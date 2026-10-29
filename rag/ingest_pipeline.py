"""Offline ingestion. Run from repo root: python -m rag.ingest_pipeline --help."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import unicodedata
import uuid

from .domain_config import ROOT_DIR, RAGConfigurationError, load_domain_config, validate_profile


from rag.timing import timed, span, emit, TimedLock, trace_headers, TimingMiddleware, trace_id

def normalize_header(value):
    return " ".join(unicodedata.normalize("NFC", str(value)).split()).casefold()


def cell_text(value):
    import pandas as pd
    if value is None or pd.isna(value):
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value).strip()


@timed('rag.ingest_pipeline.read_input')
def read_input(path, sheet=0):
    import pandas as pd
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm"):
        return pd.read_excel(path, sheet_name=sheet, dtype=object, keep_default_na=False)
    if suffix in (".csv", ".tsv"):
        # Strings preserve textual IDs such as 00123. Literal 'NA' remains data.
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", dtype=str,
                           keep_default_na=False, encoding="utf-8-sig")
    if suffix in ('.json', '.jsonl'):
        with path.open(encoding='utf-8-sig') as file:
            data = [json.loads(line) for line in file if line.strip()] if suffix == '.jsonl' else json.load(file)
        if not isinstance(data, list) or not data or any(not isinstance(row, dict) for row in data):
            raise ValueError('JSON must be a nonempty array of row objects; JSONL must have one object per line')
        if any(isinstance(value, (dict, list)) for row in data for value in row.values()):
            raise ValueError('Nested JSON must be flattened into tabular fields first')
        return pd.DataFrame(data)
    raise ValueError("Supported inputs: .xlsx, .xlsm, .csv, .tsv, .json, .jsonl")


@timed('rag.ingest_pipeline.prepare_records')
def prepare_records(df, profile):
    validate_profile(profile)
    if "_raw_json" in profile["schema_mapping"]:
        raise RAGConfigurationError("_raw_json is reserved for original row data")
    # Ignore only unnamed AND entirely empty columns.
    df = df.loc[:, [not (str(column).startswith("Unnamed:") and all(not cell_text(v) for v in df[column]))
                    for column in df.columns]]
    header_lookup = {}
    for column in df.columns:
        normalized = normalize_header(column)
        if normalized in header_lookup:
            raise RAGConfigurationError(f"Ambiguous duplicate header: {column!r}")
        header_lookup[normalized] = column
    resolved = {}
    for key, label in profile["schema_mapping"].items():
        column = header_lookup.get(normalize_header(label))
        if column is None:
            raise RAGConfigurationError(f"Mapped input column is missing: {label!r} ({key})")
        resolved[key] = column
    records, missing_entity_rows = [], []
    empty_rows = 0
    duplicate_counts = {}
    seen_ids = set()
    for position, (_, row) in enumerate(df.iterrows(), start=2):
        raw = {str(column): cell_text(row[column]) for column in df.columns}
        if not any(raw.values()):
            empty_rows += 1
            continue
        fields = {key: cell_text(row[column]) for key, column in resolved.items()}
        if not fields[profile["primary_entity"]]:
            missing_entity_rows.append(position)
            continue
        id_field = profile.get("source_id_field")
        if id_field:
            record_id = fields[id_field]
            if not record_id or record_id in seen_ids:
                raise RAGConfigurationError(f"Missing or duplicate source ID at input row {position}")
        else:
            # Content-addressed fallback: stable across row reordering; changes on content edits.
            digest = hashlib.sha256(json.dumps(raw, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            duplicate_counts[digest] = duplicate_counts.get(digest, 0) + 1
            record_id = f"{digest}:{duplicate_counts[digest]}"
        seen_ids.add(record_id)
        records.append({"id": record_id, **fields, "_raw_json": json.dumps(raw, ensure_ascii=False)})
    if not records:
        raise RAGConfigurationError("No rows with a primary entity; active data was not changed")
    report = {"input_rows": len(df), "indexed_rows": len(records), "empty_rows": empty_rows,
              "missing_primary_entity_rows": missing_entity_rows,
              "unmapped_columns_preserved_in_raw_json": [str(c) for c in df.columns if c not in resolved.values()]}
    return records, report


@timed('rag.ingest_pipeline.write_database')
def write_database(records, schema, path):
    columns = ["id"] + list(schema) + ["_raw_json"]
    quoted = ", ".join(f'"{column}"' for column in columns)
    definitions = ', '.join(f'"{column}" TEXT' for column in columns[1:])
    with sqlite3.connect(path) as connection:
        connection.execute(f'CREATE TABLE records (id TEXT PRIMARY KEY, {definitions})')
        connection.executemany(f'INSERT INTO records ({quoted}) VALUES ({", ".join("?" for _ in columns)})',
                               [[row[column] for column in columns] for row in records])
        count = connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        if count != len(records):
            raise RuntimeError("SQL row count mismatch")


@timed('rag.ingest_pipeline.build_vectors')
def build_vectors(records, profile, destination):
    from llama_index.core import Document, VectorStoreIndex
    from .gpu_runtime import embedding_model
    primary = profile["primary_entity"]
    fields = profile.get("embedding_fields") or [primary]
    # Retain the legacy legal title representation unless configured otherwise.
    prefix = profile.get("embedding_prefix", "Thủ tục: " if primary == "procedure_name" else "")
    documents = []
    for row in records:
        text = prefix + row[primary]
        for key in fields:
            if key != primary and row[key]:
                text += f"\n{profile['schema_mapping'][key]}: {row[key]}"
        metadata = {"id": row["id"], primary: row[primary]}
        documents.append(Document(text=text, metadata=metadata,
                                  excluded_llm_metadata_keys=list(metadata), doc_id=row["id"]))
    with span('ingest.embedding_model_load'):
        embedding = embedding_model(profile.get("embedding_model", "BAAI/bge-m3"))
    with span('ingest.embedding_and_index_build'):
        index = VectorStoreIndex.from_documents(documents, embed_model=embedding, show_progress=True)
    index.storage_context.persist(persist_dir=str(destination))
    # Verify the persisted index is readable and retains every source ID.
    from llama_index.core import StorageContext, load_index_from_storage
    restored = load_index_from_storage(StorageContext.from_defaults(persist_dir=str(destination)), embed_model=embedding)
    stored_ids = {str(node.metadata.get("id")) for node in restored.docstore.docs.values()}
    if not {row["id"] for row in records}.issubset(stored_ids):
        raise RuntimeError("Persisted vectors are missing source IDs")


@timed('rag.ingest_pipeline.publish_generation')
def publish_generation(config, records, report, *, vector_builder=build_vectors, progress=None):
    generation = uuid.uuid4().hex
    directory = config.pointer_path.parent / generation
    directory.mkdir(parents=True, exist_ok=False)
    # A failure leaves an unreferenced directory for diagnosis, never a partial active version.
    if progress:
        progress('Writing SQL snapshot')
    write_database(records, config.schema, directory / "records.db")
    if progress:
        progress('Building and verifying vector index')
    vector_builder(records, config.profile, directory / "vectors")
    for filename, value in (("profile.json", config.profile), ("ingestion_report.json", report)):
        (directory / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    # Check current runtime settings against any new schema before publishing.
    from .domain_config import validate_runtime, read_json
    runtime_path = config.pointer_path.parents[2] / 'domain_settings' / f'{config.name}.json'
    if runtime_path.exists():
        validate_runtime(read_json(runtime_path), config.profile)
    if progress:
        progress('Activating completed snapshot')
    temporary_pointer = config.pointer_path.parent / f"{generation}.tmp"
    with temporary_pointer.open("w", encoding="utf-8") as file:
        json.dump({"generation": generation}, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_pointer, config.pointer_path)
    return generation


@timed('rag.ingest_pipeline.main')
def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / "backend/.env")
    load_dotenv(ROOT_DIR / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--input", '--file', type=Path, default=ROOT_DIR / "data/raw/Tổng hợp các thủ tục hành chính (22.12) (1).xlsx")
    parser.add_argument("--sheet", default="0", help="Worksheet index or name")
    parser.add_argument("--dry-run", action="store_true", help="Validate rows without models or writes")
    parser.add_argument('--auto-profile', action='store_true', help='Use Gemini to propose a NEW profile; do not overwrite existing domain schemas')
    parser.add_argument('--profile-file', type=Path, help='Reviewed JSON profile to use for this ingestion')
    parser.add_argument('--save-draft', type=Path, help='Save the proposed profile without ingestion')
    args = parser.parse_args()
    sheet = int(args.sheet) if args.sheet.isdecimal() else args.sheet
    df = read_input(args.input, sheet)
    from .domain_onboarding import (propose_profile, config_from_profile,
                                   validate_dataset_profile, save_profile_registry)
    from .domain_config import list_domain_configs, read_json, atomic_json
    from .ingestion_jobs import domain_lock
    known = {c.name for c in list_domain_configs(include_inactive=True)}
    if args.auto_profile and args.profile_file:
        parser.error('Choose --auto-profile or --profile-file')
    if args.auto_profile and (args.domain in known or (not args.domain and known)):
        parser.error('Existing domains reuse their profile. Use a reviewed --profile-file for a deliberate schema change.')
    if args.profile_file:
        profile = validate_dataset_profile(read_json(args.profile_file), df, args.domain)
        config = config_from_profile(profile)
    elif args.auto_profile or (args.domain and args.domain not in known) or not known:
        profile = propose_profile(df, args.domain)
        if profile['domain_name'] in known:
            parser.error('Generated domain ID already exists. Supply a new --domain ID.')
        config = config_from_profile(profile)
    else:
        config = load_domain_config(args.domain, use_active=False, allow_inactive=True)
    records, report = prepare_records(df, config.profile)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.save_draft:
        atomic_json(args.save_draft, config.profile)
        print(f'Saved profile draft: {args.save_draft}')
        return
    if not args.dry_run:
        from .process_lock import process_lock
        os.environ['LEGAL_INGESTION'] = '1'
        with domain_lock(config.name), process_lock(ROOT_DIR/'data/ops_locks/ingestion.lock',timeout=86400):
            generation = publish_generation(config, records, report, progress=lambda stage: print(stage, flush=True))
            save_profile_registry(config.profile)
        print(f"Activated {config.name} generation {generation}")


if __name__ == "__main__":
    main()
