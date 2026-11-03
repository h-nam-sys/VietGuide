"""Seed private Docker volumes once, then exec the existing service supervisor."""
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile

ROOT = Path('/app')


def copy_asset(source, target):
    source, target = Path(source), Path(target)
    if source.is_symlink():
        raise ValueError('Runtime seeding does not follow symlinks: ' + str(source))
    if source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            if child.name in {'ops_locks', '__pycache__', 'locks'} or child.name.endswith(('-wal','-shm')):
                continue
            copy_asset(child, target/child.name)
    elif source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix.lower() in {'.db','.sqlite','.sqlite3'}:
            with sqlite3.connect(source.resolve().as_uri()+'?mode=ro', uri=True) as original:
                with sqlite3.connect(target) as output:
                    original.backup(output)
                    if output.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                        raise ValueError('Invalid source database: ' + str(source))
        else:
            shutil.copy2(source, target)


def seed_runtime(root=ROOT, seed=Path('/seed')):
    root, seed = Path(root), Path(seed)
    data = root/'data'
    data.mkdir(parents=True, exist_ok=True)
    marker = data/'.docker_initialized.json'
    if marker.is_file():
        return False
    existing = [p for p in data.iterdir() if p.name != 'ops_locks']
    if existing:
        raise RuntimeError('Docker data volume is nonempty but initialization is incomplete; refusing to overwrite it.')
    source_data = seed/'data'
    if not (source_data/'domain_profile.json').is_file():
        raise RuntimeError('Existing project data/domain_profile.json is missing from the read-only seed mount.')
    forms = root/'backend/static/forms'
    # A stage keeps the native source untouched and validates SQLite copies
    # before the seeded data becomes visible to application services.
    with tempfile.TemporaryDirectory(prefix='.docker-seed-', dir=data) as temporary:
        stage = Path(temporary)
        copy_asset(source_data, stage/'data')
        runtime = stage/'data/docker_runtime'
        if runtime.exists():
            raise RuntimeError('Native data already contains docker_runtime; refusing an ambiguous import.')
        runtime.mkdir()
        source_backend = seed/'backend'
        source_db = source_backend/'legal_llm.db'
        if source_db.is_file():
            copy_asset(source_db, runtime/'legal_llm.db')
        for name in ('all_feedback_dataset.jsonl','sft_positive_dataset.jsonl','review_negative_dataset.jsonl'):
            if (source_backend/name).is_file():
                copy_asset(source_backend/name, runtime/name)
        if (source_backend/'static/forms').is_dir():
            copy_asset(source_backend/'static/forms', stage/'forms')
        if forms.exists() and any(forms.iterdir()):
            raise RuntimeError('Docker forms volume is already populated; refusing an ambiguous first import.')
        forms.mkdir(parents=True, exist_ok=True)
        if (stage/'forms').exists():
            # Forms and data are different Docker volumes: do not rename across
            # their filesystem boundary (EXDEV).
            copy_asset(stage/'forms', forms)
        for path in (stage/'data').iterdir():
            path.replace(data/path.name)
        marker.write_text(json.dumps({'schema':1,'source':'native snapshot; copied once'}), encoding='utf-8')
    return True


def main():
    arguments = sys.argv[1:]
    if arguments not in ([], ['--web-only']):
        raise ValueError('Container accepts only the optional --web-only argument')
    initialized = seed_runtime()
    print('Docker runtime snapshot initialized.' if initialized else 'Using existing Docker runtime data.', flush=True)
    os.chdir(ROOT)
    os.execv(sys.executable, [sys.executable, '-u', str(ROOT/'run_services.py'), '--production', *arguments])


if __name__ == '__main__':
    main()
