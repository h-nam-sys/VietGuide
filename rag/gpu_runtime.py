"""Existing DirectML inference with CPU fallback on session initialization failure."""
import json
import os
from pathlib import Path
from threading import RLock

import numpy as np

from .domain_config import ROOT_DIR, RAGUnavailableError
from .timing import emit, span
from .hardware_diagnostics import memory_stage

EMBED_MODEL = 'BAAI/bge-m3'
RERANK_MODEL = 'BAAI/bge-reranker-v2-m3'
_GPU_LOCK = RLock()  # DirectML forbids concurrent Run calls on one session.


def runtime_root():
    configured = Path(os.getenv('RAG_GPU_MODEL_DIR', str(ROOT_DIR / 'models' / 'rag_directml')))
    return configured if configured.is_absolute() else ROOT_DIR / configured


def read_active():
    try:
        root = runtime_root()
        pointer = json.loads((root / 'current.json').read_text(encoding='utf-8'))
        generation = pointer['generation']
        if not isinstance(generation, str) or not generation.isalnum():
            raise ValueError('Invalid GPU generation')
        directory = root / generation
        meta = json.loads((directory / 'validated.json').read_text(encoding='utf-8'))
        if meta.get('schema') != 1 or meta.get('models') != [EMBED_MODEL, RERANK_MODEL]:
            raise ValueError('Unrecognized GPU model manifest')
        if meta.get('provider') != 'DmlExecutionProvider' or meta.get('precision') not in ('fp32', 'fp16'):
            raise ValueError('Unvalidated GPU configuration')
        import onnxruntime as ort
        if meta.get('onnxruntime_version') != ort.__version__:
            raise ValueError('ONNX Runtime changed; repeat local GPU validation')
        return directory, meta
    except Exception as exc:
        raise RAGUnavailableError('Retrieval assets are not prepared/validated. Run python -m rag.prepare_gpu --export, then --validate.') from exc


def make_session(path, *, device_id=0, profile=False, precision='fp32'):
    import onnxruntime as ort
    if precision not in ('fp32', 'fp16'):
        raise ValueError('Unsupported GPU precision')
    if device_id < 0:
        raise ValueError('DirectML device_id must be nonnegative')
    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = profile
    if profile:
        prefix = ROOT_DIR / 'logs' / 'gpu_profiles'
        prefix.mkdir(parents=True, exist_ok=True)
        options.profile_file_prefix = str(prefix / path.parent.name)
    from .inference_fallback import initialize

    def create(provider):
        session = None
        try:
            if provider not in ort.get_available_providers():
                raise RAGUnavailableError(provider + ' is unavailable')
            providers = ([('DmlExecutionProvider', {'device_id': str(device_id)})]
                         if provider == 'DmlExecutionProvider' else ['CPUExecutionProvider'])
            with memory_stage('rag.directml.session_load', model=path.parent.name,
                              precision=precision), span('rag.directml.session_load'):
                session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
            session.disable_fallback()
            if not session.get_providers() or session.get_providers()[0] != provider:
                raise RAGUnavailableError('ONNX Runtime did not activate ' + provider)
            emit('rag.directml', 'session_ready', model=path.parent.name, device_id=device_id,
                 providers=session.get_providers(), precision=precision,
                 cpu_model_fallback=provider == 'CPUExecutionProvider')
            return session
        except Exception:
            # Drop any silently substituted session before a CPU retry.
            del session
            raise

    # Profiling is used for export validation: it must still prove DirectML works,
    # never publish a supposedly GPU-validated export using a CPU fallback.
    return initialize(lambda: create('DmlExecutionProvider'),
                      lambda: create('CPUExecutionProvider'),
                      component=path.parent.name, gpu_path='DirectML', strict=profile)


def numpy_feed(session, encoded):
    dtypes = {'tensor(int64)': np.int64, 'tensor(int32)': np.int32,
              'tensor(float)': np.float32}
    feed = {}
    for item in session.get_inputs():
        if item.name == 'token_type_ids' and item.name not in encoded:
            encoded = dict(encoded, token_type_ids=np.zeros_like(encoded['input_ids']))
        if item.name not in encoded:
            raise ValueError('Tokenizer is missing ONNX input: ' + item.name)
        if item.type not in dtypes:
            raise ValueError('Unsupported ONNX input type: ' + item.type)
        feed[item.name] = np.ascontiguousarray(encoded[item.name], dtype=dtypes[item.type])
    return feed


def encode_embeddings(session, tokenizer, texts, metadata, prompt='text'):
    if not texts:
        return []
    prefix = metadata['prompts'].get(prompt, '') or ''
    prepared = [(prefix + text).strip() for text in texts]
    if metadata.get('do_lower_case'):
        prepared = [text.lower() for text in prepared]
    vectors = []
    # Query inference is normally one item. Bounded batches avoid large GPU activations.
    for text in prepared:
        encoded = tokenizer([text], padding=True, truncation='longest_first',
                            max_length=metadata['max_length'], return_tensors='np')
        with _GPU_LOCK, span('rag.directml.embedding_forward'):
            result = np.asarray(session.run(['dense_embedding'], numpy_feed(session, encoded))[0], dtype=np.float32)
        if result.shape != (1, 1024) or not np.isfinite(result).all():
            raise RAGUnavailableError('DirectML returned invalid BGE-M3 embeddings')
        vectors.append(result[0].tolist())
    return vectors


def prepare_pairs(tokenizer, pairs, metadata):
    """FlagReranker defaults: query cap 384, passage cap 512, pair cap 512."""
    if not pairs:
        return []
    length = metadata['max_length']
    if metadata.get('pair_mode') == 'joint':
        encoded = tokenizer([list(pair) for pair in pairs], padding=False,
                            truncation=True, max_length=length)
        return [{key: value[i] for key, value in encoded.items()} for i in range(len(pairs))]
    query_length = metadata['query_max_length']
    queries = tokenizer([p[0] for p in pairs], add_special_tokens=False,
                        max_length=query_length, truncation=True)['input_ids']
    passages = tokenizer([p[1] for p in pairs], add_special_tokens=False,
                         max_length=length, truncation=True)['input_ids']
    rows = []
    for query, passage in zip(queries, passages):
        # Tokenizer's pair special tokens/truncation stay authoritative.
        prepare = getattr(tokenizer, 'prepare_for_model', None)
        if prepare is not None:
            row = prepare(query, passage, truncation='only_second', max_length=length, padding=False)
        else:
            from FlagEmbedding.utils.tokenizer_compat import prepare_for_model_compat
            row = prepare_for_model_compat(tokenizer, query, passage, truncation='only_second', max_length=length, padding=False)
        rows.append(row)
    return rows


def pad_rows(tokenizer, rows):
    pad = getattr(tokenizer, 'pad', None)
    if callable(pad):
        return pad(rows, padding=True, return_tensors='np')
    # FlagEmbedding supplies the Transformers 5 tokenizer compatibility path.
    from FlagEmbedding.utils.tokenizer_compat import pad_with_compat
    return pad_with_compat(tokenizer, rows, padding=True, return_tensors='np')


def score_pairs(session, tokenizer, pairs, metadata, batch_size=1):
    if not pairs:
        return []
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    rows = prepare_pairs(tokenizer, pairs, metadata)
    order = sorted(range(len(rows)), key=lambda i: -len(rows[i]['input_ids']))
    scores = [None] * len(rows)
    for offset in range(0, len(order), batch_size):
        selected = order[offset:offset + batch_size]
        encoded = pad_rows(tokenizer, [rows[i] for i in selected])
        with _GPU_LOCK, span('rag.directml.reranker_forward'):
            result = np.asarray(session.run(['logits'], numpy_feed(session, encoded))[0], dtype=np.float32).reshape(-1)
        if len(result) != len(selected) or not np.isfinite(result).all():
            raise RAGUnavailableError('DirectML returned invalid reranker logits')
        for index, value in zip(selected, result):
            scores[index] = float(value)
    return scores  # Raw logits: existing retrieval code applies sigmoid exactly once.


class LocalDirectMLReranker:
    def __init__(self):
        from transformers import AutoTokenizer
        directory, metadata = read_active()
        self.metadata = metadata['reranker']
        self.batch_size = metadata['reranker_batch_size']
        self.tokenizer = AutoTokenizer.from_pretrained(directory / 'reranker', local_files_only=True)
        self.session = make_session(directory / 'reranker' / 'model.onnx', device_id=metadata['device_id'], precision=metadata['precision'])

    def compute_score(self, sentence_pairs):
        return score_pairs(self.session, self.tokenizer, sentence_pairs, self.metadata, self.batch_size)


def embedding_model(model_name):
    if model_name != EMBED_MODEL:
        raise RAGUnavailableError(f'No validated export for {model_name}.')
    from .gpu_client import RemoteDirectMLEmbedding
    return RemoteDirectMLEmbedding()


class DirectMLReranker:
    """Shared GPU service client: ingestion/backend never duplicate GPU weights."""
    def compute_score(self, sentence_pairs):
        from .gpu_client import gpu_call
        result = gpu_call('/v1/rag/rerank', {'pairs': sentence_pairs})['logits']
        if len(result) != len(sentence_pairs):
            raise RAGUnavailableError('GPU service returned the wrong number of logits')
        return result
