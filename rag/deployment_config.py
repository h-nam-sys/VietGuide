"""Small deployment checks; no model construction or inference."""
import importlib.util
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


def model_assets(root, check_runtime=False):
    root = Path(root)
    model = Path(os.getenv('MODEL_PATH', 'models/qwen3.5_4b.gguf'))
    model = model if model.is_absolute() else root / model
    gpu = Path(os.getenv('RAG_GPU_MODEL_DIR', 'models/rag_directml'))
    gpu = gpu if gpu.is_absolute() else root / gpu
    result = {'qwen_file': model.is_file(), 'retrieval_assets': False}
    try:
        pointer = json.loads((gpu/'current.json').read_text(encoding='utf-8'))
        name = pointer['generation']
        if not isinstance(name, str) or not name.isalnum():
            return result
        directory = gpu/name
        metadata = json.loads((directory/'validated.json').read_text(encoding='utf-8'))
        result['retrieval_assets'] = (metadata.get('schema') == 1 and metadata.get('provider') == 'DmlExecutionProvider'
            and all((directory/kind/'model.onnx').is_file() for kind in ('embedding','reranker')))
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if check_runtime and result['retrieval_assets']:
        try:
            import onnxruntime as ort
            result['retrieval_assets'] = ('DmlExecutionProvider' in ort.get_available_providers()
                and metadata.get('onnxruntime_version') == ort.__version__)
        except Exception:
            result['retrieval_assets'] = False
    return result


def database_path(root):
    value = os.getenv('DATABASE_URL', 'sqlite+aiosqlite:///./legal_llm.db')
    prefix = 'sqlite+aiosqlite:///'
    if not value.startswith(prefix) or '?' in value:
        raise ValueError('Operational backup currently requires a SQLite DATABASE_URL without URI options')
    path = Path(value[len(prefix):])
    return path if path.is_absolute() else Path(root)/'backend'/path


def validate(root, production=False):
    from .service_config import llm_port, llm_url
    from .request_capacity import RequestCapacityMiddleware
    from .operations import RequestBodyLimitMiddleware
    errors, warnings = [], []
    root = Path(root)
    if not (root/'backend/static/index.html').is_file():
        errors.append('Chat frontend is missing')
    try:
        port = llm_port()
        if port in {8000,8501}:
            errors.append('LLM_PORT conflicts with backend/admin')
        uri = urlsplit(llm_url())
        if uri.scheme not in {'http','https'} or not uri.hostname:
            errors.append('Invalid LLM service URL')
        RequestCapacityMiddleware(None,kind='chat')
        RequestCapacityMiddleware(None,kind='inference')
        RequestBodyLimitMiddleware(None)
        if not 0 <= int(os.getenv('SERVICE_RESTART_LIMIT','3')) <= 20:
            errors.append('SERVICE_RESTART_LIMIT must be between 0 and 20')
        if not 10 <= float(os.getenv('SERVICE_STARTUP_GRACE','300')) <= 1800:
            errors.append('SERVICE_STARTUP_GRACE must be between 10 and 1800 seconds')
        if not 1 <= float(os.getenv('GPU_SERVICE_TIMEOUT','300')) <= 1800:
            errors.append('GPU_SERVICE_TIMEOUT must be between 1 and 1800 seconds')
        if not 1 <= int(os.getenv('INGEST_EMBED_BATCH','4')) <= 8:
            errors.append('INGEST_EMBED_BATCH must be between 1 and 8')
        database_path(root)
    except (ValueError, TypeError) as exc:
        errors.append(str(exc))
    if os.getenv('CADDY_BINARY'):
        import shutil
        if not os.getenv('PUBLIC_HOST'):
            errors.append('Set PUBLIC_HOST when using CADDY_BINARY')
        if not shutil.which(os.environ['CADDY_BINARY']):
            errors.append('CADDY_BINARY was not found')
    assets = model_assets(root)
    errors.extend(name + ' is missing/unvalidated' for name, valid in assets.items() if not valid)
    for module in ('fastapi','uvicorn','streamlit','sqlalchemy','aiosqlite','httpx','llama_cpp','transformers','onnxruntime','psutil'):
        if importlib.util.find_spec(module) is None:
            errors.append('Missing Python dependency: ' + module)
    if os.getenv('PUBLIC_HOST'):
        import re
        if not re.fullmatch(r'[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?',os.environ['PUBLIC_HOST']):
            errors.append('PUBLIC_HOST must be a hostname, without https:// or a path')
    if production and not os.getenv('ADMIN_PASSWORD'):
        errors.append('Set ADMIN_PASSWORD before production startup')
    if os.getenv('PUBLIC_HOST') and os.getenv('SESSION_COOKIE_SECURE','auto').lower() not in {'1','true'}:
        errors.append('Set SESSION_COOKIE_SECURE=1 for the public HTTPS site')
    if production and os.getenv('LLM_BIND_HOST','127.0.0.1') not in {'127.0.0.1','localhost','::1'}:
        warnings.append('Model service is not bound to loopback; restrict it with a firewall')
    return {'ok': not errors, 'errors': errors, 'warnings': warnings, 'assets': assets}
