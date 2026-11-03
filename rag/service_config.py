"""Shared inference endpoint defaults for launcher, service and clients."""
import os

DEFAULT_LLM_PORT = 8001


def llm_port():
    try:
        port=int(os.getenv('LLM_PORT',str(DEFAULT_LLM_PORT)))
    except ValueError as exc:
        raise ValueError('LLM_PORT must be an integer between 1 and 65535') from exc
    if not 1 <= port <= 65535:
        raise ValueError('LLM_PORT must be an integer between 1 and 65535')
    return port


def llm_url():
    # Preserve existing URL overrides and their precedence.
    value=os.getenv('LLM_SERVICE_URLS',os.getenv('LLM_SERVICE_URL'))
    if value is None:
        value=f'http://127.0.0.1:{llm_port()}'
    return value.split(',')[0].strip().rstrip('/')


def model_http_timeout(default=30):
    """Optional slower-runtime allowance; native call-site defaults stay intact."""
    try:
        value = float(os.getenv('MODEL_HTTP_TIMEOUT', str(default)))
    except ValueError as exc:
        raise ValueError('MODEL_HTTP_TIMEOUT must be between 1 and 1800 seconds') from exc
    if not 1 <= value <= 1800:
        raise ValueError('MODEL_HTTP_TIMEOUT must be between 1 and 1800 seconds')
    return value
