"""Preserve the existing GPU initialization; try CPU only after it fails."""
import gc
import logging

logger = logging.getLogger(__name__)
_startup = {}


def startup_status():
    # These are initialization paths, not claims about measured GPU utilization.
    return {name: dict(value) for name, value in _startup.copy().items()}


def initialize(gpu, cpu, *, component, gpu_path, strict=False):
    try:
        result = gpu()
    except Exception as exc:
        if strict:
            raise
        error_type = type(exc).__name__
        logger.warning('%s: original %s initialization failed (%s). Trying CPU; inference may be slower.',
                       component, gpu_path, error_type)
    else:
        _startup[component] = {'path': 'original_gpu_settings', 'configured_backend': gpu_path}
        return result
    # Leave the exception handler before CPU allocation: do not retain a failed
    # constructor's traceback/model objects while allocating another model.
    gc.collect()
    result = cpu()  # A CPU failure propagates; there is no request replay.
    _startup[component] = {'path': 'cpu_fallback', 'configured_backend': 'CPU',
                           'gpu_error_type': error_type}
    logger.warning('%s: CPU fallback activated.', component)
    return result


def load_llama(factory, model_path):
    # Exactly the working constructor arguments on the first attempt. No device
    # enumeration, vendor filtering, main_gpu override, or split_mode override.
    common = dict(model_path=str(model_path), n_ctx=4096, n_threads=8, verbose=True)
    return initialize(
        lambda: factory(**common, n_gpu_layers=-1),
        lambda: factory(**common, n_gpu_layers=0, offload_kqv=False),
        component='qwen', gpu_path='llama.cpp default GPU/Vulkan')
