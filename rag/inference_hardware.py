"""Startup-only hardware selection; no request replay or model downloads."""
import ctypes
import gc
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def hardware_mode():
    mode = os.getenv('INFERENCE_DEVICE', 'auto').strip().lower()
    if mode not in {'auto', 'cpu'}:
        raise ValueError('INFERENCE_DEVICE must be auto or cpu')
    return mode


def vendor_rank(name):
    name = name.lower()
    if any(word in name for word in ('amd', 'radeon', 'advanced micro devices')):
        return 0
    if any(word in name for word in ('nvidia', 'geforce', 'quadro', 'tesla')):
        return 1
    return 2


def llama_devices(api):
    """Use ggml's GPU order, which is the order used by main_gpu."""
    import llama_cpp
    api.llama_backend_init()
    libraries = [getattr(api, '_lib', api)]
    directory = Path(llama_cpp.__file__).parent / 'lib'
    for pattern in ('*ggml-base*.dll', '*ggml-base*.so*', '*ggml-base*.dylib'):
        for path in directory.glob(pattern):
            try:
                libraries.append(ctypes.CDLL(str(path)))
            except OSError:
                continue
    for lib in libraries:
        try:
            count = lib.ggml_backend_dev_count
            get = lib.ggml_backend_dev_get
            kind = lib.ggml_backend_dev_type
            description = lib.ggml_backend_dev_description
            count.argtypes, count.restype = [], ctypes.c_size_t
            get.argtypes, get.restype = [ctypes.c_size_t], ctypes.c_void_p
            kind.argtypes, kind.restype = [ctypes.c_void_p], ctypes.c_int
            description.argtypes, description.restype = [ctypes.c_void_p], ctypes.c_char_p
            devices = []
            for index in range(count()):
                device = get(index)
                if kind(device) in (1, 2):  # GGML_BACKEND_DEVICE_TYPE_GPU / IGPU
                    devices.append({'id': len(devices), 'name': description(device).decode(errors='replace')})
            return devices
        except AttributeError:
            continue
    logger.warning('Cannot enumerate llama GPU devices with this build; using CPU safely')
    return []


def load_llama(factory, api, model_path):
    mode = hardware_mode()
    common = dict(model_path=str(model_path), n_ctx=4096, n_threads=8, verbose=True)
    candidates = []
    if mode == 'auto':
        try:
            candidates = sorted(llama_devices(api), key=lambda d: vendor_rank(d['name']))
        except Exception as exc:
            logger.warning('GPU enumeration failed (%s); using CPU', type(exc).__name__)
    for device in candidates:
        if vendor_rank(device['name']) > 1:
            continue
        try:
            model = factory(**common, n_gpu_layers=-1, main_gpu=device['id'],
                            split_mode=getattr(api, 'LLAMA_SPLIT_MODE_NONE', 0))
            selected = {'device': device['name'], 'device_id': device['id'], 'backend': 'gpu'}
            logger.info('Qwen inference selected: %s', selected)
            return model, selected
        except Exception as exc:
            logger.warning('Qwen initialization failed on %s (%s); trying next device',
                           device['name'], type(exc).__name__)
            gc.collect()
    # Explicitly disable GPU offload and GPU K/Q/V offload for the CPU attempt.
    model = factory(**common, n_gpu_layers=0, offload_kqv=False)
    selected = {'device': 'CPU', 'backend': 'cpu'}
    logger.warning('Qwen inference selected: CPU; responses may be slower')
    return model, selected


def directml_devices():
    """DXGI EnumAdapters order, as required by DirectML device_id."""
    if os.name != 'nt':
        return []
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [('a', wintypes.DWORD), ('b', wintypes.WORD),
                    ('c', wintypes.WORD), ('d', ctypes.c_ubyte * 8)]

    class DESC(ctypes.Structure):
        _fields_ = [('description', wintypes.WCHAR * 128), ('vendor', wintypes.UINT),
                    ('device', wintypes.UINT), ('subsys', wintypes.UINT),
                    ('revision', wintypes.UINT), ('video', ctypes.c_size_t),
                    ('system', ctypes.c_size_t), ('shared', ctypes.c_size_t),
                    ('luid_low', wintypes.DWORD), ('luid_high', wintypes.LONG)]

    def method(obj, index, restype, *args):
        table = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *args)(table[index])

    iid = GUID(0x7b7166ec, 0x21c7, 0x44ae, (ctypes.c_ubyte * 8)(0xb2, 0x1a, 0xc9, 0xae, 0x32, 0x1a, 0xe3, 0x69))
    factory = ctypes.c_void_p()
    dxgi = ctypes.WinDLL('dxgi')
    create = dxgi.CreateDXGIFactory
    create.argtypes = [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
    create.restype = ctypes.c_long
    if create(ctypes.byref(iid), ctypes.byref(factory)) < 0:
        raise RuntimeError('DXGI factory creation failed')
    devices = []
    try:
        for index in range(32):
            adapter = ctypes.c_void_p()
            result = method(factory, 7, ctypes.c_long, wintypes.UINT,
                            ctypes.POINTER(ctypes.c_void_p))(factory, index, ctypes.byref(adapter))
            if result < 0:
                if result & 0xffffffff == 0x887a0002:  # DXGI_ERROR_NOT_FOUND
                    break
                raise RuntimeError('DXGI adapter enumeration failed')
            try:
                desc = DESC()
                if method(adapter, 8, ctypes.c_long, ctypes.POINTER(DESC))(adapter, ctypes.byref(desc)) < 0:
                    continue
                vendor = {0x1002: 'AMD', 0x10de: 'NVIDIA'}.get(desc.vendor)
                if vendor:
                    devices.append({'id': index, 'name': vendor + ' ' + desc.description})
            finally:
                method(adapter, 2, wintypes.ULONG)(adapter)
    finally:
        method(factory, 2, wintypes.ULONG)(factory)
    return sorted(devices, key=lambda d: vendor_rank(d['name']))


def onnx_candidates(ort, device_id=0, strict=False):
    available = ort.get_available_providers()
    if strict:
        if 'DmlExecutionProvider' not in available:
            raise RuntimeError('DirectML is required for GPU export validation')
        return [('DmlExecutionProvider', {'device_id': str(device_id)})]
    if hardware_mode() == 'auto':
        if 'DmlExecutionProvider' in available:
            try:
                devices = directml_devices()
            except Exception as exc:
                logger.warning('DXGI detection failed (%s); trying saved DirectML device', type(exc).__name__)
                devices = [{'id': device_id, 'name': 'saved device'}]
            candidates = [('DmlExecutionProvider', {'device_id': str(d['id'])}) for d in devices]
        else:
            candidates = []
        if 'CUDAExecutionProvider' in available:
            candidates.append(('CUDAExecutionProvider', {'device_id': '0'}))
    else:
        candidates = []
    if 'CPUExecutionProvider' in available:
        candidates.append(('CPUExecutionProvider', {}))
    return candidates
