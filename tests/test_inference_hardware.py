"""Hardware selection tests; actual Windows/GPU inference is a separate check."""
import importlib.util
import os
from pathlib import Path
import types
import unittest
import sys
import numpy  # Load before temporary module patches to avoid repeated imports.
from contextlib import nullcontext
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('hardware_test', ROOT / 'rag/inference_hardware.py')
hardware = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hardware)


class HardwareTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'INFERENCE_DEVICE': 'auto'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_vendor_order(self):
        names = ['Intel GPU', 'NVIDIA RTX', 'AMD Radeon']
        self.assertEqual(sorted(names, key=hardware.vendor_rank), ['AMD Radeon', 'NVIDIA RTX', 'Intel GPU'])

    def load(self, fail=()):
        attempts = []
        def factory(**kwargs):
            attempts.append(kwargs)
            if kwargs.get('main_gpu') in fail:
                raise RuntimeError('Device unavailable')
            return object()
        # IDs intentionally differ from preferred vendor order.
        devices = [{'id': 0, 'name': 'NVIDIA RTX'}, {'id': 1, 'name': 'AMD Radeon'}]
        with patch.object(hardware, 'llama_devices', return_value=devices):
            _, selected = hardware.load_llama(factory, types.SimpleNamespace(), 'model.gguf')
        return attempts, selected

    def test_amd_is_preferred_without_loading_second_model(self):
        attempts, selected = self.load()
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]['main_gpu'], 1)
        self.assertEqual(attempts[0]['split_mode'], 0)
        self.assertEqual(selected['device'], 'AMD Radeon')

    def test_failed_amd_tries_nvidia(self):
        attempts, selected = self.load(fail=(1,))
        self.assertEqual([x['main_gpu'] for x in attempts], [1, 0])
        self.assertEqual(selected['device'], 'NVIDIA RTX')

    def test_failed_gpus_use_cpu(self):
        attempts, selected = self.load(fail=(1, 0))
        self.assertEqual(len(attempts), 3)
        self.assertEqual(attempts[-1]['n_gpu_layers'], 0)
        self.assertFalse(attempts[-1]['offload_kqv'])
        self.assertEqual(selected['backend'], 'cpu')

    def test_explicit_cpu_never_probes_gpu(self):
        with patch.dict(os.environ, {'INFERENCE_DEVICE': 'cpu'}):
            with patch.object(hardware, 'llama_devices', side_effect=AssertionError('Unexpected GPU probe')):
                _, selected = hardware.load_llama(lambda **kw: object(), None, 'model.gguf')
        self.assertEqual(selected['backend'], 'cpu')

    def test_cpu_error_is_not_hidden(self):
        with patch.dict(os.environ, {'INFERENCE_DEVICE': 'cpu'}):
            def fail(**kwargs):
                raise RuntimeError('Missing or corrupt model')
            with self.assertRaisesRegex(RuntimeError, 'Missing or corrupt model'):
                hardware.load_llama(fail, None, 'missing.gguf')

    def test_bad_mode_fails_clearly(self):
        with patch.dict(os.environ, {'INFERENCE_DEVICE': 'typo'}):
            with self.assertRaises(ValueError):
                hardware.hardware_mode()

    def test_onnx_amd_nvidia_cpu_order_uses_dxgi_ids(self):
        ort = types.SimpleNamespace(get_available_providers=lambda: ['DmlExecutionProvider', 'CPUExecutionProvider'])
        devices = [{'id': 2, 'name': 'AMD'}, {'id': 0, 'name': 'NVIDIA'}]
        with patch.object(hardware, 'directml_devices', return_value=devices):
            candidates = hardware.onnx_candidates(ort)
        self.assertEqual(candidates, [('DmlExecutionProvider', {'device_id': '2'}),
                                     ('DmlExecutionProvider', {'device_id': '0'}),
                                     ('CPUExecutionProvider', {})])

    def test_onnx_cpu_only_environment(self):
        ort = types.SimpleNamespace(get_available_providers=lambda: ['CPUExecutionProvider'])
        self.assertEqual(hardware.onnx_candidates(ort), [('CPUExecutionProvider', {})])

    def test_onnx_cuda_available(self):
        ort = types.SimpleNamespace(get_available_providers=lambda: ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.assertEqual(hardware.onnx_candidates(ort)[0][0], 'CUDAExecutionProvider')

    def test_export_validation_does_not_fall_back(self):
        ort = types.SimpleNamespace(get_available_providers=lambda: ['CPUExecutionProvider'])
        with self.assertRaises(RuntimeError):
            hardware.onnx_candidates(ort, strict=True)

    def test_export_validation_preserves_requested_adapter(self):
        ort = types.SimpleNamespace(get_available_providers=lambda: ['DmlExecutionProvider', 'CPUExecutionProvider'])
        self.assertEqual(hardware.onnx_candidates(ort, device_id=3, strict=True),
                         [('DmlExecutionProvider', {'device_id': '3'})])

    def runtime(self, failures=(), silent=False):
        calls = []
        class Session:
            def __init__(self, path, sess_options, providers):
                self.provider = providers[0][0]
                calls.append(self.provider)
                if self.provider in failures:
                    raise RuntimeError('Unavailable device')
            def disable_fallback(self):
                pass
            def get_providers(self):
                return ['CPUExecutionProvider'] if silent else [self.provider]
        ort = types.SimpleNamespace(SessionOptions=lambda: types.SimpleNamespace(),
            ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL=0),
            GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL=0),
            InferenceSession=Session)
        candidates = [('DmlExecutionProvider', {'device_id': '0'}), ('CPUExecutionProvider', {})]
        config = types.SimpleNamespace(ROOT_DIR=ROOT, RAGUnavailableError=RuntimeError)
        timing = types.SimpleNamespace(emit=lambda *a, **kw: None, span=lambda *a, **kw: nullcontext())
        diagnostics = types.SimpleNamespace(memory_stage=lambda *a, **kw: nullcontext())
        modules = {'onnxruntime': ort, 'rag.domain_config': config, 'rag.timing': timing,
                   'rag.hardware_diagnostics': diagnostics, 'rag.inference_hardware': hardware}
        spec = importlib.util.spec_from_file_location('rag.runtime_test', ROOT / 'rag/gpu_runtime.py')
        runtime = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules), patch.object(hardware, 'onnx_candidates', return_value=candidates):
            spec.loader.exec_module(runtime)
            session = runtime.make_session(ROOT / 'embedding/model.onnx')
        return calls, session

    def test_retrieval_session_load_failure_falls_back_to_cpu(self):
        calls, session = self.runtime(failures=('DmlExecutionProvider',))
        self.assertEqual(calls, ['DmlExecutionProvider', 'CPUExecutionProvider'])
        self.assertEqual(session.get_providers(), ['CPUExecutionProvider'])

    def test_retrieval_success_does_not_load_cpu_copy(self):
        calls, session = self.runtime()
        self.assertEqual(calls, ['DmlExecutionProvider'])
        self.assertEqual(session.get_providers(), ['DmlExecutionProvider'])

    def test_retrieval_detects_silent_provider_fallback(self):
        calls, session = self.runtime(silent=True)
        self.assertEqual(calls, ['DmlExecutionProvider', 'CPUExecutionProvider'])

    def test_retrieval_all_load_failures_still_fail(self):
        with self.assertRaisesRegex(RuntimeError, 'No inference provider'):
            self.runtime(failures=('DmlExecutionProvider', 'CPUExecutionProvider'))


if __name__ == '__main__':
    unittest.main()
