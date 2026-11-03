"""Constructor-path tests, not a claim of live GPU validation."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch
import numpy  # Import before temporary module patches.

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fallback_test', ROOT / 'rag/inference_fallback.py')
fallback = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fallback)


class FallbackTests(unittest.TestCase):
    def setUp(self):
        fallback._startup.clear()

    def test_working_llama_constructor_is_unchanged(self):
        factory = Mock(return_value=object())
        result = fallback.load_llama(factory, 'model.gguf')
        factory.assert_called_once_with(model_path='model.gguf', n_ctx=4096,
            n_threads=8, n_gpu_layers=-1, verbose=True)
        self.assertIs(result, factory.return_value)
        self.assertEqual(fallback.startup_status()['qwen']['path'], 'original_gpu_settings')

    def test_llama_cpu_attempt_only_after_original_constructor_raises(self):
        result = object()
        factory = Mock(side_effect=[RuntimeError('GPU unavailable'), result])
        self.assertIs(fallback.load_llama(factory, 'model.gguf'), result)
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(factory.call_args.kwargs, dict(model_path='model.gguf',
            n_ctx=4096, n_threads=8, n_gpu_layers=0, offload_kqv=False, verbose=True))
        self.assertEqual(fallback.startup_status()['qwen']['path'], 'cpu_fallback')

    def test_cpu_failure_propagates(self):
        factory = Mock(side_effect=[RuntimeError('GPU failure'), ValueError('Invalid model')])
        with self.assertRaisesRegex(ValueError, 'Invalid model'):
            fallback.load_llama(factory, 'model.gguf')
        self.assertEqual(factory.call_count, 2)
        self.assertNotIn('qwen', fallback.startup_status())

    def session(self, *, available=None, fail_gpu=False, silent=False, profile=False):
        attempts = []
        sessions = []
        def constructor(path, sess_options, providers):
            attempts.append(providers)
            name = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
            if fail_gpu and name == 'DmlExecutionProvider':
                raise RuntimeError('GPU allocation failed')
            actual = 'CPUExecutionProvider' if silent else name
            session = types.SimpleNamespace(get_providers=lambda: [actual], disable_fallback=Mock())
            sessions.append((session, sess_options))
            return session
        ort = types.SimpleNamespace(
            get_available_providers=lambda: available if available is not None else ['DmlExecutionProvider','CPUExecutionProvider'],
            SessionOptions=types.SimpleNamespace,
            ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL='sequential'),
            GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL='all'),
            InferenceSession=constructor)
        timing = types.SimpleNamespace(emit=lambda *a, **kw: None, span=lambda *a, **kw: nullcontext())
        modules = {'onnxruntime': ort,
            'rag.domain_config': types.SimpleNamespace(ROOT_DIR=ROOT, RAGUnavailableError=RuntimeError),
            'rag.timing': timing,
            'rag.hardware_diagnostics': types.SimpleNamespace(memory_stage=lambda *a, **kw: nullcontext()),
            'rag.inference_fallback': fallback}
        spec = importlib.util.spec_from_file_location('rag.fallback_runtime_test', ROOT / 'rag/gpu_runtime.py')
        runtime = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(runtime)
            result = runtime.make_session(ROOT/'embedding/model.onnx', device_id=2, profile=profile)
        return attempts, sessions, result

    def test_working_directml_keeps_adapter_and_options(self):
        attempts, sessions, result = self.session()
        self.assertEqual(attempts, [[('DmlExecutionProvider', {'device_id':'2'})]])
        options = sessions[0][1]
        self.assertFalse(options.enable_mem_pattern)
        self.assertEqual(options.execution_mode, 'sequential')
        self.assertEqual(options.graph_optimization_level, 'all')
        self.assertFalse(options.enable_profiling)
        result.disable_fallback.assert_called_once()

    def test_directml_failure_tries_cpu_once(self):
        attempts, _, result = self.session(fail_gpu=True)
        self.assertEqual(attempts, [[('DmlExecutionProvider', {'device_id':'2'})], ['CPUExecutionProvider']])
        self.assertEqual(result.get_providers(), ['CPUExecutionProvider'])
        self.assertEqual(fallback.startup_status()['embedding']['path'], 'cpu_fallback')

    def test_missing_directml_uses_cpu(self):
        attempts, _, _ = self.session(available=['CPUExecutionProvider'])
        self.assertEqual(attempts, [['CPUExecutionProvider']])

    def test_silent_onnx_cpu_substitution_is_made_explicit(self):
        attempts, _, _ = self.session(silent=True)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(fallback.startup_status()['embedding']['path'], 'cpu_fallback')

    def test_export_validation_never_uses_cpu(self):
        with self.assertRaisesRegex(RuntimeError, 'GPU allocation failed'):
            self.session(fail_gpu=True, profile=True)
        self.assertNotIn('embedding', fallback.startup_status())

    def test_missing_all_onnx_providers_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'CPUExecutionProvider is unavailable'):
            self.session(available=[])

    def test_returned_status_cannot_mutate_recorded_status(self):
        fallback.load_llama(Mock(return_value=object()), 'model.gguf')
        snapshot = fallback.startup_status()
        snapshot['qwen']['path'] = 'changed'
        self.assertEqual(fallback.startup_status()['qwen']['path'], 'original_gpu_settings')


if __name__ == '__main__':
    unittest.main()
