"""Lifecycle simulation; live Docker/Windows/GPU testing is separate."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import threading

import pytest

from deployment.container import hybrid, configure
import run_services


class AutoStop(threading.Event):
    def wait(self, timeout=None):
        if timeout == 5:  # End the monitoring loop after a successful startup.
            self.set()
            return True
        return super().wait(timeout)


class FakeDocker:
    def __init__(self, existing=None, *, probe_error=False, base_missing=False):
        self.current = copy.deepcopy(existing)
        self.calls = []
        self.probe_error = probe_error
        self.base_missing = base_missing

    def state(self):
        return copy.deepcopy(self.current)

    def call(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if args[:2] == ['image','inspect']:
            return NS(returncode=int(self.base_missing), stdout='', stderr='')
        if args[:2] == ['stop','app']:
            self.current = None
        if args[0] == 'run' and self.probe_error:
            raise RuntimeError('Connection refused')
        if args[0] == 'up':
            if not self.current or not self.current['running']:
                self.current = {'id':'new-app','running':True,'command':['--web-only'],
                                'state':{'Health':{'Status':'healthy'}}}
        return NS(returncode=0, stdout='', stderr='')


def execute(tmp_path, monkeypatch, *, reuse_model=False, existing=None, probe_error=False, base_missing=False):
    stop = AutoStop()
    ready = threading.Event()
    if reuse_model:
        ready.set()
    docker = FakeDocker(existing, probe_error=probe_error, base_missing=base_missing)
    events, starts = [], []
    def supervise(production, alert, **kwargs):
        starts.append(kwargs)
        ready.set()
        while not stop.wait(.01):
            pass
        return 0
    launcher = NS(ROOT=tmp_path, STOP=stop, show=lambda *args: events.append(args),
        open_chat_browser=lambda: events.append(('Browser','opened')),
        runtime_manifest=lambda: {'packages':{}}, run_group=supervise,
        service_definitions=lambda production: [('LLM',tmp_path,['python','model.py'],8001,'/v1/rag/status'),
                                              ('Backend',tmp_path,['python','backend.py'],8000,'/health')])
    monkeypatch.setattr(hybrid, 'Docker', lambda root, port: docker)
    monkeypatch.setattr(hybrid, 'llm_port', lambda: 8001)
    monkeypatch.setattr(hybrid, 'ensure_engine', lambda *args: None)
    monkeypatch.setattr(hybrid, 'port_open', lambda port: reuse_model and port == 8001)
    monkeypatch.setattr(hybrid, 'status', lambda *a, **kw: (
        {'gpu_protocol':1,'qwen_ready':True,'retrieval_assets_ready':True} if ready.is_set() else None))
    monkeypatch.setattr(configure, 'prepare', lambda root: {})
    result = hybrid.run_hybrid(launcher, lambda *a: None)
    return result, starts, docker, events


def test_owned_model_and_container_start_and_stop_once(tmp_path, monkeypatch):
    result, starts, docker, events = execute(tmp_path, monkeypatch)
    assert result == 0
    assert len(starts) == 1
    assert events.count(('Browser','opened')) == 1
    assert [item[0] for item in starts[0]['definitions']] == ['LLM']
    assert starts[0]['environment_overrides'] == {'LLM_BIND_HOST':'0.0.0.0'}
    assert any(args == ['stop','--time','45','new-app'] for args, _ in docker.calls)
    assert sum(args[0] == 'up' for args, _ in docker.calls) == 1
    assert not any(args[:2] == ['compose','-f'] for args, _ in docker.calls)


def test_reused_model_and_container_are_left_running(tmp_path, monkeypatch):
    existing = {'id':'existing-app','running':True,'command':['--web-only'],
                'state':{'Health':{'Status':'healthy'}}}
    result, starts, docker, _ = execute(tmp_path, monkeypatch, reuse_model=True, existing=existing)
    assert result == 0
    assert not starts
    assert not any(args[0] == 'stop' for args, _ in docker.calls)


def test_old_cpu_container_is_stopped_before_native_model(tmp_path, monkeypatch):
    existing = {'id':'old-cpu','running':True,'command':None,'state':{}}
    result, starts, docker, _ = execute(tmp_path, monkeypatch, existing=existing)
    assert result == 0 and len(starts) == 1
    stop_index = next(i for i, (args, _) in enumerate(docker.calls) if args == ['stop','app'])
    build_index = next(i for i, (args, _) in enumerate(docker.calls) if args == ['build','app'])
    assert stop_index < build_index


def test_failed_container_probe_does_not_start_web_app(tmp_path, monkeypatch):
    result, starts, docker, events = execute(tmp_path, monkeypatch, probe_error=True)
    assert result == 1 and len(starts) == 1
    assert not any(args[0] == 'up' for args, _ in docker.calls)
    assert any('cannot reach Windows inference' in text for _, text in events)


def test_missing_base_builds_it_once_before_hybrid(tmp_path, monkeypatch):
    result, _, docker, _ = execute(tmp_path, monkeypatch, base_missing=True)
    assert result == 0
    assert sum(args == ['compose','-f','compose.yaml','build','app'] for args, _ in docker.calls) == 1


def test_occupied_unready_model_port_never_starts_duplicate(tmp_path, monkeypatch):
    docker = FakeDocker()
    monkeypatch.setattr(hybrid, 'Docker', lambda *a: docker)
    monkeypatch.setattr(hybrid, 'ensure_engine', lambda *a: None)
    monkeypatch.setattr(hybrid, 'port_open', lambda port: port == 8001)
    monkeypatch.setattr(hybrid, 'status', lambda *a: None)
    launcher = NS(ROOT=tmp_path, STOP=threading.Event(), show=lambda *a: None)
    assert hybrid.run_hybrid(launcher, lambda *a: None) == 1
    assert not any(args[0] in {'up','build'} for args, _ in docker.calls)


def test_web_only_definition_does_not_start_model(monkeypatch):
    monkeypatch.delenv('CADDY_BINARY', raising=False)
    assert [s[0] for s in run_services.service_definitions(True, web_only=True)] == ['Backend','Admin']
    assert [s[0] for s in run_services.service_definitions(True)] == ['LLM','Backend','Admin']


def test_native_binding_defaults_unchanged(monkeypatch):
    monkeypatch.delenv('BACKEND_BIND_HOST', raising=False)
    monkeypatch.delenv('ADMIN_BIND_HOST', raising=False)
    definitions = run_services.service_definitions(True)
    for definition, flag in zip(definitions[1:], ('--host','--server.address')):
        command = definition[2]
        assert command[command.index(flag)+1] == '127.0.0.1'


def test_host_inference_preflight_does_not_require_admin_packages(tmp_path, monkeypatch):
    from rag import deployment_config as config
    frontend = tmp_path/'backend/static/index.html'
    frontend.parent.mkdir(parents=True)
    frontend.write_text('frontend')
    monkeypatch.setenv('ADMIN_PASSWORD','test-password')
    monkeypatch.setenv('LLM_SERVICE_URLS','http://localhost:8001')
    monkeypatch.delenv('CADDY_BINARY', raising=False)
    monkeypatch.delenv('PUBLIC_HOST', raising=False)
    monkeypatch.setattr(config,'model_assets',lambda root: {'qwen_file':True,'retrieval_assets':True})
    checked = []
    def find(name):
        checked.append(name)
        return None if name in {'streamlit','sqlalchemy','aiosqlite'} else object()
    monkeypatch.setattr(config.importlib.util, 'find_spec', find)
    assert config.validate(tmp_path, True, include_web=False)['ok']
    assert not {'streamlit','sqlalchemy','aiosqlite'} & set(checked)
    assert not config.validate(tmp_path, True)['ok']


def test_container_preflight_does_not_require_model_runtime(tmp_path, monkeypatch):
    from rag import deployment_config as config
    frontend = tmp_path/'backend/static/index.html'
    frontend.parent.mkdir(parents=True)
    frontend.write_text('frontend')
    monkeypatch.setenv('ADMIN_PASSWORD','test-password')
    monkeypatch.setenv('LLM_SERVICE_URLS','http://host.docker.internal:8001')
    monkeypatch.delenv('CADDY_BINARY', raising=False)
    monkeypatch.delenv('PUBLIC_HOST', raising=False)
    def no_models(root):
        raise AssertionError('Web-only mode must not inspect local model assets')
    monkeypatch.setattr(config, 'model_assets', no_models)
    checked = []
    def find(name):
        checked.append(name)
        return object()
    monkeypatch.setattr(config.importlib.util, 'find_spec', find)
    assert config.validate(tmp_path, True, include_inference=False)['ok']
    assert not {'llama_cpp','onnxruntime'} & set(checked)
