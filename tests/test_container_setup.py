"""Local setup checks; Docker build/start still needs the target Docker engine."""
import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'deployment/container'/f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


entry = module('entrypoint')
configure = module('configure')
requirements = module('build_requirements')


def native_project(root):
    root.mkdir(parents=True, exist_ok=True)
    gpu = root/'models/rag_directml'
    for kind in ('embedding','reranker'):
        path = gpu/'abc123'/kind/'model.onnx'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'model')
    (gpu/'current.json').write_text('{"generation":"abc123"}')
    (gpu/'abc123/validated.json').write_text(json.dumps({'onnxruntime_version':'1.23.2'}))
    (root/'models/qwen3.5_4b.gguf').write_bytes(b'gguf')
    (root/'data').mkdir()
    (root/'data/domain_profile.json').write_text('{"domain_name":"demo"}')
    (root/'data/session_secret.key').write_text('preserved-session-key')
    backend = root/'backend'
    (backend/'static/forms').mkdir(parents=True)
    (backend/'static/forms/example.docx').write_bytes(b'form')
    with sqlite3.connect(backend/'legal_llm.db') as db:
        db.execute('CREATE TABLE sample(value TEXT)')
        db.execute('INSERT INTO sample VALUES (?)', ('original history',))
    (backend/'all_feedback_dataset.jsonl').write_text('{"rating":1}\n')
    (backend/'.env').write_text('SECRET=do-not-copy')
    log = root/'logs/operations/runtime_dependencies.json'
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps({'packages': {'llama-cpp-python':'0.3.36'}}))
    return root


def test_build_pins_match_validated_onnx_and_native_llama(tmp_path):
    root = native_project(tmp_path/'native')
    log = root/'logs/operations/runtime_dependencies.json'
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({'packages': {'llama_cpp_python':'0.3.35',
        'numpy':'2.2.6','torch':'2.7.1+cu128','onnxruntime-directml':'1.23.2'}}))
    original = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    report = configure.prepare(root)
    assert report['onnxruntime'] == '1.23.2'
    assert report['llama_cpp_python'] == '0.3.35'
    folder = root/'deployment/container'
    assert 'torch==2.7.1\n' in (folder/'constraints.generated.txt').read_text()
    assert 'onnxruntime==1.23.2' in (folder/'constraints.generated.txt').read_text()
    assert 'do-not-copy' not in (folder/'constraints.generated.txt').read_text()
    for name, content in original.items():
        assert (root/name).read_bytes() == content


def test_missing_assets_fail_before_build(tmp_path):
    root = native_project(tmp_path/'native')
    (root/'models/qwen3.5_4b.gguf').unlink()
    with pytest.raises(ValueError, match='Required runtime asset'):
        configure.prepare(root)


def test_missing_native_runtime_snapshot_fails_instead_of_upgrading(tmp_path):
    root = native_project(tmp_path/'native')
    (root/'logs/operations/runtime_dependencies.json').unlink()
    with pytest.raises(ValueError, match='Native runtime snapshot is missing'):
        configure.prepare(root)


def test_pointer_cannot_escape_models_folder(tmp_path):
    root = native_project(tmp_path/'native')
    (root/'models/rag_directml/current.json').write_text('{"generation":"../outside"}')
    with pytest.raises(ValueError, match='Invalid retrieval'):
        configure.prepare(root)


def test_linux_requirements_drop_windows_wheel_policy_and_keep_cpu_runtime():
    result = requirements.serving_requirements((ROOT/'requirements.txt').read_text())
    assert '--extra-index-url' not in result
    assert '--only-binary' not in result
    assert 'onnxruntime-directml' not in result
    assert 'llama-cpp-python' not in result
    assert 'onnxruntime>=' in result
    assert 'FlagEmbedding' in result


def test_first_seed_copies_history_forms_and_session_key(tmp_path):
    native = native_project(tmp_path/'native')
    app = tmp_path/'container'
    before = {p.relative_to(native): p.read_bytes() for p in native.rglob('*') if p.is_file()}
    assert entry.seed_runtime(app, native)
    with sqlite3.connect(app/'data/docker_runtime/legal_llm.db') as db:
        assert db.execute('SELECT value FROM sample').fetchone()[0] == 'original history'
    assert (app/'data/session_secret.key').read_text() == 'preserved-session-key'
    assert (app/'backend/static/forms/example.docx').read_bytes() == b'form'
    assert not (app/'data/docker_runtime/.env').exists()
    for name, content in before.items():
        assert (native/name).read_bytes() == content


def test_restart_does_not_overwrite_docker_edits(tmp_path):
    native = native_project(tmp_path/'native')
    app = tmp_path/'container'
    entry.seed_runtime(app, native)
    (app/'data/domain_profile.json').write_text('docker edits')
    (native/'data/domain_profile.json').write_text('later native edits')
    assert not entry.seed_runtime(app, native)
    assert (app/'data/domain_profile.json').read_text() == 'docker edits'


def test_nonempty_uninitialized_volume_is_not_overwritten(tmp_path):
    native = native_project(tmp_path/'native')
    app = tmp_path/'container'
    (app/'data').mkdir(parents=True)
    (app/'data/keep.txt').write_text('keep')
    with pytest.raises(RuntimeError, match='refusing to overwrite'):
        entry.seed_runtime(app, native)
    assert (app/'data/keep.txt').read_text() == 'keep'


def test_sqlite_copy_includes_committed_wal(tmp_path):
    source, target = tmp_path/'source.db', tmp_path/'copy.db'
    with sqlite3.connect(source) as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE sample(value INTEGER)')
        db.execute('INSERT INTO sample VALUES (42)')
        db.commit()
        entry.copy_asset(source, target)
    with sqlite3.connect(target) as db:
        assert db.execute('SELECT value FROM sample').fetchone()[0] == 42


def test_symlink_source_is_rejected(tmp_path):
    target = tmp_path/'private'
    target.write_text('private')
    source = tmp_path/'link'
    source.symlink_to(target)
    with pytest.raises(ValueError, match='does not follow symlinks'):
        entry.copy_asset(source, tmp_path/'copy')


def test_seeding_ignores_native_locks(tmp_path):
    native = native_project(tmp_path/'native')
    lock = native/'data/ops_locks/services.lock'
    lock.parent.mkdir()
    lock.write_text('native lock')
    app = tmp_path/'container'
    entry.seed_runtime(app, native)
    assert not (app/'data/ops_locks/services.lock').exists()


def test_native_timeouts_unchanged_and_docker_override_is_bounded(monkeypatch):
    from rag.service_config import model_http_timeout
    monkeypatch.delenv('MODEL_HTTP_TIMEOUT', raising=False)
    assert model_http_timeout() == 30
    assert model_http_timeout(300) == 300
    monkeypatch.setenv('MODEL_HTTP_TIMEOUT', '600')
    assert model_http_timeout() == 600
    assert model_http_timeout(300) == 600
    for value in ('nan','infinity','0','1900','wrong'):
        monkeypatch.setenv('MODEL_HTTP_TIMEOUT', value)
        with pytest.raises(ValueError):
            model_http_timeout()
