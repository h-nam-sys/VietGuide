import importlib.util
import json
from pathlib import Path
import subprocess
from unittest.mock import patch
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('bootstrap', ROOT/'deployment/container/bootstrap_inference.py')
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)

def assets(root):
    folder = root/'models/rag_directml/abc123'
    folder.mkdir(parents=True)
    (folder.parent/'current.json').write_text(json.dumps({'generation':'abc123'}))
    (folder/'validated.json').write_text(json.dumps({'onnxruntime_version':'1.23.2'}))

def test_existing_environment_is_untouched(tmp_path):
    python = tmp_path/'env/Scripts/python.exe'
    python.parent.mkdir(parents=True)
    python.touch()
    with patch.object(bootstrap.subprocess, 'run') as run:
        assert bootstrap.ensure(tmp_path) is False
        run.assert_not_called()

def test_new_environment_installs_matching_runtime(tmp_path):
    assets(tmp_path)
    calls=[]
    def run(command, **kwargs):
        calls.append(command)
        if command[1:3] == ['-m','venv']:
            python=tmp_path/'env/Scripts/python.exe'
            python.parent.mkdir(parents=True)
            python.touch()
        if 'install' in command:
            constraints=Path(command[command.index('-c')+1]).read_text()
            assert constraints == 'onnxruntime-directml==1.23.2\n'
    with patch.object(bootstrap.subprocess, 'run', side_effect=run):
        assert bootstrap.ensure(tmp_path) is True
    assert len(calls)==3
    assert calls[-1][-2:]==['pip','check']
    assert not (tmp_path/'env/.hybrid_install_pending').exists()

def test_failed_install_can_resume(tmp_path):
    assets(tmp_path)
    python=tmp_path/'env/Scripts/python.exe'
    python.parent.mkdir(parents=True)
    python.touch()
    pending=python.parents[1]/'.hybrid_install_pending'
    pending.touch()
    with patch.object(bootstrap.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'pip')):
        with pytest.raises(subprocess.CalledProcessError):
            bootstrap.ensure(tmp_path)
    assert pending.exists()
    with patch.object(bootstrap.subprocess, 'run') as run:
        bootstrap.ensure(tmp_path)
    assert len(run.call_args_list)==2
    assert not pending.exists()

def test_embedding_adapter_preserves_existing_runtime(monkeypatch):
    import sys
    from types import SimpleNamespace
    import rag.inference_embedding as embedding
    meta={'embedding':{'normalize':True},'device_id':2,'precision':'fp16'}
    monkeypatch.setattr(embedding,'read_active',lambda:(Path('/models/current'),meta))
    sessions=[]
    monkeypatch.setattr(embedding,'make_session',lambda path,**kw:sessions.append((path,kw)) or 'session')
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:'tokenizer')))
    calls=[]
    monkeypatch.setattr(embedding,'encode_embeddings',lambda *args:calls.append(args) or [[1.,2.]])
    adapter=embedding.LocalInferenceEmbedding()
    assert adapter.get_query_embedding('query')==[1.,2.]
    assert adapter.get_text_embedding_batch(['text'])==[[1.,2.]]
    assert sessions==[(Path('/models/current/embedding/model.onnx'),{'device_id':2,'precision':'fp16'})]
    assert calls==[('session','tokenizer',['query'],meta['embedding'],'query'),('session','tokenizer',['text'],meta['embedding'],'text')]
