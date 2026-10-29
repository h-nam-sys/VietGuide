"""Deployment boundaries with real files/SQLite/ASGI; no GPU or live DNS required."""
import asyncio
from collections import Counter, deque
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
import zipfile

import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'backend')]
from rag.process_lock import process_lock
from rag.request_capacity import PrioritySlots
from rag.operations import RequestBodyLimitMiddleware, OperationsMiddleware
from rag import operations
from rag.deployment_config import model_assets, database_path
spec=importlib.util.spec_from_file_location('deployment_manage',ROOT/'deployment/manage.py')
manage=importlib.util.module_from_spec(spec);spec.loader.exec_module(manage)


def test_priority_and_cancellation():
    async def run():
        slots=PrioritySlots(1)
        await slots.acquire()
        order=[]
        async def work(label,priority):
            await slots.acquire(priority)
            order.append(label)
            slots.release()
        background=asyncio.create_task(work('ingestion',1))
        await asyncio.sleep(0)
        cancelled=asyncio.create_task(work('cancelled',0))
        await asyncio.sleep(0)
        foreground=asyncio.create_task(work('chat',0))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError): await cancelled
        slots.release()
        await asyncio.gather(background,foreground)
        assert order==['chat','ingestion'] and slots.used==0 and not slots.waiters
    asyncio.run(run())


@pytest.mark.parametrize('kind',['header','chunk','timeout','ok'])
def test_bounded_incoming_body(monkeypatch,kind):
    monkeypatch.setenv('API_MAX_BODY_BYTES','1024')
    monkeypatch.setenv('API_BODY_TIMEOUT','0.01')
    async def run():
        sent=[];called=[]
        async def send(event):sent.append(event)
        async def receive():
            if kind=='timeout': await asyncio.sleep(1)
            return {'type':'http.request','body':b'x'*(1025 if kind=='chunk' else 3),'more_body':False}
        async def app(scope,receive,send): called.append((await receive())['body'])
        gate=RequestBodyLimitMiddleware(app)
        await gate({'type':'http','method':'POST','headers':[(b'content-length',b'2000')] if kind=='header' else []},receive,send)
        if kind=='ok':assert called==[b'xxx']
        else:
            assert not called
            assert sent[0]['status']==(408 if kind=='timeout' else 413)
    asyncio.run(run())


def test_metrics_ignore_metadata_and_do_not_store_answers(monkeypatch):
    monkeypatch.setattr(operations,'_counts',Counter())
    monkeypatch.setattr(operations,'_durations',deque(maxlen=200))
    monkeypatch.setattr(operations,'_first_text',deque(maxlen=200))
    async def run():
        async def app(scope,receive,send):
            await send({'type':'http.response.start','status':200})
            for data in ['{"type":"metadata","sources":[]}','{"type":"chunk","text":"private answer"}','[DONE]']:
                await send({'type':'http.response.body','body':('data: '+data+'\n\n').encode()})
        async def send(event):pass
        await OperationsMiddleware(app,service='backend')({'type':'http','method':'POST','path':'/chat/1'},None,send)
    asyncio.run(run())
    report=operations.snapshot()
    assert report['counts']['requests']==1 and report['first_answer_text']['samples']==1
    assert 'private answer' not in json.dumps(report)


def test_lock_released_after_exception(tmp_path):
    path=tmp_path/'worker.lock'
    with pytest.raises(RuntimeError):
        with process_lock(path):
            with pytest.raises(TimeoutError):
                with process_lock(path):pass
            raise RuntimeError('worker failure')
    with process_lock(path):pass


def fake_runtime(root):
    (root/'backend/static/forms').mkdir(parents=True)
    (root/'backend/static/forms/form.docx').write_bytes(b'fake-public-form')
    (root/'backend/prompts.json').write_text('{}')
    (root/'data/session_secret.key').parent.mkdir(parents=True,exist_ok=True)
    (root/'data/session_secret.key').write_text('secret-for-test')
    (root/'.env').write_text('ADMIN_PASSWORD=test-only')
    with sqlite3.connect(root/'backend/legal_llm.db') as db:
        db.execute('CREATE TABLE conversations (id INTEGER, text TEXT)')
        db.execute("INSERT INTO conversations VALUES (1,'saved chat')")
    folder=root/'data/rag_generations/demo'/('a'*32)
    (folder/'vectors').mkdir(parents=True)
    (folder/'vectors/index.json').write_text('{"vectors":[1,2]}')
    (folder/'profile.json').write_text('{}')
    (folder.parent/'current.json').write_text(json.dumps({'generation':folder.name}))
    with sqlite3.connect(folder/'records.db') as db:
        db.execute('CREATE TABLE records (id INTEGER)')
    return root


def test_real_backup_restore_and_retention(tmp_path,monkeypatch):
    root=fake_runtime(tmp_path/'project')
    monkeypatch.delenv('DATABASE_URL',raising=False)
    monkeypatch.setattr(manage,'require_stopped',lambda *a:None)
    destination=tmp_path/'backups'
    first=manage.backup(root,destination,keep=1)
    restored=tmp_path/'restored'
    report=manage.restore_check(first,restored)
    assert report['sqlite_databases']==2 and report['active_datasets']==1
    assert (restored/'data/session_secret.key').read_text()=='secret-for-test'
    assert (restored/'backend/static/forms/form.docx').read_bytes()==b'fake-public-form'
    with sqlite3.connect(restored/'backend/legal_llm.db') as db:
        assert db.execute('SELECT text FROM conversations').fetchone()[0]=='saved chat'
    with pytest.raises(ValueError): manage.restore_check(first,restored)
    second=manage.backup(root,destination,keep=1)
    assert second.is_file() and not first.exists()
    assert (root/'backend/legal_llm.db').is_file()


def test_backup_does_not_race_launcher(tmp_path,monkeypatch):
    root=fake_runtime(tmp_path/'project')
    monkeypatch.setattr(manage,'require_stopped',lambda *a:None)
    with process_lock(root/'data/ops_locks/services.lock'):
        with pytest.raises(TimeoutError):manage.backup(root,tmp_path/'backups')


@pytest.mark.parametrize('name',['../outside','/absolute','C:/outside','DATA/x','back\\slash'])
def test_restore_rejects_unsafe_or_case_colliding_archive(tmp_path,name):
    path=tmp_path/'bad.zip'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr(name,'bad')
        z.writestr('data/x','safe')
        z.writestr('BACKUP_MANIFEST.json',json.dumps({'schema':1,'chat_database':'data/x','files':{name:'fake','data/x':'fake'}}))
    with pytest.raises(ValueError): manage.restore_check(path,tmp_path/'target')
    assert not (tmp_path/'target').exists()


def test_asset_readiness_no_model_construction(tmp_path,monkeypatch):
    monkeypatch.delenv('MODEL_PATH',raising=False);monkeypatch.delenv('RAG_GPU_MODEL_DIR',raising=False)
    assert not model_assets(tmp_path)['retrieval_assets']
    (tmp_path/'models').mkdir()
    (tmp_path/'models/qwen3.5_4b.gguf').write_bytes(b'model')
    gpu=tmp_path/'models/rag_directml'
    folder=gpu/'abc123'
    for kind in ['embedding','reranker']:
        (folder/kind).mkdir(parents=True)
        (folder/kind/'model.onnx').write_bytes(b'not-loaded')
    (gpu/'current.json').write_text('{"generation":"abc123"}')
    (folder/'validated.json').write_text(json.dumps({'schema':1,'provider':'DmlExecutionProvider','onnxruntime_version':'test'}))
    assert all(model_assets(tmp_path).values())
    fake=NS(__version__='test',get_available_providers=lambda:['CPUExecutionProvider'])
    monkeypatch.setitem(sys.modules,'onnxruntime',fake)
    assert not model_assets(tmp_path,check_runtime=True)['retrieval_assets']
    fake.get_available_providers=lambda:['DmlExecutionProvider']
    assert model_assets(tmp_path,check_runtime=True)['retrieval_assets']


def test_readiness_checks_dependencies_without_inference(tmp_path,monkeypatch):
    from services import operations_service as svc
    vector=tmp_path/'vectors';vector.mkdir()
    database=tmp_path/'records.db';database.write_bytes(b'')
    monkeypatch.setattr(svc,'list_domain_configs',lambda:[NS(db_path=database,vector_dir=vector)])
    monkeypatch.setattr(svc,'model_status',AsyncMock(return_value={'qwen_ready':True,'retrieval_assets_ready':True}))
    assert asyncio.run(svc.readiness(NS(execute=AsyncMock())))['ready']
    monkeypatch.setattr(svc,'model_status',AsyncMock(return_value=None))
    report=asyncio.run(svc.readiness(NS(execute=AsyncMock())))
    assert not report['ready'] and not report['checks']['model']


def test_production_commands_and_optional_proxy(monkeypatch):
    import run_services as launch
    monkeypatch.delenv('CADDY_BINARY',raising=False)
    development=launch.service_definitions(False)
    production=launch.service_definitions(True)
    assert '--reload' in development[1][2] and '--reload' not in production[1][2]
    assert '--proxy-headers' in production[1][2]
    monkeypatch.setenv('CADDY_BINARY','C:/tools/caddy.exe')
    assert launch.service_definitions(True)[-1][0]=='Proxy'


@pytest.mark.parametrize('production,limit,expected',[(True,'3',2),(True,'0',1),(False,'3',1)])
def test_supervisor_stops_old_process_before_replacement(tmp_path,monkeypatch,production,limit,expected):
    import run_services as launch
    monkeypatch.setattr(launch,'ROOT',tmp_path)
    monkeypatch.setenv('SERVICE_RESTART_LIMIT',limit)
    monkeypatch.setattr(launch,'runtime_manifest',lambda:{})
    monkeypatch.setattr(launch,'prune_logs',lambda:None)
    monkeypatch.setattr(launch,'service_definitions',lambda *a:[('LLM',tmp_path,['unused'],8123,'/health')])
    class Socket:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def settimeout(self,*a):pass
        def connect_ex(self,*a):return 111
    monkeypatch.setattr(launch.socket,'socket',Socket)
    class Stop:
        def __init__(self):self.flag=False;self.calls=0
        def wait(self,*a):self.calls+=1;return self.flag or self.calls>=3
        def is_set(self):return self.flag
        def set(self):self.flag=True
    monkeypatch.setattr(launch,'STOP',Stop())
    class Thread:
        def __init__(self,*a,**k):pass
        def start(self):pass
        def join(self,*a,**k):pass
    monkeypatch.setattr(launch.threading,'Thread',Thread)
    events=[]
    def spawn(*a,**k):
        index=sum(e.startswith('spawn') for e in events)+1
        events.append('spawn'+str(index))
        return NS(pid=index,returncode=1 if index==1 else None,poll=lambda:1 if index==1 else None)
    monkeypatch.setattr(launch.subprocess,'Popen',spawn)
    monkeypatch.setattr(launch,'stop_process',lambda p:events.append('stop'+str(p.pid)))
    launch.run_group(production,lambda *a:None)
    assert sum(e.startswith('spawn') for e in events)==expected
    if expected==2:assert events.index('stop1')<events.index('spawn2')


def test_first_answer_measurement_preserves_qa_parser():
    import qa_suite
    data=[b'data: {"type":"metadata","sources":[]}\n',b'\n',
          b'data: {"type":"chunk","text":"answer"}\n',b'\n',b'data: [DONE]\n',b'\n']
    result=qa_suite.parse_sse(data)
    assert result['answer']=='answer' and result['done']
    assert result['first_text_ms'] is not None and result['first_text_ms']>=0


def test_model_http_timeout_and_ingestion_priority(monkeypatch):
    import ast
    import httpx
    source=ast.parse((ROOT/'rag/gpu_client.py').read_text())
    nodes=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in {'_client','gpu_call'}]
    # Run the actual boundary functions without importing model/ONNX dependencies.
    from functools import lru_cache
    from contextlib import nullcontext
    headers=[]
    original=httpx.Client
    def client(**kwargs):
        assert kwargs['timeout'].read==17
        assert kwargs['timeout'].connect==5
        return original(transport=httpx.MockTransport(lambda request:(headers.append(dict(request.headers)) or httpx.Response(200,json={'ok':True}))),**kwargs)
    monkeypatch.setattr(httpx,'Client',client)
    monkeypatch.setenv('GPU_SERVICE_TIMEOUT','17')
    monkeypatch.setenv('LEGAL_INGESTION','1')
    ns={'os':os,'lru_cache':lru_cache,'span':lambda *a:nullcontext(),'trace_headers':lambda:{},
        'llm_url':lambda:'http://local-model','RAGUnavailableError':RuntimeError}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'gpu_boundary','exec'),ns)
    assert ns['gpu_call']('/v1/rag/embeddings',{})=={'ok':True}
    assert headers[0]['x-inference-priority']=='background'
    monkeypatch.delenv('LEGAL_INGESTION')
    ns['gpu_call']('/v1/rag/embeddings',{})
    assert 'x-inference-priority' not in headers[1]
    ns['_client']('http://local-model').close()


def test_process_lock_released_when_worker_is_killed(tmp_path):
    import subprocess
    lock=tmp_path/'cross-process.lock'
    code='from rag.process_lock import process_lock\nimport time\nwith process_lock('+repr(str(lock))+'):\n print("locked",flush=True)\n time.sleep(30)\n'
    child=subprocess.Popen([sys.executable,'-c',code],cwd=ROOT,stdout=subprocess.PIPE,text=True)
    try:
        assert child.stdout.readline().strip()=='locked'
        with pytest.raises(TimeoutError):
            with process_lock(lock):pass
    finally:
        child.terminate();child.wait(timeout=5)
    with process_lock(lock):pass


def test_load_tool_preserves_results_and_first_text(tmp_path,monkeypatch):
    import qa_suite
    monkeypatch.delenv('ADMIN_PASSWORD',raising=False)
    class Client:
        def __init__(self,*a,**k):pass
        def request(self,path,payload=None,stream=False):
            if path=='/start-conversation':return {'conversation_id':1},'trace',0
            assert payload['bypass_answer_cache'] is True
            return {'done':True,'stream_errors':[],'answer':'answer','first_text_ms':12},'trace',20
    monkeypatch.setattr(qa_suite,'Client',Client)
    output=tmp_path/'results.json'
    report=manage.load_test('http://local','question',2,1,output)
    assert report['successful']==2 and report['first_answer_text']['mean_ms']==12
    saved=json.loads(output.read_text())
    assert len(saved['results'])==2 and saved['results'][0]['response']['answer']=='answer'
    assert saved['observed_peak_model_rss_mb'] is None


def test_only_one_ingestion_worker_runs_across_domains(tmp_path,monkeypatch):
    import threading
    import time
    from rag import ingestion_jobs as jobs
    folder=tmp_path/'data/admin_jobs';folder.mkdir(parents=True)
    paths=[]
    for name in ('first','second'):
        path=folder/(name+'.json')
        path.write_text(json.dumps({'id':name,'domain':name,'status':'queued'}));paths.append(path)
    entered,release=threading.Event(),threading.Event()
    order=[]
    def run(path,**kwargs):
        order.append(Path(path).stem)
        if Path(path).stem=='first':
            entered.set();release.wait(2)
        return True
    monkeypatch.setattr(jobs,'_run_job',run)
    first=threading.Thread(target=jobs.run_job,args=(paths[0],))
    second=threading.Thread(target=jobs.run_job,args=(paths[1],))
    first.start();assert entered.wait(2);second.start()
    try:
        time.sleep(.05)
        assert order==['first']
        assert json.loads(paths[1].read_text())['stage']=='Waiting for ingestion slot'
    finally:
        release.set();first.join(2);second.join(2)
    assert order==['first','second'] and not first.is_alive() and not second.is_alive()


def test_native_model_stream_is_measured_without_changing_chunks(monkeypatch):
    monkeypatch.setattr(operations,'_counts',Counter())
    monkeypatch.setattr(operations,'_first_text',deque(maxlen=200))
    sent=[]
    async def run():
        async def app(scope,receive,send):
            await send({'type':'http.response.start','status':200})
            await send({'type':'http.response.body','body':b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'})
        async def send(event):sent.append(event)
        await OperationsMiddleware(app,service='llm')({'type':'http','method':'POST','path':'/v1/chat/completions'},None,send)
    asyncio.run(run())
    assert b'"content":"hi"' in sent[-1]['body']
    assert operations.snapshot()['first_answer_text']['samples']==1
