"""Operational commands only; never installs/replaces application source."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import threading
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT/'backend/.env')
load_dotenv(ROOT/'.env')
from rag.process_lock import process_lock
from rag.deployment_config import database_path, validate
from rag.service_config import llm_port


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024*1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def require_stopped(root):
    for port in (8000,8501,llm_port()):
        with socket.socket() as sock:
            sock.settimeout(.2)
            if sock.connect_ex(('127.0.0.1',port)) == 0:
                raise RuntimeError('Stop services before backup (port '+str(port)+' is active)')


def backup(root, destination, keep=5):
    root,destination = Path(root).resolve(),Path(destination).resolve()
    if keep < 1:
        raise ValueError('Keep at least one verified backup')
    if destination.is_relative_to(root/'data') or destination.is_relative_to(root/'backend/static/forms'):
        raise ValueError('Store backups outside the data/forms folders')
    require_stopped(root)
    destination.mkdir(parents=True,exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = destination/('legal-runtime-'+stamp+'.zip')
    temporary = output.with_suffix('.tmp')
    with ExitStack() as locks:
        locks.enter_context(process_lock(root/'data/ops_locks/services.lock'))
        locks.enter_context(process_lock(root/'data/ops_locks/ingestion.lock',timeout=2))
        try:
            with tempfile.TemporaryDirectory(dir=destination) as temp:
                stage=Path(temp)
                names={}
                for folder in ('data','backend/static/forms'):
                    for path in (root/folder).rglob('*'):
                        if path.is_symlink():
                            raise ValueError('Backup does not follow symlinks: '+str(path.relative_to(root)))
                        if not path.is_file():
                            continue
                        relative=path.relative_to(root)
                        if relative.parts[:2] in {('data','ops_locks')} or relative.parts[:3] == ('data','admin_jobs','locks'):
                            continue
                        if path.name.endswith(('-wal','-shm')):
                            continue
                        names[relative.as_posix()] = path
                for name in ('.env','backend/.env','backend/prompts.json','backend/all_feedback_dataset.jsonl','all_feedback_dataset.jsonl'):
                    if (root/name).is_file(): names[name]=root/name
                from rag.domain_config import list_domain_configs
                for config in list_domain_configs(root=root,include_inactive=True):
                    for asset in (config.db_path,config.vector_dir):
                        asset=asset.resolve()
                        if not asset.is_relative_to(root):
                            raise ValueError('Dataset assets outside project are not supported by this backup')
                        for path in ([asset] if asset.is_file() else asset.rglob('*')):
                            if path.is_symlink(): raise ValueError('Dataset asset symlinks are not supported')
                            if path.is_file() and not path.name.endswith(('-wal','-shm')):
                                names[path.relative_to(root).as_posix()]=path
                chat=database_path(root)
                if not chat.is_file():
                    raise ValueError('Conversation database not found; check DATABASE_URL')
                if chat.is_relative_to(root):
                    chat_name=chat.relative_to(root).as_posix()
                else:
                    raise ValueError('Database outside project: move/configure it inside the project before this backup')
                names[chat_name]=chat
                # SQLite backup API includes committed WAL content and produces standalone files.
                for name,path in list(names.items()):
                    target=stage/name
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if path.suffix.lower() in {'.db','.sqlite','.sqlite3'}:
                        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as source, sqlite3.connect(target) as out:
                            source.backup(out)
                            if out.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                                raise ValueError('Invalid SQLite database: '+name)
                    else:
                        import shutil
                        shutil.copy2(path,target)
                for pointer in (stage/'data/rag_generations').glob('*/current.json'):
                    generation=json.loads(pointer.read_text())['generation']
                    if not isinstance(generation,str) or not generation.isalnum(): raise ValueError('Invalid active generation')
                    folder=pointer.parent/generation
                    if not (folder/'records.db').is_file() or not (folder/'vectors').is_dir() or not (folder/'profile.json').is_file():
                        raise ValueError('Active dataset is incomplete; backup was not published')
                manifest={'schema':1,'created':stamp,'chat_database':chat_name,
                    'files':{n:digest(stage/n) for n in sorted(names)}}
                (stage/'BACKUP_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
                with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as archive:
                    for path in stage.rglob('*'):
                        if path.is_file(): archive.write(path,path.relative_to(stage).as_posix())
                verify_archive(temporary)
                temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    # Only prune this tool's own archives, after a new backup passes verification.
    candidates=sorted(destination.glob('legal-runtime-*.zip'),reverse=True)
    owned=[]
    for path in candidates:
        try:
            with zipfile.ZipFile(path) as z:
                if json.loads(z.read('BACKUP_MANIFEST.json')).get('schema') == 1: owned.append(path)
        except Exception:
            continue
    for path in owned[keep:]: path.unlink()
    return output


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        seen=set()
        for entry in archive.infolist():
            name=entry.filename
            p=PurePosixPath(name)
            if ('\\' in name or p.is_absolute() or '..' in p.parts or PureWindowsPath(name).drive
                    or name.casefold() in seen or ((entry.external_attr>>16)&0o170000)==0o120000):
                raise ValueError('Unsafe or duplicate backup path')
            seen.add(name.casefold())
        manifest=json.loads(archive.read('BACKUP_MANIFEST.json'))
        if manifest.get('schema') != 1 or not isinstance(manifest.get('files'),dict):
            raise ValueError('Unrecognized backup manifest')
        if manifest.get('chat_database') not in manifest['files']:
            raise ValueError('Chat database is not present in the backup manifest')
        if set(archive.namelist()) != set(manifest['files'])|{'BACKUP_MANIFEST.json'}:
            raise ValueError('Backup contents do not match manifest')
        for name, expected in manifest['files'].items():
            hasher=hashlib.sha256()
            with archive.open(name) as file:
                for block in iter(lambda:file.read(1024*1024),b''): hasher.update(block)
            if hasher.hexdigest()!=expected: raise ValueError('Backup checksum mismatch: '+name)
        return manifest


def restore_check(archive, target):
    manifest=verify_archive(archive)
    target=Path(target).resolve()
    if target.exists():
        raise ValueError('Restore target must be a NEW folder; existing data is never overwritten')
    target.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z: z.extractall(target)
    dbs=0
    for path in target.rglob('*'):
        if path.suffix.lower() in {'.db','.sqlite','.sqlite3'} and path.is_file():
            with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
                if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok': raise ValueError('SQLite restore failed')
            dbs+=1
    if not (target/manifest['chat_database']).is_file(): raise ValueError('Missing conversation database')
    active=0
    for pointer in (target/'data/rag_generations').glob('*/current.json'):
        generation=json.loads(pointer.read_text())['generation']
        if not isinstance(generation,str) or not generation.isalnum(): raise ValueError('Invalid dataset generation')
        folder=pointer.parent/generation
        if not (folder/'records.db').is_file() or not (folder/'vectors').is_dir() or not (folder/'profile.json').is_file():
            raise ValueError('Incomplete restored active dataset: '+pointer.parent.name)
        active+=1
    return {'verified_files':len(manifest['files']),'sqlite_databases':dbs,'active_datasets':active,
            'session_key_in_backup':(target/'data/session_secret.key').is_file(),
            'note':'Data restore verified; model weights and a live GPU check are still required before running a restored system.'}


def probe(base):
    result={}
    for path in ('/health','/readyz'):
        try:
            with urllib.request.urlopen(base.rstrip('/')+path,timeout=10) as response:
                result[path]={'status':response.status,'result':json.load(response)}
        except Exception as exc:
            result[path]={'error':type(exc).__name__}
    return result


def load_test(base,prompt,users,rounds,output):
    if not 1<=users<=8 or not 1<=rounds<=5:
        raise ValueError('Use 1-8 users and 1-5 requests per user for a bounded acceptance run')
    from qa_suite import Client
    def worker(index):
        client=Client(base,'deployment-'+str(index),interval=6.5,timeout=360)
        try:
            cid=client.request('/start-conversation',{'title':'Deployment capacity check'})[0]['conversation_id']
        except Exception as exc:
            return [{'user':index,'turn':0,'stage':'session','ok':False,'error':str(exc)[:500]}]
        results=[]
        for turn in range(rounds):
            started=time.monotonic()
            try:
                # Existing parser preserves real answers and completion/error information.
                response,trace,elapsed=client.request('/chat/'+str(cid),{'user_prompt':prompt,'bypass_answer_cache':True},stream=True)
                ok=response['done'] and not response['stream_errors'] and 'Lỗi hệ thống:' not in response['answer']
                results.append({'user':index,'turn':turn,'ok':ok,'elapsed_ms':elapsed,'first_answer_ms':response.get('first_text_ms'),'trace':trace,'response':response})
            except Exception as exc:
                # HTTP failures are recorded; no spoofed IPs or bypass of production limits.
                results.append({'user':index,'turn':turn,'ok':False,'elapsed_ms':round((time.monotonic()-started)*1000,2),'error':str(exc)[:500]})
        return results
    samples=[]
    stop=threading.Event()
    def sample():
        password=os.getenv('ADMIN_PASSWORD')
        if not password: return
        while not stop.is_set():
            try:
                request=urllib.request.Request(base.rstrip('/')+'/admin/operations',headers={'X-Admin-Key':password})
                with urllib.request.urlopen(request,timeout=10) as response: status=json.load(response)
                samples.append({'time':time.time(),'backend_memory':status['backend'].get('memory'),
                                'model_memory':(status.get('model') or {}).get('memory'),
                                'queues':status['backend'].get('queues')})
            except Exception:
                pass
            stop.wait(2)
    sampler=threading.Thread(target=sample,daemon=True)
    sampler.start()
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=users) as pool:
        results=[r for group in pool.map(worker,range(users)) for r in group]
    elapsed=time.monotonic()-started
    stop.set()
    sampler.join(timeout=12)
    report={'memory_samples':samples,'users':users,'rounds':rounds,'requests':len(results),'successful':sum(r['ok'] for r in results),
            'wall_seconds':round(elapsed,2),'cacheless':True,'results':results,
            'note':'Acceptance samples, not a capacity/uptime guarantee. 429 is a real rate-limit outcome; do not bypass it.'}
    def stats(values):
        values=sorted(values)
        return {'samples':len(values),'mean_ms':round(sum(values)/len(values),2) if values else None,
                'p95_ms':values[max(0,(95*len(values)+99)//100-1)] if values else None}
    report['successful_duration']=stats([r['elapsed_ms'] for r in results if r['ok']])
    report['first_answer_text']=stats([r['first_answer_ms'] for r in results if r['ok'] and r.get('first_answer_ms') is not None])
    model_rss=[sample['model_memory']['process_rss_mb'] for sample in samples if sample.get('model_memory')]
    report['observed_peak_model_rss_mb']=max(model_rss) if model_rss else None
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return {**{k:v for k,v in report.items() if k not in {'results','memory_samples'}}, 'saved_report':str(output)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('check')
    sub.add_parser('freeze')
    p=sub.add_parser('probe');p.add_argument('--base-url',default='http://127.0.0.1:8000')
    p=sub.add_parser('backup');p.add_argument('--destination',type=Path,default=ROOT/'backups');p.add_argument('--keep',type=int,default=5)
    p=sub.add_parser('restore-check');p.add_argument('archive',type=Path);p.add_argument('--target',required=True,type=Path)
    p=sub.add_parser('load-test');p.add_argument('--base-url',default='http://127.0.0.1:8000');p.add_argument('--prompt',required=True);p.add_argument('--users',type=int,default=2);p.add_argument('--rounds',type=int,default=2);p.add_argument('--output',type=Path,default=ROOT/'qa_exports/deployment_load.json')
    args=parser.parse_args()
    if args.command=='check':
        result=validate(ROOT,True); print(json.dumps(result,indent=2));return 0 if result['ok'] else 1
    elif args.command=='freeze':
        import run_services
        target=ROOT/'deployment/runtime_dependencies.local.json'
        target.write_text(json.dumps(run_services.runtime_manifest(),indent=2),encoding='utf-8')
        print(target);return 0
    elif args.command=='probe': result=probe(args.base_url)
    elif args.command=='backup': result={'backup':str(backup(ROOT,args.destination,args.keep))}
    elif args.command=='restore-check': result=restore_check(args.archive,args.target)
    else: result=load_test(args.base_url,args.prompt,args.users,args.rounds,args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr)
        raise SystemExit(1)
