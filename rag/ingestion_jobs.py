"""Durable subprocess jobs. One ingestion/rollback writer per domain."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

from .domain_config import ROOT_DIR, RAGConfigurationError, atomic_json, read_json, validate_profile, domain_status


from rag.timing import timed, span, emit, TimedLock, trace_headers, TimingMiddleware, trace_id

def now():
    return datetime.now(timezone.utc).isoformat()


def lock_path(domain, root):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', domain):
        raise RAGConfigurationError('Invalid domain ID')
    return Path(root) / 'data/admin_jobs/locks' / f'{domain}.lock'


@timed('rag.ingestion_jobs.claim')
def claim(domain, owner, root):
    path = lock_path(domain, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('x', encoding='utf-8') as f:
            json.dump({'owner': owner, 'pid': os.getpid(), 'created_at': now()}, f)
    except FileExistsError as exc:
        raise RAGConfigurationError('Another ingestion or rollback is active for this domain. Refresh job status first.') from exc
    # Check after taking the lock so deletion cannot race a new writer.
    try:
        if domain_status(domain, root=root) == 'deleted':
            raise RAGConfigurationError('Lĩnh vực đã bị xóa. Dùng mã mới khi nhập lại dữ liệu.')
    except Exception:
        release(path, owner)
        raise
    return path


@timed('rag.ingestion_jobs.release')
def release(path, owner):
    if path.exists() and read_json(path).get('owner') == owner:
        path.unlink()


@contextmanager
def domain_lock(domain, *, root=None):
    root = Path(root or ROOT_DIR)
    owner = uuid.uuid4().hex
    path = claim(domain, owner, root)
    try:
        yield
    finally:
        release(path, owner)


@timed('rag.ingestion_jobs.start_job')
def start_job(input_path, profile, *, sheet=0, root=None):
    root = Path(root or ROOT_DIR).resolve()
    input_path = Path(input_path).resolve()
    profile = validate_profile(profile)
    if domain_status(profile['domain_name'], root=root) == 'deleted':
        raise RAGConfigurationError('Lĩnh vực đã bị xóa. Dùng mã mới khi nhập lại dữ liệu.')
    from .ingest_pipeline import prepare_records, read_input
    _, report = prepare_records(read_input(input_path, sheet), profile)
    job_id = uuid.uuid4().hex
    path = root / 'data/admin_jobs' / f'{job_id}.json'
    lock = claim(profile['domain_name'], job_id, root)
    job = {'id': job_id, 'domain': profile['domain_name'], 'input_path': str(input_path),
           'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
           'profile': profile, 'sheet': sheet, 'status': 'queued', 'stage': 'Waiting for worker',
           'created_at': now(), 'updated_at': now(), 'report': report, 'pid': None}
    atomic_json(path, job)
    log_path = path.with_suffix('.log')
    try:
        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1', 'LEGAL_TIMING_TRACE_ID': trace_id(), 'LEGAL_INGESTION': '1'}
        with log_path.open('w', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, '-u', '-m', 'rag.ingestion_jobs', '--job-file', str(path)],
                cwd=ROOT_DIR, stdout=log, stderr=subprocess.STDOUT, env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        # Worker owns subsequent status writes; keep PID separately to avoid races.
        atomic_json(path.with_suffix('.pid'), {'pid': process.pid})
    except Exception:
        job.update(status='failed', stage='Could not start worker', updated_at=now())
        atomic_json(path, job)
        release(lock, job_id)
        raise
    return job_id


@timed('rag.ingestion_jobs.list_jobs')
def list_jobs(*, root=None):
    root = Path(root or ROOT_DIR)
    result = []
    for path in (root / 'data/admin_jobs').glob('*.json'):
        job = read_json(path)
        pid_path = path.with_suffix('.pid')
        if job['status'] in {'queued', 'running'} and pid_path.exists():
            import psutil
            pid = read_json(pid_path)['pid']
            if not psutil.pid_exists(pid):
                job.update(status='failed', stage='Worker stopped before completion', updated_at=now(),
                           error='The worker exited unexpectedly. Active data was not intentionally changed; inspect its log.')
                atomic_json(path, job)
                release(lock_path(job['domain'], root), job['id'])
        result.append(job)
    return sorted(result, key=lambda j: j['created_at'], reverse=True)


@timed('rag.ingestion_jobs.run_job')
def _run_job(path, *, vector_builder=None):
    from .domain_onboarding import config_from_profile, save_profile_registry
    from .ingest_pipeline import read_input, prepare_records, publish_generation, build_vectors
    path = Path(path)
    root = path.parents[2]
    job = read_json(path)
    lock = lock_path(job['domain'], root)
    if not lock.exists() or read_json(lock).get('owner') != job['id']:
        raise RAGConfigurationError('Job does not hold its domain writer lock')
    def progress(stage):
        job.update(status='running', stage=stage, updated_at=now(), pid=os.getpid())
        atomic_json(path, job)
        print(stage, flush=True)
    try:
        progress('Reading and validating uploaded file')
        input_path = Path(job['input_path'])
        if hashlib.sha256(input_path.read_bytes()).hexdigest() != job['input_sha256']:
            raise RAGConfigurationError('Uploaded file changed after validation')
        config = config_from_profile(job['profile'], root=root)
        records, report = prepare_records(read_input(input_path, job['sheet']), config.profile)
        generation = publish_generation(config, records, report, vector_builder=vector_builder or build_vectors, progress=progress)
        save_profile_registry(config.profile, root=root)
        job.update(status='succeeded', stage='Domain is ready for chat', generation=generation,
                   report=report, updated_at=now())
        atomic_json(path, job)
    except Exception as exc:
        # Don't persist credentials or HTTP URLs from third-party exception strings.
        job.update(status='failed', stage='Ingestion failed', error=(str(exc) if isinstance(exc, (RAGConfigurationError, ValueError))
                    else f'{type(exc).__name__}: see worker log'), updated_at=now())
        atomic_json(path, job)
        import traceback
        traceback.print_exc()
        return False
    finally:
        release(lock, job['id'])
    return True


def run_job(path, *, vector_builder=None):
    from .process_lock import process_lock
    path = Path(path)
    root = path.parents[2]
    job = read_json(path)
    job.update(stage='Waiting for ingestion slot', updated_at=now())
    atomic_json(path, job)
    # Only one dataset builds embeddings at a time, across all worker processes.
    try:
        with process_lock(root / 'data/ops_locks/ingestion.lock', timeout=86400):
            return _run_job(path, vector_builder=vector_builder)
    except Exception as exc:
        job.update(status='failed', stage='Ingestion failed', error=type(exc).__name__ + ': see worker log', updated_at=now())
        atomic_json(path, job)
        release(lock_path(job['domain'], root), job['id'])
        return False


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / 'backend/.env')
    load_dotenv(ROOT_DIR / '.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-file', type=Path, required=True)
    args = parser.parse_args()
    os.environ['LEGAL_INGESTION'] = '1'
    sys.exit(0 if run_job(args.job_file) else 1)
