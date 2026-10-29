"""Run the existing three services in one terminal; retain full output on disk."""
import os
import argparse
import json
import importlib.metadata
import platform
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from dotenv import load_dotenv
from rag.service_config import llm_port

ROOT = Path(__file__).resolve().parent
STOP = threading.Event()
OUTPUT_LOCK = threading.Lock()


def show(service, text):
    with OUTPUT_LOCK:
        print(f'{time.strftime("%H:%M:%S")} [{service}] {text}', flush=True)


def capture(name, process, path):
    handler = RotatingFileHandler(path, maxBytes=10_000_000, backupCount=3, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(message)s'))
    try:
        in_traceback = False
        for line in process.stdout:
            handler.emit(logging.LogRecord(name, logging.INFO, '', 0, line.rstrip('\r\n'), (), None))
            if line.lstrip().startswith('Traceback (most recent call last):'):
                in_traceback = True
                continue
            if in_traceback and '[activity]' in line:
                in_traceback = False
            if in_traceback:
                if re.match(r'^(?:[\w.]*(?:Error|Exception|ExceptionGroup)|KeyboardInterrupt|SystemExit)(?::|$)', line):
                    show(name, line.strip() + f' (full traceback: {path})')
                    in_traceback = False
                continue
            if '[activity]' in line:
                show(name, line.split('[activity]', 1)[1].strip())
            elif re.search(r'\b(ERROR|WARNING|CRITICAL|Traceback)\b|\w+(?:Error|Exception):', line) or re.search(r'"level"\s*:\s*"(?:warn|error|fatal)"', line):
                show(name, line.strip())
    except Exception as exc:
        show(name, f'ERROR reading service output: {exc}')
        STOP.set()
    finally:
        handler.close()


def watch_ready(name, process, url):
    while not STOP.is_set() and process.poll() is None:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    show(name, 'Ready: ' + url)
                    return
        except Exception:
            pass
        STOP.wait(2)


def stop_process(process):
    if process.poll() is not None:
        return
    if os.name == 'nt':
        # Includes uvicorn reload children; never targets unrelated Python jobs.
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        # The process-group leader may have exited before its children. taskkill
        # is best-effort and uses this launcher's original PID only.
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def service_definitions(production=None):
    if production is None:
        production = os.getenv('APP_ENV', 'development') == 'production'
    inference_port=llm_port()
    backend = [sys.executable, '-u', '-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', '8000']
    backend += ['--proxy-headers', '--forwarded-allow-ips', '127.0.0.1,::1'] if production else ['--reload']
    services = [
        ('LLM', ROOT, [sys.executable, '-u', 'llm_service/main.py'], inference_port, '/v1/rag/status'),
        ('Backend', ROOT / 'backend', backend, 8000, '/health'),
        ('Admin', ROOT / 'backend', [sys.executable, '-u', '-m', 'streamlit', 'run', 'admin_dashboard.py', '--server.address', '127.0.0.1', '--server.port', '8501'], 8501, '/_stcore/health'),
    ]
    if production and os.getenv('CADDY_BINARY'):
        services.append(('Proxy',ROOT,[os.environ['CADDY_BINARY'],'run','--config',str(ROOT/'deployment/Caddyfile'),'--adapter','caddyfile'],443,'http://127.0.0.1:2019/config/'))
    return services


def healthy(port, path='/health'):
    try:
        with urllib.request.urlopen(path if path.startswith('http') else f'http://127.0.0.1:{port}{path}', timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def runtime_manifest():
    return {'python': sys.version, 'platform': platform.platform(), 'packages':
            {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}}


def prune_logs(days=30):
    cutoff = time.time() - days * 86400
    for folder in ('services', 'timing', 'operations'):
        for path in (ROOT/'logs'/folder).rglob('*.log*'):
            if path.is_file() and not path.is_symlink() and path.stat().st_mtime < cutoff:
                try:
                    path.unlink()
                except OSError:
                    pass


def main(argv=None):
    load_dotenv(ROOT / 'backend/.env')
    load_dotenv(ROOT / '.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--production', action='store_true')
    parser.add_argument('--development', action='store_true')
    args = parser.parse_args(argv)
    production = not args.development and (args.production or os.getenv('APP_ENV') == 'production')
    from rag.process_lock import process_lock
    from rag.operations import alert
    from rag.deployment_config import validate
    STOP.clear()
    if production:
        report = validate(ROOT, True)
        for warning in report['warnings']:
            show('Config', 'WARNING: ' + warning)
        if not report['ok']:
            for error in report['errors']:
                show('Config', 'ERROR: ' + error)
            return 1
    try:
        with process_lock(ROOT/'data/ops_locks/services.lock'):
            return run_group(production, alert)
    except TimeoutError:
        show('Services', 'ERROR: another launcher or maintenance operation is active.')
        return 1
    except (ValueError, OSError) as exc:
        show('Services', 'ERROR: ' + str(exc))
        return 1


def run_group(production, alert):
    services = service_definitions(production)
    for name, _, _, port, _ in services:
        with socket.socket() as probe:
            probe.settimeout(0.5)
            if probe.connect_ex(('127.0.0.1', port)) == 0:
                show(name, f'ERROR: port {port} is already occupied. Stop the old services first.')
                return 1
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    directory = ROOT / 'logs/services' / stamp
    directory.mkdir(parents=True)
    if production:
        prune_logs()
        manifest = ROOT/'logs/operations/runtime_dependencies.json'
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(runtime_manifest(), indent=2), encoding='utf-8')
    show('Services', ('Production' if production else 'Development') + ' mode. Ctrl+C stops all services.')
    children, readers = {}, []
    attempts = {name: 0 for name, *_ in services}
    started, failures = {}, {}
    max_restarts = int(os.getenv('SERVICE_RESTART_LIMIT','3'))
    grace = float(os.getenv('SERVICE_STARTUP_GRACE','300'))
    if max_restarts < 0 or not 10 <= grace <= 1800:
        raise ValueError('Invalid service restart/startup settings')

    def launch(definition):
        name, cwd, command, port, health = definition
        environment = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        process = subprocess.Popen(command, cwd=cwd, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            text=True, encoding='utf-8', errors='replace', creationflags=flags, start_new_session=os.name != 'nt')
        children[name] = process
        started[name], failures[name] = time.monotonic(), 0
        reader = threading.Thread(target=capture, args=(name, process, directory/(name.lower()+'.log')), daemon=True)
        reader.start()
        readers.append(reader)
        threading.Thread(target=watch_ready, args=(name,process,health if health.startswith('http') else f'http://127.0.0.1:{port}{health}'),daemon=True).start()
        show(name, 'Starting...')

    result, last_probe = 0, 0
    try:
        for definition in services:
            launch(definition)
        while not STOP.wait(.5):
            now = time.monotonic()
            probe_now = production and now-last_probe >= 10
            if probe_now:
                last_probe = now
            for definition in services:
                name, _, _, port, health = definition
                process = children[name]
                if probe_now and now-started[name] > grace:
                    failures[name] = 0 if healthy(port, health) else failures[name]+1
                    if failures[name] >= 3:
                        alert('service_unresponsive', name)
                        show(name, 'ERROR: health checks failed; stopping before restart.')
                        stop_process(process)
                if process.poll() is None:
                    if now-started[name] > 600:
                        attempts[name] = 0
                    continue
                alert('service_exited', name)
                if not production or attempts[name] >= max_restarts:
                    show(name, 'ERROR: service exited; restart limit reached or development mode. See ' + str(directory))
                    result = process.returncode or 1
                    STOP.set()
                    break
                # Stop the old process group fully before allocating replacement GPU models.
                stop_process(process)
                attempts[name] += 1
                delay = min(30, 5 * 2**(attempts[name]-1))
                show(name, f'Restarting in {delay}s (attempt {attempts[name]}/{max_restarts}).')
                if STOP.wait(delay):
                    break
                launch(definition)
    except KeyboardInterrupt:
        show('Services', 'Stopping...')
    except Exception as exc:
        show('Services', 'ERROR: ' + str(exc))
        result = 1
    finally:
        STOP.set()
        stoppers = [threading.Thread(target=stop_process,args=(p,)) for p in children.values()]
        for thread in stoppers:
            thread.start()
        for thread in stoppers:
            thread.join()
        for thread in readers:
            thread.join(timeout=2)
        show('Services','Stopped.')
    return result


if __name__ == '__main__':
    raise SystemExit(main())
