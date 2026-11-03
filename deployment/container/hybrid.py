"""Coordinate native inference and the existing Docker web image in one launcher."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request

from rag.service_config import llm_port


def status(url, timeout=3):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=timeout) as response:
            value = json.load(response)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def compatible_model(value):
    return bool(value and value.get('gpu_protocol') == 1 and value.get('qwen_ready')
                and value.get('retrieval_assets_ready'))


def port_open(port):
    with socket.socket() as connection:
        connection.settimeout(.5)
        return connection.connect_ex(('127.0.0.1', port)) == 0


class Docker:
    def __init__(self, root, port):
        self.root = Path(root)
        self.env = dict(os.environ, HYBRID_LLM_PORT=str(port))
        self.prefix = ['docker', 'compose', '-f', 'compose.yaml', '-f', 'compose.hybrid.yaml']

    def call(self, args, *, capture=True, timeout=120, check=True, compose=True):
        command = (self.prefix if compose else ['docker']) + args
        result = subprocess.run(command, cwd=self.root, env=self.env, text=True,
                                capture_output=capture, timeout=timeout)
        if check and result.returncode:
            # Do not echo rendered Compose config or inspect environment values.
            detail = (result.stderr or '').strip()[-1500:] if capture else 'See Docker output above.'
            raise RuntimeError('Docker command failed: ' + ' '.join(args[:2]) + '\n' + detail)
        return result or model_outcome.get('code', 0)

    def state(self):
        ids = self.call(['ps', '-aq', 'app']).stdout.strip().splitlines()
        if not ids:
            return None
        if len(ids) != 1:
            raise RuntimeError('Hybrid mode expects one app container, not multiple replicas.')
        template = '{"id":"{{.Id}}","running":{{.State.Running}},"command":{{json .Config.Cmd}},"state":{{json .State}}}'
        return json.loads(self.call(['inspect', '--format', template, ids[0]], compose=False).stdout)


def ensure_engine(docker, stop, show):
    if not shutil.which('docker'):
        raise RuntimeError('Docker CLI is missing. Install Docker Desktop, then reopen this terminal.')
    def available():
        result = docker.call(['info', '--format', '{{.OSType}}'], compose=False, check=False, timeout=10)
        if result.returncode == 0 and result.stdout.strip() != 'linux':
            raise RuntimeError('Switch Docker Desktop to Linux containers.')
        return result.returncode == 0
    if available():
        return
    desktop = Path(os.environ.get('ProgramFiles', 'C:/Program Files'))/'Docker/Docker/Docker Desktop.exe'
    if os.name != 'nt' or not desktop.is_file():
        raise RuntimeError('Start Docker Desktop and wait for its Linux engine, then retry.')
    show('Docker', 'Starting Docker Desktop; waiting for its Linux engine...')
    subprocess.Popen([str(desktop)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic()+180
    while not stop.wait(2):
        if available():
            return
        if time.monotonic() >= deadline:
            raise RuntimeError('Docker Desktop did not become ready. Check its window and retry.')
    raise RuntimeError('Startup cancelled.')


def run_hybrid(launcher, alert):
    port = llm_port()
    docker = Docker(launcher.ROOT, port)
    model_url = f'http://127.0.0.1:{port}/v1/rag/status'
    stop, show = launcher.STOP, launcher.show
    worker = None
    model_outcome = {}
    owned_container = None
    result = 0
    try:
        ensure_engine(docker, stop, show)
        docker.call(['version'])
        docker.call(['config', '--quiet'])
        before = docker.state()
        owned_ports = bool(before and before['running'])
        for web_port in (8000, 8501):
            if port_open(web_port) and not owned_ports:
                raise RuntimeError(f'Port {web_port} is occupied. Stop native backend/admin before hybrid launch.')
        # Stop the previous CPU container before loading native GPU models.
        if before and before['running'] and before.get('command') != ['--web-only']:
            show('Docker', 'Stopping the existing all-in-one container; saved volumes are preserved.')
            docker.call(['stop', 'app'], timeout=90)
            before = None
        model_busy = port_open(port)
        if model_busy and not compatible_model(status(model_url)):
            raise RuntimeError(f'Port {port} is occupied by an unready/incompatible service. Wait for it or stop it first; no duplicate model will be started.')
        # Record versions from the actual environment selected by the existing BAT.
        manifest = launcher.ROOT/'logs/operations/runtime_dependencies.json'
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(launcher.runtime_manifest(), indent=2), encoding='utf-8')
        from deployment.container.configure import prepare
        prepare(launcher.ROOT)
        base = docker.call(['image','inspect','--format','{{.Id}}','legal-llm:docker-v1'], compose=False, check=False)
        if base.returncode:
            show('Docker', 'First setup: building the base image. This downloads dependencies and compiles inference support once.')
            docker.call(['compose','-f','compose.yaml','build','app'], compose=False,
                        capture=False, timeout=3600)
        show('Docker', 'Building the web update on the existing image; no inference compilation.')
        docker.call(['build', 'app'], capture=False, timeout=600)
        if stop.is_set():
            return 0
        if not model_busy:
            definitions = [item for item in launcher.service_definitions(True) if item[0] == 'LLM']
            def serve_model():
                try:
                    model_outcome['code'] = launcher.run_group(True, alert, definitions=definitions,
                                       environment_overrides={'LLM_BIND_HOST':'0.0.0.0'})
                finally:
                    stop.set()
            worker = threading.Thread(target=serve_model, name='native-inference-supervisor')
            worker.start()
            deadline = time.monotonic()+float(os.getenv('SERVICE_STARTUP_GRACE', '300'))
            show('Hybrid', 'Waiting for native inference and retrieval assets...')
            while not stop.is_set() and not compatible_model(status(model_url)):
                if time.monotonic() >= deadline:
                    raise RuntimeError('Native inference did not become ready. Inspect logs/services before retrying.')
                stop.wait(1)
            if stop.is_set():
                raise RuntimeError('Native inference exited before becoming ready.')
        else:
            show('Hybrid', 'Reusing compatible native inference; it will not be stopped on exit.')
        # Prove the host GPU service is reachable from the actual container network.
        probe = ("import json,urllib.request; o=urllib.request.build_opener(urllib.request.ProxyHandler({})); v=json.load(o.open("
                 f"'http://host.docker.internal:{port}/v1/rag/status',timeout=10)); "
                 "assert v.get('gpu_protocol')==1 and v.get('qwen_ready') and v.get('retrieval_assets_ready')")
        try:
            docker.call(['run','--rm','--no-deps','--entrypoint','python','app','-c',probe], timeout=60)
        except RuntimeError as exc:
            raise RuntimeError('Docker cannot reach Windows inference. Check the Windows firewall for the private Docker connection. If reusing a loopback-only model service, stop it and let --docker launch it with the required binding.') from exc
        show('Docker', 'Starting backend/admin; inference stays on Windows.')
        # Mark a newly created/recreated app as owned, including partial startup failures.
        try:
            docker.call(['up','-d','--no-build','app'], capture=False, timeout=180)
        finally:
            current = docker.state()
            if current and (not before or not before['running'] or current['id'] != before['id']):
                owned_container = current['id']
        deadline = time.monotonic()+180
        while not stop.is_set():
            state = docker.state()
            if not state or not state['running']:
                raise RuntimeError('Docker app stopped. Run docker compose logs --tail 100 app.')
            if state['state'].get('Health', {}).get('Status') == 'healthy':
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('Docker backend/admin did not become ready. Inspect Docker app logs.')
            stop.wait(2)
        if stop.is_set():
            return 1
        show('Hybrid', 'Ready: http://localhost:8000/chat | admin http://localhost:8501')
        launcher.open_chat_browser()
        model_status = status(model_url) or {}
        fallback = [name for name, info in model_status.get('inference_startup', {}).items()
                    if info.get('path') == 'cpu_fallback']
        if fallback:
            show('Hybrid', 'WARNING: native CPU fallback is active for ' + ', '.join(fallback))
        show('Hybrid', 'Ctrl+C stops only the components started by this launcher; saved data remains.')
        while not stop.wait(5):
            state = docker.state()
            if not state or not state['running']:
                raise RuntimeError('Docker app stopped unexpectedly. Inspect its logs before restarting.')
    except KeyboardInterrupt:
        show('Hybrid', 'Stopping...')
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
        show('Hybrid', 'ERROR: ' + str(exc))
        result = 1
    finally:
        stop.set()
        try:
            if owned_container:
                # Stop the exact container we started, not an independently replaced instance.
                docker.call(['stop','--time','45',owned_container], compose=False, timeout=60)
        except Exception as exc:
            show('Docker', 'WARNING: could not stop the owned container: ' + str(exc))
        finally:
            if worker:
                worker.join(timeout=20)
                if worker.is_alive():
                    show('Hybrid', 'WARNING: native inference shutdown is still in progress.')
    return result or model_outcome.get('code', 0)
