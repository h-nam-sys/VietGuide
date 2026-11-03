"""First-run Windows inference install; never upgrade an existing working env."""
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def ensure(root=ROOT):
    root = Path(root)
    environment = root/'env'
    python = environment/'Scripts/python.exe'
    pending = environment/'.hybrid_install_pending'
    if python.is_file() and not pending.exists():
        return False
    # Match the validated asset's ORT version instead of silently upgrading it.
    models = root/'models/rag_directml'
    generation = json.loads((models/'current.json').read_text(encoding='utf-8-sig'))['generation']
    if not isinstance(generation, str) or not generation.isalnum():
        raise ValueError('Invalid retrieval model generation')
    metadata = json.loads((models/generation/'validated.json').read_text(encoding='utf-8-sig'))
    version = metadata['onnxruntime_version']
    if not isinstance(version, str) or not re.fullmatch(r'\d+\.\d+\.\d+', version):
        raise ValueError('Expected a released ONNX Runtime version in validated.json')
    environment.mkdir(parents=True, exist_ok=True)
    pending.write_text('Install incomplete; rerun --docker to resume.', encoding='utf-8')
    if not python.is_file():
        subprocess.run([sys.executable,'-m','venv',str(environment)],check=True)
    with tempfile.TemporaryDirectory(prefix='legal-inference-') as temporary:
        constraints = Path(temporary)/'constraints.txt'
        constraints.write_text('onnxruntime-directml=='+version+'\n', encoding='utf-8')
        print('Installing Windows inference dependencies. Backend/admin packages stay in Docker.',flush=True)
        subprocess.run([str(python),'-m','pip','install','-r',str(root/'requirements-inference.txt'),
                        '-c',str(constraints)],cwd=root,check=True)
        subprocess.run([str(python),'-m','pip','check'],cwd=root,check=True)
    pending.unlink()
    return True


if __name__ == '__main__':
    try:
        ensure()
    except Exception as exc:
        print('Inference setup failed: '+str(exc),file=sys.stderr)
        raise SystemExit(1)
