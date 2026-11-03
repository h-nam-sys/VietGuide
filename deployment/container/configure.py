"""Generate non-secret Linux build pins from the current validated native runtime."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def prepare(root=ROOT):
    root = Path(root)
    gpu = root / 'models/rag_directml'
    generation = json.loads((gpu/'current.json').read_text(encoding='utf-8-sig'))['generation']
    if not isinstance(generation, str) or not generation.isalnum():
        raise ValueError('Invalid retrieval model generation')
    metadata = json.loads((gpu/generation/'validated.json').read_text(encoding='utf-8-sig'))
    version = metadata['onnxruntime_version']
    if not isinstance(version, str) or not re.fullmatch(r'\d+\.\d+\.\d+', version):
        raise ValueError('Expected a released ONNX Runtime version in validated.json')
    for path in (root/'models/qwen3.5_4b.gguf', gpu/generation/'embedding/model.onnx',
                 gpu/generation/'reranker/model.onnx', root/'data/domain_profile.json'):
        if not path.is_file():
            raise ValueError('Required runtime asset is missing: ' + str(path.relative_to(root)))
    manifest = root/'logs/operations/runtime_dependencies.json'
    if not manifest.is_file():
        raise ValueError('Native runtime snapshot is missing. Start the working native services once, stop them, then rerun this command.')
    packages = json.loads(manifest.read_text(encoding='utf-8'))['packages']
    packages = {name.lower().replace('_','-'): value for name, value in packages.items()}
    names = ('fastapi','uvicorn','httpx','pydantic','numpy','transformers','tokenizers',
             'sentence-transformers','flagembedding','llama-index-core',
             'llama-index-embeddings-huggingface','sqlalchemy','streamlit','torch')
    constraints = ['onnxruntime==' + version]
    for name in names:
        installed = packages.get(name)
        if installed and re.fullmatch(r'\d+(?:\.\d+)+(?:[a-zA-Z0-9.+-]*)', installed):
            # Linux CPU PyTorch wheels have their own local-version suffix.
            installed = installed.split('+')[0] if name == 'torch' else installed
            constraints.append(name + '==' + installed)
    llama = packages.get('llama-cpp-python', '')
    if not re.fullmatch(r'\d+\.\d+\.\d+', llama):
        raise ValueError('Expected a released llama-cpp-python version')
    folder = root/'deployment/container'
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'constraints.generated.txt').write_text('\n'.join(constraints)+'\n', encoding='utf-8')
    (folder/'llama.generated.txt').write_text('llama-cpp-python=='+llama+'\n', encoding='utf-8')
    return {'onnxruntime': version, 'llama_cpp_python': llama,
            'native_package_versions_found': bool(packages)}


if __name__ == '__main__':
    print(json.dumps(prepare(), indent=2))
    print('Docker build configuration prepared. No native dependencies or data were modified.')
