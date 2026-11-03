"""Derive Linux serving dependencies from the existing requirements file."""
from pathlib import Path
import sys


def serving_requirements(text):
    result = []
    for line in text.splitlines():
        value = line.strip()
        if not value or value.startswith(('#', '--')):
            continue
        if value.lower().startswith(('llama-cpp-python', 'onnxruntime-directml')):
            continue
        result.append(value)
    # The container must have an explicit CPU EP even for older native requirements.
    if not any(line.startswith('onnxruntime') for line in result):
        result.append('onnxruntime>=1.20,<2')
    return '\n'.join(result) + '\n'


if __name__ == '__main__':
    Path(sys.argv[2]).write_text(serving_requirements(Path(sys.argv[1]).read_text()), encoding='utf-8')
