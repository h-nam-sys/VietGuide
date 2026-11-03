# Legal LLM

Vietnamese question answering over structured datasets, with local model inference, source records and a web administration interface.

## Features

- Multiple datasets with automatic routing and contextual follow-up questions.
- Streamed answers with inspectable sources, browser-based history and feedback.
- Spreadsheet/JSON ingestion, configurable field mappings and dataset rollback.
- Procedure form downloads and optional permission-based online search.
- Admin controls for dataset scope, answer settings and service monitoring.

<!-- Screenshot: completed answer with its source record expanded. -->
<!-- Screenshot: domain management workspace in the admin interface. -->

## Quick start

Prerequisites: Windows, Python 3.12, compatible graphics drivers, the Qwen GGUF,
prepared BGE ONNX exports, and dataset assets. Runtime files are separate from
source control. Model preparation is covered in [Setup](docs/SETUP.md).

For a new installation, create `.env` from `.env.example` and set `ADMIN_PASSWORD`.
Existing `.env` files and working environments should be retained.

### Docker backend/admin + Windows inference

Install Docker Desktop with Linux containers, then run from PowerShell:

```powershell
.\start_all_services.bat --docker
```

On first use, the launcher creates `env` and installs only the Windows inference
requirements. Existing environments are reused without reinstalling packages.
Docker supplies the backend/admin dependencies. The launcher builds the image
when needed, starts inference, checks connectivity and opens chat when ready.
The first installation/build can take time; later starts reuse the environment
and image layers.

### Run entirely from source

Run in Windows Command Prompt:

```bat
py -3.12 -m venv env
env\Scripts\activate.bat
python -m pip install -r requirements.txt
start_all_services.bat
```

This installs the full native stack. `requirements-inference.txt` is for hybrid
mode only; it does not include the admin or model-export toolchain.

| Interface | Address |
|---|---|
| Chat | <http://localhost:8000/chat> |
| Admin | <http://localhost:8501> |
| API docs | <http://localhost:8000/docs> |

Keep the launcher terminal open. Ctrl+C stops components started by that launch.
Do not run native and Docker web services together on ports 8000/8501.

## Models and architecture

| Component | Implementation |
|---|---|
| Generation | Qwen3.5-4B IQ4_XS GGUF, llama-cpp-python/Vulkan, 4,096-token context |
| Embedding | BAAI/bge-m3, ONNX Runtime |
| Reranking | BAAI/bge-reranker-v2-m3, ONNX Runtime |
| Backend / admin | FastAPI / Streamlit |
| Storage | SQLite and persisted vector indexes |
| Optional services | Gemini profile proposals and MCP external search |

The backend selects a domain, resolves references, retrieves and reranks records,
then generates an answer from the retained public fields. Generation, embedding
and reranking share one inference process. Browser cookies identify chat owners;
clearing the cookie starts a separate history.

Hybrid mode keeps inference on Windows and runs backend/admin in Docker. The
local GPU setup has been verified with an RX 5500 XT. NVIDIA configurations
require compatible runtimes/drivers and have not been validated on project
hardware. CPU fallback is attempted after initialization failure; interrupted
inference is not automatically replayed.

## Configuration and documentation

- [Installation, runtime assets and troubleshooting](docs/SETUP.md)
- [Admin workflows and setting descriptions](docs/ADMIN.md)
- [Docker modes and persistent storage](deployment/container/README.md)
- [Backend API](backend/spec.md)

Domain switching can occasionally require clarification. Docker and native data
stores are separate after the initial Docker snapshot. See the setup guide
before transferring runtime assets between machines.

## Development

The repository contains the application source and regression tests. For example:

```bat
python -m unittest discover -s tests -p test_llm_domain_routing.py
```

Automated contract checks do not replace live checks of answers, sources and
GPU behavior. Development history is recorded in the existing repository;
research evaluation is separate from application engineering.

## Team

**Team:** `Team 10`

| Member | GitHub | Responsibilities |
|---|---|---|
| Nguyen Hai Nam | [@h-nam-sys](https://github.com/h-nam-sys) | System architecture, backend development, RAG integration and refinement, task coordination, experiment review and system debugging. |
| Nguyen Minh Thinh | [@minhthinhn](https://github.com/minhthinhn) | Development of the initial working RAG module; ongoing research and paper development. |
| Le Thien Khang | [@thienkhang1234](https://github.com/thienkhang1234) | Model selection, fine-tuning and quantization experiments, AI evaluation; ongoing research and paper development. |

Contributions and development history are available in the [commit history](https://github.com/h-nam-sys/Legal_LLM/commits) and [pull requests](https://github.com/h-nam-sys/Legal_LLM/pulls).

## License

Project code is distributed under the [MIT License](LICENSE). Model weights and dependencies retain their respective upstream licenses.
