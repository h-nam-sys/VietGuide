# Docker / Windows hybrid mode

```powershell
.\start_all_services.bat --docker
```

Docker runs backend/admin. Windows runs the shared Qwen, embedding and reranking
service. Python 3.12, Docker Desktop, runtime assets and `.env` are prerequisites;
the first launch installs the inference-only environment when absent.

- `requirements.txt`: full source installation.
- `requirements-inference.txt`: Windows model runtime and launcher.
- `Dockerfile`: base application image; backend/admin dependencies install here.
- `Dockerfile.hybrid`: app source layer on the existing base image.
- `compose.hybrid.yaml`: routes model requests to `host.docker.internal:8001`.

The launcher reuses a compatible model service and stops only components it starts.
Ctrl+C retains data volumes. The plain BAT remains the full native launch mode.

The initial image build downloads packages and compiles llama.cpp. Subsequent
hybrid source updates reuse the base image. No image has been published to a
registry by this setup; a new machine currently builds locally.

The all-in-container CPU configuration remains available through `compose.yaml`.
It is separate from hybrid mode and does not give an RX 5500 XT GPU access through
Windows Docker Desktop. GPU-capable host drivers/runtime access are platform
requirements, not properties of the GGUF file.

Native seed folders are read-only. First startup copies data/forms/default history
into named Docker volumes; later native and Docker edits are independent. Models
are mounted separately and are not copied into the image. `docker compose down`
retains volumes. Adding `-v` deletes Docker's saved state.

Inspect hybrid services with:

```powershell
docker compose -f compose.yaml -f compose.hybrid.yaml ps
docker compose -f compose.yaml -f compose.hybrid.yaml logs --tail 100 app
```

See [setup](../../docs/SETUP.md) and [admin settings](../../docs/ADMIN.md).
