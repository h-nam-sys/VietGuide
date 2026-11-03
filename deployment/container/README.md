# Local Docker deployment

Supports Windows inference with containerized backend/admin, plus an optional
fully containerized CPU mode. Both use the existing service supervisor. The
normal native launcher remains available.

## Windows GPU + Docker backend/admin (recommended for this project machine)

The RX 5500 XT / Windows 10 setup runs inference natively using the confirmed
Vulkan and DirectML code. Docker runs backend/admin only. This is a hybrid
deployment, not GPU inference inside a Linux container.

One-time prerequisites:

- Docker Desktop with Linux containers installed; sign-in is not required for
  local images. The launcher can open the standard Windows Docker Desktop
  installation if its engine is stopped. Per-user/custom installations may need
  Docker Desktop opened manually.
- The existing project Python environment at `env/Scripts/python.exe`, with the
  working inference dependencies. Docker does not install native GPU drivers or
  Windows Python packages. A new machine must prepare/validate its BGE exports
  for its own runtime/device; do not assume another machine's validation applies.
- The current `.env` (including `ADMIN_PASSWORD`), model files and ingested data.
  Use the default `models/qwen3.5_4b.gguf`, `models/rag_directml`, and native
  `backend/legal_llm.db` layout for the automatic first import.

Stop the current native launcher/old Docker app before the first migration.
From the project folder run:

```powershell
.\start_all_services.bat --docker
```

The launcher records the actual native dependency versions, generates build
pins, and builds the base image only if `legal-llm:docker-v1` is missing. Existing
users reuse that built image. `Dockerfile.hybrid` only copies updated source on
it, avoiding another Vulkan compilation. If Python requirements change later,
rebuild the base deliberately before rebuilding the hybrid image.

It starts/supervises native inference first, waits for model/assets readiness,
checks connectivity from a temporary container, then starts Docker backend/admin.
It reuses a compatible existing inference service when reachable and never loads
another model on the occupied port. An occupied unrelated/unready port fails
with an explanation. An existing old all-in-one app container is stopped before
native inference starts; its saved volumes are reused.

Chat: http://localhost:8000/chat. Admin: http://localhost:8501.
Keep the launcher terminal open. Ctrl+C stops the native inference and container
that this invocation started. Independently running components that were reused
are left running. Docker Desktop itself is left running.

The native inference child binds `0.0.0.0` in hybrid mode so Docker Desktop can
reach it using `host.docker.internal:8001`. No `.env` binding is changed. Allow
only the needed private Docker connection in Windows Firewall; do not disable
it or expose the unauthenticated model service publicly. A previously running
loopback-only inference service may need stopping so the launcher can start it
with the required binding.

The plain `start_all_services.bat` retains the normal native startup mode.

Vulkan and DirectML can use compatible AMD or NVIDIA hardware and drivers. The
locally verified GPU is the RX 5500 XT; NVIDIA hardware has not been validated
in this release. Initialization tries the existing GPU settings first and CPU
only after an initialization failure. It does not replay interrupted answers
or distribute inference automatically across multiple GPUs.

## Fully containerized CPU option

The base configuration still includes all services. Stop hybrid/native services,
then run these from the project root if choosing the slower CPU deployment:

```powershell
python deployment/container/configure.py
docker compose config --quiet
docker compose up -d --build
docker compose logs -f app
```

First build downloads dependencies and compiles llama.cpp. First launch may
download the configured chat tokenizer. The configure command writes two ignored
build dependency files and does not install anything in Windows.

## Data and models

On first start, existing datasets, forms, the session signing key and the default
`backend/legal_llm.db` are copied into named Docker volumes. SQLite copies use
the backup API. Native folders are mounted read-only. Later starts reuse the
Docker snapshot, including edits made through Docker's admin interface; changes
to native data after the first import are not synchronized automatically.

Models stay in `./models`, mounted read-only. This includes the GGUF and the
validated `rag_directml` ONNX exports. Model files, private datasets, `.env`,
history and logs are excluded from the image. Recipients need the image plus
the required model/data assets; the image alone does not contain those assets.

Named volumes hold datasets, chat/feedback data, forms, logs, tokenizer cache
and backups. `docker compose down` preserves them. **Do not add `--volumes`
or `-v` unless intentionally deleting Docker's saved state.**

Custom native database locations and custom model-directory layouts require
adjusting the seed/build configuration before the first import.

## Inference behavior

The confirmed native initializer is unchanged. In hybrid mode, the returned
status belongs to the Windows inference process. For the fully containerized
CPU option, the Linux image includes Vulkan;
the GPU must still be exposed by the host. On Windows/AMD, this configuration
does not establish GPU passthrough; expect CPU inference unless actual GPU use
is verified. ONNX retrieval uses the CPU provider in this Linux image. No
DirectML-to-Linux conversion or unverified CUDA provider is assumed. Docker gives
model HTTP calls up to 600 seconds; native 30/300-second defaults are unchanged. CPU work
uses system RAM and may be significantly slower than the native AMD service.

Check initialization paths with:

```powershell
docker compose -f compose.yaml -f compose.hybrid.yaml exec app python -c "import json,urllib.request; print(json.dumps(json.load(urllib.request.urlopen('http://host.docker.internal:8001/v1/rag/status')),indent=2))"
```

The `original_gpu_settings` path records the attempted settings, not measured
GPU utilization. Inspect model startup logs for actual offload. This local test
publishes backend/admin only on localhost; model port 8001 is not published by Docker.

## Docker inspection and manual stop

```powershell
docker compose -f compose.yaml -f compose.hybrid.yaml logs --tail 100 app
docker compose -f compose.yaml -f compose.hybrid.yaml ps
docker compose -f compose.yaml -f compose.hybrid.yaml stop app
```

Stop Docker before starting native services, because both use host ports 8000
and 8501. Public HTTPS and Linux GPU exposure are separate deployment steps.
