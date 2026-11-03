# Docker Hub hybrid deployment

Docker runs backend/admin. Windows runs generation, embedding and reranking.

## Users

Install Python 3.12, Docker Desktop (Linux containers) and GPU drivers. Download
the project source, prepared model/data assets and create `.env` as shown in the
[root quick start](../../README.md#quick-start). Then run:

```powershell
.\start_all_services.bat --docker
```

The launcher pulls `namnhedu/legal_llm-chat-service:hybrid-v1` when it is absent, installs
missing Windows inference dependencies and starts both sides. Existing Python
environments are reused. The source overlay build reuses downloaded dependencies.
No model weights are included in the image.

The repository must be public for anonymous pulls. For a private repository, run
`docker login` first with an account that has access. A `manifest unknown` error
means the maintainer has not published this tag yet. Tags from an older
three-container architecture are not interchangeable with this hybrid runtime.

## Publish from the working development machine

Run from the project root. The generated build configuration uses the working
native runtime snapshot and prepared model manifest:

```powershell
.\env\Scripts\python.exe deployment\container\configure.py
docker compose -f compose.yaml build app
docker login
docker tag legal-llm:docker-v1 namnhedu/legal_llm-chat-service:hybrid-v1
docker push namnhedu/legal_llm-chat-service:hybrid-v1
```

The runtime snapshot is produced by the existing service launcher. If it is
missing, launch the working services once and stop them before preparing the build.
If the current `legal-llm:docker-v1` image is already built from this version,
only login, tag and push are needed.

Only one image is published for this layout. Do not publish separate LLM/RAG
images for the Windows host inference process. The `hybrid-v1` tag distinguishes
this layout from older `latest` images. Docker Hub page:
https://hub.docker.com/r/namnhedu/legal_llm-chat-service

For an intentional runtime update, stop services and refresh the cached image:

```powershell
docker pull namnhedu/legal_llm-chat-service:hybrid-v1
.\start_all_services.bat --docker
```

Publishing a new tag also requires updating the default image in
`Dockerfile.hybrid` and `deployment/container/hybrid.py` together. Docker's
[publish reference](https://docs.docker.com/reference/cli/docker/image/push/)
covers tag and push usage.

## Model/data distribution and storage

Google Drive can host a separate ready-to-extract bundle with
`models/rag_directml/` (including tokenizers, `current.json` and `validated.json`),
clean demo `data/` and optional `backend/static/forms/`. Keep their project-relative
paths. Qwen's GGUF has a direct download link in the README. Prepared retrieval
assets must satisfy runtime-version/device validation on the target machine.
Keep upstream model licenses with redistributed weights.

Exclude private histories, feedback, credentials, signing keys, `.env`, `env/`,
logs and backups from public demo assets. Docker imports demo data once into
named volumes; later native and Docker edits are independent. Ctrl+C retains
volumes; `docker compose down -v` deletes Docker's saved state.

The all-in-container CPU configuration remains separate in `compose.yaml`.
Publishing and launch commands here preserve the Windows-GPU hybrid setup.
