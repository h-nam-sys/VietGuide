# Setup

## Runtime assets

The source archive does not contain model weights, private datasets or secrets.
The standard layout is:

- `models/qwen3.5_4b.gguf`: Qwen3.5-4B IQ4_XS from
  [bartowski's GGUF repository](https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF).
- `models/rag_directml/`: BGE ONNX exports, tokenizer files and `current.json`
  pointing to a generation with `validated.json`.
- `data/`: domain profiles, SQL records, vector indexes and runtime settings.
- `backend/static/forms/`: configured form files, when used.

Prepared retrieval assets must match the ONNX Runtime version recorded in their
validation manifest. Hardware/device compatibility must be checked on a new
machine. Export and formal validation tools belong to the full source environment.
They are not included in the smaller hybrid runtime requirements.

## Hybrid installation

Install Python 3.12 with the Windows Python launcher and Docker Desktop using
Linux containers. Place the runtime assets in the project, create `.env` from
`.env.example`, and set the admin password. Then run:

```powershell
.\start_all_services.bat --docker
```

For a missing `env`, the BAT invokes the first-run inference installer. It pins
DirectML's ONNX Runtime version to the prepared manifest, installs
`requirements-inference.txt` and checks dependency consistency. An interrupted
installation is retried on the next launch. Existing working environments are
not reinstalled or upgraded.

The Windows environment contains the model HTTP service, tokenizers, ONNX runtime
and compatibility dependencies. FlagEmbedding is retained for the existing
reranker tokenizer compatibility path; its dependencies include PyTorch. Streamlit,
LlamaIndex and backend database packages are supplied by the Docker image instead.

The launcher prepares/builds the base image if missing, applies the web source
layer, starts native inference and checks it from a temporary container before
starting backend/admin. No separate inference command is required.

## Full source installation

In Command Prompt:

```bat
py -3.12 -m venv env
env\Scripts\activate.bat
python -m pip install -r requirements.txt
```

For a fresh model setup, export the BGE models:

```bat
python -m rag.prepare_gpu --export
```

In that environment, start the model service directly:

```bat
python llm_service/main.py
```

Leave it running without sending chat/embedding requests. In another activated
terminal at the project root, validate the pending exports:

```bat
python -m rag.prepare_gpu --validate
```

Validation requires Qwen to be loaded, checks outputs and activates the completed
export generation. `--device-id` selects the DirectML adapter. An optional
`--precision fp16` can be supplied during export; changing export precision needs
new validation. Stop the temporary model service, then start normally:

```bat
start_all_services.bat
```

For an installation without ingested data, open admin at localhost:8501 and import
the first dataset. `/readyz` and automatic chat opening require an available dataset.
Gemini proposals require `GEMINI_API_KEY`; a reviewed JSON profile can be imported
instead. Dataset updates replace the selected domain's active data after a
successful build. The previous generation remains available for rollback.

## Environment and storage

Admin edits apply through saved domain settings. `.env` changes require restarting
services. Per-domain and per-mode values can override environment defaults.
`RAG_LOW_SCORE_RECOVERY=0` is the selected release setting; omitting it enables
the code's fallback default.

Native conversations use `backend/legal_llm.db`. Hybrid mode imports native
runtime data once into Docker volumes. Later edits are not synchronized between
the two stores. Preserve the session signing key with its conversation database.
Never distribute private histories, `.env` files or signing keys as demo assets.

## Troubleshooting

- Missing Python launcher: install Python 3.12 or create the project environment
  explicitly before starting hybrid mode.
- Model assets missing: add the prepared files; installing Python packages alone
  does not download or validate those exports.
- Docker cannot reach inference: check the private Docker connection in Windows
  Firewall. Hybrid inference binds `0.0.0.0`; keep model port 8001 private.
- Occupied port: stop the other launcher/container. The launcher does not kill
  unrelated processes or load a duplicate model on an occupied port.
- Slow answers: inspect initialization logs/status for CPU fallback before
  changing retrieval parameters.
