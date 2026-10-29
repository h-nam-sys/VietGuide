# Deployment operations

This update keeps the current native Windows/Vulkan/DirectML model stack. No Redis, Qdrant or retrieval redesign.

## Use locally first

Keep your current `.env`, models, databases, data, forms and session key. Add just:

```dotenv
APP_ENV=production
```

Keep your existing `ADMIN_PASSWORD`. Leave `PUBLIC_HOST` unset and `SESSION_COOKIE_SECURE=auto` while using localhost HTTP.

Start with the usual `start_all_services.bat`. Alternatively, `start_all_services.bat --production` selects production mode for one run. `--development` retains development reload behavior. No new Python runtime dependencies are required.

Production removes reload, validates configuration/assets, prevents duplicate launchers, records installed Python package versions, restarts failed services with bounded backoff and monitors liveness. Three failed restarts stop the group with an error; it does not loop forever or silently switch models. Allow the default five-minute startup grace for cold model initialization.

Admin → Tổng quan now shows queue counts, recent timings, process memory, readiness and local alerts. Figures reset when a process restarts. RSS is not VRAM; first answer text is not an internal model token timestamp. `/health` means alive; `/readyz` checks database, an available dataset and model/validated retrieval assets without generating an answer. First-use model allocation can still fail, so run the live check below.

## Public HTTPS, when the hostname is ready

Install Caddy from its official distribution on the deployment machine. Add to the existing `.env`:

```dotenv
PUBLIC_HOST=chat.your-domain.example
CADDY_BINARY=C:/tools/caddy/caddy.exe
SESSION_COOKIE_SECURE=1
```

Replace the example hostname/path. DNS must point to this machine and ports 80/443 must reach Caddy. The existing launcher then supervises Caddy too. Its config is `deployment/Caddyfile`; chat/API/static stay on the same origin. SSE flushes immediately through Caddy; disconnect cancellation remains enabled. Admin routes/API docs are blocked at this proxy. Keep backend 8000, model 8001, admin 8501 and Caddy admin 2019 private. Use local access or a private tunnel for administration.

Do not set public HTTPS options just to test localhost. No public domain, certificate, firewall or router changes were applied by the update. The shipped config was validated and tested with temporary local HTTPS; real public certificate issuance still needs the target host.

## One operations command

Use the project environment's Python from the project root:

| Command | Purpose |
|---|---|
| `python deployment/manage.py check` | Check production settings, dependencies and model assets |
| `python deployment/manage.py freeze` | Save actual package versions to `deployment/runtime_dependencies.local.json` |
| `python deployment/manage.py probe` | Check liveness/readiness; use `--base-url https://your-host` for the public route |
| `python deployment/manage.py backup` | With services stopped, make a verified backup under `backups/`; keep the latest five |
| `python deployment/manage.py restore-check backups/FILE.zip --target C:/temp/legal-restore-check` | Restore and verify into a NEW folder, never over the running project |
| `python deployment/manage.py load-test --users 2 --rounds 2 --prompt "Đăng ký khai sinh cần những giấy tờ gì?"` | Small cacheless live run; saves full responses, durations, first-text timing and sampled RSS/queues |

Use a prompt supported by your dataset. Start at one user, then two. Real rate limits remain enabled: a 429 is recorded rather than bypassed. Reports go to `qa_exports/deployment_load.json`; they are measurements, not a promised supported capacity. Full GPU/VRAM usage still needs the deployment machine's GPU tools.

Backups include project-contained runtime data, conversation/dataset SQLite snapshots, forms, `.env` files and the session key. They contain private data: store them privately and copy to another disk for protection against disk loss. Models, Python environments, Git history and Caddy certificate storage are not included. Custom database/assets outside the project are rejected rather than silently omitted. Restore-check verifies hashes, SQLite integrity and active dataset files; it does not launch the restored system. Check paths in restored `.env` before ever running it. Source rollback files are separate from runtime backups.

## Startup after reboot

Only on the deployment PC, optionally run PowerShell:

```powershell
powershell -File deployment/windows_startup.ps1 -Mode Logon
```

This registers the existing launcher for the current Windows user's logon. For unattended boot, use `-Mode Boot` from an elevated PowerShell: Windows prompts locally for a deployment-account credential. The password is not written to a script or project file. Verify the GPU, model/tokenizer cache and file permissions under that account, then perform a real reboot check. Boot mode was not executed in the development environment; logon mode does not promise operation before Windows login. `-Mode Remove` removes only the named scheduled task. Task registration fails if its name already exists; remove/review that task before replacing it.

## Limits and recovery

GPU HTTP calls now have a finite default 300-second idle/read wait; generation is not automatically replayed. Only one ingestion worker builds at a time across domains. Ingestion uses smaller embedding batches and gives waiting chat work priority between batches. A running native batch cannot be preempted; ingestion can still affect latency and may fail safely if the queue stays busy. An interrupted build does not intentionally activate a partial generation.

Service/timing logs rotate at 10 MB with three rotated copies per file. Closed service/timing/alert logs older than 30 days are pruned at production startup. Local alert logs rotate separately. Conversation database/history is not pruned. Counters and queues are process-local: run one backend and one model process with this launcher. This is not a distributed queue or an uptime guarantee.

See `ACCEPTANCE.md` for the short target-machine check.
