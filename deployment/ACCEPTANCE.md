# Target-machine acceptance

Code checks and local simulated/model-stub tests are complete. These checks remain for your actual Windows/GPU machine:

1. Run `check`, start production normally, then run `probe`. Readiness should become true. Check admin status and the recorded dependency manifest.
2. Ask a supported question, a follow-up and a domain switch. Check source expansion, feedback, a form and one approved MCP search. Your previous manual results are the baseline.
3. Use a private window for a second user; verify separate history. Run one user then two users with the load-test command. Keep the report; observe VRAM separately. Do not claim higher capacity without testing it.
4. Upload a small dataset while asking a question. Confirm chat completes and ingestion progress/activation/rollback still work. Some slowdown is expected on the shared GPU.
5. On an idle test instance, terminate ONLY its LLM service process using its PID from admin diagnostics. Confirm a single replacement starts, readiness returns, and the next new question works. Do not terminate unrelated Python processes or replay a half-finished answer automatically.
6. Stop services. Make a backup and run restore-check into a separate NEW folder. Confirm your session key/forms/data are represented. Keep this backup privately on another disk as well.
7. Configure startup under the intended Windows account and reboot once. Check all services and history afterward. Test the public HTTPS address separately after DNS/firewall/certificates are configured.

Record host OS, GPU, RAM/VRAM, dependency snapshot and test report with the release. Set response-time/concurrency targets from these measurements. No public certificate issuance, real reboot, GPU recovery or GPU throughput claim is implied by the local automated checks.
