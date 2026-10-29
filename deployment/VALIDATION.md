# Release validation

134 automated checks passed locally:

- 26 deployment-operation checks: configuration, bounded requests, metrics, timeout/priority headers, cross-process locks, ingestion serialization, supervisor restart ordering, backup/restore safety and load-report parsing.
- 1 real local Caddy HTTPS integration: temporary internal CA, Secure/HttpOnly session cookie, first streamed chunk before answer completion, saved history and blocked public admin routes. The internal CA was not installed into system trust.
- 21 existing cookie-session/capacity/admin checks.
- 4 existing ingestion snapshot/rollback checks, including failed vector construction preserving active data.
- 82 existing routing, follow-up, context, structured-output, record and service-configuration checks in isolated runs.

Python compilation, frontend script syntax and Caddy configuration validation passed. Existing regression fixtures omitted from the review ZIP were reused from the previous supplied source for local checks.

Model/search I/O was stubbed. No real Windows GPU model benchmark, public ACME certificate issuance, Windows scheduled-task execution or reboot was performed here. Supervisor restart ordering was tested with fake child processes; cross-process lock release after killing a worker was also tested with a real subprocess on the development OS. The Windows locking/task branches still require the target-machine check.

Core retrieval, reranking/selection, domain routing, reference resolution, answer-generation code, MCP implementation, session identity and database schema were byte-compared with the verified cookie-session source and left unchanged. Ingestion scheduling/batch request size and operational boundaries changed deliberately.

The built-in load-test tool produces measurements and records real rate-limit failures. It does not certify an accuracy score, concurrent-user capacity or uptime target. RSS measurements are not VRAM measurements.
