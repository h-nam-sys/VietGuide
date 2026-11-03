# Administration

Admin is available at localhost:8501 and requires `ADMIN_PASSWORD`.

## Domain workspace

Select the domain once, then choose data update, answer configuration, versions
or management. Creating a new domain starts the upload/profile workflow.

| Setting | Purpose | Requires re-ingestion? |
|---|---|---|
| Scope description / routing examples | Describe which questions belong to the domain | No |
| Field mapping | Connect input columns to the internal field keys | Yes |
| Primary title/topic | Identify the record's main subject | Yes |
| Search fields / embedding prefix | Define the text used to build retrieval vectors | Yes |
| Excluded source fields | Omit configured fields from answer/source context; stored data remains | Set during ingestion |
| Enabled/disabled | Include or exclude the domain from chat routing without deleting data | No |

Uploading to an existing domain replaces that domain's active dataset after
successful ingestion. It is not an append operation. JSON profile import/export
supports reviewed configurations without a second raw JSON editing interface.

The versions view shows completed dataset generations. Activating another version
changes the active SQL/vector snapshot. This differs from restoring answer settings.
Deleting a domain archives its configuration and generations; the default domain
is protected. Confirmation and in-progress ingestion checks remain enforced.

## Answer settings

Choose Strict or Friendly before editing. Saving one mode preserves the other.
The modes use their saved prompts and sampling settings; a label alone does not
guarantee factual correctness.

| Setting | Effect |
|---|---|
| Default mode | Mode used when a request does not supply one |
| Source-answer prompt | Instructions for answering from retrieved records |
| Conversation prompt | Friendly-mode instructions for ordinary conversation |
| Temperature / top-p | Sampling controls; lower randomness does not guarantee accuracy |
| Maximum answer tokens | Output length budget; too low can truncate answers |
| Minimum retrieval score | Filters low-scoring candidates; too high can reject valid records |
| Candidate count | Retrieval/reranking candidate budget; increasing it can increase latency |
| Result count | Limit on records retained for answering; cannot exceed candidate count |
| User template | Must retain `{context}` and `{query}` placeholders |
| Missing-data message | Domain fallback text when information cannot be found |
| Direct field output | Returns source values directly for supported explicit lookups; other questions still use generation |

Saving changes applies to subsequent requests without rebuilding indexes.
Restoring settings removes runtime overrides and uses the active dataset version's
configuration. Existing `.env` values and per-domain/per-mode overrides should be
checked together when a value appears unexpected.

## Online search

The operational controls are enable/disable, permitted domains, allowed websites,
location use, source limit, timeout, answer length and answer prompt. The bundled
provider also has a Gemini model setting. These settings do not change the local
Qwen model.

The search test performs a real provider request with the displayed draft. It can
incur API usage and does not save the draft. Chat search still follows its existing
permission flow. Editing a domain's websites and then selecting another domain
retains those draft rules until saving.

MCP transport, endpoint, token environment variable, tool name and argument
mapping remain in `data/mcp_settings.json`. Existing connection values are
preserved by the admin form. Credentials belong in environment variables rather
than that JSON file.

## Monitoring

Select a history/feedback row to inspect its answer and sources. The global
refresh action reloads the selected page's data. Service counters and timings
reset with the process; RSS measures system memory, not GPU VRAM.

`/health` checks liveness. `/readyz` also checks the conversation database,
available data and the inference service. Readiness does not prove answer accuracy.
