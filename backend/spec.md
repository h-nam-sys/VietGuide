# Backend API

FastAPI serves the chat UI at `/chat` and interactive API documentation at `/docs`.
The default backend address is `http://localhost:8000`.

## Browser sessions

`GET /session` establishes the anonymous browser session and returns its CSRF
token. Keep the session cookie and send the token in the `X-CSRF-Token` header
for mutations. Conversation and message access is restricted to the current
cookie owner. There is no public login requirement.

## Chat endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/domains` | Available domains |
| POST | `/start-conversation` | Create an owned conversation |
| POST | `/chat/{conversation_id}` | Send a question; streamed SSE response |
| GET | `/conversations` | List the current owner's conversations |
| GET | `/history/{conversation_id}` | Read owned conversation history |
| POST | `/feedback/{message_id}` | Record a rating for an owned answer |
| POST | `/chat/{conversation_id}/action/mcp` | Respond to an offered external-search action |
| POST | `/chat/{conversation_id}/action/form` | Respond to an offered form action |

Conversation creation accepts `title`, `domain` and `allow_switching`.
A chat request requires `user_prompt`; optional `domain`, `allow_switching`,
`reset_domain` and `mode` control routing and answer policy. Omitted values
preserve the conversation preference or configured default. `mode` accepts
`strict` or `friendly`. Automatic switching must be enabled to switch domains
within a conversation.

Responses use Server-Sent Events, including metadata and answer chunks, followed
by `[DONE]`. Consume the event stream rather than expecting one JSON answer.
External search and form downloads follow the offered action and its permission
flow. See `/docs` for the current request schemas.

## Operations

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Backend liveness |
| GET | `/readyz` | Readiness including required dependencies/data |
| GET | `/admin/operations` | Operational status; requires `X-Admin-Key` |
| GET | `/admin/failed-logs` | Failed-answer records; requires `X-Admin-Key` |

A conversation already processing a request can return 409; exhausted capacity
can return 503. Inspect service logs using the request ID when diagnosing errors.

## Model service

The model service normally listens at `localhost:8001`. It exposes
`/v1/chat/completions`, `/v1/rag/embeddings`, `/v1/rag/rerank`, `/v1/rag/status`
and `/v1/operations`. Backend retrieval calls the shared model service for
embedding and reranking. In hybrid mode Docker reaches it through
`host.docker.internal`; the browser continues using the backend.
