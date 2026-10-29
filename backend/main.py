import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from fastapi import FastAPI, Request, Response, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
load_dotenv(ROOT / "backend/.env")
load_dotenv(ROOT / ".env")
from routers import chat
from database import init_db, get_db
from rate_limiter import limiter
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

load_dotenv()

from rag.service_config import llm_url
LLM_SERVICE_URL = llm_url()
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", LLM_SERVICE_URL)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

from rag.timing import timed, span, emit, TimedLock, trace_headers, TimingMiddleware, trace_id

app = FastAPI(
    title="Legal LLM Backend",
    version="1.0",
    description="Vietnamese legal consulting system with RAG and LLM integration"
)

# 0. Initialize the async database when the server starts
@app.on_event("startup")
@timed('backend.main.startup_event')
async def startup_event():
    await init_db()

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 1. Mount static assets
app.mount("/static", StaticFiles(directory=str(ROOT / "backend/static")), name="static")

# 2. Serve the chat interface
@app.get("/chat")
def serve_chat_ui():
    return FileResponse(ROOT / "backend/static/index.html")

app.include_router(chat.router)

# Anonymous chat: one signed persistent cookie per browser, no username/login.
from services.session_auth import bootstrap_session, PrivateResponsesMiddleware
from rag.request_capacity import RequestCapacityMiddleware
from rag.operations import OperationsMiddleware, RequestBodyLimitMiddleware
app.add_middleware(RequestCapacityMiddleware, kind='chat')
app.add_middleware(PrivateResponsesMiddleware)
app.add_middleware(RequestBodyLimitMiddleware)
app.add_middleware(OperationsMiddleware, service='backend')
app.add_middleware(TimingMiddleware, service='backend')


@app.get('/session')
def browser_session(request: Request, response: Response):
    return bootstrap_session(request, response)


@app.get("/")
def health_check():
    return {
        "status": "Backend server is running live",
        "chat_ui": "http://localhost:8000/chat"
    }

@app.get("/health")
def quick_health():
    return {"status": "ok"}


@app.get('/readyz')
async def ready(response: Response, db=Depends(get_db)):
    from services.operations_service import readiness
    report = await readiness(db)
    response.status_code = 200 if report['ready'] else 503
    return report


@app.get('/admin/operations', dependencies=[Depends(chat.require_admin)])
async def operational_status(db=Depends(get_db)):
    from services.operations_service import operations_status
    return await operations_status(db)
