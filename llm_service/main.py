"""Existing port/startup command; supports both JSON and SSE responses."""
import json
import os
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from llama_cpp import Llama
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
import sys
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from rag.timing import timed, span, emit, TimedLock, trace_headers, TimingMiddleware, trace_id
from rag.hardware_diagnostics import memory_stage, NativeStartupLog, llama_runtime, finish_llama_startup
from rag.service_config import llm_port
import llama_cpp.llama_cpp as llama_api
load_dotenv(ROOT_DIR / "backend/.env")
load_dotenv(ROOT_DIR / ".env")
app = FastAPI(title="Legal LLM Service")
from rag.request_capacity import RequestCapacityMiddleware
from rag.operations import OperationsMiddleware
app.add_middleware(RequestCapacityMiddleware, kind='inference')
app.add_middleware(OperationsMiddleware, service='llm')
app.add_middleware(TimingMiddleware, service='llm')
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/qwen3.5_4b.gguf"))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = ROOT_DIR / MODEL_PATH
# Preserve the existing tokenizer choice; override TOKENIZER_NAME if your GGUF needs another.
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "Qwen/Qwen3-0.6B")
with memory_stage('llm.tokenizer_load'), span('llm.tokenizer_load'):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
with memory_stage('llm.model_load'), span('llm.model_load'), NativeStartupLog(llama_api):
    llm = Llama(model_path=str(MODEL_PATH), n_ctx=4096, n_threads=8, n_gpu_layers=-1, verbose=True)
# Verbosity changes only during startup; retain the original inference verbosity.
finish_llama_startup(llm)
llama_runtime(llama_api)
_model_lock = Lock()


class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIRequest(BaseModel):
    messages: list[OpenAIMessage]
    stream: bool = True
    max_tokens: int = Field(default=800, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0, le=1)


@timed('llm_service.main.format_prompt')
def format_prompt(payload):
    return tokenizer.apply_chat_template(
        [{"role": message.role, "content": message.content} for message in payload.messages],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


@app.post("/v1/chat/completions")
@timed('llm_service.main.chat_completions')
def chat_completions(payload: OpenAIRequest):
    # Sync endpoint/generator keeps CPU inference off FastAPI's event loop.
    if not payload.stream:
        with TimedLock(_model_lock, 'llm.inference_queue'):
            with span('llm.nonstream_inference'):
                result = llm(format_prompt(payload), max_tokens=payload.max_tokens,
                             temperature=payload.temperature, top_p=payload.top_p, stream=False, stop=["<|im_end|>"])
        return {"choices": [{"index": 0, "message": {"role": "assistant", "content": result["choices"][0]["text"]},
                              "finish_reason": result["choices"][0].get("finish_reason", "stop")} ]}

    @timed('llm_service.main.generate')
    def generate():
        with TimedLock(_model_lock, 'llm.inference_queue'):
            with span('llm.stream_setup'):
                output = llm(format_prompt(payload), max_tokens=payload.max_tokens,
                             temperature=payload.temperature, top_p=payload.top_p, stream=True, stop=["<|im_end|>"])
            try:
                for chunk in output:
                    text = chunk["choices"][0]["text"]
                    if text:
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': text}}]}, ensure_ascii=False)}\n\n"
            finally:
                close = getattr(output, "close", None)
                if close:
                    close()
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# Shared DirectML retrieval lives in this existing model process. Backend/admin
# restarts and ingestion subprocesses do not create duplicate GPU model copies.
from functools import lru_cache
from typing import Literal
from fastapi import HTTPException


class GPUEmbeddingRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=32)
    kind: Literal['query', 'text'] = 'text'


class GPURerankRequest(BaseModel):
    pairs: list[tuple[str, str]] = Field(min_length=1)


@lru_cache(maxsize=1)
def _gpu_embedding():
    from rag.gpu_embedding import LocalDirectMLEmbedding
    with memory_stage('llm.rag.embedding_load'):
        return LocalDirectMLEmbedding()


@lru_cache(maxsize=1)
def _gpu_reranker():
    from rag.gpu_runtime import LocalDirectMLReranker
    with memory_stage('llm.rag.reranker_load'):
        return LocalDirectMLReranker()


@app.get('/health')
def health():
    return {'alive': True}


@app.get('/v1/rag/status')
def gpu_status():
    from rag.deployment_config import model_assets
    return {'gpu_protocol': 1, 'qwen_ready': True,
            'retrieval_assets_ready': model_assets(ROOT_DIR, check_runtime=True)['retrieval_assets'],
            'embedding_loaded': bool(_gpu_embedding.cache_info().currsize),
            'reranker_loaded': bool(_gpu_reranker.cache_info().currsize),
            'gpu_queue_busy': _model_lock.locked()}


@app.post('/v1/rag/embeddings')
@timed('llm.rag.embeddings')
def gpu_embeddings(payload: GPUEmbeddingRequest):
    try:
        with TimedLock(_model_lock, 'llm.rag.gpu_queue'):
            model = _gpu_embedding()
            values = ([model.get_query_embedding(text) for text in payload.texts]
                      if payload.kind == 'query' else model.get_text_embedding_batch(payload.texts))
        return {'embeddings': values}
    except Exception as exc:
        emit('llm.rag.embeddings', 'failure', error=type(exc).__name__, error_message=str(exc)[:500])
        raise HTTPException(503, 'DirectML embedding unavailable; inspect LLM timing logs. CPU fallback is disabled.') from exc


@app.post('/v1/rag/rerank')
@timed('llm.rag.rerank')
def gpu_rerank(payload: GPURerankRequest):
    try:
        with TimedLock(_model_lock, 'llm.rag.gpu_queue'):
            values = _gpu_reranker().compute_score([list(pair) for pair in payload.pairs])
        return {'logits': values}
    except Exception as exc:
        emit('llm.rag.rerank', 'failure', error=type(exc).__name__, error_message=str(exc)[:500])
        raise HTTPException(503, 'DirectML reranker unavailable; inspect LLM timing logs. CPU fallback is disabled.') from exc


@app.get('/v1/operations')
def operational_status():
    from rag.operations import snapshot
    return snapshot()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("LLM_BIND_HOST", "127.0.0.1"), port=llm_port())
