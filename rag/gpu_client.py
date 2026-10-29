"""Share GPU models in the existing LLM service, including ingestion subprocesses."""
import asyncio
from functools import lru_cache
import math
import os
from llama_index.core.base.embeddings.base import BaseEmbedding

from .domain_config import RAGUnavailableError, llm_url
from .timing import trace_headers, span
from .gpu_runtime import EMBED_MODEL


@lru_cache(maxsize=2)
def _client(base_url):
    import httpx
    # Local model requests can wait for the same serialized GPU queue as Qwen.
    return httpx.Client(base_url=base_url, timeout=httpx.Timeout(float(os.getenv('GPU_SERVICE_TIMEOUT', '300')), connect=5, pool=10), trust_env=False)


def gpu_call(path, payload):
    try:
        with span('rag.directml.service_call'):
            headers = trace_headers()
            if os.getenv('LEGAL_INGESTION') == '1':
                headers['X-Inference-Priority'] = 'background'
            response = _client(llm_url()).post(path, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
    except Exception as exc:
        raise RAGUnavailableError('Shared GPU inference unavailable. Check the LLM service and GPU setup logs; CPU fallback is disabled.') from exc


class RemoteDirectMLEmbedding(BaseEmbedding):
    def __init__(self):
        super().__init__(model_name=EMBED_MODEL, embed_batch_size=8)

    def _vectors(self, texts, kind):
        if not texts:
            return []
        result = gpu_call('/v1/rag/embeddings', {'texts': texts, 'kind': kind})['embeddings']
        if len(result) != len(texts) or any(len(row) != 1024 or not all(math.isfinite(x) for x in row) for row in result):
            raise RAGUnavailableError('Shared GPU service returned invalid embedding vectors')
        return result

    def _get_query_embedding(self, query):
        return self._vectors([query], 'query')[0]

    async def _aget_query_embedding(self, query):
        return await asyncio.to_thread(self._get_query_embedding, query)

    def _get_text_embedding(self, text):
        return self._vectors([text], 'text')[0]

    async def _aget_text_embedding(self, text):
        return await asyncio.to_thread(self._get_text_embedding, text)

    def _get_text_embeddings(self, texts):
        result = []
        batch = int(os.getenv('INGEST_EMBED_BATCH', '4')) if os.getenv('LEGAL_INGESTION') == '1' else 32
        if not 1 <= batch <= 32:
            raise ValueError('Invalid embedding batch size')
        for start in range(0, len(texts), batch):
            result.extend(self._vectors(texts[start:start + batch], 'text'))
        return result
