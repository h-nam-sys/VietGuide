"""Bound active/waiting HTTP work, holding admission through the final SSE body."""
import asyncio
import json
import os
import re
import time
from .operations import queue_state, alert


class PrioritySlots:
    """FIFO within each class; foreground work precedes queued ingestion batches."""
    def __init__(self, limit):
        self.limit, self.used, self.sequence = limit, 0, 0
        self.waiters = []

    async def acquire(self, priority=0):
        if self.used < self.limit and not self.waiters:
            self.used += 1
            return
        future = asyncio.get_running_loop().create_future()
        self.sequence += 1
        entry = (priority, self.sequence, future)
        self.waiters.append(entry)
        self.waiters.sort(key=lambda x: x[:2])
        try:
            await future
        except BaseException:
            if entry in self.waiters:
                self.waiters.remove(entry)
            elif future.done() and not future.cancelled():
                self.release()
            raise

    def release(self):
        self.used -= 1
        while self.waiters and self.used < self.limit:
            _, _, future = self.waiters.pop(0)
            if not future.cancelled():
                self.used += 1
                future.set_result(None)


class RequestCapacityMiddleware:
    def __init__(self, app, *, kind):
        self.app = app
        self.kind = kind
        prefix = 'CHAT' if kind == 'chat' else 'INFERENCE'
        self.active = int(os.getenv(prefix + '_MAX_ACTIVE', '2' if kind == 'chat' else '1'))
        self.waiting = int(os.getenv(prefix + '_MAX_WAITING', '2' if kind == 'chat' else '3'))
        self.timeout = float(os.getenv(prefix + '_QUEUE_TIMEOUT', '60'))
        if self.active < 1 or self.waiting < 0 or not 0 < self.timeout <= 300:
            raise ValueError('Invalid ' + prefix + ' request capacity settings')
        if kind == 'inference' and self.active != 1:
            raise ValueError('INFERENCE_MAX_ACTIVE must be 1 for the shared model process')
        self.semaphore = PrioritySlots(self.active)
        self.admitted = 0
        self.conversations = set()

    async def reject(self, send, status, detail):
        queue_state(self.kind, self.semaphore.used, max(0, self.admitted-self.semaphore.used), rejected=True)
        if status == 503:
            alert('queue_capacity_exceeded', self.kind)
        body = json.dumps({'detail': detail}, ensure_ascii=False).encode()
        await send({'type': 'http.response.start', 'status': status,
                    'headers': [(b'content-type', b'application/json; charset=utf-8'),
                                (b'retry-after', b'5'), (b'cache-control', b'no-store')]})
        await send({'type': 'http.response.body', 'body': body})

    async def __call__(self, scope, receive, send):
        path = scope.get('path', '')
        match = re.fullmatch(r'/chat/(\d+)(?:/action/(?:mcp|form))?', path)
        selected = bool(match) if self.kind == 'chat' else path in {
            '/v1/chat/completions', '/v1/rag/embeddings', '/v1/rag/rerank'}
        if scope['type'] != 'http' or scope.get('method') != 'POST' or not selected:
            return await self.app(scope, receive, send)
        conversation = match.group(1) if match else None
        # No await between testing and reserving capacity: atomic on the ASGI loop.
        if conversation and conversation in self.conversations:
            return await self.reject(send, 409, 'Cuộc trò chuyện đang xử lý một yêu cầu. Vui lòng chờ hoàn tất.')
        if self.admitted >= self.active + self.waiting:
            return await self.reject(send, 503, 'Hệ thống đang bận. Vui lòng thử lại sau vài giây.')
        self.admitted += 1
        if conversation:
            self.conversations.add(conversation)
        acquired = False
        started = time.monotonic()
        priority = 1 if self.kind == 'inference' and dict(scope.get('headers', [])).get(b'x-inference-priority') == b'background' else 0
        queue_state(self.kind, self.semaphore.used, max(0, self.admitted-self.semaphore.used))
        try:
            try:
                async with asyncio.timeout(self.timeout):
                    await self.semaphore.acquire(priority)
                    acquired = True
                queue_state(self.kind, self.semaphore.used, max(0, self.admitted-self.semaphore.used),
                            wait_ms=(time.monotonic()-started)*1000)
            except asyncio.TimeoutError:
                return await self.reject(send, 503, 'Hệ thống đang bận. Vui lòng thử lại sau vài giây.')
            await self.app(scope, receive, send)
        finally:
            if acquired:
                self.semaphore.release()
            self.admitted -= 1
            if conversation:
                self.conversations.discard(conversation)
            queue_state(self.kind, self.semaphore.used, max(0, self.admitted-self.semaphore.used))
