"""Admission remains held until SSE completes; rejected work never reaches the app."""
import asyncio
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.request_capacity import RequestCapacityMiddleware


def test_capacity_stream_lifetime_queue_and_health(monkeypatch):
    monkeypatch.setenv('INFERENCE_MAX_WAITING', '1')
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []
        async def app(scope, receive, send):
            calls.append(scope['path'])
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            if scope['path'] == '/v1/chat/completions':
                entered.set()
                await send({'type': 'http.response.body', 'body': b'data: first\n\n', 'more_body': True})
                await release.wait()
            await send({'type': 'http.response.body', 'body': b'data: [DONE]\n\n'})
        gate = RequestCapacityMiddleware(app, kind='inference')
        async def request(path='/v1/chat/completions'):
            sent = []
            async def send(event): sent.append(event)
            await gate({'type': 'http', 'method': 'POST', 'path': path}, None, send)
            return sent
        first = asyncio.create_task(request())
        await entered.wait()
        second = asyncio.create_task(request('/v1/rag/embeddings'))
        await asyncio.sleep(0)
        assert gate.admitted == 2
        rejected = await request('/v1/rag/rerank')
        assert rejected[0]['status'] == 503
        assert (b'retry-after', b'5') in rejected[0]['headers']
        assert '/v1/rag/rerank' not in calls
        assert (await request('/v1/rag/status'))[0]['status'] == 200
        assert '/v1/rag/embeddings' not in calls  # SSE headers alone did not release the slot
        release.set()
        assert (await first)[-1]['body'] == b'data: [DONE]\n\n'
        assert (await second)[0]['status'] == 200
        assert gate.admitted == 0
    asyncio.run(run())


def test_wait_timeout_and_cancellation_release_capacity(monkeypatch):
    monkeypatch.setenv('INFERENCE_QUEUE_TIMEOUT', '0.02')
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        async def app(scope, receive, send):
            entered.set()
            await release.wait()
        gate = RequestCapacityMiddleware(app, kind='inference')
        async def request():
            sent = []
            async def send(event): sent.append(event)
            await gate({'type': 'http', 'method': 'POST', 'path': '/v1/chat/completions'}, None, send)
            return sent
        first = asyncio.create_task(request())
        await entered.wait()
        assert (await request())[0]['status'] == 503
        assert gate.admitted == 1
        waiting = asyncio.create_task(request())
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError): await waiting
        assert gate.admitted == 1
        first.cancel()
        with pytest.raises(asyncio.CancelledError): await first
        assert gate.admitted == 0
        release.set()
        await request()
        assert gate.admitted == 0
    asyncio.run(run())


def test_same_conversation_cannot_interleave_actions(monkeypatch):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        async def app(scope, receive, send):
            entered.set()
            await release.wait()
        gate = RequestCapacityMiddleware(app, kind='chat')
        async def request(path):
            sent = []
            async def send(event): sent.append(event)
            await gate({'type': 'http', 'method': 'POST', 'path': path}, None, send)
            return sent
        first = asyncio.create_task(request('/chat/1'))
        await entered.wait()
        for path in ['/chat/1', '/chat/1/action/mcp', '/chat/1/action/form']:
            assert (await request(path))[0]['status'] == 409
        other = asyncio.create_task(request('/chat/2'))
        await asyncio.sleep(0.01)
        assert gate.admitted == 2
        release.set()
        await asyncio.gather(first, other)
        assert not gate.conversations
    asyncio.run(run())


def test_exception_releases_slot():
    async def run():
        async def app(*args): raise RuntimeError('failure')
        gate = RequestCapacityMiddleware(app, kind='chat')
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await gate({'type': 'http', 'method': 'POST', 'path': '/chat/1'}, None, None)
            assert gate.admitted == 0 and not gate.conversations
    asyncio.run(run())


@pytest.mark.parametrize('name,value', [('CHAT_MAX_ACTIVE','0'), ('CHAT_MAX_WAITING','-1'),
                                      ('CHAT_QUEUE_TIMEOUT','0'), ('INFERENCE_MAX_ACTIVE','2')])
def test_bad_configuration_fails_clearly(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        RequestCapacityMiddleware(None, kind='inference' if name.startswith('INFERENCE') else 'chat')
