"""Bounded operational counters, no prompts, cookies, source records or credentials."""
import asyncio
from collections import Counter, deque
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
_lock = threading.Lock()
_counts = Counter()
_durations = deque(maxlen=200)
_first_text = deque(maxlen=200)
_queues = {}
_last_alert = {}


def alert(event, service='backend'):
    """Local alerts only, deduplicated; never send messages to outside services."""
    try:
        key = service + ':' + event
        with _lock:
            now = time.time()
            if now - _last_alert.get(key, 0) < 30:
                return
            _last_alert[key] = now
        logger = logging.getLogger('legal_llm.ops.alerts')
        if not logger.handlers:
            directory = ROOT / 'logs/operations'
            directory.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(directory / ('alerts_'+str(os.getpid())+'.log'), maxBytes=2_000_000, backupCount=3, encoding='utf-8')
            logger.addHandler(handler)
            logger.setLevel(logging.WARNING)
            logger.propagate = False
        logger.warning(json.dumps({'time': now, 'service': service, 'event': event}))
    except Exception:
        pass


def queue_state(kind, active, waiting, rejected=False, wait_ms=None):
    with _lock:
        prior = _queues.setdefault(kind, {'active': 0, 'waiting': 0, 'rejected': 0, 'last_wait_ms': 0})
        prior.update(active=active, waiting=waiting)
        if rejected:
            prior['rejected'] += 1
        if wait_ms is not None:
            prior['last_wait_ms'] = round(wait_ms, 2)


def snapshot():
    def stats(values):
        data = sorted(values)
        return {'samples': len(data), 'mean_ms': round(sum(data)/len(data), 2) if data else None,
                'p95_ms': round(data[max(0, (95*len(data)+99)//100-1)], 2) if data else None}
    with _lock:
        result = {'pid': os.getpid(), 'counts': dict(_counts), 'queues': {k: dict(v) for k, v in _queues.items()},
                  'request_duration': stats(_durations), 'first_answer_text': stats(_first_text)}
    try:
        import psutil
        result['memory'] = {'process_rss_mb': round(psutil.Process().memory_info().rss/1024**2, 1),
                            'host_available_mb': round(psutil.virtual_memory().available/1024**2, 1)}
    except Exception:
        result['memory'] = None
    return result


class OperationsMiddleware:
    def __init__(self, app, *, service):
        self.app, self.service = app, service

    async def __call__(self, scope, receive, send):
        selected = (scope.get('path', '').startswith('/chat/') if self.service == 'backend' else
                    scope.get('path') in {'/v1/chat/completions','/v1/rag/embeddings','/v1/rag/rerank'})
        if scope['type'] != 'http' or not selected or scope.get('method') != 'POST':
            return await self.app(scope, receive, send)
        start = time.monotonic()
        status, first, failed, done, streaming = 500, None, False, False, False
        buffer = ''
        async def measured_send(message):
            nonlocal status, first, buffer, failed, done, streaming
            if message['type'] == 'http.response.start':
                status = message['status']
                streaming = b'text/event-stream' in dict(message.get('headers',[])).get(b'content-type',b'')
            elif message['type'] == 'http.response.body':
                # Inspect SSE frames in bounded memory; never persist answer text.
                buffer += message.get('body', b'').decode('utf-8', errors='replace')
                frames = buffer.split('\n\n')
                buffer = frames.pop()[-65536:]
                for frame in frames:
                    for line in frame.splitlines():
                        if not line.startswith('data:'):
                            continue
                        text = line[5:].strip()
                        if text == '[DONE]':
                            done = True
                        else:
                            try:
                                data = json.loads(text)
                                content = data.get('text') if isinstance(data,dict) and data.get('type') == 'chunk' else None
                                if isinstance(data,dict) and not content:
                                    choices=data.get('choices')
                                    if isinstance(choices,list) and choices and isinstance(choices[0],dict):
                                        content=choices[0].get('delta',{}).get('content')
                                if isinstance(content,str) and content:
                                    if first is None:
                                        first = (time.monotonic()-start)*1000
                                    if 'Lỗi hệ thống:' in content:
                                        failed = True
                            except (ValueError, TypeError, AttributeError):
                                pass
            await send(message)
        try:
            await self.app(scope, receive, measured_send)
        except BaseException:
            failed = True
            raise
        finally:
            failed = failed or (streaming and not done)
            with _lock:
                _counts['requests'] += 1
                _counts['http_' + str(status)] += 1
                if status >= 500 or failed:
                    _counts['errors'] += 1
                _durations.append((time.monotonic()-start)*1000)
                if first is not None and status < 400 and not failed:
                    _first_text.append(first)
            if status >= 500 or failed:
                alert('chat_request_failed', self.service)


class RequestBodyLimitMiddleware:
    """Read only bounded incoming JSON before reserving scarce chat capacity."""
    def __init__(self, app):
        self.app = app
        self.limit = int(os.getenv('API_MAX_BODY_BYTES', '262144'))
        self.timeout = float(os.getenv('API_BODY_TIMEOUT', '15'))
        if not 1024 <= self.limit <= 4_194_304 or not 0 < self.timeout <= 120:
            raise ValueError('Invalid API request body limit/timeout')

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('method') not in {'POST', 'PUT', 'PATCH'}:
            return await self.app(scope, receive, send)
        async def reject(status, detail):
            await send({'type': 'http.response.start', 'status': status,
                        'headers': [(b'content-type', b'application/json'), (b'cache-control', b'no-store')]})
            await send({'type': 'http.response.body', 'body': json.dumps({'detail': detail}).encode()})
        headers = dict(scope.get('headers', []))
        try:
            if int(headers.get(b'content-length', b'0')) > self.limit:
                return await reject(413, 'Request is too large')
        except ValueError:
            return await reject(400, 'Invalid Content-Length')
        body = bytearray()
        async def read():
            while True:
                event = await receive()
                if event['type'] == 'http.disconnect':
                    return 'disconnected'
                body.extend(event.get('body', b''))
                if len(body) > self.limit:
                    return 'large'
                if not event.get('more_body'):
                    return 'ok'
        try:
            outcome = await asyncio.wait_for(read(), self.timeout)
        except asyncio.TimeoutError:
            return await reject(408, 'Request body timed out')
        if outcome == 'disconnected':
            return
        if outcome == 'large':
            return await reject(413, 'Request is too large')
        sent = False
        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
            return await receive()
        await self.app(scope, replay, send)
