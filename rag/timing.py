"""Diagnostic wall-clock timers only; no prompts, credentials, or response changes."""
import contextvars
import functools
import inspect
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import re
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

class ActivityFilter(logging.Filter):
    """Console only: detailed JSON still goes to the existing timing files."""
    def filter(self, record):
        try:
            data = json.loads(record.getMessage().split('TIMING ', 1)[1])
            task, event = data['task'], data['event']
            failed = data.get('status') == 'error' or event == 'failure' or (data.get('status_code') or 0) >= 400
            labels = {
                'llm.model_load': 'Qwen model load',
                'llm.tokenizer_load': 'Tokenizer load',
                'llm.rag.embeddings': 'Embedding',
                'llm.rag.rerank': 'Reranking',
                'rag.retrieval_service.retrieve_and_rerank': 'Retrieval',
                'llm_service.main.generate': 'Answer generation',
                'llm.nonstream_inference': 'Internal LLM call',
            }
            label = labels.get(task)
            if task == 'backend.http_request':
                if not (data.get('method') == 'POST' and '/chat/' in data.get('path', '')) and not failed:
                    return False
                label = 'Chat'
            if data.get('status') == 'control_flow':
                return False
            if failed:
                label = label or task
                message = label + ': ERROR ' + str(data.get('error_message') or data.get('error') or data.get('status_code'))
            elif label and event in ('start', 'end'):
                if event == 'start' and label not in ('Chat', 'Qwen model load'):
                    return False
                ending = ' cancelled' if data.get('status') == 'cancelled' else ' finished'
                message = label + (' working...' if event == 'start' else ending)
                if 'elapsed_ms' in data:
                    message += ' (' + format(data['elapsed_ms'] / 1000, '.2f') + ' s)'
            else:
                return False
            identity = data.get('request_id', '-')
            record.activity_message = message + (' [request ' + identity + ']' if identity != '-' else '')
            return True
        except Exception:
            return False


class ActivityFormatter(logging.Formatter):
    def format(self, record):
        return '[activity] ' + record.activity_message


_trace = contextvars.ContextVar('legal_llm_timing_trace', default='')
_logger = logging.getLogger('legal_llm.timing')
if not _logger.handlers:
    handler = logging.StreamHandler()  # stderr: stdout belongs to MCP's protocol.
    handler.addFilter(ActivityFilter())
    handler.setFormatter(ActivityFormatter())
    _logger.addHandler(handler)
    try:
        directory = Path(__file__).resolve().parents[1] / 'logs' / 'timing'
        directory.mkdir(parents=True, exist_ok=True)
        filename = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + f'_{os.getpid()}.log'
        file_handler = RotatingFileHandler(directory / filename, maxBytes=10_000_000, backupCount=3, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        _logger.addHandler(file_handler)
    except OSError:
        pass  # Console timings remain available if this directory is read-only.
_logger.setLevel(logging.INFO)
_logger.propagate = False


def trace_id():
    return _trace.get() or os.getenv('LEGAL_TIMING_TRACE_ID', '') or '-'


def emit(task, event='info', **fields):
    # Diagnostics must never turn a successful operation into a failed one.
    try:
        _logger.info('TIMING %s', json.dumps(dict(
            request_id=trace_id(), pid=os.getpid(), task=task, event=event, **fields),
            ensure_ascii=True, default=str))
    except Exception:
        pass


def exception_outcome(exc):
    # Streamlit rerun/stop deliberately unwind the script. Keep re-raising them;
    # only their diagnostic classification changes.
    kind = type(exc)
    if getattr(kind, '_timing_control_flow', False):
        return 'control_flow', None
    if kind.__module__.startswith('streamlit.') and kind.__name__ in {'RerunException', 'StopException'}:
        return 'control_flow', None
    if isinstance(exc, GeneratorExit) or (kind.__name__ == 'CancelledError' and kind.__module__ in {'asyncio.exceptions', 'concurrent.futures._base'}):
        return 'cancelled', kind.__name__
    return 'error', kind.__name__


@contextmanager
def span(task):
    token = None
    if trace_id() == '-':
        token = _trace.set(uuid.uuid4().hex[:16])
    started = time.perf_counter()
    status, error = 'ok', None
    emit(task, 'start')
    try:
        yield
    except BaseException as exc:
        status, error = exception_outcome(exc)
        raise
    finally:
        emit(task, 'end', elapsed_ms=round((time.perf_counter()-started)*1000, 3),
             status=status, error=error)
        if token is not None:
            _trace.reset(token)


def timed(task):
    """Time task calls, preserving signatures, return values and exceptions."""
    def decorate(fn):
        if inspect.isasyncgenfunction(fn):
            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                identity = trace_id()
                if identity == '-':
                    identity = uuid.uuid4().hex[:16]
                async def iterate():
                    started = time.perf_counter()
                    source = fn(*args, **kwargs)
                    status, error, first = 'ok', None, True
                    token = _trace.set(identity)
                    emit(task, 'start')
                    _trace.reset(token)
                    try:
                        while True:
                            token = _trace.set(identity)
                            try:
                                value = await source.__anext__()
                                if first:
                                    emit(task, 'first_yield', elapsed_ms=round((time.perf_counter()-started)*1000, 3))
                                    first = False
                            except StopAsyncIteration:
                                break
                            finally:
                                _trace.reset(token)
                            yield value
                    except BaseException as exc:
                        status, error = exception_outcome(exc)
                        raise
                    finally:
                        token = _trace.set(identity)
                        try:
                            await source.aclose()
                        finally:
                            emit(task, 'end', elapsed_ms=round((time.perf_counter()-started)*1000, 3), status=status, error=error)
                            _trace.reset(token)
                return iterate()
            return wrapped
        if inspect.isgeneratorfunction(fn):
            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                identity = trace_id()
                if identity == '-':
                    identity = uuid.uuid4().hex[:16]
                def iterate():
                    started = time.perf_counter()
                    source = fn(*args, **kwargs)
                    status, error, first = 'ok', None, True
                    token = _trace.set(identity)
                    emit(task, 'start')
                    _trace.reset(token)
                    try:
                        while True:
                            token = _trace.set(identity)
                            try:
                                value = next(source)
                                if first:
                                    emit(task, 'first_yield', elapsed_ms=round((time.perf_counter()-started)*1000, 3))
                                    first = False
                            except StopIteration:
                                break
                            finally:
                                _trace.reset(token)
                            yield value
                    except BaseException as exc:
                        status, error = exception_outcome(exc)
                        raise
                    finally:
                        token = _trace.set(identity)
                        try:
                            source.close()
                        finally:
                            emit(task, 'end', elapsed_ms=round((time.perf_counter()-started)*1000, 3), status=status, error=error)
                            _trace.reset(token)
                return iterate()
            return wrapped
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def wrapped(*args, **kwargs):
                with span(task):
                    return await fn(*args, **kwargs)
            return wrapped
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            with span(task):
                return fn(*args, **kwargs)
        return wrapped
    return decorate


class TimedLock:
    """Delegate to the existing lock; record wait and held time separately."""
    def __init__(self, lock, task):
        self.lock, self.task = lock, task

    def __enter__(self):
        with span(self.task + '.wait'):
            result = self.lock.__enter__()
        self.hold = span(self.task + '.held')
        self.hold.__enter__()
        return result

    def __exit__(self, *exc):
        try:
            return self.lock.__exit__(*exc)
        finally:
            self.hold.__exit__(*exc)

    async def __aenter__(self):
        with span(self.task + '.wait'):
            result = await self.lock.__aenter__()
        self.hold = span(self.task + '.held')
        self.hold.__enter__()
        return result

    async def __aexit__(self, *exc):
        try:
            return await self.lock.__aexit__(*exc)
        finally:
            self.hold.__exit__(*exc)


def trace_headers():
    return {'X-Legal-Timing-Trace': trace_id()}


class TimingMiddleware:
    """Pure ASGI: total request time includes streaming and persistence."""
    def __init__(self, app, service):
        self.app, self.service = app, service

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        supplied = dict(scope.get('headers', [])).get(b'x-legal-timing-trace', b'').decode('ascii', 'ignore')
        identity = supplied if re.fullmatch(r'[a-f0-9]{16}', supplied) else uuid.uuid4().hex[:16]
        token = _trace.set(identity)
        started = time.perf_counter()
        task = self.service + '.http_request'
        first_body, first_text, buffer, status_code = True, True, b'', None
        emit(task, 'start', method=scope['method'], path=scope.get('path', ''))
        async def observed_send(message):
            nonlocal first_body, first_text, buffer, status_code
            if message['type'] == 'http.response.start':
                status_code = message['status']
                emit(task, 'headers', elapsed_ms=round((time.perf_counter()-started)*1000, 3), status_code=status_code)
            elif message['type'] == 'http.response.body' and message.get('body'):
                if first_body:
                    emit(task, 'first_body', elapsed_ms=round((time.perf_counter()-started)*1000, 3))
                    first_body = False
                if first_text:
                    buffer += message['body']
                    lines = buffer.split(b'\n')
                    buffer = lines.pop()[-65536:]
                    for line in lines:
                        if line.startswith(b'data:'):
                            try:
                                data = json.loads(line[5:].strip())
                                choices = data.get('choices', [])
                                text = data.get('text') if data.get('type') == 'chunk' else (
                                    choices[0].get('delta', {}).get('content') if choices else None)
                                if text:
                                    emit(task, 'first_answer_text', elapsed_ms=round((time.perf_counter()-started)*1000, 3))
                                    first_text = False
                                    break
                            except (ValueError, TypeError, AttributeError, IndexError, KeyError):
                                pass
            await send(message)  # Original ASGI message, unchanged.
        outcome, error = 'ok', None
        try:
            await self.app(scope, receive, observed_send)
        except BaseException as exc:
            outcome, error = exception_outcome(exc)
            raise
        finally:
            emit(task, 'end', elapsed_ms=round((time.perf_counter()-started)*1000, 3),
                 status=outcome, error=error, status_code=status_code,
                 method=scope['method'], path=scope.get('path', ''))
            _trace.reset(token)
