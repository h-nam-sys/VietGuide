"""Real HTTP/SQLite session and isolation checks; model/search I/O is stubbed."""
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backend')]
from database import Base, Message, User
from database import get_db
from routers import chat as routes
from services import session_auth as auth
from services import chat_orchestrator as orchestrator
from services.db_service import add_message
from rag.request_capacity import RequestCapacityMiddleware


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv('SESSION_SECRET', 'test-secret-' * 8)
    monkeypatch.setenv('ADMIN_PASSWORD', 'admin-test-password')
    monkeypatch.delenv('SESSION_COOKIE_SECURE', raising=False)
    auth.session_secret.cache_clear()
    engine = create_async_engine('sqlite+aiosqlite:///' + str(tmp_path / 'chat.db'))
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async def setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    asyncio.run(setup())
    app = FastAPI()
    app.state.limiter = routes.limiter
    app.include_router(routes.router)
    app.add_api_route('/session', auth.bootstrap_session)
    app.add_middleware(RequestCapacityMiddleware, kind='chat')
    async def db():
        async with factory() as session:
            yield session
    app.dependency_overrides[get_db] = db
    async def answer(cid, payload, db):
        await add_message(db, cid, 'user', payload.user_prompt)
        await add_message(db, cid, 'assistant', 'saved answer', ['source'])
        async def stream():
            yield 'data: {"type":"chunk","text":"saved answer"}\n\n'
            yield 'data: [DONE]\n\n'
        return StreamingResponse(stream(), media_type='text/event-stream')
    monkeypatch.setattr(orchestrator, 'orchestrate_chat', AsyncMock(side_effect=answer))
    yield app, factory
    asyncio.run(engine.dispose())
    auth.session_secret.cache_clear()


def open_browser(app, base='http://testserver'):
    client = TestClient(app, base_url=base)
    response = client.get('/session')
    assert response.status_code == 200
    client.headers['X-Session-CSRF'] = response.json()['csrf_token']
    return client, response


def start(client):
    response = client.post('/start-conversation', json={'title': 'test'})
    assert response.status_code == 200, response.text
    return response.json()['conversation_id']


def test_cookie_flags_and_restart_identity(server):
    app, _ = server
    browser, response = open_browser(app)
    cookie = response.headers['set-cookie']
    assert 'HttpOnly' in cookie and 'SameSite=lax' in cookie and 'Max-Age=15552000' in cookie
    assert response.headers['cache-control'] == 'no-store'
    assert 'Secure' not in cookie  # normal localhost HTTP still works
    cid = start(browser)
    before = browser.cookies.get(auth.COOKIE_NAME)
    auth.session_secret.cache_clear()  # simulate process reload using the same key
    fresh_client, _ = open_browser(app)
    assert fresh_client.get('/conversations').json() == []
    restored = TestClient(app)
    restored.cookies.update(browser.cookies)
    assert restored.get('/session').json()['csrf_token'] == browser.headers['X-Session-CSRF']
    assert 'set-cookie' not in restored.get('/session').headers
    assert restored.get(f'/history/{cid}').status_code == 200
    assert restored.cookies.get(auth.COOKIE_NAME) == before


def test_https_cookie_is_secure(server):
    _, response = open_browser(server[0], 'https://testserver')
    assert 'Secure' in response.headers['set-cookie']


def test_persistent_generated_key(tmp_path, monkeypatch):
    monkeypatch.delenv('SESSION_SECRET', raising=False)
    monkeypatch.setattr(auth, 'ROOT', tmp_path)
    auth.session_secret.cache_clear()
    first = auth.session_secret()
    auth.session_secret.cache_clear()
    assert auth.session_secret() == first
    assert (tmp_path / 'data/session_secret.key').is_file()
    auth.session_secret.cache_clear()


def test_no_header_login_or_missing_csrf(server):
    app, _ = server
    visitor = TestClient(app, headers={'X-User-ID': 'someone-else'})
    assert visitor.get('/conversations').status_code == 401
    assert visitor.post('/mock-login', json={'username': 'old-user'}).status_code == 404
    owner, _ = open_browser(app)
    owner.headers.pop('X-Session-CSRF')
    assert owner.post('/start-conversation', json={}).status_code == 403
    owner.headers['X-Session-CSRF'] = 'wrong'
    assert owner.post('/start-conversation', json={}).status_code == 403


def test_expired_tampered_and_non_ascii_cookie_replaced(server):
    browser, _ = open_browser(server[0])
    token = browser.cookies.get(auth.COOKIE_NAME)
    issued, nonce, mac = token.split('.')
    assert auth.read_session(token + '0') is None
    assert auth.read_session(issued + '.' + nonce + '.é') is None
    old = str(int(issued) - auth.SESSION_SECONDS - 10) + '.' + nonce
    assert auth.read_session(old + '.' + auth.signature(old)) is None
    forged = TestClient(server[0], headers={'Cookie': auth.COOKIE_NAME + '=' + token + '0'})
    assert forged.get('/conversations').status_code == 401
    response = forged.get('/session')
    assert response.status_code == 200 and 'set-cookie' in response.headers


def test_conversation_and_message_isolation_before_side_effects(server, monkeypatch):
    app, _ = server
    a, _ = open_browser(app)
    b, _ = open_browser(app)
    cid = start(a)
    response = a.post(f'/chat/{cid}', json={'user_prompt': 'hello'})
    assert response.status_code == 200 and '[DONE]' in response.text
    history = a.get(f'/history/{cid}').json()
    assert len(history['messages']) == 2
    mid = history['messages'][-1]['id']
    assert a.post(f'/feedback/{mid}', json={'rating': 1}).status_code == 200
    generation = orchestrator.orchestrate_chat.call_count
    assert b.get('/conversations').json() == []
    for path, payload in [(f'/chat/{cid}', {'user_prompt': 'intrude'}),
                          (f'/chat/{cid}/action/form', None),
                          (f'/chat/{cid}/action/mcp', {'offer_message_id': mid}),
                          (f'/feedback/{mid}', {'rating': 0})]:
        assert b.post(path, json=payload).status_code == 404
    assert b.get(f'/history/{cid}').status_code == 404
    b.headers['X-User-ID'] = auth.read_session(a.cookies.get(auth.COOKIE_NAME))
    assert b.get(f'/history/{cid}').status_code == 404
    assert orchestrator.orchestrate_chat.call_count == generation
    unchanged = a.get(f'/history/{cid}').json()
    assert len(unchanged['messages']) == 2
    assert a.post(f'/feedback/{mid}', json={'rating': 9}).status_code == 422


def test_mcp_owned_offer_and_duplicate_approval(server, monkeypatch):
    app, factory = server
    owner, _ = open_browser(app)
    cid = start(owner)
    async def offer():
        async with factory() as db:
            conv = await routes.get_conversation(db, cid)
            message = await add_message(db, cid, 'assistant', 'search?', ['RAG_DOMAIN:demo'])
            conv.checklist = {'_domain_state': {'name': 'demo'}, '_mcp_pending': {
                'domain': 'demo', 'query': 'lookup', 'message_id': message.id}}
            conv.needs_search_permission = True
            await db.commit()
            return message.id
    mid = asyncio.run(offer())
    monkeypatch.setattr('rag.domain_config.load_domain_config', lambda *a: SimpleNamespace(name='demo'))
    monkeypatch.setattr(routes, 'load_settings', lambda: {})
    monkeypatch.setattr(routes, 'search_enabled', lambda *a: True)
    search = AsyncMock(return_value=('online answer', ['WEB_SOURCE:{"title":"official","url":"https://example.com"}']))
    monkeypatch.setattr(routes, 'execute_mcp_search', search)
    stranger, _ = open_browser(app)
    assert stranger.post(f'/chat/{cid}/action/mcp', json={'offer_message_id': mid}).status_code == 404
    assert search.call_count == 0
    assert owner.post(f'/chat/{cid}/action/mcp', json={'offer_message_id': mid + 1}).status_code == 409
    response = owner.post(f'/chat/{cid}/action/mcp', json={'offer_message_id': mid})
    assert response.status_code == 200, response.text
    assert response.json()['text'] == 'online answer'
    assert owner.post(f'/chat/{cid}/action/mcp', json={'offer_message_id': mid}).status_code == 409
    assert search.call_count == 1
    assert owner.get(f'/history/{cid}').json()['needs_search_permission'] is False


def test_form_action_still_works_for_owner(server, monkeypatch):
    owner, _ = open_browser(server[0])
    cid = start(owner)
    monkeypatch.setattr('rag.domain_config.load_domain_config', lambda *a: SimpleNamespace(
        name='demo', primary_entity='topic', profile={'allow_forms': True}))
    # Match an existing public form; no actual document is generated.
    from services import forms_matcher
    monkeypatch.setattr(forms_matcher, 'find_matching_form', lambda **k: {
        'display_name': 'Form', 'filename': 'form.docx'})
    response = owner.post(f'/chat/{cid}/action/form')
    assert response.status_code == 200
    assert '/static/forms/form.docx' in response.json()['text']


def test_admin_endpoint_requires_separate_secret(server, monkeypatch):
    browser, _ = open_browser(server[0])
    assert browser.get('/admin/failed-logs').status_code == 403
    assert browser.get('/admin/failed-logs', headers={'X-Admin-Key': 'bad'}).status_code == 403
    assert browser.get('/admin/failed-logs', headers={'X-Admin-Key': 'admin-test-password'}).status_code == 200
    assert browser.get('/admin/failed-logs?limit=999999', headers={'X-Admin-Key': 'admin-test-password'}).status_code == 422
    monkeypatch.delenv('ADMIN_PASSWORD')
    assert browser.get('/admin/failed-logs').status_code == 503


def test_qa_runner_uses_real_cookie_jar_over_http(server):
    import socket
    import threading
    import time
    import uvicorn
    from qa_suite import Client
    app, _ = server
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    host, port = sock.getsockname()
    instance = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level='error'))
    thread = threading.Thread(target=instance.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not instance.started and time.monotonic() < deadline:
            time.sleep(.01)
        assert instance.started
        first = Client(f'http://{host}:{port}', 'qa-run', interval=0)
        cid = first.request('/start-conversation', {'title': 'QA smoke'})[0]['conversation_id']
        result = first.request(f'/chat/{cid}', {'user_prompt': 'hello'}, stream=True)[0]
        assert result['done'] and result['answer'] == 'saved answer'
        assert len(first.request(f'/history/{cid}')[0]['messages']) == 2
        second = Client(f'http://{host}:{port}', 'other-run', interval=0)
        second.request('/session')
        assert second.request('/conversations')[0] == []
        with pytest.raises(RuntimeError, match='HTTP 404'):
            second.request(f'/history/{cid}')
    finally:
        instance.should_exit = True
        thread.join(timeout=5)
        sock.close()


def test_cross_site_session_bootstrap_rejected(server):
    response = TestClient(server[0]).get('/session', headers={'Sec-Fetch-Site': 'cross-site'})
    assert response.status_code == 403 and 'set-cookie' not in response.headers
