"""Anonymous, signed browser sessions. Never trust a client-supplied user ID."""
import hashlib
import hmac
import os
from pathlib import Path
import re
import secrets
import time
from functools import lru_cache

from fastapi import HTTPException, Request, Response

COOKIE_NAME = 'legal_session'
SESSION_SECONDS = 180 * 24 * 60 * 60
ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def session_secret():
    configured = os.getenv('SESSION_SECRET', '')
    if configured:
        if len(configured.encode()) < 32:
            raise RuntimeError('SESSION_SECRET must contain at least 32 bytes')
        return configured.encode()
    # Persist across restarts; do not derive identity from IP, username or headers.
    path = ROOT / 'data/session_secret.key'
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, 'w', encoding='ascii') as file:
            file.write(secrets.token_hex(32))
    value = path.read_bytes().strip()
    if len(value) < 32:
        raise RuntimeError('Session secret is missing or incomplete; inspect data/session_secret.key')
    return value


def signature(value):
    return hmac.new(session_secret(), value.encode(), hashlib.sha256).hexdigest()


def read_session(token):
    if not isinstance(token, str) or len(token) > 200:
        return None
    parts = token.split('.')
    if len(parts) != 3:
        return None
    issued, nonce, mac = parts
    if not re.fullmatch(r'[0-9]{1,12}', issued) or not re.fullmatch(r'[0-9a-f]{64}', nonce):
        return None
    age = time.time() - int(issued)
    if not re.fullmatch(r'[0-9a-f]{64}', mac):
        return None
    if age < -60 or age >= SESSION_SECONDS or not hmac.compare_digest(signature(issued + '.' + nonce), mac):
        return None
    return 'anon:' + hashlib.sha256(nonce.encode()).hexdigest()


def csrf_token(token):
    return signature('csrf:' + token)


def bootstrap_session(request: Request, response: Response):
    if request.headers.get('Sec-Fetch-Site') == 'cross-site':
        raise HTTPException(403, 'Open the chat page to start a browser session')
    token = request.cookies.get(COOKIE_NAME)
    if not read_session(token):
        body = str(int(time.time())) + '.' + secrets.token_hex(32)
        token = body + '.' + signature(body)
        secure = os.getenv('SESSION_COOKIE_SECURE', 'auto').strip().lower()
        if secure not in {'auto', '0', '1', 'false', 'true'}:
            raise RuntimeError('SESSION_COOKIE_SECURE must be auto, 0 or 1')
        response.set_cookie(COOKIE_NAME, token, max_age=SESSION_SECONDS, path='/',
                            httponly=True, secure=request.url.scheme == 'https' if secure == 'auto' else secure in {'1', 'true'},
                            samesite='lax')
    response.headers['Cache-Control'] = 'no-store'
    return {'csrf_token': csrf_token(token)}


def get_current_user(request: Request):
    token = request.cookies.get(COOKIE_NAME)
    owner = read_session(token)
    if not owner:
        raise HTTPException(401, 'Phiên trình duyệt đã hết hạn. Vui lòng tải lại trang.')
    if request.method not in {'GET', 'HEAD', 'OPTIONS'}:
        supplied = request.headers.get('X-Session-CSRF', '')
        if not hmac.compare_digest(supplied.encode(), csrf_token(token).encode()):
            raise HTTPException(403, 'Yêu cầu không hợp lệ. Vui lòng tải lại trang.')
    return owner


class PrivateResponsesMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('path', '').startswith('/static/'):
            return await self.app(scope, receive, send)
        async def private_send(message):
            if message['type'] == 'http.response.start':
                message = dict(message)
                message['headers'] = [(k, v) for k, v in message.get('headers', [])
                                      if k.lower() != b'cache-control'] + [(b'cache-control', b'no-store')]
            await send(message)
        await self.app(scope, receive, private_send)
