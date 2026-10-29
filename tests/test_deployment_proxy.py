"""Optional real Caddy + HTTPS + SSE integration; uses a temporary internal CA."""
import asyncio
import json
import os
from pathlib import Path
import socket
import ssl
import subprocess
import threading
import time
import urllib.error
import urllib.request
from fastapi.responses import StreamingResponse
import pytest
import uvicorn
from test_browser_sessions import server
from services import chat_orchestrator
from services.db_service import add_message
from qa_suite import Client

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not os.getenv('CADDY_TEST_BINARY'),reason='Set CADDY_TEST_BINARY to run real local HTTPS proxy checks')
def test_real_https_cookie_streaming_and_private_routes(server,tmp_path,monkeypatch):
    app,_=server
    app.add_api_route('/health',lambda:{'ok':True})
    release=threading.Event()
    async def answer(cid,payload,db):
        await add_message(db,cid,'user',payload.user_prompt)
        await add_message(db,cid,'assistant','first second',['source'])
        async def stream():
            yield 'data: '+json.dumps({'type':'chunk','text':'first '})+'\n\n'
            await asyncio.to_thread(release.wait,3)
            yield 'data: '+json.dumps({'type':'chunk','text':'second'})+'\n\n'
            yield 'data: [DONE]\n\n'
        return StreamingResponse(stream(),media_type='text/event-stream')
    monkeypatch.setattr(chat_orchestrator,'orchestrate_chat',answer)
    backend=socket.socket();backend.bind(('127.0.0.1',0));backend_port=backend.getsockname()[1]
    spare=socket.socket();spare.bind(('127.0.0.1',0));proxy_port=spare.getsockname()[1];spare.close()
    uv=uvicorn.Server(uvicorn.Config(app,log_level='error'))
    thread=threading.Thread(target=uv.run,kwargs={'sockets':[backend]},daemon=True);thread.start()
    template=(ROOT/'deployment/Caddyfile').read_text()
    config='{\n admin off\n skip_install_trust\n auto_https disable_redirects\n}\n'+template.replace('{$PUBLIC_HOST} {',f'https://localhost:{proxy_port} {{\n tls internal').replace('127.0.0.1:8000',f'127.0.0.1:{backend_port}')
    path=tmp_path/'Caddyfile';path.write_text(config)
    environment=dict(os.environ,XDG_DATA_HOME=str(tmp_path/'data'),XDG_CONFIG_HOME=str(tmp_path/'config'))
    log=(tmp_path/'proxy.log').open('w')
    proxy=subprocess.Popen([os.environ['CADDY_TEST_BINARY'],'run','--config',str(path),'--adapter','caddyfile'],env=environment,stdout=log,stderr=subprocess.STDOUT)
    try:
        ca=tmp_path/'data/caddy/pki/authorities/local/root.crt'
        deadline=time.monotonic()+10
        while not ca.exists() and time.monotonic()<deadline:
            if proxy.poll() is not None:raise AssertionError((tmp_path/'proxy.log').read_text())
            time.sleep(.02)
        assert ca.exists()
        context=ssl.create_default_context(cafile=str(ca))
        base=f'https://localhost:{proxy_port}'
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),urllib.request.HTTPSHandler(context=context))
        while time.monotonic()<deadline:
            try:
                with opener.open(base+'/health',timeout=1) as response:
                    assert response.status==200
                    break
            except urllib.error.URLError:time.sleep(.02)
        else:raise AssertionError('Proxy did not become ready')
        client=Client(base,'proxy-test',interval=0,timeout=4)
        client.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),urllib.request.HTTPCookieProcessor(client.cookies),urllib.request.HTTPSHandler(context=context))
        session=client.request('/session')[0]
        cookie=next(c for c in client.cookies if c.name=='legal_session')
        assert cookie.secure and cookie.has_nonstandard_attr('HttpOnly')
        cid=client.request('/start-conversation',{'title':'HTTPS check'})[0]['conversation_id']
        request=urllib.request.Request(base+f'/chat/{cid}',data=json.dumps({'user_prompt':'test'}).encode(),headers={'Content-Type':'application/json','X-Session-CSRF':session['csrf_token']})
        with client.opener.open(request,timeout=2) as response:
            first=response.readline()
            assert b'first' in first  # A buffered stream would wait for the release event and time out.
            release.set()
            rest=response.read()
            assert b'second' in rest and b'[DONE]' in rest
        assert len(client.request(f'/history/{cid}')[0]['messages'])==2
        with pytest.raises(urllib.error.HTTPError) as error:
            opener.open(base+'/admin/operations')
        assert error.value.code==404
    finally:
        release.set()
        proxy.terminate();proxy.wait(timeout=5);log.close()
        uv.should_exit=True;thread.join(timeout=5);backend.close()
