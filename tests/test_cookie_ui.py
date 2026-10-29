"""Browser-session behavior and safe rendering contracts, without real inference."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backend')]


def test_real_entrypoint_has_no_login_and_serves_local_sanitizer(monkeypatch):
    from fastapi.testclient import TestClient
    import main
    monkeypatch.setenv('SESSION_SECRET', 'test-secret-' * 8)
    from services.session_auth import session_secret
    session_secret.cache_clear()
    client = TestClient(main.app)
    page = client.get('/chat')
    assert page.status_code == 200
    assert 'id="login-overlay"' not in page.text
    assert '/mock-login' not in page.text and 'X-User-ID' not in page.text
    assert 'window.onload=async' in page.text
    assert 'DOMPurify.sanitize' in page.text
    assert client.get('/static/vendor/purify.min.js').status_code == 200
    assert client.get('/session').status_code == 200
    assert client.post('/mock-login', json={'username': 'test'}).status_code == 404
    assert client.get('/health').headers['cache-control'] == 'no-store'
    session_secret.cache_clear()


def test_admin_gate_before_data_and_password_removal(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import rag.admin_access as access
    # Running the actual dashboard: when blocked, domain/database work must not run.
    monkeypatch.delenv('ADMIN_PASSWORD', raising=False)
    monkeypatch.setattr(access, 'admin_password', lambda: '')
    app = AppTest.from_file(str(ROOT / 'backend/admin_dashboard.py'), default_timeout=20).run()
    assert not app.exception
    assert not app.tabs and len(app.error) == 1
    monkeypatch.undo()
    monkeypatch.setenv('ADMIN_PASSWORD', 'private-admin-test')
    app = AppTest.from_file(str(ROOT / 'backend/admin_dashboard.py'), default_timeout=20).run()
    assert not app.exception and not app.tabs
    assert len(app.text_input) == 1
    app.text_input[0].set_value('wrong')
    app.button[0].click().run()
    assert not app.tabs and len(app.error) == 1
    # Only the gate is under test; replace database-heavy dashboard content after login.
    wrapper = tmp_path / 'admin_gate.py'
    source = (ROOT / 'backend/admin_dashboard.py').read_text()
    # Retain the real gate verbatim, replace subsequent application work with a marker.
    source = source[:source.index('# --- DATA FETCHING ---')]
    source += "\nst.success('Authenticated dashboard reached')\n"
    wrapper.write_text(source.replace('PROJECT_ROOT = Path(__file__).resolve().parents[1]', 'PROJECT_ROOT = Path(' + repr(str(ROOT)) + ')'))
    app = AppTest.from_file(str(wrapper), default_timeout=20).run()
    app.text_input[0].set_value('private-admin-test')
    app.button[0].click().run()
    assert not app.exception and len(app.success) == 1
    assert not app.text_input
    app.sidebar.button[0].click().run()
    assert not app.exception and not app.success
    assert len(app.text_input) == 1
    assert app.text_input[0].value == ''
