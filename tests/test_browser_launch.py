"""Browser opening is once per host startup, after readiness."""
from unittest.mock import Mock

import run_services as launcher


def test_browser_opens_local_chat_in_default_browser(monkeypatch):
    monkeypatch.setenv('DISPLAY', ':test')
    monkeypatch.delenv('OPEN_CHAT_BROWSER', raising=False)
    browser = Mock(return_value=True)
    monkeypatch.setattr(launcher.webbrowser, 'open', browser)
    launcher.open_chat_browser()
    browser.assert_called_once_with('http://localhost:8000/chat', new=2)


def test_disabled_browser_does_not_open(monkeypatch):
    monkeypatch.setenv('OPEN_CHAT_BROWSER', '0')
    browser = Mock()
    monkeypatch.setattr(launcher.webbrowser, 'open', browser)
    launcher.open_chat_browser()
    browser.assert_not_called()


def test_browser_failure_does_not_stop_services(monkeypatch):
    monkeypatch.setenv('DISPLAY', ':test')
    monkeypatch.delenv('OPEN_CHAT_BROWSER', raising=False)
    monkeypatch.setattr(launcher.webbrowser, 'open', Mock(side_effect=OSError('No browser')))
    launcher.open_chat_browser()


def test_waits_for_readiness_then_opens_once(monkeypatch):
    stop = Mock()
    stop.wait.return_value = False
    stop.is_set.return_value = False
    healthy = Mock(side_effect=[False, False, True])
    browser = Mock()
    monkeypatch.setattr(launcher, 'STOP', stop)
    monkeypatch.setattr(launcher, 'healthy', healthy)
    monkeypatch.setattr(launcher, 'open_chat_browser', browser)
    launcher.open_chat_when_ready()
    assert healthy.call_count == 3
    healthy.assert_called_with(8000, '/readyz')
    browser.assert_called_once()


def test_shutdown_before_readiness_does_not_open(monkeypatch):
    stop = Mock()
    stop.wait.return_value = True
    browser = Mock()
    monkeypatch.setattr(launcher, 'STOP', stop)
    monkeypatch.setattr(launcher, 'open_chat_browser', browser)
    launcher.open_chat_when_ready()
    browser.assert_not_called()
