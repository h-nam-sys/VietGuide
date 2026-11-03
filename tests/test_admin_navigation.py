"""Exercise the actual Streamlit dashboard with temporary records and mocked APIs."""
import copy
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'backend')]


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    monkeypatch.setenv('ADMIN_PASSWORD', 'admin-layout-test')
    monkeypatch.setenv('CHAT_DB_PATH', str(tmp_path/'chat.db'))
    monkeypatch.setenv('FEEDBACK_DIR', str(tmp_path))
    with sqlite3.connect(tmp_path/'chat.db') as db:
        db.executescript('''
            CREATE TABLE conversations(id INTEGER, created_at TEXT);
            CREATE TABLE messages(id INTEGER, role TEXT, created_at TEXT);
            CREATE TABLE audit_logs(created_at TEXT, conversation_id INTEGER,
                user_query TEXT, llm_final_response TEXT, used_online_search INTEGER,
                retrieved_local_context TEXT, rag_score REAL, failure_type TEXT);
            INSERT INTO conversations VALUES(1,'2026-11-03');
            INSERT INTO messages VALUES(1,'user','2026-11-03');
            INSERT INTO audit_logs VALUES('2026-11-03',1,'Library question',
                'First complete answer',0,'Full original source',0.9,'SUCCESS');
            INSERT INTO audit_logs VALUES('2026-11-02',1,'Second question',
                'Second complete answer',0,'Second original source',0.8,'SUCCESS');
        ''')
    (tmp_path/'all_feedback_dataset.jsonl').write_text(json.dumps({
        'timestamp':'2026-11-03','rating':1,'input':'Feedback question',
        'output':'Feedback complete answer','rag_context':['Feedback source']})+'\n')
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        value = ({'readiness':{'ready':True},'backend':{'queues':{'chat':{'active':0,'waiting':0,'rejected':0}}}}
                 if url.endswith('/operations') else [{'created_at':'2026-11-03',
                 'failure_type':'LOW_SCORE','rag_score':0.01,'user_query':'Review question',
                 'llm_response':'Review complete answer'}])
        return NS(status_code=200, json=lambda: value, raise_for_status=lambda: None)
    monkeypatch.setattr(requests, 'get', get)
    import domain_admin_ui as data
    monkeypatch.setattr(data, 'list_domain_configs', lambda **kw: [])
    monkeypatch.setattr(data, 'list_jobs', lambda: [])
    import mcp_admin_ui as mcp
    from rag.mcp_config import DEFAULTS
    monkeypatch.setattr(mcp, 'load_settings', lambda: copy.deepcopy(DEFAULTS))
    monkeypatch.setattr(mcp, 'list_domain_configs', lambda: [])
    save = Mock()
    monkeypatch.setattr(mcp, 'save_settings', save)
    st.cache_data.clear()
    app = AppTest.from_file(str(ROOT/'backend/admin_dashboard.py'), default_timeout=20)
    app.session_state['_admin_authenticated'] = 'admin-layout-test'
    app.run()
    assert not app.exception
    yield app, calls, save
    st.cache_data.clear()


def go(app, section):
    app.radio(key='admin_section').set_value(section).run()
    assert not app.exception


def test_overview_is_focused_and_only_fetches_operations(dashboard):
    app, calls, _ = dashboard
    assert app.title[0].value == 'Bảng quản trị'
    assert not app.sidebar.radio
    assert app.radio(key='admin_section').proto.horizontal
    assert not app.tabs
    assert all(url.endswith('/operations') for url in calls)
    assert len(app.metric) == 7


def test_activity_list_and_selected_answer_keep_full_sources(dashboard):
    app, calls, _ = dashboard
    calls.clear()
    go(app, 'Hội thoại và đánh giá')
    assert not calls
    assert list(app.dataframe[0].value.columns) == ['created_at','user_query','used_online_search','rag_score','failure_type']
    assert len(app.dataframe) == 1
    assert not app.selectbox
    assert any(item.value == 'First complete answer' for item in app.markdown)
    assert any(item.value == 'Full original source' for item in app.code)


def test_feedback_keeps_counts_answer_and_source(dashboard):
    app, _, _ = dashboard
    go(app, 'Hội thoại và đánh giá')
    app.radio(key='admin_conversation_view').set_value('Đánh giá').run()
    assert not app.exception
    assert app.metric[0].value == '1'
    assert any(item.value == 'Feedback complete answer' for item in app.markdown)
    assert any('Feedback source' in item.value for item in app.code)


def test_review_list_fetches_only_its_endpoint(dashboard):
    app, calls, _ = dashboard
    go(app, 'Hội thoại và đánh giá')
    calls.clear()
    app.radio(key='admin_conversation_view').set_value('Cần xem lại').run()
    assert not app.exception
    assert calls and all(url.endswith('/failed-logs') for url in calls)
    assert any(item.value == 'Review complete answer' for item in app.markdown)


def test_all_existing_data_actions_remain_on_data_page(dashboard):
    app, calls, _ = dashboard
    calls.clear()
    go(app, 'Dữ liệu và lĩnh vực')
    assert not app.tabs
    assert app.selectbox(key='admin_domain').options == ['Tạo lĩnh vực mới']
    assert app.get('file_uploader')
    assert not calls


def test_online_search_settings_do_not_save_on_navigation(dashboard):
    app, calls, save = dashboard
    calls.clear()
    go(app, 'Tìm kiếm trực tuyến')
    assert app.toggle(key='mcp_enabled').value is False
    assert any(button.label == 'Lưu cấu hình MCP' for button in app.button)
    save.assert_not_called()
    assert not calls


def test_logout_still_blocks_all_admin_content(dashboard):
    app, calls, _ = dashboard
    calls.clear()
    next(button for button in app.button if button.label == 'Đóng phiên quản trị').click().run()
    assert not app.exception
    assert not app.radio
    assert len(app.text_input) == 1
    assert app.text_input[0].value == ''
    assert not calls
