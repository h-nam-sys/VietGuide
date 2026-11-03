"""Actual admin forms with temporary domain configs and intercepted writes."""
import ast
import copy
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from test_admin_navigation import dashboard, go, ROOT


@pytest.fixture
def domains(dashboard, tmp_path, monkeypatch):
    app, calls, save = dashboard
    import domain_admin_ui as ui
    from rag.domain_config import DomainConfig, validate_profile, validate_runtime
    configs = []
    for name in ('alpha','beta'):
        profile = validate_profile({
            'domain_name':name,'primary_entity':'topic','schema_mapping':{'topic':'Topic','content':'Content'},
            'embedding_fields':['topic'],'excluded_fields':[],'description':name+' scope',
            'minimum_score':0.01,'candidate_k':20,'result_k':5,'exact_field_output':False,
            'record_selection_prompt':'Preserve this selector instruction',
            'modes':{
                'strict':{'system_prompt':'Strict instruction','temperature':0.15,'top_p':0.88,
                          'max_tokens':600,'user_template':'{context}\n{query}'},
                'friendly':{'system_prompt':'Friendly instruction','direct_prompt':'Friendly conversation',
                            'temperature':0.65,'top_p':0.9,'max_tokens':700,
                            'user_template':'{context}\n{query}'}
            }})
        db = tmp_path/(name+'.db')
        db.write_bytes(b'')
        vectors = tmp_path/(name+'-vectors')
        vectors.mkdir()
        configs.append(DomainConfig(name, profile, db, vectors, 'records','id','a'*32,
                                    'revision-'+name,tmp_path/name/'current.json'))
    monkeypatch.setattr(ui, 'list_domain_configs', lambda **kw: configs)
    monkeypatch.setattr(ui, 'domain_status', lambda name: 'enabled')
    monkeypatch.setattr(ui, 'ROOT_DIR', tmp_path)
    monkeypatch.setattr(ui, 'generations', lambda name: [])
    monkeypatch.setattr(ui, 'deletion_targets', lambda name: [])
    published = []
    def publish(name, value):
        validate_runtime(value, next(c.profile for c in configs if c.name == name))
        published.append((name, copy.deepcopy(value)))
    monkeypatch.setattr(ui, 'publish_settings', publish)
    go(app, 'Dữ liệu và lĩnh vực')
    return app, configs, published, ui


def button(app, label):
    return next(item for item in app.button if item.label == label)


def test_one_domain_selection_is_shared_across_actions(domains):
    app, _, _, _ = domains
    app.selectbox(key='admin_domain').set_value('beta').run()
    for action in ('Cập nhật dữ liệu','Cấu hình trả lời','Phiên bản','Quản lý'):
        app.radio(key='domain_action').set_value(action).run()
        assert not app.exception
        assert app.selectbox(key='admin_domain').value == 'beta'
        assert sum(item.label == 'Lĩnh vực' for item in app.selectbox) == 1


def test_only_one_mode_form_and_no_raw_json_editor(domains):
    app, _, published, _ = domains
    app.radio(key='domain_action').set_value('Cấu hình trả lời').run()
    assert not app.exception
    assert sum(item.label == 'Hướng dẫn trả lời từ dữ liệu' for item in app.text_area) == 1
    assert not any('JSON' in item.label for item in app.text_area)
    assert len(app.number_input) == 6
    app.selectbox(key='edit_mode_alpha').set_value('friendly').run()
    assert not app.exception
    assert sum(item.label == 'Hướng dẫn trò chuyện thông thường' for item in app.text_area) == 1
    assert not published


def test_saving_strict_preserves_friendly_and_unexposed_settings(domains):
    app, configs, published, _ = domains
    original = copy.deepcopy(configs[0].profile)
    app.radio(key='domain_action').set_value('Cấu hình trả lời').run()
    next(item for item in app.number_input if item.label == 'Ngưỡng điểm tra cứu').set_value(0.05)
    button(app,'Lưu cấu hình').click().run()
    assert not app.exception and len(published) == 1
    name, value = published[0]
    assert name == 'alpha'
    assert value['modes']['strict']['minimum_score'] == 0.05
    assert value['modes']['friendly'] == original['modes']['friendly']
    assert value['record_selection_prompt'] == original['record_selection_prompt']


def test_invalid_candidate_result_limits_do_not_publish(domains):
    app, _, published, _ = domains
    app.radio(key='domain_action').set_value('Cấu hình trả lời').run()
    next(item for item in app.number_input if item.label == 'Số bản ghi kết quả').set_value(50)
    button(app,'Lưu cấu hình').click().run()
    assert not app.exception and app.error
    assert not published


def test_enable_disable_action_targets_selected_domain(domains, monkeypatch):
    app, _, _, ui = domains
    toggle = Mock()
    monkeypatch.setattr(ui, 'set_domain_enabled', toggle)
    app.selectbox(key='admin_domain').set_value('beta').run()
    app.radio(key='domain_action').set_value('Quản lý').run()
    button(app,'Tắt lĩnh vực').click().run()
    assert not app.exception
    toggle.assert_called_once_with('beta', False)


def test_mcp_keeps_connection_configuration_and_domain_drafts(domains, monkeypatch):
    app, configs, _, _ = domains
    import mcp_admin_ui as ui
    from rag.mcp_config import DEFAULTS, validate_settings
    original = copy.deepcopy(DEFAULTS)
    original.update(transport='http',server_url='http://localhost:8877/mcp',
                    token_env='MY_TOKEN',tool_name='search_custom',query_argument='q',
                    count_argument='limit',extra_arguments={'kind':'search'},
                    domain_rules={c.name:{'enabled':True,'allowed_hosts':['example.org'],
                                          'use_location':False} for c in configs})
    monkeypatch.setattr(ui, 'load_settings', lambda: copy.deepcopy(original))
    monkeypatch.setattr(ui, 'list_domain_configs', lambda: configs)
    saved = []
    monkeypatch.setattr(ui, 'save_settings', lambda value: saved.append(validate_settings(value)))
    go(app,'Tìm kiếm trực tuyến')
    assert not any(item.label == 'Kiểm tra kết nối' for item in app.button)
    assert not any(item.label == 'Kết nối MCP' for item in app.selectbox)
    app.text_area(key='mcp_sites_alpha').set_value('alpha.example.org').run()
    app.selectbox(key='mcp_rule_domain').set_value('beta').run()
    app.text_area(key='mcp_sites_beta').set_value('beta.example.org').run()
    button(app,'Lưu cấu hình MCP').click().run()
    assert not app.exception and len(saved) == 1
    for key in ('transport','server_url','token_env','tool_name','query_argument','count_argument','extra_arguments'):
        assert saved[0][key] == original[key]
    assert saved[0]['domain_rules']['alpha']['allowed_hosts'] == ['alpha.example.org']
    assert saved[0]['domain_rules']['beta']['allowed_hosts'] == ['beta.example.org']


def test_mcp_real_search_test_does_not_save_configuration(domains, monkeypatch):
    app, configs, _, _ = domains
    import mcp_admin_ui as ui
    from rag.mcp_config import DEFAULTS
    settings = copy.deepcopy(DEFAULTS)
    settings['domain_rules'] = {'alpha':{'enabled':True,'allowed_hosts':[],'use_location':False}}
    monkeypatch.setattr(ui, 'load_settings', lambda: copy.deepcopy(settings))
    monkeypatch.setattr(ui, 'list_domain_configs', lambda: configs)
    monkeypatch.setattr(ui, 'load_domain_config', lambda name: configs[0])
    retrieve, save = AsyncMock(return_value=[]), Mock()
    monkeypatch.setattr(ui, 'retrieve_external', retrieve)
    monkeypatch.setattr(ui, 'save_settings', save)
    go(app,'Tìm kiếm trực tuyến')
    app.text_input(key='mcp_test_query').set_value('Test question').run()
    button(app,'Tra cứu thử').click().run()
    assert not app.exception
    retrieve.assert_awaited_once()
    save.assert_not_called()


def test_record_selection_shows_answer_and_source_from_same_row():
    tree = ast.parse((ROOT/'backend/admin_dashboard.py').read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {'display_value','record_table'}]
    markdown, code = [], []
    st = NS(caption=lambda *a,**kw:None, dataframe=lambda *a,**kw:NS(selection=NS(rows=[1])),
            markdown=lambda value:markdown.append(value), write=lambda *a:None,
            expander=lambda *a,**kw:nullcontext(),code=lambda value,**kw:code.append(value))
    namespace = {'st':st,'pd':pd,'json':json}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'admin-records','exec'),namespace)
    frame = pd.DataFrame([{'q':'first','a':'answer one','s':'source one'},
                          {'q':'second','a':'answer two','s':'source two'}])
    namespace['record_table'](frame,['q'],{},question='q',answer='a',source='s',key='test')
    assert 'answer two' in markdown and 'answer one' not in markdown
    assert code == ['source two']
