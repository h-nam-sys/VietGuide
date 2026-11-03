"""Simple MCP administration with advanced connection settings collapsed."""
import asyncio
import html
import copy
import json
import os
import streamlit as st
from rag.domain_config import list_domain_configs, load_domain_config
from rag.mcp_config import load_settings, save_settings, validate_settings, rule_for, MCPConfigError
from backend.services.mcp_service import retrieve_external


from rag.timing import timed, span, emit, TimedLock, trace_headers, TimingMiddleware, trace_id

@timed('backend.mcp_admin_ui.render_mcp_admin')
def render_mcp_admin():
    st.subheader('Tìm kiếm trực tuyến (MCP)')
    st.caption('Chỉ tra cứu khi dữ liệu nội bộ không đủ và người dùng chọn Có. Cấu hình được áp dụng cho lượt tra cứu tiếp theo.')
    try:
        current = load_settings()
        configs = list_domain_configs()
    except Exception as exc:
        st.error(str(exc))
        return
    draft = copy.deepcopy(current)
    revision = json.dumps(current, sort_keys=True, ensure_ascii=False)
    if st.session_state.get('_mcp_saved_revision') != revision:
        st.session_state['_mcp_saved_revision'] = revision
        st.session_state['_mcp_rule_draft'] = copy.deepcopy(current['domain_rules'])
    draft['domain_rules'] = copy.deepcopy(st.session_state.get('_mcp_rule_draft', current['domain_rules']))
    draft['enabled'] = st.toggle('Cho phép tìm kiếm trực tuyến', value=current['enabled'], key='mcp_enabled')
    by_name = {c.name: c for c in configs}
    selected = st.multiselect('Lĩnh vực được phép tìm kiếm', options=list(by_name),
        default=[c.name for c in configs if rule_for(c, current)['enabled']], key='mcp_domains')
    # Preserve disabled domain rules; rendering this page never writes configuration.
    for name, config in by_name.items():
        draft['domain_rules'][name] = {**rule_for(config, draft), 'enabled': name in selected}
    if selected:
        domain = st.selectbox('Website tìm kiếm của lĩnh vực', selected, key='mcp_rule_domain')
        rule = draft['domain_rules'][domain]
        sites = st.text_area('Chỉ tìm trên các website này (mỗi dòng một tên miền)',
            value='\n'.join(rule['allowed_hosts']), key='mcp_sites_' + domain,
            placeholder='dichvucong.gov.vn')
        rule['allowed_hosts'] = [s.strip() for s in sites.splitlines() if s.strip()]
        st.caption('Để trống để tìm trên web công khai. Ví dụ lĩnh vực trường học: website chính thức của trường.')
        rule['use_location'] = st.checkbox('Thêm địa điểm của cuộc trò chuyện vào câu hỏi tìm kiếm',
            value=rule['use_location'], key='mcp_location_' + domain)
    # Retain edited website/location rules when the admin switches the domain selector.
    for name in by_name:
        if 'mcp_sites_' + name in st.session_state:
            draft['domain_rules'][name]['allowed_hosts'] = [x.strip() for x in st.session_state['mcp_sites_' + name].splitlines() if x.strip()]
        if 'mcp_location_' + name in st.session_state:
            draft['domain_rules'][name]['use_location'] = st.session_state['mcp_location_' + name]
    st.session_state['_mcp_rule_draft'] = copy.deepcopy(draft['domain_rules'])
    transport = current['transport']
    st.caption('Nguồn tìm kiếm: ' + ('Gemini / Google Search' if transport == 'bundled' else 'Máy chủ MCP đã cấu hình'))
    if transport == 'http':
        st.caption('Máy chủ MCP: ' + current['server_url'])
        st.caption('Thông số kết nối được cấu hình trong data/mcp_settings.json.')
    if transport == 'bundled':
        st.caption('Khóa Gemini: ' + ('Đã cấu hình' if os.getenv('GEMINI_API_KEY') else 'Chưa cấu hình'))
    with st.expander('Tham số tìm kiếm', expanded=False):
        if transport == 'bundled':
            draft['gemini_model'] = st.text_input('Model Gemini', value=current['gemini_model'], key='mcp_gemini_model',
                help='Model dùng cho tìm kiếm có dẫn nguồn; không thay đổi model Qwen cục bộ.')
        draft['result_limit'] = st.number_input('Số nguồn tối đa', 1, 10, value=current['result_limit'], key='mcp_limit',
            help='Giới hạn số nguồn được đưa vào bước tổng hợp câu trả lời.')
        draft['timeout_seconds'] = st.number_input('Thời gian chờ tìm kiếm (giây)', 5, 180,
            value=current['timeout_seconds'], key='mcp_timeout', help='Giới hạn thời gian chờ máy chủ tìm kiếm.')
        draft['max_tokens'] = st.number_input('Giới hạn token trả lời', 100, 2000,
            value=current['max_tokens'], key='mcp_tokens')
        draft['answer_prompt'] = st.text_area('Hướng dẫn trả lời từ nguồn ngoài',
            value=current['answer_prompt'], height=140, key='mcp_prompt')
    if st.button('Lưu cấu hình MCP', type='primary'):
        try:
            save_settings(draft)
            st.success('Đã lưu cấu hình tìm kiếm.')
        except Exception as exc:
            st.error(str(exc))
    with st.expander('Kiểm tra bằng một câu hỏi', expanded=False):
        st.caption('Thực hiện tìm kiếm thật bằng cấu hình đang hiển thị. Không lưu thay đổi và có thể phát sinh phí API.')
        test_domain = st.selectbox('Lĩnh vực thử', selected, key='mcp_test_domain') if selected else None
        question = st.text_input('Câu hỏi thử', key='mcp_test_query')
        if st.button('Tra cứu thử', disabled=not test_domain or not question.strip()):
            try:
                clean = validate_settings(draft)
                # An explicit admin test is authorized even before the global toggle is published.
                clean['enabled'] = True
                with st.spinner('Đang tra cứu qua MCP…'):
                    records = asyncio.run(retrieve_external(question, load_domain_config(test_domain), settings=clean))
                if not records:
                    st.warning('Không có đoạn trích phù hợp từ các website được phép.')
                if getattr(records, 'search_suggestions', ''):
                    doc = html.escape(records.search_suggestions, quote=True)
                    st.markdown(f'<iframe title="Google Search" sandbox="allow-popups allow-popups-to-escape-sandbox" srcdoc="{doc}" style="width:100%;height:150px;border:0"></iframe>', unsafe_allow_html=True)
                for r in records:
                    st.write(r['title'])
                    st.code(r['url'], language=None)
                    if r.get('evidence_kind') == 'gemini_grounded_summary':
                        st.caption('Tóm tắt Gemini có dẫn nguồn (không phải trích nguyên văn website)')
                    st.text(r['snippet'])
            except Exception as exc:
                st.error(str(exc))
