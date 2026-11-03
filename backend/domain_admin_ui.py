"""Local Streamlit domain onboarding and configuration, alongside existing monitoring."""
import hashlib
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import re
import pandas as pd
import streamlit as st

from rag.domain_config import (ROOT_DIR, RUNTIME_KEYS, RAGConfigurationError, atomic_json,
    list_domain_configs, load_domain_config, mode_settings, read_json, validate_profile, domain_status)
from rag.domain_management import set_domain_enabled, delete_domain, deletion_targets
from rag.domain_onboarding import (propose_profile, publish_settings, generations,
    activate_generation, GENERIC_EVIDENCE_PROMPT)
from rag.ingest_pipeline import read_input, prepare_records
from rag.ingestion_jobs import start_job, list_jobs


from rag.timing import timed, span, emit, TimedLock, trace_headers, TimingMiddleware, trace_id

def dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def ui_error(exc):
    st.error(str(exc))


@timed('backend.domain_admin_ui.upload_page')
def upload_page(domain=None, *, create_new=False):
    st.subheader('Thêm dữ liệu hoặc tạo lĩnh vực')
    st.caption('Tải tệp → tạo cấu hình → kiểm tra → nhập dữ liệu. '
               'Nhập dữ liệu cho lĩnh vực đã có sẽ thay thế toàn bộ dữ liệu của lĩnh vực đó sau khi hoàn tất.')
    configs = list_domain_configs(include_inactive=True)
    existing = {c.name: c for c in configs}
    choice = ('Tạo lĩnh vực mới' if create_new else domain if domain is not None
              else st.selectbox('Lĩnh vực đích', ['Tạo lĩnh vực mới'] + list(existing), key='upload_domain'))
    upload_context = 'create' if create_new else 'existing_' + domain if domain else 'standalone'
    new = choice == 'Tạo lĩnh vực mới'
    domain_id = st.text_input('Mã lĩnh vực mới (để trống để Gemini đề xuất)',
                            placeholder='university / it_support / hr', key='new_domain_id') if new else choice
    file = st.file_uploader('Tệp dữ liệu', type=['xlsx', 'xlsm', 'csv', 'tsv', 'json', 'jsonl'], key='domain_upload_' + upload_context)
    sheet = 0
    if file and Path(file.name).suffix.lower() in {'.xlsx', '.xlsm'}:
        sheet_text = st.text_input('Trang tính Excel (số thứ tự hoặc tên)', value='0', key='upload_sheet')
        sheet = int(sheet_text) if sheet_text.isdecimal() else sheet_text
    if not file:
        st.info('Chọn tệp dữ liệu để bắt đầu.')
        return
    digest = hashlib.sha256(file.getvalue()).hexdigest()
    suffix = Path(file.name).suffix.lower()
    path = ROOT_DIR / 'data/admin_uploads' / (digest + suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(file.getvalue())
    token = digest + '|' + choice + '|' + domain_id + '|' + str(sheet)
    if st.session_state.get('upload_token') != token:
        st.session_state['upload_token'] = token
        st.session_state.pop('profile_editor', None)
        st.session_state.pop('mapping_editor', None)
        if not new:
            st.session_state['profile_editor'] = dump(existing[choice].profile)
    try:
        df = read_input(path, sheet)
        st.write(f'{len(df)} dòng • {len(df.columns)} cột')
        st.dataframe(df.head(8), hide_index=True, use_container_width=True)
    except Exception as exc:
        ui_error(exc)
        return
    if new:
        st.caption('Gemini nhận tên cột và tối đa 8 dòng mẫu giới hạn độ dài; mô tả này phục vụ nhận diện lĩnh vực, '
                   'không thay thế dữ liệu gốc. Khóa API sử dụng cấu hình trên máy chủ.')
        if st.button('Tạo cấu hình bằng Gemini', type='primary'):
            try:
                with st.spinner('Gemini đang phân tích cấu trúc và phạm vi dữ liệu…'):
                    profile = propose_profile(df, domain_id.strip() or None)
                if profile['domain_name'] in existing:
                    raise RAGConfigurationError('Mã Gemini đề xuất đã tồn tại. Nhập mã mới rồi tạo lại.')
                st.session_state['profile_editor'] = dump(profile)
                st.session_state.pop('mapping_editor', None)
                st.success('Đã tạo bản nháp. Kiểm tra cấu hình bên dưới rồi nhập dữ liệu.')
            except Exception as exc:
                ui_error(exc)
    else:
        st.info('Dùng lại cấu hình hiện tại; không gọi Gemini và không tự đổi ánh xạ cột.')
    with st.expander('Nhập cấu hình có sẵn', expanded=False):
        st.caption('Tùy chọn cho cấu hình đã chuẩn bị trước; không cần dùng khi cấu hình được đề xuất bằng Gemini.')
        supplied = st.file_uploader('Tệp cấu hình JSON', type=['json'], key='import_profile_' + upload_context)
        if supplied:
            supplied_token = token + hashlib.sha256(supplied.getvalue()).hexdigest()
            if st.session_state.get('imported_profile_token') != supplied_token:
                try:
                    st.session_state['profile_editor'] = dump(validate_profile(json.loads(supplied.getvalue().decode('utf-8-sig'))))
                    st.session_state['imported_profile_token'] = supplied_token
                except Exception as exc:
                    ui_error(exc)
    if 'profile_editor' not in st.session_state:
        return
    text = st.session_state['profile_editor']
    try:
        profile = validate_profile(json.loads(text))
        if not new and profile['domain_name'] != choice:
            raise RAGConfigurationError('Không được đổi mã lĩnh vực hiện tại trong cấu hình.')
        if new and profile['domain_name'] in existing:
            raise RAGConfigurationError('Lĩnh vực đã tồn tại. Chọn nó ở Lĩnh vực đích để cập nhật.')
        if new and domain_id.strip() and profile['domain_name'] != domain_id.strip():
            raise RAGConfigurationError('Mã trong cấu hình phải khớp mã lĩnh vực mới đã nhập.')
        st.markdown('**Kiểm tra cấu hình dữ liệu**')
        description_key = 'upload_desc_' + hashlib.sha256(text.encode()).hexdigest()[:12]
        profile['description'] = st.text_area('Mô tả phạm vi dữ liệu', profile.get('description', ''), key=description_key)
        st.caption('Mã lĩnh vực: ' + profile['domain_name'])
        with st.expander('Ánh xạ cột dữ liệu', expanded=True):
            mapping = pd.DataFrame([{'Mã trường': k, 'Cột dữ liệu': v} for k, v in profile['schema_mapping'].items()])
            mapping_digest = hashlib.sha256(dump(profile['schema_mapping']).encode()).hexdigest()[:10]
            edited = st.data_editor(mapping, hide_index=True, num_rows='dynamic', use_container_width=True,
                                   key='map_' + token[:12] + mapping_digest,
                                   column_config={'Cột dữ liệu': st.column_config.SelectboxColumn(options=[str(c) for c in df.columns], required=True)})
            entries = [(str(row['Mã trường']).strip(), str(row['Cột dữ liệu']).strip()) for _, row in edited.iterrows()]
            if len({k for k, _ in entries}) != len(entries):
                raise RAGConfigurationError('Mã trường không được trùng nhau.')
            profile['schema_mapping'] = dict(entries)
        keys = list(profile['schema_mapping'])
        if not keys:
            raise RAGConfigurationError('Cần ít nhất một trường dữ liệu.')
        primary = profile['primary_entity']
        profile['primary_entity'] = st.selectbox('Cột tiêu đề hoặc chủ đề', keys,
            index=keys.index(primary) if primary in keys else 0,
            format_func=lambda k: profile['schema_mapping'][k], key='primary_' + token[:12] + mapping_digest)
        profile['embedding_fields'] = st.multiselect('Các cột dùng để tìm kiếm', keys,
            default=[k for k in profile.get('embedding_fields', [profile['primary_entity']]) if k in keys],
            format_func=lambda k: profile['schema_mapping'][k],
            key='embed_' + token[:12] + mapping_digest,
            help='Các cột dùng để tạo chỉ mục tìm kiếm. Thay đổi lựa chọn này cần nhập lại dữ liệu.')
        with st.expander('Cấu hình nâng cao: tìm kiếm và nguồn', expanded=False):
            profile['embedding_prefix'] = st.text_input('Tiền tố văn bản tìm kiếm', value=profile.get('embedding_prefix', ''),
                                                      key='prefix_' + token[:12])
            profile['excluded_fields'] = st.multiselect('Các trường ẩn trong nguồn dữ liệu', keys,
                default=[k for k in profile.get('excluded_fields', ['timestamp', 'email', 'email_address']) if k in keys],
                key='excluded_' + token[:12] + mapping_digest,
                help='Giữ trong dữ liệu gốc nhưng không đưa vào ngữ cảnh trả lời và nguồn hiển thị.')
        validate_profile(profile)
        _, report = prepare_records(df, profile)
        st.success(f"Hợp lệ: {report['indexed_rows']} bản ghi sẵn sàng để nhập.")
        with st.expander('Chi tiết xác thực và tải cấu hình', expanded=False):
            st.json(report)
            st.download_button('Tải cấu hình đã kiểm tra', dump(profile), file_name=profile['domain_name'] + '_profile.json')
        if st.button('Nhập dữ liệu', type='primary'):
            job_id = start_job(path, profile, sheet=sheet)
            st.success(f'Đã tạo tác vụ {job_id}. Xem tab Tiến độ và phiên bản để theo dõi. Dữ liệu cũ tiếp tục được sử dụng đến khi hoàn tất.')
    except Exception as exc:
        ui_error(exc)


@timed('backend.domain_admin_ui.settings_page')
def settings_page(domain=None):
    st.subheader('Cấu hình trả lời')
    configs = {c.name: c for c in list_domain_configs(include_inactive=True)}
    if not configs:
        st.info('Chưa có lĩnh vực để cấu hình.')
        return
    name = domain or st.selectbox('Lĩnh vực', list(configs), key='settings_domain')
    config = configs[name]
    mode = st.selectbox('Chế độ cần chỉnh', ['strict', 'friendly'],
        format_func=lambda value: 'Strict' if value == 'strict' else 'Friendly',
        key='edit_mode_' + name,
        help='Chỉ lưu thay đổi cho chế độ đang chọn. Chế độ còn lại được giữ nguyên.')
    runtime = {key: copy.deepcopy(value) for key, value in config.profile.items() if key in RUNTIME_KEYS}
    modes = copy.deepcopy(config.profile.get('modes', {}))
    baseline = {}
    prompt_path = ROOT_DIR / 'backend/prompts.json'
    if config.primary_entity == 'procedure_name' and prompt_path.exists():
        baseline = read_json(prompt_path).get('baseline_qa', {})
    settings = {**baseline, **mode_settings(config, mode)}
    settings = {key: value for key, value in settings.items() if key in {
        'system_prompt', 'user_template', 'direct_prompt', 'temperature', 'top_p',
        'max_tokens', 'minimum_score', 'candidate_k', 'result_k'}}
    form_key = name + config.revision[:10] + '_' + mode
    st.caption('Có hiệu lực sau khi lưu. Thay đổi cột hoặc ánh xạ dữ liệu phải thực hiện ở mục Cập nhật dữ liệu.')
    with st.form('settings_' + form_key):
        description = st.text_area('Mô tả phạm vi dữ liệu', config.profile.get('description', ''),
            key='desc_' + form_key, help='Dùng để nhận diện lĩnh vực; không thay thế nội dung các bản ghi.')
        settings['system_prompt'] = st.text_area('Hướng dẫn trả lời từ dữ liệu',
            settings.get('system_prompt', GENERIC_EVIDENCE_PROMPT), height=150, key='prompt_' + form_key)
        if mode == 'friendly':
            settings['direct_prompt'] = st.text_area('Hướng dẫn trò chuyện thông thường',
                settings.get('direct_prompt', 'Bạn là trợ lý thân thiện. Trả lời ngắn gọn. Không tuyên bố đã tra cứu nguồn khi chưa tra cứu.'),
                key='direct_' + form_key, help='Áp dụng cho phần trò chuyện không dùng dữ liệu tra cứu.')
        defaults = st.selectbox('Chế độ mặc định', ['strict', 'friendly'],
            index=0 if config.profile.get('default_mode', os.getenv('CHAT_MODE', 'strict')) == 'strict' else 1,
            key='default_' + form_key, help='Dùng khi lượt hỏi không chỉ định chế độ.')
        with st.expander('Tham số trả lời và tra cứu', expanded=False):
            st.caption('Các giá trị được nạp từ cấu hình hiện tại; mở biểu mẫu không thay đổi cấu hình.')
            settings['temperature'] = st.number_input('Temperature', 0.0, 2.0,
                float(settings.get('temperature', 0.1 if mode == 'strict' else 0.3)), step=0.05,
                key='temp_' + form_key, help='Mức ngẫu nhiên khi sinh câu trả lời. Giá trị thấp làm đầu ra ổn định hơn, không bảo đảm tính chính xác.')
            settings['top_p'] = st.number_input('Top-p', 0.01, 1.0,
                float(settings.get('top_p', 0.95)), step=0.01, key='top_' + form_key,
                help='Giới hạn tập token được lấy mẫu khi sinh câu trả lời.')
            settings['max_tokens'] = st.number_input('Giới hạn token trả lời', 1, 4096,
                int(settings.get('max_tokens', 800)), key='tokens_' + form_key,
                help='Giới hạn độ dài đầu ra; quá thấp có thể làm câu trả lời bị cắt.')
            settings['minimum_score'] = st.number_input('Ngưỡng điểm tra cứu', 0.0, 1.0,
                float(settings.get('minimum_score', config.minimum_score)), step=0.01, format='%.4f',
                key='score_' + form_key, help='Ngưỡng loại ứng viên có điểm thấp. Tăng ngưỡng có thể loại cả bản ghi phù hợp.')
            settings['candidate_k'] = st.number_input('Số bản ghi ứng viên', 1, 200,
                int(settings.get('candidate_k', config.retrieval_option('candidate_k', 'RAG_CANDIDATE_K', '20'))),
                key='candidate_' + form_key, help='Số ứng viên phục vụ tìm kiếm và xếp hạng. Tăng giá trị có thể tăng thời gian xử lý.')
            settings['result_k'] = st.number_input('Số bản ghi kết quả', 1, 200,
                int(settings.get('result_k', config.retrieval_option('result_k', 'RAG_RESULT_K', '5'))),
                key='result_' + form_key, help='Giới hạn bản ghi giữ lại cho bước trả lời; không được vượt số ứng viên.')
            settings['user_template'] = st.text_area('Mẫu câu hỏi gửi mô hình',
                settings.get('user_template', 'RECORDS:\n{context}\nQUESTION:\n{query}'),
                key='template_' + form_key, help='Giữ các biến {context} và {query} để truyền dữ liệu và câu hỏi.')
        with st.expander('Nhận diện lĩnh vực và xử lý thiếu dữ liệu', expanded=False):
            examples = st.text_area('Câu hỏi minh họa cho lĩnh vực',
                '\n'.join(config.profile.get('routing_examples', [])), key='examples_' + form_key,
                help='Mỗi dòng một câu, dùng làm ví dụ cho bộ nhận diện lĩnh vực.')
            fallback = st.text_area('Thông báo khi thiếu dữ liệu',
                config.profile.get('fallback_message', ''), key='fallback_' + form_key)
            exact = st.checkbox('Trả trực tiếp giá trị trường khi phù hợp',
                value=config.profile.get('exact_field_output', os.getenv('RAG_EXACT_FIELD_OUTPUT', '0') == '1' and config.primary_entity == 'procedure_name'),
                key='exact_' + form_key,
                help='Chỉ áp dụng khi câu hỏi và dữ liệu đáp ứng điều kiện trả nguyên văn. Câu hỏi khác vẫn dùng mô hình.')
        if st.form_submit_button('Lưu cấu hình', type='primary'):
            try:
                modes[mode] = settings
                runtime.update(description=description,
                    routing_examples=[line.strip() for line in examples.splitlines() if line.strip()],
                    default_mode=defaults, exact_field_output=exact, modes=modes)
                if fallback:
                    runtime['fallback_message'] = fallback
                else:
                    runtime.pop('fallback_message', None)
                publish_settings(name, runtime)
                st.success('Đã lưu cấu hình.')
                st.rerun()
            except Exception as exc:
                ui_error(exc)
    with st.expander('Khôi phục cài đặt', expanded=False):
        confirmed = st.checkbox('Xác nhận bỏ các thay đổi cài đặt và dùng lại cấu hình của phiên bản dữ liệu',
                                key='reset_confirm_' + name)
        if st.button('Khôi phục cài đặt mặc định', disabled=not confirmed):
            (ROOT_DIR / 'data/domain_settings' / f'{name}.json').unlink(missing_ok=True)
            st.rerun()


@timed('backend.domain_admin_ui.status_page')
def status_page(domain=None):
    st.subheader('Tiến độ nhập dữ liệu')
    status_labels = {'queued': 'Đang chờ', 'running': 'Đang xử lý', 'failed': 'Thất bại', 'completed': 'Hoàn tất', 'succeeded': 'Hoàn tất'}
    stage_labels = {'Waiting for ingestion slot': 'Chờ tác vụ nhập dữ liệu trước hoàn tất', 'Waiting for worker': 'Chờ tiến trình xử lý', 'Worker stopped before completion': 'Tiến trình dừng trước khi hoàn tất', 'Could not start worker': 'Không thể khởi chạy tiến trình', 'Reading and validating uploaded file': 'Đọc và xác thực tệp', 'Writing SQL snapshot': 'Lưu phiên bản dữ liệu', 'Building and verifying vector index': 'Tạo và kiểm tra chỉ mục tìm kiếm', 'Activating completed snapshot': 'Kích hoạt phiên bản dữ liệu', 'Domain is ready for chat': 'Sẵn sàng sử dụng', 'Ingestion failed': 'Nhập dữ liệu thất bại'}
    jobs = [job for job in list_jobs() if domain is None or job['domain'] == domain]
    if not jobs:
        st.info('Chưa có tác vụ nhập dữ liệu.')
    for job in jobs[:20]:
        st.write(job['domain'] + ' — ' + status_labels.get(job['status'], job['status']))
        with st.expander(f"{job['domain']} • {status_labels.get(job['status'], job['status'])} • {stage_labels.get(job['stage'], job['stage'])}", expanded=False):
            st.write('Mã tác vụ:', job['id'])
            st.write('Cập nhật:', job['updated_at'])
            if job.get('error'):
                st.error('Tiến trình xử lý đã dừng ngoài dự kiến. Kiểm tra nhật ký bên dưới.' if job['stage'] == 'Worker stopped before completion' else job['error'])
            st.json(job.get('report', {}))
            log_path = ROOT_DIR / 'data/admin_jobs' / (job['id'] + '.log')
            if log_path.exists():
                st.code(log_path.read_text(encoding='utf-8', errors='replace')[-12000:], language='text')
    st.subheader('Dữ liệu đang sử dụng và khôi phục phiên bản')
    configs = list_domain_configs(include_inactive=True)
    if not configs:
        return
    name = domain or st.selectbox('Lĩnh vực', [c.name for c in configs], key='rollback_domain')
    snapshots = generations(name)
    if not snapshots:
        st.caption('Lĩnh vực này dùng dữ liệu cũ; phiên bản sẽ được tạo sau lần nhập dữ liệu tiếp theo.')
        return
    generation = st.selectbox('Phiên bản', [g['generation'] for g in snapshots],
        format_func=lambda value: datetime.fromtimestamp(next(g for g in snapshots if g['generation'] == value)['modified']).strftime('%d/%m/%Y %H:%M') + ' · ' + value[:8] + (' (đang dùng)' if next(g for g in snapshots if g['generation'] == value)['active'] else ''),
        key='version_' + name)
    active = next(g for g in snapshots if g['generation'] == generation)['active']
    if st.button('Dùng phiên bản này', disabled=active):
        try:
            activate_generation(name, generation)
            st.success('Đã kích hoạt phiên bản dữ liệu đã chọn.')
            st.rerun()
        except Exception as exc:
            ui_error(exc)


@timed('backend.domain_admin_ui.management_page')
def management_page(domain=None):
    st.subheader('Bật, tắt hoặc xóa lĩnh vực')
    configs = {c.name: c for c in list_domain_configs(include_inactive=True)}
    if not configs:
        st.info('Chưa có lĩnh vực để quản lý.')
        return
    name = domain or st.selectbox('Lĩnh vực', list(configs), key='manage_domain')
    enabled = domain_status(name) == 'enabled'
    st.write('Trạng thái: ' + ('Đang bật' if enabled else 'Đã tắt'))
    st.caption('Tắt lĩnh vực giữ nguyên dữ liệu và cấu hình, nhưng bỏ lĩnh vực khỏi danh sách trò chuyện '
               'và nhận diện tự động. Các lượt trò chuyện tiếp theo sẽ chọn lại lĩnh vực.')
    default_path = ROOT_DIR / 'data/domain_profile.json'
    protected = default_path.exists() and read_json(default_path).get('domain_name') == name
    if protected:
        st.info('Lĩnh vực mặc định hiện tại được bảo vệ: không thể tắt hoặc xóa.')
    if st.button('Tắt lĩnh vực' if enabled else 'Bật lĩnh vực',
                 key='toggle_domain_' + name, disabled=protected and enabled):
        try:
            set_domain_enabled(name, not enabled)
            st.session_state['domain_management_notice'] = 'Đã tắt lĩnh vực.' if enabled else 'Đã bật lĩnh vực.'
            st.rerun()
        except Exception as exc:
            ui_error(exc)
    with st.expander('Xóa lĩnh vực', expanded=False):
        st.caption('Xóa khỏi hệ thống hoạt động và chuyển cấu hình cùng toàn bộ phiên bản SQL/chỉ mục '
                   'của lĩnh vực vào bản lưu. Không xóa lịch sử trò chuyện hoặc tệp tải lên dùng chung. '
                   'Không thể xóa khi đang nhập dữ liệu. Khi tạo lại, dùng mã lĩnh vực mới.')
        if not protected:
            try:
                paths = deletion_targets(name)
                st.write('Các mục sẽ được chuyển vào bản lưu:')
                st.code('\n'.join(str(p.relative_to(ROOT_DIR)) for p in paths), language='text')
            except Exception as exc:
                ui_error(exc)
                return
        confirmation = st.text_input('Nhập mã lĩnh vực để xác nhận xóa', key='delete_confirm_' + name)
        if st.button('Xóa lĩnh vực đã chọn', key='delete_domain_' + name,
                     disabled=protected or confirmation != name):
            try:
                archive = delete_domain(name, confirmation)
                st.session_state['domain_management_notice'] = 'Đã xóa lĩnh vực. Bản lưu: ' + str(archive)
                for key in ('upload_domain', 'settings_domain', 'rollback_domain', 'manage_domain'):
                    st.session_state.pop(key, None)
                st.rerun()
            except Exception as exc:
                ui_error(exc)


@timed('backend.domain_admin_ui.render_domain_admin')
def render_domain_admin():
    st.header('Dữ liệu và lĩnh vực')
    notice = st.session_state.pop('domain_management_notice', None)
    if notice:
        st.success(notice)
    configs = {config.name: config for config in list_domain_configs(include_inactive=True)}
    names = list(configs) + ['Tạo lĩnh vực mới']
    selected = st.selectbox('Lĩnh vực', names, key='admin_domain')
    if selected == 'Tạo lĩnh vực mới':
        upload_page(create_new=True)
        return
    config = configs[selected]
    enabled = domain_status(selected) == 'enabled'
    ready = config.db_path.is_file() and config.vector_dir.is_dir()
    st.caption(('Đang bật' if enabled else 'Đã tắt') + ' · ' + ('Dữ liệu sẵn sàng' if ready else 'Chưa có dữ liệu sẵn sàng'))
    action = st.radio('Thao tác', ['Cập nhật dữ liệu', 'Cấu hình trả lời', 'Phiên bản', 'Quản lý'],
                      horizontal=True, key='domain_action')
    if action == 'Cập nhật dữ liệu':
        upload_page(selected)
    elif action == 'Cấu hình trả lời':
        settings_page(selected)
    elif action == 'Phiên bản':
        status_page(selected)
    else:
        management_page(selected)
