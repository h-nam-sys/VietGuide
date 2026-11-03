import streamlit as st
import pandas as pd
import sqlite3
import json
import os
import requests

# Load project modules and .env for the domain admin, regardless of launch directory.
import sys
from pathlib import Path
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / 'backend/.env')
load_dotenv(PROJECT_ROOT / '.env')

# --- CONFIG ---
DB_PATH = os.getenv("CHAT_DB_PATH", "./legal_llm.db")
FEEDBACK_FILE = str(Path(os.getenv("FEEDBACK_DIR", ".")) / "all_feedback_dataset.jsonl")
API_URL = "http://localhost:8000" # Update this if your FastAPI runs on a different port/host

st.set_page_config(page_title="Bảng quản trị", layout="wide")

# Admin access is separate from anonymous chat. An unset password fails closed.
from rag.admin_access import admin_password, valid_admin_password
if not admin_password():
    st.error('Đặt ADMIN_PASSWORD trong .env rồi khởi động lại dịch vụ để mở quản trị.')
    st.stop()
if not valid_admin_password(st.session_state.get('_admin_authenticated', '')):
    st.title('Quản trị hệ thống')
    with st.form('admin_access', clear_on_submit=True):
        password = st.text_input('Mật khẩu quản trị', type='password')
        submitted = st.form_submit_button('Mở quản trị')
    if submitted:
        if valid_admin_password(password):
            st.session_state['_admin_authenticated'] = password
            st.rerun()
        else:
            st.error('Mật khẩu không đúng.')
    st.stop()
heading, refresh, logout = st.columns([6, 1, 1])
heading.title('Bảng quản trị')
if refresh.button('Làm mới', key='admin_refresh'):
    st.cache_data.clear()
    st.rerun()
if logout.button('Đóng phiên quản trị'):
    st.session_state.pop('_admin_authenticated', None)
    st.rerun()
section = st.radio('Khu vực',
    ['Tổng quan', 'Dữ liệu và lĩnh vực', 'Hội thoại và đánh giá', 'Tìm kiếm trực tuyến'],
    horizontal=True, key='admin_section', label_visibility='collapsed')
st.divider()


# --- DATA FETCHING ---
from rag.timing import timed, span, emit, TimedLock, trace_headers, TimingMiddleware, trace_id

@st.cache_data(ttl=60) # Cache for 60 seconds to prevent DB locking
@timed('backend.admin_dashboard.load_db_data')
def load_db_data(query):
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=60)
@timed('backend.admin_dashboard.load_feedback_data')
def load_feedback_data():
    if not os.path.exists(FEEDBACK_FILE):
        return pd.DataFrame()

    data = []
    with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except:
                pass
    return pd.DataFrame(data)

def display_value(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 'Chưa có thông tin.'
    return json.dumps(value, ensure_ascii=False, indent=2) if isinstance(value, (list, dict)) else str(value)


def record_table(frame, columns, labels, *, question, answer, source=None, key):
    """Show one compact list with the selected answer and source below it."""
    columns = [name for name in columns if name in frame.columns]
    st.caption(f'{len(frame)} lượt. Chọn một dòng để xem câu trả lời và nguồn đầy đủ.')
    event = st.dataframe(frame[columns], column_config={name: labels.get(name, name) for name in columns},
        hide_index=True, use_container_width=True, height=320, key=key + '_table',
        on_select='rerun', selection_mode='single-row')
    rows = event.selection.rows
    selected = rows[0] if rows and 0 <= rows[0] < len(frame) else 0
    row = frame.iloc[selected]
    metadata = [labels.get(name, name) + ': ' + display_value(row[name])
                for name in ('created_at', 'timestamp', 'conversation_id', 'rag_score', 'failure_type')
                if name in frame.columns]
    if metadata:
        st.caption(' · '.join(metadata))
    st.markdown('**Câu hỏi**')
    st.write(display_value(row.get(question)))
    st.markdown('**Câu trả lời**')
    st.markdown(display_value(row.get(answer)))
    if source and source in frame.columns:
        with st.expander('Nguồn và dữ liệu cung cấp', expanded=False):
            st.code(display_value(row.get(source)), language=None)


def overview_page():
    st.subheader("Chỉ số hệ thống")

    conv_df = load_db_data("SELECT id, created_at FROM conversations")
    msg_df = load_db_data("SELECT id, role, created_at FROM messages")
    audit_df = load_db_data("SELECT id, used_online_search FROM audit_logs")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng cuộc trò chuyện", len(conv_df))
    col2.metric("Tổng tin nhắn", len(msg_df))

    mcp_count = len(audit_df[audit_df['used_online_search'] == 1]) if not audit_df.empty else 0
    col3.metric("Lượt dùng nguồn ngoài", mcp_count)

    user_msgs = len(msg_df[msg_df['role'] == 'user']) if not msg_df.empty else 0
    col4.metric("Câu hỏi từ người dùng", user_msgs)

    st.divider()
    st.subheader('Vận hành dịch vụ')
    try:
        response = requests.get(f'{API_URL}/admin/operations', headers={'X-Admin-Key': admin_password()}, timeout=12)
        response.raise_for_status()
        status = response.json()
        ready = status['readiness']['ready']
        (st.success if ready else st.warning)('Sẵn sàng trả lời' if ready else 'Có thành phần chưa sẵn sàng')
        queue = status['backend']['queues'].get('chat', {})
        a, b, c = st.columns(3)
        a.metric('Đang xử lý', queue.get('active', 0))
        b.metric('Đang chờ', queue.get('waiting', 0))
        c.metric('Lượt từ chối do giới hạn', queue.get('rejected', 0))
        with st.expander('Thời gian, bộ nhớ và trạng thái thành phần'):
            st.caption('Số liệu trong tiến trình hiện tại; thời gian chữ trả lời đầu tiên không phải token đầu tiên của model. RSS không phải VRAM.')
            st.json(status)
    except Exception:
        st.warning('Chưa lấy được trạng thái backend. Kiểm tra dịch vụ và ADMIN_PASSWORD.')

    if not msg_df.empty:
        st.subheader("Lưu lượng truy cập theo ngày")
        msg_df['created_at'] = pd.to_datetime(msg_df['created_at'])
        daily_counts = msg_df.groupby(msg_df['created_at'].dt.date).size()
        st.bar_chart(daily_counts)


def activity_page():
    st.subheader("Nhật ký tra cứu và trả lời")
    st.markdown("Xem câu trả lời và dữ liệu được sử dụng.")

    try:
        logs = load_db_data("SELECT created_at, conversation_id, user_query, llm_final_response, used_online_search, retrieved_local_context, rag_score, failure_type FROM audit_logs ORDER BY created_at DESC LIMIT 100")
    except:
         logs = load_db_data("SELECT created_at, conversation_id, user_query, llm_final_response, used_online_search, retrieved_local_context FROM audit_logs ORDER BY created_at DESC LIMIT 100")


    if not logs.empty:
        logs['used_online_search'] = logs['used_online_search'].apply(lambda x: "Nguồn ngoài" if x == 1 else "Dữ liệu nội bộ")

        column_cfg = {
                "created_at": "Thời gian",
                "conversation_id": "Mã cuộc trò chuyện",
                "user_query": "Câu hỏi",
                "llm_final_response": "Câu trả lời",
                "used_online_search": "Loại nguồn",
                "retrieved_local_context": "Dữ liệu trích xuất"
            }

        if 'rag_score' in logs.columns:
            column_cfg["rag_score"] = "Điểm RAG"
        if 'failure_type' in logs.columns:
             column_cfg["failure_type"] = "Trạng thái"

        record_table(logs, ['created_at', 'user_query', 'used_online_search', 'rag_score', 'failure_type'],
            column_cfg, question='user_query', answer='llm_final_response',
            source='retrieved_local_context', key='activity')

    else:
        st.info("Chưa có nhật ký hoạt động.")


def feedback_page():
    st.subheader("Đánh giá từ người dùng")
    st.markdown("Xem phản hồi của người dùng về chất lượng câu trả lời.")

    feedback_df = load_feedback_data()

    if not feedback_df.empty:
        feedback_df['rating_display'] = feedback_df['rating'].apply(lambda x: "Phù hợp" if x == 1 else "Không phù hợp")

        good_count = len(feedback_df[feedback_df['rating'] == 1])
        bad_count = len(feedback_df[feedback_df['rating'] == 0])

        col1, col2 = st.columns(2)
        col1.metric("Lượt đánh giá phù hợp", good_count)
        col2.metric("Lượt đánh giá không phù hợp", bad_count)

        record_table(feedback_df, ['timestamp', 'rating_display', 'input'],
            {'timestamp':'Thời gian', 'rating_display':'Đánh giá', 'input':'Câu hỏi',
             'output':'Câu trả lời', 'rag_context':'Dữ liệu cung cấp'},
            question='input', answer='output', source='rag_context', key='feedback')

    else:
        st.info("Chưa có đánh giá nào từ người dùng.")


def review_page():
    st.subheader("Truy vấn cần xem lại")
    st.markdown("Các câu hỏi có điểm tra cứu thấp hoặc được ghi nhận cần xử lý thêm. Dùng danh sách này để xem lại và bổ sung dữ liệu.")


    try:
        response = requests.get(f"{API_URL}/admin/failed-logs", headers={"X-Admin-Key": admin_password()}, timeout=10)
        if response.status_code == 200:
            failed_logs = response.json()
            if failed_logs:
                df_failed = pd.DataFrame(failed_logs)

                record_table(df_failed, ['created_at', 'failure_type', 'rag_score', 'user_query'],
                    {'created_at':'Thời gian', 'failure_type':'Loại lỗi', 'rag_score':'Điểm RAG',
                     'user_query':'Câu hỏi', 'llm_response':'Câu trả lời'},
                    question='user_query', answer='llm_response', key='review')

            else:
                st.success("Chưa có truy vấn cần xem lại.")
        else:
            st.error(f"Lỗi API: {response.status_code}. Kiểm tra máy chủ backend đã chạy chưa.")
    except requests.exceptions.RequestException:
        st.error("Không thể kết nối đến máy chủ backend tại localhost:8000.")


if section == 'Tổng quan':
    st.caption('Tình trạng dịch vụ và hoạt động của trợ lý.')
    overview_page()
elif section == 'Dữ liệu và lĩnh vực':
    from domain_admin_ui import render_domain_admin
    render_domain_admin()
elif section == 'Hội thoại và đánh giá':
    st.subheader('Hội thoại và đánh giá')
    view = st.radio('Nội dung', ['Nhật ký', 'Đánh giá', 'Cần xem lại'],
                    horizontal=True, key='admin_conversation_view')
    if view == 'Nhật ký':
        activity_page()
    elif view == 'Đánh giá':
        feedback_page()
    else:
        review_page()
else:
    from mcp_admin_ui import render_mcp_admin
    render_mcp_admin()
