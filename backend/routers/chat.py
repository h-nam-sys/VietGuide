from fastapi import APIRouter, HTTPException, Depends, Header, Request, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func
from sqlalchemy.future import select
from pydantic import BaseModel, Field
from schemas import ChatRequest, ChatResponse, StartConversationRequest, ConversationSchema, MCPActionRequest
from services.llm_service import get_legal_response_stream
from services.mcp_service import execute_mcp_search, MCPSearchError
from rag.mcp_config import load_settings, search_enabled, MCPConfigError
from services.mcp_permission import authorize_pending, search_locks
from database import User, Message
from rate_limiter import limiter
from services.db_service import (
    create_conversation,
    get_conversation,
    get_conversation_history,
    get_user_conversations,
    add_message,
    update_conversation_title,
    update_conversation_state,
    update_domain_state,
    update_message_rating,
    log_interaction,
    Conversation
)
from database import get_db, AuditLog
import json
import random
import unicodedata
import re
import difflib
import ast

router = APIRouter(tags=["Chat"])


from rag.timing import timed, span, emit, TimedLock, trace_headers, TimingMiddleware, trace_id

@router.get('/domains')
@timed('backend.routers.chat.available_domains')
async def available_domains():
    from services.domain_router import ready_domains
    return [{'id': c.name, 'description': c.profile.get('description', ''),
             'default_mode': c.profile.get('default_mode', 'strict')} for c in ready_domains()]

class FeedbackRequest(BaseModel):
    rating: int = Field(ge=0, le=1)

from services.session_auth import get_current_user
from rag.admin_access import admin_password, valid_admin_password


async def owned_conversation(db, conversation_id, current_user_id):
    conversation = await get_conversation(db, conversation_id)
    if conversation:
        owner = (await db.execute(select(User.user_id).where(User.id == conversation.user_id))).scalar_one_or_none()
        if owner == current_user_id:
            return conversation
    raise HTTPException(404, 'Conversation not found')


def require_admin(x_admin_key: str = Header(default='')):
    if not admin_password():
        raise HTTPException(503, 'Configure ADMIN_PASSWORD before using administration')
    if not valid_admin_password(x_admin_key):
        raise HTTPException(403, 'Administrator access required')


def parse_sources(sources_data):
    if not sources_data:
        return []
    if isinstance(sources_data, list):
        return sources_data
    if isinstance(sources_data, str):
        try:
            parsed = json.loads(sources_data)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            try:
                parsed = ast.literal_eval(sources_data)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
    return []

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return re.sub(r'\s+', ' ', text).strip()

def fuzzy_match_any(query: str, normalized_vocab_list: list[str], threshold: float = 0.85) -> bool:
    norm_query = normalize_text(query)
    for vocab in normalized_vocab_list:
        if vocab in norm_query:
            return True
    query_words = norm_query.split()
    for vocab in normalized_vocab_list:
        vocab_word_count = len(vocab.split())
        if vocab_word_count == 0:
            continue
        for i in range(len(query_words) - vocab_word_count + 1):
            ngram = " ".join(query_words[i:i + vocab_word_count])
            similarity = difflib.SequenceMatcher(None, vocab, ngram).ratio()
            if similarity >= threshold:
                return True
    return False

RAW_FORM_TRIGGERS = ["giấy tờ", "làm giấy", "cần gì", "hồ sơ", "biểu mẫu", "tờ khai"]
NORMALIZED_FORM_TRIGGERS = [normalize_text(w) for w in RAW_FORM_TRIGGERS if normalize_text(w)]

def is_form_trigger(query: str) -> bool:
    return fuzzy_match_any(query, NORMALIZED_FORM_TRIGGERS)

# ==========================================
# ENDPOINTS
# ==========================================

@router.post("/start-conversation", response_model=dict)
@timed('backend.routers.chat.start_conversation')
async def start_conversation(
    request: StartConversationRequest,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    try:
        if request.domain and request.domain != 'auto':
            from services.domain_router import ready_domains
            if request.domain not in {c.name for c in ready_domains()}:
                raise HTTPException(400, 'Selected domain is unavailable or not ingested')
        conversation = await create_conversation(db, current_user_id, request.title)
        state = {'allow_switching': request.allow_switching}
        if request.domain and request.domain != 'auto':
            state.update({'name': request.domain, 'selection': 'manual', 'locked': True})
        await update_domain_state(db, conversation.id, state)
        return {
            "conversation_id": conversation.id,
            "title": conversation.title,
            "created_at": conversation.created_at.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")


@router.post("/chat/{conversation_id}")
@limiter.limit("10/minute")
@timed('backend.routers.chat.handle_chat')
async def handle_chat(
    request: Request,
    conversation_id: int,
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    await owned_conversation(db, conversation_id, current_user_id)
    from services.chat_orchestrator import orchestrate_chat
    return await orchestrate_chat(conversation_id, payload, db)



@router.post("/chat/{conversation_id}/action/mcp")
@timed('backend.routers.chat.action_mcp')
async def action_mcp(
    conversation_id: int,
    payload: MCPActionRequest | None = None,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    if payload is None:
        raise HTTPException(400, 'Vui lòng làm mới giao diện chat và chọn Có trên đề nghị tìm kiếm cụ thể.')
    async with search_locks.hold(conversation_id):
        conversation = await owned_conversation(db, conversation_id, current_user_id)
        owner = current_user_id
        pending = authorize_pending(conversation, current_user_id, owner, payload.offer_message_id)
        from rag.domain_config import load_domain_config, RAGConfigurationError
        try:
            config = load_domain_config(pending['domain'])
            settings = load_settings()
            if not search_enabled(config, settings):
                raise HTTPException(400, 'Tìm kiếm trực tuyến đang tắt cho lĩnh vực này.')
        except (RAGConfigurationError, MCPConfigError) as exc:
            raise HTTPException(400, 'Cấu hình lĩnh vực hoặc MCP không khả dụng. Kiểm tra admin.') from exc
        try:
            answer, sources = await execute_mcp_search(pending['query'], conversation.location,
                config=config, settings=settings)
        except MCPSearchError as exc:
            await log_interaction(db, conversation_id, pending['query'], str(exc), None, False, None, 'MCP_FAILED')
            raise HTTPException(503, str(exc)) from exc
        # Do not consume a newer offer if another chat turn arrived during the search.
        with span('db.refresh'):
            await db.refresh(conversation)
        if (conversation.checklist or {}).get('_mcp_pending') != pending:
            raise HTTPException(409, 'Câu hỏi đã thay đổi. Vui lòng dùng đề nghị tìm kiếm mới nhất.')
        saved_sources = sources + ['RAG_DOMAIN:' + config.name]
        await add_message(db, conversation_id, 'assistant', answer, saved_sources)
        checklist = dict(conversation.checklist or {})
        checklist.pop('_mcp_pending', None)
        conversation.checklist = checklist
        conversation.needs_search_permission = False
        with span('db.commit'):
            await db.commit()
        await log_interaction(db, conversation_id, pending['query'], answer,
            json.dumps(saved_sources, ensure_ascii=False), True, None,
            'MCP_SEARCH_EXECUTED' if sources else 'MCP_NO_RESULTS')
        return {'text': answer, 'sources': saved_sources}


@router.post("/chat/{conversation_id}/action/form")
@timed('backend.routers.chat.action_form')
async def action_form(
    conversation_id: int,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    conversation = await owned_conversation(db, conversation_id, current_user_id)
    from rag.domain_config import load_domain_config, RAGConfigurationError
    name = (conversation.checklist or {}).get('_domain_state', {}).get('name')
    try:
        config = load_domain_config(name)
    except RAGConfigurationError as exc:
        raise HTTPException(400, 'Domain is unavailable; send a new message to select another domain') from exc
    if not config.profile.get('allow_forms', config.primary_entity == 'procedure_name'):
        raise HTTPException(400, 'Forms are not configured for this domain')

    context_messages = await get_conversation_history(db, conversation_id)

    last_proc_title = conversation.current_procedure or ""
    last_user_msg = ""

    for msg in reversed(context_messages):
        if msg.role == "user" and not last_user_msg:
            last_user_msg = msg.content
            break

    try:
        from services.forms_matcher import find_matching_form
        query = f"{last_proc_title} {last_user_msg}".strip() if last_proc_title else last_user_msg
        matched_form = find_matching_form(procedure_title=last_proc_title, query=query)
        if matched_form:
            link_msg = (
                f"Đây là biểu mẫu bạn cần: \n\n"
                f"📄 **[{matched_form['display_name']} (.docx)](/static/forms/{matched_form['filename']})**"
            )
        else:
            link_msg = "Xin lỗi, hiện tại tôi chưa có sẵn file Word cho biểu mẫu này."
    except ImportError:
        link_msg = "Xin lỗi, module biểu mẫu chưa được cấu hình."

    await add_message(db, conversation_id, "assistant", link_msg, [])

    return {"text": link_msg, "sources": []}


@router.get("/conversations", response_model=list[dict])
@timed('backend.routers.chat.get_user_chats')
async def get_user_chats(
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    try:
        conversations = await get_user_conversations(db, current_user_id)
        return [
            {
                "id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat() if conv.created_at else "",
                "message_count": 0
            }
            for conv in conversations
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching conversations: {str(e)}")


@router.get("/history/{conversation_id}", response_model=ConversationSchema)
@timed('backend.routers.chat.get_chat_history')
async def get_chat_history(
    conversation_id: int,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    try:
        conversation = await owned_conversation(db, conversation_id, current_user_id)

        messages = await get_conversation_history(db, conversation_id)

        return ConversationSchema(**{
            "id": conversation.id,
            "title": conversation.title,
            "location": conversation.location,
            "current_procedure": conversation.current_procedure,
            "checklist": conversation.checklist or {},
            "needs_search_permission": conversation.needs_search_permission,
            "created_at": conversation.created_at.isoformat() if conversation.created_at else "",
            "updated_at": conversation.updated_at.isoformat() if conversation.updated_at else "",
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "sources": parse_sources(msg.sources),
                    "created_at": msg.created_at.isoformat() if msg.created_at else "",
                    "rating": getattr(msg, 'rating', None)
                }
                for msg in messages
            ]
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")


@router.post("/feedback/{message_id}")
@timed('backend.routers.chat.submit_feedback')
async def submit_feedback(
    message_id: int,
    payload: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    message = (await db.execute(select(Message).where(Message.id == message_id))).scalar_one_or_none()
    if not message:
        raise HTTPException(404, 'Message not found')
    await owned_conversation(db, message.conversation_id, current_user_id)
    if message.role != 'assistant':
        raise HTTPException(400, 'Only assistant answers can be rated')
    try:
        await update_message_rating(db, message_id, payload.rating)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/admin/failed-logs")
@timed('backend.routers.chat.get_failed_queries')
async def get_failed_queries(
    limit: int = Query(default=50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
    admin: None = Depends(require_admin)
):
    with span('db.execute'):
        result = await db.execute(
            select(AuditLog)
            .where(AuditLog.failure_type.in_(["LOW_CONFIDENCE", "MCP_PROMPT", "ONLINE_SEARCH_EXECUTED"]))
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        )
    logs = result.scalars().all()

    return [
        {
            "id": log.id,
            "conversation_id": log.conversation_id,
            "user_query": log.user_query,
            "rag_score": log.rag_score,
            "failure_type": log.failure_type,
            "llm_response": log.llm_final_response,
            "created_at": log.created_at.isoformat() if log.created_at else None
        } for log in logs
    ]
