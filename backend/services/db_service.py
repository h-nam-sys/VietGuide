from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from database import User, Conversation, Message, AuditLog
import json
import os
from pathlib import Path
from datetime import datetime

# ============ User Operations ============

from rag.timing import timed, span, emit, TimedLock, trace_headers, TimingMiddleware, trace_id
from rag.conversation_diagnostics import record as diagnostic_record

@timed('backend.services.db_service.get_or_create_user')
async def get_or_create_user(db: AsyncSession, user_id: str) -> User:
    with span('db.execute'):
        result = await db.execute(select(User).where(User.user_id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        user = User(user_id=user_id)
        db.add(user)
        with span('db.commit'):
            await db.commit()
        with span('db.refresh'):
            await db.refresh(user)
    return user

# ============ Conversation Operations ============

@timed('backend.services.db_service.create_conversation')
async def create_conversation(db: AsyncSession, user_id: str, title: str = "Legal Consultation") -> Conversation:
    user = await get_or_create_user(db, user_id)
    conversation = Conversation(user_id=user.id, title=title)
    db.add(conversation)
    with span('db.commit'):
        await db.commit()
    with span('db.refresh'):
        await db.refresh(conversation)
    return conversation

@timed('backend.services.db_service.get_conversation')
async def get_conversation(db: AsyncSession, conversation_id: int) -> Conversation:
    with span('db.execute'):
        result = await db.execute(select(Conversation).where(Conversation.id == conversation_id))
    return result.scalar_one_or_none()

@timed('backend.services.db_service.get_user_conversations')
async def get_user_conversations(db: AsyncSession, user_id: str) -> list[Conversation]:
    with span('db.execute'):
        result = await db.execute(select(User).where(User.user_id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        return []

    with span('db.execute'):
        result = await db.execute(
            select(Conversation)
            .where(Conversation.user_id == user.id)
            .order_by(Conversation.created_at.desc())
        )
    return list(result.scalars().all())

@timed('backend.services.db_service.update_conversation_title')
async def update_conversation_title(db: AsyncSession, conversation_id: int, new_title: str):
    conv = await get_conversation(db, conversation_id)
    if conv:
        conv.title = new_title
        with span('db.commit'):
            await db.commit()

@timed('backend.services.db_service.update_search_permission')
async def update_search_permission(db: AsyncSession, conversation_id: int, permission: bool):
    conv = await get_conversation(db, conversation_id)
    if conv:
        conv.needs_search_permission = permission
        with span('db.commit'):
            await db.commit()

@timed('backend.services.db_service.update_form_offer_flag')
async def update_form_offer_flag(db: AsyncSession, conversation_id: int, state: bool):
    conv = await get_conversation(db, conversation_id)
    if conv:
        conv.offered_form_download = state
        with span('db.commit'):
            await db.commit()

@timed('backend.services.db_service.update_conversation_state')
async def update_conversation_state(db: AsyncSession, conversation_id: int, current_procedure: str = None, location: str = None):
    """Updates the explicit memory state of the conversation (TC10)"""
    conv = await get_conversation(db, conversation_id)
    if conv:
        if current_procedure is not None:
            conv.current_procedure = current_procedure
        if location is not None:
            conv.location = location
        with span('db.commit'):
            await db.commit()


@timed('backend.services.db_service.update_domain_state')
async def update_domain_state(db, conversation_id, state, *, clear_procedure=False):
    conv = await get_conversation(db, conversation_id)
    if conv:
        # Reassign JSON so SQLAlchemy detects the change; preserve existing checklist values.
        checklist = dict(conv.checklist or {})
        checklist['_domain_state'] = state
        conv.checklist = checklist
        if clear_procedure:
            conv.current_procedure = None
        with span('db.commit'):
            await db.commit()
        diagnostic_record('domain_state', conversation_id=conversation_id, state=dict(state))

# ============ Message Operations ============

@timed('backend.services.db_service.add_message')
async def add_message(db: AsyncSession, conversation_id: int, role: str, content: str, sources: list[str] = None) -> Message:
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        sources=json.dumps(sources or [])
    )
    db.add(message)
    with span('db.commit'):
        await db.commit()
    with span('db.refresh'):
        await db.refresh(message)
    diagnostic_record('message', conversation_id=conversation_id, message_id=message.id,
                      role=role, content=content, sources=sources or [])
    return message

@timed('backend.services.db_service.get_conversation_history')
async def get_conversation_history(db: AsyncSession, conversation_id: int) -> list[Message]:
    with span('db.execute'):
        result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        )
    return list(result.scalars().all())

@timed('backend.services.db_service.get_last_n_messages')
async def get_last_n_messages(db: AsyncSession, conversation_id: int, n: int = 10) -> list[Message]:
    with span('db.execute'):
        result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(n)
        )
    messages = list(result.scalars().all())
    return messages[::-1]

# ============ Audit Log Operations ============

@timed('backend.services.db_service.log_interaction')
async def log_interaction(
    db: AsyncSession,
    conversation_id: int,
    user_query: str,
    llm_response: str,
    local_context: str = None,
    used_online_search: bool = False,
    rag_score: float = None,
    failure_type: str = None
):
    audit_entry = AuditLog(
        conversation_id=conversation_id,
        user_query=user_query,
        retrieved_local_context=local_context,
        llm_final_response=llm_response,
        used_online_search=used_online_search,
        rag_score=rag_score,
        failure_type=failure_type
    )
    diagnostic_record('interaction', conversation_id=conversation_id,
                      used_online_search=used_online_search, rag_score=rag_score,
                      outcome=failure_type, local_context=local_context)
    db.add(audit_entry)
    with span('db.commit'):
        await db.commit()

# ============ Feedback Operations ============

@timed('backend.services.db_service.update_message_rating')
async def update_message_rating(db: AsyncSession, message_id: int, rating: int):
    with span('db.execute'):
        result = await db.execute(select(Message).where(Message.id == message_id))
    msg = result.scalar_one_or_none()

    if not msg or not hasattr(msg, 'rating'):
        return

    msg.rating = rating
    with span('db.commit'):
        await db.commit()

    if msg.role == "assistant":
        # Find the preceding user message
        with span('db.execute'):
            result = await db.execute(
                select(Message)
                .where(
                    Message.conversation_id == msg.conversation_id,
                    Message.id < msg.id,
                    Message.role == "user"
                )
                .order_by(Message.id.desc())
                .limit(1)
            )
        user_msg = result.scalar_one_or_none()

        if user_msg:
            raw_sources = json.loads(msg.sources) if msg.sources else []
            clean_sources = [s for s in raw_sources if not s.startswith("RAG_SCORE:")]

            entry = {
                "timestamp": datetime.now().isoformat(),
                "rating": rating,
                "instruction": "Bạn là cán bộ hướng dẫn thủ tục hành chính công. Trả lời ngắn gọn, lịch sự, đúng trọng tâm và báo rõ giấy tờ còn thiếu.",
                "input": user_msg.content,
                "output": msg.content,
                "rag_context": clean_sources
            }

            with open(Path(os.getenv("FEEDBACK_DIR", ".")) / "all_feedback_dataset.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            if rating == 1:
                with open(Path(os.getenv("FEEDBACK_DIR", ".")) / "sft_positive_dataset.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            elif rating == 0:
                with open(Path(os.getenv("FEEDBACK_DIR", ".")) / "review_negative_dataset.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
