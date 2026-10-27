"""Chat coordination extracted from the existing route. Public contracts unchanged."""
import ast
import json
import random
import re
import unicodedata
import difflib
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from services.llm_service import get_legal_response_stream
from services.answer_policy import resolve_mode, needs_retrieval
from services.llm_service import get_friendly_response_stream
from rag.mcp_config import search_enabled, MCPConfigError
from services.db_service import (get_conversation, add_message, get_conversation_history,
    update_conversation_title, update_conversation_state, update_domain_state, log_interaction)
from services.domain_router import choose_domain, DomainClarification, DomainNoMatch, DomainConversation, property_followup, default_name
from rag.conversation_diagnostics import capture_turn
from rag.turn_context import resolve_references
from rag.runtime_options import enabled
from rag.domain_config import RAGUnavailableError

from rag.timing import timed, span, emit, TimedLock, trace_headers, TimingMiddleware, trace_id

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

@timed('backend.services.chat_orchestrator.active_procedures')
def active_procedures(messages, legacy_title="", domain=None):
    """Use the most recent local answer's source titles; no schema migration."""
    for message in reversed(messages):
        if message.role != "assistant":
            continue
        sources = parse_sources(message.sources)
        if 'RAG_ROUTE:none' in sources:
            return []
        source_domain = next((s[11:] for s in sources if isinstance(s, str) and s.startswith('RAG_DOMAIN:')), None)
        has_records = any(isinstance(s, str) and s.startswith('RAG_CONTENT:') for s in sources)
        if domain and (source_domain or (default_name() if has_records else domain)) != domain:
            return []
        if not has_records:
            continue
        titles = list(dict.fromkeys(s for s in sources
            if isinstance(s, str) and s.strip() and not s.startswith("RAG_")))
        if titles:
            return titles
    return [legacy_title] if legacy_title and (domain is None or domain == default_name()) else []


@timed('backend.services.chat_orchestrator.domain_history')
def domain_history(messages, domain):
    """Bound model history to the current domain segment, including direct/fallback turns."""
    for position in range(len(messages) - 1, -1, -1):
        message = messages[position]
        if message.role != 'assistant':
            continue
        sources = parse_sources(message.sources)
        if 'RAG_ROUTE:none' in sources:
            return messages[position + 1:]
        source_domain = next((s[11:] for s in sources if isinstance(s, str) and s.startswith('RAG_DOMAIN:')), None)
        if source_domain is None and any(isinstance(s, str) and s.startswith('RAG_CONTENT:') for s in sources):
            source_domain = default_name()
        if source_domain and source_domain != domain:
            return messages[position + 1:]
    return messages


@timed('backend.services.chat_orchestrator.resolve_search_query')
def resolve_search_query(prompt, active_titles, config=None, *, followup=False):
    """Reuse subject memory only for an identified continuation.

    A property-only continuation pins the previous records. Action continuations
    may retrieve another record (e.g. borrowing -> renewal), so contextualize
    those without pinning. Explicit new questions receive no old subjects.
    """
    from rag.domain_config import load_domain_config
    config = config or load_domain_config()
    if property_followup(prompt, active_titles, config):
        return '; '.join(active_titles) + '. ' + prompt.strip(), active_titles
    if followup and active_titles:
        return '; '.join(active_titles) + '. ' + prompt.strip(), []
    return prompt.strip(), []


@timed('backend.services.chat_orchestrator.orchestrate_chat')
@capture_turn
async def orchestrate_chat(conversation_id, payload, db):
    if not payload.user_prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    conversation = await get_conversation(db, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # A new turn invalidates the old search offer, including old UI buttons.
    checklist = dict(conversation.checklist or {})
    checklist.pop('_mcp_pending', None)
    pending_question = checklist.pop('_rag_pending_question', None)
    if getattr(payload, 'reset_domain', False) or not enabled('RAG_RESUME_CLARIFICATION'):
        pending_question = None
    if not isinstance(pending_question, str) or len(pending_question) > 2000:
        pending_question = None
    conversation.checklist = checklist
    conversation.needs_search_permission = False
    with span('db.commit'):
        await db.commit()
    await add_message(db, conversation_id, "user", payload.user_prompt)
    context_messages = await get_conversation_history(db, conversation_id)

    if len(context_messages) == 1:
        raw_title = payload.user_prompt.strip()
        new_title = raw_title.capitalize()
        if len(new_title) > 35:
            new_title = new_title[:35] + "..."
        await update_conversation_title(db, conversation_id, new_title)

    # 1. READ STATE DIRECTLY FROM DATABASE
    last_proc_title = conversation.current_procedure or ""
    current_location = conversation.location or ""

    user_prompt_clean = payload.user_prompt.strip().lower()

    # Update location memory if mentioned
    if "tp.hcm" in user_prompt_clean or "hồ chí minh" in user_prompt_clean:
        current_location = "TP.HCM"
        await update_conversation_state(db, conversation_id, location=current_location)

    state = (conversation.checklist or {}).get('_domain_state', {})
    reset_domain = getattr(payload, 'reset_domain', False)
    if reset_domain:
        state = {'allow_switching': state.get('allow_switching', False)}
        last_proc_title = ''
    switching = getattr(payload, 'allow_switching', None)
    if switching is not None:
        state = {**state, 'allow_switching': switching}
    # Save preference/reset even if detection asks for clarification.
    if reset_domain or switching is not None:
        await update_domain_state(db, conversation_id, state, clear_procedure=reset_domain)
    prior_domain = state.get('name') or default_name()
    active_titles = active_procedures(context_messages, last_proc_title, prior_domain)
    if reset_domain:
        active_titles = []
    question = payload.user_prompt.strip()
    allow_followup = True
    while True:
        try:
            pending_args = {'pending_question': pending_question} if pending_question else {}
            if not allow_followup:
                pending_args['allow_followup'] = False
            config, state, switched = await choose_domain(question, state, active_titles,
                                                          getattr(payload, 'domain', None), **pending_args)
        except DomainConversation:
            # A social interlude does not consume an unanswered clarification.
            if pending_question:
                checklist = dict(conversation.checklist or {})
                checklist['_rag_pending_question'] = pending_question
                conversation.checklist = checklist
                with span('db.commit'):
                    await db.commit()
            # Keep domain/subject memory for a later factual follow-up, but do not
            # supply prior dataset facts to a greeting/thanks response.
            @timed('backend.services.chat_orchestrator.conversation_stream')
            async def conversation_stream():
                parts = []
                complete = False
                async for frame in get_friendly_response_stream(payload.user_prompt.strip(), context_messages[-1:]):
                    if frame.strip() == 'data: [DONE]':
                        complete = True
                        continue
                    if frame.startswith('data:'):
                        try:
                            event = json.loads(frame[5:].strip())
                            if event.get('type') == 'chunk':
                                parts.append(event.get('text', ''))
                        except (ValueError, AttributeError):
                            pass
                    yield frame
                answer = ''.join(parts)
                await add_message(db, conversation_id, 'assistant', answer, ['RAG_ROUTE:conversation'])
                await log_interaction(db, conversation_id, payload.user_prompt, answer, None, False, None,
                                      'CONVERSATION' if complete else 'INCOMPLETE_STREAM')
                yield 'data: [DONE]\n\n'
            return StreamingResponse(conversation_stream(), media_type='text/event-stream')
        except DomainClarification as exc:
            text = str(exc)
            no_match = isinstance(exc, DomainNoMatch)
            if no_match:
                # A no-dataset result must not leave old evidence as the next turn's subject.
                await update_domain_state(db, conversation_id,
                    {'selection': 'auto', 'locked': False,
                     'allow_switching': state.get('allow_switching', False)}, clear_procedure=True)
            if not no_match and enabled('RAG_RESUME_CLARIFICATION'):
                checklist = dict(conversation.checklist or {})
                # Replace rather than append: repeated clarification cannot grow context indefinitely.
                unresolved = pending_question or question
                # Never truncate away a late condition/negation and later answer a different question.
                if len(unresolved) <= 2000:
                    checklist['_rag_pending_question'] = unresolved
                conversation.checklist = checklist
                with span('db.commit'):
                    await db.commit()
            await add_message(db, conversation_id, 'assistant', text, ['RAG_ROUTE:none'] if no_match else [])
            await log_interaction(db, conversation_id, payload.user_prompt, text, None, False, None,
                                  'DOMAIN_NONE' if no_match else 'DOMAIN_CLARIFICATION')
            # Copy exception data before Python clears the except variable.
            choices = exc.domains
            @timed('backend.services.chat_orchestrator.clarification_stream')
            async def clarification_stream():
                yield f"data: {json.dumps({'type': 'metadata', 'sources': [], 'domain_choices': choices}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'chunk', 'text': text}, ensure_ascii=False)}\n\n"
                yield 'data: [DONE]\n\n'
            return StreamingResponse(clarification_stream(), media_type='text/event-stream')
        resumed = bool(state.pop('resume_pending', False) and pending_question)
        if resumed:
            question = 'Câu hỏi cần trả lời: ' + pending_question + '\nĐối tượng được người dùng làm rõ: ' + question
            active_titles = []
            emit('rag.clarification', 'resumed', pending_question=pending_question, question=payload.user_prompt)
        switched = switched or reset_domain
        await update_domain_state(db, conversation_id, state, clear_procedure=switched)
        if switched or not allow_followup:
            active_titles = []
        elif not resumed:
            active_titles = active_procedures(context_messages, last_proc_title, config.name)
        mode = resolve_mode(getattr(payload, "mode", None), config)
        retrieve = mode == "strict" or await needs_retrieval(question, active_titles, config)
        if retrieve:
            # Routing chooses a dataset. Reference resolution separately decides
            # whether a same-domain question still needs the previous subject.
            search_query, followup_titles = resolve_search_query(question, active_titles, config)
            if active_titles and not followup_titles:
                try:
                    search_query = await resolve_references(question, active_titles, config)
                except RAGUnavailableError as exc:
                    raise HTTPException(503, 'Không thể xác định ngữ cảnh câu hỏi. Vui lòng thử lại hoặc nêu rõ đối tượng.') from exc
            if (allow_followup and state.get('route_action') == 'followup'
                    and state.get('selection') == 'auto' and state.get('allow_switching')
                    and not followup_titles and search_query.strip() == question.strip()):
                # The route claimed a continuation, but the existing reference call
                # returned no subjects. Resolve this disagreement before any RAG call.
                emit('rag.routing_consistency', 'retry_without_followup',
                     rejected_domain=config.name, question=question,
                     reason='followup_without_resolved_subject')
                allow_followup = False
                active_titles = []
                pending_question = None
                continue
            stream_gen, max_score = await get_legal_response_stream(
                prompt=question, search_query=search_query,
                active_procedures=followup_titles,
                domain=config.name, mode=mode, use_cache=not getattr(payload, 'bypass_answer_cache', False),
                answer_guidance=getattr(payload, 'answer_guidance', None),
            )
        else:
            # A switched domain starts a fresh model context; previous dataset facts stay isolated.
            friendly_history = domain_history(context_messages, config.name) if not switched else context_messages[-1:]
            stream_gen = get_friendly_response_stream(payload.user_prompt.strip(), friendly_history, config)
            max_score = None  # No retrieval was performed; never invent a relevance score.

        break

    @timed('backend.services.chat_orchestrator.stream_and_save_to_db')
    async def stream_and_save_to_db():
        # Report routing/mode even on direct Friendly answers and no-evidence paths.
        yield f"data: {json.dumps({'type': 'metadata', 'sources': [], 'domain': config.name, 'mode': mode}, ensure_ascii=False)}\n\n"
        if max_score == 0.0:
            try:
                offer_search = search_enabled(config)
            except (MCPConfigError, RuntimeError):
                offer_search = False
            if offer_search:
                ask_msg = "Tôi không tìm thấy đủ thông tin trong dữ liệu nội bộ. Bạn có muốn tôi tìm kiếm trực tuyến từ các website được quản trị viên cho phép không?"
                offer = await add_message(db, conversation_id, 'assistant', ask_msg, ['RAG_DOMAIN:' + config.name])
                checklist = dict(conversation.checklist or {})
                checklist['_mcp_pending'] = {'domain': config.name, 'query': search_query,
                    'question': payload.user_prompt.strip(), 'message_id': offer.id}
                conversation.checklist = checklist
                conversation.needs_search_permission = True
                with span('db.commit'):
                    await db.commit()
                await log_interaction(db, conversation_id, payload.user_prompt, ask_msg, None, False, max_score, "MCP_PROMPT")

                yield f"data: {json.dumps({'type': 'metadata', 'sources': [], 'needs_search_permission': True, 'offer_message_id': offer.id})}\n\n"
                yield f"data: {json.dumps({'type': 'chunk', 'text': ask_msg})}\n\n"
                yield "data: [DONE]\n\n"
                return
            else:
                nudge_msg = config.profile.get('fallback_message') or random.choice([
                    "Xin chào! Tôi là trợ lý hướng dẫn thủ tục hành chính. Vui lòng cho tôi biết bạn cần làm thủ tục gì nhé.",
                    "Chào bạn! Tôi ở đây để hỗ trợ giải đáp các thủ tục hành chính công. Bạn đang cần thực hiện thủ tục nào?"
                ])
                await add_message(db, conversation_id, 'assistant', nudge_msg, ['RAG_DOMAIN:' + config.name])
                await log_interaction(db, conversation_id, payload.user_prompt, nudge_msg, None, False, max_score, "LOW_CONFIDENCE")

                yield f"data: {json.dumps({'type': 'metadata', 'sources': []})}\n\n"
                yield f"data: {json.dumps({'type': 'chunk', 'text': nudge_msg})}\n\n"
                yield "data: [DONE]\n\n"
                return

        full_ai_answer = ""
        final_sources = []
        extracted_procedure = None

        stream_complete = False
        async for raw_chunk in stream_gen:
            if raw_chunk.strip() == "data: [DONE]":
                stream_complete = True
                continue
            if raw_chunk.startswith("data: "):
                data_str = raw_chunk.replace("data: ", "").strip()
                if data_str and data_str != "[DONE]":
                    try:
                        parsed = json.loads(data_str)
                        if parsed.get("type") == "metadata":
                            final_sources = list(parsed.get('sources', []))
                            final_sources.append('RAG_DOMAIN:' + config.name)
                            parsed['sources'] = final_sources
                            parsed['domain'] = config.name
                            parsed['mode'] = mode
                            raw_chunk = f"data: {json.dumps(parsed, ensure_ascii=False)}\n\n"
                            actual_sources = [s for s in final_sources if isinstance(s, str) and not s.startswith("RAG_")]
                            if actual_sources:
                                extracted_procedure = actual_sources[0]

                        elif parsed.get("type") == "chunk":
                            full_ai_answer += parsed.get("text", "")
                    except json.JSONDecodeError:
                        pass
            yield raw_chunk

        db_sources = final_sources.copy()
        if max_score is not None:
            db_sources.append(f"RAG_SCORE:{max_score}")

        if extracted_procedure and stream_complete and config.profile.get('allow_forms', config.primary_entity == 'procedure_name') and is_form_trigger(question):
            offer_text = "\n\n**Bạn có muốn tải biểu mẫu (file Word) cho thủ tục này không?**"
            full_ai_answer += offer_text
            yield f"data: {json.dumps({'type': 'chunk', 'text': offer_text})}\n\n"

        # 3. SAVE STATE FOR THE NEXT TURN
        if extracted_procedure and stream_complete:
            await update_conversation_state(db, conversation_id, current_procedure=extracted_procedure)

        await add_message(db, conversation_id, "assistant", full_ai_answer, db_sources)
        await log_interaction(db, conversation_id, payload.user_prompt, full_ai_answer, str(final_sources), False, max_score, ("FRIENDLY_DIRECT" if not retrieve else "SUCCESS") if stream_complete else "INCOMPLETE_STREAM")
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_and_save_to_db(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
