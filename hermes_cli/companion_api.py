import asyncio
import hashlib
import json
import logging
import os
import queue
import shutil
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Generator, List, Optional

import anyio
import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette._utils import collapse_excgroups
from starlette.requests import ClientDisconnect

from hermes_cli.companion_profile_policy import (
    STYLE_GUARD_V1_POLICY,
    ensure_companion_profile,
    profile_lock,
    read_conversation_policy,
)
from hermes_cli.companion_prompt import build_companion_system_prompt
from hermes_cli.config import load_config
from hermes_cli.companion_turn_guard import (
    TurnReviewStore,
    assistant_sha256,
    build_guarded_history,
    review_turn,
)

router = APIRouter(prefix="/companion/v1")
logger = logging.getLogger(__name__)
_SESSION_LOCKS = {}
_SESSION_LOCKS_GUARD = threading.Lock()
_UNRESOLVED_REVIEW_TASKS = {}
_UNRESOLVED_REVIEW_TASKS_GUARD = threading.Lock()
_stream_event_hook = None
# 上下文压缩阈值：保留最近 N 个用户轮次的原文，更早的压成摘要。
# 原 30/4/160 过于激进，长聊（单日 200+ 轮）会把关键剧情设定压成 160 字片段，
# 导致 AI "记不得前面说过的"。放宽以保留更多早期细节。
_COMPANION_HISTORY_RECENT_USER_TURNS = 80
_COMPANION_EARLY_SUMMARY_EDGE_MESSAGES = 8
_COMPANION_EARLY_SUMMARY_CONTENT_CHARS = 500


def _style_guard_new_profiles_enabled() -> bool:
    try:
        config = load_config()
    except Exception:
        logger.exception("Failed to load companion profile feature flags")
        return False
    companion_config = config.get("companion") or {}
    return bool(companion_config.get("style_guard_new_profiles_enabled", True))


def _log_style_guard_event(
    *,
    profile_dir: str,
    turn_id: str,
    policy: str,
    status: str,
    elapsed_ms: int,
    model: str,
) -> None:
    profile_hash = hashlib.sha256(
        str(Path(profile_dir).resolve()).encode("utf-8")
    ).hexdigest()[:16]
    payload = {
        "profile_hash": profile_hash,
        "turn_id": str(turn_id or ""),
        "policy": str(policy or ""),
        "status": str(status or ""),
        "elapsed_ms": max(0, int(elapsed_ms)),
        "model": str(model or ""),
    }
    logger.info(
        "savana_companion.style_guard %s",
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
    )

class ChatRequest(BaseModel):
    user_id: str
    character_id: str
    message_id: str
    user_message: str
    character_profile: Dict[str, Any]
    stream: bool = True
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    directives: Optional[str] = None
    companion_directives: Optional[str] = None
    request_overrides: Optional[Dict[str, Any]] = None
    reasoning_config: Optional[Dict[str, Any]] = None

class MemorySyncRequest(BaseModel):
    target: str
    action: str
    content: Optional[str] = None
    old_text: Optional[str] = None


class WeixinQrStartRequest(BaseModel):
    session_id: str
    character_profile: Optional[Dict[str, Any]] = None
    bot_type: str = "3"
    timeout_seconds: int = 480


class WeixinQrStatusRequest(BaseModel):
    session_id: str
    qrcode: str
    base_url: Optional[str] = None


class WeixinQrSessionRequest(BaseModel):
    session_id: str


def _safe_char_len(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(value, ensure_ascii=False))
    except Exception:
        return len(str(value))


def _read_text_if_exists(path: Path) -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8")
    except Exception:
        return ""
    return ""


def _load_companion_memory_snapshots(profile_dir: str) -> Dict[str, str]:
    memories_dir = Path(profile_dir) / "memories"
    return {
        "memory": _read_text_if_exists(memories_dir / "MEMORY.md"),
        "user": _read_text_if_exists(memories_dir / "USER.md"),
    }


def _get_latest_session_stats(session_db, session_id: str) -> Dict[str, Any]:
    try:
        session = session_db.get_session(session_id) or {}
    except Exception:
        logger.exception("Failed to load session stats for %s", session_id)
        return {}

    return {
        "message_count": int(session.get("message_count") or 0),
        "input_tokens": int(session.get("input_tokens") or 0),
        "output_tokens": int(session.get("output_tokens") or 0),
        "cache_read_tokens": int(session.get("cache_read_tokens") or 0),
        "cache_write_tokens": int(session.get("cache_write_tokens") or 0),
        "model": str(session.get("model") or ""),
        "stored_system_prompt_chars": _safe_char_len(session.get("system_prompt") or ""),
    }


def _build_history_preview(conversation_history: List[Dict[str, Any]], limit: int = 4, preview_chars: int = 80) -> List[Dict[str, str]]:
    preview: List[Dict[str, str]] = []
    for msg in (conversation_history or [])[-limit:]:
        role = str((msg or {}).get("role") or "")
        content = str((msg or {}).get("content") or "")
        preview.append(
            {
                "role": role,
                "content_preview": content[:preview_chars],
            }
        )
    return preview


def extract_evolved_persona_from_text(content: str) -> Optional[str]:
    if not content:
        return None
    lines = content.splitlines()
    evolved_lines = []
    in_section = False

    for line in lines:
        if line.startswith("## Evolved Persona"):
            in_section = True
            continue
        if in_section:
            if line.startswith("#"):
                break
            evolved_lines.append(line)

    if not in_section:
        return None

    return "\n".join(evolved_lines).strip()


def _build_soul_section_stats(soul_text: str) -> Dict[str, int]:
    text = str(soul_text or "")
    sections = [line for line in text.splitlines() if line.startswith("## ")]
    evolved_persona = extract_evolved_persona_from_text(text)
    return {
        "total_sections": len(sections),
        "evolved_persona_chars": _safe_char_len(evolved_persona or ""),
    }


def _log_companion_prompt_diagnostics(
    *,
    session_id: str,
    provider: str,
    model: str,
    user_message: str,
    conversation_history: List[Dict[str, Any]],
    character_profile: Dict[str, Any],
    directives: Optional[str],
    soul_text: str,
    memory_text: str,
    user_profile_text: str,
    session_stats: Dict[str, Any],
) -> None:
    history_user_messages = 0
    history_assistant_messages = 0
    history_tool_messages = 0
    history_chars = 0

    for msg in conversation_history or []:
        role = str((msg or {}).get("role") or "")
        content = (msg or {}).get("content") or ""
        history_chars += _safe_char_len(content)
        if role == "user":
            history_user_messages += 1
        elif role == "assistant":
            history_assistant_messages += 1
        elif role in {"tool", "function"}:
            history_tool_messages += 1

    profile_sample_dialogues = character_profile.get("sample_dialogues") or []
    profile_relationship = character_profile.get("relationship") or {}
    relationship_stage = str(profile_relationship.get("relationship_stage") or "")
    directives_text = str(directives or "")

    payload = {
        "session_id": session_id,
        "provider": str(provider or ""),
        "model": str(model or ""),
        "user_message_chars": _safe_char_len(user_message),
        "directives_chars": _safe_char_len(directives_text),
        "directives_preview": directives_text[:200],
        "history_messages": len(conversation_history or []),
        "history_user_messages": history_user_messages,
        "history_assistant_messages": history_assistant_messages,
        "history_tool_messages": history_tool_messages,
        "history_chars": history_chars,
        "history_preview": _build_history_preview(conversation_history),
        "soul_chars": _safe_char_len(soul_text),
        "soul_sections": _build_soul_section_stats(soul_text),
        "memory_chars": _safe_char_len(memory_text),
        "user_profile_chars": _safe_char_len(user_profile_text),
        "profile_name": str(character_profile.get("name") or ""),
        "profile_personality_chars": _safe_char_len(character_profile.get("personality") or ""),
        "profile_background_chars": _safe_char_len(character_profile.get("background") or ""),
        "profile_speaking_style_chars": _safe_char_len(character_profile.get("speaking_style") or ""),
        "profile_sample_dialogues": len(profile_sample_dialogues) if isinstance(profile_sample_dialogues, list) else 0,
        "profile_has_relationship": bool(profile_relationship),
        "relationship_stage": relationship_stage,
        "session_message_count": int(session_stats.get("message_count") or 0),
        "session_input_tokens": int(session_stats.get("input_tokens") or 0),
        "session_output_tokens": int(session_stats.get("output_tokens") or 0),
        "session_cache_read_tokens": int(session_stats.get("cache_read_tokens") or 0),
        "session_cache_write_tokens": int(session_stats.get("cache_write_tokens") or 0),
        "session_stored_model": str(session_stats.get("model") or ""),
        "session_stored_system_prompt_chars": int(session_stats.get("stored_system_prompt_chars") or 0),
    }
    logger.info(
        "savana_companion.prompt_diagnostics %s",
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
    )
    print(
        "[SAVANA PROMPT DIAGNOSTICS] {}".format(
            json.dumps(payload, ensure_ascii=False, sort_keys=True)
        )
    )


def get_profile_path(session_id: str) -> str:
    from hermes_constants import get_default_hermes_root
    return str(get_default_hermes_root() / "profiles" / session_id)


async def _start_weixin_qr_session(
    hermes_home: str,
    session_id: str,
    bot_type: str = "3",
    timeout_seconds: int = 480,
) -> Dict[str, Any]:
    from gateway.platforms import weixin

    if not weixin.check_weixin_requirements():
        raise RuntimeError("Weixin QR login dependencies are missing")

    async with weixin.aiohttp.ClientSession(
        trust_env=True,
        connector=weixin._make_ssl_connector(),
    ) as session:
        response = await weixin._api_get(
            session,
            base_url=weixin.ILINK_BASE_URL,
            endpoint="{0}?bot_type={1}".format(weixin.EP_GET_BOT_QR, bot_type),
            timeout_ms=weixin.QR_TIMEOUT_MS,
        )

    qrcode_value = str(response.get("qrcode") or "")
    qrcode_url = str(response.get("qrcode_img_content") or "")
    if not qrcode_value:
        raise RuntimeError("Weixin QR response missing qrcode")

    return {
        "session_id": session_id,
        "qrcode": qrcode_value,
        "qrcode_img_content": qrcode_url,
        "status": "wait",
        "bot_type": bot_type,
        "timeout_seconds": timeout_seconds,
    }


async def _query_weixin_qr_status(
    hermes_home: str,
    session_id: str,
    qrcode: str,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    from gateway.platforms import weixin

    if not weixin.check_weixin_requirements():
        raise RuntimeError("Weixin QR login dependencies are missing")

    resolved_base_url = (base_url or weixin.ILINK_BASE_URL).rstrip("/")
    async with weixin.aiohttp.ClientSession(
        trust_env=True,
        connector=weixin._make_ssl_connector(),
    ) as session:
        response = await weixin._api_get(
            session,
            base_url=resolved_base_url,
            endpoint="{0}?qrcode={1}".format(weixin.EP_GET_QR_STATUS, qrcode),
            timeout_ms=weixin.QR_TIMEOUT_MS,
        )

    response["session_id"] = session_id
    return response


def _persist_weixin_binding_account(
    hermes_home: str,
    *,
    account_id: str,
    token: str,
    base_url: str,
    user_id: str,
) -> None:
    from gateway.platforms import weixin

    weixin.save_weixin_account(
        hermes_home,
        account_id=account_id,
        token=token,
        base_url=base_url,
        user_id=user_id,
    )

def extract_evolved_persona(soul_path: str) -> Optional[str]:
    if not os.path.exists(soul_path):
        return None
    try:
        with open(soul_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return None

    lines = content.splitlines()
    evolved_lines = []
    in_section = False
    
    for line in lines:
        if line.startswith("## Evolved Persona"):
            in_section = True
            continue
        if in_section:
            if line.startswith("#"):
                break
            evolved_lines.append(line)
            
    if not in_section:
        return None
        
    return "\n".join(evolved_lines).strip()


def _load_companion_history(session_db, session_id: str, current_message_id: str) -> List[Dict[str, Any]]:
    try:
        messages = session_db.get_messages_as_conversation(session_id)
    except Exception as exc:
        logger.exception(
            "Failed to load companion history for session %s: %s",
            session_id,
            exc,
        )
        raise

    history = []
    for msg in messages:
        if (
            current_message_id
            and isinstance(msg, dict)
            and msg.get("role") == "user"
            and msg.get("message_id") == current_message_id
        ):
            continue
        history.append(msg)
    return history


def _count_user_turns(messages: List[Dict[str, Any]]) -> int:
    return sum(1 for msg in messages or [] if (msg or {}).get("role") == "user")


def _truncate_text_for_summary(value: Any, limit: int = _COMPANION_EARLY_SUMMARY_CONTENT_CHARS) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _summarize_early_companion_history(omitted: List[Dict[str, Any]]) -> str:
    omitted = list(omitted or [])
    omitted_count = len(omitted)
    omitted_user_turns = _count_user_turns(omitted)
    if not omitted_count:
        return ""

    edge = _COMPANION_EARLY_SUMMARY_EDGE_MESSAGES
    sample_messages = omitted[:edge]
    if omitted_count > edge:
        sample_messages = sample_messages + omitted[-edge:]

    lines = [
        "【早期对话摘要（自动压缩，仅用于承接事实）】",
        "此前省略了 {0} 条历史消息，其中用户轮次 {1} 个。".format(
            omitted_count,
            omitted_user_turns,
        ),
    ]
    for msg in sample_messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "unknown")
        content = _truncate_text_for_summary(msg.get("content"))
        if not content:
            continue
        lines.append("- {0}: {1}".format(role, content))
    return "\n".join(lines)


def _compact_companion_history_for_prompt(
    messages: List[Dict[str, Any]],
    recent_user_turns: int = _COMPANION_HISTORY_RECENT_USER_TURNS,
) -> List[Dict[str, Any]]:
    messages = list(messages or [])
    if recent_user_turns <= 0:
        return messages
    if _count_user_turns(messages) <= recent_user_turns:
        return messages

    seen_user_turns = 0
    recent_start = 0
    for index in range(len(messages) - 1, -1, -1):
        msg = messages[index]
        if isinstance(msg, dict) and msg.get("role") == "user":
            seen_user_turns += 1
            if seen_user_turns == recent_user_turns:
                recent_start = index
                break

    omitted = messages[:recent_start]
    recent = messages[recent_start:]
    summary = _summarize_early_companion_history(omitted)
    if not summary:
        return recent
    return [
        {"role": "user", "content": summary},
        {"role": "assistant", "content": "收到，我会基于这份早期摘要承接当前对话。"},
    ] + recent


def _get_session_lock(session_id: str) -> threading.Lock:
    with _SESSION_LOCKS_GUARD:
        lock = _SESSION_LOCKS.get(session_id)
        if lock is None:
            lock = threading.Lock()
            _SESSION_LOCKS[session_id] = lock
        return lock


def _emit_stream_event(name: str, payload: Optional[Dict[str, Any]] = None) -> None:
    hook = _stream_event_hook
    if callable(hook):
        try:
            hook(name, payload)
        except Exception:
            logger.exception("stream event hook failed for %s", name)


def _start_background_thread(target) -> threading.Thread:
    worker = threading.Thread(target=target)
    worker.start()
    return worker


def _safe_acquire_lock(lock: Optional[threading.Lock], timeout: float = 5.0) -> bool:
    if lock is None:
        return True
    try:
        return lock.acquire(timeout=timeout)
    except TypeError:
        try:
            return lock.acquire(True, timeout)
        except TypeError:
            return lock.acquire()


class _CompanionStreamResponse(StreamingResponse):
    def __init__(
        self,
        *args,
        final_marker_consumed: threading.Event,
        producer_done: threading.Event,
        session_lock: Optional[threading.Lock],
        session_lock_released_ref: Dict[str, bool],
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._final_marker_consumed = final_marker_consumed
        self._producer_done = producer_done
        self._session_lock = session_lock
        self._session_lock_released_ref = session_lock_released_ref

    async def _cleanup_stream_resources(self) -> None:
        self._final_marker_consumed.set()
        _emit_stream_event("final_marker_set")
        await asyncio.get_running_loop().run_in_executor(None, self._producer_done.wait)
        if self._session_lock is not None and not self._session_lock_released_ref["released"]:
            try:
                self._session_lock.release()
            except RuntimeError:
                pass
            self._session_lock_released_ref["released"] = True
            _emit_stream_event("lock_released")

    async def __call__(self, scope, receive, send) -> None:
        try:
            spec_version = tuple(map(int, scope.get("asgi", {}).get("spec_version", "2.0").split(".")))

            if spec_version >= (2, 4):
                try:
                    await self.stream_response(send)
                except OSError:
                    raise
            else:
                with collapse_excgroups():
                    async with anyio.create_task_group() as task_group:

                        async def wrap(func: Callable[[], Awaitable[None]]) -> None:
                            await func()
                            task_group.cancel_scope.cancel()

                        task_group.start_soon(wrap, partial(self.stream_response, send))
                        await wrap(partial(self.listen_for_disconnect, receive))

            if self.background is not None:
                await self.background()
        except ClientDisconnect:
            return
        finally:
            cleanup_task = asyncio.create_task(self._cleanup_stream_resources())
            await asyncio.shield(cleanup_task)


async def _acquire_session_lock_cancellation_safe(session_lock: threading.Lock, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    acquire_future = loop.run_in_executor(None, lambda: _safe_acquire_lock(session_lock, timeout=timeout))

    def _release_if_acquired(future) -> None:
        try:
            acquired = bool(future.result())
        except Exception:
            logger.exception("session lock acquire future cleanup failed")
            return
        if acquired:
            try:
                session_lock.release()
            except Exception:
                logger.exception("session lock orphan acquire release failed")

    try:
        acquired = await asyncio.shield(acquire_future)
        if not acquired:
            logger.warning("session_lock.acquire_timeout_exceeded_breaking_stale_lock")
            try:
                session_lock.release()
            except Exception:
                pass
            _safe_acquire_lock(session_lock, timeout=0.1)
    except asyncio.CancelledError:
        if acquire_future.done():
            _release_if_acquired(acquire_future)
        else:
            acquire_future.add_done_callback(_release_if_acquired)
        raise


def _build_style_guard_system_prompt(
    soul_text: str,
    memory_text: str,
    user_profile_text: str,
    companion_directives: Optional[str],
) -> str:
    prompt = build_companion_system_prompt(
        soul_text,
        memory_text,
        user_profile_text,
    )
    directives_text = str(companion_directives or "").strip()
    if not directives_text:
        return prompt
    return (
        prompt
        + "\n\n对话指引\n"
        + directives_text
    ).strip()


def _index_history_turns(raw_history: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed = {}
    previous_user_turn = None
    for idx, message in enumerate(list(raw_history or [])):
        role = (message or {}).get("role")
        if role == "user":
            previous_user_turn = {
                "turn_id": (message or {}).get("message_id"),
                "user_message": (message or {}).get("content") or "",
                "assistant_prefix_messages": list(raw_history[:idx]),
            }
            continue
        if role == "assistant" and previous_user_turn and previous_user_turn.get("turn_id"):
            indexed[previous_user_turn["turn_id"]] = {
                "turn_id": previous_user_turn["turn_id"],
                "user_message": previous_user_turn["user_message"],
                "assistant_text": (message or {}).get("content") or "",
                "messages": previous_user_turn["assistant_prefix_messages"],
            }
            previous_user_turn = None
            continue
        if role not in ("tool", "function"):
            previous_user_turn = None
    return indexed


def _restore_unresolved_reviews(
    *,
    profile_dir: str,
    raw_history: List[Dict[str, Any]],
    provider: str,
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    memory_store: Any,
) -> None:
    turn_store = TurnReviewStore(profile_dir)
    unresolved = turn_store.list_unresolved()
    if not unresolved:
        return
    indexed_turns = _index_history_turns(raw_history)
    for item in unresolved:
        turn_id = str(item.get("turn_id") or "")
        history_turn = indexed_turns.get(turn_id)
        if not history_turn:
            continue
        assistant_text = history_turn.get("assistant_text") or ""
        if item.get("assistant_sha256") != assistant_sha256(assistant_text):
            continue
        try:
            review_turn(
                profile_dir=profile_dir,
                turn_id=turn_id,
                assistant_text=assistant_text,
                user_message=history_turn.get("user_message") or "",
                messages=history_turn.get("messages") or [],
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=api_key,
                memory_store=memory_store,
                store=turn_store,
            )
        except Exception:
            logger.exception("Failed to restore unresolved companion turn %s", turn_id)


def _schedule_unresolved_reviews(**kwargs) -> bool:
    profile_key = str(Path(kwargs["profile_dir"]).resolve())
    with _UNRESOLVED_REVIEW_TASKS_GUARD:
        current = _UNRESOLVED_REVIEW_TASKS.get(profile_key)
        if current is not None and current.is_alive():
            return False

        def run_restore():
            try:
                _restore_unresolved_reviews(**kwargs)
            except Exception:
                logger.exception("Failed to run unresolved companion review task for %s", profile_key)
            finally:
                with _UNRESOLVED_REVIEW_TASKS_GUARD:
                    if _UNRESOLVED_REVIEW_TASKS.get(profile_key) is threading.current_thread():
                        _UNRESOLVED_REVIEW_TASKS.pop(profile_key, None)

        worker = threading.Thread(
            target=run_restore,
            name="companion-review-restore",
            daemon=True,
        )
        _UNRESOLVED_REVIEW_TASKS[profile_key] = worker
    try:
        worker.start()
    except Exception:
        with _UNRESOLVED_REVIEW_TASKS_GUARD:
            if _UNRESOLVED_REVIEW_TASKS.get(profile_key) is worker:
                _UNRESOLVED_REVIEW_TASKS.pop(profile_key, None)
        logger.exception("Failed to start unresolved companion review task for %s", profile_key)
        return False
    return True


def _fallback_review_metadata(turn_id: str, error: Exception) -> Dict[str, Any]:
    logger.exception("Companion style review failed for turn %s", turn_id)
    return {
        "turn_id": str(turn_id or ""),
        "review_status": "pending",
        "memory_status": "pending",
        "memory_modifications": [],
        "error": str(error),
    }


def _mark_companion_turn_pending(profile_dir: str, turn_id: str, assistant_text: str) -> None:
    store = TurnReviewStore(profile_dir)
    current = store.begin(turn_id, assistant_text)
    if current and current.get("status") != "pending":
        store._commit_status(
            turn_id,
            current["assistant_sha256"],
            "pending",
            "",
            "",
            {},
            "pending",
            "",
        )


def _review_companion_turn(
    *,
    style_guard_enabled: bool,
    profile_dir: str,
    turn_id: str,
    assistant_text: str,
    user_message: str,
    raw_history: List[Dict[str, Any]],
    provider: str,
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    memory_store: Any,
) -> Dict[str, Any]:
    if not style_guard_enabled:
        modifications = []
        if memory_store is not None and hasattr(memory_store, "modifications"):
            modifications = memory_store.modifications
        return {
            "turn_id": str(turn_id or ""),
            "review_status": None,
            "memory_status": None,
            "memory_modifications": modifications,
        }
    started_at = time.monotonic()
    try:
        result = review_turn(
            profile_dir=profile_dir,
            turn_id=turn_id,
            assistant_text=assistant_text,
            user_message=user_message,
            messages=raw_history,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            memory_store=memory_store,
        )
        _log_style_guard_event(
            profile_dir=profile_dir,
            turn_id=turn_id,
            policy=STYLE_GUARD_V1_POLICY,
            status=result.get("review_status") or "completed",
            elapsed_ms=int((time.monotonic() - started_at) * 1000),
            model=model,
        )
        return result
    except Exception as exc:
        try:
            _mark_companion_turn_pending(profile_dir, turn_id, assistant_text)
        except Exception:
            logger.exception("Failed to preserve pending companion turn %s", turn_id)
        result = _fallback_review_metadata(turn_id, exc)
        _log_style_guard_event(
            profile_dir=profile_dir,
            turn_id=turn_id,
            policy=STYLE_GUARD_V1_POLICY,
            status="pending",
            elapsed_ms=int((time.monotonic() - started_at) * 1000),
            model=model,
        )
        return result


def _get_weixin_bridge_runtime():
    # type: () -> Dict[str, Any]
    bridge_url = (
        os.environ.get("SAVANA_WECHAT_BRIDGE_URL", "").strip()
        or "http://127.0.0.1:8005/api/v1/wechat-role-binding/bridge/inbound"
    )
    bridge_secret = os.environ.get("SAVANA_WECHAT_BRIDGE_SECRET", "").strip()
    enabled_raw = os.environ.get("SAVANA_WECHAT_BRIDGE_ENABLED", "").strip().lower()
    enabled = enabled_raw in {"1", "true", "yes", "on"}
    if not enabled and bridge_secret:
        enabled = True
    return {
        "enabled": enabled,
        "bridge_url": bridge_url,
        "bridge_secret": bridge_secret,
    }


async def _relay_inbound_weixin_message(
    hermes_home: str,
    bridge_url: str,
    bridge_secret: str,
    account_id: str,
    account_payload: Dict[str, Any],
    message: Dict[str, Any],
) -> Optional[str]:
    from gateway.platforms import weixin

    sender_id = str(message.get("from_user_id") or "").strip()
    if not sender_id:
        return None

    item_list = message.get("item_list") or []
    text = weixin._extract_text(item_list).strip()
    if not text:
        return None

    chat_type, effective_chat_id = weixin._guess_chat_type(message, account_id)
    if chat_type != "dm":
        return None

    token_store = weixin.ContextTokenStore(hermes_home)
    token_store.restore(account_id)
    context_token = str(message.get("context_token") or "").strip()
    if context_token:
        token_store.set(account_id, sender_id, context_token)

    headers = {"X-WeChat-Bridge-Secret": bridge_secret}
    payload = {
        "wechat_channel_user_id": sender_id,
        "message_text": text,
    }
    base_url = str(account_payload.get("base_url") or weixin.ILINK_BASE_URL).rstrip("/")
    token = str(account_payload.get("token") or "")
    typing_ticket = ""

    try:
        async with weixin.aiohttp.ClientSession(
            trust_env=True,
            connector=weixin._make_ssl_connector(),
        ) as typing_session:
            config = await weixin._get_config(
                typing_session,
                base_url=base_url,
                token=token,
                user_id=sender_id,
                context_token=context_token or None,
            )
            typing_ticket = str(config.get("typing_ticket") or "")
            if typing_ticket:
                await weixin._send_typing(
                    typing_session,
                    base_url=base_url,
                    token=token,
                    to_user_id=sender_id,
                    typing_ticket=typing_ticket,
                    status=weixin.TYPING_START,
                )
    except Exception as exc:
        logger.debug("weixin typing start skipped for %s: %s", sender_id, exc)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                bridge_url,
                headers=headers,
                json=payload,
            )
    finally:
        if typing_ticket:
            try:
                async with weixin.aiohttp.ClientSession(
                    trust_env=True,
                    connector=weixin._make_ssl_connector(),
                ) as typing_session:
                    await weixin._send_typing(
                        typing_session,
                        base_url=base_url,
                        token=token,
                        to_user_id=sender_id,
                        typing_ticket=typing_ticket,
                        status=weixin.TYPING_STOP,
                    )
            except Exception as exc:
                logger.debug("weixin typing stop skipped for %s: %s", sender_id, exc)

    if response.status_code != 200:
        raise RuntimeError(
            "WeChat inbound bridge failed: HTTP {0} - {1}".format(
                response.status_code,
                response.text,
            )
        )

    response_json = response.json()
    reply = str(
        ((response_json or {}).get("data") or {}).get("reply") or ""
    ).strip()
    if not reply:
        return None

    send_result = await weixin.send_weixin_direct(
        extra={
            "account_id": account_id,
            "base_url": base_url,
            "token": token,
        },
        token=token,
        chat_id=effective_chat_id,
        message=reply,
    )
    if isinstance(send_result, dict) and send_result.get("error"):
        raise RuntimeError(
            "Weixin direct reply failed: {0}".format(send_result["error"])
        )

    logger.info(
        "weixin bridge relayed message for %s -> %s (message_id=%s)",
        account_id,
        sender_id,
        isinstance(send_result, dict) and send_result.get("message_id") or "",
    )

    return reply


class WeixinInboundBridgeWorker(object):
    def __init__(self):
        self._task = None
        self._stop_event = None
        self._poll_tasks = {}

    async def start(self):
        runtime = _get_weixin_bridge_runtime()
        if not runtime["enabled"] or not runtime["bridge_secret"]:
            logger.info("weixin bridge worker disabled: missing enable flag or secret")
            return
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run(runtime))
        logger.info("weixin bridge worker started")

    async def stop(self):
        if self._task is None:
            return
        self._stop_event.set()
        for task in self._poll_tasks.values():
            task.cancel()
        await asyncio.gather(*self._poll_tasks.values(), return_exceptions=True)
        self._poll_tasks = {}
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None
        logger.info("weixin bridge worker stopped")

    async def _run(self, runtime: Dict[str, Any]):
        from hermes_constants import get_default_hermes_root

        hermes_home = str(get_default_hermes_root())
        accounts_dir = Path(hermes_home) / "weixin" / "accounts"
        while not self._stop_event.is_set():
            known_accounts = {}
            if accounts_dir.exists():
                for account_file in accounts_dir.glob("*.json"):
                    if account_file.name.endswith(".context-tokens.json"):
                        continue
                    if account_file.name.endswith(".sync.json"):
                        continue
                    try:
                        known_accounts[account_file.stem] = json.loads(
                            account_file.read_text(encoding="utf-8")
                        )
                    except Exception as exc:
                        logger.warning(
                            "failed to load weixin account file %s: %s",
                            account_file,
                            exc,
                        )

            for account_id, payload in known_accounts.items():
                poll_task = self._poll_tasks.get(account_id)
                if poll_task is None or poll_task.done():
                    self._poll_tasks[account_id] = asyncio.create_task(
                        self._poll_account(
                            hermes_home=hermes_home,
                            bridge_url=runtime["bridge_url"],
                            bridge_secret=runtime["bridge_secret"],
                            account_id=account_id,
                            account_payload=payload,
                        )
                    )

            inactive_accounts = [
                account_id
                for account_id in list(self._poll_tasks.keys())
                if account_id not in known_accounts
            ]
            for account_id in inactive_accounts:
                self._poll_tasks.pop(account_id).cancel()

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                continue

    async def _poll_account(
        self,
        hermes_home: str,
        bridge_url: str,
        bridge_secret: str,
        account_id: str,
        account_payload: Dict[str, Any],
    ):
        from gateway.platforms import weixin
        from gateway.platforms.helpers import MessageDeduplicator

        token = str(account_payload.get("token") or "").strip()
        if not token:
            logger.warning("skip weixin bridge polling for %s: missing token", account_id)
            return

        base_url = str(
            account_payload.get("base_url") or weixin.ILINK_BASE_URL
        ).strip().rstrip("/")
        sync_buf = weixin._load_sync_buf(hermes_home, account_id)
        timeout_ms = weixin.LONG_POLL_TIMEOUT_MS
        dedup = MessageDeduplicator(ttl_seconds=weixin.MESSAGE_DEDUP_TTL_SECONDS)

        async with weixin.aiohttp.ClientSession(
            trust_env=True,
            connector=weixin._make_ssl_connector(),
        ) as session:
            while not self._stop_event.is_set():
                try:
                    response = await weixin._get_updates(
                        session,
                        base_url=base_url,
                        token=token,
                        sync_buf=sync_buf,
                        timeout_ms=timeout_ms,
                    )
                    suggested_timeout = response.get("longpolling_timeout_ms")
                    if isinstance(suggested_timeout, int) and suggested_timeout > 0:
                        timeout_ms = suggested_timeout

                    ret = response.get("ret", 0)
                    errcode = response.get("errcode", 0)
                    if ret not in {0, None} or errcode not in {0, None}:
                        await asyncio.sleep(2)
                        continue

                    new_sync_buf = str(response.get("get_updates_buf") or "")
                    if new_sync_buf:
                        sync_buf = new_sync_buf
                        weixin._save_sync_buf(hermes_home, account_id, sync_buf)

                    for message in response.get("msgs") or []:
                        message_id = str(message.get("message_id") or "").strip()
                        if message_id and dedup.is_duplicate(message_id):
                            continue
                        text = weixin._extract_text(message.get("item_list") or [])
                        sender_id = str(message.get("from_user_id") or "").strip()
                        if text and sender_id:
                            content_key = "content:{0}:{1}:{2}".format(
                                account_id,
                                sender_id,
                                text,
                            )
                            if dedup.is_duplicate(content_key):
                                continue
                        try:
                            reply = await _relay_inbound_weixin_message(
                                hermes_home=hermes_home,
                                bridge_url=bridge_url,
                                bridge_secret=bridge_secret,
                                account_id=account_id,
                                account_payload=account_payload,
                                message=message,
                            )
                            if reply:
                                logger.info(
                                    "weixin bridge relayed message for %s -> %s",
                                    account_id,
                                    sender_id,
                                )
                        except Exception as exc:
                            logger.error(
                                "weixin bridge relay failed for %s: %s",
                                account_id,
                                exc,
                                exc_info=True,
                            )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error(
                        "weixin bridge poll failed for %s: %s",
                        account_id,
                        exc,
                        exc_info=True,
                    )
                    await asyncio.sleep(2)


_WEIXIN_BRIDGE_WORKER = WeixinInboundBridgeWorker()


async def start_weixin_bridge_worker():
    await _WEIXIN_BRIDGE_WORKER.start()


async def stop_weixin_bridge_worker():
    await _WEIXIN_BRIDGE_WORKER.stop()

def sync_soul_file(profile_dir: str, profile_data: Dict[str, Any]) -> None:
    with profile_lock(Path(profile_dir), "soul"):
        _sync_soul_file_unlocked(profile_dir, profile_data)


def _sync_soul_file_unlocked(profile_dir: str, profile_data: Dict[str, Any]) -> None:
    soul_path = os.path.join(profile_dir, "SOUL.md")
    evolved_persona = extract_evolved_persona(soul_path)
    if not evolved_persona or not evolved_persona.strip():
        evolved_persona = "(暂无自主进化，人设遵循基础设定)"
    os.makedirs(os.path.dirname(soul_path), exist_ok=True)
    
    name = str(profile_data.get("name", "Companion"))
    
    # 兼容处理 speaking_style 可能是 List[str] 或 Dict[str, str] 的情况
    speaking_style = profile_data.get("speaking_style", "")
    if isinstance(speaking_style, list):
        speaking_style = ", ".join(speaking_style)
    elif isinstance(speaking_style, dict):
        speaking_style = ", ".join("{}: {}".format(k, v) for k, v in speaking_style.items())
        
    personality = str(profile_data.get("personality", "")).strip()
    background = str(profile_data.get("background", "")).strip()
    
    # 新增扩展字段处理
    behavior_tags = profile_data.get("behavior_tags") or ""
    if isinstance(behavior_tags, list):
        behavior_tags = ", ".join(behavior_tags)
    behavior_tags = str(behavior_tags).strip()
    
    appearance_details = str(profile_data.get("appearance_details") or "").strip()
    initial_scenario = str(profile_data.get("initial_scenario") or "").strip()
    
    sample_dialogues = profile_data.get("sample_dialogues") or []
    dialogue_text = ""
    if isinstance(sample_dialogues, list):
        dialogue_lines = []
        for item in sample_dialogues:
            if isinstance(item, dict):
                user_val = item.get("user") or item.get("user_input")
                char_val = item.get("character") or item.get("character_response")
                if user_val and char_val:
                    dialogue_lines.append("User: {}".format(user_val))
                    dialogue_lines.append("{}: {}".format(name, char_val))
                    dialogue_lines.append("")
        dialogue_text = "\n".join(dialogue_lines).strip()

    # Parse relationship metrics
    rel_data = profile_data.get("relationship") or {}
    relationship_section = ""
    if rel_data:
        stage = rel_data.get("relationship_stage")
        intimacy = rel_data.get("intimacy_score")
        trust = rel_data.get("trust_score")
        nickname = rel_data.get("preferred_nickname")
        profile = rel_data.get("persona_profile")
        constraints = rel_data.get("persona_prompt_constraints")
        
        stage_names = {
            "stranger": "stranger (初次相遇，礼貌疏离)",
            "acquaintance": "acquaintance (逐渐熟悉，互动自然)",
            "ambiguous": "ambiguous (暧昧期，关系推拉有张力)",
            "early_relationship": "early_relationship (热恋陪伴，甜蜜依恋)",
            "soulmate": "soulmate (灵魂伴侣，深度信任与脆弱感)",
            "deep_relationship": "deep_relationship (长线羁绊，生命中不可或缺)"
        }
        stage_desc = stage_names.get(stage, stage)
        
        parts = ["## Relationship with User"]
        if stage_desc:
            parts.append("- Current Stage: {}".format(stage_desc))
        if intimacy is not None:
            parts.append("- Intimacy level: {}/10".format(intimacy))
        if trust is not None:
            parts.append("- Trust level: {}/10".format(trust))
        if nickname:
            parts.append("- Preferred Nickname: {} (When referring to the user, you MUST call them by this nickname)".format(nickname))
        if profile:
            parts.append("- User Profile: {} (Your perception of the user's role/background in your life)".format(profile))
        if constraints:
            parts.append("- Prompt Constraints: {}".format(constraints))
            
        relationship_section = "\n".join(parts)

    content_parts = [
        "# {}".format(name),
        "## Personality\n{}".format(personality) if personality else "",
        "## Evolved Persona\n{}".format(evolved_persona),
        "## Behavior Tags\n{}".format(behavior_tags) if behavior_tags else "",
        "## Speaking Style\n{}".format(speaking_style) if speaking_style else "",
        "## Appearance\n{}".format(appearance_details) if appearance_details else "",
        "## Scenario\n{}".format(initial_scenario) if initial_scenario else "",
        "## Background\n{}".format(background) if background else "",
        "## Sample Dialogues\n{}".format(dialogue_text) if dialogue_text else "",
        relationship_section if relationship_section else "",
    ]
    
    # 过滤掉空部分，并以两个换行符连接
    content = "\n\n".join([p for p in content_parts if p]) + "\n"
    
    # 仅在内容有差异时覆写，优化缓存
    if not os.path.exists(soul_path) or open(soul_path, "r", encoding="utf-8").read() != content:
        with open(soul_path, "w", encoding="utf-8") as f:
            f.write(content)

@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    session_id = "savana_{}_{}".format(req.user_id.lower(), req.character_id.lower())
    profile_dir = get_profile_path(session_id)
    profile_path = Path(profile_dir)
    if not profile_path.exists() and not _style_guard_new_profiles_enabled():
        raise HTTPException(
            status_code=503,
            detail="Companion profile initialization is temporarily unavailable; retry later.",
            headers={"Retry-After": "5"},
        )
    ensure_companion_profile(profile_path)
    conversation_policy = read_conversation_policy(profile_dir)
    style_guard_enabled = conversation_policy == STYLE_GUARD_V1_POLICY
    session_lock = None
    if style_guard_enabled:
        session_lock = _get_session_lock(session_id)
        await _acquire_session_lock_cancellation_safe(session_lock)
    session_lock_released_ref = {"released": False}
    stream_lock_owned_by_response = False
    
    # 1. 动态物理隔离 Profile 目录 (使用线程安全的 ContextVar 覆盖)
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    token = set_hermes_home_override(profile_dir)
    
    try:
        sync_soul_file(profile_dir, req.character_profile)

        # 2. 导入与运行 AIAgent
        from hermes_state import SessionDB
        from run_agent import AIAgent

        # 初始化本地 state.db
        db_path = Path(profile_dir) / "state.db"
        session_db = SessionDB(db_path=db_path)
        
        # 确保 session 存在
        try:
            session_db.create_session(session_id, "savana")
        except Exception:
            pass
        raw_history = _load_companion_history(
            session_db, session_id, req.message_id
        )
        soul_text = _read_text_if_exists(Path(profile_dir) / "SOUL.md")
        memory_snapshots = _load_companion_memory_snapshots(profile_dir)
        session_stats = _get_latest_session_stats(session_db, session_id)
        conversation_history = _compact_companion_history_for_prompt(raw_history)

        # 动态解析模型及 API 配置
        api_key = req.api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = req.api_base or os.environ.get("DEEPSEEK_API_BASE") or "https://api.deepseek.com"
        provider = req.provider or "deepseek"
        model = req.model or "deepseek-v4-flash"
        effective_directives = ""
        diagnostics_directives = req.companion_directives or ""
        enabled_toolsets = []
        if not style_guard_enabled:
            effective_directives = req.directives
            diagnostics_directives = req.directives
            enabled_toolsets = ["memory"]

        # 3. 实例化 AI 代理 (硬编码工具限制为 memory，完全封死危险操作)
        agent = AIAgent(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            api_mode="chat_completions",
            enabled_toolsets=enabled_toolsets,
            quiet_mode=True,
            platform="savana",
            session_db=session_db,
            session_id=session_id,
            load_soul_identity=True,
            skip_context_files=True,
            ephemeral_system_prompt=effective_directives,
            request_overrides=req.request_overrides,
            reasoning_config=req.reasoning_config,
        )
        agent.suppress_status_output = True
        if style_guard_enabled:
            # Offload unresolved turn reviews to a background task so pending turn audits
            # never block the interactive streaming chat response (prevents 45s multi-turn lag).
            _schedule_unresolved_reviews(
                profile_dir=profile_dir,
                raw_history=raw_history,
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=api_key,
                memory_store=getattr(agent, "_memory_store", None),
            )
            conversation_history = build_guarded_history(conversation_history, TurnReviewStore(profile_dir))
            agent._system_prompt_override = _build_style_guard_system_prompt(
                soul_text,
                memory_snapshots.get("memory", ""),
                memory_snapshots.get("user", ""),
                req.companion_directives,
            )
            agent._cached_system_prompt = agent._system_prompt_override

        # Savana 伴侣场景：覆盖 memory 工具描述为角色扮演专用中文版本
        # 原始通用描述是面向开发者的英文，DeepSeek 在角色扮演模式下无法关联到"记住用户偏好"这一触发场景
        _SAVANA_MEMORY_DESCRIPTION = (
            "将用户的个人信息永久写入记忆，跨会话持久保存。"
            "触发条件（以下情况必须立即调用，不得用台词代替）：\n"
            "- 用户说\"记住\"、\"帮我记\"、\"别忘了\"等要求记忆的话\n"
            "- 用户提到自己的喜好、厌恶、口味、习惯、运动、食物偏好\n"
            "- 用户透露职业、年龄、所在城市、作息规律等个人信息\n"
            "- 用户分享宠物、兴趣爱好、重要的人或事\n"
            "只写入 target='user'（用户档案），action='add' 新增，action='replace' 更新旧条目。"
            "content 用简短中文写明事实，例如：'用户最喜欢的运动是攀岩'、'用户讨厌吃芹菜和胡萝卜'。"
        )
        if not style_guard_enabled and agent.tools:
            for _tool in agent.tools:
                if isinstance(_tool, dict) and _tool.get("function", {}).get("name") == "memory":
                    _tool["function"]["description"] = _SAVANA_MEMORY_DESCRIPTION
                    break

        # Debug print to trace tools loading and memory settings
        print("[DEBUG] AIAgent initialized. model={}, provider={}, base_url={}, tools={}, skip_memory={}, memory_enabled={}, memory_store={}, directives={}".format(
            agent.model, agent.provider, agent.base_url,
            [t["function"]["name"] for t in agent.tools] if agent.tools else [],
            getattr(agent, "skip_memory", None),
            getattr(agent, "_memory_enabled", None),
            agent._memory_store is not None,
            repr(getattr(agent, "ephemeral_system_prompt", None))
        ))

        if style_guard_enabled:
            _log_style_guard_event(
                profile_dir=profile_dir,
                turn_id=req.message_id,
                policy=conversation_policy,
                status="started",
                elapsed_ms=0,
                model=model,
            )
        else:
            _log_companion_prompt_diagnostics(
                session_id=session_id,
                provider=provider,
                model=model,
                user_message=req.user_message,
                conversation_history=conversation_history,
                character_profile=req.character_profile,
                directives=diagnostics_directives,
                soul_text=soul_text,
                memory_text=memory_snapshots.get("memory", ""),
                user_profile_text=memory_snapshots.get("user", ""),
                session_stats=session_stats,
            )


        if req.stream:
            q = queue.Queue()
            review_metadata = {
                "review_status": "pending" if style_guard_enabled else None,
                "memory_status": "pending" if style_guard_enabled else None,
                "memory_modifications": [],
            }
            final_marker_consumed = threading.Event()
            producer_done = threading.Event()

            def _build_stream_metadata():
                modifications = review_metadata.get("memory_modifications", [])
                if not style_guard_enabled and not modifications:
                    memory_store = getattr(agent, "_memory_store", None)
                    if memory_store is not None and hasattr(memory_store, "modifications"):
                        modifications = memory_store.modifications
                metadata = {
                    "session_id": session_id,
                    "status": "completed",
                    "memory_modifications": modifications,
                }
                if style_guard_enabled:
                    metadata["review_status"] = review_metadata.get("review_status")
                    metadata["memory_status"] = review_metadata.get("memory_status")
                return metadata

            def run_chat_thread():
                token_thread = set_hermes_home_override(profile_dir)
                try:
                    def stream_callback(delta: str) -> None:
                        q.put(("token", delta))

                    # 触发对话生成
                    result = agent.run_conversation(
                        req.user_message,
                        conversation_history=conversation_history,
                        stream_callback=stream_callback,
                        platform_message_id=req.message_id,
                    )
                    final_reply = result.get("final_response", "")
                    q.put(("final", final_reply))

                    if style_guard_enabled:
                        _bg_mem_store = getattr(agent, "_memory_store", None)
                        threading.Thread(
                            target=_review_companion_turn,
                            kwargs={
                                "style_guard_enabled": style_guard_enabled,
                                "profile_dir": profile_dir,
                                "turn_id": req.message_id,
                                "assistant_text": final_reply,
                                "user_message": req.user_message,
                                "raw_history": raw_history,
                                "provider": provider,
                                "model": model,
                                "base_url": base_url,
                                "api_key": api_key,
                                "memory_store": _bg_mem_store,
                            },
                            daemon=True,
                        ).start()

                    q.put(("metadata", _build_stream_metadata()))
                    q.put(("end", None))
                except Exception as e:
                    q.put(("error", e))
                    q.put(("metadata", _build_stream_metadata()))
                    q.put(("end", None))
                finally:
                    try:
                        try:
                            reset_hermes_home_override(token_thread)
                        except Exception:
                            logger.exception("Failed to reset Hermes home override for companion stream thread")
                    finally:
                        producer_done.set()
                        _emit_stream_event("producer_done")

            _start_background_thread(run_chat_thread)

            def sse_generator() -> Generator[str, None, None]:
                while True:
                    item_type, item_value = q.get()
                    if item_type == "error":
                        _emit_stream_event("error")
                        yield "event: error\ndata: {}\n\n".format(json.dumps({"detail": str(item_value)}))
                        continue
                    if item_type == "token":
                        yield_data = {"delta": item_value}
                        _emit_stream_event("token", yield_data)
                        yield "event: token\ndata: {}\n\n".format(json.dumps(yield_data))
                        continue
                    if item_type == "final":
                        final_marker_consumed.set()
                        _emit_stream_event("final_marker_set")
                        continue
                    if item_type == "metadata":
                        _emit_stream_event("metadata", item_value)
                        yield "event: metadata\ndata: {}\n\n".format(json.dumps(item_value))
                        continue
                    if item_type == "end":
                        break

            response = _CompanionStreamResponse(
                sse_generator(),
                media_type="text/event-stream",
                final_marker_consumed=final_marker_consumed,
                producer_done=producer_done,
                session_lock=session_lock,
                session_lock_released_ref=session_lock_released_ref,
            )
            stream_lock_owned_by_response = True
            return response
        else:
            result = agent.run_conversation(
                req.user_message,
                conversation_history=conversation_history,
                platform_message_id=req.message_id,
            )
            reply = result.get("final_response", "")
            review_metadata = _review_companion_turn(
                style_guard_enabled=style_guard_enabled,
                profile_dir=profile_dir,
                turn_id=req.message_id,
                assistant_text=reply,
                user_message=req.user_message,
                raw_history=raw_history,
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=api_key,
                memory_store=getattr(agent, "_memory_store", None),
            )
            payload = {
                "reply": reply,
                "session_id": session_id,
                "memory_modifications": review_metadata.get("memory_modifications", []),
            }
            if style_guard_enabled:
                payload["review_status"] = review_metadata.get("review_status")
                payload["memory_status"] = review_metadata.get("memory_status")
            return JSONResponse(payload)
            
    finally:
        if session_lock is not None and not session_lock_released_ref["released"] and not stream_lock_owned_by_response:
            session_lock.release()
            session_lock_released_ref["released"] = True
        reset_hermes_home_override(token)

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    profile_dir = get_profile_path(session_id)
    if os.path.exists(profile_dir):
        # 使用 shutil.rmtree 删除物理 Profile 目录
        shutil.rmtree(profile_dir)
        return JSONResponse({"success": True, "message": "Session directory removed"})
    raise HTTPException(status_code=404, detail="Session profile not found")

@router.post("/sessions/{session_id}/memories")
async def sync_memory(session_id: str, req: MemorySyncRequest):
    profile_dir = get_profile_path(session_id)
    ensure_companion_profile(Path(profile_dir))
    
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    token = set_hermes_home_override(profile_dir)
    try:
        from tools.memory_tool import MemoryStore, memory_tool
        store = MemoryStore()
        store.load_from_disk()
        
        result_str = memory_tool(
            action=req.action,
            target=req.target,
            content=req.content,
            old_text=req.old_text,
            store=store
        )
        result = json.loads(result_str)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        reset_hermes_home_override(token)


@router.post("/weixin/qr/start")
async def start_weixin_qr(req: WeixinQrStartRequest):
    hermes_home = str(Path(get_profile_path(req.session_id)).parent.parent)
    try:
        if isinstance(req.character_profile, dict):
            profile_dir = get_profile_path(req.session_id)
            ensure_companion_profile(Path(profile_dir))
            sync_soul_file(profile_dir, req.character_profile)
        result = await _start_weixin_qr_session(
            hermes_home=hermes_home,
            session_id=req.session_id,
            bot_type=req.bot_type,
            timeout_seconds=req.timeout_seconds,
        )
        return JSONResponse(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/weixin/qr/status")
async def get_weixin_qr_status(req: WeixinQrStatusRequest):
    hermes_home = str(Path(get_profile_path(req.session_id)).parent.parent)
    try:
        result = await _query_weixin_qr_status(
            hermes_home=hermes_home,
            session_id=req.session_id,
            qrcode=req.qrcode,
            base_url=req.base_url,
        )
        if str(result.get("status") or "") == "confirmed":
            account_id = str(result.get("ilink_bot_id") or "")
            token = str(result.get("bot_token") or "")
            base_url = str(result.get("baseurl") or "")
            user_id = str(result.get("ilink_user_id") or "")
            if account_id and token:
                _persist_weixin_binding_account(
                    hermes_home,
                    account_id=account_id,
                    token=token,
                    base_url=base_url,
                    user_id=user_id,
                )
        return JSONResponse(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.delete("/sessions/{session_id}/messages/{message_id}")
async def delete_message(session_id: str, message_id: str):
    profile_dir = get_profile_path(session_id)
    
    # 动态覆盖 HOME 路径
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    token = set_hermes_home_override(profile_dir)
    
    try:
        from hermes_state import SessionDB
        db_path = Path(profile_dir) / "state.db"
        session_db = SessionDB(db_path=db_path)
        
        conn = session_db._conn
        cursor = conn.cursor()
        
        # 1. 查找对应 platform_message_id 消息的自增 id
        cursor.execute(
            "SELECT id FROM messages WHERE session_id = ? AND platform_message_id = ?",
            (session_id, message_id)
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Message not found in local db")
            
        msg_internal_id = row[0]
        
        # 2. 删除该消息及之后的所有消息 (撤回此消息及其产生的回复)
        cursor.execute(
            "DELETE FROM messages WHERE session_id = ? AND id >= ?",
            (session_id, msg_internal_id)
        )
        conn.commit()
        return JSONResponse({"success": True, "message": "Message and subsequent replies deleted from local db"})
        
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        reset_hermes_home_override(token)


class MessageUpdateRequest(BaseModel):
    content: str


@router.put("/sessions/{session_id}/messages/{message_id}")
async def update_message_endpoint(session_id: str, message_id: str, req: MessageUpdateRequest):
    profile_dir = get_profile_path(session_id)
    
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    token = set_hermes_home_override(profile_dir)
    
    session_db = None
    try:
        from hermes_state import SessionDB
        db_path = Path(profile_dir) / "state.db"
        session_db = SessionDB(db_path=db_path)

        updated = session_db.update_message_content(
            session_id=session_id,
            platform_message_id=message_id,
            content=req.content
        )
        
        if not updated:
            raise HTTPException(status_code=404, detail="Message not found in local db")
            
        return JSONResponse({"success": True, "message": "Message content updated in local db"})
        
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if session_db is not None:
            try:
                session_db.close()
            except Exception:
                pass
        reset_hermes_home_override(token)
