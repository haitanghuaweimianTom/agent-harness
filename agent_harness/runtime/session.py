"""
多会话管理器 (SessionManager)
实现多窗口独立 Session 隔离、无缝切换与状态持久化
"""
from __future__ import annotations
import time
import uuid
from typing import Any, Dict, List, Optional
from agent_harness.core.context_manager import ContextManager
from agent_harness.core.types import SessionState


class Session:
    """单个独立的会话窗口"""

    def __init__(
        self,
        session_id: str,
        user_id: str = "default_user",
        title: str = "新会话",
        system_prompt: str = "",
        max_turns: int = 10,
    ):
        self.state = SessionState(
            session_id=session_id,
            user_id=user_id,
            title=title,
            created_at=time.time(),
            updated_at=time.time(),
        )
        self.context_manager = ContextManager(
            system_prompt=system_prompt,
            max_turns=max_turns,
        )
        self.custom_data: Dict[str, Any] = {}

    def touch(self) -> None:
        self.state.updated_at = time.time()


class SessionManager:
    """会话管理器，提供多会话独立隔离与调度"""

    def __init__(self, default_system_prompt: str = ""):
        self.default_system_prompt = default_system_prompt
        self._sessions: Dict[str, Session] = {}
        self._active_session_id: Optional[str] = None

    def create_session(
        self,
        session_id: Optional[str] = None,
        user_id: str = "user_a",
        title: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Session:
        sid = session_id or f"sess_{uuid.uuid4().hex[:8]}"
        sess_title = title or f"窗口 {len(self._sessions) + 1}"
        sys_prompt = system_prompt if system_prompt is not None else self.default_system_prompt

        session = Session(
            session_id=sid,
            user_id=user_id,
            title=sess_title,
            system_prompt=sys_prompt,
        )
        self._sessions[sid] = session
        self._active_session_id = sid
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def get_or_create_session(
        self,
        session_id: str,
        user_id: str = "user_a",
        title: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Session:
        if session_id in self._sessions:
            return self._sessions[session_id]
        return self.create_session(session_id=session_id, user_id=user_id, title=title, system_prompt=system_prompt)

    def switch_active_session(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            raise KeyError(f"会话不存在: {session_id}")
        self._active_session_id = session_id
        return self._sessions[session_id]

    def get_active_session(self) -> Optional[Session]:
        if self._active_session_id and self._active_session_id in self._sessions:
            return self._sessions[self._active_session_id]
        return None

    def list_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        result = []
        for sid, sess in self._sessions.items():
            if user_id and sess.state.user_id != user_id:
                continue
            result.append({
                "session_id": sid,
                "user_id": sess.state.user_id,
                "title": sess.state.title,
                "turns_count": sess.context_manager.count_user_turns(),
                "created_at": sess.state.created_at,
                "updated_at": sess.state.updated_at,
                "is_active": (sid == self._active_session_id),
            })
        return result

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            if self._active_session_id == session_id:
                self._active_session_id = next(iter(self._sessions.keys())) if self._sessions else None
            return True
        return False
