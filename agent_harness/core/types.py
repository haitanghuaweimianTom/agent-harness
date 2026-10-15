"""
Agent 核心数据类型定义 (零框架依赖)
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """工具调用请求结构体"""
    id: str
    name: str
    arguments: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResult:
    """工具执行结果结构体"""
    tool_call_id: str
    name: str
    content: str
    success: bool = True
    error: Optional[str] = None
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Message:
    """统一消息结构体，兼顾原生标准与 ReAct 格式"""
    role: Role
    content: str
    thought: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "role": self.role.value if isinstance(self.role, Role) else self.role,
            "content": self.content,
        }
        if self.thought:
            d["thought"] = self.thought
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class AgentStep:
    """单步执行轨迹"""
    step_num: int
    thought: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    final_answer: Optional[str] = None
    is_finished: bool = False
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_num": self.step_num,
            "thought": self.thought,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "tool_results": [tr.to_dict() for tr in self.tool_results],
            "final_answer": self.final_answer,
            "is_finished": self.is_finished,
        }


@dataclass
class SessionState:
    """会话窗口状态元数据"""
    session_id: str
    user_id: str
    title: str = "New Session"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
