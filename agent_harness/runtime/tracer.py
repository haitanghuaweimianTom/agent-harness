"""
工具调用追踪与执行日志记录器 (Tracer)
记录 Agent Loop 全链路轨迹、性能耗时与执行状态
"""
from __future__ import annotations
import json
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional
from agent_harness.core.types import AgentStep, ToolCall, ToolResult


class ExecutionTracer:
    """Agent 执行追踪器"""

    def __init__(self, session_id: str = "default", enable_console: bool = True):
        self.session_id = session_id
        self.enable_console = enable_console
        self.steps: List[AgentStep] = []
        self.start_time: float = time.time()
        self.end_time: Optional[float] = None

    def start_step(self, step_num: int) -> AgentStep:
        step = AgentStep(step_num=step_num)
        self.steps.append(step)
        if self.enable_console:
            print(f"\n\033[1;36m[Step {step_num}]\033[0m 开始规划...")
        return step

    def record_thought(self, step: AgentStep, thought: Optional[str]) -> None:
        step.thought = thought
        if self.enable_console and thought:
            print(f"  \033[1;33m💭 [Thought]:\033[0m {thought}")

    def record_tool_calls(self, step: AgentStep, tool_calls: List[ToolCall]) -> None:
        step.tool_calls.extend(tool_calls)
        if self.enable_console:
            for tc in tool_calls:
                args_preview = json.dumps(tc.arguments, ensure_ascii=False)
                if len(args_preview) > 80:
                    args_preview = args_preview[:77] + "..."
                print(f"  \033[1;34m🔧 [Tool Call]:\033[0m {tc.name} -> {args_preview}")

    def record_tool_result(self, step: AgentStep, tool_result: ToolResult) -> None:
        step.tool_results.append(tool_result)
        if self.enable_console:
            status_icon = "✅" if tool_result.success else "❌"
            content_preview = (tool_result.content or tool_result.error or "")[:90].replace("\n", " ")
            print(
                f"  {status_icon} \033[1;32m[Observation ({tool_result.name}) - {tool_result.execution_time_ms:.1f}ms]:\033[0m {content_preview}"
            )

    def record_final_answer(self, step: AgentStep, final_answer: str) -> None:
        step.final_answer = final_answer
        step.is_finished = True
        self.end_time = time.time()
        if self.enable_console:
            print(f"  \033[1;35m🎯 [Final Answer]:\033[0m {final_answer[:120]}...\n")

    def finish_trace(self) -> Dict[str, Any]:
        if not self.end_time:
            self.end_time = time.time()
        total_duration_ms = (self.end_time - self.start_time) * 1000
        return {
            "session_id": self.session_id,
            "total_steps": len(self.steps),
            "total_duration_ms": round(total_duration_ms, 2),
            "steps": [s.to_dict() for s in self.steps],
        }

    def export_json(self) -> str:
        return json.dumps(self.finish_trace(), ensure_ascii=False, indent=2)
