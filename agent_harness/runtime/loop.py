"""
核心 Agent Runtime 与 ReAct 循环驱动引擎
从零手写实现，无任何第三方 Agent 框架依赖
"""
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
from agent_harness.core.llm import BaseLLMClient
from agent_harness.core.parser import LLMOutputParser
from agent_harness.core.prompt import build_system_prompt_with_tools
from agent_harness.core.types import Message, Role, ToolCall
from agent_harness.runtime.session import Session, SessionManager
from agent_harness.runtime.tracer import ExecutionTracer
from agent_harness.tools.registry import ToolRegistry


class AgentRuntime:
    """自研最小可用 Agent 核心运行时"""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        tool_registry: ToolRegistry,
        session_manager: Optional[SessionManager] = None,
        max_steps_per_turn: int = 5,
        enable_console_log: bool = True,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.max_steps_per_turn = max_steps_per_turn
        self.enable_console_log = enable_console_log

        # 生成并挂载包含所有已注册工具 Schema 的基础 System Prompt
        self.system_prompt = build_system_prompt_with_tools(self.tool_registry.get_schemas())
        if session_manager:
            self.session_manager = session_manager
            if not self.session_manager.default_system_prompt:
                self.session_manager.default_system_prompt = self.system_prompt
        else:
            self.session_manager = SessionManager(default_system_prompt=self.system_prompt)

    def run(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        user_id: str = "default_user",
    ) -> Dict[str, Any]:
        """
        执行完整的 Agent 循环交互
        Step 1: 接收用户输入
        Step 2: 判断是直接回复，还是调用工具
        Step 3: 调用工具并获取观察结果
        Step 4: 根据工具结果判断是继续 loop，还是返回结果给用户
        """
        # ==========================================================
        # Step 1: 接收用户输入与 Session 定位
        # ==========================================================
        if not session_id:
            active_sess = self.session_manager.get_active_session()
            if not active_sess:
                active_sess = self.session_manager.create_session(
                    user_id=user_id,
                    system_prompt=self.system_prompt,
                )
            session = active_sess
        else:
            session = self.session_manager.get_or_create_session(
                session_id=session_id,
                user_id=user_id,
                title=f"Session_{session_id}",
                system_prompt=self.system_prompt,
            )
            self.session_manager.switch_active_session(session_id)

        session.touch()
        ctx = session.context_manager
        tracer = ExecutionTracer(session_id=session.state.session_id, enable_console=self.enable_console_log)

        if self.enable_console_log:
            print(f"\n\033[1;32m=== [Session: {session.state.session_id}] 收到用户输入 ===\033[0m")
            print(f"用户: {user_input}")

        # 将用户输入追加到 Context 中
        ctx.add_user_message(user_input)

        step_count = 0
        final_answer: Optional[str] = None

        # ==========================================================
        # 核心 Loop: Step 2 ~ Step 4
        # ==========================================================
        while step_count < self.max_steps_per_turn:
            step_count += 1
            step_trace = tracer.start_step(step_count)

            # 获取当前会话上下文并请求 LLM
            messages = ctx.get_messages()
            tools_schema = self.tool_registry.get_schemas()

            try:
                llm_resp = self.llm_client.generate(
                    messages=messages,
                    tools_schema=tools_schema,
                )
            except Exception as e:
                # 异常处理：LLM API 调用失败
                err_msg = f"LLM 调用异常: {str(e)}"
                tracer.record_thought(step_trace, f"发生异常: {err_msg}")
                final_answer = f"抱歉，系统在思考时遇到故障: {err_msg}"
                tracer.record_final_answer(step_trace, final_answer)
                ctx.add_assistant_message(content=final_answer)
                break

            raw_text = llm_resp.get("content") or ""
            native_tool_calls = llm_resp.get("tool_calls")
            step_trace.raw_response = raw_text

            # Step 2: 解析模型响应，判断是直接回复还是调用工具
            thought, tool_calls, candidate_answer = LLMOutputParser.parse(
                raw_text=raw_text,
                native_tool_calls=native_tool_calls,
            )

            tracer.record_thought(step_trace, thought)

            # 判断分支 A: 模型决定直接回复用户 (无需调用工具)
            if not tool_calls and candidate_answer:
                final_answer = candidate_answer
                tracer.record_final_answer(step_trace, final_answer)
                ctx.add_assistant_message(
                    content=final_answer,
                    thought=thought,
                )
                break

            # 判断分支 B: 模型决定调用工具
            if tool_calls:
                tracer.record_tool_calls(step_trace, tool_calls)

                # 将当前 Assistant 的工具调用意图写入上下文
                ctx.add_assistant_message(
                    content=raw_text,
                    thought=thought,
                    tool_calls=tool_calls,
                )

                # Step 3: 执行工具并收集 Observation
                for tc in tool_calls:
                    tool_result = self.tool_registry.execute_tool_call(tc)
                    tracer.record_tool_result(step_trace, tool_result)

                    # 将 Observation 写入 Context (Role.TOOL)
                    obs_content = tool_result.content if tool_result.success else f"Error: {tool_result.error}"
                    ctx.add_tool_result(
                        tool_call_id=tc.id,
                        name=tc.name,
                        result_content=obs_content,
                    )

                # Step 4: 根据工具结果判断是继续 loop 还是返回
                # 工具执行完毕后，进入下一轮循环 (下一步由 LLM 观察结果后决定是否继续调用或交付答复)
                continue

            # 容错兜底：既无工具调用也未提取出最终回答
            if not final_answer:
                final_answer = raw_text or "已为您处理完成。"
                tracer.record_final_answer(step_trace, final_answer)
                ctx.add_assistant_message(content=final_answer, thought=thought)
                break

        # 超过最大轮次保护
        if not final_answer and step_count >= self.max_steps_per_turn:
            final_answer = "已达到单轮最大工具调用步数上限，操作已部分执行，请查验当前状态。"
            if step_trace:
                tracer.record_final_answer(step_trace, final_answer)
            ctx.add_assistant_message(content=final_answer)

        trace_summary = tracer.finish_trace()

        return {
            "session_id": session.state.session_id,
            "answer": final_answer,
            "steps_taken": step_count,
            "trace": trace_summary,
            "total_user_turns": ctx.count_user_turns(),
        }
