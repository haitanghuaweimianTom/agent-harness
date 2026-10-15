"""
上下文有效管理器 (ContextManager)
实现轮次限制、追问状态保持、滑动窗口截断与增量摘要压缩
"""
from __future__ import annotations
import copy
from typing import Any, Callable, Dict, List, Optional
from agent_harness.core.types import Message, Role


class ContextManager:
    """会话上下文与窗口管理器"""

    def __init__(
        self,
        system_prompt: str,
        max_turns: int = 10,
        max_tokens_budget: int = 4000,
        enable_auto_compression: bool = True,
    ):
        """
        :param system_prompt: 固定的系统角色与工具说明
        :param max_turns: 触发压缩或截断的最大用户轮次数
        :param max_tokens_budget: 字符/token 预算上限
        :param enable_auto_compression: 是否在超出窗口时执行滚动摘要压缩
        """
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.max_tokens_budget = max_tokens_budget
        self.enable_auto_compression = enable_auto_compression

        # 核心上下文序列 (首条为 System)
        self._messages: List[Message] = [
            Message(role=Role.SYSTEM, content=self.system_prompt)
        ]
        # 累积的历史摘要 (用于滑动压缩)
        self._history_summary: Optional[str] = None
        # 统计指标
        self.total_compressions: int = 0

    def add_message(self, message: Message) -> None:
        """追加一条消息"""
        self._messages.append(message)
        # 每次追加后进行预算检查与压缩
        if self.enable_auto_compression:
            self.maybe_compress()

    def add_user_message(self, content: str) -> None:
        self.add_message(Message(role=Role.USER, content=content))

    def add_assistant_message(
        self,
        content: str,
        thought: Optional[str] = None,
        tool_calls: Optional[List[Any]] = None,
    ) -> None:
        self.add_message(Message(
            role=Role.ASSISTANT,
            content=content,
            thought=thought,
            tool_calls=tool_calls,
        ))

    def add_tool_result(self, tool_call_id: str, name: str, result_content: str) -> None:
        self.add_message(Message(
            role=Role.TOOL,
            content=result_content,
            tool_call_id=tool_call_id,
            name=name,
        ))

    def get_messages(self) -> List[Message]:
        """获取当前提供给 LLM 的全量上下文序列"""
        return list(self._messages)

    def count_user_turns(self) -> int:
        """计算当前上下文中包含的用户输入轮次数"""
        return sum(1 for m in self._messages if m.role == Role.USER)

    def estimate_tokens(self) -> int:
        """粗粒度估算当前上下文的 token 数 (以字符/3.5 估算)"""
        total_chars = sum(len(m.content or "") + len(m.thought or "") for m in self._messages)
        return int(total_chars / 3.5)

    def maybe_compress(self) -> bool:
        """
        检查是否需要进行基础上下文压缩
        如果超出最大轮次限制或 token 预算，采用工业级'保留首尾、摘要中间'策略：
        1. 保持 System 消息不变 (Index 0)
        2. 保留最近保留的 K 轮消息 (默认保留最近 3 轮完整对话)
        3. 将中间较老的交互压缩为一段语义概括标签 <context_summary>
        """
        user_turn_count = self.count_user_turns()
        est_tokens = self.estimate_tokens()

        if user_turn_count <= self.max_turns and est_tokens <= self.max_tokens_budget:
            return False

        # 需要压缩：保留最近 3 轮（大约 6~12 条消息，包含 user, assistant, tool）
        preserve_recent_turns = 3
        # 寻找最近 preserve_recent_turns 个 user message 的起始索引
        user_indices = [i for i, m in enumerate(self._messages) if m.role == Role.USER]
        if len(user_indices) <= preserve_recent_turns:
            return False

        split_index = user_indices[-preserve_recent_turns]

        # 需要被压缩的老消息集合 (跳过 index 0 的 system 消息)
        old_messages = self._messages[1:split_index]
        if not old_messages:
            return False

        # 生成滚动压缩摘要
        summary_text = self._compress_messages_to_summary(old_messages)

        # 组装新的消息序列
        new_messages: List[Message] = [self._messages[0]]  # 保持 System Prompt

        # 插入摘要节点 (以 System 附加上下文或带有标记的 User 上下文注入)
        summary_msg = Message(
            role=Role.SYSTEM,
            content=f"【前情会话提要与状态汇总】\n{summary_text}",
        )
        new_messages.append(summary_msg)

        # 追加最近 preserve_recent_turns 轮未经修改的高保真消息
        new_messages.extend(self._messages[split_index:])

        self._messages = new_messages
        self.total_compressions += 1
        return True

    def _compress_messages_to_summary(self, messages_to_compress: List[Message]) -> str:
        """
        轻量高效的确定性摘要编译器
        提取用户目标、已调用的工具记录及关键成果，避免再次调用昂贵模型
        """
        actions = []
        user_intents = []
        tool_results_brief = []

        for m in messages_to_compress:
            if m.role == Role.USER:
                user_intents.append(m.content[:80])
            elif m.role == Role.ASSISTANT:
                if m.tool_calls:
                    for tc in m.tool_calls:
                        name = tc.name if hasattr(tc, "name") else tc.get("name")
                        actions.append(f"调用工具[{name}]")
            elif m.role == Role.TOOL:
                name = m.name or "tool"
                brief = (m.content or "")[:60].replace("\n", " ")
                tool_results_brief.append(f"{name}返回: {brief}")

        summary_parts = []
        if self._history_summary:
            summary_parts.append(f"先前历史：{self._history_summary}")

        if user_intents:
            summary_parts.append(f"用户早期需求涉及：{'; '.join(user_intents[-3:])}")
        if actions:
            summary_parts.append(f"曾执行的操作：{', '.join(set(actions))}")
        if tool_results_brief:
            summary_parts.append(f"产生的结果关键数据：{'; '.join(tool_results_brief[-3:])}")

        consolidated = " | ".join(summary_parts)
        self._history_summary = consolidated
        return consolidated

    def clear_history_keep_system(self) -> None:
        """清空历史仅保留系统提示"""
        self._messages = [Message(role=Role.SYSTEM, content=self.system_prompt)]
        self._history_summary = None
