"""
上下文管理与压缩测试 (ContextManager & Sliding Compression)
"""
import unittest
from agent_harness.core.context_manager import ContextManager
from agent_harness.core.types import Message, Role


class TestContextManagement(unittest.TestCase):

    def test_basic_turns_and_memory(self):
        ctx = ContextManager(system_prompt="You are an assistant.", max_turns=5)
        # 初始只有 1 条 System
        self.assertEqual(len(ctx.get_messages()), 1)
        self.assertEqual(ctx.count_user_turns(), 0)

        # 追加用户输入与回复
        ctx.add_user_message("你好")
        ctx.add_assistant_message("你好！有什么我可以帮你的？", thought="分析问候")
        self.assertEqual(ctx.count_user_turns(), 1)
        self.assertEqual(len(ctx.get_messages()), 3)

        # 追问
        ctx.add_user_message("帮我推荐一本书")
        ctx.add_assistant_message("推荐《设计数据密集型应用》")
        self.assertEqual(ctx.count_user_turns(), 2)

    def test_auto_compression_when_exceeding_turns(self):
        # 设置 max_turns=3，测试超过后自动压缩
        ctx = ContextManager(
            system_prompt="System Instruction",
            max_turns=3,
            max_tokens_budget=10000,
            enable_auto_compression=True,
        )

        # 模拟前 4 轮对话 (每轮带工具调用和结果)
        for i in range(1, 5):
            ctx.add_user_message(f"任务 {i}: 计算 {i} + 10")
            ctx.add_assistant_message(f"准备计算", thought=f"计算第{i}步", tool_calls=[{"name": "calculator"}])
            ctx.add_tool_result(f"call_{i}", "calculator", f"{i + 10}")
            ctx.add_assistant_message(f"计算结果是 {i + 10}")

        # 检查是否成功触发压缩
        self.assertGreater(ctx.total_compressions, 0)

        messages = ctx.get_messages()
        # 验证第一条仍然是原始 System Prompt
        self.assertEqual(messages[0].role, Role.SYSTEM)
        self.assertEqual(messages[0].content, "System Instruction")

        # 验证第二条是前情摘要消息
        self.assertEqual(messages[1].role, Role.SYSTEM)
        self.assertIn("【前情会话提要与状态汇总】", messages[1].content)
        self.assertIn("任务 1", messages[1].content)

        # 验证最新几轮的高保真消息得到了保留
        last_user_msg = [m for m in messages if m.role == Role.USER][-1]
        self.assertIn("任务 4", last_user_msg.content)

    def test_tool_followup_context_continuity(self):
        """测试带工具的追问，上下文完整保留 Observation 供下一步使用"""
        ctx = ContextManager(system_prompt="System Prompt", max_turns=10)

        # 第 1 步：查天气
        ctx.add_user_message("查询北京今天天气")
        ctx.add_assistant_message("调用天气工具", tool_calls=[{"name": "weather"}])
        ctx.add_tool_result("call_w1", "weather", '{"city": "北京", "condition": "晴", "temp": "26°C"}')
        ctx.add_assistant_message("北京今天晴天，气温26度。")

        # 第 2 步：追问——“基于刚才的天气，给我推荐穿衣建议”
        ctx.add_user_message("基于刚才的天气，给我推荐穿衣建议")

        all_msgs = ctx.get_messages()
        # 确认当前上下文中包含了前一步的观察结果
        obs_msgs = [m for m in all_msgs if m.role == Role.TOOL]
        self.assertEqual(len(obs_msgs), 1)
        self.assertIn("26°C", obs_msgs[0].content)


if __name__ == "__main__":
    unittest.main()
