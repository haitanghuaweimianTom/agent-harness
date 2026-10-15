"""
多窗口独立会话与状态隔离测试 (Session Isolation & Management)
专门验证：用户 A 开窗口 1 (查天气记待办) 与窗口 2 (写周报记待办) 的完全隔离与随切随聊
"""
import unittest
from agent_harness.core.llm import MockLLMClient
from agent_harness.runtime.loop import AgentRuntime
from agent_harness.runtime.session import SessionManager
from agent_harness.tools.builtin.todo import TodoStore, TodoTool
from agent_harness.tools.builtin.weather import WeatherTool
from agent_harness.tools.builtin.calculator import CalculatorTool
from agent_harness.tools.registry import ToolRegistry


class TestSessionIsolation(unittest.TestCase):

    def setUp(self):
        self.session_mgr = SessionManager()
        self.registry = ToolRegistry()

        # 将 TodoTool 的 session getter 绑定到当前活动的 session_id
        def get_current_session_id():
            sess = self.session_mgr.get_active_session()
            return sess.state.session_id if sess else "default"

        self.registry.register(TodoTool(active_session_id_getter=get_current_session_id))
        self.registry.register(WeatherTool())
        self.registry.register(CalculatorTool())

        self.runtime = AgentRuntime(
            llm_client=MockLLMClient(),
            tool_registry=self.registry,
            session_manager=self.session_mgr,
            enable_console_log=False,
        )

        # 清空测试 store
        TodoStore.get_for_session("win_1_weather").clear()
        TodoStore.get_for_session("win_2_report").clear()

    def test_multi_window_independence(self):
        """
        验证双窗口完全独立，彼此不会影响，且能随时切回继续追问
        """
        # ==========================================
        # 1. 窗口 1 (win_1_weather): 查天气记待办
        # ==========================================
        resp1 = self.runtime.run(
            user_input="帮我查一下北京天气，并且把带雨伞写进待办",
            session_id="win_1_weather",
            user_id="user_A",
        )
        self.assertEqual(resp1["session_id"], "win_1_weather")
        self.assertIn("任务处理完成", resp1["answer"])

        # 检查窗口 1 的待办列表
        win1_todos = TodoStore.get_for_session("win_1_weather").list_all()
        self.assertEqual(len(win1_todos), 1)
        self.assertIn("雨伞", win1_todos[0]["title"])

        # 此时窗口 2 的待办列表必须为空！
        win2_todos = TodoStore.get_for_session("win_2_report").list_all()
        self.assertEqual(len(win2_todos), 0, "窗口 2 此时绝不应该受到窗口 1 的待办污染")

        # ==========================================
        # 2. 窗口 2 (win_2_report): 写周报记待办
        # ==========================================
        resp2 = self.runtime.run(
            user_input="今天写周报，添加待办：本周技术架构评审总结",
            session_id="win_2_report",
            user_id="user_A",
        )
        self.assertEqual(resp2["session_id"], "win_2_report")
        self.assertTrue("任务处理完成" in resp2["answer"] or "顺利完成" in resp2["answer"])

        # 检查窗口 2 的待办列表
        win2_todos = TodoStore.get_for_session("win_2_report").list_all()
        self.assertEqual(len(win2_todos), 1)
        self.assertIn("周报", win2_todos[0]["title"])

        # 检查窗口 1 的待办列表依然是 1 个，未被窗口 2 串味
        win1_todos = TodoStore.get_for_session("win_1_weather").list_all()
        self.assertEqual(len(win1_todos), 1)

        # ==========================================
        # 3. 用户 A 随时切回窗口 1 继续追问待办
        # ==========================================
        resp1_followup = self.runtime.run(
            user_input="查一下我刚才记的待办清单",
            session_id="win_1_weather",
            user_id="user_A",
        )
        self.assertEqual(resp1_followup["session_id"], "win_1_weather")
        # 窗口 1 返回的内容里包含雨伞待办
        self.assertIn("雨伞", resp1_followup["answer"])
        self.assertNotIn("周报", resp1_followup["answer"], "窗口 1 的回答中不应出现窗口 2 的周报待办")

        # ==========================================
        # 4. 用户 A 随时切回窗口 2 继续追问周报
        # ==========================================
        resp2_followup = self.runtime.run(
            user_input="查看有哪些待办任务",
            session_id="win_2_report",
            user_id="user_A",
        )
        self.assertEqual(resp2_followup["session_id"], "win_2_report")
        self.assertIn("周报", resp2_followup["answer"])
        self.assertNotIn("雨伞", resp2_followup["answer"], "窗口 2 的回答中不应出现窗口 1 的雨伞待办")

        # 验证会话列表元数据
        sessions = self.session_mgr.list_sessions(user_id="user_A")
        self.assertEqual(len(sessions), 2)


if __name__ == "__main__":
    unittest.main()
