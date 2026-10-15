"""
核心 Agent Runtime 循环状态机测试 (Agent Loop & Tracer)
"""
import unittest
from agent_harness.core.llm import MockLLMClient
from agent_harness.runtime.loop import AgentRuntime
from agent_harness.runtime.session import SessionManager
from agent_harness.tools.builtin.calculator import CalculatorTool
from agent_harness.tools.builtin.todo import TodoStore, TodoTool
from agent_harness.tools.builtin.weather import WeatherTool
from agent_harness.tools.builtin.search import SearchTool
from agent_harness.tools.registry import ToolRegistry


class TestRuntimeLoop(unittest.TestCase):

    def setUp(self):
        self.registry = ToolRegistry()
        self.session_mgr = SessionManager()

        def get_current_sid():
            s = self.session_mgr.get_active_session()
            return s.state.session_id if s else "test_runtime"

        self.registry.register(TodoTool(active_session_id_getter=get_current_sid))
        self.registry.register(CalculatorTool())
        self.registry.register(WeatherTool())
        self.registry.register(SearchTool())

        self.runtime = AgentRuntime(
            llm_client=MockLLMClient(),
            tool_registry=self.registry,
            session_manager=self.session_mgr,
            enable_console_log=False,
        )

        TodoStore.get_for_session("test_runtime").clear()

    def test_direct_reply_without_tools(self):
        """测试无需工具的直接问答，1 步即产出结果"""
        res = self.runtime.run("你好，请介绍一下你自己", session_id="test_runtime")
        self.assertEqual(res["steps_taken"], 1)
        self.assertIn("AI 助手", res["answer"])
        self.assertTrue(res["trace"]["steps"][0]["is_finished"])
        self.assertEqual(len(res["trace"]["steps"][0]["tool_calls"]), 0)

    def test_single_tool_execution_flow(self):
        """测试单工具调用 (Step 1 提问 -> Step 2 决策调用 -> Step 3 执行工具 -> Step 4 生成总结)"""
        res = self.runtime.run("算一下 12 * (34 + 56) 等于多少", session_id="test_runtime")
        self.assertEqual(res["steps_taken"], 2)
        self.assertIn("1080", res["answer"])

        # 检查 trace
        step1 = res["trace"]["steps"][0]
        self.assertEqual(len(step1["tool_calls"]), 1)
        self.assertEqual(step1["tool_calls"][0]["name"], "calculator")
        self.assertEqual(len(step1["tool_results"]), 1)
        self.assertTrue(step1["tool_results"][0]["success"])

        step2 = res["trace"]["steps"][1]
        self.assertTrue(step2["is_finished"])
        self.assertIsNotNone(step2["final_answer"])

    def test_chained_multi_tool_execution(self):
        """测试级联工具调用 (查天气并记待办)"""
        res = self.runtime.run("帮我查一下北京天气，并且把带雨伞写进待办", session_id="test_runtime")
        # 应该连续调用 weather 和 todo 工具
        self.assertGreaterEqual(res["steps_taken"], 2)
        self.assertIn("任务处理完成", res["answer"])

        # 验证 Todo 已经持久化成功
        todos = TodoStore.get_for_session("test_runtime").list_all()
        self.assertGreaterEqual(len(todos), 1)
        self.assertIn("雨伞", todos[0]["title"])

    def test_trace_duration_and_structure(self):
        """测试 Trace 耗时统计与 JSON 结构完整性"""
        res = self.runtime.run("搜索 JEV 架构", session_id="test_runtime")
        trace = res["trace"]
        self.assertIn("session_id", trace)
        self.assertIn("total_steps", trace)
        self.assertIn("total_duration_ms", trace)
        self.assertGreater(trace["total_duration_ms"], 0)


if __name__ == "__main__":
    unittest.main()
