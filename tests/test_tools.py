"""
工具模块单元测试 (ToolRegistry & Builtin Tools)
"""
import unittest
from agent_harness.tools.base import tool
from agent_harness.tools.builtin.calculator import CalculatorTool
from agent_harness.tools.builtin.todo import TodoTool, TodoStore
from agent_harness.tools.builtin.weather import WeatherTool
from agent_harness.tools.builtin.search import SearchTool
from agent_harness.tools.registry import ToolRegistry
from agent_harness.core.types import ToolCall


class TestTools(unittest.TestCase):

    def setUp(self):
        self.registry = ToolRegistry()
        # 清理待办状态
        TodoStore.get_for_session("test_sess").clear()

    def test_calculator(self):
        calc = CalculatorTool()
        self.registry.register(calc)

        # 基础运算
        res = calc.execute(expression="12 * (34 + 56)")
        self.assertTrue(res["success"])
        self.assertEqual(res["result"], 1080)

        # 复杂函数
        res = calc.execute(expression="sqrt(144) + 2**3")
        self.assertTrue(res["success"])
        self.assertEqual(res["result"], 20)

        # 除零保护
        res = calc.execute(expression="10 / 0")
        self.assertFalse(res["success"])
        self.assertIn("除零错误", res["error"])

        # 代码注入防护 (eval 防御)
        res = calc.execute(expression="__import__('os').system('ls')")
        self.assertFalse(res["success"])

    def test_todo_tool(self):
        todo = TodoTool(active_session_id_getter=lambda: "test_sess")
        self.registry.register(todo)

        # 添加待办
        add_res = todo.execute(action="add", title="准备 Agent 面试代码", priority="high")
        self.assertTrue(add_res["success"])
        item_id = add_res["todo"]["id"]

        # 查询待办
        list_res = todo.execute(action="list")
        self.assertTrue(list_res["success"])
        self.assertEqual(list_res["total"], 1)

        # 完成待办
        comp_res = todo.execute(action="complete", todo_id=item_id)
        self.assertTrue(comp_res["success"])
        self.assertEqual(comp_res["todo"]["status"], "completed")

        # 状态筛选查询
        pending_list = todo.execute(action="list", status_filter="pending")
        self.assertEqual(pending_list["total"], 0)

        completed_list = todo.execute(action="list", status_filter="completed")
        self.assertEqual(completed_list["total"], 1)

    def test_weather_and_search(self):
        weather = WeatherTool()
        w_res = weather.execute(city="北京", date="今天")
        self.assertTrue(w_res["success"])
        self.assertIn("北京", w_res["city"])
        self.assertIn("°C", w_res["temperature"])

        search = SearchTool()
        s_res = search.execute(query="JEV 决策模型")
        self.assertTrue(s_res["success"])
        self.assertGreater(s_res["results_count"], 0)

    def test_custom_tool_decorator(self):
        @tool(name="reverse_text", description="将输入的字符串倒序排列")
        def reverse_text(text: str) -> str:
            return text[::-1]

        self.registry.register(reverse_text)
        schema = reverse_text.to_schema()
        self.assertEqual(schema["function"]["name"], "reverse_text")

        # 通过 ToolRegistry 执行
        call = ToolCall(id="call_1", name="reverse_text", arguments={"text": "hello agent"})
        result = self.registry.execute_tool_call(call)
        self.assertTrue(result.success)
        self.assertEqual(result.content, "tnega olleh")


if __name__ == "__main__":
    unittest.main()
