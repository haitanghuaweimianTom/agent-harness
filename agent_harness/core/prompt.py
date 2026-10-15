"""
Agent 提示词工程与协议规范
"""
from __future__ import annotations
import json
from typing import Any, Dict, List


DEFAULT_SYSTEM_PROMPT = """你是一个智能、可靠且具备自主工具调用能力的 AI Assistant。
你能够理解用户的复杂意图，并在必要时自主决策调用已注册的外部工具来获取实时信息或执行操作。

【运行机制与交互规范】
在面对用户的请求时，你应当按照如下步骤进行思考与决策：
1. 意图分析：理解用户的问题。如果是日常闲聊或已有充分知识回答，直接给出清晰答复。
2. 工具决策：如果需要外部真实数据（如数学计算、查询天气、管理待办事项、搜索技术文档等），应当调用相应的工具。
3. 思考与格式输出：
   - 你必须先在 `<thought>` 标签内清晰写出你的思考过程，阐述为什么需要调用工具、选什么工具及参数。
   - 如果需要调用工具，请在 `<tool_call>` 标签内输出标准 JSON 格式的工具调用（支持单个或按需调用）。
     格式：
     <thought>这里阐述思考分析</thought>
     <tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>
   - 如果信息已经充分，不需要再调用工具，请直接在 `<final_answer>` 标签内输出给用户的最终答复：
     <thought>分析当前观察到的结果，确认已满足用户需求</thought>
     <final_answer>这里是直接回复给用户的最终答案</final_answer>

【重要原则】
- 工具调用的参数必须严格符合工具的 Schema 规范。
- 严禁臆造不存在的工具或随意篡改参数名称。
- 当工具返回 Observation（观察结果）后，你会再次审视结果，如需进一步处理可继续调用工具，若已解决则输出 <final_answer>。
- 始终保持友好、专业、实事求是的语气。
"""


def build_system_prompt_with_tools(tools_schema: List[Dict[str, Any]], custom_prompt: str = "") -> str:
    """构建注入了工具 Schema 的完整 System Prompt"""
    base = custom_prompt or DEFAULT_SYSTEM_PROMPT

    tools_doc_list = []
    for t in tools_schema:
        fn = t.get("function", t)
        name = fn.get("name")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        tools_doc_list.append(f"- 工具名称: `{name}`\n  说明: {desc}\n  参数 Schema: {json.dumps(params, ensure_ascii=False)}")

    tools_section = "\n\n".join(tools_doc_list)

    return f"""{base}

【当前可使用的工具列表】
{tools_section}
"""
