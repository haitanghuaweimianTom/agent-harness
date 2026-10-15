"""
原生 LLM 客户端实现 (支持真实 OpenAI-compatible API 与 Mock Client)
零外部重型依赖，完全基于 Python 标准库 urllib.request
"""
from __future__ import annotations
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional
from agent_harness.core.types import Message


class BaseLLMClient:
    """LLM 客户端抽象基类"""
    def generate(
        self,
        messages: List[Message],
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        生成响应，返回结构:
        {
            "content": str,
            "tool_calls": Optional[List[Dict]],
            "raw_response": Any
        }
        """
        raise NotImplementedError


class OpenAILikeClient(BaseLLMClient):
    """支持任何 OpenAI-compatible 端点 (DeepSeek, Qwen, GLM, OpenAI 等) 的轻量客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
        mode: str = "react",  # "react" (通用纯文本ReAct标签模式) 或 "native" (OpenAI 原生 tool_calls)
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or ""
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.timeout = timeout
        self.mode = mode

    def generate(
        self,
        messages: List[Message],
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError(
                "未检测到有效的 API Key。请设置环境变量 OPENAI_API_KEY 或在初始化时传入 api_key。"
            )

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # 格式化消息体 (自适应 react 模式与 native 模式)
        formatted_msgs = []
        for m in messages:
            role_val = m.role.value if hasattr(m.role, "value") else str(m.role)

            if self.mode == "react" and role_val == "tool":
                # 在 ReAct 纯文本模式下，将 tool 角色转为用户观察，彻底兼容 Gemini 等各类 API 代理
                formatted_msgs.append({
                    "role": "user",
                    "content": f"【工具执行结果 Observation ({m.name or 'tool'})】:\n{m.content}",
                })
                continue

            msg_dict: Dict[str, Any] = {
                "role": role_val,
                "content": m.content or "",
            }
            if self.mode == "native":
                if m.tool_calls:
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else str(tc.arguments),
                            },
                        }
                        for tc in m.tool_calls
                    ]
                if m.tool_call_id:
                    msg_dict["tool_call_id"] = m.tool_call_id
                if m.name:
                    msg_dict["name"] = m.name

            formatted_msgs.append(msg_dict)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted_msgs,
            "temperature": temperature,
        }

        # 仅在 native 模式且传入 schema 时启用 OpenAI 原生 tools 字段
        if self.mode == "native" and tools_schema:
            payload["tools"] = tools_schema
            payload["tool_choice"] = "auto"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                res_json = json.loads(body)
                choice = res_json["choices"][0]["message"]

                content = choice.get("content") or ""
                reasoning = choice.get("reasoning_content") or ""

                # 智能融合 content 与 reasoning_content (针对 Gemini / DeepSeek-R1 等)
                if reasoning and not content:
                    final_text = reasoning
                elif reasoning and content and reasoning not in content:
                    final_text = f"<thought>\n{reasoning}\n</thought>\n{content}"
                else:
                    final_text = content

                return {
                    "content": final_text,
                    "tool_calls": choice.get("tool_calls") if self.mode == "native" else None,
                    "raw_response": res_json,
                }
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM API HTTP Error {e.code}: {err_body}") from e
        except Exception as e:
            raise RuntimeError(f"LLM API Network Error: {str(e)}") from e


class MockLLMClient(BaseLLMClient):
    """
    智能确定性 Mock LLM 客户端
    用于无 API Key 或离线自动化验证环境，覆盖真实完整的多轮对话与工具调用生命周期
    """

    def __init__(self):
        self.call_history: List[List[Message]] = []

    def generate(
        self,
        messages: List[Message],
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        self.call_history.append(messages)
        last_msg = messages[-1]

        # 1. 如果上一条是工具执行结果 (Role.TOOL)
        if (last_msg.role.value if hasattr(last_msg.role, "value") else str(last_msg.role)) == "tool":
            tool_content = last_msg.content
            tool_name = last_msg.name or ""

            # 判断前序工具并根据观察结果决定是继续还是结束
            # 检查上下文里是否有待后续执行的组合任务（如“查天气并且记待办”）
            user_intent = ""
            for m in reversed(messages):
                r = m.role.value if hasattr(m.role, "value") else str(m.role)
                if r == "user":
                    user_intent = m.content
                    break

            # 场景：如果是天气查询刚返回，且用户说要“查天气并把带伞写进待办”
            if tool_name == "weather" and ("待办" in user_intent or "todo" in user_intent.lower()):
                # 如果尚未执行 todo 工具，则继续调用 todo 工具
                already_called_todo = any(m.name == "todo" for m in messages)
                if not already_called_todo:
                    return {
                        "content": (
                            "<thought>天气查询已完成。根据用户要求，现在需要将相关提醒事项记入待办事项清单。</thought>\n"
                            "<tool_call>{\"name\": \"todo\", \"arguments\": {\"action\": \"add\", \"title\": \"明天出门带雨伞（根据天气预报）\", \"priority\": \"high\"}}</tool_call>"
                        ),
                        "tool_calls": None,
                        "raw_response": {},
                    }

            # 最终总结回复
            thought = f"工具 {tool_name} 执行完毕，获取到结果：{tool_content[:60]}... 信息已充分，生成最终回答。"
            answer = f"任务处理完成！执行结果详情如下：\n{tool_content}"
            if "周报" in user_intent:
                answer = f"周报相关任务已顺利完成：{tool_content}"

            return {
                "content": f"<thought>{thought}</thought>\n<final_answer>{answer}</final_answer>",
                "tool_calls": None,
                "raw_response": {},
            }

        # 2. 如果最后一条是用户发来的消息
        user_text = last_msg.content.strip()

        # A. 纯闲聊/常识问答（不调用工具）
        if any(greet in user_text for greet in ["你好", "hello", "hi", "你是谁", "介绍一下自己"]):
            return {
                "content": (
                    "<thought>用户正在进行问候，无需调用任何外部工具，直接以友好专业的态度进行自我介绍并回复。</thought>\n"
                    "<final_answer>你好！我是基于自研 Agent Runtime 运行的 AI 助手。我具备自主规划思考和工具调用能力，支持待办事项管理、精确数学计算、实时天气查询和知识搜索等功能。请问今天有什么我可以帮您的？</final_answer>"
                ),
                "tool_calls": None,
                "raw_response": {},
            }

        # B. 计算器工具意图
        if any(k in user_text for k in ["计算", "算一下", "多少", "+", "-", "*", "/", "sqrt"]):
            expr_match = re.search(r"([0-9\.\+\-\*\/\(\)\s\^sqrt]+)", user_text)
            expr = expr_match.group(1).strip() if expr_match else "12 * (34 + 56)"
            # 清理常见汉字
            expr = re.sub(r"[^\d\+\-\*\/\(\)\.\s]", "", expr) or "12 * (34 + 56)"
            return {
                "content": (
                    f"<thought>用户请求进行数学计算，识别到表达式。为了保证计算绝对精确，调用 calculator 工具执行。</thought>\n"
                    f"<tool_call>{{\"name\": \"calculator\", \"arguments\": {{\"expression\": \"{expr}\"}}}}</tool_call>"
                ),
                "tool_calls": None,
                "raw_response": {},
            }

        # C. 天气查询意图
        if "天气" in user_text:
            city = "北京"
            for c in ["北京", "上海", "深圳", "广州", "杭州", "成都", "武汉"]:
                if c in user_text:
                    city = c
                    break
            return {
                "content": (
                    f"<thought>用户询问 {city} 的天气情况。需要调用天气工具获取实时预报。</thought>\n"
                    f"<tool_call>{{\"name\": \"weather\", \"arguments\": {{\"city\": \"{city}\", \"date\": \"今天\"}}}}</tool_call>"
                ),
                "tool_calls": None,
                "raw_response": {},
            }

        # D. 待办事项管理意图 (Todo)
        if any(k in user_text for k in ["待办", "todo", "任务", "清单"]):
            if any(k in user_text for k in ["完成", "做完", "done"]):
                # 标记完成
                id_match = re.search(r"\d+", user_text)
                tid = int(id_match.group(0)) if id_match else 1
                return {
                    "content": (
                        f"<thought>用户希望将待办事项 #{tid} 标记为完成。调用 todo 工具 complete 动作。</thought>\n"
                        f"<tool_call>{{\"name\": \"todo\", \"arguments\": {{\"action\": \"complete\", \"todo_id\": {tid}}}}} </tool_call>"
                    ),
                    "tool_calls": None,
                    "raw_response": {},
                }
            elif any(k in user_text for k in ["查看", "列出", "有哪些", "查一下", "查询", "看下", "有哪些任务", "清单"]):
                return {
                    "content": (
                        "<thought>用户需要查看当前的待办事项列表。调用 todo 工具 list 动作。</thought>\n"
                        "<tool_call>{\"name\": \"todo\", \"arguments\": {\"action\": \"list\"}}</tool_call>"
                    ),
                    "tool_calls": None,
                    "raw_response": {},
                }
            else:
                # 添加待办
                title = user_text.replace("添加待办", "").replace("记待办", "").replace("记个待办", "").replace("待办", "").strip() or "默认待办任务"
                return {
                    "content": (
                        f"<thought>用户要求新增一条待办事项。调用 todo 工具的 add 动作进行持久化。</thought>\n"
                        f"<tool_call>{{\"name\": \"todo\", \"arguments\": {{\"action\": \"add\", \"title\": \"{title}\", \"priority\": \"medium\"}}}}</tool_call>"
                    ),
                    "tool_calls": None,
                    "raw_response": {},
                }

        # E. 搜索意图
        if any(k in user_text for k in ["搜索", "查询", "查找", "什么是"]):
            return {
                "content": (
                    f"<thought>用户需要检索外部知识。调用 search 工具获取相关资料。</thought>\n"
                    f"<tool_call>{{\"name\": \"search\", \"arguments\": {{\"query\": \"{user_text}\"}}}}</tool_call>"
                ),
                "tool_calls": None,
                "raw_response": {},
            }

        # 默认：直接回复
        return {
            "content": f"<thought>该请求可直接基于当前知识库理解并回答。</thought>\n<final_answer>我理解了您的输入：'{user_text}'。请问接下来需要我为您执行哪些操作？</final_answer>",
            "tool_calls": None,
            "raw_response": {},
        }
