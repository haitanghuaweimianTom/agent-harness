"""
LLM 输出解析器 (Parser)
提取思考过程 (Thought)、工具调用 (Tool Call) 与最终答案 (Final Answer)
具备工业级容错与多种格式自适应能力
"""
from __future__ import annotations
import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple
from agent_harness.core.types import ToolCall


class LLMOutputParser:
    """LLM 响应鲁棒解析器"""

    @staticmethod
    def parse(raw_text: str, native_tool_calls: Optional[List[Dict[str, Any]]] = None) -> Tuple[Optional[str], List[ToolCall], Optional[str]]:
        """
        解析 LLM 原始文本与原生 tool_calls
        返回: (thought, list_of_tool_calls, final_answer)
        """
        # 1. 优先检查是否有原生 OpenAI tool_calls
        if native_tool_calls:
            calls: List[ToolCall] = []
            for tc in native_tool_calls:
                call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args_raw = fn.get("arguments", "{}")
                if isinstance(args_raw, str):
                    try:
                        args = json.loads(args_raw)
                    except Exception:
                        args = {}
                else:
                    args = args_raw or {}
                calls.append(ToolCall(id=call_id, name=name, arguments=args))
            return None, calls, None

        if not raw_text:
            return None, [], ""

        text = raw_text.strip()
        thought: Optional[str] = None
        tool_calls: List[ToolCall] = []
        final_answer: Optional[str] = None

        # 2. 尝试提取 <thought>...</thought>
        thought_match = re.search(r"<thought>(.*?)</thought>", text, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()
        else:
            # 兼容模型只写了 <thought> 没有闭合
            open_thought = re.search(r"<thought>(.*?)(?:<tool_call>|<final_answer>|$)", text, re.DOTALL | re.IGNORECASE)
            if open_thought and "<tool_call>" in text:
                thought = open_thought.group(1).strip()

        # 3. 尝试提取 <tool_call>...</tool_call>
        tool_matches = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL | re.IGNORECASE)
        for t_raw in tool_matches:
            call = LLMOutputParser._parse_single_tool_json(t_raw.strip())
            if call:
                tool_calls.append(call)

        # 4. 如果没匹配到 <tool_call> 标签，但有明确的 ```json 代码块包含 name/arguments
        if not tool_calls and "<tool_call>" not in text:
            code_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            for block in code_blocks:
                call = LLMOutputParser._parse_single_tool_json(block.strip())
                if call:
                    tool_calls.append(call)

        # 5. 提取 <final_answer>...</final_answer>
        ans_match = re.search(r"<final_answer>(.*?)</final_answer>", text, re.DOTALL | re.IGNORECASE)
        if ans_match:
            final_answer = ans_match.group(1).strip()
        elif not tool_calls:
            # 如果既没有 tool_call，也没有显式的 <final_answer> 标签
            # 移除 <thought> 标签内容后剩余的部分即为最终答案
            clean_text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
            # 移除未闭合的 <thought> 前缀
            clean_text = re.sub(r"<thought>.*", "", clean_text, flags=re.DOTALL | re.IGNORECASE).strip()
            final_answer = clean_text or text

        return thought, tool_calls, final_answer

    @staticmethod
    def _parse_single_tool_json(json_str: str) -> Optional[ToolCall]:
        """尝试解析单个工具调用的 JSON 字符串"""
        clean = json_str.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()

        try:
            data = json.loads(clean)
            if isinstance(data, dict):
                name = data.get("name") or data.get("tool") or data.get("action")
                args = data.get("arguments") or data.get("parameters") or data.get("args") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"input": args}
                if name:
                    return ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        name=str(name),
                        arguments=args if isinstance(args, dict) else {},
                    )
        except Exception:
            # 容错：尝试用正则提取 name 和 arguments
            name_m = re.search(r'["\']name["\']\s*:\s*["\']([^"\']+)["\']', clean)
            if name_m:
                return ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    name=name_m.group(1),
                    arguments={},
                )
        return None
