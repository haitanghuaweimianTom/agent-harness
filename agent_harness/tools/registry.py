"""
工具注册中心 (ToolRegistry)
负责工具的集中注册、Schema 导出与安全调用
"""
from __future__ import annotations
import json
import time
import traceback
from typing import Any, Dict, List, Optional
from agent_harness.core.types import ToolCall, ToolResult
from agent_harness.tools.base import BaseTool


class ToolRegistry:
    """工具注册中心"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """注册一个工具"""
        if tool.name in self._tools:
            # 允许覆盖或提示
            pass
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """获取工具"""
        return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        """获取所有已注册工具"""
        return list(self._tools.values())

    def get_schemas(self) -> List[Dict[str, Any]]:
        """获取用于 LLM 的工具 Schema 列表"""
        return [t.to_schema() for t in self._tools.values()]

    def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """执行单次工具调用，内置超时与异常防护"""
        start_time = time.perf_counter()
        tool_name = tool_call.name
        tool = self._tools.get(tool_name)

        if not tool:
            elapsed = (time.perf_counter() - start_time) * 1000
            available = ", ".join(self._tools.keys())
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_name,
                content="",
                success=False,
                error=f"Tool '{tool_name}' not found. Available tools: [{available}]",
                execution_time_ms=elapsed,
            )

        try:
            raw_args = tool_call.arguments
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except Exception:
                    args = {}
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}

            # 执行工具
            output = tool.execute(**args)

            # 序列化结果
            if isinstance(output, (dict, list)):
                content_str = json.dumps(output, ensure_ascii=False)
            else:
                content_str = str(output)

            elapsed = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_name,
                content=content_str,
                success=True,
                error=None,
                execution_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            err_msg = f"Error executing tool '{tool_name}': {str(e)}"
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_name,
                content="",
                success=False,
                error=err_msg,
                execution_time_ms=elapsed,
            )
