"""
工具基础抽象与装饰器机制 (支持基于 Schema 的工具定义)
"""
from __future__ import annotations
import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type, get_type_hints


PYTHON_TYPE_TO_JSON_SCHEMA = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class BaseTool(ABC):
    """工具基类"""

    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """执行工具逻辑"""
        pass

    def to_schema(self) -> Dict[str, Any]:
        """导出符合 OpenAI / 标准规范的工具描述"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __repr__(self) -> str:
        return f"<Tool {self.name}: {self.description[:30]}...>"


class FunctionTool(BaseTool):
    """基于普通函数包装的工具实现"""

    def __init__(
        self,
        func: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self.func = func
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "").strip().split("\n")[0]
        params_schema = parameters or self._generate_schema(func)
        super().__init__(name=tool_name, description=tool_desc, parameters=params_schema)

    @staticmethod
    def _generate_schema(func: Callable[..., Any]) -> Dict[str, Any]:
        sig = inspect.signature(func)
        try:
            type_hints = get_type_hints(func)
        except Exception:
            type_hints = {}

        properties: Dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = type_hints.get(param_name, str)
            json_type = PYTHON_TYPE_TO_JSON_SCHEMA.get(param_type, "string")

            prop_def: Dict[str, Any] = {"type": json_type}
            
            # 尝试从 docstring 中获取参数说明（简单解析）
            doc = func.__doc__ or ""
            for line in doc.split("\n"):
                line_str = line.strip()
                if line_str.startswith(f":param {param_name}:"):
                    prop_def["description"] = line_str[len(f":param {param_name}:"):].strip()
                elif line_str.startswith(f"- {param_name}:"):
                    prop_def["description"] = line_str[len(f"- {param_name}:"):].strip()

            properties[param_name] = prop_def

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def execute(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)


def tool(name: Optional[str] = None, description: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None):
    """装饰器：将普通函数注册为可用工具"""
    def decorator(fn: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(func=fn, name=name, description=description, parameters=parameters)
    return decorator
