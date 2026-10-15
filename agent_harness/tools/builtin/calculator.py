"""
安全的计算器工具 (CalculatorTool)
"""
from __future__ import annotations
import ast
import math
import operator
from typing import Any, Dict
from agent_harness.tools.base import BaseTool


# 安全算符映射表
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        op_type = type(node.op)
        if op_type in _OPERATORS:
            return _OPERATORS[op_type](left, right)
        raise ValueError(f"不支持的操作符: {op_type.__name__}")
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval_node(node.operand)
        op_type = type(node.op)
        if op_type in _OPERATORS:
            return _OPERATORS[op_type](operand)
        raise ValueError(f"不支持的一元操作符: {op_type.__name__}")
    elif isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCTIONS:
            return _SAFE_FUNCTIONS[node.id]
        raise ValueError(f"未知或禁止使用的变量: {node.id}")
    elif isinstance(node, ast.Call):
        func = _safe_eval_node(node.func)
        args = [_safe_eval_node(arg) for arg in node.args]
        if callable(func):
            return func(*args)
        raise ValueError("不可调用的对象")
    else:
        raise ValueError(f"不安全的表达式节点: {type(node).__name__}")


class CalculatorTool(BaseTool):
    """数学计算器工具"""

    def __init__(self):
        parameters = {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "需要计算的数学表达式，例如: '12 * (34 + 56)', 'sqrt(144) + 2**8'",
                }
            },
            "required": ["expression"],
        }
        super().__init__(
            name="calculator",
            description="执行精确数学运算的计算器工具。支持基础四则运算、括号、幂运算、开方及常用数学函数。",
            parameters=parameters,
        )

    def execute(self, expression: str, **kwargs: Any) -> Dict[str, Any]:
        expr = expression.strip()
        try:
            tree = ast.parse(expr, mode="eval")
            result = _safe_eval_node(tree.body)
            # 格式化输出
            if isinstance(result, float) and result.is_integer():
                formatted = int(result)
            elif isinstance(result, float):
                formatted = round(result, 6)
            else:
                formatted = result

            return {
                "success": True,
                "expression": expr,
                "result": formatted,
            }
        except ZeroDivisionError:
            return {"success": False, "error": "除零错误 (Division by zero)"}
        except Exception as e:
            return {"success": False, "error": f"计算表达式失败: {str(e)}"}
