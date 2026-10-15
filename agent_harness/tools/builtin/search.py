"""
搜索/知识库检索工具 (SearchTool / Mock Search)
"""
from __future__ import annotations
from typing import Any, Dict, List
from agent_harness.tools.base import BaseTool


_MOCK_KNOWLEDGE_BASE = [
    {
        "keywords": ["agent", "runtime", "react"],
        "title": "Agent Runtime 架构核心原理",
        "snippet": "Agent 运行时核心包括环境感知、状态循环（Thought-Action-Observation）、工具调用分发以及跨轮会话上下文管理。",
    },
    {
        "keywords": ["jev", "system one", "decision model"],
        "title": "System One 决策模型 JEV 原理",
        "snippet": "JEV 模型是单次前向传导输出类别概率分布的非生成式决策模型，具备毫秒级延迟与极高概率校准度，适合作为 Agent 的快动作执行器。",
    },
    {
        "keywords": ["周报", "汇报", "template"],
        "title": "高效技术周报模版与规范",
        "snippet": "周报应包含：1. 本周核心进展与产出；2. 关键工程指标与测试通过率；3. 遇到的技术阻塞点及解决方案；4. 下周工作规划。",
    },
]


class SearchTool(BaseTool):
    """搜索检索工具 (支持 Mock 知识检索)"""

    def __init__(self):
        parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索关键词或查询语句，例如 '周报 规范', 'JEV 架构'",
                }
            },
            "required": ["query"],
        }
        super().__init__(
            name="search",
            description="搜索外部网络知识库或文档，返回相关段落和参考摘要。",
            parameters=parameters,
        )

    def execute(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        q_lower = query.lower()
        results: List[Dict[str, str]] = []

        for item in _MOCK_KNOWLEDGE_BASE:
            if any(k in q_lower for k in item["keywords"]) or any(w in item["title"].lower() for w in q_lower.split()):
                results.append({"title": item["title"], "snippet": item["snippet"]})

        if not results:
            results.append({
                "title": f"关于 '{query}' 的综合检索结果",
                "snippet": f"在本地与网络索引中检索关键词 '{query}'，找到了相关技术讨论与上下文，请结合当前任务判断。",
            })

        return {
            "success": True,
            "query": query,
            "results_count": len(results),
            "results": results,
        }
