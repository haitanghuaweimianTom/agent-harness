"""
Todo 待办事项管理工具 (按 Session 状态隔离，支持增删改查)
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from agent_harness.tools.base import BaseTool


class TodoStore:
    """多 Session 隔离的 Todo 存储器"""
    _instances: Dict[str, TodoStore] = {}

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self._todos: List[Dict[str, Any]] = []
        self._counter: int = 1

    @classmethod
    def get_for_session(cls, session_id: str) -> TodoStore:
        if session_id not in cls._instances:
            cls._instances[session_id] = TodoStore(session_id)
        return cls._instances[session_id]

    def add(self, title: str, priority: str = "medium") -> Dict[str, Any]:
        item = {
            "id": self._counter,
            "title": title,
            "status": "pending",
            "priority": priority,
        }
        self._counter += 1
        self._todos.append(item)
        return item

    def list_all(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        if status:
            return [t for t in self._todos if t["status"] == status]
        return list(self._todos)

    def complete(self, todo_id: int) -> Optional[Dict[str, Any]]:
        for t in self._todos:
            if t["id"] == todo_id:
                t["status"] = "completed"
                return t
        return None

    def delete(self, todo_id: int) -> bool:
        for i, t in enumerate(self._todos):
            if t["id"] == todo_id:
                self._todos.pop(i)
                return True
        return False

    def clear(self) -> None:
        self._todos.clear()
        self._counter = 1


class TodoTool(BaseTool):
    """待办事项管理工具"""

    def __init__(self, active_session_id_getter=None):
        self.active_session_id_getter = active_session_id_getter
        parameters = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "操作类型：'add' (添加待办), 'list' (列出待办), 'complete' (完成待办), 'delete' (删除待办)",
                    "enum": ["add", "list", "complete", "delete"],
                },
                "title": {
                    "type": "string",
                    "description": "待办事项内容 (当 action='add' 时必填)",
                },
                "todo_id": {
                    "type": "integer",
                    "description": "待办事项的 ID (当 action='complete' 或 'delete' 时必填)",
                },
                "priority": {
                    "type": "string",
                    "description": "优先级：'high', 'medium', 'low'，默认为 'medium'",
                    "enum": ["high", "medium", "low"],
                },
                "status_filter": {
                    "type": "string",
                    "description": "按状态筛选待办：'pending' 或 'completed' (当 action='list' 时可选)",
                    "enum": ["pending", "completed"],
                },
            },
            "required": ["action"],
        }
        super().__init__(
            name="todo",
            description="待办事项清单管理工具。支持创建任务、查询待办、完成待办或删除任务。",
            parameters=parameters,
        )

    def _get_store(self) -> TodoStore:
        session_id = "default"
        if self.active_session_id_getter:
            try:
                session_id = self.active_session_id_getter() or "default"
            except Exception:
                session_id = "default"
        return TodoStore.get_for_session(session_id)

    def execute(
        self,
        action: str,
        title: Optional[str] = None,
        todo_id: Optional[int] = None,
        priority: str = "medium",
        status_filter: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        store = self._get_store()

        if action == "add":
            if not title:
                return {"success": False, "error": "添加待办事项必须提供 'title' 参数"}
            item = store.add(title=title, priority=priority)
            return {
                "success": True,
                "message": f"成功创建待办事项 #{item['id']}: '{title}' (优先级: {priority})",
                "todo": item,
            }

        elif action == "list":
            items = store.list_all(status=status_filter)
            return {
                "success": True,
                "total": len(items),
                "items": items,
            }

        elif action == "complete":
            if todo_id is None:
                return {"success": False, "error": "完成待办必须提供 'todo_id' 参数"}
            try:
                tid = int(todo_id)
            except Exception:
                return {"success": False, "error": f"无效的 todo_id: {todo_id}"}
            item = store.complete(tid)
            if item:
                return {"success": True, "message": f"待办事项 #{tid} 已标记为已完成", "todo": item}
            return {"success": False, "error": f"未找到 ID 为 #{tid} 的待办事项"}

        elif action == "delete":
            if todo_id is None:
                return {"success": False, "error": "删除待办必须提供 'todo_id' 参数"}
            try:
                tid = int(todo_id)
            except Exception:
                return {"success": False, "error": f"无效的 todo_id: {todo_id}"}
            ok = store.delete(tid)
            if ok:
                return {"success": True, "message": f"待办事项 #{tid} 已删除"}
            return {"success": False, "error": f"未找到 ID 为 #{tid} 的待办事项"}

        else:
            return {"success": False, "error": f"不支持的 action: '{action}'"}
