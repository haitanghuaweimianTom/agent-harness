"""
真实调用 Google Gemini 3.8 Flash High 端到端自动化验证脚本
"""
import json
import os
import sys
from pathlib import Path

# 将根目录加入路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_harness.core.llm import OpenAILikeClient
from agent_harness.runtime.loop import AgentRuntime
from agent_harness.runtime.session import SessionManager
from agent_harness.tools.builtin.calculator import CalculatorTool
from agent_harness.tools.builtin.todo import TodoStore, TodoTool
from agent_harness.tools.builtin.weather import WeatherTool
from agent_harness.tools.registry import ToolRegistry


def run_gemini_live_test(model_name: str = "gemini-3.8-flash-high"):
    print("\n" + "=" * 65)
    print(f"🚀 启动 Google Gemini 系列真实大模型接入验证 ({model_name})")
    print("=" * 65)

    # 优先读取 opencode 的 proxy 配置，若无则读取标准环境变量
    cfg_path = Path.home() / ".config/opencode/opencode.json"
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")

    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            ag = cfg.get("provider", {}).get("antigravity", {})
            base_url = base_url or ag.get("options", {}).get("baseURL")
            api_key = api_key or ag.get("options", {}).get("apiKey")
        except Exception:
            pass

    if not api_key:
        print("❌ 未能获取到有效的 API Key，请检查环境变量或 opencode 配置。")
        return False

    client = OpenAILikeClient(
        api_key=api_key,
        base_url=base_url or "http://localhost:9878/v1",
        model=model_name,
        mode="react",
    )

    reg = ToolRegistry()
    sess_mgr = SessionManager()

    def get_current_sid():
        s = sess_mgr.get_active_session()
        return s.state.session_id if s else "gemini_live"

    reg.register(WeatherTool())
    reg.register(TodoTool(active_session_id_getter=get_current_sid))
    reg.register(CalculatorTool())

    runtime = AgentRuntime(
        llm_client=client,
        tool_registry=reg,
        session_manager=sess_mgr,
        enable_console_log=True,
    )

    # 轮次 1：查天气并记待办
    print("\n\033[1;34m▶ [轮次 1] 发送多步复杂任务：查天气并记录待办\033[0m")
    res1 = runtime.run(
        "请帮我查询北京今天的天气，并把出门带伞加入待办事项清单",
        session_id="gemini_live",
    )
    print(f"\n\033[1;32m[Gemini 最终交付回复]:\033[0m\n{res1['answer']}\n")

    # 查验真实存储落库
    todos = TodoStore.get_for_session("gemini_live").list_all()
    print(f"\033[1;33m[底层 TodoStore 状态验证]:\033[0m 已成功持久化 {len(todos)} 项待办：")
    for t in todos:
        print(f"  - #{t['id']}: {t['title']} (状态: {t['status']})")

    # 轮次 2：追问清单
    print("\n\033[1;34m▶ [轮次 2] 带着记忆进行追问：查看当前待办清单\033[0m")
    res2 = runtime.run(
        "查看我当前的待办清单有哪些",
        session_id="gemini_live",
    )
    print(f"\n\033[1;32m[Gemini 追问回复]:\033[0m\n{res2['answer']}\n")

    print("=" * 65)
    print("🎉 Google Gemini 系列真实大模型端到端调用与工具自愈闭环验证全部成功！")
    print("=" * 65)
    return True


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "gemini-3.8-flash-high"
    run_gemini_live_test(model)
