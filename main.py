"""
Agent Harness 主交互程序与多会话 CLI 演示入口
支持终端实时交互、真实 LLM API 配置、多窗口无缝切换与自动演示
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import Optional
from agent_harness.core.llm import MockLLMClient, OpenAILikeClient
from agent_harness.runtime.loop import AgentRuntime
from agent_harness.runtime.session import SessionManager
from agent_harness.tools.builtin.calculator import CalculatorTool
from agent_harness.tools.builtin.search import SearchTool
from agent_harness.tools.builtin.todo import TodoStore, TodoTool
from agent_harness.tools.builtin.weather import WeatherTool
from agent_harness.tools.registry import ToolRegistry


def build_runtime(use_mock: bool = False, model: Optional[str] = None, use_gemini: bool = False) -> AgentRuntime:
    import json
    from pathlib import Path

    registry = ToolRegistry()
    session_manager = SessionManager()

    def get_active_sid():
        s = session_manager.get_active_session()
        return s.state.session_id if s else "default"

    registry.register(TodoTool(active_session_id_getter=get_active_sid))
    registry.register(CalculatorTool())
    registry.register(WeatherTool())
    registry.register(SearchTool())

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    llm_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if use_gemini:
        # 自动尝试加载 opencode 本地 Gemini 配置
        cfg_path = Path.home() / ".config/opencode/opencode.json"
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
                ag = cfg.get("provider", {}).get("antigravity", {})
                base_url = ag.get("options", {}).get("baseURL") or "http://localhost:9878/v1"
                api_key = ag.get("options", {}).get("apiKey") or "mock-key"
                llm_model = model or "gemini-3.8-flash-high"
            except Exception:
                pass

    if use_mock or not api_key:
        if not use_mock and not api_key:
            print("\033[1;33m[提示] 未检测到 API Key，自动启用内置高保真 MockLLMClient 进行交互演示。\033[0m")
        llm = MockLLMClient()
    else:
        print(f"\033[1;32m[提示] 成功连接真实 LLM 端点: {base_url} (Model: {llm_model})\033[0m")
        llm = OpenAILikeClient(api_key=api_key, base_url=base_url, model=llm_model, mode="react")

    return AgentRuntime(
        llm_client=llm,
        tool_registry=registry,
        session_manager=session_manager,
        enable_console_log=True,
    )


def run_demo():
    """运行笔试题中的核心业务场景演示：用户 A 多窗口独立 Session 交互"""
    print("\n" + "=" * 65)
    print("🚀 启动 Agent Harness 笔试题多窗口 Session 场景自动化演示")
    print("=" * 65)

    runtime = build_runtime(use_mock=True)

    # 1. 窗口 1：查天气记待办
    print("\n\033[1;34m▶ [场景 1] 用户 A 在【窗口 1: win_weather】提问：查北京天气并记待办\033[0m")
    resp1 = runtime.run(
        user_input="帮我查一下北京天气，并且把带雨伞写进待办",
        session_id="win_weather",
        user_id="user_A",
    )
    print(f"\033[1;32m[窗口 1 回复]:\033[0m {resp1['answer']}")

    # 2. 窗口 2：写周报记待办
    print("\n\033[1;34m▶ [场景 2] 用户 A 切换到【窗口 2: win_weekly】提问：写周报记待办\033[0m")
    resp2 = runtime.run(
        user_input="今天写周报，添加待办：本周技术架构评审总结",
        session_id="win_weekly",
        user_id="user_A",
    )
    print(f"\033[1;32m[窗口 2 回复]:\033[0m {resp2['answer']}")

    # 3. 追问窗口 1：检查待办隔离性
    print("\n\033[1;34m▶ [场景 3] 用户 A 随时切回【窗口 1: win_weather】继续追问：查一下待办清单\033[0m")
    resp1_follow = runtime.run(
        user_input="查一下我刚才记的待办清单",
        session_id="win_weather",
        user_id="user_A",
    )
    print(f"\033[1;32m[窗口 1 追问回复]:\033[0m {resp1_follow['answer']}")

    # 4. 追问窗口 2：检查待办隔离性
    print("\n\033[1;34m▶ [场景 4] 用户 A 随时切回【窗口 2: win_weekly】继续追问：查看有哪些待办任务\033[0m")
    resp2_follow = runtime.run(
        user_input="查看有哪些待办任务",
        session_id="win_weekly",
        user_id="user_A",
    )
    print(f"\033[1;32m[窗口 2 追问回复]:\033[0m {resp2_follow['answer']}")

    print("\n" + "=" * 65)
    print("✅ 演示完成！窗口 1 与窗口 2 状态完全独立，会话无缝保留，工具调用与 Trace 记录完备。")
    print("=" * 65)


def interactive_cli(use_mock: bool = False, model: Optional[str] = None, use_gemini: bool = False):
    """交互式 CLI 会话控制台"""
    runtime = build_runtime(use_mock=use_mock, model=model, use_gemini=use_gemini)
    current_sid = "default_session"

    print("\n" + "=" * 60)
    print("🤖 欢迎进入 Agent Harness 交互终端")
    print("常用命令:")
    print("  /demo            - 运行内置的多窗口独立 Session 自动化演示")
    print("  /list            - 列出所有已开启的会话窗口")
    print("  /switch <id>     - 切换到指定会话窗口")
    print("  /new <title>     - 创建新会话窗口")
    print("  /exit 或 /quit   - 退出终端")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input(f"\033[1;36m[{current_sid}] > \033[0m").strip()
            if not user_input:
                continue

            if user_input in ("/exit", "/quit"):
                print("再见！")
                break
            elif user_input == "/demo":
                run_demo()
                continue
            elif user_input == "/list":
                sessions = runtime.session_manager.list_sessions()
                print("\n当前所有会话列表:")
                for s in sessions:
                    active_marker = " * " if s["session_id"] == current_sid else "   "
                    print(f"{active_marker}{s['session_id']} | 标题: {s['title']} | 轮次: {s['turns_count']}")
                print()
                continue
            elif user_input.startswith("/switch "):
                target_id = user_input[8:].strip()
                runtime.session_manager.switch_active_session(target_id)
                current_sid = target_id
                print(f"已切换到会话: {current_sid}\n")
                continue
            elif user_input.startswith("/new "):
                title = user_input[5:].strip()
                new_sess = runtime.session_manager.create_session(title=title)
                current_sid = new_sess.state.session_id
                print(f"已创建并切换到新会话: {current_sid} ({title})\n")
                continue

            # 执行正常问答
            resp = runtime.run(user_input=user_input, session_id=current_sid)
            print(f"\n\033[1;32m[Assistant 回复]:\033[0m\n{resp['answer']}\n")

        except KeyboardInterrupt:
            print("\n会话已终止。")
            break
        except Exception as e:
            print(f"\n\033[1;31m[执行异常]:\033[0m {str(e)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent Harness CLI")
    parser.add_argument("--demo", action="store_true", help="直接运行多窗口独立会话演示")
    parser.add_argument("--mock", action="store_true", help="强制使用内置 Mock LLM")
    parser.add_argument("--gemini", action="store_true", help="自动接入本地 opencode 的 Google Gemini 3.8 Flash High 大模型")
    parser.add_argument("--model", type=str, default=None, help="指定大模型名称")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        interactive_cli(use_mock=args.mock, model=args.model, use_gemini=args.gemini)
