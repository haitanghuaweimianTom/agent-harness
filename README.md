# Agent Harness: 从零实现的轻量高扩展 Agent 核心 Runtime

> 🚀 **零依赖第三方 Agent 框架**（无 LangGraph / OpenHands / OpenClaw），完全基于原生 Python 构建的工业级、高扩展、双引擎（真实 LLM API + 高保真 Mock）最小可用 Agent 运行时。

---

## 🌟 核心特性与架构概览

1. **纯手写自主 Agent 循环 (ReAct ToolLoop)**:
   - **Step 1**: 接收用户意图与多轮上下文拼装；
   - **Step 2**: LLM 自主决策判定（直接自然语言回复 vs 调用注册工具）；
   - **Step 3**: 动态分发工具执行，具备参数校验、执行计时与异常防护；
   - **Step 4**: 观察结果（Observation）回填，自主判断继续循环或收敛交付最终答案。
2. **多窗口会话隔离 (Multi-Session Management)**:
   - 彻底解决多窗口并发场景下的数据污染问题：用户 A 开窗口 1（查天气记待办），开窗口 2（写周报记待办），两窗口上下文与内部待办存储完全独立，支持随时无缝切换继续追问。
3. **上下文智能管理与滑动压缩 (Context Management & Compression)**:
   - 支持纯对话追问与带工具结果的上下文连续追问；
   - 内置轮次监控与自动压缩机制（首尾锚定 + 滚动语义摘要），保障超长对话不超窗口且不迷失目标。
4. **完备的可观测性与追踪器 (Execution Tracer)**:
   - 记录每一步的思维链（Thought）、工具参数、毫秒级执行耗时、工具 Observation 与最终 Answer，支持结构化 JSON 导出。
5. **双模式 LLM 引擎支持**:
   - 原生支持任何 OpenAI-compatible API（DeepSeek, Qwen, Moonshot, GLM, OpenAI 等），仅需配置标准环境变量；
   - 内置确定性高保真 `MockLLMClient`，实现离线 0 成本、毫秒级自动化测试。

---

## 📁 目录结构

```
projects/agent-harness/
├── agent_harness/
│   ├── core/
│   │   ├── types.py            # Message, ToolCall, ToolResult, AgentStep 等核心数据契约
│   │   ├── llm.py              # 原生 LLM 客户端 (OpenAILikeClient + MockLLMClient)
│   │   ├── prompt.py           # 动态系统提示词与工具 Schema 注入
│   │   ├── parser.py           # 工业级鲁棒输出解析器 (Thought / ToolCall / FinalAnswer)
│   │   └── context_manager.py  # 轮次管理、追问状态保持与滚动摘要压缩
│   ├── runtime/
│   │   ├── loop.py             # 核心 Agent Runtime 循环状态机
│   │   ├── session.py          # 多会话隔离管理器
│   │   └── tracer.py           # 执行追踪器与性能耗时记录
│   └── tools/
│       ├── base.py             # BaseTool 抽象基类与 @tool 自动 Schema 装饰器
│       ├── registry.py         # 工具注册中心 (ToolRegistry)
│       └── builtin/
│           ├── todo.py         # 待办事项管理 (按 Session 状态物理隔离)
│           ├── calculator.py   # 安全数学表达式计算器 (AST 防注入)
│           ├── weather.py      # 实时/模拟天气查询工具
│           └── search.py       # 知识库检索与文档搜索工具
├── tests/
│   ├── test_tools.py           # 工具单元测试
│   ├── test_session.py         # 多窗口独立 Session 隔离与切换测试
│   ├── test_context.py         # 上下文管理与自动压缩测试
│   ├── test_runtime_loop.py    # 核心 Loop 端到端链路测试
│   └── run_all.py              # 一键自动化运行全套测试套件
├── main.py                     # CLI 交互入口与多窗口演示
└── README.md                   # 系统设计与使用文档
```

---

## 🧠 Memory 的召回时机与放置方式说明

在 Agent 系统中，Memory（记忆）是解决长周期、跨会话信息连贯性的核心机制。本系统采用**层次化认知记忆**设计理念：

### 1. 记忆的分层模型
- **工作记忆（Working Memory）**:
  - 范围：当前会话最近 3~5 轮高保真上下文消息（User、Assistant Thought、Tool Observation）；
  - 存储：由 `ContextManager` 驻留在内存中。
- **短期摘要记忆（Buffer Summary Memory）**:
  - 范围：被滑动窗口滑出的较早轮次；
  - 处理：触发压缩时，由 `_compress_messages_to_summary` 抽象出核心事实，固化为 `<context_summary>` 锚点。
- **长期记忆（Long-Term Profile / Episodic Memory）**:
  - 范围：跨越数天至数周的用户画像、偏好、重要事实（例如用户的常住城市、工作岗位、常用项目）。

### 2. 记忆的召回时机（When to Recall）
- **避免“盲目全局检索”**：严禁在每一轮对话中无差别进行向量库 Top-K 检索（会引入无关噪音、消耗显存和延迟）。
- **召回触发条件（两级门控策略）**：
  1. **意图路由判定**：当用户输入包含跨周期指代词（如“上次说的”、“之前那个项目”、“我习惯的格式”），或初次规划发现当前会话上下文信息不足时；
  2. **混合召回策略（Hybrid Retrieval）**：
     $$\text{FinalScore} = 0.5 \times \text{DenseVectorSim} + 0.3 \times \text{BM25Keyword} + 0.2 \times \text{RecencyDecay}$$

### 3. 记忆的放置方式（Where & How to Place）
- **严禁直接假装成用户的即时输入**（否则大模型会发生时空倒错，误以为是刚才发生的事）；
- **标准隔离注入范式**：
  在组织请求消息序列时，将召回的长期记忆以独立的系统上下文块放置在 **System Prompt 之后、历史消息之前**，形如：
  ```xml
  <system>
    你是一个智能助手...
  </system>
  <system_recalled_memory source="long_term_profile" timestamp="15 days ago">
    - 用户偏好: 汇报偏好结构化 Markdown 表格形式
    - 历史待办事实: 上周讨论过分布式状态机设计方案
  </system_recalled_memory>
  <user>用户当前问题</user>
  ```
  这种放置方式既能充分享受现代服务端的 **Prefix KV Cache** 命中优化，又能让模型清晰分辨“历史事实”与“当前上下文”。

---

## ⚡ 运行方式与快速上手

### 1. 运行自动化测试套件 (100% 离线通过)
```bash
python3 tests/run_all.py
```
> 输出包含 12 项覆盖工具、Session 隔离、上下文压缩、端到端循环的完整测试报告。

### 2. 运行笔试题业务场景演示 (多窗口隔离演示)
```bash
python3 main.py --demo
```
> 自动演示：用户 A 在窗口 1 查天气记待办，在窗口 2 写周报记待办，并展示切回窗口 1/2 追问时的彻底状态隔离。

### 3. 启动交互式终端 (CLI Console)
```bash
# 模式 A: 使用内置 Mock LLM 体验
python3 main.py --mock

# 模式 B: 连接真实 LLM API (支持 DeepSeek / OpenAI / Qwen 等)
export OPENAI_API_KEY="sk-your-key-here"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 或国内镜像端点
export OPENAI_MODEL="gpt-4o-mini"                   # 或 deepseek-chat
python3 main.py
```

在终端中可使用以下交互指令：
- `/demo` : 重新运行多窗口独立演示
- `/list` : 查看当前所有开启的会话窗口
- `/switch <session_id>` : 切换当前活动窗口
- `/new <title>` : 开启全新独立会话窗口
- `/exit` : 退出

---

## 🛠 自定义工具扩展方法

通过 `@tool` 装饰器，仅需几行原生 Python 代码即可轻松将任意函数注册为 Agent 工具：

```python
from agent_harness.tools.base import tool

@tool(name="github_repo_info", description="获取 GitHub 仓库基本信息")
def get_repo_info(owner: str, repo: str) -> dict:
    # 你的业务逻辑
    return {"repo": f"{owner}/{repo}", "stars": 1280, "language": "Python"}

# 注册到 ToolRegistry
runtime.tool_registry.register(get_repo_info)
```
系统将自动通过类型注解推导出标准的 JSON Schema，注入至 System Prompt 中供 LLM 自主选择调用。

