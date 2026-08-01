# 企业级 RAG Agent → MCP 生态改造说明

把你项目里「可被 LLM 调用的能力（Skill）」从**单体进程内调用**，升级为**标准 MCP 协议暴露**，
让任意兼容 MCP 的 AI 客户端（Claude Desktop / Cursor / 自研 Agent）零改造复用你的工具。

---

## 一、改造前 vs 改造后（架构对比）

### 改造前：能力锁在单体进程内

```
┌──────────────────────────────────────────────┐
│  单个 Python 进程 (advanced_rag_agent.py)      │
│                                                │
│   用户提问                                     │
│     └─ PlanningAgent ─ ReActAgent              │
│            └─ SkillRegistry (内存字典)          │
│                  ├─ DocSearchSkill             │
│                  └─ CalculatorSkill            │
│                                                │
│   只能被「自己这个 Agent」调用；               │
│   别的 AI 客户端进不来。                        │
└──────────────────────────────────────────────┘
```

- LLM 通过**自研 ReAct 提示词循环**决定调哪个 Skill，再 `skill.execute()`
- 没有协议层、没有 client/server 拆分、没有标准传输
- 工具「发现 / 调用 / 描述」的语义都是你自己的约定

### 改造后：Skill 内核统一，协议对外暴露

```
                   ┌─────────────────────────────────────┐
                   │   skill_framework.py（协议无关的共享内核） │
                   │   BaseSkill / CalculatorSkill /       │
                   │   SkillRegistry / safe_eval           │
                   └───────────────┬─────────────────────┘
                      ┌────────────┴────────────┐
        in-process 调用│                        │  MCP 协议暴露
                      ▼                        ▼
   ┌──────────────────────────────┐   ┌──────────────────────────────────┐
   │ 原 ReAct / Planning Agent    │   │  mcp_server.py (FastMCP)          │
   │ （零改动，照旧 import 使用）  │   │  Tools:    calculator / doc_search │
   │                              │   │  Resource: skills://list          │
   │                              │   │  Prompt:   security_review         │
   └──────────────────────────────┘   └──────────────┬───────────────────┘
                                                     │  stdio / HTTP
                       ┌─────────────────────────────┼─────────────────────────────┐
                       ▼                             ▼                             ▼
               Claude Desktop                  Cursor / 其它 IDE            自研 AI 客户端
              （mcpServers 配置）            （MCP 配置）                （mcp_client_example.py）
```

**关键变化**：`skill_framework.py` 成为唯一事实来源（single source of truth），
in-process Agent 与 MCP Server **复用同一份实现**，新增一个 Skill 两处自动生效。

---

## 二、能力映射表（Skill → MCP 原语）

| 你项目里的概念 | 改造前 | 改造后（MCP 原语） |
|---------------|--------|--------------------|
| `CalculatorSkill` | `registry.get_skill("calculator").execute()` | **Tool** `calculator(expression)` |
| `DocSearchSkill` | `registry.get_skill("doc_search").execute()` | **Tool** `doc_search(query, top_k)` |
| `SkillRegistry.get_all_descriptions()` | Agent 内部拼提示词 | **Resource** `skills://list` |
| 安全评审口径 | 散落在代码注释 | **Prompt** `security_review(skill_name)` |
| `BaseSkill.validate_params()` | 调用前守门 | Tool 内部**照旧先校验**（安全延续） |
| `safe_eval()` | CalculatorSkill 内部 | Tool 内部**照旧调用**（杜绝 eval） |

---

## 三、改造前后对比（维度一览）

| 维度 | 改造前 | 改造后 |
|------|--------|--------|
| 调用边界 | 仅本进程内的 Agent | 任意 MCP 客户端，跨进程 / 跨机器 |
| 协议 | 无（自研 Python 调用） | MCP（JSON-RPC 2.0） |
| 工具发现 | 代码里硬编码清单 | `list_tools()` / `skills://list` 自动发现 |
| 传输 | — | stdio（本地）/ Streamable HTTP（远程） |
| 复用成本 | 复制粘贴代码 | 配一行 `mcpServers` 即可 |
| 安全模型 | validate_params + safe_eval | **原样复用**，未降级 |
| 耦合度 | Skill 与 Agent 强耦合 | Skill 内核独立（`skill_framework`） |
| 扩展新能力 | 改 Agent 代码 | 在 `skill_framework` 注册，Agent 与 MCP 双生效 |

---

## 四、别的 AI 客户端如何复用你的工具

### 方式 A：Claude Desktop / Cursor（改配置即可）

把下面这段加进对应客户端的 MCP 配置文件（`claude_desktop_config.json` 或 Cursor 的 MCP 设置）：

```json
{
  "mcpServers": {
    "enterprise-rag": {
      "command": "python",
      "args": ["D:/work/workspace/pythonspace/mcp_server.py"]
    }
  }
}
```

重启客户端后，它会在后台拉起 `mcp_server.py` 子进程，自动发现
`calculator` / `doc_search` 两个工具——你不用写一行客户端代码。

### 方式 B：远程暴露（Streamable HTTP）

```bash
python mcp_server.py --http --host 0.0.0.0 --port 8000
```

其它机器上的客户端用 `http://<你的IP>:8000/mcp` 连接，实现团队级工具共享。

### 方式 C：自研 AI 客户端（代码集成）

见 `mcp_client_example.py`——它用官方 `mcp` SDK 的 `stdio_client` + `ClientSession`
完成「握手 → list_tools → call_tool → read_resource → get_prompt」，
证明**任何语言/框架只要实现 MCP 客户端，就能调你的工具**。

---

## 五、本地运行验证

```bash
# 1) 安装（需在 Python 3.10+ 环境）
pip install fastmcp

# 2) 跑通「别的客户端复用」最小示例
python mcp_client_example.py
```

预期输出要点：
- 列出工具 `calculator` / `doc_search`
- `calculator("120/24")` → `计算结果：120/24 = 5`
- `calculator("__import__('os').system('rm -rf /')")` → 被 `validate_params` 拦截
- `doc_search("JM-S509 定位精度")` → 复用同一份安全守门
- 读取 `skills://list`、获取 `security_review` 提示词模板

---

## 六、改造涉及的文件

| 文件 | 作用 |
|------|------|
| `skill_framework.py` | **新增**：协议无关的 Skill 内核（single source of truth） |
| `advanced_rag_agent.py` | 改造：原 `BaseSkill`/`safe_eval`/`CalculatorSkill`/`SkillRegistry` 改为从 `skill_framework` 导入，逻辑零变化 |
| `mcp_server.py` | **新增**：FastMCP Server，把 Skill 暴露为 Tools / Resource / Prompt |
| `mcp_client_example.py` | **新增**：外部客户端复用示例（stdio 调用） |

---

## 七、迁移要点（给后续维护者）

1. **新增 Skill**：在 `skill_framework.py` 里写 `class XxxSkill(BaseSkill)` 并 `registry.register()`，
   Agent 与 MCP Server 自动都能用，无需改两处。
2. **安全不可回退**：所有 Tool 调用都必须先过 `validate_params()`；计算器必须走 `safe_eval()`，
   禁止在任何新代码里写 `eval()`。
3. **凭据已在 `.env` 外部化**，Server 本身不持有任何密钥，对外暴露也安全。
