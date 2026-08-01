# 企业级大模型网关 (LLM Gateway)

> 把「直连单一 Ollama 实例」升级为具备**多模型路由、限流、熔断降级、连接复用、Token 计费**的统一入口。
> 核心模块 `llm_gateway.py` **仅依赖 Python 标准库**，可被 Agent、MCP Server、单元测试任意端复用。

- 开源仓库：https://github.com/lingluo1hao/enterprise-ai
- 核心文件：`llm_gateway.py`（1343 行）· `llm_gateway.yaml`（配置）· `test_llm_gateway.py`（32 项验证）

---

## 一、改造前 vs 改造后

| 维度 | 改造前 | 改造后 |
|---|---|---|
| **多模型路由** | 单一 `qwen2:7b`，所有任务同一个模型 | 按任务类型路由：经实测 `grade` 可稳定走 1.5b，`classify/rewrite/compress` 仍走 7b；`generate/write/plan/synthesize` 走大模型 |
| **模型切换** | 改代码 | 改 `llm_gateway.yaml` 即可，**支持热重载**（不重启进程） |
| **供应商可插拔** | 只有 Ollama | `ollama` + `openai_compat`（OpenAI / 通义千问 / DeepSeek / Kimi 共用同一适配器） |
| **流控限流** | 无 | 令牌桶，**全局 + 单模型** 两级，**RPM + TPM** 双维度 |
| **熔断降级** | 无 | 三态熔断器 `CLOSED → OPEN → HALF_OPEN`，主模型挂了自动 fallback 到备选链 |
| **兜底** | 抛异常 / 500 | 全链失败返回可配置的降级文案，不把异常甩给用户 |
| **连接复用** | 每次 `chat()` 新建 `ChatPromptTemplate` + `chain.invoke()` | HTTP 长连接池，按 host 分桶复用（实测 3 次调用 0 新建 / 3 次复用） |
| **Token 计算** | 无 | 优先取供应商返回的**真实用量**，缺失时才估算；分模型统计 prompt / completion / 成本 |
| **统一接口** | 只有 `BaseLLM.chat()` | `chat()` / `chat_detailed()` / `stream_chat()` 三个出口，流式与非流式同源 |
| **可观测性** | 只有 `call_count` / `total_time` | `metrics()` 输出连接池、分模型用量、成本、熔断健康状态 |

**关键约束：16 个 `.chat()` 调用点一行业务逻辑都没改。** 网关实现了同样的 `BaseLLM` 接口，是 drop-in 替换。

---

## 二、架构分层

```
                    业务层（Agent / Web / LangGraph）
                    llm.chat(system, user, task="grade")
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │                    LLMGateway                     │
        │  路由表 task→模型链  ·  fallback  ·  降级文案      │
        ├───────────────────────────────────────────────────┤
        │  治理层  TokenBucket(RPM/TPM) · CircuitBreaker     │
        │          CostTracker(真实token+成本)               │
        ├───────────────────────────────────────────────────┤
        │  适配层  OllamaProvider · OpenAICompatProvider     │
        ├───────────────────────────────────────────────────┤
        │  连接层  HttpConnectionPool（按 host 复用长连接）  │
        └───────────────────────────────────────────────────┘
```

---

## 三、快速使用

### 1. 零改造接入（默认已启用）

```python
from advanced_rag_agent import create_llm

llm = create_llm()                       # 返回 LLMGateway，满足 BaseLLM 接口
llm.chat("你是助手", "1+1=?")             # 走默认链
llm.chat("你是助手", "给文档打分", task="grade")   # 走 grade 的模型链（1.5b，实测稳定）
```

`create_llm()` 内置双保险：

- 环境变量 `USE_LLM_GATEWAY=false` → 一键回滚到改造前的单模型直连
- 网关自身初始化失败 → 自动回退 `OllamaLLM`，**网关绝不成为新的单点故障**

### 2. 查看运行指标

```python
m = llm.metrics()
m["pool"]    # {'reused': 4, 'created': 1, 'idle': {...}}
m["usage"]   # 分模型 calls/failures/tokens/avg_latency_s/cost_usd
m["health"]  # {'local-qwen': 'closed'}  熔断状态
```

### 3. 流式输出

```python
for chunk in llm.stream_chat("你是助手", "讲讲 MCP", task="generate"):
    print(chunk, end="", flush=True)
```

---

## 四、配置说明（`llm_gateway.yaml`）

```yaml
# ---- 模型注册表 ----
# provider 可选 ollama / openai / deepseek / dashscope / moonshot / vllm
# 后面几个都是 openai_compat 适配器的别名，走 /chat/completions
models:
  local-qwen:                  # 主力模型（当前唯一实际可用）
    provider: ollama
    model: qwen2:7b
    base_url: http://192.168.200.128:11434
    tier: large
    rpm: 60                    # 每分钟 60 次
    tpm: 150000                # 每分钟 15 万 token
    fail_threshold: 3          # 连续失败 3 次触发熔断
    recovery_sec: 30.0         # 熔断 30 秒后半开探测
    price_in_per_1m: 0.0       # 美元 / 每百万 token，本地模型填 0
    price_out_per_1m: 0.0
    enabled: true

  local-small:                 # 小模型：经实测只给 grade 稳定，其他判断任务仍走 7b
    provider: ollama
    model: qwen2:1.5b
    temperature: 0.0           # 判断题要确定性，温度拉到 0
    rpm: 200
    enabled: true              # ollama pull qwen2:1.5b 后已可用

  qwen-plus:                   # 云端备选，改配置即可插拔
    provider: dashscope
    model: qwen-plus
    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    api_key_env: DASHSCOPE_API_KEY   # 只写环境变量名，密钥不落盘
    price_in_per_1m: 0.11
    price_out_per_1m: 0.28
    enabled: false

# ---- 路由表：任务类型 -> 模型链（第一个是主模型，后面依次 fallback）----
routing:
  classify:   [local-qwen]           # 决定要不要走 RAG，1.5b 会把"需搜索"判成"否"
  grade:      [local-small, local-qwen]
  rewrite:    [local-qwen]           # 1.5b 理解成"列举答案"且更慢
  compress:   [local-qwen]           # 1.5b 不遵循指令
  generate:   [local-qwen, deepseek-chat, qwen-plus]
  write:      [local-qwen, deepseek-chat, qwen-plus]
# 未命中的任务走这条链（保证老代码不传 task 时行为不变）
default_chain: [local-qwen, deepseek-chat, qwen-plus]

# ---- 全局流控（所有模型的总闸门）----
global_rpm: 120
global_tpm: 400000
acquire_timeout: 5.0     # 拿不到令牌最多阻塞 5 秒，超时才判失败

# ---- 连接池 ----
pool_size_per_host: 8    # 每个 host:port 最多缓存 8 条 keep-alive 连接
pool_idle_timeout: 60.0  # 空闲超 60 秒的连接丢弃重建，防拿到死连接

# ---- 重试与降级 ----
max_retries: 1           # 同一模型内重试次数，之后才切 fallback
retry_backoff: 0.5       # 指数退避基数，实际等待 = backoff * 2^n * 抖动
degraded_reply: "抱歉，当前模型服务繁忙，请稍后重试。（网关已尝试全部备选模型）"
reload_interval: 10.0    # 配置热重载检查间隔
```

> 注意：熔断参数（`fail_threshold` / `recovery_sec`）和限流参数（`rpm` / `tpm`）都是
> **按模型单独配置**的，不是全局统一值——小模型便宜可以给大配额，云端模型贵就收紧。

**自愈设计**：配置里写了但实际拉取不到的模型会被自动过滤出链路。
例如 `grade: [local-small, local-qwen]`，若 `qwen2:1.5b` 未拉取，链路自动变成 `[local-qwen]`，
**不会因为配了不存在的模型导致全线崩溃**。

热重载：改完 YAML 无需重启，网关按文件 mtime 自动检测并重建运行时。

---

## 五、验证

```bash
python test_llm_gateway.py
```

32 项断言全部通过，覆盖：

| # | 验证项 | 关键实测结果 |
|---|---|---|
| 1 | 真实调用 + 真实 token 计数 | prompt=30 / completion=7，取自 Ollama 返回值 |
| 2 | 连接池复用 | 3 次调用 **0 新建 / 3 次复用** |
| 3 | 流式输出 | 9 个分片，首字节早于总耗时（真流式） |
| 4 | 令牌桶限流 | 耗尽拒绝 → 按时间补充 → 阻塞等待可拿到；`rate=0` 不限流 |
| 5 | 熔断器状态机 | `CLOSED→OPEN→HALF_OPEN→CLOSED`，半开只放行 1 个探测，探测失败重新 OPEN |
| 6 | 熔断降级 | 主模型失败 → 备选应答，失败/成功分别正确计数，主模型被隔离 |
| 7 | 全链失败兜底 | 返回降级文案；未配置时抛 `AllModelsFailed` |
| 8 | 多模型路由 | 轻/重任务解析出不同链，未知任务回退默认链 |
| 9 | 配置热重载 | 改配置后新模型生效、路由链同步更新 |
| 10 | **全局配额只扣一次** | 3 模型 fallback 真实扣减 **1.00**（修复前为 3.00） |

> 第 10 项是改造过程中发现并修复的缺陷：原实现把全局令牌获取放在 fallback 循环**内**，
> 一次请求试 3 个模型就扣 3 个全局令牌，会让全局限流比配置值严格 3 倍。
> 修复方式是把 `_admit_global()` 提到循环外只调一次，循环内只做 `_admit_model()`。

---

## 六、任务类型标注一览

16 个调用点已按语义标注。实测后，判断类里只有 `grade` 稳定走 1.5b，`classify/rewrite/compress` 仍走 7b，生成类走大模型：

| 文件 | 函数 | task |
|---|---|---|
| advanced_rag_agent.py | `_query_rewrite` | `rewrite` |
| advanced_rag_agent.py | `ReActAgent.run` ×2 | `react` / `generate` |
| advanced_rag_agent.py | `PlannerAgent.plan` | `plan` |
| advanced_rag_agent.py | `PlannerAgent.synthesize` | `synthesize` |
| langgraph_rag_agent.py | `node_classify` | `classify` |
| langgraph_rag_agent.py | `node_direct_llm` | `direct` |
| langgraph_rag_agent.py | `node_reviewer` | `review` |
| langgraph_rag_agent.py | `node_writer` | `write` |
| langgraph_rag_agent.py | `_do_rewrite` ×2 | `rewrite` |
| langgraph_rag_agent.py | `_do_grade` | `grade` |
| langgraph_rag_agent.py | `_do_generate` | `generate` |
| langgraph_rag_agent.py | `_do_plan` / `_do_plan_supplement` | `plan` |
| langgraph_rag_agent.py | `_compress_history` | `compress` |

---

## 七、后续可延伸

网关建好后，下面几件事都变成了「配置问题」而非「架构问题」：

- **Agent 并行调度**：连接池 + 限流已就位，并发调用不会打爆后端
- **成本治理**：`price_in/price_out` 配上即可按模型出账单
- **灰度与 A/B**：路由表里调整链路顺序即可切流量
- **多租户**：令牌桶按 key 分桶即可扩展成租户级配额
