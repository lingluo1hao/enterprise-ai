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

### 4. Token 用量持久化与按用户查询

真实 token 不能只打日志——进程重启就丢，也无法区分用户。配置 `usage_db` 后，每次调用的真实 token 数会落盘到 SQLite（纯标准库 `sqlite3`，零新增依赖）。

`user` 标识从 Web 层（真实用户名/角色）经 Agent 一路透传到 `chat()`，所以天然支持「某用户查自己历史用量」：

```python
# llm_gateway.yaml
usage_db: ./llm_usage.db   # 非空落盘；留空则仅进程内累计，重启即丢

# 查询（gateway 实例上）
gw.user_usage("alice")              # 累计：calls / prompt+completion tokens / 成本 / 最近活跃时间
gw.usage_log("alice", limit=50)     # 最近明细（不传 user 看全部，管理员视角）
gw.usage_range(start_ts, end_ts)    # 某时间区间（如「本月烧了多少」）
gw.top_users(limit=50)              # 全用户排行（按 token 降序，后台看板用）
gw.metrics()["usage_persisted"]     # True=已落盘，重启后历史仍在
```

> 不配置 `usage_db` 也能跑，但用量只在内存累计，重启即丢。
> 注意：流式调用（`stream_chat`）的 token 是估算值（Ollama 的 usage 在末帧，当前流式实现未抓真实数），日志里标了 `(est)`；非流式是真实值。

#### 网页入口（无需写代码）

| 入口 | 位置 | 内容 |
|------|------|------|
| **我的用量** | 主页右上角 📊 | 当前账号的调用数 / token 总量 / 输入输出拆分 / 成本 + 最近 100 条明细，支持 今日 / 7天 / 30天 / 全部 |
| **Token 用量看板** | `/admin` → 📊 Token 用量 | 全站汇总 + 用户排行榜 + 全量明细（需管理员 Token） |

主页 👤 chip 可切换用量归属账号（存 localStorage），提问时随请求上报。对应接口：

```
GET /api/usage/me?user=alice&range=today|7d|30d|all&limit=100   # 单用户（公开）
GET /api/admin/usage/top?range=7d&limit=100                     # 全用户排行（管理员）
```

#### 为什么用 SQLite 而不是 MySQL？

这是本项目**最容易被质疑的一个选型**——毕竟系统里已经跑着 MySQL（`memory_store.py` 用它存对话历史和断点快照），为什么用量不顺手写进去？

**答案是：这是一次刻意的边界隔离，不是「没有数据库可用」。**

| 维度 | 本场景实际情况 | 结论 |
|------|---------------|------|
| **依赖边界** | 网关的设计契约是「仅依赖 Python 标准库」，可被 Agent / MCP Server / 单测任意端独立复用 | 一旦引入 `pymysql`，网关就绑死在本项目的基础设施上，**不再是可搬走的组件** |
| **写入模型** | 进程内单例、单写者、纯 INSERT 追加；一次 LLM 调用才写一条，而调用本身耗时 0.5~3s | 写库那 0.1ms 完全被淹没，SQLite「写并发差」的短板压根没被触及 |
| **数据量级** | 运行一段时间后 `.db` 文件仅 12KB；按日均 1 万次调用估算，一年也就几十 MB | SQLite 单文件撑到几十 GB 无压力，远未触及天花板 |
| **查询复杂度** | 只有 `SUM` / `GROUP BY user` / `WHERE ts BETWEEN` / `ORDER BY ... LIMIT` 四类 | 全是标准 SQL，两边写法一模一样，**换 MySQL 收益为零** |
| **部署门槛** | 开源项目，别人 clone 下来应当能直接跑 | 用量是辅助功能，不该让人为了看 token 先装一个数据库 |

还有一条关键的**故障域**考虑：MySQL 挂了，业务侧记忆功能会降级（这是已知且可接受的）；但如果用量也写 MySQL，就等于**让计量系统和业务系统同生共死**。计量应该是最后一个倒下的东西——它是你排查故障时唯一还能看的账本。

**并发安全怎么保证？** `threading.Lock` 串行化写入 + `check_same_thread=False`：

```python
with self._lock:
    self._conn.execute("INSERT INTO usage_log (...) VALUES (...)")
    self._conn.commit()
```

SQLite 被诟病的并发问题出在**多进程/多连接同时写**，单进程多线程加锁后完全安全。

**迁移成本被刻意压到最低。** 所有 DB 操作都收敛在 `UsageStore` 一个类里，外部只调 `record()` / `user_usage()` / `top_users()` / `usage_range()`。真要换 MySQL，**只需重写这一个类**——16 处调用点、两个 Web API、前端页面一行都不用动。

#### 什么时候该换掉 SQLite

不回避它的天花板。以下任一条成立就该迁移：

- **网关多实例部署** —— 两个进程写同一个 `.db` 会锁冲突。上 K8s 多副本时**必须换**，这是最常见的触发点。
- **写入超千 QPS** —— 单写者模型成为瓶颈，开 WAL 也顶不住。
- **要和业务库联表** —— 比如按部门归集成本、关联用户表做计费，跨库 JOIN 做不了。
- **长周期趋势分析** —— 按小时粒度留存半年以上，更适合 ClickHouse / TimescaleDB 这类时序或列存。

> 选型不是选最强的，是选**当前规模下复杂度最低、同时留好退路的**。单机部署上 MySQL 是过度设计，多实例还用 SQLite 是欠考虑——关键是看清自己在哪个阶段，并把切换成本提前锁死在一个类里。

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

# ---- Token 用量持久化（按用户查历史）----
usage_db: ./llm_usage.db  # 非空 → 落盘 SQLite，支持按用户/按时间查历史用量；
                          # 留空 → 仅进程内累计，重启即丢
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

43 项断言全部通过，覆盖：

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
| 11 | Token 用量持久化 + 按用户查询 | 落盘 SQLite 重启可查；`alice`/`bob` 用量隔离；`usage_log`/`usage_range` 正确；`chat_detailed(user=...)` 写入用户；`usage_persisted=True`；`top_users` 按 token 降序聚合；异常 latency 被钳制、不污染看板 |

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
