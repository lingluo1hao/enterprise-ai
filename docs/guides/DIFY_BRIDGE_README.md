# Dify 双向桥：让 Dify 和你的 RAG 互相吃对方的成果

> 出口：把本项目的混合检索做成 Dify「外部知识库 API」（`POST /retrieval`），Dify 应用直接检索你带 ACL 的存量文档。
> 入口：把 Dify 应用注册成 `llm_gateway` 的第 9 个 provider，白拿限流 / 熔断 / 计费 / 降级。
> 两个方向各自成立，串起来才是完整闭环。

---

## 0. 先说结论：为什么不"Dify 替掉你的 RAG"

这是最容易走歪的一步。看起来更省事的路是：把文档传进 Dify，用它的知识库，自己的检索层就可以退休了。但对着代码量一遍就知道不行：

| 能力 | 本项目（`advanced_rag_agent.py` + `langgraph_rag_agent.py`） | Dify 自带知识库 |
|---|---|---|
| 召回 | dense 向量 + sparse BM25 双路，RRF 融合（`advanced_rag_agent.py:866` `_milvus_search`） | 单路向量召回，可选混合 |
| 精排 | bge-reranker-v2-m3 cross-encoder，带 5xx 重试与优雅回退（`langgraph_rag_agent.py:1983`） | 需另接 rerank 模型 |
| 跨 query | RRF 跨多个改写 query 融合，原句置顶（`_rrf_fuse_queries`） | 无 |
| 租户隔离 | Milvus `partition_key(tenant_id)` + scalar `expr` 下推（`advanced_rag_agent.py:878-884`） | 知识库一级 |
| 密级 | `access_level` = public / restricted，`config/access_rules.yaml` fail-closed | 无 |
| 拥有者 | `user_id` 维度过滤（自己上传的可见） | 无 |
| 应用层兜底 | `AccessControlFilter.filter_results` 第二道闸 | 无 |

**结论：检索留在本项目，Dify 只做编排。** 把 Dify 摆在"检索层"的位置，等于把三层 ACL 换成一个知识库下拉框。

那 Dify 的价值在哪？在两个缺口：

1. **编排缺口**：多步工作流、条件分支、工具调用、可视化调试，Dify 比手写 LangGraph 快一个数量级；
2. **交付缺口**：把能力包成一个「应用」，给出 API Key、会话、引用溯源，业务方自己能接。

于是桥的两个方向就有了明确分工。

---

## 1. 架构

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              出口：Dify → 你的知识库                          │
│                                                                              │
│   Dify 应用（chatflow / agent / workflow）                                    │
│        │  检索节点                                                            │
│        ▼                                                                     │
│   POST {endpoint}/retrieval          ← dify_kb_server.py                     │
│        Authorization: Bearer <DIFY_KB_API_KEY>                               │
│        {knowledge_id, query, retrieval_setting, metadata_condition}          │
│        │                                                                     │
│        ├─ ① knowledge_id 白名单  ──→ tenant_id / role / user_id（写死，不可传） │
│        ├─ ② top_k 钳制到 20                                                  │
│        ├─ ③ _do_retrieve(host, [query], role, keep_score=True)               │
│        │      ↓ 线上那份检索代码本身（不重写）                                  │
│        │   dense+BM25 → RRF → bge-reranker 精排 → ACL 过滤                    │
│        └─ ④ score 换算：relevance_score → 0~1 越大越相关                       │
│        ▼                                                                     │
│   {"records":[{content, score, title, metadata}]}                            │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                              入口：你的代码 → Dify 应用                        │
│                                                                              │
│   业务代码 / Agent 节点                                                       │
│        ▼  gw.chat(system, user, task="dify", user="alice")                    │
│   llm_gateway.py  ── 令牌桶(RPM/TPM) → 熔断器 → 重试 → fallback 链 → 计费      │
│        ▼                                                                     │
│   DifyProvider（llm_gateway.py:716）                                          │
│        ├─ chat        → POST /v1/chat-messages                               │
│        ├─ agent       → POST /v1/chat-messages（强制 streaming）              │
│        ├─ completion  → POST /v1/completion-messages                         │
│        └─ workflow    → POST /v1/workflows/run                               │
│        ▼                                                                     │
│   Dify 应用（内部还可以再调上面的 /retrieval，形成闭环）                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

一句话概括两边各自拿到了什么：

- **Dify 拿到的**：带租户 + 密级 + 拥有者三重过滤的检索结果，不用迁移数据；
- **你的代码拿到的**：Dify 编排好的流程，但限流、熔断、计费、降级仍然归网关管，不会多出一个绕过治理的野通道。

---

## 2. 出口：Dify 外部知识库 API

### 2.1 协议（照官方文档逐字实现）

请求：

```http
POST /retrieval
Authorization: Bearer <DIFY_KB_API_KEY>
Content-Type: application/json

{
  "knowledge_id": "kb-default",
  "query": "定位方式有哪几种",
  "retrieval_setting": {"top_k": 4, "score_threshold": 0.5},
  "metadata_condition": {
    "logical_operator": "and",
    "conditions": [
      {"name": "file_name", "comparison_operator": "contains", "value": "指令表"}
    ]
  }
}
```

响应：

```json
{
  "records": [
    {
      "content": "定位方式有三种：基站定位、WiFi 定位、GPS 定位。",
      "score": 0.93,
      "title": "指令表.pdf · §3 定位",
      "metadata": {"file_name": "指令表.pdf", "page": 12, "tenant_id": "default", "knowledge_id": "kb-default"}
    }
  ]
}
```

三个**不能想当然**的字段约束：

| 字段 | 约束 | 违反的后果 |
|---|---|---|
| `score` | `0 ~ 1` 浮点，**越大越相关** | 越界 → Dify 报 `score must be between 0 and 1`，整次检索失败 |
| `metadata` | **必须是对象**，不能 `null` | Dify 解析异常，整个知识库不可用 |
| `records` | 无命中返回 `[]`，不是 404 / 不是省略字段 | Dify 侧被当成故障 |

错误码（官方固定值，`dify_kb_server.py:117`）：

| code | HTTP | 含义 | 本实现的触发条件 |
|---|---|---|---|
| 1001 | 403 | Authorization 头格式错 | 没有 `Bearer ` 前缀 / 只有前缀没值 / 用了 `Basic` |
| 1002 | 403 | 鉴权失败 | key 不匹配；**或服务端没配 key**（此时返回 500，绝不裸奔放行） |
| 2001 | 404 | 知识库不存在 | `knowledge_id` 未登记 / `enabled: false` / body 缺该字段 |

### 2.2 三个必须堵的口子

这三条不是"最佳实践"，是**不堵就出事**。

**口子 1：`knowledge_id` 必须白名单（最要命）**

`knowledge_id` 直接决定 `tenant_id` / `role` / `user_id` 三个检索参数，而 `tenant_id` 在 `advanced_rag_agent.py:878-884` 是 **f-string 拼进 Milvus expr** 的：

```python
expr = (f'(tenant_id == "{tenant_id}") and '
        f'((access_level == "public") or (user_id == "{user_id}"))')
```

如果允许 Dify 传任意字符串，等于把 Milvus 表达式注入的入口开放给外部系统。所以：

```python
kb = self.cfg.kb(knowledge_id)          # 只认 config/dify_bridge.yaml 里登记过的
if kb is None:
    self._error(404, ERR_KB_NOT_FOUND, ...)   # 直接 2001，不回落、不透传
```

注意**不能**"找不到就回落 default 租户"——那一行代码等于把默认租户对所有人开放。

**口子 2：`top_k` 必须钳制**

Dify 表单里能填 100。本项目单 query 召回深度是 `RETRIEVE_CANDIDATE_K = 20`（`langgraph_rag_agent.py:140`），填 100 的收益是 0，代价是 reranker 白跑。所以 `top_k = max(1, min(top_k, cfg.max_top_k))`。

**口子 3：`score` 语义必须换算（最阴）**

Dify：`0~1`，越大越相关。
本项目：距离，越小越相关；hybrid 下是 `-fused`，量级 `1/61 ≈ 0.016`。

把 `0.016` 当 score 传过去，结果不是报错，而是**「全部低于阈值 → 0 命中」**——一个不报错的假失败。三条换算路径（`dify_kb_server.py:395` `_to_dify_score`）：

| 优先级 | 数据来源 | 换算 | 说明 |
|---|---|---|---|
| ① | `metadata["_rerank_score"]`（reranker 的 `relevance_score`） | 在 `[0,1]` 原样用；越界则 `sigmoid` | 唯一能诚实当"相关度"用的量 |
| ② | 纯 dense 的余弦距离（`[0,2]`） | `1 - dist/2` | 方向不能反：距离小 = 相关度高 |
| ③ | hybrid 的 `-fused` | `rank_base - rank_step * rank_idx` | **只保证排序正确**，不代表真实相关度，会打一次 warning |

**这里是整件事最容易被做错的地方**：`_rerank` 原本在 `langgraph_rag_agent.py:2046` 附近把 `relevance_score` **直接丢掉**了：

```python
# 改造前
order = sorted(resp.json().get("results", []), key=lambda x: x["relevance_score"], reverse=True)
reranked = [docs[idx_map[it["index"]]] for it in order[:top_n]]   # ← 分数没了
```

真实相关度算完就扔，只留排序。所以这次给它加了 `keep_score` 开关（默认 `False`，线上一个字节不变）：

```python
# 改造后（langgraph_rag_agent.py:2046-2057）
for it in order[:top_n]:
    doc, rrf_score = docs[idx_map[it["index"]]]
    if keep_score:
        doc.metadata["_rerank_score"] = float(it["relevance_score"])
    reranked.append((doc, rrf_score))
```

为什么不干脆把 `(doc, score)` 里的 score 改成 `relevance_score`？因为下游 `_mmr_rerank`、`RANK_BUDGET` 上下文预算、`_build_context` 的章节排序全都约定"越小越相关"，改语义等于全线改口径。加一个私有 metadata 键，代价最小、影响面可控。

> ⚠️ 万一 llama.cpp 的 `/v1/rerank` 吐的是 **logits**（`-12 ~ 8`）而不是概率，直接当 score 会越界。所以换算函数自动判断值域，越界才压 `sigmoid`。这个坑不同版本行为不一致，已实测（见 §6）。

### 2.3 `metadata_condition` 支持到什么程度

Dify 的算子全实现了（`dify_kb_server.py:470` `_cmp_one`），中英文两套都认：

```
contains / not contains / start with / end with / is / is not
empty / not empty / = / ≠ / > / < / ≥ / ≤ / before / after
```

`name` 先按本项目 metadata 原生键取（`file_name` / `section_path` / `access_level` / `chunk_type` / `page` / `chunk_index`），取不到再看 `source`；都取不到按空串处理，所以 `empty` / `not empty` 也能正常用。`logical_operator` 支持 `and` / `or`。

未知算子**按不匹配处理**并打 warning，不静默放行——静默放行会让"过滤条件写错了"表现为"过滤没生效"，非常难查。

### 2.4 目录结构上的一个副作用

`retrieve()` 里对 `pairs` 做了一次**单调性修正**，只影响"混装"这一种情况：

```python
if rank_idx > 0 and meta.get("_rerank_score") is None:
    score = min(score, max(0.0, prev_score - 0.01))
```

正常情况下整批候选要么全有 reranker 分、要么全没有（reranker 挂了就整批退化），天然单调。唯一混装的是 **figure-aware 二次召回**：图页文档被 `_do_retrieve` 直接插到最前面（`langgraph_rag_agent.py:1931` ⑤ 分支），**没经过 reranker**，于是它拿排名归一化的 0.95，而紧跟其后的真实 `rerank 0.52` 反而更低 —— 用户在 Dify 的引用列表里会看到分数倒序，像 bug。

压在"上一条之下 0.01"而不是整体 running-min，是为了保住分辨率（running-min 会把后面几条全部抹平成 0.52）。

---

## 3. 入口：Dify 应用当 provider

### 3.1 四种应用类型，四套协议

`DifyProvider`（`llm_gateway.py:716`）按 `dify_app_type` 分流：

| `dify_app_type` | 端点 | 正文取值 | 备注 |
|---|---|---|---|
| `chat` | `/v1/chat-messages` | `answer` | 含 chatflow |
| `agent` | `/v1/chat-messages` | `answer` | **只支持 streaming**，blocking 直接 400 |
| `completion` | `/v1/completion-messages` | `answer` | query 放进 `inputs.query` |
| `workflow` | `/v1/workflows/run` | `data.outputs.{变量}` | 出参结构完全不同 |

`agent` 那条坑是这个 provider 存在的一半理由：Dify 的智能体应用**不支持 blocking**。所以 `invoke()` 内部对 `agent` 类型会自己走流式再把增量拼起来，对上层 `BaseProvider.invoke` 的契约保持不变：

```python
if self._app_type() == "agent":
    buf = []
    for piece in self.stream(system_prompt, user_prompt):
        buf.append(piece)
    ...  # 拼成 LLMResponse 返回
```

Token 用量取 `metadata.usage`（真实值）；workflow 只有总量 `data.total_tokens`，拆不出输入/输出，就全算到 `completion_tokens` —— 宁可高估也不漏计，TPM 限流不能少扣。

> Dify 的 `usage` 里还有 `prompt_unit_price` / `total_price`，那是 **Dify 按它自己登记的模型价**算的，与我们网关的计费口径不同。混用会让"token 用量"和"成本"两套数字互相矛盾，所以只取 token 数。

### 3.2 `system prompt` 往哪放

网关的调用方（`langgraph_rag_agent` 的 generate / grade / rewrite 节点）会传自己的 system prompt，而 Dify 应用在界面上也配了 system prompt。三种模式：

| 模式 | 行为 | 什么时候用 |
|---|---|---|
| `prepend`（默认） | `system + "\n\n" + user` 一起当 query | 通用；保住"资料里没有就说不知道"这类护栏 |
| `input` | 写进 `inputs[dify_sys_input]` | 应用里显式引用该变量 |
| `ignore` | 丢弃网关的 system prompt | 完全交给 Dify 应用自己控 |

默认 `prepend` 而不是 `ignore` 是刻意的：网关侧 system prompt 里带着拒答护栏，丢了会退化成裸模型，RAG 的拒答能力跟着一起没。

### 3.3 配置

`config/llm_gateway.yaml` 里已经加好了块，**默认 `enabled: false`**，且不写进 `default_chain`，所以没装 Dify 的环境完全感知不到：

```yaml
  dify-flow:
    provider: dify
    model: dify-flow
    base_url: http://192.168.200.128/v1     # 指向 Dify 的 /v1，路径由 provider 拼
    api_key_env: DIFY_APP_API_KEY           # Dify 应用「访问 API」里那个 app- 开头的 key
    timeout: 120.0
    rpm: 30
    tpm: 100000
    enabled: false                          # 打开它 + 在 routing.dify 里启用即可
    dify_app_type: chat
    dify_user: llm-gateway
    dify_sys_prompt_mode: prepend
    dify_app_inputs: {}
    dify_query_input: query
    dify_output_key: ""
    dify_conversation_id: ""
```

路由链单独一个键，不碰任何现有任务：

```yaml
  dify:       [dify-flow, local-qwen]
```

`dify-flow` 关掉时 `chain_for()` 会把它过滤掉，自动落到 `local-qwen` —— **Dify 没装或挂了，业务照跑**，而不是抛异常。这正是网关降级该有的样子。

`dify_conversation_id` 默认空是刻意的：网关本身无状态，记住一个 `conversation_id` 会把不同用户的上下文串到一起。这个坑比"少一点记忆"严重得多。

---

## 4. 部署

### 4.1 配密钥

```bash
# .env 追加两行（值自己生成，别用示例）
#   python -c "import secrets;print(secrets.token_urlsafe(32))"
DIFY_KB_API_KEY=<随机串>          # 出口：Dify 来调我们时带的 key
DIFY_APP_API_KEY=app-xxxxxxxx     # 入口：我们调 Dify 应用时带的 key
```

密钥只写环境变量名进 yaml，值永远不进配置文件——与 `llm_gateway.yaml` 同一套纪律。

### 4.2 自检（不连 Milvus）

```bash
python dify_kb_server.py --check
```

输出会直接告诉你 Dify 侧该填什么：

```
监听地址      : 0.0.0.0:8300
密钥是否已配  : 是
top_k 上限    : 20
knowledge_id          tenant    role    enabled  label
kb-jm                 jm        user    是        JM 租户 · 指令表/协议文档
kb-yh                 yh        user    是        YH 租户 · 学生证设备通讯协议
kb-default            default   user    否        默认租户 · 当前 0 篇（停用，见注释）
kb-default-admin      default   admin   否        默认租户 · 含机密文档（admin 视角）
Endpoint（填进 Dify）: （监听 0.0.0.0，下面每个地址都可填）
    http://192.168.0.106:8300
    http://192.168.200.1:8300
    http://192.168.55.1:8300
外部知识库 ID 候选   : kb-jm, kb-yh
```

**注意两处口径变化（2026-10-25 实测后改的）**：

1. `kb-default` 已停用。`config/tenants.yaml` 里写着它是「存量/公共文档所在租户」，
   但 Milvus `rag_docs` 实际统计是 **default=0 / jm=526 / yh=322**（合计 848）。
   登记表指向一个空租户，调用方只会看到「200 + 空结果」，把「配置指错」
   误判成「检索坏了」。现在换成有真实数据的两条：`kb-jm` / `kb-yh`。
2. Endpoint 不再只打印一个 IP。`_local_ip()` 取的是默认路由网卡，在
   「Dify 跑 VMware NAT 虚拟机」的场景下会给出**宿主机办公网地址**
   （实测 `192.168.0.106`），而虚拟机只能走 NAT 网关那一张网卡
   （`192.168.200.1`）。照旧日志填进 Dify 就是 `failed to connect`，
   还会被误判成防火墙问题。现在把本机所有 IPv4 都列出来，自己挑同网段的那个。

退出码 0 = 配置可用；2 = 有知识库没登记或全被 disable。

### 4.3 起服务

```bash
python dify_kb_server.py --config config/dify_bridge.yaml
# 改监听地址/端口：
python dify_kb_server.py --host 0.0.0.0 --port 8300
```

服务**懒连接** Milvus：起进程时不连，第一次检索才连。好处是容器编排里"先起服务、后起向量库"不会卡住，Dify 那边也能先建好外部知识库再点"测试连接"。

`--host 0.0.0.0` 只在 Dify 跑在别的机器/容器时才用，并且**必须**用防火墙或安全组限制来源网段。数据不出内网不等于端口对全网开放。

### 4.4 Dify 侧配置

`知识库 → 连接外部知识库`：

| 字段 | 值 |
|---|---|
| API Endpoint | `http://<与 Dify 同网段的宿主机IP>:8300`（不要带 `/retrieval`，Dify 自己拼） |
| API Key | 与 `DIFY_KB_API_KEY` 一致 |
| 外部知识库 ID | `kb-jm` 或 `kb-yh`（必须在 `config/dify_bridge.yaml` 白名单里且 `enabled: true`） |

**Endpoint 该填哪个 IP**：`python dify_kb_server.py --check` 会把本机所有 IPv4 都列出来。
Dify 在 VMware NAT 虚拟机里时，虚拟机侧只能访问 **NAT 网关网卡**那个地址
（本项目是 `192.168.200.1`），宿主机办公网地址（`192.168.0.x`）在虚拟机里不通。

填完点「测试连接」。失败时先看 `curl` 能不能打通（下一步），再回头看 Dify 的网络能不能到宿主机。

### 4.5 手工验证

```bash
# 出口：模拟 Dify 打一发
curl -sS -X POST http://127.0.0.1:8300/retrieval \
  -H "Authorization: Bearer $DIFY_KB_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"knowledge_id":"kb-jm","query":"定位方式",
       "retrieval_setting":{"top_k":3,"score_threshold":0.0}}'

# 报错时看错误码：1001 头格式 / 1002 密钥 / 2001 知识库 ID

# 健康检查（含累计请求数、当前登记的库）
curl -sS http://127.0.0.1:8300/health
```

### 4.6 入口侧验证

```python
import llm_gateway as G
gw = G.LLMGateway()          # 走 config/llm_gateway.yaml
print(gw.chat("你是简洁助手。", "用一句话说明什么是混合检索", task="dify"))
```

`dify-flow` 的 `enabled` 还是 `false` 时，这条会落到 `local-qwen` —— 这就是降级链在工作，不是 bug。

---

### 4.7 Web 控制台（`/dify`）

桥服务是纯 HTTP、要敲 curl 才能看状态。为此在 `rag_web_server.py` 里加了一个
运维控制台，**仅管理员可见**（请求流水带原始 query、演练能读租户正文）。

```bash
python rag_web_server.py            # 或 waitress-serve --port=8080 rag_web_server:app
# 打开 http://127.0.0.1:8080/dify ，用管理员账号登录
# 入口：/kb 顶栏的「Dify 接入」
```

五个页签，数据**全部来自真实出口**，取不到就显示「未取到」，不填估算值：

| 页签 | 内容 | 数据来源 |
|---|---|---|
| 概览 | 6 个 KPI（状态/知识库/请求/命中/错误/p50·p95）、依赖健康矩阵、精排状态、限额、最近 5 条请求 | 桥 `/health` + 服务端并发 TCP/HTTP 探活 |
| 知识库 | 登记表：knowledge_id / 租户 / 角色 / 启用 / **实测 chunk 数** | 桥 `/health` + Milvus `count(*)`（5 分钟缓存） |
| 检索演练 | 选库 + query + top_k + 阈值 → 真实检索，结果卡片带分数条与元数据 | `POST /api/dify/retrieval` → 桥 `POST /retrieval` |
| 请求流水 | 最近 40 条：时间/库/query/命中/耗时/HTTP/备注，8 秒自动刷新 | 桥 `/health` 的 `recent`（内存环形缓冲 40 条） |
| 接入指引 | 多网卡 Endpoint（标出「虚拟机用这个」）、错误码表、可复制的 curl | 服务端枚举本机 IPv4 |

**为什么探活不用 `urllib.request`**：它读 `http_proxy` 环境变量。本机挂着
Clash（`127.0.0.1:7897`）时，访问内网 `192.168.x` 会走代理并返回 **502**，
看起来像「虚拟机服务挂了」。实测同一个端口：`curl` 报 `HTTP=502`，
`curl --noproxy '*'` 才是 `000`（真连不上）。所以控制台一律用 `http.client` 直连。

**精排状态单独一张卡**的原因：`RERANK_ENABLED=true` 但端点不可达时，检索**仍然成功**，
只是所有 score 退化成 `0.95/0.90/0.85…` 的名次阶梯。这时 `score_threshold`
不再代表相关度，而日志里只有一句「已退化为排名归一化」。控制台把开关、端点、
可达性、退化结论并列显示，一眼能看出是「没开」还是「开了但服务没起」。

## 5. 排障表

| 现象 | 根因 | 修法 |
|---|---|---|
| Dify 里检索永远 0 命中，且不报错 | score 语义反了 / 量级 0.016 直接当 score | 看 `_to_dify_score` 走了哪条路径；确认 `RERANK_ENABLED=true` 让 `_rerank_score` 出现 |
| 日志里刷 "退化为排名归一化 score" | reranker 没开或挂了 → 整批没有 `_rerank_score` | 检查 `RERANK_URL` 可达、`RERANK_ENABLED`；warning 每进程只打一次 |
| 引用分数倒序（第 3 条 0.52、第 4 条 0.80） | figure 召回插队且未经 rerank | 已由 §2.4 的单调性修正处理；若复现请检查 `_do_retrieve` 的 figure 分支 |
| 每次请求都 1001 | `Authorization` 少了 `Bearer ` 或用了 `Basic` | Dify 侧 API Key 原样填，别自己拼前缀 |
| 全 2001 | `knowledge_id` 与 yaml 里不一致（**大小写敏感**） | `--check` 看候选列表，逐字符对 |
| 服务端 500 + `API Key（环境变量 ... 为空）` | `.env` 没加载 / 变量名拼错 | 确认进程环境里有 `DIFY_KB_API_KEY`；这是**故意**不裸奔放行的行为 |
| Dify 侧报 `score must be between 0 and 1` | 某条记录的 score 越界 | 跑 `python tests/test_dify_bridge.py`，第 2 节会把所有路径的 score 值域打出来 |
| agent 应用 400 | 用了 blocking | provider 会自动绕，若仍 400 说明被当成 `chat` 类型了，改 `dify_app_type: agent` |

---

## 6. 验证到了哪一步（诚实版）

跑：

```bash
python tests/test_dify_bridge.py
```

**116 项全绿**，覆盖：

| 组 | 验的是什么 | 怎么验 |
|---|---|---|
| 1 | 配置解析、`knowledge_id` 白名单（含大小写敏感）、密钥从环境变量取 | 纯函数 |
| 2 | score 三条换算路径、值域 `[0,1]`、方向不能反 | 纯函数 + 全路径取值域 |
| 3 | `metadata_condition` 全部算子 + and/or + 未知算子 | 纯函数 |
| 4 | `/retrieval` 端到端：1001/1002/2001、钳制、阈值、过滤、出参整形、后端异常→500、`/health` | 本地起服务 + stub 掉 `_retrieve_pairs`（**其余是生产代码本身**） |
| 5 | `DifyProvider`：四种应用 × blocking/streaming、agent 绕法、sys prompt 三模式、error 事件转异常 | 本地 stub Dify（按真实报文格式回包） |
| 6 | 与生产代码的接缝：`keep_score` 参数存在、默认 `False`、确实转传给 `_rerank` | AST 校验，不拉依赖 |
| 7 | **真实 `_rerank` 代码**：`keep_score` 写入 `_rerank_score`、默认不开时零改动、原始 logits 经桥换算后顺序不变 | 只 stub reranker 的 HTTP 端点，其余全是线上代码 |

第 7 组值得单独说：stub reranker 故意回 **logits**（`5.2 / 0.4 / -3.1`）而不是概率，因为线上不同版本的 llama.cpp 行为不一致，按下限（最难的一种）验，能过就两种都能过。实测换算结果 `[0.9945, 0.4, 0.0431]`，顺序与精排一致。

**未能验证的部分（环境限制，不掩盖）**：

- 真实 Milvus（`192.168.200.128:19530`）与真实 reranker（`:11436`）的端到端检索 —— 本次验证时虚拟机未开机，三个端口（含 Ollama `:11434`）全部超时。所以「真实文档 → 真实 rerank 分 → Dify score」这条完整链路**只验到了 reranker 的 HTTP 边界**，边界之外的 stub 是我构造的。
- 真实 Dify 实例的「测试连接」与引用溯源展示 —— 本机没有 Dify 部署。

要补齐这两项，前置条件就一句话：**把虚拟机开起来，跑一次 `curl` 和一次 Dify 的连接测试**。

---

## 7. 改了哪些文件

| 文件 | 改动 | 位置 |
|---|---|---|
| `dify_kb_server.py` | **新增**。出口服务：`/retrieval` + `/health`，标准库 `http.server`，零新增依赖 | 全文 |
| `config/dify_bridge.yaml` | **新增**。端口 / 密钥变量名 / 限流上限 / `knowledge_id` 白名单 / score 参数 | 全文 |
| `langgraph_rag_agent.py` | `_do_retrieve` 与 `_rerank` 加 `keep_score`（默认 `False`）；开启时写 `metadata["_rerank_score"]` | `:1877` / `:1983` / `:2046` |
| `llm_gateway.py` | `ModelConfig` 加 9 个 `dify_*` 字段；新增 `DifyProvider`；注册表加 `"dify"` | `:134` / `:716` / `:971` |
| `config/llm_gateway.yaml` | 新增 `dify-flow` 模型块（`enabled: false`）+ `routing.dify` | models / routing |
| `tests/test_dify_bridge.py` | **新增**。116 项自测，零外部依赖 | 全文 |

**为什么用标准库 `http.server` 而不是 Flask/FastAPI**：这是旁挂在 `rag_web_server.py` 之外的一个进程，为一个接口把整套 Web 框架再拉一遍不值；更重要的是避免 `import rag_web_server` 的副作用——那个模块 **import 时就替换 `builtins.print`** 并装全局 stdout 复用器。

**为什么不自己写一套检索**：`DifyRetrievalService._retrieve_pairs` 直接调 `LangGraphRAGApp._do_retrieve`，宿主用 `_BridgeHost`（抄 `evalkit/harness_retrieval.py:66` 的 `_LiteHost` 手法，`__getattr__` 把兄弟方法绑到自身）。线上以后改检索（加一路召回、换 reranker），桥自动跟着变，不需要同步改两份。

---

## 8. 一句话记住这件事

> **Dify 负责"怎么用"，你负责"知道什么"。**
> 桥的两个方向，就是让这两件事各归各位——而不是让其中一个吞掉另一个。
