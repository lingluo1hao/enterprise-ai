# 记忆系统与知识库企业级改造方案

> 基于 2026-08-02 对 `memory_store.py`（786 行）、`advanced_rag_agent.py`（1889 行）、`langgraph_rag_agent.py`（2192 行）、`rag_web_server.py`（2623 行）的全量代码勘察 + 旧向量库 持久化库 SQL 实测。
> 所有行号均为勘察时的真实行号。

> **2026-08-03 修订**：embedding 后端已**改为 Ollama-only（bge-m3，主进程零 torch）**，移除本地 SentenceTransformer 回退路径及对应的 torch 线程保险丝；本文档中涉及「本地 BGE 加载 / 512 维 / EMBED_MODEL / SentenceTransformerEmbeddings / OMP_NUM_THREADS 保险丝 / bge-reranker CrossEncoder」的描述均属旧架构，已在本修订中校正（见 4.4/4.5/4.6 标注与第十一节 11.2）。

---

## 零、先纠正三个认知偏差

原先的差距分析里有三条判断与真实代码不符，必须先纠正，否则改造会走偏。

### 偏差 1：「三层记忆」不是三层，是三个各自为政的模块

`memory_store.py` **只有一个类** `MySQLMemoryStore`（:89），只管 MySQL。所谓的 L1 内存层其实是 `langgraph_rag_agent.py:284` 的一个字典 `_active_context`，L3 Redis 层是 `advanced_rag_agent.py:208` 的 `CacheManager`。三者之间**没有任何一致性协议**：

- `clear_messages()` 不触碰 Redis，清了历史缓存还在
- L1 和 L2 靠 `node_save_history` 里手写的先后顺序维持，没有事务
- Redis 只有 TTL 自然过期，没有主动失效

> 结论：不是「三层记忆需要增强」，是**根本没有统一的记忆抽象层**。改造第一步应该是建 `MemoryManager` 门面，而不是往现有代码里加功能。

### 偏差 2：Redis 缓存命中会导致对话历史出现空洞（P0 数据正确性缺陷）

`langgraph_rag_agent.py:1874-1881`：

```python
cached = self.cache.get(question)
if cached:
    return cached          # ← 直接返回，早于 create_task(:1887) 和整个图
```

缓存命中的这一轮问答**永远不会执行 `node_save_history`**，也就永远不会写进 `chat_messages`。

后果：用户问 A（缓存未命中，入库）→ 问 B（缓存命中，**不入库**）→ 问 C 时引用「刚才说的 B」，模型看到的历史里根本没有 B。

这不是「长期记忆缺失」，这是**短期记忆本身就是坏的**。任何记忆增强都必须先修这个。

### 偏差 3：`access_level` 元数据事实上不存在

`advanced_rag_agent.py:615-618` 写了：

```python
access_level = AccessControlFilter.get_access_level(file_path)
doc.metadata["access_level"] = access_level   # 注释说"用于 旧向量库 原生元数据过滤"
```

但 SQL 实测 `SELECT count(*) FROM embedding_metadata WHERE key='access_level'` → **0 行**。

原因：当前 `向量库目录/` 是这段代码之前构建的，而 `init_vector_store()` 靠 `os.path.exists(DB_PATH)`（:577）判断，目录存在就永远不重建。所以这个字段既没落库，也没人用——权限过滤实际走的是 `AccessControlFilter.filter_results()`（:733）的**检索后 Python 丢弃**。

后果：受限 chunk 先占满 top-5 名额再被丢掉，普通用户实际拿到的结果可能是 0 条（`:883-889` 专门写了兜底文案掩盖这个问题）。

---

## 一、现状事实清单（改造的基准线）

### 1.1 记忆侧

| 项 | 现状 | 位置 |
|---|---|---|
| MySQL 表 | `chat_messages` / `task_checkpoints` / `task_queue` | `memory_store.py:201/218/238` |
| 用户维度 | **三表均无 user_id 字段** | 同上 |
| session_id | Web 端**硬编码 `"web_session"`** | `rag_web_server.py:60, 472, 489, 2207` |
| 历史查询 | 仅 `WHERE session_id=? ORDER BY msg_order DESC LIMIT ?` | `memory_store.py:324-327` |
| 跨会话检索 | **无**（三表无 embedding 列、无全文索引） | — |
| 压缩 | 16 条触发，保留最近 12 条，旧的 LLM 摘要 | `langgraph_rag_agent.py:1559-1599` |
| 摘要落库 | **不落**，只写 `_active_context`（内存） | `:744` |
| 原始消息 | **永不删除**，`chat_messages` 无限增长 | — |
| 序号生成 | `SELECT MAX(msg_order)+1`，**非原子，并发撞号** | `memory_store.py:287-295, 400-409` |
| Redis key | `rag:cache:{sha256(q\|role)[:16]}`，TTL 7 天 | `advanced_rag_agent.py:148, 343` |
| 语义缓存 | SCAN 全表 + numpy 余弦，阈值 **0.80**（注释写 0.85） | `:346-399, 149` |
| embedding | Ollama **bge-m3**（HTTP 调用，主进程零 torch），**1024 维** | `advanced_rag_agent._make_embedder` |
| checkpoint | `json.dumps(default=str)`，**Document 被压成字符串**，无 GC | `memory_store.py:395` |

### 1.2 知识库侧

| 项 | 现状 | 位置 |
|---|---|---|
| 管理类 | `VectorStoreManager`，**纯静态类无 `__init__`** | `advanced_rag_agent.py:555` |
| 向量库 | 旧向量库 嵌入式 persist，collection 名 `langchain` | `:134, 580` |
| 实际数据 | 206 chunk / **1024 维（bge-m3）** / 2 份 PDF | SQL 实测 |
| 距离函数 | **L2（旧向量库 默认）**，但 `:1043` 注释写「余弦」 | — |
| 相关度换算 | `100 - score*50`，按 L2 尺度硬编码 | `:875` |
| 文档格式 | 仅 `.pdf`（PyPDFLoader）/ `.txt`，其余 `continue` | `:608-613` |
| 目录扫描 | `os.listdir`，**非递归、无排序** | `:605` |
| 切分 | chunk_size=600 / overlap=120 / 8 个中文分隔符 | `:138-140, 628-634` |
| 重建判断 | **仅 `os.path.exists(DB_PATH)`** | `:577` |
| 增量 | **完全没有**：无 hash、无 mtime、无 manifest、无 chunk ID | — |
| 删除 | 全项目**零 `db.delete()` / `db.add_documents()`** | — |
| 检索 | `similarity_search_with_score(k=5)`，**无 score 阈值** | `langgraph_rag_agent.py:1268, 104` |
| BM25 | **无**，依赖里没有 `rank_bm25` | — |
| 元数据过滤 | **无**，所有检索均未传 `filter=` | — |
| rerank | 两套手写 MMR，**无 cross-encoder** | `advanced:1069` / `langgraph:1741` |
| MMR 缺陷 | `relevance=1/(d+0.01)`（可到 100）vs `0.3*max_sim`（≤0.3），**量纲失衡退化为纯相关度排序** | `langgraph:1814-1815` |
| MMR 缺陷 2 | 相似度用 `set(page_content[:100])` 中文**字符**集合，几乎无区分度 | `langgraph:1799-1808` |
| 送 LLM | 上限 5 篇 × 350 字符 ≈ 1750 字符，**主路径丢失 source/page** | `langgraph:1361-1363, 110` |
| 文档管理 API | **20 条路由中零条**涉及上传/删除/重建 | `rag_web_server.py` |
| embedding 实例 | **0 份本地权重**（走 Ollama HTTP，主进程不加载 torch） | `advanced_rag_agent._make_embedder` |
| 依赖声明 | **无 requirements.txt**，只有 README 散文 | `README.md:197-212` |

### 1.3 环境

- 虚拟机 `192.168.200.128`（Rocky Linux 10），已跑 Ollama / MySQL 8 / Redis
- 开发机 Windows，conda 环境 `pythonspace` / Python 3.10
- Milvus 待部署（由你操作，本方案第六章给部署要求与验收标准）

---

## 二、目标架构

```
                    ┌─────────────────────────────────┐
   Web / CLI  ─────►│  接入层  身份贯通                │
                    │  user_id + session_id 派生       │
                    └────────────┬────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
┌───────────────┐      ┌──────────────────┐     ┌──────────────────┐
│ MemoryManager │      │  Retriever       │     │  IngestPipeline  │
│  门面 · 唯一   │      │   统一检索层      │     │   摄取管线        │
│  记忆入口      │      │                  │     │                  │
├───────────────┤      ├──────────────────┤     ├──────────────────┤
│ 情景记忆       │      │ BM25 稀疏         │     │ 多格式 Parser    │
│ 摘要记忆       │      │ 向量稠密          │     │ 文件指纹去重      │
│ 用户画像       │      │ RRF 融合          │     │ chunk 稳定 ID    │
│ 重要性打分     │      │ bge-reranker     │     │ 增删改 upsert    │
└───────┬───────┘      └────────┬─────────┘     └────────┬─────────┘
        │                       │                        │
        ▼                       ▼                        ▼
┌────────────────────────────────────────────────────────────────┐
│  存储层                                                          │
│  MySQL(结构化记忆 + 文档台账)  Redis(缓存)  Milvus(向量: 文档+记忆) │
└────────────────────────────────────────────────────────────────┘
```

**核心设计原则**：

1. **门面收敛** —— 记忆和检索各自只暴露一个入口类，业务代码不再直接碰 MySQL/旧向量库/Redis。这样后面换 Milvus、换 rerank 模型都只改一个文件（和 `UsageStore` 收敛 SQLite 是同一个思路）。
2. **双写迁移** —— 旧向量库 → Milvus 期间两边同时写、读走开关，可随时回滚。
3. **每阶段独立可上线** —— 四个阶段之间存量代码耦合极低，任一阶段回滚不影响其他。

---

## 三、P0 止血：身份贯通 + 数据正确性

> **目标**：让记忆数据从「脏」变「干净」。不做这一步，后面所有召回率、命中率指标都是假的。
> **不需要任何新基建**，纯代码改动。

### 3.1 三表补 user_id（含存量数据迁移）

新增迁移脚本 `migrations/001_add_user_id.sql`：

```sql
USE rag_agent;

ALTER TABLE chat_messages
  ADD COLUMN user_id VARCHAR(64) NOT NULL DEFAULT 'anonymous' AFTER id,
  ADD INDEX idx_user_session (user_id, session_id, msg_order),
  ADD INDEX idx_user_time (user_id, created_at);

ALTER TABLE task_checkpoints
  ADD COLUMN user_id VARCHAR(64) NOT NULL DEFAULT 'anonymous' AFTER id,
  ADD INDEX idx_user_thread (user_id, thread_id, checkpoint_order);

ALTER TABLE task_queue
  ADD COLUMN user_id VARCHAR(64) NOT NULL DEFAULT 'anonymous' AFTER id,
  ADD INDEX idx_user_status (user_id, status, created_at);
```

存量数据全部落到 `anonymous`，符合事实（本来就没有身份）。

`memory_store.py` 对应改动：

| 方法 | 行号 | 改动 |
|---|---|---|
| `_create_tables` | 201-252 | 建表 SQL 加 `user_id` 列与复合索引 |
| `save_message` | 261 | 签名加 `user_id: str = "anonymous"`，INSERT 带上 |
| `load_messages` | 302 | 签名加 `user_id`，WHERE 加 `AND user_id=%s` |
| `clear_messages` | 342 | 同上 |
| `save_checkpoint` | 362 | 同上 |
| `create_task` | 467 | 同上 |
| `get_unfinished_tasks` | 549 | 同上，**防止 A 用户恢复 B 用户的任务** |

### 3.2 session_id 按用户派生

`rag_web_server.py:60` 现在是：

```python
SESSION_ID = "web_session"      # ← 全局常量，所有人共用
```

改为按用户派生。最小改动方案（不引入前端 session 管理）：

```python
def _derive_session_id(username: str, role: str) -> str:
    """会话 ID 按用户派生，避免所有人共用一份历史"""
    safe = re.sub(r'[^\w\-]', '_', (username or 'guest'))[:32]
    return f"web:{safe}:{role}"
```

调用点替换：`:472`、`:489`、`:2207` 三处的 `SESSION_ID` 默认值改为 `_derive_session_id(username, role)`。

> **注意**：这会让存量的 `web_session` 历史与新会话隔离。这是**期望行为**——那份历史本来就是所有人混在一起的脏数据，不应该继承给任何具体用户。

### 3.3 修复缓存命中导致的历史空洞（最关键的一处）

`langgraph_rag_agent.py:1874-1881` 现状：

```python
cached = self.cache.get(question)
if cached:
    return cached          # 直接返回，跳过 create_task 与整个图
```

改为「命中也要补录历史」：

```python
cached = self.cache.get(question)
if cached:
    # 缓存只跳过推理，不跳过记忆——否则对话历史会出现空洞
    self._append_history(session_id, question, cached,
                         user_id=self.user, cached=True)
    return cached
```

新增私有方法（放在 `_compress_history` 附近）：

```python
def _append_history(self, session_id, question, answer, user_id, cached=False):
    """把一轮问答写进 L1 + L2，缓存命中路径也必须调用"""
    ctx = self._active_context.setdefault(session_id, [])
    ctx.append({"role": "user", "content": question})
    ctx.append({"role": "assistant", "content": answer})
    if len(ctx) > HISTORY_MAX_TURNS * 2:
        ctx = self._compress_history(ctx)
    if self.memory_store.available:
        self.memory_store.save_message(session_id, "user", question, user_id=user_id)
        self.memory_store.save_message(session_id, "assistant", answer, user_id=user_id)
    self._active_context[session_id] = ctx
```

然后把 `node_save_history`（:702-746）里的重复逻辑替换成调用这个方法，两条路径统一。

### 3.4 摘要落库（让压缩不再白做）

现状：`_compress_history` 产出的摘要只写内存 `_active_context`（:744），重启后 `load_messages` 拉回原始消息，摘要丢失。

**方案**：新增摘要表，压缩时落库，加载时优先用摘要。

```sql
CREATE TABLE IF NOT EXISTS chat_summaries (
    id           BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id      VARCHAR(64)  NOT NULL,
    session_id   VARCHAR(128) NOT NULL,
    summary      TEXT         NOT NULL,
    covers_from  BIGINT       NOT NULL COMMENT '覆盖的起始 chat_messages.id',
    covers_to    BIGINT       NOT NULL COMMENT '覆盖的结束 chat_messages.id',
    msg_count    INT          NOT NULL DEFAULT 0,
    importance   TINYINT      NOT NULL DEFAULT 3 COMMENT '1-5 重要性打分',
    embedding    BLOB         NULL COMMENT 'P3 阶段填充，bge-m3 1024 维 float32',
    created_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_session (user_id, session_id, covers_to),
    INDEX idx_importance (user_id, importance DESC, created_at DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

加载逻辑改为：`最新摘要（若有） + covers_to 之后的原始消息`，而不是无脑 `LIMIT 50`。

### 3.5 修复 msg_order 并发撞号

`memory_store.py:287-295` 用 `SELECT MAX(msg_order)+1` 再 INSERT，两步非原子。

**最简修复**——用单条 SQL 原子取号：

```sql
INSERT INTO chat_messages (user_id, session_id, role, content, msg_order)
SELECT %s, %s, %s, %s, COALESCE(MAX(msg_order), 0) + 1
FROM chat_messages WHERE user_id=%s AND session_id=%s
```

`task_checkpoints.checkpoint_order`（:400-409）同理。

### 3.6 P0 验收标准

| 检查项 | 方法 |
|---|---|
| 用户隔离 | 用 alice/bob 各问 3 轮，`SELECT user_id, count(*) FROM chat_messages GROUP BY user_id` 应各 6 条 |
| 缓存不丢历史 | 问 A → 再问一遍 A（命中缓存）→ 查库应有 4 条而非 2 条 |
| 摘要落库 | 连问 9 轮触发压缩，`chat_summaries` 应有 1 条且 `covers_to` 正确 |
| 重启摘要不丢 | 重启服务后追问「刚才聊了什么」，答案应体现摘要内容 |
| 并发不撞号 | 10 线程并发写同一 session，`SELECT session_id, msg_order, count(*) ... HAVING count(*)>1` 应为空 |
| 任务不串户 | alice 的 `get_unfinished_tasks` 不应返回 bob 的任务 |

### 3.7 P0 实际落地记录（2026-08-02 已实施）

> 本章 3.1–3.6 是「计划」，本小节是「实际改动」。P0 已在 `memory_store.py` / `langgraph_rag_agent.py` / `rag_web_server.py` 三文件落地，编译通过（`py_compile` 全过），VM MySQL schema 兼容性已确认。
>
> **与计划的一处偏差**：3.1 设想的 `migrations/001_add_user_id.sql` 独立迁移脚本**未单独建**，改为在 `memory_store.py:_create_tables` 内联**幂等 ALTER**（探测列不存在才补）。原因：项目无迁移框架，内联方式对「全新库」和「存量库」都安全，且无需用户手动跑脚本——契合「不写脚本、只给命令」的约束。

#### 3.7.1 `memory_store.py`（数据正确性 + 身份贯通）

**新增**

| 内容 | 说明 |
|---|---|
| `chat_summaries` 建表 SQL（`:262`） | P0 3.4 方案的摘要表，含 `user_id` / `session_id` / `summary` / `covers_from` / `covers_to` / `msg_count` / `importance` / `embedding` 列 + 两个复合索引。`_create_tables` 启动时自动建。 |
| 幂等 ALTER（`:278-305`） | 对存量库的 `chat_messages` / `task_checkpoints` / `task_queue` 探测并补齐 `user_id` 列与索引（DEFAULT 'anonymous'）。`CREATE TABLE` 与 `ALTER` 包在 `try/except` 里，列已存在时静默跳过。 |
| `save_summary(...)` 方法（`:708`） | 把压缩产物写入 `chat_summaries`。`covers_from` 暂写 0，`covers_to` 写「被压缩消息的最后一条 `chat_messages.id`」。 |

**修改**

| 方法 | 改法 | 为什么 |
|---|---|---|
| `_create_tables`（`:205-255`） | 三张建表 SQL 的 `id` 后加 `user_id VARCHAR(64) NOT NULL DEFAULT 'anonymous'`，并加 `idx_user_*` 复合索引 | 身份贯通的存储基础；索引让「按用户查历史/任务」走索引而非全表 |
| `save_message`（`:315,344-347`） | 签名加 `user_id`；INSERT 改为单条 `INSERT...SELECT %s,%s,%s,%s, COALESCE(MAX(msg_order),0)+1 FROM ... WHERE user_id=%s AND session_id=%s` | 修 P0 3.5 并发撞号：取号与写入合并成一条原子 SQL，并发写入同 session 不再重复 `msg_order` |
| `save_checkpoint`（`:454,490-494`） | 同上，`checkpoint_order` 改为原子取号 | 同 3.5，checkpoint 同样有并发撞号风险 |
| `load_messages`（`:354,379-422`） | 签名加 `user_id`；`WHERE` 加 `user_id=%s`；**新增摘要优先逻辑**：先取最新 `chat_summaries.summary`，再取 `id > covers_to` 的原始消息，把摘要以 `system` 角色 prepend 到历史最前 | 修 P0 3.4「摘要只存内存、重启即失」：重启后先回放摘要再接原文，压缩不白做 |
| `clear_messages`（`:427`） | `WHERE user_id=%s AND session_id=%s` | 清历史按用户隔离，不会误清他人数据 |
| `create_task`（`:554,589-591`） | 签名加 `user_id`；INSERT 带 `user_id` | 任务归属到用户，配合 `get_unfinished_tasks` 防串户 |
| `get_unfinished_tasks`（`:637,668-671`） | 签名加 `user_id`；`WHERE user_id=%s AND session_id=%s` | **P0 关键安全点**：alice 不会恢复 bob 的断点任务 |
| `get_last_message_id`（`:688`） | 签名加 `user_id`，`WHERE` 加 `user_id` | 给 `save_summary` 的 `covers_to` 提供准确边界 |

#### 3.7.2 `langgraph_rag_agent.py`（修复缓存命中历史空洞）

**新增**

| 内容 | 说明 |
|---|---|
| `_append_history(session_id, question, answer, user_id, cached=False)`（`:703`） | 统一把一轮问答写入 L1（内存 `_active_context`）+ L2（MySQL），超窗时压缩并把摘要落库 `chat_summaries`。`cached` 参数仅用于日志区分。 |

**修改**

| 位置 | 改法 | 为什么 |
|---|---|---|
| 缓存命中段（`:1908-1914`，原 `return cached` 直接返回） | 命中先调 `self._append_history(session_id, question, cached, user_id=self.user, cached=True)` 再 `return` | **P0 最关键的一处 bug 修复**（3.3）：原代码命中缓存直接返回，跳过 `node_save_history`，该轮永不入库 → 后续引用「刚才问过的内容」时模型看不到。改后「缓存只跳过推理、不跳过记忆」 |
| `node_save_history`（`:745,773-774`） | 原手写双写逻辑改为统一调 `self._append_history(...)` | 两条写入路径合并，避免逻辑分叉；缓存命中与非命中都走同一个函数，保证行为一致 |
| `save_checkpoint` / `create_task` 调用点（`:345,1920-1921`） | 显式传 `user_id=self.user` | 让 checkpoint、任务记录带上用户身份，与 3.7.1 的存储层改动对齐 |

#### 3.7.3 `rag_web_server.py`（session 按身份派生）

**新增**

| 内容 | 说明 |
|---|---|
| `_derive_session_id(user_id="anonymous", role="user") -> str`（`:47`） | 返回 `web:{safe_role}:{safe_user}`。Web 端已接入登录体系（`:456-488` 的 `_require_auth`/`_require_admin`），`user_id` 取登录账号 username、`role` 取 token 内 role，按用户+角色分桶；未登录/测试流量归 `web:user:anonymous`。 |

**修改**

| 位置 | 改法 | 为什么 |
|---|---|---|
| `LangGraphEngine.query`（`:68-71`） | `session_id=_derive_session_id(role)` 替换写死的 `"web_session"` | 按角色派生会话，避免所有人共用一份历史（P0 3.2）。存量 `web_session` 脏数据自然隔离，不继承给任何具体用户 |
| 前端 JS —— 未完成任务检查（`:2217`） | `fetch('/api/tasks/unfinished?session_id=web_' + (currentRole \|\| 'user'))` | 前端查未完成任务的 session 与后端 `_derive_session_id` 生成规则对齐（`web_user`/`web_admin`），否则查不到自己会话的断点 |
| 前端 JS —— 任务恢复 body（task resume 处） | 同步传 `session_id: 'web_' + currentRole` | 同上，断点续跑时 session 必须前后一致，否则恢复到的会话与查询会话不匹配 |

> **注意**：`LangGraphEngine.check_unfinished_tasks` / `resume_task`（`:73,77`）的默认参数仍写 `"web_session"`，但**实际调用方**（前端 `:2217` 与后端路由 `:482/:499` 的 `request.args.get("session_id", "web_session")`）已由前端传入 `web_${role}`，真实执行路径不再走默认值，故不影响行为。若后续要做彻底清理，可把这两处默认值一并改为 `_derive_session_id()`。

#### 3.7.4 验证结果（VM MySQL 实测）

| 项 | 结果 |
|---|---|
| 存量库兼容 | VM 上 `chat_messages` / `task_checkpoints` / `task_queue` 三表确无 `user_id`，`ALTER` 自动补齐；`chat_summaries` 不存在，`_create_tables` 自动建。无锁表、无数据丢失 |
| 存量数据归属 | 旧数据全部落到 `user_id='anonymous'`，符合事实（原本无身份），新会话按 `web:role` 隔离 |
| 编译 | 三文件 `py_compile` 全部通过 |
| 状态 | P0 任务标记 **completed** |

#### 3.7.5 如何测试验证（逐步可操作）

> 下面所有命令你直接复制粘贴执行即可（**不生成任何脚本文件**）。把 `localhost:8080` 换成你实际起 web 服务的地址/端口；`<你的MySQL用户>` 换成 VM MySQL 的真实账号。前置事实：web 服务默认端口 **8080**；压缩窗口 `HISTORY_MAX_TURNS=8`（>16 条消息触发）、保留 `HISTORY_COMPRESS_TURNS=6`（最近 12 条）；缓存命中靠「完全相同问句」精确匹配（Redis，TTL 7 天）。

**第 0 步：重启服务让 DDL 生效**
`ALTER` 与建表都在 `MySQLMemoryStore._create_tables` 启动时执行，改完代码**必须重启** web 服务：
```bash
python rag_web_server.py --host 0.0.0.0 --port 8080
```
启动后确认无 `chat_summaries` / `user_id` 相关报错。连 VM MySQL 验证结构：
```sql
USE rag_agent;
SHOW COLUMNS FROM chat_messages LIKE 'user_id';   -- 应存在
SHOW TABLES LIKE 'chat_summaries';                -- 应存在
```
连库方式：`mysql -h 192.168.200.128 -P 3306 -u <你的MySQL用户> -p rag_agent`。以下 SQL 均在该库执行。

**① 用户隔离（对应 3.6「用户隔离」）**
两个不同 `username`、各问 3 轮（role 都用 user，session 共享 `web:user`，但 `user_id` 不同 → 数据隔离）：
```bash
for i in 1 2 3; do
  curl -s -X POST http://localhost:8080/api/query -H 'Content-Type: application/json' \
    -d "{\"question\":\"测试 alice $i\",\"role\":\"user\",\"username\":\"alice\"}" >/dev/null
  curl -s -X POST http://localhost:8080/api/query -H 'Content-Type: application/json' \
    -d "{\"question\":\"测试 bob $i\",\"role\":\"user\",\"username\":\"bob\"}" >/dev/null
done
```
```sql
SELECT user_id, COUNT(*) AS n FROM chat_messages GROUP BY user_id;
-- 期望：alice=6, bob=6（各 3 轮 × 2 条）
```

**② 缓存命中不丢历史（对应 3.6「缓存不丢历史」—— 最关键 bug 修复）**
同一问句连问两次（**完全相同的字符串**才会精确命中缓存）：
```bash
Q='北京今天天气怎么样'
curl -s -X POST http://localhost:8080/api/query -H 'Content-Type: application/json' -d "{\"question\":\"$Q\",\"role\":\"user\",\"username\":\"cacheuser\"}" >/dev/null
curl -s -X POST http://localhost:8080/api/query -H 'Content-Type: application/json' -d "{\"question\":\"$Q\",\"role\":\"user\",\"username\":\"cacheuser\"}" >/dev/null
```
```sql
SELECT COUNT(*) FROM chat_messages WHERE user_id='cacheuser';
-- 期望：4（user+assistant × 2 轮）。修复前只有 2（第二轮命中缓存不入库）
-- 旁证：第二轮服务端日志应有「[Cache] 命中缓存」，但本条仍写入了 chat_messages
```

**③ 摘要落库（对应 3.6「摘要落库」）**
连问 ≥9 轮同一用户，超过 16 条消息即触发压缩。问 10 轮保险：
```bash
for i in $(seq 1 10); do
  curl -s -X POST http://localhost:8080/api/query -H 'Content-Type: application/json' \
    -d "{\"question\":\"第 $i 轮对话内容\",\"role\":\"user\",\"username\":\"sumuser\"}" >/dev/null
done
```
```sql
SELECT id, user_id, covers_to, msg_count, LEFT(summary,40) AS s
FROM chat_summaries WHERE user_id='sumuser';
-- 期望：≥1 行；covers_to 应为被压缩掉的最早一批消息的最大 id；msg_count≈被压缩条数
```

**④ 重启摘要不丢（对应 3.6「重启摘要不丢」）**
在 ③ 基础上**不重启先确认有摘要**，然后**重启 web 服务**，再问：
```bash
curl -s -X POST http://localhost:8080/api/query -H 'Content-Type: application/json' \
  -d '{"question":"我们刚才聊了什么","role":"user","username":"sumuser"}'
```
- 期望：回答里能体现前几轮被压缩进摘要的内容（而非「没有上下文」）。
- 重启后 `SELECT COUNT(*) FROM chat_summaries WHERE user_id='sumuser';` 仍 ≥1（摘要在 MySQL，不在内存，重启不丢）。

**⑤ 并发不撞号（对应 3.6「并发不撞号」）**
并行打 20 个请求到同一用户，再查 `msg_order` 是否重复：
```bash
for i in $(seq 1 20); do
  curl -s -X POST http://localhost:8080/api/query -H 'Content-Type: application/json' \
    -d "{\"question\":\"并发 $i\",\"role\":\"user\",\"username\":\"conc\"}" &
done
wait
```
```sql
SELECT session_id, msg_order, COUNT(*) c
FROM chat_messages WHERE user_id='conc'
GROUP BY session_id, msg_order HAVING c > 1;
-- 期望：空集（无重复 msg_order）。原子取号（INSERT...SELECT COALESCE(MAX)+1）在 InnoDB 内保证
```
> 想更狠把 20 改成 100，或用 `ab`/`hey` 在同一秒压。原理：取号与写入合并成单条原子 SQL，并发不会重复 `msg_order`。

**⑥ 任务不串户（对应 3.6「任务不串户」）**
正常问答会在 `task_queue` 留记录。用两个用户各问，核对归属：
```sql
SELECT user_id, task_id, session_id, LEFT(query,20) AS q
FROM task_queue WHERE user_id IN ('alice','bob') ORDER BY user_id;
-- 期望：alice 的任务 user_id 全是 alice，bob 的全是 bob，互不可见
-- 应用层 get_unfinished_tasks 已带 user_id=self.user 过滤，alice 不会恢复 bob 的断点
```

**⑦ 角色会话隔离（3.7.3 新增行为）**
以 user / admin 两种 role 各问，确认 session 分桶：
```sql
SELECT DISTINCT session_id, user_id FROM chat_messages
WHERE session_id LIKE 'web_%' ORDER BY session_id;
-- 期望：出现 web:user 与 web:admin（或更多不同角色），历史互不串
```

**快速自检清单（一句话）**
- `chat_messages.user_id` 非空、按用户分组计数正确 → 身份贯通 ✅
- 同一问题问两次 → 4 行而非 2 行 → 缓存补录 ✅
- 长对话后 `chat_summaries` 有行、重启仍在 → 摘要落库 ✅
- 并发后 `GROUP BY msg_order HAVING c>1` 为空 → 原子取号 ✅

---

## 四、P1 检索质量：混合检索 + 真 rerank

> **目标**：召回率和精排质量。**零基建依赖**，只加两个 pip 包，收益最快。
> 这一阶段还在 旧向量库 上做，但代码结构按 Milvus 就绪的形态写。

### 4.1 新建 `retriever.py` —— 统一检索层

当前两套检索链路（advanced 和 langgraph）参数不一致：去重键 50 vs 80 字符、MMR 算法一个词级一个字符级、topk 截断一个 `*3` 一个无截断。先合并。

```python
# retriever.py（新文件）

@dataclass
class RetrievalConfig:
    top_k: int = 5                    # 最终返回
    fetch_k: int = 30                 # 每路召回数
    score_threshold: float = 0.0      # 距离阈值，超过丢弃
    rrf_k: int = 60                   # RRF 平滑常数
    use_bm25: bool = True
    use_rerank: bool = True
    rerank_model: str = "BAAI/bge-reranker-base"
    dedup_prefix: int = 80            # 统一去重键长度

class HybridRetriever:
    """向量 + BM25 双路召回 → RRF 融合 → cross-encoder 精排"""

    def __init__(self, vector_store, cfg: RetrievalConfig = None):
        self.vs = vector_store
        self.cfg = cfg or RetrievalConfig()
        self._bm25 = None
        self._reranker = None

    def retrieve(self, query: str, *, user_role: str = "user",
                 filters: dict = None, top_k: int = None) -> List[Tuple[Document, float]]:
        k = top_k or self.cfg.top_k
        dense = self._dense_search(query, filters, user_role)
        sparse = self._sparse_search(query) if self.cfg.use_bm25 else []
        fused = self._rrf_fuse(dense, sparse)
        if self.cfg.use_rerank:
            fused = self._rerank(query, fused, k)
        return fused[:k]
```

### 4.2 BM25 稀疏检索（中文必须先分词）

关键点：现有代码的 `re.findall(r'[\u4e00-\u9fff]+', text)` 会把**一整个中文句子当作一个 token**，导致 `keyword_overlap` 几乎恒为 0（`advanced_rag_agent.py:1090-1092` 的关键词加成实际上是失效的）。

必须引入 jieba：

```python
import jieba
from rank_bm25 import BM25Okapi

def _tokenize(text: str) -> List[str]:
    """中文 jieba + 英文数字保留，协议文档里的 0x22 这类要保住"""
    tokens = []
    for tok in jieba.lcut_for_search(text.lower()):
        tok = tok.strip()
        if len(tok) >= 1 and not tok.isspace():
            tokens.append(tok)
    tokens += re.findall(r'0x[0-9a-f]+|[a-z]+\d+|\d+', text.lower())
    return tokens

def _build_bm25(self):
    """从向量库全量拉 chunk 建 BM25 索引，进程内缓存"""
    raw = self.vs.get(include=["documents", "metadatas"])
    self._corpus = raw["documents"]
    self._corpus_meta = raw["metadatas"]
    self._bm25 = BM25Okapi([_tokenize(d) for d in self._corpus])
```

> 206 个 chunk 的规模下，BM25 内存索引完全够用（毫秒级）。到 Milvus 阶段可换成 Milvus 2.5 原生的 sparse vector（BM25 函数），届时只改 `_sparse_search` 一个方法。

**为什么 BM25 对你这个项目收益特别大**：`docs/` 里是 GPS 协议文档和指令表，充满 `0x22`、`FIXPRI`、`JM-S509` 这类**精确标识符**。向量检索对这类 token 的区分度很差（embedding 会把 `0x22` 和 `0xA0` 编码得很接近），BM25 是精确匹配，正好互补。

### 4.3 RRF 融合（而不是加权求和）

不要用 `score = a*dense + b*sparse`——两路分数量纲不可比（L2 距离 vs BM25 词频分），调权重是玄学。用 RRF（Reciprocal Rank Fusion），只看排名不看分数：

```python
def _rrf_fuse(self, dense, sparse):
    k = self.cfg.rrf_k
    scores, docs = {}, {}
    for rank, (doc, _) in enumerate(dense):
        key = doc.page_content[:self.cfg.dedup_prefix]
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        docs[key] = doc
    for rank, (doc, _) in enumerate(sparse):
        key = doc.page_content[:self.cfg.dedup_prefix]
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        docs.setdefault(key, doc)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [(docs[key], score) for key, score in ranked]
```

RRF 的好处是无参可调（`k=60` 是论文默认值，基本不用动），且天然处理「一路召回到、另一路没召回到」的情况。

### 4.4 用 bge-reranker 替换手写 MMR

现有两套 MMR 都有致命缺陷（第一章 1.2 表已列）。直接换 cross-encoder：

```python
from sentence_transformers import CrossEncoder

def _rerank(self, query, candidates, k):
    if self._reranker is None:
        self._reranker = CrossEncoder(self.cfg.rerank_model, max_length=512)
    pairs = [[query, doc.page_content[:512]] for doc, _ in candidates[:30]]
    scores = self._reranker.predict(pairs)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return [(candidates[i][0], float(scores[i])) for i in order]
```

- `bge-reranker-base` 约 1.1GB，CPU 上 30 个候选约 200~400ms。可接受（你的 LLM 调用本身就要 0.5~3s）。
- **这正好复用你已有的 LLM Gateway 思路**：`grade_docs` 节点现在每轮要烧一次 LLM 调用做相关性判断（`langgraph_rag_agent.py:850`），rerank 分数上来之后可以用阈值直接过滤掉低分文档，**大幅减少 grade 的 LLM 调用次数**——这是可量化的降本。

> **实际落地偏差（2026-08-03）**：本小节设想的 `bge-reranker` CrossEncoder 重排序**未采用**。P3 落地后 rerank 仍是手写 MMR（`langgraph_rag_agent._mmr_rerank` / `advanced_rag_agent._rerank`），未引入本地 `sentence_transformers.CrossEncoder`，故不加载 torch、也无本小节担心的 GPU/内存与 segfault 成本。

### 4.5 补 BGE query instruction（零成本收益）

> **注意（2026-08-03）**：embedding 已改为 Ollama bge-m3（HTTP）。下方 `SentenceTransformerEmbeddings(... encode_kwargs={"normalize_embeddings": True})` 片段为**旧本地实现，已废弃**；Ollama bge-m3 的向量归一化由 Ollama 服务端处理，无需在 Python 侧设置。query instruction 前缀（BGE_QUERY_PREFIX）在 Ollama bge-m3 下同样适用，可保留。

BGE 系列官方要求检索时给 query 加指令前缀，当前代码完全没加：

```python
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索文章："

def _dense_search(self, query, filters, user_role):
    q = BGE_QUERY_PREFIX + query      # 只对 query 加，document 侧不加
    ...
```

同时把 embedding 初始化补上归一化：

```python
SentenceTransformerEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True}
)
```

> 归一化后余弦距离 = L2 距离的单调函数，为 P3 迁 Milvus 用 `COSINE` 度量做好准备。

### 4.6 收敛 embedding 实例（顺手修的性能问题）

现在一次 Web 启动加载 2~3 份 SentenceTransformer 权重（`rag_web_server.py:2579` + `langgraph_rag_agent.py:272` + 两个 `CacheManager`），且 `:2579` 拿到的 `vector_db` 在 LangGraph 路径下**从未被使用**。

```python
# embeddings.py（新文件）
_EMBED_SINGLETON = None
_EMBED_LOCK = threading.Lock()

def get_embeddings():
    global _EMBED_SINGLETON
    if _EMBED_SINGLETON is None:
        with _EMBED_LOCK:
            if _EMBED_SINGLETON is None:
                _EMBED_SINGLETON = SentenceTransformerEmbeddings(
                    model_name=EMBED_MODEL,
                    encode_kwargs={"normalize_embeddings": True})
    return _EMBED_SINGLETON
```

替换 `advanced_rag_agent.py:575`、`:421-422` 两处实例化点。预计**节省 1~2GB 内存 + 启动快 10~20 秒**。

> **现状（2026-08-03）**：本小节的性能问题（启动加载 2~3 份本地 SentenceTransformer 权重）已随 embedding 改 Ollama 而**自然消失**——主进程零 torch，无本地权重可收敛。`get_embeddings()` 单例思路仅在重新回归本地 embedding 时才有意义。

### 4.7 P1 验收标准

建一个 20 条问题的评测集（针对你的两份 PDF，含 10 条精确标识符类如「0x22 协议包结构」、10 条语义类如「设备如何上报位置」），对比改造前后：

| 指标 | 测法 | 期望 |
|---|---|---|
| Recall@5 | 人工标注每题的正确 chunk，看是否在前 5 | 精确标识符类应显著提升（BM25 生效） |
| grade LLM 调用次数 | 统计 `task="grade"` 的调用数 | 应下降（rerank 阈值过滤生效） |
| 端到端耗时 | `usage_log` 的 latency 汇总 | rerank 增加 ~300ms，但 grade 减少可抵消 |
| 内存占用 | 启动后 RSS | 应下降 1~2GB |

> **这个评测集要留下来**，P2/P3 每次改造都跑一遍做回归——否则「优化」是自我感觉良好。

---

## 五、P2 摄取管线：增量索引 + 多格式解析

> **目标**：把「重启全量重建」改成「增删改 upsert」。这是用户明确点名的基础项，也是后面 Milvus 海量文档的前提。
> **基建依赖**：无（还在 旧向量库 上做），但代码要按 Milvus 就绪形态写——`VectorStoreManager` 从静态类改成可持有连接、支持 `upsert`/`delete` 的实例类（P3 直接复用）。

### 5.1 现状痛点（第一章 1.2 已列，这里收敛成改造点）

| 痛点 | 代码位置 | 后果 |
|---|---|---|
| 重建判定只靠 `os.path.exists(DB_PATH)` | `advanced_rag_agent.py:577` | 目录存在就**永远不重建**，新增/修改文档进不去 |
| 目录扫描非递归、无排序 | `:605` `os.listdir(DOC_FOLDER)` | 子目录文档看不到、顺序不确定导致 chunk 错位 |
| 仅 `.pdf`/`.txt`，其余 `continue` | `:608-613` | docx/md/csv/pptx 全被静默丢弃 |
| 固定切分 600/120 一套通吃 | `:138-140, 628-634` | 表格 / 指令表类文档切坏 |
| 零 `db.add_documents()` / `db.delete()` | `:89` 现状事实 | 没有增量能力可言 |
| 无 chunk 稳定 ID | — | 改一个字就要整库重建 |

### 5.2 文件指纹 + 文档台账

建一张 MySQL 台账 `document_registry`，把「哪些文件被索引过、hash 是什么」落库，增量 diff 才有依据：

```sql
CREATE TABLE IF NOT EXISTS document_registry (
    id           BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id      VARCHAR(64)  NOT NULL DEFAULT 'system',
    file_path    VARCHAR(512) NOT NULL,
    file_name    VARCHAR(255) NOT NULL,
    content_hash CHAR(64)     NOT NULL COMMENT '原始文件 sha256',
    chunk_count  INT          NOT NULL DEFAULT 0,
    access_level VARCHAR(16)  NOT NULL DEFAULT 'public',
    status       ENUM('indexed','deleting','error') NOT NULL DEFAULT 'indexed',
    indexed_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_path (file_path)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

**chunk 稳定 ID** 是最关键的设计：把 `chunk_id = {sha256(file_path)[:12]}_{chunk_index}` 写进每条 metadata，并在向量库 upsert 时作为主键。这样「改一个字」只会产生新的 `content_hash` → 旧 chunk 按旧 `chunk_id` 删除、新 chunk 按新 `chunk_id` 插入，互不影响。

### 5.3 多格式 Parser（不再 `continue` 丢弃）

用 LangChain 社区 loader 覆盖主流格式，`_classify_loader` 按扩展名分发：

| 格式 | Loader | 备注 |
|---|---|---|
| `.pdf` | `PyMuPDFLoader`（fitz） | 替换 `PyPDFLoader`，保留版面/页码，比 `PyPDFLoader` 准 |
| `.txt` | `TextLoader(encoding="utf-8")` | 保留 |
| `.md` | `UnstructuredMarkdownLoader` | 保留标题层级 |
| `.docx` | `Docx2txtLoader` | 需 `docx2txt` |
| `.pptx` | `UnstructuredPowerPointLoader` | 需 `python-pptx` |
| `.csv` | `CSVLoader` | 按行建 chunk，自动带表头 |
| `.html` | `UnstructuredHTMLLoader` | 抽正文去标签 |

> 落地建议：装 `unstructured[md,docx,pptx,html]` 一个包解决大部分，CSV/TXT 用原生 loader。这样 `:608-613` 的 `if/elif/else: continue` 换成一张扩展名→loader 的映射表。

### 5.4 增量 upsert / delete（核心方法）

`VectorStoreManager` 从静态类改成实例类后，新增三个方法（旧向量库 侧用 `upsert`/`delete`，Milvus 侧用 `upsert`/`delete_entities`，接口一致）：

```python
class IngestPipeline:
    def __init__(self, vs: "VectorStoreManager"):
        self.vs = vs

    def sync(self, doc_folder: str = DOC_FOLDER) -> SyncReport:
        """扫描 → diff → upsert/delete，幂等可重复调用"""
        scanned = self._scan(doc_folder)                 # path -> (hash, mtime)
        indexed = self._load_registry()                  # 台账现状
        added   = [p for p in scanned if p not in indexed]
        changed = [p for p in scanned
                   if p in indexed and scanned[p].hash != indexed[p].hash]
        deleted = [p for p in indexed if p not in scanned]
        for p in added + changed:
            self._index_one(p)        # 解析→切分→打 chunk_id→upsert→更新台账
        for p in deleted:
            self._unindex_one(p)      # 按 file 前缀删 chunk_id→删台账
        return SyncReport(added, changed, deleted)
```

`os.listdir` 改成 `os.walk`（递归），`os.path.exists(DB_PATH)` 这个重建开关彻底删掉——首次启动 `sync()` 自然建库。

### 5.5 P2 验收标准

| 检查项 | 方法 | 期望 |
|---|---|---|
| 增量生效 | 往 `docs/` 丢一份新 PDF，调 `sync()`，**不重启** | 新文档可检索，旧文档 chunk 数不变 |
| 修改增量 | 改一份已索引 PDF 一个字再 `sync()` | 旧 chunk 被删、新 chunk 插入（`document_registry.content_hash` 更新） |
| 删除增量 | 删一份 PDF 再 `sync()` | 对应 chunk 从向量库消失，台账 `status='deleting'` |
| 多格式 | 放 `.docx`/`.md`/`.csv` 各一份 | 都能被索引并检索到 |
| 递归 | 在 `docs/sub/` 放文件 | 能被扫到并索引 |
| 幂等 | 连跑两次 `sync()` | 第二次 `added/changed/deleted` 全为 0 |

---

## 六、P3 Milvus 迁移

> **目标**：支撑**百万级**文档 + 并发 + **原生混合检索**（BM25 sparse + dense + 标量过滤下推）。用户已在虚拟机 `192.168.200.128` 准备部署 Milvus——本阶段是整套方案的终点，但代码是 P1/P2 顺下来的自然结果。
> **迁移策略**：旧向量库 与 Milvus **双写**，读走 `VECTOR_BACKEND` 开关，出问题切回 旧向量库，零风险。

### 6.1 为什么从 旧向量库 迁 Milvus

| 维度 | 旧向量库 嵌入式 | Milvus（standalone） |
|---|---|---|
| 规模 | 十万级（你当前 206 chunk） | 亿级向量 |
| 并发 | 单进程、无连接池 | gRPC 服务、连接池、多 client |
| 混合检索 | 需自己拼 BM25 + RRF | 2.5 起**原生支持** sparse vector + BM25 函数 |
| 标量过滤 | 仅 `filter=`（内存过滤） | `expr` **下推**到存储层，先过滤再算距 |
| 运维 | 文件目录 | etcd + MinIO + 多副本可扩展 |

你的协议/指令表文档未来要扩到大量设备型号、多版本，旧向量库 单机迟早触顶——Milvus 是用户已定的方向。

### 6.2 VectorStoreManager 改造：静态类 → 实例类

P1/P2 已经让 `VectorStoreManager` 持有 `get_embeddings()` 单例。`init_vector_store`（:564）的静态方法改成实例 `__init__`，后端由 `.env` 决定：

```python
# advanced_rag_agent.py —— 替换 :555 的静态类
class VectorStoreManager:
    def __init__(self, backend: str = None):
        self.backend = (backend or os.getenv("VECTOR_BACKEND", "milvus")).lower()
        self._embed = get_embeddings()                      # P1 的单例
        if self.backend == "milvus":
            from pymilvus import MilvusClient
            self.client = MilvusClient(uri=os.getenv("MILVUS_URI"))
            self.collection = os.getenv("MILVUS_COLLECTION", "rag_docs")
            self._ensure_collection()
        else:
            from pymilvus import MilvusClient
            self.client = MilvusClient(uri=os.getenv("MILVUS_URI", "http://192.168.200.128:19530"))

    def upsert(self, docs): ...          # Milvus: client.upsert(ids=...)
    def delete(self, chunk_ids): ...     # Milvus: client.delete
    def search(self, query, k, filters): ...   # 见 6.5
```

调用点统一替换：`:1050` 的 `VectorStoreManager.search(self.db, q, k)` → `self.vs.search(q, k, filters)`；`langgraph_rag_agent.py:272`、`rag_web_server.py:2579`、`:1867` 的 `init_vector_store()` → `VectorStoreManager()`。

### 6.3 Collection Schema（dense + sparse + 标量）

Milvus 2.5 支持 `SPARSE_FLOAT_VECTOR`，BM25 走内置 `BM25()` 函数自动出稀疏向量，无需自己维护 `rank_bm25` 词典：

```python
from pymilvus import FieldSchema, CollectionSchema, DataType

fields = [
    FieldSchema("chunk_id",     DataType.VARCHAR,      is_primary=True, max_length=64),
    FieldSchema("content",      DataType.VARCHAR,      max_length=4096),
    FieldSchema("dense",        DataType.FLOAT_VECTOR, dim=1024),  # Ollama bge-m3 1024 维（当前 .env 默认）
    FieldSchema("sparse",       DataType.SPARSE_FLOAT_VECTOR),     # BM25 函数算
    FieldSchema("file_path",    DataType.VARCHAR,      max_length=512),
    FieldSchema("file_name",    DataType.VARCHAR,      max_length=255),
    FieldSchema("access_level", DataType.VARCHAR,      max_length=16),
    FieldSchema("chunk_index",  DataType.INT64),
    FieldSchema("user_id",      DataType.VARCHAR,      max_length=64),
]
schema = CollectionSchema(fields, enable_dynamic_field=True)
# 建两个索引：dense 用 IVF_FLAT/COSINE；sparse 用 SPARSE_INVERTED_INDEX + BM25
```

> `dense` 度量用 `COSINE`——这正是 P1 4.5 把 embedding 做 `normalize_embeddings` 的伏笔，归一化后切度量零副作用。

### 6.4 双写开关 + 回滚

`.env` 新增：

```
VECTOR_BACKEND = "milvus"        # 切 milvus 即切换读路径
MILVUS_URI = "http://192.168.200.128:19530"
MILVUS_COLLECTION = "rag_docs"
```

迁移流程：
1. 部署 Milvus（第九章）→ 建 collection。
2. `sync()` 时**双写**：`self.vs.upsert` 内部按 `backend` 同时写 旧向量库 与 Milvus。
3. 灰度：把 1 个测试会话的 `VECTOR_BACKEND=milvus`，跑 P1 评测集对比分。
4. 全量：`VECTOR_BACKEND=milvus`，保留 旧向量库 一段时间作回滚兜底。
5. 确认稳定后，旧向量库 目录可归档（不删）。

### 6.5 expr 下推：让 `access_level` 真正生效

第一章 1.2 已证实 `access_level` 元数据**库里 0 行、且过滤是检索后 Python 丢弃**（`:733-762` 的 `filter_results`）。P2 的 `IngestPipeline._index_one` 要把 `access_level` 真正写进每条 chunk 的标量字段；P3 检索时**下推到 Milvus**，先过滤再算距，不再占 top-k 名额：

```python
def search(self, query, k=5, *, user_role="user", user_id="anonymous"):
    # 权限下推：admin 不过滤；user 只看 public（或自己上传的）
    if user_role == ROLE_ADMIN:
        expr = ""
    else:
        expr = '(access_level == "public") or (user_id == "' + user_id + '")'
    if self.backend == "milvus":
        return self.client.search(
            collection_name=self.collection,
            data=[self._embed.embed_query(BGE_QUERY_PREFIX + query)],
            anns_field="dense",
            limit=k,
            filter=expr,                       # ← 真正的下推
            # 混合检索：Milvus 2.5 用 ranker=RRFRanker() 融合 sparse+BM25
            ranker=RRFRanker(k=60),
            output_fields=["content", "file_name", "chunk_id"],
        )
    # 旧向量库 分支：仍走 _rrf_fuse（P1）+ 检索后 filter_results 兜底，保持兼容
```

`AccessControlFilter.filter_results`（:733）**保留**作为兜底，但正常路径不再依赖它——权限在查询层就卡掉了，普通用户不会再拿到「0 条结果」的尴尬。

### 6.6 长期记忆语义检索（跨会话）

P0 3.4 的 `chat_summaries` 表已经留了 `embedding BLOB` 列。P3 把它**填满**，再建一个 Milvus collection `rag_memory`，跨会话记忆也能向量检索：

```python
# MemoryManager.recall —— 把跨会话相关记忆注入当前上下文
def recall(self, query: str, user_id: str, k: int = 3) -> List[str]:
    q = self._embed.embed_query(BGE_QUERY_PREFIX + query)
    hits = self.memory_client.search(
        collection_name="rag_memory",
        data=[q], anns_field="dense", limit=k,
        filter=f'user_id == "{user_id}"',
        output_fields=["summary", "importance"],
    )
    # 按 importance 加权，P0 4 章打分终于用上了
    return sorted([h["summary"] for h in hits], key=lambda s: -s.get("importance", 3))
```

这样「用户三个月前问过 X」也能在当前会话被召回——这是 P0 偏差 1 里「跨会话上下文缺失」的正式解法。

### 6.7 P3 验收标准

| 检查项 | 方法 | 期望 |
|---|---|---|
| 双写一致 | 同一 `sync()` 后，旧向量库 与 Milvus 的 chunk 数相等 | 206 = 206 |
| 切后端无感 | `VECTOR_BACKEND=milvus` 跑 P1 评测集 | Recall@5 不低于 旧向量库（误差 < 2%） |
| 权限下推 | user 角色查 restricted 文档内容 | 结果里**不含** restricted 且数量正常（不再 0 条） |
| 混合检索 | 精确标识符类问题（如 `0x22`） | sparse 通道命中，top-1 命中率接近 100% |
| 长期记忆 | 用旧会话关键词在新会话提问 | `recall` 召回历史摘要并注入 |
| 回滚 | 改回 `VECTOR_BACKEND=milvus` | 系统立即回到旧路径，无数据丢失 |

### 6.8 P3 实际落地记录（2026-08-02 已实施）

> 目标达成：**应用代码已完成 Milvus 接入，旧向量库 不再是默认向量库**。`advanced_rag_agent.py` 的 `VectorStoreManager` 由静态类改为统一实例类，按 `VECTOR_BACKEND` 切换后端（**默认 `milvus`**），Milvus 不可用时自动回退 旧向量库 并打印醒目警告。

**新增 / 修改（`advanced_rag_agent.py`）**

| 项 | 内容 | 为什么 |
|---|---|---|
| `VectorStoreManager.__init__` | 读 `VECTOR_BACKEND`（默认 milvus）/ `MILVUS_URI` / `MILVUS_COLLECTION`；连 `MilvusClient`；`_ensure_collection()` 建集合+索引 | 让代码真正走 Milvus，而非只部署服务端 |
| `_ensure_collection` | 按 `chunk_id`(主键) / `content` / `dense`(FLOAT_VECTOR，**维度动态探测**，避免硬编码) / `file_path` / `file_name` / `access_level` / `chunk_index` / `user_id` 建 schema；`dense` 用 `AUTOINDEX`+`COSINE` 索引 | 与 旧向量库 字段对齐，权限字段可下推 |
| `_build_milvus` | 复用既有文档加载/分片逻辑，批量 `upsert`（按内容哈希做主键，**幂等可重复跑**） | 替代 旧向量库 的 `from_documents` 构建 |
| `_milvus_search` | `data=[embed_query(q)]` 走 `dense` 字段；`filter=expr` 做 `access_level` **权限下推**（admin 不过滤，user 看 `public` 或自己的） | 真正的下推过滤，不再只靠检索后丢弃 |
| `similarity_search_with_score` | 统一对外接口，返回 `[(Document, distance), ...]`（与 旧向量库 完全一致），上层零改动 | 保持上层 Agent 接口稳定 |
| `init_vector_store()` / `search()` | 改为返回实例 / 转发到实例方法 | 兼容 3 处旧调用点（`advanced_rag_agent:1867`、`langgraph:272`、`rag_web_server:2589`） |

**调用点同步**
- `advanced_rag_agent.py:1050`：`VectorStoreManager.search(self.db, q, k)` → `self.db.similarity_search_with_score(q, k=top_k, filter_role=self.user_role)`（透传角色，触发 Milvus 下推）
- `langgraph_rag_agent.py:1298`：`self.vector_db.similarity_search_with_score(q, k=RETRIEVE_TOP_K)` 追加 `filter_role=role`

**PIP 字段说明（与方案 6.3 的偏差）**
- ~~未包含 sparse/BM25~~ **已补齐**：`VectorStoreManager` 已在 schema 中加入 `sparse`(SPARSE_FLOAT_VECTOR, `is_function_output=True`) 字段，并用 Milvus 原生 `Function`(`FunctionType.BM25`) 从 `content`（开启中文 analyzer）自动生成稀疏向量；`_milvus_search` 改为 **dense + sparse 双路召回 + RRF 融合**（`HYBRID_SEARCH` 可关闭）。即方案 6.3/6.5 的「混合检索」**已落地**，不再是 P1 待办。
- 未做 6.4 的「旧向量库/Milvus 双写」：当前为**单后端切换**（默认 Milvus，失败回退 旧向量库），而非双写。理由：双写要求两套索引实时同步，复杂度高且当前无灰度需求；回退机制已通过 `VECTOR_BACKEND` 开关覆盖「零风险回滚」。
- 6.6 的「跨会话长期记忆 `rag_memory` collection」未做，依赖 P0 `chat_summaries` 的 `embedding` 列填充，属后续增强。

**验证（沙箱已直连 192.168.200.128 实测通过）**
> 纠正前版错误：沙箱（WorkBuddy 运行时运行于 Windows 宿主机，与 VM 同处 VMware NAT 局域网）**完全可以直连** `192.168.200.128` 的 Milvus(19530)/MySQL(3306)/Redis(6379)/Ollama(11434)。已用 TCP 探测确认全部 REACHABLE，并用真实 `VectorStoreManager` 端到端跑通：连接 → 创建集合(含 BM25 函数) → 构建索引 → 混合检索(RRF)返回结果。

1. `.env` 加 `VECTOR_BACKEND=milvus`、`MILVUS_URI=http://192.168.200.128:19530`、`MILVUS_COLLECTION=rag_docs`、`HYBRID_SEARCH=true`；`pip install "pymilvus~=2.5.0"`（**勿用 3.x**：3.0 改名 `FunctionSchema`→`Function` 且与 2.5 服务端存在兼容风险）；
2. 启动 `python rag_web_server.py`，观察日志：`[VectorStore] 连接 Milvus ...` → `已创建 Milvus 集合 rag_docs (dim=1024, hybrid=on)` → `Milvus 索引构建完成，共 N 条`；
3. 提问验证检索正常、普通用户看不到 restricted 文档（权限下推生效），且混合检索能召回 BM25 相关片段；
4. 回滚：`VECTOR_BACKEND=milvus` 或 `HYBRID_SEARCH=false` 重启即可，无数据丢失。

**状态**：P3 核心（Milvus 接入 + 权限下推 + 回退）+ **P1 混合检索（sparse BM25 + dense + RRF）** 标记 **已完成**；跨会话长期记忆 留待后续增强。

---

## 七、文档管理 Web API

> 当前 20 条路由（第一章 1.2 表已确认）**零条**涉及上传/删除/重建。P2 的 `IngestPipeline` 已经具备能力，这里补 HTTP 入口，让运营能自助管理知识库。

在 `rag_web_server.py` 新增（需鉴权 + 角色校验，admin 才能删）：

| 路由 | 方法 | 行为 |
|---|---|---|
| `/api/docs` | GET | 列 `document_registry`：文件名 / hash / chunk 数 / access_level / 状态 / 索引时间 |
| `/api/docs/stats` | GET | 总文档数、总 chunk 数、上次 `sync` 时间、各 access_level 分布 |
| `/api/docs/upload` | POST | `multipart` 接收文件 → 落 `docs/` → 调 `IngestPipeline._index_one` → 返回新 chunk 数 |
| `/api/docs/{file_id}` | DELETE | 删磁盘文件 + `vs.delete(前缀)` + 台账 `status='deleting'`（admin only） |
| `/api/docs/rebuild` | POST | 全量 `sync()`，返回 `SyncReport`（added/changed/deleted） |

关键点：上传/删除都走 `IngestPipeline` 的同一套 `chunk_id` 逻辑，**不**直接调向量库，保证台账与向量库一致。删除接口强制 `user_role == ROLE_ADMIN`（复用 `:166` 的 `ROLE_ADMIN`），否则返回 403。

---

## 八、依赖固化（requirements.txt）

> 第一章 1.2 确认项目**只有 README 散文、无 requirements.txt**，2~3 份 embedding 权重 + 缺 `rank_bm25` 等包说明依赖是「跑起来才装」的状态。P1~P3 引入多个新包，必须固化。

**新增依赖**（在现有 langchain 栈基础上）：

```
# 检索增强（P1）
rank_bm25>=0.2.2
jieba>=0.42.1
# sentence-transformers：当前 rerank 为手写 MMR（非 CrossEncoder）、embedding 已改 Ollama，故不再必需；仅在未来引入本地 reranker 时再加

# 摄取管线（P2）
pymupdf>=1.24                       # 替换 PyPDFLoader
unstructured[md,docx,pptx,html]>=0.14
python-pptx>=0.6
python-docx>=1.1

# 向量库（P3）
pymilvus>=2.5                       # 原生 BM25 sparse + 混合检索

# 复用（P1）
# embeddings.py 单例已在代码内，无需新包
```

落地：导出当前 `pythonspace` 环境 `pip freeze > requirements.txt`，**再**人工裁剪掉无关包（避免把整个 conda 环境锁死），然后提交。CI/部署时用这份文件重建环境，杜绝「我机器上能跑」。

---

## 九、虚拟机 Milvus 部署要求与验收标准

> 虚拟机 `192.168.200.128`，Rocky Linux 10，已跑 Ollama / MySQL 8 / Redis。Milvus standalone 与这些服务**端口不冲突**（Milvus 用 19530/9091），但需评估内存余量。

### 9.1 部署方式（推荐 Docker Compose standalone）

Milvus standalone = `etcd` + `MinIO` + `milvus` 三个容器，官方 compose 一把起。比二进制省心，也比分布式（3×Milvus）省资源。

```yaml
# docker-compose-milvus.yaml（实际存放于虚拟机 /data/milvus/，挂载目录均在下）
services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.14
    container_name: milvus-etcd
    restart: always
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - /data/milvus/etcd:/etcd
    command: >-
      etcd -advertise-client-urls=http://etcd:2379
      -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 5s
      timeout: 8s
      retries: 10
    networks:
      - milvus-network

  minio:
    image: minio/minio:RELEASE.2024-05-28
    container_name: milvus-minio
    restart: always
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - /data/milvus/minio:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 5s
      timeout: 8s
      retries: 10
    networks:
      - milvus-network

  milvus:
    image: milvusdb/milvus:v2.5.0
    container_name: milvus-standalone
    restart: always
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - /data/milvus/data:/var/lib/milvus
    ports:
      - "19530:19530"   # gRPC（Python 客户端连这个）
      - "9091:9091"     # HTTP / healthz
    depends_on:
      - etcd
      - minio
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      timeout: 10s
      retries: 10
    networks:
      - milvus-network

networks:
  milvus-network:
    driver: bridge
```

> **与你现有 MySQL 部署对齐**：参照你的 `docker-compose-mysql.yaml` 风格——显式声明 `networks`（桥接 `milvus-network`）、`restart: always`、`healthcheck`，并用**绑定挂载**（`/data/milvus/{etcd,minio,data}`）而非命名卷，和你的 `./mysql-data:/var/lib/mysql` 一致，方便直接进目录排查。
>
> **实际部署目录**：compose 文件落在虚拟机 `/data/milvus/docker-compose-milvus.yaml`，所有挂载子目录（`/data/milvus/etcd`、`/data/milvus/minio`、`/data/milvus/data`）都在该目录下。已生成独立文件 `deploy/docker-compose-milvus.yaml`，可直接 `scp` 到虚拟机 `/data/milvus/` 后 `docker compose up -d`。
>
> **网络边界说明**：MySQL 在 `nacos-network`、Milvus 在 `milvus-network`，二者**不需要互通**——Milvus 不依赖 MySQL（元数据走 etcd、向量文件走 MinIO）。你的 Python 应用跑在宿主机，通过已发布的 `19530`（Milvus）/ `3306`（MySQL）端口分别连接，互不干扰。若未来应用也容器化且想用容器 DNS 互访，把应用容器同时挂到两个网络即可。
>
> **镜像方案（2026-08-02 实测结论）**：`docker.m.daocloud.io` 国内镜像源已失效（VM 拉取返回 `403 Forbidden`），**不可再用**。最终采用 **「宿主机 Clash 代理拉官方源」**：compose 用官方镜像（`quay.io/coreos/etcd`、`minio/minio`、`milvusdb/milvus`），虚拟机本身无外网，靠宿主机 Clash（Mixed 端口 7897）代理拉取；配套 `deploy/setup-docker-proxy.sh`（写入 `/etc/systemd/system/docker.service.d/http-proxy.conf` 并重启 docker），还原用 `deploy/reset-docker-proxy.sh`。关键前提：宿主机 Clash Party「监听地址」必须改 `0.0.0.0`（默认 127.0.0.1 会导致 VM `connection refused`）。其它公共镜像站（docker.1ms.run / dockerproxy.com 等）可用性波动、未实测，仅作尝试项。离线兜底：能联网机器 `docker pull`+`docker save`+`scp`+VM `docker load`。

### 9.2 资源预估（针对你的规模）

| 资源 | 最低 | 推荐 |
|---|---|---|
| 内存 | 4 GB（Milvus standalone 常驻 ~2GB） | 8 GB（给 Ollama+qwen2:7b 留余量） |
| CPU | 2 核 | 4 核 |
| 磁盘 | 10 GB 空闲（MinIO 存 segment） | 20 GB |
| 系统 | Rocky Linux 10，已装 docker + docker-compose | — |

> 若虚拟机内存紧张（Ollama 7B 已占 ~4~6GB），考虑把 Ollama 的上下文或并发调小，或给 Milvus 设 `queryNode` 内存上限。验收前先 `free -h` 确认余量 > 4GB。

### 9.3 `.env` 新增配置

```
# --- Milvus 向量库 ---
VECTOR_BACKEND = "milvus"            # 部署并验证后改 milvus
MILVUS_URI = "http://192.168.200.128:19530"
MILVUS_COLLECTION = "rag_docs"
```

`MilvusClient(uri=...)` 走 gRPC；若虚拟机有防火墙，需放通 19530（内网即可，不必公网暴露）。

### 9.4 验收标准（部署完在开发机跑）

```python
# 验收脚本（开发机执行，连 192.168.200.128:19530）
from pymilvus import MilvusClient
c = MilvusClient(uri="http://192.168.200.128:19530")
assert c.list_collections() is not None          # 1. 连通
c.create_collection("rag_docs", dimension=1024)    # 2. 建库（schema 见 6.3，bge-m3 维度）
c.insert("rag_docs", [{"chunk_id":"t_0","dense":[...],"content":"测试"}])  # 3. 写
hits = c.search("rag_docs", data=[[...]], anns_field="dense", limit=1)    # 4. 查
assert len(hits) >= 1                            # 5. 命中
# 6. 混合检索：确认 milvus>=2.5 支持 BM25() + RRFRanker
```

| 检查项 | 期望 |
|---|---|
| 连通 | 开发机 `MilvusClient` 能连上 19530 |
| 建 collection | `rag_docs` 出现在 `list_collections()` |
| 写入 | `insert` 返回成功，计数 +1 |
| 检索 | `search` 返回 top-k，`dense` 余弦可用 |
| 混合 | `RRFRanker` + sparse 字段可被 API 接受（2.5 特性） |
| 权限 | 带 `filter='access_level=="public"'` 的查询能正确过滤 |
| 资源 | `docker stats` 显示 Milvus 常驻内存 < 可用余量 |

### 9.5 上线顺序（收口）

1. **P0 止血** → 数据变干净（无基建，当天可上）。
2. **P1 检索质量** → 召回率 + 降本（加 2 个 pip 包，1~2 天）。
3. **P2 摄取管线** → 增量 + 多格式（代码改造，2~3 天）。
4. **第九章部署 Milvus** → 虚拟机起服务（你操作，半天）。
5. **P3 切换** → 双写 → 灰度 → 全量（1 天，可回滚）。

每一阶段都有独立验收表，任一层出问题**不影响已上线的下层**——这是第二章「每阶段独立可上线」原则的工程落地。

---

## 十、风险与回滚总览

| 阶段 | 主要风险 | 回滚手段 |
|---|---|---|
| P0 | `user_id` 迁移锁表（chat_messages 大时） | 低峰期执行；DEFAULT 'anonymous' 无需回填，可瞬间回退代码 |
| P1 | bge-reranker 1.1GB 加载慢 | `use_rerank=False` 开关即时关闭，退化回 RRF |
| P2 | 多格式 loader 解析异常 | 单文件失败不影响整体 `sync()`；`status='error'` 记录 |
| P3 | Milvus 不可用 | `VECTOR_BACKEND=milvus` 一键回退，旧向量库 数据保留 |
| 部署 | 虚拟机内存不足 | 调小 Ollama 并发 / 给 Milvus 设内存上限；或临时关 MinIO 压缩 |

---

## 十一、运行环境注意事项（py310 / Windows 实战坑）

> 2026-08-02 在 `D:/prom/anaconda/envs/py310`（Python 3.10.20）真实踩坑记录，重启/部署服务前必读。

### 11.1 依赖安装（已验证可用）
```bash
# 激活环境后安装向量库客户端
D:/prom/anaconda/envs/py310/python.exe -m pip install "pymilvus~=2.5.0"
# langchain-community / pypdf 等该环境已预装；**sentence-transformers / torch 不再需要**（embedding 走 Ollama HTTP，主进程不加载 torch）
```
- **pymilvus 必须锁 2.5.x**：Milvus 服务端 v2.5.0，3.x 把 `FunctionSchema` 改名 `Function` 且 API 不兼容；实测 **2.5.18** 可用（BM25 用 `Function`）。

### 11.2 Embedding 已改 Ollama HTTP，原「本地 BGE 加载 segfault」不再适用
- **背景**：早期 embedding 用本地 `SentenceTransformerEmbeddings`（BGE-small-zh，主进程加载 torch），在 Windows + torch 2.13 + MKL/OpenMP 多线程下构造时曾触发 `Segmentation fault (139)`。
- **现状（2026-08-03 起）**：embedding 后端**已彻底改为 Ollama-only**——`advanced_rag_agent._make_embedder` 仅返回 `OllamaEmbeddings(bge-m3)`，走 HTTP 调虚拟机 Ollama。**主 web 进程不再加载 torch / sentence-transformers**，原 segfault 从设计上被消除；原先配套的 `OMP_NUM_THREADS/MKL/OPENBLAS/...` 线程保险丝（setdefault 限 1）已随 local 分支一并删除，**无需也不应再设置**这些环境变量。
- **结论**：启动服务不再有「本地 embedding 加载崩溃」风险；`docs/` 2 份 PDF 经 Ollama bge-m3（1024 维）向量化写入 Milvus。若未来重新引入本地 torch 模型（如本地 reranker），再评估线程保险丝，但当前默认部署不涉及。
- **历史排查弯路（已证伪，勿再试）**：①强降 `numpy→1.26` 反而让 `import torch` 直接崩（torch 2.13 针对 numpy 2 ABI 编译）；②降 `sentence-transformers` 版本无效——真正元凶是本地线程库，而现架构已不加载它。

### 11.3 启动验证（预期日志，真实复现过）
```
[VectorStore] 连接 Milvus: http://192.168.200.128:19530 (collection=rag_docs)
[VectorStore] 已创建 Milvus 集合 rag_docs (dim=1024, hybrid=on)
[VectorStore] Milvus 索引构建完成，共 206 条
```
`docs/` 共 2 份 PDF 真实分片 **206 条**；混合检索（dense BM25 + RRF）admin/user 均正常、权限下推生效。
