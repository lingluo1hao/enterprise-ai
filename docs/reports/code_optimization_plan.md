# 代码优化方案（按优先级）— enterprise-ai

> 范围：基于全模块深读 + 13 个已定位到 `file:line` 的问题，输出**可落地的具体修复方案**。
> 约定：本文件只列方案，**未改动任何代码**，待评审后实施。
> 验证环境说明：所有 `file:line` 均来自当前工作区真实代码（非推测）。

---

## 0. 总览与实施顺序

| 编号 | 优先级 | 类别 | 一句话风险 | 修复体量 |
|---|---|---|---|---|
| P0-1 | P0 | 数据丢失 | Milvus 瞬时报错 → 误删整个 `rag_docs` 索引（锁未覆盖 check+drop+create） | 中 |
| P0-2 | P0 | 越权 | **仅 legacy 模式**（`--no-langgraph`）编排器按请求改实例字段+`cache.current_role` → 并发串味；**默认 LangGraph 入口已由 ContextVar 修掉** | 中（只改 legacy 路径） |
| P0-3 | P0 | 认证旁路 | MySQL 抖动 → 硬编码 `admin/admin123` 直接登录 | 小 |
| P0-4 | P0 | 路径穿越 | 租户名未校验 → `../` 穿越写文件 | 小 |
| P0-5 | P0 | fail-open | ACL 规则加载失败 → 受限文档全变 public | 小 |
| P1-6 | P1 | 闭环真实性 | complex/chitchat 被当"检索失败"灌进 Bad Case | 中 |
| P1-7 | P1 | 闭环真实性 | 三级信号 L2/L3 线上是死的（README 立论需修正） | 大 |
| P1-8 | P1 | 数据虚增 | 一次请求 `success_count` 双计，自进化数字翻倍 | 小 |
| P1-9 | P1 | 经验丢失 | `patch_success` delete+insert 非原子，崩溃/并发丢经验 | 中 |
| P1-10 | P1 | 经验失效 | playbook 不随 KB 重 ingest 失效，复用过期经验 | 中 |
| P1-11 | P1 | 凭据外置 | 硬编码 IP/密码散落 ~14 处 | 大（收敛） |
| P1-12 | P1 | 凭据外置 | `init_db.sql` 明文 6 套默认密码入库 | 小 |
| P1-13 | P1 | 权限一致 | 访问控制三套实现互相打架 | 大（统一） |

**建议实施顺序**：先 P0（1→5，都是小/中改动、止血安全），再 P1 中"闭环真实性"组（6→10，因为直接影响方案/稿子立论），最后凭据与配置组（11→13，纯重构、可独立发版）。

---

## P0 · 立即修（越权 / 数据丢失 / 认证旁路）

### P0-1 Milvus 瞬时错误触发全量重建 → 可能丢整个索引
**位置**：`advanced_rag_agent.py:627-645`
```python
if self.client.has_collection(self.collection):
    try:
        desc = self.client.describe_collection(self.collection)
        ...
        if "sparse" in existing and existing_dim == dim and "tenant_id" in existing:
            return
        reason = (...)
        print(f"[VectorStore] 检测到集合需重建({reason}...)")
    except Exception:
        pass                       # ← 问题：describe 抛异常被吞掉
    self.client.drop_collection(self.collection)   # ← 无条件 drop
```
**根因**：`describe_collection` 抛异常（网络抖动/超时）走 `except: pass`，落入下面的 `drop_collection`。`has_collection` 为真时，任何瞬时错误都会把整个集合删了重建。gunicorn 4 worker 并发 init 时更会抢删。

**修复方案（两条铁律：瞬时错误要重试→fail-fast，锁必须覆盖整段）**：

1. **瞬时错误：重试 3 次带指数退避 → 仍失败就 fail-fast 抛错启动失败，绝不静默 `return`/静默 `continue`**。
   原方案"describe 失败就跳过重建、留待下次启动"是错的：那样进程会带着 schema 不匹配的老集合继续跑（搜索报错/结果错），等于用坏集合服务。正确语义是——瞬时错误只是"这一下没查到"，重试后大概率恢复；重试耗尽说明 Milvus 真病了，应当**启动失败**让人来修，而不是偷偷跑一个会出错的集合。

```python
def _describe_with_retry(self, collection, retries=3, backoff=1.0):
    last_e = None
    for i in range(retries):
        try:
            return self.client.describe_collection(collection)
        except Exception as e:
            last_e = e
            if i < retries - 1:
                time.sleep(backoff * (2 ** i))   # 1s, 2s, 4s
    raise RuntimeError(f"describe_collection 重试 {retries} 次仍失败: {last_e}")

def ensure_collection(self):
    with self._init_lock():          # ★ 分布式锁覆盖 check+drop+create 整段（见下）
        if not self.client.has_collection(self.collection):
            self._create_collection()
            return
        try:
            desc = self._describe_with_retry(self.collection)   # 瞬时错误 → 重试 → 耗尽抛 RuntimeError
            fields = desc.get("fields", [])
            existing = [f.get("name") for f in fields]
            dense = next((f for f in fields if f.get("name") == "dense"), None)
            existing_dim = dense.get("params", {}).get("dim") if dense else None
            if "sparse" in existing and existing_dim == dim and "tenant_id" in existing:
                return               # ★ schema 满足，绝不重建
            reason = (...)
            print(f"[VectorStore] 集合需重建({reason})")
        except RuntimeError as e:
            # ★ 瞬时错误耗尽重试：fail-fast，启动失败，绝不静默 continue
            print(f"[VectorStore] ⚠ describe 瞬时错误耗尽重试，中止启动(fail-fast): {e}")
            raise
        self.client.drop_collection(self.collection)   # 到这里只可能是"确认 schema 不满足"
        self._create_collection()
```

2. **分布式锁必须覆盖 `check + drop + create` 整段**，否则"1 号 worker 独占"方案下，其他 worker 在重建完成前 `has_collection=True` 仍会触发自己的 drop。
   - 锁用 Redis `SET NX` + TTL（key=`rag:init_lock:{collection}`），获锁 worker 跑整段 `ensure_collection` 逻辑，其余 worker **阻塞等待**而非"各自判断"；锁释放后其余 worker 再获锁时 `has_collection=True` 且 `describe` 成功、schema 已满足 → 直接 `return`，不会重复 drop。
   - 若不想用 Redis 锁，也必须在 `post_worker_init` 里做"仅 1 号 worker 做 ensure，其余 worker `sleep` 轮询直到集合就绪再退出 init"，**不能**让未就绪的 worker 各自进入 `ensure_collection` 的 drop 分支。

**验证**：单测注入 `describe_collection` 前两次抛错、第三次成功 → 断言**不 drop**、正常返回；注入每次都抛错 → 断言 `ensure_collection` 抛 `RuntimeError`（启动失败）。并发 4 进程同启，断言只重建 1 次（锁生效）。

---

### P0-2 越权根因修正：默认 LangGraph 入口已安全，**只重构 legacy 路径**
**前提修正（重要）**：原方案把"共享编排器改实例字段"当作越权根因，这是不准确的。

**默认入口（`LangGraphRAGApp`）早已用 `contextvars.ContextVar` 修掉了同类问题**：
`langgraph_rag_agent.py:505-599` 专门把 `user / username / tenant_id / current_task_id` 做成 `ContextVar` property，注释直写"P0 级安全问题，SSE 多线程串号"。每个请求线程拿到一份独立 `Context`，天然隔离——`self.user = x` 实际写进当前线程的 `ContextVar`，不会污染别的线程。**所以默认路径（Web 走 `LangGraphRAGApp`）不存在串味问题，无需改动。**

**真正有坑的只有 legacy `RAGOrchestrator`（`--no-langgraph` 模式）**：
`advanced_rag_agent.py:1955-1974`（`gunicorn_config.py:12-14` 的 `workers=4, worker_class=gthread, threads=8`）
```python
# query() 内直接改写实例字段（每个 gthread worker 只有 1 个 orchestrator 实例）
if user:
    self.user = user
    self.planning_agent.user = user
    ...
if user_role and user_role != self.user_role:
    self.user_role = user_role
    self.cache.current_role = user_role          # ★ 连 cache.current_role 也按请求改写，同一家族
    doc_skill.user_role = user_role
```
**根因（仅限 legacy）**：8 线程共用 1 个 `RAGOrchestrator` 实例，`query()` 把请求级的 `user/user_role` 写进**实例字段**，且**连 `cache.current_role` 也按请求改写**。admin 与 user 请求并发时，后到请求会覆盖先到请求的 `user_role`/`cache.current_role`，导致普通用户短暂拿到 admin 角色、读到 restricted 文档。

**修复方案（收紧改动面，只动 legacy 路径 + 收口 `cache.current_role`，**不要**铺到 LangGraph）**：
- **（推荐）把 `--no-langgraph` 的 legacy 编排器也改成 ContextVar 隔离**，与 LangGraph 同构：把 `user / user_role / cache.current_role / doc_skill.user_role` 的请求级状态全部走 `contextvars`，与 `langgraph_rag_agent.py:526-599` 的写法对齐。调用点语法不变（`self.user = x` 照旧），只是底层路由到 `ContextVar`——回归风险最低，且复用已验证过的隔离模式。
- 若不想大改 legacy，至少把 `cache.current_role` 的"按请求改写"去掉：legacy 路径的 `cache.current_role` 改为从登录态 `g.current_user`（见 `rag_web_server.py:365` "role/user 一律来自登录态，客户端不可伪造"）读取，不再在 `query()` 里赋值。
- **明确排除**：默认 LangGraph 入口（`LangGraphRAGApp`）**不在本次改动范围**，验证脚本也不要跑它，避免扩大改动面引入回归。

**验证（必须显式跑 legacy 模式）**：用 `--no-langgraph` 启动，admin token 与 user token 交替高频并发请求，断言 user 请求**永不可**在返回里出现 restricted 文档内容，且 `cache.current_role` 不被另一请求的 `user_role` 污染。验证脚本启动参数写明 `--no-langgraph`，别误测默认 LangGraph 路径。

---

### P0-3 MySQL 挂了仍可用硬编码 admin/admin123 登录（认证后门）
**位置**：`prompt_manager.py:855-864`（`login`）+ `:936-939`（`change_password`）
```python
def login(self, username, password):
    if not self.available:
        # 降级：本地硬编码认证（仅 admin）
        if username == "admin" and password == "admin123":
            ... return admin session
        return None
# change_password 同理：if not available: if admin/admin123: return True
```
**根因**：DB 不可达（`self.available=False`）时 fallback 到写死口令，等于把认证边界在故障期完全敞开。

**修复方案**：fail-closed，**拒绝登录**，绝不接受写死口令：
```python
def login(self, username, password):
    if not self.available:
        # DB 不可达：认证边界不能降级为写死口令 —— 拒绝登录（fail-closed）
        print("[AuthManager] ⚠ MySQL 不可达，拒绝登录（认证后门已关闭）")
        return None
```
`change_password` 同理改为 `return False`。
- 若确有"无 DB 本地应急 admin"需求，必须显式开关 + 随机口令：env `AUTH_LOCAL_FALLBACK=1` 且口令从 `env AUTH_LOCAL_PASSWORD`（随机生成、部署时注入）读取，**严禁源码写死**。

**验证**：单测 mock `available=False`，断言 `login("admin","admin123")` 返回 `None`；集成测试断掉 MySQL，登录接口返回 401。

---

### P0-4 租户目录名未校验 → 路径穿越写入
**位置**：`rag_web_server.py:1641`（`tenant` 来自表单）+ `:1658-1663`
```python
tenant = (request.form.get("tenant", "") or "").strip() or my_tenant   # super_admin 可控
...
dest_dir = os.path.join(DOC_FOLDER, tenant)
os.makedirs(dest_dir, exist_ok=True)
f.save(os.path.join(dest_dir, filename))
```
**根因**：`tenant` 仅做非空判断，未做白名单校验。`super_admin` 可传 `tenant = "../../etc"` 实现目录穿越写文件。

**修复方案**：白名单正则 + 路径回锚双重校验：
```python
import re
TENANT_RE = re.compile(r"^[a-z0-9_-]{1,64}$")
...
raw_tenant = (request.form.get("tenant", "") or "").strip()
if role == ROLE_SUPER_ADMIN and raw_tenant:
    tenant = raw_tenant
else:
    tenant = my_tenant
if not TENANT_RE.match(tenant):
    return jsonify({"error": "非法租户名"}), 400
dest_dir = os.path.abspath(os.path.join(DOC_FOLDER, tenant))
doc_root = os.path.abspath(DOC_FOLDER)
if os.path.commonpath([doc_root, dest_dir]) != doc_root:   # 二次回锚，防穿越
    return jsonify({"error": "非法路径"}), 400
os.makedirs(dest_dir, exist_ok=True)
```
文件名侧已有 `_kb_safe_name()`，保持。建议把 `TENANT_RE` 也复用到 `rag_web_server.py:1514/1516` 的 Milvus `tenant` 表达式拼接（防注入）。

**验证**：传 `tenant=../../etc`，断言返回 400 且文件未写出 `DOC_FOLDER` 之外。

---

### P0-5 ACL 解析失败静默"全公开"（fail-open）
**位置**：`ingest/loaders.py:29-39`
```python
DOC_ACCESS_RULES: dict = {}
try:
    ... yaml.safe_load(access_rules.yaml) ...
except Exception:
    DOC_ACCESS_RULES = {}   # ← 加载失败 → 受限文档全部变 public
```
**根因**：yaml 缺失/格式错/不可读时，`DOC_ACCESS_RULES={}`，`get_access_level()` 一律返回 `public`。受限文档（如 `JM-S509` 客户指令表）在故障时全部暴露。

**修复方案**：fail-closed + 启动告警。
- 加载失败或文件缺失时，默认级别改为 **`restricted`**（宁可误伤公开文档，不可泄露受限文档）；并提供显式开关 `ACCESS_RULES_FAIL_OPEN=1` 供明确选择宽松模式（默认关闭）。
- 启动时（模块 import / init_system）打印 **ERROR 级**日志明确告知 ACL 未生效、当前为受限默认。
- 进一步：把 `access_rules.yaml` 的加载集中到统一 `AccessControl` 模块（见 P1-13），单点加载、单点失败处理。

**验证**：删掉 `access_rules.yaml`， ingest 一个文件名命中受限规则的文档，断言其 `access_level` 入库为 `restricted` 且启动日志出现 ERROR。

---

## P1 · 自进化闭环真实性（影响方案和稿子的立论）

### P1-6 complex / chitchat 被当作"检索失败"灌进 Bad Case
**位置**：`langgraph_rag_agent.py:889`（`direct_llm→respond→save_history`）+ `:1189-1207`（`save_history` 内 `evaluate_success`→空 `doc_grades` 即写 Bad Case）+ `:2093`（complex 的 grade 只写子任务局部变量，从不回写顶层 `state["doc_grades"]`）；顶层 `doc_grades` 在 `:2614/2739/2749` 初始化为 `[]`。
```python
# node_save_history 公共出口
ok, _level = ext.evaluate_success(state)   # state["doc_grades"]==[] → ok=False
if ok: store.save_or_merge(...)
else:
    fail = ext.extract_failure(state, ...)  # relevant=0 < 阈值 → is_retrieval_fail=True
    ms.add_bad_case(...)                    # ← 每个 complex / 每句闲聊都成假 bad case
```
**根因**：顶层 `state["doc_grades"]` 对 complex（检索在子 agent 内完成）和 chitchat（本就无检索）恒为空 → `evaluate_success` 判"检索未命中" → 假 Bad Case 灌库，污染自进化闭环与评测数字。

**修复方案**：
1. **chitchat 直接跳过自进化闭环**（无检索，谈不上检索失败）：
```python
qtype = state.get("query_type")
if qtype == "chitchat":
    pass  # 闲聊无检索，不写 playbook、不写 bad case
elif qtype == "complex":
    # complex 的检索发生在子 agent，需把子任务 grades 聚合回顶层再判
    ok, _level = ext.evaluate_success(state, grades=aggregated_subtask_grades)
else:
    ok, _level = ext.evaluate_success(state)
```
2. **complex 聚合子任务 grades**：在 complex 分支收口处，把各子任务 `grade_docs` 结果汇总（`all_sub_grades`）写入顶层 `state["doc_grades"]`（或新增 `evaluate_success(state, grades=...)` 入参，避免动全局 state）。**关键**：聚合后 `state["doc_grades"]` 反映的是 complex 各子任务真实的检索成败——不再恒为空。
3. **`extract_failure` 只修"误判"，不一刀切禁止 complex 落库**：
   - 原误判根因是"顶层 `doc_grades` 为空 → 判检索失败"。修复后 complex 已聚合子任务 grades，**非空**，所以真正检索失败的 complex 子任务应当照常落 Bad Case。
   - 守卫只加在"确实无任何检索动作"的情形（chitchat），以及"顶层 `doc_grades` 为空且无法聚合"的兜底——**不要**写 `query_type in ("simple",)` 这类把 complex 整类排除的规则。complex 子任务真实检索失败（如某子任务 `relevant=0`）仍应进库，只是不能因"顶层为空"被错杀。

**验证**：跑 N 条 chitchat → 断言 `bad_cases` 新增 0；跑 N 条 complex（部分子任务真实检索失败、部分成功）→ 断言**真实失败的那几条** sub-task 进了 `bad_cases`、成功的不进；跑 simple 真实检索失败 → 仍正常落 Bad Case。

---

### P1-7 三级信号里的 L2/L3 在线上是死的
**位置**：`evolution.py:357`（`fs = state.get("faithfulness_score")` 全工程无写入点 → L2 恒 `None`）、`evolution.py:280-310`（`reinforce_feedback` 定义但**零调用**）、`rag_web_server.py:1280-1308`（`/api/feedback` 只 `save_feedback`+落 Bad Case，**不碰 playbook**、不传 `feedback_rating`）、`langgraph_rag_agent.py:402`（`extract()` 调 `evaluate_success(state)` 未传 `feedback_rating`）。
**根因（诚实结论）**：当前线上**只有 L1（检索级 `doc_grades`）真正生效**。
- L2：`evaluate_success` 逻辑已写，但管线从未把 `faithfulness_score` 写进 `state` → 恒为 `None` → `answer_ok=None`（中性）→ L2 形同虚设。
- L3：`reinforce_feedback` 完全没接线；`/api/feedback` 不知道本次问答命中了哪个 playbook（`used_playbook_pk` 算出来却没存进 task/feedback），无法回灌。
**影响**：README / 方案稿中"三级成功信号"的对外表述**目前不属实**，需先接线或先改文案。

**修复方案（最大隐患先说清：L2 绝不能进答题热路径）**：

> ⚠️ **原方案"每轮问答都加 `node_faithfulness` 打分"是错的，会引入两个致命问题**：
> 1. **延迟/成本崩坏**：每次回答都要多一次 DeepSeek 调用。本地 7b 跑几分钟的答案，再叠一个 judge，用户体验直接崩。
> 2. **回落陷阱**：DeepSeek 没配 key 时，`evalgrade` 自动回落 `local-small`（qwen2.5:1.5b）——**正是当初被判"幻觉相关=0.9"的那个弱评委**。生产 L2 反而因弱评委失准，比不接还糟。
> 所以 L2 的接线方式必须**离开热路径**。

- **L2 离线化 / 可采样 / 默认关**（三选一或组合，核心是"不阻塞主链路"）：
  - **异步后台打分**：`node_faithfulness` 改为回答**生成并返回用户之后**才触发的后台任务（线程池/异步队列），打分结果只用于离线沉淀/治理，**绝不**影响本次返回。
  - **采样 + 开关**：新增 `FAITHFULNESS_GRADE_ENABLED=false`（**默认关闭**），且默认只对采样流量（如 5%）打分；只有显式 `=true` 且有 DeepSeek key 时才参与。
  - **严禁回落弱评委**：DeepSeek 未配 key 时，**直接跳过 L2 打分**，绝不回落 `local-small`。
  - 阈值 `ANSWER_FAITH_THRESHOLD`（`evolution.py:46`）保留，但只在"开启且有强评委"时生效。
- **L3 接线（本身是反馈回调，不在热路径，可正常接；且可复用现成先例，无需新设计）**：
  - **沿用 `last_task_id` ContextVar 套路**：`langgraph_rag_agent.py:530-533`（`_ctx_last_task_id`）与 `:596-599`（`last_task_id` property）已是"请求级任务 ID 供前端点赞/点踩关联全链路 trace"的现成通道——**这正是 L3 要的「task → feedback 关联」机制**。L3 直接复用它：前端反馈带上 `last_task_id` → 后端据此查到对应的 `used_playbook_pk`（存进 task 时一并落库）→ 调 `reinforce_feedback`，不必另行设计关联键。
  1. `query()` 算出的 `used_playbook_pk` 随 task 落到 `task_queue`/`qa_feedback`（新增字段或塞进 `state` 透传，关联键即复用 `last_task_id`）。
  2. `rag_web_server.py:/api/feedback` 中：用请求里的 `task_id`（= `last_task_id`）反查本次问答命中的 `used_playbook_pk` → 若 `rating==1` 且能关联到 pk → `playbook_store.reinforce_feedback(pk, positive=True)`；`rating==-1` → `positive=False`；调用走 P1-9 的原子化改造。
  3. `extract()` 调 `evaluate_success(state, feedback_rating=...)` 时带上真实反馈（异步 triage 阶段回填）。
- **对外文案（必须同步改）**：在 L2/L3 真正接好前，README/稿子的"三级成功信号"改为**"L1 检索级已生效；L2 答案可信度、L3 用户反馈级为已设计、待采样/异步接线"**，消除立论被考古打脸的风险。

**验证**：集成测试确认开启 `FAITHFULNESS_GRADE_ENABLED=false` 时，单次问答**不再**触发额外 DeepSeek 调用（trace 里无 judge 请求、延迟无叠加）；单测注入 `state["faithfulness_score"]=0.3` 断言 L2 拦住正向沉淀；集成测试点"赞"断言对应 playbook `success_count` 变化且 `used_playbook_pk` 被正确关联。

---

### P1-8 一次请求 success_count 双计 + 失败回答也 +1
**位置**：`langgraph_rag_agent.py:982-991`（`node_classify` 命中即 `patch_success`）+ `:2576-2594`（`query()` 顶层命中又 `patch_success` 一次，且这处**在 `graph.invoke` 之前**）。
**根因**：
1. 双计：同一 playbook 命中时，classify 阶段 + 顶层 query 阶段各调一次 `patch_success` → "越用越快"数字虚增一倍。
2. **更隐蔽的 bug**：顶层 `patch_success`（`:2588`）发生在 `graph.invoke(...)` **之前**——只要命中了 playbook 就 +1，**哪怕后面 LLM 全挂、走熔断降级返回失败答案，这次也计了成功**。自进化数字被失败回答灌水。

**修复方案**：
- **去双计**：删除 `node_classify` 中的 `patch_success` 调用（`:982-991` 整段 try/except 块移除），仅保留 `prefill` 预填逻辑。
- **移到 `graph.invoke` 成功之后**：顶层命中只先**记录** `used_playbook_pk`，把 `store.patch_success(used_playbook_pk)` 挪到 `graph.invoke(...)` 跑通**且成功作答**之后，失败回答绝不 +1。
```python
# query() 顶层：先查、先记 pk（不 +1）
try:
    store = getattr(self, "playbook_store", None)
    if store is not None:
        hit = store.query_similar(question, self.tenant_id, top_k=1)
        if hit:
            prefill_rewrites = json.loads(hit.get("rewrite_text") or "[]")
            used_playbook_pk = hit.get("pk")   # ★ 只记录，不在此 patch_success
except Exception: ...
# ... 后续跑 graph.invoke（硬失败会抛异常）...
try:
    final_state = self.graph.invoke(initial_state, {"recursion_limit": 50})
except Exception:
    final_state = {}
# ★ AgentState 无 ok 字段：成功判定 = graph.invoke 未抛异常 且 final_state 取到非空 answer
answer = final_state.get("answer", "")
if answer:                                     # ★ 只有成功作答（非空 answer）才回写计数
    try:
        if used_playbook_pk:
            store.patch_success(used_playbook_pk)
    except Exception: ...
```

**验证**：对同一问题连续 2 次成功请求，断言该 playbook `success_count` 精确 +2（而非 +4）；构造 LLM 全挂熔断场景，断言该次**不 +1**。

---

### P1-9 patch_success 的 delete+insert 非原子，中途挂掉即丢经验
**位置**：`evolution.py:252-278`（核心 `:273-274`）
```python
self.client.delete(self.collection, filter=f'pk == "{pk}"')
self.client.insert(self.collection, [row])
self.client.flush(self.collection)
```
**根因**：read-modify-write 非原子。① `delete` 后 `insert` 前进程崩溃 → 该行永久丢失；② 并发命中同 `pk`：两线程都 `query` 到旧值、都 `delete`、都 `insert`，可能互相覆盖/丢一次计数。当前注释说"类缓存可接受"，但至少应消除崩溃丢行。

**修复方案**：
1. **用 `upsert` 替代 delete+insert**（Milvus 2.5 支持按主键 upsert，原子覆盖）：
```python
row["success_count"] = int(row.get("success_count", 0)) + 1
row["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
self.client.upsert(self.collection, [row])   # pk 为主键 → 按主键覆盖，无 delete 窗口
```
   - **前提①：主键确认**：`pk` 字段必须是集合 PRIMARY KEY VARCHAR（当前 `filter='pk == "..."'` 已按它查，需确认 schema 里 `pk` 确为主键；若不是，改用真实主键字段做 upsert）。
   - **前提②：`intent_vector` 必须显式回读**：原 `query(..., output_fields=["*"])` 在部分 pymilvus 版本下**不会**把 `FloatVector` 字段（`intent_vector`）包含在 `"*"` 里，导致 read-back 的 `row` 缺向量 → `upsert` 把这条经验的向量弄丢（经验检索退化/失效）。修复：把 `output_fields` 显式列出并**强制包含 `intent_vector`**：
     ```python
     res = self.client.query(
         self.collection, filter=f'pk == "{pk}"',
         output_fields=["pk","intent_text","query_type","rewrite_text",
                        "node_path","intent_vector","kb_version",  # ★ 显式点名，确保向量回读
                        "success_count","success_level","tenant_id","user_id"],
     )
     ```
2. **进程内 per-pk 锁**串行化同 pk 的 read-modify-write，消除并发自增丢失（跨进程仍靠 upsert 幂等兜底，计数偶发少 +1 可接受）。
3. `reinforce_feedback`（`:304-305`）同步改为 `upsert` + 同样显式 `intent_vector` 回读。
4. **运维步骤（上线必做）**：既有 `skill_playbooks` 集合需要一次**迁移/重建**才能稳定走 upsert（尤其确认 `intent_vector` 字段存在且维度匹配，否则首条 upsert 会因 schema 不符报错）。因 playbook 属"经验类缓存"、丢失可接受，**推荐上线时对 `skill_playbooks` 做一次 drop+重建**（首次运行 `_ensure` 已能自愈维度漂移，但显式重建最干净）；若不允许清空，先做全量 `query` 导出备份再重建回灌。

**验证**：并发 10 线程对同一 pk 各 +1，崩溃注入在 delete 后，断言最终计数 = 10-n（不丢行，最多少计）而非偶发 0；回读后断言 `row["intent_vector"]` 为非空向量（未被 `"*"` 漏掉）；上线后对老集合执行迁移，断言 upsert 不再报 schema 错。

---

### P1-10 playbook 不随文档更新失效（**用全局版本，不做 per-tenant**）
**位置**：`evolution.py` `RetrievalPlaybook`（字段：`intent_text/query_type/rewrite_text/node_path/relevant_sources/tenant_id/user_id/success_level`，**无 kb 版本**）；`kb_version.py:46-57` 已有 `bump_kb_version()`（ingest 成功末尾调用），但 playbook 从未读写它。
**根因**：KB 重 ingest 后，`relevant_sources` / `rewrite_text` 经验过期，但没有任何失效机制，旧经验仍被复用，可能把答案带偏。

**修复方案（澄清：版本是全局粒度，per-tenant 不成立）**：
> ⚠️ **方案修正**：原稿写"按 tenant 打版本"不符合现状。实测 `kb_version.py:46-57` 的 `get_kb_version(tenant)` **无论传入什么都强制读全局键 `rag:kbver:global`**（`bump` 端也只写全局）。原因见模块 docstring：现有 `CacheManager` 缓存键不含 tenant、ingest 多为整库操作，**全局版本最稳妥**。所以 playbook 失效直接复用全局版本即可，**不要**做 per-tenant 版本（做了也读不到，反而失效机制不触发）。

1. `RetrievalPlaybook` 增加 `kb_version` 字段；`save/save_or_merge` 时打上 `get_kb_version()`（**全局，不传 tenant**）。
2. `query_similar` 命中后，比较 `hit["kb_version"]` 与当前 `get_kb_version()`（**全局**）：
   - 一致 → 正常复用；
   - 不一致（任何 KB 已更新）→ **跳过复用**（fall through 到实时检索），或降权复用（乘衰减系数）并打标 `stale`。推荐"跳过复用"最安全。
   - 语义：任何一个租户的文档变更都会 bump 全局版本 → **全部** playbook 一次性失效。这是有意为之的"宁可多失效、不可复用过期经验"策略，简单且不会因版本粒度错配导致旧答案不刷新。
3. 集合 schema 增加 `kb_version` VARCHAR 字段（随 playbook 集合重建迁移，见 P1-9 运维步骤）。

**验证**：ingest 一批文档 → bump 全局版本 → 对同一问题再问，断言不再命中旧 playbook（走实时检索）。

---

## P1 · 凭据与配置外置（复制粘贴到处是）

### P1-11 硬编码 IP/密码散落 ~14 处
**已核实位置**：`memory_store.py:75-79`（`192.168.200.128`/`Root@2026`）、`kb_version.py:23-25`（`192.168.200.128`/`dev0619`）、`ingest/embed.py:74`（`http://192.168.200.128:11434`）、`ingest/cli.py:32`（`http://192.168.200.128:19530`）；其余见全模块扫描（`advanced_rag_agent.py`、`llm_gateway.py:181/800-802`、`prompt_manager.py:64-68` 等，均带 env 默认值兜底但默认值即写死 IP/口令）。
**根因**：基础设施地址与口令直接以 `os.getenv("X", "192.168.200.128")` 形式散落源码，改 IP 要改十几个文件；且默认口令进仓库。

**修复方案（与"开箱即用"对齐：默认值是 dev-profile，不是删掉）**：

> ⚠️ **方案修正**：原稿"全仓默认值全部移除、缺失即报错"会与**开箱即用体验**冲突——README 一键安装 + 首次启动指向 VM 演示路径，依赖那些 `os.getenv("X","192.168.200.128")` 默认值能直接连上演示环境。删光默认值会让开源用户首次启动即失败，演示路径断掉。正确做法：**默认值下沉到 `settings.yaml` 的 dev-profile，生产 profile 才 fail-fast**。

1. 新建**单一配置入口** `config/settings.yaml`，内分 **`profile: dev` / `profile: prod`** 两段：
   - `dev` 段保留现有演示默认值（`192.168.200.128` / `dev0619` / `Root@2026` 等），让 README 一键安装 + 首次启动开箱即连 VM 演示环境，开源体验保住。
   - `prod` 段**不写口令**，靠 env 注入；缺关键项（DB/Redis/Milvus 口令）启动直接 fail-fast。
2. 新增 `config_loader`（模块级单例）：`env 覆盖 > 当前 profile 的 yaml > 兜底`。**仅当 `profile=prod` 且关键项缺失/为空时**才启动失败并打印告警；`dev` 模式即使缺失也允许用内置默认值。
3. 全仓替换：`os.getenv("MYSQL_HOST","192.168.200.128")` 这类散落默认值**统一收口到 `settings.yaml:dev`**，`config_loader` 按当前 profile 解析；代码侧不再各自硬编码 IP/口令，只读 `settings.*`。保留 env 覆盖能力供容器部署。
4. 启动自检：若 `profile=prod` 且解析出的 host 仍是 `192.168.200.128` 这类开发默认值 → 打印 ERROR 提醒"生产环境请改配置/切换 profile"。

**验证**：`profile=dev` + 清空所有相关 env → 启动成功且连到 dev 演示地址（README 一键路径不破）；`profile=prod` 且删掉 `settings.yaml` 中 MySQL 口令 → 启动直接报错退出。

---

### P1-12 init_db.sql 头注释+seed SQL 明文写死 6 套默认密码
**位置**：`config/init_db.sql:5-9`（注释列出 `admin123/reader123/viewer123/jm123/yh123/Super@2026`）+ `:165-171`（6 条 `INSERT IGNORE` 密码哈希，salt+sha256 已知算法 → 口令可逆）。
**根因**：默认口令随仓库公开，运维若直接执行不改密即对外开放。

**修复方案**：
1. 文件头**醒目警告**（已部分存在，加强）："⚠ 以下为示例口令，生产环境对外开放前**必须**改密（`UPDATE admin_users SET password_hash=... WHERE username=...`），或部署脚本首次启动随机生成。"
2. 提供 `scripts/bootstrap_admin.py`：首次启动检测默认口令未改 → 自动为各账号生成随机口令并打印一次（写部署日志），不再使用明文示例。
3. README 部署章节增加"改密 checklist"步骤。

**验证**：执行 init_db.sql 后跑 `bootstrap_admin.py`，断言 6 个账号 `password_hash` 不再是脚本里那串固定值。

---

### P1-13 访问控制三套实现、互相不一致
**位置**：`advanced_rag_agent.py:182-185`（硬编码 `DOC_ACCESS_RULES={"JM-S509":"restricted"}`）vs `ingest/loaders.py:29-48`（读 `config/access_rules.yaml`）vs `rag_web_server.py:1417`（`access_fn=(lambda s: access_level)` 旁路，上传时直接用表单 `access_level` 覆盖，忽略文件名规则）。
**根因**：ACL 有且三处来源，谁有权限取决于走了哪条 ingest 入口。例如 `JM-S509` 文件若经 web 上传且表单选 `public`，会被存成 public，绕过代码里的 `restricted` 硬编码规则——权限语义随入口漂移。

**修复方案**：收敛为**单一 `AccessControl` 模块**作为唯一事实来源。
1. 抽离 `AccessControlFilter`（已在 `advanced_rag_agent` 存在）为独立模块 `access_control.py`，提供 `get_access_level(source, tenant_id, uploader_level) -> str`。
2. 三处统一调用它：
   - 删 `advanced_rag_agent.py:182` 的硬编码 dict（或仅作"yaml 缺失时的最后兜底"，并打告警）；
   - `ingest/loaders.py` 的 yaml 加载迁到该模块；
   - `rag_web_server.py:1417` 的 `lambda s: access_level` 改为 `lambda s: AccessControlFilter.get_access_level(s, tenant_id, uploader_level)`。
3. **权限取交集/取严**：上传表单的 `access_level` 与文件名规则**同时生效，取更严者**（文件名命中 `restricted` 关键字 → 强制 `restricted`，即使上传者选了 public，防权限降级）。
4. 单套 `access_rules.yaml` schema + 单点加载（与 P0-5 的 fail-closed 合并处理）。

**验证**：用 `JM-S509` 文件名经 web 上传并选 `public`，断言入库 `access_level=restricted`；三个入口对同一文件返回一致级别。

---

## 附录 A · 改动文件清单（预计）

| 文件 | 涉及项 |
|---|---|
| `advanced_rag_agent.py` | P0-1, P0-2(仅 legacy 路径+`cache.current_role`), P0-5(规则兜底), P1-13 |
| `prompt_manager.py` | P0-3 |
| `rag_web_server.py` | P0-4, P1-7(/api/feedback), P1-13, P1-8(若有调用) |
| `ingest/loaders.py` | P0-5, P1-13 |
| `langgraph_rag_agent.py` | P1-6, P1-7(L3接线), P1-8（**不含 P0-2**，默认入口已 ContextVar 隔离） |
| `evolution.py` | P1-6, P1-7, P1-9, P1-10 |
| `kb_version.py` | P1-10, P1-11 |
| `memory_store.py` / `ingest/embed.py` / `ingest/cli.py` / `llm_gateway.py` | P1-11 |
| `config/init_db.sql` + 新增 `scripts/bootstrap_admin.py` | P1-12 |
| 新增 `config/settings.yaml` + `config_loader.py` / `.env` | P1-11 |
| 新增/调整 `access_control.py` | P0-5, P1-13 |
| README / 方案稿 | P1-7（立论修正） |

## 附录 B · 实施顺序建议（发版节奏）

1. **发版 1（安全止血，必须先行）**：P0-1 ~ P0-5。互不依赖，可一次 PR。
2. **发版 2（闭环真实性）**：P1-6 → P1-8 → P1-9 → P1-10 → P1-7。先堵假 Bad Case 与数字虚增，再接线 L2/L3，并同步修 README 立论。
3. **发版 3（配置与权限收敛）**：P1-11 → P1-12 → P1-13。纯重构/外置，独立发版、低风险。

> 所有改动均**不自动提交**；按你的 git 纪律，改动留工作区，提交指令由你下发。
