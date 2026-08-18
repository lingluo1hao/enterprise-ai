# P1 自进化修复 · 测试报告（含 P1-R1 审查整改复测）

> 报告时间：2026-08-15
> 背景：P1 修复（P1-6/7/8/9/10）经 Claude Code 代码审查后，产出 7 项整改（P1-R1，含 1 个高优先级真 bug、2 个中危、4 个建议）；随后补齐测试用例并全量回归。
> 执行环境：`D:\prom\anaconda\envs\py310\python.exe`（Python 3.10.20）；测试零外部依赖——mock Milvus client / MySQL / kb_version，**不连 VM、不调 LLM**。
> 运行方式（项目根目录）：
> ```
> python tests/test_evolution_p1.py
> python tests/test_p1_review_fixes.py
> ```
> 改动纪律：本报告只跑测试 + 落文档，未提交代码。

---

## 0. 一句话结论

**45 个断言全部通过**（`test_evolution_p1.py` 16/16 + `test_p1_review_fixes.py` 29/29）。P1 五条修复 + P1-R1 七项审查整改的行为均已锁定；其中测试额外抓到 1 个隐蔽的 Python 运算符优先级 bug（写新代码时引入，非审查原始指出的点）。**后续补充测试（§6）补齐 10 组覆盖率缺口，再增 36 断言全绿，合计 81/81 无回归。**

---

## 1. 测试结果总览

| 用例组 | 对应项 | 结果 | 说明 |
|---|---|---|---|
| P1-6 | 三级成功信号守卫 | ✅ 3/3 | chitchat / 空 doc_grades → 不沉淀；L2 默认关不拦 |
| P1-7 | L2 开关接线 | ✅ 2/2 | 开启后 faithfulness 阈值生效 |
| P1-9 | upsert 原子化 + per-pk 锁 | ✅ 4/4 | 无 delete 窗口、向量回读、并发 +10 |
| P1-10 | kb_version 时效校验 | ✅ 3/3 | 过期跳过、当前复用、allowed_stale 放行 |
| R1-#4 | `_pk_locks` LRU 上限 | ✅ 4/4 | 超限淘汰最久未用、不超上限 |
| R1-#5 | `answer_failed` 显式标志 | ✅ 10/10 | 各分支失败置位、字符串启发式误判回归 |
| R1-#3 | `get_task_playbook_pk` 归属校验 | ✅ 7/7 | fallback + MySQL 双分支越权拦截 |
| R1-#6 | 迁移仅 errno=1054 触发 ALTER | ✅ 3/3 | 2013 连接异常不误操作 |
| R1-#1/#2 | kb_version=0 / fail-open 回归 | ✅ 2/2 | 合法 0 复用、读失败不过期 |
| 附带 | kb_version 失败冷却 | ✅ 5/5 | 失败进冷却、冷却期不碰 Redis |
| 补充 | 覆盖率补强（10 组缺口） | ✅ 36/36 | 见 §6：真守卫 / P1-8 双计 / _is_match 兜底 / L3 信号 / 冷却重试等 |
| **合计** | | ✅ **81/81** | 原 45 + 补充 36，无回归 |

---

## 2. 测试用例明细

### 2.1 P1-6 · 三级成功信号 `evaluate_success` 守卫（test_evolution_p1.py）

| 用例 | 断言 |
|---|---|
| 空 doc_grades → ok=False | 守卫触发点（chitchat / 无检索命中不沉淀） |
| doc_grades 2 相关 → ok=True, level=1 | 正常检索成功 |
| L2 默认关（faithfulness=0.2 仍 ok） | `FAITHFULNESS_GRADE_ENABLED=false` 不拦正向 |

### 2.2 P1-7 · L2 答案级开关（test_evolution_p1.py）

| 用例 | 断言 |
|---|---|
| L2 开：0.2 < 0.5 → ok=False | 阈值拦截 |
| L2 开：0.9 >= 0.5 → ok=True, level=2 | 达到阈值放行 |

### 2.3 P1-9 · `patch_success` upsert 原子化（test_evolution_p1.py）

| 用例 | 断言 |
|---|---|
| 走 upsert（无 delete 调用） | delete_calls=0, upsert_calls=1 |
| success_count 1→2 | 计数回写 |
| intent_vector 未丢 | 显式回读 1024 维保持 |
| 并发 10 线程同 pk → >=10 | per-pk 锁串行化 |

### 2.4 P1-10 · `query_similar` kb_version 时效校验（test_evolution_p1.py）

| 用例 | 断言 |
|---|---|
| kb_version=1 过期（当前=32）→ None | 跳过复用走实时检索 |
| kb_version=当前 → 复用 | 命中 pk-new |
| allowed_stale=True → 过期也放行 | 去重合并场景 |

### 2.5 R1-#4 · `_pk_locks` LRU 上限（test_p1_review_fixes.py）

> 修复：`_pk_locks` 从无限 dict 改为 `OrderedDict` + 上限 4096，超限淘汰最久未用。
> 测试：临时把上限压到 3，验证淘汰语义。

```
a b c 填满(上限3) → 访问a(移到末尾) → 加d(超限)
断言: b 被淘汰 / d 已加入 / a、c 保留 / 字典大小 <= 3
```

| 用例 | 断言 |
|---|---|
| 超上限后淘汰最久未用的 b | `"b" not in _pk_locks` |
| d 已加入 | `"d" in _pk_locks` |
| 仍保留近期访问的 a/c | a、c 均在 |
| 锁字典大小不超上限 | len <= 3 |

### 2.6 R1-#5 · `answer_failed` 显式标志（test_p1_review_fixes.py）

> 修复：AgentState 加 `answer_failed: bool`；writer/generate_simple/respond 失败分支置 True；`query()` 用 `final_state.get("answer_failed")` 替代字符串启发式（`"抱歉，无法回答这个问题。" not in answer`）。
> 测试：`object.__new__(LangGraphRAGApp)` 绕过 `__init__`，只测节点方法（mock `_do_generate` / `pm` / `llm`）。

| 用例 | 断言 |
|---|---|
| writer 空 results | answer_failed=True，answer 以"未检索到"开头 |
| writer 全子任务未检索到 | answer_failed=True |
| writer 有真实结果 | 不标失败，正常调用 llm |
| generate_simple 检索无果 | answer_failed=True |
| generate_simple 正常答案 | 不标失败 |
| **generate_simple 答案含"无法回答"字样** | 不标失败（**旧字符串启发式会误判**——回归点） |
| respond 无 answer 兜底 | answer_failed=True |
| respond 已有 answer | 不标失败 |
| query 判定：answer_failed=True | 不 patch_success |
| query 判定：正常答案含"无法回答"字样 | 仍 patch（显式标志兜底） |

### 2.7 R1-#3 · `get_task_playbook_pk` 归属校验（test_p1_review_fixes.py）

> 修复：`get_task_playbook_pk(task_id, user_id=None)`——MySQL 分支 `WHERE task_id=%s AND user_id=%s`；内存 fallback 分支同样校验归属；`/api/feedback` 传入 `user["user_id"]`。
> 测试：构造 `MySQLMemoryStore.__new__` 实例 + fake cursor/conn。

| 用例 | 断言 |
|---|---|
| fallback：归属用户（user_id=101） | 返回 pk-shared |
| fallback：他人 user_id=999 | **None（越权拦截）** |
| fallback：不带 user_id | 返回 pk（兼容旧调用） |
| fallback：task 不存在 | None |
| MySQL：SQL 含 `AND user_id` | 归属过滤下推 |
| MySQL：归属用户 | 返回 pk-shared |
| MySQL：他人 user_id=999 | **None（越权拦截）** |

### 2.8 R1-#6 · 迁移仅 errno=1054 触发 ALTER（test_p1_review_fixes.py）

> 修复：`_migrate_task_used_playbook_pk` 裸 `except` 改为——SELECT 抛 `pymysql.err.OperationalError` 且 `errno==1054`（unknown column）才 ALTER；其他异常 `raise` 抛给外层统一告警。
> 测试：构造 `_FakeMigConn` 模拟不同异常。

| 用例 | 断言 |
|---|---|
| errno=1054（缺列） | ALTER + commit |
| errno=2013（连接断开） | **不 ALTER（修复前裸 except 会误操作）** |
| 无异常（列已存在） | 不 ALTER |

### 2.9 R1-#1/#2 · kb_version=0 合法 + 读失败 fail-open（test_evolution_p1.py）

> 修复：`(int(hit_ver) if hit_ver is not None else -1)` 括号修正；`cur_version=None` 时跳过时效校验。
> 测试：monkeypatch `kb_version.get_kb_version` 返回 None / 0。

| 用例 | 断言 |
|---|---|
| cur_version=None（读失败）→ fail-open | 跳过校验，允许复用旧经验 |
| kb_version=0（当前=0）→ 合法复用 | 全新部署不误杀（falsy-zero 回归） |

### 2.10 附带 · kb_version 失败冷却（test_p1_review_fixes.py）

> 背景：实测发现 redis-py 8 默认 5s connect_timeout + 内置重试，Redis 不可达时单次 `get_kb_version()` 阻塞 **58.6s**；若在检索热路径上每次实时检索都调它，会把问答拖死。
> 修复：读侧加进程内缓存 + 失败冷却（5s）——失败后冷却期内快速返回 None（fail-open），冷却后重试一次。验证：首次失败 58.6s → 第二次 0.0s。

| 用例 | 断言 |
|---|---|
| 读成功返回 32 | int 值 |
| 读成功后无冷却 | `_cache_fail_until == 0` |
| 失败返回 None | fail-open 语义 |
| 失败后进入冷却 | `_cache_fail_until > now` |
| 冷却期内不碰 Redis | get 调用计数不变，立即 None |

---

## 3. 测试过程中额外抓到的问题

> 均由测试驱动暴露，非审查原始指出的点。

### 3.1 三元表达式优先级 bug（R1-#1 修复时引入）

**位置**：`evolution.py` `query_similar` 时效校验（P1-R1 #1 初次写法）。

```python
# 初次写法（有 bug）——三元优先级低于 !=，条件恒为 int(hit_ver) 的真值
if int(hit_ver) if hit_ver is not None else -1 != cur_version:
    ...
```

`A if C else B != D` 被解析为 `A if C else (B != D)`：当 `hit_ver` 非 None 时，条件恒为 `int(hit_ver)` 的真值（非 0 即 True）→ **所有命中经验无条件判过期**。

**暴露方式**：`test_evolution_p1.py` 用例 "kb_version=当前 → 正常复用" 失败（日志 `kb_version=32 vs 当前 32` 仍判过期）。

**修复**：加括号 `(int(hit_ver) if hit_ver is not None else -1) != cur_version`。修复后 16/16 全过。

### 3.2 `node_writer` 对 `state["query"]` 的 eager 默认值依赖

`state.get("resolved_query", state["query"])` 中 dict.get 的默认参数是 **eager 求值**——`resolved_query` 缺失时仍会访问 `state["query"]`。测试传入缺失 `query` 的 state 触发 `KeyError`。非缺陷（真实调用总带 query），仅测试构造 state 时需补全字段。

---

## 4. 结论与建议

- **行为锁定**：P1 五条修复 + P1-R1 七项整改的语义已被 45 个断言固化，后续改动出现回归会立即暴露。
- **建议后续**：
  1. 在线（连 VM）跑一条真实问答验证 `query_similar` 时效校验与 `answer_failed` 的端到端表现（当前测试全为机制级 mock）。
  2. `node_respond` 兜底文案 "抱歉，我无法回答这个问题。" 与 `query()` 默认 "抱歉，无法回答这个问题。" 措辞不一致（差一个"我"），建议统一常量，避免未来字符串比对又踩坑。
  3. 若希望 `_pk_locks` 上限可配置，可将 `_PK_LOCKS_MAX` 改为从环境变量读取。

## 5. 清理记录

- 本次仅新增/修改代码与文档，无临时脚本残留。
- 相关测试文件：`tests/test_evolution_p1.py`（16 断言）、`tests/test_p1_review_fixes.py`（29 断言）、`tests/test_p1_coverage_gaps.py`（36 断言，本次新增）。
- 所有改动未提交（git 纪律）。

---

## 6. 补充测试（覆盖率补强）

> 任务：对原报告（45 断言）做覆盖率审计，定位零/低覆盖分支，补齐用例并全量回归。
> 新增文件：`tests/test_p1_coverage_gaps.py`（零外部依赖，同机制级 mock 风格，运行方式同上）。

### 6.1 覆盖率审计结果（原报告缺口清单）

原报告 §0 称"45 全过 ≠ 全覆盖"但未展开。审计发现以下 10 类分支在原两套测试中**零或极低覆盖**：

| # | 缺口 | 原状态 | 风险 |
|---|---|---|---|
| ① | 真守卫 `Extractor.extract`（line ~501） | 零用例 | 报告 §2.1 反复称"守卫在 save_history"，但 `evolution.py` **无 `save_history` 函数**——真守卫从未被任何用例验证 |
| ② | P1-8 `node_classify` 命中后不 patch_success（防与 `query()` 双计） | 总览提过，零用例 | 双计致 success_count 虚高、经验被重复加权 |
| ③ | `query_similar._is_match` 距离阈值 + 文本兜底 | mock 永远 distance=0.0，未真验证 | Milvus 余弦距离失真（相同向量=1.0）时兜底是否生效未知 |
| ④ | L3 `feedback_rating` 信号（赞/踩/非数字） | 零用例 | 三级信号最高层未锁定 |
| ⑤ | `extract_failure` 边界（query 空 / bad_sources / root_cause） | 零用例 | 失败经验沉淀边界未验证 |
| ⑥ | `query_similar` 命中字段透传（rewrite_text/query_type/kb_version/success_count） | 零用例 | 经验复用时字段丢失无感知 |
| ⑦ | R1-#3 MySQL 无 user_id 旧调用分支（SQL 不含 AND user_id） | 仅测带 user_id | 旧调用路径未覆盖 |
| ⑧ | P1-9 `reinforce_feedback` 踩 → success_count=0 | 零用例 | 负反馈清空逻辑未验证 |
| ⑨ | P1-9 `patch_success` 不存在的 pk → 直接 return | 零用例 | 越界 pk 行为未验证 |
| ⑩ | kb_version 失败冷却到期后重试一次 | 原只测冷却期不碰 Redis | 到期后重试成功路径未验证 |

### 6.2 新增用例明细（10 组，36 断言）

**A. P1-8 · `node_classify` 命中后不双计**（2 断言）
- 命中 playbook → `used_playbook_pk` 透传进 state
- `node_classify` 内**绝不**调 `patch_success`（防与 `query()` 重复计数）

**B. P1-6 · `Extractor.extract` 真守卫**（7 断言）
- chitchat / 空相关 → `extract` 返回 None（不沉淀）
- 空 `doc_grades` → None
- 有相关 + 有 `intent_text` → 返回经验对象
- `success_level` 透传为 1（检索级）
- `intent_text` 取自 `resolved_query` 且非空
- `kb_version` 快照为当前版本 32
- `intent_text` 为空 → None

**C. `query_similar._is_match` 距离 + 文本兜底**（5 断言）
- `dist=0.0 <= HIT_DIST` → 命中
- `dist=0.99 > HIT_DIST` 且文本不同 → None（不命中）
- dist 失真但 `intent_text` 完全一致 → 文本兜底命中
- dist 失真但 `difflib>=0.92` 近重复（"心跳间隔测试。" vs "心跳间隔测试"，相似度 0.941）→ 命中
- `dist=None` 走文本兜底，一致文本 → 命中

**D. `evaluate_success` L3 `feedback_rating` 信号**（4 断言，注意该参数是**位置/关键字参数**，非 state 字段）
- `feedback_rating>=1` → level 升到 3
- `feedback_rating<=-1` → L3 为负，ok=False（不沉淀负经验）
- `feedback_rating` 非数字 → 回退中性（不拦正向）
- L2+L3 同时为真 → level=3（取最高层级）

**E. `extract_failure` 边界**（3 断言）
- query 为空 → 返回 None
- 被召回但不相关的源收集进 `bad_sources`
- `root_cause` 标记为"检索未命中"

**F. `query_similar` 命中字段透传**（4 断言）
- 命中透传 `rewrite_text` / `query_type` / `kb_version` / `success_count`

**G. R1-#3 · MySQL 无 user_id 旧调用**（2 断言）
- 无 user_id 旧调用 → 返回 pk
- 无 user_id → SQL **不含** `AND user_id`（兼容旧调用）

**H. P1-9 · `reinforce_feedback` 踩 → 0**（2 断言）
- 负反馈 → `success_count` 压 0
- 负反馈 → 走 upsert（原子覆盖）

**I. P1-9 · `patch_success` 不存在的 pk**（2 断言）
- 不存在的 pk → 不抛异常
- 不存在的 pk → 不触发 upsert/delete（`_read_pk_row` 返回 None 直接 return）

**J. 附带 · kb_version 失败冷却到期后重试**（5 断言，仅 mock 底层 `_redis`，**保留真实冷却逻辑**）
- 首次失败 → 返回 None（fail-open）
- 首次失败后进入冷却
- 冷却期内快速失败（不触发 Redis）
- 冷却到期 → 重新重试 Redis 并成功返回 32
- 成功后清除冷却（`_cache_fail_until` 归零）

### 6.3 验证结果

- 补充测试：`tests/test_p1_coverage_gaps.py` → **36/36 PASS**（EXIT=0）
- 回归原两套：`test_evolution_p1.py` 16/16 + `test_p1_review_fixes.py` 29/29，均 EXIT=0（**无回归**）
- **合计 81/81 断言全绿**

### 6.4 文档勘误（原报告表述修正）

P1-6 的「守卫」实际是**两道防线**，分别位于两个文件，本次核对确认均已落地：

1. **`Extractor.extract`**（`evolution.py` 约 line 501）：三级成功信号的真实前置守卫——`relevant < GRADE_THRESHOLD` 时 `return None`，不沉淀经验。**这条此前零测试**（原测试只测了 `evaluate_success` 与 `extract_failure`），本次 §6.2 B 组首次补齐精确断言。✅
2. **`node_save_history`**（`langgraph_rag_agent.py:1156`，即此前所称 "save_history 守卫"）：图节点内对 chitchat / 空 doc_grades 的二次拦截（`:1193-1194` P1-6 守卫注释），与 `extract` 形成双保险。

> 修正说明：初版审计稿误称「`save_history` 函数根本不存在」。实际它存在于 `langgraph_rag_agent.py`（节点 `node_save_history`），审计方在 `evolution.py` 内检索未命中导致误判。本节予以更正——**不是函数不存在，而是其守卫此前未被测试触达**；`Extractor.extract` 的守卫本次已补齐（§6.2 B 组）。
