# 本项目 AgentWorkflow · Bad Case 自动诊断方案

> 日期：2026-08-17 ｜ 用途：开源项目方案文档，供评审与复用
> 一句话：用「Workflow 标准诊断流水线 + ReAct 开放探查」的 agent workflow，把进入 bad_cases 库的失败样本从**人工逐条 triage**变成**自动诊断回写、人工只确认**——两种 Agent 形态在这个场景里有真实分工，不是演示。

---

# 第一部分：背景与问题

## 一、现状链路与三个具体的洞

bad case 闭环现状（三个入口 → 库 → 人工处理）：

```
入口① 用户点踩     POST /api/feedback（rag_web_server.py:1328）
入口② pipeline 异常 langgraph_rag_agent.py:1215
入口③ 评测失败      evalkit run
        │
        ▼
bad_cases 表（root_cause 可空、status=open）
        │
        ▼
人工在 /admin/bad_cases 逐条看：读 query / 故障答案 / 期望答案 →
凭经验判根因 → PATCH 更新状态与诊断 → resolved
```

**洞 1 · 点踩样本零诊断**：入口①②进来的 case `root_cause=None`、diagnosis 只有一句"待 triage"。现有 `evalkit/triage.py` 的 `classify()` 是规则阈值判定，**依赖 golden 集信号**（`Recall@5` / `bury` / `nDCG@5` / `forbidden_hits`），点踩 case 没有 gold 标注，这套规则根本跑不了——于是全部落在人工头上。

**洞 2 · 根因写不进库**：`memory_store.update_bad_case_status()`（memory_store.py:847）只支持 `status / resolved_by / diagnosis / expected` 四个字段，**没有 `root_cause` 参数**——连人工通过 PATCH 接口都设不了根因（`rag_web_server.py:1364` 的 PATCH 同样透传不了）。GET 的 `?root_cause=R5` 筛选形同虚设。

**洞 3 · 诊断没有执行体**：一次像样的诊断需要连续动作——复跑检索看召回、用强 judge 复判当时的答案、对比矛盾信号、按 R1~R8 归因、写出带证据的结论。这是多步、有分支、可能要临场换角度查证的**agent 任务**，现有系统没有承载它的模块。

## 二、为什么用 agent workflow 的两种形态（真实分工）

| 形态 | 在本场景的分工 | 为什么合适 |
|---|---|---|
| **Workflow（预定义 DAG）** | 承载标准诊断五步：复跑检索 → 探查文档相关性 → judge 复判 → 规则归因 → 回写 | 步骤固定可枚举；LLM 调用数恒定（3~4 次），成本可预测；失败可定位到节点；大部分 case（有明确信号）在这层结案 |
| **ReAct（动态决策）** | 标准流水线**低置信/信号矛盾**时的开放探查：该查什么事先不知道——可能要换关键词重搜、查文档是否入库、查相似历史 case | 任务不可预知，需要模型逐步决定下一步；复用现有 `ReActAgent` + `SkillRegistry`，每步走网关 `react` 链 |

这正是业界"Workflow 做骨架 + ReAct 做局部智能节点"生产共识（见附录）在本项目的落地位置——**分支规则写在代码里（何时升级到 ReAct），LLM 只在节点内部干活**。

## 三、改造后的人工工作量变化

| 步骤 | 现状 | 改造后 |
|---|---|---|
| 读 query / 答案 | 人工 | — |
| 复跑检索验证 | 人工（或跳过） | 自动（DAG `rerun_retrieval`） |
| 复判答案质量 | 人工 | 自动（`JudgeLLM.grade`，`evalgrade` 链） |
| 判根因 R1~R8 | 人工凭经验 | 自动（规则归因，复用 `triage._ROOT_CAUSES` 文案库）；低置信自动转 ReAct 探查 |
| 写诊断 + 建议 | 人工打字 | 自动生成，含证据与置信度 |
| **确认 / 修复 / resolve** | 人工 | **仍人工**（诊断是建议，结论归人） |

---

# 第二部分：目标与非目标

## 一、目标

1. 新增 `agentworkflow/` 包：诊断流水线（Workflow DAG）+ 探查引擎（ReAct）+ 统一轨迹（Trace）。
2. 两个触发入口：CLI `python -m agentworkflow.diagnose --bc-id N`；Web `/admin/bad_cases` 一键「自动诊断」按钮。
3. 自动产出并回写：`root_cause`（R1~R8）+ `diagnosis`（证据 + 建议，复用现有文案库）+ 置信度 + `status: open → in_progress`。
4. 补洞 2：`update_bad_case_status()` 增加 `root_cause` 参数（向后兼容），PATCH 接口同步透传。
5. 全部 LLM 调用走网关现有链（`evalgrade / react / generate`，**零新增路由、零改 yaml**），Token 按触发者归因。
6. 每次诊断落盘 run json（轨迹 + 证据 + 结论），可复查、可回放。

## 二、非目标

- **不做自动修复执行**：诊断产出"根因 + 建议"；playbook 强化仍走既有 `evolution.patch_success` 流程（P2 才考虑对接）。
- **不取代人工确认**：`resolved` 仍由管理员操作；诊断建议附带置信度，页面明示"机器建议"。
- **不动 evalkit 门禁语义**：golden 集评测与 CI 退出码不变。
- **不动生产问答链路**：`/api/query` 与 LangGraph 主图零改动。
- **不做定时自动批量**（P2 再议）：先做单条按需触发，控制网关配额消耗。

---

# 第三部分：总体设计

## 一、架构总览

```
触发       CLI：python -m agentworkflow.diagnose --bc-id 17
           Web：/admin/bad_cases → 「自动诊断」按钮
                 → POST /api/admin/bad_cases/<id>/diagnose（admin token）
    │
    ▼
agentworkflow/ 诊断运行时
    ├─ diagnose.py   诊断入口：读 bad case → 驱动 DAG → 低置信升级 ReAct → 回写
    ├─ pipeline.py   诊断 DAG（langgraph StateGraph，预定义 6 节点）
    ├─ probe.py      ReAct 探查（复用 advanced_rag_agent.ReActAgent + SkillRegistry）
    ├─ trace.py      StepRecord 统一轨迹（两引擎同构，落 runs/）
    └─ runs/         诊断 run json 落盘（gitignore）
    │
    ▼                    复用（零改或最小改）
DocSearchSkill（复跑检索）   evalkit.judge.JudgeLLM（evalgrade 复判）
evalkit.triage._ROOT_CAUSES（R1~R8 文案库）
memory_store.update_bad_case_status（+root_cause 参数 ← 本方案唯一存储层改动）
LLM Gateway（task=react/evalgrade/generate）  audit_logger（+1 类操作）
```

## 二、诊断 DAG（Workflow 形态，6 节点）

| # | 节点 | 职责 | 类型 / 网关链 |
|---|---|---|---|
| 1 | `load_case` | 读 bad case（query / answer / expected / source），构造诊断上下文 | 纯代码 |
| 2 | `rerun_retrieval` | 以 admin 视角复跑检索（绕过 Redis 缓存，取真实当前召回），记录命中文件与租户归属 | 工具（`DocSearchSkill`） |
| 3 | `probe_docs` | 判定当前检索结果**能否回答 query**（docs_relevant + 缺什么） | LLM，`evalgrade` 链（强模型判定，压误判） |
| 4 | `judge_answer` | 复用 `JudgeLLM.grade(query, 故障答案, contexts=复跑命中)` → faithfulness / relevancy / reason | LLM，`evalgrade` 链 |
| 5 | `classify_root_cause` | **代码规则**归因（见下表），产出 R 码 + 证据 + 建议 + 置信度 | 纯代码（复用 `_ROOT_CAUSES` 文案） |
| 6 | `writeback` | 回写 `root_cause / diagnosis / status=in_progress`，落 run json | 纯代码 |

### 归因规则（节点 5，分支全部写在代码里）

| 优先级 | 条件（基于 2/3/4 节点信号） | 结论 | 置信度 |
|---|---|---|---|
| 1 | 复跑命中了**其他租户**的文档（文件名→租户映射不符） | **R8** 跨租户泄漏 | 高 |
| 2 | 复跑零命中 或 docs_relevant=False | **R1** 召回丢失（附：可能原因=未入库/切片/embedding，建议 `--mode raw` 复核） | 高 |
| 3 | docs_relevant=True 且 faithfulness<0.6 | **R5** 生成幻觉（附 judge reason 作证据） | 高 |
| 4 | faithfulness≥0.6 且 relevancy<0.6 | **R6** 答非所问 | 中 |
| 5 | 全部指标正常但用户点了踩（source=feedback） | 低置信 → **升级 ReAct 探查** | — |
| 6 | judge 不可用 / 检索执行异常 | 结论"无法自动归因"，保留现场，转人工 | — |

> 规则优先级与 `evalkit/triage.py` 的判定精神一致（安全类 R8 → 召回类 R1 → 生成类 R5/R6），但信号源从 golden 指标换成"复跑实测 + LLM 探查"，这正是点踩 case 能被自动诊断的关键。

## 三、ReAct 探查（形态二，低置信分支）

- 触发：归因置信度低 / 信号矛盾（如检索正常、判分正常、用户仍点踩——可能是 expected 有隐含要求、时间敏感问题、或多跳没拆）。
- 实现：实例化 `ReActAgent(create_llm(), skill_registry, max_steps=5)`，任务提示词=「对给定 bad case 做根因探查」，工具=现有 `SkillRegistry` 全量技能（doc_search 等），`user=` 置为触发者用于 Token 归因。
- 产出：要求 Final Answer 输出结构化结论（R 码 + 证据 + 建议），解析失败则降级为"探查完成但未归因，转人工"（诚实边界）。
- 升级动作本身在 DAG 的**代码分支**里（`classify_root_cause` 置信度不足 → 进 `probe` 支路），LLM 不能改变流程形状。

## 四、统一 Trace

沿用双形态统一轨迹设计：`StepRecord{engine, seq, step_type(think/act/observe/node_enter/node_exit), node_or_tool, input, output, llm_task, tokens_in/out, latency_ms, ts}`。DAG 节点包装收集 + ReAct 的 `ReActStep` 列表归一。落 `agentworkflow/runs/diag-<bc_id>-<ts>.json`，run json 即"诊断证据链"，页面与报表共用。

## 五、安全与集成约束

| 项 | 约束 |
|---|---|
| 权限 | Web 触发仅 `admin / super_admin`（同现有 bad_cases 接口）；CLI 本地运维用 |
| 限流 | 新路由登记进 `_get_rate_limit_for_route()`，IP 令牌桶 6/min（单次诊断 3~6 次 LLM 调用，串行数十秒） |
| 审计 | `audit_logger` 新增第 8 类操作 `bad_case_diagnose`（target=bc_id） |
| 缓存 | 复跑检索**绕过 Redis 两级缓存**（诊断要看真实召回，不是缓存命中） |
| 网关配额 | 单条诊断 LLM 调用 ≤6 次（DAG 3 次 + ReAct 兜底 3~5 步封顶）；批量触发在 P2 前不开放 |
| 租户 | 复跑以 admin 视角取全库召回用于 R1/R4/R8 判断；**R4（权限误杀）只能"提示"不能"定论"**——点踩样本没有记录当时用户角色，页面明示 |

---

# 第四部分：详细设计

## 一、包结构与文件职责

```
agentworkflow/
  __init__.py      导出 diagnose_bad_case()
  diagnose.py      入口：编排（读 case → DAG → 可选 ReAct → 回写 → 落盘）
  pipeline.py      诊断 DAG（StateGraph + AgentState 定义 + 节点实现 + 归因规则表）
  probe.py         ReAct 探查适配器 + 结论解析
  trace.py         StepRecord 协议 + 两引擎轨迹归一 + run json 落盘
  __main__.py      CLI：python -m agentworkflow --bc-id N [--dry-run] [--json]
agentworkflow/runs/        诊断产物（.gitignore）
tests/test_agentworkflow.py  纯 Python PASS/FAIL（零外部依赖，项目测试惯例）
```

## 二、存储层唯一改动：`update_bad_case_status` 增加 `root_cause`

```text
签名：update_bad_case_status(bc_id, status=None, resolved_by=None,
                             diagnosis=None, expected=None, root_cause=None)
```

- SQL 分支增加 `root_cause = %s`；内存降级分支同步；`rag_web_server.py` PATCH 接口透传 `root_cause`（人工也能改根因了，顺带修洞 2）。
- 不改表结构（`root_cause` 列本就存在，VARCHAR(8)）。

## 三、Web 触发入口

- 新路由 `POST /api/admin/bad_cases/<int:bc_id>/diagnose`：
  - `_require_auth()` + 角色校验（同现有 admin 接口）；记忆层不可用 503。
  - 请求体可选 `{"dry_run": true}`——只出诊断不落库（预演模式）。
  - 响应：`{ok, root_cause, title, diagnosis, confidence, engine("workflow"|"react"), trace摘要, run_id}`；耗时较长（30~90s，CPU 模型下更久），前端按钮置 loading 态。
- `_BADCASE_PAGE`（rag_web_server.py:1032）改造：右抽屉加「⚡ 自动诊断」按钮 + 结果区（R 码徽标 / 置信度 / 诊断文本 / 建议动作 / "机器建议，确认后请手动流转状态"提示）；诊断后自动刷新列表使 `?root_cause=` 筛选生效。

## 四、网关任务链映射（全部现有链）

| 诊断动作 | task | 链（config/llm_gateway.yaml） |
|---|---|---|
| `probe_docs` 文档相关性判定 | `evalgrade` | `[deepseek-chat, local-small, local-qwen]` |
| `judge_answer` 答案复判 | `evalgrade`（`JudgeLLM` 默认） | 同上 |
| ReAct 探查每步 | `react` | `[local-qwen, deepseek-chat]` |
| ReAct 兜底总结 | `generate` | `[local-qwen-gen, deepseek-chat, qwen-plus]` |

## 五、CLI

```bash
python -m agentworkflow --bc-id 17            # 诊断并回写
python -m agentworkflow --bc-id 17 --dry-run  # 只出诊断不落库
python -m agentworkflow --bc-id 17 --json     # 机器可读输出（含完整 trace）
```

---

# 第五部分：实施计划

## P0 · 诊断运行时（CLI 可用）

| # | 事项 | 涉体 |
|---|---|---|
| 1 | `trace.py` StepRecord + run json 落盘 | 新增 |
| 2 | `pipeline.py` 诊断 DAG（6 节点 + 归因规则） | 新增 |
| 3 | `probe.py` ReAct 探查适配 | 新增（复用 `ReActAgent`） |
| 4 | `memory_store.update_bad_case_status` 增加 `root_cause` | 小改（唯一存储层改动） |
| 5 | `diagnose.py` + `__main__.py` CLI | 新增 |
| 6 | `tests/test_agentworkflow.py`（归因规则表驱动单测 / trace 归一 / 结论解析降级，零外部依赖） | 新增 |

## P1 · Web 集成

| # | 事项 | 涉体 |
|---|---|---|
| 1 | `POST /api/admin/bad_cases/<id>/diagnose` + 限流登记 | `rag_web_server.py` 小改 |
| 2 | PATCH 接口透传 `root_cause`（修洞 2 的人工侧） | `rag_web_server.py` 小改 |
| 3 | `_BADCASE_PAGE` 自动诊断按钮 + 结果区 | `rag_web_server.py` 小改 |
| 4 | `audit_logger` 新增 `bad_case_diagnose` 操作 | `audit_logger.py` 小改 |

## P2 · 增强（可选，另行评估）

1. 批量诊断：`--status open --limit N` 批量跑（受网关配额约束，需节流）。
2. 诊断建议直达自进化：R1 类建议生成 playbook `rewrite_text` 草稿，接 `evolution.save_or_merge`。
3. 新增 `KbInventorySkill`（查文件是否入库/chunk 数）供 ReAct 探查 R1 时使用。
4. 诊断通过率报表：按根因统计"机器诊断 vs 人工最终确认"一致率，作为自进化体系新指标。

---

# 第六部分：验收标准

**功能**

1. 点踩产生一条 bad case → CLI 或页面「自动诊断」→ `root_cause / diagnosis / status=in_progress` 自动落库，页面刷新可见 R 码徽标与诊断文本。
2. `--dry-run` 不写库；低置信 case 自动走 ReAct 探查并在结果中标注 `engine=react`。
3. 每次 run 落 `agentworkflow/runs/*.json`，含完整 StepRecord 轨迹与证据。
4. PATCH 接口可更新 `root_cause`（洞 2 修复）。

**质量**

5. 构造 3 类已知根因种子样本（R1：检索不到的生僻问题；R5：答案含编造字段；R6：跑题答案）——自动诊断 3/3 命中正确 R 码（种子构造脚本随测试交付）。
6. 现有 `tests/` 全绿；生产问答链路、16 个网关调用点、evalkit 门禁零改动。

**诚实边界**

7. 诊断正确率**不承诺** R1~R8 全类覆盖（R2/R3/R4 需 golden 对比信号，点踩场景拿不到）；页面明示置信度与"机器建议"属性；R4 仅提示不定论。

---

# 第七部分：风险与边界

| 风险 | 缓解 |
|---|---|
| 复跑检索时知识库已更新，当时故障无法复现 | 诊断结论标注"基于当前库状态"；run json 留存完整证据链供人工核对 |
| 本地 7b 做 docs_relevant 判定噪声大 | 该节点固定走 `evalgrade` 链（DeepSeek 优先），未配 key 时回落本地并降低置信度档位 |
| Redis 缓存污染复跑结果 | 复跑显式绕过缓存（直连检索），实现时验证 `DocSearchSkill` 的缓存路径 |
| ReAct 探查格式不守约 | 沿用 `_parse_react_output` 降级 + 结构化结论解析失败即转人工（不硬猜） |
| CPU 模型下单次诊断慢（分钟级） | 页面 loading 态 + 限流 6/min；批量是 P2 且必须节流 |
| 网关配额被诊断打满 | 单条 LLM 调用 ≤6 次硬顶；诊断串行执行 |

---

# 第八部分：与现有架构契合点核对表

| 现有约定（AGENTS.md / README） | 本方案符合性 |
|---|---|
| 所有 LLM 调用必穿 LLM Gateway | ✓ 全部 task= 走现有链 |
| MySQL 不自动建表 | ✓ 零 schema 变更（root_cause 列已存在） |
| 缓存键按角色/租户隔离 | ✓ 诊断复跑显式绕缓存，不碰缓存键规则 |
| 审计 7 类操作全覆盖敏感动作 | ✓ 新增第 8 类 `bad_case_diagnose` |
| 测试为纯 Python PASS/FAIL 脚本、零外部依赖 | ✓ `tests/test_agentworkflow.py` 同风格 |
| 文档/注释/提交信息中文 | ✓ |
| 温度纪律（决策 temp=0 / 生成 0.3） | ✓ 判定走 evalgrade(0.1)，探查走 react(0)，总结走 generate(0.3) |

---

# 附：实施状态（2026-08-17 · P0 + P1 已落地）

## 一、已交付

| 交付物 | 位置 |
|---|---|
| 诊断运行时包（DAG / ReAct 探查 / 统一轨迹 / CLI） | `agentworkflow/`（`pipeline.py` 6 节点 DAG、`probe.py`、`trace.py`、`rules.py` 纯规则层、`diagnose.py` + `__main__.py`） |
| 存储层扩展：`root_cause` 可写 + 按主键读取 | `memory_store.py`（`update_bad_case_status` 增 `root_cause` 参数；新增 `get_bad_case`） |
| Web 触发接口 | `rag_web_server.py`：`POST /api/admin/bad_cases/<id>/diagnose`（admin 鉴权 + 组件注入 + 审计 `bad_case_diagnose` + 专用限流 6/min） |
| PATCH 透传 root_cause（修洞 2 的人工侧） | `rag_web_server.py` `api_admin_bad_case_update` |
| 管理页一键诊断 | `_BADCASE_PAGE`：「⚡ 自动诊断」按钮 + 结果区（R 码/置信度/引擎/诊断文本）+ 根因改判下拉；根因筛选标签同步为 triage 现行口径（R1~R8） |
| 零依赖单测 | `tests/test_agentworkflow.py`（归因规则表驱动 / 解析器 / 统一轨迹） |

## 二、验证结果

| 项 | 结果 |
|---|---|
| `python tests/test_agentworkflow.py` | 25 PASS / 0 FAIL |
| 回归：`test_gen_routing` / `test_harness_grading` | 7/7、10/10 全过 |
| 回归：`test_ingest` | 67 PASS / 2 FAIL（`删实体>0`、`父窗口含层级路径`——**既有失败**，本次未触碰 `ingest/`，与诊断改动无关） |
| 模块导入 + 路由注册（真实 anaconda py310 环境） | `agentworkflow` 可导入；`/api/admin/bad_cases/<id>/diagnose` 已注册；`update_bad_case_status` 签名含 `root_cause`；`get_bad_case` 存在 |
| root_cause 回写实测 | MySQL 不可达自动降级内存模式，建→改（`root_cause=R5`）→查回环通过 |
| 语法检查（py_compile 全部改动文件） | 通过 |

## 三、live 验证结果（2026-08-17 · VM 192.168.200.128 在线实测）

| # | 验证项 | 结果 |
|---|---|---|
| 1 | 端到端真实诊断（真实点踩 case #9，yh 租户） | ✅ **R7 拒答错误（误拒）**，中置信：yh 视角复跑成功召回学生证协议 V5.0（含工作模式内容），但当时答案表现为拒答——与 BadCase9 历史报告在当前库状态下吻合；root_cause/diagnosis/status 已回写 |
| 2 | R1 种子（#10，财务问题） | ✅ R1 高置信：scoped 召回 8 条协议片段，LLM 探查正确判定"无财务数据"→ 规则先于 judge 阈值触发 |
| 3 | R5 种子（#14，编造停止位数值） | ✅ **R5 高置信**：faithfulness=0.00，judge 证据"停止位数值与上下文不符，且编码方式无出处" |
| 4 | R6 种子（#15，答 UTF-8 不答 SOS 格式） | ✅ **R6 中置信**：faithfulness=1.00 + relevancy=0.00 的教科书信号组合，规则正确路由 |
| 5 | ReAct 升级分支（#16，信号全正常的点踩） | ✅ 完整走通：Workflow 判低置信 → 升级 ReAct → 2 步探查确认"停止位信息与系统答案一致"（点踩存疑）→ 未产出 JSON 结论 → 按设计**不硬猜、携证据转人工**（engine=react，双引擎轨迹落 run json） |
| 6 | Web 路由全链路（test client） | ✅ `POST /api/admin/bad_cases/13/diagnose`：200 + R1 + 组件注入正常；**dry_run 实测不落库**（#13 保持 open/无 rc）；审计 `bad_case_diagnose` 完整落盘（含 root_cause/engine/dry_run 明细）；鉴权失败路径（错误密码 401）亦验证并审计 |
| — | 验收第 5 条（3 类种子 3/3） | ✅ 基于可召回内容构造的种子 **3/3 命中**（#10 R1 / #14 R5 / #15 R6） |

### live 验证中的两个真实发现

1. **诊断系统比种子假设更准**：最初用"定位方式"类 query 构造的 R5/R6 种子（#11/#12/#13）被系统判为 R1——追查发现"定位方式"内容只存在于**其他租户**的文档（学生证协议 V5.0），jm 租户视角确实召不回。系统对"当前检索状态"忠实，种子的 ground truth 假设错了。这本身验证了双视角检索设计的价值。
2. **知识库数据质量线索**：jm 协议文档的 top 召回里混有近乎空内容的页眉 chunk（"个人定位终端通讯协议"重复数十字）排在有效段落之前——是潜在的召回噪声源，建议后续在 ingest 侧过滤空 chunk（与 BadCase9 报告的检索层议题同族，非本方案范围）。

### 试用与清理

- 种子样本 #10~#16 与真实 case #9 均已在 `/admin/bad_cases` 可见（诊断字段带"（R5重造）"等标记便于识别）；确认后可正常流转状态，或直接删除测试行。
- CLI 复跑：`python -m agentworkflow --bc-id <N>`；`--dry-run` 不落库；`--json` 机器可读（注：stdout 混有节点进度日志，程序化取值建议读 `agentworkflow/runs/` 下最新 json）。

## 四、实现与方案的偏差说明

- 方案中「登记限流」实际落地为前缀规则内的专用阈值（`RATE_LIMIT_DIAGNOSE=6`），`/api/admin/` 前缀的通用限流本就自动覆盖，无需逐路由登记。
- 审计无需扩展动作枚举：`AuditLogger.log()` 的 action 为自由字符串，直接写 `bad_case_diagnose`（文档口径从 7 类变 8 类）。
- 存储层比方案多一个只读方法 `get_bad_case`（按主键取单条，诊断入口必需；`list_bad_cases` 全量拉取再过滤不可靠）。

---

# 附录：业界参考（Agent 两种形态与生产共识）

- [Anthropic《Building effective agents》范式解读（workflows vs agents）](https://zhuanlan.zhihu.com/p/1899127131549733019)
- [AI Agent 架构怎么选？ReAct / Plan-Execute / 多 Agent 协作的 5 个落地场景](https://www.kaiyan.net/blog/ai-agent-architecture-pattern/)
- [大模型 Agent 和 workflow 的区别在哪里？（知乎）](https://www.zhihu.com/question/1896707093580448857)
- [从 workflow 到 ReAct 提升 Agent 智能化水平（CSDN）](https://blog.csdn.net/m0_59164520/article/details/147721893)

---

**关联文档**：[Harness_BadCase_自进化体系改造方案.md](Harness_BadCase_自进化体系改造方案.md) · [README.md](../../README.md) · [LLM_GATEWAY_README.md](../guides/LLM_GATEWAY_README.md)
