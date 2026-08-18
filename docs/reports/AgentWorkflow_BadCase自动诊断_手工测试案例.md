# Bad Case 自动诊断 · 手工测试案例（一步一步走）

> 配套方案：`docs/reports/AgentWorkflow_BadCase自动诊断方案.md`（P0 + P1 已落地）
> 本文目的：**动手跑一遍完整闭环**，感受「点踩 → 自动诊断 → 根因回写 → 轨迹落盘」
> 每个案例独立可执行，建议按 T1 → T10 顺序走（前面案例造的数据后面会复用）。

---

## 〇、被测对象 30 秒速览

**链路**：用户点踩（`/api/feedback`，rating=-1）→ 写入 `bad_cases`（open）
→ 管理员触发诊断（Web 按钮 或 CLI）→ 诊断 DAG 七节点执行
→ 回写 `root_cause` / `diagnosis`，状态 `open → in_progress`
→ 轨迹落盘 `agentworkflow/runs/diag-<bc_id>-<时间戳>.json`

**诊断 DAG**：`prepare → rerun_retrieval(双视角复跑) → probe_docs(LLM 判文档相关性)
→ judge_answer(JudgeLLM 复判) → classify_root_cause(纯规则归因)
→ [低置信则 react_probe] → writeback`

**归因优先级**：R8 跨租户泄漏 → R1 召回丢失（含 R4 方向提示）→ R7 误拒
→ R5 幻觉 → R6 答非所问 → 全正常则升级 ReAct 探查

| R 码 | 含义 | 典型证据 | 置信度 |
|------|------|----------|--------|
| R8 | 跨租户泄漏 | 本租户视角复跑命中**其他租户**文档 | 高 |
| R1 | 召回侧丢失 | 零召回 / LLM 判文档答不了 | 高 |
| R1+R4提示 | 权限误杀方向 | 本租户零召回但**全库能搜到** | 中（只能提示） |
| R7 | 误拒 | 文档能答，当时答案却是拒答话术 | 中 |
| R5 | 生成幻觉 | faithfulness < 0.6 | 高 |
| R6 | 答非所问 | relevancy < 0.6 | 中 |
| null | 转人工 | 信号矛盾/探查未归因 | — |

**节奏预期**：CPU/本地模型下单次诊断 **30 秒 ~ 2 分钟**（含最多 6 次 LLM 调用），
Web 端限流 **6 次/分钟**，测试时别连点。

---

## 一、第 0 步：环境自检（全部通过再往下走）

| # | 检查项 | 验证方法 | 通过标准 |
|---|--------|----------|----------|
| 1 | MySQL 记忆层 | 见下方命令① | `available=True` |
| 2 | Milvus 向量库 | 启动 Web 服务时无报错 | 正常启动 |
| 3 | LLM 网关 | `python -m agentworkflow --bc-id 1 --dry-run` 能跑（不要求归因成功） | 不抛连接异常 |
| 4 | Web 服务 | `python rag_web_server.py`（默认 8080） | 控制台打印访问地址 |
| 5 | 管理员账号 | 浏览器登录 admin 账号 | 能打开 `/admin/bad_cases` |

```bash
# ① MySQL 记忆层自检（在仓库根目录执行）
python -c "from memory_store import MySQLMemoryStore; ms=MySQLMemoryStore(); print('available=', ms.available); print(ms.list_bad_cases(limit=3))"
```

```bash
# ② 看看现在库里有哪些 bad case 可用（记下几条 id 备用）
python -c "from memory_store import MySQLMemoryStore; [print(r['id'], r['status'], r['root_cause'], r['query'][:30]) for r in MySQLMemoryStore().list_bad_cases(limit=10)]"
```

> 下文所有 `<bc_id>` 请替换成你实际拿到的 id；`<port>` 默认 8080。

---

## 二、造数工具：可控 bad case 种子脚本

点踩产生的 case 是「自然样本」，query/answer 不可控；要稳定触发特定 R 码，
需要**手工控制 query / answer / 诊断文本里的租户**。把下面内容存为仓库根目录的
`tmp_seed_cases.py`（测完删除）：

```python
# -*- coding: utf-8 -*-
"""手工测试用：批量插入可控 bad case，返回各自 id。用法 python tmp_seed_cases.py"""
from memory_store import MySQLMemoryStore

ms = MySQLMemoryStore()

# ⚠ 先改成你环境里的真实情况：
REAL_TENANT = "jm"            # ← 改成 knowledge/ 下真实存在的租户目录名
COVERED_QUERY = "公司报销流程是什么"   # ← 改成一条你的知识库【能答好】的问题

cases = [
    # T4-R1：问库里压根没有的内容
    dict(query="2027年公司春游安排去了哪里", answer="抱歉，我不太清楚。",
         diagnosis="用户点踩（tenant=%s），待 triage。" % REAL_TENANT),
    # T5-R4方向：诊断文本里给一个【不存在的租户】→ 本租户视角必零召回
    dict(query=COVERED_QUERY, answer="这个我不了解。",
         diagnosis="用户点踩（tenant=tenant_not_exist_404），待 triage。"),
    # T6-R7：文档明明能答，当时答案是拒答话术（误拒）
    dict(query=COVERED_QUERY,
         answer="抱歉，知识库中没有找到与您问题相关的信息，无法回答该问题。",
         diagnosis="用户点踩（tenant=%s），待 triage。" % REAL_TENANT),
    # T7-R5：文档能答，答案是编造内容（幻觉）
    dict(query=COVERED_QUERY,
         answer="根据规定，报销额度为每人每月 8888 元，需在 3 个工作日内提交纸质单据，由董事会直接审批。",
         diagnosis="用户点踩（tenant=%s），待 triage。" % REAL_TENANT),
    # T8-升级ReAct：一条健康的问答被「误点踩」
    dict(query=COVERED_QUERY,
         answer="报销流程为：提交申请单 → 主管审批 → 财务复核 → 打款，一般在 5 个工作日内完成。",
         diagnosis="用户点踩（tenant=%s），待 triage。" % REAL_TENANT),
]

for c in cases:
    cid = ms.add_bad_case(query=c["query"], source="feedback", answer=c["answer"],
                          diagnosis=c["diagnosis"], status="open")
    print(f"bad_case id={cid}  query={c['query'][:24]}")
```

```bash
python tmp_seed_cases.py    # 记下输出里的 5 个 id，后面分别叫 T4~T8 的 bc_id
```

> 原理：`prepare` 节点用正则 `tenant=xxx` 从 diagnosis 文本里解析租户（`rules.parse_tenant`），
> 所以改 diagnosis 就能控制「以哪个租户视角复跑检索」。

---

## 三、测试案例

### T1 冒烟 · CLI dry-run（不动库，先感受输出形态）

**【生产场景】** 诊断功能上线前，运维想先在不污染数据的情况下验证链路通不通。

1. 挑一条第 0 步查到的已有 case（或 T4~T8 任一 id）：

```bash
python -m agentworkflow --bc-id <bc_id> --dry-run
```

2. **预期输出**（形态，R 码因 case 而异）：

```
============================================================
  Bad Case #<bc_id> 自动诊断（dry-run，未回写）
============================================================
  根因     : R1 召回侧完全丢失
  引擎     : workflow   置信度: 高
  回写     : 未写入
  轨迹     : ...agentworkflow\runs\diag-<bc_id>-<时间戳>.json（workflow 10 步 / react 0 步）
------------------------------------------------------------
【自动诊断·workflow】R1 召回侧完全丢失
证据：…
```

3. 再跑一次加 `--json`，确认返回结构含 `ok/bc_id/root_cause/engine/confidence/diagnosis/written/run_file/steps` 六类字段。

4. **验证不落库**：

```bash
python -c "from memory_store import MySQLMemoryStore; r=MySQLMemoryStore().get_bad_case(<bc_id>); print(r['status'], r['root_cause'])"
```

✅ 通过标准：仍是 `open` / 原 root_cause，未被修改。

---

### T2 错误路径 · 诊断不存在的 case

**【生产场景】** 管理页数据被人删了，诊断入口要优雅报错而不是 500。

```bash
python -m agentworkflow --bc-id 999999
```

✅ 通过标准：输出 `✗ 诊断失败：bad case 999999 不存在`，退出码 1，不产生 runs 文件。

---

### T3 全链路 · 聊天页点踩 → 管理页诊断 → 回写（核心闭环）

**【生产场景】** 客服同事反馈「机器人答错了」，你在管理页一键定位根因。

1. 浏览器打开 `http://localhost:<port>`，用普通用户登录；
2. 问一个知识库**答得一般**的问题（随意），在回答卡片点 👎，填写问题描述并提交；
   - 提示条出现「已记入 Bad Case（open）」；
3. 用第 0 步命令② 找到这条新 case 的 id（query 是你刚问的）；
4. 打开 `http://localhost:<port>/admin/bad_cases`（admin 账号），找到该条，点 **诊断** 按钮；
   - 按钮进入 loading 态「⏳ 诊断中（约 0.5~2 分钟）…」，**等它跑完**；
5. **验证三件事**：

```bash
# ① 状态被推进、根因被写入
python -c "from memory_store import MySQLMemoryStore; r=MySQLMemoryStore().get_bad_case(<bc_id>); print(r['status']); print(r['root_cause']); print(r['diagnosis'][:300])"
```

✅ 通过标准：
- `status` = `in_progress`（仅当原来是 open）；
- `root_cause` 是 R 码或 None；
- `diagnosis` 以 `【自动诊断·workflow】`（或 `·react`）开头，含证据、复跑召回文件名、建议、置信度四段。

```bash
# ② 轨迹文件生成（目录里找最新的 diag-<bc_id>-*.json）
ls -t agentworkflow/runs | head -3
```

✅ 通过标准：文件存在，用编辑器打开能看到 `steps[]`（每步 engine/step_type/node/latency_ms）和 `payload.triage`。

```sql
-- ③ 审计留痕（有条件连 MySQL 时）
SELECT * FROM audit_logs WHERE action='bad_case_diagnose' ORDER BY id DESC LIMIT 1;
```

✅ 通过标准：一条 `result=success`，detail 里带 root_cause/engine/触发者。

---

### T4 R1 · 召回侧丢失（库里没有的内容）

**【生产场景】** 新产品资料还没入库，用户已经在问，点踩进来一片「召回空白」。

用种子脚本插入的第 1 条 case：

```bash
python -m agentworkflow --bc-id <T4的id>
```

✅ 通过标准：
- `根因: R1 召回侧完全丢失`，置信度**高**；
- 证据行体现「租户视角命中 N 条、相关性判定=False（LLM 探查：…）」；
- `written=true`，状态 open → in_progress。

> 若你们库里真有该内容，会归因成别的码——换一个确定没有的话题重测即可。

---

### T5 R4 方向 · 本租户零召回但全库命中（权限误杀提示）

**【生产场景】** 用户抱怨「问了自己租户的文档却搜不到」，怀疑权限过滤把内容挡了。
点踩样本没有当时用户角色，系统**只能提示方向、不能定论**——这是刻意设计的诚实边界。

用种子脚本第 2 条（diagnosis 里租户是 `tenant_not_exist_404`，本租户视角必然零召回，
而全库视角能命中真实文档）：

```bash
python -m agentworkflow --bc-id <T5的id> --dry-run
```

✅ 通过标准：
- 归因 **R1**，置信度**中**；
- 证据行明确写出「本租户视角零召回，但全库可检索到 N 条相关内容，**疑似租户/权限过滤误杀（R4 方向），需人工确认**」；
- 没有把 R4 写成定论（root_cause 仍是 R1）。

---

### T6 R7 · 误拒（文档能答，答案却是拒答）

**【生产场景】** 用户怒点踩：「明明手册里写着，机器人却说没有！」——拒答阈值过严。

用种子脚本第 3 条（answer 是标准拒答话术，query 是库内能答的问题）：

```bash
python -m agentworkflow --bc-id <T6的id>
```

✅ 通过标准：
- `根因: R7 拒答错误`，置信度**中**；
- 证据行：「检索内容足以回答，但当时答案表现为拒答（误拒）」；
- 拒答判定走的是零 LLM 的语义正则（`JudgeLLM._grade_refusal`），所以这一条**结果稳定**。

---

### T7 R5/R6 · 生成类根因（答案编造 / 答非所问）

**【生产场景】** 最危险的一类：检索全对，模型张嘴就来，用户拿着编造的「8888 元额度」去财务报销。

用种子脚本第 4 条（answer 是编造内容）：

```bash
python -m agentworkflow --bc-id <T7的id>
```

✅ 通过标准：
- 归因 **R5 生成幻觉**（faithfulness < 0.6），诊断文本里能看到 judge 复判行
  `judge 复判 faithfulness=0.xx, relevancy=0.xx`；
- 若你的 judge 打分偏松导致两条分都 ≥0.6，会落到 T8 的「升级 ReAct」分支——**这也算通过**，
  观察点变为：证据行说明「信号与失败标记矛盾」。把编造内容改得更离谱一点可稳定触发 R5。

---

### T8 低置信升级 · ReAct 开放探查（Workflow → ReAct 形态切换）

**【生产场景】** 用户「心情不好乱点踩」或失败原因在流水线信号之外——所有自动信号都正常，
这时不该硬给结论，而是放 ReAct Agent 去换关键词、核对文档，查完仍不确定就转人工。

用种子脚本第 5 条（健康答案被误点踩）：

```bash
python -m agentworkflow --bc-id <T8的id> --json
```

✅ 通过标准：
- `"engine": "react"`（说明走了升级分支）；
- `"steps": {"workflow": N, "react": M}` 中 **M > 0**；
- 打开对应 runs 文件，`steps[]` 里能看到 `engine="react"` 的 think/act/observe 记录
  （act 的工具应是 doc_search / calculator）；
- 无论 ReAct 是否给出 R 码，结论都**附证据**；未归因时明确写「转人工」，不硬猜。

---

### T9 状态机保护 · 已处理的 case 不被降级

**【生产场景】** 运维已经把某条 case 标成 resolved，自动诊断绝不能把它拽回处理中。

1. 把 T4 那条 case 人工改成 resolved：

```bash
python -c "from memory_store import MySQLMemoryStore; print(MySQLMemoryStore().update_bad_case_status(<T4的id>, status='resolved', resolved_by='tester'))"
```

2. 再诊断一次：

```bash
python -m agentworkflow --bc-id <T4的id> --dry-run
```

3. 查状态：

```bash
python -c "from memory_store import MySQLMemoryStore; r=MySQLMemoryStore().get_bad_case(<T4的id>); print(r['status'])"
```

✅ 通过标准：仍是 `resolved`（dry-run 本就不写库；即便去掉 --dry-run，代码也只推进 open → in_progress，不动其他状态）。

---

### T10 Web 边界 · 权限与限流

**【生产场景】** 诊断要烧 LLM Token，必须只有管理员能点、且不能被连点刷爆。

1. **权限**：用普通 user 角色的 token 调接口：

```bash
curl -s -X POST http://localhost:<port>/api/admin/bad_cases/<bc_id>/diagnose \
  -H "Content-Type: application/json" -H "Authorization: Bearer <普通用户token>" \
  -d "{}"
```

✅ 通过标准：HTTP 403 `{"ok":false,"error":"需要管理员权限"}`。

2. **限流**：用 admin token 在 1 分钟内**连续**调第 7 次（可用循环）：

```bash
for i in $(seq 1 7); do curl -s -o /dev/null -w "%{http_code}\n" -X POST \
  http://localhost:<port>/api/admin/bad_cases/<bc_id>/diagnose \
  -H "Content-Type: application/json" -H "Authorization: Bearer <admin token>" -d '{"dry_run":true}'; done
```

✅ 通过标准：前 6 次返回 200，第 7 次返回 429（限流阈值 RATE_LIMIT_DIAGNOSE=6/min）。
注意每次 dry-run 也要 30s~2min，嫌慢可把并发开大或改小后端阈值临时测。

---

## 四、收尾清理

```bash
# 1. 删除种子脚本
rm tmp_seed_cases.py

# 2. 清掉本次测试的 bad case（把 id 列表替换进去；确认别删到真实线上样本）
python -c "
from memory_store import MySQLMemoryStore
ms = MySQLMemoryStore()
for i in [<T4的id>, <T5的id>, <T6的id>, <T7的id>, <T8的id>]:
    print(i, ms.update_bad_case_status(i, status='closed', resolved_by='tester'))"

# 3. runs 轨迹文件可保留作复查证据；确认不要可整目录清空
# rm agentworkflow/runs/diag-*.json
```

---

## 五、诊断本身失败了怎么看

| 现象 | 排查点 |
|------|--------|
| CLI 直接 `✗ 诊断失败：记忆层不可用` | MySQL 没起 / 连接配置错（第 0 步检查①） |
| `复跑检索失败：MilvusException…` | 向量库挂了；runs 文件里 rerun_retrieval 节点 output 有异常串 |
| 归因一直是 null + 「judge 复判不可用」 | evalkit JudgeLLM 初始化失败（网关/eval 链配置），看控制台报错 |
| probe_docs 步 `docs_relevant: null` | evalgrade 链 LLM 输出没解析成 JSON，规则层按「无法判定」处理——属设计内降级 |
| Web 点诊断 500 | 查 audit_logs 里 `action=bad_case_diagnose, result=failure` 的 detail |

所有「卡在哪一步」的问题，先开最新的 `agentworkflow/runs/diag-*.json`：
`steps[]` 按时间序记录了每个节点的进入/退出、输出摘要和 latency_ms，
`payload.triage` 是最终归因、`payload.writeback` 是回写结果。

---

## 六、覆盖度对照（本手册 vs 归因规则）

| 归因分支 | 本手册案例 | 方式 |
|----------|-----------|------|
| 复跑彻底失败 → null | （未手工覆盖） | 断 Milvus 复现，或看单测 |
| R8 泄漏 | （未手工覆盖） | 见下方说明 |
| R1 零召回/不相关 | T4 | 真实复跑 |
| R1 + R4 方向提示 | T5 | 假租户制造零召回 |
| R7 误拒 | T6 | 拒答话术 answer |
| R5 幻觉 | T7 | 编造 answer + judge |
| R6 答非所问 | T7 变体（answer 换成离题但流畅的内容） | judge |
| 全正常 → ReAct | T8 | 健康 answer |

**R8 为什么不手工造**：它需要租户过滤 expr 真实失效（越权缺陷）才能触发，
人为制造等于往生产索引里掺别的租户文档——污染数据且不可控。该分支由
`tests/test_agentworkflow.py` 的表驱动单测覆盖（leak_hits 非空 → R8），
线上真触发时它会以最高优先级、critical 级别报出。

```bash
# 单测验证规则层（零外部依赖，秒级）
python -m pytest tests/test_agentworkflow.py -q
```
