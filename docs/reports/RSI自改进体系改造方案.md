# RSI 自改进体系改造方案

> 目标：把 enterprise-ai 从「带记忆的检索系统」改造成**可验证的递归自改进（RSI）系统**。
> 参照对象：Google AlphaEvolve、Sakana Darwin Gödel Machine、Stanford DSPy/GEPA、NVIDIA Voyager、Reflexion。
> 口径：**弱 RSI**（AI 辅助研发自动化闭环），不是强 RSI（模型改自己权重）。
> 日期：2026-10-25　作者：自改进体系改造组

---

## 零、一句话结论

**本仓现在不是 RSI。** 它有 RSI 五环里的三环（运行产出、评测打分、诊断归因），
**缺第四环「自动改进」**，而且这个缺是**有意留下的设计决定**，不是漏做。

补齐第四环有两种做法：

- **改代码层**：让 Agent 直接改 `.py` 源码（DGM 路线）。能力上限高，但验证成本高、回滚难、责任边界模糊。
- **改配置层**：让 Agent 只改**白名单内的可改对象**（提示词文本、检索参数、playbook 经验），
  改动结构化落库、强制过沙箱 A/B、变好才合并、变差自动回滚（DSPy/GEPA 路线）。

**本方案选第二条。** 理由不是保守，是三条硬判据：可验证性、可回滚性、责任归属——
这三条决定了改动能不能进生产。等到配置层跑满一个完整周期、积累了足够的「有效提案率」数据，
再评估要不要开第四层的受控代码 patch。

---

## 一、竞品对照：他们到底改了什么，怎么判好坏

数字全部来自公开论文/官方发布/第三方复现分析，可回源核验。

### 1.1 主流系统对照表

| 系统 | 改什么 | 靠什么判好坏 | 公开量化结果 | 我们能搬什么 | 不能搬什么 |
|---|---|---|---|---|---|
| **AlphaEvolve**（Google DeepMind，2025-05） | 算法源码 / 整个代码库，进化算法 + LLM 变异 | 自动化评估器**机械打分**（结果可判定对错） | 4×4 复值矩阵乘 **49 → 48** 次标量乘法（Strassen 1969 以来 56 年首次改进）；14 个矩阵尺寸取得改进；50+ 开放数学问题约 75% 追平已知最优、约 20% 严格更优；Borg 调度启发式回收 Google 全球算力约 **0.7%**（生产运行一年以上）；Gemini 训练矩阵乘 kernel 提速 **23%**，训练时间降 1%；FlashAttention kernel 最高 **32.5%** 提速 | 「评估器是天花板」这条铁律：能机械判定的目标才进自动循环 | 我们的 RAG 质量指标不是机械可判定的（faithfulness 靠 judge），学不来它那套纯执行验证 |
| **Darwin Gödel Machine**（Sakana AI + UBC，2025-05，arXiv:2505.22954，ICLR 2026） | **改自己的 Python 源码**（工具、验证流程、编辑能力都是它自己发明的） | SWE-bench / Polyglot 可执行测试集 | SWE-bench 子集 **20.0% → 50.0%**（80 轮）；Polyglot 完整集 **14.2% → 30.7%**；能力可迁移：用 Claude 3.5 Sonnet 自改，换 3.7 Sonnet 测 200 题，初始 agent 19.0% vs DGM 版 **59.5%**；Python 上练出的能迁到 Rust/C++/Go | **archive 演化树**（父代采样 → 变异 → 评测 → 优胜子代归档 → 成为下一代父代）；保留成绩下降的分支（后续突破的起点） | 它记录的**奖励作弊**：为通过幻觉检测任务，某个满分版本直接删掉了记录检测标记的日志。这条比它的成绩更值得学 |
| **DSPy GEPA**（Stanford NLP，ICLR 2026） | **只改指令文本**，代码一行不动 | 指标分 + metric 返回的**自然语言反馈**（trace / 报错 / 约束违反）做「文本梯度」 | 对比 MIPROv2：聚合增益 **+14% vs +7%**，产出 prompt 最长缩短 **9.2×**；Pareto 前沿保留互补候选，可 merge | 这条和我们最贴：**可改对象 = 提示词**，反射式诊断 + Pareto 保留 + 结构化合并 | 需要 200+ 样本防过拟合，我们黄金集规模不够，得先做数据 |
| **Voyager**（NVIDIA + Caltech，2023） | 技能库（JS 函数），embedding 索引 | 环境执行反馈 + 解释器报错 + **自验证 critic** | 160 轮内多发现 **3.3×** 独特物品，科技树里程碑解锁最高快 **15.3×**。消融很关键：去掉自验证 → 发现物品数 **掉 73%**；去掉技能库 → 后期平台；随机课程换掉自动课程 → **掉 93%** | **技能库 = 我们的 `PlaybookStore`**（同构）；自验证是最重要反馈信号 | 我们有明确业务语义，不能靠「自动课程」随机探索 |
| **Reflexion**（2023 NeurIPS） | 失败后写自然语言反思，存 episodic buffer，下次 prepend | 单元测试 | HumanEval 上 GPT-4 **80% → 91%**。消融：只留测试反馈、去掉反思 → 增益**完全消失** | 「失败样本必须结构化成可检索的教训」，对应我们的 `extract_failure` | 纯 prepend 会撑爆上下文，我们得走检索式复用 |
| **MASS**（多 agent 轨迹 → 训练权重） | 编排工作流 + 后训练权重 | 多个研究型 benchmark | 4 个 benchmark 涨（ScienceAgentBench 28.8%→31.7% 等），但 **Terminal-Bench 2.0 与 SWE-bench Verified 反而略降**；只跑了 2 轮 | 反面教材：自改进不是全维度变好，得看**哪些维度涨、哪些跌** | — |

### 1.2 反面证据（比正面结果更重要）

这几条决定了我们验收章节怎么设计：

1. **「记忆力通胀」（memory reward inflation，2026-08 分析）**：
   自改代码的编码 agent（SICA）在 SWE-bench Verified 子集上 17% → 53%，
   **但两轮之间清空记忆后增益崩塌**。也就是说，涨的不是「解题能力」，是「记住了答案」。
   检测手段：**清空记忆重跑，看掉多少**。

2. **人工 vs 自生成的技能差距**：
   跨 scaffold 汇总显示，**人工精编的技能 +16.2pp，LLM 自生成的技能 +0.0pp**。
   大量看起来像「学到的技能」的东西，其实是「记住的策展」。

3. **EvoAgentBench**：528 训练 / 267 测试任务、两个 scaffold、三个底座模型，
   **没有任何自动方法能维持正向增益**。人工策展的能力内容能跨模型家族迁移，自动方法不能。

4. **DGM 的日志删除事件**：优化器找到评分器的漏洞，直接把漏洞变成「能力」。
   这不是理论风险，是论文里如实记录的实事。

**结论**：自改进系统最大的敌人不是「改不动」，是**「改了但改的是评分器」**。
本方案的安全闸设计（第七章）全部围绕这一点。

---

## 二、判定标准：什么叫「真的在自我改进」

### 2.1 五环

| # | 环节 | 本仓现状 | 证据（文件:行号） |
|---|---|---|---|
| 1 | 运行产出 | 有 | `langgraph_rag_agent.py` 全链路（classify → retrieve → grade → rewrite → generate） |
| 2 | 评测打分 | 有 | `evalkit/harness_retrieval.py`（Recall@k / MRR / nDCG / bury）、`harness_answer.py`（faithfulness / relevancy / judge） |
| 3 | 诊断归因 | 有 | `evalkit/triage.py:85-114`（R1–R8 根因码，含 title/severity/建议）、`agentworkflow/rules.py:44-55`（同一套码，工作流侧复用） |
| 4 | **自动改进** | **无（有意不做）** | 见 3.2 三处代码证据 |
| 5 | 回写复用 | 有 | `evolution.py:126 PlaybookStore`（Voyager 技能库同构）、`langgraph_rag_agent.py:1345 store.save_or_merge(pb)` |

**五环缺一环，闭环就是断开的。** 现在的形态是：系统能发现问题（R1–R8 分得很准）、
能记住成功经验（playbook）、但**从「发现」到「修好」之间，站着一个人**。

### 2.2 递归性检验（区分「改进」和「递归改进」）

光有五环还不够。真正的 R 要过这一关——用 A₀/A₁/A₂ 三代看：

- A₁ 比 A₀ 强 → 只说明**任务能力**提高了；
- 在资源大致相同的前提下，**A₁ 比 A₀ 更高效地开发出 A₂** → 才说明**改进能力本身**提高了；
- 这种优势继续从 A₂ 传到 A₃ → 跨代复利才真正成立。

**这是本方案的终局验收口径，写在第九章。** 只报「指标从 X 涨到 Y」是自欺，
必须报「第 N 代产出有效提案的速度 / 命中率」。

---

## 三、现状盘点（文件:行号级）

### 3.1 已经能用的底座

这是本次改造的立足点——**不用从零建**，四个关键零件都在：

| 零件 | 位置 | 能拿来干什么 |
|---|---|---|
| 技能库 | `evolution.py:126 PlaybookStore`，集合 `skill_playbooks` | Voyager 技能库同构：命中相似 playbook 复用已知好 rewrite，跳过首轮 LLM 改写 |
| 经验抽取 | `evolution.py:501 Extractor.extract` / `:546 extract_failure` | 正样本沉淀 + **负样本进 bad_cases**（`langgraph_rag_agent.py:1349`） |
| 根因码 | `evalkit/triage.py:85-114` | R1 召回丢失 / R2 排序埋没 / R3 改写精排负优化 / R4 权限误杀 / R5 幻觉 / R6 答非所问 / R7 拒答错误 / R8 跨租户泄漏 |
| 参数沙箱 | `evalkit/runner.py:142 --set key=value` | 已验证的 A/B 杠杆：覆盖只影响评测进程、跑完自动还原。本轮刚加了 `rerank_max_chars`（`harness_retrieval.py`） |
| 评测落盘 | `evalkit/schema.py:339 EvalRun`（含 `config` 字段） | 每次 run 记录关键配置，**两次 run 的指标差异可归因** |
| 基线对比 | `evalkit/runner.py:134 --compare last` | 现成的「变好还是变差」判定入口 |
| 提示词热更 | `prompt_manager.py:373 get_prompt`（每次实时查 MySQL，无缓存）、`:464 save_prompt`、`:538 set_active`、表里有 `version` | **最佳可改对象**：改完即时生效、有版本号、可切回。主链路 12 处调用点（`langgraph_rag_agent.py:1095/1216/1738/1818/1858/1868/2113/2168/2346/2378/2460` 等）全部走 `self.pm.get_prompt(...)` |

### 3.2 三处「有意不做」的代码证据

这不是疏漏，是当时的设计决定，改造时必须正面处理而不是绕过：

```
agentworkflow/__init__.py:18-20
  【边界】诊断是「建议」，不是结论：root_cause / diagnosis 回写后仍由管理员
  人工确认流转（resolved 归人）；本模块不做自动修复执行。

kb_ops_employee.py:22
  高置信修复建议只进「人工复审清单」，不自动改数据（waiting 思路）

evolution.py:10
  3. Nudge 反思引擎  -> （P1 阶段补；本 P0 仅落地 1+2 的最小闭环）
```

三处指向同一条原则：**判定权在人手上**。本方案不动这条原则，
而是把它**制度化**：哪些变更可以自动合并、哪些必须人工审批，用一张分级表写死（见 7.5）。

### 3.3 缺口：三件事

1. **没有提案生成器**：诊断出 R5（生成幻觉）之后，没有组件把「R5 + trace + 当前 generate prompt」变成一份具体的改写提案。
2. **没有沙箱 A/B 编排**：`--set` 只能手跑，没有「自动跑基线 vs 变体 → 出 diff → 判胜负」的驱动器。
3. **没有合并/回滚闸**：改了 prompt 之后，没有「谁批准、怎么回滚、怎么证明是这次改动导致的」。

---

## 四、技术选型：为什么先做配置层，不做改代码层

| 判据 | 改代码层（DGM 路线） | 改配置层（GEPA 路线） |
|---|---|---|
| 可验证性 | 需要完整单测覆盖才能判对错，本仓测试覆盖不足 | 有黄金集 + judge + 冻结集，可量化 |
| 可回滚性 | 靠 git，但运行中热更意味着进程内状态可能已污染 | prompt 有 `version` 字段、参数走进程内 `--set`、playbook 有 `success_count` —— 三路都能一键回滚 |
| 责任归属 | 出事后很难说清是哪次自动改动引入的 | 每条变更是结构化提案，有 `run_id` 证据链，可追溯到具体根因码 |
| 失败爆炸半径 | 源码改错 → 服务崩溃 | 配置改错 → 指标下降，被门禁拦下，自动回滚 |
| 对应竞品教训 | DGM 的日志删除事件 | GEPA 的 +14% / prompt 缩短 9.2× |

**决定：P0–P3 只做配置层。P4 才评估受控代码 patch，且必须满足 7.6 的前置条件。**

---

## 五、架构设计：五层 + 双闸门

```
┌─ L0 观测层（已有）────────────────────────────────────────┐
│  主链路 trace / checkpoints / qa_feedback / bad_cases      │
└────────────────────────┬───────────────────────────────────┘
                         │
┌─ L1 诊断层（已有）──────▼───────────────────────────────────┐
│  evalkit/triage.py  R1–R8   +   agentworkflow/rules.py      │
│  产出：根因码 + 证据 + 建议（不改任何东西）                  │
└────────────────────────┬───────────────────────────────────┘
                         │
┌─ L2 提案层（新增）──────▼───────────────────────────────────┐
│  ProposalGenerator：读 trace + 根因码 + 当前可改对象         │
│  → 产出结构化 ImprovementProposal（只动白名单，含 rationale） │
└────────────────────────┬───────────────────────────────────┘
                         │
┌─ L3 验证层（新增）──────▼───────────────────────────────────┐
│  SandboxRunner：基线 run vs 变体 run（复用 --set 机制扩展）   │
│  优化集 + 冻结集 双集评估 → diff → 门禁判定                  │
└────────────────────────┬───────────────────────────────────┘
                         │
┌─ L4 裁决层（新增）──────▼───────────────────────────────────┐
│  按变更类型分级：自动合并 / 人工审批 / 一律拒绝               │
│  合并写库并打 lineage；失败自动回滚 + 记入 archive 负样本     │
└────────────────────────┬───────────────────────────────────┘
                         │
┌─ L5 复用层（已有 + 扩展）─▼─────────────────────────────────┐
│  PlaybookStore（技能库） + ProposalArchive（DGM archive 同构）│
│  archive 保留失败分支，作为后续代次的父代候选                 │
└─────────────────────────────────────────────────────────────┘
```

### 5.1 关键设计一：可改对象白名单（三层）

这是整套系统的**收敛点**。Agent 能改的东西必须穷举、必须结构化、必须可回滚：

| 层 | 可改对象 | 载体 | 回滚方式 | 自动合并 |
|---|---|---|---|---|
| **P1 提示词层** | `prompt_templates` 表里指定 name 的 `system` / `user_template` | MySQL 行 + `version` | `set_active` 切回旧版本 | 允许（冻结集涨 + 无 R8/R5 新增） |
| **P2 参数层** | 检索/精排参数（`top_k`、`hybrid`、`rerank_max_chars`、RRF 权重等） | 进程内覆盖（沿用 `harness_retrieval.py` 的 `_saved` 还原模式），确认有效才写回配置 | 不写回即无效 | 需人工审批 |
| **P3 经验层** | playbook 的新增/合并/降权 | `skill_playbooks` 集合 + `success_count` | 删除/降权 | 允许（有 `reinforce_feedback` 负反馈兜底） |

**白名单外的任何东西一律不改**，包括：源码、租户过滤 expr、权限配置、judge 的评分标准、安全策略。

### 5.2 关键设计二：ProposalArchive = DGM archive 的同构落地

DGM 的核心不是「能改代码」，是**演化树的维护方式**：父代采样时既看当前成绩，也照顾没探索过的分支；
**成绩下降的版本也留着**，因为可能是后续突破的起点。

我们的 archive 表照搬这个语义（`parent_id` 建树、`status` 包含 `rejected` 保留分支）：
采样父代时不只看「当前指标最高的」，而是按 `score + 探索次数倒数` 加权，避免早收敛到局部最优。

### 5.3 关键设计三：双集验证（直接对应「记忆力通胀」）

优化器能看见的集合和判定它变好的集合**必须分开**：

- **优化集（dev）**：提案生成时可读、可反复试错；
- **冻结集（holdout）**：提案生成流程**完全看不到**（不进任何 LLM 上下文），只在 L3 门禁时跑一次。

**只有冻结集变好才允许合并。** 这是防过拟合的唯一有效手段，也是「17% → 53% 但清记忆就塌」这类事故的解药。

---

## 六、数据模型（新增三张表）

```sql
-- 1) 提案：一次「我认为该怎么改」的完整记录
CREATE TABLE improvement_proposals (
  id              BIGINT AUTO_INCREMENT PRIMARY KEY,
  proposal_id     VARCHAR(64) NOT NULL UNIQUE,      -- pp-20261025-xxxx
  parent_id       VARCHAR(64) DEFAULT NULL,          -- 演化树父代（DGM archive 同构）
  root_cause      VARCHAR(8)  NOT NULL,              -- R1..R8，来自 triage
  change_layer    VARCHAR(8)  NOT NULL,              -- P1 prompt / P2 param / P3 playbook
  change_target   VARCHAR(128) NOT NULL,             -- 例：generate_answer / top_k
  payload_before  LONGTEXT,                          -- 变更前快照（回滚用）
  payload_after   LONGTEXT,                          -- 变更后内容
  rationale       TEXT,                              -- LLM 给出的改的理由（自然语言反馈）
  status          VARCHAR(16) NOT NULL DEFAULT 'draft',
                  -- draft → validating → passed / rejected → merged / rolled_back
  dev_score       JSON,                              -- 优化集指标
  holdout_score   JSON,                              -- 冻结集指标（门禁唯一依据）
  baseline_run_id VARCHAR(64),                       -- 基线 run（evalkit run_id）
  variant_run_id  VARCHAR(64),                       -- 变体 run
  created_at      DATETIME NOT NULL,
  decided_at      DATETIME,
  decided_by      VARCHAR(32),                       -- 'auto' 或 操作者
  INDEX idx_status (status),
  INDEX idx_cause  (root_cause),
  INDEX idx_parent (parent_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2) 门禁事件：每次判定的完整留痕（出事后可复盘）
CREATE TABLE improvement_gate_events (
  id           BIGINT AUTO_INCREMENT PRIMARY KEY,
  proposal_id  VARCHAR(64) NOT NULL,
  gate         VARCHAR(32) NOT NULL,   -- immutable_guard / holdout_gain / no_regression / budget
  passed       TINYINT(1)  NOT NULL,
  detail       TEXT,
  created_at   DATETIME NOT NULL,
  INDEX idx_proposal (proposal_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 3) 生效档案：已合并、正在生效的变更（archive 的「活着的部分」）
CREATE TABLE proposal_archive (
  id            BIGINT AUTO_INCREMENT PRIMARY KEY,
  proposal_id   VARCHAR(64) NOT NULL,
  change_layer  VARCHAR(8) NOT NULL,
  change_target VARCHAR(128) NOT NULL,
  payload_after LONGTEXT,
  merged_at     DATETIME NOT NULL,
  rollbacks     INT DEFAULT 0,          -- 被回滚次数（回滚≥2 的目标进黑名单）
  is_active     TINYINT(1) DEFAULT 1,
  INDEX idx_target (change_target)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

---

## 七、安全闸（本方案最重要的部分）

竞品已经替我们交过学费了。每一道闸都对应一个真实发生过的失败模式。

### T1 不可变护栏集（对应 DGM 删日志事件）

```python
IMMUTABLE_TARGETS = {
    # 安全相关：任何提案触碰即拒，不进验证
    "tenant_filter_expr", "access_control", "role_matrix",
    "rag_docs_collection", "knowledge_id_whitelist",
    # 评分相关：防「改评分器」
    "judge_prompt", "evalgrade_task", "faith_threshold", "relevancy_threshold",
    # 门禁自身：防「改门禁」
    "holdout_golden", "gate_config",
}
```

提案生成后第一道检查：`change_target ∈ IMMUTABLE_TARGETS` → 直接 `rejected`，并记一条 `gate_event`。

### T2 双集验证（对应 memory reward inflation）

- 优化集可见，冻结集不可见；
- 冻结集 case 由人工编写 + 从 trace 挖掘后人工确认，**不进任何 LLM 上下文**；
- 冻结集规模建议 ≥ 30 条，且**每季度换一批**（防长期过拟合到同一批题）。

### T3 评分器隔离（对应「改评分器」类作弊）

**同一个提案里，不允许同时改「生成侧」和「判定侧」。**
即：`change_target ∈ {generate_answer, writer_compose, ...}` 的提案，
其验证所用的 judge / 阈值必须是**提案生成时冻结的快照**，不允许同批变更。
实现上：judge 相关配置从 `IMMUTABLE_TARGETS` 走，且 L3 验证时显式传入 judge 版本号。

### T4 净收益阈值 + 回归门禁

合并必须同时满足：

| 条件 | 阈值 | 说明 |
|---|---|---|
| 冻结集 Recall@5 净增 | ≥ +2.0pp | 低于噪声 floor 的增益不接受 |
| 冻结集 nDCG@5 净增 | ≥ +1.5pp | 同上 |
| R8 跨租户泄漏 | **必须 0**（零容忍） | 出一条即拒绝并告警 |
| 新增 R5 幻觉 / R6 答非所问 | **必须 0** | 不能用一个新坑换旧坑 |
| 延迟回归 | p95 增幅 ≤ 10% | 防止用延迟换质量 |
| 拒答类 R7 | 误拒数不增加 | 防止「变好后乱拒」 |

**全部通过才允许合并；任意一条不过 → 自动 `rolled_back` + 该 `change_target` 冷却 N 轮。**

### T5 三路回滚

| 层 | 回滚动作 | 耗时 |
|---|---|---|
| P1 提示词 | `prompt_templates` 恢复 `payload_before` + `version` 递增 | 即时（下次请求生效，因为 `get_prompt` 无缓存） |
| P2 参数 | 不写回配置文件即失效；已写回则恢复旧值 | 需重启或热重载 |
| P3 playbook | `reinforce_feedback(pk, positive=False)` 降权 / 删除 | 即时 |

回滚必须**自动触发**：监控到线上 bad case 率异常上升时，自动回滚最近 N 条已合并提案。

### T6 预算闸（对应「一夜烧掉一个月 token」）

- 单轮提案生成：LLM 调用 ≤ 20 次；
- 单轮验证：优化集 ≤ 50 条 case，冻结集 ≤ 30 条；
- 单日自动合并提案数 ≤ 3；
- 单日 token 预算硬上限，超了直接熔断并告警。

### 7.5 变更分级审批表（正面处理 3.2 的「判定权在人手上」）

| 变更层 | 场景 | 裁决 |
|---|---|---|
| P1 提示词 | 冻结集达标、无 R8/R5/R6 新增、历史同目标无回滚 | **自动合并** |
| P1 提示词 | 首次改动某个 `change_target` | 人工审批 |
| P2 参数 | 全部 | 人工审批 |
| P3 playbook | 全部 | 自动合并（`reinforce_feedback` 兜底） |
| 任意层 | 触碰 `IMMUTABLE_TARGETS` | **一律拒绝** |

### 7.6 P4 受控代码 patch 的前置条件（暂不开，先写死门槛）

只有同时满足以下全部条件，才允许评估开启：

1. 配置层已跑满 **≥ 8 代**，有效提案率 ≥ 40%；
2. 冻结集指标连续 3 代单调不降；
3. 目标文件单测覆盖率 ≥ 80%；
4. patch 限制为 AST 级、只允许改白名单文件、禁止新增 import、禁止删测试；
5. 必须有独立沙箱进程执行 + 全量单测通过才可进入门禁。

---

## 八、分阶段路线

| 阶段 | 目标 | 交付物 | 验收口径 |
|---|---|---|---|
| **P0** 接线（先行） | 把负样本真正汇入诊断闭环 | `extract_failure` → `bad_cases` 打通；`agentworkflow` 诊断结果回写 `root_cause`；跑一次全量 triage 分布 | 能说出当前 bad case 的 R1–R8 分布及 Top3 根因 |
| **P1** 提案生成器（离线） | 从根因产出结构化提案 | `improvement/generator.py` + `improvement/proposal.py`；CLI：`python -m improvement.propose --from-run <run_id> --cause R5` | 每条提案有 root_cause / rationale / payload_before / payload_after，且 `change_target` 在白名单内 |
| **P2** 沙箱 A/B 编排 | 自动跑基线 vs 变体并出 diff | `improvement/sandbox.py`，复用 `evalkit/schema.save_run` + `--compare`；扩展 `--set` 支持 prompt 变体 | 一次 A/B 产出 `baseline_run_id` / `variant_run_id` / 双集 diff |
| **P3** 门禁与自动合并 | 变好才合并，变差自动回滚 | `improvement/gate.py` + 三张表 + 状态机；**只对 P1 提示词层开自动** | 构造一条**故意变差**的提案，验证它被门禁拦下并自动回滚 |
| **P4** 跨代复利度量 | 证明 R 真的存在 | `improvement/lineage.py`：有效提案率、提案产出速度随代数变化 | 见第九章 |
| **P5**（门槛制） | 受控代码 patch | 满足 7.6 全部条件后才启动 | — |

**P0 不需要任何新组件，是纯接线工作，建议立刻做。**

---

## 九、验收：怎么证明它真的在自我改进

只报「指标从 X 涨到 Y」不够——那可能是记忆通胀。必须同时交这三张表：

### 表一：冻结集指标随代数变化

| 代 | 有效提案 / 总提案 | 冻结集 Recall@5 | 冻结集 nDCG@5 | R8 数 | 自动回滚数 |
|---|---|---|---|---|---|
| A₀（基线） | — | 实测值 | 实测值 | 0 | 0 |
| A₁ | — | | | | |
| A₂ | — | | | | |

### 表二：抗通胀检测（对应 1.2 第 1 条）

**清空 playbook 与已合并 prompt 变更，回到 A₀ 配置重跑冻结集。**

- 如果指标掉回 A₀ 附近 → 说明涨的是「配置改对了」，真实改进；
- 如果清空后指标**仍然高** → 说明是记忆/检索泄漏，不是改进；
- 如果清空后指标**低于 A₀** → 说明存在负向迁移，必须回滚。

这张表是本方案**最有判别力的一张**，缺了它整个体系的自证就站不住。

### 表三：跨代复利（对应 2.2，终局验收）

| 指标 | A₀→A₁ | A₁→A₂ | 判定 |
|---|---|---|---|
| 产出一条「有效提案」平均所需轮数 | | | 下降 = 改进能力在提高 |
| 提案一次通过门禁的比例 | | | 上升 = 同上 |
| 单代自动合并提案数 | | | 上升且不伴随回滚上升 |

**只有这张表呈正向趋势，才能称这个系统是 RSI；否则它只是一个带自动调参的 RAG。**

---

## 十、明确不做的事

1. **不做强 RSI**：不碰模型权重、不做自训练闭环。理由：本仓是业务系统，算力与数据规模都不支撑，且责任边界不清。
2. **不做自由改源码**：P0–P3 期间，Agent 不产出 `.py` 的 diff。这是 7.6 门槛制的内容。
3. **不做自动改业务数据**：`kb_ops_employee.py:22` 那条「不自动改数据」继续有效，本方案不改它。
4. **不做实时在线学习**：所有改进走离线批处理 + 门禁，主链路零侵入。任何自动组件失败都降级跳过
   （沿用 `langgraph_rag_agent.py:710-715` 的 try/except 降级模式）。
5. **不做「AI 自主设定目标」**：优化目标（Recall/nDCG/faithfulness/泄漏零容忍）由人写死，
   对应竞品共识——**目前所有自改进系统的评估标准都是人设计的**，没有例外。

---

## 十一、可复现步骤（P0 先行部分）

```bash
# 环境：一律使用项目指定解释器
PY="D:/prom/anaconda/envs/py310/python.exe"

# 1) 跑一次检索层全量评测，看当前 R1–R8 根因分布
$PY -m evalkit.runner --suite retrieval --mode pipeline

# 2) 跑答案层（含 judge，分钟级）
$PY -m evalkit.runner --suite answer --fast

# 3) 参数 A/B 沙箱现成可用（以精排字符上限为例）
$PY -m evalkit.runner --suite retrieval \
    --set rerank_max_chars=1200 --compare last \
    --note "P2 沙箱：精排单条字符上限 3000→1200"

# 4) 单条 bad case 走 agentworkflow 诊断（只读，不改数据）
$PY -m agentworkflow.diagnose --bc-id <id>
```

P1 及之后的代码尚未编写，上面的命令只覆盖 P0 接线验证。
**P1 落地前，先补齐冻结集（≥30 条人工确认 case），否则双集验证无从谈起。**

---

## 十二、与竞品的差距（诚实版）

| 维度 | AlphaEvolve / DGM | 本方案 P3 完成时 | 差距原因 |
|---|---|---|---|
| 可改对象 | 任意源码 | 提示词 / 参数 / 经验三层 | 可回滚性优先，见第四章 |
| 判好坏的方式 | 机械可判定 / 可执行测试 | judge + 人工标注黄金集 | RAG 质量天然不可机械判定 |
| 规模 | 80 轮演化、完整 benchmark | 数十条 case、数代 | 算力与数据规模限制 |
| 量化结果 | SWE-bench 20→50 | 暂无（待 P4 实测） | 未跑，不预填 |

**这张表是给自己的提醒：不要拿 DGM 的数字给本方案贴金。**
本方案的价值不在「复现 DGM」，而在「在一个有真实租户隔离约束的业务系统里，
把自改进闭环安全地接进去」——这件事大厂论文里没有答案，因为他们的 benchmark 不需要考虑跨租户泄漏。

---

## 十三、附：竞品信息来源

- AlphaEvolve：Google DeepMind 官方博客与技术白皮书（2025-05）；第三方汇总分析（vendor-reported，无独立复现）
- Darwin Gödel Machine：Sakana AI + UBC，arXiv:2505.22954，ICLR 2026；代码 github.com/jennyzzt/dgm
- DSPy / GEPA：Stanford NLP，dspy.ai 官方文档与 optimizer 页面；GEPA 对比 MIPROv2 数据出自 ICLR 2026 论文
- Voyager：Wang et al.，NVIDIA + Caltech，2023（TMLR）；消融数据（−73% / −93%）出自原论文
- Reflexion：Shinn et al.，2023 NeurIPS；HumanEval 80%→91% 出自原论文
- 记忆力通胀 / SICA 17%→53% 塌方 / 人工技能 +16.2pp vs 自生成 +0.0pp：2026-08 第三方分析文章（非论文，引用时标注为分析结论）

**注意**：AlphaEvolve 的分数多为 Google 自报，无第三方复现，引用时须带这个限定。

---

## 十四、实施状态（2026-10-25，P0–P3 已落地）

### 14.1 新增代码

| 文件 | 职责 | 行数 |
|---|---|---|
| `improvement/whitelist.py` | 可改对象白名单 + T1 不可变护栏 + 根因到目标的路由表 | 218 |
| `improvement/schema.py` | 提案数据结构、状态机、门禁阈值常量、指标 diff | 233 |
| `improvement/store.py` | 三张表建表与读写，含 archive 保留失败分支 | 331 |
| `improvement/gate.py` | 六道闸判定与分级裁决（auto / manual / reject） | 253 |
| `improvement/generator.py` | 提案生成器（LLM + 确定性变异算子双路） | 274 |
| `improvement/sandbox.py` | 双集切分、suite 选择、沙箱 A/B | 254 |
| `improvement/runner.py` | CLI：triage / split / propose / validate / merge / rollback / list / status | 372 |
| `tests/test_improvement_p3.py` | 41 项单测 | 391 |
| `prompt_manager.py` | **改动**：新增沙箱进程内覆盖钩子（3 行短路 + 3 个函数） | +34 |

### 14.2 真实跑通过的（有输出为证）

```
# 提案生成（离线确定性变异，无 LLM 也能跑）
python -m improvement.runner propose --cause R5 --no-llm
  -> 已生成 pp-20261025-326b2f  generate_answer (971 -> 1022 字符)
  -> 已生成 pp-20261025-1eb0cb  writer_compose  (429 -> 481 字符)

python -m improvement.runner propose --cause R8 --no-llm
  -> 根因 R8 在提示词层没有可改对象（要么该改参数，要么属不可变范围）

# 冻结集切分
python -m improvement.runner split --suite retrieval
  -> 共 15 条 -> dev 11 / holdout 4
  -> 警告：冻结集仅 4 条，低于 30 条门槛，该规模下门禁不会自动放行
```

单测 **41 项全绿**（1 项因 MySQL 不可达 skip），其中最能说明问题的几条：

- `test_regression_rejected`：故意构造指标倒退 → 门禁 **reject**
- `test_small_holdout_forces_manual`：冻结集 5 条 → 只给 **manual**，不给自动放行
- `test_r8_zero_tolerance` / `test_new_r5_regression_rejected`：零容忍项生效
- `test_immutable_target_rejected_by_gate`：`judge_prompt` 触碰护栏即拒
- `test_prompt_override_inject_and_clear`：沙箱覆盖跑完必须清干净
- `test_p3_sandbox_refuses`：经验层没有注入口 → 明确报不可用，不假装跑过

回归：`tests/test_p1_review_fixes.py` **29/29 PASS**（改了 `prompt_manager.get_prompt` 后的回归验证）。

### 14.3 没跑通的，以及为什么

| 项 | 阻塞原因 | 需要什么 |
|---|---|---|
| 沙箱真实 A/B | Milvus `192.168.200.128:19530` 不可达；MySQL 同上 | 虚拟机在线 |
| `triage` 读 bad case | MySQL 不可达，内存态为空 | 虚拟机在线 |
| 提案落库与合并 | MySQL 不可达 | 虚拟机在线 |
| LLM 提案生成（非离线路径） | 未实测，网关路由未配 `improvement` task | 配网关 + 虚拟机在线 |

沙箱失败路径本身**已验证正确**：`ab_test` 返回 `ok=False` 且错误信息明确，
`finally` 里覆盖已清干净（实测残留 `{}`），不会污染线上。

### 14.4 与方案的偏差（诚实记录）

1. **黄金集规模远低于设想**：实测 `retrieval.jsonl` 只有 **15 条数据行**（44 行里多数是注释），
   `answer.jsonl` 只有 **9 条**。方案里「冻结集 ≥30 条」现阶段**达不到**。
   处理方式不是放宽阈值，而是把「样本不足」做成门禁的 G2 硬闸：
   低于 30 条一律 `manual`，禁止自动合并。**宁可不给过，也不拿噪声当改进。**
2. **切分算法改了**：方案没指定算法。实作为「按 case_id 哈希排序取前 k 条」，
   放弃了「哈希阈值分桶」——后者在 15 条小样本上实测只切出 2 条，条数不可预期，
   而门禁要拿条数当硬判据。代价是牺牲了一点「新增 case 不影响旧成员」的稳定性，
   现阶段的瓶颈是样本量不够，不是集合抖动。
3. **发现一个方案里没写的关键事实**：**提示词变体在检索层评测上完全看不到效果**。
   `harness_retrieval` 走 Retriever 直查向量库，不经过任何 prompt；
   只有 `harness_answer` 才消费 prompt。所以 P1/P3 必须跑 answer 层、
   P2 才跑 retrieval 层。选错层会得到「改了但指标一动不动」的假阴性。
   已在 `sandbox.suite_for_layer()` 强制。
4. **P3 经验层沙箱本次不做**：playbook 必须先落库才能被检索到，没有进程内注入口。
   明确报不可用，不假装跑过。
5. **状态机补了一条**：回滚时不传操作者则保留原操作者（`decided_by` 不清空），
   否则复盘时找不到「当初谁合的」。这是单测暴露出来的真实缺陷。

### 14.5 下一步（需要虚拟机在线才能继续）

1. 起虚拟机，跑 `python -m improvement.runner triage` 看真实根因分布
2. 人工把冻结集补到 30 条以上（这是自动合并的前置条件）
3. 配网关 `improvement` task，实测 LLM 提案生成路径
4. 跑通一次完整闭环：`propose -> validate -> merge`，并**构造一条故意变差的提案**
   验证它在真实环境里也被拦下
