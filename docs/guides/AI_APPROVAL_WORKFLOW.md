# AI 自动审批流 —— 用 Dify 编排物料采购审核

> 定位：`ai_approval.py` + `config/ai_approval.yaml` + Dify 工作流。
> 人工审批链在 `approval.py`（不动它），AI 预审插在它**前面**。
>
> 本文回答三件事：**为什么是这条业务**、**审的是哪些数字**、**AI 凭什么不能越权**。

---

## 0. 先纠正上一轮的方向

上一轮把 Dify 当成「外部知识库」接到 `/retrieval` 上，那是把 Dify 降格成检索代理了。
Dify 真正值钱的是**编排**：可视化工作流、条件分支、HTTP 节点、LLM 节点、变量在节点间流转、
每一步可回放。这些能力对应的业务形态，不是「查资料」，而是**有明确多级判定规则的流程**。

所以方向应该反过来：不是「把 Dify 塞进 RAG」，而是「用 Dify 编排一个审批流，
让它在关键节点回头调用本项目的 RAG 取数」。**RAG 不跟 Dify 冲突，RAG 是 Dify 的下游数据源。**

---

## 1. 反推：什么业务适合交给 Dify 编排

### 1.1 Dify 工作流能提供的六种原语

| 原语 | 用途 | 在审批里的角色 |
|---|---|---|
| Start 变量 | 接外部入参 | 接 `po_id` / `request_id` |
| HTTP 请求节点 | 调任意 REST | 拉本项目「数字包」和制度条款 |
| LLM 节点 | 结构化输出 | 出 `decision / risk_level / confidence` |
| 条件分支（if-else） | 按变量分流 | 三分支：自动批 / 转人工 / 建议拒绝 |
| 变量聚合 / 代码节点 | 拼装、换算 | 把数字包拼成 prompt，把多分支结果归一 |
| End 节点 | 固定出参 | 返回唯一结论给调用方 |

一个容易走的弯路：**想用 Dify 的「迭代节点」把整条审批路线卷在画布里跑完**。
不这么做的原因见 §5.1——路线与裁决必须留在代码里，所以 Dify 每次只处理**一站**，
由 Python 侧循环调用。这不影响画布的可视化价值：单站内部照样是
「检索 → 拼提示词 → 调模型 → 校验输出」的可视化链路。

### 1.2 什么样的业务值得这么干（四条判据）

一条业务同时满足这四条，才适合 AI 自动审批：

1. **判定依据是数字，不是文字。** 能写成 `值 vs 阈值`，才能被规则层复核；
   纯定性的业务（「这个文案好不好」）交给 AI 自动决策就是赌博。
2. **量大到人工审不过来。** 一天三张单，人工审毫无压力，上 AI 是自找麻烦。
3. **错了代价可控，或者能兜住。** 审批错了要么金额小，要么有人工二审。
4. **有明确的多级权限。** 系统里已经有「谁能批、批多大额度」的规则，
   AI 才能被塞进这个框架里当一个「有额度的审批人」，而不是游离在外。

### 1.3 业务候选对比

本项目里现成的高危动作有 7 个（`approval.py::build_default_executors`），逐个按上面四条打分：

| 动作 | 判定依据是数字 | 单量 | 错判代价 | 有多级权限 | 结论 |
|---|---|---|---|---|---|
| `purchase_order.approve` 采购审批 | 强（金额/单价/库存/供应商） | 高 | 可控（金额分层） | 有（>10万须 level=3） | **选它** |
| `sales_order.complete` 销售关单 | 中（金额、账期） | 中 | 高（钱收不回） | 有 | 二期 |
| `sales_delivery.release` 发货放行 | 中（库存、授信） | 高 | 中 | 有 | 二期 |
| `inventory.adjust` 盘点调整 | 弱（差异原因千奇百怪） | 低 | 高 | 有 | 不适合 |
| `repair_order.resolve` 维修结算 | 弱（工时主观） | 中 | 低 | 有 | 不适合 |
| `sales_order.cancel` 取消订单 | 弱 | 低 | 高 | 有 | 不适合 |
| `repair_order.cancel` 取消工单 | 弱 | 低 | 低 | 有 | 不适合 |

**采购审批被选中，是因为它每一条判据都能落地成字段**：金额在 `purchase_orders.amount`、
单价在 `purchase_order_items.unit_price`、库存消耗在 `inventory_transactions`、
供应商资质在 `suppliers`。这不是「挑了个好讲的例子」，
而是这个项目里唯一一个「数字密度足够高、能撑起自动判定」的动作。

所以 `ApprovalGate.ai_review()` 里写死了一句拒绝：

```python
AI_REVIEW_ACTIONS = ("purchase_order.approve",)
# 其他动作 —— 该动作还没有定义取数口径 ... 拒绝用编造的指标做审批。
```

宁可只覆盖一个动作，也不给六个动作编六套假指标。

---

## 2. 物料审核这条业务长什么样

### 2.1 现状（没有 AI 之前）

```
采购员建单 PO(draft)  →  submit_po  →  PO(submitted)
                                          ↓
                              数字员工/采购员发起审批单
                              approval_requests(status=pending)
                                          ↓
                              部门经理打开待审列表，看一眼，点通过
                                          ↓
                              decide() → execute_approve → PO(approved) → 收货入库
```

问题不在流程，在**「看一眼」看的是什么**。待审列表只有
`#12 [purchase_order.approve] 发起: digital_employee 金额: 98000 理由: 产线补料`。
经理要靠记忆判断「这个供应商靠不靠谱」「这个价格贵不贵」「上个月是不是也买过一批」——
记不住，最后就变成看金额没超线就点同意。

### 2.2 加了 AI 预审之后

```
PO(submitted)
   ↓
ApprovalGate.ai_review(request_id)
   ↓
┌──────────────────────────────────────────────────────────────────────┐
│ ① DataPackBuilder   → 12 个硬指标（真实 SQL，逐项可回源）              │
│ ② RuleEngine        → findings + risk_score + hard_block              │
│ ③ HistoryDigest     → 采购历史档案（同料成交、价格趋势、供应商近期单）   │
│ ④ RoutePlanner      → 这张单走哪几站（金额分段 + 历史信号插站）         │
│ ⑤ NodeAdvisor       → 逐站调 Dify：每站只看自己的指标、只出本站结论      │
│ ⑥ arbitrate_route   → 多站汇总裁决（保守取严）                         │
│ ⑦ build_handover    → 转人工时生成交接单                              │
└──────────────────────────────────────────────────────────────────────┘
   ↓                                    ↓
auto_approve                    human_review / recommend_reject / degraded
   ↓                                    ↓
gate.decide(ai_approver)        单据仍 pending，但带一张交接单：
（走同一套权限链）              卡在哪站、为什么不敢定、各站依据、
                                需采购员回答什么、建议怎么处置
```

注意最后一步：**AI 自动批也要调 `gate.decide()`**，不是直接 UPDATE 状态。
这样「等级够不够」「是不是自审自批」「执行器有没有跑成功」「审计有没有落」这四件事
全部复用原有代码，AI 拿不到任何特权路径。

---

## 3. 审的是哪些数字（12 个硬指标）

每一条都能落回具体表和字段。这张表就是「可回源」的凭据。

| # | 指标 | 怎么算 | 数据源 | 为什么它影响审批 |
|---|---|---|---|---|
| 1 | `amount` 本单金额 | 直接取 | `purchase_orders.amount` | 金额决定审批等级（>10万须 level=3） |
| 2 | `price_deviation_pct` 单价偏离 | 本单单价 vs 该物料近 90 天已成交加权均价 | `purchase_order_items.unit_price` ÷ (`SUM(quantity*unit_price)/SUM(quantity)`) | 涨价没有理由就是有问题；也抓围标抬价 |
| 3 | `unit_price_vs_cost_pct` 高于参考进价 | 本单单价 vs 主数据参考进价 | `products.cost_price` | 主数据里的进价是已知底线，超了要解释 |
| 4 | `qty_ratio` 采购量倍数 | 本单量 ÷ 近 30 天该物料净消耗 | `inventory_transactions`（排除 RESERVE/RELEASE） | 一次买 6 个月的量，不是囤货就是错单 |
| 5 | `stock_days` 收货后可用天数 | (在库 − 已预留 + 在途 + 本单) ÷ 日均消耗 | `inventory` + `inventory_transactions` | 和 #4 互为校验：量大但确实缺料就合理 |
| 6 | `supplier_age_days` 供应商账龄 | 距今建档天数 | `suppliers.created_at` | 新供应商首单风险天然高 |
| 7 | `supplier_po_90d` 近90天成交单数 | 计数 | `purchase_orders`（已成交状态） | 没有历史成交 = 没有履约证据 |
| 8 | `on_time_rate_pct` 准时交付率 | 按期收货单数 ÷ 有期望到货日的已收单数 | `purchase_orders.expected_date` vs `updated_at` | 老拖期的供应商，救急单不能给它 |
| 9 | `reject_rate_pct` 历史审批被拒率 | 该供应商被拒审批单 ÷ 总审批单 | `approval_requests` + JSON 反查 `payload.po_id` | 被拒过的单还要再报，说明没改 |
| 10 | `budget_used_pct` 本月预算占用 | (本月已批额 + 本单) ÷ 月预算 | `purchase_orders` + 配置 | 预算见底时每笔都要看清楚 |
| 11 | `split_amount_pct` 在途合计占审批线 | (本单 + 同供应商窗口期在途) ÷ 审批线 | `purchase_orders` | **拆单识别**，见下 |
| 12 | `safety_shortage_*` 安全库存缺口 | 可用量 − `safety_stock` | `products.safety_stock` | 缺料的单要快批，不能和囤货同等对待 |

### 3.1 第 11 条为什么单独拿出来说

这是整条流里最有价值的一条，因为它抓的是**故意规避**，不是**判断失误**：

- 审批规则是「单笔 >10 万须总经理批」。
- 采购员把 12 万的单拆成 9.8 万 + 2.2 万，两张单都在部门经理权限内。
- 单看任何一张，金额、单价、库存全都正常——**任何只审单张单据的系统都抓不到**。

所以判定要跨单看：

```python
bypass = bool(amount < line <= total and rows)
# 本单自己没超线，但和同供应商窗口期内的单合起来超了 → 拆单规避
```

反过来说，**本单自己就超线不算拆单**。这条区分很重要：把「明目张胆的大额单」
也标成拆单，规则会天天误报，人就又回到「反正都是红的，直接点通过」。

拆单在规则层是 `hard_block`，`arbitrate()` 第一条就拦，AI 说得再好听也翻不了案。

### 3.2 算不出来的指标怎么处理

`DataPackBuilder._compute()` 的纪律：**算不出来就是 `None`，不是 `0`**。

- 没有历史成交价 → `price_deviation_pct = None`，进 `missing`
- 没有近 30 天消耗 → `qty_ratio` / `stock_days` 都是 `None`，进 `missing`

然后 `arbitrate()` 第二步：`missing` 里只要含核心指标（`_CORE_METRICS`），直接转人工。
理由很直白——「0% 价格偏离」和「没有历史价所以算不出偏离」是两件完全不同的事，
把后者当后者处理，AI 就会在一张没有任何参照的单据上信心满满地签字。

---

## 4. 六层分离，为什么不能合并

```
┌─────────────────────────────────────────────────────────────────┐
│ ⑥ arbitrate_route() / build_handover()   纯函数，可单测           │
│    规则 ∨ 每站意见 ∨ 权限额度 → 一个可执行动作 + 一张交接单        │
├─────────────────────────────────────────────────────────────────┤
│ ⑤ NodeAdvisor   → llm_gateway(task=dify_approve) → Dify 工作流   │
│    每站一次调用。只读数字、只给本站结论、只补依据。                │
│    不碰算数、不碰一票否决、不知道别的站说了什么                    │
├─────────────────────────────────────────────────────────────────┤
│ ④ RoutePlanner   金额分段 + 历史信号 → 这张单走哪几站              │
│ ③ HistoryDigest  7 块采购历史（Dify 不会查的那些）                │
├─────────────────────────────────────────────────────────────────┤
│ ② RuleEngine    阈值 → findings → risk_score / hard_block        │
│    纯确定性，同一输入永远同一输出，可复现                          │
├─────────────────────────────────────────────────────────────────┤
│ ① DataPackBuilder  11 条 SQL → 12 个指标                         │
│    唯一 DB 出口 _q()，测试可注入                                  │
└─────────────────────────────────────────────────────────────────┘
```

四个「为什么不合并」：

**① 为什么算数不用 LLM。** 金额、单价偏离、采购量倍数必须可复现：
同一张单跑两次要一样，换台机器要一样，审计调出来要一样。LLM 做不到，
所以它**不产生数字，只消费数字**。

**② 为什么每站互相看不见对方的结论。** 如果采购主管能看到「部门经理已同意」，
它会倾向于附和——这是真实的组织行为，模型也会复现。所以各站独立出结论，
**只在最后一层汇总**。这样「四站全部同意」才是一条有意义的信息，
而不是一串串味的回声。`NodeAdvisor` 给每站的 prompt 里确实只含本站材料。

**③ 为什么 LLM 不参与一票否决。** 如果「AI 觉得还行」能推翻拆单判定，
那所有风控规则都可以被一句好话绕过去。规则层的 `hard_block` 在 `arbitrate_route()`
第 1 条，物理上排在所有站点意见之前。

**④ 为什么裁决要独立成纯函数。** `arbitrate_route()` 不碰 IO、不读时钟、不连任何服务，
输入输出都是字典。它的 10 条判据每条都有独立用例（`t_arbitrate_route`），
每个分支都能单独验——这种地方留着 DB 连接或者网络调用就没法测了。

---

## 5. Dify 工作流怎么搭

### 5.1 编排分两半，这是最关键的一个取舍

上一版把「一个 LLM 节点出一个结论」当作整条流，那是**一个判官**，不是一条审批流。
真实的采购审批是**多站串联**：部门经理问「这料该不该买」、采购主管问「从这家买、这价谈到位没」、
财务问「预算还够不够」、总经理问「这钱该不该花」。四个岗位看同一段 prompt、出同一段建议，等于没做。

所以编排被劈成两半，各归各位：

```
┌─ Python（不可外包的那一半）─────────────────────────────────────────┐
│  ① DataPackBuilder   12 指标          ② RuleEngine   findings       │
│  ③ HistoryDigest     采购历史档案      ④ RoutePlanner 这张单走哪几站  │
│  ⑤ NodeAdvisor  ── 逐站调 Dify ──►    ⑥ arbitrate_route  汇总裁决    │
│  ⑦ build_handover    转人工的交接单                                  │
└─────────────────────────────────────────────────────────────────────┘
                          │ 每站一次调用
                          ▼
┌─ Dify 工作流（画布那一半）───────────────────────────────────────────┐
│  [HTTP] 拉制度条款 → [LLM] 按本站职责研判 → 严格 JSON 回传           │
└─────────────────────────────────────────────────────────────────────┘
```

**为什么路线和裁决必须留在 Python、不能画在画布上。**
画布是配置，配置能被改。如果「哪几站」「任一站反对怎么办」「平均置信度够不够」这些规则
写在画布里，那么任何人打开 Dify 把连线的条件分支改一下，风控就换了。而这三件事恰好是
风控本身。留在 Python 里，它们才有单测（`t_route_plan` / `t_arbitrate_route` 共 60 余项），
才能被 `--check` 逐条打印出来对账，才改不动。

**为什么单站研判放进 Dify。**
因为那部分真的适合可视化编排：检索制度条款（HTTP 节点）、拼 prompt（变量聚合）、
调模型（LLM 节点）、校验输出格式（代码节点）。这些改动频率高、需要业务方一起看，
放在画布上比改 Python 快得多。**能改的那部分交给画布，不能改的那部分留在代码**，这是这两层的分界线。

### 5.2 单次调用图（一个节点内部）

```
[Start]  variables: po_id, knowledge_id, node_id
      │
      ▼
[HTTP 请求] 拉数字包（一次拿全，含历史档案与整条路线）
  POST {本项目}/ai/pack
  body: {"po_id": {{#start.po_id#}}, "knowledge_id": {{#start.knowledge_id#}}}
  → {{#http_pack.body.metrics#}} .missing .items .history .route .gate
      │
      ▼
[代码节点] 从 route.nodes 里挑出本站
  node = filter(route.nodes, node_id == {{#start.node_id#}})
  material = node.material      ← 后端已按本站 focus_* 渲染好，画布不拼内容
      │
      ▼
[HTTP 请求] 拉制度条款（走 /retrieval）
  POST {本项目}/retrieval
  body: {"knowledge_id":"kb-default","query":"采购审批 制度 单价 拆单",
         "retrieval_setting":{"top_k":6,"score_threshold":0.3}}
  → {{#http_kb.records#}}
      │
      ▼
[LLM] 按本站职责研判
  system: 见 ai_approval.py::_NODE_SYSTEM（title / duty / hint 三段由节点配置注入）
  user:   node.material + 制度条款
  → {{#llm.text#}}（严格 JSON，见 §5.5）
      │
      ▼
[End] 直接把这一段 JSON 回给 NodeAdvisor，由它收集齐所有站再裁决
```

注意最后一步：**Dify 不裁决、不汇总、不下发动作**，它只负责「这一站怎么看」。
汇总与裁决在 `arbitrate_route()` —— 这是故意的，理由见 §5.1。

### 5.3 为什么路线要变长：金额分段 + 历史插站

上一版一条路走到底，5 千的单和 50 万的单走同样的站，这就是「流太短」的根源。
现在路线由两层配置生成（`config/ai_approval.yaml`）：

**① 金额分段**（`routes`，从上往下匹配，第一个 `amount <= up_to` 生效）

| 金额 | 站点数 | 路线 |
|---|---|---|
| ≤ 5 万 | 1 | 部门经理 |
| ≤ 10 万 | 2 | 部门经理 → 财务审核 |
| ≤ 30 万 | 4 | 部门经理 → 采购主管 → 财务审核 → 总经理 |
| 无上限（`up_to: 0`） | 4 | 同链，逐站都要更高等级确认 |

**② 历史信号动态插站**（`insert_rules`，命中就把节点插到指定位置，已存在不重复插）

| 信号 | 判定条件 | 插入节点 | 位置 | 为什么 |
|---|---|---|---|---|
| `split_bypass` | 本单未超线，但 + 同供应商窗口期在途 ≥ 线 | 合规复核 | 部门经理之后 | 疑似拆单规避，须合规复核 |
| `new_supplier` | 建档 < 90 天或无历史成交 | 供应商准入复核 | 部门经理之后 | 新供应商首单风险天然高 |
| `price_spike` | 单价偏离超 `high` 阈值 | 采购主管 | 部门经理之后 | 价格异常须重新比价 |
| `budget_tight` | 本月预算占用 ≥ `medium` 阈值 | 财务审核 | 部门经理之后 | 预算见底要提前看 |
| `late_supplier` | 准时率低于 `medium` 阈值 | 采购主管 | 部门经理之后 | 交付不稳要评估备选 |

效果示例：一张 16,000 的新供应商首单 → `[部门经理, 供应商准入复核]` 两站；
一张 250,000 且疑似拆单的单 → `[部门经理, 合规复核, 采购主管, 财务, 总经理]` 五站。

**这条路线的意义不是「多问了几个问题」，而是把判断权放到该拿的人手里。**
1 站的流等于让部门经理替全厂签字；5 站的流才对应真实的工厂审批。

### 5.4 职责隔离：每站只看自己那一摊

每个节点在配置里写死四样东西：`duty`（负责什么）、`hint`（怎么判、倾向如何、明确禁止评什么）、
`focus_metrics`（只看哪几个指标）、`focus_history`（只看哪一块历史）。

| 节点 | level | 职责 | 只看这些指标 | 只看这些历史 |
|---|---|---|---|---|
| 部门经理 | 2 | 需求真实性与用量合理性 | `qty_ratio` `stock_days` `safety_shortage_*` | `issued_trend` `material_purchases` |
| 采购主管 | 2 | 供应商选择与成交价格 | `price_deviation_pct` `unit_price_vs_cost_pct` `supplier_po_90d` `on_time_rate_pct` `reject_rate_pct` | `price_trend` `supplier_recent_orders` `supplier_share_pct` |
| 财务审核 | 2 | 预算占用与资金时点 | `amount` `budget_used_pct` | `quarterly_amount` |
| 总经理 | 3 | 综合风险与经营影响 | `amount` `split_amount_pct` `price_deviation_pct` `supplier_age_days` `budget_used_pct` | `supplier_recent_orders` `quarterly_amount` `price_trend` |
| 合规复核 | 3 | 审批合规性 | `split_amount_pct` `amount` `split_sibling_amount` | `supplier_recent_orders` |
| 供应商准入复核 | 2 | 新供应商首单风险 | `supplier_age_days` `supplier_po_90d` `on_time_rate_pct` `reject_rate_pct` | `supplier_recent_orders` |

`render_node_material()` 只渲染本站 `focus_*` 里列出的项，其余一律不出现。
测试里专门验了两个方向（`t_material_isolation`）：部门经理的材料里**不含**价格偏离，
财务的材料里**不含**采购量倍数。这不是锦上添花——如果四站看到的是同一份材料，
它们必然给出同一段建议，那多站就是纯浪费 token，还稀释了责任。

### 5.5 逐站输出契约

每站必须回严格 JSON，`parse_node_advice()` 只容忍 Markdown 围栏和前后寒暄：

```json
{
  "advice": "agree | agree_with_conditions | oppose | need_info",
  "confidence": 0.87,
  "key_points": ["指标=值 → 影响"],
  "conditions": ["若同意需附加的条件"],
  "questions": ["需要采购员补充说明的问题"],
  "one_liner": "一句话给审批人看的建议（不超过 40 字）"
}
```

四条硬要求写进 system prompt：

1. 只能依据给出的指标与历史，**不得编造数字**；
2. 只评自己有权管的维度（部门经理不评价格、财务不评该不该买）；
3. 每条结论必须落到具体数字，格式「指标=值 → 影响」；
4. 数据不足就写进 `questions` 点名要补什么，**不许把缺数据当成没问题**。

不合规 → 该站 `ok: False` → 汇总裁决直接 `degraded`（见 §5.6），不是「跳过这站继续算」。

### 5.6 与代码的连接点：`/ai/pack` 的返回契约

`/ai/pack` 挂在 `dify_kb_server.py` 同一个进程上（和 `/retrieval` 共用一个 Bearer 密钥、一个端口）。
一次请求把**整条路线 + 每站材料 + 历史档案**全带回去，Dify 不用二次取数：

```json
{
  "ok": true,
  "po":      { "po_no": "PO-202610-0101", "amount": 250000.0, "supplier_name": "..." },
  "items":   [ { "product_name": "...", "quantity": 200, "unit_price": 80.0 } ],
  "metrics": { "amount": 250000.0, "price_deviation_pct": 2.56, "...": "..." },
  "missing": [],
  "rule":    { "risk_score": 0, "risk_level": "low", "hard_block": false,
               "blockers": [], "findings": [] },
  "history": { "_material": "...", "material_purchases": [], "price_trend": {},
               "supplier_recent_orders": [], "supplier_share_pct": 62.0,
               "quarterly_amount": {}, "issued_trend": [] },
  "route":   { "nodes": [
                 { "node_id": "dept_manager", "title": "部门经理", "level": 2,
                   "duty": "需求真实性与用量合理性",
                   "material": "（只含 qty_ratio / stock_days / 安全库存 + 领用趋势）" },
                 { "node_id": "compliance", "title": "合规复核", "level": 3,
                   "duty": "审批合规性", "reason": "疑似拆单规避审批，须合规复核",
                   "material": "（只含 split_amount_pct / 金额 / 兄弟单）" }
               ],
               "max_level": 3, "base_note": "10~30 万 · 采购主管比价 + 财务 + 总经理",
               "inserted": ["compliance"], "signals": {"split_bypass": true},
               "warnings": [] },
  "gate":    { "required_level": 2, "route_max_level": 3, "effective_level": 3,
               "ai_level": 2, "max_auto_amount": 50000.0, "auto_eligible": false }
}
```

三个设计取舍值得说明：

**为什么把指标和规则都放在服务端算。** 如果让 Dify 用代码节点自己算
「单价偏离」，那同一张单在 Dify 里和在本项目里会算出两个数，出了争议没法对账。
数字只在一个地方产生，Dify 只负责把它们喂给 LLM、把结论带回来。

**为什么 `material` 是每站一份而不是共用一份。** 见 §5.4。画布侧只做
`filter(route.nodes, node_id == 本站)`，不拼内容——内容一进画布，职责隔离就没人保证了。

**`auto_eligible` 只是提示，不是通行证。** 它让画布上的条件分支好写一点，
但即便有人把画布改成「一律走自动批」，服务端的 `arbitrate_route()` 和 `decide()`
照样各拦一道。注意 `effective_level = max(required_level, route_max_level)`——
路线里只要有总经理站（level 3），即便金额只有 2 万，AI 也不够格自动批。

`knowledge_id` 在这里不是用来检索的，是用来**反查租户**的——不接受调用方直传
`tenant_id`，否则「查哪个租户」的决定权就跑到 Dify 画布上去了。

### 5.7 汇总裁决：多站的结论怎么合成一个动作

`arbitrate_route()` 是纯函数，判定顺序**一步都不能调**：

| # | 判据 | 结论 | 理由 |
|---|---|---|---|
| 1 | `rule.hard_block` | `recommend_reject` | 一票否决，AI 无权翻案 |
| 2 | 核心指标缺失 | `human_review` | 不许闭眼签字 |
| 3 | `effective_level > ai_level` 或 `amount > max_auto` | `human_review` | **权限必须排在 AI 意见之前** |
| 4 | 存在 `high` finding | `recommend_reject` | 多站不放宽这条：四个岗位都说好，也谈不下客观高危指标 |
| 5 | 有站没出意见 | `degraded` | 不能用「剩下的站都说好」凑过去 |
| 6 | 任一站 `oppose` | `recommend_reject` | `veto_on_any_oppose: true` |
| 7 | 任一站 `need_info` | `human_review` | 停在那站，要补资料 |
| 8 | 任一站 `agree_with_conditions` | `human_review` | 条件须落实后才放行 |
| 9 | 各站**平均**置信度 < `min_avg_confidence` | `human_review` | 看平均不看最高，防「一站很自信掩盖三站很虚」 |
| 10 | `rule.risk_level != low` | `human_review` | 规则层不放行 |
| — | 全过 | `auto_approve` | reason 里写明「N 站全部同意（平均置信度 x）」 |

五个分支和十个判据都有独立用例（`t_arbitrate_route` / `t_arbitrate_strict`），
同一输入永远同一输出——这是「审批动作可复现」的前提。

第 5 条最容易被误解成过度保守，但反过来看：**如果一个岗位的 AI 挂了，
它挂的那一刻恰恰是最该看数据的岗位没看**。这时候用其余站的意见去自动批，
等于拿掉了一整道职责。所以宁可 `degraded`。

### 5.8 接进网关而不是直连

`NodeAdvisor` 走的是 `llm_gateway.chat_detailed(task="dify_approve")`，
不是直接 `requests.post` 到 Dify。白拿四样东西：限流（rpm/tpm）、熔断、
失败降级、token 计量归因到 `ai_approver`。

**这条路由必须显式写在 `config/llm_gateway.yaml` 的 `routing` 里：**

```yaml
  dify_approve: [dify-flow, local-qwen]
```

不写会怎样？实测过，两个后果都很严重：

1. **采购单据数据会被发到云端。** 任务名未命中 `routing` 时，网关回退到
   `default_chain: [local-qwen, deepseek-v4-pro, qwen-plus]`。本地模型一旦不可用，
   供应商、单价、预算这些数据就发给外部模型了——直接违反工厂「数据不出内网」。
2. **装了 Dify 也不会被调用。** `dify-flow` 不在候选链上，整条工作流是死代码，
   而且不报错，看起来一切正常。

写完可以一行验证（`resolve_chain` 返回的是**过滤后**真正会试的链）：

```
>>> LLMGateway().resolve_chain("dify_approve")
['local-qwen']        # dify-flow 的 enabled=false，被过滤掉
```

`config/llm_gateway.yaml` 里 `dify-flow` 默认 `enabled: false`，
**不会因为没配 Dify 就影响现有链路**。此时链上只剩 `local-qwen`；
本地模型也不可用时整条链失败 → 各站 `ok=False` → 裁决 `degraded` → 转人工。
失败会明说，不会假装成功（见 §9 排障第一行）。

### 5.9 输出校验：宁可白跑一趟

`parse_node_advice()` 只容忍两种格式噪音：Markdown 围栏、前后寒暄。
**不容忍**：缺键、`advice` 不在白名单（`agree / agree_with_conditions / oppose / need_info`）、
`confidence` 非数字或越界、`key_points` 不是数组。
任何一项不合规 → 该站 `{"ok": False}` → 裁决第 5 条 `degraded` → 转人工。

一句话：宁可让 AI 白跑一趟，也不让一个含糊的结论影响审批。

---

## 6. 安全边界：四道闸

这是整篇文档最该被记住的部分。AI 想干坏事，得连过四关。

### 闸一：额度，等级与业务线取严

```python
route_level = int(route.get("max_level") or 0)
need = max([x for x in (required_level, route_level) if x] or [2])
if need > ai_level:                                    # level 3 > 2
    return out("human_review", "超出 AI 审批等级，必须更高角色审批")
if amount > max_auto:                                 # 120000 > 50000
    return out("human_review", "超出 AI 自动批金额上限")
```

`required_level` 来自 `ApprovalGate.required_level()`，读的是
`config/access_rules.yaml` 的 `approval_rules`——和真人审批**同一个数据源**。
AI 角色是 `ai_approver`（level=2），采购 >10 万要 level=3，所以 10 万以上它连
「申请自动批」的资格都没有。改 `ai_approval.yaml` 的 `max_auto_amount` 也绕不过去，
因为那是另一道独立的检查。

**多站之后这一闸多了一维：路线最高等级也参与取严。** 只要路线里排进了总经理站
（`level: 3`），哪怕金额只有 2 万、`required_level` 还是 2，AI 也不够格自动批。
否则会出现「金额小 → AI 自动批掉了 → 总经理根本没看到这张单」的漏洞：
风险高不高，不只看金额，还看这张单为什么会被排到高等级那一站。

### 闸二：权限链，即使绕过裁决也会被拒

裁决层拦是一回事，`decide()` 自己拦是另一回事。测试里专门验了这一条
（`t_real_chain`）：直接调 `gate.decide(..., decided_role="ai_approver")` 批一张 12 万的单：

```
PermissionError: 审批角色「ai_approver」等级不足（当前 2，须 ≥3）——
                「purchase_order.approve」 金额 120000.0 超 10 万须 level=3；
                须更高角色审批（已审计）
```

**两道闸都验，因为前者是逻辑、后者是底线。** 逻辑会被人改，底线不能。

### 闸三：禁自审自批对 AI 同样生效

`decide()` 里 `if decided_by == r.get("requested_by")` 这条规则没有为 AI 开口子。
AI 发起的单，AI 自己批不了——测试也验了。

### 闸四：AI 建议拒绝 ≠ 自动拒绝

`arbitrate()` 里所有「应该拒」的分支返回的都是 `recommend_reject`，
单据仍然是 `pending`，等人工确认。

理由是不对称的代价：误杀一张正常单（产线停工等料）远比漏放一张异常单
（人还有二审、还能追责）严重。AI 在这里的定位是**筛掉显然正常的 80%**，
把人的注意力省下来给剩下 20%，不是替人做决定。

---

## 7. 四种结论怎么落地

| action | 单据状态 | 谁动手 | 人的负担 |
|---|---|---|---|
| `auto_approve` | → approved，执行器已跑 | AI（`ai:ai_approver`） | 事后抽查 |
| `human_review` | 仍 pending | 人 | 读 AI 意见后决策 |
| `recommend_reject` | 仍 pending | 人 | 重点看 AI 给的拒绝理由，确认即可 |
| `degraded` | 仍 pending | 人 | AI 挂了，回到纯人工 |

`recommend_reject` 的单据，`reason` 里会写明触发原因（拆单、高风险指标、AI 建议拒绝），
`ai_approval_reviews` 表里还存了完整快照：12 个指标、findings、各站建议原文、
耗时、用的哪个 task；多站时 `ai_approval_node_advices` 表按节点各存一行。人可以只核对关键那一条。

### 7.1 转人工必须带交接单（不能只把球丢过去）

「转人工」和「把活推给人」的区别就在一张交接单上。`build_handover()` 输出七块，
`format_handover()` 把它渲染成审批人一屏能读完的文本：

| 字段 | 回答什么问题 |
|---|---|
| `conclusion` | AI 到底给了结论没有（明确同意 / 明确建议拒 / 没定下来 / 降级） |
| `stage` | **卡在哪一站**（不是笼统的「有风险」，是「卡在财务审核」） |
| `avg_confidence` | 各站平均置信度是多少 |
| `why_uncertain` | 为什么不敢定（blockers 原文，含数字与依据） |
| `recommended_action` | **建议人工怎么处置**（按卡点给动作，不是一句「请审核」） |
| `questions_for_requester` | 需要采购员回答的问题（按站归集，可直接转给申请人） |
| `advices` | 每站：结论 + 一句话 + 依据 + 条件 + 待问 |

`recommended_action` 是随卡点变的，这是它有用的原因：

- 卡在**权限** → 「该单超出 AI 授权范围（等级或额度），须由更高角色审批；AI 意见可作参考，但不能代替等级要求」
- 卡在**资料不全** → 「让采购员回答 questions_for_requester 后重新提交」
- 卡在**条件** → 「确认条件落实即可放行」
- `recommend_reject` → 「核实 why_uncertain 里的依据；确认属实即驳回，让采购员按各站条件重新提交」
- `degraded` → 「按传统方式人工审核。已算好的指标可直接引用，不必重新取数」

最后一条尤其重要：**AI 挂了，它算好的数字仍然有效。** 取数与规则是确定性代码，
不依赖模型；降级只是没有意见，不是没干活。

命令行看交接单：

```bash
python approval.py --ai-advice 12          # 打印某张单的交接单
python approval.py --ai-review 12 --ai-advice
```

渲染样张（下面这段是用构造的 advices 真实调用 `format_handover(build_handover(row))`
打印出来的原文，不是手写示意；真实单据的字段相同，数字来自 DB）：

```
========================================================================
AI 交接单：AI 未取得一致结论，转人工判断（下方为各站建议）
========================================================================
卡住的环节：财务审核
各站平均置信度：0.865

── 各站意见 ──
[同意] 部门经理（agree，置信度 0.91）
    结论：缺料确认，采购量在合理范围，建议立即采购
    · qty_ratio=1.8 → 相当于 1.8 个月用量，仍在常备区间
    · stock_days=21.0 → 无停机风险
[有条件同意] 财务审核（agree_with_conditions，置信度 0.82）
    结论：预算可批，但批完余量不足 5%
    · budget_used_pct=94.2 → 占用超 90% 警戒线
    + 条件：批完剩余额度 42,000，本月后续采购须先报预算调整
    ? 待问：本月是否还有已承诺未下单的采购计划？

── 为什么不敢直接定 ──
  · 【财务审核】有条件同意：批完剩余额度 42,000，本月后续采购须先报预算调整

── 需要采购员回答的问题 ──
  1. 【财务审核】本月是否还有已承诺未下单的采购计划？

── 建议你这样做 ──
  请先看 stage 指明的那一站。若卡在资料不全，让采购员回答 questions_for_requester 后重新提交；若卡在条件，确认条件落实即可放行。
========================================================================
```

**这是一张「可执行」的交接单，不是一份分析报告。** 审批人不缺权力，缺的是判断依据；
把依据、卡点、待问问题、推荐处置一次交清，人工那一关可能只需要三十秒。

---

## 8. 部署

```bash
# 1. 配置
cp .env.example .env          # 填 DIFY_APP_API_KEY（Dify 应用密钥）
vi config/ai_approval.yaml    # 按租户改预算、阈值、max_auto_amount

# 2. 自检（不连库，纯配置校验）
python ai_approval.py --check
#   等级自检：AI level=2，审批线 100,000 元以上须 level=3；
#   AI 自动批上限 50,000 元 → 在权限内
#   审批路线（按金额分段）
#     ≤ 50,000   1  部门经理
#     ≤ 100,000  2  部门经理 → 财务审核
#     ≤ 300,000  4  部门经理 → 采购主管 → 财务审核 → 总经理
#     无上限      4  部门经理 → 采购主管 → 财务审核 → 总经理
#   路线自检：4 段金额路线 / 5 条插站规则 / 6 个节点，引用全部有定义
#   汇总裁决：任一站反对是否直接否决自动批 = True；各站平均置信度门槛 = 0.8
#   ↑ 引用未定义的节点（比如 insert_rules 里写了 nodes 里没有的名字）→ exit 1

# 3. 有库之后先验 SQL 取数（这一步千万别跳）
python ai_approval.py --sql-check 12        # 打印 12 个指标 + 中间量 + 历史档案

# 3b. 起桥服务后，按下 Dify 工作流会打的那一发（确认端到端通）
python dify_kb_server.py --config config/dify_bridge.yaml &
curl -sS -X POST http://127.0.0.1:8300/ai/pack \
  -H "Authorization: Bearer $DIFY_KB_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"po_id": 12, "knowledge_id": "kb-default"}' | python -m json.tool
#   → 检查 route.nodes 的站数与 order，以及每站 material 是否只含自己的指标

# 4. 试跑（不写库、不自动批）
python approval.py --ai-review 12 --dry-run

# 5. 单站排查（只看某一站怎么判，partial 一律不自动批）
python approval.py --ai-review 12 --node finance --dry-run

# 6. 正式跑
python approval.py --ai-review 12

# 7. 看交接单（转人工时给人看的那一页）
python approval.py --ai-advice 12
```

Dify 侧：工作流应用建好后，在 `config/llm_gateway.yaml` 把 `dify-flow`
的 `enabled` 改成 `true`，`api_key_env` 指向的变量填上 app key。

---

## 9. 排障

| 现象 | 大概率原因 | 怎么查 |
|---|---|---|
| 理由含「**LLM 网关全链失败**」 | Dify 没装/没启用，且本地 Ollama 也不可用——**这是没有 Dify 环境的正常表现，不是 bug** | `resolve_chain("dify_approve")` 只剩 `local-qwen`；起 Ollama 或启用 `dify-flow` 即可 |
| ⚠️ 理由含「**AI 输出里找不到 JSON 对象**」 | 曾把网关降级当模型乱输出（已修）。现在还看到这句，多半是模型真的没按 JSON 回 | `ai_approval_node_advices` 里那行 `raw` 字段的原文 |
| `action=degraded`，理由「AI 研判失败」 | 网关/ Dify 没起来，或返回不是 JSON | 看 `row["ai_raw"]["error"]` |
| `degraded` 且理由含「N/M 个节点 AI 未给出有效意见」 | 某几站的 JSON 不合规，不是全挂 | `ai_approval_node_advices` 里 `ok=0` 那几行的 `error` |
| `recommend_reject` 且理由含「拆单」 | 同供应商 7 天内有其它在途单 | 看 `metrics["split_siblings"]` 的 po_no |
| 路线站数比预期多 | 命中插站规则（新供应商/价格异动/预算紧张…） | `/ai/pack` 返回的 `route.inserted` 与 `route.signals` |
| 路线里某站重复出现 | 该节点已在金额分段路线里，`insert_rules` 命中后不再插 | `route.inserted` 不含它即正常 |
| 2 万的小单也 `human_review`「超出 AI 审批等级」 | 路线里排进了 `level: 3` 的站（如合规复核/总经理） | `gate.route_max_level` 与 `gate.effective_level` |
| 大量 `human_review`「核心指标缺失」 | 该物料没有历史采购记录（新品） | `missing` 列表；新品首采应走人工，符合预期 |
| `auto_approve` 但执行失败 | executor 报错（状态不对/权限不足） | `gate.get(rid)["result"]["error"]` |
| `--sql-check` 报 Unknown column | 表结构版本不一致 | 对照 `config/init_db.sql` |

---

## 10. 验证到哪一步（诚实版）

**已验证（真跑过）**

- `python ai_approval.py --check` → exit 0，输出与本文件 §8 一致（含路线表、插站规则、
  节点职责、引用自洽性检查）。
- `python tests/test_ai_approval.py` → **262/262 通过，0 失败**，19 组：

  | 组 | 验什么 |
  |---|---|
  | `t_config` | 配置自洽：等级/额度/阈值/权重、`approval_amount_line` 与 `access_rules` 一致 |
  | `t_metrics_healthy` | 12 指标逐项手算核对（加权均价、可用天数、安全库存缺口） |
  | `t_split_order` | 拆单三个方向：命中 / 本单自己超线不算 / 无兄弟单不算 |
  | `t_missing` | 核心缺失阻止自动批；非核心缺失不阻止 |
  | `t_rules` | 阈值方向（越大越险 / 越小越险）、边界值、风险分级由最严重 finding 兜底 |
  | `t_parse` | `parse_verdict` 13 种输入：围栏、寒暄、缺键、白名单外、越界、非 JSON、空 |
  | `t_arbitrate` | 单站裁决全部分支：auto / human / recommend_reject / degraded |
  | `t_render` | 材料渲染：指标表、缺失标注、制度条款拼装 |
  | `t_real_chain` | **真实 `ApprovalGate` 链路**（见下） |
  | `t_cli` | `_check(policy)` 返回 0——配置自检能跑通 |
  | `t_ai_pack_endpoint` | **`POST /ai/pack` 端点端到端**（真起 server、真发 HTTP） |
  | `t_history` | 历史档案 7 块：同料历史、价格趋势、供应商近期单、份额、季度额、领用趋势 |
  | `t_route` | 金额分段 1/2/4 站（25 万路线 `max_level=3`）+ 拆单/新供应商/价格异动/预算 四类插站 + **去重且原因合并** + 路线与插站引用的节点都有定义 |
  | `t_node_material` | **职责隔离**：部门经理材料不含价格偏离、财务材料不含用量倍数 |
  | `t_parse_advice` | 逐站 JSON 校验：白名单、置信度越界、缺键、围栏 |
  | `t_arbitrate_route` | 多站裁决：单站同意 / 无意见 / 有站失败 / 反对 / 资料不足 / 有条件 / 置信度不足 / 等级不足 / **路线 level=3 拦住** / 规则 high 不放宽 |
  | `t_handover` | 交接单七字段、`recommended_action` 随卡点变、待问问题按站归属、四种 action 的结论措辞、空单兜底 |
  | `t_degraded_gateway` | **网关全链失败**（`LLMResponse.model == "<degraded>"`）：各站 `ok=False` 且错误信息点名「全链失败」而不是误报「找不到 JSON」；裁决 `degraded`；交接单打出失败原因 |
  | `t_e2e_multistage` | 端到端（`_NoDb` 强制走内存兜底、真 `ApprovalGate`）：8 万单走两站 → 财务有条件同意 → 转人工且单据仍 `pending` → 交接单点名财务；单站查询标 `partial` 且**永不自动批**；查不在路线上的节点报错并列出本单路线；两站全同意 → 真自动批 → 单据 `approved` |

  `t_real_chain` 与 `t_ai_pack_endpoint` 的细项：
  - AI 自动批 → executor 真被调用 → 状态变 approved → `decided_by` 落 `ai:ai_approver`；
  - AI 批 12 万被权限链拒（`PermissionError`，见 §6 闸二）；`super_admin` 能批同一张单；
  - AI 自审自批被拒；非采购动作被明确拒绝；缺 `po_id` 报错；
  - 端点：无密钥 403/1001、密钥错 403/1002、`po_id` 非整数 400、
    `knowledge_id` 不在白名单 404/2001、正常 200 且 `route.nodes[0].material`
    含真实金额且**不含**价格偏离、**DB 不可用返 503 而不是空指标**、
    未知端点 404 带提示、`GET /health` 正常。
- `python dify_kb_server.py --check` → exit 0，输出含 `/ai/pack` 端点行。
- `python tests/test_dify_bridge.py` → 116/116 通过（改动无回归）。

**没有 Dify 环境的实跑结果（这机器上就是没有）**

「需要 Dify 才能测吗」——不需要，而且**必须**先把没有 Dify 的路径跑通，
否则等于系统只在理想环境里成立。下面是在本机（无 Dify、无 Ollama、虚拟机未开机）
**真调 `LLMGateway`**（不注入 chat、不 mock）的结果：

```
=== dify_approve 实际生效链 ===
    ['local-qwen']                     ← dify-flow 未启用被过滤，链上不含云端 deepseek

=== 逐站调用（1.6 万单，单站）===
   部门经理 ok=False  调用失败：RuntimeError: LLM 网关全链失败（Dify 与本地模型均不可用），
                     拿到的是降级文案而不是模型输出

=== 汇总裁决 ===
   action = degraded | stage = AI
   reason = 1/1 个节点 AI 未给出有效意见，不能按剩余节点自动批

=== 交接单 ===
AI 交接单：AI 未能给出完整意见（已降级），下列建议仅供参照
卡住的环节：AI
── 各站意见 ──
[未产出] 部门经理
    原因：调用失败：RuntimeError: LLM 网关全链失败（Dify 与本地模型均不可用）…
── 建议你这样做 ──
  按传统方式人工审核。已算好的指标可直接引用（amount、budget_used_pct、
  on_time_rate_pct、price_deviation_pct、qty_ratio、reject_rate_pct、
  safety_shortage_items、safety_short…），不必重新取数。
```

（上面把交接单里那一长行折了行，其余逐字照抄终端输出。`safety_short…` 的截断
是 `build_handover()` 里对指标名列表做的 120 字符截断，不是抄漏。）

三件事被这次实跑证明了：

1. **没有 Dify 不会假装成功**，也不会自动批——`degraded` 后回人工。
2. **指标和规则仍然算完了**：取数与打分是确定性代码，不依赖模型。
   降级只损失「意见」，不损失「数字」，所以交接单里那句「不必重新取数」是真的。
3. **失败原因是可诊断的**：一眼看出是「网关全链失败」，而不是去猜模型为什么不听话。

**这次实跑抓到两个真 bug（单测覆盖不到、也没写在之前的文档里）**

| # | 问题 | 后果 | 修法 |
|---|---|---|---|
| 1 | `routing` 里**没有** `dify_approve` 这条路由 | task 未命中 → 回退 `default_chain`（含 `deepseek-v4-pro` 云端）；且 `dify-flow` 不在链上，**装了 Dify 也永远不被调用** | 显式加 `dify_approve: [dify-flow, local-qwen]`（`config/llm_gateway.yaml`） |
| 2 | 网关全链失败时返回 `degraded_reply` 文案，`NodeAdvisor` 把它当模型输出解析 | 报错是「AI 输出里找不到 JSON 对象」，把「模型连不上」误诊成「模型乱输出」，排障方向全错 | `_call()` 识别 `resp.model == "<degraded>"` 并显式抛错；`format_handover()` 把失败原因打进交接单 |

两个都补了测试（`t_degraded_gateway`，9 项），**第 1 个尤其重要**：
它意味着本次交付前，Dify 编排在真实运行时是死代码，而采购数据反而可能外发云端。

**未验证（环境限制，不掩盖）**

- **17 条取数 SQL（11 条指标 + 6 条历史）没在真实库上跑过。** MySQL 位于虚拟机
  `192.168.200.128:3306`，本次验证时未开机（同机 19530/11436/11434 三个端口全部超时）。
  字段名与 `config/init_db.sql` 逐条人工核对一致，但**语法与语义只做到这**——
  有库后必须补 `python ai_approval.py --sql-check <po_id>`。
- 真实 Dify 工作流的「导入画布 → 跑通一次 → 回 JSON」没验，本机无 Dify 实例。
  验证只到 `NodeAdvisor._call()` 的边界（测试里注入 chat 函数替代，`_FakeAdvisor`
  可指定逐站返回或指定某站失败）与 `/ai/pack` 端点本身
  （真起 server 真发 HTTP，但取数用构造数据）。
- `ai_approval_reviews` / `ai_approval_node_advices` 两张表的建表语句没在真实 MySQL 执行过（同上）。
- 真实 LLM 的逐站建议质量没评估过——没有 Dify 实例、本地 `local-qwen` 也没在本次跑。
  测试里所有建议都是 `_FakeAdvisor` 造的，**验的是协议与裁决，不是文风与判断水平**。

**顺手修掉的一个真缺陷**（跑真实渲染时发现，不是测试抓到的）：
`format_handover()` 里 `recommended_action` 的字符串跨行拼接丢了空格，
打印出来是「让采购员回答questions_for_requester」，已修（`ai_approval.py` 内补尾随空格）。
这类问题单测抓不到——只有把渲染结果真打出来看才发现，所以 §7.1 的样张是从函数输出复制的。

---

## 11. 改动清单

| 文件 | 类型 | 行数 | 说明 |
|---|---|---|---|
| `config/ai_approval.yaml` | 新增 | 301 | 12 指标阈值 / 权重 / AI 额度 / 预算 / Dify 契约 + **多站三块**：`routes`（金额分段）/ `insert_rules`（历史插站）/ `nodes`（6 节点职责与话术）/ `route_rules`（裁决门槛） |
| `ai_approval.py` | 新增 | 2511 | 取数 → 规则 → 历史档案 → 路线规划 → 逐站建议 → 汇总裁决 → 交接单 → 落库 |
| `config/llm_gateway.yaml` | 修改 | — | **新增 `routing.dify_approve: [dify-flow, local-qwen]`**。缺这条时 task 回退 `default_chain`（含云端 deepseek，采购数据外发）+ dify-flow 永不生效，是两个真问题，不是配置美化 |
| `tests/test_ai_approval.py` | 新增 | 1554 | **262 项 / 19 组**，零外部依赖（DB 用 SQL 替身，LLM 用 chat/gateway 替身） |
| `docs/guides/AI_APPROVAL_WORKFLOW.md` | 新增 | 本文 | 11 章 |
| `approval.py` | 修改 | 551 | +`ai_review()` 钩子；+CLI `--ai-review / --ai-advice / --dry-run / --node`；`required_level` 抽成模块级 `required_level_for()`（同一权威实现给 `/ai/pack` 用）；人工路径零改动 |
| `dify_kb_server.py` | 修改 | 1012 | +`POST /ai/pack`（返回 `history` + `route` + 每站 `material` + `gate.effective_level`）；端点分发抽 `_handle_retrieval` / `_handle_ai_pack`；body 读取抽 `_read_json_body`；`_send_json` 容 datetime/Decimal |
| `config/access_rules.yaml` | 修改 | — | 注册 `ai_approver` 角色（level=2，ops 只有 query/approve）；注释写明缺这条记录时的 fail-closed 行为 |

新增的两张表（`_DDL` / `_NODE_DDL`，均 `CREATE TABLE IF NOT EXISTS`，首次落库时自动建）：

- `ai_approval_reviews` —— 一张单一行：12 指标、findings、路线、裁决、耗时、用的哪个 task
- `ai_approval_node_advices` —— **一张单 N 行（每站一行）**：`node_id / title / level / advice /
  confidence / one_liner / key_points / conditions / questions / material / raw / latency_ms`

为什么建议单独一张表而不是塞进 `ai_approval_reviews` 的 JSON 字段：
「哪一站最容易反对」「哪个站的置信度长期偏低」「哪一站的 AI 老是解析失败」这三个问题
要能直接 SQL 聚合。塞进 JSON 就只能全表拉出来在内存里数。
