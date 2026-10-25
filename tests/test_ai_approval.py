#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 自动审批流自测（零外部依赖：不连 MySQL、不连 Dify、不起端口）
================================================================================

运行：
    python tests/test_ai_approval.py

【这份测试守的是什么】

这条链最容易翻车的地方不是「AI 判断得准不准」，而是「AI 判断完之后，
有没有绕过原有风控」。所以测试重点放在**边界**而不是**准确性**：

    1. 拆单识别：本单不超线、同供应商几天内几张单合起来超线 → 必须一票否决。
       而且「本单自己就超线」不能算拆单（那是明目张胆，不是规避），
       这两件事混一起会让规则天天误报。
    2. 数据缺失：算不出来的指标必须进 missing 并阻止自动批。
       绝不能把「算不出」当成「没问题」——这是 AI 审批最阴的假成功。
    3. AI 越权：金额超 AI 等级能批的范围时，必须在调 decide() **之前**拦住；
       而且就算有人绕过裁决直接调 decide()，ApprovalGate 的权限链也得拒。
       两道闸都要验，因为前者是逻辑、后者是底线。
    4. AI 输出容错：只容忍「带 Markdown 围栏 / 前后有寒暄」，
       不容忍缺键、值不在白名单、confidence 越界。这些猜不得。
    5. AI 挂了：一律降级转人工，绝不假装通过。

【怎么做到「跑了真实代码」而不连库】

    · DataPackBuilder 的唯一 DB 出口是 _q()。这里注入一个按 SQL 特征分派的
      假查询函数，于是 `_compute()` 里 12 个指标的计算、加权、拆单扫描
      全部是**生产代码**在跑，测试只提供原始行。
    · ApprovalGate 这里用「临时屏蔽 pymysql」的方式强制它走内存兜底分支，
      于是 request/decide/_row/required_level 也都是生产代码。
      屏蔽是 try/finally 的，跑完立刻恢复，不影响同进程其他东西。

【没验到的部分（诚实标注）】

    11 条取数 SQL 的语法与语义**没有在真实库上跑过**——本次验证时
    MySQL 位于虚拟机 192.168.200.128，未开机。所以「指标算得对不对」验了，
    「SQL 写得对不对」只做到人工核对字段名与 config/init_db.sql 一致。
    有库后请补：python ai_approval.py --sql-check <po_id>
"""

import builtins
import datetime as dt
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ai_approval as A          # noqa: E402

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ===========================================================================
# 工具：按 SQL 特征分派的假查询（让 DataPackBuilder 的计算逻辑真跑）
# ===========================================================================
class FakeERP:
    """
    模拟取数 SQL 的返回（11 条数字包 SQL + 6 条历史档案 SQL）。

    分派依据用「最具体的唯一片段」，并且**顺序从具体到宽泛**：
    比如 history 里查 purchase_orders 的那条（supplier_recent_orders）
    必须排在数字包查 purchase_orders 的那几条之前，否则会被误吃。
    """

    def __init__(self, po, items, hist=None, issued30=None, stock=None,
                 transit=0, sup_po_cnt=0, ontime=None, reject=None,
                 used=0.0, siblings=None, material_purchases=None,
                 supplier_orders=None, supplier_share=None,
                 quarterly=None, issued_rows=None):
        self.po, self.items = po, items
        self.hist = hist or {"qty": 0, "amt": 0, "rows_cnt": 0}
        self.issued30 = issued30
        self.stock = stock or {"stock": 0, "reserved": 0}
        self.transit = transit
        self.sup_po_cnt = sup_po_cnt
        self.ontime = ontime or {"ontime": 0, "total": 0}
        self.reject = reject or {"rejected": 0, "total": 0}
        self.used = used
        self.siblings = siblings or []
        # ---- 历史档案（HistoryDigest）----
        self.material_purchases = material_purchases or []
        self.supplier_orders = supplier_orders or []
        self.supplier_share = supplier_share
        self.quarterly = quarterly or {"this_quarter": 0, "this_year": 0,
                                       "last_year": 0}
        self.issued_rows = issued_rows or []
        self.calls = []

    def __call__(self, sql, params=()):
        s = " ".join(sql.split())
        self.calls.append(s[:60])
        # ---- 历史档案优先（它们的表与数字包重叠）----
        if "sup_amt" in s:
            total = self.supplier_share
            if total is None:
                return [{"sup_amt": 0, "total_amt": 0}]
            return [{"sup_amt": round(total, 2), "total_amt": 100.0}]
        if "this_quarter" in s:
            return [dict(self.quarterly)]
        if "s.name AS supplier_name" in s and "i.quantity, i.unit_price" in s:
            return list(self.material_purchases)
        if "expected_date, created_at, updated_at" in s:
            return list(self.supplier_orders)
        if s.startswith("SELECT created_at, qty FROM inventory_transactions"):
            return list(self.issued_rows)
        # ---- 数字包 ----
        if "rows_cnt" in s:
            return [dict(self.hist)]
        if "SUM(-t.qty)" in s:
            return [{"used": self.issued30}]
        if "SUM(stock_qty)" in s:
            return [dict(self.stock)]
        if "AS in_transit" in s:
            return [{"in_transit": self.transit}]
        if "COUNT(*) AS c" in s:
            return [{"c": self.sup_po_cnt}]
        if "AS ontime" in s:
            return [dict(self.ontime)]
        if "AS rejected" in s:
            return [dict(self.reject)]
        if "AS used FROM purchase_orders" in s:
            return [{"used": self.used}]
        if s.startswith("SELECT id, po_no, amount, status, created_at"):
            return list(self.siblings)
        if "LEFT JOIN suppliers s ON" in s:
            return [dict(self.po)]
        if "LEFT JOIN products p ON" in s:
            return [dict(i) for i in self.items]
        raise AssertionError(f"FakeERP 未识别的 SQL：{s[:120]}")


def _dt(days_ago):
    return dt.datetime.now() - dt.timedelta(days=days_ago)


class _FakeAdvisor:
    """
    替身 NodeAdvisor：按路线节点数生成建议，每站内容可以不同。

    :param per_node: {node_id: {"advice":..,"confidence":..,"one_liner":..}}
                     不在表里的节点用 default_*
    """

    def __init__(self, per_node=None, advice="agree", confidence=0.9,
                 one_liner="可批", fail_nodes=None):
        self.per_node = per_node or {}
        self.advice = advice
        self.confidence = confidence
        self.one_liner = one_liner
        self.fail_nodes = set(fail_nodes or [])

    def advise(self, pack, history, rule, route):
        out = []
        for n in (route.get("nodes") or []):
            nid = n["node_id"]
            if nid in self.fail_nodes:
                out.append({"node_id": nid, "title": n["title"],
                            "level": n["level"], "ok": False,
                            "error": "模拟：调用超时", "material": "m",
                            "latency_ms": 5000})
                continue
            cfg = {"advice": self.advice, "confidence": self.confidence,
                   "one_liner": self.one_liner}
            cfg.update(self.per_node.get(nid) or {})
            out.append({"node_id": nid, "title": n["title"],
                        "level": n["level"], "ok": True,
                        "key_points": [f"{nid}: 指标=值 → 影响"],
                        "conditions": cfg.pop("conditions", []),
                        "questions": cfg.pop("questions", []),
                        "material": f"{n['title']} 的材料", "latency_ms": 800,
                        **cfg})
        return out


def good_case(**over):
    """
    基准场景：一张没有毛病的采购单（应当自动批准）。

    数字全部手算过，见场景 A 的断言。
    """
    po = {"id": 101, "tenant_id": "default", "po_no": "PO-202610-0101",
          "supplier_id": 7, "supplier_code": "SUP-007",
          "supplier_name": "华鑫五金", "supplier_active": 1,
          "supplier_created_at": _dt(730), "amount": 16000.0,
          "status": "submitted", "buyer": "张三", "created_by": "张三",
          "expected_date": dt.date.today(), "created_at": _dt(1)}
    items = [{"id": 1, "product_id": 11, "product_code": "M-A",
              "product_name": "不锈钢螺栓 M8", "quantity": 200,
              "unit_price": 80.0, "received_qty": 0, "cost_price": 75.0,
              "safety_stock": 100, "product_unit": "件"}]
    kw = dict(po=po, items=items,
              hist={"qty": 600, "amt": 46800.0, "rows_cnt": 3},
              issued30=150.0, stock={"stock": 40, "reserved": 10},
              transit=0, sup_po_cnt=8,
              ontime={"ontime": 18, "total": 20},
              reject={"rejected": 0, "total": 10},
              used=200000.0, siblings=[],
              # ---- 历史档案 ----
              # 真实 SQL 是 ORDER BY o.id DESC，所以这里也必须**新的在前**；
              # 写成旧的在前会让 price_trend 的「上次成交价」取错方向。
              material_purchases=[
                  {"po_no": "PO-202609-0011", "created_at": _dt(40),
                   "status": "received", "supplier_name": "恒达标准件",
                   "quantity": 300, "unit_price": 79.0},
                  {"po_no": "PO-202608-0003", "created_at": _dt(70),
                   "status": "received", "supplier_name": "华鑫五金",
                   "quantity": 300, "unit_price": 76.5},
              ],
              supplier_orders=[
                  {"po_no": "PO-202609-0011", "amount": 23700.0,
                   "status": "received", "expected_date": (dt.date.today()
                                                           - dt.timedelta(days=40)),
                   "created_at": _dt(45), "updated_at": _dt(38)},
                  {"po_no": "PO-202608-0003", "amount": 22950.0,
                   "status": "received", "expected_date": (dt.date.today()
                                                           - dt.timedelta(days=70)),
                   "created_at": _dt(75), "updated_at": _dt(72)},
              ],
              supplier_share=62.5,
              quarterly={"this_quarter": 180000.0, "this_year": 520000.0,
                         "last_year": 960000.0},
              issued_rows=[{"created_at": _dt(10), "qty": -60},
                           {"created_at": _dt(40), "qty": -90},
                           {"created_at": _dt(70), "qty": -50}])
    kw.update(over)
    return FakeERP(**kw)


AI_OK = {"ok": True, "verdict": {
    "decision": "approve", "risk_level": "low", "confidence": 0.9,
    "key_factors": ["单价偏离历史均价=2.56% → 与历史一致"],
    "evidence": [], "missing_data": [], "suggestion": "可批准"},
    "latency_ms": 850}


# ===========================================================================
# 1. 配置与工具
# ===========================================================================
def t_config():
    section("1. 配置加载 / 阈值方向 / 小工具")
    p = A.load_policy()
    check("load_policy 读到 yaml（不是兜底）",
          p["limits"]["max_auto_amount"] == 50000, str(p["limits"]))
    check("AI 角色 level=2（够批 10 万以内）", p["ai_approver"]["level"] == 2)
    check("审批金额线 10 万与 access_rules 对齐",
          p["approval_amount_line"] == 100000)

    # 阈值方向表必须与 yaml 的 thresholds 键一一对应，否则配了阈值的指标会被静默跳过
    yaml_keys = set((p.get("thresholds") or {}).keys())
    code_keys = set(A._THRESHOLD_DIR.keys())
    check("阈值方向表与 yaml 指标键一致", yaml_keys == code_keys,
          f"仅yaml={sorted(yaml_keys - code_keys)} 仅代码={sorted(code_keys - yaml_keys)}")

    # 缺文件时的兜底不能放宽：额度、门槛、等级三项取严
    tmp = A.load_policy(os.path.join(ROOT, "config", "__not_exist__.yaml"))
    check("yaml 缺失 → 兜底策略生效", tmp["limits"]["max_auto_amount"] == 50000)
    check("兜底策略 AI 等级仍为 2（不放权）", tmp["ai_approver"]["level"] == 2)

    # 深合并：只覆盖一项，其余保留
    m = A._deep_merge({"a": {"x": 1, "y": 2}, "b": 3}, {"a": {"y": 9}})
    check("深合并只覆盖指定键", m == {"a": {"x": 1, "y": 9}, "b": 3}, str(m))

    # 月预算租户覆盖
    p2 = A.load_policy()
    p2["budget"]["tenants"] = {"jm": {"monthly": 300000}}
    check("月预算按租户覆盖", A.monthly_budget(p2, "jm") == 300000)
    check("月预算未覆盖时回落 default",
          A.monthly_budget(p2, "other") == 800000)

    # 工具函数：None 不能变成 0
    check("_f(None) → None（不冒充 0）", A._f(None) is None)
    check("_f('abc') → None", A._f("abc") is None)
    check("_f('12.5') → 12.5", A._f("12.5") == 12.5)
    check("_days_since(None) → None", A._days_since(None) is None)
    check("_days_since(30天前) ≈ 30", A._days_since(_dt(30)) == 30)
    check("_weighted_avg 按金额加权",
          A._weighted_avg([{"v": 10, "w": 1}, {"v": 20, "w": 3}], "v", "w") == 17.5)
    check("_weighted_avg 全 None → None",
          A._weighted_avg([{"v": None, "w": 1}], "v", "w") is None)
    check("_min_of 取最紧的一条", A._min_of([{"v": 9}, {"v": 3}], "v") == 3)


# ===========================================================================
# 2. 取数与指标计算（场景 A：健康单）
# ===========================================================================
def t_metrics_healthy():
    section("2. DataPackBuilder：12 个指标的真实计算（健康单）")
    fake = good_case()
    b = A.DataPackBuilder(q=fake)
    pack = b.build(101)
    check("取数成功", pack["ok"] is True, pack.get("error") or "")
    m = pack["metrics"]

    check("金额 = 16000", m["amount"] == 16000.0, str(m["amount"]))
    # 历史加权均价 46800/600 = 78.0 → (80-78)/78*100 = 2.5641
    check("单价偏离历史均价 = 2.56%", m["price_deviation_pct"] == 2.56,
          str(m["price_deviation_pct"]))
    # (80-75)/75*100 = 6.6667
    check("单价高于参考进价 = 6.67%", m["unit_price_vs_cost_pct"] == 6.67,
          str(m["unit_price_vs_cost_pct"]))
    # 200/150
    check("采购量倍数 = 1.33", m["qty_ratio"] == 1.33, str(m["qty_ratio"]))
    # (40-10+0+200)/(150/30) = 230/5 = 46
    check("收货后可用天数 = 46.0", m["stock_days"] == 46.0, str(m["stock_days"]))
    check("安全库存缺口 = -70（低于安全线）",
          pack["evidence"]["items"][0]["safety_gap"] == -70)
    check("供应商建档 730 天", m["supplier_age_days"] == 730,
          str(m["supplier_age_days"]))
    check("供应商 90 天成交 8 单", m["supplier_po_90d"] == 8)
    check("准时交付率 = 90%", m["on_time_rate_pct"] == 90.0,
          str(m["on_time_rate_pct"]))
    check("历史被拒率 = 0%", m["reject_rate_pct"] == 0.0)
    # (200000+16000)/800000*100
    check("本月预算占用 = 27%", m["budget_used_pct"] == 27.0,
          str(m["budget_used_pct"]))
    check("拆单线占比 = 16%（本单/10万）", m["split_amount_pct"] == 16.0)
    check("拆单标记 False", m["split_bypass"] is False)
    check("无缺失指标", pack["missing"] == [], str(pack["missing"]))
    check("低于安全库存明细数 = 1", m["safety_shortage_items"] == 1)

    rule = A.RuleEngine().score(pack)
    check("健康单无 findings", rule["findings"] == [], str(rule["findings"]))
    check("健康单风险分 0", rule["risk_score"] == 0)
    check("健康单风险等级 low", rule["risk_level"] == "low")


# ===========================================================================
# 3. 拆单识别（最有价值的一条规则）
# ===========================================================================
def t_split_order():
    section("3. 拆单规避识别")
    # 本单 9.8 万没超线，但同供应商 7 天内还有 3 万 + 1.2 万 → 合计 14 万超线
    sibs = [{"id": 102, "po_no": "PO-202610-0102", "amount": 30000.0,
             "status": "submitted", "created_at": _dt(2)},
            {"id": 103, "po_no": "PO-202610-0103", "amount": 12000.0,
             "status": "approved", "created_at": _dt(4)}]
    po = dict(good_case().po)
    po.update({"id": 104, "po_no": "PO-202610-0104", "amount": 98000.0})
    items = [dict(good_case().items[0])]
    items[0].update({"quantity": 1225, "unit_price": 80.0})   # 1225×80 = 98000
    fake = FakeERP(po=po, items=items,
                   hist={"qty": 600, "amt": 46800.0, "rows_cnt": 3},
                   issued30=150.0, stock={"stock": 40, "reserved": 10},
                   sup_po_cnt=8, ontime={"ontime": 18, "total": 20},
                   reject={"rejected": 0, "total": 10}, used=200000.0,
                   siblings=sibs)
    pack = A.DataPackBuilder(q=fake).build(104)
    m = pack["metrics"]
    check("兄弟单合计 42000", m["split_sibling_amount"] == 42000.0,
          str(m["split_sibling_amount"]))
    check("合计占审批线 140%", m["split_amount_pct"] == 140.0,
          str(m["split_amount_pct"]))
    check("判定为拆单规避", m["split_bypass"] is True)

    rule = A.RuleEngine().score(pack)
    check("拆单 → hard_block", rule["hard_block"] is True)
    check("hard_block 原因点名兄弟单号",
          any("PO-202610-0102" in b for b in rule["blockers"]),
          str(rule["blockers"]))
    check("拆单 → 风险等级 high", rule["risk_level"] == "high")

    v = A.arbitrate(pack, rule, AI_OK, A.load_policy(), required_level=2)
    check("拆单 → 建议拒绝（不是自动拒）", v["action"] == "recommend_reject",
          v["action"])
    check("拆单 → 理由说明 AI 无权翻案", "一票否决" in v["reason"], v["reason"])

    # 反例：本单自己就超线，不算拆单（否则规则天天误报）
    po2 = dict(po)
    po2.update({"id": 105, "po_no": "PO-202610-0105", "amount": 120000.0})
    items2 = [dict(items[0])]
    items2[0].update({"quantity": 1500, "unit_price": 80.0})   # 12 万
    fake2 = FakeERP(po=po2, items=items2,
                    hist={"qty": 600, "amt": 46800.0, "rows_cnt": 3},
                    issued30=150.0, stock={"stock": 40, "reserved": 10},
                    sup_po_cnt=8, ontime={"ontime": 18, "total": 20},
                    reject={"rejected": 0, "total": 10}, used=200000.0,
                    siblings=[])
    pack2 = A.DataPackBuilder(q=fake2).build(105)
    check("本单自己超线 → 不算拆单（bypass=False）",
          pack2["metrics"]["split_bypass"] is False)
    check("但占比仍算出 120%", pack2["metrics"]["split_amount_pct"] == 120.0)

    # 兄弟单里没有本单以外的单 → 即使超线也不算拆单
    check("无兄弟单时 bypass 必为 False",
          pack2["metrics"]["split_bypass"] is False)


# ===========================================================================
# 4. 数据缺失（不许当没问题）
# ===========================================================================
def t_missing():
    section("4. 数据缺失：必须阻止自动批")
    # 没有历史成交、没有近 30 天领用 → 三个核心指标算不出
    fake = good_case(hist={"qty": 0, "amt": 0, "rows_cnt": 0}, issued30=None)
    pack = A.DataPackBuilder(q=fake).build(101)
    m = pack["metrics"]
    check("无历史价 → 偏离为 None（不是 0）",
          m["price_deviation_pct"] is None)
    check("无领用 → 采购量倍数为 None", m["qty_ratio"] is None)
    check("无领用 → 可用天数为 None", m["stock_days"] is None)
    check("三项进 missing",
          set(["price_deviation_pct", "qty_ratio", "stock_days"])
          <= set(pack["missing"]), str(pack["missing"]))

    rule = A.RuleEngine().score(pack)
    check("缺失指标产生 DATA.missing finding",
          any(f["code"] == "DATA.missing" for f in rule["findings"]))

    v = A.arbitrate(pack, rule, AI_OK, A.load_policy(), required_level=2)
    check("核心数据缺失 → 转人工", v["action"] == "human_review", v["action"])
    check("blockers 点名缺了哪些指标",
          any("核心指标缺失" in b for b in v["blockers"]), str(v["blockers"]))

    # 供应商维度缺失不算核心（它影响判断但不阻止算数）
    fake2 = good_case(ontime={"ontime": 0, "total": 0},
                      reject={"rejected": 0, "total": 0})
    pack2 = A.DataPackBuilder(q=fake2).build(101)
    check("准时率无样本 → None 且进 missing",
          pack2["metrics"]["on_time_rate_pct"] is None
          and "on_time_rate_pct" in pack2["missing"])
    check("非核心缺失不阻止流程（仍 ok）", pack2["ok"] is True)


# ===========================================================================
# 5. 规则打分的阈值边界
# ===========================================================================
def t_rules():
    section("5. RuleEngine：阈值方向与边界")
    pol = A.load_policy()
    eng = A.RuleEngine(pol)

    def score_with(**metrics):
        base = {"amount": 1000.0, "supplier_active": 1}
        base.update(metrics)
        return eng.score({"ok": True, "po": {}, "metrics": base,
                          "missing": [], "evidence": {}})

    r = score_with(price_deviation_pct=8.0)     # 恰好等于 medium 阈值
    check("单价偏离=8.0 命中 medium（>= 即触发）",
          any(f["level"] == "medium" for f in r["findings"]))
    r = score_with(price_deviation_pct=7.99)
    check("单价偏离=7.99 不触发", r["findings"] == [], str(r["findings"]))
    r = score_with(price_deviation_pct=20.0)
    check("单价偏离=20.0 命中 high",
          any(f["level"] == "high" for f in r["findings"]))
    check("high 一条即 40 分", r["risk_score"] == 40, str(r["risk_score"]))
    check("一条 high → 等级就是 high（不被总分线吃掉）",
          r["risk_level"] == "high", r["risk_level"])

    # 分级必须由「最严重的那条」兜底：只有一条 medium 时总分仅 15 分，
    # 仍要判 medium，否则权重表一调低，中风险项就被判成 low 放过去了
    r = score_with(qty_ratio=4.0)
    check("一条 medium 仅 15 分", r["risk_score"] == 15, str(r["risk_score"]))
    check("但等级仍是 medium（不按总分线判）",
          r["risk_level"] == "medium", r["risk_level"])

    # 越小越危险的指标：方向反了就会把「优秀」判成「危险」
    r = score_with(on_time_rate_pct=85.0)
    check("准时率=85 命中 medium（<= 即触发）",
          any(f["level"] == "medium" for f in r["findings"]))
    r = score_with(on_time_rate_pct=99.0)
    check("准时率=99 不触发（方向正确）", r["findings"] == [])

    r = score_with(supplier_age_days=30)
    check("新供应商 30 天 → high",
          any(f["level"] == "high" for f in r["findings"]))

    # 供应商停用是硬规则，与阈值无关
    r = score_with(supplier_active=0)
    check("供应商停用 → hard_block", r["hard_block"] is True)
    check("停用理由写明 is_active=0",
          any("is_active=0" in b for b in r["blockers"]), str(r["blockers"]))

    # 风险分累加
    r = score_with(price_deviation_pct=25.0, qty_ratio=7.0)
    check("两条 high 累加到 80 分", r["risk_score"] == 80, str(r["risk_score"]))
    check("80 分 → high 等级", r["risk_level"] == "high")

    # 取数失败（pack.ok=False）必须一票否决，不能当低风险放过去
    bad = eng.score({"ok": False, "error": "连接超时", "metrics": {},
                     "missing": [], "po": {}})
    check("取数失败 → hard_block", bad["hard_block"] is True)
    check("取数失败理由带原始错误",
          any("连接超时" in b for b in bad["blockers"]), str(bad["blockers"]))


# ===========================================================================
# 6. AI 输出解析容错
# ===========================================================================
def t_parse():
    section("6. parse_verdict：只容忍格式噪音，不容忍内容含糊")
    good = json.dumps({"decision": "approve", "risk_level": "low",
                       "confidence": 0.88, "key_factors": ["a=1 → ok"],
                       "evidence": ["制度 3.2"], "missing_data": [],
                       "suggestion": "批准"})
    v, err = A.parse_verdict(good)
    check("标准 JSON 解析成功", err == "" and v["decision"] == "approve", err)

    v, err = A.parse_verdict(f"```json\n{good}\n```")
    check("剥掉 Markdown 围栏", err == "" and v["confidence"] == 0.88, err)

    v, err = A.parse_verdict(f"好的，我的判断如下：\n{good}\n以上。")
    check("剥掉前后寒暄", err == "" and v["decision"] == "approve", err)

    v, err = A.parse_verdict("完全不是 JSON")
    check("非 JSON → 报错", v is None and "找不到 JSON" in err, err)

    v, err = A.parse_verdict("")
    check("空内容 → 报错", v is None and "空内容" in err, err)

    v, err = A.parse_verdict('{"decision":"approve","risk_level":"low"}')
    check("缺 required_keys → 报错（不猜默认值）",
          v is None and "缺少必需字段" in err, err)

    v, err = A.parse_verdict(json.dumps(
        {"decision": "maybe", "risk_level": "low", "confidence": 0.9,
         "key_factors": []}))
    check("decision 不在白名单 → 报错", v is None and "白名单" in err, err)

    v, err = A.parse_verdict(json.dumps(
        {"decision": "approve", "risk_level": "severe", "confidence": 0.9,
         "key_factors": []}))
    check("risk_level 越界 → 报错", v is None and "白名单" in err, err)

    v, err = A.parse_verdict(json.dumps(
        {"decision": "approve", "risk_level": "low", "confidence": 1.7,
         "key_factors": []}))
    check("confidence 越界 → 报错", v is None and "越界" in err, err)

    v, err = A.parse_verdict(json.dumps(
        {"decision": "approve", "risk_level": "low", "confidence": "high",
         "key_factors": []}))
    check("confidence 非数字 → 报错", v is None and "不是数字" in err, err)

    v, err = A.parse_verdict(json.dumps(
        {"decision": "approve", "risk_level": "low", "confidence": 0.9,
         "key_factors": "just one"}))
    check("key_factors 非数组 → 报错", v is None and "不是数组" in err, err)

    v, err = A.parse_verdict(json.dumps([1, 2]))
    check("顶层是数组 → 报错（不是合法 verdict）", v is None and err != "", err)

    # 大小写与首尾空格应被规范化，模型常这么写
    v, err = A.parse_verdict(json.dumps(
        {"decision": " Approve ", "risk_level": "LOW", "confidence": 0.9,
         "key_factors": []}))
    check("decision/risk 大小写与空格被规范化",
          err == "" and v["decision"] == "approve" and v["risk_level"] == "low",
          err)


# ===========================================================================
# 7. 裁决：八个分支
# ===========================================================================
def t_arbitrate():
    section("7. arbitrate：三分支 + 降级的所有出口")
    pol = A.load_policy()
    good_pack = A.DataPackBuilder(q=good_case()).build(101)
    good_rule = A.RuleEngine(pol).score(good_pack)

    v = A.arbitrate(good_pack, good_rule, AI_OK, pol, required_level=2)
    check("全绿 → auto_approve", v["action"] == "auto_approve", v["reason"])

    # 金额超自动批额度
    v = A.arbitrate(good_pack, good_rule, AI_OK, pol, required_level=2)
    pol_low = json.loads(json.dumps(pol))
    pol_low["limits"]["max_auto_amount"] = 1000
    v = A.arbitrate(good_pack, good_rule, AI_OK, pol_low, required_level=2)
    check("金额超自动批额度 → 转人工", v["action"] == "human_review", v["reason"])
    check("理由写明额度", "自动批金额上限" in v["reason"], v["reason"])

    # 等级不足：AI level=2，单据要 level=3
    v = A.arbitrate(good_pack, good_rule, AI_OK, pol, required_level=3)
    check("单据要求 level=3 → 转人工", v["action"] == "human_review", v["reason"])
    check("理由写明等级不足", "等级" in v["reason"], v["reason"])

    # AI 不可用 → 降级
    v = A.arbitrate(good_pack, good_rule, {"ok": False, "error": "网关超时"},
                    pol, required_level=2)
    check("AI 不可用 → degraded", v["action"] == "degraded", v["action"])
    check("降级理由带原始错误", "网关超时" in v["blockers"][0], str(v["blockers"]))

    # AI 建议拒绝
    ai_rej = json.loads(json.dumps(AI_OK))
    ai_rej["verdict"]["decision"] = "reject"
    v = A.arbitrate(good_pack, good_rule, ai_rej, pol, required_level=2)
    check("AI 建议拒绝 → recommend_reject（仍转人工）",
          v["action"] == "recommend_reject", v["action"])

    # AI 要求升级
    ai_esc = json.loads(json.dumps(AI_OK))
    ai_esc["verdict"]["decision"] = "escalate"
    v = A.arbitrate(good_pack, good_rule, ai_esc, pol, required_level=2)
    check("AI 要求升级 → human_review", v["action"] == "human_review", v["reason"])

    # 置信度不足
    ai_low = json.loads(json.dumps(AI_OK))
    ai_low["verdict"]["confidence"] = 0.5
    v = A.arbitrate(good_pack, good_rule, ai_low, pol, required_level=2)
    check("置信度不足 → 转人工", v["action"] == "human_review", v["reason"])
    check("理由写置信度门槛", "置信度" in v["reason"], v["reason"])

    # 规则层有 high（造一个 high finding 的 pack）
    hi_pack = A.DataPackBuilder(q=good_case(
        hist={"qty": 600, "amt": 46800.0, "rows_cnt": 3})).build(101)
    hi_pack["metrics"]["price_deviation_pct"] = 45.0    # 直接改指标，模拟极端行情
    hi_rule = A.RuleEngine(pol).score(hi_pack)
    check("构造出 high finding",
          any(f["level"] == "high" for f in hi_rule["findings"]))
    v = A.arbitrate(hi_pack, hi_rule, AI_OK, pol, required_level=2)
    check("规则有 high → recommend_reject", v["action"] == "recommend_reject",
          v["action"])

    # 规则 medium：AI 说通过也不许自动批
    mid_pack = A.DataPackBuilder(q=good_case()).build(101)
    mid_pack["metrics"]["qty_ratio"] = 4.0              # 3.0~6.0 → medium
    mid_rule = A.RuleEngine(pol).score(mid_pack)
    check("构造出 medium 且无 high",
          mid_rule["risk_level"] == "medium"
          and not any(f["level"] == "high" for f in mid_rule["findings"]))
    v = A.arbitrate(mid_pack, mid_rule, AI_OK, pol, required_level=2)
    check("规则 medium → 转人工（AI 说了不算）",
          v["action"] == "human_review", v["reason"])

    # hard_block 优先级最高：即使 AI 置信度满分也不放
    blk_pack = A.DataPackBuilder(q=good_case()).build(101)
    blk_pack["metrics"]["supplier_active"] = 0
    blk_rule = A.RuleEngine(pol).score(blk_pack)
    hi_conf = json.loads(json.dumps(AI_OK))
    hi_conf["verdict"]["confidence"] = 1.0
    v = A.arbitrate(blk_pack, blk_rule, hi_conf, pol, required_level=2)
    check("hard_block 压过 AI 满置信度 → recommend_reject",
          v["action"] == "recommend_reject", v["action"])


# ===========================================================================
# 8. 给 LLM 的材料渲染
# ===========================================================================
def t_render():
    section("8. render_material：AI 拿到的必须是数字，不是散文")
    pol = A.load_policy()
    pack = A.DataPackBuilder(q=good_case()).build(101)
    rule = A.RuleEngine(pol).score(pack)
    txt = A.render_material(pack, rule, [{"title": "采购管理办法 3.2",
                                          "content": "单笔超 10 万须总经理审批"}])
    check("含单号", "PO-202610-0101" in txt)
    check("含金额", "16,000.00" in txt, txt[:200])
    check("含单价偏离数值", "2.56%" in txt)
    check("含可用天数", "46.0天" in txt or "46.0 天" in txt)
    check("含规则结论", "风险分" in txt)
    check("含制度条款", "单笔超 10 万须总经理审批" in txt)
    check("健康单不虚构缺失项（无 missing 行）", "缺失指标：" not in txt)
    check("材料长度合理（< 4000 字符，避免挤爆上下文）", len(txt) < 4000,
          f"{len(txt)} 字符")

    # 缺数据时材料里必须显式写「数据不足」，不能留空
    pack2 = A.DataPackBuilder(q=good_case(hist={"qty": 0, "amt": 0,
                                                "rows_cnt": 0},
                                          issued30=None)).build(101)
    txt2 = A.render_material(pack2, A.RuleEngine(pol).score(pack2))
    check("缺失指标在材料里显示「数据不足」", "数据不足" in txt2)


# ===========================================================================
# 9. 真实链路：ApprovalGate.ai_review → decide 权限链
# ===========================================================================
class _NoDb:
    """临时屏蔽 pymysql，让 ApprovalGate / AIApprovalFlow 走内存兜底分支。"""

    def __enter__(self):
        self._real = builtins.__import__

        def _imp(name, *a, **kw):
            if name == "pymysql":
                raise ImportError("测试中屏蔽 pymysql（强制内存模式）")
            return self._real(name, *a, **kw)

        builtins.__import__ = _imp
        for m in ("pymysql",):
            sys.modules.pop(m, None)
        return self

    def __exit__(self, *exc):
        builtins.__import__ = self._real
        return False


def t_real_chain():
    section("9. 真实链路：ApprovalGate.ai_review 与 decide 权限链")
    from approval import ApprovalGate

    executed = []

    def fake_executor(payload, actor):
        executed.append({"po_id": payload.get("po_id"), "actor": actor})
        return {"po_no": "PO-202610-0101", "status": "approved"}

    with _NoDb():
        gate = ApprovalGate(executors={"purchase_order.approve": fake_executor},
                            verbose=False)
        check("无 pymysql → 内存兜底模式", gate.available is False)

        # 小金额单：应当自动批准
        rid_small = gate.request("purchase_order.approve",
                                 {"tenant_id": "default", "po_id": 101,
                                  "amount": 16000.0, "supplier_id": 7},
                                 reason="产线补料", requested_by="purchase_user",
                                 requested_role="purchase_user")
        check("审批单创建成功", isinstance(rid_small, int) and rid_small > 0)

        flow = A.AIApprovalFlow.__new__(A.AIApprovalFlow)   # 只注依赖，不连库
        flow.policy = A.load_policy()
        flow.verbose = False
        flow.gate = gate
        flow.rule = A.RuleEngine(flow.policy)
        flow.planner = A.RoutePlanner(flow.policy)
        flow._builder = A.DataPackBuilder(flow.policy,
                                          q=good_case())    # 健康单数据
        flow._reviewer = None
        flow._advisor = _FakeAdvisor()
        flow.available = False
        flow._pymysql = None
        flow._conn_kw = None

        row = flow.review_po(101, tenant_id="default", request_id=rid_small)
        check("预审动作为 auto_approve", row["action"] == "auto_approve",
              row["reason"])
        check("确实调用了 executor（不是只改状态）",
              len(executed) == 1 and executed[0]["po_id"] == 101, str(executed))
        check("executor 拿到的 actor 是 AI 身份",
              executed[0]["actor"] == "ai:ai_approver", str(executed))
        saved = gate.get(rid_small)
        check("审批单状态已变 approved", saved["status"] == "approved",
              str(saved.get("status")))
        check("decided_by 落 AI 身份（审计可区分人批/机批）",
              saved["decided_by"] == "ai:ai_approver", str(saved.get("decided_by")))
        check("decided_role 落 ai_approver",
              saved["decided_role"] == "ai_approver")
        check("审批结果回填了执行结果",
              (saved.get("result") or {}).get("ok") is True)

        # ---- 第二道闸：绕过裁决直接调 decide，也必须被权限链拒绝 ----
        rid_big = gate.request("purchase_order.approve",
                               {"tenant_id": "default", "po_id": 105,
                                "amount": 120000.0, "supplier_id": 7},
                               reason="大额采购", requested_by="purchase_user",
                               requested_role="purchase_user")
        err = None
        try:
            gate.decide(rid_big, approve=True, decided_by="ai:ai_approver",
                        decided_role="ai_approver")
        except PermissionError as e:
            err = str(e)
        check("AI 以 level=2 批 12 万单 → 被权限链拒绝",
              err is not None and "等级不足" in err, err or "居然通过了")
        check("被拒后单据仍为 pending",
              gate.get(rid_big)["status"] == "pending",
              gate.get(rid_big)["status"])

        # 换成真人 level=3 就能批（证明不是把功能锁死了）
        ok = gate.decide(rid_big, approve=True, decided_by="王总",
                         decided_role="super_admin")
        check("super_admin（level=3）能正常批", ok["status"] == "approved")

        # ---- 第三道闸：AI 不能给自己发起的单审批 ----
        rid_self = gate.request("purchase_order.approve",
                                {"tenant_id": "default", "po_id": 106,
                                 "amount": 1000.0, "supplier_id": 7},
                                requested_by="ai:ai_approver",
                                requested_role="ai_approver")
        err2 = None
        try:
            gate.decide(rid_self, approve=True, decided_by="ai:ai_approver",
                        decided_role="ai_approver")
        except PermissionError as e:
            err2 = str(e)
        check("AI 不能自审自批（禁自审自批规则对 AI 同样生效）",
              err2 is not None and "自审自批" in err2, err2 or "居然通过了")

        # ---- 不支持的动作必须明确拒绝，不许假装能审 ----
        gate.register_executor("inventory.adjust", lambda p, actor: {"ok": 1})
        rid_adj = gate.request("inventory.adjust",
                               {"tenant_id": "default", "warehouse_id": 1,
                                "product_id": 11, "delta": -5},
                               requested_by="warehouse_user",
                               requested_role="warehouse_user")
        err3 = None
        try:
            gate.ai_review(rid_adj, flow=flow)
        except ValueError as e:
            err3 = str(e)
        check("非采购审批动作 → 明确拒绝（不编指标）",
              err3 is not None and "暂不支持" in err3, err3 or "居然审了")

        # ---- 已处理的单不能再审 ----
        err4 = None
        try:
            gate.ai_review(rid_small, flow=flow)
        except ValueError as e:
            err4 = str(e)
        check("已 approved 的单再审 → 报错",
              err4 is not None and "已处理" in err4, err4 or "居然审了")

        # ---- 缺 po_id 必须报错，不能猜 ----
        rid_nopo = gate.request("purchase_order.approve",
                                {"tenant_id": "default", "amount": 5000.0},
                                requested_by="purchase_user",
                                requested_role="purchase_user")
        err5 = None
        try:
            gate.ai_review(rid_nopo, flow=flow)
        except ValueError as e:
            err5 = str(e)
        check("payload 缺 po_id → 报错",
              err5 is not None and "po_id" in err5, err5 or "居然审了")


# ===========================================================================
# 10. CLI 自检
# ===========================================================================
def t_cli():
    section("10. CLI --check")
    rc = A._check(A.load_policy())
    check("--check 返回 0", rc == 0)


# ===========================================================================
# 11. POST /ai/pack（Dify 工作流拉数字包的端点）
# ===========================================================================
def _free_port() -> int:
    """占一个空闲端口再释放（Windows 下 SO_REUSEADDR 会静默抢占，
    所以必须显式指定端口，不能指望 port=0）。"""
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class _FakeBuilder:
    """
    替身 DataPackBuilder：内部仍调**原始真实实现**算指标，只接管可用性开关。

    注意 `_real` 必须在打补丁前存好——否则 build() 里再取
    `A.DataPackBuilder` 拿到的是被替换后的自己，直接无限递归。

    `_q` 也要委托：HistoryDigest 会拿着这个 builder 实例再查历史，
    没有 `_q` 就直接 AttributeError。
    """
    mode = "ok"          # ok / nodb
    _real = None

    def __init__(self, policy=None, tenant_id="default", q=None, verbose=False):
        self.policy = policy or A.load_policy()
        self.tenant = tenant_id
        self.tenant_id = tenant_id
        self.available = (_FakeBuilder.mode != "nodb")
        self.error = "模拟：连不上 MySQL"
        self._inner = None

    def build(self, po_id):
        self._inner = _FakeBuilder._real(self.policy, self.tenant,
                                        q=good_case())
        return self._inner.build(po_id)

    def _q(self, sql, params=()):
        if self._inner is None:
            self._inner = _FakeBuilder._real(self.policy, self.tenant,
                                            q=good_case())
        return self._inner._q(sql, params)


def t_ai_pack_endpoint():
    section("11. POST /ai/pack：Dify 工作流拉数字包")
    import threading
    import urllib.error
    import urllib.request
    import dify_kb_server as KB

    raw = KB._load_config(KB.DEFAULT_CONFIG_PATH)
    cfg = raw if isinstance(raw, KB.BridgeConfig) else KB.BridgeConfig(raw)
    cfg.port = _free_port()
    os.environ["DIFY_KB_API_KEY"] = "test-bridge-key"

    class _Svc:
        """假 service：/ai/pack 用到 note_error；/health 还要 stats 与 _vector_db。"""
        _vector_db = None

        def __init__(self):
            self.stats = {"requests": 0, "records": 0, "errors": 0,
                          "last_error": "", "started_at": time.time()}

        def note_error(self, msg):
            self.stats["errors"] += 1
            self.stats["last_error"] = msg

    svc = _Svc()
    httpd = KB._build_server(cfg, svc)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()

    base = f"http://127.0.0.1:{cfg.port}"

    def post(path, body, key="test-bridge-key"):
        data = json.dumps(body).encode() if not isinstance(body, bytes) else body
        req = urllib.request.Request(
            base + path, data=data, method="POST",
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {key}"} if key else {})})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode() or "{}")

    _orig = A.DataPackBuilder
    _FakeBuilder._real = _orig
    A.DataPackBuilder = _FakeBuilder
    try:
        # ---- 鉴权 ----
        st, body = post("/ai/pack", {"po_id": 101}, key=None)
        check("无 Authorization 头 → 403/1001",
              st == 403 and body.get("error_code") == 1001, f"{st} {body}")
        st, body = post("/ai/pack", {"po_id": 101}, key="wrong")
        check("密钥不对 → 403/1002",
              st == 403 and body.get("error_code") == 1002, f"{st} {body}")

        # ---- 入参校验 ----
        st, body = post("/ai/pack", {"po_id": "abc"})
        check("po_id 非整数 → 400", st == 400 and "整数" in body.get("error", ""),
              f"{st} {body}")
        st, body = post("/ai/pack", {"po_id": 101, "knowledge_id": "kb-hack"})
        check("knowledge_id 不在白名单 → 404/2001",
              st == 404 and body.get("error_code") == 2001, f"{st} {body}")

        # ---- 正常路径 ----
        st, body = post("/ai/pack", {"po_id": 101})
        check("正常取数 → 200", st == 200 and body.get("ok") is True, f"{st} {body}")
        m = body.get("metrics") or {}
        check("返回 12 个指标里的金额", m.get("amount") == 16000.0, str(m.get("amount")))
        check("返回规则结论（low/0分）",
              body["rule"]["risk_level"] == "low"
              and body["rule"]["risk_score"] == 0, str(body["rule"]))
        # ---- 多站路线 ----
        route = body.get("route") or {}
        nodes = route.get("nodes") or []
        check("返回审批路线（16000 元走单站部门经理）",
              len(nodes) == 1 and nodes[0]["node_id"] == "dept_manager",
              str([n["node_id"] for n in nodes]))
        check("路线节点带职责与关注指标",
              nodes and nodes[0].get("duty") and nodes[0].get("focus_metrics"),
              str(nodes[0] if nodes else {}))
        check("每站都有自己的 material（Dify 迭代节点直接用）",
              nodes and isinstance(nodes[0].get("material"), str)
              and "16,000.00" in nodes[0]["material"],
              (nodes[0].get("material") or "")[:120] if nodes else "")
        check("部门经理站的材料含用量与库存天数",
              nodes and "采购量 / 近30天用量" in nodes[0]["material"]
              and "可用天数" in nodes[0]["material"])
        check("部门经理材料**不含**价格偏离（职责隔离）",
              nodes and "单价偏离历史均价" not in nodes[0]["material"])
        check("返回采购历史档案",
              (body.get("history") or {}).get("supplier_share_pct") == 62.5,
              str((body.get("history") or {}).get("supplier_share_pct")))
        check("历史档案含价格走势",
              (body.get("history") or {}).get("price_trend", {}).get("last_price")
              == 79.0,
              str((body.get("history") or {}).get("price_trend", {})))
        check("返回 gate 边界信息（含路线最高等级）",
              body["gate"]["effective_level"] == 2
              and body["gate"]["ai_level"] == 2
              and body["gate"]["max_auto_amount"] == 50000.0, str(body["gate"]))
        check("16000 元单 auto_eligible=True", body["gate"]["auto_eligible"] is True)

        # ---- DB 不可用：必须 503，不能返回空 metrics 让 LLM 瞎编 ----
        _FakeBuilder.mode = "nodb"
        st, body = post("/ai/pack", {"po_id": 101})
        check("数据库不可用 → 503（不返回空指标）",
              st == 503 and body.get("ok") is False, f"{st} {body}")
        check("503 文案说明是取数失败",
              "数据库不可用" in (body.get("error") or ""), str(body.get("error")))
        _FakeBuilder.mode = "ok"

        # ---- 未知端点 ----
        st, body = post("/nope", {"po_id": 101})
        check("未知端点 → 404 且提示可用端点",
              st == 404 and "/ai/pack" in json.dumps(body, ensure_ascii=False),
              str(body))

        # ---- health 里能看到端点信息 ----
        with urllib.request.urlopen(base + "/health", timeout=10) as r:
            h = json.loads(r.read().decode())
        check("GET /health 正常", h.get("status") == "ok", str(h)[:80])
        check("health 报告密钥已配", h.get("api_key_configured") is True)
    finally:
        A.DataPackBuilder = _orig
        httpd.shutdown()
        httpd.server_close()


# 在任何测试跑之前就把「原始实现」钉住：_FakeBuilder.build() 内部要调它，
# 否则先跑的测试会因为 _real 还是 None 而 TypeError。
_FakeBuilder._real = A.DataPackBuilder


class _AmountBuilder(_FakeBuilder):
    """按指定金额造一张单（用来跑不同金额分段的多站路线）。"""
    amount = 80000.0

    def build(self, po_id):
        qty = int(round(self.amount / 80.0))
        # issued30 与 qty 对齐，避免用量倍数把规则层打成 high，掩盖要测的路径
        fake = good_case(issued30=1000.0, stock={"stock": 0, "reserved": 0})
        fake.po["amount"] = self.amount
        fake.po["po_no"] = "PO-202610-0199"
        fake.items[0]["quantity"] = qty
        self._inner = _FakeBuilder._real(self.policy, self.tenant, q=fake)
        return self._inner.build(po_id)


# ===========================================================================
# 12. 采购历史档案
# ===========================================================================
def t_history():
    section("12. HistoryDigest：把历史摊成 AI 能引用的数字")
    pol = A.load_policy()
    b = A.DataPackBuilder(q=good_case())
    pack = b.build(101)
    h = A.HistoryDigest(b, pol).build(pack)

    check("标注了以哪条明细为主料", "不锈钢螺栓" in h["_material"], h["_material"])
    check("同料历史 2 条", len(h["material_purchases"]) == 2)
    check("历史明细带供应商名",
          h["material_purchases"][0]["supplier"] == "恒达标准件",
          str(h["material_purchases"][0]))
    # 真实 SQL 是 ORDER BY id DESC，reversed 后应是时间正序
    check("价格序列按时间正序（旧→新）",
          [s["unit_price"] for s in h["price_trend"]["series"]] == [76.5, 79.0],
          str([s["unit_price"] for s in h["price_trend"]["series"]]))
    check("上次成交价 = 79.0", h["price_trend"]["last_price"] == 79.0)
    check("本次相比上次涨 1.27%",
          h["price_trend"]["direction"] == "涨"
          and h["price_trend"]["delta_vs_last_pct"] == 1.27,
          str(h["price_trend"]))
    check("供应商近期 2 单", len(h["supplier_recent_orders"]) == 2)
    check("供应商集中度 62.5%", h["supplier_share_pct"] == 62.5)
    check("季度金额取到", h["quarterly_amount"]["this_year"] == 520000.0)
    check("月度领用 3 个月", len(h["issued_trend"]) == 3,
          str(h["issued_trend"]))
    check("无取数异常", h["_error"] is None, str(h["_error"]))

    # 无历史时必须是空而不是编造
    b2 = A.DataPackBuilder(q=good_case(material_purchases=[],
                                       supplier_orders=[],
                                       supplier_share=None,
                                       issued_rows=[]))
    h2 = A.HistoryDigest(b2, pol).build(b2.build(101))
    check("无历史成交 → 空列表", h2["material_purchases"] == [])
    check("无历史成交 → 集中度为 None（不是 0）",
          h2["supplier_share_pct"] is None)
    check("无历史成交 → 方向标「无历史」",
          h2["price_trend"]["direction"] == "无历史", str(h2["price_trend"]))


# ===========================================================================
# 13. 审批路线规划
# ===========================================================================
def t_route():
    section("13. RoutePlanner：按金额分段 + 按历史插站")
    pol = A.load_policy()
    planner = A.RoutePlanner(pol)

    def plan(amount, **metric_over):
        pack = A.DataPackBuilder(q=good_case()).build(101)
        pack["metrics"]["amount"] = amount
        pack["metrics"].update(metric_over)
        rule = A.RuleEngine(pol).score(pack)
        return planner.plan(pack, rule), rule, pack

    ids = lambda r: [n["node_id"] for n in r["nodes"]]

    r, _, _ = plan(16000)
    check("1.6 万 → 单站部门经理", ids(r) == ["dept_manager"], str(ids(r)))
    r, _, _ = plan(80000)
    check("8 万 → 部门经理 + 财务", ids(r) == ["dept_manager", "finance"],
          str(ids(r)))
    r, _, _ = plan(250000)
    check("25 万 → 4 站",
          ids(r) == ["dept_manager", "purchase_lead", "finance", "gm"],
          str(ids(r)))
    check("25 万路线最高等级 = 3（总经理站）", r["max_level"] == 3)

    # ---- 插站：拆单 ----
    r, _, _ = plan(98000, split_bypass=True, split_amount_pct=140.0)
    check("拆单 → 合规复核插在部门经理之后",
          ids(r)[:2] == ["dept_manager", "compliance"], str(ids(r)))
    check("插站原因写明「拆单」",
          any("拆单" in n["reason"] for n in r["nodes"]),
          str([n["reason"] for n in r["nodes"]]))
    check("拆单插站记进 inserted", any(x["node"] == "compliance"
                                       for x in r["inserted"]))

    # ---- 插站：新供应商 ----
    r, _, _ = plan(16000, supplier_age_days=10, supplier_po_90d=0)
    check("新供应商 → 插准入复核",
          "supplier_onboard" in ids(r), str(ids(r)))

    # ---- 插站：价格异动透过规则层的 high finding 触发 ----
    r, rule, _ = plan(16000, price_deviation_pct=45.0)
    check("价格异动 → 规则给出 high",
          any(f["level"] == "high" for f in rule["findings"]))
    check("价格异动 → 插采购主管",
          "purchase_lead" in ids(r), str(ids(r)))
    check("信号 price_spike = True", r["signals"]["price_spike"] is True)

    # ---- 插站去重：财务已在路线上就不再插，但原因要合并 ----
    r, _, _ = plan(80000, budget_used_pct=95.0)
    check("财务已在路线 → 不重复插", ids(r).count("finance") == 1, str(ids(r)))
    fin = next(n for n in r["nodes"] if n["node_id"] == "finance")
    check("财务站原因合并了预算告警", "预算" in fin["reason"], fin["reason"])

    # ---- 每站都带完整定义 ----
    r, _, _ = plan(250000)
    check("每站都有职责、话术、关注指标",
          all(n["duty"] and n["hint"] and n["focus_metrics"] for n in r["nodes"]))
    check("无配置告警", r["warnings"] == [], str(r["warnings"]))

    # ---- 配置自洽：路线/插站引用的节点必须都有定义 ----
    nodes_def = pol.get("nodes") or {}
    check("yaml 路线引用的节点都有定义",
          all(str(x) in nodes_def for rr in (pol.get("routes") or [])
              for x in (rr.get("nodes") or [])))
    check("yaml 插站引用的节点都有定义",
          all(str(ir.get("node")) in nodes_def
              for ir in (pol.get("insert_rules") or [])))


# ===========================================================================
# 14. 每站材料必须职责隔离
# ===========================================================================
def t_node_material():
    section("14. render_node_material：四个岗位不能看到同一份材料")
    pol = A.load_policy()
    b = A.DataPackBuilder(q=good_case())
    pack = b.build(101)
    rule = A.RuleEngine(pol).score(pack)
    h = A.HistoryDigest(b, pol).build(pack)

    def mat(nid):
        node = dict(pol["nodes"][nid])
        node["node_id"] = nid
        return A.render_node_material(pack, h, rule, node)

    dm, pl_, fin = mat("dept_manager"), mat("purchase_lead"), mat("finance")

    check("部门经理看到用量倍数", "采购量 / 近30天用量" in dm)
    check("部门经理看到月度领用", "月度领用" in dm)
    check("部门经理**看不到**价格偏离（职责隔离）",
          "单价偏离历史均价" not in dm)
    check("部门经理看不到预算占用", "本月预算占用率" not in dm)

    check("采购主管看到价格偏离", "单价偏离历史均价" in pl_)
    check("采购主管看到历史成交单价序列", "历史成交单价序列" in pl_)
    check("采购主管看到供应商集中度", "62.5%" in pl_)
    check("采购主管看不到库存天数", "可用天数" not in pl_)

    check("财务看到预算占用", "本月预算占用率" in fin)
    check("财务看到季度金额", "本季度" in fin)
    check("财务看不到用量倍数", "采购量 / 近30天用量" not in fin)

    check("三段材料互不相同", len({dm, pl_, fin}) == 3)

    # 一票否决所有站都必须知道
    bad_pack = A.DataPackBuilder(q=good_case()).build(101)
    bad_pack["metrics"]["supplier_active"] = 0
    bad_rule = A.RuleEngine(pol).score(bad_pack)
    node = dict(pol["nodes"]["dept_manager"])
    node["node_id"] = "dept_manager"
    txt = A.render_node_material(bad_pack, h, bad_rule, node)
    check("全局一票否决会透传给每一站", "供应商已停用" in txt)


# ===========================================================================
# 15. 单站建议解析
# ===========================================================================
def t_parse_advice():
    section("15. parse_node_advice：单站建议的严格解析")
    good = json.dumps({"advice": "agree", "confidence": 0.9,
                       "key_points": ["倍数=1.33 → 合理"], "conditions": [],
                       "questions": [], "one_liner": "可批"})
    v, e = A.parse_node_advice(good)
    check("合法建议解析成功", e == "" and v["advice"] == "agree", e)
    v, e = A.parse_node_advice(f"```json\n{good}\n```")
    check("剥 Markdown 围栏", e == "" and v["confidence"] == 0.9, e)
    v, e = A.parse_node_advice(f"我的意见：\n{good}")
    check("剥前置寒暄", e == "", e)
    v, e = A.parse_node_advice(json.dumps(
        {"advice": "maybe", "confidence": 0.9, "key_points": []}))
    check("advice 不在白名单 → 报错", v is None and "白名单" in e, e)
    v, e = A.parse_node_advice(json.dumps(
        {"advice": "agree", "confidence": 1.5, "key_points": []}))
    check("confidence 越界 → 报错", v is None and "越界" in e, e)
    v, e = A.parse_node_advice(json.dumps(
        {"advice": "agree", "confidence": "0.9", "key_points": []}))
    check("confidence 是字符串数字也能接受（float 可转）",
          e == "" and v["confidence"] == 0.9, e)
    v, e = A.parse_node_advice(json.dumps(
        {"advice": "agree", "confidence": 0.9, "key_points": "a"}))
    check("key_points 非数组 → 报错", v is None and "不是数组" in e, e)
    v, e = A.parse_node_advice("")
    check("空内容 → 报错", v is None and "空内容" in e, e)
    v, e = A.parse_node_advice("不是JSON")
    check("非 JSON → 报错", v is None and "找不到 JSON" in e, e)


# ===========================================================================
# 16. 多站裁决
# ===========================================================================
def t_arbitrate_route():
    section("16. arbitrate_route：四站全同意才自动批")
    pol = A.load_policy()
    pack = A.DataPackBuilder(q=good_case()).build(101)
    rule = A.RuleEngine(pol).score(pack)
    route = A.RoutePlanner(pol).plan(pack, rule)     # 1.6 万 → 单站

    def adv(nid="dept_manager", title="部门经理", **kw):
        d = {"node_id": nid, "title": title, "level": 2, "ok": True,
             "advice": "agree", "confidence": 0.9, "one_liner": "可批",
             "key_points": ["倍数=1.33 → 合理"], "conditions": [],
             "questions": []}
        d.update(kw)
        return d

    v = A.arbitrate_route(pack, rule, route, [adv()], pol, 2)
    check("单站同意 → auto_approve", v["action"] == "auto_approve", v["reason"])
    check("自动批时 stage 标「全部」", v["stage"] == "全部", v["stage"])

    v = A.arbitrate_route(pack, rule, route, [], pol, 2)
    check("没有任何站意见 → degraded", v["action"] == "degraded", v["action"])

    v = A.arbitrate_route(pack, rule, route,
                          [adv(ok=False, error="调用超时")], pol, 2)
    check("有站失败 → degraded", v["action"] == "degraded", v["action"])
    check("degraded 点名哪一站没意见",
          any("部门经理" in b for b in v["blockers"]), str(v["blockers"]))

    v = A.arbitrate_route(pack, rule, route,
                          [adv(advice="oppose", one_liner="一次买太多")], pol, 2)
    check("某站反对 → recommend_reject", v["action"] == "recommend_reject",
          v["action"])
    check("理由点名是哪一站反对", "部门经理" in v["reason"], v["reason"])
    check("blockers 带该站结论", any("一次买太多" in b for b in v["blockers"]))

    v = A.arbitrate_route(pack, rule, route,
                          [adv(advice="need_info", questions=["用在哪张工单"])],
                          pol, 2)
    check("某站资料不足 → human_review", v["action"] == "human_review",
          v["action"])
    check("blockers 转述待问问题", any("用在哪张工单" in b for b in v["blockers"]))

    v = A.arbitrate_route(pack, rule, route,
                          [adv(advice="agree_with_conditions",
                               conditions=["首批验货"])], pol, 2)
    check("有条件同意 → human_review", v["action"] == "human_review")
    check("blockers 带条件", any("首批验货" in b for b in v["blockers"]))

    v = A.arbitrate_route(pack, rule, route, [adv(confidence=0.5)], pol, 2)
    check("平均置信度不足 → human_review", v["action"] == "human_review")

    # 权限优先于 AI 意见：AI 全同意也不行
    v = A.arbitrate_route(pack, rule, route, [adv()], pol, 3)
    check("等级不足 → 转人工（哪怕 AI 全同意）",
          v["action"] == "human_review" and v["stage"] == "权限", v["stage"])

    # 路线最高等级也会拦住
    big = A.DataPackBuilder(q=good_case()).build(101)
    big["metrics"]["amount"] = 250000
    big_rule = A.RuleEngine(pol).score(big)
    broute = A.RoutePlanner(pol).plan(big, big_rule)
    check("25 万路线最高等级 3", broute["max_level"] == 3)
    allok = [adv(n["node_id"], title=n["title"]) for n in broute["nodes"]]
    v = A.arbitrate_route(big, big_rule, broute, allok, pol, 2)
    check("路线本身要求 level=3 → 转人工", v["action"] == "human_review")
    check("理由说明是路线等级不够",
          any("路线" in b for b in v["blockers"]), str(v["blockers"]))

    # 规则层 high 不因多站而放宽
    hp = A.DataPackBuilder(q=good_case()).build(101)
    hp["metrics"]["price_deviation_pct"] = 45.0
    hrule = A.RuleEngine(pol).score(hp)
    v = A.arbitrate_route(hp, hrule, route, [adv()], pol, 2)
    check("规则 high → recommend_reject（AI 全同意也不放）",
          v["action"] == "recommend_reject", v["action"])


# ===========================================================================
# 17. 交接单：转人工也要给建议
# ===========================================================================
def t_handover():
    section("17. build_handover：转人工不是丢球，是交一份判断依据")
    pol = A.load_policy()
    pack = A.DataPackBuilder(q=good_case()).build(101)
    row = {
        "action": "human_review", "stage": "财务审核",
        "blockers": ["【财务审核】有条件同意：批完剩余额度只剩 3 万"],
        "metrics": pack["metrics"],
        "advices": [
            {"node_id": "dept_manager", "title": "部门经理", "ok": True,
             "advice": "agree", "confidence": 0.9, "one_liner": "用量合理",
             "key_points": ["采购量/用量=1.33 → 合理"], "conditions": [],
             "questions": []},
            {"node_id": "finance", "title": "财务审核", "ok": True,
             "advice": "agree_with_conditions", "confidence": 0.85,
             "one_liner": "额度紧张", "key_points": ["预算占用=27% → 在阈值内"],
             "conditions": ["批完剩余额度写清"], "questions": ["是否分两次付款"]},
        ],
    }
    h = A.build_handover(row, pol)
    check("结论说明是转人工", "转人工" in h["conclusion"], h["conclusion"])
    check("平均置信度 = 0.875", h["avg_confidence"] == 0.875, str(h["avg_confidence"]))
    check("保留卡在哪一站", h["stage"] == "财务审核")
    check("两站建议都在", len(h["advices"]) == 2)
    check("汇总了需要采购员回答的问题",
          any("分两次付款" in q for q in h["questions_for_requester"]),
          str(h["questions_for_requester"]))
    check("问题带站名归属",
          any("财务审核" in q for q in h["questions_for_requester"]))
    check("给出可执行的处置建议",
          bool(h["recommended_action"]) and "stage" in h["recommended_action"],
          h["recommended_action"])
    check("why_uncertain 原样带走 blockers", h["why_uncertain"] == row["blockers"])

    txt = A.format_handover(h)
    check("可读版带站点标记", "[有条件同意]" in txt and "财务审核" in txt)
    check("可读版带处置建议段", "建议你这样做" in txt)
    check("可读版带待确认问题段", "需要采购员回答的问题" in txt)

    # 其他 action 的结论措辞
    for act, kw in [("recommend_reject", "建议拒批"),
                    ("degraded", "未能给出完整意见"),
                    ("auto_approve", "自动批准")]:
        hh = A.build_handover({"action": act, "stage": "", "blockers": [],
                               "metrics": {}, "advices": []}, pol)
        check(f"{act} 的结论措辞正确", kw in hh["conclusion"], hh["conclusion"])
    check("空交接单有兜底文案", "没有 AI 交接单" in A.format_handover({}))


# ===========================================================================
# 17b. 网关全链失败（没装 Dify、本地模型也不可用）时的行为
# ===========================================================================
class _DegradedGateway:
    """网关「全链失败」时的替身：不抛异常，返回 degraded_reply 文案 + model=<degraded>。

    这不是编出来的场景——llm_gateway 在全链失败且配置了 degraded_reply 时
    就是这么返回的（`chat_detailed` 末尾那段）。ai_approval 必须认出来，
    否则会把「模型连不上」误诊成「模型乱输出」。
    """

    def __init__(self):
        self.calls = 0

    def chat_detailed(self, system_prompt, user_prompt, task="default",
                      user="anonymous"):
        from llm_gateway import LLMResponse
        self.calls += 1
        return LLMResponse(text="AI 服务暂时不可用，请稍后再试。",
                           model="<degraded>", provider="<none>")


def t_degraded_gateway():
    section("17b. 网关全链失败 → 各站明确报「网关失败」而不是「JSON 不对」")
    pol = A.load_policy()
    pack = A.DataPackBuilder(q=good_case()).build(101)     # 1.6 万 → 单站
    rule = A.RuleEngine(pol).score(pack)
    route = A.RoutePlanner(pol).plan(pack, rule)
    hist = A.HistoryDigest(A.DataPackBuilder(q=good_case()), pol).build(pack)

    gw = _DegradedGateway()
    adv = A.NodeAdvisor(policy=pol, gateway=gw, verbose=False)
    advices = adv.advise(pack, hist, rule, route)

    check("网关真被调用了（不是空转）", gw.calls == len(route["nodes"]),
          f"calls={gw.calls}")
    check("所有站 ok=False", all(a["ok"] is False for a in advices))
    check("错误信息点名「网关全链失败」",
          all("全链失败" in (a.get("error") or "") for a in advices),
          str([a.get("error") for a in advices]))
    check("错误信息不再误报「找不到 JSON」",
          not any("找不到 JSON" in (a.get("error") or "") for a in advices))

    v = A.arbitrate_route(pack, rule, route, advices, pol, required_level=2)
    check("裁决降级（不是自动批，也不是建议拒）", v["action"] == "degraded",
          v["action"])
    check("降级 stage 标 AI", v["stage"] == "AI", v["stage"])

    row = {"action": v["action"], "stage": v["stage"], "blockers": v["blockers"],
           "metrics": pack["metrics"], "advices": advices}
    h = A.build_handover(row, pol)
    txt = A.format_handover(h)
    check("交接单保留 ok/error 字段",
          h["advices"][0].get("ok") is False
          and "全链失败" in (h["advices"][0].get("error") or ""),
          str(h["advices"][0].get("error")))
    check("交接单把失败原因打出来（不只是一句「无结论」）",
          "原因：" in txt and "全链失败" in txt)
    check("交接单仍给了人工处置建议", "建议你这样做" in txt)


# ===========================================================================
# 18. 端到端：多站 + 站级反对 + 交接单
# ===========================================================================
def t_e2e_multistage():
    section("18. 端到端：8 万单走两站，财务反对 → 交接单点名财务")
    from approval import ApprovalGate

    pol = A.load_policy()
    check("测试策略：把自动批额度抬到 10 万（否则会先被额度拦住）",
          True)

    with _NoDb():
        gate = ApprovalGate(executors={"purchase_order.approve":
                                       lambda p, actor: {"ok": 1}},
                            verbose=False)
        rid = gate.request("purchase_order.approve",
                           {"tenant_id": "default", "po_id": 199,
                            "amount": 80000.0, "supplier_id": 7},
                           reason="季度备料", requested_by="purchase_user",
                           requested_role="purchase_user")

        flow = A.AIApprovalFlow.__new__(A.AIApprovalFlow)
        flow.policy = json.loads(json.dumps(pol))
        flow.policy["limits"]["max_auto_amount"] = 100000.0
        flow.verbose = False
        flow.gate = gate
        flow.rule = A.RuleEngine(flow.policy)
        flow.planner = A.RoutePlanner(flow.policy)
        flow._builder = _AmountBuilder(flow.policy)
        flow._reviewer = None
        flow._advisor = _FakeAdvisor(per_node={
            "finance": {"advice": "agree_with_conditions", "confidence": 0.82,
                        "one_liner": "预算可批，但剩余额度只剩 3 万",
                        "conditions": ["批完把剩余额度报财务备案"],
                        "questions": ["是否可拆成两次到货"]}})
        flow.available = False
        flow._pymysql = None
        flow._conn_kw = None

        row = flow.review_po(199, tenant_id="default", request_id=rid,
                             dry_run=True)

        check("8 万单走两站（部门经理 + 财务）",
              [n["node_id"] for n in row["route"]["nodes"]]
              == ["dept_manager", "finance"],
              str([n["node_id"] for n in row["route"]["nodes"]]))
        check("两站都有结论",
              len(row["advices"]) == 2
              and all(a["advice"] for a in row["advices"]),
              str([(a["node_id"], a.get("advice")) for a in row["advices"]]))
        check("每站都有一句话结论",
              all(a.get("one_liner") for a in row["advices"]),
              str([a.get("one_liner") for a in row["advices"]]))
        check("财务有条件同意 → 转人工",
              row["action"] == "human_review", row["action"])
        check("卡住环节 = 财务审核", row["stage"] == "财务审核", row["stage"])
        check("reason 里带上了 AI 建议",
              "AI建议" in row["reason"], row["reason"])
        check("交接单存在", bool(row.get("handover")))
        check("交接单含两站建议", len(row["handover"]["advices"]) == 2)
        check("交接单带财务附的条件",
              any("备案" in c for c in
                  row["handover"]["advices"][1]["conditions"]),
              str(row["handover"]["advices"][1]["conditions"]))
        check("交接单汇总了待问问题",
              any("拆成两次" in q for q in
                  row["handover"]["questions_for_requester"]),
              str(row["handover"]["questions_for_requester"]))
        check("交接单给了处置建议",
              "财务审核" in row["handover"]["recommended_action"]
              or "条件" in row["handover"]["recommended_action"],
              row["handover"]["recommended_action"])
        check("转人工时单据仍是 pending",
              gate.get(rid)["status"] == "pending", gate.get(rid)["status"])

        # 单站查询：审批人只看自己那一站
        one = flow.review_po(199, tenant_id="default", request_id=rid,
                             dry_run=True, node_id="finance")
        check("单站查询只返回一站", len(one["advices"]) == 1
              and one["advices"][0]["node_id"] == "finance")
        check("单站结果标 partial", one["route"].get("partial") is True)
        check("单站结果不会自动批",
              one["action"] == "human_review", one["action"])

        err = None
        try:
            flow.review_po(199, tenant_id="default", dry_run=True,
                           node_id="compliance")
        except ValueError as e:
            err = str(e)
        check("查一个不在路线上的节点 → 报错并列出本单路线",
              err is not None and "不在本单路线" in err, err or "居然通过了")

        # 全部同意时可自动批（额度内、路线 level=2）
        flow._advisor = _FakeAdvisor()
        ok = flow.review_po(199, tenant_id="default", request_id=rid)
        check("两站全同意 → auto_approve", ok["action"] == "auto_approve",
              ok["reason"])
        check("自动批后单据 approved", gate.get(rid)["status"] == "approved")


# ===========================================================================
def main():
    tests = [t_config, t_metrics_healthy, t_split_order, t_missing, t_rules,
             t_parse, t_arbitrate, t_render, t_real_chain, t_history, t_route,
             t_node_material, t_parse_advice, t_arbitrate_route, t_handover,
             t_degraded_gateway, t_e2e_multistage, t_ai_pack_endpoint, t_cli]
    for t in tests:
        try:
            t()
        except Exception:
            print(f"\n[EXCEPTION] {t.__name__}")
            traceback.print_exc()
            FAIL.append(f"{t.__name__} 抛异常")

    print("\n" + "=" * 78)
    total = len(PASS) + len(FAIL)
    print(f"结果：{len(PASS)}/{total} 通过，{len(FAIL)} 失败")
    if FAIL:
        for f in FAIL:
            print(f"  ✗ {f}")
    print("=" * 78)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
