"""
================================================================================
 ai_approval.py — AI 自动审批流（物料采购审核）
================================================================================

 为什么要有这个模块
 ----------------
 现有的 `approval.py::ApprovalGate` 是一条完整的**人工**审批链：
 申请 → pending → 人批 → 执行。它解决的是「谁都别乱动」，
 但没解决「线上一天几十张采购单，人审不过来就点同意」这个真实问题——
 审批一旦退化成橡皮图章，风控等于没有。

 这个模块补的是审批门**前面**的一层：AI 预审。
 但 AI 不能凭感觉审，审批必须落在数字上。所以链路是：

     采购单 submitted
        ↓  ① 取数（DataPackBuilder，真实 SQL，12 个硬指标）
     数字包 pack（每个指标都能回源到具体表和字段）
        ↓  ② 确定性规则（RuleEngine，不依赖 LLM，随时可复现）
     findings + risk_score + hard_block
        ↓  ③ AI 研判（DifyReviewer → llm_gateway → Dify 工作流）
     结构化 verdict（decision / risk_level / confidence / key_factors）
        ↓  ④ 裁决（arbitrate，纯函数）
     auto_approve / human_review / recommend_reject / degraded
        ↓  ⑤ 落库 + 必要时调 ApprovalGate.decide()

 四层分开的理由（这是本模块最重要的设计决策）
 --------------------------------------------
 1. **算数不用 LLM**。金额、单价偏离、采购量倍数、可用天数这些必须可复现：
    同一张单跑两次、换台机器跑，结果必须一模一样。LLM 做不到这点，
    所以它不碰数，只读数。
 2. **LLM 不参与一票否决**。硬规则（供应商停用、拆单规避、数据缺失）
    在规则层就判死，AI 只能补充理由，不能推翻。否则「AI 觉得还行」
    就成了绕开风控的后门。
 3. **AI 建议拒绝 ≠ 自动拒绝**。误杀一张正常单的代价（停工等料）
    远高于漏放一张异常单（人还有二审），所以 high 风险也只落
    「建议拒绝 + 强制人工复核」。
 4. **AI 不越权**。它能自动批的额度 = min(自己角色的审批等级额度,
    limits.max_auto_amount)。采购 >10 万须 level=3（access_rules.yaml 定稿），
    AI 角色是 level=2，所以 10 万以上它连申请自动批的资格都没有——
    这一条不是本模块自己的规矩，是复用 ApprovalGate.decide() 的权限链，
    改配置也绕不过去。

 配置
 ----
 策略全在 config/ai_approval.yaml（阈值/权重/额度/门槛）；
 审批等级的权威源仍是 config/access_rules.yaml 的 approval_rules。

 CLI
 ---
   python ai_approval.py --check                    # 自检配置（不连库）
   python ai_approval.py --review 12                # 对审批单 #12 做 AI 预审
   python ai_approval.py --dry-run 12               # 只算不写库
   python ai_approval.py --sql-check 12             # 只跑取数 SQL，打印数字包
================================================================================
"""

import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from erp_common import audit

POLICY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "config", "ai_approval.yaml")

# ---------------------------------------------------------------------------
# 兜底策略：yaml 缺失/损坏时用它（额度与等级一律取严，绝不放宽）
# ---------------------------------------------------------------------------
_DEFAULT_POLICY: Dict[str, Any] = {
    "ai_approver": {"role": "ai_approver", "level": 2,
                    "display_name": "AI 审批员（Dify 工作流）"},
    "limits": {"max_auto_amount": 50000, "min_ai_confidence": 0.80,
               "require_no_high_finding": True, "max_top_k": 6},
    "budget": {"default": {"monthly": 800000}, "tenants": {}},
    "thresholds": {
        "price_deviation_pct": {"medium": 8.0, "high": 20.0},
        "unit_price_vs_cost_pct": {"medium": 10.0, "high": 25.0},
        "qty_ratio": {"medium": 3.0, "high": 6.0},
        "stock_days": {"medium": 60.0, "high": 120.0},
        "supplier_age_days": {"medium": 90.0, "high": 30.0},
        "supplier_po_90d": {"medium": 3, "high": 1},
        "on_time_rate_pct": {"medium": 85.0, "high": 70.0},
        "reject_rate_pct": {"medium": 20.0, "high": 40.0},
        "budget_used_pct": {"medium": 90.0, "high": 100.0},
        "split_amount_pct": {"medium": 90.0, "high": 100.0},
    },
    "split_window_days": 7,
    "approval_amount_line": 100000,
    "weights": {"hard_block": 100, "high": 40, "medium": 15, "low": 3},
    # 多站路线的兜底：yaml 读不到时退化成「单站部门经理」。
    # 刻意不在这里复制 yaml 里那几段长 hint——兜底要的是「能跑、不越权」，
    # 不是「完整复现业务话术」。真丢了 yaml 应该在 --check 里被发现。
    "routes": [{"up_to": 0, "nodes": ["dept_manager"],
                "note": "兜底：单站部门经理"}],
    "insert_rules": [],
    "nodes": {
        "dept_manager": {
            "title": "部门经理", "level": 2,
            "duty": "需求真实性与用量合理性",
            "focus_metrics": ["qty_ratio", "stock_days", "safety_shortage_items"],
            "focus_history": ["issued_trend", "material_purchases"],
            "hint": "你是生产部门负责人，只判断这笔料该不该买、量合不合理；"
                    "不看价格与付款条件。",
        },
    },
    "route_rules": {"advice_values": ["agree", "agree_with_conditions",
                                      "oppose", "need_info"],
                    "min_avg_confidence": 0.80,
                    "veto_on_any_oppose": True},
    "risk_levels": {"low": 20, "medium": 60},
    "dify": {"enabled": True, "task": "dify_approve", "timeout_sec": 120,
             "required_keys": ["decision", "risk_level", "confidence",
                               "key_factors"],
             "decision_values": ["approve", "reject", "escalate"],
             "risk_values": ["low", "medium", "high"]},
    "knowledge_id": "kb-default",
}

# ---------------------------------------------------------------------------
# 指标元信息：标签 / 单位 / 阈值方向
#   up   = 数值越大越危险（>= 阈值触发）
#   down = 数值越小越危险（<= 阈值触发）
# 方向写死在代码里，改阈值不要改方向——方向错了阈值就失去意义。
# ---------------------------------------------------------------------------
_THRESHOLD_DIR: Dict[str, str] = {
    "price_deviation_pct": "up",
    "unit_price_vs_cost_pct": "up",
    "qty_ratio": "up",
    "stock_days": "up",
    "supplier_age_days": "down",
    "supplier_po_90d": "down",
    "on_time_rate_pct": "down",
    "reject_rate_pct": "up",
    "budget_used_pct": "up",
    "split_amount_pct": "up",
}

_METRIC_LABEL: Dict[str, str] = {
    "amount": "本单金额",
    "price_deviation_pct": "单价偏离历史均价",
    "unit_price_vs_cost_pct": "单价高于参考进价",
    "qty_ratio": "采购量 / 近30天用量",
    "stock_days": "收货后预计可用天数",
    "supplier_age_days": "供应商建档天数",
    "supplier_po_90d": "供应商近90天成交单数",
    "on_time_rate_pct": "供应商准时交付率",
    "reject_rate_pct": "供应商历史审批被拒率",
    "budget_used_pct": "本月预算占用率",
    "split_amount_pct": "同供应商7天在途合计 / 审批线",
}

_METRIC_UNIT: Dict[str, str] = {
    "amount": "元",
    "price_deviation_pct": "%",
    "unit_price_vs_cost_pct": "%",
    "qty_ratio": "倍",
    "stock_days": "天",
    "supplier_age_days": "天",
    "supplier_po_90d": "单",
    "on_time_rate_pct": "%",
    "reject_rate_pct": "%",
    "budget_used_pct": "%",
    "split_amount_pct": "%",
}

# 缺了这几项就不许自动批：没有历史价、没有用量口径，等于闭着眼签字
_CORE_METRICS = ("amount", "price_deviation_pct", "qty_ratio", "stock_days",
                 "split_amount_pct")


# ===========================================================================
# 配置加载
# ===========================================================================
def _deep_merge(base: Dict, over: Dict) -> Dict:
    """字典深合并：over 覆盖 base，base 不被原地改。"""
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_policy(path: str = "") -> Dict[str, Any]:
    """
    读策略配置。yaml 缺失/解析失败/为空 → 用内置兜底（取严，不放宽）。

    注意：这里不做「部分字段缺失就用默认值补齐」的静默处理之外的任何事情，
    配置写错只会退化成默认值，不会退化成「无限制」。
    """
    p = path or POLICY_PATH
    try:
        import yaml
        with open(p, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            return dict(_DEFAULT_POLICY)
        return _deep_merge(_DEFAULT_POLICY, raw)
    except Exception:
        return dict(_DEFAULT_POLICY)


def monthly_budget(policy: Dict, tenant_id: str = "default") -> float:
    """取租户月预算：tenants 下同名 key 优先，否则 default。"""
    b = policy.get("budget") or {}
    t = (b.get("tenants") or {}).get(tenant_id)
    if isinstance(t, dict) and t.get("monthly") is not None:
        return float(t["monthly"])
    return float(((b.get("default") or {}).get("monthly")) or 0.0)


# ===========================================================================
# ① 取数：12 个硬指标
# ===========================================================================
class DataPackBuilder:
    """
    把一张采购单摊成「可核对的数字包」。

    唯一 DB 出口是 `_q(sql, params)`。测试通过子类覆盖 `_q` 注入构造数据，
    这样指标计算逻辑被真跑，而 SQL 的正确性只能在有库时验证——两者不混为一谈。

    :param q: 可选，直接注入查询函数（签名 (sql, params) -> List[dict]）。
    """

    def __init__(self, policy: Dict[str, Any] = None, tenant_id: str = "default",
                 q: Callable = None, verbose: bool = False):
        self.policy = policy or load_policy()
        self.tenant_id = tenant_id
        self.verbose = verbose
        self._q_override = q
        self.available = q is not None
        self._pymysql = None
        self.error: Optional[str] = None
        if q is None:
            self._connect()

    # -- 基础设施 -------------------------------------------------------
    def _connect(self):
        try:
            import pymysql
            from memory_store import (MYSQL_HOST, MYSQL_PORT, MYSQL_USER,
                                      MYSQL_PASSWORD, MYSQL_DATABASE,
                                      MYSQL_CHARSET)
            self._pymysql = pymysql
            self._conn_kw = dict(host=MYSQL_HOST, port=MYSQL_PORT,
                                 user=MYSQL_USER, password=MYSQL_PASSWORD,
                                 database=MYSQL_DATABASE, charset=MYSQL_CHARSET,
                                 autocommit=True)
            conn = pymysql.connect(**self._conn_kw)
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            finally:
                conn.close()
            self.available = True
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            self.available = False

    def _q(self, sql: str, params: tuple = ()) -> List[Dict]:
        """唯一 DB 出口：SELECT → 字典行列表。"""
        if self._q_override is not None:
            return self._q_override(sql, params) or []
        if not self.available:
            raise RuntimeError(f"数据库不可用：{self.error}")
        conn = self._pymysql.connect(**self._conn_kw)
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                if not cur.description:
                    return []
                cols = [c[0] for c in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]
        finally:
            conn.close()

    # -- 主入口 ---------------------------------------------------------
    def build(self, po_id: int) -> Dict[str, Any]:
        """
        取一张采购单的完整数字包。

        返回结构::

            {
              "ok": True/False,
              "error": None/"...",
              "po":  {...单据头...},
              "items": [...明细...],
              "metrics": {...12 个指标，算不出来的为 None...},
              "missing": ["price_deviation_pct", ...],   # 数据不足的指标
              "evidence": {"history_price": {...}, ...},  # 中间量，供人工核对
            }
        """
        pack: Dict[str, Any] = {
            "ok": False, "error": None, "tenant_id": self.tenant_id,
            "po": {}, "items": [], "metrics": {}, "missing": [],
            "evidence": {},
        }
        try:
            po = self._fetch_po(po_id)
            if not po:
                pack["error"] = f"采购单 #{po_id} 不存在（tenant={self.tenant_id}）"
                return pack
            items = self._fetch_items(po_id)
            pack["po"], pack["items"] = po, items
            metrics, missing, evidence = self._compute(po, items)
            pack["metrics"], pack["missing"] = metrics, missing
            pack["evidence"] = evidence
            pack["ok"] = True
            return pack
        except Exception as e:
            pack["error"] = f"{type(e).__name__}: {e}"
            return pack

    # -- 取数 SQL -------------------------------------------------------
    def _fetch_po(self, po_id: int) -> Optional[Dict]:
        rows = self._q(
            "SELECT o.id, o.tenant_id, o.po_no, o.supplier_id, o.amount, "
            "       o.status, o.buyer, o.expected_date, o.created_by, "
            "       o.created_at, s.supplier_code, s.name AS supplier_name, "
            "       s.is_active AS supplier_active, "
            "       s.created_at AS supplier_created_at "
            "FROM purchase_orders o "
            "LEFT JOIN suppliers s ON s.id = o.supplier_id "
            "                      AND s.tenant_id = o.tenant_id "
            "WHERE o.id = %s AND o.tenant_id = %s",
            (po_id, self.tenant_id))
        return rows[0] if rows else None

    def _fetch_items(self, po_id: int) -> List[Dict]:
        return self._q(
            "SELECT i.id, i.product_id, i.quantity, i.unit_price, "
            "       i.received_qty, p.product_code, p.name AS product_name, "
            "       p.cost_price, p.safety_stock, p.unit AS product_unit "
            "FROM purchase_order_items i "
            "LEFT JOIN products p ON p.id = i.product_id "
            "                    AND p.tenant_id = i.tenant_id "
            "WHERE i.po_id = %s AND i.tenant_id = %s ORDER BY i.id",
            (po_id, self.tenant_id))

    # -- 指标计算 -------------------------------------------------------
    def _compute(self, po: Dict, items: List[Dict]
                 ) -> Tuple[Dict[str, Any], List[str], Dict[str, Any]]:
        """
        由「单据 + 明细」算出 12 个指标。

        每个指标算不出来时写 None 并进 missing——绝不填 0 冒充正常，
        因为「0% 价格偏离」和「没有历史价所以算不出偏离」是两件事。
        """
        th = self.policy.get("thresholds") or {}
        line = float(self.policy.get("approval_amount_line") or 100000)
        win = int(self.policy.get("split_window_days") or 7)
        metrics: Dict[str, Any] = {}
        missing: List[str] = []
        ev: Dict[str, Any] = {}

        amount = _f(po.get("amount"))
        metrics["amount"] = amount

        # ---- 供应商维度 ----
        age_days = _days_since(po.get("supplier_created_at"))
        metrics["supplier_age_days"] = age_days
        if age_days is None:
            missing.append("supplier_age_days")
        metrics["supplier_active"] = int(po.get("supplier_active") or 0)

        cnt90 = self._supplier_po_90d(po)
        metrics["supplier_po_90d"] = cnt90
        if cnt90 is None:
            missing.append("supplier_po_90d")

        ontime = self._on_time_rate(po)
        metrics["on_time_rate_pct"] = ontime
        if ontime is None:
            missing.append("on_time_rate_pct")

        rej = self._reject_rate(po)
        metrics["reject_rate_pct"] = rej
        if rej is None:
            missing.append("reject_rate_pct")

        # ---- 金额 / 预算维度 ----
        used = self._budget_used()
        budget = monthly_budget(self.policy, self.tenant_id)
        ev["budget"] = {"monthly": budget, "used_before": used,
                        "used_with_this": None if used is None else used + amount}
        if budget > 0 and used is not None:
            metrics["budget_used_pct"] = round((used + amount) / budget * 100, 2)
        else:
            metrics["budget_used_pct"] = None
            missing.append("budget_used_pct")

        # ---- 拆单识别 ----
        split = self._split_scan(po, amount, line, win)
        metrics["split_amount_pct"] = split["pct"]
        metrics["split_siblings"] = split["siblings"]
        metrics["split_sibling_amount"] = split["sibling_amount"]
        metrics["split_bypass"] = split["bypass"]
        if split["pct"] is None:
            missing.append("split_amount_pct")

        # ---- 物料维度（多明细按金额加权）----
        item_calc = [self._compute_item(it, po, amount) for it in items]
        ev["items"] = item_calc
        weighted = _weighted_avg(item_calc, "price_deviation_pct", "line_amount")
        metrics["price_deviation_pct"] = weighted
        if weighted is None:
            missing.append("price_deviation_pct")

        cost_dev = _weighted_avg(item_calc, "unit_price_vs_cost_pct",
                                 "line_amount")
        metrics["unit_price_vs_cost_pct"] = cost_dev
        if cost_dev is None:
            missing.append("unit_price_vs_cost_pct")

        qty_ratio = _weighted_avg(item_calc, "qty_ratio", "line_amount")
        metrics["qty_ratio"] = qty_ratio
        if qty_ratio is None:
            missing.append("qty_ratio")

        stock_days = _min_of(item_calc, "stock_days_after")
        metrics["stock_days"] = stock_days
        if stock_days is None:
            missing.append("stock_days")

        # 低于安全库存的明细（紧缺，不是风险，但要写进理由）
        gaps = [c for c in item_calc if c.get("safety_gap") is not None
                and c["safety_gap"] < 0]
        metrics["safety_shortage_items"] = len(gaps)
        metrics["safety_shortage_qty"] = sum(abs(c["safety_gap"]) for c in gaps)
        return metrics, missing, ev

    def _compute_item(self, it: Dict, po: Dict, po_amount: float) -> Dict:
        """单条明细的指标（供应商维度不在这里算）。"""
        pid = it.get("product_id")
        qty = int(it.get("quantity") or 0)
        price = _f(it.get("unit_price"))
        cost = _f(it.get("cost_price"))
        out: Dict[str, Any] = {
            "product_id": pid, "product_code": it.get("product_code"),
            "product_name": it.get("product_name"),
            "quantity": qty, "unit_price": price, "cost_price": cost,
            "line_amount": round(qty * price, 2),
            "price_deviation_pct": None, "unit_price_vs_cost_pct": None,
            "qty_ratio": None, "stock_days_after": None, "safety_gap": None,
        }
        # 单价 vs 参考进价
        if cost and cost > 0 and price is not None:
            out["unit_price_vs_cost_pct"] = round((price - cost) / cost * 100, 2)
        # 单价 vs 近 90 天已成交加权均价
        hist = self._history_price(pid, po.get("id"))
        out["history_avg_price"] = hist["avg_price"]
        out["history_qty"] = hist["qty"]
        if hist["avg_price"] and price is not None:
            out["price_deviation_pct"] = round(
                (price - hist["avg_price"]) / hist["avg_price"] * 100, 2)
        # 采购量 / 近 30 天用量
        used30 = self._issued_30d(pid)
        out["issued_30d"] = used30
        if used30 and used30 > 0 and qty > 0:
            out["qty_ratio"] = round(qty / used30, 2)
        # 收货后可用天数
        stock = self._stock(pid)
        out.update({"stock_qty": stock["stock"], "reserved_qty": stock["reserved"],
                    "in_transit": stock["in_transit"]})
        avail_after = (stock["stock"] - stock["reserved"]
                       + stock["in_transit"] + qty)
        out["avail_after"] = avail_after
        if used30 and used30 > 0:
            out["stock_days_after"] = round(avail_after / (used30 / 30.0), 1)
        # 与安全库存的缺口（负=低于安全线）
        ss = it.get("safety_stock")
        if ss is not None:
            out["safety_gap"] = int((stock["stock"] - stock["reserved"])
                                    - int(ss))
        return out

    # -- 单项取数 -------------------------------------------------------
    def _history_price(self, product_id, exclude_po_id) -> Dict[str, Any]:
        """该物料近 90 天**已成交**（approved/received/partial/closed）加权均价。"""
        rows = self._q(
            "SELECT COALESCE(SUM(i.quantity), 0) AS qty, "
            "       COALESCE(SUM(i.quantity * i.unit_price), 0) AS amt, "
            "       COUNT(*) AS rows_cnt "
            "FROM purchase_order_items i "
            "JOIN purchase_orders o ON o.id = i.po_id "
            "                      AND o.tenant_id = i.tenant_id "
            "WHERE i.tenant_id = %s AND i.product_id = %s AND o.id <> %s "
            "  AND o.status IN ('approved','received','partial','closed') "
            "  AND o.created_at >= DATE_SUB(NOW(), INTERVAL 90 DAY)",
            (self.tenant_id, product_id, exclude_po_id))
        qty = _f((rows[0] if rows else {}).get("qty"))
        amt = _f((rows[0] if rows else {}).get("amt"))
        avg = round(amt / qty, 4) if qty and qty > 0 else None
        return {"avg_price": avg, "qty": qty,
                "rows": int((rows[0] if rows else {}).get("rows_cnt") or 0)}

    def _issued_30d(self, product_id) -> Optional[float]:
        """近 30 天该物料的**净消耗**（出库为正，排除 RESERVE/RELEASE 这类占用动作）。"""
        rows = self._q(
            "SELECT COALESCE(SUM(-t.qty), 0) AS used "
            "FROM inventory_transactions t "
            "WHERE t.tenant_id = %s AND t.product_id = %s AND t.qty < 0 "
            "  AND t.txn_type NOT IN ('RESERVE','RELEASE') "
            "  AND t.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)",
            (self.tenant_id, product_id))
        v = _f((rows[0] if rows else {}).get("used"))
        return v if v else None

    def _stock(self, product_id) -> Dict[str, int]:
        """当前在库 / 已预留 / 在途（已批未收）。三者都算，否则可用天数会偏乐观。"""
        r1 = self._q(
            "SELECT COALESCE(SUM(stock_qty), 0) AS stock, "
            "       COALESCE(SUM(reserved_qty), 0) AS reserved "
            "FROM inventory WHERE tenant_id = %s AND product_id = %s",
            (self.tenant_id, product_id))
        r2 = self._q(
            "SELECT COALESCE(SUM(i.quantity - i.received_qty), 0) AS in_transit "
            "FROM purchase_order_items i "
            "JOIN purchase_orders o ON o.id = i.po_id "
            "                      AND o.tenant_id = i.tenant_id "
            "WHERE i.tenant_id = %s AND i.product_id = %s "
            "  AND o.status IN ('approved','partial')",
            (self.tenant_id, product_id))
        return {"stock": int(_f((r1[0] if r1 else {}).get("stock"))),
                "reserved": int(_f((r1[0] if r1 else {}).get("reserved"))),
                "in_transit": int(_f((r2[0] if r2 else {}).get("in_transit")))}

    def _supplier_po_90d(self, po: Dict) -> Optional[int]:
        rows = self._q(
            "SELECT COUNT(*) AS c FROM purchase_orders "
            "WHERE tenant_id = %s AND supplier_id = %s AND id <> %s "
            "  AND status IN ('approved','received','partial','closed') "
            "  AND created_at >= DATE_SUB(NOW(), INTERVAL 90 DAY)",
            (self.tenant_id, po.get("supplier_id"), po.get("id")))
        return int(_f((rows[0] if rows else {}).get("c")) or 0)

    def _on_time_rate(self, po: Dict) -> Optional[float]:
        """
        准时交付率（%）。

        口径说明：本项目没有独立的「收货时间」字段，这里用 updated_at 作为
        收货动作落库时间的**近似**——receive_po 会 UPDATE 状态，updated_at
        随之刷新。只在状态已是 received/closed 的单上算，避免在途单污染。
        """
        rows = self._q(
            "SELECT "
            "  SUM(CASE WHEN expected_date IS NOT NULL "
            "            AND DATE(updated_at) <= expected_date THEN 1 ELSE 0 END) "
            "    AS ontime, "
            "  SUM(CASE WHEN expected_date IS NOT NULL THEN 1 ELSE 0 END) AS total "
            "FROM purchase_orders "
            "WHERE tenant_id = %s AND supplier_id = %s "
            "  AND status IN ('received','closed')",
            (self.tenant_id, po.get("supplier_id")))
        total = int(_f((rows[0] if rows else {}).get("total")) or 0)
        if total <= 0:
            return None
        return round(int(_f((rows[0] if rows else {}).get("ontime")) or 0)
                     / total * 100, 2)

    def _reject_rate(self, po: Dict) -> Optional[float]:
        """
        该供应商历史采购审批被拒率（%）。

        approval_requests.payload 是 JSON，含 po_id 不含 supplier_id，
        所以要用 JSON_EXTRACT 反查采购单再归到供应商。
        这里只看 action_type='purchase_order.approve' 的单，别的动作不混入。
        """
        rows = self._q(
            "SELECT "
            "  SUM(CASE WHEN a.status = 'rejected' THEN 1 ELSE 0 END) AS rejected, "
            "  COUNT(*) AS total "
            "FROM approval_requests a "
            "JOIN purchase_orders o "
            "  ON o.id = CAST(JSON_UNQUOTE(JSON_EXTRACT(a.payload, '$.po_id')) "
            "                 AS UNSIGNED) "
            " AND o.tenant_id = a.tenant_id "
            "WHERE a.tenant_id = %s AND a.action_type = 'purchase_order.approve' "
            "  AND o.supplier_id = %s",
            (self.tenant_id, po.get("supplier_id")))
        total = int(_f((rows[0] if rows else {}).get("total")) or 0)
        if total <= 0:
            return None
        return round(int(_f((rows[0] if rows else {}).get("rejected")) or 0)
                     / total * 100, 2)

    def _budget_used(self) -> Optional[float]:
        """本月已批采购额（不含本单）。"""
        rows = self._q(
            "SELECT COALESCE(SUM(amount), 0) AS used FROM purchase_orders "
            "WHERE tenant_id = %s "
            "  AND status IN ('approved','received','partial','closed') "
            "  AND YEAR(created_at) = YEAR(CURDATE()) "
            "  AND MONTH(created_at) = MONTH(CURDATE())",
            (self.tenant_id,))
        if not rows:
            return None
        return _f(rows[0].get("used")) or 0.0

    def _split_scan(self, po: Dict, amount: float, line: float,
                    window_days: int) -> Dict[str, Any]:
        """
        拆单识别：把「本单 + 同供应商窗口期内其它在途单」合起来看。

        判据（这是整条审批流里最有价值的一条）：
          本单单独**没超**审批线，但和同供应商几天内的其它单合起来**超了**
          → 疑似拆单规避审批，落到 hard_block。
        如果本单自己就超线，那不叫拆单（那是明目张胆），走金额/等级规则即可。
        """
        rows = self._q(
            "SELECT id, po_no, amount, status, created_at FROM purchase_orders "
            "WHERE tenant_id = %s AND supplier_id = %s AND id <> %s "
            "  AND status IN ('submitted','approved','partial') "
            "  AND created_at >= DATE_SUB(NOW(), INTERVAL %s DAY) "
            "ORDER BY id DESC",
            (self.tenant_id, po.get("supplier_id"), po.get("id"),
             int(window_days)))
        sib_amt = round(sum(_f(r.get("amount")) for r in rows), 2)
        total = round(amount + sib_amt, 2)
        pct = round(total / line * 100, 2) if line > 0 else None
        bypass = bool(amount < line <= total and rows)
        return {"siblings": rows, "sibling_amount": sib_amt,
                "total": total, "pct": pct, "bypass": bypass}


# ===========================================================================
# ①b 采购历史档案（多站审批的证据来源）
# ===========================================================================
class HistoryDigest:
    """
    「这个物料以前怎么买的、这家供应商以前表现如何」摊成可引用的具体数字。

    为什么单独一层而不是塞进 metrics：

    审批建议里最有用的一句话是「上次从华鑫买同料 76.5 元/件，这次 80 元，涨 4.6%」。
    这句话包含三样东西：规则只知道「偏离 4.6%」，LLM 凭空不知道「上次谁家多少钱」，
    而历史明细里有。三者必须在这里汇合，所以历史必须是一等公民。

    另一个理由是**按站分发**：部门经理不需要看价格趋势，财务不需要看领用曲线。
    历史块按节点 focus_history 取子集，才不会出现「四个岗位看到同一段材料
    然后说出同一段话」——那是上一版最大的毛病。

    复用 DataPackBuilder 的 _q()，所以测试注入一个假查询就能全跑。
    """

    def __init__(self, builder: "DataPackBuilder", policy: Dict = None):
        self.b = builder
        self.policy = policy or builder.policy

    def _q(self, sql, params=()):
        return self.b._q(sql, params)

    # -- 主入口 ---------------------------------------------------------
    def build(self, pack: Dict[str, Any]) -> Dict[str, Any]:
        """
        返回结构（每块都可能为空，空就是空，不编）::

            {
              "_material": "不锈钢螺栓 M8（M-A）",   # 历史档案以哪条明细为主料
              "material_purchases":      [近 6 次同料采购明细],
              "price_trend":             {series, delta_pct, direction},
              "supplier_recent_orders":  [该供应商近 6 单],
              "supplier_share_pct":      该供应商占该物料年采购额比例,
              "quarterly_amount":        {this_quarter, this_year, last_year},
              "issued_trend":            [{month, qty}] 近 6 个月领用,
            }

        多明细单据按**金额最大**的那条明细取历史，并在 _material 里写明——
        不写清楚就等于让人以为算的是整单。
        """
        out: Dict[str, Any] = {
            "_material": "", "material_purchases": [], "price_trend": {},
            "supplier_recent_orders": [], "supplier_share_pct": None,
            "quarterly_amount": {}, "issued_trend": [], "_error": None,
        }
        try:
            po = pack.get("po") or {}
            items = (pack.get("evidence") or {}).get("items") or []
            if not items:
                out["_error"] = "无采购明细，无法取历史"
                return out
            main = max(items, key=lambda x: _f(x.get("line_amount")) or 0)
            pid = main.get("product_id")
            sup = po.get("supplier_id")
            out["_material"] = f"{main.get('product_name')}（{main.get('product_code')}）"

            out["material_purchases"] = self._material_purchases(pid, po.get("id"))
            out["price_trend"] = self._price_trend(out["material_purchases"],
                                                   main.get("unit_price"))
            out["supplier_recent_orders"] = self._supplier_orders(sup, po.get("id"))
            out["quarterly_amount"] = self._quarterly()
            out["supplier_share_pct"] = self._supplier_share(pid, sup)
            out["issued_trend"] = self._issued_trend(pid)
            return out
        except Exception as e:
            out["_error"] = f"{type(e).__name__}: {e}"
            return out

    # -- 各块取数 -------------------------------------------------------
    def _material_purchases(self, product_id, exclude_po_id) -> List[Dict]:
        """该物料近 6 次采购（含本次之外的成交历史），带供应商名。"""
        if product_id is None:
            return []
        rows = self._q(
            "SELECT o.po_no, o.created_at, o.status, "
            "       s.name AS supplier_name, i.quantity, i.unit_price "
            "FROM purchase_order_items i "
            "JOIN purchase_orders o ON o.id = i.po_id "
            "                      AND o.tenant_id = i.tenant_id "
            "LEFT JOIN suppliers s ON s.id = o.supplier_id "
            "                     AND s.tenant_id = o.tenant_id "
            "WHERE i.tenant_id = %s AND i.product_id = %s AND o.id <> %s "
            "ORDER BY o.id DESC LIMIT 6",
            (self.b.tenant_id, product_id, exclude_po_id))
        return [{"po_no": r.get("po_no"), "date": _dstr(r.get("created_at")),
                 "status": r.get("status"),
                 "supplier": r.get("supplier_name"),
                 "quantity": int(_f(r.get("quantity")) or 0),
                 "unit_price": _f(r.get("unit_price"))}
                for r in rows]

    def _price_trend(self, purchases: List[Dict], current_price) -> Dict[str, Any]:
        """把历史单价排成时间序列，算出这次相对上次/均价的涨跌。"""
        cur = _f(current_price)
        series = [{"po_no": p["po_no"], "date": p["date"],
                   "unit_price": p["unit_price"], "supplier": p["supplier"]}
                  for p in reversed(purchases) if p.get("unit_price")]
        prices = [s["unit_price"] for s in series]
        if not prices or cur is None:
            return {"series": series, "last_price": prices[-1] if prices else None,
                    "delta_vs_last_pct": None, "direction": "无历史"}
        last = prices[-1]
        delta = round((cur - last) / last * 100, 2) if last else None
        if delta is None:
            direction = "无历史"
        elif delta > 1:
            direction = "涨"
        elif delta < -1:
            direction = "跌"
        else:
            direction = "持平"
        return {"series": series, "last_price": last,
                "delta_vs_last_pct": delta, "direction": direction,
                "last_supplier": series[-1]["supplier"] if series else None}

    def _supplier_orders(self, supplier_id, exclude_po_id) -> List[Dict]:
        """该供应商近 6 单（金额/状态/期望到货），用于评估交付与集中度。"""
        if supplier_id is None:
            return []
        rows = self._q(
            "SELECT po_no, amount, status, expected_date, created_at, updated_at "
            "FROM purchase_orders "
            "WHERE tenant_id = %s AND supplier_id = %s AND id <> %s "
            "ORDER BY id DESC LIMIT 6",
            (self.b.tenant_id, supplier_id, exclude_po_id))
        return [{"po_no": r.get("po_no"), "amount": _f(r.get("amount")),
                 "status": r.get("status"),
                 "expected_date": _dstr(r.get("expected_date")),
                 "created": _dstr(r.get("created_at")),
                 "updated": _dstr(r.get("updated_at"))}
                for r in rows]

    def _supplier_share(self, product_id, supplier_id) -> Optional[float]:
        """该供应商占该物料近一年采购额的比例（%）。集中度风险的核心数字。"""
        if product_id is None or supplier_id is None:
            return None
        rows = self._q(
            "SELECT "
            "  COALESCE(SUM(CASE WHEN o.supplier_id = %s "
            "                    THEN i.quantity * i.unit_price ELSE 0 END), 0) "
            "    AS sup_amt, "
            "  COALESCE(SUM(i.quantity * i.unit_price), 0) AS total_amt "
            "FROM purchase_order_items i "
            "JOIN purchase_orders o ON o.id = i.po_id "
            "                      AND o.tenant_id = i.tenant_id "
            "WHERE i.tenant_id = %s AND i.product_id = %s "
            "  AND o.created_at >= DATE_SUB(NOW(), INTERVAL 365 DAY)",
            (supplier_id, self.b.tenant_id, product_id))
        r = rows[0] if rows else {}
        total = _f(r.get("total_amt")) or 0.0
        if total <= 0:
            return None
        return round((_f(r.get("sup_amt")) or 0.0) / total * 100, 2)

    def _quarterly(self) -> Dict[str, Any]:
        """本季度 / 今年 / 去年同期的已发生采购额（用于财务站的时点判断）。"""
        rows = self._q(
            "SELECT "
            "  COALESCE(SUM(CASE WHEN YEAR(created_at) = YEAR(CURDATE()) "
            "                    AND QUARTER(created_at) = QUARTER(CURDATE()) "
            "                    THEN amount ELSE 0 END), 0) AS this_quarter, "
            "  COALESCE(SUM(CASE WHEN YEAR(created_at) = YEAR(CURDATE()) "
            "                    THEN amount ELSE 0 END), 0) AS this_year, "
            "  COALESCE(SUM(CASE WHEN YEAR(created_at) = YEAR(CURDATE()) - 1 "
            "                    THEN amount ELSE 0 END), 0) AS last_year "
            "FROM purchase_orders WHERE tenant_id = %s "
            "  AND status IN ('approved','received','partial','closed')",
            (self.b.tenant_id,))
        r = rows[0] if rows else {}
        return {"this_quarter": _f(r.get("this_quarter")) or 0.0,
                "this_year": _f(r.get("this_year")) or 0.0,
                "last_year": _f(r.get("last_year")) or 0.0}

    def _issued_trend(self, product_id) -> List[Dict]:
        """
        近 6 个月该物料的月度领用量。

        刻意在 Python 里按 (年, 月) 聚合而不是用 DATE_FORMAT：
        pymysql 走参数化时 SQL 里的 `%Y-%m` 会被当成占位符报错，得写成 `%%Y-%%m`，
        这个坑一次踩过就不想再踩——而且换 SQLite/PG 时字符串格式化还不通用。
        """
        if product_id is None:
            return []
        rows = self._q(
            "SELECT created_at, qty FROM inventory_transactions "
            "WHERE tenant_id = %s AND product_id = %s AND qty < 0 "
            "  AND txn_type NOT IN ('RESERVE','RELEASE') "
            "  AND created_at >= DATE_SUB(NOW(), INTERVAL 6 MONTH)",
            (self.b.tenant_id, product_id))
        buckets: Dict[str, float] = {}
        for r in rows:
            ym = _dstr(r.get("created_at"))[:7]      # YYYY-MM
            if not ym:
                continue
            buckets[ym] = buckets.get(ym, 0.0) + abs(_f(r.get("qty")) or 0.0)
        return [{"month": k, "qty": round(v, 2)} for k, v in sorted(buckets.items())]


# ===========================================================================
# ② 确定性规则打分（不依赖 LLM，可复现）
# ===========================================================================
class RuleEngine:
    """按阈值把数字包翻译成 findings，并给出 risk_score / risk_level。"""

    def __init__(self, policy: Dict[str, Any] = None):
        self.policy = policy or load_policy()

    def score(self, pack: Dict[str, Any]) -> Dict[str, Any]:
        """
        返回::

            {
              "findings": [{code, level, metric, value, threshold, text}],
              "risk_score": 55,
              "risk_level": "medium",
              "hard_block": False,
              "blockers": [],          # hard_block 的原因
            }
        """
        th = self.policy.get("thresholds") or {}
        w = self.policy.get("weights") or {}
        metrics = pack.get("metrics") or {}
        findings: List[Dict[str, Any]] = []
        blockers: List[str] = []

        # ---- 硬规则（代码级，配置改不掉）----
        if not pack.get("ok"):
            blockers.append(f"数字包取数失败：{pack.get('error')}")
        amt = _f(metrics.get("amount"))
        if amt is None or amt <= 0:
            blockers.append("采购金额为空或非正，无法定价审批")
        if metrics.get("supplier_active") == 0:
            blockers.append("供应商已停用（suppliers.is_active=0）")
        if metrics.get("split_bypass"):
            sib = metrics.get("split_siblings") or []
            nos = "、".join(str(s.get("po_no")) for s in sib[:3]) or "-"
            blockers.append(
                f"疑似拆单规避审批：本单 {amt:,.2f} 未超审批线，"
                f"但与同供应商 {self.policy.get('split_window_days')} 天内的 "
                f"{nos} 合计 {_f(metrics.get('split_sibling_amount')) + amt:,.2f} 已超线")

        # ---- 阈值规则 ----
        for key, direction in _THRESHOLD_DIR.items():
            v = metrics.get(key)
            if v is None:
                continue
            rule = th.get(key) or {}
            lvl = _level_of(key, v, rule, direction)
            if not lvl:
                continue
            findings.append({
                "code": f"TH.{key}",
                "level": lvl,
                "metric": key,
                "label": _METRIC_LABEL.get(key, key),
                "value": v,
                "unit": _METRIC_UNIT.get(key, ""),
                "threshold": rule.get(lvl),
                "text": self._finding_text(key, v, lvl, rule),
            })

        # ---- 数据缺失：算不出来的核心指标本身就是风险 ----
        for m in pack.get("missing") or []:
            if m in _CORE_METRICS:
                findings.append({
                    "code": "DATA.missing",
                    "level": "medium",
                    "metric": m,
                    "label": _METRIC_LABEL.get(m, m),
                    "value": None, "unit": "", "threshold": None,
                    "text": f"{_METRIC_LABEL.get(m, m)}无法计算（历史数据不足），"
                            f"该项无法核验",
                })

        score = sum(int(w.get(f["level"], 0)) for f in findings)
        score += int(w.get("hard_block", 0)) if blockers else 0
        rl = self.policy.get("risk_levels") or {}
        low_max = int(rl.get("low", 20))
        medium_max = int(rl.get("medium", 60))
        # 分级**不只看总分**：只要出现一条 high finding 就是 high，
        # 出现 medium finding 至少是 medium。原因是权重表可以被配置调低
        # （比如把 medium 权重从 15 改成 5），如果只看分数，一条真实的
        # 中风险项就会被总分线吃掉、判成 low，进而被自动批准。
        # 分数只用来排优先级，等级必须由「最严重的那条」兜底。
        has_high = any(f["level"] == "high" for f in findings)
        has_medium = any(f["level"] == "medium" for f in findings)
        if blockers or has_high or score > medium_max:
            level = "high"
        elif has_medium or score > low_max:
            level = "medium"
        else:
            level = "low"
        return {"findings": findings, "risk_score": score, "risk_level": level,
                "hard_block": bool(blockers), "blockers": blockers}

    def _finding_text(self, key: str, v: Any, lvl: str, rule: Dict) -> str:
        unit = _METRIC_UNIT.get(key, "")
        label = _METRIC_LABEL.get(key, key)
        thr = rule.get(lvl)
        cmp_ = "高于" if _THRESHOLD_DIR.get(key) == "up" else "低于"
        return f"{label} {v}{unit}，{cmp_}阈值 {thr}{unit}"


# ===========================================================================
# ②b 审批路线规划：这张单走哪几站
# ===========================================================================
class RoutePlanner:
    """
    按「金额分段 + 历史信号」决定这张单要过哪几站。

    两层设计的好处：想调「多少钱走几家」改 routes，想调「什么情况加一站」
    改 insert_rules，都不用碰代码。而「每站看什么说什么」在 nodes 里，
    改话术不会影响路线。

    `node.level` 是「这一站要求几级的人来点」的声明，只用于判断
    「AI 够不够资格自动批」——真实审批谁有资格点，仍由 access_rules.yaml 的
    order_acl 决定。两套东西混在一起就会自相矛盾。
    """

    def __init__(self, policy: Dict[str, Any] = None):
        self.policy = policy or load_policy()

    # -- 历史信号 -------------------------------------------------------
    def _signals(self, pack: Dict[str, Any], rule: Dict[str, Any]) -> Dict[str, bool]:
        """
        从数字包里抽出几个布尔信号，作为「要不要插站」的判据。

        刻意用布尔而不是原始数值：插站规则是业务配置，让它直接读原始数值
        会逼着配置里写比较表达式，等于把代码搬进 yaml。
        """
        m = pack.get("metrics") or {}
        th = self.policy.get("thresholds") or {}
        sig: Dict[str, bool] = {}

        sig["split_bypass"] = bool(m.get("split_bypass"))

        age = _f(m.get("supplier_age_days"))
        cnt = _f(m.get("supplier_po_90d"))
        sig["new_supplier"] = bool((age is not None and age < 90)
                                   or (cnt is not None and cnt <= 0))

        # 价格异动：直接问规则层「你有没有给出 high 级的单价偏离」
        sig["price_spike"] = any(
            f.get("metric") == "price_deviation_pct" and f.get("level") == "high"
            for f in (rule.get("findings") or []))

        bu = _f(m.get("budget_used_pct"))
        bu_med = _f((th.get("budget_used_pct") or {}).get("medium"))
        sig["budget_tight"] = bool(bu is not None and bu_med is not None
                                   and bu >= bu_med)

        ot = _f(m.get("on_time_rate_pct"))
        ot_med = _f((th.get("on_time_rate_pct") or {}).get("medium"))
        sig["late_supplier"] = bool(ot is not None and ot_med is not None
                                    and ot <= ot_med)
        return sig

    # -- 主入口 ---------------------------------------------------------
    def plan(self, pack: Dict[str, Any], rule: Dict[str, Any]) -> Dict[str, Any]:
        """
        返回::

            {
              "nodes": [{node_id, title, level, duty, focus_metrics,
                         focus_history, hint, reason}],
              "max_level": 3,
              "base_note": "5~10 万 · 加财务确认预算",
              "inserted": [{node, because}],
              "signals": {...},
              "warnings": [],
            }
        """
        amount = _f((pack.get("metrics") or {}).get("amount")) or 0.0
        base = self._pick_route(amount)
        node_defs = self.policy.get("nodes") or {}
        warnings: List[str] = []

        ids: List[str] = []
        reasons: Dict[str, str] = {}
        for nid in (base.get("nodes") or []):
            if nid in node_defs:
                ids.append(nid)
                reasons[nid] = base.get("note") or "按金额路线"
            else:
                warnings.append(f"路线引用了未定义的节点「{nid}」，已跳过")

        signals = self._signals(pack, rule)
        inserted: List[Dict[str, str]] = []
        for ir in (self.policy.get("insert_rules") or []):
            nid = str(ir.get("node") or "")
            when = str(ir.get("when") or "")
            if not signals.get(when):
                continue
            if nid not in node_defs:
                warnings.append(f"插站规则引用了未定义的节点「{nid}」，已跳过")
                continue
            if nid in ids:
                reasons[nid] = (reasons.get(nid, "") + "；" +
                                str(ir.get("because") or "")).strip("；")
                continue
            after = str(ir.get("after") or "")
            pos = ids.index(after) + 1 if after in ids else len(ids)
            ids.insert(pos, nid)
            reasons[nid] = str(ir.get("because") or "动态插入")
            inserted.append({"node": nid, "because": reasons[nid]})

        nodes = []
        for nid in ids:
            d = node_defs[nid]
            nodes.append({
                "node_id": nid,
                "title": str(d.get("title") or nid),
                "level": int(d.get("level") or 2),
                "duty": str(d.get("duty") or ""),
                "focus_metrics": list(d.get("focus_metrics") or []),
                "focus_history": list(d.get("focus_history") or []),
                "hint": str(d.get("hint") or "").strip(),
                "reason": reasons.get(nid, ""),
            })

        max_level = max([n["level"] for n in nodes], default=2)
        return {"nodes": nodes, "max_level": max_level,
                "base_note": base.get("note") or "",
                "inserted": inserted, "signals": signals,
                "warnings": warnings}

    def _pick_route(self, amount: float) -> Dict[str, Any]:
        """按金额选基础路线：第一个 amount <= up_to 的段生效；up_to=0 视为无上限。"""
        for r in (self.policy.get("routes") or []):
            up = r.get("up_to")
            try:
                up_f = float(up) if up is not None else 0.0
            except (TypeError, ValueError):
                up_f = 0.0
            if up_f <= 0 or amount <= up_f:
                return r
        # 配置里没有兜底段 → 退回最后一段，再不济退成单站，绝不返回空路线
        routes = self.policy.get("routes") or []
        if routes:
            return routes[-1]
        return {"nodes": ["dept_manager"], "note": "无路线配置，退回单站"}


def _level_of(key: str, value: Any, rule: Dict, direction: str) -> Optional[str]:
    """判一条指标落在哪一级；不触发返回 None。"""
    v = _f(value)
    if v is None:
        return None
    high = rule.get("high")
    medium = rule.get("medium")
    if direction == "up":
        if high is not None and v >= float(high):
            return "high"
        if medium is not None and v >= float(medium):
            return "medium"
    else:
        if high is not None and v <= float(high):
            return "high"
        if medium is not None and v <= float(medium):
            return "medium"
    return None


# ===========================================================================
# ③ AI 研判：Dify 工作流（经 llm_gateway，白拿限流/熔断/降级）
# ===========================================================================
_SYSTEM_PROMPT = """你是一名制造业采购审批员，负责审核物料采购申请。

硬性要求：
1. 你只依据下面给出的数字和制度条款做判断，不得引入外部假设，不得编造数字。
2. 每一个结论都必须能追溯到给出的某个指标或某条条款，写进 key_factors。
3. 数据缺失（missing 列表非空）时必须在 missing_data 里点名，不得当作正常。
4. 只输出一个 JSON 对象，不要任何解释性文字、不要 Markdown 代码围栏。

输出 JSON 结构：
{
  "decision": "approve | reject | escalate",
  "risk_level": "low | medium | high",
  "confidence": 0.0 到 1.0 之间的小数,
  "key_factors": ["每个元素必须是『指标名=数值 → 影响』的格式"],
  "evidence": ["引用的制度条款或历史依据，无则空数组"],
  "missing_data": ["缺失项"],
  "suggestion": "一句话给审批人的处置建议"
}"""


def render_material(pack: Dict[str, Any], rule: Dict[str, Any],
                    clauses: List[Dict] = None) -> str:
    """
    把数字包 + 规则结论 + 制度条款渲染成给 LLM 的输入材料。

    刻意用「表格 + 逐条 findings」而不是塞 JSON：模型对
    「指标名 = 数值（阈值 X）」这种一行一事的格式最不容易读错。
    """
    po = pack.get("po") or {}
    m = pack.get("metrics") or {}
    lines = [
        "## 采购申请",
        f"单号：{po.get('po_no')}  状态：{po.get('status')}  "
        f"采购员：{po.get('buyer') or '-'}",
        f"供应商：{po.get('supplier_name')}（{po.get('supplier_code')}）  "
        f"本单金额：{_f(m.get('amount')):,.2f} 元",
        "",
        "## 明细",
    ]
    for it in (pack.get("evidence") or {}).get("items") or []:
        lines.append(
            f"- {it.get('product_name')}({it.get('product_code')}) "
            f"{it.get('quantity')} × {it.get('unit_price')} 元 "
            f"= {it.get('line_amount'):,.2f} 元")
    lines += ["", "## 硬指标（本单实算，可回源）"]
    for k in ("amount", "price_deviation_pct", "unit_price_vs_cost_pct",
              "qty_ratio", "stock_days", "supplier_age_days",
              "supplier_po_90d", "on_time_rate_pct", "reject_rate_pct",
              "budget_used_pct", "split_amount_pct"):
        v = m.get(k)
        unit = _METRIC_UNIT.get(k, "")
        shown = "数据不足" if v is None else f"{v}{unit}"
        lines.append(f"- {_METRIC_LABEL.get(k, k)} = {shown}")
    if m.get("safety_shortage_items"):
        lines.append(f"- 低于安全库存的明细数 = {m['safety_shortage_items']} 项，"
                     f"合计缺口 {m['safety_shortage_qty']}")
    lines += ["", "## 规则引擎结论",
              f"风险分：{rule.get('risk_score')}  等级：{rule.get('risk_level')}"]
    if rule.get("blockers"):
        lines.append("一票否决项（不可被推翻）：")
        lines += [f"- {b}" for b in rule["blockers"]]
    for f in rule.get("findings") or []:
        lines.append(f"- [{f['level']}] {f['text']}")
    if not rule.get("findings") and not rule.get("blockers"):
        lines.append("- 无异常指标")
    if pack.get("missing"):
        lines.append(f"缺失指标：{', '.join(pack['missing'])}")
    if clauses:
        lines += ["", "## 制度条款（知识库检索）"]
        for i, c in enumerate(clauses, 1):
            lines.append(f"[{i}] {c.get('title') or '-'}：{c.get('content')}")
    return "\n".join(lines)


class DifyReviewer:
    """
    调 Dify 工作流出结构化判定。

    :param chat: 可选，注入 (system, user) -> str 的调用函数（测试用）。
                 None 时走 llm_gateway.chat_detailed(task=policy.dify.task)。
    """

    def __init__(self, policy: Dict[str, Any] = None, chat: Callable = None,
                 gateway=None, verbose: bool = False):
        self.policy = policy or load_policy()
        self.verbose = verbose
        self._chat = chat
        self._gateway = gateway
        self.last_error: Optional[str] = None
        self.last_latency_ms: int = 0

    def _call(self, system_prompt: str, user_prompt: str) -> str:
        if self._chat is not None:
            return self._chat(system_prompt, user_prompt)
        if self._gateway is None:
            from llm_gateway import LLMGateway
            self._gateway = LLMGateway(verbose=False)
        d = self.policy.get("dify") or {}
        resp = self._gateway.chat_detailed(
            system_prompt, user_prompt,
            task=str(d.get("task") or "dify_approve"),
            user=str((self.policy.get("ai_approver") or {}).get("role")
                     or "ai_approver"))
        # 网关「全链失败」时**不抛异常**，而是返回配置里的 degraded_reply 文案。
        # 那样下游只报「AI 输出里找不到 JSON 对象」，把「模型根本连不上」
        # 误诊成「模型乱输出」，排障会往完全错误的方向查。
        # 这里显式识别（网关把 model 标成 <degraded>）并抛错，让 error 说真相。
        if getattr(resp, "model", "") == "<degraded>":
            raise RuntimeError(
                "LLM 网关全链失败（Dify 与本地模型均不可用），"
                "拿到的是降级文案而不是模型输出")
        return getattr(resp, "text", resp) or ""

    def review(self, pack: Dict[str, Any], rule: Dict[str, Any],
               clauses: List[Dict] = None) -> Dict[str, Any]:
        """
        返回::

            {"ok": True,  "verdict": {decision, risk_level, confidence, ...}}
            {"ok": False, "error": "..."}     # 任何异常都不得当成功
        """
        d = self.policy.get("dify") or {}
        if not d.get("enabled", True):
            return {"ok": False, "error": "AI 研判已在配置中关闭（dify.enabled=false）"}
        self.last_error = None
        t0 = time.time()
        try:
            raw = self._call(_SYSTEM_PROMPT, render_material(pack, rule, clauses))
        except Exception as e:
            self.last_error = f"调用 AI 失败：{type(e).__name__}: {e}"
            return {"ok": False, "error": self.last_error}
        finally:
            self.last_latency_ms = int((time.time() - t0) * 1000)
        verdict, err = parse_verdict(raw, d)
        if err:
            self.last_error = err
            return {"ok": False, "error": err, "raw": (raw or "")[:2000]}
        return {"ok": True, "verdict": verdict, "latency_ms": self.last_latency_ms,
                "prompt_len": len(raw or "")}


def parse_verdict(raw: str, dify_cfg: Dict = None) -> Tuple[Optional[Dict], str]:
    """
    解析 AI 输出。返回 (verdict, error)；error 非空即判失败。

    容错只做「取 JSON 子串」这一步——模型偶尔会带 Markdown 围栏或前置寒暄。
    **不做**的错误容忍：缺键、值不在白名单、confidence 非数字。
    这些东西一旦被「猜」出来，审批就变成了猜谜。
    """
    cfg = dify_cfg or (_DEFAULT_POLICY.get("dify") or {})
    if not raw or not str(raw).strip():
        return None, "AI 返回空内容"
    text = str(raw).strip()
    # 剥 Markdown 围栏
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    # 取第一个 { 到最后一个 }（容忍前后寒暄）
    if not text.startswith("{"):
        i, j = text.find("{"), text.rfind("}")
        if i < 0 or j <= i:
            return None, "AI 输出里找不到 JSON 对象"
        text = text[i:j + 1]
    try:
        obj = json.loads(text)
    except Exception as e:
        return None, f"AI 输出不是合法 JSON：{e}"
    if not isinstance(obj, dict):
        return None, "AI 输出的 JSON 顶层不是对象"

    for k in (cfg.get("required_keys") or []):
        if k not in obj:
            return None, f"AI 输出缺少必需字段「{k}」"
    dec = str(obj.get("decision", "")).strip().lower()
    if dec not in (cfg.get("decision_values") or ["approve"]):
        return None, f"AI 的 decision「{obj.get('decision')}」不在白名单内"
    rl = str(obj.get("risk_level", "")).strip().lower()
    if rl not in (cfg.get("risk_values") or ["low"]):
        return None, f"AI 的 risk_level「{obj.get('risk_level')}」不在白名单内"
    try:
        conf = float(obj.get("confidence"))
    except Exception:
        return None, f"AI 的 confidence「{obj.get('confidence')}」不是数字"
    if not (0.0 <= conf <= 1.0):
        return None, f"AI 的 confidence {conf} 越界（须 0~1）"
    kf = obj.get("key_factors")
    if not isinstance(kf, list):
        return None, "AI 的 key_factors 不是数组"
    return {
        "decision": dec, "risk_level": rl, "confidence": round(conf, 4),
        "key_factors": [str(x) for x in kf],
        "evidence": [str(x) for x in (obj.get("evidence") or [])],
        "missing_data": [str(x) for x in (obj.get("missing_data") or [])],
        "suggestion": str(obj.get("suggestion") or ""),
    }, ""


# ===========================================================================
# ③b 节点级 AI 建议：每一站只看自己那一摊
# ===========================================================================
_NODE_SYSTEM = """你是{title}，在这笔物料采购审批中负责「{duty}」。

{extra}
职责边界（这条最重要，越权评价会让你这一站的意见失去价值）：
{hint}

输出要求：
1. 只依据下面给出的指标与历史记录判断，不得引入外部假设，不得编造数字。
2. 只评价你有权管的维度。比如部门经理不要评价格、财务不要评该不该买——
   那不是你的职责，写了对审批人没有价值。
3. 每条结论必须落到具体数字上，格式用「指标=值 → 影响」。
4. 数据不足时在 questions 里点名要补什么，不要猜，也不要把缺数据当成没问题。
5. 只输出一个 JSON 对象。不要 Markdown 代码围栏，不要前后解释性文字。

JSON 结构：
{{
  "advice": "agree | agree_with_conditions | oppose | need_info",
  "confidence": 0.0 到 1.0 之间的小数,
  "key_points": ["指标=值 → 影响"],
  "conditions": ["若同意需附加的条件"],
  "questions": ["需要采购员补充说明的问题"],
  "one_liner": "一句话给审批人看的建议（不超过 40 字）"
}}"""

_NODE_SIGNALS_DESC = {
    "split_bypass": "本单未超审批线，但与同供应商窗口期内的在途单合计已超线",
    "new_supplier": "该供应商建档不足 90 天或没有历史成交",
    "price_spike": "本单单价相对历史成交均价异常偏高",
    "budget_tight": "本月采购预算已接近用尽",
    "late_supplier": "该供应商历史准时交付率偏低",
}


def render_history_block(key: str, h: Dict[str, Any]) -> List[str]:
    """把历史档案里的一块渲染成人能读的几行（也让 LLM 好引用）。"""
    out: List[str] = []
    if key == "material_purchases":
        rows = h.get("material_purchases") or []
        if not rows:
            return ["同料历史采购：无成交记录（这是首次采购）"]
        out.append("同料历史采购（近 %d 次）：" % len(rows))
        for r in rows:
            out.append(f"  · {r['date']} {r['supplier'] or '-'} "
                       f"{r['quantity']} × {_fmt(r['unit_price'])} 元 "
                       f"[{r['status']}] {r['po_no']}")
        return out
    if key == "price_trend":
        pt = h.get("price_trend") or {}
        series = pt.get("series") or []
        if not series:
            return ["价格走势：无历史成交价，无法判断涨跌"]
        seq = " → ".join(_fmt(s["unit_price"]) for s in series)
        out.append(f"历史成交单价序列：{seq} 元")
        if pt.get("last_price") is not None:
            out.append(f"上次成交 {_fmt(pt['last_price'])} 元"
                       f"（供应商 {pt.get('last_supplier') or '-'}），"
                       f"本次相比上次 {pt.get('direction')} "
                       f"{_fmt(pt.get('delta_vs_last_pct'))}%")
        return out
    if key == "supplier_recent_orders":
        rows = h.get("supplier_recent_orders") or []
        if not rows:
            return ["该供应商近期订单：无（新供应商）"]
        out.append(f"该供应商近期订单（近 {len(rows)} 单）：")
        for r in rows:
            overdue = ""
            if r.get("expected_date") and r.get("updated") \
                    and r["updated"][:10] > r["expected_date"]:
                overdue = f" 超期(期望{r['expected_date']} 实收{r['updated'][:10]})"
            out.append(f"  · {r['po_no']} 金额 {_fmt(r['amount'])} 元 "
                       f"[{r['status']}]{overdue}")
        return out
    if key == "supplier_share_pct":
        v = h.get("supplier_share_pct")
        if v is None:
            return ["供应商集中度：无历史数据"]
        return [f"该供应商占本物料近一年采购额 {v}%"
                + ("（集中度偏高，建议引入备选）" if v >= 70 else "")]
    if key == "quarterly_amount":
        q = h.get("quarterly_amount") or {}
        return [f"采购额：本季度 {_fmt(q.get('this_quarter'))} 元 / "
                f"今年累计 {_fmt(q.get('this_year'))} 元 / "
                f"去年全年 {_fmt(q.get('last_year'))} 元"]
    if key == "issued_trend":
        rows = h.get("issued_trend") or []
        if not rows:
            return ["月度领用：无出库记录"]
        return ["月度领用：" + "  ".join(f"{r['month']}:{_fmt(r['qty'])}"
                                        for r in rows)]
    return []


def render_node_material(pack: Dict[str, Any], history: Dict[str, Any],
                         rule: Dict[str, Any], node: Dict[str, Any]) -> str:
    """
    给某一站渲染它该看的材料。

    关键取舍：**只给这一站 focus 里的东西**。上一版把所有指标一股脑塞进去，
    结果四个岗位说的都是同一段话——那不是多站审批，那是把一份建议复制四遍。
    """
    po = pack.get("po") or {}
    m = pack.get("metrics") or {}
    lines = [
        f"## 本单（{node.get('title')}视角）",
        f"单号：{po.get('po_no')}  金额：{_fmt(m.get('amount'))} 元  "
        f"供应商：{po.get('supplier_name')}（{po.get('supplier_code')}）  "
        f"采购员：{po.get('buyer') or '-'}",
        f"你的职责：{node.get('duty')}",
    ]
    if node.get("reason"):
        lines.append(f"你这一站为什么存在：{node['reason']}")

    lines += ["", "## 你该看的指标"]
    focus = node.get("focus_metrics") or []
    if not focus:
        lines.append("- （未配置关注指标）")
    for k in focus:
        v = m.get(k)
        unit = _METRIC_UNIT.get(k, "")
        if v is None:
            lines.append(f"- {_METRIC_LABEL.get(k, k)} = 数据不足/无法计算")
        else:
            lines.append(f"- {_METRIC_LABEL.get(k, k)} = {v}{unit}")

    lines += ["", "## 采购历史记录"]
    fh = node.get("focus_history") or []
    if not fh:
        lines.append("- （未配置关注历史）")
    for k in fh:
        lines += render_history_block(k, history)

    # 规则层里与本站关注点相关的 findings；全局硬伤所有站都要知道
    related = [f for f in (rule.get("findings") or [])
               if f.get("metric") in focus]
    if related:
        lines += ["", "## 规则引擎在本站关注点上的告警"]
        lines += [f"- [{f['level']}] {f['text']}" for f in related]
    if rule.get("blockers"):
        lines += ["", "## 全局一票否决（与你这一站无关也必须知道）"]
        lines += [f"- {b}" for b in rule["blockers"]]
    if pack.get("missing"):
        lines.append(f"\n注意，以下指标数据缺失：{', '.join(pack['missing'])}")
    return "\n".join(lines)


class NodeAdvisor:
    """
    对路线上每一站各出一份结构化建议。

    :param chat: 可选，注入 (system, user) -> str（测试用）。
    """

    def __init__(self, policy: Dict[str, Any] = None, chat: Callable = None,
                 gateway=None, verbose: bool = False):
        self.policy = policy or load_policy()
        self.verbose = verbose
        self._chat = chat
        self._gateway = gateway

    def _call(self, system_prompt: str, user_prompt: str) -> str:
        if self._chat is not None:
            return self._chat(system_prompt, user_prompt)
        if self._gateway is None:
            from llm_gateway import LLMGateway
            self._gateway = LLMGateway(verbose=False)
        d = self.policy.get("dify") or {}
        resp = self._gateway.chat_detailed(
            system_prompt, user_prompt,
            task=str(d.get("task") or "dify_approve"),
            user=str((self.policy.get("ai_approver") or {}).get("role")
                     or "ai_approver"))
        # 网关「全链失败」时**不抛异常**，而是返回配置里的 degraded_reply 文案。
        # 那样下游只报「AI 输出里找不到 JSON 对象」，把「模型根本连不上」
        # 误诊成「模型乱输出」，排障会往完全错误的方向查。
        # 这里显式识别（网关把 model 标成 <degraded>）并抛错，让 error 说真相。
        if getattr(resp, "model", "") == "<degraded>":
            raise RuntimeError(
                "LLM 网关全链失败（Dify 与本地模型均不可用），"
                "拿到的是降级文案而不是模型输出")
        return getattr(resp, "text", resp) or ""

    def advise(self, pack: Dict[str, Any], history: Dict[str, Any],
               rule: Dict[str, Any], route: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        逐站出建议。返回列表与 route["nodes"] 等长、同序。

        某站失败不影响其它站（记 ok=False），但整单裁决会把「有站没出意见」
        当作降级——少一站意见就自动批，等于这一站白设。
        """
        rr = self.policy.get("route_rules") or {}
        values = rr.get("advice_values") or ["agree", "agree_with_conditions",
                                             "oppose", "need_info"]
        out: List[Dict[str, Any]] = []
        for node in (route.get("nodes") or []):
            t0 = time.time()
            material = render_node_material(pack, history, rule, node)
            system_prompt = _NODE_SYSTEM.format(
                title=node.get("title"), duty=node.get("duty"),
                hint=node.get("hint"),
                extra=(f"（本单触发了额外审查：{node['reason']}；"
                       f"请围绕这一点重点看）\n" if node.get("reason") else ""))
            try:
                raw = self._call(system_prompt, material)
            except Exception as e:
                out.append({"node_id": node.get("node_id"),
                            "title": node.get("title"),
                            "level": node.get("level"),
                            "ok": False,
                            "error": f"调用失败：{type(e).__name__}: {e}",
                            "material": material,
                            "latency_ms": int((time.time() - t0) * 1000)})
                continue
            adv, err = parse_node_advice(raw, values)
            rec = {"node_id": node.get("node_id"), "title": node.get("title"),
                   "level": node.get("level"),
                   "material": material,
                   "latency_ms": int((time.time() - t0) * 1000),
                   "ok": bool(adv), "error": err, "raw": (raw or "")[:1500]}
            if adv:
                rec.update(adv)
            out.append(rec)
        return out


def parse_node_advice(raw: str, allowed: List[str] = None,
                      ) -> Tuple[Optional[Dict], str]:
    """
    解析单站建议。容错范围与 parse_verdict 一致：只容忍格式噪音，
    不容忍 advice 不合规、confidence 非数字、key_points 不是数组。
    """
    if not raw or not str(raw).strip():
        return None, "该站 AI 返回空内容"
    text = str(raw).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    if not text.startswith("{"):
        i, j = text.find("{"), text.rfind("}")
        if i < 0 or j <= i:
            return None, "该站 AI 输出里找不到 JSON 对象"
        text = text[i:j + 1]
    try:
        obj = json.loads(text)
    except Exception as e:
        return None, f"该站 AI 输出不是合法 JSON：{e}"
    if not isinstance(obj, dict):
        return None, "该站 AI 输出的 JSON 顶层不是对象"

    vals = allowed or ["agree", "agree_with_conditions", "oppose", "need_info"]
    adv = str(obj.get("advice", "")).strip().lower()
    if adv not in vals:
        return None, f"advice「{obj.get('advice')}」不在白名单 {vals} 内"
    try:
        conf = float(obj.get("confidence"))
    except Exception:
        return None, f"confidence「{obj.get('confidence')}」不是数字"
    if not (0.0 <= conf <= 1.0):
        return None, f"confidence {conf} 越界（须 0~1）"
    kp = obj.get("key_points")
    if not isinstance(kp, list):
        return None, "key_points 不是数组"
    return {
        "advice": adv, "confidence": round(conf, 4),
        "key_points": [str(x) for x in kp],
        "conditions": [str(x) for x in (obj.get("conditions") or [])],
        "questions": [str(x) for x in (obj.get("questions") or [])],
        "one_liner": str(obj.get("one_liner") or ""),
    }, ""


# ===========================================================================
# ④ 裁决：纯函数，可单测，不碰 IO
# ===========================================================================
def arbitrate_route(pack: Dict[str, Any], rule: Dict[str, Any],
                    route: Dict[str, Any],
                    advices: List[Dict[str, Any]],
                    policy: Dict[str, Any],
                    required_level: int = None) -> Dict[str, Any]:
    """
    多站裁决：把「路线 + 每站建议 + 规则结论 + 权限额度」合成一个动作。

    与单站 arbitrate 的差别不只是「多了几个节点」：
      · 单站时 AI 只要说 agree 就能过；多站时**每一站**都得 agree，
        任何一站 oppose 就停在那站。
      · 平均置信度要过门槛（min_avg_confidence），不是看某一站。
      · 有站没出意见（AI 挂了/解析失败）→ 降级，不能用「剩下的站都说好」凑过去。

    安全边界仍是那四道，且顺序不变：硬规则 → 数据完整性 → 权限额度 →
    节点意见 → 规则风险等级。**权限永远在 AI 意见之前判**。
    """
    lim = policy.get("limits") or {}
    rr = policy.get("route_rules") or {}
    ai_role = policy.get("ai_approver") or {}
    ai_level = int(ai_role.get("level") or 2)
    max_auto = _f(lim.get("max_auto_amount")) or 0.0
    min_avg = _f(rr.get("min_avg_confidence")) or 1.0
    amount = _f((pack.get("metrics") or {}).get("amount")) or 0.0
    blockers: List[str] = []
    nodes = route.get("nodes") or []

    def out(action: str, reason: str, stage: str = "") -> Dict[str, Any]:
        return {"action": action, "reason": reason, "blockers": blockers,
                "stage": stage or "裁决", "route_nodes": len(nodes)}

    # 1) 一票否决：AI 无权翻案
    if rule.get("hard_block"):
        blockers.extend(rule.get("blockers") or [])
        return out("recommend_reject", "规则层一票否决；AI 意见仅作参考，须人工复核",
                   "规则层")

    # 2) 核心数据不全：不许闭眼签字
    core_missing = [m for m in (pack.get("missing") or []) if m in _CORE_METRICS]
    if core_missing:
        blockers.append(f"核心指标缺失：{', '.join(core_missing)}")
        return out("human_review", "关键数据不足，无法自动批", "规则层")

    # 3) 权限额度：等级（含路线最高等级）与业务额度取严 —— 必须在 AI 意见之前
    route_level = int(route.get("max_level") or 0)
    need = max([x for x in (required_level, route_level) if x] or [2])
    if need > ai_level:
        blockers.append(f"本单路线要求审批等级 level≥{need}"
                        f"（路线最高站 {route_level}），AI 审批员等级 {ai_level} 不足")
        return out("human_review", "超出 AI 审批等级，必须更高角色审批", "权限")
    if amount > max_auto:
        blockers.append(f"金额 {amount:,.2f} 超 AI 自动批额度 {max_auto:,.2f}")
        return out("human_review", "超出 AI 自动批金额上限", "权限")

    # 4) 规则层有 high → 建议拒绝。**多站不放宽这条**：即便每个岗位都说同意，
    #    规则层已经算出客观高危指标，就不该由 AI 把它的严重性"谈下来"。
    highs = [f for f in (rule.get("findings") or []) if f.get("level") == "high"]
    if highs:
        blockers.extend(f["text"] for f in highs)
        return out("recommend_reject",
                   "存在高风险指标，建议拒绝；由人工最终确认", "规则层")

    # 5) 有站没出意见 → 降级，不能用其余站凑过去
    missing_nodes = [a.get("title") or a.get("node_id") for a in advices
                     if not a.get("ok")]
    if not advices:
        blockers.append("路线上没有任何节点意见")
        return out("degraded", "AI 未产出任何节点意见，回落人工", "AI")
    if missing_nodes:
        blockers.append(f"以下节点未产出有效意见：{'、'.join(missing_nodes)}")
        return out("degraded",
                   f"{len(missing_nodes)}/{len(advices)} 个节点 AI 未给出有效意见，"
                   f"不能按剩余节点自动批", "AI")

    # 5) 逐站意见
    opposes = [a for a in advices if a.get("advice") == "oppose"]
    if opposes and rr.get("veto_on_any_oppose", True):
        for a in opposes:
            blockers.append(f"【{a['title']}】{a.get('one_liner') or '反对'}"
                            + (f"（依据：{'；'.join(a.get('key_points') or [])}）"
                               if a.get("key_points") else ""))
        return out("recommend_reject",
                   f"{opposes[0]['title']} 站明确反对，建议拒批；由人工最终确认",
                   opposes[0]["title"])

    need_info = [a for a in advices if a.get("advice") == "need_info"]
    if need_info:
        for a in need_info:
            blockers.append(f"【{a['title']}】需要补充："
                            f"{'；'.join(a.get('questions') or []) or '未说明'}")
        return out("human_review",
                   f"{need_info[0]['title']} 站要求补充信息", need_info[0]["title"])

    cond = [a for a in advices if a.get("advice") == "agree_with_conditions"]
    if cond:
        for a in cond:
            blockers.append(f"【{a['title']}】有条件同意："
                            f"{'；'.join(a.get('conditions') or []) or '未说明条件'}")
        return out("human_review",
                   f"{cond[0]['title']} 站有条件同意，须确认条件后放行",
                   cond[0]["title"])

    # 6) 置信度：所有站平均，不是取某站最高
    confs = [float(a.get("confidence") or 0) for a in advices]
    avg = round(sum(confs) / len(confs), 4) if confs else 0.0
    if avg < min_avg:
        blockers.append(f"各站平均置信度 {avg} < 门槛 {min_avg}")
        return out("human_review", "AI 各站平均置信度不足", "AI")

    # 7) 规则层风险等级
    if rule.get("risk_level") != "low":
        return out("human_review",
                   f"规则风险等级为 {rule.get('risk_level')}，不满足自动通过条件",
                   "规则层")

    return out("auto_approve",
               f"{len(nodes)} 站全部同意（平均置信度 {avg}），"
               f"规则风险 {rule.get('risk_level')}（{rule.get('risk_score')} 分）",
               "全部")


def build_handover(row: Dict[str, Any], policy: Dict[str, Any] = None
                   ) -> Dict[str, Any]:
    """
    生成「交给人工的交接单」。

    这是整个模块对人工最有用的一块，也是「转人工」和「把球丢过去」的分界线：

      · AI 已经查到什么 → advices（每站的结论、依据、条件、待问问题）
      · 卡在哪一站      → stage
      · 为什么不敢定    → why_uncertain（blockers 原文）
      · 建议人工怎么处置 → recommended_action

    人拿到这一页就能决策，不需要把分析重跑一遍。这也是「自动化审批」在工厂
    能落地的关键——审批人不缺权力，缺的是判断依据。
    """
    advices = [a for a in (row.get("advices") or []) if a.get("ok")]
    action = row.get("action") or "human_review"
    stage = row.get("stage") or ""
    confs = [float(a.get("confidence") or 0) for a in advices
             if a.get("confidence") is not None]
    avg_conf = round(sum(confs) / len(confs), 4) if confs else None

    if action == "auto_approve":
        conclusion = "AI 已给出明确结论并自动批准"
    elif action == "recommend_reject":
        conclusion = "AI 给出明确结论：建议拒批"
    elif action == "degraded":
        conclusion = "AI 未能给出完整意见（已降级），下列建议仅供参照"
    else:
        conclusion = "AI 未取得一致结论，转人工判断（下方为各站建议）"

    # 推荐处置：按「卡住的原因」给动作，不是给一句「请人工审核」
    if action == "auto_approve":
        rec = "无需处置；如需抽查，优先看下方各站 key_points。"
    elif action == "degraded":
        rec = ("按传统方式人工审核。已算好的指标可直接引用"
               f"（{'、'.join(sorted((row.get('metrics') or {}).keys()))[:120]}…），"
               "不必重新取数。")
    elif stage in ("权限",):
        rec = ("该单超出 AI 授权范围（等级或额度），须由更高角色审批；"
               "AI 各站意见可作参考，但不能代替等级要求。")
    elif action == "recommend_reject":
        rec = ("存在明确反对或一票否决项。请核实 why_uncertain 里的依据；"
               "确认属实即驳回，并让采购员按各站的条件/问题重新提交。")
    else:
        rec = ("请先看 stage 指明的那一站。若卡在资料不全，让采购员回答 "
               "questions_for_requester 后重新提交；若卡在条件，确认条件落实即可放行。")

    return {
        "conclusion": conclusion,
        "avg_confidence": avg_conf,
        "stage": stage,
        "why_uncertain": list(row.get("blockers") or []),
        "recommended_action": rec,
        "questions_for_requester": [
            f"【{a.get('title')}】{q}"
            for a in advices for q in (a.get("questions") or [])],
        "advices": [{
            "stage": a.get("node_id"), "title": a.get("title"),
            "ok": bool(a.get("ok")), "error": a.get("error"),
            "advice": a.get("advice"), "confidence": a.get("confidence"),
            "one_liner": a.get("one_liner"),
            "key_points": a.get("key_points") or [],
            "conditions": a.get("conditions") or [],
            "questions": a.get("questions") or [],
        } for a in (row.get("advices") or [])],
    }


def format_handover(h: Dict[str, Any]) -> str:
    """把交接单渲染成人能一眼读完的文本（给 `approval.py --ai-advice` 用）。"""
    if not h:
        return "（该单没有 AI 交接单：可能尚未预审）"
    lines = [
        "=" * 72,
        f"AI 交接单：{h.get('conclusion')}",
        "=" * 72,
        f"卡住的环节：{h.get('stage') or '-'}",
        f"各站平均置信度：{h.get('avg_confidence')}",
        "",
        "── 各站意见 ──",
    ]
    advs = h.get("advices") or []
    if not advs:
        lines.append("（无）")
    for a in advs:
        # 没出意见的站要把**原因**写出来。只写「无结论」等于让人再去翻日志，
        # 而这一行恰恰是排障最需要的（网关全链失败 / 输出不合规 / 调用超时）。
        if a.get("ok") is False or not a.get("advice"):
            lines.append(f"[未产出] {a.get('title')}")
            if a.get("error"):
                lines.append(f"    原因：{a['error']}")
            continue
        mark = {"agree": "[同意]", "agree_with_conditions": "[有条件同意]",
                "oppose": "[反对]", "need_info": "[资料不足]"}.get(
                    a.get("advice") or "", "[未给出]")
        lines.append(f"{mark} {a.get('title')}（{a.get('advice') or '无结论'}"
                     f"，置信度 {a.get('confidence')}）")
        if a.get("one_liner"):
            lines.append(f"    结论：{a['one_liner']}")
        for kp in (a.get("key_points") or [])[:6]:
            lines.append(f"    · {kp}")
        for c in (a.get("conditions") or []):
            lines.append(f"    + 条件：{c}")
        for q in (a.get("questions") or []):
            lines.append(f"    ? 待问：{q}")
    if h.get("why_uncertain"):
        lines += ["", "── 为什么不敢直接定 ──"]
        lines += [f"  · {b}" for b in h["why_uncertain"]]
    if h.get("questions_for_requester"):
        lines += ["", "── 需要采购员回答的问题 ──"]
        lines += [f"  {i}. {q}" for i, q in
                  enumerate(h["questions_for_requester"], 1)]
    lines += ["", "── 建议你这样做 ──", f"  {h.get('recommended_action')}",
              "=" * 72]
    return "\n".join(lines)


def arbitrate(pack: Dict[str, Any], rule: Dict[str, Any],
              ai: Dict[str, Any], policy: Dict[str, Any],
              required_level: int = None) -> Dict[str, Any]:
    """
    把「规则结论 + AI 结论 + 权限额度」合成一个可执行动作。

    :param required_level: 该动作在当前金额下要求的审批等级（来自 ApprovalGate）
    :return: {"action": ..., "reason": ..., "blockers": [...]}

    action 语义：
      auto_approve      规则与 AI 都无异议，且额度在 AI 权限内 → 可以自动批
      human_review      有疑点但不够拒，转人工（附 AI 意见）
      recommend_reject  有硬伤，AI 建议拒绝；**仍然转人工**，只是带上拒绝建议
      degraded          AI 不可用，回落纯规则结论，全部转人工
    """
    lim = policy.get("limits") or {}
    ai_role = policy.get("ai_approver") or {}
    ai_level = int(ai_role.get("level") or 2)
    max_auto = _f(lim.get("max_auto_amount")) or 0.0
    min_conf = _f(lim.get("min_ai_confidence")) or 1.0
    amount = _f((pack.get("metrics") or {}).get("amount")) or 0.0
    blockers: List[str] = []

    def out(action: str, reason: str) -> Dict[str, Any]:
        return {"action": action, "reason": reason, "blockers": blockers}

    # 1) 硬规则一票否决：AI 无权翻案
    if rule.get("hard_block"):
        blockers.extend(rule.get("blockers") or [])
        return out("recommend_reject",
                   "规则层一票否决；AI 意见仅作参考，须人工复核")

    # 2) 数据不全：不许闭眼签字
    core_missing = [m for m in (pack.get("missing") or []) if m in _CORE_METRICS]
    if core_missing:
        blockers.append(f"核心指标缺失：{', '.join(core_missing)}")
        return out("human_review", "关键数据不足，无法自动批")

    # 3) AI 不可用 → 降级，不假装通过
    if not ai.get("ok"):
        blockers.append(f"AI 不可用：{ai.get('error') or '未知原因'}")
        return out("degraded", "AI 研判失败，回落规则结论并转人工")

    v = ai.get("verdict") or {}

    # 4) 额度：等级够不够 + 业务线额度够不够，两者取严
    if required_level is not None and required_level > ai_level:
        blockers.append(
            f"该单要求审批等级 level≥{required_level}，AI 审批员等级 {ai_level} 不足")
        return out("human_review", "超出 AI 审批等级，必须更高角色审批")
    if amount > max_auto:
        blockers.append(f"金额 {amount:,.2f} 超 AI 自动批额度 {max_auto:,.2f}")
        return out("human_review", "超出 AI 自动批金额上限")

    # 5) 规则层有 high → 建议拒绝，但仍转人工
    highs = [f for f in (rule.get("findings") or []) if f.get("level") == "high"]
    if highs:
        blockers.extend(f["text"] for f in highs)
        return out("recommend_reject",
                   "存在高风险指标，建议拒绝；由人工最终确认")

    # 6) AI 自己的结论
    if v.get("decision") == "reject":
        blockers.extend(v.get("key_factors") or [])
        return out("recommend_reject", "AI 判定建议拒绝，由人工最终确认")
    if v.get("decision") == "escalate":
        blockers.extend(v.get("key_factors") or [])
        return out("human_review", "AI 认为须升级审批")

    # 7) 置信度与风险等级门槛
    if float(v.get("confidence") or 0) < min_conf:
        blockers.append(f"AI 置信度 {v.get('confidence')} < 门槛 {min_conf}")
        return out("human_review", "AI 置信度不足")
    if rule.get("risk_level") != "low":
        return out("human_review",
                   f"规则风险等级为 {rule.get('risk_level')}，不满足自动通过条件")
    if lim.get("require_no_high_finding", True) and highs:
        return out("human_review", "存在高风险指标")

    # 8) 全过：自动批
    return out("auto_approve",
               f"规则风险 {rule.get('risk_level')}（{rule.get('risk_score')} 分）、"
               f"AI {v.get('decision')}（置信度 {v.get('confidence')}）")


# ===========================================================================
# ⑤ 编排 + 落库
# ===========================================================================
_REVIEW_DDL = """
CREATE TABLE IF NOT EXISTS `ai_approval_reviews` (
  `id`             BIGINT AUTO_INCREMENT PRIMARY KEY,
  `tenant_id`      VARCHAR(64)   NOT NULL DEFAULT 'default',
  `request_id`     BIGINT        NULL,
  `po_id`          BIGINT        NULL,
  `po_no`          VARCHAR(32)   NULL,
  `amount`         DECIMAL(14,2) NULL,
  `rule_score`     INT           NULL,
  `rule_level`     VARCHAR(16)   NULL,
  `hard_block`     TINYINT       NOT NULL DEFAULT 0,
  `ai_decision`    VARCHAR(16)   NULL,
  `ai_risk`        VARCHAR(16)   NULL,
  `ai_confidence`  DECIMAL(6,4)  NULL,
  `action`         VARCHAR(24)   NOT NULL,
  `reason`         VARCHAR(1000) NULL,
  `blockers`       JSON          NULL,
  `findings`       JSON          NULL,
  `metrics`        JSON          NULL,
  `missing`        JSON          NULL,
  `ai_raw`         JSON          NULL,
  `model_task`     VARCHAR(64)   NULL,
  `latency_ms`     INT           NULL,
  `created_at`     DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX `idx_tenant_po` (`tenant_id`, `po_id`),
  INDEX `idx_action` (`action`, `created_at` DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

# 节点级建议单独一张表：一次审批 = N 行（每站一行）。
# 好处是能按站查「这个岗位历史上都反对过什么」，也能算「哪一站最常卡单」，
# 这是改善审批流程的原始数据。全部塞进主表 JSON 就查不动了。
_NODE_DDL = """
CREATE TABLE IF NOT EXISTS `ai_approval_node_advices` (
  `id`             BIGINT AUTO_INCREMENT PRIMARY KEY,
  `tenant_id`      VARCHAR(64)   NOT NULL DEFAULT 'default',
  `review_id`      BIGINT        NULL,
  `request_id`     BIGINT        NULL,
  `po_id`          BIGINT        NULL,
  `po_no`          VARCHAR(32)   NULL,
  `amount`         DECIMAL(14,2) NULL,
  `seq`            INT           NOT NULL DEFAULT 0,
  `node_id`        VARCHAR(32)   NOT NULL,
  `node_title`     VARCHAR(64)   NULL,
  `node_level`     INT           NULL,
  `stage_reason`   VARCHAR(500)  NULL,
  `advice`         VARCHAR(32)   NULL,
  `confidence`     DECIMAL(6,4)  NULL,
  `one_liner`      VARCHAR(200)  NULL,
  `key_points`     JSON          NULL,
  `conditions`     JSON          NULL,
  `questions`      JSON          NULL,
  `material`       TEXT          NULL,
  `ok`             TINYINT       NOT NULL DEFAULT 0,
  `error`          VARCHAR(500)  NULL,
  `model_task`     VARCHAR(64)   NULL,
  `latency_ms`     INT           NULL,
  `created_at`     DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX `idx_tenant_po` (`tenant_id`, `po_id`),
  INDEX `idx_node_advice` (`node_id`, `advice`, `created_at` DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


class AIApprovalFlow:
    """
    一次完整的 AI 预审 = 取数 → 规则 → 规划路线 → 逐站建议 → 汇总裁决 → 落库
                        （→ 必要时自动批）。

    多站是这一版的重点：同一张单，部门经理 / 采购主管 / 财务 / 总经理
    各看到一份**只属于自己职责**的建议，而不是一段被复制四遍的结论。

    :param gate: approval.ApprovalGate 实例（自动批准要通过它，走同一套权限链）
    :param builder: 可选，注入 DataPackBuilder（测试用）
    :param advisor: 可选，注入 NodeAdvisor（测试用）
    :param reviewer: 可选，旧版单站 reviewer；给了它就走单站退化路径（兼容用）
    """

    def __init__(self, gate=None, policy: Dict[str, Any] = None,
                 builder=None, reviewer=None, advisor=None,
                 verbose: bool = True):
        self.policy = policy or load_policy()
        self.verbose = verbose
        self.gate = gate
        self.rule = RuleEngine(self.policy)
        self.planner = RoutePlanner(self.policy)
        self._builder = builder
        self._reviewer = reviewer
        self._advisor = advisor
        self.available = False
        self._pymysql = None
        self._conn_kw = None
        self._ensure_store()

    # -- 基础设施 -------------------------------------------------------
    def _ensure_store(self):
        try:
            import pymysql
            from memory_store import (MYSQL_HOST, MYSQL_PORT, MYSQL_USER,
                                      MYSQL_PASSWORD, MYSQL_DATABASE,
                                      MYSQL_CHARSET)
            self._pymysql = pymysql
            self._conn_kw = dict(host=MYSQL_HOST, port=MYSQL_PORT,
                                 user=MYSQL_USER, password=MYSQL_PASSWORD,
                                 database=MYSQL_DATABASE, charset=MYSQL_CHARSET,
                                 autocommit=True)
            conn = pymysql.connect(**self._conn_kw)
            try:
                with conn.cursor() as cur:
                    cur.execute(_REVIEW_DDL)
                    cur.execute(_NODE_DDL)
            finally:
                conn.close()
            self.available = True
        except Exception:
            self.available = False

    def _save(self, row: Dict[str, Any]) -> Optional[int]:
        """落 AI 预审快照。库不可用则只打印，不静默丢。"""
        if not self.available:
            if self.verbose:
                print("  [AIApprovalFlow] 审批库不可用，AI 预审快照未落库")
            return None
        sql = ("INSERT INTO ai_approval_reviews (tenant_id, request_id, po_id, "
               "po_no, amount, rule_score, rule_level, hard_block, ai_decision, "
               "ai_risk, ai_confidence, action, reason, blockers, findings, "
               "metrics, missing, ai_raw, model_task, latency_ms) "
               "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"
               "%s,%s)")
        d = lambda o: json.dumps(o, ensure_ascii=False, default=str)
        params = (row.get("tenant_id"), row.get("request_id"),
                  row.get("po_id"), row.get("po_no"), row.get("amount"),
                  row.get("rule_score"), row.get("rule_level"),
                  1 if row.get("hard_block") else 0,
                  row.get("ai_decision"), row.get("ai_risk"),
                  row.get("ai_confidence"), row.get("action"),
                  (row.get("reason") or "")[:1000],
                  d(row.get("blockers")), d(row.get("findings")),
                  d(row.get("metrics")), d(row.get("missing")),
                  d(row.get("ai_raw")), row.get("model_task"),
                  row.get("latency_ms"))
        conn = self._pymysql.connect(**self._conn_kw)
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.lastrowid
        finally:
            conn.close()

    def _save_node_advices(self, row: Dict[str, Any]) -> List[int]:
        """逐站落建议（一次审批 N 行）。库不可用时返回空表，不静默当成功。"""
        advices = row.get("advices") or []
        if not self.available or not advices:
            return []
        route_nodes = {n["node_id"]: n for n in
                       ((row.get("route") or {}).get("nodes") or [])}
        sql = ("INSERT INTO ai_approval_node_advices (tenant_id, review_id, "
               "request_id, po_id, po_no, amount, seq, node_id, node_title, "
               "node_level, stage_reason, advice, confidence, one_liner, "
               "key_points, conditions, questions, material, ok, error, "
               "model_task, latency_ms) "
               "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"
               "%s,%s,%s,%s,%s)")
        ids: List[int] = []
        d = lambda o: json.dumps(o, ensure_ascii=False, default=str)
        try:
            conn = self._pymysql.connect(**self._conn_kw)
            try:
                with conn.cursor() as cur:
                    for i, a in enumerate(advices):
                        nid = a.get("node_id")
                        cur.execute(sql, (
                            row.get("tenant_id"), row.get("id"),
                            row.get("request_id"), row.get("po_id"),
                            row.get("po_no"), row.get("amount"), i,
                            nid, a.get("title"), a.get("level"),
                            (route_nodes.get(nid) or {}).get("reason"),
                            a.get("advice"), a.get("confidence"),
                            (a.get("one_liner") or "")[:200],
                            d(a.get("key_points") or []),
                            d(a.get("conditions") or []),
                            d(a.get("questions") or []),
                            (a.get("material") or "")[:60000],
                            1 if a.get("ok") else 0,
                            (a.get("error") or "")[:500],
                            row.get("model_task"), a.get("latency_ms")))
                        ids.append(cur.lastrowid)
            finally:
                conn.close()
        except Exception as e:
            if self.verbose:
                print(f"  [AIApprovalFlow] 节点建议落库失败：{e}")
            return []
        return ids

    # -- 主流程 ---------------------------------------------------------
    def review_po(self, po_id: int, tenant_id: str = "default",
                  request_id: int = None, clause_fetcher: Callable = None,
                  dry_run: bool = False,
                  node_id: str = None) -> Dict[str, Any]:
        """
        对一张采购单跑完整 AI 预审（多站）。

        :param request_id: 关联的审批单号（有则自动批准走 gate.decide）
        :param clause_fetcher: 可选，(query) -> [{title, content}] 的制度检索函数
        :param node_id: 只出某一站的建议（审批人打开单据时看"我这一站"）。
                        指定它时 route 被裁成单站并标 partial=True，
                        **partial 一律不许自动批**——只跑一站就签字等于没跑。
        """
        builder = self._builder or DataPackBuilder(self.policy, tenant_id)
        pack = builder.build(po_id)

        if not pack.get("ok"):
            rule = self.rule.score(pack)
            route = self.planner.plan(pack, rule)
            verdict = arbitrate_route(pack, rule, route, [], self.policy, None)
            row = self._compose(pack, rule, None, verdict, request_id, route, [],
                                None)
            row["handover"] = build_handover(row, self.policy)
            row["ai_raw"]["handover"] = row["handover"]
            row["reason"] = (f"{row['reason']}｜AI建议："
                             f"{row['handover']['conclusion']}")
            if not dry_run:
                row["id"] = self._save(row)
            return row

        rule = self.rule.score(pack)
        route = self.planner.plan(pack, rule)
        if node_id:
            picked = [n for n in route["nodes"] if n["node_id"] == node_id]
            if not picked:
                raise ValueError(
                    f"节点「{node_id}」不在本单路线上。本单路线："
                    f"{[n['node_id'] for n in route['nodes']]}"
                    f"（触发信号：{route.get('signals')}）")
            route = {**route, "nodes": picked, "partial": True,
                     "full_route_nodes": [n["node_id"]
                                          for n in self.planner.plan(pack, rule)["nodes"]]}

        history = HistoryDigest(builder, self.policy).build(pack)
        clauses = []
        if clause_fetcher is not None and not rule.get("hard_block"):
            try:
                m = pack.get("metrics") or {}
                q = (f"{pack['po'].get('supplier_name')} 采购审批 制度 "
                     f"物料 {m.get('price_deviation_pct')}")
                clauses = clause_fetcher(q) or []
            except Exception:
                clauses = []

        if self._reviewer is not None and self._advisor is None:
            # 旧版单站退化路径：只给了一个 DifyReviewer 的调用方仍能跑，
            # 但结果会被标成 single_stage，不参与多站裁决的自动批。
            ai = self._reviewer.review(pack, rule, clauses)
            advices = [{"node_id": "single", "title": "AI 审批员", "level": 2,
                        "ok": bool(ai.get("ok")),
                        "error": ai.get("error"),
                        "advice": (ai.get("verdict") or {}).get("decision"),
                        "confidence": (ai.get("verdict") or {}).get("confidence"),
                        "key_points": (ai.get("verdict") or {}).get("key_factors"),
                        "one_liner": (ai.get("verdict") or {}).get("suggestion"),
                        "conditions": [], "questions": []}]
            route = {**route, "single_stage": True}
        else:
            advisor = self._advisor or NodeAdvisor(self.policy,
                                                   verbose=self.verbose)
            advices = advisor.advise(pack, history, rule, route)

        required = self._required_level(request_id, pack)
        verdict = arbitrate_route(pack, rule, route, advices, self.policy,
                                  required)
        row = self._compose(pack, rule, None, verdict, request_id, route,
                            advices, history)
        row["clauses"] = clauses
        # 交接单：转人工时必须把「AI 查到了什么、卡在哪、建议怎么处置」一起交出去，
        # 否则「转人工」就只是把球丢过去，人还得从零重算一遍。
        row["handover"] = build_handover(row, self.policy)
        row["ai_raw"]["handover"] = row["handover"]
        if row["action"] != "auto_approve":
            row["reason"] = (
                f"{row['reason']}｜AI建议：{row['handover']['conclusion']}"
                + (f"｜卡在：{row['stage']}" if row.get("stage") else ""))

        if dry_run:
            return row

        row["id"] = self._save(row)
        row["advice_ids"] = self._save_node_advices(row)

        # 只有全路线、全站同意才真正动手；动作仍要经 gate 的权限链
        if verdict["action"] == "auto_approve" and request_id and self.gate \
                and not route.get("partial"):
            row["decided"] = self._auto_decide(request_id, row)
        else:
            row["decided"] = None
        if self.verbose:
            print(f"  [AIApprovalFlow] PO #{po_id} → {verdict['action']}"
                  f"（{verdict['stage']}）：{verdict['reason']}")
        return row

    def _auto_decide(self, request_id: int, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        调 ApprovalGate 自动批准。

        这里**复用** ApprovalGate.decide()，而不是直接 UPDATE 状态——
        因为权限等级校验、禁自审自批、执行器调用、审计落盘都在那里面。
        换句话说：AI 不是绕过审批门，它是拿到了一个 level=2 的审批人身份。
        """
        role = str((self.policy.get("ai_approver") or {}).get("role")
                   or "ai_approver")
        actor = f"ai:{role}"
        note = (f"AI 自动批准（规则 {row.get('rule_level')}/"
                f"{row.get('rule_score')}分；AI {row.get('ai_decision')}, "
                f"置信度 {row.get('ai_confidence')}）：{row.get('reason')}")
        try:
            decided = self.gate.decide(request_id, approve=True,
                                       decided_by=actor, decided_role=role,
                                       note=note)
            audit(actor, "ai_approval.auto_approved", target=str(request_id),
                  result="success", detail=note)
            return {"ok": True, "status": decided.get("status"),
                    "result": decided.get("result")}
        except Exception as e:
            audit(actor, "ai_approval.auto_approve_failed",
                  target=str(request_id), result="blocked", detail=str(e))
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def _required_level(self, request_id, pack) -> Optional[int]:
        """从 gate 拿该动作在当前金额下要求的审批等级；拿不到返回 None。"""
        if not self.gate:
            return None
        try:
            amt = (pack.get("metrics") or {}).get("amount")
            return self.gate.required_level("purchase_order.approve", amt)
        except Exception:
            return None

    def _compose(self, pack, rule, ai, verdict, request_id,
                 route=None, advices=None, history=None) -> Dict[str, Any]:
        """
        组装最终结果行。

        多站信息（route / advices / history）不新建表列，而是塞进主表的
        `ai_raw` JSON 字段——`ai_raw` 本来就是 JSON 类型，塞得下，
        也省掉一次 ALTER TABLE 迁移（生产库上改结构是要排窗口的）。
        节点级细粒度建议另外落 `ai_approval_node_advices`，方便按站查询。
        """
        po = pack.get("po") or {}
        v = (ai or {}).get("verdict") or {}
        d = self.policy.get("dify") or {}
        advices = advices or []
        route = route or {}
        confs = [float(a.get("confidence") or 0) for a in advices
                 if a.get("ok") and a.get("confidence") is not None]
        return {
            "tenant_id": pack.get("tenant_id") or "default",
            "request_id": request_id, "po_id": po.get("id"),
            "po_no": po.get("po_no"), "amount": po.get("amount"),
            "rule_score": rule.get("risk_score"),
            "rule_level": rule.get("risk_level"),
            "hard_block": rule.get("hard_block"),
            "ai_decision": v.get("decision"), "ai_risk": v.get("risk_level"),
            "ai_confidence": v.get("confidence"),
            "action": verdict["action"], "reason": verdict["reason"],
            "stage": verdict.get("stage"),
            "blockers": verdict.get("blockers") or [],
            "findings": rule.get("findings") or [],
            "metrics": pack.get("metrics") or {},
            "missing": pack.get("missing") or [],
            "ai_raw": {
                "ok": (ai or {}).get("ok"),
                "error": (ai or {}).get("error"),
                "verdict": v,
                "latency_ms": (ai or {}).get("latency_ms"),
                # 多站快照：谁规划的路线、每站说了什么、为什么插这一站
                "route": route,
                "advices": advices,
                "avg_confidence": (round(sum(confs) / len(confs), 4)
                                   if confs else None),
            },
            "model_task": str(d.get("task") or ""),
            "latency_ms": sum(int(a.get("latency_ms") or 0) for a in advices)
                          or (ai or {}).get("latency_ms"),
            # 方便调用方直接读，不用翻 ai_raw
            "route": route,
            "advices": advices,
            "history": history or {},
            "pack": pack,
        }


# ===========================================================================
# 小工具
# ===========================================================================
def _f(v) -> Optional[float]:
    """安全转 float：None/''/非数字 → None（不返回 0 冒充数值）。"""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _dstr(v) -> str:
    """
    日期/时间 → 'YYYY-MM-DD'。

    刻意只保留日期：审批判断里「哪一天」够用了，带上时分秒会让
    「实收 vs 期望」的比较逻辑变复杂（一个带时间一个不带，字符串比较就错）。
    """
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()[:10]
    try:
        return v.strftime("%Y-%m-%d")
    except Exception:
        return str(v)[:10]


def _fmt(v, nd: int = 2) -> str:
    """数字格式化；算不出来就显示「数据不足」，不显示 0。"""
    x = _f(v)
    if x is None:
        return "数据不足"
    return f"{x:,.{nd}f}"


def _days_since(dt) -> Optional[int]:
    """距今天数；dt 为 None/无效 → None。"""
    if not dt:
        return None
    try:
        import datetime as _dt
        if isinstance(dt, str):
            s = dt.strip().replace("T", " ")[:19]
            dt = _dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        elif isinstance(dt, _dt.date) and not isinstance(dt, _dt.datetime):
            dt = _dt.datetime(dt.year, dt.month, dt.day)
        return max(0, int((_dt.datetime.now() - dt).total_seconds() // 86400))
    except Exception:
        return None


def _weighted_avg(items: List[Dict], key: str, weight_key: str) -> Optional[float]:
    """按金额加权平均；全部为 None 或权重为 0 → None。"""
    num = den = 0.0
    for it in items:
        v = _f(it.get(key))
        w = _f(it.get(weight_key)) or 0.0
        if v is None or w <= 0:
            continue
        num += v * w
        den += w
    return round(num / den, 2) if den > 0 else None


def _min_of(items: List[Dict], key: str) -> Optional[float]:
    """取最小值（可用天数取最紧的那条物料）；全 None → None。"""
    vals = [_f(it.get(key)) for it in items]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None


# ===========================================================================
# CLI
# ===========================================================================
def _check(policy: Dict[str, Any]) -> int:
    print("=" * 74)
    print("AI 自动审批流 · 配置自检")
    print("=" * 74)
    ai = policy.get("ai_approver") or {}
    lim = policy.get("limits") or {}
    th = policy.get("thresholds") or {}
    print(f"AI 审批员角色  : {ai.get('role')}（level={ai.get('level')}）")
    print(f"自动批金额上限 : {lim.get('max_auto_amount'):,.2f} 元")
    print(f"AI 置信度门槛  : {lim.get('min_ai_confidence')}")
    print(f"审批金额线     : {policy.get('approval_amount_line'):,} 元")
    print(f"拆单识别窗口   : {policy.get('split_window_days')} 天")
    print(f"Dify 路由 key  : {(policy.get('dify') or {}).get('task')}"
          f"（enabled={(policy.get('dify') or {}).get('enabled')}）")
    print(f"知识库         : {policy.get('knowledge_id')}")
    print("-" * 74)
    print(f"{'指标':<26}{'中风险':>10}{'高风险':>10}  方向")
    for k, d in _THRESHOLD_DIR.items():
        r = th.get(k) or {}
        print(f"{_METRIC_LABEL.get(k, k):<26}{str(r.get('medium')):>10}"
              f"{str(r.get('high')):>10}  {'越大越险' if d == 'up' else '越小越险'}")
    print("-" * 74)
    print("月预算：", {t: monthly_budget(policy, t)
                     for t in ["default", *(policy.get("budget") or {}).get(
                         "tenants", {})]})
    # 等级自检：AI 的 level 必须付得起审批线以内的金额
    line = float(policy.get("approval_amount_line") or 100000)
    cap = float(lim.get("max_auto_amount") or 0)
    print(f"等级自检：AI level={ai.get('level')}，审批线 {line:,.0f} 元以上须 level=3；"
          f"AI 自动批上限 {cap:,.0f} 元 "
          f"→ {'在权限内' if cap < line else '⚠ 上限超过 AI 等级可批范围，会被 ApprovalGate 拒绝'}")

    # ---- 多站路线 ----
    nodes_def = policy.get("nodes") or {}
    print("-" * 74)
    print("审批路线（按金额分段）")
    print(f"  {'金额区间':<18}{'站数':>4}  路线")
    prev = 0
    for r in (policy.get("routes") or []):
        up = r.get("up_to")
        try:
            up_f = float(up) if up is not None else 0.0
        except (TypeError, ValueError):
            up_f = 0.0
        rng = f"≤ {up_f:,.0f}" if up_f > 0 else "无上限"
        ids = [str(x) for x in (r.get("nodes") or [])]
        titles = [(nodes_def.get(i) or {}).get("title") or f"⚠{i}" for i in ids]
        print(f"  {rng:<18}{len(ids):>4}  {' → '.join(titles)}")
        prev = up_f
    print("-" * 74)
    print("动态插站（命中信号就插一站）")
    irs = policy.get("insert_rules") or []
    if not irs:
        print("  （无）")
    for ir in irs:
        nid = str(ir.get("node") or "")
        title = (nodes_def.get(nid) or {}).get("title") or f"⚠未定义:{nid}"
        print(f"  {str(ir.get('when') or ''):<16}→ {title:<10}"
              f"（插在 {ir.get('after')} 之后）{ir.get('because') or ''}")
    print("-" * 74)
    print("节点职责")
    for nid, d in nodes_def.items():
        fm = ",".join(d.get("focus_metrics") or [])
        fh = ",".join(d.get("focus_history") or [])
        print(f"  {nid:<17}level={d.get('level', '-'):<2}"
              f"{str(d.get('title')):<8}{d.get('duty') or ''}")
        print(f"  {'':<17}指标[{fm}]  历史[{fh}]")
    # 配置自洽性：插站/路线引用的节点必须都定义过
    bad = []
    for ir in irs:
        if str(ir.get("node") or "") not in nodes_def:
            bad.append(f"insert_rules 引用了未定义节点 {ir.get('node')}")
    for r in (policy.get("routes") or []):
        for nid in (r.get("nodes") or []):
            if str(nid) not in nodes_def:
                bad.append(f"routes 引用了未定义节点 {nid}")
    rr = policy.get("route_rules") or {}
    if not (policy.get("routes") or []):
        bad.append("routes 为空，任何单都会退回单站")
    print("-" * 74)
    if bad:
        print("⚠ 配置问题：")
        for b in bad:
            print(f"  · {b}")
    else:
        print(f"路线自检：{len(policy.get('routes') or [])} 段金额路线 / "
              f"{len(irs)} 条插站规则 / {len(nodes_def)} 个节点，引用全部有定义")
    print(f"汇总裁决：任一站反对是否直接否决自动批 = "
          f"{rr.get('veto_on_any_oppose')}；各站平均置信度门槛 = "
          f"{rr.get('min_avg_confidence')}")
    print("=" * 74)
    return 0 if not bad else 1


def main():
    import argparse
    ap = argparse.ArgumentParser(description="AI 自动审批流（物料采购审核 · 多站）")
    ap.add_argument("--check", action="store_true", help="自检配置（不连库）")
    ap.add_argument("--review", type=int, help="对采购单 id 做 AI 预审并落库")
    ap.add_argument("--request", type=int, help="关联的审批单 id（配合 --review）")
    ap.add_argument("--dry-run", type=int, help="只算不写库")
    ap.add_argument("--sql-check", type=int,
                    help="只跑取数 SQL，打印数字包 + 历史档案 + 审批路线")
    ap.add_argument("--node", default=None,
                    help="只出某一站的建议（如 finance / gm / compliance）；"
                         "单站结果 partial，不会自动批")
    ap.add_argument("--tenant", default="default")
    args = ap.parse_args()

    policy = load_policy()
    if args.check:
        raise SystemExit(_check(policy))

    po_id = args.review or args.dry_run or args.sql_check
    if not po_id:
        ap.print_help()
        return

    builder = DataPackBuilder(policy, args.tenant, verbose=True)
    if not builder.available:
        print(f"⚠ 数据库不可用，无法取数：{builder.error}")
        raise SystemExit(2)
    if args.sql_check:
        pack = builder.build(po_id)
        hist = HistoryDigest(builder, policy).build(pack) if pack.get("ok") else {}
        route = RoutePlanner(policy).plan(pack, RuleEngine(policy).score(pack))
        print(json.dumps({"pack": pack, "history": hist, "route": route},
                         ensure_ascii=False, indent=2, default=str))
        return

    from approval import ApprovalGate, build_default_executors
    gate = ApprovalGate(build_default_executors(), verbose=False)
    flow = AIApprovalFlow(gate=gate, policy=policy, builder=builder)
    row = flow.review_po(po_id, tenant_id=args.tenant,
                         request_id=args.request,
                         dry_run=bool(args.dry_run),
                         node_id=args.node)
    out = {k: v for k, v in row.items() if k != "pack"}
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
