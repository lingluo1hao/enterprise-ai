"""
================================================================================
 approval.py — 高危操作审批门（数字员工的「缰绳」）
================================================================================

 定位
 ----
 通用审批层：数字员工的任何高危写操作不允许直接执行，必须先落一张
 pending 审批单，人工批准后才真正执行，执行结果回填——完整审计链：

     谁申请 → 做什么（payload）→ 为什么（reason）
       → 谁批的（decided_by，什么角色批的）→ 执行成什么样（result）

 审批规则（config/access_rules.yaml approval_rules 段）
 ----
   min_approver_level : 审批人最低审批等级（取 biz_roles.level：1/2/3）
   amount_rules       : 金额分级——payload.amount 超阈值时提高审批等级
                        （销售关单/采购审批 >10 万须 level=3，已定稿）
   硬规则（代码级，不可配置）：
     1. 发起人与审批人不得为同一人（禁自审自批）
     2. 未知角色 / 未注册动作 → 连审批都不给发起（fail-closed）
     3. 拒绝同样须达标等级（防止低权限者恶意否决他人单据）

 执行器（executor）注册制：按 action_type 分发，新增高危动作零侵入。

 AI 预审（ai_review）—— 2026-10-22 新增
 ----
 人工审批链解决「谁都别乱动」，但解决不了「一天几十张单、人点同意点到麻木」。
 `ai_review()` 在人工审批**之前**插一层 AI 预审：取真实数字 → 规则打分
 → Dify 工作流出结论 → 三分支（自动批 / 转人工 / 建议拒绝）。实现见
 `ai_approval.py`，设计说明见 `docs/guides/AI_APPROVAL_WORKFLOW.md`。

 关键边界：AI 自动批准**不是后门**——它调用的是本文件同一个 `decide()`，
 用的是 order_acl 里注册的 `ai_approver`（level=2）。采购单 >10 万要求
 level=3，AI 付不起，改 ai_approval.yaml 的额度上限也绕不过权限链。

 人工侧 CLI：
   python approval.py --list
   python approval.py --approve 3 --by 王工 --role dept_manager
   python approval.py --reject 3 --by 李经理 --role dept_manager --note "证据不足"
   python approval.py --ai-review 3            # 对采购审批单做 AI 预审
================================================================================
"""

import os
import json
import argparse
from typing import Callable, Dict, Optional

from memory_store import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE, MYSQL_CHARSET,
)
from erp_common import get_role_engine, audit

RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "config", "access_rules.yaml")

# 内置兜底规则（yaml 缺失时用；金额线 10 万，与定稿一致）
_DEFAULT_RULES = {
    "sales_order.complete": {"min_approver_level": 2,
                             "amount_rules": [{"above": 100000,
                                               "min_approver_level": 3}]},
    "sales_order.cancel": {"min_approver_level": 2},
    "sales_delivery.release": {"min_approver_level": 2},
    "repair_order.resolve": {"min_approver_level": 2},
    "repair_order.cancel": {"min_approver_level": 2},
    "purchase_order.approve": {"min_approver_level": 2,
                               "amount_rules": [{"above": 100000,
                                                 "min_approver_level": 3}]},
    "inventory.adjust": {"min_approver_level": 2},
}

_APPROVAL_DDL = """
CREATE TABLE IF NOT EXISTS `approval_requests` (
  `id`             BIGINT AUTO_INCREMENT PRIMARY KEY,
  `tenant_id`      VARCHAR(64)   NOT NULL DEFAULT 'default',
  `action_type`    VARCHAR(64)   NOT NULL,
  `payload`        JSON          NOT NULL,
  `reason`         VARCHAR(500)  NULL,
  `requested_by`   VARCHAR(64)   NOT NULL DEFAULT 'digital_employee',
  `requested_role` VARCHAR(64)   NULL,
  `status`         VARCHAR(16)   NOT NULL DEFAULT 'pending',
  `decided_by`     VARCHAR(64)   NULL,
  `decided_role`   VARCHAR(64)   NULL,
  `decided_at`     DATETIME      NULL,
  `result`         JSON          NULL,
  `created_at`     DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX `idx_status` (`status`, `created_at` DESC),
  INDEX `idx_tenant_status` (`tenant_id`, `status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

_COLS = ("id, tenant_id, action_type, payload, reason, requested_by, "
         "requested_role, status, decided_by, decided_role, decided_at, "
         "result, created_at")


def _loads(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def load_approval_rules() -> Dict:
    """读 yaml 的 approval_rules；缺失/异常用内置兜底（不放权）。"""
    try:
        import yaml
        with open(RULES_PATH, "r", encoding="utf-8") as f:
            rules = (yaml.safe_load(f) or {}).get("approval_rules")
        if isinstance(rules, dict) and rules:
            return rules
    except Exception:
        pass
    return _DEFAULT_RULES


def required_level_for(action_type: str, amount: float = None,
                       rules: Dict = None) -> int:
    """
    该动作在当前金额下要求的审批等级（1/2/3）。

    模块级函数而不是只做方法——因为 `dify_kb_server.py` 的 `/ai/pack` 端点
    也要用同一套算法把「这张单要几级」告诉 Dify 工作流。抽出来的目的是让
    **等级判定只有一个权威实现**，避免副本各改各的导致 AI 侧和审批侧口径不一。
    """
    rs = rules if isinstance(rules, dict) else load_approval_rules()
    rule = rs.get(action_type) or {}
    level = int(rule.get("min_approver_level", 2))
    for ar in rule.get("amount_rules") or []:
        if amount is not None and float(amount) > float(ar.get("above", 0)):
            level = max(level, int(ar.get("min_approver_level", level)))
    return level


class ApprovalGate:
    """
    审批门：request（挂起）→ decide（人工批准/拒绝，批准即执行）。

    :param executors: {action_type: fn(payload: dict, actor: str) -> Any}
                      执行结果须可 JSON 序列化。
    """

    def __init__(self, executors: Dict[str, Callable] = None,
                 ensure_schema: bool = True, verbose: bool = True):
        self.verbose = verbose
        self.executors: Dict[str, Callable] = dict(executors or {})
        self.rules = load_approval_rules()
        self.roles = get_role_engine()
        self.available = False
        self._fallback_requests: list = []
        self._fallback_seq = 0
        try:
            import pymysql
            self._pymysql = pymysql
            self._conn_kw = dict(
                host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
                password=MYSQL_PASSWORD, database=MYSQL_DATABASE,
                charset=MYSQL_CHARSET, autocommit=True,
            )
            if ensure_schema:
                self._execute(_APPROVAL_DDL)
            self.available = True
            if verbose:
                print(f"  [ApprovalGate] 连接成功: "
                      f"{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}")
        except Exception as e:
            self._pymysql = None
            if verbose:
                print(f"  [ApprovalGate] 连接失败，降级为内存模式: {e}")

    def register_executor(self, action_type: str, fn: Callable):
        """注册/替换高危动作执行器（新动作零侵入接入）。"""
        self.executors[action_type] = fn

    # ------------------------------------------------------------------
    # 基础设施
    # ------------------------------------------------------------------
    def _execute(self, sql: str, params: tuple = ()):
        """SELECT → dict 行列表；INSERT → cursor.lastrowid（D5 修复口径一致）。"""
        conn = self._pymysql.connect(**self._conn_kw)
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            if cursor.description:
                cols = [c[0] for c in cursor.description]
                return [dict(zip(cols, r)) for r in cursor.fetchall()]
            return cursor.lastrowid or None
        finally:
            conn.close()

    @staticmethod
    def _normalize(row: Dict) -> Dict:
        row = dict(row)
        row["payload"] = _loads(row.get("payload"))
        row["result"] = _loads(row.get("result"))
        return row

    def _row(self, request_id) -> Optional[Dict]:
        if self.available:
            rows = self._execute(
                f"SELECT {_COLS} FROM approval_requests WHERE id = %s",
                (request_id,))
            return self._normalize(rows[0]) if rows else None
        for r in self._fallback_requests:
            if r["id"] == request_id:
                return dict(r)
        return None

    # ------------------------------------------------------------------
    # 规则：该动作在当前金额下要求的审批等级
    # ------------------------------------------------------------------
    def required_level(self, action_type: str, amount: float = None) -> int:
        """转调模块级 required_level_for（唯一权威实现，见其 docstring）。"""
        return required_level_for(action_type, amount, self.rules)

    # ------------------------------------------------------------------
    # 数字员工侧：发起高危操作（只挂起，不执行）
    # ------------------------------------------------------------------
    def request(self, action_type: str, payload: Dict, reason: str = "",
                requested_by: str = "digital_employee",
                requested_role: str = None,
                tenant_id: str = "default") -> int:
        """
        创建 pending 审批单。返回审批单 id。绝不执行动作。

        :param amount: 放 payload["amount"] 里（金额分级审批依据）
        """
        if action_type not in self.executors:
            raise ValueError(
                f"未注册的高危动作「{action_type}」，拒绝发起审批。"
                f"已注册：{sorted(self.executors)}")
        if requested_role and self.roles.level_of(requested_role, tenant_id) < 1:
            raise ValueError(f"未知发起角色「{requested_role}」，拒绝发起")
        if self.available:
            return self._execute(
                "INSERT INTO approval_requests (tenant_id, action_type, "
                "payload, reason, requested_by, requested_role) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (tenant_id, action_type,
                 json.dumps(payload or {}, ensure_ascii=False),
                 reason, requested_by, requested_role))
        self._fallback_seq += 1
        self._fallback_requests.append({
            "id": self._fallback_seq, "tenant_id": tenant_id,
            "action_type": action_type, "payload": payload or {},
            "reason": reason, "requested_by": requested_by,
            "requested_role": requested_role, "status": "pending",
            "decided_by": None, "decided_role": None, "decided_at": None,
            "result": None, "created_at": "fallback",
        })
        return self._fallback_seq

    # ------------------------------------------------------------------
    # 人工侧：审批
    # ------------------------------------------------------------------
    def decide(self, request_id, approve: bool, decided_by: str,
               decided_role: str = "user", note: str = "") -> Dict:
        """
        审批。批准 = 立即执行 executor 并回填结果；拒绝 = 动作不执行。

        校验链（任一不过即拒绝处理，落审计）：
          1. 单据须 pending（不可重复审批）
          2. 审批角色等级 ≥ 该动作在当前金额下要求的等级
          3. 审批人 ≠ 发起人（禁自审自批）
        """
        r = self._row(request_id)
        if not r:
            raise ValueError(f"审批单 #{request_id} 不存在")
        tenant_id = r.get("tenant_id", "default")
        if r["status"] != "pending":
            raise ValueError(
                f"审批单 #{request_id} 已处理（status={r['status']}），不可重复审批")

        level = self.roles.level_of(decided_role, tenant_id)
        required = self.required_level(r["action_type"],
                                       (r.get("payload") or {}).get("amount"))
        if level < required:
            audit(decided_by, f"approval.{r['action_type']}",
                  target=str(request_id), result="blocked",
                  detail=f"审批角色「{decided_role}」等级 {level} < 要求 {required}")
            raise PermissionError(
                f"审批角色「{decided_role}」等级不足（当前 {level}，须 ≥{required}）——"
                f"「{r['action_type']}」"
                + (f" 金额 {(r.get('payload') or {}).get('amount')} 超 10 万"
                   "须 level=3；" if required == 3 else "")
                + " 须更高角色审批（已审计）")
        if decided_by == r.get("requested_by"):
            audit(decided_by, f"approval.{r['action_type']}",
                  target=str(request_id), result="blocked",
                  detail="发起人=审批人，自审自批被拒")
            raise PermissionError("发起人与审批人为同一人，禁止自审自批（已审计）")

        if not approve:
            result = {"rejected": True, "note": note or ""}
            self._write(request_id, "rejected", decided_by, decided_role, result)
            audit(decided_by, f"approval.{r['action_type']}.rejected",
                  target=str(request_id), result="success",
                  detail=note or "")
            print(f"  [ApprovalGate] 审批单 #{request_id} 已拒绝"
                  f"（{decided_by}/{decided_role}）——动作不执行")
            return self._row(request_id)

        # 批准 → 立即执行；失败如实回填（不装成功）
        fn = self.executors.get(r["action_type"])
        try:
            outcome = fn(r["payload"], actor=decided_by)
            result = {"ok": True, "outcome": outcome}
        except Exception as e:
            outcome = None
            result = {"ok": False, "error": str(e)}
            print(f"  [ApprovalGate] ⚠ 审批单 #{request_id} 批准后执行失败：{e}")
        self._write(request_id, "approved", decided_by, decided_role, result)
        audit(decided_by, f"approval.{r['action_type']}.approved",
              target=str(request_id), result="success",
              detail=f"executor {'ok' if result['ok'] else 'failed'}")
        return self._row(request_id)

    def _write(self, request_id, status, decided_by, decided_role, result):
        if self.available:
            self._execute(
                "UPDATE approval_requests SET status = %s, decided_by = %s, "
                "decided_role = %s, decided_at = NOW(), result = %s "
                "WHERE id = %s",
                (status, decided_by, decided_role,
                 json.dumps(result, ensure_ascii=False, default=str),
                 request_id))
        else:
            r = next(x for x in self._fallback_requests if x["id"] == request_id)
            r.update({"status": status, "decided_by": decided_by,
                      "decided_role": decided_role, "decided_at": "fallback",
                      "result": result})

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def list_pending(self, tenant_id: str = None) -> list:
        if self.available:
            sql = (f"SELECT {_COLS} FROM approval_requests "
                   "WHERE status = 'pending'")
            params: tuple = ()
            if tenant_id:
                sql += " AND tenant_id = %s"
                params = (tenant_id,)
            sql += " ORDER BY id DESC"
            return [self._normalize(r)
                    for r in (self._execute(sql, params) or [])]
        return [dict(r) for r in self._fallback_requests
                if r["status"] == "pending"
                and (not tenant_id or r["tenant_id"] == tenant_id)]

    def get(self, request_id) -> Optional[Dict]:
        return self._row(request_id)

    # ------------------------------------------------------------------
    # AI 预审（物料采购审核）
    # ------------------------------------------------------------------
    # AI 预审目前**只覆盖** purchase_order.approve。理由是取数口径必须与业务
    # 真实字段一一对应，其他动作（关单/报修结算/盘点调整）还没有定义「用哪些
    # 数字判断是否合理」，硬做就是编指标。这里宁可拒绝，也不假装支持。
    AI_REVIEW_ACTIONS = ("purchase_order.approve",)

    def ai_review(self, request_id, flow=None, clause_fetcher=None,
                  dry_run: bool = False, node_id: str = None) -> Dict:
        """
        对一张 pending 审批单跑 AI 预审（多站）。

        :param flow: 可选，注入 ai_approval.AIApprovalFlow（测试/复用）
        :param clause_fetcher: 可选，(query) -> [{title, content}] 制度检索
        :param dry_run: True 只算不写库、不执行
        :param node_id: 只出某一站建议（审批人看"我这一站"）；单站结果不会自动批
        :return: 预审结果行，含：
                   action        auto_approve / human_review / recommend_reject
                                 / degraded
                   stage         路线上卡在哪一站
                   advices       每站结论（agree / 有条件 / 反对 / 资料不足）
                   handover      给人工的交接单（结论 + 依据 + 待问问题 + 处置建议）
        """
        r = self._row(request_id)
        if not r:
            raise ValueError(f"审批单 #{request_id} 不存在")
        if r["status"] != "pending":
            raise ValueError(
                f"审批单 #{request_id} 已处理（status={r['status']}），无需预审")
        action_type = r.get("action_type")
        if action_type not in self.AI_REVIEW_ACTIONS:
            raise ValueError(
                f"AI 预审暂不支持动作「{action_type}」——"
                f"该动作还没有定义取数口径（支持：{list(self.AI_REVIEW_ACTIONS)}）。"
                f"拒绝用编造的指标做审批。")
        payload = r.get("payload") or {}
        po_id = payload.get("po_id")
        if not po_id:
            raise ValueError(
                f"审批单 #{request_id} 的 payload 缺 po_id，无法取数（payload={payload}）")

        from ai_approval import AIApprovalFlow
        flow = flow or AIApprovalFlow(gate=self, verbose=False)
        return flow.review_po(int(po_id),
                              tenant_id=r.get("tenant_id") or "default",
                              request_id=request_id,
                              clause_fetcher=clause_fetcher,
                              dry_run=dry_run,
                              node_id=node_id)


# ----------------------------------------------------------------------
# 业务执行器注册表（7 个高危动作 → 各 Service 的 execute_*）
# ----------------------------------------------------------------------
def build_default_executors(md=None, inv=None, sales=None, repair=None,
                            purchase=None) -> Dict[str, Callable]:
    """
    组装全量业务审批执行器。payload 统一带 tenant_id（执行时校验租户）。

    executor 签名 fn(payload: dict, actor: str) -> Any（actor = 审批人，
    落进单据审计字段——「谁批的」全程可追溯）。
    """
    from master_data import MasterData
    from inventory_system import InventoryService
    from sales_system import SalesService
    from repair_system import RepairService
    from purchase_system import PurchaseService

    md = md or MasterData(verbose=False)
    inv = inv or InventoryService(verbose=False)
    sales = sales or SalesService(inventory=inv, master=md, verbose=False)
    repair = repair or RepairService(inventory=inv, master=md, verbose=False)
    purchase = purchase or PurchaseService(inventory=inv, master=md,
                                           verbose=False)

    def t(payload) -> str:
        return (payload or {}).get("tenant_id", "default")

    def sales_complete(p, actor):
        o = sales.execute_complete(p["order_id"], t(p), actor)
        return {"order_no": o["order_no"], "status": o["status"]}

    def sales_cancel(p, actor):
        o = sales.execute_cancel(p["order_id"], t(p), actor)
        return {"order_no": o["order_no"], "status": o["status"]}

    def delivery_release(p, actor):
        d = sales.execute_release(p["delivery_id"], t(p), actor)
        return {"delivery_no": d["delivery_no"], "status": d["status"]}

    def repair_resolve(p, actor):
        o = repair.execute_resolve(p["order_id"], t(p), actor,
                                   resolution=p.get("resolution", ""),
                                   downtime_hours=p.get("downtime_hours"))
        return {"order_no": o["order_no"], "status": o["status"]}

    def repair_cancel(p, actor):
        o = repair.execute_cancel(p["order_id"], t(p), actor)
        return {"order_no": o["order_no"], "status": o["status"]}

    def po_approve(p, actor):
        o = purchase.execute_approve(p["po_id"], t(p), actor)
        return {"po_no": o["po_no"], "status": o["status"]}

    def inv_adjust(p, actor):
        lv = inv.adjust(t(p), p["warehouse_id"], p["product_id"],
                        int(p["delta"]), ref_no=p.get("ref_no", "ADJ"),
                        operator=actor, remark=p.get("remark", ""))
        return lv

    return {
        "sales_order.complete": sales_complete,
        "sales_order.cancel": sales_cancel,
        "sales_delivery.release": delivery_release,
        "repair_order.resolve": repair_resolve,
        "repair_order.cancel": repair_cancel,
        "purchase_order.approve": po_approve,
        "inventory.adjust": inv_adjust,
    }


# ----------------------------------------------------------------------
# CLI：人工审批入口
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="高危操作审批（数字员工缰绳）")
    parser.add_argument("--list", action="store_true", help="查看待审清单")
    parser.add_argument("--show", type=int, help="查看某张审批单详情")
    parser.add_argument("--approve", type=int, help="批准并立即执行")
    parser.add_argument("--reject", type=int, help="拒绝（动作不执行）")
    parser.add_argument("--ai-review", type=int, dest="ai_review",
                        help="对采购审批单做 AI 预审（物料审核·多站）")
    parser.add_argument("--ai-advice", type=int, dest="ai_advice",
                        help="打印该单的 AI 交接单（各站建议 + 待确认问题）")
    parser.add_argument("--dry-run", action="store_true",
                        help="配合 --ai-review：只算不写库、不自动批")
    parser.add_argument("--node", default=None,
                        help="配合 --ai-review：只出某一站的建议"
                             "（dept_manager/purchase_lead/finance/gm/compliance"
                             "/supplier_onboard）")
    parser.add_argument("--by", default="admin", help="审批人（写入审计链）")
    parser.add_argument("--role", default="gm", help="审批人业务角色（定审批等级）")
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--note", default="", help="备注 / 拒绝理由")
    args = parser.parse_args()

    if not any([args.list, args.show, args.approve, args.reject,
                args.ai_review, args.ai_advice]):
        parser.print_help()
        return

    gate = ApprovalGate(build_default_executors(), verbose=False)

    if args.ai_review:
        row = gate.ai_review(args.ai_review, dry_run=args.dry_run,
                             node_id=args.node)
        out = {k: v for k, v in row.items() if k != "pack"}
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
        return

    if args.ai_advice:
        row = gate.ai_review(args.ai_advice, dry_run=True)
        from ai_approval import format_handover
        print(format_handover(row.get("handover") or {}))
        return

    if args.list:
        rows = gate.list_pending(args.tenant)
        if not rows:
            print("（无待审单）")
            return
        for r in rows:
            amt = (r["payload"] or {}).get("amount")
            req = gate.required_level(r["action_type"], amt)
            print(f"#{r['id']} [{r['action_type']}] 发起: {r['requested_by']}"
                  f"({r.get('requested_role') or '-'})"
                  f"  金额: {amt if amt is not None else '-'}"
                  f"  须 level≥{req}  理由: {r['reason'] or '-'}")
        return
    if args.show:
        print(json.dumps(gate.get(args.show), ensure_ascii=False,
                         indent=2, default=str))
        return
    if args.approve:
        r = gate.decide(args.approve, approve=True, decided_by=args.by,
                        decided_role=args.role, note=args.note)
    else:
        r = gate.decide(args.reject, approve=False, decided_by=args.by,
                        decided_role=args.role, note=args.note)
    print(json.dumps(r, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
