"""
================================================================================
 purchase_system.py — 采购域（采购单 → 审批 → 收货入库 → 回写）
================================================================================

 流程（工厂业务系统设计 §3.4）
 ----
   create_po（draft，价格按 products.cost_price 快照）
     → submit_po（submitted，金额固化；触发审批）
     → execute_approve（approved，审批 executor；金额>10万须 level=3，已定稿）
     → receive_po（received/partial：生成入库单并收货——库存 PURCHASE_IN
       流水 + received_qty 回写；部分收货一期按全量收简化）

 与库存联动经 InventoryService（入库事务 + 流水）。
 表结构权威源 config/init_db.sql（表 28-29）。
================================================================================
"""

from typing import Dict, List, Optional

from erp_common import ErpDb

_DDL_PO = """
CREATE TABLE IF NOT EXISTS `purchase_orders` (
  `id`            BIGINT AUTO_INCREMENT PRIMARY KEY,
  `tenant_id`     VARCHAR(64)   NOT NULL DEFAULT 'default',
  `po_no`         VARCHAR(32)   NOT NULL,
  `supplier_id`   BIGINT        NOT NULL,
  `amount`        DECIMAL(14,2) NOT NULL DEFAULT 0,
  `status`        VARCHAR(16)   NOT NULL DEFAULT 'draft',
  `buyer`         VARCHAR(64)   NULL,
  `expected_date` DATE          NULL,
  `created_by`    VARCHAR(64)   NULL,
  `created_at`    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at`    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX `idx_tenant_status` (`tenant_id`, `status`, `id` DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

_DDL_PO_ITEMS = """
CREATE TABLE IF NOT EXISTS `purchase_order_items` (
  `id`          BIGINT AUTO_INCREMENT PRIMARY KEY,
  `tenant_id`   VARCHAR(64)   NOT NULL DEFAULT 'default',
  `po_id`       BIGINT        NOT NULL,
  `product_id`  BIGINT        NOT NULL,
  `quantity`    INT           NOT NULL,
  `unit_price`  DECIMAL(12,2) NOT NULL,
  `received_qty` INT          NOT NULL DEFAULT 0,
  INDEX `idx_po` (`po_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


class PurchaseService(ErpDb):
    """采购域唯一入口。"""

    RESOURCE = "purchase_order"

    def __init__(self, inventory=None, master=None, verbose: bool = True):
        self.TABLE, self.DDL = "purchase_orders", _DDL_PO
        super().__init__(verbose=verbose)
        if self.available:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(_DDL_PO_ITEMS)
                conn.commit()
        from inventory_system import InventoryService
        from master_data import MasterData
        self.inv = inventory or InventoryService(verbose=False)
        self.md = master or MasterData(verbose=False)

    # ==================================================================
    # 建单 / 提交
    # ==================================================================
    def create_po(self, acting_role: str, acting_user: str, tenant_id: str,
                  supplier_code: str, items: List[Dict],
                  expected_date: str = None) -> Dict:
        """
        创建采购单（draft）。items: [{product_code, quantity}]。
        采购价按 products.cost_price 快照（缺省 0——须先补主数据价格）。
        """
        self._check_op("create", acting_role, acting_user, tenant_id)
        sup = self.md.suppliers.get_by_code(supplier_code, tenant_id)
        if not sup:
            raise ValueError(f"供应商「{supplier_code}」不存在（先建主数据）")
        if not items:
            raise ValueError("采购明细不能为空")
        rows = []
        for it in items:
            prod = self.md.products.get_by_code(it.get("product_code", ""),
                                                tenant_id)
            if not prod:
                raise ValueError(f"产品「{it.get('product_code')}」不存在")
            qty = int(it.get("quantity", 0))
            if qty <= 0:
                raise ValueError(f"采购数量必须为正：{it}")
            rows.append((prod, qty, float(prod.get("cost_price") or 0)))
        amount = round(sum(q * p for _, q, p in rows), 2)
        if amount <= 0:
            raise ValueError("采购金额为 0——请先在产品主数据补 cost_price")

        if self.available:
            pid = self._execute(
                "INSERT INTO purchase_orders (tenant_id, po_no, supplier_id, "
                "amount, status, buyer, expected_date, created_by) "
                "VALUES (%s,'PENDING',%s,%s,'draft',%s,%s,%s)",
                (tenant_id, sup["id"], amount, acting_user, expected_date,
                 acting_user))
            no = self._next_no("PO", pid)
            self._execute("UPDATE purchase_orders SET po_no = %s WHERE id = %s",
                          (no, pid))
            for prod, qty, price in rows:
                self._execute(
                    "INSERT INTO purchase_order_items (tenant_id, po_id, "
                    "product_id, quantity, unit_price) VALUES (%s,%s,%s,%s,%s)",
                    (tenant_id, pid, prod["id"], qty, price))
            return self.get_po(pid, tenant_id)
        self._fallback_seq += 1
        pid = self._fallback_seq
        po = {"id": pid, "po_no": self._next_no("PO", pid),
              "tenant_id": tenant_id, "supplier_id": sup["id"],
              "amount": amount, "status": "draft", "buyer": acting_user,
              "created_by": acting_user, "items": [
                  {"product_id": p["id"], "product_code": p["product_code"],
                   "quantity": q, "unit_price": pr, "received_qty": 0}
                  for p, q, pr in rows]}
        self._fallback_rows.append(po)
        return dict(po)

    def submit_po(self, acting_role: str, acting_user: str, po_id: int,
                  tenant_id: str = "default") -> Dict:
        """提交审批：draft → submitted（审批在 MCP/编排层发起）。"""
        self._check_op("update", acting_role, acting_user, tenant_id)
        po = self.get_po(po_id, tenant_id)
        if not po:
            raise ValueError(f"采购单 #{po_id} 不存在")
        if po["status"] != "draft":
            raise ValueError(f"采购单 {po['po_no']} 状态 {po['status']}，仅 draft 可提交")
        return self._set_status(po_id, tenant_id, "submitted")

    def execute_approve(self, po_id: int, tenant_id: str,
                        approver: str) -> Dict:
        """审批通过（executor 调用）：submitted → approved。"""
        po = self.get_po(po_id, tenant_id)
        if not po:
            raise ValueError(f"采购单 #{po_id} 不存在")
        if po["status"] != "submitted":
            raise ValueError(f"采购单 {po['po_no']} 状态 {po['status']}，须 submitted 才能审批")
        return self._set_status(po_id, tenant_id, "approved")

    # ==================================================================
    # 收货入库（库存联动 + received_qty 回写）
    # ==================================================================
    def receive_po(self, acting_role: str, acting_user: str, po_id: int,
                   tenant_id: str = "default") -> Dict:
        """
        收货：approved → received（一期整单收）。
        生成入库单（ref_type=purchase）并立即收货：
        事务加库存（PURCHASE_IN 流水）+ 明细 received_qty 回写。
        """
        self._check_op("update", acting_role, acting_user, tenant_id)
        po = self.get_po(po_id, tenant_id)
        if not po:
            raise ValueError(f"采购单 #{po_id} 不存在")
        if po["status"] != "approved":
            raise ValueError(
                f"采购单 {po['po_no']} 状态 {po['status']}，须审批通过后才能收货")
        wh = self.inv.default_warehouse(tenant_id, md=self.md)
        if not wh:
            raise ValueError("租户无可用仓库")
        items = self._po_items(po_id, tenant_id)
        pending = [it for it in items if it["received_qty"] < it["quantity"]]
        if not pending:
            raise ValueError("采购单已全部入库")
        rc = self.inv.create_receipt(
            "warehouse_user", acting_user, tenant_id, wh["id"],
            [{"product_id": it["product_id"],
              "quantity": it["quantity"] - it["received_qty"],
              "unit_cost": it["unit_price"]} for it in pending],
            ref_type="purchase", ref_id=po_id,
            supplier_id=po.get("supplier_id"),
            remark=f"采购收货 {po['po_no']}")
        self.inv.receive_receipt(rc["id"], tenant_id, operator=acting_user)
        for it in pending:
            if self.available:
                self._execute(
                    "UPDATE purchase_order_items SET received_qty = quantity "
                    "WHERE id = %s AND tenant_id = %s",
                    (it["id"], tenant_id))
        return self._set_status(po_id, tenant_id, "received")

    # ==================================================================
    # 查询
    # ==================================================================
    def query_pos(self, acting_role: str, acting_user: str,
                  tenant_id: str = "default", status: str = None,
                  limit: int = 20) -> List[Dict]:
        rule = self._check_op("query", acting_role, acting_user, tenant_id)
        if self.available:
            sql = ("SELECT o.*, s.name AS supplier_name, s.supplier_code "
                   "FROM purchase_orders o LEFT JOIN suppliers s "
                   "ON o.supplier_id = s.id AND o.tenant_id = s.tenant_id "
                   "WHERE o.tenant_id = %s")
            params: list = [tenant_id]
            if rule.get("scope") == "own":
                sql += " AND o.created_by = %s"
                params.append(acting_user)
            if status:
                sql += " AND o.status = %s"
                params.append(status)
            sql += " ORDER BY o.id DESC LIMIT %s"
            params.append(limit)
            return self._execute(sql, tuple(params)) or []
        scope = rule.get("scope", "own")
        rows = [dict(r) for r in self._fallback_rows
                if "po_no" in r and r["tenant_id"] == tenant_id
                and (scope != "own" or r.get("created_by") == acting_user)
                and (not status or r["status"] == status)]
        return list(reversed(rows))[:limit]

    def get_po(self, po_id: int, tenant_id: str) -> Optional[Dict]:
        if self.available:
            rows = self.query_pos("gm", "system", tenant_id=tenant_id,
                                  limit=10**6)
            po = next((r for r in rows if r["id"] == po_id), None)
            if po:
                po["items"] = self._po_items(po_id, tenant_id)
            return po
        for r in self._fallback_rows:
            if r.get("id") == po_id and r["tenant_id"] == tenant_id \
                    and "po_no" in r:
                return r   # 原引用：execute_approve 等的内存态状态更新直接生效
        return None

    # ==================================================================
    # 内部
    # ==================================================================
    def _po_items(self, po_id: int, tenant_id: str) -> List[Dict]:
        if self.available:
            return self._execute(
                "SELECT i.*, p.product_code, p.name AS product_name "
                "FROM purchase_order_items i LEFT JOIN products p "
                "ON i.product_id = p.id AND i.tenant_id = p.tenant_id "
                "WHERE i.po_id = %s AND i.tenant_id = %s ORDER BY i.id",
                (po_id, tenant_id)) or []
        po = next((r for r in self._fallback_rows
                   if r.get("id") == po_id and "po_no" in r), None)
        return (po or {}).get("items", [])

    def _set_status(self, po_id, tenant_id, status) -> Dict:
        if self.available:
            self._execute("UPDATE purchase_orders SET status = %s "
                          "WHERE id = %s AND tenant_id = %s",
                          (status, po_id, tenant_id))
        else:
            po = self.get_po(po_id, tenant_id)   # 原引用，直接改内存行
            if po:
                po["status"] = status
        return self.get_po(po_id, tenant_id)
