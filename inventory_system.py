"""
================================================================================
 inventory_system.py — 库存域（余额 / 流水 / 入库单 / 出库单）
================================================================================

 铁律（工厂业务系统设计 §3.3）
 ----
   1. 余额是快照，流水才是真相：inventory_transactions 只增不改，
      每行带 balance_after；对账以流水为准。
   2. 涉库存复合操作全走事务（erp_common.execute_txn）：
      校验 + 余额更新 + 流水，要么全成要么全回——
      绝不允许「扣了库存没写流水」的中间态。
   3. 占用语义（防超卖）：confirm 订单 / 领料预留 → reserved_qty += ；
      出库扣 stock 并同步释放 reserved。

 事务类型：PURCHASE_IN 采购入库 / SALE_OUT 销售出库 / RESERVE 占用 /
   RELEASE 释放 / RETURN_IN 退货回补 / PARTS_OUT 维修领料 / ADJUST 盘点调整

 一期简化：占用与出库默认「主仓」（租户第一个启用仓库），保证占用/扣减
 同仓一致；inventory 表已是 仓×产品 粒度，多仓分配（allocation）二期开。

 表结构权威源 config/init_db.sql（表 16-21）。
================================================================================
"""

import time
from typing import Dict, List, Optional

from erp_common import ErpDb

_DDL_INVENTORY = """
CREATE TABLE IF NOT EXISTS `inventory` (
  `id`            BIGINT AUTO_INCREMENT PRIMARY KEY,
  `tenant_id`     VARCHAR(64)   NOT NULL DEFAULT 'default',
  `warehouse_id`  BIGINT        NOT NULL,
  `product_id`    BIGINT        NOT NULL,
  `stock_qty`     INT           NOT NULL DEFAULT 0,
  `reserved_qty`  INT           NOT NULL DEFAULT 0,
  `updated_at`    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY `uk_wh_prod` (`tenant_id`, `warehouse_id`, `product_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

_DDL_TXN = """
CREATE TABLE IF NOT EXISTS `inventory_transactions` (
  `id`           BIGINT AUTO_INCREMENT PRIMARY KEY,
  `tenant_id`    VARCHAR(64)   NOT NULL DEFAULT 'default',
  `txn_type`     VARCHAR(16)   NOT NULL,
  `warehouse_id` BIGINT        NOT NULL,
  `product_id`   BIGINT        NOT NULL,
  `qty`          INT           NOT NULL,
  `balance_after` INT          NOT NULL,
  `ref_type`     VARCHAR(32)   NULL,
  `ref_no`       VARCHAR(32)   NULL,
  `operator`     VARCHAR(64)   NULL,
  `remark`       VARCHAR(256)  NULL,
  `created_at`   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX `idx_tenant_wh_prod` (`tenant_id`, `warehouse_id`, `product_id`, `id` DESC),
  INDEX `idx_ref` (`ref_type`, `ref_no`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

_DDL_RECEIPTS = """
CREATE TABLE IF NOT EXISTS `stock_receipts` (
  `id`           BIGINT AUTO_INCREMENT PRIMARY KEY,
  `tenant_id`    VARCHAR(64)   NOT NULL DEFAULT 'default',
  `receipt_no`   VARCHAR(32)   NOT NULL,
  `ref_type`     VARCHAR(16)   NOT NULL,
  `ref_id`       BIGINT        NULL,
  `warehouse_id` BIGINT        NOT NULL,
  `supplier_id`  BIGINT        NULL,
  `status`       VARCHAR(16)   NOT NULL DEFAULT 'draft',
  `remark`       VARCHAR(256)  NULL,
  `created_by`   VARCHAR(64)   NULL,
  `created_at`   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at`   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `received_at`  DATETIME      NULL,
  INDEX `idx_tenant_status` (`tenant_id`, `status`, `id` DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

_DDL_RECEIPT_ITEMS = """
CREATE TABLE IF NOT EXISTS `stock_receipt_items` (
  `id`          BIGINT AUTO_INCREMENT PRIMARY KEY,
  `tenant_id`   VARCHAR(64)   NOT NULL DEFAULT 'default',
  `receipt_id`  BIGINT        NOT NULL,
  `product_id`  BIGINT        NOT NULL,
  `quantity`    INT           NOT NULL,
  `unit_cost`   DECIMAL(12,2) NULL,
  `remark`      VARCHAR(256)  NULL,
  INDEX `idx_receipt` (`receipt_id`),
  INDEX `idx_product` (`product_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

_DDL_ISSUES = """
CREATE TABLE IF NOT EXISTS `stock_issues` (
  `id`           BIGINT AUTO_INCREMENT PRIMARY KEY,
  `tenant_id`    VARCHAR(64)   NOT NULL DEFAULT 'default',
  `issue_no`     VARCHAR(32)   NOT NULL,
  `ref_type`     VARCHAR(16)   NOT NULL,
  `ref_id`       BIGINT        NULL,
  `warehouse_id` BIGINT        NOT NULL,
  `status`       VARCHAR(16)   NOT NULL DEFAULT 'draft',
  `remark`       VARCHAR(256)  NULL,
  `created_by`   VARCHAR(64)   NULL,
  `created_at`   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at`   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `issued_at`    DATETIME      NULL,
  INDEX `idx_tenant_status` (`tenant_id`, `status`, `id` DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

_DDL_ISSUE_ITEMS = """
CREATE TABLE IF NOT EXISTS `stock_issue_items` (
  `id`         BIGINT AUTO_INCREMENT PRIMARY KEY,
  `tenant_id`  VARCHAR(64)   NOT NULL DEFAULT 'default',
  `issue_id`   BIGINT        NOT NULL,
  `product_id` BIGINT        NOT NULL,
  `quantity`   INT           NOT NULL,
  `remark`     VARCHAR(256)  NULL,
  INDEX `idx_issue` (`issue_id`),
  INDEX `idx_product` (`product_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


class InventoryService(ErpDb):
    """
    库存域唯一入口：余额 / 流水 / 占用释放 / 出入库单据。

    单据动作语义：
      create_receipt → draft（不动物料）；receive_receipt → received
        （事务：逐行 stock_in + 流水 PURCHASE_IN/RETURN_IN）
      create_issue → draft；issue_issue → issued
        （事务：逐行校验 stock_qty 足量 → 扣减 + 流水 SALE_OUT/PARTS_OUT）
    """

    RESOURCE = "inventory"

    def __init__(self, verbose: bool = True):
        # 该 Store 管多张表：DDL 逐条幂等建
        self.TABLE = "inventory"
        self.DDL = _DDL_INVENTORY
        # 内存降级态：余额与流水（MySQL 不可用时保证同进程闭环可测/可演示）
        self._fb_inv: Dict[tuple, Dict] = {}     # (wh_id, prod_id) -> {stock_qty, reserved_qty}
        self._fb_txns: List[Dict] = []
        super().__init__(verbose=verbose)
        if self.available:
            for ddl in (_DDL_TXN, _DDL_RECEIPTS, _DDL_RECEIPT_ITEMS,
                        _DDL_ISSUES, _DDL_ISSUE_ITEMS):
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(ddl)
                    conn.commit()

    # ==================================================================
    # 余额与可用量
    # ==================================================================
    def level(self, tenant_id: str, warehouse_id: int,
              product_id: int) -> Dict:
        """查某仓某产品的库存水位 {stock_qty, reserved_qty, available}。"""
        if self.available:
            rows = self._execute(
                "SELECT stock_qty, reserved_qty FROM inventory "
                "WHERE tenant_id = %s AND warehouse_id = %s AND product_id = %s",
                (tenant_id, warehouse_id, product_id))
            stock = rows[0]["stock_qty"] if rows else 0
            reserved = rows[0]["reserved_qty"] if rows else 0
        else:
            lv = self._fb_inv.get((warehouse_id, product_id),
                                   {"stock_qty": 0, "reserved_qty": 0})
            stock, reserved = lv["stock_qty"], lv["reserved_qty"]
        return {"stock_qty": stock, "reserved_qty": reserved,
                "available": stock - reserved}

    def default_warehouse(self, tenant_id: str = "default",
                          md=None) -> Optional[Dict]:
        """
        主仓 = 租户第一个启用仓库（一期占用/出库默认仓）。
        F9 修复：内部联动走系统通道（此前冒充 "warehouse_user" 身份查询）。
        """
        from master_data import WarehouseStore
        whs = (md.warehouses if md else WarehouseStore(verbose=False)) \
            ._list_for_system(tenant_id=tenant_id)
        return whs[0] if whs else None

    # ==================================================================
    # 占用 / 释放（销售确认、取消；预留不动 stock）
    # ==================================================================
    def reserve(self, tenant_id: str, warehouse_id: int, product_id: int,
                qty: int, ref_type: str, ref_no: str,
                operator: str = "system") -> Dict:
        """
        占用（防超卖）：available >= qty 才放行，reserved += qty，写 RESERVE 流水。
        与校验同事务——并发下不会双重占用到超卖。
        """
        if qty <= 0:
            raise ValueError("占用数量必须为正")
        lv = self.level(tenant_id, warehouse_id, product_id)
        if lv["available"] < qty:
            raise ValueError(
                f"可用库存不足：产品 {product_id} 可用 {lv['available']}，"
                f"需占用 {qty}（防超卖保护，订单被拒绝）")
        new_reserved = lv["reserved_qty"] + qty
        if self.available:
            stmts = [
                ("INSERT INTO inventory (tenant_id, warehouse_id, product_id, "
                 "stock_qty, reserved_qty) VALUES (%s,%s,%s,%s,%s) "
                 "ON DUPLICATE KEY UPDATE reserved_qty = %s",
                 (tenant_id, warehouse_id, product_id, lv["stock_qty"],
                  new_reserved, new_reserved)),
                ("INSERT INTO inventory_transactions (tenant_id, txn_type, "
                 "warehouse_id, product_id, qty, balance_after, ref_type, "
                 "ref_no, operator) VALUES (%s,'RESERVE',%s,%s,%s,%s,%s,%s,%s)",
                 (tenant_id, warehouse_id, product_id, qty, lv["stock_qty"],
                  ref_type, ref_no, operator)),
            ]
            self._execute_txn(stmts)
        else:
            self._fb_inv[(warehouse_id, product_id)] = {
                "stock_qty": lv["stock_qty"], "reserved_qty": new_reserved}
            self._fb_log("RESERVE", warehouse_id, product_id, qty,
                         lv["stock_qty"], ref_type, ref_no, operator)
        return self.level(tenant_id, warehouse_id, product_id)

    def release(self, tenant_id: str, warehouse_id: int, product_id: int,
                qty: int, ref_type: str, ref_no: str,
                operator: str = "system") -> Dict:
        """释放占用：reserved -= qty（下限 0），写 RELEASE 流水。"""
        if qty <= 0:
            raise ValueError("释放数量必须为正")
        lv = self.level(tenant_id, warehouse_id, product_id)
        new_reserved = max(0, lv["reserved_qty"] - qty)
        if self.available:
            self._execute_txn([
                ("UPDATE inventory SET reserved_qty = %s WHERE tenant_id = %s "
                 "AND warehouse_id = %s AND product_id = %s",
                 (new_reserved, tenant_id, warehouse_id, product_id)),
                ("INSERT INTO inventory (tenant_id, warehouse_id, product_id, "
                 "stock_qty, reserved_qty) VALUES (%s,%s,%s,%s,%s) "
                 "ON DUPLICATE KEY UPDATE reserved_qty = %s",
                 (tenant_id, warehouse_id, product_id, lv["stock_qty"],
                  new_reserved, new_reserved)),
                ("INSERT INTO inventory_transactions (tenant_id, txn_type, "
                 "warehouse_id, product_id, qty, balance_after, ref_type, "
                 "ref_no, operator) VALUES (%s,'RELEASE',%s,%s,%s,%s,%s,%s,%s)",
                 (tenant_id, warehouse_id, product_id, -qty, lv["stock_qty"],
                  ref_type, ref_no, operator)),
            ])
        else:
            self._fb_inv[(warehouse_id, product_id)] = {
                "stock_qty": lv["stock_qty"], "reserved_qty": new_reserved}
            self._fb_log("RELEASE", warehouse_id, product_id, -qty,
                         lv["stock_qty"], ref_type, ref_no, operator)
        return self.level(tenant_id, warehouse_id, product_id)

    def _fb_log(self, txn_type, warehouse_id, product_id, qty, balance_after,
                ref_type, ref_no, operator, remark=None):
        """内存模式流水（与 MySQL 流水字段同构）。"""
        import time as _t
        self._fb_txns.append({
            "txn_type": txn_type, "warehouse_id": warehouse_id,
            "product_id": product_id, "qty": qty,
            "balance_after": balance_after, "ref_type": ref_type,
            "ref_no": ref_no, "operator": operator, "remark": remark,
            "created_at": _t.strftime("%Y-%m-%d %H:%M:%S")})

    # ==================================================================
    # 出入库核心（动 stock，全部事务）
    # ==================================================================
    def _stock_in(self, tenant_id: str, warehouse_id: int, product_id: int,
                  qty: int, txn_type: str, ref_type: str, ref_no: str,
                  operator: str, remark: str = None) -> int:
        """入库：stock += qty + 流水（返回变动后 stock）。须在调用方事务内。"""
        lv = self.level(tenant_id, warehouse_id, product_id)
        new_stock = lv["stock_qty"] + qty
        if self.available:
            self._execute_txn([
                ("INSERT INTO inventory (tenant_id, warehouse_id, product_id, "
                 "stock_qty, reserved_qty) VALUES (%s,%s,%s,%s,%s) "
                 "ON DUPLICATE KEY UPDATE stock_qty = %s",
                 (tenant_id, warehouse_id, product_id, new_stock,
                  lv["reserved_qty"], new_stock)),
                ("INSERT INTO inventory_transactions (tenant_id, txn_type, "
                 "warehouse_id, product_id, qty, balance_after, ref_type, "
                 "ref_no, operator, remark) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                 (tenant_id, txn_type, warehouse_id, product_id, qty,
                  new_stock, ref_type, ref_no, operator, remark)),
            ])
        else:
            self._fb_inv[(warehouse_id, product_id)] = {
                "stock_qty": new_stock, "reserved_qty": lv["reserved_qty"]}
            self._fb_log(txn_type, warehouse_id, product_id, qty, new_stock,
                         ref_type, ref_no, operator, remark)
        return new_stock

    def _stock_out(self, tenant_id: str, warehouse_id: int, product_id: int,
                   qty: int, txn_type: str, ref_type: str, ref_no: str,
                   operator: str, also_release: int = 0) -> int:
        """
        出库：校验 stock 足量 → stock -= qty（+ 可选同步释放占用）+ 流水。
        also_release > 0 时 reserved -= also_release（销售出库：占用转出库）。
        """
        if qty <= 0:
            raise ValueError("出库数量必须为正")
        lv = self.level(tenant_id, warehouse_id, product_id)
        if lv["stock_qty"] < qty:
            raise ValueError(
                f"实物库存不足：产品 {product_id} 在库 {lv['stock_qty']}，"
                f"需出库 {qty}")
        new_stock = lv["stock_qty"] - qty
        new_reserved = max(0, lv["reserved_qty"] - also_release)
        if self.available:
            self._execute_txn([
                ("UPDATE inventory SET stock_qty = %s, reserved_qty = %s "
                 "WHERE tenant_id = %s AND warehouse_id = %s AND product_id = %s",
                 (new_stock, new_reserved, tenant_id, warehouse_id, product_id)),
                ("INSERT INTO inventory (tenant_id, warehouse_id, product_id, "
                 "stock_qty, reserved_qty) VALUES (%s,%s,%s,%s,%s) "
                 "ON DUPLICATE KEY UPDATE stock_qty = %s, reserved_qty = %s",
                 (tenant_id, warehouse_id, product_id, new_stock, new_reserved,
                  new_stock, new_reserved)),
                ("INSERT INTO inventory_transactions (tenant_id, txn_type, "
                 "warehouse_id, product_id, qty, balance_after, ref_type, "
                 "ref_no, operator) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                 (tenant_id, txn_type, warehouse_id, product_id, -qty,
                  new_stock, ref_type, ref_no, operator)),
            ])
        else:
            self._fb_inv[(warehouse_id, product_id)] = {
                "stock_qty": new_stock, "reserved_qty": new_reserved}
            self._fb_log(txn_type, warehouse_id, product_id, -qty, new_stock,
                         ref_type, ref_no, operator)
        return new_stock

    def adjust(self, tenant_id: str, warehouse_id: int, product_id: int,
               delta: int, ref_no: str, operator: str,
               remark: str = "") -> Dict:
        """盘点调整（高危，须审批后调用）：delta 正负皆可，写 ADJUST 流水。"""
        if delta == 0:
            raise ValueError("调整量不能为 0")
        if delta > 0:
            self._stock_in(tenant_id, warehouse_id, product_id, delta,
                           "ADJUST", "adjust", ref_no, operator, remark)
        else:
            self._stock_out(tenant_id, warehouse_id, product_id, -delta,
                            "ADJUST", "adjust", ref_no, operator)
        return self.level(tenant_id, warehouse_id, product_id)

    # ==================================================================
    # 入库单（采购收货 / 退货入库）
    # ==================================================================
    def create_receipt(self, acting_role: str, acting_user: str,
                       tenant_id: str, warehouse_id: int,
                       items: List[Dict], ref_type: str = "purchase",
                       ref_id: int = None, supplier_id: int = None,
                       remark: str = "") -> Dict:
        """
        创建入库单（draft，不动物料）。
        items: [{product_id, quantity, unit_cost?, remark?}]
        """
        self._check_op("update", acting_role, acting_user, tenant_id,
                       resource="stock")
        if not items:
            raise ValueError("入库明细不能为空")
        for it in items:
            if int(it.get("quantity", 0)) <= 0:
                raise ValueError(f"入库数量必须为正：{it}")
        if self.available:
            rid = self._execute(
                "INSERT INTO stock_receipts (tenant_id, receipt_no, ref_type, "
                "ref_id, warehouse_id, supplier_id, status, remark, created_by) "
                "VALUES (%s,'PENDING',%s,%s,%s,%s,'draft',%s,%s)",
                (tenant_id, ref_type, ref_id, warehouse_id, supplier_id,
                 remark, acting_user))
            no = self._next_no("RC", rid)
            self._execute(
                "UPDATE stock_receipts SET receipt_no = %s WHERE id = %s",
                (no, rid))
            for it in items:
                self._execute(
                    "INSERT INTO stock_receipt_items (tenant_id, receipt_id, "
                    "product_id, quantity, unit_cost, remark) "
                    "VALUES (%s,%s,%s,%s,%s,%s)",
                    (tenant_id, rid, it["product_id"], it["quantity"],
                     it.get("unit_cost"), it.get("remark")))
            return self.get_receipt(rid, tenant_id)
        self._fallback_seq += 1
        rid = self._fallback_seq
        row = {"id": rid, "receipt_no": self._next_no("RC", rid),
               "tenant_id": tenant_id, "ref_type": ref_type, "ref_id": ref_id,
               "warehouse_id": warehouse_id, "supplier_id": supplier_id,
               "status": "draft", "remark": remark, "created_by": acting_user,
               "items": [dict(i) for i in items]}
        self._fallback_rows.append(row)
        return dict(row)

    def receive_receipt(self, receipt_id: int, tenant_id: str,
                        operator: str = "system") -> Dict:
        """
        收货入库：draft → received，事务性逐行加库存 + 写流水。
        采购入库 txn=PURCHASE_IN；退货入库 txn=RETURN_IN。
        """
        rc = self.get_receipt(receipt_id, tenant_id)
        if not rc:
            raise ValueError(f"入库单 #{receipt_id} 不存在")
        if rc["status"] != "draft":
            raise ValueError(f"入库单 {rc['receipt_no']} 状态 {rc['status']}，不可重复收货")
        txn_type = "RETURN_IN" if rc["ref_type"] == "return" else "PURCHASE_IN"
        for it in self._receipt_items(receipt_id, tenant_id):
            self._stock_in(tenant_id, rc["warehouse_id"], it["product_id"],
                           it["quantity"], txn_type, "stock_receipt",
                           rc["receipt_no"], operator)
        if self.available:
            self._execute(
                "UPDATE stock_receipts SET status = 'received', "
                "received_at = NOW() WHERE id = %s AND tenant_id = %s",
                (receipt_id, tenant_id))
        else:
            rc["status"] = "received"
        return self.get_receipt(receipt_id, tenant_id)

    def get_receipt(self, rid: int, tenant_id: str) -> Optional[Dict]:
        if self.available:
            rows = self._execute(
                "SELECT * FROM stock_receipts WHERE id = %s AND tenant_id = %s",
                (rid, tenant_id))
            if not rows:
                return None
            r = rows[0]
            r["items"] = self._receipt_items(rid, tenant_id)
            return r
        for r in self._fallback_rows:
            if r.get("id") == rid and r["tenant_id"] == tenant_id \
                    and "receipt_no" in r:
                return r   # 原引用：receive_receipt 的内存态状态更新直接生效
        return None

    def _receipt_items(self, rid: int, tenant_id: str) -> List[Dict]:
        if self.available:
            return self._execute(
                "SELECT * FROM stock_receipt_items WHERE receipt_id = %s "
                "AND tenant_id = %s ORDER BY id", (rid, tenant_id)) or []
        # F5 修复：内存模式从单据行取明细（create_receipt 已随行存 items），
        # 否则 receive_receipt 的逐行 _stock_in 循环空转——收货不回写库存
        row = next((r for r in self._fallback_rows
                    if r.get("id") == rid and r.get("tenant_id") == tenant_id
                    and "receipt_no" in r), None)
        return (row or {}).get("items", [])

    # ==================================================================
    # 出库单（销售出库 / 维修领料）
    # ==================================================================
    def create_issue(self, acting_role: str, acting_user: str,
                     tenant_id: str, warehouse_id: int,
                     items: List[Dict], ref_type: str = "sales_delivery",
                     ref_id: int = None, remark: str = "") -> Dict:
        """创建出库单（draft，不动物料）。"""
        self._check_op("update", acting_role, acting_user, tenant_id,
                       resource="stock")
        if not items:
            raise ValueError("出库明细不能为空")
        for it in items:
            if int(it.get("quantity", 0)) <= 0:
                raise ValueError(f"出库数量必须为正：{it}")
        if self.available:
            iid = self._execute(
                "INSERT INTO stock_issues (tenant_id, issue_no, ref_type, "
                "ref_id, warehouse_id, status, remark, created_by) "
                "VALUES (%s,'PENDING',%s,%s,%s,'draft',%s,%s)",
                (tenant_id, ref_type, ref_id, warehouse_id, remark, acting_user))
            no = self._next_no("IS", iid)
            self._execute("UPDATE stock_issues SET issue_no = %s WHERE id = %s",
                          (no, iid))
            for it in items:
                self._execute(
                    "INSERT INTO stock_issue_items (tenant_id, issue_id, "
                    "product_id, quantity, remark) VALUES (%s,%s,%s,%s,%s)",
                    (tenant_id, iid, it["product_id"], it["quantity"],
                     it.get("remark")))
            return self.get_issue(iid, tenant_id)
        self._fallback_seq += 1
        iid = self._fallback_seq
        row = {"id": iid, "issue_no": self._next_no("IS", iid),
               "tenant_id": tenant_id, "ref_type": ref_type, "ref_id": ref_id,
               "warehouse_id": warehouse_id, "status": "draft",
               "remark": remark, "created_by": acting_user,
               "items": [dict(i) for i in items]}
        self._fallback_rows.append(row)
        return dict(row)

    def issue_issue(self, issue_id: int, tenant_id: str,
                    operator: str = "system") -> Dict:
        """
        执行出库：draft → issued，事务性逐行校验足量 → 扣库存 + 写流水。
        销售出库 txn=SALE_OUT；维修领料 txn=PARTS_OUT。
        """
        iss = self.get_issue(issue_id, tenant_id)
        if not iss:
            raise ValueError(f"出库单 #{issue_id} 不存在")
        if iss["status"] != "draft":
            raise ValueError(f"出库单 {iss['issue_no']} 状态 {iss['status']}，不可重复出库")
        txn_type = "PARTS_OUT" if iss["ref_type"] == "repair" else "SALE_OUT"
        for it in self._issue_items(issue_id, tenant_id):
            self._stock_out(tenant_id, iss["warehouse_id"], it["product_id"],
                            it["quantity"], txn_type, "stock_issue",
                            iss["issue_no"], operator)
        if self.available:
            self._execute(
                "UPDATE stock_issues SET status = 'issued', issued_at = NOW() "
                "WHERE id = %s AND tenant_id = %s", (issue_id, tenant_id))
        else:
            iss["status"] = "issued"
        return self.get_issue(issue_id, tenant_id)

    def get_issue(self, iid: int, tenant_id: str) -> Optional[Dict]:
        if self.available:
            rows = self._execute(
                "SELECT * FROM stock_issues WHERE id = %s AND tenant_id = %s",
                (iid, tenant_id))
            if not rows:
                return None
            r = rows[0]
            r["items"] = self._issue_items(iid, tenant_id)
            return r
        for r in self._fallback_rows:
            if r.get("id") == iid and r["tenant_id"] == tenant_id \
                    and "issue_no" in r:
                return r   # 原引用：issue_issue 的内存态状态更新直接生效
        return None

    def _issue_items(self, iid: int, tenant_id: str) -> List[Dict]:
        if self.available:
            return self._execute(
                "SELECT * FROM stock_issue_items WHERE issue_id = %s "
                "AND tenant_id = %s ORDER BY id", (iid, tenant_id)) or []
        # F5 修复：内存模式从单据行取明细（issue_issue 扣库存依赖它）
        row = next((r for r in self._fallback_rows
                    if r.get("id") == iid and r.get("tenant_id") == tenant_id
                    and "issue_no" in r), None)
        return (row or {}).get("items", [])

    # ==================================================================
    # 查询（L2 行级裁剪 + 低库存预警）
    # ==================================================================
    def query_inventory(self, acting_role: str, acting_user: str,
                        tenant_id: str = "default",
                        low_only: bool = False,
                        md=None) -> List[Dict]:
        """库存列表（联产品名/编码/安全库存）。low_only=True 只看低于安全线的。"""
        self._check_op("query", acting_role, acting_user, tenant_id)
        if self.available:
            sql = ("SELECT i.warehouse_id, i.product_id, i.stock_qty, "
                   "i.reserved_qty, p.product_code, p.name, p.safety_stock "
                   "FROM inventory i LEFT JOIN products p "
                   "ON i.product_id = p.id AND i.tenant_id = p.tenant_id "
                   "WHERE i.tenant_id = %s")
            rows = self._execute(sql, (tenant_id,)) or []
        else:
            rows = []
        out = []
        for r in rows:
            available = r["stock_qty"] - r["reserved_qty"]
            rec = {**r, "available": available,
                   "low": available < (r.get("safety_stock") or 0)}
            if low_only and not rec["low"]:
                continue
            out.append(rec)
        return out

    def transactions_of(self, acting_role: str, acting_user: str,
                        tenant_id: str = "default",
                        product_id: int = None, limit: int = 50) -> List[Dict]:
        """库存流水（审计视图：谁在什么时候动了什么，余额轨迹）。"""
        self._check_op("query", acting_role, acting_user, tenant_id)
        if not self.available:
            rows = [t for t in self._fb_txns
                    if product_id is None or t["product_id"] == product_id]
            return list(reversed(rows))[:limit]
        sql = ("SELECT t.*, p.product_code FROM inventory_transactions t "
               "LEFT JOIN products p ON t.product_id = p.id "
               "AND t.tenant_id = p.tenant_id WHERE t.tenant_id = %s")
        params: list = [tenant_id]
        if product_id:
            sql += " AND t.product_id = %s"
            params.append(product_id)
        sql += " ORDER BY t.id DESC LIMIT %s"
        params.append(limit)
        return self._execute(sql, tuple(params)) or []
