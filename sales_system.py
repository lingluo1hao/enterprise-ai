"""
================================================================================
 sales_system.py — 销售域（订单→出库→物流→退货，单据分离）
================================================================================

 单据分离（ERP 通则，设计 §3.2）
 ----
   sales_orders        计划单（客户意图）——confirm 时占用库存（防超卖），
                       但不动实物
   sales_deliveries    执行单（仓库动作）——release 时才真正扣库存；
                       支持一单多次出库（delivered_qty 回写）
   shipments           物流交接；sales_returns 退货（质检 ok 回库 / scrap 报废）

 高危节点（走审批门 approval.py，金额分级 >10 万须 level=3）：
   sales_delivery.release（扣库存）/ sales_order.complete（关单）/
   sales_order.cancel（已确认订单取消须释放占用）
 表结构权威源 config/init_db.sql（表 22-27）。
================================================================================
"""

from typing import Dict, List, Optional

from erp_common import ErpDb, PermissionDenied

_DDL_ORDERS = """
CREATE TABLE IF NOT EXISTS `sales_orders` (
  `id`             BIGINT AUTO_INCREMENT PRIMARY KEY,
  `tenant_id`      VARCHAR(64)   NOT NULL DEFAULT 'default',
  `order_no`       VARCHAR(32)   NOT NULL,
  `customer_id`    BIGINT        NOT NULL,
  `amount`         DECIMAL(14,2) NOT NULL DEFAULT 0,
  `status`         VARCHAR(16)   NOT NULL DEFAULT 'draft',
  `sales_rep`      VARCHAR(64)   NULL,
  `expected_date`  DATE          NULL,
  `remark`         VARCHAR(256)  NULL,
  `created_by`     VARCHAR(64)   NULL,
  `created_at`     DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at`     DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `completed_at`   DATETIME      NULL,
  INDEX `idx_tenant_status` (`tenant_id`, `status`, `id` DESC),
  INDEX `idx_order_no` (`order_no`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

_DDL_ORDER_ITEMS = """
CREATE TABLE IF NOT EXISTS `sales_order_items` (
  `id`            BIGINT AUTO_INCREMENT PRIMARY KEY,
  `tenant_id`     VARCHAR(64)   NOT NULL DEFAULT 'default',
  `order_id`      BIGINT        NOT NULL,
  `product_id`    BIGINT        NOT NULL,
  `quantity`      INT           NOT NULL,
  `unit_price`    DECIMAL(12,2) NOT NULL,
  `line_amount`   DECIMAL(14,2) NOT NULL,
  `delivered_qty` INT           NOT NULL DEFAULT 0,
  INDEX `idx_order` (`order_id`),
  INDEX `idx_product` (`product_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

_DDL_DELIVERIES = """
CREATE TABLE IF NOT EXISTS `sales_deliveries` (
  `id`            BIGINT AUTO_INCREMENT PRIMARY KEY,
  `tenant_id`     VARCHAR(64)   NOT NULL DEFAULT 'default',
  `delivery_no`   VARCHAR(32)   NOT NULL,
  `order_id`      BIGINT        NOT NULL,
  `warehouse_id`  BIGINT        NOT NULL,
  `status`        VARCHAR(16)   NOT NULL DEFAULT 'draft',
  `released_by`   VARCHAR(64)   NULL,
  `remark`        VARCHAR(256)  NULL,
  `created_by`    VARCHAR(64)   NULL,
  `created_at`    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at`    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX `idx_tenant_status` (`tenant_id`, `status`, `id` DESC),
  INDEX `idx_order` (`order_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

_DDL_DELIVERY_ITEMS = """
CREATE TABLE IF NOT EXISTS `sales_delivery_items` (
  `id`             BIGINT AUTO_INCREMENT PRIMARY KEY,
  `tenant_id`      VARCHAR(64)   NOT NULL DEFAULT 'default',
  `delivery_id`    BIGINT        NOT NULL,
  `order_item_id`  BIGINT        NOT NULL,
  `product_id`     BIGINT        NOT NULL,
  `quantity`       INT           NOT NULL,
  `snapshot_price` DECIMAL(12,2) NULL,
  INDEX `idx_delivery` (`delivery_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

_DDL_SHIPMENTS = """
CREATE TABLE IF NOT EXISTS `shipments` (
  `id`           BIGINT AUTO_INCREMENT PRIMARY KEY,
  `tenant_id`    VARCHAR(64)   NOT NULL DEFAULT 'default',
  `shipment_no`  VARCHAR(32)   NOT NULL,
  `delivery_id`  BIGINT        NOT NULL,
  `carrier`      VARCHAR(64)   NOT NULL,
  `tracking_no`  VARCHAR(64)   NULL,
  `status`       VARCHAR(16)   NOT NULL DEFAULT 'created',
  `shipped_at`   DATETIME      NULL,
  `delivered_at` DATETIME      NULL,
  `created_by`   VARCHAR(64)   NULL,
  `created_at`   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX `idx_tenant_status` (`tenant_id`, `status`),
  INDEX `idx_delivery` (`delivery_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

_DDL_RETURNS = """
CREATE TABLE IF NOT EXISTS `sales_returns` (
  `id`            BIGINT AUTO_INCREMENT PRIMARY KEY,
  `tenant_id`     VARCHAR(64)   NOT NULL DEFAULT 'default',
  `return_no`     VARCHAR(32)   NOT NULL,
  `order_id`      BIGINT        NOT NULL,
  `product_id`    BIGINT        NOT NULL,
  `quantity`      INT           NOT NULL,
  `reason`        VARCHAR(256)  NULL,
  `status`        VARCHAR(16)   NOT NULL DEFAULT 'draft',
  `inspect_result` VARCHAR(8)   NULL,
  `warehouse_id`  BIGINT        NULL,
  `created_by`    VARCHAR(64)   NULL,
  `created_at`    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `received_at`   DATETIME      NULL,
  INDEX `idx_tenant_status` (`tenant_id`, `status`),
  INDEX `idx_order` (`order_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


class SalesService(ErpDb):
    """销售域唯一入口。库存联动经 InventoryService（构造注入，避免重复连接）。"""

    RESOURCE = "sales_order"

    def __init__(self, inventory=None, master=None, verbose: bool = True):
        self.TABLE, self.DDL = "sales_orders", _DDL_ORDERS
        super().__init__(verbose=verbose)
        if self.available:
            for ddl in (_DDL_ORDER_ITEMS, _DDL_DELIVERIES, _DDL_DELIVERY_ITEMS,
                        _DDL_SHIPMENTS, _DDL_RETURNS):
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(ddl)
                    conn.commit()
        from inventory_system import InventoryService
        from master_data import MasterData
        self.inv = inventory or InventoryService(verbose=False)
        self.md = master or MasterData(verbose=False)

    # ==================================================================
    # 下单（计划单：价格快照，不动物料）
    # ==================================================================
    def create_order(self, acting_role: str, acting_user: str,
                     tenant_id: str, customer_code: str,
                     items: List[Dict], expected_date: str = None,
                     remark: str = "") -> Dict:
        """
        创建销售订单（draft）。items: [{product_code, quantity}]。
        单价从 products 快照——此后改价不影响已下明细。
        """
        self._check_op("create", acting_role, acting_user, tenant_id)
        cust = self.md.customers.get_by_code(customer_code, tenant_id)
        if not cust:
            raise ValueError(f"客户「{customer_code}」不存在（先建主数据）")
        if not items:
            raise ValueError("订单明细不能为空")
        rows = []
        for it in items:
            prod = self.md.products.get_by_code(it.get("product_code", ""),
                                                tenant_id)
            if not prod:
                raise ValueError(f"产品「{it.get('product_code')}」不存在")
            qty = int(it.get("quantity", 0))
            if qty <= 0:
                raise ValueError(f"数量必须为正：{it}")
            rows.append((prod, qty, float(prod["unit_price"] or 0)))
        amount = round(sum(q * p for _, q, p in rows), 2)

        if self.available:
            oid = self._execute(
                "INSERT INTO sales_orders (tenant_id, order_no, customer_id, "
                "amount, status, sales_rep, expected_date, remark, created_by) "
                "VALUES (%s,'PENDING',%s,%s,'draft',%s,%s,%s,%s)",
                (tenant_id, cust["id"], amount, acting_user, expected_date,
                 remark, acting_user))
            no = self._next_no("SO", oid)
            self._execute("UPDATE sales_orders SET order_no = %s WHERE id = %s",
                          (no, oid))
            for prod, qty, price in rows:
                self._execute(
                    "INSERT INTO sales_order_items (tenant_id, order_id, "
                    "product_id, quantity, unit_price, line_amount) "
                    "VALUES (%s,%s,%s,%s,%s,%s)",
                    (tenant_id, oid, prod["id"], qty, price,
                     round(qty * price, 2)))
            return self.get_order(oid, tenant_id)
        self._fallback_seq += 1
        oid = self._fallback_seq
        order = {"id": oid, "order_no": self._next_no("SO", oid),
                 "tenant_id": tenant_id, "customer_id": cust["id"],
                 "amount": amount, "status": "draft",
                 "sales_rep": acting_user, "remark": remark,
                 "created_by": acting_user, "items": [
                     {"id": i + 1, "product_id": p["id"],
                      "product_code": p["product_code"],
                      "quantity": q, "unit_price": pr,
                      "line_amount": round(q * pr, 2), "delivered_qty": 0}
                     for i, (p, q, pr) in enumerate(rows)]}
        self._fallback_rows.append(order)
        return order   # 原引用（内存态状态更新直接生效）

    # ==================================================================
    # 确认（占用库存 + 信用校验）
    # ==================================================================
    def confirm_order(self, acting_role: str, acting_user: str,
                      order_id: int, tenant_id: str = "default") -> Dict:
        """
        draft → confirmed：信用额度校验 + 逐行占用库存（防超卖）。

        信用口径：该客户 draft/confirmed/delivering 状态订单的 amount 汇总
        + 本单 ≤ customers.credit_limit（credit_limit=0 不限制，demo 方便）。
        """
        o = self._own_order(acting_role, acting_user, order_id, tenant_id)
        if o["status"] != "draft":
            raise ValueError(f"订单 {o['order_no']} 状态 {o['status']}，仅 draft 可确认")
        cust = self._customer(o, tenant_id)
        if cust and float(cust.get("credit_limit") or 0) > 0:
            outstanding = float(o["amount"])
            for other in self.query_orders("gm", "system", tenant_id):
                if (other["customer_id"] == o["customer_id"]
                        and other["status"] in ("draft", "confirmed", "delivering")
                        and other["id"] != o["id"]):
                    outstanding += float(other["amount"])
            if outstanding > float(cust["credit_limit"]):
                raise ValueError(
                    f"超出客户信用额度：{cust['name']} 在途+本单 "
                    f"{outstanding:.2f} > 额度 {cust['credit_limit']}，订单被拒绝")
        wh = self.inv.default_warehouse(tenant_id, md=self.md)
        if not wh:
            raise ValueError("租户无可用仓库（先建仓库主数据）")
        for it in self._order_items(order_id, tenant_id):
            # 先全量校验再占用（一期简化；多行并发窗口二期以事务化收口）
            lv = self.inv.level(tenant_id, wh["id"], it["product_id"])
            if lv["available"] < it["quantity"]:
                raise ValueError(
                    f"可用库存不足：产品 #{it['product_id']} 可用 "
                    f"{lv['available']} < 需占用 {it['quantity']}（防超卖，"
                    f"订单确认被拒——可先采购补货）")
        for it in self._order_items(order_id, tenant_id):
            self.inv.reserve(tenant_id, wh["id"], it["product_id"],
                             it["quantity"], "sales_order", o["order_no"],
                             operator=acting_user)
        return self._set_order_status(order_id, tenant_id, "confirmed")

    # ==================================================================
    # 出库（执行单：审批后 release 才扣库存）
    # ==================================================================
    def create_delivery(self, acting_role: str, acting_user: str,
                        order_id: int, tenant_id: str = "default",
                        lines: List[Dict] = None) -> Dict:
        """
        创建出库单（draft）。lines: [{order_item_id, quantity}]，
        缺省 = 全部未出量一次性出库。校验 quantity ≤ 行未出量。
        """
        o = self._own_order(acting_role, acting_user, order_id, tenant_id)
        if o["status"] not in ("confirmed", "delivering"):
            raise ValueError(
                f"订单 {o['order_no']} 状态 {o['status']}，须 confirmed 后才能出库")
        wh = self.inv.default_warehouse(tenant_id, md=self.md)
        items = self._order_items(order_id, tenant_id)
        if not lines:
            lines = [{"order_item_id": it["id"], "quantity":
                      it["quantity"] - it["delivered_qty"]} for it in items
                     if it["quantity"] > it["delivered_qty"]]
        if not lines:
            raise ValueError("订单已全部出库，无可出明细")
        by_id = {it["id"]: it for it in items}
        for ln in lines:
            it = by_id.get(ln.get("order_item_id"))
            if not it:
                raise ValueError(f"订单明细 #{ln.get('order_item_id')} 不存在")
            remain = it["quantity"] - it["delivered_qty"]
            if int(ln["quantity"]) <= 0 or int(ln["quantity"]) > remain:
                raise ValueError(
                    f"出库量非法：行 #{it['id']} 未出 {remain}，"
                    f"本次 {ln['quantity']}（支持分批，但不可超未出量）")
        if self.available:
            did = self._execute(
                "INSERT INTO sales_deliveries (tenant_id, delivery_no, order_id, "
                "warehouse_id, status, created_by) "
                "VALUES (%s,'PENDING',%s,%s,'draft',%s)",
                (tenant_id, order_id, wh["id"], acting_user))
            no = self._next_no("SD", did)
            self._execute("UPDATE sales_deliveries SET delivery_no = %s "
                          "WHERE id = %s", (no, did))
            for ln in lines:
                it = by_id[ln["order_item_id"]]
                self._execute(
                    "INSERT INTO sales_delivery_items (tenant_id, delivery_id, "
                    "order_item_id, product_id, quantity, snapshot_price) "
                    "VALUES (%s,%s,%s,%s,%s,%s)",
                    (tenant_id, did, it["id"], it["product_id"],
                     ln["quantity"], it["unit_price"]))
            return self.get_delivery(did, tenant_id)
        self._fallback_seq += 1
        did = self._fallback_seq
        d = {"id": did, "delivery_no": self._next_no("SD", did),
             "tenant_id": tenant_id, "order_id": order_id,
             "warehouse_id": wh["id"] if wh else None, "status": "draft",
             "created_by": acting_user,
             "items": [{"order_item_id": ln["order_item_id"],
                        "product_id": by_id[ln["order_item_id"]]["product_id"],
                        "quantity": ln["quantity"]} for ln in lines]}
        self._fallback_rows.append(d)
        return dict(d)

    def execute_release(self, delivery_id: int, tenant_id: str,
                        approver: str) -> Dict:
        """
        放行出库（审批通过后由 executor 调用）：
        扣库存（同步释放对应占用）→ 回写 delivered_qty → 订单转 delivering。
        """
        d = self.get_delivery(delivery_id, tenant_id)
        if not d:
            raise ValueError(f"出库单 #{delivery_id} 不存在")
        if d["status"] != "draft":
            raise ValueError(f"出库单 {d['delivery_no']} 状态 {d['status']}，不可重复放行")
        for it in d["items"]:
            # 出库扣实物 + 释放等量占用（占用转出库）
            self.inv._stock_out(
                tenant_id, d["warehouse_id"], it["product_id"], it["quantity"],
                "SALE_OUT", "sales_delivery", d["delivery_no"], approver,
                also_release=it["quantity"])
            if self.available:
                self._execute(
                    "UPDATE sales_order_items SET delivered_qty = "
                    "delivered_qty + %s WHERE id = %s AND tenant_id = %s",
                    (it["quantity"], it["order_item_id"], tenant_id))
        if self.available:
            self._execute(
                "UPDATE sales_deliveries SET status = 'released', "
                "released_by = %s WHERE id = %s AND tenant_id = %s",
                (approver, delivery_id, tenant_id))
            # 订单首次出库 → delivering
            self._execute(
                "UPDATE sales_orders SET status = 'delivering' WHERE id = %s "
                "AND tenant_id = %s AND status = 'confirmed'",
                (d["order_id"], tenant_id))
        else:
            d["status"] = "released"
            d["released_by"] = approver
            # 内存态回写：delivered_qty 累加 + 订单首次出库转 delivering
            o = self.get_order(d["order_id"], tenant_id)
            if o:
                by_item = {it["id"]: it for it in o.get("items", [])}
                for it in d["items"]:
                    row = by_item.get(it.get("order_item_id"))
                    if row is not None:
                        row["delivered_qty"] = row.get("delivered_qty", 0) \
                            + it["quantity"]
                if o.get("status") == "confirmed":
                    o["status"] = "delivering"
        return self.get_delivery(delivery_id, tenant_id)

    # ==================================================================
    # 物流
    # ==================================================================
    def ship_delivery(self, acting_role: str, acting_user: str,
                      delivery_id: int, carrier: str,
                      tenant_id: str = "default") -> Dict:
        """released → shipped：交接承运商，生成物流单（TRK-）。"""
        self._check_op("update", acting_role, acting_user, tenant_id,
                       resource="shipment")
        d = self.get_delivery(delivery_id, tenant_id)
        if not d:
            raise ValueError(f"出库单 #{delivery_id} 不存在")
        if d["status"] != "released":
            raise ValueError(f"出库单 {d['delivery_no']} 须 released 后才能发运")
        if not (carrier or "").strip():
            raise ValueError("承运商不能为空")
        if self.available:
            sid = self._execute(
                "INSERT INTO shipments (tenant_id, shipment_no, delivery_id, "
                "carrier, status, created_by) VALUES (%s,'PENDING',%s,%s,"
                "'created',%s)", (tenant_id, delivery_id, carrier, acting_user))
            no = self._next_no("TRK", sid)
            self._execute("UPDATE shipments SET shipment_no = %s, "
                          "shipped_at = NOW(), status = 'in_transit' "
                          "WHERE id = %s", (no, sid))
            self._execute("UPDATE sales_deliveries SET status = 'shipped' "
                          "WHERE id = %s AND tenant_id = %s",
                          (delivery_id, tenant_id))
            return self.get_shipment(sid, tenant_id)
        self._fallback_seq += 1
        sid = self._fallback_seq
        s = {"id": sid, "shipment_no": self._next_no("TRK", sid),
             "tenant_id": tenant_id, "delivery_id": delivery_id,
             "carrier": carrier, "status": "in_transit"}
        self._fallback_rows.append(s)
        return dict(s)

    def deliver_shipment(self, acting_role: str, acting_user: str,
                         shipment_id: int, tenant_id: str = "default") -> Dict:
        """物流签收：in_transit → delivered。"""
        s = self.get_shipment(shipment_id, tenant_id)
        if not s:
            raise ValueError(f"物流单 #{shipment_id} 不存在")
        if s["status"] != "in_transit":
            raise ValueError(f"物流单 {s['shipment_no']} 状态 {s['status']}")
        if self.available:
            self._execute("UPDATE shipments SET status = 'delivered', "
                          "delivered_at = NOW() WHERE id = %s AND tenant_id = %s",
                          (shipment_id, tenant_id))
            s["status"] = "delivered"
        return s

    # ==================================================================
    # 关单 / 取消（高危：审批 executor 调用）
    # ==================================================================
    def execute_complete(self, order_id: int, tenant_id: str,
                         approver: str) -> Dict:
        """delivering → completed（审批通过后执行；金额>10万须 level=3 审批）。"""
        o = self.get_order(order_id, tenant_id)
        if not o:
            raise ValueError(f"订单 #{order_id} 不存在")
        if o["status"] != "delivering":
            raise ValueError(f"订单 {o['order_no']} 状态 {o['status']}，"
                             "须出库中才能关单")
        for it in self._order_items(order_id, tenant_id):
            if it["delivered_qty"] < it["quantity"]:
                raise ValueError(
                    f"订单 {o['order_no']} 存在未出完明细（产品 "
                    f"#{it['product_id']} 已出 {it['delivered_qty']}/"
                    f"{it['quantity']}），不可关单")
        return self._set_order_status(order_id, tenant_id, "completed")

    def execute_cancel(self, order_id: int, tenant_id: str,
                       approver: str) -> Dict:
        """
        取消订单（审批通过后执行）：draft 直接收口；
        confirmed 须先释放全部占用（防超卖语义闭环）。
        """
        o = self.get_order(order_id, tenant_id)
        if not o:
            raise ValueError(f"订单 #{order_id} 不存在")
        if o["status"] not in ("draft", "confirmed"):
            raise ValueError(f"订单 {o['order_no']} 状态 {o['status']}，不可取消")
        if o["status"] == "confirmed":
            wh = self.inv.default_warehouse(tenant_id, md=self.md)
            for it in self._order_items(order_id, tenant_id):
                remain = it["quantity"] - it["delivered_qty"]
                if remain > 0 and wh:
                    self.inv.release(tenant_id, wh["id"], it["product_id"],
                                     remain, "sales_order", o["order_no"],
                                     operator=approver)
        return self._set_order_status(order_id, tenant_id, "cancelled")

    # ==================================================================
    # 退货
    # ==================================================================
    def create_return(self, acting_role: str, acting_user: str,
                      order_id: int, product_code: str, quantity: int,
                      reason: str, tenant_id: str = "default") -> Dict:
        """创建退货单（draft）。校验退货量 ≤ 该订单已出量。"""
        self._check_op("create", acting_role, acting_user, tenant_id)
        o = self.get_order(order_id, tenant_id)
        if not o:
            raise ValueError(f"订单 #{order_id} 不存在")
        prod = self.md.products.get_by_code(product_code, tenant_id)
        if not prod:
            raise ValueError(f"产品「{product_code}」不存在")
        shipped = sum(it["delivered_qty"] for it in
                      self._order_items(order_id, tenant_id)
                      if it["product_id"] == prod["id"])
        if int(quantity) <= 0 or int(quantity) > shipped:
            raise ValueError(f"退货量非法：该产品已出 {shipped}，本次 {quantity}")
        if self.available:
            rid = self._execute(
                "INSERT INTO sales_returns (tenant_id, return_no, order_id, "
                "product_id, quantity, reason, status, created_by) "
                "VALUES (%s,'PENDING',%s,%s,%s,%s,'draft',%s)",
                (tenant_id, order_id, prod["id"], quantity, reason, acting_user))
            no = self._next_no("SR", rid)
            self._execute("UPDATE sales_returns SET return_no = %s WHERE id = %s",
                          (no, rid))
            return self.get_return(rid, tenant_id)
        self._fallback_seq += 1
        rid = self._fallback_seq
        r = {"id": rid, "return_no": self._next_no("SR", rid),
             "tenant_id": tenant_id, "order_id": order_id,
             "product_id": prod["id"], "quantity": quantity,
             "reason": reason, "status": "draft"}
        self._fallback_rows.append(r)
        return dict(r)

    def receive_return(self, acting_role: str, acting_user: str,
                       return_id: int, inspect_result: str,
                       tenant_id: str = "default") -> Dict:
        """
        退货收货：draft → received。
        质检 ok → 回补库存（RETURN_IN）；scrap → 报废不回库（只记流水备注）。
        """
        self._check_op("update", acting_role, acting_user, tenant_id)
        r = self.get_return(return_id, tenant_id)
        if not r:
            raise ValueError(f"退货单 #{return_id} 不存在")
        if r["status"] != "draft":
            raise ValueError(f"退货单 {r['return_no']} 状态 {r['status']}")
        if inspect_result not in ("ok", "scrap"):
            raise ValueError("质检结论取值 ok（重入库）/ scrap（报废）")
        wh = self.inv.default_warehouse(tenant_id, md=self.md)
        if wh:
            if inspect_result == "ok":
                self.inv._stock_in(tenant_id, wh["id"], r["product_id"],
                                   r["quantity"], "RETURN_IN", "sales_return",
                                   r["return_no"], acting_user)
            else:
                # 报废：实物不回库，落一条 0 量流水留痕（备注报废）
                self.inv._stock_in(tenant_id, wh["id"], r["product_id"], 0,
                                   "RETURN_IN", "sales_return", r["return_no"],
                                   acting_user,
                                   remark=f"报废 {r['quantity']} 件，不回库")
        if self.available:
            self._execute(
                "UPDATE sales_returns SET status = 'received', "
                "inspect_result = %s, warehouse_id = %s, received_at = NOW() "
                "WHERE id = %s AND tenant_id = %s",
                (inspect_result, wh["id"] if wh else None, return_id, tenant_id))
            r = self.get_return(return_id, tenant_id)
        else:
            r["status"], r["inspect_result"] = "received", inspect_result
        return r

    # ==================================================================
    # 查询
    # ==================================================================
    def query_orders(self, acting_role: str, acting_user: str,
                     tenant_id: str = "default", status: str = None,
                     limit: int = 20) -> List[Dict]:
        """订单列表（L2 行级裁剪：销售只看自己的，经理看本租户全部）。"""
        rule = self._check_op("query", acting_role, acting_user, tenant_id)
        conds = self._visibility_where(rule, acting_user, tenant_id)
        if self.available:
            sql, params = ("SELECT o.*, c.name AS customer_name, c.customer_code "
                           "FROM sales_orders o LEFT JOIN customers c "
                           "ON o.customer_id = c.id AND o.tenant_id = c.tenant_id "
                           "WHERE o.tenant_id = %s"), [tenant_id]
            for frag, p in conds:
                if frag.startswith("tenant_id"):
                    continue   # 已有 o.tenant_id 条件（避免列名歧义）
                sql += f" AND o.{frag}"
                params.append(p)
            if status:
                sql += " AND o.status = %s"
                params.append(status)
            sql += " ORDER BY o.id DESC LIMIT %s"
            params.append(limit)
            return self._execute(sql, tuple(params)) or []
        scope = rule.get("scope", "own")
        rows = [dict(r) for r in self._fallback_rows
                if "order_no" in r
                and (r["tenant_id"] == tenant_id)
                and (scope != "own" or r.get("created_by") == acting_user)
                and (not status or r["status"] == status)]
        return list(reversed(rows))[:limit]

    def get_order(self, order_id: int, tenant_id: str) -> Optional[Dict]:
        if self.available:
            rows = self._execute(
                "SELECT o.*, c.name AS customer_name FROM sales_orders o "
                "LEFT JOIN customers c ON o.customer_id = c.id "
                "AND o.tenant_id = c.tenant_id WHERE o.id = %s "
                "AND o.tenant_id = %s", (order_id, tenant_id))
            if not rows:
                return None
            o = rows[0]
            o["items"] = self._order_items(order_id, tenant_id)
            return o
        for r in self._fallback_rows:
            if r.get("id") == order_id and r["tenant_id"] == tenant_id \
                    and "order_no" in r:
                return r   # 原引用：execute_* 的内存态状态更新直接生效
        return None

    def get_delivery(self, did: int, tenant_id: str) -> Optional[Dict]:
        if self.available:
            rows = self._execute(
                "SELECT * FROM sales_deliveries WHERE id = %s AND tenant_id = %s",
                (did, tenant_id))
            if not rows:
                return None
            d = rows[0]
            d["items"] = self._execute(
                "SELECT * FROM sales_delivery_items WHERE delivery_id = %s "
                "AND tenant_id = %s ORDER BY id", (did, tenant_id)) or []
            return d
        for r in self._fallback_rows:
            if r.get("id") == did and r["tenant_id"] == tenant_id \
                    and "delivery_no" in r:
                return r   # 原引用
        return None

    def get_shipment(self, sid: int, tenant_id: str) -> Optional[Dict]:
        if self.available:
            rows = self._execute(
                "SELECT * FROM shipments WHERE id = %s AND tenant_id = %s",
                (sid, tenant_id))
            return rows[0] if rows else None
        for r in self._fallback_rows:
            if r.get("id") == sid and r["tenant_id"] == tenant_id \
                    and "shipment_no" in r:
                return r   # 原引用
        return None

    def get_return(self, rid: int, tenant_id: str) -> Optional[Dict]:
        if self.available:
            rows = self._execute(
                "SELECT * FROM sales_returns WHERE id = %s AND tenant_id = %s",
                (rid, tenant_id))
            return rows[0] if rows else None
        for r in self._fallback_rows:
            if r.get("id") == rid and r["tenant_id"] == tenant_id \
                    and "return_no" in r:
                return r   # 原引用：receive_return 的内存态状态更新直接生效
        return None

    # ==================================================================
    # 内部
    # ==================================================================
    def _order_items(self, order_id: int, tenant_id: str) -> List[Dict]:
        if self.available:
            return self._execute(
                "SELECT i.*, p.product_code, p.name AS product_name "
                "FROM sales_order_items i LEFT JOIN products p "
                "ON i.product_id = p.id AND i.tenant_id = p.tenant_id "
                "WHERE i.order_id = %s AND i.tenant_id = %s ORDER BY i.id",
                (order_id, tenant_id)) or []
        o = next((r for r in self._fallback_rows
                  if r.get("id") == order_id and "order_no" in r), None)
        return (o or {}).get("items", [])

    def _own_order(self, acting_role, acting_user, order_id, tenant_id) -> Dict:
        o = self.get_order(order_id, tenant_id)
        if not o:
            raise ValueError(f"订单 #{order_id} 不存在")
        self._check_op("update", acting_role, acting_user, tenant_id)
        return o

    def _customer(self, o: Dict, tenant_id: str) -> Optional[Dict]:
        """订单客户的信用额度档案（MySQL join 查询；内存态走 fallback rows）。"""
        if self.md.customers.available:
            rows = self.md.customers._execute(
                "SELECT * FROM customers WHERE id = %s AND tenant_id = %s",
                (o["customer_id"], tenant_id))
            return rows[0] if rows else None
        for r in self.md.customers._fallback_rows:
            if r.get("id") == o["customer_id"] and r["tenant_id"] == tenant_id:
                return r
        return None

    def _set_order_status(self, order_id, tenant_id, status) -> Dict:
        if self.available:
            done = ", completed_at = NOW()" if status == "completed" else ""
            self._execute(
                f"UPDATE sales_orders SET status = %s{done} "
                "WHERE id = %s AND tenant_id = %s",
                (status, order_id, tenant_id))
        else:
            o = self.get_order(order_id, tenant_id)   # 原引用，直接改内存行
            if o:
                o["status"] = status
        return self.get_order(order_id, tenant_id)
