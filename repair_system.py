"""
================================================================================
 repair_system.py — 设备维修域（工单 / 故障归类 / 派单 / 领料 / 保养计划）
================================================================================

 核心联动（工厂业务系统设计 §3.5）
 ----
   设备台账联动：创建工单按 equipment 自动判——
     · 保内/保外：warranty_until ≥ 今天 → in（材料费记保修成本）否则 out
     · 关键度：criticality=A 的设备故障自动升 urgent
   故障代码库：classify 落 fault_code_id（标准 SOP 与 RAG 检索联动）
   派单：assign 按 fault_code.category 与工程师 skill 匹配给建议（不强制）
   领料：use_repair_parts → 出库单（ref_type=repair）→ 扣库存（PARTS_OUT 流水）
   保养计划（PM）：trigger_due 到期手动生成工单（source=pm，已定稿先手动）

 状态机：open → assigned → in_progress → resolved → verified；旁路 cancelled
   resolved / cancel 属高危（审批门，level≥2）；verified = 使用方确认修复。
 表结构权威源 config/init_db.sql（表 30-33）。
================================================================================
"""

import time
from datetime import datetime
from typing import Dict, List, Optional

from erp_common import ErpDb

_DDL_REPAIR = """
CREATE TABLE IF NOT EXISTS `repair_orders` (
  `id`              BIGINT AUTO_INCREMENT PRIMARY KEY,
  `tenant_id`       VARCHAR(64)   NOT NULL DEFAULT 'default',
  `order_no`        VARCHAR(32)   NOT NULL,
  `equipment_id`    BIGINT        NOT NULL,
  `fault_code_id`   BIGINT        NULL,
  `fault_desc`      TEXT          NOT NULL,
  `priority`        VARCHAR(16)   NOT NULL DEFAULT 'normal',
  `warranty`        VARCHAR(8)    NOT NULL DEFAULT 'out',
  `source`          VARCHAR(16)   NOT NULL DEFAULT 'report',
  `technician_id`   BIGINT        NULL,
  `status`          VARCHAR(16)   NOT NULL DEFAULT 'open',
  `downtime_hours`  DECIMAL(8,2)  NULL,
  `resolution`      TEXT          NULL,
  `resolved_at`     DATETIME      NULL,
  `created_by`      VARCHAR(64)   NULL,
  `created_at`      DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at`      DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX `idx_tenant_status` (`tenant_id`, `status`, `priority`, `id` DESC),
  INDEX `idx_equipment` (`equipment_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

_DDL_PARTS = """
CREATE TABLE IF NOT EXISTS `repair_parts` (
  `id`               BIGINT AUTO_INCREMENT PRIMARY KEY,
  `tenant_id`        VARCHAR(64)   NOT NULL DEFAULT 'default',
  `repair_order_id`  BIGINT        NOT NULL,
  `product_id`       BIGINT        NOT NULL,
  `quantity`         INT           NOT NULL,
  `issued_at`        DATETIME      NULL,
  `remark`           VARCHAR(256)  NULL,
  `created_by`       VARCHAR(64)   NULL,
  INDEX `idx_order` (`repair_order_id`),
  INDEX `idx_product` (`product_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

_DDL_PM = """
CREATE TABLE IF NOT EXISTS `pm_plans` (
  `id`           BIGINT AUTO_INCREMENT PRIMARY KEY,
  `tenant_id`    VARCHAR(64)   NOT NULL DEFAULT 'default',
  `plan_no`      VARCHAR(32)   NOT NULL,
  `equipment_id` BIGINT        NOT NULL,
  `name`         VARCHAR(128)  NOT NULL,
  `cycle_type`   VARCHAR(16)   NOT NULL DEFAULT 'monthly',
  `last_done_at` DATETIME      NULL,
  `next_due_at`  DATETIME      NOT NULL,
  `checklist`    JSON          NULL,
  `assignee_id`  BIGINT        NULL,
  `is_active`    TINYINT       NOT NULL DEFAULT 1,
  `created_by`   VARCHAR(64)   NULL,
  `created_at`   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX `idx_tenant_due` (`tenant_id`, `is_active`, `next_due_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

REPAIR_TRANSITIONS = {
    "open": {"assigned", "cancelled"},
    "assigned": {"in_progress", "cancelled"},
    "in_progress": {"resolved", "cancelled"},
    "resolved": {"verified", },
    "verified": set(),
    "cancelled": set(),
}

_CYCLE_DAYS = {"daily": 1, "weekly": 7, "monthly": 30, "quarterly": 90}


class RepairService(ErpDb):
    """设备维修域唯一入口。库存联动经 InventoryService。"""

    RESOURCE = "repair_order"

    def __init__(self, inventory=None, master=None, verbose: bool = True):
        self.TABLE, self.DDL = "repair_orders", _DDL_REPAIR
        super().__init__(verbose=verbose)
        if self.available:
            for ddl in (_DDL_PARTS, _DDL_PM):
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(ddl)
                    conn.commit()
        from inventory_system import InventoryService
        from master_data import MasterData
        self.inv = inventory or InventoryService(verbose=False)
        self.md = master or MasterData(verbose=False)

    # ==================================================================
    # 创建（设备台账联动：保内/保外 + 关键度升 urgent）
    # ==================================================================
    def create_order(self, acting_role: str, acting_user: str,
                     tenant_id: str, equipment_code: str, fault_desc: str,
                     priority: str = None,
                     source: str = "digital_employee") -> Dict:
        """
        创建维修工单。

        联动规则（不许人工覆盖保内/保外；关键度只升不降）：
          warranty_until ≥ 今天 → warranty=in（材料走保修成本），否则 out
          criticality=A → priority 自动 urgent（传参也会被覆盖）
        """
        self._check_op("create", acting_role, acting_user, tenant_id)
        eq = self.md.equipment.get_by_code(equipment_code, tenant_id)
        if not eq:
            raise ValueError(f"设备「{equipment_code}」不在台账（先建设备主数据）")
        if not (fault_desc or "").strip():
            raise ValueError("故障现象不能为空")
        # 保内/保外：按台账自动判（人工不可指定）
        warranty = "out"
        if eq.get("warranty_until"):
            try:
                due = eq["warranty_until"]
                due = due if isinstance(due, datetime) else datetime.strptime(
                    str(due)[:10], "%Y-%m-%d")
                warranty = "in" if due.date() >= datetime.now().date() else "out"
            except Exception:
                warranty = "out"
        # 关键度 A → urgent（只升不降）
        if eq.get("criticality") == "A":
            priority = "urgent"
        priority = priority or "normal"
        if priority not in ("low", "normal", "high", "urgent"):
            raise ValueError("priority 取值 low/normal/high/urgent")

        if self.available:
            oid = self._execute(
                "INSERT INTO repair_orders (tenant_id, order_no, equipment_id, "
                "fault_desc, priority, warranty, source, status, created_by) "
                "VALUES (%s,'PENDING',%s,%s,%s,%s,%s,'open',%s)",
                (tenant_id, eq["id"], fault_desc, priority, warranty, source,
                 acting_user))
            no = self._next_no("RO", oid)
            self._execute("UPDATE repair_orders SET order_no = %s WHERE id = %s",
                          (no, oid))
            return self.get_order(oid, tenant_id)
        self._fallback_seq += 1
        oid = self._fallback_seq
        row = {"id": oid, "order_no": self._next_no("RO", oid),
               "tenant_id": tenant_id, "equipment_id": eq["id"],
               "equipment_code": equipment_code, "fault_desc": fault_desc,
               "priority": priority, "warranty": warranty, "source": source,
               "status": "open", "created_by": acting_user}
        self._fallback_rows.append(row)
        return dict(row)

    # ==================================================================
    # 诊断 → 派单 → 开工
    # ==================================================================
    def classify_fault(self, acting_role: str, acting_user: str,
                       order_id: int, fault_code: str,
                       tenant_id: str = "default") -> Dict:
        """故障归类：落 fault_code_id（open/assigned/in_progress 均可归类）。"""
        self._check_op("update", acting_role, acting_user, tenant_id)
        o = self.get_order(order_id, tenant_id)
        if not o:
            raise ValueError(f"工单 #{order_id} 不存在")
        if o["status"] not in ("open", "assigned", "in_progress"):
            raise ValueError(f"工单 {o['order_no']} 状态 {o['status']}，不可归类")
        fc = self.md.fault_codes.get_by_code(fault_code, tenant_id)
        if not fc:
            raise ValueError(f"故障代码「{fault_code}」不存在")
        if self.available:
            self._execute(
                "UPDATE repair_orders SET fault_code_id = %s WHERE id = %s "
                "AND tenant_id = %s", (fc["id"], order_id, tenant_id))
            return self.get_order(order_id, tenant_id)
        o["fault_code_id"] = fc["id"]
        return o

    def assign_technician(self, acting_role: str, acting_user: str,
                          order_id: int, engineer_name: str,
                          tenant_id: str = "default") -> Dict:
        """
        派单：open → assigned；assigned 状态可改派（换人）。

        技能匹配建议：故障类别（fault_code.category）与工程师 skill 对口
        优先；不强制（老师傅跨工种常见）。不匹配时结果里带 hint。
        """
        self._check_op("update", acting_role, acting_user, tenant_id)
        o = self.get_order(order_id, tenant_id)
        if not o:
            raise ValueError(f"工单 #{order_id} 不存在")
        if o["status"] not in ("open", "assigned"):
            raise ValueError(f"工单 {o['order_no']} 状态 {o['status']}，"
                             "仅 open/assigned 可（改）派单")
        eng = None
        # F9 修复：内部联动查在册名单走系统通道（此前冒充 "repair_user"
        # 身份，其无 engineer 资源 → PermissionDenied 段崩溃）
        for e in self.md.engineers._list_for_system(tenant_id=tenant_id):
            if e["name"] == engineer_name:
                eng = e
                break
        if not eng:
            raise ValueError(f"工程师「{engineer_name}」不在册")
        hint = ""
        if o.get("fault_code_id"):
            # 技能匹配提示：按故障类别 vs 工程师专长（两模式通用的 get_by_code
            # 不行——这里按 id 查；MySQL 走 SQL，内存态扫 fallback rows）
            fc = None
            if self.md.fault_codes.available:
                fcs = self.md.fault_codes._execute(
                    "SELECT category FROM fault_codes WHERE id = %s "
                    "AND tenant_id = %s", (o["fault_code_id"], tenant_id))
                fc = fcs[0] if fcs else None
            else:
                for r in self.md.fault_codes._fallback_rows:
                    if r.get("id") == o["fault_code_id"] \
                            and r["tenant_id"] == tenant_id:
                        fc = r
                        break
            if fc and fc.get("category") != eng.get("skill"):
                hint = (f"⚠ 技能提示：故障类别 {fc['category']}，"
                        f"{eng['name']} 专长 {eng.get('skill')}（跨工种派单）")
        if self.available:
            self._execute(
                "UPDATE repair_orders SET technician_id = %s, status='assigned' "
                "WHERE id = %s AND tenant_id = %s",
                (eng["id"], order_id, tenant_id))
        else:
            o["technician_id"], o["status"] = eng["id"], "assigned"
        # 展示层字段（assign_hint）写入**副本**——不污染工单数据行
        # （内存态 get_order 返回原引用，直接写会让下一次派单读到残留提示）
        result = dict(self.get_order(order_id, tenant_id) or {})
        if hint:
            result["assign_hint"] = hint
        return result

    def start_repair(self, acting_role: str, acting_user: str,
                     order_id: int, tenant_id: str = "default") -> Dict:
        """开工：assigned → in_progress。"""
        self._check_op("update", acting_role, acting_user, tenant_id)
        o = self._must(order_id, tenant_id, "assigned", "开工")
        if self.available:
            self._execute("UPDATE repair_orders SET status='in_progress' "
                          "WHERE id = %s AND tenant_id = %s",
                          (order_id, tenant_id))
        else:
            o["status"] = "in_progress"
        return self.get_order(order_id, tenant_id)

    # ==================================================================
    # 领料（扣库存：出库单 ref_type=repair → PARTS_OUT 流水）
    # ==================================================================
    def use_repair_parts(self, acting_role: str, acting_user: str,
                         order_id: int, lines: List[Dict],
                         tenant_id: str = "default") -> Dict:
        """
        维修领料：lines = [{product_code, quantity}]。
        每次领料：repair_parts 落行 + 出库单（draft 立即 issued）→ 事务扣库存。
        保内工单材料记保修成本，保外记客户收费（remark 注明，财务口径二期）。
        """
        self._check_op("update", acting_role, acting_user, tenant_id)
        o = self._must(order_id, tenant_id, None, "领料",
                       ok_statuses=("assigned", "in_progress"))
        if not lines:
            raise ValueError("领料明细不能为空")
        wh = self.inv.default_warehouse(tenant_id, md=self.md)
        if not wh:
            raise ValueError("租户无可用仓库")
        items, prods = [], []
        for ln in lines:
            prod = self.md.products.get_by_code(ln.get("product_code", ""),
                                                tenant_id)
            if not prod:
                raise ValueError(f"备件「{ln.get('product_code')}」不存在")
            qty = int(ln.get("quantity", 0))
            if qty <= 0:
                raise ValueError(f"领料数量必须为正：{ln}")
            lv = self.inv.level(tenant_id, wh["id"], prod["id"])
            if lv["stock_qty"] < qty:
                raise ValueError(
                    f"备件库存不足：{prod['product_code']} 在库 "
                    f"{lv['stock_qty']} < 需领 {qty}（可先采购补货）")
            items.append({"product_id": prod["id"], "quantity": qty})
            prods.append(prod)
        # 出库单 + 立即执行（事务扣库存，PARTS_OUT 流水）
        cost_note = "保内：材料记保修成本" if o["warranty"] == "in" \
            else "保外：材料记客户收费"
        iss = self.inv.create_issue(
            acting_role, acting_user, tenant_id, wh["id"], items,
            ref_type="repair", ref_id=order_id,
            remark=f"{o['order_no']} 领料（{cost_note}）")
        self.inv.issue_issue(iss["id"], tenant_id, operator=acting_user)
        # 领料行落库（issued_at = 现在）
        for it, prod in zip(items, prods):
            if self.available:
                self._execute(
                    "INSERT INTO repair_parts (tenant_id, repair_order_id, "
                    "product_id, quantity, issued_at, remark, created_by) "
                    "VALUES (%s,%s,%s,%s,NOW(),%s,%s)",
                    (tenant_id, order_id, it["product_id"], it["quantity"],
                     cost_note, acting_user))
        return self.list_parts(order_id, tenant_id)

    def list_parts(self, order_id: int, tenant_id: str) -> List[Dict]:
        """工单的领料明细。"""
        if self.available:
            return self._execute(
                "SELECT r.*, p.product_code, p.name AS product_name "
                "FROM repair_parts r LEFT JOIN products p "
                "ON r.product_id = p.id AND r.tenant_id = p.tenant_id "
                "WHERE r.repair_order_id = %s AND r.tenant_id = %s ORDER BY r.id",
                (order_id, tenant_id)) or []
        return []

    # ==================================================================
    # 完工 / 取消（高危：审批 executor 调用）/ 验证
    # ==================================================================
    def execute_resolve(self, order_id: int, tenant_id: str, approver: str,
                        resolution: str = "", downtime_hours: float = None
                        ) -> Dict:
        """
        完工（审批通过后执行）：in_progress → resolved。
        resolution（处置结论）与 downtime_hours 随审批 payload 带入。
        """
        o = self._must(order_id, tenant_id, "in_progress", "完工")
        if self.available:
            self._execute(
                "UPDATE repair_orders SET status='resolved', resolution=%s, "
                "downtime_hours=%s, resolved_at=NOW() WHERE id = %s "
                "AND tenant_id = %s",
                (resolution, downtime_hours, order_id, tenant_id))
        else:
            o.update({"status": "resolved", "resolution": resolution,
                      "downtime_hours": downtime_hours})
        return self.get_order(order_id, tenant_id)

    def execute_cancel(self, order_id: int, tenant_id: str,
                       approver: str) -> Dict:
        """取消工单（审批通过后执行）：open/assigned/in_progress → cancelled。"""
        o = self.get_order(order_id, tenant_id)
        if not o:
            raise ValueError(f"工单 #{order_id} 不存在")
        if o["status"] not in ("open", "assigned", "in_progress"):
            raise ValueError(f"工单 {o['order_no']} 状态 {o['status']}，不可取消")
        if self.available:
            self._execute("UPDATE repair_orders SET status='cancelled' "
                          "WHERE id = %s AND tenant_id = %s",
                          (order_id, tenant_id))
        else:
            o["status"] = "cancelled"
        return self.get_order(order_id, tenant_id)

    def verify_repair(self, acting_role: str, acting_user: str,
                      order_id: int, tenant_id: str = "default") -> Dict:
        """使用方确认修复：resolved → verified（防「修完又坏」无确认关单）。"""
        self._check_op("update", acting_role, acting_user, tenant_id)
        o = self._must(order_id, tenant_id, "resolved", "验证")
        if self.available:
            self._execute("UPDATE repair_orders SET status='verified' "
                          "WHERE id = %s AND tenant_id = %s",
                          (order_id, tenant_id))
        else:
            o["status"] = "verified"
        return self.get_order(order_id, tenant_id)

    # ==================================================================
    # 查询（own 语义特殊：维修工程师看「自己被指派的 + 自己报的」）
    # ==================================================================
    def query_orders(self, acting_role: str, acting_user: str,
                     tenant_id: str = "default", status: str = None,
                     limit: int = 20) -> List[Dict]:
        rule = self._check_op("query", acting_role, acting_user, tenant_id)
        if self.available:
            sql = ("SELECT o.*, e.equipment_code, e.name AS equipment_name, "
                   "e.criticality, eng.name AS technician_name, fc.code AS "
                   "fault_code FROM repair_orders o "
                   "LEFT JOIN equipment e ON o.equipment_id = e.id "
                   "AND o.tenant_id = e.tenant_id "
                   "LEFT JOIN engineers eng ON o.technician_id = eng.id "
                   "AND o.tenant_id = eng.tenant_id "
                   "LEFT JOIN fault_codes fc ON o.fault_code_id = fc.id "
                   "AND o.tenant_id = fc.tenant_id WHERE o.tenant_id = %s")
            params: list = [tenant_id]
            if rule.get("scope") == "own":
                sql += " AND (o.created_by = %s OR eng.name = %s)"
                params += [acting_user, acting_user]
            if status:
                sql += " AND o.status = %s"
                params.append(status)
            sql += " ORDER BY o.id DESC LIMIT %s"
            params.append(limit)
            return self._execute(sql, tuple(params)) or []
        scope = rule.get("scope", "own")
        rows = [dict(r) for r in self._fallback_rows
                if "order_no" in r and r["tenant_id"] == tenant_id
                and (scope != "own" or r.get("created_by") == acting_user)
                and (not status or r["status"] == status)]
        return list(reversed(rows))[:limit]

    def get_order(self, order_id: int, tenant_id: str) -> Optional[Dict]:
        if self.available:
            rows = self.query_orders("gm", "system", tenant_id=tenant_id,
                                     limit=10**6)
            return next((r for r in rows if r["id"] == order_id), None)
        for r in self._fallback_rows:
            if r.get("id") == order_id and r["tenant_id"] == tenant_id \
                    and "order_no" in r:
                return r   # 原引用：execute_* 的内存态状态更新直接生效
        return None

    # ==================================================================
    # 保养计划（PM，已定稿：先手动触发）
    # ==================================================================
    def create_pm_plan(self, acting_role: str, acting_user: str,
                       tenant_id: str, equipment_code: str, name: str,
                       cycle_type: str, first_due_at: str,
                       checklist: List[str] = None,
                       assignee_name: str = None) -> Dict:
        """创建保养计划。first_due_at 格式 YYYY-MM-DD HH:MM:SS。"""
        self._check_op("create", acting_role, acting_user, tenant_id,
                       resource="pm_plan")
        eq = self.md.equipment.get_by_code(equipment_code, tenant_id)
        if not eq:
            raise ValueError(f"设备「{equipment_code}」不在台账")
        if cycle_type not in _CYCLE_DAYS:
            raise ValueError(f"cycle_type 取值 {sorted(_CYCLE_DAYS)}")
        import json as _json
        if self.available:
            pid = self._execute(
                "INSERT INTO pm_plans (tenant_id, plan_no, equipment_id, name, "
                "cycle_type, next_due_at, checklist, assignee_id, created_by) "
                "VALUES (%s,'PENDING',%s,%s,%s,%s,%s,%s,%s)",
                (tenant_id, eq["id"], name, cycle_type, first_due_at,
                 _json.dumps(checklist or [], ensure_ascii=False), None,
                 acting_user))
            no = self._next_no("PM", pid)
            self._execute("UPDATE pm_plans SET plan_no = %s WHERE id = %s",
                          (no, pid))
            return self.get_pm_plan(pid, tenant_id)
        self._fallback_seq += 1
        pid = self._fallback_seq
        row = {"id": pid, "plan_no": self._next_no("PM", pid),
               "tenant_id": tenant_id, "equipment_id": eq["id"], "name": name,
               "cycle_type": cycle_type, "next_due_at": first_due_at,
               "checklist": checklist or [], "is_active": 1}
        self._fallback_rows.append(row)
        return dict(row)

    def list_due_plans(self, acting_role: str, acting_user: str,
                       tenant_id: str = "default") -> List[Dict]:
        """列出已到期的保养计划（数字员工巡检「该保养了」。"""
        self._check_op("query", acting_role, acting_user, tenant_id,
                       resource="pm_plan")
        if not self.available:
            return []
        rows = self._execute(
            "SELECT p.*, e.equipment_code, e.name AS equipment_name FROM "
            "pm_plans p LEFT JOIN equipment e ON p.equipment_id = e.id "
            "AND p.tenant_id = e.tenant_id WHERE p.tenant_id = %s "
            "AND p.is_active = 1 AND p.next_due_at <= NOW() "
            "ORDER BY p.next_due_at", (tenant_id,)) or []
        return rows

    def trigger_due_pm(self, acting_role: str, acting_user: str,
                       tenant_id: str = "default") -> List[Dict]:
        """
        到期保养生成工单（手动触发，已定稿；自动调度等数字员工 P1）。
        生成后滚动 next_due_at（按周期 +1）并刷新 last_done_at。
        """
        due = self.list_due_plans(acting_role, acting_user, tenant_id)
        created = []
        for p in due:
            desc = f"[PM] {p['name']}（计划 {p['plan_no']}，周期 {p['cycle_type']}）"
            o = self.create_order(acting_role, acting_user, tenant_id,
                                  p["equipment_code"], desc,
                                  priority="normal", source="pm")
            created.append(o)
            if self.available:
                import json as _json
                days = _CYCLE_DAYS[p["cycle_type"]]
                self._execute(
                    "UPDATE pm_plans SET last_done_at = NOW(), next_due_at = "
                    "DATE_ADD(NOW(), INTERVAL %s DAY) WHERE id = %s "
                    "AND tenant_id = %s", (days, p["id"], tenant_id))
        return created

    def get_pm_plan(self, pid: int, tenant_id: str) -> Optional[Dict]:
        if self.available:
            rows = self._execute(
                "SELECT * FROM pm_plans WHERE id = %s AND tenant_id = %s",
                (pid, tenant_id))
            return rows[0] if rows else None
        return None

    # ==================================================================
    # 内部
    # ==================================================================
    def _must(self, order_id, tenant_id, expect_status, action,
              ok_statuses=None) -> Dict:
        o = self.get_order(order_id, tenant_id)
        if not o:
            raise ValueError(f"工单 #{order_id} 不存在")
        ok = ok_statuses if ok_statuses is not None else (
            [expect_status] if expect_status else None)
        if ok and o["status"] not in ok:
            raise ValueError(
                f"工单 {o['order_no']} 状态 {o['status']}，{action}要求 {ok}")
        return o
