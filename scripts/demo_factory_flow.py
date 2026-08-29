# -*- coding: utf-8 -*-
"""
demo_factory_flow.py — 工厂业务系统全流程演示（数字员工数据面）
================================================================================

 演示内容（docs/guides/factory_business_system_design.md 的活样例）：

   第 0 幕   种子数据：角色/仓库/产品/设备/工程师/客户/故障代码 + 期初库存
   第 1 幕   采购补货：采购单 → 提交审批 → 经理批准 → 收货入库（库存+流水）
   第 2 幕   销售流：销售建单 → 确认（信用+防超卖占用）→ 出库单
             → 放行审批（经理批→扣库存）→ 发运 → 大额关单（>10万须超管级）
   第 3 幕   维修流：报修（A类自动urgent/保内自动判）→ 故障归类 → 派单（技能
             匹配提示）→ 开工 → 领料（扣库存）→ 完工审批 → 使用方验证
   第 4 幕   隔离演示：销售查维修工单被拒 / 采购员查销售单被拒（越权审计）
   第 5 幕   库存流水审计视图（对账以流水为准）

 运行：
   python scripts/demo_factory_flow.py                # 全流程（幂等：种子已存在则跳过）
   python scripts/demo_factory_flow.py --skip-seed    # 跳过种子

 环境：MySQL 可达最佳（数据落库）；不可达时自动内存降级，同进程闭环仍完整。
================================================================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from master_data import MasterData                        # noqa: E402
from inventory_system import InventoryService             # noqa: E402
from sales_system import SalesService                     # noqa: E402
from repair_system import RepairService                   # noqa: E402
from purchase_system import PurchaseService               # noqa: E402
from approval import ApprovalGate, build_default_executors  # noqa: E402
from erp_common import PermissionDenied                   # noqa: E402

TENANT = "default"

# C8（demo 幂等化）：开跑前清理本租户**业务单据**（保留主数据与种子角色——
# WorkBuddy 复核验证过此清法安全）。否则上一轮 demo 的 confirmed 订单挂着
# 占用、库存被吃空，非首次运行必崩（多轮/CI/多人共用 VM 场景假绿）。
# --keep-data 可跳过清理（想保留演示数据时用，自负库存充足）。
BIZ_TABLES = [
    "inventory", "inventory_transactions",
    "stock_receipts", "stock_receipt_items",
    "stock_issues", "stock_issue_items",
    "sales_orders", "sales_order_items",
    "sales_deliveries", "sales_delivery_items",
    "shipments", "sales_returns",
    "purchase_orders", "purchase_order_items",
    "repair_orders", "repair_parts", "pm_plans",
    "approval_requests",
]


def clean_business_data(md):
    """清 default 租户业务单据（主数据/角色/审计日志保留），幂等可重复跑。"""
    if not md.products.available:
        return
    n = 0
    for t in BIZ_TABLES:
        try:
            md.products._execute(f"DELETE FROM {t} WHERE tenant_id = %s",
                                 (TENANT,))
            n += 1
        except Exception as e:
            print(f"  [清理] {t}: {e}")
    print(f"  [清理] {TENANT} 租户业务单据 {n}/{len(BIZ_TABLES)} 张表已清"
          f"（主数据/角色/审计保留）——demo 幂等保障")

# 身份（biz_roles 角色，与 access_rules.yaml 种子一致）
MEI = {"user": "xiaomei", "role": "sales_user"}          # 销售小美
GM = {"user": "王总", "role": "gm"}                       # 总经理（level 2）
SUPER = {"user": "superadmin", "role": "super_admin"}     # 超管（level 3）
ZHAO = {"user": "赵工", "role": "repair_user"}            # 维修工程师（电气）
DISP = {"user": "调度台", "role": "dept_manager"}         # 维修调度（level 2）
BUYER = {"user": "采购老李", "role": "purchase_user"}


def banner(t):
    print("\n" + "=" * 74)
    print(f"  {t}")
    print("=" * 74)


def who(x):
    return f"{x['user']}({x['role']})"


def main():
    skip_seed = "--skip-seed" in sys.argv

    md = MasterData(verbose=False)
    if "--keep-data" not in sys.argv:
        clean_business_data(md)   # C8：幂等保障（--keep-data 跳过）
    inv = InventoryService(verbose=False)
    sales = SalesService(inv, md, verbose=False)
    repair = RepairService(inv, md, verbose=False)
    purchase = PurchaseService(inv, md, verbose=False)
    gate = ApprovalGate(build_default_executors(md, inv, sales, repair, purchase),
                        verbose=False)

    # ==================================================================
    if not skip_seed:
        banner("第 0 幕 · 种子数据（角色 + 主数据 + 期初库存）")
        n = md.seed_builtin_roles(TENANT)
        print(f"种子角色：装入 {n} 个（已存在则跳过）")
        wh = md.warehouses.list("gm", "seed", tenant_id=TENANT)
        if not wh:
            md.warehouses.create("gm", "seed", TENANT, warehouse_code="WH-01",
                                 name="成品仓")
            wh = md.warehouses.list("gm", "seed", tenant_id=TENANT)
        WH = wh[0]
        prods = md.products.list("gm", "seed", tenant_id=TENANT)
        if not prods:
            md.products.create("gm", "seed", TENANT, product_code="LBL-100",
                               name="自动贴标机", spec="JM-S509", category="finished",
                               unit_price=250000, cost_price=180000, safety_stock=2)
            md.products.create("gm", "seed", TENANT, product_code="SPR-SRV",
                               name="伺服电机备件", spec="750W", category="spare",
                               unit_price=2400, cost_price=1600, safety_stock=5)
            md.products.create("gm", "seed", TENANT, product_code="SEN-T1",
                               name="温度传感器", spec="PT100", category="spare",
                               unit_price=899, cost_price=500, safety_stock=10)
        eqs = md.equipment.list("gm", "seed", tenant_id=TENANT)
        if not eqs:
            # A 类关键设备 + 保修期内（工单自动 urgent + 保内）
            md.equipment.create("gm", "seed", TENANT, equipment_code="JM-S509-L3",
                                name="3号线贴标机", location="1车间/S3线/贴标工位",
                                model="JM-S509", warranty_until="2027-12-31",
                                criticality="A", status="running")
            md.equipment.create("gm", "seed", TENANT, equipment_code="AGV-07",
                                name="AGV 小车 7 号", location="2车间/物流区",
                                criticality="B", status="running")
        engs = md.engineers.list("gm", "seed", tenant_id=TENANT)
        if not engs:
            md.engineers.create("gm", "seed", TENANT, name="赵工",
                                skill="electrical")
            md.engineers.create("gm", "seed", TENANT, name="钱师傅",
                                skill="mechanical")
        custs = md.customers.list("gm", "seed", tenant_id=TENANT)
        if not custs:
            md.customers.create("gm", "seed", TENANT, customer_code="C-001",
                                name="华中医疗器械厂", contact="张主任",
                                credit_limit=500000)
            md.customers.create("gm", "seed", TENANT, customer_code="C-002",
                                name="华东食品集团", credit_limit=300000)
        sups = md.suppliers.list("gm", "seed", tenant_id=TENANT)
        if not sups:
            md.suppliers.create("gm", "seed", TENANT, supplier_code="S-001",
                                name="深圳伺服科技")
        fcs = md.fault_codes.list("gm", "seed", tenant_id=TENANT)
        if not fcs:
            md.fault_codes.create("gm", "seed", TENANT, code="E-SRV-01",
                                  category="electrical", name="伺服过流报警",
                                  standard_solution="检查负载卡滞→测绝缘→更换伺服电机备件（详见手册 7.2 节）",
                                  avg_repair_hours=2.5)
        # 期初库存：按「可用量」补货（可用 = 在库 − 占用）——demo 可重复跑：
        # 此前失败轮次的 confirmed 订单会挂着占用，只看 stock 会导致补货失效
        for code, qty, cost in (("LBL-100", 5, 180000),
                                ("SPR-SRV", 20, 1600),
                                ("SEN-T1", 50, 500)):
            prod = md.products.get_by_code(code, TENANT)
            if not prod:
                continue
            lv = inv.level(TENANT, WH["id"], prod["id"])
            if lv["available"] < 5:      # 可用不足则补一批期初
                rc = inv.create_receipt("gm", "seed", TENANT, WH["id"],
                                        [{"product_id": prod["id"],
                                          "quantity": qty, "unit_cost": cost}],
                                        ref_type="purchase", remark="期初库存")
                inv.receive_receipt(rc["id"], TENANT, operator="seed")
        print(f"期初库存就绪（仓库 {WH['warehouse_code']}）；MySQL 模式：{inv.available}")
    WH = inv.default_warehouse(TENANT, md=md)
    print(f"主仓：{WH['warehouse_code']}（{WH['name']}）")

    # ==================================================================
    banner("第 1 幕 · 采购补货（采购员建单 → 审批 → 收货入库）")
    po = purchase.create_po(BUYER["role"], BUYER["user"], TENANT, "S-001",
                            [{"product_code": "SPR-SRV", "quantity": 10}])
    print(f"✓ 采购单 {po['po_no']}（{po.get('supplier_name')}，金额 {po['amount']}）")
    po = purchase.submit_po(BUYER["role"], BUYER["user"], po["id"], TENANT)
    rid_po = gate.request("purchase_order.approve",
                          {"po_id": po["id"], "amount": float(po["amount"]),
                           "tenant_id": TENANT},
                          reason=f"补货 {po['po_no']}", requested_by=BUYER["user"],
                          requested_role=BUYER["role"], tenant_id=TENANT)
    print(f"⏸ 审批单 #{rid_po}（金额 {po['amount']} 未超 10 万线，经理级可批）")
    r = gate.decide(rid_po, approve=True, decided_by=GM["user"],
                    decided_role=GM["role"], note="同意补货")
    print(f"✓ {who(GM)} 批准 → {r['result']['outcome']}")
    po = purchase.receive_po("warehouse_user", "库管老周", po["id"], TENANT)
    print(f"✓ 收货入库：{po['po_no']} → {po['status']}（SPR-SRV 库存 +10，流水已记）")

    # ==================================================================
    banner("第 2 幕 · 销售流（小美建单 → 确认占用 → 出库审批 → 发运 → 关单）")
    o1 = sales.create_order(MEI["role"], MEI["user"], TENANT, "C-002", [
        {"product_code": "SEN-T1", "quantity": 20},
        {"product_code": "SPR-SRV", "quantity": 2},
    ])
    print(f"✓ 订单 {o1['order_no']}（{o1.get('customer_name')}，金额 {o1['amount']}，draft）")
    o1 = sales.confirm_order(MEI["role"], MEI["user"], o1["id"], TENANT)
    lv = inv.level(TENANT, WH["id"], 3)
    print(f"✓ 已确认（信用通过 + 库存已占用）；传感器可用余量 {lv['available']}")

    d1 = sales.create_delivery(MEI["role"], MEI["user"], o1["id"], TENANT)
    print(f"✓ 出库单 {d1['delivery_no']}（draft，放行须审批）")
    rid_d = gate.request("sales_delivery.release",
                         {"delivery_id": d1["id"], "tenant_id": TENANT},
                         reason="订单确认后首次发货", requested_by="库管老周",
                         requested_role="warehouse_user", tenant_id=TENANT)
    r = gate.decide(rid_d, approve=True, decided_by=GM["user"],
                    decided_role=GM["role"], note="同意发货")
    print(f"✓ 放行审批通过 → 扣库存 + 回写出库量：{r['result']['outcome']}")
    s1 = sales.ship_delivery("warehouse_user", "库管老周", d1["id"],
                             "顺丰速运", TENANT)
    print(f"✓ 已发运：物流单 {s1['shipment_no']}（顺丰速运，运输中）")
    # 大额订单关单演示（金额线）
    o2 = sales.create_order(MEI["role"], MEI["user"], TENANT, "C-001", [
        {"product_code": "LBL-100", "quantity": 1},
    ])
    o2 = sales.confirm_order(MEI["role"], MEI["user"], o2["id"], TENANT)
    d2 = sales.create_delivery(MEI["role"], MEI["user"], o2["id"], TENANT)
    rid2 = gate.request("sales_delivery.release",
                        {"delivery_id": d2["id"], "tenant_id": TENANT},
                        requested_by="库管老周", requested_role="warehouse_user",
                        tenant_id=TENANT)
    gate.decide(rid2, approve=True, decided_by=GM["user"], decided_role=GM["role"])
    sales.ship_delivery("warehouse_user", "库管老周", d2["id"], "德邦物流", TENANT)
    print(f"✓ 大额订单 {o2['order_no']}（{o2['amount']} > 10 万线）已发货")
    rid3 = gate.request("sales_order.complete",
                        {"order_id": o2["id"], "amount": float(o2["amount"]),
                         "tenant_id": TENANT},
                        reason="客户已签收，申请关单", requested_by=MEI["user"],
                        requested_role=MEI["role"], tenant_id=TENANT)
    print(f"⏸ 关单审批 #{rid3}——金额超 10 万，经理级批会被拦：")
    try:
        gate.decide(rid3, approve=True, decided_by=GM["user"],
                    decided_role=GM["role"], note="我批")
    except PermissionError as e:
        print(f"  ⛔ 被金额分级拦下：{e}")
    r = gate.decide(rid3, approve=True, decided_by=SUPER["user"],
                    decided_role=SUPER["role"], note="大额复核通过")
    print(f"✓ {who(SUPER)} 批准 → {r['result']['outcome']}")

    # ==================================================================
    banner("第 3 幕 · 维修流（报修→归类→派单→领料→完工审批→验证）")
    ro = repair.create_order("repair_user", "车间主任刘", TENANT,
                             "JM-S509-L3", "开机自检报伺服过流 F-201，复位复发")
    print(f"✓ 工单 {ro['order_no']}（{ro.get('equipment_code')}，"
          f"优先级 {ro['priority']}（A 类设备自动 urgent），"
          f"{'保内' if ro['warranty'] == 'in' else '保外'}（按台账自动判）")
    ro = repair.classify_fault(DISP["role"], DISP["user"], ro["id"],
                               "E-SRV-01", TENANT)
    print(f"✓ 归类 E-SRV-01（伺服过流报警，SOP 见故障库；工单当前归类代码："
          f"{ro.get('fault_code') or '-'}）")
    ro = repair.assign_technician(DISP["role"], DISP["user"], ro["id"],
                                  "钱师傅", TENANT)
    hint = ro.get("assign_hint")
    print(f"✓ 派单 钱师傅（机械专长）{hint or '——跨工种派单提示见 hint'}")
    ro = repair.assign_technician(DISP["role"], DISP["user"], ro["id"],
                                  "赵工", TENANT)
    print("✓ 改派 赵工（电气，与故障类别 E-SRV-01 匹配，无跨工种提示）")
    ro = repair.start_repair(ZHAO["role"], ZHAO["user"], ro["id"], TENANT)
    print(f"✓ {who(ZHAO)} 开工（in_progress）")
    parts = repair.use_repair_parts(ZHAO["role"], ZHAO["user"], ro["id"],
                                    [{"product_code": "SPR-SRV", "quantity": 1}],
                                    TENANT)
    print(f"✓ 领料 1×SPR-SRV（扣库存记 PARTS_OUT 流水，保内→记保修成本）")
    rid_r = gate.request("repair_order.resolve",
                         {"order_id": ro["id"], "resolution": "更换伺服电机，"
                          "负载卡滞已排除，试机 2h 正常", "downtime_hours": 3.5,
                          "tenant_id": TENANT},
                         reason="维修完成待确认", requested_by=ZHAO["user"],
                         requested_role=ZHAO["role"], tenant_id=TENANT)
    r = gate.decide(rid_r, approve=True, decided_by=DISP["user"],
                    decided_role=DISP["role"], note="试机确认")
    print(f"✓ 完工审批通过 → {r['result']['outcome']}（停机 3.5h 记录在案）")
    ro = repair.verify_repair("gm", "车间主任刘", ro["id"], TENANT)
    print(f"✓ 使用方验证：{ro['order_no']} → {ro['status']}（闭环）")

    # ==================================================================
    banner("第 4 幕 · 隔离演示（越权尝试 → 拒绝 + 审计）")
    for desc, fn in [
        ("销售小美查维修工单", lambda: repair.query_orders(MEI["role"], MEI["user"],
                                                     tenant_id=TENANT)),
        ("采购老李查销售订单", lambda: sales.query_orders(BUYER["role"], BUYER["user"],
                                                     tenant_id=TENANT)),
        ("维修赵工建销售单", lambda: sales.create_order(ZHAO["role"], ZHAO["user"],
                                                   TENANT, "C-001",
                                                   [{"product_code": "SEN-T1",
                                                     "quantity": 1}])),
    ]:
        try:
            fn()
            print(f"  ⚠ {desc}：竟然成功了（不该发生！）")
        except PermissionDenied as e:
            print(f"  ⛔ {desc}：被拒（{e}）")
    print("  （以上三次拒绝均已落 logs/audit.log，result=blocked）")

    # ==================================================================
    banner("第 5 幕 · 库存流水审计（余额是快照，流水才是真相）")
    for t in inv.transactions_of("warehouse_user", "库管老周",
                                  tenant_id=TENANT, limit=15):
        print(f"  [{t['txn_type']:<12}] 产品#{t['product_id']} "
              f"{t['qty']:+d} → 余 {t['balance_after']}"
              f"（{t.get('ref_no') or '-'}，{t.get('operator') or '-'}）")

    banner("演示完毕：单据分离 / 防超卖 / 台账联动 / 分级审批 / 四层隔离 / 流水审计")
    print("人工审批 CLI：python approval.py --list")
    print("岗位与工具：config/employee_profile.yaml + mcp_server.py（19 个 MCP 工具）")


if __name__ == "__main__":
    main()
