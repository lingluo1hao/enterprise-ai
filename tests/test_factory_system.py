# -*- coding: utf-8 -*-
"""
test_factory_system.py — 工厂业务系统自动化测试（v2，修复 D4/D6/D7/D10）
================================================================================

 对应测试案例文档：docs/reports/factory_system_test_cases.md
 执行：
   python tests/test_factory_system.py            # 全部
   python tests/test_factory_system.py T21        # 只跑指定编号（前缀匹配）

 v2 修复（对照 WorkBuddy 测试报告 docs/reports/factory_system_test_result.md）：
   D4  运行前自动清理 test_fac 租户全部业务表（可重复执行，无 1062 残留）
   D6  T17 取数方向修正：取每产品**最新**流水（原字典推导取到了最早一条）
   D7  T24 补齐漏传的 TENANT 实参（create_return）
   D10 段级错误隔离：任一段崩溃只记 CRASH 并继续后续段，全量报告

 双模式：MySQL 可达 → 真库测试；不可达 → 自动内存降级（须 MySQL 的
 案例自动 SKIP 并注明）。数据统一租户 test_fac，不污染 default。
================================================================================
"""

import os
import sys
import py_compile
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL, SKIP = [], [], []
CRASHED_SECTIONS = []
ONLY = sys.argv[1] if len(sys.argv) > 1 else None
TENANT = "test_fac"

# 清理清单：全部业务表（D4；均带 tenant_id 列）
ALL_TABLES = [
    "biz_roles", "customers", "suppliers", "products", "equipment",
    "engineers", "warehouses", "inventory", "inventory_transactions",
    "stock_receipts", "stock_receipt_items", "stock_issues",
    "stock_issue_items", "sales_orders", "sales_order_items",
    "sales_deliveries", "sales_delivery_items", "shipments", "sales_returns",
    "purchase_orders", "purchase_order_items", "fault_codes",
    "repair_orders", "repair_parts", "pm_plans", "approval_requests",
]


def check(tid, name, cond, note=""):
    if ONLY and not tid.startswith(ONLY):
        return
    (PASS if cond else FAIL).append(tid)
    mark = "✓" if cond else "✗ FAIL"
    print(f"  [{tid}] {mark} {name}" + (f"  —— {note}" if note and not cond else ""))


def expect_error(tid, name, fn, contains="", exc=Exception):
    """断言 fn 抛异常且消息含 contains（越权/业务规则类案例）。"""
    if ONLY and not tid.startswith(ONLY):
        return
    try:
        fn()
        check(tid, name, False, "未抛异常（应被拒绝）")
    except exc as e:
        ok = (not contains) or (contains in str(e))
        check(tid, name, ok, f"异常消息不含「{contains}」：{e}")
    except Exception as e:
        check(tid, name, False, f"异常类型 {type(e).__name__} 非 {exc.__name__}：{e}")


def skip(tid, name, why):
    if ONLY and not tid.startswith(ONLY):
        return
    SKIP.append(tid)
    print(f"  [{tid}] - SKIP {name} —— {why}")


SECTIONS = []


def section(name):
    def deco(fn):
        SECTIONS.append((name, fn))
        return fn
    return deco


# ============================================================================
print("=" * 74)
print("  工厂业务系统自动化测试 v2（案例：docs/reports/factory_system_test_cases.md）")
print("=" * 74)

# ---- T01 语法编译（独立于段隔离：编译失败则硬退出） ----
_files = ["erp_common.py", "master_data.py", "inventory_system.py",
          "sales_system.py", "repair_system.py", "purchase_system.py",
          "approval.py", "digital_employee.py", "kb_ops_employee.py",
          "scripts/demo_factory_flow.py"]
_err = None
try:
    for f in _files:
        py_compile.compile(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), f), doraise=True)
except Exception as e:
    _err = str(e)
check("T01", f"py_compile {len(_files)} 个文件全部通过", _err is None, _err or "")
if _err:
    print("语法错误，终止（其余案例无意义）")
    sys.exit(1)

# ---- T02 模块导入 ----
try:
    from master_data import MasterData
    from inventory_system import InventoryService
    from sales_system import SalesService
    from repair_system import RepairService
    from purchase_system import PurchaseService
    from approval import ApprovalGate, build_default_executors
    from erp_common import RoleEngine, PermissionDenied, get_role_engine, RES_NAMES
    check("T02", "六域 + 审批门 + 角色引擎导入成功", True)
except Exception as e:
    check("T02", "模块导入", False, str(e))
    print(traceback.format_exc())
    sys.exit(1)

md = MasterData(verbose=False)
inv = InventoryService(verbose=False)
sales = SalesService(inv, md, verbose=False)
repair = RepairService(inv, md, verbose=False)
purchase = PurchaseService(inv, md, verbose=False)
gate = ApprovalGate(build_default_executors(md, inv, sales, repair, purchase),
                    verbose=False)
MODE = "MySQL" if inv.available else "内存降级"
print(f"  [环境] {MODE} 模式（{'真库断言' if inv.available else '同进程闭环断言'}）")

# ---- D4：清理 test_fac 残留（可重复执行） ----
if inv.available:
    n_del = 0
    for t in ALL_TABLES:
        try:
            md.products._execute(f"DELETE FROM {t} WHERE tenant_id = %s",
                                 (TENANT,))
            n_del += 1
        except Exception as e:
            print(f"  [清理] {t}: {e}")
    print(f"  [清理] test_fac 租户 {n_del}/{len(ALL_TABLES)} 张表已清（幂等可重跑）")

GM, MEI, LI, ZHAO, BUYER = "王总", "小美", "小李", "赵工", "采购老李"

# 公共测试数据（多段共用；各段自建自用则放段内）
WH = None
p_sen = p_spr = None


@section("S1 角色引擎")
def s1():
    rules = get_role_engine().rules_for(TENANT)
    check("T03", f"种子角色加载（≥7 个，实得 {len(rules)}）", len(rules) >= 7,
          f"仅 {list(rules)}")
    check("T04a", "未知角色 fail-closed（rule_of → None）",
          get_role_engine().rule_of("hacker_role", TENANT) is None)
    check("T04b", "未知角色等级为 0（任何审批都过不了）",
          get_role_engine().level_of("hacker_role", TENANT) == 0)
    if not inv.available:
        skip("T05", "自建角色（须 MySQL）", "内存降级模式不适用")
        skip("T06", "角色资源白名单（须 MySQL）", "同上")
        check("T07", "内置角色可用（sales_user 查销售单返回列表）",
              isinstance(sales.query_orders("sales_user", MEI,
                                            tenant_id=TENANT), list))
        return
    r = md.create_biz_role("gm", GM, "qc_inspector", "质检员", 1,
                           ["product", "repair_order"], "tenant", ["query"],
                           TENANT)
    check("T05a", "经理级自建角色成功（qc_inspector）",
          r["role_key"] == "qc_inspector")
    expect_error("T05b", "普通角色（L1）自建角色被拒（防给自己造权限）",
                 lambda: md.create_biz_role("sales_user", MEI, "god_role",
                                            "上帝", 3, list(RES_NAMES), "all",
                                            ["approve"], TENANT),
                 contains="经理级", exc=PermissionDenied)
    expect_error("T06", "自建角色资源白名单（未注册资源拒绝）",
                 lambda: get_role_engine().create_role(
                     "bad_r", "坏角色", 1, ["not_a_resource"], "own", ["query"],
                     TENANT),
                 contains="未注册的资源")
    # D2 回归：建了自建角色后，内置角色必须仍在（非空即不兜底的旧缺陷）
    rules2 = get_role_engine().rules_for(TENANT)
    check("T05c", "D2 回归：自建 qc_inspector 后内置 gm/sales_user 仍在",
          "qc_inspector" in rules2 and "gm" in rules2 and "sales_user" in rules2)
    check("T07", "新角色立即生效（qc_inspector 可查维修工单）",
          isinstance(repair.query_orders("qc_inspector", "QC小张",
                                         tenant_id=TENANT, limit=5), list))


@section("S2 主数据")
def s2():
    global WH, p_sen, p_spr
    if not md.products.get_by_code("LBL-TEST", TENANT):
        md.products.create("gm", GM, TENANT, product_code="LBL-TEST",
                           name="测试贴标机", category="finished",
                           unit_price=250000, cost_price=180000, safety_stock=2)
        md.products.create("gm", GM, TENANT, product_code="SPR-TEST",
                           name="测试伺服备件", category="spare",
                           unit_price=2400, cost_price=1600, safety_stock=5)
        md.products.create("gm", GM, TENANT, product_code="SEN-TEST",
                           name="测试传感器", category="spare",
                           unit_price=899, cost_price=500, safety_stock=10)
    check("T08a", "产品创建（3 个测试品）",
          md.products.get_by_code("LBL-TEST", TENANT) is not None)
    expect_error("T08b", "产品编码唯一约束（重复创建被拒）",
                 lambda: md.products.create("gm", GM, TENANT,
                                            product_code="LBL-TEST", name="重复品"),
                 contains="已存在")
    expect_error("T08c", "非法 category 拒绝（字段白名单校验）",
                 lambda: md.products.create("gm", GM, TENANT,
                                            product_code="X-1", name="坏类别",
                                            category="weapon"),
                 contains="category")
    md.equipment.create("gm", GM, TENANT, equipment_code="EQ-A-WARR",
                        name="A类保内设备", warranty_until="2099-12-31",
                        criticality="A", status="running")
    md.equipment.create("gm", GM, TENANT, equipment_code="EQ-B-OUT",
                        name="B类保外设备", warranty_until="2020-01-01",
                        criticality="B", status="running")
    check("T09", "设备台账（保内 A 类 + 保外 B 类各 1）",
          md.equipment.get_by_code("EQ-A-WARR", TENANT) is not None
          and md.equipment.get_by_code("EQ-B-OUT", TENANT) is not None)
    md.customers.create("gm", GM, TENANT, customer_code="C-TEST1",
                        name="测试客户甲", credit_limit=300000)
    md.customers.create("gm", GM, TENANT, customer_code="C-TEST2",
                        name="测试客户乙", credit_limit=50000)
    check("T10", "客户主数据（信用额度 30 万 / 5 万）",
          float(md.customers.get_by_code("C-TEST1", TENANT)["credit_limit"]) == 300000)
    wh = inv.default_warehouse(TENANT, md=md)
    if not wh:
        md.warehouses.create("gm", GM, TENANT, warehouse_code="WH-TEST",
                             name="测试仓")
        wh = inv.default_warehouse(TENANT, md=md)
    WH = wh["id"]
    p_sen = md.products.get_by_code("SEN-TEST", TENANT)["id"]
    p_spr = md.products.get_by_code("SPR-TEST", TENANT)["id"]


@section("S3 库存域")
def s3():
    rc = inv.create_receipt("warehouse_user", "库管", TENANT, WH, [
        {"product_id": p_sen, "quantity": 100, "unit_cost": 500},
        {"product_id": p_spr, "quantity": 30, "unit_cost": 1600}],
        ref_type="purchase", remark="测试期初")
    rc = inv.receive_receipt(rc["id"], TENANT, operator="库管")
    lv0 = inv.level(TENANT, WH, p_sen)
    check("T11", f"入库：收货后 SEN 在库 100（实得 {lv0['stock_qty']}）"
          f" + PURCHASE_IN 流水",
          lv0["stock_qty"] == 100 and any(
              t["txn_type"] == "PURCHASE_IN" and t["product_id"] == p_sen
              for t in inv.transactions_of("warehouse_user", "库管",
                                           tenant_id=TENANT)))
    inv.reserve(TENANT, WH, p_sen, 20, "sales_order", "SO-T12", "测试")
    lv1 = inv.level(TENANT, WH, p_sen)
    check("T12", f"占用：reserved 20，可用 {lv1['available']}（=100-20）",
          lv1["reserved_qty"] == 20 and lv1["available"] == 80)
    expect_error("T13", "防超卖：占用 81 > 可用 80 被拒",
                 lambda: inv.reserve(TENANT, WH, p_sen, 81, "sales_order",
                                     "SO-T13", "测试"),
                 contains="可用库存不足")
    inv.release(TENANT, WH, p_sen, 10, "sales_order", "SO-T12", "测试")
    lv2 = inv.level(TENANT, WH, p_sen)
    check("T14", f"释放：reserved 回落 10（实得 {lv2['reserved_qty']}）"
          f" + RELEASE 流水",
          lv2["reserved_qty"] == 10 and any(
              t["txn_type"] == "RELEASE" and t["qty"] == -10
              for t in inv.transactions_of("warehouse_user", "库管",
                                           tenant_id=TENANT)))
    inv._stock_out(TENANT, WH, p_sen, 15, "SALE_OUT", "sales_delivery",
                   "SD-T15", "测试", also_release=10)
    lv3 = inv.level(TENANT, WH, p_sen)
    check("T15", f"出库：stock 100→85（实得 {lv3['stock_qty']}），"
          f"占用同步释放 10→0（实得 {lv3['reserved_qty']}）",
          lv3["stock_qty"] == 85 and lv3["reserved_qty"] == 0)
    inv.adjust(TENANT, WH, p_spr, 5, "ADJ-T16", "盘点", "盘盈")
    check("T16", f"盘点调整：SPR 库存 +5（实得 "
          f"{inv.level(TENANT, WH, p_spr)['stock_qty']}，=35）+ ADJUST 流水",
          inv.level(TENANT, WH, p_spr)["stock_qty"] == 35 and any(
              t["txn_type"] == "ADJUST" and t["qty"] == 5
              for t in inv.transactions_of("warehouse_user", "库管",
                                           tenant_id=TENANT)))
    # D6 修正：取每产品「最新」一条流水（transactions_of 返回 id DESC，
    # 第一次遇到的即最新；原实现的字典推导留成了最早一条）
    latest = {}
    for t in inv.transactions_of("warehouse_user", "库管", tenant_id=TENANT,
                                 limit=10 ** 6):
        latest.setdefault(t["product_id"], t)
    ok17 = all(latest[pid]["balance_after"]
               == inv.level(TENANT, WH, pid)["stock_qty"]
               for pid in (p_sen, p_spr) if pid in latest)
    check("T17", "流水余额连续：每产品最新 balance_after == 当前库存", ok17,
          f"最新流水 {[(pid, t['balance_after']) for pid, t in latest.items()]}"
          f" vs 实时 "
          f"{[(pid, inv.level(TENANT, WH, pid)['stock_qty']) for pid in (p_sen, p_spr)]}")


@section("S4 销售域")
def s4():
    o1 = sales.create_order("sales_user", MEI, TENANT, "C-TEST1", [
        {"product_code": "SEN-TEST", "quantity": 10},
        {"product_code": "SPR-TEST", "quantity": 2}])
    check("T18a", f"建单 {o1['order_no']}：amount=10×899+2×2400=13790"
          f"（实得 {o1['amount']}）",
          float(o1["amount"]) == 13790.0)
    if inv.available:   # 改价不影响快照（内存模式改主数据直接生效，只测真库）
        md.products._execute(
            "UPDATE products SET unit_price = 999999 "
            "WHERE product_code = 'SEN-TEST' AND tenant_id = %s", (TENANT,))
        snap = [it for it in sales.get_order(o1["id"], TENANT)["items"]
                if it["product_id"] == p_sen][0]
        check("T18b", "价格快照：主数据改价后订单明细单价仍 899",
              float(snap["unit_price"]) == 899.0)
        md.products._execute(
            "UPDATE products SET unit_price = 899 "
            "WHERE product_code = 'SEN-TEST' AND tenant_id = %s", (TENANT,))
    o1 = sales.confirm_order("sales_user", MEI, o1["id"], TENANT)
    lv_sen = inv.level(TENANT, WH, p_sen)
    check("T19a", f"确认：占用生效（SEN 可用 85-10={lv_sen['available']}）",
          lv_sen["available"] == 75)
    o_big = sales.create_order("sales_user", MEI, TENANT, "C-TEST2", [
        {"product_code": "LBL-TEST", "quantity": 1}])   # 25 万 > 客户乙 5 万额度
    expect_error("T19b", "信用额度拒绝：客户乙 5 万额度下 25 万单确认被拒",
                 lambda: sales.confirm_order("sales_user", MEI, o_big["id"],
                                             TENANT),
                 contains="信用额度")
    d1 = sales.create_delivery("sales_user", MEI, o1["id"], TENANT)
    check("T20a", f"出库单 {d1['delivery_no']}（默认全量未出）",
          len(d1["items"]) == 2)
    o1_items = sales._order_items(o1["id"], TENANT)
    expect_error("T20b", "分批校验：出库量超未出量被拒",
                 lambda: sales.create_delivery(
                     "sales_user", MEI, o1["id"], TENANT,
                     lines=[{"order_item_id": o1_items[0]["id"],
                             "quantity": 999}]),
                 contains="出库量非法")
    d1 = sales.execute_release(d1["id"], TENANT, GM)
    o1r = sales.get_order(o1["id"], TENANT)
    snap = [it for it in o1r["items"] if it["product_id"] == p_sen][0]
    lv_sen2 = inv.level(TENANT, WH, p_sen)
    check("T21", f"放行：扣库存（85→{lv_sen2['stock_qty']}）+ delivered_qty 回写"
          f"（SEN 行 {snap['delivered_qty']}/10）+ 订单 {o1r['status']}",
          lv_sen2["stock_qty"] == 75 and snap["delivered_qty"] == 10
          and o1r["status"] == "delivering")
    o2 = sales.create_order("sales_user", LI, TENANT, "C-TEST1", [
        {"product_code": "SEN-TEST", "quantity": 5}])
    o2 = sales.confirm_order("sales_user", LI, o2["id"], TENANT)
    d2 = sales.create_delivery(
        "sales_user", LI, o2["id"], TENANT,
        lines=[{"order_item_id": sales._order_items(o2["id"], TENANT)[0]["id"],
                "quantity": 2}])
    sales.execute_release(d2["id"], TENANT, GM)
    expect_error("T22", "未出完不可关单（2/5 已出）",
                 lambda: sales.execute_complete(o2["id"], TENANT, GM),
                 contains="未出完")
    d3 = sales.create_delivery("sales_user", LI, o2["id"], TENANT)
    sales.execute_release(d3["id"], TENANT, GM)
    o2c = sales.execute_complete(o2["id"], TENANT, GM)
    check("T22b", f"全部出完后可关单（{o2c['status']}）",
          o2c["status"] == "completed")
    o3 = sales.create_order("sales_user", MEI, TENANT, "C-TEST1", [
        {"product_code": "SPR-TEST", "quantity": 3}])
    sales.confirm_order("sales_user", MEI, o3["id"], TENANT)
    lv_before = inv.level(TENANT, WH, p_spr)
    o3x = sales.execute_cancel(o3["id"], TENANT, GM)
    lv_after = inv.level(TENANT, WH, p_spr)
    check("T23", f"取消释放占用：SPR reserved {lv_before['reserved_qty']}→"
          f"{lv_after['reserved_qty']}（订单 {o3x['status']}）",
          lv_after["reserved_qty"] == lv_before["reserved_qty"] - 3)
    # D7 修复：create_return 显式传 TENANT
    rt = sales.create_return("sales_user", MEI, o1["id"], "SEN-TEST", 3,
                             "质量问题", TENANT)
    rt = sales.receive_return("gm", GM, rt["id"], "ok", TENANT)
    # F6 修正：T21 后 75，T22b 的 o2(SEN×5) 已全量出库扣 5 → 70，退货 +3 = 73
    check("T24a", f"退货质检 ok：回补库存（SEN → "
          f"{inv.level(TENANT, WH, p_sen)['stock_qty']}，=70+3）",
          inv.level(TENANT, WH, p_sen)["stock_qty"] == 73)
    expect_error("T24b", "退货量 > 已出量被拒",
                 lambda: sales.create_return("sales_user", MEI, o1["id"],
                                             "SEN-TEST", 999, "贪心", TENANT),
                 contains="退货量非法")


@section("S5 维修域")
def s5():
    ro = repair.create_order("repair_user", "车间刘", TENANT, "EQ-A-WARR",
                             "A类保内设备故障")
    check("T25a", f"A 类保内：priority={ro['priority']} warranty={ro['warranty']}",
          ro["priority"] == "urgent" and ro["warranty"] == "in")
    ro2 = repair.create_order("repair_user", "车间刘", TENANT, "EQ-B-OUT",
                              "B类保外设备故障")
    check("T25b", f"B 类保外：priority={ro2['priority']} "
          f"warranty={ro2['warranty']}",
          ro2["priority"] == "normal" and ro2["warranty"] == "out")
    expect_error("T25c", "台账外设备报修被拒",
                 lambda: repair.create_order("repair_user", "车间刘", TENANT,
                                             "NO-SUCH-EQ", "幽灵设备"),
                 contains="不在台账")
    md.fault_codes.create("gm", GM, TENANT, code="E-TST-01",
                          category="electrical", name="测试电气故障",
                          standard_solution="换件")
    ro = repair.classify_fault("dept_manager", "调度", ro["id"], "E-TST-01",
                               TENANT)
    check("T26a", "故障归类（fault_code_id 落库）",
          ro.get("fault_code_id") is not None)
    md.engineers.create("gm", GM, TENANT, name="测试钱师傅", skill="mechanical")
    md.engineers.create("gm", GM, TENANT, name="测试赵工", skill="electrical")
    ro = repair.assign_technician("dept_manager", "调度", ro["id"],
                                  "测试钱师傅", TENANT)
    check("T26b", "跨工种派单带提示（电气故障派机械师傅）",
          "跨工种" in (ro.get("assign_hint") or ""))
    ro = repair.assign_technician("dept_manager", "调度", ro["id"],
                                  "测试赵工", TENANT)
    check("T26c", "改派匹配无提示", not ro.get("assign_hint"))
    repair.start_repair("repair_user", "测试赵工", ro["id"], TENANT)
    repair.use_repair_parts("repair_user", "测试赵工", ro["id"],
                            [{"product_code": "SPR-TEST", "quantity": 2}],
                            TENANT)
    lv_spr = inv.level(TENANT, WH, p_spr)
    # 账目：入库30 → T16盘+5=35 → T21放行出SPR×2=33 → 本次领料×2 → **31**
    check("T27a", f"领料扣库存（SPR → {lv_spr['stock_qty']}，=33-2=31）"
          f"+ PARTS_OUT 流水",
          lv_spr["stock_qty"] == 31 and any(
              t["txn_type"] == "PARTS_OUT" and t["qty"] == -2
              for t in inv.transactions_of("warehouse_user", "库管",
                                           tenant_id=TENANT)))
    expect_error("T27b", "领料超库存被拒",
                 lambda: repair.use_repair_parts(
                     "repair_user", "测试赵工", ro["id"],
                     [{"product_code": "SPR-TEST", "quantity": 9999}], TENANT),
                 contains="库存不足")
    rov = repair.execute_resolve(ro["id"], TENANT, GM,
                                 resolution="更换伺服备件", downtime_hours=2.0)
    check("T28a", f"完工：{rov['status']}，结论与停时落库",
          rov["status"] == "resolved" and rov["resolution"] == "更换伺服备件"
          and float(rov["downtime_hours"]) == 2.0)
    expect_error("T28b", "重复完工拒绝（已 resolved）",
                 lambda: repair.execute_resolve(ro["id"], TENANT, GM, "再来一次"),
                 contains="状态")
    rov = repair.verify_repair("gm", "车间刘", ro["id"], TENANT)
    check("T29", f"使用方验证：{rov['status']}", rov["status"] == "verified")


@section("S6 采购域")
def s6():
    md.suppliers.create("gm", GM, TENANT, supplier_code="S-TEST",
                        name="测试供应商")
    po = purchase.create_po("purchase_user", BUYER, TENANT, "S-TEST", [
        {"product_code": "SPR-TEST", "quantity": 10}])
    check("T30a", f"采购单（cost_price 快照 1600×10=16000，实得 {po['amount']}）",
          float(po["amount"]) == 16000.0)
    po = purchase.submit_po("purchase_user", BUYER, po["id"], TENANT)
    expect_error("T30b", "未审批不可收货",
                 lambda: purchase.receive_po("warehouse_user", "库管", po["id"],
                                             TENANT),
                 contains="审批通过")
    purchase.execute_approve(po["id"], TENANT, GM)
    po = purchase.receive_po("warehouse_user", "库管", po["id"], TENANT)
    lv_spr2 = inv.level(TENANT, WH, p_spr)
    # 账目：T27a 后 31 → 采购收货 +10 → **41**
    check("T30c", f"收货入库：SPR 31→{lv_spr2['stock_qty']}（=41）+ 单据 "
          f"{po['status']}",
          lv_spr2["stock_qty"] == 41 and po["status"] == "received")


@section("S7 审批门")
def s7():
    # F7 修复：大额关单场景须用 LBL-TEST（25 万 > 10 万线），先备货入库
    p_lbl = md.products.get_by_code("LBL-TEST", TENANT)["id"]
    rc_lbl = inv.create_receipt("warehouse_user", "库管", TENANT, WH,
                                [{"product_id": p_lbl, "quantity": 2,
                                  "unit_cost": 180000}],
                                ref_type="purchase", remark="S7 大额场景备货")
    inv.receive_receipt(rc_lbl["id"], TENANT, operator="库管")
    o_ap = sales.create_order("sales_user", MEI, TENANT, "C-TEST1", [
        {"product_code": "SEN-TEST", "quantity": 1}])
    sales.confirm_order("sales_user", MEI, o_ap["id"], TENANT)
    d_ap = sales.create_delivery("sales_user", MEI, o_ap["id"], TENANT)
    rid = gate.request("sales_delivery.release",
                       {"delivery_id": d_ap["id"], "tenant_id": TENANT},
                       reason="测试", requested_by="库管",
                       requested_role="warehouse_user", tenant_id=TENANT)
    expect_error("T31", "审批等级不足：L1 角色批放行被拒",
                 lambda: gate.decide(rid, approve=True, decided_by="小美",
                                     decided_role="sales_user"),
                 contains="等级不足", exc=PermissionError)
    expect_error("T32", "自审自批被拒（发起人=审批人）",
                 lambda: gate.decide(rid, approve=True, decided_by="库管",
                                     decided_role="gm"),
                 contains="自审自批", exc=PermissionError)
    o_big2 = sales.create_order("sales_user", MEI, TENANT, "C-TEST1", [
        {"product_code": "LBL-TEST", "quantity": 1}])       # 25 万 > 10 万线
    sales.confirm_order("sales_user", MEI, o_big2["id"], TENANT)
    d_big = sales.create_delivery("sales_user", MEI, o_big2["id"], TENANT)
    rid_big_d = gate.request("sales_delivery.release",
                             {"delivery_id": d_big["id"], "tenant_id": TENANT},
                             requested_by="库管",
                             requested_role="warehouse_user", tenant_id=TENANT)
    gate.decide(rid_big_d, approve=True, decided_by=GM, decided_role="gm")
    sales.ship_delivery("warehouse_user", "库管", d_big["id"], "测试承运",
                        TENANT)
    rid_big = gate.request("sales_order.complete",
                           {"order_id": o_big2["id"],
                            "amount": float(o_big2["amount"]),
                            "tenant_id": TENANT},
                           requested_by=MEI, requested_role="sales_user",
                           tenant_id=TENANT)
    expect_error("T33a", "金额分级：25 万关单 L2（gm）审批被拒",
                 lambda: gate.decide(rid_big, approve=True, decided_by=GM,
                                     decided_role="gm"),
                 contains="等级不足", exc=PermissionError)
    r_big = gate.decide(rid_big, approve=True, decided_by="超管",
                        decided_role="super_admin", note="大额复核")
    check("T33b", f"金额分级：L3（super_admin）批准通过 → "
          f"{r_big['result']['outcome']}",
          r_big["status"] == "approved" and r_big["result"]["ok"] is True)
    o_rej = sales.create_order("sales_user", LI, TENANT, "C-TEST1", [
        {"product_code": "SEN-TEST", "quantity": 1}])
    sales.confirm_order("sales_user", LI, o_rej["id"], TENANT)
    rid_rej = gate.request("sales_order.cancel",
                           {"order_id": o_rej["id"], "tenant_id": TENANT},
                           requested_by=LI, requested_role="sales_user",
                           tenant_id=TENANT)
    r_rej = gate.decide(rid_rej, approve=False, decided_by=GM,
                        decided_role="gm", note="证据不足")
    o_rej2 = sales.get_order(o_rej["id"], TENANT)
    check("T34", f"拒绝路径：订单保持 {o_rej2['status']}（未被取消）",
          o_rej2["status"] == "confirmed" and r_rej["result"]["rejected"] is True)
    expect_error("T35", "重复审批拒绝（已决单据）",
                 lambda: gate.decide(rid_rej, approve=True, decided_by=GM,
                                     decided_role="gm"),
                 contains="已处理")
    rid_bad = gate.request("sales_order.complete",
                           {"order_id": 999999, "tenant_id": TENANT},
                           requested_by=MEI, requested_role="sales_user",
                           tenant_id=TENANT)
    r_bad = gate.decide(rid_bad, approve=True, decided_by=GM,
                        decided_role="gm")
    check("T36", "executor 失败如实回填（result.ok=False，不装成功）",
          r_bad["status"] == "approved" and r_bad["result"]["ok"] is False)


@section("S8 四层隔离")
def s8():
    expect_error("T37", "L2 资源隔离：销售查维修工单被拒",
                 lambda: repair.query_orders("sales_user", MEI, tenant_id=TENANT),
                 contains="无权访问", exc=PermissionDenied)
    expect_error("T38", "资源隔离：采购查销售单被拒",
                 lambda: sales.query_orders("purchase_user", BUYER,
                                            tenant_id=TENANT),
                 contains="无权访问", exc=PermissionDenied)
    expect_error("T39", "操作隔离：维修工程师建销售单被拒",
                 lambda: sales.create_order("repair_user", ZHAO, TENANT,
                                            "C-TEST1",
                                            [{"product_code": "SEN-TEST",
                                              "quantity": 1}]),
                 contains="无权", exc=PermissionDenied)
    _audit = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "logs", "audit.log")
    _blocked_seen = False
    if os.path.isfile(_audit):
        with open(_audit, "r", encoding="utf-8", errors="ignore") as f:
            tail = f.readlines()[-200:]
        _blocked_seen = any(
            '"result": "blocked"' in ln or '"result":"blocked"' in ln
            for ln in tail)
    check("T40", "越权审计落盘（logs/audit.log 尾部含 blocked 记录）",
          _blocked_seen)
    rows_mei = sales.query_orders("sales_user", MEI, tenant_id=TENANT, limit=100)
    rows_li = sales.query_orders("sales_user", LI, tenant_id=TENANT, limit=100)
    check("T41", f"own 范围：小美 {len(rows_mei)} 单不含小李的，"
          f"小李 {len(rows_li)} 单不含小美的",
          all(r.get("created_by") != LI for r in rows_mei)
          and all(r.get("created_by") != MEI for r in rows_li))


@section("S9 数字员工层（环境依赖项）")
def s9():
    try:
        from digital_employee import EmployeeProfile, ReviewGate

        class _FakeVDB:
            def __init__(self, hits, dist):
                self.hits, self.dist = hits, dist

            def similarity_search_with_score(self, q, k=4, **kwargs):
                # kwargs：filter_role/user_id/tenant_id（与真实签名兼容）
                return [("doc", self.dist)] * self.hits

        prof = EmployeeProfile.load("researcher")
        gate42 = ReviewGate(prof)
        v_rej = gate42.review("目标", "未找到相关资料。", _FakeVDB(0, 0.9))
        v_deg = gate42.review("目标", "这是一个足够长的正常答案。" * 5,
                              _FakeVDB(3, 0.9))
        v_ok = gate42.review("目标", "这是一个足够长的正常答案。" * 5,
                             _FakeVDB(3, 0.3))
        # T-D2：距离失真（负数，实测 -0.0164）→ distance_valid=False →
        # 文本兜底：问题与垃圾片段（"doc"）相似度低 → low_quality（KB 污染场景）
        v_dis = gate42.review("企业级数字员工的设计原则", "这是一段无关的长答案。" * 5,
                              _FakeVDB(1, -0.0164))
        check("T42a", "岗位档案加载（researcher）", prof.name == "资料研究员")
        check("T42b", f"验收门三态：查空→{v_rej['verdict']} "
              f"远距→{v_deg['verdict']} 正常→{v_ok['verdict']}",
              v_rej["verdict"] == "rejected"
              and v_deg["verdict"] == "low_quality"
              and v_ok["verdict"] == "passed")
        check("T42c", f"T-D2 距离失真兜底：负距离→{v_dis['verdict']}"
              f"（distance_valid={v_dis['signals'].get('distance_valid')}，"
              f"goal_doc_sim={v_dis['signals'].get('goal_doc_sim')}）",
              v_dis["verdict"] == "low_quality"
              and v_dis["signals"].get("distance_valid") is False)
    except ImportError as e:
        skip("T42", "验收门（ReviewGate 三态）",
             f"advanced_rag_agent 依赖缺失：{e}")
    try:
        import kb_ops_employee  # noqa
        check("T43", "知识库运营专员模块导入成功", True)
    except ImportError as e:
        skip("T43", "kb_ops_employee 导入", f"依赖缺失：{e}")
    try:
        import mcp_server  # noqa
        check("T44", "MCP Server 导入成功（19 个业务工具注册）", True)
    except ImportError as e:
        skip("T44", "mcp_server 导入（fastmcp）", f"依赖缺失：{e}")


# ============================================================================
# D10：段级隔离执行（任一段崩溃记 CRASH 并继续）
for _name, _fn in SECTIONS:
    print(f"\n—— {_name} ——")
    try:
        _fn()
    except Exception:
        CRASHED_SECTIONS.append(_name)
        FAIL.append(f"{_name}(段崩溃)")
        print(f"  ✗ 段级崩溃（已隔离，继续后续段）：")
        print("    " + traceback.format_exc(limit=4).replace("\n", "\n    "))

print("\n" + "=" * 74)
print(f"  汇总：PASS {len(PASS)} ｜ FAIL {len(FAIL)} ｜ SKIP {len(SKIP)}"
      f" ｜ 段崩溃 {len(CRASHED_SECTIONS)}（模式：{MODE}）")
if FAIL:
    print(f"  失败清单：{FAIL}")
print("=" * 74)
print("  T45（端到端 demo，default 租户）请单独执行："
      "python scripts/demo_factory_flow.py")
print("  T46（审批 CLI 冒烟）请单独执行：python approval.py --list")
sys.exit(1 if FAIL else 0)
