# -*- coding: utf-8 -*-
"""给 jm / yh 两个租户造**演示用**主数据与采购单（供 Dify 审批流跑通）。

为什么要它：`/ai/pack` 是审批流取数的唯一出口，但库里 `purchase_orders` 只有 2 条
（tenant=test_fac / default），而登记的知识库是 kb-jm / kb-yh —— 拿 kb-jm 查任何
单号都是 404「不存在（tenant=jm）」，业务闭环断在数据上，不是代码上。

造什么（每租户一套，全部带 DEMO 标记，一眼能看出是样例）：
    warehouses      WH-DEMO            演示仓
    suppliers       SUP-DEMO-A/B       两家供应商
    products        P-DEMO-1/2/3       三个商品（含成本价与安全库存）
    inventory       上述商品各一条库存
    purchase_orders PO-DEMO-<T>-001    小额，应走自动批
                    PO-DEMO-<T>-002    大额，应转人工/升级

幂等：按 (tenant_id, code) 或 (tenant_id, po_no) 判存在，重复跑不会重复插。

用法：
    python scripts/seed_demo_tenants.py            # 造数据
    python scripts/seed_demo_tenants.py --dry-run  # 只打印将要插入的内容
"""
import argparse
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TENANTS = ("jm", "yh")

# 商品：(code, 名称, 规格, 单位, 单价, 成本价, 安全库存, 当前库存)
PRODUCTS = [
    ("P-DEMO-1", "演示物料·定位终端", "4G/GPS", "台", 420.00, 380.00, 20, 35),
    ("P-DEMO-2", "演示物料·通讯模组", "Cat.1", "片", 96.00, 82.00, 100, 60),
    ("P-DEMO-3", "演示物料·外壳套件", "ABS", "套", 18.50, 15.00, 200, 480),
]
SUPPLIERS = [
    ("SUP-DEMO-A", "演示供应商甲（长期合作）", 1),
    ("SUP-DEMO-B", "演示供应商乙（新引入）", 1),
]
# 两张单：(后缀, 状态, [(商品序号, 数量, 单价)])
ORDERS = [
    ("001", "submitted", [(0, 10, 420.00), (2, 40, 18.50)]),      # ≈4930 元
    ("002", "submitted", [(0, 300, 420.00), (1, 800, 96.00)]),    # ≈202800 元
]


def load_env():
    env = {}
    with io.open(os.path.join(ROOT, ".env"), encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k, _, v = ln.partition("=")
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import pymysql
    env = load_env()
    conn = pymysql.connect(host=env["MYSQL_HOST"], port=int(env["MYSQL_PORT"]),
                           user=env["MYSQL_USER"], password=env["MYSQL_PASSWORD"],
                           database=env["MYSQL_DATABASE"], charset="utf8mb4",
                           autocommit=False)
    cur = conn.cursor()
    stats = {"warehouse": 0, "supplier": 0, "product": 0, "inventory": 0,
             "po": 0, "items": 0}

    def one(sql, params):
        cur.execute(sql, params)
        return cur.fetchone()

    for t in TENANTS:
        print("== 租户 %s" % t)
        # ---- 仓库 ----
        r = one("select id from warehouses where tenant_id=%s and warehouse_code=%s",
                (t, "WH-DEMO"))
        if r:
            wh = r[0]
        else:
            if args.dry_run:
                wh = None
            else:
                cur.execute("insert into warehouses(tenant_id,warehouse_code,name,"
                            "location,is_active,created_by) values(%s,%s,%s,%s,1,%s)",
                            (t, "WH-DEMO", "演示仓", "演示地址", "seed"))
                wh = cur.lastrowid      # ★ 必须回读，否则下游 inventory.warehouse_id 为 NULL
            stats["warehouse"] += 1
            print("   + 仓库 WH-DEMO  (id=%s)" % wh)

        # ---- 供应商 ----
        sup_ids = {}
        for code, name, active in SUPPLIERS:
            r = one("select id from suppliers where tenant_id=%s and supplier_code=%s",
                    (t, code))
            if r:
                sup_ids[code] = r[0]
            else:
                if not args.dry_run:
                    cur.execute("insert into suppliers(tenant_id,supplier_code,name,"
                                "contact,phone,is_active,created_by) "
                                "values(%s,%s,%s,%s,%s,%s,%s)",
                                (t, code, name, "演示联系人", "13800000000",
                                 active, "seed"))
                    sup_ids[code] = cur.lastrowid
                stats["supplier"] += 1
                print("   + 供应商 %s" % code)

        # ---- 商品 + 库存 ----
        prod_ids = []
        for i, (code, name, spec, unit, price, cost, safety, stock) in enumerate(PRODUCTS, 1):
            r = one("select id from products where tenant_id=%s and product_code=%s",
                    (t, code))
            if r:
                pid = r[0]
            else:
                if not args.dry_run:
                    cur.execute("insert into products(tenant_id,product_code,name,spec,"
                                "category,unit,unit_price,cost_price,safety_stock,"
                                "is_active,created_by) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,1,%s)",
                                (t, code, name, spec, "demo", unit, price, cost,
                                 safety, "seed"))
                    pid = cur.lastrowid
                else:
                    pid = None
                stats["product"] += 1
                print("   + 商品 %s %s" % (code, name))
            prod_ids.append(pid)
            if not one("select id from inventory where tenant_id=%s and warehouse_id=%s "
                       "and product_id=%s", (t, wh, pid)):
                if not args.dry_run:
                    cur.execute("insert into inventory(tenant_id,warehouse_id,product_id,"
                                "stock_qty,reserved_qty) values(%s,%s,%s,%s,0)",
                                (t, wh, pid, stock))
                stats["inventory"] += 1
                print("   + 库存 %s = %d" % (code, stock))

        # ---- 采购单 + 明细 ----
        for suffix, status, lines in ORDERS:
            po_no = "PO-DEMO-%s-%s" % (t.upper(), suffix)
            amount = round(sum(q * p for _, q, p in lines), 2)
            r = one("select id from purchase_orders where tenant_id=%s and po_no=%s",
                    (t, po_no))
            if r:
                print("   = 采购单已存在 %s (id=%s, %.2f 元)" % (po_no, r[0], amount))
                continue
            if not args.dry_run:
                cur.execute("insert into purchase_orders(tenant_id,po_no,supplier_id,"
                            "amount,status,buyer,expected_date,created_by) "
                            "values(%s,%s,%s,%s,%s,%s,%s,%s)",
                            (t, po_no, sup_ids["SUP-DEMO-A" if suffix == "001"
                                               else "SUP-DEMO-B"],
                             amount, status, "演示采购员",
                             "2026-11-15", "seed"))
                po_id = cur.lastrowid
                for pi, qty, price in lines:
                    cur.execute("insert into purchase_order_items(tenant_id,po_id,"
                                "product_id,quantity,unit_price,received_qty) "
                                "values(%s,%s,%s,%s,%s,0)",
                                (t, po_id, prod_ids[pi], qty, price))
                    stats["items"] += 1
            stats["po"] += 1
            print("   + 采购单 %s  %.2f 元  %d 行" % (po_no, amount, len(lines)))

    if args.dry_run:
        print("\n[dry-run] 未写入。预计新增：", stats)
        conn.rollback()
        return 0
    conn.commit()
    print("\n完成：新增 %s" % stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
