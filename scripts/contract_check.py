"""
contract_check.py —— 独立契约检查器（WorkBuddy 视角，不依赖 glm 的 test_factory_system.py）

设计目的：一次性暴露「缺省即静默坏配置 / 静默坏契约」类缺陷，作为持续护栏。
这类缺陷的共同特征：定义时不报错、只在特定运行时路径崩或悄悄行为错误——
打地鼠逐个修永远修不完。本检查器在 import/静态层把整类风险扫出来。

检查维度：
  C1  MasterStore 子类必须声明非空 FIELDS（否则继承基类 [] → create 拒绝所有字段 → 运行时崩，F8 类）
  C2  Store.FIELDS 必须是对应 DDL 表的列子集（防止代码字段与 DDL 漂移，F4 类）
  C3  角色双真源一致性：access_rules.yaml(order_acl) 与 erp_common._SEED_ROLES 必须逐字段一致
  C4  角色 resources 必须是已知资源名（RES_NAMES），防止引用不存在的资源（F2 类）
  C5  角色若拥有「自身创建型资源」，ops 必须含 create（F1 类）
  C6  tenant_id 默认 "default" 的调用点扫描（D7 根因，架构层未治理）
  C7  内存模式空转扫描：各 System 方法内存分支不得 `return []` 静默空操作（F5 类）

退出码：发现 FAIL 级问题返回 1，否则 0（可接 CI 门禁）。
"""
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import erp_common
import master_data

try:
    import yaml
except Exception:
    yaml = None

OK, WARN, FAIL = [], [], []


def _ok(cat, msg):
    OK.append(f"[{cat}] {msg}")


def _warn(cat, msg):
    WARN.append(f"[{cat}] {msg}")


def _fail(cat, msg):
    FAIL.append(f"[{cat}] {msg}")


# ---------------------------------------------------------------- C1 / C2
def _parse_ddl_cols(sql_text):
    cols = {}
    for m in re.finditer(
        r"CREATE TABLE IF NOT EXISTS `?(\w+)`?\s*\((.*?)\)\s*ENGINE",
        sql_text, re.S | re.I,
    ):
        tbl = m.group(1)
        body = m.group(2)
        cset = set()
        for line in body.split("\n"):
            line = line.strip().strip(",")
            cm = re.match(
                r"`?(\w+)`?\s+(?:VARCHAR|INT|BIGINT|TINYINT|TEXT|DATETIME|DATE|"
                r"DECIMAL|JSON|FLOAT|DOUBLE|TIMESTAMP|CHAR)",
                line, re.I,
            )
            if cm:
                cset.add(cm.group(1))
        cols[tbl] = cset
    return cols


ddl_path = os.path.join(ROOT, "config", "init_db.sql")
ddl_text = open(ddl_path, encoding="utf-8").read() if os.path.exists(ddl_path) else ""
ddl_cols = _parse_ddl_cols(ddl_text)

base = master_data._MasterBase
store_count = 0
for name in dir(master_data):
    obj = getattr(master_data, name)
    if isinstance(obj, type) and issubclass(obj, base) and obj is not base:
        store_count += 1
        f = getattr(obj, "FIELDS", None)
        tbl = getattr(obj, "TABLE", "")
        if not (isinstance(f, list) and f):
            _fail("C1-STORE_FIELDS",
                  f"{name}: FIELDS={f!r}（继承基类 [] → create 拒绝所有字段 → 运行时崩，F8 类）")
            continue
        # C2 对齐 DDL
        dcols = ddl_cols.get(tbl)
        if dcols is not None:
            drift = [c for c in f if c not in dcols]
            if drift:
                _fail("C2-FIELDS_DDL_DRIFT",
                      f"{name}: FIELDS 含 DDL({tbl})不存在的列 {drift}（代码/DDL 漂移，F4 类）")
        else:
            _warn("C2-DDL_MISSING", f"{name}: 未在 init_db.sql 找到表 {tbl!r}，无法对齐校验")
        for attr in ("TABLE", "RESOURCE", "CODE_FIELD"):
            if not getattr(obj, attr, ""):
                _fail("C1-STORE_META", f"{name}: {attr} 为空")

# ---------------------------------------------------------------- C3 / C4 / C5
yaml_acl = {}
if yaml is not None and os.path.exists(erp_common.ACCESS_RULES_PATH):
    try:
        doc = yaml.safe_load(open(erp_common.ACCESS_RULES_PATH, encoding="utf-8")) or {}
        yaml_acl = doc.get("order_acl") or {}
    except Exception as e:
        _warn("C3-YAML", f"读取 access_rules.yaml 失败：{e}")

seed = erp_common._SEED_ROLES
all_keys = set(seed) | set(yaml_acl)
res_names = set(erp_common.RES_NAMES)

# 哪些资源要求角色 ops 含 create（自身创建型业务对象）
CREATE_REQUIRED = {
    "sales_order": ["sales_user", "dept_manager", "gm", "super_admin"],
    "repair_order": ["repair_user", "dept_manager", "gm", "super_admin"],
    "purchase_order": ["purchase_user", "dept_manager", "gm", "super_admin"],
    "product": ["gm", "super_admin", "dept_manager"],
    "customer": ["sales_user", "gm", "super_admin"],
    "supplier": ["purchase_user", "gm", "super_admin"],
    "equipment": ["gm", "super_admin"],
    "engineer": ["gm", "super_admin"],
    "warehouse": ["gm", "super_admin"],
    "fault_code": ["gm", "super_admin"],
}

for key in sorted(all_keys):
    s = seed.get(key)
    y = yaml_acl.get(key)
    if s is None or y is None:
        _fail("C3-DUAL_SOURCE",
              f"角色 {key}: 仅存在于 {('yaml' if y else 'code')} 一侧（双侧不一致 → 双真源漂移风险）")
        continue
    for field in ("resources", "ops", "level", "scope"):
        if s.get(field) != y.get(field):
            _fail("C3-DUAL_SOURCE",
                  f"角色 {key}.{field}: code={s.get(field)} ≠ yaml={y.get(field)}（双真源不一致）")
    # C4 资源名合法
    bad_res = [r for r in (y.get("resources") or []) if r not in res_names]
    if bad_res:
        _fail("C4-UNKNOWN_RES",
              f"角色 {key}: resources 含未知资源 {bad_res}（不在 RES_NAMES，F2 类）")
    # C5 create 权限
    for res in (y.get("resources") or []):
        expecters = CREATE_REQUIRED.get(res, [])
        if key in expecters and "create" not in (y.get("ops") or []):
            _fail("C5-MISSING_CREATE",
                  f"角色 {key}: 拥有资源 {res} 但 ops 缺 create（F1 类）")

# ---------------------------------------------------------------- C6 tenant_id 默认 default
hits = []
for py in ("master_data.py", "inventory_system.py", "sales_system.py",
           "repair_system.py", "purchase_system.py", "approval.py", "erp_common.py"):
    p = os.path.join(ROOT, py)
    if not os.path.exists(p):
        continue
    for i, line in enumerate(open(p, encoding="utf-8"), 1):
        if re.search(r"tenant_id\s*=\s*[\"']default[\"']", line):
            hits.append(f"{py}:{i}: {line.strip()}")
if hits:
    _fail("C6-TENANT_DEFAULT",
          f"发现 {len(hits)} 处 tenant_id 默认 'default'（D7 根因，架构层未治理）：")
    for h in hits:
        _fail("C6-TENANT_DEFAULT", "    " + h)
else:
    _ok("C6-TENANT_DEFAULT", "未发现 tenant_id 默认 'default' 调用点")

# ---------------------------------------------------------------- C7 内存模式空转
for py in ("inventory_system.py", "repair_system.py", "sales_system.py", "purchase_system.py"):
    p = os.path.join(ROOT, py)
    if not os.path.exists(p):
        continue
    src = open(p, encoding="utf-8").read()
    # 找形如 def _xxx_items(...) 之类的方法体内 return [] 分支（启发式）
    for m in re.finditer(r"def (\w+)\([^)]*\):\s*(.*?)(?=\n\S|\Z)", src, re.S):
        body = m.group(2)
        if re.search(r"return\s*\[\]", body) and ("内存" in body or "_fb" in body
                                                   or "fallback" in body or "memory" in body):
            _warn("C7-MEM_EMPTY",
                  f"{py}:{m.start()} 方法 {m.group(1)} 内存分支 return []（静默空操作风险，F5 类）")

# ---------------------------------------------------------------- 汇总
print("=" * 72)
print("独立契约检查器（WorkBuddy 视角）结果")
print("=" * 72)
print(f"\n[扫描统计] MasterStore 子类 {store_count} 个；角色 {len(all_keys)} 个")
print(f"\n[OK] {len(OK)}")
for x in OK:
    print("  ✓", x)
print(f"\n[WARN] {len(WARN)}")
for x in WARN:
    print("  !", x)
print(f"\n[FAIL] {len(FAIL)}")
for x in FAIL:
    print("  ✗", x)
print("\n" + "=" * 72)
print(f"结论：FAIL={len(FAIL)}  WARN={len(WARN)}  OK={len(OK)}")
print("=" * 72)
sys.exit(1 if FAIL else 0)
