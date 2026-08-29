"""
================================================================================
 master_data.py — 主数据域（客户/供应商/产品/设备/工程师/仓库）+ 角色管理
================================================================================

 一切单据的字典源（工厂业务系统设计 §3.1）：
   customers 客户（信用额度） / suppliers 供应商 / products 产品与备件
   （统一主数据，含安全库存）/ equipment 设备台账（保修+关键度+层级位置）
   engineers 工程师（技能派单）/ warehouses 仓库

 角色管理（已定稿「系统自建角色」）：
   MasterData.create_biz_role / list_biz_roles / seed_builtin_roles
   —— 工厂按需创建车间主任/质检员/计划员，不改代码。

 所有写操作过 L3 操作矩阵 + 越权审计（erp_common 四层隔离）。
 表结构权威源 config/init_db.sql（表 9-15）。
================================================================================
"""

import json
import argparse
from typing import Dict, List, Optional

from erp_common import ErpDb, PermissionDenied, get_role_engine, RES_NAMES

# 业务校验取值域
PRODUCT_CATEGORIES = ("finished", "spare", "material")
EQUIP_CRITICALITY = ("A", "B", "C")
ENGINEER_SKILLS = ("electrical", "mechanical", "software")


class _MasterBase(ErpDb):
    """
    主数据通用 Store：create（字段白名单 + 编码唯一）/ get_by_code / list。

    子类声明 FIELDS（合法字段白名单，防任意列注入）与 CODE_FIELD。
    """

    CODE_FIELD = ""
    FIELDS: List[str] = []

    def __init_subclass__(cls, **kwargs):
        """
        fail-fast 契约护栏（T-02，根治 F8 类缺陷）：
        子类漏声明非空 FIELDS 时在 **import 阶段即 TypeError**——
        「定义态静默坏配置」不再流到运行时让测试/用户发现。
        """
        super().__init_subclass__(**kwargs)
        if not (isinstance(getattr(cls, "FIELDS", None), list) and cls.FIELDS):
            raise TypeError(
                f"{cls.__name__} 必须声明非空 FIELDS"
                f"（防继承基类 [] 的静默坏配置——F8 教训）")
        if not getattr(cls, "CODE_FIELD", ""):
            raise TypeError(f"{cls.__name__} 必须声明 CODE_FIELD")
        if not getattr(cls, "CODE_FIELD", "") in cls.FIELDS:
            raise TypeError(
                f"{cls.__name__}.CODE_FIELD「{cls.CODE_FIELD}」不在 FIELDS 内"
                f"（编码字段必须是可写白名单成员）")

    def __init__(self, ensure_schema: bool = True, verbose: bool = True):
        super().__init__(ensure_schema, verbose)
        if self.available:
            self._ensure_is_active_column()

    def _ensure_is_active_column(self):
        """
        幂等列迁移（F4 修复）：equipment / fault_codes 旧表缺 is_active 列
        （list 的 active_only 过滤需要），缺则 ALTER 补列。
        权威源 init_db.sql 已含该列；此迁移保证已建过旧表的环境免手工 DDL。
        """
        if not self.TABLE:
            return
        try:
            rows = self._execute(
                "SELECT COUNT(*) AS n FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s "
                "AND COLUMN_NAME = 'is_active'", (self.TABLE,))
            if rows and rows[0]["n"] == 0:
                self._execute(
                    f"ALTER TABLE `{self.TABLE}` ADD COLUMN `is_active` "
                    "TINYINT NOT NULL DEFAULT 1 COMMENT '是否启用（主数据通用列）'")
                if self.verbose:
                    print(f"  [{type(self).__name__}] 已补 is_active 列（幂等迁移）")
        except Exception as e:
            if self.verbose:
                print(f"  [{type(self).__name__}] is_active 列检查失败：{e}")

    def _validate(self, fields: Dict) -> Dict:
        """子类覆写：业务取值域校验（非法值抛 ValueError）。"""
        return fields

    def create(self, acting_role: str, acting_user: str,
               tenant_id: str = "default", **fields) -> Dict:
        """创建主数据记录（L3 校验 create 权 + 字段白名单 + 编码唯一）。"""
        self._check_op("create", acting_role, acting_user, tenant_id)
        bad = [k for k in fields if k not in self.FIELDS]
        if bad:
            raise ValueError(f"非法字段 {bad}，合法字段：{self.FIELDS}")
        fields = self._validate(fields)
        code = (fields.get(self.CODE_FIELD) or "").strip()
        if not code:
            raise ValueError(f"{self.CODE_FIELD} 不能为空")
        if self.get_by_code(code, tenant_id):
            raise ValueError(f"{self.CODE_FIELD}「{code}」已存在")
        cols = ["tenant_id", "created_by"] + list(fields.keys())
        vals = [tenant_id, acting_user] + list(fields.values())
        if self.available:
            ph = ", ".join(["%s"] * len(cols))
            self._execute(
                f"INSERT INTO {self.TABLE} ({', '.join(cols)}) VALUES ({ph})",
                tuple(vals))
            return self.get_by_code(code, tenant_id)
        self._fallback_seq += 1
        row = {"id": self._fallback_seq, "tenant_id": tenant_id,
               "created_by": acting_user, **fields}
        self._fallback_rows.append(row)
        return dict(row)

    def get_by_code(self, code: str, tenant_id: str = "default") -> Optional[Dict]:
        """按编码取（租户内唯一）。"""
        if self.available:
            rows = self._execute(
                f"SELECT * FROM {self.TABLE} WHERE tenant_id = %s "
                f"AND {self.CODE_FIELD} = %s", (tenant_id, code))
            return rows[0] if rows else None
        for r in self._fallback_rows:
            if r.get(self.CODE_FIELD) == code and r["tenant_id"] == tenant_id:
                return dict(r)
        return None

    def get(self, rid, tenant_id: str = None) -> Optional[Dict]:
        if self.available:
            sql, params = f"SELECT * FROM {self.TABLE} WHERE id = %s", (rid,)
            if tenant_id:
                sql += " AND tenant_id = %s"
                params = (rid, tenant_id)
            rows = self._execute(sql, params)
            return rows[0] if rows else None
        for r in self._fallback_rows:
            if r["id"] == rid and (not tenant_id or r["tenant_id"] == tenant_id):
                return dict(r)
        return None

    def list(self, acting_role: str, acting_user: str,
             tenant_id: str = "default", active_only: bool = True,
             limit: int = 100) -> List[Dict]:
        """用户级列表（L3 校验 query 权 + 行级裁剪）。"""
        self._check_op("query", acting_role, acting_user, tenant_id)
        return self._list_for_system(tenant_id, active_only, limit)

    def _list_for_system(self, tenant_id: str = "default",
                         active_only: bool = True,
                         limit: int = 100) -> List[Dict]:
        """
        系统内部联动查询（F9 修复）：调用方是 Service 业务逻辑（如
        default_warehouse 查主仓、assign_technician 查工程师在册名单），
        **不是人**，因此不经用户 ACL——此前内部联动冒充硬编码用户身份
        （"warehouse_user"/"repair_user"），身份无对应资源即崩（F9 教训）。
        用户级查询必须走 list()（过四层隔离）；主数据字典本身非敏感资源。
        """
        if self.available:
            sql, params = f"SELECT * FROM {self.TABLE} WHERE tenant_id = %s", [tenant_id]
            if active_only:
                sql += " AND is_active = 1"
            sql += " ORDER BY id DESC LIMIT %s"
            params.append(limit)
            return self._execute(sql, tuple(params)) or []
        rows = [dict(r) for r in self._fallback_rows
                if r["tenant_id"] == tenant_id
                and (not active_only or r.get("is_active", 1) == 1)]
        return list(reversed(rows))[:limit]


class CustomerStore(_MasterBase):
    TABLE, RESOURCE, CODE_FIELD = "customers", "customer", "customer_code"
    FIELDS = ["customer_code", "name", "contact", "phone", "address",
              "credit_limit", "is_active"]


class SupplierStore(_MasterBase):
    TABLE, RESOURCE, CODE_FIELD = "suppliers", "supplier", "supplier_code"
    FIELDS = ["supplier_code", "name", "contact", "phone", "is_active"]


class ProductStore(_MasterBase):
    TABLE, RESOURCE, CODE_FIELD = "products", "product", "product_code"
    FIELDS = ["product_code", "name", "spec", "category", "unit",
              "unit_price", "cost_price", "safety_stock", "is_active"]

    def _validate(self, fields: Dict) -> Dict:
        if fields.get("category") and fields["category"] not in PRODUCT_CATEGORIES:
            raise ValueError(f"category 取值 {PRODUCT_CATEGORIES}")
        if fields.get("unit_price") is not None and float(fields["unit_price"]) < 0:
            raise ValueError("unit_price 不能为负")
        return fields


class EquipmentStore(_MasterBase):
    """设备台账（CMMS asset register）——保内/保外与关键度的判定源。"""
    TABLE, RESOURCE, CODE_FIELD = "equipment", "equipment", "equipment_code"
    FIELDS = ["equipment_code", "name", "location", "model", "manufacturer",
              "installed_date", "warranty_until", "criticality", "status",
              "owner_dept", "is_active"]

    def _validate(self, fields: Dict) -> Dict:
        if fields.get("criticality") and fields["criticality"] not in EQUIP_CRITICALITY:
            raise ValueError(f"criticality 取值 {EQUIP_CRITICALITY}（A=关键）")
        if fields.get("status") and fields["status"] not in ("running", "down", "retired"):
            raise ValueError("status 取值 running/down/retired")
        return fields


class EngineerStore(_MasterBase):
    # F8 修复：与 engineers 表 DDL 对齐（name/skill/phone/is_active）。
    # 注：WorkBuddy 建议里的 contact 列在 DDL 中不存在（C2 会抓 FIELDS ⊄ DDL），
    # 以 DDL 为准——如需联系人字段应先改 DDL 再加此处。
    TABLE, RESOURCE, CODE_FIELD = "engineers", "engineer", "name"
    FIELDS = ["name", "skill", "phone", "is_active"]

    def _validate(self, fields: Dict) -> Dict:
        if fields.get("skill") and fields["skill"] not in ENGINEER_SKILLS:
            raise ValueError(f"skill 取值 {ENGINEER_SKILLS}")
        return fields


class WarehouseStore(_MasterBase):
    TABLE, RESOURCE, CODE_FIELD = "warehouses", "warehouse", "warehouse_code"
    FIELDS = ["warehouse_code", "name", "location", "is_active"]


class FaultCodeStore(_MasterBase):
    """故障代码库——standard_solution 与 RAG 知识库联动的锚点。"""
    TABLE, RESOURCE, CODE_FIELD = "fault_codes", "fault_code", "code"
    FIELDS = ["code", "category", "name", "standard_solution",
              "avg_repair_hours", "is_active"]

    def _validate(self, fields: Dict) -> Dict:
        if fields.get("category") and fields["category"] not in (
                "mechanical", "electrical", "software", "hydraulic"):
            raise ValueError("category 取值 mechanical/electrical/software/hydraulic")
        return fields


class MasterData:
    """门面：一次初始化持有全部主数据 Store + 角色管理（demo/MCP/Service 共用）。"""

    def __init__(self, verbose: bool = True):
        self.customers = CustomerStore(verbose=verbose)
        self.suppliers = SupplierStore(verbose=verbose)
        self.products = ProductStore(verbose=verbose)
        self.equipment = EquipmentStore(verbose=verbose)
        self.engineers = EngineerStore(verbose=verbose)
        self.warehouses = WarehouseStore(verbose=verbose)
        self.fault_codes = FaultCodeStore(verbose=verbose)
        self.roles = get_role_engine()

    # ------------------------------------------------------------------
    # 角色管理（已定稿：系统自建角色）
    # ------------------------------------------------------------------
    def create_biz_role(self, acting_role: str, acting_user: str,
                        role_key: str, name: str, level: int,
                        resources: List[str], scope: str, ops: List[str],
                        tenant_id: str = "default") -> Dict:
        """
        自建业务角色（仅 level≥2 的角色可操作——防止普通用户给自己造权限）。
        """
        actor_level = self.roles.level_of(acting_role, tenant_id)
        if actor_level < 2:
            from erp_common import audit
            audit(acting_user, "biz_role.create", result="blocked",
                  detail=f"角色「{acting_role}」等级不足（{actor_level}），"
                         "自建角色须经理级及以上")
            raise PermissionDenied(
                f"角色「{acting_role}」无权自建角色（须经理级及以上，越权已审计）")
        return self.roles.create_role(role_key, name, level, resources, scope,
                                      ops, tenant_id=tenant_id,
                                      created_by=acting_user)

    def list_biz_roles(self, acting_role: str, acting_user: str,
                       tenant_id: str = "default") -> List[Dict]:
        """列出本租户全部业务角色（含 resources/scope/ops，供前端配角色用）。"""
        actor_level = self.roles.level_of(acting_role, tenant_id)
        if actor_level < 1:
            raise PermissionDenied(f"未知角色「{acting_role}」")
        rules = self.roles.rules_for(tenant_id)
        return [{"role_key": k, **v} for k, v in sorted(rules.items())]

    def seed_builtin_roles(self, tenant_id: str = "default") -> int:
        """把 YAML 种子角色装入 biz_roles（初始化/演示用，已存在跳过）。"""
        return self.roles.seed_builtin(tenant_id)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="主数据管理（客户/产品/设备/工程师/角色…）")
    parser.add_argument("entity", choices=["customers", "products", "equipment",
                                           "engineers", "warehouses", "suppliers",
                                           "fault-codes", "roles", "seed-roles"])
    parser.add_argument("--tenant", default="default")
    args = parser.parse_args()

    md = MasterData()
    if args.entity == "seed-roles":
        n = md.seed_builtin_roles(args.tenant)
        print(f"已装入 {n} 个种子角色（已存在则跳过）")
        return
    if args.entity == "roles":
        for r in md.list_biz_roles("gm", "cli", tenant_id=args.tenant):
            print(f"{r['role_key']:<16} L{r['level']} [{r['scope']:<7}] "
                  f"{r['name']}  资源:{','.join(r['resources'])}")
        return
    store = {"customers": md.customers, "products": md.products,
             "equipment": md.equipment, "engineers": md.engineers,
             "warehouses": md.warehouses, "suppliers": md.suppliers,
             "fault-codes": md.fault_codes}[args.entity]
    rows = store.list("gm", "cli", tenant_id=args.tenant)
    if not rows:
        print("（无数据）")
        return
    for r in rows:
        code = r.get(store.CODE_FIELD) or r.get("id")
        print(f"{code:<16} {r.get('name', '')}")


if __name__ == "__main__":
    main()
