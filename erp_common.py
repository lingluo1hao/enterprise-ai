"""
================================================================================
 erp_common.py — 工厂业务系统公共基座（DB + 事务 + 动态角色 ACL + 审计）
================================================================================

 分层（生产级三段式）
 ----
   erp_common    基座：连接 / 事务执行器 / biz_roles 角色引擎 / 越权审计
   master_data   主数据：客户/供应商/产品/设备/工程师/仓库 + 角色管理
   inventory_system 库存域：余额 / 流水 / 出入库单据
   sales_system  销售域 / repair_system 维修域 / purchase_system 采购域
   mcp_server    工具暴露层（数字员工经 MCP 调用，同一份逻辑零分叉）

 角色（已定稿：两层 + 系统自建）
 ----
   系统角色 admin_users.role（user/admin/super_admin）→ 审批等级
   业务角色 biz_roles 表（租户级自建：车间主任/质检员/计划员…）→ 行级 ACL
   加载优先级：DB biz_roles（本租户）> access_rules.yaml 种子 > 代码兜底
   fail-closed：查不到的角色一律拒绝，绝不默认放行。

 多用户隔离（四层）
 ----
   L1 租户隔离 tenant_id / L2 行级 ACL（角色→resources×scope）
   L3 操作矩阵（ops） / L4 越权审计（logs/audit.log，result=blocked）

 库存事务铁律
 ----
   涉库存复合操作必须走 execute_txn（单连接事务），
   绝不允许「扣了库存没写流水」的中间态落库。

 表结构权威源 config/init_db.sql；各 Store init 时幂等建表兜底。
 MySQL 不可用时降级内存模式（同进程可演示，跨进程不共享）。
================================================================================
"""

import os
import time
import json
from typing import Dict, List, Optional

from memory_store import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE, MYSQL_CHARSET,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
ACCESS_RULES_PATH = os.path.join(ROOT, "config", "access_rules.yaml")

RES_NAMES = {
    "sales_order": "销售订单", "repair_order": "维修工单",
    "product": "产品主数据", "inventory": "库存", "shipment": "物流单",
    "engineer": "工程师", "customer": "客户", "supplier": "供应商",
    "purchase_order": "采购单", "equipment": "设备台账", "warehouse": "仓库",
    "fault_code": "故障代码", "pm_plan": "保养计划", "stock": "出入库单据",
    "biz_role": "业务角色",
}

# ---- YAML 种子角色（装库用；DB 里有 biz_roles 后以 DB 为准） ----
_SEED_ROLES = {
    "sales_user": {"name": "销售跟单员", "level": 1,
                   "resources": ["sales_order", "shipment", "product", "customer"],
                   "scope": "own", "ops": ["create", "query", "update", "request"]},
    "repair_user": {"name": "维修工程师", "level": 1,
                    "resources": ["repair_order", "fault_code", "product",
                                  "equipment", "stock"],  # stock=自助领料出库单（own 范围仅自己的领料单）
                    "scope": "own",
                    "ops": ["create", "query", "update", "request"]},  # create=报修建单
    "warehouse_user": {"name": "库管员", "level": 1,
                       "resources": ["stock", "shipment", "inventory", "warehouse",
                                     "purchase_order"],   # 采购收货是仓储动作
                       "scope": "tenant", "ops": ["update", "query"]},
    "purchase_user": {"name": "采购员", "level": 1,
                      "resources": ["purchase_order", "supplier", "inventory"],
                      "scope": "own", "ops": ["create", "query", "update", "request"]},
    "dept_manager": {"name": "部门经理", "level": 2,
                     "resources": ["sales_order", "repair_order", "product", "inventory",
                                   "shipment", "engineer", "purchase_order", "stock"],
                     "scope": "tenant",
                     "ops": ["create", "query", "update", "request", "approve"]},
    "gm": {"name": "总经理", "level": 2,
           "resources": ["sales_order", "repair_order", "product", "inventory",
                         "shipment", "engineer", "purchase_order", "customer",
                         "supplier", "equipment", "warehouse", "fault_code",
                         "pm_plan", "stock"],
           "scope": "tenant",
           "ops": ["create", "query", "update", "request", "approve"]},
    "super_admin": {"name": "超级管理员", "level": 3,
                    "resources": ["sales_order", "repair_order", "product", "inventory",
                                  "shipment", "engineer", "purchase_order", "customer",
                                  "supplier", "equipment", "warehouse", "fault_code",
                                  "pm_plan", "stock"],
                    "scope": "all",
                    "ops": ["create", "query", "update", "request", "approve"]},
}

_BIZ_ROLES_DDL = """
CREATE TABLE IF NOT EXISTS `biz_roles` (
  `id`         BIGINT AUTO_INCREMENT PRIMARY KEY,
  `tenant_id`  VARCHAR(64)   NOT NULL DEFAULT 'default',
  `role_key`   VARCHAR(64)   NOT NULL,
  `name`       VARCHAR(128)  NOT NULL,
  `level`      TINYINT       NOT NULL DEFAULT 1,
  `resources`  JSON          NOT NULL,
  `scope`      VARCHAR(16)   NOT NULL DEFAULT 'own',
  `ops`        JSON          NOT NULL,
  `is_system`  TINYINT       NOT NULL DEFAULT 0,
  `created_by` VARCHAR(64)   NULL,
  `created_at` DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY `uk_tenant_role` (`tenant_id`, `role_key`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


def load_seed_roles() -> Dict:
    """
    种子角色加载（T-03 单真源治理，根治双源漂移）：

    `_SEED_ROLES`（代码）是**唯一真源**；access_rules.yaml 的 order_acl
    仅作**可选覆盖层**（逐角色覆盖，字段级合并）。两侧 ops/resources
    不一致时**启动即告警**（warnings.warn）——下次改角色忘同步一侧，
    会在日志里立刻暴露，而不是等运行时炸。
    """
    merged = {k: dict(v) for k, v in _SEED_ROLES.items()}   # 代码真源打底
    try:
        import yaml
        with open(ACCESS_RULES_PATH, "r", encoding="utf-8") as f:
            yacl = (yaml.safe_load(f) or {}).get("order_acl") or {}
        if isinstance(yacl, dict) and yacl:
            for key, v in yacl.items():
                base = merged.get(key, {})
                merged[key] = {**base, **v}                  # yaml 覆盖层
            # 漂移告警：两侧同名字段的 ops/resources 不一致
            import warnings
            for key in set(_SEED_ROLES) & set(yacl):
                for f_ in ("ops", "resources"):
                    if (list(_SEED_ROLES[key].get(f_) or [])
                            != list(yacl[key].get(f_) or [])):
                        warnings.warn(
                            f"[角色配置漂移] {key}.{f_}：code="
                            f"{_SEED_ROLES[key].get(f_)} vs yaml="
                            f"{yacl[key].get(f_)}，以 yaml 覆盖——"
                            f"请同步两侧或仅保留代码真源")
    except Exception:
        pass   # yaml 缺失/解析失败：仅用代码真源（fail-closed 语义不变）
    return merged


def audit(username: str, action: str, target: str = "",
          result: str = "success", detail: str = "", quiet: bool = False):
    """操作/越权审计（复用项目 audit_logger，落 logs/audit.log）。"""
    try:
        import contextlib, io
        from audit_logger import get_audit_logger
        ctx = (contextlib.redirect_stdout(io.StringIO())
               if quiet else contextlib.nullcontext())
        with ctx:
            get_audit_logger().log(ip="digital_employee", username=username,
                                   action=action, target=target,
                                   result=result, detail=detail)
    except Exception:
        pass  # 审计失败不阻断业务（审计器自身有兜底打印）


class PermissionDenied(PermissionError):
    """ACL 拒绝（L2/L3），消息带原因——同时已落审计日志。"""


# ============================================================================
# 角色引擎（biz_roles 表驱动，租户级自建角色）
# ============================================================================
class RoleEngine:
    """
    业务角色引擎：读 biz_roles 表（带进程内 TTL 缓存），支持系统自建角色。

    缓存策略：tenant → {role_key: rule}，TTL 60s；create_role 后主动失效。
    MySQL 不可用时退化为 YAML 种子（同构、不放权）。
    """

    TTL = 60

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._cache: Dict[str, tuple] = {}   # tenant -> (ts, {role_key: rule})
        self._pymysql = None
        self._conn_kw = None
        try:
            import pymysql
            self._pymysql = pymysql
            self._conn_kw = dict(
                host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
                password=MYSQL_PASSWORD, database=MYSQL_DATABASE,
                charset=MYSQL_CHARSET, autocommit=True,
            )
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(_BIZ_ROLES_DDL)
        except Exception as e:
            if verbose:
                print(f"  [RoleEngine] biz_roles 表不可用，退化为种子角色：{e}")

    def _connect(self):
        return self._pymysql.connect(**self._conn_kw)

    # ------------------------------------------------------------------
    def rules_for(self, tenant_id: str = "default") -> Dict[str, Dict]:
        """
        某租户的全量角色规则（含 level/resources/scope/ops）。

        合并策略（D2 修复）：**内置种子打底 + DB 自建角色覆盖/追加**——
        DB 一旦有任意自建角色，内置角色不再「消失」；同名时以 DB 为准。
        """
        cached = self._cache.get(tenant_id)
        if cached and time.time() - cached[0] < self.TTL:
            return cached[1]
        rules = {k: dict(v) for k, v in load_seed_roles().items()}   # 内置打底
        if self._pymysql is not None:
            try:
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT role_key, name, level, resources, scope, ops "
                            "FROM biz_roles WHERE tenant_id = %s", (tenant_id,))
                        cols = [c[0] for c in cur.description]
                        for r in cur.fetchall():
                            d = dict(zip(cols, r))
                            rules[d["role_key"]] = {
                                "name": d["name"], "level": int(d["level"]),
                                "resources": json.loads(d["resources"]),
                                "scope": d["scope"], "ops": json.loads(d["ops"]),
                            }
            except Exception as e:
                if self.verbose:
                    print(f"  [RoleEngine] 读 biz_roles 失败，仅用种子角色：{e}")
        self._cache[tenant_id] = (time.time(), rules)
        return rules

    def rule_of(self, role_key: str, tenant_id: str = "default") -> Optional[Dict]:
        """单个角色规则；查不到返回 None（fail-closed 由调用方拒绝）。"""
        return self.rules_for(tenant_id).get(role_key)

    def level_of(self, role_key: str, tenant_id: str = "default") -> int:
        """角色审批等级（1/2/3）；未知角色返回 0（任何审批都过不了）。"""
        r = self.rule_of(role_key, tenant_id)
        return int(r["level"]) if r else 0

    # ------------------------------------------------------------------
    def create_role(self, role_key: str, name: str, level: int,
                    resources: List[str], scope: str, ops: List[str],
                    tenant_id: str = "default", created_by: str = "admin") -> Dict:
        """
        系统自建角色（租户级）。工厂可按需创建车间主任/质检员/计划员等。

        :param level: 审批等级 1/2/3（2 及以上才有审批权）
        :param resources: 可见资源（须在 RES_NAMES 已注册的资源内）
        :param scope: own / tenant / all
        """
        if not role_key or not name:
            raise ValueError("role_key 与 name 不能为空")
        if level not in (1, 2, 3):
            raise ValueError("level 取值 1/2/3")
        if scope not in ("own", "tenant", "all"):
            raise ValueError("scope 取值 own/tenant/all")
        bad_res = [r for r in resources if r not in RES_NAMES]
        if bad_res:
            raise ValueError(f"未注册的资源 {bad_res}，合法值：{sorted(RES_NAMES)}")
        bad_ops = [o for o in ops if o not in
                   ("create", "query", "update", "request", "approve")]
        if bad_ops:
            raise ValueError(f"非法操作 {bad_ops}")
        if self._pymysql is None:
            raise RuntimeError("MySQL 不可用，自建角色须数据库支持")
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO biz_roles (tenant_id, role_key, name, level, "
                        "resources, scope, ops, created_by) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                        (tenant_id, role_key, name, level,
                         json.dumps(resources), scope, json.dumps(ops), created_by))
                conn.commit()
        except Exception as e:
            raise ValueError(f"创建角色失败（role_key 可能已存在）：{e}")
        self._cache.pop(tenant_id, None)   # 失效缓存
        return {"role_key": role_key, "name": name, "level": level,
                "resources": resources, "scope": scope, "ops": ops}

    def seed_builtin(self, tenant_id: str = "default") -> int:
        """
        内置种子角色 upsert 进 biz_roles（T-03 单真源的落库侧）：

        - 不存在 → 插入（is_system=1）
        - 已存在且 is_system=1 → **以代码真源覆盖更新**（防旧种子行
          压掉新代码——demo 连 default 租户曾踩雷：旧 DB 种子无
          warehouse_user.purchase_order，覆盖了 F2 修复）
        - 已存在且 is_system=0（用户自建同名角色）→ 不动
        返回写入/更新数。
        """
        if self._pymysql is None:
            return 0
        # 旧版种子迁移：早期 create_role 装的种子行 is_system=0（列默认值），
        # 会被下面的「用户自建不动」逻辑误跳过——先按 created_by='seed' 归位
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE biz_roles SET is_system = 1 "
                        "WHERE tenant_id = %s AND created_by = 'seed' "
                        "AND is_system = 0", (tenant_id,))
                conn.commit()
        except Exception:
            pass
        n = 0
        for key, r in load_seed_roles().items():
            try:
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT is_system FROM biz_roles "
                            "WHERE tenant_id = %s AND role_key = %s",
                            (tenant_id, key))
                        existing = cur.fetchone()
                        if existing is None:
                            cur.execute(
                                "INSERT INTO biz_roles (tenant_id, role_key, "
                                "name, level, resources, scope, ops, "
                                "is_system, created_by) VALUES (%s,%s,%s,%s,"
                                "%s,%s,%s,1,'seed')",
                                (tenant_id, key, r["name"], r["level"],
                                 json.dumps(r["resources"]),
                                 r["scope"], json.dumps(r["ops"])))
                        elif existing[0] == 1:
                            cur.execute(
                                "UPDATE biz_roles SET name=%s, level=%s, "
                                "resources=%s, scope=%s, ops=%s WHERE "
                                "tenant_id=%s AND role_key=%s",
                                (r["name"], r["level"],
                                 json.dumps(r["resources"]), r["scope"],
                                 json.dumps(r["ops"]), tenant_id, key))
                        else:
                            continue   # 用户自建同名角色，代码不碰
                    conn.commit()
                n += 1
            except Exception as e:
                if self.verbose:
                    print(f"  [RoleEngine] seed {key} 失败：{e}")
                continue
        self._cache.pop(tenant_id, None)
        return n


# 模块级单例（各 Store / 审批门共享同一份角色缓存）
_role_engine: Optional[RoleEngine] = None


def get_role_engine() -> RoleEngine:
    global _role_engine
    if _role_engine is None:
        _role_engine = RoleEngine()
    return _role_engine


# ============================================================================
# 单 Store 数据基座
# ============================================================================
class ErpDb:
    """
    单 Store 数据基座：连接 / 幂等建表 / 事务 / ACL（走 RoleEngine）。

    子类声明 TABLE / DDL / RESOURCE；带 created_by 的表自动获得行级可见性。
    """

    TABLE = ""
    DDL = ""
    RESOURCE = ""

    def __init__(self, ensure_schema: bool = True, verbose: bool = True):
        self.verbose = verbose
        self.available = False
        self._fallback_rows: List[Dict] = []
        self._fallback_seq = 0
        self.roles = get_role_engine()
        try:
            import pymysql
            self._pymysql = pymysql
            self._conn_kw = dict(
                host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
                password=MYSQL_PASSWORD, database=MYSQL_DATABASE,
                charset=MYSQL_CHARSET, autocommit=False,   # 事务由 execute_txn 管理
            )
            if ensure_schema and self.DDL:
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(self.DDL)
                    conn.commit()
            else:
                # D9 修复：DDL 为空的 Store（主数据等）也做一次真实连通探测，
                # 避免「available 误 True、首次 _execute 才崩」的假降级
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                    conn.commit()
            self.available = True
            if verbose:
                print(f"  [{type(self).__name__}] 连接成功: "
                      f"{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}")
        except Exception as e:
            self._pymysql = None
            if verbose:
                print(f"  [{type(self).__name__}] 连接失败，降级为内存模式: {e}")

    # ------------------------------------------------------------------
    # 连接与执行
    # ------------------------------------------------------------------
    def _connect(self):
        """取一个新连接（autocommit=False）。配合 execute / execute_txn 使用。"""
        return self._pymysql.connect(**self._conn_kw)

    def _execute(self, sql: str, params: tuple = ()):
        """
        单语句自动提交执行。

        :return: SELECT → dict 行列表；INSERT → **cursor.lastrowid**（自增 id，
                 无自增列时 None）。

        D5 修复：此前用「INSERT 后另发 SELECT LAST_INSERT_ID()」取 id——但
        LAST_INSERT_ID() 是**连接级**变量，_connect() 每语句新建连接导致其
        恒为 0（级联崩溃 + id 错位，P0 数据正确性缺陷）。改为同一游标内
        cursor.lastrowid，跨连接语义安全。
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                result = None
                if cur.description:
                    cols = [c[0] for c in cur.description]
                    result = [dict(zip(cols, r)) for r in cur.fetchall()]
                else:
                    result = cur.lastrowid or None
            conn.commit()
            return result

    def _execute_txn(self, statements: List[tuple]):
        """
        事务执行器：statements = [(sql, params), ...] 单连接依次执行，
        全部成功才 COMMIT，任何异常 ROLLBACK 后上抛。

        库存类复合操作（占用/扣减/释放 + 写流水）必须走这里——
        绝不允许「扣了库存没写流水」的中间态落库。
        """
        with self._connect() as conn:
            try:
                with conn.cursor() as cur:
                    for sql, params in statements:
                        cur.execute(sql, params)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    # ------------------------------------------------------------------
    # L2/L3：ACL 引擎（角色查 RoleEngine，租户维度）
    # ------------------------------------------------------------------
    def _check_op(self, op: str, acting_role: str, acting_user: str,
                  tenant_id: str = "default", resource: str = None) -> Dict:
        """
        操作矩阵校验：角色须存在、资源须可见、op 须在白名单。
        违规抛 PermissionDenied 并落审计（L4）。
        """
        # T-04（D7 收尾）：租户必填断言——空/None 直接拒绝，
        # 防「静默落 default 租户」类错位（调用点已全显式传参，此为防回潮护栏）
        if not tenant_id or not str(tenant_id).strip():
            raise ValueError("tenant_id 不能为空（多租户必填，防静默错位）")
        resource = resource or self.RESOURCE
        rule = self.roles.rule_of(acting_role, tenant_id)
        if rule is None:
            audit(acting_user, f"{resource}.{op}", result="blocked",
                  detail=f"未知/未授权角色「{acting_role}」",
                  quiet=not self.verbose)
            raise PermissionDenied(
                f"角色「{acting_role}」不存在或未授权（租户 {tenant_id}），拒绝操作")
        if resource not in rule.get("resources", []):
            audit(acting_user, f"{resource}.{op}", result="blocked",
                  detail=f"角色「{acting_role}」无权访问资源 {resource}",
                  quiet=not self.verbose)
            raise PermissionDenied(
                f"角色「{acting_role}」无权访问{RES_NAMES.get(resource, resource)}"
                "（越权已审计）")
        if op not in rule.get("ops", []):
            audit(acting_user, f"{resource}.{op}", result="blocked",
                  detail=f"角色「{acting_role}」无权执行 {op}",
                  quiet=not self.verbose)
            raise PermissionDenied(
                f"角色「{acting_role}」无权对{RES_NAMES.get(resource, resource)}"
                f"执行「{op}」（越权已审计）")
        return rule

    def _visibility_where(self, rule: Dict, acting_user: str,
                          tenant_id: str) -> List[tuple]:
        """
        可见范围（L1+L2）→ 条件片段 [(frag, param), ...]。
        own：仅自己创建（created_by）且本租户；tenant：本租户；all：跨租户。
        """
        scope = rule.get("scope", "own")
        conds = []
        if scope == "own":
            conds.append(("tenant_id = %s", tenant_id))
            conds.append(("created_by = %s", acting_user))
        elif scope == "tenant":
            conds.append(("tenant_id = %s", tenant_id))
        return conds

    # ------------------------------------------------------------------
    # 内存降级模式的公共助手
    # ------------------------------------------------------------------
    def fallback_visible(self, rule: Dict, row: Dict, acting_user: str,
                         tenant_id: str) -> bool:
        scope = rule.get("scope", "own")
        if scope == "own":
            return row.get("tenant_id") == tenant_id and \
                row.get("created_by") == acting_user
        if scope == "tenant":
            return row.get("tenant_id") == tenant_id
        return True

    def _next_no(self, prefix: str, oid: int) -> str:
        """单号：前缀-日期-四位序号。"""
        return f"{prefix}-{time.strftime('%Y%m%d')}-{oid:04d}"
