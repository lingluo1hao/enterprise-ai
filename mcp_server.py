"""
================================================================================
 mcp_server.py — 企业级 RAG Agent 的 MCP Server（标准协议暴露层）
================================================================================

 作用
 ----
 把项目里「协议无关」的 Skill 内核（skill_framework.py）包装成标准 MCP Tools，
 任何兼容 MCP 的客户端（Claude Desktop / Cursor / 自研 Agent / 其它 AI 应用）
 都能零改造地复用你的工具。

 暴露能力
 --------
   Tools   : calculator(expression)        → 复用 CalculatorSkill + AST 沙箱
             doc_search(query, top_k)      → 复用 DocSearchSkill 的安全守门
   Resource: skills://list                → 技能清单（让客户端自动发现能力）
   Prompt  : security_review              → 复用审计/安全口径的可复用提示词

 传输方式
 --------
   默认 stdio（本地子进程，最常用）
   想远程暴露时改用 HTTP：python mcp_server.py --http   （Streamable HTTP）

 安全延续
 --------
   - 所有工具先过 BaseSkill.validate_params() 参数白名单
   - calculator 仍走 safe_eval()（AST 白名单），绝不出现 eval()
   - 凭据已外部化（见 .env），Server 本身不持有任何密钥
================================================================================
"""

import os
import sys
import argparse

from fastmcp import FastMCP

# 复用同一份 Skill 内核（与 in-process Agent 完全一致，逻辑不分叉）
from skill_framework import CalculatorSkill, SkillRegistry

mcp = FastMCP("EnterpriseRAGSkills")

# 复用真实的工具沙箱实例
_calc = CalculatorSkill()
_registry = SkillRegistry()
_registry.register(_calc)


# ----------------------------------------------------------------------------
# Tool 1：计算器（完整可运行，安全求值）
# ----------------------------------------------------------------------------
@mcp.tool()
def calculator(expression: str) -> str:
    """
    执行数学计算。适用于需要数值运算、单位换算等场景。
    输入数学表达式（如 120/24 或 5*24），返回计算结果。
    仅允许数字与 + - * / // % ** ( ) 运算符，杜绝任意代码执行。
    """
    return _calc.execute(expression)


# ----------------------------------------------------------------------------
# Tool 2：文档检索（复用安全守门；真实检索在挂载向量库后自动生效）
# ----------------------------------------------------------------------------
@mcp.tool()
def doc_search(query: str, top_k: int = 5) -> str:
    """
    搜索企业文档知识库（Milvus 向量数据库），返回与查询相关的文档片段。
    输入：搜索关键词或问题描述。输出：相关文档片段列表。
    """
    # 延迟导入：没有 ollama 的环境也能正常启动 Server
    try:
        from advanced_rag_agent import DocSearchSkill
    except Exception as e:  # pragma: no cover
        return f"[doc_search] 未找到 DocSearchSkill：{e}"

    # 复用与 in-process Agent 完全相同的参数白名单校验（同一份逻辑）
    probe = DocSearchSkill.__new__(DocSearchSkill)  # 仅构造外壳，不触发重依赖
    err = probe.validate_params(query)
    if err:
        return err

    # 真实部署：复用与 in-process Agent 完全相同的 DocSearchSkill（真连 Milvus）。
    # redirect_stdout 抑制其内部进度 print，避免污染 MCP stdio 协议；
    # 只取 execute() 的纯文本返回值 —— 与 in-process Agent 同一份逻辑，零分叉。
    try:
        import contextlib
        import io
        from advanced_rag_agent import create_llm, VectorStoreManager

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            llm = create_llm(verbose=False)
            vector_db = VectorStoreManager.init_vector_store()
            skill = DocSearchSkill(llm, vector_db, fast_mode=True)
            return skill.execute(query)
    except Exception as e:
        # 无 Milvus/Ollama 环境时优雅降级，仍证明工具总线被复用
        return (
            f"[doc_search] 参数校验通过（top_k={top_k}），但真实检索暂不可用：{e}。"
            "请确认 Milvus/Ollama 已部署可达后重启 MCP Server。"
        )


# ----------------------------------------------------------------------------
# Tool 3+：工厂业务工具（销售/库存/维修/采购/角色，五岗位工具集）
# ----------------------------------------------------------------------------
# 数字员工的「写能力」数据面（docs/guides/factory_business_system_design.md）。
# 分级管控：
#   低危（直接执行）：建单 / 查询 / 流转 / 派单 / 领料…
#   高危（挂起等审）：出库放行 / 关单 / 取消 / 采购审批 / 盘点调整——
#     内部只创建审批单（approval.py），人工批准后才执行。
#
# 身份参数（每个写操作必带，四层隔离依据）：
#   acting_user  登录用户名（数字员工代表谁操作）
#   biz_role     业务角色（biz_roles 表 key，如 sales_user / gm）
#   tenant       租户
# 复杂参数（订单明细等）用 JSON 字符串传入，工具内解析校验。
_erp: dict = None


def _layer():
    """懒初始化工厂业务层（主数据/库存/销售/维修/采购/审批门），单例复用。"""
    global _erp
    if _erp is None:
        import contextlib, io
        from master_data import MasterData
        from inventory_system import InventoryService
        from sales_system import SalesService
        from repair_system import RepairService
        from purchase_system import PurchaseService
        from approval import ApprovalGate, build_default_executors
        from erp_common import PermissionDenied
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):   # 防 Store 初始化 print 污染 stdio
            md = MasterData(verbose=False)
            inv = InventoryService(verbose=False)
            _erp = {
                "md": md, "inv": inv,
                "sales": SalesService(inv, md, verbose=False),
                "repair": RepairService(inv, md, verbose=False),
                "purchase": PurchaseService(inv, md, verbose=False),
                "gate": ApprovalGate(
                    build_default_executors(md, inv), verbose=False),
                "PermissionDenied": PermissionDenied,
            }
    return _erp


def _items(items_json: str):
    import json
    items = json.loads(items_json or "[]")
    if not isinstance(items, list):
        raise ValueError("items_json 须为 JSON 数组")
    return items


def _deny(e) -> str:
    """越权/业务错误统一出口（错误信息已含原因，审计已落 audit.log）。"""
    return f"⛔ {e}"


# ---------------- 角色管理（经理级） ----------------
@mcp.tool()
def list_biz_roles(acting_user: str = "anonymous",
                   biz_role: str = "gm", tenant: str = "default") -> str:
    """列出本租户全部业务角色（role_key/等级/可见资源/操作）。须任意有效角色。"""
    try:
        erp = _layer()
        rows = erp["md"].list_biz_roles(biz_role, acting_user, tenant_id=tenant)
        return "\n".join(
            f"{r['role_key']:<16} L{r['level']} [{r['scope']:<7}] {r['name']}"
            f"  资源:{','.join(r['resources'])}  操作:{','.join(r['ops'])}"
            for r in rows) or "（无角色，请先 seed）"
    except Exception as e:
        return _deny(e)


@mcp.tool()
def create_biz_role(role_key: str, name: str, level: int,
                    resources_json: str, scope: str, ops_json: str,
                    acting_user: str = "anonymous", biz_role: str = "gm",
                    tenant: str = "default") -> str:
    """
    自建业务角色（工厂不限三类角色：车间主任/质检员/计划员…）。须经理级及以上。
    resources_json 如 ["sales_order","product"]；ops_json 如 ["create","query"]。
    """
    try:
        import json
        erp = _layer()
        r = erp["md"].create_biz_role(
            biz_role, acting_user, role_key, name, level,
            json.loads(resources_json), scope, json.loads(ops_json),
            tenant_id=tenant)
        return f"✓ 角色已创建：{r['role_key']}（{r['name']}，L{r['level']}，" \
               f"{r['scope']}）"
    except Exception as e:
        return _deny(e)


# ---------------- 产品与库存 ----------------
@mcp.tool()
def list_products(acting_user: str = "anonymous", biz_role: str = "sales_user",
                  tenant: str = "default") -> str:
    """查询产品与备件主数据（编码/规格/类别/售价/安全库存）。"""
    try:
        erp = _layer()
        rows = erp["md"].products.list(biz_role, acting_user, tenant_id=tenant)
        return "\n".join(
            f"{p['product_code']:<14} {p['name']}（{p.get('spec') or '-'}，"
            f"{p['category']}，¥{p['unit_price']}，安全库存"
            f"{p.get('safety_stock') or 0}）" for p in rows) or "（无产品）"
    except Exception as e:
        return _deny(e)


@mcp.tool()
def query_inventory(acting_user: str = "anonymous",
                    biz_role: str = "warehouse_user", tenant: str = "default",
                    low_only: bool = False) -> str:
    """
    查询库存水位（在库/占用/可用，联产品名）。low_only=True 只看低于安全线的。
    库管/经理/采购角色可查。
    """
    try:
        erp = _layer()
        rows = erp["inv"].query_inventory(biz_role, acting_user,
                                          tenant_id=tenant, low_only=low_only)
        if not rows:
            return "（无库存记录）"
        return "\n".join(
            f"{r.get('product_code')}：在库 {r['stock_qty']} / 占用 "
            f"{r['reserved_qty']} / 可用 {r['available']}"
            + ("  ⚠ 低于安全线" if r["low"] else "") for r in rows)
    except Exception as e:
        return _deny(e)


@mcp.tool()
def query_stock_transactions(acting_user: str = "anonymous",
                              biz_role: str = "warehouse_user",
                              tenant: str = "default", limit: int = 20) -> str:
    """查询库存流水（类型/数量/变动后余额/关联单号——库存审计视图）。"""
    try:
        erp = _layer()
        rows = erp["inv"].transactions_of(biz_role, acting_user,
                                          tenant_id=tenant, limit=limit)
        return "\n".join(
            f"[{r['txn_type']}] {r.get('product_code') or r['product_id']} "
            f"{r['qty']:+d} → 余 {r['balance_after']}（{r.get('ref_no') or '-'}，"
            f"{r.get('operator') or '-'}）" for r in rows) or "（无流水）"
    except Exception as e:
        return _deny(e)


# ---------------- 销售跟单员 ----------------
@mcp.tool()
def create_sales_order(customer_code: str, items_json: str,
                       acting_user: str = "anonymous",
                       biz_role: str = "sales_user",
                       tenant: str = "default") -> str:
    """
    创建销售订单（draft 计划单，不扣库存）。
    items_json 如 [{"product_code":"LBL-100","quantity":2}]；单价自动按主数据快照。
    """
    try:
        erp = _layer()
        o = erp["sales"].create_order(
            biz_role, acting_user, tenant, customer_code, _items(items_json))
        return (f"✓ 销售订单已创建 {o['order_no']}（{o.get('customer_name')}，"
                f"金额 {o['amount']}，状态 {o['status']}；确认须可用库存足够）")
    except Exception as e:
        return _deny(e)


@mcp.tool()
def confirm_sales_order(order_id: int, acting_user: str = "anonymous",
                        biz_role: str = "sales_user",
                        tenant: str = "default") -> str:
    """
    确认销售订单：客户信用额度校验 + 逐行占用库存（防超卖）。
    库存不足/超信用会被拒绝并说明原因。
    """
    try:
        erp = _layer()
        o = erp["sales"].confirm_order(biz_role, acting_user, order_id, tenant)
        return f"✓ 订单 {o['order_no']} 已确认（库存已占用，金额 {o['amount']}）"
    except Exception as e:
        return _deny(e)


@mcp.tool()
def query_sales_orders(acting_user: str = "anonymous",
                       biz_role: str = "sales_user", tenant: str = "default",
                       status: str = None) -> str:
    """查询销售订单（行级裁剪：销售只看自己的，经理看本租户全部）。"""
    try:
        erp = _layer()
        rows = erp["sales"].query_orders(biz_role, acting_user,
                                         tenant_id=tenant, status=status)
        return "\n".join(
            f"{o['order_no']} [{o['status']}] {o.get('customer_name')} "
            f"¥{o['amount']}（{o.get('sales_rep') or '-'}）" for o in rows
        ) or "（无符合条件的订单）"
    except Exception as e:
        return _deny(e)


@mcp.tool()
def create_delivery(order_id: int, acting_user: str = "anonymous",
                    biz_role: str = "sales_user",
                    tenant: str = "default") -> str:
    """
    创建出库单（draft）：缺省把订单全部未出量开成一张出库单（支持分批）。
    注意：出库单须审批放行（release）后才真正扣库存。
    """
    try:
        erp = _layer()
        d = erp["sales"].create_delivery(biz_role, acting_user, order_id, tenant)
        return (f"✓ 出库单已创建 {d['delivery_no']}（订单 #{order_id}，"
                f"{len(d['items'])} 行；放行须走审批）")
    except Exception as e:
        return _deny(e)


@mcp.tool()
def request_release_delivery(delivery_id: int, reason: str = "",
                             acting_user: str = "anonymous",
                             biz_role: str = "warehouse_user",
                             tenant: str = "default") -> str:
    """
    申请放行出库（⚠ 高危：扣库存）。不直接执行——创建审批单挂起，
    经理级批准后扣库存并回写出库量。返回审批单号与批准命令。
    """
    try:
        erp = _layer()
        d = erp["sales"].get_delivery(delivery_id, tenant)
        if not d:
            return f"⛔ 出库单 #{delivery_id} 不存在"
        rid = erp["gate"].request(
            "sales_delivery.release",
            {"delivery_id": delivery_id, "tenant_id": tenant},
            reason=reason or f"数字员工申请放行出库 {d['delivery_no']}",
            requested_by=acting_user, requested_role=biz_role, tenant_id=tenant)
        return (f"⏸ 已挂起：出库单 {d['delivery_no']} 保持 draft，审批单 #{rid} "
                f"等经理级批准。批准：python approval.py --approve {rid} "
                f"--by <审批人> --role dept_manager")
    except Exception as e:
        return _deny(e)


@mcp.tool()
def ship_delivery(delivery_id: int, carrier: str,
                  acting_user: str = "anonymous",
                  biz_role: str = "warehouse_user",
                  tenant: str = "default") -> str:
    """发运：放行后的出库单交接承运商，生成物流单（TRK-）。"""
    try:
        erp = _layer()
        s = erp["sales"].ship_delivery(biz_role, acting_user, delivery_id,
                                       carrier, tenant)
        return f"✓ 已发运：物流单 {s['shipment_no']}（{s['carrier']}，运输中）"
    except Exception as e:
        return _deny(e)


# ---------------- 维修调度员 / 维修工程师 ----------------
@mcp.tool()
def create_repair_order(equipment_code: str, fault_desc: str,
                        acting_user: str = "anonymous",
                        biz_role: str = "repair_user",
                        tenant: str = "default") -> str:
    """
    创建维修工单（设备台账联动：自动判保内/保外；A 类设备自动升 urgent）。
    """
    try:
        erp = _layer()
        o = erp["repair"].create_order(biz_role, acting_user, tenant,
                                       equipment_code, fault_desc)
        return (f"✓ 维修工单已创建 {o['order_no']}（{o.get('equipment_code')}，"
                f"优先级 {o['priority']}，{'保内' if o['warranty'] == 'in' else '保外'}）")
    except Exception as e:
        return _deny(e)


@mcp.tool()
def classify_fault(order_id: int, fault_code: str,
                   acting_user: str = "anonymous", biz_role: str = "repair_user",
                   tenant: str = "default") -> str:
    """故障归类：把工单挂到标准故障代码（标准 SOP 可经 doc_search 检索）。"""
    try:
        erp = _layer()
        o = erp["repair"].classify_fault(biz_role, acting_user, order_id,
                                         fault_code, tenant)
        return f"✓ 工单 {o['order_no']} 已归类为 {fault_code}"
    except Exception as e:
        return _deny(e)


@mcp.tool()
def assign_technician(order_id: int, engineer_name: str,
                      acting_user: str = "anonymous",
                      biz_role: str = "gm", tenant: str = "default") -> str:
    """派单：指派维修工程师（按故障类别给技能匹配提示，不强制）。"""
    try:
        erp = _layer()
        o = erp["repair"].assign_technician(biz_role, acting_user, order_id,
                                            engineer_name, tenant)
        return (f"✓ 工单 {o['order_no']} 已派给 {engineer_name}"
                + (f"  {o['assign_hint']}" if o.get("assign_hint") else ""))
    except Exception as e:
        return _deny(e)


@mcp.tool()
def start_repair(order_id: int, acting_user: str = "anonymous",
                 biz_role: str = "repair_user", tenant: str = "default") -> str:
    """维修开工：assigned → in_progress。"""
    try:
        erp = _layer()
        o = erp["repair"].start_repair(biz_role, acting_user, order_id, tenant)
        return f"✓ 工单 {o['order_no']} 已开工"
    except Exception as e:
        return _deny(e)


@mcp.tool()
def use_repair_parts(order_id: int, items_json: str,
                     acting_user: str = "anonymous",
                     biz_role: str = "repair_user",
                     tenant: str = "default") -> str:
    """
    维修领料（直接扣库存，写 PARTS_OUT 流水）。
    items_json 如 [{"product_code":"SPR-SRV","quantity":1}]；库存不足会被拒绝。
    """
    try:
        erp = _layer()
        parts = erp["repair"].use_repair_parts(biz_role, acting_user, order_id,
                                               _items(items_json), tenant)
        return f"✓ 领料完成（{len(parts)} 行，库存已扣减并记流水）"
    except Exception as e:
        return _deny(e)


@mcp.tool()
def submit_resolution(order_id: int, resolution: str,
                      downtime_hours: float = None,
                      acting_user: str = "anonymous",
                      biz_role: str = "repair_user",
                      tenant: str = "default") -> str:
    """
    提交维修完工（⚠ 高危：关单）。不直接执行——创建审批单挂起，
    经理级批准后工单才转 resolved（结论/停机时长随审批落库）。
    """
    try:
        erp = _layer()
        o = erp["repair"].get_order(order_id, tenant)
        if not o:
            return f"⛔ 工单 #{order_id} 不存在"
        rid = erp["gate"].request(
            "repair_order.resolve",
            {"order_id": order_id, "resolution": resolution,
             "downtime_hours": downtime_hours, "tenant_id": tenant},
            reason=f"维修完工：{resolution[:80]}",
            requested_by=acting_user, requested_role=biz_role, tenant_id=tenant)
        return (f"⏸ 完工申请已挂起：工单 {o['order_no']} 保持 "
                f"{o['status']}，审批单 #{rid} 等经理级批准")
    except Exception as e:
        return _deny(e)


@mcp.tool()
def query_repair_orders(acting_user: str = "anonymous",
                        biz_role: str = "repair_user",
                        tenant: str = "default", status: str = None) -> str:
    """查询维修工单（工程师看自己被指派的+自己报的；经理看本租户全部）。"""
    try:
        erp = _layer()
        rows = erp["repair"].query_orders(biz_role, acting_user,
                                          tenant_id=tenant, status=status)
        return "\n".join(
            f"{o['order_no']} [{o['status']}·{o['priority']}·"
            f"{'保内' if o['warranty'] == 'in' else '保外'}] "
            f"{o.get('equipment_code')}：{(o.get('fault_desc') or '')[:24]}"
            f"（工程师 {o.get('technician_name') or '-'}）" for o in rows
        ) or "（无符合条件的工单）"
    except Exception as e:
        return _deny(e)


@mcp.tool()
def trigger_due_pm(acting_user: str = "anonymous", biz_role: str = "gm",
                   tenant: str = "default") -> str:
    """检查到期保养计划并生成维修工单（PM 手动触发，已定稿）。"""
    try:
        erp = _layer()
        created = erp["repair"].trigger_due_pm(biz_role, acting_user, tenant)
        if not created:
            return "（无到期保养计划）"
        return "\n".join(f"✓ {o['order_no']} 已生成（{o['fault_desc'][:40]}）"
                         for o in created)
    except Exception as e:
        return _deny(e)


# ---------------- 采购员 ----------------
@mcp.tool()
def create_purchase_order(supplier_code: str, items_json: str,
                          acting_user: str = "anonymous",
                          biz_role: str = "purchase_user",
                          tenant: str = "default") -> str:
    """
    创建采购单（draft；采购价按主数据 cost_price 快照）。
    items_json 如 [{"product_code":"SPR-SRV","quantity":50}]。
    """
    try:
        erp = _layer()
        po = erp["purchase"].create_po(biz_role, acting_user, tenant,
                                       supplier_code, _items(items_json))
        return (f"✓ 采购单已创建 {po['po_no']}（{po.get('supplier_name')}，"
                f"金额 {po['amount']}；提交后须审批，>10 万须超管级）")
    except Exception as e:
        return _deny(e)


@mcp.tool()
def submit_purchase_order(po_id: int, acting_user: str = "anonymous",
                          biz_role: str = "purchase_user",
                          tenant: str = "default") -> str:
    """提交采购审批：draft → submitted，并自动创建审批单（>10 万须 level=3）。"""
    try:
        erp = _layer()
        po = erp["purchase"].submit_po(biz_role, acting_user, po_id, tenant)
        rid = erp["gate"].request(
            "purchase_order.approve",
            {"po_id": po_id, "amount": float(po["amount"]),
             "tenant_id": tenant},
            reason=f"采购审批 {po['po_no']} 金额 {po['amount']}",
            requested_by=acting_user, requested_role=biz_role, tenant_id=tenant)
        return (f"⏸ 采购单 {po['po_no']} 已提交，审批单 #{rid} "
                f"（金额 {po['amount']}"
                f"{'——超 10 万须超管级审批' if float(po['amount']) > 100000 else ''}）")
    except Exception as e:
        return _deny(e)


# ---------------- 审批查询（人工侧入口） ----------------
@mcp.tool()
def list_pending_approvals(acting_user: str = "anonymous",
                           biz_role: str = "dept_manager",
                           tenant: str = "default") -> str:
    """查看待审清单（审批人用；批准/拒绝经 approval.py CLI 或 Web 端）。"""
    try:
        erp = _layer()
        rows = erp["gate"].list_pending(tenant)
        if not rows:
            return "（无待审单）"
        out = []
        for r in rows:
            amt = (r["payload"] or {}).get("amount")
            req = erp["gate"].required_level(r["action_type"], amt)
            out.append(f"#{r['id']} [{r['action_type']}] "
                       f"{r['requested_by']}({r.get('requested_role') or '-'})"
                       f" 金额:{amt if amt is not None else '-'} "
                       f"须L{req}  {r['reason'] or ''}")
        return "\n".join(out)
    except Exception as e:
        return _deny(e)


# ----------------------------------------------------------------------------
# Resource：能力清单（让 MCP 客户端自动发现你的工具）
# ----------------------------------------------------------------------------
@mcp.resource("skills://list")
def list_skills() -> list:
    """返回当前 Server 暴露的全部技能清单（名称 + 描述）。"""
    return _registry.list_skills()


# ----------------------------------------------------------------------------
# Prompt：可复用的安全评审提示词模板
# ----------------------------------------------------------------------------
@mcp.prompt()
def security_review(skill_name: str) -> str:
    """为某个 Skill 生成一份「上线前安全体检」提示词，供客户端直接套用。"""
    return (
        f"请对技能 `{skill_name}` 做一次上线前安全体检，逐项确认：\n"
        "1. 输入是否经过白名单校验（非空 / 长度 / 危险模式）？\n"
        "2. 是否杜绝了 eval / exec / os.system 等任意代码执行？\n"
        "3. 密钥是否从 .env 外部化，未硬编码在代码里？\n"
        "4. 调用是否有限流与结构化审计日志？\n"
        "5. 结果是否按用户角色做了权限过滤？"
    )


def _run():
    parser = argparse.ArgumentParser(description="Enterprise RAG Agent MCP Server")
    parser.add_argument(
        "--http", action="store_true",
        help="使用 Streamable HTTP 传输（默认 stdio）",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.http:
        print(f"[MCP] 启动 Streamable HTTP Server → http://{args.host}:{args.port}/mcp")
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        # 默认 stdio：由客户端作为子进程拉起，通过 stdin/stdout 通信
        mcp.run(transport="stdio")


if __name__ == "__main__":
    _run()
