# -*- coding: utf-8 -*-
"""
pipeline —— 诊断 DAG（Agent 形态之一：Workflow 预定义编排）
================================================================================

【图结构】（分支规则全部写在代码里，LLM 只在节点内部干活）

  START → prepare → rerun_retrieval → probe_docs → judge_answer
         → classify_root_cause ──(escalate)──→ react_probe → writeback → END
                                └─(正常)──────────────────────→ writeback

【与生产图的关系】

独立于 LangGraphRAGApp（13 节点生产图）：诊断图小而专注，节点复用生产组件
（VectorStoreManager 检索、evalkit JudgeLLM 复判、网关 evalgrade/react 链）。

【两次检索的设计】（R8 / R4 判定的关键）

  · scoped：以 case 租户的 admin 视角复跑 —— 系统「应该」给该租户看什么
  · full：  以 super_admin 视角全库复跑 —— 内容到底存不存在
  · R8 实锤：scoped 结果里出现其他租户文档（租户过滤 expr 失效）
  · R4 方向：scoped 零命中但 full 有命中（内容在库里、租户视角看不到），
    点踩样本没有当时用户角色，只能提示、不能定论（诚实边界）
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, TypedDict

from agentworkflow import rules
from agentworkflow.trace import TraceCollector, truncate

# 复跑检索参数
RETRIEVE_K = 8                 # 每路取 8 条候选（诊断要看全貌，比问答 top-k 宽）
PROBE_CONTEXTS = 5             # 喂给 judge / probe_docs 的上下文条数


class DiagState(TypedDict, total=False):
    """诊断图状态（LangGraph 各节点返回增量合并）。"""
    bc: Dict[str, Any]             # bad case 原始行
    bc_id: int
    dry_run: bool
    actor: str
    query: str
    answer: str
    expected: str
    source: str
    case_tenant: Optional[str]     # 从点踩诊断文本里解析出的租户
    scoped_hits: List[Dict]        # [{file,page,tenant,snippet,score}]
    full_hits: List[Dict]
    leak_hits: List[Dict]
    retrieval_error: str
    docs_relevant: Optional[bool]
    docs_gap: str
    refused: bool
    scores: Dict[str, Any]         # judge 复判结果
    triage: Dict[str, Any]         # 归因结论
    escalate: bool
    writeback: Dict[str, Any]      # 回写结果


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------

def _hit_records(hits, limit: int = RETRIEVE_K) -> List[Dict]:
    """[(Document, distance), ...] → 轻量记录列表（file/page/tenant/snippet）。"""
    out: List[Dict] = []
    for item in (hits or [])[:limit]:
        doc, dist = (item[0], item[1]) if isinstance(item, (tuple, list)) else (item, 0.0)
        meta = getattr(doc, "metadata", {}) or {}
        path = meta.get("file_path") or meta.get("source") or meta.get("file_name") or "?"
        out.append({
            "file": (meta.get("file_name") or str(path).rsplit("/", 1)[-1]),
            "page": meta.get("page", "?"),
            "tenant": rules.tenant_of_path(path),
            "snippet": truncate(getattr(doc, "page_content", ""), 400),
            "score": round(float(dist), 4) if dist is not None else None,
        })
    return out


def _contexts(state: DiagState, n: int = PROBE_CONTEXTS) -> List[str]:
    return [h.get("snippet", "") for h in (state.get("scoped_hits") or [])[:n]]


def _extract_bool_json(text: str) -> Dict[str, Any]:
    """宽松解析 probe_docs 的 JSON 输出：{"relevant":bool,"gap":"..."}"""
    import json as _json
    import re as _re
    out: Dict[str, Any] = {"relevant": None, "gap": ""}
    m = _re.search(r"\{.*\}", text or "", _re.DOTALL)
    if m:
        try:
            obj = _json.loads(m.group(0))
            if isinstance(obj, dict):
                out["relevant"] = obj.get("relevant") if isinstance(obj.get("relevant"), bool) else None
                out["gap"] = str(obj.get("gap") or "")[:200]
                return out
        except Exception:
            pass
    # 退路：文本里找 是/否 判定
    if _re.search(r"是|true|yes|能回答", (text or ""), _re.IGNORECASE):
        out["relevant"] = True
    elif _re.search(r"否|false|no|无法回答|不能", (text or ""), _re.IGNORECASE):
        out["relevant"] = False
    return out


# ----------------------------------------------------------------------
# 图构建
# ----------------------------------------------------------------------

def build_diagnosis_graph(comp, collector: TraceCollector):
    """
    构建并编译诊断图。

    :param comp: 组件命名空间（llm / vector_db / memory_store），由调用方注入：
                 Web 模式复用 rag_web_server 全局 orchestrator 的组件；
                 CLI 模式由 diagnose.build_components() 独立构建。
    :param collector: 统一轨迹收集器
    """
    from langgraph.graph import StateGraph, START, END  # 懒加载（重依赖）

    # ---- 节点包装：进/出轨迹 + 计时（轻量版 _wrap_node_with_checkpoint 思路）----
    def traced(name: str, fn):
        def wrapped(state: DiagState) -> Dict:
            t0 = time.time()
            collector.add("workflow", "node_enter", name,
                          input=state.get("query", ""))
            try:
                delta = fn(state) or {}
            except Exception as e:            # 单节点失败不拖垮整图：降级为错误增量
                delta = {"retrieval_error": f"{name} 节点异常: {e}"} if name == "rerun_retrieval" \
                    else {"triage": rules.classify_signals(retrieval_error=str(e))}
                print(f"  [agentworkflow] ⚠ {name} 节点异常: {e}")
            collector.add("workflow", "node_exit", name,
                          output={k: delta.get(k) for k in list(delta)[:4]},
                          latency_ms=(time.time() - t0) * 1000)
            return delta
        return wrapped

    # ---- 节点 1：prepare（归一化 + 解析租户）----
    def node_prepare(state: DiagState) -> Dict:
        bc = state.get("bc") or {}
        return {
            "bc_id": bc.get("id") or state.get("bc_id"),
            "query": (bc.get("query") or "").strip(),
            "answer": (bc.get("answer") or "").strip(),
            "expected": (bc.get("expected") or "").strip(),
            "source": bc.get("source") or "",
            # 点踩时诊断文本里带了 tenant=xxx（rag_web_server /api/feedback 写入）
            "case_tenant": rules.parse_tenant(bc.get("diagnosis") or ""),
        }

    # ---- 节点 2：rerun_retrieval（双视角复跑，绕过 Redis 缓存直连检索）----
    def node_rerun_retrieval(state: DiagState) -> Dict:
        from advanced_rag_agent import VectorStoreManager  # 懒加载
        query = state.get("query", "")
        tenant = state.get("case_tenant")
        db = comp.vector_db
        err = ""
        scoped, full = [], []
        try:
            # scoped：本租户 admin 视角（tenant_id expr 下推）
            if tenant:
                scoped = VectorStoreManager.search(
                    db, query, k=RETRIEVE_K, filter_role="admin",
                    user_id="agent-diagnosis", tenant_id=tenant)
            # full：全库视角（super_admin → 无 expr）
            full = VectorStoreManager.search(
                db, query, k=RETRIEVE_K, filter_role="super_admin",
                user_id="agent-diagnosis", tenant_id="__global__")
            if not tenant:
                scoped = full            # 租户未知：只有全库一票视角，R8 不判（防误报）
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

        scoped_recs = _hit_records(scoped)
        full_recs = _hit_records(full)
        # R8 实锤：本租户视角命中了「能解析出租户且 ≠ 本租户」的文档
        leak = [h for h in scoped_recs
                if tenant and h.get("tenant") and h["tenant"] != tenant] if tenant else []
        return {
            "scoped_hits": scoped_recs, "full_hits": full_recs,
            "leak_hits": leak, "retrieval_error": err,
        }

    # ---- 节点 3：probe_docs（LLM 判定检索结果能否回答 query，走 evalgrade 强链）----
    def node_probe_docs(state: DiagState) -> Dict:
        scoped = state.get("scoped_hits") or []
        if not scoped:
            return {"docs_relevant": False, "docs_gap": "租户视角零召回"}
        system = (
            "你是 RAG 检索质量审查员。给定用户问题和检索到的文档片段，判断这些片段"
            "是否包含足以回答该问题的内容。只输出一个 JSON 对象（不要解释、不要代码块）："
            '{"relevant":true或false,"gap":"若不足，缺什么，≤40字"}'
        )
        ctx = "\n\n".join(f"[片段{i}] {h.get('snippet','')}"
                          for i, h in enumerate(scoped[:PROBE_CONTEXTS], 1))
        user_p = f"【问题】{state.get('query','')}\n\n【检索片段】\n{ctx}"
        try:
            from llm_gateway import get_gateway
            resp = get_gateway().chat_detailed(system, user_p, task="evalgrade")
            parsed = _extract_bool_json(getattr(resp, "text", "") or "")
            return {"docs_relevant": parsed["relevant"], "docs_gap": parsed["gap"]}
        except Exception as e:
            # 判定失败不硬猜：None 表示未知（归因规则里按「无法判定」处理）
            print(f"  [agentworkflow] ⚠ probe_docs 失败: {e}")
            return {"docs_relevant": None, "docs_gap": f"探查失败: {e}"}

    # ---- 节点 4：judge_answer（复用 evalkit.JudgeLLM 复判当时的故障答案）----
    def node_judge_answer(state: DiagState) -> Dict:
        answer = state.get("answer", "")
        if not answer:
            return {"scores": {"error": "无故障答案快照，无法复判"},
                    "refused": False}
        try:
            from evalkit.judge import JudgeLLM
            judge = JudgeLLM()
            scores = judge.grade(state.get("query", ""), answer, _contexts(state))
            refused = JudgeLLM._grade_refusal(answer)   # 语义正则判拒答，零 LLM
            return {"scores": scores, "refused": refused}
        except Exception as e:
            return {"scores": {"error": f"judge 初始化/执行失败: {e}"},
                    "refused": False}

    # ---- 节点 5：classify_root_cause（纯代码规则归因——Workflow 形态的纪律点）----
    def node_classify(state: DiagState) -> Dict:
        triage = rules.classify_signals(
            leak_hits=state.get("leak_hits"),
            scoped_hits=state.get("scoped_hits"),
            full_hits=state.get("full_hits"),
            docs_relevant=state.get("docs_relevant"),
            docs_gap=state.get("docs_gap", ""),
            scores=state.get("scores"),
            refused=state.get("refused", False),
            source=state.get("source", ""),
            retrieval_error=state.get("retrieval_error", ""),
        )
        triage["engine"] = "workflow"
        return {"triage": triage, "escalate": bool(triage.get("escalate"))}

    def route_after_classify(state: DiagState) -> str:
        return "react_probe" if state.get("escalate") else "writeback"

    # ---- 节点 6：react_probe（升级分支：低置信时开放探查）----
    def node_react_probe(state: DiagState) -> Dict:
        from agentworkflow.probe import run_react_probe, merge_probe_into_triage
        probe = run_react_probe(
            state.get("bc") or {}, comp.llm, comp.vector_db, collector,
            actor=state.get("actor") or "agent-diagnosis",
            tenant_id=state.get("case_tenant") or "default",
            max_steps=5,                     # 单条诊断 LLM 调用 ≤6 次的硬顶之一
        )
        merged = merge_probe_into_triage(state.get("triage") or {}, probe)
        return {"triage": merged, "escalate": False}

    # ---- 节点 7：writeback（回写 bad_cases；dry_run 只出结论不落库）----
    def node_writeback(state: DiagState) -> Dict:
        ms = comp.memory_store
        bc = state.get("bc") or {}
        triage = state.get("triage") or {}
        code = triage.get("code")
        engine = triage.get("engine", "workflow")
        conf = triage.get("confidence", "低")

        files = "、".join(h.get("file", "?") for h in (state.get("scoped_hits") or [])[:3]) or "无"
        scores = state.get("scores") or {}
        judge_line = ""
        if scores.get("faithfulness") is not None:
            judge_line = (f"；judge 复判 faithfulness={scores['faithfulness']:.2f}, "
                          f"relevancy={scores.get('relevancy')}")
        reason = (triage.get("reason") or "").rstrip("。；; ")
        diagnosis_text = (
            f"【自动诊断·{engine}】{code or '未能归因'} {triage.get('title', '')}\n"
            f"证据：{reason}{judge_line}\n"
            f"复跑召回（本租户视角）：{len(state.get('scoped_hits') or [])} 条（{files}）\n"
            f"建议：{triage.get('suggested_fix', '')}\n"
            f"置信度：{conf}（机器建议，确认后请人工流转状态）"
        )

        written = False
        if not state.get("dry_run") and ms is not None:
            # 只把 open 推到 in_progress；已在处理/已解决的不动（不降级人工状态）
            new_status = "in_progress" if (bc.get("status") == "open") else None
            written = ms.update_bad_case_status(
                state.get("bc_id"), status=new_status,
                diagnosis=diagnosis_text, root_cause=code,
            )
        return {"writeback": {"written": bool(written),
                              "root_cause": code, "engine": engine,
                              "confidence": conf,
                              "diagnosis": diagnosis_text}}

    # ---- 组装图 ----
    g = StateGraph(DiagState)
    g.add_node("prepare", traced("prepare", node_prepare))
    g.add_node("rerun_retrieval", traced("rerun_retrieval", node_rerun_retrieval))
    g.add_node("probe_docs", traced("probe_docs", node_probe_docs))
    g.add_node("judge_answer", traced("judge_answer", node_judge_answer))
    g.add_node("classify_root_cause", traced("classify_root_cause", node_classify))
    g.add_node("react_probe", traced("react_probe", node_react_probe))
    g.add_node("writeback", traced("writeback", node_writeback))

    g.add_edge(START, "prepare")
    g.add_edge("prepare", "rerun_retrieval")
    g.add_edge("rerun_retrieval", "probe_docs")
    g.add_edge("probe_docs", "judge_answer")
    g.add_edge("judge_answer", "classify_root_cause")
    g.add_conditional_edges("classify_root_cause", route_after_classify,
                            {"react_probe": "react_probe", "writeback": "writeback"})
    g.add_edge("react_probe", "writeback")
    g.add_edge("writeback", END)
    return g.compile()
