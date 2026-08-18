# -*- coding: utf-8 -*-
"""
probe —— ReAct 开放探查（Agent 形态之二）
================================================================================

【什么时候进来】

诊断 DAG 的 classify_root_cause 节点判定「低置信」时（复跑检索正常、judge
复判也正常，但样本被标记失败——信号矛盾），由**图的代码分支**升级到本模块。
LLM 自己不能决定进不进来（分支规则写在代码里，这是 Workflow 形态的纪律）。

【为什么用 ReAct】

这种 case 该查什么事先不知道：可能要换关键词重搜、可能要核对文档是否入库、
可能 expected 里藏着特殊要求——需要模型逐步决定下一步。完全复用现有
advanced_rag_agent.ReActAgent（Think→Act→Observe 循环 + _parse_react_output
容错 + max_steps 兜底），每步走网关 react 链，Token 按触发者归因。

【结论契约】

任务提示词要求 Final Answer 输出 JSON {code, evidence, suggestion}；
解析失败不硬猜——返回 code=None，上层标注「探查完成但未归因，转人工」。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agentworkflow.rules import parse_conclusion
from agentworkflow.trace import TraceCollector, truncate


def build_probe_task(case: Dict[str, Any]) -> str:
    """构造给 ReAct Agent 的诊断任务提示词（Final Answer 格式契约写进任务里）。"""
    query = truncate(case.get("query"), 300)
    answer = truncate(case.get("answer") or "（无答案快照）", 400)
    expected = truncate(case.get("expected") or "（未提供）", 200)
    diagnosis = truncate(case.get("diagnosis") or "", 150)
    return (
        f"你是 RAG 系统的 Bad Case 诊断员。下面这条问答被用户标记为失败，"
        f"但系统的检索与判分信号看起来正常，请你主动查证真实根因。\n\n"
        f"【用户问题】{query}\n"
        f"【系统当时的答案】{answer}\n"
        f"【期望答案】{expected}\n"
        f"【原始记录】{diagnosis}\n\n"
        f"请先用 doc_search 换不同关键词检索验证文档里到底有没有能回答该问题的内容，"
        f"再对比答案找问题（常见根因：答案遗漏要点/答非所问/编造字段/文档缺内容）。\n\n"
        f"最终必须输出且只输出一行 JSON（不要 markdown 代码块）：\n"
        f'{{"code":"R1~R8 之一或 null","evidence":"≤80字证据",'
        f'"suggestion":"≤60字修复建议"}}\n'
        f"根因对照：R1 检索缺失；R2 排序埋没；R3 改写负优化；R4 权限误杀；"
        f"R5 生成幻觉；R6 答非所问；R7 拒答错误；R8 跨租户泄漏；"
        f"确实无法归因填 null。"
    )


def run_react_probe(case: Dict[str, Any], llm, vector_db,
                    collector: TraceCollector, actor: str = "agent-diagnosis",
                    tenant_id: str = "default",
                    max_steps: int = 5) -> Dict[str, Any]:
    """
    执行 ReAct 探查。

    :param case: bad case 行（query/answer/expected/diagnosis）
    :param llm:  BaseLLM 兼容对象（走网关 create_llm()）
    :param vector_db: VectorStoreManager 实例
    :param collector: 统一轨迹收集器（steps 会归一为 react 引擎记录）
    :return: {ok, code, evidence, suggestion, engine, raw}
    """
    # 懒加载：advanced_rag_agent / skill_framework 是重模块，本函数才 import
    try:
        from advanced_rag_agent import DocSearchSkill, ReActAgent
        from skill_framework import CalculatorSkill, SkillRegistry
    except Exception as e:
        return {"ok": False, "code": None, "evidence": f"探查组件加载失败: {e}",
                "suggestion": "", "engine": "react", "raw": ""}

    # 组装与生产同源的工具集（SkillRegistry + 沙箱校验原样复用）
    registry = SkillRegistry()
    registry.register(DocSearchSkill(
        llm, vector_db, fast_mode=True,          # fast 模式：跳过查询重写，省 LLM 调用
        user_role="admin", tenant_id=tenant_id or "default"))
    registry.register(CalculatorSkill())

    agent = ReActAgent(llm, registry, max_steps=max_steps)
    agent.user = actor or "agent-diagnosis"      # Token 用量归因到触发者

    task = build_probe_task(case)
    try:
        final_answer, steps = agent.run(task)
    except Exception as e:
        return {"ok": False, "code": None, "evidence": f"ReAct 探查执行失败: {e}",
                "suggestion": "", "engine": "react", "raw": ""}

    # ReActStep 列表 → 统一轨迹（think/act/observe）
    collector.from_react_steps(steps)

    conclusion = parse_conclusion(final_answer)
    return {
        "ok": conclusion.get("code") is not None,
        "code": conclusion.get("code"),
        "evidence": conclusion.get("evidence") or truncate(final_answer, 300),
        "suggestion": conclusion.get("suggestion") or "",
        "engine": "react",
        "raw": truncate(final_answer, 300),
    }


def merge_probe_into_triage(triage: Dict[str, Any],
                            probe: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """把 ReAct 探查结论合并进 DAG 归因结论（探查成功则以其为准，engine 标 react）。"""
    if not probe or not probe.get("ok"):
        # 探查未归因：维持「转人工」，但把探查证据附上
        triage = dict(triage)
        triage["engine"] = "react"
        triage["reason"] = (f"{triage.get('reason', '')} "
                            f"ReAct 探查未归因（{truncate(probe.get('evidence', ''), 150)}），转人工。").strip()
        triage["escalate"] = False
        return triage
    from agentworkflow.rules import _ROOT_CAUSES  # noqa: WPS433 — 同包内聚
    code = probe["code"]
    title, sev, _, fix = _ROOT_CAUSES.get(
        code, ("未知根因", "low", "", "请人工分析"))
    return {
        "code": code, "title": title, "severity": sev,
        "reason": f"[ReAct 探查] {probe.get('evidence', '')}",
        "suggested_fix": probe.get("suggestion") or fix,
        "confidence": "中",          # 探查结论：有证据但无 judge 定量背书
        "escalate": False, "engine": "react",
    }
