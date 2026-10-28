# -*- coding: utf-8 -*-
"""
improvement.generator —— 提案生成器（L2）
================================================================================

把「诊断」翻译成「改进」：读根因码 + 失败样本 + 当前提示词，产出一份结构化提案。

两条铁律
--------
1. **只产出白名单内的变更**。生成完立刻过 whitelist.validate_target，
   不合法直接丢掉，不进验证流程 —— 护栏在提案侧就生效，不等到门禁。
2. **不许碰评分侧**。系统提示里明确禁止：不得要求放宽判定标准、
   不得提到 judge / 阈值 / 分数。对应 T3「生成侧与判定侧禁止同批变更」。

降级设计
--------
网关不可用（配置缺失 / 离线）时，走**确定性变异算子** fallback：
按根因码在提示词尾部追加一条已知有效的约束句。这不是兜底糊弄——
它是进化算法里的固定变异算子，每次产出一致、可回归、可断言。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from . import whitelist as W
from .schema import ImprovementProposal, BUDGET_MAX_LLM_CALLS_PER_ROUND


# ============================================================================
# 确定性变异算子（fallback / 离线可用）
# ============================================================================
# 每条都针对「该类根因最常见的失败模式」写死一句约束，改动最小、可逆、可断言。

MUTATION_OPERATORS: Dict[str, str] = {
    "R1": "若首次改写后仍未召回，保留原查询的专有名词与编号原样再检索一次，"
          "不要为了通顺而替换术语。",
    "R3": "改写只做同义扩展与拼写归一，禁止改变查询中的实体、编号与时间范围；"
          "若不确定，宁可不改写。",
    "R5": "只依据提供的上下文作答。上下文中没有的信息，明确回答「知识库中没有相关内容」，"
          "禁止用常识或推测补全。",
    "R6": "先重述用户问题中被问到的具体对象，再作答，确保答案与问题主语一致。",
    "R7": "只有当上下文完全无法支撑任何回答时才拒答；若只能部分回答，"
          "先给出可确认的部分，再说明缺失部分。",
}

# 不同提示词适用的变异（避免把「只据上下文」塞进分类提示词里）
OPERATOR_APPLIES: Dict[str, tuple] = {
    "rewrite_first": ("R1", "R3"),
    "rewrite_retry": ("R1", "R3"),
    "generate_answer": ("R5", "R6"),
    "writer_compose": ("R5", "R6"),
    "classify": ("R6", "R7"),
    "planner_decompose": ("R6",),
}


def deterministic_mutate(cause: str, target: str, current: str) -> Optional[str]:
    """
    在提示词尾部追加一条针对该根因的约束句。

    已在原文里出现过就不重复追加（幂等）。
    目标不适用该算子则返回 None。
    """
    cause = (cause or "").strip().upper()
    applies = OPERATOR_APPLIES.get(target)
    if not applies or cause not in applies:
        return None
    sentence = MUTATION_OPERATORS.get(cause)
    if not sentence:
        return None
    if sentence[:18] in (current or ""):
        return None
    sep = "\n" if current and not current.endswith("\n") else ""
    return f"{current}{sep}{sentence}\n"


# ============================================================================
# LLM 提案生成
# ============================================================================

_SYSTEM = """你是一个 RAG 系统的提示词改进器。

任务：针对给定的失败根因，改进一条提示词的**指令文本**。

硬性约束（违反即作废）：
1. 只改指令文本本身。不得改动代码、参数、阈值、租户与权限规则、检索配置。
2. 不得要求放宽判定标准，不得提到 judge、评分、分数、阈值。
   （改判定侧属于作弊，一律拒绝。）
3. 不得新增外部知识、示例答案或任何上下文里没有的事实。
4. 改动尽量小：只补一条约束或澄清一句歧义，不要重写整条提示词。
5. 输出必须是严格 JSON，不要代码块、不要解释、不要前后缀文字。

输出格式：
{"rationale": "一句话说明这次改动的理由，指出原提示词哪里会导致该根因",
 "new_system": "改动后的完整提示词文本"}
"""

_USER = """根因码：{cause}
根因含义：{cause_title}
提示词名：{target}
提示词用途：{purpose}

当前提示词全文：
---
{current}
---

失败样本（最多 5 条）：
{samples}

请输出改动后的完整提示词，以及一句话理由。"""


CAUSE_TITLE = {
    "R1": "召回侧完全丢失：正确文档根本没进召回",
    "R2": "排序侧埋没：文档召回到了但排太后面被截断",
    "R3": "改写或精排负优化：改写把原意带偏",
    "R4": "权限过滤误杀：该看的文档被挡在门外",
    "R5": "生成幻觉：答案出现上下文里没有的内容",
    "R6": "答非所问：路由或拆解把问题送错节点",
    "R7": "拒答错误：该拒的没拒或不该拒的拒了",
    "R8": "跨租户泄漏：召回了其他租户的文档",
}


def llm_mutate(cause: str, target: str, current: str,
               samples: List[Dict[str, Any]],
               task: str = "improvement") -> Optional[Dict[str, str]]:
    """
    调网关生成新提示词。返回 {"rationale":..., "new_system":...}，失败返回 None。
    task 走网关路由（默认 "improvement"，未配置时回落 default 链）。
    """
    try:
        from llm_gateway import get_gateway
        gw = get_gateway()
    except Exception:
        return None

    sample_lines = []
    for s in (samples or [])[:5]:
        q = (s.get("query") or s.get("question") or "").strip()
        exp = (s.get("expected") or s.get("note") or s.get("reason") or "").strip()
        if q:
            sample_lines.append(f"- 问：{q[:120]}" + (f"\n  注：{exp[:160]}" if exp else ""))
    if not sample_lines:
        sample_lines = ["（无样本，依据根因含义做保守改进）"]

    user = _USER.format(
        cause=(cause or "").upper(),
        cause_title=CAUSE_TITLE.get((cause or "").upper(), "未分类"),
        target=target,
        purpose=W.PROMPT_WHITELIST.get(target, "未登记用途"),
        current=(current or "")[:4000],
        samples="\n".join(sample_lines),
    )
    try:
        resp = gw.chat_detailed(_SYSTEM, user, task=task)
        text = (resp.text or "").strip()
    except Exception:
        # 未配置 improvement 任务时回落默认链再试一次
        try:
            resp = gw.chat_detailed(_SYSTEM, user)
            text = (resp.text or "").strip()
        except Exception:
            return None

    return _parse_llm_json(text)


def _parse_llm_json(text: str) -> Optional[Dict[str, str]]:
    """从 LLM 输出里抠出 JSON。优先整体解析，退化到找第一个 {...} 块。"""
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
    except Exception:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None
    if not isinstance(obj, dict):
        return None
    new_system = obj.get("new_system")
    if not isinstance(new_system, str) or not new_system.strip():
        return None
    return {"rationale": str(obj.get("rationale", ""))[:500],
            "new_system": new_system}


# ============================================================================
# 对外主入口
# ============================================================================

def propose_for_cause(
    cause: str,
    target: str,
    current_text: str,
    samples: Optional[List[Dict[str, Any]]] = None,
    parent_id: Optional[str] = None,
    use_llm: bool = True,
    llm_calls: Optional[List[int]] = None,
    verbose: bool = False,
) -> Optional[ImprovementProposal]:
    """
    针对一个根因 + 一个可改对象，产出一条提案。

    护栏前置：目标不在白名单 / 触碰不可变 → 直接返回 None（不生成、不入库）。
    """
    cause = (cause or "").strip().upper()
    try:
        layer = W.validate_target(target)
    except W.TargetRejected as e:
        if verbose:
            print(f"  [generator] 目标被拒：{e}")
        return None

    if layer != W.LAYER_PROMPT:
        # 本版只做提示词层的自动生成（参数层与经验层由 runner 显式构造）
        if verbose:
            print(f"  [generator] 暂不自动生成 {layer} 层提案")
        return None

    current = current_text or ""
    rationale = ""
    new_text: Optional[str] = None

    if use_llm:
        # 预算闸：单轮 LLM 调用上限
        budget = llm_calls if llm_calls is not None else []
        if len(budget) >= BUDGET_MAX_LLM_CALLS_PER_ROUND:
            if verbose:
                print("  [generator] 已达单轮 LLM 调用预算，转确定性变异")
        else:
            budget.append(1)
            got = llm_mutate(cause, target, current, samples or [])
            if got:
                new_text = got["new_system"]
                rationale = got["rationale"]
                rationale = f"[LLM] {rationale}"

    if new_text is None:
        new_text = deterministic_mutate(cause, target, current)
        if new_text is None:
            if verbose:
                print(f"  [generator] {cause} 对 {target} 无适用变异算子")
            return None
        rationale = f"[确定性变异] 针对 {cause} 追加约束：{MUTATION_OPERATORS.get(cause, '')[:40]}..."

    # 改动必须真的发生
    if new_text.strip() == current.strip():
        if verbose:
            print("  [generator] 改动后内容未变化，丢弃")
        return None

    return ImprovementProposal.new(
        root_cause=cause,
        change_layer=W.LAYER_PROMPT,
        change_target=target,
        payload_before=current,
        payload_after=new_text,
        rationale=rationale,
        parent_id=parent_id,
    )


def plan_targets(cause: str, layer: Optional[str] = None,
                 limit: int = 2) -> List[str]:
    """给定根因，返回本轮打算动哪些可改对象（已过滤不可变）。"""
    ts = W.targets_for_cause(cause, layer=layer or W.LAYER_PROMPT)
    return ts[:limit]
