# -*- coding: utf-8 -*-
"""
improvement.gate —— 六道安全闸与分级裁决
================================================================================

每一道闸都对应一个真实发生过的失败模式，不是凭空设的护栏：

  G1 immutable_guard  对应 DGM 删检测日志刷分：优化器会去找评分器的漏洞
  G2 sample_size      对应「记忆力通胀」：样本不够时，涨的可能是噪声不是能力
  G3 holdout_gain     只看冻结集，dev 集的成绩不作数（防过拟合到优化集）
  G4 no_regression    R8 零容忍；不许用新增 R5/R6 去换别的指标
  G5 blacklist        同一个目标被回滚两次就拉黑，别再让优化器在这儿打转
  G6 budget           对应「一夜烧掉一个月 token」：自动合并数量必须有上限

裁决分级（正面处理「判定权在人手上」这条既有设计决定）
------------------------------------------------------
  auto_merge  提示词层 / 经验层，且全闸通过、非首次改动该目标、样本量达标
  manual      参数层、首次改动的目标、或样本量不足
  reject      触碰护栏 / 指标倒退 / 触发零容忍
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import whitelist as W
from .schema import (
    ImprovementProposal, GateEvent,
    MIN_HOLDOUT_RECALL_GAIN_PP, MIN_HOLDOUT_NDCG_GAIN_PP, MIN_HOLDOUT_CASES,
    ZERO_TOLERANCE_CAUSES, NO_REGRESSION_CAUSES, MAX_P95_LATENCY_INC_PCT,
    BUDGET_MAX_AUTO_MERGE_PER_DAY, GATE_METRICS_BY_SUITE,
)

# 裁决
DECISION_AUTO = "auto_merge"
DECISION_MANUAL = "manual"
DECISION_REJECT = "reject"

# 闸名
G_IMMUTABLE = "immutable_guard"
G_SAMPLE = "sample_size"
G_HOLDOUT = "holdout_gain"
G_REGRESSION = "no_regression"
G_BLACKLIST = "blacklist"
G_BUDGET = "budget"


@dataclass
class GateResult:
    """一次门禁判定的完整结果。"""
    passed: bool
    decision: str                              # auto_merge / manual / reject
    events: List[GateEvent] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def failed_gates(self) -> List[str]:
        return [e.gate for e in self.events if not e.passed]

    def summary_line(self) -> str:
        mark = {DECISION_AUTO: "自动合并", DECISION_MANUAL: "人工审批",
                DECISION_REJECT: "拒绝"}.get(self.decision, self.decision)
        extra = ""
        if self.reasons:
            extra = "；" + "；".join(self.reasons[:3])
        return f"{mark}{extra}"


def _ev(pid: str, gate: str, passed: bool, detail: str) -> GateEvent:
    return GateEvent(proposal_id=pid, gate=gate, passed=passed, detail=detail)


def evaluate(
    p: ImprovementProposal,
    baseline_metrics: Dict[str, float],
    variant_metrics: Dict[str, float],
    holdout_cases: int,
    baseline_causes: Optional[Dict[str, int]] = None,
    variant_causes: Optional[Dict[str, int]] = None,
    store=None,
    seen_target_before: bool = False,
    verbose: bool = False,
) -> GateResult:
    """
    对一条已完成沙箱 A/B 的提案做门禁判定。

    参数
    ----
    baseline_metrics / variant_metrics  冻结集上基线 / 变体的聚合指标
    holdout_cases                       冻结集样本量（G2 用）
    baseline_causes / variant_causes    两次 run 的根因分布（G4 用）
    store                               ImprovementStore，用于黑名单与预算闸
    seen_target_before                  该 change_target 此前是否改过（首次改动需人工）
    """
    baseline_causes = baseline_causes or {}
    variant_causes = variant_causes or {}
    events: List[GateEvent] = []
    reasons: List[str] = []
    hard_fail = False
    low_power = False

    # ---------- G1 不可变护栏 ----------
    try:
        W.validate_target(p.change_target)
        events.append(_ev(p.proposal_id, G_IMMUTABLE, True,
                          f"{p.change_target} 在白名单内（{p.change_layer}）"))
    except W.TargetRejected as e:
        events.append(_ev(p.proposal_id, G_IMMUTABLE, False, str(e)))
        reasons.append(f"G1 {e}")
        hard_fail = True

    # ---------- G2 样本量 ----------
    if holdout_cases >= MIN_HOLDOUT_CASES:
        events.append(_ev(p.proposal_id, G_SAMPLE, True,
                          f"冻结集 {holdout_cases} 条，达标（>= {MIN_HOLDOUT_CASES}）"))
    else:
        low_power = True
        events.append(_ev(
            p.proposal_id, G_SAMPLE, False,
            f"冻结集仅 {holdout_cases} 条，低于 {MIN_HOLDOUT_CASES} 条门槛："
            f"统计效力不足，结论不可靠，禁止自动合并"))
        reasons.append(f"G2 冻结集 {holdout_cases} < {MIN_HOLDOUT_CASES}，统计效力不足")

    # ---------- G3 冻结集净增 ----------
    def _delta(key: str) -> Optional[float]:
        b, v = baseline_metrics.get(key), variant_metrics.get(key)
        if b is None or v is None:
            return None
        return round(v - b, 4)

    # 主指标按评测层分流：P2 参数变体只在检索层看得出效果，P1/P3 提示词变体
    # 必须看答案层（检索层差值实测精确为 0，会得假阴性）。
    suite_key = "retrieval" if p.change_layer == W.LAYER_PARAM else "answer"
    pairs = GATE_METRICS_BY_SUITE.get(suite_key) or GATE_METRICS_BY_SUITE["answer"]
    deltas = [(k, th, _delta(k)) for k, th in pairs]

    if all(d is None for _, _, d in deltas):
        events.append(_ev(p.proposal_id, G_HOLDOUT, False,
                          f"suite={suite_key} 取不到冻结集指标，按失败处理"
                          f"（宁可拦下，不可误放）"))
        reasons.append(f"G3 冻结集指标缺失（suite={suite_key}）")
        hard_fail = True
    else:
        measured = [(k, th, d) for k, th, d in deltas if d is not None]
        passed = (bool(measured)
                  and any(d >= th for _, th, d in measured)
                  and all(d > 0 for _, _, d in measured))
        detail = " / ".join(f"{k} {d}（门槛 +{th}）" for k, th, d in deltas)
        events.append(_ev(p.proposal_id, G_HOLDOUT, passed,
                          f"suite={suite_key} {detail}"))
        if not passed:
            reasons.append(f"G3 冻结集净增不足：{detail}")
            hard_fail = True

    # ---------- G4 零容忍与回归 ----------
    reg_problems: List[str] = []
    for c in ZERO_TOLERANCE_CAUSES:
        n = int(variant_causes.get(c, 0))
        if n > 0:
            reg_problems.append(f"{c} 出现 {n} 条（零容忍）")
    for c in NO_REGRESSION_CAUSES:
        b = int(baseline_causes.get(c, 0))
        v = int(variant_causes.get(c, 0))
        if v > b:
            reg_problems.append(f"{c} 由 {b} 增至 {v}")
    b_lat = baseline_metrics.get("p95_latency_ms")
    v_lat = variant_metrics.get("p95_latency_ms")
    if b_lat and v_lat and b_lat > 0:
        inc = (v_lat - b_lat) / b_lat * 100.0
        if inc > MAX_P95_LATENCY_INC_PCT:
            reg_problems.append(
                f"p95 延迟 +{inc:.1f}%（上限 {MAX_P95_LATENCY_INC_PCT}%）")
    if reg_problems:
        events.append(_ev(p.proposal_id, G_REGRESSION, False, "；".join(reg_problems)))
        reasons.append("G4 " + "；".join(reg_problems))
        hard_fail = True
    else:
        events.append(_ev(p.proposal_id, G_REGRESSION, True,
                          "无 R8，无新增 R5/R6，延迟未超阈值"))

    # ---------- G5 黑名单 ----------
    if store is not None and hasattr(store, "is_blacklisted"):
        if store.is_blacklisted(p.change_target):
            events.append(_ev(p.proposal_id, G_BLACKLIST, False,
                              f"{p.change_target} 已被回滚多次，进黑名单"))
            reasons.append(f"G5 {p.change_target} 在黑名单")
            hard_fail = True
        else:
            events.append(_ev(p.proposal_id, G_BLACKLIST, True, "未拉黑"))
    else:
        events.append(_ev(p.proposal_id, G_BLACKLIST, True, "无 store，跳过"))

    # ---------- G6 预算 ----------
    if store is not None and hasattr(store, "today_auto_merge_count"):
        used = store.today_auto_merge_count()
        if used >= BUDGET_MAX_AUTO_MERGE_PER_DAY:
            events.append(_ev(p.proposal_id, G_BUDGET, False,
                              f"今日已自动合并 {used} 条，达上限 {BUDGET_MAX_AUTO_MERGE_PER_DAY}"))
            reasons.append(f"G6 今日自动合并已达上限 {used}")
            hard_fail = True
        else:
            events.append(_ev(p.proposal_id, G_BUDGET, True,
                              f"今日已用 {used}/{BUDGET_MAX_AUTO_MERGE_PER_DAY}"))
    else:
        events.append(_ev(p.proposal_id, G_BUDGET, True, "无 store，跳过"))

    # ---------- 裁决 ----------
    if hard_fail:
        decision = DECISION_REJECT
    elif low_power:
        # 样本不足：不拒绝，但不给自动放行，交给人
        decision = DECISION_MANUAL
    elif p.change_layer == W.LAYER_PARAM:
        decision = DECISION_MANUAL
        reasons.append("参数层变更一律人工审批")
    elif not seen_target_before:
        decision = DECISION_MANUAL
        reasons.append(f"首次改动 {p.change_target}，需人工审批")
    elif p.change_layer in (W.LAYER_PROMPT, W.LAYER_PLAYBOOK):
        decision = DECISION_AUTO
    else:
        decision = DECISION_MANUAL

    mdelta = {f"delta_{k}": d for k, _, d in deltas}
    mdelta["suite"] = suite_key
    mdelta["holdout_cases"] = holdout_cases
    mdelta["baseline_causes"] = baseline_causes
    mdelta["variant_causes"] = variant_causes

    result = GateResult(
        passed=(not hard_fail),
        decision=decision,
        events=events,
        reasons=reasons,
        metrics=mdelta,
    )
    if verbose:
        print(f"  [gate] {p.proposal_id} -> {result.summary_line()}")
    return result
