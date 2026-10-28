# -*- coding: utf-8 -*-
"""
improvement.schema —— 提案数据结构、状态机与门禁阈值
================================================================================

一次「我认为该怎么改」的完整记录，以及判定它能不能合并的全部常量。

设计要点
--------
1. **payload_before / payload_after 必须同在**：没有 before 就没有回滚，
   缺 before 的提案在入库前就该被拒（见 store.save）。
2. **dev_score 与 holdout_score 分开存**：只有 holdout（冻结集）能作为合并依据，
   dev 只用于提案生成阶段的试错。这是防过拟合的核心（T2）。
3. **状态机单向推进，不回退**：rolled_back 是终态，想再来一次得开新提案，
   父代指向旧提案（parent_id），形成 DGM 式演化树。
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# ============================================================================
# 状态机
# ============================================================================

ST_DRAFT = "draft"              # 已生成，未验证
ST_VALIDATING = "validating"    # 沙箱 A/B 进行中
ST_PASSED = "passed"            # 门禁全过，等待裁决
ST_REJECTED = "rejected"        # 门禁未过（终态，但仍留在 archive 里当父代候选）
ST_MERGED = "merged"            # 已合并生效
ST_ROLLED_BACK = "rolled_back"  # 合并后被回滚（终态）

ALL_STATUSES = (ST_DRAFT, ST_VALIDATING, ST_PASSED,
                ST_REJECTED, ST_MERGED, ST_ROLLED_BACK)

# 合法流转
_TRANSITIONS = {
    ST_DRAFT: (ST_VALIDATING, ST_REJECTED),
    ST_VALIDATING: (ST_PASSED, ST_REJECTED),
    ST_PASSED: (ST_MERGED, ST_REJECTED),
    ST_MERGED: (ST_ROLLED_BACK,),
    ST_REJECTED: (),
    ST_ROLLED_BACK: (),
}


def can_transit(src: str, dst: str) -> bool:
    return dst in _TRANSITIONS.get(src, ())


# ============================================================================
# 门禁阈值（T4）
# ============================================================================

# 冻结集净增门槛（百分点）。低于此值视为噪声，不接受。
MIN_HOLDOUT_RECALL_GAIN_PP = 2.0
MIN_HOLDOUT_NDCG_GAIN_PP = 1.5
# 答案层（P1/P3 提示词跑 answer suite）对应门槛：忠实度与相关性。
# 忠实度是「幻觉」的直接量化，门槛不低于检索层召回。
MIN_HOLDOUT_FAITH_GAIN_PP = 2.0
MIN_HOLDOUT_RELEV_GAIN_PP = 1.5

# 冻结集最小样本量。低于此值判定为「统计效力不足」，禁止自动合并，只能人工审批。
# 这是诚实处理：当前 retrieval 黄金集只有 15 条数据行，切完 holdout 远达不到 30。
MIN_HOLDOUT_CASES = 30

# 零容忍项
ZERO_TOLERANCE_CAUSES = ("R8",)          # 跨租户泄漏
NO_REGRESSION_CAUSES = ("R5", "R6")      # 幻觉 / 答非所问：不允许「用一个新坑换旧坑」

# 延迟回归上限（p95 增幅百分比）
MAX_P95_LATENCY_INC_PCT = 10.0

# 预算闸（T6）
BUDGET_MAX_LLM_CALLS_PER_ROUND = 20
BUDGET_MAX_DEV_CASES = 50
BUDGET_MAX_HOLDOUT_CASES = 30
BUDGET_MAX_AUTO_MERGE_PER_DAY = 3

# 同一个 change_target 被回滚达到此次数后，进黑名单，不再接受自动提案
ROLLBACK_BLACKLIST_THRESHOLD = 2


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class ImprovementProposal:
    """一次改进提案。"""

    proposal_id: str
    root_cause: str                       # R1..R8
    change_layer: str                     # P1 / P2 / P3
    change_target: str                    # 例：generate_answer / top_k / playbook
    payload_before: str                   # 变更前快照（回滚的唯一依据，不可为空）
    payload_after: str                    # 变更后内容
    rationale: str = ""                   # LLM 给出的改的理由
    parent_id: Optional[str] = None       # 演化树父代
    status: str = ST_DRAFT

    dev_score: Dict[str, Any] = field(default_factory=dict)
    holdout_score: Dict[str, Any] = field(default_factory=dict)
    baseline_run_id: str = ""
    variant_run_id: str = ""

    created_at: float = field(default_factory=time.time)
    decided_at: Optional[float] = None
    decided_by: str = ""                  # 'auto' 或操作者标识

    # ---- 构造 ----

    @staticmethod
    def new(root_cause: str, change_layer: str, change_target: str,
            payload_before: str, payload_after: str, rationale: str = "",
            parent_id: Optional[str] = None) -> "ImprovementProposal":
        pid = "pp-" + time.strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:6]
        return ImprovementProposal(
            proposal_id=pid,
            root_cause=(root_cause or "").strip().upper(),
            change_layer=change_layer,
            change_target=change_target,
            payload_before=payload_before,
            payload_after=payload_after,
            rationale=rationale,
            parent_id=parent_id,
        )

    # ---- 指纹 ----

    def fingerprint(self) -> str:
        """内容指纹：同一 target + 同一 after 只保留一条，防重复提案。"""
        raw = f"{self.change_target}\x00{self.payload_after}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    # ---- 状态推进 ----

    def transit(self, dst: str, by: str = "") -> bool:
        if not can_transit(self.status, dst):
            return False
        self.status = dst
        if dst in (ST_PASSED, ST_REJECTED, ST_MERGED, ST_ROLLED_BACK):
            self.decided_at = time.time()
            # by 为空时保留原操作者：回滚通常只记动作、不覆盖「当初谁合的」，
            # 否则复盘时会丢掉最该知道的那个名字。
            if by:
                self.decided_by = by
        return True

    # ---- 序列化 ----

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ImprovementProposal":
        known = {f for f in ImprovementProposal.__dataclass_fields__}  # type: ignore[attr-defined]
        return ImprovementProposal(**{k: v for k, v in (d or {}).items() if k in known})


@dataclass
class GateEvent:
    """一次门禁判定的留痕。出事后可复盘：到底卡在哪一道。"""

    proposal_id: str
    gate: str          # immutable_guard / holdout_gain / no_regression / budget / sample_size
    passed: bool
    detail: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# 指标 diff
# ============================================================================

# 从 EvalRun.summary 里取数的键（对齐 evalkit/schema.compute_* 与 harness_answer 的输出命名）。
# 2026-10-26 实测补：原先只有检索层四项，导致 P1/P3（提示词层，必须跑 answer 层）
# 在门禁眼里永远「指标缺失」。这里补上 answer 层四项。
METRIC_KEYS = ("recall@5", "ndcg@5", "mrr", "p95_latency_ms",
               "faithfulness", "relevancy", "refuse_accuracy",
               "hard_pass_rate", "pass_rate")

# 比率型指标：evalkit 输出 0–1 口径（如 recall@5=0.875），而门禁门槛与单测一律用
# 0–100 百分点口径（如 60.0 表示 60%）。两套口径没对齐时，真实数据永远达不到
# 「+2.0」的门槛 —— 2026-10-26 实测抓到。extract_metrics 负责统一到百分点。
RATIO_METRIC_KEYS = frozenset({
    "recall@5", "ndcg@5", "mrr", "faithfulness", "relevancy",
    "refuse_accuracy", "hard_pass_rate", "pass_rate",
})

# 门禁主指标按评测层分流（gate.G3 用）：
#   检索层（P2 参数）看召回与排序；答案层（P1/P3 提示词）看忠实度与相关性。
# 选错评测层会得假阴性 —— 提示词变体在检索层差值精确为 0（2026-10-26 实测）。
GATE_METRICS_BY_SUITE = {
    "retrieval": (("recall@5", MIN_HOLDOUT_RECALL_GAIN_PP),
                  ("ndcg@5", MIN_HOLDOUT_NDCG_GAIN_PP)),
    "answer": (("faithfulness", MIN_HOLDOUT_FAITH_GAIN_PP),
               ("relevancy", MIN_HOLDOUT_RELEV_GAIN_PP)),
}


def _to_percent_point(key: str, v: float) -> float:
    """把比率型指标统一到 0–100 百分点口径。

    evalkit 输出 0–1（recall@5=0.875），门禁门槛与单测用 0–100（87.5）。
    只对「比率型且当前值 <= 1.0」的指标换算；已经是百分点的（如 60.0）不动，
    延迟类（p95_latency_ms）与计数类一律不动。
    """
    if key in RATIO_METRIC_KEYS and -1.0 <= v <= 1.0:
        return round(v * 100.0, 4)
    return v


def extract_metrics(run_dict: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    从一个 EvalRun.to_dict() 里抽出门禁关心的指标。

    取不到就返回空 dict —— 门禁侧会把「取不到指标」当成验证失败，
    而不是当成 0 分通过。宁可拦下，不可误放。
    """
    if not run_dict:
        return {}
    summary = (run_dict or {}).get("summary") or {}
    out: Dict[str, float] = {}
    for k in METRIC_KEYS:
        v = summary.get(k)
        if v is None:
            # 兼容嵌套写法：summary.metrics.recall@5
            m = summary.get("metrics")
            if isinstance(m, dict):
                v = m.get(k)
        if isinstance(v, (int, float)):
            out[k] = _to_percent_point(k, float(v))
    # 安全相关：R8 计数与失败 case 数
    for k in ("r8_count", "failures", "cases"):
        v = summary.get(k)
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def diff_metrics(base: Dict[str, float],
                 variant: Dict[str, float]) -> Dict[str, float]:
    """变体 - 基线。Recall/nDCG 按百分点计（输入已是 0-100 口径则直接相减）。"""
    out: Dict[str, float] = {}
    for k in set(base) | set(variant):
        b, v = base.get(k), variant.get(k)
        if b is None or v is None:
            continue
        out[k] = round(v - b, 4)
    return out


def count_causes(run_dict: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """统计一次 run 里的根因分布（依赖 evalkit.triage.triage_run）。"""
    if not run_dict:
        return {}
    try:
        from evalkit import triage as T
        t = T.triage_run(run_dict)
        return dict(t.get("by_code") or {})
    except Exception:
        return {}
