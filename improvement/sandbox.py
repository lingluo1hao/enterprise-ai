# -*- coding: utf-8 -*-
"""
improvement.sandbox —— 沙箱 A/B 编排（L3）
================================================================================

把「改了之后到底变好还是变差」变成一次可复现的机械判定。

关键事实（决定了验证怎么跑）
----------------------------
**提示词变体在检索层评测上完全看不到效果。**
`evalkit/harness_retrieval.py` 走的是 Retriever 直查向量库，不经过任何 prompt；
只有 `harness_answer.py` 跑完整链路才会消费 prompt。因此：

    P1 提示词层 / P3 经验层  -> suite = answer
    P2 参数层                -> suite = retrieval（走 --set 覆盖键）

用错 suite 会得到「改了但指标一动不动」的假阴性，这里按层级强制选对。

双集
----
dev（优化集）与 holdout（冻结集）分开。提案生成阶段只能看 dev，
门禁判定只认 holdout —— 这是防过拟合的唯一有效手段（T2）。

诚实约束
--------
当前 retrieval 黄金集只有 15 条数据行，answer 集 9 条。
切完 holdout 远达不到 MIN_HOLDOUT_CASES=30 的门槛，
G2 闸会把这种情况判成「统计效力不足」，**禁止自动合并**（交人工）。
宁可不给过，也不拿噪声当改进。
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import whitelist as W
from .schema import (
    ImprovementProposal, extract_metrics, count_causes,
    BUDGET_MAX_DEV_CASES, BUDGET_MAX_HOLDOUT_CASES,
)

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "evalkit", "golden")
DEFAULT_RETRIEVAL_GOLDEN = os.path.join(GOLDEN_DIR, "retrieval.jsonl")
DEFAULT_ANSWER_GOLDEN = os.path.join(GOLDEN_DIR, "answer.jsonl")
HOLDOUT_RETRIEVAL = os.path.join(GOLDEN_DIR, "holdout_retrieval.jsonl")
HOLDOUT_ANSWER = os.path.join(GOLDEN_DIR, "holdout_answer.jsonl")
DEV_RETRIEVAL = os.path.join(GOLDEN_DIR, "dev_retrieval.jsonl")
DEV_ANSWER = os.path.join(GOLDEN_DIR, "dev_answer.jsonl")

HOLDOUT_RATIO = 0.3


# ============================================================================
# 切分（确定性，可复现）
# ============================================================================

def _bucket(case_id: str) -> int:
    """按 case_id 做稳定哈希分桶，保证同一条 case 永远落在同一侧。"""
    h = hashlib.md5(str(case_id).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 100


def split_cases(cases: List[Any], ratio: float = HOLDOUT_RATIO) -> Tuple[List[Any], List[Any]]:
    """
    确定性切分。返回 (dev, holdout)。

    做法：按 case_id 的稳定哈希排序，取前 k 条作 holdout（k = round(n*ratio)）。

    为什么不用「按哈希阈值分桶」：阈值分桶在 15 条这种小样本上条数波动很大
    （实测同样 30% 只切出 2 条），而门禁要拿「冻结集有多少条」当硬判据，
    条数必须可预期。排序取前 k 牺牲了一点「新增 case 不影响旧成员」的稳定性，
    换来条数精确 —— 现阶段的瓶颈是样本量不够，不是集合抖动。
    """
    n = len(cases)
    if n == 0:
        return [], []
    k = max(1, min(n - 1 if n > 1 else 1, int(round(n * ratio))))
    ordered = sorted(
        cases,
        key=lambda c: (_bucket(getattr(c, "case_id", None)
                               or (c.get("case_id") if isinstance(c, dict) else "")),
                       str(getattr(c, "case_id", None)
                           or (c.get("case_id") if isinstance(c, dict) else ""))),
    )
    hold = ordered[:k]
    dev = ordered[k:]
    return dev, hold


def load_cases(suite: str, golden_path: Optional[str] = None):
    """复用 evalkit 自己的加载器（不自建等价取数）。"""
    from evalkit import schema as S
    path = golden_path or (DEFAULT_ANSWER_GOLDEN if suite == "answer"
                           else DEFAULT_RETRIEVAL_GOLDEN)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"黄金集不存在：{path}")
    if suite == "answer":
        return S.load_answer_cases(path)
    return S.load_retrieval_cases(path)


def split_and_persist(suite: str, golden_path: Optional[str] = None,
                      ratio: float = HOLDOUT_RATIO, verbose: bool = True
                      ) -> Tuple[List[Any], List[Any]]:
    """切分并把结果落成文件，方便人工审阅与补充（冻结集要人工确认）。"""
    from evalkit import schema as S
    cases = load_cases(suite, golden_path)
    dev, hold = split_cases(cases, ratio)
    if suite == "answer":
        S.save_cases(dev, DEV_ANSWER) if hasattr(S, "save_cases") else None
        S.save_cases(hold, HOLDOUT_ANSWER) if hasattr(S, "save_cases") else None
    else:
        S.save_cases(dev, DEV_RETRIEVAL) if hasattr(S, "save_cases") else None
        S.save_cases(hold, HOLDOUT_RETRIEVAL) if hasattr(S, "save_cases") else None
    if verbose:
        print(f"  [sandbox] {suite}: 共 {len(cases)} 条 -> dev {len(dev)} / holdout {len(hold)}")
        if len(hold) < 30:
            print(f"  [sandbox] 警告：冻结集仅 {len(hold)} 条，低于 30 条门槛，"
                  f"该规模下门禁不会自动放行")
    return dev, hold


# ============================================================================
# 跑一次评测
# ============================================================================

def suite_for_layer(layer: str) -> str:
    """按层级选对评测层。选错会得到假阴性。"""
    return "retrieval" if layer == W.LAYER_PARAM else "answer"


def run_one(suite: str, cases: List[Any], note: str = "",
            overrides: Optional[Dict[str, Any]] = None,
            fetch_k: int = 20, fast: bool = True, judge_task: str = "evalgrade"):
    """跑一次评测，返回 EvalRun。失败抛异常（由调用方决定怎么处理）。"""
    if suite == "answer":
        from evalkit.harness_answer import AnswerHarness
        h = AnswerHarness(fast_mode=fast, judge_task=judge_task)
        return h.run_suite(cases, note=note, quiet=True)
    from evalkit.harness_retrieval import run_suite
    return run_suite(cases, mode="pipeline", fetch_k=fetch_k,
                     overrides=overrides, note=note, quiet=True)


# ============================================================================
# A/B
# ============================================================================

@dataclass
class ABResult:
    ok: bool
    suite: str
    error: str = ""
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    variant_metrics: Dict[str, float] = field(default_factory=dict)
    baseline_causes: Dict[str, int] = field(default_factory=dict)
    variant_causes: Dict[str, int] = field(default_factory=dict)
    holdout_cases: int = 0
    dev_cases: int = 0
    baseline_run_id: str = ""
    variant_run_id: str = ""

    def delta(self) -> Dict[str, float]:
        out = {}
        for k in set(self.baseline_metrics) | set(self.variant_metrics):
            b, v = self.baseline_metrics.get(k), self.variant_metrics.get(k)
            if b is None or v is None:
                continue
            out[k] = round(v - b, 4)
        return out


def ab_test(p: ImprovementProposal, holdout_cases: List[Any],
            dev_cases: Optional[List[Any]] = None,
            suite: Optional[str] = None, fetch_k: int = 20,
            fast: bool = True, verbose: bool = False) -> ABResult:
    """
    在冻结集上跑 基线 vs 变体。

    变体注入方式按层级分：
      P1 提示词 -> prompt_manager.set_prompt_override（进程内，跑完必清）
      P2 参数   -> overrides 传给 Retriever（与 --set 同路径）
      P3 经验   -> 暂无进程内注入口，需先落库再跑（本版不支持，直接报不可用）

    任何一步失败都返回 ok=False，门禁侧按「指标缺失」处理（宁可拦下，不可误放）。
    """
    suite = suite or suite_for_layer(p.change_layer)
    res = ABResult(ok=False, suite=suite,
                   holdout_cases=len(holdout_cases),
                   dev_cases=len(dev_cases or []))

    # 预算闸：验证规模上限
    if len(holdout_cases) > BUDGET_MAX_HOLDOUT_CASES:
        res.error = f"冻结集 {len(holdout_cases)} 条超过预算上限 {BUDGET_MAX_HOLDOUT_CASES}"
        return res
    if len(dev_cases or []) > BUDGET_MAX_DEV_CASES:
        res.error = f"优化集 {len(dev_cases)} 条超过预算上限 {BUDGET_MAX_DEV_CASES}"
        return res

    overrides = None
    if p.change_layer == W.LAYER_PARAM:
        try:
            overrides = {p.change_target: W.cast_param(p.change_target, p.payload_after)}
        except Exception as e:
            res.error = f"参数变体解析失败: {e}"
            return res
    elif p.change_layer == W.LAYER_PLAYBOOK:
        res.error = ("P3 经验层暂无进程内注入口：playbook 必须先落库才能被检索到，"
                     "本版沙箱不支持，请走人工验证")
        return res

    injected = False
    try:
        # ---- 基线 ----
        base_run = run_one(suite, holdout_cases, note=f"baseline for {p.proposal_id}",
                           overrides=None, fetch_k=fetch_k, fast=fast)
        res.baseline_run_id = getattr(base_run, "run_id", "")

        # ---- 注入变体 ----
        if p.change_layer == W.LAYER_PROMPT:
            import prompt_manager
            prompt_manager.set_prompt_override(p.change_target, p.payload_after)
            injected = True

        var_run = run_one(suite, holdout_cases, note=f"variant for {p.proposal_id}",
                          overrides=overrides, fetch_k=fetch_k, fast=fast)
        res.variant_run_id = getattr(var_run, "run_id", "")

        bd, vd = base_run.to_dict(), var_run.to_dict()
        res.baseline_metrics = extract_metrics(bd)
        res.variant_metrics = extract_metrics(vd)
        res.baseline_causes = count_causes(bd)
        res.variant_causes = count_causes(vd)
        res.ok = True
        if verbose:
            print(f"  [sandbox] 基线 {res.baseline_metrics}")
            print(f"  [sandbox] 变体 {res.variant_metrics}")
            print(f"  [sandbox] 差值 {res.delta()}")
    except Exception as e:
        res.error = f"{type(e).__name__}: {e}"
        if verbose:
            print(f"  [sandbox] A/B 失败: {res.error}")
    finally:
        if injected:
            try:
                import prompt_manager
                prompt_manager.clear_prompt_overrides(p.change_target)
            except Exception:
                pass
    return res
