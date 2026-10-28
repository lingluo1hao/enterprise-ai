# -*- coding: utf-8 -*-
"""
improvement.runner —— 自改进闭环命令行入口
================================================================================

【典型用法】

  # 0) 看当前 bad case 的根因分布（P0 接线是否打通）
  python -m improvement.runner triage

  # 1) 切冻结集（只需做一次；切完请人工审阅并补充到 30 条以上）
  python -m improvement.runner split --suite retrieval

  # 2) 针对某个根因生成提案（离线可用 --no-llm 走确定性变异）
  python -m improvement.runner propose --cause R5 --limit 2 --no-llm

  # 3) 跑沙箱 A/B + 门禁判定
  python -m improvement.runner validate <proposal_id>

  # 4) 合并 / 回滚 / 查看
  python -m improvement.runner merge <proposal_id>
  python -m improvement.runner rollback <proposal_id>
  python -m improvement.runner list
  python -m improvement.runner status

【退出码】
  0 = 正常
  2 = 门禁拒绝或执行失败
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from improvement import whitelist as W          # noqa: E402
from improvement import schema as SC            # noqa: E402
from improvement import store as ST             # noqa: E402
from improvement import sandbox as SB           # noqa: E402
from improvement import gate as GT              # noqa: E402


# ============================================================================
# 子命令
# ============================================================================

def cmd_triage(args) -> int:
    """P0 接线检查：bad case 的根因分布。"""
    try:
        from memory_store import get_memory_store
        ms = get_memory_store()
        if not getattr(ms, "available", False):
            print("[triage] memory_store 不可用（降级内存态），读到的 bad_cases 可能为空")
        rows = ms.list_bad_cases(limit=200)
    except Exception as e:
        print(f"[triage] 读取失败: {type(e).__name__}: {e}")
        return 2

    if not rows:
        if not getattr(ms, "available", False):
            print("[triage] bad_cases 为空，但**库本身不可用**，"
                  "因此不能据此判断接线是否打通 —— 请在库在线时重试")
        else:
            print("[triage] bad_cases 为空 —— P0 接线未打通："
                  "主链路失败样本没有落进 bad_cases，闭环起点就断了")
        return 2

    dist: Dict[str, int] = {}
    for r in rows:
        c = (r.get("root_cause") or "").strip().upper() or "未分类"
        dist[c] = dist.get(c, 0) + 1
    print(f"[triage] bad_cases 共 {len(rows)} 条，根因分布：")
    for c in sorted(dist, key=lambda k: -dist[k]):
        title = {"R1": "召回丢失", "R2": "排序埋没", "R3": "改写/精排负优化",
                 "R4": "权限误杀", "R5": "生成幻觉", "R6": "答非所问",
                 "R7": "拒答错误", "R8": "跨租户泄漏"}.get(c, "")
        print(f"  {c} {title} ×{dist[c]}")
    top = sorted(dist.items(), key=lambda kv: -kv[1])[:3]
    print(f"[triage] Top3 根因：{', '.join(f'{k}({v})' for k, v in top)}")
    for k, _ in top:
        ts = W.targets_for_cause(k)
        if ts:
            print(f"  {k} 建议动：{', '.join(ts)}")
    return 0


def cmd_split(args) -> int:
    dev, hold = SB.split_and_persist(args.suite, args.golden, args.ratio)
    print(f"[split] dev {len(dev)} 条 / holdout {len(hold)} 条")
    if len(hold) < SC.MIN_HOLDOUT_CASES:
        print(f"[split] 冻结集不足 {SC.MIN_HOLDOUT_CASES} 条："
              f"门禁不会自动放行，需人工补 case 后再跑")
    return 0


def cmd_propose(args) -> int:
    store = ST.ImprovementStore()
    cause = (args.cause or "").strip().upper()

    targets = [args.target] if args.target else W.targets_for_cause(
        cause, layer=W.LAYER_PROMPT)
    if not targets:
        print(f"[propose] 根因 {cause} 在提示词层没有可改对象"
              f"（要么该改参数，要么属不可变范围）")
        return 2
    targets = targets[:args.limit]

    # 取失败样本
    samples: List[Dict[str, Any]] = []
    try:
        from memory_store import get_memory_store
        ms = get_memory_store()
        if getattr(ms, "available", False):
            samples = ms.list_bad_cases(root_cause=cause, limit=5) or []
    except Exception as e:
        print(f"[propose] 取 bad case 失败（继续，样本为空）: {e}")

    # 取当前提示词
    try:
        from prompt_manager import get_prompt_manager
        pm = get_prompt_manager()
    except Exception as e:
        print(f"[propose] PromptManager 不可用: {e}")
        return 2

    from improvement import generator as G
    llm_calls: List[int] = []
    made = 0
    for t in targets:
        cur = pm.get_prompt(t)
        current_text = (cur or {}).get("system", "") or ""
        if not current_text.strip():
            print(f"[propose] {t} 当前内容为空，跳过")
            continue
        p = G.propose_for_cause(cause, t, current_text, samples,
                                use_llm=not args.no_llm,
                                llm_calls=llm_calls, verbose=True)
        if p is None:
            continue
        if store.available and store.has_fingerprint(p.fingerprint()):
            print(f"[propose] {t} 已存在相同内容的提案，跳过（去重）")
            continue
        store.save_proposal(p)
        made += 1
        print(f"[propose] 已生成 {p.proposal_id}  {p.change_target} "
              f"({len(p.payload_before)} -> {len(p.payload_after)} 字符)")
        print(f"          理由：{p.rationale[:100]}")
    print(f"[propose] 本轮共 {made} 条提案")
    return 0 if made else 2


def cmd_validate(args) -> int:
    store = ST.ImprovementStore()
    p = store.load_proposal(args.proposal_id)
    if p is None:
        print(f"[validate] 找不到提案 {args.proposal_id}")
        return 2

    suite = args.suite or SB.suite_for_layer(p.change_layer)
    print(f"[validate] {p.proposal_id} {p.change_target} "
          f"({p.change_layer} / {p.root_cause}) -> suite={suite}")

    # 冻结集
    golden_hold = args.holdout_golden or (
        SB.HOLDOUT_ANSWER if suite == "answer" else SB.HOLDOUT_RETRIEVAL)
    if os.path.isfile(golden_hold):
        hold = SB.load_cases(suite, golden_hold)
    else:
        print(f"[validate] 未找到冻结集文件 {golden_hold}，改用全量切分（临时）")
        _, hold = SB.split_cases(SB.load_cases(suite))

    p.transit(SC.ST_VALIDATING)
    store.save_proposal(p)

    ab = SB.ab_test(p, hold, suite=suite, fetch_k=args.fetch_k,
                    fast=not args.full, verbose=True)
    if not ab.ok:
        print(f"[validate] 沙箱 A/B 失败：{ab.error}")
        print("[validate] 按门禁口径：指标缺失 = 验证失败，不合并")
        p.transit(SC.ST_REJECTED, by="auto")
        store.save_proposal(p)
        store.add_gate_event(SC.GateEvent(
            p.proposal_id, "sandbox", False, ab.error))
        store.archive(p, active=False)
        return 2

    p.baseline_run_id = ab.baseline_run_id
    p.variant_run_id = ab.variant_run_id
    p.dev_score = {"note": "dev 集不参与门禁判定"}
    p.holdout_score = {"baseline": ab.baseline_metrics,
                       "variant": ab.variant_metrics,
                       "delta": ab.delta(), "cases": ab.holdout_cases}

    seen_before = bool(store.list_proposals(status=SC.ST_MERGED, limit=200))
    g = GT.evaluate(p, ab.baseline_metrics, ab.variant_metrics, ab.holdout_cases,
                    ab.baseline_causes, ab.variant_causes,
                    store=store, seen_target_before=seen_before, verbose=True)
    for ev in g.events:
        store.add_gate_event(ev)

    print(f"[validate] 差值：{ab.delta()}")
    print(f"[validate] 裁决：{g.summary_line()}")

    if g.passed:
        p.transit(SC.ST_PASSED, by="gate")
    else:
        p.transit(SC.ST_REJECTED, by="gate")
    store.save_proposal(p)
    store.archive(p, active=False)
    return 0 if g.passed else 2


def cmd_merge(args) -> int:
    store = ST.ImprovementStore()
    p = store.load_proposal(args.proposal_id)
    if p is None:
        print(f"[merge] 找不到提案 {args.proposal_id}")
        return 2
    if p.status != SC.ST_PASSED:
        print(f"[merge] 提案状态为 {p.status}，只有 passed 才能合并")
        return 2

    if not args.approve:
        print(f"[merge] 该提案需人工确认（--approve 后执行）："
              f"{p.change_target} {p.change_layer}")
        return 2

    if p.change_layer == W.LAYER_PROMPT:
        try:
            from prompt_manager import get_prompt_manager
            pm = get_prompt_manager()
            cur = pm.get_prompt(p.change_target)
            ok = pm.save_prompt(
                p.change_target, p.payload_after,
                user_template=(cur or {}).get("user_template", ""),
                display_name=(cur or {}).get("display_name", p.change_target),
                description=(cur or {}).get("description", ""),
                category=(cur or {}).get("category", "general"),
            )
            if not ok:
                print("[merge] 写库失败")
                return 2
        except Exception as e:
            print(f"[merge] 写库异常: {e}")
            return 2
    else:
        print(f"[merge] {p.change_layer} 层的自动落库尚未实现，需人工执行")
        return 2

    p.transit(SC.ST_MERGED, by=args.by or "manual")
    store.save_proposal(p)
    store.archive(p, active=True)
    print(f"[merge] 已合并 {p.proposal_id} -> {p.change_target}"
          f"（回滚请用 rollback 子命令）")
    return 0


def cmd_rollback(args) -> int:
    store = ST.ImprovementStore()
    p = store.load_proposal(args.proposal_id)
    if p is None:
        print(f"[rollback] 找不到提案 {args.proposal_id}")
        return 2
    if p.change_layer == W.LAYER_PROMPT:
        try:
            from prompt_manager import get_prompt_manager
            pm = get_prompt_manager()
            cur = pm.get_prompt(p.change_target)
            pm.save_prompt(
                p.change_target, p.payload_before,
                user_template=(cur or {}).get("user_template", ""),
                display_name=(cur or {}).get("display_name", p.change_target),
                description=(cur or {}).get("description", ""),
                category=(cur or {}).get("category", "general"))
        except Exception as e:
            print(f"[rollback] 回滚写库异常: {e}")
            return 2
    p.transit(SC.ST_ROLLED_BACK, by=args.by or "manual")
    store.save_proposal(p)
    n = store.bump_rollback(p.proposal_id)
    print(f"[rollback] 已回滚 {p.proposal_id} -> {p.change_target}"
          f"（该目标累计回滚 {n} 次，达到 {SC.ROLLBACK_BLACKLIST_THRESHOLD} 次会拉黑）")
    return 0


def cmd_list(args) -> int:
    store = ST.ImprovementStore()
    rows = store.list_proposals(status=args.status, limit=args.limit)
    if not rows:
        print("[list] 无提案（或库不可用）")
        return 0
    print(f"{'proposal_id':<22} {'cause':<5} {'layer':<5} {'target':<20} {'status':<12} by")
    for r in rows:
        print(f"{r['proposal_id']:<22} {r['root_cause']:<5} {r['change_layer']:<5} "
              f"{r['change_target']:<20} {r['status']:<12} {r['decided_by']}")
    return 0


def cmd_status(args) -> int:
    store = ST.ImprovementStore()
    act = store.active_changes()
    print(f"[status] 生效中的变更 {len(act)} 条")
    for a in act[:20]:
        print(f"  {a['proposal_id']}  {a['change_layer']}  {a['change_target']}"
              f"  {a['status']}  {a['merged_at']}")
    used = store.today_auto_merge_count()
    print(f"[status] 今日自动合并 {used}/{SC.BUDGET_MAX_AUTO_MERGE_PER_DAY}")
    try:
        import prompt_manager
        ov = prompt_manager.active_prompt_overrides()
        if ov:
            print(f"[status] 警告：当前进程存在沙箱覆盖 {list(ov)}，"
                  f"正常情况下线上进程应为空")
        else:
            print("[status] 沙箱覆盖：空（正常）")
    except Exception:
        pass
    return 0


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="improvement.runner",
                                 description="自改进闭环命令行入口")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("triage", help="查看 bad case 根因分布（P0 接线检查）")

    sp = sub.add_parser("split", help="切分冻结集与优化集")
    sp.add_argument("--suite", choices=["retrieval", "answer"], default="retrieval")
    sp.add_argument("--golden", default=None)
    sp.add_argument("--ratio", type=float, default=SB.HOLDOUT_RATIO)

    pp = sub.add_parser("propose", help="针对根因生成提案")
    pp.add_argument("--cause", required=True, help="根因码 R1..R8")
    pp.add_argument("--target", default=None, help="指定提示词名（默认按根因路由）")
    pp.add_argument("--limit", type=int, default=2)
    pp.add_argument("--no-llm", action="store_true", help="离线：走确定性变异算子")

    vp = sub.add_parser("validate", help="沙箱 A/B + 门禁判定")
    vp.add_argument("proposal_id")
    vp.add_argument("--suite", choices=["retrieval", "answer"], default=None)
    vp.add_argument("--holdout-golden", default=None)
    vp.add_argument("--fetch-k", type=int, default=20)
    vp.add_argument("--full", action="store_true", help="答案层跑完整 judge（慢）")

    mp = sub.add_parser("merge", help="合并已通过的提案")
    mp.add_argument("proposal_id")
    mp.add_argument("--approve", action="store_true", help="人工确认后才真正写库")
    mp.add_argument("--by", default="manual")

    rp = sub.add_parser("rollback", help="回滚已合并的提案")
    rp.add_argument("proposal_id")
    rp.add_argument("--by", default="manual")

    lp = sub.add_parser("list", help="列出提案")
    lp.add_argument("--status", default=None)
    lp.add_argument("--limit", type=int, default=50)

    sub.add_parser("status", help="查看生效变更与预算")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    fn = {
        "triage": cmd_triage, "split": cmd_split, "propose": cmd_propose,
        "validate": cmd_validate, "merge": cmd_merge, "rollback": cmd_rollback,
        "list": cmd_list, "status": cmd_status,
    }[args.cmd]
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
