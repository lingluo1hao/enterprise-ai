"""
evalkit.runner —— 评测命令行入口
================================================================================

把整个 evalkit 串起来：加载黄金集 → 跑 harness → 落盘 run → 生成 HTML 报表
→ 可选基线 diff → 输出 bad case 根因分类。

【典型用法】

  # 检索层（零 LLM 成本，秒级，改检索参数后常驻跑）
  python -m evalkit.runner --suite retrieval --mode pipeline
  python -m evalkit.runner --suite retrieval --mode raw        # 只看召回侧
  python -m evalkit.runner --suite retrieval --compare last     # 和上次比

  # 答案层（跑完整链路 + judge，分钟级，发版前跑）
  python -m evalkit.runner --suite answer
  python -m evalkit.runner --suite answer --judge-task evalgrade --fast

  # 两层都跑
  python -m evalkit.runner --suite both

  # 自定义黄金集 / 检索深度 / 报告名
  python -m evalkit.runner --suite retrieval --golden ./my.jsonl --fetch-k 25
  python -m evalkit.runner --suite retrieval --report ./out.html

【退出码】
  0 = 全部通过（或检索无失败）
  2 = 存在失败 case（便于 CI 门禁；可用 --no-fail 关闭）
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalkit import schema as S          # noqa: E402
from evalkit import report as R          # noqa: E402
from evalkit import triage as T          # noqa: E402


# ============================================================================
# 子流程
# ============================================================================

def _run_retrieval(args, cases) -> S.EvalRun:
    from evalkit.harness_retrieval import run_suite
    return run_suite(
        cases, mode=args.mode, fetch_k=args.fetch_k,
        overrides=_parse_overrides(args), note=args.note, quiet=not args.verbose,
    )


def _run_answer(args, cases) -> S.EvalRun:
    from evalkit.harness_answer import AnswerHarness
    h = AnswerHarness(fast_mode=args.fast, judge_task=args.judge_task)
    return h.run_suite(cases, note=args.note, quiet=not args.verbose)


def _parse_overrides(args) -> dict:
    """把 --set hybrid=0 --set top_k=8 这类覆盖解析成 dict。"""
    ov = {}
    for kv in (args.set or []):
        if "=" not in kv:
            print(f"[warn] 忽略无效覆盖（需 key=value）: {kv}")
            continue
        k, v = kv.split("=", 1)
        # 自动转 bool/int
        if v.lower() in ("true", "false"):
            v = (v.lower() == "true")
        else:
            try:
                v = int(v)
            except ValueError:
                pass
        ov[k] = v
    return ov


def _resolve_baseline(suite: str, spec: str):
    if not spec:
        return None
    if spec == "last":
        runs = S.list_runs(suite)
        if not runs:
            print("[warn] 没有历史 run 可作基线对比")
            return None
        return S.load_run(runs[0])
    path = spec if spec.endswith(".json") else None
    if path is None:
        # 当成 run_id 前缀模糊匹配
        for p in S.list_runs(suite):
            if spec in os.path.basename(p):
                path = p
                break
    if path and os.path.isfile(path):
        return S.load_run(path)
    print(f"[warn] 基线未找到: {spec}")
    return None


def _print_triage(run: S.EvalRun):
    triage = T.triage_run(run.to_dict())
    if triage["failures"] == 0:
        return
    print(f"\n[evalkit] Bad Case 根因分布（{triage['failures']} 条失败）")
    for code in sorted(triage["by_code"]):
        n = triage["by_code"][code]
        title, sev, _, _ = T._ROOT_CAUSES[code]
        print(f"  {code} {title} [{sev}] ×{n}")
    # 打印前 3 条明细
    for it in triage["items"][:3]:
        print(f"   - {it['case_id']}: {it['code']} {it['title']} — {it['reason']}")


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="evalkit.runner", description="RAG 评测 Harness 命令行入口")
    ap.add_argument("--suite", choices=["retrieval", "answer", "both"],
                    default="retrieval", help="跑哪一层（默认 retrieval）")
    ap.add_argument("--mode", choices=["raw", "pipeline"], default="pipeline",
                    help="检索模式：raw=仅向量库，pipeline=raw+融合+精排")
    ap.add_argument("--golden", default=None,
                    help="黄金集路径（默认 evalkit/golden/<suite>.jsonl）")
    ap.add_argument("--fetch-k", type=int, default=20,
                    help="检索深度（需 > 线上 top_k 才能算出真实 bury）")
    ap.add_argument("--compare", default=None,
                    help="基线 run：'last' 或 run_id/路径，报表出 diff")
    ap.add_argument("--report", default=None, help="HTML 报表输出路径")
    ap.add_argument("--note", default="", help="本次 run 备注")
    ap.add_argument("--judge-task", default="evalgrade",
                    help="答案层 judge 路由任务（默认 evalgrade）")
    ap.add_argument("--fast", action="store_true",
                    help="答案层用规则分类（不调 LLM 分类节点）")
    ap.add_argument("--set", action="append",
                    help="配置覆盖 key=value，可重复（如 --set hybrid=0 --set top_k=8）")
    ap.add_argument("--verbose", action="store_true", help="显示底层管线日志")
    ap.add_argument("--no-fail", action="store_true",
                    help="即使有失败 case 也返回退出码 0（CI 默认会因失败非零退出）")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    suites = (["retrieval", "answer"] if args.suite == "both"
              else [args.suite])

    any_failure = False
    baseline_by_suite = {}

    for suite in suites:
        # ---- 选黄金集 ----
        golden_path = args.golden or (
            S.DEFAULT_ANSWER_GOLDEN if suite == "answer"
            else S.DEFAULT_RETRIEVAL_GOLDEN)
        if not os.path.isfile(golden_path):
            print(f"[evalkit] 黄金集不存在：{golden_path}（跳过 {suite}）")
            continue

        cases = (S.load_answer_cases(golden_path) if suite == "answer"
                 else S.load_retrieval_cases(golden_path))
        print(f"[evalkit] 载入 {len(cases)} 条 {suite} 黄金集：{golden_path}")

        # ---- 基线 ----
        base = _resolve_baseline(suite, args.compare) if args.compare else None
        if base is not None:
            baseline_by_suite[suite] = base

        # ---- 跑 ----
        t0 = time.time()
        if suite == "answer":
            run = _run_answer(args, cases)
        else:
            run = _run_retrieval(args, cases)
        elapsed = time.time() - t0

        # ---- 落盘 + 报表 ----
        run_path = S.save_run(run)
        base_for_report = baseline_by_suite.get(suite)
        report_path = R.write_report(run.to_dict(), base_for_report, args.report)
        print(f"[evalkit] run 落盘：{run_path}")
        print(f"[evalkit] 报表生成：{report_path}（{elapsed:.1f}s）")

        # ---- 门禁 ----
        # 两个 suite 都以 harness 落盘的 _passed 为准：检索层的正例（看召回）
        # 与隔离负例（看有没有泄漏）判定方向相反，不能再用 bury 一刀切。
        failed = sum(1 for r in run.results if not r.get("_passed"))
        if failed:
            any_failure = True
            print(f"[evalkit] ⚠ {suite} 有 {failed} 条失败 case")
            _print_triage(run)

    return 0 if (args.no_fail or not any_failure) else 2


if __name__ == "__main__":
    sys.exit(main())
