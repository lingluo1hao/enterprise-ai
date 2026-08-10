"""
evalkit.report —— 评测结果 HTML 报表生成
================================================================================

【为什么要 HTML 报表，终端打印不够吗】

终端输出适合"跑的时候看"，不适合"过两周回头看"。
一次评测有 20 条 case、每条 6 个指标、外加召回明细，
终端刷过去就没了，也没法发给同事、没法贴进评审记录。

HTML 报表解决三件事：
  1. 归档：每次 run 存一份，半年后还能查"当时到底什么水平"
  2. 定位：失败 case 直接展开看召回了什么，不用重跑一遍去 debug
  3. 对比：两次 run 并排 diff，改动到底是变好还是变坏一目了然

报表是**自包含单文件**（CSS 内联、无任何外部依赖），
可以直接丢进邮件附件或塞进 CI 产物，打开就能看，不用起服务器。

【配色约定】
指标改善用绿色、退化用红色 —— 这里是技术指标不是股票行情，
沿用国际通用的"绿好红坏"，并在图例中明确标注，避免与 A 股红涨绿跌混淆。
"""

from __future__ import annotations

import html
import json
import os
import time
from typing import Any, Dict, List, Optional

from evalkit.schema import REPORTS_DIR


# ============================================================================
# 样式
# ============================================================================

_CSS = """
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px;
  font-family: -apple-system, "Segoe UI", "Microsoft YaHei", Roboto, sans-serif;
  background: #f5f7fa; color: #1f2937; line-height: 1.6;
}
.wrap { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 24px; margin: 0 0 4px; letter-spacing: -0.3px; }
h2 { font-size: 17px; margin: 32px 0 12px; padding-left: 10px;
     border-left: 3px solid #2563eb; }
.sub { color: #6b7280; font-size: 13px; margin-bottom: 22px; }
.cfg { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
       padding: 12px 16px; font-size: 12.5px; color: #4b5563;
       font-family: "Cascadia Mono", Consolas, monospace; word-break: break-all; }
.cards { display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0 8px; }
.card { flex: 1 1 150px; background: #fff; border: 1px solid #e5e7eb;
        border-radius: 10px; padding: 14px 16px; }
.card .label { font-size: 12px; color: #6b7280; margin-bottom: 6px; }
.card .value { font-size: 26px; font-weight: 650; letter-spacing: -0.5px; }
.card .delta { font-size: 12.5px; margin-top: 4px; font-weight: 600; }
.up { color: #059669; }      /* 指标改善 */
.down { color: #dc2626; }    /* 指标退化 */
.flat { color: #9ca3af; }
table { width: 100%; border-collapse: collapse; background: #fff;
        border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden;
        font-size: 13px; }
th { background: #f9fafb; text-align: left; padding: 9px 12px;
     font-weight: 600; color: #374151; border-bottom: 1px solid #e5e7eb;
     white-space: nowrap; }
td { padding: 9px 12px; border-bottom: 1px solid #f3f4f6; vertical-align: top; }
tr:last-child td { border-bottom: none; }
tr.fail { background: #fef2f2; }
tr.fail:hover { background: #fee2e2; }
tr:hover { background: #f9fafb; }
.pill { display: inline-block; padding: 1px 7px; border-radius: 10px;
        font-size: 11px; font-weight: 600; }
.pill.ok { background: #d1fae5; color: #065f46; }
.pill.no { background: #fee2e2; color: #991b1b; }
.pill.tag { background: #eef2ff; color: #3730a3; margin-right: 4px; }
.q { color: #111827; }
details { margin-top: 6px; }
summary { cursor: pointer; color: #2563eb; font-size: 12px; outline: none; }
.docs { margin-top: 8px; border-left: 2px solid #e5e7eb; padding-left: 10px; }
.doc { font-size: 12px; color: #4b5563; padding: 4px 0;
       border-bottom: 1px dashed #f3f4f6; }
.doc b { color: #111827; }
.hint { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px;
        padding: 12px 16px; font-size: 13px; color: #1e40af; margin: 12px 0; }
.legend { font-size: 12px; color: #6b7280; margin-top: 8px; }
footer { margin-top: 36px; color: #9ca3af; font-size: 12px; text-align: center; }
"""


def _esc(s: Any) -> str:
    return html.escape(str(s if s is not None else ""))


# ============================================================================
# 指标卡
# ============================================================================

# 指标方向：True 表示"越大越好"，False 表示"越小越好"。
# 判断改善/退化必须依赖这张表 —— miss_count 变大是坏事，recall 变大是好事，
# 不区分方向就会把红绿标反，比不标还糟。
_METRIC_DIRECTION = {
    "pass_rate": True, "mrr": True, "avg_bury": False,
    "miss_count": False, "buried_count": False, "avg_latency_ms": False,
    "faithfulness": True, "relevancy": True, "refuse_accuracy": True,
    "hard_pass_rate": True,
}


def _direction(key: str) -> bool:
    if key in _METRIC_DIRECTION:
        return _METRIC_DIRECTION[key]
    if key.startswith("recall@") or key.startswith("ndcg@"):
        return True
    return True


def _fmt(key: str, val: Any) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        if key in ("pass_rate", "refuse_accuracy", "hard_pass_rate"):
            return f"{val:.1%}"
        return f"{val:.3f}"
    return str(val)


def _card(key: str, val: Any, base: Optional[Any] = None) -> str:
    """渲染一张指标卡。给了 base 就同时显示与基线的差值。"""
    delta_html = ""
    if base is not None and isinstance(val, (int, float)) and isinstance(base, (int, float)):
        d = val - base
        if abs(d) < 1e-9:
            delta_html = '<div class="delta flat">— 无变化</div>'
        else:
            better = (d > 0) == _direction(key)
            cls = "up" if better else "down"
            arrow = "▲" if d > 0 else "▼"
            sign = "+" if d > 0 else ""
            word = "改善" if better else "退化"
            dv = f"{sign}{d:.3f}" if isinstance(d, float) else f"{sign}{d}"
            delta_html = f'<div class="delta {cls}">{arrow} {dv} {word}</div>'
    return (f'<div class="card"><div class="label">{_esc(key)}</div>'
            f'<div class="value">{_fmt(key, val)}</div>{delta_html}</div>')


def _cards(summary: Dict[str, Any], base: Optional[Dict[str, Any]] = None,
           keys: Optional[List[str]] = None) -> str:
    keys = keys or [k for k in summary.keys() if k != "count"]
    base = base or {}
    return ('<div class="cards">' +
            "".join(_card(k, summary.get(k), base.get(k)) for k in keys) +
            "</div>")


# ============================================================================
# 检索报表
# ============================================================================

def _retrieval_rows(results: List[Dict[str, Any]]) -> str:
    rows = []
    # 失败的排前面：报表打开就该先看到问题，而不是滚半天找红色行
    ordered = sorted(results, key=lambda r: (r.get("bury", -1) > 0, r.get("bury", 99)))
    for r in ordered:
        bury = r.get("bury", -1)
        ok = bury > 0 and not r.get("error")
        cls = "" if ok else "fail"
        if r.get("error"):
            status = '<span class="pill no">异常</span>'
        elif bury <= 0:
            status = '<span class="pill no">未召回</span>'
        elif bury > 5:
            status = '<span class="pill no">排序过深</span>'
        else:
            status = '<span class="pill ok">通过</span>'

        tags = "".join(f'<span class="pill tag">{_esc(t)}</span>'
                       for t in r.get("tags", []))

        docs = "".join(
            f'<div class="doc"><b>#{d.get("rank")}</b> '
            f'{_esc(d.get("file_name"))} p{_esc(d.get("page"))} — '
            f'{_esc(d.get("preview"))}</div>'
            for d in r.get("retrieved", [])
        ) or '<div class="doc">（无召回结果）</div>'

        err = (f'<div class="doc" style="color:#dc2626">错误：{_esc(r["error"])}</div>'
               if r.get("error") else "")

        rows.append(f"""<tr class="{cls}">
  <td><code>{_esc(r.get('case_id'))}</code></td>
  <td class="q">{_esc(r.get('query'))}<br>{tags}
    <details><summary>查看 top-10 召回</summary>
      <div class="docs">{err}{docs}</div>
    </details>
  </td>
  <td>{status}</td>
  <td>{'未召回' if bury <= 0 else f'第 {bury} 位'}</td>
  <td>{r.get('mrr', 0):.3f}</td>
  <td>{r.get('recall_at_k', {}).get('5', 0):.2f}</td>
  <td>{r.get('ndcg_at_k', {}).get('5', 0):.3f}</td>
  <td>{r.get('latency_ms', 0):.0f}ms</td>
</tr>""")
    return "".join(rows)


def _diagnosis(summary: Dict[str, Any]) -> str:
    """
    把指标翻译成"下一步该干什么"。

    报表只给数字是不够的 —— 看到 recall@5=0.62 大部分人不知道该改哪里。
    这里按 miss / buried 的比例给出明确的排查方向。
    """
    n = summary.get("count", 0) or 1
    miss = summary.get("miss_count", 0)
    buried = summary.get("buried_count", 0)
    tips = []
    if miss / n > 0.2:
        tips.append(f"<b>{miss} 条完全没召回（{miss/n:.0%}）→ 召回侧问题</b>："
                    "优先查 embedding 模型是否匹配中文技术文档、切片是否把答案劈开、"
                    "以及权限 expr 是否误杀。可先用 <code>--mode raw</code> 复跑确认。")
    if buried / n > 0.2:
        tips.append(f"<b>{buried} 条召回了但排在 5 名之外（{buried/n:.0%}）→ 排序侧问题</b>："
                    "检索能力没问题，是排序把正确答案压下去了。查 rerank 服务是否真的生效、"
                    "RRF 融合权重、以及候选池是否过早截断。")
    if not tips:
        tips.append("未发现系统性问题。逐条看下方失败 case 即可。")
    return '<div class="hint">' + "<br><br>".join(tips) + "</div>"


def render_retrieval(run: Dict[str, Any],
                     base: Optional[Dict[str, Any]] = None) -> str:
    """渲染检索评测报表 HTML。base 为基线 run（可选），给了就出 diff。"""
    s = run.get("summary", {})
    bs = (base or {}).get("summary", {})
    cfg = run.get("config", {})
    dur = run.get("finished_at", 0) - run.get("started_at", 0)

    keys = ["pass_rate", "mrr", "recall@1", "recall@5", "ndcg@5",
            "miss_count", "buried_count", "avg_bury", "avg_latency_ms"]
    keys = [k for k in keys if k in s]

    base_line = ""
    if base:
        base_line = (f'<div class="sub">对比基线：<code>{_esc(base.get("run_id"))}</code>'
                     f'（{_esc(json.dumps(base.get("config", {}).get("mode", "?")))}，'
                     f'{len(base.get("results", []))} 条）</div>')

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>检索评测报表 {_esc(run.get('run_id'))}</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
  <h1>检索评测报表</h1>
  <div class="sub">run <code>{_esc(run.get('run_id'))}</code> ·
    {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run.get('started_at', 0)))} ·
    耗时 {dur:.1f}s · {s.get('count', 0)} 条 case</div>
  {base_line}
  <div class="cfg">{_esc(json.dumps(cfg, ensure_ascii=False))}</div>

  <h2>总览</h2>
  {_cards(s, bs, keys)}
  <div class="legend">绿色 = 指标改善，红色 = 指标退化（技术指标口径，非股市红涨绿跌）。
    <code>bury</code> = 第一个正确文档的排名，越小越好，-1 表示完全未召回。</div>

  <h2>诊断建议</h2>
  {_diagnosis(s)}

  <h2>逐条明细<span style="font-weight:400;color:#6b7280;font-size:13px">
    （失败项已置顶）</span></h2>
  <table>
    <tr><th>case</th><th>问题</th><th>状态</th><th>首个命中</th>
        <th>MRR</th><th>R@5</th><th>nDCG@5</th><th>耗时</th></tr>
    {_retrieval_rows(run.get('results', []))}
  </table>
  <footer>evalkit · 企业级 RAG 评测 Harness</footer>
</div></body></html>"""


# ============================================================================
# 答案报表
# ============================================================================

def _answer_rows(results: List[Dict[str, Any]]) -> str:
    rows = []
    ordered = sorted(results, key=lambda r: bool(r.get("_passed", True)))
    for r in ordered:
        passed = r.get("_passed", True)
        cls = "" if passed else "fail"
        status = ('<span class="pill ok">通过</span>' if passed
                  else '<span class="pill no">失败</span>')

        problems = []
        if r.get("error"):
            problems.append(f"异常：{_esc(r['error'])}")
        if r.get("missing_points"):
            problems.append("缺要点：" + _esc("、".join(r["missing_points"])))
        if r.get("forbidden_hits"):
            problems.append("出现禁词：" + _esc("、".join(r["forbidden_hits"])))
        if r.get("refuse_correct") is False:
            problems.append("拒答判断错误：" +
                            ("该拒答却编了答案" if not r.get("refused") else "不该拒答却拒了"))
        prob_html = ("<br>".join(f'<span style="color:#dc2626">{p}</span>'
                                 for p in problems)) or ""

        f_score = r.get("faithfulness")
        rel = r.get("relevancy")
        rows.append(f"""<tr class="{cls}">
  <td><code>{_esc(r.get('case_id'))}</code></td>
  <td class="q">{_esc(r.get('query'))}
    {('<br>' + prob_html) if prob_html else ''}
    <details><summary>查看回答</summary>
      <div class="docs"><div class="doc">{_esc(r.get('answer'))}</div>
      {f'<div class="doc"><b>judge：</b>{_esc(r.get("judge_reason"))}</div>' if r.get('judge_reason') else ''}
      </div>
    </details>
  </td>
  <td>{status}</td>
  <td>{'—' if f_score is None else f'{f_score:.2f}'}</td>
  <td>{'—' if rel is None else f'{rel:.2f}'}</td>
  <td>{r.get('latency_ms', 0)/1000:.1f}s</td>
</tr>""")
    return "".join(rows)


def render_answer(run: Dict[str, Any],
                  base: Optional[Dict[str, Any]] = None) -> str:
    """渲染答案评测报表 HTML。"""
    s = run.get("summary", {})
    bs = (base or {}).get("summary", {})
    cfg = run.get("config", {})
    dur = run.get("finished_at", 0) - run.get("started_at", 0)
    keys = [k for k in ["pass_rate", "hard_pass_rate", "faithfulness", "relevancy",
                        "refuse_accuracy", "avg_latency_ms"] if k in s]

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>答案评测报表 {_esc(run.get('run_id'))}</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
  <h1>答案评测报表</h1>
  <div class="sub">run <code>{_esc(run.get('run_id'))}</code> ·
    {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run.get('started_at', 0)))} ·
    耗时 {dur:.1f}s · {s.get('count', 0)} 条 case</div>
  <div class="cfg">{_esc(json.dumps(cfg, ensure_ascii=False))}</div>

  <h2>总览</h2>
  {_cards(s, bs, keys)}
  <div class="legend">
    <code>hard_pass_rate</code> 只看字符串硬判据（零成本）；
    <code>pass_rate</code> 综合硬判据 + LLM judge 打分。
    <code>faithfulness</code> 越高说明越少幻觉，
    <code>refuse_accuracy</code> 衡量"该拒答时有没有老实拒答"。</div>

  <h2>逐条明细<span style="font-weight:400;color:#6b7280;font-size:13px">
    （失败项已置顶）</span></h2>
  <table>
    <tr><th>case</th><th>问题</th><th>状态</th>
        <th>忠实度</th><th>相关性</th><th>耗时</th></tr>
    {_answer_rows(run.get('results', []))}
  </table>
  <footer>evalkit · 企业级 RAG 评测 Harness</footer>
</div></body></html>"""


# ============================================================================
# 落盘
# ============================================================================

def write_report(run: Dict[str, Any], base: Optional[Dict[str, Any]] = None,
                 out_path: Optional[str] = None) -> str:
    """生成并写出 HTML 报表，返回文件路径。"""
    suite = run.get("suite", "retrieval")
    html_text = (render_answer(run, base) if suite == "answer"
                 else render_retrieval(run, base))
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = out_path or os.path.join(
        REPORTS_DIR, f"{suite}-{run.get('run_id')}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_text)
    return out_path
