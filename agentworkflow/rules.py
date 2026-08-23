# -*- coding: utf-8 -*-
"""
rules —— 诊断归因的纯规则层（零第三方依赖）
================================================================================

【为什么单独一层】

归因规则是「Workflow 形态」的灵魂：分支全部写在代码里，LLM 只在节点内部
干活、永远不能改变流程形状。把规则从 pipeline.py 拆出来：

  1. tests/test_agentworkflow.py 可以零外部依赖做表驱动单测（项目测试惯例）；
  2. 规则的调整（阈值 / 优先级）与图结构解耦，改规则不动图。

【归因优先级】（与 evalkit/triage.py 的判定精神一致：安全类 → 召回类 → 拒答类 → 生成类）

  安全类 R8 跨租户泄漏 → 召回类 R1 → 拒答类 R7 → 生成类 R5/R6 → 全正常则低置信升级 ReAct

【信号来源】（与 golden 集评测不同——点踩 case 没有 gold 标注，无法算
Recall@5 / bury / nDCG，因此信号换成「复跑实测 + LLM 探查」）：

  leak_hits       本租户视角复跑却命中其他租户文档（R8 实锤）
  scoped_hits     以 case 租户视角复跑的命中
  full_hits       全库视角（super_admin）复跑的命中——用于区分
                  「内容根本不在库里」vs「内容在但本租户看不到」（R4 方向）
  docs_relevant   LLM 判定当前检索结果能否回答 query（evalgrade 链）
  scores          JudgeLLM 复判故障答案的 faithfulness / relevancy
  refused         故障答案表现为拒答（JudgeLLM._grade_refusal 语义正则）

R 码文案复用 evalkit.triage._ROOT_CAUSES（R1~R8 的 title/severity/建议集中维护），
不在本文件重复维护一份。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# 复用 evalkit 的根因文案库（title / severity / 修复建议集中维护处）
try:
    from evalkit.triage import _ROOT_CAUSES
except Exception:  # pragma: no cover - evalkit 缺失时兜底（不应发生）
    _ROOT_CAUSES = {
        "R1": ("召回侧完全丢失", "high", "", "① 检查知识库是否入库 ② 重新 ingest ③ 用 --mode raw 复核"),
        "R5": ("生成幻觉", "high", "", "① 强化 generate prompt 约束 ② 补充相关文档 ③ 降温度/加 cite 要求"),
        "R6": ("答非所问", "medium", "", "① 修 classify 路由 ② 校验 generate 对齐问题"),
        "R7": ("拒答错误", "medium", "", "① 调整拒答判定阈值 ② 校准 should_refuse 标注"),
        "R8": ("跨租户泄漏", "critical", "", "① 立即检查租户过滤 expr ② 确认索引是否混库"),
    }

# judge 阈值与 evalkit.triage / harness 判分口径保持一致
FAITH_TH = 0.6
REL_TH = 0.6

VALID_CODES = ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8")


def _result(code: Optional[str], reason: str, confidence: str,
            escalate: bool = False, extra: str = "") -> Dict[str, Any]:
    """组装归因结论。code=None 表示无法自动归因（转人工或升级探查）。"""
    if code and code in _ROOT_CAUSES:
        title, sev, _, fix = _ROOT_CAUSES[code]
    else:
        title, sev, fix = "未能自动归因", "low", "请人工分析"
    d = {
        "code": code, "title": title, "severity": sev,
        "reason": reason, "suggested_fix": fix,
        "confidence": confidence, "escalate": escalate,
    }
    if extra:
        d["reason"] = f"{reason} {extra}".strip()
    return d


def classify_signals(*,
                     leak_hits: List[Dict] = None,
                     scoped_hits: List[Dict] = None,
                     full_hits: List[Dict] = None,
                     docs_relevant: Optional[bool] = None,
                     docs_gap: str = "",
                     scores: Dict[str, Any] = None,
                     refused: bool = False,
                     source: str = "",
                     retrieval_error: str = "",
                     ) -> Dict[str, Any]:
    """
    纯规则归因：输入复跑/复判信号，输出 R 码结论。

    返回 {code,title,severity,reason,suggested_fix,confidence,escalate}
    escalate=True 表示标准流水线低置信，应升级 ReAct 探查（分支由图代码控制）。
    """
    leak_hits = leak_hits or []
    scoped_hits = scoped_hits or []
    full_hits = full_hits or []
    scores = scores or {}
    faith = scores.get("faithfulness")
    rel = scores.get("relevancy")

    # ---- 0. 复跑彻底失败：没有可用信号，不硬猜 ----
    if retrieval_error and not scoped_hits and not full_hits:
        return _result(None, f"复跑检索失败：{retrieval_error}", "低")

    # ---- 1. 安全类：R8 跨租户泄漏（最优先）----
    if leak_hits:
        files = "、".join(h.get("file", "?") for h in leak_hits[:3])
        return _result("R8",
                       f"本租户视角复跑命中其他租户文档（{files}），存在越权风险。",
                       "高")

    # ---- 2. 召回类：R1（含「内容在库里但本租户看不到」的 R4 方向提示）----
    no_recall = (not scoped_hits) or (docs_relevant is False)
    if no_recall:
        if not scoped_hits and full_hits:
            # 全库能看到、本租户视角看不到 → 更像权限/租户配置问题（R4 方向），
            # 但点踩样本没有记录当时用户角色，只能提示不能定论（诚实边界）。
            return _result(
                "R1",
                f"本租户视角零召回，但全库可检索到 {len(full_hits)} 条相关内容，"
                f"疑似租户/权限过滤误杀（R4 方向），需人工确认。",
                "中")
        gap = f"（LLM 探查：{docs_gap}）" if (docs_relevant is False and docs_gap) else ""
        return _result(
            "R1",
            f"当前检索无法支撑该问题（租户视角命中 {len(scoped_hits)} 条、"
            f"相关性判定={docs_relevant}）{gap}。",
            "高")

    # ---- 3. 拒答类：R7（文档明明能答，答案却拒了 = 误拒）----
    if refused:
        return _result("R7",
                       "检索内容足以回答，但当时答案表现为拒答（误拒）。",
                       "中")

    # ---- 4. 生成类：judge 不可用时不下生成类结论 ----
    if scores.get("error") or (faith is None and rel is None):
        return _result(None,
                       f"检索正常但 judge 复判不可用（{scores.get('error') or '分数缺失'}），"
                       f"生成类根因无法自动判定。",
                       "低")

    if faith is not None and faith < FAITH_TH:
        judge_reason = scores.get("reason") or ""
        return _result("R5",
                       f"faithfulness={faith:.2f} 低于阈值 {FAITH_TH}，"
                       f"答案不忠于上下文。",
                       "高",
                       extra=f"（judge：{judge_reason}）" if judge_reason else "")

    if rel is not None and rel < REL_TH:
        return _result("R6",
                       f"relevancy={rel:.2f} 低于阈值 {REL_TH}，答非所问。",
                       "中")

    # ---- 5. 全部信号正常但仍被标记失败 → 低置信，升级 ReAct 探查 ----
    return _result(None,
                   f"复跑检索与 judge 复判均正常（faith={faith}, rel={rel}），"
                   f"与「失败样本」标记矛盾（source={source or '?'}）。",
                   "低",
                   escalate=True)


# ----------------------------------------------------------------------
# 租户解析：从点踩时写入的诊断文本里抠出 tenant=xxx
# ----------------------------------------------------------------------
_TENANT_RE = re.compile(r"tenant[=＝]\s*([A-Za-z0-9_\-]+)")


def parse_tenant(text: str) -> Optional[str]:
    """"用户点踩（tenant=jm），待 triage。" → "jm"；解析不到返回 None。"""
    m = _TENANT_RE.search(text or "")
    return m.group(1) if m else None


# ----------------------------------------------------------------------
# ReAct 探查结论解析（宽松 JSON：模型偶发夹带文字）
# ----------------------------------------------------------------------
_CONCLUSION_RE = re.compile(r"\{.*\}", re.DOTALL)


def normalize_code(v: Any) -> Optional[str]:
    """"R5" / "r5" / "5" → "R5"；非法值返回 None。"""
    s = str(v or "").strip().upper()
    m = re.match(r"^R?([1-8])$", s)
    return f"R{m.group(1)}" if m else None


def parse_conclusion(text: str) -> Dict[str, Any]:
    """
    解析 ReAct 探查的 Final Answer。

    期望 JSON：{"code":"R1"|"R5"|...|null, "evidence":"...", "suggestion":"..."}
    解析失败 → code=None（上层据此「探查完成但未归因，转人工」，不硬猜）。
    """
    out: Dict[str, Any] = {"code": None, "evidence": "", "suggestion": ""}
    raw = (text or "").strip()
    m = _CONCLUSION_RE.search(raw)
    obj = None
    if m:
        try:
            obj = json.loads(m.group(0))
        except Exception:
            obj = None
    if not isinstance(obj, dict):
        return out
    out["code"] = normalize_code(obj.get("code"))
    out["evidence"] = str(obj.get("evidence") or "")[:500]
    out["suggestion"] = str(obj.get("suggestion") or "")[:500]
    return out


# ----------------------------------------------------------------------
# 命中记录的租户归属：knowledge/{tenant}/... → tenant
# ----------------------------------------------------------------------
def tenant_of_path(path: str) -> Optional[str]:
    """从文件路径推断租户：含 knowledge/<tenant>/ 段取 <tenant>，否则 None。"""
    if not path:
        return None
    parts = str(path).replace("\\", "/").split("/")
    if "knowledge" in parts:
        i = parts.index("knowledge")
        if i + 1 < len(parts) and parts[i + 1]:
            return parts[i + 1]
    return None
