"""
evalkit.triage —— Bad Case 自动根因分类（R1~R8）
================================================================================

【为什么需要自动 triage】

评测跑完，alert 亮了，但"红灯"只告诉你"又挂了"，不告诉你"为什么挂、改哪"。
人工逐条看失败 case 在 case 多时不可持续。triage 把"失败信号"翻译成
"根因分类 + 该改哪里"，让 bad case 闭环真正能驱动自进化。

【八类根因（与 harness 指标一一对应）】

  R1 召回侧完全丢失   检索 bury<=0 / Recall@5=0
                      → embedding 不匹配中文 / 切片把答案劈开 / 索引未更新
  R2 排序侧埋没       召回到了但排在第 5 名之后（或 nDCG@5 偏低）
                      → rerank 把对的压下去 / RRF 融合权重不对 / 候选池过早截断
  R3 改写/精排负优化   raw 模式表现明显好于 pipeline（同一条 query）
                      → 查询改写把问题带偏 / rerank 模型在该领域水土不服
  R4 权限过滤误杀     raw 命中但 pipeline 未命中（或 admin 命中、user 未命中）
                      → 租户/角色 expr 写错，把该看的文档挡在门外
  R5 生成幻觉         检索正常（bury>0）但答案出现禁词 / faithfulness<0.6
                      → 上下文不足 / 模型编造 / 生成 prompt 没约束"只据上下文答"
  R6 答非所问         检索正常但 relevancy<0.6
                      → 路由/分类把问题送错节点 / 多跳没拆解 / 生成偏题
  R7 拒答错误         should_refuse 却编了答案（漏拒），或不该拒却拒了（误拒）
                      → 安全策略 / 拒答判定阈值
  R8 跨租户泄漏       本租户 query 召回了其他租户文档（最严重，安全风险）
                      → 租户过滤 expr 缺失 / 索引混库

【判定优先级】

安全类（R8）→ 召回类（R1/R4/R2/R3）→ 拒答类（R7）→ 生成类（R5/R6）。
即：先确认"有没有把错的文档放出来 / 有没有把对的挡住"，再看"生成质量"。
因为召回错了，后面生成再好也是建立在错误材料上，先修召回。

【输入信号】

classify() 接受检索结果 + 答案结果 + 可选 ctx（承载需要"双模式对比"才知道的信号）：
  ctx = {
    "raw_bury": int,            # 同 case 在 raw 模式下的 bury（用于 R3/R4 对比）
    "cross_tenant_hit": bool,   # 是否命中了其他租户文档（R8）
    "should_refuse": bool,      # 该 case 是否本应拒答（R7）
  }
没传 raw_bury 时，R3/R4 退化为只判 R1/R2（不误报）。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalkit.schema import RetrievalCaseResult, AnswerCaseResult  # noqa: E402

# judge 失败阈值，与 AnswerCaseResult.passed 对齐
_FAITH_TH = 0.6
_REL_TH = 0.6
_NDCG_TH = 0.5
_BURY_DEEP = 5  # 排在第 5 名之后算"埋没"


@dataclass
class TriageResult:
    """单条 bad case 的根因判定。"""
    code: str                       # "R1".."R8" 或 "OK"
    title: str
    reason: str
    suggested_fix: str
    severity: str = "high"          # high | medium | low
    is_failure: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code, "title": self.title, "reason": self.reason,
            "suggested_fix": self.suggested_fix, "severity": self.severity,
            "is_failure": self.is_failure,
        }


# 各类根因的元信息（文案集中维护，方便报表/前端复用）
_ROOT_CAUSES = {
    "R1": ("召回侧完全丢失", "high",
           "正确文档根本没进召回。优先查：embedding 模型是否适配中文技术文档、"
           "切片策略是否把答案劈成两半、索引是否是最新的。可用 `--mode raw` 复跑确认。",
           "① 换/微调 embedding ② 调 chunk_size 与重叠 ③ 重新 ingest"),
    "R2": ("排序侧埋没", "medium",
           "文档召回到了，但被排在很后面，截断后丢失。查 rerank 服务是否真生效、"
           "RRF 融合权重、候选池是否过早截断。",
           "① 调 rerank 阈值/模型 ② 调 RRF 权重 ③ 加大 candidate_k"),
    "R3": ("改写/精排负优化", "medium",
           "raw 模式明显好于 pipeline：查询改写把原意带偏，或 rerank 在该领域水土不服。",
           "① 关掉/弱化 query rewrite ② 单独验证 rerank 模型 ③ A/B 对比 raw vs pipeline"),
    "R4": ("权限过滤误杀", "high",
           "raw 命中但 pipeline 未命中（或 admin 能看、user 看不到）：租户/角色 expr 写错，"
           "把该看的文档挡在门外。",
           "① 检查租户/角色过滤 expr ② 验证 AccessControlFilter 逻辑 ③ 补单测"),
    "R5": ("生成幻觉", "high",
           "检索正常，但答案出现知识库没有的内容（禁词命中 / faithfulness 偏低）。"
           "查生成 prompt 是否约束'只据上下文作答'、上下文是否足够。",
           "① 强化 generate prompt 约束 ② 补充相关文档 ③ 降温度/加 cite 要求"),
    "R6": ("答非所问", "medium",
           "检索正常但相关性低：路由/分类把问题送错节点，或多跳没拆好。",
           "① 修 classify 路由 ② 强化 multi-hop 拆解 ③ 校验 generate 对齐问题"),
    "R7": ("拒答错误", "medium",
           "该拒答的陷阱题却编了答案（漏拒），或不该拒的被拒了（误拒）。",
           "① 调整拒答判定阈值 ② 补充安全策略样例 ③ 校准 should_refuse 标注"),
    "R8": ("跨租户泄漏", "critical",
           "本租户的问题召回了其他租户的文档，存在数据越权风险。",
           "① 立即检查租户过滤 expr ② 确认索引是否混库 ③ 加跨租户隔离回归测试"),
}


def _make(code: str, reason: str, is_failure: bool = True) -> TriageResult:
    title, sev, _, fix = _ROOT_CAUSES[code]
    return TriageResult(code=code, title=title, reason=reason,
                        suggested_fix=fix, severity=sev, is_failure=is_failure)


def classify(retrieval: Optional[RetrievalCaseResult] = None,
             answer: Optional[AnswerCaseResult] = None,
             ctx: Optional[Dict[str, Any]] = None) -> TriageResult:
    """
    综合检索结果 + 答案结果 + 可选对比信号，给出根因分类。

    传入 None 的维度会被跳过（例如只跑检索 harness 时只传 retrieval）。
    返回 TriageResult；如果都正常，返回 code="OK"。
    """
    ctx = ctx or {}

    # ---- 安全类：跨租户泄漏（最优先）----
    if ctx.get("cross_tenant_hit"):
        return _make("R8", "本租户 query 召回了其他租户文档，存在越权风险。")

    # 隔离负例命中禁止规格 = 实锤泄漏，等价于 R8
    if retrieval is not None and getattr(retrieval, "forbidden_hits", None):
        hits = "、".join(retrieval.forbidden_hits[:3])
        return _make("R8", f"命中禁止召回规格（{hits}），存在跨租户/越权泄漏。")

    # 隔离负例且未泄漏 = 行为正确。必须在检索类判定前短路，
    # 否则"什么都没召回"会被 R1 误报成召回丢失。
    if retrieval is not None and getattr(retrieval, "is_negative", False):
        if not retrieval.error:
            return _make("OK", "隔离负例：未召回到禁止内容，行为正确。")

    # ---- 检索类 ----
    if retrieval is not None and not retrieval.error:
        bury = retrieval.bury
        recall5 = retrieval.recall_at_k.get("5", 0.0)
        ndcg5 = retrieval.ndcg_at_k.get("5", 0.0)
        raw_bury = ctx.get("raw_bury")

        # R4：raw 命中但 pipeline 未命中 → 权限误杀（比 R1 更具体，优先判定）
        # 必须放在 R1 之前：pipeline 未召回时 bury<=0，若不先判 R4 会被 R1 抢走，
        # 而"权限误杀"比"泛化的召回丢失"更能直接指导修复。
        if raw_bury is not None and raw_bury > 0 and bury <= 0:
            return _make("R4",
                         f"raw 模式能召回（bury={raw_bury}）但 pipeline 未召回，"
                         f"疑似权限/角色过滤误杀。")

        # R1：完全没召回（无 raw 对比信号，或 raw 也未召回）
        if bury <= 0 or recall5 == 0.0:
            return _make("R1",
                         f"正确文档未进入召回（bury={bury}, Recall@5={recall5:.2f}）。")

        # R3：raw 明显优于 pipeline（改写/rerank 负优化）
        if raw_bury is not None and raw_bury > 0 and raw_bury < bury:
            return _make("R3",
                         f"raw 模式 bury={raw_bury} 明显优于 pipeline bury={bury}，"
                         f"查询改写或 rerank 对该问题负优化。")

        # R2：召回到了但排太深 / nDCG 偏低
        if bury > _BURY_DEEP or ndcg5 < _NDCG_TH:
            return _make("R2",
                         f"文档召回到了但排序过深（bury={bury}, nDCG@5={ndcg5:.3f}），"
                         f"被截断导致丢失。")

    # 检索整体异常（取数失败等）
    if retrieval is not None and retrieval.error:
        return _make("R1", f"检索执行异常：{retrieval.error}")

    # ---- 拒答类 ----
    if answer is not None:
        should_refuse = ctx.get("should_refuse", False)
        if should_refuse and answer.refuse_correct is False:
            return _make("R7",
                         "该拒答的陷阱题却生成了答案（漏拒），疑似幻觉或安全策略未生效。")
        if (not should_refuse) and answer.refuse_correct is False:
            return _make("R7", "不该拒答的问题被拒了（误拒）。")

    # ---- 生成类 ----
    if answer is not None and not answer.error:
        if answer.forbidden_hits:
            return _make("R5",
                         f"答案出现禁词 {answer.forbidden_hits}，存在幻觉/编造。")
        if answer.faithfulness is not None and answer.faithfulness < _FAITH_TH:
            return _make("R5",
                         f"faithfulness={answer.faithfulness:.2f} 低于阈值"
                         f"{_FAITH_TH}，答案不忠于上下文。")
        if answer.relevancy is not None and answer.relevancy < _REL_TH:
            return _make("R6",
                         f"relevancy={answer.relevancy:.2f} 低于阈值{_REL_TH}，答非所问。")

    if answer is not None and answer.error:
        return _make("R5", f"答案生成异常：{answer.error}")

    # ---- 都正常 ----
    return TriageResult(
        code="OK", title="通过", reason="检索与答案均达标，无需处理。",
        suggested_fix="", severity="low", is_failure=False,
    )


def triage_run(run: Dict[str, Any],
               ctx_by_case: Optional[Dict[str, Dict[str, Any]]] = None
               ) -> Dict[str, Any]:
    """
    对一个 EvalRun 的失败 case 批量分类，返回汇总。

    参数：
        run:         EvalRun.to_dict()（含 suite / results）
        ctx_by_case: 可选的 {case_id: ctx} 对比信号（raw_bury / cross_tenant_hit / should_refuse）
    返回：
        {"suite", "count", "failures", "by_code": {R1: n, ...},
         "items": [{case_id, code, title, reason, suggested_fix, severity}]}
    """
    ctx_by_case = ctx_by_case or {}
    suite = run.get("suite", "retrieval")
    items: List[Dict[str, Any]] = []
    by_code: Dict[str, int] = {}

    for r in run.get("results", []):
        case_id = r.get("case_id", "?")
        # 已通过的跳过。_passed 由 harness 显式落盘，统一了正例/负例两种口径；
        # 缺字段时（老 run json）才退回按 bury 判定。
        if suite == "retrieval":
            retrieval = RetrievalCaseResult(**{k: r.get(k) for k in (
                "case_id", "query", "bury", "recall_at_k", "ndcg_at_k", "error")})
            retrieval.hit_ranks = r.get("hit_ranks", [])
            retrieval.is_negative = bool(r.get("is_negative"))
            retrieval.forbidden_hits = r.get("forbidden_hits") or []
            ok = r["_passed"] if "_passed" in r else retrieval.passed
            answer = None
        else:
            ok = r.get("_passed", True)
            retrieval = None
            answer = AnswerCaseResult(**{k: r.get(k) for k in (
                "case_id", "query", "missing_points", "forbidden_hits",
                "refuse_correct", "faithfulness", "relevancy", "error")})

        if ok:
            continue

        ctx = ctx_by_case.get(case_id, {})
        # should_refuse 只存在于答案 case 标注里；若 ctx 没给，triage 退化为不看 R7
        res = classify(retrieval=retrieval, answer=answer, ctx=ctx)
        d = res.to_dict()
        d["case_id"] = case_id
        items.append(d)
        by_code[res.code] = by_code.get(res.code, 0) + 1

    return {
        "suite": suite,
        "count": len(run.get("results", [])),
        "failures": len(items),
        "by_code": by_code,
        "items": items,
    }
