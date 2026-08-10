"""
evalkit.schema —— 黄金集与评测结果的数据结构定义
================================================================================

本模块是整个 harness 的地基，定义三类东西：

  1. 黄金集 case（输入）：RetrievalCase / AnswerCase
  2. 单条评测结果（输出）：RetrievalCaseResult / AnswerCaseResult
  3. 一次完整评测（聚合）：EvalRun

以及 jsonl 的读写工具。


【核心设计决策：黄金集怎么标注「哪些文档是对的」】

最直觉的做法是记 chunk_id 或 chunk_index：
    "relevant_chunks": [1024, 1025]
这是个陷阱。只要重新 ingest 一次（换切片策略、加一页 PDF、改 chunk_size），
所有 chunk 编号全部重排，黄金集当场作废，之前的人工标注全白干。

所以本项目采用「抗重建」的三重标识：

    file_name   哪个文件      —— 换切片策略不会变
    page        第几页        —— 除非重新排版，否则不会变
    keywords    关键词兜底     —— 连页码都不可靠时（比如换了 PDF 版本），
                                 用内容特征判断，最抗变化

一条 relevant 规则命中的判定是「与」关系里带兜底：
    file 匹配  且  (pages 为空 或 page ∈ pages 或 内容命中 keywords)

这样即使某次 ingest 后页码元数据丢失，靠 keywords 仍能正确判分，
黄金集的生命周期可以跨越多次索引重建。


【为什么 relevant 要带 gain（增益）】

不是所有"相关文档"的价值一样：
  - 直接写着答案的那一段          → gain=3（高度相关）
  - 提到了但要推理才能得出的段落    → gain=2（中等相关）
  - 只是背景介绍                  → gain=1（弱相关）

Recall/MRR 是二值指标（相关 or 不相关），看不出这个差别。
nDCG 能：它按 gain 加权、按排名折损，
「把 gain=3 的排第 1」和「把 gain=1 的排第 1」得分完全不同。
这正是评价排序质量（而不只是召回能力）需要的。
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ============================================================================
# 1. 黄金集：相关性标注规则
# ============================================================================

@dataclass
class RelevantSpec:
    """
    一条「什么样的文档算命中」的标注规则。

    字段：
        file:     文件名（支持子串匹配，例如写 "Jimi IoT" 即可匹配全名）
        pages:    页码列表；为空表示「该文件任意页都算命中」
        keywords: 内容关键词；命中任一即算命中（页码不可靠时的兜底判据）
        gain:     相关性增益，用于 nDCG。3=直接答案 / 2=需推理 / 1=背景
        note:     人工备注，说明为什么这段是答案（给后来标注者看）
    """
    file: str = ""
    pages: List[int] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    gain: float = 1.0
    note: str = ""

    def matches(self, meta: Dict[str, Any], content: str) -> bool:
        """
        判断一条检索结果是否命中本规则。

        参数：
            meta:    检索结果的元数据（含 file_name / page 等）
            content: 检索结果的正文

        判定逻辑（见模块头部说明）：
            文件必须匹配；页码与关键词是「或」关系，任一成立即命中。
            若 pages 和 keywords 都为空，则只要文件匹配就算命中
            （用于「这个问题的答案只可能在这份文档里」的粗粒度标注）。
        """
        # ---- 文件匹配（子串，容忍路径前缀和版本号差异）----
        if self.file:
            file_name = str(meta.get("file_name") or meta.get("source") or "")
            file_path = str(meta.get("file_path") or "")
            if self.file not in file_name and self.file not in file_path:
                return False

        # 没有更细的条件 → 文件匹配即命中
        if not self.pages and not self.keywords:
            return True

        # ---- 页码匹配 ----
        if self.pages:
            page = meta.get("page")
            try:
                if page is not None and int(page) in self.pages:
                    return True
            except (TypeError, ValueError):
                pass  # 页码元数据异常时，交给关键词兜底

        # ---- 关键词兜底 ----
        if self.keywords:
            body = content or ""
            for kw in self.keywords:
                if kw and kw in body:
                    return True

        return False


# ============================================================================
# 2. 黄金集：检索 case 与答案 case
# ============================================================================

@dataclass
class RetrievalCase:
    """
    检索层黄金集的一条题目。

    只关心「该找出来的文档有没有被找出来、排在第几」，不关心最终答案措辞。
    因此评测这层完全不需要调用 LLM，零成本、秒级。
    """
    case_id: str
    query: str
    relevant: List[RelevantSpec] = field(default_factory=list)

    # 「禁止召回」标注：命中任意一条即判失败。
    # 用于隔离类负例——例如在 yh 租户下提问只存在于 jm 文档的内容，
    # 正确行为是召不到；一旦召回到 jm 的文件，就是跨租户泄漏（严重安全问题）。
    # 没有 relevant 只有 forbidden 的 case 即「负例」，判定逻辑整体反转。
    forbidden: List[RelevantSpec] = field(default_factory=list)

    # 检索上下文：不同租户/角色的检索结果不同，必须固定下来才可复现
    tenant_id: str = "default"
    role: str = "admin"
    user_id: str = "anonymous"

    # tags 用于分组看指标，例如 ["fact-lookup"] / ["multi-hop"] / ["table"]
    # 能回答「我们在哪类问题上弱」，而不只是一个笼统的总分
    tags: List[str] = field(default_factory=list)

    # source 标记这条 case 从哪来：
    #   manual   人工编写
    #   mined    从历史 trace 自动挖掘
    #   feedback 用户点踩转化而来（最珍贵：真实失败样本）
    source: str = "manual"
    note: str = ""

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RetrievalCase":
        rel = [RelevantSpec(**r) for r in d.get("relevant", [])]
        forb = [RelevantSpec(**r) for r in d.get("forbidden", [])]
        return RetrievalCase(
            case_id=d["case_id"],
            query=d["query"],
            relevant=rel,
            forbidden=forb,
            tenant_id=d.get("tenant_id", "default"),
            role=d.get("role", "admin"),
            user_id=d.get("user_id", "anonymous"),
            tags=d.get("tags", []),
            source=d.get("source", "manual"),
            note=d.get("note", ""),
        )

    def total_gain(self) -> float:
        """所有相关文档的增益之和，算 IDCG 时用。"""
        return sum(r.gain for r in self.relevant)

    @property
    def is_negative(self) -> bool:
        """负例：没有正确答案可召回，考察的是「不该召回的有没有漏出来」。"""
        return not self.relevant


@dataclass
class AnswerCase:
    """
    答案层黄金集的一条题目。

    比检索 case 多了对「最终回答」的要求，评测时需要跑完整链路并调 judge。

    字段说明：
        reference:        参考答案（不要求逐字一致，judge 用它做语义比对）
        must_include:     答案必须包含的要点（字符串匹配，硬性判据，零成本）
        must_not_include: 答案不能出现的内容（例如错误型号、竞品名、幻觉常客）
        should_refuse:    这题「应该拒答」。知识库里没有的东西硬编一个答案，
                          比说"我不知道"危害大得多。这类 case 专门抓幻觉。
    """
    case_id: str
    query: str
    reference: str = ""
    must_include: List[str] = field(default_factory=list)
    must_not_include: List[str] = field(default_factory=list)
    should_refuse: bool = False

    tenant_id: str = "default"
    role: str = "admin"
    user_id: str = "anonymous"
    tags: List[str] = field(default_factory=list)
    source: str = "manual"
    note: str = ""

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AnswerCase":
        return AnswerCase(
            case_id=d["case_id"],
            query=d["query"],
            reference=d.get("reference", ""),
            must_include=d.get("must_include", []),
            must_not_include=d.get("must_not_include", []),
            should_refuse=d.get("should_refuse", False),
            tenant_id=d.get("tenant_id", "default"),
            role=d.get("role", "admin"),
            user_id=d.get("user_id", "anonymous"),
            tags=d.get("tags", []),
            source=d.get("source", "manual"),
            note=d.get("note", ""),
        )


# ============================================================================
# 3. 评测结果
# ============================================================================

@dataclass
class RetrievalCaseResult:
    """
    单条检索 case 的评测结果。

    指标口径（k 默认取 1/3/5/10）：

      Recall@k  = 命中的相关文档数 / 相关文档总数
                  回答「该找的东西找到了几成」

      MRR       = 1 / 第一个相关文档的排名
                  回答「用户要往下翻多久才看到有用的」
                  排第 1 得 1.0，排第 5 只有 0.2，惩罚很陡

      nDCG@k    = DCG@k / IDCG@k，DCG = Σ gain_i / log2(rank_i + 1)
                  回答「排序质量如何」，考虑相关性分级，是最全面的单一指标

      bury      = 第一个相关文档的排名（1-based），-1 表示压根没召回
                  这是本项目自定义的诊断指标，专门区分两类完全不同的故障：
                    bury = -1   → 召回阶段就丢了（该查 embedding / 切片 / 权限过滤）
                    bury > top_k → 召回到了但排太后被截断（该查 rerank / 融合权重）
                  只看 Recall 会把这两种混为一谈，修错方向。
    """
    case_id: str
    query: str
    retrieved: List[Dict[str, Any]] = field(default_factory=list)  # 精简后的命中列表
    hit_ranks: List[int] = field(default_factory=list)  # 每条 relevant 规则命中的排名，-1=未命中
    recall_at_k: Dict[str, float] = field(default_factory=dict)
    ndcg_at_k: Dict[str, float] = field(default_factory=dict)
    mrr: float = 0.0
    bury: int = -1
    latency_ms: float = 0.0
    tags: List[str] = field(default_factory=list)
    error: str = ""

    # 负例（隔离类）：没有正确答案，考察的是「不该出现的有没有漏出来」，判定反转
    is_negative: bool = False
    # 命中的禁止规格描述，非空即表示泄漏
    forbidden_hits: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """
        及格线分两种口径：

          正例：至少召回一个相关文档（bury > 0）。
          负例：一条禁止规格都没命中 —— 「什么都没召回」对负例才是正确行为，
                若沿用正例口径会把隔离成功误报成失败，直接污染 pass_rate 与
                miss_count，把人引向不存在的召回问题。
        """
        if self.error:
            return False
        if self.is_negative:
            return not self.forbidden_hits
        return self.bury > 0


@dataclass
class AnswerCaseResult:
    """单条答案 case 的评测结果。"""
    case_id: str
    query: str
    answer: str = ""
    # 硬性判据（字符串匹配，零成本，先跑这层能过滤掉一大半明显问题）
    missing_points: List[str] = field(default_factory=list)   # must_include 里没出现的
    forbidden_hits: List[str] = field(default_factory=list)   # must_not_include 里出现了的
    refused: bool = False
    refuse_correct: Optional[bool] = None   # None 表示这条 case 不考察拒答
    # LLM judge 打分（0~1），judge 不可用时为 None
    faithfulness: Optional[float] = None    # 答案是否忠于检索到的上下文（抓幻觉）
    relevancy: Optional[float] = None       # 答案是否切题（抓答非所问）
    judge_reason: str = ""
    latency_ms: float = 0.0
    task_id: str = ""                       # 关联 task_checkpoints 的全链路 trace
    tags: List[str] = field(default_factory=list)
    error: str = ""

    @property
    def passed(self) -> bool:
        """
        综合及格判定：
          - 无异常
          - 该包含的要点都在、不该出现的都没出现
          - 拒答判断正确（如果考察）
          - judge 打分（如果有）不低于 0.6
        """
        if self.error:
            return False
        if self.missing_points or self.forbidden_hits:
            return False
        if self.refuse_correct is False:
            return False
        if self.faithfulness is not None and self.faithfulness < 0.6:
            return False
        if self.relevancy is not None and self.relevancy < 0.6:
            return False
        return True


@dataclass
class EvalRun:
    """
    一次完整评测的记录。

    run_id 用「时间戳 + 短 uuid」，既能按时间排序，又不会撞名。
    config 必须记录本次评测时的关键配置（hybrid 开关、top_k、rerank 开关、模型名），
    否则两次 run 的指标差异无法归因 —— 你不知道是代码变了还是配置变了。
    """
    run_id: str
    suite: str                     # "retrieval" | "answer"
    started_at: float
    finished_at: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    results: List[Dict[str, Any]] = field(default_factory=list)
    note: str = ""

    @staticmethod
    def new(suite: str, config: Optional[Dict[str, Any]] = None) -> "EvalRun":
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        return EvalRun(run_id=run_id, suite=suite, started_at=time.time(),
                       config=config or {})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# 4. 指标计算
# ============================================================================

def compute_retrieval_metrics(case: RetrievalCase,
                              hit_ranks: List[int],
                              ks: List[int]) -> Dict[str, Any]:
    """
    根据「每条 relevant 规则命中的排名」计算全部检索指标。

    参数：
        case:      黄金集题目（提供 gain 分布）
        hit_ranks: 与 case.relevant 等长，第 i 个元素是第 i 条规则命中的排名
                   （1-based），未命中为 -1
        ks:        要计算的 k 值列表，如 [1, 3, 5, 10]

    返回 dict，含 recall_at_k / ndcg_at_k / mrr / bury。

    ---- nDCG 的实现细节（容易写错的地方）----
    DCG  = Σ  gain_i / log2(rank_i + 1)     只累加排名 ≤ k 的命中
    IDCG = Σ  gain_j / log2(j + 1)          把所有 gain 按降序排列后的理想值
    nDCG = DCG / IDCG                        归一化到 0~1，可跨 case 平均

    注意 IDCG 要用「按 gain 降序」的理想排列，而不是原始顺序，
    否则当高 gain 文档在标注里排后面时，nDCG 会算出大于 1 的值。
    """
    total = len(case.relevant)
    out: Dict[str, Any] = {"recall_at_k": {}, "ndcg_at_k": {}, "mrr": 0.0, "bury": -1}
    if total == 0:
        return out

    gains = [r.gain for r in case.relevant]

    # ---- bury：第一个命中的排名 ----
    valid = [r for r in hit_ranks if r > 0]
    out["bury"] = min(valid) if valid else -1

    # ---- MRR ----
    out["mrr"] = 1.0 / min(valid) if valid else 0.0

    # ---- IDCG（与 k 无关的理想排列）----
    ideal_gains = sorted(gains, reverse=True)

    for k in ks:
        # Recall@k：排名在前 k 的命中数 / 相关文档总数
        hit_k = sum(1 for r in hit_ranks if 0 < r <= k)
        out["recall_at_k"][str(k)] = hit_k / total

        # DCG@k
        dcg = 0.0
        for gain, rank in zip(gains, hit_ranks):
            if 0 < rank <= k:
                dcg += gain / math.log2(rank + 1)
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains[:k]))
        out["ndcg_at_k"][str(k)] = (dcg / idcg) if idcg > 0 else 0.0

    return out


def aggregate(results: List[Dict[str, Any]], ks: List[int]) -> Dict[str, Any]:
    """
    把逐条结果聚合成总览指标。

    除了各项均值，额外统计三个「诊断计数」，这才是真正能指导下一步动作的：
      miss_count   完全没召回的 case 数     → 召回侧问题（embedding / 切片 / 权限）
      buried_count 召回到了但排在 5 名之后   → 排序侧问题（rerank / 融合权重）
      pass_rate    至少召回一条的比例
    """
    n = len(results)
    if n == 0:
        return {"count": 0}

    # 负例（隔离类）没有正确答案，Recall/nDCG/MRR 对它恒为 0。
    # 若混进均值会凭空拉低分数，且让指标随负例条数漂移、跨 run 不可比，
    # 所以质量类指标只在正例上计算；负例单独用 leak_count 体现。
    pos = [r for r in results if not r.get("is_negative")]
    neg = [r for r in results if r.get("is_negative")]
    np_ = len(pos) or 1

    summary: Dict[str, Any] = {"count": n, "positive_count": len(pos),
                               "negative_count": len(neg)}
    for k in ks:
        key = str(k)
        summary[f"recall@{k}"] = round(
            sum(r["recall_at_k"].get(key, 0.0) for r in pos) / np_, 4)
        summary[f"ndcg@{k}"] = round(
            sum(r["ndcg_at_k"].get(key, 0.0) for r in pos) / np_, 4)
    summary["mrr"] = round(sum(r.get("mrr", 0.0) for r in pos) / np_, 4)

    buries = [r.get("bury", -1) for r in pos]
    summary["miss_count"] = sum(1 for b in buries if b <= 0)
    summary["buried_count"] = sum(1 for b in buries if b > 5)
    # 泄漏数：负例中命中了禁止规格的条数，安全类指标，必须为 0
    summary["leak_count"] = sum(1 for r in neg if r.get("forbidden_hits"))
    # 通过率覆盖全部 case（正例看召回、负例看隔离），口径由 _passed 统一给出
    summary["pass_rate"] = round(
        sum(1 for r in results if r.get("_passed")) / n, 4)
    hit_buries = [b for b in buries if b > 0]
    summary["avg_bury"] = round(sum(hit_buries) / len(hit_buries), 2) if hit_buries else -1
    summary["avg_latency_ms"] = round(
        sum(r.get("latency_ms", 0.0) for r in results) / n, 1)
    return summary


def aggregate_answer(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    把逐条答案结果聚合成总览指标（供 HTML 报表与门禁使用）。

    指标口径：
      pass_rate        综合及格率（硬判据 + judge 阈值，见 AnswerCaseResult.passed）
      hard_pass_rate   只看字符串硬判据（must_include / must_not_include / 拒答），
                       不依赖 judge，零外部依赖，适合 CI 常驻跑
      faithfulness     平均忠实度（仅统计 judge 成功打分的 case）
      relevancy        平均相关性
      refuse_accuracy  在考察拒答的 case 中，拒答判断正确的比例；无此类 case 记 1.0
      avg_latency_ms   平均耗时
    """
    n = len(results)
    if n == 0:
        return {"count": 0}

    hard_pass = sum(1 for r in results
                    if not r.get("missing_points") and not r.get("forbidden_hits")
                    and r.get("refuse_correct") is not False)
    passed = sum(1 for r in results if r.get("_passed"))
    refs = [r for r in results if r.get("refuse_correct") is not None]
    refuse_ok = sum(1 for r in refs if r.get("refuse_correct") is True)
    faith = [r["faithfulness"] for r in results
             if isinstance(r.get("faithfulness"), (int, float))]
    rel = [r["relevancy"] for r in results
           if isinstance(r.get("relevancy"), (int, float))]

    return {
        "count": n,
        "pass_rate": round(passed / n, 4),
        "hard_pass_rate": round(hard_pass / n, 4),
        "faithfulness": round(sum(faith) / len(faith), 4) if faith else None,
        "relevancy": round(sum(rel) / len(rel), 4) if rel else None,
        "refuse_accuracy": round(refuse_ok / len(refs), 4) if refs else 1.0,
        "refuse_cases": len(refs),
        "avg_latency_ms": round(
            sum(r.get("latency_ms", 0.0) for r in results) / n, 1),
    }


# ============================================================================
# 5. jsonl 读写
# ============================================================================
# 为什么用 jsonl 而不是 json 数组或 yaml：
#   - 一行一条，git diff 清爽，评审新增/修改的 case 一目了然
#   - 追加新 case 不用改动已有行，多人并行标注不易冲突
#   - 大文件可以流式读，不必一次载入内存

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """读取 jsonl，自动跳过空行和 # 开头的注释行。"""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                # 单行坏了不该让整个评测挂掉，跳过并报位置，方便定位
                print(f"[evalkit] ⚠ {path}:{lineno} JSON 解析失败，已跳过: {e}")
    return out


def dump_jsonl(path: str, rows: List[Dict[str, Any]]):
    """写 jsonl（覆盖）。目录不存在时自动创建。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: str, rows: List[Dict[str, Any]]):
    """追加写 jsonl（挖掘脚本增量补充 case 时用）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_retrieval_cases(path: str) -> List[RetrievalCase]:
    """加载检索黄金集。"""
    return [RetrievalCase.from_dict(d) for d in load_jsonl(path)]


def load_answer_cases(path: str) -> List[AnswerCase]:
    """加载答案黄金集。"""
    return [AnswerCase.from_dict(d) for d in load_jsonl(path)]


# ---- 目录约定 ----
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.path.join(PKG_DIR, "golden")
RUNS_DIR = os.path.join(PKG_DIR, "runs")        # 每次评测的原始结果 json
REPORTS_DIR = os.path.join(PKG_DIR, "reports")  # 生成的 HTML 报表

DEFAULT_RETRIEVAL_GOLDEN = os.path.join(GOLDEN_DIR, "retrieval.jsonl")
DEFAULT_ANSWER_GOLDEN = os.path.join(GOLDEN_DIR, "answer.jsonl")


def save_run(run: EvalRun) -> str:
    """把一次评测结果落盘为 json，返回文件路径。供后续 diff 对比。"""
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, f"{run.suite}-{run.run_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run.to_dict(), f, ensure_ascii=False, indent=2)
    return path


def load_run(path: str) -> Dict[str, Any]:
    """读取一次历史评测结果。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_runs(suite: Optional[str] = None) -> List[str]:
    """按时间倒序列出历史 run 文件路径（用于 --compare last 之类的便捷用法）。"""
    if not os.path.isdir(RUNS_DIR):
        return []
    files = [os.path.join(RUNS_DIR, f) for f in os.listdir(RUNS_DIR)
             if f.endswith(".json") and (suite is None or f.startswith(suite + "-"))]
    return sorted(files, key=os.path.getmtime, reverse=True)
