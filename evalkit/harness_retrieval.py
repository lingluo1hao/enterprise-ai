"""
evalkit.harness_retrieval —— 检索层评测 Harness
================================================================================

【为什么先做检索层】

RAG 系统答错，绝大多数不是"模型笨"，而是"根本没把正确的段落喂给它"。
拿一个答错的问题去调 prompt、换更大的模型，往往是在错误的方向上使劲。
检索层评测能在**零 LLM 成本、秒级**的前提下告诉你：正确段落到底进没进上下文。

【三种检索模式，用来定位故障在哪一层】

    raw       只做向量库检索（dense + BM25 混合），不做跨 query 融合、不做精排
              → 考察 embedding 质量、切片策略、权限过滤是否误杀

    pipeline  raw + RRF 融合 + cross-encoder 精排（复用线上真实代码）
              → 考察排序质量

    full      pipeline + LLM query 改写（要调 LLM，有成本，默认不跑）
              → 考察改写会不会把问题带偏

分层对比能直接给出结论，而不用靠猜：

    raw 差                  → 召回侧问题：换 embedding / 调切片 / 查权限 expr
    raw 好但 pipeline 差     → 排序侧问题：rerank 把对的压下去了 / 融合权重不对
    pipeline 好但 full 差    → 改写侧问题：LLM 把问题改跑偏了

这三条结论分别对应 bad case 根因分类里的 R1、R2、R3。

【为什么要复用线上代码而不是自己重写一遍检索逻辑】

Harness 自己实现一套"差不多的检索"，就会与线上实现慢慢漂移，
最后评测分数很好看，线上依然拉胯。所以 pipeline 模式直接调
LangGraphRAGApp._do_retrieve —— 评的就是线上那份代码本身。

实现手法：构造一个只带必要字段的轻量宿主对象，把未绑定方法挂上去调用。
这样既拿到真实逻辑，又不必启动完整 App（不用连 MySQL / Redis / LLM 网关）。
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional

# 允许从项目根目录直接 `python -m evalkit.runner` 运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalkit.schema import (  # noqa: E402
    RetrievalCase, RetrievalCaseResult, EvalRun,
    compute_retrieval_metrics, aggregate,
)

DEFAULT_KS = [1, 3, 5, 10]
# 检索深度：比线上实际使用的 top_k 深，这样才能区分
#   "压根没召回（bury=-1）"  vs  "召回了但排在第 12 位（bury=12）"
# 如果只取 top 5，两者都表现为"没命中"，会把人引向错误的修复方向。
DEFAULT_FETCH_K = 20


# ============================================================================
# 检索器封装
# ============================================================================

class _LiteHost:
    """
    轻量宿主对象：为复用 LangGraphRAGApp 的检索方法提供最小依赖集。

    检索链路（_do_retrieve）只需要 self.vector_db / self.user / self.tenant_id
    这三份**状态**，因此不必构造完整的 App（省掉 MySQL / Redis / LLM 网关 /
    提示词管理器的初始化）。

    但 _do_retrieve 内部还会回调若干**兄弟方法**（_rrf_fuse_queries、_rerank，
    以及未来可能新增的辅助方法）。为了避免"线上加一个 self._xxx 调用，评测就崩"
    这种漂移，这里用 __getattr__ 兜底：凡本对象没有的属性，自动到真实的
    LangGraphRAGApp 上取同名函数并绑定到 self。

    效果：评测跑的永远是线上那份检索代码，harness 不需要跟着改。
    """

    def __init__(self, vector_db, user_id: str, tenant_id: str):
        self.vector_db = vector_db
        self.user = user_id
        self.tenant_id = tenant_id
        # 其余属性（如 username、current_task_id）走 __getattr__ 兜底

    def __getattr__(self, name: str):
        # 注意：__getattr__ 仅在常规查找失败后才被调用，
        # 所以 vector_db / user / tenant_id 不会走到这里。
        if name.startswith("__"):
            raise AttributeError(name)
        import langgraph_rag_agent as L
        attr = getattr(L.LangGraphRAGApp, name, None)
        if attr is None:
            raise AttributeError(
                f"_LiteHost 与 LangGraphRAGApp 均无属性 {name!r}；"
                f"若线上检索链路新增了依赖，请在此补齐最小状态。")
        if callable(attr):
            return attr.__get__(self, type(self))   # 绑定为本对象的方法
        return attr


class Retriever:
    """
    评测用检索器。负责按指定模式执行检索，并把结果规整成统一结构。

    参数：
        mode:      raw | pipeline
        fetch_k:   检索深度
        overrides: 临时覆盖的全局配置，用于做 A/B 对比，可用键：
                     hybrid        bool  是否开启 dense+BM25 混合召回
                     rerank        bool  是否开启 cross-encoder 精排
                     top_k         int   线上 RETRIEVE_TOP_K
                     candidate_k   int   RRF 候选池 RETRIEVE_CANDIDATE_K
    """

    def __init__(self, mode: str = "pipeline", fetch_k: int = DEFAULT_FETCH_K,
                 overrides: Optional[Dict[str, Any]] = None):
        self.mode = mode
        self.fetch_k = fetch_k
        self.overrides = overrides or {}
        self._lg = None          # langgraph_rag_agent 模块
        self._acl = None         # AccessControlFilter
        self.vector_db = None
        self._saved: Dict[str, Any] = {}   # 记录被覆盖前的原值，用于还原

    # ---------------------------------------------------------------- setup
    def setup(self) -> Dict[str, Any]:
        """
        连接向量库并应用配置覆盖。返回本次生效的配置快照（写进 run.config，
        保证事后能解释指标差异从何而来）。
        """
        import advanced_rag_agent as A
        import langgraph_rag_agent as L

        self._lg = L
        self._acl = A.AccessControlFilter
        self.vector_db = A.VectorStoreManager.init_vector_store()

        # ---- 应用覆盖，并记录原值以便还原 ----
        ov = self.overrides
        if "hybrid" in ov:
            self._saved["hybrid"] = getattr(self.vector_db, "hybrid", None)
            self.vector_db.hybrid = bool(ov["hybrid"])
        if "rerank" in ov:
            self._saved["RERANK_ENABLED"] = L.RERANK_ENABLED
            L.RERANK_ENABLED = bool(ov["rerank"])
        if "top_k" in ov:
            self._saved["RETRIEVE_TOP_K"] = L.RETRIEVE_TOP_K
            L.RETRIEVE_TOP_K = int(ov["top_k"])
        if "candidate_k" in ov:
            self._saved["RETRIEVE_CANDIDATE_K"] = L.RETRIEVE_CANDIDATE_K
            L.RETRIEVE_CANDIDATE_K = int(ov["candidate_k"])

        # pipeline 模式下，检索深度要顶到 fetch_k，否则算不出真实的 bury
        if self.mode == "pipeline":
            self._saved.setdefault("RETRIEVE_TOP_K", L.RETRIEVE_TOP_K)
            self._saved.setdefault("RETRIEVE_CANDIDATE_K", L.RETRIEVE_CANDIDATE_K)
            L.RETRIEVE_TOP_K = max(L.RETRIEVE_TOP_K, self.fetch_k)
            L.RETRIEVE_CANDIDATE_K = max(L.RETRIEVE_CANDIDATE_K, self.fetch_k * 2)

        return {
            "mode": self.mode,
            "fetch_k": self.fetch_k,
            "hybrid": getattr(self.vector_db, "hybrid", None),
            "rerank_enabled": L.RERANK_ENABLED,
            "retrieve_top_k": L.RETRIEVE_TOP_K,
            "retrieve_candidate_k": L.RETRIEVE_CANDIDATE_K,
            "collection": getattr(self.vector_db, "collection", "?"),
            "embed_model": os.getenv("OLLAMA_EMBED_MODEL", "?"),
        }

    def teardown(self):
        """还原被覆盖的全局配置，避免污染同进程内的后续评测。"""
        L = self._lg
        if L is None:
            return
        for key, val in self._saved.items():
            if key == "hybrid":
                if self.vector_db is not None and val is not None:
                    self.vector_db.hybrid = val
            else:
                setattr(L, key, val)
        self._saved.clear()

    # ------------------------------------------------------------- retrieve
    def retrieve(self, case: RetrievalCase) -> List[Dict[str, Any]]:
        """
        执行一次检索，返回规整后的命中列表（按排名顺序）。

        每个元素：{rank, content, file_name, page, chunk_index, score, ...}
        """
        if self.mode == "raw":
            pairs = self.vector_db.similarity_search_with_score(
                case.query, k=self.fetch_k, filter_role=case.role,
                user_id=case.user_id, tenant_id=case.tenant_id,
            )
            # 线上在向量检索后还有一层应用层权限过滤，评测必须保持一致，
            # 否则会漏掉"被权限误杀"这类故障（根因 R5）
            pairs = self._acl.filter_results(pairs, case.role)
        else:
            host = _LiteHost(self.vector_db, case.user_id, case.tenant_id)
            # 只传原句，不做 LLM 改写 —— 保证本层评测零 LLM 成本
            pairs = self._lg.LangGraphRAGApp._do_retrieve(host, [case.query], case.role)

        out: List[Dict[str, Any]] = []
        for i, (doc, score) in enumerate(pairs[: self.fetch_k], start=1):
            meta = getattr(doc, "metadata", {}) or {}
            content = getattr(doc, "page_content", "") or ""
            out.append({
                "rank": i,
                "score": float(score) if isinstance(score, (int, float)) else None,
                "content": content,
                "file_name": meta.get("file_name") or meta.get("source") or "",
                "page": meta.get("page"),
                "chunk_index": meta.get("chunk_index"),
                "section_path": meta.get("section_path"),
                "access_level": meta.get("access_level"),
            })
        return out


# ============================================================================
# 单条 case 评测
# ============================================================================

def evaluate_case(retriever: Retriever, case: RetrievalCase,
                  ks: List[int] = None) -> RetrievalCaseResult:
    """
    跑一条检索 case，返回带全部指标的结果。

    命中判定：对每条 relevant 规则，从上往下扫描检索结果，
    找到第一个满足该规则的排名。注意是「每条规则各自找自己的首个命中」，
    而不是「一个检索结果只能命中一条规则」——
    因为同一段文本理论上可以同时满足多条标注规则，
    分开统计才能正确反映召回完整性。
    """
    ks = ks or DEFAULT_KS
    t0 = time.time()
    try:
        retrieved = retriever.retrieve(case)
        err = ""
    except Exception as e:
        import traceback
        traceback.print_exc()
        retrieved, err = [], f"{type(e).__name__}: {e}"
    latency = (time.time() - t0) * 1000

    hit_ranks: List[int] = []
    for spec in case.relevant:
        rank = -1
        for item in retrieved:
            meta = {
                "file_name": item["file_name"],
                "file_path": item["file_name"],
                "page": item["page"],
            }
            if spec.matches(meta, item["content"]):
                rank = item["rank"]
                break
        hit_ranks.append(rank)

    metrics = compute_retrieval_metrics(case, hit_ranks, ks)

    # 禁止规格扫描：命中即为泄漏（隔离类负例的核心判据）。
    # 与 relevant 扫描分开做，因为一条 case 可以同时有正例要求和禁止要求。
    forbidden_hits: List[str] = []
    for spec in case.forbidden:
        for item in retrieved:
            meta = {
                "file_name": item["file_name"],
                "file_path": item["file_name"],
                "page": item["page"],
            }
            if spec.matches(meta, item["content"]):
                forbidden_hits.append(
                    f"rank{item['rank']}:{item['file_name']}"
                    f"{'p' + str(item['page']) if item.get('page') is not None else ''}")
                break

    # 结果里只保留每条命中的摘要，避免 run json 膨胀到几十 MB。
    # 正文截断到 160 字符足够人工在报表里判断"这条召回对不对"。
    slim = [{
        "rank": it["rank"],
        "file_name": it["file_name"],
        "page": it["page"],
        "score": it["score"],
        "preview": (it["content"] or "").replace("\n", " ")[:160],
    } for it in retrieved[:10]]

    return RetrievalCaseResult(
        case_id=case.case_id,
        query=case.query,
        retrieved=slim,
        hit_ranks=hit_ranks,
        recall_at_k=metrics["recall_at_k"],
        ndcg_at_k=metrics["ndcg_at_k"],
        mrr=metrics["mrr"],
        bury=metrics["bury"],
        latency_ms=round(latency, 1),
        tags=case.tags,
        error=err,
        is_negative=case.is_negative,
        forbidden_hits=forbidden_hits,
    )


# ============================================================================
# 批量评测
# ============================================================================

def run_suite(cases: List[RetrievalCase],
              mode: str = "pipeline",
              fetch_k: int = DEFAULT_FETCH_K,
              ks: List[int] = None,
              overrides: Optional[Dict[str, Any]] = None,
              note: str = "",
              quiet: bool = True) -> EvalRun:
    """
    跑完整检索评测套件。

    参数：
        cases:     黄金集
        mode:      raw | pipeline
        fetch_k:   检索深度
        ks:        计算指标的 k 值列表
        overrides: 配置覆盖（做 A/B 对比用）
        quiet:     静音底层管线的 print（评测跑几十条时日志会淹没进度条）

    返回 EvalRun，含逐条结果与聚合指标。
    """
    ks = ks or DEFAULT_KS
    retriever = Retriever(mode=mode, fetch_k=fetch_k, overrides=overrides)
    config = retriever.setup()
    config["ks"] = ks

    run = EvalRun.new("retrieval", config)
    run.note = note

    # 底层检索链路 print 很多（rerank / figure-aware / BM25 回退提示），
    # 批量跑时全刷出来会看不见进度。默认吞掉，出错时仍会打印堆栈。
    import io, contextlib
    total = len(cases)
    results: List[Dict[str, Any]] = []

    print(f"\n[evalkit] 检索评测开始：{total} 条 case，mode={mode}，fetch_k={fetch_k}")
    print(f"[evalkit] 配置：hybrid={config.get('hybrid')} "
          f"rerank={config.get('rerank_enabled')} "
          f"top_k={config.get('retrieve_top_k')}")

    try:
        for i, case in enumerate(cases, 1):
            if quiet:
                with contextlib.redirect_stdout(io.StringIO()):
                    res = evaluate_case(retriever, case, ks)
            else:
                res = evaluate_case(retriever, case, ks)
            # passed 是 property，不在 __dict__ 里；显式落一份，
            # 否则 aggregate / 报表 / 门禁全都取不到判定结果。
            row = dict(res.__dict__)
            row["_passed"] = res.passed
            results.append(row)

            flag = "✓" if res.passed else "✗"
            if res.is_negative:
                bury_txt = "泄漏!" if res.forbidden_hits else "已隔离"
                metric_txt = f"泄漏={len(res.forbidden_hits)}"
            else:
                bury_txt = "未召回" if res.bury <= 0 else f"rank {res.bury}"
                metric_txt = f"MRR={res.mrr:.3f}"
            print(f"  [{i}/{total}] {flag} {case.case_id:<16} {bury_txt:<8} "
                  f"{metric_txt}  {case.query[:34]}")
    finally:
        retriever.teardown()

    run.results = results
    run.summary = aggregate(results, ks)
    run.finished_at = time.time()

    s = run.summary
    print(f"\n[evalkit] 完成：{s['count']} 条 | "
          f"通过率 {s['pass_rate']:.1%} | MRR {s['mrr']:.3f} | "
          f"Recall@5 {s.get('recall@5', 0):.3f} | nDCG@5 {s.get('ndcg@5', 0):.3f}")
    print(f"[evalkit] 诊断：完全未召回 {s['miss_count']} 条（召回侧问题）| "
          f"召回但排在 5 名外 {s['buried_count']} 条（排序侧问题）")
    leak = s.get("leak_count", 0)
    if s.get("negative_count"):
        tip = "❌ 存在跨租户泄漏，属安全问题需立即修复" if leak else "✔ 隔离正常"
        print(f"[evalkit] 隔离负例：{s['negative_count']} 条，泄漏 {leak} 条 —— {tip}")
    return run
