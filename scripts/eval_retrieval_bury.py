"""
最小检索量化脚本：量化「原句被追加到改写列表末尾」导致精确文档被埋的问题。

只复刻 langgraph_rag_agent._do_retrieve 的「跨多个改写 query 合并」这一步
（底层 per-query 检索直接复用项目真实的 Milvus + Ollama bge-m3 hybrid+RRF），
公平对比三种合并策略对「原句精确命中」的影响：

  CURRENT : 改写词在前、原句末尾；去重锁首命中分数；按分数升序 (现有代码 line 1472)
  FIXED-A : 原句置顶 + 去重保留最佳(最小)分数；按分数升序 (最小对齐改动)
  RRF     : 跨 query 做 Reciprocal Rank Fusion，每 query 等权投票 (大厂做法)

gold = 原句单独检索的 top-1 文档（即「最精准的信息」）。
指标：gold 在各策略合并结果中的排名(1=最相关) 与 是否进入 top-5。

依赖：anaconda py310（含 pymilvus / langchain_ollama）+ 虚拟机 Milvus/Ollama 在线。
运行：D:\prom\anaconda\envs\py310\python.exe scripts/eval_retrieval_bury.py
"""
import os, re, sys
from collections import defaultdict

# ---------- 连接配置（与项目一致） ----------
MILVUS_URI = os.getenv("MILVUS_URI", "http://192.168.200.128:19530")
COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_docs")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.200.128:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
RETRIEVE_TOP_K = 5
HYBRID = os.getenv("HYBRID_SEARCH", "true").lower() != "false"
ROLE_ADMIN = "admin"
RRF_K = 60

# ---------- 最小检索实现（复刻 advanced_rag_agent._milvus_search） ----------
from pymilvus import MilvusClient
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

_client = MilvusClient(uri=MILVUS_URI)
_embed = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)

FIELDS = ["content", "file_name", "file_path", "access_level", "chunk_index",
          "chunk_type", "figure_paths", "page", "section_path",
          "parent_id", "parent_content", "is_parent"]


def _parse_hits(hits):
    out = []
    for hit in hits:
        e = getattr(hit, "entity", {}) or {}
        meta = {
            "source": e.get("file_path", ""), "file_name": e.get("file_name", ""),
            "access_level": e.get("access_level", "public"),
            "chunk_index": e.get("chunk_index", 0), "parent_id": e.get("parent_id", ""),
            "page": e.get("page", None), "chunk_type": e.get("chunk_type", "prose"),
            "figure_paths": list(e.get("figure_paths") or []),
            "section_path": e.get("section_path", "") or "",
        }
        page_content = e.get("parent_content") or e.get("content", "")
        out.append((hit.id, Document(page_content=page_content[:8192], metadata=meta),
                    float(getattr(hit, "distance", 0.0))))
    return out


def _rrf_fuse(dense_list, sparse_list, k, rrf_k=RRF_K):
    score, docs = {}, {}
    for rank, (hid, doc, _) in enumerate(dense_list):
        score[hid] = score.get(hid, 0.0) + 1.0 / (rrf_k + rank + 1)
        docs.setdefault(hid, doc)
    for rank, (hid, doc, _) in enumerate(sparse_list):
        score[hid] = score.get(hid, 0.0) + 1.0 / (rrf_k + rank + 1)
        docs.setdefault(hid, doc)
    ranked = sorted(score.keys(), key=lambda h: score[h], reverse=True)
    return [(docs[h], -score[h]) for h in ranked[:k]]


def vs_search(query, k, tenant_id):
    """复刻 _milvus_search：admin + 单租户 + is_parent==false；hybrid+RRF。返回 [(doc,dist)]，dist 越小越相关。"""
    expr = f'(tenant_id == "{tenant_id}") and (is_parent == false)'
    top = max(k * 2, 8)
    qvec = _embed.embed_query(query)
    dense = _parse_hits(_client.search(collection_name=COLLECTION, data=[qvec],
                                       anns_field="dense", limit=top, filter=expr,
                                       output_fields=FIELDS)[0])
    if not HYBRID:
        dense.sort(key=lambda x: x[2])
        return [(d, dist) for (_, d, dist) in dense[:k]]
    try:
        sparse = _parse_hits(_client.search(collection_name=COLLECTION, data=[query],
                                            anns_field="sparse", limit=top, filter=expr,
                                            output_fields=FIELDS)[0])
        return _rrf_fuse(dense, sparse, k)
    except Exception as e:
        print(f"  [warn] sparse 失败回退 dense: {e}")
        dense.sort(key=lambda x: x[2])
        return [(d, dist) for (_, d, dist) in dense[:k]]


def detect_tenant(query):
    best_t, best_dist = None, 1e9
    for t in ["yh", "jm", "default"]:
        res = vs_search(query, 5, t)
        if res and res[0][1] < best_dist:
            best_dist = res[0][1]; best_t = t
    return best_t or "default"


# ---------- 三种跨 query 合并策略 ----------
def keyof(doc):
    return doc.page_content[:80]


def merge_current(per_q):
    seen, out = set(), []
    for _, res in per_q:
        for doc, dist in res:
            k = keyof(doc)
            if k not in seen:
                seen.add(k); out.append((doc, dist))
    out.sort(key=lambda x: x[1])
    return out


def merge_fixedA(per_q):
    best = {}
    for _, res in per_q:
        for doc, dist in res:
            k = keyof(doc)
            if k not in best or dist < best[k][1]:
                best[k] = (doc, dist)
    out = list(best.values()); out.sort(key=lambda x: x[1]); return out


def merge_rrf(per_q, rrf_k=RRF_K):
    score, docs = defaultdict(float), {}
    for _, res in per_q:
        for rank, (doc, _) in enumerate(res):
            k = keyof(doc)
            score[k] += 1.0 / (rrf_k + rank + 1)
            docs.setdefault(k, doc)
    ranked = sorted(score.keys(), key=lambda k: score[k], reverse=True)
    return [(docs[k], -score[k]) for k in ranked]


def rank_of(merged, gold_key, window=15):
    for i, (doc, _) in enumerate(merged):
        if keyof(doc) == gold_key:
            return i + 1
    return f">{len(merged)}" if len(merged) <= window else f">{window}"


def snippet(doc, n=46):
    return re.sub(r"\s+", " ", doc.page_content[:n]).strip()


# ---------- 评估集（来自真实日志） ----------
# Case1：diag_e2e.log line46 真实生产改写（已去掉 "1. 2. 3." 编号噪声）
CASE1_Q = "基站信息格式是什么"
CASE1_REWRITES = ["基站信息格式解析", "基站数据格式说明", "GSM基站信息结构"]

# Case2/3：日志里高频 query，改写由 Ollama 实时生成（复用真实 rewrite 思路）
EXTRA = ["基站信息格式", "VI 基站信息格式"]


def llm_rewrites(query):
    try:
        from langchain_ollama import ChatOllama
        chat = ChatOllama(model="qwen2:7b", base_url=OLLAMA_URL)
        sys_p = ("你是查询重写助手。将用户问题改写为3个更适合向量检索的短关键词组合，"
                 "每行一个，不要加编号。只输出关键词，不要解释。")
        txt = chat.invoke([
            {"role": "system", "content": sys_p},
            {"role": "user", "content": query},
        ]).content
        return [l.strip(" .、") for l in txt.strip().split("\n") if l.strip()][:3]
    except Exception as e:
        print(f"  [warn] 改写生成失败({e})，回退为原句复制")
        return [query + " 格式", query + " 说明", query + " 结构"]


def main():
    cases = [(CASE1_Q, CASE1_REWRITES)]
    for q in EXTRA:
        cases.append((q, llm_rewrites(q)))

    print("=" * 92)
    print(f"{'query':<22}{'gold_rank_CURRENT':>18}{'gold_rank_FIXED-A':>20}{'gold_rank_RRF':>14}")
    print("=" * 92)
    summary = {"bury_current": 0, "fix_a_recover": 0, "rrf_recover": 0, "n": 0}

    for q, rewrites in cases:
        tenant = detect_tenant(q)
        original = q
        # CURRENT 顺序：改写词在前、原句末尾；FIXED 顺序：原句置顶
        order_cur = rewrites + [original]
        order_fix = [original] + rewrites
        per_cur = [(qq, vs_search(qq, RETRIEVE_TOP_K, tenant)) for qq in order_cur]
        per_fix = [(qq, vs_search(qq, RETRIEVE_TOP_K, tenant)) for qq in order_fix]
        if not per_cur[-1][1]:
            print(f"{q:<22} [skip] 原句检索为空(tenant={tenant})")
            continue

        m_cur = merge_current(per_cur)
        m_fixa = merge_fixedA(per_fix)
        m_rrf = merge_rrf(per_cur)  # RRF 与顺序无关

        gold_key = keyof(per_cur[-1][1][0][0])  # 原句单独检索 top-1
        r_cur = rank_of(m_cur, gold_key)
        r_fixa = rank_of(m_fixa, gold_key)
        r_rrf = rank_of(m_rrf, gold_key)

        summary["n"] += 1
        try:
            cur_num = int(r_cur)
        except Exception:
            cur_num = 99
        if cur_num > 5:
            summary["bury_current"] += 1
        if int(r_fixa) <= 5 if str(r_fixa).isdigit() else False:
            summary["fix_a_recover"] += 1
        if str(r_rrf).isdigit() and int(r_rrf) <= 5:
            summary["rrf_recover"] += 1

        print(f"{q:<22}{str(r_cur):>18}{str(r_fixa):>20}{str(r_rrf):>14}")
        print(f"    tenant={tenant}  gold='{snippet(per_cur[-1][1][0][0], 60)}'")
        print(f"    CURRENT top3 : " + " | ".join(snippet(d) for d, _ in m_cur[:3]))
        print(f"    FIXED-A top3: " + " | ".join(snippet(d) for d, _ in m_fixa[:3]))
        print(f"    RRF     top3: " + " | ".join(snippet(d) for d, _ in m_rrf[:3]))
        print("-" * 92)

    print("\n=== 汇总（gold = 原句精确命中文档）===")
    print(f"评估 query 数            : {summary['n']}")
    print(f"当前代码 gold 被埋(>top5): {summary['bury_current']} / {summary['n']}")
    print(f"FIXED-A 救回(<=top5)     : {summary['fix_a_recover']} / {summary['n']}")
    print(f"RRF     救回(<=top5)     : {summary['rrf_recover']} / {summary['n']}")


if __name__ == "__main__":
    main()
