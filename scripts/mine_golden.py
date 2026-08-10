# -*- coding: utf-8 -*-
"""
从历史 trace 自动挖掘检索黄金集（evalkit 数据来源之二）
============================================================================

为什么要挖：
    手写黄金集覆盖的是「我们以为用户会问的」，而 trace 里是「用户真正问的」。
    后者才是线上质量的真实分布，也是回归测试最该守住的阵地。

挖掘依据：
    task_checkpoints.state_json 里保存了 LangGraph 每个节点后的完整状态，
    其中 grade_docs 节点产出的 doc_grades（LLM 判定每篇召回文档是否相关）
    就是一份**免费的人工标注替代品**——被判为相关的文档，
    正是这条 query 应该召回的目标。

抗重建标注（关键设计）：
    不用 chunk_index 当标识——重新 ingest 后切片边界会变，标注立即失效。
    改用 file_name + pages + keywords 三重定位（或关系兜底），
    跨越多次索引重建依然有效。

质量门槛（宁缺毋滥，脏数据比没数据更糟）：
    1. 至少有 1 篇被判定相关的文档
    2. query 长度 >= MIN_QUERY_LEN，过滤 "你好" 之类的闲聊
    3. 同一 query 只保留相关文档最多的那次（同题多次问，取信息最全的）
    4. 与既有黄金集按归一化 query 去重，不产生重复题目

用法：
    python scripts/mine_golden.py --dry-run          # 只预览，不落盘
    python scripts/mine_golden.py                    # 写入 mined_retrieval.jsonl
    python scripts/mine_golden.py --merge            # 追加进主黄金集 retrieval.jsonl
    python scripts/mine_golden.py --limit 200 --min-grade 2
"""

import os
import re
import sys
import json
import argparse
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymysql

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from evalkit.schema import GOLDEN_DIR, load_jsonl

# ---------------------------------------------------------------- 配置
MIN_QUERY_LEN = 4          # 过滤闲聊
MAX_KEYWORDS = 2           # 每条 relevant 最多留几个关键词（多了容易过拟合切片）
KEYWORD_MIN_LEN = 4        # 关键词最短长度
MAX_PAGES_PER_SPEC = 4     # 单条 spec 最多记几页，超了说明定位太散、参考价值低


def _conn():
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "192.168.200.128"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "Root@2026"),
        database=os.getenv("MYSQL_DATABASE", "rag_agent"),
        charset="utf8mb4", connect_timeout=10)


def norm_query(q):
    """归一化 query 用于去重：去空白、去标点、转小写。"""
    return re.sub(r"[\s\W_]+", "", (q or "").lower())


def scan_knowledge_base():
    """
    扫描当前知识库，返回 {tenant: {file_name, ...}}。

    为什么必须做这一步：
        trace 是历史记录，里面会引用**早已被删除或改名的文档**
        （实测挖到过 Jimi_IoT__V1.21.pdf，当前库里根本不存在）。
        把这种标注写进黄金集，等于制造了一条永远失败的 case——
        评测会一直报红，但代码毫无问题，纯属自己骗自己。
        所以只保留能在当前知识库里找到的文件。
    """
    root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge")
    kb = {}
    if not os.path.isdir(root):
        return kb
    for tenant in os.listdir(root):
        tdir = os.path.join(root, tenant)
        if os.path.isdir(tdir):
            kb[tenant] = {f for f in os.listdir(tdir) if not f.startswith(".")}
    return kb


def guess_tenant(file_path, session_id):
    """
    推断租户。优先从文档路径推断（knowledge/<tenant>/xxx.pdf），
    这是最可靠的信号；退而求其次从 session_id（web:admin:yh_admin）里取。
    """
    if file_path:
        m = re.search(r"knowledge[/\\]([^/\\]+)[/\\]", file_path)
        if m:
            return m.group(1)
    if session_id:
        m = re.search(r"(jm|yh)", session_id)
        if m:
            return m.group(1)
    return "default"


# 这些片段是切片器留下的结构标记，不是文档内容。
# 用它们当关键词等于标了个永远匹配不上的锚点（实测挖到过 keywords=["[Page 4]"]），
# 会伪造出一条永远失败的 case。
_JUNK_KW = re.compile(
    r"^(\[?page\s*\d+\]?|第?\s*\d+\s*页|图\s*\d+|表\s*\d+|[\d\s\W_]+)$",
    re.IGNORECASE)


def _clean_keyword(text):
    """清洗候选关键词，返回 None 表示不可用。"""
    if not text:
        return None
    t = re.sub(r"\[Page\s*\d+\]", "", text, flags=re.IGNORECASE).strip()
    t = re.sub(r"^[\s\d０-９一二三四五六七八九十、.．\-]+", "", t).strip()
    if len(t) < KEYWORD_MIN_LEN or _JUNK_KW.match(t):
        return None
    return t[:24]


def extract_keywords(section_path, content):
    """
    抽取抗重建的关键词。

    优先用 section_path（章节标题）——它由文档结构决定，
    重新切片后依然存在，比正文片段稳定得多。
    退化时才从正文取首个较长的短语，并过滤掉页码/图表编号这类结构噪声。
    """
    kws = []
    kw = _clean_keyword(section_path)
    if kw:
        kws.append(kw)
    if not kws and content:
        for seg in re.split(r"[\n。；;，,：:]", content):
            kw = _clean_keyword(seg)
            if kw:
                kws.append(kw)
                break
    return kws[:MAX_KEYWORDS]


def mine(limit=500, min_grade=1, verbose=False):
    """
    扫描 trace，产出候选 case 列表。

    返回 [{case_id, query, relevant, tenant_id, role, tags, source, note}, ...]
    """
    conn = _conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    try:
        # 每个 thread 取含 doc_grades 的最新一条快照（状态最完整）
        cur.execute("""
            SELECT c.thread_id, c.session_id, c.state_json, c.created_at,
                   q.role, q.status
            FROM task_checkpoints c
            LEFT JOIN task_queue q ON q.task_id = c.thread_id
            WHERE c.state_json LIKE %s
            ORDER BY c.id DESC
            LIMIT %s
        """, ("%doc_grades%", limit))
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    kb = scan_knowledge_base()
    print(f"[mine] 扫描 {len(rows)} 条快照…")
    print(f"[mine] 当前知识库："
          + "，".join(f"{t}({len(fs)}份)" for t, fs in kb.items()))

    # query 归一化 -> 最佳候选（相关文档最多的那次）
    best = OrderedDict()
    skipped = {"parse": 0, "no_grade": 0, "short_query": 0, "no_relevant": 0,
               "stale_file": 0, "cross_tenant": 0}

    for row in rows:
        try:
            st = json.loads(row["state_json"])
        except Exception:
            skipped["parse"] += 1
            continue

        query = (st.get("query") or "").strip()
        if len(query) < MIN_QUERY_LEN:
            skipped["short_query"] += 1
            continue

        grades = st.get("doc_grades") or []
        docs = st.get("retrieved_docs") or []
        if not grades or not docs:
            skipped["no_grade"] += 1
            continue

        # doc_grades 可能因多轮检索比 retrieved_docs 长，按较短的对齐，
        # 避免越界把不相关文档误标成相关
        n = min(len(grades), len(docs))

        # file_name -> {pages, keywords}
        by_file = OrderedDict()
        for i in range(n):
            if not grades[i]:
                continue
            item = docs[i]
            doc = item[0] if isinstance(item, (list, tuple)) else item
            if not isinstance(doc, dict):
                continue
            md = doc.get("metadata") or {}
            fname = md.get("file_name") or ""
            if not fname:
                continue
            slot = by_file.setdefault(fname, {
                "pages": [], "keywords": [],
                "file_path": md.get("source") or md.get("file_path") or "",
            })
            page = md.get("page")
            if isinstance(page, int) and page not in slot["pages"]:
                slot["pages"].append(page)
            for kw in extract_keywords(md.get("section_path"), doc.get("page_content")):
                if kw not in slot["keywords"]:
                    slot["keywords"].append(kw)

        if not by_file:
            skipped["no_relevant"] += 1
            continue

        # ---- 知识库校验：剔除已不存在的历史文档 ----
        owner = {}          # fname -> tenant
        for fname in list(by_file):
            hit = [t for t, files in kb.items() if fname in files]
            if hit:
                owner[fname] = hit[0]
            else:
                by_file.pop(fname)
                skipped["stale_file"] += 1
        if not by_file:
            skipped["no_relevant"] += 1
            continue

        # ---- 租户归属：由文件实际所在目录决定，比 session_id 猜测可靠 ----
        tenants = {owner[f] for f in by_file}
        if len(tenants) > 1:
            # 跨租户混标（多为 admin 越权全库检索留下的记录）。
            # 保留占多数的租户，其余剔除；平票则整条丢弃，避免制造歧义 case。
            counts = {}
            for f in by_file:
                counts[owner[f]] = counts.get(owner[f], 0) + 1
            top = max(counts.values())
            winners = [t for t, c in counts.items() if c == top]
            if len(winners) > 1:
                skipped["cross_tenant"] += 1
                continue
            keep = winners[0]
            for f in [f for f in by_file if owner[f] != keep]:
                by_file.pop(f)
            tenant = keep
        else:
            tenant = next(iter(tenants))

        relevant_cnt = len(by_file)
        if relevant_cnt < min_grade:
            skipped["no_relevant"] += 1
            continue

        key = norm_query(query)
        prev = best.get(key)
        if prev and prev["_relevant_cnt"] >= relevant_cnt:
            continue    # 已有更好的同题样本

        relevant = []
        for fname, slot in by_file.items():
            spec = {"file": fname, "gain": 3}
            if slot["pages"]:
                spec["pages"] = sorted(slot["pages"])[:MAX_PAGES_PER_SPEC]
            if slot["keywords"]:
                spec["keywords"] = slot["keywords"][:MAX_KEYWORDS]
            relevant.append(spec)

        best[key] = {
            "query": query,
            "relevant": relevant,
            "tenant_id": tenant,
            "role": row.get("role") or st.get("role") or "user",
            "_relevant_cnt": relevant_cnt,
            "_thread": row["thread_id"],
            "_created": str(row.get("created_at") or ""),
        }

    print(f"[mine] 去重后候选 {len(best)} 条 | 跳过：{skipped}")
    return best


def to_cases(best, existing_norm, prefix="mined"):
    """把候选转成标准 case，并剔除与既有黄金集重复的题目。"""
    cases, dup = [], 0
    idx = 0
    for key, c in best.items():
        if key in existing_norm:
            dup += 1
            continue
        idx += 1
        cases.append({
            "case_id": f"{prefix}-{idx:03d}",
            "query": c["query"],
            "tenant_id": c["tenant_id"],
            "role": c["role"],
            "tags": ["mined", c["tenant_id"]],
            "relevant": c["relevant"],
            "source": "mined",
            "note": f"挖自 trace {c['_thread']}（{c['_created']}），"
                    f"LLM 判定相关文档 {c['_relevant_cnt']} 篇",
        })
    if dup:
        print(f"[mine] 与既有黄金集重复，跳过 {dup} 条")
    return cases


def verify(cases, fetch_k=20):
    """
    用当前检索器把挖出来的 case 跑一遍，分成「可信」与「存疑」两组。

    为什么必须做：
        标注来自 LLM 的 doc_grades，本身带噪声——实测挖到过
        「输出一下通信流程图」被标到"设备信息结构""白名单获取"这种
        明显无关的章节上。这类标注进了黄金集，评测就会长期报红，
        而问题其实出在标注、不在系统，最终导致所有人不再相信这份报告。

    注意：验证不通过 ≠ 系统有 bug，只是「这条标注不够可信」，
    因此不丢弃，而是单独存到 *_review.jsonl 供人工裁决。
    """
    from evalkit.harness_retrieval import Retriever, evaluate_case
    from evalkit.schema import RetrievalCase
    import contextlib, io

    retriever = Retriever(mode="pipeline", fetch_k=fetch_k)
    retriever.setup()
    trusted, review = [], []
    try:
        for c in cases:
            case = RetrievalCase.from_dict(c)
            with contextlib.redirect_stdout(io.StringIO()):
                res = evaluate_case(retriever, case)
            if res.passed:
                trusted.append(c)
                print(f"  ✓ {c['case_id']} rank {res.bury:<3} {c['query'][:30]}")
            else:
                c["note"] = (c.get("note", "") +
                             " | ⚠ 自校验未命中：标注可信度存疑，需人工确认")
                review.append(c)
                print(f"  ? {c['case_id']} 未命中   {c['query'][:30]}")
    finally:
        retriever.teardown()
    return trusted, review


def main():
    ap = argparse.ArgumentParser(description="从历史 trace 挖掘检索黄金集")
    ap.add_argument("--limit", type=int, default=500, help="扫描的快照条数上限")
    ap.add_argument("--min-grade", type=int, default=1, help="至少几篇相关文档才收")
    ap.add_argument("--dry-run", action="store_true", help="只预览不落盘")
    ap.add_argument("--merge", action="store_true",
                    help="追加进主黄金集 retrieval.jsonl（默认写独立文件）")
    ap.add_argument("--out", default="", help="输出路径")
    ap.add_argument("--no-verify", action="store_true",
                    help="跳过自校验（默认会跑一遍检索过滤标注噪声）")
    args = ap.parse_args()

    main_golden = os.path.join(GOLDEN_DIR, "retrieval.jsonl")
    existing = load_jsonl(main_golden) if os.path.isfile(main_golden) else []
    existing_norm = {norm_query(d.get("query", "")) for d in existing}
    print(f"[mine] 既有黄金集 {len(existing)} 条")

    best = mine(limit=args.limit, min_grade=args.min_grade)
    cases = to_cases(best, existing_norm)

    if not cases:
        print("[mine] 没有可新增的 case（可能都已在黄金集中）")
        return 0

    print(f"\n[mine] 产出 {len(cases)} 条新 case：")
    for c in cases:
        files = "、".join(r["file"][:26] for r in c["relevant"])
        print(f"  {c['case_id']}  [{c['tenant_id']}] {c['query'][:34]:<36} "
              f"→ {len(c['relevant'])} 文档：{files[:60]}")

    review = []
    if not args.no_verify:
        print(f"\n[mine] 自校验：用当前检索器跑一遍，过滤标注噪声…")
        cases, review = verify(cases)
        print(f"[mine] 可信 {len(cases)} 条 | 存疑 {len(review)} 条")

    if args.dry_run:
        print("\n[mine] --dry-run，未落盘")
        return 0

    if not cases and not review:
        print("[mine] 无可落盘内容")
        return 0

    out = args.out or (main_golden if args.merge
                       else os.path.join(GOLDEN_DIR, "mined_retrieval.jsonl"))
    if cases:
        mode = "a" if args.merge else "w"
        with open(out, mode, encoding="utf-8") as f:
            for c in cases:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"\n[mine] 已{'追加' if args.merge else '写入'} {len(cases)} 条可信 case → {out}")

    if review:
        rpath = os.path.join(GOLDEN_DIR, "mined_review.jsonl")
        with open(rpath, "w", encoding="utf-8") as f:
            for c in review:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"[mine] {len(review)} 条存疑 case → {rpath}")
        print("[mine] 存疑不等于系统有 bug，多为 LLM 标注噪声；"
              "人工确认后可手工并入主黄金集。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
