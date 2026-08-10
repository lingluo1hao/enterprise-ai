# -*- coding: utf-8 -*-
"""一次性探针：查看 task_checkpoints.state_json 的真实结构，供挖黄金集设计字段映射。"""
import os
import sys
import json
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pymysql

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

conn = pymysql.connect(
    host=os.getenv("MYSQL_HOST", "192.168.200.128"),
    port=int(os.getenv("MYSQL_PORT", "3306")),
    user=os.getenv("MYSQL_USER", "root"),
    password=os.getenv("MYSQL_PASSWORD", "Root@2026"),
    database=os.getenv("MYSQL_DATABASE", "rag_agent"),
    charset="utf8mb4", connect_timeout=10)
cur = conn.cursor()

cur.execute("SELECT node_name, COUNT(*) FROM task_checkpoints GROUP BY node_name ORDER BY 2 DESC")
print("节点分布：")
for name, cnt in cur.fetchall():
    print(f"  {name:<28} {cnt}")

cur.execute("SELECT status, COUNT(*) FROM task_queue GROUP BY status")
print("\n任务状态分布：", dict(cur.fetchall()))

# 找一条含 doc_grades 的快照
cur.execute("""SELECT thread_id, node_name, state_json FROM task_checkpoints
               WHERE state_json LIKE '%doc_grades%'
               ORDER BY id DESC LIMIT 1""")
row = cur.fetchone()
if not row:
    print("\n未找到含 doc_grades 的快照")
else:
    thread_id, node, sj = row
    st = json.loads(sj)
    print(f"\n样本快照 thread={thread_id} node={node}")
    print("顶层键：")
    for k, v in st.items():
        t = type(v).__name__
        if isinstance(v, list):
            desc = f"list[{len(v)}]"
        elif isinstance(v, str):
            desc = f"str({len(v)}) {v[:60]!r}"
        else:
            desc = f"{t} {str(v)[:60]}"
        print(f"  {k:<22} {desc}")

    print("\ndoc_grades:", st.get("doc_grades"))
    docs = st.get("retrieved_docs") or []
    print(f"retrieved_docs 共 {len(docs)} 条，首条结构：")
    if docs:
        d = docs[0]
        print("  type:", type(d).__name__)
        print("  raw:", json.dumps(d, ensure_ascii=False)[:600])

# 统计有多少任务可挖（completed + 有 grade=True）
cur.execute("""SELECT COUNT(DISTINCT thread_id) FROM task_checkpoints
               WHERE state_json LIKE '%"doc_grades"%'""")
print("\n含 doc_grades 的任务数：", cur.fetchone()[0])

cur.execute("SELECT query, status FROM task_queue ORDER BY id DESC LIMIT 15")
print("\n最近 15 条提问：")
for q, s in cur.fetchall():
    print(f"  [{s:<10}] {q[:60]}")

cur.close()
conn.close()
