# -*- coding: utf-8 -*-
"""
把 config/init_db.sql 里新增的表（qa_feedback / bad_cases）应用到 MySQL。

幂等：DDL 全部是 CREATE TABLE IF NOT EXISTS，重复执行安全。
只抽取这两张表的建表语句执行，不动既有 6 张表与种子数据。

用法：
    python scripts/apply_new_tables.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymysql

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

HOST = os.getenv("MYSQL_HOST", "192.168.200.128")
PORT = int(os.getenv("MYSQL_PORT", "3306"))
USER = os.getenv("MYSQL_USER", "root")
PWD = os.getenv("MYSQL_PASSWORD", "Root@2026")
DB = os.getenv("MYSQL_DATABASE", "rag_agent")

TARGET_TABLES = ["qa_feedback", "bad_cases"]

SQL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "config", "init_db.sql")


def extract_ddl(sql_text, table):
    """从完整 init_db.sql 中抽出指定表的 CREATE TABLE 语句（到第一个分号）。"""
    pattern = re.compile(
        r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+`?" + re.escape(table) + r"`?\s*\(.*?;",
        re.IGNORECASE | re.DOTALL)
    m = pattern.search(sql_text)
    return m.group(0) if m else None


def main():
    with open(SQL_PATH, "r", encoding="utf-8") as f:
        sql_text = f.read()

    conn = pymysql.connect(host=HOST, port=PORT, user=USER, password=PWD,
                           database=DB, charset="utf8mb4", connect_timeout=10)
    cur = conn.cursor()
    try:
        for t in TARGET_TABLES:
            ddl = extract_ddl(sql_text, t)
            if not ddl:
                print(f"  ✗ 在 init_db.sql 中未找到表 {t} 的 DDL")
                continue
            cur.execute(ddl)
            print(f"  ✔ 已应用 {t}")

        conn.commit()

        cur.execute("SHOW TABLES")
        tables = sorted(r[0] for r in cur.fetchall())
        print(f"\n当前库 {DB} 共 {len(tables)} 张表：")
        for t in tables:
            cur.execute(f"SELECT COUNT(*) FROM `{t}`")
            print(f"  - {t:<20} {cur.fetchone()[0]} 行")

        for t in TARGET_TABLES:
            cur.execute(f"SHOW COLUMNS FROM `{t}`")
            cols = [r[0] for r in cur.fetchall()]
            print(f"\n{t} 字段（{len(cols)}）：{', '.join(cols)}")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
