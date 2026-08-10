# -*- coding: utf-8 -*-
"""一次性种子脚本：向 bad_cases 表注入 6 条 demo 数据，便于在管理后台看到完整 master-detail 效果。

覆盖根因 R1~R6 + 三种状态（open / in_progress / resolved）+ 完整字段
（expected / diagnosis / resolved_by / resolved_at）。

注意：
- 这不是真实用户反馈，仅为演示。删除用 `scripts/clean_bad_cases.py` 或后台手动标记 resolved 后再清。
- 重复执行会先清空本脚本注入的 seed 标记数据（source='seed'），不会重复堆积。
- 依赖 .env 里的 MySQL 凭据；VM 192.168.200.128 需在线。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import pymysql

DB = os.getenv("MYSQL_DATABASE", "rag_agent")
HOST = os.getenv("MYSQL_HOST", "192.168.200.128")
PORT = int(os.getenv("MYSQL_PORT", "3306"))
USER = os.getenv("MYSQL_USER", "root")
PASS = os.getenv("MYSQL_PASSWORD", "")

# 6 条 demo，覆盖 R1~R6 + open/in_progress/resolved
SEED = [
    {
        "source": "seed", "suite": "retrieval", "root_cause": "R1",
        "query": "设备的 GPS 定位坐标系是什么？经纬度采用 WGS84 还是 GCJ02？",
        "answer": "根据文档，GPS 定位数据包含经纬度字段，但未明确说明坐标系类型。",
        "expected": "应明确说明坐标系（WGS84/GCJ02）及经纬度精度与小数位。",
        "diagnosis": '{"relevant":0,"bad_sources":[],"query_type":"fact"}',
        "status": "open", "resolved_by": "", "age_days": 0,
    },
    {
        "source": "seed", "suite": "answer", "root_cause": "R2",
        "query": "WIFIMAC 字段的含义是什么？",
        "answer": "WIFIMAC 是设备的 WiFi MAC 地址字段，用于唯一标识设备。",
        "expected": "WIFIMAC 为 WiFi MAC 地址，16 进制，用于设备唯一标识与定位辅助。",
        "diagnosis": '{"relevant":1,"bad_sources":["心跳协议说明.pdf","基站信息格式.pdf"],"query_type":"fact"}',
        "status": "open", "resolved_by": "", "age_days": 1,
    },
    {
        "source": "seed", "suite": "answer", "root_cause": "R4",
        "query": "0xFF 协议包的字段定义是什么？",
        "answer": "0xFF 协议包包含以下字段：0xB2 数据头、0xA0 数据体、0x80 校验位。",
        "expected": "0xFF 协议包应包含：包长度、设备ID、时间戳、数据区、CRC 校验。",
        "diagnosis": '{"relevant":2,"bad_sources":[],"query_type":"protocol"}',
        "status": "in_progress", "resolved_by": "", "age_days": 2,
    },
    {
        "source": "seed", "suite": "answer", "root_cause": "R5",
        "query": "心跳包的协议号是多少？",
        "answer": "心跳包使用的协议号是 0x01。",
        "expected": "心跳包协议号是 0x36（正确值）。",
        "diagnosis": '{"relevant":1,"bad_sources":[],"query_type":"protocol"}',
        "status": "open", "resolved_by": "", "age_days": 3,
    },
    {
        "source": "seed", "suite": "answer", "root_cause": "R6",
        "query": "WIFI 定位的完整流程是什么？",
        "answer": "WIFI 定位流程见文档第 3.2.1 节（实际上该章节不存在），首先扫描周围 AP。",
        "expected": "WIFI 定位流程：扫描 AP → 上报信号强度 → 服务端三角定位 → 返回坐标。",
        "diagnosis": '{"relevant":1,"bad_sources":["WIFI定位白皮书.pdf"],"query_type":"process"}',
        "status": "resolved", "resolved_by": "jm_admin", "age_days": 5,
    },
    {
        "source": "seed", "suite": "answer", "root_cause": "R7",
        "query": "请对这份 50 页的技术手册做完整摘要",
        "answer": "（生成超时，60s 内未完成）",
        "expected": "应分块摘要或给出超时降级提示，而非直接失败。",
        "diagnosis": '{"relevant":3,"bad_sources":[],"query_type":"summary"}',
        "status": "resolved", "resolved_by": "admin", "age_days": 7,
    },
]


def main():
    conn = pymysql.connect(host=HOST, port=PORT, user=USER, password=PASS,
                           database=DB, charset="utf8mb4")
    cur = conn.cursor()
    # 清掉上次 seed 的残留（避免重复堆积）
    cur.execute("DELETE FROM bad_cases WHERE source='seed'")
    print(f"  清理旧 seed 行: {cur.rowcount}")
    now = time.time()
    n = 0
    for r in SEED:
        created = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now - r["age_days"] * 86400))
        resolved = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now - (r["age_days"] - 1) * 86400)) \
            if r["status"] == "resolved" else None
        cur.execute(
            "INSERT INTO bad_cases (source, suite, case_id, query, answer, expected, "
            "root_cause, diagnosis, status, resolved_by, created_at, resolved_at) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (r["source"], r["suite"], "seed-" + r["root_cause"], r["query"], r["answer"],
             r["expected"], r["root_cause"], r["diagnosis"], r["status"],
             r["resolved_by"] or None, created, resolved),
        )
        n += 1
    conn.commit()
    cur.execute("SELECT status, COUNT(*) FROM bad_cases GROUP BY status")
    dist = cur.fetchall()
    conn.close()
    print(f"  注入 {n} 条 demo bad case")
    print("  当前状态分布:", dist)
    print("  打开 /admin/bad_cases 即可看到效果（需管理员登录）。")


if __name__ == "__main__":
    main()
