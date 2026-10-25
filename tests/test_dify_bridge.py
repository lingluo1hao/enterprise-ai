#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dify 双向桥自测（零外部依赖）
================================================================================

运行：
    python tests/test_dify_bridge.py          # 全跑
    python tests/test_dify_bridge.py -v       # 带 WARNING 日志（看退化路径告警）

【为什么这份测试一个外部服务都不连】

桥的两侧都是 HTTP 协议对接，真实环境要 Milvus + reranker + Dify 三样都在线
才能跑一次端到端。那样这测试就只能"偶尔跑一次"，等于没有。
所以这里把边界画在两处接缝上：

    · dify_kb_server.DifyRetrievalService._retrieve_pairs
      —— 唯一调线上检索的地方。子类覆盖它，喂入构造好的 (Document, score)，
         于是 retrieve() 里的分数换算 / 阈值过滤 / top_k 截断 / 出参整形
         全部是**生产代码本身**在跑，不是测试重写的一份。
    · HttpConnectionPool 的下游
      —— 起一个本地 stub，按 Dify 的真实报文格式回 chat-messages /
         workflows/run 两种响应（blocking 与 SSE 流式都回）。
         验证的是 DifyProvider 的协议翻译，不是 Dify 本身。

【覆盖的东西（每条都对应一个会真出问题的地方）】

    1. knowledge_id 白名单：未知 / 已禁用 一律 2001，且**不回落**默认租户
    2. Bearer 鉴权：头格式错 1001、密钥错 1002、密钥未配 500（不许放行）
    3. score 语义换算：reranker logits → sigmoid；无 rerank 时退化路径方向正确
       （距离越小越相关 → 分数越大越相关，搞反了阈值一设就全灭）
    4. metadata 对象不能为 null（Dify 会解析失败）
    5. top_k 钳制（Dify 能填 100，本项目召回深度只有 20）
    6. metadata_condition 全部算子
    7. DifyProvider：chat/completion/workflow × blocking/streaming；
       agent 应用只支持 streaming 的绕法；error 事件转异常
    8. system prompt 三种投递模式（prepend / input / ignore）

用法约定：只读代码、只起本地临时端口，不写任何文件、不碰数据库。
"""

import ast
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import dify_kb_server as KB      # noqa: E402
import llm_gateway as G          # noqa: E402

PASS, FAIL, SKIP = [], [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def skip(name, why):
    SKIP.append(name)
    print(f"  [SKIP] {name}  {why}")


def section(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


# ============================================================================
# 第一部分：配置与白名单（纯函数）
# ============================================================================

RAW_CFG = {
    "server": {"host": "127.0.0.1", "port": 0, "request_timeout": 10,
               "max_concurrency": 2},
    "auth": {"api_key_env": "TEST_DIFY_KB_KEY"},
    "limits": {"max_top_k": 20, "max_query_len": 50, "max_records": 20},
    "score": {"rank_base": 0.95, "rank_step": 0.05,
              "sigmoid_when_out_of_range": True},
    "knowledge_bases": [
        {"knowledge_id": "kb-test", "label": "测试库", "tenant_id": "default",
         "role": "user", "user_id": "dify-svc", "enabled": True},
        {"knowledge_id": "kb-jm", "label": "JM 库", "tenant_id": "jm",
         "role": "user", "user_id": "dify-svc", "enabled": True},
        {"knowledge_id": "kb-off", "label": "关掉的库", "tenant_id": "hr",
         "role": "admin", "user_id": "svc", "enabled": False},
    ],
}


def test_config():
    section("1. 配置解析与 knowledge_id 白名单")
    cfg = KB.BridgeConfig(RAW_CFG)
    check("3 个知识库入表", len(cfg.knowledge_bases) == 3,
          f"keys={sorted(cfg.knowledge_bases)}")
    check("登记且启用的可取到", cfg.kb("kb-test") is not None)
    check("未登记的返回 None（→ 2001，不回落默认租户）",
          cfg.kb("kb-not-exist") is None)
    check("enabled:false 的返回 None", cfg.kb("kb-off") is None)
    check("大小写敏感（kb-TEST 不等于 kb-test）", cfg.kb("kb-TEST") is None)
    check("租户/角色/用户按白名单取出",
          cfg.kb("kb-jm")["tenant_id"] == "jm"
          and cfg.kb("kb-jm")["role"] == "user"
          and cfg.kb("kb-jm")["user_id"] == "dify-svc")

    # 密钥：只认环境变量名，没配就是空
    os.environ.pop("TEST_DIFY_KB_KEY", None)
    check("密钥未配时 api_key 为空串（服务端据此拒绝，不放行）", cfg.api_key == "")
    os.environ["TEST_DIFY_KB_KEY"] = "secret-abc"
    check("密钥从环境变量读取", cfg.api_key == "secret-abc")

    # 默认值兜底：配置里缺字段不能让服务起不来
    minimal = KB.BridgeConfig({})
    check("空配置也能构造（默认 127.0.0.1:8300 / top_k≤20）",
          minimal.host == "127.0.0.1" and minimal.port == 8300
          and minimal.max_top_k == 20 and minimal.knowledge_bases == {})


# ============================================================================
# 第二部分：score 换算（最容易写错的地方）
# ============================================================================

def test_score():
    section("2. score 语义换算（本项目「越小越相关」→ Dify「越大越相关」）")
    cfg = KB.BridgeConfig(RAW_CFG)

    # ① reranker 已经给出 0~1 概率：原样用
    s = KB._to_dify_score({"_rerank_score": 0.87}, -0.016, 0, cfg)
    check("relevance_score 在 [0,1] 时原样使用", abs(s - 0.87) < 1e-9, f"score={s}")

    # ② reranker 给的是 logits（llama.cpp 不同版本行为不一）
    s = KB._to_dify_score({"_rerank_score": 8.0}, -0.016, 0, cfg)
    check("logits=8.0 经 sigmoid 压进 [0,1]", 0.99 < s <= 1.0, f"score={s:.6f}")
    s = KB._to_dify_score({"_rerank_score": -3.0}, -0.016, 0, cfg)
    check("logits=-3.0 经 sigmoid 压到低位", 0.0 < s < 0.1, f"score={s:.6f}")

    # ③ 无 rerank：hybrid 的 -fused（量级 0.016）绝对值无意义 → 排名归一化
    s0 = KB._to_dify_score({}, -0.0164, 0, cfg)
    s1 = KB._to_dify_score({}, -0.0150, 1, cfg)
    s5 = KB._to_dify_score({}, -0.0140, 5, cfg)
    check("hybrid 分数走排名归一化（rank0=0.95 / rank1=0.90 / rank5=0.70）",
          abs(s0 - 0.95) < 1e-9 and abs(s1 - 0.90) < 1e-9 and abs(s5 - 0.70) < 1e-9,
          f"{s0}/{s1}/{s5}")
    check("排名归一化单调递减（Dify 排序依赖这个）", s0 > s1 > s5)

    # ④ 无 rerank：纯 dense 的余弦距离（[0,2]，越小越相似）→ 必须反向
    s = KB._to_dify_score({}, 0.2, 3, cfg)
    check("余弦距离 0.2 → 0.90（方向不能反：距离小=相关度高）",
          abs(s - 0.90) < 1e-9, f"score={s}")

    # ⑤ 关闭 sigmoid 时的 clamp 兜底
    cfg2 = KB.BridgeConfig({**RAW_CFG, "score": {"sigmoid_when_out_of_range": False}})
    s = KB._to_dify_score({"_rerank_score": 8.0}, -0.016, 0, cfg2)
    check("关掉 sigmoid 后越界值被 clamp 到 1.0（不越界、不丢请求）",
          s == 1.0, f"score={s}")
    s = KB._to_dify_score({"_rerank_score": "not-a-number"}, -0.016, 0, cfg2)
    check("relevance_score 非数值时优雅退化到排名归一化", abs(s - 0.95) < 1e-9)

    # ⑥ 所有路径的分数都必须落在 [0,1]（Dify 会校验，越界整次检索失败）
    vals = [KB._to_dify_score({"m": 1}, x, i, cfg)
            for x in (8.0, -8.0, 0.5, -0.0164, 0.0, -2.0, 1.99)
            for i in (0, 3, 19)]
    vals += [KB._to_dify_score({"_rerank_score": x}, -0.01, 0, cfg)
             for x in (8.0, -8.0, 0.0, 1.0, 0.999)]
    check("全部路径的 score 均在 [0,1]", all(0.0 <= v <= 1.0 for v in vals),
          f"min={min(vals):.4f} max={max(vals):.4f}")


# ============================================================================
# 第三部分：metadata_condition 过滤
# ============================================================================

def test_metadata_condition():
    section("3. metadata_condition 算子")
    meta = {"file_name": "JM-S509 指令表.pdf", "section_path": "§3 定位",
            "access_level": "restricted", "page": 12, "chunk_index": 14}

    def m(name, op, value=None):
        return KB._match_metadata(meta, {
            "logical_operator": "and",
            "conditions": [{"name": name, "comparison_operator": op,
                            "value": value}]})

    check("contains 命中", m("file_name", "contains", "指令表"))
    check("contains 未命中", not m("file_name", "contains", "协议规范"))
    check("not contains", m("file_name", "not contains", "协议规范"))
    check("start with", m("file_name", "start with", "JM-"))
    check("end with", m("file_name", "end with", ".pdf"))
    check("is 精确相等", m("access_level", "is", "restricted"))
    check("is not", m("access_level", "is not", "public"))
    check("= / ≠ 符号式", m("access_level", "=", "restricted")
          and m("access_level", "≠", "public"))
    check("中文字段名兜底（document_name → source）",
          KB._match_metadata({"source": "指令表.pdf"},
                             {"conditions": [{"name": "document_name",
                                              "comparison_operator": "contains",
                                              "value": "指令表"}]}))
    check("数值比较 >", m("page", ">", 10))
    check("数值比较 ≤", m("page", "≤", 12))
    check("empty / not empty", m("not_exist", "empty")
          and m("file_name", "not empty"))
    check("未知算子按不匹配处理（不静默放行）",
          not m("file_name", "regex_match", ".*"))

    # and / or 逻辑
    cond_or = {"logical_operator": "or", "conditions": [
        {"name": "file_name", "comparison_operator": "contains", "value": "协议"},
        {"name": "access_level", "comparison_operator": "is", "value": "restricted"},
    ]}
    cond_and = dict(cond_or, logical_operator="and")
    check("or：任一满足即通过", KB._match_metadata(meta, cond_or))
    check("and：全部满足才通过", not KB._match_metadata(meta, cond_and))
    check("无条件时不过滤（返回 True）", KB._match_metadata(meta, {}))
    check("条件不是 dict 时不过滤", KB._match_metadata(meta, "garbage"))


# ============================================================================
# 第四部分：知识库桥 HTTP 端到端（stub 掉 Milvus，其余全是生产代码）
# ============================================================================

class Doc:
    def __init__(self, text, meta):
        self.page_content = text
        self.metadata = meta


# 前三条带 reranker 真实分，后三条只有 RRF 分数（验证退化路径）
PAIRS = [
    (Doc("定位方式有三种：基站定位、WiFi 定位、GPS 定位。",
         {"file_name": "指令表.pdf", "page": 12, "access_level": "public",
          "section_path": "§3 定位", "chunk_index": 14, "_rerank_score": 0.93}),
     -0.0164),
    (Doc("GPS 定位精度受遮挡影响，室内建议用 WiFi 定位兜底。",
         {"file_name": "指令表.pdf", "page": 13, "access_level": "public",
          "section_path": "§3 定位", "chunk_index": 15, "_rerank_score": 0.81}),
     -0.0160),
    (Doc("设备上报协议要求每条指令带序列号。",
         {"file_name": "协议规范.pdf", "page": 4, "access_level": "restricted",
          "section_path": "§2 协议", "chunk_index": 3, "_rerank_score": 0.52}),
     -0.0155),
    (Doc("心跳包默认 60 秒一次。",
         {"file_name": "协议规范.pdf", "page": 9, "access_level": "public",
          "chunk_index": 8}), -0.0164),
    (Doc("补传机制在断网恢复后触发。",
         {"file_name": "协议规范.pdf", "page": 10, "access_level": "public",
          "chunk_index": 9}), -0.0150),
    (Doc("设备休眠时不上报。",
         {"file_name": "运维手册.pdf", "page": 3, "access_level": "public",
          "chunk_index": 2}), -0.0140),
]


class StubService(KB.DifyRetrievalService):
    """只替换 _retrieve_pairs —— retrieve() 的其余部分是生产代码"""

    def __init__(self, cfg, pairs=None):
        super().__init__(cfg)
        self.pairs = list(PAIRS if pairs is None else pairs)
        self.seen = {}

    def _retrieve_pairs(self, kb, query):
        self.seen = {"kb": dict(kb), "query": query}
        return list(self.pairs)


def _free_port():
    """拿一个空闲端口。

    必须显式拿：BridgeConfig 里 port 用的是 `server.get("port") or 8300`，
    写 0 会回落到 8300（部署配置里"随机端口"没意义，代码里刻意这么定）。
    测试里若两个服务都想绑 8300，Windows 的 SO_REUSEADDR 不会报错而是
    静默抢占 —— 结果是请求被另一个服务接走，测试假绿。这个坑踩过。
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def http_post(url, payload, api_key=None, raw_auth=None, timeout=10):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if raw_auth is not None:
        req.add_header("Authorization", raw_auth)
    elif api_key is not None:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"raw": body}


def http_get(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, {}


def test_kb_server():
    section("4. 知识库桥 HTTP 端到端（POST /retrieval）")
    os.environ["TEST_DIFY_KB_KEY"] = "secret-abc"
    cfg = KB.BridgeConfig(RAW_CFG)
    cfg.port = _free_port()          # 见 _free_port 的说明
    svc = StubService(cfg)
    httpd = KB._build_server(cfg, svc)
    port = httpd.server_address[1]
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"
    url = base + "/retrieval"

    try:
        # ---- 鉴权 ----
        st, body = http_post(url, {"knowledge_id": "kb-test", "query": "定位"},
                             raw_auth="secret-abc")
        check("Authorization 缺 Bearer 前缀 → 1001",
              st == 403 and body.get("error_code") == 1001, f"http={st} {body}")

        st, body = http_post(url, {"knowledge_id": "kb-test", "query": "定位"},
                             raw_auth="Basic xx")
        check("Authorization 规则错（Basic）→ 1001",
              st == 403 and body.get("error_code") == 1001)

        st, body = http_post(url, {"knowledge_id": "kb-test", "query": "定位"},
                             api_key="wrong-key")
        check("密钥不匹配 → 1002", st == 403 and body.get("error_code") == 1002,
              f"http={st}")

        st, body = http_post(url, {"knowledge_id": "kb-test", "query": "定位"},
                             api_key="secret-abc")
        check("正确密钥 + 已登记库 → 200", st == 200 and "records" in body,
              f"http={st}")

        # ---- 白名单 ----
        st, body = http_post(url, {"knowledge_id": "kb-hacker", "query": "定位"},
                             api_key="secret-abc")
        check("未登记 knowledge_id → 2001（不回落默认租户）",
              st == 404 and body.get("error_code") == 2001, f"http={st}")

        st, body = http_post(url, {"knowledge_id": "kb-off", "query": "定位"},
                             api_key="secret-abc")
        check("enabled:false 的库 → 2001", st == 404
              and body.get("error_code") == 2001)

        # ★ 2026-10-23 契约变更（本用例原断言是 404 + 2001，已过期）：
        #   **缺失 / 空 knowledge_id = Dify 保存外部知识库 API 时的连通性探测**。
        #   实测 body 92 字节、knowledge_id 为空串；Dify 只看响应码，404 就直接
        #   判定「连不上」并抛 `ValueError: Not Found: failed to connect`，
        #   而 POST /retrieval 本身是好的 —— 表现为「接口能调但永远存不进去」。
        #   ⇒ 这里必须 200 + 空 records。这不是放水：空 ID 不带 tenant/role/user
        #     任何检索参数，不查任何数据、零信息泄露；
        #     真正的白名单防线由上面「未登记 kb-hacker → 2001」「enabled:false → 2001」
        #     两条守着，那两条一条都不能放宽。
        st, body = http_post(url, {"query": "定位"}, api_key="secret-abc")
        check("缺 knowledge_id → 200 空结果（Dify 探测约定，非白名单放水）",
              st == 200 and body.get("records") == [], f"http={st} {body}")

        # 关键：白名单生效后，检索用的租户/角色必须是白名单里写死的，
        # 不能是请求体带来的（请求体里塞 tenant_id 也没用）
        st, body = http_post(url, {"knowledge_id": "kb-jm", "query": "定位",
                                   "tenant_id": "__global__", "role": "admin"},
                             api_key="secret-abc")
        check("请求体里的 tenant_id/role 被忽略，以白名单为准",
              svc.seen["kb"]["tenant_id"] == "jm"
              and svc.seen["kb"]["role"] == "user",
              f"seen={svc.seen['kb']['tenant_id']}/{svc.seen['kb']['role']}")

        # ---- 出参整形 ----
        st, body = http_post(url, {"knowledge_id": "kb-test", "query": "定位方式",
                                   "retrieval_setting": {"top_k": 6,
                                                         "score_threshold": 0.0}},
                             api_key="secret-abc")
        recs = body.get("records") or []
        check("返回 6 条（全部候选）", len(recs) == 6, f"got={len(recs)}")
        check("每条都有 content/score/title/metadata 四字段",
              all(set(r) >= {"content", "score", "title", "metadata"} for r in recs))
        check("metadata 是对象而非 null（Dify 会解析失败）",
              all(isinstance(r["metadata"], dict) for r in recs))
        check("score 单调不增（Dify 依赖顺序）",
              all(recs[i]["score"] >= recs[i + 1]["score"]
                  for i in range(len(recs) - 1)),
              f"scores={[r['score'] for r in recs]}")
        check("reranker 分数被透出（第一条 0.93）",
              abs(recs[0]["score"] - 0.93) < 1e-6, f"score={recs[0]['score']}")
        check("title 拼成「文件名 · 章节」", "指令表.pdf" in recs[0]["title"],
              recs[0]["title"])
        check("metadata 带 tenant_id/knowledge_id（排障时能看出被哪个租户卡掉）",
              recs[0]["metadata"].get("tenant_id") == "default"
              and recs[0]["metadata"].get("knowledge_id") == "kb-test")

        # ---- top_k 与阈值 ----
        st, body = http_post(url, {"knowledge_id": "kb-test", "query": "x",
                                   "retrieval_setting": {"top_k": 2}},
                             api_key="secret-abc")
        check("top_k=2 → 只回 2 条", len(body.get("records") or []) == 2)

        st, body = http_post(url, {"knowledge_id": "kb-test", "query": "x",
                                   "retrieval_setting": {"top_k": 100}},
                             api_key="secret-abc")
        check("top_k=100 被钳制到 max_top_k=20（本项目召回深度就 20）",
              svc.seen.get("query") is not None, "钳制在 HTTP 层前完成")
        check("top_k=100 时返回条数不超过候选数（6）",
              len(body.get("records") or []) == 6,
              f"got={len(body.get('records') or [])}")

        st, body = http_post(url, {"knowledge_id": "kb-test", "query": "x",
                                   "retrieval_setting": {"top_k": 6,
                                                         "score_threshold": 0.85}},
                             api_key="secret-abc")
        recs = body.get("records") or []
        check("score_threshold=0.85 → 只剩 1 条（0.93）",
              len(recs) == 1 and recs[0]["score"] >= 0.85, f"got={len(recs)}")

        # ---- metadata_condition ----
        st, body = http_post(url, {
            "knowledge_id": "kb-test", "query": "x",
            "retrieval_setting": {"top_k": 6, "score_threshold": 0.0},
            "metadata_condition": {"logical_operator": "and", "conditions": [
                {"name": "file_name", "comparison_operator": "contains",
                 "value": "协议规范"}]}}, api_key="secret-abc")
        recs = body.get("records") or []
        check("metadata_condition 过滤生效（只剩协议规范 3 条）",
              len(recs) == 3 and all("协议规范" in r["title"] for r in recs),
              f"got={len(recs)}")

        st, body = http_post(url, {
            "knowledge_id": "kb-test", "query": "x",
            "retrieval_setting": {"top_k": 6},
            "metadata_condition": {"logical_operator": "and", "conditions": [
                {"name": "file_name", "comparison_operator": "contains",
                 "value": "不存在的文件"}]}}, api_key="secret-abc")
        check("过滤到空 → records:[]（不是报错）",
              st == 200 and body.get("records") == [])

        # ---- 入参校验 ----
        st, body = http_post(url, {"knowledge_id": "kb-test", "query": ""},
                             api_key="secret-abc")
        check("query 为空 → 400", st == 400, f"http={st}")

        st, body = http_post(url, {"knowledge_id": "kb-test", "query": "长" * 60},
                             api_key="secret-abc")
        check("query 超长（60 > 50）→ 400", st == 400, f"http={st}")

        st, body = http_post(url, {"knowledge_id": "kb-test", "query": "x",
                                   "retrieval_setting": "garbage"},
                             api_key="secret-abc")
        check("retrieval_setting 类型错时用默认值而不是崩",
              st == 200, f"http={st}")

        # ---- 无 rerank 的整批退化：分数应按排名归一化 0.95/0.90/0.85 ----
        svc.pairs = [
            (Doc("甲", {"file_name": "a.pdf", "chunk_index": 0}), -0.0164),
            (Doc("乙", {"file_name": "b.pdf", "chunk_index": 1}), -0.0150),
            (Doc("丙", {"file_name": "c.pdf", "chunk_index": 2}), -0.0140),
        ]
        st, body = http_post(url, {"knowledge_id": "kb-test", "query": "x",
                                   "retrieval_setting": {"top_k": 3}},
                             api_key="secret-abc")
        got = [r["score"] for r in (body.get("records") or [])]
        check("整批无 rerank 时按排名归一化（0.95/0.90/0.85）",
              len(got) == 3 and abs(got[0] - 0.95) < 1e-6
              and abs(got[1] - 0.90) < 1e-6 and abs(got[2] - 0.85) < 1e-6,
              f"{got}")
        svc.pairs = list(PAIRS)

        # ---- 检索后端抛异常 → 500 而不是空 records ----
        svc.pairs = []
        st, body = http_post(url, {"knowledge_id": "kb-test", "query": "x"},
                             api_key="secret-abc")
        check("无命中 → 200 + records:[]", st == 200
              and body.get("records") == [])
        svc.pairs = list(PAIRS)

        class Boom(StubService):
            def _retrieve_pairs(self, kb, query):
                raise RuntimeError("milvus down")

        boom = Boom(cfg)
        cfg2 = KB.BridgeConfig(RAW_CFG)
        cfg2.port = _free_port()
        # 用同一个 server 不能换 service，另起一个端口
        httpd2 = KB._build_server(cfg2, boom)
        port2 = httpd2.server_address[1]
        threading.Thread(target=httpd2.serve_forever, daemon=True).start()
        st, body = http_post(f"http://127.0.0.1:{port2}/retrieval",
                            {"knowledge_id": "kb-test", "query": "x"},
                            api_key="secret-abc")
        check("检索后端炸了 → 500 + 明确文案（不拿空 records 冒充没找到）",
              st == 500 and "retrieval backend error" in str(body.get("error_msg")),
              f"http={st}")
        httpd2.shutdown()

        # ---- /health ----
        st, body = http_get(base + "/health")
        check("GET /health 200 且列出知识库",
              st == 200 and len(body.get("knowledge_bases") or []) == 3)
        check("/health 里能看到累计请求数（排障用）",
              (body.get("stats") or {}).get("requests", 0) > 0)

        st, body = http_get(base + "/nope")
        check("未知 GET 路径 → 404", st == 404)

        st, body = http_post(base + "/wrong-path",
                             {"knowledge_id": "kb-test", "query": "x"},
                             api_key="secret-abc")
        check("未知 POST 路径 → 404", st == 404, f"http={st}")

        # ---- 密钥未配时不许放行 ----
        os.environ["TEST_DIFY_KB_KEY"] = ""
        st, body = http_post(url, {"knowledge_id": "kb-test", "query": "x"},
                             api_key="secret-abc")
        check("服务端密钥未配 → 500 拒服务（绝不裸奔放行）",
              st == 500 and body.get("error_code") == 1002, f"http={st}")
        os.environ["TEST_DIFY_KB_KEY"] = "secret-abc"

    finally:
        httpd.shutdown()
        httpd.server_close()
        th.join(timeout=3)


# ============================================================================
# 第五部分：DifyProvider（stub 一个 Dify，验协议翻译）
# ============================================================================

class StubDifyHandler(BaseHTTPRequestHandler):
    """按 Dify 真实报文格式回包的最小 stub。

    故意用 HTTP/1.0 + 不带 Content-Length 的方式回流式响应：
    客户端靠「连接关闭」判定结束，这正好覆盖真实 Dify SSE 的行为。
    """
    protocol_version = "HTTP/1.0"
    seen = {}

    def log_message(self, *a):
        pass

    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, events):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for ev in events:
            self.wfile.write(
                ("data: " + json.dumps(ev, ensure_ascii=False) + "\n\n").encode("utf-8"))
        self.wfile.flush()

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            body = {}
        path = urlparse(self.path).path
        StubDifyHandler.seen = {"path": path, "body": body,
                                "auth": self.headers.get("Authorization") or ""}

        inputs = body.get("inputs") or {}
        query = str(body.get("query") or inputs.get("query") or "")
        streaming = body.get("response_mode") == "streaming"

        # 错误路径：query 里带「触发错误」就回错误
        if "触发错误" in query:
            if streaming:
                self._sse([{"event": "error", "message": "stub boom"}])
            else:
                self._json(400, {"code": "invalid_param", "message": "stub boom"})
            return

        if path.endswith("/chat-messages"):
            if streaming:
                self._sse([
                    {"event": "message", "answer": "HALF-"},
                    {"event": "message", "answer": "TWO"},
                    {"event": "message_end", "metadata": {"usage": {
                        "prompt_tokens": 7, "completion_tokens": 3}}},
                ])
            else:
                self._json(200, {"event": "message", "answer": "STUB ANSWER",
                                 "conversation_id": "c1", "mode": "chat",
                                 "metadata": {"usage": {
                                     "prompt_tokens": 11, "completion_tokens": 4,
                                     "total_tokens": 15}}})
            return

        if path.endswith("/completion-messages"):
            self._json(200, {"event": "message", "answer": "COMPLETION ANSWER",
                             "metadata": {"usage": {"prompt_tokens": 5,
                                                    "completion_tokens": 2}}})
            return

        if path.endswith("/workflows/run"):
            if streaming:
                self._sse([
                    {"event": "workflow_started"},
                    {"event": "text_chunk", "data": {"text": "WF-"}},
                    {"event": "workflow_finished", "data": {
                        "status": "succeeded", "outputs": {"text": "WF-DONE"},
                        "total_tokens": 9}},
                ])
            else:
                self._json(200, {"data": {
                    "status": "succeeded", "total_tokens": 42,
                    "outputs": {"text": "WF BLOCK", "node_count": 3}}})
            return

        self._json(404, {"error": "unknown path"})


def _mk_model_cfg(port, **over):
    kw = dict(name="dify-test", provider="dify", model="dify-test",
              base_url=f"http://127.0.0.1:{port}/v1",
              api_key_env="TEST_DIFY_APP_KEY", timeout=10.0,
              tier="large")
    kw.update(over)
    return G.ModelConfig(**kw)


def test_dify_provider():
    section("5. DifyProvider（网关把 Dify 应用当成一个模型）")
    os.environ["TEST_DIFY_APP_KEY"] = "app-test-key"

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), StubDifyHandler)
    httpd.daemon_threads = True
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    pool = G.HttpConnectionPool(max_per_host=4, idle_timeout=30)
    try:
        check("provider=dify 能在注册表里找到",
              G._PROVIDER_REGISTRY.get("dify") is G.DifyProvider)
        check("build_provider('dify') 走 DifyProvider",
              isinstance(G.build_provider(_mk_model_cfg(port), pool), G.DifyProvider))

        # ---- chat / blocking ----
        p = G.build_provider(_mk_model_cfg(port), pool)
        r = p.invoke("你是助手。", "定位方式有哪些？")
        check("chat blocking：正文取 answer 字段", r.text == "STUB ANSWER", r.text)
        check("chat blocking：token 取 metadata.usage（真实值非估算）",
              r.prompt_tokens == 11 and r.completion_tokens == 4,
              f"{r.prompt_tokens}/{r.completion_tokens}")
        check("chat blocking：provider 标为 dify", r.provider == "dify")
        check("请求打到 /v1/chat-messages（路径由 provider 拼）",
              StubDifyHandler.seen["path"] == "/v1/chat-messages",
              StubDifyHandler.seen["path"])
        check("自动带 Bearer 鉴权头",
              StubDifyHandler.seen["auth"] == "Bearer app-test-key")
        check("system prompt 默认 prepend 进 query",
              "你是助手。" in StubDifyHandler.seen["body"].get("query", ""))
        check("带上 Dify 必填的 user 标识",
              StubDifyHandler.seen["body"].get("user") == "llm-gateway",
              str(StubDifyHandler.seen["body"].get("user")))

        # ---- chat / streaming ----
        pieces = list(p.stream("你是助手。", "定位方式有哪些？"))
        check("chat streaming：逐段 yield 增量",
              "".join(pieces) == "HALF-TWO", repr(pieces))
        check("streaming 时 response_mode=streaming",
              StubDifyHandler.seen["body"].get("response_mode") == "streaming")

        # ---- system prompt 三种模式 ----
        p_ignore = G.build_provider(
            _mk_model_cfg(port, dify_sys_prompt_mode="ignore"), pool)
        p_ignore.invoke("你是助手。", "原始问题")
        check("sys_prompt_mode=ignore：query 里没有 system prompt",
              StubDifyHandler.seen["body"].get("query") == "原始问题",
              StubDifyHandler.seen["body"].get("query"))

        p_input = G.build_provider(
            _mk_model_cfg(port, dify_sys_prompt_mode="input",
                          dify_sys_input="sys_prompt"), pool)
        p_input.invoke("你是助手。", "原始问题")
        b = StubDifyHandler.seen["body"]
        check("sys_prompt_mode=input：写进指定变量，query 保持干净",
              b.get("inputs", {}).get("sys_prompt") == "你是助手。"
              and b.get("query") == "原始问题", str(b.get("inputs")))

        # ---- 应用自定义输入变量 ----
        p_inputs = G.build_provider(
            _mk_model_cfg(port, dify_app_inputs={"tenant": "jm"}), pool)
        p_inputs.invoke("s", "q")
        check("dify_app_inputs 原样透传（键名必须与 Dify 一致）",
              StubDifyHandler.seen["body"].get("inputs", {}).get("tenant") == "jm")

        # ---- agent 应用：只支持 streaming，invoke 要自己绕 ----
        p_agent = G.build_provider(_mk_model_cfg(port, dify_app_type="agent"), pool)
        r = p_agent.invoke("s", "q")
        check("agent 应用：invoke 自动走流式再拼成非流式",
              r.text == "HALF-TWO", r.text)
        check("agent 应用请求被标成 streaming（blocking 会被 Dify 400）",
              StubDifyHandler.seen["body"].get("response_mode") == "streaming")

        # ---- completion 应用 ----
        p_comp = G.build_provider(_mk_model_cfg(port, dify_app_type="completion"), pool)
        r = p_comp.invoke("s", "q")
        check("completion 应用：打到 /v1/completion-messages",
              StubDifyHandler.seen["path"] == "/v1/completion-messages",
              StubDifyHandler.seen["path"])
        check("completion 应用：query 放进 inputs.query",
              StubDifyHandler.seen["body"].get("inputs", {}).get("query"))
        check("completion 应用：正文取 answer", r.text == "COMPLETION ANSWER")

        # ---- workflow 应用 ----
        p_wf = G.build_provider(_mk_model_cfg(port, dify_app_type="workflow"), pool)
        r = p_wf.invoke("s", "q")
        check("workflow blocking：打到 /v1/workflows/run",
              StubDifyHandler.seen["path"] == "/v1/workflows/run")
        check("workflow blocking：正文从 data.outputs 拼出（非字符串值被跳过）",
              r.text == "WF BLOCK", r.text)
        check("workflow blocking：token 取 data.total_tokens",
              r.completion_tokens == 42, str(r.completion_tokens))
        check("workflow blocking：finish_reason = status",
              r.finish_reason == "succeeded")

        p_wf_key = G.build_provider(
            _mk_model_cfg(port, dify_app_type="workflow", dify_output_key="text"),
            pool)
        check("workflow 指定 dify_output_key 时只取那一个变量",
              p_wf_key.invoke("s", "q").text == "WF BLOCK")

        p_wf_empty = G.build_provider(
            _mk_model_cfg(port, dify_app_type="workflow",
                          dify_output_key="no_such_key"), pool)
        try:
            p_wf_empty.invoke("s", "q")
            ok, detail = True, "取不到 key 时退化为拼接全部字符串"
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        check("workflow 指定的 key 不存在时退化为拼接（不炸）", ok, detail)

        pieces = list(p_wf.stream("s", "q"))
        check("workflow streaming：text_chunk 逐段出",
              "".join(pieces).startswith("WF-"), repr(pieces))
        check("workflow streaming：结尾补 outputs 里的正文（应用最后一刻才出结果）",
              "WF-DONE" in "".join(pieces), repr("".join(pieces)))

        # ---- 错误处理 ----
        try:
            p.invoke("s", "触发错误")
            check("blocking 非 200 → 抛 ProviderError", False, "没抛异常")
        except G.ProviderError as e:
            check("blocking 非 200 → 抛 ProviderError", True, str(e)[:60])
        except Exception as e:
            check("blocking 非 200 → 抛 ProviderError", False,
                  f"抛了 {type(e).__name__}")

        try:
            list(p.stream("s", "触发错误"))
            check("流式 error 事件 → 抛 ProviderError", False, "没抛异常")
        except G.ProviderError as e:
            check("流式 error 事件 → 抛 ProviderError", True, str(e)[:60])
        except Exception as e:
            check("流式 error 事件 → 抛 ProviderError", False,
                  f"抛了 {type(e).__name__}")

    finally:
        httpd.shutdown()
        httpd.server_close()
        pool.close()


# ============================================================================
# 第六部分：与生产代码的接缝（不 import 重依赖，用 AST 校验）
# ============================================================================

def test_seams():
    section("6. 与生产代码的接缝（AST 校验，不拉起 langgraph 依赖）")
    src_path = os.path.join(ROOT, "langgraph_rag_agent.py")
    with open(src_path, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)

    methods = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "LangGraphRAGApp":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef):
                    methods[fn.name] = fn
    check("找到 LangGraphRAGApp._do_retrieve", "_do_retrieve" in methods)
    check("找到 LangGraphRAGApp._rerank", "_rerank" in methods)

    def argnames(fn):
        return [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]

    if "_do_retrieve" in methods:
        args = argnames(methods["_do_retrieve"])
        check("_do_retrieve 有 keep_score 参数（桥靠它拿 reranker 分）",
              "keep_score" in args, str(args))
        check("keep_score 默认 False（线上主链路零回归）",
              _default_is_false(methods["_do_retrieve"], "keep_score"))
        # 它必须把 keep_score 继续传给 _rerank，否则开关形同虚设
        passed = False
        for node in ast.walk(methods["_do_retrieve"]):
            if isinstance(node, ast.Call):
                fn = node.func
                name = getattr(fn, "attr", None) or getattr(fn, "id", None)
                if name == "_rerank":
                    if any(getattr(k, "arg", None) == "keep_score"
                           for k in node.keywords):
                        passed = True
        check("_do_retrieve 把 keep_score 转传给 _rerank", passed)

    if "_rerank" in methods:
        args = argnames(methods["_rerank"])
        check("_rerank 有 keep_score 参数", "keep_score" in args, str(args))
        check("_rerank 写入 metadata['_rerank_score']",
              "_rerank_score" in src)

    # bridge 里真的按协议取这两个键
    kb_src = open(os.path.join(ROOT, "dify_kb_server.py"), encoding="utf-8").read()
    check("桥按 Dify 协议返回 records/content/score/title/metadata",
          all(k in kb_src for k in
              ('"records"', '"content"', '"score"', '"title"', '"metadata"')))
    check("桥实现了三个官方错误码 1001/1002/2001",
          all(str(c) in kb_src for c in (1001, 1002, 2001)))
    check("桥用 hmac.compare_digest 做定长比较（防时序侧信道）",
          "hmac.compare_digest" in kb_src)
    check("桥屏蔽业务 print（redirect_stdout），日志不被打花",
          "redirect_stdout" in kb_src)


def _default_is_false(fn, name):
    """判断某参数默认值是不是 False（AST 层面，不执行代码）"""
    pos = [a.arg for a in fn.args.args]
    offs = len(pos) - len(fn.args.defaults)
    for i, d in enumerate(fn.args.defaults):
        if pos[offs + i] == name:
            return isinstance(d, ast.Constant) and d.value is False
    for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
        if a.arg == name:
            return isinstance(d, ast.Constant) and d.value is False
    return False


# ============================================================================
# 第七部分：真实 _rerank 链路（只 stub reranker 的 HTTP 端点）
# ============================================================================

class StubReranker(BaseHTTPRequestHandler):
    """按 llama.cpp /v1/rerank 协议回包。

    故意回 **logits**（不是 0~1 概率）——线上不同版本的 llama.cpp 行为不一致，
    这里按下限（最难的一种）来验，能过就两种都能过。
    """
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        docs = body.get("documents") or []
        # 打分规则：文档里出现「高」→ 高分 logits，其余按位置递减
        scores = []
        for i, d in enumerate(docs):
            if "高" in d:
                scores.append(5.2)
            elif "中" in d:
                scores.append(0.4)
            else:
                scores.append(-3.1)
        payload = {"results": [{"index": i, "relevance_score": s}
                               for i, s in enumerate(scores)]}
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def test_real_rerank():
    section("7. 真实 _rerank + _BridgeHost（只 stub reranker 的 HTTP 端点）")
    try:
        import inspect
        import langgraph_rag_agent as L
    except Exception as e:
        skip("真实 _rerank 链路", f"langgraph_rag_agent 导入失败: {type(e).__name__}: {e}")
        return

    # 调用约定校验：_retrieve_pairs 用的是 _do_retrieve(host, [q], role, True)
    try:
        inspect.signature(L.LangGraphRAGApp._do_retrieve).bind(
            None, ["q"], "user", True)
        check("_do_retrieve 支持 (self, queries, role, keep_score) 位置调用",
              True)
    except TypeError as e:
        check("_do_retrieve 支持 (self, queries, role, keep_score) 位置调用",
              False, str(e))

    # _BridgeHost 的 __getattr__ 绑定（检索链路靠它拿到兄弟方法）
    host = KB._BridgeHost(None, "u1", "jm")
    bound = host._rrf_fuse_queries
    check("_BridgeHost 能把 LangGraphRAGApp 的兄弟方法绑到自身上",
          callable(bound) and getattr(bound, "__self__", None) is host)

    # 真跑 _rerank（只把 HTTP 端点换成 stub）
    srv = ThreadingHTTPServer(("127.0.0.1", 0), StubReranker)
    srv.daemon_threads = True
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    old_url, old_enabled = L.RERANK_URL, L.RERANK_ENABLED
    try:
        L.RERANK_URL = f"http://127.0.0.1:{port}/v1/rerank"
        L.RERANK_ENABLED = True
        docs = [
            (Doc("低相关内容（甲）", {"file_name": "a.pdf"}), -0.0164),
            (Doc("高相关内容（乙）", {"file_name": "b.pdf"}), -0.0150),
            (Doc("中相关内容（丙）", {"file_name": "c.pdf"}), -0.0140),
        ]

        out = L.LangGraphRAGApp._rerank(None, "测试问题", list(docs), 10,
                                        keep_score=True)
        names = [d.metadata.get("file_name") for d, _ in out]
        check("keep_score=True：按 relevance_score 降序重排",
              names == ["b.pdf", "c.pdf", "a.pdf"], str(names))
        got = [d.metadata.get("_rerank_score") for d, _ in out]
        check("keep_score=True：原始 logits 被写进 metadata['_rerank_score']",
              got == [5.2, 0.4, -3.1], str(got))
        # 关键：RRF 距离本身不能被改动（下游 MMR / 预算都依赖它）
        check("原有 (doc, 距离) 里的距离不被改写",
              [s for _, s in out] == [-0.0150, -0.0140, -0.0164],
              str([s for _, s in out]))

        docs2 = [
            (Doc("低相关内容（甲）", {"file_name": "a.pdf"}), -0.0164),
            (Doc("高相关内容（乙）", {"file_name": "b.pdf"}), -0.0150),
        ]
        out2 = L.LangGraphRAGApp._rerank(None, "测试问题", docs2, 10)
        check("keep_score 默认 False：metadata 里不出现 _rerank_score（线上零回归）",
              all("_rerank_score" not in d.metadata for d, _ in out2))
        check("keep_score 默认 False：排序结果与 True 完全一致",
              [d.metadata["file_name"] for d, _ in out2] == ["b.pdf", "a.pdf"])

        # 端到端：logits 经桥换算后必须落在 [0,1] 且顺序仍是 b > c > a
        cfg = KB.BridgeConfig(RAW_CFG)
        scores = [KB._to_dify_score(d.metadata, s, i, cfg)
                  for i, (d, s) in enumerate(out)]
        check("真实 logits → Dify score 落在 [0,1]",
              all(0.0 <= s <= 1.0 for s in scores),
              str([round(s, 4) for s in scores]))
        check("换算后顺序不变（高相关仍第一）",
              scores[0] > scores[1] > scores[2],
              str([round(s, 4) for s in scores]))
    finally:
        L.RERANK_URL, L.RERANK_ENABLED = old_url, old_enabled
        srv.shutdown()
        srv.server_close()


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 74)
    print("Dify 双向桥自测（出口：/retrieval 外部知识库 API；入口：DifyProvider）")
    print("=" * 74)

    test_config()
    test_score()
    test_metadata_condition()
    test_kb_server()
    test_dify_provider()
    test_seams()
    test_real_rerank()

    total = len(PASS) + len(FAIL)
    print(f"\n{'=' * 74}")
    print(f"结果：{len(PASS)}/{total} 通过，{len(FAIL)} 失败，{len(SKIP)} 跳过，"
          f"耗时 {time.time() - t0:.2f}s")
    if FAIL:
        print("\n失败项：")
        for name in FAIL:
            print(f"  - {name}")
    print("=" * 74)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
