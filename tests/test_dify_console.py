# -*- coding: utf-8 -*-
"""Dify 接入控制台（web）测试 —— 页面路由 / API 契约 / 桥服务端到端。

为什么单独一个文件：
  `tests/test_dify_bridge.py` 测的是**桥服务进程**（HTTP server、白名单、score 换算），
  不涉及 Flask。控制台是后加的一层，它自己会踩三类坑，都必须被钉住：
    1) 鉴权：这是运维面板，请求流水带原始 query、演练能读业务正文 → 必须仅管理员；
    2) 契约：页面 JS 认的字段（deps[].ok / bridge.stats / latency_ms / limits / recent）
       一旦后端改名，页面会静默变成一堆「—」，不会报错；
    3) 代理：/api/dify/retrieval 必须由服务端转发（浏览器不直连 8300、不暴露密钥）。

跑法：
    python tests/test_dify_console.py            # 快速（不触发真实检索）
    python tests/test_dify_console.py --live     # 追加一次真实检索（十几秒）

依赖：桥服务需在 127.0.0.1:8300 起着（`python dify_kb_server.py`）。
      没起时相关用例标记 SKIP，不判 FAIL —— 但会明确打出来，别当全绿。
"""
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import rag_web_server as W          # noqa: E402
import prompt_manager as PM         # noqa: E402

PASS = FAIL = SKIP = 0
FAILS = []


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  [PASS] %s%s" % (name, ("  " + str(extra)) if extra else ""))
    else:
        FAIL += 1
        FAILS.append(name)
        print("  [FAIL] %s%s" % (name, ("  " + str(extra)) if extra else ""))


def skip(name, why=""):
    global SKIP
    SKIP += 1
    print("  [SKIP] %s%s" % (name, ("  " + str(why)) if why else ""))


def sect(t):
    print("\n" + "=" * 74 + "\n" + t + "\n" + "=" * 74)


# --------------------------------------------------------------------------
# 鉴权桩：只替换 token 校验，路由/渲染/探活全走真实代码
# --------------------------------------------------------------------------
class StubAuth:
    def verify_token(self, tok):
        if tok == "tok-admin":
            return {"username": "tester", "role": "admin", "tenant_id": "default"}
        if tok == "tok-user":
            return {"username": "u1", "role": "user", "tenant_id": "default"}
        return None


PM.get_auth_manager = lambda: StubAuth()
ADMIN = {"Authorization": "Bearer tok-admin"}
USER = {"Authorization": "Bearer tok-user"}


def jbody(resp):
    try:
        return json.loads(resp.data.decode("utf-8"))
    except Exception:
        return {}


def main():
    live = "--live" in sys.argv
    c = W.app.test_client()

    # ---------------------------------------------------------------- 1 鉴权
    sect("1. 鉴权：运维面板必须仅管理员可见")
    r = c.get("/dify")
    check("未登录访问 /dify → 401", r.status_code == 401, "http=%s" % r.status_code)
    r = c.get("/api/dify/overview")
    check("未登录访问 /api/dify/overview → 401", r.status_code == 401,
          "http=%s" % r.status_code)
    r = c.get("/dify", headers=USER)
    check("普通用户访问 /dify → 403（面板带原始 query 与业务正文）",
          r.status_code == 403, "http=%s" % r.status_code)
    r = c.post("/api/dify/retrieval", headers=USER,
               json={"knowledge_id": "kb-jm", "query": "x"})
    check("普通用户调检索演练 → 403", r.status_code == 403,
          "http=%s" % r.status_code)

    # ---------------------------------------------------------------- 2 页面
    sect("2. 页面：渲染完整、占位符替换干净、不泄露密钥")
    r = c.get("/dify", headers=ADMIN)
    html = r.data.decode("utf-8")
    check("管理员访问 /dify → 200", r.status_code == 200, "http=%s" % r.status_code)
    check("Content-Type 是 html", "text/html" in (r.headers.get("Content-Type") or ""))
    for mark, label in [("Dify 接入控制台", "标题"),
                        ('data-p="ov"', "页签-概览"),
                        ('data-p="kb"', "页签-知识库"),
                        ('data-p="pg"', "页签-检索演练"),
                        ('data-p="lg"', "页签-请求流水"),
                        ('data-p="gd"', "页签-接入指引"),
                        ('id="pgRes"', "演练结果容器"),
                        ('id="deps"', "依赖矩阵容器")]:
        check("页面含 %s" % label, mark in html)
    left = [k for k in ("__ENDPOINTS__", "__BRIDGE__", "__USER__", "__ROLE__")
            if k in html]
    check("占位符全部替换（无残留）", not left, "残留=%s" % left)
    check("登录用户名已注入页面", "tester" in html)
    endpoints = json.loads(
        html.split("const ENDPOINTS = ")[1].split(";\n")[0]) if "const ENDPOINTS = " in html else []
    check("ENDPOINTS 至少 1 条且是 http 地址",
          len(endpoints) >= 1 and all(x.startswith("http://") for x in endpoints),
          endpoints)
    key = W.DIFY_KEY
    if key:
        check("页面不含 DIFY_KB_API_KEY 明文", key not in html)
    else:
        skip("页面不含密钥明文", "DIFY_KB_API_KEY 未配置")
    check("页面无 emoji/特殊符号硬伤", "✅" not in html and "❌" not in html)

    # ---------------------------------------------------------------- 3 概览
    sect("3. /api/dify/overview：字段契约（页面 JS 认这些名字）")
    t0 = time.time()
    r = c.get("/api/dify/overview", headers=ADMIN)
    d = jbody(r)
    ms = (time.time() - t0) * 1000
    check("HTTP 200", r.status_code == 200, "http=%s" % r.status_code)
    check("首屏耗时可接受（<8000ms）", ms < 8000, "%.0fms" % ms)
    check("有 ts / clock", isinstance(d.get("ts"), (int, float)) and bool(d.get("clock")),
          d.get("clock"))
    deps = d.get("deps") or []
    check("deps 数量 = 7（桥/Milvus/Ollama/精排/MySQL/Redis/Dify）",
          len(deps) == 7, len(deps))
    check("每条 dep 有 name/ok/target",
          all(set(("name", "ok", "target")) <= set(x) for x in deps))
    names = [x["name"] for x in deps]
    check("七项都在：%s" % "、".join(names), len(names) == 7)
    okn = sum(1 for x in deps if x.get("ok"))
    print("       可达 %d / %d：%s" % (
        okn, len(deps),
        "、".join("%s%s" % (x["name"], "" if x.get("ok") else "(不可达)") for x in deps)))

    br = d.get("bridge")
    if not br:
        skip("桥服务字段契约", "桥服务不可达：%s" % d.get("bridge_error"))
    else:
        st = br.get("stats") or {}
        check("bridge.stats 有 requests/records/errors/uptime_sec",
              all(k in st for k in ("requests", "records", "errors", "uptime_sec")))
        lat = br.get("latency_ms") or {}
        check("bridge.latency_ms 有 n/p50/p95",
              all(k in lat for k in ("n", "p50", "p95")), "n=%s" % lat.get("n"))
        lm = br.get("limits") or {}
        check("bridge.limits 有 max_top_k/candidate_k/max_concurrency",
              all(k in lm for k in ("max_top_k", "candidate_k", "max_concurrency")))
        rr = br.get("rerank") or {}
        check("bridge.rerank 有 enabled/url", "enabled" in rr and "url" in rr,
              "enabled=%s" % rr.get("enabled"))
        check("bridge.recent 是数组", isinstance(br.get("recent"), list),
              len(br.get("recent") or []))
        kbs = br.get("knowledge_bases") or []
        check("knowledge_bases 有 kb-jm 且 enabled",
              any(k.get("knowledge_id") == "kb-jm" and k.get("enabled") for k in kbs))
        check("kb-default 已停用（库里 0 篇，不该再对外暴露）",
              any(k.get("knowledge_id") == "kb-default" and not k.get("enabled")
                  for k in kbs))
        check("vector_db_ready 是布尔", isinstance(br.get("vector_db_ready"), bool),
              br.get("vector_db_ready"))

    # ------------------------------------------------------------- 4 chunk 数
    sect("4. /api/dify/kb_counts：Milvus 实际条数")
    t0 = time.time()
    r = c.get("/api/dify/kb_counts", headers=ADMIN)
    kc = jbody(r)
    ms = (time.time() - t0) * 1000
    check("HTTP 200", r.status_code == 200, "http=%s" % r.status_code)
    counts = kc.get("counts") or {}
    if counts:
        check("counts 是 {租户: 整数}", all(isinstance(v, int) for v in counts.values()),
              counts)
        check("jm / yh 都有数（>0）",
              (counts.get("jm") or 0) > 0 and (counts.get("yh") or 0) > 0, counts)
        check("default = 0（与 Milvus 实测一致，页面据此提示「查不到不是坏了」）",
              counts.get("default") == 0, counts.get("default"))
        print("       实测 chunk 数：%s（%0.fms，含 collection.load）"
              % (counts, ms))
    else:
        skip("chunk 数契约", kc.get("error") or "Milvus 未取到")

    # ------------------------------------------------------------- 5 参数校验
    sect("5. /api/dify/retrieval：参数校验（不触发真实检索）")
    r = c.post("/api/dify/retrieval", headers=ADMIN, json={"query": "x"})
    check("缺 knowledge_id → 400", r.status_code == 400, "http=%s" % r.status_code)
    r = c.post("/api/dify/retrieval", headers=ADMIN,
               json={"knowledge_id": "kb-jm", "query": ""})
    check("query 为空 → 400", r.status_code == 400, "http=%s" % r.status_code)
    r = c.post("/api/dify/retrieval", headers=ADMIN,
               json={"knowledge_id": "kb-jm", "query": "x", "top_k": "abc"})
    check("top_k 非数字 → 400", r.status_code == 400, "http=%s" % r.status_code)

    # ------------------------------------------------------ 5.5 审批演练
    sect("5.5 /api/dify/ai_pack：审批演练（取数 → 规则 → 路径）")
    r = c.post("/api/dify/ai_pack", headers=ADMIN, json={"knowledge_id": "kb-jm"})
    check("缺 po_id → 400", r.status_code == 400, "http=%s" % r.status_code)
    r = c.post("/api/dify/ai_pack", headers=ADMIN,
               json={"knowledge_id": "kb-jm", "po_id": "abc"})
    check("po_id 非数字 → 400", r.status_code == 400, "http=%s" % r.status_code)
    r = c.post("/api/dify/ai_pack", headers=ADMIN, json={"po_id": 33})
    check("缺 knowledge_id → 400", r.status_code == 400, "http=%s" % r.status_code)

    d = jbody(c.post("/api/dify/ai_pack", headers=ADMIN,
                     json={"knowledge_id": "kb-jm", "po_id": 33}))
    pk = d.get("response") or {}
    if d.get("http") == 200 and pk.get("ok"):
        g = pk.get("gate") or {}
        rt = pk.get("route") or {}
        nodes = rt.get("nodes") or []
        check("po=33 取数成功（demo 小额单）", pk.get("ok") is True)
        check("小额单 auto_eligible=true（仅提示，真判由 Gate.decide）",
              g.get("auto_eligible") is True, g.get("auto_eligible"))
        check("审批路径 ≥2 站", len(nodes) >= 2, len(nodes))
        check("每站都有 material（Dify 迭代节点直接用）",
              all(n.get("material") for n in nodes), len(nodes))
        check("metrics 有金额且 = 4940", float((pk.get("metrics") or {}).get("amount") or 0) == 4940.0,
              (pk.get("metrics") or {}).get("amount"))
        d2 = jbody(c.post("/api/dify/ai_pack", headers=ADMIN,
                          json={"knowledge_id": "kb-jm", "po_id": 34}))
        pk2 = d2.get("response") or {}
        check("大额单 auto_eligible=false（转人工）",
              (pk2.get("gate") or {}).get("auto_eligible") is False,
              (pk2.get("gate") or {}).get("auto_eligible"))
        check("大额单路径站数 > 小额单",
              len((pk2.get("route") or {}).get("nodes") or []) >
              len(nodes))
        # 跨租户：yh 的单用 kb-jm 查，必须拿不到
        d3 = jbody(c.post("/api/dify/ai_pack", headers=ADMIN,
                          json={"knowledge_id": "kb-jm", "po_id": 35}))
        check("跨租户取数被拒（yh 的单 #35 用 kb-jm 查 → ok=false）",
              (d3.get("response") or {}).get("ok") is False,
              (d3.get("response") or {}).get("error"))
    else:
        skip("审批演练数据契约", "演示数据缺失或 /ai/pack 不可用：%s"
             % (pk.get("error") or d.get("error") or d.get("http")))


    # ------------------------------------------------------------- 6 桥服务契约
    sect("6. 桥服务 HTTP 契约（直连 8300，不经 Flask）")
    try:
        st, body, _ = W._hc(W.DIFY_BRIDGE + "/health", timeout=3)
        h = json.loads(body)
        check("GET /health → 200 且 status=ok", st == 200 and h.get("status") == "ok")
        for k in ("rerank", "limits", "latency_ms", "recent", "knowledge_bases"):
            check("/health 含 %s" % k, k in h)
        st, body, _ = W._hc(W.DIFY_BRIDGE + "/retrieval", timeout=3,
                            headers={"Authorization": "Bearer " + W.DIFY_KEY})
        check("GET /retrieval（Dify 保存时的探测）→ 200", st == 200, "http=%s" % st)
        st, body, _ = W._hc(W.DIFY_BRIDGE + "/retrieval", method="POST", timeout=5,
                            payload={"knowledge_id": "", "query": "probe"},
                            headers={"Authorization": "Bearer " + W.DIFY_KEY})
        check("POST 空 knowledge_id → 200 + 空 records（探测约定）",
              st == 200 and json.loads(body).get("records") == [], "http=%s" % st)
        st, body, _ = W._hc(W.DIFY_BRIDGE + "/retrieval", method="POST", timeout=5,
                            payload={"knowledge_id": "kb-nope", "query": "x"},
                            headers={"Authorization": "Bearer " + W.DIFY_KEY})
        check("POST 未登记 id → 404 + 2001（白名单不留口子）",
              st == 404 and json.loads(body).get("error_code") == 2001,
              "http=%s" % st)
        st, body, _ = W._hc(W.DIFY_BRIDGE + "/retrieval", method="POST", timeout=5,
                            payload={"knowledge_id": "kb-jm", "query": "x"})
        check("POST 无鉴权 → 403 + 1001",
              st == 403 and json.loads(body).get("error_code") == 1001,
              "http=%s" % st)
    except Exception as e:
        skip("桥服务契约", "%s: %s" % (type(e).__name__, e))

    # ------------------------------------------------------------- 7 真实检索
    sect("7. 真实检索（--live 才跑；走完整 桥→Milvus→精排 链路）")
    if not live:
        skip("真实检索", "未加 --live")
    else:
        t0 = time.time()
        r = c.post("/api/dify/retrieval", headers=ADMIN,
                   json={"knowledge_id": "kb-jm", "query": "系统支持哪些通信协议",
                         "top_k": 3, "score_threshold": 0})
        d = jbody(r)
        ms = (time.time() - t0) * 1000
        check("HTTP 200（代理层）", r.status_code == 200, "http=%s" % r.status_code)
        check("桥服务 http=200", d.get("http") == 200, d.get("http"))
        recs = ((d.get("response") or {}).get("records")) or []
        check("命中 > 0", len(recs) > 0, "%d 条 / %.0fms" % (len(recs), ms))
        if recs:
            check("每条有 content/score/title/metadata",
                  all(set(("content", "score", "title", "metadata")) <= set(x)
                      for x in recs))
            sc = [x["score"] for x in recs]
            check("score 全在 [0,1]", all(0 <= s <= 1 for s in sc), sc)
            check("score 单调不增（Dify 引用列表按分数降序显示）",
                  all(sc[i] >= sc[i + 1] for i in range(len(sc) - 1)), sc)
            print("       前 3 条：%s" % [
                (round(x["score"], 4), (x["content"] or "")[:26].replace("\n", " "))
                for x in recs[:3]])
        # top_k 越界钳制：传 999 应被压到 20，且不报错
        #   注意取值路径：top_k 是塞进 retrieval_setting 的（Dify 协议如此），
        #   不在 body 顶层 —— 断言写 request["top_k"] 会永远 None。
        d2 = jbody(c.post("/api/dify/retrieval", headers=ADMIN,
                          json={"knowledge_id": "kb-jm", "query": "通信协议",
                                "top_k": 999}))
        sent = (((d2.get("request") or {}).get("retrieval_setting")) or {}).get("top_k")
        check("top_k=999 被钳到 20（不报错）",
              d2.get("http") == 200 and sent == 20, sent)

    # ------------------------------------------------------------- 汇总
    print("\n" + "=" * 74)
    print("结果：%d 通过，%d 失败，%d 跳过，耗时 %.2fs"
          % (PASS, FAIL, SKIP, time.time() - _T0))
    print("=" * 74)
    if FAILS:
        print("失败项：")
        for f in FAILS:
            print("  - %s" % f)
    return 0 if FAIL == 0 else 2


_T0 = time.time()
if __name__ == "__main__":
    sys.exit(main())
