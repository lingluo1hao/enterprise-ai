# -*- coding: utf-8 -*-
"""Dify 接入控制台 · 视觉质检（真实数据 + headless Chrome 截图）

为什么需要它：`/dify` 要管理员登录，截图工具没法自己登；而空页面截图看不出
「表头错位、正文尾部空段、KPI 小数太吵、概览下半截空一片」这类版面问题。
做法：用 Flask `test_client` + 桩掉 token 校验，拿**当前真实**的 HTML 与 API JSON，
把 `fetch` 换成回放，再交给 Chrome `--headless=new --screenshot`。

跑法：
    python scripts/preview_dify_console.py
产出：`docs/reports/dify_console_<日期>/0N_*.png`

前提：桥服务在 127.0.0.1:8300（不然概览里的桥字段会是空的，截图没有参考价值）。
"""
import io
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import rag_web_server as W   # noqa: E402
import prompt_manager as PM  # noqa: E402

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUTDIR = os.path.join(ROOT, "docs", "reports",
                      "dify_console_%s" % time.strftime("%Y%m%d"))
SHOTS = [("ov", "01_概览.png", 1500, 1180),
         ("kb", "02_知识库.png", 1500, 900),
         ("pg", "03_检索演练.png", 1500, 1500),
         ("lg", "04_请求流水.png", 1500, 1100),
         ("gd", "05_接入指引.png", 1500, 1400),
         ("ap", "06_审批演练.png", 1500, 1700)]


class _StubAuth:
    def verify_token(self, tok):
        return ({"username": "tester", "role": "admin", "tenant_id": "default"}
                if tok == "tok-admin" else None)


def main():
    PM.get_auth_manager = lambda: _StubAuth()
    h = {"Authorization": "Bearer tok-admin"}
    c = W.app.test_client()

    html = c.get("/dify", headers=h).data.decode("utf-8")
    ov = json.loads(c.get("/api/dify/overview", headers=h).data.decode("utf-8"))
    kbc = json.loads(c.get("/api/dify/kb_counts", headers=h).data.decode("utf-8"))
    pack = json.loads(c.post("/api/dify/ai_pack", headers=h,
                             json={"knowledge_id": "kb-jm", "po_id": 33}).data.decode("utf-8"))
    ret = json.loads(c.post("/api/dify/retrieval", headers=h, json={
        "knowledge_id": "kb-jm", "query": "系统支持哪些通信协议",
        "top_k": 5, "score_threshold": 0}).data.decode("utf-8"))

    mock = json.dumps({"overview": ov, "kbc": kbc, "retrieval": ret,
                       "pack": pack},
                      ensure_ascii=False)
    inject = """<script>
window.__MOCK__ = %s;
window.fetch = function(u){
  var k = String(u), b = null;
  if (k.indexOf('/api/dify/overview') >= 0) b = window.__MOCK__.overview;
  else if (k.indexOf('/api/dify/kb_counts') >= 0) b = window.__MOCK__.kbc;
  else if (k.indexOf('/api/dify/ai_pack') >= 0) b = window.__MOCK__.pack;
  else if (k.indexOf('/api/dify/retrieval') >= 0) b = window.__MOCK__.retrieval;
  return Promise.resolve(new Response(JSON.stringify(b),
    {status:200, headers:{'Content-Type':'application/json'}}));
};
</script>
""" % mock
    html = html.replace("<script>", inject + "<script>", 1)
    tail = """<script>
(function(){
  var t = new URLSearchParams(location.search).get('t');
  setTimeout(function(){
    if (t) { var el = document.querySelector('.tab[data-p="' + t + '"]'); if (el) el.click(); }
    setTimeout(function(){
      if (t === 'pg') { document.getElementById('pgRes').innerHTML = '';
        renderHits(window.__MOCK__.retrieval); }
      if (t === 'ap') { document.getElementById('apRes').innerHTML = '';
        renderPack(window.__MOCK__.pack); }
    }, 400);
  }, 500);
})();
</script></body>"""
    html = html.replace("</body>", tail, 1)

    os.makedirs(OUTDIR, exist_ok=True)
    page = os.path.join(ROOT, "_preview_dify.html")
    io.open(page, "w", encoding="utf-8").write(html)
    print("preview -> %s (%d bytes)" % (page, len(html)))

    url_base = "file:///" + page.replace("\\", "/")
    for tab, name, w, hh in SHOTS:
        dst = os.path.join(OUTDIR, name)
        r = subprocess.run([CHROME, "--headless=new", "--disable-gpu",
                            "--hide-scrollbars", "--force-device-scale-factor=1.4",
                            "--window-size=%d,%d" % (w, hh),
                            "--virtual-time-budget=4000",
                            "--screenshot=" + dst, url_base + "?t=" + tab],
                           capture_output=True, text=True)
        print("  %-18s %s %s" % (name, "OK" if os.path.exists(dst) else "FAIL",
                                 "" if os.path.exists(dst) else (r.stderr or "")[-160:]))
    print("-> %s" % OUTDIR)


if __name__ == "__main__":
    main()
