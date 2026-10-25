#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dify_kb_server.py —— Dify 外部知识库 API 桥（双向桥的「出口」侧）
================================================================================

【这个文件解决的问题】

Dify 的「外部知识库」允许把检索外包出去：Dify 只管编排应用、写提示词、管会话，
**检索这一步发 HTTP 到你的服务**。协议就一个接口：

    POST {endpoint}/retrieval
    Authorization: Bearer {API_KEY}
    {
      "knowledge_id": "kb-default",
      "query": "定位方式有哪几种",
      "retrieval_setting": {"top_k": 4, "score_threshold": 0.5},
      "metadata_condition": {"logical_operator": "and", "conditions": [...]}
    }

    →
    {"records": [{"content": "...", "score": 0.93,
                  "title": "指令表 第12页",
                  "metadata": {"file_name": "指令表.pdf"}}]}

也就是说：**Dify 应用可以直接吃你带 ACL 的存量文档**，不需要把它们重新
传到 Dify 的知识库里（重新上传 = 权限模型作废 + 一份数据两处维护）。

【为什么不干脆让 Dify 自己管知识库】

本项目检索栈是 dense + sparse(BM25) 双路召回 → RRF 跨 query 融合 →
bge-reranker-v2-m3 cross-encoder 精排 → 租户/密级/拥有者三重过滤。
Dify 自带知识库只有单路向量召回 + 可选 rerank，且租户隔离粒度到
「知识库」一级，做不到本项目这种 partition_key + scalar expr + 应用层
ACL 的三层防线。所以正确姿势是**把检索留在本项目**，Dify 只做编排。

【三个必须堵的口子（不堵就等于开了后门）】

  1. knowledge_id 必须白名单。
     它直接决定 tenant_id / role / user_id 三个检索参数，而 tenant_id 在
     advanced_rag_agent.py:874-884 是要 **f-string 拼进 Milvus expr** 的。
     透传外部字符串 = 开放 Milvus 表达式注入。
     → 未登记的 knowledge_id 一律 2001，不做任何 fallback。

  2. top_k 必须钳制。
     Dify 表单里能填 100，本项目单 query 召回深度只有 20
     （RETRIEVE_CANDIDATE_K），不钳制就是让 reranker 白跑几十条。

  3. score 语义必须换算。
     Dify: 0~1，越大越相关。
     本项目: 距离，越小越相关；hybrid 下是 -fused，量级 1/61 ≈ 0.016。
     直接把 0.016 当 score 传给 Dify，结果是「全部低于阈值 → 0 命中」，
     而且**不报错**，是最阴的一种假失败。
     → 取 reranker 的 relevance_score（真实语义相关度）换算。

【怎么跑】

    # 1) 配密钥（.env）
    #    DIFY_KB_API_KEY=<随机串>
    #    python -c "import secrets;print(secrets.token_urlsafe(32))"

    # 2) 自检配置（不连 Milvus）
    python dify_kb_server.py --check

    # 3) 起服务
    python dify_kb_server.py --config config/dify_bridge.yaml

    # 4) 手工打一发（确认不是"日志 ok 但没结果"）
    curl -sS -X POST http://127.0.0.1:8300/retrieval \
      -H "Authorization: Bearer $DIFY_KB_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{"knowledge_id":"kb-default","query":"定位方式",
           "retrieval_setting":{"top_k":3,"score_threshold":0.0}}'

    # 5) Dify 后台 → 知识库 → 连接外部知识库
    #    API Endpoint: http://<本机IP>:8300
    #    API Key:      $DIFY_KB_API_KEY
    #    外部知识库 ID: kb-default

【设计取舍】

  · 用标准库 http.server，不引 Flask/FastAPI。
    本文件是「旁挂」在 rag_web_server.py 之外的一个进程，不该为了一个
    接口把整套 Web 框架依赖再拉一遍；顺带也避免了 import rag_web_server
    带来的副作用（它 import 时就替换 builtins.print 并装全局 stdout 复用器）。
  · 直接调 LangGraphRAGApp._do_retrieve，而不是自己写一套检索。
    自己写「差不多的检索」必然与线上漂移，最后 Dify 里效果和主站对不上。
    手法抄 evalkit/harness_retrieval.py:66 的 _LiteHost：只喂
    vector_db / user / tenant_id 三份状态，不起 MySQL/Redis/LLM 网关。
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import hmac
import io
import json
import logging
import math
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

LOG = logging.getLogger("dify_kb")

# Dify 外部知识库协议的错误码（官方文档固定值，不要自己发明）
ERR_BAD_AUTH_HEADER = 1001   # Authorization 头格式不对（不是 Bearer xxx）
ERR_AUTH_FAILED = 1002       # API Key 不匹配
ERR_KB_NOT_FOUND = 2001      # knowledge_id 不存在 / 未启用

DEFAULT_CONFIG_PATH = os.path.join(_BASE_DIR, "config", "dify_bridge.yaml")


# ============================================================================
# 第一部分：配置
# ============================================================================

class BridgeConfig:
    """config/dify_bridge.yaml 的内存形态"""

    def __init__(self, raw: Dict[str, Any]):
        server = raw.get("server") or {}
        self.host: str = str(server.get("host") or "127.0.0.1")
        # 注意：这里用的是 `or`，所以写 port: 0 **不会**得到随机端口，
        # 而是回落到 8300。"随机端口"在部署配置里没有意义，只会让人误判。
        # 要用随机端口（测试场景）请在代码里显式覆盖 cfg.port。
        self.port: int = int(server.get("port") or 8300)
        self.request_timeout: int = int(server.get("request_timeout") or 30)
        self.max_concurrency: int = int(server.get("max_concurrency") or 4)

        auth = raw.get("auth") or {}
        self.api_key_env: str = str(auth.get("api_key_env") or "DIFY_KB_API_KEY")

        limits = raw.get("limits") or {}
        self.max_top_k: int = int(limits.get("max_top_k") or 20)
        self.max_query_len: int = int(limits.get("max_query_len") or 512)
        self.max_records: int = int(limits.get("max_records") or 20)
        # 单 query 送 reranker 的候选池宽度。**不要**跟主链路的
        # RETRIEVE_CANDIDATE_K=20 对齐：Dify 外部检索是硬超时，而 reranker
        # 在 CPU 上随候选数超线性变慢（见 _retrieve_pairs 的实测数据）。
        self.candidate_k: int = int(limits.get("candidate_k") or 5)
        # ★ 2026-10-25：送进精排的单条字符上限（只对本进程生效）。
        #   实测精排耗时≈正比于送检总字符数：20 条 ~30 字→0.9s，~358 字→7.1s，
        #   ~1704 字→51.7s。Dify 召回的 parent chunk 就有 1701 字的 → 一次检索 35 秒。
        #   0 / 未配 = 不覆盖，沿用主链路默认（3000），行为与改造前一致。
        #   填 400 后：评测 A/B 显示 Recall@5 0.962→1.000、nDCG@5 0.973→0.975、
        #   整轮耗时 766s→270s（-64.7%），MRR 1.000→0.962（单条 jm-008 从第1掉到第2）。
        self.rerank_max_chars: int = int(limits.get("rerank_max_chars") or 0)

        score = raw.get("score") or {}
        self.rank_base: float = float(score.get("rank_base", 0.95))
        self.rank_step: float = float(score.get("rank_step", 0.05))
        self.sigmoid_when_out_of_range: bool = bool(
            score.get("sigmoid_when_out_of_range", True))

        # knowledge_id -> 检索身份。key 大小写敏感（Dify 里填什么这里就得是什么）
        self.knowledge_bases: Dict[str, Dict[str, Any]] = {}
        for item in (raw.get("knowledge_bases") or []):
            kid = str(item.get("knowledge_id") or "").strip()
            if not kid:
                continue
            self.knowledge_bases[kid] = {
                "knowledge_id": kid,
                "label": str(item.get("label") or ""),
                "tenant_id": str(item.get("tenant_id") or "default"),
                "role": str(item.get("role") or "user"),
                "user_id": str(item.get("user_id") or "dify-svc"),
                "enabled": bool(item.get("enabled", True)),
            }

    @property
    def api_key(self) -> str:
        return os.getenv(self.api_key_env, "")

    def kb(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """白名单查询：只有这里返回非 None 的 knowledge_id 才允许检索"""
        item = self.knowledge_bases.get(knowledge_id)
        if not item or not item.get("enabled"):
            return None
        return item


def _load_config(path: str) -> BridgeConfig:
    if not os.path.exists(path):
        raise SystemExit(f"配置文件不存在: {path}")
    try:
        import yaml  # pyyaml；项目里 llm_gateway 也用它
    except ImportError:
        raise SystemExit("缺少 pyyaml：pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"配置文件格式不对（顶层必须是映射）: {path}")
    return BridgeConfig(raw)


# ============================================================================
# 第二部分：检索宿主（复用线上代码，不重写检索逻辑）
# ============================================================================

class _BridgeHost:
    """
    轻量宿主：为复用 LangGraphRAGApp._do_retrieve 提供最小依赖集。

    与 evalkit/harness_retrieval.py 的 _LiteHost 同一手法、同一目的：
    _do_retrieve 只需要 self.vector_db / self.user / self.tenant_id 三份状态，
    但它内部会回调兄弟方法（_rrf_fuse_queries / _rerank），所以留 __getattr__
    兜底到真实的 LangGraphRAGApp 上取同名函数并绑定。

    这样做的收益是「不会漂移」：线上以后改了 _do_retrieve（加一路召回、
    换 reranker），这个桥自动跟着变，不需要同步改两份。
    """

    def __init__(self, vector_db, user_id: str, tenant_id: str):
        self.vector_db = vector_db
        self.user = user_id
        self.tenant_id = tenant_id
        self.username = user_id          # _do_retrieve 上游偶尔会读，补齐

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        import langgraph_rag_agent as L
        attr = getattr(L.LangGraphRAGApp, name, None)
        if attr is None:
            raise AttributeError(
                f"_BridgeHost 与 LangGraphRAGApp 均无属性 {name!r}；"
                f"若线上检索链路新增了依赖，请在此补齐最小状态。")
        if callable(attr):
            return attr.__get__(self, type(self))
        return attr


class DifyRetrievalService:
    """把「一次 Dify 检索请求」翻译成「一次线上检索调用」"""

    def __init__(self, cfg: BridgeConfig):
        self.cfg = cfg
        self._vector_db = None
        self._init_lock = threading.Lock()
        # 限流：reranker 只有 1 个进程，并发打过去收益为负
        self._slots = threading.BoundedSemaphore(max(1, cfg.max_concurrency))
        self.stats = {"requests": 0, "records": 0, "errors": 0,
                      "last_error": "", "started_at": time.time()}
        self._stats_lock = threading.Lock()
        # ★ 控制台用的滑动窗口（内存恒定，不落盘）：
        #   recent  —— 最近 40 条请求，供 web 控制台的「请求流水」直接渲染；
        #   latencies —— 最近 200 次耗时，用来算 p50 / p95。
        #   为什么不从日志文件里 grep：日志是给人看的（还混着 ua/长度），
        #   解析日志当数据源，改一行日志格式就把面板打崩。
        self.recent: "collections.deque[dict]" = collections.deque(maxlen=40)
        self.latencies: "collections.deque[float]" = collections.deque(maxlen=200)

    # ------------------------------------------------------------------ init
    def vector_db(self):
        """懒初始化向量库：起服务时不连 Milvus，第一次检索才连。

        好处是 `--check` 模式与容器健康检查不会因为 Milvus 还没起来就挂掉，
        Dify 那边也就能先建好外部知识库再去点"测试连接"。
        """
        if self._vector_db is not None:
            return self._vector_db
        with self._init_lock:
            if self._vector_db is not None:
                return self._vector_db
            import advanced_rag_agent as A
            self._quietly(A.VectorStoreManager.init_vector_store,
                          label="init_vector_store")
            return self._vector_db

    def _quietly(self, fn, *args, label: str = "", **kwargs):
        """执行并吞掉业务 print（_do_retrieve / VectorStoreManager 全程 print）。

        为什么要吞：本进程是 stdout 被打到日志/终端的守护进程，业务 print
        会与访问日志搅在一起；更麻烦的是 rag_web_server 家族代码里的 print
        带 emoji 和进度条，重定向到日志后极难检索。
        """
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                result = fn(*args, **kwargs)
        except Exception:
            captured = buf.getvalue().strip()
            if captured:
                LOG.debug("[%s] suppressed output:\n%s", label, captured)
            raise
        captured = buf.getvalue().strip()
        if captured:
            LOG.debug("[%s] suppressed output:\n%s", label, captured)
        # init_vector_store 的返回值是向量库实例，只有它需要被留下
        if label == "init_vector_store":
            self._vector_db = result
        return result

    # -------------------------------------------------------------- retrieval
    def _retrieve_pairs(self, kb: Dict[str, Any], query: str) -> List:
        """
        真正调线上检索的地方，也是**测试里唯一需要替换的接缝**。

        单独成方法的原因：`import langgraph_rag_agent` 会拉起 langgraph /
        memory_store / Milvus 客户端一整串依赖，自测不该为了验证
        「score 换算 / 白名单 / 出参整形」把那一串全拉起来。
        自测子类只覆盖这一个方法，就能把 retrieve() 的其余逻辑（分数字段、
        阈值过滤、top_k 截断、record 整形）原样跑一遍。

        返回：[(Document, rrf_score), ...]
        """
        vdb = self.vector_db()
        host = _BridgeHost(vdb, kb["user_id"], kb["tenant_id"])
        import langgraph_rag_agent as L
        # keep_score=True 的关键：只有它能把 reranker 的 relevance_score
        # 带出来。不改这个开关，我们拿到的分数是 RRF 的 -fused（≈0.016），
        # 换算成 Dify 的 0~1 只能是编的。
        #
        # ★ 候选池必须按桥自己的配置收窄，不能沿用主链路的
        #   RETRIEVE_CANDIDATE_K=20。原因是 Dify 外部检索是硬超时
        #   （实测约 6s、失败重试 3 次），而 llama.cpp reranker 在单 CPU 上
        #   随候选数**超线性**膨胀 —— 同一 query、同一批文档实测：
        #       5 条 → 1.03s ／ 10 条 → 6.30s ／ 20 条 → 22.09s
        #   20 条稳定 22s，Dify 侧必然耗光 3 次重试还判「操作失败」。
        #   收窄到 5 条后全链路 ~1.6s（embed 0.2 + search 0.5 + rerank 1.0）。
        # 实现方式：临时代改模块全局再还原（_do_retrieve 内部按全局名查找，
        #   这样不用复制一份检索逻辑，将来主链路改召回/换 reranker 仍然跟随）。
        #   并发安全：retrieve() 用 _slots 把 _retrieve_pairs 串行化了，
        #   max_concurrency>1 时最坏也只是互相把值设成同一个 candidate_k。
        prev_k = L.RETRIEVE_CANDIDATE_K
        L.RETRIEVE_CANDIDATE_K = self.cfg.candidate_k
        try:
            return self._quietly(
                L.LangGraphRAGApp._do_retrieve, host, [query], kb["role"],
                True,  # keep_score
                label="do_retrieve",
            )
        finally:
            L.RETRIEVE_CANDIDATE_K = prev_k

    def retrieve(self, kb: Dict[str, Any], query: str, top_k: int,
                 score_threshold: float,
                 metadata_condition: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        执行一次检索，返回 Dify 协议格式的 records。

        参数：
            kb:                 白名单条目（tenant_id / role / user_id 已确定）
            query:              原始问题
            top_k:              已钳制过的条数
            score_threshold:    低于此分的丢弃（Dify 语义，0~1）
            metadata_condition: 元数据过滤条件（可选）
        返回：
            [{"content","score","title","metadata"}, ...]，无命中返回 []
        """
        with self._slots:
            pairs = self._retrieve_pairs(kb, query)

        if not pairs:
            return []

        # 融合后的上限：本项目 _do_retrieve 返回 RETRIEVE_CANDIDATE_K(20) 条，
        # Dify 可能只要 3 条，这里先按 max_records 卡一刀再逐条算分。
        cap = min(len(pairs), self.cfg.max_records)

        out: List[Dict[str, Any]] = []
        prev_score = 1.0
        for rank_idx, (doc, rrf_score) in enumerate(pairs[:cap]):
            meta = getattr(doc, "metadata", None) or {}
            if metadata_condition and not _match_metadata(meta, metadata_condition):
                continue
            score = _to_dify_score(meta, rrf_score, rank_idx, self.cfg)
            # ---- 单调性修正（只影响"混装"这一种情况）----
            # 正常情况下整批候选要么全有 reranker 分，要么全没有（reranker 挂了
            # 就整批退化），分数天然单调。唯一会混装的是 figure-aware 二次召回：
            # 图页文档被直接插到最前面，**没经过 reranker**，于是它拿到排名
            # 归一化的 0.95，而紧跟其后的真实 rerank 0.52 反而更低 —— 用户
            # 在 Dify 引用列表里看到的分数是倒序的，像 bug。
            # 处理：无 rerank 分的记录，分数压在**上一条**之下 0.01，保住分辨率
            # （比整体 running-min 抹平到 0.52 更保留信息）。
            if rank_idx > 0 and meta.get("_rerank_score") is None:
                score = min(score, max(0.0, prev_score - 0.01))
            if score < score_threshold:
                continue
            prev_score = score
            out.append({
                "content": (getattr(doc, "page_content", "") or ""),
                "score": round(score, 6),
                "title": _record_title(meta),
                # 注意：Dify 明确要求 metadata 是「对象」，不能是 null。
                # 没内容就给 {}，千万别给 None（Dify 会抛 parsing error）。
                "metadata": _record_metadata(meta, kb),
            })
            if len(out) >= top_k:
                break

        with self._stats_lock:
            self.stats["requests"] += 1
            self.stats["records"] += len(out)
        return out

    def note_error(self, msg: str) -> None:
        with self._stats_lock:
            self.stats["errors"] += 1
            self.stats["last_error"] = msg[:300]

    def note_request(self, kb: str, query: str, hit: int, ms: float,
                     top_k: int = 0, http: int = 200,
                     code: int = 0, note: str = "", thr: float = 0.0) -> None:
        """记一条请求进滑动窗口（控制台数据源）。

        `code` 是 Dify 官方错误码（1001/1002/2001），0 表示成功；
        `note` 给「探测 / 鉴权失败」这类非检索请求留一句人话说明；
        `thr` 记 Dify 这次实际传进来的 score_threshold ——
        🔴 它是排查「检索成功但 0 命中」的第一现场证据：分数是 0.27~0.37 时，
           阈值只要填 0.5 就全灭，而日志里不记这个值就只能猜。
        """
        with self._stats_lock:
            self.recent.appendleft({
                "ts": round(time.time(), 3),
                "clock": time.strftime("%H:%M:%S"),
                "kb": kb or "-",
                "q": (query or "")[:80],
                "qlen": len(query or ""),
                "top_k": top_k,
                "thr": round(float(thr or 0.0), 3),
                "hit": hit,
                "ms": round(ms, 1),
                "http": http,
                "code": code,
                "note": note,
            })
            if http == 200 and not note:
                self.latencies.append(ms)

    def latency_summary(self) -> Dict[str, Any]:
        """p50 / p95 / avg（毫秒）。样本不足就返回实际样本数，不编数。"""
        with self._stats_lock:
            xs = sorted(self.latencies)
        if not xs:
            return {"n": 0, "p50": None, "p95": None, "avg": None,
                    "min": None, "max": None}
        def _pct(p: float) -> float:
            k = max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1)))))
            return round(xs[k], 1)
        return {"n": len(xs), "p50": _pct(0.50), "p95": _pct(0.95),
                "avg": round(sum(xs) / len(xs), 1),
                "min": round(xs[0], 1), "max": round(xs[-1], 1)}

    def recent_requests(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._stats_lock:
            return list(self.recent)[:max(0, limit)]


# ============================================================================
# 第三部分：score 语义换算（本文件最容易写错的地方）
# ============================================================================

# 排名归一化的告警只打一次（见 _to_dify_score ③）
_RANK_FALLBACK_WARNED = [False]


def _sigmoid(x: float) -> float:
    # 防溢出：x 很负时 exp(-x) 会 inf
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _to_dify_score(meta: Dict[str, Any], rrf_score: Any, rank_idx: int,
                   cfg: BridgeConfig) -> float:
    """
    把本项目的「距离越小越相关」换算成 Dify 的「0~1 越大越相关」。

    三条路径，优先级从高到低：

    ① reranker 的 relevance_score（首选）
       bge-reranker-v2-m3 输出的是 cross-encoder 的真实语义相关度，
       这是唯一能诚实地当「相关度」用的量。
       ⚠️ 但它的原始值可能是 logits（-12~8）也可能是已 sigmoid 过的
       概率（0~1）—— 取决于 llama.cpp 的版本与是否有 --rerank-normalize。
       所以这里自动判断：值在 [0,1] 就照用，越界才压 sigmoid。
       不判断的后果：logits=8 直接当 score 传给 Dify → Dify 校验
       「score must be between 0 and 1」→ 整次检索失败。

    ② 距离换算（没有 reranker 时的退化）
       两种可能形态，都是「越小越相关」：
         · 纯 dense（HYBRID_SEARCH=false）：`1 - cosine相似度`，值域 [0,2]，
           是余弦距离 → 用 1 - dist/2 换回相似度；
         · hybrid：`-fused`，量级 1/61 ≈ 0.016 且为负 → 绝对值无意义
           （不能设 0.5 阈值），只能走 ③。
       注意别把「距离」直接当「相似度」返回：距离 0.05 会被当成"很不相关"，
       而它其实是"几乎完全相同"。这个方向搞反了，检索结果顺序依然是
       对的（Dify 只是拿来过滤），但阈值一设就全灭 —— 属于最难查的那类。

    ③ 排名归一化（最后兜底）
       score = rank_base - rank_step * rank_idx
       只保证「排序正确、阈值可设」，不代表真实相关度。
       这条路径会打 warning —— 因为这时候 Dify 里的 score 是"假"的，
       出现「阈值调半天没反应」时要知道根因在这里。
    """
    raw = meta.get("_rerank_score")
    if raw is not None:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            val = None
        if val is not None:
            if 0.0 <= val <= 1.0:
                return val
            if cfg.sigmoid_when_out_of_range:
                return _sigmoid(val)
            # 不压 sigmoid 就只能 clamp，会丢掉分辨率，但至少不越界
            return max(0.0, min(1.0, val))

    # ② cosine 距离形态（纯 dense 召回下出现，值域 [0,2]）
    try:
        dist = float(rrf_score)
    except (TypeError, ValueError):
        dist = None
    if dist is not None and 0.0 <= dist <= 2.0:
        return max(0.0, min(1.0, 1.0 - dist / 2.0))

    # ③ 排名归一化
    if rank_idx == 0 and not _RANK_FALLBACK_WARNED[0]:
        # 只警告一次：这条路径每次检索都会走到，逐条打日志会把日志淹掉，
        # 反而看不到真正有用的请求日志。
        _RANK_FALLBACK_WARNED[0] = True
        # ★ 2026-10-25：原文案只猜「可能 RERANK_ENABLED=false」，实测踩过一个反例 ——
        #   开关是 true、端点却连不上（llama.cpp 没起 / 只绑了 127.0.0.1），
        #   结果同样是「有结果、分数是 0.95/0.90 的阶梯」，看日志还以为精排开着。
        #   所以把两个真实环境值一并打出来，让「开关开了但服务没起」一眼可辨。
        LOG.warning("无 _rerank_score，已退化为排名归一化 score（此时 score 只反映名次，"
                    "Dify 里的 score_threshold 不具语义）。RERANK_ENABLED=%s "
                    "RERANK_URL=%s —— 开关为 true 却仍走到这里，通常是精排端点不可达。"
                    "（本进程只提示这一次）",
                    os.getenv("RERANK_ENABLED", "(未设)"),
                    os.getenv("RERANK_URL", "(未设)"))
    return max(0.0, cfg.rank_base - cfg.rank_step * rank_idx)


# ============================================================================
# 第四部分：metadata 过滤与出参整形
# ============================================================================

def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _cmp_one(actual: Any, operator: str, expected: Any) -> bool:
    """单个条件的比较。operator 取值见 Dify 文档（中英文两套都认）。"""
    op = (operator or "").strip().lower()
    a = "" if actual is None else actual

    if op in ("empty", "为空"):
        return a == "" or a is None
    if op in ("not empty", "不为空"):
        return not (a == "" or a is None)

    # 数值/日期比较：能转 float 就按数值比，否则按字符串比（ISO 日期字符串
    # 的字典序与时间序一致，所以 "before"/"after" 直接用字符串比较是对的）
    if op in (">", ">=", "≥", "<", "<=", "≤", "before", "after"):
        try:
            x, y = float(a), float(expected)
        except (TypeError, ValueError):
            x, y = str(a), str(expected)
        if op == ">":
            return x > y
        if op in (">=", "≥"):
            return x >= y
        if op == "<":
            return x < y
        if op in ("<=", "≤"):
            return x <= y
        if op == "before":
            return x < y
        return x > y  # after

    sa, se = str(a), str(expected)
    if op in ("contains", "包含"):
        return se in sa
    if op in ("not contains", "不包含"):
        return se not in sa
    if op in ("start with", "startswith", "开始是", "开头是"):
        return sa.startswith(se)
    if op in ("end with", "endswith", "结束是", "结尾是"):
        return sa.endswith(se)
    if op in ("is", "=", "等于", "是"):
        # 布尔/数字走宽松比较，避免 "true" vs True 这种坑
        if _as_bool(sa) == _as_bool(se) and str(se).lower() in ("true", "false"):
            return True
        return sa == se
    if op in ("is not", "≠", "!=", "不等于", "不是"):
        if str(se).lower() in ("true", "false"):
            return _as_bool(sa) != _as_bool(se)
        return sa != se
    LOG.warning("未知的 metadata 比较算子 %r，按不匹配处理", operator)
    return False


def _match_metadata(meta: Dict[str, Any], cond: Dict[str, Any]) -> bool:
    """
    执行 Dify 的 metadata_condition。

    结构：
        {"logical_operator": "and" | "or",
         "conditions": [{"name": "file_name", "comparison_operator": "contains",
                         "value": "指令表"}]}

    约定：name 先按本项目 metadata 的原生键取（file_name / section_path /
    access_level / chunk_type / page / chunk_index），取不到再看
    source（等价 file_name 的另一个名字）。都取不到时视为空串，
    这样 "empty"/"not empty" 也能正常用。
    """
    if not isinstance(cond, dict):
        return True
    conditions = cond.get("conditions") or []
    if not conditions:
        return True
    logic = str(cond.get("logical_operator") or "and").strip().lower()

    results = []
    for c in conditions:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name") or "").strip()
        if not name:
            continue
        expected = c.get("value")
        actual = meta.get(name)
        if actual is None and name in ("file_name", "document_name"):
            actual = meta.get("source")
        results.append(_cmp_one(actual, str(c.get("comparison_operator") or "is"),
                                expected))
    if not results:
        return True
    return all(results) if logic == "and" else any(results)


def _record_title(meta: Dict[str, Any]) -> str:
    """
    title 是给人看的。Dify 在「引用与归属」里直接展示它，
    所以拼成「文件名 第N页」比只给文件名更容易对上原文。
    """
    fname = meta.get("file_name") or os.path.basename(
        str(meta.get("source") or "")) or "未命名文档"
    page = meta.get("page")
    section = (meta.get("section_path") or "").strip()
    if section:
        return f"{fname} · {section}"
    if page not in (None, "", 0):
        return f"{fname} 第{page}页"
    return str(fname)


def _record_metadata(meta: Dict[str, Any], kb: Dict[str, Any]) -> Dict[str, Any]:
    """
    出参 metadata：Dify 要求「对象」，不能 null。

    带上 tenant_id / access_level 是刻意的：Dify 侧排查"为什么这条没出来"
    时，一眼能看出是不是被租户或密级卡掉了。不要为了"少暴露信息"把它删掉，
    这个接口本身已经要求 Bearer 鉴权。
    """
    keep = ("file_name", "source", "page", "chunk_index", "section_path",
            "access_level", "chunk_type", "parent_id")
    out: Dict[str, Any] = {}
    for k in keep:
        v = meta.get(k)
        if v in (None, ""):
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
    out["tenant_id"] = kb.get("tenant_id", "")
    out["knowledge_id"] = kb.get("knowledge_id", "")
    return out


# ============================================================================
# 第五部分：HTTP 层
# ============================================================================

class DifyKBHandler(BaseHTTPRequestHandler):
    server_version = "DifyKBBridge/1.0"
    # 由 StreamRequestHandler.setup() 用来 settimeout，挡住半开连接
    timeout = 30

    # ---- 由 ThreadingHTTPServer 注入（见 main）----
    cfg: BridgeConfig = None       # type: ignore[assignment]
    service: DifyRetrievalService = None  # type: ignore[assignment]

    # -- 工具 ------------------------------------------------------------
    def log_message(self, fmt: str, *args):  # noqa: A003
        LOG.info("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        # default=str：/ai/pack 的 metrics 里可能带 datetime / Decimal（来自
        # 取数 SQL 的原始类型）。宁可序列化成字符串，也不要因为一个日期类型
        # 把整个审批请求打成 500。
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # 客户端（Dify）提前断开：不是错误，不要把它算进统计
            pass

    def _error(self, status: int, code: int, msg: str) -> None:
        self.service.note_error(f"{code}: {msg}")
        LOG.warning("拒绝请求 http=%s code=%s msg=%s", status, code, msg)
        self._send_json(status, {"error_code": code, "error_msg": msg})

    def _check_auth(self) -> bool:
        """Bearer 鉴权。错误码严格按 Dify 文档：1001 格式错 / 1002 不匹配。"""
        header = self.headers.get("Authorization") or ""
        expected = self.cfg.api_key
        if not expected:
            # 没配密钥 = 配置事故，不能"放行"。宁可不可用也不要裸奔。
            self.service.note_request("-", "", 0, 0.0, http=500,
                                      code=ERR_AUTH_FAILED,
                                      note="服务端未配置 API Key")
            self._error(500, ERR_AUTH_FAILED,
                        f"服务端未配置 API Key（环境变量 {self.cfg.api_key_env} 为空）")
            return False
        parts = header.split(None, 1)
        if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
            self.service.note_request("-", "", 0, 0.0, http=403,
                                      code=ERR_BAD_AUTH_HEADER,
                                      note="Authorization 格式错")
            self._error(403, ERR_BAD_AUTH_HEADER,
                        "Invalid Authorization header format. "
                        "Expected 'Bearer <api-key>' format.")
            return False
        # 定长比较，避免时序侧信道
        if not hmac.compare_digest(parts[1].strip(), expected):
            self.service.note_request("-", "", 0, 0.0, http=403,
                                      code=ERR_AUTH_FAILED, note="密钥不匹配")
            self._error(403, ERR_AUTH_FAILED, "Authorization failed.")
            return False
        return True

    # -- GET /health ------------------------------------------------------
    def do_GET(self):  # noqa: N802
        # ★ 原始请求日志：Dify 的「保存外部知识库 API」会先做一次探测，
        #   但它请求的路径/方法跟文档写的不一定一致（实测会带前缀差异），
        #   不加这行日志就永远只能看着 "Not Found: failed to connect" 猜。
        LOG.info("REQ %s %s  auth=%s  ua=%s", self.command, self.path,
                 "yes" if self.headers.get("Authorization") else "no",
                 (self.headers.get("User-Agent") or "-")[:40])
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path in ("/health", "/healthz"):
            st = dict(self.service.stats)
            st["uptime_sec"] = round(time.time() - st.pop("started_at"), 1)
            # ★ 精排状态必须在这里暴露：RERANK_ENABLED 写了 true 但端点挂了时，
            #   检索仍然「成功」，只是所有 score 退化成排名阶梯（0.95/0.90/…）。
            #   不把这两个值亮出来，控制台就只能看到「有结果但分数很假」。
            rerank_on = (os.getenv("RERANK_ENABLED", "false").lower()
                         in ("1", "true", "yes", "on"))
            self._send_json(200, {
                "status": "ok",
                "vector_db_ready": self.service._vector_db is not None,
                "api_key_configured": bool(self.cfg.api_key),
                "knowledge_bases": [
                    {"knowledge_id": k, "label": v["label"],
                     "tenant_id": v["tenant_id"], "role": v["role"],
                     "enabled": v["enabled"]}
                    for k, v in self.cfg.knowledge_bases.items()
                ],
                "rerank": {
                    "enabled": rerank_on,
                    "url": os.getenv("RERANK_URL", ""),
                    "timeout_sec": os.getenv("RERANK_TIMEOUT", ""),
                },
                "limits": {
                    "max_concurrency": self.cfg.max_concurrency,
                    "candidate_k": self.cfg.candidate_k,
                    "rerank_max_chars": self.cfg.rerank_max_chars,
                    "max_top_k": self.cfg.max_top_k,
                    "max_query_len": self.cfg.max_query_len,
                    "max_records": self.cfg.max_records,
                },
                "latency_ms": self.service.latency_summary(),
                "recent": self.service.recent_requests(20),
                "stats": st,
            })
            return

        # ★ Dify 保存「外部知识库 API」时会先做一次**连通性探测**：
        #     GET {endpoint}/retrieval   （带 Authorization: Bearer <key>）
        #   只要这一步不是 200，Dify 就拒绝保存，并报
        #     ValueError: Not Found: failed to connect to the endpoint: {endpoint}
        #   而 POST /retrieval 其实是好的 —— 表现为「检索接口明明能调，
        #   但填进 Dify 永远『操作失败』」，而且错误信息说的是"连不上"，
        #   完全不提方法不匹配，极容易误判成网络/防火墙问题。
        #   （2026-10-23 实测踩坑）
        #   这里同时容忍 endpoint 误填成 .../retrieval 的情况（那会变成
        #   GET /retrieval/retrieval），免得用户改了地址还得再试一轮。
        if path in ("/retrieval", "/retrieval/retrieval", "/ai/pack"):
            if not self._check_auth():
                return
            self._send_json(200, {
                "status": "ok",
                "probe": path,
                "hint": "连通性探测通过。检索请用 POST /retrieval；"
                        "审批数字包请用 POST /ai/pack",
            })
            return

        self._send_json(404, {"error": "not found",
                              "hint": "GET /health；POST /retrieval；POST /ai/pack"})

    # -- POST 路由 -------------------------------------------------------
    def do_POST(self):  # noqa: N802
        LOG.info("REQ %s %s  auth=%s  len=%s", self.command, self.path,
                 "yes" if self.headers.get("Authorization") else "no",
                 self.headers.get("Content-Length") or 0)
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path == "/retrieval":
            self._handle_retrieval()
            return
        if path == "/ai/pack":
            self._handle_ai_pack()
            return
        self._send_json(404, {
            "error_code": ERR_KB_NOT_FOUND,
            "error_msg": f"unknown endpoint {path}",
            "hint": "可用：POST /retrieval（外部知识库）、POST /ai/pack"
                    "（审批数字包）、GET /health"})

    # -- 公共：读 JSON body（限制大小，防止被打爆内存）---------------------
    def _read_json_body(self) -> Optional[Dict[str, Any]]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            self._send_json(400, {"error_code": 2001, "error_msg": "empty body"})
            return None
        if length > 1024 * 256:
            self._send_json(413, {"error_code": 2001, "error_msg": "body too large"})
            return None
        raw = self.rfile.read(length)
        try:
            req = json.loads(raw.decode("utf-8"))
        except Exception as e:
            self._send_json(400, {"error_code": 2001, "error_msg": f"invalid JSON: {e}"})
            return None
        if not isinstance(req, dict):
            self._send_json(400, {"error_code": 2001, "error_msg": "body must be object"})
            return None
        return req

    # -- POST /retrieval（Dify 外部知识库）--------------------------------
    def _handle_retrieval(self):
        if not self._check_auth():
            return

        req = self._read_json_body()
        if req is None:
            return

        # ---- knowledge_id 白名单（最关键的防线）----
        knowledge_id = str(req.get("knowledge_id") or "").strip()

        # ★ 空 knowledge_id = Dify 的**连通性探测**，不是业务请求。
        #   Dify 在「添加外部知识库 API」保存时会先发一个
        #     POST {endpoint}/retrieval
        #     {"knowledge_id": "", "query": "...", "retrieval_setting": {...}}
        #   （实测 body 92 字节，knowledge_id 为空——因为那一步还没指定知识库）。
        #   它只看响应码：**404 就直接判定「连不上」**，并抛出
        #     ValueError: Not Found: failed to connect to the endpoint
        #   表现为「检索接口明明能调，但 API 永远存不进去」。
        #   所以空 ID 必须回 200 + 空结果，这不是放水：
        #   没有 tenant/role/user 任何检索参数，不查任何数据，零信息泄露。
        #   （2026-10-23 实测踩坑：这一条卡了整整两轮）
        if not knowledge_id:
            LOG.info("连通性探测（knowledge_id 为空）→ 200 空结果")
            self.service.note_request("-", req.get("query") or "", 0, 0.0,
                                      http=200, note="连通性探测")
            self._send_json(200, {"records": []})
            return

        kb = self.cfg.kb(knowledge_id)
        if kb is None:
            # 注意：这里**不能**回落到 default 租户。回落等于"随便填个 ID
            # 就能查默认租户"，而 tenant_id 是要拼进 Milvus expr 的。
            self.service.note_request(knowledge_id, req.get("query") or "", 0,
                                      0.0, http=404, code=ERR_KB_NOT_FOUND,
                                      note="知识库未登记/已停用")
            self._error(404, ERR_KB_NOT_FOUND,
                        f"knowledge base {knowledge_id!r} not found or disabled")
            return

        query = str(req.get("query") or "").strip()
        if not query:
            self._send_json(400, {"error_code": 2001, "error_msg": "query is empty"})
            return
        if len(query) > self.cfg.max_query_len:
            self._send_json(400, {
                "error_code": 2001,
                "error_msg": f"query too long ({len(query)} > {self.cfg.max_query_len})"})
            return

        # ---- retrieval_setting（Dify 不传时的默认值：top_k=4, threshold=0）----
        rs = req.get("retrieval_setting") or {}
        if not isinstance(rs, dict):
            rs = {}
        try:
            top_k = int(rs.get("top_k", 4))
        except (TypeError, ValueError):
            top_k = 4
        top_k = max(1, min(top_k, self.cfg.max_top_k))   # 钳制：见文件头「口子 2」
        try:
            score_threshold = float(rs.get("score_threshold", 0.0) or 0.0)
        except (TypeError, ValueError):
            score_threshold = 0.0

        meta_cond = req.get("metadata_condition")
        if meta_cond is not None and not isinstance(meta_cond, dict):
            meta_cond = None

        t0 = time.time()
        try:
            records = self.service.retrieve(kb, query, top_k,
                                            score_threshold, meta_cond)
        except Exception as e:
            # 检索链路炸了（Milvus 掉线 / reranker 挂）→ 500 + 明确文案。
            # 绝不返回空 records 冒充"没找到"，那会让 Dify 侧误判为知识库没内容。
            import traceback
            LOG.error("检索失败: %s\n%s", e, traceback.format_exc())
            self.service.note_error(f"retrieve failed: {e}")
            self.service.note_request(knowledge_id, query, 0,
                                      (time.time() - t0) * 1000, top_k=top_k,
                                      http=500, code=ERR_KB_NOT_FOUND,
                                      note=f"后端异常 {type(e).__name__}")
            self._send_json(500, {
                "error_code": 2001,
                "error_msg": f"retrieval backend error: {type(e).__name__}: {e}"})
            return

        ms = (time.time() - t0) * 1000
        LOG.info("retrieval ok kb=%s top_k=%d thr=%s hit=%d %.0fms query=%r",
                 knowledge_id, top_k, score_threshold, len(records), ms, query[:60])
        self.service.note_request(knowledge_id, query, len(records), ms,
                                  top_k=top_k, http=200, thr=score_threshold)
        self._send_json(200, {"records": records})

    # -- POST /ai/pack（AI 审批用的数字包，给 Dify 工作流拉）--------------
    def _handle_ai_pack(self):
        """
        给 Dify 工作流的 HTTP 节点用：一次拿到「算好的数字 + 规则结论 + 渲染好的材料」。

        返回里的 `material` 是直接可以塞进 Dify LLM 节点 user prompt 的纯文本，
        Dify 侧不需要自己拼字符串、也不需要自己算指标——那样既容易算错，
        也会让「同一张单在两个地方算出两个数」。

        `gate.auto_eligible` 只是个**提示**（方便 Dify 画布上做条件分支），
        真正的能不能批由服务端 `ApprovalGate.decide()` 说了算：即便 Dify 画布
        被人改成「一律走自动批」，权限链照样会拒。
        """
        if not self._check_auth():
            return
        req = self._read_json_body()
        if req is None:
            return

        try:
            po_id = int(req.get("po_id"))
        except (TypeError, ValueError):
            self._send_json(400, {"ok": False,
                                  "error": "po_id 必须是整数"})
            return

        # 租户从 knowledge_id 白名单反查——不接受调用方直传 tenant_id，
        # 否则等于把「查哪个租户」的决定权交给了 Dify 画布。
        kb_id = str(req.get("knowledge_id") or "kb-default").strip()
        kb = self.cfg.kb(kb_id)
        if kb is None:
            self._error(404, ERR_KB_NOT_FOUND,
                        f"knowledge base {kb_id!r} not found or disabled")
            return
        tenant = kb["tenant_id"]

        try:
            import ai_approval as A
            from approval import required_level_for
        except Exception as e:
            self._send_json(500, {"ok": False,
                                  "error": f"审批模块加载失败：{type(e).__name__}: {e}"})
            return

        policy = A.load_policy()
        builder = A.DataPackBuilder(policy, tenant)
        if not builder.available:
            # 取数不了就不能给 Dify 任何数字。返回空 metrics 会让 LLM
            # 在「无数据」上编出判断，比直接报错危险得多。
            self._send_json(503, {"ok": False,
                                  "error": f"数据库不可用，无法取数：{builder.error}"})
            return

        pack = builder.build(po_id)
        if not pack.get("ok"):
            self._send_json(404, {"ok": False, "error": pack.get("error")})
            return

        rule = A.RuleEngine(policy).score(pack)
        history = A.HistoryDigest(builder, policy).build(pack)
        route = A.RoutePlanner(policy).plan(pack, rule)
        # 每站一份只属于自己职责的材料：Dify 的迭代节点遍历 route，
        # 每一项都有现成的 material 可以塞进 LLM 节点的 user。
        for n in route.get("nodes") or []:
            n["material"] = A.render_node_material(pack, history, rule, n)

        po = pack.get("po") or {}
        amount = (pack.get("metrics") or {}).get("amount")
        required = required_level_for("purchase_order.approve", amount)
        lim = policy.get("limits") or {}
        ai_level = int((policy.get("ai_approver") or {}).get("level") or 2)
        max_auto = float(lim.get("max_auto_amount") or 0)
        route_level = int(route.get("max_level") or 2)
        need_level = max(required, route_level)

        LOG.info("ai/pack po=%s tenant=%s amount=%s rule=%s/%s route=%s站(malx_level=%s) "
                 "material=%d字",
                 po_id, tenant, amount, rule.get("risk_level"),
                 rule.get("risk_score"), len(route.get("nodes") or []),
                 route_level,
                 sum(len(n.get("material") or "") for n in route.get("nodes") or []))
        self._send_json(200, {
            "ok": True,
            "po": {"id": po.get("id"), "po_no": po.get("po_no"),
                   "supplier_id": po.get("supplier_id"),
                   "supplier_code": po.get("supplier_code"),
                   "supplier_name": po.get("supplier_name"),
                   "amount": amount, "status": po.get("status"),
                   "buyer": po.get("buyer"),
                   "expected_date": str(po.get("expected_date") or "")},
            "items": (pack.get("evidence") or {}).get("items") or [],
            "metrics": pack.get("metrics") or {},
            "missing": pack.get("missing") or [],
            "rule": {"risk_score": rule.get("risk_score"),
                     "risk_level": rule.get("risk_level"),
                     "hard_block": rule.get("hard_block"),
                     "blockers": rule.get("blockers") or [],
                     "findings": rule.get("findings") or []},
            "history": history,
            "route": route,
            "gate": {"required_level": required,
                     "route_max_level": route_level,
                     "effective_level": need_level,
                     "ai_level": ai_level,
                     "max_auto_amount": max_auto,
                     "auto_eligible": bool(need_level <= ai_level
                                           and amount is not None
                                           and amount <= max_auto
                                           and not rule.get("hard_block"))},
        })


def _build_server(cfg: BridgeConfig, service: DifyRetrievalService) -> ThreadingHTTPServer:
    handler = type("_BoundHandler", (DifyKBHandler,), {
        "cfg": cfg,
        "service": service,
        "timeout": cfg.request_timeout,
    })
    ThreadingHTTPServer.allow_reuse_address = True
    httpd = ThreadingHTTPServer((cfg.host, cfg.port), handler)
    httpd.daemon_threads = True
    return httpd


# ============================================================================
# 第六部分：入口
# ============================================================================

def _load_env_file(path: Optional[str] = None) -> int:
    """把项目根 `.env` 注入 os.environ（**只填缺失的键**，不覆盖已有值）。

    🔴 为什么必须有这个：本文件的文档一直写着「密钥在 .env 里配 DIFY_KB_API_KEY」，
    但进程从来不读 .env —— 此前能跑起来全靠启动时在 shell 里手动 export。
    一旦换成 systemd / 计划任务 / 后台托管启动，就会变成「密钥为空 → 所有请求 1002」，
    而现象是一句 `服务端未配置 API Key（环境变量 ... 为空）` 的 500，
    看着像代码 bug，其实是环境没进来。（2026-10-25 实测踩到）
    """
    if path is None:
        path = os.path.join(_BASE_DIR, ".env")
    if not os.path.isfile(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
                n += 1
    return n


def _local_ip() -> str:
    """取本机对外 IP（只为在启动日志里提示 Dify 该填什么地址）"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _local_ips() -> List[str]:
    """列出本机所有 IPv4（去环回），供「Dify 该填哪个地址」逐条核对。

    ★ 2026-10-25 为什么要这个：原来是 `_local_ip()` 单值，它取默认路由网卡，
    在「Dify 跑 VMware NAT 虚拟机」的场景下会给出**宿主机办公网地址**
    （实测 192.168.0.106），而虚拟机只能走 NAT 网关那一个（192.168.200.1）。
    照日志填进 Dify 就是 failed to connect，还以为是防火墙。
    """
    out: List[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127.") and ip not in out:
                out.append(ip)
    except Exception:
        pass
    primary = _local_ip()
    if primary and not primary.startswith("127.") and primary not in out:
        out.insert(0, primary)
    # 默认路由网卡排最前（同机访问最常用），其余按字符串排，保证输出稳定
    out.sort(key=lambda x: (x != primary, x))
    return out or ["127.0.0.1"]


def _print_check(cfg: BridgeConfig) -> int:
    """--check：只验配置与密钥，不打扰 Milvus。返回进程退出码。"""
    print("=" * 74)
    print("Dify 外部知识库桥 · 配置自检")
    print("=" * 74)
    print(f"监听地址      : {cfg.host}:{cfg.port}")
    print(f"密钥环境变量  : {cfg.api_key_env}")
    print(f"密钥是否已配  : {'是' if cfg.api_key else '否  ← 必须配，否则所有请求都会 1002'}")
    print(f"top_k 上限    : {cfg.max_top_k}")
    print(f"query 长度上限: {cfg.max_query_len}")
    print(f"返回条数上限  : {cfg.max_records}")
    print(f"并发上限      : {cfg.max_concurrency}")
    print(f"精排截断上限  : {cfg.rerank_max_chars or '(不覆盖，沿用主链路 3000)'}")
    print("-" * 74)
    print(f"{'knowledge_id':<22}{'tenant':<10}{'role':<8}{'enabled':<9}label")
    ok = True
    for kid, v in cfg.knowledge_bases.items():
        print(f"{kid:<22}{v['tenant_id']:<10}{v['role']:<8}"
              f"{('是' if v['enabled'] else '否'):<9}{v['label']}")
    if not cfg.knowledge_bases:
        print("⚠ 未登记任何 knowledge_id —— Dify 那边填什么都只会收到 2001")
        ok = False
    ids = [k for k, v in cfg.knowledge_bases.items() if v["enabled"]]
    if not ids:
        print("⚠ 没有任何 enabled: true 的知识库")
        ok = False
    print("-" * 74)
    bind_note = ("（监听 0.0.0.0，下面每个地址都可填；"
                 "Dify 在虚拟机里时用它与宿主机同网段的那个）"
                 if cfg.host == "0.0.0.0" else "")
    print(f"Endpoint（填进 Dify）: {bind_note}")
    for ip in (_local_ips() if cfg.host == "0.0.0.0" else [cfg.host]):
        print(f"    http://{ip}:{cfg.port}")
    print(f"外部知识库 ID 候选   : {', '.join(ids) if ids else '（无）'}")
    print(f"AI 审批数字包端点    : POST /ai/pack"
          f"（Dify 工作流 HTTP 节点拉；见 docs/guides/AI_APPROVAL_WORKFLOW.md）")
    print("=" * 74)
    return 0 if ok else 2


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Dify 外部知识库 API 桥（POST /retrieval）")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                    help="配置文件路径（默认 config/dify_bridge.yaml）")
    ap.add_argument("--host", default=None, help="覆盖监听地址")
    ap.add_argument("--port", type=int, default=None, help="覆盖监听端口")
    ap.add_argument("--check", action="store_true",
                    help="只自检配置并退出，不连 Milvus、不起服务")
    ap.add_argument("-v", "--verbose", action="store_true", help="DEBUG 日志")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    n_env = _load_env_file()
    cfg = _load_config(args.config)
    if args.host:
        cfg.host = args.host
    if args.port:
        cfg.port = args.port
    if n_env:
        LOG.info("已从 .env 注入 %d 个环境变量（只补缺失的键）", n_env)
    # ★ 精排送检截断：必须在 langgraph_rag_agent 首次 import 之前写进环境，
    #   它是该模块的模块级常量（import 期读取）。
    if cfg.rerank_max_chars > 0:
        os.environ["RERANK_MAX_CHARS"] = str(cfg.rerank_max_chars)
        LOG.info("精排送检文本截断：%d 字符（只影响本进程，主链路仍用默认值）",
                 cfg.rerank_max_chars)

    if args.check:
        return _print_check(cfg)

    if not cfg.api_key:
        LOG.error("环境变量 %s 为空。先在 .env 里配密钥，否则所有请求都会 1002。",
                  cfg.api_key_env)
        # 仍然允许启动：容器编排里常见「先起服务后注密钥」，
        # 但每个请求都会被挡，不会裸奔。

    service = DifyRetrievalService(cfg)
    try:
        httpd = _build_server(cfg, service)
    except OSError as e:
        LOG.error("端口 %s:%s 绑定失败: %s", cfg.host, cfg.port, e)
        return 1

    LOG.info("Dify 知识库桥已启动 http://%s:%s  endpoint=/retrieval", cfg.host, cfg.port)
    LOG.info("Dify 侧填写 → API Endpoint: %s  知识库 ID: %s",
             " | ".join("http://%s:%s" % (ip, cfg.port)
                        for ip in (_local_ips() if cfg.host == "0.0.0.0"
                                   else [cfg.host])),
             ", ".join(k for k, v in cfg.knowledge_bases.items() if v["enabled"]) or "（未登记）")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        LOG.info("收到 Ctrl-C，正在停止…")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
