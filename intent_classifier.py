# =============================================================================
# 意图识别组件（v2 语义路由范式）
# -----------------------------------------------------------------------------
# 设计目标：
#   - 线上线下统一，不再受 fast_mode 决定是否走 LLM 的规则网关约束。
#   - 主路径 L1 语义路由（bge-m3 向量余弦）：零 LLM 成本，天然抗改写/同义。
#   - L2 仅当 L1 低置信/歧义时，用 LLM JSON 结构化输出兜底（gateway 暂无 tools）。
#   - embedder 离线时降级到传入的 lexical_fn（即原 _quick_classify）。
#   - 每次判定记录 classify_source / confidence，喂给 bad case 归因。
#
# 用法（在 LangGraphRAGApp.node_classify 内）：
#     res = self.intent_classifier.classify(
#         query,
#         lexical_fn=self._quick_classify,
#         llm_fn=self._llm_classify_json)
#     # res.intent / res.confidence / res.source
# =============================================================================

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class IntentResult:
    intent: str
    confidence: float = 0.0
    source: str = "unknown"
    candidates: List = field(default_factory=list)
    # source ∈ {
    #   "semantic"                L1 语义路由高置信判定
    #   "semantic:ambiguous->llm" L1 歧义，L2 LLM 兜底判定
    #   "lexical"                 embedder 不可用/未就绪，降级 lexical 规则判定
    #   "cache"                   命中缓存
    #   "fallback:default"        全失败，取 default_intent
    # }
    # candidates: [(intent, score), ...] 按分数降序，供 bad case 归因看"模型在哪些意图间纠结"


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class IntentClassifier:
    def __init__(self, config_path: str = None, embedder=None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "config", "intents.yaml")
        self._cfg = self._load_config()
        self._meta = self._cfg.get("_meta", {})
        self._intents = self._cfg.get("intents", {})
        self._embedder = embedder  # 可注入（测试用）；默认懒加载 OllamaEmbeddings
        self._ready = False        # centroids 是否已构建
        self._centroids: Dict[str, List[float]] = {}
        # 并发保护（web 服务多线程共享同一实例）+ 构建失败冷却
        # （Ollama 离线时若无冷却，每个请求都会重试 N 次 embed HTTP —— 复用
        #   kb_version 的 fail-cooldown 模式：失败后 60s 内直接走降级不重试）
        self._build_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._centroid_fail_until = 0.0
        self._centroid_fail_cooldown = float(
            self._meta.get("centroid_fail_cooldown", 60.0))
        self._cache: Dict[str, IntentResult] = {}
        self._cache_max = int(self._meta.get("cache_size", 1024))
        self._threshold = float(self._meta.get("threshold", 0.60))
        self._ambiguity_gap = float(self._meta.get("ambiguity_gap", 0.08))
        self._llm_tiebreak = bool(self._meta.get("llm_tiebreak", True))
        self._default_intent = self._meta.get("default_intent", "simple")

    # ---------- 配置 ----------
    def _load_config(self) -> dict:
        if yaml is None:  # pragma: no cover
            raise RuntimeError("PyYAML 未安装，无法加载 intents.yaml")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # ---------- embedder（懒加载，避免 import torch）----------
    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        try:
            from advanced_rag_agent import _make_embedder
            self._embedder = _make_embedder()
            # 显式超时：httpx 默认 5s 覆盖大多数场景，但半开 TCP/模型冷加载时
            # 一次挂起的 embed 会拖住整个 classify（流式场景表现为"永远无回复"）。
            # 分类器专用超时可配置，不影响检索管线的共享 embedder。
            try:
                kw = dict(getattr(self._embedder, "client_kwargs", None) or {})
                kw.setdefault("timeout",
                              float(os.getenv("INTENT_EMBED_TIMEOUT", "10")))
                self._embedder.client_kwargs = kw
            except Exception:  # noqa: BLE001
                pass  # 赋值失败仅意味着用库默认超时，不影响功能
        except Exception as e:
            print(f"[IntentClassifier] embedder 初始化失败: {e}")
            self._embedder = None
        return self._embedder

    def _embed(self, text: str) -> Optional[List[float]]:
        emb = self._get_embedder()
        if emb is None:
            return None
        # 关键：OllamaEmbeddings 默认无 per-call timeout，Ollama 卡死时 embed_query
        # 会无限挂起，进而把 _ensure_centroids 的 _build_lock 锁死、整个服务冻结。
        # 用守护线程 + join(timeout) 强制有界等待，超时即视为失败 → 走降级/冷却。
        timeout = float(self._meta.get("embed_timeout", 8.0))
        box: List = []
        failed: List = []
        def _run():
            try:
                box.append(emb.embed_query(text))
            except Exception:  # noqa: BLE001
                failed.append(True)
        th = threading.Thread(target=_run, daemon=True)
        th.start()
        th.join(timeout)
        if th.is_alive():
            print(f"[IntentClassifier] ⚠ embed_query 超时({timeout:.0f}s)，"
                  f"按失败降级（防 Ollama 卡死冻结服务）")
            return None
        if failed:
            print(f"[IntentClassifier] embed_query 失败，按失败降级")
            return None
        return box[0] if box else None

    # ---------- centroids（懒构建，带失败冷却与并发保护）----------
    def _ensure_centroids(self):
        if self._ready:
            return
        with self._build_lock:
            if self._ready:          # 双重检查：等锁期间可能已被其他线程构建
                return
            if time.time() < self._centroid_fail_until:
                # 冷却期内不重试（Ollama 离线时避免每个请求都打一轮超时 HTTP）
                return
            try:
                all_vecs: Dict[str, List[List[float]]] = {}
                for intent, spec in self._intents.items():
                    examples = spec.get("examples", [])
                    if not examples:
                        # 该意图未配置示例 → 整体不可用
                        print(f"[IntentClassifier] ⚠ 意图「{intent}」无示例，"
                              f"语义路由不可用，{self._centroid_fail_cooldown:.0f}s 内降级 lexical 不重试")
                        self._centroid_fail_until = (
                            time.time() + self._centroid_fail_cooldown)
                        return
                    vecs = []
                    for ex in examples:
                        v = self._embed(ex)   # 已带超时，Ollama 卡死时快速失败
                        if v is None:
                            # 任一示例 embed 失败/超时 → 整轮不可用，立即进入冷却；
                            # 不在循环中逐条重试，避免把单次延迟放大成「每请求数十秒」
                            print(f"[IntentClassifier] ⚠ 意图「{intent}」示例 embed 失败/超时，"
                                  f"语义路由不可用，{self._centroid_fail_cooldown:.0f}s 内降级 lexical 不重试")
                            self._centroid_fail_until = (
                                time.time() + self._centroid_fail_cooldown)
                            return
                        vecs.append(v)
                    dim = len(vecs[0])
                    centroid = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
                    all_vecs[intent] = centroid
                    # 进度可见：冷构建期间流式对话框不再完全静默
                    # （print 经 stdout 多路复用器转发为 SSE 日志事件）
                    print(f"[IntentClassifier] 意图路由预热 "
                          f"{len(all_vecs)}/{len(self._intents)}: {intent} ✓")
                if all_vecs:
                    self._centroids = all_vecs
                    self._ready = True
                    self._centroid_fail_until = 0.0
            except Exception as e:  # noqa: BLE001
                print(f"[IntentClassifier] centroid 构建异常(进入冷却): {e}")
                self._centroid_fail_until = (
                    time.time() + self._centroid_fail_cooldown)

    # ---------- 预热 ----------
    def warmup(self) -> bool:
        """
        【启动期预热】预构建 centroid，把冷构建成本从首个用户请求挪到服务启动。

        由 LangGraphRAGApp 在后台线程调用；失败返回 False（运行时自动降级
        lexical + 冷却重试），绝不抛异常拖死启动流程。
        """
        try:
            self._ensure_centroids()
        except Exception as e:  # noqa: BLE001
            print(f"[IntentClassifier] 预热异常(运行时降级): {e}")
        return self._ready

    # ---------- 主入口 ----------
    def classify(self, query: str,
                 lexical_fn: Callable[[str], str] = None,
                 llm_fn: Callable[[str, List[str]], Optional[str]] = None,
                 use_cache: bool = True) -> IntentResult:
        q_norm = (query or "").strip().lower()
        cache_key = hashlib.md5(q_norm.encode("utf-8")).hexdigest()

        # 0. 缓存（读；命中直接短路）
        if use_cache:
            with self._cache_lock:
                hit = self._cache.get(cache_key)
            if hit is not None:
                return IntentResult(hit.intent, hit.confidence, "cache",
                                    candidates=hit.candidates)

        # 1. L1 语义路由（主路径）
        self._ensure_centroids()
        if self._ready:
            qv = self._embed(query)
            if qv is None:
                # 本次 embed 失败 → 降级
                return self._lexical_or_default(query, 0.0, lexical_fn, use_cache, cache_key)
            sims = {it: _cosine(qv, c) for it, c in self._centroids.items()}
            ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
            top_cands = [(it, round(s, 3)) for it, s in ranked[:3]]
            top1, s1 = ranked[0]
            top2, s2 = ranked[1] if len(ranked) > 1 else (None, -1.0)
            confident = s1 >= self._threshold
            unambiguous = (top2 is None) or ((s1 - s2) >= self._ambiguity_gap)
            if confident and unambiguous:
                return self._finish(query, top1, s1, "semantic", use_cache, cache_key, top_cands)

            # 歧义或低置信 → L2 LLM 兜底
            if self._llm_tiebreak and llm_fn is not None:
                raw = llm_fn(query, list(self._intents.keys()))
                parsed = self._parse_llm_intent(raw)
                if parsed and parsed in self._intents:
                    return self._finish(query, parsed, max(s1, 0.5),
                                        "semantic:ambiguous->llm", use_cache, cache_key, top_cands)
            # L2 不可用/失败 → lexical 降级或兜底
            return self._lexical_or_default(query, max(s1, 0.4), lexical_fn, use_cache, cache_key, top_cands)

        # 2. embedder 不可用 → lexical 降级
        return self._lexical_or_default(query, 0.0, lexical_fn, use_cache, cache_key)

    def _lexical_or_default(self, query, conf, lexical_fn, use_cache, cache_key,
                            candidates=None) -> IntentResult:
        if lexical_fn is not None:
            return self._finish(query, lexical_fn(query), conf, "lexical",
                                use_cache, cache_key, candidates)
        return self._finish(query, self._default_intent, conf,
                            "fallback:default", use_cache, cache_key, candidates)

    def _finish(self, query, intent, conf, source, use_cache, cache_key,
                candidates=None) -> IntentResult:
        res = IntentResult(intent=intent, confidence=round(float(conf), 3),
                           source=source, candidates=candidates or [])
        if use_cache:
            with self._cache_lock:
                self._cache[cache_key] = res
                if len(self._cache) > self._cache_max:
                    items = list(self._cache.items())
                    self._cache = dict(items[len(items) // 2:])
        return res

    # ---------- L2 解析 ----------
    @staticmethod
    def _parse_llm_intent(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        raw = raw.strip()
        raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
        try:
            data = json.loads(raw)
            return data.get("intent") or data.get("type")
        except (json.JSONDecodeError, TypeError):
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                    return data.get("intent") or data.get("type")
                except Exception:
                    return None
        return None
