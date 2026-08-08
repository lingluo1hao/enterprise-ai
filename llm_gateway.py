# -*- coding: utf-8 -*-
"""
企业级大模型网关 (LLM Gateway)
================================================================================

解决什么问题
--------------------------------------------------------------------------------
改造前，整个系统通过 `OllamaLLM` 直接怼死一个 Ollama 实例、一个 qwen2:7b 模型：

    - 换模型要改代码，接 OpenAI/DeepSeek/通义千问 要重写一遍
    - 每次 chat() 都新建 ChatPromptTemplate + chain，连接不复用
    - 没有限流，一个死循环 Agent 能把 GPU 打满
    - 没有熔断，模型一挂全站 500，而且会持续雪崩式重试
    - 不知道烧了多少 token，更不知道花了多少钱
    - 流式和非流式两套写法，出口不统一

本模块把这些能力收敛到一个统一入口，并且**保持 BaseLLM 接口不变**，
让现有 16 处 `llm.chat(system, user)` 调用点零改动即可接入。

分层设计
--------------------------------------------------------------------------------
    ┌──────────────────────────────────────────────────────┐
    │  LLMGateway            统一出口 chat() / stream_chat() │
    ├──────────────────────────────────────────────────────┤
    │  治理层   TokenBucket(RPM/TPM) · CircuitBreaker ·     │
    │           CostTracker · Router(任务→模型链)            │
    ├──────────────────────────────────────────────────────┤
    │  适配层   OllamaProvider · OpenAICompatProvider        │
    ├──────────────────────────────────────────────────────┤
    │  连接层   HttpConnectionPool (keep-alive 复用)         │
    ├──────────────────────────────────────────────────────┤
    │  配置层   llm_gateway.yaml (热重载，改配置不重启)       │
    └──────────────────────────────────────────────────────┘

为什么是纯标准库
--------------------------------------------------------------------------------
与 `skill_framework.py` 保持同一约定：核心不依赖任何第三方库
(仅 YAML 解析可选依赖 pyyaml，缺失时自动回退 JSON / 内置默认配置)。
好处是这个网关能被 Agent、MCP Server、单元测试、压测脚本任意导入，
不会因为某一端缺少 langchain 就跑不起来。

作者: enterprise-ai
"""

from __future__ import annotations

import os
import re
import json
import time
import queue
import random
import sqlite3
import threading
import http.client
import redis
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlparse

__all__ = [
    "LLMGateway", "GatewayConfig", "ModelConfig",
    "LLMGatewayError", "RateLimitExceeded", "CircuitOpen", "AllModelsFailed",
    "get_gateway",
]


# ============================================================================
# 异常体系
# ============================================================================
# 分成不同异常类型，方便上层区分「该重试」还是「该降级」还是「该报错」。

class LLMGatewayError(Exception):
    """网关基础异常"""


class RateLimitExceeded(LLMGatewayError):
    """超出限流阈值（等待超时后仍拿不到令牌）"""


class CircuitOpen(LLMGatewayError):
    """熔断器打开，该模型暂时不可用"""


class ProviderError(LLMGatewayError):
    """下游 Provider 调用失败（网络错误 / HTTP 非 200 / 响应格式错误）"""


class AllModelsFailed(LLMGatewayError):
    """路由链上所有模型都失败了，且没有可用降级内容"""


# ============================================================================
# 第一部分：配置层
# ============================================================================
# 目标：换模型 / 调限流 / 改路由，只动配置文件，不动代码，且支持热重载。

@dataclass
class ModelConfig:
    """单个模型的完整配置"""
    name: str                              # 逻辑名，路由表里引用的就是它
    provider: str = "ollama"               # ollama | openai_compat
    model: str = "qwen2:7b"                # 下游真实模型名
    base_url: str = "http://localhost:11434"
    api_key_env: str = ""                  # 从哪个环境变量取 key（不写死密钥）
    tier: str = "large"                    # small | large，仅作标注与可读性
    timeout: float = 120.0                 # 单次请求超时（秒）
    temperature: float = 0.0
    max_tokens: int = 0                    # 0 = 不限制
    seed: int = 42                         # 固定随机种子，让 classify/rewrite/grade 可复现
    # --- Ollama 专属 ---
    # num_ctx：上下文窗口 token 数。**必须显式设置**——Ollama 默认只有 2048，
    # 超出部分会从 prompt **开头**静默截断（实测：把关键信息放开头会被整段丢弃，
    # 模型只能看到末尾的无关内容，答非所问且毫无报错）。RAG 场景 prompt 动辄
    # 4000+ token，不设这个值等于把最相关的文档喂进黑洞。
    num_ctx: int = 0                       # 0 = 不传，沿用 Ollama 默认 2048
    # keep_alive：模型在显存/内存里的常驻时长。默认 5m，过期后下次调用要重新
    # 冷加载（实测 7B/Q4_0 冷加载 ~6s）。设长一点省掉这笔固定开销。
    keep_alive: str = ""                   # 空 = 不传，沿用 Ollama 默认 5m
    # 成本：美元 / 每百万 token。本地 Ollama 填 0
    price_in_per_1m: float = 0.0
    price_out_per_1m: float = 0.0
    # 限流：该模型独立的每分钟请求数 / 每分钟 token 数，0 = 不限
    rpm: int = 0
    tpm: int = 0
    # 熔断参数
    fail_threshold: int = 5                # 连续失败多少次后打开熔断
    recovery_sec: float = 30.0             # 打开后多久进入半开探测
    enabled: bool = True

    @property
    def api_key(self) -> str:
        """延迟读取环境变量，保证密钥永远不落配置文件"""
        return os.getenv(self.api_key_env, "") if self.api_key_env else ""


@dataclass
class GatewayConfig:
    """网关全局配置"""
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    # 路由表：任务类型 -> 模型链（第一个是主模型，其余为 fallback）
    routing: Dict[str, List[str]] = field(default_factory=dict)
    default_chain: List[str] = field(default_factory=list)
    # 全局限流（所有模型合计）
    global_rpm: int = 0
    global_tpm: int = 0
    # 拿不到令牌时最多阻塞等待多久，0 = 不等待直接拒绝
    acquire_timeout: float = 5.0
    # 连接池
    pool_size_per_host: int = 8
    pool_idle_timeout: float = 60.0
    # 单模型失败后的重试次数（同一模型内重试，指数退避）
    max_retries: int = 1
    retry_backoff: float = 0.5
    # 降级：所有模型都失败时返回的兜底文案，为空则抛异常
    degraded_reply: str = ""
    # 配置热重载检查间隔（秒）
    reload_interval: float = 10.0
    # Token 用量持久化：SQLite 文件路径（标准库自带，零依赖）。
    # 留空 = 仅进程内内存累计（重启即丢，但 metrics() 仍可用）；
    # 设路径（如 ./llm_usage.db）= 落盘，支持按用户/按时间查询历史用量。
    usage_db: str = ""

    def chain_for(self, task: str) -> List[str]:
        """按任务类型取模型链，未命中则回退到默认链"""
        chain = self.routing.get(task) or self.default_chain
        # 过滤掉未启用或未定义的模型，避免路由到不存在的配置
        return [m for m in chain if m in self.models and self.models[m].enabled]


# ---------------------------------------------------------------------------
# 配置加载：YAML 优先，缺 pyyaml 时回退 JSON，都没有则用内置默认值
# ---------------------------------------------------------------------------

def _builtin_default_config() -> dict:
    """
    内置默认配置——保证没有任何配置文件时网关也能跑起来。
    行为等价于改造前的单模型 OllamaLLM，做到「无感兼容」。
    """
    ollama_url = os.getenv("OLLAMA_URL", "http://192.168.200.128:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2:7b")
    return {
        "models": {
            "local-qwen": {
                "provider": "ollama",
                "model": ollama_model,
                "base_url": ollama_url,
                "tier": "large",
                "timeout": 120.0,
                "rpm": 60,
                "tpm": 120000,
            }
        },
        "routing": {},
        "default_chain": ["local-qwen"],
        "global_rpm": 120,
        "acquire_timeout": 5.0,
    }


def _parse_config_file(path: str) -> Optional[dict]:
    """读取并解析配置文件，失败返回 None（由调用方决定回退策略）"""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError:
        return None

    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # 可选依赖
        except ImportError:
            print(f"[Gateway] 未安装 pyyaml，无法解析 {path}，尝试同名 .json")
            alt = re.sub(r"\.(yaml|yml)$", ".json", path)
            return _parse_config_file(alt)
        try:
            return yaml.safe_load(raw) or {}
        except Exception as e:
            print(f"[Gateway] YAML 解析失败 {path}: {e}")
            return None
    try:
        return json.loads(raw)
    except Exception as e:
        print(f"[Gateway] JSON 解析失败 {path}: {e}")
        return None


def load_config(path: str = "") -> GatewayConfig:
    """
    加载配置。查找顺序：
        1. 显式传入的 path
        2. 环境变量 LLM_GATEWAY_CONFIG
        3. 同目录 llm_gateway.yaml / .yml / .json
        4. 内置默认配置
    """
    candidates: List[str] = []
    if path:
        candidates.append(path)
    env_path = os.getenv("LLM_GATEWAY_CONFIG", "")
    if env_path:
        candidates.append(env_path)
    here = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(here, "config")
    for name in ("llm_gateway.yaml", "llm_gateway.yml", "llm_gateway.json"):
        candidates.append(os.path.join(config_dir, name))
        candidates.append(os.path.join(here, name))

    data: Optional[dict] = None
    used = "<builtin>"
    for c in candidates:
        data = _parse_config_file(c)
        if data is not None:
            used = c
            break
    if data is None:
        data = _builtin_default_config()

    cfg = GatewayConfig()
    cfg.source_path = used  # type: ignore[attr-defined]

    for name, spec in (data.get("models") or {}).items():
        spec = dict(spec or {})
        spec.pop("name", None)
        # 只保留 ModelConfig 认识的字段，避免配置里多写字段直接崩掉
        valid = {k: v for k, v in spec.items()
                 if k in ModelConfig.__dataclass_fields__}
        cfg.models[name] = ModelConfig(name=name, **valid)

    cfg.routing = {k: list(v) for k, v in (data.get("routing") or {}).items()}
    cfg.default_chain = list(data.get("default_chain") or [])
    if not cfg.default_chain and cfg.models:
        cfg.default_chain = [next(iter(cfg.models))]

    for key in ("global_rpm", "global_tpm", "acquire_timeout",
                "pool_size_per_host", "pool_idle_timeout",
                "max_retries", "retry_backoff", "degraded_reply",
                "reload_interval", "usage_db"):
        if key in data and data[key] is not None:
            setattr(cfg, key, data[key])
    return cfg


# ============================================================================
# 第二部分：连接层 —— HTTP 连接池
# ============================================================================
# 改造前每次调用都新建 chain、底层重新握手 TCP。对本地大模型来说，
# 一次 TCP + HTTP 握手看似只有几十毫秒，但在 Agent 一轮问答要调 6+ 次 LLM
# 的场景下，累积开销相当可观，而且高并发时会把端口耗尽（TIME_WAIT 堆积）。
#
# 这里实现一个按 host:port 分池的 keep-alive 连接池：
#   - 复用 http.client.HTTPConnection，避免重复握手
#   - 空闲超时自动丢弃，防止服务端单方面关连接后拿到坏连接
#   - 拿到坏连接时自动重建一次，对上层透明

class _PooledConn:
    """池中的一条连接，记录最后使用时间用于空闲淘汰"""
    __slots__ = ("conn", "last_used")

    def __init__(self, conn, last_used: float):
        self.conn = conn
        self.last_used = last_used


class HttpConnectionPool:
    """
    线程安全的 HTTP/HTTPS 连接池。

    用法::

        pool = HttpConnectionPool(max_per_host=8)
        status, body = pool.request("POST", url, headers, payload, timeout=60)

    流式场景用 `stream_lines()`，它会在迭代结束后才把连接还池。
    """

    def __init__(self, max_per_host: int = 8, idle_timeout: float = 60.0):
        self.max_per_host = max_per_host
        self.idle_timeout = idle_timeout
        self._pools: Dict[str, queue.LifoQueue] = {}
        self._lock = threading.Lock()
        # 统计：命中复用 vs 新建，用来验证连接池真的生效了
        self.reused = 0
        self.created = 0

    # -- 内部：取/还连接 ---------------------------------------------------
    def _pool_for(self, key: str) -> queue.LifoQueue:
        with self._lock:
            if key not in self._pools:
                self._pools[key] = queue.LifoQueue(maxsize=self.max_per_host)
            return self._pools[key]

    def _new_conn(self, scheme: str, host: str, port: int, timeout: float):
        if scheme == "https":
            return http.client.HTTPSConnection(host, port, timeout=timeout)
        return http.client.HTTPConnection(host, port, timeout=timeout)

    def _acquire(self, scheme: str, host: str, port: int, timeout: float):
        key = f"{scheme}://{host}:{port}"
        pool = self._pool_for(key)
        now = time.time()
        while True:
            try:
                item: _PooledConn = pool.get_nowait()
            except queue.Empty:
                break
            # 空闲太久的连接大概率已被服务端关闭，直接丢弃重建
            if now - item.last_used > self.idle_timeout:
                try:
                    item.conn.close()
                except Exception:
                    pass
                continue
            item.conn.timeout = timeout
            with self._lock:
                self.reused += 1
            return item.conn
        with self._lock:
            self.created += 1
        return self._new_conn(scheme, host, port, timeout)

    def _release(self, scheme: str, host: str, port: int, conn) -> None:
        key = f"{scheme}://{host}:{port}"
        pool = self._pool_for(key)
        try:
            pool.put_nowait(_PooledConn(conn, time.time()))
        except queue.Full:
            # 池满了就直接关掉，不无限增长
            try:
                conn.close()
            except Exception:
                pass

    @staticmethod
    def _split(url: str) -> Tuple[str, str, int, str]:
        u = urlparse(url)
        scheme = u.scheme or "http"
        host = u.hostname or "localhost"
        port = u.port or (443 if scheme == "https" else 80)
        path = u.path or "/"
        if u.query:
            path = f"{path}?{u.query}"
        return scheme, host, port, path

    # -- 对外：普通请求 -----------------------------------------------------
    def request(self, method: str, url: str, headers: Dict[str, str],
                body: Optional[bytes], timeout: float) -> Tuple[int, bytes]:
        """
        发起一次请求并读完响应体。内部对「拿到已被关闭的旧连接」做一次重试。
        """
        scheme, host, port, path = self._split(url)
        last_err: Optional[Exception] = None
        for attempt in range(2):  # 第 2 次是针对坏连接的重建重试
            conn = self._acquire(scheme, host, port, timeout)
            try:
                hdrs = dict(headers)
                hdrs.setdefault("Connection", "keep-alive")
                conn.request(method, path, body=body, headers=hdrs)
                resp = conn.getresponse()
                data = resp.read()
                if resp.will_close:
                    try:
                        conn.close()
                    except Exception:
                        pass
                else:
                    self._release(scheme, host, port, conn)
                return resp.status, data
            except (http.client.HTTPException, OSError) as e:
                last_err = e
                try:
                    conn.close()
                except Exception:
                    pass
                if attempt == 0:
                    continue
        raise ProviderError(f"HTTP 请求失败 {url}: {last_err}")

    # -- 对外：流式请求 -----------------------------------------------------
    def stream_lines(self, method: str, url: str, headers: Dict[str, str],
                     body: Optional[bytes], timeout: float) -> Iterator[bytes]:
        """
        流式读取响应，逐行 yield。连接在迭代彻底结束后才归还，
        中途异常则直接关闭，避免把半读状态的脏连接放回池里。
        """
        scheme, host, port, path = self._split(url)
        conn = self._acquire(scheme, host, port, timeout)
        try:
            hdrs = dict(headers)
            hdrs.setdefault("Connection", "keep-alive")
            conn.request(method, path, body=body, headers=hdrs)
            resp = conn.getresponse()
            if resp.status != 200:
                detail = resp.read()[:500].decode("utf-8", "ignore")
                raise ProviderError(f"HTTP {resp.status}: {detail}")
            for raw in resp:
                line = raw.strip()
                if line:
                    yield line
            if resp.will_close:
                conn.close()
            else:
                self._release(scheme, host, port, conn)
        except GeneratorExit:
            # 上层提前 break，连接状态不确定，直接关掉最安全
            try:
                conn.close()
            except Exception:
                pass
            raise
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            raise

    def stats(self) -> Dict[str, int]:
        with self._lock:
            idle = {k: q.qsize() for k, q in self._pools.items()}
            return {"reused": self.reused, "created": self.created,
                    "idle": idle}  # type: ignore[dict-item]

    def close(self) -> None:
        with self._lock:
            pools = list(self._pools.values())
            self._pools.clear()
        for p in pools:
            while True:
                try:
                    item = p.get_nowait()
                except queue.Empty:
                    break
                try:
                    item.conn.close()
                except Exception:
                    pass


# ============================================================================
# 第三部分：Provider 适配层
# ============================================================================
# 不同厂商的 HTTP 协议不一样，但对上层来说都应该是「输入 system+user，
# 输出文本 + token 用量」。这一层负责抹平差异。
#
#   OllamaProvider      -> POST /api/chat        （本地部署）
#   OpenAICompatProvider-> POST /v1/chat/completions
#                          兼容 OpenAI / DeepSeek / 通义千问 / Moonshot / vLLM

@dataclass
class LLMResponse:
    """统一的调用结果"""
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    provider: str = ""
    latency: float = 0.0
    finish_reason: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _estimate_tokens(text: str) -> int:
    """
    兜底的 token 估算——只在下游没返回 usage 时使用。
    中文按 ~1.5 字符/token，英文按 ~4 字符/token 粗算。
    真实用量优先取下游返回值，估算值仅用于限流预扣。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return int(cjk / 1.5 + other / 4) + 1


class BaseProvider(ABC):
    """Provider 抽象：所有下游后端都实现这两个方法"""

    def __init__(self, cfg: ModelConfig, pool: HttpConnectionPool):
        self.cfg = cfg
        self.pool = pool

    @abstractmethod
    def invoke(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """非流式调用"""

    @abstractmethod
    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """流式调用，逐段 yield 增量文本"""


class OllamaProvider(BaseProvider):
    """
    Ollama 后端。

    用 /api/chat 而不是 langchain 封装，好处有三：
      1. 响应里带 prompt_eval_count / eval_count，是**真实 token 数**，不用估算
      2. 走我们自己的连接池，keep-alive 真正复用
      3. 流式就是 NDJSON，逐行解析即可，不需要额外抽象
    """

    def _payload(self, system_prompt: str, user_prompt: str, stream: bool) -> bytes:
        opts: Dict[str, Any] = {"temperature": self.cfg.temperature, "seed": self.cfg.seed}
        if self.cfg.max_tokens > 0:
            opts["num_predict"] = self.cfg.max_tokens
        # 上下文窗口：不传则 Ollama 用默认 2048，长 prompt 会从开头静默截断
        if self.cfg.num_ctx > 0:
            opts["num_ctx"] = self.cfg.num_ctx
        body = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": stream,
            "options": opts,
        }
        # 常驻时长：避免每次调用重新冷加载权重
        if self.cfg.keep_alive:
            body["keep_alive"] = self.cfg.keep_alive
        return json.dumps(body, ensure_ascii=False).encode("utf-8")

    def invoke(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        url = self.cfg.base_url.rstrip("/") + "/api/chat"
        headers = {"Content-Type": "application/json"}
        start = time.time()
        status, data = self.pool.request(
            "POST", url, headers,
            self._payload(system_prompt, user_prompt, False),
            self.cfg.timeout,
        )
        if status != 200:
            raise ProviderError(
                f"Ollama HTTP {status}: {data[:300].decode('utf-8', 'ignore')}")
        try:
            obj = json.loads(data)
        except Exception as e:
            raise ProviderError(f"Ollama 响应非法 JSON: {e}")

        text = (obj.get("message") or {}).get("content", "")
        return LLMResponse(
            text=text,
            # Ollama 直接给出真实 token 数，无需估算
            prompt_tokens=int(obj.get("prompt_eval_count") or 0),
            completion_tokens=int(obj.get("eval_count") or 0),
            model=self.cfg.model,
            provider="ollama",
            latency=time.time() - start,
            finish_reason=obj.get("done_reason", ""),
        )

    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        url = self.cfg.base_url.rstrip("/") + "/api/chat"
        headers = {"Content-Type": "application/json"}
        for line in self.pool.stream_lines(
            "POST", url, headers,
            self._payload(system_prompt, user_prompt, True),
            self.cfg.timeout,
        ):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            piece = (obj.get("message") or {}).get("content", "")
            if piece:
                yield piece
            if obj.get("done"):
                break


class OpenAICompatProvider(BaseProvider):
    """
    OpenAI 兼容后端 —— 一份实现覆盖 OpenAI / DeepSeek / 通义千问 / Moonshot / vLLM。

    这些厂商都实现了 /v1/chat/completions 与相同的 usage 字段，
    所以「可插拔」不需要为每家写一个类，只要在配置里改 base_url + model 即可。
    """

    def _payload(self, system_prompt: str, user_prompt: str, stream: bool) -> bytes:
        body: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.cfg.temperature,
            "seed": self.cfg.seed,
            "stream": stream,
        }
        if self.cfg.max_tokens > 0:
            body["max_tokens"] = self.cfg.max_tokens
        if stream:
            # 要求返回用量统计，否则流式下拿不到 token 数
            body["stream_options"] = {"include_usage": True}
        return json.dumps(body, ensure_ascii=False).encode("utf-8")

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        key = self.cfg.api_key
        if key:
            h["Authorization"] = f"Bearer {key}"
        return h

    def invoke(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        start = time.time()
        status, data = self.pool.request(
            "POST", url, self._headers(),
            self._payload(system_prompt, user_prompt, False),
            self.cfg.timeout,
        )
        if status != 200:
            raise ProviderError(
                f"{self.cfg.name} HTTP {status}: "
                f"{data[:300].decode('utf-8', 'ignore')}")
        try:
            obj = json.loads(data)
        except Exception as e:
            raise ProviderError(f"{self.cfg.name} 响应非法 JSON: {e}")

        choices = obj.get("choices") or []
        if not choices:
            raise ProviderError(f"{self.cfg.name} 返回空 choices")
        text = (choices[0].get("message") or {}).get("content", "") or ""
        usage = obj.get("usage") or {}
        return LLMResponse(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            model=self.cfg.model,
            provider="openai_compat",
            latency=time.time() - start,
            finish_reason=choices[0].get("finish_reason", ""),
        )

    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        for line in self.pool.stream_lines(
            "POST", url, self._headers(),
            self._payload(system_prompt, user_prompt, True),
            self.cfg.timeout,
        ):
            # SSE 格式：以 "data: " 开头，[DONE] 表示结束
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            for ch in obj.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece:
                    yield piece


_PROVIDER_REGISTRY: Dict[str, type] = {
    "ollama": OllamaProvider,
    "openai_compat": OpenAICompatProvider,
    # 下面这些都是 OpenAI 兼容协议，写成别名纯粹为了配置文件可读性更好
    "openai": OpenAICompatProvider,
    "deepseek": OpenAICompatProvider,
    "dashscope": OpenAICompatProvider,
    "qwen": OpenAICompatProvider,
    "moonshot": OpenAICompatProvider,
    "vllm": OpenAICompatProvider,
}


def build_provider(cfg: ModelConfig, pool: HttpConnectionPool) -> BaseProvider:
    cls = _PROVIDER_REGISTRY.get(cfg.provider.lower())
    if cls is None:
        raise LLMGatewayError(
            f"未知 provider: {cfg.provider}（可选: {sorted(_PROVIDER_REGISTRY)}）")
    return cls(cfg, pool)


# ============================================================================
# 第四部分：治理层
# ============================================================================

class TokenBucket:
    """
    令牌桶限流器（线程安全）。

    与 rag_web_server 里那个按 IP 限 HTTP 请求的不同，这里限的是
    **LLM 调用本身**，而且同时支持两个维度：

        RPM — 每分钟请求数，防止调用次数失控
        TPM — 每分钟 token 数，这才是真正的成本与 GPU 压力来源

    支持阻塞等待：拿不到令牌时短暂等待而不是立刻失败，
    因为 LLM 调用本来就慢，等 1 秒远好过直接报错。
    """

    def __init__(self, rate_per_min: int, capacity: Optional[int] = None):
        self.rate = float(rate_per_min)              # 每分钟补充速率
        self.capacity = float(capacity or rate_per_min)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()
        self.rejected = 0

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        self._tokens = min(self.capacity,
                           self._tokens + elapsed * (self.rate / 60.0))

    def try_acquire(self, n: float = 1.0) -> bool:
        if self.rate <= 0:
            return True  # 未配置限流
        with self._lock:
            self._refill()
            if self._tokens >= n:
                self._tokens -= n
                return True
            self.rejected += 1
            return False

    def acquire(self, n: float = 1.0, timeout: float = 0.0) -> bool:
        """带超时的阻塞获取。timeout=0 等价于 try_acquire。"""
        if self.rate <= 0:
            return True
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            if self.try_acquire(n):
                return True
            if time.monotonic() >= deadline:
                return False
            # 睡到理论上够补一个令牌的时间，加抖动避免线程齐步走
            need = n
            with self._lock:
                need = max(0.0, n - self._tokens)
            wait = min(0.5, max(0.02, need * 60.0 / max(self.rate, 1e-6)))
            time.sleep(wait * (0.8 + 0.4 * random.random()))

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            self._refill()
            return {"tokens": round(self._tokens, 2),
                    "capacity": self.capacity,
                    "rate_per_min": self.rate,
                    "rejected": self.rejected}


def _get_shared_redis():
    """获取共享限流用的 Redis 客户端（与 CacheManager 同源：REDIS_HOST/PORT/DB/PASSWORD）。
    连不上返回 None，调用方降级为内存限流。"""
    try:
        r = redis.Redis(
            host=os.getenv("REDIS_HOST", "192.168.200.128"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD", "dev0619") or None,
            db=int(os.getenv("REDIS_DB", "0")),
            socket_connect_timeout=3, socket_timeout=3,
            decode_responses=True,
        )
        r.ping()
        return r
    except Exception as e:
        print(f"[Gateway] 共享限流 Redis 不可用（{e}），降级内存限流")
        return None


class RedisTokenBucket:
    """
    Redis 令牌桶（多实例共享限流）。

    用 Lua 脚本在 Redis 端原子完成 refill + consume，保证多 gunicorn worker /
    多机部署时全局 RPM/TPM 配额一致。rate 内部换算为每秒补充速率。
    client 为 None 时 try_acquire 直接放行（兼容未配置 Redis）。
    """

    _LUA = """
    local rate = tonumber(ARGV[1])
    local cap  = tonumber(ARGV[2])
    local now  = tonumber(ARGV[3])
    local cost = tonumber(ARGV[4])
    local d = redis.call('HMGET', KEYS[1], 't', 'ts')
    local tokens = d[1]
    local ts = d[2]
    if not tokens then tokens = cap; ts = now end
    tokens = math.min(cap, tokens + (now - ts) * rate)
    if tokens >= cost then
        tokens = tokens - cost
        redis.call('HMSET', KEYS[1], 't', tokens, 'ts', now)
        redis.call('EXPIRE', KEYS[1], 120)
        return 1
    end
    redis.call('HMSET', KEYS[1], 't', tokens, 'ts', now)
    redis.call('EXPIRE', KEYS[1], 120)
    return 0
    """

    def __init__(self, rate_per_min, capacity=None, prefix="b", client=None):
        self.rate = float(rate_per_min)
        self.capacity = float(capacity or rate_per_min)
        self.prefix = prefix
        self.client = client
        self.rejected = 0
        self._script = client.register_script(self._LUA) if client else None

    def try_acquire(self, n: float = 1.0) -> bool:
        if self.rate <= 0 or self.client is None:
            return True
        ok = self._script(keys=[self.prefix], args=[self.rate / 60.0, self.capacity, time.time(), n])
        if not ok:
            self.rejected += 1
        return bool(ok)

    def acquire(self, n: float = 1.0, timeout: float = 0.0) -> bool:
        if self.rate <= 0 or self.client is None:
            return True
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            if self.try_acquire(n):
                return True
            if time.monotonic() >= deadline:
                return False
            wait = min(0.5, max(0.02, n * 60.0 / max(self.rate, 1e-6)))
            time.sleep(wait * (0.8 + 0.4 * random.random()))

    def snapshot(self) -> Dict[str, float]:
        if self.client is None:
            return {"tokens": self.capacity, "capacity": self.capacity,
                    "rate_per_min": self.rate, "rejected": self.rejected}
        d = self.client.hmget(self.prefix, "t", "ts")
        tokens = float(d[0]) if d[0] else self.capacity
        return {"tokens": round(tokens, 2), "capacity": self.capacity,
                "rate_per_min": self.rate, "rejected": self.rejected}


class CircuitBreaker:
    """
    熔断器：三态状态机。

        CLOSED   正常放行。连续失败达到阈值 -> OPEN
        OPEN     直接拒绝，不再打下游（关键：给故障服务喘息，避免雪崩）
                 距离打开时间超过 recovery_sec -> HALF_OPEN
        HALF_OPEN 只放行一个探测请求。成功 -> CLOSED；失败 -> 重新 OPEN

    为什么必须有：主模型挂掉时，如果继续无脑重试，不仅拖慢每个请求
    （每次都要等超时），还会把下游彻底压死。熔断让失败**快速返回**，
    从而给 fallback 让出时间。
    """

    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, fail_threshold: int = 5, recovery_sec: float = 30.0):
        self.fail_threshold = max(1, fail_threshold)
        self.recovery_sec = recovery_sec
        self._state = self.CLOSED
        self._fails = 0
        self._opened_at = 0.0
        self._probing = False
        self._lock = threading.Lock()
        self.open_count = 0

    @property
    def state(self) -> str:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def _maybe_half_open(self) -> None:
        if (self._state == self.OPEN
                and time.monotonic() - self._opened_at >= self.recovery_sec):
            self._state = self.HALF_OPEN
            self._probing = False

    def allow(self) -> bool:
        """是否放行本次请求"""
        with self._lock:
            self._maybe_half_open()
            if self._state == self.CLOSED:
                return True
            if self._state == self.OPEN:
                return False
            # HALF_OPEN：只放一个探测请求
            if not self._probing:
                self._probing = True
                return True
            return False

    def on_success(self) -> None:
        with self._lock:
            self._state = self.CLOSED
            self._fails = 0
            self._probing = False

    def on_failure(self) -> None:
        with self._lock:
            if self._state == self.HALF_OPEN:
                # 探测失败，立刻重新打开并重新计时
                self._state = self.OPEN
                self._opened_at = time.monotonic()
                self._probing = False
                self.open_count += 1
                return
            self._fails += 1
            if self._fails >= self.fail_threshold:
                self._state = self.OPEN
                self._opened_at = time.monotonic()
                self.open_count += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            self._maybe_half_open()
            return {"state": self._state, "fails": self._fails,
                    "open_count": self.open_count}


@dataclass
class ModelStats:
    """单模型的累计指标"""
    calls: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency: float = 0.0
    cost_usd: float = 0.0

    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.calls if self.calls else 0.0


class CostTracker:
    """
    Token 与成本统计。

    按「每百万 token 单价」计算，本地 Ollama 单价为 0，
    但 token 数照样统计——它反映的是 GPU 算力消耗，同样是成本。
    """

    def __init__(self):
        self._stats: Dict[str, ModelStats] = {}
        self._lock = threading.Lock()

    def record(self, model_name: str, cfg: ModelConfig,
               resp: LLMResponse) -> float:
        cost = (resp.prompt_tokens / 1e6 * cfg.price_in_per_1m
                + resp.completion_tokens / 1e6 * cfg.price_out_per_1m)
        with self._lock:
            st = self._stats.setdefault(model_name, ModelStats())
            st.calls += 1
            st.prompt_tokens += resp.prompt_tokens
            st.completion_tokens += resp.completion_tokens
            st.total_latency += resp.latency
            st.cost_usd += cost
        return cost

    def record_failure(self, model_name: str) -> None:
        with self._lock:
            self._stats.setdefault(model_name, ModelStats()).failures += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            out: Dict[str, Any] = {}
            total_cost = total_tokens = total_calls = 0
            for name, st in self._stats.items():
                out[name] = {
                    "calls": st.calls,
                    "failures": st.failures,
                    "prompt_tokens": st.prompt_tokens,
                    "completion_tokens": st.completion_tokens,
                    "total_tokens": st.prompt_tokens + st.completion_tokens,
                    "avg_latency_s": round(st.avg_latency, 2),
                    "cost_usd": round(st.cost_usd, 6),
                }
                total_cost += st.cost_usd
                total_tokens += st.prompt_tokens + st.completion_tokens
                total_calls += st.calls
            return {"per_model": out,
                    "total_calls": total_calls,
                    "total_tokens": total_tokens,
                    "total_cost_usd": round(total_cost, 6)}


# ============================================================================
# 第四点五部分：用量持久化（SQLite，标准库自带，零依赖）
# ============================================================================

class UsageStore:
    """
    Token 用量持久化 —— 让「用户查自己历史用量」成为可能。

    为什么不用内存 dict：进程一重启就没了，且无法按用户/时间检索。
    为什么不用 MySQL/Redis：网关坚持纯标准库，sqlite3 是 Python 自带，
    一个文件落盘即可支撑「按用户聚合 + 按时间区间查询」，完全够用。

    - db_path 为空字符串或 None → 用内存表（进程内可查，重启即丢）
    - db_path 为文件路径        → 落盘 SQLite，重启后历史仍在，跨进程可查
    """

    def __init__(self, db_path: str = ""):
        self._mem: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._conn = None
        self._db_path = (db_path or "").strip()
        if self._db_path:
            # 目录不存在先建，避免直接崩
            parent = os.path.dirname(os.path.abspath(self._db_path))
            if parent and not os.path.isdir(parent):
                os.makedirs(parent, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS usage_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,                 -- 调用时间戳
                    user TEXT NOT NULL DEFAULT 'anonymous',
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL DEFAULT '',
                    task TEXT NOT NULL DEFAULT 'default',
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    latency_s REAL NOT NULL DEFAULT 0,
                    cost_usd REAL NOT NULL DEFAULT 0
                )"""
            )
            self._conn.commit()

    # -- 写入 --------------------------------------------------------------
    def record(self, user: str, model: str, provider: str, task: str,
               prompt_tokens: int, completion_tokens: int,
               latency_s: float, cost_usd: float) -> None:
        total = prompt_tokens + completion_tokens
        # 防御：latency 只可能是「秒」，出现天文数字必然是把时间戳当耗时传了，
        # 与其让看板显示 1785606461s，不如记 0 并留给日志排查
        try:
            latency_s = float(latency_s)
        except (TypeError, ValueError):
            latency_s = 0.0
        if latency_s < 0 or latency_s > 86400:
            latency_s = 0.0
        row = {
            "ts": time.time(), "user": user or "anonymous", "model": model,
            "provider": provider, "task": task or "default",
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": total, "latency_s": latency_s, "cost_usd": cost_usd,
        }
        with self._lock:
            if self._conn is not None:
                self._conn.execute(
                    """INSERT INTO usage_log
                       (ts, user, model, provider, task,
                        prompt_tokens, completion_tokens, total_tokens,
                        latency_s, cost_usd)
                       VALUES (:ts, :user, :model, :provider, :task,
                               :prompt_tokens, :completion_tokens, :total_tokens,
                               :latency_s, :cost_usd)""", row)
                self._conn.commit()
            else:
                self._mem.append(row)

    # -- 读取：单用户聚合 ------------------------------------------------
    def user_usage(self, user: str) -> Dict[str, Any]:
        """某用户的累计用量（调用次数 / token / 成本 / 最近一次时间）"""
        user = user or "anonymous"
        calls = total_p = total_c = 0
        cost = 0.0
        last_ts = 0.0
        if self._conn is not None:
            cur = self._conn.execute(
                """SELECT COUNT(*), SUM(prompt_tokens), SUM(completion_tokens),
                          SUM(cost_usd), MAX(ts)
                   FROM usage_log WHERE user = ?""", (user,))
            row = cur.fetchone()
            if row and row[0]:
                calls, total_p, total_c, cost, last_ts = row
        else:
            with self._lock:
                rows = [r for r in self._mem if r["user"] == user]
            calls = len(rows)
            total_p = sum(r["prompt_tokens"] for r in rows)
            total_c = sum(r["completion_tokens"] for r in rows)
            cost = sum(r["cost_usd"] for r in rows)
            last_ts = max((r["ts"] for r in rows), default=0.0)
        return {
            "user": user, "calls": calls or 0,
            "prompt_tokens": total_p or 0, "completion_tokens": total_c or 0,
            "total_tokens": (total_p or 0) + (total_c or 0),
            "cost_usd": round(cost or 0.0, 6),
            "last_active_ts": last_ts or 0.0,
        }

    # -- 读取：明细列表 --------------------------------------------------
    def usage_log(self, user: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """最近用量明细；user 为 None 时返回全部用户（管理员视角）"""
        if self._conn is not None:
            if user:
                cur = self._conn.execute(
                    """SELECT ts, user, model, provider, task,
                              prompt_tokens, completion_tokens, total_tokens,
                              latency_s, cost_usd
                       FROM usage_log WHERE user = ?
                       ORDER BY ts DESC LIMIT ?""", (user, limit))
            else:
                cur = self._conn.execute(
                    """SELECT ts, user, model, provider, task,
                              prompt_tokens, completion_tokens, total_tokens,
                              latency_s, cost_usd
                       FROM usage_log ORDER BY ts DESC LIMIT ?""", (limit,))
            keys = ["ts", "user", "model", "provider", "task",
                    "prompt_tokens", "completion_tokens", "total_tokens",
                    "latency_s", "cost_usd"]
            return [dict(zip(keys, r)) for r in cur.fetchall()]
        with self._lock:
            rows = [r for r in self._mem if user is None or r["user"] == user]
        rows = sorted(rows, key=lambda r: r["ts"], reverse=True)[:limit]
        return rows

    # -- 读取：全用户排行（管理员视角）------------------------------------
    def top_users(self, limit: int = 50) -> List[Dict[str, Any]]:
        """按 token 消耗从多到少列出所有用户，用于后台「谁在烧钱」看板。"""
        if self._conn is not None:
            cur = self._conn.execute(
                """SELECT user, COUNT(*), SUM(prompt_tokens), SUM(completion_tokens),
                          SUM(total_tokens), SUM(cost_usd), MAX(ts)
                   FROM usage_log GROUP BY user
                   ORDER BY SUM(total_tokens) DESC LIMIT ?""", (limit,))
            rows = cur.fetchall()
        else:
            with self._lock:
                agg: Dict[str, Dict[str, Any]] = {}
                for r in self._mem:
                    a = agg.setdefault(r["user"], {
                        "calls": 0, "p": 0, "c": 0, "t": 0, "cost": 0.0, "last": 0.0})
                    a["calls"] += 1
                    a["p"] += r["prompt_tokens"]
                    a["c"] += r["completion_tokens"]
                    a["t"] += r["total_tokens"]
                    a["cost"] += r["cost_usd"]
                    a["last"] = max(a["last"], r["ts"])
            rows = sorted(
                [(u, a["calls"], a["p"], a["c"], a["t"], a["cost"], a["last"])
                 for u, a in agg.items()],
                key=lambda x: x[4], reverse=True)[:limit]
        return [{
            "user": r[0], "calls": r[1] or 0,
            "prompt_tokens": r[2] or 0, "completion_tokens": r[3] or 0,
            "total_tokens": r[4] or 0, "cost_usd": round(r[5] or 0.0, 6),
            "last_active_ts": r[6] or 0.0,
        } for r in rows]

    # -- 读取：时间区间 --------------------------------------------------
    def usage_range(self, start_ts: float, end_ts: float,
                    user: str = None) -> List[Dict[str, Any]]:
        """某时间区间内的用量明细（用于「这个月我烧了多少」）"""
        if self._conn is not None:
            if user:
                cur = self._conn.execute(
                    """SELECT ts, user, model, provider, task,
                              prompt_tokens, completion_tokens, total_tokens,
                              latency_s, cost_usd
                       FROM usage_log
                       WHERE ts >= ? AND ts <= ? AND user = ?
                       ORDER BY ts DESC""", (start_ts, end_ts, user))
            else:
                cur = self._conn.execute(
                    """SELECT ts, user, model, provider, task,
                              prompt_tokens, completion_tokens, total_tokens,
                              latency_s, cost_usd
                       FROM usage_log
                       WHERE ts >= ? AND ts <= ?
                       ORDER BY ts DESC""", (start_ts, end_ts))
            keys = ["ts", "user", "model", "provider", "task",
                    "prompt_tokens", "completion_tokens", "total_tokens",
                    "latency_s", "cost_usd"]
            return [dict(zip(keys, r)) for r in cur.fetchall()]
        with self._lock:
            return [r for r in self._mem
                    if start_ts <= r["ts"] <= end_ts
                    and (user is None or r["user"] == user)]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ============================================================================
# 第五部分：网关主体
# ============================================================================

@dataclass
class _ModelRuntime:
    """一个模型在运行期的全部附属组件"""
    cfg: ModelConfig
    provider: BaseProvider
    breaker: CircuitBreaker
    rpm_bucket: TokenBucket
    tpm_bucket: TokenBucket


class LLMGateway:
    """
    企业级大模型网关 —— 系统里所有 LLM 调用的唯一出口。

    对上：完全兼容原 `BaseLLM.chat(system_prompt, user_prompt) -> str`，
          现有调用点一行不改即可接入；想启用多模型路由时再补 `task=` 参数。
    对下：统一管理连接池、限流、熔断、降级、计费。

    典型用法::

        gw = LLMGateway()                         # 读 llm_gateway.yaml
        gw.chat(sys_p, user_p)                    # 走 default_chain
        gw.chat(sys_p, user_p, task="classify")   # 路由到小模型
        for piece in gw.stream_chat(sys_p, user_p, task="generate"):
            print(piece, end="")
        gw.metrics()                              # 看 token / 成本 / 熔断状态
    """

    def __init__(self, config_path: str = "", verbose: bool = True):
        self._config_path = config_path
        self._verbose = verbose
        self._lock = threading.RLock()
        self._runtimes: Dict[str, _ModelRuntime] = {}
        self._cfg_mtime = 0.0
        self._last_reload_check = 0.0

        self.cost = CostTracker()
        self.cfg = load_config(config_path)
        self.pool = HttpConnectionPool(
            max_per_host=int(self.cfg.pool_size_per_host),
            idle_timeout=float(self.cfg.pool_idle_timeout),
        )
        self._shared_redis = _get_shared_redis()
        self._global_rpm = self._make_bucket(int(self.cfg.global_rpm or 0), "gw:global:rpm")
        self._global_tpm = self._make_bucket(int(self.cfg.global_tpm or 0), "gw:global:tpm")
        self._build_runtimes()
        self._remember_mtime()

        # 用量持久化：文件落盘或进程内内存，热重载不重建（保留历史）
        self._usage_store = UsageStore(self.cfg.usage_db)

        # 兼容原 OllamaLLM 的统计字段，避免上层读这些属性时报错
        self.call_count = 0
        self.total_time = 0.0

        if self._verbose:
            src = getattr(self.cfg, "source_path", "<builtin>")
            print(f"[Gateway] 配置来源: {src}")
            print(f"[Gateway] 已加载 {len(self._runtimes)} 个模型: "
                  f"{list(self._runtimes)}")
            print(f"[Gateway] 默认链: {self.cfg.default_chain} | "
                  f"路由任务: {list(self.cfg.routing)}")
            if self.cfg.usage_db:
                print(f"[Gateway] Token 用量持久化已开启 -> {self.cfg.usage_db}")
            else:
                print("[Gateway] Token 用量仅进程内累计（未配置 usage_db，重启即丢）")

    # -- 限流桶工厂（Redis 共享 / 内存降级） -------------------------------
    _warned_no_redis = False  # 类级标志，避免重复告警刷屏

    def _make_bucket(self, rate_per_min: int, prefix: str):
        """构造令牌桶：配置了 Redis 则用共享桶（多实例全局配额一致），
        否则降级为进程内内存桶并告警（多实例时配额会被放大 N 倍）。"""
        if rate_per_min <= 0:
            return TokenBucket(0)
        if self._shared_redis is not None:
            return RedisTokenBucket(rate_per_min, prefix=prefix, client=self._shared_redis)
        if not LLMGateway._warned_no_redis:
            print("[Gateway] ⚠ 未检测到 Redis，LLM 限流为进程内内存态；"
                  "多实例部署时全局 RPM/TPM 配额会被放大 N 倍。"
                  "建议设置 REDIS_HOST 启用共享限流。")
            LLMGateway._warned_no_redis = True
        return TokenBucket(rate_per_min)

    # -- 初始化与热重载 ----------------------------------------------------
    def _build_runtimes(self) -> None:
        runtimes: Dict[str, _ModelRuntime] = {}
        for name, mc in self.cfg.models.items():
            if not mc.enabled:
                continue
            try:
                provider = build_provider(mc, self.pool)
            except LLMGatewayError as e:
                print(f"[Gateway] 跳过模型 {name}: {e}")
                continue
            runtimes[name] = _ModelRuntime(
                cfg=mc,
                provider=provider,
                breaker=CircuitBreaker(mc.fail_threshold, mc.recovery_sec),
                rpm_bucket=self._make_bucket(mc.rpm, f"gw:m:{name}:rpm"),
                tpm_bucket=self._make_bucket(mc.tpm, f"gw:m:{name}:tpm"),
            )
        with self._lock:
            self._runtimes = runtimes

    def _config_file(self) -> str:
        return getattr(self.cfg, "source_path", "") or ""

    def _remember_mtime(self) -> None:
        p = self._config_file()
        try:
            self._cfg_mtime = os.path.getmtime(p) if os.path.isfile(p) else 0.0
        except OSError:
            self._cfg_mtime = 0.0

    def maybe_reload(self) -> bool:
        """
        配置热重载：检测配置文件 mtime 变化则重建模型运行时。

        这就是「模型切换不用改代码」的落地方式——
        运维改一行 yaml，下一次调用自动生效，进程不用重启。
        熔断状态与统计数据会重置（因为模型集合可能已经变了）。
        """
        now = time.monotonic()
        if now - self._last_reload_check < float(self.cfg.reload_interval):
            return False
        self._last_reload_check = now
        p = self._config_file()
        if not p or not os.path.isfile(p):
            return False
        try:
            mtime = os.path.getmtime(p)
        except OSError:
            return False
        if mtime <= self._cfg_mtime:
            return False

        try:
            new_cfg = load_config(self._config_path or p)
        except Exception as e:
            print(f"[Gateway] 热重载失败，沿用旧配置: {e}")
            self._cfg_mtime = mtime
            return False

        with self._lock:
            self.cfg = new_cfg
            self._global_rpm = self._make_bucket(int(new_cfg.global_rpm or 0), "gw:global:rpm")
            self._global_tpm = self._make_bucket(int(new_cfg.global_tpm or 0), "gw:global:tpm")
        self._build_runtimes()
        self._cfg_mtime = mtime
        if self._verbose:
            print(f"[Gateway] 配置已热重载，当前模型: {list(self._runtimes)}")
        return True

    # -- 路由 --------------------------------------------------------------
    def resolve_chain(self, task: str) -> List[str]:
        """
        解析任务对应的模型调用链。

        路由策略非常直白，但这正是企业级要的东西：
            classify / grade / rewrite  -> 小模型（快、便宜，够用就行）
            generate / write / synthesize -> 大模型（质量优先）
        配错或模型不存在时自动过滤，最后兜底到任意可用模型，
        保证「路由表写错不至于让整个系统不可用」。
        """
        with self._lock:
            chain = [m for m in self.cfg.chain_for(task) if m in self._runtimes]
            if not chain:
                chain = list(self._runtimes)[:1]
            return chain

    # -- 准入检查：限流 + 熔断 ---------------------------------------------
    def _admit_global(self, est_tokens: int) -> Optional[str]:
        """
        全局配额检查——每个请求只做一次。

        为什么必须独立出来：如果放在按模型循环里，一次请求走完
        「主模型失败 -> 备选1失败 -> 备选2成功」的 fallback 链，
        就会扣掉 3 个全局令牌。结果是配了 120 RPM，实际只能跑 40 RPM，
        限流越用越严。全局配额衡量的是「进来多少请求」，
        与内部重试了几次无关。
        """
        timeout = float(self.cfg.acquire_timeout)
        if not self._global_rpm.acquire(1, timeout):
            return "全局 RPM 超限"
        if est_tokens > 0 and not self._global_tpm.acquire(est_tokens, timeout):
            return "全局 TPM 超限"
        return None

    def _admit_model(self, rt: _ModelRuntime, est_tokens: int) -> Optional[str]:
        """
        单模型准入：熔断 + 该模型自己的配额。

        顺序很重要：先查熔断（本地判断、零成本），再扣令牌，
        避免给一个已经熔断的模型白白消耗配额。
        """
        if not rt.breaker.allow():
            return f"熔断器打开({rt.cfg.name})"
        timeout = float(self.cfg.acquire_timeout)
        if not rt.rpm_bucket.acquire(1, timeout):
            return f"模型 RPM 超限({rt.cfg.name})"
        if est_tokens > 0 and not rt.tpm_bucket.acquire(est_tokens, timeout):
            return f"模型 TPM 超限({rt.cfg.name})"
        return None

    # -- 单模型调用（含同模型重试） -----------------------------------------
    def _call_one(self, rt: _ModelRuntime, system_prompt: str,
                  user_prompt: str) -> LLMResponse:
        last: Optional[Exception] = None
        attempts = max(1, int(self.cfg.max_retries) + 1)
        for i in range(attempts):
            try:
                resp = rt.provider.invoke(system_prompt, user_prompt)
                rt.breaker.on_success()
                return resp
            except Exception as e:
                last = e
                rt.breaker.on_failure()
                self.cost.record_failure(rt.cfg.name)
                if i < attempts - 1:
                    # 指数退避 + 抖动，避免重试风暴
                    time.sleep(float(self.cfg.retry_backoff) * (2 ** i)
                               * (0.8 + 0.4 * random.random()))
        raise ProviderError(f"{rt.cfg.name} 调用失败: {last}")

    # -- 统一出口：非流式 ---------------------------------------------------
    def chat(self, system_prompt: str, user_prompt: str,
             task: str = "default", user: str = "anonymous",
             **_ignored) -> str:
        """
        与原 `BaseLLM.chat` 完全同签名，可直接替换 OllamaLLM。

        `task` 是新增的可选参数，用来启用多模型路由；
        不传就走 default_chain，因此**老代码零改动**。
        `user` 把 token 用量归因到具体用户，支撑「查自己历史用量」。
        """
        return self.chat_detailed(system_prompt, user_prompt, task, user).text

    def chat_detailed(self, system_prompt: str, user_prompt: str,
                      task: str = "default",
                      user: str = "anonymous") -> LLMResponse:
        """和 chat 相同，但返回包含 token 用量与耗时的完整结果"""
        self.maybe_reload()
        chain = self.resolve_chain(task)
        if not chain:
            raise AllModelsFailed("没有任何可用模型，请检查配置")

        est = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt)
        errors: List[str] = []
        started = time.time()

        # 全局配额只扣一次，与后面试了几个模型无关
        gerr = self._admit_global(est)
        if gerr:
            raise RateLimitExceeded(gerr)

        for name in chain:
            with self._lock:
                rt = self._runtimes.get(name)
            if rt is None:
                continue

            reason = self._admit_model(rt, est)
            if reason:
                errors.append(reason)
                if self._verbose:
                    print(f"[Gateway] 跳过 {name}: {reason} -> 尝试下一个")
                continue

            try:
                resp = self._call_one(rt, system_prompt, user_prompt)
            except Exception as e:
                errors.append(f"{name}: {e}")
                if self._verbose:
                    print(f"[Gateway] {name} 失败: {e} -> fallback")
                continue

            # 用真实 token 数补扣配额：预扣的是估算值，这里补齐差额，
            # 保证 TPM 统计长期准确（多退少补，不阻塞本次请求）
            real = resp.total_tokens
            if real > est:
                self._global_tpm.try_acquire(real - est)
                rt.tpm_bucket.try_acquire(real - est)

            cost = self.cost.record(name, rt.cfg, resp)
            self.call_count += 1
            self.total_time += time.time() - started
            # 持久化用量（落盘或内存），支撑「按用户/按时间查历史」
            self._usage_store.record(
                user=user, model=name, provider=rt.cfg.provider, task=task,
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
                latency_s=resp.latency, cost_usd=cost)
            if self._verbose:
                print(f"[Gateway] ✓ {name} | user={user} task={task} | "
                      f"tokens: prompt={resp.prompt_tokens} "
                      f"completion={resp.completion_tokens} "
                      f"total={resp.total_tokens} | {resp.latency:.2f}s")
            return resp

        # 全链失败 —— 要么降级，要么抛异常，绝不静默返回空串
        if self.cfg.degraded_reply:
            if self._verbose:
                print(f"[Gateway] 全链失败，返回降级内容。原因: {errors}")
            return LLMResponse(text=str(self.cfg.degraded_reply),
                               model="<degraded>", provider="<none>")
        raise AllModelsFailed(
            f"任务 [{task}] 全部模型失败: " + " | ".join(errors))

    # -- 统一出口：流式 -----------------------------------------------------
    def stream_chat(self, system_prompt: str, user_prompt: str,
                    task: str = "default",
                    user: str = "anonymous") -> Iterator[str]:
        """
        流式输出。和非流式共用同一套路由 / 限流 / 熔断逻辑，
        真正做到「一个出口，按需切换」。

        注意：流式下 fallback 只在**首个 chunk 到达前**生效——
        一旦已经吐字给用户，再切模型会造成前后文风断裂，
        所以此时选择直接向上抛异常，由调用方决定怎么处理。
        """
        self.maybe_reload()
        chain = self.resolve_chain(task)
        est = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt)
        errors: List[str] = []

        gerr = self._admit_global(est)
        if gerr:
            raise RateLimitExceeded(gerr)

        for name in chain:
            with self._lock:
                rt = self._runtimes.get(name)
            if rt is None:
                continue
            reason = self._admit_model(rt, est)
            if reason:
                errors.append(reason)
                continue

            # 注意：started 是「是否已吐字」的布尔标志，不是时间戳。
            # 计时必须另用 t0，否则 time.time() - started 会把 True 当 1 秒减，
            # 结果把一个绝对时间戳写进 latency（曾经踩过这个坑）。
            started = False
            t0 = time.time()
            out_chars = 0
            try:
                for piece in rt.provider.stream(system_prompt, user_prompt):
                    started = True
                    out_chars += len(piece)
                    yield piece
            except Exception as e:
                rt.breaker.on_failure()
                self.cost.record_failure(name)
                if started:
                    # 已经吐过内容，不能再换模型重来
                    raise ProviderError(f"{name} 流式中断: {e}")
                errors.append(f"{name}: {e}")
                continue

            rt.breaker.on_success()
            # 流式下 Ollama 的 usage 分散在末帧，这里用估算值记账，
            # 保证成本统计不因为走流式就出现黑洞
            _comp_est = _estimate_tokens("x" * out_chars)
            cost = self.cost.record(name, rt.cfg, LLMResponse(
                text="", prompt_tokens=est,
                completion_tokens=_comp_est,
                model=rt.cfg.model, provider=rt.cfg.provider))
            self._usage_store.record(
                user=user, model=name, provider=rt.cfg.provider, task=task,
                prompt_tokens=est, completion_tokens=_comp_est,
                latency_s=time.time() - t0, cost_usd=cost)
            self.call_count += 1
            if self._verbose:
                print(f"[Gateway] ✓ {name} (stream) | user={user} task={task} | "
                      f"tokens(est): prompt={est} "
                      f"completion={_comp_est} total={est + _comp_est}")
            return

        if self.cfg.degraded_reply:
            yield str(self.cfg.degraded_reply)
            return
        raise AllModelsFailed(
            f"任务 [{task}] 流式全部失败: " + " | ".join(errors))

    # -- 可观测性 -----------------------------------------------------------
    def metrics(self) -> Dict[str, Any]:
        """一次性拿到所有运行指标，可直接挂到 /api/admin/llm_metrics"""
        with self._lock:
            models = {
                name: {
                    "provider": rt.cfg.provider,
                    "model": rt.cfg.model,
                    "tier": rt.cfg.tier,
                    "circuit": rt.breaker.snapshot(),
                    "rpm": rt.rpm_bucket.snapshot(),
                    "tpm": rt.tpm_bucket.snapshot(),
                }
                for name, rt in self._runtimes.items()
            }
            routing = dict(self.cfg.routing)
            default_chain = list(self.cfg.default_chain)
        return {
            "models": models,
            "routing": routing,
            "default_chain": default_chain,
            "pool": self.pool.stats(),
            "usage": self.cost.snapshot(),
            "usage_db": self.cfg.usage_db or "",
            "usage_persisted": bool(self.cfg.usage_db),
            "global_rpm": self._global_rpm.snapshot(),
            "global_tpm": self._global_tpm.snapshot(),
        }

    # -- 用量查询：让「用户查自己历史 token」成为可能 ----------------------
    def user_usage(self, user: str) -> Dict[str, Any]:
        """某用户累计用量（无则全 0）。user 默认 'anonymous'。"""
        return self._usage_store.user_usage(user)

    def usage_log(self, user: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        最近用量明细。

        - user=None（管理员视角）→ 全部用户
        - user='alice'          → 只该用户
        - limit                  → 返回条数上限
        """
        return self._usage_store.usage_log(user, limit)

    def usage_range(self, start_ts: float, end_ts: float,
                    user: str = None) -> List[Dict[str, Any]]:
        """某时间区间内的用量明细（如「这个月我烧了多少 token」）"""
        return self._usage_store.usage_range(start_ts, end_ts, user)

    def top_users(self, limit: int = 50) -> List[Dict[str, Any]]:
        """全用户 token 排行（管理后台看板用）"""
        return self._usage_store.top_users(limit)

    def health(self) -> Dict[str, str]:
        """各模型熔断状态速览：closed=健康 / open=已隔离"""
        with self._lock:
            return {n: rt.breaker.state for n, rt in self._runtimes.items()}

    def close(self) -> None:
        self.pool.close()


# ============================================================================
# 全局单例 —— 让任意模块都能拿到同一个网关（共享连接池与配额）
# ============================================================================
_gateway_singleton: Optional[LLMGateway] = None
_singleton_lock = threading.Lock()


def get_gateway(config_path: str = "", verbose: bool = False) -> LLMGateway:
    """
    获取全局网关单例。

    必须是单例：限流配额、熔断状态、连接池如果每个模块各持一份，
    就等于没有全局管控——这是很多「假网关」翻车的地方。
    """
    global _gateway_singleton
    if _gateway_singleton is None:
        with _singleton_lock:
            if _gateway_singleton is None:
                _gateway_singleton = LLMGateway(config_path, verbose=verbose)
    return _gateway_singleton


# ============================================================================
# 自检脚本：python llm_gateway.py
# ============================================================================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="LLM Gateway 自检")
    ap.add_argument("--config", default="", help="配置文件路径")
    ap.add_argument("--task", default="default", help="任务类型")
    ap.add_argument("--stream", action="store_true", help="使用流式输出")
    ap.add_argument("--prompt", default="用一句话说明什么是令牌桶限流。")
    args = ap.parse_args()

    gw = LLMGateway(args.config, verbose=True)
    sys_p = "你是一个简洁的技术助手，回答控制在 50 字以内。"

    print("\n--- 路由结果 ---")
    print(f"task={args.task} -> {gw.resolve_chain(args.task)}")

    print("\n--- 调用 ---")
    if args.stream:
        for piece in gw.stream_chat(sys_p, args.prompt, task=args.task):
            print(piece, end="", flush=True)
        print()
    else:
        r = gw.chat_detailed(sys_p, args.prompt, task=args.task)
        print(r.text)
        print(f"\n[tokens] in={r.prompt_tokens} out={r.completion_tokens} "
              f"latency={r.latency:.2f}s model={r.model}")

    print("\n--- 指标 ---")
    print(json.dumps(gw.metrics(), ensure_ascii=False, indent=2))
    gw.close()
