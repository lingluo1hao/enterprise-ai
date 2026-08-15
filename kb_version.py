# -*- coding: utf-8 -*-
"""
kb_version.py — 知识库版本号（Cache Versioning，方案乙核心）

为什么需要它：
  原 Redis 精确缓存用固定 TTL（曾 7 天），文档更新后相似/重复问题仍可能回旧答案。
  方案乙改用「版本号缓存键」：缓存 key 嵌入 kb_version，文档一 ingest/rebuild 成功，
  对该版本号 INCR → 旧 key 全体瞬间失联（O(1)，无需 SCAN/DELETE 海量 key）。
  再叠加一个短 TTL（30 分钟）作兜底，即使版本号漏 bump 最坏也只过时 30 分钟。

设计：
  - 零耦合：仅依赖 redis，不 import 项目内其他模块，避免循环依赖。
  - 单例连接：首次使用建连接，之后复用。
  - 全部异常 try/except 包裹：Redis 不可用也不影响主流程（降级为无版本号，靠 TTL 兜底）。

版本粒度：全局（rag:kbver:global）。原因：现有 CacheManager 缓存键只含 role、不含 tenant，
  且 ingest/rebuild 多为整库操作；全局版本最稳妥——任何文档变更即让所有缓存失效。
  如需按租户隔离，bump/get 传入对应 tenant 即可（key 为 rag:kbver:{tenant}）。
"""

import os
import threading
import time
from typing import Optional

REDIS_HOST = os.getenv("REDIS_HOST", "192.168.200.128")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "dev0619")
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# 版本号前缀键：rag:kbver:{tenant}
KBVER_KEY_PREFIX = "rag:kbver:"

# 读取侧内存缓存：避免热路径每次连 Redis。
#   _cache_value: 最近一次成功读到的版本（int）；None 表示尚未成功读过
#   _cache_fail_until: 失败后的冷却截止（epoch 秒）——Redis 黑洞/抖动时
#                      快速失败返回 None，不再每次重试拖死热路径
#                      （redis-py 8 默认 5s connect_timeout + 内置重试，
#                       失败一次可能阻塞数十秒，必须冷却）。
_cache_lock = threading.Lock()
_cache_value: Optional[int] = None
_cache_fail_until: float = 0.0
_CACHE_FAIL_COOLDOWN = 5.0  # 失败冷却秒数（期间快速失败，之后重试一次）

_conn = None


def _redis():
    """懒加载、复用的 Redis 连接。"""
    global _conn
    if _conn is None:
        import redis as redis_pkg
        _conn = redis_pkg.Redis(
            host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, db=REDIS_DB,
            socket_connect_timeout=5, socket_timeout=5, decode_responses=True,
        )
    return _conn


def get_kb_version(tenant: str = "global") -> Optional[int]:
    """读取当前知识库版本号。

    返回 Optional[int]：
      - int      >= 0：读成功。0 = Redis 里尚无版本键（全新部署），是合法初始版本。
      - None          ：读失败（Redis 不可用 / 连接异常 / 失败冷却期内）。

    调用方应区分对待：缓存键场景可用 `or 0` 兜底（继续走 TTL 降级）；
    自进化时效校验场景应 fail-open（见 evolution.py）。

    P1-R1：读取侧加内存缓存 + 失败冷却。redis-py 8 默认 5s connect_timeout
    且内置重试，Redis 不可达时一次调用可能阻塞数十秒——若在检索热路径上
    每次实时检索都调这里，会把问答拖死。缓存避免重复连接；失败冷却期内
    快速返回 None（fail-open 语义），冷却后重试一次，兼顾新鲜度与可用性。

    注意：无论传入 tenant 是什么，一律读全局键 rag:kbver:global。
    原因：bump 端（pipeline/rebuild）只用全局粒度写入，若读取端按各自
    tenant 去读 rag:kbver:{tenant}，会读不到 → 永远返回 None/0 → 缓存 key 卡在
    v0 前缀 → bump 失效、旧答案不随 ingest 刷新（仅 30min TTL 兜底）。
    全局单版本最稳妥（见模块 docstring），故强制对齐到 global。
    """
    global _cache_value, _cache_fail_until
    now = time.time()
    with _cache_lock:
        # 失败冷却期内：不碰 Redis，快速失败
        if now < _cache_fail_until:
            return None
    try:
        v = _redis().get(f"{KBVER_KEY_PREFIX}global")
        result = int(v) if v is not None else 0
        with _cache_lock:
            _cache_value = result
            _cache_fail_until = 0.0  # 成功后清冷却
        return result
    except Exception:
        with _cache_lock:
            _cache_fail_until = time.time() + _CACHE_FAIL_COOLDOWN
        return None


def bump_kb_version(tenant: str = "global") -> int:
    """文档 ingest/rebuild 成功末尾调用：版本号 +1，使旧缓存 key 全体失效。

    返回 bump 后的版本号（失败返回 0）。
    强制写入全局键 rag:kbver:global，与 get_kb_version 对齐。
    """
    try:
        new_v = _redis().incr(f"{KBVER_KEY_PREFIX}global")
        print(f"[kb_version] ✔ 知识库版本 +1 -> v{new_v} (tenant={tenant})")
        return new_v
    except Exception as e:
        print(f"[kb_version] ⚠ bump 失败(忽略,靠 TTL 兜底): {e}")
        return 0
