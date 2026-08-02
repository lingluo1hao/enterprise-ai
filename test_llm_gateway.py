# -*- coding: utf-8 -*-
"""
LLM Gateway 端到端验证脚本
================================================================================
连真实 Ollama (192.168.200.128:11434) 逐项验证网关的企业级能力：

    1. 真实调用 + 真实 token 计数（不是估算）
    2. 连接池复用（keep-alive 命中率）
    3. 流式输出与非流式共用同一出口
    4. 令牌桶限流真的会拦截
    5. 熔断器 OPEN -> HALF_OPEN -> CLOSED 状态流转
    6. 主模型不可用时自动 fallback
    7. 全链失败时降级返回，而不是抛 500
    8. 多模型路由按任务类型分流
    9. 配置热重载

运行：python test_llm_gateway.py
"""
import os
import sys
import time
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import llm_gateway as G

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = ""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def section(title: str):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


# ---------------------------------------------------------------------------
# 1. 真实调用 + token 计数
# ---------------------------------------------------------------------------
def test_real_call(gw):
    section("1. 真实调用 + 真实 token 计数")
    r = gw.chat_detailed("你是简洁助手，20字内作答。", "1+1等于几？", task="classify")
    print(f"  回复: {r.text.strip()[:60]}")
    print(f"  模型: {r.model} | 耗时: {r.latency:.2f}s")
    print(f"  token: prompt={r.prompt_tokens} completion={r.completion_tokens} "
          f"total={r.total_tokens}")
    check("能拿到非空回复", bool(r.text.strip()))
    check("prompt_tokens 为真实值(>0)", r.prompt_tokens > 0, f"={r.prompt_tokens}")
    check("completion_tokens 为真实值(>0)", r.completion_tokens > 0,
          f"={r.completion_tokens}")
    return r


# ---------------------------------------------------------------------------
# 2. 连接池复用
# ---------------------------------------------------------------------------
def test_pool_reuse(gw):
    section("2. 连接池复用（改造前每次都新建连接）")
    before = gw.pool.stats()
    for i in range(3):
        gw.chat("你是简洁助手，10字内作答。", f"第{i+1}次：说个数字", task="classify")
    after = gw.pool.stats()
    print(f"  调用前: created={before['created']} reused={before['reused']}")
    print(f"  调用后: created={after['created']} reused={after['reused']}")
    delta_new = after["created"] - before["created"]
    delta_reuse = after["reused"] - before["reused"]
    print(f"  本轮 3 次调用: 新建 {delta_new} 条, 复用 {delta_reuse} 次")
    check("连接被复用而非每次新建", delta_reuse >= 2,
          f"复用{delta_reuse}次 / 新建{delta_new}条")


# ---------------------------------------------------------------------------
# 3. 流式输出
# ---------------------------------------------------------------------------
def test_streaming(gw):
    section("3. 流式输出（与非流式统一出口）")
    pieces, first_at = [], None
    t0 = time.time()
    for p in gw.stream_chat("你是简洁助手。", "从1数到5，只输出数字", task="classify"):
        if first_at is None:
            first_at = time.time() - t0
        pieces.append(p)
    total = time.time() - t0
    text = "".join(pieces)
    print(f"  首字节延迟: {first_at:.2f}s | 总耗时: {total:.2f}s | 分片数: {len(pieces)}")
    print(f"  内容: {text.strip()[:60]}")
    check("流式返回多个分片", len(pieces) > 1, f"{len(pieces)} 片")
    check("首字节早于总耗时（真流式）", first_at is not None and first_at < total)


# ---------------------------------------------------------------------------
# 4. 令牌桶限流
# ---------------------------------------------------------------------------
def test_rate_limit():
    section("4. 令牌桶限流（RPM / TPM 双维度）")
    tb = G.TokenBucket(rate_per_min=60)   # 每秒补 1 个，容量 60
    got = sum(1 for _ in range(60) if tb.try_acquire())
    blocked = sum(1 for _ in range(10) if not tb.try_acquire())
    print(f"  容量 60：连续取 60 次成功 {got} 次；再取 10 次被拒 {blocked} 次")
    check("令牌耗尽后正确拒绝", got == 60 and blocked == 10)

    time.sleep(1.1)
    check("按时间自动补充令牌", tb.try_acquire(), "等待1.1s后重新获取成功")

    tb2 = G.TokenBucket(rate_per_min=60)
    for _ in range(60):
        tb2.try_acquire()
    t0 = time.time()
    ok = tb2.acquire(1, timeout=2.0)
    print(f"  阻塞获取: 成功={ok} 等待={time.time() - t0:.2f}s")
    check("阻塞等待能拿到令牌而非直接失败", ok)

    tb3 = G.TokenBucket(rate_per_min=0)
    check("rate=0 表示不限流", tb3.try_acquire(999))


# ---------------------------------------------------------------------------
# 5. 熔断器状态机
# ---------------------------------------------------------------------------
def test_circuit_breaker():
    section("5. 熔断器 CLOSED -> OPEN -> HALF_OPEN -> CLOSED")
    cb = G.CircuitBreaker(fail_threshold=3, recovery_sec=1.0)
    check("初始为 CLOSED", cb.state == cb.CLOSED, cb.state)

    for _ in range(2):
        cb.on_failure()
    check("失败2次(未达阈值)仍 CLOSED", cb.state == cb.CLOSED, cb.state)

    cb.on_failure()
    check("失败3次达阈值 -> OPEN", cb.state == cb.OPEN, cb.state)
    check("OPEN 状态拒绝放行", not cb.allow())

    time.sleep(1.05)
    check("恢复期后 -> HALF_OPEN", cb.state == cb.HALF_OPEN, cb.state)
    check("HALF_OPEN 放行第1个探测", cb.allow())
    check("HALF_OPEN 拒绝第2个请求", not cb.allow())

    cb.on_success()
    check("探测成功 -> CLOSED", cb.state == cb.CLOSED, cb.state)

    cb2 = G.CircuitBreaker(fail_threshold=1, recovery_sec=0.5)
    cb2.on_failure()
    time.sleep(0.55)
    cb2.allow()
    cb2.on_failure()
    check("HALF_OPEN 探测失败 -> 重新 OPEN", cb2.state == cb2.OPEN, cb2.state)


# ---------------------------------------------------------------------------
# 6. Fallback：主模型不可用时自动切备选
# ---------------------------------------------------------------------------
def test_fallback():
    section("6. 熔断降级：主模型不可用自动 fallback 到备选")
    cfg_path = os.path.join(tempfile.gettempdir(),
                            "_test_fallback.json")
    ollama = os.getenv("OLLAMA_URL", "http://192.168.200.128:11434")
    cfg = {
        "models": {
            # 故意指向一个不存在的端口，模拟主模型宕机
            "broken-primary": {
                "provider": "ollama", "model": "qwen2:7b",
                "base_url": "http://127.0.0.1:9", "timeout": 3.0,
                "fail_threshold": 2, "recovery_sec": 5.0,
            },
            "healthy-backup": {
                "provider": "ollama", "model": "qwen2:7b",
                "base_url": ollama, "timeout": 120.0,
            },
        },
        "routing": {"generate": ["broken-primary", "healthy-backup"]},
        "default_chain": ["broken-primary", "healthy-backup"],
        "max_retries": 0,
        "acquire_timeout": 1.0,
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False)

    gw = G.LLMGateway(cfg_path, verbose=False)
    try:
        print(f"  路由链: {gw.resolve_chain('generate')}")
        t0 = time.time()
        r = gw.chat_detailed("你是简洁助手，10字内作答。", "说个成语", task="generate")
        print(f"  实际应答模型: {r.model} (provider={r.provider}) "
              f"耗时 {time.time() - t0:.2f}s")
        print(f"  回复: {r.text.strip()[:40]}")
        check("主模型失败后由备选模型成功应答", bool(r.text.strip()))

        usage = gw.cost.snapshot()["per_model"]
        broken = usage.get("broken-primary", {})
        backup = usage.get("healthy-backup", {})
        print(f"  broken-primary: 失败 {broken.get('failures', 0)} 次, "
              f"成功 {broken.get('calls', 0)} 次")
        print(f"  healthy-backup: 成功 {backup.get('calls', 0)} 次")
        check("失败被正确计数到主模型", broken.get("failures", 0) >= 1)
        check("成功被正确计数到备选模型", backup.get("calls", 0) >= 1)

        # 连续失败后主模型应被熔断隔离
        for _ in range(3):
            try:
                gw.chat("x", "y", task="generate")
            except Exception:
                pass
        health = gw.health()
        print(f"  熔断状态: {health}")
        check("主模型被熔断隔离(open/half_open)",
              health.get("broken-primary") in ("open", "half_open"),
              health.get("broken-primary", "?"))
    finally:
        gw.close()
        try:
            os.remove(cfg_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 7. 全链失败 -> 降级返回
# ---------------------------------------------------------------------------
def test_degraded():
    section("7. 全链失败 -> 返回降级文案（而不是 500）")
    cfg_path = os.path.join(tempfile.gettempdir(),
                            "_test_degraded.json")
    cfg = {
        "models": {
            "dead-a": {"provider": "ollama", "model": "m",
                       "base_url": "http://127.0.0.1:9", "timeout": 2.0},
            "dead-b": {"provider": "ollama", "model": "m",
                       "base_url": "http://127.0.0.1:10", "timeout": 2.0},
        },
        "default_chain": ["dead-a", "dead-b"],
        "max_retries": 0,
        "acquire_timeout": 0.5,
        "degraded_reply": "服务繁忙，请稍后重试。",
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False)
    gw = G.LLMGateway(cfg_path, verbose=False)
    try:
        out = gw.chat("s", "u")
        print(f"  返回: {out}")
        check("全挂时返回降级文案", out == "服务繁忙，请稍后重试。")
    finally:
        gw.close()

    # 不配 degraded_reply 时应显式抛异常，不能静默返回空串
    cfg["degraded_reply"] = ""
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False)
    gw2 = G.LLMGateway(cfg_path, verbose=False)
    try:
        gw2.chat("s", "u")
        check("未配降级文案时抛 AllModelsFailed", False, "竟然没抛异常")
    except G.AllModelsFailed as e:
        check("未配降级文案时抛 AllModelsFailed", True, str(e)[:50])
    except Exception as e:
        check("未配降级文案时抛 AllModelsFailed", False, f"抛了{type(e).__name__}")
    finally:
        gw2.close()
        try:
            os.remove(cfg_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 8. 多模型路由
# ---------------------------------------------------------------------------
def test_routing(gw):
    section("8. 多模型路由（任务类型 -> 模型链）")
    for task in ["classify", "grade", "generate", "write", "plan", "没配过的任务"]:
        print(f"  {task:<14} -> {gw.resolve_chain(task)}")
    check("轻任务与重任务都能解析出可用链",
          all(gw.resolve_chain(t) for t in ["classify", "generate", "xxx"]))
    check("未配置任务回退到默认链", gw.resolve_chain("xxx") == gw.resolve_chain("default"))


# ---------------------------------------------------------------------------
# 9. 配置热重载
# ---------------------------------------------------------------------------
def test_hot_reload():
    section("9. 配置热重载（改配置不重启）")
    cfg_path = os.path.join(tempfile.gettempdir(),
                            "_test_reload.json")
    ollama = os.getenv("OLLAMA_URL", "http://192.168.200.128:11434")
    base = {
        "models": {"m1": {"provider": "ollama", "model": "qwen2:7b",
                          "base_url": ollama}},
        "default_chain": ["m1"],
        "reload_interval": 0.0,
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(base, f)
    gw = G.LLMGateway(cfg_path, verbose=False)
    try:
        print(f"  重载前模型: {list(gw._runtimes)}")
        check("初始只有 m1", list(gw._runtimes) == ["m1"])

        time.sleep(1.05)  # 保证 mtime 变化能被文件系统感知
        base["models"]["m2"] = {"provider": "ollama", "model": "qwen2:1.5b",
                                "base_url": ollama}
        base["default_chain"] = ["m2", "m1"]
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(base, f)

        changed = gw.maybe_reload()
        print(f"  重载触发: {changed} | 重载后模型: {list(gw._runtimes)}")
        print(f"  新的默认链: {gw.cfg.default_chain}")
        check("检测到配置变更并重载", changed)
        check("新模型已生效", "m2" in gw._runtimes)
        check("路由链同步更新", gw.resolve_chain("default")[0] == "m2")
    finally:
        gw.close()
        try:
            os.remove(cfg_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 10. 全局配额只扣一次（fallback 不应重复扣全局令牌）
# ---------------------------------------------------------------------------
def test_global_quota_once():
    section("10. 全局配额每请求只扣一次（fallback 不重复扣）")
    cfg_path = os.path.join(tempfile.gettempdir(), "_test_quota.json")
    ollama = os.getenv("OLLAMA_URL", "http://192.168.200.128:11434")
    cfg = {
        "models": {
            "dead-1": {"provider": "ollama", "model": "m",
                       "base_url": "http://127.0.0.1:9", "timeout": 2.0},
            "dead-2": {"provider": "ollama", "model": "m",
                       "base_url": "http://127.0.0.1:10", "timeout": 2.0},
            "alive": {"provider": "ollama", "model": "qwen2:7b",
                      "base_url": ollama, "timeout": 120.0},
        },
        "default_chain": ["dead-1", "dead-2", "alive"],
        "global_rpm": 60,
        "max_retries": 0,
        "acquire_timeout": 1.0,
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    gw = G.LLMGateway(cfg_path, verbose=False)
    try:
        rate = cfg["global_rpm"] / 60.0          # 每秒回填量
        # 先把桶压到半满，避免回填触顶被截断，导致净扣减失真
        gw._global_rpm.acquire(cfg["global_rpm"] // 2, 0)

        before = gw._global_rpm.snapshot()["tokens"]
        t0 = time.time()
        gw.chat("你是简洁助手，5字内作答。", "说个字")
        elapsed = time.time() - t0
        after = gw._global_rpm.snapshot()["tokens"]

        net = before - after                     # 净变化（已被回填抵消一部分）
        refill = elapsed * rate                  # 期间自动回填量
        gross = net + refill                     # 还原出真实扣减
        print(f"  链路: {gw.resolve_chain('default')}（前两个必失败）")
        print(f"  全局令牌: {before:.2f} -> {after:.2f}，耗时 {elapsed:.2f}s")
        print(f"  净变化 {net:.2f} + 期间回填 {refill:.2f} = 真实扣减 {gross:.2f}")
        # 修复前每个模型都扣一次，3 模型链会扣满 3 个
        check("走完3个模型只扣1个全局令牌", 0.5 <= gross <= 1.5,
              f"真实扣减 {gross:.2f}（修复前应为 3.00）")
    finally:
        gw.close()
        try:
            os.remove(cfg_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
def test_usage_persistence():
    """Token 用量持久化 + 按用户查询（不依赖 Ollama，纯本地 SQLite 验证）"""
    section("11. Token 用量持久化 + 按用户查询")
    tmp = tempfile.gettempdir()
    db = os.path.join(tmp, f"_test_usage_{int(time.time() * 1000)}.db")
    if os.path.exists(db):
        os.remove(db)
    try:
        # 11.1 落盘 + 重启可查 + 按用户隔离
        store = G.UsageStore(db)
        store.record("alice", "qwen2.5:1.5b", "ollama", "grade", 10, 20, 0.3, 0.0001)
        store.record("alice", "qwen2:7b", "ollama", "generate", 30, 50, 0.8, 0.0005)
        store.record("bob", "qwen2:7b", "ollama", "generate", 25, 40, 0.7, 0.0004)
        ua = store.user_usage("alice")
        check("按用户聚合(alice=2次/110tok)",
              ua["calls"] == 2 and ua["total_tokens"] == 110, str(ua))
        ub = store.user_usage("bob")
        check("按用户隔离(bob 不含 alice)", ub["calls"] == 1, str(ub))
        store2 = G.UsageStore(db)  # 模拟进程重启
        check("重启后仍可查历史", store2.user_usage("alice")["calls"] == 2)

        # 11.2 明细 + 时间区间
        now = time.time()
        store.record("carol", "qwen2:7b", "ollama", "generate", 5, 5, 0.1, 0.0001)
        log = store.usage_log(user="alice")
        check("usage_log 返回用户明细", len(log) == 2 and "total_tokens" in log[0],
              f"len={len(log)}")
        rng = store.usage_range(now - 1, now + 1)
        check("usage_range 按时间筛选", len(rng) >= 1, f"hits={len(rng)}")

        # 11.3 内存模式（未配置 usage_db）也能聚合但不落盘
        mem = G.UsageStore("")
        mem.record("x", "m", "p", "t", 1, 2, 0.1, 0.0)
        check("内存模式可聚合", mem.user_usage("x")["calls"] == 1)

        # 11.4 Gateway 整合：chat 写入 user
        gw = G.LLMGateway(config_path="", verbose=False)
        gw._usage_store = G.UsageStore(db)

        def fake_call(rt, sp, up):
            return G.LLMResponse(text="ok", prompt_tokens=12, completion_tokens=8,
                                 model=rt.cfg.model, provider=rt.cfg.provider,
                                 latency=0.2)

        gw._call_one = fake_call
        try:
            gw.chat_detailed("你是助手", "hi", task="generate", user="dave")
            ud = gw.user_usage("dave")
            check("gateway.chat_detailed 写入 user",
                  ud["calls"] == 1 and ud["total_tokens"] == 20, str(ud))
        finally:
            gw.close()

        # 11.5 配置 usage_db 后指标标记持久化
        gw2 = G.LLMGateway(config_path="", verbose=False)
        gw2.cfg.usage_db = db
        gw2._usage_store = G.UsageStore(db)
        check("配置 usage_db 后 usage_persisted=True",
              gw2.metrics()["usage_persisted"] is True)
        gw2.close()

        # 11.6 全用户排行（后台看板数据源）
        tops = store.top_users(10)
        names = [t["user"] for t in tops]
        check("top_users 按 token 降序",
              names and tops == sorted(tops, key=lambda x: x["total_tokens"],
                                       reverse=True),
              str(names))
        alice_row = next((t for t in tops if t["user"] == "alice"), None)
        check("top_users 聚合值正确(alice=2次/110tok)",
              alice_row is not None and alice_row["calls"] == 2
              and alice_row["total_tokens"] == 110, str(alice_row))

        # 11.7 latency 防御：把时间戳误当耗时传进来，必须被拦成 0
        store.record("eve", "m", "p", "t", 1, 1, time.time(), 0.0)
        eve_log = store.usage_log("eve", 1)
        check("异常 latency 被清零（防时间戳污染）",
              eve_log and eve_log[0]["latency_s"] == 0.0, str(eve_log))
    finally:
        try:
            os.remove(db)
        except OSError:
            pass


# ---------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("LLM Gateway 端到端验证")
    print("=" * 74)

    gw = G.LLMGateway(verbose=True)
    try:
        test_real_call(gw)
        test_pool_reuse(gw)
        test_streaming(gw)
        test_rate_limit()
        test_circuit_breaker()
        test_fallback()
        test_degraded()
        test_routing(gw)
        test_hot_reload()
        test_global_quota_once()
        test_usage_persistence()

        section("最终指标快照")
        m = gw.metrics()
        print(json.dumps({"pool": m["pool"], "usage": m["usage"],
                          "health": gw.health()},
                         ensure_ascii=False, indent=2))
    finally:
        gw.close()

    section("汇总")
    print(f"  通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        for f in FAIL:
            print(f"    FAILED: {f}")
        sys.exit(1)
    print("  全部通过")


if __name__ == "__main__":
    main()
