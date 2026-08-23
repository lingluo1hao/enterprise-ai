# =============================================================================
# 意图识别组件测试（v2 语义路由范式）
# -----------------------------------------------------------------------------
# 覆盖：
#   - 余弦相似度 / L2 JSON 解析工具
#   - L1 语义路由（确定性 fake embedder，无需 Ollama）：case #17 与「心跳间隔是多少」
#     字面都含「是多少」，但向量空间分离，一判即分（证明「词典救不了、语义能救」）
#   - 歧义 → L2 LLM 兜底
#   - 缓存命中
#   - embedder 离线 → lexical 降级 / fallback
#   - node_classify 集成（线上线下统一，不依赖 fast_mode 决定走哪条路）
#   - P3a：candidates 归因（top3 排序透传、缓存命中也带）
#   - P3b：低置信（<0.5 且非 lexical）进 bad case；lexical 降级不刷屏
#   - case #17 回归（_quick_classify 仍是可靠 lexical 兜底）
# =============================================================================

import os
import sys

# 把项目根目录加入 path，保证 import intent_classifier / langgraph_rag_agent 成功
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intent_classifier import IntentClassifier, IntentResult, _cosine  # noqa: E402


class FakeEmbedder:
    """测试用确定性 embedder：按关键词把文本映射到 2D 簇心向量。"""
    def embed_query(self, text):
        t = (text or "").lower()
        if text == "ambiguous_probe":
            return [0.0, 0.0]  # 与所有 centroid 余弦=0，强制歧义 → L2
        if any(w in t for w in ["你好", "在吗", "多大了", "天气", "几岁", "谢谢", "你是谁", "最近"]):
            return [1.0, 0.0]
        if any(w in t for w in ["写诗", "股票", "外卖", "爬虫"]):
            return [0.0, 1.0]
        if any(w in t for w in ["怎么弄", "什么意思", "有什么用", "怎么设置"]):
            return [0.0, -1.0]
        if any(w in t for w in ["不对", "很好", "说错了", "不准确"]):
            return [-1.0, -1.0]
        if any(w in t for w in ["续航", "有哪些", "怎么配置", "升级"]):
            return [1.0, 1.0]
        if any(w in t for w in ["区别", "对比", "差在哪"]):
            return [1.0, -1.0]
        return [-1.0, 0.0]  # simple / 默认


class _Results:
    def __init__(self):
        self.calls = []

    def __call__(self, query, intent_list):
        self.calls.append((query, intent_list))
        return '{"intent": "simple", "confidence": 0.7}'


class BrokenEmbedder:
    """模拟 embedder 离线：embed_query 永远返回 None。"""
    def embed_query(self, text):
        return None


class FakeClassifier:
    """测试用：忽略 query，返回预设 IntentResult（验证 node_classify 的 P3b 分支）。"""
    def __init__(self, result):
        self.result = result
    def classify(self, query, **kwargs):
        return self.result


def test_cosine():
    assert abs(_cosine([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9
    assert abs(_cosine([1.0, 0.0], [-1.0, 0.0]) + 1.0) < 1e-9
    assert _cosine([1.0, 0.0], [0.0, 0.0]) == 0.0
    assert _cosine([], []) == 0.0


def test_parse_llm_intent():
    assert IntentClassifier._parse_llm_intent('{"intent":"simple"}') == "simple"
    assert IntentClassifier._parse_llm_intent('```json\n{"intent":"complex"}\n```') == "complex"
    assert IntentClassifier._parse_llm_intent('{"type":"chitchat"}') == "chitchat"
    assert IntentClassifier._parse_llm_intent("no json here") is None
    assert IntentClassifier._parse_llm_intent(None) is None


def test_semantic_routing_separation():
    clf = IntentClassifier(embedder=FakeEmbedder())
    # case #17：字面含「是多少」，但语义离 chitchat 近
    r1 = clf.classify("你今年多大了")
    assert r1.intent == "chitchat", r1
    assert r1.source == "semantic", r1
    # 真正的领域问题：语义离 simple 近
    r2 = clf.classify("心跳间隔是多少")
    assert r2.intent == "simple", r2
    assert r2.source == "semantic", r2
    # comparison 独立意图
    r3 = clf.classify("A 和 B 的定位方式有什么区别")
    assert r3.intent == "comparison", r3
    # complex 独立意图
    r4 = clf.classify("定位方式有哪些，各自精度如何，续航怎样")
    assert r4.intent == "complex", r4


def test_ambiguity_triggers_l2():
    clf = IntentClassifier(embedder=FakeEmbedder())
    res_calls = _Results()
    r = clf.classify("ambiguous_probe", llm_fn=res_calls)
    assert r.intent == "simple", r
    assert r.source == "semantic:ambiguous->llm", r
    assert len(res_calls.calls) == 1


def test_cache_hit():
    clf = IntentClassifier(embedder=FakeEmbedder())
    r1 = clf.classify("你今年多大了")
    r2 = clf.classify("你今年多大了")
    assert r2.source == "cache", r2
    assert r1.intent == r2.intent


def test_lexical_fallback_when_embedder_down():
    # embedder 离线（返回 None）→ centroid 构建失败 → 降级 lexical
    clf = IntentClassifier(embedder=BrokenEmbedder())
    r = clf.classify("你今年多大了", lexical_fn=lambda q: "chitchat")
    assert r.intent == "chitchat", r
    assert r.source == "lexical", r


def test_fallback_default_when_no_lexical():
    clf = IntentClassifier(embedder=None)
    r = clf.classify("anything", lexical_fn=None)
    assert r.intent == clf._default_intent, r
    assert r.source == "fallback:default", r


def test_integration_node_classify():
    """node_classify 委托 IntentClassifier，线上线下统一（fast_mode=True 也走语义）。"""
    import langgraph_rag_agent as ra

    app = object.__new__(ra.LangGraphRAGApp)
    app.fast_mode = True
    app.playbook_store = None
    app.intent_classifier = IntentClassifier()  # 真实组件；Ollama 不可达则自动 lexical
    app._format_history = lambda messages, max_turns=4: ""

    # case #17：闲聊（语义或 lexical 兜底都应得 chitchat）
    out = app.node_classify({"query": "你今年多大了", "messages": []})
    assert out["query_type"] == "chitchat", out
    assert "classify_source" in out and "classify_confidence" in out

    # 领域问题 → simple
    out = app.node_classify({"query": "心跳间隔是多少", "messages": []})
    assert out["query_type"] == "simple", out

    # 多问句 → complex（节点级确定性规则，先于组件）
    out = app.node_classify({"query": "心跳是多少？波特率是？", "messages": []})
    assert out["query_type"] == "complex", out
    assert out["classify_source"] == "rule:multi_question", out

    # 若 Ollama 可达，证明 case #17 走的是「语义」而非 lexical 规则
    # （用 use_cache=False 直测分类器，绕开 node_classify 的缓存命中）
    if app.intent_classifier._ready:
        r = app.intent_classifier.classify("你今年多大了", use_cache=False,
                                           lexical_fn=app._quick_classify,
                                           llm_fn=app._llm_classify_json)
        assert r.intent == "chitchat", r
        assert r.source == "semantic", r
    else:
        print("[warn] Ollama 不可达，node_classify 走 lexical 降级（结论仍正确）")


def test_quick_classify_case17_regression():
    """_quick_classify 仍是可靠的 lexical 兜底（早期修复的回归保障）。"""
    import langgraph_rag_agent as ra
    app = object.__new__(ra.LangGraphRAGApp)
    assert app._quick_classify("你今年多大了") == "chitchat"
    assert app._quick_classify("你几岁") == "chitchat"
    assert app._quick_classify("心跳间隔是多少") == "simple"
    assert app._quick_classify("心跳是多少？波特率是？") == "complex"


# ============================================================================
# P3a：candidates 归因
# ============================================================================

def test_candidates_attribution():
    """语义路由应返回 top3 候选（降序），且 top1 与判定意图一致。"""
    clf = IntentClassifier(embedder=FakeEmbedder())
    r = clf.classify("你今年多大了", use_cache=False)
    assert isinstance(r.candidates, list) and len(r.candidates) >= 2, r
    scores = [s for _, s in r.candidates]
    assert scores == sorted(scores, reverse=True), r  # 降序
    assert r.candidates[0][0] == r.intent              # top1 == 判定意图
    assert 0.0 <= r.candidates[0][1] <= 1.0            # 置信在合法区间


def test_cache_carries_candidates():
    """缓存命中也应携带 candidates（不能因为走 cache 就丢归因）。"""
    clf = IntentClassifier(embedder=FakeEmbedder())
    r1 = clf.classify("你今年多大了", use_cache=True)  # 首次：计算并写缓存
    r2 = clf.classify("你今年多大了", use_cache=True)  # 二次：命中缓存
    assert r2.source == "cache", r2
    assert r2.candidates == r1.candidates, (r1, r2)


# ============================================================================
# P3b：低置信意图进 bad case 库（仅语义路由在线但没把握时）
# ============================================================================

def _build_app_with_ms(fake_clf_result=None):
    """构造最小 node_classify 运行环境（fast_mode=True，与 web 服务一致）。"""
    import langgraph_rag_agent as ra
    app = object.__new__(ra.LangGraphRAGApp)
    app.fast_mode = True
    app.playbook_store = None
    app.username = "test"
    app._format_history = lambda messages, max_turns=4: ""
    rows = []
    class _FakeMS:
        def add_bad_case(self, query, source, root_cause=None, diagnosis=None,
                         suite=None, expected=None):
            rows.append((query, source, root_cause, suite))
            return len(rows)
    app.memory_store = _FakeMS()
    if fake_clf_result is None:
        app.intent_classifier = IntentClassifier(embedder=FakeEmbedder())
    else:
        app.intent_classifier = FakeClassifier(fake_clf_result)
    return app, rows


def test_low_confidence_triggers_badcase():
    """语义路由置信 <0.5 → 进 bad case（source=intent_lowconf, root_cause=来源）。"""
    from intent_classifier import IntentResult
    res = IntentResult("comparison", 0.42, "semantic",
                       candidates=[("comparison", 0.42), ("simple", 0.40), ("chitchat", 0.30)])
    app, rows = _build_app_with_ms(res)
    out = app.node_classify({"query": "这个和那个比哪个好", "messages": []})
    assert out["query_type"] == "comparison", out
    assert len(rows) == 1, rows
    assert rows[0][1] == "intent_lowconf", rows
    assert rows[0][2] == "semantic", rows
    assert rows[0][3] == "intent", rows


def test_lexical_lowconf_no_badcase():
    """embedder 离线 → 走 lexical:fallback 低置信，但不应刷屏 bad case。"""
    app, rows = _build_app_with_ms()        # intent_classifier=None → 走 lexical 降级
    app.intent_classifier = None
    out = app.node_classify({"query": "今天心情怎么样啊", "messages": []})
    assert out["classify_source"] == "lexical:fallback", out
    assert len(rows) == 0, "lexical 降级不应入库 bad case（避免刷屏）"


# ============================================================================
# CR 修复验证：centroid 构建失败冷却 + 消解归因一致性
# ============================================================================

class CountingBrokenEmbedder:
    """每次 embed_query 都失败并计数——验证失败后进入冷却、不重复打 HTTP。"""
    def __init__(self):
        self.calls = 0
    def embed_query(self, text):
        self.calls += 1
        return None


def test_centroid_fail_cooldown():
    """Ollama 离线：首次构建失败进冷却，冷却期内 classify 不再触发 embed 重试。"""
    emb = CountingBrokenEmbedder()
    clf = IntentClassifier(embedder=emb)
    clf._centroid_fail_cooldown = 60.0
    r1 = clf.classify("你今年多大了", lexical_fn=lambda q: "chitchat")
    assert r1.source == "lexical", r1
    first_calls = emb.calls
    assert first_calls > 0, "首次应至少尝试过一次 embed"
    r2 = clf.classify("心跳间隔是多少", use_cache=False,
                      lexical_fn=lambda q: "simple")
    assert r2.source == "lexical", r2
    assert emb.calls == first_calls, (
        f"冷却期内不应重试 embed：{first_calls} -> {emb.calls}")


def _build_app_for_resolve(clf_result, llm_json):
    """构造非 fast_mode + 有历史的 node_classify 环境（验证消解归因一致性）。"""
    from types import SimpleNamespace
    import langgraph_rag_agent as ra
    app = object.__new__(ra.LangGraphRAGApp)
    app.fast_mode = False
    app.playbook_store = None
    app.username = "test"
    app.intent_classifier = FakeClassifier(clf_result)
    app._format_history = lambda messages, max_turns=4: "历史上下文"
    app.pm = SimpleNamespace(
        get_prompt=lambda name: {"system": "s", "user_template": "u"},
        format_user_message=lambda t, **kw: "U")
    # 与 LLMGateway.chat 同签名：system_prompt / user_prompt / task / user
    app.llm = SimpleNamespace(
        chat=lambda system_prompt, user_prompt=None, task=None, user=None: llm_json)
    return app


def test_semantic_decision_not_overridden_by_resolve_llm():
    """语义路由高置信判定不被消解 LLM 翻盘（classify_source 必须反映真实决策者）。"""
    from intent_classifier import IntentResult
    res = IntentResult("simple", 0.9, "semantic",
                       candidates=[("simple", 0.9), ("chitchat", 0.5)])
    # 消解 LLM 硬判 chitchat —— 不允许翻盘
    app = _build_app_for_resolve(res, '{"type": "chitchat", "resolved": "你今年多大了"}')
    out = app.node_classify({"query": "这个设备的波特率是多少", "messages": []})
    assert out["query_type"] == "simple", out          # 保留语义判定
    assert out["classify_source"] == "semantic", out   # 归因不被污染
    assert out["resolved_query"] == "你今年多大了", out  # 但消解结果仍被采纳


def test_lexical_path_allows_resolve_override():
    """lexical 降级路径保留旧行为：消解 LLM 可纠偏，但 source 追加标记。"""
    from intent_classifier import IntentResult
    res = IntentResult("simple", 0.0, "lexical")
    app = _build_app_for_resolve(res, '{"type": "chitchat", "resolved": "你好呀"}')
    out = app.node_classify({"query": "hi，在吗", "messages": []})
    assert out["query_type"] == "chitchat", out              # 允许纠偏
    assert out["classify_source"].startswith("lexical"), out
    assert out["classify_source"].endswith("|resolve_override"), out


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n结果：{passed} 通过 / {failed} 失败 / 共 {len(tests)}")
    sys.exit(1 if failed else 0)
