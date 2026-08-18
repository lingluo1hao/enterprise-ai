"""P1 自进化修复 · 覆盖率补强测试（零外部依赖：mock Milvus/MySQL/kb_version）。

背景：test_evolution_p1.py(16) + test_p1_review_fixes.py(29) 共 45 断言全过，
但"全过"≠"全覆盖"。本文件补齐对真实代码逐函数核对后发现的零覆盖 / 低覆盖分支：

  A. P1-8   node_classify 命中 playbook 后【不再】patch_success（防与 query() 顶层双计）
  B. P1-6   Extractor.extract 守卫（chitchat / 空 doc_grades / intent_text 空 → 不沉淀）
  C.        query_similar 命中判定核心 _is_match：距离阈值 + 文本兜底（dist>thresh 不命中 /
            完全一致文本命中 / difflib>=0.92 命中 / dist=None 走文本兜底）
  D.        evaluate_success 的 L3 feedback_rating 信号（赞→level3 / 踩→拦截 / 非数字中性）
  E.        extract_failure 边界（query 空 → None / bad_sources 收集 / root_cause 正确）
  F.        query_similar 命中后字段透传（rewrite_text / query_type / kb_version）
  G. R1-#3  get_task_playbook_pk MySQL 分支【无 user_id 的旧调用】SQL 不含 AND user_id
  H. P1-9   reinforce_feedback 踩 → success_count 压 0（与 patch_success 同族原子写）
  I. P1-9   patch_success 对【不存在的 pk】直接返回（不抛、不 upsert）
  J. 附带   kb_version 失败冷却【到期后重试一次】（不再无限快速失败）

运行（项目根目录）：
  python tests/test_p1_coverage_gaps.py
"""

import os
import sys
import json
import time
import threading
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("FAITHFULNESS_GRADE_ENABLED", "false")

import kb_version  # noqa: E402


# 注意：不要整体替换 kb_version.get_kb_version（会蒙掉带冷却逻辑的真实函数）。
# 只 mock 底层 _redis，让真实 get_kb_version 跑通并返回 32。
class _FakeRedisOk:
    def get(self, key):
        return "32"


kb_version._redis = lambda: _FakeRedisOk()  # type: ignore

import evolution as ev  # noqa: E402
from evolution import (  # noqa: E402
    Extractor, PlaybookStore, RetrievalPlaybook,
    _get_pk_lock, _get_current_kb_version, GRADE_THRESHOLD,
)
import langgraph_rag_agent as ra  # noqa: E402
import memory_store as ms_mod  # noqa: E402
import pymysql  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  PASS  " + name)
    else:
        FAIL += 1
        print("  FAIL  " + name)


# ---------------------------------------------------------------- mock Milvus client
class FakeMilvus:
    """按 pk 的 fake 集合：实现 search/query/upsert/flush/has_collection 等最小接口。"""

    def __init__(self):
        self.rows = {}  # pk -> row dict
        self.lock = threading.Lock()
        self.upsert_calls = 0
        self.delete_calls = 0
        self._load_state = "Loaded"
        self._next_hit_pk = None
        self.next_dist = 0.0  # 可控搜索距离，用于验证 _is_match 阈值

    def has_collection(self, name):
        return True

    def describe_collection(self, name):
        return {"fields": [
            {"name": "intent_vector", "type": "FloatVector", "params": {"dim": 1024}},
            {"name": "kb_version", "type": "INT64"},
        ]}

    def get_load_state(self, name):
        return self._load_state

    def load_collection(self, name):
        return None

    def query(self, collection, filter=None, output_fields=None):
        pk = filter.split('"')[1] if '"' in filter else None
        row = self.rows.get(pk)
        if row is None:
            return []
        return [row]

    def upsert(self, collection, rows):
        with self.lock:
            self.upsert_calls += 1
            for r in rows:
                self.rows[r["pk"]] = dict(r)

    def insert(self, collection, rows):
        self.upsert(collection, rows)

    def flush(self, collection):
        return None

    def search(self, collection, data=None, anns_field="", limit=3, filter="",
               consistency_level="", output_fields=None):
        pk = self._next_hit_pk
        row = self.rows.get(pk)
        if row is None:
            return [[]]
        ent = {k: row.get(k) for k in output_fields}
        return [[{
            "id": pk,
            "distance": self.next_dist,
            "entity": ent,
        }]]


class FakeVDB:
    def __init__(self):
        self.client = FakeMilvus()

    class _Embed:
        @staticmethod
        def embed_query(text):
            return [0.1] * 1024

    _embed = _Embed()


def _make_store(fake_milvus=None):
    vdb = FakeVDB()
    if fake_milvus is not None:
        vdb.client = fake_milvus
    ps = PlaybookStore.__new__(PlaybookStore)
    ps.vdb = vdb
    ps.client = vdb.client
    ps.collection = "skill_playbooks"
    return ps


# ==================================================================
# A. P1-8：node_classify 命中 playbook 后【不再】patch_success（防双计）
# ==================================================================
print("== A. P1-8: node_classify 命中后不 patch_success（防与 query() 双计） ==")


class _FakeStoreNoPatch:
    def __init__(self):
        self.patch_calls = 0
        self.last_pk = None

    def query_similar(self, intent_text, tenant_id, top_k=1):
        return {"pk": "pkX", "score": 0.95, "rewrite_text": '["心跳间隔"]'}

    def patch_success(self, pk):
        self.patch_calls += 1
        self.last_pk = pk


app = object.__new__(ra.LangGraphRAGApp)
app.fast_mode = True          # 走规则分类，不调 LLM
app.tenant_id = "default"
app.playbook_store = _FakeStoreNoPatch()

out = app.node_classify({"query": "心跳间隔是多少"})
check("命中 playbook 后 used_playbook_pk 透传到 state",
      out.get("used_playbook_pk") == "pkX")
check("P1-8：node_classify 绝不在此 patch_success（防双计）",
      app.playbook_store.patch_calls == 0)


# ==================================================================
# B. P1-6：Extractor.extract 守卫（不沉淀假经验）
# ==================================================================
print("== B. P1-6: Extractor.extract 守卫（chitchat / 空 doc_grades / intent 空） ==")

# chitchat / 全不相关 → relevant=0 < 阈值 → 返回 None
st_chitchat = {
    "doc_grades": [False, False], "resolved_query": "你好", "query": "你好",
    "query_type": "chitchat", "rewritten_queries": [],
}
pb = Extractor.extract(st_chitchat, "default", "u")
check("chitchat / 空相关 → extract 返回 None（不沉淀）", pb is None)

# 空 doc_grades（复杂路径无检索命中）
st_empty = {
    "doc_grades": [], "resolved_query": "心跳", "query": "心跳",
    "query_type": "complex", "rewritten_queries": [],
}
check("空 doc_grades → extract 返回 None", Extractor.extract(st_empty, "default", "u") is None)

# 正常：有相关 + 有 intent_text → 返回 RetrievalPlaybook，success_level 透传 evaluate_success
st_ok = {
    "doc_grades": [True], "resolved_query": "心跳间隔", "query": "心跳间隔",
    "query_type": "simple", "rewritten_queries": ["心跳"], "retrieved_docs": [],
}
pb_ok = Extractor.extract(st_ok, "default", "u")
check("有相关 + 有 intent_text → 返回经验对象", pb_ok is not None)
check("success_level 透传为 1（检索级）",
      pb_ok is not None and pb_ok.success_level == 1)
check("intent_text 正确取自 resolved_query",
      pb_ok is not None and pb_ok.intent_text == "心跳间隔")
check("kb_version 快照为当前版本 32",
      pb_ok is not None and pb_ok.kb_version == 32)

# intent_text 两处都为空 → 返回 None（避免空 anchor 沉淀）
st_no_intent = {
    "doc_grades": [True], "resolved_query": "", "query": "",
    "query_type": "simple", "rewritten_queries": [],
}
check("intent_text 为空 → extract 返回 None",
      Extractor.extract(st_no_intent, "default", "u") is None)


# ==================================================================
# C. query_similar 命中判定核心 _is_match（距离阈值 + 文本兜底）
# ==================================================================
print("== C. query_similar _is_match：距离阈值 + 文本兜底 ==")

m = FakeMilvus()
m.rows["pk-hit"] = {
    "pk": "pk-hit", "intent_vector": [0.5] * 1024, "intent_text": "心跳间隔",
    "query_type": "simple", "rewrite_text": '["心跳间隔"]', "node_path": "simple",
    "relevant_sources": "[]", "success_count": 1, "tenant_id": "default",
    "user_id": "u", "updated_at": "", "kb_version": 32,
}
m.rows["pk-near"] = {
    "pk": "pk-near", "intent_vector": [0.5] * 1024, "intent_text": "心跳间隔测试",
    "query_type": "simple", "rewrite_text": '["心跳间隔测试"]', "node_path": "simple",
    "relevant_sources": "[]", "success_count": 1, "tenant_id": "default",
    "user_id": "u", "updated_at": "", "kb_version": 32,
}
ps = _make_store(m)

# C1：距离 0.0 <= 0.22 → 命中
m._next_hit_pk = "pk-hit"
m.next_dist = 0.0
hit = ps.query_similar("心跳间隔", "default", top_k=1, dist_thresh=ev.HIT_DIST)
check("dist=0.0 <= HIT_DIST → 命中", hit is not None and hit["pk"] == "pk-hit")

# C2：距离 0.99 > 0.22 且文本不同 → 不命中（_is_match 返回 False）
m._next_hit_pk = "pk-hit"
m.next_dist = 0.99
miss = ps.query_similar("完全无关的问题xyz", "default", top_k=1, dist_thresh=ev.HIT_DIST)
check("dist=0.99 > HIT_DIST 且文本不同 → 返回 None（不命中）", miss is None)

# C3：距离失真（0.99）但 intent_text 完全一致 → 文本兜底命中
m._next_hit_pk = "pk-hit"
m.next_dist = 0.99
text_hit = ps.query_similar("心跳间隔", "default", top_k=1, dist_thresh=ev.HIT_DIST)
check("dist 失真但 intent_text 完全一致 → 文本兜底命中",
      text_hit is not None and text_hit["pk"] == "pk-hit")

# C4：距离失真但近重复（difflib>=0.92）→ 命中
#     "心跳间隔测试。" vs "心跳间隔测试" 相似度 = 2*8/(9+8) = 0.941 >= 0.92
m._next_hit_pk = "pk-near"
m.next_dist = 0.99
near = ps.query_similar("心跳间隔测试。", "default", top_k=1, dist_thresh=ev.HIT_DIST)
check("dist 失真但 difflib>=0.92 近重复 → 命中",
      near is not None and near["pk"] == "pk-near")

# C5：dist 为 None（Milvus 未返回距离）→ 走文本兜底，完全一致仍命中
m._next_hit_pk = "pk-hit"
m.next_dist = None
none_dist = ps.query_similar("心跳间隔", "default", top_k=1, dist_thresh=ev.HIT_DIST)
check("dist=None 走文本兜底，一致文本 → 命中",
      none_dist is not None and none_dist["pk"] == "pk-hit")


# ==================================================================
# D. evaluate_success 的 L3 feedback_rating 信号（第二位置参数，非 state 字段）
# ==================================================================
print("== D. evaluate_success L3 feedback_rating 信号 ==")

ok3, lvl3 = Extractor.evaluate_success({"doc_grades": [True]}, feedback_rating=1)
check("feedback 赞(>=1) → level 升到 3", ok3 is True and lvl3 == 3)

ok_neg, lvl_neg = Extractor.evaluate_success({"doc_grades": [True]}, feedback_rating=-1)
check("feedback 踩(<=-1) → L3 为负，ok=False（不沉淀负经验）", ok_neg is False)

ok_bad, _ = Extractor.evaluate_success({"doc_grades": [True]}, feedback_rating="bad")
check("feedback 非数字 → 回退中性（不拦正向）", ok_bad is True)

# L2 + L3 同时为真 → level = max(1,2,3) = 3
ev.FAITHFULNESS_GRADE_ENABLED = True
ok_both, lvl_both = Extractor.evaluate_success(
    {"doc_grades": [True], "faithfulness_score": 0.9}, feedback_rating=1)
ev.FAITHFULNESS_GRADE_ENABLED = False
check("L2+L3 同时为真 → level=3（最高层级）", ok_both is True and lvl_both == 3)


# ==================================================================
# E. extract_failure 边界
# ==================================================================
print("== E. extract_failure 边界（query 空 / bad_sources 收集 / root_cause） ==")

# query 为空（resolved_query/query 都空）→ 无法构造，返回 None
fail_empty = Extractor.extract_failure(
    {"doc_grades": [False], "resolved_query": "", "query": ""}, "default", "u")
check("query 为空 → extract_failure 返回 None", fail_empty is None)

# 全部不相关（检索失败）→ 收集被召回但不相关的源进 bad_sources，root_cause 标记未命中
doc_a = SimpleNamespace(metadata={"file_name": "a.pdf"})
doc_b = SimpleNamespace(metadata={"file_name": "b.pdf"})
st_bad = {
    "doc_grades": [False, False],
    "resolved_query": "续航", "query": "续航",
    "retrieved_docs": [(doc_a,), (doc_b,)],
}
fail_bad = Extractor.extract_failure(st_bad, "default", "u")
check("被召回但不相关的源收集进 bad_sources",
      fail_bad is not None and "a.pdf" in fail_bad["diagnosis"])
check("root_cause 标记为检索未命中",
      fail_bad is not None and "未命中" in fail_bad["root_cause"])


# ==================================================================
# F. query_similar 命中后字段透传
# ==================================================================
print("== F. query_similar 命中字段透传 ==")

m2 = FakeMilvus()
m2.rows["pk-f"] = {
    "pk": "pk-f", "intent_vector": [0.5] * 1024, "intent_text": "新经验",
    "query_type": "simple", "rewrite_text": '["新改写"]', "node_path": "simple",
    "relevant_sources": "[]", "success_count": 7, "tenant_id": "default",
    "user_id": "u", "updated_at": "", "kb_version": 32,
}
ps2 = _make_store(m2)
m2._next_hit_pk = "pk-f"
m2.next_dist = 0.0
hit_f = ps2.query_similar("新经验", "default", top_k=1, dist_thresh=ev.HIT_DIST)
check("命中透传 rewrite_text", hit_f is not None and hit_f["rewrite_text"] == '["新改写"]')
check("命中透传 query_type", hit_f is not None and hit_f["query_type"] == "simple")
check("命中透传 kb_version", hit_f is not None and hit_f["kb_version"] == 32)
check("命中透传 success_count", hit_f is not None and hit_f["success_count"] == 7)


# ==================================================================
# G. R1-#3：get_task_playbook_pk MySQL 分支【无 user_id 旧调用】
# ==================================================================
print("== G. R1-#3: MySQL 分支无 user_id 旧调用（SQL 不含 AND user_id） ==")


class _FakeCursorNoUid:
    def __init__(self, rows):
        self._rows = rows
        self.last_sql = None

    def execute(self, sql, params=None):
        self.last_sql = sql

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def close(self):
        pass


class _FakeConnNoUid:
    def __init__(self, rows):
        self._rows = rows
        self.cursor_obj = _FakeCursorNoUid(rows)

    def cursor(self):
        return self.cursor_obj

    def close(self):
        pass


ms_mysql = object.__new__(ms_mod.MySQLMemoryStore)
ms_mysql.available = True
ms_mysql._conn = _FakeConnNoUid([("pk-legacy",)])
ms_mysql._get_conn = lambda: ms_mysql._conn
pk_legacy = ms_mysql.get_task_playbook_pk("t-legacy")  # 不传 user_id（旧调用）
check("MySQL 无 user_id 旧调用 → 返回 pk", pk_legacy == "pk-legacy")
check("MySQL 无 user_id → SQL 不含 AND user_id（兼容旧调用）",
      " AND user_id" not in ms_mysql._conn.cursor_obj.last_sql)


# ==================================================================
# H. P1-9：reinforce_feedback 踩 → success_count 压 0（同族原子写）
# ==================================================================
print("== H. P1-9: reinforce_feedback 踩 → success_count=0 ==")

mh = FakeMilvus()
mh.rows["pkH"] = {
    "pk": "pkH", "intent_vector": [0.5] * 1024, "intent_text": "反馈",
    "query_type": "simple", "rewrite_text": "[]", "node_path": "simple",
    "relevant_sources": "[]", "success_count": 5, "tenant_id": "default",
    "user_id": "u", "updated_at": "", "kb_version": 32,
}
psh = _make_store(mh)
psh.reinforce_feedback("pkH", positive=False)
check("踩负反馈 → success_count 压 0", mh.rows["pkH"]["success_count"] == 0)
check("踩负反馈 → 走 upsert（原子覆盖）", mh.upsert_calls == 1 and mh.delete_calls == 0)


# ==================================================================
# I. P1-9：patch_success 对不存在的 pk 直接返回（不抛、不 upsert）
# ==================================================================
print("== I. P1-9: patch_success 不存在的 pk → 直接返回 ==")

mi = FakeMilvus()
psi = _make_store(mi)
try:
    psi.patch_success("pk-not-exist")
    check("patch_success 不存在的 pk → 不抛异常", True)
except Exception:
    check("patch_success 不存在的 pk → 不抛异常", False)
check("patch_success 不存在的 pk → 不触发 upsert/delete",
      mi.upsert_calls == 0 and mi.delete_calls == 0)


# ==================================================================
# J. 附带：kb_version 失败冷却【到期后重试一次】
# ==================================================================
print("== J. 附带: kb_version 失败冷却到期后重试一次 ==")


class _FakeRedisRetry:
    def __init__(self):
        self.calls = 0
        self.down = True

    def get(self, key):
        self.calls += 1
        if self.down:
            raise ConnectionError("redis down")
        return "32"


_orig_redis = kb_version._redis
_orig_fail_until = kb_version._cache_fail_until
_orig_lock = kb_version._cache_lock
_orig_cooldown = kb_version._CACHE_FAIL_COOLDOWN
try:
    kb_version._CACHE_FAIL_COOLDOWN = 0.05  # 冷却缩到 50ms，避免测试等 5s
    kb_version._cache_lock = threading.Lock()
    kb_version._cache_fail_until = 0.0

    fr = _FakeRedisRetry()
    kb_version._redis = lambda: fr  # 临时换成「首次失败」的 fake

    v1 = kb_version.get_kb_version()
    check("首次失败 → 返回 None（fail-open）", v1 is None)
    check("首次失败后进入冷却", kb_version._cache_fail_until > time.time())

    # 冷却期内：不碰 Redis，立即 None
    before_cool = fr.calls
    v_cool = kb_version.get_kb_version()
    check("冷却期内快速失败（不触发 Redis）",
          v_cool is None and fr.calls == before_cool)

    # 等冷却到期 + 恢复 Redis
    time.sleep(0.06)
    fr.down = False
    before_retry = fr.calls
    v2 = kb_version.get_kb_version()
    check("冷却到期 → 重新重试 Redis 并成功返回 32",
          v2 == 32 and fr.calls == before_retry + 1)
    check("成功后清除冷却（_cache_fail_until 归零）",
          kb_version._cache_fail_until == 0.0)
finally:
    kb_version._redis = _orig_redis
    kb_version._cache_fail_until = _orig_fail_until
    kb_version._cache_lock = _orig_lock
    kb_version._CACHE_FAIL_COOLDOWN = _orig_cooldown


# ==================================================================
# L. Case #17 修复：_quick_classify 闲聊词表扩充 + 复合句防误伤
# ==================================================================
print("\n== L. Case #17: _quick_classify 闲聊词表扩充 ==")

_appq = object.__new__(ra.LangGraphRAGApp)

# 闲聊类（本应直接 LLM 回答，且不再是"未检索到"生硬回复）
for _q, _kw in [
    ("你今年多大了", "年龄"),
    ("你几岁", "年龄"),
    ("你叫什么名字", "你是谁"),
    ("今天天气怎么样", "这是当日闲聊"),
    ("最近怎么样", "寒暄"),
    ("在吗", "寒暄"),
]:
    check("chitchat: %s（%s）→ _quick_classify 判 chitchat" % (_q, _kw),
          _appq._quick_classify(_q) == "chitchat")

# 防误伤：打招呼+问正事 → simple（不是 chitchat）
for _q in ["你好，心跳间隔是多少", "你好，请介绍一下JM-S509的功能"]:
    check("复合句: %s → _quick_classify 判 simple（不误判闲聊）" % _q,
          _appq._quick_classify(_q) == "simple")

# 领域正常问题仍走 simple / complex
check("simple 不受扩充影响: 心跳间隔是多少",
      _appq._quick_classify("心跳间隔是多少") == "simple")
check("complex 不受扩充影响: 心跳是多少？波特率是？",
      _appq._quick_classify("心跳是多少？波特率是？") == "complex")


print("\n===== 结果: PASS=%d FAIL=%d =====" % (PASS, FAIL))
sys.exit(1 if FAIL else 0)
