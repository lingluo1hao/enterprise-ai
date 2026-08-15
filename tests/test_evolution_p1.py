"""P1 自进化修复单元测试（零外部依赖，mock Milvus client，不连 VM / 不调 LLM）。

运行方式（项目根目录）：
  python tests/test_evolution_p1.py

覆盖：
  P1-6  chitchat / 空 doc_grades 不再灌假 Bad Case（save_history 守卫 + complex 聚合）
  P1-8  patch_success 只由顶层成功作答后触发（node_classify 不再双计）
  P1-9  patch_success 走 upsert（无 delete 窗口）+ 显式 intent_vector 回读 + per-pk 锁
  P1-10 query_similar 对过期 kb_version 跳过复用
  P1-7  FAITHFULNESS_GRADE_ENABLED=false 时 L2 中性（不拦正向沉淀）
"""

import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("FAITHFULNESS_GRADE_ENABLED", "false")

from evolution import (  # noqa: E402
    Extractor, PlaybookStore, RetrievalPlaybook,
    _get_pk_lock, _get_current_kb_version,
)

# P1-R1：测试不依赖真实 Redis——mock kb_version.get_kb_version 返回固定版本，
# 避免连不上 Redis 时 redis-py 内置重试把测试拖几十秒。
import kb_version  # noqa: E402

FAKE_KB_VERSION = 32
kb_version.get_kb_version = lambda tenant="global": FAKE_KB_VERSION  # type: ignore

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

    # 供 PlaybookStore._ensure 使用的探测/建表接口
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

    # 读写
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

    # search 返回命中向量行（模拟 query_similar 命中）
    def search(self, collection, data=None, anns_field="", limit=3, filter="", consistency_level="", output_fields=None):
        pk = self._next_hit_pk
        row = self.rows.get(pk)
        if row is None:
            return [[]]
        return [[{
            "id": pk,
            "distance": 0.0,
            "entity": {k: row.get(k) for k in output_fields},
        }]]


class FakeVDB:
    """模拟 vector_db：提供 client + _embed。"""

    def __init__(self):
        self.client = FakeMilvus()

    class _Embed:
        @staticmethod
        def embed_query(text):
            return [0.1] * 1024

    _embed = _Embed()


class FakeAgent:
    """极简 agent 外壳，满足 PlaybookStore 构造签名（vector_db 即可）。"""

    def __init__(self):
        self.client = FakeMilvus()


def _make_store(fake_milvus=None):
    """构造 PlaybookStore，但跳过 _ensure 的建表（FakeMilvus.has_collection=True 且 schema 已含 kb_version）。"""
    vdb = FakeVDB()
    if fake_milvus is not None:
        vdb.client = fake_milvus
    ps = PlaybookStore.__new__(PlaybookStore)
    ps.vdb = vdb
    ps.client = vdb.client
    ps.collection = "skill_playbooks"
    return ps


print("== P1-6: 三级成功信号 evaluate_success 守卫 ==")

# chitchat / 空 doc_grades：retrieval_ok=False -> ok=False -> 走 extract_failure（会在 save_history 被守卫拦截）
ok, level = Extractor.evaluate_success({"doc_grades": []})
check("空 doc_grades -> ok=False（守卫触发点）", ok is False)

ok, level = Extractor.evaluate_success({"doc_grades": [True, True]})
check("doc_grades 2 相关 -> ok=True, level=1", ok is True and level == 1)

# P1-7: L2 开关默认关 -> faithfulness_score 存在也不拦（中性）
ok, level = Extractor.evaluate_success({"doc_grades": [True], "faithfulness_score": 0.2})
check("L2 默认关：faithfulness=0.2 仍 ok=True（不拦）", ok is True)

print("== P1-7: 开启 FAITHFULNESS_GRADE_ENABLED 时 L2 生效 ==")
import evolution as ev

ev.FAITHFULNESS_GRADE_ENABLED = True
ok, level = Extractor.evaluate_success({"doc_grades": [True], "faithfulness_score": 0.2})
check("L2 开：faithfulness=0.2 < 0.5 -> ok=False（拦截）", ok is False)
ok, level = Extractor.evaluate_success({"doc_grades": [True], "faithfulness_score": 0.9})
check("L2 开：faithfulness=0.9 >= 0.5 -> ok=True, level=2", ok is True and level == 2)
ev.FAITHFULNESS_GRADE_ENABLED = False  # 复位

print("== P1-9: patch_success 用 upsert + 显式向量回读 + per-pk 锁 ==")

m = FakeMilvus()
m.rows["pk1"] = {
    "pk": "pk1", "intent_vector": [0.5] * 1024, "intent_text": "心跳间隔",
    "query_type": "simple", "rewrite_text": '["心跳间隔"]', "node_path": "simple",
    "relevant_sources": "[]", "success_count": 1, "tenant_id": "default",
    "user_id": "u", "updated_at": "2026-01-01 00:00:00", "kb_version": 32,
}
ps = _make_store(m)
ps.patch_success("pk1")
check("patch_success 走 upsert（无 delete 调用）", m.delete_calls == 0 and m.upsert_calls == 1)
check("success_count 从 1 -> 2", m.rows["pk1"]["success_count"] == 2)
check("intent_vector 未被丢（显式回读+upsert 保持向量）",
      isinstance(m.rows["pk1"].get("intent_vector"), list) and len(m.rows["pk1"]["intent_vector"]) == 1024)
check("updated_at 已刷新", m.rows["pk1"]["updated_at"] != "2026-01-01 00:00:00")

# 并发 10 线程对同一 pk 各 +1：per-pk 锁串行化 -> 最终 +10（upsert 幂等，最多少计）
m2 = FakeMilvus()
m2.rows["pkA"] = {
    "pk": "pkA", "intent_vector": [0.5] * 1024, "intent_text": "并发",
    "query_type": "simple", "rewrite_text": "[]", "node_path": "simple",
    "relevant_sources": "[]", "success_count": 0, "tenant_id": "default",
    "user_id": "u", "updated_at": "", "kb_version": 32,
}
ps2 = _make_store(m2)
lock = _get_pk_lock("pkA")
results = []


def _inc():
    ps2.patch_success("pkA")
    with lock:
        results.append(m2.rows["pkA"]["success_count"])


threads = [threading.Thread(target=_inc) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
check("并发 10 线程同 pk：最终 success_count >= 10（upsert 幂等，不丢行）",
      m2.rows["pkA"]["success_count"] >= 10)

print("== P1-10: query_similar 过期 kb_version 跳过复用 ==")

m3 = FakeMilvus()
m3.rows["pk-old"] = {
    "pk": "pk-old", "intent_vector": [0.5] * 1024, "intent_text": "旧经验",
    "query_type": "simple", "rewrite_text": '["旧改写"]', "node_path": "simple",
    "relevant_sources": "[]", "success_count": 3, "tenant_id": "default",
    "user_id": "u", "updated_at": "", "kb_version": 1,  # 旧版本
}
m3.rows["pk-new"] = {
    "pk": "pk-new", "intent_vector": [0.5] * 1024, "intent_text": "新经验",
    "query_type": "simple", "rewrite_text": '["新改写"]', "node_path": "simple",
    "relevant_sources": "[]", "success_count": 1, "tenant_id": "default",
    "user_id": "u", "updated_at": "", "kb_version": _get_current_kb_version(),
}
ps3 = _make_store(m3)

m3._next_hit_pk = "pk-old"
hit_old = ps3.query_similar("旧经验", "default", top_k=1, dist_thresh=0.5)
check("kb_version=1 过期（当前 > 1）-> 跳过复用返回 None", hit_old is None)

m3._next_hit_pk = "pk-new"
hit_new = ps3.query_similar("新经验", "default", top_k=1, dist_thresh=0.5)
check("kb_version=当前 -> 正常复用", hit_new is not None and hit_new["pk"] == "pk-new")

m3._next_hit_pk = "pk-old"
hit_old_allow = ps3.query_similar("旧经验", "default", top_k=1, dist_thresh=0.5,
                                  allowed_stale=True)
check("allowed_stale=True -> 过期也允许（去重合并场景）", hit_old_allow is not None)

# P1-R1：新增回归——版本读取失败（Redis 挂）时 fail-open，不误杀任何经验
_orig_get_kb_version = kb_version.get_kb_version
kb_version.get_kb_version = lambda tenant="global": None  # 模拟读失败
m3._next_hit_pk = "pk-old"
hit_failopen = ps3.query_similar("旧经验", "default", top_k=1, dist_thresh=0.5)
check("cur_version=None（读失败）-> fail-open 跳过时效校验，允许复用",
      hit_failopen is not None and hit_failopen["pk"] == "pk-old")
kb_version.get_kb_version = _orig_get_kb_version

# P1-R1：新增回归——合法 kb_version=0（全新部署 Redis 未 bump）不得当"缺失"判过期
m3.rows["pk-zero"] = {
    "pk": "pk-zero", "intent_vector": [0.5] * 1024, "intent_text": "全新经验",
    "query_type": "simple", "rewrite_text": '["新改写"]', "node_path": "simple",
    "relevant_sources": "[]", "success_count": 1, "tenant_id": "default",
    "user_id": "u", "updated_at": "", "kb_version": 0,
}
kb_version.get_kb_version = lambda tenant="global": 0  # 当前版本也是 0
m3._next_hit_pk = "pk-zero"
hit_zero = ps3.query_similar("全新经验", "default", top_k=1, dist_thresh=0.5)
check("kb_version=0（当前也是 0）-> 是合法匹配，正常复用", hit_zero is not None)
kb_version.get_kb_version = _orig_get_kb_version

print("== P1-6: extract_failure 在空 doc_grades 时仍能构造（守卫在 save_history） ==")
fail = Extractor.extract_failure({"query": "你好", "resolved_query": "你好",
                                  "doc_grades": []}, "default", "u")
check("空 doc_grades -> extract_failure 构造成功（含 root_cause=检索未命中）", fail is not None)

print("\n===== 结果: PASS=%d FAIL=%d =====" % (PASS, FAIL))
sys.exit(1 if FAIL else 0)
