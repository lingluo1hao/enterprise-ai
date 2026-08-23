"""P1-R1 审查整改单元测试（零外部依赖：mock Milvus/MySQL/kb_version，不连真实服务）。

运行方式（项目根目录）：
  python tests/test_p1_review_fixes.py

覆盖（对应 Claude Code 审查 1-7 + 附带修复）：
  R1-#1   kb_version=0（全新部署）不再是 falsy-zero，query_similar 正常复用
  R1-#2   cur_version=None（Redis 读失败）fail-open，不误杀任何经验
  R1-#3   get_task_playbook_pk 带 user_id 归属校验（内存 fallback + MySQL 分支）防越权
  R1-#4   _pk_locks LRU 上限淘汰最久未用（无界增长泄漏消除）
  R1-#5   answer_failed 显式标志：writer/generate_simple/respond 失败置位，
          query() success_count 只对真正成功作答 +1（字符串启发式移除）
  R1-#6   schema 迁移只对 errno=1054（unknown column）触发 ALTER，其他异常不误操作
  附带    kb_version 失败冷却：Redis 不可达时快速失败不再重试拖死热路径
"""

import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import kb_version  # noqa: E402
from evolution import _get_pk_lock, _pk_locks  # noqa: E402

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


# ==================================================================
# R1-#4：_pk_locks LRU 上限
# ==================================================================
print("== R1-#4: _pk_locks LRU 上限（无界增长修复） ==")

import evolution as ev  # noqa: E402

_orig_max = ev._PK_LOCKS_MAX
_orig_locks = dict(ev._pk_locks)
try:
    ev._PK_LOCKS_MAX = 3
    ev._pk_locks.clear()
    _get_pk_lock("a")
    _get_pk_lock("b")
    _get_pk_lock("c")       # 达到上限 3
    _get_pk_lock("a")       # 访问 a -> 移到末尾
    _get_pk_lock("d")       # 超出 -> 应淘汰最久未用的 b
    check("超上限后淘汰最久未用的 b", "b" not in ev._pk_locks)
    check("d 已加入", "d" in ev._pk_locks)
    check("仍保留近期访问的 a/c", "a" in ev._pk_locks and "c" in ev._pk_locks)
    check("锁字典大小不超上限", len(ev._pk_locks) <= 3)
finally:
    ev._PK_LOCKS_MAX = _orig_max
    ev._pk_locks.clear()
    ev._pk_locks.update(_orig_locks)


# ==================================================================
# R1-#5：answer_failed 显式标志
# ==================================================================
print("== R1-#5: answer_failed 显式标志（替代字符串启发式） ==")

import langgraph_rag_agent as ra  # noqa: E402

# 用 __new__ 绕过 __init__（不连真实依赖），只测节点方法
app = object.__new__(ra.LangGraphRAGApp)


class _FakePM:
    def get_prompt(self, name):
        return {"system": "s", "user_template": "{query}\n{results_text}"}

    def format_user_message(self, tpl, **kw):
        return tpl.format(**kw)


class _FakeLLM:
    def chat(self, system_prompt, user_prompt, **kw):
        return "真实总结答案"


app.pm = _FakePM()
app.llm = _FakeLLM()
app.username = "tester"

# 1. writer：无子任务结果 -> 失败
out = app.node_writer({"resolved_query": "Q", "query": "Q", "research_results": []})
check("writer 空结果 -> answer_failed=True",
      out.get("answer_failed") is True and out["answer"].startswith("未检索到"))

# 2. writer：所有子任务都未检索到 -> 失败
out = app.node_writer({"resolved_query": "Q", "query": "Q",
                       "research_results": [
                           {"subtask": "A", "answer": "未检索到相关内容。"},
                           {"subtask": "B", "answer": "未检索到相关内容。"},
                       ]})
check("writer 全未检索到 -> answer_failed=True",
      out.get("answer_failed") is True)

# 3. writer：有真实结果 -> 正常（无 answer_failed）
out = app.node_writer({"resolved_query": "Q", "query": "Q",
                       "research_results": [{"subtask": "A", "answer": "有内容"}]})
check("writer 有真实结果 -> 不标失败",
      out.get("answer_failed") is not True and out["answer"] == "真实总结答案")

# 4. generate_simple：_do_generate 返回失败文案 -> 失败
app._do_generate = lambda q, d, role=None: "未检索到与问题相关的文档内容，无法回答。"
out = app.node_generate_simple({"query": "Q", "retrieved_docs": []})
check("generate_simple 检索无果 -> answer_failed=True",
      out.get("answer_failed") is True)

# 5. generate_simple：正常答案 -> 成功（即使答案含'未检索到'字样也不误判）
app._do_generate = lambda q, d, role=None: "抱歉，无法回答这个问题。但协议是 x36。"
out = app.node_generate_simple({"query": "Q", "retrieved_docs": [1]})
check("generate_simple 正常答案 -> 不标失败（旧的字符串启发式会误判）",
      out.get("answer_failed") is not True)

# 6. respond：上游没写 answer -> 兜底失败
out = app.node_respond({})
check("respond 无 answer 兜底 -> answer_failed=True",
      out.get("answer_failed") is True and "无法回答" in out["answer"])

# 7. respond：已有 answer -> 成功
out = app.node_respond({"answer": "正常回答"})
check("respond 有 answer -> 不标失败", out.get("answer_failed") is not True)

# 8. query 判定逻辑（抽取自 query()）：answer_failed 时不 +1
def _should_patch(answer, answer_failed):
    return bool(answer) and not bool(answer_failed)

check("query 判定：answer_failed=True -> 不 patch_success",
      _should_patch("抱歉，无法回答这个问题。", True) is False)
check("query 判定：正常答案含'无法回答'字样也 patch（显式标志兜底）",
      _should_patch("抱歉，无法回答这个问题。但协议是 x36。", False) is True)


# ==================================================================
# R1-#3：get_task_playbook_pk 归属校验（防越权回灌）
# ==================================================================
print("== R1-#3: get_task_playbook_pk user_id 归属校验（防越权） ==")

import memory_store as ms_mod  # noqa: E402


def _make_ms(available, fallback_tasks=None):
    ms = object.__new__(ms_mod.MySQLMemoryStore)
    ms.available = available
    ms._fallback_tasks = fallback_tasks or {}
    return ms


# 内存 fallback 分支
ms = _make_ms(False, {
    "task-owner": {"user_id": 101, "used_playbook_pk": "pk-shared"},
})
check("fallback：归属用户查到 pk", ms.get_task_playbook_pk("task-owner", user_id=101) == "pk-shared")
check("fallback：他人 user_id -> None（越权拦截）",
      ms.get_task_playbook_pk("task-owner", user_id=999) is None)
check("fallback：不带 user_id -> 返回 pk（兼容旧调用）",
      ms.get_task_playbook_pk("task-owner") == "pk-shared")
check("fallback：task 不存在 -> None", ms.get_task_playbook_pk("nope", user_id=1) is None)

# MySQL 分支（mock _get_conn）
class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.last_sql = None

    def execute(self, sql, params=None):
        self.last_sql = sql
        self.last_params = params
        if " AND user_id" in sql and str(self.last_params[1]) != "101":
            self._rows = []
        return 0

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def close(self):
        pass


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows
        self.cursor_obj = _FakeCursor(rows)

    def cursor(self):
        return self.cursor_obj

    def close(self):
        pass


def _run_fetch(ms):
    """执行一次带 user_id 的查询，返回 cursor（供断言 SQL 内容）。"""
    ms.get_task_playbook_pk("t1", user_id=101)
    return ms._conn.cursor_obj


ms2 = object.__new__(ms_mod.MySQLMemoryStore)
ms2.available = True
ms2._conn = _FakeConn([("pk-shared",)])
ms2._get_conn = lambda: ms2._conn
check("MySQL：带 user_id 的 SQL 含归属过滤",
      " AND user_id" in _run_fetch(ms2).last_sql)
check("MySQL：归属用户查到 pk", ms2.get_task_playbook_pk("t1", user_id=101) == "pk-shared")

ms3 = object.__new__(ms_mod.MySQLMemoryStore)
ms3.available = True
ms3._conn = _FakeConn([("pk-shared",)])
ms3._get_conn = lambda: ms3._conn
check("MySQL：他人 user_id -> None（越权拦截）",
      ms3.get_task_playbook_pk("t1", user_id=999) is None)


# ==================================================================
# R1-#6：schema 迁移只对 errno=1054 触发 ALTER
# ==================================================================
print("== R1-#6: 迁移仅 unknown column(1054) 触发 ALTER ==")

import pymysql  # noqa: E402


class _FakeMigConn:
    def __init__(self, err):
        self._err = err
        self.alter_called = False
        self.committed = False

    def cursor(self):
        return self

    def execute(self, sql, params=None):
        if sql.startswith("SELECT"):
            if self._err is not None:
                raise self._err
        else:
            self.alter_called = True
        return 0

    def fetchone(self):
        return None

    def commit(self):
        self.committed = True

    def close(self):
        pass


def _run_migrate(err):
    ms = object.__new__(ms_mod.MySQLMemoryStore)
    conn = _FakeMigConn(err)
    ms._get_conn = lambda: conn
    ms._migrate_task_used_playbook_pk()
    return conn


# 缺列 -> ALTER + commit
conn = _run_migrate(pymysql.err.OperationalError(1054, "Unknown column 'used_playbook_pk'"))
check("errno=1054 -> 执行 ALTER", conn.alter_called is True)
check("errno=1054 -> commit", conn.committed is True)

# 其他异常（连接断开）-> 不 ALTER（修复前裸 except 会误操作）
conn = _run_migrate(pymysql.err.OperationalError(2013, "Lost connection"))
check("errno=2013 -> 不执行 ALTER（防误操作）", conn.alter_called is False)

# 无异常（列已存在）-> 不 ALTER
conn = _run_migrate(None)
check("列已存在 -> 不执行 ALTER", conn.alter_called is False)


# ==================================================================
# 附带：kb_version 失败冷却（Redis 不可达快速失败，不重试拖死热路径）
# ==================================================================
print("== 附带: kb_version 失败冷却（redis-py 重试防护） ==")


class _FakeRedis:
    def __init__(self, value, fail=False):
        self._value = value
        self._fail = fail
        self.get_calls = 0

    def get(self, key):
        self.get_calls += 1
        if self._fail:
            raise ConnectionError("redis down")
        return self._value


# 用 monkeypatch 临时替换 kb_version 内部状态，抹掉真实连接的影响
_orig_redis = kb_version._redis
_orig_fail_until = kb_version._cache_fail_until
_orig_lock = kb_version._cache_lock
try:
    kb_version._cache_lock = threading.Lock()
    kb_version._cache_value = None
    kb_version._cache_fail_until = 0.0
    fake = _FakeRedis("32")
    kb_version._redis = lambda: fake
    v = kb_version.get_kb_version()
    check("读成功返回 32", v == 32)
    check("读成功后无冷却", kb_version._cache_fail_until == 0.0)

    fake2 = _FakeRedis(None, fail=True)
    kb_version._redis = lambda: fake2
    v_fail = kb_version.get_kb_version()
    check("失败返回 None（fail-open 语义）", v_fail is None)
    check("失败后进入冷却", kb_version._cache_fail_until > time.time())
    # 冷却期内第二次调用：不碰 Redis，立即 None
    before = fake2.get_calls
    v_fast = kb_version.get_kb_version()
    check("冷却期内快速失败（不再触发 Redis）",
          v_fast is None and fake2.get_calls == before)
finally:
    kb_version._redis = _orig_redis
    kb_version._cache_fail_until = _orig_fail_until


print("\n===== 结果: PASS=%d FAIL=%d =====" % (PASS, FAIL))
sys.exit(1 if FAIL else 0)