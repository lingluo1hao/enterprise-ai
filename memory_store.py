#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MySQL 多层记忆持久化模块
========================

本模块实现三层记忆架构中的 Layer 2（MySQL 持久化层），
解决两个核心问题：

  1. 对话历史持久化 — 服务重启后对话不丢失
  2. 断点重续 — 服务宕机或用户关闭客户端后，下次登录可恢复未完成的任务

三层记忆架构：
  Layer 1  内存（_active_context）   — 最快，重启丢失（已有，保留）
  Layer 2  MySQL（本模块）           — 持久化，重启不丢（★新增）
  Layer 3  Redis（CacheManager）     — Q&A 缓存（已有，保留）

三张 MySQL 表：
  chat_messages      — 对话历史（每条 user/assistant 消息一行）
  task_checkpoints   — 断点快照（每个 LangGraph 节点执行后保存 state）
  task_queue         — 任务队列（记录任务状态：pending/running/completed/failed/interrupted）

断点重续原理：
  1. 用户提问时，先在 task_queue 创建一条 status=running 的任务记录
  2. LangGraph 每个节点执行完毕后，把当前 state 序列化为 JSON 存入 task_checkpoints
  3. 如果服务宕机/用户关闭客户端，task_queue 中的 status 仍为 running
  4. 下次用户登录时，查询 status=running 的任务
  5. 读取 task_checkpoints 最后一条快照，恢复 state
  6. 从中断的节点继续执行（不需要重新分类、检索）

容错策略：
  如果 MySQL 不可用，自动降级为内存模式（与旧版行为一致），
  打印警告但不阻断服务。所有方法都有 try-except 兜底。
"""

import json
import os
import time
import uuid
import threading
from typing import List, Dict, Any, Optional
from datetime import datetime

# ---- 加载 .env 文件 ----
def _load_dotenv(dotenv_path=None):
    if dotenv_path is None:
        import __main__
        dotenv_path = os.path.join(
            os.path.dirname(os.path.abspath(
                __main__.__file__ if hasattr(__main__, '__file__') else __file__
            )),
            ".env"
        )
    if not os.path.isfile(dotenv_path):
        return
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

_load_dotenv()

# ============================================================================
# 配置区 — MySQL 连接参数
# ============================================================================

MYSQL_HOST = os.getenv("MYSQL_HOST", "192.168.200.128")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "Root@2026")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "rag_agent")
MYSQL_CHARSET = "utf8mb4"

# 对话历史最大保留条数（超过则触发压缩）
HISTORY_MAX_TURNS = 8

# 压缩时保留的近期轮数（其余压缩为摘要）
HISTORY_COMPRESS_TURNS = 6


# ============================================================================
# JSON 序列化辅助（用于断点快照的 Document/tuple 兼容）
# ============================================================================
def _json_default(obj):
    """
    json.dumps 的 default 回调。
    - langchain_core.documents.Document → {"__type__": "Document", "page_content": ..., "metadata": ...}
    - tuple → {"__type__": "tuple", "items": [...]}（JSON 不支持 tuple）
    - 其他不可序列化对象 → 退化为 str(obj)
    """
    # Document 对象识别：duck-type 检查 page_content + metadata
    if hasattr(obj, "page_content") and hasattr(obj, "metadata"):
        try:
            meta = dict(obj.metadata) if obj.metadata else {}
        except Exception:
            meta = {}
        return {
            "__type__": "Document",
            "page_content": obj.page_content,
            "metadata": meta,
        }
    if isinstance(obj, tuple):
        return {"__type__": "tuple", "items": list(obj)}
    return str(obj)


def _json_object_hook(d):
    """
    json.loads 的 object_pairs_hook/object_hook 回调：把 _json_default 打的标记还原。
    - {"__type__": "Document", ...} → langchain Document
    - {"__type__": "tuple", "items": [...]} → tuple
    - 其他普通 dict 原样返回
    """
    if not isinstance(d, dict) or "__type__" not in d:
        return d
    t = d.get("__type__")
    if t == "Document":
        try:
            from langchain_core.documents import Document
            return Document(
                page_content=d.get("page_content", ""),
                metadata=d.get("metadata") or {},
            )
        except Exception:
            # 极端情况：langchain_core 不在。降级为简单命名空间
            class _Doc:
                pass
            doc = _Doc()
            doc.page_content = d.get("page_content", "")
            doc.metadata = d.get("metadata") or {}
            return doc
    if t == "tuple":
        return tuple(d.get("items", []))
    return d


class MySQLMemoryStore:
    """
    MySQL 持久化记忆存储

    职责：
      1. 对话历史的增删改查（替代旧的内存 _history_store）
      2. 任务断点快照的保存与恢复
      3. 任务队列状态管理

    线程安全：
      使用连接池 + threading.local() 保证多线程安全。
      Flask Web 服务器是多线程的，每个请求在不同线程中执行。
    """

    def __init__(
        self,
        host: str = MYSQL_HOST,
        port: int = MYSQL_PORT,
        user: str = MYSQL_USER,
        password: str = MYSQL_PASSWORD,
        database: str = MYSQL_DATABASE,
    ):
        """
        初始化 MySQL 连接并校验表结构。

        建表 DDL 统一由 init_db.sql 负责（单一权威源），本类不在代码里
        CREATE TABLE。如果连接失败或表缺失，self.available = False，
        所有方法降级为空操作，不影响 Agent 主流程（容错降级策略）。
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.available = False  # MySQL 是否可用标志

        # 内存降级存储（MySQL 不可用时使用）
        self._fallback_history: Dict[str, List[Dict]] = {}
        self._fallback_checkpoints: Dict[str, List[Dict]] = {}
        self._fallback_tasks: Dict[str, Dict] = {}
        self._fallback_feedback: List[Dict] = []
        self._fallback_bad_cases: List[Dict] = []

        try:
            import pymysql
            from dbutils.pooled_db import PooledDB

            # 创建数据库（如果不存在）
            # 先连 MySQL 服务器（不指定 database），创建 rag_agent 库
            conn = pymysql.connect(
                host=host, port=port, user=user, password=password,
                charset=MYSQL_CHARSET,
            )
            cursor = conn.cursor()
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{database}` "
                f"DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            conn.commit()
            cursor.close()
            conn.close()

            # 创建连接池（Web 多线程场景需要）
            # PooledDB 维护一组 TCP 连接，每个线程从池中借用、用完归还，
            # 避免每次请求都新建连接（TCP 握手 + MySQL 认证约 10-50ms）
            self._pool = PooledDB(
                creator=pymysql,
                maxconnections=5,       # 最大连接数
                mincached=1,            # 初始空闲连接数
                maxcached=3,            # 最大空闲连接数
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                charset=MYSQL_CHARSET,
                autocommit=True,        # 自动提交，简化事务管理
            )

            # 校验表结构：建表职责统一交给 init_db.sql（单一权威源），
            # 这里仅检查 6 张核心表是否已由 init_db.sql 建好
            if self._verify_schema():
                self.available = True
                print(f"  [MySQLMemoryStore] 连接成功: {host}:{port}/{database}")

        except Exception as e:
            print(f"  [MySQLMemoryStore] 连接失败，降级为内存模式: {e}")
            self._pool = None

    def _get_conn(self):
        """
        从连接池获取一个连接（线程安全）。

        PooledDB.connection() 返回一个共享连接对象，
        多线程调用时内部会自动分配空闲连接。
        用完后调用 conn.close() 会将连接归还池中（而非真正关闭）。
        """
        return self._pool.connection()


    def _verify_schema(self):
        """
        校验数据库表结构是否已由 init_db.sql 初始化。

        设计原则：建表 DDL 的单一权威源是 init_db.sql，不在代码里重复
        CREATE TABLE（避免两份 DDL 不同步、注释不一致）。本方法只做存在性
        检查：
          - 6 张核心表都在 -> 返回 True，正常可用
          - 任意表缺失    -> 降级为内存模式，并打印明确提示「请先执行 init_db.sql」
        """
        required = [
            "chat_messages", "task_checkpoints", "task_queue",
            "chat_summaries", "prompt_templates", "admin_users",
            "qa_feedback", "bad_cases",
        ]
        # 连接失败直接抛，由 __init__ 的 except 统一降级为内存模式
        conn = self._get_conn()
        cursor = conn.cursor()
        missing = []
        for t in required:
            try:
                cursor.execute("SHOW TABLES LIKE %s", (t,))
                if not cursor.fetchone():
                    missing.append(t)
            except Exception:
                missing.append(t)
        cursor.close()
        conn.close()

        if missing:
            self.available = False
            print("  [MySQLMemoryStore] 数据库表缺失，已降级为内存模式。")
            print("  [MySQLMemoryStore] 缺失表: " + ", ".join(missing))
            print("  [MySQLMemoryStore] 请先执行 init_db.sql 初始化数据库后再启动服务。")
            return False
        return True

    # ========================================================================
    # 对话历史 CRUD
    # ========================================================================

    def save_message(self, session_id: str, speaker_role: str, content: str, user_id: int = 0):
        """
        保存一条对话消息到 MySQL。

        作用：替代旧的 self._history_store[session_id].append(...)。
        现在消息直接写入 MySQL，服务重启后依然存在。

        参数：
            session_id:   会话标识符
            speaker_role: 消息说话方（user / assistant / system），不是账号权限等级
            content:      消息内容
            user_id:      登录账号的 ID（admin_users.id），外键关联，不冗余存用户名
        """
        if not self.available:
            # 降级：写入内存
            if session_id not in self._fallback_history:
                self._fallback_history[session_id] = []
            order = len(self._fallback_history[session_id])
            self._fallback_history[session_id].append(
                {"role": speaker_role, "content": content, "msg_order": order}
            )
            return

        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            # 原子取号：单条 SQL 同时完成「取最大 order + 插入」，
            # 避免 SELECT MAX + INSERT 两步非原子导致并发撞号（P0 止血 3.5）
            cursor.execute(
                "INSERT INTO chat_messages (user_id, session_id, speaker_role, content, msg_order) "
                "SELECT %s, %s, %s, %s, COALESCE(MAX(msg_order), 0) + 1 "
                "FROM chat_messages WHERE user_id = %s AND session_id = %s",
                (user_id, session_id, speaker_role, content, user_id, session_id),
            )
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"  [MySQLMemoryStore] save_message 失败: {e}")

    def load_messages(self, session_id: str, limit: int = 50, user_id: int = 0) -> List[Dict]:
        """
        从 MySQL 加载对话历史。

        作用：替代旧的 self._history_store.get(session_id, [])。
        从 MySQL 读取该会话的所有消息，按 msg_order 排序。
        P0 止血 3.4：优先带出「最新摘要」，再拼接 covers_to 之后的原始消息，
        这样重启后压缩摘要不丢（原来摘要只存内存，重启即失）。

        参数：
            session_id: 会话标识符
            limit: 最多加载多少条（默认 50，避免超长历史拖慢加载）
            user_id: 登录账号的 ID（admin_users.id），与写入时一致
        返回：
            [{"role": "user", "content": "..."}, ...] 格式的消息列表
            （返回仍用 OpenAI 约定的 "role" 键，便于直接喂给 LLM）
        """
        if not self.available:
            return [
                {"role": m["role"], "content": m["content"]}
                for m in self._fallback_history.get(session_id, [])[-limit:]
            ]

        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            # 1) 取最新一条摘要（P0 止血）
            summary = None
            try:
                cursor.execute(
                    "SELECT summary FROM chat_summaries "
                    "WHERE user_id = %s AND session_id = %s "
                    "ORDER BY covers_to DESC LIMIT 1",
                    (user_id, session_id),
                )
                srow = cursor.fetchone()
                if srow:
                    summary = srow[0]
            except Exception:
                summary = None

            # 2) 取原始消息：始终返回最近 limit 条（不再用 covers_to 屏蔽，
            #    否则历史会被越压越短，重启后只剩最后一轮）。
            #    摘要仅作为“前情提要”前缀，不替代原始消息本身。
            cursor.execute(
                "SELECT speaker_role, content FROM chat_messages "
                "WHERE user_id = %s AND session_id = %s "
                "ORDER BY msg_order DESC LIMIT %s",
                (user_id, session_id, limit),
            )
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            # DESC 查询后需要反转为正序；返回键仍用 OpenAI 约定的 "role"
            messages = [
                {"role": row[0], "content": row[1]}
                for row in reversed(rows)
            ]
            if summary:
                messages = [{"role": "system", "content": summary}] + messages
            return messages
        except Exception as e:
            print(f"  [MySQLMemoryStore] load_messages 失败: {e}")
            return []

    def clear_messages(self, session_id: str, user_id: int = 0):
        """清空指定会话的所有对话历史"""
        if not self.available:
            self._fallback_history.pop(session_id, None)
            return
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM chat_messages WHERE user_id = %s AND session_id = %s",
                (user_id, session_id),
            )
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"  [MySQLMemoryStore] clear_messages 失败: {e}")

    # ========================================================================
    # 断点快照 CRUD
    # ========================================================================

    def save_checkpoint(
        self,
        thread_id: str,
        session_id: str,
        node_name: str,
        state: Dict[str, Any],
        user_id: int = 0,
    ):
        """
        保存一个 LangGraph 断点快照。

        作用：在每个 LangGraph 节点执行完毕后调用，把当前 state 序列化存入 MySQL。
        如果服务在此之后宕机，下次可以通过 load_latest_checkpoint 恢复到这个状态。

        序列化：state 中 retrieved_docs 元素是 (Document, score) 元组，
        Document 对象不能直接 json.dumps，需要走 _json_default 拆解为字典，
        并打上 __type__ 标记，反序列化时用 object_hook 还原。

        参数：
            thread_id: 任务 ID（与 task_queue.task_id 对应）
            session_id: 会话 ID
            node_name: 刚执行完的节点名称（如 "classify", "retrieve"）
            state: LangGraph 的 AgentState 字典
            user_id: 登录账号的 ID（admin_users.id），外键关联
        """
        if not self.available:
            if thread_id not in self._fallback_checkpoints:
                self._fallback_checkpoints[thread_id] = []
            order = len(self._fallback_checkpoints[thread_id])
            self._fallback_checkpoints[thread_id].append({
                "node_name": node_name,
                "state_json": json.dumps(state, ensure_ascii=False, default=_json_default),
                "checkpoint_order": order,
            })
            return

        try:
            # state 中可能包含不可 JSON 序列化的对象（如 Document、tuple），
            # 用 _json_default 把 Document 转 dict、tuple 转 list 并打 __type__ 标记
            state_json = json.dumps(state, ensure_ascii=False, default=_json_default)

            conn = self._get_conn()
            cursor = conn.cursor()
            # 原子取号（P0 止血 3.5）：单条 SQL 避免并发撞号
            cursor.execute(
                "INSERT INTO task_checkpoints "
                "(user_id, thread_id, session_id, node_name, state_json, checkpoint_order) "
                "SELECT %s, %s, %s, %s, %s, COALESCE(MAX(checkpoint_order), 0) + 1 "
                "FROM task_checkpoints WHERE user_id = %s AND thread_id = %s",
                (user_id, thread_id, session_id, node_name, state_json,
                 user_id, thread_id),
            )
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"  [MySQLMemoryStore] save_checkpoint 失败: {e}")

    def load_latest_checkpoint(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        加载某个任务的最后一条断点快照。

        作用：断点恢复时调用。读取 task_checkpoints 中该 thread_id 的最新快照，
        返回 state 字典和中断节点名称。

        反序列化：通过 _json_object_hook 把 {"__type__": "Document"} 还原为
        langchain_core.documents.Document 对象；{"__type__": "tuple"} 还原为 tuple，
        否则后续节点访问 d[0].page_content 会报 'str' object has no attribute。

        参数：
            thread_id: 任务 ID
        返回：
            {"node_name": "retrieve", "state": {...}, "checkpoint_order": 3}
            如果没有快照，返回 None
        """
        if not self.available:
            checkpoints = self._fallback_checkpoints.get(thread_id, [])
            if not checkpoints:
                return None
            last = checkpoints[-1]
            return {
                "node_name": last["node_name"],
                "state": json.loads(last["state_json"], object_hook=_json_object_hook),
                "checkpoint_order": last["checkpoint_order"],
            }

        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT node_name, state_json, checkpoint_order "
                "FROM task_checkpoints WHERE thread_id = %s "
                "ORDER BY checkpoint_order DESC LIMIT 1",
                (thread_id,),
            )
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            if not row:
                return None
            return {
                "node_name": row[0],
                "state": json.loads(row[1], object_hook=_json_object_hook),
                "checkpoint_order": row[2],
            }
        except Exception as e:
            print(f"  [MySQLMemoryStore] load_latest_checkpoint 失败: {e}")
            return None

    # ========================================================================
    # 任务队列 CRUD
    # ========================================================================

    def create_task(
        self, session_id: str, query: str, role: str = "user",
        user_id: int = 0,
    ) -> str:
        """
        创建一个新任务，返回 task_id。

        作用：用户每次提问时调用，在 task_queue 中创建一条 status=running 的记录。
        如果后续执行中断，这条记录的 status 会保持 running，
        下次用户登录时可以通过 get_unfinished_tasks 检测到。

        参数：
            session_id: 会话 ID
            query: 用户问题
            role: 用户角色
            user_id: 登录账号的 ID（admin_users.id），防止 A 用户恢复 B 用户的任务
        返回：
            task_id（UUID 格式，全局唯一）
        """
        task_id = str(uuid.uuid4())[:8]  # 短 UUID，足够区分

        if not self.available:
            self._fallback_tasks[task_id] = {
                "task_id": task_id,
                "session_id": session_id,
                "query": query,
                "role": role,
                "status": "running",
                "answer": None,
            }
            return task_id

        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO task_queue "
                "(user_id, task_id, session_id, query, role, status) "
                "VALUES (%s, %s, %s, %s, %s, 'running')",
                (user_id, task_id, session_id, query, role),
            )
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"  [MySQLMemoryStore] create_task 失败: {e}")

        return task_id

    def update_task_status(
        self, task_id: str, status: str,
        answer: str = None, error_msg: str = None,
    ):
        """
        更新任务状态。

        作用：任务完成/失败/中断时调用，更新 task_queue 中的 status 字段。

        参数：
            task_id: 任务 ID
            status: 新状态（completed / failed / interrupted）
            answer: 如果完成，填入最终答案
            error_msg: 如果失败，填入错误信息
        """
        if not self.available:
            if task_id in self._fallback_tasks:
                self._fallback_tasks[task_id]["status"] = status
                if answer:
                    self._fallback_tasks[task_id]["answer"] = answer
                if error_msg:
                    self._fallback_tasks[task_id]["error_msg"] = error_msg
            return

        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE task_queue SET status = %s, answer = %s, error_msg = %s "
                "WHERE task_id = %s",
                (status, answer, error_msg, task_id),
            )
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"  [MySQLMemoryStore] update_task_status 失败: {e}")

    # ========================================================================
    # 用户反馈 & Bad Case 库（bad case 闭环 / 强化自进化）
    # ========================================================================
    # 这两类数据直接驱动 evalkit 的 triage + evolution：
    #   qa_feedback  用户在生产环境踩到的失败（最珍贵）
    #   bad_cases    沉淀后的失败样本 + 根因 + 修复记录

    def save_feedback(
        self, query: str, answer: str = None, rating: int = 0,
        feedback_text: str = None, task_id: str = None,
        user_id: int = 0, tenant_id: str = "default", session_id: str = None,
        root_cause: str = None,
    ) -> int:
        """
        保存一条用户反馈。rating: -1=踩 / 0=无 / 1=赞。
        返回新插入行的自增 id；MySQL 不可用时降级内存并返回内存 id。
        """
        if not self.available:
            fb = {
                "id": len(self._fallback_feedback) + 1, "query": query,
                "answer": answer, "rating": rating, "feedback_text": feedback_text,
                "task_id": task_id, "user_id": user_id, "tenant_id": tenant_id,
                "session_id": session_id, "root_cause": root_cause,
                "created_at": time.time(),
            }
            self._fallback_feedback.append(fb)
            return fb["id"]

        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO qa_feedback "
                "(task_id, user_id, tenant_id, session_id, query, answer, rating, "
                " feedback_text, root_cause) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (task_id, user_id, tenant_id, session_id, query, answer,
                 rating, feedback_text, root_cause),
            )
            new_id = cursor.lastrowid
            cursor.close()
            conn.close()
            return new_id
        except Exception as e:
            print(f"  [MySQLMemoryStore] save_feedback 失败: {e}")
            return -1

    def list_feedback(self, limit: int = 100, tenant_id: str = None,
                      rating: int = None) -> List[Dict]:
        """查询反馈。可按月租户 / 评分筛选（rating=-1 即只看点踩）。"""
        if not self.available:
            rows = list(self._fallback_feedback)
        else:
            try:
                conn = self._get_conn()
                cursor = conn.cursor()
                sql = ("SELECT id, task_id, user_id, tenant_id, session_id, query, "
                       "answer, rating, feedback_text, root_cause, created_at "
                       "FROM qa_feedback WHERE 1=1")
                params = []
                if tenant_id:
                    sql += " AND tenant_id = %s"
                    params.append(tenant_id)
                if rating is not None:
                    sql += " AND rating = %s"
                    params.append(rating)
                sql += " ORDER BY created_at DESC LIMIT %s"
                params.append(limit)
                cursor.execute(sql, params)
                cols = [c[0] for c in cursor.description]
                rows = [dict(zip(cols, r)) for r in cursor.fetchall()]
                cursor.close()
                conn.close()
                return rows
            except Exception as e:
                print(f"  [MySQLMemoryStore] list_feedback 失败: {e}")
                return []
        # 内存降级路径的筛选
        if tenant_id:
            rows = [r for r in rows if r.get("tenant_id") == tenant_id]
        if rating is not None:
            rows = [r for r in rows if r.get("rating") == rating]
        return rows[:limit]

    def add_bad_case(
        self, query: str, source: str = "feedback", suite: str = "answer",
        case_id: str = None, answer: str = None, expected: str = None,
        root_cause: str = None, diagnosis: str = None, status: str = "open",
    ) -> int:
        """写入一条 bad case。返回自增 id。"""
        if not self.available:
            bc = {
                "id": len(self._fallback_bad_cases) + 1, "source": source,
                "suite": suite, "case_id": case_id, "query": query,
                "answer": answer, "expected": expected,
                "root_cause": root_cause, "diagnosis": diagnosis,
                "status": status, "created_at": time.time(),
            }
            self._fallback_bad_cases.append(bc)
            return bc["id"]

        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO bad_cases "
                "(source, suite, case_id, query, answer, expected, root_cause, "
                " diagnosis, status) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (source, suite, case_id, query, answer, expected,
                 root_cause, diagnosis, status),
            )
            new_id = cursor.lastrowid
            cursor.close()
            conn.close()
            return new_id
        except Exception as e:
            print(f"  [MySQLMemoryStore] add_bad_case 失败: {e}")
            return -1

    def list_bad_cases(self, status: str = None, root_cause: str = None,
                       limit: int = 200) -> List[Dict]:
        """查询 bad case 库。可按状态 / 根因筛选。"""
        if not self.available:
            rows = list(self._fallback_bad_cases)
        else:
            try:
                conn = self._get_conn()
                cursor = conn.cursor()
                sql = ("SELECT id, source, suite, case_id, query, answer, expected, "
                       "root_cause, diagnosis, status, resolved_by, created_at, resolved_at "
                       "FROM bad_cases WHERE 1=1")
                params = []
                if status:
                    sql += " AND status = %s"
                    params.append(status)
                if root_cause:
                    sql += " AND root_cause = %s"
                    params.append(root_cause)
                sql += " ORDER BY created_at DESC LIMIT %s"
                params.append(limit)
                cursor.execute(sql, params)
                cols = [c[0] for c in cursor.description]
                rows = [dict(zip(cols, r)) for r in cursor.fetchall()]
                cursor.close()
                conn.close()
                return rows
            except Exception as e:
                print(f"  [MySQLMemoryStore] list_bad_cases 失败: {e}")
                return []
        if status:
            rows = [r for r in rows if r.get("status") == status]
        if root_cause:
            rows = [r for r in rows if r.get("root_cause") == root_cause]
        return rows[:limit]

    def update_bad_case_status(self, bc_id, status: str = None,
                               resolved_by: str = None, diagnosis: str = None,
                               expected: str = None) -> bool:
        """更新 bad case 状态 / 处理记录（bad case 闭环的「修复→验证」落点）。

        返回是否成功。status 仅允许 open/in_progress/resolved；
        status=resolved 时自动写 resolved_at。
        """
        if status not in (None, "open", "in_progress", "resolved"):
            status = None
        # 降级：直接改内存对象
        if not self.available:
            for bc in self._fallback_bad_cases:
                if bc.get("id") == bc_id:
                    if status:
                        bc["status"] = status
                    if resolved_by is not None:
                        bc["resolved_by"] = resolved_by
                    if diagnosis is not None:
                        bc["diagnosis"] = diagnosis
                    if expected is not None:
                        bc["expected"] = expected
                    if status == "resolved":
                        bc["resolved_at"] = time.time()
                    return True
            return False
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            sets, params = [], []
            if status:
                sets.append("status = %s")
                params.append(status)
            if resolved_by is not None:
                sets.append("resolved_by = %s")
                params.append(resolved_by)
            if diagnosis is not None:
                sets.append("diagnosis = %s")
                params.append(diagnosis)
            if expected is not None:
                sets.append("expected = %s")
                params.append(expected)
            if status == "resolved":
                sets.append("resolved_at = NOW()")
            if not sets:
                cursor.close()
                conn.close()
                return True
            params.append(bc_id)
            cursor.execute(
                f"UPDATE bad_cases SET {', '.join(sets)} WHERE id = %s", params)
            affected = cursor.rowcount
            cursor.close()
            conn.close()
            return affected > 0
        except Exception as e:
            print(f"  [MySQLMemoryStore] update_bad_case_status 失败: {e}")
            return False

    def get_unfinished_tasks(self, session_id: str, user_id: int = 0) -> List[Dict]:
        """
        查询指定会话的所有未完成任务（status=running 或 interrupted）。

        作用：用户登录/连接时调用。如果有 running 或 interrupted 状态的任务，
        说明上次执行被中断（服务宕机或用户关闭客户端），
        可以提示用户并尝试恢复。

        为什么查两个状态？
        - running: 任务正在执行中（可能是上次执行时服务突然宕机，还没来得及标记）
        - interrupted: 服务重启时由 mark_interrupted_tasks 自动标记的
        两种情况都需要提示用户恢复。

        参数：
            session_id: 会话 ID
            user_id: 登录账号的 ID（admin_users.id），防止 A 用户恢复 B 用户的任务
        返回：
            [{"task_id": "...", "query": "...", "created_at": "...", "status": "..."}, ...]
        """
        if not self.available:
            return [
                {"task_id": t["task_id"], "query": t["query"], "created_at": "", "status": t["status"]}
                for t in self._fallback_tasks.values()
                if t["session_id"] == session_id and t["status"] in ("running", "interrupted")
            ]

        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT task_id, query, created_at, status "
                "FROM task_queue WHERE user_id = %s AND session_id = %s "
                "AND status IN ('running', 'interrupted') "
                "ORDER BY created_at DESC",
                (user_id, session_id),
            )
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            return [
                {"task_id": r[0], "query": r[1], "created_at": str(r[2]), "status": r[3]}
                for r in rows
            ]
        except Exception as e:
            print(f"  [MySQLMemoryStore] get_unfinished_tasks 失败: {e}")
            return []

    # ========================================================================
    # 对话摘要 CRUD（P0 止血 3.4：压缩摘要落库，重启不丢）
    # ========================================================================

    def get_last_message_id(self, session_id: str, user_id: int = 0) -> int:
        """
        返回指定会话最后一条 chat_messages.id，用于记录摘要覆盖到的位置。
        """
        if not self.available:
            return 0
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT MAX(id) FROM chat_messages WHERE user_id = %s AND session_id = %s",
                (user_id, session_id),
            )
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            return int(row[0]) if row and row[0] is not None else 0
        except Exception:
            return 0

    def save_summary(
        self,
        user_id: int,
        session_id: str,
        summary: str,
        covers_to: int,
        msg_count: int,
        importance: int = 3,
    ):
        """
        保存一条对话历史摘要（压缩产物）。

        参数：
            user_id / session_id: 归属
            summary: 摘要文本（形如 "历史摘要: ..."）
            covers_to: 该摘要覆盖到的最后一条 chat_messages.id
            msg_count: 被压缩掉的原始消息条数
            importance: 1-5 重要性打分（P3 跨会话召回时用于加权）
        """
        if not self.available:
            return
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chat_summaries "
                "(user_id, session_id, summary, covers_from, covers_to, msg_count, importance) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (user_id, session_id, summary, 0, covers_to, msg_count, importance),
            )
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"  [MySQLMemoryStore] save_summary 失败: {e}")

    def get_task_by_id(self, task_id: str) -> Optional[Dict]:
        """根据 task_id 查询单个任务详情"""
        if not self.available:
            return self._fallback_tasks.get(task_id)

        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT task_id, session_id, query, role, status, answer, "
                "created_at, updated_at "
                "FROM task_queue WHERE task_id = %s",
                (task_id,),
            )
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            if not row:
                return None
            return {
                "task_id": row[0], "session_id": row[1], "query": row[2],
                "role": row[3], "status": row[4], "answer": row[5],
                "created_at": str(row[6]), "updated_at": str(row[7]),
            }
        except Exception as e:
            print(f"  [MySQLMemoryStore] get_task_by_id 失败: {e}")
            return None

    def mark_interrupted_tasks(self, session_id: str):
        """
        将指定会话的所有 running 任务标记为 interrupted。

        作用：服务重启时调用。上次运行中由于宕机未能正常完成的任务，
        其 status 仍为 running。启动时批量改为 interrupted，
        这样用户下次登录看到的是 interrupted 而非 running（更准确）。

        参数：
            session_id: 会话 ID（传 None 则标记所有会话）
        """
        if not self.available:
            for t in self._fallback_tasks.values():
                if session_id is None or t["session_id"] == session_id:
                    if t["status"] == "running":
                        t["status"] = "interrupted"
            return

        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            if session_id:
                cursor.execute(
                    "UPDATE task_queue SET status = 'interrupted' "
                    "WHERE session_id = %s AND status = 'running'",
                    (session_id,),
                )
            else:
                cursor.execute(
                    "UPDATE task_queue SET status = 'interrupted' "
                    "WHERE status = 'running'"
                )
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"  [MySQLMemoryStore] mark_interrupted_tasks 失败: {e}")

    def get_all_sessions(self) -> List[str]:
        """获取所有有对话记录的 session_id 列表"""
        if not self.available:
            return list(set(
                m.get("session_id", "")
                for m_list in [self._fallback_history.values()]
                for m in m_list
            ))

        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT session_id FROM chat_messages")
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            return [r[0] for r in rows]
        except Exception as e:
            print(f"  [MySQLMemoryStore] get_all_sessions 失败: {e}")
            return []

    def close(self):
        """关闭连接池"""
        if self._pool:
            # PyMySQL 的 ConnectionPool 没有 close 方法，
            # 但连接会在 GC 时自动释放
            pass


# ============================================================================
# 便捷函数：创建全局单例
# ============================================================================

_global_store: Optional[MySQLMemoryStore] = None
_store_lock = threading.Lock()


def get_memory_store() -> MySQLMemoryStore:
    """
    获取全局 MySQLMemoryStore 单例。

    使用双重检查锁保证线程安全，
    整个应用生命周期只创建一个 MySQL 连接池。
    """
    global _global_store
    if _global_store is None:
        with _store_lock:
            if _global_store is None:
                _global_store = MySQLMemoryStore()
    return _global_store


# ============================================================================
# 自测入口
# ========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MySQL 多层记忆持久化模块 — 自测")
    print("=" * 60)

    store = MySQLMemoryStore()

    if not store.available:
        print("\n[警告] MySQL 不可用，仅测试降级模式")
    else:
        print("\n[OK] MySQL 连接成功，开始测试...")

    # 测试 1: 对话历史
    print("\n--- 测试 1: 对话历史 ---")
    sid = "test_session_001"
    store.save_message(sid, "user", "JM-S509 的定位方式有哪些？")
    store.save_message(sid, "assistant", "支持 GPS、LBS、WiFi 三种定位方式。")
    store.save_message(sid, "user", "那它的续航呢？")
    msgs = store.load_messages(sid)
    print(f"  加载 {len(msgs)} 条消息:")
    for m in msgs:
        print(f"    [{m['role']}] {m['content'][:40]}")

    # 测试 2: 断点快照
    print("\n--- 测试 2: 断点快照 ---")
    tid = "test_task_001"
    store.save_checkpoint(tid, sid, "classify", {
        "query": "续航呢？", "query_type": "simple", "resolved_query": "JM-S509 续航"
    })
    store.save_checkpoint(tid, sid, "retrieve", {
        "query": "续航呢？", "retrieved_docs": ["doc1", "doc2"]
    })
    ckpt = store.load_latest_checkpoint(tid)
    print(f"  最新快照: node={ckpt['node_name']}, order={ckpt['checkpoint_order']}")
    print(f"  state: {ckpt['state']}")

    # 测试 3: 任务队列
    print("\n--- 测试 3: 任务队列 ---")
    task_id = store.create_task(sid, "JM-S509 心跳间隔是多少？", "admin")
    print(f"  创建任务: {task_id}")
    unfinished = store.get_unfinished_tasks(sid)
    print(f"  未完成任务: {len(unfinished)} 个")
    for t in unfinished:
        print(f"    task_id={t['task_id']}, query={t['query'][:30]}")

    # 标记完成
    store.update_task_status(task_id, "completed", answer="心跳间隔为 60 秒。")
    unfinished = store.get_unfinished_tasks(sid)
    print(f"  标记完成后，未完成任务: {len(unfinished)} 个")

    # 测试 4: 断点恢复模拟
    print("\n--- 测试 4: 断点恢复模拟 ---")
    # 模拟: 创建任务但未完成
    task_id2 = store.create_task(sid, "定位精度是多少？", "admin")
    print(f"  创建未完成任务: {task_id2}")
    # 模拟服务重启
    store.mark_interrupted_tasks(sid)
    unfinished = store.get_unfinished_tasks(sid)
    print(f"  标记 interrupted 后，running 任务: {len(unfinished)} 个")
    # 恢复
    ckpt = store.load_latest_checkpoint(task_id2)
    print(f"  可恢复的快照: {'有' if ckpt else '无（任务创建后尚未执行任何节点）'}")

    # 清理测试数据
    store.clear_messages(sid)
    print("\n[完成] 测试数据已清理")
    print("\n" + "=" * 60)
    print("三层记忆架构:")
    print("  Layer 1 (内存)  — 已有，保留")
    print("  Layer 2 (MySQL) — ★ 本次新增，已就绪")
    print("  Layer 3 (Redis) — 已有，保留")
    print("=" * 60)
