"""
================================================================================
  RAG Agent Web 界面 — Flask + SSE 实时进度推送
================================================================================

  为非技术人员提供友好的人工智能问答界面。
  - 支持普通用户 / 特权用户两种角色（由登录账号决定）
  - 实时显示推理进度（SSE 推送）
  - 自动连接 Redis 缓存加速重复问题
  - 后端：LangGraph 引擎（默认）+ LLM 网关多模型路由
    （qwen2:7b 生成/规划、qwen2.5:1.5b 打分/改写/压缩）
  - 向量检索：Milvus 默认 + Chroma 兜底；Embedding 默认 Ollama bge-m3

  启动方式：
    python rag_web_server.py              # 默认端口 8080
    python rag_web_server.py --port 9090  # 指定端口

================================================================================
"""

import io
import os
import sys
import json
import time
import queue
import threading
from pathlib import Path

# ====== 环境配置（必须在所有 import 之前） ======
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, Response, g

# ====== 导入核心模块 ======
from audit_logger import get_audit_logger
from advanced_rag_agent import (
    OLLAMA_URL, MODEL_NAME, DB_PATH,
    ROLE_ADMIN, DEFAULT_ROLE,
    AccessControlFilter, CacheManager,
    RAGOrchestrator,
    OllamaLLM,
    create_llm,
    VectorStoreManager,
    REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB,
)


def _derive_session_id(user_id: str = "anonymous", role: str = "user") -> str:
    """会话 ID 按用户 + 角色派生，避免同角色用户共用历史（P0 止血 3.2）。

    Web 端当前用 username（如 guest/admin）+ role 共同区分会话。
    这样不同 guest 用户、或同一浏览器切换账号后，历史不会串。
    """
    safe_user = "".join(c if c.isalnum() or c in "-_" else "_" for c in (user_id or "anonymous"))[:32]
    safe_role = "".join(c if c.isalnum() or c in "-_" else "_" for c in (role or "user"))[:16]
    return f"web:{safe_role}:{safe_user}"


class LangGraphEngine:
    """适配器：让 LangGraphRAGApp 兼容 rag_web_server 的 RAGOrchestrator 接口"""

    def __init__(self, fast_mode=True, user_role=DEFAULT_ROLE):
        from langgraph_rag_agent import LangGraphRAGApp
        self.app = LangGraphRAGApp(fast_mode=fast_mode)
        self.user_role = user_role
        self.cache = self.app.cache
        # 兼容角色切换中的 skill_registry.get_skill() 调用
        self.skill_registry = type("SR", (), {"get_skill": lambda self, name: None})()

    def query(self, question, user_role=None, user=None, user_id=None):
        role = user_role or self.user_role
        user = user or role
        return self.app.query(question, role=role,
                              session_id=_derive_session_id(user, role),
                              user=user, user_id=user_id)

    def check_unfinished_tasks(self, session_id="web_session", user_id=0):
        """查询指定会话的未完成任务（断点检测）"""
        return self.app.check_unfinished_tasks(session_id, user_id=user_id)

    def resume_task(self, task_id, session_id="web_session", user_id=None):
        """从断点恢复执行指定任务"""
        return self.app.resume(task_id, session_id=session_id, user_id=user_id)


# ====== Flask 应用 ======
app = Flask(__name__)

# 全局变量
orchestrator = None
llm = None
vector_db = None
# LangGraph 引擎开关：默认开启。
# 注意 gunicorn 多 worker 不执行 __main__，必须用环境变量控制（不能写死 False），
# 否则所有 worker 会回退到旧版 RAGOrchestrator。可用 RAG_LANGGRAPH=0/False 关闭。
use_langgraph = os.getenv("RAG_LANGGRAPH", "true").lower() not in ("0", "false", "no", "off")


# ======================================================================
# 进度捕获器 — 拦截 print 输出并转发到队列，供 SSE 实时推送
# ======================================================================

class ProgressWriter(io.StringIO):
    """替换 sys.stdout，把每次 write 写入到队列中，同时保留到原始 stdout"""

    def __init__(self, _queue: queue.Queue, original_stdout):
        super().__init__()
        self._queue = _queue
        self._original = original_stdout

    def write(self, s):
        if s.strip():
            self._queue.put({"type": "log", "text": s.strip()})
        self._original.write(s)
        self._original.flush()

    def flush(self):
        self._original.flush()


# ======================================================================
# 输入校验 — 安全防护
# ======================================================================

# ---- 字段长度限制 ----
MAX_QUESTION_LEN = 2000          # 用户问题最大字符数
MAX_PROMPT_NAME_LEN = 100        # 提示词名称
MAX_PROMPT_SYSTEM_LEN = 10000    # 系统提示词
MAX_PROMPT_USER_TEMPLATE_LEN = 10000  # 用户模板
MAX_PROMPT_DISPLAY_LEN = 200     # 显示名称
MAX_PROMPT_DESC_LEN = 500        # 描述
MAX_PROMPT_CATEGORY_LEN = 50     # 分类
MAX_USERNAME_LEN = 50            # 用户名
MAX_PASSWORD_LEN = 128           # 密码

# 禁止在用户输入中出现的危险模式（简单的注入防护）
_DANGEROUS_PATTERNS = [
    "__import__", "exec(", "eval(", "os.system", "subprocess",
    "open(", "compile(", "globals(", "locals(", "getattr(",
]

def validate_input(value: str, max_len: int, field_name: str = "输入") -> str | None:
    """
    校验单个输入字段，返回错误消息（None 表示通过）。
    做三层检查：空值、长度、危险模式。
    """
    if not value or not value.strip():
        return f"{field_name}不能为空"
    if len(value) > max_len:
        return f"{field_name}过长（最大 {max_len} 字符，当前 {len(value)} 字符）"
    # 对所有用户输入做危险模式检查
    lower = value.lower()
    for pattern in _DANGEROUS_PATTERNS:
        if pattern in lower:
            return f"{field_name}包含不被允许的字符模式"
    return None


# ---- API 限流（令牌桶算法） ----
import threading as _threading
import time as _time

# 默认限流配置：每分钟请求数
RATE_LIMIT_DEFAULT = 60          # 通用接口
RATE_LIMIT_QUERY = 30            # 查询接口（较重）
RATE_LIMIT_STREAM = 10           # 流式查询（最重）
RATE_LIMIT_ADMIN = 30            # 管理接口

_RATE_WINDOW = 60                # 时间窗口（秒）


class RateLimiter:
    """基于 IP 的令牌桶限流器，线程安全。"""

    def __init__(self):
        self._buckets: dict[str, dict] = {}
        self._lock = _threading.Lock()

    def is_allowed(self, ip: str, max_req_per_minute: int) -> bool:
        """检查某 IP 是否允许请求，自动补充令牌。新 IP 初始满令牌。"""
        now = _time.time()
        with self._lock:
            bucket = self._buckets.get(ip)
            if bucket is None:
                # 新 IP：初始给满令牌后消耗 1 个
                self._buckets[ip] = {"tokens": max_req_per_minute - 1, "last_time": now}
                return True

            elapsed = now - bucket["last_time"]
            bucket["last_time"] = now
            # 按时间比例补充令牌（最多补满到上限）
            bucket["tokens"] = min(
                max_req_per_minute,
                bucket["tokens"] + elapsed * (max_req_per_minute / _RATE_WINDOW)
            )
            if bucket["tokens"] >= 1.0:
                bucket["tokens"] -= 1.0
                return True
            return False


_rate_limiter = RateLimiter()


def _get_client_ip() -> str:
    """获取真实客户端 IP，支持反向代理。"""
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "127.0.0.1"


def _get_rate_limit_for_route() -> int:
    """根据请求路径返回对应的限流阈值。"""
    path = request.path
    if path == "/api/query":
        return RATE_LIMIT_QUERY
    if path == "/api/query/stream":
        return RATE_LIMIT_STREAM
    if path.startswith("/api/admin/"):
        return RATE_LIMIT_ADMIN
    return RATE_LIMIT_DEFAULT


@app.before_request
def _rate_limit_before_request():
    """Flask 全局前置钩子：对每个 API 请求执行 IP 限流。"""
    if not request.path.startswith("/api/"):
        return None  # 静态页面不限流

    ip = _get_client_ip()
    max_req = _get_rate_limit_for_route()

    if not _rate_limiter.is_allowed(ip, max_req):
        get_audit_logger().log(
            ip=ip, username="anonymous", action="rate_limit_blocked",
            target=request.path, result="blocked",
            detail=f"超过 {max_req} 次/分钟限制"
        )
        return jsonify({
            "error": f"请求过于频繁，请稍后再试（{max_req} 次/分钟）",
            "retry_after": 3,
        }), 429


def _audit_log(action: str, target: str = "",
               result: str = "success", detail: str = "",
               username: str = "anonymous"):
    """便捷方法：记录审计日志，自动获取 IP。"""
    get_audit_logger().log(
        ip=_get_client_ip(), username=username,
        action=action, target=target, result=result, detail=detail,
    )


# ======================================================================
# API 路由
# ======================================================================

@app.route("/")
def index():
    """返回聊天界面（受保护，无 token 会由前端重定向到 /login）"""
    return _HTML_PAGE


@app.route("/login")
def login_page():
    """统一登录入口：登录后由前端按角色跳转（admin→/admin，其他→/）"""
    return _LOGIN_PAGE


@app.route("/api/health")
def health():
    """健康检查"""
    return jsonify({
        "status": "ok",
        "llm": MODEL_NAME,
        "db_path": DB_PATH,
        "role": orchestrator.user_role if orchestrator else "unknown",
    })


@app.route("/api/query", methods=["POST"])
def api_query():
    """
    同步查询（简单模式）：直接运行并返回结果，不流式推送。
    需登录；role/user 一律来自登录态（token），客户端不可伪造。
    """
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "请求体格式错误，需要 JSON"}), 400

    auth_result = _require_auth()
    if auth_result:
        return auth_result

    question = data.get("question", "").strip()
    # role/user/user_id 一律来自登录态，客户端不可伪造（防普通用户提权看受限文档）
    user_role = g.current_user["role"]
    user = g.current_user["username"]
    user_id = g.current_user["user_id"]

    err = validate_input(question, MAX_QUESTION_LEN, "问题")
    if err:
        return jsonify({"error": err}), 400

    result = orchestrator.query(question, user_role=user_role, user=user, user_id=user_id)
    _audit_log("query", target=question[:80], username=user)
    return jsonify({"answer": result, "role": user_role})


@app.route("/api/query/stream", methods=["POST"])
def api_query_stream():
    """
    流式查询 — SSE 实时推送进度。
    启动后台线程执行查询，前端通过 EventSource 接收进度事件。
    """

    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "请求体格式错误，需要 JSON"}), 400

    auth_result = _require_auth()
    if auth_result:
        return auth_result

    question = data.get("question", "").strip()
    # role/user/user_id 一律来自登录态，客户端不可伪造（防普通用户提权看受限文档）
    user_role = g.current_user["role"]
    user = g.current_user["username"]
    user_id = g.current_user["user_id"]

    err = validate_input(question, MAX_QUESTION_LEN, "问题")
    if err:
        return jsonify({"error": err}), 400

    # 创建队列用于接收进度
    progress_queue = queue.Queue()

    # 流式响应的生成器会在请求上下文外执行，因此提前捕获 IP，
    # 避免生成器内部访问 request 对象导致 RuntimeError。
    client_ip = _get_client_ip()

    def generate():
        try:
            # 发送初始事件（建立连接）
            evt = json.dumps({
                "type": "start",
                "question": question,
                "role": user_role,
            }, ensure_ascii=False)
            yield f"data: {evt}\n\n"

            # 在后台线程执行查询，stdout 重定向到进度队列
            output_queue = queue.Queue()
            original_stdout = sys.stdout
            sys.stdout = ProgressWriter(output_queue, original_stdout)

            result_holder = {"answer": None, "error": None}

            def run_query():
                try:
                    result_holder["answer"] = orchestrator.query(
                        question, user_role=user_role, user=user, user_id=user_id
                    )
                except Exception as e:
                    import traceback
                    result_holder["error"] = str(e)
                    output_queue.put({
                        "type": "log",
                        "text": f"[ERROR] {traceback.format_exc()}"
                    })

            thread = threading.Thread(target=run_query, daemon=True)
            thread.start()

            # 轮询队列，把 print 输出转发为 SSE 事件
            finished = False
            while not finished or not output_queue.empty():
                try:
                    msg = output_queue.get(timeout=0.1)
                    evt = json.dumps(msg, ensure_ascii=False)
                    yield f"data: {evt}\n\n"
                except queue.Empty:
                    if not thread.is_alive():
                        finished = True

            # 恢复 stdout
            sys.stdout = original_stdout

            # 审计日志（生成器内无 request 上下文，使用提前捕获的 client_ip）
            if result_holder["error"]:
                get_audit_logger().log(
                    ip=client_ip, username=user_role, action="query_stream",
                    target=question[:80], result="failure",
                    detail=result_holder["error"][:200],
                )
            else:
                get_audit_logger().log(
                    ip=client_ip, username=user_role, action="query_stream",
                    target=question[:80], result="success",
                )

            # 发送最终结果
            if result_holder["error"]:
                yield f"data: {json.dumps({'type': 'error', 'text': result_holder['error']}, ensure_ascii=False)}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'done', 'answer': result_holder['answer'], 'role': orchestrator.user_role}, ensure_ascii=False)}\n\n"

        except GeneratorExit:
            pass

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


@app.route("/api/role", methods=["POST"])
def set_role():
    """
    返回当前登录用户的角色（只读）。

    角色由登录账号决定，不可由客户端自行切换——避免普通用户
    通过直接调用 API 把全局 orchestrator 角色提升为 admin 而访问受限文档。
    """
    auth_result = _require_auth()
    if auth_result:
        return auth_result
    role = g.current_user["role"]
    return jsonify({
        "role": role,
        "description": AccessControlFilter.get_role_description(role),
    })


# ======================================================================
# 内部辅助函数
# ======================================================================

def _get_token_from_request():
    """从 Authorization: Bearer 头 或 rag_token cookie 取 token。"""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return request.cookies.get("rag_token") or ""


def _require_auth():
    """
    校验是否已登录（任意角色）。

    通过返回 None 并把 session 写入 g.current_user；否则返回 (响应, 状态码)。
    token 经 Redis 校验（admin/user 各自独立、带 role）。
    """
    token = _get_token_from_request()
    from prompt_manager import get_auth_manager
    auth = get_auth_manager()
    session = auth.verify_token(token)
    if not session:
        return jsonify({"error": "未登录或登录已过期，请先登录"}), 401
    g.current_user = session
    return None


def _require_admin():
    """校验是否为管理员（role=admin）。通过返回 None，否则返回 (响应, 状态码)。"""
    denied = _require_auth()
    if denied:
        return denied
    if g.current_user.get("role") != ROLE_ADMIN:
        return jsonify({"error": "需要管理员权限"}), 403
    return None


# ======================================================================
# 断点重续 API — 多层记忆核心功能
# ======================================================================

@app.route("/api/tasks/unfinished")
def get_unfinished_tasks():
    """查询当前登录用户的未完成任务（需登录）。"""
    auth_result = _require_auth()
    if auth_result:
        return auth_result
    # session_id 一律按当前登录用户派生，忽略客户端传入，防串会话
    session_id = _derive_session_id(g.current_user["username"], g.current_user["role"])
    if not use_langgraph or not isinstance(orchestrator, LangGraphEngine):
        return jsonify({"tasks": [], "message": "当前引擎不支持断点恢复"})
    tasks = orchestrator.check_unfinished_tasks(session_id, user_id=g.current_user["user_id"])
    return jsonify({"tasks": tasks, "count": len(tasks)})


@app.route("/api/tasks/resume", methods=["POST"])
def resume_task():
    """
    从断点恢复执行指定任务（需登录），以 SSE 流返回。

    复用 /api/query/stream 的 ProgressWriter + 后台线程模式：resume 内部
    要重跑 LangGraph 图（30-90s），改成流式后前端能实时看到进度日志，
    并能用刚加的 ⏹ 中断按钮中止（AbortController）。
    """
    auth_result = _require_auth()
    if auth_result:
        return auth_result

    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "请求体格式错误，需要 JSON"}), 400

    task_id = (data.get("task_id") or "").strip()
    if not task_id:
        return jsonify({"error": "缺少 task_id 参数"}), 400

    if not use_langgraph or not isinstance(orchestrator, LangGraphEngine):
        return jsonify({"error": "当前引擎不支持断点恢复"}), 400

    # session_id 一律按当前登录用户派生，忽略客户端传入，防串会话
    session_id = _derive_session_id(g.current_user["username"], g.current_user["role"])
    user_id = g.current_user["user_id"]

    # 提前捕获 IP（生成器在请求上下文外执行）
    client_ip = _get_client_ip()
    # 提前捕获用户名（生成器在请求上下文外执行，不能在 generate() 内访问 g.current_user）
    current_username = g.current_user["username"]

    def generate():
        output_queue = queue.Queue()
        original_stdout = sys.stdout
        sys.stdout = ProgressWriter(output_queue, original_stdout)

        result_holder = {"answer": None, "error": None}

        def run_resume():
            try:
                result_holder["answer"] = orchestrator.resume_task(
                    task_id, session_id, user_id=user_id
                )
            except Exception as e:
                import traceback as _tb
                result_holder["error"] = str(e)
                output_queue.put({
                    "type": "log",
                    "text": f"[ERROR] {_tb.format_exc()}",
                })

        # 推送首事件
        evt = json.dumps({"type": "start", "task_id": task_id}, ensure_ascii=False)
        yield f"data: {evt}\n\n"

        thread = threading.Thread(target=run_resume, daemon=True)
        thread.start()

        # 轮询 stdout 队列，转发为 SSE
        finished = False
        while not finished or not output_queue.empty():
            try:
                msg = output_queue.get(timeout=0.1)
                evt = json.dumps(msg, ensure_ascii=False)
                yield f"data: {evt}\n\n"
            except queue.Empty:
                if not thread.is_alive():
                    finished = True

        sys.stdout = original_stdout

        if result_holder["error"]:
            get_audit_logger().log(
                ip=client_ip, username=current_username,
                action="task_resume", target=task_id, result="failure",
                detail=result_holder["error"][:200],
            )
            yield f"data: {json.dumps({'type': 'error', 'text': result_holder['error']}, ensure_ascii=False)}\n\n"
        else:
            get_audit_logger().log(
                ip=client_ip, username=current_username,
                action="task_resume", target=task_id, result="success",
            )
            yield f"data: {json.dumps({'type': 'done', 'answer': result_holder['answer']}, ensure_ascii=False)}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


# ======================================================================
# 管理后台 API — 提示词工程管理 + 用户认证
# ======================================================================

# ---- 认证相关 ----

@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    """管理员登录"""
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "请求体格式错误，需要 JSON"}), 400

    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    for val, limit, lbl in [
        (username, MAX_USERNAME_LEN, "用户名"),
        (password, MAX_PASSWORD_LEN, "密码"),
    ]:
        err = validate_input(val, limit, lbl)
        if err:
            _audit_log("login", target=username, result="failure", detail=err)
            return jsonify({"error": err}), 400

    from prompt_manager import AuthManager, get_auth_manager
    auth = get_auth_manager()
    user = auth.login(username, password)

    if user:
        _audit_log("login", target=username, result="success", username=username)
        resp = jsonify({
            "success": True,
            "user": {
                "username": user["username"],
                "display_name": user["display_name"],
                "role": user["role"],
            },
            "token": user["token"],
        })
        # 同时写入 HttpOnly Cookie，浏览器后续请求自动携带（双保险，前端也可用 localStorage）
        resp.set_cookie("rag_token", user["token"], httponly=True,
                        max_age=auth.TOKEN_TTL, samesite="Lax", path="/")
        return resp
    else:
        _audit_log("login", target=username, result="failure",
                   detail="用户名或密码错误", username=username)
        return jsonify({"success": False, "error": "用户名或密码错误"}), 401


@app.route("/api/admin/me", methods=["GET"])
def admin_me():
    """
    获取当前登录管理员信息（需 admin 角色）。

    前端页面刷新后，用 token 调用此接口恢复登录状态，返回真实登录账号。
    """
    auth_result = _require_admin()
    if auth_result:
        return auth_result
    return jsonify({
        "username": g.current_user["username"],
        "display_name": g.current_user.get("display_name"),
        "role": g.current_user["role"],
    })


# ---- 统一登录 / 登出 / 当前用户（admin / user 通用） ----
@app.route("/api/login", methods=["POST"])
def api_login():
    """统一登录入口：admin_users 表中任意启用账号（role=admin/user 均可）。

    成功后 token 写入 Redis（带 TTL），同时下发 HttpOnly Cookie，
    浏览器后续请求自动携带；前端也可把 token 存 localStorage 自行携带。
    """
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "请求体格式错误，需要 JSON"}), 400

    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    for val, limit, lbl in [
        (username, MAX_USERNAME_LEN, "用户名"),
        (password, MAX_PASSWORD_LEN, "密码"),
    ]:
        err = validate_input(val, limit, lbl)
        if err:
            _audit_log("login", target=username, result="failure", detail=err)
            return jsonify({"error": err}), 400

    from prompt_manager import get_auth_manager
    auth = get_auth_manager()
    user = auth.login(username, password)
    if user:
        _audit_log("login", target=username, result="success", username=username)
        resp = jsonify({
            "success": True,
            "token": user["token"],
            "user": {
                "username": user["username"],
                "display_name": user["display_name"],
                "role": user["role"],
            },
        })
        resp.set_cookie("rag_token", user["token"], httponly=True,
                        max_age=auth.TOKEN_TTL, samesite="Lax", path="/")
        return resp
    _audit_log("login", target=username, result="failure",
               detail="用户名或密码错误", username=username)
    return jsonify({"success": False, "error": "用户名或密码错误"}), 401


@app.route("/api/logout", methods=["POST"])
def api_logout():
    """登出：删除 Redis 中的 token，并清除 Cookie。"""
    token = _get_token_from_request()
    if token:
        from prompt_manager import get_auth_manager
        get_auth_manager().logout(token)
    resp = jsonify({"success": True, "message": "已登出"})
    resp.delete_cookie("rag_token", path="/")
    return resp


@app.route("/api/me", methods=["GET"])
def api_me():
    """返回当前登录用户信息（任意已登录角色）。前端据此恢复会话、判断特权。"""
    auth_result = _require_auth()
    if auth_result:
        return auth_result
    return jsonify({
        "username": g.current_user["username"],
        "display_name": g.current_user.get("display_name"),
        "role": g.current_user["role"],
    })


def _get_memory_store():
    """兼容两种编排器，取到统一的 MySQLMemoryStore 实例。"""
    o = orchestrator
    if o is None:
        return None
    ms = getattr(o, "memory_store", None)
    if ms is None and getattr(o, "app", None) is not None:
        ms = getattr(o.app, "memory_store", None)
    return ms


@app.route("/api/history", methods=["GET"])
def get_history():
    """
    返回当前登录用户的历史对话，供前端刷新/重登后恢复显示。

    历史已落库（chat_messages 表，按 username + session_id 隔离），
    这里只负责按当前用户派生 session_id 拉取并回传。
    system 角色的摘要（压缩产物，仅供 LLM 续聊）不渲染给用户。
    """
    auth_result = _require_auth()
    if auth_result:
        return auth_result
    user = g.current_user
    session_id = _derive_session_id(user["username"], user["role"])
    ms = _get_memory_store()
    if ms is None:
        return jsonify({"session_id": session_id, "messages": []})
    try:
        raw = ms.load_messages(session_id, user_id=user["user_id"])
        visible = [
            {"role": m["role"], "content": m["content"]}
            for m in raw
            if m.get("role") in ("user", "assistant")
        ]
        return jsonify({"session_id": session_id, "messages": visible})
    except Exception as e:
        return jsonify({"session_id": session_id, "messages": [], "error": str(e)})


@app.route("/api/change-password", methods=["POST"])
@app.route("/api/admin/change-password", methods=["POST"])
def change_password_api():
    """
    修改密码（需登录，所有角色通用；只能改当前登录账号自己的密码）。

    普通用户（聊天页）和 admin（后台）共用此接口。任意登录账号都能改
    自己的密码，不能改他人——username 一律取自 token 中的 g.current_user，
    忽略客户端传入，防越权。
    """
    auth_result = _require_auth()
    if auth_result:
        return auth_result
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "请求体格式错误，需要 JSON"}), 400

    # 只能改当前登录账号，忽略客户端传的 username，防改他人密码
    username = g.current_user["username"]
    old_password = data.get("old_password", "")
    new_password = data.get("new_password", "")

    for val, limit, lbl in [
        (new_password, MAX_PASSWORD_LEN, "新密码"),
        (old_password, MAX_PASSWORD_LEN, "原密码"),
    ]:
        err = validate_input(val, limit, lbl)
        if err:
            return jsonify({"error": err}), 400

    if len(new_password) < 6:
        return jsonify({"error": "新密码至少6位"}), 400

    from prompt_manager import get_auth_manager
    auth = get_auth_manager()
    if auth.change_password(username, old_password, new_password):
        _audit_log("change_password", target=username, result="success", username=username)
        return jsonify({"success": True, "message": "密码修改成功"})
    else:
        _audit_log("change_password", target=username, result="failure",
                   detail="原密码错误", username=username)
        return jsonify({"error": "原密码错误或用户不存在"}), 400


# ---- 提示词管理 ----

@app.route("/api/admin/prompts")
def admin_list_prompts():
    """列出所有提示词"""
    from prompt_manager import get_prompt_manager
    pm = get_prompt_manager()
    category = request.args.get("category", "")
    prompts = pm.list_prompts(category if category else None)
    return jsonify({"prompts": prompts, "total": len(prompts)})


@app.route("/api/admin/prompts/<name>")
def admin_get_prompt(name):
    """获取单个提示词"""
    from prompt_manager import get_prompt_manager
    pm = get_prompt_manager()
    prompt = pm.get_prompt(name)
    return jsonify(prompt)


@app.route("/api/admin/prompts", methods=["POST"])
def admin_save_prompt():
    """保存/更新提示词"""
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "请求体格式错误，需要 JSON"}), 400

    name = data.get("name", "").strip()
    system = data.get("system", "")
    user_template = data.get("user_template", "")
    display_name = data.get("display_name", "")
    description = data.get("description", "")
    category = data.get("category", "general")

    # 逐字段校验
    for val, limit, lbl in [
        (name, MAX_PROMPT_NAME_LEN, "提示词名称"),
        (system, MAX_PROMPT_SYSTEM_LEN, "系统提示词"),
        (user_template, MAX_PROMPT_USER_TEMPLATE_LEN, "用户模板"),
        (display_name, MAX_PROMPT_DISPLAY_LEN, "显示名称"),
        (description, MAX_PROMPT_DESC_LEN, "描述"),
        (category, MAX_PROMPT_CATEGORY_LEN, "分类"),
    ]:
        err = validate_input(val, limit, lbl)
        if err:
            return jsonify({"error": err}), 400

    from prompt_manager import get_prompt_manager
    pm = get_prompt_manager()
    ok = pm.save_prompt(
        name=name, system=system, user_template=user_template,
        display_name=display_name, description=description, category=category,
    )
    if ok:
        _audit_log("save_prompt", target=name, username="admin")
        return jsonify({"success": True, "message": f"提示词 '{name}' 已保存"})
    else:
        _audit_log("save_prompt", target=name, result="failure",
                   detail="数据库不可用", username="admin")
        return jsonify({"error": "保存失败，数据库不可用"}), 500


@app.route("/api/admin/prompts/<name>", methods=["DELETE"])
def admin_delete_prompt(name):
    """删除提示词"""
    from prompt_manager import get_prompt_manager
    pm = get_prompt_manager()
    ok = pm.delete_prompt(name)
    if ok:
        _audit_log("delete_prompt", target=name, username="admin")
        return jsonify({"success": True, "message": f"提示词 '{name}' 已删除"})
    else:
        _audit_log("delete_prompt", target=name, result="failure",
                   detail="不可删除", username="admin")
        return jsonify({"error": "删除失败（系统内置提示词不可删除）"}), 400


@app.route("/api/admin/prompts/<name>/toggle", methods=["POST"])
def admin_toggle_prompt(name):
    """启用/禁用提示词"""
    data = request.get_json(force=True)
    active = data.get("active", True)
    from prompt_manager import get_prompt_manager
    pm = get_prompt_manager()
    ok = pm.set_active(name, active)
    if ok:
        return jsonify({"success": True, "active": active})
    else:
        return jsonify({"error": "操作失败"}), 500


@app.route("/api/admin/prompts/import-defaults", methods=["POST"])
def admin_import_defaults():
    """导入默认提示词（覆盖已有记录）"""
    from prompt_manager import get_prompt_manager
    pm = get_prompt_manager()
    count = pm.import_defaults()
    _audit_log("import_defaults", target=f"共 {count} 个提示词", username="admin")
    return jsonify({"success": True, "imported": count})


@app.route("/api/admin/categories")
def admin_categories():
    """获取提示词分类列表"""
    from prompt_manager import get_prompt_manager
    pm = get_prompt_manager()
    return jsonify({"categories": pm.get_categories()})


# ======================================================================
# Token 用量查询 API —— 让「网关记了多少 token」在网页上直接看得见
# ======================================================================

def _gateway_or_none():
    """拿全局网关单例；网关未启用时返回 None，接口降级而不是 500。"""
    try:
        from llm_gateway import get_gateway
        return get_gateway()
    except Exception:
        return None


def _range_start_ts(rng: str) -> float:
    """把前端传的 today / 7d / 30d / all 翻译成起始时间戳。"""
    now = time.time()
    if rng == "today":
        lt = time.localtime(now)
        return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday,
                            0, 0, 0, 0, 0, -1))
    if rng == "7d":
        return now - 7 * 86400
    if rng == "30d":
        return now - 30 * 86400
    return 0.0


def _summarize(rows):
    """对明细做一次聚合，供前端统计卡直接渲染。"""
    p = sum(r["prompt_tokens"] for r in rows)
    c = sum(r["completion_tokens"] for r in rows)
    return {
        "calls": len(rows),
        "prompt_tokens": p,
        "completion_tokens": c,
        "total_tokens": p + c,
        "cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
        "avg_latency_s": round(
            sum(r["latency_s"] for r in rows) / len(rows), 3) if rows else 0.0,
    }


@app.route("/api/usage/me")
def api_usage_me():
    """
    当前登录用户的 token 用量（网页「我的用量」面板数据源）。

    需登录；只返回当前登录账号的用量（忽略客户端 user 参数，防越权查询他人用量）。
    """
    auth_result = _require_auth()
    if auth_result:
        return auth_result

    gw = _gateway_or_none()
    if gw is None:
        return jsonify({"error": "LLM 网关未启用，无法查询用量"}), 503

    user = g.current_user["username"]
    rng = request.args.get("range", "all")
    try:
        limit = max(1, min(int(request.args.get("limit", 50)), 200))
    except (TypeError, ValueError):
        limit = 50

    lifetime = gw.user_usage(user)          # 全周期累计（不受 range 影响）
    if rng == "all":
        rows = gw.usage_log(user, limit)
        window = {
            "calls": lifetime["calls"],
            "prompt_tokens": lifetime["prompt_tokens"],
            "completion_tokens": lifetime["completion_tokens"],
            "total_tokens": lifetime["total_tokens"],
            "cost_usd": lifetime["cost_usd"],
            "avg_latency_s": round(
                sum(r["latency_s"] for r in rows) / len(rows), 3) if rows else 0.0,
        }
    else:
        all_rows = gw.usage_range(_range_start_ts(rng), time.time(), user)
        window = _summarize(all_rows)
        rows = all_rows[:limit]

    m = gw.metrics()
    return jsonify({
        "user": user,
        "range": rng,
        "lifetime": lifetime,
        "window": window,
        "rows": rows,
        "persisted": m.get("usage_persisted", False),
        "db": m.get("usage_db", ""),
    })


@app.route("/api/admin/usage/top")
def api_admin_usage_top():
    """全用户 token 排行 + 最近明细（管理后台看板，需要管理员 Token）。"""
    auth_result = _require_admin()
    if auth_result:
        return auth_result

    gw = _gateway_or_none()
    if gw is None:
        return jsonify({"error": "LLM 网关未启用，无法查询用量"}), 503

    rng = request.args.get("range", "all")
    try:
        limit = max(1, min(int(request.args.get("limit", 50)), 200))
    except (TypeError, ValueError):
        limit = 50

    if rng == "all":
        users = gw.top_users(50)
        rows = gw.usage_log(None, limit)
        window = {
            "calls": sum(u["calls"] for u in users),
            "prompt_tokens": sum(u["prompt_tokens"] for u in users),
            "completion_tokens": sum(u["completion_tokens"] for u in users),
            "total_tokens": sum(u["total_tokens"] for u in users),
            "cost_usd": round(sum(u["cost_usd"] for u in users), 6),
            "avg_latency_s": round(
                sum(r["latency_s"] for r in rows) / len(rows), 3) if rows else 0.0,
        }
    else:
        all_rows = gw.usage_range(_range_start_ts(rng), time.time(), None)
        agg = {}
        for r in all_rows:
            a = agg.setdefault(r["user"], {
                "user": r["user"], "calls": 0, "prompt_tokens": 0,
                "completion_tokens": 0, "total_tokens": 0,
                "cost_usd": 0.0, "last_active_ts": 0.0})
            a["calls"] += 1
            a["prompt_tokens"] += r["prompt_tokens"]
            a["completion_tokens"] += r["completion_tokens"]
            a["total_tokens"] += r["total_tokens"]
            a["cost_usd"] = round(a["cost_usd"] + r["cost_usd"], 6)
            a["last_active_ts"] = max(a["last_active_ts"], r["ts"])
        users = sorted(agg.values(), key=lambda x: x["total_tokens"], reverse=True)
        window = _summarize(all_rows)
        rows = all_rows[:limit]

    m = gw.metrics()
    return jsonify({
        "range": rng,
        "users": users,
        "window": window,
        "rows": rows,
        "persisted": m.get("usage_persisted", False),
        "db": m.get("usage_db", ""),
    })


# ---- 管理后台页面 ----

@app.route("/admin")
def admin_page():
    """管理后台页面"""
    return _ADMIN_PAGE


# ======================================================================
# HTML 页面（内嵌单文件模板）
# ======================================================================

_ADMIN_PAGE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>系统管理 - RAG Agent</title>
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  :root{
    --bg:#f5f6fa;--surface:#fff;--border:#e2e5ed;
    --text:#1a1a2e;--text-2:#6b7280;--text-3:#9ca3af;
    --primary:#2563eb;--primary-light:#dbeafe;--primary-hover:#1d4ed8;
    --danger:#dc2626;--danger-light:#fef2f2;
    --success:#16a34a;--success-light:#f0fdf4;
    --warning:#d97706;--warning-light:#fffbeb;
    --shadow:0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.06);
    --shadow-lg:0 10px 25px rgba(0,0,0,.1);
    --radius:12px;--radius-sm:8px;
  }
  html,body{height:100%}
  body{
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans SC",sans-serif;
    background:var(--bg);color:var(--text);
  }

  /* ===== Login Page ===== */
  .login-page{
    display:flex;align-items:center;justify-content:center;min-height:100vh;
    background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
  }
  .login-card{
    background:var(--surface);border-radius:var(--radius);padding:40px;width:380px;
    box-shadow:var(--shadow-lg);
  }
  .login-card h2{text-align:center;margin-bottom:8px;font-size:24px}
  .login-card .sub{text-align:center;color:var(--text-2);margin-bottom:24px;font-size:14px}
  .form-group{margin-bottom:16px}
  .form-group label{display:block;margin-bottom:6px;font-size:14px;font-weight:500;color:var(--text-2)}
  .form-group input{
    width:100%;padding:10px 14px;border:1.5px solid var(--border);border-radius:var(--radius-sm);
    font-size:14px;outline:none;transition:border-color .2s;
  }
  .form-group input:focus{border-color:var(--primary)}
  .btn{
    display:inline-flex;align-items:center;justify-content:center;gap:6px;
    padding:10px 20px;border-radius:var(--radius-sm);font-size:14px;font-weight:500;
    cursor:pointer;border:none;transition:all .2s;
  }
  .btn-primary{background:var(--primary);color:#fff;width:100%}
  .btn-primary:hover{background:var(--primary-hover)}
  .btn-sm{padding:6px 12px;font-size:12px}
  .btn-danger{background:var(--danger);color:#fff}
  .btn-danger:hover{opacity:.9}
  .btn-outline{background:var(--surface);border:1.5px solid var(--border);color:var(--text)}
  .btn-outline:hover{background:var(--bg)}
  .error-msg{color:var(--danger);font-size:13px;text-align:center;margin-top:8px;display:none}
  .hidden{display:none !important}

  /* ===== App Layout ===== */
  .app-page{display:flex;flex-direction:column;height:100vh}
  .app-header{
    background:var(--surface);border-bottom:1px solid var(--border);
    padding:12px 24px;display:flex;align-items:center;justify-content:space-between;
    box-shadow:var(--shadow);
  }
  .app-header h1{font-size:18px;font-weight:600;display:flex;align-items:center;gap:8px}
  .user-info{display:flex;align-items:center;gap:12px;font-size:13px;color:var(--text-2)}
  .user-info .name{font-weight:500;color:var(--text)}

  /* ===== Tabs ===== */
  .tabs{display:flex;gap:0;border-bottom:2px solid var(--border);background:var(--surface);padding:0 24px}
  .tab{
    padding:12px 24px;font-size:14px;font-weight:500;color:var(--text-2);
    cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-2px;
    transition:all .2s;
  }
  .tab:hover{color:var(--text)}
  .tab.active{color:var(--primary);border-bottom-color:var(--primary)}

  /* ===== Content ===== */
  .app-content{flex:1;overflow:hidden;display:flex;flex-direction:column}
  .tab-panel{flex:1;overflow:hidden;padding:24px;display:none;flex-direction:column}
  .tab-panel.active{display:flex;flex-direction:column}

  /* ===== Prompt List ===== */
  .toolbar{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;gap:12px;flex-wrap:wrap}

  /* ===== Token 用量看板 ===== */
  .toolbar h2{font-size:17px;font-weight:600}
  .toolbar p.sub{font-size:12px;color:var(--text-2);margin-top:2px}
  .range-btn{
    padding:5px 14px;border-radius:16px;font-size:12px;cursor:pointer;
    border:1.5px solid var(--border);background:var(--surface);color:var(--text-2);
    transition:all .15s;user-select:none
  }
  .range-btn:hover{border-color:var(--primary);color:var(--primary)}
  .range-btn.active{background:var(--primary);border-color:var(--primary);color:#fff}
  .stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
  .stat-card{
    background:var(--surface);border:1px solid var(--border);
    border-radius:var(--radius);padding:14px 16px
  }
  .stat-card .label{font-size:11px;color:var(--text-2);margin-bottom:6px}
  .stat-card .value{font-size:22px;font-weight:700;font-family:var(--font-mono,monospace)}
  .stat-card .sub{font-size:11px;color:var(--text-3);margin-top:4px}
  .stat-card.primary{background:var(--primary-light);border-color:var(--primary)}
  .stat-card.primary .value{color:var(--primary)}
  .stat-card.success{background:#f0fdf4;border-color:#16a34a}
  .stat-card.success .value{color:#16a34a}
  .stat-card.warning{background:#fffbeb;border-color:#d97706}
  .stat-card.warning .value{color:#d97706}
  .usage-section-title{font-size:13px;font-weight:600;margin:16px 0 10px;color:var(--text-2)}
  .usage-table{width:100%;border-collapse:collapse;font-size:12px;background:var(--surface)}
  .usage-table th{
    text-align:left;padding:8px 10px;background:#f5f6fa;color:var(--text-2);
    font-weight:600;border-bottom:1px solid var(--border);white-space:nowrap;
    position:sticky;top:0;z-index:1
  }
  .usage-table td{
    padding:8px 10px;border-bottom:1px solid var(--border);
    font-family:var(--font-mono,monospace);white-space:nowrap
  }
  .usage-table tr:hover td{background:#f5f6fa}
  .usage-table .num{text-align:right}
  .tag-task{
    display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;
    background:var(--primary-light);color:var(--primary)
  }
  .rank-badge{
    display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;
    border-radius:50%;background:var(--border);color:var(--text-2);font-size:11px;font-weight:700
  }
  .rank-badge.top1{background:#fef3c7;color:#b45309}
  .usage-empty{text-align:center;padding:36px 12px;color:var(--text-3);font-size:13px}
  .table-wrap{max-height:340px;overflow:auto;border:1px solid var(--border);border-radius:var(--radius-sm)}
  .search-box{
    padding:8px 14px;border:1.5px solid var(--border);border-radius:var(--radius-sm);
    font-size:13px;outline:none;width:240px
  }
  .search-box:focus{border-color:var(--primary)}
  .filter-select{
    padding:8px 12px;border:1.5px solid var(--border);border-radius:var(--radius-sm);
    font-size:13px;outline:none;background:var(--surface);
  }
  .prompt-list{display:flex;flex-direction:column;gap:8px}
  .prompt-card{
    background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-sm);
    padding:16px;cursor:pointer;transition:all .2s;
  }
  .prompt-card:hover{border-color:var(--primary);box-shadow:var(--shadow)}
  .prompt-card .top{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}
  .prompt-card .name{font-weight:600;font-size:15px}
  .prompt-card .name .tag{
    font-size:11px;padding:2px 8px;border-radius:10px;margin-left:8px;
    background:var(--primary-light);color:var(--primary);font-weight:500;
  }
  .prompt-card .desc{color:var(--text-2);font-size:13px;margin-bottom:6px}
  .prompt-card .meta{display:flex;gap:16px;font-size:11px;color:var(--text-3)}
  .prompt-card.inactive{opacity:.5}

  /* ===== Editor Modal ===== */
  .modal-overlay{
    position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.4);
    display:flex;align-items:center;justify-content:center;z-index:100;
  }
  .modal{
    background:var(--surface);border-radius:var(--radius);width:720px;max-height:85vh;
    box-shadow:var(--shadow-lg);display:flex;flex-direction:column;
  }
  .modal-header{
    padding:16px 24px;border-bottom:1px solid var(--border);
    display:flex;align-items:center;justify-content:space-between;
  }
  .modal-header h3{font-size:17px}
  .modal-body{flex:1;overflow-y:auto;padding:24px}
  .modal-footer{
    padding:16px 24px;border-top:1px solid var(--border);
    display:flex;justify-content:flex-end;gap:8px;
  }
  .field{margin-bottom:16px}
  .field label{display:block;margin-bottom:6px;font-size:13px;font-weight:500;color:var(--text-2)}
  .field input,.field select,.field textarea{
    width:100%;padding:10px 14px;border:1.5px solid var(--border);border-radius:var(--radius-sm);
    font-size:14px;outline:none;font-family:inherit;
  }
  .field textarea{min-height:120px;resize:vertical;font-size:13px}
  .field input:focus,.field select:focus,.field textarea:focus{border-color:var(--primary)}

  /* ===== Q&A Area ===== */
  .qa-container{
    max-width:900px;margin:0 auto;display:flex;flex-direction:column;height:100%;
    background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
    box-shadow:var(--shadow);overflow:hidden
  }
  .qa-header{
    padding:14px 20px;border-bottom:1px solid var(--border);
    background:linear-gradient(90deg,var(--primary-light),transparent);
    display:flex;align-items:center;justify-content:space-between
  }
  .qa-header-title{display:flex;align-items:center;gap:10px;font-weight:600;color:var(--text-1)}
  .qa-header-badge{
    font-size:11px;padding:3px 10px;border-radius:999px;background:var(--primary);color:#fff;font-weight:500
  }
  .qa-messages{flex:1;overflow-y:auto;padding:20px;background:var(--bg)}
  .qa-empty{
    height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;
    text-align:center;color:var(--text-2);padding:40px 20px
  }
  .qa-empty-icon{
    width:72px;height:72px;border-radius:50%;background:var(--primary-light);color:var(--primary);
    display:flex;align-items:center;justify-content:center;font-size:32px;margin-bottom:16px
  }
  .qa-empty h3{font-size:18px;color:var(--text-1);margin-bottom:8px;font-weight:600}
  .qa-empty p{max-width:420px;line-height:1.6;margin-bottom:20px;font-size:14px}
  .qa-tips{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;max-width:480px}
  .qa-tip{
    padding:8px 14px;border-radius:var(--radius-sm);background:var(--surface);border:1px solid var(--border);
    font-size:13px;color:var(--text-2);cursor:pointer;transition:all .2s
  }
  .qa-tip:hover{border-color:var(--primary);color:var(--primary);background:var(--primary-light)}
  .qa-msg{margin-bottom:16px;display:flex;gap:10px}
  .qa-msg.user{flex-direction:row-reverse}
  .qa-msg .avatar{
    width:36px;height:36px;border-radius:50%;display:flex;align-items:center;
    justify-content:center;font-size:16px;flex-shrink:0;
  }
  .qa-msg.assistant .avatar{background:var(--primary-light);color:var(--primary)}
  .qa-msg.user .avatar{background:#e0e7ff;color:#6366f1}
  .qa-bubble{
    max-width:75%;padding:12px 16px;border-radius:var(--radius-sm);
    font-size:14px;line-height:1.6;white-space:pre-wrap;
  }
  .qa-msg.assistant .qa-bubble{background:var(--surface);border:1px solid var(--border)}
  .qa-msg.user .qa-bubble{background:var(--primary);color:#fff}
  .qa-progress-head{font-size:13px;font-weight:600;color:var(--primary);margin-bottom:8px;display:flex;align-items:center;gap:6px}
  .qa-progress-head .spinner{display:inline-block;width:12px;height:12px;border:2px solid var(--primary-light);border-top-color:var(--primary);border-radius:50%;animation:qaSpin .8s linear infinite}
  .qa-progress-logs{max-height:240px;overflow-y:auto;font-size:12px;color:var(--text-2);line-height:1.6;font-family:ui-monospace,Menlo,Consolas,monospace}
  .qa-progress-logs .log-line{padding:2px 0;border-bottom:1px dashed var(--border)}
  .qa-progress-logs .log-line:last-child{border-bottom:none}
  @keyframes qaSpin{to{transform:rotate(360deg)}}
  .qa-input-area{
    display:flex;gap:10px;padding:16px 20px;border-top:1px solid var(--border);background:var(--surface)
  }
  .qa-input{
    flex:1;padding:12px 16px;border:1.5px solid var(--border);border-radius:var(--radius-sm);
    font-size:14px;outline:none;resize:none;font-family:inherit;min-height:46px
  }
  .qa-input:focus{border-color:var(--primary);box-shadow:0 0 0 3px var(--primary-light)}
  .qa-send-btn{
    padding:0 24px;border-radius:var(--radius-sm);background:var(--primary);color:#fff;
    border:none;font-size:14px;font-weight:500;cursor:pointer;transition:background .2s
  }
  .qa-send-btn:hover{background:var(--primary-dark)}
  .qa-send-btn:disabled{background:var(--text-3);cursor:not-allowed}

  /* ===== Toast ===== */
  .toast{
    position:fixed;top:20px;right:20px;z-index:200;
    padding:12px 20px;border-radius:var(--radius-sm);font-size:14px;
    box-shadow:var(--shadow-lg);animation:slideIn .3s;max-width:360px;
  }
  .toast.success{background:var(--success);color:#fff}
  .toast.error{background:var(--danger);color:#fff}
  @keyframes slideIn{from{transform:translateX(100%);opacity:0}to{transform:translateX(0);opacity:1}}

  /* ===== 修改密码模态框（admin 后台用） ===== */
  .modal-mask{position:fixed;inset:0;background:rgba(15,23,42,.45);display:none;align-items:center;justify-content:center;z-index:9999;animation:fadeIn .15s}
  .modal-mask.show{display:flex}
  .modal{background:var(--surface);border-radius:14px;box-shadow:var(--shadow-lg);width:100%;max-width:420px;animation:popIn .2s}
  .modal-head{display:flex;align-items:center;justify-content:space-between;padding:18px 22px 12px;border-bottom:1px solid var(--border)}
  .modal-head h3{font-size:17px;font-weight:600;display:flex;align-items:center;gap:8px;margin:0}
  .modal-close{background:none;border:none;cursor:pointer;color:var(--text-3);font-size:22px;line-height:1;padding:2px 6px;border-radius:6px}
  .modal-close:hover{background:var(--bg);color:var(--text)}
  .modal-body{padding:18px 22px 22px}
  .pwd-row{display:flex;flex-direction:column;gap:6px;margin-bottom:12px}
  .pwd-row label{font-size:13px;font-weight:500;color:var(--text-2)}
  .pwd-row input{padding:10px 12px;border:1.5px solid var(--border);border-radius:8px;font-size:14px;font-family:inherit;background:var(--bg);color:var(--text);transition:border-color .15s}
  .pwd-row input:focus{outline:none;border-color:var(--primary)}
  .pwd-msg{font-size:13px;min-height:18px;margin-bottom:8px}
  .pwd-msg.err{color:var(--danger)}
  .pwd-msg.ok{color:var(--success)}
  .pwd-actions{display:flex;gap:10px;justify-content:flex-end;margin-top:6px}
  .pwd-actions .btn{min-width:88px}
  @keyframes fadeIn{from{opacity:0}to{opacity:1}}
  @keyframes popIn{from{transform:scale(.95);opacity:0}to{transform:scale(1);opacity:1}}
  /* ===== 在线问答：断点重续横条 + 中断按钮 ===== */
  .qa-resume-bar{
    display:flex;align-items:center;gap:10px;
    margin:0 14px 8px;padding:10px 14px;
    background:linear-gradient(90deg,#fff7e6,#fffbe9);
    border:1px solid #ffd591;border-left:4px solid #fa8c16;border-radius:10px;
    font-size:13px;color:#614700;box-shadow:0 1px 4px rgba(250,140,22,.12);
  }
  .qa-resume-bar .resume-icon{font-size:16px}
  .qa-resume-bar .resume-text{flex:1;line-height:1.4}
  .qa-resume-bar .resume-text b{color:#ad4e00}
  .qa-resume-bar .btn-resume{
    border:none;background:#fa8c16;color:#fff;font-weight:600;
    padding:5px 16px;border-radius:7px;cursor:pointer;font-size:13px
  }
  .qa-resume-bar .btn-resume:hover{background:#d8760b}
  .qa-resume-bar .btn-resume-dismiss{
    border:1px solid #d9d9d9;background:#fff;color:#595959;
    padding:5px 12px;border-radius:7px;cursor:pointer;font-size:13px
  }
  .qa-resume-bar .btn-resume-dismiss:hover{background:#f5f5f5}
  .qa-send-btn.stopping{background:var(--danger);color:#fff}
  .qa-send-btn.stopping:hover{background:var(--danger-hover,#dc2626)}
</style>
</head>
<body>

<!-- 本页为受保护页面：仅 admin 角色可访问；无 token / 失效 / 非 admin 时由脚本重定向 -->
<script>
(function(){
  var t = localStorage.getItem('rag_token');
  if(!t){ location.replace('/login'); return; }
  document.documentElement.style.visibility = 'hidden';
})();
</script>

<!-- ===== App Page ===== -->
<div id="appPage" class="app-page hidden">
  <div class="app-header">
    <h1>⚙️ RAG Agent 系统管理</h1>
    <div class="user-info">
      <span>👤</span>
      <span class="name" id="displayName"></span>
      <a href="/admin" style="color:var(--primary);text-decoration:none;font-size:13px">🔧 提示词管理</a>
      <button class="btn btn-sm btn-outline" onclick="changePassword()">🔑 修改密码</button>
      <button class="btn btn-sm btn-outline" onclick="doLogout()">退出</button>
    </div>
  </div>
  <div class="tabs">
    <div class="tab active" data-tab="prompts" onclick="switchTab('prompts')">📝 提示词管理</div>
    <div class="tab" data-tab="qa" onclick="switchTab('qa')">💬 在线问答</div>
    <div class="tab" data-tab="usage" onclick="switchTab('usage')">📊 Token 用量</div>
  </div>
  <div class="app-content">
    <!-- Prompt Management Tab -->
    <div id="tabPrompts" class="tab-panel active">
      <div class="toolbar">
        <div style="display:flex;gap:8px;align-items:center">
          <input class="search-box" id="searchInput" placeholder="搜索提示词..." oninput="filterPrompts()">
          <select class="filter-select" id="filterCategory" onchange="filterPrompts()">
            <option value="">全部分类</option>
          </select>
        </div>
        <div style="display:flex;gap:8px">
          <button class="btn btn-sm btn-outline" onclick="importDefaults()">🔄 恢复默认</button>
        </div>
      </div>
      <div class="prompt-list" id="promptList"></div>
    </div>
    <!-- Q&A Tab -->
    <div id="tabQa" class="tab-panel">
      <div class="qa-container">
        <div class="qa-header">
          <div class="qa-header-title">
            <span>💬</span>
            <span>管理员在线问答</span>
          </div>
          <span class="qa-header-badge">admin 权限 · 可访问全部文档</span>
        </div>
        <div class="qa-messages" id="qaMessages">
          <div class="qa-empty" id="qaEmpty">
            <div class="qa-empty-icon">🤖</div>
            <h3>管理员问答测试台</h3>
            <p>在这里以 admin 权限测试知识库问答效果，可访问公开文档与受限文档。修改提示词后可立即来此验证。</p>
            <div class="qa-tips">
              <span class="qa-tip" onclick="setQAQuestion('项目支持哪些文档格式？')">📄 项目支持哪些文档格式？</span>
              <span class="qa-tip" onclick="setQAQuestion('断点重续是怎么实现的？')">🧠 断点重续是怎么实现的？</span>
              <span class="qa-tip" onclick="setQAQuestion('三层记忆架构是什么？')">🗂️ 三层记忆架构是什么？</span>
            </div>
          </div>
        </div>
        <!-- 断点重续横条：检测到上次未完成任务时显示，跟聊天页同款体验 -->
        <div id="qaResumeBar" class="qa-resume-bar" style="display:none">
          <span class="resume-icon">⏸️</span>
          <span class="resume-text">检测到上次未完成的任务：<b class="resume-query"></b></span>
          <button class="btn-resume" onclick="resumeFromBarQA()">继续</button>
          <button class="btn-resume-dismiss" onclick="dismissResumeBarQA()">忽略</button>
        </div>
        <div class="qa-input-area">
          <textarea class="qa-input" id="qaInput" rows="2" placeholder="输入你的问题，按 Enter 发送..." onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();askQA()}"></textarea>
          <button class="qa-send-btn" id="qaSendBtn" onclick="askQA()">发送</button>
        </div>
      </div>
    </div>
    <!-- Token 用量 Tab -->
    <div id="tabUsage" class="tab-panel">
      <div class="toolbar">
        <div>
          <h2>Token 用量看板</h2>
          <p class="sub">网关记录的每一次模型调用，按用户聚合。数据源：SQLite 持久化，重启不丢。</p>
        </div>
        <div style="display:flex;gap:8px;align-items:center">
          <div class="range-btn active" data-range="today" onclick="setAdminRange('today')">今日</div>
          <div class="range-btn" data-range="7d" onclick="setAdminRange('7d')">近 7 天</div>
          <div class="range-btn" data-range="30d" onclick="setAdminRange('30d')">近 30 天</div>
          <div class="range-btn" data-range="all" onclick="setAdminRange('all')">全部</div>
          <button class="btn btn-sm btn-outline" onclick="loadAdminUsage()">🔄 刷新</button>
        </div>
      </div>
      <div id="adminUsageContent" style="overflow-y:auto;flex:1">
        <div class="usage-empty">加载中…</div>
      </div>
    </div>
  </div>
</div>

<!-- ===== Modal ===== -->
<div id="modalOverlay" class="modal-overlay hidden">
  <div class="modal">
    <div class="modal-header">
      <h3 id="modalTitle">编辑提示词</h3>
      <button class="btn btn-sm btn-outline" onclick="closeModal()">✕</button>
    </div>
    <div class="modal-body">
      <div class="field">
        <label>名称（唯一标识）</label>
        <input id="editName" placeholder="如: classify, rewrite_first">
      </div>
      <div class="field">
        <label>显示名称</label>
        <input id="editDisplayName" placeholder="如: 问题分类器">
      </div>
      <div class="field">
        <label>分类</label>
        <select id="editCategory">
          <option value="路由">路由</option>
          <option value="检索">检索</option>
          <option value="生成">生成</option>
          <option value="规划">规划</option>
          <option value="记忆">记忆</option>
        </select>
      </div>
      <div class="field">
        <label>用途说明</label>
        <input id="editDescription" placeholder="简短描述提示词的用途">
      </div>
      <div class="field">
        <label>系统提示词 (System Prompt)</label>
        <textarea id="editSystem" placeholder="输入系统提示词..." rows="6"></textarea>
      </div>
      <div class="field">
        <label>用户消息模板 (User Template) 
          <span style="color:var(--text-3);font-weight:400">— 使用 {variable} 作为占位符</span>
        </label>
        <textarea id="editUserTemplate" placeholder="输入用户消息模板，如: 问题: {query}..." rows="3"></textarea>
      </div>
      <div class="field">
        <label>版本号</label>
        <input id="editVersion" readonly style="background:var(--bg);color:var(--text-2)">
      </div>
    </div>
    <div class="modal-footer">
      <button class="btn btn-outline" onclick="closeModal()">取消</button>
      <button class="btn btn-primary" onclick="savePrompt()">💾 保存</button>
    </div>
  </div>
</div>

<script>
// ===== State =====
let token = '';
let currentUser = null;
// 在线问答断点重续 + 中断控制（admin 后台独立命名空间，跟聊天页的 currentAbortController 不冲突）
let currentAbortControllerQA = null;
let isQueryingQA = false;

// ===== Init =====
// 页面加载时校验登录态（无 token / 失效 / 非 admin 时由函数内重定向）
tryAutoLogin();

// ===== Auth =====
// 登录统一在 /login 页完成；本页只校验已有 token（见 tryAutoLogin）。
function doLogout() {
  token = '';
  currentUser = null;
  localStorage.removeItem('rag_token');
  localStorage.removeItem('rag_admin_user');
  location.replace('/login');
}

// ===== 无权限提示(普通用户访问 /admin 时显示,不静默跳转) =====
function showNoPermission(u){
  document.getElementById('appPage').classList.add('hidden');
  document.getElementById('loginPage').classList.add('hidden');
  let np = document.getElementById('noPermissionPage');
  if(!np){
    np = document.createElement('div');
    np.id = 'noPermissionPage';
    np.style.cssText = 'position:fixed;inset:0;z-index:9000;display:flex;flex-direction:column;align-items:center;justify-content:center;background:var(--bg);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;color:var(--text);';
    np.innerHTML =
      '<div style="background:var(--surface);padding:40px 48px;border-radius:16px;box-shadow:var(--shadow-lg);text-align:center;max-width:440px">' +
      '<div style="font-size:48px;margin-bottom:12px">🚫</div>' +
      '<h2 style="margin:0 0 10px;font-size:20px">无访问权限</h2>' +
      '<p style="margin:0 0 6px;color:var(--text-2);font-size:14px">当前账号 <b style="color:var(--primary)">' + (u && u.username ? u.username : 'unknown') + '</b> 不是管理员,无法访问系统管理页。</p>' +
      '<p style="margin:0 0 22px;color:var(--text-3);font-size:13px">如需管理权限,请联系系统管理员开通。</p>' +
      '<div style="display:flex;gap:12px;justify-content:center">' +
      '<button onclick="location.replace(\'/\')" style="padding:8px 20px;border:1.5px solid var(--primary);border-radius:8px;background:var(--primary-light);color:var(--primary);font-size:14px;cursor:pointer">返回问答</button>' +
      '<button onclick="doLogout()" style="padding:8px 20px;border:1.5px solid #cbd5e1;border-radius:8px;background:transparent;color:var(--text-2);font-size:14px;cursor:pointer">退出登录</button>' +
      '</div>' +
      '</div>';
    document.body.appendChild(np);
  } else {
    np.style.display = 'flex';
  }
}

function showApp() {
  document.getElementById('displayName').textContent = currentUser.display_name;
  document.documentElement.style.visibility = 'visible';  // 校验通过，显示页面
  document.getElementById('appPage').classList.remove('hidden');
  loadPrompts();
  loadCategories();
  loadQAHistory();      // 在线问答：刷新/重登后恢复历史问答
  checkUnfinishedQA();  // 在线问答：检测上次未完成的任务，弹横条让用户确认是否继续
}

async function tryAutoLogin() {
  const savedToken = localStorage.getItem('rag_token');
  if (!savedToken){ location.replace('/login'); return; }

  try {
    const res = await fetch('/api/me', {
      headers: {'Authorization': 'Bearer ' + savedToken}
    });
    if (res.ok) {
      const u = await res.json();
      if (u.role !== 'admin'){ showNoPermission(u); return; }  // 非 admin 提示无权限,不让页面"消失"
      token = savedToken;
      currentUser = {
        username: u.username,
        display_name: u.display_name || '管理员'
      };
      showApp();
    } else {
      localStorage.removeItem('rag_token');
      location.replace('/login');
    }
  } catch(e) {
    localStorage.removeItem('rag_token');
    location.replace('/login');
  }
}

// ===== Tabs =====
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelector(`.tab[data-tab="${name}"]`).classList.add('active');
  document.getElementById(`tab${name.charAt(0).toUpperCase() + name.slice(1)}`).classList.add('active');
  if (name === 'usage') loadAdminUsage();
}

// ===== Token 用量看板 =====
let adminUsageRange = 'today';

function setAdminRange(r) {
  adminUsageRange = r;
  document.querySelectorAll('.range-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.range === r));
  loadAdminUsage();
}

function uFmt(n) { return (n || 0).toLocaleString('en-US'); }

function uTime(ts) {
  if (!ts) return '—';
  const d = new Date(ts * 1000);
  const p = n => String(n).padStart(2, '0');
  return `${p(d.getMonth()+1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

async function loadAdminUsage() {
  const box = document.getElementById('adminUsageContent');
  box.innerHTML = '<div class="usage-empty">加载中…</div>';
  try {
    const res = await fetch(`/api/admin/usage/top?range=${adminUsageRange}&limit=100`, {
      headers: {'Authorization': 'Bearer ' + token}
    });
    const data = await res.json();
    if (!res.ok) {
      box.innerHTML = `<div class="usage-empty">❌ ${data.error || '查询失败'}</div>`;
      return;
    }
    renderAdminUsage(box, data);
  } catch(e) {
    box.innerHTML = '<div class="usage-empty">❌ 网络异常，无法获取用量</div>';
  }
}

function renderAdminUsage(box, data) {
  const w = data.window || {};
  const users = data.users || [];
  const rows = data.rows || [];

  const cards = `
    <div class="stat-grid">
      <div class="stat-card primary">
        <div class="label">调用次数</div>
        <div class="value">${uFmt(w.calls)}</div>
        <div class="sub">${users.length} 个用户</div>
      </div>
      <div class="stat-card success">
        <div class="label">Token 总量</div>
        <div class="value">${uFmt(w.total_tokens)}</div>
        <div class="sub">入 ${uFmt(w.prompt_tokens)} / 出 ${uFmt(w.completion_tokens)}</div>
      </div>
      <div class="stat-card warning">
        <div class="label">累计成本</div>
        <div class="value" style="font-size:18px">$${(w.cost_usd||0).toFixed(4)}</div>
        <div class="sub">按配置单价估算</div>
      </div>
      <div class="stat-card">
        <div class="label">平均耗时</div>
        <div class="value" style="font-size:18px">${(w.avg_latency_s||0).toFixed(2)}s</div>
        <div class="sub">${data.persisted ? '已持久化 · ' + (data.db||'') : '仅内存'}</div>
      </div>
    </div>`;

  let userTable;
  if (!users.length) {
    userTable = '<div class="usage-empty">该时间范围内没有任何调用记录</div>';
  } else {
    const trs = users.map((u, i) => `
      <tr>
        <td><span class="rank-badge ${i===0?'top1':''}">${i+1}</span></td>
        <td><b>${u.user}</b></td>
        <td class="num">${uFmt(u.calls)}</td>
        <td class="num">${uFmt(u.prompt_tokens)}</td>
        <td class="num">${uFmt(u.completion_tokens)}</td>
        <td class="num"><b>${uFmt(u.total_tokens)}</b></td>
        <td class="num">$${(u.cost_usd||0).toFixed(5)}</td>
        <td>${uTime(u.last_active_ts)}</td>
      </tr>`).join('');
    userTable = `
      <div class="table-wrap">
        <table class="usage-table">
          <thead><tr>
            <th>#</th><th>用户</th><th class="num">调用</th>
            <th class="num">输入</th><th class="num">输出</th><th class="num">合计</th>
            <th class="num">成本</th><th>最近活跃</th>
          </tr></thead>
          <tbody>${trs}</tbody>
        </table>
      </div>`;
  }

  let logTable;
  if (!rows.length) {
    logTable = '<div class="usage-empty">暂无明细</div>';
  } else {
    const trs = rows.map(r => `
      <tr>
        <td>${uTime(r.ts)}</td>
        <td><b>${r.user}</b></td>
        <td style="color:var(--text-2)">${r.model || '—'}</td>
        <td><span class="tag-task">${r.task || 'default'}</span></td>
        <td class="num">${uFmt(r.prompt_tokens)}</td>
        <td class="num">${uFmt(r.completion_tokens)}</td>
        <td class="num"><b>${uFmt(r.total_tokens)}</b></td>
        <td class="num">${(r.latency_s||0).toFixed(2)}s</td>
        <td class="num">$${(r.cost_usd||0).toFixed(5)}</td>
      </tr>`).join('');
    logTable = `
      <div class="table-wrap">
        <table class="usage-table">
          <thead><tr>
            <th>时间</th><th>用户</th><th>模型</th><th>任务</th>
            <th class="num">输入</th><th class="num">输出</th><th class="num">合计</th>
            <th class="num">耗时</th><th class="num">成本</th>
          </tr></thead>
          <tbody>${trs}</tbody>
        </table>
      </div>`;
  }

  box.innerHTML = cards +
    '<div class="usage-section-title">🏆 用户排行（按 Token 消耗）</div>' + userTable +
    `<div class="usage-section-title">🧾 调用明细（最近 ${rows.length} 条）</div>` + logTable;
}

// ===== Prompts =====
async function loadPrompts() {
  try {
    const res = await fetch('/api/admin/prompts');
    const data = await res.json();
    window._allPrompts = data.prompts;
    renderPrompts(data.prompts);
  } catch(e) {
    showToast('加载提示词列表失败', 'error');
  }
}

async function loadCategories() {
  try {
    const res = await fetch('/api/admin/categories');
    const data = await res.json();
    const sel = document.getElementById('filterCategory');
    sel.innerHTML = '<option value="">全部分类</option>'
      + data.categories.map(c => `<option value="${c}">${c}</option>`).join('');
  } catch(e) {}
}

function renderPrompts(prompts) {
  const el = document.getElementById('promptList');
  el.innerHTML = prompts.map(p => `
    <div class="prompt-card ${p.is_active?'':'inactive'}" onclick="editPrompt('${p.name}')">
      <div class="top">
        <span class="name">${p.display_name}<span class="tag">${p.category}</span></span>
        <span style="font-size:11px;color:var(--text-3)">v${p.version} ${p.is_active?'':'[已禁用]'}</span>
      </div>
      <div class="desc">${p.description || '无描述'}</div>
      <div class="meta">
        <span>名称: ${p.name}</span>
        <span>系统提示词: ${p.system.substring(0,50)}...</span>
      </div>
    </div>
  `).join('');
}

function filterPrompts() {
  const search = document.getElementById('searchInput').value.toLowerCase();
  const cat = document.getElementById('filterCategory').value;
  let prompts = window._allPrompts || [];
  if (search) prompts = prompts.filter(p =>
    p.name.toLowerCase().includes(search) ||
    p.display_name.toLowerCase().includes(search) ||
    p.system.toLowerCase().includes(search)
  );
  if (cat) prompts = prompts.filter(p => p.category === cat);
  renderPrompts(prompts);
}

// ===== Editor =====
async function editPrompt(name) {
  try {
    const res = await fetch(`/api/admin/prompts/${name}`);
    const p = await res.json();
    document.getElementById('modalTitle').textContent = '编辑: ' + p.display_name;
    document.getElementById('editName').value = p.name;
    document.getElementById('editDisplayName').value = p.display_name;
    document.getElementById('editCategory').value = p.category;
    document.getElementById('editDescription').value = p.description;
    document.getElementById('editSystem').value = p.system;
    document.getElementById('editUserTemplate').value = p.user_template;
    document.getElementById('editVersion').value = p.version;
    document.getElementById('modalOverlay').classList.remove('hidden');
  } catch(e) {
    showToast('加载提示词失败', 'error');
  }
}

function closeModal() {
  document.getElementById('modalOverlay').classList.add('hidden');
}

async function savePrompt() {
  const data = {
    name: document.getElementById('editName').value.trim(),
    display_name: document.getElementById('editDisplayName').value.trim(),
    category: document.getElementById('editCategory').value,
    description: document.getElementById('editDescription').value.trim(),
    system: document.getElementById('editSystem').value,
    user_template: document.getElementById('editUserTemplate').value,
  };
  if (!data.name || !data.system) {
    showToast('名称和系统提示词不能为空', 'error');
    return;
  }
  try {
    const res = await fetch('/api/admin/prompts', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(data)
    });
    const result = await res.json();
    if (result.success) {
      closeModal();
      loadPrompts();
      showToast(result.message, 'success');
    } else {
      showToast(result.error, 'error');
    }
  } catch(e) {
    showToast('保存失败', 'error');
  }
}

async function importDefaults() {
  if (!confirm('将用默认提示词覆盖当前数据库中的所有记录，确定？')) return;
  try {
    const res = await fetch('/api/admin/prompts/import-defaults', {method:'POST'});
    const data = await res.json();
    loadPrompts();
    showToast(`已导入 ${data.imported} 个默认提示词`, 'success');
  } catch(e) {
    showToast('导入失败', 'error');
  }
}

function changePassword() {
  openChangePwd();
}

// ===== Q&A =====
function setQAQuestion(text) {
  const input = document.getElementById('qaInput');
  input.value = text;
  input.focus();
}

async function loadQAHistory() {
  try {
    const r = await fetch('/api/history', {
      headers: {'Authorization': 'Bearer ' + token}
    });
    if (!r.ok) return;
    const data = await r.json();
    const msgs = data.messages || [];
    if (!msgs.length) return;
    const qaMsgs = document.getElementById('qaMessages');
    if (!qaMsgs) return;
    // 有历史时移除空状态
    const emptyEl = document.getElementById('qaEmpty');
    if (emptyEl) emptyEl.remove();
    for (const m of msgs) {
      if (m.role === 'user') {
        qaMsgs.innerHTML +=
          '<div class="qa-msg user">' +
            '<div class="avatar">👤</div>' +
            '<div class="qa-bubble">' + escapeHtml(m.content) + '</div>' +
          '</div>';
      } else if (m.role === 'assistant') {
        qaMsgs.innerHTML +=
          '<div class="qa-msg assistant">' +
            '<div class="avatar">🤖</div>' +
            '<div class="qa-bubble" style="white-space:pre-wrap">' + escapeHtml(m.content) + '</div>' +
          '</div>';
      }
    }
    qaMsgs.scrollTop = qaMsgs.scrollHeight;
  } catch (e) {
    // 历史拉取失败不影响新对话，静默忽略
  }
}

// ===== 在线问答：断点重续（跟聊天页 checkUnfinishedTasks 同款体验）=====
function checkUnfinishedQA() {
  fetch('/api/tasks/unfinished', {headers: {'Authorization': 'Bearer ' + token}})
    .then(r => r.json())
    .then(data => {
      if (!data.tasks || data.tasks.length === 0) return;
      const task = data.tasks[0];
      const bar = document.getElementById('qaResumeBar');
      if (!bar) return;
      bar.querySelector('.resume-query').textContent = task.query;
      bar.dataset.taskId = task.task_id;
      bar.style.display = 'flex';
    })
    .catch(() => {/* 接口失败不影响主流程 */});
}

function resumeFromBarQA() {
  const bar = document.getElementById('qaResumeBar');
  if (!bar) return;
  const taskId = bar.dataset.taskId;
  if (!taskId) return;
  bar.style.display = 'none';
  sendQuestionWithQA(taskId, bar.querySelector('.resume-query').textContent);
}

function dismissResumeBarQA() {
  const bar = document.getElementById('qaResumeBar');
  if (bar) bar.style.display = 'none';
}

// 中断当前流式查询：跟聊天页 abortCurrentQuery 同款
function abortCurrentQA() {
  if (currentAbortControllerQA) {
    try { currentAbortControllerQA.abort(); } catch (e) {}
  }
}

function updateQASendButton() {
  const btn = document.getElementById('qaSendBtn');
  if (!btn) return;
  btn.disabled = false;
  if (isQueryingQA) {
    btn.classList.add('stopping');
    btn.textContent = '⏹';
    btn.title = '中断当前回答';
    btn.onclick = function(){ abortCurrentQA(); };
  } else {
    btn.classList.remove('stopping');
    btn.textContent = '发送';
    btn.title = '发送';
    btn.onclick = function(){ askQA(); };
  }
}

async function askQA() {
  const input = document.getElementById('qaInput');
  const sendBtn = document.getElementById('qaSendBtn');
  const question = input.value.trim();
  if (!question || isQueryingQA) return;

  // 隐藏空状态提示
  const emptyEl = document.getElementById('qaEmpty');
  if (emptyEl) emptyEl.remove();

  input.value = '';
  isQueryingQA = true;
  updateQASendButton();

  const msgs = document.getElementById('qaMessages');
  msgs.innerHTML += `
    <div class="qa-msg user">
      <div class="avatar">👤</div>
      <div class="qa-bubble">${escapeHtml(question)}</div>
    </div>`;

  // 流式进度气泡：进度头 + 日志列表，跟聊天页 handleSSEEvent 同款体验
  const loadingId = 'loading_' + Date.now();
  msgs.innerHTML += `
    <div class="qa-msg assistant" id="${loadingId}">
      <div class="avatar">🤖</div>
      <div class="qa-bubble">
        <div class="qa-progress-head"><span class="spinner"></span><span class="head-text">准备中...</span></div>
        <div class="qa-progress-logs"></div>
      </div>
    </div>`;
  msgs.scrollTop = msgs.scrollHeight;

  const bubble = document.getElementById(loadingId).querySelector('.qa-bubble');
  const headText = bubble.querySelector('.head-text');
  const logsEl = bubble.querySelector('.qa-progress-logs');
  let finalAnswer = null;

  // 把 controller 挂到全局，让"⏹ 停止"按钮能真中止请求
  currentAbortControllerQA = new AbortController();

  try {
    const resp = await fetch('/api/query/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + token
      },
      body: JSON.stringify({question, role: 'admin',
                            username: (currentUser && currentUser.username) || 'admin'}),
      signal: currentAbortControllerQA.signal
    });

    if (!resp.ok || !resp.body) {
      throw new Error('HTTP ' + resp.status);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let data;
        try { data = JSON.parse(line.slice(6)); } catch (e) { continue; }
        if (data.type === 'start') {
          headText.textContent = '正在分析问题...';
        } else if (data.type === 'log') {
          const text = data.text || '';
          // 阶段判定（跟聊天页 handleSSEEvent 同步）
          if (text.includes('用户提问')) headText.textContent = '正在检索文档...';
          else if (text.includes('子任务') && text.includes('开始执行')) headText.textContent = '正在执行子任务...';
          else if (text.includes('最终回答') || text.includes('从缓存返回')) headText.textContent = '正在生成答案...';
          if (text && text.length < 200) {
            const div = document.createElement('div');
            div.className = 'log-line';
            div.textContent = text;
            logsEl.appendChild(div);
            logsEl.scrollTop = logsEl.scrollHeight;
          }
        } else if (data.type === 'done') {
          finalAnswer = data.answer || '';
          headText.textContent = '✓ 回答完成';
          bubble.querySelector('.spinner').style.display = 'none';
        } else if (data.type === 'error') {
          throw new Error(data.text || data.error || '未知错误');
        }
      }
    }

    // 流结束：用最终答案替换整个进度区
    if (finalAnswer !== null) {
      bubble.innerHTML = '<div class="qa-progress-head" style="margin-bottom:6px"><span style="color:var(--success)">✓</span><span>回答完成</span></div>' +
                         '<div style="white-space:pre-wrap;font-size:14px;line-height:1.7">' + escapeHtml(finalAnswer) + '</div>';
    } else {
      bubble.innerHTML = '<div style="color:var(--text-3)">无响应</div>';
    }
  } catch(e) {
    if (e.name === 'AbortError') {
      bubble.innerHTML = '<div style="color:var(--text-3)">⏹ 已中断当前回答</div>';
    } else {
      bubble.innerHTML = '<div style="color:var(--danger)">请求失败: ' + escapeHtml(e.message) + '</div>';
    }
  } finally {
    isQueryingQA = false;
    currentAbortControllerQA = null;
    updateQASendButton();
    msgs.scrollTop = msgs.scrollHeight;
  }
}

// 断点重续入口：复用原问题，调 /api/tasks/resume 恢复上次未完成的任务
async function sendQuestionWithQA(taskId, question) {
  if (!question || isQueryingQA) return;
  const emptyEl = document.getElementById('qaEmpty');
  if (emptyEl) emptyEl.remove();

  const msgs = document.getElementById('qaMessages');
  msgs.innerHTML += `
    <div class="qa-msg user">
      <div class="avatar">👤</div>
      <div class="qa-bubble">${escapeHtml(question)}</div>
    </div>`;

  // 流式进度气泡，跟 askQA 风格一致
  const loadingId = 'resume_' + Date.now();
  msgs.innerHTML += `
    <div class="qa-msg assistant" id="${loadingId}">
      <div class="avatar">🤖</div>
      <div class="qa-bubble">
        <div class="qa-progress-head"><span class="spinner"></span><span class="head-text">⏳ 正在从断点恢复...</span></div>
        <div class="qa-progress-logs"></div>
      </div>
    </div>`;
  msgs.scrollTop = msgs.scrollHeight;

  const bubble = document.getElementById(loadingId).querySelector('.qa-bubble');
  const headText = bubble.querySelector('.head-text');
  const logsEl = bubble.querySelector('.qa-progress-logs');
  let finalAnswer = null;
  let errored = false;

  isQueryingQA = true;
  updateQASendButton();
  currentAbortControllerQA = new AbortController();

  try {
    const resp = await fetch('/api/tasks/resume', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + token
      },
      body: JSON.stringify({task_id: taskId}),
      signal: currentAbortControllerQA.signal
    });

    if (!resp.ok || !resp.body) {
      let errText = '恢复失败';
      try { const j = await resp.json(); errText = j.error || errText; } catch (_) {}
      throw new Error(errText);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let data;
        try { data = JSON.parse(line.slice(6)); } catch (e) { continue; }
        if (data.type === 'start') {
          headText.textContent = '正在分析问题...';
        } else if (data.type === 'log') {
          const text = data.text || '';
          if (text.includes('用户提问')) headText.textContent = '正在检索文档...';
          else if (text.includes('子任务') && text.includes('开始执行')) headText.textContent = '正在执行子任务...';
          else if (text.includes('最终回答') || text.includes('从缓存返回')) headText.textContent = '正在生成答案...';
          if (text && text.length < 200) {
            const div = document.createElement('div');
            div.className = 'log-line';
            div.textContent = text;
            logsEl.appendChild(div);
            logsEl.scrollTop = logsEl.scrollHeight;
          }
        } else if (data.type === 'done') {
          finalAnswer = data.answer || '';
          headText.textContent = '✓ 恢复完成';
          bubble.querySelector('.spinner').style.display = 'none';
        } else if (data.type === 'error') {
          errored = true;
          throw new Error(data.text || data.error || '未知错误');
        }
      }
    }

    if (finalAnswer !== null) {
      bubble.innerHTML = '<div class="qa-progress-head" style="margin-bottom:6px"><span style="color:var(--success)">✓</span><span>恢复完成</span></div>' +
                         '<div style="white-space:pre-wrap;font-size:14px;line-height:1.7">' + escapeHtml(finalAnswer) + '</div>';
    } else if (errored) {
      bubble.innerHTML = '<div style="color:var(--danger)">恢复过程出现异常，请稍后重试。</div>';
    } else {
      bubble.innerHTML = '<div style="color:var(--text-3)">无响应</div>';
    }
  } catch(e) {
    if (e.name === 'AbortError') {
      bubble.innerHTML = '<div style="color:var(--text-3)">⏹ 已中断断点恢复</div>';
    } else {
      bubble.innerHTML = '<div style="color:var(--danger)">恢复失败：' + escapeHtml(e.message) + '</div>';
    }
  } finally {
    isQueryingQA = false;
    currentAbortControllerQA = null;
    updateQASendButton();
    msgs.scrollTop = msgs.scrollHeight;
  }
}

// ===== Toast =====
function showToast(msg, type) {
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3000);
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// ===== 修改密码模态框 =====
function openChangePwd() {
  const m = document.getElementById('changePwdModal');
  if (!m) return;
  document.getElementById('oldPwdInput').value = '';
  document.getElementById('newPwdInput').value = '';
  document.getElementById('newPwdInput2').value = '';
  document.getElementById('changePwdMsg').textContent = '';
  document.getElementById('changePwdMsg').className = 'pwd-msg';
  m.classList.add('show');
  setTimeout(() => document.getElementById('oldPwdInput').focus(), 50);
}

function closeChangePwd() {
  document.getElementById('changePwdModal').classList.remove('show');
}

async function submitChangePwd() {
  const oldPwd = document.getElementById('oldPwdInput').value;
  const newPwd = document.getElementById('newPwdInput').value;
  const newPwd2 = document.getElementById('newPwdInput2').value;
  const msgEl = document.getElementById('changePwdMsg');
  msgEl.className = 'pwd-msg';
  msgEl.textContent = '';

  if (!oldPwd || !newPwd || !newPwd2) {
    msgEl.className = 'pwd-msg err';
    msgEl.textContent = '请填写完整';
    return;
  }
  if (newPwd.length < 6) {
    msgEl.className = 'pwd-msg err';
    msgEl.textContent = '新密码至少 6 位';
    return;
  }
  if (newPwd !== newPwd2) {
    msgEl.className = 'pwd-msg err';
    msgEl.textContent = '两次输入的新密码不一致';
    return;
  }

  try {
    const r = await fetch('/api/change-password', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({old_password: oldPwd, new_password: newPwd})
    });
    const d = await r.json();
    if (d.success) {
      msgEl.className = 'pwd-msg ok';
      msgEl.textContent = '✅ ' + d.message;
      showToast(d.message, 'success');
      setTimeout(closeChangePwd, 800);
    } else {
      msgEl.className = 'pwd-msg err';
      msgEl.textContent = d.error || '修改失败';
    }
  } catch (e) {
    msgEl.className = 'pwd-msg err';
    msgEl.textContent = '网络错误：' + e.message;
  }
}
</script>

<!-- 修改密码模态框（admin 后台用） -->
<div class="modal-mask" id="changePwdModal" onclick="if(event.target===this)closeChangePwd()">
  <div class="modal">
    <div class="modal-head">
      <h3>🔑 修改密码</h3>
      <button class="modal-close" onclick="closeChangePwd()" title="关闭">×</button>
    </div>
    <div class="modal-body">
      <div class="pwd-row">
        <label for="oldPwdInput">当前密码</label>
        <input type="password" id="oldPwdInput" placeholder="请输入当前密码" autocomplete="current-password">
      </div>
      <div class="pwd-row">
        <label for="newPwdInput">新密码</label>
        <input type="password" id="newPwdInput" placeholder="至少 6 位" autocomplete="new-password">
      </div>
      <div class="pwd-row">
        <label for="newPwdInput2">确认新密码</label>
        <input type="password" id="newPwdInput2" placeholder="再次输入新密码" autocomplete="new-password">
      </div>
      <div class="pwd-msg" id="changePwdMsg"></div>
      <div class="pwd-actions">
        <button class="btn btn-outline" onclick="closeChangePwd()">取消</button>
        <button class="btn btn-primary" onclick="submitChangePwd()">确认修改</button>
      </div>
    </div>
  </div>
</div>
</body>
</html>
"""

_LOGIN_PAGE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>登录 - RAG 企业知识库</title>
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  html,body{height:100%}
  body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans SC",sans-serif;
    background:linear-gradient(135deg,#1e3a8a 0%,#0f172a 100%);display:flex;align-items:center;justify-content:center}
  .card{background:#1e293b;padding:38px 40px;border-radius:16px;width:340px;box-shadow:0 20px 60px rgba(0,0,0,.5)}
  h2{margin:0 0 6px;color:#f8fafc;font-size:21px}
  .sub{margin:0 0 24px;color:#94a3b8;font-size:13px}
  input{width:100%;padding:12px 13px;margin-bottom:14px;border-radius:9px;border:1px solid #334155;
    background:#0f172a;color:#e2e8f0;font-size:14px;box-sizing:border-box}
  button{width:100%;padding:12px;border:0;border-radius:9px;background:#2563eb;color:#fff;font-size:15px;cursor:pointer}
  button:hover{background:#1d4ed8}
  .err{color:#f87171;font-size:13px;margin-top:14px;min-height:18px;text-align:center}
</style>
</head>
<body>
  <div class="card">
    <h2>🔐 系统登录</h2>
    <p class="sub">账号由管理员分配，登录后按角色进入对应页面</p>
    <input id="lu" placeholder="用户名" autocomplete="username">
    <input id="lp" type="password" placeholder="密码" autocomplete="current-password">
    <button onclick="doLogin()">登 录</button>
    <div class="err" id="le"></div>
  </div>
<script>
const LS='rag_token';
// 已登录则直接跳对应页，避免重复登录
(function(){
  var t=localStorage.getItem(LS);
  if(t){
    fetch('/api/me',{headers:{'Authorization':'Bearer '+t}}).then(function(r){
      if(r.ok) return r.json().then(function(u){
        location.replace(u.role==='admin' ? '/admin' : '/');
      });
    }).catch(function(){});
  }
})();
function doLogin(){
  var u=document.getElementById('lu').value.trim();
  var p=document.getElementById('lp').value;
  var e=document.getElementById('le');
  e.textContent='';
  if(!u||!p){e.textContent='请输入用户名和密码';return;}
  fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({username:u,password:p})})
    .then(function(r){return r.json().then(function(j){return {r:r,j:j};});})
    .then(function(o){
      var r=o.r,j=o.j;
      if(r.ok && j.token){
        localStorage.setItem(LS,j.token);
        location.replace((j.user && j.user.role==='admin') ? '/admin' : '/');
      }else{
        e.textContent=(j.error||'登录失败，请检查账号或密码');
      }
    }).catch(function(){e.textContent='网络错误，无法连接服务';});
}
document.getElementById('lp').addEventListener('keydown',function(ev){if(ev.key==='Enter')doLogin();});
</script>
</body></html>
"""

_HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG 企业知识��问答</title>
<style>
  /* ===== Reset & Base ===== */
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  :root{
    --bg:#f5f6fa;--surface:#fff;--border:#e2e5ed;
    --text:#1a1a2e;--text-2:#6b7280;--text-3:#9ca3af;
    --primary:#2563eb;--primary-light:#dbeafe;--primary-hover:#1d4ed8;
    --danger:#dc2626;--danger-light:#fef2f2;
    --success:#16a34a;--success-light:#f0fdf4;
    --warning:#d97706;--warning-light:#fffbeb;
    --shadow:0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.06);
    --shadow-lg:0 10px 25px rgba(0,0,0,.1);
    --radius:12px;--radius-sm:8px;
    --font-mono:'JetBrains Mono','Fira Code','Cascadia Code',monospace;
  }
  html,body{height:100%}
  body{
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans SC",sans-serif;
    background:var(--bg);color:var(--text);display:flex;flex-direction:column;
  }

  /* ===== Header ===== */
  .header{
    background:var(--surface);border-bottom:1px solid var(--border);
    padding:12px 24px;display:flex;align-items:center;justify-content:space-between;
    box-shadow:var(--shadow);z-index:10;
  }
  .header-left{display:flex;align-items:center;gap:10px}
  .logo{width:32px;height:32px;background:var(--primary);border-radius:8px;
    display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;font-size:16px}
  .header h1{font-size:18px;font-weight:600}
  .header-right{display:flex;align-items:center;gap:16px}
  .role-badge{
    display:flex;align-items:center;gap:6px;padding:6px 14px;border-radius:20px;
    font-size:13px;font-weight:500;cursor:pointer;border:1.5px solid var(--border);
    transition:all .2s;user-select:none;background:var(--surface);
  }
  .role-badge:hover{background:var(--bg)}
  .role-badge.admin{border-color:var(--danger);background:var(--danger-light);color:var(--danger)}
  .role-badge.user{border-color:var(--primary);background:var(--primary-light);color:var(--primary)}
  .role-dot{width:8px;height:8px;border-radius:50%}
  .role-dot.admin{background:var(--danger)}
  .role-dot.user{background:var(--primary)}
  .status-indicator{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--text-2)}
  .status-dot{width:8px;height:8px;border-radius:50%;background:var(--success);animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}

  /* ===== Main Chat Area ===== */
  .main{flex:1;display:flex;flex-direction:column;max-width:860px;width:100%;margin:0 auto;padding:0 16px;overflow:hidden}
  .chat-area{flex:1;overflow-y:auto;padding:24px 0;scroll-behavior:smooth}
  .chat-area::-webkit-scrollbar{width:6px}
  .chat-area::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}

  /* ===== Welcome ===== */
  .welcome{text-align:center;padding:60px 20px 40px}
  .welcome-icon{font-size:48px;margin-bottom:16px}
  .welcome h2{font-size:22px;font-weight:600;margin-bottom:8px}
  .welcome p{color:var(--text-2);font-size:14px;line-height:1.6;max-width:500px;margin:0 auto 20px}
  .suggestions{display:flex;flex-wrap:wrap;gap:8px;justify-content:center}
  .suggestion{
    padding:8px 16px;border-radius:20px;border:1px solid var(--border);
    font-size:13px;color:var(--text-2);cursor:pointer;transition:all .15s;
    background:var(--surface);white-space:nowrap;
  }
  .suggestion:hover{background:var(--primary-light);color:var(--primary);border-color:var(--primary)}

  /* ===== Messages ===== */
  .message{display:flex;gap:10px;margin-bottom:20px;animation:slideIn .3s ease}
  @keyframes slideIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
  .msg-avatar{width:36px;height:36px;border-radius:10px;flex-shrink:0;
    display:flex;align-items:center;justify-content:center;font-size:16px}
  .msg-avatar.user{background:var(--primary-light);color:var(--primary)}
  .msg-avatar.assistant{background:#eef2ff;color:#4f46e5}
  .msg-body{flex:1;min-width:0}
  .msg-role-name{font-size:12px;font-weight:600;margin-bottom:4px;color:var(--text-2)}
  .msg-content{
    background:var(--surface);border-radius:var(--radius);
    padding:14px 18px;font-size:14px;line-height:1.7;box-shadow:var(--shadow);
    word-break:break-word;white-space:pre-wrap;
  }
  .msg-content p{margin-bottom:8px}
  .msg-content p:last-child{margin-bottom:0}
  .msg-content h3{font-size:15px;font-weight:600;margin:12px 0 6px;color:var(--primary)}
  .msg-content h3:first-child{margin-top:0}
  .msg-content strong{color:var(--text)}
  .msg-time{font-size:11px;color:var(--text-3);margin-top:6px}

  /* ===== Progress (streaming) ===== */
  .progress-box{
    background:var(--surface);border-radius:var(--radius);padding:12px 16px;
    margin-bottom:20px;border:1.5px dashed var(--primary);animation:slideIn .3s ease
  }
  .progress-header{display:flex;align-items:center;gap:8px;margin-bottom:8px}
  .spinner{
    width:16px;height:16px;border:2.5px solid var(--border);border-top-color:var(--primary);
    border-radius:50%;animation:spin .7s linear infinite
  }
  @keyframes spin{to{transform:rotate(360deg)}}
  .progress-header span{font-size:13px;font-weight:500;color:var(--primary)}
  .progress-logs{max-height:180px;overflow-y:auto;font-family:var(--font-mono);font-size:11px;color:var(--text-2)}
  .progress-logs .log-line{padding:2px 0;border-bottom:1px solid var(--border);line-height:1.5}
  .progress-logs .log-line:last-child{border-bottom:none}

  /* ===== Error ===== */
  .error-box{
    background:var(--danger-light);border:1px solid var(--danger);border-radius:var(--radius);
    padding:14px 18px;margin-bottom:20px;color:var(--danger);font-size:13px
  }

  /* ===== Input ===== */
  .input-area{
    background:var(--surface);border-top:1px solid var(--border);
    padding:16px 0 20px;box-shadow:0 -1px 3px rgba(0,0,0,.04)
  }
  .input-row{display:flex;gap:10px}
  .input-row textarea{
    flex:1;resize:none;border:1.5px solid var(--border);border-radius:var(--radius);
    padding:12px 16px;font-size:14px;font-family:inherit;line-height:1.5;
    outline:none;transition:border-color .2s;min-height:48px;max-height:120px
  }
  .input-row textarea:focus{border-color:var(--primary);box-shadow:0 0 0 3px rgba(37,99,235,.1)}
  .input-row textarea::placeholder{color:var(--text-3)}
  .btn-send{
    flex-shrink:0;width:48px;height:48px;border:none;border-radius:var(--radius);
    background:var(--primary);color:#fff;font-size:20px;cursor:pointer;
    transition:all .2s;display:flex;align-items:center;justify-content:center
  }
  .btn-send:hover{background:var(--primary-hover);transform:scale(1.03)}
  .btn-send:disabled{background:var(--border);cursor:not-allowed;transform:none}
  /* 主动中断态：查询中按钮变红色"停止符" */
  .btn-send.stopping{background:#ef4444}
  .btn-send.stopping:hover{background:#dc2626}
  .input-hint{font-size:11px;color:var(--text-3);margin-top:8px;text-align:center}

  /* ===== 用量入口（Header）===== */
  .user-chip{
    display:flex;align-items:center;gap:6px;padding:6px 12px;border-radius:20px;
    font-size:13px;border:1.5px solid var(--border);background:var(--surface);
    cursor:pointer;user-select:none;transition:all .2s;color:var(--text-2)
  }
  .user-chip:hover{background:var(--bg);color:var(--text)}
  .btn-usage{
    display:flex;align-items:center;gap:6px;padding:6px 14px;border-radius:20px;
    font-size:13px;font-weight:500;cursor:pointer;border:1.5px solid var(--primary);
    background:var(--primary-light);color:var(--primary);transition:all .2s
  }
  .btn-usage:hover{background:var(--primary);color:#fff}

  /* ===== 退出登录按钮 ===== */
  .btn-logout{
    display:flex;align-items:center;gap:6px;padding:6px 14px;border-radius:20px;
    font-size:13px;font-weight:500;cursor:pointer;border:1.5px solid #cbd5e1;
    background:transparent;color:var(--text-2);transition:all .2s
  }
  .btn-logout:hover{background:#fee2e2;border-color:#ef4444;color:#b91c1c}

  /* ===== Modal ===== */
  .modal-mask{
    position:fixed;inset:0;background:rgba(15,23,42,.45);z-index:100;
    display:none;align-items:center;justify-content:center;padding:24px;
    backdrop-filter:blur(2px)
  }
  .modal-mask.show{display:flex}
  .modal{
    background:var(--surface);border-radius:16px;box-shadow:var(--shadow-lg);
    width:100%;max-width:880px;max-height:88vh;display:flex;flex-direction:column;
    animation:slideIn .25s ease;overflow:hidden
  }
  .modal-head{
    padding:18px 24px;border-bottom:1px solid var(--border);
    display:flex;align-items:center;justify-content:space-between
  }
  .modal-head h3{font-size:17px;font-weight:600;display:flex;align-items:center;gap:8px}
  .modal-close{
    border:none;background:transparent;font-size:22px;line-height:1;cursor:pointer;
    color:var(--text-3);padding:4px 8px;border-radius:6px
  }
  .modal-close:hover{background:var(--bg);color:var(--text)}
  .modal-body{padding:20px 24px 24px;overflow-y:auto}
  /* ===== 修改密码表单 ===== */
  .pwd-row{display:flex;flex-direction:column;gap:6px;margin-bottom:14px}
  .pwd-row label{font-size:13px;font-weight:500;color:var(--text-2)}
  .pwd-row input{
    width:100%;padding:10px 12px;border:1.5px solid var(--border);border-radius:8px;
    font-size:14px;background:var(--bg);color:var(--text);transition:border-color .2s
  }
  .pwd-row input:focus{outline:none;border-color:var(--primary)}
  .pwd-msg{font-size:13px;min-height:18px;margin-bottom:6px}
  .pwd-msg.err{color:var(--danger)}
  .pwd-msg.ok{color:var(--success)}
  /* ===== 轻量 toast 提示 ===== */
  .toast{
    position:fixed;top:20px;right:20px;z-index:200;
    padding:12px 20px;border-radius:10px;font-size:14px;
    box-shadow:var(--shadow-lg);animation:slideIn .3s;max-width:360px;
  }
  .toast.success{background:var(--success);color:#fff}
  .toast.error{background:var(--danger);color:#fff}

  /* ===== 用量统计卡 ===== */
  .usage-toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:16px}
  .range-btn{
    padding:5px 14px;border-radius:16px;font-size:12px;cursor:pointer;
    border:1.5px solid var(--border);background:var(--surface);color:var(--text-2);transition:all .15s
  }
  .range-btn:hover{border-color:var(--primary);color:var(--primary)}
  .range-btn.active{background:var(--primary);border-color:var(--primary);color:#fff}
  .usage-db-tag{
    margin-left:auto;font-size:11px;color:var(--text-3);font-family:var(--font-mono);
    background:var(--bg);padding:4px 10px;border-radius:12px
  }
  .stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
  .stat-card{
    background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);
    padding:14px 16px
  }
  .stat-card .label{font-size:11px;color:var(--text-2);margin-bottom:6px}
  .stat-card .value{font-size:22px;font-weight:700;color:var(--text);font-family:var(--font-mono)}
  .stat-card .sub{font-size:11px;color:var(--text-3);margin-top:4px}
  .stat-card.primary{background:var(--primary-light);border-color:var(--primary)}
  .stat-card.primary .value{color:var(--primary)}
  .stat-card.success{background:var(--success-light);border-color:var(--success)}
  .stat-card.success .value{color:var(--success)}
  .stat-card.warning{background:var(--warning-light);border-color:var(--warning)}
  .stat-card.warning .value{color:var(--warning)}

  .usage-section-title{font-size:13px;font-weight:600;margin:4px 0 10px;color:var(--text-2)}
  .usage-table{width:100%;border-collapse:collapse;font-size:12px}
  .usage-table th{
    text-align:left;padding:8px 10px;background:var(--bg);color:var(--text-2);
    font-weight:600;border-bottom:1px solid var(--border);white-space:nowrap;
    position:sticky;top:0
  }
  .usage-table td{
    padding:8px 10px;border-bottom:1px solid var(--border);
    font-family:var(--font-mono);color:var(--text);white-space:nowrap
  }
  .usage-table tr:hover td{background:var(--bg)}
  .usage-table .num{text-align:right}
  .tag-task{
    display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;
    background:var(--primary-light);color:var(--primary);font-family:inherit
  }
  .tag-model{color:var(--text-2)}
  .usage-empty{text-align:center;padding:36px 12px;color:var(--text-3);font-size:13px}
  .table-wrap{max-height:320px;overflow:auto;border:1px solid var(--border);border-radius:var(--radius-sm)}

  /* ===== Responsive ===== */
  @media (max-width:640px){
    .header{padding:10px 16px}
    .header h1{font-size:16px}
    .suggestions{flex-direction:column;align-items:stretch}
    .suggestion{text-align:center}
    .stat-grid{grid-template-columns:repeat(2,1fr)}
    .user-chip span.uname{max-width:70px;overflow:hidden;text-overflow:ellipsis}
  }

  /* ===== 断点重续横条 ===== */
  .resume-bar{
    display:flex;align-items:center;gap:10px;
    margin:0 14px 8px;padding:10px 14px;
    background:linear-gradient(90deg,#fff7e6,#fffbe9);
    border:1px solid #ffd591;border-left:4px solid #fa8c16;border-radius:10px;
    font-size:13px;color:#614700;box-shadow:0 1px 4px rgba(250,140,22,.12);
  }
  .resume-bar .resume-icon{font-size:16px}
  .resume-bar .resume-text{flex:1;line-height:1.4}
  .resume-bar .resume-text b{color:#ad4e00}
  .resume-bar .btn-resume{
    border:none;background:#fa8c16;color:#fff;font-weight:600;
    padding:5px 16px;border-radius:7px;cursor:pointer;font-size:13px
  }
  .resume-bar .btn-resume:hover{background:#d8760b}
  .resume-bar .btn-resume-dismiss{
    border:1px solid #d9d9d9;background:#fff;color:#595959;
    padding:5px 12px;border-radius:7px;cursor:pointer;font-size:13px
  }
  .resume-bar .btn-resume-dismiss:hover{background:#f5f5f5}
</style>
</head>
<body>

<!-- 本页为受保护页面：无 token 或 token 失效时由 <body> 顶部脚本重定向到 /login -->
<script>
(function(){
  var t = localStorage.getItem('rag_token');
  if(!t){ location.replace('/login'); return; }
  document.documentElement.style.visibility = 'hidden';  // 先隐藏，校验通过再显示
})();
</script>

<!-- Header -->
<header class="header">
  <div class="header-left">
    <div class="logo">AI</div>
    <h1>企业知识库问答</h1>
  </div>
  <div class="header-right">
    <a href="/admin" id="adminLink" style="display:none;text-decoration:none;color:var(--primary);font-size:13px;font-weight:500;margin-right:8px" title="进入系统管理后台">⚙️ 系统管理</a>
    <div class="user-chip" id="userChip" title="当前登录账号(只读)">
      <span>👤</span><span class="uname" id="userLabel">guest</span>
    </div>
    <div class="btn-usage" onclick="openUsage()" title="查看我的 Token 使用记录">
      <span>📊</span><span>我的用量</span>
    </div>
    <div class="status-indicator">
      <div class="status-dot"></div>
      <span>服务就绪</span>
    </div>
    <div class="role-badge user" id="roleBadge" title="当前为普通用户模式">
      <div class="role-dot user"></div>
      <span id="roleLabel">普通用户</span>
    </div>
    <button class="btn-logout" id="changePwdBtn" onclick="openChangePwd()" title="修改密码">
      <span>🔑</span><span>修改密码</span>
    </button>
    <button class="btn-logout" id="logoutBtn" onclick="doLogout()" title="退出登录">
      <span>🚪</span><span>退出</span>
    </button>
  </div>
</header>

<!-- Token 用量弹窗 -->
<div class="modal-mask" id="usageModal" onclick="if(event.target===this)closeUsage()">
  <div class="modal">
    <div class="modal-head">
      <h3>📊 我的 Token 用量 <span style="font-size:12px;font-weight:400;color:var(--text-2)">— <span id="usageUser">guest</span></span></h3>
      <button class="modal-close" onclick="closeUsage()">×</button>
    </div>
    <div class="modal-body">
      <div class="usage-toolbar">
        <div class="range-btn active" data-range="today" onclick="setRange('today')">今日</div>
        <div class="range-btn" data-range="7d" onclick="setRange('7d')">近 7 天</div>
        <div class="range-btn" data-range="30d" onclick="setRange('30d')">近 30 天</div>
        <div class="range-btn" data-range="all" onclick="setRange('all')">全部</div>
        <div class="usage-db-tag" id="usageDbTag">—</div>
      </div>
      <div id="usageContent">
        <div class="usage-empty">加载中…</div>
      </div>
    </div>
  </div>
</div>

<!-- 修改密码弹窗 -->
<div class="modal-mask" id="changePwdModal" onclick="if(event.target===this)closeChangePwd()">
  <div class="modal" style="max-width:420px">
    <div class="modal-head">
      <h3>🔑 修改密码</h3>
      <button class="modal-close" onclick="closeChangePwd()">×</button>
    </div>
    <div class="modal-body">
      <div class="pwd-row">
        <label>原密码</label>
        <input type="password" id="oldPwdInput" placeholder="请输入当前密码" autocomplete="off">
      </div>
      <div class="pwd-row">
        <label>新密码</label>
        <input type="password" id="newPwdInput" placeholder="至少 6 位" autocomplete="off">
      </div>
      <div class="pwd-row">
        <label>确认新密码</label>
        <input type="password" id="newPwdInput2" placeholder="再次输入新密码" autocomplete="off">
      </div>
      <div id="changePwdMsg" class="pwd-msg"></div>
      <button class="btn btn-primary" onclick="submitChangePwd()" style="margin-top:6px">确认修改</button>
    </div>
  </div>
</div>

<!-- Main Chat -->
<div class="main">
  <div class="chat-area" id="chatArea">
    <!-- Welcome -->
    <div class="welcome" id="welcome">
      <div class="welcome-icon">📚</div>
      <h2>欢迎使用企业知识库问答</h2>
      <p>基于 AI 的智能文档问答系统，支持自然语言提问，从企业文档中精准检索答案。</p>
      <div class="suggestions">
        <div class="suggestion" onclick="askSuggestion(this)">定位精度多少？几种定位方式？续航如何？</div>
        <div class="suggestion" onclick="askSuggestion(this)">通讯协议端口和心跳间隔是多少？</div>
        <div class="suggestion" onclick="askSuggestion(this)">设备支持哪些报警功能？</div>
        <div class="suggestion" onclick="askSuggestion(this)">待机时间120小时换算成天是多少？</div>
      </div>
    </div>
  </div>

    <!-- Input -->
  <div class="input-area">
    <!-- 断点重续横条：检测到上次未完成任务时显示 -->
    <div id="resumeBar" class="resume-bar" style="display:none">
      <span class="resume-icon">⏸️</span>
      <span class="resume-text">检测到上次未完成的任务：<b class="resume-query"></b></span>
      <button class="btn-resume" onclick="resumeFromBar()">继续</button>
      <button class="btn-resume-dismiss" onclick="dismissResumeBar()">忽略</button>
    </div>
    <div class="input-row">
      <textarea id="questionInput" rows="1" placeholder="输入您的问题，按 Enter 发送（Shift+Enter 换行）..."
        onkeydown="handleKeydown(event)"></textarea>
      <button class="btn-send" id="sendBtn" onclick="onSendBtnClick()" title="发送">➤</button>
    </div>
    <div class="input-hint" id="docHint">当前为普通用户模式，仅可访问公开文档；管理员请从右上角进入系统管理</div>
  </div>
</div>

<script>
// ============================================================
// Token 管理（登录态由统一登录页 /login 处理，本页仅校验与重定向）
// ============================================================
const RAG_TOKEN_KEY = 'rag_token';
function getRagToken(){ return localStorage.getItem(RAG_TOKEN_KEY) || ''; }
function setRagToken(t){ if(t) localStorage.setItem(RAG_TOKEN_KEY, t); else localStorage.removeItem(RAG_TOKEN_KEY); }

// 包装 fetch：自动带 Authorization；401/403 时只清 token + 控制台告警，不再自动弹回登录页
// （原行为会把任何 401/403 误判为「登录态失效」并 showLogin，导致登录后页面卡死）
(function(){
  const _orig = window.fetch.bind(window);
  window.fetch = function(url, opts){
    opts = opts || {};
    const t = getRagToken();
    if(t){
      opts.headers = Object.assign({}, opts.headers);
      opts.headers['Authorization'] = 'Bearer ' + t;
    }
    return _orig(url, opts).then(function(res){
      if(res.status === 401 || res.status === 403){
        console.warn('[rag-auth] 接口 ' + url + ' 返回 ' + res.status + '，可能是登录态失效');
        // 不再自动清 token / 弹登录页：避免误伤（任意接口 401 都会触发）
        // 改由各业务函数自行决定如何处理（如 openUsage 在 401 时显示错误）
      }
      return res;
    });
  };
})();

async function bootstrapAuth(){
  const token = getRagToken();
  if(!token){ location.replace('/login'); return; }
  try{
    const r = await fetch('/api/me', {headers:{'Authorization':'Bearer '+token}});
    if(r.ok){
      const u = await r.json();
      currentUser = u.username;
      currentRole = u.role;
      // admin 应在 /admin 后台工作，不在聊天页——强制跳走
      if (currentRole === 'admin') {
        location.replace('/admin');
        return;
      }
      document.documentElement.style.visibility = 'visible';  // 校验通过，显示页面
      renderUser();
      loadHistory();  // 刷新/重登后恢复历史对话
      checkUnfinishedTasks();  // 刷新/重登后检测上次未完成的任务，弹横条让用户确认是否继续
      return;
    }
  }catch(e){}
  setRagToken(''); location.replace('/login');
}

// ============================================================
// 退出登录：清 token + 调 /api/logout + 跳登录页
// ============================================================
function doLogout(){
  fetch('/api/logout', {method:'POST'}).catch(function(){});
  setRagToken('');
  location.replace('/login');
}

// ============================================================
// 修改密码（聊天页，普通用户/任意登录用户通用）
// ============================================================
function openChangePwd(){
  document.getElementById('oldPwdInput').value = '';
  document.getElementById('newPwdInput').value = '';
  document.getElementById('newPwdInput2').value = '';
  const msg = document.getElementById('changePwdMsg');
  msg.textContent = ''; msg.className = 'pwd-msg';
  document.getElementById('changePwdModal').classList.add('show');
}
function closeChangePwd(){
  document.getElementById('changePwdModal').classList.remove('show');
}
async function submitChangePwd(){
  const oldPwd = document.getElementById('oldPwdInput').value;
  const newPwd = document.getElementById('newPwdInput').value;
  const newPwd2 = document.getElementById('newPwdInput2').value;
  const msg = document.getElementById('changePwdMsg');
  msg.className = 'pwd-msg';

  if (!oldPwd || !newPwd || !newPwd2){
    msg.className = 'pwd-msg err';
    msg.textContent = '请填写完整';
    return;
  }
  if (newPwd.length < 6){
    msg.className = 'pwd-msg err';
    msg.textContent = '新密码至少 6 位';
    return;
  }
  if (newPwd !== newPwd2){
    msg.className = 'pwd-msg err';
    msg.textContent = '两次输入的新密码不一致';
    return;
  }

  try{
    const r = await fetch('/api/change-password', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({old_password: oldPwd, new_password: newPwd})
    });
    const d = await r.json();
    if (d.success){
      msg.className = 'pwd-msg ok';
      msg.textContent = d.message || '密码修改成功';
      showToast(d.message || '密码修改成功', 'success');
      setTimeout(closeChangePwd, 900);
    } else {
      msg.className = 'pwd-msg err';
      msg.textContent = d.error || '修改失败';
    }
  }catch(e){
    msg.className = 'pwd-msg err';
    msg.textContent = '网络错误: ' + e.message;
  }
}

// 聊天页轻量 toast（与后台 showToast 同款）
function showToast(msg, type){
  const t = document.createElement('div');
  t.className = 'toast ' + (type || 'info');
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(function(){ t.remove(); }, 3000);
}

document.addEventListener('DOMContentLoaded', function(){
  // userChip 仅展示(显示当前登录账号),不绑定任何点击行为
  // 退出登录改由独立的 #logoutBtn 处理(doLogout)
  bootstrapAuth();
});

// ============================================================
// 全局状态
// ============================================================
let currentRole = 'user';
let isQuerying = false;
// 用量归属账号：登录后由 token 决定（见 bootstrapAuth），不再允许游客随意填写
let currentUser = '';
let usageRange = 'today';
// 当前查询的 AbortController：用于主动中断对话（点击"停止"按钮时 abort）
let currentAbortController = null;

// ============================================================
// 用量归属账号
// ============================================================
function renderUser() {
  document.getElementById('userLabel').textContent = currentUser;
  document.getElementById('usageUser').textContent = currentUser;
  // 系统管理入口只对 admin 可见（普通用户不应看见后台按钮）
  const adminLink = document.getElementById('adminLink');
  if (adminLink) {
    adminLink.style.display = (currentRole === 'admin') ? '' : 'none';
  }
}

function changeUser() {
  // 已改为登录态驱动，不再允许游客随意修改用量归属账号
  alert('用量归属账号以登录账号为准，无需手动切换。退出请点击右上角账号。');
}

// ============================================================
// 「我的用量」弹窗
// ============================================================
function openUsage() {
  document.getElementById('usageModal').classList.add('show');
  loadUsage();
}

function closeUsage() {
  document.getElementById('usageModal').classList.remove('show');
}

function setRange(r) {
  usageRange = r;
  document.querySelectorAll('.range-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.range === r));
  loadUsage();
}

function fmtNum(n) {
  return (n || 0).toLocaleString('en-US');
}

function fmtTime(ts) {
  if (!ts) return '—';
  const d = new Date(ts * 1000);
  const p = n => String(n).padStart(2, '0');
  return `${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

async function loadUsage() {
  const box = document.getElementById('usageContent');
  box.innerHTML = '<div class="usage-empty">加载中…</div>';
  try {
    const url = `/api/usage/me?user=${encodeURIComponent(currentUser)}&range=${usageRange}&limit=100`;
    const resp = await fetch(url);
    const data = await resp.json();
    if (!resp.ok) {
      box.innerHTML = `<div class="usage-empty">❌ ${data.error || '查询失败'}</div>`;
      return;
    }
    document.getElementById('usageDbTag').textContent =
      data.persisted ? `已持久化 · ${data.db}` : '仅内存（重启即丢）';
    renderUsage(box, data);
  } catch (e) {
    box.innerHTML = '<div class="usage-empty">❌ 网络异常，无法获取用量</div>';
  }
}

function renderUsage(box, data) {
  const w = data.window || {};
  const lt = data.lifetime || {};
  const cards = `
    <div class="stat-grid">
      <div class="stat-card primary">
        <div class="label">调用次数</div>
        <div class="value">${fmtNum(w.calls)}</div>
        <div class="sub">累计 ${fmtNum(lt.calls)} 次</div>
      </div>
      <div class="stat-card success">
        <div class="label">Token 总量</div>
        <div class="value">${fmtNum(w.total_tokens)}</div>
        <div class="sub">累计 ${fmtNum(lt.total_tokens)}</div>
      </div>
      <div class="stat-card">
        <div class="label">输入 / 输出</div>
        <div class="value" style="font-size:16px">${fmtNum(w.prompt_tokens)} / ${fmtNum(w.completion_tokens)}</div>
        <div class="sub">prompt / completion</div>
      </div>
      <div class="stat-card warning">
        <div class="label">累计成本</div>
        <div class="value" style="font-size:18px">$${(w.cost_usd || 0).toFixed(4)}</div>
        <div class="sub">平均耗时 ${(w.avg_latency_s || 0).toFixed(2)}s</div>
      </div>
    </div>`;

  const rows = data.rows || [];
  let table;
  if (!rows.length) {
    table = '<div class="usage-empty">该时间范围内没有调用记录<br><span style="font-size:12px">提问几次后再回来看看 👀</span></div>';
  } else {
    const trs = rows.map(r => `
      <tr>
        <td>${fmtTime(r.ts)}</td>
        <td class="tag-model">${r.model || '—'}</td>
        <td><span class="tag-task">${r.task || 'default'}</span></td>
        <td class="num">${fmtNum(r.prompt_tokens)}</td>
        <td class="num">${fmtNum(r.completion_tokens)}</td>
        <td class="num"><b>${fmtNum(r.total_tokens)}</b></td>
        <td class="num">${(r.latency_s || 0).toFixed(2)}s</td>
        <td class="num">$${(r.cost_usd || 0).toFixed(5)}</td>
      </tr>`).join('');
    table = `
      <div class="table-wrap">
        <table class="usage-table">
          <thead><tr>
            <th>时间</th><th>模型</th><th>任务</th>
            <th class="num">输入</th><th class="num">输出</th><th class="num">合计</th>
            <th class="num">耗时</th><th class="num">成本</th>
          </tr></thead>
          <tbody>${trs}</tbody>
        </table>
      </div>`;
  }

  box.innerHTML = cards +
    `<div class="usage-section-title">调用明细（最近 ${rows.length} 条）</div>` + table;
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeUsage();
});

// ============================================================
// 初始化
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
  renderUser();
  fetch('/api/health')
    .then(r => r.json())
    .then(data => {
      if (data.role) {
        currentRole = data.role;
        updateRoleUI();
      }
    })
    .catch(() => console.warn('Health check failed'));
});

// ============================================================
// 断点重续：登录后检测上次未完成的任务，弹出友好横条让用户确认是否继续
// （不再用浏览器原生 confirm，避免被弹窗拦截；session_id 由后端按 token 派生）
// ============================================================
function checkUnfinishedTasks() {
  fetch('/api/tasks/unfinished')
    .then(r => r.json())
    .then(data => {
      if (!data.tasks || data.tasks.length === 0) return;
      const task = data.tasks[0];
      const bar = document.getElementById('resumeBar');
      if (!bar) return;
      bar.querySelector('.resume-query').textContent = task.query;
      bar.dataset.taskId = task.task_id;
      bar.style.display = 'flex';
    })
    .catch(() => console.warn('Unfinished tasks check failed'));
}

function resumeFromBar() {
  const bar = document.getElementById('resumeBar');
  if (!bar) return;
  const taskId = bar.dataset.taskId;
  if (!taskId) return;
  bar.style.display = 'none';
  // 把原问题当作一次新提问渲染出来，并恢复执行
  sendQuestionWith(taskId, bar.querySelector('.resume-query').textContent);
}

function dismissResumeBar() {
  const bar = document.getElementById('resumeBar');
  if (bar) bar.style.display = 'none';
}

// ============================================================
// 角色状态：聊天页只服务非 admin 用户，admin 在 /admin 后台使用
// ============================================================
function updateRoleUI() {
  const badge = document.getElementById('roleBadge');
  const label = document.getElementById('roleLabel');
  const dot = badge.querySelector('.role-dot');
  const hint = document.getElementById('docHint');

  // 聊天页强制按普通用户展示（admin 也只看文档问答，不会在这里看到管理入口）
  badge.className = 'role-badge user';
  dot.className = 'role-dot user';
  label.textContent = '普通用户';
  hint.innerHTML = '当前为普通用户模式，仅可访问公开文档';
}

// ============================================================
// 发送问题
// ============================================================
function handleKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
}

function askSuggestion(el) {
  document.getElementById('questionInput').value = el.textContent;
  sendQuestion();
}

// 刷新/重登后从后端拉取历史对话并渲染（数据在 MySQL，不在前端内存）
async function loadHistory() {
  try {
    const r = await fetch('/api/history', {
      headers: {'Authorization': 'Bearer ' + getRagToken()}
    });
    if (!r.ok) return;
    const data = await r.json();
    const msgs = data.messages || [];
    if (!msgs.length) return;
    // 有历史时隐藏欢迎页
    const welcome = document.getElementById('welcome');
    if (welcome) welcome.style.display = 'none';
    for (const m of msgs) {
      if (m.role === 'user') addUserMessage(m.content);
      else if (m.role === 'assistant') addAssistantMessage(m.content);
    }
  } catch (e) {
    // 历史拉取失败不影响新对话，静默忽略
  }
}

async function sendQuestion() {
  const input = document.getElementById('questionInput');
  const question = input.value.trim();
  if (!question || isQuerying) return;

  // 清空输入
  input.value = '';
  input.style.height = 'auto';

  // 隐藏欢迎页
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.style.display = 'none';

  // 添加用户消息
  addUserMessage(question);

  // 开始查询
  isQuerying = true;
  updateSendButton();

  // 添加进度框
  const progressBox = addProgressBox();

  // 启动 SSE 流
  // 主动中断用：把 controller 存到 currentAbortController，让"停止"按钮可以 abort
  currentAbortController = new AbortController();
  try {
    const resp = await fetch('/api/query/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question, role: currentRole, username: currentUser}),
      signal: currentAbortController.signal
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finalAnswer = null;

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            handleSSEEvent(data, progressBox, question);
            if (data.type === 'done') {
              finalAnswer = data.answer;
              if (data.role) currentRole = data.role;
            } else if (data.type === 'error') {
              finalAnswer = null;
            }
          } catch (e) {
            // 忽略解析错误
          }
        }
      }
    }

    // 移除进度框
    if (progressBox.parentNode) {
      progressBox.remove();
    }

    // 添加最终答案
    if (finalAnswer) {
      addAssistantMessage(finalAnswer);
    } else {
      addErrorMessage('查询过程出现异常，请稍后重试。');
    }

  } catch (e) {
    if (progressBox.parentNode) progressBox.remove();
    if (e.name === 'AbortError') {
      // 用户主动中断：友好提示（不是错误）
      addAssistantMessage('⏹ 已中断当前回答。');
    } else {
      addErrorMessage('连接中断: ' + e.message);
    }
  }

  currentAbortController = null;
  isQuerying = false;
  updateSendButton();
  updateRoleUI();
}

// 从断点重续横条触发：复用原问题，调 /api/tasks/resume 恢复上次未完成的任务
async function sendQuestionWith(taskId, question) {
  if (!question || isQuerying) return;
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.style.display = 'none';

  addUserMessage(question);

  // 用进度框替换原"⏳ 正在恢复..."占位气泡，跟普通 query 风格一致
  removeLastAssistant();
  const progressBox = addProgressBox();
  document.getElementById('chatArea').appendChild(progressBox);
  progressBox.scrollIntoView({behavior: 'smooth', block: 'end'});

  isQuerying = true;
  updateSendButton();

  // 关键：把 AbortController 挂到全局，让刚加的 ⏹ 中断按钮能真正中止
  currentAbortController = new AbortController();

  try {
    const resp = await fetch('/api/tasks/resume', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({task_id: taskId}),
      signal: currentAbortController.signal,
    });

    if (!resp.ok && resp.status !== 200) {
      // 非流式错误（如 400/500）
      let errText = '恢复失败';
      try {
        const j = await resp.json();
        errText = j.error || errText;
      } catch (_) {}
      if (progressBox.parentNode) progressBox.remove();
      addErrorMessage('恢复失败：' + errText);
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finalAnswer = null;
    let errored = false;

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const data = JSON.parse(line.slice(6));
          handleSSEEvent(data, progressBox, question);
          if (data.type === 'done') finalAnswer = data.answer;
          else if (data.type === 'error') errored = true;
        } catch (_) {}
      }
    }

    if (progressBox.parentNode) progressBox.remove();
    if (errored) {
      addErrorMessage('恢复过程出现异常，请稍后重试。');
    } else if (finalAnswer) {
      addAssistantMessage(finalAnswer);
    } else {
      addErrorMessage('恢复过程出现异常，请稍后重试。');
    }
  } catch (e) {
    if (progressBox.parentNode) progressBox.remove();
    if (e.name === 'AbortError') {
      addAssistantMessage('⏹ 已中断断点恢复');
    } else {
      addErrorMessage('连接中断: ' + e.message);
    }
  } finally {
    isQuerying = false;
    currentAbortController = null;
    updateSendButton();
  }
}

function removeLastAssistant() {
  const area = document.getElementById('chatArea');
  if (!area) return;
  const last = area.lastElementChild;
  if (last && last.classList.contains('message')) {
    last.remove();
  }
}

function handleSSEEvent(data, progressBox, question) {
  const logsEl = progressBox.querySelector('.progress-logs');
  const headerSpan = progressBox.querySelector('.progress-header span');

  switch (data.type) {
    case 'start':
      headerSpan.textContent = '正在分析问题...';
      break;
    case 'log':
      const text = data.text;
      // 更新状态
      if (text.includes('用户提问')) {
        headerSpan.textContent = '正在检索文档...';
      } else if (text.includes('子任务') && text.includes('开始执行')) {
        headerSpan.textContent = '正在执行子任务...';
      } else if (text.includes('最终回答') || text.includes('从缓存返回')) {
        headerSpan.textContent = '正在生成答案...';
      }
      // 添加日志行
      if (text.length < 200) {
        const div = document.createElement('div');
        div.className = 'log-line';
        div.textContent = text;
        logsEl.appendChild(div);
        logsEl.scrollTop = logsEl.scrollHeight;
      }
      break;
    case 'done':
      headerSpan.textContent = '✓ 回答完成';
      progressBox.querySelector('.spinner').style.display = 'none';
      break;
    case 'error':
      headerSpan.textContent = '✗ 处理出错';
      progressBox.querySelector('.spinner').style.display = 'none';
      const errDiv = document.createElement('div');
      errDiv.className = 'log-line';
      errDiv.style.color = 'var(--danger)';
      errDiv.textContent = data.text;
      logsEl.appendChild(errDiv);
      break;
  }
}

// ============================================================
// UI 工具函数
// ============================================================
function addUserMessage(text) {
  const area = document.getElementById('chatArea');
  const div = document.createElement('div');
  div.className = 'message';
  div.innerHTML = `
    <div class="msg-avatar user">U</div>
    <div class="msg-body">
      <div class="msg-role-name">你</div>
      <div class="msg-content">${escapeHtml(text)}</div>
    </div>
  `;
  area.appendChild(div);
  scrollToBottom();
}

function addAssistantMessage(text) {
  const area = document.getElementById('chatArea');
  const div = document.createElement('div');
  div.className = 'message';
  // 简单 Markdown 渲染（标题、粗体、段落）
  const html = simpleMarkdown(text);
  div.innerHTML = `
    <div class="msg-avatar assistant">AI</div>
    <div class="msg-body">
      <div class="msg-role-name">AI 助手</div>
      <div class="msg-content">${html}</div>
      <div class="msg-time">${formatTime()}</div>
    </div>
  `;
  area.appendChild(div);
  scrollToBottom();
}

function addProgressBox() {
  const area = document.getElementById('chatArea');
  const div = document.createElement('div');
  div.className = 'progress-box';
  div.innerHTML = `
    <div class="progress-header">
      <div class="spinner"></div>
      <span>正在分析问题...</span>
    </div>
    <div class="progress-logs"></div>
  `;
  area.appendChild(div);
  scrollToBottom();
  return div;
}

function addErrorMessage(text) {
  const area = document.getElementById('chatArea');
  const div = document.createElement('div');
  div.className = 'error-box';
  div.textContent = '⚠ ' + text;
  area.appendChild(div);
  scrollToBottom();
}

function updateSendButton() {
  const btn = document.getElementById('sendBtn');
  if (isQuerying) {
    // 查询中：变成"停止"按钮（红色 ⏹，点击触发 abort）
    btn.classList.add('stopping');
    btn.textContent = '⏹';
    btn.title = '点击中断当前回答';
    btn.disabled = false;  // 必须可点，所以单独控制
  } else {
    btn.classList.remove('stopping');
    btn.textContent = '➤';
    btn.title = '发送';
    btn.disabled = false;
  }
}

// 发送按钮的真正入口：查询中点击=abort，空闲点击=发送
function onSendBtnClick() {
  if (isQuerying) {
    abortCurrentQuery();
  } else {
    sendQuestion();
  }
}

function abortCurrentQuery() {
  if (currentAbortController) {
    try { currentAbortController.abort(); } catch (e) {}
  }
  // 后端 daemon 线程会继续跑完，但前端立刻停止显示（用户视角的"中断"）
  // 若需要"硬中断"（停后端 LangGraph 任务），需在后端节点协作 CheckpointSaver
}

function scrollToBottom() {
  const area = document.getElementById('chatArea');
  requestAnimationFrame(() => {
    area.scrollTop = area.scrollHeight;
  });
}

// ============================================================
// 简单 Markdown 渲染
// ============================================================
function simpleMarkdown(text) {
  if (!text) return '';
  const lines = text.split('\n');
  let html = '';
  let inList = false;

  for (let i = 0; i < lines.length; i++) {
    let line = lines[i].trim();
    if (!line) {
      if (inList) { html += '</ul>'; inList = false; }
      continue;
    }

    // 标题
    if (/^### /.test(line)) {
      if (inList) { html += '</ul>'; inList = false; }
      html += `<h3>${escapeHtml(line.slice(4))}</h3>`;
      continue;
    }
    if (/^## /.test(line)) {
      if (inList) { html += '</ul>'; inList = false; }
      html += `<h3>${escapeHtml(line.slice(3))}</h3>`;
      continue;
    }

    // 列表项
    if (/^[-*]\s/.test(line)) {
      if (!inList) { html += '<ul>'; inList = true; }
      const content = line.slice(2);
      html += `<li>${inlineFormat(content)}</li>`;
      continue;
    }

    // 编号列表
    if (/^\d+[\.、]\s/.test(line)) {
      if (!inList) { html += '<ul>'; inList = true; }
      const content = line.replace(/^\d+[\.、]\s*/, '');
      html += `<li>${inlineFormat(content)}</li>`;
      continue;
    }

    // 普通段落
    if (inList) { html += '</ul>'; inList = false; }
    html += `<p>${inlineFormat(line)}</p>`;
  }

  if (inList) html += '</ul>';
  return html;
}

function inlineFormat(text) {
  let t = escapeHtml(text);
  // 粗体
  t = t.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // 行内代码
  t = t.replace(/`([^`]+)`/g, '<code style="background:#f3f4f6;padding:1px 4px;border-radius:3px;font-family:monospace">$1</code>');
  return t;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function formatTime() {
  const now = new Date();
  return now.toLocaleTimeString('zh-CN', {hour:'2-digit', minute:'2-digit'});
}
</script>

</body>
</html>
"""


# ======================================================================
# 启动函数
# ======================================================================

def init_system():
    """初始化 LLM + 向量数据库 + 编排器"""
    global orchestrator, llm, vector_db, use_langgraph

    print("\n" + "=" * 60)
    print("  初始化 RAG Agent 系统...")
    print("=" * 60)

    # 1. LLM（企业级网关：多模型路由 / 限流 / 熔断 / token 计费）
    try:
        llm = create_llm()
        if hasattr(llm, "resolve_chain"):
            print(f"  ✓ LLM 网关: 默认链 {llm.resolve_chain('default')}")
        else:
            print(f"  ✓ LLM: {MODEL_NAME} @ {OLLAMA_URL}")
    except Exception as e:
        print(f"  ✗ LLM 初始化失败: {e}")
        print("    请确认 Ollama 已运行且已加载模型")
        sys.exit(1)

    # 2. 向量数据库（多 worker 安全：Redis 锁防并发重建 Milvus 集合）
    try:
        vector_db = _init_vector_store_locked()
        print(f"  ✓ 向量数据库: VECTOR_BACKEND={os.getenv('VECTOR_BACKEND', 'milvus')}"
              f"（Chroma 兜底路径 {DB_PATH}）")
    except Exception as e:
        print(f"  ✗ 向量数据库初始化失败: {e}")
        sys.exit(1)

    # 3. 编排器
    if use_langgraph:
        orchestrator = LangGraphEngine(fast_mode=True, user_role=DEFAULT_ROLE)
        print(f"  ✓ LangGraph Engine 就绪（多轮检索+多智能体+多轮对话）")
    else:
        orchestrator = RAGOrchestrator(llm, vector_db, fast_mode=True, user_role=DEFAULT_ROLE)
        print(f"  ✓ RAG Orchestrator 就绪")
    print(f"  ✓ 用户角色: {DEFAULT_ROLE}")
    print(f"  ✓ Web 界面已启动\n")


def _init_vector_store_locked():
    """
    多 worker 安全初始化向量库。

    gunicorn 多 worker 时每个 worker 都会调 init_system()，若同时触发
    Milvus 集合重建（维度变更/集合不存在），并发 drop+create 会冲突报错。
    用 Redis 分布式锁保证只有一个 worker 执行完整重建，其余等待后只连接。
    无 Redis（单实例开发）时直接初始化，无并发风险。
    """
    try:
        import redis as _redis
        import time as _t
        r = _redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD,
                         db=REDIS_DB, socket_connect_timeout=5, socket_timeout=5)
        r.ping()
    except Exception:
        return VectorStoreManager.init_vector_store()

    lock_key = "rag:init:vectorstore:lock"
    if r.set(lock_key, "1", nx=True, ex=600):
        try:
            return VectorStoreManager.init_vector_store()
        finally:
            r.delete(lock_key)
    # 未持锁：等待持锁 worker 建好集合（最多 2 分钟），再只连接不重建
    deadline = _t.time() + 120
    while _t.time() < deadline and r.exists(lock_key):
        _t.sleep(1)
    return VectorStoreManager.init_vector_store()


def main():
    global use_langgraph
    import argparse
    parser = argparse.ArgumentParser(description="RAG Agent Web Server")
    parser.add_argument("--port", type=int, default=8080, help="Web 服务器端口 (默认 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="绑定的主机地址")
    parser.add_argument(
        "--langgraph",
        action=argparse.BooleanOptionalAction,
        default=use_langgraph,
        help="使用 LangGraph 引擎（默认开启，--no-langgraph 使用旧版）",
    )
    args = parser.parse_args()

    use_langgraph = args.langgraph

    init_system()

    url = f"http://localhost:{args.port}"
    print(f"  🌐 浏览器打开: {url}")
    print(f"  按 Ctrl+C 停止服务器\n")

    # 开发/单进程模式。生产高并发请用 gunicorn（见 gunicorn_config.py）：
    #   gunicorn -c gunicorn_config.py rag_web_server:app
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
