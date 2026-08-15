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
  - 向量检索：Milvus（唯一向量后端）；Embedding 默认 Ollama bge-m3

  启动方式：
    python rag_web_server.py              # 默认端口 8080
    python rag_web_server.py --port 9090  # 指定端口

================================================================================
"""

# ====== 全局日志加时间戳（必须在所有 import 之前） ======
# 包一层 builtins.print，使后续所有 print 自动带 [YYYY-MM-DD HH:MM:SS] 前缀
import logutil  # noqa: F401  仅副作用：替换 builtins.print

import io
import os
import sys
import json
import time
import queue
import threading
import secrets
import uuid
from pathlib import Path

# ====== 环境配置（必须在所有 import 之前） ======
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, Response, g, send_from_directory
from werkzeug.security import safe_join

# ====== 导入核心模块 ======
from audit_logger import get_audit_logger
from advanced_rag_agent import (
    OLLAMA_URL, MODEL_NAME,
    ROLE_ADMIN, ROLE_SUPER_ADMIN, DEFAULT_ROLE, DOC_FOLDER,
    AccessControlFilter, CacheManager,
    RAGOrchestrator,
    OllamaLLM,
    create_llm,
    VectorStoreManager,
    REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB,
)

# ====== 导入摄取管线（数据面） ======
from ingest.pipeline import IngestPipeline
from ingest.store import MilvusStoreBackend
from ingest.fingerprint import ManifestStore


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

    def query(self, question, user_role=None, user=None, user_id=None, tenant_id=None):
        role = user_role or self.user_role
        username = user or role
        return self.app.query(question, role=role,
                              session_id=_derive_session_id(username, role),
                              user=username, user_id=user_id, tenant_id=tenant_id)

    @property
    def last_task_id(self):
        """
        透传底层 App 最近一次问答的任务 ID（请求级 ContextVar）。

        必须在**执行 query 的那个线程内**读取，跨线程读会拿到 None。
        前端用它把用户反馈（点赞/点踩/纠错）挂到对应的全链路 trace 上。
        """
        return getattr(self.app, "last_task_id", None)

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

class ThreadRoutedStdout(io.StringIO):
    """
    按「线程」路由的 stdout 多路复用器。

    【为什么要有它 —— 原实现的并发缺陷】
    旧写法是每个 SSE 请求各建一个 ProgressWriter 并 `sys.stdout = writer`，
    结束时再 `sys.stdout = original`。但 sys.stdout 是**进程级全局**的：

        请求A: sys.stdout = WriterA
        请求B: sys.stdout = WriterB          # 覆盖了 A
        A 的管线 print(...)  → 全部流进 B 的 SSE 通道（A 页面卡住不动，B 看到别人的日志）
        请求A 结束: sys.stdout = original    # B 还在跑，却被提前恢复，B 后半段日志全丢

    也就是说：日志串台 + 日志丢失 + 潜在的隐私泄漏（A 的问题内容出现在 B 的屏幕上）。

    【怎么修】
    全局只安装一次，永不恢复。内部维护 {线程ID: 队列} 路由表：
      - 工作线程开始前调用 register(q) 把自己登记进去
      - write() 时按 threading.get_ident() 找到本线程的队列，只投递给它
      - 没登记的线程（如 Flask 主线程、后台定时任务）照常写原始 stdout，不进任何队列
    注册与写入都发生在同一个工作线程内，所以路由天然正确。
    """

    def __init__(self, original_stdout):
        super().__init__()
        self._original = original_stdout
        self._routes: dict[int, queue.Queue] = {}
        self._lock = threading.Lock()

    def register(self, q: queue.Queue):
        """在当前线程内登记接收队列（必须由工作线程自己调用）。"""
        with self._lock:
            self._routes[threading.get_ident()] = q

    def unregister(self):
        """注销当前线程的接收队列（放在 finally 里，确保线程退出前清理）。"""
        with self._lock:
            self._routes.pop(threading.get_ident(), None)

    def write(self, s):
        if s.strip():
            # 不加锁读：dict.get 在 CPython 下是原子的，
            # 且此处只读自己线程的 key，避免每行日志都抢锁拖慢管线。
            q = self._routes.get(threading.get_ident())
            if q is not None:
                q.put({"type": "log", "text": s.strip()})
        self._original.write(s)
        self._original.flush()

    def flush(self):
        self._original.flush()


# 全局唯一实例：进程启动时安装一次，此后 sys.stdout 不再被任何请求替换。
# 注意要在 logutil 替换 builtins.print 之后取 sys.stdout，
# 这样原始输出链路（控制台/日志文件）保持不变。
_stdout_mux = ThreadRoutedStdout(sys.stdout)
sys.stdout = _stdout_mux


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
        "vector_db": "milvus",
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
    # role/user/user_id/tenant_id 一律来自登录态，客户端不可伪造（防普通用户提权看受限文档）
    user_role = g.current_user["role"]
    user = g.current_user["username"]
    user_id = g.current_user["user_id"]
    tenant_id = g.current_user.get("tenant_id", "default")

    err = validate_input(question, MAX_QUESTION_LEN, "问题")
    if err:
        return jsonify({"error": err}), 400

    result = orchestrator.query(question, user_role=user_role, user=user,
                                user_id=user_id, tenant_id=tenant_id)
    # 非流式接口在 Flask 请求线程内同步执行，可直接读到本上下文的 task_id
    task_id = getattr(orchestrator, "last_task_id", None)
    _audit_log("query", target=question[:80], username=user)
    return jsonify({"answer": result, "role": user_role, "task_id": task_id})


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
    # role/user/user_id/tenant_id 一律来自登录态，客户端不可伪造（防普通用户提权看受限文档）
    user_role = g.current_user["role"]
    user = g.current_user["username"]
    user_id = g.current_user["user_id"]
    tenant_id = g.current_user.get("tenant_id", "default")

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

            # 在后台线程执行查询。该线程的 print 输出由 stdout 多路复用器
            # 按线程 ID 精确路由到本请求的队列 —— 不再全局替换 sys.stdout，
            # 因此多个用户并发提问时日志不会串台、也不会被提前恢复而丢失。
            output_queue = queue.Queue()

            result_holder = {"answer": None, "error": None, "task_id": None}

            def run_query():
                _stdout_mux.register(output_queue)
                try:
                    result_holder["answer"] = orchestrator.query(
                        question, user_role=user_role, user=user,
                        user_id=user_id, tenant_id=tenant_id
                    )
                    # task_id 必须在**工作线程内**读取：它存放在 ContextVar 中，
                    # 而 SSE 生成器跑在 Flask 请求线程，读不到本线程的上下文。
                    result_holder["task_id"] = getattr(orchestrator, "last_task_id", None)
                except Exception as e:
                    import traceback
                    result_holder["error"] = str(e)
                    output_queue.put({
                        "type": "log",
                        "text": f"[ERROR] {traceback.format_exc()}"
                    })
                finally:
                    _stdout_mux.unregister()

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

            # 无需恢复 stdout：多路复用器是全局常驻的，
            # 工作线程退出前已在 finally 中 unregister 自己的路由。

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
                # role 用请求局部变量，不读全局 orchestrator.user_role（并发下会是别人的角色）。
                # task_id 透出给前端：点赞/点踩/纠错时带回来，即可关联到
                # task_checkpoints 里那一整条 13 节点全链路 trace。
                done_evt = {
                    "type": "done",
                    "answer": result_holder["answer"],
                    "role": user_role,
                    "task_id": result_holder["task_id"],
                }
                yield f"data: {json.dumps(done_evt, ensure_ascii=False)}\n\n"

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
    if g.current_user.get("role") not in (ROLE_ADMIN, ROLE_SUPER_ADMIN):
        return jsonify({"error": "需要管理员或超级管理员权限"}), 403
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

    复用 /api/query/stream 的 stdout 多路复用 + 后台线程模式：resume 内部
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

        result_holder = {"answer": None, "error": None}

        def run_resume():
            # 与 /api/query/stream 同理：由工作线程自己登记日志路由，
            # 退出前在 finally 注销，避免并发 resume 之间日志串台。
            _stdout_mux.register(output_queue)
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
            finally:
                _stdout_mux.unregister()

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

        # stdout 无需恢复（全局多路复用器常驻，路由已由工作线程自行注销）

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
                "tenant_id": user.get("tenant_id", "default"),
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
        "tenant_id": g.current_user.get("tenant_id", "default"),
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
                "tenant_id": user.get("tenant_id", "default"),
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


# --------------------------------------------------------------------------- #
# PDF 页面渲染图静态访问（PDF 整页 PNG，由 ingest 阶段预渲染到 assets/figures/）
# --------------------------------------------------------------------------- #
FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "figures"
)


@app.route("/api/figures/<path:subpath>", methods=["GET"])
def api_figures(subpath):
    """安全返回 PDF 真图裁剪图（PyMuPDF 在 ingest 阶段连通分量抽取到 assets/figures/）。

    仅允许访问 .png 结尾、且 safe_join 校验在 FIGURES_DIR 内的文件，
    防止路径穿越（../、绝对路径等）。
    """
    # 拒绝非 png（其他后缀一律 404，避免被当成目录泄露）
    if not subpath.lower().endswith(".png"):
        return jsonify({"error": "only png allowed"}), 404
    safe = safe_join(FIGURES_DIR, subpath.replace("\\", "/"))
    if not safe or not os.path.isfile(safe):
        return jsonify({"error": "not found"}), 404
    # 必须落在 FIGURES_DIR 之下（防止 safe_join 之外的手法）
    safe_abs = os.path.abspath(safe)
    if not safe_abs.startswith(os.path.abspath(FIGURES_DIR)):
        return jsonify({"error": "forbidden"}), 403
    return send_from_directory(os.path.dirname(safe_abs),
                               os.path.basename(safe_abs),
                               mimetype="image/png",
                               max_age=3600)


# ====================================================================== #
# 图片水印（admin 专用：上传图片 + 斜铺水印，存 knowledge/pic）
# ====================================================================== #
PIC_DIR = os.path.join(DOC_FOLDER, "pic")


@app.route("/api/admin/watermark", methods=["POST"])
def api_admin_watermark():
    """管理员上传图片并生成斜铺水印图，存到 knowledge/pic（不存在自动创建）。"""
    denied = _require_admin()
    if denied:
        return denied
    if "file" not in request.files:
        return jsonify({"error": "缺少图片文件"}), 400
    f = request.files["file"]
    text = (request.form.get("text") or "").strip()
    date = (request.form.get("date") or "").strip()
    if not text:
        return jsonify({"error": "水印文字不能为空"}), 400
    if not f.filename:
        return jsonify({"error": "无效文件"}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        return jsonify({"error": "仅支持 png/jpg/jpeg/webp/bmp 图片"}), 400
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageFilter
        img = Image.open(f.stream).convert("RGBA")
        W, H = img.size
        # 字体：优先微软雅黑（含中文），缺失则回退默认，保证中文不方块
        try:
            font = ImageFont.truetype(
                "C:/Windows/Fonts/msyh.ttc",
                max(12, min(18, int(min(W, H) * 0.015))),
            )
        except Exception:
            font = ImageFont.load_default()
        content = text
        if date:
            # 自定义模式：日期追加到最后一行（保留多行能力）
            parts = content.split("\n")
            parts[-1] = f"{parts[-1]} · {date}"
            content = "\n".join(parts)
        lines = content.split("\n")
        # 透明水印层放大到 2 边距，旋转后仍能铺满整图
        pad = max(W, H)
        layer = Image.new("RGBA", (W + 2 * pad, H + 2 * pad), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        # 计算每行宽度与统一行高（支持多行水印）
        bboxes = [d.textbbox((0, 0), ln, font=font) for ln in lines]
        tw = max((b[2] - b[0]) for b in bboxes) if bboxes else 0
        th = max((b[3] - b[1]) for b in bboxes) if bboxes else 0
        line_gap = max(4, int(th * 0.3))
        block_h = th * len(lines) + line_gap * (len(lines) - 1)
        step_x = tw + max(15, int(W * 0.05))
        step_y = block_h + max(15, int(H * 0.05))
        # 网格双重绘制：深灰阴影 + 半透明白主色，任意背景可读
        for yy in range(0, layer.size[1], step_y):
            for xx in range(0, layer.size[0], step_x):
                y_off = 0
                for ln in lines:
                    d.text((xx + 2, yy + 2 + y_off), ln, font=font, fill=(40, 40, 40, 110))
                    d.text((xx, yy + y_off), ln, font=font, fill=(255, 255, 255, 90))
                    y_off += th + line_gap
        # 旋转 -30° 实现斜铺
        layer = layer.rotate(-30, expand=True, resample=Image.BICUBIC)
        lw, lh = layer.size
        layer = layer.crop(((lw - W) // 2, (lh - H) // 2,
                            (lw - W) // 2 + W, (lh - H) // 2 + H))
                # 合成到原图
        base = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        base.alpha_composite(img)
        base.alpha_composite(layer)
        os.makedirs(PIC_DIR, exist_ok=True)
        out_name = f"{os.path.splitext(f.filename)[0]}_watermark.png"
        base.convert("RGB").save(os.path.join(PIC_DIR, out_name), "PNG")
        rel = f"pic/{out_name}"
        try:
            get_audit_logger().log(action="watermark", target=rel,
                                   result="success", detail=f"生成水印图 {rel}")
        except Exception:
            pass
        return jsonify({"success": True, "url": f"/api/pic/{out_name}", "file": rel})
    except Exception as e:
        return jsonify({"error": f"生成失败：{e}"}), 500


@app.route("/api/pic/<path:filename>", methods=["GET"])
def api_pic(filename):
    """安全返回 knowledge/pic 下的水印图（仅 png，防路径穿越）。"""
    if not filename.lower().endswith(".png"):
        return jsonify({"error": "only png allowed"}), 404
    safe = safe_join(PIC_DIR, filename.replace("\\", "/"))
    if not safe or not os.path.isfile(safe):
        return jsonify({"error": "not found"}), 404
    safe_abs = os.path.abspath(safe)
    if not safe_abs.startswith(os.path.abspath(PIC_DIR)):
        return jsonify({"error": "forbidden"}), 403
    return send_from_directory(os.path.dirname(safe_abs),
                               os.path.basename(safe_abs),
                               mimetype="image/png",
                               max_age=3600)


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
        "tenant_id": g.current_user.get("tenant_id", "default"),
    })


@app.route("/api/tenants")
def api_tenants():
    """返回已注册租户列表（任意登录用户可见，用于 /kb 等页面下拉框）。"""
    auth_result = _require_auth()
    if auth_result:
        return auth_result
    return jsonify({"tenants": _load_tenants()})


def _get_memory_store():
    """兼容两种编排器，取到统一的 MySQLMemoryStore 实例。"""
    o = orchestrator
    if o is None:
        return None
    ms = getattr(o, "memory_store", None)
    if ms is None and getattr(o, "app", None) is not None:
        ms = getattr(o.app, "memory_store", None)
    return ms


def _get_playbook_store():
    """P1-7 L3：取到 PlaybookStore（进化经验库），用于反馈级信号回灌。

    仅 LangGraph 引擎持有 playbook_store；旧版 RAGOrchestrator 没有 → 返回 None。
    """
    o = orchestrator
    if o is None:
        return None
    store = getattr(o, "playbook_store", None)
    if store is None and getattr(o, "app", None) is not None:
        store = getattr(o.app, "playbook_store", None)
    return store


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
        # UI 展示最近 8 条 + 历史摘要卡片（避免滚太长）；
        # system 摘要转为 summary 角色，对用户可见，避免历史“消失”。
        raw = ms.load_messages(session_id, user_id=user["user_id"], limit=8)
        visible = []
        for m in raw:
            role = m.get("role")
            if role in ("user", "assistant"):
                visible.append({"role": role, "content": m["content"]})
            elif role == "system":
                # 压缩产物（前情提要）对用户可见，避免历史“消失”
                visible.append({"role": "summary", "content": m["content"]})
        return jsonify({"session_id": session_id, "messages": visible})
    except Exception as e:
        return jsonify({"session_id": session_id, "messages": [], "error": str(e)})


_BADCASE_PAGE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bad Case 复盘 - RAG Agent</title>
<style>
  :root{
    --bg:#f5f6fa;--surface:#fff;--border:#e2e5ed;
    --text:#1a1a2e;--text-2:#6b7280;--text-3:#9ca3af;
    --primary:#2563eb;--danger:#dc2626;--success:#16a34a;
    --radius:12px;--radius-sm:8px;
  }
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Noto Sans SC',sans-serif;background:var(--bg);color:var(--text)}
  .topbar{display:flex;align-items:center;justify-content:space-between;background:var(--surface);border-bottom:1px solid var(--border);padding:12px 22px}
  .topbar h1{font-size:17px;font-weight:600;margin:0;display:flex;align-items:center;gap:8px}
  .topbar .links{display:flex;gap:14px;align-items:center;font-size:13px}
  .topbar a{color:var(--primary);text-decoration:none}
  .wrap{max-width:1200px;margin:0 auto;padding:18px 22px}
  .stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px}
  .stat{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:14px 16px}
  .stat .l{font-size:13px;color:var(--text-2)}
  .stat .n{font-size:24px;font-weight:600;margin-top:4px}
  .stat .s{font-size:12px;margin-top:2px;color:#854F0B}
  .filterbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:12px 14px;margin-bottom:14px}
  .seg{display:inline-flex;border:1px solid var(--border);border-radius:var(--radius-sm);overflow:hidden}
  .seg button{border:none;background:var(--surface);padding:6px 14px;font-size:13px;color:var(--text-2);cursor:pointer}
  .seg button.active{background:var(--primary);color:#fff}
  .filterbar select,.filterbar input{padding:7px 10px;border:1px solid var(--border);border-radius:var(--radius-sm);font-size:13px;color:var(--text)}
  .filterbar .spacer{margin-left:auto}
  .layout{display:flex;gap:14px;align-items:flex-start}
  .listcard{flex:1.6;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;min-width:0}
  .detailcard{flex:1;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px;min-width:0;position:sticky;top:14px}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{text-align:left;padding:10px 12px;border-bottom:1px solid #f0f1f4}
  th{color:var(--text-2);font-weight:600;background:#fafbfc;font-size:12px}
  tr:hover td{background:#fafbff}
  .badge{font-size:12px;padding:2px 9px;border-radius:11px;display:inline-block}
  .b-open{background:#FAEEDA;color:#854F0B}
  .b-in_progress{background:#E6F1FB;color:#185FA5}
  .b-resolved{background:#EAF3DE;color:#3B6D11}
  .rc{font-size:12px;padding:2px 8px;border-radius:10px;background:#E6F1FB;color:#185FA5}
  .link{color:var(--primary);cursor:pointer}
  .muted{color:var(--text-3);font-size:12px}
  .detail .row{margin-bottom:12px}
  .detail .k{font-size:12px;color:var(--text-2);margin-bottom:3px}
  .detail .v{font-size:13px;line-height:1.6;white-space:pre-wrap;word-break:break-word}
  textarea{width:100%;border:1px solid var(--border);border-radius:var(--radius-sm);padding:8px;font-size:13px;font-family:inherit;resize:vertical}
  .btn{border:none;border-radius:var(--radius-sm);padding:8px 14px;font-size:13px;cursor:pointer;font-weight:500}
  .btn-primary{background:var(--success);color:#fff}
  .btn-blue{background:var(--primary);color:#fff}
  .btn-ghost{background:var(--surface);border:1px solid var(--border);color:var(--text)}
  .btn-danger{background:var(--surface);border:1px solid var(--danger);color:var(--danger)}
  .actions{display:flex;gap:8px;margin-top:8px;flex-wrap:wrap}
  .empty{color:var(--text-3);text-align:center;padding:40px;font-size:13px}
</style>
</head>
<body>
  <div class="topbar">
    <h1>🐞 Bad Case 复盘</h1>
    <div class="links">
      <a href="/admin">← 返回管理后台</a>
      <button class="btn btn-ghost" onclick="loadCases()">↻ 刷新</button>
    </div>
  </div>
  <div class="wrap">
    <div class="stats" id="stats"></div>
    <div class="filterbar">
      <div class="seg" id="segStatus">
        <button data-s="" class="active" onclick="setStatus('')">全部</button>
        <button data-s="open" onclick="setStatus('open')">待处理</button>
        <button data-s="in_progress" onclick="setStatus('in_progress')">处理中</button>
        <button data-s="resolved" onclick="setStatus('resolved')">已解决</button>
      </div>
      <select id="fRoot" onchange="applyFilter()">
        <option value="">根因: 全部</option>
        <option value="R1">R1 检索缺失</option>
        <option value="R2">R2 检索噪声</option>
        <option value="R3">R3 改写失败</option>
        <option value="R4">R4 生成偏离</option>
        <option value="R5">R5 答案不符</option>
        <option value="R6">R6 引用错误</option>
        <option value="R7">R7 超时/异常</option>
        <option value="OK">OK 隔离负例</option>
      </select>
      <input id="fQ" placeholder="搜索问题 / case id" oninput="applyFilter()" style="min-width:200px">
      <span class="spacer"></span>
      <span class="muted" id="countTip"></span>
    </div>
    <div class="layout">
      <div class="listcard">
        <table>
          <thead><tr><th>状态</th><th>问题</th><th>来源</th><th>根因</th><th>时间</th><th></th></tr></thead>
          <tbody id="tbody"></tbody>
        </table>
        <div class="empty" id="emptyTip" style="display:none">暂无 bad case</div>
      </div>
      <div class="detailcard detail" id="detail">
        <div class="muted">点击左侧任意一条查看详情</div>
      </div>
    </div>
  </div>
<script>
let token = localStorage.getItem('rag_token') || '';
let allCases = [];
let curStatus = '';
let curId = null;

function esc(s){ return String(s==null?'':s).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function fmtTime(v){
  if(v==null) return '—';
  let t = (typeof v === 'number') ? new Date(v*1000) : new Date(v);
  if(isNaN(t.getTime())) return String(v);
  const p=n=>String(n).padStart(2,'0');
  return t.getFullYear()+'-'+p(t.getMonth()+1)+'-'+p(t.getDate())+' '+p(t.getHours())+':'+p(t.getMinutes());
}
function statusBadge(s){ const m={open:['b-open','待处理'],in_progress:['b-in_progress','处理中'],resolved:['b-resolved','已解决']}; const x=m[s]||['b-open',s]; return '<span class="badge '+x[0]+'">'+x[1]+'</span>'; }

async function loadCases(){
  try{
    const r = await fetch('/api/admin/bad_cases', {headers:{'Authorization':'Bearer '+token}});
    if(r.status===403){ document.getElementById('tbody').innerHTML=''; const e=document.getElementById('emptyTip'); e.style.display='block'; e.textContent='需要管理员权限'; return; }
    const d = await r.json();
    allCases = d.items || [];
    applyFilter();
  }catch(e){ console.error(e); }
}

function applyFilter(){
  const q = (document.getElementById('fQ').value||'').trim().toLowerCase();
  const rc = document.getElementById('fRoot').value;
  const rows = allCases.filter(c=>{
    if(curStatus && c.status!==curStatus) return false;
    if(rc && (c.root_cause||'')!==rc) return false;
    if(q && !((c.query||'').toLowerCase().includes(q)) && !String(c.id).includes(q)) return false;
    return true;
  });
  renderStats();
  renderTable(rows);
  document.getElementById('countTip').textContent = '共 '+rows.length+' 条';
}

function renderStats(){
  const open = allCases.filter(c=>c.status==='open').length;
  const prog = allCases.filter(c=>c.status==='in_progress').length;
  const resolved = allCases.filter(c=>c.status==='resolved').length;
  const rate = allCases.length ? Math.round(resolved/allCases.length*100) : 0;
  document.getElementById('stats').innerHTML =
    '<div class="stat"><div class="l">待处理 open</div><div class="n">'+open+'</div></div>'+
    '<div class="stat"><div class="l">处理中 in_progress</div><div class="n">'+prog+'</div></div>'+
    '<div class="stat"><div class="l">已解决率</div><div class="n">'+rate+'%</div><div class="s">近全部 '+allCases.length+' 条</div></div>';
}

function renderTable(rows){
  const tb = document.getElementById('tbody');
  const empty = document.getElementById('emptyTip');
  if(!rows.length){ tb.innerHTML=''; empty.style.display='block'; empty.textContent='暂无匹配的 bad case'; return; }
  empty.style.display='none';
  tb.innerHTML = rows.map(c=>
    '<tr onclick="openDetail('+c.id+')" style="cursor:pointer">'+
      '<td>'+statusBadge(c.status)+'</td>'+
      '<td style="max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'+esc(c.query)+'</td>'+
      '<td class="muted">'+esc(c.source||'')+'</td>'+
      '<td>'+(c.root_cause?'<span class="rc">'+esc(c.root_cause)+'</span>':'<span class="muted">—</span>')+'</td>'+
      '<td class="muted">'+fmtTime(c.created_at)+'</td>'+
      '<td class="link">查看</td>'+
    '</tr>').join('');
}

function setStatus(s){
  curStatus = s;
  document.querySelectorAll('#segStatus button').forEach(b=>b.classList.toggle('active', b.dataset.s===s));
  applyFilter();
}

function openDetail(id){
  curId = id;
  const c = allCases.find(x=>x.id===id);
  if(!c) return;
  const detail = document.getElementById('detail');
  detail.innerHTML =
    '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">'+
      '<strong style="font-size:15px">Case #'+c.id+'</strong>'+statusBadge(c.status)+
    '</div>'+
    '<div class="row"><div class="k">问题</div><div class="v">'+esc(c.query)+'</div></div>'+
    '<div class="row"><div class="k">用户得到的答案</div><div class="v">'+(esc(c.answer)||'<span class="muted">（空）</span>')+'</div></div>'+
    '<div class="row"><div class="k">标准答案 / 期望</div><div class="v">'+(esc(c.expected)||'<span class="muted">（待补）</span>')+'</div></div>'+
    '<div class="row"><div class="k">自动根因 (triage)</div><div class="v">'+(c.root_cause?'<span class="rc">'+esc(c.root_cause)+'</span> ':'')+(esc(c.diagnosis)||'<span class="muted">（待 triage）</span>')+'</div></div>'+
    '<div class="row"><div class="k">来源 / 套件</div><div class="v">'+esc(c.source||'')+' · '+esc(c.suite||'')+'</div></div>'+
    '<div class="row"><div class="k">创建时间</div><div class="v muted">'+fmtTime(c.created_at)+'</div></div>'+
    '<div class="row"><div class="k">处理人</div><div class="v muted">'+esc(c.resolved_by||'—')+(c.resolved_at?(' · '+fmtTime(c.resolved_at)):'')+'</div></div>'+
    '<div class="row"><div class="k">处理记录</div><textarea id="detDiag" rows="3" placeholder="补充根因分析 / 修复说明...">'+esc(c.diagnosis||'')+'</textarea></div>'+
    '<div class="row"><div class="k">标准答案（可选，用于回归验证）</div><textarea id="detExp" rows="2" placeholder="填写标准答案...">'+esc(c.expected||'')+'</textarea></div>'+
    '<div class="actions">'+
      '<button class="btn btn-primary" onclick="updateCase(\'resolved\')">✓ 标记为已解决</button>'+
      '<button class="btn btn-blue" onclick="updateCase(\'in_progress\')">处理中</button>'+
      '<button class="btn btn-danger" onclick="updateCase(\'open\')">退回重测</button>'+
    '</div>';
}

async function updateCase(status){
  if(!curId) return;
  const diag = document.getElementById('detDiag').value;
  const exp = document.getElementById('detExp').value;
  try{
    const r = await fetch('/api/admin/bad_cases/'+curId, {
      method:'PATCH', headers:{'Content-Type':'application/json','Authorization':'Bearer '+token},
      body: JSON.stringify({status:status, diagnosis:diag, expected:exp})
    });
    const d = await r.json();
    if(d.ok){ await loadCases(); openDetail(curId); }
    else alert('更新失败：'+(d.error||'未知'));
  }catch(e){ alert('更新失败：'+e.message); }
}

loadCases();
</script>
</body>
</html>
"""

@app.route("/admin/bad_cases")
def admin_bad_cases_page():
    """Bad Case 复盘页（独立管理页，角色感知）。仅管理员可访问。"""
    auth_result = _require_auth()
    if auth_result:
        return auth_result
    if g.current_user.get("role") not in ("admin", "super_admin"):
        return jsonify({"ok": False, "error": "需要管理员权限"}), 403
    return _BADCASE_PAGE


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """
    用户反馈入口 —— bad case 闭环的「源头」。

    请求体：{task_id?, query, answer?, rating(-1=踩/0=无/1=赞), feedback_text?}
    行为：
      1. 落库 qa_feedback（关联 task_id，可回溯完整 task_checkpoints trace）
      2. 若 rating == -1（点踩），同步写入一条 open 状态的 bad_cases，
         待 evalkit.triage 结合检索结果补 root_cause，或管理员在后台处理
    返回：{ok, feedback_id, bad_case_id?}
    """
    auth_result = _require_auth()
    if auth_result:
        return auth_result
    user = g.current_user
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    try:
        rating = int(data.get("rating", 0) or 0)
    except (TypeError, ValueError):
        rating = 0
    if not query:
        return jsonify({"ok": False, "error": "query 必填"}), 400

    ms = _get_memory_store()
    if ms is None:
        return jsonify({"ok": False, "error": "记忆层不可用"}), 503

    fb_id = ms.save_feedback(
        query=query,
        answer=data.get("answer"),
        rating=rating,
        feedback_text=data.get("feedback_text"),
        task_id=data.get("task_id"),
        user_id=user["user_id"],
        tenant_id=user.get("tenant_id", "default"),
        session_id=data.get("session_id"),
    )

    # P1-7 L3：用户反馈级信号回灌——若本次问答命中了 playbook，赞/踩都同步到经验。
    # 关联键 = task_id（= last_task_id，前端已在反馈里带上）：
    #   1. 按 task_id 反查本次问答命中的 used_playbook_pk
    #   2. 赞(1) → reinforce_feedback(pk, positive=True) 计数 +2 强确认
    #      踩(-1) → reinforce_feedback(pk, positive=False) 计数清零标存疑
    # 全程 try/except 降级，绝不因经验回灌失败影响反馈主流程。
    task_id = data.get("task_id")
    if rating in (1, -1) and task_id:
        try:
            # P1-R1：回灌反查必须校验 task 归属（user_id），防越权清/抬他人经验计数
            pk = ms.get_task_playbook_pk(task_id, user_id=user["user_id"])
            pb_store = _get_playbook_store()
            if pk and pb_store is not None:
                pb_store.reinforce_feedback(pk, positive=(rating == 1))
                print(f"  [L3反馈级] task={task_id} pk={pk} "
                      f"reinforce_feedback(positive={rating == 1})")
        except Exception as e:
            print(f"  [L3反馈级] 回灌异常(忽略): {e}")

    # 点踩 = 真实失败样本，沉淀到 bad_cases 驱动自进化
    bad_case_id = None
    if rating == -1:
        bad_case_id = ms.add_bad_case(
            query=query, source="feedback", suite="answer",
            case_id=None, answer=data.get("answer"), expected=None,
            root_cause=None,
            diagnosis=f"用户点踩（tenant={user.get('tenant_id','default')}），待 triage。",
            status="open",
        )

    # 静默失败防护：记忆层写入失败时明确报错，避免前端误判成功
    if fb_id == -1:
        return jsonify({"ok": False, "error": "反馈写入失败（记忆层不可用）"}), 503
    if rating == -1 and bad_case_id == -1:
        return jsonify({"ok": False, "error": "点踩已记录，但 Bad Case 写入失败（记忆层不可用）"}), 503
    return jsonify({"ok": True, "feedback_id": fb_id, "bad_case_id": bad_case_id})


@app.route("/api/admin/bad_cases", methods=["GET"])
def api_admin_bad_cases():
    """
    管理后台：列出 bad case 库，供「Bad Case 复盘」Tab 使用。
    支持 ?status=open / ?root_cause=R5 筛选。仅管理员可访问。
    """
    auth_result = _require_auth()
    if auth_result:
        return auth_result
    if g.current_user.get("role") not in ("admin", "super_admin"):
        return jsonify({"ok": False, "error": "需要管理员权限"}), 403
    ms = _get_memory_store()
    if ms is None:
        return jsonify({"ok": False, "error": "记忆层不可用"}), 503
    status = request.args.get("status")
    root_cause = request.args.get("root_cause")
    rows = ms.list_bad_cases(status=status, root_cause=root_cause, limit=500)
    return jsonify({"ok": True, "count": len(rows), "items": rows})


@app.route("/api/admin/bad_cases/<int:bc_id>", methods=["PATCH"])
def api_admin_bad_case_update(bc_id):
    """管理后台：更新某条 bad case 的状态 / 处理记录（bad case 闭环「修复→验证」落点）。

    请求体：{status?(open|in_progress|resolved), diagnosis?, expected?}
    仅管理员可访问；resolved_by 自动取当前登录用户。
    """
    auth_result = _require_auth()
    if auth_result:
        return auth_result
    if g.current_user.get("role") not in ("admin", "super_admin"):
        return jsonify({"ok": False, "error": "需要管理员权限"}), 403
    ms = _get_memory_store()
    if ms is None:
        return jsonify({"ok": False, "error": "记忆层不可用"}), 503
    data = request.get_json(silent=True) or {}
    resolved_by = g.current_user.get("username") or g.current_user.get("user_id")
    ok = ms.update_bad_case_status(
        bc_id, status=data.get("status"), resolved_by=resolved_by,
        diagnosis=data.get("diagnosis"), expected=data.get("expected"))
    return jsonify({"ok": bool(ok)})


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

# ======================================================================
# 知识库文档管理 API（P0-1：多租户隔离 + 角色感知上传/删除/重建）
# ======================================================================

SUPPORTED_UPLOAD_EXT = (".txt", ".md", ".pdf", ".html", ".htm",
                        ".docx", ".xlsx", ".xls", ".pptx")


def _kb_build_pipeline(tenant_id: str, user_id, access_level: str = "public"):
    """构造一次摄取管线（单文件上传 / 全量重建共用）。"""
    store = MilvusStoreBackend(vector_db.client, vector_db.collection)
    return IngestPipeline(
        folder=DOC_FOLDER,
        embedder=vector_db._embed.embed_documents,
        store=store,
        tenant_id=tenant_id,
        user_id=str(user_id),
        access_fn=(lambda s: access_level),
    )


def _kb_derive_tenant(rel_path: str) -> str:
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) > 1 and parts[0]:
        return parts[0]
    return "default"


def _kb_safe_name(filename: str) -> str:
    """保留原文件名的安全文件名处理。

    与 werkzeug.secure_filename 不同：不删除中文、空格、括号，
    仅替换 Windows/Linux 路径非法字符，确保不同租户同名文件可共存。
    """
    forbidden = '\\/:*?"<>|'
    for ch in forbidden:
        filename = filename.replace(ch, '_')
    filename = filename.strip('. ')
    if not filename:
        filename = "unnamed"
    return filename


def _kb_scan_files():
    """递归扫描 knowledge/ 下受支持文件，附带 manifest 统计。"""
    out = []
    manifest = ManifestStore(os.path.join(DOC_FOLDER, ".ingest_manifest.sqlite"))
    recs = manifest.load_all()
    # 额外建立规范化查找表，处理 Windows 盘符大小写、正/反斜杠差异
    norm_recs = {}
    for k, v in recs.items():
        try:
            norm_recs[os.path.normcase(os.path.normpath(os.path.abspath(k)))] = v
        except Exception:
            norm_recs[os.path.normcase(k)] = v
    for root, _dirs, files in os.walk(DOC_FOLDER):
        if os.path.basename(root).startswith("."):
            continue
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() not in SUPPORTED_UPLOAD_EXT:
                continue
            full = os.path.join(root, name)
            rel = os.path.relpath(full, DOC_FOLDER).replace("\\", "/")
            # manifest key 可能是绝对路径、相对路径（正/反斜杠）或旧版的 ./knowledge\... 形式
            keys = [full, rel, rel.lstrip("./"), "./" + rel,
                    full.replace("/", "\\"), rel.replace("/", "\\"),
                    ("./" + rel).replace("/", "\\")]
            rec = {}
            for k in keys:
                if k in recs:
                    rec = recs[k]
                    break
            if not rec:
                # 兜底：按规范化绝对路径再匹配一次
                rec = norm_recs.get(os.path.normcase(os.path.normpath(os.path.abspath(full))), {})
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            out.append({
                "path": rel,
                "tenant": _kb_derive_tenant(rel),
                "size": size,
                "chunks": rec.get("chunk_count", 0),
            })
    manifest.close()
    return out


def _kb_file_owner(file_path: str):
    """查询某文件首个 chunk 的拥有者（Milvus user_id），用于普通用户删除鉴权。"""
    if vector_db.backend != "milvus":
        return None
    safe = file_path.replace("\\", "\\\\").replace('"', '\\"')
    try:
        res = vector_db.client.query(
            vector_db.collection, f'file_path == "{safe}"',
            output_fields=["user_id"], limit=1)
        if res:
            return res[0].get("user_id")
    except Exception:
        pass
    return None


def _kb_visible_files(tenant, role, uid):
    """通过 Milvus 标量过滤下推「角色 + 用户id」权限，返回可见文件集合与 ACL 元信息。

    返回 (paths:set, meta:dict[path]={owner,access_level})；Milvus 不可用时抛异常，
    由调用方回退到仅按租户过滤。
    """
    if vector_db.backend != "milvus":
        raise RuntimeError("知识库后端非 Milvus，无法下推用户级权限")
    if role == ROLE_ADMIN:
        expr = f"(tenant_id == '{tenant}')"
    else:
        expr = (f"(tenant_id == '{tenant}') and "
                f"((access_level == 'public') or (user_id == '{str(uid)}'))")
    paths, meta = set(), {}
    offset, page = 0, 1000
    print(f"[docs/acl] query expr={expr!r}")
    while True:
        res = vector_db.client.query(
            vector_db.collection, expr,
            output_fields=["file_path", "user_id", "access_level"],
            limit=page, offset=offset)
        print(f"[docs/acl] query returned {len(res)} rows (offset={offset})")
        if not res:
            break
        for r in res:
            fp = (r.get("file_path") or "").replace("\\", "/")
            print(f"[docs/acl] row file_path={fp!r} user_id={r.get('user_id')!r} access_level={r.get('access_level')!r}")
            if fp:
                paths.add(fp)
                if fp not in meta:
                    meta[fp] = {"owner": r.get("user_id"),
                                "access_level": r.get("access_level")}
        if len(res) < page:
            break
        offset += page
    return paths, meta


@app.route("/api/docs", methods=["GET"])
def api_docs_list():
    """列出知识库文档（按 角色 + 用户id 下推隔离）：
    - super_admin：全部租户文件；
    - admin：本租户全部；
    - 普通用户：本租户内 (access_level==public) 或 (user_id==本人) 的文件。
    返回字段含 tenant / owner / access_level，前端可据此渲染归属与可见性。
    """
    auth_result = _require_auth()
    if auth_result:
        return auth_result
    role = g.current_user["role"]
    uid = g.current_user["user_id"]
    my_tenant = g.current_user.get("tenant_id", "default")
    files = _kb_scan_files()
    print(f"[docs/list] user={g.current_user.get('username')} role={role} uid={uid} tenant={my_tenant} scanned={len(files)} (全盘扫描，后续按租户/权限过滤)")
    for _f in files:
        # 非 super_admin 看到的其他租户文件一定会被过滤掉，打标避免误判为越权可见
        cross = (_f.get("tenant") != my_tenant and role != ROLE_SUPER_ADMIN)
        tag = " [跨租户-将过滤]" if cross else ""
        print(f"[docs/list] scan path={_f['path']!r} tenant={_f['tenant']}{tag}")

    def _norm(p: str) -> str:
        """把绝对/相对路径统一为相对 DOC_FOLDER 的 / 分隔路径。"""
        try:
            return os.path.relpath(os.path.abspath(p), DOC_FOLDER).replace("\\", "/")
        except Exception:
            return (p or "").replace("\\", "/")

    if role == ROLE_SUPER_ADMIN:
        # 全租户：附 ACL 元信息便于展示
        try:
            res = vector_db.client.query(
                vector_db.collection, "",
                output_fields=["file_path", "user_id", "access_level"],
                limit=10000)
            meta = {}
            for r in res:
                fp = _norm(r.get("file_path", ""))
                if fp and fp not in meta:
                    meta[fp] = {"owner": r.get("user_id"),
                                "access_level": r.get("access_level")}
            for f in files:
                m = meta.get(f["path"].replace("\\", "/"))
                if m:
                    f["owner"] = m["owner"]
                    f["access_level"] = m["access_level"]
        except Exception as _e:
            print(f"[docs] super_admin ACL 标注失败(忽略): {_e}")
        return jsonify({"files": files, "count": len(files)})

    # admin：本租户全部可见（直接按租户过滤，避免 Milvus expr/路径不一致导致空白）
    if role == ROLE_ADMIN:
        files = [f for f in files if f.get("tenant") == my_tenant]
        print(f"[docs/list] admin role: return tenant={my_tenant} count={len(files)}")
        return jsonify({"files": files, "count": len(files)})

    # 普通用户：Milvus 下推「用户id + access_level」权限
    try:
        vis_paths, vis_meta = _kb_visible_files(my_tenant, role, uid)
    except Exception as _e:
        print(f"[docs] 权限下推失败，回退租户过滤: {_e}")
        vis_paths, vis_meta = None, None

    if vis_paths is None:
        files = [f for f in files if f.get("tenant") == my_tenant]
    else:
        vis_rel = {_norm(p) for p in vis_paths}
        vis_rel_meta = {}
        for p, m in vis_meta.items():
            vis_rel_meta[_norm(p)] = m
        print(f"[docs/list] vis_rel={vis_rel!r}")
        files = [f for f in files if f["path"].replace("\\", "/") in vis_rel]
        for f in files:
            m = vis_rel_meta.get(f["path"].replace("\\", "/"))
            if m:
                f["owner"] = m["owner"]
                f["access_level"] = m["access_level"]
    print(f"[docs/list] returning count={len(files)}")
    return jsonify({"files": files, "count": len(files)})


@app.route("/api/docs/upload", methods=["POST"])
def api_docs_upload():
    """上传并增量入库（角色感知）：
    - 普通用户：落自己租户，仅能删自己上传的；
    - 租户管理员：落本租户，可管理本租户全部；
    - super-admin：可指定 tenant 表单字段，落到任意租户。
    文档归属（tenant + owner）一律来自登录态/服务端，绝不信任客户端。
    """
    auth_result = _require_auth()
    if auth_result:
        return auth_result
    role = g.current_user["role"]
    uid = g.current_user["user_id"]
    my_tenant = g.current_user.get("tenant_id", "default")

    if role == ROLE_SUPER_ADMIN:
        tenant = (request.form.get("tenant", "") or "").strip() or my_tenant
    else:
        tenant = my_tenant  # 非超级管理员只能落到自己租户
    if not tenant:
        tenant = "default"
    if len(tenant) > 64 or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for ch in tenant):
        return jsonify({"error": "非法租户名"}), 400


    access_level = request.form.get("access_level", "public")
    if access_level not in ("public", "restricted"):
        access_level = "public"

    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "未收到文件"}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in SUPPORTED_UPLOAD_EXT:
        return jsonify({"error": f"不支持的格式：{ext}"}), 400

    dest_dir = os.path.abspath(os.path.join(DOC_FOLDER, tenant))
    doc_root = os.path.abspath(DOC_FOLDER)
    if os.path.commonpath([doc_root, dest_dir]) != doc_root:
        return jsonify({"error": "非法路径"}), 400

    os.makedirs(dest_dir, exist_ok=True)
    filename = _kb_safe_name(f.filename)
    save_path = os.path.join(dest_dir, filename)
    print(f"[docs/upload] 开始接收文件 user={g.current_user['username']} tenant={tenant} name={filename}")
    f.save(save_path)
    print(f"[docs/upload] 已保存到 {save_path} ({os.path.getsize(save_path)} 字节)，开始解析入库...")

    try:
        pipe = _kb_build_pipeline(tenant, uid, access_level)

        def _progress(text: str):
            print(f"[docs/upload] {text}")

        rep = pipe.run(files=[save_path], progress_cb=_progress)
        pipe.close()
        print(f"[docs/upload] 入库完成 file={save_path} chunks={rep.entities_upserted} duration={rep.duration_sec:.2f}s")
    except Exception as e:
        print(f"[docs/upload] 入库失败: {e}")
        return jsonify({"error": f"入库失败：{e}"}), 500
    _audit_log("docs_upload", target=save_path, username=g.current_user["username"],
               result="success")
    return jsonify({
        "success": True,
        "file": os.path.relpath(save_path, DOC_FOLDER).replace("\\", "/"),
        "tenant": tenant,
        "access_level": access_level,
        "chunks": rep.entities_upserted,
    })


@app.route("/api/docs/<path:file_id>", methods=["DELETE"])
def api_docs_delete(file_id):
    """删除文档（角色感知）：
    - super-admin：任意；admin：同租户；普通用户：仅自己上传的。
    """
    auth_result = _require_auth()
    if auth_result:
        return auth_result
    role = g.current_user["role"]
    uid = g.current_user["user_id"]
    my_tenant = g.current_user.get("tenant_id", "default")

    rel = file_id.replace("\\", "/")
    full = os.path.normpath(os.path.join(DOC_FOLDER, rel))
    base = os.path.normpath(DOC_FOLDER)
    if not (full == base or full.startswith(base + os.sep)):
        return jsonify({"error": "非法路径"}), 400

    file_tenant = _kb_derive_tenant(rel)
    if role not in (ROLE_ADMIN, ROLE_SUPER_ADMIN):
        owner = _kb_file_owner(full)
        if owner != str(uid):
            return jsonify({"error": "只能删除自己上传的文档"}), 403
    elif role == ROLE_ADMIN and file_tenant != my_tenant:
        return jsonify({"error": "只能管理本租户文档"}), 403

    deleted = 0
    if os.path.isfile(full):
        try:
            store = MilvusStoreBackend(vector_db.client, vector_db.collection)
            deleted = store.delete_by_file(full)
        except Exception:
            pass
        try:
            manifest = ManifestStore(os.path.join(DOC_FOLDER, ".ingest_manifest.sqlite"))
            manifest.remove(full)
            manifest.close()
        except Exception:
            pass
        os.remove(full)
    _audit_log("docs_delete", target=rel, username=g.current_user["username"],
               result="success")
    return jsonify({"success": True, "deleted": deleted})


@app.route("/api/docs/rebuild", methods=["POST"])
def api_docs_rebuild():
    """全量重建（仅 admin/super-admin）：后台线程跑，返回 job_id。
    注：仅重建默认命名空间（knowledge/ 平铺文件）；租户文档由上传 API 管理，避免覆盖 owner。
    """
    auth_result = _require_admin()
    if auth_result:
        return auth_result
    job_id = uuid.uuid4().hex[:12]

    def _do():
        try:
            pipe = _kb_build_pipeline("default", "anonymous", "public")
            pipe.run(force=True)
            pipe.close()
            print(f"[docs] 全量重建完成 job={job_id}")
        except Exception as e:
            print(f"[docs] 全量重建失败 job={job_id}: {e}")

    threading.Thread(target=_do, daemon=True).start()
    return jsonify({"success": True, "job_id": job_id})


@app.route("/api/docs/stats", methods=["GET"])
def api_docs_stats():
    """统计：总文档数、总 chunk 数、按租户拆分。"""
    auth_result = _require_auth()
    if auth_result:
        return auth_result
    files = _kb_scan_files()
    by_tenant = {}
    for fmeta in files:
        by_tenant[fmeta["tenant"]] = by_tenant.get(fmeta["tenant"], 0) + 1
    total_chunks = 0
    try:
        total_chunks = vector_db._milvus_count()
    except Exception:
        pass
    return jsonify({
        "total_docs": len(files),
        "total_chunks": total_chunks,
        "by_tenant": by_tenant,
    })


def _load_tenants():
    """读取 config/tenants.yaml 的租户清单，供 /kb 超级管理员下拉选择。"""
    try:
        import yaml
        _p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "config", "tenants.yaml")
        if os.path.isfile(_p):
            with open(_p, "r", encoding="utf-8") as _f:
                _data = yaml.safe_load(_f) or {}
            return [t.get("name") for t in (_data.get("tenants") or [])
                    if t.get("name")]
    except Exception:
        pass
    return ["default"]


@app.route("/kb")
def kb_page():
    """知识库管理页（server-rendered，角色感知）。任意已登录用户可访问。"""
    auth_result = _require_auth()
    if auth_result:
        return auth_result
    role = g.current_user["role"]
    is_admin = role in (ROLE_ADMIN, ROLE_SUPER_ADMIN)
    can_choose_tenant = (role == ROLE_SUPER_ADMIN)
    tenant_options = "".join(
        f'<option value="{t}">{t}</option>' for t in _load_tenants())
    html = '''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>知识库管理</title>
<style>
  body{font-family:-apple-system,Segoe UI,Roboto,"Microsoft YaHei",sans-serif;margin:0;background:#f5f7fa;color:#222}
  .wrap{max-width:960px;margin:0 auto;padding:24px}
  header{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px}
  h1{font-size:20px;margin:0}
  .badge{font-size:12px;padding:2px 10px;border-radius:12px;background:#e6f1fb;color:#185fa5;border:1px solid #bcdcf5}
  .card{background:#fff;border:1px solid #e6e8eb;border-radius:10px;padding:18px;margin-bottom:18px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
  .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
  input[type=file]{font-size:14px}
  select,input[type=text]{padding:7px 10px;border:1px solid #ccd2d9;border-radius:7px;font-size:14px}
  button{background:#185fa5;color:#fff;border:none;padding:8px 16px;border-radius:7px;font-size:14px;cursor:pointer}
  button.ghost{background:#fff;color:#185fa5;border:1px solid #185fa5}
  button.danger{background:#fff;color:#a32d2d;border:1px solid #a32d2d}
  table{width:100%;border-collapse:collapse;font-size:14px}
  th,td{text-align:left;padding:9px 8px;border-bottom:1px solid #eef0f2}
  th{color:#666;font-weight:600}
  .muted{color:#888;font-size:13px}
  #msg{margin-top:10px;font-size:13px;color:#0f6e56}
  #err{margin-top:10px;font-size:13px;color:#a32d2d}
  .topnav{display:flex;align-items:center;justify-content:space-between;background:#185fa5;color:#fff;padding:0 18px;height:54px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
  .nav-left{display:flex;align-items:center;gap:10px}
  .nav-left .logo{font-size:22px}
  .nav-left h1{font-size:17px;margin:0;font-weight:600}
  .nav-right{display:flex;align-items:center;gap:6px}
  .nav-link{color:#fff;text-decoration:none;font-size:14px;padding:7px 12px;border-radius:7px;opacity:.85}
  .nav-link:hover{background:rgba(255,255,255,.15);opacity:1}
  .nav-link.active{background:rgba(255,255,255,.22);opacity:1;font-weight:600}
  .nav-user{font-size:13px;padding:5px 10px;background:rgba(255,255,255,.15);border-radius:12px;margin-left:6px}
  .nav-logout{background:rgba(255,255,255,.15);color:#fff;border:none;padding:7px 12px;border-radius:7px;font-size:13px;cursor:pointer;margin-left:6px}
  .nav-logout:hover{background:rgba(255,255,255,.28)}
</style></head>
<body>
<header class="topnav">
  <div class="nav-left">
    <div class="logo">📚</div>
    <h1>企业知识库</h1>
  </div>
  <div class="nav-right">
    <a href="/" class="nav-link">💬 问答</a>
    <a href="/kb" class="nav-link active">📚 知识库</a>''' + ('''
    <a href="/admin" class="nav-link">⚙️ 系统管理</a>''' if is_admin else '') + '''
    <span class="nav-user" id="who"></span>
    <button class="nav-logout" onclick="doLogout()">🚪 退出</button>
  </div>
</header>
<div class="wrap">

  <div class="card">
    <div class="row">
      <input type="file" id="file">
      <select id="access">
        <option value="public">公开(本租户可读)</option>
        <option value="restricted">受限(仅自己+管理员)</option>
      </select>
      <select id="tenant" style="display:''' + ("inline-block" if can_choose_tenant else "none") + '''">''' + tenant_options + '''</select>
      <button onclick="upload()">⬆️ 上传并入库</button>
    </div>
    <div id="msg"></div><div id="err"></div>
  </div>

  <div class="card">
    <div class="row" style="justify-content:space-between">
      <strong>文档列表</strong>
      ''' + ('''<button class="ghost" onclick="rebuild()">♻️ 全量重建</button>''' if is_admin else '') + '''
    </div>
    <div id="stats" class="muted" style="margin:8px 0"></div>
    <table>
      <thead><tr><th>路径</th><th>租户</th><th>分片</th><th>大小</th><th></th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
</div>
<script>
const me = ''' + json.dumps({
        "username": g.current_user["username"],
        "role": role,
        "tenant_id": g.current_user.get("tenant_id", "default"),
        "tenants": _load_tenants(),
    }) + ''';
document.getElementById("who").textContent = me.username + " · " + me.role + " · 租户:" + me.tenant_id;
load();

function load(){
  fetch("/api/docs").then(r=>r.json()).then(d=>{
    const tb = document.getElementById("rows"); tb.innerHTML="";
    (d.files||[]).forEach(f=>{
      const tr=document.createElement("tr");
      tr.innerHTML = "<td>"+f.path+"</td><td>"+f.tenant+"</td><td>"+f.chunks+"</td><td>"+(f.size/1024).toFixed(1)+"KB</td>";
      const td=document.createElement("td");
      const b=document.createElement("button"); b.className="danger"; b.textContent="删除";
      b.onclick=()=>del(f.path); td.appendChild(b); tr.appendChild(td); tb.appendChild(tr);
    });
  });
  fetch("/api/docs/stats").then(r=>r.json()).then(s=>{
    document.getElementById("stats").textContent =
      "共 "+s.total_docs+" 篇文档 · "+s.total_chunks+" 分片 · 租户分布: "+
      Object.entries(s.by_tenant||{}).map(([k,v])=>k+":"+v).join("  ");
  });
}

function upload(){
  const fd=new FormData();
  const file=document.getElementById("file").files[0];
  if(!file){err("请选择文件");return;}
  const btn=document.querySelector('button[onclick="upload()"]');
  if(btn){btn.disabled=true;btn.textContent='⏳ 上传入库中...';}
  msg('正在上传 '+file.name+'，请稍候（大文件 PDF 解析+向量化可能需要几十秒）...');
  err('');
  fd.append("file",file);
  fd.append("access_level",document.getElementById("access").value);
  const t=document.getElementById("tenant").value;
  if(t) fd.append("tenant",t);
  fetch("/api/docs/upload",{method:"POST",body:fd}).then(r=>r.json()).then(d=>{
    if(d.success){msg('✅ 已入库: '+d.file+' ('+d.chunks+' 分片)');load();}
    else err(d.error||'上传失败');
  }).catch(e=>err(e)).finally(()=>{if(btn){btn.disabled=false;btn.textContent='⬆️ 上传并入库';}});
}

function del(path){
  if(!confirm("确认删除 "+path+"？")) return;
  fetch("/api/docs/"+encodeURIComponent(path),{method:"DELETE"})
    .then(r=>r.json()).then(d=>{ if(d.success){msg("已删除");load();} else err(d.error||"删除失败"); })
    .catch(e=>err(e));
}

function rebuild(){
  if(!confirm("全量重建将重刷默认命名空间，耗时较长，继续？")) return;
  fetch("/api/docs/rebuild",{method:"POST"}).then(r=>r.json()).then(d=>{
    if(d.success) msg("重建已启动 job="+d.job_id);
  });
}

function msg(t){document.getElementById("msg").textContent=t;document.getElementById("err").textContent="";}
function err(t){document.getElementById("err").textContent=t;document.getElementById("msg").textContent="";}
function doLogout(){
  fetch('/api/logout',{method:'POST'}).catch(function(){});
  try{ localStorage.removeItem('rag_token'); }catch(e){}
  location.replace('/login');
}
</script>
</body></html>'''
    return html


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

    # ===== 角色感知过滤：admin 仅看本租户用户，super_admin 看全部 =====
    role = g.current_user.get("role")
    my_tenant = g.current_user.get("tenant_id", "default")
    if role != ROLE_SUPER_ADMIN:
        try:
            from prompt_manager import get_auth_manager
            pool = get_auth_manager()._pool
            if pool is not None:
                conn = pool.connection()
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT username, tenant_id FROM admin_users")
                    tmap = {r[0]: (r[1] or "default") for r in cur.fetchall()}
                finally:
                    conn.close()
                users = [u for u in users if tmap.get(u["user"], "default") == my_tenant]
                rows = [r for r in rows if tmap.get(r.get("user"), "default") == my_tenant]
                window = {
                    "calls": sum(u["calls"] for u in users),
                    "prompt_tokens": sum(u["prompt_tokens"] for u in users),
                    "completion_tokens": sum(u["completion_tokens"] for u in users),
                    "total_tokens": sum(u["total_tokens"] for u in users),
                    "cost_usd": round(sum(u["cost_usd"] for u in users), 6),
                    "avg_latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 3)
                                     if rows else 0.0,
                }
        except Exception as _e:
            print(f"[usage] 租户过滤失败(忽略): {_e}")

    # ===== 可选：按用户筛选（user_id 或 username），非 super_admin 仍受本租户约束 =====
    req_user = (request.args.get("user_id") or request.args.get("user") or "").strip()
    if req_user:
        target_username = req_user
        try:
            from prompt_manager import get_auth_manager
            pool = get_auth_manager()._pool
            if pool is not None:
                conn = pool.connection()
                try:
                    cur = conn.cursor()
                    if req_user.isdigit():
                        cur.execute("SELECT username, tenant_id FROM admin_users WHERE id=%s", (int(req_user),))
                    else:
                        cur.execute("SELECT username, tenant_id FROM admin_users WHERE username=%s", (req_user,))
                    row = cur.fetchone()
                    if row:
                        target_username = row[0]
                        if role != ROLE_SUPER_ADMIN and (row[1] or "default") != my_tenant:
                            target_username = None  # 跨租户越权，忽略筛选
                    else:
                        target_username = None
                finally:
                    conn.close()
        except Exception as _e:
            print(f"[usage] 用户解析失败(忽略筛选): {_e}")
            target_username = None
        if target_username:
            users = [u for u in users if u["user"] == target_username]
            rows = [r for r in rows if r.get("user") == target_username]
            window = {
                "calls": sum(u["calls"] for u in users),
                "prompt_tokens": sum(u["prompt_tokens"] for u in users),
                "completion_tokens": sum(u["completion_tokens"] for u in users),
                "total_tokens": sum(u["total_tokens"] for u in users),
                "cost_usd": round(sum(u["cost_usd"] for u in users), 6),
                "avg_latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 3)
                                 if rows else 0.0,
            }

    m = gw.metrics()
    return jsonify({
        "range": rng,
        "users": users,
        "window": window,
        "rows": rows,
        "persisted": m.get("usage_persisted", False),
        "db": m.get("usage_db", ""),
    })


# ---- 用户管理 API（仅超级管理员，体现 tab/数据权限分层）----

@app.route("/api/admin/users", methods=["GET"])
def api_admin_users_list():
    """列出用户：超级管理员看全部；租户管理员仅看本租户。"""
    denied = _require_admin()
    if denied:
        return denied
    role = g.current_user.get("role")
    my_tenant = g.current_user.get("tenant_id", "default")
    if role not in (ROLE_ADMIN, ROLE_SUPER_ADMIN):
        return jsonify({"error": "无权限"}), 403
    try:
        from prompt_manager import get_auth_manager
        pool = get_auth_manager()._pool
        if pool is None:
            return jsonify({"error": "数据库不可用"}), 503
        conn = pool.connection()
        try:
            cur = conn.cursor()
            if role == ROLE_SUPER_ADMIN:
                cur.execute(
                    "SELECT id, username, display_name, role, tenant_id, is_active "
                    "FROM admin_users ORDER BY id")
            else:
                cur.execute(
                    "SELECT id, username, display_name, role, tenant_id, is_active "
                    "FROM admin_users WHERE tenant_id=%s ORDER BY id", (my_tenant,))
            cols = ["id", "username", "display_name", "role", "tenant_id", "is_active"]
            users = [dict(zip(cols, row)) for row in cur.fetchall()]
        finally:
            conn.close()
        return jsonify({"users": users,
                        "scope": "all" if role == ROLE_SUPER_ADMIN else "tenant",
                        "can_create_super": role == ROLE_SUPER_ADMIN})
    except Exception as e:
        return jsonify({"error": f"查询失败：{e}"}), 500


@app.route("/api/admin/users", methods=["POST"])
def api_admin_users_create():
    """新增用户。

    - 超级管理员：可建任意租户、任意角色（user/admin/super_admin）。
    - 租户管理员：仅能建本租户用户，角色限于 普通用户 / 租户管理员。
    服务端强制约束，绝不信任客户端传入的 tenant_id / role。
    """
    denied = _require_admin()
    if denied:
        return denied
    role = g.current_user.get("role")
    my_tenant = g.current_user.get("tenant_id", "default")
    if role not in (ROLE_ADMIN, ROLE_SUPER_ADMIN):
        return jsonify({"error": "无权限"}), 403

    data = request.get_json(force=True, silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    display_name = (data.get("display_name") or "").strip()
    new_role = (data.get("role") or "user").strip()
    new_tenant = (data.get("tenant_id") or "").strip()

    err = validate_input(username, MAX_USERNAME_LEN, "用户名")
    if err:
        return jsonify({"error": err}), 400
    if len(password) < 6:
        return jsonify({"error": "密码至少 6 位"}), 400

    if role == ROLE_SUPER_ADMIN:
        tenant = new_tenant or my_tenant
        if new_role not in (ROLE_ADMIN, ROLE_SUPER_ADMIN, ROLE_USER):
            return jsonify({"error": "非法角色"}), 400
    else:
        # 租户管理员：锁定本租户，角色仅限 普通用户 / 租户管理员
        tenant = my_tenant
        if new_role not in (ROLE_ADMIN, ROLE_USER):
            return jsonify({"error": "租户管理员只能创建普通用户或租户管理员"}), 400

    from prompt_manager import get_auth_manager
    ok, msg = get_auth_manager().create_user(username, password, display_name, new_role, tenant)
    if not ok:
        return jsonify({"error": msg}), 400
    return jsonify({"success": True, "username": username, "tenant": tenant, "role": new_role})


@app.route("/api/admin/users/<int:uid>", methods=["PATCH"])
def api_admin_users_update(uid):
    """修改用户角色 / 启用状态。

    - 超级管理员：可改任意用户（含设为 super_admin）。
    - 租户管理员：只能改本租户用户，且不能升为 super_admin。
    """
    denied = _require_admin()
    if denied:
        return denied
    role = g.current_user.get("role")
    my_tenant = g.current_user.get("tenant_id", "default")
    if role not in (ROLE_ADMIN, ROLE_SUPER_ADMIN):
        return jsonify({"error": "无权限"}), 403
    data = request.get_json(force=True, silent=True) or {}
    new_role = data.get("role")
    is_active = data.get("is_active")
    sets, params = [], []
    if new_role is not None:
        if new_role not in (ROLE_ADMIN, ROLE_SUPER_ADMIN, ROLE_USER):
            return jsonify({"error": "非法角色"}), 400
        # 租户管理员不能赋予 super_admin 角色（防越权提权）
        if role != ROLE_SUPER_ADMIN and new_role == ROLE_SUPER_ADMIN:
            return jsonify({"error": "租户管理员不能赋予超级管理员角色"}), 403
        sets.append("role=%s"); params.append(new_role)
    if is_active is not None:
        sets.append("is_active=%s"); params.append(1 if is_active else 0)
    if not sets:
        return jsonify({"error": "无可更新字段"}), 400
    try:
        from prompt_manager import get_auth_manager
        pool = get_auth_manager()._pool
        if pool is None:
            return jsonify({"error": "数据库不可用"}), 503
        conn = pool.connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT tenant_id FROM admin_users WHERE id=%s", (uid,))
            trow = cur.fetchone()
            if not trow:
                return jsonify({"error": "用户不存在"}), 404
            if role != ROLE_SUPER_ADMIN and (trow[0] or "default") != my_tenant:
                return jsonify({"error": "无权修改其他租户用户"}), 403
            cur.execute(
                "UPDATE admin_users SET " + ", ".join(sets) + " WHERE id=%s",
                params + [uid])
        finally:
            conn.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": f"更新失败：{e}"}), 500


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
  .fb-bar{display:flex;gap:6px;align-items:center;margin-top:10px}
  .fb-btn{font-size:12px;color:#6b7280;border:1px solid #e2e5ed;background:#fff;border-radius:7px;padding:4px 11px;cursor:pointer;transition:all .15s;display:inline-flex;align-items:center;gap:4px}
  .fb-btn:hover{background:#f5f6fa;color:#1a1a2e}
  .fb-up.on{color:#3B6D11;border-color:#3B6D11;background:#EAF3DE}
  .fb-down.on{color:#A32D2D;border-color:#A32D2D;background:#FCEBEB}
  .fb-tip{font-size:11px;color:#9ca3af;margin-left:4px}
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
  /* ===== KB Tab ===== */
  .kb-toolbar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:16px}
  .kb-toolbar select,.kb-toolbar input[type=file]{padding:6px 10px;border:1px solid var(--border);border-radius:var(--radius-sm);font-size:14px;background:var(--surface);color:var(--text)}
  .kb-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px;flex:1;overflow:hidden;display:flex;flex-direction:column}
  .kb-card-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
  .kb-card-header h2{font-size:17px;font-weight:600}
  .kb-stats{color:var(--text-2);font-size:13px;margin-bottom:12px}
  .kb-table{width:100%;border-collapse:collapse;font-size:14px}
  .kb-table th,.kb-table td{text-align:left;padding:10px 8px;border-bottom:1px solid var(--border)}
  .kb-table th{color:var(--text-2);font-weight:600;background:var(--bg)}
  .kb-table tr:hover{background:var(--bg)}
  .kb-msg{color:var(--success);font-size:13px;margin-top:8px}
  .kb-err{color:var(--danger);font-size:13px;margin-top:8px}
  .kb-btn-danger{border:1px solid var(--danger);color:var(--danger);background:#fff;padding:5px 12px;border-radius:var(--radius-sm);cursor:pointer;font-size:13px}
  .kb-btn-danger:hover{background:var(--danger-light)}
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
      <a href="/admin/bad_cases" style="color:var(--danger);text-decoration:none;font-size:13px">🐞 Bad Case</a>
      <button class="btn btn-sm btn-outline" onclick="changePassword()">🔑 修改密码</button>
      <button class="btn btn-sm btn-outline" onclick="doLogout()">退出</button>
    </div>
  </div>
  <div class="tabs">
    <div class="tab active" data-tab="prompts" onclick="switchTab('prompts')">📝 提示词管理</div>
    <div class="tab" data-tab="qa" onclick="switchTab('qa')">💬 在线问答</div>
    <div class="tab" data-tab="usage" onclick="switchTab('usage')">📊 Token 用量</div>
    <div class="tab" data-tab="kb" onclick="switchTab('kb')">📚 知识库</div>
    <div class="tab" data-tab="users" id="tabUsersBtn" style="display:none" onclick="switchTab('users')">👥 用户管理</div>
    <div class="tab" data-tab="watermark" onclick="switchTab('watermark')">🖼️ 图片水印</div>
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
    <!-- Knowledge Base Tab -->
    <div id="tabKb" class="tab-panel">
      <div class="toolbar kb-toolbar">
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
          <input type="file" id="kbFile">
          <select id="kbAccess">
            <option value="public">公开(本租户可读)</option>
            <option value="restricted">受限(仅自己+管理员)</option>
          </select>
          <select id="kbTenant" style="display:none"></select>
          <button class="btn btn-primary btn-sm" onclick="uploadKb()">⬆️ 上传并入库</button>
        </div>
      </div>
      <div class="kb-card">
        <div class="kb-card-header">
          <h2>📚 文档列表</h2>
          <button class="btn btn-sm btn-outline" onclick="rebuildKb()">♻️ 全量重建</button>
        </div>
        <div id="kbStats" class="kb-stats"></div>
        <div style="overflow:auto;flex:1">
          <table class="kb-table">
            <thead>
              <tr><th>路径</th><th>租户</th><th>分片</th><th>大小</th><th></th></tr>
            </thead>
            <tbody id="kbRows"></tbody>
          </table>
        </div>
        <div id="kbMsg" class="kb-msg"></div>
        <div id="kbErr" class="kb-err"></div>
      </div>
    </div>
    <!-- User Management Tab (super_admin only) -->
    <div id="tabUsers" class="tab-panel">
      <div class="toolbar">
        <div>
          <h2>👥 用户管理</h2>
          <p class="sub">管理本租户用户（超级管理员可管理全部租户）。可新增用户、调整角色与启用状态。</p>
        </div>
        <div style="display:flex;gap:8px">
          <button class="btn btn-sm btn-primary" onclick="openCreateUser()">➕ 新增用户</button>
          <button class="btn btn-sm btn-outline" onclick="loadUsers()">🔄 刷新</button>
        </div>
      </div>
      <div style="overflow:auto;flex:1">
        <table class="kb-table">
          <thead>
            <tr><th>ID</th><th>账号</th><th>显示名</th><th>角色</th><th>租户</th><th>状态</th><th></th></tr>
          </thead>
          <tbody id="usersRows"></tbody>
        </table>
      </div>
      <div id="usersMsg" class="kb-msg"></div>
      <div id="usersErr" class="kb-err"></div>
    </div>

    <!-- ===== 图片水印面板 ===== -->
    <div id="tabWatermark" class="tab-panel">
      <div class="toolbar">
        <div>
          <h2>🖼️ 图片加水印</h2>
          <p class="sub">上传图片并填写水印文字/日期，生成斜铺整图的水印图，自动存入 knowledge/pic 目录。</p>
        </div>
      </div>
      <div style="display:flex;gap:24px;flex-wrap:wrap;flex:1;overflow:auto">
        <div style="flex:1;min-width:300px;max-width:420px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px">
          <div class="form-group">
            <label>① 选择图片</label>
            <input type="file" id="wmFile" accept="image/*">
          </div>
          <div class="form-group">
            <label>② 水印模板</label>
            <select id="wmPreset" onchange="applyPreset()">
              <option value="preset" selected>🛡️ 证件/身份证专用（推荐）</option>
              <option value="outsource">🏢 外包公司入职专用</option>
              <option value="custom">✏️ 自定义输入</option>
            </select>
          </div>
          <div class="form-group">
            <label>③ 水印文字</label>
            <textarea id="wmText" rows="3" placeholder="如：企业机密 · 请勿外传" style="width:100%;font:inherit;padding:8px;border:1px solid var(--border);border-radius:6px;background:var(--bg-2);color:var(--text-1);box-sizing:border-box;resize:vertical"></textarea>
            <small style="color:var(--text-3);font-size:11px">预设模板请替换【公司全称】为实际公司工商全称，可自由增删文字。</small>
          </div>
          <div class="form-group">
            <label>④ 水印日期</label>
            <input type="date" id="wmDate">
            <small id="wmDateHint" style="color:var(--text-3);font-size:11px"></small>
          </div>
          <button class="btn btn-primary" id="wmBtn" onclick="generateWatermark()">🔆 生成水印图片</button>
          <div id="wmMsg" class="kb-msg" style="margin-top:10px"></div>
          <div id="wmErr" class="kb-err" style="margin-top:6px"></div>
        </div>
        <div style="flex:1;min-width:300px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px;display:flex;flex-direction:column;align-items:center;justify-content:center">
          <div id="wmPreviewWrap" style="color:var(--text-3);font-size:13px">预览区（生成后显示）</div>
          <img id="wmPreview" style="display:none;max-width:100%;max-height:420px;border:1px solid var(--border);border-radius:var(--radius-sm);margin-top:12px">
          <a id="wmDownload" class="btn btn-outline" style="display:none;margin-top:14px" download>⬇️ 下载水印图</a>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ===== 新增用户模态框 ===== -->
<div id="createUserModal" class="modal-overlay hidden">
  <div class="modal">
    <div class="modal-header">
      <h3>➕ 新增用户</h3>
      <button class="btn btn-sm btn-outline" onclick="closeCreateUser()">✕</button>
    </div>
    <div class="modal-body">
      <div class="field">
        <label>账号（登录用户名）</label>
        <input id="cuUsername" placeholder="如: zhangsan" autocomplete="off">
      </div>
      <div class="field">
        <label>显示名称</label>
        <input id="cuDisplayName" placeholder="如: 张三">
      </div>
      <div class="field">
        <label>初始密码（至少 6 位）</label>
        <input id="cuPassword" type="password" placeholder="至少 6 位">
      </div>
      <div class="field">
        <label>角色</label>
        <select id="cuRole"></select>
      </div>
      <div class="field" id="cuTenantWrap">
        <label>所属租户</label>
        <select id="cuTenant"></select>
      </div>
      <div class="field">
        <div id="cuMsg" class="kb-msg"></div>
      </div>
    </div>
    <div class="modal-footer">
      <button class="btn btn-outline" onclick="closeCreateUser()">取消</button>
      <button class="btn btn-primary" onclick="submitCreateUser()">💾 创建</button>
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
  // 超级管理员 / 租户管理员 均显示"用户管理"tab
  const usersBtn = document.getElementById('tabUsersBtn');
  if (usersBtn) {
    const canManage = (currentUser.role === 'super_admin' || currentUser.role === 'admin');
    usersBtn.style.display = canManage ? 'inline-flex' : 'none';
  }
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
      if (u.role !== 'admin' && u.role !== 'super_admin'){ showNoPermission(u); return; }  // 仅 admin/super_admin 可进后台
      token = savedToken;
      currentUser = {
        username: u.username,
        display_name: u.display_name || '管理员',
        role: u.role,
        tenant_id: u.tenant_id || 'default'
      };
      showApp();
      loadKbTenants();
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
  if (name === 'kb') loadKb();
  if (name === 'users') loadUsers();
  if (name === 'watermark') applyPreset();
}

function applyPreset() {
  const sel = document.getElementById('wmPreset');
  const textEl = document.getElementById('wmText');
  const dateEl = document.getElementById('wmDate');
  const hint = document.getElementById('wmDateHint');
  const mode = sel ? sel.value : 'preset';
  const pad = n => String(n).padStart(2, '0');
  const fmt = d => `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}`;
  const fmtCn = d => `${d.getFullYear()} 年 ${pad(d.getMonth()+1)} 月 ${pad(d.getDate())} 日`;
  if (mode === 'preset') {
    const exp = new Date(); exp.setDate(exp.getDate() + 15);
    textEl.value = `仅限【公司全称】入职身份核验使用\n不作其他任何用途・再次复印无效\n有效期至：${fmtCn(exp)}`;
    dateEl.value = '';
    dateEl.disabled = true;
    if (hint) hint.textContent = '预设模板已内嵌有效期（提交日+15天）；如需调整请直接改上方文字。';
  } else if (mode === 'outsource') {
    const exp = new Date(); exp.setDate(exp.getDate() + 15);
    textEl.value = `仅限【公司名词】入职身份核验使用\n不作其他任何用途・再次复印无效\n有效期至：${fmtCn(exp)}`;
    dateEl.value = '';
    dateEl.disabled = true;
    if (hint) hint.textContent = '外包公司模板已填示例「公司名词」，请替换为实际签约公司工商全称。';
  } else {
    textEl.value = '';
    dateEl.value = fmt(new Date());
    dateEl.disabled = false;
    if (hint) hint.textContent = '自定义模式：日期将作为「 · 日期」追加到水印文字之后。';
  }
}

async function generateWatermark() {
  const fileEl = document.getElementById('wmFile');
  const textEl = document.getElementById('wmText');
  const dateEl = document.getElementById('wmDate');
  const msg = document.getElementById('wmMsg');
  const err = document.getElementById('wmErr');
  const btn = document.getElementById('wmBtn');
  const preview = document.getElementById('wmPreview');
  const wrap = document.getElementById('wmPreviewWrap');
  const dl = document.getElementById('wmDownload');
  msg.textContent = ''; err.textContent = '';
  if (!fileEl.files || !fileEl.files[0]) { err.textContent = '请先选择图片'; return; }
  if (!textEl.value.trim()) { err.textContent = '请填写水印文字'; return; }
  btn.disabled = true; btn.textContent = '⏳ 生成中…';
  try {
    const fd = new FormData();
    fd.append('file', fileEl.files[0]);
    fd.append('text', textEl.value.trim());
    fd.append('date', dateEl.value || '');
    const r = await fetch('/api/admin/watermark', {
      method: 'POST', headers: { 'Authorization': 'Bearer ' + (localStorage.getItem('admin_token') || '') }, body: fd
    });
    const data = await r.json();
    if (!r.ok || !data.success) throw new Error(data.error || ('HTTP ' + r.status));
    preview.src = data.url + '?t=' + Date.now();
    preview.style.display = 'block';
    wrap.style.display = 'none';
    dl.href = data.url;
    dl.style.display = 'inline-flex';
    msg.textContent = '✅ 已生成：' + data.file + '（已存入 knowledge/pic）';
  } catch (e) {
    err.textContent = '生成失败：' + e.message;
  } finally {
    btn.disabled = false; btn.textContent = '🔆 生成水印图片';
  }
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

function fbToken(){ try{ if(typeof getRagToken==='function'){const t=getRagToken(); if(t) return t;} }catch(e){} try{ if(typeof token!=='undefined'&&token) return token; }catch(e){} try{ return localStorage.getItem('rag_token')||''; }catch(e){ return ''; } }
function buildFeedbackBar(query, answer){
  const bar=document.createElement('div'); bar.className='fb-bar';
  bar.setAttribute('data-query', query||''); bar.setAttribute('data-answer', answer||'');
  bar.innerHTML='<button class="fb-btn fb-up" onclick="fbVote(this,1)">👍 有帮助</button>'+
                '<button class="fb-btn fb-down" onclick="fbVote(this,-1)">👎 没帮助</button>'+
                '<button class="fb-btn" onclick="fbText(this)">💬 反馈</button>'+
                '<span class="fb-tip">点踩自动记入 Bad Case</span>';
  return bar;
}
async function fbVote(btn, rating){
  const bar=btn.parentElement; if(bar.getAttribute('data-voted')) return;
  const query=bar.getAttribute('data-query')||''; const answer=bar.getAttribute('data-answer')||'';
  try{
    const resp=await fetch('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+fbToken()},body:JSON.stringify({query:query,answer:answer,rating:rating})});
    const data=await resp.json().catch(()=>({}));
    if(!resp.ok || data.ok===false){ throw new Error('HTTP '+resp.status); }
    bar.setAttribute('data-voted','1');
    if(rating>0){ bar.querySelector('.fb-up').classList.add('on'); bar.querySelector('.fb-tip').textContent='已记录反馈'; }
    else { bar.querySelector('.fb-down').classList.add('on'); bar.querySelector('.fb-tip').textContent='已记入 Bad Case（open）'; }
  }catch(e){ console.error(e); bar.querySelector('.fb-tip').textContent='反馈失败，请重试'; }
}
async function fbText(btn){
  const bar=btn.parentElement; const txt=prompt('请描述问题（可选）：',''); if(txt===null) return;
  const query=bar.getAttribute('data-query')||''; const answer=bar.getAttribute('data-answer')||'';
  if(bar.getAttribute('data-voted')) return;
  try{
    const resp=await fetch('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+fbToken()},body:JSON.stringify({query:query,answer:answer,rating:-1,feedback_text:txt})});
    const data=await resp.json().catch(()=>({}));
    if(!resp.ok || data.ok===false){ throw new Error('HTTP '+resp.status); }
    bar.setAttribute('data-voted','1'); bar.querySelector('.fb-down').classList.add('on'); bar.querySelector('.fb-tip').textContent='已记入 Bad Case（open）';
  }catch(e){ console.error(e); bar.querySelector('.fb-tip').textContent='反馈失败，请重试'; }
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
      bubble.appendChild(buildFeedbackBar(question, finalAnswer));
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
      bubble.appendChild(buildFeedbackBar(question, finalAnswer));
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

// ===== Knowledge Base Tab =====
let kbTenants = ['default'];

async function loadKbTenants() {
  try {
    const res = await fetch('/api/tenants', {headers: {'Authorization': 'Bearer ' + token}});
    if (res.ok) {
      const d = await res.json();
      kbTenants = d.tenants && d.tenants.length ? d.tenants : ['default'];
    }
  } catch (e) { console.warn('加载租户列表失败', e); }
  const sel = document.getElementById('kbTenant');
  if (!currentUser || currentUser.role !== 'super_admin') {
    sel.style.display = 'none';
    return;
  }
  sel.innerHTML = '';
  kbTenants.forEach(t => {
    const opt = document.createElement('option');
    opt.value = t; opt.textContent = t;
    if (t === currentUser.tenant_id) opt.selected = true;
    sel.appendChild(opt);
  });
  sel.style.display = 'inline-block';
}

async function loadKb() {
  try {
    const res = await fetch('/api/docs', {headers: {'Authorization': 'Bearer ' + token}});
    const d = await res.json();
    const tb = document.getElementById('kbRows');
    tb.innerHTML = '';
    (d.files || []).forEach(f => {
      const tr = document.createElement('tr');
      tr.innerHTML = '<td>' + escapeHtml(f.path) + '</td><td>' + escapeHtml(f.tenant) + '</td><td>' + f.chunks + '</td><td>' + (f.size / 1024).toFixed(1) + 'KB</td>';
      const td = document.createElement('td');
      const b = document.createElement('button');
      b.className = 'kb-btn-danger';
      b.textContent = '删除';
      b.onclick = () => deleteKb(f.path);
      td.appendChild(b);
      tr.appendChild(td);
      tb.appendChild(tr);
    });
  } catch (e) { kbErr('加载文档列表失败: ' + e.message); }
  try {
    const res = await fetch('/api/docs/stats', {headers: {'Authorization': 'Bearer ' + token}});
    const s = await res.json();
    document.getElementById('kbStats').textContent =
      '共 ' + (s.total_docs || 0) + ' 篇文档 · ' + (s.total_chunks || 0) + ' 分片' +
      (s.by_tenant ? ' · 租户分布: ' + Object.entries(s.by_tenant).map(([k, v]) => k + ':' + v).join('  ') : '');
  } catch (e) { document.getElementById('kbStats').textContent = ''; }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

async function uploadKb() {
  const file = document.getElementById('kbFile').files[0];
  if (!file) { kbErr('请选择文件'); return; }
  const btn = document.querySelector('#tabKb button[onclick="uploadKb()"]');
  if (btn) { btn.disabled = true; btn.textContent = '⏳ 上传入库中...'; }
  kbMsg('正在上传 ' + file.name + '，请稍候（大文件 PDF 解析+向量化可能需要几十秒）...');
  kbErr('');
  const fd = new FormData();
  fd.append('file', file);
  fd.append('access_level', document.getElementById('kbAccess').value);
  const t = document.getElementById('kbTenant');
  if (currentUser.role === 'super_admin' && t.value) fd.append('tenant', t.value);
  try {
    const res = await fetch('/api/docs/upload', {method: 'POST', headers: {'Authorization': 'Bearer ' + token}, body: fd});
    const d = await res.json();
    if (d.success) { kbMsg('✅ 已入库: ' + d.file + ' (' + d.chunks + ' 分片)'); loadKb(); }
    else kbErr(d.error || '上传失败');
  } catch (e) { kbErr('上传失败: ' + e.message); }
  finally { if (btn) { btn.disabled = false; btn.textContent = '⬆️ 上传并入库'; } }
}

async function deleteKb(path) {
  if (!confirm('确认删除 ' + path + ' ?')) return;
  try {
    const res = await fetch('/api/docs/' + encodeURIComponent(path), {method: 'DELETE', headers: {'Authorization': 'Bearer ' + token}});
    const d = await res.json();
    if (d.success) { kbMsg('已删除'); loadKb(); }
    else kbErr(d.error || '删除失败');
  } catch (e) { kbErr('删除失败: ' + e.message); }
}

async function rebuildKb() {
  if (!confirm('全量重建将重刷默认命名空间，耗时较长，继续？')) return;
  try {
    const res = await fetch('/api/docs/rebuild', {method: 'POST', headers: {'Authorization': 'Bearer ' + token}});
    const d = await res.json();
    if (d.success) kbMsg('重建已启动 job=' + d.job_id);
    else kbErr(d.error || '重建失败');
  } catch (e) { kbErr('重建失败: ' + e.message); }
}

function kbMsg(t) { document.getElementById('kbMsg').textContent = t; document.getElementById('kbErr').textContent = ''; }
function kbErr(t) { document.getElementById('kbErr').textContent = t; document.getElementById('kbMsg').textContent = ''; }

// ===== User Management Tab (super_admin only) =====
async function loadUsers() {
  const tb = document.getElementById('usersRows');
  tb.innerHTML = '<tr><td colspan="7">加载中…</td></tr>';
  try {
    const res = await fetch('/api/admin/users', {headers: {'Authorization': 'Bearer ' + token}});
    const d = await res.json();
    if (!res.ok) { usersErr(d.error || '加载失败'); tb.innerHTML = ''; return; }
    tb.innerHTML = '';
    (d.users || []).forEach(u => {
      const tr = document.createElement('tr');
      tr.innerHTML = '<td>' + u.id + '</td><td>' + escapeHtml(u.username) + '</td><td>' + escapeHtml(u.display_name || '') + '</td>' +
        '<td>' + userRoleSelect(u.id, u.role) + '</td><td>' + escapeHtml(u.tenant_id || 'default') + '</td>' +
        '<td>' + userActiveSelect(u.id, u.is_active) + '</td><td><button class="btn btn-sm btn-primary" onclick="saveUser(' + u.id + ')">保存</button></td>';
      tb.appendChild(tr);
    });
    usersMsg(''); usersErr('');
  } catch (e) { usersErr('加载用户列表失败: ' + e.message); }
}

function userRoleSelect(id, role) {
  let opts = [['user','普通用户'],['admin','租户管理员']];
  // 仅超级管理员可把用户设为超级管理员（防租户管理员越权提权）
  if (currentUser && currentUser.role === 'super_admin') {
    opts.push(['super_admin','超级管理员']);
  }
  let html = '<select id="role_' + id + '">';
  opts.forEach(([v, l]) => { html += '<option value="' + v + '"' + (role === v ? ' selected' : '') + '>' + l + '</option>'; });
  return html + '</select>';
}

function userActiveSelect(id, isActive) {
  const opts = [[1,'启用'],[0,'禁用']];
  let html = '<select id="active_' + id + '">';
  opts.forEach(([v, l]) => { html += '<option value="' + v + '"' + (isActive == v ? ' selected' : '') + '>' + l + '</option>'; });
  return html + '</select>';
}

async function saveUser(id) {
  const role = document.getElementById('role_' + id).value;
  const isActive = parseInt(document.getElementById('active_' + id).value, 10);
  try {
    const res = await fetch('/api/admin/users/' + id, {
      method: 'PATCH',
      headers: {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'},
      body: JSON.stringify({role: role, is_active: isActive})
    });
    const d = await res.json();
    if (res.ok) { usersMsg('用户 #' + id + ' 已更新'); loadUsers(); }
    else usersErr(d.error || '保存失败');
  } catch (e) { usersErr('保存失败: ' + e.message); }
}

function usersMsg(t) { document.getElementById('usersMsg').textContent = t; document.getElementById('usersErr').textContent = ''; }
function usersErr(t) { document.getElementById('usersErr').textContent = t; document.getElementById('usersMsg').textContent = ''; }

// ===== 新增用户 =====
async function openCreateUser() {
  const modal = document.getElementById('createUserModal');
  modal.classList.remove('hidden');
  // 角色选项按当前操作者权限：超级管理员多一个「超级管理员」
  const roleSel = document.getElementById('cuRole');
  roleSel.innerHTML = '<option value="user">普通用户</option><option value="admin">租户管理员</option>'
    + ((currentUser && currentUser.role === 'super_admin') ? '<option value="super_admin">超级管理员</option>' : '');
  // 租户：超级管理员可选全部租户；租户管理员锁定本租户
  const tenantWrap = document.getElementById('cuTenantWrap');
  const tenantSel = document.getElementById('cuTenant');
  if (currentUser && currentUser.role === 'super_admin') {
    tenantWrap.style.display = 'block';
    try {
      const res = await fetch('/api/tenants', {headers: {'Authorization': 'Bearer ' + token}});
      const d = await res.json();
      tenantSel.innerHTML = (d.tenants || []).map(t => '<option value="' + t + '">' + t + '</option>').join('');
    } catch (e) { tenantSel.innerHTML = '<option value="default">default</option>'; }
  } else {
    tenantWrap.style.display = 'none';
  }
  document.getElementById('cuMsg').textContent = '';
}

function closeCreateUser() {
  document.getElementById('createUserModal').classList.add('hidden');
}

async function submitCreateUser() {
  const username = document.getElementById('cuUsername').value.trim();
  const displayName = document.getElementById('cuDisplayName').value.trim();
  const password = document.getElementById('cuPassword').value;
  const role = document.getElementById('cuRole').value;
  const tenant = (currentUser && currentUser.role === 'super_admin')
    ? document.getElementById('cuTenant').value : (currentUser ? currentUser.tenant_id : 'default');
  const msg = document.getElementById('cuMsg');
  if (!username) { msg.style.color = 'crimson'; msg.textContent = '请填写账号'; return; }
  if (password.length < 6) { msg.style.color = 'crimson'; msg.textContent = '密码至少 6 位'; return; }
  try {
    const res = await fetch('/api/admin/users', {
      method: 'POST',
      headers: {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'},
      body: JSON.stringify({username: username, display_name: displayName, password: password, role: role, tenant_id: tenant})
    });
    const d = await res.json();
    if (res.ok) {
      msg.style.color = 'green';
      msg.textContent = '已创建：' + d.username + '（' + d.role + ' / 租户 ' + d.tenant + '）';
      closeCreateUser();
      loadUsers();
    } else {
      msg.style.color = 'crimson'; msg.textContent = d.error || '创建失败';
    }
  } catch (e) { msg.style.color = 'crimson'; msg.textContent = '创建失败: ' + e.message; }
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
  .fb-bar{display:flex;gap:6px;align-items:center;margin-top:10px}
  .fb-btn{font-size:12px;color:#6b7280;border:1px solid #e2e5ed;background:#fff;border-radius:7px;padding:4px 11px;cursor:pointer;transition:all .15s;display:inline-flex;align-items:center;gap:4px}
  .fb-btn:hover{background:#f5f6fa;color:#1a1a2e}
  .fb-up.on{color:#3B6D11;border-color:#3B6D11;background:#EAF3DE}
  .fb-down.on{color:#A32D2D;border-color:#A32D2D;background:#FCEBEB}
  .fb-tip{font-size:11px;color:#9ca3af;margin-left:4px}

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

  /* ===== Tabs（与 /admin 风格一致） ===== */
  .tabs{display:flex;gap:2px;padding:0 24px;background:var(--surface);border-bottom:1px solid var(--border);flex-shrink:0}
  .tab{padding:13px 18px;font-size:14px;font-weight:500;color:var(--text-2);cursor:pointer;border-bottom:2px solid transparent;transition:all .15s;user-select:none;white-space:nowrap}
  .tab:hover{color:var(--text)}
  .tab.active{color:var(--primary);border-bottom-color:var(--primary)}
  .tab-panel{display:none;flex:1;min-height:0;flex-direction:column}
  .tab-panel.active{display:flex}

  /* ===== 知识库 tab 内嵌 ===== */
  .kb-toolbar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:16px 24px 0;max-width:1100px;width:100%;margin:0 auto}
  .kb-toolbar select,.kb-toolbar input[type=file]{padding:6px 10px;border:1px solid var(--border);border-radius:var(--radius-sm);font-size:14px;background:var(--surface);color:var(--text)}
  .kb-body{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px;margin:16px 24px 24px;flex:1;overflow:hidden;display:flex;flex-direction:column;max-width:1100px;width:100%;margin-left:auto;margin-right:auto}
  .kb-card-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
  .kb-card-header h2{font-size:17px;font-weight:600}
  .kb-stats{color:var(--text-2);font-size:13px;margin-bottom:12px}
  .kb-table{width:100%;border-collapse:collapse;font-size:14px}
  .kb-table th,.kb-table td{text-align:left;padding:10px 8px;border-bottom:1px solid var(--border)}
  .kb-table th{color:var(--text-2);font-weight:600;background:var(--bg)}
  .kb-table tr:hover{background:var(--bg)}
  .kb-msg{color:var(--success);font-size:13px;margin-top:8px}
  .kb-err{color:var(--danger);font-size:13px;margin-top:8px}
  .kb-btn-danger{border:1px solid var(--danger);color:var(--danger);background:#fff;padding:5px 12px;border-radius:var(--radius-sm);cursor:pointer;font-size:13px}
  .kb-btn-danger:hover{background:var(--danger-light)}

  /* ===== 用量 tab 内嵌 ===== */
  .usage-wrap{padding:20px 24px;flex:1;overflow-y:auto;max-width:1100px;width:100%;margin:0 auto}
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

<!-- Tabs（与 /admin 风格一致） -->
<div class="tabs">
  <div class="tab active" data-tab="chat" onclick="switchTab('chat')">💬 知识问答</div>
  <div class="tab" data-tab="kb" onclick="switchTab('kb')">📚 知识库</div>
  <div class="tab" data-tab="usage" onclick="switchTab('usage')">📊 我的用量</div>
</div>

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

<!-- Tab: 知识问答 -->
<div id="tabChat" class="tab-panel active">
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
</div>

<!-- Tab: 知识库 -->
<div id="tabKb" class="tab-panel">
  <div class="kb-toolbar">
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <input type="file" id="kbFile">
      <select id="kbAccess">
        <option value="public">公开(本租户可读)</option>
        <option value="restricted">受限(仅自己+管理员)</option>
      </select>
      <button class="btn btn-primary btn-sm" onclick="uploadKb()">⬆️ 上传并入库</button>
    </div>
  </div>
  <div class="kb-body">
    <div class="kb-card-header">
      <h2>📚 文档列表</h2>
      <button class="btn btn-sm btn-outline" onclick="rebuildKb()">♻️ 全量重建</button>
    </div>
    <div id="kbStats" class="kb-stats"></div>
    <div style="overflow:auto;flex:1">
      <table class="kb-table">
        <thead>
          <tr><th>路径</th><th>租户</th><th>分片</th><th>大小</th><th></th></tr>
        </thead>
        <tbody id="kbRows"></tbody>
      </table>
    </div>
    <div id="kbMsg" class="kb-msg"></div>
    <div id="kbErr" class="kb-err"></div>
  </div>
</div>

<!-- Tab: 我的用量 -->
<div id="tabUsage" class="tab-panel">
  <div class="usage-wrap">
    <div class="usage-toolbar">
      <div class="range-btn active" data-range="today" onclick="setRangeInline('today')">今日</div>
      <div class="range-btn" data-range="7d" onclick="setRangeInline('7d')">近 7 天</div>
      <div class="range-btn" data-range="30d" onclick="setRangeInline('30d')">近 30 天</div>
      <div class="range-btn" data-range="all" onclick="setRangeInline('all')">全部</div>
      <div class="usage-db-tag" id="usageDbTagInline">—</div>
    </div>
    <div id="usageInline">
      <div class="usage-empty">加载中…</div>
    </div>
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
// 顶部 Tab 切换（与 /admin 风格一致：知识问答 / 知识库 / 我的用量）
// ============================================================
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(function(t){ t.classList.remove('active'); });
  document.querySelectorAll('.tab-panel').forEach(function(p){ p.classList.remove('active'); });
  document.querySelector('.tab[data-tab="' + name + '"]').classList.add('active');
  document.getElementById('tab' + name.charAt(0).toUpperCase() + name.slice(1)).classList.add('active');
  if (name === 'usage') loadUsage('#usageInline', '#usageDbTagInline');
  if (name === 'kb') loadKb();
}

// ============================================================
// 知识库 tab（内嵌上传 / 列表 / 删除 / 重建，隔离规则与普通用户一致）
// ============================================================
async function loadKb() {
  try {
    const res = await fetch('/api/docs', {headers: {'Authorization': 'Bearer ' + getRagToken()}});
    const d = await res.json();
    const tb = document.getElementById('kbRows');
    tb.innerHTML = '';
    (d.files || []).forEach(function(f) {
      const tr = document.createElement('tr');
      tr.innerHTML = '<td>' + escapeHtml(f.path) + '</td><td>' + escapeHtml(f.tenant) + '</td><td>' + f.chunks + '</td><td>' + (f.size / 1024).toFixed(1) + 'KB</td>';
      const td = document.createElement('td');
      const b = document.createElement('button');
      b.className = 'kb-btn-danger';
      b.textContent = '删除';
      b.onclick = function() { deleteKb(f.path); };
      td.appendChild(b);
      tr.appendChild(td);
      tb.appendChild(tr);
    });
  } catch (e) { kbErr('加载文档列表失败: ' + e.message); }
  try {
    const res = await fetch('/api/docs/stats', {headers: {'Authorization': 'Bearer ' + getRagToken()}});
    const s = await res.json();
    document.getElementById('kbStats').textContent =
      '共 ' + (s.total_docs || 0) + ' 篇文档 · ' + (s.total_chunks || 0) + ' 分片' +
      (s.by_tenant ? ' · 租户分布: ' + Object.entries(s.by_tenant).map(function(kv){return kv[0] + ':' + kv[1];}).join('  ') : '');
  } catch (e) { document.getElementById('kbStats').textContent = ''; }
}

async function uploadKb() {
  const file = document.getElementById('kbFile').files[0];
  if (!file) { kbErr('请选择文件'); return; }
  const btn = document.querySelector('#tabKb button[onclick="uploadKb()"]');
  if (btn) { btn.disabled = true; btn.textContent = '⏳ 上传入库中...'; }
  kbMsg('正在上传 ' + file.name + '，请稍候（大文件 PDF 解析+向量化可能需要几十秒）...');
  kbErr('');
  const fd = new FormData();
  fd.append('file', file);
  fd.append('access_level', document.getElementById('kbAccess').value);
  try {
    const res = await fetch('/api/docs/upload', {method: 'POST', headers: {'Authorization': 'Bearer ' + getRagToken()}, body: fd});
    const d = await res.json();
    if (d.success) { kbMsg('✅ 已入库: ' + d.file + ' (' + d.chunks + ' 分片)'); loadKb(); }
    else kbErr(d.error || '上传失败');
  } catch (e) { kbErr('上传失败: ' + e.message); }
  finally { if (btn) { btn.disabled = false; btn.textContent = '⬆️ 上传并入库'; } }
}

async function deleteKb(path) {
  if (!confirm('确认删除 ' + path + ' ?')) return;
  try {
    const res = await fetch('/api/docs/' + encodeURIComponent(path), {method: 'DELETE', headers: {'Authorization': 'Bearer ' + getRagToken()}});
    const d = await res.json();
    if (d.success) { kbMsg('已删除'); loadKb(); }
    else kbErr(d.error || '删除失败');
  } catch (e) { kbErr('删除失败: ' + e.message); }
}

async function rebuildKb() {
  if (!confirm('全量重建将重刷默认命名空间，耗时较长，继续？')) return;
  try {
    const res = await fetch('/api/docs/rebuild', {method: 'POST', headers: {'Authorization': 'Bearer ' + getRagToken()}});
    const d = await res.json();
    if (d.success) kbMsg('重建已启动 job=' + d.job_id);
    else kbErr(d.error || '重建失败');
  } catch (e) { kbErr('重建失败: ' + e.message); }
}

function kbMsg(t) { document.getElementById('kbMsg').textContent = t; document.getElementById('kbErr').textContent = ''; }
function kbErr(t) { document.getElementById('kbErr').textContent = t; document.getElementById('kbMsg').textContent = ''; }

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

async function loadUsage(boxSel, dbTagSel) {
  const box = document.getElementById(boxSel || 'usageContent');
  box.innerHTML = '<div class="usage-empty">加载中…</div>';
  try {
    const url = `/api/usage/me?user=${encodeURIComponent(currentUser)}&range=${usageRange}&limit=100`;
    const resp = await fetch(url);
    const data = await resp.json();
    if (!resp.ok) {
      box.innerHTML = `<div class="usage-empty">❌ ${data.error || '查询失败'}</div>`;
      return;
    }
    const tag = document.getElementById(dbTagSel || 'usageDbTag');
    if (tag) tag.textContent = data.persisted ? `已持久化 · ${data.db}` : '仅内存（重启即丢）';
    renderUsage(box, data);
  } catch (e) {
    box.innerHTML = '<div class="usage-empty">❌ 网络异常，无法获取用量</div>';
  }
}

// 用量 tab 内联切换（与弹窗 openUsage 共用 loadUsage/renderUsage）
function setRangeInline(r) {
  usageRange = r;
  document.querySelectorAll('#tabUsage .range-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.range === r));
  loadUsage('#usageInline', '#usageDbTagInline');
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
    let lastUser = '';
    for (const m of msgs) {
      if (m.role === 'user') { lastUser = m.content; addUserMessage(m.content); }
      else if (m.role === 'assistant') addAssistantMessage(m.content, lastUser);
      else if (m.role === 'summary') addSummaryMessage(m.content);
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
      addAssistantMessage(finalAnswer, question);
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
      addAssistantMessage(finalAnswer, question);
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

function fbToken(){ try{ if(typeof getRagToken==='function'){const t=getRagToken(); if(t) return t;} }catch(e){} try{ if(typeof token!=='undefined'&&token) return token; }catch(e){} try{ return localStorage.getItem('rag_token')||''; }catch(e){ return ''; } }
function buildFeedbackBar(query, answer){
  const bar=document.createElement('div'); bar.className='fb-bar';
  bar.setAttribute('data-query', query||''); bar.setAttribute('data-answer', answer||'');
  bar.innerHTML='<button class="fb-btn fb-up" onclick="fbVote(this,1)">👍 有帮助</button>'+
                '<button class="fb-btn fb-down" onclick="fbVote(this,-1)">👎 没帮助</button>'+
                '<button class="fb-btn" onclick="fbText(this)">💬 反馈</button>'+
                '<span class="fb-tip">点踩自动记入 Bad Case</span>';
  return bar;
}
async function fbVote(btn, rating){
  const bar=btn.parentElement; if(bar.getAttribute('data-voted')) return;
  const query=bar.getAttribute('data-query')||''; const answer=bar.getAttribute('data-answer')||'';
  try{
    const resp=await fetch('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+fbToken()},body:JSON.stringify({query:query,answer:answer,rating:rating})});
    const data=await resp.json().catch(()=>({}));
    if(!resp.ok || data.ok===false){ throw new Error('HTTP '+resp.status); }
    bar.setAttribute('data-voted','1');
    if(rating>0){ bar.querySelector('.fb-up').classList.add('on'); bar.querySelector('.fb-tip').textContent='已记录反馈'; }
    else { bar.querySelector('.fb-down').classList.add('on'); bar.querySelector('.fb-tip').textContent='已记入 Bad Case（open）'; }
  }catch(e){ console.error(e); bar.querySelector('.fb-tip').textContent='反馈失败，请重试'; }
}
async function fbText(btn){
  const bar=btn.parentElement; const txt=prompt('请描述问题（可选）：',''); if(txt===null) return;
  const query=bar.getAttribute('data-query')||''; const answer=bar.getAttribute('data-answer')||'';
  if(bar.getAttribute('data-voted')) return;
  try{
    const resp=await fetch('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+fbToken()},body:JSON.stringify({query:query,answer:answer,rating:-1,feedback_text:txt})});
    const data=await resp.json().catch(()=>({}));
    if(!resp.ok || data.ok===false){ throw new Error('HTTP '+resp.status); }
    bar.setAttribute('data-voted','1'); bar.querySelector('.fb-down').classList.add('on'); bar.querySelector('.fb-tip').textContent='已记入 Bad Case（open）';
  }catch(e){ console.error(e); bar.querySelector('.fb-tip').textContent='反馈失败，请重试'; }
}

function addAssistantMessage(text, query) {
  const area = document.getElementById('chatArea');
  const div = document.createElement('div');
  div.className = 'message';
  // 先把 [[FIG:assets/figures/xxx.png]] 切成 (文本段, 图段) 交错渲染
  // （simpleMarkdown 不识别自定义占位符，会包到 <p> 里破坏布局，所以前置拆分）
  const html = renderAssistantContent(text);
  div.innerHTML = `
    <div class="msg-avatar assistant">AI</div>
    <div class="msg-body">
      <div class="msg-role-name">AI 助手</div>
      <div class="msg-content">${html}</div>
      <div class="msg-time">${formatTime()}</div>
    </div>
  `;
  area.appendChild(div);
  // 聊天反馈按钮（赞/踩/反馈）：挂在答案下方，点踩自动沉淀 bad case
  if (query) {
    const body = div.querySelector('.msg-body');
    if (body) body.appendChild(buildFeedbackBar(query, text));
  }
  scrollToBottom();
}

/* 把文本切成 markdown 段 + 图段（按 [[FIG:path]] 占位符切分），分别渲染。
   图段渲染为 <div class="chat-figure"><img src="/api/figures/path" loading="lazy"></div>。
   路径必须是 assets/figures/ 开头的相对路径；其他路径一律忽略（防注入）。 */
function renderAssistantContent(text) {
  const parts = String(text || '').split(/\[\[FIG:([^\]]+)\]\]/);
  let out = '';
  for (let i = 0; i < parts.length; i++) {
    if (i % 2 === 0) {
      // 文本段：走 markdown
      if (parts[i].trim()) out += simpleMarkdown(parts[i]);
    } else {
      // 图段：渲染 <img>
      // 注意：rel 含中文/空格（如 assets/figures/27 【技术对接】.../fig_p001_1.png），
      // 旧正则 [\w\-./%]+ 不含中文与空格会整条判不通过导致图被静默丢弃。
      // 放宽到「assets/figures/ 开头 + .png 结尾」即可；越权防护由服务端
      // /api/figures 的 safe_join + abspath 包含校验兜底。
      const rel = parts[i].trim();
      if (!/^assets\/figures\/.+\.png$/i.test(rel)) continue;
      const url = '/api/figures/' + rel.split('/').map(encodeURIComponent).join('/');
      out += `<div class="chat-figure" style="margin:10px 0;border:1px solid #e1e5ea;border-radius:6px;overflow:hidden;background:#f8f9fa;">
        <img src="${url}" loading="lazy" alt="${rel}" style="display:block;max-width:100%;height:auto;">
        <div style="padding:4px 10px;font-size:12px;color:#5f5e5a;background:#fff;">📄 ${rel}</div>
      </div>`;
    }
  }
  return out || '';
}

function addSummaryMessage(text) {
  const area = document.getElementById('chatArea');
  const div = document.createElement('div');
  div.className = 'message summary-card';
  const html = simpleMarkdown(text);
  div.innerHTML = `
    <div class="msg-body" style="background:var(--primary-light, #e8f1fb); border-left:4px solid var(--primary, #185fa5); border-radius:8px;">
      <div class="msg-role-name" style="color:var(--primary, #185fa5);">📋 历史摘要（前情提要）</div>
      <div class="msg-content">${html}</div>
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
        print(f"  ✓ 向量数据库: Milvus（唯一后端，URI={os.getenv('MILVUS_URI', 'http://192.168.200.128:19530')}）")
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
