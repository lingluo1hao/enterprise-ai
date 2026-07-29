"""
================================================================================
  RAG Agent Web 界面 — Flask + SSE 实时进度推送
================================================================================

  为非技术人员提供友好的人工智能问答界面。
  - 支持普通用户 / 特权用户角色切换
  - 实时显示推理进度（SSE 推送）
  - 自动连接 Redis 缓存加速重复问题

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

from flask import Flask, request, jsonify, Response

# ====== 导入核心模块 ======
from advanced_rag_agent import (
    OLLAMA_URL, MODEL_NAME, DB_PATH, DOC_FOLDER,
    ROLE_ADMIN, ROLE_USER, DEFAULT_ROLE,
    AccessControlFilter, CacheManager,
    RAGOrchestrator,
    OllamaLLM,
    VectorStoreManager,
)


class LangGraphEngine:
    """适配器：让 LangGraphRAGApp 兼容 rag_web_server 的 RAGOrchestrator 接口"""

    def __init__(self, fast_mode=True, user_role=DEFAULT_ROLE):
        from langgraph_rag_agent import LangGraphRAGApp
        self.app = LangGraphRAGApp(fast_mode=fast_mode)
        self.user_role = user_role
        self.cache = self.app.cache
        # 兼容角色切换中的 skill_registry.get_skill() 调用
        self.skill_registry = type("SR", (), {"get_skill": lambda self, name: None})()

    def query(self, question, user_role=None):
        role = user_role or self.user_role
        return self.app.query(question, role=role, session_id="web_session")

    def check_unfinished_tasks(self, session_id="web_session"):
        """查询指定会话的未完成任务（断点检测）"""
        return self.app.check_unfinished_tasks(session_id)

    def resume_task(self, task_id, session_id="web_session"):
        """从断点恢复执行指定任务"""
        return self.app.resume(task_id, session_id=session_id)


# ====== Flask 应用 ======
app = Flask(__name__)

# 全局变量
orchestrator = None
llm = None
vector_db = None
use_langgraph = False


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
# API 路由
# ======================================================================

@app.route("/")
def index():
    """返回聊天界面"""
    return _HTML_PAGE


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
    前端用 fetch + 轮询 /api/progress。
    """
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    user_role = data.get("role", orchestrator.user_role)

    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    # 特权角色需要管理员 Token
    if user_role == ROLE_ADMIN:
        auth_result = _require_admin_token()
        if auth_result:
            return auth_result

    result = orchestrator.query(question, user_role=user_role)
    return jsonify({"answer": result, "role": orchestrator.user_role})


@app.route("/api/query/stream", methods=["POST"])
def api_query_stream():
    """
    流式查询 — SSE 实时推送进度。
    启动后台线程执行查询，前端通过 EventSource 接收进度事件。
    """

    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    user_role = data.get("role", orchestrator.user_role)

    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    # 特权角色需要管理员 Token
    if user_role == ROLE_ADMIN:
        auth_result = _require_admin_token()
        if auth_result:
            return auth_result

    # 创建队列用于接收进度
    progress_queue = queue.Queue()

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
                        question, user_role=user_role
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
    切换用户角色。

    安全策略：
      - 切换到普通用户 (user) 无需认证；
      - 切换到特权用户 (admin) 必须在请求头携带有效管理 Token，
        防止未授权用户通过直接调用 API 访问受限文档。
    """
    data = request.get_json(force=True)
    new_role = data.get("role", ROLE_USER)

    if new_role not in (ROLE_ADMIN, ROLE_USER):
        return jsonify({"error": "无效角色"}), 400

    # 切换到 admin 需要认证
    if new_role == ROLE_ADMIN:
        auth_result = _require_admin_token()
        if auth_result:
            return auth_result

    orchestrator.user_role = new_role
    orchestrator.cache.current_role = new_role
    doc_skill = orchestrator.skill_registry.get_skill("doc_search")
    if doc_skill:
        doc_skill.user_role = new_role

    return jsonify({
        "role": new_role,
        "description": AccessControlFilter.get_role_description(new_role),
    })


# ======================================================================
# 内部辅助函数
# ======================================================================

def _require_admin_token():
    """
    校验请求是否携带有效的管理员 Token。

    用于保护 admin 角色相关的接口，防止未授权用户直接调用 API
    访问受限文档。返回 None 表示校验通过，否则返回 (响应, 状态码)。
    """
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    from prompt_manager import get_auth_manager
    auth = get_auth_manager()
    if not auth.verify_token(token):
        return jsonify({"error": "需要管理员权限"}), 403
    return None


# ======================================================================
# 断点重续 API — 多层记忆核心功能
# ======================================================================

@app.route("/api/tasks/unfinished")
def get_unfinished_tasks():
    """
    查询当前会话的未完成任务。

    用户登录/连接时调用。如果有 interrupted 状态的任务，
    说明上次执行被中断（服务宕机或用户关闭客户端）。
    前端收到后可弹窗提示用户："上次有未完成的任务，是否恢复？"
    """
    session_id = request.args.get("session_id", "web_session")
    if not use_langgraph or not isinstance(orchestrator, LangGraphEngine):
        return jsonify({"tasks": [], "message": "当前引擎不支持断点恢复"})
    tasks = orchestrator.check_unfinished_tasks(session_id)
    return jsonify({"tasks": tasks, "count": len(tasks)})


@app.route("/api/tasks/resume", methods=["POST"])
def resume_task():
    """
    从断点恢复执行指定任务。

    前端用户点击"恢复"按钮后调用此接口。
    后端读取 MySQL task_checkpoints 最后一条快照，恢复 state，重新执行图。
    """
    data = request.get_json(force=True)
    task_id = data.get("task_id")
    session_id = data.get("session_id", "web_session")

    if not task_id:
        return jsonify({"error": "缺少 task_id 参数"}), 400

    if not use_langgraph or not isinstance(orchestrator, LangGraphEngine):
        return jsonify({"error": "当前引擎不支持断点恢复"}), 400

    try:
        answer = orchestrator.resume_task(task_id, session_id)
        return jsonify({"answer": answer, "task_id": task_id, "status": "completed"})
    except Exception as e:
        return jsonify({"error": str(e), "task_id": task_id, "status": "failed"}), 500


# ======================================================================
# 管理后台 API — 提示词工程管理 + 用户认证
# ======================================================================

# ---- 认证相关 ----

@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    """管理员登录"""
    data = request.get_json(force=True)
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400

    from prompt_manager import AuthManager, get_auth_manager
    auth = get_auth_manager()
    user = auth.login(username, password)

    if user:
        return jsonify({
            "success": True,
            "user": {"username": user["username"], "display_name": user["display_name"]},
            "token": user["token"],
        })
    else:
        return jsonify({"success": False, "error": "用户名或密码错误"}), 401


@app.route("/api/admin/me", methods=["GET"])
def admin_me():
    """
    获取当前登录管理员信息。

    前端页面刷新后，用 localStorage 中的 token 调用此接口恢复登录状态。
    """
    auth_result = _require_admin_token()
    if auth_result:
        return auth_result

    from prompt_manager import get_auth_manager
    auth = get_auth_manager()
    # 当前 token 只做格式校验，返回默认管理员信息
    return jsonify({
        "username": "admin",
        "display_name": "管理员",
        "role": "admin"
    })


@app.route("/api/admin/change-password", methods=["POST"])
def admin_change_password():
    """修改密码"""
    data = request.get_json(force=True)
    username = data.get("username", "").strip()
    old_password = data.get("old_password", "")
    new_password = data.get("new_password", "")

    if not new_password or len(new_password) < 6:
        return jsonify({"error": "新密码至少6位"}), 400

    from prompt_manager import AuthManager, get_auth_manager
    auth = get_auth_manager()
    if auth.change_password(username, old_password, new_password):
        return jsonify({"success": True, "message": "密码修改成功"})
    else:
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
    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    system = data.get("system", "")
    user_template = data.get("user_template", "")
    display_name = data.get("display_name", "")
    description = data.get("description", "")
    category = data.get("category", "general")

    if not name or not system:
        return jsonify({"error": "名称和系统提示词不能为空"}), 400

    from prompt_manager import get_prompt_manager
    pm = get_prompt_manager()
    ok = pm.save_prompt(
        name=name, system=system, user_template=user_template,
        display_name=display_name, description=description, category=category,
    )
    if ok:
        return jsonify({"success": True, "message": f"提示词 '{name}' 已保存"})
    else:
        return jsonify({"error": "保存失败，数据库不可用"}), 500


@app.route("/api/admin/prompts/<name>", methods=["DELETE"])
def admin_delete_prompt(name):
    """删除提示词"""
    from prompt_manager import get_prompt_manager
    pm = get_prompt_manager()
    ok = pm.delete_prompt(name)
    if ok:
        return jsonify({"success": True, "message": f"提示词 '{name}' 已删除"})
    else:
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
    return jsonify({"success": True, "imported": count})


@app.route("/api/admin/categories")
def admin_categories():
    """获取提示词分类列表"""
    from prompt_manager import get_prompt_manager
    pm = get_prompt_manager()
    return jsonify({"categories": pm.get_categories()})


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
</style>
</head>
<body>

<!-- ===== Login Page ===== -->
<div id="loginPage" class="login-page">
  <div class="login-card">
    <h2>🔐 系统管理</h2>
    <p class="sub">RAG Agent 提示词工程管理</p>
    <div class="form-group">
      <label>用户名</label>
      <input id="loginUser" type="text" placeholder="请输入用户名" value="admin" autocomplete="username">
    </div>
    <div class="form-group">
      <label>密码</label>
      <input id="loginPwd" type="password" placeholder="请输入密码" autocomplete="current-password">
    </div>
    <p id="loginError" class="error-msg"></p>
    <button class="btn btn-primary" style="margin-top:8px" onclick="doLogin()">登 录</button>
    <p style="text-align:center;margin-top:12px;font-size:12px;color:var(--text-3)">
      默认账号: admin / admin123
    </p>
  </div>
</div>

<!-- ===== App Page ===== -->
<div id="appPage" class="app-page hidden">
  <div class="app-header">
    <h1>⚙️ RAG Agent 系统管理</h1>
    <div class="user-info">
      <span>👤</span>
      <span class="name" id="displayName"></span>
      <a href="/" style="color:var(--primary);text-decoration:none;font-size:13px;margin-left:8px">💬 知识问答</a>
      <a href="/admin" style="color:var(--primary);text-decoration:none;font-size:13px">🔧 提示词管理</a>
      <button class="btn btn-sm btn-outline" onclick="doLogout()">退出</button>
    </div>
  </div>
  <div class="tabs">
    <div class="tab active" data-tab="prompts" onclick="switchTab('prompts')">📝 提示词管理</div>
    <div class="tab" data-tab="qa" onclick="switchTab('qa')">💬 在线问答</div>
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
          <button class="btn btn-sm btn-outline" onclick="changePassword()">🔑 修改密码</button>
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
        <div class="qa-input-area">
          <textarea class="qa-input" id="qaInput" rows="2" placeholder="输入你的问题，按 Enter 发送..." onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();askQA()}"></textarea>
          <button class="qa-send-btn" id="qaSendBtn" onclick="askQA()">发送</button>
        </div>
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

// ===== Init =====
document.getElementById('loginPwd').addEventListener('keydown', e => {
  if (e.key === 'Enter') doLogin();
});

// 页面加载时尝试自动登录
tryAutoLogin();

// ===== Auth =====
async function doLogin() {
  const username = document.getElementById('loginUser').value.trim();
  const password = document.getElementById('loginPwd').value.trim();
  const errEl = document.getElementById('loginError');

  if (!username || !password) {
    errEl.textContent = '请输入用户名和密码';
    errEl.style.display = 'block';
    return;
  }

  try {
    const res = await fetch('/api/admin/login', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({username, password})
    });
    const data = await res.json();
    if (data.success) {
      token = data.token;
      currentUser = data.user;
      localStorage.setItem('rag_admin_token', token);
      localStorage.setItem('rag_admin_user', JSON.stringify(currentUser));
      showApp();
    } else {
      errEl.textContent = data.error || '登录失败';
      errEl.style.display = 'block';
    }
  } catch(e) {
    errEl.textContent = '网络错误，请稍后重试';
    errEl.style.display = 'block';
  }
}

function doLogout() {
  token = '';
  currentUser = null;
  localStorage.removeItem('rag_admin_token');
  localStorage.removeItem('rag_admin_user');
  document.getElementById('loginPage').classList.remove('hidden');
  document.getElementById('appPage').classList.add('hidden');
  document.getElementById('loginPwd').value = '';
}

function showApp() {
  document.getElementById('displayName').textContent = currentUser.display_name;
  document.getElementById('loginPage').classList.add('hidden');
  document.getElementById('appPage').classList.remove('hidden');
  loadPrompts();
  loadCategories();
}

async function tryAutoLogin() {
  const savedToken = localStorage.getItem('rag_admin_token');
  if (!savedToken) return;

  try {
    const res = await fetch('/api/admin/me', {
      headers: {'Authorization': 'Bearer ' + savedToken}
    });
    if (res.ok) {
      const serverUser = await res.json();
      const localUser = JSON.parse(localStorage.getItem('rag_admin_user') || '{}');
      token = savedToken;
      currentUser = {
        username: serverUser.username || localUser.username || 'admin',
        display_name: serverUser.display_name || localUser.display_name || '管理员'
      };
      showApp();
    } else {
      localStorage.removeItem('rag_admin_token');
      localStorage.removeItem('rag_admin_user');
    }
  } catch(e) {
    localStorage.removeItem('rag_admin_token');
    localStorage.removeItem('rag_admin_user');
  }
}

// ===== Tabs =====
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelector(`.tab[data-tab="${name}"]`).classList.add('active');
  document.getElementById(`tab${name.charAt(0).toUpperCase() + name.slice(1)}`).classList.add('active');
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
  const oldPwd = prompt('请输入原密码:');
  if (!oldPwd) return;
  const newPwd = prompt('请输入新密码（至少6位）:');
  if (!newPwd || newPwd.length < 6) {
    alert('新密码至少6位');
    return;
  }
  const newPwd2 = prompt('请再次输���新密码:');
  if (newPwd !== newPwd2) {
    alert('两次密码不一致');
    return;
  }
  fetch('/api/admin/change-password', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({username: currentUser.username, old_password: oldPwd, new_password: newPwd})
  }).then(r => r.json()).then(d => {
    if (d.success) showToast(d.message, 'success');
    else showToast(d.error, 'error');
  });
}

// ===== Q&A =====
function setQAQuestion(text) {
  const input = document.getElementById('qaInput');
  input.value = text;
  input.focus();
}

async function askQA() {
  const input = document.getElementById('qaInput');
  const sendBtn = document.getElementById('qaSendBtn');
  const question = input.value.trim();
  if (!question) return;

  // 隐藏空状态提示
  const emptyEl = document.getElementById('qaEmpty');
  if (emptyEl) emptyEl.remove();

  input.value = '';
  sendBtn.disabled = true;
  sendBtn.textContent = '发送中...';

  const msgs = document.getElementById('qaMessages');
  msgs.innerHTML += `
    <div class="qa-msg user">
      <div class="avatar">👤</div>
      <div class="qa-bubble">${escapeHtml(question)}</div>
    </div>`;

  // Add loading
  const loadingId = 'loading_' + Date.now();
  msgs.innerHTML += `
    <div class="qa-msg assistant" id="${loadingId}">
      <div class="avatar">🤖</div>
      <div class="qa-bubble">思考中...</div>
    </div>`;
  msgs.scrollTop = msgs.scrollHeight;

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + token
      },
      body: JSON.stringify({question, role: 'admin'})
    });
    const data = await res.json();
    document.getElementById(loadingId).querySelector('.qa-bubble').textContent = data.answer || data.error || '无响应';
    document.getElementById(loadingId).scrollIntoView({behavior:'smooth'});
  } catch(e) {
    document.getElementById(loadingId).querySelector('.qa-bubble').textContent = '请求失败: ' + e.message;
  } finally {
    sendBtn.disabled = false;
    sendBtn.textContent = '发送';
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
</script>
</body>
</html>
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
  .input-hint{font-size:11px;color:var(--text-3);margin-top:8px;text-align:center}

  /* ===== Responsive ===== */
  @media (max-width:640px){
    .header{padding:10px 16px}
    .header h1{font-size:16px}
    .suggestions{flex-direction:column;align-items:stretch}
    .suggestion{text-align:center}
  }
</style>
</head>
<body>

<!-- Header -->
<header class="header">
  <div class="header-left">
    <div class="logo">AI</div>
    <h1>企业知识库问答</h1>
  </div>
  <div class="header-right">
    <a href="/admin" style="text-decoration:none;color:var(--primary);font-size:13px;font-weight:500;margin-right:8px">⚙️ 系统管理</a>
    <div class="status-indicator">
      <div class="status-dot"></div>
      <span>服务就绪</span>
    </div>
    <div class="role-badge user" id="roleBadge" title="当前为普通用户模式">
      <div class="role-dot user"></div>
      <span id="roleLabel">普通用户</span>
    </div>
  </div>
</header>

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
    <div class="input-row">
      <textarea id="questionInput" rows="1" placeholder="输入您的问题，按 Enter 发送（Shift+Enter 换行）..."
        onkeydown="handleKeydown(event)"></textarea>
      <button class="btn-send" id="sendBtn" onclick="sendQuestion()" title="发送">➤</button>
    </div>
    <div class="input-hint" id="docHint">当前为普通用户模式，仅可访问公开文档；管理员请从右上角进入系统管理</div>
  </div>
</div>

<script>
// ============================================================
// 全局状态
// ============================================================
let currentRole = 'user';
let isQuerying = false;

// ============================================================
// 初始化
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
  fetch('/api/health')
    .then(r => r.json())
    .then(data => {
      if (data.role) {
        currentRole = data.role;
        updateRoleUI();
      }
    })
    .catch(() => console.warn('Health check failed'));

  // ===== 断点重续：页面加载时检查未完成任务 =====
  fetch('/api/tasks/unfinished?session_id=web_session')
    .then(r => r.json())
    .then(data => {
      if (data.count && data.count > 0) {
        const task = data.tasks[0];
        const msg = '检测到上次有未完成的任务：\n"' + task.query + '"\n\n是否恢复执行？';
        if (confirm(msg)) {
          // 用户点击"确定"，恢复执行
          addMessage('user', task.query);
          isQuerying = true;
          toggleUI(true);
          addMessage('assistant', '正在从断点恢复执行...');
          fetch('/api/tasks/resume', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({task_id: task.task_id, session_id: 'web_session'})
          })
          .then(r => r.json())
          .then(result => {
            if (result.answer) {
              addMessage('assistant', result.answer);
            } else if (result.error) {
              addMessage('assistant', '恢复失败：' + result.error);
            }
          })
          .catch(e => addMessage('assistant', '恢复请求失败：' + e))
          .finally(() => {
            isQuerying = false;
            toggleUI(false);
          });
        }
      }
    })
    .catch(() => console.warn('Unfinished tasks check failed'));
});

// ============================================================
// 角色状态（聊天页固定为普通用户，不允许切换）
// ============================================================
function updateRoleUI() {
  const badge = document.getElementById('roleBadge');
  const label = document.getElementById('roleLabel');
  const dot = badge.querySelector('.role-dot');
  const hint = document.getElementById('docHint');

  badge.className = 'role-badge user';
  dot.className = 'role-dot user';
  label.textContent = '普通用户';
  hint.innerHTML = '当前为普通用户模式，仅可访问公开文档；管理员请从右上角进入系统管理';
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
  try {
    const resp = await fetch('/api/query/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question, role: currentRole})
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
    addErrorMessage('连接中断: ' + e.message);
  }

  isQuerying = false;
  updateSendButton();
  updateRoleUI();
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
  btn.disabled = isQuerying;
  btn.textContent = isQuerying ? '...' : '➤';
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

    # 1. LLM
    try:
        llm = OllamaLLM()
        print(f"  ✓ LLM: {MODEL_NAME} @ {OLLAMA_URL}")
    except Exception as e:
        print(f"  ✗ LLM 初始化失败: {e}")
        print("    请确认 Ollama 已运行且已加载模型")
        sys.exit(1)

    # 2. 向量数据库
    try:
        vector_db = VectorStoreManager.init_vector_store()
        print(f"  ✓ 向量数据库: {DB_PATH}")
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG Agent Web Server")
    parser.add_argument("--port", type=int, default=8080, help="Web 服务器端口 (默认 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="绑定的主机地址")
    parser.add_argument(
        "--langgraph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="使用 LangGraph 引擎（默认开启，--no-langgraph 使用旧版）",
    )
    args = parser.parse_args()

    global use_langgraph
    use_langgraph = args.langgraph

    init_system()

    url = f"http://localhost:{args.port}"
    print(f"  🌐 浏览器打开: {url}")
    print(f"  按 Ctrl+C 停止服务器\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
