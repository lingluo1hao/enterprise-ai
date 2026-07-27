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

# ====== Flask 应用 ======
app = Flask(__name__)

# 全局变量
orchestrator: RAGOrchestrator = None
llm = None
vector_db = None


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
    """切换用户角色"""
    data = request.get_json(force=True)
    new_role = data.get("role", ROLE_USER)

    if new_role not in (ROLE_ADMIN, ROLE_USER):
        return jsonify({"error": "无效角色"}), 400

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
# HTML 页面（内嵌单文件模板）
# ======================================================================

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
    <div class="status-indicator">
      <div class="status-dot"></div>
      <span>服务就绪</span>
    </div>
    <div class="role-badge user" id="roleBadge" onclick="toggleRole()" title="点击切换权限">
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
    <div class="input-hint" id="docHint">当前可访问：<strong>公开文档</strong> · 切换权限可解锁更多</div>
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
});

// ============================================================
// 角色切换
// ============================================================
async function toggleRole() {
  if (isQuerying) return;
  const newRole = currentRole === 'admin' ? 'user' : 'admin';

  try {
    const resp = await fetch('/api/role', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({role: newRole})
    });
    const data = await resp.json();
    if (data.role) {
      currentRole = data.role;
      updateRoleUI();
    }
  } catch (e) {
    console.error('Role switch failed:', e);
  }
}

function updateRoleUI() {
  const badge = document.getElementById('roleBadge');
  const label = document.getElementById('roleLabel');
  const dot = badge.querySelector('.role-dot');
  const hint = document.getElementById('docHint');

  badge.className = 'role-badge ' + currentRole;
  dot.className = 'role-dot ' + currentRole;
  label.textContent = currentRole === 'admin' ? '特权用户' : '普通用户';

  if (currentRole === 'admin') {
    hint.innerHTML = '当前可访问：<strong>全部文档</strong>（含受限文档）';
  } else {
    hint.innerHTML = '当前可访问：<strong>公开文档</strong> · 点击右上角切换权限';
  }
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
    global orchestrator, llm, vector_db

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
    orchestrator = RAGOrchestrator(llm, vector_db, fast_mode=True, user_role=DEFAULT_ROLE)
    print(f"  ✓ RAG Orchestrator 就绪")
    print(f"  ✓ 用户角色: {DEFAULT_ROLE}")
    print(f"  ✓ Web 界面已启动\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG Agent Web Server")
    parser.add_argument("--port", type=int, default=8080, help="Web 服务器端口 (默认 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="绑定的主机地址")
    args = parser.parse_args()

    init_system()

    url = f"http://localhost:{args.port}"
    print(f"  🌐 浏览器打开: {url}")
    print(f"  按 Ctrl+C 停止服务器\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
