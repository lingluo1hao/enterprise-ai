# gunicorn 生产配置：替换 app.run(threaded=True) 的单进程开发服务器
# 启动：gunicorn -c gunicorn_config.py rag_web_server:app
#
# 高并发要点：
#   - workers 多进程，突破 Flask 开发服务器单线程瓶颈
#   - gthread worker + threads，兼容 SSE 长连接（/api/tasks/resume）与同步 LLM 调用
#   - post_worker_init 在每个 worker 内调用 init_system()，否则 gunicorn 不跑 __main__，
#     向量库/编排器不会初始化（且 use_langgraph 顶层已用环境变量正确默认开启）
import os

bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '8080')}"
workers = int(os.getenv("GUNICORN_WORKERS", "4"))            # 进程数，建议 2*CPU+1
worker_class = os.getenv("GUNICORN_WORKER_CLASS", "gthread") # gthread 兼容 SSE 长连接
threads = int(os.getenv("GUNICORN_THREADS", "8"))            # 每 worker 线程数
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))          # LLM 调用可能慢，给足
graceful_timeout = 30
keepalive = 5


def post_worker_init(worker):
    """每个 worker 启动后执行系统初始化（gunicorn 不执行 __main__）。"""
    import rag_web_server
    rag_web_server.init_system()
