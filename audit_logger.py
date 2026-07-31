#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
审计日志模块
============

记录所有关键操作的结构化审计日志，输出到 `logs/audit.log` + 控制台。

日志格式：每行一条 JSON，字段：
  timestamp  — ISO 8601 时间戳
  ip         — 客户端 IP
  username   — 操作用户（匿名请求用 "anonymous"）
  action     — 操作类型（login/save_prompt/delete_prompt/change_password/query/import_defaults）
  target     — 操作对象（提示词名称、查询问题摘要等）
  result     — success / failure / blocked
  detail     — 补充信息（失败原因等）

使用方式：
  from audit_logger import AuditLogger
  logger = AuditLogger()
  logger.log(ip="192.168.1.1", username="admin", action="login", target="/api/admin/login", result="success")
"""

import os
import json
import threading
from datetime import datetime, timezone


LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "audit.log")

# UTC+8 时区
_TZ = timezone(__import__("datetime").timedelta(hours=8))


class AuditLogger:
    """线程安全的审计日志记录器。"""

    def __init__(self, log_path: str | None = None):
        self._log_path = log_path or LOG_FILE
        self._lock = threading.Lock()
        # 确保日志目录存在
        _dir = os.path.dirname(self._log_path)
        if _dir:
            os.makedirs(_dir, exist_ok=True)

    def log(self, ip: str = "unknown",
            username: str = "anonymous",
            action: str = "unknown",
            target: str = "",
            result: str = "success",
            detail: str = ""):
        """
        记录一条审计日志。

        参数：
          ip       — 客户端 IP
          username — 操作用户名
          action   — 操作类型（login / query / save_prompt / delete_prompt / change_password / import_defaults）
          target   — 操作对象
          result   — success / failure / blocked
          detail   — 补充说明
        """
        entry = {
            "timestamp": datetime.now(_TZ).isoformat(timespec="seconds"),
            "ip": ip,
            "username": username,
            "action": action,
            "target": target,
            "result": result,
            "detail": detail,
        }
        line = json.dumps(entry, ensure_ascii=False)

        # 同时输出到控制台和文件
        print(f"[AUDIT] {line}")
        with self._lock:
            try:
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError as e:
                print(f"[AuditLogger] 写入日志文件失败: {e}")


# 全局单例
_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """获取全局审计日志单例。"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
