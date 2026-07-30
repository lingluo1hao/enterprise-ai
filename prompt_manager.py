#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提示词工程管理模块
==================

将硬编码在代码中的 AI 提示词（Prompt）迁移到 MySQL 数据库，
支持通过 Web 管理界面在线编辑、版本管理。

核心功能：
  1. 提示词模板的 CRUD（增删改查）
  2. 从数据库动态加载提示词，DB 不可用时回退到默认值
  3. 管理后台认证（admin_users 表）

数据表：
  prompt_templates — 提示词模板（name, system_prompt, user_template, category, version）
  admin_users      — 管理员账号（username, password_hash, role）

使用方式：
  from prompt_manager import PromptManager, AuthManager
  pm = PromptManager()
  prompt = pm.get_prompt("classify")  # 返回 {"system": "...", "user_template": "..."}
"""

import json
import hashlib
import secrets
import time
from typing import Dict, List, Optional, Tuple

# ============================================================================
# 配置区 — MySQL 连接参数（与 memory_store.py 保持一致）
# ============================================================================

MYSQL_HOST = "192.168.200.128"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = "Root@2026"
MYSQL_DATABASE = "rag_agent"
MYSQL_CHARSET = "utf8mb4"


# ============================================================================
# 默认提示词 — 数据库不可用时的兜底方案
# ============================================================================

DEFAULT_PROMPTS = {
    "classify": {
        "display_name": "问题分类器",
        "description": "判断用户问题是简单查询、复杂查询还是闲聊，并进行上下文消解",
        "category": "路由",
        "version": 10,
        "system": (
            "你是问题分类器。判断用户问题的类型，输出 JSON：\n"
            '{{"type": "simple|complex|chitchat", "resolved": "消解后的完整问题"}}\n\n'
            "分类规则：\n"
            '- simple：单一事实查询（如"心跳间隔是多少？"）\n'
            '- complex：多维度复合问题（如"定位精度？几种方式？续航如何？"）\n'
            '- chitchat：闲聊/打招呼\n\n'
            "如果问题是追问（依赖上文），resolved 要补全为完整的独立问题。\n"
            "如果不依赖上文，resolved 等于原问题。\n"
            "只输出 JSON，不要其他文字。"
        ),
        "user_template": "对话历史:\n{history}\n\n当前问题: {query}",
    },
    "chitchat": {
        "display_name": "闲聊回答",
        "description": "对问候、感谢等非知识类问题直接用 LLM 回答",
        "category": "路由",
        "version": 10,
        "system": "你是友好的企业助手。简短自然地回答用户的话。",
        "user_template": "{query}",
    },
    "rewrite_first": {
        "display_name": "查询改写（首轮）",
        "description": "将用户问题改写为更适合向量检索的关键词",
        "category": "检索",
        "version": 10,
        "system": (
            "你是查询重写专家。将用户问题改写为 2-3 个更利于向量检索的搜索词。\n"
            "每个搜索词独占一行，不要编号，不要解释。"
        ),
        "user_template": "{query}",
    },
    "rewrite_retry": {
        "display_name": "查询改写（重试）",
        "description": "在之前检索结果不够相关时，换角度重新改写搜索词",
        "category": "检索",
        "version": 10,
        "system": (
            "之前的检索结果不够相关。请换一个角度改写搜索词，输出 2-3 个。\n"
            "每行一个，不要编号。"
        ),
        "user_template": "原问题: {query}\n\n已检索片段:\n{prev_text}",
    },
    "grade_docs": {
        "display_name": "文档相关性评分",
        "description": "判断每个检索到的文档片段是否与问题相关",
        "category": "检索",
        "version": 10,
        "system": (
            "判断每个文档片段是否与问题有直接关联，能够用来回答该问题。\n"
            "注意：仅包含问题中的个别词语（如'文档'、'格式'）但内容无法直接回答问题，不算关联。\n"
            "输出有直接关联的文档编号，逗号分隔，如: 0,2,3\n"
            "如果没有，输出: none"
        ),
        "user_template": "问题: {query}\n\n文档:\n{docs}",
    },
    "generate_answer": {
        "display_name": "生成答案",
        "description": "基于检索到的文档片段让 LLM 生成最终回答",
        "category": "生成",
        "version": 10,
        "system": (
            "你是企业文档问答助手。根据检索到的文档片段回答问题。\n\n"
            "严格要求：\n"
            "- 你只能根据下面提供的「检索到的文档」回答问题，禁止依赖常识或外部知识。\n"
            "- 如果文档中没有直接回答问题所需的信息，必须明确回答：未检索到相关内容。\n"
            "- 不要为了让回答看起来完整而拼凑、推断或编造文档中没有的信息。\n"
            "- 回答必须基于文档内容，不要编造\n"
            "- 如果信息不足，如实说明\n"
            "- 用中文回答，条理清晰"
        ),
        "user_template": "问题: {query}\n\n检索到的文档:\n{context}",
    },
    "planner_decompose": {
        "display_name": "任务拆解",
        "description": "将复杂问题拆解为 2-4 个独立可检索的子问题",
        "category": "规划",
        "version": 10,
        "system": (
            "你是任务规划器。将复杂问题拆解为 2-4 个独立的子问题，"
            "每个子问题可以独立检索回答。\n\n"
            '输出 JSON: {{"subtasks": [{{"id": 1, "task": "子问题"}}]}}\n'
            "只输出 JSON，不要其他文字。"
        ),
        "user_template": "{query}",
    },
    "planner_supplement": {
        "display_name": "补充拆解",
        "description": "审查不通过时，基于已有结果补充新的子任务",
        "category": "规划",
        "version": 10,
        "system": (
            "之前的回答不够充分。请补充 1-2 个新的子问题来填补信息缺口。\n\n"
            '输出 JSON: {{"subtasks": [{{"id": 1, "task": "新子问题"}}]}}\n'
            "只输出 JSON。"
        ),
        "user_template": "原问题: {query}\n\n已有结果:\n{existing_text}",
    },
    "reviewer_check": {
        "display_name": "审查把关",
        "description": "审查子任务研究结果是否充分回答了原始问题",
        "category": "规划",
        "version": 10,
        "system": (
            "你是严格的审查员。判断以上子任务结果是否充分回答了原始问题。\n"
            '只回答"充分"或"不充分"。'
        ),
        "user_template": "原始问题: {query}\n\n子任务结果:\n{results_text}",
    },
    "writer_compose": {
        "display_name": "汇总撰写",
        "description": "汇总所有子任务的研究结果，撰写结构化最终答案",
        "category": "生成",
        "version": 10,
        "system": (
            "你是技术文档撰写员。根据各子任务的研究结果，撰写一份完整的回答。\n\n"
            "严格要求：\n"
            "- 你只能根据各子任务的研究结果撰写回答，禁止依赖常识或外部知识。\n"
            "- 如果所有子任务结果都无法回答原始问题，必须明确回答：未检索到相关内容。\n"
            "- 不要为了让回答看起来完整而拼凑、推断或编造研究结果中没有的信息。\n"
            "- 整合所有子任务结果，按逻辑组织，可分点\n"
            "- 回答必须基于研究结果，不要编造\n"
            "- 用中文，条理清晰\n"
            "- 如果某方面信息不足，如实说明"
        ),
        "user_template": "原始问题: {query}\n\n各子任务研究结果:\n{results_text}",
    },
    "compress_history": {
        "display_name": "历史压缩",
        "description": "对话消息过多时，将旧消息压缩为摘要",
        "category": "记忆",
        "version": 10,
        "system": "将以下对话历史压缩为一段简短摘要（不超过100字），保留关键信息。",
        "user_template": "{history_text}",
    },
}


# ============================================================================
# PromptManager — 提示词 CRUD
# ============================================================================

class PromptManager:
    """
    提示词管理器

    职责：
      1. 从 MySQL 动态加载提示词模板
      2. 数据库不可用时回退到 DEFAULT_PROMPTS 内存字典
      3. 提供 CRUD 方法供管理后台调用

    使用模式：
      pm = PromptManager()
      prompt = pm.get_prompt("classify")
      # prompt = {"system": "...", "user_template": "...", "name": "classify", ...}
    """

    def __init__(self):
        self._pool = None
        self.available = False
        self._init_db()

    def _init_db(self):
        """初始化 MySQL 连接并创建表"""
        try:
            import pymysql
            from dbutils.pooled_db import PooledDB

            conn = pymysql.connect(
                host=MYSQL_HOST, port=MYSQL_PORT,
                user=MYSQL_USER, password=MYSQL_PASSWORD,
                charset=MYSQL_CHARSET,
            )
            cursor = conn.cursor()
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` "
                f"DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            conn.commit()
            cursor.close()
            conn.close()

            self._pool = PooledDB(
                creator=pymysql,
                maxconnections=5, mincached=1, maxcached=3,
                host=MYSQL_HOST, port=MYSQL_PORT,
                user=MYSQL_USER, password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE, charset=MYSQL_CHARSET,
                autocommit=True,
            )

            self._create_tables()
            self.available = True
            print(f"  [PromptManager] 连接成功: {MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}")

        except Exception as e:
            print(f"  [PromptManager] 连接失败，使用默认提示词: {e}")
            self._pool = None

    def _get_conn(self):
        return self._pool.connection() if self._pool else None

    def _create_tables(self):
        conn = self._get_conn()
        cursor = conn.cursor()

        # 表 1: 提示词模板
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_templates (
                id          BIGINT AUTO_INCREMENT PRIMARY KEY,
                name        VARCHAR(64)   NOT NULL UNIQUE COMMENT '唯一标识名',
                display_name VARCHAR(128) NOT NULL COMMENT '显示名称',
                description TEXT                    COMMENT '用途说明',
                category    VARCHAR(32)   NOT NULL DEFAULT 'general' COMMENT '分类',
                system_prompt TEXT        NOT NULL COMMENT '系统提示词',
                user_template TEXT        COMMENT '用户消息模板',
                version     INT           NOT NULL DEFAULT 1 COMMENT '版本号',
                is_active   TINYINT       NOT NULL DEFAULT 1 COMMENT '是否启用',
                created_at  DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at  DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_name (name),
                INDEX idx_category (category)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # 表 2: 管理员用户
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admin_users (
                id          BIGINT AUTO_INCREMENT PRIMARY KEY,
                username    VARCHAR(64)   NOT NULL UNIQUE COMMENT '用户名',
                password_hash VARCHAR(256) NOT NULL COMMENT '密码哈希(salt:hash)',
                display_name VARCHAR(128) DEFAULT '' COMMENT '显示名称',
                role        VARCHAR(32)   NOT NULL DEFAULT 'admin' COMMENT '角色',
                is_active   TINYINT       NOT NULL DEFAULT 1 COMMENT '是否启用',
                last_login  DATETIME      DEFAULT NULL COMMENT '最后登录时间',
                created_at  DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_username (username)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        cursor.close()
        conn.close()

    # ---- 提示词 CRUD ----

    def get_prompt(self, name: str) -> Dict:
        """
        按名称获取提示词。

        优先从 MySQL 读取，如果 DB 不可用或不存在，回退到 DEFAULT_PROMPTS。
        """
        if self.available:
            try:
                conn = self._get_conn()
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name, display_name, description, category, "
                    "system_prompt, user_template, version, is_active "
                    "FROM prompt_templates WHERE name=%s AND is_active=1",
                    (name,)
                )
                row = cursor.fetchone()
                cursor.close()
                conn.close()
                if row:
                    return {
                        "name": row[0],
                        "display_name": row[1],
                        "description": row[2] or "",
                        "category": row[3],
                        "system": row[4],
                        "user_template": row[5] or "",
                        "version": row[6],
                    }
            except Exception as e:
                print(f"  [PromptManager] get_prompt({name}) 失败: {e}")

        # 回退到默认值
        default = DEFAULT_PROMPTS.get(name, {})
        return {
            "name": name,
            "display_name": default.get("display_name", name),
            "description": default.get("description", ""),
            "category": default.get("category", "general"),
            "system": default.get("system", ""),
            "user_template": default.get("user_template", ""),
            "version": 0,
        }

    def list_prompts(self, category: str = None) -> List[Dict]:
        """列出所有提示词（或其分类子集）"""
        if self.available:
            try:
                conn = self._get_conn()
                cursor = conn.cursor()
                if category:
                    cursor.execute(
                        "SELECT name, display_name, description, category, "
                        "system_prompt, user_template, version, is_active "
                        "FROM prompt_templates WHERE category=%s ORDER BY name",
                        (category,)
                    )
                else:
                    cursor.execute(
                        "SELECT name, display_name, description, category, "
                        "system_prompt, user_template, version, is_active "
                        "FROM prompt_templates ORDER BY category, name"
                    )
                rows = cursor.fetchall()
                cursor.close()
                conn.close()
                return [
                    {
                        "name": r[0], "display_name": r[1],
                        "description": r[2] or "", "category": r[3],
                        "system": r[4], "user_template": r[5] or "",
                        "version": r[6], "is_active": bool(r[7]),
                    }
                    for r in rows
                ]
            except Exception as e:
                print(f"  [PromptManager] list_prompts 失败: {e}")

        # 回退
        return [
            {
                "name": k, "display_name": v["display_name"],
                "description": v.get("description", ""),
                "category": v.get("category", "general"),
                "system": v["system"],
                "user_template": v.get("user_template", ""),
                "version": 0, "is_active": True,
            }
            for k, v in DEFAULT_PROMPTS.items()
        ]

    def save_prompt(self, name: str, system: str, user_template: str = "",
                    display_name: str = "", description: str = "",
                    category: str = "general", version: int = None) -> bool:
        """
        保存或更新提示词。

        如果 name 已存在：
          - 若传入 version，则更新为该 version；
          - 若未传入 version，则递增 version。
        否则插入新记录，version 默认为 1。
        """
        if not self.available:
            return False
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            if version is None:
                # 未指定版本：插入为 1，存在则递增
                cursor.execute(
                    "INSERT INTO prompt_templates (name, display_name, description, "
                    "category, system_prompt, user_template, version) "
                    "VALUES (%s, %s, %s, %s, %s, %s, 1) "
                    "ON DUPLICATE KEY UPDATE "
                    "display_name=VALUES(display_name), "
                    "description=VALUES(description), "
                    "category=VALUES(category), "
                    "system_prompt=VALUES(system_prompt), "
                    "user_template=VALUES(user_template), "
                    "version=version+1",
                    (name, display_name or name, description or "",
                     category, system, user_template or "")
                )
            else:
                # 指定版本：强制覆盖为传入版本（用于工厂默认升级）
                cursor.execute(
                    "INSERT INTO prompt_templates (name, display_name, description, "
                    "category, system_prompt, user_template, version) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE "
                    "display_name=VALUES(display_name), "
                    "description=VALUES(description), "
                    "category=VALUES(category), "
                    "system_prompt=VALUES(system_prompt), "
                    "user_template=VALUES(user_template), "
                    "version=VALUES(version)",
                    (name, display_name or name, description or "",
                     category, system, user_template or "", version)
                )
            conn.commit()  # PooledDB autocommit=True 但显式 commit 更安全
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            print(f"  [PromptManager] save_prompt({name}) 失败: {e}")
            return False

    def delete_prompt(self, name: str) -> bool:
        """删除提示词（受保护的系统提示词不可删除）"""
        if not self.available:
            return False
        if name in DEFAULT_PROMPTS:
            # 系统内置提示词不允许删除，只能编辑
            return False
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM prompt_templates WHERE name=%s", (name,))
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            print(f"  [PromptManager] delete_prompt({name}) 失败: {e}")
            return False

    def set_active(self, name: str, active: bool) -> bool:
        """启用/禁用提示词"""
        if not self.available:
            return False
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE prompt_templates SET is_active=%s WHERE name=%s",
                (1 if active else 0, name)
            )
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            print(f"  [PromptManager] set_active({name}) 失败: {e}")
            return False

    def get_categories(self) -> List[str]:
        """获取所有分类"""
        if self.available:
            try:
                conn = self._get_conn()
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT DISTINCT category FROM prompt_templates WHERE is_active=1 ORDER BY category"
                )
                rows = cursor.fetchall()
                cursor.close()
                conn.close()
                return [r[0] for r in rows]
            except Exception:
                pass
        cats = set()
        for v in DEFAULT_PROMPTS.values():
            cats.add(v.get("category", "general"))
        return sorted(cats)

    def import_defaults(self) -> int:
        """
        将 DEFAULT_PROMPTS 中的所有条目导入 MySQL。

        逻辑：
          - 若数据库中不存在该提示词，直接插入。
          - 若数据库中已存在，但 DEFAULT_PROMPTS 中的 version 更高，则覆盖升级。
          - 若数据库中 version 大于等于默认 version，保留用户版本不覆盖。

        首次部署时调用，或在管理后台点击"恢复默认"按钮时调用。
        """
        if not self.available:
            return 0

        # 先批量读取当前数据库中的版本号
        current_versions: Dict[str, int] = {}
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT name, version FROM prompt_templates")
            for row in cursor.fetchall():
                current_versions[row[0]] = row[1]
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"  [PromptManager] import_defaults 读取版本失败: {e}")

        count = 0
        for name, data in DEFAULT_PROMPTS.items():
            default_version = data.get("version", 1)
            db_version = current_versions.get(name, 0)

            # 数据库版本更高或相等：保留用户编辑，跳过
            if db_version >= default_version:
                continue

            # 数据库版本更低或不存在：覆盖/插入为工厂默认版本
            if self.save_prompt(
                name=name,
                system=data["system"],
                user_template=data.get("user_template", ""),
                display_name=data["display_name"],
                description=data.get("description", ""),
                category=data.get("category", "general"),
                version=default_version,
            ):
                count += 1
        print(f"  [PromptManager] 导入/升级 {count} 个默认提示词")
        return count

    def format_user_message(self, user_template: str, **kwargs) -> str:
        """
        格式化用户消息模板。

        例如 user_template = "问题: {query}\n\n文档:\n{docs}"
        format_user_message(user_template, query="xxx", docs="yyy")
        → "问题: xxx\n\n文档:\nyyy"
        """
        if not user_template:
            return kwargs.get("query", "")
        try:
            return user_template.format(**kwargs)
        except KeyError as e:
            print(f"  [PromptManager] format_user_message 缺少参数: {e}")
            # 兜底：至少填入 query
            return f"问题: {kwargs.get('query', '')}"


# ============================================================================
# AuthManager — 用户认证
# ============================================================================

class AuthManager:
    """
    管理员认证管理器

    密码存储格式：salt:sha256(salt + password)
    """

    def __init__(self):
        self._pool = None
        self.available = False
        self._init_db()

    def _init_db(self):
        """初始化 MySQL 连接"""
        try:
            import pymysql
            from dbutils.pooled_db import PooledDB
            self._pool = PooledDB(
                creator=pymysql,
                maxconnections=3, mincached=1, maxcached=2,
                host=MYSQL_HOST, port=MYSQL_PORT,
                user=MYSQL_USER, password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE, charset=MYSQL_CHARSET,
                autocommit=True,
            )
            self.available = True
            self._ensure_default_admin()
        except Exception as e:
            print(f"  [AuthManager] 连接失败: {e}")
            self._pool = None

    def _get_conn(self):
        return self._pool.connection() if self._pool else None

    def _hash_password(self, password: str) -> str:
        """生成密码哈希：salt:hash格式"""
        salt = secrets.token_hex(16)
        h = hashlib.sha256((salt + password).encode()).hexdigest()
        return f"{salt}:{h}"

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """验证密码"""
        try:
            salt, h = password_hash.split(":", 1)
            return hashlib.sha256((salt + password).encode()).hexdigest() == h
        except Exception:
            return False

    def _ensure_default_admin(self):
        """确保至少有一个管理员账号"""
        if not self.available:
            return
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM admin_users WHERE is_active=1")
            count = cursor.fetchone()[0]
            if count == 0:
                h = self._hash_password("admin123")
                cursor.execute(
                    "INSERT INTO admin_users (username, password_hash, display_name, role) "
                    "VALUES (%s, %s, %s, %s)",
                    ("admin", h, "管理员", "admin")
                )
                conn.commit()
                print("  [AuthManager] 已创建默认管理员: admin / admin123")
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"  [AuthManager] 初始化默认管理员失败: {e}")

    def login(self, username: str, password: str) -> Optional[Dict]:
        """
        用户登录验证。

        返回: {"username": ..., "display_name": ..., "token": ...} 或 None
        """
        if not self.available:
            # 降级：本地硬编码认证
            if username == "admin" and password == "admin123":
                return {"username": "admin", "display_name": "管理员", "token": "local_fallback"}
            return None

        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, username, password_hash, display_name, role "
                "FROM admin_users WHERE username=%s AND is_active=1",
                (username,)
            )
            row = cursor.fetchone()
            cursor.close()

            if row and self._verify_password(password, row[2]):
                # 更新最后登录时间，生成 token
                token = secrets.token_hex(32)
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE admin_users SET last_login=NOW() WHERE id=%s",
                    (row[0],)
                )
                cursor.close()
                conn.close()
                return {
                    "username": row[1],
                    "display_name": row[3] or row[1],
                    "role": row[4],
                    "token": token,
                }

            conn.close()
            return None

        except Exception as e:
            print(f"  [AuthManager] login 失败: {e}")
            return None

    def verify_token(self, token: str) -> bool:
        """验证 token 是否有效（当前使用简单的会话 token）"""
        # 简化版：如果 auth manager 可用且 token 不为空，视为有效
        # 生产环境应使用 JWT 或 Redis session
        return bool(token and len(token) >= 32)

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """修改密码"""
        if not self.available:
            if username == "admin" and old_password == "admin123":
                return True
            return False

        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT password_hash FROM admin_users WHERE username=%s AND is_active=1",
                (username,)
            )
            row = cursor.fetchone()
            if not row or not self._verify_password(old_password, row[0]):
                cursor.close()
                conn.close()
                return False

            new_hash = self._hash_password(new_password)
            cursor.execute(
                "UPDATE admin_users SET password_hash=%s WHERE username=%s",
                (new_hash, username)
            )
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            print(f"  [AuthManager] change_password 失败: {e}")
            return False


# ============================================================================
# 全局单例
# ============================================================================

_prompt_manager: Optional[PromptManager] = None
_auth_manager: Optional[AuthManager] = None


def get_prompt_manager() -> PromptManager:
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def get_auth_manager() -> AuthManager:
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager
