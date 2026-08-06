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
import os
import secrets
import time
from typing import Dict, List, Optional, Tuple

# ---- 加载 .env 文件（轻量实现，零依赖） ----
def _load_dotenv(dotenv_path=None):
    """解析 .env 文件并将未设置的变量注入 os.environ。"""
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
# 配置区 — MySQL 连接参数（与 memory_store.py 保持一致）
# ============================================================================

MYSQL_HOST = os.getenv("MYSQL_HOST", "192.168.200.128")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "Root@2026")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "rag_agent")
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
        "version": 15,
        "system": (
            "你是企业文档问答助手。根据检索到的文档片段回答问题。\n\n"
            "严格要求：\n"
            "- 你只能根据下面提供的「检索到的文档」回答问题，禁止依赖常识或外部知识编造文档中不存在的事实。\n"
            "- 文档信息可能分散在多个片段中（例如分别提及 GPS、LBS、WiFi 定位，或只在报文结构里出现相关字段），\n"
            "  应将这些片段综合归纳后回答，不要因为单个片段没有把答案完整列出一次就判为“未检索到”。\n"
            "- 仅当检索到的文档确实完全不包含回答所需的任何相关信息时，才说明：未检索到相关内容。\n"
            "- 可以基于多个片段合理归纳总结，但不得编造文档中不存在的数据、参数或结论。\n"
            "- 用中文回答，条理清晰\n"
            "- 引用具体文档时，使用文档片段开头标注的文件名（如 `JM-S509 学生证产品客户指令表_V1.0.pdf`），不要使用「文档1/文档2」这种笼统标签\n"
            "- 如果检索片段中包含 `[[FIG:assets/figures/...]]` 占位符（说明该页有真实图示或表格截图），请在回答里**原样保留**该占位符（按行放置即可），前端会自动渲染为图片。判断说明文字：若占位符路径中包含 `table_`（例如 `table_p053_1.png`），说明这是表格截图，请写「参见第 X 页表格图」；否则才是普通图示/流程图，可写「参见第 X 页通信流程图」。不要删除占位符、不要改写路径，尤其不要把路径改写成 `.pdf` 之类不存在的文件。\n"
            "- 重要：只要回答中存在 `[[FIG:...]]` 占位符，就说明文档中**确有对应图示/表格截图**。请直接写「参见下方图示/表格图」并保留占位符，**绝对禁止**写出「文档中未直接提供…流程图/表格」「未检索到相关图示」等否认语句——图就在占位符里，不要自我否认。也**不要**凭空把无关协议字段（如 SMS/RFID 指令）拼凑成「流程图概述」来填充。\n"
            "- 如果检索片段中包含 Markdown 表格（以 `|` 和分隔线 `|---|---|` 构成），请在回答中**保持表格形式输出**，不要把它压缩成纯文字罗列。表格内容与描述文字可并用，确保字段、长度、取值等一一对应、清晰可读。若同一段落中还存在 `[[FIG:...table_...png]]` 表格图片占位符，请一并保留，并用文字说明「字段详情参见下方表格图」\n"
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
        "version": 12,
        "system": (
            "你是技术文档撰写员。根据各子任务的研究结果，撰写一份完整的回答。\n\n"
            "严格要求：\n"
            "- 你只能根据各子任务的研究结果撰写回答，禁止依赖常识或外部知识。\n"
            "- 如果所有子任务结果都无法回答原始问题，必须明确回答：未检索到相关内容。\n"
            "- 不要为了让回答看起来完整而拼凑、推断或编造研究结果中没有的信息。\n"
            "- 整合所有子任务结果，按逻辑组织，可分点\n"
            "- 回答必须基于研究结果，不要编造\n"
            "- 用中文，条理清晰\n"
            "- 如果某方面信息不足，如实说明\n"
            "- 如果研究结果中包含 Markdown 表格（字段、长度、取值等以 `|` 分隔），请在最终回答中**保持表格形式输出**，不要把它压缩成纯文字列表，确保一一对应、清晰可读。\n"
            "- 如果研究结果中包含 `[[FIG:assets/figures/...]]` 占位符（含 `table_...png` 表格截图），请**原样保留**该占位符，前端会自动渲染为图片；附一句说明如「字段详情参见下方表格图」。不要删除、不要改写路径。"
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
                role        VARCHAR(32)   NOT NULL DEFAULT 'admin' COMMENT '角色(admin/user/super_admin)',
                tenant_id   VARCHAR(64)   NOT NULL DEFAULT 'default' COMMENT '所属租户(多租户隔离)',
                is_active   TINYINT       NOT NULL DEFAULT 1 COMMENT '是否启用',
                last_login  DATETIME      DEFAULT NULL COMMENT '最后登录时间',
                created_at  DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_username (username)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        # 老库迁移：缺 tenant_id 列则补齐（不丢数据）
        try:
            cursor.execute("SELECT tenant_id FROM admin_users LIMIT 1")
        except Exception:
            cursor.execute(
                "ALTER TABLE admin_users ADD COLUMN tenant_id "
                "VARCHAR(64) NOT NULL DEFAULT 'default' "
                "COMMENT '所属租户(多租户隔离)' AFTER role"
            )

        # 同步默认提示词：当代码中 DEFAULT_PROMPTS 的 version 高于数据库时自动升级
        try:
            for name, cfg in DEFAULT_PROMPTS.items():
                cursor.execute(
                    "SELECT version FROM prompt_templates WHERE name=%s", (name,)
                )
                row = cursor.fetchone()
                db_version = row[0] if row else 0
                code_version = cfg.get("version", 1)
                if db_version < code_version:
                    cursor.execute(
                        "INSERT INTO prompt_templates "
                        "(name, display_name, description, category, system_prompt, user_template, version) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                        "ON DUPLICATE KEY UPDATE "
                        "display_name=VALUES(display_name), "
                        "description=VALUES(description), "
                        "category=VALUES(category), "
                        "system_prompt=VALUES(system_prompt), "
                        "user_template=VALUES(user_template), "
                        "version=VALUES(version)",
                        (name, cfg["display_name"], cfg.get("description", ""),
                         cfg.get("category", "general"), cfg["system"],
                         cfg.get("user_template", ""), code_version)
                    )
                    print(f"  [PromptManager] 同步提示词 {name}: {db_version} -> {code_version}")
        except Exception as e:
            print(f"  [PromptManager] 默认提示词同步失败(忽略): {e}")

        cursor.close()
        conn.commit()
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
    认证管理器（管理员 + 普通用户通用）

    密码存储格式：salt:sha256(salt + password)
    Token 存储：登录成功后写入 Redis（auth:token:{token} → session JSON），带 TTL。
      - admin 与 user 各自生成独立随机 token，且 token 内携带 role，
        普通用户 token 无法调用需要 admin 角色的接口。
      - Redis 不可用时回退到进程内字典（仅单实例有效，会打印告警）。
    """

    # Token 有效期（秒），可被环境变量 AUTH_TOKEN_TTL 覆盖
    TOKEN_TTL = int(os.environ.get("AUTH_TOKEN_TTL", "604800"))  # 默认 7 天

    def __init__(self):
        self._pool = None
        self.available = False
        self._redis = None
        self._token_store = {}  # Redis 不可用时的回退存储
        self._init_db()
        self._init_redis()

    def _init_redis(self):
        """初始化 Redis 连接（用于 token 存储）"""
        try:
            import redis as redis_pkg
            self._redis = redis_pkg.Redis(
                host=os.environ.get("REDIS_HOST", "127.0.0.1"),
                port=int(os.environ.get("REDIS_PORT", "6379")),
                password=os.environ.get("REDIS_PASSWORD") or None,
                db=int(os.environ.get("REDIS_DB", "0")),
                socket_connect_timeout=5, socket_timeout=5,
                decode_responses=True,
            )
            self._redis.ping()
            print(f"  [AuthManager] Redis 已连接（token 存储）")
        except Exception as e:
            self._redis = None
            self._token_store = {}
            print(f"  [AuthManager] ⚠ Redis 不可用，token 回退到内存存储: {e}")

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

    # ---- Token 存储（Redis 优先，内存回退）----
    # 索引：auth:user_tokens:{username} (Set) → 该账号当前在用的所有 token
    # 用于：登录时清空同账号旧 token、登出时按 username 定位
    @staticmethod
    def _user_index_key(username: str) -> str:
        return f"auth:user_tokens:{username}" if username else ""

    def _save_token(self, token: str, session: Dict):
        import json as _json
        payload = _json.dumps(session)
        username = session.get("username") if isinstance(session, dict) else None
        if self._redis is not None:
            pipe = self._redis.pipeline()
            pipe.setex(f"auth:token:{token}", self.TOKEN_TTL, payload)
            if username:
                idx = self._user_index_key(username)
                pipe.sadd(idx, token)
                pipe.expire(idx, self.TOKEN_TTL)
            pipe.execute()
        else:
            self._token_store[token] = (time.time() + self.TOKEN_TTL, payload)

    def _load_token(self, token: str) -> Optional[Dict]:
        import json as _json
        if self._redis is not None:
            raw = self._redis.get(f"auth:token:{token}")
            if not raw:
                return None
            try:
                return _json.loads(raw)
            except Exception:
                return None
        # 内存回退
        item = self._token_store.get(token)
        if not item:
            return None
        exp, payload = item
        if exp < time.time():
            self._token_store.pop(token, None)
            return None
        try:
            return _json.loads(payload)
        except Exception:
            return None

    def _evict_user_tokens(self, username: str):
        """登录前调用：清空该 username 在 Redis 里的所有旧 token（避免一账号多 token 堆积）。"""
        if not username or self._redis is None:
            return
        idx = self._user_index_key(username)
        try:
            old_tokens = self._redis.smembers(idx) or set()
            if old_tokens:
                pipe = self._redis.pipeline()
                for t in old_tokens:
                    pipe.delete(f"auth:token:{t}")
                pipe.delete(idx)
                pipe.execute()
        except Exception as e:
            print(f"  [AuthManager] ⚠ 清旧 token 失败（忽略）: {e}")

    def _delete_token(self, token: str, username: str = None):
        if self._redis is not None:
            pipe = self._redis.pipeline()
            pipe.delete(f"auth:token:{token}")
            if username:
                pipe.srem(self._user_index_key(username), token)
            pipe.execute()
        else:
            self._token_store.pop(token, None)

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

    def create_user(self, username: str, password: str, display_name: str = "",
                   role: str = "user", tenant_id: str = "default") -> "tuple[bool, str]":
        """创建新用户（管理后台「新增用户」调用）。

        成功返回 (True, "")；失败返回 (False, 错误原因)。
        调用方（rag_web_server）负责按操作者角色约束 tenant_id / role，
        本方法只做唯一性检查与落地写入。
        """
        if not self.available:
            return False, "认证后端不可用（数据库未连接）"
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM admin_users WHERE username=%s", (username,))
            if cursor.fetchone():
                cursor.close(); conn.close()
                return False, "用户名已存在"
            pw_hash = self._hash_password(password)
            cursor.execute(
                "INSERT INTO admin_users (username, password_hash, display_name, role, tenant_id) "
                "VALUES (%s, %s, %s, %s, %s)",
                (username, pw_hash, display_name or username, role, tenant_id))
            conn.commit()
            cursor.close(); conn.close()
            return True, ""
        except Exception as e:
            return False, str(e)

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
        用户登录验证。成功后把 token 写入 Redis（带 TTL），token 内携带 role。

        同账号已有 token 时，全部先清掉再发新 token（防 1 账号多 token 堆积）。

        返回: {"username", "display_name", "role", "token"} 或 None
        """
        if not self.available:
            # 降级：本地硬编码认证（仅 admin）
            if username == "admin" and password == "admin123":
                token = secrets.token_hex(32)
                session = {"username": "admin", "display_name": "管理员",
                           "role": "admin", "user_id": 0, "tenant_id": "default"}
                self._save_token(token, session)
                return {"username": "admin", "display_name": "管理员",
                        "role": "admin", "tenant_id": "default", "token": token}
            return None

        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, username, password_hash, display_name, role, tenant_id "
                "FROM admin_users WHERE username=%s AND is_active=1",
                (username,)
            )
            row = cursor.fetchone()
            cursor.close()

            if row and self._verify_password(password, row[2]):
                # 同账号已有 token 时全部清掉（防 1 账号多 token 堆积）
                self._evict_user_tokens(row[1])
                # 校验通过：生成独立随机 token，写入 Redis（admin/user 各自独立）
                token = secrets.token_hex(32)
                session = {
                    "username": row[1],
                    "display_name": row[3] or row[1],
                    "role": row[4] or "user",
                    "user_id": row[0],
                    "tenant_id": row[5] or "default",
                }
                self._save_token(token, session)
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE admin_users SET last_login=NOW() WHERE id=%s",
                        (row[0],)
                    )
                    cursor.close()
                except Exception:
                    pass
                conn.close()
                return {
                    "username": session["username"],
                    "display_name": session["display_name"],
                    "role": session["role"],
                    "tenant_id": session["tenant_id"],
                    "token": token,
                }

            conn.close()
            return None

        except Exception as e:
            print(f"  [AuthManager] login 失败: {e}")
            return None

    def verify_token(self, token: str) -> Optional[Dict]:
        """
        验证 token 是否有效（查 Redis / 内存回退）。

        返回 session 字典 {"username","display_name","role","user_id"} 或 None。
        """
        if not token or len(token) < 32:
            return None
        return self._load_token(token)

    def logout(self, token: str):
        """注销：删除 token 存储（Redis 或内存回退），并从用户名反向索引中移除。"""
        if not token:
            return
        # 拿 username 以便从反向索引 SREM（避免该 username 的索引长期悬挂）
        session = self._load_token(token)
        username = session.get("username") if isinstance(session, dict) else None
        self._delete_token(token, username=username)

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
