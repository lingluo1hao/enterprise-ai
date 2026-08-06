-- ============================================================================
-- RAG 企业知识库 — MySQL 初始化脚本（可直接复制执行）
-- 说明：
--   1. 在 MySQL 客户端执行本文件即可建立 rag_agent 库及全部业务表；
--   2. 默认管理员 admin / admin123 若已存在则跳过（INSERT IGNORE）；
--   3. 已预置两个普通用户（role='user'）：reader / reader123、viewer / viewer123；
--   4. 已预置两个租户管理员（role='admin'，各属一个租户）：jm_admin(租户jm,密码jm123) / yh_admin(租户yh,密码yh123)；
--      普通用户只能访问 public 文档，无法查看 restricted（受限）文档；
--   5. 已预置超级管理员 superadmin(role='super_admin',密码Super@2026)：可跨 jm/yh/default 全部租户查看与上传文档；
--   6. 登录 token 不落 MySQL，而是写入 Redis（见下方 .env 配置），本脚本不含；
--   7. usage_log 表由 LLM 网关用本地 SQLite 自动创建，不在此处。
--   8. 所有表与字段均带中文注释（COMMENT），便于阅读与维护。
-- ============================================================================

CREATE DATABASE IF NOT EXISTS `rag_agent`
  DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE `rag_agent`;

-- ---------- 表 1: 提示词模板 ----------
CREATE TABLE IF NOT EXISTS `prompt_templates` (
  `id`            BIGINT AUTO_INCREMENT PRIMARY KEY                       COMMENT '自增主键',
  `name`          VARCHAR(64)   NOT NULL UNIQUE COMMENT '唯一标识名（代码中引用提示词用）',
  `display_name`  VARCHAR(128)  NOT NULL COMMENT '显示名称（给人看的）',
  `description`   TEXT                    COMMENT '用途说明',
  `category`      VARCHAR(32)   NOT NULL DEFAULT 'general' COMMENT '分类（如 general / 问答 / 摘要）',
  `system_prompt` TEXT          NOT NULL COMMENT '系统提示词（注入给 LLM 的 system 消息）',
  `user_template` TEXT          COMMENT '用户消息模板（支持占位符）',
  `version`       INT           NOT NULL DEFAULT 1 COMMENT '版本号',
  `is_active`     TINYINT       NOT NULL DEFAULT 1 COMMENT '是否启用（1=启用 / 0=禁用）',
  `created_at`    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at`    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  INDEX `idx_name` (`name`),
  INDEX `idx_category` (`category`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='提示词模板表：存储系统/用户提示词配置，供不同场景切换';

-- ---------- 表 2: 管理员 / 用户账号 ----------
-- role 取值：admin（特权，可看受限文档） / user（普通，仅 public 文档）
CREATE TABLE IF NOT EXISTS `admin_users` (
  `id`            BIGINT AUTO_INCREMENT PRIMARY KEY                       COMMENT '自增主键',
  `username`      VARCHAR(64)   NOT NULL UNIQUE COMMENT '用户名（登录名，唯一）',
  `password_hash` VARCHAR(256)  NOT NULL COMMENT '密码哈希（格式：salt:sha256(salt+password)）',
  `display_name`  VARCHAR(128)  DEFAULT '' COMMENT '显示名称',
  `role`          VARCHAR(32)   NOT NULL DEFAULT 'admin' COMMENT '账号角色：admin(特权) / user(普通) / super_admin(跨租户)',
  `tenant_id`     VARCHAR(64)   NOT NULL DEFAULT 'default' COMMENT '所属租户(多租户隔离，super_admin 可跨租户)',
  `is_active`     TINYINT       NOT NULL DEFAULT 1 COMMENT '是否启用（1=启用 / 0=禁用）',
  `last_login`    DATETIME      DEFAULT NULL COMMENT '最后登录时间',
  `created_at`    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  INDEX `idx_username` (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='账号表：管理员与普通用户，role 决定可见文档范围';

-- ---------- 表 3: 对话消息 ----------
-- user_id     : 外键 → admin_users.id（存数字ID，不冗余存用户名，符合第三范式）
-- speaker_role: 消息说话方，固定取值 user(人问)/assistant(AI答)/system(系统提示)
--               ——与 admin_users.role(账号权限等级) 是两回事，别混
-- session_id  : 会话隔离 key（格式 web:角色:账号），仅用于会话区分，不是外键
CREATE TABLE IF NOT EXISTS `chat_messages` (
  `id`            BIGINT AUTO_INCREMENT PRIMARY KEY                       COMMENT '自增主键',
  `user_id`       BIGINT        NOT NULL                                 COMMENT '用户ID（外键 → admin_users.id）',
  `session_id`    VARCHAR(128) NOT NULL                                  COMMENT '会话ID（格式：web:角色:账号）',
  `speaker_role`  VARCHAR(20)   NOT NULL
                  COMMENT '消息说话方：user=人问 / assistant=AI答 / system=系统提示',
  `content`       TEXT          NOT NULL                                 COMMENT '消息内容（问题或回答原文）',
  `msg_order`     INT           NOT NULL DEFAULT 0                       COMMENT '消息序号（同一会话内按此排序）',
  `created_at`    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP       COMMENT '创建时间',
  INDEX `idx_user_session` (`user_id`, `session_id`, `msg_order`),
  INDEX `idx_user_time` (`user_id`, `created_at`),
  CONSTRAINT `fk_chat_messages_user` FOREIGN KEY (`user_id`) REFERENCES `admin_users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='对话消息表：每条问答一行，按 user_id+session_id 隔离不同用户/会话';

-- ---------- 表 4: 断点快照（多层记忆 / 断点续作） ----------
CREATE TABLE IF NOT EXISTS `task_checkpoints` (
  `id`               BIGINT AUTO_INCREMENT PRIMARY KEY                   COMMENT '自增主键',
  `user_id`          BIGINT        NOT NULL                              COMMENT '用户ID（外键 → admin_users.id）',
  `thread_id`        VARCHAR(128) NOT NULL                               COMMENT '线程ID（= 任务ID task_id）',
  `session_id`       VARCHAR(128) NOT NULL                               COMMENT '会话ID',
  `node_name`        VARCHAR(64)   NOT NULL                              COMMENT '节点名（快照生成于哪个 LangGraph 节点执行之后）',
  `state_json`       LONGTEXT     NOT NULL                               COMMENT '节点状态 JSON（LangGraph AgentState 完整序列化，含消息历史）',
  `checkpoint_order` INT           NOT NULL DEFAULT 0                     COMMENT '快照序号（同一任务内取最新一条恢复）',
  `created_at`       DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP    COMMENT '创建时间',
  INDEX `idx_user_thread` (`user_id`, `thread_id`, `checkpoint_order`),
  CONSTRAINT `fk_task_checkpoints_user` FOREIGN KEY (`user_id`) REFERENCES `admin_users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='任务断点快照表：LangGraph 每个节点执行后保存状态，用于断点续作';

-- ---------- 表 5: 任务队列 ----------
CREATE TABLE IF NOT EXISTS `task_queue` (
  `id`          BIGINT AUTO_INCREMENT PRIMARY KEY                       COMMENT '自增主键',
  `user_id`     BIGINT        NOT NULL                                 COMMENT '用户ID（外键 → admin_users.id，发起任务的用户）',
  `task_id`     VARCHAR(128) NOT NULL UNIQUE                           COMMENT '任务ID（唯一，对应 thread_id）',
  `session_id`  VARCHAR(128) NOT NULL                                  COMMENT '会话ID',
  `query`       TEXT         NOT NULL                                  COMMENT '用户原始提问',
  `role`        VARCHAR(20)  NOT NULL DEFAULT 'user'                   COMMENT '发起账号角色（admin/user，决定可见文档范围）',
  `status`      VARCHAR(20)  NOT NULL DEFAULT 'pending'                COMMENT '任务状态：pending→running→completed/failed/interrupted',
  `answer`      TEXT                                                   COMMENT '任务最终答案（completed 时写入）',
  `error_msg`   TEXT                                                   COMMENT '错误信息（failed / interrupted 时记录）',
  `created_at`  DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP       COMMENT '创建时间',
  `updated_at`  DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  INDEX `idx_user_status` (`user_id`, `status`, `created_at`),
  CONSTRAINT `fk_task_queue_user` FOREIGN KEY (`user_id`) REFERENCES `admin_users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='任务队列表：异步/长任务的生命周期管理';

-- ---------- 表 6: 对话历史摘要（跨会语义召回） ----------
CREATE TABLE IF NOT EXISTS `chat_summaries` (
  `id`           BIGINT AUTO_INCREMENT PRIMARY KEY                     COMMENT '自增主键',
  `user_id`      BIGINT        NOT NULL                               COMMENT '用户ID（外键 → admin_users.id）',
  `session_id`   VARCHAR(128) NOT NULL                                COMMENT '会话ID',
  `summary`      TEXT         NOT NULL                                COMMENT '摘要文本（对话压缩产物，形如"历史摘要: ..."）',
  `covers_from`  BIGINT       NOT NULL DEFAULT 0                      COMMENT '覆盖起始的 chat_messages.id',
  `covers_to`    BIGINT       NOT NULL DEFAULT 0                      COMMENT '覆盖结束的 chat_messages.id',
  `msg_count`    INT          NOT NULL DEFAULT 0                      COMMENT '被压缩掉的原始消息条数',
  `importance`   TINYINT      NOT NULL DEFAULT 3                       COMMENT '重要性打分（1-5，跨会话召回时加权）',
  `embedding`    LONGBLOB     NULL                                     COMMENT '摘要向量（跨会话语义召回用）',
  `created_at`   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP     COMMENT '创建时间',
  INDEX `idx_user_session` (`user_id`, `session_id`, `covers_to`),
  INDEX `idx_importance` (`user_id`, `importance` DESC, `created_at` DESC),
  CONSTRAINT `fk_chat_summaries_user` FOREIGN KEY (`user_id`) REFERENCES `admin_users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='对话摘要表：压缩历史对话的产物，支持跨会话语义召回';

-- ============================================================================
-- 种子数据：账号（role 区分特权 / 普通）
-- 密码哈希算法：salt = 16字节十六进制；hash = sha256(salt + password)；格式 "salt:hash"
-- 如需重置某账号密码，请用项目内 prompt_manager._hash_password 重新生成后 UPDATE。
-- ============================================================================
INSERT IGNORE INTO `admin_users` (`username`, `password_hash`, `display_name`, `role`, `tenant_id`, `is_active`) VALUES
  ('admin',    '77163c0565c20104947c6d5a11cc1c07:63fff56895488aac04e0473dd4c5502eeb57a992e9ecbb483fc12c149e6bd78d', '系统管理员',   'admin', 'default', 1),
  ('reader',   '11576e1a65fe467d591628950dec7ad5:f9a10e3c305b4547e55bcc4a9fedb4200a520ce99467f4347614f5a4da0f21ab', '普通读者',     'user',  'default', 1),
  ('viewer',   '07a4b33c0ec699024ecd561dca87b6e5:e4516c5ffc42a3e616574b75bb436c96395c385ae7bca8c8ea119af031ac4221', '只读访客',     'user',  'default', 1),
  ('jm_admin', '0ffbc101953c00d273d561c109ee6296:bb9b6b0e3bff6d157739c582cb06731877d3a6d153ab31a633b920ae941048c4', 'jm租户管理员', 'admin', 'jm',     1),
  ('yh_admin', 'f5280115223307c9efa1de2153a68830:6de9b218cf902e5318379e462c07e64b12d3e281802ca76ca245173c465686b2', 'yh租户管理员', 'admin', 'yh',     1),
  ('superadmin', '2dc9a3c36925e6b9a0096b1abfce01e2:6ed5f0d33448a4d4863d21143b0a72b1207b44c6031d6778a1aee08fe18c4d26', '超级管理员', 'super_admin', 'default', 1);

-- 查看结果
SELECT username, display_name, role, is_active FROM `admin_users`;
