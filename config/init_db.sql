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
  `used_playbook_pk` VARCHAR(64) NULL                                  COMMENT 'P1-7 L3：本次问答命中的 playbook pk，供 /api/feedback 回灌 reinforce_feedback',
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

-- ---------- 表 7: 用户反馈（答案点赞 / 点踩 / 文字） ----------
-- 这是 bad case 闭环的「源头」：真实用户在生产环境踩到的失败，比挖掘 trace 更珍贵。
-- 一条 feedback 可关联 task_id，回溯完整 task_checkpoints 全链路 trace；
-- 经 triage 自动归类根因（root_cause = R1..R8）后，可转成 bad_cases 驱动自进化。
CREATE TABLE IF NOT EXISTS `qa_feedback` (
  `id`           BIGINT AUTO_INCREMENT PRIMARY KEY                     COMMENT '自增主键',
  `task_id`      VARCHAR(64)  NULL                                     COMMENT '关联任务ID（→ task_queue.id），回溯全链路trace',
  `user_id`      BIGINT        NULL                                    COMMENT '反馈用户ID（→ admin_users.id）',
  `tenant_id`    VARCHAR(32)   NULL                                    COMMENT '租户ID（多租户隔离）',
  `session_id`   VARCHAR(128)  NULL                                    COMMENT '会话ID',
  `query`        TEXT          NOT NULL                                COMMENT '用户原始问题',
  `answer`       TEXT          NULL                                    COMMENT '系统给出的答案（落库快照，避免task被清后丢失）',
  `rating`       TINYINT       NOT NULL DEFAULT 0                       COMMENT '评分：-1=踩(差) / 0=无 / 1=赞(好)',
  `feedback_text` TEXT         NULL                                    COMMENT '用户自由文字反馈',
  `root_cause`   VARCHAR(8)    NULL                                    COMMENT 'triage 自动归类根因：R1..R8（NULL=未分类）',
  `created_at`   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP     COMMENT '反馈时间',
  INDEX `idx_tenant_created` (`tenant_id`, `created_at` DESC),
  INDEX `idx_rating` (`rating`, `created_at` DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户反馈表：答案点赞/点踩/文字，bad case 闭环源头';

-- ---------- 表 8: Bad Case 库（驱动自进化） ----------
-- 沉淀「失败样本 + 根因 + 修复记录」，是强化自进化（evolution.py）的输入。
-- 来源：用户反馈转（source=feedback）/ 评测失败（source=eval）/ 人工录入（source=manual）。
-- status 流转：open → triaged → fixed(wontfix)。resolved_by 记录修复它的 evolution patch / 评测 run。
CREATE TABLE IF NOT EXISTS `bad_cases` (
  `id`           BIGINT AUTO_INCREMENT PRIMARY KEY                     COMMENT '自增主键',
  `source`       VARCHAR(16)   NOT NULL DEFAULT 'feedback'             COMMENT '来源：feedback / eval / manual',
  `suite`        VARCHAR(16)   NOT NULL DEFAULT 'answer'              COMMENT '所属评测层：retrieval / answer',
  `case_id`      VARCHAR(64)   NULL                                    COMMENT '关联黄金集case_id（若是评测失败转来）',
  `query`        TEXT          NOT NULL                                COMMENT '问题',
  `answer`       TEXT          NULL                                    COMMENT '系统故障答案（快照）',
  `expected`     TEXT          NULL                                    COMMENT '期望答案 / 期望召回规范（参考）',
  `root_cause`   VARCHAR(8)    NULL                                    COMMENT '根因分类：R1..R8',
  `diagnosis`    TEXT          NULL                                    COMMENT 'triage 诊断理由 + 建议修复方向',
  `status`       VARCHAR(16)   NOT NULL DEFAULT 'open'                 COMMENT '状态：open / triaged / fixed / wontfix',
  `resolved_by`  VARCHAR(64)   NULL                                    COMMENT '修复它的 evolution patch id / 评测 run_id',
  `created_at`   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP     COMMENT '入库时间',
  `resolved_at`  DATETIME      NULL                                    COMMENT '修复时间',
  INDEX `idx_status` (`status`, `root_cause`),
  INDEX `idx_source` (`source`, `created_at` DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Bad Case 库：失败样本+根因+修复记录，驱动强化自进化';

-- ============================================================================
-- 工厂业务系统（数字员工数据面，设计见 docs/guides/factory_business_system_design.md）
-- 六域 25 表：主数据 / 销售 / 库存 / 采购 / 设备维修 / 平台(biz_roles)
-- 公共约定：
--   * 所有表带 tenant_id（L1 租户隔离）+ created_by（数字员工代表谁操作，L2 own 依据）；
--   * 逻辑外键（索引 + 应用层保证），不建物理 CONSTRAINT；
--   * 单号插入后按 id 回填（SO-/SD-/TRK-/SR-/RC-/IS-/PO-/RO- 前缀）。
-- ============================================================================

-- ---------- 表 9: 业务角色（租户级自建角色，行级 ACL 之源） ----------
CREATE TABLE IF NOT EXISTS `biz_roles` (
  `id`         BIGINT AUTO_INCREMENT PRIMARY KEY                       COMMENT '自增主键',
  `tenant_id`  VARCHAR(64)   NOT NULL DEFAULT 'default'                COMMENT '租户（角色按租户自定义）',
  `role_key`   VARCHAR(64)   NOT NULL                                 COMMENT '角色标识（如 sales_user / workshop_director）',
  `name`       VARCHAR(128)  NOT NULL                                 COMMENT '角色显示名（销售跟单员 / 车间主任）',
  `level`      TINYINT       NOT NULL DEFAULT 1                       COMMENT '审批等级：1=普通 / 2=经理级 / 3=超管级（分级审批依据）',
  `resources`  JSON          NOT NULL                                 COMMENT '可见资源数组（如 ["sales_order","product"]）',
  `scope`      VARCHAR(16)   NOT NULL DEFAULT 'own'                    COMMENT '可见范围：own=仅自己创建 / tenant=本租户 / all=跨租户',
  `ops`        JSON          NOT NULL                                 COMMENT '可执行操作数组（create/query/update/request/approve）',
  `is_system`  TINYINT       NOT NULL DEFAULT 0                       COMMENT '内置种子角色=1（不可删）',
  `created_by` VARCHAR(64)   NULL                                     COMMENT '创建人',
  `created_at` DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP       COMMENT '创建时间',
  `updated_at` DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  UNIQUE KEY `uk_tenant_role` (`tenant_id`, `role_key`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='业务角色表：租户自建角色（行级 ACL + 审批等级），工厂不限三类角色';

-- ---------- 表 10: 客户主数据 ----------
CREATE TABLE IF NOT EXISTS `customers` (
  `id`           BIGINT AUTO_INCREMENT PRIMARY KEY                     COMMENT '自增主键',
  `tenant_id`    VARCHAR(64)   NOT NULL DEFAULT 'default'              COMMENT '租户',
  `customer_code` VARCHAR(64) NOT NULL                                 COMMENT '客户编码（租户内唯一）',
  `name`         VARCHAR(128)  NOT NULL                                COMMENT '客户名称',
  `contact`      VARCHAR(64)   NULL                                    COMMENT '联系人',
  `phone`        VARCHAR(32)   NULL                                    COMMENT '电话',
  `address`      VARCHAR(256)  NULL                                    COMMENT '地址',
  `credit_limit` DECIMAL(14,2) NOT NULL DEFAULT 0                      COMMENT '信用额度（confirm 时校验超信用拒单）',
  `is_active`    TINYINT       NOT NULL DEFAULT 1                      COMMENT '是否启用',
  `created_by`   VARCHAR(64)   NULL                                    COMMENT '创建人',
  `created_at`   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP      COMMENT '创建时间',
  `updated_at`   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  UNIQUE KEY `uk_tenant_code` (`tenant_id`, `customer_code`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客户主数据表';

-- ---------- 表 11: 供应商主数据 ----------
CREATE TABLE IF NOT EXISTS `suppliers` (
  `id`           BIGINT AUTO_INCREMENT PRIMARY KEY                     COMMENT '自增主键',
  `tenant_id`    VARCHAR(64)   NOT NULL DEFAULT 'default'              COMMENT '租户',
  `supplier_code` VARCHAR(64) NOT NULL                                 COMMENT '供应商编码（租户内唯一）',
  `name`         VARCHAR(128)  NOT NULL                                COMMENT '供应商名称',
  `contact`      VARCHAR(64)   NULL                                    COMMENT '联系人',
  `phone`        VARCHAR(32)   NULL                                    COMMENT '电话',
  `is_active`    TINYINT       NOT NULL DEFAULT 1                      COMMENT '是否启用',
  `created_by`   VARCHAR(64)   NULL                                    COMMENT '创建人',
  `created_at`   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP      COMMENT '创建时间',
  `updated_at`   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  UNIQUE KEY `uk_tenant_code` (`tenant_id`, `supplier_code`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='供应商主数据表';

-- ---------- 表 12: 产品与备件（统一主数据） ----------
CREATE TABLE IF NOT EXISTS `products` (
  `id`           BIGINT AUTO_INCREMENT PRIMARY KEY                     COMMENT '自增主键',
  `tenant_id`    VARCHAR(64)   NOT NULL DEFAULT 'default'              COMMENT '租户',
  `product_code` VARCHAR(64)   NOT NULL                                 COMMENT '产品编码（租户内唯一）',
  `name`         VARCHAR(128)  NOT NULL                                 COMMENT '产品名称',
  `spec`         VARCHAR(128)  NULL                                    COMMENT '规格型号',
  `category`     VARCHAR(16)   NOT NULL DEFAULT 'finished'             COMMENT '类别：finished=成品 / spare=备件 / material=原料',
  `unit`         VARCHAR(16)   NOT NULL DEFAULT '件'                   COMMENT '计量单位',
  `unit_price`   DECIMAL(12,2) NOT NULL DEFAULT 0                      COMMENT '标准售价（销售/领料计价基准）',
  `cost_price`   DECIMAL(12,2) NULL                                    COMMENT '参考进价（采购计价基准）',
  `safety_stock` INT           NOT NULL DEFAULT 0                      COMMENT '安全库存（低于此值触发低库存预警/建议采购）',
  `is_active`    TINYINT       NOT NULL DEFAULT 1                      COMMENT '是否启用',
  `created_by`   VARCHAR(64)   NULL                                    COMMENT '创建人',
  `created_at`   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP      COMMENT '创建时间',
  `updated_at`   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  UNIQUE KEY `uk_tenant_code` (`tenant_id`, `product_code`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='产品与备件统一主数据表';

-- ---------- 表 13: 设备台账（CMMS asset register） ----------
CREATE TABLE IF NOT EXISTS `equipment` (
  `id`             BIGINT AUTO_INCREMENT PRIMARY KEY                   COMMENT '自增主键',
  `tenant_id`      VARCHAR(64)   NOT NULL DEFAULT 'default'            COMMENT '租户',
  `equipment_code` VARCHAR(64)   NOT NULL                               COMMENT '设备编码（租户内唯一）',
  `name`           VARCHAR(128)  NOT NULL                               COMMENT '设备名称',
  `location`       VARCHAR(128)  NULL                                  COMMENT '位置（车间/产线/工位层级路径，如 1车间/S3线/贴标工位）',
  `model`          VARCHAR(128)  NULL                                  COMMENT '型号',
  `manufacturer`   VARCHAR(128)  NULL                                  COMMENT '制造商',
  `installed_date` DATE          NULL                                  COMMENT '投运日期',
  `warranty_until` DATE          NULL                                  COMMENT '保修截止（工单自动判保内/保外的依据）',
  `criticality`    VARCHAR(4)    NOT NULL DEFAULT 'B'                  COMMENT '关键度：A=关键（故障自动升 urgent）/ B=普通 / C=辅助',
  `status`         VARCHAR(16)   NOT NULL DEFAULT 'running'            COMMENT '设备状态：running / down / retired',
  `is_active`      TINYINT       NOT NULL DEFAULT 1                    COMMENT '是否启用（主数据通用列）',
  `owner_dept`     VARCHAR(64)   NULL                                  COMMENT '归属部门',
  `created_by`     VARCHAR(64)   NULL                                  COMMENT '创建人',
  `created_at`     DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP    COMMENT '创建时间',
  `updated_at`     DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  UNIQUE KEY `uk_tenant_code` (`tenant_id`, `equipment_code`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='设备台账表（CMMS 核心：保修判定+关键度+层级位置）';

-- ---------- 表 14: 维修工程师 ----------
CREATE TABLE IF NOT EXISTS `engineers` (
  `id`         BIGINT AUTO_INCREMENT PRIMARY KEY                       COMMENT '自增主键',
  `tenant_id`  VARCHAR(64)   NOT NULL DEFAULT 'default'                COMMENT '租户',
  `name`       VARCHAR(64)   NOT NULL                                 COMMENT '姓名',
  `skill`      VARCHAR(32)   NOT NULL DEFAULT 'mechanical'            COMMENT '技能专长：electrical=电气 / mechanical=机械 / software=软件（派单匹配）',
  `phone`      VARCHAR(32)   NULL                                     COMMENT '电话',
  `is_active`  TINYINT       NOT NULL DEFAULT 1                       COMMENT '是否在岗',
  `created_by` VARCHAR(64)   NULL                                     COMMENT '创建人',
  `created_at` DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP       COMMENT '创建时间',
  UNIQUE KEY `uk_tenant_name` (`tenant_id`, `name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='维修工程师表（派单按 skill 匹配）';

-- ---------- 表 15: 仓库 ----------
CREATE TABLE IF NOT EXISTS `warehouses` (
  `id`            BIGINT AUTO_INCREMENT PRIMARY KEY                    COMMENT '自增主键',
  `tenant_id`     VARCHAR(64)   NOT NULL DEFAULT 'default'             COMMENT '租户',
  `warehouse_code` VARCHAR(32) NOT NULL                                COMMENT '仓库编码（租户内唯一）',
  `name`          VARCHAR(128)  NOT NULL                                COMMENT '仓库名称',
  `location`      VARCHAR(128)  NULL                                   COMMENT '位置',
  `is_active`     TINYINT       NOT NULL DEFAULT 1                     COMMENT '是否启用',
  `created_by`    VARCHAR(64)   NULL                                   COMMENT '创建人',
  `created_at`    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP     COMMENT '创建时间',
  UNIQUE KEY `uk_tenant_code` (`tenant_id`, `warehouse_code`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='仓库表';

-- ---------- 表 16: 库存余额（仓库 × 产品） ----------
CREATE TABLE IF NOT EXISTS `inventory` (
  `id`            BIGINT AUTO_INCREMENT PRIMARY KEY                    COMMENT '自增主键',
  `tenant_id`     VARCHAR(64)   NOT NULL DEFAULT 'default'             COMMENT '租户',
  `warehouse_id`  BIGINT        NOT NULL                               COMMENT '仓库ID（→ warehouses.id）',
  `product_id`    BIGINT        NOT NULL                               COMMENT '产品ID（→ products.id）',
  `stock_qty`     INT           NOT NULL DEFAULT 0                     COMMENT '实物在库数量',
  `reserved_qty`  INT           NOT NULL DEFAULT 0                     COMMENT '已占用数量（已确认未出库的订单/工单）',
  `updated_at`    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  UNIQUE KEY `uk_wh_prod` (`tenant_id`, `warehouse_id`, `product_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='库存余额表：可用量 = stock_qty - reserved_qty（余额是快照，对账以流水为准）';

-- ---------- 表 17: 库存流水（每次变动一行，审计之源） ----------
CREATE TABLE IF NOT EXISTS `inventory_transactions` (
  `id`           BIGINT AUTO_INCREMENT PRIMARY KEY                     COMMENT '自增主键',
  `tenant_id`    VARCHAR(64)   NOT NULL DEFAULT 'default'              COMMENT '租户',
  `txn_type`     VARCHAR(16)   NOT NULL                                 COMMENT '类型：PURCHASE_IN采购入/SALE_OUT销售出/RESERVE占用/RELEASE释放/RETURN_IN退货回补/PARTS_OUT维修领料/ADJUST盘点调',
  `warehouse_id` BIGINT        NOT NULL                                 COMMENT '仓库ID',
  `product_id`   BIGINT        NOT NULL                                 COMMENT '产品ID',
  `qty`          INT           NOT NULL                                 COMMENT '变动数量（正=入，负=出）',
  `balance_after` INT          NOT NULL                                 COMMENT '变动后库存快照（stock_qty）',
  `ref_type`     VARCHAR(32)   NULL                                     COMMENT '关联单据类型：stock_receipt/sales_delivery/repair_order/purchase_order/adjust',
  `ref_no`       VARCHAR(32)   NULL                                     COMMENT '关联单号（RC-/SD-/RO-/PO-…）',
  `operator`     VARCHAR(64)   NULL                                     COMMENT '操作者（人工或 digital_employee:岗位）',
  `remark`       VARCHAR(256)  NULL                                     COMMENT '备注',
  `created_at`   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP      COMMENT '时间',
  INDEX `idx_tenant_wh_prod` (`tenant_id`, `warehouse_id`, `product_id`, `id` DESC),
  INDEX `idx_ref` (`ref_type`, `ref_no`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='库存流水表：只增不改，库存对账唯一可信来源';

-- ---------- 表 18: 入库单（表头） ----------
CREATE TABLE IF NOT EXISTS `stock_receipts` (
  `id`           BIGINT AUTO_INCREMENT PRIMARY KEY                     COMMENT '自增主键',
  `tenant_id`    VARCHAR(64)   NOT NULL DEFAULT 'default'              COMMENT '租户',
  `receipt_no`   VARCHAR(32)   NOT NULL                                 COMMENT '入库单号（RC-日期-序号）',
  `ref_type`     VARCHAR(16)   NOT NULL                                 COMMENT '来源：purchase=采购收货 / return=退货入库 / adjust=盘点调入',
  `ref_id`       BIGINT        NULL                                     COMMENT '来源单据ID（采购单/退货单）',
  `warehouse_id` BIGINT        NOT NULL                                 COMMENT '入库仓库',
  `supplier_id`  BIGINT        NULL                                     COMMENT '供应商（采购入库时）',
  `status`       VARCHAR(16)   NOT NULL DEFAULT 'draft'                 COMMENT '状态：draft → received（received 时才加库存）',
  `remark`       VARCHAR(256)  NULL                                     COMMENT '备注',
  `created_by`   VARCHAR(64)   NULL                                     COMMENT '创建人',
  `created_at`   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP      COMMENT '创建时间',
  `updated_at`   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `received_at`  DATETIME      NULL                                     COMMENT '入库时间',
  INDEX `idx_tenant_status` (`tenant_id`, `status`, `id` DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='入库单表头：draft→received，received 时事务性加库存+写流水';

-- ---------- 表 19: 入库单明细（独立表） ----------
CREATE TABLE IF NOT EXISTS `stock_receipt_items` (
  `id`          BIGINT AUTO_INCREMENT PRIMARY KEY                      COMMENT '自增主键',
  `tenant_id`   VARCHAR(64)   NOT NULL DEFAULT 'default'               COMMENT '租户',
  `receipt_id`  BIGINT        NOT NULL                                 COMMENT '入库单ID（→ stock_receipts.id）',
  `product_id`  BIGINT        NOT NULL                                 COMMENT '产品ID（→ products.id）',
  `quantity`    INT           NOT NULL                                 COMMENT '入库数量',
  `unit_cost`   DECIMAL(12,2) NULL                                     COMMENT '采购成本单价',
  `remark`      VARCHAR(256)  NULL                                     COMMENT '备注',
  INDEX `idx_receipt` (`receipt_id`),
  INDEX `idx_product` (`product_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='入库单明细表';

-- ---------- 表 20: 出库单（表头） ----------
CREATE TABLE IF NOT EXISTS `stock_issues` (
  `id`           BIGINT AUTO_INCREMENT PRIMARY KEY                     COMMENT '自增主键',
  `tenant_id`    VARCHAR(64)   NOT NULL DEFAULT 'default'              COMMENT '租户',
  `issue_no`     VARCHAR(32)   NOT NULL                                 COMMENT '出库单号（IS-日期-序号）',
  `ref_type`     VARCHAR(16)   NOT NULL                                 COMMENT '用途：sales_delivery=销售出库 / repair=维修领料 / adjust=盘点调出',
  `ref_id`       BIGINT        NULL                                     COMMENT '来源单据ID（出库单/维修工单）',
  `warehouse_id` BIGINT        NOT NULL                                 COMMENT '出库仓库',
  `status`       VARCHAR(16)   NOT NULL DEFAULT 'draft'                 COMMENT '状态：draft → issued（issued 时才扣库存）',
  `remark`       VARCHAR(256)  NULL                                     COMMENT '备注',
  `created_by`   VARCHAR(64)   NULL                                     COMMENT '创建人',
  `created_at`   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP      COMMENT '创建时间',
  `updated_at`   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `issued_at`    DATETIME      NULL                                     COMMENT '出库时间',
  INDEX `idx_tenant_status` (`tenant_id`, `status`, `id` DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='出库单表头：draft→issued，issued 时事务性扣库存+写流水';

-- ---------- 表 21: 出库单明细（独立表） ----------
CREATE TABLE IF NOT EXISTS `stock_issue_items` (
  `id`         BIGINT AUTO_INCREMENT PRIMARY KEY                       COMMENT '自增主键',
  `tenant_id`  VARCHAR(64)   NOT NULL DEFAULT 'default'                COMMENT '租户',
  `issue_id`   BIGINT        NOT NULL                                   COMMENT '出库单ID（→ stock_issues.id）',
  `product_id` BIGINT        NOT NULL                                   COMMENT '产品ID',
  `quantity`   INT           NOT NULL                                   COMMENT '出库数量',
  `remark`     VARCHAR(256)  NULL                                       COMMENT '备注',
  INDEX `idx_issue` (`issue_id`),
  INDEX `idx_product` (`product_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='出库单明细表';

-- ---------- 表 22: 销售订单（表头，计划单——不扣库存） ----------
CREATE TABLE IF NOT EXISTS `sales_orders` (
  `id`             BIGINT AUTO_INCREMENT PRIMARY KEY                   COMMENT '自增主键',
  `tenant_id`      VARCHAR(64)   NOT NULL DEFAULT 'default'            COMMENT '租户',
  `order_no`       VARCHAR(32)   NOT NULL                               COMMENT '订单号（SO-日期-序号）',
  `customer_id`    BIGINT        NOT NULL                               COMMENT '客户ID（→ customers.id）',
  `amount`         DECIMAL(14,2) NOT NULL DEFAULT 0                    COMMENT '订单金额（confirm 时按明细汇总固化）',
  `status`         VARCHAR(16)   NOT NULL DEFAULT 'draft'               COMMENT '状态：draft/confirmed/delivering/completed/cancelled/closed',
  `sales_rep`      VARCHAR(64)   NULL                                   COMMENT '销售员',
  `expected_date`  DATE          NULL                                   COMMENT '交期',
  `remark`         VARCHAR(256)  NULL                                   COMMENT '备注',
  `created_by`     VARCHAR(64)   NULL                                   COMMENT '发起登录用户（own 范围依据）',
  `created_at`     DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP     COMMENT '创建时间',
  `updated_at`     DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `completed_at`   DATETIME      NULL                                   COMMENT '完结时间',
  INDEX `idx_tenant_status` (`tenant_id`, `status`, `id` DESC),
  INDEX `idx_order_no` (`order_no`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='销售订单表头（计划单）：confirm 占库存，出库走 sales_deliveries';

-- ---------- 表 23: 销售订单明细 ----------
CREATE TABLE IF NOT EXISTS `sales_order_items` (
  `id`            BIGINT AUTO_INCREMENT PRIMARY KEY                    COMMENT '自增主键',
  `tenant_id`     VARCHAR(64)   NOT NULL DEFAULT 'default'             COMMENT '租户',
  `order_id`      BIGINT        NOT NULL                                COMMENT '订单ID（→ sales_orders.id）',
  `product_id`    BIGINT        NOT NULL                                COMMENT '产品ID（→ products.id）',
  `quantity`      INT           NOT NULL                                COMMENT '数量',
  `unit_price`    DECIMAL(12,2) NOT NULL                                COMMENT '单价（下单时从 products 快照，此后改价不影响）',
  `line_amount`   DECIMAL(14,2) NOT NULL                                COMMENT '行金额 = quantity × unit_price',
  `delivered_qty` INT           NOT NULL DEFAULT 0                      COMMENT '已出库数量（出库单回写，支持分批出库）',
  INDEX `idx_order` (`order_id`),
  INDEX `idx_product` (`product_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='销售订单明细表（价格快照 + 分批出库回写）';

-- ---------- 表 24: 销售出库单（表头，执行单——在这里扣库存） ----------
CREATE TABLE IF NOT EXISTS `sales_deliveries` (
  `id`            BIGINT AUTO_INCREMENT PRIMARY KEY                    COMMENT '自增主键',
  `tenant_id`     VARCHAR(64)   NOT NULL DEFAULT 'default'             COMMENT '租户',
  `delivery_no`   VARCHAR(32)   NOT NULL                                COMMENT '出库单号（SD-日期-序号）',
  `order_id`      BIGINT        NOT NULL                                COMMENT '销售订单ID（→ sales_orders.id）',
  `warehouse_id`  BIGINT        NOT NULL                                COMMENT '发货仓库',
  `status`        VARCHAR(16)   NOT NULL DEFAULT 'draft'                COMMENT '状态：draft→released(已扣库存)→shipped(已交接物流)/cancelled',
  `released_by`   VARCHAR(64)   NULL                                    COMMENT '放行人（审批后落）',
  `remark`        VARCHAR(256)  NULL                                    COMMENT '备注',
  `created_by`    VARCHAR(64)   NULL                                    COMMENT '创建人',
  `created_at`    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP     COMMENT '创建时间',
  `updated_at`    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  INDEX `idx_tenant_status` (`tenant_id`, `status`, `id` DESC),
  INDEX `idx_order` (`order_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='销售出库单表头（执行单）：release 时事务性扣库存，支持一单多次出库';

-- ---------- 表 25: 销售出库明细（独立表） ----------
CREATE TABLE IF NOT EXISTS `sales_delivery_items` (
  `id`             BIGINT AUTO_INCREMENT PRIMARY KEY                   COMMENT '自增主键',
  `tenant_id`      VARCHAR(64)   NOT NULL DEFAULT 'default'            COMMENT '租户',
  `delivery_id`    BIGINT        NOT NULL                               COMMENT '出库单ID（→ sales_deliveries.id）',
  `order_item_id`  BIGINT        NOT NULL                               COMMENT '订单明细行ID（→ sales_order_items.id）',
  `product_id`     BIGINT        NOT NULL                               COMMENT '产品ID',
  `quantity`       INT           NOT NULL                               COMMENT '本次出库数量（≤ 订单行未出量）',
  `snapshot_price` DECIMAL(12,2) NULL                                   COMMENT '出库价快照',
  INDEX `idx_delivery` (`delivery_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='销售出库明细表';

-- ---------- 表 26: 物流单 ----------
CREATE TABLE IF NOT EXISTS `shipments` (
  `id`           BIGINT AUTO_INCREMENT PRIMARY KEY                     COMMENT '自增主键',
  `tenant_id`    VARCHAR(64)   NOT NULL DEFAULT 'default'              COMMENT '租户',
  `shipment_no`  VARCHAR(32)   NOT NULL                                 COMMENT '物流单号（TRK-日期-序号）',
  `delivery_id`  BIGINT        NOT NULL                                 COMMENT '出库单ID（→ sales_deliveries.id）',
  `carrier`      VARCHAR(64)   NOT NULL                                 COMMENT '承运商',
  `tracking_no`  VARCHAR(64)   NULL                                     COMMENT '承运商运单号',
  `status`       VARCHAR(16)   NOT NULL DEFAULT 'created'               COMMENT '状态：created → in_transit → delivered',
  `shipped_at`   DATETIME      NULL                                     COMMENT '发运时间',
  `delivered_at` DATETIME      NULL                                     COMMENT '签收时间',
  `created_by`   VARCHAR(64)   NULL                                     COMMENT '创建人',
  `created_at`   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP      COMMENT '创建时间',
  INDEX `idx_tenant_status` (`tenant_id`, `status`),
  INDEX `idx_delivery` (`delivery_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='物流单表';

-- ---------- 表 27: 退货单 ----------
CREATE TABLE IF NOT EXISTS `sales_returns` (
  `id`            BIGINT AUTO_INCREMENT PRIMARY KEY                    COMMENT '自增主键',
  `tenant_id`     VARCHAR(64)   NOT NULL DEFAULT 'default'             COMMENT '租户',
  `return_no`     VARCHAR(32)   NOT NULL                                COMMENT '退货单号（SR-日期-序号）',
  `order_id`      BIGINT        NOT NULL                                COMMENT '原订单ID',
  `product_id`    BIGINT        NOT NULL                                COMMENT '退货产品',
  `quantity`      INT           NOT NULL                                COMMENT '退货数量',
  `reason`        VARCHAR(256)  NULL                                    COMMENT '退货原因',
  `status`        VARCHAR(16)   NOT NULL DEFAULT 'draft'                COMMENT '状态：draft→received(收货)→refunded',
  `inspect_result` VARCHAR(8)   NULL                                    COMMENT '质检结论：ok=重入库 / scrap=报废不回库',
  `warehouse_id`  BIGINT        NULL                                    COMMENT '退货入库仓库',
  `created_by`    VARCHAR(64)   NULL                                    COMMENT '创建人',
  `created_at`    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP     COMMENT '创建时间',
  `received_at`   DATETIME      NULL                                    COMMENT '收货时间',
  INDEX `idx_tenant_status` (`tenant_id`, `status`),
  INDEX `idx_order` (`order_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='退货单表：received 时按质检结论决定是否回补库存';

-- ---------- 表 28: 采购订单（表头） ----------
CREATE TABLE IF NOT EXISTS `purchase_orders` (
  `id`            BIGINT AUTO_INCREMENT PRIMARY KEY                    COMMENT '自增主键',
  `tenant_id`     VARCHAR(64)   NOT NULL DEFAULT 'default'             COMMENT '租户',
  `po_no`         VARCHAR(32)   NOT NULL                                COMMENT '采购单号（PO-日期-序号）',
  `supplier_id`   BIGINT        NOT NULL                                COMMENT '供应商ID（→ suppliers.id）',
  `amount`        DECIMAL(14,2) NOT NULL DEFAULT 0                     COMMENT '采购金额（submit 时按明细固化）',
  `status`        VARCHAR(16)   NOT NULL DEFAULT 'draft'                COMMENT '状态：draft→submitted→approved→received/partial/closed',
  `buyer`         VARCHAR(64)   NULL                                    COMMENT '采购员',
  `expected_date` DATE          NULL                                    COMMENT '期望到货日',
  `created_by`    VARCHAR(64)   NULL                                    COMMENT '创建人',
  `created_at`    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP     COMMENT '创建时间',
  `updated_at`    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  INDEX `idx_tenant_status` (`tenant_id`, `status`, `id` DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='采购订单表头：金额>10万须 super_admin 审批';

-- ---------- 表 29: 采购订单明细 ----------
CREATE TABLE IF NOT EXISTS `purchase_order_items` (
  `id`          BIGINT AUTO_INCREMENT PRIMARY KEY                      COMMENT '自增主键',
  `tenant_id`   VARCHAR(64)   NOT NULL DEFAULT 'default'               COMMENT '租户',
  `po_id`       BIGINT        NOT NULL                                  COMMENT '采购单ID（→ purchase_orders.id）',
  `product_id`  BIGINT        NOT NULL                                  COMMENT '产品ID',
  `quantity`    INT           NOT NULL                                  COMMENT '采购数量',
  `unit_price`  DECIMAL(12,2) NOT NULL                                  COMMENT '采购单价',
  `received_qty` INT          NOT NULL DEFAULT 0                        COMMENT '已入库数量（入库单回写）',
  INDEX `idx_po` (`po_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='采购订单明细表';

-- ---------- 表 30: 故障代码库（维修知识沉淀） ----------
CREATE TABLE IF NOT EXISTS `fault_codes` (
  `id`                  BIGINT AUTO_INCREMENT PRIMARY KEY              COMMENT '自增主键',
  `tenant_id`           VARCHAR(64)   NOT NULL DEFAULT 'default'       COMMENT '租户',
  `code`                VARCHAR(32)   NOT NULL                          COMMENT '故障代码（如 E-SRV-01）',
  `category`            VARCHAR(16)   NOT NULL                          COMMENT '类别：mechanical/electrical/software/hydraulic',
  `name`                VARCHAR(128)  NOT NULL                          COMMENT '标准故障名',
  `standard_solution`   TEXT          NULL                              COMMENT '标准处置SOP摘要（可联动 RAG 知识库检索）',
  `avg_repair_hours`    DECIMAL(6,1)  NULL                              COMMENT '历史平均维修工时（派单参考）',
  `is_active`           TINYINT       NOT NULL DEFAULT 1                COMMENT '是否启用（主数据通用列）',
  `created_by`          VARCHAR(64)   NULL                              COMMENT '创建人',
  `created_at`          DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  UNIQUE KEY `uk_tenant_code` (`tenant_id`, `code`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='故障代码库：标准故障+标准SOP+平均工时（与 RAG 检索联动）';

-- ---------- 表 31: 维修工单（纠正性维修） ----------
CREATE TABLE IF NOT EXISTS `repair_orders` (
  `id`              BIGINT AUTO_INCREMENT PRIMARY KEY                  COMMENT '自增主键',
  `tenant_id`       VARCHAR(64)   NOT NULL DEFAULT 'default'           COMMENT '租户',
  `order_no`        VARCHAR(32)   NOT NULL                              COMMENT '工单号（RO-日期-序号）',
  `equipment_id`    BIGINT        NOT NULL                              COMMENT '设备ID（→ equipment.id，自动判保内外/关键度）',
  `fault_code_id`   BIGINT        NULL                                  COMMENT '故障代码ID（→ fault_codes.id，诊断归类）',
  `fault_desc`      TEXT          NOT NULL                              COMMENT '故障现象描述',
  `priority`        VARCHAR(16)   NOT NULL DEFAULT 'normal'             COMMENT '优先级：low/normal/high/urgent（A类设备自动urgent）',
  `warranty`        VARCHAR(8)    NOT NULL DEFAULT 'out'                COMMENT '保修：in=保内 / out=保外（按台账 warranty_until 自动判）',
  `source`          VARCHAR(16)   NOT NULL DEFAULT 'report'             COMMENT '来源：report=人工报修 / digital_employee / pm=保养转单',
  `technician_id`   BIGINT        NULL                                  COMMENT '维修工程师ID（→ engineers.id，assigned 时落）',
  `status`          VARCHAR(16)   NOT NULL DEFAULT 'open'               COMMENT '状态：open/assigned/in_progress/resolved/verified/cancelled',
  `downtime_hours`  DECIMAL(8,2)  NULL                                  COMMENT '停机时长（resolved 时计算）',
  `resolution`      TEXT          NULL                                  COMMENT '处置结论',
  `resolved_at`     DATETIME      NULL                                  COMMENT '修复时间',
  `created_by`      VARCHAR(64)   NULL                                  COMMENT '发起登录用户',
  `created_at`      DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP    COMMENT '创建时间',
  `updated_at`      DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  INDEX `idx_tenant_status` (`tenant_id`, `status`, `priority`, `id` DESC),
  INDEX `idx_equipment` (`equipment_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='维修工单表：open→assigned→in_progress→resolved→verified，关单须审批';

-- ---------- 表 32: 维修领料（工单 BOM） ----------
CREATE TABLE IF NOT EXISTS `repair_parts` (
  `id`               BIGINT AUTO_INCREMENT PRIMARY KEY                 COMMENT '自增主键',
  `tenant_id`        VARCHAR(64)   NOT NULL DEFAULT 'default'          COMMENT '租户',
  `repair_order_id`  BIGINT        NOT NULL                              COMMENT '工单ID（→ repair_orders.id）',
  `product_id`       BIGINT        NOT NULL                              COMMENT '备件ID（→ products.id）',
  `quantity`         INT           NOT NULL                              COMMENT '领用数量',
  `issued_at`        DATETIME      NULL                                  COMMENT '领料时间（同时扣库存）',
  `remark`           VARCHAR(256)  NULL                                  COMMENT '备注（保内记保修成本/保外记客户收费）',
  `created_by`       VARCHAR(64)   NULL                                  COMMENT '领料人',
  INDEX `idx_order` (`repair_order_id`),
  INDEX `idx_product` (`product_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='维修领料表：领料即事务性扣库存并写流水';

-- ---------- 表 33: 保养计划（预防性维修 PM） ----------
CREATE TABLE IF NOT EXISTS `pm_plans` (
  `id`           BIGINT AUTO_INCREMENT PRIMARY KEY                     COMMENT '自增主键',
  `tenant_id`    VARCHAR(64)   NOT NULL DEFAULT 'default'              COMMENT '租户',
  `plan_no`      VARCHAR(32)   NOT NULL                                 COMMENT '计划号（PM-日期-序号）',
  `equipment_id` BIGINT        NOT NULL                                 COMMENT '设备ID（→ equipment.id）',
  `name`         VARCHAR(128)  NOT NULL                                 COMMENT '计划名（如 贴标机月度润滑）',
  `cycle_type`   VARCHAR(16)   NOT NULL DEFAULT 'monthly'              COMMENT '周期：daily/weekly/monthly/quarterly',
  `last_done_at` DATETIME      NULL                                     COMMENT '上次执行时间',
  `next_due_at`  DATETIME      NOT NULL                                 COMMENT '下次到期时间（到期生成工单）',
  `checklist`    JSON          NULL                                     COMMENT '保养项清单',
  `assignee_id`  BIGINT        NULL                                     COMMENT '默认指派工程师ID（→ engineers.id）',
  `is_active`    TINYINT       NOT NULL DEFAULT 1                      COMMENT '是否启用',
  `created_by`   VARCHAR(64)   NULL                                     COMMENT '创建人',
  `created_at`   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP      COMMENT '创建时间',
  INDEX `idx_tenant_due` (`tenant_id`, `is_active`, `next_due_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='保养计划表：到期手动/调度生成维修工单（source=pm）';

-- ---------- 表 10: 高危操作审批单（数字员工写操作的缰绳） ----------
-- 任何高危动作（如关单/取消/改价）先在这里挂 pending，人工 approve 后才执行；
-- 执行结果回填 result 列，形成完整审计链：谁申请→为什么→谁批→执行成什么样。
CREATE TABLE IF NOT EXISTS `approval_requests` (
  `id`             BIGINT AUTO_INCREMENT PRIMARY KEY                     COMMENT '自增主键',
  `tenant_id`      VARCHAR(64)   NOT NULL DEFAULT 'default'              COMMENT '租户（审批单按租户隔离）',
  `action_type`    VARCHAR(64)   NOT NULL                                 COMMENT '动作类型：sales_order.complete / sales_order.cancel / repair_order.resolve / repair_order.cancel / ...',
  `payload`        JSON          NOT NULL                                 COMMENT '动作参数（批准后按此执行，含 tenant_id 供执行时校验归属）',
  `reason`         VARCHAR(500)  NULL                                     COMMENT '数字员工申请该操作的理由',
  `requested_by`   VARCHAR(64)   NOT NULL DEFAULT 'digital_employee'      COMMENT '发起者（数字员工岗位标识）',
  `requested_role` VARCHAR(32)   NULL                                     COMMENT '发起者登录角色（审批分级校验：须更高角色审批）',
  `status`         VARCHAR(16)   NOT NULL DEFAULT 'pending'               COMMENT '审批状态：pending / approved / rejected',
  `decided_by`     VARCHAR(64)   NULL                                     COMMENT '审批人',
  `decided_role`   VARCHAR(32)   NULL                                     COMMENT '审批人角色（审计：证明批准时角色达标）',
  `decided_at`     DATETIME      NULL                                     COMMENT '审批时间',
  `result`         JSON          NULL                                     COMMENT '批准后的执行结果回填（含失败信息，诚实审计）',
  `created_at`     DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP       COMMENT '申请时间',
  INDEX `idx_status` (`status`, `created_at` DESC),
  INDEX `idx_tenant_status` (`tenant_id`, `status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='高危操作审批表：数字员工写操作须人工批准后才执行';

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
