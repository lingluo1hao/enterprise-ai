---
name: code-review-cn
description: >-
  统一的代码审查清单。当用户要求 review 代码、检查 PR、分析代码质量，
  或提到 "review / 审查 / 找 bug / 安全检查 / PR / 全量审核" 时使用。
  覆盖安全漏洞、错误处理、命名与重复代码。
allowed-tools: Read, Grep, Glob, Bash(git:*), Bash(bash .claude/skills/code-review-cn/scripts/secret_scan.sh:*)
---

# 代码审查清单

按以下顺序审查当前改动或全量代码库。

## 审查模式（根据用户意图选择）

- **全量审核**：用户未指定文件 / diff、或说"全量审核/扫一遍/检查整个项目"时，
  扫描整个项目源码（排除 `.git`、`node_modules`、`dist`、`build`、`venv`、
  `.venv`、`__pycache__`、`.claude` 等生成目录）。
- **增量审查**：用户提到"改动 / PR / diff / 最近提交"时，基于 `git diff`
  （review PR 用 `git diff <base>...HEAD`）只审当前改动。

## 1. 安全（最高优先级）
- 是否泄露密钥 / API Key / Token / 连接串（重点扫 `.env*`、`settings.*`、
  `config.*`、`init_db.sql`、硬编码 IP/密码）
- 是否存在命令注入、SQL 注入、路径穿越
- 输入是否经过校验与转义
- 是否越权访问 / 认证绕过（多租户项目重点看 tenant/权限过滤是否下推）

## 2. 正确性
- 边界条件与空值处理
- 并发 / 竞态
- 错误是否被吞掉（空 catch / except: pass）
- 数据库读写是否原子（先删后插、read-modify-write 是否有锁）

## 3. 质量
- 函数是否单一职责、命名清晰
- 是否有明显重复代码
- 是否有对应测试

## 全量审核的节奏（防止输出爆炸）

- 先列项目目录结构，按模块分批审查。
- 优先审高风险文件：含网络/上传/SQL/exec/eval/文件写/权限过滤的模块。
- 每批最多输出 N 条发现（N=10），审完一批询问用户是否继续下一批。

## 输出格式

每条发现给出：文件:行号 → 一句话问题 → 建议修复。
按严重程度排序（🔴 高危 / 🟡 中危 / 🟢 建议）。
若没有问题，明确说「未发现明显问题」，不要编造。

详细安全规则见 [REFERENCE.md](REFERENCE.md)。

## 密钥初筛（可选快速通道）
- 先执行 `bash .claude/skills/code-review-cn/scripts/secret_scan.sh` 做密钥正则初筛（结果作为信号，不进入上下文）。
- 有命中 → 按 REFERENCE.md 密钥规则逐条核验；无命中 → 正常继续审查。
