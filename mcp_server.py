"""
================================================================================
 mcp_server.py — 企业级 RAG Agent 的 MCP Server（标准协议暴露层）
================================================================================

 作用
 ----
 把项目里「协议无关」的 Skill 内核（skill_framework.py）包装成标准 MCP Tools，
 任何兼容 MCP 的客户端（Claude Desktop / Cursor / 自研 Agent / 其它 AI 应用）
 都能零改造地复用你的工具。

 暴露能力
 --------
   Tools   : calculator(expression)        → 复用 CalculatorSkill + AST 沙箱
             doc_search(query, top_k)      → 复用 DocSearchSkill 的安全守门
   Resource: skills://list                → 技能清单（让客户端自动发现能力）
   Prompt  : security_review              → 复用审计/安全口径的可复用提示词

 传输方式
 --------
   默认 stdio（本地子进程，最常用）
   想远程暴露时改用 HTTP：python mcp_server.py --http   （Streamable HTTP）

 安全延续
 --------
   - 所有工具先过 BaseSkill.validate_params() 参数白名单
   - calculator 仍走 safe_eval()（AST 白名单），绝不出现 eval()
   - 凭据已外部化（见 .env），Server 本身不持有任何密钥
================================================================================
"""

import os
import sys
import argparse

from fastmcp import FastMCP

# 复用同一份 Skill 内核（与 in-process Agent 完全一致，逻辑不分叉）
from skill_framework import CalculatorSkill, SkillRegistry

mcp = FastMCP("EnterpriseRAGSkills")

# 复用真实的工具沙箱实例
_calc = CalculatorSkill()
_registry = SkillRegistry()
_registry.register(_calc)


# ----------------------------------------------------------------------------
# Tool 1：计算器（完整可运行，安全求值）
# ----------------------------------------------------------------------------
@mcp.tool()
def calculator(expression: str) -> str:
    """
    执行数学计算。适用于需要数值运算、单位换算等场景。
    输入数学表达式（如 120/24 或 5*24），返回计算结果。
    仅允许数字与 + - * / // % ** ( ) 运算符，杜绝任意代码执行。
    """
    return _calc.execute(expression)


# ----------------------------------------------------------------------------
# Tool 2：文档检索（复用安全守门；真实检索在挂载向量库后自动生效）
# ----------------------------------------------------------------------------
@mcp.tool()
def doc_search(query: str, top_k: int = 5) -> str:
    """
    搜索企业文档知识库（ChromaDB 向量数据库），返回与查询相关的文档片段。
    输入：搜索关键词或问题描述。输出：相关文档片段列表。
    """
    # 延迟导入：没有 chromadb/ollama 的环境也能正常启动 Server
    try:
        from advanced_rag_agent import DocSearchSkill
    except Exception as e:  # pragma: no cover
        return f"[doc_search] 未找到 DocSearchSkill：{e}"

    # 复用与 in-process Agent 完全相同的参数白名单校验（同一份逻辑）
    probe = DocSearchSkill.__new__(DocSearchSkill)  # 仅构造外壳，不触发重依赖
    err = probe.validate_params(query)
    if err:
        return err

    # 真实部署时，这里应传入已构造好的 llm + vector_db 实例：
    #     skill = DocSearchSkill(llm, vector_db, fast_mode=True)
    #     return skill.execute(query)
    # 当前演示环境未挂载向量库，故仅完成安全守门，证明「工具沙箱」被复用。
    return (
        f"[doc_search] 参数校验通过（top_k={top_k}）。"
        "该工具在真实部署中连接 ChromaDB 返回文档片段；"
        "当前演示环境未挂载向量库，已复用与 in-process Agent 同一份"
        "validate_params 安全守门。"
    )


# ----------------------------------------------------------------------------
# Resource：能力清单（让 MCP 客户端自动发现你的工具）
# ----------------------------------------------------------------------------
@mcp.resource("skills://list")
def list_skills() -> list:
    """返回当前 Server 暴露的全部技能清单（名称 + 描述）。"""
    return _registry.list_skills()


# ----------------------------------------------------------------------------
# Prompt：可复用的安全评审提示词模板
# ----------------------------------------------------------------------------
@mcp.prompt()
def security_review(skill_name: str) -> str:
    """为某个 Skill 生成一份「上线前安全体检」提示词，供客户端直接套用。"""
    return (
        f"请对技能 `{skill_name}` 做一次上线前安全体检，逐项确认：\n"
        "1. 输入是否经过白名单校验（非空 / 长度 / 危险模式）？\n"
        "2. 是否杜绝了 eval / exec / os.system 等任意代码执行？\n"
        "3. 密钥是否从 .env 外部化，未硬编码在代码里？\n"
        "4. 调用是否有限流与结构化审计日志？\n"
        "5. 结果是否按用户角色做了权限过滤？"
    )


def _run():
    parser = argparse.ArgumentParser(description="Enterprise RAG Agent MCP Server")
    parser.add_argument(
        "--http", action="store_true",
        help="使用 Streamable HTTP 传输（默认 stdio）",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.http:
        print(f"[MCP] 启动 Streamable HTTP Server → http://{args.host}:{args.port}/mcp")
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        # 默认 stdio：由客户端作为子进程拉起，通过 stdin/stdout 通信
        mcp.run(transport="stdio")


if __name__ == "__main__":
    _run()
