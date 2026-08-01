"""
================================================================================
 mcp_client_example.py — 「别的 AI 客户端」如何复用你的工具（最小可运行示例）
================================================================================

 这个脚本模拟一个外部的 AI 客户端：它不关心你的 Skill 是怎么实现的，
 只通过 MCP 协议连接 mcp_server.py，发现工具、调用工具、读取资源、获取提示词。

 运行方式（在项目根目录）：
     python mcp_client_example.py

 你会看到：客户端通过 stdio 拉起 Server 子进程，列出可用工具，并成功调用
 calculator 工具完成一次安全计算——这正是 Claude Desktop / Cursor 等客户端
 内部做的事，区别只是它们由各自的应用拉起 Server。
================================================================================
"""

import asyncio
import os
import sys

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession


async def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    # 用与本项目相同的解释器拉起 Server，并把项目目录加入 PYTHONPATH
    python = sys.executable
    env = dict(os.environ)
    env["PYTHONPATH"] = project_dir + os.pathsep + env.get("PYTHONPATH", "")

    params = StdioServerParameters(
        command=python,
        args=[os.path.join(project_dir, "mcp_server.py")],
        env=env,
    )

    print(">>> 客户端：通过 stdio 拉起 MCP Server 子进程 ...")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1) 握手
            await session.initialize()
            print(">>> 客户端：协议握手完成 (initialize)")

            # 2) 发现工具（模拟 LLM 看到「我能调什么」）
            tools = await session.list_tools()
            print("\n[发现工具] 服务端暴露了：")
            for t in tools.tools:
                print(f"   - {t.name}: {t.description.splitlines()[0]}")

            # 3) 调用 calculator 工具（复用你的 AST 安全求值）
            print("\n[调用工具] calculator('120/24')")
            res = await session.call_tool("calculator", {"expression": "120/24"})
            print("   返回:", res.content[0].text)

            # 4) 调用 calculator 工具（投递恶意 payload，验证沙箱生效）
            print("\n[调用工具] calculator(\"__import__('os').system('rm -rf /')\")")
            res = await session.call_tool(
                "calculator",
                {"expression": "__import__('os').system('rm -rf /')"},
            )
            print("   返回:", res.content[0].text)

            # 5) 调用 doc_search 工具（复用安全守门）
            print("\n[调用工具] doc_search('JM-S509 定位精度')")
            res = await session.call_tool(
                "doc_search", {"query": "JM-S509 定位精度", "top_k": 3}
            )
            print("   返回:", res.content[0].text)

            # 6) 读取 Resource：能力清单
            print("\n[读取资源] skills://list")
            res = await session.read_resource("skills://list")
            print("   返回:", res.contents[0].text)

            # 7) 获取 Prompt 模板
            print("\n[获取提示词] security_review(calculator)")
            res = await session.get_prompt("security_review", {"skill_name": "calculator"})
            print("   返回:", res.messages[0].content.text.replace("\n", " "))

    print("\n>>> 客户端：会话结束，Server 子进程已退出。")


if __name__ == "__main__":
    asyncio.run(main())
