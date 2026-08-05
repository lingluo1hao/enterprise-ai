#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一次 RAG 回合的端到端时延对比：新路由(小模型接管 rewrite/compress) vs 旧路由(全走 7b)。"""
import time
import llm_gateway as g

SYS = {
    "classify": "判断是否需要联网搜索才能回答，只回答“是”或“否”。",
    "rewrite":  "你是一名查询改写助手。请把用户的问题改写成更适合检索的独立问句，只输出改写后的问题。",
    "grade":    "评估文档片段对回答用户问题是否有帮助，只回答“相关”“部分相关”或“不相关”。",
    "compress": "你是一名对话历史压缩助手。请把对话历史压缩成一段简洁摘要，只输出摘要。",
}
USR = {
    "classify": "2024年诺贝尔物理学奖得主是谁？",
    "rewrite":  "Python怎么学？",
    "grade":    "用户问题：如何学习Python？\n文档片段：Python 是一门高级编程语言，适合初学者。",
    "compress": "用户：你好\n助手：你好。\n用户：我想学 Python\n助手：Python 适合初学者。",
}
TURN = ["classify", "rewrite", "grade", "compress"]  # 一次典型 RAG 回合的任务顺序


def timed_call(gw, rt, task):
    s = time.time()
    r = gw._call_one(rt, SYS[task], USR[task])
    return time.time() - s, len(r.text)


def main():
    gw = g.LLMGateway(verbose=False)
    small = gw._runtimes["local-small"]
    large = gw._runtimes["local-qwen"]

    # 预热
    for t in TURN:
        gw._call_one(large, SYS[t], USR[t])
        gw._call_one(small, SYS[t], USR[t])

    # 新路由：classify->large, rewrite/grade/compress->small(优先)
    new_total = 0.0
    print("== 新路由（rewrite/grade/compress 走 1.5b）==")
    for t in TURN:
        rt = large if t == "classify" else small
        dt, _ = timed_call(gw, rt, t)
        print(f"  {t:8s} {dt:.2f}s")
        new_total += dt

    # 旧路由基线：全走 7b
    old_total = 0.0
    print("== 旧路由基线（全走 7b）==")
    for t in TURN:
        dt, _ = timed_call(gw, large, t)
        print(f"  {t:8s} {dt:.2f}s")
        old_total += dt

    print("-" * 40)
    print(f"新路由单回合：{new_total:.2f}s")
    print(f"旧路由单回合：{old_total:.2f}s")
    print(f"加速：{(1 - new_total / old_total) * 100:.0f}%  "
          f"（{old_total / new_total:.2f}x）")
    gw.close()


if __name__ == "__main__":
    main()
