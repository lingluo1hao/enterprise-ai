#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
local-small vs local-qwen 模型对比测试脚本

用法：
    python test_llm_gateway_models.py

作用：
    用固定任务集分别请求 local-small（当前 qwen2.5:1.5b）和 local-qwen（qwen2:7b），
    输出延迟、token 用量和首条回答，帮助判断哪些任务适合下放给小模型。
"""

import statistics
import time

import llm_gateway as g


TASKS = [
    (
        "is_math",
        "判断以下问题是否属于数学计算类，只回答“是”或“否”。",
        "一辆汽车以60km/h行驶2小时，走了多少公里？",
    ),
    (
        "need_search",
        "判断是否需要联网搜索才能回答，只回答“是”或“否”。",
        "2024年诺贝尔物理学奖得主是谁？",
    ),
    (
        "sentiment",
        "判断情感倾向，只回答“正面”“负面”或“中性”。",
        "这次更新简直太棒了，速度提升明显。",
    ),
    (
        "grade_doc",
        "评估这份文档片段对回答用户问题是否有帮助，只回答“相关”“部分相关”或“不相关”。",
        "用户问题：如何学习Python？\n文档片段：Python 是一门高级编程语言，适合初学者。",
    ),
    (
        "rewrite",
        "你是一名查询改写助手。请把用户的问题改写成更适合检索的独立问句，只输出改写后的问题，不要解释。",
        "Python怎么学？",
    ),
    (
        "compress",
        "你是一名对话历史压缩助手。请把下面的对话历史压缩成一段简洁的摘要，保留关键信息，只输出摘要。",
        "用户：你好\n助手：你好，有什么可以帮你的？\n用户：我想学 Python\n助手：Python 是一门很适合初学者的语言。",
    ),
]


def benchmark(gw, model_key, system, user, n=3):
    rt = gw._runtimes[model_key]

    # 预热一次，避免把模型加载时间算进统计
    gw._call_one(rt, system, user)

    times = []
    outputs = []
    tokens = []
    for _ in range(n):
        start = time.time()
        resp = gw._call_one(rt, system, user)
        times.append(time.time() - start)
        outputs.append(resp.text.strip().replace("\n", " "))
        tokens.append(resp.total_tokens)
    return times, outputs, tokens


def main():
    gw = g.LLMGateway(verbose=False)

    small_name = gw._runtimes["local-small"].cfg.model
    large_name = gw._runtimes["local-qwen"].cfg.model

    print("=" * 80)
    print(f"{small_name} vs {large_name} 对比测试")
    print("=" * 80)

    for task_id, system, user in TASKS:
        print(f"\n任务: {task_id}")
        for model_key, real_name in [("local-small", small_name), ("local-qwen", large_name)]:
            times, outputs, tokens = benchmark(gw, model_key, system, user, n=3)
            print(
                f"  {real_name}: "
                f"avg={statistics.mean(times):.2f}s "
                f"min={min(times):.2f}s "
                f"max={max(times):.2f}s "
                f"tokens={statistics.mean(tokens):.0f}"
            )
            print(f"     输出: {outputs[0]}")

    gw.close()


if __name__ == "__main__":
    main()
