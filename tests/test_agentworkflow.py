#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_agentworkflow —— Bad Case 自动诊断单测（零外部依赖）
================================================================================

项目测试惯例：纯 Python PASS/FAIL 脚本，直接 `python tests/test_agentworkflow.py`，
不依赖 Ollama / Milvus / MySQL / langgraph（重模块不在本测试的 import 范围内，
只测 rules.py / trace.py 的纯逻辑层）。
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentworkflow import rules            # noqa: E402
from agentworkflow.trace import TraceCollector, truncate  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


# ----------------------------------------------------------------------
# 一、归因规则（表驱动：rules.classify_signals）
# ----------------------------------------------------------------------
def test_classify_signals():
    print("\n[1] 归因规则 classify_signals")

    hit = {"file": "a.pdf", "page": 1, "tenant": "jm", "snippet": "x"}

    # R8：本租户视角命中其他租户文档 → 实锤泄漏，最高优先级
    r = rules.classify_signals(leak_hits=[hit], scoped_hits=[hit], full_hits=[hit],
                               docs_relevant=True, scores={"faithfulness": 0.9, "relevancy": 0.9})
    check("R8 泄漏优先级最高", r["code"] == "R8" and r["confidence"] == "高", f"got {r['code']}")

    # R1（高置信）：租户视角零召回、全库也无
    r = rules.classify_signals(scoped_hits=[], full_hits=[], docs_relevant=False,
                               scores={"faithfulness": None, "relevancy": None})
    check("R1 零召回=高置信", r["code"] == "R1" and r["confidence"] == "高", f"got {r}")

    # R1（中置信 + R4 方向提示）：本租户看不到、全库能看到
    r = rules.classify_signals(scoped_hits=[], full_hits=[hit], docs_relevant=False,
                               scores={"faithfulness": None, "relevancy": None})
    check("R1 内容在他库=中置信+R4提示", r["code"] == "R1" and r["confidence"] == "中"
          and "R4" in r["reason"], f"got {r}")

    # R1：有命中但 LLM 判定内容不相关
    r = rules.classify_signals(scoped_hits=[hit], full_hits=[hit],
                               docs_relevant=False, docs_gap="片段与问题无关",
                               scores={"faithfulness": None, "relevancy": None})
    check("R1 相关性判定否=高置信", r["code"] == "R1" and "片段与问题无关" in r["reason"], f"got {r}")

    # R7：文档能答但答案拒了（误拒）
    r = rules.classify_signals(scoped_hits=[hit], full_hits=[hit],
                               docs_relevant=True, refused=True,
                               scores={"faithfulness": 0.2, "relevancy": 0.2})
    check("R7 误拒优先于生成类", r["code"] == "R7", f"got {r['code']}")

    # R5：faithfulness 低 → 幻觉，附 judge reason
    r = rules.classify_signals(scoped_hits=[hit], full_hits=[hit],
                               docs_relevant=True,
                               scores={"faithfulness": 0.35, "relevancy": 0.9,
                                       "reason": "字段无出处"})
    check("R5 幻觉=高置信+judge证据", r["code"] == "R5" and r["confidence"] == "高"
          and "字段无出处" in r["reason"], f"got {r}")

    # R6：faithfulness 过线但 relevancy 低
    r = rules.classify_signals(scoped_hits=[hit], full_hits=[hit],
                               docs_relevant=True,
                               scores={"faithfulness": 0.9, "relevancy": 0.3})
    check("R6 答非所问=中置信", r["code"] == "R6" and r["confidence"] == "中", f"got {r}")

    # 全正常 → 无法归因 + 升级 ReAct
    r = rules.classify_signals(scoped_hits=[hit], full_hits=[hit],
                               docs_relevant=True, source="feedback",
                               scores={"faithfulness": 0.9, "relevancy": 0.9})
    check("全正常→escalate", r["code"] is None and r["escalate"] is True, f"got {r}")

    # judge 不可用 → 不下生成类结论、不升级
    r = rules.classify_signals(scoped_hits=[hit], full_hits=[hit],
                               docs_relevant=True,
                               scores={"error": "judge 未初始化"})
    check("judge不可用→转人工不硬猜", r["code"] is None and r["escalate"] is False, f"got {r}")

    # 复跑彻底失败 → 无信号不硬猜
    r = rules.classify_signals(retrieval_error="Milvus 不可达")
    check("复跑失败→无信号不硬猜", r["code"] is None and r["confidence"] == "低", f"got {r}")


# ----------------------------------------------------------------------
# 二、租户解析 / 结论解析 / 路径租户推断
# ----------------------------------------------------------------------
def test_parsers():
    print("\n[2] 解析器")
    check("点踩诊断文本解析租户",
          rules.parse_tenant("用户点踩（tenant=jm），待 triage。") == "jm")
    check("无租户返回 None", rules.parse_tenant("用户点踩，待 triage。") is None)

    c = rules.parse_conclusion('{"code":"R5","evidence":"编了字段","suggestion":"强化prompt"}')
    check("探查结论 JSON 解析", c["code"] == "R5" and c["evidence"] == "编了字段")

    c = rules.parse_conclusion('根因是 r3，建议关掉改写 {"code":"r3","evidence":"raw优于pipeline"}')
    check("夹带文字的宽松解析+小写码归一", c["code"] == "R3", f"got {c}")

    c = rules.parse_conclusion('{"code":"9"}')
    check("非法 R 码归 None", c["code"] is None)

    c = rules.parse_conclusion("模型自由发挥没有 JSON")
    check("无 JSON→code None（不硬猜）", c["code"] is None)

    check("normalize 数字码", rules.normalize_code("5") == "R5")
    check("normalize 小写", rules.normalize_code("r8") == "R8")
    check("normalize 非法", rules.normalize_code("R99") is None)

    check("路径租户推断（正斜杠）", rules.tenant_of_path("knowledge/jm/JM-S509.pdf") == "jm")
    check("路径租户推断（反斜杠）", rules.tenant_of_path("knowledge\\yh\\doc.pdf") == "yh")
    check("无 knowledge 段返回 None", rules.tenant_of_path("JM-S509.pdf") is None)


# ----------------------------------------------------------------------
# 三、统一轨迹（TraceCollector）
# ----------------------------------------------------------------------
def test_trace():
    print("\n[3] 统一轨迹 TraceCollector")
    c = TraceCollector()
    c.add("workflow", "node_enter", "prepare", input="q", output="ok")
    c.add("workflow", "node_exit", "prepare", output={"k": 1}, latency_ms=12)
    c.add("react", "think", "react", output="先搜一下")

    class FakeStep:   # 鸭子类型模拟 ReActStep（不 import 重模块）
        def __init__(self, step_num, thought, action, action_input, observation, is_final):
            self.step_num, self.thought = step_num, thought
            self.action, self.action_input = action, action_input
            self.observation, self.is_final = observation, is_final

    c.from_react_steps([
        FakeStep(1, "需要检索", "doc_search", "定位精度", "", False),
        FakeStep(2, "信息充足", "", "", "最终结论……", True),
    ])
    check("轨迹计数按引擎分列",
          c.count_by_engine("workflow") == 2 and c.count_by_engine("react") == 6,
          f"wf={c.count_by_engine('workflow')} react={c.count_by_engine('react')}"
          "（react = 1 条手工 + 步骤1的 think/act/observe + 步骤2的 think/observe）")

    long_text = "x" * 2000
    t = truncate(long_text, 500)
    check("字段截断 500", len(t) < 520 and t.endswith("…(截断)"))

    with tempfile.TemporaryDirectory() as td:
        fn = c.save_run(prefix="diag-test", payload={"bc_id": 1}, runs_dir=td)
        import json as _json
        with open(os.path.join(td, fn), encoding="utf-8") as f:
            data = _json.load(f)
        check("run json 落盘可读", data["payload"]["bc_id"] == 1
              and len(data["steps"]) == len(c.records))


# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  test_agentworkflow · Bad Case 自动诊断纯逻辑单测")
    print("=" * 60)
    test_classify_signals()
    test_parsers()
    test_trace()
    print("-" * 60)
    print(f"  结果: {PASS} PASS / {FAIL} FAIL")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
