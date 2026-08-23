# -*- coding: utf-8 -*-
"""
diagnose —— Bad Case 自动诊断入口（编排 + CLI）
================================================================================

用法：
  python -m agentworkflow --bc-id 17             # 诊断并回写 bad_cases
  python -m agentworkflow --bc-id 17 --dry-run   # 只出诊断不落库
  python -m agentworkflow --bc-id 17 --json      # 机器可读输出（含轨迹摘要）

Web 侧由 rag_web_server.py 的 POST /api/admin/bad_cases/<id>/diagnose 调用
diagnose_bad_case()，并注入生产组件（复用全局 orchestrator 的 llm / vector_db），
避免重复初始化向量库连接。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentworkflow.trace import RUNS_DIR, TraceCollector  # noqa: E402


def build_components() -> SimpleNamespace:
    """CLI 独立模式：自建组件（网关 LLM + Milvus + MySQL 记忆层）。

    Web 模式不要用这个——用 diagnose_bad_case(components=...) 注入
    rag_web_server 已初始化的组件，避免重复建向量库连接。
    """
    from advanced_rag_agent import VectorStoreManager, create_llm
    from memory_store import MySQLMemoryStore
    return SimpleNamespace(
        llm=create_llm(verbose=False),
        vector_db=VectorStoreManager.init_vector_store(),
        memory_store=MySQLMemoryStore(),
    )


def diagnose_bad_case(bc_id: int, components: Optional[SimpleNamespace] = None,
                      dry_run: bool = False, actor: str = "cli") -> Dict[str, Any]:
    """
    对一条 bad case 执行自动诊断。

    :param bc_id: bad_cases 表主键
    :param components: 组件注入（llm/vector_db/memory_store）；None 则 CLI 自建
    :param dry_run: True 只产出结论不回写
    :param actor: 触发者（Token 用量归因 & 审计）
    :return: {ok, bc_id, root_cause, title, engine, confidence, diagnosis,
              written, run_file, steps}
    """
    comp = components or build_components()
    ms = comp.memory_store
    if ms is None:
        return {"ok": False, "error": "记忆层不可用"}

    bc = ms.get_bad_case(bc_id)
    if not bc:
        return {"ok": False, "error": f"bad case {bc_id} 不存在"}

    from agentworkflow.pipeline import build_diagnosis_graph
    collector = TraceCollector()
    graph = build_diagnosis_graph(comp, collector)
    final = graph.invoke({"bc": bc, "dry_run": dry_run, "actor": actor})

    wb = final.get("writeback") or {}
    triage = final.get("triage") or {}
    run_file = collector.save_run(
        prefix=f"diag-{bc_id}",
        payload={"bc_id": bc_id, "dry_run": dry_run, "actor": actor,
                 "triage": triage, "writeback": wb},
    )
    return {
        "ok": True,
        "bc_id": bc_id,
        "root_cause": wb.get("root_cause"),
        "title": triage.get("title", ""),
        "engine": wb.get("engine", "workflow"),
        "confidence": wb.get("confidence", "低"),
        "diagnosis": wb.get("diagnosis", ""),
        "written": wb.get("written", False),
        "run_file": os.path.join(RUNS_DIR, run_file),
        "steps": {
            "workflow": collector.count_by_engine("workflow"),
            "react": collector.count_by_engine("react"),
        },
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agentworkflow",
        description="Bad Case 自动诊断（AgentWorkflow：Workflow 流水线 + ReAct 探查）")
    parser.add_argument("--bc-id", type=int, required=True, help="bad_cases 表主键")
    parser.add_argument("--dry-run", action="store_true", help="只出诊断不回写")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = parser.parse_args(argv)

    result = diagnose_bad_case(args.bc_id, dry_run=args.dry_run, actor="cli")
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if not result.get("ok"):
            print(f"✗ 诊断失败：{result.get('error')}")
            return 1
        print("=" * 60)
        print(f"  Bad Case #{result['bc_id']} 自动诊断"
              f"{'（dry-run，未回写）' if args.dry_run else ''}")
        print("=" * 60)
        print(f"  根因     : {result['root_cause'] or '未能归因（转人工）'}"
              f"  {result['title']}")
        print(f"  引擎     : {result['engine']}   置信度: {result['confidence']}")
        print(f"  回写     : {'已写入 bad_cases（open→in_progress）' if result['written'] else '未写入'}")
        print(f"  轨迹     : {result['run_file']}"
              f"（workflow {result['steps']['workflow']} 步 / react {result['steps']['react']} 步）")
        print("-" * 60)
        print(result["diagnosis"])
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
