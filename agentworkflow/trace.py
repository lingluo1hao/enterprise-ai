# -*- coding: utf-8 -*-
"""
trace —— 统一轨迹协议（StepRecord）
================================================================================

【为什么需要统一轨迹】

诊断 DAG（Workflow 形态）产出的是「节点流转」，ReAct 探查产出的是
Think→Act→Observe 步骤——两种异构轨迹要进同一个 run json、同一段页面展示，
必须先归一到同一条记录。

一份轨迹，三处消费：
  1. agentworkflow/runs/*.json   诊断证据链落盘（可复查、可回放）
  2. CLI / Web 接口返回           诊断摘要附步骤数
  3. 排查诊断本身失败时           定位卡在哪个节点 / 哪一步

【零依赖】本模块只依赖标准库，tests 直接可测。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

# 轨迹字段截断长度：防止长 Observation 把 run json 和页面撑爆
MAX_FIELD_LEN = 500

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def truncate(text: Any, n: int = MAX_FIELD_LEN) -> str:
    """轨迹字段统一截断（超长尾部加省略标记）。"""
    s = "" if text is None else str(text)
    s = s.replace("\r", "").strip()
    return s if len(s) <= n else s[:n] + " …(截断)"


@dataclass
class StepRecord:
    """两引擎同构的单步轨迹记录。"""
    engine: str                 # "workflow" | "react"
    seq: int                    # 引擎内步骤序号（各自从 1 递增）
    step_type: str              # workflow: node_enter/node_exit；react: think/act/observe
    node_or_tool: str           # workflow 记节点名；react 记工具名
    input: str = ""             # 截断后的输入摘要
    output: str = ""            # 截断后的输出摘要
    llm_task: str = ""          # 该步走的网关任务链（非 LLM 步为空）
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TraceCollector:
    """收集 StepRecord 并落盘 run json。"""

    def __init__(self):
        self.records: List[StepRecord] = []
        self._seq: Dict[str, int] = {}   # engine -> 当前序号

    # ------------------------------------------------------------------
    def add(self, engine: str, step_type: str, node_or_tool: str,
            input: Any = "", output: Any = "", llm_task: str = "",
            latency_ms: int = 0) -> StepRecord:
        self._seq[engine] = self._seq.get(engine, 0) + 1
        rec = StepRecord(
            engine=engine, seq=self._seq[engine], step_type=step_type,
            node_or_tool=node_or_tool,
            input=truncate(input), output=truncate(output),
            llm_task=llm_task, latency_ms=int(latency_ms),
        )
        self.records.append(rec)
        return rec

    # ------------------------------------------------------------------
    def from_react_steps(self, steps) -> None:
        """把 advanced_rag_agent.ReActStep 列表归一进统一轨迹。

        不 import advanced_rag_agent（保持零依赖），按鸭子类型读字段：
        step_num / thought / action / action_input / observation / is_final
        """
        for s in steps:
            n = getattr(s, "step_num", 0) or 0
            self.add("react", "think", "react",
                     output=getattr(s, "thought", ""), llm_task="react")
            if getattr(s, "is_final", False):
                self.add("react", "observe", "final_answer",
                         output=getattr(s, "observation", ""), llm_task="react")
                continue
            self.add("react", "act", getattr(s, "action", "") or "?",
                     input=getattr(s, "action_input", ""))
            self.add("react", "observe", getattr(s, "action", "") or "?",
                     output=getattr(s, "observation", ""))
            _ = n  # seq 由 collector 自增，step_num 仅备查

    # ------------------------------------------------------------------
    def save_run(self, prefix: str = "diag", payload: Dict[str, Any] = None,
                 runs_dir: str = None) -> str:
        """落盘 run json，返回文件名（不含目录）。"""
        runs_dir = runs_dir or RUNS_DIR
        os.makedirs(runs_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{prefix}-{ts}-{int(time.time() * 1000) % 1000:03d}.json"
        data = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "steps": [r.to_dict() for r in self.records],
            "payload": payload or {},
        }
        path = os.path.join(runs_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return filename

    # ------------------------------------------------------------------
    def count_by_engine(self, engine: str) -> int:
        return sum(1 for r in self.records if r.engine == engine)
