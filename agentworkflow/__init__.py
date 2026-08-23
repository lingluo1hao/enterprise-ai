# -*- coding: utf-8 -*-
"""
agentworkflow —— Bad Case 自动诊断（AgentWorkflow 生产落地）
================================================================================

【定位】（见 docs/reports/AgentWorkflow_BadCase自动诊断方案.md）

把进入 bad_cases 库的失败样本从「人工逐条 triage」变成「自动诊断回写、人工只确认」。
两种 Agent 形态在本场景有真实分工：

  · Workflow（预定义 DAG，pipeline.py）：标准诊断五步固定——
        复跑检索 → 文档相关性探查 → judge 复判 → 规则归因 → 回写
    成本可预测（3~4 次 LLM 调用），失败可定位到节点。

  · ReAct（动态决策，probe.py）：标准流水线低置信 / 信号矛盾时的开放探查。
    复用 advanced_rag_agent.ReActAgent + SkillRegistry，每步走网关 react 链。

【边界】诊断是「建议」，不是结论：root_cause / diagnosis 回写后仍由管理员
人工确认流转（resolved 归人）；本模块不做自动修复执行。

【依赖纪律】rules.py / trace.py 零第三方依赖（tests/test_agentworkflow.py
零外部依赖可跑）；probe / pipeline / diagnose 对 langgraph、advanced_rag_agent
等重模块一律函数内懒加载。
"""

from agentworkflow.diagnose import diagnose_bad_case, build_components  # noqa: F401

__all__ = ["diagnose_bad_case", "build_components"]
