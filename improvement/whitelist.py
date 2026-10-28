# -*- coding: utf-8 -*-
"""
improvement.whitelist —— 可改对象白名单与不可变护栏
================================================================================

RSI 改造的**收敛点**：Agent 能改的东西必须穷举、必须结构化、必须可回滚。
本模块就是那张穷举表，也是 T1 不可变护栏的唯一事实来源。

三条纪律
--------
1. 白名单外的任何东西一律不改（含源码、租户过滤 expr、评分器、门禁自身）。
2. 判定权在人手上：哪些层能自动合并，见 gate.py 的分级表，不在这里放宽。
3. 新增可改对象必须显式登记到本文件，并同步 RSI 改造方案文档的白名单表。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# ============================================================================
# 层级
# ============================================================================

LAYER_PROMPT = "P1"      # 提示词层：改 prompt_templates 的 system / user_template
LAYER_PARAM = "P2"       # 参数层：改检索/精排参数（进程内覆盖，不写回配置文件）
LAYER_PLAYBOOK = "P3"    # 经验层：playbook 的新增 / 合并 / 降权

LAYERS = (LAYER_PROMPT, LAYER_PARAM, LAYER_PLAYBOOK)

LAYER_LABEL = {
    LAYER_PROMPT: "提示词层",
    LAYER_PARAM: "参数层",
    LAYER_PLAYBOOK: "经验层",
}


# ============================================================================
# P1 提示词层白名单
# ============================================================================
# 来源：langgraph_rag_agent.py 里 self.pm.get_prompt(...) 的实际调用点
# （1095 classify / 1216 chitchat / 1738 reviewer_check / 1818 writer_compose /
#   1858 rewrite_first / 1868 rewrite_retry / 2113 grade_docs /
#   2168 generate_answer / 2346 planner_decompose / 2378 planner_supplement /
#   2460 compress_history）。
# 不在表里、也不在主链路被调用的提示词名，提案一律拒绝（防改到不知道谁在用）。

PROMPT_WHITELIST: Dict[str, str] = {
    "classify": "意图分类路由",
    "chitchat": "闲聊判定",
    "rewrite_first": "首轮查询改写",
    "rewrite_retry": "重试轮查询改写",
    "grade_docs": "文档相关性打分",
    "generate_answer": "最终答案生成",
    "reviewer_check": "多智能体复核",
    "writer_compose": "多智能体成稿",
    "planner_decompose": "多跳问题拆解",
    "planner_supplement": "拆解补充",
    "compress_history": "历史压缩",
}


# ============================================================================
# P2 参数层白名单
# ============================================================================
# 只列 evalkit/harness_retrieval.py 的 Retriever.setup() 真实消费的覆盖键
# （harness_retrieval.py:143-161），改这些键不需要动源码。
# 键 -> (类型, 安全区间, 说明)

PARAM_WHITELIST: Dict[str, Tuple[type, str, str]] = {
    "hybrid": (bool, "0/1", "混合检索开关（稀疏 BM25 + 稠密）"),
    "rerank": (bool, "0/1", "精排开关"),
    "top_k": (int, "1-50", "最终返回条数"),
    "candidate_k": (int, "1-200", "召回候选池深度"),
    "rerank_max_chars": (int, "200-8000", "送进精排的单条字符上限（压延迟的杠杆）"),
}


# ============================================================================
# T1 不可变护栏集 —— 触碰即拒，不进验证
# ============================================================================
# 每一条都对应一个真实事故或一类作弊：
#   · 安全类：租户隔离是本项目最硬的约束，任何自动改动都不得靠近
#   · 评分类：对应 DGM 删检测日志刷分 —— 优化器找到评分器漏洞就会把漏洞当能力
#   · 门禁类：防「改门禁」这条作弊路径本身

IMMUTABLE_TARGETS = {
    # ---- 安全相关 ----
    "tenant_filter_expr",
    "access_control",
    "role_matrix",
    "rag_docs_collection",
    "knowledge_id_whitelist",
    # ---- 评分相关（防改评分器）----
    "judge_prompt",
    "evalgrade_task",
    "faith_threshold",
    "relevancy_threshold",
    # ---- 门禁自身 ----
    "holdout_golden",
    "gate_config",
}

# 提示词里凡是名字带这些片段的，一律不可改（judge / 评分 / 安全策略）
IMMUTABLE_PROMPT_SUBSTRINGS = ("judge", "grade_task", "evalgrade", "safety", "security")


# ============================================================================
# 根因 -> 可改对象路由表
# ============================================================================
# 这是「诊断」通向「改进」的桥：拿到一个 R 码，就知道该去动哪些东西。
# 空列表 = 这个根因不该由配置层解决（要么改参数、要么属于不可变范围）。

CAUSE_TO_PROMPTS: Dict[str, List[str]] = {
    "R1": ["rewrite_first"],                       # 召回丢失：改写把原意带偏
    "R2": [],                                      # 排序埋没：改参数（RRF / 精排）
    "R3": ["rewrite_first", "rewrite_retry"],      # 改写/精排负优化
    "R4": [],                                      # 权限误杀：属不可变范围，人工处理
    "R5": ["generate_answer", "writer_compose"],   # 生成幻觉：收紧「只据上下文」约束
    "R6": ["classify", "planner_decompose", "generate_answer"],  # 答非所问：路由/拆解
    "R7": ["classify"],                            # 拒答错误：校准判定边界
    "R8": [],                                      # 跨租户泄漏：零容忍，只告警不自动改
}

CAUSE_TO_PARAMS: Dict[str, List[str]] = {
    "R1": ["candidate_k", "hybrid"],
    "R2": ["rerank", "top_k"],
    "R3": ["rerank", "rerank_max_chars"],
    "R4": [],
    "R5": [],
    "R6": [],
    "R7": [],
    "R8": [],
}


# ============================================================================
# 校验
# ============================================================================

class TargetRejected(Exception):
    """目标不在白名单内 / 触碰不可变护栏。"""


def normalize_target(target: str) -> str:
    return (target or "").strip()


def is_immutable(target: str) -> bool:
    """是否触碰 T1 不可变护栏。"""
    t = normalize_target(target)
    if not t:
        return True
    if t in IMMUTABLE_TARGETS:
        return True
    low = t.lower()
    return any(s in low for s in IMMUTABLE_PROMPT_SUBSTRINGS)


def layer_of(target: str) -> Optional[str]:
    """返回目标所属层级；不在任何白名单里则返回 None。"""
    t = normalize_target(target)
    if t in PROMPT_WHITELIST:
        return LAYER_PROMPT
    if t in PARAM_WHITELIST:
        return LAYER_PARAM
    if t == "playbook":
        return LAYER_PLAYBOOK
    return None


def validate_target(target: str) -> str:
    """
    校验一个可改对象，返回其层级。

    抛出 TargetRejected 的情况：
      · 空目标
      · 触碰不可变护栏（T1）
      · 不在任何白名单里
    """
    t = normalize_target(target)
    if not t:
        raise TargetRejected("目标为空")
    if is_immutable(t):
        raise TargetRejected(f"触碰不可变护栏（T1），一律拒绝：{t}")
    layer = layer_of(t)
    if layer is None:
        raise TargetRejected(
            f"目标不在白名单内：{t}（可改对象必须显式登记到 whitelist.py）")
    return layer


def param_bounds(target: str):
    """返回参数的安全区间描述；非参数层返回 None。"""
    item = PARAM_WHITELIST.get(normalize_target(target))
    return item[1] if item else None


def cast_param(target: str, value):
    """把参数值转成白名单声明的类型，转不了就原样返回。"""
    item = PARAM_WHITELIST.get(normalize_target(target))
    if not item:
        return value
    typ = item[0]
    try:
        if typ is bool:
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if typ is int:
            return int(value)
    except (TypeError, ValueError):
        return value
    return value


def targets_for_cause(cause: str, layer: Optional[str] = None) -> List[str]:
    """
    给定一个根因码，返回该动哪些可改对象。

    layer 为空则返回该根因在所有层上的候选（按 提示词 -> 参数 顺序）。
    """
    cause = (cause or "").strip().upper()
    if layer == LAYER_PROMPT:
        return [t for t in CAUSE_TO_PROMPTS.get(cause, []) if not is_immutable(t)]
    if layer == LAYER_PARAM:
        return [t for t in CAUSE_TO_PARAMS.get(cause, []) if not is_immutable(t)]
    out = []
    out.extend(targets_for_cause(cause, LAYER_PROMPT))
    out.extend(targets_for_cause(cause, LAYER_PARAM))
    return out
