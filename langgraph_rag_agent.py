#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangGraph 版高级 RAG Agent
==========================

从 LangChain 手写 ReAct 循环迁移到 LangGraph StateGraph，实现四大升级：

  1. 显式状态机 + 条件边精细分支（classify → simple/complex/chitchat 三路）
  2. 多轮检索（query_rewrite → retrieve → grade_docs 反馈循环，最多 3 轮）
  3. 多智能体协作（planner 拆解 → researcher 并行检索 → reviewer 把关 → writer 成稿）
  4. 多轮对话（历史记忆 + 上下文消解 + 超窗摘要压缩）

复用 advanced_rag_agent.py 中的 OllamaLLM / VectorStoreManager / CacheManager / AccessControlFilter，
不重复造轮子。

图结构：
  START → load_history → classify → [条件边]
    ├ simple  → query_rewrite → retrieve → grade_docs → [条件边]
    │            ↑__________________ rewrite(不足且<3轮) ___|
    │            → rerank_mmr → generate_simple → respond
    ├ complex → planner → reviewer → [条件边]
    │            ↑_______ insufficient(不充分且<2轮) ______|
    │            → writer → respond
    └ chitchat → direct_llm → respond
  respond → save_history → END

运行方式：
  python langgraph_rag_agent.py                          # 交互模式
  python langgraph_rag_agent.py "定位方式有哪些？"        # 直接提问
  python langgraph_rag_agent.py "问题" --admin           # 特权用户
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import sys
import time
import json
import re
import traceback
from typing import TypedDict, List, Dict, Any, Optional

import warnings
warnings.filterwarnings("ignore")

from langgraph.graph import StateGraph, END, START

# 复用现有模块的配置和工具类（不重复造轮子）
from advanced_rag_agent import (
    OllamaLLM,
    VectorStoreManager,
    CacheManager,
    AccessControlFilter,
    OLLAMA_URL,
    MODEL_NAME,
    ROLE_ADMIN,
    ROLE_USER,
    DEFAULT_ROLE,
)

# ============================================================================
# 配置区 — 所有可调参数集中管理
# ============================================================================

# 多轮检索最大轮次。
# 当 grade_docs 判定没有足够的相关文档时，会自动换词重新检索。
# 设为 3 意味着最多检索 3 次——避免无限循环，同时给足够机会找到答案。
# 第 3 轮后不管有没有结果都强制进入下一步。
MAX_RETRIEVAL_ROUNDS = 3

# 相关文档阈值。
# grade_docs 判定为相关的文档数量 >= 此值，立即停止检索循环进入重排序。
# 1 是最宽松的：只要有一个文档被判定相关就停止。
# 设宽松是为了减少不必要的 LLM 调用（每次 grade_docs 都要调 LLM），
# 同时避免因评分过严导致"明明有相关信息却被判定为不相关"。
GRADE_THRESHOLD = 1

# 多智能体审查最大轮次。
# reviewer 判定"不充分"时，planner 会补充拆解子任务重新检索。
# 最多审查 2 轮，超过则强制输出，避免无限循环。
MAX_REVIEW_ROUNDS = 2

# 对话历史保留的最大轮数。
# 1 轮 = 用户一条消息 + 助手一条回复，所以实际存储 2 * N 条消息。
# 8 轮足够覆盖大部分多轮对话场景，超出后会触发摘要压缩。
HISTORY_MAX_TURNS = 8

# 摘要压缩触发阈值。
# 当对话历史超过此轮数时，将旧消息压缩为一条摘要，只保留最近的对话。
# 6 轮约等于 12 条消息，在此之前的旧消息会被 LLM 压缩。
HISTORY_COMPRESS_TURNS = 6

# 每次向量检索返回的文档片段数量。
# ChromaDB 的 similarity_search_with_score 的 k 参数。
# 5 是一个平衡值：太少可能遗漏关键信息，太多会塞满 LLM 上下文窗口。
RETRIEVE_TOP_K = 5

# 文档片段截断长度。
# 给 LLM 生成答案时，每个文档片段只取前 N 个字符。
# LLM 的上下文窗口有限（qwen2:7b 约 32K tokens），
# 截断可避免 token 溢出，同时保留足够的语义信息。
DOC_TRUNCATE = 350


# ============================================================================
# State 定义 — LangGraph 核心：所有节点共享的可变状态
# ============================================================================
#
# LangGraph 的核心概念：
#   - 一个 StateGraph 由多个"节点"（函数）组成
#   - 所有节点读写同一个状态字典（AgentState）
#   - 每个节点返回一个 dict，LangGraph 自动合并到全局 state 中
#   - 条件边根据 state 的某个字段值决定下一步走哪个节点
#
# 这个 TypedDict 定义了所有状态字段及其类型。
# total=False 表示所有字段都是可选的（不必在初始状态中全部提供）。

class AgentState(TypedDict, total=False):
    """
    LangGraph 全局状态字典。

    这个字典在整个图执行过程中被所有节点共享。
    每个节点读取需要的字段，返回要更新的字段，LangGraph 自动合并。

    字段分为四组：对话上下文、路由、多轮检索、多智能体、输出。
    """

    # ===== 对话上下文 =====

    # 会话 ID，用于区分不同用户/会话的历史记录。
    # 命令行模式用 "cli_session"，Web 模式用 "web_session"。
    session_id: str

    # 完整对话历史，格式为 [{role: "user"|"assistant", content: "..."}, ...]。
    # load_history 节点从内存加载，save_history 节点写回内存。
    # 超过 HISTORY_MAX_TURNS 轮后自动压缩，旧消息合并为一条 system 摘要。
    messages: List[Dict]

    # 用户原始问题（未经任何处理）。
    # 例如用户在输入框敲的 "心跳间隔是多少？"。
    query: str

    # 上下文消解后的完整问题。
    # 例如上轮问了"JM-S509 的定位方式"，本轮追问"那它的续航呢？"
    # classify 节点会把 "那它的续航呢？" 消解为 "JM-S509 的续航如何？"。
    # 这样即使追问不完整，后续检索也能拿到完整语义。
    resolved_query: str

    # ===== 路由 =====

    # 问题类型，classify 节点输出的三类之一：
    #   "simple"   — 单事实查询，走多轮检索分支
    #   "complex"  — 多维度复合问题，走多智能体分支
    #   "chitchat" — 闲聊，直接 LLM 回答
    # 这个字段是条件边 route_after_classify 的决策依据。
    query_type: str

    # ===== 多轮检索（simple 分支使用） =====

    # 当前轮次的改写查询词列表。
    # 第 1 轮改写 2-3 个搜索词，后续轮次换角度重新改写。
    # 每个搜索词都会分别去 ChromaDB 检索，结果合并去重。
    rewritten_queries: List[str]

    # 累积检索到的文档列表，格式为 [(langchain Document, 距离分数), ...]。
    # 多轮检索的结果会不断追加（不会覆盖），所以叫"累积"。
    # 距离分数越小表示越相似（ChromaDB 返回的是 L2 欧氏距离）。
    retrieved_docs: List[Any]

    # 文档相关性评分列表，每个元素是 bool，与 retrieved_docs 一一对应。
    # True = LLM 判定该文档与问题相关（包括间接相关）。
    # grade_docs 节点一次性批量评分，route_after_grade 根据评分决定是否继续循环。
    doc_grades: List[bool]

    # 检索轮次计数，每经过一次 query_rewrite → retrieve → grade_docs 循环 +1。
    # 用于判断是否达到 MAX_RETRIEVAL_ROUNDS 上限。
    retrieval_iterations: int

    # ===== 多智能体（complex 分支使用） =====

    # planner 拆解出的子任务列表，格式为 [{"id": 1, "task": "子问题描述"}, ...]。
    # 每个子任务由一个内部 researcher 独立完成多轮检索 RAG。
    subtasks: List[Dict]

    # 各子任务的研究结果列表，格式为 [{"subtask": "...", "answer": "...", "doc_count": N}, ...]。
    # researcher 对每个子任务做完检索+生成后，结果存入此列表。
    # writer 节点汇总所有结果生成最终答案。
    research_results: List[Dict]

    # 审查轮次计数，每经过一次 planner → reviewer 循环 +1。
    # reviewer 判定"不充分"时回到 planner 补充拆解，重新检索。
    # 超过 MAX_REVIEW_ROUNDS 后强制输出。
    review_rounds: int

    # reviewer 的审查结论。
    # True = 回答充分，进入 writer 撰写最终答案
    # False = 回答不充分，回到 planner 补充拆解
    review_passed: bool

    # ===== 输出 =====

    # 最终答案，所有分支的终点都是生成这个字段。
    # generate_simple / writer / direct_llm 三个节点各产生一种答案。
    answer: str

    # 用户角色，取值 "admin"（特权）或 "user"（普通）。
    # 用于 AccessControlFilter 做文档级权限过滤。
    role: str

    # 错误信息，仅在异常时非空。
    # 当前版本主要是 query() 方法在 catch 异常时设置。
    error: Optional[str]


# ============================================================================
# LangGraph RAG 应用主类
# ============================================================================
class LangGraphRAGApp:
    """
    LangGraph 版 RAG Agent — 状态图驱动的知识库问答引擎。

    核心思想：
    把原来手写的 ReAct 循环 + Planning Agent 重构为显式状态图（StateGraph）。
    图中每个节点是一个独立的处理函数，负责一个明确的任务。
    节点之间通过"边"连接，条件边根据 state 状态值决定走哪条分支。
    循环通过条件边回到上游节点实现（如 grade_docs → query_rewrite）。

    优势：
    - 可读性：控制流一目了然，不再是一大段 while 循环里的 if/else
    - 可扩展：新增功能只需加节点和边，不影响现有流程
    - 可观测：节点级追踪，每一步的执行记录清晰可见
    - 可测试：每个节点可独立单元测试
    """

    def __init__(self, fast_mode: bool = False):
        """
        初始化 LangGraph RAG Agent。

        执行顺序：
        1. 连接 Ollama LLM（复用 advanced_rag_agent 的 OllamaLLM 类）
        2. 加载 ChromaDB 向量数据库（复用 VectorStoreManager）
        3. 初始化 Redis 缓存 + 内存对话历史存储
        4. 构建 LangGraph 状态图（注册节点 + 连线 + 条件边）

        参数：
            fast_mode: True 时 classify 使用规则分类（不调 LLM），速度更快但准确性降低。
                       适合对响应速度要求高、问题类型简单的场景。
        """
        print("=" * 70)
        print("  LangGraph RAG Agent 初始化")
        print("=" * 70)

        # 1. LLM（复用现有 OllamaLLM）
        # OllamaLLM 封装了对 Ollama 的 HTTP API 调用。
        # chat(system_prompt, user_prompt) 方法发送 prompt 并返回文本响应。
        print("\n[1/3] 连接 LLM...")
        self.llm = OllamaLLM()

        # 2. 向量数据库（复用现有 VectorStoreManager）
        # VectorStoreManager 封装了 ChromaDB 的初始化、文档索引、向量检索。
        # init_vector_store() 会扫描 docs/ 目录，首次运行时自动构建索引。
        print("\n[2/3] 加载向量数据库...")
        self.vector_db = VectorStoreManager.init_vector_store()

        # 3. 缓存 + 对话历史
        # CacheManager: Redis 两级缓存（精确匹配 + 语义匹配）
        # _history_store: 内存字典，以 session_id 为 key，存储在内存中
        #                  注意：重启程序后历史会丢失，仅用于单次运行的多轮对话
        print("\n[3/3] 初始化缓存与对话历史...")
        self.cache = CacheManager()
        self._history_store: Dict[str, List[Dict]] = {}
        self.fast_mode = fast_mode

        # 4. 构建 StateGraph
        # _build_graph() 定义所有节点、边、条件边，最后 compile() 生成可执行的图。
        # compile() 会校验图结构（无孤立节点、无死循环等），返回 CompiledStateGraph。
        print("\n[系统] 构建 LangGraph 状态图...")
        self.graph = self._build_graph()
        print("[系统] 初始化完成\n")

    # ========================================================================
    # 图构建 — 定义整个 Agent 的执行流程
    # ========================================================================
    def _build_graph(self):
        """
        构建 LangGraph 状态图：注册节点 + 连边 + 条件边。

        原理：
        - StateGraph(AgentState) 创建一个以 AgentState 为状态类型的新图
        - add_node(name, fn) 注册一个处理函数为图节点，name 是节点名
        - add_edge(from, to) 添加一条固定边（无条件跳转）
        - add_conditional_edges(from, router, mapping) 添加条件边：
          router 函数接收 state 返回字符串，mapping 把字符串映射到目标节点
        - compile() 编译并验证图结构，返回可执行的 CompiledStateGraph

        图结构概览：
          START
            │
            ▼
          load_history  ←── 从 _history_store 加载该会话的历史消息
            │
            ▼
          classify      ←── LLM 判断问题类型 + 上下文消解（追问补全）
            │
            ├── query_type == "simple" ──► query_rewrite（多轮检索反馈循环）
            │                                  │
            │                                  ▼
            │                               retrieve（ChromaDB 检索）
            │                                  │
            │                                  ▼
            │                               grade_docs（LLM 评分相关性）
            │                                  │
            │                          ┌───────┴───────┐
            │                 不足且<3轮    │               相关或≥3轮
            │                          ▼                ▼
            │                    query_rewrite     rerank_mmr
            │                     (换词重写)        (MMR去冗余)
            │                                          │
            │                                          ▼
            │                                    generate_simple
            │                                    (基于文档生成答案)
            │                                          │
            ├── query_type == "complex" ─► planner（拆解子任务）
            │                                  │
            │                                  ▼
            │                               reviewer（审查充分性）
            │                                  │
            │                          ┌───────┴───────┐
            │                    不充分且<2轮    │       充分或≥2轮
            │                          ▼                ▼
            │                      planner          writer
            │                    (补充拆解)       (汇总撰写)
            │                                          │
            └── query_type == "chitchat" ─► direct_llm
                                                  │
                              三条分支汇合 ────────┘
                                                  │
                                                  ▼
                                               respond（最终回答）
                                                  │
                                                  ▼
                                              save_history（保存历史）
                                                  │
                                                  ▼
                                                 END

        返回值：
            CompiledStateGraph — 编译后的可执行图，通过 graph.invoke(state) 运行
        """
        # 创建一个以 AgentState 为状态类型的状态图
        graph = StateGraph(AgentState)

        # ================== 注册所有节点 ==================
        # 每个节点是一个函数，签名统一为 (state: AgentState) -> dict
        # 返回的 dict 会被 LangGraph 自动合并到全局 state 中

        # --- 公共入口节点 ---
        graph.add_node("load_history", self.node_load_history)
        graph.add_node("classify", self.node_classify)
        graph.add_node("direct_llm", self.node_direct_llm)

        # --- simple 分支节点（多轮检索） ---
        graph.add_node("query_rewrite", self.node_query_rewrite)
        graph.add_node("retrieve", self.node_retrieve)
        graph.add_node("grade_docs", self.node_grade_docs)
        graph.add_node("rerank_mmr", self.node_rerank_mmr)
        graph.add_node("generate_simple", self.node_generate_simple)

        # --- complex 分支节点（多智能体） ---
        graph.add_node("planner", self.node_planner)
        graph.add_node("reviewer", self.node_reviewer)
        graph.add_node("writer", self.node_writer)

        # --- 公共出口节点 ---
        graph.add_node("respond", self.node_respond)
        graph.add_node("save_history", self.node_save_history)

        # ================== 连线 — 图的数据流向 ==================

        # 入口边：START → load_history → classify
        # START 是 LangGraph 内置的特殊起点，固定边（无条件跳转）
        graph.add_edge(START, "load_history")
        graph.add_edge("load_history", "classify")

        # --- 分类条件边（三路分支） ---
        # route_after_classify 函数读取 state["query_type"]，返回 "simple"/"complex"/"chitchat"
        # mapping 把返回值映射到目标节点名称
        graph.add_conditional_edges(
            "classify",
            self.route_after_classify,
            {
                "simple": "query_rewrite",   # 简单查询 → 多轮检索
                "complex": "planner",        # 复杂查询 → 多智能体
                "chitchat": "direct_llm",    # 闲聊 → 直接回答
            },
        )

        # --- simple 分支：多轮检索反馈循环 ---
        # 改写 → 检索 → 评分 的线性链
        graph.add_edge("query_rewrite", "retrieve")
        graph.add_edge("retrieve", "grade_docs")
        # 评分后的条件边：相关 → 重排序，不相关 → 重写查询（形成循环）
        graph.add_conditional_edges(
            "grade_docs",
            self.route_after_grade,
            {
                "relevant": "rerank_mmr",     # 相关文档足够 → 去冗余
                "rewrite": "query_rewrite",   # 不足 → 回到改写，换词重新检索
            },
        )
        # 重排序后生成答案，然后进入公共出口
        graph.add_edge("rerank_mmr", "generate_simple")
        graph.add_edge("generate_simple", "respond")

        # --- complex 分支：多智能体协作 ---
        # planner → reviewer 固定边
        graph.add_edge("planner", "reviewer")
        # 审查后条件边：充分 → 写答案，不充分 → 回 planner 补充拆解（形成循环）
        graph.add_conditional_edges(
            "reviewer",
            self.route_after_review,
            {
                "sufficient": "writer",       # 回答充分 → 汇总撰写
                "insufficient": "planner",    # 不充分 → 补充拆解
            },
        )
        graph.add_edge("writer", "respond")

        # --- chitchat 分支 ---
        # 闲聊直接 LLM 回答，然后进入公共出口
        graph.add_edge("direct_llm", "respond")

        # --- 公共出口 ---
        # 所有分支汇聚到 respond → save_history → END
        graph.add_edge("respond", "save_history")
        graph.add_edge("save_history", END)

        # compile() 编译图：校验结构完整性，将节点和边编译为可执行的计算图
        return graph.compile()

    # ========================================================================
    # 第 1 号节点：load_history — 加载对话历史
    # ========================================================================

    def node_load_history(self, state: AgentState) -> dict:
        """
        【节点：加载对话历史】

        作用：根据 session_id 从内存中加载该会话的历史消息。

        原理：
        多轮对话需要"记住"之前聊了什么。
        这里用 session_id 隔离不同会话（不同用户/不同浏览器窗口）。
        _history_store 是一个内存字典，key 是 session_id，value 是消息列表。

        这个节点在任何问题处理之前执行，确保 classify 等后续节点能看到历史上下文。

        输入：
            state["session_id"] — 会话标识符
        输出：
            {"messages": [...]} — 该会话的历史消息列表
        """
        session_id = state.get("session_id", "default")
        history = self._history_store.get(session_id, [])
        print(f"  [load_history] 会话 {session_id}：{len(history)} 条历史消息")
        return {"messages": history}

    # ========================================================================
    # 第 2 号节点：classify — 问题分类 + 上下文消解
    # ========================================================================

    def node_classify(self, state: AgentState) -> dict:
        """
        【节点：问题分类 + 上下文消解】

        作用：判断问题类型，同时把依赖上文的追问补全为独立问题。

        原理 — 上下文消解：
        用户可能先问"A 的定位方式有哪些？"，得到答案后追问"那它的续航呢？"。
        如果直接拿"那它的续航呢？"去向量检索，几乎不可能找到相关文档。
        所以需要借助 LLM 理解历史上下文，把追问消解为"JM-S509 的续航如何？"。

        原理 — 问题分类：
        - simple: 单一事实查询，如"心跳间隔是多少？"
        - complex: 多维度复合问题，如"定位精度？几种方式？续航如何？"
        - chitchat: 闲聊、问候、感谢

        分类结果决定后续走哪条处理分支：
        - simple → 多轮检索（快速、精确）
        - complex → 多智能体协作（耗时长、回答全）
        - chitchat → 直接 LLM 回答（不触发检索）

        快速模式（fast_mode=True）或无历史上下文时，跳过 LLM 调用，
        用规则快速分类（_quick_classify），不消耗 LLM tokens。

        输入：
            state["query"] — 用户原始问题
            state["messages"] — 对话历史
        输出：
            {"query_type": "simple"|"complex"|"chitchat", "resolved_query": "消解后问题"}
        """
        query = state["query"]
        messages = state.get("messages", [])

        # 构建最近 4 轮对话历史的文本摘要
        history_text = self._format_history(messages, max_turns=4)

        if self.fast_mode or not history_text:
            # 快速模式或无历史：跳过 LLM，用规则快速分类
            qtype = self._quick_classify(query)
            print(f"  [classify] 类型={qtype}（快速分类）")
            return {"query_type": qtype, "resolved_query": query}

        # 构造 LLM prompt：让 LLM 同时进行分类和上下文消解
        # 注意：{{ 和 }} 是 Python 字符串中花括号的转义写法，
        # 因为 f-string 中的 {} 有特殊含义，这里虽然不是 f-string，
        # 但 LangChain 的 prompt 模板也会解析花括号，所以需要双花括号转义
        system = (
            "你是问题分类器。判断用户问题的类型，输出 JSON：\n"
            '{{"type": "simple|complex|chitchat", "resolved": "消解后的完整问题"}}\n\n'
            "分类规则：\n"
            '- simple：单一事实查询（如"心跳间隔是多少？"）\n'
            '- complex：多维度复合问题（如"定位精度？几种方式？续航如何？"）\n'
            '- chitchat：闲聊/打招呼\n\n'
            "如果问题是追问（依赖上文），resolved 要补全为完整的独立问题。\n"
            "如果不依赖上文，resolved 等于原问题。\n"
            "只输出 JSON，不要其他文字。"
        )
        user = f"对话历史:\n{history_text}\n\n当前问题: {query}"
        result = self.llm.chat(system, user)

        # 解析 LLM 输出的 JSON（含容错兜底）
        qtype, resolved = self._parse_classify(result, query)
        print(f"  [classify] 类型={qtype}, 消解问题={resolved[:40]}")
        return {"query_type": qtype, "resolved_query": resolved}

    # ========================================================================
    # 条件边路由函数
    # ========================================================================

    def route_after_classify(self, state: AgentState) -> str:
        """
        【条件边：分类后路由】

        作用：根据 classify 节点的输出结果，决定下一步走哪个分支。

        原理：
        这是 LangGraph 条件边的路由函数。
        它读取 state["query_type"]，返回一个字符串 key。
        _build_graph() 中的 mapping 将这个 key 映射到目标节点名称。

        输入：
            state["query_type"] — classify 节点输出的问题类型
        返回：
            "simple" | "complex" | "chitchat"
        """
        return state.get("query_type", "simple")

    # ========================================================================
    # 闲聊分支：direct_llm
    # ========================================================================

    def node_direct_llm(self, state: AgentState) -> dict:
        """
        【节点：闲聊直接回答】

        作用：对问候、感谢等非知识类问题，直接用 LLM 回答，不触发检索。

        原理：
        闲聊问题不需要查找企业文档，调用 ChromaDB 检索纯属浪费。
        这里直接让 LLM 以"友好助手"的角色简短自然地回答。

        为什么不用检索？
        - "你好" → 检索结果全是文档噪音，加入反而会污染 LLM 的回答
        - 闲聊的预期是快速、自然、友好的回复

        输入：
            state["resolved_query"] — 消解后问题（闲聊一般为原问题）
        输出：
            {"answer": "LLM 的回答"}
        """
        query = state.get("resolved_query", state["query"])
        print(f"  [direct_llm] 闲聊直接回答")
        system = "你是友好的企业助手。简短自然地回答用户的话。"
        answer = self.llm.chat(system, query)
        return {"answer": answer}

    # ========================================================================
    # 第 12 号节点：respond — 最终回答（所有分支汇聚）
    # ========================================================================

    def node_respond(self, state: AgentState) -> dict:
        """
        【节点：最终回答】

        作用：所有三条分支的汇聚点。确保 state["answer"] 有值，提供兜底文案。

        原理：
        三条分支（generate_simple / writer / direct_llm）各自生成 answer 字段。
        但万一某个分支没有设置 answer（异常、bug），这里做最后的兜底。
        这样 save_history 和最终输出不会出现空 answer。

        输入：
            state["answer"] — 由上一个节点设置的回答文本
        输出：
            {"answer": "..."} — 确保非空的最终答案
        """
        answer = state.get("answer", "抱歉，我无法回答这个问题。")
        return {"answer": answer}

    # ========================================================================
    # 第 14 号节点：save_history — 保存对话历史
    # ========================================================================

    def node_save_history(self, state: AgentState) -> dict:
        """
        【节点：保存对话历史】

        作用：将本轮问答追加到会话历史中，超过窗口限制时自动压缩。

        原理：
        每次问答结束后，将用户问题和助手回复追加到 messages 列表。
        当消息数量超过 HISTORY_MAX_TURNS * 2 时（每轮=用户+助手=2条），
        触发 _compress_history() 把旧消息压缩为一段摘要。

        为什么需要压缩？
        LLM 的上下文窗口有限（qwen2:7b 约 32K tokens）。
        如果不压缩，多轮对话会快速占满窗口，导致 LLM "忘记"早期的关键信息。
        压缩策略：用 LLM 把旧消息总结为一段简短摘要（≤100 字），
        以 system 角色的消息形式放在对话历史的开头。

        输入：
            state 中的所有相关字段
        输出：
            {"messages": [...]} — 更新后的对话历史
        """
        session_id = state.get("session_id", "default")
        messages = list(state.get("messages", []))
        query = state["query"]
        answer = state.get("answer", "")

        # 追加本轮对话：先用户问题，再助手回复
        # 两段式存储是为了后续 format_history 能区分角色
        messages.append({"role": "user", "content": query})
        messages.append({"role": "assistant", "content": answer})

        # 检查是否需要压缩
        max_msgs = HISTORY_MAX_TURNS * 2
        if len(messages) > max_msgs:
            messages = self._compress_history(messages)

        # 写回内存存储
        self._history_store[session_id] = messages
        print(f"  [save_history] 已保存，历史 {len(messages)} 条消息")
        return {"messages": messages}

    # ========================================================================
    # simple 分支节点：多轮检索反馈循环
    #
    # 循环逻辑：
    #   query_rewrite → retrieve → grade_docs → [条件判断]
    #     ↑                                        │
    #     └──── 不相关 且 <3轮 ──────────────────────┘
    #                                              │
    #                              相关 或 ≥3轮 ──→ rerank_mmr → generate_simple
    #
    # 为什么需要这个循环？
    # 单次检索经常漏掉关键信息。原因：
    # 1. 用户问题用词和文档用词不一致（"心跳" vs "心跳间隔"）
    # 2. 向量检索的语义匹配有误差
    # 3. 问题可能涉及文档的不同章节
    #
    # 循环策略：
    # - 第 1 轮：把用户问题改写为 2-3 个搜索词，全部检索
    # - 第 2+ 轮：LLM 看到上一轮的检索结果后，换角度重新改写搜索词
    # - 每轮检索结果累积（不丢弃旧结果）
    # - 直到找到足够相关文档或达到轮次��限
    # ========================================================================

    def node_query_rewrite(self, state: AgentState) -> dict:
        """
        【节点：查询重写 — 简单问题分支，simple】

        作用：将用户问题改写为 2-3 个更适合向量检索的搜索词。

        原理：
        向量检索对自然语言长问句的召回效果不如短关键词。
        例如 "请问这个设备的定位方式具体有哪些？" → 改写为 "定位方式"、"GPS定位"、"设备定位技术"。

        第 1 轮 vs 第 N 轮的区别：
        - 第 1 轮（iteration==0）：LLM 只看原始问题，正常改写
        - 第 N 轮（iteration>0）：LLM 看原始问题 + 上一轮检索到的文档片段，
          发现哪些方面还没检索到，换角度改写

        这个设计实现了"反馈循环"：
        检索不到 → 换个角度 → 重新检索 → 评分 → 还是不够 → 再换角度……

        输入：
            state["resolved_query"] — 消解后问题
            state["retrieval_iterations"] — 当前是第几轮检索
            state["retrieved_docs"] — 上轮累积的检索结果（第 2+ 轮用）
        输出：
            {"rewritten_queries": [...], "retrieval_iterations": N}
        """
        query = state.get("resolved_query", state["query"])
        iteration = state.get("retrieval_iterations", 0)

        if iteration == 0:
            # 第 1 轮：正常改写
            queries = self._do_rewrite(query, None)
        else:
            # 第 2+ 轮：基于上轮检索结果，换角度改写
            prev_docs = state.get("retrieved_docs", [])
            queries = self._do_rewrite(query, prev_docs)

        print(f"  [query_rewrite] 第 {iteration + 1} 轮改写: {queries}")
        return {"rewritten_queries": queries, "retrieval_iterations": iteration + 1}

    def node_retrieve(self, state: AgentState) -> dict:
        """
        【节点：向量检索 — 简单问题分支，simple】

        作用：用改写后的查询词去 ChromaDB 做向量相似度检索，并对结果做权限过滤。

        原理：
        每个改写查询词独立发送到 ChromaDB 的 similarity_search_with_score()。
        这个函数先对查询词做 embedding（转为向量），然后在向量空间中找最近的 top-k 个文档。
        返回的是 (Document, 距离分数) 元组，距离越小表示越相似。

        权限过滤：
        AccessControlFilter.filter_results() 根据用户角色剔除无权限的文档片段。
        例如普通用户看不到标记为 "JM-S509" 的受限文档。

        去重逻辑：
        多个查询词的检索结果可能有重叠（同一个文档片段被不同查询词召回）。
        用 page_content 的前 80 个字符做去重标识，避免重复文档降低 MMR 效果。

        输入：
            state["rewritten_queries"] — 改写后的搜索词列表
            state["role"] — 用户角色（用于权限过滤）
            state["retrieved_docs"] — 已有文档（新结果追加，不覆盖）
        输出：
            {"retrieved_docs": [(Doc, score), ...]}
        """
        queries = state.get(
            "rewritten_queries", [state.get("resolved_query", state["query"])]
        )
        role = state.get("role", DEFAULT_ROLE)

        new_docs = self._do_retrieve(queries, role)

        # 累积到已有文档列表（不覆盖旧结果，实现"多轮累积"）
        existing = state.get("retrieved_docs", [])
        merged = existing + new_docs

        print(f"  [retrieve] 本轮 {len(new_docs)} 个新片段，累计 {len(merged)} 个")
        return {"retrieved_docs": merged}

    def node_grade_docs(self, state: AgentState) -> dict:
        """
        【节点：文档相关性评分 — 简单问题分支，simple】

        作用：用 LLM 判断每个检索到的文档片段是否与问题相关。

        原理：
        向量检索返回的 top-k 结果不一定都和问题相关（语义有噪声）。
        需要 LLM 做一次"相关性筛选"，标记哪些文档真的有用。

        为什么一次性批量评分？
        如果对每个文档单独调一次 LLM，10 个文档就是 10 次 API 调用。
        批量发送（所有文档编号后放在一个 prompt 里）只需 1 次调用。
        虽然 prompt 更长，但总体耗时少得多。

        评分标准（在 prompt 中设定）：
        - "包括间接相关"：不要求文档直接回答问题，只要包含相关主题词即可
        - 例如问"心跳间隔"，文档中提到"心跳机制"、"心跳包"都算相关
        - 宽松评分是为了避免误杀——宁可把不相关的留在后面 MMR 过滤，也不能漏掉相关的

        输入：
            state["resolved_query"] — 消解后问题
            state["retrieved_docs"] — 累积的检索结果
        输出：
            {"doc_grades": [True, False, True, ...]}
        """
        query = state.get("resolved_query", state["query"])
        docs = state.get("retrieved_docs", [])

        grades = self._do_grade(query, docs)
        relevant_count = sum(grades)
        print(f"  [grade_docs] {relevant_count}/{len(grades)} 个文档相关")
        return {"doc_grades": grades}

    def route_after_grade(self, state: AgentState) -> str:
        """
        【条件边：评分后路由 — 简单问题分支，simple】

        作用：根据文档评分结果，决定是继续检索还是进入重排序。

        原理：
        两个条件，满足任意一个即停止检索循环：
        1. 相关文档数 >= GRADE_THRESHOLD（默认 1）：找到了足够的相关信息
        2. retrieval_iterations >= MAX_RETRIEVAL_ROUNDS（默认 3）：达到最大轮次上限

        返回值决定下一步：
        - "relevant" → 走 rerank_mmr 节点（去冗余 + 生成答案）
        - "rewrite" → 走 query_rewrite 节点（换词重新检索，形成循环）

        输入：
            state["doc_grades"] — 评分标记列表
            state["retrieval_iterations"] — 已执行轮数
        返回：
            "relevant" | "rewrite"
        """
        relevant_count = sum(state.get("doc_grades", []))
        iterations = state.get("retrieval_iterations", 0)

        if relevant_count >= GRADE_THRESHOLD or iterations >= MAX_RETRIEVAL_ROUNDS:
            return "relevant"
        return "rewrite"

    def node_rerank_mmr(self, state: AgentState) -> dict:
        """
        【节点：MMR 重排序 — 简单问题分支，simple】

        作用：过滤不相关文档，然后用 MMR 算法重排序，保证结果的多样性和相关性。

        原理 — 为什么要 MMR？
        向量检索容易"扎堆"：top-5 结果可能来自文档的同一段落，内容高度重复。
        如果直接把这些相似内容塞给 LLM，浪费 token 且没有信息增量。

        MMR（Maximal Marginal Relevance）在相关性和多样性之间找平衡：
        - 既选择与查询最相关的文档
        - 又避免选择与已选中文档过于相似的内容

        分两步：
        1. 过滤：丢弃 grade_docs 判定为不相关的文档
        2. MMR 重排序：贪心算法，每轮选 MMR 分数最高的文档加入结果集

        兜底逻辑：
        如果过滤后一个相关文档都没有（极端情况），回退到使用前 RETRIEVE_TOP_K 个文档。

        输入：
            state["resolved_query"] — 消解后问题
            state["retrieved_docs"] — 累积检索结果
            state["doc_grades"] — 相关性评分
        输出：
            {"retrieved_docs": [...MMR排序后的文档...]}
        """
        query = state.get("resolved_query", state["query"])
        docs = state.get("retrieved_docs", [])
        grades = state.get("doc_grades", [])

        # 第一步：过滤出相关文档
        relevant = []
        for i, (doc, score) in enumerate(docs):
            if i < len(grades) and grades[i]:
                relevant.append((doc, score))

        # 第二步：兜底 — 如果没有相关文档，用全部文档
        if not relevant:
            relevant = docs[:RETRIEVE_TOP_K]

        # 第三步：MMR 重排序
        reranked = self._mmr_rerank(query, relevant)
        print(f"  [rerank_mmr] 重排序后 {len(reranked)} 个文档")
        return {"retrieved_docs": reranked}

    def node_generate_simple(self, state: AgentState) -> dict:
        """
        【节点：生成答案 — 简单问题分支，simple】

        作用：基于检索到的文档片段，让 LLM 生成最终回答。

        原理：
        把重排序后的文档片段格式化为上下文，和用户问题一起发给 LLM。
        LLM 被要求"基于文档内容回答"（RAG 的 Grounding 原则），不能编造。

        上下文格式：
        [文档1] 内容省略...\n\n[文档2] 内容省略...
        每个文档截取前 DOC_TRUNCATE 个字符，最多取 5 个文档。

        为什么限 5 个文档？
        LLM 上下文窗口有限。5 个文档 × 350 字符 ≈ 1750 字符，
        加上 system prompt 和用户问题，通常在 3000 字符以内，远低于窗口上限。
        如果需要更多上下文，可调高 RETRIEVE_TOP_K 和 DOC_TRUNCATE。

        输入：
            state["resolved_query"] — 消解后问题
            state["retrieved_docs"] — MMR 排序后的文档
        输出：
            {"answer": "基于文档生成的回答"}
        """
        query = state.get("resolved_query", state["query"])
        docs = state.get("retrieved_docs", [])

        answer = self._do_generate(query, docs)
        print(f"  [generate_simple] 生成答案 ({len(answer)} 字)")
        return {"answer": answer}

    # ========================================================================
    # complex 分支节点：多智能体协作
    #
    # 设计理念：
    # 复杂问题（多维度复合查询）拆分给多个"智能体"角色协同完成。
    #
    # 四个角色的分工：
    #   Planner   — 拆解任务："这个问题可以分解为哪几个独立子问题？"
    #   Researcher — 研究检索：对每个子问题独立做多轮检索 RAG
    #   Reviewer  — 审查把关："这些结果是否充分回答了原始问题？"
    #   Writer    — 汇总撰写：把各子问题的研究结果整合为一篇完整答案
    #
    # 为什么不用"真正的并发"？
    # Ollama 是单卡串行推理，多个线程同时调 LLM 不会加速（反而可能因锁竞争变慢）。
    # 所以 researcher 对子任务做串行处理，简单可靠。
    #
    # 循环逻辑：
    #   planner → reviewer → [条件判断]
    #     ↑                      │
    #     └──── 不充分 且<2轮 ────┘
    #                             │
    #                充分 或≥2轮 → writer → respond
    # ========================================================================

    def node_planner(self, state: AgentState) -> dict:
        """
        【节点：Planner Agent — 复杂问题分支，complex】

        作用：将复杂问题拆解为 2-4 个独立子任务，然后对每个子任务串行执行多轮检索 RAG。

        原理：
        Planner 是"项目经理"，负责把用户的问题分解为可以独立处理的子任务。
        例如 "定位精度？几种方式？续航如何？" 拆解为：
          1. "JM-S509 的定位精度是多少？"
          2. "JM-S509 有哪些定位方式？"
          3. "JM-S509 的续航能力如何？"

        首次 vs 补充拆解的区别：
        - review_rounds == 0：首次拆解，LLM 基于原始问题创建 2-4 个子任务
        - review_rounds > 0：reviewer 判定不充分，基于已有研究结果补充新的子任务

        每个子任务由 _research_subtask() 独立完成：
        - 子任务级的 query_rewrite → retrieve → grade_docs 循环（最多 2 轮）
        - 子任务级的 MMR 重排序
        - 子任务级的 answer 生成

        输入：
            state["resolved_query"] — 消解后问题
            state["review_rounds"] — 当前审查轮次（0=首次，>0=补充）
            state["role"] — 用户角色
        """
        query = state.get("resolved_query", state["query"])
        review_rounds = state.get("review_rounds", 0)
        role = state.get("role", DEFAULT_ROLE)

        if review_rounds == 0:
            # 首次拆解：让 LLM 把问题分解为独立子任务
            subtasks = self._do_plan(query)
        else:
            # 补充拆解（审查不通过时）：基于已有结果的不足，补充新问题
            existing = state.get("research_results", [])
            subtasks = self._do_plan_supplement(query, existing)

        print(f"  [planner] 第 {review_rounds + 1} 轮：拆解出 {len(subtasks)} 个子任务")

        # 对每个子任务串行执行多轮检索 RAG（researcher 角色）
        results = []
        for st in subtasks:
            print(f"    [researcher] 处理子任务: {st['task'][:30]}")
            result = self._research_subtask(st, role)
            results.append(result)

        return {
            "subtasks": subtasks,
            "research_results": results,
            "review_rounds": review_rounds + 1,
        }

    def node_reviewer(self, state: AgentState) -> dict:
        """
        【节点：Reviewer Agent — 复杂问题分支，complex】

        作用：审查所有子任务的研究结果是否充分回答了原始问题。

        原理：
        Reviewer 是"质检员"，不参与检索和生成，只做判断：
        - 把原始问题和各子任务的研究结果发给 LLM
        - LLM 只回答"充分"或"不充分"
        - "充分" = 原始问题的每个方面都有对应的研究结果覆盖
        - "不充分" = 某个方面信息缺失或回答不够深入

        为什么需要 Reviewer？
        复杂问题的拆解可能遗漏某些维度。
        例如问"定位方式、精度、续航"，planner 可能只拆了"定位方式"和"精度"，
        漏掉了"续航"。Reviewer 会发现这一点，触发 planner 补充拆解。

        输入：
            state["resolved_query"] — 原始问题
            state["research_results"] — 各子任务的研究结果
        输出：
            {"review_passed": True|False}
        """
        query = state.get("resolved_query", state["query"])
        results = state.get("research_results", [])

        # 格式化各子任务的结果摘要
        results_text = "\n".join(
            [f"- 子任务: {r['subtask']}\n  回答: {r['answer'][:200]}" for r in results]
        )

        system = (
            "你是严格的审查员。判断以上子任务结果是否充分回答了原始问题。\n"
            '只回答"充分"或"不充分"。'
        )
        user = f"原始问题: {query}\n\n子任务结果:\n{results_text}"
        result = self.llm.chat(system, user)

        passed = "充分" in result
        print(f"  [reviewer] 审查结果: {'通过' if passed else '不通过'}")
        return {"review_passed": passed}

    def route_after_review(self, state: AgentState) -> str:
        """
        【条件边：审查后路由 — 复杂问题分支，complex】

        作用：根据 reviewer 的结论决定下一步。

        原理：
        两个条件，满足任意一个即进入 writer：
        1. review_passed == True：回答充分，可以汇总写作
        2. review_rounds >= MAX_REVIEW_ROUNDS：达到审查轮次上限，强制输出

        如果不满足，返回 planner 做补充拆解（循环）。

        输入：
            state["review_passed"] — 审查结论
            state["review_rounds"] — 审查轮次
        返回：
            "sufficient" | "insufficient"
        """
        if state.get("review_passed", False):
            return "sufficient"
        if state.get("review_rounds", 0) >= MAX_REVIEW_ROUNDS:
            return "sufficient"  # 超过最大轮次，强制输出（避免死循环）
        return "insufficient"

    def node_writer(self, state: AgentState) -> dict:
        """
        【节点：Writer Agent — 复杂问题分支，complex】

        作用：汇总所有子任务的研究结果，撰写一份结构化的最终答案。

        原理：
        Writer 是"技术撰稿人"，不参与检索，只负责整合和润色。
        把各子任务的研究结果按逻辑组织，形成一篇条理清晰的回答。

        Writer 的 prompt 强调：
        - "整合所有子任务结果"：确保不遗漏任何维度的信息
        - "基于研究结果，不要编造"：RAG 的核心约束
        - "如果某方面信息不足，如实说明"：诚实原则
        - "按逻辑组织，可分点"：结构化输出

        输入：
            state["resolved_query"] — 原始问题
            state["research_results"] — 各子任务的研究结果
        输出：
            {"answer": "汇总后的最终答案"}
        """
        query = state.get("resolved_query", state["query"])
        results = state.get("research_results", [])

        # 格式化所有子任务结果为统一格式
        results_text = "\n\n".join(
            [f"【{r['subtask']}】\n{r['answer']}" for r in results]
        )

        system = (
            "你是技术文档撰写员。根据各子任务的研究结果，撰写一份完整的回答。\n\n"
            "要求：\n"
            "- 整合所有子任务结果，按逻辑组织，可分点\n"
            "- 回答必须基于研究结果，不要编造\n"
            "- 用中文，条理清晰\n"
            "- 如果某方面信息不足，如实说明"
        )
        user = f"原始问题: {query}\n\n各子任务研究结果:\n{results_text}"

        answer = self.llm.chat(system, user)
        print(f"  [writer] 生成最终答案 ({len(answer)} 字)")
        return {"answer": answer}

    # ========================================================================
    # 检索辅助方法 — 被图节点和 researcher 共用
    # ========================================================================

    def _do_rewrite(self, query: str, prev_docs: Optional[List]) -> List[str]:
        """
        【辅助：查询改写】

        作用：把自然语言问题改写为更适合向量检索的短关键词。

        原理：
        向量检索对短关键词的召回效果通常优于长问句。
        因为文档 embedding 是按句子粒度计算的，短关键词能更精确地命中相关句子。

        第 1 轮 vs 第 N 轮的区别：
        - prev_docs is None → 第 1 轮：LLM 只看问题，自由改写 2-3 个搜索词
        - prev_docs is not None → 第 N 轮：LLM 看到之前检索的词和文档片段，
          发现哪些方向已经搜过了，换一个不同的角度改写

        最终结果：
        改写词 + 原始问题一起作为搜索词（原始问题兜底，确保不丢信息）。

        参数：
            query: 要改写的问题
            prev_docs: 上轮检索结果（None=首轮，非None=后续轮）
        返回：
            ["搜索词1", "搜索词2", "搜索词3", "原始问题"]
        """
        if prev_docs is None:
            # 第 1 轮：正常改写
            system = (
                "你是查询重写专家。将用户问题改写为 2-3 个更利于向量检索的搜索词。\n"
                "每个搜索词独占一行，不要编号，不要解释。"
            )
            result = self.llm.chat(system, query)
        else:
            # 第 N 轮：基于之前的检索结果，换角度改写
            # 只取前 3 个文档的前 120 字符，避免 prompt 过长
            prev_text = "\n".join(
                [d[0].page_content[:120] for d in prev_docs[:3]]
            )
            system = (
                "之前的检索结果不够相关。请换一个角度改写搜索词，输出 2-3 个。\n"
                "每行一个，不要编号。"
            )
            user = f"原问题: {query}\n\n已检索片段:\n{prev_text}"
            result = self.llm.chat(system, user)

        # 解析 LLM 输出：按行拆分，取前 3 个非空行
        queries = [q.strip() for q in result.strip().split("\n") if q.strip()][:3]
        # 兜底：始终保留原始问题作为搜索词
        queries.append(query)
        return queries

    def _do_retrieve(self, queries: List[str], role: str) -> List:
        """
        【辅助：向量检索 + 去重 + 权限过滤】

        作用：对多个查询词分别做 ChromaDB 向量检索，合并结果后去重。

        原理：
        1. 每个查询词独立调用 similarity_search_with_score() 取 top-k 个文档
        2. 对每个查询词的结果做权限过滤（AccessControlFilter）
        3. 用 page_content 前 80 个字符做去重 key，避免同一文档在多个查询词的结果中出现

        去重的重要性：
        如果不去重，同一段文档可能被 3 个查询词分别返回，在最终上下文里出现 3 次。
        这浪费 LLM 的上下文窗口，且会导致答案偏向这段重复内容。

        参数：
            queries: 搜索词列表
            role: 用户角色（"admin" 或 "user"，决定可见文档范围）
        返回：
            [(Document, score), ...] — 去重后的文档列表
        """
        all_results = []
        seen = set()  # 去重集合：记录已经见过的文档内容摘要
        for q in queries:
            # ChromaDB 相似度检索：返回 (Document, 距离分数) 的列表
            results = self.vector_db.similarity_search_with_score(q, k=RETRIEVE_TOP_K)
            # 根据用户角色过滤无权限文档
            results = AccessControlFilter.filter_results(results, role)
            for doc, score in results:
                # 用前 80 字符作为文档唯一标识
                content_key = doc.page_content[:80]
                if content_key not in seen:
                    seen.add(content_key)
                    all_results.append((doc, score))
        return all_results

    def _do_grade(self, query: str, docs: List) -> List[bool]:
        """
        【辅助：文档相关性批量评分】

        作用：用 LLM 一次性判断所有文档片段是否与问题相关。

        原理：
        将所有文档编号后一次性发给 LLM，让它返回相关文档的编号列表。
        这样做只需一次 LLM 调用，而不是每个文档调一次（N 次调用）。

        评分标准（宽松）：
        - "包括间接相关"：不要求文档直接回答，只要包含相关概念就算
        - 例如问"心跳间隔"，文档中出现"心跳机制"、"心跳包"都算相关
        - 宽松是为了避免误杀——不相关的留在后面 MMR 过滤阶段处理

        容错处理：
        - LLM 返回 "none" → 所有文档标记为不相关
        - LLM 返回不完整的编号 → 只标记能匹配上的，其余默认不相关
        - LLM 返回非法编号 → 忽略，不影响整体评分

        参数：
            query: 问题文本
            docs: [(Document, score), ...] 检索结果列表
        返回：
            [True, False, True, ...] — 与 docs 一一对应的评分标记
        """
        if not docs:
            return []

        # 格式化文档：为每个文档编号（[0], [1], ...）
        doc_texts = [
            f"[{i}] {d[0].page_content[:200]}" for i, d in enumerate(docs)
        ]
        system = (
            "判断每个文档片段是否与问题有关联（包括间接相关）。\n"
            "只要文档包含问题涉及的主题词或相关概念，就算关联。\n"
            "输出有关联的文档编号，逗号分隔，如: 0,2,3\n"
            "如果没有，输出: none"
        )
        user = f"问题: {query}\n\n文档:\n" + "\n".join(doc_texts)
        result = self.llm.chat(system, user)

        # 初始化：全部标记为 False
        grades = [False] * len(docs)

        # 解析 LLM 返回的编号
        if "none" not in result.lower():
            # 用正则提取所有数字（应对 LLM 输出格式不一致的情况）
            nums = re.findall(r"\d+", result)
            for n in nums:
                idx = int(n)
                if 0 <= idx < len(docs):
                    grades[idx] = True
        return grades

    def _do_generate(self, query: str, docs: List) -> str:
        """
        【辅助：基于检索文档生成答案】

        作用：把检索到的文档格式化为上下文，让 LLM 基于上下文生成回答。

        原理：
        这是 RAG 的 "G" (Generation) 阶段。
        核心原则是 Grounded Generation：LLM 必须基于提供的上下文回答，不能编造。
        文档上下文作为"外部知识"注入到提示词中，约束 LLM 的输出。

        格式规范：
        - 每个文档片段标注序号 [文档1], [文档2] ...
        - 文档内容截断到 DOC_TRUNCATE 字符（避免 token 溢出）
        - 最多取前 5 个文档（再多占太多上下文空间）

        参数：
            query: 用户问题
            docs: [(Document, score), ...] 检索到的文档列表
        返回：
            LLM 生成的回答文本
        """
        # 构建上下文：最多 5 个文档，每个截断到 350 字符
        context = "\n\n".join(
            [f"[文档{i+1}] {d[0].page_content[:DOC_TRUNCATE]}" for i, d in enumerate(docs[:5])]
        )
        system = (
            "你是企业文档问答助手。根据检索到的文档片段回答问题。\n\n"
            "要求：\n"
            "- 回答必须基于文档内容，不要编造\n"
            "- 如果信息不足，如实说明\n"
            "- 用中文回答，条理清晰"
        )
        user = f"问题: {query}\n\n检索到的文档:\n{context}"
        return self.llm.chat(system, user)

    def _research_subtask(self, subtask: Dict, role: str) -> Dict:
        """
        【Researcher Agent：对单个子任务做多轮检索 RAG】

        作用：这是 complex 分支中 Researcher 角色的核心实现。
        对 planner 拆解出的单个子任务，独立完成"改写→检索→评分→重排→生成"的完整流程。

        流程（5 步）：
        1. query_rewrite — 将子任务改写为 2-3 个搜索词
        2. retrieve — ChromaDB 向量检索 + 权限过滤
        3. grade_docs — LLM 批量评分文档相关性
        4. MMR 重排序 — 过滤不相关 + 去冗余
        5. generate — 基于最终文档生成子回答

        第 1-3 步形成一个小的反馈循环（最多 2 轮）：
        - 评分达标 → 跳出循环
        - 评分不足 → 换词重新检索

        为什么子任务级最多只检索 2 轮？
        复杂问题通常有 2-4 个子任务，每个子任务 2 轮检索，
        总共 4-8 轮 LLM 调用。如果每个子任务 3 轮，总调用数会增加 50%。
        2 轮是在"回答质量"和"响应速度"之间的折中。

        参数：
            subtask: {"id": 1, "task": "子问题"} 或纯字符串
            role: 用户角色
        返回：
            {"subtask": "子问题", "answer": "子回答", "doc_count": N}
        """
        query = subtask.get("task", str(subtask))
        all_docs = []

        # 子任务级检索最多 2 轮（控制总耗时）
        max_rounds = min(MAX_RETRIEVAL_ROUNDS, 2)
        for iteration in range(max_rounds):
            # 步骤 1：改写查询词
            if iteration == 0:
                queries = self._do_rewrite(query, None)
            else:
                queries = self._do_rewrite(query, all_docs)

            # 步骤 2：向量检索
            new_docs = self._do_retrieve(queries, role)
            all_docs.extend(new_docs)

            # 步骤 3：评分，达标则跳出循环
            grades = self._do_grade(query, all_docs)
            relevant_count = sum(grades)
            print(f"      第 {iteration + 1} 轮: {relevant_count} 个相关")

            if relevant_count >= GRADE_THRESHOLD:
                break

        # 步骤 4：过滤相关文档 + MMR 重排序
        relevant = [
            (all_docs[i][0], all_docs[i][1])
            for i in range(len(all_docs))
            if i < len(grades) and grades[i]
        ]
        # 兜底：如果全部不相关，用前几个文档
        if not relevant:
            relevant = all_docs[:RETRIEVE_TOP_K]
        reranked = self._mmr_rerank(query, relevant)

        # 步骤 5：生成子回答
        answer = self._do_generate(query, reranked)

        return {
            "subtask": query,
            "answer": answer,
            "doc_count": len(reranked),
        }

    # ========================================================================
    # 多智能体辅助方法
    # ========================================================================

    def _do_plan(self, query: str) -> List[Dict]:
        """
        【辅助：Planner 首次拆解】

        作用：让 LLM 将复杂问题拆解为 2-4 个独立可检索的子问题。

        原理：
        复杂问题通常包含多个维度（如"定位精度？几种方式？续航如何？"），
        如果一次性检索，很难同时覆盖所有维度的信息。
        拆解后每个子问题独立检索，能更精准地找到对应文档片段。

        LLM 输出格式：
        {"subtasks": [{"id": 1, "task": "子问题1"}, {"id": 2, "task": "子问题2"}, ...]}

        容错处理（_parse_json_list）：
        - 正常 JSON → 直接解析
        - LLM 输出了代码块标记（```json```）→ 清理后解析
        - 输出不是合法 JSON → 用正则提取花括号内容
        - 全部失败 → 兜底：把原问题作为单个子任务

        参数：
            query: 原始复杂问题
        返回：
            [{"id": 1, "task": "..."}, ...]
        """
        system = (
            "你是任务规划器。将复杂问题拆解为 2-4 个独立的子问题，"
            "每个子问题可以独立检索回答。\n\n"
            '输出 JSON: {{"subtasks": [{{"id": 1, "task": "子问题"}}]}}\n'
            "只输出 JSON，不要其他文字。"
        )
        result = self.llm.chat(system, query)
        return self._parse_json_list(result, "subtasks", query)

    def _do_plan_supplement(self, query: str, existing: List[Dict]) -> List[Dict]:
        """
        【辅助：Planner 补充拆解】

        作用：reviewer 判定不充分时，基于已有研究结果的不足，补充 1-2 个新子任务。

        原理：
        reviewer 说"不充分"，意味着某些维度的信息缺失。
        这时 Planner 需要"查漏补缺"——看已有的研究覆盖了哪些方面，
        还有哪些方面没涉及，生成新的子任务去检索。

        这个方法的 prompt 关键点：
        - "之前的回答不够充分" → 告诉 LLM 已有结果有问题
        - "补充 1-2 个新的子问题" → 限制数量，避免无限扩张
        - "填补信息缺口" → 明确目标是补漏，不是重新拆解

        参数：
            query: 原始问题
            existing: 已有的研究结果列表
        返回：
            [{"id": 1, "task": "新子问题"}, ...]
        """
        # 格式化已有结果摘要（每个子任务取前 100 字符，避免 prompt 过长）
        existing_text = "\n".join(
            [f"- {r['subtask']}: {r['answer'][:100]}" for r in existing]
        )
        system = (
            "之前的回答不够充分。请补充 1-2 个新的子问题来填补信息缺口。\n\n"
            '输出 JSON: {{"subtasks": [{{"id": 1, "task": "新子问题"}}]}}\n'
            "只输出 JSON。"
        )
        user = f"原问题: {query}\n\n已有结果:\n{existing_text}"
        result = self.llm.chat(system, user)
        return self._parse_json_list(result, "subtasks", query)

    # ========================================================================
    # 多轮对话辅助方法
    # ========================================================================

    def _format_history(self, messages: List[Dict], max_turns: int = 4) -> str:
        """
        【辅助：格式化对话历史为文本】

        作用：将对话历史列表转换为 LLM 可读的文本格式。

        原理：
        classify 节点需要用对话历史做上下文消解（把追问补全）。
        但 messages 是结构化的 dict 列表，需要转为纯文本给 LLM。

        格式示例：
        用户: JM-S509 的定位方式有哪些？
        助手: JM-S509 支持 GPS、LBS、WiFi 三种定位方式...
        用户: 那它的续航呢？
        助手: JM-S509 在 GPS 常开模式下续航约 120 小时...

        第 4 个问"那它的续航呢？"时，LLM 看到这段历史，
        就能推理出"它"指的是 JM-S509，从而消解为完整问题。

        参数：
            messages: 对话历史 [{role, content}, ...]
            max_turns: 最多取多少轮（默认 4 轮 = 8 条消息）
        返回：
            格式化后的历史文本
        """
        if not messages:
            return ""
        # 取最近 N 轮的消息（从后往前取，保证拿到的是最新对话）
        recent = messages[-(max_turns * 2):]
        lines = []
        for m in recent:
            role = "用户" if m["role"] == "user" else "助手"
            # 每条消息截断到 150 字符，避免历史文本过长
            lines.append(f"{role}: {m['content'][:150]}")
        return "\n".join(lines)

    def _compress_history(self, messages: List[Dict]) -> List[Dict]:
        """
        【辅助：对话历史压缩】

        作用：当消息数量超过窗口限制时，将旧消息压缩为一段摘要。

        原理：
        LLM 的上下文窗口有限（qwen2:7b 约 32K tokens ≈ 约 2 万汉字）。
        如果不压缩，聊天历史会不断增长，最终撑满窗口，导致 LLM 无法处理新问题。

        压缩策略：
        1. 保留最近的 K 条消息（K = HISTORY_COMPRESS_TURNS * 2）
        2. 将更早的旧消息发给 LLM，让它输出一段简短摘要（≤100 字）
        3. 摘要以 system 角色的消息形式放在对话历史最前面

        这样既保住了"历史上下文"（摘要里有关键信息），
        又不会占用太多窗口（摘要只有 100 字）。

        参数：
            messages: 完整对话历史
        返回：
            [{"role": "system", "content": "历史摘要: ..."}, ...近期消息...]
        """
        # 计算保留数量：保留最近 N 轮对话
        keep_count = HISTORY_COMPRESS_TURNS * 2
        old = messages[:-keep_count]      # 待压缩的旧消息
        recent = messages[-keep_count:]   # 保留的近期消息

        # 格式化旧消息为纯文本
        old_text = "\n".join(
            [f"{m['role']}: {m['content'][:100]}" for m in old]
        )
        # 让 LLM 压缩旧历史为摘要
        system = "将以下对话历史压缩为一段简短摘要（不超过100字），保留关键信息。"
        summary = self.llm.chat(system, old_text)

        print(f"  [save_history] 历史压缩: {len(old)} 条 → 1 条摘要")
        # 摘要放在最前面，近期消息附加在后面
        return [{"role": "system", "content": f"历史摘要: {summary}"}] + recent

    # ========================================================================
    # 通用辅助方法 — 解析、分类、排序
    # ========================================================================

    def _quick_classify(self, query: str) -> str:
        """
        【辅助：快速规则分类（不用 LLM）】

        作用：基于纯文本规则判断问题类型，用于 fast_mode 或无历史上下文时。

        原理：
        逐个检查关键词，速度极快（<1ms），不消耗 LLM tokens。

        分类规则（按优先级）：
        1. 包含"你好/谢谢/hi/hello/再见" → chitchat（闲聊）
        2. 包含多个问号（"？"） → complex（多问句，大概率多维度）
        3. 包含"和/且/以及/同时"等并列词且有问号 → complex（并列查询）
        4. 以上都不满足 → simple（默认，最安全）

        局限性：
        规则分类是"粗略的"，有些边界 case 会分错。
        比如"心跳间隔是多少秒和毫秒？"会被判为 complex，但实际上一个检索就能找到答案。
        但规则分类的目的不是 100% 准确，而是快速过滤。
        在完整模式下，classify 会用 LLM 做更准确的分类。

        参数：
            query: 用户问题
        返回：
            "simple" | "complex" | "chitchat"
        """
        q = query.lower()
        # 规则 1：闲聊关键词
        if any(w in q for w in ["你好", "谢谢", "你是谁", "hi", "hello", "再见"]):
            return "chitchat"
        # 规则 2：多个问号 → 复杂问题
        if query.count("？") >= 2 or query.count("?") >= 2:
            return "complex"
        # 规则 3：并列词 + 问号 → 复杂问题
        if any(w in q for w in ["和", "且", "以及", "同时"]) and any(
            w in q for w in ["？", "?"]
        ):
            return "complex"
        # 规则 4：默认简单
        return "simple"

    def _parse_classify(self, result: str, fallback_query: str):
        """
        【辅助：解析 classify 节点的 LLM 输出】

        作用：把 LLM 返回的 JSON 字符串解析为 (问题类型, 消解后问题)。

        原理：
        LLM 的输出格式不可靠，可能包含：
        - 纯 JSON：{"type": "simple", "resolved": "心跳间隔是多少？"}
        - JSON 带代码块：```json\n{...}\n```
        - 不完整 JSON：{...（缺少尾部花括号）
        - 完全非 JSON：普通文本

        容错策略（三层兜底）：
        1. 直接 json.loads：正常情况，一步成功
        2. 正则提取花括号内的内容再解析：处理带代码块标记的情况
        3. 完全失败 → 回退到规则分类 + 原问题作为消解结果

        参数：
            result: LLM 原始输出
            fallback_query: 兜底用的原始问题
        返回：
            (query_type, resolved_query) 元组
        """
        result = result.strip()
        # 清理可能的 Markdown 代码块标记
        result = re.sub(r"```(?:json)?\s*", "", result)
        result = result.replace("```", "").strip()

        try:
            # 第一层：直接解析
            data = json.loads(result)
            return data.get("type", "simple"), data.get("resolved", fallback_query)
        except (json.JSONDecodeError, TypeError):
            # 第二层：正则提取花括号内容再解析
            match = re.search(r'\{.*\}', result, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    return data.get("type", "simple"), data.get("resolved", fallback_query)
                except (json.JSONDecodeError, TypeError):
                    pass

        # 第三层：全部失败 → 规则分类兜底
        return self._quick_classify(fallback_query), fallback_query

    def _parse_json_list(self, result: str, key: str, fallback_query: str) -> List[Dict]:
        """
        【辅助：从 LLM 输出解析 JSON 列表（用于 planner）】

        作用：解析 planner 的 JSON 输出，提取子任务列表。

        原理：
        与 _parse_classify 类似，但目标是从 JSON 中提取一个 list 字段。
        planner 的典型输出：
        {"subtasks": [{"id": 1, "task": "子问题1"}, {"id": 2, "task": "子问题2"}]}

        容错策略（三层）：
        1. 直接 json.loads + 验证 list 类型 + 验证非空
        2. 正则提取花括号内容再解析
        3. 兜底：把原问题包装成单个子任务的列表

        参数：
            result: LLM 原始输出
            key: JSON 中要提取的字段名（如 "subtasks"）
            fallback_query: 兜底用的原始问题
        返回：
            [{"id": 1, "task": "..."}, ...]
        """
        result = result.strip()
        # 清理可能的 Markdown 代码块标记
        result = re.sub(r"```(?:json)?\s*", "", result)
        result = result.replace("```", "").strip()

        try:
            # 第一层：直接解析
            data = json.loads(result)
            items = data.get(key, [])
            if isinstance(items, list) and items:
                return items
        except (json.JSONDecodeError, TypeError):
            # 第二层：正则提取
            match = re.search(r'\{.*\}', result, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    items = data.get(key, [])
                    if isinstance(items, list) and items:
                        return items
                except (json.JSONDecodeError, TypeError):
                    pass

        # 第三层：兜底 — 把原问题作为单个子任务
        return [{"id": 1, "task": fallback_query}]

    def _mmr_rerank(self, query: str, results: List, k: int = 5) -> List:
        """
        【辅助：MMR (Maximal Marginal Relevance) 重排序】

        作用：在相关性和多样性之间做平衡的文档排序算法。

        原理：
        向量检索容易返回内容相似的多个文档（来自同一段落的不同切片）。
        如果全部塞给 LLM，既浪费 token 空间，又让答案偏向重复内容。

        MMR 同时优化两个目标：
        - 相关性（Relevance）：选与查询最相关的文档
        - 多样性（Diversity）：选与已选中文档不重复的新文档

        算法（贪心）：
        1. 按距离排序，选最相关的作为第一个选中项
        2. 循环选择剩余项：
           a. 对每个候选文档，计算它和已选中集合的最大相似度（Jaccard）
           b. MMR 分数 = λ × 相关性 - (1-λ) × 冗余度
           c. 选 MMR 分数最高的加入选中集合
        3. 直到选中 k 个或剩余为空

        λ（lambda_param=0.7）的含义：
        - λ 越接近 1，越看重相关性（可能选出 5 个几乎一样的内容）
        - λ 越接近 0，越看重多样性（可能选出不太相关的新内容）
        - 0.7 是经验值，在"相关"和"多样"之间 7:3 的权衡

        相似度计算（简化版 Jaccard）：
        用前 100 个字符的字符级 Jaccard 系数衡量文档间相似度。
        Jaccard = 交集字符数 / 并集字符数。
        这不是最优的语义相似度，但速度极快，适合在线计算。

        参数：
            query: 查询文本
            results: [(Document, distance_score), ...] 待排序的文档列表
            k: 最终保留的文档数（默认 5）
        返回：
            MMR 排序后的文档列表
        """
        # 如果文档数已经 <= k，不需要排序
        if len(results) <= k:
            return results

        # 按距离分数升序排列（距离越小 = 越相关）
        results = sorted(results, key=lambda x: x[1])

        # 贪心算法：从最相关的开始，逐个加入选中集合
        selected = [results[0]]      # 已选中的文档
        remaining = list(results[1:])  # 剩余候选
        lambda_param = 0.7           # 相关性权重（0.7 = 70% 看相关性，30% 看多样性）

        while len(selected) < k and remaining:
            best_idx = 0
            best_score = -float("inf")

            # 遍历每个候选文档，计算 MMR 分数
            for i, (doc, score) in enumerate(remaining):
                # 取文档的前 100 字符作为内容指纹
                content = set(doc.page_content[:100])

                # 计算该文档与已选中文档的最大相似度（找最相似的作为冗余度）
                max_sim = 0.0
                for s_doc, _ in selected:
                    s_content = set(s_doc.page_content[:100])
                    union = len(content | s_content)
                    if union > 0:
                        # Jaccard 相似度 = 交集 / 并集
                        sim = len(content & s_content) / union
                        max_sim = max(max_sim, sim)

                # MMR 公式：λ × 相关性 - (1-λ) × 最大冗余度
                # relevance = 1 / (distance + 0.01)，距离越小相关性越大
                # +0.01 防止除以 0（距离可能为 0）
                relevance = 1.0 / (score + 0.01)
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr > best_score:
                    best_score = mmr
                    best_idx = i

            # 把 MMR 分数最高的候选移到选中集合
            selected.append(remaining.pop(best_idx))

        return selected

    # ========================================================================
    # 对外查询入口 — 唯一的外部调用接口
    # ========================================================================
    def query(
        self,
        question: str,
        role: str = DEFAULT_ROLE,
        session_id: str = "default",
    ) -> str:
        """
        【对外入口：用户提问】

        这是整个 Agent 的唯一对外接口。
        CLI 模式和 Web 模式都通过这个方法来提问。

        完整流程（4 步）：
        1. Redis 缓存检查 — 命中则直接返回，跳过所有 LLM 调用
        2. LangGraph 状态图执行 — 分类 → 检索/智能体 → 生成
        3. Redis 缓存写入 — 把本次问答结果存入 Redis（加速下次）
        4. 输出结果

        为什么先查缓存？
        缓存命中率较高的问题（如"心跳间隔是多少？"），
        走缓存从十几秒降到几毫秒，极大提升用户体验。

        recursion_limit=50 的含义：
        LangGraph 的递归深度限制。每个节点执行计为一次递归。
        14 个节点 + 3 轮检索循环 + 2 轮审查循环 ≈ 可能达到 20+ 次递归。
        设为 50 留有足够余量，同时防止真正的死循环。

        参数：
            question: 用户输入的原始问题
            role: 用户角色（"admin" 可访问全部文档，"user" 仅公开文档）
            session_id: 会话标识符（多轮对话用，相同 session_id 共享历史）
        返回：
            生成的回答文本
        """
        total_start = time.time()

        print("\n" + "=" * 70)
        print(f"用户提问: {question}")
        role_desc = AccessControlFilter.get_role_description(role)
        print(f"用户角色: {role_desc}")
        print("=" * 70)

        # ---- 第一步：缓存检查 ----
        # CacheManager 内部会做标准化 → SHA256 精确匹配 → BGE 语义匹配
        self.cache.current_role = role
        cached = self.cache.lookup(question)
        if cached:
            print(f"\n[Cache] 命中缓存，直接返回（耗时 {time.time() - total_start:.1f}s）")
            print(f"\n{'─' * 70}\n{cached}\n{'─' * 70}")
            return cached

        # ---- 第二步：执行 LangGraph 状态图 ----
        # 构建初始状态字典，提供给图的起点（START → load_history）
        initial_state = {
            "query": question,
            "role": role,
            "session_id": session_id,
            "messages": self._history_store.get(session_id, []),
            "retrieved_docs": [],
            "doc_grades": [],
            "retrieval_iterations": 0,
            "research_results": [],
            "review_rounds": 0,
        }

        try:
            # graph.invoke() 是 LangGraph 的执行入口。
            # 它把初始状态注入图，自动按节点→边→条件边的顺序执行，
            # 直到到达 END 节点，返回最终状态。
            # recursion_limit=50 防止无限循环（节点调用次数上限）。
            final_state = self.graph.invoke(initial_state, {"recursion_limit": 50})
        except Exception as e:
            print(f"\n[Error] 图执行出错: {e}")
            traceback.print_exc()
            return f"处理过程中出现错误: {e}"

        answer = final_state.get("answer", "抱歉，无法回答这个问题。")
        elapsed = time.time() - total_start

        # ---- 第三步：写入缓存 ----
        self.cache.save(question, answer)

        # ---- 第四步：输出结果 ----
        print(f"\n[完成] 总耗时 {elapsed:.1f}s")
        print(f"\n{'─' * 70}\n{answer}\n{'─' * 70}")

        return answer


# ============================================================================
# CLI 入口 — 命令行交互模式
# ============================================================================
def run_interactive(app: LangGraphRAGApp, role: str):
    """
    【CLI：命令行交互模式】

    作用：提供一个简单的终端聊天界面，方便开发者测试和调试。

    支持的命令：
    - /admin 或 /特权    — 切换为特权用户（可访问全部文档）
    - /user 或 /普通     — 切换为普通用户（仅公开文档）
    - /history 或 /历史  — 查看当前会话的对话历史
    - /clear 或 /清空    — 清空当前会话的对话历史
    - exit / quit / 退出  — 退出程序

    会话 ID 固定为 "cli_session"，所有命令行交互共享同一会话历史。

    参数：
        app: 已初始化的 LangGraphRAGApp 实例
        role: 初始用户角色
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║       LangGraph RAG Agent — 交互模式                             ║
    ║       StateGraph + 多轮检索 + 多智能体 + 多轮对话                ║
    ╚══════════════════════════════════════════════════════════════════╝

    输入问题提问，输入 exit 退出。

    命令：
      /admin    切换为特权用户
      /user     切换为普通用户
      /history  查看当前会话历史
      /clear    清空当前会话历史

    示例问题：
      - 心跳间隔是多少？
      - 定位精度？几种定位方式？续航如何？（复杂问题）
      - 那它的续航呢？（多轮对话追问）
    """)

    current_role = role
    session_id = "cli_session"

    while True:
        try:
            # 动态生成提示符（显示当前角色）
            prompt = f"\n[{AccessControlFilter.get_role_description(current_role)[:4]}] >> "
            question = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "退出"):
            print("再见！")
            break

        # ---- 命令处理 ----
        if question.lower() in ("/admin", "/特权"):
            current_role = ROLE_ADMIN
            print("  已切换为特权用户（可访问所有文档）")
            continue
        if question.lower() in ("/user", "/普通"):
            current_role = ROLE_USER
            print("  已切换为普通用户（仅公开文档）")
            continue
        if question.lower() in ("/history", "/历史"):
            history = app._history_store.get(session_id, [])
            print(f"  当前会话历史（{len(history)} 条）:")
            for m in history:
                r = "用户" if m["role"] == "user" else "助手"
                print(f"    {r}: {m['content'][:80]}")
            continue
        if question.lower() in ("/clear", "/清空"):
            app._history_store[session_id] = []
            print("  已清空会话历史")
            continue

        # ---- 提问 ----
        try:
            app.query(question, role=current_role, session_id=session_id)
        except Exception as e:
            print(f"\n  处理出错: {e}")
            traceback.print_exc()


def main():
    """
    【程序入口】

    作用：解析命令行参数，初始化 Agent，启动 CLI 交互或直接提问。

    参数格式：
    - 位置参数：提问文本（可选，提供则直接回答后退出）
    - --admin：以特权用户身份运行
    - --fast：快速模式（跳过 classify 的 LLM 调用，用规则分类）

    示例：
    - python langgraph_rag_agent.py                          # 交互模式
    - python langgraph_rag_agent.py "心跳间隔是多少？"        # 直接提问
    - python langgraph_rag_agent.py "问题" --admin           # 特权用户
    - python langgraph_rag_agent.py "问题" --fast             # 快速模式
    """
    args = sys.argv[1:]
    fast_mode = "--fast" in args
    admin_mode = "--admin" in args
    role = ROLE_ADMIN if admin_mode else DEFAULT_ROLE

    # 提取直接提问（第一个不以 "--" 开头的参数）
    direct_question = None
    for arg in args:
        if not arg.startswith("--"):
            direct_question = arg
            break

    mode_label = "快速模式" if fast_mode else "完整模式"
    print(f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║       LangGraph RAG Agent — 状态图驱动                          ║
    ║       多轮检索 + 多智能体协作 + 多轮对话  [{mode_label}]                    ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # 初始化应用（连接 LLM、加载向量库、编译状态图）
    app = LangGraphRAGApp(fast_mode=fast_mode)

    if direct_question:
        # 直接提问模式：回答后退出
        app.query(direct_question, role=role)
    else:
        # 交互模式：进入命令行循环
        run_interactive(app, role)


if __name__ == "__main__":
    main()
