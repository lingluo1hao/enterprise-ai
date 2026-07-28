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
# 配置区
# ============================================================================
MAX_RETRIEVAL_ROUNDS = 3        # 多轮检索最大轮次（超过则兜底输出）
GRADE_THRESHOLD = 1             # 相关文档数达到此值即停止循环（子任务级宽松）
MAX_REVIEW_ROUNDS = 2           # 多智能体审查最大轮次（超过则强制输出）
HISTORY_MAX_TURNS = 8           # 对话历史保留的最大轮数（1 轮 = user + assistant）
HISTORY_COMPRESS_TURNS = 6      # 超过此轮数触发摘要压缩
RETRIEVE_TOP_K = 5              # 每次检索取 top_k
DOC_TRUNCATE = 350              # 文档片段截断长度（给 LLM 看的）


# ============================================================================
# State 定义 — LangGraph 核心：所有节点共享的可变状态
# ============================================================================
class AgentState(TypedDict, total=False):
    """LangGraph 全局状态，所有节点读写这个字典"""

    # --- 对话上下文 ---
    session_id: str                       # 会话 ID（多轮对话隔离）
    messages: List[Dict]                  # 完整对话历史 [{role, content}]
    query: str                            # 用户原始问题
    resolved_query: str                   # 上下文消解后的问题

    # --- 路由 ---
    query_type: str                       # simple / complex / chitchat

    # --- 多轮检索 ---
    rewritten_queries: List[str]          # 当前轮次的改写查询
    retrieved_docs: List[Any]             # 累积检索到的文档 [(Document, score)]
    doc_grades: List[bool]                # 文档相关性评分
    retrieval_iterations: int             # 检索轮次计数

    # --- 多智能体 ---
    subtasks: List[Dict]                  # planner 拆解的子任务
    research_results: List[Dict]          # 各子任务的检索结果
    review_rounds: int                    # 审查轮次计数
    review_passed: bool                   # reviewer 是否通过

    # --- 输出 ---
    answer: str                           # 最终答案
    role: str                             # 用户角色
    error: Optional[str]                  # 错误信息


# ============================================================================
# LangGraph RAG 应用主类
# ============================================================================
class LangGraphRAGApp:
    """
    LangGraph 版 RAG Agent

    把原来手写的 ReAct 循环 + Planning Agent 重构为显式状态图：
    - 每个节点是一个函数，接收 state 返回 state 更新
    - 条件边（conditional_edges）实现精细分支路由
    - 循环通过条件边回到上游节点实现（多轮检索、多智能体审查）
    """

    def __init__(self, fast_mode: bool = False):
        print("=" * 70)
        print("  LangGraph RAG Agent 初始化")
        print("=" * 70)

        # 1. LLM（复用现有 OllamaLLM）
        print("\n[1/3] 连接 LLM...")
        self.llm = OllamaLLM()

        # 2. 向量数据库（复用现有 VectorStoreManager）
        print("\n[2/3] 加载向量数据库...")
        self.vector_db = VectorStoreManager.init_vector_store()

        # 3. 缓存 + 对话历史
        print("\n[3/3] 初始化缓存与对话历史...")
        self.cache = CacheManager()
        self._history_store: Dict[str, List[Dict]] = {}  # session_id → messages
        self.fast_mode = fast_mode

        # 4. 构建 StateGraph
        print("\n[系统] 构建 LangGraph 状态图...")
        self.graph = self._build_graph()
        print("[系统] 初始化完成\n")

    # ========================================================================
    # 图构建
    # ========================================================================
    def _build_graph(self):
        """构建 LangGraph 主图：节点 + 边 + 条件边"""
        graph = StateGraph(AgentState)

        # --- 注册所有节点 ---
        graph.add_node("load_history", self.node_load_history)
        graph.add_node("classify", self.node_classify)
        graph.add_node("direct_llm", self.node_direct_llm)

        # simple 分支节点（多轮检索）
        graph.add_node("query_rewrite", self.node_query_rewrite)
        graph.add_node("retrieve", self.node_retrieve)
        graph.add_node("grade_docs", self.node_grade_docs)
        graph.add_node("rerank_mmr", self.node_rerank_mmr)
        graph.add_node("generate_simple", self.node_generate_simple)

        # complex 分支节点（多智能体）
        graph.add_node("planner", self.node_planner)
        graph.add_node("reviewer", self.node_reviewer)
        graph.add_node("writer", self.node_writer)

        # 公共出口节点
        graph.add_node("respond", self.node_respond)
        graph.add_node("save_history", self.node_save_history)

        # --- 入口边 ---
        graph.add_edge(START, "load_history")
        graph.add_edge("load_history", "classify")

        # --- 分类条件边：精细分支路由 ---
        graph.add_conditional_edges(
            "classify",
            self.route_after_classify,
            {
                "simple": "query_rewrite",
                "complex": "planner",
                "chitchat": "direct_llm",
            },
        )

        # --- simple 分支：多轮检索反馈循环 ---
        graph.add_edge("query_rewrite", "retrieve")
        graph.add_edge("retrieve", "grade_docs")
        # 评分后条件边：相关则继续，不足则回到改写（循环）
        graph.add_conditional_edges(
            "grade_docs",
            self.route_after_grade,
            {
                "relevant": "rerank_mmr",
                "rewrite": "query_rewrite",
            },
        )
        graph.add_edge("rerank_mmr", "generate_simple")
        graph.add_edge("generate_simple", "respond")

        # --- complex 分支：多智能体协作 ---
        graph.add_edge("planner", "reviewer")
        # 审查后条件边：充分则写答案，不充分则回 planner 补搜（循环）
        graph.add_conditional_edges(
            "reviewer",
            self.route_after_review,
            {
                "sufficient": "writer",
                "insufficient": "planner",
            },
        )
        graph.add_edge("writer", "respond")

        # --- chitchat 分支 ---
        graph.add_edge("direct_llm", "respond")

        # --- 公共出口 ---
        graph.add_edge("respond", "save_history")
        graph.add_edge("save_history", END)

        return graph.compile()

    # ========================================================================
    # 主图节点
    # ========================================================================

    def node_load_history(self, state: AgentState) -> dict:
        """加载当前会话的对话历史"""
        session_id = state.get("session_id", "default")
        history = self._history_store.get(session_id, [])
        print(f"  [load_history] 会话 {session_id}：{len(history)} 条历史消息")
        return {"messages": history}

    def node_classify(self, state: AgentState) -> dict:
        """
        问题分类 + 上下文消解

        判断问题类型（simple/complex/chitchat），同时如果是追问，
        把"那它的续航呢？"消解为"JM-S509 的续航如何？"。
        """
        query = state["query"]
        messages = state.get("messages", [])

        # 构建最近几轮的历史摘要
        history_text = self._format_history(messages, max_turns=4)

        if self.fast_mode or not history_text:
            # 快速模式或无历史：跳过消解，简单分类
            qtype = self._quick_classify(query)
            print(f"  [classify] 类型={qtype}（快速分类）")
            return {"query_type": qtype, "resolved_query": query}

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

        qtype, resolved = self._parse_classify(result, query)
        print(f"  [classify] 类型={qtype}, 消解问题={resolved[:40]}")
        return {"query_type": qtype, "resolved_query": resolved}

    def route_after_classify(self, state: AgentState) -> str:
        """分类后的条件边路由函数"""
        return state.get("query_type", "simple")

    def node_direct_llm(self, state: AgentState) -> dict:
        """闲聊分支：直接用 LLM 回答，不做检索"""
        query = state.get("resolved_query", state["query"])
        print(f"  [direct_llm] 闲聊直接回答")
        system = "你是友好的企业助手。简短自然地回答用户的话。"
        answer = self.llm.chat(system, query)
        return {"answer": answer}

    def node_respond(self, state: AgentState) -> dict:
        """最终回答节点（所有分支汇聚于此）"""
        answer = state.get("answer", "抱歉，我无法回答这个问题。")
        return {"answer": answer}

    def node_save_history(self, state: AgentState) -> dict:
        """保存对话历史，超窗时做摘要压缩"""
        session_id = state.get("session_id", "default")
        messages = list(state.get("messages", []))
        query = state["query"]
        answer = state.get("answer", "")

        # 追加本轮对话
        messages.append({"role": "user", "content": query})
        messages.append({"role": "assistant", "content": answer})

        # 超过最大轮数 → 压缩旧历史
        max_msgs = HISTORY_MAX_TURNS * 2
        if len(messages) > max_msgs:
            messages = self._compress_history(messages)

        self._history_store[session_id] = messages
        print(f"  [save_history] 已保存，历史 {len(messages)} 条消息")
        return {"messages": messages}

    # ========================================================================
    # simple 分支节点：多轮检索反馈循环
    # ========================================================================

    def node_query_rewrite(self, state: AgentState) -> dict:
        """
        查询重写节点

        第 1 轮：把用户问题改写为多个搜索词
        第 2+ 轮：基于上轮检索结果，换角度改写
        """
        query = state.get("resolved_query", state["query"])
        iteration = state.get("retrieval_iterations", 0)

        if iteration == 0:
            queries = self._do_rewrite(query, None)
        else:
            prev_docs = state.get("retrieved_docs", [])
            queries = self._do_rewrite(query, prev_docs)

        print(f"  [query_rewrite] 第 {iteration + 1} 轮改写: {queries}")
        return {"rewritten_queries": queries, "retrieval_iterations": iteration + 1}

    def node_retrieve(self, state: AgentState) -> dict:
        """检索节点：ChromaDB 向量检索 + 权限过滤"""
        queries = state.get(
            "rewritten_queries", [state.get("resolved_query", state["query"])]
        )
        role = state.get("role", DEFAULT_ROLE)

        new_docs = self._do_retrieve(queries, role)

        # 累积到已有文档（多轮检索会叠加）
        existing = state.get("retrieved_docs", [])
        merged = existing + new_docs

        print(f"  [retrieve] 本轮 {len(new_docs)} 个新片段，累计 {len(merged)} 个")
        return {"retrieved_docs": merged}

    def node_grade_docs(self, state: AgentState) -> dict:
        """
        文档评分节点：用 LLM 判断每个文档片段是否与问题相关

        一次性把所有文档发给 LLM 评分（减少调用次数）。
        """
        query = state.get("resolved_query", state["query"])
        docs = state.get("retrieved_docs", [])

        grades = self._do_grade(query, docs)
        relevant_count = sum(grades)
        print(f"  [grade_docs] {relevant_count}/{len(grades)} 个文档相关")
        return {"doc_grades": grades}

    def route_after_grade(self, state: AgentState) -> str:
        """
        评分后的条件边路由

        相关文档数 >= 阈值 或 检索轮次已达上限 → 进入重排序
        否则 → 回到查询改写（循环）
        """
        relevant_count = sum(state.get("doc_grades", []))
        iterations = state.get("retrieval_iterations", 0)

        if relevant_count >= GRADE_THRESHOLD or iterations >= MAX_RETRIEVAL_ROUNDS:
            return "relevant"
        return "rewrite"

    def node_rerank_mmr(self, state: AgentState) -> dict:
        """MMR 重排序：只保留相关文档，按最大边际相关性排序"""
        query = state.get("resolved_query", state["query"])
        docs = state.get("retrieved_docs", [])
        grades = state.get("doc_grades", [])

        # 过滤出相关文档
        relevant = []
        for i, (doc, score) in enumerate(docs):
            if i < len(grades) and grades[i]:
                relevant.append((doc, score))

        # 兜底：如果没有相关文档，用全部
        if not relevant:
            relevant = docs[:RETRIEVE_TOP_K]

        reranked = self._mmr_rerank(query, relevant)
        print(f"  [rerank_mmr] 重排序后 {len(reranked)} 个文档")
        return {"retrieved_docs": reranked}

    def node_generate_simple(self, state: AgentState) -> dict:
        """simple 分支最终生成：基于检索文档回答"""
        query = state.get("resolved_query", state["query"])
        docs = state.get("retrieved_docs", [])

        answer = self._do_generate(query, docs)
        print(f"  [generate_simple] 生成答案 ({len(answer)} 字)")
        return {"answer": answer}

    # ========================================================================
    # complex 分支节点：多智能体协作
    # ========================================================================

    def node_planner(self, state: AgentState) -> dict:
        """
        Planner Agent：拆解子任务 + 串行执行多轮检索

        每个子任务由内部 researcher 逻辑做多轮检索 RAG。
        （Ollama 单机串行，真正并行无性能收益，这里串行执行更可靠）
        """
        query = state.get("resolved_query", state["query"])
        review_rounds = state.get("review_rounds", 0)
        role = state.get("role", DEFAULT_ROLE)

        if review_rounds == 0:
            # 首次拆解
            subtasks = self._do_plan(query)
        else:
            # 补充拆解（审查不通过时）
            existing = state.get("research_results", [])
            subtasks = self._do_plan_supplement(query, existing)

        print(f"  [planner] 第 {review_rounds + 1} 轮：拆解出 {len(subtasks)} 个子任务")

        # 对每个子任务执行多轮检索 RAG（researcher 逻辑）
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
        Reviewer Agent：审查研究结果是否充分回答了原始问题
        """
        query = state.get("resolved_query", state["query"])
        results = state.get("research_results", [])

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
        """审查后的条件边路由"""
        if state.get("review_passed", False):
            return "sufficient"
        if state.get("review_rounds", 0) >= MAX_REVIEW_ROUNDS:
            return "sufficient"  # 超过最大轮次，强制输出
        return "insufficient"

    def node_writer(self, state: AgentState) -> dict:
        """
        Writer Agent：汇总所有子任务结果，撰写最终答案
        """
        query = state.get("resolved_query", state["query"])
        results = state.get("research_results", [])

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
    # 检索辅助方法（节点和 researcher 共用）
    # ========================================================================

    def _do_rewrite(self, query: str, prev_docs: Optional[List]) -> List[str]:
        """查询改写：第 1 轮正常改写，后续轮换角度"""
        if prev_docs is None:
            system = (
                "你是查询重写专家。将用户问题改写为 2-3 个更利于向量检索的搜索词。\n"
                "每个搜索词独占一行，不要编号，不要解释。"
            )
            result = self.llm.chat(system, query)
        else:
            prev_text = "\n".join(
                [d[0].page_content[:120] for d in prev_docs[:3]]
            )
            system = (
                "之前的检索结果不够相关。请换一个角度改写搜索词，输出 2-3 个。\n"
                "每行一个，不要编号。"
            )
            user = f"原问题: {query}\n\n已检索片段:\n{prev_text}"
            result = self.llm.chat(system, user)

        queries = [q.strip() for q in result.strip().split("\n") if q.strip()][:3]
        queries.append(query)  # 保留原始问题
        return queries

    def _do_retrieve(self, queries: List[str], role: str) -> List:
        """向量检索 + 去重 + 权限过滤"""
        all_results = []
        seen = set()
        for q in queries:
            results = self.vector_db.similarity_search_with_score(q, k=RETRIEVE_TOP_K)
            results = AccessControlFilter.filter_results(results, role)
            for doc, score in results:
                content_key = doc.page_content[:80]
                if content_key not in seen:
                    seen.add(content_key)
                    all_results.append((doc, score))
        return all_results

    def _do_grade(self, query: str, docs: List) -> List[bool]:
        """文档相关性评分：一次性发给 LLM 评分（减少调用次数）"""
        if not docs:
            return []

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

        grades = [False] * len(docs)
        if "none" not in result.lower():
            nums = re.findall(r"\d+", result)
            for n in nums:
                idx = int(n)
                if 0 <= idx < len(docs):
                    grades[idx] = True
        return grades

    def _do_generate(self, query: str, docs: List) -> str:
        """基于检索文档生成回答"""
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
        Researcher 逻辑：对单个子任务执行多轮检索 RAG

        内部循环：query_rewrite → retrieve → grade_docs
        最多 MAX_RETRIEVAL_ROUNDS 轮，相关文档达标即停止。
        """
        query = subtask.get("task", str(subtask))
        all_docs = []

        # 子任务级检索最多 2 轮（控制总耗时）
        max_rounds = min(MAX_RETRIEVAL_ROUNDS, 2)
        for iteration in range(max_rounds):
            # 1. 改写
            if iteration == 0:
                queries = self._do_rewrite(query, None)
            else:
                queries = self._do_rewrite(query, all_docs)

            # 2. 检索
            new_docs = self._do_retrieve(queries, role)
            all_docs.extend(new_docs)

            # 3. 评分
            grades = self._do_grade(query, all_docs)
            relevant_count = sum(grades)
            print(f"      第 {iteration + 1} 轮: {relevant_count} 个相关")

            if relevant_count >= GRADE_THRESHOLD:
                break

        # 4. 过滤 + 重排序
        relevant = [
            (all_docs[i][0], all_docs[i][1])
            for i in range(len(all_docs))
            if i < len(grades) and grades[i]
        ]
        if not relevant:
            relevant = all_docs[:RETRIEVE_TOP_K]
        reranked = self._mmr_rerank(query, relevant)

        # 5. 生成子答案
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
        """Planner：首次拆解子任务"""
        system = (
            "你是任务规划器。将复杂问题拆解为 2-4 个独立的子问题，"
            "每个子问题可以独立检索回答。\n\n"
            '输出 JSON: {{"subtasks": [{{"id": 1, "task": "子问题"}}]}}\n'
            "只输出 JSON，不要其他文字。"
        )
        result = self.llm.chat(system, query)
        return self._parse_json_list(result, "subtasks", query)

    def _do_plan_supplement(self, query: str, existing: List[Dict]) -> List[Dict]:
        """Planner：补充拆解（审查不通过时）"""
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
        """格式化最近几轮对话历史"""
        if not messages:
            return ""
        recent = messages[-(max_turns * 2):]
        lines = []
        for m in recent:
            role = "用户" if m["role"] == "user" else "助手"
            lines.append(f"{role}: {m['content'][:150]}")
        return "\n".join(lines)

    def _compress_history(self, messages: List[Dict]) -> List[Dict]:
        """压缩旧历史为摘要，保留最近对话"""
        keep_count = HISTORY_COMPRESS_TURNS * 2
        old = messages[:-keep_count]
        recent = messages[-keep_count:]

        old_text = "\n".join(
            [f"{m['role']}: {m['content'][:100]}" for m in old]
        )
        system = "将以下对话历史压缩为一段简短摘要（不超过100字），保留关键信息。"
        summary = self.llm.chat(system, old_text)

        print(f"  [save_history] 历史压缩: {len(old)} 条 → 1 条摘要")
        return [{"role": "system", "content": f"历史摘要: {summary}"}] + recent

    # ========================================================================
    # 通用辅助方法
    # ========================================================================

    def _quick_classify(self, query: str) -> str:
        """快速分类（不用 LLM，基于规则）"""
        q = query.lower()
        if any(w in q for w in ["你好", "谢谢", "你是谁", "hi", "hello", "再见"]):
            return "chitchat"
        # 包含多个问号或"和""且"→ complex
        if query.count("？") >= 2 or query.count("?") >= 2:
            return "complex"
        if any(w in q for w in ["和", "且", "以及", "同时"]) and any(
            w in q for w in ["？", "?"]
        ):
            return "complex"
        return "simple"

    def _parse_classify(self, result: str, fallback_query: str):
        """解析分类 LLM 输出"""
        result = result.strip()
        result = re.sub(r"```(?:json)?\s*", "", result)
        result = result.replace("```", "").strip()
        try:
            data = json.loads(result)
            return data.get("type", "simple"), data.get("resolved", fallback_query)
        except (json.JSONDecodeError, TypeError):
            match = re.search(r'\{.*\}', result, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    return data.get("type", "simple"), data.get("resolved", fallback_query)
                except (json.JSONDecodeError, TypeError):
                    pass
        # 兜底：规则分类
        return self._quick_classify(fallback_query), fallback_query

    def _parse_json_list(self, result: str, key: str, fallback_query: str) -> List[Dict]:
        """从 LLM 输出解析 JSON 列表"""
        result = result.strip()
        result = re.sub(r"```(?:json)?\s*", "", result)
        result = result.replace("```", "").strip()
        try:
            data = json.loads(result)
            items = data.get(key, [])
            if isinstance(items, list) and items:
                return items
        except (json.JSONDecodeError, TypeError):
            match = re.search(r'\{.*\}', result, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    items = data.get(key, [])
                    if isinstance(items, list) and items:
                        return items
                except (json.JSONDecodeError, TypeError):
                    pass
        # 兜底：把原问题作为单个子任务
        return [{"id": 1, "task": fallback_query}]

    def _mmr_rerank(self, query: str, results: List, k: int = 5) -> List:
        """
        MMR (Maximal Marginal Relevance) 重排序

        在相关性和多样性之间做平衡：既选与查询相关的，又避免内容重复。
        简化实现：用字符级 Jaccard 相似度衡量文档间冗余度。
        """
        if len(results) <= k:
            return results

        # 按距离排序（距离越小越相关）
        results = sorted(results, key=lambda x: x[1])
        selected = [results[0]]
        remaining = list(results[1:])
        lambda_param = 0.7

        while len(selected) < k and remaining:
            best_idx = 0
            best_score = -float("inf")
            for i, (doc, score) in enumerate(remaining):
                content = set(doc.page_content[:100])
                # 与已选文档的最大相似度
                max_sim = 0.0
                for s_doc, _ in selected:
                    s_content = set(s_doc.page_content[:100])
                    union = len(content | s_content)
                    if union > 0:
                        sim = len(content & s_content) / union
                        max_sim = max(max_sim, sim)
                # MMR 分数 = λ * 相关性 - (1-λ) * 冗余度
                relevance = 1.0 / (score + 0.01)
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i
            selected.append(remaining.pop(best_idx))

        return selected

    # ========================================================================
    # 对外查询入口
    # ========================================================================
    def query(
        self,
        question: str,
        role: str = DEFAULT_ROLE,
        session_id: str = "default",
    ) -> str:
        """
        用户提问入口

        完整流程：
          检查 Redis 缓存（命中直接返回）
          → LangGraph 状态图执行（分类 → 多轮检索/多智能体 → 生成）
          → 写入 Redis 缓存
          → 保存对话历史
        """
        total_start = time.time()

        print("\n" + "=" * 70)
        print(f"用户提问: {question}")
        role_desc = AccessControlFilter.get_role_description(role)
        print(f"用户角色: {role_desc}")
        print("=" * 70)

        # --- 缓存检查 ---
        self.cache.current_role = role
        cached = self.cache.lookup(question)
        if cached:
            print(f"\n[Cache] 命中缓存，直接返回（耗时 {time.time() - total_start:.1f}s）")
            print(f"\n{'─' * 70}\n{cached}\n{'─' * 70}")
            return cached

        # --- 执行 LangGraph 状态图 ---
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
            final_state = self.graph.invoke(initial_state, {"recursion_limit": 50})
        except Exception as e:
            print(f"\n[Error] 图执行出错: {e}")
            traceback.print_exc()
            return f"处理过程中出现错误: {e}"

        answer = final_state.get("answer", "抱歉，无法回答这个问题。")
        elapsed = time.time() - total_start

        # --- 写入缓存 ---
        self.cache.save(question, answer)

        # --- 输出结果 ---
        print(f"\n[完成] 总耗时 {elapsed:.1f}s")
        print(f"\n{'─' * 70}\n{answer}\n{'─' * 70}")

        return answer


# ============================================================================
# CLI 入口
# ============================================================================
def run_interactive(app: LangGraphRAGApp, role: str):
    """交互模式"""
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

        # 命令处理
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

        # 提问
        try:
            app.query(question, role=current_role, session_id=session_id)
        except Exception as e:
            print(f"\n  处理出错: {e}")
            traceback.print_exc()


def main():
    """主入口"""
    args = sys.argv[1:]
    fast_mode = "--fast" in args
    admin_mode = "--admin" in args
    role = ROLE_ADMIN if admin_mode else DEFAULT_ROLE

    # 提取直接提问
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

    # 初始化应用
    app = LangGraphRAGApp(fast_mode=fast_mode)

    if direct_question:
        app.query(direct_question, role=role)
    else:
        run_interactive(app, role)


if __name__ == "__main__":
    main()
