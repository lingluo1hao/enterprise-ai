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

关键依赖（最终架构）：
  - LLM：经 LLM 网关（llm_gateway）统一路由 —— qwen2:7b 负责 generate/plan，
    qwen2.5:1.5b 负责 grade/rewrite/compress（见 create_llm()）。
  - Embedding：统一使用 Ollama bge-m3（advanced_rag_agent._make_embedder，主进程不加载
    torch，只提供 Ollama 模式，无本地 SentenceTransformer 回退）。
  - 向量库：Milvus（唯一向量后端，检索接口统一）。

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

# ---- 加载 .env 文件（轻量实现，零依赖；与 advanced_rag_agent / prompt_manager / memory_store 一致） ----
def _load_dotenv(dotenv_path: str | None = None):
    """解析 .env 文件并将「未设置」的变量注入 os.environ（不覆盖已 export 的环境变量）。"""
    if dotenv_path is None:
        dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(dotenv_path):
        return
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

_load_dotenv()

import sys
import time
import json
import re
import traceback
import contextvars
from typing import TypedDict, List, Dict, Any, Optional

import warnings
warnings.filterwarnings("ignore")

from langgraph.graph import StateGraph, END, START

# 复用现有模块的配置和工具类（不重复造轮子）
from advanced_rag_agent import (
    OllamaLLM,
    create_llm,
    VectorStoreManager,
    CacheManager,
    AccessControlFilter,
    ROLE_ADMIN,
    ROLE_USER,
    DEFAULT_ROLE,
)

# Layer 2: MySQL 多层记忆持久化模块
from memory_store import MySQLMemoryStore

# 提示词工程管理 — 从 MySQL 动态加载提示词（替代硬编码）
from prompt_manager import PromptManager, get_prompt_manager

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
# 向量库（Milvus）的 similarity_search_with_score 的 k 参数。
# 5 是一个平衡值：太少可能遗漏关键信息，太多会塞满 LLM 上下文窗口。
RETRIEVE_TOP_K = 5

# RRF 跨 query 融合后的候选池宽度（两阶段精排前不截断）。
# 关键：裸原句单独检索时 gold 可能已 rank1，但 RRF 融合会被噪声改写 query 稀释到 rank>5；
# 若直接截断到 RETRIEVE_TOP_K=5 再送 reranker，gold 会在精排前被丢弃（图错 bug 根因）。
# 故先融合到较宽候选池，让 cross-encoder 在宽池中把 gold 拉回顶部，再收窄到 top5。
RETRIEVE_CANDIDATE_K = 20

# ===== 两阶段 cross-encoder 精排（reranker via llama.cpp server）=====
# 见改造方案文档 §4.2。Ollama 0.32.x 不提供 /api/rerank 路由，故 reranker 由
# llama.cpp 独立托管（已在 192.168.200.128:11436 起好并通过 /v1/rerank 验证：
# 相关 +3.07 / 无关 -6.77）。启动命令（VM 上一次性）：
#   nohup /data/llama/build/bin/llama-server \
#     -m /data/models/bge-reranker-v2-m3.Q4_K_M.gguf \
#     --reranking --port 11436 -c 2048 -b 2048 -ub 2048 &
# Reranker（两阶段精排）配置 —— 从 .env 读取，缺失时回退到以下生产默认值
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_URL = os.getenv("RERANK_URL", "http://192.168.200.128:11436/v1/rerank")
RERANK_TIMEOUT = int(os.getenv("RERANK_TIMEOUT", "20"))
RERANK_RETRIES = int(os.getenv("RERANK_RETRIES", "2"))          # 5xx 瞬时失败重试次数
RERANK_RETRY_BACKOFF = float(os.getenv("RERANK_RETRY_BACKOFF", "1.0"))  # 退避基数(秒)

# ===== 生成侧难度路由：难的 query 走 deepseek 生成，简单走本地 qwen2:7b =====
# 依据：tenant 命中硬 tenant 集合（jm/yh 等技术协议库，幻觉高发）或 query 命中
# 技术/结构类关键词（协议号/字段/组成/优先级…）。目标：难的给强模型、简单的给本地
# 模型，既控成本又压幻觉。开关/名单/模式均可用环境变量覆盖。
GEN_ROUTING_ENABLED = os.getenv("GENERATION_ROUTING_ENABLED", "true").lower() == "true"
GEN_HARD_TENANTS = {t.strip() for t in os.getenv("GENERATION_HARD_TENANTS", "jm,yh").split(",") if t.strip()}
GEN_HARD_PATTERN = os.getenv(
    "GENERATION_HARD_PATTERN",
    r"0x[0-9a-fA-F]+|由哪几部分组成|组成|分别代表|优先级|上报|字段|协议包|协议号|起始位|校验位|格式|结构",
)
_GEN_HARD_RE = re.compile(GEN_HARD_PATTERN)


def _select_gen_task(query: str, tenant: str = "") -> str:
    """难度路由：返回 'generate' 或 'generate-hard'。

    难的 query（硬 tenant 或命中技术关键词）→ generate-hard（deepseek 优先）；
    简单的 → 本地 qwen2:7b。关闭开关时一律走默认 generate。
    """
    if not GEN_ROUTING_ENABLED:
        return "generate"
    if (tenant in GEN_HARD_TENANTS) or bool(_GEN_HARD_RE.search(query or "")):
        return "generate-hard"
    return "generate"

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
    # 每个搜索词都会分别去向量库（Milvus）检索，结果合并去重。
    rewritten_queries: List[str]

    # 自进化层（方案 A）预填的"已知好改写"列表。
    # 由 node_classify 查相似 playbook 命中后注入；
    # node_query_rewrite 第 1 轮优先直接复用，跳过 LLM 改写（省 token、越用越快）。
    prefill_rewrites: List[str]

    # 累积检索到的文档列表，格式为 [(langchain Document, 距离分数), ...]。
    # 多轮检索的结果会不断追加（不会覆盖），所以叫"累积"。
    # 距离分数越小表示越相似。
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


# --------------------------------------------------------------------------- #
# figure 查询识别（heuristic）— 模块级常量与函数（在 LangGraphRAGApp 类外定义）
# --------------------------------------------------------------------------- #
# 含图类名词 / 渲染动作的 query 走 figure-aware 召回路径——专门捞图页（chunk_type="page"），
# 避免 caption 文本极短被正文页挤掉导致 [[FIG:...]] 缺失。
# 真图类名词（意图=figure）：优先返回 fig_p* 真图（流程图/架构图/示意图等）
_FIGURE_INTENT_FIGURE_KW = (
    "流程图", "架构图", "拓扑", "示意图", "框图", "时序图", "状态图",
    "类图", "原理图", "接线图", "信号流", "数据流", "消息流", "协议栈",
    "配图", "插图",
    "diagram", "chart", "graph", "architecture", "topology",
    "flowchart", "sequence",
)
# 表格类名词（意图=table）：优先返回 table_p* 表格图
_FIGURE_INTENT_TABLE_KW = (
    "表格", "表",
)
# 通用「要图」触发词（含渲染动作）：仅决定是否走 figure 召回路径，不区分意图
_FIGURE_QUERY_KEYWORDS = (
    _FIGURE_INTENT_FIGURE_KW
    + _FIGURE_INTENT_TABLE_KW
    + ("输出", "展示", "画出", "渲染", "看看", "render", "display")
)


def _is_figure_query(query: str) -> bool:
    """识别「要图」的查询（大小写不敏感）。False 时走常规检索。"""
    if not query:
        return False
    ql = query.lower()
    return any(kw in ql for kw in _FIGURE_QUERY_KEYWORDS)


def _figure_intent(query: str) -> str:
    """判断图查询意图：'figure'(真图) / 'table'(表格图) / 'any'(未明确)。

    用于图选择阶段区分 fig_p*(真图) 与 table_p*(表格图)，避免「通信流程图」
    这类真图查询被协议细节表格 chunk 的文本相关度挤掉，返回错图。
    """
    if not query:
        return "any"
    ql = query.lower()
    if any(kw in ql for kw in _FIGURE_INTENT_FIGURE_KW):
        return "figure"
    if any(kw in ql for kw in _FIGURE_INTENT_TABLE_KW):
        return "table"
    return "any"


def _fig_is_table(fp: str) -> bool:
    """路径是否指向表格图（table_p*）。真图/整页图返回 False。"""
    return "table_p" in (fp or "").lower()


def _fig_sort_key(fp: str, intent: str):
    """图选择排序键：(意图优先级, score) 中的意图优先级部分，越小越优先。
    - figure 意图：fig_p* 真图(0) > 其他真图/整页图(1) > table_p* 表格图(2)
    - table  意图：table_p*(0) > 其他真图(1) > fig_p*(2)
    - any    意图：不区分(1)
    """
    low = (fp or "").lower()
    is_fig = "fig_p" in low
    is_tab = "table_p" in low
    if intent == "figure":
        if is_fig:
            return 0
        if is_tab:
            return 2
        return 1
    if intent == "table":
        if is_tab:
            return 0
        if is_fig:
            return 2
        return 1
    return 1


def _norm_figs(val) -> List[str]:
    """归一化 figure_paths：兼容 Milvus 动态字段返回 list / 字符串 JSON 两种形态。

    - 已是 list/tuple：逐元素转 str 过滤空值；
    - 是字符串：尝试 json.loads（如 '["a.png"]'），失败则当作单路径；
    - 其余：返回 []。
    """
    if not val:
        return []
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        try:
            import json
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if x]
        except Exception:
            pass
        return [s]
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val if x]
    return []


_FIG_PLACEHOLDER_RE = re.compile(r"\[\[FIG:([^\]]+)\]\]")


def _extract_figs_from_text(text: str) -> List[str]:
    """从 chunk 文本中提取内嵌的 [[FIG:...]] 占位符路径。"""
    if not text:
        return []
    return [m.group(1) for m in _FIG_PLACEHOLDER_RE.finditer(text)]


# ============================================================================
# LangGraph RAG 应用主类
# ============================================================================
# ============================================================================
# 请求级上下文（并发隔离）
# ============================================================================
# 【为什么需要】
# LangGraphRAGApp 在 Web 模式下是「全局单例」（rag_web_server 的 orchestrator），
# 而每个 SSE 请求都会起一个后台线程调用 app.query(...)。
# 如果把「当前用户 / 租户 / 任务 ID」存成实例属性（self.user = ...），
# 两个用户同时提问时会互相覆盖：
#   T1: A 设 self.tenant_id="acme"  →  T2: B 设 self.tenant_id="globex"
#   →  A 的检索被下推到 B 的租户，A 的 checkpoint 写进 B 的 task_id。
# 这是数据串户 + 归因错乱，属于 P0 级安全问题。
#
# 【怎么修】
# 用 contextvars.ContextVar 把这些值绑定到「执行上下文」而不是「对象」。
# threading.Thread 启动时会拿到一份独立的空 Context，
# 因此每个请求线程读写这些变量互不干扰，天然隔离。
#
# 【为什么用 property 包一层而不是直接改 40 处调用点】
# 类里有 40+ 处 self.user / self.username / self.tenant_id / self.current_task_id。
# 定义成 property 后，读写语法完全不变（self.user 照旧），
# 底层自动路由到 ContextVar —— 调用点零改动，回归风险最低。
#
# 注意：default 值必须与 __init__ 中的初始赋值一致。
# 因为新线程拿到的是空 Context，读不到主线程 __init__ 时设过的值，会 fallback 到 default。
_ctx_user_id = contextvars.ContextVar("rag_user_id", default=1)
_ctx_username = contextvars.ContextVar("rag_username", default="anonymous")
_ctx_tenant_id = contextvars.ContextVar("rag_tenant_id", default="default")
_ctx_task_id = contextvars.ContextVar("rag_task_id", default=None)
# last_task_id：query() 结束时 current_task_id 会被清空（用于表示「当前无运行中任务」），
# 但前端点赞/点踩需要拿到刚才那次问答的 task_id 去关联全链路 trace，
# 所以额外留一个「最近一次任务 ID」，query() 返回后调用方仍可读到。
_ctx_last_task_id = contextvars.ContextVar("rag_last_task_id", default=None)


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

    # ------------------------------------------------------------------
    # 请求级上下文属性（读写语法不变，底层走 ContextVar，多请求并发不串号）
    # ------------------------------------------------------------------
    @property
    def user(self) -> int:
        """当前调用用户的 ID（admin_users.id）。请求级隔离。"""
        return _ctx_user_id.get()

    @user.setter
    def user(self, value: int):
        _ctx_user_id.set(value)

    @property
    def username(self) -> str:
        """当前调用用户名，用于 token 用量归因。请求级隔离。"""
        return _ctx_username.get()

    @username.setter
    def username(self, value: str):
        _ctx_username.set(value)

    @property
    def tenant_id(self) -> str:
        """当前租户，用于检索下推隔离。请求级隔离（串了会跨租户泄漏）。"""
        return _ctx_tenant_id.get()

    @tenant_id.setter
    def tenant_id(self, value: str):
        _ctx_tenant_id.set(value)

    @property
    def current_task_id(self) -> Optional[str]:
        """当前正在执行的任务 ID，节点 checkpoint 按它归档。请求级隔离。"""
        return _ctx_task_id.get()

    @current_task_id.setter
    def current_task_id(self, value: Optional[str]):
        _ctx_task_id.set(value)
        # 只记录「非空」的任务 ID：query() 收尾会把 current_task_id 置 None，
        # 但 last_task_id 要保留下来给前端做反馈关联。
        if value:
            _ctx_last_task_id.set(value)

    @property
    def last_task_id(self) -> Optional[str]:
        """最近一次问答的任务 ID（只读）。供前端点赞/点踩关联全链路 trace。"""
        return _ctx_last_task_id.get()

    def __init__(self, fast_mode: bool = False):
        """
        初始化 LangGraph RAG Agent。

        执行顺序：
        1. 连接 Ollama LLM（复用 advanced_rag_agent 的 OllamaLLM 类）
        2. 加载向量数据库（复用 VectorStoreManager，唯一后端 Milvus）
        3. 初始化 Redis 缓存 + 内存对话历史存储
        4. 构建 LangGraph 状态图（注册节点 + 连线 + 条件边）

        参数：
            fast_mode: True 时 classify 使用规则分类（不调 LLM），速度更快但准确性降低。
                       适合对响应速度要求高、问题类型简单的场景。
        """
        print("=" * 70)
        print("  LangGraph RAG Agent 初始化")
        print("=" * 70)

        # 1. LLM（走企业级网关，初始化失败自动回退单模型直连）
        # create_llm() 返回的对象一定满足 BaseLLM 接口，
        # 所以本文件里 11 处 self.llm.chat(system, user) 一行都不用改。
        print("\n[1/3] 连接 LLM...")
        self.llm = create_llm()
        # 下面三行走的是本类的 property setter，实际写入 ContextVar（请求级隔离）。
        # 这里赋值只是让 CLI 模式有个明确起点；Web 模式每次 query() 都会按真实登录态覆盖。
        self.user = 1              # 当前调用用户的 ID（admin_users.id）
        self.username = "anonymous"  # 当前调用用户的用户名，用于 token 用量归因
        self.tenant_id = "default" # 当前调用用户所属租户

        # 2. 向量数据库（复用现有 VectorStoreManager）
        # VectorStoreManager 封装了 Milvus 的初始化、文档索引、向量检索，
        # 本项目仅此一个后端。
        # init_vector_store() 会扫描 knowledge/ 目录，首次运行时自动构建索引。
        print("\n[2/3] 加载向量数据库...")
        self.vector_db = VectorStoreManager.init_vector_store()

        # 3. 缓存 + 对话历史 + MySQL 持久化记忆
        # CacheManager: Redis 两级缓存（精确匹配 + 语义匹配）— Layer 3
        # MySQLMemoryStore: MySQL 持久化记忆 — Layer 2（★新增）
        #   - 对话历史持久化（替代内存 _history_store）
        #   - 任务断点快照（每个节点执行后保存 state）
        #   - 任务队列管理（running/completed/interrupted）
        # _active_context: 内存活跃上下文 — Layer 1（保留，用于加速）
        print("\n[3/5] 初始化缓存与多层记忆...")
        self.cache = CacheManager()
        self.memory_store = MySQLMemoryStore()
        self._active_context: Dict[str, List[Dict]] = {}  # Layer 1: 内存加速
        self.current_task_id = None  # 当前正在执行的任务 ID（property → ContextVar）
        self.fast_mode = fast_mode

        # 4. 提示词管理器（从 MySQL 动态加载，DB 不可用时回退到默认值）
        print("\n[4/5] 初始化提示词管理器...")
        self.pm = get_prompt_manager()
        # 首次运行：将默认提示词导入 MySQL
        if self.pm.available:
            self.pm.import_defaults()

        # 4.5 自进化层（方案 A：嫁接 Hermes 式自进化 —— 越用越快）
        # 复用 self.vector_db（Milvus 客户端 + Ollama embedding），独立集合 skill_playbooks。
        # 任何失败都降级跳过，绝不影响主问答链路。
        self.playbook_store = None
        self._Extractor = None
        try:
            from evolution import PlaybookStore, Extractor
            self.playbook_store = PlaybookStore(self.vector_db)
            self._Extractor = Extractor
            print("  [4.5] 自进化层 PlaybookStore 已挂载（skill_playbooks）")
        except Exception as e:
            print(f"  [4.5] ⚠ 自进化层挂载失败(降级,不影响主流程): {e}")

        # 5. 构建 StateGraph
        print("\n[5/5] 构建 LangGraph 状态图...")

        # 服务重启时，将上次未完成的 running 任务标记为 interrupted
        # 这样用户下次登录时能看到准确的中断状态
        if self.memory_store.available:
            self.memory_store.mark_interrupted_tasks(None)
            print("  [记忆] 已检查并标记上次中断的任务")

        # _build_graph() 定义所有节点、边、条件边，最后 compile() 生成可执行的图。
        # compile() 会校验图结构（无孤立节点、无死循环等），返回 CompiledStateGraph。
        # 节点函数通过 _wrap_node_with_checkpoint 包装，每个节点执行后自动保存断点到 MySQL。
        self.graph = self._build_graph()
        print("[系统] 初始化完成\n")

    # ========================================================================
    # 断点包装器 — 每个节点执行后自动保存 state 到 MySQL
    # ========================================================================

    def _wrap_node_with_checkpoint(self, node_name: str, fn):
        """
        【断点包装器：给节点函数包一层自动存盘逻辑】

        作用：每个 LangGraph 节点执行完毕后，自动把当前 state 保存到 MySQL。
        如果服务在此之后宕机，下次可以通过 load_latest_checkpoint 恢复到这个状态。

        原理：
        包装器是一个高阶函数：接收原始节点函数 fn，返回一个新的函数 wrapped。
        wrapped 先调用原始 fn 执行节点逻辑，拿到返回的 state 增量，
        然后把「旧 state + 增量」合并后保存到 MySQL task_checkpoints 表。

        为什么用包装器而不是手动在每个节点里写存盘代码？
        - DRY 原则：14 个节点都需要存盘，包装器写一次即可
        - 可维护：未来加减节点不需要改存盘逻辑
        - 可测试：包装器与业务逻辑解耦

        参数：
            node_name: 节点名称（如 "classify", "retrieve"）
            fn: 原始节点函数
        返回：
            包装后的函数（签名与 fn 相同）
        """
        def wrapped(state):
            # 1. 调用原始节点函数，执行业务逻辑
            result = fn(state)

            # 2. 如果当前有活跃任务，保存断点快照到 MySQL
            if self.current_task_id and self.memory_store.available:
                # 合并旧 state 和节点返回的增量
                merged_state = {**state, **result}
                self.memory_store.save_checkpoint(
                    thread_id=self.current_task_id,
                    session_id=state.get("session_id", "default"),
                    node_name=node_name,
                    state=merged_state,
                    user_id=self.user,
                )
                print(f"  [checkpoint] {node_name} → 已保存断点到 MySQL")

            # 3. 返回原始结果（不影响 LangGraph 的 state 合并逻辑）
            return result

        return wrapped

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
          load_history  ←── 从 MySQL/内存 加载该会话的历史消息
            │
            ▼
          classify      ←── LLM 判断问题类型 + 上下文消解（追问补全）
            │
            ├── query_type == "simple" ──► query_rewrite（多轮检索反馈循环）
            │                                  │
            │                                  ▼
            │                               retrieve（向量库 Milvus 检索）
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
        # 返回的 dict 会被 LangGraph 自动合并到全局 state 中。
        #
        # ★ 断点机制：所有节点通过 _wrap_node_with_checkpoint 包装，
        # 每个节点执行后自动把 state 保存到 MySQL task_checkpoints 表。
        # 如果服务宕机，下次可通过 load_latest_checkpoint 恢复。

        # --- 公共入口节点 ---
        graph.add_node("load_history", self._wrap_node_with_checkpoint("load_history", self.node_load_history))
        graph.add_node("classify", self._wrap_node_with_checkpoint("classify", self.node_classify))
        graph.add_node("direct_llm", self._wrap_node_with_checkpoint("direct_llm", self.node_direct_llm))

        # --- simple 分支节点（多轮检索） ---
        graph.add_node("query_rewrite", self._wrap_node_with_checkpoint("query_rewrite", self.node_query_rewrite))
        graph.add_node("retrieve", self._wrap_node_with_checkpoint("retrieve", self.node_retrieve))
        graph.add_node("grade_docs", self._wrap_node_with_checkpoint("grade_docs", self.node_grade_docs))
        graph.add_node("rerank_mmr", self._wrap_node_with_checkpoint("rerank_mmr", self.node_rerank_mmr))
        graph.add_node("generate_simple", self._wrap_node_with_checkpoint("generate_simple", self.node_generate_simple))

        # --- complex 分支节点（多智能体） ---
        graph.add_node("planner", self._wrap_node_with_checkpoint("planner", self.node_planner))
        graph.add_node("reviewer", self._wrap_node_with_checkpoint("reviewer", self.node_reviewer))
        graph.add_node("writer", self._wrap_node_with_checkpoint("writer", self.node_writer))

        # --- 公共出口节点 ---
        graph.add_node("respond", self._wrap_node_with_checkpoint("respond", self.node_respond))
        graph.add_node("save_history", self._wrap_node_with_checkpoint("save_history", self.node_save_history))

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

        作用：根据 session_id 加载该会话的历史消息。

        原理 — 三层记忆架构：
        1. 先查 Layer 1（内存 _active_context）：最快，用于加速单次会话
        2. 再查 Layer 2（MySQL chat_messages 表）：持久化，服务重启不丢失
        3. 如果两层都没有，返回空列表（首次对话）

        为什么需要两层？
        - 内存层：读写 < 0.1ms，但服务重启丢失
        - MySQL 层：读写 ~1-5ms，但持久化
        - 组合使用：首次从 MySQL 加载到内存，后续直接读内存

        输入：
            state["session_id"] — 会话标识符
        输出：
            {"messages": [...]} — 该会话的历史消息列表
        """
        session_id = state.get("session_id", "default")

        # Layer 1: 先查内存（如果同会话已经加载过，直接用缓存）
        if session_id in self._active_context:
            history = self._active_context[session_id]
            print(f"  [load_history] 会话 {session_id}：{len(history)} 条（内存命中）")
            return {"messages": history}

        # Layer 2: 查 MySQL（服务重启后的首次加载，或新会话）
        history = self.memory_store.load_messages(session_id, user_id=self.user)
        # 写入内存缓存，后续同会话直接读内存
        self._active_context[session_id] = history
        print(f"  [load_history] 会话 {session_id}：{len(history)} 条（MySQL 加载）")
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

        # —— 自进化（方案 A）：查相似 playbook，命中则预填已知好 rewrite ——
        prefill: List[str] = []
        used_playbook_pk: Optional[str] = None
        try:
            store = getattr(self, "playbook_store", None)
            if store is not None:
                hit = store.query_similar(query, self.tenant_id, top_k=1)
                if hit and hit.get("rewrite_text"):
                    prefill = json.loads(hit["rewrite_text"])
                    used_playbook_pk = hit.get("pk")
                    # 命中即复用：success_count +1（越用越快，强化自进化 #168）
                    try:
                        store.patch_success(used_playbook_pk)
                    except Exception as _e:
                        print(f"  [classify] patch_success 异常(忽略): {_e}")
                    print(f"  [classify] ♻ 命中经验 playbook(相似度={hit['score']:.2f})，预填 rewrite: {prefill}")
        except Exception as e:
            print(f"  [classify] 经验查询异常(忽略): {e}")

        # —— 确定性修复：多问句问题强制走 complex 且跳过历史消解 ——
        # 同一问题在「有/无历史」两种状态下会被 LLM 消解出不同 query，
        # 导致两次检索结果不一致。多问号问题几乎都是自包含复杂问题，
        # 直接判定 complex 并用原始 query，消除历史依赖带来的随机性。
        if query.count("？") >= 2 or query.count("?") >= 2:
            print(f"  [classify] 类型=complex（多问句强制，跳过 LLM 消解）")
            return {"query_type": "complex", "resolved_query": query,
                    "prefill_rewrites": prefill, "used_playbook_pk": used_playbook_pk}

        # 构建最近 4 轮对话历史的文本摘要
        history_text = self._format_history(messages, max_turns=4)

        if self.fast_mode or not history_text:
            # 快速模式或无历史：跳过 LLM，用规则快速分类
            qtype = self._quick_classify(query)
            print(f"  [classify] 类型={qtype}（快速分类）")
            return {"query_type": qtype, "resolved_query": query,
                    "prefill_rewrites": prefill, "used_playbook_pk": used_playbook_pk}

        # 从提示词管理器获取 classify 模板
        prompt = self.pm.get_prompt("classify")
        system = prompt["system"]
        user = self.pm.format_user_message(
            prompt["user_template"],
            history=history_text, query=query
        )
        result = self.llm.chat(system, user, task="classify", user=self.username)

        # 解析 LLM 输出的 JSON（含容错兜底）
        qtype, resolved = self._parse_classify(result, query)
        print(f"  [classify] 类型={qtype}, 消解问题={resolved[:40]}")
        return {"query_type": qtype, "resolved_query": resolved,
                "prefill_rewrites": prefill, "used_playbook_pk": used_playbook_pk}

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
        闲聊问题不需要查找企业文档，调用 Milvus 检索纯属浪费。
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
        prompt = self.pm.get_prompt("chitchat")
        system = prompt["system"]
        user = self.pm.format_user_message(prompt["user_template"], query=query)
        answer = self.llm.chat(system, user, task="direct", user=self.username)
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
    # 对话历史统一写入（P0 止血 3.3：缓存命中与正常路径共用，杜绝历史空洞）
    # ========================================================================

    def _append_history(self, session_id, question, answer, user_id: int = 0, cached=False):
        """
        把一轮问答写入 Layer 1（内存）+ Layer 2（MySQL）。

        关键：缓存命中路径也必须调用本方法（P0 止血 3.3）。
        原本缓存命中直接 return，跳过 node_save_history，导致该轮永不入库——
        用户问 A（未命中）→ 问 B（命中）→ 问 C 时引用「刚才的 B」，历史里却没有 B。

        压缩发生时，摘要同时落库 chat_summaries（P0 止血 3.4），重启不丢。
        """
        ctx = self._active_context.get(session_id, [])
        ctx = ctx + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

        # 超过窗口则压缩，压缩产物落库
        max_msgs = HISTORY_MAX_TURNS * 2
        if len(ctx) > max_msgs:
            old_count = len(ctx)
            ctx = self._compress_history(ctx)
            if self.memory_store.available:
                summary_text = ctx[0]["content"] if ctx and ctx[0].get("role") == "system" else ""
                covers_to = self.memory_store.get_last_message_id(session_id, user_id)
                if summary_text:
                    self.memory_store.save_summary(
                        user_id, session_id, summary_text, covers_to, old_count, importance=3
                    )

        # Layer 2: 写入 MySQL（持久化，重启不丢）
        if self.memory_store.available:
            self.memory_store.save_message(session_id, "user", question, user_id=user_id)
            self.memory_store.save_message(session_id, "assistant", answer, user_id=user_id)

        # Layer 1: 写入内存（加速后续读取）
        self._active_context[session_id] = ctx
        return ctx

    # ========================================================================
    # 第 14 号节点：save_history — 保存对话历史
    # ========================================================================

    def node_save_history(self, state: AgentState) -> dict:
        """
        【节点：保存对话历史】

        作用：将本轮问答追加到会话历史中，同时写入 MySQL 和内存，超过窗口限制时自动压缩。

        原理 — 三层记忆架构写入策略：
        1. 写 Layer 2（MySQL）：持久化保存，服务重启不丢失
        2. 写 Layer 1（内存）：加速后续同会话的读取
        3. 超过 HISTORY_MAX_TURNS * 2 条消息时触发压缩

        为什么需要压缩？
        LLM 的上下文窗口有限（qwen2:7b 约 32K tokens）。
        如果不压缩，多轮对话会快速占满窗口，导致 LLM "忘记"早期的关键信息。
        压缩策略：用 LLM 把旧消息总结为一段简短摘要（≤100 字），
        以 system 角色的消息形式放在对话历史的开头。

        输入：
            state 中的所有相关字段
        输出：
            {"messages": [...]} — 更新后的对话历史

        注意：实际写入统一走 self._append_history（与缓存命中路径共用，P0 止血 3.3）。
        """
        session_id = state.get("session_id", "default")
        query = state["query"]
        answer = state.get("answer", "")

        # 统一走 _append_history：本轮问答写 L1 + L2，超窗压缩并落库摘要
        messages = self._append_history(session_id, query, answer, user_id=self.user)
        print(f"  [save_history] 已保存到 MySQL + 内存，历史 {len(messages)} 条消息")

        # —— 自进化（方案 A / 强化 #168）：沉淀成功经验 + 失败样本闭环 ——
        try:
            store = getattr(self, "playbook_store", None)
            ext = getattr(self, "_Extractor", None)
            if store is not None and ext is not None:
                # 三级成功信号评估：是否值得作为正向经验沉淀
                ok, _level = ext.evaluate_success(state)
                if ok:
                    pb = ext.extract(state, self.tenant_id, self.username)
                    if pb is not None:
                        # save_or_merge：同问题去重合并，避免重复插（越用越快）
                        store.save_or_merge(pb)
                else:
                    # 检索未走通 / 答案级或反馈级为负 -> 沉淀为负样本到 bad_cases，
                    # 形成「失败样本 -> triage 诊断 -> 修复 -> 回归验证」自进化闭环
                    fail = ext.extract_failure(state, self.tenant_id, self.username)
                    if fail is not None:
                        ms = getattr(self, "memory_store", None)
                        if ms is not None and getattr(ms, "available", False):
                            ms.add_bad_case(
                                query=fail["query"], source=fail["source"],
                                suite=fail["suite"], expected=fail["expected"],
                                root_cause=fail["root_cause"],
                                diagnosis=fail["diagnosis"], status="open",
                            )
                            print(f"  [evolution] ⚠ 本次未走通，已沉淀负样本到 bad_cases: "
                                  f"{fail['query'][:40]}")
        except Exception as e:
            print(f"  [evolution] 沉淀异常(忽略): {e}")

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

        if iteration == 0 and state.get("prefill_rewrites"):
            # 第 1 轮：复用经验 playbook 已知好的首轮改写，跳过 LLM 改写（省 token）
            queries = state["prefill_rewrites"]
            print(f"  [query_rewrite] ♻ 复用经验 playbook 首轮改写，跳过 LLM: {queries}")
        elif iteration == 0:
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

        作用：用改写后的查询词去向量库（Milvus）做向量相似度检索，并对结果做权限过滤。

        原理：
        每个改写查询词独立发送到向量库（Milvus）的 similarity_search_with_score()。
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
        role = state.get("role", DEFAULT_ROLE)

        answer = self._do_generate(query, docs, role=role)
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

        prompt = self.pm.get_prompt("reviewer_check")
        system = prompt["system"]
        user = self.pm.format_user_message(
            prompt["user_template"],
            query=query, results_text=results_text
        )
        result = self.llm.chat(system, user, task="review", user=self.username)

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

        # 如果没有任何子任务结果，直接返回“未检索到相关内容”
        if not results:
            return {"answer": "未检索到与问题相关的文档内容，无法回答。"}

        # 如果所有子任务都返回“未检索到相关内容”，则无需调用 LLM，直接兜底
        all_unknown = all(
            "未检索到" in str(r.get("answer", "")) for r in results
        )
        if all_unknown:
            return {"answer": "未检索到与问题相关的文档内容，无法回答。"}

        # 格式化所有子任务结果为统一格式
        results_text = "\n\n".join(
            [f"【{r['subtask']}】\n{r['answer']}" for r in results]
        )

        prompt = self.pm.get_prompt("writer_compose")
        system = prompt["system"]
        user = self.pm.format_user_message(
            prompt["user_template"],
            query=query, results_text=results_text
        )
        answer = self.llm.chat(system, user, task="write", user=self.username)
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
            prompt = self.pm.get_prompt("rewrite_first")
            system = prompt["system"]
            user = self.pm.format_user_message(prompt["user_template"], query=query)
            result = self.llm.chat(system, user, task="rewrite", user=self.username)
        else:
            # 第 N 轮：基于之前的检索结果，换角度改写
            # 只取前 3 个文档的前 120 字符，避免 prompt 过长
            prev_text = "\n".join(
                [d[0].page_content[:120] for d in prev_docs[:3]]
            )
            prompt = self.pm.get_prompt("rewrite_retry")
            system = prompt["system"]
            user = self.pm.format_user_message(
                prompt["user_template"],
                query=query, prev_text=prev_text
            )
            result = self.llm.chat(system, user, task="rewrite", user=self.username)

        # 解析 LLM 输出：按行拆分，取前 3 个非空行
        # 去「1. 2. 3.」或「1、2、3、」编号噪声（LLM 常给有序列表，编号会干扰检索）
        queries = [re.sub(r"^\d+[.、]\s*", "", q.strip())
                   for q in result.strip().split("\n") if q.strip()][:3]
        # 兜底：始终保留原始问题，且置于列表首位（原句是最高精度信号，
        # 放末尾会被改写 query 抢注分数而沉底；见改造方案 §4.1）
        queries.insert(0, query.strip())
        return queries

    def _do_retrieve(self, queries: List[str], role: str) -> List:
        """
        【辅助：向量检索 + RRF 跨 query 融合 + 权限过滤 + 两阶段精排】

        作用：对多个查询词分别做向量检索，用 RRF（Reciprocal Rank Fusion）
        跨 query 融合排名，再做可选 cross-encoder 精排，最后权限过滤、去重。

        原理：
        1. 每个查询词独立调用 similarity_search_with_score() 取 top-k 个文档
        2. 不同 query 的距离分数分布不同、不可直接比较（改造方案 §5.2），
           改用 RRF：每个 query 内部按返回顺序给 1/(k+rank+1) 的分，跨 query 累加
        3. 原句已在 _do_rewrite 中置顶（insert(0)），其召回的 gold 文档在首个
           query 即 rank 1，RRF 累加后稳居顶部 —— 根治「精准文档被改写 query 挤沉」
        4. （可选）cross-encoder reranker 对 RRF 候选池用原句精排，进一步提升精度

        参数：
            queries: 搜索词列表（第 0 项始终为原句）
            role: 用户角色（"admin" 或 "user"，决定可见文档范围）
        返回：
            [(Document, score), ...] — 融合 + 精排后的文档列表（score 越小越相关）
        """
        # ① 收集每个 query 的排名（带权限过滤）
        per_query_results = []
        for q in queries:
            results = self.vector_db.similarity_search_with_score(
                q, k=RETRIEVE_TOP_K, filter_role=role,
                user_id=self.user, tenant_id=self.tenant_id)
            results = AccessControlFilter.filter_results(results, role)
            per_query_results.append(results)

        # ② RRF 跨 query 融合（原生句置顶 → gold 文档浮顶）
        #    融合到较宽候选池 RETRIEVE_CANDIDATE_K，避免 gold 在进 reranker 前被 top5 截断丢弃
        rrf_results = self._rrf_fuse_queries(per_query_results, RETRIEVE_CANDIDATE_K)

        # ③④ 两阶段精排（可选，失败优雅回退到 RRF 顺序）
        candidate = self._rerank(queries[0] if queries else "", rrf_results, RETRIEVE_TOP_K)

        # ⑤ figure-aware 二次召回：原句（queries[0]）含图关键词时，图页顶到最前
        original_q = queries[0] if queries else ""
        figure_results = []
        if _is_figure_query(original_q):
            try:
                figure_results = self.vector_db.search_figure_pages(
                    original_q, k=2, filter_role=role,
                    user_id=self.user, tenant_id=self.tenant_id
                )
                print(f"  [retrieve] figure-aware topk={len(figure_results)} (query={original_q[:30]!r})")
            except Exception as e:
                print(f"  [retrieve] figure-page 召回失败(忽略): {e}")
        if figure_results:
            merged = []
            seen2 = set()
            for doc, score in figure_results:
                key = doc.page_content[:80]
                if key not in seen2:
                    seen2.add(key)
                    merged.append((doc, score))
            for doc, score in candidate:
                key = doc.page_content[:80]
                if key not in seen2:
                    seen2.add(key)
                    merged.append((doc, score))
            return merged
        return candidate

    def _rrf_fuse_queries(self, per_query_results, top_k, rrf_k=60):
        """
        RRF（Reciprocal Rank Fusion）跨 query 融合。

        不同 query 的距离分布不同、不可直接比较（改造方案 §5.2），故用排名倒数而非
        原始距离。原句置顶后，其召回的 gold 文档在第一个 query 即 rank 1，累加分最高，
        浮到顶部（改造方案 §5.3 的小算例）。

        返回 [(Document, score), ...]，score = -fused（越小越相关），与下游约定一致。
        """
        fused = {}
        docs = {}
        for qres in per_query_results:
            for rank, (doc, _) in enumerate(qres):
                key = doc.page_content[:80]
                docs[key] = doc
                fused[key] = fused.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        ranked = sorted(fused, key=lambda h: fused[h], reverse=True)
        return [(docs[h], -fused[h]) for h in ranked[:top_k]]

    def _rerank(self, query: str, docs: List, top_n: int) -> List:
        """
        两阶段 cross-encoder 精排（改造方案 §4.2）。

        用 bge-reranker-v2-m3（经 llama.cpp server 的 OpenAI 兼容 /v1/rerank 接口）
        对 RRF 候选池做精排。原句 query 直接作为 reranker 的 query（最高精度信号）。

        reranker 服务不可用时优雅回退到 RRF 顺序，不影响主链路。

        参数：
            query: 原始问题
            docs: RRF 融合后的候选 [(Document, score), ...]
            top_n: 返回条数
        返回：
            精排后的 [(Document, score), ...]
        """
        if not RERANK_ENABLED or not docs:
            return docs[:top_n]
        try:
            import requests
            # 防御性清洗：过滤空串/None，截断超长文档，避免 reranker 报 500。
            # 截断只影响"发给 reranker 打分的内容"，返回时经 idx_map 取回完整原始
            # Document，不丢上下文。
            RERANK_MAX_CHARS = 3000
            cleaned, idx_map = [], []
            for i, (d, _s) in enumerate(docs):
                text = (getattr(d, "page_content", None) or "").strip()
                if not text:
                    continue
                if len(text) > RERANK_MAX_CHARS:
                    text = text[:RERANK_MAX_CHARS]
                cleaned.append(text)
                idx_map.append(i)
            if not cleaned:
                return docs[:top_n]
            payload = {
                "model": "bge-reranker-v2-m3",
                "query": (query or "").strip()[:512],
                "documents": cleaned,
            }
            last_err = None
            for _attempt in range(RERANK_RETRIES + 1):
                try:
                    resp = requests.post(RERANK_URL, json=payload, timeout=RERANK_TIMEOUT)
                    if resp.status_code >= 500:
                        # 瞬时 500（VM 上 reranker 抖动）：退避后重试，不直接回退 RRF
                        last_err = f"HTTP {resp.status_code}"
                        if _attempt < RERANK_RETRIES:
                            time.sleep(RERANK_RETRY_BACKOFF * (_attempt + 1))
                            continue
                        break
                    resp.raise_for_status()
                    order = sorted(resp.json().get("results", []),
                                   key=lambda x: x["relevance_score"], reverse=True)
                    reranked = [docs[idx_map[it["index"]]] for it in order[:top_n]]
                    print(f"  [rerank] {len(docs)} 候选 → 精排 {len(reranked)} 条 (query={query[:30]!r})")
                    return reranked
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                    if _attempt < RERANK_RETRIES:
                        time.sleep(RERANK_RETRY_BACKOFF * (_attempt + 1))
                        continue
                    break
            print(f"  [rerank] 失败，回退 RRF 顺序: {last_err}")
            return docs[:top_n]
        except Exception as e:
            # 覆盖 import requests / 候选清洗等前置阶段的异常，同样优雅回退 RRF
            print(f"  [rerank] 失败，回退 RRF 顺序: {e}")
            return docs[:top_n]

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
        prompt = self.pm.get_prompt("grade_docs")
        system = prompt["system"]
        user = self.pm.format_user_message(
            prompt["user_template"],
            query=query, docs="\n".join(doc_texts)
        )
        result = self.llm.chat(system, user, task="grade", user=self.username)

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

    def _do_generate(self, query: str, docs: List, role: str = None) -> str:
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
        # 文档为空或没有实质内容时，直接返回“未检索到相关内容”，不再调用 LLM
        # 避免 LLM 在零上下文或弱上下文下胡编乱造
        if not docs:
            return "未检索到与问题相关的文档内容，无法回答。"

        # 构建上下文：最多 5 个文档，每个截断到 DOC_TRUNCATE 字符。
        # 带 figure_paths 的文档放宽到 2000 字符，避免 PyPDF 抽取的 caption
        # （如「I 通信流程图」）落在截断点之后被吃掉，导致 LLM 看不到图上下文。
        # 同时剥离 chunk 文本里内嵌的 [[FIG:assets/...]] 占位符——
        # 图由本方法末尾「服务端确定性追加」逻辑统一处理（1581 行起），
        # 不应出现在喂给 LLM 的正文里，否则 LLM 会把它当噪音或原样吐回答案。
        import re as _re
        _FIG_RE = _re.compile(r"\[\[FIG:[^\]]*\]\]")

        def _clean_body(text: str) -> str:
            return _FIG_RE.sub("", text).strip()

        # 章节优先（B3）：把「与问题同属一个章节」的文档排到最前，
        # 避免被同文档其他章节（如命令集表格）的噪声带偏。
        # 匹配规则：doc 的 section_path（"§" 连接）中包含 query 里的连续中文字片段越多，
        # 优先级越高；query 不含章节名时（如"MCC 是什么"）全部为 0，退化为按原 score 序。
        _q_segs = [s for s in re.findall(r"[\u4e00-\u9fff]{2,}", query)]

        def _section_priority(doc) -> int:
            sp = doc.metadata.get("section_path", "") or ""
            if not sp:
                return 0
            return sum(1 for seg in sp.split("§") if any(seg in qseg for qseg in _q_segs))

        docs = sorted(docs, key=lambda d: _section_priority(d[0]), reverse=True)

        # ---- 分级上下文预算（按上面 B3 排序后的名次分配字数）----------------
        # 旧逻辑是「带图就放宽到 2000 字」，本意只覆盖极少数图页；但 chunker 会把
        # 整章图片清单透传给该章每一个子 chunk（实测 97% 的 chunk 都带图），
        # 这条规则等于对全体生效。再叠加 _parse_hits 优先返回整章 parent_content
        # （中位数 5468 字，子 chunk 自身只有 269 字），上下文直接涨到 8348 字符
        # / 4000+ token。
        # 致命后果：Ollama 默认 num_ctx=2048，超出部分**从 prompt 开头静默截断**，
        # 而 B3 恰好把最相关的章节排在第一位 —— 正确答案第一个被丢掉，模型只剩
        # 末尾的无关章节可看，于是答成「心跳/补传/GPRS 数传」。
        # 现在按名次给预算：命中章节给足，旁证给少量，兜底给最少。
        RANK_BUDGET = (1200, 500, 500, 350, 350)  # 合计 ≈ 2900 字，约 1700 token
        parts = []
        top_docs = docs[:5]
        for i, d in enumerate(top_docs):
            doc = d[0]
            src = os.path.basename(doc.metadata.get("source", "未知"))
            page = doc.metadata.get("page")
            label = f"[{src} 第{page}页]" if page else f"[{src}]"
            trunc = RANK_BUDGET[i] if i < len(RANK_BUDGET) else DOC_TRUNCATE
            body = _clean_body(doc.page_content[:trunc])
            parts.append(f"{label} {body}")
        context = "\n\n".join(parts)
        # 过滤后上下文仍为空（理论上不会，但做兜底）
        if not context.strip():
            return "未检索到与问题相关的文档内容，无法回答。"

        prompt = self.pm.get_prompt("generate_answer")
        system = prompt["system"]
        user = self.pm.format_user_message(
            prompt["user_template"],
            query=query, context=context
        )
        # ---- 生成侧难度路由：难的 query 走 deepseek 生成，简单走本地 qwen2:7b ----
        gen_task = _select_gen_task(query, getattr(self, "tenant_id", "") or "")
        answer = self.llm.chat(system, user, task=gen_task, user=self.username)
        if gen_task != "generate":
            print(f"  [generate] 难度路由 → task={gen_task} (tenant={getattr(self, 'tenant_id', '')!r})")

        # ===== 图渲染：服务端确定性追加，取「本次检索涉及的最相关图」拼到答案末尾 =====
        # 之前把 [[FIG:...]] 放进 context 指望 LLM 原样保留，但 LLM 常把占位符当噪音删掉，
        # 导致前端拿不到图。这里直接把「本次检索到的图页 + 图查询兜底召回到的图页」
        # 收集并按相似度排序，去重后取前 MAX_FIGS 张（当前 2）拼到答案末尾，避免图片泛滥成噪音。
        # LLM 走到降级分支（全链失败）时，答案是网关的固定文案，与文档无关，
        # 此时再追加图片只会制造「一句道歉 + 一堆无关截图」的观感，直接返回。
        if answer and answer.startswith("抱歉，当前模型服务繁忙"):
            print("  [generate] LLM 降级，跳过图页追加")
            return answer

        figs_with_score = []  # (fp, score)
        # 只从**真正进入 context 的 top_docs** 收集，不再扫全量 docs：
        # chunker 把整章图透传给每个子 chunk，5 个文档去重后能收出 16 张图，
        # 全糊到答案末尾就是「一问基站格式、末尾挂满报警/心跳表格截图」。
        for d in top_docs:
            sc = d[1] if len(d) > 1 else 0.0
            doc = d[0]
            # 元数据 figure_paths + 文本中内嵌 [[FIG:...]] 双重收集
            fig_sources = set(_norm_figs(doc.metadata.get("figure_paths")))
            fig_sources.update(_extract_figs_from_text(doc.page_content or ""))
            for fp in fig_sources:
                figs_with_score.append((fp, sc))
        # figure 查询再兜底一次：覆盖 grade_docs 把 page chunk 过滤掉、主检索未带图页的情况
        if _is_figure_query(query):
            try:
                rescued = self.vector_db.search_figure_pages(
                    query, k=4, filter_role=role,
                    user_id=self.user, tenant_id=self.tenant_id
                )
                for doc, sc in rescued:
                    fps = set(_norm_figs(doc.metadata.get("figure_paths")))
                    fps.update(_extract_figs_from_text(doc.page_content or ""))
                    for fp in fps:
                        figs_with_score.append((fp, sc))
            except Exception as e:
                print(f"  [generate] figure 兜底追加失败(忽略): {e}")

        if figs_with_score:
            # 同一 fp 保留最小 score（最相关）
            best = {}
            for fp, sc in figs_with_score:
                if fp not in best or sc < best[fp]:
                    best[fp] = sc
            # 过滤掉 LLM 已经在答案里保留的占位符，避免重复追加
            existing_figs = set(_extract_figs_from_text(answer))
            remaining = [(fp, sc) for fp, sc in best.items() if fp not in existing_figs]

            intent = _figure_intent(query)
            MAX_FIGS = 2
            # 按「意图优先级 + score」排序：figure 意图优先 fig_p* 真图、最后才回退
            # table_p* 表格图；table 意图反之。any 意图保持原 score 排序。
            # 这样「通信流程图」即使真图 chunk 的 BM25 分数低于协议表格 chunk，
            # 也会因为意图=figure 而被顶上来，返回正确配图。
            ordered = sorted(remaining, key=lambda kv: (_fig_sort_key(kv[0], intent), kv[1]))
            figs = [fp for fp, _ in ordered][:MAX_FIGS]
            if figs:
                answer += "\n\n" + "\n".join(f"[[FIG:{fp}]]" for fp in figs)
            print(f"  [generate] 确定性追加 {len(figs)} 个图页占位符: {figs} (intent={intent})")

        return answer

    def _research_subtask(self, subtask: Dict, role: str) -> Dict:
        """
        【Researcher Agent：对单个子任务做多轮检索 RAG】

        作用：这是 complex 分支中 Researcher 角色的核心实现。
        对 planner 拆解出的单个子任务，独立完成"改写→检索→评分→重排→生成"的完整流程。

        流程（5 步）：
        1. query_rewrite — 将子任务改写为 2-3 个搜索词
        2. retrieve — 向量库（Milvus）向量检索 + 权限过滤
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
        answer = self._do_generate(query, reranked, role=role)

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
        prompt = self.pm.get_prompt("planner_decompose")
        system = prompt["system"]
        user = self.pm.format_user_message(prompt["user_template"], query=query)
        result = self.llm.chat(system, user, task="plan", user=self.username)
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
        prompt = self.pm.get_prompt("planner_supplement")
        system = prompt["system"]
        user = self.pm.format_user_message(
            prompt["user_template"],
            query=query, existing_text=existing_text
        )
        result = self.llm.chat(system, user, task="plan", user=self.username)
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
        prompt = self.pm.get_prompt("compress_history")
        system = prompt["system"]
        user = self.pm.format_user_message(prompt["user_template"], history_text=old_text)
        summary = self.llm.chat(system, user, task="compress", user=self.username)

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
        user: str = None,
        user_id: int = None,
        tenant_id: str = "default",
        username: str = None,
    ) -> str:
        """
        【对外入口：用户提问】

        这是整个 Agent 的唯一对外接口。
        CLI 模式和 Web 模式都通过这个方法来提问。

        完整流程（5 步）：
        1. Redis 缓存检查 — 命中则直接返回，跳过所有 LLM 调用
        2. 创建任务记录 — 在 MySQL task_queue 中创建 status=running 的任务
        3. LangGraph 状态图执行 — 分类 → 检索/智能体 → 生成
           （每个节点执行后自动保存断点到 MySQL task_checkpoints）
        4. 更新任务状态为 completed + 写入 Redis 缓存
        5. 输出结果

        断点重续保障：
        - 步骤 2 创建任务后，如果步骤 3 中途宕机，task_queue 中的 status 仍为 running
        - 下次服务重启时，__init__ 会把所有 running 标记为 interrupted
        - 用户下次登录时可通过 check_unfinished_tasks() 检测到中断的任务
        - 通过 resume() 方法从最后断点恢复执行

        参数：
            question: 用户输入的原始问题
            role: 用户角色（"admin" 可访问全部文档，"user" 仅公开文档）
            session_id: 会话标识符（多轮对话用，相同 session_id 共享历史）
        返回：
            生成的回答文本
        """
        # 把 user_id（admin_users.id，外键）写进请求级上下文，
        # 本次问答里所有 memory_store 写入都用它做用户隔离（不再冗余存用户名）。
        if user_id is not None:
            self.user = user_id
        # 用户名：用于 token 用量归因（用量表按 username 关联租户）
        self.username = username or user or str(self.user)
        # 租户：请求级隔离，检索下推用（super-admin 传 "__global__" 做跨租户巡检）
        self.tenant_id = tenant_id or "default"
        # 清空上一轮遗留的任务 ID：同一线程被连接池复用时，
        # 若本轮 create_task 失败，前端不应该拿到上一轮的过期 task_id 去打反馈。
        _ctx_task_id.set(None)
        _ctx_last_task_id.set(None)
        total_start = time.time()

        print("\n" + "=" * 70)
        print(f"用户提问: {question}")
        role_desc = AccessControlFilter.get_role_description(role)
        print(f"用户角色: {role_desc}")
        print("=" * 70)

        # ---- 第一步：缓存检查（Layer 3: Redis）----
        # 方案 A：CacheManager 仅精确匹配（语义答案缓存已删除）。
        self.cache.current_role = role
        self.cache.current_tenant = self.tenant_id   # 方案乙：决定缓存键里的 kb_version 粒度
        cached = self.cache.lookup(question)
        if cached:
            print(f"\n[Cache] 命中缓存，直接返回（耗时 {time.time() - total_start:.1f}s）")
            print(f"\n{'─' * 70}\n{cached}\n{'─' * 70}")
            # P0 止血 3.3：缓存只跳过「推理」，不跳过「记忆」。
            # 否则命中缓存的这一轮永不入库，对话历史出现空洞（模型看不到刚问过的内容）。
            self._append_history(session_id, question, cached, user_id=self.user, cached=True)
            return cached

        # ---- 第一步·5（方案 A 自进化读路径，提到缓存短路之前）----
        # 精确缓存未命中后、跑管线前，先查 PlaybookStore：
        # 命中相似问题 → 带「经验改写词」跑完整管线，答案仍实时生成（永远新鲜）。
        # 注意：这里复用的是「检索策略」，不是「答案」，所以绝不回旧答案。
        prefill_rewrites: List[str] = []
        used_playbook_pk: Optional[str] = None
        try:
            store = getattr(self, "playbook_store", None)
            if store is not None:
                hit = store.query_similar(question, self.tenant_id, top_k=1)
                if hit:
                    try:
                        prefill_rewrites = json.loads(hit["rewrite_text"])
                    except Exception:
                        prefill_rewrites = []
                    used_playbook_pk = hit.get("pk")
                    # 命中即复用：success_count +1（越用越快）
                    try:
                        store.patch_success(used_playbook_pk)
                    except Exception as _e:
                        print(f"[evolution] patch_success 异常(忽略): {_e}")
                    print(f"[evolution] ♻ 顶层命中 playbook (score={hit.get('score')}, "
                          f"success_count={hit.get('success_count')})")
        except Exception as e:
            print(f"[evolution] ⚠ 顶层 playbook 查询失败(忽略): {e}")

        # ---- 第二步：创建任务记录（Layer 2: MySQL task_queue）----
        # 在 MySQL 中创建一条 status=running 的任务记录。
        # 如果后续执行中断，这条记录的 status 会保持 running，
        # 下次用户登录时可以通过 check_unfinished_tasks() 检测到。
        self.current_task_id = self.memory_store.create_task(
            session_id=session_id, query=question, role=role, user_id=self.user
        )
        print(f"  [任务] 已创建任务 {self.current_task_id}（status=running）")

        # ---- 第三步：执行 LangGraph 状态图 ----
        # 构建初始状态字典，提供给图的起点（START → load_history）
        # 注意：messages 从 Layer 1 内存加载（如果同会话已加载过）或 Layer 2 MySQL
        initial_state = {
            "query": question,
            "role": role,
            "session_id": session_id,
            "messages": self._active_context.get(session_id, self.memory_store.load_messages(session_id, user_id=self.user)),
            "retrieved_docs": [],
            "doc_grades": [],
            "retrieval_iterations": 0,
            "research_results": [],
            "review_rounds": 0,
            # 方案 A：若顶层命中 Playbook，则带上经验改写词；node_query_rewrite 首轮优先复用。
            "prefill_rewrites": prefill_rewrites,
            # 强化自进化 #168：记录本次复用的 playbook pk，供用户反馈级信号回调强化
            "used_playbook_pk": used_playbook_pk,
        }

        try:
            # graph.invoke() 是 LangGraph 的执行入口。
            # 它把初始状态注入图，自动按节点→边→条件边的顺序执行，
            # 直到到达 END 节点，返回最终状态。
            # recursion_limit=50 防止无限循环（节点调用次数上限）。
            # 每个节点执行后，_wrap_node_with_checkpoint 会自动保存断点到 MySQL。
            final_state = self.graph.invoke(initial_state, {"recursion_limit": 50})
        except Exception as e:
            print(f"\n[Error] 图执行出错: {e}")
            traceback.print_exc()
            # 任务标记为失败
            self.memory_store.update_task_status(
                self.current_task_id, "failed", error_msg=str(e)
            )
            self.current_task_id = None
            return f"处理过程中出现错误: {e}"

        answer = final_state.get("answer", "抱歉，无法回答这个问题。")
        elapsed = time.time() - total_start

        # ---- 第四步：更新任务状态 + 写入缓存 ----
        # 任务标记为已完成
        self.memory_store.update_task_status(
            self.current_task_id, "completed", answer=answer
        )
        print(f"  [任务] 任务 {self.current_task_id} 已完成（status=completed）")
        self.current_task_id = None

        # 写入 Redis 缓存（Layer 3）
        self.cache.save(question, answer)

        # ---- 第五步：输出结果 ----
        print(f"\n[完成] 总耗时 {elapsed:.1f}s")
        print(f"\n{'─' * 70}\n{answer}\n{'─' * 70}")

        return answer

    def check_unfinished_tasks(self, session_id: str, user_id: int = 0) -> List[Dict]:
        """
        【断点检测：查询指定会话的未完成任务】

        作用：用户登录/连接时调用。如果有 interrupted 状态的任务，
        说明上次执行被中断（服务宕机或用户关闭客户端），
        可以提示用户并尝试恢复。

        参数：
            session_id: 会话 ID
            user_id: 登录账号的 ID（admin_users.id），防止 A 用户恢复 B 用户的任务
        返回：
            [{"task_id": "...", "query": "...", "created_at": "..."}, ...]
            空列表表示没有未完成任务
        """
        return self.memory_store.get_unfinished_tasks(session_id, user_id=user_id)

    def resume(self, task_id: str, session_id: str = "default", user_id: int = None) -> str:
        """
        【断点重续：从上次中断的位置恢复执行】

        作用：读取 MySQL task_checkpoints 中该任务的最后一条快照，
        恢复 state 字典，重新注入 LangGraph 图执行。

        原理：
        1. 从 MySQL 读取 task_checkpoints 最后一条快照（包含完整 state）
        2. 恢复 state 中的 query、role、session_id、retrieved_docs 等字段
        3. 重新调用 graph.invoke()，图会从 load_history 开始重新执行
        4. 但由于 state 中已有 retrieved_docs、query_type 等结果，
        - classify 节点会看到已有的 query_type（虽然会重新分类，但结果通常一致）
        - retrieve 节点会重新检索（因为没有"跳过已检索"的逻辑）
        - 实际效果：相当于重新执行，但对话历史和上下文都在
        5. 执行完成后更新任务状态为 completed

        局限性说明（对初学者透明）：
        真正的"从断点继续执行"需要 LangGraph 的持久化 CheckpointSaver
        （如 PostgresSaver），可以在任意节点暂停/恢复。
        本方案采用"恢复状态 + 重新执行"的简化策略，
        虽然会重复执行部分节点，但保证了数据一致性和实现简洁性。
        对于 RAG 场景（大部分耗时在 LLM 调用），Redis 缓存可以加速重复查询。

        参数：
            task_id: 要恢复的任务 ID
            session_id: 会话 ID（默认 "default"）
        返回：
            生成的回答文本
        """
        print(f"\n{'=' * 70}")
        print(f"[断点重续] 恢复任务 {task_id}")
        print(f"{'=' * 70}")

        # 把 user_id 记到实例上，resume 内部 load_messages 用它做用户隔离
        if user_id is not None:
            self.user = user_id

        # 1. 查询任务信息
        task = self.memory_store.get_task_by_id(task_id)
        if not task:
            return f"任务 {task_id} 不存在"

        print(f"  原始问题: {task['query']}")
        print(f"  任务状态: {task['status']}")

        # 2. 加载最后一条断点快照
        ckpt = self.memory_store.load_latest_checkpoint(task_id)
        if ckpt:
            print(f"  断点位置: {ckpt['node_name']}（第 {ckpt['checkpoint_order']} 个快照）")
            # 恢复 state
            restored_state = ckpt["state"]
            # 确保关键字段存在
            restored_state["session_id"] = session_id
            restored_state.setdefault("messages", self.memory_store.load_messages(session_id, user_id=self.user))
            # 【兜底】旧脏 checkpoint 的 retrieved_docs 元素是 str（老版本用 default=str 序列化），
            # 访问 .page_content 会炸。清空让 retrieve 节点重跑（影响极小，文档可重检）。
            rd = restored_state.get("retrieved_docs", [])
            if rd and any(isinstance(d, str) or not hasattr(d, "__iter__") for d in rd):
                print(f"  [resume] 检测到旧版脏 checkpoint（retrieved_docs 非 Document 对象），清空后重跑")
                restored_state["retrieved_docs"] = []
                restored_state["doc_grades"] = []
        else:
            # 没有快照，从头开始
            print(f"  无断点快照，从头执行")
            restored_state = {
                "query": task["query"],
                "role": task.get("role", DEFAULT_ROLE),
                "session_id": session_id,
                "messages": self.memory_store.load_messages(session_id, user_id=self.user),
                "retrieved_docs": [],
                "doc_grades": [],
                "retrieval_iterations": 0,
                "research_results": [],
                "review_rounds": 0,
            }

        # 3. 设置当前任务 ID（用于断点包装器继续保存快照）
        self.current_task_id = task_id

        # 4. 重新执行图
        total_start = time.time()
        try:
            final_state = self.graph.invoke(restored_state, {"recursion_limit": 50})
        except Exception as e:
            print(f"\n[Error] 恢复执行出错: {e}")
            traceback.print_exc()
            self.memory_store.update_task_status(task_id, "failed", error_msg=str(e))
            self.current_task_id = None
            return f"恢复执行失败: {e}"

        answer = final_state.get("answer", "抱歉，无法回答这个问题。")
        elapsed = time.time() - total_start

        # 5. 更新任务状态
        self.memory_store.update_task_status(task_id, "completed", answer=answer)
        self.current_task_id = None

        print(f"\n[断点重续完成] 耗时 {elapsed:.1f}s")
        print(f"\n{'─' * 70}\n{answer}\n{'─' * 70}")
        return answer


# ============================================================================
# CLI 入口 — 命令行交互模式
# ============================================================================
def run_interactive(app: LangGraphRAGApp, role: str, tenant: str = "default"):
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
            history = app._active_context.get(session_id, app.memory_store.load_messages(session_id, user_id=app.user))
            print(f"  当前会话历史（{len(history)} 条）:")
            for m in history:
                r = "用户" if m["role"] == "user" else "助手"
                print(f"    {r}: {m['content'][:80]}")
            continue
        if question.lower() in ("/clear", "/清空"):
            app._active_context.pop(session_id, None)
            app.memory_store.clear_messages(session_id, user_id=app.user)
            print("  已清空会话历史（内存 + MySQL）")
            continue

        # ---- 提问 ----
        try:
            app.query(question, role=current_role, session_id=session_id, tenant_id=tenant)
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

    # 提取 --tenant 参数（支持 "--tenant yh" 与 "--tenant=yh" 两种写法），
    # 用于多租户环境下把 CLI 验证指向真实租户（默认 default 无文档）。
    tenant = "default"
    for i, a in enumerate(args):
        if a == "--tenant" and i + 1 < len(args):
            tenant = args[i + 1]
            break
        if a.startswith("--tenant="):
            tenant = a.split("=", 1)[1]
            break

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
    app.tenant_id = tenant   # CLI 显式指定租户，便于多租户环境验证（默认 default 无文档）

    if direct_question:
        # 直接提问模式：回答后退出
        app.query(direct_question, role=role, tenant_id=tenant)
    else:
        # 交互模式：进入命令行循环
        run_interactive(app, role, tenant)


if __name__ == "__main__":
    main()
