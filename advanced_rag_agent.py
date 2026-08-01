"""
================================================================================
  高级 RAG Agent — 真实环境版（ChromaDB + Ollama）
================================================================================

  本文件是advanced_rag.py 的真实环境版本：
    - 不使用任何 Mock 数据或 Mock LLM
    - 直接连接本地 ChromaDB 向量数据库
    - 直接调用本地 Ollama（qwen2:7b）进行推理
    - 保留完整的 ReAct + Planning Agent + 智能 RAG + Skill 架构

  ┌──────────────────────────────────────────────────────────────────┐
  │                         整体架构                                  │
  │                                                                  │
  │    用户提问                                                      │
  │       │                                                          │
  │       ▼                                                          │
  │  ┌──────────────┐                                                │
  │  │ Planning Agent│  ← 调用 Ollama LLM 拆解复杂问题为子任务        │
  │  └──────┬───────┘                                                │
  │         │                                                        │
  │    ┌────┼────┐                                                   │
  │    ▼    ▼    ▼                                                   │
  │  ┌────┐┌────┐┌────┐  每个子任务由独立的 ReAct Agent 执行          │
  │  │Sub1││Sub2││Sub3│                                              │
  │  └─┬──┘└─┬──┘└─┬──┘                                              │
  │    │     │     │                                                  │
  │    ▼     ▼     ▼                                                  │
  │  ┌──────────────────────────────────┐                            │
  │  │     ReAct 循环（Think→Act→Observe）│  ← Ollama LLM 逐步推理     │
  │  └──────────┬───────────────────────┘                            │
  │             │                                                    │
  │       ┌─────┴─────┐                                              │
  │       ▼           ▼                                              │
  │  ┌──────────┐ ┌──────────┐                                       │
  │  │DocSearch │ │Calculator│  可插拔 Skill                          │
  │  │  Skill   │ │  Skill   │                                       │
  │  └────┬─────┘ └──────────┘                                       │
  │       │                                                          │
  │       ▼                                                          │
  │  ┌──────────────────────────────┐                                │
  │  │ 查询重写 → 多跳检索 → 重排序   │  智能 RAG 内部流程              │
  │  │     （ChromaDB 向量检索）      │                                │
  │  └──────────────────────────────┘                                │
  │             │                                                    │
  │             ▼                                                    │
  │  ┌──────────────────────────────┐                                │
  │  │  汇总子任务结果 → Ollama 生成   │                                │
  │  │  最终回答                      │                                │
  │  └──────────────────────────────┘                                │
  └──────────────────────────────────────────────────────────────────┘

  运行方式：
    python advanced_rag_agent.py                          # 交互模式（普通用户）
    python advanced_rag_agent.py --admin                  # 交互模式（特权用户）
    python advanced_rag_agent.py --demo                   # 运行内置演示（1题）
    python advanced_rag_agent.py "通讯协议端口是多少？"    # 直接提问
    python advanced_rag_agent.py "问题" --fast            # 快速模式（跳过查询重写）
    python advanced_rag_agent.py "问题" --admin           # 以特权用户身份提问

  --admin 模式说明：
    特权用户(admin)可访问所有文档（含受限文档）。
    普通用户(user)只能访问公开文档。
    交互模式下可输入 /admin 和 /user 动态切换角色。

  --fast 模式说明：
    跳过 LLM 查询重写步骤，直接用原始问题做向量检索。
    每个 subtask 少一次 LLM 调用，速度提升 30-50%。
    适合问题比较明确、不需要术语转换的场景。

  环境要求：
    - Python 3.10（conda env: pythonspace）
    - ChromaDB 已构建（./chroma_db）
    - Ollama 已启动并加载 qwen2:7b 模型

================================================================================
"""

# ============================================================================
# 环境配置（必须在所有 import 之前）
# ============================================================================
import os

# ---- 加载 .env 文件（轻量实现，零依赖） ----
def _load_dotenv(dotenv_path: str | None = None):
    """解析 .env 文件并将未设置的变量注入 os.environ。"""
    if dotenv_path is None:
        # 从当前文件所在目录向上查找 .env
        import __main__
        dotenv_path = os.path.join(
            os.path.dirname(os.path.abspath(__main__.__file__ if hasattr(__main__, '__file__') else __file__)),
            ".env"
        )
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

# HF_ENDPOINT 指向国内镜像，加速 embedding 模型下载
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 屏蔽 LangChain 的弃用警告，让输出更干净
import warnings
warnings.filterwarnings("ignore")

import re
import sys
import json
import ast
import operator as op
import time
import hashlib
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

# ============================================================================
# 配置区
# ============================================================================
DOC_FOLDER = "./docs"
DB_PATH = "./chroma_db"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.200.128:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2:7b")
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
SEPARATORS = ["\n\n", "\n", "。", "；", "？", "！", "，", "、"]

# Redis 缓存配置
REDIS_HOST = os.getenv("REDIS_HOST", "192.168.200.128")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "dev0619")
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
CACHE_TTL = 7 * 24 * 3600          # 缓存过期时间：7 天
CACHE_PREFIX = "rag:cache:"        # Redis 键前缀
SEMANTIC_THRESHOLD = 0.80          # 语义匹配阈值（余弦相似度 > 0.80 即命中）
CACHE_MAX_SCAN = 500               # 语义查找时最多扫描的缓存条目数

# ============================================================================
# 文档访问权限配置
# ============================================================================
# 文档按权限级别分两类：
#   "public"    — 所有用户可访问
#   "restricted"— 仅特权用户(admin)可访问
#
# 配置方式：文件名包含关键字 → 权限级别
# 未匹配的文档默认为 public
DOC_ACCESS_RULES = {
    "JM-S509": "restricted",   # JM-S509 学生证产品客户指令表 → 仅特权用户
    # Jimi IoT 个人定位终端通讯协议 → 默认 public，所有用户可访问
}

ROLE_ADMIN = "admin"      # 特权用户：可访问所有文档（含 restricted）
ROLE_USER = "user"        # 普通用户：只能访问 public 文档
DEFAULT_ROLE = ROLE_USER   # 默认角色


# ============================================================================
# 第零部分：Redis 智能缓存层
# ============================================================================
#
#  为什么需要缓存？每次问答都要调 6+ 次 LLM（qwen2:7b），耗时 90s+。
#  同样的问题问两遍，完全没有必要重新跑一轮——缓存直接返回。
#
#  缓存架构 — 两级匹配（先精确后语义）：
#
#  ┌─────────────────────────────────────────────────────────────┐
#  │  用户提问                                                    │
#  │     │                                                        │
#  │     ├── 第1级：精确匹配                                       │
#  │     │   对问题做标准化（去空格/标点/转小写）→ SHA256 哈希      │
#  │     │   Redis Key: rag:cache:{sha256_hex}                    │
#  │     │   命中 → 直接返回缓存答案（<1ms）                        │
#  │     │                                                        │
#  │     └── 第2级：语义匹配（精确未命中时触发）                     │
#  │         把问题转成 embedding 向量                              │
#  │         扫描 Redis 中所有缓存条目的 embedding                  │
#  │         计算余弦相似度，找到 > 0.85 的最近条目                  │
#  │         命中 → 返回缓存答案，同时补一条精确匹配键方便下次命中    │
#  │                                                              │
#  │  两级都未命中 → 正常走 ReAct + PlanningAgent 流程             │
#  │  流程结束后 → 把问题和答案写入缓存（同时写精确键 + embedding）   │
#  └─────────────────────────────────────────────────────────────┘
#
#  Redis 数据格式（每个缓存条目，JSON 字符串）：
#  {
#    "q": "原始问题",
#    "a": "LLM 生成的答案",
#    "emb": [0.12, -0.34, ...],   // 768 维 embedding 向量
#    "ts": "2025-07-22T10:30:00"  // 缓存时间
#  }
#  TTL: 7 天（CACHE_TTL）


class CacheManager:
    """
    Redis 智能缓存管理器

    两级匹配策略：
      1. 精确匹配：问题标准化 → SHA256 → Redis GET，最快
      2. 语义匹配：问题 → embedding → 扫描历史 embedding → 余弦相似度

    为什么不用 Redis 自带的向量搜索（RediSearch）？
      RediSearch 需要单独安装模块，不是所有 Redis 部署都有。
      对于几百到几千条缓存的场景，Python 端计算余弦相似度（几十毫秒）完全够用。
      如果缓存条目上了万，再考虑升级到 RediSearch。
    """

    def __init__(self, host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD,
                 db=REDIS_DB, ttl=CACHE_TTL, threshold=SEMANTIC_THRESHOLD):
        import redis as redis_pkg

        try:
            self.redis = redis_pkg.Redis(
                host=host, port=port, password=password, db=db,
                socket_connect_timeout=5, socket_timeout=5,
                decode_responses=True  # 自动把 bytes → str
            )
            self.redis.ping()
            print(f"[CacheManager] Redis 已连接 {host}:{port} (DB{db})")
        except Exception as e:
            print(f"[CacheManager] ⚠ Redis 连接失败: {e}")
            print(f"[CacheManager] 缓存功能已禁用，每次都会调用 LLM")
            self.redis = None

        self.ttl = ttl
        self.threshold = threshold
        self._hit_count = 0       # 命中计数
        self._miss_count = 0      # 未命中计数
        self._embed_fn = None     # 延迟加载 embedding 模型
        self.current_role = DEFAULT_ROLE  # 当前用户角色（影响缓存键，防止跨角色泄漏）

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def lookup(self, query: str) -> Optional[str]:
        """
        查找缓存，返回答案字符串；如果没命中返回 None。

        先精确匹配，再语义匹配。
        """
        if self.redis is None:
            return None

        # === 第1级：精确匹配 ===
        exact_key = self._exact_key(query)
        cached = self.redis.get(exact_key)
        if cached:
            self._hit_count += 1
            data = json.loads(cached)
            print(f"\n  [CacheManager] ✓ 精确命中，跳过 LLM 调用")
            print(f"  [CacheManager] 缓存时间: {data.get('ts', '?')}")
            print(f"  [CacheManager] 原问题: {data.get('q', '?')}")
            return data["a"]

        # === 第2级：语义匹配 ===
        result = self._semantic_lookup(query)
        if result:
            # 命中了语义匹配，同时补一条精确匹配键，下次直接第一级命中
            self._hit_count += 1
            answer, matched_q = result
            # 为新查询补写精确键（复用已有的 answer 和 embedding）
            # 需要从原始缓存条目中取 embedding
            orig_key = self._exact_key(matched_q)
            orig_data = json.loads(self.redis.get(orig_key) or "{}")
            emb = orig_data.get("emb", [])
            self._save_entry(query, answer, emb)
            print(f"\n  [CacheManager] ✓ 语义匹配命中（相似问题）")
            print(f"  [CacheManager] 匹配到: \"{matched_q}\"")
            return answer

        self._miss_count += 1
        return None

    def save(self, query: str, answer: str):
        """
        把问题和答案写入缓存。

        会同时生成：
          - 精确匹配键（SHA256 哈希）
          - embedding 向量（用于后续语义匹配）
        """
        if self.redis is None:
            return

        # 生成 embedding（用于语义匹配）
        emb = self._embed(query)

        self._save_entry(query, answer, emb)
        print(f"  [CacheManager] 已写入缓存（精确+语义）")

    @property
    def stats(self) -> dict:
        """返回缓存统计信息"""
        total = self._hit_count + self._miss_count
        hit_rate = f"{self._hit_count / total * 100:.0f}%" if total > 0 else "N/A"
        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "total": total,
            "hit_rate": hit_rate,
            "redis_keys": self.redis.dbsize() if self.redis else 0,
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _exact_key(self, query: str) -> str:
        """
        生成精确匹配的 Redis 键

        对问题做标准化处理（去多余空白、全角转半角、转小写），
        然后拼接用户角色，最后 SHA256 哈希。

        为什么要把 role 拼进哈希？
          admin 和 user 可能问同一个问题（如"定位方式"），
          但 admin 能看到 JM-S509 指令表的额外信息，答案不同。
          如果共用一个缓存键，admin 的答案会泄漏给 user。
          拼入 role 后，两个角色各自有独立的缓存空间。
        """
        # 标准化：去首尾空白、合并连续空白、全角转半角、转小写
        import unicodedata
        normalized = query.strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = unicodedata.normalize('NFKC', normalized).lower()
        # 拼入用户角色，确保不同角色的缓存互不干扰
        hash_input = f"{normalized}|{self.current_role}"
        hash_hex = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]
        return f"{CACHE_PREFIX}{hash_hex}"

    def _semantic_lookup(self, query: str) -> Optional[Tuple[str, str]]:
        """
        语义匹配：把当前问题转为 embedding，和 Redis 中所有缓存条目比较。

        返回 (answer, matched_query) 或 None。

        性能说明：
          - SCAN 操作 O(N)，但每次只扫 CACHE_MAX_SCAN 个键
          - 余弦相似度计算是向量点积，几百条 <50ms
          - 如果缓存条目超过 CACHE_MAX_SCAN，只扫最近的
        """
        query_emb = self._embed(query)

        # 收集所有缓存条目
        entries = []
        cursor = 0
        while True:
            cursor, keys = self.redis.scan(cursor, match=f"{CACHE_PREFIX}*", count=100)
            for key in keys:
                data = self.redis.get(key)
                if data:
                    try:
                        entry = json.loads(data)
                        # 角色隔离：只匹配同一角色的缓存条目
                        # 防止 admin 的答案通过语义匹配泄漏给 user
                        if entry.get("role") != self.current_role:
                            continue
                        if entry.get("emb") and len(entry["emb"]) == len(query_emb):
                            entries.append(entry)
                    except json.JSONDecodeError:
                        pass
            if cursor == 0 or len(entries) >= CACHE_MAX_SCAN:
                break

        if not entries:
            return None

        # 批量计算余弦相似度（用 numpy 向量化加速）
        import numpy as np
        q_vec = np.array(query_emb)
        best_score = -1
        best_entry = None

        for entry in entries:
            e_vec = np.array(entry["emb"])
            # 余弦相似度 = dot(A, B) / (||A|| * ||B||)
            cos_sim = float(np.dot(q_vec, e_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(e_vec) + 1e-10))
            if cos_sim > best_score:
                best_score = cos_sim
                best_entry = entry

        if best_score >= self.threshold and best_entry:
            return (best_entry["a"], best_entry["q"])
        return None

    def _save_entry(self, query: str, answer: str, emb: List[float]):
        """写入一条缓存记录（精确键 + embedding + 角色）"""
        entry = json.dumps({
            "q": query,
            "a": answer,
            "emb": emb,
            "role": self.current_role,  # 记录角色，语义匹配时按角色过滤
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, ensure_ascii=False)
        key = self._exact_key(query)
        self.redis.setex(key, self.ttl, entry)

    def _embed(self, text: str) -> List[float]:
        """
        把文本转为 embedding 向量（768 维）

        延迟加载 embedding 模型，避免 import 时触发模型下载。
        使用与 ChromaDB 相同的 BAAI/bge-small-zh-v1.5 模型。
        """
        if self._embed_fn is None:
            from langchain_community.embeddings import SentenceTransformerEmbeddings
            self._embed_fn = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
        return self._embed_fn.embed_query(text)


# ============================================================================
# 第一部分：LLM 抽象层
# ============================================================================
# 定义统一的 LLM 接口，目前只有 OllamaLLM 一个实现。
# 为什么要抽象？如果将来想换成 OpenAI / 通义千问 / DeepSeek，
# 只需新增一个子类，Agent 代码完全不用改。

class BaseLLM(ABC):
    """LLM 抽象基类 — 所有 LLM 后端都要实现 chat 方法"""

    @abstractmethod
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """
        发送对话请求，返回 LLM 生成的文本

        :param system_prompt: 系统提示词（定义角色和规则）
        :param user_prompt:   用户输入
        :return: LLM 回复文本
        """
        pass


class OllamaLLM(BaseLLM):
    """
    Ollama LLM 后端 — 连接你本地部署的 Ollama 服务

    使用 langchain_ollama.ChatOllama 封装，支持流式输出（这里用同步调用）。
    temperature=0.1 保持低随机性，让推理过程更稳定可预测。
    """

    def __init__(self, base_url: str = OLLAMA_URL, model: str = MODEL_NAME):
        from langchain_ollama import ChatOllama

        self.base_url = base_url
        self.model = model
        self._llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0.1  # 低温度 = 更确定性的输出，适合 Agent 推理
        )
        # 调用统计：帮助用户理解时间花在哪里
        self.call_count = 0
        self.total_time = 0.0
        print(f"[LLM] 已连接 Ollama: {base_url}, 模型: {model}")

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """
        调用 Ollama 生成回复

        内部用 LangChain 的 prompt template 拼接 system + user 消息，
        然后通过 chain 调用 LLM 并解析输出。
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        # 构造消息模板：system 定义角色，human 是用户输入
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        # LangChain LCEL 语法：prompt → llm → 字符串解析
        chain = prompt | self._llm | StrOutputParser()

        # 同步调用（会阻塞直到 LLM 返回）
        self.call_count += 1
        start = time.time()
        result = chain.invoke({"input": user_prompt})
        elapsed = time.time() - start
        self.total_time += elapsed
        return result


# ============================================================================
# 第二部分：向量数据库管理
# ============================================================================
# 这部分封装了 ChromaDB 的初始化和检索操作。
# 但封装成类方便 Skill 调用。

class VectorStoreManager:
    """
    向量数据库管理器 — 封装 ChromaDB 的初始化和检索

    如果 ./chroma_db 已存在则直接加载，否则从 docs/ 构建。
    使用 BAAI/bge-small-zh-v1.5 作为 embedding 模型（中文优化）。
    """

    @staticmethod
    def init_vector_store():
        """
        初始化 ChromaDB 向量数据库

        返回: Chroma 实例，可用于 similarity_search
        """
        from langchain_chroma import Chroma
        from langchain_community.embeddings import SentenceTransformerEmbeddings

        # 加载 embedding 模型（首次会下载，之后从缓存加载）
        print(f"[VectorStore] 加载 embedding 模型: {EMBED_MODEL}")
        embedding = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)

        if os.path.exists(DB_PATH):
            # 已有向量库，直接加载
            print(f"[VectorStore] 加载已有向量数据库: {DB_PATH}")
            db = Chroma(persist_directory=DB_PATH, embedding_function=embedding)

            # 验证数据库是否有效
            try:
                test_results = db.similarity_search("测试", k=1)
                print(f"[VectorStore] 向量数据库正常，包含文档片段")
            except Exception as e:
                print(f"[VectorStore] ⚠ 向量数据库可能损坏: {e}")
                raise

            return db
        else:
            # 首次运行，需要从 docs/ 构建
            print(f"[VectorStore] 首次运行，从 {DOC_FOLDER} 构建向量数据库...")
            return VectorStoreManager._build_vector_store(embedding)

    @staticmethod
    def _build_vector_store(embedding):
        """从 docs/ 目录构建向量数据库"""
        from langchain_community.document_loaders import PyPDFLoader, TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_chroma import Chroma

        # 1. 读取原始文档
        raw_docs = []
        for filename in os.listdir(DOC_FOLDER):
            file_path = os.path.join(DOC_FOLDER, filename)
            try:
                if filename.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif filename.endswith(".txt"):
                    loader = TextLoader(file_path, encoding="utf-8")
                else:
                    continue
                docs = loader.load()
                # 给每个文档打上访问权限标签（用于 ChromaDB 原生元数据过滤）
                access_level = AccessControlFilter.get_access_level(file_path)
                for doc in docs:
                    doc.metadata["access_level"] = access_level
                raw_docs.extend(docs)
                print(f"  ✅ 载入: {filename} (权限: {access_level})")
            except Exception as e:
                print(f"  ❌ 读取失败 {filename}: {e}")

        if not raw_docs:
            raise Exception(f"{DOC_FOLDER} 文件夹内没有可识别文档！")

        # 2. 智能分片
        text_splitter = RecursiveCharacterTextSplitter(
            separators=SEPARATORS,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            strip_whitespace=True
        )
        split_docs = text_splitter.split_documents(raw_docs)
        print(f"  分片完成: {len(raw_docs)} 个原始文档 → {len(split_docs)} 个片段")

        # 3. 写入向量库
        db = Chroma.from_documents(
            documents=split_docs,
            embedding=embedding,
            persist_directory=DB_PATH
        )
        print(f"[VectorStore] 向量数据库构建完成")
        return db

    @staticmethod
    def search(db, query: str, k: int = 4) -> List[Tuple[Any, float]]:
        """
        向量相似度检索

        :param db: ChromaDB 实例
        :param query: 搜索文本
        :param k: 返回结果数
        :return: [(Document, distance), ...]  distance 越小越相似
        """
        # similarity_search_with_score 返回 (Document, float) 列表
        # Document 有 page_content 和 metadata 属性
        # float 是 L2 距离（越小越相似，0 = 完全相同）
        results = db.similarity_search_with_score(query, k=k)
        return results


# ============================================================================
# 第 2.5 部分：文档访问权限过滤器
# ============================================================================
#
#  ┌──────────────────────────────────────────────────────────────────┐
#  │  用户提问（带 user_role）                                         │
#  │     │                                                             │
#  │     ▼                                                             │
#  │  DocSearchSkill.execute(query)                                    │
#  │     │                                                             │
#  │     ▼                                                             │
#  │  ChromaDB 向量检索（top_k=5）                                      │
#  │     │                                                             │
#  │     ▼                                                             │
#  │  AccessControlFilter.filter_results(results, user_role)           │
#  │     │                                                             │
#  │     ├── admin → 全部保留（可访问所有文档）                          │
#  │     └── user  → 移除 source 含 "JM-S509" 的文档片段                │
#  │               （只能访问 public 文档）                              │
#  │                                                                   │
#  │  过滤后的结果 → MMR 重排序 → 返回给 ReAct Agent                    │
#  └──────────────────────────────────────────────────────────────────┘
#
#  为什么在后端过滤而不是前端拒绝？
#    用户的问题本身不包含权限信息（"定位方式"对所有人都一样），
#    权限差异体现在「能看到哪些文档的检索结果」。
#    后端过滤 = 检索到了但根据权限丢弃，用户无感知被过滤了哪些内容。
#
#  缓存隔离：
#    admin 和 user 的缓存 key 不同（包含 role），
#    即使问同样的问题，admin 的答案不会泄漏给 user。


class AccessControlFilter:
    """
    文档访问权限过滤器

    根据用户角色过滤检索结果：
      - admin（特权用户）：可访问所有文档
      - user（普通用户）：只能访问 public 文档

    权限判断基于文档来源文件名：
      - 文件名匹配 DOC_ACCESS_RULES 中的关键字 → 对应权限级别
      - 未匹配 → 默认 public

    示例：
      source = "D:/docs/JM-S509 学生证产品客户指令表_V1.0.pdf"
      → 匹配 "JM-S509" → access_level = "restricted"
      → 普通用户无法访问

      source = "D:/docs/Jimi IoT_个人定位终端通讯协议 V1.2(1).pdf"
      → 不匹配任何规则 → access_level = "public"
      → 所有用户可访问
    """

    @staticmethod
    def get_access_level(source: str) -> str:
        """
        根据文档来源路径判断访问级别

        :param source: 文档文件路径
        :return: "public" 或 "restricted"
        """
        basename = os.path.basename(source)
        for keyword, level in DOC_ACCESS_RULES.items():
            if keyword in basename:
                return level
        return "public"

    @staticmethod
    def filter_results(results: List[Tuple[Any, float]],
                       user_role: str) -> List[Tuple[Any, float]]:
        """
        过滤检索结果，移除用户无权访问的文档

        :param results: ChromaDB 检索结果 [(Document, score), ...]
        :param user_role: 用户角色 ("admin" 或 "user")
        :return: 过滤后的结果列表
        """
        if user_role == ROLE_ADMIN:
            # 特权用户可访问所有文档，无需过滤
            return results

        # 普通用户：过滤掉 restricted 文档
        filtered = []
        blocked_sources = set()
        for doc, score in results:
            source = doc.metadata.get("source", "")
            level = AccessControlFilter.get_access_level(source)
            if level == "public":
                filtered.append((doc, score))
            else:
                blocked_sources.add(os.path.basename(source))

        if blocked_sources:
            print(f"    [AccessControl] 🚫 已过滤无权限文档: "
                  f"{', '.join(blocked_sources)}（用户角色: {user_role}）")

        return filtered

    @staticmethod
    def get_role_description(user_role: str) -> str:
        """返回用户角色的中文描述"""
        if user_role == ROLE_ADMIN:
            return "特权用户（可访问所有文档）"
        return "普通用户（仅可访问公开文档）"


# ============================================================================
# 第三部分：Skill 系统（可插拔技能）
# ============================================================================
#
# Skill 是 Agent 可以调用的「工具」。每个 Skill 封装了一种能力。
# Agent 通过 ReAct 循环自主决定调用哪个 Skill。
#
# 设计要点：
#   - 所有 Skill 继承 BaseSkill，实现 execute 方法
#   - Skill 注册到 SkillRegistry，Agent 从中查找和调用
#   - 新增能力只需写一个新 Skill 类并注册，无需改 Agent 代码
#       → 这就是「可插拔」的含义

# ============================================================================
# Skill 内核已抽到 skill_framework.py（协议无关、可被 MCP Server 复用）
# 这里只做再导出，保证文件内其余代码（DocSearchSkill / ReActAgent 等）零改动
# ============================================================================
from skill_framework import BaseSkill  # noqa: E402  (置于类定义原位置，保证下方引用不变)


class DocSearchSkill(BaseSkill):
    """
    文档检索技能 — 智能 RAG 的入口

    这里实现了「智能 RAG」三步流程：

    ┌─────────────────────────────────────────────────────────────┐
    │  第1步：查询重写（Query Rewriting）                           │
    │  用户说"续航怎么样" → 重写为"JM-S509 电池容量 待机时间 mAh"    │
    │  原因：用户用语和文档用语可能不一致，重写后提高召回率           │
    │  注意：此步需要调用 LLM，会额外增加 1 次推理耗时               │
    │  fast 模式下会跳过此步，直接用原始问题检索                     │
    │                                                              │
    │  第2步：多跳检索（Multi-hop Retrieval）                       │
    │  第1跳：用重写后的查询检索 ChromaDB                           │
    │  如果结果不足 → 从第1跳结果提取关键词 → 第2跳检索              │
    │  合并两跳结果并去重                                           │
    │                                                              │
    │  第3步：结果重排序（Reranking）                               │
    │  对所有检索片段按与原始问题的相关度重新排列                     │
    │  综合 ChromaDB 距离分数 + 关键词匹配数                         │
    └─────────────────────────────────────────────────────────────┘
    """

    name = "doc_search"
    description = (
        "搜索企业文档知识库（ChromaDB向量数据库），返回与查询相关的文档片段。"
        "适用于需要查找产品参数、协议说明、功能规格等文档内容时。"
        "输入：搜索关键词或问题描述。输出：相关文档片段列表。"
    )

    def __init__(self, llm: BaseLLM, vector_db, fast_mode: bool = False,
                 user_role: str = DEFAULT_ROLE):
        self.llm = llm          # 用于查询重写
        self.db = vector_db     # ChromaDB 实例
        self.fast_mode = fast_mode  # True=跳过查询重写，直接检索
        self.user_role = user_role  # 当前用户角色，用于文档访问权限过滤

    def execute(self, query: str) -> str:
        """执行智能 RAG 检索流程"""
        # 参数安全校验
        err = self.validate_params(query)
        if err:
            return err

        print(f"\n    [DocSearchSkill] 开始智能检索: \"{query}\"")

        # ====== 第1步：查询重写（fast 模式跳过）======
        if self.fast_mode:
            # 快速模式：直接用原始问题检索，不调 LLM 做查询重写
            rewrites = [query]
            print(f"    [DocSearchSkill] 快速模式：跳过查询重写，直接检索")
        else:
            # 完整模式：让 LLM 把用户问题改写成多个搜索变体
            rewrites = self._query_rewrite(query)
            print(f"    [DocSearchSkill] 查询重写生成 {len(rewrites)} 个搜索变体:")
            for i, rw in enumerate(rewrites):
                print(f"      变体{i+1}: \"{rw}\"")

        # ====== 第2步：第一跳检索 ======
        # top_k 从 3 提高到 5，增加结果丰富度
        # 例如搜索"定位方式"时，k=5 才能同时捕获 GPS/LBS/WiFi 相关的 chunk
        first_hop = self._retrieve(rewrites, top_k=5)
        print(f"    [DocSearchSkill] 第一跳检索到 {len(first_hop)} 个文档片段")

        # ====== 第3步：多样性检查 + 多跳检索 ======
        # 不仅看结果数量，更看结果是否覆盖了不同维度
        # 如果所有结果都是关于同一种定位方式（如全是GPS），说明需要补搜
        first_hop = self._diversity_hop(query, first_hop, rewrites)

        # ====== 第4步：重排序（含多样性提升）======
        # 综合 ChromaDB 距离分数 + 关键词匹配数 + 多样性奖励，重新排列
        reranked = self._rerank(query, first_hop)
        print(f"    [DocSearchSkill] 重排序完成，Top {len(reranked)} 结果:")

        # 格式化输出，返回给 ReAct Agent 作为 Observation
        # 注意：截断每个片段到 200 字符，避免 prompt 过长导致 LLM 生成极慢
        result_parts = []
        for i, (doc, score) in enumerate(reranked):
            source = doc.metadata.get("source", "未知")
            page = doc.metadata.get("page", "?")
            # ChromaDB 的 distance 越小越相似，转换为 0-100 的相关度分数
            relevance = max(0, round(100 - score * 50))
            print(f"      [{i+1}] {os.path.basename(source)} P{page} (相关度:{relevance}%)")
            # 截断文档内容，保留前 350 字符（足够包含协议号+描述+关键字段）
            content_truncated = doc.page_content[:350]
            result_parts.append(
                f"[文档{i+1}] 来源:{os.path.basename(source)} 第{page}页\n{content_truncated}"
            )

        if not result_parts:
            # 区分"没有相关文档"和"有文档但无权限访问"两种情况
            if self.user_role != ROLE_ADMIN:
                return ("未检索到相关文档。"
                        "部分文档可能需要特权用户权限才能访问，"
                        "如需查看完整信息请联系管理员。")
            return "未检索到相关文档。"

        return "\n\n".join(result_parts)

    def _query_rewrite(self, query: str) -> List[str]:
        """
        查询重写：让 LLM 把用户问题改写为多个搜索变体

        为什么需要重写？
        - 用户问"续航怎么样" → 文档里写的是"待机时间""电池容量"
        - 用户问"定位准不准" → 文档里写的是"定位精度""GPS误差"
        - 重写后用文档术语检索，召回率更高
        """
        system_prompt = """你是查询重写助手。将用户问题改写为2-3个更适合向量检索的关键词组合。

要求：
- 保留原始问题的核心意图
- 使用文档中可能出现的专业术语和关键词
- 每个变体是一个简短的搜索词组合（不要完整句子）

特殊情况——当问题在问"有哪几种/支持哪些/什么方式"这类列举型问题时：
- 必须为每一种可能的类型生成独立的搜索词组合
- 例如问"定位方式"→ 分别生成 "GPS 定位包 0x22 0xA0"、"LBS 基站定位包 0x28 0xA1"、"WiFi 定位包 0x2C 0xA2"、"FIXPRI 定位方式优先级"
- 例如问"报警类型"→ 分别生成 "围栏报警"、"低电量报警"、"SOS报警"

特殊情况——当问题在问"定位精度/准确度/误差"时：
- 生成包含卫星状态、定位星数、GPS模块状态等术语的搜索词
- 例如问"定位精度"→ "卫星状态 GPS模块 2D定位 3D定位 搜星 定位星数 0x09"

输出格式（必须是合法JSON，不要包含```json等markdown标记）：
{{"rewrites": ["关键词组合1", "关键词组合2"]}}"""

        try:
            result = self.llm.chat(system_prompt, query)
            # 清理可能存在的 markdown 标记
            result = result.strip()
            if result.startswith("```"):
                result = re.sub(r'^```(?:json)?\s*', '', result)
                result = re.sub(r'\s*```$', '', result)

            data = json.loads(result)
            rewrites = data.get("rewrites", [query])
            # 始终包含原始查询作为兜底
            if query not in rewrites:
                rewrites.insert(0, query)
            return rewrites[:4]  # 最多4个变体
        except (json.JSONDecodeError, Exception) as e:
            # LLM 输出格式不对，退化为只用原始查询
            print(f"    [DocSearchSkill] 查询重写失败({e})，使用原始查询")
            return [query]

    def _diversity_hop(self, query: str, first_hop: List[Tuple[Any, float]],
                       original_rewrites: List[str]) -> List[Tuple[Any, float]]:
        """
        多样性检索：检查第一跳结果是否只覆盖了单一维度，如果是则补搜其他维度。

        核心思路：
        ——————————————————————————————————————————————
        用户问"定位方式" → 第一跳全是GPS → 但文档里还有LBS和WiFi
        传统多跳只看结果数量（≥2就跳过），这里额外看内容多样性
        ——————————————————————————————————————————————

        实现方式：
        1. 从第一跳结果中提取已有的主题词（如"GPS"）
        2. 从查询意图中推断可能遗漏的主题（如"LBS", "WiFi"）
        3. 针对性补搜遗漏主题，合并到结果集中
        """
        # 收集第一跳中已有的关键术语
        # 注意：逐 chunk 检查，而非合并后查找，避免第一个匹配点不是定位描述
        existing_terms = set()

        type_patterns = {
            "GPS": ["GPS", "gps", "卫星定位", "0x22", "0xA0", "GPS定位包"],
            "LBS": ["LBS", "lbs", "基站定位", "多基站", "0x28", "0xA1", "LBS定位"],
            "WiFi": ["WIFI", "wifi", "无线定位", "0x2C", "0xA2", "WiFi定位包"],
            "FIXPRI": ["FIXPRI", "fixpri", "定位方式优先级", "WIFI定位优先于GPS"],
        }
        for term_type, keywords in type_patterns.items():
            found = False
            for doc, _ in first_hop:
                text = doc.page_content
                for kw in keywords:
                    if kw in text:
                        # 在该 chunk 中找到关键词，检查上下文是否为定位描述
                        idx = text.find(kw)
                        if idx >= 0:
                            context = text[max(0, idx - 50):idx + len(kw) + 50]
                            if any(hint in context for hint in [
                                "定位包", "定位方式", "定位类型", "定位模式",
                                "用于传输", "描述", "2.4", "2.5", "2.8", "2.11",
                                "2.12", "2.13", "FIXPRI", "优先级",
                            ]):
                                existing_terms.add(term_type)
                                found = True
                                break
                if found:
                    break

        # 判断用户查询是否在问"定位方式/定位类型"这类需要枚举的问题
        category_indicators = ["定位方式", "定位类型", "几种定位", "支持哪些定位",
                               "定位模式", "方式", "类型", "种类"]
        is_category_question = any(ind in query for ind in category_indicators)

        # 判断是否需要补搜：查询是"列举类"问题，且第一跳中缺少某些类型
        missing_types = []
        if is_category_question:
            for term_type in type_patterns:
                if term_type not in existing_terms:
                    missing_types.append(term_type)

        if not missing_types:
            return first_hop

        print(f"    [DocSearchSkill] 多样性检查：第一跳已有 {existing_terms}，"
              f"缺失 {missing_types}，启动补搜...")

        # 对每个缺失的类型构造专门查询（使用协议文档中的精确术语）
        additional_queries = []
        for mt in missing_types:
            if mt == "GPS":
                additional_queries.append("GPS 定位包 0x22 0xA0 描述 用于传输终端位置")
            elif mt == "LBS":
                additional_queries.append("LBS 多基站 定位包 0x28 0xA1 描述 用于传输终端")
            elif mt == "WiFi":
                additional_queries.append("WiFi 定位包 0x2C 0xA2 描述 用于传输终端接收")
            elif mt == "FIXPRI":
                additional_queries.append("FIXPRI 定位方式优先级 WIFI定位优先 GPS定位 LBS定位方式")

        # 始终追加 FIXPRI 命令搜索（这是最明确的定位方式说明）
        if "FIXPRI" not in existing_terms:
            additional_queries.append("FIXPRI 定位方式优先级 WIFI定位优先于GPS LBS")

        # 执行补搜
        extra_hop = self._retrieve(additional_queries, top_k=3)

        # 合并并去重
        seen_keys = set()
        for doc, _ in first_hop:
            seen_keys.add(doc.page_content[:50])

        for doc, dist in extra_hop:
            key = doc.page_content[:50]
            if key not in seen_keys:
                seen_keys.add(key)
                first_hop.append((doc, dist))

        print(f"    [DocSearchSkill] 多样性补搜后共 {len(first_hop)} 个片段")
        return first_hop

    def _retrieve(self, queries: List[str], top_k: int = 3) -> List[Tuple[Any, float]]:
        """
        向量检索：用多个查询变体分别检索 ChromaDB，合并去重

        真实实现：用 embedding 模型把 query 编码成向量，
                  在 ChromaDB 中做余弦相似度搜索。
        """
        all_results = []
        seen_contents = set()  # 用于去重

        for q in queries:
            try:
                results = VectorStoreManager.search(self.db, q, k=top_k)
                for doc, distance in results:
                    # 用内容前50字符作为去重键
                    key = doc.page_content[:50]
                    if key not in seen_contents:
                        seen_contents.add(key)
                        all_results.append((doc, distance))
            except Exception as e:
                print(f"    [DocSearchSkill] 检索查询 \"{q}\" 失败: {e}")

        # 按距离排序（距离越小越相似）
        all_results.sort(key=lambda x: x[1])

        # 访问权限过滤：移除用户无权访问的文档片段
        # 例如普通用户检索到 JM-S509 指令表的内容 → 过滤掉
        all_results = AccessControlFilter.filter_results(all_results, self.user_role)

        return all_results[:top_k * 3]  # 限制总结果数（放宽以支持多样性检索）

    def _rerank(self, original_query: str, results: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
        """
        重排序：MMR (Maximal Marginal Relevance) 多样性重排

        传统重排序只按相关度分数排列 → 可能导致 Top5 全是 GPS，LBS/WiFi 不见踪影。
        MMR 算法在保证相关度的同时，惩罚与已选结果过于相似的内容。

        公式：MMR = λ × 相关度 - (1-λ) × 与已选结果的最大相似度
        其中 λ=0.7 表示相关度权重 70%，多样性权重 30%

        生产环境可以用 Cross-Encoder（如 bge-reranker-base）做更精准的相关度评估。
        """
        if len(results) <= 3:
            # 结果本就少，不需要多样性排重
            return results

        # 1. 计算每个文档的基线相关度分数
        query_words = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', original_query.lower()))
        scored = []
        for doc, distance in results:
            content_words = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', doc.page_content.lower()))
            keyword_overlap = len(query_words & content_words)
            # 相关度 = 向量距离的倒数 + 关键词匹配奖励
            relevance = (1.0 / max(distance, 0.01)) + keyword_overlap * 0.5
            scored.append((doc, distance, relevance))

        # 2. MMR 贪婪选择
        selected = []
        remaining = scored[:]

        # 第一个选相关度最高的
        remaining.sort(key=lambda x: x[2], reverse=True)
        best = remaining.pop(0)
        selected.append(best)

        LAMBDA = 0.7  # 相关度权重

        while remaining and len(selected) < min(8, len(results)):
            max_mmr = -float('inf')
            best_idx = 0

            for i, (doc, dist, rel) in enumerate(remaining):
                # 计算与已选结果的最大相似度（基于共享词数）
                cur_words = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', doc.page_content.lower()))
                max_sim = 0.0
                for sel_doc, _, _ in selected:
                    sel_words = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+',
                                               sel_doc.page_content.lower()))
                    if cur_words and sel_words:
                        sim = len(cur_words & sel_words) / max(len(cur_words | sel_words), 1)
                        max_sim = max(max_sim, sim)

                # MMR = λ*相关度 - (1-λ)*相似度（相似度高 = 扣分多）
                mmr = LAMBDA * rel - (1 - LAMBDA) * max_sim * rel
                if mmr > max_mmr:
                    max_mmr = mmr
                    best_idx = i

            # 选 MMR 最高的加入已选列表
            selected.append(remaining.pop(best_idx))

        # 3. 剩余未选的追加到末尾
        for r in remaining:
            selected.append(r)

        return [(doc, dist) for doc, dist, _ in selected]

    def _extract_keywords(self, text: str) -> str:
        """从文本中提取关键词用于第二轮检索"""
        # 提取中文词和英文词
        words = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{2,}', text)
        # 取前8个去重
        seen = set()
        unique = []
        for w in words:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        return " ".join(unique[:8]) if unique else ""


# ============================================================
# 安全数学求值器 + CalculatorSkill 已抽到 skill_framework.py
# （与 MCP Server 共用同一份实现，避免逻辑分叉）
# ============================================================
from skill_framework import safe_eval, CalculatorSkill  # noqa: E402


# Skill 注册表已抽到 skill_framework.py（与 MCP Server 共用同一份实现）
from skill_framework import SkillRegistry  # noqa: E402


# ============================================================================
# 第四部分：ReAct Agent（推理-行动-观察循环）
# ============================================================================
#
# ReAct = "Reasoning + Acting"
#
# 核心思想：Agent 不是一次性给出答案，而是：
#   1. Thought（思考）：分析当前状态，决定下一步做什么
#   2. Action（行动）：调用某个工具/技能
#   3. Observation（观察）：查看工具返回的结果
#   4. 重复 1-3，直到信息充足，给出 Final Answer

#   现有方式：问题 → 检索 → 生成答案（一步到位，无法应对复杂问题）
#   ReAct方式：问题 → 思考 → 检索 → 观察 → 思考 → 答案
#              （多步推理，可以处理需要组合多种信息源的复杂问题）

@dataclass
class ReActStep:
    """记录 ReAct 循环中的每一步，用于调试和展示推理过程"""
    step_num: int
    thought: str = ""
    action: str = ""
    action_input: str = ""
    observation: str = ""
    is_final: bool = False


class ReActAgent:
    """
    ReAct Agent — 在 Sub Agent 内部使用

    接收一个子任务，通过 ReAct 循环完成它。

    工作流程：
      1. 构造 prompt，包含：子任务描述 + 可用技能列表 + 历史观察结果
      2. 调用 LLM（Ollama qwen2:7b）生成下一步的 Thought + Action
      3. 如果 Action 是某个技能，执行该技能获得 Observation
      4. 如果 LLM 输出 Final Answer，结束循环
      5. 否则把新的 Observation 加入上下文，回到第1步
    """

    # ReAct 系统提示词
    # 这是最关键的部分：告诉 LLM 如何按 ReAct 格式推理
    SYSTEM_PROMPT = """你是一个使用 ReAct（推理+行动）模式的智能助手。

你需要完成以下任务：
{task}

你可以使用以下技能：
{skills}

请严格按照以下格式输出（每行一个字段，不要省略任何字段）：

Thought: 分析当前状态，决定下一步
Action: 技能名称
Action Input: 传给技能的搜索关键词（必须是具体的搜索词，不能为空）

当你通过技能获取到足够信息后，输出：

Thought: 信息已充足
Final Answer: 基于检索结果的回答

示例1（搜索）：
Thought: 我需要搜索文档查找信息
Action: doc_search
Action Input: JM-S509 定位精度 GPS 卫星状态

示例2（列出所有类型）：
Thought: 我需要搜索文档确认所有定位方式类型
Action: doc_search
Action Input: GPS定位包 LBS基站 WiFi定位 FIXPRI 定位方式优先级

示例3（回答）：
Thought: 已获得相关信息，可以回答了
Final Answer: 根据文档，JM-S509支持三种定位方式：GPS定位（协议号0x22/0xA0）、LBS基站定位（协议号0x28/0xA1）、WiFi定位（协议号0x2C/0xA2）。定位精度方面，文档未给出具体数值，但提供了卫星状态信息（GPS模块状态包括搜星、2D定位、3D定位）及定位星数。

重要规则：
1. Action Input 不能为空，必须是具体的搜索关键词
2. 如果任务要求"列出所有类型/方式"，请确保在观察结果中看到了所有类型再回答
3. 如果已有观察结果，直接给出 Final Answer，不要重复搜索
4. Final Answer 要精炼（不超过200字），回答关键信息，如果问题包含多个部分请分别回答
5. 回答必须基于检索到的实际内容，不要编造
6. 回答用中文，引用协议号或文档章节有助于可信度"""

    # 当接近最大步数时，强制要求给出最终答案的提示
    FORCE_FINAL_PROMPT = """你已经有以下观察结果，请直接基于这些结果给出最终答案，不要再调用技能。

观察结果：
{observations}

请输出：
Thought: 基于已有的观察结果，我可以回答了
Final Answer: [你的回答]"""

    def __init__(self, llm: BaseLLM, skill_registry: SkillRegistry, max_steps: int = 3):
        self.llm = llm
        self.skill_registry = skill_registry
        self.max_steps = max_steps  # 防止无限循环（通常2步足够：搜索→回答）

    def run(self, task: str) -> Tuple[str, List[ReActStep]]:
        """
        执行 ReAct 循环来完成子任务

        :param task: 子任务描述
        :return: (最终回答, 推理步骤列表)
        """
        steps: List[ReActStep] = []
        observations: List[str] = []  # 累积的观察结果

        print(f"\n    ┌─ ReAct Agent 启动 ─────────────────────────")
        print(f"    │ 子任务: {task}")
        print(f"    │ 最大步数: {self.max_steps}")

        for step_num in range(1, self.max_steps + 1):
            # --- 构造上下文 ---
            # 把任务、技能列表、历史观察结果都放进 prompt
            skills_desc = self.skill_registry.get_all_descriptions()
            system_prompt = self.SYSTEM_PROMPT.format(task=task, skills=skills_desc)

            context = f"[子任务]: {task}\n"
            if observations:
                # 截断每个观察结果，保留 500 字符避免信息丢失
                truncated_obs = [obs[:500] for obs in observations]
                context += "[已观察到]:\n" + "\n".join(truncated_obs) + "\n"
            context += f"\n请决定第 {step_num} 步的操作。"

            # 接近最大步数时，强制要求给出最终答案
            if step_num >= self.max_steps - 1 and observations:
                context = self.FORCE_FINAL_PROMPT.format(
                    observations="\n".join(observations)
                )

            # --- 调用 LLM 生成下一步 ---
            print(f"\n    │ Step {step_num}: 正在思考...")
            start_time = time.time()
            llm_output = self.llm.chat(system_prompt, context)
            elapsed = time.time() - start_time
            print(f"    │ LLM 响应 ({elapsed:.1f}s)")

            # --- 解析 LLM 输出 ---
            step = self._parse_react_output(llm_output, step_num)
            steps.append(step)

            print(f"    │   Thought: {step.thought}")

            # 检查是否已得出最终答案
            if step.is_final:
                print(f"    │   ★ Final Answer: {step.observation[:200]}")
                print(f"    └─ ReAct Agent 完成（共 {step_num} 步）")
                return step.observation, steps

            # --- 执行 Action ---
            print(f"    │   Action: {step.action}")
            print(f"    │   Action Input: {step.action_input}")

            # 容错：LLM 经常输出空的 Action Input，用子任务描述作为搜索词
            if not step.action_input or len(step.action_input.strip()) < 2:
                step.action_input = task
                print(f"    │   ⚠ Action Input 为空，改用子任务描述: \"{task}\"")

            skill = self.skill_registry.get_skill(step.action)
            if skill:
                # 调用技能，获取观察结果
                step.observation = skill.execute(step.action_input)
                observations.append(step.observation)
                obs_preview = step.observation.replace("\n", " ")[:120]
                print(f"    │   Observation: {obs_preview}...")
            else:
                # 未知技能
                step.observation = f"错误：未知技能 '{step.action}'，可用技能: {list(self.skill_registry._skills.keys())}"
                observations.append(step.observation)
                print(f"    │   ⚠ {step.observation}")

        # 超过最大步数，用最后的观察结果作为答案
        print(f"    └─ ReAct Agent 达到最大步数 {self.max_steps}")
        if observations:
            # 最后再调一次 LLM 让它总结
            final_prompt = f"请根据以下信息简要回答问题「{task}」：\n\n{observations[-1]}"
            final_answer = self.llm.chat("你是文档问答助手，请根据提供的信息简要回答问题。", final_prompt)
            return final_answer, steps
        return "未能完成任务", steps

    def _parse_react_output(self, output: str, step_num: int) -> ReActStep:
        """
        解析 LLM 输出的 ReAct 格式文本

        预期格式1（继续行动）：
          Thought: xxx
          Action: xxx
          Action Input: xxx

        预期格式2（最终答案）：
          Thought: xxx
          Final Answer: xxx

        但 LLM 可能不完全遵循格式，需要做容错处理。
        """
        step = ReActStep(step_num=step_num)
        output = output.strip()

        # 提取 Thought
        thought_match = re.search(r'Thought[:：]\s*(.+?)(?=\n(?:Action|Final|最终)|$)', output, re.DOTALL)
        if thought_match:
            step.thought = thought_match.group(1).strip()

        # 检查是否是最终答案
        final_match = re.search(r'Final Answer[:：]\s*(.+)', output, re.DOTALL)
        if final_match:
            step.is_final = True
            step.observation = final_match.group(1).strip()
            return step

        # 也检查中文 "最终答案"
        final_match_cn = re.search(r'最终答案[:：]\s*(.+)', output, re.DOTALL)
        if final_match_cn:
            step.is_final = True
            step.observation = final_match_cn.group(1).strip()
            return step

        # 提取 Action
        action_match = re.search(r'Action[:：]\s*(.+?)(?=\n(?:Action Input|Action|Final|$))', output, re.DOTALL)
        if action_match:
            step.action = action_match.group(1).strip()
            # 清理 action 中的多余描述（如 "doc_search（文档检索）" → "doc_search"）
            step.action = re.split(r'[（(]', step.action)[0].strip()

        # 提取 Action Input
        input_match = re.search(r'Action Input[:：]\s*(.+?)(?=\n(?:Thought|Action|Final|$))', output, re.DOTALL)
        if input_match:
            step.action_input = input_match.group(1).strip()

        # 容错：如果没提取到 Action 但看起来像是直接回答（且已有观察结果）
        if not step.action and not step.is_final:
            # 如果输出看起来像是一段回答文本，当作 Final Answer
            if len(output) > 20 and "搜索" not in output and "检索" not in output:
                step.is_final = True
                step.observation = output
                return step

        return step


# ============================================================================
# 第五部分：Planning Agent（规划智能体）
# ============================================================================
#
# Planning Agent 的职责：
#   接收用户的复杂问题，拆解成多个可独立执行的子任务。
#
# 为什么需要 Planning？
#   问题："JM-S509定位精度多少？支持几种定位方式？续航如何？"
#   这个问题包含 3 个独立的子问题。
#   如果直接用一个 RAG 检索，可能只检索到其中一部分信息。
#   但如果拆成 3 个子任务，每个子任务独立检索，就能各自找到精准答案。
#
#   现有方式：一个问题 → 一次检索 → 一次生成（容易遗漏信息）
#   Planning方式：一个问题 → 拆解 → 多个子任务各自检索 → 汇总（更完整）

@dataclass
class SubTask:
    """子任务数据结构"""
    id: int
    task: str                # 任务描述
    skill_hint: str = ""     # 建议使用的技能（可选）
    result: str = ""         # 执行结果
    steps: List[ReActStep] = field(default_factory=list)  # ReAct推理步骤


class PlanningAgent:
    """
    Planning Agent — 任务规划器

    工作流程：
      1. 接收用户问题
      2. 调用 LLM（Ollama）分析问题，拆解为子任务列表
      3. 为每个子任务创建 Sub Agent（ReAct Agent）执行
      4. 收集所有子任务结果
      5. 调用 LLM 汇总生成最终答案
    """

    # Planning 系统提示词
    SYSTEM_PROMPT = """你是一个任务规划智能体（Planning Agent）。

用户会提出一个问题，你需要判断是否需要将其拆解为多个独立的子任务。

输出格式（必须是合法JSON，不要包含```json等markdown标记）：
{{"subtasks": [{{"id": 1, "task": "子任务描述", "skill_hint": "doc_search"}}]}}

规则：
1. 如果问题简单（只问一个点），只输出一个子任务
2. 如果问题包含多个独立部分（如"定位精度？定位方式？续航？"），分别拆解为多个子任务，每个子任务只关注一个维度
3. 子任务描述必须包含具体产品型号（如JM-S509）和检索关键词
4. 对于"几种/哪些/多少种"类问题，子任务描述中明确要求"列出所有类型"
5. 对于"精度/准确度"类问题，子任务描述中明确要求"查找卫星状态、定位星数、GPS模块状态等精度相关信息"
6. 对于"续航/电池"类问题，子任务描述中明确要求"查找电量等级、心跳间隔、待机模式、定时设置等续航相关信息"
7. 如果需要计算或换算，添加一个 calculator 类型的子任务，task 中包含数学表达式
8. skill_hint 可选值: "doc_search"（文档检索）、"calculator"（数学计算）
9. 最多拆解为4个子任务"""

    def __init__(self, llm: BaseLLM, skill_registry: SkillRegistry, max_steps: int = 3):
        self.llm = llm
        self.skill_registry = skill_registry
        self.react_agent = ReActAgent(llm, skill_registry, max_steps=max_steps)

    def plan(self, user_query: str) -> List[SubTask]:
        """
        将用户问题拆解为子任务列表

        :param user_query: 用户原始问题
        :return: 子任务列表
        """
        print(f"\n  [PlanningAgent] 开始规划: \"{user_query}\"")

        # 调用 LLM 进行任务拆解
        start_time = time.time()
        result = self.llm.chat(self.SYSTEM_PROMPT, user_query)
        elapsed = time.time() - start_time
        print(f"  [PlanningAgent] LLM 规划完成 ({elapsed:.1f}s)")

        # 解析 LLM 输出的 JSON
        subtasks = self._parse_plan_result(result, user_query)

        print(f"  [PlanningAgent] 拆解为 {len(subtasks)} 个子任务:")
        for st in subtasks:
            print(f"    子任务 {st.id}: {st.task}")
            if st.skill_hint:
                print(f"      → 建议技能: {st.skill_hint}")

        return subtasks

    def _parse_plan_result(self, result: str, fallback_query: str) -> List[SubTask]:
        """解析 LLM 输出的规划结果（JSON），带容错"""
        result = result.strip()

        # 清理可能存在的 markdown 标记
        if "```" in result:
            result = re.sub(r'^```(?:json)?\s*', '', result)
            result = re.sub(r'\s*```$', '', result)

        try:
            data = json.loads(result)
            subtask_dicts = data.get("subtasks", [])

            subtasks: List[SubTask] = []
            for st in subtask_dicts:
                subtask = SubTask(
                    id=st.get("id", len(subtasks) + 1),
                    task=st.get("task", fallback_query),
                    skill_hint=st.get("skill_hint", "doc_search")
                )
                subtasks.append(subtask)

            if not subtasks:
                # JSON 解析成功但子任务列表为空
                subtasks.append(SubTask(id=1, task=fallback_query, skill_hint="doc_search"))

            return subtasks

        except json.JSONDecodeError:
            # JSON 解析失败，退化为单个搜索任务
            print(f"  [PlanningAgent] ⚠ LLM 输出非合法JSON，使用单任务模式")
            return [SubTask(id=1, task=fallback_query, skill_hint="doc_search")]

    def execute_subtask(self, subtask: SubTask) -> SubTask:
        """
        执行单个子任务：创建 Sub Agent 运行 ReAct 循环

        :param subtask: 子任务对象
        :return: 更新后的子任务（包含结果）
        """
        print(f"\n  [PlanningAgent] 执行子任务 {subtask.id}: \"{subtask.task}\"")
        print(f"  {'='*60}")

        # 调用 ReAct Agent 执行子任务
        result, steps = self.react_agent.run(subtask.task)

        subtask.result = result
        subtask.steps = steps
        return subtask

    def synthesize(self, user_query: str, subtasks: List[SubTask]) -> str:
        """
        汇总所有子任务的结果，生成最终答案

        调用 LLM 阅读所有子任务结果，生成一个自然、连贯的最终回答。
        """
        print(f"\n  [PlanningAgent] 汇总 {len(subtasks)} 个子任务结果...")

        # 如果只有一个子任务，直接返回它的结果
        if len(subtasks) == 1 and subtasks[0].result:
            return subtasks[0].result

        # 多个子任务：调用 LLM 汇总
        # 注意：截断每个子任务结果，避免 prompt 过长导致 LLM 生成极慢
        parts = []
        for st in subtasks:
            if st.result:
                # 取前 500 字符，保留足够信息用于准确汇总
                result_truncated = st.result[:500]
                parts.append(f"【子任务{st.id}: {st.task}】\n{result_truncated}")

        if not parts:
            return "未能获取到相关信息。"

        synthesis_system = """你是企业文档问答助手。请根据以下各子任务的检索结果，回答用户的问题。

要求：
- 回答必须基于检索到的实际文档内容，不要编造
- 如果问题包含多个部分（如问定位精度/定位方式/续航），请分别分段回答，每段用小标题标注
- 对于"有几种/哪些方式"类问题，请明确列出每种方式及其协议号
- 回答要精炼（不超过400字），只包含关键信息
- 如果文档中确实没有某项信息，请如实说明，但尽量提供相关线索（如卫星状态、星数等间接信息）
- 用中文回答"""


        synthesis_user = f"用户问题：{user_query}\n\n各子任务检索结果：\n\n" + "\n\n".join(parts) + "\n\n请给出完整回答："

        start_time = time.time()
        final_answer = self.llm.chat(synthesis_system, synthesis_user)
        elapsed = time.time() - start_time
        print(f"  [PlanningAgent] LLM 汇总完成 ({elapsed:.1f}s)")

        return final_answer

    def run(self, user_query: str) -> Tuple[str, List[SubTask]]:
        """
        完整流程：规划 → 执行子任务 → 汇总

        这是 Planning Agent 的主入口。
        """
        # 第1步：规划
        subtasks = self.plan(user_query)

        # 第2步：逐个执行子任务
        for st in subtasks:
            self.execute_subtask(st)

        # 第3步：汇总
        final_answer = self.synthesize(user_query, subtasks)

        return final_answer, subtasks


# ============================================================================
# 第六部分：Orchestrator（编排器）— 系统入口
# ============================================================================

class RAGOrchestrator:
    """
    系统编排器 — 把所有组件组装在一起

    职责：
      1. 初始化 LLM 后端（Ollama）
      2. 初始化向量数据库（ChromaDB）
      3. 注册所有 Skill
      4. 创建 Planning Agent
      5. 提供统一的 query 入口
    """

    def __init__(self, llm: BaseLLM, vector_db, fast_mode: bool = False,
                 cache: CacheManager = None, user_role: str = DEFAULT_ROLE):
        self.llm = llm
        self.vector_db = vector_db
        self.fast_mode = fast_mode
        self.user_role = user_role
        self.cache = cache or CacheManager()  # 默认自动连接 Redis，连不上则跳过

        # 设置缓存管理器的当前角色（影响缓存键，防止跨角色泄漏）
        self.cache.current_role = user_role

        # --- 注册技能 ---
        print("\n[系统] 注册技能...")
        self.skill_registry = SkillRegistry()

        # 文档检索技能（智能 RAG，使用真实 ChromaDB）
        # fast_mode=True 时跳过查询重写，减少 LLM 调用
        # user_role 传入 DocSearchSkill，检索时按权限过滤结果
        self.skill_registry.register(
            DocSearchSkill(llm, vector_db, fast_mode=fast_mode, user_role=user_role)
        )
        # 计算器技能
        self.skill_registry.register(CalculatorSkill())

        # --- 创建 Planning Agent ---
        self.planning_agent = PlanningAgent(llm, self.skill_registry)

        role_desc = AccessControlFilter.get_role_description(user_role)
        print(f"[系统] 当前用户角色: {user_role}（{role_desc}）")
        print("[系统] 初始化完成\n")

    def query(self, user_question: str, user_role: str = None) -> str:
        """
        用户提问入口

        完整流程：
          用户问题 → 检查 Redis 缓存（命中直接返回）
          → PlanningAgent 规划 → Sub Agent + ReAct 执行
          → 智能 RAG 检索（ChromaDB）→ LLM 汇总答案
          → 写入 Redis 缓存

        :param user_question: 用户问题
        :param user_role: 可选，覆盖默认用户角色（"admin" 或 "user"）
        """
        # 如果传入了 user_role，临时切换角色
        if user_role and user_role != self.user_role:
            self.user_role = user_role
            self.cache.current_role = user_role
            # 更新 DocSearchSkill 的角色
            doc_skill = self.skill_registry.get_skill("doc_search")
            if doc_skill:
                doc_skill.user_role = user_role
            role_desc = AccessControlFilter.get_role_description(user_role)
            print(f"\n[系统] 已切换用户角色: {user_role}（{role_desc}）")

        total_start = time.time()

        print("\n" + "=" * 70)
        print(f"用户提问: {user_question}")
        print(f"用户角色: {self.user_role}（{AccessControlFilter.get_role_description(self.user_role)}）")
        print("=" * 70)

        # ====== 缓存检查 ======
        # 两级匹配：先精确匹配（SHA256哈希），再语义匹配（embedding余弦相似度）
        cached_answer = self.cache.lookup(user_question)
        if cached_answer:
            total_elapsed = time.time() - total_start
            print("\n" + "=" * 70)
            print(f"从缓存返回答案（耗时 {total_elapsed:.1f}s，跳过了 LLM 推理）:")
            print("=" * 70)
            print(cached_answer)
            stats = self.cache.stats
            print(f"\n  [缓存统计] 命中:{stats['hits']} 未命中:{stats['misses']} 命中率:{stats['hit_rate']}")
            return cached_answer

        # 缓存未命中 → 执行完整的 Agent 流程
        final_answer, subtasks = self.planning_agent.run(user_question)

        total_elapsed = time.time() - total_start

        # 打印最终结果
        print("\n" + "=" * 70)
        print("最终回答:")
        print("=" * 70)
        print(final_answer)

        # 打印推理过程摘要
        print("\n" + "-" * 70)
        print(f"推理过程摘要（总耗时 {total_elapsed:.1f}s）:")
        print("-" * 70)
        for st in subtasks:
            print(f"\n  子任务 {st.id}: {st.task}")
            print(f"  推理步数: {len(st.steps)}")
            for step in st.steps:
                status = "✓ 最终答案" if step.is_final else f"→ {step.action}({step.action_input[:30]})"
                print(f"    Step {step.step_num}: {status}")
            result_preview = st.result.replace("\n", " ")[:100]
            print(f"  结果: {result_preview}...")

        print(f"\n  总耗时: {total_elapsed:.1f}s")

        # 打印 LLM 调用统计
        if hasattr(self.llm, 'call_count'):
            print(f"  LLM 调用次数: {self.llm.call_count}")
            print(f"  LLM 总耗时: {self.llm.total_time:.1f}s (占 {self.llm.total_time/total_elapsed*100:.0f}%)")

        stats = self.cache.stats
        if stats["total"] > 0:
            print(f"  [缓存统计] 命中:{stats['hits']} 未命中:{stats['misses']} 命中率:{stats['hit_rate']}")

        # ====== 写入缓存 ======
        # 把这次 LLM 推理的结果缓存起来，下次同样/相似问题直接返回
        self.cache.save(user_question, final_answer)

        return final_answer


# ============================================================================
# 第七部分：主程序入口
# ============================================================================

def run_demo(orchestrator: RAGOrchestrator):
    """运行内置演示问题"""
    demo_questions = [
        "JM-S509学生证的定位精度是多少？支持几种定位方式？电池续航如何？",
    ]

    for i, question in enumerate(demo_questions, 1):
        print(f"\n\n{'█' * 70}")
        print(f"█  演示 {i}")
        print(f"{'█' * 70}")
        orchestrator.query(question)


def run_interactive(orchestrator: RAGOrchestrator):
    """交互模式"""
    role_desc = AccessControlFilter.get_role_description(orchestrator.user_role)
    print(f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║          高级 RAG Agent — 交互模式                               ║
    ║          ChromaDB + Ollama(qwen2:7b) + ReAct + Planning Agent   ║
    ║          用户角色: {orchestrator.user_role}（{role_desc}）
    ╚══════════════════════════════════════════════════════════════════╝

    输入问题提问，输入 exit 退出。
    输入 /admin 切换为特权用户，输入 /user 切换为普通用户。

    示例问题：
      - JM-S509的定位精度是多少？支持几种定位方式？
      - 设备的电池续航和待机时间是多少？
      - 通讯协议端口和心跳间隔是多少？
      - 待机时间120小时换算成天是多少？
    """)

    while True:
        try:
            prompt = f"\n[{orchestrator.user_role}] 请输入问题 >> "
            question = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "退出"):
            print("再见！")
            break

        # 角色切换命令
        if question.lower() in ("/admin", "/特权"):
            orchestrator.user_role = ROLE_ADMIN
            orchestrator.cache.current_role = ROLE_ADMIN
            doc_skill = orchestrator.skill_registry.get_skill("doc_search")
            if doc_skill:
                doc_skill.user_role = ROLE_ADMIN
            print(f"  ✓ 已切换为特权用户（可访问所有文档）")
            continue
        if question.lower() in ("/user", "/普通"):
            orchestrator.user_role = ROLE_USER
            orchestrator.cache.current_role = ROLE_USER
            doc_skill = orchestrator.skill_registry.get_skill("doc_search")
            if doc_skill:
                doc_skill.user_role = ROLE_USER
            print(f"  ✓ 已切换为普通用户（仅可访问公开文档）")
            continue

        try:
            orchestrator.query(question)
        except Exception as e:
            print(f"\n❌ 处理出错: {e}")
            traceback.print_exc()


def main():
    """主入口"""
    # 解析命令行参数
    args = sys.argv[1:]
    demo_mode = "--demo" in args
    fast_mode = "--fast" in args
    admin_mode = "--admin" in args  # 特权用户模式
    # 提取直接提问的问题（非 -- 开头的参数）
    direct_question = None
    for arg in args:
        if not arg.startswith("--"):
            direct_question = arg
            break

    # 确定用户角色
    user_role = ROLE_ADMIN if admin_mode else DEFAULT_ROLE
    role_desc = AccessControlFilter.get_role_description(user_role)
    mode_label = "快速模式" if fast_mode else "完整模式"
    print(f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║          高级 RAG Agent — 真实环境版                              ║
    ║          ReAct + Planning Agent + 智能 RAG + Skill 系统          ║
    ║          ChromaDB + Ollama(qwen2:7b)  [{mode_label}]                     ║
    ║          用户角色: {user_role}（{role_desc}）
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # 1. 初始化 LLM
    try:
        llm = OllamaLLM()
    except Exception as e:
        print(f"\n❌ 无法连接 Ollama: {e}")
        print(f"   请确认 Ollama 已启动: {OLLAMA_URL}")
        print(f"   并已加载模型: {MODEL_NAME}")
        return

    # 2. 初始化向量数据库
    try:
        vector_db = VectorStoreManager.init_vector_store()
    except Exception as e:
        print(f"\n❌ 初始化向量数据库失败: {e}")
        print(f"   请确认 {DB_PATH} 存在或 {DOC_FOLDER} 中有文档")
        return

    # 3. 创建编排器（传入 fast_mode 和 user_role）
    orchestrator = RAGOrchestrator(llm, vector_db, fast_mode=fast_mode,
                                   user_role=user_role)

    # 4. 运行
    if direct_question:
        # 命令行直接提问
        orchestrator.query(direct_question)
    elif demo_mode:
        run_demo(orchestrator)
    else:
        run_interactive(orchestrator)


if __name__ == "__main__":
    main()
