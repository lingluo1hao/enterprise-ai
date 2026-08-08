# -*- coding: utf-8 -*-
"""
自进化层（Evolution Layer）— 方案 A：在现有 RAG 引擎上嫁接 Hermes 式自进化
============================================================================

核心思想（对齐 Hermes 三子系统）：
  1. Skill 自动生成  -> Extractor：每次成功问答后，抽取一个 RetrievalPlaybook
  2. Skill 持续进化  -> PlaybookStore：命中相似 playbook 时复用已知好 rewrite，
                        下次同类问题跳过首轮 LLM 改写（越用越快）
  3. Nudge 反思引擎  -> （P1 阶段补；本 P0 仅落地 1+2 的最小闭环）

设计原则：
  - 零新依赖：复用现有 VectorStoreManager（Milvus 客户端 self.client）与
    self._embed（Ollama bge-m3 embedding），不引任何新包。
  - 零风险：所有外部调用（Milvus 建表 / 写入 / 查询）均 try/except 包裹，
    任何失败只打印告警，绝不影响主问答链路。
  - 租户隔离天然延续：playbook 存储与查询都带 tenant_id 过滤，
    与 rag_docs 集合物理隔离、互不干扰。

触发信号（复用现有逻辑，不新增任何前端反馈端点）：
  route_after_grade 在 relevant_count >= GRADE_THRESHOLD 时判定"走通"。
  Extractor.extract 据此判定本次问答是否值得沉淀为经验。
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

# 复用主文件的成功信号常量（本地定义，避免与 langgraph_rag_agent 形成 import 环）
GRADE_THRESHOLD = 1
MAX_RETRIEVAL_ROUNDS = 3

# Milvus 集合名（独立于 rag_docs，互不影响）
PLAYBOOK_COLLECTION = "skill_playbooks"

# COSINE 距离阈值：pymilvus search 返回 distance，COSINE 度量下
#   distance = 0   -> 完全相同（相似度 1.0）
#   distance = 2   -> 完全相反（相似度 -1.0）
# distance <= HIT_DIST 视为"同类问题"，复用其已知好的 rewrite。
# 0.22 即 cosine 相似度 >= 0.78（足够区分不同意图，又不至于过严漏命中）。
HIT_DIST = 0.22


@dataclass
class RetrievalPlaybook:
    """一次成功检索问答沉淀出的经验。"""
    intent_text: str                 # 消解后问题（作为向量检索 anchor）
    query_type: str                  # simple / complex
    rewrite_text: str                # 已知好的首轮改写（JSON 数组字符串）
    node_path: str                   # 走的分支（simple/complex）
    relevant_sources: str            # 相关源文档信息（JSON 数组字符串，用于治理/溯源）
    tenant_id: str
    user_id: str
    success_count: int = 1
    updated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    pb_id: Optional[str] = None


class PlaybookStore:
    """playbook 持久化（Milvus skill_playbooks 集合）。"""

    def __init__(self, vector_db, collection_name: str = PLAYBOOK_COLLECTION):
        # vector_db 即 LangGraphRAGApp 持有的 VectorStoreManager / AdvancedRagAgent 实例
        self.vdb = vector_db
        self.client = getattr(vector_db, "client", None)
        self.collection = collection_name
        self._ensure()

    # ---------------------------------------------------------------- 建表
    def _ensure(self):
        if self.client is None:
            print("[Evolution] ⚠ vector_db.client 为 None，PlaybookStore 不可用（降级跳过）")
            return
        try:
            if self.client.has_collection(self.collection):
                self._load_if_needed()
                return
            # 维度动态探测（与 AdvancedRagAgent._ensure_collection 同思路，避免硬编码）
            dim = len(self.vdb._embed.embed_query("维度探测"))
            from pymilvus import FieldSchema, CollectionSchema, DataType
            fields = [
                FieldSchema("pk", DataType.VARCHAR, is_primary=True,
                            max_length=36, auto_id=False),
                FieldSchema("intent_vector", DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema("intent_text", DataType.VARCHAR, max_length=4096),
                FieldSchema("query_type", DataType.VARCHAR, max_length=16),
                FieldSchema("rewrite_text", DataType.VARCHAR, max_length=2048),
                FieldSchema("node_path", DataType.VARCHAR, max_length=16),
                FieldSchema("relevant_sources", DataType.VARCHAR, max_length=8192),
                FieldSchema("success_count", DataType.INT64),
                FieldSchema("tenant_id", DataType.VARCHAR, max_length=64),
                FieldSchema("user_id", DataType.VARCHAR, max_length=64),
                FieldSchema("updated_at", DataType.VARCHAR, max_length=32),
            ]
            schema = CollectionSchema(fields, enable_dynamic_field=True)
            self.client.create_collection(collection_name=self.collection, schema=schema)
            idx_params = self.client.prepare_index_params()
            idx_params.add_index(field_name="intent_vector", index_type="AUTOINDEX",
                                 metric_type="COSINE")
            self.client.create_index(self.collection, index_params=idx_params)
            self._load_if_needed()
            print(f"[Evolution] ✔ 已创建 playbook 集合 {self.collection} (dim={dim})")
        except Exception as e:
            print(f"[Evolution] ⚠ 建表/索引失败(降级跳过): {e}")

    # ---------------------------------------------------------------- 写入
    def save(self, pb: RetrievalPlaybook):
        if self.client is None:
            return
        try:
            vec = self.vdb._embed.embed_query(pb.intent_text)
            data = [{
                "pk": pb.pb_id or str(uuid.uuid4()),
                "intent_vector": vec,
                "intent_text": pb.intent_text[:4096],
                "query_type": pb.query_type,
                "rewrite_text": pb.rewrite_text,
                "node_path": pb.node_path,
                "relevant_sources": pb.relevant_sources,
                "success_count": pb.success_count,
                "tenant_id": pb.tenant_id,
                "user_id": pb.user_id,
                "updated_at": pb.updated_at,
            }]
            self.client.insert(self.collection, data)
            self.client.flush(self.collection)
            print(f"[Evolution] ✔ 沉淀 playbook: {pb.intent_text[:40]} (tenant={pb.tenant_id})")
        except Exception as e:
            print(f"[Evolution] ⚠ save 失败(忽略): {e}")

    # ---------------------------------------------------------------- 加载
    def _load_if_needed(self):
        """确保集合已 load 到 Milvus 内存；未 loaded 则 load，全 try/except 降级。

        Milvus 集合 create+index 后必须 load 才能 search，否则报
        code=101 collection not loaded。Milvus 重启后集合会变 NotLoaded，
        这里每次 query 前兜底确保，进程启动建表时也 load。
        """
        if self.client is None:
            return
        try:
            state = self.client.get_load_state(self.collection)
            if state != "Loaded":
                self.client.load_collection(self.collection)
                print(f"[Evolution] ✔ 已 load playbook 集合 {self.collection}")
        except Exception as e:
            print(f"[Evolution] ⚠ load 失败(降级跳过): {e}")

    # ---------------------------------------------------------------- 查询
    def query_similar(self, intent_text: str, tenant_id: str, top_k: int = 3):
        """返回最相似的命中 playbook（含已知好 rewrite），无命中返回 None。"""
        if self.client is None:
            return None
        self._load_if_needed()
        try:
            vec = self.vdb._embed.embed_query(intent_text)
            expr = f'(tenant_id == "{tenant_id}")'
            hits = self.client.search(
                self.collection, data=[vec], anns_field="intent_vector",
                limit=top_k, filter=expr,
                output_fields=["intent_text", "query_type", "rewrite_text",
                               "node_path", "success_count"],
            )[0]
            for h in hits:
                dist = h.get("distance", h.get("score"))
                if dist is None:
                    continue
                if dist <= HIT_DIST:
                    ent = h.get("entity", h.get("fields", {}))
                    return {
                        "pk": h.get("id"),
                        "score": round(1.0 - dist, 4),  # 转成易读的相似度
                        "query_type": ent.get("query_type"),
                        "rewrite_text": ent.get("rewrite_text"),
                        "node_path": ent.get("node_path"),
                        "success_count": ent.get("success_count", 0),
                    }
            return None
        except Exception as e:
            print(f"[Evolution] ⚠ query 失败(忽略): {e}")
            return None

    # ---------------------------------------------------------------- 进化（P1 预留）
    def patch_success(self, pk: str):
        """命中并复用后回调：success_count +1（P1 阶段实现，P0 先留接口）。

        Milvus 不支持原地 update，P1 用 delete + insert 或 collection alias 实现。
        P0 阶段复用已足够产生"越用越快"效果，计数暂不回写。
        """
        pass


class Extractor:
    """从一次（走通的）问答 state 中抽取 RetrievalPlaybook。"""

    @staticmethod
    def extract(state: dict, tenant_id: str, user_id: str):
        grades = state.get("doc_grades", []) or []
        relevant = sum(1 for g in grades if g)
        # 成功信号：真相关（relevant >= 1），不沉淀"硬凑满 3 轮仍不相关"的失败路径
        if relevant < GRADE_THRESHOLD:
            return None

        intent_text = (state.get("resolved_query") or state.get("query") or "").strip()
        if not intent_text:
            return None

        qtype = state.get("query_type", "simple")
        rewrites = state.get("rewritten_queries", []) or []

        # 相关源：grade=True 的文档 metadata（用于经验溯源 / 治理）
        docs = state.get("retrieved_docs", []) or []
        sources = []
        for i, item in enumerate(docs):
            if i < len(grades) and grades[i]:
                doc = item[0] if isinstance(item, (list, tuple)) else item
                md = getattr(doc, "metadata", {}) or {}
                sources.append({
                    "file_path": md.get("file_path"),
                    "file_name": md.get("file_name"),
                    "chunk_index": md.get("chunk_index"),
                    "tenant_id": md.get("tenant_id"),
                })

        return RetrievalPlaybook(
            intent_text=intent_text,
            query_type=qtype,
            rewrite_text=json.dumps(rewrites, ensure_ascii=False),
            node_path=qtype,
            relevant_sources=json.dumps(sources, ensure_ascii=False),
            tenant_id=tenant_id,
            user_id=user_id,
        )
