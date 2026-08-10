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

# 三级成功信号阈值（强化自进化 #168）
# L2 答案级：若管线产出了 faithfulness_score（0~1），需 >= 该阈值才视为答案可信
ANSWER_FAITH_THRESHOLD = 0.5
# L3 用户反馈级：qa_feedback.rating 取值 -1=踩 / 0=无 / 1=赞
FEEDBACK_POSITIVE = 1      # 赞 -> 正向强化（确认经验有效）
FEEDBACK_NEGATIVE = -1     # 踩 -> 负样本（该经验应被否定，沉淀到 bad_cases）
# 去重合并阈值：比 HIT_DIST 更紧。distance <= MERGE_DIST 视为"同一问题"，
# 不重复插新 playbook，而是 patch_success（计数 +1）。
MERGE_DIST = 0.10


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
    # 三级成功信号达到的最高正向层级：1=检索级 / 2=答案级 / 3=用户反馈级。
    # 仅作经验置信度标记，供治理/排序参考（不影响复用逻辑）。
    success_level: int = 1
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
    def _collection_dim(self) -> int:
        """探测已存在集合 intent_vector 的实际维度（与 AdvancedRagAgent 同思路）。"""
        try:
            fs = self.client.describe_collection(self.collection).get("fields", [])
            for f in fs:
                if f.get("type") == "FloatVector" or f.get("data_type") == 101:
                    return int(f.get("params", {}).get("dim", 0))
        except Exception:
            pass
        return 0

    def _ensure(self):
        if self.client is None:
            print("[Evolution] ⚠ vector_db.client 为 None，PlaybookStore 不可用（降级跳过）")
            return
        try:
            # 维度动态探测（与 AdvancedRagAgent._ensure_collection 同思路，避免硬编码）
            dim = len(self.vdb._embed.embed_query("维度探测"))
            if self.client.has_collection(self.collection):
                # 维度漂移防护：若集合是旧 embedder 建的（维度不一致），整集合重建。
                # 旧 playbook 维度与当前 embedder 不兼容，无法复用，重建是唯一正确路径。
                old_dim = self._collection_dim()
                if old_dim and old_dim != dim:
                    print(f"[Evolution] ⚠ 检测到 skill_playbooks 维度漂移 "
                          f"(旧={old_dim}, 新={dim})，重建集合以自修复自进化能力")
                    try:
                        self.client.drop_collection(self.collection)
                    except Exception as _e:
                        print(f"[Evolution] ⚠ drop_collection 失败(降级跳过): {_e}")
                        return
                else:
                    self._load_if_needed()
                    return
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
    def _is_match(self, query_text: str, hit_text: str, dist, dist_thresh: float) -> bool:
        """判定一次命中是否算「同类问题」。

        优先用 Milvus 余弦距离（dist <= dist_thresh）。但该环境 standalone Milvus
        的 AUTOINDEX+COSINE 在集合较小/索引未充分构建时，会返回失真距离（相同向量
        也可能返回 1.0），不可全信。因此叠加「文本相似度」兜底：
          - 完全一致的 intent_text 直接判命中（最可靠，不依赖向量）
          - difflib 之比 >= 0.92 视为近重复（去重/复用场景足够）
        这样无论 Milvus 距离是否失真，去重与复用都正确。
        """
        if dist is not None and dist <= dist_thresh:
            return True
        if query_text and hit_text:
            qt, ht = query_text.strip(), hit_text.strip()
            if qt and ht and qt == ht:
                return True
            try:
                import difflib
                if difflib.SequenceMatcher(None, qt, ht).ratio() >= 0.92:
                    return True
            except Exception:
                pass
        return False

    def query_similar(self, intent_text: str, tenant_id: str, top_k: int = 3,
                      dist_thresh: float = HIT_DIST):
        """返回最相似的命中 playbook（含已知好 rewrite），无命中返回 None。

        dist_thresh 可覆盖（去重合并时用更紧的 MERGE_DIST）。命中判定结合
        Milvus 距离 + intent_text 文本相似度兜底（见 _is_match）。
        """
        if self.client is None:
            return None
        self._load_if_needed()
        try:
            vec = self.vdb._embed.embed_query(intent_text)
            expr = f'(tenant_id == "{tenant_id}")'
            hits = self.client.search(
                self.collection, data=[vec], anns_field="intent_vector",
                limit=top_k, filter=expr, consistency_level="Strong",
                output_fields=["intent_text", "query_type", "rewrite_text",
                               "node_path", "success_count"],
            )[0]
            best = None
            for h in hits:
                dist = h.get("distance", h.get("score"))
                ent = h.get("entity", h.get("fields", {})) or {}
                hit_text = ent.get("intent_text") or h.get("intent_text") or ""
                if self._is_match(intent_text, hit_text, dist, dist_thresh):
                    best = {
                        "pk": h.get("id"),
                        "score": round(1.0 - dist, 4) if dist is not None else 1.0,
                        "query_type": ent.get("query_type"),
                        "rewrite_text": ent.get("rewrite_text"),
                        "node_path": ent.get("node_path"),
                        "success_count": ent.get("success_count", 0),
                    }
                    break
            return best
        except Exception as e:
            print(f"[Evolution] ⚠ query 失败(忽略): {e}")
            return None

    # ---------------------------------------------------------------- 进化（强化自进化 #168）
    def patch_success(self, pk: str):
        """命中并复用后回调：success_count +1。

        Milvus 不支持原地 update，用 delete(old) + insert(updated) 实现计数回写。
        全程 try/except 降级，绝不抛异常影响主链路。
        """
        if self.client is None or not pk:
            return
        try:
            self._load_if_needed()
            res = self.client.query(
                self.collection, filter=f'pk == "{pk}"', output_fields=["*"]
            )
            if not res:
                return
            row = dict(res[0])
            # 清理 Milvus query 可能夹带的元字段
            for _k in ("distance", "score", "id"):
                row.pop(_k, None)
            row["success_count"] = int(row.get("success_count", 0)) + 1
            row["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.client.delete(self.collection, filter=f'pk == "{pk}"')
            self.client.insert(self.collection, [row])
            self.client.flush(self.collection)
            print(f"[Evolution] ✔ pk={pk} success_count -> {row['success_count']}")
        except Exception as e:
            print(f"[Evolution] ⚠ patch_success 失败(忽略): {e}")

    def reinforce_feedback(self, pk: str, positive: bool):
        """用户反馈级信号回调：赞 -> success_count +2（强确认）；踩 -> 标记降权。

        踩的负反馈不直接删经验（避免误杀），而是把 success_count 压到 0 并在
        updated_at 打上反馈时间戳，供 triage/治理识别为"存疑经验"。
        """
        if self.client is None or not pk:
            return
        try:
            self._load_if_needed()
            res = self.client.query(
                self.collection, filter=f'pk == "{pk}"', output_fields=["*"]
            )
            if not res:
                return
            row = dict(res[0])
            for _k in ("distance", "score", "id"):
                row.pop(_k, None)
            cur = int(row.get("success_count", 0))
            if positive:
                row["success_count"] = cur + 2
            else:
                row["success_count"] = 0
            row["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.client.delete(self.collection, filter=f'pk == "{pk}"')
            self.client.insert(self.collection, [row])
            self.client.flush(self.collection)
            print(f"[Evolution] ✔ pk={pk} 反馈强化(positive={positive}) -> "
                  f"success_count={row['success_count']}")
        except Exception as e:
            print(f"[Evolution] ⚠ reinforce_feedback 失败(忽略): {e}")

    def save_or_merge(self, pb: "RetrievalPlaybook"):
        """沉淀经验：若已存在高度相似（dist <= MERGE_DIST）的 playbook，则去重
        合并（patch_success 计数 +1，不重复插）；否则新建。

        返回被写入/命中的 pk；失败返回 None。
        """
        if self.client is None:
            return None
        try:
            hit = self.query_similar(pb.intent_text, pb.tenant_id,
                                     top_k=1, dist_thresh=MERGE_DIST)
            if hit and hit.get("pk"):
                # 已存在同一问题经验，去重：复用计数 +1（越用越快）
                self.patch_success(hit["pk"])
                return hit["pk"]
            pb.pb_id = pb.pb_id or str(uuid.uuid4())
            self.save(pb)
            return pb.pb_id
        except Exception as e:
            print(f"[Evolution] ⚠ save_or_merge 失败(忽略): {e}")
            return None


class Extractor:
    """从一次（走通的）问答 state 中抽取 RetrievalPlaybook。

    强化自进化（#168）：引入三级成功信号判定，只有"值得学"的经验才沉淀。
      L1 检索级（硬性闸门）：doc_grades 中真相关数 >= GRADE_THRESHOLD
      L2 答案级（软性）：若管线产出 faithfulness_score，需 >= ANSWER_FAITH_THRESHOLD
      L3 用户反馈级（异步）：qa_feedback.rating == 赞 正向 / 踩 负向
    任意负向信号（L2/L3 为 False）都会阻止正向沉淀，并改走 extract_failure。
    """

    @staticmethod
    def evaluate_success(state: dict, feedback_rating=None):
        """三级成功信号评估。返回 (ok, level)。

        ok   —— 是否可作为正向经验沉淀（检索级硬闸门 + L2/L3 不为负）
        level—— 达到的最高正向层级 1/2/3（仅正向时有效；全无信号为 1）
        """
        grades = state.get("doc_grades", []) or []
        relevant = sum(1 for g in grades if g)
        retrieval_ok = relevant >= GRADE_THRESHOLD

        # L2 答案级（可选信号，无则中性）
        fs = state.get("faithfulness_score")
        answer_ok = None
        if fs is not None:
            try:
                answer_ok = float(fs) >= ANSWER_FAITH_THRESHOLD
            except (TypeError, ValueError):
                answer_ok = None

        # L3 用户反馈级（可选信号，无则中性）
        feedback_ok = None
        if feedback_rating is not None:
            try:
                r = int(feedback_rating)
            except (TypeError, ValueError):
                r = 0
            if r >= FEEDBACK_POSITIVE:
                feedback_ok = True
            elif r <= FEEDBACK_NEGATIVE:
                feedback_ok = False

        # 综合：检索级是硬闸门；L2/L3 任一为 False 即不可作为正向经验
        ok = retrieval_ok and (answer_ok is not False) and (feedback_ok is not False)
        level = 1 if retrieval_ok else 0
        if answer_ok:
            level = max(level, 2)
        if feedback_ok:
            level = max(level, 3)
        return ok, level

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

        # 三级成功信号 -> 经验置信层级（仅作标记，不影响复用）
        _, level = Extractor.evaluate_success(state)

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
            success_level=level,
        )

    @staticmethod
    def extract_failure(state: dict, tenant_id: str, user_id: str, reason: str = None):
        """抽取一次失败问答的元信息，供写入 bad_cases（负样本闭环）。

        触发场景：检索未命中相关文档，或答案级/反馈级信号为负。
        返回可直接喂给 memory_store.add_bad_case 的字段字典；无法构造返回 None。
        """
        query = (state.get("resolved_query") or state.get("query") or "").strip()
        if not query:
            return None
        grades = state.get("doc_grades", []) or []
        relevant = sum(1 for g in grades if g)
        docs = state.get("retrieved_docs", []) or []

        # 收集被召回但不相关的源，辅助 root_cause 诊断
        bad_sources = []
        for i, item in enumerate(docs):
            if i >= len(grades) or not grades[i]:
                doc = item[0] if isinstance(item, (list, tuple)) else item
                md = getattr(doc, "metadata", {}) or {}
                fn = md.get("file_name")
                if fn and fn not in bad_sources:
                    bad_sources.append(fn)

        is_retrieval_fail = relevant < GRADE_THRESHOLD
        root_cause = reason or (
            "检索未命中任何相关文档" if is_retrieval_fail else "答案级/反馈级信号为负"
        )
        diagnosis = json.dumps(
            {"relevant": relevant, "bad_sources": bad_sources,
             "query_type": state.get("query_type")},
            ensure_ascii=False,
        )
        return {
            "query": query,
            "source": "pipeline",
            "suite": "retrieval",
            "expected": f"相关文档数应 >= {GRADE_THRESHOLD}，实际 = {relevant}",
            "root_cause": root_cause,
            "diagnosis": diagnosis,
            "tenant_id": tenant_id,
            "user_id": user_id,
        }
