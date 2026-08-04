"""百万级 RAG 数据面 —— 集成测试（需 Milvus + Ollama，VM 环境运行）。

自动跳过：Milvus/Ollama 不可达时整个模块 skip，不影响普通开发机跑测试。

运行（VM 上）：
  python -m pytest tests/test_ingest_integration.py -s

它会在临时 knowledge/ 里放一个真实 .txt，走完整的
「多格式 loader → 批量 embedding(Ollama bge-m3) → Milvus upsert」链路，
验证生产路径与单元测试的引擎一致。
"""

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest  # 仅集成测试依赖 pytest

try:
    from pymilvus import MilvusClient
    from ingest.embed import make_ollama_embedder
    from ingest.store import MilvusStoreBackend
    from ingest.pipeline import IngestPipeline
except Exception as e:  # noqa
    pytest.skip(f"缺少依赖：{e}", allow_module_level=True)


def _reachable() -> bool:
    try:
        c = MilvusClient(uri=os.getenv("MILVUS_URI", "http://192.168.200.128:19530"))
        c.has_collection(os.getenv("MILVUS_COLLECTION", "rag_docs"))
        # 顺便探一下 Ollama embedding 是否通
        make_ollama_embedder().embed_query("ping")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _reachable(),
    reason="Milvus/Ollama 不可达（需在已部署向量库的 VM 环境运行）",
)


def test_incremental_against_milvus():
    folder = tempfile.mkdtemp()
    f1 = os.path.join(folder, "a.txt")
    with open(f1, "w", encoding="utf-8") as f:
        f.write("集成测试文档一 " * 30)

    client = MilvusClient(uri=os.getenv("MILVUS_URI", "http://192.168.200.128:19530"))
    collection = os.getenv("MILVUS_COLLECTION", "rag_docs")
    store = MilvusStoreBackend(client, collection)
    pipe = IngestPipeline(
        folder=folder,
        embedder=make_ollama_embedder().embed_documents,
        store=store,
    )
    rep = pipe.run(progress_cb=print)
    pipe.close()
    assert rep.entities_upserted > 0, rep.summary()
    # 二次运行应无新增（增量）
    pipe2 = IngestPipeline(
        folder=folder,
        embedder=make_ollama_embedder().embed_documents,
        store=store,
    )
    rep2 = pipe2.run()
    pipe2.close()
    assert rep2.entities_upserted == 0, rep2.summary()
