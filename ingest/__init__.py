"""百万级 RAG 数据面 ingestion 引擎（自包含，零循环依赖）。

对外暴露核心类，便于主模块与测试直接引用：
  IngestPipeline / StoreBackend / MilvusStoreBackend / MemoryStoreBackend
  BatchEmbedder / make_ollama_embedder
  compute_fingerprint / ManifestStore / diff_fingerprints
  RawDoc / Chunk / Entity / RunReport
"""

from .types import RawDoc, Chunk, Entity, RunReport
from .fingerprint import compute_fingerprint, ManifestStore, diff_fingerprints
from .loaders import load_file, get_access_level
from .chunk import chunk_documents
from .embed import BatchEmbedder, make_ollama_embedder
from .store import StoreBackend, MilvusStoreBackend, MemoryStoreBackend
from .pipeline import IngestPipeline, SUPPORTED_EXT

__all__ = [
    "RawDoc", "Chunk", "Entity", "RunReport",
    "compute_fingerprint", "ManifestStore", "diff_fingerprints",
    "load_file", "get_access_level",
    "chunk_documents",
    "BatchEmbedder", "make_ollama_embedder",
    "StoreBackend", "MilvusStoreBackend", "MemoryStoreBackend",
    "IngestPipeline", "SUPPORTED_EXT",
]
