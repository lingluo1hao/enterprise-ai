"""批量 embedding（改造点 4）：攒批 + 并发 worker 池 + 失败重试。

为什么这么改：
- 现状每一 chunk 一次 `embed_documents([c])` HTTP 往返；百万文档 = 上亿次往返，
  瓶颈在 RTT 而非算力。
- 改成攒批（如 64 条/请求）+ 线程池并发多批，把 RTT 摊薄到 N 分之一；
  单批失败按指数退避重试，保证大批量任务能从瞬时错误中自愈、可断点续跑。

这是 embedding 服务的基本常识：OpenAI/LangChain 的 embed_documents 本就接收 batch，
GPU/CUDA embedding 更是强制 batch 才能跑满显存。
"""

import os
import time
import threading
from typing import Callable, List


class BatchEmbedder:
    def __init__(self, embed_fn: Callable[[List[str]], List[List[float]]],
                 batch_size: int = 64, max_concurrency: int = 4,
                 max_retries: int = 3, retry_backoff: float = 1.0):
        """
        :param embed_fn: 底层 embedding 函数，接收一批文本返回一批向量
                         （与 OllamaEmbeddings.embed_documents 签名一致）
        :param batch_size: 每批文本条数
        :param max_concurrency: 同时进行的批次数（并发上限，避免打爆 Ollama）
        :param max_retries: 单批最大重试次数
        :param retry_backoff: 重试基础退避秒数（实际等待 = backoff * 2^attempt）
        """
        self.embed_fn = embed_fn
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._sem = threading.Semaphore(max_concurrency)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """对一批文本批量向量化，返回与输入顺序一致的向量列表。"""
        if not texts:
            return []
        batches = [texts[i:i + self.batch_size]
                   for i in range(0, len(texts), self.batch_size)]

        def worker(batch):
            last = None
            for attempt in range(self.max_retries + 1):
                try:
                    with self._sem:
                        return self.embed_fn(batch)
                except Exception as e:  # 瞬时错误：退避后重试
                    last = e
                    if attempt < self.max_retries:
                        time.sleep(self.retry_backoff * (2 ** attempt))
            raise RuntimeError(
                f"embed batch 失败（重试 {self.max_retries + 1} 次）：{last}"
            )

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
            results = list(ex.map(worker, batches))

        # 按批次顺序展平，保证向量与原文一一对应
        flat = [vec for batch in results for vec in batch]
        return flat


def make_ollama_embedder(base_url: str = None, model: str = None):
    """构造走 Ollama HTTP 的 embedding 函数（与项目 _make_embedder 一致）。"""
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError:
        from langchain_community.embeddings import OllamaEmbeddings
    base = base_url or os.getenv("OLLAMA_BASE_URL", "http://192.168.200.128:11434")
    mdl = model or os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
    return OllamaEmbeddings(model=mdl, base_url=base)
