"""向量库写入后端（改造点 1 的解耦关键）。

把「往哪写 / 怎么删」抽象成 StoreBackend，使 ingestion 管线与具体向量库解耦：
- MemoryStoreBackend：内存实现，供单元测试零依赖验证管线逻辑；
- MilvusStoreBackend：生产实现，复用项目现有 Milvus 客户端与 schema
  （chunk_id 主键幂等 upsert + batch=200 + flush + 按 file_path 删除）。
"""

import os
import shutil
from typing import List, Dict, Any, Optional


def _figures_root() -> str:
    """项目根下 `assets/figures` 的绝对路径（图片按 `assets/figures/{文件名stem}/` 组织）。"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "assets", "figures")


def clean_figures_for_file(file_path: str) -> int:
    """删除某文档切片产生的全部图片目录（assets/figures/{stem}/）。

    所有删除入口（Web API / CLI / 管线同步 / rebuild）最终都汇聚到
    MilvusStoreBackend.delete_by_file，因此在此集中清理，避免磁盘残留孤儿图片
    （table_p*.png、fig_p*.png、整页图等）。返回删除的文件数；目录不存在或路径
    非法时返回 0，且不抛出任何异常（删除图片是清理副作用，不应阻断文档删除）。
    """
    stem = os.path.splitext(os.path.basename(file_path))[0]
    if not stem or "/" in stem or "\\" in stem:
        return 0
    root = _figures_root()
    target = os.path.normpath(os.path.join(root, stem))
    # 二次校验：target 必须严格位于 figures root 之下，防止任何路径穿越
    if target == root or not target.startswith(root + os.sep):
        return 0
    if not os.path.isdir(target):
        return 0
    removed = 0
    for _r, _d, _files in os.walk(target):
        removed += len(_files)
    shutil.rmtree(target, ignore_errors=True)
    return removed


def _trunc_bytes(text: str, limit: int = 8192) -> str:
    """按 UTF-8 字节截断到 limit（Milvus VARCHAR max_length 实际按字节校验）。

    content/parent_content 的中文内容在字符级截断（[:8192]）下仍可能超过字节上限，
    必须按字节截断，否则 upsert 报 1100 (varchar length exceeds max length)。
    """
    if not isinstance(text, str):
        return text
    b = text.encode("utf-8")
    if len(b) <= limit:
        return text
    return b[:limit].decode("utf-8", "ignore")


class StoreBackend:
    def upsert(self, entities: List[Dict[str, Any]]) -> None:
        raise NotImplementedError

    def delete_by_file(self, file_path: str) -> int:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


class MemoryStoreBackend(StoreBackend):
    """内存后端（测试用）。chunk_id -> entity。"""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def upsert(self, entities: List[Dict[str, Any]]) -> None:
        for e in entities:
            self._store[e["chunk_id"]] = dict(e)

    def delete_by_file(self, file_path: str) -> int:
        n = 0
        for k in [k for k, e in self._store.items()
                  if e.get("file_path") == file_path]:
            del self._store[k]
            n += 1
        return n

    def count(self) -> int:
        return len(self._store)


class MilvusStoreBackend(StoreBackend):
    """Milvus 生产后端。

    实体字段与项目 Milvus schema 对齐：
    chunk_id(主键, 幂等) / content / dense / file_path / file_name /
    access_level / chunk_index / user_id
    """

    def __init__(self, client, collection: str, batch: int = 200):
        self.client = client
        self.collection = collection
        self.batch = batch

    def upsert(self, entities: List[Dict[str, Any]]) -> None:
        # 防御性字节截断：Milvus VARCHAR max_length 按 UTF-8 字节校验，
        # 中文内容字符级截断会失效，统一在此兜底，避免 upsert 报 1100。
        for e in entities:
            if isinstance(e.get("content"), str):
                e["content"] = _trunc_bytes(e["content"])
            if isinstance(e.get("parent_content"), str):
                e["parent_content"] = _trunc_bytes(e["parent_content"])
        for s in range(0, len(entities), self.batch):
            self.client.upsert(self.collection, entities[s:s + self.batch])
        self.client.flush(self.collection)

    def delete_by_file(self, file_path: str) -> int:
        # MilvusClient.delete 用 filter= 参数（非 expr=）；表达式里 \ 是转义符，
        # 路径里的反斜杠必须先转成 \\，否则 Windows 路径（docs\xxx.pdf）解析失败。
        safe = file_path.replace("\\", "\\\\").replace('"', '\\"')
        expr = f'file_path == "{safe}"'
        res = self.client.delete(self.collection, filter=expr)
        # 返回删除条数（pymilvus 新版返回 DeleteResult，含 delete_cnt）
        if isinstance(res, dict):
            n = int(res.get("delete_cnt", 0))
        else:
            n = 0
        # 清理该文档切片生成的所有图片（孤儿图回收）：table_p*.png / fig_p*.png / 整页图
        try:
            clean_figures_for_file(file_path)
        except Exception:
            pass
        return n

    def count(self) -> int:
        try:
            stats = self.client.get_collection_stats(self.collection)
            if isinstance(stats, dict):
                rc = stats.get("row_count", stats.get("num_entities"))
                if rc is not None:
                    return int(rc)
        except Exception:
            pass
        return 0
