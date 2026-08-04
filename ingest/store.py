"""向量库写入后端（改造点 1 的解耦关键）。

把「往哪写 / 怎么删」抽象成 StoreBackend，使 ingestion 管线与具体向量库解耦：
- MemoryStoreBackend：内存实现，供单元测试零依赖验证管线逻辑；
- MilvusStoreBackend：生产实现，复用项目现有 Milvus 客户端与 schema
  （chunk_id 主键幂等 upsert + batch=200 + flush + 按 file_path 删除）。
"""

from typing import List, Dict, Any, Optional


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
            return int(res.get("delete_cnt", 0))
        return 0

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
