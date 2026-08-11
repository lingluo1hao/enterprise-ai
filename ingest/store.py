"""向量库写入后端（改造点 1 的解耦关键）。

把「往哪写 / 怎么删」抽象成 StoreBackend，使 ingestion 管线与具体向量库解耦：
- MemoryStoreBackend：内存实现，供单元测试零依赖验证管线逻辑；
- MilvusStoreBackend：生产实现，复用项目现有 Milvus 客户端与 schema
  （chunk_id 主键幂等 upsert + batch=200 + flush + 按 file_path 删除）。
"""

import os
import shutil
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("ingest.store")


def _delete_cnt(res) -> int:
    """从 pymilvus 各版本 delete 返回值里稳定取出删除条数。"""
    if isinstance(res, dict):
        return int(res.get("delete_cnt", 0))
    return 0


def _norm_path_key(p: str) -> str:
    """把任意形态的路径归一化为「knowledge/子目录/文件名」稳定 key。

    覆盖三种历史形态：① 带 ./ 前缀；② 反斜杠分隔符（旧 Windows 直接写入）；
    ③ 绝对路径（含盘符 / 项目根）。归一化后只保留 knowledge/ 之后的相对部分，
    使「同一文档」不论以哪种写法传入都能精确对应，互不误伤不同子目录的同名文件。
    """
    p = (p or "").replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    idx = p.find("knowledge/")
    if idx >= 0:
        p = p[idx:]
    return p.strip("/")


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
        # 归一化路径分隔符：pipeline 写入时把 \\ 替成 /（见 pipeline.run），但调用方传的可能仍是
        # Windows 风格带 \\ 的路径；同时 Milvus filter 里 \\ 又是转义，先归一再转义，否则回退到
        # 「删除实体 0、duplicate 叠加」静默失败的连锁。
        norm = file_path.replace("\\", "/").replace('"', '\\"')
        expr = f'file_path == "{norm}"'
        res = self.client.delete(self.collection, filter=expr)
        n = _delete_cnt(res)
        # ---- 兜底：历史实体可能带 ./ 前缀或反斜杠（旧 Windows 直接写入），
        # 精确匹配必失 → 静默删除 0 条，旧实体沦为删不掉的孤儿。改用 Python 侧
        # 归一化匹配：拉取全量 (chunk_id, file_path)，把存储值与入参都归一成
        # 「knowledge/子目录/文件名」稳定 key 后精确比对，再按主键批量删除。
        # 选 Python 侧而非 Milvus LIKE：LIKE 转义对含 ( ) / _ 的中文文件名解析不稳
        # （实测 `file_path like "%Jimi IoT\_...V1.2(1).pdf"` 直接报 1100 表达式解析失败），
        # 而 Python 侧匹配精确可控、不依赖 Milvus 表达式解析器。
        if n == 0:
            ids = self._match_ids_by_path(file_path)
            if ids:
                self.client.delete(self.collection, ids=ids)
                logger.warning(
                    "delete_by_file 精确匹配落空，已用 Python 侧归一化兜底删除 %d 条"
                    "（历史 ./ 前缀或反斜杠路径实体）",
                    len(ids),
                )
                n = len(ids)
        # 清理该文档切片生成的所有图片（孤儿图回收）：table_p*.png / fig_p*.png / 整页图
        try:
            clean_figures_for_file(file_path)
        except Exception:
            pass
        return n

    def _match_ids_by_path(self, file_path: str) -> List[str]:
        """按归一化路径 key 找出所有匹配的 chunk_id（主键）。

        分页拉取全量 (chunk_id, file_path) 在 Python 侧比对，避免一次性大查询与
        Milvus 表达式转义坑。命中即返回主键列表（无匹配返回空列表，删除 0 条）。
        """
        target = _norm_path_key(file_path)
        if not target:
            return []
        matched: List[str] = []
        offset = 0
        page = 5000
        while True:
            rows = self.client.query(
                self.collection,
                filter="",
                output_fields=["chunk_id", "file_path"],
                limit=page,
                offset=offset,
            )
            if not rows:
                break
            for r in rows:
                fp = r.get("file_path", "")
                if _norm_path_key(fp) == target:
                    matched.append(r["chunk_id"])
            if len(rows) < page:
                break
            offset += page
        return matched

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
