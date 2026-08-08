"""增量 ingestion 管线（改造点 1+2+3+4 的总装）。

流程：扫 knowledge/ → 指纹增量(mtime+size+md5) → 多格式 loader → 结构切分 →
      批量 embedding(并发+重试) → 幂等 upsert → 更新清单。

特点：
- 可重复触发、仅处理变更文件（增量），全量用 force=True；
- 带进度回调与统计报告（RunReport）；
- dry_run 只计算「会做什么」不落库，便于上线前预检；
- 文件被删时自动从向量库删除对应实体并清理清单；
- 支持单文件 delete / rebuild。
"""

import os
import time
import hashlib
from typing import List, Dict, Any, Callable, Optional

from .types import RawDoc, Chunk, RunReport
from .fingerprint import compute_fingerprint, ManifestStore, diff_fingerprints
from .loaders import load_file, get_access_level
from .chunk import make_chunker
from .embed import BatchEmbedder
from .store import StoreBackend, _trunc_bytes

# 支持的文件后缀（与 loaders.load_file 对齐）
SUPPORTED_EXT = (".txt", ".md", ".pdf", ".html", ".htm",
                 ".docx", ".xlsx", ".xls", ".pptx")


class IngestPipeline:
    def __init__(self, folder: str,
                 embedder: Callable[[List[str]], List[List[float]]],
                 store: StoreBackend,
                 manifest_path: Optional[str] = None,
                 tenant_id: str = "default",
                 user_id: str = "anonymous",
                 access_fn: Callable[[str], str] = None,
                 structure_aware: bool = True,
                 child_size: int = 400, child_overlap: int = 80,
                 parent_size: int = 1200, parent_overlap: int = 150,
                 chunk_size: int = 600, chunk_overlap: int = 120,
                 separators: Optional[List[str]] = None,
                 batch_size: int = 64, max_concurrency: int = 4,
                 max_retries: int = 3, dry_run: bool = False):
        self.folder = folder
        self.embedder_callable = embedder
        self.store = store
        self.manifest_path = manifest_path or os.path.join(
            folder, ".ingest_manifest.sqlite")
        self.access_fn = access_fn or get_access_level
        self.structure_aware = structure_aware
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self._chunker = make_chunker(
            structure_aware=structure_aware,
            child_size=child_size, child_overlap=child_overlap,
            parent_size=parent_size, parent_overlap=parent_overlap,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=separators)
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.dry_run = dry_run
        self.tenant_id = tenant_id      # 默认租户（供全量构建；上传时由调用方覆盖）
        self.user_id = user_id          # 文档拥有者（供全量构建=anonymous；上传=真实上传者）
        self._manifest = ManifestStore(self.manifest_path)
        self._embedder = BatchEmbedder(
            embedder, batch_size=batch_size,
            max_concurrency=max_concurrency, max_retries=max_retries)
        self._progress_cb = None

    # ------------------------------------------------------------------ #
    # 文件清单 / 指纹
    # ------------------------------------------------------------------ #
    def _list_files(self, files: Optional[List[str]] = None) -> List[str]:
        if files is not None:
            return [f for f in files if os.path.isfile(f)]
        out = []
        if not os.path.isdir(self.folder):
            return out
        for root, _dirs, names in os.walk(self.folder):
            # 跳过隐藏目录（如 .git）
            if os.path.basename(root).startswith("."):
                continue
            for name in sorted(names):
                if name.startswith("."):
                    continue
                if os.path.splitext(name)[1].lower() in SUPPORTED_EXT:
                    out.append(os.path.join(root, name))
        return out

    def scan_and_diff(self, files: Optional[List[str]] = None, force: bool = False) \
            -> Dict[str, Any]:
        paths = self._list_files(files)
        current = {p: compute_fingerprint(p) for p in paths}
        manifest = self._manifest.load_all()
        added, updated, unchanged, removed = diff_fingerprints(current, manifest)
        if files is not None:
            # 单文件/指定文件上传：不推断磁盘删除，避免误删其他已入库文件
            removed = []
        if force:
            # 全量：所有在盘文件都(重)处理；新增/更新仅用于报告展示
            to_process = list(current.keys())
            added = [p for p in to_process if p not in manifest]
            updated = [p for p in to_process if p in manifest]
            unchanged = []
        else:
            to_process = added + updated
        return {
            "current": current, "manifest": manifest,
            "added": added, "updated": updated,
            "unchanged": unchanged, "removed": removed,
            "to_process": to_process, "all_paths": paths,
        }

    def _derive_tenant(self, path: str) -> str:
        """由文件相对 folder 的路径推断租户：首层子目录即租户名（knowledge/{tenant}/x.pdf）；

        平铺文件（knowledge/x.pdf）归 default。
        """
        try:
            rel = os.path.relpath(path, self.folder)
        except ValueError:
            return "default"
        parts = rel.split(os.sep)
        if len(parts) > 1 and parts[0]:
            return parts[0]
        return "default"

    # ------------------------------------------------------------------ #
    # 主运行
    # ------------------------------------------------------------------ #
    def run(self, force: bool = False, files: Optional[List[str]] = None,
            progress_cb: Optional[Callable[[str], None]] = None) -> RunReport:
        if progress_cb:
            self._progress_cb = progress_cb
        t0 = time.time()
        rep = RunReport(dry_run=self.dry_run)
        diff = self.scan_and_diff(files=files, force=force)
        rep.files_scanned = len(diff["all_paths"])
        rep.files_added = len(diff["added"])
        rep.files_updated = len(diff["updated"])
        rep.files_unchanged = len(diff["unchanged"])
        rep.files_removed = len(diff["removed"])

        if self._progress_cb:
            self._progress_cb(
                f"待处理 {len(diff['to_process'])} 个文件，待删除 {len(diff['removed'])} 个")

        all_chunks: List[Chunk] = []
        file_chunk_counts: Dict[str, int] = {}
        for fp in diff["to_process"]:
            raw = load_file(fp)
            for d in raw:
                d.access_level = self.access_fn(d.source)
            chunks = []
            for d in raw:
                chunks.extend(self._chunker.split(d))
            file_chunk_counts[fp] = len(chunks)
            all_chunks.extend(chunks)
            if self._progress_cb:
                self._progress_cb(f"  {os.path.basename(fp)}: {len(chunks)} 分片")

        rep.chunks_total = len(all_chunks)

        entities: List[Dict[str, Any]] = []
        if all_chunks:
            texts = [c.text for c in all_chunks]
            vectors = self._embedder.embed(texts)
            for c, vec in zip(all_chunks, vectors):
                # 路径分隔符归一化：Windows 上 / 与 \ 混用会让同一文档算出不同 chunk_id，
                # 导致 upsert 不幂等（残留叠加）、delete 按一种形式删不净、父子 id 跨分隔符失配。
                # 统一为 / 后：chunk_id 稳定、删除可精确匹配、tenant 推断一致。
                norm_source = c.source.replace("\\", "/")
                # chunk_id = md5(content + source)：内容相同 → 同一 id → 幂等 upsert
                cid = hashlib.md5((c.text + norm_source).encode("utf-8")).hexdigest()
                # 租户从文件相对 folder 的路径推断：首层子目录即租户名（knowledge/{tenant}/x.pdf），
                # 平铺文件（knowledge/x.pdf）归 default。拥有者一律取 pipeline 构造时传入的 user_id。
                tenant = self._derive_tenant(norm_source)
                entities.append({
                    "chunk_id": cid,
                    "content": _trunc_bytes(c.text),
                    "dense": vec,
                    "file_path": norm_source,
                    "file_name": c.file_name,
                    "access_level": c.access_level,
                    "chunk_index": c.chunk_index,
                    "user_id": self.user_id,
                    "tenant_id": tenant,
                    "parent_id": c.parent_id or "",
                    "parent_content": _trunc_bytes(c.parent_content or c.text),
                    "is_parent": c.is_parent,
                    "page": c.page,
                    "chunk_type": c.chunk_type,
                    "section_path": "§".join(c.section_path) if c.section_path else "",
                    "figure_paths": list(c.figure_paths),  # 动态字段，存为 list
                })
        rep.entities_upserted = len(entities)

        if self.dry_run:
            rep.entities_upserted = 0  # 未真正落库，仅预检
            rep.duration_sec = time.time() - t0
            if self._progress_cb:
                self._progress_cb("[DRY-RUN] 未写入向量库/清单")
            return rep

        if entities:
            self.store.upsert(entities)
        deleted = 0
        for fp in diff["removed"]:
            deleted += self.store.delete_by_file(fp)
            self._manifest.remove(fp)
        rep.entities_deleted = deleted

        # 更新清单中已处理文件的指纹
        for fp in diff["to_process"]:
            self._manifest.upsert(fp, diff["current"][fp],
                                  file_chunk_counts.get(fp, 0))

        # 方案乙：文档真正变更后，bump 知识库版本号 → 所有旧精确缓存 key 瞬间失效。
        # 覆盖 web 全量重建(_kb_build_pipeline().run) 与 CLI ingest(vector_store.ingest_documents) 两条路径。
        if rep.entities_upserted + rep.entities_deleted > 0:
            try:
                from kb_version import bump_kb_version
                bump_kb_version()
            except Exception as e:
                print(f"[pipeline] ⚠ kb_version bump 失败(忽略): {e}")

        rep.duration_sec = time.time() - t0
        return rep

    # ------------------------------------------------------------------ #
    # 运维命令
    # ------------------------------------------------------------------ #
    def status(self) -> None:
        m = self._manifest.load_all()
        print(f"[ingest] 清单: {self.manifest_path}")
        print(f"[ingest] 已追踪文件数: {len(m)}")
        for p, fp in sorted(m.items()):
            print(f"  - {os.path.basename(p)} "
                  f"(分片={fp.get('chunk_count')}, md5={fp.get('md5')[:8]})")

    def delete(self, file_path: str) -> int:
        """从向量库与清单中删除单个文件（如误传/过期文档）。"""
        n = self.store.delete_by_file(file_path)
        self._manifest.remove(file_path)
        return n

    def rebuild(self, file_path: str) -> RunReport:
        """删除单个文件后强制重新 ingest（如文档内容大改需全量重切）。"""
        self.store.delete_by_file(file_path)
        self._manifest.remove(file_path)
        return self.run(force=True, files=[file_path])

    def close(self) -> None:
        self._manifest.close()
