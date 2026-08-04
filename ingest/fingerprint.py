"""文件指纹 + 清单持久化（增量 ingestion 的核心）。

指纹 = mtime + size + md5(content)。
- mtime/size 廉价，先快速排除绝大多数未变文件；
- md5 兜底：同大小/同 mtime 但内容被改（如编辑器原地保存）时仍能识别为"已变更"。

清单用 sqlite 落地在 knowledge/.ingest_manifest.sqlite，保证进程重启后增量状态不丢、
可断点续跑。
"""

import os
import time
import hashlib
import sqlite3
from typing import Dict, Tuple


def compute_fingerprint(path: str) -> dict:
    """计算单个文件的指纹。"""
    st = os.stat(path)
    mtime = st.st_mtime
    size = st.st_size
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return {"mtime": mtime, "size": size, "md5": h.hexdigest()}


class ManifestStore:
    """清单存储（sqlite）。记录每个已 ingest 文件的指纹与分片数。"""

    def __init__(self, path: str):
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS fingerprints ("
            "file_path TEXT PRIMARY KEY, mtime REAL, size INTEGER, md5 TEXT, "
            "chunk_count INTEGER, updated_at REAL)"
        )
        self._conn.commit()

    def load_all(self) -> Dict[str, dict]:
        rows = self._conn.execute(
            "SELECT file_path, mtime, size, md5, chunk_count FROM fingerprints"
        ).fetchall()
        return {
            r[0]: {"mtime": r[1], "size": r[2], "md5": r[3], "chunk_count": r[4]}
            for r in rows
        }

    def upsert(self, file_path: str, fp: dict, chunk_count: int):
        self._conn.execute(
            "INSERT INTO fingerprints(file_path, mtime, size, md5, chunk_count, updated_at) "
            "VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(file_path) DO UPDATE SET "
            "mtime=excluded.mtime, size=excluded.size, md5=excluded.md5, "
            "chunk_count=excluded.chunk_count, updated_at=excluded.updated_at",
            (file_path, fp["mtime"], fp["size"], fp["md5"], chunk_count, time.time()),
        )
        self._conn.commit()

    def remove(self, file_path: str):
        self._conn.execute("DELETE FROM fingerprints WHERE file_path=?", (file_path,))
        self._conn.commit()

    def close(self):
        self._conn.close()


def diff_fingerprints(current: Dict[str, dict], manifest: Dict[str, dict]) \
        -> Tuple[list, list, list, list]:
    """对比当前磁盘指纹与已记录清单，返回 (added, updated, unchanged, removed)。

    - added：磁盘有、清单无
    - updated：磁盘有、清单有，但 mtime/size/md5 任一不同
    - unchanged：完全一致
    - removed：清单有、磁盘无（文件被删）
    """
    added, updated, unchanged, removed = [], [], [], []
    for path, fp in current.items():
        if path not in manifest:
            added.append(path)
        elif (manifest[path].get("mtime") != fp["mtime"]
              or manifest[path].get("size") != fp["size"]
              or manifest[path].get("md5") != fp["md5"]):
            updated.append(path)
        else:
            unchanged.append(path)
    for path in manifest:
        if path not in current:
            removed.append(path)
    return added, updated, unchanged, removed
