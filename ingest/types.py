"""数据面引擎共享数据类型（零依赖）。

这些类型刻意不依赖 langchain，便于在无 langchain 的环境下也能跑单元测试。
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RawDoc:
    """loader 产出的单篇原始文档（归一化后的统一结构）。"""
    text: str
    source: str
    file_name: str
    access_level: str = "public"
    page: Optional[int] = None  # PDF 等分页格式的来源页码（1-based，人类可读）
    figure_paths: List[str] = field(default_factory=list)
    # PDF 页面渲染图的文件路径（相对项目根），前端可直接 <img src> 引用。
    # PDF 真图裁剪图的文件路径（相对项目根，PyMuPDF 连通分量抽取），前端可直接 <img src> 引用。
    # 缺失 pymupdf/numpy/scipy 或非 PDF 来源时为 []。


@dataclass
class Chunk:
    """切分后的片段。

    结构感知 + 父子文档模式下新增字段：
    - section_path：标题/章节层级路径（如 ["第一章","安装"]），用于层级感知；
    - parent_id / parent_content：父子链接，子片段携带父窗口文本；
    - is_parent：是否为父窗口（本实现仅存子片段，父上下文随子片段透传）；
    - chunk_type：prose / code / table，标记是否被结构保护（代码块/表格不切断）。
    """
    text: str
    source: str
    file_name: str
    access_level: str
    chunk_index: int
    parent_id: Optional[str] = None
    parent_content: Optional[str] = None
    is_parent: bool = False
    section_path: Optional[List[str]] = None
    chunk_type: str = "prose"
    page: Optional[int] = None  # 来源页码（1-based），便于回答时定位到具体页
    figure_paths: List[str] = field(default_factory=list)


@dataclass
class Entity:
    """写入向量库的一行实体（与 Milvus schema 字段一一对应）。

    父子文档模式下额外携带 parent_id / parent_content / is_parent，
    便于检索时「小片段精确匹配、父窗口上下文透传」（small-to-big）。
    """
    chunk_id: str
    content: str
    dense: List[float]
    file_path: str
    file_name: str
    access_level: str
    chunk_index: int
    user_id: str = "anonymous"
    parent_id: Optional[str] = None
    parent_content: Optional[str] = None
    is_parent: bool = False
    page: Optional[int] = None  # 来源页码（1-based）
    chunk_type: str = "prose"  # prose / code / table / figure_block / page
    figure_paths: List[str] = field(default_factory=list)


@dataclass
class RunReport:
    """一次 ingestion 运行的统计。"""
    files_scanned: int = 0
    files_added: int = 0
    files_updated: int = 0
    files_unchanged: int = 0
    files_removed: int = 0
    chunks_total: int = 0
    entities_upserted: int = 0
    entities_deleted: int = 0
    duration_sec: float = 0.0
    dry_run: bool = False

    def summary(self) -> str:
        return (
            f"扫描 {self.files_scanned} | 新增 {self.files_added} | 更新 {self.files_updated} | "
            f"不变 {self.files_unchanged} | 删除 {self.files_removed} | "
            f"分片 {self.chunks_total} | upsert {self.entities_upserted} | "
            f"删除实体 {self.entities_deleted} | 耗时 {self.duration_sec:.1f}s"
            + (" [DRY-RUN]" if self.dry_run else "")
        )
