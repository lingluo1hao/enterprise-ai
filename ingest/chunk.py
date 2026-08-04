"""结构感知切分（百万级 RAG 数据面改造点：分块升级）。

相对旧版（固定 chunk_size=600/overlap=120 的朴素 RecursiveCharacterTextSplitter），
本模块提供 `StructureAwareChunker`，解决三大痛点：

1. **结构感知（按标题/章节边界）**
   - Markdown：用 `MarkdownHeaderTextSplitter` 按 `#`~`####` 切分章节，标题路径
     作为 `section_path` 前置到正文，保证「同一章节内聚、跨章节不混」；
   - HTML：用 `HTMLHeaderTextSplitter` 按 `h1`~`h6` 切分；
   - 其余格式（pdf/txt/docx/xlsx/pptx）：整体作为单 section（降级路径）。

2. **代码块 / 表格不切断**
   - 先用正则把 ```fenced code``` 与 Markdown 表格整段抽成「原子片段」，
     不进入递归切分，避免把一行代码或一个单元格从中间劈开；
   - 原子片段超过阈值（默认 2000 字）才按空行安全切，仍不切断单行。

3. **父子文档（small-to-big / parent-child）**
   - 子片段：小窗口（默认 400 字，精确匹配）；
   - 父窗口：每个 section 的完整文本，作为 `parent_content` 随子片段一起落库；
   - 检索时命中子片段，但透传 `parent_content` 给 LLM，兼顾「召回准」与「上下文足」。

`chunk_documents` 保留为 legacy 接口（固定 600/120），供 `structure_aware=False`
时回退，确保行为可对比、可降级。
"""

import os
import re
import hashlib
from typing import List, Optional

from .types import RawDoc, Chunk

# 默认分隔符（与主模块 SEPARATORS 一致）；末尾 "" 为字符级兜底，
# 保证无标点长文（如代码块）也能被切碎，而非整段作为一个 chunk。
DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "；", "？", "！", "，", "、", ""]

# 子片段（被 embedding、被检索）窗口
DEFAULT_CHILD_SIZE = 400
DEFAULT_CHILD_OVERLAP = 80
# 父窗口（透传给 LLM 的上下文）窗口
DEFAULT_PARENT_SIZE = 1200
DEFAULT_PARENT_OVERLAP = 150
# 原子片段（代码/表格）超过该长度才按空行安全切
MAX_ATOMIC = 2000

_CODE_RE = re.compile(r"```.*?```", re.DOTALL)

# --------------------------------------------------------------------------- #
# 图区邻近合并：把「图标题 + 邻近上下文」合并成独立图块，提升图表类内容的召回
# --------------------------------------------------------------------------- #
# 图标题锚点：流程图 / 架构图 / 图1 / Figure 2 等，命中即视为「此处有图」
# 「通信流程」「网络架构」等组合关键词单独列出，避免依赖「图」字尾
FIGURE_ANCHOR_RE = re.compile(
    r"(图\s*\d+|图\s*[一二三四五六七八九十]+|Figure\s*\d+|"
    r"流程图|示意图|架构图|拓扑图|时序图|结构图|关系图|部署图|网络图|框图|图表|"
    r"通信流程|网络架构|系统架构|逻辑结构|整体架构|功能结构|模块结构|物理拓扑)",
    re.IGNORECASE,
)
# 图块上下文窗口：锚点前后各取若干字符，覆盖图标题与图内/图旁文字
FIGURE_BEFORE = 400
FIGURE_AFTER = 1000
# 整页文本若短于该值（典型图表页），直接整页作为图块，避免漏掉图内零散文字
# 提高到 3000：很多 PDF 真图页 1500-3000 字（说明+图注+周围段落）
FIGURE_PAGE_MAX = 3000


def _extract_figure_blocks(text: str) -> List[str]:
    """扫描文本抽出图块列表。

    - 整页较短（≤ FIGURE_PAGE_MAX）：整页即图块（图表页文字少，整页合并最稳，
      保证图内矢量文字/标注（如「设备登录」「控制指令」）不遗漏）；
    - 否则：每个图标题锚点取前后窗口合并成一块。
    返回空列表表示无图，调用方走原逻辑。
    """
    if not FIGURE_ANCHOR_RE.search(text):
        return []
    if len(text) <= FIGURE_PAGE_MAX:
        return [text]
    blocks = []
    for m in FIGURE_ANCHOR_RE.finditer(text):
        start = max(0, m.start() - FIGURE_BEFORE)
        end = min(len(text), m.end() + FIGURE_AFTER)
        blk = text[start:end].strip()
        if blk:
            blocks.append(blk)
    return blocks


def _is_separator_row(s: str) -> bool:
    """判断一行是否为 Markdown 表格分隔行（| --- | --- |）。"""
    s = s.strip()
    if "|" not in s or "-" not in s:
        return False
    return bool(re.match(r"^\s*\|?[\s:\-|]+\|?\s*$", s))


# --------------------------------------------------------------------------- #
# legacy：固定窗口切分（保留，用于回退与对照）
# --------------------------------------------------------------------------- #
def chunk_documents(raw_docs: List, chunk_size: int = 600,
                    chunk_overlap: int = 120,
                    separators: List[str] = None) -> List[Chunk]:
    """把 RawDoc 列表切成 Chunk 列表（朴素固定窗口，无结构感知）。"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if separators is None:
        separators = DEFAULT_SEPARATORS

    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        strip_whitespace=True,
    )
    out = []
    for doc in raw_docs:
        if not doc.text or not doc.text.strip():
            continue
        parts = splitter.split_text(doc.text)
        for i, p in enumerate(parts):
            out.append(Chunk(
                text=p,
                source=doc.source,
                file_name=doc.file_name,
                access_level=doc.access_level,
                chunk_index=i,
            ))
    return out


# --------------------------------------------------------------------------- #
# 结构感知：把文本切成 (kind, text) 片段，保护代码块/表格不被切断
# --------------------------------------------------------------------------- #
def _segment_tables(text: str) -> List[tuple]:
    """在普通文本中识别 Markdown 表格，返回 prose/table 交替片段。"""
    lines = text.split("\n")
    segs: List[tuple] = []
    i, n = 0, len(lines)
    buf_start = 0
    while i < n:
        if "|" in lines[i] and i + 1 < n and _is_separator_row(lines[i + 1]):
            pre = "\n".join(lines[buf_start:i])
            if pre.strip():
                segs.append(("prose", pre))
            j = i
            while j < n and "|" in lines[j]:
                j += 1
            segs.append(("table", "\n".join(lines[i:j])))
            i = j
            buf_start = j
        else:
            i += 1
    tail = "\n".join(lines[buf_start:])
    if tail.strip():
        segs.append(("prose", tail))
    return segs


def _segment(text: str) -> List[tuple]:
    """先用正则抽出 fenced code 块，其余再按表格切分。返回 (kind, text) 列表。"""
    segs: List[tuple] = []
    pos = 0
    for m in _CODE_RE.finditer(text):
        if m.start() > pos:
            segs.extend(_segment_tables(text[pos:m.start()]))
        segs.append(("code", m.group(0)))
        pos = m.end()
    if pos < len(text):
        segs.extend(_segment_tables(text[pos:]))
    return segs


# --------------------------------------------------------------------------- #
# 结构感知切分器
# --------------------------------------------------------------------------- #
class StructureAwareChunker:
    """Markdown/HTML 层级感知 + 代码块/表格保护 + 父子文档。

    用法：
        chunker = StructureAwareChunker()
        chunks = chunker.split(raw_doc)   # raw_doc: ingest.types.RawDoc
    每个返回的 Chunk 都带 parent_id / parent_content（父窗口文本），
    便于下游「小片段检索 + 父窗口透传」。
    """

    def __init__(self, child_size: int = DEFAULT_CHILD_SIZE,
                 child_overlap: int = DEFAULT_CHILD_OVERLAP,
                 parent_size: int = DEFAULT_PARENT_SIZE,
                 parent_overlap: int = DEFAULT_PARENT_OVERLAP,
                 separators: Optional[List[str]] = None,
                 max_atomic: int = MAX_ATOMIC):
        self.child_size = child_size
        self.child_overlap = child_overlap
        self.parent_size = parent_size
        self.parent_overlap = parent_overlap
        self.separators = separators or DEFAULT_SEPARATORS
        self.max_atomic = max_atomic
        self._splitter = None  # 惰性构建（避免无 langchain 时导入失败）

    def _rec_split(self, text: str) -> List[str]:
        if self._splitter is None:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self._splitter = RecursiveCharacterTextSplitter(
                separators=self.separators,
                chunk_size=self.child_size,
                chunk_overlap=self.child_overlap,
                length_function=len,
                strip_whitespace=True,
            )
        return self._splitter.split_text(text)

    # -- 格式分发 --------------------------------------------------------- #
    def split(self, raw: RawDoc) -> List[Chunk]:
        ext = os.path.splitext(raw.source)[1].lower()
        if ext in (".md", ".markdown"):
            return self._split_markdown(raw)
        if ext in (".html", ".htm"):
            return self._split_html(raw)
        # PDF 及其它无显式结构的格式走「图感知」分块：
        # 先抽图块（图级召回），再按 section 切 child（细节召回）
        return self._split_figure_aware(raw)

    # -- Markdown：按标题切章节 ------------------------------------------ #
    def _split_markdown(self, raw: RawDoc) -> List[Chunk]:
        try:
            from langchain_text_splitters import MarkdownHeaderTextSplitter
            splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[("#", "h1"), ("##", "h2"),
                                     ("###", "h3"), ("####", "h4")])
            docs = splitter.split_text(raw.text)
        except Exception:
            return self._split_section(raw.text, [], raw)
        out: List[Chunk] = []
        for d in docs:
            path = [str(d.metadata[k]) for k in ("h1", "h2", "h3", "h4")
                    if d.metadata.get(k)]
            content = d.page_content
            if path:
                content = " > ".join(path) + "\n\n" + content
            out.extend(self._split_section(content, path, raw))
        return out

    # -- HTML：按 h1~h6 切章节 ------------------------------------------- #
    def _split_html(self, raw: RawDoc) -> List[Chunk]:
        try:
            from langchain_text_splitters import HTMLHeaderTextSplitter
            splitter = HTMLHeaderTextSplitter(
                headers_to_split_on=[("h1", "h1"), ("h2", "h2"),
                                     ("h3", "h3"), ("h4", "h4"),
                                     ("h5", "h5"), ("h6", "h6")])
            docs = splitter.split_text(raw.text)
        except Exception:
            return self._split_section(raw.text, [], raw)
        out: List[Chunk] = []
        for d in docs:
            path = [str(v) for k, v in sorted(d.metadata.items())
                    if str(k).lower().startswith("h") and v]
            content = d.page_content
            if path:
                content = " > ".join(path) + "\n\n" + content
            out.extend(self._split_section(content, path, raw))
        return out

    # -- 单 section：保护代码/表格 + 父子链接 --------------------------- #
    def _split_section(self, content: str, path: List[str],
                       raw: RawDoc) -> List[Chunk]:
        pid = hashlib.md5(
            ("§".join(path) + "\n" + content).encode("utf-8")).hexdigest()[:16]
        pcontent = content  # 父窗口 = 整段 section 文本（含标题路径前缀）
        segs = _segment(content)
        children: List[Chunk] = []
        idx = 0
        for kind, seg_text in segs:
            if kind == "prose":
                pieces = self._rec_split(seg_text)
            elif len(seg_text) > self.max_atomic:
                # 仅超长代码/表格才按空行安全切，仍不切断单行
                pieces = [p for p in seg_text.split("\n\n") if p.strip()]
            else:
                pieces = [seg_text]
            for p in pieces:
                if not p.strip():
                    continue
                children.append(Chunk(
                    text=p,
                    source=raw.source,
                    file_name=raw.file_name,
                    access_level=raw.access_level,
                    chunk_index=idx,
                    parent_id=pid,
                    parent_content=pcontent,
                    is_parent=False,
                    section_path=path or None,
                    chunk_type=kind,
                    page=raw.page,
                    figure_paths=list(raw.figure_paths),
                ))
                idx += 1
        return children

    # -- 图感知：先抽图块，再切 child ----------------------------------- #
    def _split_figure_aware(self, raw: RawDoc) -> List[Chunk]:
        """PDF 等无结构文本：抽出图块（图级召回）后，再切 child（细节召回）。

        入库三类 chunk（PDF 特有）：
        - figure_block：图标题 + 邻近上下文（或整页），负责「通信流程图」这类图查询；
          仅在锚点命中时存在（PyPDF text 流不含真图 caption 时可能为 0）。
        - page：每页整页作为兜底索引，PyPDF 抽不出图 caption 的图也能通过页级
          embedding 召回，LLM 可看到图周围段落文字作为上下文。
        - child：常规 section 细切，负责协议字段、报文格式等细节查询。
        """
        out: List[Chunk] = []
        fig_blocks = _extract_figure_blocks(raw.text)
        base = hashlib.md5(
            (raw.source + "\n" + raw.text[:64]).encode("utf-8")).hexdigest()[:12]
        for i, fb in enumerate(fig_blocks):
            pid = f"{base}-fig{i}"
            out.append(Chunk(
                text=fb,
                source=raw.source,
                file_name=raw.file_name,
                access_level=raw.access_level,
                chunk_index=0,
                parent_id=pid,
                parent_content=fb,
                is_parent=True,
                section_path=None,
                chunk_type="figure_block",
                page=raw.page,
                figure_paths=list(raw.figure_paths),
            ))
        # 整页兜底 chunk（page chunk）：PyPDF text 流不含图 caption 时，图查询召回
        # 仍能命中该页 page chunk，LLM 可结合上下文推断该页是否有图。
        # 与 child 平级（is_parent=False），避免打乱原 parent-child 关系。
        if raw.page is not None and raw.text and raw.text.strip():
            out.append(Chunk(
                text=raw.text,
                source=raw.source,
                file_name=raw.file_name,
                access_level=raw.access_level,
                chunk_index=0,
                parent_id="",
                parent_content="",
                is_parent=False,
                section_path=None,
                chunk_type="page",
                page=raw.page,
                figure_paths=list(raw.figure_paths),
            ))
        # 常规 child 切分（细节召回）；page 随每块透传
        out.extend(self._split_section(raw.text, [], raw))
        return out


# --------------------------------------------------------------------------- #
# legacy 包装（供 structure_aware=False 时回退）
# --------------------------------------------------------------------------- #
class LegacyChunker:
    def __init__(self, chunk_size=600, chunk_overlap=120, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def split(self, raw: RawDoc) -> List[Chunk]:
        return chunk_documents([raw], self.chunk_size,
                               self.chunk_overlap, self.separators)


def make_chunker(structure_aware: bool = True,
                 child_size: int = DEFAULT_CHILD_SIZE,
                 child_overlap: int = DEFAULT_CHILD_OVERLAP,
                 parent_size: int = DEFAULT_PARENT_SIZE,
                 parent_overlap: int = DEFAULT_PARENT_OVERLAP,
                 chunk_size: int = 600, chunk_overlap: int = 120,
                 separators: Optional[List[str]] = None):
    """工厂：结构感知 or legacy 朴素切分。"""
    if structure_aware:
        return StructureAwareChunker(child_size, child_overlap,
                                     parent_size, parent_overlap, separators)
    return LegacyChunker(chunk_size, chunk_overlap, separators)
