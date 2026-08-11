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
   - 父窗口（P1 方案 a）：**不再存整章**，改存「子片段前后各 `PARENT_WINDOW_CHARS` 字滑动窗口」
     （`_window_around`），每个子片段自足、天然 < 8192 字节、不共享不截断不膨胀；
   - 检索时命中子片段，透传 `parent_content` 窗口给 LLM，兼顾「召回准」与「上下文足」。

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

# P1（方案 a）：父窗口改存「子片段前后各 N 字滑动窗口」而非整章。
# 每个子片段自足、天然 < 8192 字节、不共享不截断不膨胀；
# Plan A 尾部锚点（child[-80:]）必落在窗口内 → 命令集等被截断章节也能吃到 [章节续文]。
PARENT_WINDOW_CHARS = 550  # 前后各 550 字，窗口总长 ≈ 子片段 + 1100 字 ≪ 8192 字节上限


def _window_around(child: str, parent: str, n: int) -> str:
    """返回 child 在 parent 中的滑动窗口（前 n + child + 后 n 字）。

    child 是 parent 的连续子串（切分保证），用 parent.find(child) 定位。
    若找不到（极端规范化/子串被改写），返回空串——宁可不补也不拼整章，
    避免重新引入 8192 字节截断与同章膨胀两个老问题。
    """
    if not child or not child.strip() or not parent:
        return ""
    pos = parent.find(child)
    if pos < 0:
        return ""
    start = max(0, pos - n)
    end = min(len(parent), pos + len(child) + n)
    return parent[start:end]

_CODE_RE = re.compile(r"```.*?```", re.DOTALL)


# --------------------------------------------------------------------------- #
# 纯 Python 递归切分（替代 langchain_text_splitters）
# 原因：langchain_text_splitters 在部分 Python 环境下 import 即段错误，且段错误
# 无法被 try/except 捕获，会导致整个切片进程崩溃。以下实现行为对齐
# RecursiveCharacterTextSplitter（按分隔符递归、末级硬切、相邻块重叠）。
# --------------------------------------------------------------------------- #
def _hard_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """末级硬切（字符级）：按 chunk_size 切，相邻块重叠 chunk_overlap。"""
    if not text or not text.strip():
        return []
    if len(text) <= chunk_size:
        return [text]
    step = max(1, chunk_size - chunk_overlap)
    out: List[str] = []
    for i in range(0, len(text), step):
        piece = text[i:i + chunk_size]
        if piece.strip():
            out.append(piece)
    return out


def _py_recursive_split(text: str, separators: List[str],
                        chunk_size: int, chunk_overlap: int) -> List[str]:
    """纯 Python 递归字符切分：超长片段递归下一级分隔符，末级硬切。"""
    if not text or not text.strip():
        return []
    if len(text) <= chunk_size:
        return [text]
    for idx, sep in enumerate(separators):
        if sep and sep in text:
            parts = text.split(sep)
            pieces = [p + sep if i < len(parts) - 1 else p
                      for i, p in enumerate(parts)]
            # 合并相邻小段至接近 chunk_size
            merged: List[str] = []
            buf = ""
            for pc in pieces:
                if not buf:
                    buf = pc
                elif len(buf) + len(pc) <= chunk_size:
                    buf += pc
                else:
                    merged.append(buf)
                    buf = pc
            if buf:
                merged.append(buf)
            result: List[str] = []
            for m in merged:
                if len(m) <= chunk_size:
                    result.append(m)
                else:
                    deeper = separators[idx + 1:]
                    if deeper:
                        result.extend(_py_recursive_split(
                            m, deeper, chunk_size, chunk_overlap))
                    else:
                        result.extend(_hard_split(m, chunk_size, chunk_overlap))
            return result
    return _hard_split(text, chunk_size, chunk_overlap)

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

# 内嵌 [[FIG:...]] 占位符提取：loaders.py 把表格图路径直接写进 [TABLE] 块，
# chunker 扫描每个 chunk 文本，把真实出现的占位符同步到 figure_paths，
# 避免"一页所有 chunk 都带全页所有图"的错配问题。
_EMBEDDED_FIG_RE = re.compile(r"\[\[FIG:([^\]]+)\]\]")


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
    """把 RawDoc 列表切成 Chunk 列表（朴素固定窗口，无结构感知）。

    纯 Python 实现（不再依赖 langchain_text_splitters，避免其在部分环境下
    import 即段错误）。
    """
    if separators is None:
        separators = DEFAULT_SEPARATORS
    out = []
    for doc in raw_docs:
        if not doc.text or not doc.text.strip():
            continue
        parts = _py_recursive_split(doc.text, separators, chunk_size, chunk_overlap)
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
    """在普通文本中识别 Markdown 表格 / [TABLE] 原子块，返回 prose/table 交替片段。"""
    lines = text.split("\n")
    segs: List[tuple] = []
    i, n = 0, len(lines)
    buf_start = 0
    while i < n:
        # loaders.py 已经把「章节标题 + [[FIG:...]] + Markdown 表格」打包成 [TABLE] 原子块，
        # 直接整体作为 table 片段，避免标题/图占位符被拆到前面的 prose chunk。
        if lines[i].strip() == "[TABLE]":
            pre = "\n".join(lines[buf_start:i])
            if pre.strip():
                segs.append(("prose", pre))
            j = i + 1
            while j < n and lines[j].strip() != "[/TABLE]":
                j += 1
            # 包含 [TABLE] 与 [/TABLE] 标记
            end = min(j + 1, n)
            block = "\n".join(lines[i:end])
            if block.strip():
                segs.append(("table", block))
            i = end
            buf_start = end
        elif "|" in lines[i] and i + 1 < n and _is_separator_row(lines[i + 1]):
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
# 纯 Python Markdown / HTML 标题切分（替代 langchain_text_splitters 的
# MarkdownHeaderTextSplitter / HTMLHeaderTextSplitter——后者在部分 Python 环境下
# import 即段错误，且段错误无法被 try/except 捕获）。返回 [(content, path), ...]，
# path 为标题层级链（如 ["第一章", "安装"]）。
# --------------------------------------------------------------------------- #
_MD_HEADER_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$", re.MULTILINE)
_HTML_HEADER_RE = re.compile(r"<h([1-6])[^>]*>(.*?)</h\1>", re.IGNORECASE | re.DOTALL)


def _split_md_by_headers(text: str) -> List[tuple]:
    """按 #~#### 标题把 Markdown 切成 (content, path) 列表。"""
    lines = text.split("\n")
    headers = []
    for i, ln in enumerate(lines):
        m = _MD_HEADER_RE.match(ln)
        if m:
            headers.append((i, len(m.group(1)), m.group(2).strip()))
    if not headers:
        return [(text, [])]
    sections: List[tuple] = []
    stack: List[tuple] = []
    for hidx, (line_idx, level, title) in enumerate(headers):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        path = [t for _, t in stack]
        end = headers[hidx + 1][0] if hidx + 1 < len(headers) else len(lines)
        content = "\n".join(lines[line_idx:end]).strip()
        if content:
            sections.append((content, path))
    return sections


def _split_html_by_headers(text: str) -> List[tuple]:
    """按 <h1>~<h6> 把 HTML 切成 (content, path)，内容做基础标签剥离。"""
    headers = [(m.start(), int(m.group(1)),
                re.sub(r"<[^>]+>", "", m.group(2)).strip(), m.end())
               for m in _HTML_HEADER_RE.finditer(text)]
    if not headers:
        return [(text, [])]
    sections: List[tuple] = []
    stack: List[tuple] = []
    for hidx, (_, level, title, end) in enumerate(headers):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        path = [t for _, t in stack]
        nxt = headers[hidx + 1][0] if hidx + 1 < len(headers) else len(text)
        content = re.sub(r"<[^>]+>", " ", text[end:nxt])
        content = re.sub(r"\s+", " ", content).strip()
        if content:
            sections.append((content, path))
    return sections


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
        # 纯 Python 递归切分（不再依赖 langchain_text_splitters）：
        # 该包在部分 Python 环境下 import 即段错误，且段错误无法被 try/except 捕获，
        # 会导致整条切片链路崩溃。改用 _py_recursive_split 行为等价实现。
        return _py_recursive_split(text, self.separators, self.child_size, self.child_overlap)

    # -- 格式分发 --------------------------------------------------------- #
    def split(self, raw: RawDoc) -> List[Chunk]:
        # 已按章节聚合的 PDF 单元（loader 设置了 section_path）：
        # 走「整章为父、细切为子」的 small-to-big 父子切片
        if raw.section_path is not None:
            return self._split_prebuilt_section(raw)
        ext = os.path.splitext(raw.source)[1].lower()
        if ext in (".md", ".markdown"):
            return self._split_markdown(raw)
        if ext in (".html", ".htm"):
            return self._split_html(raw)
        # PDF 降级路径（PyPDFLoader 无结构）及其它无显式结构格式：图感知分块
        return self._split_figure_aware(raw)

    # -- Markdown：按标题切章节 ------------------------------------------ #
    def _split_markdown(self, raw: RawDoc) -> List[Chunk]:
        # 纯 Python 按 #~#### 标题切分（不再依赖 langchain 的
        # MarkdownHeaderTextSplitter，避免 import 段错误）
        out: List[Chunk] = []
        for content, path in _split_md_by_headers(raw.text):
            out.extend(self._split_section(content, path, raw))
        return out

    # -- HTML：按 h1~h6 切章节 ------------------------------------------- #
    def _split_html(self, raw: RawDoc) -> List[Chunk]:
        # 纯 Python 按 <h1>~<h6> 标题切分（不再依赖 langchain 的
        # HTMLHeaderTextSplitter，避免 import 段错误）
        out: List[Chunk] = []
        for content, path in _split_html_by_headers(raw.text):
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
                # 只把本 chunk 文本里真实出现的 [[FIG:...]] 占位符作为该 chunk 的图，
                # 不再整章透传（解决 97% 子 chunk 带无关图、二次关联噪音大的问题）。
                fig_paths = _EMBEDDED_FIG_RE.findall(p)
                children.append(Chunk(
                    text=p,
                    source=raw.source,
                    file_name=raw.file_name,
                    access_level=raw.access_level,
                    chunk_index=idx,
                    parent_id=pid,
                    parent_content=_window_around(p, pcontent, PARENT_WINDOW_CHARS),
                    is_parent=False,
                    section_path=path or None,
                    chunk_type=kind,
                    page=raw.page,
                    figure_paths=fig_paths,
                ))
                idx += 1
        return children

    # -- 已分章节单元：整章为父、细切为子（small-to-big 真正落地） -------- #
    def _split_prebuilt_section(self, raw: RawDoc) -> List[Chunk]:
        """loader 已把 PDF 按章节聚合为带 section_path 的 RawDoc。

        这里做真正的父子文档：
        - **父 chunk** = 整章全文（含下属小章节的文字/表格/图片），
          即「大章节包含小章节」；父 chunk 仅入库不检索（主检索已排除 is_parent）。
        - **子 chunk** = 整章细切的小片段（检索单元）；每个子 chunk 透传
          **滑动窗口** parent_content（`_window_around`，子片段前后各 N 字），
          而非整章——根治整章超 8192 字节被静默截断、长章节答案入库即丢的问题。
          figure_paths = 整章所有图片，生成答案时可把本章全部相关图片一并还原。

        表格块 [TABLE]...[/TABLE] 经 `_segment` 保护为原子片段，不被切断。
        每个章节单元长度已在 loader 端约束在 ~8000 字内；滑动窗口总长 ≪ 8192，
        Milvus 字段上限（content/parent_content 均为 8192 字节）永不会触发。
        """
        CONTENT_CAP = 8192  # Milvus 字段上限保护
        pcontent = raw.text[:CONTENT_CAP]
        pid = hashlib.md5(
            ("§".join(raw.section_path or []) + "\n" + raw.text[:200]).encode("utf-8")
        ).hexdigest()[:16]
        # 父 chunk（is_parent=True，仅存不检索）
        parent = Chunk(
            text=pcontent,
            source=raw.source,
            file_name=raw.file_name,
            access_level=raw.access_level,
            chunk_index=0,
            parent_id="",
            parent_content="",
            is_parent=True,
            section_path=raw.section_path or None,
            chunk_type="section",
            page=raw.page,
            figure_paths=_EMBEDDED_FIG_RE.findall(pcontent),
        )
        out: List[Chunk] = [parent]
        # 子 chunk：先保护 [TABLE] 块原子不切，再对 prose 细切
        segs = _segment(pcontent)
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
                out.append(Chunk(
                    text=p,
                    source=raw.source,
                    file_name=raw.file_name,
                    access_level=raw.access_level,
                    chunk_index=idx,
                    parent_id=pid,
                    parent_content=_window_around(p, pcontent, PARENT_WINDOW_CHARS),
                    is_parent=False,
                    section_path=raw.section_path or None,
                    chunk_type=kind,
                    page=raw.page,
                    figure_paths=_EMBEDDED_FIG_RE.findall(p),
                ))
                idx += 1
        return out

    # -- 图感知：先抽图块，再切 child ----------------------------------- #
    def _split_figure_aware(self, raw: RawDoc) -> List[Chunk]:
        """PDF 等无结构文本：抽出图块（图级召回）后，再切 child（细节召回）。

        关键：把 figure_paths 转换成 [[FIG:...]] 文本占位符并写入 chunk.text，
        这样 LLM 在 context 里能看到图引用，回答时按 prompt 保留占位符，
        前端即可渲染为真实图片。

        入库三类 chunk（PDF 特有）：
        - figure_block：图标题 + 邻近上下文（或整页），负责「通信流程图」这类图查询；
          仅在锚点命中时存在（PyPDF text 流不含真图 caption 时可能为 0）。
        - page：每页整页作为兜底索引，PyPDF 抽不出图 caption 的图也能通过页级
          embedding 召回，LLM 可看到图周围段落文字作为上下文。
        - child：常规 section 细切，负责协议字段、报文格式等细节查询。
        """
        out: List[Chunk] = []
        # 把本页抽到的真图/渲染图路径转成 [[FIG:...]] 占位符，并插入到页文本开头。
        # 这样 figure_block / page / 首个 child 都会携带占位符，确保 LLM 可见。
        fig_placeholders = ""
        if raw.figure_paths:
            fig_placeholders = "\n".join(
                f"[[FIG:{p}]]" for p in raw.figure_paths)
        if fig_placeholders:
            text_with_figs = fig_placeholders + "\n\n" + raw.text
        else:
            text_with_figs = raw.text

        fig_blocks = _extract_figure_blocks(text_with_figs)
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
                text=text_with_figs,
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
        # 常规 child 切分（细节召回）；基于带占位符的文本，首个 child 会携带占位符
        out.extend(self._split_section(text_with_figs, [], raw))
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
