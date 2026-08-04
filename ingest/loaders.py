"""多格式文档加载器（改造点 3）。

统一从「仅 .pdf/.txt」扩展到 docx/xlsx/pptx/html/md，并保留表格结构：
- .txt/.md   → TextLoader（零额外依赖）
- .pdf       → PyPDFLoader（项目已有）
- .html/.htm → BSHTMLLoader（带 bs4）；缺失则降级 TextLoader
- .docx      → Docx2txtLoader；缺失则降级 UnstructuredWordDocumentLoader
- .xlsx/.xls → openpyxl 逐 Sheet 逐行读（保留表头/单元格，利于表格问答）
- .pptx      → python-pptx 逐页读文本+表格

所有 loader 均为「懒导入」：缺失某个可选依赖时只跳过该格式，不拖垮整个模块，
保证 ingest 引擎在无全套解析依赖的机器上仍能跑核心链路（txt/md/pdf）。
"""

import os
from typing import List, Optional

# 默认权限规则（与 advanced_rag_agent.DOC_ACCESS_RULES 语义一致）。
# 生产路径下由 VectorStoreManager 传入 AccessControlFilter.get_access_level 覆盖。
DOC_ACCESS_RULES: dict = {}


def get_access_level(source: str) -> str:
    """按文件名关键字判定权限：命中 → 对应级别，否则 public。"""
    basename = os.path.basename(source)
    for keyword, level in DOC_ACCESS_RULES.items():
        if keyword in basename:
            return level
    return "public"


def _raw(text: str, source: str) -> List["object"]:
    from .types import RawDoc
    return [RawDoc(text=text, source=source, file_name=os.path.basename(source))]


def _load_xlsx(path: str) -> Optional[list]:
    try:
        from openpyxl import load_workbook
    except ImportError:
        return None
    wb = load_workbook(path, read_only=True, data_only=True)
    sheets = []
    for ws in wb.worksheets:
        lines = [f"# Sheet: {ws.title}"]
        for r, row in enumerate(ws.iter_rows(values_only=True), 1):
            cells = [str(c).strip() for c in row
                     if c is not None and str(c).strip() != ""]
            if cells:
                lines.append(f"R{r}: " + " | ".join(cells))
        if len(lines) > 1:
            sheets.append("\n".join(lines))
    if not sheets:
        return None
    return _raw("\n\n".join(sheets), path)


def _load_pptx(path: str) -> Optional[list]:
    try:
        from pptx import Presentation
    except ImportError:
        return None
    prs = Presentation(path)
    slides = []
    for i, slide in enumerate(prs.slides, 1):
        parts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                t = shape.text_frame.text.strip()
                if t:
                    parts.append(t)
            if shape.has_table:
                for row in shape.table.rows:
                    parts.append(" | ".join(c.text.strip()
                                            for c in row.cells if c.text.strip()))
        if parts:
            slides.append(f"[Slide {i}]\n" + "\n".join(parts))
    if not slides:
        return None
    return _raw("\n\n".join(slides), path)


def load_file(path: str) -> List["object"]:
    """加载单个文件，返回 RawDoc 列表。不支持/缺依赖的格式返回 []（打印警告）。"""
    from .types import RawDoc
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".txt", ".md"):
            from langchain_community.document_loaders import TextLoader
            docs = TextLoader(path, encoding="utf-8").load()
        elif ext == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            docs = PyPDFLoader(path).load()
        elif ext in (".html", ".htm"):
            try:
                from langchain_community.document_loaders import BSHTMLLoader
                docs = BSHTMLLoader(path).load()
            except Exception:
                docs = TextLoader(path, encoding="utf-8").load()
        elif ext == ".docx":
            try:
                from langchain_community.document_loaders import Docx2txtLoader
                docs = Docx2txtLoader(path).load()
            except Exception:
                from langchain_community.document_loaders import (
                    UnstructuredWordDocumentLoader,
                )
                docs = UnstructuredWordDocumentLoader(path).load()
        elif ext in (".xlsx", ".xls"):
            res = _load_xlsx(path)
            if res is None:
                print(f"[ingest] 跳过 {path}：缺少 openpyxl 依赖")
                return []
            return res
        elif ext == ".pptx":
            res = _load_pptx(path)
            if res is None:
                print(f"[ingest] 跳过 {path}：缺少 python-pptx 依赖")
                return []
            return res
        else:
            print(f"[ingest] 跳过不支持的格式: {path}")
            return []
    except ImportError as e:
        print(f"[ingest] 缺少依赖，跳过 {path}: {e}")
        return []
    except Exception as e:
        print(f"[ingest] 读取失败 {path}: {e}")
        return []

    out = []
    for d in docs:
        # PyPDFLoader 按页返回；page 转 1-based 人类可读页码（langchain 默认 0-based）
        pg = d.metadata.get("page") if hasattr(d, "metadata") else None
        page = (int(pg) + 1) if pg is not None else None
        out.append(RawDoc(
            text=d.page_content or "",
            source=path,
            file_name=os.path.basename(path),
            page=page,
        ))
    # 给每页渲染 PNG（PyPDF text 流不包含真图 caption，必须靠渲染图让前端"看见"）
    figures_dir = _render_pdf_pages(path, len(docs))
    if figures_dir:
        for raw_doc, pg_dir in zip(out, figures_dir):
            # 单页对应一张 PNG
            if pg_dir:
                raw_doc.figure_paths.append(pg_dir)
    return out


# --------------------------------------------------------------------------- #
# PDF 页面渲染（图级检索可视化支撑）
# --------------------------------------------------------------------------- #
def _render_pdf_pages(path: str, page_count: int) -> List[Optional[str]]:
    """把 PDF 每页渲染成 PNG，存到 assets/figures/<file_stem>/page_<p:03d>.png。

    返回长度 == page_count 的列表；某页渲染失败/整体失败时对应位置为 None。
    - 缺失 pypdfium2：返回 []（graceful 降级，pipeline 仍可运行，仅 figure 不可视）；
    - 非 PDF 文件：返回 []；
    - assets/figures/ 与同 stem 子目录自动创建。
    """
    if not path.lower().endswith(".pdf"):
        return []
    try:
        import pypdfium2 as pdfium
    except ImportError:
        print(f"[ingest] 跳过 PDF 渲染 {os.path.basename(path)}：缺少 pypdfium2（pip install pypdfium2）")
        return []

    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 也兼容从项目根运行（__file__ 路径即项目根的子目录）
        if not os.path.isdir(os.path.join(project_root, "knowledge")):
            project_root = os.getcwd()
        stem = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.join(project_root, "assets", "figures", stem)
        os.makedirs(out_dir, exist_ok=True)

        # 用 with 上下文显式关闭 PdfDocument，避免函数返回后靠 GC 关闭时
        # pypdfium2 打印 "still open" 警告（且警告里的文件名会被控制台误解码成乱码）。
        with pdfium.PdfDocument(path) as pdf:
            total = len(pdf)
            results: List[Optional[str]] = []
            for i in range(total):
                rel = f"assets/figures/{stem}/page_{i+1:03d}.png"
                abs_path = os.path.join(out_dir, f"page_{i+1:03d}.png")
                if os.path.exists(abs_path) and os.path.getsize(abs_path) > 0:
                    # 已渲染过：跳过（增量友好，避免每次重建都重渲染）
                    results.append(rel)
                    continue
                try:
                    page = pdf[i]
                    # scale=1.5 ≈ 150 DPI，A4 渲染约 1240×1754 px，体积 200~500 KB
                    pil = page.render(scale=1.5).to_pil()
                    pil.save(abs_path, format="PNG", optimize=True)
                    results.append(rel)
                except Exception as e:
                    print(f"[ingest] PDF 第 {i+1} 页渲染失败: {e}")
                    results.append(None)
            return results
    except Exception as e:
        print(f"[ingest] PDF 渲染异常 {os.path.basename(path)}: {e}")
        return []
