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

import glob
import os
import re
from typing import List, Optional, Any

from .types import RawDoc
from .chunk import FIGURE_ANCHOR_RE

# 默认权限规则（与 advanced_rag_agent.DOC_ACCESS_RULES 语义一致）。
# 生产路径下由 VectorStoreManager 传入 AccessControlFilter.get_access_level 覆盖。
# 规则外置到 access_rules.yaml：filename 关键字 -> 级别（public/restricted）；
# 缺失文件或 yaml 不可用则退化为空（默认全部 public）。
DOC_ACCESS_RULES: dict = {}
try:
    import yaml  # 可选依赖：未安装时退化为全 public
    _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "config", "access_rules.yaml")
    if os.path.isfile(_cfg_path):
        with open(_cfg_path, "r", encoding="utf-8") as _fh:
            _cfg = yaml.safe_load(_fh) or {}
        DOC_ACCESS_RULES = {str(k): v for k, v in (_cfg.get("access_rules") or {}).items()}
except Exception:
    DOC_ACCESS_RULES = {}


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


def _table_to_markdown(rows: List[List[Any]]) -> Optional[str]:
    """把 PyMuPDF 表格二维数组转成标准 Markdown 表格。"""
    if not rows or len(rows) < 1:
        return None
    # 清理单元格：去空值、压平换行、转义 |
    def _cell(v):
        if v is None:
            return ""
        s = str(v).replace("\n", " ").replace("\r", " ").replace("|", "\\|")
        return s.strip()
    cleaned = [[_cell(c) for c in row] for row in rows]
    n_cols = max(len(r) for r in cleaned)
    # 补齐列数，避免 Markdown 格式错乱
    for r in cleaned:
        while len(r) < n_cols:
            r.append("")
    lines = []
    for i, row in enumerate(cleaned):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            lines.append("| " + " | ".join(["---"] * n_cols) + " |")
    return "\n".join(lines)


def _figures_dir(path: str) -> tuple:
    """返回 PDF 图/表输出目录的 (绝对路径, 相对项目根目录的 URL 前缀)。"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isdir(os.path.join(project_root, "knowledge")):
        project_root = os.getcwd()
    stem = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(project_root, "assets", "figures", stem)
    rel_prefix = f"assets/figures/{stem}"
    return out_dir, rel_prefix


def _load_pdf(path: str) -> Optional[list]:
    """使用 PyMuPDF 提取 PDF 文本与表格。

    - 优先把页面内表格识别出来并转为 Markdown 表格；
    - 表格区域同时渲染为图片保存，作为 LLM 可视化兜底；
    - 表格区域从普通文本中剔除，避免重复；
    - 按阅读顺序（y 坐标）把段落与表格拼接；
    - 每页返回一个 RawDoc，便于与 figure_paths 按页对应；
    - 缺失 PyMuPDF 或提取失败时返回 None，由调用方降级到 PyPDFLoader。
    """
    try:
        import fitz
    except ImportError:
        return None
    try:
        out_dir, rel_prefix = _figures_dir(path)
        os.makedirs(out_dir, exist_ok=True)
        # 清理旧版表格图（避免同名文件重建后残留过期图片）
        for old in glob.glob(os.path.join(out_dir, "table_p*.png")):
            try:
                os.remove(old)
            except OSError:
                pass

        pages: List["object"] = []
        with fitz.open(path) as doc:
            for page_idx, page in enumerate(doc, 1):
                # 1) 找表格
                table_objs = []
                try:
                    tabs = page.find_tables()
                    table_objs = tabs.tables if tabs else []
                except Exception as te:
                    print(f"[ingest] 第 {page_idx} 页表格检测失败: {te}")

                table_markdowns = []   # (Rect, markdown_text)
                table_bboxes = []
                table_fig_paths: List[str] = []
                for tidx, tab in enumerate(table_objs):
                    try:
                        rows = tab.extract()
                    except Exception:
                        continue
                    md = _table_to_markdown(rows)
                    if not md:
                        continue
                    bbox = fitz.Rect(tab.bbox)
                    table_bboxes.append(bbox)
                    # 用 [TABLE] 标签包裹，便于 chunker 识别为保护片段
                    table_markdowns.append((bbox, f"\n\n[TABLE]\n{md}\n[/TABLE]\n\n"))

                    # 把表格区域渲染成图片，作为 Markdown 表格的可视化兜底
                    try:
                        scale = 2.0
                        pad = 6
                        clip = fitz.Rect(
                            max(0, bbox.x0 - pad),
                            max(0, bbox.y0 - pad),
                            min(page.rect.width, bbox.x1 + pad),
                            min(page.rect.height, bbox.y1 + pad),
                        )
                        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip)
                        fname = f"table_p{page_idx:03d}_{tidx+1}.png"
                        pix.save(os.path.join(out_dir, fname))
                        table_fig_paths.append(f"{rel_prefix}/{fname}")
                    except Exception as te:
                        print(f"[ingest] 第 {page_idx} 页表格 {tidx+1} 渲染失败: {te}")

                # 2) 提取非表格文本块
                text_blocks = []
                for b in page.get_text("blocks"):
                    bbox = fitz.Rect(b[:4])
                    txt = b[4].strip()
                    if not txt:
                        continue
                    # 若文本块与表格 bbox 显著相交，则视为表格内容，跳过
                    if any(bbox.intersects(tb) for tb in table_bboxes):
                        continue
                    text_blocks.append((bbox.y0, txt))

                # 3) 表格也作为独立块加入，按 y0 排序实现阅读顺序
                for bbox, md in table_markdowns:
                    text_blocks.append((bbox.y0, md))

                text_blocks.sort(key=lambda x: x[0])
                page_text = "\n\n".join(t for _, t in text_blocks if t.strip())
                if page_text.strip():
                    page_text = f"[Page {page_idx}]\n{page_text.strip()}"
                else:
                    page_text = f"[Page {page_idx}]"
                pages.append(RawDoc(
                    text=page_text,
                    source=path,
                    file_name=os.path.basename(path),
                    page=page_idx,
                    figure_paths=table_fig_paths,
                ))

        if not pages:
            return None
        return pages
    except Exception as e:
        print(f"[ingest] PyMuPDF 读取 PDF 失败 {os.path.basename(path)}: {e}")
        return None


def load_file(path: str) -> List["object"]:
    """加载单个文件，返回 RawDoc 列表。不支持/缺依赖的格式返回 []（打印警告）。"""
    from .types import RawDoc
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".txt", ".md"):
            from langchain_community.document_loaders import TextLoader
            docs = TextLoader(path, encoding="utf-8").load()
        elif ext == ".pdf":
            res = _load_pdf(path)
            if res is not None:
                out = res
            else:
                # 降级：PyPDFLoader（纯文本，表格结构会丢失）
                from langchain_community.document_loaders import PyPDFLoader
                docs = PyPDFLoader(path).load()
                out = []
                for d in docs:
                    pg = d.metadata.get("page") if hasattr(d, "metadata") else None
                    page = (int(pg) + 1) if pg is not None else None
                    out.append(RawDoc(
                        text=d.page_content or "",
                        source=path,
                        file_name=os.path.basename(path),
                        page=page,
                    ))
            # 通用 PDF 图抽取：每页可能多张图（PyMuPDF + 像素墨迹 + 连通分量）
            # 按页对齐：out 已按 1-based 页码排序，figs_per_page[i] 对应第 i+1 页
            figs_per_page = _extract_figures(path)
            for raw_doc, figs in zip(out, figs_per_page):
                if figs:
                    raw_doc.figure_paths.extend(figs)
            return out
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
    # 通用 PDF 图抽取：每页可能多张图（PyMuPDF + 像素墨迹 + 连通分量）
    # 与语言/排版/有无 caption 无关；表格自然被文字密度过滤；模板背景在像素层自动消失
    figs_per_page = _extract_figures(path)
    for raw_doc, figs in zip(out, figs_per_page):
        if figs:
            raw_doc.figure_paths.extend(figs)
    return out


# --------------------------------------------------------------------------- #
# 通用 PDF 图抽取（PyMuPDF + numpy + scipy，与语言/caption/排版无关）
# --------------------------------------------------------------------------- #
# 算法：
#   1) PyMuPDF 渲染整页 → 像素墨迹
#   2) page.get_text("blocks") 拿文字块 bbox → 文字遮罩（膨胀 4-5 px）
#   3) 图形墨迹 = 整页墨迹 − 文字墨迹
#   4) scipy.ndimage.label 连通分量 → 逐分量判定面积/宽高比/墨迹密度/文字密度
#   5) 通过的真图分量 → 从原图裁剪（保留图内文字标签）→ fig_p{NNN}_{k}.png
#
# 为什么通用：
#   - 无 caption 正则：不依赖页面文本是否含「图N / XX图」，英文 PDF / 无 caption 图照样识别
#   - 模板/背景图层自动消失：get_drawings 返回的伪图元在渲染像素层不存在，就不进 graphic ink
#   - 表格自然被过滤：表格连通分量 bbox 内塞满文字（高密度）→ 丢弃
#   - logo/分隔线/细线：面积阈值与墨迹密度过滤
# --------------------------------------------------------------------------- #
def _extract_figures(path: str) -> List[List[str]]:
    """对 PDF 每一页做通用图抽取；每页返回 0..N 张裁剪图相对路径列表。

    - 缺失 fitz/numpy/scipy/Pillow：返回 []（graceful 降级，pipeline 仍可跑）
    - 非 PDF：返回 []
    - 自动清理旧版 page_*.png（整页渲染）与 fig_pNNN.png（旧单图命名）
    """
    if not path.lower().endswith(".pdf"):
        return []
    try:
        import fitz  # PyMuPDF
        import numpy as np
        from PIL import Image, ImageDraw, ImageFilter
        from scipy import ndimage
    except ImportError as e:
        print(f"[ingest] 跳过 PDF 图抽取 {os.path.basename(path)}：依赖缺失 {e.name}（pip install pymupdf numpy scipy pillow）")
        return []

    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isdir(os.path.join(project_root, "knowledge")):
            project_root = os.getcwd()
        stem = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.join(project_root, "assets", "figures", stem)
        os.makedirs(out_dir, exist_ok=True)
        # 清理旧版产物：page_*.png（整页渲染）、fig_pNNN.png（旧单图命名）
        for old in (
            glob.glob(os.path.join(out_dir, "page_*.png"))
            + glob.glob(os.path.join(out_dir, "fig_p[0-9][0-9][0-9].png"))
        ):
            try:
                os.remove(old)
            except OSError:
                pass

        scale = 2.0
        pad = 24
        MIN_AREA_RATIO = 0.01   # 图区面积 / 页面面积
        MAX_INK_DENSITY = 0.12  # 墨迹密度上限（过高=填充色块/水印，非真图）
        MIN_INK_DENSITY = 0.015 # 图区墨迹像素 / bbox 面积（过滤稀疏虚线）
        MAX_LONG_ROWS = 0       # 跨 80% bbox 宽度的长横线行数（>此值=表格，丢弃）；流程图箭头是斜的→不存在长横线；表格分隔线是横的→必然命中

        results: List[List[str]] = []
        with fitz.open(path) as doc:
            for i in range(doc.page_count):
                page_figs: List[str] = []
                try:
                    page = doc[i]
                    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
                    img = pix.pil_image()
                    W, H = img.size
                    # 文字遮罩
                    blocks = page.get_text("blocks")
                    m = Image.new("L", (W, H), 0)
                    md = ImageDraw.Draw(m)
                    for b in blocks:
                        x0, y0, x1, y1 = b[:4]
                        md.rectangle(
                            [x0*scale-4, y0*scale-4, x1*scale+4, y1*scale+4],
                            fill=255,
                        )
                    m = m.filter(ImageFilter.MaxFilter(5))
                    tm = np.asarray(m) > 128
                    ink = np.asarray(img.convert("L")) < 200
                    graphic = ink & ~tm
                    # 连通分量
                    labeled, ncomp = ndimage.label(graphic)
                    for k in range(1, ncomp + 1):
                        ys, xs = np.where(labeled == k)
                        if len(xs) == 0:
                            continue
                        bx0, bx1 = int(xs.min()), int(xs.max())
                        by0, by1 = int(ys.min()), int(ys.max())
                        bw, bh = bx1 - bx0, by1 - by0
                        if bw <= 0 or bh <= 0:
                            continue
                        if (bw * bh) / (W * H) < MIN_AREA_RATIO:
                            continue
                        asp = bw / bh if bh else 0
                        if asp < 0.1 or asp > 10:
                            continue
                        ink_density = len(xs) / (bw * bh)
                        if ink_density < MIN_INK_DENSITY or ink_density > MAX_INK_DENSITY:
                            continue
                        # 表格判别：长横线（跨 80% bbox 宽度的连续墨迹行）数量
                        sub = np.ascontiguousarray(graphic[by0:by1+1, bx0:bx1+1])
                        long_rows = 0
                        for row in sub:
                            if not row.any():
                                continue
                            diffs = np.diff(np.r_[0, row.astype(np.int8), 0])
                            starts = np.where(diffs == 1)[0]
                            ends = np.where(diffs == -1)[0]
                            max_run = max((e - s) for s, e in zip(starts, ends))
                            if max_run >= 0.8 * bw:
                                long_rows += 1
                        if long_rows > MAX_LONG_ROWS:
                            continue
                        # 裁剪原图（保留图内文字标签）
                        crop = img.crop((
                            max(0, bx0 - pad),
                            max(0, by0 - pad),
                            min(W, bx1 + pad),
                            min(H, by1 + pad),
                        ))
                        fname = f"fig_p{i+1:03d}_{len(page_figs)+1}.png"
                        crop.save(os.path.join(out_dir, fname), format="PNG", optimize=True)
                        page_figs.append(f"assets/figures/{stem}/{fname}")

                    # 兜底：页面含图标题锚点但连通分量没抽出真图时（常见流程图/时序图
                    # 被长横线过滤误伤），直接渲染整页作为 fallback 图，确保 LLM 能看到。
                    if not page_figs and FIGURE_ANCHOR_RE.search(page.get_text() or ""):
                        try:
                            fname = f"page_{i+1:03d}.png"
                            page.get_pixmap(matrix=fitz.Matrix(scale, scale)).save(
                                os.path.join(out_dir, fname))
                            page_figs.append(f"assets/figures/{stem}/{fname}")
                        except Exception as fe:
                            print(f"[ingest] PDF 第 {i+1} 页整页渲染兜底失败: {fe}")
                except Exception as e:
                    print(f"[ingest] PDF 第 {i+1} 页图抽取失败: {e}")
                results.append(page_figs)
        return results
    except Exception as e:
        print(f"[ingest] PDF 图抽取异常 {os.path.basename(path)}: {e}")
        return []
