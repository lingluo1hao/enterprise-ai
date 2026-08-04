"""ingest CLI：生产环境手动/定时增量 ingestion 入口。

用法（在 VM 上，项目根目录）：
  python -m ingest.cli ingest            # 增量 ingestion（仅处理变更文件）
  python -m ingest.cli ingest --force    # 全量重建
  python -m ingest.cli ingest --dry-run  # 预检：只打印会做什么，不落库
  python -m ingest.cli status            # 查看已追踪文件清单
  python -m ingest.cli delete <文件>      # 删除单个文件（向量库+清单）
  python -m ingest.cli rebuild <文件>     # 强制重建单个文件

环境变量沿用主模块：
  MILVUS_URI / MILVUS_COLLECTION / OLLAMA_BASE_URL / OLLAMA_EMBED_MODEL
"""

import os
import argparse
import sys

# 让 `python -m ingest.cli` 从项目根跑时能 import 到 ingest 与 advanced_rag_agent
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _build_pipeline(folder: str, dry_run: bool):
    from .pipeline import IngestPipeline
    from .store import MilvusStoreBackend
    from .embed import make_ollama_embedder

    embedder = make_ollama_embedder()
    from pymilvus import MilvusClient
    uri = os.getenv("MILVUS_URI", "http://192.168.200.128:19530")
    collection = os.getenv("MILVUS_COLLECTION", "rag_docs")
    client = MilvusClient(uri=uri)
    store = MilvusStoreBackend(client, collection)

    # 生产环境沿用主模块的权限规则
    try:
        from advanced_rag_agent import AccessControlFilter
        access_fn = AccessControlFilter.get_access_level
    except Exception:
        access_fn = None

    return IngestPipeline(
        folder=folder, embedder=embedder.embed_documents,
        store=store, access_fn=access_fn, dry_run=dry_run)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="ingest", description="百万级 RAG 数据面：增量 ingestion CLI")
    sub = p.add_subparsers(dest="cmd")

    pi = sub.add_parser("ingest", help="增量/全量 ingestion")
    pi.add_argument("--folder", default="./knowledge")
    pi.add_argument("--force", action="store_true", help="全量重建")
    pi.add_argument("--dry-run", action="store_true", help="只预检不落库")
    pi.add_argument("--files", nargs="*", help="仅处理指定文件")

    ps = sub.add_parser("status", help="查看已追踪文件清单")
    ps.add_argument("--folder", default="./knowledge")

    pd = sub.add_parser("delete", help="删除单个文件")
    pd.add_argument("file")
    pd.add_argument("--folder", default="./knowledge")

    pr = sub.add_parser("rebuild", help="强制重建单个文件")
    pr.add_argument("file")
    pr.add_argument("--folder", default="./knowledge")

    args = p.parse_args(argv)

    if args.cmd == "ingest":
        pipe = _build_pipeline(args.folder, args.dry_run)
        rep = pipe.run(force=args.force, files=args.files,
                       progress_cb=lambda m: print("[ingest]", m))
        print(">>>", rep.summary())
        pipe.close()
    elif args.cmd == "status":
        pipe = _build_pipeline(args.folder, False)
        pipe.status()
        pipe.close()
    elif args.cmd == "delete":
        pipe = _build_pipeline(args.folder, False)
        n = pipe.delete(args.file)
        print(f">>> 已删除 {n} 条实体: {args.file}")
        pipe.close()
    elif args.cmd == "rebuild":
        pipe = _build_pipeline(args.folder, False)
        rep = pipe.rebuild(args.file)
        print(">>>", rep.summary())
        pipe.close()
    else:
        p.print_help()


if __name__ == "__main__":
    main()
