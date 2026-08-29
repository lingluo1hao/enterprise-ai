"""
================================================================================
 kb_ops_employee.py — 知识库运营专员（第一岗位，P0.5 巡检 MVP）
================================================================================

 定位
 ----
 数字员工 v2 方案（docs/guides/digital_employee_plan.md §6）选定的第一岗位：
 把「bad case 自动诊断」从人工逐条触发，升级为岗位化的巡检班次——

   巡检 bad_cases 未处理条目
     → 逐条跑 agentworkflow 诊断流水线（复用 diagnose_bad_case，零分叉）
     → 高置信 case 整理为人工复审清单（只标记，不自动应用修复）
     → 产出当日运营日报并落盘

 全部工作方法均为既有积木（bad_cases / agentworkflow / evolution），本文件
 只做「岗位化编排」：加载岗位档案 + 班次上限（防失控）+ 日报产出。

 安全边界（P0.5 最小形态）
 ----
   - 单班次处理上限 max_cases（岗位档案 shift.max_cases，默认 20）
   - 高置信修复建议只进「人工复审清单」，不自动改数据（waiting 思路）
   - dry_run 模式：诊断不回写 bad_cases 状态

 运行方式
 --------
   python kb_ops_employee.py --shift                  # 跑一轮巡检
   python kb_ops_employee.py --shift --limit 5        # 只处理前 5 条
   python kb_ops_employee.py --shift --dry-run        # 只诊断不回写
   python kb_ops_employee.py --shift --json           # 机器可读输出
   日报落盘：logs/shift_reports/shift_YYYYmmdd_HHMMSS.md（P1 接 APScheduler 定时）

 环境要求：同 agentworkflow（Milvus + Ollama 网关 + MySQL 可达）。
================================================================================
"""

import os
import json
import time
import argparse
from types import SimpleNamespace

from advanced_rag_agent import create_llm, VectorStoreManager
from memory_store import MySQLMemoryStore
from agentworkflow.diagnose import diagnose_bad_case
from digital_employee import EmployeeProfile

ROOT = os.path.dirname(os.path.abspath(__file__))


class KnowledgeOpsEmployee:
    """
    知识库运营专员（岗位：kb_ops）

    职责（见 config/employee_profile.yaml kb_ops 段）：
      1. 巡检 bad_cases 未处理条目
      2. 逐条跑 agentworkflow 诊断流水线，产出根因
      3. 高置信 case 整理为人工复审清单（不自动修改数据）
      4. 产出当日运营日报并落盘
    """

    def __init__(self, dry_run: bool = False, actor: str = "kb_ops_employee"):
        self.profile = EmployeeProfile.load("kb_ops")
        shift = self.profile.data.get("shift") or {}
        self.source_status = shift.get("source_status", "open")
        self.max_cases = int(shift.get("max_cases", 20))
        self.high_confidence = set(shift.get("high_confidence", ["高", "中高"]))
        self.report_dir = os.path.join(ROOT, shift.get("report_dir",
                                                      "logs/shift_reports"))
        self.dry_run = dry_run
        self.actor = actor  # token 用量归因 & 审计标识

        # 组件自建（CLI 独立模式）；诊断流水线按依赖注入方式接收，
        # 与 rag_web_server 注入生产组件同一套接口（agentworkflow 既有约定）
        self.llm = create_llm(verbose=False)
        self.vector_db = VectorStoreManager.init_vector_store()
        self.memory_store = MySQLMemoryStore()
        self.components = SimpleNamespace(
            llm=self.llm, vector_db=self.vector_db, memory_store=self.memory_store
        )

    # ------------------------------------------------------------------
    # 巡检班次
    # ------------------------------------------------------------------
    def run_shift(self, limit: int = None) -> dict:
        """
        跑一轮巡检：捞未处理 bad case → 逐条诊断 → 汇总班次结果。

        :param limit: 覆盖岗位档案的单班次上限（防失控的硬顶）
        :return: 班次 dict（stats / cases / review_list / report_path）
        """
        max_cases = min(int(limit), self.max_cases) if limit else self.max_cases
        print(f"\n[kb_ops] 岗位「{self.profile.name}」开始巡检班次"
              f"（捞取状态：{self.source_status}，上限 {max_cases} 条，"
              f"dry_run={self.dry_run}）")

        cases = self.memory_store.list_bad_cases(
            status=self.source_status, limit=max_cases
        )
        if not cases:
            print(f"[kb_ops] 无状态为「{self.source_status}」的待处理 bad case，本班次空转。")
        else:
            print(f"[kb_ops] 捞到 {len(cases)} 条待处理 bad case，逐条诊断...")

        shift = {
            "employee": self.profile.name,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dry_run": self.dry_run,
            "memory_available": bool(getattr(self.memory_store, "available", True)),
            "cases": [],
            "review_list": [],   # 高置信 → 人工复审清单（不自动应用）
        }

        for bc in cases:
            bc_id = bc.get("id")
            print(f"\n[kb_ops] --- 诊断 bad case #{bc_id}：{(bc.get('query') or '')[:40]}")
            try:
                # 复用 agentworkflow 完整诊断（组件注入，不重复建连接）；
                # 非 dry_run 时内部会把 open → in_progress 回写
                r = diagnose_bad_case(bc_id, components=self.components,
                                      dry_run=self.dry_run, actor=self.actor)
            except Exception as e:  # 单条诊断异常不炸整轮班次
                r = {"ok": False, "bc_id": bc_id, "error": str(e)}
                print(f"[kb_ops] ✗ bad case #{bc_id} 诊断异常：{e}")

            entry = {
                "bc_id": bc_id,
                "query": bc.get("query"),
                "ok": bool(r.get("ok")),
                "root_cause": r.get("root_cause"),
                "confidence": r.get("confidence"),
                "engine": r.get("engine"),
                "written": bool(r.get("written")),
                "run_file": r.get("run_file"),
                "error": r.get("error"),
            }
            shift["cases"].append(entry)

            # 高置信 → 人工复审清单（岗位 JD：只标记，等 P2 接审批流）
            if (r.get("ok") and r.get("root_cause")
                    and r.get("confidence") in self.high_confidence):
                shift["review_list"].append({
                    **entry,
                    "diagnosis": r.get("diagnosis", ""),
                    "suggestion": "根因置信度高，建议人工复审并评估修复 patch"
                                  "（数字员工不自动应用）",
                })

        diagnosed = [c for c in shift["cases"] if c["ok"]]
        attributed = [c for c in diagnosed if c["root_cause"]]
        shift["stats"] = {
            "total": len(shift["cases"]),
            "diagnosed": len(diagnosed),
            "attributed": len(attributed),
            "unattributed": len(diagnosed) - len(attributed),
            "errors": len(shift["cases"]) - len(diagnosed),
            "high_confidence": len(shift["review_list"]),
        }
        shift["report_path"] = self.write_report(shift)
        print(f"\n[kb_ops] 班次结束：{shift['stats']}；日报已落盘 "
              f"{shift['report_path']}")
        return shift

    # ------------------------------------------------------------------
    # 运营日报（写能力最小形态：Markdown 落盘）
    # ------------------------------------------------------------------
    def write_report(self, shift: dict) -> str:
        """把班次结果渲染为 Markdown 日报并落盘，返回文件路径。"""
        os.makedirs(self.report_dir, exist_ok=True)
        fname = time.strftime("shift_%Y%m%d_%H%M%S") + ".md"
        path = os.path.join(self.report_dir, fname)

        s = shift["stats"]
        lines = [
            f"# 知识库运营日报 — {shift['started_at']}",
            "",
            f"- 岗位：{shift['employee']}（kb_ops_employee.py）",
            f"- 模式：{'dry-run（未回写 bad_cases 状态）' if shift['dry_run'] else '正式班次（诊断后 open→in_progress）'}",
            f"- 记忆层：{'MySQL 正常' if shift['memory_available'] else '⚠ MySQL 不可用（fallback 内存态，结果可能不完整）'}",
            "",
            "## 班次统计",
            "",
            "| 待处理 | 已诊断 | 归因成功 | 未归因 | 异常 | 高置信待复审 |",
            "|---|---|---|---|---|---|",
            f"| {s['total']} | {s['diagnosed']} | {s['attributed']} "
            f"| {s['unattributed']} | {s['errors']} | {s['high_confidence']} |",
            "",
        ]

        if not shift["cases"]:
            lines += ["本班次无待处理 bad case。", ""]
        else:
            lines += ["## 诊断明细", "",
                      "| # | 问题摘要 | 根因 | 置信度 | 回写 | 轨迹 |",
                      "|---|---|---|---|---|---|"]
            for c in shift["cases"]:
                q = (c.get("query") or "")[:30]
                rc = c.get("root_cause") or ("✗ " + (c.get("error") or "未归因")[:40])
                lines.append(
                    f"| {c['bc_id']} | {q} | {rc} | {c.get('confidence') or '-'} "
                    f"| {'是' if c.get('written') else '否'} "
                    f"| {os.path.basename(c['run_file']) if c.get('run_file') else '-'} |"
                )
            lines.append("")

        if shift["review_list"]:
            lines += ["## 人工复审清单（高置信，数字员工不自动应用修复）", ""]
            for c in shift["review_list"]:
                lines += [
                    f"### bad case #{c['bc_id']}（{c['confidence']}）",
                    f"- 问题：{c.get('query')}",
                    f"- 根因：{c['root_cause']}",
                    f"- 建议：{c['suggestion']}",
                    f"- 诊断详情：{c.get('diagnosis', '')[:300]}",
                    "",
                ]
        else:
            lines += ["## 人工复审清单", "", "本班次无高置信条目。", ""]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return path


def main():
    parser = argparse.ArgumentParser(
        description="知识库运营专员 — bad case 巡检班次（数字员工第一岗位）")
    parser.add_argument("--shift", action="store_true", help="跑一轮巡检班次")
    parser.add_argument("--limit", type=int, help="本班次最多处理条数（仍受岗位档案 max_cases 硬顶）")
    parser.add_argument("--dry-run", action="store_true", help="只诊断不回写 bad_cases 状态")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出班次结果")
    args = parser.parse_args()

    if not args.shift:
        parser.print_help()
        print("\n用法示例：python kb_ops_employee.py --shift --limit 5")
        return

    try:
        emp = KnowledgeOpsEmployee(dry_run=args.dry_run)
    except Exception as e:
        print(f"\n❌ 运营专员初始化失败（请确认 Milvus/Ollama/MySQL 可达）：{e}")
        return

    shift = emp.run_shift(limit=args.limit)
    if args.json:
        print(json.dumps(shift, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
