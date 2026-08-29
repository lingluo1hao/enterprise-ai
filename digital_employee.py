"""
================================================================================
 digital_employee.py — 数字员工（Digital Employee）目标驱动执行层（P0）
================================================================================

 定位
 ----
 「数字员工」在 enterprise-ai 中的 P0 入口。设计原则：**叠加而非重写**——
 完全复用 advanced_rag_agent 已有的 RAGOrchestrator（内含 PlanningAgent 拆解、
 ReActAgent 执行、DocSearchSkill 真连 Milvus、CalculatorSkill）。

 与现有 advanced_rag_agent.py 的关系
 ------------------------------------
   advanced_rag_agent.RAGOrchestrator.query()  = 单轮「问题 → 答案」反应式入口
   digital_employee.DigitalEmployee           = 目标驱动型：
                                                 · 单个复合目标 / 多个独立目标
                                                 · 逐个交给 RAGOrchestrator 完成
                                                 · 聚合为结构化报告 + 维护 task 状态

 数字员工 = 在已有 RAG 问答智能体之上，叠一层「自主目标执行」编排层。
 底层检索 / 路由 / 记忆 / 工具全部复用同一份逻辑，遵守 AGENTS.md「同一份逻辑」铁律。

 运行方式
 --------
   python digital_employee.py "JM-S509 的定位精度和续航分别是什么？"
   python digital_employee.py "目标1" "目标2" --fast
   python digital_employee.py --goal "帮我整理产品 FAQ" --admin
   python digital_employee.py "目标" --json        # 机器可读报告
   python digital_employee.py "目标" --role researcher   # 指定岗位档案

 P0.5 新增（v2 方案 §8）：
   1. 岗位档案 EmployeeProfile —— 从 config/employee_profile.yaml 加载
      （JD / 职责边界 / 工具白名单 / 验收阈值）；文件缺失时用内置缺省。
   2. 验收门 ReviewGate —— done 不再只看「没抛异常」：
        检索信号（独立复核命中数 / 最优距离）+ 文本信号（长度 / 查空措辞）
        → verdict ∈ {passed, low_quality, rejected}
        → status  ∈ {done, degraded, rejected}
      查空 / 低分 / 答案像道歉 → 不得标 done（员工本质八要素之「验收」）。
   双链路约定：验收门只消费 vector_db 与答案文本，不往 RAGOrchestrator 加东西。

 环境要求：同 advanced_rag_agent（Milvus + Ollama 网关可达）。
================================================================================
"""

import os
import sys
import time
import json
import argparse

from advanced_rag_agent import create_llm, VectorStoreManager, RAGOrchestrator

# 岗位档案默认路径（可被环境变量 EMPLOYEE_PROFILE 覆盖）
PROFILE_PATH = os.getenv(
    "EMPLOYEE_PROFILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "config", "employee_profile.yaml"),
)

# 内置缺省阈值：YAML 缺失 / 岗位未配置时兜底，保证验收门永远在场
_DEFAULT_REVIEW = {
    "min_hits": 1,
    "max_distance": 0.75,
    "min_answer_chars": 30,
    "reject_phrases": ["未找到", "没有找到", "未检索到", "没有相关",
                       "无法回答", "抱歉", "查不到"],
    # T-D2 文本兜底（Milvus AUTOINDEX+COSINE dense 失真——纯 dense 通道实跑
    # 见过 -0.0325=1-1.0325、历史记忆同向量返回 1.0——距离分支不可盲信）：
    "min_goal_doc_sim": 0.10,     # 问题↔Top1片段 difflib 相似度下限（抓 KB 污染/答非所问）
    "min_answer_coverage": 0.15,  # 答案 bigram 命中片段比例下限（抓答案游离检索）
}


class EmployeeProfile:
    """
    岗位档案（P0.5 配置文件版）

    数字员工的「身份」要素：JD + 职责边界 + 工具白名单 + 验收阈值。
    user_role 是权限位（admin/user），岗位档案才是「这个员工是干什么的」。
    """

    def __init__(self, data: dict):
        self.data = data or {}
        self.name = self.data.get("name", "未命名岗位")
        self.mission = self.data.get("mission", "")
        self.responsibilities = self.data.get("responsibilities", [])
        self.tool_allowlist = self.data.get("tool_allowlist", [])
        self.review = {**_DEFAULT_REVIEW, **(self.data.get("review") or {})}

    @classmethod
    def load(cls, role: str = "researcher", path: str = None) -> "EmployeeProfile":
        """
        从 YAML 加载指定岗位档案。

        :param role: 岗位 key（如 researcher / kb_ops）
        :param path: YAML 路径，缺省 PROFILE_PATH
        :return: 岗位档案；文件缺失 / 岗位未配置时返回带缺省阈值的空档案
                （验收门不因配置缺失而失效——宁可误判低分，不可假绿）
        """
        try:
            import yaml
            with open(path or PROFILE_PATH, "r", encoding="utf-8") as f:
                data = (yaml.safe_load(f) or {}).get("employees", {}).get(role)
            if data is None:
                print(f"  [EmployeeProfile] ⚠ 岗位「{role}」未在档案中定义，使用内置缺省")
                return cls({})
            print(f"  [EmployeeProfile] 已加载岗位「{role}」：{data.get('name', role)}")
            return cls(data)
        except FileNotFoundError:
            print(f"  [EmployeeProfile] ⚠ 档案文件不存在（{path or PROFILE_PATH}），使用内置缺省")
            return cls({})
        except Exception as e:  # yaml 解析失败等：降级但不放行
            print(f"  [EmployeeProfile] ⚠ 档案加载失败（{e}），使用内置缺省")
            return cls({})


class ReviewGate:
    """
    验收门（Reviewer，P0.5 + T-D2 兜底）

    判定一个目标是否「真完成」，两路独立信号（不信任生成侧自我汇报）：
      检索信号：用目标原文独立复核一次向量检索（命中数 / 最优距离 /
                距离合法性 / 文本相似度兜底）——抓「查空也硬合成答案」
                与「KB 污染导致答非所问」。
      文本信号：答案长度 + 查空/道歉措辞（开头命中视为 rejected，
                正文中出现视为 low_quality——引用语境不算全错）。

    T-D2 距离兜底（验收门纯 dense 通道实测 Milvus AUTOINDEX+COSINE 失真：
      相似度>1 → dist=1-sim 出负，P0 终验见过 -0.0325=1-1.0325；根治后
      复跑稳定合法正值，本兜底保留作防御）：
      best_distance 不在 COSINE 合法区间 [0,2] → distance_valid=False，
      距离分支不参与判定，改用文本相似度：
        goal_doc_sim   问题↔Top1片段 difflib 相似度——过低=检索与问题无关
                       （KB 污染/答非所问的主信号）
        answer_coverage 答案 bigram 命中片段比例——过低=答案游离于检索之外
                       （简化版忠实度）

    verdict 映射：
      rejected    查空 / 答案为空或过短 / 开头就是道歉 → 不标 done
      low_quality 有答案但检索距离差 / 距离失真且文本兜底不过线 /
                  正文含查空措辞 → degraded
      passed      两路信号均过线 → done

    检索复核失败（如 Milvus 抖动）时只降级用文本信号，并在 signals 里
    如实标注 retrieval_check=unavailable——不假装检查过。
    """

    def __init__(self, profile: EmployeeProfile):
        self.cfg = profile.review

    @staticmethod
    def _doc_text(doc) -> str:
        """容错取检索片段文本（langchain Document / 裸 str / 其他对象）。"""
        return getattr(doc, "page_content", None) or str(doc or "")

    @staticmethod
    def _text_coverage(answer: str, context: str) -> float:
        """答案字符 bigram 在检索片段中的命中率（简化忠实度信号）。"""
        text, ctx = (answer or "").strip(), context or ""
        if len(text) < 2 or not ctx:
            return 0.0
        grams = [text[i:i + 2] for i in range(len(text) - 1)]
        hit = sum(1 for g in grams if g in ctx)
        return round(hit / len(grams), 4)

    def review(self, goal: str, answer: str, vector_db,
               user_role: str = None, tenant_id: str = "default") -> dict:
        """
        :param user_role/tenant_id: 复核检索与主链路同权（同租户同密级）——
            否则 restricted 文档（如 Jimi 手册）复核时查空，误判 rejected。
        """
        import difflib
        signals = {}

        # --- 检索信号：独立复核 ---
        # 优先走纯 dense 通道拿**真实余弦距离**（1-sim ∈ [0,2]，可与阈值比较）；
        # 混合检索的 RRF -fused 分数无量纲不可比（负数是它取负所致，非失真）
        best_distance, hits, top_text = None, 0, ""
        try:
            dense_fn = getattr(vector_db, "dense_search_with_distance", None)
            if dense_fn is not None:
                results = dense_fn(goal, k=4, filter_role=user_role,
                                   user_id="digital_employee",
                                   tenant_id=tenant_id)
            else:
                results = vector_db.similarity_search_with_score(
                    goal, k=4, filter_role=user_role,
                    user_id="digital_employee", tenant_id=tenant_id)
            hits = len(results or [])
            if hits:
                best = min(results, key=lambda r: r[1])
                best_distance = round(best[1], 4)
                top_text = self._doc_text(best[0])
            signals["retrieval_check"] = "ok"
        except Exception as e:
            signals["retrieval_check"] = f"unavailable: {e}"
        signals["retrieval_hits"] = hits
        signals["best_distance"] = best_distance

        # T-D2：距离合法性（COSINE 合法区间 [0,2]；负数=失真，分支不可用）
        dist_valid = (best_distance is not None
                      and -1e-6 <= best_distance <= 2.0)
        signals["distance_valid"] = dist_valid
        signals["goal_doc_sim"] = round(difflib.SequenceMatcher(
            None, goal or "", top_text or "").ratio(), 4)
        signals["answer_coverage"] = self._text_coverage(answer, top_text)

        # --- 文本信号 ---
        text = (answer or "").strip()
        head = text[:60]
        phrase_hit = next(
            (p for p in self.cfg["reject_phrases"] if p in text), None
        )
        signals["answer_chars"] = len(text)
        signals["phrase_hit"] = phrase_hit

        # --- 判定（先硬后软） ---
        if not text or len(text) < self.cfg["min_answer_chars"]:
            verdict = "rejected"                      # 空答案 / 过短
        elif signals["retrieval_check"] == "ok" and hits < self.cfg["min_hits"]:
            verdict = "rejected"                      # 查空
        elif phrase_hit and phrase_hit in head:
            verdict = "rejected"                      # 开头就是道歉
        elif dist_valid and best_distance > self.cfg["max_distance"]:
            verdict = "low_quality"                   # 检索距离差（距离可信时）
        elif not dist_valid and best_distance is not None:
            # T-D2：距离失真 → 文本相似度兜底（双低才判差——实测协议表格类
            # 片段 goal_doc_sim 天然低至 0.02，单看会误杀正确答案）
            if (signals["goal_doc_sim"] < self.cfg["min_goal_doc_sim"]
                    and signals["answer_coverage"] < self.cfg["min_answer_coverage"]):
                verdict = "low_quality"   # 检索无关 + 答案游离（KB 污染/答非所问）
            elif phrase_hit:
                verdict = "low_quality"   # 正文含查空措辞
            else:
                verdict = "passed"
        elif phrase_hit:
            verdict = "low_quality"                   # 正文含查空措辞
        else:
            verdict = "passed"

        return {"verdict": verdict, "signals": signals}



class DigitalEmployee:
    """
    数字员工协调器（P0 骨架 + P0.5 验收门）

    职责：
      1. 初始化底层能力（LLM 网关 + Milvus + RAGOrchestrator）—— 全部复用，零分叉
      2. 加载岗位档案（JD / 职责边界 / 工具白名单 / 验收阈值）
      3. 接收目标（单个复合目标 / 多个独立目标）
      4. 逐个目标交给 RAGOrchestrator 完成「查资料并总结」
      5. 验收门复核：done 由质量信号决定，不再由「没抛异常」决定
      6. 维护 task 状态、聚合为结构化报告

    status 四态：
      done      执行成功且验收通过
      degraded  有结果但质量存疑（检索距离差 / 正文含查空措辞）
      rejected  验收不通过（查空 / 空答案 / 开头道歉）——不得算完成
      failed    执行异常
    """

    def __init__(self, fast_mode: bool = False, user_role: str = "user",
                 profile_role: str = "researcher", tenant_id: str = "default"):
        # 岗位档案：数字员工的「身份」（P0.5）
        self.profile = EmployeeProfile.load(profile_role)
        self.review_gate = ReviewGate(self.profile)
        # LLM 出口：优先 llm_gateway 多模型路由（回答/规划 DeepSeek-V4-PRO
        # 优先，本地 qwen2:7b 兜底——2026-08-29 应用户要求接入；gateway.chat
        # 与 BaseLLM.chat 完全同签名，PlanningAgent/ReActAgent 零改动）。
        # 网关不可用时回退 Ollama 直连（create_llm），保证可用性。
        try:
            from llm_gateway import get_gateway
            self.llm = get_gateway(verbose=False)
            print("[系统] LLM 出口：llm_gateway 多模型路由（DeepSeek 优先）")
        except Exception as e:
            print(f"[系统] ⚠ llm_gateway 不可用（{e}），回退本地 Ollama 直连")
            self.llm = create_llm()
        # 复用唯一向量后端 Milvus
        self.vector_db = VectorStoreManager.init_vector_store()
        # 复用编排器（已含 PlanningAgent + ReActAgent + DocSearchSkill + CalculatorSkill）
        # tenant_id 透传到 DocSearchSkill 检索下推（多租户隔离——终验踩过：
        # 文档在 jm/yh 租户而执行跑 default 租户 → 0 命中）
        self.tenant_id = tenant_id
        self.user_role = user_role
        self.orchestrator = RAGOrchestrator(
            self.llm, self.vector_db, fast_mode=fast_mode, user_role=user_role,
            tenant_id=tenant_id
        )
        self.tasks = []  # 简单 task 状态（P1 接 user_memories 持久化）

    def execute_goal(self, goal: str, user_role: str = None,
                     user: str = None) -> dict:
        """
        执行单个目标，返回结构化结果。

        P0.5：query 成功只是「跑完」，还要过验收门才算 done——
        查空 / 答案像道歉 / 检索距离差，分别落 rejected / degraded。
        T-D3：岗位能力门——本执行路径是 RAG 问答（doc_search），岗位工具
        白名单不含 doc_search 的业务岗（库管/采购）明确拒绝，不再假装跑
        （它们的业务操作应经 MCP 工具总线，不走知识问答）。
        """
        allow = self.profile.tool_allowlist or []
        if allow and "doc_search" not in allow:
            task = {
                "goal": goal, "status": "rejected", "elapsed_s": 0.0,
                "result": "",
                "error": f"岗位「{self.profile.name}」工具白名单不含 doc_search，"
                         "不具备知识问答执行路径（业务操作请经 MCP 工具总线）",
                "review": None,
            }
            self.tasks.append(task)
            return task
        t0 = time.time()
        err = None
        review = None
        try:
            # Web 多用户共用实例：每次执行显式传当前用户角色/身份
            #（与 rag_web_server /api/query 同款防提权模型），防角色串
            result = self.orchestrator.query(
                goal, user_role=user_role or self.user_role,
                user=user or "digital_employee")
        except Exception as e:  # 单个目标失败不影响其余目标
            result = ""
            status = "failed"
            err = str(e)
        else:
            # 复核检索与本次执行同权（此前误用实例构造时的 base 角色——
            # super_admin 执行主链路跨租户命中、复核却按 user 查空 → 误拒）
            review = self.review_gate.review(
                goal, result, self.vector_db,
                user_role=user_role or self.user_role,
                tenant_id=self.tenant_id)
            status = {"passed": "done",
                      "low_quality": "degraded",
                      "rejected": "rejected"}[review["verdict"]]
        elapsed = round(time.time() - t0, 1)
        task = {
            "goal": goal,
            "status": status,
            "elapsed_s": elapsed,
            "result": result,
            "error": err,
            "review": review,  # {verdict, signals}——验收证据随报告留痕
        }
        self.tasks.append(task)
        return task

    def execute_goals(self, goals: list, user_role: str = None,
                      user: str = None) -> list:
        """批量执行多个目标并聚合报告。"""
        reports = []
        for g in goals:
            g = (g or "").strip()
            if not g:
                continue
            reports.append(self.execute_goal(g, user_role=user_role, user=user))
        return reports

    def report(self, reports: list, as_json: bool = False) -> str:
        """把执行报告渲染为可读文本或 JSON。"""
        if as_json:
            return json.dumps(
                {"tasks": reports, "summary": self.summary(reports)},
                ensure_ascii=False,
                indent=2,
            )
        lines = []
        s = self.summary(reports)
        lines.append("=" * 70)
        lines.append(
            f"数字员工执行报告（岗位：{self.profile.name}）："
            f"{len(reports)} 个目标，验收通过 {s['done']}"
        )
        if s["degraded"] or s["rejected"]:
            lines.append(
                f"  ⚠ 质量存疑 {s['degraded']} · 验收不通过 {s['rejected']}"
                "（这两类不计入完成）"
            )
        lines.append("=" * 70)
        for i, r in enumerate(reports, 1):
            mark = {"done": "✓", "degraded": "△", "rejected": "✗",
                    "failed": "✗"}[r["status"]]
            lines.append(
                f"\n目标 {i} [{mark} {r['status']} · {r['elapsed_s']}s]：{r['goal']}"
            )
            if r["status"] == "done":
                lines.append("-" * 60)
                lines.append(r["result"])
            elif r["status"] in ("degraded", "rejected"):
                lines.append("-" * 60)
                lines.append(r["result"])
                rv = (r.get("review") or {}).get("signals", {})
                lines.append(f"  ⚠ 验收信号：{json.dumps(rv, ensure_ascii=False)}")
            else:
                lines.append(f"  ✗ 失败：{r['error']}")
        lines.append("\n" + "=" * 70)
        lines.append(
            f"汇总：验收通过 {s['done']}/{s['total']}（degraded {s['degraded']}"
            f" / rejected {s['rejected']} / failed {s['failed']}），"
            f"总耗时 {s['total_elapsed_s']}s"
        )
        lines.append("=" * 70)
        return "\n".join(lines)

    @staticmethod
    def summary(reports: list) -> dict:
        return {
            "total": len(reports),
            "done": sum(1 for r in reports if r["status"] == "done"),
            "degraded": sum(1 for r in reports if r["status"] == "degraded"),
            "rejected": sum(1 for r in reports if r["status"] == "rejected"),
            "failed": sum(1 for r in reports if r["status"] == "failed"),
            "total_elapsed_s": round(sum(r["elapsed_s"] for r in reports), 1),
        }


def main():
    parser = argparse.ArgumentParser(description="企业级数字员工 — 目标驱动执行层")
    parser.add_argument("goals", nargs="*", help="一个或多个目标（自然语言）")
    parser.add_argument("--goal", help="单个复合目标（与位置参数二选一）")
    parser.add_argument("--fast", action="store_true", help="快速模式（跳过查询重写）")
    parser.add_argument("--admin", action="store_true", help="以特权用户身份执行")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出报告")
    parser.add_argument("--role", default="researcher",
                        help="岗位档案 key（config/employee_profile.yaml），如 researcher / kb_ops")
    parser.add_argument("--tenant", default="default",
                        help="租户（多租户隔离：知识库按租户分库，如 jm / yh）")
    args = parser.parse_args()

    goals = list(args.goals)
    if args.goal:
        goals.append(args.goal)
    if not goals:
        print("⚠ 未提供目标。用法：python digital_employee.py \"目标1\" \"目标2\" [--role kb_ops]")
        return

    user_role = "admin" if args.admin else "user"
    try:
        emp = DigitalEmployee(fast_mode=args.fast, user_role=user_role,
                              profile_role=args.role, tenant_id=args.tenant)
    except Exception as e:
        print(f"\n❌ 数字员工初始化失败（请确认 Milvus/Ollama 可达）：{e}")
        return

    reports = emp.execute_goals(goals)
    print(emp.report(reports, as_json=args.json))


if __name__ == "__main__":
    main()
