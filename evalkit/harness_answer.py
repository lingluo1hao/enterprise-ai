"""
evalkit.harness_answer —— 答案层评测 Harness
================================================================================

【这一层在测什么】

检索 harness 只保证"正确的段落进了上下文"。但它管不了"模型拿到段落之后，
有没有忠实作答、有没有答非所问、有没有幻觉"。答案 harness 补上这最后一环：
端到端跑一遍真实问答链路，再让独立 judge 模型打分。

【为什么复用 LangGraphRAGApp.query 而不是自己写一遍生成】

和检索 harness 复用 `_do_retrieve` 是同一个道理——答案层更要复用线上代码，
否则评出来分数漂亮、线上依旧拉胯。这里直接调 `app.query()`，
评的就是生产环境那套「分类 → 检索 → 精排 → 改写 → 生成」全链路。

【两个必须处理好的隔离点】

  1. 缓存必须 neutral 化：app.query 有 Redis 答案缓存，命中就直接返回旧答案，
     既污染黄金集对比（改了 prompt 却测到旧答案），又让相邻 case 互相串味。
     所以在 harness 里把 lookup 强制返回 None、save 变 no-op，
     每次都跑真实生成，且绝不写回缓存。

  2. 记忆持久化（MySQL）层的降级：LangGraphRAGApp 初始化时
     MySQLMemoryStore 在 MySQL 不可用时**自动降级为内存模式**，
     create_task / update_task_status 在内存里跑，不阻塞评测。
     因此答案 harness 不依赖 MySQL 在线（只依赖 Milvus + Ollama，
     与检索 harness 同条件）。

【judge 打分维度】

  faithfulness  答案是否忠于检索到的上下文（抓幻觉）
  relevancy     答案是否切题（抓答非所问）
  硬判据        must_include / must_not_include 字符串匹配 + 拒答正确性（零成本）

硬判据不调 LLM，先 cheap 过滤掉一大半明显问题；judge 分数只做增强与排序依据。
"""

from __future__ import annotations

import os
import sys
import re
import time
import uuid
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalkit.schema import (  # noqa: E402
    AnswerCase, AnswerCaseResult, EvalRun, aggregate_answer,
)
from evalkit.judge import JudgeLLM  # noqa: E402

JUDGE_THRESHOLD = 0.6  # 与 AnswerCaseResult.passed 的门槛一致


# ------------------------------------------------------------------ 硬判据辅助
_NEG_TOKENS = ("不知道", "未", "无", "没", "不", "并非", "并不", "不会",
               "不予", "禁止", "不是", "未包含", "未涉及", "未提及", "没有直接")


def _clause_negated(clause: str, pos: int) -> bool:
    """clause[:pos] 是否含否定/拒答措辞 —— 用于否定感知禁词匹配。"""
    return any(tok in clause[:pos] for tok in _NEG_TOKENS)


def _forbidden_hits(answer: str, forbidden_list: List[str]) -> List[str]:
    """否定感知的禁词检查（零成本、确定性强）。

    - 支持 ``regex:`` 前缀做精确语义断言（如「心跳包句内出现 0x01」）。
    - 普通串：按标点/换行分句，仅当禁词所在子句『无否定词』时才算命中，
      避免「并不会发送短信」里的「会发送短信」被误杀。
    """
    if not answer:
        return []
    hits: List[str] = []
    for f in forbidden_list:
        if not f:
            continue
        if f.startswith("regex:"):
            if re.search(f[6:], answer):
                hits.append(f)
            continue
        if f not in answer:
            continue
        clauses = re.split(r"[。！？；\n.!?;]", answer)
        hit = False
        for clause in clauses:
            idx = clause.find(f)
            if idx == -1:
                continue
            if _clause_negated(clause, idx):
                continue
            hit = True
            break
        if hit:
            hits.append(f)
    return hits


class AnswerHarness:
    """
    答案层评测器：跑完整 RAG 链路 + judge 打分。

    参数：
        fast_mode:     True 时分类走规则（不调 LLM），加速；默认 False 走真实分类。
        judge_task:    judge 走网关的路由任务名，默认 evalgrade（deepseek → 本地小模型）。
        judge_threshold:  judge 分数低于此值判失败（仅影响 pass_rate，不影响硬判据）。
    """

    def __init__(self, fast_mode: bool = False, judge_task: str = "evalgrade",
                 judge_threshold: float = JUDGE_THRESHOLD):
        self.fast_mode = fast_mode
        self.judge = JudgeLLM(task=judge_task)
        self.judge_threshold = judge_threshold
        self.app = None
        self._orig_lookup = None
        self._orig_save = None

    # ---------------------------------------------------------------- setup
    def setup(self) -> Dict[str, Any]:
        """
        构造真实 RAG App 并 neutral 化缓存。
        返回本次生效的配置快照（写进 run.config）。
        """
        import langgraph_rag_agent as L

        # 注意：这里会真正连 Ollama + Milvus（与检索 harness 同条件）。
        # MySQL 不可用时记忆层自动降级为内存模式，不阻塞。
        self.app = L.LangGraphRAGApp(fast_mode=self.fast_mode)

        # neutral 化缓存：强制真实生成、禁止写回，避免污染黄金集与跨 case 串味。
        self._orig_lookup = self.app.cache.lookup
        self._orig_save = self.app.cache.save
        self.app.cache.lookup = lambda *a, **k: None
        self.app.cache.save = lambda *a, **k: None

        return {
            "fast_mode": self.fast_mode,
            "judge_task": self.judge.task,
            "judge_model": self.judge.model,
            "judge_available": self.judge.available,
            "judge_threshold": self.judge_threshold,
        }

    def teardown(self):
        """还原缓存行为，避免污染同进程内的后续调用（如 web server 复用）。"""
        if self.app is not None and self._orig_lookup is not None:
            try:
                self.app.cache.lookup = self._orig_lookup
                self.app.cache.save = self._orig_save
            except Exception:
                pass

    # ------------------------------------------------------------- run one
    def run_case(self, case: AnswerCase) -> AnswerCaseResult:
        """
        跑一条答案 case，返回带硬判据 + judge 分数的结果。
        """
        if self.app is None:
            raise RuntimeError("AnswerHarness 未 setup()，请先调用 setup()")

        uid = case.user_id
        try:
            uid_int = int(uid)
        except (TypeError, ValueError):
            uid_int = 1

        # 请求级上下文隔离（走 property → ContextVar，与线上一致）
        self.app.user = uid_int
        self.app.username = "eval"
        self.app.tenant_id = case.tenant_id

        # 每次用唯一 session_id，避免命中 Layer 1 内存历史（上一条答案被当成上下文）
        session_id = f"eval-{case.case_id}-{uuid.uuid4().hex[:8]}"

        t0 = time.time()
        try:
            answer = self.app.query(
                case.query, role=case.role, session_id=session_id,
                tenant_id=case.tenant_id, user_id=uid_int, username="eval",
            )
            err = ""
        except Exception as e:
            import traceback
            traceback.print_exc()
            answer, err = "", f"{type(e).__name__}: {e}"
        latency = (time.time() - t0) * 1000
        task_id = self.app.last_task_id or ""

        # ---- 硬判据（零成本，先跑）----
        missing = [m for m in case.must_include if m and m not in (answer or "")]
        forbidden = _forbidden_hits(answer, case.must_not_include)
        refused = self.judge.grade_refusal(answer).get("refused", False)
        if case.should_refuse:
            refuse_correct: Optional[bool] = refused  # 该拒答且真的拒了 => 正确
        else:
            refuse_correct = (not refused)

        # ---- 取模型实际看到的上下文，喂给 judge 做忠实度判定 ----
        contexts: List[str] = []
        try:
            pairs = self.app._do_retrieve([case.query], case.role)
            contexts = [getattr(d, "page_content", "") or "" for d, _ in pairs[:6]]
        except Exception as e:
            print(f"  [warn] 取上下文失败（不影响硬判据）: {e}")

        # ---- judge 打分 ----
        jr = self.judge.grade(case.query, answer, contexts)
        faith = jr.get("faithfulness")
        rel = jr.get("relevancy")
        jreason = jr.get("reason", "")
        jerr = jr.get("error", "")

        return AnswerCaseResult(
            case_id=case.case_id,
            query=case.query,
            answer=answer,
            missing_points=missing,
            forbidden_hits=forbidden,
            refused=refused,
            refuse_correct=refuse_correct,
            faithfulness=faith,
            relevancy=rel,
            judge_reason=jreason,
            latency_ms=round(latency, 1),
            task_id=task_id,
            tags=case.tags,
            error=(err or jerr),
        )

    # ------------------------------------------------------------- run all
    def run_suite(self, cases: List[AnswerCase], note: str = "",
                  quiet: bool = True) -> EvalRun:
        """跑完整答案评测套件，返回 EvalRun。"""
        config = self.setup()
        run = EvalRun.new("answer", config)
        run.note = note

        total = len(cases)
        results: List[Dict[str, Any]] = []
        print(f"\n[evalkit] 答案评测开始：{total} 条 case，judge={config.get('judge_model')}")
        try:
            for i, case in enumerate(cases, 1):
                res = self.run_case(case)
                d = res.__dict__
                d["_passed"] = res.passed  # 报表排序用
                results.append(d)

                flag = "✓" if res.passed else "✗"
                prob = []
                if res.missing_points:
                    prob.append(f"缺{len(res.missing_points)}点")
                if res.forbidden_hits:
                    prob.append(f"禁词{len(res.forbidden_hits)}")
                if res.refuse_correct is False:
                    prob.append("拒答错")
                if res.faithfulness is not None and res.faithfulness < self.judge_threshold:
                    prob.append(f"faith={res.faithfulness:.2f}")
                if res.relevancy is not None and res.relevancy < self.judge_threshold:
                    prob.append(f"rel={res.relevancy:.2f}")
                prob_txt = " · " + " ".join(prob) if prob else ""
                print(f"  [{i}/{total}] {flag} {case.case_id:<16} "
                      f"{prob_txt:<30} {case.query[:30]}")
        finally:
            self.teardown()

        run.results = results
        run.summary = aggregate_answer(results)
        run.finished_at = time.time()

        s = run.summary
        print(f"\n[evalkit] 完成：{s['count']} 条 | 通过率 {s['pass_rate']:.1%} | "
              f"硬通过率 {s['hard_pass_rate']:.1%}")
        print(f"[evalkit] faithfulness={s.get('faithfulness')} "
              f"relevancy={s.get('relevancy')} "
              f"拒答正确率={s.get('refuse_accuracy')}")
        return run
