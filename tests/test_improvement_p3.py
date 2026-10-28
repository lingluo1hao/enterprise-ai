# -*- coding: utf-8 -*-
"""
自改进闭环（improvement 包）测试
================================================================================

零外部依赖：不连 MySQL、不连 Milvus、不调 LLM。
DB 不可用时应走降级路径，因此本文件在本机任何状态下都能跑。

核心验证目标（对应方案的验收口径）
----------------------------------
1. 护栏真的拦得住：触碰不可变目标的提案根本生成不出来
2. 变差的提案真的会被拒：故意构造指标倒退，门禁必须 reject
3. 样本不足不给自动放行：冻结集 5 条时只能 manual，不能 auto
4. 沙箱不留痕迹：注入覆盖后必须能清干净
5. 回滚可用：merged -> rolled_back 且 before 快照完整
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from improvement import whitelist as W            # noqa: E402
from improvement import schema as SC              # noqa: E402
from improvement import gate as GT                # noqa: E402
from improvement import generator as GN           # noqa: E402
from improvement import sandbox as SB             # noqa: E402
from improvement import store as ST               # noqa: E402

import prompt_manager as PM                       # noqa: E402


def _proposal(cause="R5", target="generate_answer",
              before="旧提示词", after="新提示词", layer=W.LAYER_PROMPT):
    return SC.ImprovementProposal.new(
        root_cause=cause, change_layer=layer, change_target=target,
        payload_before=before, payload_after=after, rationale="测试")


class _FakeStore:
    """不连库的 store 替身，只提供 gate 需要的三个查询。"""
    def __init__(self, blacklist=(), merged_today=0):
        self._black = set(blacklist)
        self._merged = merged_today

    def is_blacklisted(self, target):
        return target in self._black

    def today_auto_merge_count(self):
        return self._merged


# ============================================================================
# T1 不可变护栏
# ============================================================================

class TestWhitelist(unittest.TestCase):

    def test_immutable_targets_rejected(self):
        for t in ("judge_prompt", "tenant_filter_expr", "holdout_golden",
                  "evalgrade_task"):
            with self.assertRaises(W.TargetRejected, msg=f"{t} 本应被拒"):
                W.validate_target(t)

    def test_unknown_target_rejected(self):
        with self.assertRaises(W.TargetRejected):
            W.validate_target("随便一个没登记的东西")

    def test_empty_target_rejected(self):
        with self.assertRaises(W.TargetRejected):
            W.validate_target("")

    def test_whitelisted_prompt_ok(self):
        self.assertEqual(W.validate_target("generate_answer"), W.LAYER_PROMPT)
        self.assertEqual(W.validate_target("rewrite_first"), W.LAYER_PROMPT)

    def test_param_layer(self):
        self.assertEqual(W.validate_target("top_k"), W.LAYER_PARAM)

    def test_r8_has_no_prompt_target(self):
        """跨租户泄漏零容忍：不给任何提示词层的自动改动目标。"""
        self.assertEqual(W.targets_for_cause("R8"), [])

    def test_r5_routes_to_generation(self):
        self.assertIn("generate_answer", W.targets_for_cause("R5"))

    def test_param_cast(self):
        self.assertEqual(W.cast_param("top_k", "8"), 8)
        self.assertTrue(W.cast_param("hybrid", "true"))
        self.assertFalse(W.cast_param("hybrid", "false"))


# ============================================================================
# 确定性变异算子（离线可跑）
# ============================================================================

class TestDeterministicMutation(unittest.TestCase):

    def test_r5_appends_constraint(self):
        cur = "你是助手。"
        out = GN.deterministic_mutate("R5", "generate_answer", cur)
        self.assertIsNotNone(out)
        self.assertIn("知识库中没有相关内容", out)
        self.assertTrue(out.startswith(cur))

    def test_idempotent(self):
        cur = "你是助手。"
        once = GN.deterministic_mutate("R5", "generate_answer", cur)
        twice = GN.deterministic_mutate("R5", "generate_answer", once)
        self.assertIsNone(twice, "同一约束不应追加两次")

    def test_wrong_target_returns_none(self):
        """把「只据上下文」塞进分类提示词是错的，必须有拦截。"""
        self.assertIsNone(GN.deterministic_mutate("R5", "classify", "分类。"))

    def test_propose_rejects_immutable_target(self):
        p = GN.propose_for_cause("R5", "judge_prompt", "旧", use_llm=False)
        self.assertIsNone(p, "不可变目标不该生成出提案")

    def test_propose_produces_change(self):
        p = GN.propose_for_cause("R5", "generate_answer", "你是助手。", use_llm=False)
        self.assertIsNotNone(p)
        self.assertNotEqual(p.payload_before, p.payload_after)
        self.assertEqual(p.change_layer, W.LAYER_PROMPT)

    def test_fingerprint_stable(self):
        a = _proposal(after="同样的内容")
        b = _proposal(after="同样的内容")
        self.assertEqual(a.fingerprint(), b.fingerprint())
        c = _proposal(after="不一样的内容")
        self.assertNotEqual(a.fingerprint(), c.fingerprint())


# ============================================================================
# 状态机
# ============================================================================

class TestStateMachine(unittest.TestCase):

    def test_happy_path(self):
        p = _proposal()
        self.assertEqual(p.status, SC.ST_DRAFT)
        self.assertTrue(p.transit(SC.ST_VALIDATING))
        self.assertTrue(p.transit(SC.ST_PASSED, by="gate"))
        self.assertTrue(p.transit(SC.ST_MERGED, by="auto"))
        self.assertEqual(p.decided_by, "auto")
        self.assertTrue(p.transit(SC.ST_ROLLED_BACK))
        # 回滚只记动作，不覆盖「当初谁合的」——复盘时要能找到这个人
        self.assertEqual(p.decided_by, "auto")

    def test_illegal_transition_rejected(self):
        p = _proposal()
        self.assertFalse(p.transit(SC.ST_MERGED), "draft 不能直接跳到 merged")
        self.assertEqual(p.status, SC.ST_DRAFT)

    def test_terminal_states(self):
        p = _proposal()
        p.status = SC.ST_REJECTED
        self.assertFalse(p.transit(SC.ST_MERGED))

    def test_no_before_payload_rejected(self):
        """没有变更前快照 = 无法回滚，store 必须拒。"""
        store = ST.ImprovementStore(verbose=False)
        if not store.available:
            self.skipTest("MySQL 不可用，跳过写库断言")
        with self.assertRaises(ValueError):
            store.save_proposal(_proposal(before=""))


# ============================================================================
# 门禁
# ============================================================================

class TestGate(unittest.TestCase):

    # 默认走参数层：本组用例喂的都是检索层指标，而只有 P2 参数变体的主指标
    # 才是 recall@5 / ndcg@5。P1 提示词层看的是答案层指标，见下方 TestGateAnswerLayer。
    def _run(self, base, var, holdout_cases=40, layer=W.LAYER_PARAM,
             b_causes=None, v_causes=None, store=None, seen=True):
        p = _proposal(layer=layer)
        return GT.evaluate(p, base, var, holdout_cases,
                           b_causes or {}, v_causes or {},
                           store=store if store is not None else _FakeStore(),
                           seen_target_before=seen)

    def test_improvement_passes(self):
        """提示词层走答案层主指标：忠实度与相关性双升 -> 自动放行。"""
        g = self._run({"faithfulness": 60.0, "relevancy": 50.0},
                      {"faithfulness": 65.0, "relevancy": 53.0},
                      layer=W.LAYER_PROMPT)
        self.assertTrue(g.passed)
        self.assertEqual(g.decision, GT.DECISION_AUTO)

    def test_regression_rejected(self):
        """故意变差：必须拦下。这是本套系统最该守住的一条。"""
        g = self._run({"recall@5": 65.0, "ndcg@5": 53.0},
                      {"recall@5": 60.0, "ndcg@5": 50.0})
        self.assertFalse(g.passed)
        self.assertEqual(g.decision, GT.DECISION_REJECT)
        self.assertIn(GT.G_HOLDOUT, g.failed_gates())

    def test_no_change_rejected(self):
        g = self._run({"recall@5": 60.0, "ndcg@5": 50.0},
                      {"recall@5": 60.0, "ndcg@5": 50.0})
        self.assertFalse(g.passed, "零增益不该放行")

    def test_small_holdout_forces_manual(self):
        """冻结集 5 条 -> 统计效力不足 -> 只能人工，不能自动。"""
        g = self._run({"recall@5": 60.0, "ndcg@5": 50.0},
                      {"recall@5": 70.0, "ndcg@5": 60.0}, holdout_cases=5)
        self.assertTrue(g.passed)
        self.assertEqual(g.decision, GT.DECISION_MANUAL)
        self.assertIn(GT.G_SAMPLE, g.failed_gates())

    def test_r8_zero_tolerance(self):
        g = self._run({"recall@5": 60.0}, {"recall@5": 70.0},
                      v_causes={"R8": 1})
        self.assertFalse(g.passed)
        self.assertIn(GT.G_REGRESSION, g.failed_gates())

    def test_new_r5_regression_rejected(self):
        """不许用新增幻觉去换召回。"""
        g = self._run({"recall@5": 60.0}, {"recall@5": 70.0},
                      b_causes={"R5": 1}, v_causes={"R5": 3})
        self.assertFalse(g.passed)

    def test_latency_regression_rejected(self):
        g = self._run({"recall@5": 60.0, "p95_latency_ms": 1000.0},
                      {"recall@5": 70.0, "p95_latency_ms": 1500.0})
        self.assertFalse(g.passed)

    def test_missing_metrics_rejected(self):
        """取不到指标按失败处理，宁可拦下不可误放。"""
        g = self._run({}, {})
        self.assertFalse(g.passed)

    def test_param_layer_needs_human(self):
        g = self._run({"recall@5": 60.0}, {"recall@5": 70.0},
                      layer=W.LAYER_PARAM)
        self.assertTrue(g.passed)
        self.assertEqual(g.decision, GT.DECISION_MANUAL)

    def test_first_time_target_needs_human(self):
        g = self._run({"recall@5": 60.0}, {"recall@5": 70.0}, seen=False)
        self.assertEqual(g.decision, GT.DECISION_MANUAL)

    def test_blacklist_blocks(self):
        p = _proposal()
        g = GT.evaluate(p, {"recall@5": 60.0}, {"recall@5": 70.0}, 40,
                        {}, {}, store=_FakeStore(blacklist=("generate_answer",)),
                        seen_target_before=True)
        self.assertFalse(g.passed)
        self.assertIn(GT.G_BLACKLIST, g.failed_gates())

    def test_budget_guard(self):
        p = _proposal()
        g = GT.evaluate(p, {"recall@5": 60.0}, {"recall@5": 70.0}, 40,
                        {}, {}, store=_FakeStore(merged_today=3),
                        seen_target_before=True)
        self.assertFalse(g.passed)
        self.assertIn(GT.G_BUDGET, g.failed_gates())

    def test_immutable_target_rejected_by_gate(self):
        p = _proposal(target="judge_prompt")
        g = GT.evaluate(p, {"recall@5": 60.0}, {"recall@5": 70.0}, 40,
                        {}, {}, store=_FakeStore(), seen_target_before=True)
        self.assertFalse(g.passed)
        self.assertIn(GT.G_IMMUTABLE, g.failed_gates())


# ============================================================================
# 沙箱
# ============================================================================

class TestSandbox(unittest.TestCase):

    def test_suite_selection(self):
        """提示词改动跑检索层评测会看不到效果，必须选对层。"""
        self.assertEqual(SB.suite_for_layer(W.LAYER_PROMPT), "answer")
        self.assertEqual(SB.suite_for_layer(W.LAYER_PARAM), "retrieval")

    def test_split_is_deterministic(self):
        cases = [{"case_id": f"c{i}"} for i in range(20)]
        d1, h1 = SB.split_cases(cases)
        d2, h2 = SB.split_cases(cases)
        self.assertEqual([c["case_id"] for c in h1],
                         [c["case_id"] for c in h2], "切分必须可复现")
        self.assertEqual(len(d1) + len(h1), len(cases))
        self.assertGreater(len(d1), 0)
        self.assertGreater(len(h1), 0)

    def test_holdout_size_is_predictable(self):
        """门禁拿「冻结集有多少条」当硬判据，条数必须可预期，不能靠分桶运气。"""
        for n in (10, 15, 20, 40):
            cases = [{"case_id": f"c{i}"} for i in range(n)]
            _, hold = SB.split_cases(cases)
            self.assertEqual(len(hold), int(round(n * SB.HOLDOUT_RATIO)),
                             f"n={n} 时冻结集条数应为 round(n*ratio)")
            self.assertEqual(len(hold) + len(SB.split_cases(cases)[0]), n)

    def test_holdout_never_empty_or_full(self):
        cases = [{"case_id": f"c{i}"} for i in range(15)]
        dev, hold = SB.split_cases(cases)
        self.assertGreater(len(hold), 0)
        self.assertGreater(len(dev), 0)

    def test_prompt_override_inject_and_clear(self):
        """沙箱覆盖必须能注入、能生效、能清干净。"""
        PM.clear_prompt_overrides()
        try:
            PM.set_prompt_override("generate_answer", "沙箱变体内容")
            got = PM.get_prompt_manager().get_prompt("generate_answer")
            self.assertEqual(got["system"], "沙箱变体内容")
            self.assertIn("generate_answer", PM.active_prompt_overrides())
        finally:
            PM.clear_prompt_overrides()
        self.assertEqual(PM.active_prompt_overrides(), {},
                         "跑完必须清干净，否则会污染线上")

    def test_p3_sandbox_refuses(self):
        """经验层没有进程内注入口，必须明确报不可用，而不是假装跑过。"""
        p = _proposal(layer=W.LAYER_PLAYBOOK, target="playbook")
        res = SB.ab_test(p, [])
        self.assertFalse(res.ok)
        self.assertIn("不支持", res.error)


# ============================================================================
# 降级
# ============================================================================

class TestDegrade(unittest.TestCase):

    def test_store_unavailable_is_silent(self):
        """库不可用时不能抛异常拖垮调用方。"""
        s = ST.ImprovementStore(verbose=False)
        if s.available:
            self.skipTest("MySQL 可用，跳过降级断言")
        self.assertFalse(s.save_proposal(_proposal()))
        self.assertIsNone(s.load_proposal("pp-x"))
        self.assertEqual(s.list_proposals(), [])
        self.assertFalse(s.is_blacklisted("generate_answer"))

    def test_metrics_extraction_missing(self):
        self.assertEqual(SC.extract_metrics(None), {})
        self.assertEqual(SC.extract_metrics({}), {})

    def test_metrics_extraction_from_summary(self):
        run = {"summary": {"recall@5": 60.0, "ndcg@5": 50.0}}
        m = SC.extract_metrics(run)
        self.assertEqual(m.get("recall@5"), 60.0)
        self.assertEqual(m.get("ndcg@5"), 50.0)

    def test_diff_metrics(self):
        d = SC.diff_metrics({"recall@5": 60.0}, {"recall@5": 65.0})
        self.assertEqual(d["recall@5"], 5.0)

    def test_ratio_metrics_normalized_to_percent_points(self):
        """evalkit 输出 0–1，门禁门槛是百分点 —— 不归一就永远达不到「+2.0」。

        2026-10-26 实测抓到：真实 answer 层跑出 faithfulness=0.6833，
        而 gate 打印的基线是 {}，G3 判「指标缺失」。
        """
        run = {"summary": {"recall@5": 0.875, "ndcg@5": 0.926, "mrr": 1.0,
                           "faithfulness": 0.6833, "relevancy": 0.9333}}
        m = SC.extract_metrics(run)
        self.assertAlmostEqual(m["recall@5"], 87.5, places=4)
        self.assertAlmostEqual(m["ndcg@5"], 92.6, places=4)
        self.assertAlmostEqual(m["faithfulness"], 68.33, places=4)
        self.assertAlmostEqual(m["relevancy"], 93.33, places=4)
        # 计数/延迟类不参与换算
        r2 = SC.extract_metrics({"summary": {"p95_latency_ms": 1200.0,
                                             "cases": 3, "failures": 1}})
        self.assertEqual(r2["p95_latency_ms"], 1200.0)
        self.assertEqual(r2["cases"], 3.0)

    def test_answer_metrics_extracted_from_real_answer_run(self):
        """answer suite 的指标必须进得来，否则 P1/P3 永远是「指标缺失」。"""
        self.assertIn("faithfulness", SC.METRIC_KEYS)
        self.assertIn("relevancy", SC.METRIC_KEYS)
        self.assertIn("faithfulness", SC.GATE_METRICS_BY_SUITE["answer"][0])
        self.assertEqual(
            [k for k, _ in SC.GATE_METRICS_BY_SUITE["retrieval"]],
            ["recall@5", "ndcg@5"])


class TestGateAnswerLayer(unittest.TestCase):
    """提示词层必须看答案层指标 —— 检索层差值实测精确为 0（假阴性）。"""

    def _p(self, layer):
        return _proposal(layer=layer)

    def test_prompt_layer_ignores_retrieval_only_gain(self):
        """只在检索层涨、答案层没动 -> 不算数（提示词变体在检索层本就看不出效果）。"""
        g = GT.evaluate(self._p(W.LAYER_PROMPT),
                        {"recall@5": 60.0, "faithfulness": 50.0},
                        {"recall@5": 70.0, "faithfulness": 50.0},
                        40, {}, {}, store=_FakeStore(), seen_target_before=True)
        self.assertFalse(g.passed)
        self.assertIn(GT.G_HOLDOUT, g.failed_gates())

    def test_prompt_layer_passes_on_faithfulness_gain(self):
        g = GT.evaluate(self._p(W.LAYER_PROMPT),
                        {"faithfulness": 60.0, "relevancy": 50.0},
                        {"faithfulness": 64.0, "relevancy": 53.0},
                        40, {}, {}, store=_FakeStore(), seen_target_before=True)
        self.assertTrue(g.passed)
        self.assertEqual(g.decision, GT.DECISION_AUTO)

    def test_param_layer_uses_retrieval_metrics(self):
        g = GT.evaluate(self._p(W.LAYER_PARAM),
                        {"recall@5": 60.0, "ndcg@5": 50.0},
                        {"recall@5": 65.0, "ndcg@5": 53.0},
                        40, {}, {}, store=_FakeStore(), seen_target_before=True)
        self.assertTrue(g.passed)
        # 参数层改的是检索行为，一律人工审批（设计如此）
        self.assertEqual(g.decision, GT.DECISION_MANUAL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
