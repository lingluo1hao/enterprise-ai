"""evalkit 硬判据单元测试（零外部依赖，不连 VM / 不调 LLM）。

运行方式（项目根目录）：
  python tests/test_harness_grading.py

覆盖：
  1. 否定感知禁词 _forbidden_hits（分句 + 否定词跳过 + regex: 精确断言）
  2. 语义拒答 JudgeLLM._grade_refusal（窄关键词 + 语义正则双路）
  3. 集成：用上次真实回归的「冻结答案」+ 当前黄金集，重算 4 条历史误判，
     断言它们在新逻辑下消失（证明测试侧 B 修对了，不依赖重新跑 LLM）。

设计说明：测试侧 B 要修的是「测试框架误杀正确答案」，所以只需对『答案文本』
重跑新打分逻辑即可验证，无需再花 18 分钟重跑 qwen 生成。
"""

import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evalkit.harness_answer import _forbidden_hits, _clause_negated
from evalkit.judge import JudgeLLM

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  PASS  " + name)
    else:
        FAIL += 1
        print("  FAIL  " + name)


print("== 1. 否定感知禁词 _forbidden_hits ==")
# 否定句里的禁词不应计入
check("‘并不会发送短信’ 含 ‘会发送短信’ -> 不命中",
      _forbidden_hits("白名单打不通时，设备并不会发送短信。", ["会发送短信"]) == [])
# 非否定上下文里的禁词应命中
check("‘信号极弱（0x01）’ 含 ‘0x01’ -> 命中",
      _forbidden_hits("信号极弱（0x01）表示无信号。", ["0x01"]) == ["0x01"])
# 同一句内用 ‘不是0x01’ 否定 -> 不命中
check("‘心跳包是0x36，不是0x01’ -> 不命中",
      _forbidden_hits("心跳包是0x36，不是0x01。", ["0x01"]) == [])
# 多句：否定只在某一句 -> 仍不命中
check("‘我们未发送短信，仅拨号’ -> 不命中",
      _forbidden_hits("我们未发送短信，仅循环拨号。", ["发送短信"]) == [])
# 肯定句里的禁词 -> 命中
check("‘系统会发送短信通知’ -> 命中",
      _forbidden_hits("系统会发送短信通知用户。", ["会发送短信"]) == ["会发送短信"])

print("== 2. 语义拒答 JudgeLLM._grade_refusal ==")
check("‘未检索到相关内容’ -> True",
      JudgeLLM._grade_refusal("未检索到相关内容。") is True)
check("‘文档中未直接提及价格’ -> True",
      JudgeLLM._grade_refusal("文档中未直接提及价格信息。") is True)
check("‘未明确提及电池容量’ -> True",
      JudgeLLM._grade_refusal("并未在文档中明确提及电池容量。") is True)
check("‘不在提供的资料中’ -> True",
      JudgeLLM._grade_refusal("这些信息不在提供的资料中。") is True)
check("‘心跳包的协议号是0x36’ -> False（正常作答）",
      JudgeLLM._grade_refusal("心跳包的协议号是0x36。") is False)

print("== 3. 集成：冻结答案 + 当前黄金集，重算 4 条历史误判 ==")
RUN_JSON = os.path.join(ROOT, "evalkit", "runs", "answer-20260810-205518-642f26.json")
GOLDEN = os.path.join(ROOT, "evalkit", "golden", "answer.jsonl")

if not os.path.exists(RUN_JSON):
    print("  [skip] 未找到冻结回归文件 %s" % RUN_JSON)
else:
    run = json.load(open(RUN_JSON, encoding="utf-8"))
    frozen = {r["case_id"]: r for r in run.get("results", [])}

    # 解析黄金集（跳过 # 注释行）
    gold_map = {}
    with open(GOLDEN, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                d = json.loads(line)
                gold_map[d["case_id"]] = d
            except json.JSONDecodeError:
                continue

    target = ["ans-jm-002", "ans-yh-003", "ans-refuse-001", "ans-refuse-002"]
    for cid in target:
        ans = frozen.get(cid, {}).get("answer", "")
        gold = gold_map.get(cid, {})
        forb = _forbidden_hits(ans, gold.get("must_not_include", []))
        refused = JudgeLLM._grade_refusal(ans)
        should_refuse = gold.get("should_refuse", False)
        refuse_correct = refused if should_refuse else (not refused)
        check("%s 禁词应为空 (forbidden=%r)" % (cid, forb), forb == [])
        if should_refuse:
            check("%s 拒答判定应为正确 (refused=%s)" % (cid, refused), refuse_correct is True)

print("\n== 汇总 ==")
print("PASS=%d  FAIL=%d" % (PASS, FAIL))
sys.exit(1 if FAIL else 0)
