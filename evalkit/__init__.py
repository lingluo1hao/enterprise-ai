"""
evalkit —— RAG 评测 Harness 与 Bad Case 闭环工具包
================================================================================

【这个包解决什么问题】

改 RAG 系统时最容易犯的错：凭感觉调参。
把 top_k 从 4 改成 8、把 rerank 打开、换个 embedding 模型 —— 然后随手问两个问题，
"嗯，好像变好了"，就合入了。下次再改，又"好像变好了"。
半年后没人知道系统到底是变强还是变弱了，也没人敢动核心链路。

evalkit 提供一把尺子：
  1. 固定一批带标准答案的题目（黄金集 golden set）
  2. 每次改动后自动跑一遍，输出可比较的量化指标
  3. 指标退化就报红灯，改动别合

【为什么叫 evalkit 而不是 eval】
`eval` 是 Python 内置函数名，用作包名虽然合法，但容易和内置函数混淆，
静态检查工具也常报警告。加 kit 后缀规避。

【两层 harness，成本差两个数量级】

  ┌─ 检索 harness（harness_retrieval.py）
  │    只跑向量库检索，不调用任何 LLM
  │    指标：Recall@k / MRR / nDCG@k / bury
  │    成本：零；速度：几十条 case 秒级跑完
  │    用途：每次改检索参数都跑，当单元测试用
  │
  └─ 答案 harness（harness_answer.py）
       要跑完整问答链路 + 调 LLM judge 打分
       指标：faithfulness（忠实度）/ relevancy（相关性）/ 拒答正确率
       成本：每条 case 若干次 LLM 调用；速度：分钟级
       用途：发版前跑，或检索指标动了之后验证端到端效果

先看检索层，是因为 RAG 的失败绝大多数发生在检索：
答案错，通常不是模型笨，而是根本没把正确的段落喂给它。
检索指标能在不花一分钱的情况下定位这类问题。

【模块职责】
  schema.py             数据结构：黄金集 case、评测结果、run 记录，以及 jsonl 读写
  harness_retrieval.py  检索层评测（Recall/MRR/nDCG/bury）
  harness_answer.py     答案层评测（走完整链路 + judge）
  judge.py              LLM-as-judge 打分器（用独立模型，避免自己评自己）
  triage.py             Bad case 自动根因分类（R1~R8）
  report.py             生成自包含 HTML 报表
  runner.py             命令行入口
  golden/               黄金集数据（jsonl）
"""

__all__ = ["schema", "harness_retrieval", "harness_answer", "judge",
           "triage", "report", "runner"]
