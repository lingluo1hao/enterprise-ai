"""
evalkit.judge —— LLM-as-judge 打分器
================================================================================

【为什么需要独立的 judge，而不是让生成模型自己评自己】

"自己评自己"是 RAG 评测里最危险的陷阱：模型天然倾向于给自己的输出打高分，
而且会顺着答案里的措辞自我印证，幻觉也照样拿 0.9。

所以 judge 必须用**独立模型**，且 prompt 里只看「问题 + 检索到的上下文 + 最终答案」
这三样客观材料，不让它知道"这是谁生成的"。本项目里：
  - 评分任务走网关的 `evalgrade` 路由（见 config/llm_gateway.yaml）
  - `evalgrade = [deepseek-v4-pro, local-small, local-qwen]`
    → 配了 DEEPSEEK_API_KEY 且 deepseek-v4-pro.enabled=true 时优先用云端强模型
    → 否则自动回落到本地 1.5b 小模型（零成本、可离线）
  这样"启用强 judge"对代码零侵入，只改配置。

【评哪两个维度】

  faithfulness（忠实度）：答案里每个事实性断言，是否都能在给定上下文里找到依据。
                         抓"幻觉"——凭空编造协议字段、型号、数值。0~1，越高越好。

  relevancy（相关性）：  答案是否切题、是否真正回答了问题，而不是答非所问或答一半。
                         抓"答非所问"——检索对了但生成跑偏。0~1，越高越好。

【为什么用强约束 JSON 输出】

judge 的结果要被程序直接读分数做门禁（pass_rate<0.6 算失败），
所以必须能稳定解析。prompt 强制"只输出 JSON、不要解释"，
并对非 JSON 返回做兜底抽取（找第一个 0~1 的数），避免一次格式异常把整条 case 判废。

【原则：不诱导 judge 放水】

prompt 里明确"严格按证据打分，没依据就是 0 分，不要因为答案看起来通顺就给高分"，
防止 judge 当老好人。这是评测有效性的命根子。
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

# 允许从项目根目录直接 `python -m evalkit.runner` 运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 注意：get_gateway 延迟到 JudgeLLM.__init__ 里再导入。
# 原因：llm_gateway 顶层 import redis / 连接池等重依赖，若在此处直接导入，
# evalkit 包在「仅做检索评测 / 跑报表」的轻场景下也会被迫依赖 redis，
# 既拖慢导入又让 CI 轻环境无辜报错。judge 真正用到网关时才建。


# judge 路由任务名：config/llm_gateway.yaml 里已配 [deepseek-v4-pro, local-small, local-qwen]
JUDGE_TASK = os.getenv("EVALKIT_JUDGE_TASK", "evalgrade")

_SYSTEM = (
    "你是一名严谨的 RAG 系统评审专家。你会拿到：用户问题、系统检索到的参考上下文、"
    "以及待评估的最终答案。\n"
    "你的任务是从两个维度给答案打分（0~1 浮点数，越大越好）：\n"
    "1. faithfulness（忠实度）：答案中的每个事实性断言是否都能在『参考上下文』里找到依据。"
    "凡是上下文没有、模型自己编出来的内容（虚构的字段名、型号、数值、流程），必须判低分；"
    "没依据就给 0 分，不要因为答案读起来通顺就放水。\n"
    "2. relevancy（相关性）：答案是否真正切题、完整回答了用户问题，而非答非所问、答一半或"
    "堆砌无关信息。\n\n"
    "严格遵守：\n"
    "- 只输出一个 JSON 对象，不要任何额外解释、不要 markdown 代码块标记。\n"
    "- 字段：faithfulness(数字), relevancy(数字), reason(字符串，≤40字中文说明扣分点)。\n"
    "示例：{\"faithfulness\":0.85,\"relevancy\":0.9,\"reason\":\"第3点数值上下文无出处\"}"
)

_USER_TMPL = """【用户问题】
{question}

【系统检索到的参考上下文】
{contexts}

【待评估答案】
{answer}

请按评审专家的要求，只输出 JSON。"""


def _truncate(text: str, n: int) -> str:
    text = (text or "").replace("\r", "").strip()
    return text if len(text) <= n else text[:n] + " …(截断)"


def _build_context_block(contexts: List[str], max_each: int = 1400,
                         max_total: int = 6000) -> str:
    """把若干检索片段拼成一个带编号的上下文块，超长截断。"""
    if not contexts:
        return "（无检索上下文）"
    parts: List[str] = []
    total = 0
    for i, c in enumerate(contexts[:8], 1):
        c = _truncate(c, max_each)
        seg = f"[片段{i}] {c}"
        if total + len(seg) > max_total and parts:
            parts.append(f"（上下文过长，已截断，仅展示前 {len(parts)} 段）")
            break
        parts.append(seg)
        total += len(seg)
    return "\n\n".join(parts)


def _extract_scores(text: str) -> Dict[str, Optional[float]]:
    """
    从 judge 返回里解析分数，兼容严格 JSON 与偶尔夹带文字的情况。
    返回 {"faithfulness","relevancy","reason"}，解析失败的项为 None。
    """
    out: Dict[str, Optional[float]] = {"faithfulness": None,
                                       "relevancy": None, "reason": ""}
    raw = text.strip()
    # 先尝试整段直接 json.loads
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            out["faithfulness"] = _to_score(obj.get("faithfulness"))
            out["relevancy"] = _to_score(obj.get("relevancy"))
            out["reason"] = str(obj.get("reason") or "")
            return out
    except Exception:
        pass

    # 退路 1：从文本里抠第一个 {...} 块
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                out["faithfulness"] = _to_score(obj.get("faithfulness"))
                out["relevancy"] = _to_score(obj.get("relevancy"))
                out["reason"] = str(obj.get("reason") or "")
                return out
        except Exception:
            pass

    # 退路 2：逐行抓 "faithfulness" / "relevancy" 后的数字
    for key in ("faithfulness", "relevancy"):
        km = re.search(rf"{key}\D*?([0-1](?:\.\d+)?)", raw, re.IGNORECASE)
        if km:
            out[key] = float(km.group(1))
    rm = re.search(r"reason[\"']?\s*[:：]\s*[\"']?([^\"'\n]+)", raw, re.IGNORECASE)
    if rm:
        out["reason"] = rm.group(1).strip()
    return out


def _to_score(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return max(0.0, min(1.0, f))
    except (TypeError, ValueError):
        return None


class JudgeLLM:
    """
    LLM-as-judge 打分器。

    用法::
        judge = JudgeLLM(task="evalgrade")     # 默认 evalgrade 路由
        res = judge.grade(question, answer, contexts)
        # res = {"faithfulness":0.8, "relevancy":0.9, "reason":"..."}

    若网关/模型全部不可用，grade() 返回 (None, None, "judge 不可用: ...")，
    上层据此把该 case 的 judge 维度标记为"未评分"而非直接判失败，
    保证检索硬指标依然有效（评测不被 judge 故障绑架）。
    """

    def __init__(self, task: str = JUDGE_TASK, timeout: float = 90.0):
        self.task = task
        self.timeout = timeout
        self._gw = None
        self.model = "?"
        try:
            from llm_gateway import get_gateway
            self._gw = get_gateway()
            chain = self._gw.resolve_chain(self.task)
            self.model = " > ".join(chain) if chain else "?"
        except Exception as e:  # 网关初始化失败（如配置缺失 / redis 未装）
            self._gw = None
            self._init_error = f"{type(e).__name__}: {e}"

    @property
    def available(self) -> bool:
        return self._gw is not None

    def grade(self, question: str, answer: str,
              contexts: List[str]) -> Dict[str, Any]:
        """
        对一条答案打分。返回 {"faithfulness","relevancy","reason","error"}。
        error 非空表示 judge 没跑成功，分数均为 None。
        """
        if self._gw is None:
            return {"faithfulness": None, "relevancy": None,
                    "reason": "", "error": getattr(self, "_init_error", "judge 未初始化")}

        user_p = _USER_TMPL.format(
            question=_truncate(question, 600),
            contexts=_build_context_block(contexts),
            answer=_truncate(answer, 3000),
        )
        try:
            resp = self._gw.chat_detailed(_SYSTEM, user_p, task=self.task)
            scores = _extract_scores(resp.text)
            scores["error"] = ""
            return scores
        except Exception as e:
            return {"faithfulness": None, "relevancy": None,
                    "reason": "", "error": f"{type(e).__name__}: {e}"}

    @staticmethod
    def _grade_refusal(answer: str) -> bool:
        """判定答案是否表现为拒答（模块级可测，不依赖网关）。

        双路：窄关键词兜底 + 语义正则（覆盖 未提及/未涉及/未检索到/不涉及/
        不在资料中 等多样拒答措辞），避免窄表漏判。
        """
        answer = answer or ""
        refusal_markers = ["不知道", "无法回答", "没有相关信息", "知识库中没有",
                           "无法确定", "缺乏", "未提供", "我无法", "没有找到",
                           "抱歉，", "不在我的知识范围内"]
        if any(m in answer for m in refusal_markers):
            return True
        refusal_patterns = [
            r"未检索到", r"未直接提及", r"未提及", r"未包含", r"未涉及", r"未找到",
            r"没有提供", r"无相关", r"没有提及", r"没有信息", r"没有相关",
            r"不涉及", r"无法找到", r"查不到", r"文档未", r"资料未", r"知识库未",
            r"不在.*(资料|文档|知识库|提供)",
            # 通用兜底：覆盖「未直接提供 / 未明确提及 / 未直接提及」等变体
            r"未.{0,6}(提及|提供|包含|涉及|检索到|找到)",
        ]
        return any(re.search(p, answer) for p in refusal_patterns)

    def grade_refusal(self, answer: str) -> Dict[str, Any]:
        """
        抽取"答案是否表现为拒答"。用于 should_refuse 类 case 的规则判定辅助。
        窄关键词 + 语义正则双路，避免窄表漏判（如"文档未提及"被判成未拒答）。
        """
        return {"refused": self._grade_refusal(answer)}
