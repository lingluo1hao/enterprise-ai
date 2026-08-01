"""
================================================================================
 skill_framework.py — 协议无关的 Skill 内核（企业级 RAG Agent 共享层）
================================================================================

 设计目标
 --------
 把"可被 LLM 调用的能力（Skill）"从具体运行形态里抽出来：
   - 既能被 in-process 的 ReAct / Planning Agent 直接调用
   - 也能被 MCP Server 包装成标准 MCP Tool，供任意 MCP 客户端复用

 本模块只依赖 Python 标准库（re / ast / operator），不引入 chromadb /
 ollama 等重依赖，因此可以被 MCP Server、单元测试、其它客户端安全导入。

 安全机制（与全息安全加固一致）
 ------------------------------
   - safe_eval()      : AST 白名单求值，彻底替代 eval()
   - BaseSkill.validate_params() : 参数三层校验（非空 / 长度 / 危险模式）
   - CalculatorSkill  : 复用 safe_eval，做到"输入即代码"也安全
================================================================================
"""

import re
import ast
import operator as op
from abc import ABC, abstractmethod
from typing import Dict, Optional

# ============================================================================
# 第一部分：安全数学表达式求值（替代 eval，杜绝任意代码执行）
# ============================================================================

_MAX_EXPR_LEN = 500        # 表达式最大长度
_MAX_RESULT_ABS = 1e308    # 结果最大绝对值（防止溢出）
_MAX_NODES = 100           # AST 节点数上限，防止递归炸弹

# 只允许的运算操作符
_SAFE_BINOPS = {
    ast.Add:      op.add,
    ast.Sub:      op.sub,
    ast.Mult:     op.mul,
    ast.Div:      op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod:      op.mod,
    ast.Pow:      op.pow,
}
_SAFE_UNARYOPS = {
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _eval_ast_node(node: ast.AST, depth: int = 0) -> float:
    """递归安全求值 AST 节点 — 仅允许数字、二元/一元运算。"""
    if depth > 50:
        raise ValueError("表达式嵌套层级过深")

    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body, depth)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"不支持的常量类型: {type(node.value).__name__}")

    if isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left, depth + 1)
        right = _eval_ast_node(node.right, depth + 1)
        optype = type(node.op)
        if optype in _SAFE_BINOPS:
            result = _SAFE_BINOPS[optype](left, right)
            if abs(result) > _MAX_RESULT_ABS:
                raise ValueError("计算结果溢出")
            return result
        raise ValueError(f"不支持的运算符: {optype.__name__}")

    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast_node(node.operand, depth + 1)
        optype = type(node.op)
        if optype in _SAFE_UNARYOPS:
            return _SAFE_UNARYOPS[optype](operand)
        raise ValueError(f"不支持的一元运算符: {optype.__name__}")

    raise ValueError(f"表达式包含不支持的语法: {type(node).__name__}")


def safe_eval(expr: str) -> float:
    """
    白名单 AST 安全求值数学表达式。

    仅允许：数字(int/float)、+ - * / // % **、括号、正负号。
    彻底禁止：函数调用、变量名、属性访问、import 等任何可执行代码。
    """
    if not expr or not expr.strip():
        raise ValueError("空表达式")
    if len(expr) > _MAX_EXPR_LEN:
        raise ValueError("表达式过长")

    # 先 AST 解析（仅 mode='eval'，禁止语句和多行）
    try:
        tree = ast.parse(expr.strip(), mode='eval')
    except SyntaxError as e:
        raise ValueError(f"表达式语法错误: {e}") from e

    # 节点数上限（防递归炸弹）
    node_count = sum(1 for _ in ast.walk(tree))
    if node_count > _MAX_NODES:
        raise ValueError("表达式节点数过多")

    return _eval_ast_node(tree, depth=0)


# ============================================================================
# 第二部分：Skill 抽象基类 + 工具沙箱钩子
# ============================================================================

class BaseSkill(ABC):
    """
    Skill 抽象基类

    每个 Skill 需要声明：
      - name:        技能名称（Agent / MCP 客户端用这个名字来调用）
      - description: 技能描述（Agent / LLM 根据描述决定是否使用该技能）

    安全机制：
      - validate_params()  子类覆写以实现参数白名单校验
      - execute()          先调 validate_params，通过后才执行
    """

    name: str = "base_skill"
    description: str = "基础技能"

    # 参数限制（子类覆写）
    MAX_QUERY_LEN = 2000  # 默认最大输入长度

    def validate_params(self, query: str) -> Optional[str]:
        """
        参数白名单校验。返回 None 表示通过，否则返回错误描述。

        子类应覆写此方法以实现特定校验逻辑。
        默认检查：非空 + 长度限制 + 危险模式。
        """
        if not query or not query.strip():
            return f"[{self.name}] 参数不能为空"
        if len(query) > self.MAX_QUERY_LEN:
            return f"[{self.name}] 参数过长（最大 {self.MAX_QUERY_LEN} 字符）"
        lower = query.lower()
        dangerous = ["__import__", "exec(", "eval(", "os.system", "subprocess",
                     "open(", "compile(", "globals(", "locals(", "getattr("]
        for pattern in dangerous:
            if pattern in lower:
                return f"[{self.name}] 参数包含不被允许的字符模式"
        return None

    @abstractmethod
    def execute(self, query: str) -> str:
        """
        执行技能，返回结果文本

        :param query: 传入技能的参数（搜索关键词或计算表达式）
        :return: 技能执行结果
        """
        pass


class CalculatorSkill(BaseSkill):
    """
    计算器技能 — 执行数学运算

    当 Agent 需要计算数值时调用此技能。
    例如："待机时间120小时换算成天" → 120 / 24 = 5天
    """

    name = "calculator"
    description = (
        "执行数学计算。适用于需要数值运算、单位换算等场景。"
        "输入数学表达式（如 120/24 或 5*24），返回计算结果。"
    )

    MAX_QUERY_LEN = 300  # 数学表达式不宜过长

    def execute(self, query: str) -> str:
        # 参数安全校验
        err = self.validate_params(query)
        if err:
            return f"计算失败：{err}"

        try:
            # 清理输入：只保留数字和运算符
            expr = re.sub(r'[^0-9+\-*/.()\s]', '', query).strip()
            if not expr:
                return "计算失败：无法识别有效的数学表达式"

            # 安全计算：使用 AST 白名单求值器，杜绝 eval() 风险
            result = safe_eval(expr)
            # 格式化输出：整数无小数点，浮点数保留 4 位
            if isinstance(result, float) and result == int(result):
                display = str(int(result))
            else:
                display = f"{result:.4f}".rstrip('0').rstrip('.')
            return f"计算结果：{expr} = {display}"
        except Exception as e:
            return f"计算失败：{query}，错误：{e}"


# ============================================================================
# 第三部分：Skill 注册表（协议无关的"内存版 server"）
# ============================================================================

class SkillRegistry:
    """
    Skill 注册表 — 管理所有可用的 Skill

    通过 registry.get_skill(name) 获取和调用技能。
    新增技能只需 registry.register(MySkill()) 即可。
    """

    def __init__(self):
        self._skills: Dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill):
        """注册一个 Skill"""
        self._skills[skill.name] = skill

    def get_skill(self, name: str) -> Optional[BaseSkill]:
        """按名称获取 Skill（含前缀模糊匹配）"""
        name = (name or "").strip().lower()
        if name in self._skills:
            return self._skills[name]
        for key in self._skills:
            if key in name or name in key:
                return self._skills[key]
        return None

    def get_all_descriptions(self) -> str:
        """返回所有技能的描述（供 Agent 决策时参考）"""
        lines = []
        for name, skill in self._skills.items():
            lines.append(f"  - {name}: {skill.description}")
        return "\n".join(lines)

    def list_skills(self) -> list:
        """返回技能清单（名称 + 描述），用于 MCP resource 暴露"""
        return [{"name": n, "description": s.description}
                for n, s in self._skills.items()]
