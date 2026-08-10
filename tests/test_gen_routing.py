"""生成侧难度路由单元测试（零外部依赖，不连 VM / 不调 LLM）。

运行方式（项目根目录）：
  python tests/test_gen_routing.py

覆盖 _select_gen_task：
  - 硬 tenant（jm/yh）一律走 generate-hard（deepseek 优先）
  - 任意 tenant 命中技术关键词（协议号/字段/组成/优先级…）走 generate-hard
  - 普通 query（非硬 tenant + 无关键词）走默认 generate
  - 开关关闭时一律 generate
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 在导入前固定开关，保证测试可复现
os.environ["GENERATION_ROUTING_ENABLED"] = "true"
os.environ["GENERATION_HARD_TENANTS"] = "jm,yh"

from langgraph_rag_agent import _select_gen_task  # noqa: E402

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


print("== 生成侧难度路由 _select_gen_task ==")
check("硬 tenant jm → generate-hard",
      _select_gen_task("心跳包用哪个协议号", "jm") == "generate-hard")
check("硬 tenant yh → generate-hard",
      _select_gen_task("三种定位数据的上报优先级", "yh") == "generate-hard")
check("任意 tenant + 协议号关键词 → generate-hard",
      _select_gen_task("0xFF 协议包的字段定义是什么", "xx") == "generate-hard")
check("任意 tenant + 组成关键词 → generate-hard",
      _select_gen_task("通用协议包由哪几部分组成", "zz") == "generate-hard")
check("普通 query（非硬 tenant 无关键词）→ generate",
      _select_gen_task("你好", "xx") == "generate")
check("普通 factual（非硬 tenant 无关键词）→ generate",
      _select_gen_task("今天天气怎么样", "other") == "generate")

# 关闭开关
os.environ["GENERATION_ROUTING_ENABLED"] = "false"
from importlib import reload
import langgraph_rag_agent as L  # noqa: E402
reload(L)
check("开关关闭 → 一律 generate",
      L._select_gen_task("0xFF 协议包的字段定义", "jm") == "generate")

print("\n== 汇总 ==")
print("PASS=%d  FAIL=%d" % (PASS, FAIL))
sys.exit(1 if FAIL else 0)
