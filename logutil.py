"""全局给 print() 自动加时间戳前缀。

放在入口文件最早 import 即可对整个进程生效（含被 import 的模块）。
效果：原本 `[docs/upload] 开始接收文件 ...`
      变为 `[2026-08-08 00:31:58] [docs/upload] 开始接收文件 ...`

实现：包一层 builtins.print，原样透传 sep/end/file/flush 等参数，
只在最前面拼一个 [YYYY-MM-DD HH:MM:SS]。
"""

import builtins
from datetime import datetime

_ORIG_PRINT = builtins.print


def _ts_print(*args, **kwargs):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if args:
        # 把时间戳拼到第一个参数的最前面，其余参数原样跟在后面
        _ORIG_PRINT(f"[{ts}] {args[0]}", *args[1:], **kwargs)
    else:
        # 纯 kwargs 的空打印（如 print(end="")）不加戳，避免凭空多时间戳
        _ORIG_PRINT(**kwargs)


builtins.print = _ts_print
