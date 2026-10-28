# -*- coding: utf-8 -*-
"""
improvement —— 自改进闭环（RSI 第四环）
================================================================================

把「发现 → 修好」之间站着的那个人，换成一条有闸门的自动链路。

    L1 诊断（已有 evalkit.triage R1~R8）
      -> L2 提案（generator）
      -> L3 沙箱 A/B（sandbox，双集）
      -> L4 门禁与裁决（gate，六道闸）
      -> L5 复用（store 的 archive，保留失败分支）

对外只暴露本包的四个入口，其余模块按需导入。
"""

__all__ = ["whitelist", "schema", "store", "generator", "sandbox", "gate"]

__version__ = "0.1.0"
