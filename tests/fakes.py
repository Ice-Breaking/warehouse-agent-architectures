#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试用的假模型客户端

真实链路调用 qwen 模型，同一个问题每次返回的文本都不同，控制流没法断言。
这里提供一个按脚本返回文本的假客户端，让测试能精确控制模型「说什么」，
从而验证分支走向、循环终止与错误分支。

用法:

    client = FakeClient(["Thought: 先查一下", "Final Answer: 查到了"])
    # 安装后，模型每次被调用就依次返回脚本里的下一条文本
"""

from types import SimpleNamespace
from typing import Any, List, Optional


class FakeClient:
    """脚本化的假模型客户端

    script 是响应文本队列，每次 create 弹出一条。队列用尽时抛 AssertionError，
    这样「实际调用次数超出脚本」会立刻暴露，而不是静默返回空串把问题藏起来。

    requests 记录每次收到的 messages，供测试断言提示词内容。
    call_count 记录调用次数，供测试断言「模型被调了几次」。
    """

    def __init__(self, script: Optional[List[str]] = None) -> None:
        self.script: List[str] = list(script or [])
        self.requests: List[Any] = []
        self.call_count: int = 0

    # openai 客户端的调用形状是 client.chat.completions.create(...)，
    # 用两个属性把这条链缩短到自己身上，避免为测试引入真实的 SDK 对象。
    @property
    def chat(self) -> "FakeClient":
        return self

    @property
    def completions(self) -> "FakeClient":
        return self

    def create(self, **kwargs: Any) -> Any:
        """按脚本返回下一条响应，并记录本次收到的 messages"""
        self.call_count += 1
        self.requests.append(kwargs.get("messages"))

        if not self.script:
            raise AssertionError(
                f"假客户端的响应脚本已用尽：第 {self.call_count} 次调用没有预设文本。"
                "通常是实际调用次数超出测试预期，检查链路或补充脚本。"
            )

        return _response(self.script.pop(0))


def _response(content: str) -> Any:
    """构造带 choices[0].message.content 的轻量响应对象"""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
