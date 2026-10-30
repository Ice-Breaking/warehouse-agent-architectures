#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""pytest 共享配置

做三件事：

1. 把仓库根目录与三个项目目录加入 sys.path，测试可以直接 import 四个模块，
   不必依赖运行目录。
2. 提供 llm fixture，用假客户端替换模型入口，让控制流可以断言。
3. 提供录制件的存取与「没有 API Key 就跳过」的策略。
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

TESTS_DIR = Path(__file__).resolve().parent
ROOT = TESTS_DIR.parent
PROJECT_DIRS = ["01-reactive", "02-hybrid", "03-deliberative"]

# conftest 自身的目录不一定在 sys.path 上，先补进来才能 import fakes
for _path in [str(TESTS_DIR), str(ROOT), *[str(ROOT / d) for d in PROJECT_DIRS]]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

import warehouse_data as wd  # noqa: E402  依赖上面的 sys.path 引导
from fakes import FakeClient  # noqa: E402

RECORDINGS_DIR = TESTS_DIR / "recordings"

# 三个项目当前的入口模块名
AGENT_MODULES = [
    "warehouse_react_agent",
    "warehouse_hybrid_agent",
    "warehouse_deliberative_agent",
]


def _client_providers() -> List[Any]:
    """需要替换 _build_client 的模块集合

    三个项目在重构前用 from warehouse_data import _build_client，函数对象在导入时
    就绑定到了各自的模块命名空间，只替换 warehouse_data 上的引用对它们无效。
    所以三处都要替换；raising=False 保证重构后属性消失也不会报错。
    """
    providers: List[Any] = [wd]

    for name in AGENT_MODULES:
        module = sys.modules.get(name)
        if module is None:
            try:
                module = __import__(name)
            except ImportError:
                continue
        if hasattr(module, "_build_client"):
            providers.append(module)

    return providers


@pytest.fixture
def llm(monkeypatch):
    """安装假模型客户端，返回安装函数

    用法::

        client = llm(["第一条响应", "第二条响应"])

    client.call_count 是模型被调用的次数，client.requests 是每次收到的 messages。
    """
    def install(script: List[str]) -> FakeClient:
        client = FakeClient(script)
        for module in _client_providers():
            monkeypatch.setattr(module, "_build_client", lambda: client, raising=False)
        return client

    return install


def has_api_key() -> bool:
    """是否具备真实调用模型的条件"""
    return bool(os.getenv("DASHSCOPE_API_KEY"))


def load_recording(name: str) -> Dict[str, Any]:
    """读取一份录制件

    录制件是 record.py 真实调用模型时逐步存下的原文，供回放测试使用。
    """
    path = RECORDINGS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少录制件 {path}，先运行 python tests/record.py 录制一次"
        )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def recording():
    """读取录制件，返回按名字取用的函数"""
    return load_recording
