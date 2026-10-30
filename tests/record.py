#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""录制真实模型响应，供回放测试使用

特征测试用假客户端覆盖控制流，但假客户端测的是「我们假设模型会说什么」。
录制回放补上另一半：把真实模型的输出存下来，之后的回归跑在真实语料上。

用法::

    export DASHSCOPE_API_KEY=sk-xxx
    python tests/record.py

录完在 tests/recordings/ 下生成四个 JSON 文件。录制件随代码入库，
模型版本变了就重新录一次。
"""

import json
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
ROOT = TESTS_DIR.parent

for _path in [
    str(ROOT),
    str(ROOT / "01-reactive"),
    str(ROOT / "02-hybrid"),
    str(ROOT / "03-deliberative"),
]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

import warehouse_data as wd  # noqa: E402
import warehouse_react_agent as react  # noqa: E402
import warehouse_hybrid_agent as hybrid  # noqa: E402
import warehouse_deliberative_agent as delib  # noqa: E402

RECORDINGS = TESTS_DIR / "recordings"

QUESTION_REACTIVE = "3楼2号库房的低氧库区在什么位置"
QUESTION_HYBRID_SIMPLE = "2楼1号库房有没有防磁库区"
QUESTION_HYBRID_ANALYTICAL = "对比一下各库房的能耗情况，指出能耗最高和存在环境告警的库房"
QUESTION_DELIBERATIVE = "生成一份全楼库房的季度巡检报告"

SCENARIOS = [
    ("react_position", QUESTION_REACTIVE, lambda: react.run_react_agent(QUESTION_REACTIVE, [])),
    ("hybrid_reactive", QUESTION_HYBRID_SIMPLE, lambda: hybrid.run_hybrid_agent(QUESTION_HYBRID_SIMPLE, [])),
    (
        "hybrid_analytical",
        QUESTION_HYBRID_ANALYTICAL,
        lambda: hybrid.run_hybrid_agent(QUESTION_HYBRID_ANALYTICAL, []),
    ),
    (
        "deliberative_report",
        QUESTION_DELIBERATIVE,
        lambda: delib.run_deliberative_agent(QUESTION_DELIBERATIVE),
    ),
]


class RecordingClient:
    """透传真实调用，按顺序记下每次模型返回的原文"""

    def __init__(self, real):
        self.real = real
        self.responses = []

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        response = self.real.chat.completions.create(**kwargs)
        self.responses.append(response.choices[0].message.content or "")
        return response


def main() -> None:
    if not wd.DASHSCOPE_API_KEY:
        print("未设置 DASHSCOPE_API_KEY，跳过录制")
        return

    RECORDINGS.mkdir(parents=True, exist_ok=True)
    real = wd._build_client()

    for name, question, runner in SCENARIOS:
        recorder = RecordingClient(real)
        wd._build_client = lambda client=recorder: client

        try:
            runner()
        except Exception as exc:  # 录制中途失败也要把已录到的部分写盘
            print(f"  场景 {name} 执行异常：{exc}")

        payload = {
            "scenario": name,
            "question": question,
            "model": wd.LLM_MODEL,
            "responses": recorder.responses,
        }
        path = RECORDINGS / f"{name}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"录制 {name}：{len(recorder.responses)} 条响应 -> {path.name}")


if __name__ == "__main__":
    main()
