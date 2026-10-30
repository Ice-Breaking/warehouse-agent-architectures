#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""特征：深思熟虑的五阶段流水线

这条链路的控制流完全由代码写死，模型只在每个阶段内部填内容。
锁定的是：阶段数量、模型调用次数、阶段之间的数据传递、
以及感知阶段必须是真实采集而不是模型生成。
"""

import pytest

import warehouse_deliberative_agent as delib


def empty_state() -> delib.DelibState:
    return {
        "question": "生成一份全楼库房的季度巡检报告",
        "perception": "",
        "risk_model": "",
        "plans": "",
        "decision": "",
        "report": "",
    }


def test_perception_should_collect_real_data_without_calling_model(llm):
    """第一阶段真调工具取数，模型不参与，这是整条链路可信度的基础"""
    client = llm([])

    state = delib.perception(empty_state())

    assert client.call_count == 0
    assert "【设备台账】" in state["perception"]
    assert "【能耗数据】" in state["perception"]
    assert "【环境数据】" in state["perception"]
    assert "3楼/2号库房" in state["perception"]
    assert "42.6" in state["perception"]


def test_deliberative_should_call_model_four_times(llm):
    """五阶段里感知不调模型，建模、推理、决策、成文各一次"""
    client = llm(["风险矩阵", "三套方案", "决策结论", "报告正文"])

    report = delib.run_deliberative_agent("生成一份全楼库房的季度巡检报告")

    assert report == "报告正文"
    assert client.call_count == 4


def test_deliberative_should_pass_each_stage_output_to_the_next(llm):
    """上一阶段的产出必须进入下一阶段的提示词，断了就是漏传"""
    client = llm(["风险矩阵内容", "三套方案内容", "决策结论内容", "报告正文"])

    delib.run_deliberative_agent("生成巡检报告")

    prompts = [messages[-1]["content"] for messages in client.requests]
    assert "风险矩阵内容" in prompts[1]
    assert "三套方案内容" in prompts[2]
    assert "决策结论内容" in prompts[3]


def test_deliberative_should_not_call_model_for_out_of_scope_question(llm):
    client = llm([])

    reply = delib.run_deliberative_agent("今天天气怎么样")

    assert reply is not None
    assert client.call_count == 0


def test_deliberative_should_produce_empty_report_when_every_stage_returns_empty(llm):
    """特征：四阶段全返回空时，当前实现产出空报告而不报错

    这是已知的薄弱点，如实记录。若后续改成显式抛错，这条测试会失败并提醒确认。
    """
    llm(["", "", "", ""])

    report = delib.run_deliberative_agent("生成巡检报告")

    assert report == ""
