#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""回放：用真实语料做端到端回归

假客户端测的是「我们假设模型会说什么」，回放测的是「真实模型说过什么」。
录制件由 tests/record.py 采集并随代码入库，回放时不联网、不消耗额度。

断言只针对结构事实——不抛异常、调用次数、产出里出现真实采集到的数据。
不做逐字比对，因为同一个问题模型每次的措辞都不同，逐字比对只会得到脆弱的测试。
"""

import pytest
from typing import Any

import warehouse_deliberative_agent as delib
import warehouse_hybrid_agent as hybrid
import warehouse_react_agent as react

pytestmark = pytest.mark.recorded


def require(recording, name: str) -> dict[str, Any]:
    """取录制件，缺件时跳过而不是失败——录制需要 API Key，不是所有人都具备"""
    try:
        return recording(name)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def test_react_replay_should_locate_the_low_oxygen_zone(llm, recording):
    data = require(recording, "react_position")
    client = llm(data["responses"])

    answer = react.run_react_agent(data["question"], [])

    assert "低氧库区" in answer
    assert "3-2-LO-01" in answer
    assert client.call_count == len(data["responses"])


def test_hybrid_replay_should_answer_simple_query_through_reactive_branch(llm, recording):
    data = require(recording, "hybrid_reactive")
    client = llm(data["responses"])

    answer = hybrid.run_hybrid_agent(data["question"], [])

    assert "防磁库区" in answer
    assert client.call_count == len(data["responses"])


def test_hybrid_replay_should_cite_real_collected_numbers(llm, recording):
    """回放的价值在这里：验证真实模型的产出确实引用了工具采到的数字"""
    data = require(recording, "hybrid_analytical")
    client = llm(data["responses"])

    answer = hybrid.run_hybrid_agent(data["question"], [])

    assert client.call_count == 3
    assert "3楼/2号库房" in answer

    # 采集到的真实数字必须出现在产出链路里，而不是模型自己编的
    analysis = client.requests[2][-1]["content"]
    assert "122.9" in analysis or "42.6" in analysis


def test_deliberative_replay_should_produce_all_four_sections(llm, recording):
    data = require(recording, "deliberative_report")
    client = llm(data["responses"])

    report = delib.run_deliberative_agent(data["question"])

    assert client.call_count == 4
    for section in ("巡检概况", "风险矩阵", "整改方案", "后续跟踪"):
        assert section in report


def test_deliberative_replay_should_build_on_real_perception(llm, recording):
    """报告里的每个数字都应来自感知阶段的真实采集"""
    data = require(recording, "deliberative_report")
    client = llm(data["responses"])

    report = delib.run_deliberative_agent(data["question"])

    assert "3楼/2号库房" in report
    assert "62.8" in report
    assert len(report) > 500
