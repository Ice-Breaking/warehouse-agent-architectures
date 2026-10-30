#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""特征：反应式的执行循环

这一组记录的是「工具循环现在怎么走」，用脚本化的假模型驱动，
不联网、不花钱、每次结果一致。

锁定的是对外契约：返回值、模型调用次数、观察是否回灌、轮次上限是否生效、
越界问题是否根本不进模型。终端输出文案不在锁定范围内——重构会改提示词。
"""

import pytest

import warehouse_react_agent as react


def test_react_should_answer_after_one_tool_call(llm):
    """第一轮调工具、第二轮给答案，是最常见的两轮闭环"""
    client = llm(
        [
            'Thought: 需要先定位\nAction: resolve_scene_target\nAction Input: {"target_text": "3楼2号库房低氧区"}',
            "Thought: 拿到了\nFinal Answer: 低氧库区在档案综合楼三层2号库房",
        ]
    )

    answer = react.run_react_agent("3楼2号库房的低氧区在哪个位置", [])

    assert answer == "低氧库区在档案综合楼三层2号库房"
    assert client.call_count == 2


def test_react_should_answer_directly_when_no_tool_needed(llm):
    client = llm(["Thought: 可以直接回答\nFinal Answer: 直接答复"])

    answer = react.run_react_agent("3楼2号库房的低氧区在哪", [])

    assert answer == "直接答复"
    assert client.call_count == 1


def test_react_should_feed_observation_back_into_the_next_round(llm):
    """工具返回值必须回灌进下一轮提示词，否则模型看不到结果"""
    client = llm(
        [
            'Action: resolve_scene_target\nAction Input: {"target_text": "3楼2号库房低氧区"}',
            "Final Answer: 找到了",
        ]
    )

    react.run_react_agent("低氧区在哪", [])

    second_prompt = client.requests[1][0]["content"]
    assert "Observation" in second_prompt
    assert "3-2-LO-01" in second_prompt


def test_react_should_report_unknown_tool_and_let_model_recover(llm):
    """工具名不在注册表里时回灌提示而不是抛错，模型有机会自纠"""
    client = llm(
        [
            "Action: not_a_real_tool\nAction Input: {}",
            "Final Answer: 换个工具再试",
        ]
    )

    answer = react.run_react_agent("查点东西", [])

    assert answer == "换个工具再试"
    assert "不存在" in client.requests[1][0]["content"]


def test_react_should_stop_at_the_iteration_limit(llm):
    """循环必须有上限兜底，不能无限转"""
    client = llm(
        ['Thought: 还要查\nAction: query_environment\nAction Input: {"storage_name": "3楼2号库房"}'] * 6
    )

    with pytest.raises(RuntimeError, match="超过最大工具调用轮次"):
        react.run_react_agent("3楼2号库房环境怎么样", [])

    assert client.call_count == 6


def test_react_should_not_call_model_for_out_of_scope_question(llm):
    """越界话题在入口就被拦下，一次模型调用都不该发生"""
    client = llm([])

    reply = react.run_react_agent("这份档案正文写了什么", [])

    assert reply is not None
    assert client.call_count == 0


def test_react_should_raise_when_model_output_is_unparseable(llm):
    """解析不出动作也不静默跳过，直接抛错暴露提示词或模型问题"""
    llm(["这段话既没有动作也没有答案"])

    with pytest.raises(ValueError):
        react.run_react_agent("3楼2号库房的低氧区在哪", [])
