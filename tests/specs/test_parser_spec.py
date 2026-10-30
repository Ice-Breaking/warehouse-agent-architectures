#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""规格四：ReAct 输出解析协议

模型输出的文本要变成可执行的动作，解析是这个环节的唯一入口。
这份规格锁住三件事：

    1. 认得出正常格式（Final Answer / Action + Action Input）
    2. 认得出模型常见的书写变体（加粗标记、代码围栏、标量参数）
    3. 认不出时必须抛错并带上原文，绝不静默降级成「当没看见」

第 3 条最关键：静默降级会让链路悄悄跑偏，错误暴露在当场才好排查。

解析规则只有一份，放在 warehouse_data 的 parse_react_output：
反应式用类包装它产出 langchain 的动作对象，混合式直接取它返回的二元组。
下面两组断言覆盖的是同一个协议的两个使用面。
"""

import pytest
from langchain_core.agents import AgentAction, AgentFinish

import warehouse_data as wd
import warehouse_react_agent as react


def parse(text: str):
    return react.ReActOutputParser().parse(text)


# ——— 反应式：认得出 ———

def test_parser_should_return_finish_for_final_answer():
    result = parse("Thought: 信息够了\nFinal Answer: 低氧库区在三层")
    assert isinstance(result, AgentFinish)
    assert result.return_values["output"] == "低氧库区在三层"


def test_parser_should_return_action_with_json_input():
    result = parse(
        'Thought: 需要查设备\nAction: query_device_status\nAction Input: {"keyword": "断路器"}'
    )
    assert isinstance(result, AgentAction)
    assert result.tool == "query_device_status"
    assert result.tool_input == {"keyword": "断路器"}


def test_parser_should_accept_chinese_colon():
    result = parse("Thought：够了\nFinal Answer：答案")
    assert isinstance(result, AgentFinish)
    assert result.return_values["output"] == "答案"


# ——— 反应式：常见书写变体 ———

def test_parser_should_strip_bold_marks():
    """模型常把关键字写成 **Action:**"""
    result = parse(
        '**Thought:** 需要定位\n**Action:** resolve_scene_target\n'
        '**Action Input:** {"target_text": "3楼2号库房低氧区"}'
    )
    assert isinstance(result, AgentAction)
    assert result.tool == "resolve_scene_target"


def test_parser_should_strip_code_fence_around_json():
    result = parse(
        'Action: query_energy_stat\nAction Input: ```json\n{"storage_name": "全部"}\n```'
    )
    assert result.tool_input == {"storage_name": "全部"}


def test_parser_should_fall_back_to_scalar_for_single_argument():
    """单参数工具允许直接给文本，不必强制 JSON"""
    result = parse("Action: query_environment\nAction Input: 3楼2号库房")
    assert result.tool_input == "3楼2号库房"


# ——— 反应式：认不出必须抛错 ———

def test_parser_should_raise_when_neither_action_nor_answer_present():
    with pytest.raises(ValueError, match="Action"):
        parse("这个问题我需要想想")


def test_parser_should_raise_when_action_input_missing():
    with pytest.raises(ValueError, match="Action Input"):
        parse("Thought: 需要查\nAction: query_device_status")


def test_parser_should_raise_when_final_answer_is_empty():
    with pytest.raises(ValueError, match="Final Answer"):
        parse("Thought: 够了\nFinal Answer:")


def test_parser_should_raise_when_input_is_not_valid_json():
    with pytest.raises(ValueError, match="JSON"):
        parse('Action: query_device_status\nAction Input: {"keyword": ')


def test_parser_should_reject_chinese_tool_name():
    """特征：工具名正则只认 ASCII，中文工具名会被当成无法解析而抛错

    这与「工具不存在时回灌提示」是两条不同路径：前者卡在解析，后者卡在执行。
    """
    with pytest.raises(ValueError, match="Action"):
        parse("Action: 查设备\nAction Input: {}")


def test_parser_error_should_carry_the_raw_model_output():
    """抛错必须带上原文，否则没法排查是提示词问题还是模型问题"""
    raw = "模型输出了一段完全不相关的话"
    with pytest.raises(ValueError) as excinfo:
        parse(raw)
    assert raw in str(excinfo.value)


# ——— 混合式：同一套协议的函数形态 ———

def test_hybrid_parser_should_return_finish():
    kind, payload = wd.parse_react_output("Thought: 够了\nFinal Answer: 好的")
    assert kind == "finish"
    assert payload == "好的"


def test_hybrid_parser_should_return_action():
    kind, payload = wd.parse_react_output(
        'Thought: 需要查\nAction: query_device_status\nAction Input: {"keyword": "断路器"}'
    )
    assert kind == "action"
    assert payload == ("query_device_status", {"keyword": "断路器"})


def test_hybrid_parser_should_strip_bold_marks():
    kind, payload = wd.parse_react_output(
        '**Action:** resolve_scene_target\n**Action Input:** {"target_text": "低氧区"}'
    )
    assert kind == "action"
    assert payload[0] == "resolve_scene_target"


def test_hybrid_parser_should_raise_when_nothing_recognized():
    with pytest.raises(ValueError):
        wd.parse_react_output("这段话既没有动作也没有答案")


def test_hybrid_parser_should_raise_when_final_answer_is_empty():
    with pytest.raises(ValueError, match="Final Answer"):
        wd.parse_react_output("Thought: 够了\nFinal Answer:")
