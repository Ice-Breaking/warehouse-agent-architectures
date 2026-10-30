#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""规格五：混合式协调层的三元组校验

协调层每次输出必须同时给出三项：查询类型、处理模式、优先级。
这份规格锁住两件事：

    1. 认得出模型常见的输出变体（代码围栏、前后缀说明文字）
    2. 取值不合法时抛错，不设默认值兜底

第 2 条是刻意的选择：路由分错比报错更危险。默认走反应式会让本该
深入分析的请求草草收场，且这种偏差很难被察觉。
"""

import pytest

import warehouse_hybrid_agent as hybrid


def assess(raw: str):
    return hybrid._parse_assess(raw)


# ——— 正常解析 ———

def test_assess_should_parse_plain_json():
    assert assess(
        '{"query_type": "analytical", "processing_mode": "deliberative", "priority": "high"}'
    ) == ("analytical", "deliberative", "high")


def test_assess_should_parse_emergency_triple():
    assert assess(
        '{"query_type": "emergency", "processing_mode": "reactive", "priority": "high"}'
    ) == ("emergency", "reactive", "high")


def test_assess_should_parse_informational_triple():
    assert assess(
        '{"query_type": "informational", "processing_mode": "reactive", "priority": "normal"}'
    ) == ("informational", "reactive", "normal")


# ——— 输出变体 ———

def test_assess_should_strip_json_code_fence():
    assert assess(
        '```json\n{"query_type": "analytical", "processing_mode": "deliberative", "priority": "normal"}\n```'
    ) == ("analytical", "deliberative", "normal")


def test_assess_should_tolerate_surrounding_text():
    """模型常在 JSON 前后加说明，只要花括号内的内容合法就应当解析成功"""
    assert assess(
        '我的判断如下：\n{"query_type": "informational", "processing_mode": "reactive", "priority": "normal"}\n请确认。'
    ) == ("informational", "reactive", "normal")


def test_assess_should_accept_arbitrary_key_order():
    assert assess(
        '{"priority": "normal", "processing_mode": "reactive", "query_type": "informational"}'
    ) == ("informational", "reactive", "normal")


# ——— 非法取值必须抛错 ———

def test_assess_should_raise_for_unknown_query_type():
    with pytest.raises(ValueError, match="query_type"):
        assess(
            '{"query_type": "随便填的", "processing_mode": "reactive", "priority": "normal"}'
        )


def test_assess_should_raise_for_unknown_processing_mode():
    with pytest.raises(ValueError, match="processing_mode"):
        assess(
            '{"query_type": "analytical", "processing_mode": "随便填的", "priority": "normal"}'
        )


def test_assess_should_raise_for_unknown_priority():
    with pytest.raises(ValueError, match="priority"):
        assess(
            '{"query_type": "analytical", "processing_mode": "deliberative", "priority": "紧急"}'
        )


def test_assess_should_raise_when_field_missing():
    """缺字段不能补默认值，必须暴露"""
    with pytest.raises(ValueError):
        assess('{"query_type": "analytical", "processing_mode": "deliberative"}')


def test_assess_should_raise_for_non_json_output():
    with pytest.raises(ValueError, match="JSON"):
        assess("我认为这个问题应该走反应式")


def test_assess_should_raise_for_broken_json():
    with pytest.raises(ValueError, match="JSON"):
        assess('{"query_type": "analytical", "processing_mode": ')


def test_assess_error_should_carry_the_raw_output():
    raw = "模型输出了一段不是 JSON 的话"
    with pytest.raises(ValueError) as excinfo:
        assess(raw)
    assert raw in str(excinfo.value)


# ——— 分支只认处理模式 ———

def test_pick_route_should_follow_processing_mode():
    """优先级不参与分支选择，避免与处理模式打架"""
    assert hybrid.pick_route({"processing_mode": "reactive", "priority": "high"}) == "reactive"
    assert hybrid.pick_route({"processing_mode": "deliberative", "priority": "high"}) == "deliberative"
