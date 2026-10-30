#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""规格：公共模块的复用片段

重构把三个项目间的重复实现上提到了 warehouse_data。它们现在是三个架构
共同依赖的契约，值得单独锁一层：

这些接口一旦被改动，三个项目会同时受影响。而特征层只覆盖了其中一部分——
`format_history` 与 `run_cli` 在特征测试里根本没有断言，
靠端到端去发现太晚了。
"""

import pytest

import warehouse_data as wd


# ——— 两个共用常量 ———


def test_max_rounds_should_be_six():
    """反应式分支的轮次上限，特征测试里「6 轮后抛错」的断言依赖这个值"""
    assert wd.MAX_ROUNDS == 6


def test_history_window_should_be_five():
    assert wd.HISTORY_WINDOW == 5


# ——— 环境采集 ———


def test_collect_environment_all_should_cover_every_storage():
    assert len(wd.collect_environment_all()) == 5


def test_collect_environment_all_should_return_raw_tool_output():
    """返回工具原文而不是解析后的结构，下游引用的数字才能与工具结果逐字一致"""
    joined = "\n".join(wd.collect_environment_all())
    assert "3楼/2号库房" in joined
    assert "62.8" in joined


# ——— 历史格式化 ———


def test_format_history_should_return_placeholder_when_empty():
    assert wd.format_history([]) == "无"


def test_format_history_should_keep_only_the_recent_window():
    """窗口外的历史不该进提示词，否则早期无关内容会干扰当前判断"""
    history = [(f"问{i}", f"答{i}") for i in range(8)]

    text = wd.format_history(history)

    assert "问0" not in text
    assert "问2" not in text
    assert "问3" in text
    assert "问7" in text


def test_format_history_should_label_speakers():
    assert wd.format_history([("库房在哪", "三层")]) == "用户：库房在哪\n助手：三层"


# ——— 边界预检 ———


def test_guard_scope_should_return_none_for_in_scope_question():
    assert wd.guard_scope("3楼2号库房能耗多少", "不进入工具循环") is None


def test_guard_scope_should_return_the_fixed_reply_for_out_of_scope_question():
    """返回的是 check_scope 的固定话术，不是预检节点自己另编一段"""
    question = "今天天气怎么样"

    reply = wd.guard_scope(question, "不进入工具循环")

    assert reply == wd.check_scope(question)


def test_guard_scope_should_name_the_pipeline_being_skipped(capsys):
    """label 由调用方传入，用来说明这次拦截发生在哪条链路上"""
    wd.guard_scope("今天天气怎么样", "不进入流水线")

    assert "不进入流水线" in capsys.readouterr().out


# ——— 代码围栏与动作入参 ———


def test_strip_fence_should_unwrap_fenced_block():
    assert wd.strip_fence('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_fence_should_leave_plain_text_alone():
    assert wd.strip_fence('{"a": 1}') == '{"a": 1}'


def test_parse_tool_input_should_return_dict_for_json():
    assert wd.parse_tool_input('{"storage_name": "全部"}', "原始输出") == {"storage_name": "全部"}


def test_parse_tool_input_should_fall_back_to_scalar():
    """单参数工具允许直接给文本，不必强制 JSON"""
    assert wd.parse_tool_input("3楼2号库房", "原始输出") == "3楼2号库房"


def test_parse_tool_input_should_raise_for_broken_json():
    with pytest.raises(ValueError, match="JSON"):
        wd.parse_tool_input('{"storage_name": ', "原始输出")


# ——— 交互循环 ———


def test_run_cli_should_stop_on_exit_word(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "退出")

    calls = []
    wd.run_cli("测试", lambda question: calls.append(question))

    assert calls == []


def test_run_cli_should_pass_history_to_handler(monkeypatch):
    inputs = iter(["问题1", "问题2", "退出"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    seen = []

    def handler(question, history):
        seen.append((question, list(history)))
        return f"答复-{question}"

    wd.run_cli("测试", handler, with_history=True)

    assert seen[0] == ("问题1", [])
    assert seen[1] == ("问题2", [("问题1", "答复-问题1")])


def test_run_cli_should_keep_going_when_handler_raises(monkeypatch):
    """handler 抛错不能让整个会话挂掉，用户还得能接着问下一个"""
    inputs = iter(["问题", "退出"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    def handler(question, history):
        raise RuntimeError("模型不可用")

    wd.run_cli("测试", handler, with_history=True)
