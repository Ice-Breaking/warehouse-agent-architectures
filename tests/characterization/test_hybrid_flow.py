#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""特征：混合式的两条分支

协调层只跑一次，之后路径就定了。这一组锁定：

    - 反应式分支保留模型自主循环，答复来自模型的最终答案
    - 深思熟虑分支是收集到分析到建议三段，答复来自建议节点而非分析节点
    - 优先级不参与分支选择
    - 越界问题不进图，一次模型调用都不发生
"""

import pytest

import warehouse_hybrid_agent as hybrid

ASSESS_REACTIVE = '{"query_type": "informational", "processing_mode": "reactive", "priority": "normal"}'
ASSESS_EMERGENCY = '{"query_type": "emergency", "processing_mode": "reactive", "priority": "high"}'
ASSESS_ANALYTICAL = '{"query_type": "analytical", "processing_mode": "deliberative", "priority": "normal"}'


def test_hybrid_should_take_reactive_branch_and_use_model_answer(llm):
    client = llm([ASSESS_REACTIVE, "Final Answer: 低氧库区在三层"])

    answer = hybrid.run_hybrid_agent("3楼2号库房的低氧区在哪", [])

    assert answer == "低氧库区在三层"
    assert client.call_count == 2


def test_hybrid_should_loop_tools_within_the_reactive_branch(llm):
    """反应式分支内部仍然是模型主导的多轮循环"""
    client = llm(
        [
            ASSESS_REACTIVE,
            'Action: resolve_scene_target\nAction Input: {"target_text": "3楼2号库房低氧区"}',
            "Final Answer: 低氧库区在三层，有两组密集架",
        ]
    )

    answer = hybrid.run_hybrid_agent("3楼2号库房的低氧区在哪", [])

    assert answer == "低氧库区在三层，有两组密集架"
    assert client.call_count == 3


def test_hybrid_should_chain_three_deliberative_steps(llm, capsys):
    """深思熟虑分支必须走满三段，顺序是收集、分析、建议"""
    client = llm([ASSESS_ANALYTICAL, "分析结论：3楼2号库房能耗最高", "建议：检查除湿机"])

    hybrid.run_hybrid_agent("对比一下各库房的能耗情况", [])

    output = capsys.readouterr().out
    assert output.index("[深度分析]") < output.index("[生成建议]")
    assert client.call_count == 3


def test_hybrid_should_answer_with_recommendation_not_analysis(llm):
    """答复出口取建议节点，分析结论不能直接当答复"""
    client = llm([ASSESS_ANALYTICAL, "分析结论：能耗最高是3楼2号库房", "建议：先查除湿机"])

    answer = hybrid.run_hybrid_agent("对比一下各库房的能耗情况", [])

    assert answer == "建议：先查除湿机"
    assert "分析结论" not in answer


def test_hybrid_should_pass_priority_into_the_recommendation_prompt(llm):
    """优先级不参与分支，但要传进建议提示词影响语气"""
    client = llm([ASSESS_ANALYTICAL, "分析结论", "建议：立即处置"])

    hybrid.run_hybrid_agent("对比各库房能耗", [])

    assert "high" in client.requests[-1][-1]["content"]


def test_hybrid_should_raise_when_coordinator_output_is_invalid(llm):
    """路由分错比报错更危险，所以非法输出必须抛错而不是默认走反应式"""
    llm(["我认为这个问题应该走反应式"])

    with pytest.raises(ValueError):
        hybrid.run_hybrid_agent("对比一下各库房的能耗", [])


def test_hybrid_should_not_call_model_for_out_of_scope_question(llm):
    client = llm([])

    reply = hybrid.run_hybrid_agent("今天天气怎么样", [])

    assert reply is not None
    assert client.call_count == 0


def test_hybrid_should_stop_reactive_branch_at_step_limit(llm):
    """反应式分支同样需要轮次上限"""
    client = llm(
        [ASSESS_REACTIVE]
        + ['Action: query_environment\nAction Input: {"storage_name": "3楼2号库房"}'] * 6
    )

    with pytest.raises(RuntimeError):
        hybrid.run_hybrid_agent("3楼2号库房环境怎么样", [])
