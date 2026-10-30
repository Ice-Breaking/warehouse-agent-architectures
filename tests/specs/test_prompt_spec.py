#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""规格七：提示词与调用参数的对应关系

提示词模板里的每个占位符都必须在调用处被填上，反之亦然。
两者一旦不同步，轻则 KeyError，重则某个变量静默丢失——比如边界条款没被拼进提示词，
模型就失去了约束却不报错。

这份规格用两步锁住：
    1. 占位符集合等于期望集合（改动提示词时强制人确认）
    2. 用占位值 format 一遍不报错（自洽性自动校验）
"""

from string import Formatter

import warehouse_deliberative_agent as delib
import warehouse_hybrid_agent as hybrid
import warehouse_react_agent as react


def placeholders(template: str) -> set:
    """取出模板里的全部变量名，自动跳过 {{ }} 这类转义花括号"""
    return {name for _, name, _, _ in Formatter().parse(template) if name}


def assert_renderable(template: str) -> None:
    """用占位值渲染一遍，占位符与传参不一致时会立刻抛 KeyError"""
    names = placeholders(template)
    assert names, "模板没有任何占位符，检查是否取错了对象"
    template.format(**{name: "占位值" for name in names})


# ——— 反应式 ———

def test_react_prompt_should_declare_six_variables():
    assert set(react.REACT_PROMPT.input_variables) == {
        "tool_descriptions",
        "tool_names",
        "format_example",
        "history",
        "question",
        "agent_scratchpad",
    }


def test_react_template_should_be_renderable():
    assert_renderable(react.REACT_TEMPLATE)


def test_react_template_should_state_the_knowledge_boundary():
    """边界条款必须写进提示词，不能只靠工具层兜底"""
    assert "知识边界" in react.REACT_TEMPLATE
    assert "不得推测或编造" in react.REACT_TEMPLATE


def test_react_template_should_forbid_mode_actions_in_writing():
    assert "节能模式" in react.REACT_TEMPLATE
    assert "不得表示已执行" in react.REACT_TEMPLATE


# ——— 混合式 ———

def test_assess_prompt_should_only_take_the_question():
    assert placeholders(hybrid.ASSESS_PROMPT) == {"question"}
    assert_renderable(hybrid.ASSESS_PROMPT)


def test_assess_prompt_should_list_all_three_fields():
    """协调层要输出三元组，提示词里必须三个字段都写清楚"""
    for field in ("query_type", "processing_mode", "priority"):
        assert field in hybrid.ASSESS_PROMPT


def test_reactive_prompt_should_declare_five_variables():
    assert placeholders(hybrid.REACTIVE_PROMPT) == {
        "tool_descriptions",
        "tool_names",
        "history",
        "question",
        "scratchpad",
    }


def test_analyze_prompt_should_take_data_and_question():
    assert placeholders(hybrid.ANALYZE_PROMPT) == {"energy_data", "env_data", "question"}
    assert_renderable(hybrid.ANALYZE_PROMPT)


def test_analyze_prompt_should_forbid_giving_advice():
    """分析与建议分开是刻意设计：分析节点不许顺手给建议"""
    assert "不要给出整改建议" in hybrid.ANALYZE_PROMPT


def test_recommend_prompt_should_take_analysis_and_priority():
    assert placeholders(hybrid.RECOMMEND_PROMPT) == {"analysis", "priority"}
    assert_renderable(hybrid.RECOMMEND_PROMPT)


def test_recommend_prompt_should_require_action_object_and_recheck():
    assert "动作 + 对象 + 复查条件" in hybrid.RECOMMEND_PROMPT


def test_hybrid_system_prompt_should_state_the_knowledge_boundary():
    assert "知识边界" in hybrid.SYSTEM_PROMPT


# ——— 深思熟虑 ———

def test_modeling_prompt_should_take_perception():
    assert placeholders(delib.MODELING_PROMPT) == {"perception"}
    assert_renderable(delib.MODELING_PROMPT)


def test_reasoning_prompt_should_take_risk_model():
    assert placeholders(delib.REASONING_PROMPT) == {"risk_model"}
    assert_renderable(delib.REASONING_PROMPT)


def test_decision_prompt_should_take_risk_model_and_plans():
    assert placeholders(delib.DECISION_PROMPT) == {"risk_model", "plans"}
    assert_renderable(delib.DECISION_PROMPT)


def test_report_prompt_should_take_question_model_and_decision():
    assert placeholders(delib.REPORT_PROMPT) == {"question", "risk_model", "decision"}
    assert_renderable(delib.REPORT_PROMPT)


def test_deliberative_system_prompt_should_forbid_fabrication():
    """深思熟虑的产出要归档，编造数字代价最大，铁律必须写进去"""
    assert "不得补充、推算或编造" in delib.SYSTEM_PROMPT


def test_report_prompt_should_define_four_sections():
    for section in ("巡检概况", "风险矩阵", "整改方案与决策", "后续跟踪建议"):
        assert section in delib.REPORT_PROMPT
