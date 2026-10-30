#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""规格三：知识边界的入口预检

边界三层防线的第一层：职责范围外的话题不进模型，直接给出固定话术。
这份规格同时锁住两个方向——越界话题必须命中，领域内问题绝不能误伤。

误伤比漏拦更危险：一旦领域内的问题被拦下，用户就再也拿不到答案了。
"""

import warehouse_data as wd


# ——— 越界话题必须命中 ———

def test_check_scope_should_block_question_about_archive_content():
    """正文不在值守范围"""
    assert wd.check_scope("这份档案正文写了什么") is not None


def test_check_scope_should_block_request_to_read_full_text():
    assert wd.check_scope("帮我读一下这份档案的全文") is not None


def test_check_scope_should_block_question_about_regulation():
    assert wd.check_scope("档案库房安全管理规定在制度上怎么解读") is not None


def test_check_scope_should_block_question_about_legal_responsibility():
    assert wd.check_scope("档案丢失要承担什么法律责任") is not None


def test_check_scope_should_block_question_about_staffing():
    assert wd.check_scope("这个月的排班怎么安排") is not None


def test_check_scope_should_block_question_about_weather():
    assert wd.check_scope("今天天气怎么样") is not None


def test_check_scope_reply_should_point_to_a_real_direction():
    """拦截话术不能只说「不知道」，要给出可查询的方向"""
    reply = wd.check_scope("这份档案正文写了什么")
    assert reply is not None
    assert "调阅" in reply or "咨询" in reply or "建议" in reply


# ——— 领域内问题绝不误伤 ———

def test_check_scope_should_pass_question_about_location():
    assert wd.check_scope("3楼2号库房的低氧区在什么位置") is None


def test_check_scope_should_pass_question_about_device_control():
    assert wd.check_scope("把3楼2号库房V6库区的断路器分闸") is None


def test_check_scope_should_pass_question_about_archive_location():
    assert wd.check_scope("查一下题名带决算的档案放在哪") is None


def test_check_scope_should_pass_question_about_energy():
    assert wd.check_scope("3楼2号库房这个月的能耗是多少") is None


def test_check_scope_should_pass_question_about_environment():
    assert wd.check_scope("各库房温湿度有没有告警") is None


def test_check_scope_should_pass_question_comparing_storages():
    assert wd.check_scope("对比一下各库房的能耗情况") is None


def test_check_scope_should_pass_request_for_inspection_report():
    assert wd.check_scope("生成一份全楼库房的季度巡检报告") is None
