#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""规格二：六个工具的输入输出合约

每个工具必须做到三件事：参数归一、存在性预检、越界时给出可选范围。
这份规格把每件事都变成断言，重点是「查不到时必须报错，不能编造也不能静默」。

响应码约定（当前实现）：
    0    成功
    4000 缺少必要参数
    4001 设备类型不支持远程控制
    4003 设备离线，指令未下发
    4004 目标不存在，附可选清单
    4006 两个检索条件互斥
"""

import json

import warehouse_data as wd


def call(tool_name: str, **kwargs) -> dict:
    """调用工具并把 JSON 字符串解析成字典"""
    return json.loads(wd.TOOL_MAP[tool_name].invoke(kwargs))


# ——— 工具注册 ———

def test_six_tools_should_be_registered():
    assert list(wd.TOOL_MAP) == [
        "resolve_scene_target",
        "query_device_status",
        "control_device",
        "search_archive",
        "query_energy_stat",
        "query_environment",
    ]


def test_tool_descriptions_should_mention_every_tool_name():
    text = wd.build_tool_descriptions()
    for name in wd.TOOL_MAP:
        assert name in text


# ——— 场景解析 ———

def test_resolve_scene_target_should_return_standard_path_and_racks():
    result = call("resolve_scene_target", target_text="3楼2号库房低氧区")
    assert result["code"] == 0
    assert result["data"]["path"] == "档案综合楼/三层/2号库房/低氧库区"
    assert result["data"]["racks"] == ["3-2-LO-01", "3-2-LO-02"]


def test_resolve_scene_target_should_reject_unknown_floor_with_options():
    result = call("resolve_scene_target", target_text="5楼9号库房")
    assert result["code"] == 4004
    assert result["options"] == ["一层", "二层", "三层"]


# ——— 设备查询 ———

def test_query_device_status_should_list_all_devices_when_keyword_empty():
    result = call("query_device_status", keyword="")
    assert result["code"] == 0
    assert len(result["data"]["list"]) == 11


def test_query_device_status_should_filter_by_type_keyword():
    result = call("query_device_status", keyword="断路器")
    assert result["code"] == 0
    assert len(result["data"]["list"]) == 3
    assert all(item["type"] == "断路器" for item in result["data"]["list"])


def test_query_device_status_should_mark_controllable_with_supported_actions():
    result = call("query_device_status", keyword="断路器")
    online = [item for item in result["data"]["list"] if item["online"]]
    assert online
    for item in online:
        assert item["controllable"] is True
        assert item["supported_actions"] == ["合闸", "分闸"]


def test_query_device_status_should_not_mark_sensor_as_controllable():
    result = call("query_device_status", keyword="温湿度传感器")
    for item in result["data"]["list"]:
        assert item["controllable"] is False


# ——— 设备控制 ———

def test_control_device_should_succeed_on_online_breaker():
    result = call(
        "control_device",
        tb_profile_id="8f3c1a52-7d4e-4b91-9a06-2c5d7e91b301",
        action="分闸",
    )
    assert result["code"] == 0
    assert result["status"] == "分闸"


def test_control_device_should_refuse_offline_device_with_4003():
    """离线设备必须明确拒绝，且不得声称已执行"""
    result = call(
        "control_device",
        tb_profile_id="8f3c1a52-7d4e-4b91-9a06-2c5d7e91b303",
        action="合闸",
    )
    assert result["code"] == 4003
    assert "离线" in result["message"]


def test_control_device_should_refuse_uncontrollable_type_with_4001():
    """空调不在可控类型内，必须拒绝并给出支持的类型"""
    result = call(
        "control_device",
        tb_profile_id="d42f6b70-91c3-4e58-a206-7d8b1c4e9f01",
        action="制冷",
    )
    assert result["code"] == 4001
    assert result["supported_types"] == ["断路器", "门禁"]


def test_control_device_should_refuse_mode_action_with_4001():
    """模式类动作是知识边界的重点，绝不能返回成功"""
    result = call(
        "control_device",
        tb_profile_id="d42f6b70-91c3-4e58-a206-7d8b1c4e9f01",
        action="设置成节能模式",
    )
    assert result["code"] == 4001


def test_control_device_should_refuse_unknown_device():
    result = call("control_device", tb_profile_id="not-a-device", action="分闸")
    assert result["code"] != 0


# ——— 档案检索 ———

def test_search_archive_should_match_by_title_keyword():
    result = call("search_archive", doc_no="决算")
    assert result["code"] == 0
    assert result["data"]["pages"]["total"] == 1
    assert result["data"]["pages"]["list"][0]["docNo"] == "WS-2023-Y-0012"


def test_search_archive_should_match_by_category():
    result = call("search_archive", category_name="会计档案")
    assert result["code"] == 0
    assert result["data"]["pages"]["total"] == 3


def test_search_archive_should_reject_two_conditions_with_4006():
    """档号检索与类型检索互斥，是硬约束"""
    result = call("search_archive", doc_no="决算", category_name="会计档案")
    assert result["code"] == 4006


def test_search_archive_should_reject_empty_conditions_with_4000():
    result = call("search_archive")
    assert result["code"] == 4000
    assert result["available_categories"]


# ——— 能耗 ———

def test_query_energy_stat_should_summarize_whole_building():
    result = call("query_energy_stat", storage_name="全部")
    assert result["code"] == 0
    assert result["data"]["scope"] == "全楼"
    assert len(result["data"]["storages"]) == 5


def test_query_energy_stat_should_reject_unregistered_storage_with_options():
    result = call("query_energy_stat", storage_name="9号库房")
    assert result["code"] == 4004
    assert "3楼/2号库房" in result["options"]


# ——— 环境 ———

def test_query_environment_should_report_alarm_for_humid_storage():
    result = call("query_environment", storage_name="3楼2号库房")
    assert result["code"] == 0
    entry = result["data"][0]
    assert entry["alarm"] is True
    assert any("湿度" in detail for detail in entry["alarm_detail"])


def test_query_environment_should_report_normal_for_compliant_storage():
    result = call("query_environment", storage_name="1楼1号库房")
    assert result["code"] == 0
    entry = result["data"][0]
    assert entry["alarm"] is False
    assert entry["alarm_detail"] == []


def test_query_environment_should_reject_unregistered_storage():
    result = call("query_environment", storage_name="9号库房")
    assert result["code"] == 4004
