#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""规格一：四套域数据的结构与自洽性

这份规格描述「数据本身必须成立的事实」，不涉及任何模型调用。
重构时这些断言应当始终成立——若某条失败，说明数据被改动了，属于需要人确认的变更。

覆盖：
    - 空间树的库房、库区、密集架数量
    - 能源环境数据与空间树的一一对应
    - 全楼能耗汇总等于各库房之和
    - 环境阈值可判定出唯一的告警库房
    - 设备台账的编号唯一且挂载位置真实存在
    - 档案台账的档号唯一且架位真实存在
    - 可控设备类型只有断路器与门禁
"""

import json
from typing import Any, Dict, List, Tuple

import pytest

import warehouse_data as wd


def all_storages() -> List[Tuple[int, str, Dict[str, Any]]]:
    """展开空间树，返回 (楼层号, 库房名, 库区字典) 列表"""
    result: List[Tuple[int, str, Dict[str, Any]]] = []
    for floor_no, floor in wd.SPACE_TREE["floors"].items():
        for storage_name, storage in floor["storages"].items():
            result.append((floor_no, storage_name, storage["zones"]))
    return result


def all_racks() -> set[str]:
    """空间树里的全部密集架编号"""
    return {
        rack
        for _, _, zones in all_storages()
        for zone in zones.values()
        for rack in zone["racks"]
    }


# ——— 空间树 ———

def test_space_tree_should_have_five_storages():
    assert len(all_storages()) == 5


def test_space_tree_should_have_eight_zones():
    assert sum(len(zones) for _, _, zones in all_storages()) == 8


def test_space_tree_should_have_fourteen_racks():
    assert len(all_racks()) == 14


def test_low_oxygen_zone_should_carry_two_racks():
    """低氧库区是多个端到端用例的固定落点，架位不能变"""
    matched = [zs for floor, name, zs in all_storages() if floor == 3 and name == "2号库房"]
    assert len(matched) == 1
    assert matched[0]["低氧库区"]["racks"] == ["3-2-LO-01", "3-2-LO-02"]


# ——— 能源环境 ———

def test_storage_env_keys_should_match_space_tree_exactly():
    """能源环境数据的键必须与空间树的库房一一对应，不能多也不能少"""
    tree_keys = {f"{floor}-{name}" for floor, name, _ in all_storages()}
    assert set(wd.STORAGE_ENV) == tree_keys


def test_energy_total_should_equal_the_sum_of_every_storage():
    """全楼汇总不是独立维护的数字，必须等于各库房之和"""
    today = round(sum(env["energy_today"] for env in wd.STORAGE_ENV.values()), 1)
    month = round(sum(env["energy_month"] for env in wd.STORAGE_ENV.values()), 1)

    payload = json.loads(wd.query_energy_stat.invoke({"storage_name": "全部"}))

    assert payload["data"]["energy_today"] == today
    assert payload["data"]["energy_month"] == month


def test_energy_total_should_stay_at_the_recorded_values():
    """特征值：当前全楼今日 122.9、本月 3452.9 kWh"""
    payload = json.loads(wd.query_energy_stat.invoke({"storage_name": "全部"}))
    assert payload["data"]["energy_today"] == 122.9
    assert payload["data"]["energy_month"] == 3452.9


def test_only_three_two_storage_should_raise_a_humidity_alarm():
    """阈值 18-24 度、45%-60%，当前只有 3楼2号库房 的湿度 62.8% 越界"""
    alarms: Dict[str, List[str]] = {}

    for key, env in wd.STORAGE_ENV.items():
        temp_low, temp_high = wd.ENV_LIMITS["temperature"]
        hum_low, hum_high = wd.ENV_LIMITS["humidity"]
        reasons = []
        if not temp_low <= env["temperature"] <= temp_high:
            reasons.append("temperature")
        if not hum_low <= env["humidity"] <= hum_high:
            reasons.append("humidity")
        if reasons:
            alarms[key] = reasons

    assert alarms == {"3-2号库房": ["humidity"]}


# ——— 设备台账 ———

def test_device_db_should_have_eleven_devices_with_unique_ids():
    ids = [device["tb_profile_id"] for device in wd.DEVICE_DB]
    assert len(ids) == 11
    assert len(set(ids)) == 11


def test_every_device_should_mount_on_an_existing_floor_and_storage():
    """设备不能挂在不存在的楼层或库房上；库区可能挂在入口，故只校验楼层与库房"""
    valid = {(floor, name) for floor, name, _ in all_storages()}
    for device in wd.DEVICE_DB:
        assert (device["floor"], device["storage"]) in valid


def test_offline_device_should_be_exactly_the_2f_1st_breaker():
    """离线设备是控制类用例的固定落点"""
    offline = [d for d in wd.DEVICE_DB if not d["online"]]
    assert [d["tb_profile_id"] for d in offline] == [
        "8f3c1a52-7d4e-4b91-9a06-2c5d7e91b303"
    ]


# ——— 档案台账 ———

def test_archive_db_should_have_eight_records_with_unique_doc_no():
    doc_nos = [item["doc_no"] for item in wd.ARCHIVE_DB]
    assert len(doc_nos) == 8
    assert len(set(doc_nos)) == 8


def test_every_archive_should_sit_on_an_existing_rack():
    racks = all_racks()
    for item in wd.ARCHIVE_DB:
        assert item["rack"] in racks


def test_archive_categories_should_cover_four_types():
    assert set(item["category_name"] for item in wd.ARCHIVE_DB) == {
        "会计档案",
        "科技档案",
        "文书档案",
        "人事档案",
    }


# ——— 可控类型 ———

def test_only_breaker_and_door_should_be_controllable():
    assert set(wd.CONTROLLABLE_ACTIONS) == {"断路器", "门禁"}


def test_breaker_and_door_actions_should_be_exactly_two_each():
    assert wd.CONTROLLABLE_ACTIONS["断路器"] == ["合闸", "分闸"]
    assert wd.CONTROLLABLE_ACTIONS["门禁"] == ["开门", "关门"]
