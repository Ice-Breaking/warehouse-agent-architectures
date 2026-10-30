#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
库房大屏值守助手 - 公共数据模块

反应式、混合式、深思熟虑三个架构共用这一份域数据、工具集合与模型入口，
避免同一套库房台账在多个项目里各写一遍导致口径漂移。

包含四部分：
    数据准备：空间树、设备台账、档案台账、能源环境
    工具设计：六个工具，内部完成参数归一、存在性预检、越界提示
    知识边界：职责范围预检
    模型入口：OpenAI 兼容客户端与单次对话封装

运行前设置环境变量 DASHSCOPE_API_KEY；模型与端点可通过 LLM_MODEL、LLM_BASE_URL 覆盖。
"""

import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import tool
from openai import OpenAI

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.8-27b")

# 反应式分支允许的最大工具调用轮次，超过即抛错，不靠框架的递归上限兜底
MAX_ROUNDS = 6

# 多轮会话里保留进提示词的最近轮数
HISTORY_WINDOW = 5


# ==========================================================================
# 数据准备
#
# 四套域数据按真实系统的口径建模：
#   - 空间树：楼宇 -> 楼层 -> 库房 -> 库区 -> 密集架，库区带别名（v6 / 低氧）
#   - 设备台账：tbProfileId 为 UUID，设备挂在库区下，区分在线状态与可控类型
#   - 档案台账：docNo 检索与 categoryName 检索互斥，落位精确到密集架 A/B 面
#   - 能源环境：按库房记录温湿度与能耗，超阈值即告警
# ==========================================================================

SPACE_TREE: Dict[str, Any] = {
    "building": "档案综合楼",
    "floors": {
        1: {
            "name": "一层",
            "storages": {
                "1号库房": {
                    "zones": {
                        "常温库区": {"aliases": ["常温", "普通区"], "racks": ["1-1-01", "1-1-02", "1-1-03"]},
                        "消毒暂存区": {"aliases": ["暂存", "消毒区"], "racks": ["1-1-04"]},
                    }
                },
            },
        },
        2: {
            "name": "二层",
            "storages": {
                "1号库房": {
                    "zones": {
                        "常温库区": {"aliases": ["常温"], "racks": ["2-1-01", "2-1-02"]},
                        "防磁库区": {"aliases": ["防磁"], "racks": ["2-1-03"]},
                    }
                },
                "2号库房": {
                    "zones": {
                        "常温库区": {"aliases": ["常温"], "racks": ["2-2-01"]},
                    }
                },
            },
        },
        3: {
            "name": "三层",
            "storages": {
                "2号库房": {
                    "zones": {
                        "V6库区": {
                            "aliases": ["v6", "六区", "V6区"],
                            "racks": ["3-2-V6-01", "3-2-V6-02", "3-2-V6-03"],
                        },
                        "低氧库区": {
                            "aliases": ["低氧", "低氧区", "充氮"],
                            "racks": ["3-2-LO-01", "3-2-LO-02"],
                        },
                    }
                },
                "3号库房": {
                    "zones": {
                        "常温库区": {"aliases": ["常温"], "racks": ["3-3-01"]},
                    }
                },
            },
        },
    },
}

DEVICE_DB: List[Dict[str, Any]] = [
    {
        "tb_profile_id": "8f3c1a52-7d4e-4b91-9a06-2c5d7e91b301",
        "name": "3楼2号库房V6库区断路器",
        "type": "断路器",
        "floor": 3,
        "storage": "2号库房",
        "zone": "V6库区",
        "online": True,
        "status": "合闸",
    },
    {
        "tb_profile_id": "8f3c1a52-7d4e-4b91-9a06-2c5d7e91b302",
        "name": "3楼2号库房低氧库区断路器",
        "type": "断路器",
        "floor": 3,
        "storage": "2号库房",
        "zone": "低氧库区",
        "online": True,
        "status": "合闸",
    },
    {
        "tb_profile_id": "8f3c1a52-7d4e-4b91-9a06-2c5d7e91b303",
        "name": "2楼1号库房断路器",
        "type": "断路器",
        "floor": 2,
        "storage": "1号库房",
        "zone": "防磁库区",
        "online": False,
        "status": "分闸",
    },
    {
        "tb_profile_id": "a71b0d33-5c28-4e77-8f14-9b6e2d4a5501",
        "name": "3楼2号库房入口门禁",
        "type": "门禁",
        "floor": 3,
        "storage": "2号库房",
        "zone": "入口",
        "online": True,
        "status": "关门",
    },
    {
        "tb_profile_id": "a71b0d33-5c28-4e77-8f14-9b6e2d4a5502",
        "name": "3楼3号库房入口门禁",
        "type": "门禁",
        "floor": 3,
        "storage": "3号库房",
        "zone": "入口",
        "online": True,
        "status": "关门",
    },
    {
        "tb_profile_id": "c05d9e18-2a63-4d0f-b7c8-31f4a8e27c01",
        "name": "3楼2号库房V6库区温湿度传感器",
        "type": "温湿度传感器",
        "floor": 3,
        "storage": "2号库房",
        "zone": "V6库区",
        "online": True,
        "status": "采集中",
    },
    {
        "tb_profile_id": "c05d9e18-2a63-4d0f-b7c8-31f4a8e27c02",
        "name": "3楼2号库房低氧库区温湿度传感器",
        "type": "温湿度传感器",
        "floor": 3,
        "storage": "2号库房",
        "zone": "低氧库区",
        "online": True,
        "status": "采集中",
    },
    {
        "tb_profile_id": "d42f6b70-91c3-4e58-a206-7d8b1c4e9f01",
        "name": "3楼2号库房V6库区空调",
        "type": "空调",
        "floor": 3,
        "storage": "2号库房",
        "zone": "V6库区",
        "online": True,
        "status": "制冷中",
    },
    {
        "tb_profile_id": "d42f6b70-91c3-4e58-a206-7d8b1c4e9f02",
        "name": "3楼2号库房低氧库区除湿机",
        "type": "除湿机",
        "floor": 3,
        "storage": "2号库房",
        "zone": "低氧库区",
        "online": True,
        "status": "运行中",
    },
    {
        "tb_profile_id": "e19a4c86-3f57-4b02-9d41-6a2e8b5c7d01",
        "name": "3楼2号库房V6库区摄像机",
        "type": "摄像机",
        "floor": 3,
        "storage": "2号库房",
        "zone": "V6库区",
        "online": True,
        "status": "录像中",
    },
    {
        "tb_profile_id": "e19a4c86-3f57-4b02-9d41-6a2e8b5c7d02",
        "name": "3楼2号库房低氧库区烟感",
        "type": "烟感",
        "floor": 3,
        "storage": "2号库房",
        "zone": "低氧库区",
        "online": True,
        "status": "正常",
    },
]

ARCHIVE_DB: List[Dict[str, Any]] = [
    {
        "doc_no": "WS-2023-Y-0012",
        "title": "2023年度财务决算报告",
        "category_name": "会计档案",
        "floor": 3,
        "storage": "2号库房",
        "zone": "V6库区",
        "rack": "3-2-V6-01",
        "side": "A",
        "box": "A-03",
    },
    {
        "doc_no": "WS-2023-Y-0031",
        "title": "2023年度预算执行情况说明",
        "category_name": "会计档案",
        "floor": 3,
        "storage": "2号库房",
        "zone": "V6库区",
        "rack": "3-2-V6-01",
        "side": "A",
        "box": "A-07",
    },
    {
        "doc_no": "WS-2022-K-0107",
        "title": "智能库房环境监测系统技术方案",
        "category_name": "科技档案",
        "floor": 3,
        "storage": "2号库房",
        "zone": "V6库区",
        "rack": "3-2-V6-02",
        "side": "B",
        "box": "B-02",
    },
    {
        "doc_no": "WS-2021-W-0455",
        "title": "档案库房安全管理规定",
        "category_name": "文书档案",
        "floor": 3,
        "storage": "2号库房",
        "zone": "低氧库区",
        "rack": "3-2-LO-01",
        "side": "A",
        "box": "A-01",
    },
    {
        "doc_no": "WS-2020-W-0088",
        "title": "库房巡查记录汇编",
        "category_name": "文书档案",
        "floor": 3,
        "storage": "2号库房",
        "zone": "低氧库区",
        "rack": "3-2-LO-02",
        "side": "B",
        "box": "B-05",
    },
    {
        "doc_no": "WS-2023-R-0026",
        "title": "2023年度人员岗位聘任材料",
        "category_name": "人事档案",
        "floor": 2,
        "storage": "1号库房",
        "zone": "常温库区",
        "rack": "2-1-01",
        "side": "A",
        "box": "A-11",
    },
    {
        "doc_no": "WS-2022-C-0019",
        "title": "财务凭证装订册",
        "category_name": "会计档案",
        "floor": 2,
        "storage": "2号库房",
        "zone": "常温库区",
        "rack": "2-2-01",
        "side": "B",
        "box": "B-03",
    },
    {
        "doc_no": "WS-2021-K-0203",
        "title": "密集架控制系统验收报告",
        "category_name": "科技档案",
        "floor": 1,
        "storage": "1号库房",
        "zone": "常温库区",
        "rack": "1-1-02",
        "side": "A",
        "box": "A-06",
    },
]

# 键为「楼层-库房名」，与空间树一一对应
STORAGE_ENV: Dict[str, Dict[str, Any]] = {
    "1-1号库房": {
        "temperature": 22.6,
        "humidity": 55.2,
        "updated_at": "2026-10-29 09:20:00",
        "energy_today": 18.4,
        "energy_month": 512.7,
    },
    "2-1号库房": {
        "temperature": 23.1,
        "humidity": 58.4,
        "updated_at": "2026-10-29 09:20:00",
        "energy_today": 24.9,
        "energy_month": 703.5,
    },
    "2-2号库房": {
        "temperature": 22.2,
        "humidity": 54.0,
        "updated_at": "2026-10-29 09:20:00",
        "energy_today": 20.7,
        "energy_month": 588.1,
    },
    "3-2号库房": {
        "temperature": 21.4,
        "humidity": 62.8,
        "updated_at": "2026-10-29 09:20:00",
        "energy_today": 42.6,
        "energy_month": 1180.4,
    },
    "3-3号库房": {
        "temperature": 21.9,
        "humidity": 51.6,
        "updated_at": "2026-10-29 09:20:00",
        "energy_today": 16.3,
        "energy_month": 468.2,
    },
}

# 可控类型与其支持的动作：只保留真实 RPC 已接通的动作，模式类动作一律不支持
CONTROLLABLE_ACTIONS: Dict[str, List[str]] = {
    "断路器": ["合闸", "分闸"],
    "门禁": ["开门", "关门"],
}

# 模式类动作特征词，用于区分「不支持的动作」与「不支持的模式配置」
MODE_ACTION_KEYWORDS: List[str] = ["模式", "策略", "定时", "联动", "自动", "计划", "场景"]

# 库房环境阈值，超出即告警
ENV_LIMITS: Dict[str, Tuple[float, float]] = {
    "temperature": (18.0, 24.0),
    "humidity": (45.0, 60.0),
}


# ==========================================================================
# 工具设计
#
# 每个工具内部完成三件事：参数归一、存在性预检、越界时给出可选范围。
# 不做「查不到就当没有」的静默处理，也不做「查不到就编一个」的补全。
# ==========================================================================

_CN_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _to_int(token: str) -> Optional[int]:
    """把阿拉伯数字或十以内的中文数字转成整数"""
    if token.isdigit():
        return int(token)
    if token in _CN_DIGITS:
        return _CN_DIGITS[token]
    if token.startswith("十") and len(token) == 1:
        return 10
    return None


def _iter_zones(floor: Optional[int] = None):
    """遍历空间树中的库区，产出 (楼层, 库房名, 库区名, 库区定义)"""
    for floor_no, floor_node in SPACE_TREE["floors"].items():
        if floor is not None and floor_no != floor:
            continue
        for storage_name, storage_node in floor_node["storages"].items():
            for zone_name, zone_node in storage_node["zones"].items():
                yield floor_no, storage_name, zone_name, zone_node


def _zone_matches(text: str, zone_name: str, aliases: List[str]) -> bool:
    """库区命中判断：库区全名或其别名出现在文本中即命中，大小写不敏感"""
    lowered = text.casefold()
    if zone_name.casefold() in lowered:
        return True
    return any(alias.casefold() in lowered for alias in aliases)


def _storage_options(floor: Optional[int] = None) -> List[str]:
    """返回可选的「楼层/库房」描述，供越界时提示"""
    options = []
    for floor_no, floor_node in SPACE_TREE["floors"].items():
        if floor is not None and floor_no != floor:
            continue
        for storage_name in floor_node["storages"]:
            options.append(f"{floor_node['name']}/{storage_name}")
    return options


def _locate_device(device_ref: str) -> Optional[Dict[str, Any]]:
    """按 UUID 精确匹配，或按设备名唯一子串匹配"""
    ref = device_ref.strip()
    if not ref:
        return None
    for device in DEVICE_DB:
        if device["tb_profile_id"] == ref:
            return device
    hits = [device for device in DEVICE_DB if ref in device["name"]]
    if len(hits) == 1:
        return hits[0]
    return None


@tool
def resolve_scene_target(target_text: str) -> str:
    """解析并校验三维场景目标位置，返回标准化路径与密集架清单。

    参数 target_text：自然语言位置描述，例如 "3楼2号库房低氧区"、"V6库区"、"二层1号库房防磁区"。
    库里编号与库区别名均可识别（v6 归一到 V6库区，低氧 归一到 低氧库区）。
    目标不唯一时返回候选列表，目标不存在时返回可选范围。
    """
    text = target_text.strip()
    if not text:
        return json.dumps({"code": 4000, "message": "位置描述为空，请给出楼层、库房或库区"}, ensure_ascii=False)

    floor_match = re.search(r"([0-9]+|[一二三四五六七八九十])\s*[楼层]", text)
    floor_no = _to_int(floor_match.group(1)) if floor_match else None

    storage_match = re.search(r"([0-9一二三四五六七八九十]+)\s*号库房", text)
    storage_no = _to_int(storage_match.group(1)) if storage_match else None
    storage_name = f"{storage_no}号库房" if storage_no else None

    if floor_no is not None and floor_no not in SPACE_TREE["floors"]:
        return json.dumps(
            {
                "code": 4004,
                "message": f"空间树中不存在 {floor_no} 层",
                "options": [node["name"] for node in SPACE_TREE["floors"].values()],
            },
            ensure_ascii=False,
        )

    candidates = []
    for zone_floor, zone_storage, zone_name, zone_node in _iter_zones(floor_no):
        if storage_name and zone_storage != storage_name:
            continue
        if not _zone_matches(text, zone_name, zone_node["aliases"]):
            continue
        candidates.append(
            {
                "floor": zone_floor,
                "floor_name": SPACE_TREE["floors"][zone_floor]["name"],
                "storage": zone_storage,
                "zone": zone_name,
                "racks": zone_node["racks"],
                "path": f"{SPACE_TREE['building']}/{SPACE_TREE['floors'][zone_floor]['name']}/{zone_storage}/{zone_name}",
            }
        )

    # 只给了库房、没提库区时，返回该库房下的全部库区；未给楼层则跨楼层返回
    if not candidates and storage_name:
        storages = []
        for floor_no_iter, floor_node in SPACE_TREE["floors"].items():
            if floor_no is not None and floor_no_iter != floor_no:
                continue
            storage_node = floor_node["storages"].get(storage_name)
            if not storage_node:
                continue
            for zone_name, zone_node in storage_node["zones"].items():
                storages.append(
                    {
                        "floor": floor_no_iter,
                        "floor_name": floor_node["name"],
                        "storage": storage_name,
                        "zone": zone_name,
                        "racks": zone_node["racks"],
                    }
                )
        if storages:
            scope = f"{floor_no} 层" if floor_no is not None else "多个楼层"
            return json.dumps(
                {
                    "code": 0,
                    "message": f"未指定库区，{scope} 的 {storage_name} 包含以下库区，请确认具体库区",
                    "data": storages,
                },
                ensure_ascii=False,
            )

    if not candidates:
        return json.dumps(
            {
                "code": 4004,
                "message": f"未找到与「{text}」匹配的库区，空间树中没有该位置",
                "options": _storage_options(floor_no),
            },
            ensure_ascii=False,
        )

    if len(candidates) > 1:
        return json.dumps(
            {
                "code": 0,
                "message": "匹配到多个库区，请让用户确认具体位置",
                "data": candidates,
            },
            ensure_ascii=False,
        )

    return json.dumps({"code": 0, "data": candidates[0]}, ensure_ascii=False)


@tool
def query_device_status(keyword: str) -> str:
    """查询设备台账与实时状态。

    参数 keyword：设备名关键字、设备类型、库区名或楼层，例如 "断路器"、"V6库区"、"3楼门禁"。
    关键字为空时返回全部设备。未命中时返回空列表与可选设备类型。
    """
    text = keyword.strip()
    types = sorted({device["type"] for device in DEVICE_DB})

    # 楼层单独作为过滤条件，避免 "3楼门禁" 把整层设备都命中
    floor_hint = None
    floor_match = re.search(r"([0-9]+|[一二三四五六七八九十])\s*[楼层]", text)
    if floor_match:
        floor_hint = _to_int(floor_match.group(1))
        text = (text[: floor_match.start()] + text[floor_match.end():]).strip()

    lowered = text.casefold()
    matched = []
    for device in DEVICE_DB:
        if floor_hint is not None and device["floor"] != floor_hint:
            continue
        if lowered and not (
            lowered in device["name"].casefold()
            or lowered in device["type"].casefold()
            or lowered in device["zone"].casefold()
            or lowered in device["storage"].casefold()
        ):
            continue
        matched.append(device)

    if not matched:
        return json.dumps(
            {
                "code": 0,
                "data": {"list": [], "total": 0},
                "message": f"未查询到与「{text}」相关的设备",
                "available_types": types,
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "code": 0,
            "data": {
                "list": [
                    {
                        "tb_profile_id": device["tb_profile_id"],
                        "name": device["name"],
                        "type": device["type"],
                        "position": f"{device['floor']}楼/{device['storage']}/{device['zone']}",
                        "online": device["online"],
                        "status": device["status"],
                        "controllable": device["type"] in CONTROLLABLE_ACTIONS,
                        "supported_actions": CONTROLLABLE_ACTIONS.get(device["type"], []),
                    }
                    for device in matched
                ],
                "total": len(matched),
            },
        },
        ensure_ascii=False,
    )


@tool
def control_device(tb_profile_id: str, action: str) -> str:
    """对设备下发控制指令，按响应码回报真实结果。

    参数 tb_profile_id：设备的 tbProfileId（UUID），也可传设备名做唯一匹配。
    参数 action：控制动作，断路器支持 合闸/分闸，门禁支持 开门/关门。
    设备离线、类型不可控、动作不在支持清单时返回非 0 响应码，不会改写成成功。
    """
    device = _locate_device(tb_profile_id)
    if device is None:
        return json.dumps(
            {
                "code": 4005,
                "message": f"未找到设备「{tb_profile_id.strip()}」，请先查询设备台账获取 tbProfileId",
            },
            ensure_ascii=False,
        )

    if not device["online"]:
        return json.dumps(
            {
                "code": 4003,
                "message": f"{device['name']} 当前离线，指令未下发",
                "device": device["name"],
            },
            ensure_ascii=False,
        )

    supported = CONTROLLABLE_ACTIONS.get(device["type"])
    if not supported:
        return json.dumps(
            {
                "code": 4001,
                "message": f"{device['type']} 不支持远程控制，当前仅断路器与门禁可控",
                "device": device["name"],
                "supported_types": list(CONTROLLABLE_ACTIONS.keys()),
            },
            ensure_ascii=False,
        )

    normalized = action.strip()
    if normalized not in supported:
        if any(keyword in normalized for keyword in MODE_ACTION_KEYWORDS):
            return json.dumps(
                {
                    "code": 4002,
                    "message": f"模式类动作不在支持范围，{device['type']} 仅支持 {', '.join(supported)}",
                    "device": device["name"],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "code": 4002,
                "message": f"动作「{normalized}」不受支持，{device['type']} 仅支持 {', '.join(supported)}",
                "device": device["name"],
                "supported_actions": supported,
            },
            ensure_ascii=False,
        )

    device["status"] = normalized
    return json.dumps(
        {
            "code": 0,
            "message": f"{device['name']} 已{normalized}",
            "device": device["name"],
            "status": normalized,
        },
        ensure_ascii=False,
    )


@tool
def search_archive(doc_no: str = "", category_name: str = "") -> str:
    """按档号或按档案类型检索档案存放位置。

    参数 doc_no：档号或题名关键字，例如 "决算"、"WS-2023-Y-0012"。
    参数 category_name：档案类型，例如 "会计档案"、"文书档案"、"科技档案"、"人事档案"。
    两个参数只能二选一：同时传入返回互斥错误，都不传返回参数缺失错误。
    """
    doc_keyword = doc_no.strip()
    category_keyword = category_name.strip()

    if doc_keyword and category_keyword:
        return json.dumps(
            {
                "code": 4006,
                "message": "docNo 检索与 categoryName 检索互斥，一次只能使用其中一个条件",
            },
            ensure_ascii=False,
        )

    if not doc_keyword and not category_keyword:
        return json.dumps(
            {
                "code": 4000,
                "message": "请提供 doc_no（档号或题名关键字）或 category_name（档案类型）其中一个",
                "available_categories": sorted({item["category_name"] for item in ARCHIVE_DB}),
            },
            ensure_ascii=False,
        )

    if doc_keyword:
        lowered = doc_keyword.casefold()
        matched = [
            item
            for item in ARCHIVE_DB
            if lowered in item["doc_no"].casefold() or lowered in item["title"].casefold()
        ]
    else:
        lowered = category_keyword.casefold()
        matched = [item for item in ARCHIVE_DB if lowered in item["category_name"].casefold()]

    if not matched:
        return json.dumps(
            {
                "code": 0,
                "data": {"pages": {"list": [], "total": 0}},
                "message": "未检索到符合条件的档案",
                "available_categories": sorted({item["category_name"] for item in ARCHIVE_DB}),
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "code": 0,
            "data": {
                "pages": {
                    "list": [
                        {
                            "docNo": item["doc_no"],
                            "title": item["title"],
                            "categoryName": item["category_name"],
                            "position": f"{item['floor']}楼/{item['storage']}/{item['zone']}/{item['rack']}",
                            "side": f"{item['side']} 面",
                            "box": item["box"],
                        }
                        for item in matched
                    ],
                    "total": len(matched),
                }
            },
        },
        ensure_ascii=False,
    )


def _match_storage(storage_name: str) -> List[Tuple[str, Dict[str, Any]]]:
    """按库房名匹配能源环境记录，键为「楼层-库房名」"""
    text = storage_name.strip()
    if not text:
        return []
    matched = []
    for key, env in STORAGE_ENV.items():
        floor_no, name = key.split("-", 1)
        if text.casefold() in name.casefold() or (f"{floor_no}楼" in text and name.casefold() in text.casefold()):
            matched.append((key, env))
    return matched


@tool
def query_energy_stat(storage_name: str) -> str:
    """查询库房能耗数据。

    参数 storage_name：库房名，例如 "2号库房"、"3号库房"；传 "全部" 返回全楼汇总。
    未登记库房返回可选库房清单，不做估算。
    """
    text = storage_name.strip()
    if text in ("全部", "所有", "全楼", ""):
        total_today = sum(env["energy_today"] for env in STORAGE_ENV.values())
        total_month = sum(env["energy_month"] for env in STORAGE_ENV.values())
        return json.dumps(
            {
                "code": 0,
                "data": {
                    "scope": "全楼",
                    "energy_today": round(total_today, 1),
                    "energy_month": round(total_month, 1),
                    "storages": [
                        {
                            "key": key,
                            "energy_today": env["energy_today"],
                            "energy_month": env["energy_month"],
                        }
                        for key, env in STORAGE_ENV.items()
                    ],
                },
            },
            ensure_ascii=False,
        )

    matched = _match_storage(text)
    if not matched:
        return json.dumps(
            {
                "code": 4004,
                "message": f"未登记「{text}」的能耗数据",
                "options": [f"{key.split('-', 1)[0]}楼/{key.split('-', 1)[1]}" for key in STORAGE_ENV],
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "code": 0,
            "data": [
                {
                    "storage": f"{key.split('-', 1)[0]}楼/{key.split('-', 1)[1]}",
                    "energy_today": env["energy_today"],
                    "energy_month": env["energy_month"],
                }
                for key, env in matched
            ],
        },
        ensure_ascii=False,
    )


@tool
def query_environment(storage_name: str) -> str:
    """查询库房温湿度与环境告警。

    参数 storage_name：库房名，例如 "2号库房"、"3号库房"。
    返回温湿度、采集时间与告警状态；温度超出 18-24 度或湿度超出 45%-60% 判为告警。
    """
    text = storage_name.strip()
    matched = _match_storage(text)
    if not matched:
        return json.dumps(
            {
                "code": 4004,
                "message": f"未登记「{text}」的环境数据",
                "options": [f"{key.split('-', 1)[0]}楼/{key.split('-', 1)[1]}" for key in STORAGE_ENV],
            },
            ensure_ascii=False,
        )

    results = []
    for key, env in matched:
        alarms = []
        temp_low, temp_high = ENV_LIMITS["temperature"]
        humidity_low, humidity_high = ENV_LIMITS["humidity"]
        if not temp_low <= env["temperature"] <= temp_high:
            alarms.append(f"温度 {env['temperature']} 度超出 {temp_low}-{temp_high} 度")
        if not humidity_low <= env["humidity"] <= humidity_high:
            alarms.append(f"湿度 {env['humidity']}% 超出 {humidity_low}-{humidity_high}%")
        results.append(
            {
                "storage": f"{key.split('-', 1)[0]}楼/{key.split('-', 1)[1]}",
                "temperature": env["temperature"],
                "humidity": env["humidity"],
                "updated_at": env["updated_at"],
                "alarm": bool(alarms),
                "alarm_detail": alarms,
            }
        )

    return json.dumps({"code": 0, "data": results}, ensure_ascii=False)


TOOLS = [
    resolve_scene_target,
    query_device_status,
    control_device,
    search_archive,
    query_energy_stat,
    query_environment,
]

TOOL_MAP: Dict[str, Any] = {item.name: item for item in TOOLS}


def build_tool_descriptions() -> str:
    """拼接工具清单，参数名取自工具签名，保证提示词与实现同步"""
    lines = []
    for item in TOOLS:
        params = ", ".join(item.args.keys()) if item.args else "无参数"
        description = " ".join(item.description.split())
        lines.append(f"- {item.name}({params})\n  {description}")
    return "\n".join(lines)


# ==========================================================================
# 复用片段
#
# 三个架构都会用到的采集与格式化逻辑集中在这里，避免各写一份导致口径漂移。
# ==========================================================================


def collect_environment_all() -> List[str]:
    """逐个库房采集真实环境数据，返回工具返回的原文列表

    混合式与深思熟虑都要全楼环境数据，采集方式必须一致，否则两边口径会漂移。
    返回工具的原始输出而不是解析后的结构，保证下游引用的数字与工具结果逐字一致。
    """
    snippets: List[str] = []
    for key in STORAGE_ENV:
        floor_no, name = key.split("-", 1)
        snippets.append(str(query_environment.invoke({"storage_name": f"{floor_no}楼{name}"})))
    return snippets


def format_history(history: List[Tuple[str, str]]) -> str:
    """把多轮历史拼成提示词片段，只保留最近若干轮

    轮数由 HISTORY_WINDOW 控制：全量塞进提示词既浪费 token，
    也会让早期无关内容干扰当前判断。
    """
    if not history:
        return "无"
    recent = history[-HISTORY_WINDOW:]
    return "\n".join(f"用户：{asked}\n助手：{answered}" for asked, answered in recent)


# ReAct 协议的三条解析规则：最终答案、动作名、动作入参。
# 冒号同时认中英文两种，因为模型两种都会写。
_FINAL_ANSWER_PATTERN = re.compile(r"Final\s*Answer\s*[:：]\s*(.*)", re.IGNORECASE | re.DOTALL)
_ACTION_PATTERN = re.compile(r"Action\s*[:：]\s*([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
_ACTION_INPUT_PATTERN = re.compile(r"Action\s*Input\s*[:：]\s*(.*)", re.IGNORECASE | re.DOTALL)


def strip_fence(text: str) -> str:
    """剥掉模型输出里的代码围栏

    模型常把 JSON 包在 ```json 里，解析前必须先剥掉。
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    return cleaned


def parse_tool_input(raw: str, text: str) -> Any:
    """Action Input 按 JSON 对象解析，剥掉代码围栏后解析失败再退回标量"""
    cleaned = strip_fence(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        if cleaned and "{" not in cleaned:
            return cleaned
        raise ValueError(f"Action Input 不是合法 JSON，原始内容：{cleaned}\n模型输出：\n{text}")


def parse_react_output(text: str) -> Tuple[str, Any]:
    """解析 ReAct 协议输出，返回 ("finish", 答复) 或 ("action", (工具名, 参数))

    认不出时抛错并带上原文，不做静默降级猜测：静默降级会让链路悄悄跑偏，
    错误暴露在当场才好排查。
    """
    # 模型常把关键字写成 **Action:**，去掉加粗标记后再匹配
    cleaned = text.replace("**", "")

    final_match = _FINAL_ANSWER_PATTERN.search(cleaned)
    if final_match:
        answer = final_match.group(1).strip()
        if not answer:
            raise ValueError(f"Final Answer 内容为空，模型原始输出：\n{text}")
        return "finish", answer

    action_match = _ACTION_PATTERN.search(cleaned)
    if not action_match:
        raise ValueError(
            "模型输出既没有 Action 也没有 Final Answer，无法继续执行。"
            f"请检查提示词与模型输出。原始输出：\n{text}"
        )

    input_match = _ACTION_INPUT_PATTERN.search(cleaned)
    if not input_match:
        raise ValueError(f"模型输出了 Action 但缺少 Action Input，原始输出：\n{text}")

    tool_input = parse_tool_input(input_match.group(1).strip(), text)
    return "action", (action_match.group(1).strip(), tool_input)


# ==========================================================================
# 知识边界
#
# 三层防线：
#   1. 入口预检：职责范围外的话题不进模型，直接给出明确答复
#   2. 工具硬校验：互斥参数、不可控设备、离线设备、不存在的空间，全部返回非 0 码
#   3. 提示词约束：要求模型只转述工具返回，禁止用经验补全
# ==========================================================================

OUT_OF_SCOPE_RULES: List[Tuple[Tuple[str, ...], str]] = [
    (
        ("正文", "全文", "写了什么", "内容是什么", "帮我读", "翻译"),
        "档案正文内容不在值守范围：系统只维护档号、题名、类型和架位信息。"
        "如需查阅正文，请按检索到的架位到库房调阅原件。",
    ),
    (
        ("制度", "法律责任", "合规意见", "审计结论", "政策解读"),
        "制度解读与法律意见不在值守范围：值守助手只负责位置、设备、档案定位和库房环境。"
        "如需制度依据，建议咨询档案管理部门。",
    ),
    (
        ("排班", "考勤", "工资", "人事调动", "绩效考核"),
        "人事相关事项不在值守范围：这类信息不在值守助手的可查数据内。",
    ),
    (
        ("天气", "股票", "汇率"),
        "该问题与档案库房值守无关，值守助手只处理空间位置、设备控制、档案定位和库房环境四类问题。",
    ),
]


def check_scope(question: str) -> Optional[str]:
    """职责范围预检：命中越界话题返回固定话术，未命中返回 None"""
    text = question.strip()
    for keywords, reply in OUT_OF_SCOPE_RULES:
        if any(keyword in text for keyword in keywords):
            return reply
    return None


def guard_scope(question: str, label: str) -> Optional[str]:
    """三个架构共用的入口预检：命中越界话题时打印固定话术并返回，未命中返回 None

    label 说明这次拦截发生在哪条链路上，例如「不进入工具循环」。
    抽出来是为了让三条链路的拦截行为完全一致：命中就打印、就返回固定话术、
    就终止，不会有一条链路悄悄放行。
    """
    reply = check_scope(question)
    if reply is None:
        return None
    print(f"[边界预检] 命中职责范围外话题，{label}")
    print(f"[Final Answer]\n{reply}")
    return reply


# ==========================================================================
# 模型入口
#
# 三个架构共用同一个 OpenAI 兼容客户端，模型与端点通过环境变量覆盖。
# ==========================================================================


@lru_cache(maxsize=1)
def _build_client() -> OpenAI:
    """构造 OpenAI 兼容客户端，缺少密钥时直接报错，不做降级"""
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("未设置环境变量 DASHSCOPE_API_KEY，无法调用模型")
    # max_retries=0：额度或鉴权问题立即报错，不重试刷屏
    return OpenAI(api_key=DASHSCOPE_API_KEY, base_url=LLM_BASE_URL, max_retries=0)


def chat(prompt: str) -> str:
    """发起一次对话，返回模型输出的文本

    temperature=0 保证输出格式与 JSON 参数稳定；
    enable_thinking=False 关掉思考模式，省 token 并降低延迟。
    """
    response = _build_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        extra_body={"enable_thinking": False},
    )
    return response.choices[0].message.content or ""


def chat_with_system(system: str, user: str) -> str:
    """发起一次带 system 角色的对话，返回模型输出文本

    角色与知识边界放在 system 中，与每轮的 user 内容分离，
    避免多轮之后角色被冲淡。
    """
    response = _build_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
        extra_body={"enable_thinking": False},
    )
    return response.choices[0].message.content or ""


# ==========================================================================
# 交互入口
#
# 三个架构的命令行循环完全一致，差异只在开场说明与处理函数，
# 因此把循环本身收在这里，各项目只提供自己的说明文案与入口函数。
# ==========================================================================


def run_cli(
    title: str,
    handler,
    lines: Tuple[str, ...] = (),
    prompt: str = "请输入问题：",
    with_history: bool = False,
) -> None:
    """统一的命令行交互循环

    title        架构名，显示在标题栏
    handler      处理一次请求的可调用对象
    lines        开场后逐行打印的说明与示例
    prompt       输入提示符
    with_history 是否把历史传给 handler

    handler 的签名随 with_history 变化：
        True    handler(question, history) -> 答复，答复会被记入历史
        False   handler(question)，多用于产出报告这类不需要多轮的场景
    """
    print(f"=== 库房大屏助手（{title}）===")
    print(f"模型：{LLM_MODEL}")
    for line in lines:
        print(line)
    print("输入 退出 结束会话\n")

    history: List[Tuple[str, str]] = []
    while True:
        try:
            question = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出")
            break

        if not question:
            continue
        if question.lower() in ("退出", "exit", "quit"):
            print("已退出")
            break

        try:
            if with_history:
                answer = handler(question, history)
                history.append((question, answer))
            else:
                handler(question)
        except Exception as exc:
            print(f"\n[执行失败] {exc}")
        print("-" * 60)
