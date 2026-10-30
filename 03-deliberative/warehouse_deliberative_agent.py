#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
库房大屏助手 - 深思熟虑架构

生成季度库房巡检报告。控制流完全由代码写死，五个阶段依次执行，模型不能跳过、
不能回退、也不能临时决定去调某个工具：

    感知 perception   代码调用真实工具采集设备、环境、能耗数据
    建模 modeling     把原始数据整理成风险矩阵
    推理 reasoning    提出三套整改方案
    决策 decision     选定一套并说明取舍
    成文 report       输出正式报告

五阶段与深思熟虑架构各要素的对应关系：

    Sensors              perception      采集设备、环境、能耗三类原始数据
    World Model / Memory modeling        把原始数据固化成风险矩阵，作为后续唯一依据
    Planner / Reasoner   reasoning       先生成三套投入不同的整改方案
                         decision        再选定一套，说明取舍与切换条件
    Actuators            write_report    输出可直接归档、可落到运维动作的报告

与反应式的差别在于：反应式每一步都由模型临场决定，深思熟虑把流程固定下来，
模型只在每个阶段内部负责内容，阶段之间的顺序与数据流向由代码保证。
好处是报告结构稳定、不会漏采；代价是无法中途转向，遇到意外情况只能整体失败。

域数据、工具集合、职责边界与模型入口来自仓库根目录的 warehouse_data。

运行（在仓库根目录或本项目目录下均可）：
    export DASHSCOPE_API_KEY=sk-xxx
    python 03-deliberative/warehouse_deliberative_agent.py
"""

import sys
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

# 允许在仓库根目录或本项目目录下直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warehouse_data as wd

SYSTEM_PROMPT = """你是档案库房智慧大屏的值守助手，负责把巡检数据整理成可执行的结论。

铁律：只能使用给定素材中出现过的数字、设备名与库房名，不得补充、推算或编造任何未出现的内容。
素材里没有的项目，明确写"本次巡检未涉及"或"正常"，不要凭经验填充。"""

MODELING_PROMPT = """以下是本次巡检采集到的原始数据，未经任何加工：

{perception}

请建立风险矩阵，要求：
1. 按「环境风险」「设备风险」「能耗风险」三类分别列出发现的问题。
2. 每条风险标注：涉及对象、当前数值、标准或参照范围、风险等级（高/中/低）。
3. 没有发现问题的类别要明确写"正常"，不要省略。
4. 只使用上面出现过的数字与名称。"""

REASONING_PROMPT = """风险矩阵：

{risk_model}

请提出三套整改方案，每套包含：适用范围、具体措施、预期效果、代价与局限。

三套方案必须在投入与见效速度上有明显区分：
    方案一 立即处置：低成本、见效快，用于先把高风险压下去
    方案二 分批改造：中等投入，用于在一个季度内系统性解决
    方案三 长期优化：投入较大，用于从根本上降低风险与能耗

不得引入风险矩阵之外的新数据。"""

DECISION_PROMPT = """风险矩阵：

{risk_model}

三套整改方案：

{plans}

请选定一套并给出决策理由：
1. 当前条件下为什么这一套最合适。
2. 选择它意味着放弃了什么。
3. 什么情况下应当切换到另一套。

只输出选定方案与理由，不要再提出新的方案。"""

REPORT_PROMPT = """请把本次巡检整理成一份正式报告，结构如下：

一、巡检概况（范围、数据来源与采集方式）
二、风险矩阵
三、整改方案与决策
四、后续跟踪建议

素材：
巡检诉求：{question}
风险矩阵：{risk_model}
决策结论：{decision}

要求：
1. 只使用素材中出现过的内容，不得补充任何未出现的数字、设备或库房。
2. 语言正式、可直接归档，不要出现"可能""大概"这类模糊表述。
3. 段落之间空一行，不要输出 markdown 标题符号以外的装饰字符。"""


class DelibState(TypedDict):
    """深思熟虑的共享状态，每个阶段只回写自己产出的字段"""

    question: str
    perception: str
    risk_model: str
    plans: str
    decision: str
    report: str


def perception(state: DelibState) -> dict:
    """第一阶段：代码调用真实工具采集数据，模型不参与

    这一步是整条流水线可信度的基础：后续所有阶段只能引用这里采集到的原文。
    """
    devices = str(wd.query_device_status.invoke({"keyword": ""}))
    energy = str(wd.query_energy_stat.invoke({"storage_name": "全部"}))

    env_snippets = wd.collect_environment_all()

    perception_text = (
        f"【设备台账】\n{devices}\n\n"
        f"【能耗数据】\n{energy}\n\n"
        f"【环境数据】\n" + "\n".join(env_snippets)
    )

    print(f"[阶段 1/5 感知] 采集设备 {len(devices)} 字符、环境 {len(env_snippets)} 个库房、能耗汇总 1 份")
    return {"perception": perception_text}


def modeling(state: DelibState) -> dict:
    """第二阶段：把原始数据整理成风险矩阵"""
    risk_model = wd.chat_with_system(SYSTEM_PROMPT, MODELING_PROMPT.format(perception=state["perception"])).strip()
    print(f"\n[阶段 2/5 建模]\n{risk_model}")
    return {"risk_model": risk_model}


def reasoning(state: DelibState) -> dict:
    """第三阶段：提出三套整改方案"""
    plans = wd.chat_with_system(SYSTEM_PROMPT, REASONING_PROMPT.format(risk_model=state["risk_model"])).strip()
    print(f"\n[阶段 3/5 推理]\n{plans}")
    return {"plans": plans}


def decide(state: DelibState) -> dict:
    """第四阶段：选定一套方案并说明取舍"""
    decision = wd.chat_with_system(
        SYSTEM_PROMPT,
        DECISION_PROMPT.format(risk_model=state["risk_model"], plans=state["plans"]),
    ).strip()
    print(f"\n[阶段 4/5 决策]\n{decision}")
    return {"decision": decision}


def write_report(state: DelibState) -> dict:
    """第五阶段：成文"""
    report = wd.chat_with_system(
        SYSTEM_PROMPT,
        REPORT_PROMPT.format(
            question=state["question"],
            risk_model=state["risk_model"],
            decision=state["decision"],
        ),
    ).strip()
    print(f"\n[阶段 5/5 成文]\n{report}")
    return {"report": report}


@lru_cache(maxsize=1)
def build_graph():
    """组装深思熟虑状态图

    五个节点线性串联，没有条件分支。这不是漏写——线性正是深思熟虑的特征：
    阶段顺序由代码保证，模型无法跳过或回退，失败就整体终止。
    """
    workflow = StateGraph(DelibState)
    workflow.add_node("perception", perception)
    workflow.add_node("modeling", modeling)
    workflow.add_node("reasoning", reasoning)
    workflow.add_node("decide", decide)
    workflow.add_node("write_report", write_report)

    workflow.set_entry_point("perception")
    workflow.add_edge("perception", "modeling")
    workflow.add_edge("modeling", "reasoning")
    workflow.add_edge("reasoning", "decide")
    workflow.add_edge("decide", "write_report")
    workflow.add_edge("write_report", END)

    return workflow.compile()


def run_deliberative_agent(question: str) -> str:
    """执行一次深思熟虑推理，返回巡检报告全文"""
    scope_reply = wd.guard_scope(question, "直接答复，不进入流水线")
    if scope_reply is not None:
        return scope_reply

    initial: DelibState = {
        "question": question,
        "perception": "",
        "risk_model": "",
        "plans": "",
        "decision": "",
        "report": "",
    }
    result = build_graph().invoke(initial)
    return result["report"]


def main() -> None:
    wd.run_cli(
        "深思熟虑",
        run_deliberative_agent,
        lines=(
            "按固定五阶段生成巡检报告：感知 -> 建模 -> 推理 -> 决策 -> 成文",
            "示例：生成一份全楼库房的季度巡检报告",
        ),
        prompt="请输入巡检诉求：",
    )


if __name__ == "__main__":
    main()
