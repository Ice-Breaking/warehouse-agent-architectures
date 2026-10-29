#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
库房大屏助手 - 混合式架构

入口先用一次模型调用做路由分类，之后走两条完全不同的控制流：

    reactive       单目标查询。交回模型自主决定调哪个工具，代码只给工具集合与终止条件。
    deliberative   跨库房对比分析。代码写死采集与对比两步，数据必须来自真实工具调用，
                   模型只负责归纳，不得补充任何未出现的数字。

与反应式的差别在于：反应式把控制流全程交给模型，混合式只对简单查询交出去，
复杂请求改由代码编排，避免模型在多步采集里漏采或编数。

域数据、工具集合、职责边界与模型入口来自仓库根目录的 warehouse_data。

运行（在仓库根目录或本项目目录下均可）：
    export DASHSCOPE_API_KEY=sk-xxx
    python 02-hybrid/warehouse_hybrid_agent.py
"""

import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Tuple, TypedDict

from langgraph.graph import END, StateGraph

# 允许在仓库根目录或本项目目录下直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from warehouse_data import (
    STORAGE_ENV,
    TOOL_MAP,
    _build_client,
    build_tool_descriptions,
    check_scope,
    query_energy_stat,
    query_environment,
    LLM_MODEL,
)

# 反应式分支的最大工具调用轮次，超过即抛错，不靠框架的递归上限兜底
MAX_REACTIVE_STEPS = 6

_FINAL_ANSWER_PATTERN = re.compile(r"Final\s*Answer\s*[:：]\s*(.*)", re.IGNORECASE | re.DOTALL)
_ACTION_PATTERN = re.compile(r"Action\s*[:：]\s*([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
_ACTION_INPUT_PATTERN = re.compile(r"Action\s*Input\s*[:：]\s*(.*)", re.IGNORECASE | re.DOTALL)

# 角色与知识边界放在 system 中，与每轮的 user 内容分离，避免多轮后角色被冲淡
SYSTEM_PROMPT = """你是档案库房智慧大屏的值守助手，负责回答四类问题：空间位置查询、设备状态与控制、档案存放位置检索、库房能源环境。

知识边界，必须遵守：
1. 位置只认工具返回的空间树，不得推测或编造楼层、库房、库区、密集架编号。
2. 设备控制只支持断路器合闸分闸、门禁开关门。节能模式、自动模式、定时策略等模式类动作不在支持范围，遇到时明确说明不支持，不得表示已执行。
3. 档案检索按档号或按类型二选一，不支持同时按两者查询，也不支持读取档案正文。
4. 工具返回 code 非 0 时，如实转述失败原因，不得改写成成功。
5. 工具没有返回的信息，回答"系统中未查询到"，不要用一般性经验补充。"""

ROUTE_PROMPT = """判断下面这个问题该交给哪条链路处理，只回答一个单词，不要输出其他内容。

reactive：单目标查询，一两次工具调用就能回答。例如某个库区在哪、某台设备什么状态、
某份档案放在哪、某个库房的能耗或温湿度是多少。

deliberative：需要跨多个库房采集数据再做对比分析的请求。例如对比各库房能耗、
全楼能耗排名、多个库房的环境差异、找出能耗异常或环境告警的库房。

问题：{question}"""

REACTIVE_PROMPT = """可用工具（工具名必须原样使用，区分大小写）：
{tool_descriptions}

可用工具名：{tool_names}

输出格式（每轮只能有一个 Thought 和一个 Action，或一个 Final Answer）：
Thought: 你的思考
Action: 工具名
Action Input: {{"参数名": "参数值"}}

信息足够时输出：
Thought: 已经可以回答
Final Answer: 最终答复

调用要求：
1. Action Input 必须是 JSON 对象，键名与工具参数名完全一致。
2. 一次只调用一个工具，拿到 Observation 后再决定下一步。

历史对话：
{history}

当前问题：{question}

{scratchpad}"""

COMPARE_PROMPT = """以下是系统真实采集到的库房数据，未经任何加工。

能耗数据（全楼汇总与各库房明细）：
{energy_data}

环境数据（各库房温湿度与告警）：
{env_data}

用户的问题是：{question}

请基于以上数据回答，要求：
1. 只能引用上面出现过的数字，不得补充、推算或编造任何未出现的数字。
2. 明确指出能耗最高与最低的库房，以及存在环境告警的库房。
3. 给出结论与可执行的建议，控制在 300 字以内。"""


class HybridState(TypedDict):
    """混合式的共享状态，每个节点只回写自己产出的字段"""

    question: str
    history: str
    route: str
    steps: int
    scratchpad: str
    pending_tool: str
    pending_input: Any
    final_answer: str
    energy_data: str
    env_data: str
    analysis: str
    answer: str


def _chat_with_system(system: str, user: str) -> str:
    """发起一次带 system 角色的对话，返回模型输出文本"""
    response = _build_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
        extra_body={"enable_thinking": False},
    )
    return response.choices[0].message.content or ""


def _parse_output(text: str) -> Tuple[str, Any]:
    """解析模型输出，返回 ("finish", 答复) 或 ("action", (工具名, 参数))

    解析不出时抛异常并附带原文，不做静默猜测。
    """
    cleaned = text.replace("**", "")
    final_match = _FINAL_ANSWER_PATTERN.search(cleaned)
    if final_match:
        answer = final_match.group(1).strip()
        if not answer:
            raise ValueError(f"Final Answer 内容为空，模型原始输出：\n{text}")
        return "finish", answer

    action_match = _ACTION_PATTERN.search(cleaned)
    if not action_match:
        raise ValueError(f"模型输出既没有 Action 也没有 Final Answer，原始输出：\n{text}")

    input_match = _ACTION_INPUT_PATTERN.search(cleaned)
    if not input_match:
        raise ValueError(f"模型输出了 Action 但缺少 Action Input，原始输出：\n{text}")

    return "action", (action_match.group(1).strip(), _parse_tool_input(input_match.group(1).strip(), text))


def _parse_tool_input(raw: str, text: str) -> Any:
    """Action Input 按 JSON 对象解析，去掉代码围栏后解析失败再退回标量"""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        if cleaned and "{" not in cleaned:
            return cleaned
        raise ValueError(f"Action Input 不是合法 JSON，原始内容：{cleaned}\n模型输出：\n{text}")


def assess(state: HybridState) -> dict:
    """路由分类节点：判定问题走反应式还是固定分析流程"""
    raw = _chat_with_system(SYSTEM_PROMPT, ROUTE_PROMPT.format(question=state["question"]))
    lowered = raw.strip().lower()
    print(f"[路由分类] {raw.strip()}")

    # 不做静默降级：拿不到明确的分类就抛错，而不是默认走反应式
    if "deliberative" in lowered and "reactive" not in lowered:
        route = "deliberative"
    elif "reactive" in lowered and "deliberative" not in lowered:
        route = "reactive"
    else:
        raise ValueError(f"路由分类结果不是 reactive 或 deliberative，模型原始输出：\n{raw}")

    return {"route": route}


def pick_route(state: HybridState) -> str:
    """路由分支依据"""
    return state["route"]


def reactive_agent(state: HybridState) -> dict:
    """反应式分支：模型自主决定下一步，代码只负责解析与回灌"""
    if state["steps"] >= MAX_REACTIVE_STEPS:
        raise RuntimeError(f"反应式分支已调用 {state['steps']} 轮工具，超过上限 {MAX_REACTIVE_STEPS}，终止")

    user_prompt = REACTIVE_PROMPT.format(
        tool_descriptions=build_tool_descriptions(),
        tool_names=", ".join(TOOL_MAP),
        history=state["history"],
        question=state["question"],
        scratchpad=state["scratchpad"] or "（当前还没有调用过工具）",
    )
    raw = _chat_with_system(SYSTEM_PROMPT, user_prompt)
    round_no = state["steps"] + 1
    print(f"\n[反应式分支 第 {round_no} 轮]\n{raw.strip()}")

    kind, payload = _parse_output(raw)
    if kind == "finish":
        return {"final_answer": payload, "scratchpad": state["scratchpad"] + f"{raw.strip()}\n"}

    tool_name, tool_input = payload
    return {
        "steps": round_no,
        "pending_tool": tool_name,
        "pending_input": tool_input,
        "scratchpad": state["scratchpad"] + f"{raw.strip()}\n",
    }


def execute_tools(state: HybridState) -> dict:
    """执行上一步解析出的工具调用，把返回值回灌进 scratchpad"""
    tool_item = TOOL_MAP.get(state["pending_tool"])
    if tool_item is None:
        observation = f"工具 {state['pending_tool']} 不存在，可用工具：{', '.join(TOOL_MAP)}"
    else:
        observation = str(tool_item.invoke(state["pending_input"]))

    print(f"[Action] {state['pending_tool']}")
    print(f"[Action Input] {json.dumps(state['pending_input'], ensure_ascii=False)}")
    print(f"[Observation] {observation}")

    return {
        "scratchpad": state["scratchpad"] + f"Observation: {observation}\n",
        "pending_tool": "",
        "pending_input": {},
    }


def after_reactive(state: HybridState) -> str:
    """反应式分支的循环出口：有最终答案就收尾，否则继续调工具"""
    return "answer" if state["final_answer"] else "tools"


def collect_energy(state: HybridState) -> dict:
    """固定分析分支第一步：真实调用工具采集数据，不让模型生成"""

    energy = str(query_energy_stat.invoke({"storage_name": "全部"}))

    env_snippets: List[str] = []
    for key in STORAGE_ENV:
        floor_no, name = key.split("-", 1)
        env_snippets.append(str(query_environment.invoke({"storage_name": f"{floor_no}楼{name}"})))
    env_data = "\n".join(env_snippets)

    print(f"[采集能耗] {energy[:200]}")
    print(f"[采集环境] 共 {len(env_snippets)} 个库房")
    for snippet in env_snippets:
        print(f"  {snippet[:160]}")

    return {"energy_data": energy, "env_data": env_data}


def compare_energy(state: HybridState) -> dict:
    """固定分析分支第二步：基于已采集的真实数据做对比"""
    user_prompt = COMPARE_PROMPT.format(
        energy_data=state["energy_data"],
        env_data=state["env_data"],
        question=state["question"],
    )
    analysis = _chat_with_system(SYSTEM_PROMPT, user_prompt).strip()
    print(f"\n[对比分析]\n{analysis}")
    return {"analysis": analysis}


def make_answer(state: HybridState) -> dict:
    """汇总最终答复"""
    answer = state["final_answer"] if state["route"] == "reactive" else state["analysis"]
    print(f"\n[Final Answer]\n{answer}")
    return {"answer": answer}


@lru_cache(maxsize=1)
def build_graph():
    """组装混合式状态图

    assess 只做一次分类，之后的路径是确定的：
    reactive 分支保留模型自主循环，deliberative 分支是采集到对比的固定两步。
    """
    workflow = StateGraph(HybridState)
    workflow.add_node("assess", assess)
    workflow.add_node("reactive_agent", reactive_agent)
    workflow.add_node("tools", execute_tools)
    workflow.add_node("collect_energy", collect_energy)
    workflow.add_node("compare_energy", compare_energy)
    workflow.add_node("make_answer", make_answer)

    workflow.set_entry_point("assess")
    workflow.add_conditional_edges("assess", pick_route, {"reactive": "reactive_agent", "deliberative": "collect_energy"})
    workflow.add_conditional_edges("reactive_agent", after_reactive, {"answer": "make_answer", "tools": "tools"})
    workflow.add_edge("tools", "reactive_agent")
    workflow.add_edge("collect_energy", "compare_energy")
    workflow.add_edge("compare_energy", "make_answer")
    workflow.add_edge("make_answer", END)

    return workflow.compile()


def _format_history(history: List[Tuple[str, str]]) -> str:
    """把多轮历史拼成提示词片段，只保留最近五轮"""
    if not history:
        return "无"
    return "\n".join(f"用户：{asked}\n助手：{answered}" for asked, answered in history[-5:])


def run_hybrid_agent(question: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
    """执行一次混合式推理，返回最终答复"""
    scope_reply = check_scope(question)
    if scope_reply is not None:
        print("[边界预检] 命中职责范围外话题，直接答复，不进入路由")
        print(f"[Final Answer]\n{scope_reply}")
        return scope_reply

    initial: HybridState = {
        "question": question,
        "history": _format_history(history or []),
        "route": "",
        "steps": 0,
        "scratchpad": "",
        "pending_tool": "",
        "pending_input": {},
        "final_answer": "",
        "energy_data": "",
        "env_data": "",
        "analysis": "",
        "answer": "",
    }
    result = build_graph().invoke(initial)
    return result["answer"]


def main() -> None:
    print("=== 库房大屏助手（混合式）===")
    print(f"模型：{LLM_MODEL}")
    print("简单查询走反应式分支，跨库房对比分析走固定采集流程")
    print("示例：3楼2号库房的低氧库区在什么位置")
    print("示例：对比一下各库房的能耗情况")
    print("输入 退出 结束会话\n")

    history: List[Tuple[str, str]] = []
    while True:
        try:
            question = input("请输入问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出")
            break

        if not question:
            continue
        if question.lower() in ("退出", "exit", "quit"):
            print("已退出")
            break

        try:
            answer = run_hybrid_agent(question, history)
            history.append((question, answer))
        except Exception as exc:
            print(f"\n[执行失败] {exc}")
        print("-" * 60)


if __name__ == "__main__":
    main()
