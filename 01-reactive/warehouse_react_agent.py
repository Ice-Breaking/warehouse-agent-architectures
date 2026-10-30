#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
库房大屏助手 - 反应式架构

控制流完全交给模型：每一轮由模型观察上一步的工具返回，自主决定下一步调用哪个工具，
直到它自己输出 Final Answer。代码只提供工具集合、输出解析器与终止条件，不预设路径。

四个步骤在本仓库里的落点：

    Step1 数据准备   warehouse_data.py  空间树、设备台账、档案台账、能源环境
    Step2 工具设计   warehouse_data.py  六个工具，内部完成参数校验与存在性预检
    Step3 Agent 搭建 本文件            提示词模板 + 输出解析器 + 执行循环
    Step4 知识边界   三层配合          warehouse_data 的 check_scope 入口预检、
                                      工具层非 0 响应码硬校验、本文件提示词约束

域数据、工具集合、职责边界与模型入口来自仓库根目录的 warehouse_data，
本项目只实现反应式的控制流。

langchain 1.x 已移除 AgentExecutor，因此执行环节按 ReAct 协议自行实现，
每一步的 Thought / Action / Observation 都打印到终端，便于核查结论来源。

运行（在仓库根目录或本项目目录下均可）：
    export DASHSCOPE_API_KEY=sk-xxx
    python 01-reactive/warehouse_react_agent.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, List, Tuple, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import ValidationError

# 允许在仓库根目录或本项目目录下直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from warehouse_data import TOOL_MAP, _chat, build_tool_descriptions, check_scope, LLM_MODEL


# ==========================================================================
# Step3 Agent 搭建
#
# 三件套，落点依次如下：
#   1. 提示词模板   FORMAT_EXAMPLE（格式示例）+ REACT_TEMPLATE（主模板）+ REACT_PROMPT
#   2. 输出解析器   ReActOutputParser，把模型文本解析为 AgentAction 或 AgentFinish
#   3. 执行循环     run_react_agent，调用模型 -> 解析 -> 执行工具 -> 回灌观察
# ==========================================================================

# 提示词模板之一：给模型看的调用格式样例
FORMAT_EXAMPLE = """Thought: 需要先确认库区在空间树中的标准名称
Action: resolve_scene_target
Action Input: {"target_text": "3楼2号库房低氧区"}
Observation: {"code": 0, "data": {"path": "档案综合楼/三层/2号库房/低氧库区", "racks": ["3-2-LO-01", "3-2-LO-02"]}}
Thought: 已经拿到标准路径，可以回答
Final Answer: 低氧库区位于档案综合楼三层2号库房，包含 3-2-LO-01、3-2-LO-02 两组密集架。"""

REACT_TEMPLATE = """你是档案库房智慧大屏的值守助手，负责回答四类问题：空间位置查询、设备状态与控制、档案存放位置检索、库房能源环境。

可用工具（工具名必须原样使用，区分大小写）：
{tool_descriptions}

可用工具名：{tool_names}

输出格式（每轮只能有一个 Thought 和一个 Action，或一个 Final Answer）：
{format_example}

调用要求：
1. Action Input 必须是 JSON 对象，键名与工具参数名完全一致。
2. 一次只调用一个工具，拿到 Observation 后再决定下一步，不要一次输出多个 Action。
3. 信息足够时直接输出 "Final Answer: " 加最终答复，不要继续调用工具。

知识边界，必须遵守：
1. 位置只认工具返回的空间树，不得推测或编造楼层、库房、库区、密集架编号。
2. 设备控制只支持断路器合闸分闸、门禁开关门。节能模式、自动模式、定时策略等模式类动作不在支持范围，遇到时明确说明不支持，不得表示已执行。
3. 档案检索按档号或按类型二选一，不支持同时按两者查询，也不支持读取档案正文。
4. 工具返回 code 非 0 时，如实转述失败原因，不得改写成成功。
5. 工具没有返回的信息，回答"系统中未查询到"，不要用一般性经验补充。
6. 涉及制度解读、法律意见、人事安排等超出值守范围的问题，说明不在职责范围并给出可查询的方向。

历史对话：
{history}

当前问题：{question}

{agent_scratchpad}"""

REACT_PROMPT = PromptTemplate.from_template(REACT_TEMPLATE)

_FINAL_ANSWER_PATTERN = re.compile(r"Final\s*Answer\s*[:：]\s*(.*)", re.IGNORECASE | re.DOTALL)
_ACTION_PATTERN = re.compile(r"Action\s*[:：]\s*([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
_ACTION_INPUT_PATTERN = re.compile(r"Action\s*Input\s*[:：]\s*(.*)", re.IGNORECASE | re.DOTALL)


# 输出解析器：把模型文本解析为 AgentAction 或 AgentFinish
class ReActOutputParser(BaseOutputParser[Union[AgentAction, AgentFinish]]):
    """解析 ReAct 文本输出，产出 AgentAction 或 AgentFinish。

    解析不出动作或最终答案时抛出异常并附带原文，不做静默降级猜测。
    """

    @property
    def _type(self) -> str:
        return "react"

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        # 模型常把关键字写成 **Action:**，去掉加粗标记后再匹配
        text = text.replace("**", "")
        final_match = _FINAL_ANSWER_PATTERN.search(text)
        if final_match:
            answer = final_match.group(1).strip()
            if not answer:
                raise ValueError(f"Final Answer 内容为空，模型原始输出：\n{text}")
            return AgentFinish(return_values={"output": answer}, log=text)

        action_match = _ACTION_PATTERN.search(text)
        if not action_match:
            raise ValueError(
                "模型输出既没有 Action 也没有 Final Answer，无法继续执行。"
                f"请检查提示词与模型输出。原始输出：\n{text}"
            )

        input_match = _ACTION_INPUT_PATTERN.search(text)
        if not input_match:
            raise ValueError(f"模型输出了 Action 但缺少 Action Input，原始输出：\n{text}")

        tool_input = self._parse_tool_input(input_match.group(1).strip(), text)
        return AgentAction(tool=action_match.group(1).strip(), tool_input=tool_input, log=text)

    @staticmethod
    def _parse_tool_input(raw: str, text: str) -> Any:
        """Action Input 统一按 JSON 对象解析，去掉代码围栏后再解析"""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
            cleaned = re.sub(r"```\s*$", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # 单参数工具允许直接给标量文本
            if cleaned and "{" not in cleaned:
                return cleaned
            raise ValueError(f"Action Input 不是合法 JSON，原始内容：{cleaned}\n模型输出：\n{text}")


def _execute_tool(action: AgentAction) -> str:
    """执行工具，把参数错误回灌给模型自纠，工具内部的硬错误照常抛出"""
    tool_item = TOOL_MAP.get(action.tool)
    if tool_item is None:
        return f"工具 {action.tool} 不存在，可用工具：{', '.join(TOOL_MAP)}"
    try:
        return str(tool_item.invoke(action.tool_input))
    except ValidationError as exc:
        return f"参数不合法，请按工具参数名重新调用。错误详情：{exc}"


def _format_history(history: List[Tuple[str, str]]) -> str:
    """把多轮历史拼成提示词片段，只保留最近五轮"""
    if not history:
        return "无"
    recent = history[-5:]
    return "\n".join(f"用户：{asked}\n助手：{answered}" for asked, answered in recent)


# 执行循环：调用模型 -> 解析 -> 执行工具 -> 回灌观察，直到 AgentFinish 或超过轮次上限
def run_react_agent(question: str, history: List[Tuple[str, str]], max_iterations: int = 6) -> str:
    """执行一次反应式推理，返回最终答复"""
    scope_reply = check_scope(question)
    if scope_reply is not None:
        print("[边界预检] 命中职责范围外话题，直接答复，不进入工具循环")
        print(f"[Final Answer]\n{scope_reply}")
        return scope_reply

    parser = ReActOutputParser()
    tool_descriptions = build_tool_descriptions()
    history_text = _format_history(history)
    scratchpad = ""
    raw = ""

    for step in range(1, max_iterations + 1):
        prompt = REACT_PROMPT.format(
            tool_descriptions=tool_descriptions,
            tool_names=", ".join(TOOL_MAP),
            format_example=FORMAT_EXAMPLE,
            history=history_text,
            question=question,
            agent_scratchpad=scratchpad,
        )
        raw = _chat(prompt)
        print(f"\n[第 {step} 轮模型输出]\n{raw.strip()}")

        parsed = parser.parse(raw)
        if isinstance(parsed, AgentFinish):
            print(f"\n[Final Answer]\n{parsed.return_values['output']}")
            return parsed.return_values["output"]

        observation = _execute_tool(parsed)
        print(f"[Action] {parsed.tool}")
        print(f"[Action Input] {json.dumps(parsed.tool_input, ensure_ascii=False)}")
        print(f"[Observation] {observation}")
        scratchpad += f"{raw.strip()}\nObservation: {observation}\n"

    raise RuntimeError(f"超过最大工具调用轮次 {max_iterations}，已终止。最后一次模型输出：\n{raw}")


def main() -> None:
    print("=== 库房大屏助手（反应式）===")
    print(f"模型：{LLM_MODEL}")
    print("可查询：空间位置、设备状态与控制、档案存放位置、库房能源环境")
    print("示例：3楼2号库房的低氧库区在什么位置")
    print("示例：把3楼2号库房V6库区的断路器分闸")
    print("示例：查一下题名带决算的档案放在哪")
    print("示例：3楼2号库房这个月的能耗是多少")
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
            answer = run_react_agent(question, history)
            history.append((question, answer))
        except Exception as exc:
            print(f"\n[执行失败] {exc}")
        print("-" * 60)


if __name__ == "__main__":
    main()
