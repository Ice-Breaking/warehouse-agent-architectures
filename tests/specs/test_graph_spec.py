#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""规格六：两张状态图的结构

图的形状就是架构本身，改动节点或边等于改架构，必须显式确认。
这份规格把节点集与边集锁住：

    混合式   协调层一次分派，两条分支，深思熟虑侧是收集到分析到建议三段
    深思熟虑 五阶段线性串联，没有分支——线性正是它的特征，不是漏写
"""

import warehouse_deliberative_agent as delib
import warehouse_hybrid_agent as hybrid


def node_names(graph) -> set[str]:
    """去掉 langgraph 自带的 __start__ / __end__，只留业务节点"""
    return {name for name in graph.get_graph().nodes if not name.startswith("__")}


def edge_pairs(graph) -> set[tuple[str, str]]:
    return {(edge.source, edge.target) for edge in graph.get_graph().edges}


# ——— 混合式 ———

def test_hybrid_graph_should_have_seven_business_nodes():
    assert node_names(hybrid.build_graph()) == {
        "assess",
        "reactive_agent",
        "tools",
        "collect_energy",
        "analyze_energy",
        "recommend",
        "make_answer",
    }


def test_hybrid_graph_should_dispatch_from_assess_to_both_branches():
    edges = edge_pairs(hybrid.build_graph())
    assert ("assess", "reactive_agent") in edges
    assert ("assess", "collect_energy") in edges


def test_hybrid_graph_should_loop_between_reactive_agent_and_tools():
    """反应式分支保留模型自主循环：拿不到最终答案就继续调工具"""
    edges = edge_pairs(hybrid.build_graph())
    assert ("reactive_agent", "tools") in edges
    assert ("tools", "reactive_agent") in edges
    assert ("reactive_agent", "make_answer") in edges


def test_hybrid_graph_should_chain_deliberative_branch_in_three_steps():
    """数据收集 → 深度分析 → 生成建议，三段缺一不可"""
    edges = edge_pairs(hybrid.build_graph())
    assert ("collect_energy", "analyze_energy") in edges
    assert ("analyze_energy", "recommend") in edges
    assert ("recommend", "make_answer") in edges


def test_hybrid_graph_should_share_one_exit_node():
    """两条分支汇到同一个响应生成节点"""
    edges = edge_pairs(hybrid.build_graph())
    assert ("reactive_agent", "make_answer") in edges
    assert ("recommend", "make_answer") in edges
    assert ("make_answer", "__end__") in edges


def test_hybrid_graph_should_start_from_assess():
    assert ("__start__", "assess") in edge_pairs(hybrid.build_graph())


# ——— 深思熟虑 ———

def test_deliberative_graph_should_have_five_business_nodes():
    assert node_names(delib.build_graph()) == {
        "perception",
        "modeling",
        "reasoning",
        "decide",
        "write_report",
    }


def test_deliberative_graph_should_be_a_strict_linear_chain():
    """线性是深思熟虑的特征：阶段顺序由代码保证，模型无法跳过或回退"""
    assert edge_pairs(delib.build_graph()) == {
        ("__start__", "perception"),
        ("perception", "modeling"),
        ("modeling", "reasoning"),
        ("reasoning", "decide"),
        ("decide", "write_report"),
        ("write_report", "__end__"),
    }


def test_deliberative_graph_should_have_no_branch():
    """每个节点恰好一条出边，不存在条件分支"""
    edges = edge_pairs(delib.build_graph())
    business = {src for src, _ in edges if not src.startswith("__")}
    for source in business:
        outgoing = [tgt for src, tgt in edges if src == source]
        assert len(outgoing) == 1, f"{source} 有多条出边，出现了分支"


# ——— 两个图都应可编译 ———

def test_both_graphs_should_compile_to_runnable():
    assert hybrid.build_graph().invoke is not None
    assert delib.build_graph().invoke is not None
