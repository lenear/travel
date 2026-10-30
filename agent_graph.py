"""
LangGraph 驱动的模型—工具调用循环

用 StateGraph 重新实现 prompt_engineering.PromptBuilder._run_model_with_tools
的核心流程：
    调用模型 → 如果返回 tool_calls，执行工具并把结果加入消息历史 → 再调用模型
    → 如果返回最终回答，结束并返回原始回答文本

保留现有限制：
    - 工具调用轮数上限 MAX_TOOL_ROUNDS
    - 单轮工具数量上限 MAX_TOOL_CALLS_PER_ROUND
    - 轮数耗尽后不提供工具再请求一次最终回答
    - 工具执行失败也作为工具结果反馈给模型

每个节点只调用一次模型或执行一次工具，不把整个 _run_model_with_tools 包进节点。

本模块暂不接入 Flask 页面、行程校验、自动重规划或交通耗时新功能。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from langgraph.graph import StateGraph, END

from prompt_engineering import (
    MAX_TOOL_ROUNDS,
    MAX_TOOL_CALLS_PER_ROUND,
    PromptBuilder,
)
from travel_tools import TOOL_REGISTRY, TOOL_DEFINITIONS


# ---------- State ----------

class AgentState(dict):
    """LangGraph 状态：消息历史 + 工具调用轮数计数器"""

    messages: List[Dict[str, Any]]
    tool_round: int          # 已完成的工具调用轮数（从 0 开始）
    max_tool_rounds: int     # 上限，默认 MAX_TOOL_ROUNDS


def _initial_state(
    messages: List[Dict[str, Any]],
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
) -> AgentState:
    return AgentState(
        messages=list(messages),
        tool_round=0,
        max_tool_rounds=max(0, min(int(max_tool_rounds), MAX_TOOL_ROUNDS)),
    )


# ---------- Helpers (复用现有逻辑) ----------

def _field(obj: Any, key: str, default: Any = None) -> Any:
    """安全读取字典字段"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def _build_tool_schemas() -> List[Dict[str, Any]]:
    """把 TOOL_DEFINITIONS 转成 DashScope tools 格式"""
    return [
        {"type": "function", "function": definition}
        for definition in TOOL_DEFINITIONS.values()
    ]


def _tool_message(tool_call_id: str, result: Dict[str, Any]) -> Dict[str, str]:
    """工具结果 → tool 角色消息"""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(result, ensure_ascii=False),
    }


def _normalize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    """把模型返回的 tool_calls 统一成 OpenAI 风格"""
    normalized: List[Dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls or []):
        function = _field(tool_call, "function", {}) or {}
        arguments = _field(function, "arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        normalized.append({
            "id": _field(tool_call, "id", "") or f"call_{index}",
            "type": "function",
            "function": {
                "name": _field(function, "name", "") or "",
                "arguments": arguments,
            },
        })
    return normalized


def _execute_tool_call(tool_call: Dict[str, Any]) -> Dict[str, str]:
    """执行单个工具调用，任何失败都作为工具结果返回（不抛异常）"""
    call_id = tool_call.get("id", "")
    function = tool_call.get("function", {})
    name = function.get("name", "")
    raw_arguments = function.get("arguments", "{}")

    func = TOOL_REGISTRY.get(name)
    if func is None:
        return _tool_message(call_id, {
            "ok": False,
            "error": f"工具未注册：{name}，可用工具：{list(TOOL_REGISTRY)}",
        })

    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except (json.JSONDecodeError, TypeError):
        return _tool_message(call_id, {
            "ok": False,
            "error": f"工具{name}的参数不是合法JSON",
        })
    if not isinstance(arguments, dict):
        return _tool_message(call_id, {
            "ok": False,
            "error": f"工具{name}的参数必须是JSON对象",
        })

    try:
        result = func(**arguments)
    except TypeError as e:
        result = {"ok": False, "error": f"工具{name}参数不匹配：{e}"}
    except Exception as e:
        result = {"ok": False, "error": f"工具{name}执行异常：{e}"}

    if not isinstance(result, dict):
        result = {"ok": False, "error": f"工具{name}返回结果格式异常"}
    return _tool_message(call_id, result)


# ---------- Graph Nodes ----------

def call_model_node(state: AgentState, *, builder: PromptBuilder) -> AgentState:
    """调用一次模型（带工具定义）。结果追加到 messages。"""
    messages = state["messages"]
    tool_schemas = _build_tool_schemas()
    message = builder._call_model(messages, tools=tool_schemas)
    content = _field(message, "content", "") or ""
    tool_calls = _normalize_tool_calls(_field(message, "tool_calls", []))

    entry: Dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        entry["tool_calls"] = tool_calls
    messages.append(entry)
    return AgentState(messages=messages, **{k: state[k] for k in state if k != "messages"})


def execute_tools_node(state: AgentState) -> AgentState:
    """执行上一条 assistant 消息里的 tool_calls，结果追加到 messages。"""
    messages = list(state["messages"])
    last = messages[-1] if messages else {}
    tool_calls = _normalize_tool_calls(_field(last, "tool_calls", []))

    for index, tool_call in enumerate(tool_calls):
        if index >= MAX_TOOL_CALLS_PER_ROUND:
            messages.append(_tool_message(tool_call["id"], {
                "ok": False,
                "error": f"单轮工具调用数量超过上限（{MAX_TOOL_CALLS_PER_ROUND}个），该调用未执行",
            }))
            continue
        messages.append(_execute_tool_call(tool_call))

    return AgentState(
        messages=messages,
        tool_round=state["tool_round"] + 1,
        max_tool_rounds=state["max_tool_rounds"],
    )


def final_answer_node(state: AgentState, *, builder: PromptBuilder) -> AgentState:
    """轮次耗尽：不提供工具，请求模型直接输出最终回答。"""
    messages = state["messages"]
    message = builder._call_model(messages)
    content = _field(message, "content", "") or ""
    messages.append({"role": "assistant", "content": content})
    return AgentState(messages=messages, **{k: state[k] for k in state if k != "messages"})


# ---------- Routing ----------

def _should_continue(state: AgentState) -> str:
    """从 call_model_node 出发后的路由：
    - 有 tool_calls 且轮次未耗尽 → "tools"
    - 有 tool_calls 但轮次已耗尽 → "final_answer"
    - 无 tool_calls → END
    """
    messages = state["messages"]
    last = messages[-1] if messages else {}
    tool_calls = _normalize_tool_calls(_field(last, "tool_calls", []))

    if not tool_calls:
        return END

    # 下一轮将是 tool_round + 1
    if state["tool_round"] + 1 > state["max_tool_rounds"]:
        return "final_answer"
    return "tools"


# ---------- Build & Run ----------

def build_graph(builder: PromptBuilder) -> StateGraph:
    """构建 LangGraph StateGraph，返回编译后的可执行图。"""
    graph = StateGraph(AgentState)

    graph.add_node("call_model", lambda s: call_model_node(s, builder=builder))
    graph.add_node("tools", execute_tools_node)
    graph.add_node("final_answer", lambda s: final_answer_node(s, builder=builder))

    graph.set_entry_point("call_model")

    graph.add_conditional_edges("call_model", _should_continue, {
        END: END,
        "tools": "tools",
        "final_answer": "final_answer",
    })
    graph.add_edge("tools", "call_model")
    graph.add_edge("final_answer", END)

    return graph.compile()


def run_agent(
    builder: PromptBuilder,
    messages: List[Dict[str, Any]],
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
) -> str:
    """
    运行模型—工具调用循环，返回最终回答文本。

    Args:
        builder: PromptBuilder 实例（持有 model / API Key 等配置）
        messages: 对话消息列表
        max_tool_rounds: 工具调用轮次上限

    Returns:
        str: 模型最终输出的文本内容
    """
    graph = build_graph(builder)
    result = graph.invoke(_initial_state(messages, max_tool_rounds))
    return _extract_answer(result)


def _extract_answer(state: AgentState) -> str:
    """从最终状态中提取模型最后一条 assistant 消息的文本。"""
    for msg in reversed(state["messages"]):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return msg.get("content", "") or ""
    return ""
