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

由 PromptBuilder 接入生成与优化；行程校验和重规划沿用业务层逻辑。
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langgraph.graph import StateGraph, END

from prompt_engineering import (
    MAX_TOOL_ROUNDS,
    MAX_TOOL_CALLS_PER_ROUND,
    PromptBuilder,
    MISSING_TOOLS_TEMPLATE,
    _field,
)


# ---------- State ----------

class AgentState(TypedDict):
    """LangGraph 状态：消息历史 + 工具调用轮数计数器"""

    messages: List[Dict[str, Any]]
    tool_round: int          # 已完成的工具调用轮数（从 0 开始）
    max_tool_rounds: int     # 上限，默认 MAX_TOOL_ROUNDS
    model_round: int         # 带工具的模型调用次数（含补调提醒）
    required_tools: List[str]
    called_tools: List[str]


def _initial_state(
    messages: List[Dict[str, Any]],
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
) -> AgentState:
    try:
        max_tool_rounds = int(max_tool_rounds)
    except (TypeError, ValueError):
        max_tool_rounds = MAX_TOOL_ROUNDS
    return AgentState(
        messages=list(messages),
        model_round=0,
        required_tools=PromptBuilder._detect_required_tools(messages),
        called_tools=PromptBuilder._called_tools_in_messages(messages),
        tool_round=0,
        max_tool_rounds=max(0, min(int(max_tool_rounds), MAX_TOOL_ROUNDS)),
    )


# 复用业务层的参数校验、错误脱敏和消息兼容处理。
_build_tool_schemas = PromptBuilder._build_tool_schemas
_tool_message = PromptBuilder._tool_message
_normalize_tool_calls = PromptBuilder._normalize_tool_calls


def _execute_tool_call(tool_call, builder):
    return builder._execute_tool_call(tool_call)


# ---------- Graph Nodes ----------

def call_model_node(state: AgentState, *, builder: PromptBuilder) -> AgentState:
    """调用一次模型（带工具定义）。结果追加到 messages。"""
    messages = list(state["messages"])
    tool_schemas = _build_tool_schemas()
    message = builder._call_model(messages, tools=tool_schemas)
    content = _field(message, "content", "") or ""
    tool_calls = _normalize_tool_calls(_field(message, "tool_calls", []))

    entry: Dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        entry["tool_calls"] = tool_calls
    messages.append(entry)
    return {**state, "messages": messages, "model_round": state.get("model_round", 0) + 1}


def execute_tools_node(state: AgentState, *, builder=None) -> AgentState:
    """执行上一条 assistant 消息里的 tool_calls，结果追加到 messages。"""
    messages = list(state["messages"])
    last = messages[-1] if messages else {}
    tool_calls = _normalize_tool_calls(_field(last, "tool_calls", []))

    called = list(state.get("called_tools", []))
    for index, tool_call in enumerate(tool_calls):
        if index >= MAX_TOOL_CALLS_PER_ROUND:
            messages.append(_tool_message(tool_call["id"], {
                "ok": False,
                "error": f"单轮工具调用数量超过上限（{MAX_TOOL_CALLS_PER_ROUND}个），该调用未执行",
            }))
            continue
        called.append(tool_call["function"]["name"])
        messages.append(_execute_tool_call(tool_call, builder))

    return {**state, "messages": messages, "called_tools": called,
            "tool_round": state["tool_round"] + 1}



def final_answer_node(state: AgentState, *, builder: PromptBuilder) -> AgentState:
    """轮次耗尽：不提供工具，请求模型直接输出最终回答。"""
    messages = list(state["messages"])
    missing = _missing(state)
    if missing:
        raise ValueError(f"模型未执行用户明确要求的工具：{'、'.join(missing)}")
    # 防御性处理：即使从自定义状态恢复，也不给模型留下未响应调用。
    if messages and messages[-1].get("tool_calls"):
        for call in messages[-1]["tool_calls"]:
            messages.append(_tool_message(call["id"], {
                "ok": False, "error": "工具调用轮次已耗尽，该调用未执行"
            }))
    message = builder._call_model(messages)
    content = _field(message, "content", "") or ""
    messages.append({"role": "assistant", "content": content})
    return AgentState(messages=messages, **{k: state[k] for k in state if k != "messages"})


# ---------- Routing ----------

def _should_continue(state: AgentState) -> str:
    """从 call_model_node 出发后的路由：
    - 有 tool_calls 且轮次未耗尽 → "tools"
    - 有 tool_calls 但轮次已耗尽 → "final_answer"
    - 无 tool_calls 且缺少用户指定工具 → 提醒补调或报错
    - 无 tool_calls 且要求已满足 → END
    """
    messages = list(state["messages"])
    last = messages[-1] if messages else {}
    tool_calls = _normalize_tool_calls(_field(last, "tool_calls", []))

    if tool_calls:
        return "tools" if state["tool_round"] < state["max_tool_rounds"] else "final_answer"
    if _missing(state):
        return "remind" if state.get("model_round", 0) < state["max_tool_rounds"] else "final_answer"
    return END


def _missing(state):
    return PromptBuilder._missing_required_tools(
        state.get("required_tools", []), state.get("called_tools", []))


def _next_model(state):
    return "call_model" if state.get("model_round", 0) < state["max_tool_rounds"] else "final_answer"


def remind_node(state):
    return {**state, "messages": state["messages"] + [{
        "role": "user",
        "content": MISSING_TOOLS_TEMPLATE.format(missing="、".join(_missing(state)))
    }]}


# ---------- Build & Run ----------

def build_graph(builder: PromptBuilder):
    """构建 LangGraph StateGraph，返回编译后的可执行图。"""
    graph = StateGraph(AgentState)

    graph.add_node("call_model", lambda s: call_model_node(s, builder=builder))
    graph.add_node("tools", lambda s: execute_tools_node(s, builder=builder))
    graph.add_node("final_answer", lambda s: final_answer_node(s, builder=builder))

    graph.add_node("remind", remind_node)
    graph.set_conditional_entry_point(_next_model, {
        "call_model": "call_model", "final_answer": "final_answer"})

    graph.add_conditional_edges("call_model", _should_continue, {
        END: END,
        "tools": "tools",
        "remind": "remind",
        "final_answer": "final_answer",
    })
    graph.add_conditional_edges("tools", _next_model, {
        "call_model": "call_model", "final_answer": "final_answer"})
    graph.add_edge("remind", "call_model")
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
    # 原业务依赖传入列表原地更新，使校验失败后的重规划保留完整上下文。
    messages[:] = result["messages"]
    return _extract_answer(result)


def _extract_answer(state: AgentState) -> str:
    """从最终状态中提取模型最后一条 assistant 消息的文本。"""
    for msg in reversed(state["messages"]):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return msg.get("content", "") or ""
    return ""

