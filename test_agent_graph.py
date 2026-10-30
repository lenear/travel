"""
测试 agent_graph.py：用模拟模型响应和模拟工具验证两条核心路径

路径 1：直接回答 — 模型不返回 tool_calls → 图直接结束
路径 2：调用工具后回答 — 模型返回 tool_calls → 执行工具 → 模型返回最终回答

不消耗真实 API。
"""

import json
import sys
import unittest
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List, Optional

# 确保能 import 项目模块
sys.path.insert(0, ".")

from agent_graph import (
    AgentState,
    build_graph,
    run_agent,
    _initial_state,
    _should_continue,
    _extract_answer,
    _normalize_tool_calls,
    _execute_tool_call,
    call_model_node,
    execute_tools_node,
    final_answer_node,
)
from prompt_engineering import MAX_TOOL_ROUNDS, MAX_TOOL_CALLS_PER_ROUND


# ---------- 辅助：造 PromptBuilder mock ----------

def _make_builder(call_side_effect):
    """造一个 PromptBuilder mock，_call_model 按顺序返回 side_effect 里的响应"""
    builder = MagicMock()
    builder._call_model = MagicMock(side_effect=call_side_effect)
    return builder


def _assistant_msg(content: str = "", tool_calls: Optional[List] = None) -> Dict[str, Any]:
    """构造 DashScope 风格的 assistant message"""
    msg: Dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return msg


def _tool_call_dict(
    name: str, arguments: str = "{}", call_id: str = "call_0"
) -> Dict[str, Any]:
    """构造 DashScope 风格的 tool_call"""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


# ---------- 路径 1：直接回答 ----------

class TestDirectAnswer(unittest.TestCase):
    """模型第一次就返回纯文本回答，不调用工具"""

    def test_no_tool_calls_ends_immediately(self):
        """模型无 tool_calls → 图结束，返回模型文本"""
        final_text = "北京三日行程：第一天故宫..."
        builder = _make_builder([_assistant_msg(content=final_text)])

        messages = [{"role": "user", "content": "帮我规划北京三日游"}]
        answer = run_agent(builder, messages, max_tool_rounds=3)

        self.assertEqual(answer, final_text)
        # 模型只被调用一次
        self.assertEqual(builder._call_model.call_count, 1)

    def test_messages_preserved(self):
        """最终状态的消息历史应包含原始消息 + assistant 回答"""
        final_text = "行程结果"
        builder = _make_builder([_assistant_msg(content=final_text)])

        user_msg = [{"role": "user", "content": "测试"}]
        graph = build_graph(builder)
        result = graph.invoke(_initial_state(user_msg, max_tool_rounds=3))

        msgs = result["messages"]
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[-1]["role"], "assistant")
        self.assertEqual(msgs[-1]["content"], final_text)

    def test_zero_tool_rounds(self):
        """max_tool_rounds=0 → 模型有 tool_calls 也不走工具，直接 final_answer"""
        tool_calls = [_tool_call_dict("search_poi", '{"keyword": "故宫", "city": "北京"}')]
        # 第一次返回 tool_calls，第二次返回纯文本
        builder = _make_builder([
            _assistant_msg(content="", tool_calls=tool_calls),
            _assistant_msg(content="最终回答"),
        ])

        answer = run_agent(builder, [{"role": "user", "content": "测试"}], max_tool_rounds=0)
        self.assertEqual(answer, "最终回答")
        # 第一次 call_model + 一次 final_answer = 2 次
        self.assertEqual(builder._call_model.call_count, 2)


# ---------- 路径 2：调用工具后回答 ----------

class TestToolThenAnswer(unittest.TestCase):
    """模型先调用工具，工具执行后模型给出最终回答"""

    @patch("agent_graph._execute_tool_call")
    def test_one_tool_then_answer(self, mock_exec):
        """模型调用一个工具 → 执行工具 → 模型返回最终回答"""
        tool_result = {"ok": True, "data": {"name": "故宫博物院"}}
        mock_exec.return_value = {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": json.dumps(tool_result, ensure_ascii=False),
        }

        tool_calls = [_tool_call_dict("search_poi", '{"keyword": "故宫", "city": "北京"}')]
        builder = _make_builder([
            _assistant_msg(content="", tool_calls=tool_calls),   # 第 1 次：返回 tool_calls
            _assistant_msg(content="结合工具结果的行程"),         # 第 2 次：返回纯文本
        ])

        answer = run_agent(builder, [{"role": "user", "content": "规划北京游"}], max_tool_rounds=3)
        self.assertEqual(answer, "结合工具结果的行程")
        self.assertEqual(builder._call_model.call_count, 2)
        self.assertTrue(mock_exec.called)

    @patch("agent_graph._execute_tool_call")
    def test_two_rounds_of_tools(self, mock_exec):
        """两轮工具调用：第 1 轮 1 个工具，第 2 轮 1 个工具，然后最终回答"""
        tool_result = {"ok": True, "data": {}}
        mock_exec.return_value = {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": json.dumps(tool_result, ensure_ascii=False),
        }

        tc1 = [_tool_call_dict("search_poi", '{"keyword": "故宫", "city": "北京"}', "call_0")]
        tc2 = [_tool_call_dict("get_weather", '{"city": "北京"}', "call_1")]

        builder = _make_builder([
            _assistant_msg(content="", tool_calls=tc1),      # 轮 1：调用 search_poi
            _assistant_msg(content="", tool_calls=tc2),      # 轮 2：调用 get_weather
            _assistant_msg(content="最终行程JSON"),           # 最终回答
        ])

        answer = run_agent(builder, [{"role": "user", "content": "规划北京游"}], max_tool_rounds=3)
        self.assertEqual(answer, "最终行程JSON")
        self.assertEqual(builder._call_model.call_count, 3)
        self.assertEqual(mock_exec.call_count, 2)

    @patch("agent_graph._execute_tool_call")
    def test_max_rounds_exhausted(self, mock_exec):
        """工具轮次耗尽 → 不提供工具再请求一次最终回答"""
        tool_result = {"ok": True, "data": {}}
        mock_exec.return_value = {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": json.dumps(tool_result, ensure_ascii=False),
        }

        tc = [_tool_call_dict("search_poi", '{"keyword": "故宫", "city": "北京"}', "call_0")]
        builder = _make_builder([
            _assistant_msg(content="", tool_calls=tc),   # 轮 1：调用工具
            _assistant_msg(content="", tool_calls=tc),   # 轮 2：还想调用工具
            # 轮次耗尽后 final_answer 节点调用
            _assistant_msg(content="基于已有信息的最终回答"),
        ])

        answer = run_agent(builder, [{"role": "user", "content": "规划北京游"}], max_tool_rounds=2)
        self.assertEqual(answer, "基于已有信息的最终回答")
        # 2 次 call_model (带工具) + 1 次 final_answer (不带工具)
        self.assertEqual(builder._call_model.call_count, 3)


# ---------- 路由逻辑 ----------

class TestRouting(unittest.TestCase):
    """_should_continue 路由决策"""

    def test_no_tool_calls_goes_end(self):
        state = AgentState(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "回答", "tool_calls": []},
            ],
            tool_round=0,
            max_tool_rounds=3,
        )
        from langgraph.graph import END
        self.assertEqual(_should_continue(state), END)

    def test_has_tool_calls_and_rounds_left_goes_tools(self):
        state = AgentState(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "tool_calls": [
                    _tool_call_dict("search_poi", '{"keyword": "故宫"}', "call_0")
                ]},
            ],
            tool_round=0,
            max_tool_rounds=3,
        )
        self.assertEqual(_should_continue(state), "tools")

    def test_has_tool_calls_but_rounds_exhausted_goes_final(self):
        state = AgentState(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "tool_calls": [
                    _tool_call_dict("search_poi", '{"keyword": "故宫"}', "call_0")
                ]},
            ],
            tool_round=3,
            max_tool_rounds=3,
        )
        self.assertEqual(_should_continue(state), "final_answer")


# ---------- 工具执行失败也反馈给模型 ----------

class TestToolFailure(unittest.TestCase):
    """工具执行失败时，错误作为 tool 消息反馈给模型，图不中断"""

    @patch("agent_graph._execute_tool_call")
    def test_tool_error_is_feedback(self, mock_exec):
        mock_exec.return_value = {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": json.dumps({"ok": False, "error": "搜索关键词为空"}, ensure_ascii=False),
        }

        tc = [_tool_call_dict("search_poi", '{"keyword": ""}', "call_0")]
        builder = _make_builder([
            _assistant_msg(content="", tool_calls=tc),
            _assistant_msg(content="虽然工具失败但我还是给出行程"),
        ])

        answer = run_agent(builder, [{"role": "user", "content": "测试"}], max_tool_rounds=3)
        self.assertEqual(answer, "虽然工具失败但我还是给出行程")


# ---------- 状态与工具计数 ----------

class TestStateAndLimits(unittest.TestCase):
    """tool_round 计数和单轮工具数量上限"""

    def test_tool_round_increments(self):
        """execute_tools_node 每执行一次 tool_round +1"""
        tc = _tool_call_dict("search_poi", '{"keyword": "故宫", "city": "北京"}')
        state = AgentState(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "tool_calls": [tc]},
            ],
            tool_round=0,
            max_tool_rounds=3,
        )
        with patch("agent_graph._execute_tool_call") as mock_exec:
            mock_exec.return_value = {
                "role": "tool",
                "tool_call_id": "call_0",
                "content": json.dumps({"ok": True, "data": {}}),
            }
            new_state = execute_tools_node(state)
        self.assertEqual(new_state["tool_round"], 1)

    def test_initial_state_clamps_rounds(self):
        """max_tool_rounds 被钳位到 [0, MAX_TOOL_ROUNDS]"""
        s = _initial_state([], max_tool_rounds=100)
        self.assertEqual(s["max_tool_rounds"], MAX_TOOL_ROUNDS)
        s = _initial_state([], max_tool_rounds=-1)
        self.assertEqual(s["max_tool_rounds"], 0)


if __name__ == "__main__":
    unittest.main()
