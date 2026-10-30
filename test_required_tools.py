"""
用户明确要求的工具必须被调用（mock 测试）

不依赖真实网络与真实模型：替换 dashscope.Generation.call 模拟模型返回，
替换 TOOL_REGISTRY 中的工具为假工具，验证：
  - 用户消息里出现工具名时，模型不能跳过该工具直接输出最终JSON，提醒后补调即可成功
  - 没有出现工具名时允许0次工具调用（系统提示词里列出的工具名不算用户要求）
  - 只要求一个工具时，不强制调用其他工具
  - 3轮内始终遗漏时抛出明确的ValueError，且不会死循环
  - 工具返回ok=False也算已经执行，不会反复要求

运行方式：python test_required_tools.py
"""

import json
import sys

import prompt_engineering as pe
from prompt_engineering import PromptBuilder

GOOD_ITINERARY = {
    "destination": "北京",
    "start_date": "2026-10-01",
    "end_date": "2026-10-01",
    "total_budget": 3000,
    "people": 2,
    "days": [{
        "day": 1,
        "date": "2026-10-01",
        "items": [{
            "start_time": "09:00",
            "end_time": "11:30",
            "place_name": "故宫博物院",
            "activity": "参观故宫",
            "estimated_cost": 120
        }]
    }],
    "estimated_total_cost": 1500,
    "notes": ["提前预约"]
}

USER_INPUT = {
    "destination": "北京",
    "start_date": "2026-10-01",
    "end_date": "2026-10-01",
    "budget": 3000,
    "people": 2,
    "preferences": "历史文化"
}

# 保存原始工具函数，测试结束后恢复
ORIGINAL_TOOLS = dict(pe.TOOL_REGISTRY)
executed = []


def text_response(content):
    return {"output": {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}]}}


def tool_response(calls):
    return {"output": {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": "", "tool_calls": calls}}]}}


def tool_call(name, arguments, call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
        }
    }


def plan_json():
    return text_response(json.dumps(GOOD_ITINERARY, ensure_ascii=False))


class ScriptedModel:
    """按脚本依次返回模型响应，并记录每次调用的参数"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        assert self.responses, f"模型被调用了{len(self.calls)}次，超出脚本预期"
        return self.responses.pop(0)


def fake_tool(name, result=None):
    """生成记录调用参数的假工具"""

    def _fn(**kwargs):
        executed.append((name, kwargs))
        if result is not None:
            return result
        return {"ok": True, "data": {"name": "X", "location": [116.4, 39.9], "address": "地址", "type": "类型"}}

    return _fn


def reset():
    executed.clear()
    pe.TOOL_REGISTRY.update(ORIGINAL_TOOLS)


def user_messages(messages):
    return [m for m in messages if m.get("role") == "user"]


def test_1_reminder_then_complete():
    """用户要求 get_weather + search_poi：模型漏了天气，提醒后补调并成功生成"""
    reset()
    pe.TOOL_REGISTRY["search_poi"] = fake_tool("search_poi")
    pe.TOOL_REGISTRY["get_weather"] = fake_tool("get_weather")
    user_input = {**USER_INPUT, "preferences": (
        "请先调用 get_weather 查询北京天气，并调用 search_poi 查询故宫的真实地址、坐标，"
        "拿到工具结果后再生成行程。")}
    model = ScriptedModel([
        tool_response([tool_call("search_poi", {"keyword": "故宫博物院", "city": "北京"}, "c1")]),
        plan_json(),                                              # 少了get_weather，这一版应被拒绝
        tool_response([tool_call("get_weather", {"city": "北京"}, "c2")]),
        plan_json(),
    ])
    pe.dashscope.Generation.call = model

    plan = PromptBuilder().generate_itinerary(user_input)

    assert plan == GOOD_ITINERARY
    assert [e[0] for e in executed] == ["search_poi", "get_weather"], "提醒后应补调 get_weather"
    assert len(model.calls) == 4, f"应为[搜索+被拒的JSON+补调天气+出JSON]，实际{len(model.calls)}次"

    reminder = [m for m in user_messages(model.calls[2]["messages"]) if "get_weather" in str(m.get("content", ""))]
    assert reminder, "补调前应追加一条指明缺失工具的修正消息"
    assert "你还没有调用用户明确要求的工具" in reminder[-1]["content"]
    assert model.calls[2].get("tools"), "补调轮次仍然要给模型提供工具"
    assert model.calls[3].get("tools") is None, "轮次用尽后的最终回答不再提供工具"
    print("T1 提醒后补调缺失工具并成功生成 ....... OK")


def test_2_no_required_tools_zero_calls():
    """用户没有指定工具：允许0次工具调用，直接输出JSON"""
    reset()
    model = ScriptedModel([plan_json()])
    pe.dashscope.Generation.call = model

    plan = PromptBuilder().generate_itinerary(USER_INPUT)

    assert plan == GOOD_ITINERARY
    assert len(model.calls) == 1, f"未指定工具时应一次调用直接出结果，实际{len(model.calls)}次"
    assert executed == [], "未指定工具时不应强制调用任何工具"
    print("T2 用户未指定工具时允许0次调用 ........ OK")


def test_3_system_prompt_tool_names_not_required():
    """系统提示词里的工具名不算用户要求（只检查role=user的消息）"""
    reset()
    model = ScriptedModel([plan_json()])
    pe.dashscope.Generation.call = model

    plan = PromptBuilder().generate_itinerary(USER_INPUT)

    system_text = "".join(str(m.get("content", "")) for m in model.calls[0]["messages"] if m.get("role") == "system")
    for name in pe.TOOL_DEFINITIONS:
        assert name in system_text, f"系统提示词里本来就列出了工具名 {name}"
    assert plan == GOOD_ITINERARY
    assert len(model.calls) == 1, "不能因为系统提示词提到工具名就强制调用"
    print("T3 系统提示词中的工具名不算要求 ...... OK")


def test_4_only_search_poi_required():
    """用户只指定 search_poi：不强制调用其他工具"""
    reset()
    pe.TOOL_REGISTRY["search_poi"] = fake_tool("search_poi")
    pe.TOOL_REGISTRY["get_weather"] = fake_tool("get_weather")
    pe.TOOL_REGISTRY["get_route"] = fake_tool("get_route")
    user_input = {**USER_INPUT, "preferences": "请调用 search_poi 查询故宫的位置"}
    model = ScriptedModel([
        tool_response([tool_call("search_poi", {"keyword": "故宫博物院", "city": "北京"}, "c1")]),
        plan_json(),
    ])
    pe.dashscope.Generation.call = model

    plan = PromptBuilder().generate_itinerary(user_input)

    assert plan == GOOD_ITINERARY
    assert [e[0] for e in executed] == ["search_poi"], "只要求search_poi时不应调用其他工具"
    assert len(model.calls) == 2, f"不应额外要求其他工具，实际{len(model.calls)}次调用"
    print("T4 只要求search_poi时不强制其他工具 .... OK")


def test_5_three_rounds_missing_raises():
    """连续3轮都直接输出JSON、始终没调用指定工具：抛出明确错误，不死循环"""
    reset()
    user_input = {**USER_INPUT, "preferences": "请先调用 get_weather 查询北京天气再生成行程"}
    model = ScriptedModel([plan_json() for _ in range(10)])
    pe.dashscope.Generation.call = model

    try:
        PromptBuilder().generate_itinerary(user_input)
        raise AssertionError("明确要求的工具始终未调用时应抛出ValueError")
    except ValueError as e:
        assert "模型未执行用户明确要求的工具：get_weather" in str(e), str(e)

    assert executed == [], "模型没有调用工具，工具不应被执行"
    assert len(model.calls) == 3, f"提醒也受3轮上限约束，实际调用{len(model.calls)}次"
    print("T5 3轮仍未调用指定工具时报错 ........... OK (明确报错, 无死循环)")


def test_6_failed_tool_counts_as_called():
    """工具返回ok=False也算已经执行过：不再强制重复调用"""
    reset()
    pe.TOOL_REGISTRY["get_weather"] = fake_tool(
        "get_weather", {"ok": False, "error": "未获取到'北京'的天气预报数据"})
    user_input = {**USER_INPUT, "preferences": "请先调用 get_weather 查询北京天气再生成行程"}
    model = ScriptedModel([
        tool_response([tool_call("get_weather", {"city": "北京"}, "cw")]),
        plan_json(),
    ])
    pe.dashscope.Generation.call = model

    plan = PromptBuilder().generate_itinerary(user_input)

    assert plan == GOOD_ITINERARY
    assert [e[0] for e in executed] == ["get_weather"], "失败的工具不应被反复要求重调"
    assert len(model.calls) == 2, f"ok=False也算已执行，不应出现补调轮次，实际{len(model.calls)}次"
    tool_msgs = [m for m in model.calls[1]["messages"] if m.get("role") == "tool"]
    assert json.loads(tool_msgs[0]["content"])["ok"] is False
    print("T6 工具返回ok=False也算已执行 ........... OK")


def test_7_optimize_flow_requires_tool():
    """优化流程的反馈也算用户要求：反馈里点名get_route时必须先调用"""
    reset()
    pe.TOOL_REGISTRY["get_route"] = fake_tool("get_route")
    model = ScriptedModel([
        plan_json(),                                              # 没调用工具，这一版应被拒绝
        tool_response([tool_call("get_route", {
            "origin": [116.397, 39.909], "destination": [116.407, 39.919], "mode": "walking"}, "cr")]),
        plan_json(),
    ])
    pe.dashscope.Generation.call = model

    optimized = PromptBuilder().optimize_itinerary(USER_INPUT, GOOD_ITINERARY, "请调用 get_route 查一下两地距离")

    assert optimized == GOOD_ITINERARY
    assert [e[0] for e in executed] == ["get_route"], "反馈中明确要求的工具应被执行"
    assert len(model.calls) == 3, f"应为[被拒的JSON+查路线+出JSON]，实际{len(model.calls)}次"
    reminder = [m for m in user_messages(model.calls[1]["messages"]) if "get_route" in str(m.get("content", ""))]
    assert reminder, "优化流程同样应追加指明缺失工具的修正消息"
    print("T7 优化流程同样执行用户指定的工具 ...... OK")


def main():
    tests = [
        test_1_reminder_then_complete,
        test_2_no_required_tools_zero_calls,
        test_3_system_prompt_tool_names_not_required,
        test_4_only_search_poi_required,
        test_5_three_rounds_missing_raises,
        test_6_failed_tool_counts_as_called,
        test_7_optimize_flow_requires_tool,
    ]
    failed = 0
    try:
        for test in tests:
            try:
                test()
            except Exception as e:
                failed += 1
                print(f"{test.__name__} FAILED: {type(e).__name__}: {e}")
    finally:
        pe.TOOL_REGISTRY.update(ORIGINAL_TOOLS)

    print("-" * 46)
    if failed:
        print(f"{failed}/{len(tests)} 个测试失败")
        sys.exit(1)
    print(f"全部 {len(tests)} 个测试通过")


if __name__ == "__main__":
    main()
