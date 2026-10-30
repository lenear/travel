"""
第三阶段：大模型自主工具调用的 mock 测试

不依赖真实网络与真实模型：通过替换 dashscope.Generation.call 模拟模型返回，
通过替换 TOOL_REGISTRY 中的工具或在 travel_tools 层面 mock 高德接口，
验证工具调用闭环的各项行为。

运行方式：python test_tool_calling.py
"""

import json
import sys

import requests

import prompt_engineering as pe
import travel_tools
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


def output_messages(model, call_index):
    return model.calls[call_index]["messages"]


def test_1_no_tool_call():
    """模型第一次不调用工具，直接返回合法JSON"""
    reset()
    pe.TOOL_REGISTRY["search_poi"] = fake_tool("search_poi")
    model = ScriptedModel([text_response(json.dumps(GOOD_ITINERARY, ensure_ascii=False))])
    pe.dashscope.Generation.call = model

    plan = PromptBuilder().generate_itinerary(USER_INPUT)

    assert plan == GOOD_ITINERARY
    assert len(model.calls) == 1, f"应只调用1次模型，实际{len(model.calls)}次"
    assert executed == [], "未调用工具时不应执行任何工具"
    schemas = model.calls[0]["tools"]
    assert isinstance(schemas, list) and len(schemas) == 3
    assert all(s["type"] == "function" and "parameters" in s["function"] for s in schemas)
    assert model.calls[0]["tool_choice"] == "auto", "tool_choice必须为auto"
    print("T1 不调用工具直接返回合法JSON ......... OK")


def test_2_single_tool_call():
    """模型调用search_poi，工具执行后第二次返回合法JSON"""
    reset()
    pe.TOOL_REGISTRY["search_poi"] = fake_tool("search_poi")
    model = ScriptedModel([
        tool_response([tool_call("search_poi", {"keyword": "故宫博物院", "city": "北京"}, "call_a")]),
        text_response(json.dumps(GOOD_ITINERARY, ensure_ascii=False))
    ])
    pe.dashscope.Generation.call = model

    plan = PromptBuilder().generate_itinerary(USER_INPUT)

    assert plan == GOOD_ITINERARY
    assert executed == [("search_poi", {"keyword": "故宫博物院", "city": "北京"})], "工具参数未被正确解析执行"
    messages = output_messages(model, 1)
    assistant_msg = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert assistant_msg and assistant_msg[0]["tool_calls"][0]["id"] == "call_a"
    assert len(tool_msgs) == 1 and tool_msgs[0]["tool_call_id"] == "call_a"
    assert json.loads(tool_msgs[0]["content"])["ok"] is True, "工具结果应以JSON回传"
    print("T2 调用search_poi后二次请求返回JSON ... OK")


def test_3_weather_and_poi():
    """模型连续调用天气和地点工具"""
    reset()
    pe.TOOL_REGISTRY["get_weather"] = fake_tool("get_weather", {
        "ok": True,
        "data": {"city": "北京", "type": "forecast", "forecasts": [{"date": "2026-10-01", "dayweather": "晴"}]}
    })
    pe.TOOL_REGISTRY["search_poi"] = fake_tool("search_poi")
    model = ScriptedModel([
        tool_response([
            tool_call("get_weather", {"city": "北京", "extensions": "all"}, "call_w"),
            tool_call("search_poi", {"keyword": "故宫博物院", "city": "北京"}, "call_p")
        ]),
        text_response(json.dumps(GOOD_ITINERARY, ensure_ascii=False))
    ])
    pe.dashscope.Generation.call = model

    plan = PromptBuilder().generate_itinerary(USER_INPUT)

    assert plan == GOOD_ITINERARY
    assert [e[0] for e in executed] == ["get_weather", "search_poi"]
    tool_msgs = [m for m in output_messages(model, 1) if m.get("role") == "tool"]
    assert {m["tool_call_id"] for m in tool_msgs} == {"call_w", "call_p"}
    weather = json.loads(tool_msgs[0]["content"])
    assert weather["data"]["forecasts"][0]["dayweather"] == "晴"
    print("T3 连续调用天气和地点工具 ............. OK")


def test_4_unknown_tool():
    """模型请求不存在的工具，返回错误消息且不崩溃"""
    reset()
    model = ScriptedModel([
        tool_response([tool_call("delete_all", {"x": 1}, "call_bad")]),
        text_response(json.dumps(GOOD_ITINERARY, ensure_ascii=False))
    ])
    pe.dashscope.Generation.call = model

    plan = PromptBuilder().generate_itinerary(USER_INPUT)

    assert plan == GOOD_ITINERARY
    tool_msgs = [m for m in output_messages(model, 1) if m.get("role") == "tool"]
    error = json.loads(tool_msgs[0]["content"])
    assert error["ok"] is False and "未注册" in error["error"] and "delete_all" in error["error"]
    assert executed == [], "未注册的工具不能被调用"
    print("T4 请求不存在的工具 ................... OK")


def test_5_invalid_arguments():
    """工具参数不是合法JSON，返回错误消息且不执行工具"""
    reset()
    model = ScriptedModel([
        tool_response([tool_call("search_poi", "{not-json", "call_c")]),
        text_response(json.dumps(GOOD_ITINERARY, ensure_ascii=False))
    ])
    pe.dashscope.Generation.call = model

    plan = PromptBuilder().generate_itinerary(USER_INPUT)

    assert plan == GOOD_ITINERARY
    assert executed == [], "参数非法时不应执行工具"
    tool_msgs = [m for m in output_messages(model, 1) if m.get("role") == "tool"]
    error = json.loads(tool_msgs[0]["content"])
    assert error["ok"] is False and "不是合法JSON" in error["error"]
    print("T5 工具参数不是合法JSON ............... OK")


def test_6_exceed_max_rounds():
    """工具调用超过3轮：停止调用并抛出明确的ValueError"""
    reset()
    pe.TOOL_REGISTRY["search_poi"] = fake_tool("search_poi")
    model = ScriptedModel([tool_response([tool_call("search_poi", {"keyword": "A"}, "c1")]) for _ in range(10)])
    pe.dashscope.Generation.call = model

    try:
        PromptBuilder().generate_itinerary(USER_INPUT)
        raise AssertionError("超过轮次上限时应抛出ValueError")
    except ValueError as e:
        assert "自动修复后仍无效" in str(e)
        assert len(executed) == 3, f"工具最多执行3轮，实际{len(executed)}轮"
        assert model.calls[3].get("tools") is None, "达到轮次上限后不应再提供工具"
        assert len(model.calls) == 5, f"应为3轮工具+1轮强制回答+1次修复，实际{len(model.calls)}次"
    print("T6 工具调用超过3轮 ................... OK (明确报错, 无死循环)")


def test_7_tool_returns_error():
    """工具返回ok=False时，模型仍能收到错误结果并继续生成"""
    reset()
    original_get = travel_tools.requests.get

    class ErrorResp:
        def json(self):
            return {"status": "0", "info": "INVALID_USER_KEY", "infocode": "10001"}

    travel_tools.requests.get = lambda url, params=None, timeout=None, **kw: ErrorResp()
    try:
        model = ScriptedModel([
            tool_response([tool_call("search_poi", {"keyword": "故宫"})]),
            text_response(json.dumps(GOOD_ITINERARY, ensure_ascii=False))
        ])
        pe.dashscope.Generation.call = model

        plan = PromptBuilder().generate_itinerary(USER_INPUT)

        assert plan == GOOD_ITINERARY
        tool_msgs = [m for m in output_messages(model, 1) if m.get("role") == "tool"]
        error = json.loads(tool_msgs[0]["content"])
        assert error["ok"] is False and "10001" in error["error"], "真实工具失败应回传ok=False"
    finally:
        travel_tools.requests.get = original_get
    print("T7 工具返回ok=False时模型收到错误 .... OK")


def test_8_extra_guards():
    """补充：单轮最多3个工具、消息不含API Key、优化流程同样走工具循环"""
    # 8.1 单轮超过3个工具调用时只执行前3个
    reset()
    pe.TOOL_REGISTRY["search_poi"] = fake_tool("search_poi")
    five_calls = [tool_call("search_poi", {"keyword": f"K{i}"}, f"c{i}") for i in range(5)]
    model = ScriptedModel([tool_response(five_calls), text_response(json.dumps(GOOD_ITINERARY, ensure_ascii=False))])
    pe.dashscope.Generation.call = model

    plan = PromptBuilder().generate_itinerary(USER_INPUT)

    assert plan == GOOD_ITINERARY
    assert len(executed) == 3, f"单轮最多执行3个工具，实际{len(executed)}个"
    tool_msgs = [m for m in output_messages(model, 1) if m.get("role") == "tool"]
    assert len(tool_msgs) == 5, "每个tool_call都必须有对应的tool消息"
    assert sum(1 for m in tool_msgs if "超过上限" in m["content"]) == 2

    # 8.2 工具异常时消息中不出现API Key
    reset()
    real_key = travel_tools.AMAP_KEY
    original_get = travel_tools.requests.get

    def boom_get(url, params=None, timeout=None, **kw):
        raise requests.exceptions.SSLError(f"Max retries exceeded with url: /v3/place/text?key={real_key}&keywords=x")

    travel_tools.requests.get = boom_get
    try:
        model = ScriptedModel([
            tool_response([tool_call("search_poi", {"keyword": "故宫"})]),
            text_response(json.dumps(GOOD_ITINERARY, ensure_ascii=False))
        ])
        pe.dashscope.Generation.call = model

        plan = PromptBuilder().generate_itinerary(USER_INPUT)

        assert plan == GOOD_ITINERARY
        all_messages = json.dumps(output_messages(model, 1), ensure_ascii=False)
        assert real_key not in all_messages, "模型消息中不能出现API Key"
        assert "key=***" in all_messages, "错误信息中的Key应被脱敏"
    finally:
        travel_tools.requests.get = original_get

    # 8.3 optimize_itinerary 同样支持自主工具调用并返回同结构dict
    reset()
    pe.TOOL_REGISTRY["get_weather"] = fake_tool("get_weather")
    model = ScriptedModel([
        tool_response([tool_call("get_weather", {"city": "北京"}, "cw")]),
        text_response(json.dumps(GOOD_ITINERARY, ensure_ascii=False))
    ])
    pe.dashscope.Generation.call = model

    optimized = PromptBuilder().optimize_itinerary(USER_INPUT, GOOD_ITINERARY, "天气如何？")

    assert optimized == GOOD_ITINERARY
    assert [e[0] for e in executed] == ["get_weather"]
    assert '"days"' in model.calls[0]["messages"][1]["content"], "优化请求应携带原行程JSON"
    print("T8 补充(单轮上限/Key脱敏/优化流程) .... OK")


def test_9_flask_flow():
    """现有Flask生成、优化、地图流程仍然通过"""
    import re
    import app as webapp

    client = webapp.app.test_client()
    reset()

    # 生成流程：完整走一次工具调用循环后渲染页面
    model = ScriptedModel([
        tool_response([tool_call("search_poi", {"keyword": "故宫博物院", "city": "北京"})]),
        text_response(json.dumps(GOOD_ITINERARY, ensure_ascii=False))
    ])
    pe.dashscope.Generation.call = model
    travel_tools.search_poi = ORIGINAL_TOOLS["search_poi"]
    travel_tools.requests.get = lambda url, params=None, timeout=None, **kw: type(
        "R", (), {"json": lambda self: {"status": "1", "pois": [{"name": "故宫博物院", "location": "116.397,39.918", "address": "景山前街", "type": "风景名胜"}]}}
    )()
    travel_tools.get_weather = ORIGINAL_TOOLS["get_weather"]

    form = {"destination": "北京", "start_date": "2026-10-01", "end_date": "2026-10-01",
            "budget": "3000", "people": "2", "preferences": "历史文化"}

    resp = client.post("/", data=form)
    html = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert "故宫博物院" in html and "09:00-11:30" in html, "页面应展示易读行程"
    match = re.search(r"name=\"itinerary\" id=\"itineraryJson\" value='([^']*)'", html)
    assert match, "页面应携带完整行程JSON"
    hidden_json = match.group(1)
    assert json.loads(hidden_json) == GOOD_ITINERARY

    # 优化流程
    model.responses = [text_response(json.dumps(GOOD_ITINERARY, ensure_ascii=False))]
    resp2 = client.post("/", data={**form, "action": "optimize", "itinerary": hidden_json, "feedback": "多安排美食"})
    assert resp2.status_code == 200 and "故宫博物院" in resp2.get_data(as_text=True)

    # 地图/天气接口
    resp3 = client.post("/get_locations", json={"itinerary": GOOD_ITINERARY, "destination": "北京", "start_date": "2026-10-01"})
    days = resp3.get_json()
    assert resp3.status_code == 200 and len(days) == 1
    assert days[0]["locations"][0]["name"] == "故宫博物院"
    assert days[0]["locations"][0]["time_block"] == "09:00-11:30"
    print("T9 Flask生成/优化/地图流程回归 ........ OK")


def main():
    tests = [
        test_1_no_tool_call,
        test_2_single_tool_call,
        test_3_weather_and_poi,
        test_4_unknown_tool,
        test_5_invalid_arguments,
        test_6_exceed_max_rounds,
        test_7_tool_returns_error,
        test_8_extra_guards,
        test_9_flask_flow,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            failed += 1
            print(f"{test.__name__} FAILED: {type(e).__name__}: {e}")
    print("-" * 46)
    if failed:
        print(f"{failed}/{len(tests)} 个测试失败")
        sys.exit(1)
    print(f"全部 {len(tests)} 个测试通过")


if __name__ == "__main__":
    main()
