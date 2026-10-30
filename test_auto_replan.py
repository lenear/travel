"""
第五阶段：约束失败后自动重规划一次的 mock 测试

不依赖真实网络与真实模型：通过替换 dashscope.Generation.call 按脚本返回模型响应，
验证重规划的触发条件、次数上限、反馈内容、工具可用性和异常类型。

运行方式：python test_auto_replan.py
"""

import copy
import json
import sys

import prompt_engineering as pe
from prompt_engineering import PromptBuilder, ItineraryConstraintError

# 完全合法的2天行程
GOOD_PLAN = {
    "destination": "北京",
    "start_date": "2026-10-01",
    "end_date": "2026-10-02",
    "total_budget": 5000,
    "people": 2,
    "days": [
        {
            "day": 1,
            "date": "2026-10-01",
            "items": [
                {
                    "start_time": "09:00",
                    "end_time": "11:30",
                    "place_name": "故宫博物院",
                    "activity": "参观故宫",
                    "estimated_cost": 120
                },
                {
                    "start_time": "13:00",
                    "end_time": "16:00",
                    "place_name": "天坛公园",
                    "activity": "游览天坛",
                    "estimated_cost": 60
                }
            ]
        },
        {
            "day": 2,
            "date": "2026-10-02",
            "items": [
                {
                    "start_time": "09:00",
                    "end_time": "12:00",
                    "place_name": "颐和园",
                    "activity": "游览颐和园",
                    "estimated_cost": 80
                }
            ]
        }
    ],
    "estimated_total_cost": 260,
    "notes": ["提前预约"]
}

USER_INPUT = {
    "destination": "北京",
    "start_date": "2026-10-01",
    "end_date": "2026-10-02",
    "budget": 5000,
    "people": 2,
    "preferences": "历史文化"
}


def plan():
    return copy.deepcopy(GOOD_PLAN)


def bad_budget():
    """预算超支"""
    data = plan()
    data["estimated_total_cost"] = 8000
    return data


def bad_overlap():
    """第1天时间重叠"""
    data = plan()
    data["days"][0]["items"][1]["start_time"] = "11:00"
    data["days"][0]["items"][1]["end_time"] = "13:00"
    return data


def bad_date():
    """第2天日期与前一天不连续"""
    data = plan()
    data["days"][1]["date"] = "2026-10-05"
    return data


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


def plan_response(data):
    return text_response(json.dumps(data, ensure_ascii=False))


class ScriptedModel:
    """按脚本依次返回模型响应，并记录每次调用的参数"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        assert self.responses, f"模型被调用了{len(self.calls)}次，超出脚本预期（可能存在多余调用或死循环）"
        return self.responses.pop(0)


def messages_of(model, call_index):
    return model.calls[call_index]["messages"]


def last_user_message(messages):
    return [m for m in messages if m.get("role") == "user"][-1]["content"]


def assert_replan_feedback(model):
    """断言第二次调用时携带了修正消息和违规原因"""
    messages = messages_of(model, 1)
    feedback = last_user_message(messages)
    assert "上一版行程没有通过硬性约束检查" in feedback, "重规划时应告知模型上一版未通过检查"
    assert "发现的问题：" in feedback
    assert "重新调用需要的工具" in feedback and "Markdown" in feedback, "应明确要求重新规划并禁止Markdown"
    return feedback


executed = []


def fake_tool(name):
    def _fn(**kwargs):
        executed.append((name, kwargs))
        return {"ok": True, "data": {"name": kwargs.get("keyword", "X"), "location": [116.4, 39.9], "address": "地址", "type": "类型"}}

    return _fn


def reset():
    executed.clear()
    pe.TOOL_REGISTRY["search_poi"] = fake_tool("search_poi")


def test_1_valid_first_try_no_replan():
    """第一次方案合法：只调用一次模型，不发生重规划"""
    reset()
    model = ScriptedModel([plan_response(GOOD_PLAN)])
    pe.dashscope.Generation.call = model

    result = PromptBuilder().generate_itinerary(USER_INPUT)

    assert result == GOOD_PLAN
    assert len(model.calls) == 1, f"合法方案不应触发重规划，实际调用{len(model.calls)}次"
    assert pe.MAX_REPLANS == 1
    print("T1 首次合法不重规划 .................... OK")


def test_2_budget_overrun_fixed_on_replan():
    """第一次预算超支，反馈后模型修正并通过"""
    reset()
    model = ScriptedModel([plan_response(bad_budget()), plan_response(GOOD_PLAN)])
    pe.dashscope.Generation.call = model

    result = PromptBuilder().generate_itinerary(USER_INPUT)

    assert result == GOOD_PLAN, "应返回重规划后的方案"
    assert len(model.calls) == 2, f"应恰好重规划1次，实际调用{len(model.calls)}次"
    feedback = assert_replan_feedback(model)
    assert "超过用户预算" in feedback, "修正消息应包含具体违规原因"
    assert "8000" in feedback and "5000" in feedback, "修正消息应包含具体金额"
    print("T2 预算超支→重规划后通过 ............... OK")


def test_3_time_overlap_fixed_on_replan():
    """第一次时间重叠，反馈后模型修正并通过"""
    reset()
    model = ScriptedModel([plan_response(bad_overlap()), plan_response(GOOD_PLAN)])
    pe.dashscope.Generation.call = model

    result = PromptBuilder().generate_itinerary(USER_INPUT)

    assert result == GOOD_PLAN
    assert len(model.calls) == 2
    feedback = assert_replan_feedback(model)
    assert "时间重叠" in feedback, "修正消息应指出时间重叠"
    print("T3 时间重叠→重规划后通过 ............... OK")


def test_4_date_error_fixed_on_replan():
    """第一次日期不连续，反馈后模型修正并通过"""
    reset()
    model = ScriptedModel([plan_response(bad_date()), plan_response(GOOD_PLAN)])
    pe.dashscope.Generation.call = model

    result = PromptBuilder().generate_itinerary(USER_INPUT)

    assert result == GOOD_PLAN
    assert len(model.calls) == 2
    feedback = assert_replan_feedback(model)
    assert "不连续" in feedback, "修正消息应指出日期问题"
    print("T4 日期错误→重规划后通过 ............... OK")


def test_5_both_invalid_raises():
    """两次都不合法：抛出最终的 ItineraryConstraintError，不返回半成品"""
    reset()
    second = bad_overlap()
    model = ScriptedModel([plan_response(bad_budget()), plan_response(second)])
    pe.dashscope.Generation.call = model

    try:
        PromptBuilder().generate_itinerary(USER_INPUT)
        raise AssertionError("两次都不合法时应抛出异常")
    except ItineraryConstraintError as e:
        assert isinstance(e, ValueError), "ItineraryConstraintError 应是 ValueError 的子类"
        assert e.violations, "异常应携带违规原因列表"
        assert any("时间重叠" in item for item in e.violations), f"应携带第二次方案的违规原因：{e.violations}"
        assert "行程约束检查失败" in str(e)
    assert len(model.calls) == 2, f"重规划最多1次，实际调用{len(model.calls)}次"
    print("T5 两次都不合法→抛最终异常 ............ OK")


def test_6_tools_available_during_replan():
    """重规划过程中仍可自主调用工具，且工具结果回传给模型"""
    reset()
    model = ScriptedModel([
        plan_response(bad_overlap()),
        tool_response([tool_call("search_poi", {"keyword": "天坛公园", "city": "北京"}, "call_r")]),
        plan_response(GOOD_PLAN)
    ])
    pe.dashscope.Generation.call = model

    result = PromptBuilder().generate_itinerary(USER_INPUT)

    assert result == GOOD_PLAN
    assert [e[0] for e in executed] == ["search_poi"], "重规划时应能执行工具"
    assert len(model.calls) == 3
    replan_schemas = model.calls[1]["tools"]
    assert isinstance(replan_schemas, list) and len(replan_schemas) == 3, "重规划时仍应提供工具定义"
    assert model.calls[1]["tool_choice"] == "auto", "重规划仍由模型自主决定是否调用工具"
    tool_msgs = [m for m in messages_of(model, 2) if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["tool_call_id"] == "call_r", "工具结果应回传给模型"
    print("T6 重规划中调用工具 .................... OK")


def test_7_replan_at_most_once():
    """模型连续给出不合法方案：最多重规划1次后立即停止"""
    reset()
    model = ScriptedModel([plan_response(bad_overlap()) for _ in range(6)])
    pe.dashscope.Generation.call = model

    try:
        PromptBuilder().generate_itinerary(USER_INPUT)
        raise AssertionError("始终不合法时应抛出异常")
    except ItineraryConstraintError:
        pass
    assert len(model.calls) == 2, f"最多调用模型2次（首次+重规划1次），实际{len(model.calls)}次"
    print("T7 重规划次数上限为1次 ................. OK")


def test_8_optimize_also_replans():
    """optimize_itinerary 同样支持一次自动重规划"""
    reset()
    model = ScriptedModel([plan_response(bad_budget()), plan_response(GOOD_PLAN)])
    pe.dashscope.Generation.call = model

    result = PromptBuilder().optimize_itinerary(USER_INPUT, GOOD_PLAN, "多安排美食")

    assert result == GOOD_PLAN
    assert len(model.calls) == 2
    feedback = assert_replan_feedback(model)
    assert "超过用户预算" in feedback
    assert "原始行程JSON如下" in messages_of(model, 0)[1]["content"], "优化请求应携带原行程JSON"

    # 优化流程两次都不合法时同样抛出 ItineraryConstraintError
    model = ScriptedModel([plan_response(bad_budget()), plan_response(bad_budget())])
    pe.dashscope.Generation.call = model
    try:
        PromptBuilder().optimize_itinerary(USER_INPUT, GOOD_PLAN, "多安排美食")
        raise AssertionError("优化流程两次都不合法时应抛出异常")
    except ItineraryConstraintError as e:
        assert any("超过用户预算" in item for item in e.violations)
    assert len(model.calls) == 2
    print("T8 优化流程同样支持重规划 .............. OK")


def test_9_flask_flow():
    """Flask：重规划成功时正常展示；两次失败时返回200并显示错误，不产生500"""
    import app as webapp

    client = webapp.app.test_client()
    form = {"destination": "北京", "start_date": "2026-10-01", "end_date": "2026-10-02",
            "budget": "5000", "people": "2", "preferences": "历史文化"}

    # 重规划成功
    reset()
    model = ScriptedModel([plan_response(bad_overlap()), plan_response(GOOD_PLAN)])
    pe.dashscope.Generation.call = model
    resp = client.post("/", data=form)
    html = resp.get_data(as_text=True)
    assert resp.status_code == 200 and "故宫博物院" in html, "重规划成功后应展示行程"
    assert len(model.calls) == 2

    # 两次都不合法
    model = ScriptedModel([plan_response(bad_budget()), plan_response(bad_budget())])
    pe.dashscope.Generation.call = model
    resp2 = client.post("/", data=form)
    html2 = resp2.get_data(as_text=True)
    assert resp2.status_code == 200, "约束始终不通过时页面也不能返回500"
    assert "行程约束检查失败" in html2 and "超过用户预算" in html2, "页面应显示违规原因"
    assert "故宫博物院" not in html2, "不合法的行程不能展示给前端"
    print("T9 Flask重规划成功/失败流程 ............ OK")


def main():
    tests = [
        test_1_valid_first_try_no_replan,
        test_2_budget_overrun_fixed_on_replan,
        test_3_time_overlap_fixed_on_replan,
        test_4_date_error_fixed_on_replan,
        test_5_both_invalid_raises,
        test_6_tools_available_during_replan,
        test_7_replan_at_most_once,
        test_8_optimize_also_replans,
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
