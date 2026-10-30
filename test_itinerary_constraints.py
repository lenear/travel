"""
第四阶段：行程确定性约束检查的 mock 测试

不依赖真实网络与真实模型：约束检查本身是纯代码规则，直接调用
PromptBuilder._validate_itinerary_constraints 验证；
生成/优化流程中的接入通过替换 dashscope.Generation.call 模拟模型返回
（第五阶段加入自动重规划后，流程用例需连续给出两次不合法方案，才能等到最终异常）。

运行方式：python test_itinerary_constraints.py
"""

import copy
import json
import sys

import prompt_engineering as pe
from prompt_engineering import PromptBuilder

# 完全合法的基准行程（2天）
BASE_PLAN = {
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
    """返回基准行程的深拷贝，避免用例间互相影响"""
    return copy.deepcopy(BASE_PLAN)


def check(plan_data, user_input=None):
    return PromptBuilder._validate_itinerary_constraints(plan_data, user_input if user_input is not None else USER_INPUT)


def assert_invalid(plan_data, keyword, user_input=None):
    """断言行程不通过检查，且违规原因中包含指定关键词"""
    result = check(plan_data, user_input)
    assert result["valid"] is False, f"应判定为不合法（期望包含：{keyword}），实际通过"
    assert result["violations"], "不合法时必须给出违规原因"
    joined = "\n".join(result["violations"])
    assert keyword in joined, f"违规原因中应包含{keyword!r}，实际：{joined}"
    return joined


def text_response(content):
    return {"output": {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}]}}


class ScriptedModel:
    """按脚本依次返回模型响应"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        assert self.responses, f"模型被调用了{len(self.calls)}次，超出脚本预期"
        return self.responses.pop(0)


def test_1_valid_plan():
    """完全合法的行程通过检查；重复地点名称不算违规"""
    result = check(plan())
    assert result == {"valid": True, "violations": []}, f"合法行程不应有违规项：{result}"

    duplicate = plan()
    duplicate["days"][1]["items"][0]["place_name"] = "故宫博物院"
    result = check(duplicate)
    assert result["valid"] is True, f"重复地点名称不应报错：{result['violations']}"
    print("T1 合法行程通过、重复地点不报错 ........ OK")


def test_2_day_count_mismatch():
    """天数与开始/结束日期不一致"""
    bad = plan()
    bad["end_date"] = "2026-10-03"  # 应为3天，实际2天
    joined = assert_invalid(bad, "应为3天，实际2天")

    bad2 = plan()
    bad2["days"] = bad2["days"][:1]  # 只保留1天，日期范围仍是2天
    assert_invalid(bad2, "应为2天，实际1天")
    print("T2 天数与日期范围不一致 ................ OK")


def test_3_dates_not_contiguous():
    """日期不连续（缺 2026-10-02）"""
    bad = plan()
    bad["end_date"] = "2026-10-03"
    bad["days"][1]["day"] = 3
    bad["days"][1]["date"] = "2026-10-03"
    bad["days"].append({
        "day": 3,
        "date": "2026-10-04",
        "items": [{
            "start_time": "09:00",
            "end_time": "11:00",
            "place_name": "圆明园",
            "activity": "游览圆明园",
            "estimated_cost": 50
        }]
    })
    assert_invalid(bad, "不连续")

    outside = plan()
    outside["days"][1]["date"] = "2026-10-05"  # 超出用户指定范围
    assert_invalid(outside, "超出用户指定的")

    first_mismatch = plan()
    first_mismatch["start_date"] = "2026-09-30"  # 第1天与开始日期不一致
    assert_invalid(first_mismatch, "第1天的日期与开始日期不一致")
    print("T3 日期不连续/超出范围/首日不一致 ...... OK")


def test_4_empty_items():
    """某一天 items 为空"""
    bad = plan()
    bad["days"][1]["items"] = []
    assert_invalid(bad, "第2天没有任何行程项")

    empty_days = plan()
    empty_days["days"] = []
    assert_invalid(empty_days, "days为空")
    print("T4 某天没有行程项 ...................... OK")


def test_5_invalid_time_format():
    """时间格式不是 HH:MM"""
    bad = plan()
    bad["days"][0]["items"][0]["start_time"] = "9:00"
    assert_invalid(bad, "不是合法的HH:MM时间")

    bad2 = plan()
    bad2["days"][0]["items"][0]["end_time"] = "25:00"
    assert_invalid(bad2, "不是合法的HH:MM时间")
    print("T5 时间格式非法 ........................ OK")


def test_6_time_overlap():
    """同一天内时间重叠"""
    bad = plan()
    bad["days"][0]["items"][1]["start_time"] = "11:00"  # 与 09:00-11:30 重叠
    bad["days"][0]["items"][1]["end_time"] = "13:00"
    assert_invalid(bad, "时间重叠")
    print("T6 同一天时间重叠 ...................... OK")


def test_7_end_before_start():
    """结束时间早于开始时间"""
    bad = plan()
    bad["days"][0]["items"][0]["start_time"] = "15:00"
    bad["days"][0]["items"][0]["end_time"] = "14:00"
    assert_invalid(bad, "结束时间早于开始时间")

    same = plan()
    same["days"][0]["items"][0]["start_time"] = "09:00"
    same["days"][0]["items"][0]["end_time"] = "09:00"
    assert_invalid(same, "结束时间与开始时间相同")

    unordered = plan()
    unordered["days"][0]["items"][0]["start_time"] = "13:00"
    unordered["days"][0]["items"][0]["end_time"] = "18:00"
    unordered["days"][0]["items"][1]["start_time"] = "09:00"
    unordered["days"][0]["items"][1]["end_time"] = "11:30"
    assert_invalid(unordered, "未按开始时间升序排列")
    print("T7 结束早于开始/相同/未排序 ............ OK")


def test_8_budget_exceeded():
    """预计总费用超过用户预算"""
    bad = plan()
    bad["estimated_total_cost"] = 6000  # 用户预算 5000
    joined = assert_invalid(bad, "超过用户预算")
    assert "6000" in joined and "5000" in joined, "错误信息应给出具体金额"

    negative_budget = plan()
    negative_budget["total_budget"] = -1
    assert_invalid(negative_budget, "total_budget 不能小于0")

    unknown_people = plan()
    unknown_people["people"] = 0
    assert_invalid(unknown_people, "people 必须大于0")
    print("T8 预算超支/负预算/人数非法 ............ OK")


def test_9_negative_cost():
    """单项费用为负数或非法数字"""
    bad = plan()
    bad["days"][0]["items"][0]["estimated_cost"] = -100
    assert_invalid(bad, "不能小于0")

    bad2 = plan()
    bad2["days"][0]["items"][0]["estimated_cost"] = "很贵"
    assert_invalid(bad2, "不是有效数字")

    bad3 = plan()
    bad3["estimated_total_cost"] = -50
    assert_invalid(bad3, "estimated_total_cost 不能小于0")
    print("T9 负数费用/非数字费用 ................. OK")


def test_10_flows_raise_with_all_violations():
    """生成与优化流程：约束反复不通过时抛出含全部违规原因的异常

    第五阶段加入"反馈一次、重规划一次"后，模型连续两次给出同一份不合法行程，
    最终仍抛出 ItineraryConstraintError，且不返回半成品行程。
    """
    over_budget = plan()
    over_budget["estimated_total_cost"] = 6000
    over_budget["days"][1]["items"] = []  # 第二条违规
    over_budget_response = text_response(json.dumps(over_budget, ensure_ascii=False))

    # 10.1 生成流程
    model = ScriptedModel([over_budget_response, over_budget_response])
    pe.dashscope.Generation.call = model
    try:
        PromptBuilder().generate_itinerary(USER_INPUT)
        raise AssertionError("约束不通过时应抛出ValueError")
    except ValueError as e:
        message = str(e)
        assert "行程约束检查失败" in message
        assert "超过用户预算" in message and "第2天没有任何行程项" in message, f"应包含全部违规原因：{message}"

    # 10.2 优化流程
    model = ScriptedModel([over_budget_response, over_budget_response])
    pe.dashscope.Generation.call = model
    try:
        PromptBuilder().optimize_itinerary(USER_INPUT, BASE_PLAN, "多安排美食")
        raise AssertionError("优化流程约束不通过时应抛出ValueError")
    except ValueError as e:
        assert "行程约束检查失败" in e.args[0]

    # 10.3 合法行程在两条流程中正常返回
    model = ScriptedModel([text_response(json.dumps(BASE_PLAN, ensure_ascii=False))])
    pe.dashscope.Generation.call = model
    assert PromptBuilder().generate_itinerary(USER_INPUT) == BASE_PLAN
    assert len(model.calls) == 1

    model = ScriptedModel([text_response(json.dumps(BASE_PLAN, ensure_ascii=False))])
    pe.dashscope.Generation.call = model
    assert PromptBuilder().optimize_itinerary(USER_INPUT, BASE_PLAN, "多安排美食") == BASE_PLAN
    assert len(model.calls) == 1

    # 10.4 Flask 流程：约束不通过时页面返回错误提示而不是500
    import app as webapp
    client = webapp.app.test_client()
    form = {"destination": "北京", "start_date": "2026-10-01", "end_date": "2026-10-02",
            "budget": "5000", "people": "2", "preferences": "历史文化"}

    model = ScriptedModel([over_budget_response, over_budget_response])
    pe.dashscope.Generation.call = model
    resp = client.post("/", data=form)
    html = resp.get_data(as_text=True)
    assert resp.status_code == 200, "约束不通过时页面不能返回500"
    assert "行程约束检查失败" in html and "超过用户预算" in html, "页面应展示违规原因"

    model = ScriptedModel([text_response(json.dumps(BASE_PLAN, ensure_ascii=False))])
    pe.dashscope.Generation.call = model
    resp2 = client.post("/", data=form)
    assert resp2.status_code == 200 and "故宫博物院" in resp2.get_data(as_text=True), "合法行程应正常展示"
    print("T10 生成/优化/Flask流程约束接入 ........ OK")


def main():
    tests = [
        test_1_valid_plan,
        test_2_day_count_mismatch,
        test_3_dates_not_contiguous,
        test_4_empty_items,
        test_5_invalid_time_format,
        test_6_time_overlap,
        test_7_end_before_start,
        test_8_budget_exceeded,
        test_9_negative_cost,
        test_10_flows_raise_with_all_violations,
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
