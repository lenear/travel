"""
第六阶段：短期会话记忆的 mock 测试

不依赖真实网络与真实模型：通过替换 dashscope.Generation.call 按脚本返回模型响应，
用 Flask test_client 验证 session 中的记忆读写、隔离与截断。

运行方式：python test_session_memory.py
"""

import copy
import json
import os
import sys

import prompt_engineering as pe
from flask import session as flask_session

import app as webapp

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
    "estimated_total_cost": 200,
    "notes": ["提前预约"]
}

FORM = {
    "destination": "北京",
    "start_date": "2026-10-01",
    "end_date": "2026-10-02",
    "budget": "5000",
    "people": "2",
    "preferences": "历史文化"
}

MEMORY_KEYS = {
    "destination", "start_date", "end_date", "budget", "people", "preferences",
    "feedback_history", "last_validation_errors", "revision_count"
}


def plan():
    return copy.deepcopy(GOOD_PLAN)


def bad_budget():
    data = plan()
    data["estimated_total_cost"] = 9000
    return data


def text_response(content):
    return {"output": {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}]}}


def plan_response(data):
    return text_response(json.dumps(data, ensure_ascii=False))


class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        assert self.responses, f"模型被调用了{len(self.calls)}次，超出脚本预期"
        return self.responses.pop(0)


def new_client():
    return webapp.app.test_client()


def read_memory(client):
    with client.session_transaction() as sess:
        return sess.get(webapp.SESSION_MEMORY_KEY)


def generate(client, model, form=None):
    pe.dashscope.Generation.call = model
    return client.post("/", data=form or FORM)


def optimize(client, model, feedback, form=None, plan_data=None):
    data = plan() if plan_data is None else plan_data
    pe.dashscope.Generation.call = model
    return client.post("/", data={
        "action": "optimize", "itinerary": json.dumps(data, ensure_ascii=False),
        "feedback": feedback, **(form or FORM)
    })


def messages_text(model, call_index=0):
    return json.dumps(model.calls[call_index]["messages"], ensure_ascii=False)


def test_1_generate_saves_basic_info():
    """生成行程后 session 中保存基本旅行信息"""
    client = new_client()
    assert read_memory(client) is None, "新会话不应有记忆"

    resp = generate(client, ScriptedModel([plan_response(GOOD_PLAN)]))
    assert resp.status_code == 200 and "故宫博物院" in resp.get_data(as_text=True)

    memory = read_memory(client)
    assert memory, "生成成功后应写入会话记忆"
    assert memory["destination"] == "北京"
    assert memory["start_date"] == "2026-10-01" and memory["end_date"] == "2026-10-02"
    assert memory["budget"] == 5000 and memory["people"] == 2
    assert memory["preferences"] == "历史文化"
    assert memory["feedback_history"] == [] and memory["last_validation_errors"] == []
    assert memory["revision_count"] == 0
    assert set(memory) == MEMORY_KEYS, f"记忆字段与约定不一致：{sorted(memory)}"
    print("T1 生成后保存基本旅行信息 .............. OK")


def test_2_feedback_history_appended():
    """优化后 feedback_history 正确增加"""
    client = new_client()
    generate(client, ScriptedModel([plan_response(GOOD_PLAN)]))

    optimize(client, ScriptedModel([plan_response(GOOD_PLAN)]), "多安排美食")
    assert read_memory(client)["feedback_history"] == ["多安排美食"]

    optimize(client, ScriptedModel([plan_response(GOOD_PLAN)]), "减少步行")
    assert read_memory(client)["feedback_history"] == ["多安排美食", "减少步行"]
    print("T2 优化后追加反馈历史 .................. OK")


def test_3_feedback_history_limit():
    """feedback_history 最多保留最近3条"""
    client = new_client()
    generate(client, ScriptedModel([plan_response(GOOD_PLAN)]))

    for feedback in ["反馈1", "反馈2", "反馈3", "反馈4"]:
        optimize(client, ScriptedModel([plan_response(GOOD_PLAN)]), feedback)

    memory = read_memory(client)
    assert memory["feedback_history"] == ["反馈2", "反馈3", "反馈4"], f"应只保留最近3条：{memory['feedback_history']}"
    assert pe.MAX_MEMORY_FEEDBACK == 3
    print("T3 反馈历史最多保留3条 ................. OK")


def test_4_revision_count_increases():
    """revision_count 正确递增，重新生成后归零"""
    client = new_client()
    generate(client, ScriptedModel([plan_response(GOOD_PLAN)]))
    assert read_memory(client)["revision_count"] == 0

    for expected in (1, 2, 3):
        optimize(client, ScriptedModel([plan_response(GOOD_PLAN)]), f"第{expected}次反馈")
        assert read_memory(client)["revision_count"] == expected, f"修改次数应为{expected}"

    # 同行程重新生成：仍未优化过
    generate(client, ScriptedModel([plan_response(GOOD_PLAN)]))
    memory = read_memory(client)
    assert memory["revision_count"] == 0, "重新生成后修改次数应归零"
    assert len(memory["feedback_history"]) == 3, "同行程重新生成时保留历史反馈"
    print("T4 修改次数递增/重新生成归零 ........... OK")


def test_5_constraint_failure_saves_errors():
    """约束反复不通过后保存 last_validation_errors"""
    client = new_client()
    bad = plan_response(bad_budget())
    resp = generate(client, ScriptedModel([bad, bad]))

    html = resp.get_data(as_text=True)
    assert resp.status_code == 200 and "行程约束检查失败" in html

    memory = read_memory(client)
    assert memory["last_validation_errors"], "约束失败后应保存违规原因"
    assert any("超过用户预算" in item for item in memory["last_validation_errors"])
    assert memory["revision_count"] == 0, "失败不应增加修改次数"
    assert memory["feedback_history"] == [], "失败不应写入反馈历史"
    assert memory["destination"] == "北京" and memory["preferences"] == "历史文化"
    print("T5 约束失败保存错误原因 ................ OK")


def test_6_model_receives_history():
    """下一次模型调用能收到偏好、历史反馈、错误原因和修改次数"""
    client = new_client()
    generate(client, ScriptedModel([plan_response(GOOD_PLAN)]))
    optimize(client, ScriptedModel([plan_response(GOOD_PLAN)]), "多安排美食")

    # 优化流程：模型应收到偏好、反馈历史和修改次数
    model = ScriptedModel([plan_response(GOOD_PLAN)])
    optimize(client, model, "再多安排一个博物馆")
    text = messages_text(model)
    assert "用户之前填写的旅行偏好：历史文化" in text
    assert "最近几次反馈：多安排美食" in text
    assert "当前行程已经修改的次数：1" in text
    assert "如果用户本次输入与历史信息冲突，应优先服从本次输入" in text
    assert "不要凭空增加用户没有提供的事实" in text

    # 约束失败后，生成流程应收到上一次失败原因
    bad = plan_response(bad_budget())
    generate(client, ScriptedModel([bad, bad]))
    model = ScriptedModel([plan_response(GOOD_PLAN)])
    generate(client, model)
    text = messages_text(model)
    assert "上一次约束检查失败的原因：" in text and "超过用户预算" in text
    print("T6 模型收到历史信息 .................... OK")


def test_7_new_destination_clears_history():
    """更换目的地或日期会清空反馈历史与错误记录"""
    client = new_client()
    generate(client, ScriptedModel([plan_response(GOOD_PLAN)]))
    optimize(client, ScriptedModel([plan_response(GOOD_PLAN)]), "多安排美食")
    bad = plan_response(bad_budget())
    generate(client, ScriptedModel([bad, bad]))
    assert read_memory(client)["last_validation_errors"], "先制造一条错误记录"

    other_form = {**FORM, "destination": "上海", "start_date": "2026-11-01", "end_date": "2026-11-02"}
    other_plan = {**plan(), "destination": "上海", "start_date": "2026-11-01", "end_date": "2026-11-02",
                  "days": [{**plan()["days"][0], "date": "2026-11-01"},
                           {**plan()["days"][1], "date": "2026-11-02"}]}
    generate(client, ScriptedModel([plan_response(other_plan)]), other_form)

    memory = read_memory(client)
    assert memory["destination"] == "上海"
    assert memory["feedback_history"] == [], "换目的地应清空旧反馈历史"
    assert memory["last_validation_errors"] == [], "换目的地应清空旧错误记录"
    assert memory["revision_count"] == 0
    assert memory["preferences"] == "历史文化", "用户基本信息应保留"
    print("T7 换目的地/日期清空历史 ................ OK")


def test_8_sessions_are_isolated():
    """不同用户的 session 互不影响"""
    client_a, client_b = new_client(), new_client()

    generate(client_a, ScriptedModel([plan_response(GOOD_PLAN)]))
    assert read_memory(client_a) is not None
    assert read_memory(client_b) is None, "B用户不应看到A用户的记忆"

    b_form = {**FORM, "destination": "上海"}
    generate(client_b, ScriptedModel([plan_response(GOOD_PLAN)]), b_form)
    assert read_memory(client_a)["destination"] == "北京", "B用户的操作不应影响A用户"
    assert read_memory(client_b)["destination"] == "上海"

    # 清除记忆只影响自己
    resp = client_a.post("/clear_memory")
    assert resp.status_code == 200 and resp.get_json()["ok"] is True
    assert read_memory(client_a) is None
    assert read_memory(client_b)["destination"] == "上海", "清除操作不应影响其他用户"
    print("T8 不同用户session互相隔离 ............. OK")


def test_9_no_api_key_and_small_cookie():
    """session中不含API Key、不含完整行程，且超长内容自动截断"""
    long_preferences = "历史文化" * 100          # 400字
    long_feedback = "多安排美食" * 100            # 500字
    long_form = {**FORM, "preferences": long_preferences}
    client = new_client()
    generate(client, ScriptedModel([plan_response(GOOD_PLAN)]), long_form)
    optimize(client, ScriptedModel([plan_response(GOOD_PLAN)]), long_feedback, long_form)

    memory = read_memory(client)
    raw = json.dumps(memory, ensure_ascii=False)

    # 不包含任何API Key
    for key in (os.getenv("DASHSCOPE_API_KEY", ""), os.getenv("AMAP_KEY", "")):
        assert not key or key not in raw, "session中不能出现API Key"
    assert "sk-" not in raw and "key=" not in raw

    # 不保存完整行程和工具返回结果
    assert "days" not in memory and "items" not in memory and "place_name" not in raw
    assert "estimated_total_cost" not in raw

    # 超长字段按字符数截断
    assert len(memory["preferences"]) == webapp.MEMORY_LIMITS["preferences"]
    assert len(memory["feedback_history"][0]) == webapp.MEMORY_LIMITS["feedback"]

    # 失败原因同样被截断，且最多保留5条
    many_errors = [f"第{i}天存在时间重叠" * 20 for i in range(8)]
    with webapp.app.test_request_context():
        stored = webapp.update_session_memory(
            {"destination": "北京", "start_date": "2026-10-01", "end_date": "2026-10-02",
             "budget": 5000, "people": 2, "preferences": "历史文化"},
            violations=many_errors
        )
    assert len(stored["last_validation_errors"]) == webapp.MAX_MEMORY_ERRORS, "错误原因最多保留5条"
    assert all(len(item) == webapp.MEMORY_LIMITS["error"] for item in stored["last_validation_errors"])
    print("T9 无API Key/无完整行程/字段截断 ....... OK")


def test_11_cookie_size_within_limit():
    """真实Set-Cookie长度受控：极端输入下远低于上限，且不会过度删除记忆"""
    # 11.1 极端中文输入（长偏好+多条长反馈）下测量真实响应头
    client = new_client()
    long_form = {**FORM, "preferences": "历史文化" * 100}
    resp = generate(client, ScriptedModel([plan_response(GOOD_PLAN)]), long_form)
    header = resp.headers.get("Set-Cookie", "")
    header_bytes = len(header.encode("utf-8"))
    cookie_value = header.split("session=", 1)[1].split(";", 1)[0]
    value_bytes = len(cookie_value.encode("utf-8"))
    assert value_bytes < webapp.SESSION_COOKIE_MAX_BYTES, f"极端输入下Cookie值{value_bytes}字节，超过上限"
    print(f"    [测量] 极端中文输入 Set-Cookie头 {header_bytes} 字节 / session值 {value_bytes} 字节")

    for _ in range(3):
        resp = optimize(client, ScriptedModel([plan_response(GOOD_PLAN)]), "多安排美食" * 100, long_form)
    value_bytes = len(resp.headers["Set-Cookie"].split("session=", 1)[1].split(";", 1)[0].encode("utf-8"))
    assert value_bytes < webapp.SESSION_COOKIE_MAX_BYTES
    print(f"    [测量] 3条长反馈后   session值 {value_bytes} 字节")

    # 不因阈值过小而过度删除：反馈和错误都应完整保留
    memory = read_memory(client)
    assert len(memory["feedback_history"]) == webapp.MAX_MEMORY_FEEDBACK, "正常内容不应被裁剪掉反馈"
    assert len(memory["feedback_history"][0]) == webapp.MEMORY_LIMITS["feedback"]
    assert len(memory["preferences"]) == webapp.MEMORY_LIMITS["preferences"]

    # 11.2 最坏情况：所有字段顶格且内容不可压缩（zlib压不动）
    def hexfill(n):
        return os.urandom((n // 2) + 1).hex()[:n]

    worst = {
        "destination": hexfill(webapp.MEMORY_LIMITS["destination"]),
        "start_date": "2026-10-01", "end_date": "2026-10-02",
        "budget": 5000, "people": 2,
        "preferences": hexfill(webapp.MEMORY_LIMITS["preferences"]),
        "feedback_history": [hexfill(webapp.MEMORY_LIMITS["feedback"]) for _ in range(3)],
        "last_validation_errors": [hexfill(webapp.MEMORY_LIMITS["error"]) for _ in range(5)],
        "revision_count": 3,
    }
    worst_size = webapp._session_cookie_size(worst)
    print(f"    [测量] 不可压缩满字段   session值 {worst_size} 字节")
    assert worst_size < webapp.SESSION_COOKIE_MAX_BYTES, "满字段不可压缩内容也不能超过上限"

    # 11.3 兜底裁剪确实生效，且只裁剪历史、不动基本信息
    oversized = dict(worst)
    oversized["feedback_history"] = [hexfill(400) for _ in range(3)]
    oversized["last_validation_errors"] = [hexfill(500) for _ in range(5)]
    oversized["preferences"] = hexfill(800)
    oversized["destination"] = hexfill(400)
    assert webapp._session_cookie_size(oversized) > webapp.SESSION_COOKIE_MAX_BYTES, "用例应确实超限"
    shrunk = webapp._shrink_memory(oversized)
    final_size = webapp._session_cookie_size(shrunk)
    assert final_size <= webapp.SESSION_COOKIE_TARGET_BYTES, f"裁剪应留出余量，实际{final_size}字节"
    assert final_size <= webapp.SESSION_COOKIE_MAX_BYTES, f"裁剪后仍超限：{final_size}字节"
    assert shrunk["budget"] == 5000 and shrunk["people"] == 2 and shrunk["start_date"] == "2026-10-01"
    print(f"    [测量] 超限用例裁剪后 session值 {final_size} 字节，剩余反馈{len(shrunk['feedback_history'])}条/"
          f"错误{len(shrunk['last_validation_errors'])}条")
    print("T11 Cookie长度受控且不过度裁剪 ......... OK")


def test_10_clear_memory_endpoint():
    """清除记忆接口"""
    client = new_client()
    generate(client, ScriptedModel([plan_response(GOOD_PLAN)]))
    assert read_memory(client)

    resp = client.post("/clear_memory")
    assert resp.status_code == 200
    assert resp.get_json() == {"ok": True, "memory": {}}
    assert read_memory(client) is None

    # 清除后仍可正常使用
    resp2 = generate(client, ScriptedModel([plan_response(GOOD_PLAN)]))
    assert resp2.status_code == 200 and read_memory(client)["destination"] == "北京"
    print("T10 清除会话记忆接口 ................... OK")


def main():
    tests = [
        test_1_generate_saves_basic_info,
        test_2_feedback_history_appended,
        test_3_feedback_history_limit,
        test_4_revision_count_increases,
        test_5_constraint_failure_saves_errors,
        test_6_model_receives_history,
        test_7_new_destination_clears_history,
        test_8_sessions_are_isolated,
        test_9_no_api_key_and_small_cookie,
        test_10_clear_memory_endpoint,
        test_11_cookie_size_within_limit,
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
