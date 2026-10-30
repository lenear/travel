"""
三组真实联网测试（不使用mock）

需要外网连接和 .env 中的 DASHSCOPE_API_KEY、AMAP_KEY。

运行方式：python real_scenario_test.py

包含：
  组1 正常生成：北京3天/5000元/2人/历史文化美食
       - 观察模型是否自主调用 search_poi / get_weather / get_route
       - 用真实的 /get_locations 接口验证地图所需数据（坐标+天气）
  组2 故意制造约束问题：上海3天/300元/2人
       - 观察是否发现预算问题并自动重规划一次，而不是直接输出超预算方案
  组3 会话记忆：连续三次反馈 + 切换目的地
       - 观察反馈是否保留、revision_count 是否递增、后续方案是否参考前面的要求
       - 切换目的地后旧反馈是否被清除
  组4 模型自主工具调用：用户在要求里明确指定先查天气和两个地点的坐标再生成行程
       - 观察模型是否真的自主调用 get_weather / search_poi

工具调用会同时记录两类来源（每次调用只记一次）：
  - 模型自主调用：prompt_engineering 工具循环里通过 TOOL_REGISTRY 执行
  - Flask 地图接口调用：app.py 的 /get_locations 直接调用 travel_tools 里的工具函数
"""

import inspect
import json
import re
import time

import dashscope
import travel_tools
import prompt_engineering as pe
from prompt_engineering import PromptBuilder, ItineraryConstraintError

import app as webapp

# ---------- 记录模型调用与工具调用 ----------

MODEL_CALLS = []      # 每次模型调用的参数
TOOL_RUNS = []        # 每次工具执行 {"origin", "name", "params", "ok", "error"}
_original_model_call = dashscope.Generation.call

# 工具调用的两种来源
ORIGIN_MODEL = "model"    # 模型在工具循环里自主发起
ORIGIN_FLASK = "flask"    # app.py 的 /get_locations 直接调用工具层

# 需要记录的工具（与 travel_tools.TOOL_REGISTRY 的三个工具一致）
TOOL_NAMES = ("search_poi", "get_weather", "get_route")


def recording_model_call(**kwargs):
    """转发给真实模型，同时记录本次调用的消息和响应"""
    index = len(MODEL_CALLS)
    MODEL_CALLS.append({"kwargs": kwargs, "response": None})
    response = _original_model_call(**kwargs)
    MODEL_CALLS[index]["response"] = response
    return response


def record_tool(origin, name, params, result):
    """记录一次工具调用；ok/error 取自工具层统一返回结构"""
    ok = result.get("ok") if isinstance(result, dict) else None
    error = result.get("error", "") if isinstance(result, dict) and not ok else ""
    TOOL_RUNS.append({
        "origin": origin, "name": name, "params": params,
        "ok": ok, "error": error or ""
    })


def _bound_params(func, args, kwargs):
    """把位置参数和关键字参数统一成参数字典，便于打印（Flask 那边用的是位置参数）"""
    try:
        return dict(inspect.signature(func).bind(*args, **kwargs).arguments)
    except TypeError:
        return {"args": list(args), "kwargs": kwargs}


def _tool_wrapper(origin, name, raw):
    """包一层记录调用，并原样返回工具结果"""
    def wrapper(*args, **kwargs):
        result = raw(*args, **kwargs)
        record_tool(origin, name, _bound_params(raw, args, kwargs), result)
        return result

    wrapper._recording = True
    return wrapper


def install_tool_recorder():
    """
    给两条调用路径各装一个记录包装器，保证每次调用只记录一次：

      - 模型自主调用：prompt_engineering 用 TOOL_REGISTRY[name](**参数) 执行
      - Flask 地图接口：app.py /get_locations 用 travel_tools.search_poi(位置参数)

    两个包装器都直接调用原始函数、彼此不嵌套，所以不会重复统计。
    prompt_engineering 里是 from travel_tools import TOOL_REGISTRY，拿到的是同一个
    字典对象，替换其中的函数两边都会生效。
    """
    for name in TOOL_NAMES:
        raw = pe.TOOL_REGISTRY[name]
        if getattr(raw, "_recording", False):
            continue
        pe.TOOL_REGISTRY[name] = _tool_wrapper(ORIGIN_MODEL, name, raw)
        setattr(travel_tools, name, _tool_wrapper(ORIGIN_FLASK, name, raw))


def phase(label):
    """记录当前模型调用/工具调用数量，用于判定某个阶段发生了什么"""
    return {
        "label": label,
        "models": len(MODEL_CALLS),
        "tools": len(TOOL_RUNS),
        "time": time.time(),
    }


def print_tool_run(run):
    """打印一次工具调用的名称、参数、成功状态和失败原因"""
    tag = "模型" if run["origin"] == ORIGIN_MODEL else "地图"
    line = (f"      [{tag}] {run['name']}({json.dumps(run['params'], ensure_ascii=False)})"
            f" -> ok={run['ok']}")
    if run["ok"] is False and run["error"]:
        line += f" error={run['error']}"
    print(line)


def model_tool_runs(runs):
    """从一组工具调用里挑出模型自主发起的那部分"""
    return [r for r in runs if r["origin"] == ORIGIN_MODEL]


def phase_report(mark):
    """打印本阶段的模型调用和工具调用（分来源统计），返回(模型调用次数, 本阶段工具调用)"""
    models = len(MODEL_CALLS) - mark["models"]
    tools = TOOL_RUNS[mark["tools"]:]
    elapsed = time.time() - mark["time"]
    auto_tools = model_tool_runs(tools)
    map_tools = [r for r in tools if r["origin"] == ORIGIN_FLASK]
    print(f"    模型调用 {models} 次，模型自主工具调用 {len(auto_tools)} 次，"
          f"Flask 地图接口工具调用 {len(map_tools)} 次，耗时 {elapsed:.1f} 秒")
    for run in tools:
        print_tool_run(run)
    return models, tools


def memory_messages(mark_index):
    """取出第mark_index次模型调用中携带的记忆提示（如果有）"""
    messages = MODEL_CALLS[mark_index]["kwargs"].get("messages", [])
    return [m for m in messages if m.get("role") == "system" and "当前会话中的历史信息" in str(m.get("content"))]


def replan_feedback_text():
    """找出自动重规划时追加给模型的修正消息"""
    for call in MODEL_CALLS:
        for message in call["kwargs"].get("messages", []):
            content = str(message.get("content", ""))
            if "上一版行程没有通过硬性约束检查" in content:
                return content
    return ""


def show_plan(plan):
    print(f"    目的地 {plan.get('destination')} {plan.get('start_date')}~{plan.get('end_date')}，"
          f"预算{plan.get('total_budget')}元，人数{plan.get('people')}，"
          f"预计总花费{plan.get('estimated_total_cost')}元")
    for day in plan.get("days", []):
        items = day.get("items", [])
        places = "、".join(f"{i.get('start_time')}-{i.get('end_time')} {i.get('place_name')}" for i in items)
        print(f"    第{day.get('day')}天({day.get('date')}) {len(items)}项: {places}")


# ---------- 组1：正常生成 ----------

def scenario_1():
    print("=" * 70)
    print("组1 正常生成：北京 / 3天 / 5000元 / 2人 / 历史文化、美食")
    print("=" * 70)

    user_input = {
        "destination": "北京", "start_date": "2026-10-01", "end_date": "2026-10-03",
        "budget": 5000, "people": 2, "preferences": "历史文化、美食"
    }

    mark = phase("生成")
    try:
        plan = PromptBuilder().generate_itinerary(user_input)
    except Exception as e:
        print(f"    [失败] {type(e).__name__}: {e}")
        return None

    models, tools = phase_report(mark)
    print(f"    是否自主调用工具：{'是' if model_tool_runs(tools) else '否（本次不需要外部信息）'}")
    if models > 1:
        print(f"    注意：发生了 {models - 1} 次自动重规划（首次方案未通过约束检查）")
    show_plan(plan)

    # 用真实的 /get_locations 接口验证地图数据（真实调用高德）
    print("    -- 地图数据校验（真实调用高德接口）--")
    client = webapp.app.test_client()
    mark_map = phase("地图接口")
    resp = client.post("/get_locations", json={
        "itinerary": plan, "destination": plan.get("destination"),
        "start_date": plan.get("start_date")
    })
    _, map_runs = phase_report(mark_map)
    days = resp.get_json() or []
    total_places = sum(len(d.get("locations", [])) for d in days)
    without_coord = [loc["name"] for d in days for loc in d.get("locations", []) if not loc.get("location")]
    with_weather = sum(1 for d in days if d.get("weather"))
    print(f"    /get_locations 返回 {len(days)} 天，共 {total_places} 个地点，"
          f"{with_weather} 天有天气数据，{len(without_coord)} 个地点缺坐标")
    if without_coord:
        print(f"      缺坐标的地点（地图上不会显示）: {without_coord}")
    # 天气缺失时打印工具返回的具体错误，而不是只说“0 天有天气数据”
    if with_weather < len(days):
        failed_weather = [r for r in map_runs if r["name"] == "get_weather" and r["ok"] is False]
        if failed_weather:
            print(f"    天气数据缺失 {len(days) - with_weather} 天，get_weather 失败原因：")
            for run in failed_weather:
                print(f"      - {json.dumps(run['params'], ensure_ascii=False)} -> {run['error']}")
        else:
            print(f"    天气数据缺失 {len(days) - with_weather} 天：get_weather 调用成功，"
                  f"但预报只覆盖未来几天，未包含行程日期")
    print("    浏览器里再打开 http://127.0.0.1:5000 生成一次，确认地图上确实画出标记和路线")
    return plan


# ---------- 组2：约束问题与重规划 ----------

def scenario_2():
    print()
    print("=" * 70)
    print("组2 故意制造约束问题：上海 / 3天 / 300元 / 2人 / 主题乐园、高档餐饮")
    print("=" * 70)

    user_input = {
        "destination": "上海", "start_date": "2026-10-05", "end_date": "2026-10-07",
        "budget": 300, "people": 2, "preferences": "主题乐园、高档餐饮"
    }

    mark = phase("生成")
    try:
        plan = PromptBuilder().generate_itinerary(user_input)
    except ItineraryConstraintError as e:
        models, _ = phase_report(mark)
        print(f"    约束始终不通过，已抛出异常（未输出方案）：")
        for item in e.violations:
            print(f"      - {item}")
        print(f"    模型调用 {models} 次：首次 + {models - 1} 次重规划")
        print("    结论：系统没有把超预算方案返回给前端")
        return None
    except Exception as e:
        print(f"    [失败] {type(e).__name__}: {e}")
        return None

    models, _ = phase_report(mark)
    if models > 1:
        print(f"    触发了一次自动重规划（共 {models} 次模型调用），重规划后的方案通过约束检查")
        feedback_text = replan_feedback_text()
        if feedback_text:
            print("    -- 反馈给模型的修正消息 --")
            for line in feedback_text.splitlines():
                print(f"      {line}")
    else:
        print("    首次方案即满足 300 元预算（未触发重规划，属正常情况）")
    show_plan(plan)
    total = plan.get("estimated_total_cost")
    print(f"    校验：预计总花费 {total} 元 <= 预算 300 元：{total is not None and total <= 300}")
    return plan


# ---------- 组3：会话记忆 ----------

def extract_hidden_itinerary(html):
    """从返回的页面里取出隐藏字段中的完整行程JSON（与前端提交的一致）"""
    match = re.search(r"name=\"itinerary\" id=\"itineraryJson\" value='([^']*)'", html)
    return json.loads(match.group(1)) if match else None


def read_memory(client):
    with client.session_transaction() as sess:
        return sess.get(webapp.SESSION_MEMORY_KEY)


def scenario_3():
    print()
    print("=" * 70)
    print("组3 会话记忆：连续三次反馈 + 切换目的地")
    print("=" * 70)

    client = webapp.app.test_client()
    form = {"destination": "北京", "start_date": "2026-10-10", "end_date": "2026-10-12",
            "budget": "5000", "people": "2", "preferences": "历史文化"}

    # 首次生成
    mark = phase("首次生成")
    resp = client.post("/", data=form)
    html = resp.get_data(as_text=True)
    phase_report(mark)
    plan = extract_hidden_itinerary(html)
    memory = read_memory(client)
    if not plan:
        print("    [失败] 页面没有携带行程JSON，可能是模型未返回合法JSON或约束不通过")
        print("    页面错误信息片段:", re.sub(r"<[^>]+>", " ", html)[:300])
        return
    print(f"    session记忆: {json.dumps(memory, ensure_ascii=False)}")

    feedbacks = ["不要安排太多景点。", "尽量安排室内活动。", "每天上午十点以后出发。"]
    for index, feedback in enumerate(feedbacks, 1):
        if not plan:
            print(f"    第{index}次反馈中止：没有可用的行程JSON")
            return
        before_calls = len(MODEL_CALLS)
        resp = client.post("/", data={**form, "action": "optimize",
                                      "itinerary": json.dumps(plan, ensure_ascii=False),
                                      "feedback": feedback})
        html = resp.get_data(as_text=True)
        plan = extract_hidden_itinerary(html) or plan
        memory = read_memory(client)
        models = len(MODEL_CALLS) - before_calls
        carried = memory_messages(before_calls)
        print(f"    -- 第{index}次反馈：{feedback}")
        print(f"       模型调用 {models} 次；携带历史记忆：{'是' if carried else '否'}")
        if carried:
            text = carried[0]["content"]
            for line in text.splitlines():
                if line.startswith("- ") or "历史信息" in line:
                    print(f"         {line}")
        print(f"       memory.feedback_history={memory.get('feedback_history')}")
        print(f"       memory.revision_count={memory.get('revision_count')}")

    # 检查三次反馈是否保留、计数是否递增
    memory = read_memory(client)
    history = memory.get("feedback_history") or []
    print("    -- 记忆检查 --")
    print(f"       保留反馈条数 {len(history)}（最多3条）：{history}")
    print(f"       revision_count = {memory.get('revision_count')}（连续3次优化，应为3）")

    # 检查后续方案是否参考了前面的要求
    print("    -- 最终方案是否参考了反馈（人工核对）--")
    for day in (plan or {}).get("days", []):
        starts = [i.get("start_time", "") for i in day.get("items", [])]
        places = [i.get("place_name", "") for i in day.get("items", [])]
        early = [s for s in starts if s and s < "10:00"]
        print(f"       第{day.get('day')}天 开始时间{starts} 地点{places}"
              f"{'  ← 存在10点前出发的项目' if early else '  ← 均10点后出发'}")

    # 切换目的地
    print("    -- 切换目的地后 --")
    new_form = {**form, "destination": "杭州", "start_date": "2026-11-01", "end_date": "2026-11-03"}
    client.post("/", data=new_form)
    memory = read_memory(client)
    print(f"       session记忆: {json.dumps(memory, ensure_ascii=False)}")
    print(f"       旧反馈是否清除: {memory.get('feedback_history') == []}，"
          f"错误记录是否清除: {memory.get('last_validation_errors') == []}，"
          f"revision_count: {memory.get('revision_count')}")


# ---------- 组4：模型自主工具调用 ----------

def scenario_4():
    print()
    print("=" * 70)
    print("组4 模型自主工具调用：用户明确要求先查天气和故宫/颐和园的坐标，再生成行程")
    print("=" * 70)

    user_input = {
        "destination": "北京", "start_date": "2026-10-12", "end_date": "2026-10-14",
        "budget": 6000, "people": 2,
        "preferences": ("请先调用 get_weather 查询北京天气，并调用 search_poi 查询故宫和颐和园"
                        "的真实地址、坐标，拿到工具结果后再生成行程。")
    }
    print(f"    用户要求：{user_input['preferences']}")

    mark = phase("生成")
    try:
        plan = PromptBuilder().generate_itinerary(user_input)
    except Exception as e:
        phase_report(mark)
        print(f"    [失败] {type(e).__name__}: {e}")
        return None

    _, tools = phase_report(mark)
    auto_tools = model_tool_runs(tools)
    called = [r["name"] for r in auto_tools]
    keywords = [str(r["params"].get("keyword", "")) for r in auto_tools if r["name"] == "search_poi"]

    print("    -- 自主调用检查 --")
    print(f"       get_weather：{'已调用' if 'get_weather' in called else '未调用'}"
          f"（共 {called.count('get_weather')} 次）")
    print(f"       search_poi：{'已调用' if 'search_poi' in called else '未调用'}"
          f"（共 {called.count('search_poi')} 次），查询关键词 {keywords}")
    for place in ("故宫", "颐和园"):
        print(f"       是否查询 {place}：{'是' if any(place in kw for kw in keywords) else '否'}")

    plan_places = [i.get("place_name", "") for d in plan.get("days", []) for i in d.get("items", [])]
    print("    -- 行程是否用上工具结果（人工核对）--")
    print(f"       行程地点：{plan_places}")
    for place in ("故宫", "颐和园"):
        print(f"       {place} 出现在行程中：{'是' if any(place in p for p in plan_places) else '否'}")
    show_plan(plan)
    return plan


def main():
    print("真实联网测试开始（会消耗真实API额度），每一步的模型调用和工具调用都会被记录")
    dashscope.Generation.call = recording_model_call
    install_tool_recorder()

    try:
        scenario_1()
        scenario_2()
        scenario_3()
        scenario_4()
    except KeyboardInterrupt:
        print("\n已中断")
    finally:
        dashscope.Generation.call = _original_model_call

    print()
    print("=" * 70)
    print(f"全部结束：模型调用共 {len(MODEL_CALLS)} 次，工具调用共 {len(TOOL_RUNS)} 次")
    for origin, label in ((ORIGIN_MODEL, "模型自主工具调用"),
                          (ORIGIN_FLASK, "Flask地图接口工具调用")):
        used = {}
        for run in TOOL_RUNS:
            if run["origin"] == origin:
                used[run["name"]] = used.get(run["name"], 0) + 1
        total = sum(used.values())
        print(f"{label}：{total} 次 {used if used else ''}")
    print("=" * 70)


if __name__ == "__main__":
    main()
