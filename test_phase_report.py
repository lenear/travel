"""
phase_report() 的离线测试（不联网、不调用模型）

覆盖：
  - 当前阶段没有任何工具调用时不报错，正常输出“工具调用 0 次”
  - 有工具调用时只统计当前阶段新增的部分，不把更早阶段的调用算进来
  - 模型自主调用与 Flask 地图接口调用分开计数
  - 两条调用路径各记录一次，同一次调用不会被重复统计

运行方式：python test_phase_report.py
"""

import io
import sys
from contextlib import redirect_stdout

import travel_tools
import real_scenario_test as rst


def make_mark(models=0, tools=0):
    return {"label": "测试阶段", "models": models, "tools": tools, "time": 0.0}


def make_run(origin, name, params, ok=True, error=""):
    return {"origin": origin, "name": name, "params": params, "ok": ok, "error": error}


def capture(func):
    """执行函数并返回它打印到 stdout 的内容"""
    buf = io.StringIO()
    with redirect_stdout(buf):
        func()
    return buf.getvalue()


def test_1_zero_tool_calls():
    """当前阶段没有工具调用：正常输出“0 次”，不报错"""
    rst.MODEL_CALLS[:] = [{"kwargs": {}, "response": None}]
    rst.TOOL_RUNS[:] = [make_run(rst.ORIGIN_MODEL, "search_poi", {"keyword": "故宫"})]

    mark = make_mark(models=0, tools=len(rst.TOOL_RUNS))
    output = capture(lambda: rst.phase_report(mark))

    assert "模型自主工具调用 0 次" in output, output
    assert "Flask 地图接口工具调用 0 次" in output, output
    assert "search_poi" not in output, "不应统计本阶段之前的历史调用"
    print("T1 阶段内0次工具调用不报错 ............... OK")


def test_2_counts_only_this_phase():
    """本阶段新增的调用按来源分别统计，并逐条打印参数和失败原因"""
    rst.MODEL_CALLS[:] = [{"kwargs": {}, "response": None} for _ in range(3)]
    rst.TOOL_RUNS[:] = [
        make_run(rst.ORIGIN_MODEL, "search_poi", {"keyword": "故宫"}),      # 更早阶段
        make_run(rst.ORIGIN_MODEL, "get_weather", {"city": "北京"}),        # 本阶段新增
        make_run(rst.ORIGIN_FLASK, "get_route",
                 {"origin": [116.397, 39.909], "destination": [116.407, 39.919], "mode": "walking"},
                 ok=False, error="未查询到可用路线"),                          # 本阶段新增
    ]
    mark = make_mark(models=1, tools=1)   # 从第1个工具调用之后开始统计

    output = capture(lambda: rst.phase_report(mark))

    assert "模型调用 2 次" in output, output
    assert "模型自主工具调用 1 次" in output, output
    assert "Flask 地图接口工具调用 1 次" in output, output
    assert "[模型] get_weather" in output and "[地图] get_route" in output, output
    assert "search_poi" not in output, "不应把更早阶段的调用算进来"
    assert "未查询到可用路线" in output, "失败的工具调用应打印具体错误"
    print("T2 只统计本阶段新增、按来源分开计数 ....... OK")


def test_3_no_double_counting():
    """模型路径与直接调用路径各记一次，同一次调用不会被重复统计"""
    saved = {name: (travel_tools.TOOL_REGISTRY[name], getattr(travel_tools, name))
             for name in rst.TOOL_NAMES}

    def fake_search_poi(keyword, city=""):
        return {"ok": True, "data": {"name": keyword}}

    try:
        rst.MODEL_CALLS[:] = []
        rst.TOOL_RUNS[:] = []
        for name in rst.TOOL_NAMES:
            travel_tools.TOOL_REGISTRY[name] = fake_search_poi
            setattr(travel_tools, name, fake_search_poi)

        rst.install_tool_recorder()
        travel_tools.TOOL_REGISTRY["search_poi"]("故宫", city="北京")   # 模型自主调用路径
        travel_tools.search_poi("颐和园", "北京")                      # Flask 地图接口路径
    finally:
        for name, (registry_func, module_func) in saved.items():
            travel_tools.TOOL_REGISTRY[name] = registry_func
            setattr(travel_tools, name, module_func)

    assert len(rst.TOOL_RUNS) == 2, f"两次调用应只记录两条，实际 {len(rst.TOOL_RUNS)} 条"
    assert [r["origin"] for r in rst.TOOL_RUNS] == [rst.ORIGIN_MODEL, rst.ORIGIN_FLASK], rst.TOOL_RUNS
    assert rst.TOOL_RUNS[0]["params"] == {"keyword": "故宫", "city": "北京"}, rst.TOOL_RUNS[0]
    assert rst.TOOL_RUNS[1]["params"] == {"keyword": "颐和园", "city": "北京"}, "位置参数也要记录"
    assert rst.TOOL_RUNS[0]["ok"] is True and rst.TOOL_RUNS[0]["error"] == ""
    print("T3 两条路径各记一次、不重复统计 ........... OK")


def main():
    tests = [test_1_zero_tool_calls, test_2_counts_only_this_phase, test_3_no_double_counting]
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
