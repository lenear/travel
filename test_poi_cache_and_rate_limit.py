"""
search_poi 内存缓存与请求频率控制的 mock 测试

不依赖真实网络：统一替换 travel_tools.requests.get，
验证缓存命中、最小请求间隔、配额错误不重试、超时/异常处理。
测试期间任何未经替换的真实网络调用都会被拦截并让测试失败。

运行方式：python test_poi_cache_and_rate_limit.py
"""

import sys
import time

import travel_tools

REAL_GET = travel_tools.requests.get


def _forbidden_network(*args, **kwargs):
    raise AssertionError("测试中不应发起真实网络请求")


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class Recorder:
    """记录每次 requests.get 的调用及发生时间；responder 返回异常对象时抛出该异常"""

    def __init__(self, responder):
        self.responder = responder
        self.calls = []
        self.times = []

    def __call__(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        # 用高精度时钟计时：Windows 上 time.monotonic() 刻度为15.6毫秒，量不准间隔
        self.times.append(time.perf_counter())
        result = self.responder(url, params)
        if isinstance(result, BaseException):
            raise result
        return result

    @property
    def count(self):
        return len(self.calls)


def ok_payload(name):
    return {
        "status": "1", "info": "OK", "infocode": "10000",
        "pois": [{
            "name": name, "location": "116.397428,39.90923",
            "address": "北京市东城区", "type": "风景名胜"
        }],
    }


def quota_payload():
    return {"status": "0", "info": "CUQPS_HAS_EXCEEDED_THE_LIMIT", "infocode": "10021"}


def reset_poi_state(ttl=86400.0, interval=1.0):
    """清空缓存、重置限流状态，并按测试需要调整配置"""
    travel_tools._POI_CACHE.clear()
    travel_tools._poi_last_request_at = 0.0
    travel_tools.POI_CACHE_TTL = ttl
    travel_tools.POI_MIN_INTERVAL = interval


def use_fake_requests(responder):
    recorder = Recorder(responder)
    travel_tools.requests.get = recorder
    return recorder


def test_1_same_place_requests_once():
    """相同地点（含首尾空格差异）只发起一次网络请求，命中缓存不等待，且不返回缓存本体"""
    reset_poi_state(interval=0.5)
    recorder = use_fake_requests(lambda url, params: FakeResponse(ok_payload(params["keywords"])))

    result = travel_tools.search_poi("故宫博物院", "北京")
    assert result["ok"] is True
    assert set(result) == {"ok", "data"}
    assert set(result["data"]) == {"name", "location", "address", "type"}
    assert recorder.count == 1

    started = time.monotonic()
    again = travel_tools.search_poi("  故宫博物院 ", " 北京 ")
    elapsed = time.monotonic() - started
    assert again["ok"] is True and again["data"]["name"] == "故宫博物院"
    assert recorder.count == 1, f"缓存命中不应再次请求，实际请求{recorder.count}次"
    assert elapsed < 0.25, f"缓存命中不应等待，实际耗时{elapsed:.2f}秒"

    # 调用方修改返回结果不应污染缓存
    again["data"]["name"] = "被改坏了"
    assert travel_tools.search_poi("故宫博物院", "北京")["data"]["name"] == "故宫博物院"
    assert recorder.count == 1
    print("T1 相同地点只请求一次、命中不等待 ........ OK")


def test_2_min_interval_between_different_places():
    """不同地点之间的真实网络请求满足最小间隔"""
    interval = 0.3
    reset_poi_state(interval=interval)
    recorder = use_fake_requests(lambda url, params: FakeResponse(ok_payload(params["keywords"])))

    for keyword in ("故宫", "颐和园", "天坛"):
        assert travel_tools.search_poi(keyword, "北京")["ok"] is True
    assert recorder.count == 3

    gaps = [b - a for a, b in zip(recorder.times, recorder.times[1:])]
    # 实现里的 time.monotonic() 在Windows上是15.6毫秒刻度，允许一个刻度的误差
    assert all(gap >= interval - 0.03 for gap in gaps), f"请求间隔不足：{['%.3f' % g for g in gaps]}"
    print(f"T2 不同地点满足最小间隔({interval}s) ....... OK（实际 {['%.3f' % g for g in gaps]}）")


def test_3_quota_error_no_retry():
    """高德返回配额错误时只请求一次、直接失败，且失败结果不写入缓存"""
    reset_poi_state(interval=0.05)
    recorder = use_fake_requests(lambda url, params: FakeResponse(quota_payload()))

    result = travel_tools.search_poi("故宫", "北京")
    assert result["ok"] is False
    assert "CUQPS_HAS_EXCEEDED_THE_LIMIT" in result["error"] and "10021" in result["error"]
    assert recorder.count == 1, f"不应重试，实际请求{recorder.count}次"

    # 失败不入缓存：紧接着再查同一地点应重新请求
    travel_tools.search_poi("故宫", "北京")
    assert recorder.count == 2, "失败结果不应被缓存"
    print("T3 配额错误不重试、失败不入缓存 .......... OK")


def test_4_timeout_and_exception_return_ok_false():
    """超时/异常/格式异常仍然返回 ok=False、不重试，且错误信息不泄露Key"""
    reset_poi_state(interval=0.05)

    recorder = use_fake_requests(
        lambda url, params: travel_tools.requests.exceptions.Timeout("连接超时"))
    result = travel_tools.search_poi("故宫", "北京")
    assert result["ok"] is False and "超时" in result["error"]
    assert recorder.count == 1, "超时不应重试"
    assert recorder.calls[0]["timeout"] == travel_tools.REQUEST_TIMEOUT == 10, "超时时间应仍为10秒"

    recorder = use_fake_requests(
        lambda url, params: ConnectionError(
            "连接失败 url=https://restapi.amap.com/v3/place/text?key=SECRET123456&keywords=x"))
    result = travel_tools.search_poi("天坛", "北京")
    assert result["ok"] is False and "请求失败" in result["error"]
    assert recorder.count == 1, "异常不应重试"
    assert "SECRET123456" not in result["error"] and "key=***" in result["error"], "错误信息中的Key应被脱敏"

    recorder = use_fake_requests(lambda url, params: FakeResponse(["不是字典"]))
    result = travel_tools.search_poi("长城", "北京")
    assert result["ok"] is False and "格式异常" in result["error"]
    assert recorder.count == 1
    print("T4 超时/异常返回ok=False、不重试、不泄露Key  OK")


def test_5_cache_expires():
    """缓存有效期过后重新请求（TTL=0 视为立即过期）"""
    reset_poi_state(ttl=0.0, interval=0.05)
    recorder = use_fake_requests(lambda url, params: FakeResponse(ok_payload(params["keywords"])))

    travel_tools.search_poi("故宫", "北京")
    travel_tools.search_poi("故宫", "北京")
    assert recorder.count == 2, "TTL=0时应每次重新请求"
    print("T5 缓存过期后重新请求 .................... OK")


def test_6_other_tools_unaffected():
    """get_weather/get_route 行为不变：不缓存、返回结构不变"""
    reset_poi_state(interval=0.05)

    def responder(url, params):
        if "weather" in url:
            return FakeResponse({
                "status": "1", "info": "OK",
                "forecasts": [{"city": "北京", "casts": [{"date": "2026-10-01", "dayweather": "晴"}]}],
            })
        return FakeResponse({
            "status": "1", "info": "OK",
            "route": {"paths": [{"distance": "1200", "duration": "900"}]},
        })

    recorder = use_fake_requests(responder)

    travel_tools.get_weather("北京")
    travel_tools.get_weather("北京")
    assert recorder.count == 2, "get_weather 不应被缓存"

    result = travel_tools.get_route([116.397, 39.909], [116.407, 39.919], "walking")
    assert result == {"ok": True, "data": {"distance": 1200, "duration": 900, "mode": "walking"}}
    assert recorder.count == 3
    print("T6 get_weather/get_route 行为不变 ......... OK")


def main():
    # 兜底：任何未经替换的真实网络调用都会被拦截
    travel_tools.requests.get = _forbidden_network

    tests = [
        test_1_same_place_requests_once,
        test_2_min_interval_between_different_places,
        test_3_quota_error_no_retry,
        test_4_timeout_and_exception_return_ok_false,
        test_5_cache_expires,
        test_6_other_tools_unaffected,
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
        travel_tools.requests.get = REAL_GET

    print("-" * 46)
    if failed:
        print(f"{failed}/{len(tests)} 个测试失败")
        sys.exit(1)
    print(f"全部 {len(tests)} 个测试通过")


if __name__ == "__main__":
    main()
