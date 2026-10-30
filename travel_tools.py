"""
旅游规划工具层

把外部 API（高德地图）封装成结构统一的工具函数，供行程规划和后续的
大模型工具调用使用：

    - search_poi:  地点搜索（名称、经纬度、地址、类型）
    - get_weather: 天气查询（实况/多天预报）
    - get_route:   路线规划（步行/驾车）

所有工具都返回 JSON 可序列化的字典：
    成功: {"ok": True, "data": {...}}
    失败: {"ok": False, "error": "失败原因"}

约定：
    - 所有网络请求超时时间为 10 秒
    - 工具内部不做递归重试，失败即返回错误
    - 日志与错误信息中不包含 API Key
    - search_poi 使用内存缓存并限制请求频率，避免触发高德 QPS 限制
      （AMAP_POI_CACHE_TTL 缓存有效期，AMAP_POI_MIN_INTERVAL 最小请求间隔）
"""

import copy
import os
import re
import threading
import time
import requests
from dotenv import load_dotenv
from typing import Any, Dict, Optional, Tuple

#加载 .env 中的环境变量（override=True：以 .env 为准，覆盖系统中可能存在的同名旧变量）
load_dotenv(override=True)

# 高德地图API密钥
AMAP_KEY = os.getenv('AMAP_KEY', '')

# 统一的请求超时时间（秒）
REQUEST_TIMEOUT = 10


def _env_float(name: str, default: float) -> float:
    """读取数值型配置，缺失或非法时使用默认值"""
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# search_poi 搜索结果缓存有效期（秒），可通过环境变量 AMAP_POI_CACHE_TTL 配置
POI_CACHE_TTL = _env_float("AMAP_POI_CACHE_TTL", 86400.0)

# 两次 POI 网络请求之间的最小间隔（秒），可通过环境变量 AMAP_POI_MIN_INTERVAL 配置
POI_MIN_INTERVAL = _env_float("AMAP_POI_MIN_INTERVAL", 1.0)

# POI 结果缓存：{(keyword.strip(), city.strip()): (到期时间, 结果)}，只缓存成功结果
_POI_CACHE: Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]] = {}
_POI_CACHE_LOCK = threading.Lock()

# 上一次 POI 请求时间（time.monotonic），用于控制请求频率
_POI_RATE_LOCK = threading.Lock()
_poi_last_request_at = 0.0

# 路线规划支持的模式 -> 高德接口地址
ROUTE_ENDPOINTS = {
    "walking": "https://restapi.amap.com/v3/direction/walking",
    "driving": "https://restapi.amap.com/v3/direction/driving",
}

# 匹配URL中的key参数，用于日志脱敏
_KEY_PATTERN = re.compile(r'(key=)[^&\s]+')


def sanitize_text(text: Any) -> str:
    """去掉文本中的API Key（requests异常信息里会包含带key的完整URL）"""
    return _KEY_PATTERN.sub(r'\1***', str(text))


def _error(message: Any) -> Dict[str, Any]:
    """构造统一的失败返回值"""
    return {"ok": False, "error": sanitize_text(message)}


def _as_text(value: Any) -> str:
    """把高德返回的字段统一转成字符串（空地址会返回空列表[]）"""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "".join(str(v) for v in value)
    return str(value)


def _request_amap(url: str, params: Dict[str, Any], action: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    发起高德GET请求并做基础校验

    Args:
        url: 接口地址
        params: 请求参数（包含key，不会被打日志）
        action: 操作名称，用于拼接错误信息

    Returns:
        (data, None) 请求成功且status=1
        (None, error) 超时、网络异常、返回格式异常或接口报错
    """
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        data = response.json()
    except requests.Timeout:
        return None, f"{action}超时（超过{REQUEST_TIMEOUT}秒）"
    except Exception as e:
        return None, f"{action}请求失败：{e}"

    if not isinstance(data, dict):
        return None, f"{action}返回格式异常"
    if data.get("status") != "1":
        return None, f"{action}失败：{data.get('info')}（infocode={data.get('infocode')}）"
    return data, None


def _format_coord(point: Any) -> Optional[str]:
    """
    把 [经度, 纬度] 转成高德要求的 "经度,纬度" 字符串

    Returns:
        str: 格式化后的坐标；格式非法时返回None
    """
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return None
    try:
        lng, lat = float(point[0]), float(point[1])
    except (TypeError, ValueError):
        return None
    if not (-180 <= lng <= 180) or not (-90 <= lat <= 90):
        return None
    return f"{lng:.6f},{lat:.6f}"


def _poi_cache_key(keyword: str, city: str) -> Tuple[str, str]:
    """缓存键：关键词和城市的去空格形式"""
    return keyword.strip(), city.strip()


def _get_cached_poi(key: Tuple[str, str]) -> Optional[Dict[str, Any]]:
    """读取未过期的缓存结果，未命中返回None（返回值是副本，调用方修改不影响缓存）"""
    with _POI_CACHE_LOCK:
        entry = _POI_CACHE.get(key)
        if entry is None:
            return None
        expire_at, result = entry
        if time.monotonic() >= expire_at:
            del _POI_CACHE[key]
            return None
        return copy.deepcopy(result)


def _set_cached_poi(key: Tuple[str, str], result: Dict[str, Any]) -> None:
    """缓存成功结果（失败结果不缓存，避免瞬时故障被长期记住）"""
    with _POI_CACHE_LOCK:
        _POI_CACHE[key] = (time.monotonic() + POI_CACHE_TTL, copy.deepcopy(result))


def _wait_for_poi_slot() -> None:
    """发起POI请求前先等到与上一次请求至少间隔 POI_MIN_INTERVAL 秒"""
    global _poi_last_request_at
    with _POI_RATE_LOCK:
        delay = _poi_last_request_at + POI_MIN_INTERVAL - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        # 睡醒后再打时间戳，避免sleep过冲导致实际间隔略小于设定值
        _poi_last_request_at = time.monotonic()


def search_poi(keyword: str, city: str = "") -> Dict[str, Any]:
    """
    调用高德地点搜索API查找地点信息

    带内存缓存和请求频率控制：相同关键词+城市在缓存有效期内直接返回缓存结果；
    真正发起网络请求前保证与上一次POI请求间隔至少 POI_MIN_INTERVAL 秒，
    避免连续请求触发高德 QPS 限制（CUQPS_HAS_EXCEEDED_THE_LIMIT）。

    Args:
        keyword: 搜索关键词，地点名称（如"故宫博物院"）
        city: 城市名称，用于限定搜索范围，可为空

    Returns:
        dict: 成功时 {"ok": True, "data": {"name", "location": [经度, 纬度], "address", "type"}}
              失败时 {"ok": False, "error": "失败原因"}
    """
    if not keyword or not str(keyword).strip():
        return _error("搜索关键词为空")

    keyword = str(keyword).strip()
    city = (city or "").strip()

    cache_key = _poi_cache_key(keyword, city)
    cached = _get_cached_poi(cache_key)
    if cached is not None:
        return cached

    url = "https://restapi.amap.com/v3/place/text"
    params = {
        "key": AMAP_KEY,
        "keywords": keyword,
        "city": city,
        "output": "json",
        "offset": 1,          # 只取最相关的一条
        "page": 1,
        "citylimit": "true" if city else "false"
    }

    _wait_for_poi_slot()
    data, error = _request_amap(url, params, f"搜索地点'{keyword}'")
    if error:
        return _error(error)

    pois = data.get("pois") or []
    if not pois:
        return _error(f"未找到地点：{keyword}")

    poi = pois[0]
    location = _as_text(poi.get("location"))
    try:
        lng, lat = (float(v) for v in location.split(","))
    except (ValueError, AttributeError):
        return _error(f"地点'{keyword}'的坐标格式异常")

    result = {
        "ok": True,
        "data": {
            "name": _as_text(poi.get("name")) or keyword,
            "location": [lng, lat],
            "address": _as_text(poi.get("address")),
            "type": _as_text(poi.get("type"))
        }
    }
    _set_cached_poi(cache_key, result)
    return result


def get_weather(city: str, extensions: str = "all") -> Dict[str, Any]:
    """
    调用高德天气API查询指定城市天气

    Args:
        city: 城市名称或adcode
        extensions: "all" 查询多天预报（默认），"base" 查询实况天气

    Returns:
        dict: 成功时 {"ok": True, "data": {"city", "type": "forecast"/"live", ...}}
              其中 type="forecast" 时 data["forecasts"] 为每天的天气列表（date/dayweather/
              nighttemp/daytemp 等字段），type="live" 时 data["live"] 为实况天气字典
              失败时 {"ok": False, "error": "失败原因"}
    """
    if not city or not str(city).strip():
        return _error("查询天气的城市为空")
    if extensions not in ("all", "base"):
        return _error(f"不支持的天气查询类型：{extensions}")

    city = str(city).strip()
    url = "https://restapi.amap.com/v3/weather/weatherInfo"
    params = {
        "key": AMAP_KEY,
        "city": city,
        "extensions": extensions,
        "output": "JSON"
    }

    data, error = _request_amap(url, params, f"查询'{city}'天气")
    if error:
        return _error(error)

    if extensions == "all":
        forecasts = data.get("forecasts") or []
        casts = (forecasts[0].get("casts") or []) if forecasts else []
        if not casts:
            return _error(f"未获取到'{city}'的天气预报数据")
        return {
            "ok": True,
            "data": {
                "city": _as_text(forecasts[0].get("city")) or city,
                "type": "forecast",
                "forecasts": casts
            }
        }

    lives = data.get("lives") or []
    if not lives:
        return _error(f"未获取到'{city}'的实况天气数据")
    return {
        "ok": True,
        "data": {
            "city": _as_text(lives[0].get("city")) or city,
            "type": "live",
            "live": lives[0]
        }
    }


def get_route(origin: Any, destination: Any, mode: str = "walking") -> Dict[str, Any]:
    """
    调用高德路线规划API查询两点间的距离和耗时

    Args:
        origin: 起点坐标 [经度, 纬度]
        destination: 终点坐标 [经度, 纬度]
        mode: 出行方式，支持 "walking"（步行，默认）和 "driving"（驾车）

    Returns:
        dict: 成功时 {"ok": True, "data": {"distance": 距离(米), "duration": 耗时(秒), "mode": 出行方式}}
              查询失败时 {"ok": False, "error": "失败原因"}，不伪造路线结果
    """
    if mode not in ROUTE_ENDPOINTS:
        return _error(f"不支持的出行方式：{mode}（仅支持 {'、'.join(ROUTE_ENDPOINTS)}）")

    origin_str = _format_coord(origin)
    if origin_str is None:
        return _error("起点坐标格式错误，应为[经度, 纬度]")
    destination_str = _format_coord(destination)
    if destination_str is None:
        return _error("终点坐标格式错误，应为[经度, 纬度]")

    params = {
        "key": AMAP_KEY,
        "origin": origin_str,
        "destination": destination_str,
        "output": "json"
    }

    data, error = _request_amap(ROUTE_ENDPOINTS[mode], params, f"查询{mode}路线")
    if error:
        return _error(error)

    paths = (data.get("route") or {}).get("paths") or []
    if not paths:
        return _error("未查询到可用路线")

    try:
        distance = int(float(paths[0].get("distance")))
        duration = int(float(paths[0].get("duration")))
    except (TypeError, ValueError):
        return _error("路线结果格式异常")

    return {
        "ok": True,
        "data": {
            "distance": distance,
            "duration": duration,
            "mode": mode
        }
    }


# 工具注册表：名称 -> 可调用函数，后续供大模型工具调用使用
TOOL_REGISTRY = {
    "search_poi": search_poi,
    "get_weather": get_weather,
    "get_route": get_route,
}

# 每个工具的名称、功能描述和参数说明（JSON Schema格式，兼容大模型function calling）
TOOL_DEFINITIONS = {
    "search_poi": {
        "name": "search_poi",
        "description": "根据关键词搜索地点，返回地点名称、经纬度坐标、地址和类型，用于把行程中的地点名称转换为地图坐标",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "要搜索的地点名称，如“故宫博物院”“颐和园”"
                },
                "city": {
                    "type": "string",
                    "description": "限定搜索范围的城市名称，如“北京”，可为空字符串"
                }
            },
            "required": ["keyword"]
        }
    },
    "get_weather": {
        "name": "get_weather",
        "description": "查询指定城市的天气信息，返回多天预报或实况天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称或adcode，如“北京”"
                },
                "extensions": {
                    "type": "string",
                    "enum": ["all", "base"],
                    "description": "all查询多天预报（默认），base查询实况天气"
                }
            },
            "required": ["city"]
        }
    },
    "get_route": {
        "name": "get_route",
        "description": "查询两个坐标点之间的路线距离和预计耗时，支持步行和驾车",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "起点坐标[经度, 纬度]"
                },
                "destination": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "终点坐标[经度, 纬度]"
                },
                "mode": {
                    "type": "string",
                    "enum": ["walking", "driving"],
                    "description": "出行方式，walking为步行（默认），driving为驾车"
                }
            },
            "required": ["origin", "destination"]
        }
    }
}
