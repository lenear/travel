"""
旅游行程规划Web应用

这个模块实现了一个基于Flask的Web应用，用于生成和优化旅游行程。
应用集成了AI模型来生成个性化的旅游建议。

主要功能：
- 提供Web界面供用户输入旅游偏好
- 生成定制化的旅游行程
- 根据用户反馈优化现有行程
- 在高德地图上显示行程路线
- 使用Flask session保存当前会话的简要旅行记忆（不保存完整行程）
"""

import os
import json
import secrets
from flask import Flask, render_template, request, jsonify, session
from prompt_engineering import PromptBuilder, ItineraryConstraintError, MAX_MEMORY_FEEDBACK
from dotenv import load_dotenv
from datetime import datetime, timedelta
import travel_tools

# 加载环境变量配置（override=True：以 .env 为准，覆盖系统中可能存在的同名旧变量）
load_dotenv(override=True)

# 初始化Flask应用实例
app = Flask(__name__)

# 会话签名密钥：不同用户的session靠它隔离，生产环境必须配置固定值
app.secret_key = os.getenv("FLASK_SECRET_KEY")
if not app.secret_key:
    app.secret_key = secrets.token_hex(32)
    print("[警告] 未配置 FLASK_SECRET_KEY，已生成临时随机密钥：仅适用于开发环境；"
          "生产环境必须配置固定的 FLASK_SECRET_KEY，否则服务重启后所有用户的会话记忆都会失效")

# 初始化行程规划器实例
planner = PromptBuilder()

# 高德地图API密钥（供页面加载JS地图使用）
AMAP_KEY = os.getenv('AMAP_KEY', '')

# 会话记忆在session中的键名
SESSION_MEMORY_KEY = "travel_memory"

# 会话记忆各字段的字符数上限（按字符计，不按字节；中文一个字算一个字符），防止Cookie过大
MEMORY_LIMITS = {
    "destination": 40,
    "date": 10,
    "preferences": 120,
    "feedback": 80,
    "error": 100,
}
# 记忆中最多的错误原因条数
MAX_MEMORY_ERRORS = 5
# 会话Cookie中session值的字节上限（浏览器单个Cookie上限约4096字节，这里留出余量）
SESSION_COOKIE_MAX_BYTES = 3500
# 裁剪时按更低的目标执行，避免裁剪结果贴着上限
SESSION_COOKIE_TARGET_BYTES = 3000


def _clip(value, limit):
    """把记忆字段转成字符串并按字符数截断，避免Cookie膨胀"""
    if value is None:
        return ""
    return str(value).strip()[:limit]


def _session_cookie_size(memory):
    """
    用Flask实际使用的会话序列化器，算出记忆写入Cookie后的真实字节数

    Flask会先json序列化（非ASCII会被转义），再做zlib压缩和base64编码，
    所以Cookie长度和记忆JSON的字节数并不相等，这里直接按真实结果度量。
    """
    serializer = app.session_interface.get_signing_serializer(app)
    return len(serializer.dumps({SESSION_MEMORY_KEY: memory}).encode("utf-8"))


def _shrink_memory(memory):
    """
    记忆过大时逐级精简，保证真实Cookie不超过 SESSION_COOKIE_MAX_BYTES

    先丢弃最旧的反馈，再丢弃最旧的错误原因，最后才缩短偏好和目的地；
    正常情况下（中文内容经zlib压缩后Cookie通常不到1KB）不会触发任何裁剪。
    """
    for key in ("feedback_history", "last_validation_errors"):
        while _session_cookie_size(memory) > SESSION_COOKIE_TARGET_BYTES and memory.get(key):
            memory[key].pop(0)
    if _session_cookie_size(memory) > SESSION_COOKIE_TARGET_BYTES:
        memory["preferences"] = memory["preferences"][:20]
    if _session_cookie_size(memory) > SESSION_COOKIE_TARGET_BYTES:
        memory["destination"] = memory["destination"][:10]
    return memory


def get_session_memory():
    """
    读取当前会话的旅行记忆

    session由Flask按用户签名隔离，不同用户的记忆互不影响。
    记忆只包含简要信息，不含完整行程和工具返回结果，也不含任何API Key。

    Returns:
        dict: 记忆字典；没有记忆时返回空字典
    """
    memory = session.get(SESSION_MEMORY_KEY)
    return memory if isinstance(memory, dict) else {}


def _same_trip(memory, user_input):
    """判断本次输入的目的地和日期是否与记忆中一致"""
    return (
        memory.get("destination") == _clip(user_input.get("destination"), MEMORY_LIMITS["destination"])
        and memory.get("start_date") == _clip(user_input.get("start_date"), MEMORY_LIMITS["date"])
        and memory.get("end_date") == _clip(user_input.get("end_date"), MEMORY_LIMITS["date"])
    )


def update_session_memory(user_input, feedback=None, violations=None):
    """
    更新当前会话的旅行记忆

    规则：
    - 用户更换目的地或日期时，清空旧的反馈历史、错误记录和修改次数；
    - 优化成功（feedback非空）：本次反馈加入 feedback_history（最多3条），revision_count加1；
    - 生成成功（feedback为空）：同一行程保留历史反馈，revision_count归零；
    - 约束检查失败（violations非空）：只保存违规原因，不改动反馈历史和修改次数。

    Args:
        user_input: 本次表单的用户需求参数
        feedback: 本次优化反馈，生成流程为None
        violations: 约束检查失败原因列表，成功时为None

    Returns:
        dict: 更新后的记忆字典
    """
    memory = get_session_memory()
    previous = memory if _same_trip(memory, user_input) else {}

    feedback_history = list(previous.get("feedback_history") or [])[-MAX_MEMORY_FEEDBACK:]
    revision_count = int(previous.get("revision_count") or 0)

    if violations:
        last_errors = [
            _clip(item, MEMORY_LIMITS["error"]) for item in list(violations)[:MAX_MEMORY_ERRORS]
        ]
    else:
        last_errors = []
        if feedback and str(feedback).strip():
            feedback_history = (feedback_history + [_clip(feedback, MEMORY_LIMITS["feedback"])])[-MAX_MEMORY_FEEDBACK:]
            revision_count += 1
        else:
            # 新生成的行程还没有被优化过
            revision_count = 0

    memory = {
        "destination": _clip(user_input.get("destination"), MEMORY_LIMITS["destination"]),
        "start_date": _clip(user_input.get("start_date"), MEMORY_LIMITS["date"]),
        "end_date": _clip(user_input.get("end_date"), MEMORY_LIMITS["date"]),
        "budget": user_input.get("budget"),
        "people": user_input.get("people"),
        "preferences": _clip(user_input.get("preferences"), MEMORY_LIMITS["preferences"]),
        "feedback_history": feedback_history,
        "last_validation_errors": last_errors,
        "revision_count": revision_count,
    }
    memory = _shrink_memory(memory)
    session[SESSION_MEMORY_KEY] = memory
    return memory


@app.route('/clear_memory', methods=['POST', 'GET'])
def clear_memory():
    """
    清除当前会话的旅行记忆

    只影响发起请求的用户自己的session，其他用户的记忆不受影响。
    """
    session.pop(SESSION_MEMORY_KEY, None)
    return jsonify({"ok": True, "memory": {}})

@app.route('/get_locations', methods=['POST'])
def get_locations():
    """
    处理获取行程地点坐标的API请求

    接收前端发送的行程JSON数据，直接读取其中的地点并转换为地理坐标。
    处理流程：
    1. 接收前端发送的行程JSON（统一结构）和目的地城市
    2. 直接读取 days[].items[].place_name（不调用大模型二次提取）
    3. 通过工具层 search_poi() 将地点名称转换为地理坐标
    4. 通过工具层 get_weather() 获取当天天气
    5. 返回带有坐标和当天天气的地点信息给前端

    请求参数:
        - itinerary: 行程JSON对象（或JSON字符串）
        - destination: 目的地城市名称（用于限定地点搜索范围）
        - start_date: 开始日期（行程JSON中缺少日期时的兜底）

    返回:
        JSON格式的地点信息列表，按天分组，每个地点包含:
        - name: 地点名称
        - location: (经度,纬度)坐标
        - address: 详细地址
        - type: 地点类型（景点、餐厅等）
        - time_block: 时间段（如"09:00-11:30"）
    """
    try:
        req_data = request.get_json(silent=True) or {}

        # 兼容前端传JSON字符串的情况
        itinerary_data = req_data.get('itinerary')
        if isinstance(itinerary_data, str):
            try:
                itinerary_data = json.loads(itinerary_data)
            except (json.JSONDecodeError, TypeError):
                itinerary_data = {}
        if not isinstance(itinerary_data, dict):
            itinerary_data = {}

        destination = req_data.get('destination', '')
        start_date = req_data.get('start_date', '')

        # 通过工具层获取多天天气预报
        weather_forecast = []
        weather_result = travel_tools.get_weather(destination, extensions="all")
        if weather_result.get("ok"):
            weather_forecast = weather_result["data"].get("forecasts", [])
        else:
            print(f"获取天气失败：{weather_result.get('error')}")

        days = []
        for i, day_data in enumerate(itinerary_data.get('days', [])):
            if not isinstance(day_data, dict):
                continue

            # 直接读取该天所有行程项的地点名称
            day_locations = []
            for item in day_data.get('items', []):
                if not isinstance(item, dict):
                    continue
                place_name = str(item.get('place_name', '')).strip()
                if not place_name:
                    continue
                # 通过工具层查询地点坐标
                poi_result = travel_tools.search_poi(place_name, destination)
                if poi_result.get("ok"):
                    poi_info = poi_result["data"]
                    # 组装时间段，便于前端在地图气泡中展示
                    time_block = ''
                    if item.get('start_time'):
                        time_block = str(item['start_time'])
                        if item.get('end_time'):
                            time_block += f"-{item['end_time']}"
                    day_locations.append({
                        "name": poi_info["name"],
                        "location": poi_info["location"],
                        "address": poi_info["address"],
                        "type": poi_info["type"],
                        "time_block": time_block
                    })
                else:
                    print(f"地点查询失败：{poi_result.get('error')}")

            # 确定当天日期：优先使用行程JSON中的date，缺失时按开始日期推算
            date_str = str(day_data.get('date') or '')
            weather = None
            if not date_str and start_date:
                date_obj = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=i)
                date_str = date_obj.strftime("%Y-%m-%d")
            # 匹配当天天气预报
            if date_str and weather_forecast:
                for w in weather_forecast:
                    if w.get("date") == date_str:
                        weather = w
                        break
            days.append({
                "locations": day_locations,
                "weather": weather,
                "date": date_str
            })

        # 将处理后的地点信息返回给前端
        # 前端将使用这些信息在地图上显示标记和路线
        return jsonify(days)
    except Exception as e:
        print(f"获取地点信息出错：{e}")
        return jsonify([])

def format_itinerary_for_display(data):
    """
    将结构化行程JSON转换为易读的文本形式，供前端文本框展示

    Args:
        data (dict): 统一结构的行程字典

    Returns:
        str: 人类可读的行程文本
    """
    if not isinstance(data, dict):
        return str(data)

    lines = []
    header = f"目的地：{data.get('destination', '')}"
    if data.get('start_date') or data.get('end_date'):
        header += f"    日期：{data.get('start_date', '')} ~ {data.get('end_date', '')}"
    lines.append(header)

    info = []
    if data.get('total_budget') is not None:
        info.append(f"总预算 {data['total_budget']} 元")
    if data.get('people') is not None:
        info.append(f"{data['people']} 人出行")
    if info:
        lines.append("    ".join(info))

    for day in data.get('days', []):
        date_str = day.get('date', '')
        day_title = f"\n第{day.get('day', '')}天"
        if date_str:
            day_title += f"（{date_str}）"
        lines.append(day_title)
        for item in day.get('items', []):
            time_range = item.get('start_time', '')
            if item.get('end_time'):
                time_range = f"{time_range}-{item['end_time']}" if time_range else item['end_time']
            cost = item.get('estimated_cost')
            cost_str = f"（预计 {cost} 元）" if cost not in (None, '', 0) else ""
            place = item.get('place_name', '')
            activity = item.get('activity', '')
            line = f"  {time_range}  {place}" if time_range else f"  {place}"
            if activity and activity != place:
                line += f"：{activity}"
            line += cost_str
            lines.append(line)

    if data.get('estimated_total_cost') is not None:
        lines.append(f"\n预计总花费：{data['estimated_total_cost']} 元（所有人员合计）")

    notes = data.get('notes')
    if notes:
        lines.append("\n温馨提示：")
        for note in notes:
            lines.append(f"- {note}")

    return "\n".join(lines).strip()

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    处理主页的路由函数
    
    支持两种HTTP方法：
    - GET：显示行程规划表单
    - POST：处理用户提交的表单数据并生成行程
    
    Returns:
        str: 渲染后的HTML页面
    """
    itinerary = None
    error_message = None

    if request.method == 'POST':
        # 获取并处理表单数据
        action = request.form.get('action')
        destination = request.form.get('destination')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        budget = int(request.form.get('budget', 0))
        people = int(request.form.get('people', 1))
        preferences = request.form.get('preferences')
        user_input = {
            'destination': destination,
            'start_date': start_date,
            'end_date': end_date,
            'budget': budget,
            'people': people,
            'preferences': preferences
        }

        # 当前会话的历史记忆（只含简要信息），作为参考传给模型
        memory = get_session_memory()
        try:
            if action == 'optimize':
                # 处理行程优化请求：隐藏字段携带的是完整行程JSON
                feedback = request.form.get('feedback', '')
                previous_plan = request.form.get('itinerary', '')
                try:
                    previous_plan = json.loads(previous_plan)
                except (json.JSONDecodeError, TypeError):
                    # 兼容非JSON的行程内容，按文本交给模型处理
                    pass
                itinerary = planner.optimize_itinerary(user_input, previous_plan, feedback,
                                                       memory_context=memory)
                update_session_memory(user_input, feedback=feedback)
            else:
                # 生成新的行程（返回统一结构的行程字典）
                itinerary = planner.generate_itinerary(user_input, memory_context=memory)
                update_session_memory(user_input)
        except ItineraryConstraintError as e:
            # 约束始终不通过：记录违规原因，供后续规划参考
            update_session_memory(user_input, violations=e.violations)
            error_message = f"生成行程时出错：{e}"
        except Exception as e:
            error_message = f"生成行程时出错：{e}"

    # 行程字典转为易读文本展示，完整JSON通过模板传给地图接口和优化表单
    itinerary_text = format_itinerary_for_display(itinerary) if itinerary else ''

    return render_template(
        'index.html',
        itinerary=itinerary,
        itinerary_text=itinerary_text,
        error_message=error_message,
        amap_key=AMAP_KEY
    )

if __name__ == '__main__':
    # 获取配置的端口号，默认为5000
    port = int(os.getenv('PORT', 5000))

    # 启动Flask应用服务器
    app.run(
        host='0.0.0.0',  # 允许外部访问
        port=port,       # 使用配置的端口号
        debug=True       # 启用调试模式
    )
