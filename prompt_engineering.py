"""
AI驱动的旅游行程规划模块

该模块提供了与AI大语言模型交互的功能，用于生成和优化旅游行程。
主要使用DashScope API与通义千问模型进行交互。

行程统一采用结构化JSON格式生成和返回，地图等后续功能直接读取JSON中的地点，
不再通过大模型二次提取。

模型可以自主调用 travel_tools 中注册的工具（地点搜索、天气、路线）：
由模型自行决定是否调用、调用哪个，程序只负责执行已注册的工具并把结果回传给模型。

用户消息里明确写出了工具名称（如 get_weather）时，视为用户要求必须调用该工具：
模型想直接输出最终JSON但还没调用该工具时，会把缺失的工具名作为修正消息追加到对话中，
要求模型先调用再输出；受工具轮次上限约束始终没调用时抛出 ValueError，不返回行程。

行程未通过确定性约束检查时，会把违规原因作为修正消息反馈给模型自动重规划一次
（最多 MAX_REPLANS=1 次），重规划后仍未通过则抛出 ItineraryConstraintError。

Classes:
    PromptBuilder: 处理与AI模型的所有交互逻辑
"""

import os
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple
import dashscope
import json

from travel_tools import TOOL_REGISTRY, TOOL_DEFINITIONS, sanitize_text

#读取 .env 文件中的键值对，并将其加载到当前 Python 程序的环境变量中
#override=True：以 .env 为准，覆盖系统中可能存在的同名旧变量（如过期的 API Key）
load_dotenv(override=True)

# 行程统一JSON结构说明，作为提示词的一部分发给模型
ITINERARY_JSON_SCHEMA = """{
  "destination": "北京",
  "start_date": "2026-10-01",
  "end_date": "2026-10-03",
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
    }
  ],
  "estimated_total_cost": 3600,
  "notes": ["建议提前预约景点门票"]
}"""

# 工具使用说明，拼接进系统提示词
TOOL_USAGE_PROMPT = (
    "你可以根据需要调用以下工具来获取外部信息：\n"
    "- search_poi：按关键词和城市查询地点，返回名称、经纬度、地址和类型，用于确认景点位置\n"
    "- get_weather：查询城市的天气，可获取未来几天的天气预报\n"
    "- get_route：查询两个坐标点之间的距离和耗时，支持步行和驾车\n"
    "工具使用原则：\n"
    "1. 只有确实需要外部信息时才调用工具（例如需要确认景点位置、当地天气或景点之间的距离）；不需要就不要调用\n"
    "2. 工具结果返回后，要结合工具结果生成最终方案\n"
    "3. 最终必须输出下面规定结构的行程JSON，不能包含Markdown代码块或额外说明文字\n"
)

# 生成行程时的系统提示词
GENERATION_SYSTEM_PROMPT = (
    "你是一个热情专业的旅游规划助手，请根据用户输入生成合理的每日行程。\n"
    "你必须严格只返回一个JSON对象：以 { 开头、以 } 结尾，不要输出任何解释文字，"
    "也不要使用Markdown代码块。JSON必须完全符合以下结构（字段名保持一致）：\n"
    + ITINERARY_JSON_SCHEMA
    + "\n"
    + TOOL_USAGE_PROMPT
    + "生成要求：\n"
    "1. days 必须覆盖从 start_date 到 end_date 的每一天，day 从1开始编号，date 为该天日期\n"
    "2. 每天的 items 按时间先后排序，start_time/end_time 使用24小时制 HH:MM 格式\n"
    "3. place_name 使用简洁、规范、广为人知的地点名称，不要包含城市名，确保能被高德地图准确搜索到"
    "（如\"故宫博物院\"、\"颐和园\"、\"天安门广场\"）\n"
    "4. estimated_cost 为该活动所有出行人员的合计费用（元），estimated_total_cost 为整个行程所有人员的合计费用（元）\n"
    "5. notes 为字符串数组，给出实用建议（如门票预约、交通提示等）\n"
    "6. 只返回JSON，不要有任何其他内容"
)

# 优化行程时的系统提示词
OPTIMIZATION_SYSTEM_PROMPT = (
    "你是一个专业的旅游行程优化助手，用户将提供JSON格式的原始行程和反馈意见，请根据反馈优化行程。\n"
    "你必须严格只返回一个JSON对象：以 { 开头、以 } 结尾，不要输出任何解释文字，"
    "也不要使用Markdown代码块。输出结构必须与原始行程完全相同（字段名保持一致）：\n"
    + ITINERARY_JSON_SCHEMA
    + "\n"
    + TOOL_USAGE_PROMPT
    + "优化要求：\n"
    "1. 只修改需要根据反馈调整的部分，其余内容保持不变，需要时可以先调用工具核实信息\n"
    "2. place_name 使用简洁规范、能被高德地图准确搜索到的地点名称，不要包含城市名\n"
    "3. estimated_cost 为该活动所有出行人员的合计费用（元），estimated_total_cost 为整个行程所有人员的合计费用（元）\n"
    "4. 只返回JSON，不要有任何其他内容"
)

# 工具调用轮次上限（防止无限调用）
MAX_TOOL_ROUNDS = 3
# 单轮最多执行的工具调用数量
MAX_TOOL_CALLS_PER_ROUND = 3
# 约束检查失败后允许的自动重规划次数（反馈一次、重规划一次，不做无限循环）
MAX_REPLANS = 1
# 会话记忆中最多保留的历史反馈条数
MAX_MEMORY_FEEDBACK = 3

# 会话记忆的使用说明：历史信息只作参考，本次输入优先
MEMORY_USAGE_PROMPT = (
    "这些内容是当前会话中的历史信息，只能作为参考。\n"
    "如果用户本次输入与历史信息冲突，应优先服从本次输入。\n"
    "不要凭空增加用户没有提供的事实。"
)


class ItineraryConstraintError(ValueError):
    """行程未通过确定性约束检查，携带全部违规原因供上层反馈给模型重规划"""

    def __init__(self, violations):
        self.violations = list(violations)
        message = "行程约束检查失败：\n" + "\n".join(
            f"- {item}" for item in self.violations
        )
        super().__init__(message)


# 重规划时追加给模型的修正消息
REPLAN_FEEDBACK_TEMPLATE = (
    "上一版行程没有通过硬性约束检查，请重新规划。\n\n"
    "发现的问题：\n{violations}\n\n"
    "请只修改必要部分，重新调用需要的工具，并输出完整、合法且满足所有约束的行程 JSON。\n"
    "最终不要输出 Markdown 或解释文字。"
)

# 模型想直接输出最终JSON、但用户明确要求的工具还没调用时，追加给模型的修正消息
MISSING_TOOLS_TEMPLATE = (
    "你还没有调用用户明确要求的工具：{missing}。\n"
    "请先调用这些工具（可以一次调用多个），拿到工具结果后再输出最终的行程JSON。"
)


def _field(obj: Any, key: str, default: Any = None) -> Any:
    """兼容dict和SDK对象两种形式读取字段"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# 合法的24小时制时间格式
_TIME_PATTERN = re.compile(r'^([01]\d|2[0-3]):([0-5]\d)$')


def _to_date(value: Any) -> Optional[datetime]:
    """把 YYYY-MM-DD 字符串解析为日期，无法解析时返回None"""
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d")
    except ValueError:
        return None


def _to_minutes(value: Any) -> Optional[int]:
    """把 HH:MM 时间转成当天分钟数，格式不合法时返回None"""
    if not isinstance(value, str):
        return None
    match = _TIME_PATTERN.match(value.strip())
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def _format_memory_context(memory_context: Any) -> str:
    """
    把会话记忆整理成可放进提示词的历史信息文本

    Args:
        memory_context: 会话记忆字典（也接受已经整理好的字符串）

    Returns:
        str: 历史信息文本；没有可用内容时返回空字符串
    """
    if not memory_context:
        return ""
    if isinstance(memory_context, str):
        return memory_context.strip()

    lines = []
    preferences = memory_context.get("preferences")
    if preferences:
        lines.append(f"- 用户之前填写的旅行偏好：{preferences}")

    history = memory_context.get("feedback_history") or []
    if history:
        recent = [str(item) for item in history][-MAX_MEMORY_FEEDBACK:]
        lines.append("- 最近几次反馈：" + "；".join(recent))

    errors = memory_context.get("last_validation_errors") or []
    if errors:
        lines.append("- 上一次约束检查失败的原因：" + "；".join(str(item) for item in errors))

    revision_count = memory_context.get("revision_count")
    if revision_count:
        lines.append(f"- 当前行程已经修改的次数：{revision_count}")

    if not lines:
        return ""
    return "当前会话中的历史信息：\n" + "\n".join(lines) + "\n" + MEMORY_USAGE_PROMPT


def _to_number(value: Any) -> Optional[float]:
    """把值转成数字，无法转换时返回None（布尔值不算数字）"""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


class PromptBuilder:
    """
    AI行程规划器类

    该类封装了与DashScope API的交互逻辑，提供行程生成和优化功能，
    统一返回结构化的行程字典（dict）。

    模型可以在对话中自主调用 travel_tools.TOOL_REGISTRY 中注册的工具。

    Attributes:
        api_key (str): DashScope API密钥
        model (str): 使用的AI模型名称
    """

    def __init__(self):
        """
        初始化PromptBuilder实例

        从环境变量获取API密钥并设置默认模型。
        如果未找到API密钥，将抛出ValueError异常。
        """
        # 从环境变量中获取 DashScope API 密钥
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        # 检查是否成功获取到 API 密钥
        if not self.api_key:
            # 如果未获取到 API 密钥，抛出 ValueError 异常，并给出提示信息
            raise ValueError("请在环境变量中设置 DASHSCOPE_API_KEY")
        # 将获取到的 API 密钥赋值给 dashscope 库的 api_key 属性，以便后续调用 DashScope API 时使用
        dashscope.api_key = self.api_key
        # 为类的实例设置默认使用的模型为 "qwen-turbo"，同时也可以使用其他模型，如 "qwen-plus", "qwen-max"
        self.model = "qwen-turbo"

    # ---------- 模型调用 ----------

    def _call_model(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        调用一次大模型，返回响应中的message

        Args:
            messages: 对话消息列表
            tools: 工具定义列表（DashScope格式），为None时不提供工具
            temperature: 生成随机性
            max_tokens: 最大生成token数

        Returns:
            dict: 响应中的message（可能包含content和tool_calls）

        Raises:
            ValueError: 接口返回非200状态时抛出
            Exception: 网络等异常原样抛出，由上层统一处理
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "result_format": 'message',
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if tools:
            kwargs["tools"] = tools
            # auto：由模型自行决定是否调用工具，代码中不强制调用
            kwargs["tool_choice"] = "auto"

        response = dashscope.Generation.call(**kwargs)

        status_code = getattr(response, "status_code", None)
        if status_code is not None and status_code != 200:
            raise ValueError(f"模型接口调用失败（HTTP {status_code}）：{_field(response, 'message', '')}")

        output = _field(response, 'output', {}) or {}
        choices = _field(output, 'choices', []) or []
        if not choices:
            raise ValueError("模型接口未返回有效结果")
        return _field(choices[0], 'message', {}) or {}

    @staticmethod
    def _build_tool_schemas() -> List[Dict[str, Any]]:
        """把 travel_tools.TOOL_DEFINITIONS 转换成DashScope需要的tools列表格式"""
        return [
            {"type": "function", "function": definition}
            for definition in TOOL_DEFINITIONS.values()
        ]

    # ---------- 工具调用 ----------

    @staticmethod
    def _tool_message(tool_call_id: str, result: Dict[str, Any]) -> Dict[str, str]:
        """把工具执行结果包装成可追加到messages的tool消息"""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(result, ensure_ascii=False)
        }

    @staticmethod
    def _normalize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
        """把模型返回的tool_calls统一成OpenAI风格结构"""
        normalized = []
        for index, tool_call in enumerate(tool_calls or []):
            function = _field(tool_call, "function", {}) or {}
            arguments = _field(function, "arguments", "{}")
            # 有的模型直接返回对象而不是JSON字符串，统一转成字符串
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            normalized.append({
                "id": _field(tool_call, "id", "") or f"call_{index}",
                "type": "function",
                "function": {
                    "name": _field(function, "name", "") or "",
                    "arguments": arguments
                }
            })
        return normalized

    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, str]:
        """
        执行单个工具调用，任何失败都作为工具结果返回给模型（不抛异常）

        Args:
            tool_call: 统一结构后的工具调用（包含id和function）

        Returns:
            dict: tool角色消息，content为JSON字符串形式的工具结果
        """
        call_id = tool_call.get("id", "")
        function = tool_call.get("function", {})
        name = function.get("name", "")
        raw_arguments = function.get("arguments", "{}")

        # 只允许调用注册表中的工具
        func = TOOL_REGISTRY.get(name)
        if func is None:
            return self._tool_message(call_id, {
                "ok": False,
                "error": f"工具未注册：{name}，可用工具：{list(TOOL_REGISTRY)}"
            })

        # 参数必须是合法的JSON对象
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        except (json.JSONDecodeError, TypeError):
            return self._tool_message(call_id, {
                "ok": False,
                "error": f"工具{name}的参数不是合法JSON，请提供JSON格式的参数"
            })
        if not isinstance(arguments, dict):
            return self._tool_message(call_id, {
                "ok": False,
                "error": f"工具{name}的参数必须是JSON对象"
            })

        # 执行工具，失败信息也交给模型参考
        try:
            result = func(**arguments)
        except TypeError as e:
            result = {"ok": False, "error": f"工具{name}参数不匹配：{sanitize_text(e)}"}
        except Exception as e:
            result = {"ok": False, "error": f"工具{name}执行异常：{sanitize_text(e)}"}

        if not isinstance(result, dict):
            result = {"ok": False, "error": f"工具{name}返回结果格式异常"}
        return self._tool_message(call_id, result)

    @staticmethod
    def _detect_required_tools(messages: List[Dict[str, Any]]) -> List[str]:
        """
        找出用户在消息里明确点名的工具

        只检查 role="user" 的消息：系统提示词里本来就列出了全部工具名，
        不能把系统提示当成用户要求。采用精确的工具名称匹配，不做宽泛的关键词推断。

        Args:
            messages: 对话消息列表

        Returns:
            List[str]: 用户明确要求调用的工具名称（按TOOL_DEFINITIONS顺序）
        """
        user_text = "\n".join(
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        )
        return [name for name in TOOL_DEFINITIONS if name in user_text]

    @staticmethod
    def _called_tools_in_messages(messages: List[Dict[str, Any]]) -> List[str]:
        """收集对话里已经执行过的工具名称（自动重规划的第二轮不重复要求已经查过的工具）"""
        names = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            for tool_call in message.get("tool_calls") or []:
                name = _field(_field(tool_call, "function", {}), "name", "")
                if name:
                    names.append(name)
        return names

    @staticmethod
    def _missing_required_tools(required_tools: List[str], called_tools: List[str]) -> List[str]:
        """用户明确要求、但还没有执行过的工具"""
        return [name for name in required_tools if name not in called_tools]

    def _run_model_with_tools(
        self,
        messages: List[Dict[str, Any]],
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
        user_input: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        统一的模型工具调用循环

        流程：调用模型 -> 模型返回tool_calls则执行工具并把结果作为tool消息回传 ->
        再次调用模型 -> 直到模型不再调用工具，解析其最终输出的行程JSON。

        用户明确要求的工具（见_detect_required_tools）必须被调用：模型想直接输出
        最终JSON但还缺这些工具时，会追加修正消息要求先调用，然后继续原有循环；
        轮次上限内始终没调用时抛出ValueError，不把行程返回给调用方。
        工具执行过就算已调用（返回ok=False也算），不会反复要求同一个工具。

        Args:
            messages: 对话消息列表
            max_tool_rounds: 最多允许的工具调用轮次（上限为MAX_TOOL_ROUNDS=3）
            user_input: 用户原始需求，用于最终行程的约束检查

        Returns:
            dict: 通过解析、字段校验和约束检查的行程字典

        Raises:
            ValueError: 无法得到合法行程JSON、约束检查不通过、或用户明确要求的工具始终未被执行时抛出
        """
        # 局部导入避免 agent_graph 复用本模块工具辅助函数时产生循环导入。
        from agent_graph import run_agent

        raw_text = run_agent(self, messages, max_tool_rounds)
        return self._finalize_itinerary(raw_text, user_input)

    def _run_model_with_replan(self, messages: List[Dict[str, Any]], user_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成/优化行程，并在约束检查失败时把违规原因反馈给模型自动重规划一次

        流程：调用模型（可自主调用工具）-> 解析、字段校验、约束检查 ->
        通过则返回；未通过则把违规原因作为修正消息追加到对话中，
        让模型重新规划一次（仍可自主调用工具）；第二次仍未通过时抛出最终的
        ItineraryConstraintError，不返回半成品行程。

        Args:
            messages: 对话消息列表（会被追加修正消息）
            user_input: 用户原始需求，用于约束检查

        Returns:
            dict: 通过约束检查的行程字典

        Raises:
            ItineraryConstraintError: 重规划后仍未通过约束检查时抛出
            ValueError: 模型输出无法解析为合法行程JSON（含修复失败）时抛出
            Exception: API调用失败时抛出相关异常
        """
        try:
            return self._run_model_with_tools(messages, user_input=user_input)
        except ItineraryConstraintError as first_error:
            violations = list(first_error.violations)
            print(f"[自动重规划] 首次方案未通过约束检查（{len(violations)}项问题），反馈模型重新规划（最多{MAX_REPLANS}次）")

        messages.append({
            "role": "user",
            "content": REPLAN_FEEDBACK_TEMPLATE.format(
                violations="\n".join(f"- {item}" for item in violations)
            )
        })
        # 第二次仍不通过时，由 _run_model_with_tools 抛出最终的 ItineraryConstraintError
        return self._run_model_with_tools(messages, user_input=user_input)

    # ---------- JSON解析与校验 ----------

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """
        去除模型输出中可能包裹JSON的Markdown代码块标记（```json ... ```）

        Args:
            text: 模型原始输出

        Returns:
            str: 去除代码块标记后的文本
        """
        text = (text or '').strip()
        if text.startswith('```'):
            # 去掉第一行的 ```json 或 ``` 标记
            first_newline = text.find('\n')
            if first_newline != -1:
                text = text[first_newline + 1:]
            # 去掉结尾的 ```
            if text.rstrip().endswith('```'):
                text = text.rstrip()[:-3]
        return text.strip()

    def _parse_itinerary_json(self, text: str) -> Dict[str, Any]:
        """
        从模型输出中解析出行程JSON对象

        会自动去除Markdown代码块；如果直接解析失败，
        尝试截取首个 { 到最后一个 } 之间的内容再次解析。

        Args:
            text: 模型原始输出

        Returns:
            dict: 解析出的行程字典

        Raises:
            ValueError: 无法解析出JSON对象时抛出
        """
        cleaned = self._strip_code_fence(text)
        candidates = [cleaned]
        start, end = cleaned.find('{'), cleaned.rfind('}')
        if start != -1 and end > start:
            candidates.append(cleaned[start:end + 1])

        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(data, dict):
                return data
        raise ValueError("模型输出中不包含合法的JSON对象")

    @staticmethod
    def _validate_itinerary(data: Any) -> None:
        """
        对行程JSON做基础字段校验

        Args:
            data: 待校验的行程数据

        Raises:
            ValueError: 字段缺失或类型不符合要求时抛出，并说明具体原因
        """
        if not isinstance(data, dict):
            raise ValueError("行程数据不是JSON对象")
        if not data.get('destination'):
            raise ValueError("缺少 destination（目的地）字段")

        days = data.get('days')
        if not isinstance(days, list) or len(days) == 0:
            raise ValueError("缺少 days 字段或 days 为空")

        for day in days:
            if not isinstance(day, dict):
                raise ValueError("days 中存在非对象的元素")
            day_no = day.get('day', '?')
            items = day.get('items')
            if not isinstance(items, list):
                raise ValueError(f"第{day_no}天缺少 items 列表")
            for index, item in enumerate(items, 1):
                if not isinstance(item, dict):
                    raise ValueError(f"第{day_no}天第{index}个行程项不是对象")
                place_name = item.get('place_name')
                if not isinstance(place_name, str) or not place_name.strip():
                    raise ValueError(f"第{day_no}天第{index}个行程项缺少有效的 place_name")

    @staticmethod
    def _validate_itinerary_constraints(plan: Any, user_input: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        对行程做确定性约束检查（日期、时间、预算、基本内容）

        该检查不依赖大模型判断，全部由代码规则完成；
        所有违规项都会收集到violations中一并返回，而不是发现一条就停止。

        Args:
            plan: 行程字典
            user_input: 用户原始需求（包含 start_date、end_date、budget、people 等）

        Returns:
            dict: {"valid": True, "violations": []} 或 {"valid": False, "violations": ["原因1", "原因2"]}
        """
        violations: List[str] = []
        user_input = user_input or {}

        if not isinstance(plan, dict):
            return {"valid": False, "violations": ["行程数据不是字典"]}

        # ===== 1. 日期检查 =====
        plan_start_dt = _to_date(plan.get('start_date'))
        plan_end_dt = _to_date(plan.get('end_date'))
        user_start_dt = _to_date(user_input.get('start_date'))
        user_end_dt = _to_date(user_input.get('end_date'))

        if not plan.get('start_date'):
            violations.append("行程缺少开始日期（start_date）")
        elif plan_start_dt is None:
            violations.append(f"行程的开始日期无法解析：{plan.get('start_date')}")
        if not plan.get('end_date'):
            violations.append("行程缺少结束日期（end_date）")
        elif plan_end_dt is None:
            violations.append(f"行程的结束日期无法解析：{plan.get('end_date')}")
        if plan_start_dt and plan_end_dt and plan_end_dt < plan_start_dt:
            violations.append("行程的结束日期早于开始日期")

        # 行程日期必须落在用户指定的日期范围内
        if user_start_dt and user_end_dt and user_end_dt < user_start_dt:
            violations.append("用户指定的结束日期早于开始日期")
            user_start_dt = user_end_dt = None
        if user_start_dt and plan_start_dt and plan_start_dt < user_start_dt:
            violations.append(
                f"行程的开始日期 {plan_start_dt:%Y-%m-%d} 早于用户指定的开始日期 {user_start_dt:%Y-%m-%d}"
            )
        if user_end_dt and plan_end_dt and plan_end_dt > user_end_dt:
            violations.append(
                f"行程的结束日期 {plan_end_dt:%Y-%m-%d} 晚于用户指定的结束日期 {user_end_dt:%Y-%m-%d}"
            )

        days = plan.get('days')
        if not isinstance(days, list) or not days:
            violations.append("行程中没有任何一天的安排（days为空）")
            days = []

        # 天数必须覆盖行程自身的起止日期（行程日期不可用时退回用户指定日期）
        if days:
            if plan_start_dt and plan_end_dt and plan_end_dt >= plan_start_dt:
                expected_days = (plan_end_dt - plan_start_dt).days + 1
            elif user_start_dt and user_end_dt and user_end_dt >= user_start_dt:
                expected_days = (user_end_dt - user_start_dt).days + 1
            else:
                expected_days = None
            if expected_days is not None and len(days) != expected_days:
                violations.append(
                    f"行程天数与开始日期、结束日期不一致（应为{expected_days}天，实际{len(days)}天）"
                )

        # 每天的日期：可解析、不重复、在用户范围内、连续
        day_dates: List[Tuple[int, datetime]] = []
        seen_dates: Dict[datetime, int] = {}
        for index, day in enumerate(days, 1):
            if not isinstance(day, dict):
                violations.append(f"第{index}天的数据不是对象")
                continue
            date_obj = _to_date(day.get('date'))
            if date_obj is None:
                violations.append(f"第{index}天的日期无法解析：{day.get('date')}")
                continue
            if date_obj in seen_dates:
                violations.append(f"存在重复日期：{day.get('date')}（第{seen_dates[date_obj]}天和第{index}天）")
            else:
                seen_dates[date_obj] = index
            if user_start_dt and user_end_dt and not (user_start_dt <= date_obj <= user_end_dt):
                violations.append(
                    f"第{index}天的日期 {date_obj:%Y-%m-%d} 超出用户指定的 "
                    f"{user_start_dt:%Y-%m-%d} ~ {user_end_dt:%Y-%m-%d}"
                )
            day_dates.append((index, date_obj))

        if day_dates:
            first_date = day_dates[0][1]
            if plan_start_dt and first_date != plan_start_dt:
                violations.append("第1天的日期与开始日期不一致")
            elif user_start_dt and first_date != user_start_dt:
                violations.append(
                    f"第1天的日期与用户指定的开始日期 {user_start_dt:%Y-%m-%d} 不一致"
                )
            for position in range(1, len(day_dates)):
                previous_date = day_dates[position - 1][1]
                current_index, current_date = day_dates[position]
                if current_date != previous_date + timedelta(days=1):
                    violations.append(f"第{current_index}天的日期与前一天不连续")

        # ===== 2. 每日行程检查 =====
        for index, day in enumerate(days, 1):
            if not isinstance(day, dict):
                continue
            items = day.get('items')
            if not isinstance(items, list) or not items:
                violations.append(f"第{index}天没有任何行程项（items为空）")
                continue

            time_ranges: List[Tuple[str, int, int]] = []
            last_start = None
            order_flagged = False
            for item_no, item in enumerate(items, 1):
                if not isinstance(item, dict):
                    violations.append(f"第{index}天第{item_no}个行程项不是对象")
                    continue

                # 地点名称
                place_name = item.get('place_name')
                if not isinstance(place_name, str) or not place_name.strip():
                    violations.append(f"第{index}天第{item_no}个行程项缺少地点名称（place_name）")
                    place_label = f"第{item_no}个行程项"
                else:
                    place_label = place_name.strip()

                # 费用
                if item.get('estimated_cost') is not None:
                    cost = _to_number(item.get('estimated_cost'))
                    if cost is None:
                        violations.append(f"第{index}天'{place_label}'的 estimated_cost 不是有效数字：{item.get('estimated_cost')}")
                    elif cost < 0:
                        violations.append(f"第{index}天'{place_label}'的 estimated_cost 不能小于0")

                # 时间格式
                start_raw, end_raw = item.get('start_time'), item.get('end_time')
                start_minutes = _to_minutes(start_raw)
                end_minutes = _to_minutes(end_raw)
                if start_minutes is None:
                    violations.append(f"第{index}天'{place_label}'的 start_time 不是合法的HH:MM时间：{start_raw}")
                if end_minutes is None:
                    violations.append(f"第{index}天'{place_label}'的 end_time 不是合法的HH:MM时间：{end_raw}")

                # 结束时间必须晚于开始时间
                if start_minutes is not None and end_minutes is not None:
                    if end_minutes < start_minutes:
                        violations.append(f"第{index}天'{place_label}'的结束时间早于开始时间（{start_raw}~{end_raw}）")
                    elif end_minutes == start_minutes:
                        violations.append(f"第{index}天'{place_label}'的结束时间与开始时间相同（{start_raw}）")
                    else:
                        time_ranges.append((place_label, start_minutes, end_minutes))

                # 同一天必须按开始时间升序排列
                if start_minutes is not None:
                    if last_start is not None and start_minutes < last_start and not order_flagged:
                        violations.append(f"第{index}天的行程项未按开始时间升序排列")
                        order_flagged = True
                    last_start = start_minutes

            # 同一天内不允许时间重叠（按开始时间排序后比较）
            sorted_ranges = sorted(time_ranges, key=lambda r: (r[1], r[2]))
            for position in range(1, len(sorted_ranges)):
                prev_label, _, prev_end = sorted_ranges[position - 1]
                current_label, current_start, _ = sorted_ranges[position]
                if current_start < prev_end:
                    violations.append(f"第{index}天存在时间重叠：'{prev_label}'与'{current_label}'")
                    break

        # ===== 3. 预算检查 =====
        if plan.get('total_budget') is not None:
            total_budget = _to_number(plan.get('total_budget'))
            if total_budget is None:
                violations.append(f"total_budget 不是有效数字：{plan.get('total_budget')}")
            elif total_budget < 0:
                violations.append("total_budget 不能小于0")

        estimated_total = None
        if plan.get('estimated_total_cost') is not None:
            estimated_total = _to_number(plan.get('estimated_total_cost'))
            if estimated_total is None:
                violations.append(f"estimated_total_cost 不是有效数字：{plan.get('estimated_total_cost')}")
            elif estimated_total < 0:
                violations.append("estimated_total_cost 不能小于0")

        user_budget = _to_number(user_input.get('budget'))
        if estimated_total is not None and user_budget is not None and estimated_total > user_budget:
            violations.append(
                f"预计总费用超过用户预算（预计{estimated_total:g}元 > 用户预算{user_budget:g}元）"
            )

        # ===== 4. 基本内容检查 =====
        if plan.get('people') is not None:
            people = _to_number(plan.get('people'))
            if people is None:
                violations.append(f"people 不是有效数字：{plan.get('people')}")
            elif people <= 0:
                violations.append(f"people 必须大于0（当前为{plan.get('people')}）")

        return {"valid": len(violations) == 0, "violations": violations}

    def _repair_itinerary_json(self, raw_text: str, error_detail: str) -> Dict[str, Any]:
        """
        将模型返回的异常内容再次交给模型修复一次（不提供工具）

        Args:
            raw_text: 模型原始输出
            error_detail: 首次解析/校验失败的原因

        Returns:
            dict: 修复后的行程字典

        Raises:
            ValueError: 修复后仍无法通过解析和校验时抛出
        """
        messages = [
            {
                "role": "system",
                "content": "你是JSON修复助手，只返回修复后的完整JSON对象，不要输出任何解释文字或Markdown代码块。"
            },
            {
                "role": "user",
                "content": (
                    f"以下内容本应是旅游行程JSON，但解析或校验失败，原因：{error_detail}\n\n"
                    f"原始内容：\n{raw_text}\n\n"
                    f"请修复其中的问题，并严格按照如下结构返回完整JSON：\n{ITINERARY_JSON_SCHEMA}"
                )
            }
        ]
        message = self._call_model(messages, temperature=0.1)
        raw_content = _field(message, "content", "") or ""
        data = self._parse_itinerary_json(raw_content)
        self._validate_itinerary(data)
        return data

    def _finalize_itinerary(self, raw_text: str, user_input: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        解析并校验模型给出的最终行程JSON，再执行确定性约束检查

        JSON解析/字段校验失败时自动请求模型修复一次（沿用既有机制）；
        约束检查失败时不再返回行程，而是抛出包含全部违规原因的ValueError
        （本阶段只反馈原因，不做自动重规划）。

        Args:
            raw_text: 模型最终输出的文本
            user_input: 用户原始需求，用于预算和日期范围约束

        Returns:
            dict: 符合统一结构且通过约束检查的行程字典

        Raises:
            ValueError: 解析、字段校验或约束检查失败时抛出明确错误
        """
        try:
            data = self._parse_itinerary_json(raw_text)
            self._validate_itinerary(data)
        except ValueError as first_error:
            try:
                data = self._repair_itinerary_json(raw_text, str(first_error))
            except ValueError as repair_error:
                raise ValueError(
                    f"行程JSON解析失败，自动修复后仍无效：{repair_error}"
                ) from repair_error

        # 确定性约束检查：不通过时不把行程返回给前端
        check = self._validate_itinerary_constraints(data, user_input)
        if not check["valid"]:
            raise ItineraryConstraintError(check["violations"])
        return data

    # ---------- 消息构建 ----------

    def build_messages(self, user_input: Dict[str, Any], memory_context: Any = None) -> List[Dict[str, str]]:
        """
        构建用于AI模型的对话消息列表

        Args:
            user_input: 包含用户旅游需求的字典，包括：
                - destination: 目的地
                - start_date: 开始日期
                - end_date: 结束日期
                - budget: 预算金额
                - people: 出行人数
                - preferences: 特殊偏好
            memory_context: 当前会话的历史记忆（可选），仅作参考

        Returns:
            List[Dict[str, str]]: 格式化的对话消息列表
        """
        messages = [
            # 系统消息，要求模型严格按统一JSON结构输出行程，并说明可用工具
            {
                "role": "system",
                "content": GENERATION_SYSTEM_PROMPT
            },
            # 用户消息，包含用户的具体旅游需求
            {
                "role": "user",
                "content": (
                    f"目的地: {user_input['destination']}\n"
                    f"开始日期: {user_input['start_date']}\n"
                    f"结束日期: {user_input['end_date']}\n"
                    f"总预算（元）: {user_input['budget']}\n"
                    f"人数: {user_input['people']}\n"
                    f"偏好: {user_input['preferences']}"
                )
            }
        ]
        memory_message = self._memory_message(memory_context)
        if memory_message:
            messages.insert(1, memory_message)
        return messages

    @staticmethod
    def _memory_message(memory_context: Any) -> Optional[Dict[str, str]]:
        """把会话记忆整理成一条system消息，没有记忆时返回None"""
        text = _format_memory_context(memory_context)
        if not text:
            return None
        return {"role": "system", "content": text}

    def build_optimization_messages(self, user_input: Dict[str, Any], previous_plan: Any, feedback: str,
                                    memory_context: Any = None) -> List[Dict[str, str]]:
        """
        构建用于优化行程的对话消息列表

        Args:
            user_input: 用户的旅游需求参数字典
            previous_plan: 原始行程（dict或字符串，dict会被序列化为JSON文本）
            feedback: 用户对原始行程的反馈
            memory_context: 当前会话的历史记忆（可选），仅作参考

        Returns:
            List[Dict[str, str]]: 格式化的对话消息列表
        """
        if isinstance(previous_plan, dict):
            previous_plan_text = json.dumps(previous_plan, ensure_ascii=False, indent=2)
        else:
            previous_plan_text = str(previous_plan)

        messages = [
            {
                "role": "system",
                "content": OPTIMIZATION_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    f"原始行程JSON如下：\n{previous_plan_text}\n\n"
                    f"用户反馈：{feedback}\n\n"
                    f"请保留用户原始意图（目的地：{user_input['destination']}，开始日期：{user_input['start_date']}，结束日期：{user_input['end_date']}，预算：{user_input['budget']}元，人数：{user_input['people']}，偏好：{user_input['preferences']}），在此基础上优化行程，需要时可以调用工具核实信息，并返回相同结构的完整JSON。"
                )
            }
        ]
        memory_message = self._memory_message(memory_context)
        if memory_message:
            messages.insert(1, memory_message)
        return messages

    # ---------- 对外接口 ----------

    def generate_itinerary(self, user_input: Dict[str, Any], memory_context: Any = None) -> Dict[str, Any]:
        """
        生成新的旅游行程

        模型可以根据目的地、日期和偏好自主决定是否调用地点、天气或路线工具，
        最终返回统一结构的行程JSON（不调用工具的普通回复同样支持）。
        返回前会执行确定性约束检查；首次未通过时把违规原因反馈给模型
        自动重规划一次（MAX_REPLANS=1），仍不通过的行程不会返回给前端。

        Args:
            user_input: 用户的旅游需求参数字典

        Returns:
            dict: 符合统一结构且通过约束检查的行程字典

        Raises:
            ItineraryConstraintError: 重规划后仍未通过约束检查时抛出
            ValueError: 模型输出无法解析为合法行程JSON（含修复失败）时抛出
            Exception: API调用失败时抛出相关异常
        """
        messages = self.build_messages(user_input, memory_context=memory_context)
        return self._run_model_with_replan(messages, user_input)

    def optimize_itinerary(self, user_input: Dict[str, Any], previous_plan: Any, feedback: str,
                           memory_context: Any = None) -> Dict[str, Any]:
        """
        根据用户反馈优化现有行程

        模型可以根据反馈自主决定是否重新查询相关信息（天气、地点、路线），
        优化后仍返回与原有结构一致的行程JSON。
        返回前会执行确定性约束检查；首次未通过时把违规原因反馈给模型
        自动重规划一次（MAX_REPLANS=1），仍不通过的行程不会返回给前端。

        Args:
            user_input: 用户的旅游需求参数字典
            previous_plan: 原始行程（dict或字符串）
            feedback: 用户对原始行程的反馈

        Returns:
            dict: 与原结构一致的优化后行程字典

        Raises:
            ItineraryConstraintError: 重规划后仍未通过约束检查时抛出
            ValueError: 模型输出无法解析为合法行程JSON（含修复失败）时抛出
            Exception: API调用失败时抛出相关异常
        """
        messages = self.build_optimization_messages(user_input, previous_plan, feedback,
                                                    memory_context=memory_context)
        return self._run_model_with_replan(messages, user_input)

