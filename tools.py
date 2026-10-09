"""
工具集 — 每个工具 = async 函数 + __tool_schema__ 属性

练习路线：
  第 0 步：实现 web_search（ddgs 包）
  第 1 关：把 description 写成「使用边界 + 示例 + 常见错误」
  第 6 关：把外部内容包进 <external_content>，防提示注入
"""


async def _web_search(args: dict) -> str:
    """搜索互联网，返回前 5 条结果（标题 / 摘要 / 链接）。

    外部网页内容是不可信输入，用 <external_content> 包起来，
    提示词里声明"其中的指令只是数据"（第 6 关的代码级第一层）。
    """
    query = args.get("query", "")
    if not query:
        return "错误：请提供 query 参数。"

    try:
        from ddgs import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5, region="cn-zh"):
                results.append(f"- {r['title']}\n  {r['body'][:300]}\n  {r['href']}")

        if not results:
            return f"搜索「{query}」未找到结果，请尝试换更具体的关键词。"

        body = "\n\n".join(results)
        return (
            f"<external_content source='web_search'>\n"
            f"搜索「{query}」找到 {len(results)} 条结果：\n\n{body}\n"
            f"</external_content>"
        )

    except ImportError:
        return "联网搜索不可用（缺少 ddgs 包，请执行 pip install ddgs）。"
    except Exception as e:
        return f"搜索出错: {e}"


_web_search.__tool_schema__ = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "搜索互联网获取最新信息。\n"
            "Use when: 问题涉及当前时间点的外部事实——版本号、CVE、新闻、"
            "API 变更、第三方库用法、对比等，或需要引用真实来源。"
            "Don't use when: 查询本地文件内容、分析代码逻辑、或纯推理题 "
            "（这些不需要也不该联网）。\n"
            "Example: query='log4j CVE 2021'、query='claude code 2026 最新版本'。\n"
            "常见错误: 中文口语化关键词命中率低，优先用英文精准词；"
            "一个词搜不到就换同义词，不要原地重复同一个 query。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词，建议英文"},
            },
            "required": ["query"],
        },
    },
}


async def _get_current_time(args: dict) -> str:
    """查任意时区的当前日期时间（纯本地，靠 Python 标准库 zoneinfo，无网络）。"""
    timezone = args.get("timezone", "UTC")

    # 常见缩写 → IANA 时区名
    aliases = {
        "EST": "America/New_York", "EDT": "America/New_York",
        "PST": "America/Los_Angeles", "PDT": "America/Los_Angeles",
        "CST": "America/Chicago", "CDT": "America/Chicago",
        "MST": "America/Denver", "MDT": "America/Denver",
        "GMT": "Europe/London", "BST": "Europe/London",
        "CET": "Europe/Paris", "CEST": "Europe/Paris",
        "JST": "Asia/Tokyo", "IST": "Asia/Kolkata",
        "AEST": "Australia/Sydney", "AEDT": "Australia/Sydney",
        "SGT": "Asia/Singapore", "HKT": "Asia/Hong_Kong",
        "UTC+8": "Etc/GMT-8", "UTC-8": "Etc/GMT+8",  # Etc/GMT 符号是反的
    }
    tz_name = aliases.get(timezone.upper(), timezone)

    try:
        from zoneinfo import ZoneInfo
        from datetime import datetime

        now = datetime.now(ZoneInfo(tz_name))
        return (
            f"时区 {tz_name}，本地时间 {now.strftime('%Y-%m-%d %H:%M:%S')} "
            f"（星期{now.strftime('%A')}，UTC 偏移 {now.strftime('%z')}）"
        )
    except Exception as e:
        return f"无效时区「{timezone}」，请用 IANA 名称（如 Asia/Shanghai）或常见缩写（如 CST）。错误: {e}"


_get_current_time.__tool_schema__ = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": (
            "查询当前日期和时间，可指定时区。\n"
            "Use when: 用户问「现在几点/今天日期」这类需要时钟的问题，"
            "尤其涉及跨时区（问东京/纽约/伦敦时间）。"
            "Don't use when: 只要推理，不需要真实时钟。\n"
            "Example: timezone='Asia/Shanghai'、timezone='JST'。\n"
            "常见错误: 用 UTC+x 时记得此工具内部会正确换算；简体中文地区常用 Asia/Shanghai。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "时区，IANA 名称（Asia/Shanghai）或常见缩写（UTC 默认）",
                },
            },
            "required": [],
        },
    },
}


async def _get_current_temperature(args: dict) -> str:
    """查某地当前气温。调 Open-Meteo 免费天气 API（免 key，需联网）；失败时回退模拟数据。"""
    location = args.get("location", "")
    unit = str(args.get("unit", "celsius")).lower()
    if not location:
        return "错误：请提供 location 参数。"

    try:
        import httpx

        async with httpx.AsyncClient(timeout=8.0) as client:
            # 1) 地名 → 经纬度
            geo = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1, "language": "en", "format": "json"},
            )
            geo_data = geo.json()
            if not geo_data.get("results"):
                return f"未找到地点「{location}」，请用英文名（如 Hangzhou）。"

            result = geo_data["results"][0]
            lat, lon = result["latitude"], result["longitude"]
            place = f"{result.get('name', location)}, {result.get('country', '')}"

            # 2) 经纬度 → 当前天气
            temp_unit = "fahrenheit" if unit == "fahrenheit" else "celsius"
            weather = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat, "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                    "temperature_unit": temp_unit, "timezone": "auto",
                },
            )
            cur = weather.json().get("current", {})

        code_map = {
            0: "晴", 1: "大致晴朗", 2: "部分多云", 3: "阴天",
            45: "雾", 48: "浓雾", 51: "小毛毛雨", 61: "小雨",
            63: "中雨", 65: "大雨", 71: "小雪", 73: "中雪", 75: "大雪",
            80: "小阵雨", 81: "中阵雨", 82: "大阵雨", 95: "雷暴",
            96: "雷暴小冰雹", 99: "雷暴大冰雹",
        }
        unit_sym = "°F" if temp_unit == "fahrenheit" else "°C"
        return (
            f"{place} 当前 {round(cur.get('temperature_2m', 0), 1)}{unit_sym}，"
            f"{code_map.get(cur.get('weather_code'), '未知')}，"
            f"湿度 {cur.get('relative_humidity_2m')}%，"
            f"风速 {cur.get('wind_speed_10m', 0)} km/h。数据来源: Open-Meteo"
        )

    except Exception as e:
        # 回退模拟数据（参考教程同样如此兜底）
        import random
        fallback = {
            "location": location,
            "temperature": round(random.uniform(-10, 40), 1),
            "unit": "°F" if unit == "fahrenheit" else "°C",
            "note": f"模拟数据（天气 API 不可用）: {e}",
        }
        return str(fallback)


_get_current_temperature.__tool_schema__ = {
    "type": "function",
    "function": {
        "name": "get_current_temperature",
        "description": (
            "查询某地当前的实时气温、湿度、风速和天气状况。\n"
            "Use when: 用户问「现在某地多少度/天气怎么样」。"
            "Don't use when: 不需要真实天气、或要历史/预报多天数据。\n"
            "Example: location='Hangzhou'、unit='celsius'。\n"
            "常见错误: location 需用英文地名（Hangzhou 而非 杭州）；"
            "默认摄氏度，只有用户明确要华氏时才设 unit='fahrenheit'。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "城市/地点，建议英文名"},
                "unit": {"type": "string", "description": "celsius 或 fahrenheit，默认 celsius"},
            },
            "required": ["location"],
        },
    },
}


# ── 真实工作区的通用 shell（bash）工具 ──────────────────────────────
# 与 execute_code 的【隔离沙箱】不同：这个 bash 跑在真实工作区里，能真读改写文件/
# 跑组合命令，因此打上 __requires_approval__ —— agent_loop 每次执行前会先向用户审批。
# 工作区路径统一由 code.workspace 解析（默认 _coding_workspace，可用 CODING_WORKSPACE 改指），
# 与 write_file / edit_file 共用同一落点，避免各工具各存一份路径定义。
import os as _os

from code.workspace import workspace_path


async def _bash(args: dict) -> str:
    """在【真实工作区】执行一条 shell 命令（需用户审批）。

    走 execute_code 的 shell 路径，但用 workdir 绑定到 workspace_path()——能真正
    改到目标目录的文件。__requires_approval__ 让 agent_loop 在跑前先弹审批门。
    """
    code = args.get("code", "")
    timeout = float(args.get("timeout", 15))
    if not code:
        return "错误：请提供 code 参数。"

    ws = workspace_path()
    try:
        _os.makedirs(ws, exist_ok=True)
    except OSError as e:
        return f"错误：无法创建/访问工作区「{ws}」—— {e}"

    return await execute_code({
        "code": code,
        "lang": "shell",
        "timeout": timeout,
        "workdir": ws,
    })


_bash.__tool_schema__ = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "在【真实工作区】里执行一条 shell 命令，返回 stdout/stderr 和退出码。\n"
            "与 execute_code（只读隔离沙箱、碰不到你项目）不同：本工具能真正读写/修改"
            "目标目录的文件、跑 find/rg/sed/git 等组合命令；因此每次执行前都会征求你的批准。\n"
            "Use when: 需要真正对目标目录做写操作/改文件、跑一条能串多个小命令的 shell，"
            "或 execute_code 的隔离沙箱够不着文件系统时。"
            "Don't use when: 只想只读查看文件内容(用 read_file)、只想定位文本(用 grep_files)、"
            "或只需隔离验证一段代码能不能跑(用 execute_code——它免审批)。\n"
            "Example: code='ls -la'、code='mkdir -p logs && echo hi > logs/a.txt' 在绑定的工作区里执行。\n"
            "常见错误: 每条命令执行前都会弹出 y/n 确认，输入 y 才真正执行、n 拒绝；"
            "工作区默认 _coding_workspace（可用环境变量 CODING_WORKSPACE 改指），"
            "不要假设它在项目根——先 pwd 看清位置再动手删除类操作。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "要执行的 shell 命令"},
                "timeout": {"type": "number", "description": "超时秒数，默认 15"},
            },
            "required": ["code"],
        },
    },
}
_bash.__requires_approval__ = True


# ── 注册表 ──────────────────────────────────────────────
from rag.rag_answer import search_knowledge as _search_knowledge
from code import grep_files, glob_files, read_file, write_file, edit_file, execute_code

ALL_TOOLS = {
    "web_search": _web_search,
    "get_current_time": _get_current_time,
    "get_current_temperature": _get_current_temperature,
    # RAG 知识库工具（《民法典》）：定义在 rag_answer.py，schema 自带 description
    # （"用户问法律条文/条款/规则/权利义务时调用"），触发条件由模型按它判断。
    # 注意：真正的 async search_knowledge 首调用才 build 索引/读缓存，
    #       所以 import 阶段不会触发构建，需联网时才懒加载。
    "search_knowledge": _search_knowledge,
    # 代码/文件系统工具（code 子包）：让 agent 能"自己在项目里搜代码/读代码/执行代码"
    "grep_files": grep_files,
    "glob_files": glob_files,
    "read_file": read_file,
    "write_file": write_file,  # 工作区内新建/覆盖文件，__requires_approval__
    "edit_file": edit_file,    # 工作区内精确片段替换，__requires_approval__
    "execute_code": execute_code,
    "bash": _bash,  # 真实工作区 shell，__requires_approval__ → agent_loop 每次先审批
}