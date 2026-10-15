"""
天气查询工具 (WeatherTool)
"""
from __future__ import annotations
import hashlib
from typing import Any, Dict, Optional
from agent_harness.tools.base import BaseTool


# 预置城市基础气候特征
_CITY_WEATHER_DATA = {
    "北京": {"temp_range": (15, 26), "condition": "晴转多云", "humidity": "38%", "wind": "北风3级", "aqi": 42},
    "上海": {"temp_range": (18, 25), "condition": "小雨转阴", "humidity": "78%", "wind": "东风2级", "aqi": 35},
    "深圳": {"temp_range": (23, 30), "condition": "雷阵雨转多云", "humidity": "82%", "wind": "微风", "aqi": 22},
    "杭州": {"temp_range": (17, 24), "condition": "阴天", "humidity": "72%", "wind": "东北风2级", "aqi": 38},
    "广州": {"temp_range": (22, 29), "condition": "多云", "humidity": "75%", "wind": "南风2级", "aqi": 28},
    "成都": {"temp_range": (16, 23), "condition": "阴", "humidity": "80%", "wind": "无风", "aqi": 48},
    "武汉": {"temp_range": (18, 27), "condition": "晴朗", "humidity": "55%", "wind": "东南风3级", "aqi": 52},
}


class WeatherTool(BaseTool):
    """天气查询工具"""

    def __init__(self):
        parameters = {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "需要查询天气的城市名称，如 '北京', '上海', '深圳'",
                },
                "date": {
                    "type": "string",
                    "description": "查询日期，如 '今天', '明天', '后天' 或 '2026-10-12'，默认为 '今天'",
                },
            },
            "required": ["city"],
        }
        super().__init__(
            name="weather",
            description="获取指定城市的实时天气预报，包括气温、天气状况、湿度、风力与空气质量指数。",
            parameters=parameters,
        )

    def execute(self, city: str, date: str = "今天", **kwargs: Any) -> Dict[str, Any]:
        normalized_city = city.strip().replace("市", "")
        base_data = _CITY_WEATHER_DATA.get(normalized_city)

        if not base_data:
            # 动态生成确定性的天气数据（基于城市名称 hash 保证稳定复现）
            h = int(hashlib.md5(normalized_city.encode("utf-8")).hexdigest()[:6], 16)
            min_temp = 12 + (h % 10)
            max_temp = min_temp + 6 + (h % 8)
            conditions = ["晴天", "多云", "阴天", "微风小雨"]
            base_data = {
                "temp_range": (min_temp, max_temp),
                "condition": conditions[h % len(conditions)],
                "humidity": f"{45 + (h % 40)}%",
                "wind": f"微风{(h % 3) + 1}级",
                "aqi": 30 + (h % 50),
            }

        temp_min, temp_max = base_data["temp_range"]
        return {
            "success": True,
            "city": normalized_city,
            "date": date,
            "condition": base_data["condition"],
            "temperature": f"{temp_min}°C ~ {temp_max}°C",
            "humidity": base_data["humidity"],
            "wind": base_data["wind"],
            "air_quality": f"AQI {base_data['aqi']} (优良)",
            "summary": f"{normalized_city}{date}天气：{base_data['condition']}，气温 {temp_min}°C ~ {temp_max}°C，{base_data['wind']}，空气质量优良。",
        }
