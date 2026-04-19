"""
User Behavior Analyzer —— 用户行为特征分析器

职责：
  1. 分析用户的输入行为特征（输入频率、消息长度、时间段等）
  2. 为入口 Agent 的多模态分类提供行为特征数据
  3. 记录用户行为历史，支持趋势分析

设计理由：
  - 行为特征可以辅助文本分类，提高分类准确性
  - 例如：快速短消息可能表示情绪化表达
  - 深夜消息可能情绪更敏感
  - 结合行为特征和文本向量，实现多模态分类
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from companion_ai.utils.logger import logger


class UserBehaviorAnalyzer:
    """
    用户行为特征分析器。

    记录并分析用户的输入行为，为分类决策提供辅助信息。
    """

    def __init__(self):
        self._message_history: Dict[str, List[Dict[str, Any]]] = {}

    def analyze_behavior(
        self,
        user_id: str,
        message: str,
        current_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        分析用户当前输入的行为特征。

        Args:
            user_id: 用户唯一标识
            message: 用户输入的消息
            current_time: 当前时间戳（可选，用于测试）

        Returns:
            行为特征字典
        """
        if user_id not in self._message_history:
            self._message_history[user_id] = []

        now = current_time or time.time()
        message_length = len(message)

        # 计算输入频率（距离上次输入的时间间隔）
        history = self._message_history[user_id]
        if history:
            last_time = history[-1]["timestamp"]
            time_since_last_message = now - last_time
        else:
            time_since_last_message = None

        # 记录当前消息
        self._message_history[user_id].append(
            {
                "timestamp": now,
                "message_length": message_length,
                "message": message[:100],  # 只保存前 100 字符
            }
        )

        # 保持历史记录不超过 50 条
        if len(self._message_history[user_id]) > 50:
            self._message_history[user_id] = self._message_history[user_id][-50:]

        # 分析行为特征
        behavior = {
            "message_length": message_length,
            "message_length_category": self._categorize_length(message_length),
            "time_of_day": self._get_time_of_day(now),
            "is_late_night": self._is_late_night(now),
        }

        if time_since_last_message is not None:
            behavior["time_since_last_message"] = round(time_since_last_message, 2)
            behavior["typing_speed"] = self._categorize_typing_speed(
                time_since_last_message
            )
        else:
            behavior["time_since_last_message"] = None
            behavior["typing_speed"] = "unknown"

        # 计算情绪化倾向（快速短消息可能表示情绪化）
        behavior["emotional_tendency"] = self._detect_emotional_tendency(behavior)

        logger.debug(
            f"UserBehavior | user={user_id} | "
            f"length={message_length} | "
            f"speed={behavior['typing_speed']} | "
            f"time={behavior['time_of_day']} | "
            f"emotional_tendency={behavior['emotional_tendency']}"
        )

        return behavior

    def _categorize_length(self, length: int) -> str:
        """将消息长度分类为 short/medium/long"""
        if length <= 10:
            return "short"
        elif length <= 50:
            return "medium"
        else:
            return "long"

    def _get_time_of_day(self, timestamp: float) -> str:
        """根据时间戳判断时间段"""
        hour = datetime.fromtimestamp(timestamp).hour
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"

    def _is_late_night(self, timestamp: float) -> bool:
        """判断是否为深夜（22:00 - 06:00）"""
        hour = datetime.fromtimestamp(timestamp).hour
        return hour >= 22 or hour < 6

    def _categorize_typing_speed(self, time_interval: float) -> str:
        """
        根据消息间隔判断输入速度。

        间隔 < 5 秒：fast（可能情绪化或急切）
        5-30 秒：normal
        > 30 秒：slow（可能在思考）
        """
        if time_interval < 5:
            return "fast"
        elif time_interval <= 30:
            return "normal"
        else:
            return "slow"

    def _detect_emotional_tendency(self, behavior: Dict[str, Any]) -> str:
        """
        检测情绪化倾向。

        快速 + 短消息 = 可能情绪化
        深夜 + 短消息 = 可能情绪化
        """
        typing_speed = behavior.get("typing_speed", "unknown")
        length_category = behavior.get("message_length_category", "unknown")
        is_late_night = behavior.get("is_late_night", False)

        if (typing_speed == "fast" and length_category == "short") or (
            is_late_night and length_category == "short"
        ):
            return "likely_emotional"
        elif typing_speed == "slow" and length_category == "long":
            return "likely_thoughtful"
        else:
            return "neutral"

    def get_user_history(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户的历史消息记录"""
        return self._message_history.get(user_id, [])

    def clear_history(self, user_id: Optional[str] = None):
        """清除用户行为历史"""
        if user_id:
            self._message_history.pop(user_id, None)
        else:
            self._message_history.clear()


# 全局单例
behavior_analyzer = UserBehaviorAnalyzer()
