import json
from datetime import datetime
from typing import Any, Dict, List, Optional


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def emotion_intensity(score: float) -> str:
    """
    将情感分数映射为强度等级：
    0~0.4 → 低, 0.4~0.7 → 中, 0.7~1.0 → 高
    """
    if score < 0.4:
        return "低"
    elif score < 0.7:
        return "中"
    else:
        return "高"


def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}


def safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return "{}"


def truncate_text(text: str, max_length: int = 500) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def format_memories(memories: List[Dict]) -> str:
    """
    将检索到的记忆列表格式化为可读文本，供 LLM 上下文使用。
    """
    if not memories:
        return "暂无相关历史记忆。"
    parts = []
    for i, mem in enumerate(memories, 1):
        text = mem.get("text", "")
        emotion = mem.get("emotion", "unknown")
        ts = mem.get("timestamp", "")
        parts.append(f"[记忆{i}] ({ts}, 情绪:{emotion}) {text}")
    return "\n".join(parts)


def format_user_profile(profile: Dict[str, Any]) -> str:
    """
    将用户画像格式化为可读文本。
    """
    if not profile:
        return "暂无用户画像信息。"
    lines = []
    for key, value in profile.items():
        label_map = {
            "learning_goal": "学习目标",
            "current_skill_level": "当前技能水平",
            "job_target": "求职目标",
            "emotional_trend": "近期情绪趋势",
        }
        label = label_map.get(key, key)
        if key == "emotional_trend" and isinstance(value, list):
            value = ", ".join([f"{v:.2f}" for v in value])
        lines.append(f"- {label}: {value}")
    return "\n".join(lines)
