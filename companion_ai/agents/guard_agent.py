"""
GuardAgent —— 入口 Agent

职责：
  1. 接收用户消息，调用情感分析模块获取 emotion_label 和 emotion_score
  2. 判断消息类别（coding/career/emotional/chitchat）
  3. 将情感标签、分数和消息类别附加到状态中，供后续 Agent 使用

设计理由：
  - 作为工作流的入口节点，GuardAgent 是所有消息的第一站。
  - 情感分析前置，使得后续所有 Agent 都能感知用户情绪，
    为情绪驱动的回复风格切换提供基础。
  - 消息分类决定了条件路由的走向，是 LangGraph 条件边的核心依据。
  - 分类逻辑使用向量相似度检测，结合代码检测和关键词兜底，兼顾智能性和鲁棒性。
"""

from typing import Dict

from companion_ai.agents.behavior_analyzer import behavior_analyzer
from companion_ai.emotion.sentiment_analyzer import sentiment_analyzer
from companion_ai.graph.state import State
from companion_ai.utils.logger import logger


CAREER_KEYWORDS = [
    "实习", "简历", "面试", "offer", "求职", "招聘", "投递",
    "秋招", "春招", "内推", "笔试", "HR", "薪资", "转正",
    "海康", "大疆", "华为", "禾赛", "字节", "腾讯", "阿里",
    "美团", "百度", "小米", "网易", "京东", "快手",
    "岗位", "公司", "工作", "职业", "就业",
]

CODING_KEYWORDS = [
    "python", "代码", "算法", "bug", "调试", "leetcode", "力扣",
    "函数", "类", "递归", "排序", "二叉树", "动态规划", "dp",
    "链表", "数组", "哈希", "栈", "队列", "图", "dfs", "bfs",
    "报错", "error", "exception", "traceback", "debug",
    "编程", "程序", "变量", "循环", "条件", "继承", "多态",
    "数据结构", "时间复杂度", "空间复杂度",
]

EMOTIONAL_KEYWORDS = [
    "焦虑", "累", "烦", "难过", "崩溃", "压力", "抑郁", "沮丧",
    "开心", "谢谢", "心情", "情绪", "烦死了", "受不了",
    "想哭", "好累", "太难了", "不想学了", "迷茫",
]


def _classify_message_with_keywords(text: str) -> tuple:
    """
    基于关键词匹配对消息进行分类（兜底方案）。

    优先级：coding > career > emotional > chitchat
    理由：编程问题通常包含代码特征，优先级最高；
    求职问题关键词明确，次之；情绪问题需要关注，但可由 EmotionAgent 补充；
    其余归为闲聊。

    Returns:
        (category, confidence) 类别和置信度
    """
    text_lower = text.lower()

    coding_hits = sum(1 for kw in CODING_KEYWORDS if kw in text_lower)
    career_hits = sum(1 for kw in CAREER_KEYWORDS if kw in text_lower)
    emotional_hits = sum(1 for kw in EMOTIONAL_KEYWORDS if kw in text_lower)

    scores = {
        "coding": coding_hits,
        "career": career_hits,
        "emotional": emotional_hits,
    }

    max_category = max(scores, key=scores.get)
    max_score = scores[max_category]

    if max_score == 0:
        return "chitchat", 0.5

    # 计算置信度：最高分占总分的比例
    total_score = sum(scores.values())
    confidence = max_score / total_score if total_score > 0 else 0.5

    return max_category, min(confidence, 1.0)


def _classify_message(text: str, behavior: Dict = None) -> tuple:
    """
    基于向量相似度对消息进行分类（主方案）。

    优先级：代码检测 > 向量相似度分类 > 关键词兜底
    理由：代码特征明确时直接分类为 coding；
    向量分类能理解语义、同义词、缩写等，泛化能力强；
    关键词匹配作为兜底，确保系统在向量分类失败时仍能工作。

    Args:
        text: 用户消息
        behavior: 用户行为特征（可选）

    Returns:
        (category, confidence) 类别和置信度
    """
    # 1. 代码检测（最高优先级）
    if sentiment_analyzer.contains_code(text):
        return "coding", 0.95

    # 2. 向量相似度分类（主方案）
    try:
        from companion_ai.memory.vector_store import vector_store
        category, confidence = vector_store.classify_message_with_confidence(text, top_k=3)
        
        # 3. 行为特征调整（多模态分类）
        if behavior and confidence < 0.8:
            category, confidence = _adjust_category_with_behavior(
                category, confidence, text, behavior
            )
        
        return category, confidence
    except Exception as e:
        logger.warning(f"向量分类失败，回退到关键词方案: {e}")

    # 4. 关键词兜底
    return _classify_message_with_keywords(text)


def _adjust_category_with_behavior(
    category: str,
    confidence: float,
    text: str,
    behavior: Dict,
) -> tuple:
    """
    根据行为特征调整分类结果。

    规则：
      - 快速短消息 + 深夜 → 可能情绪化
      - 情绪化倾向高 → 提升 emotional 类别权重
      - 长消息 + 慢速输入 → 可能深思熟虑，保持原分类
    """
    emotional_tendency = behavior.get("emotional_tendency", "neutral")
    is_late_night = behavior.get("is_late_night", False)
    typing_speed = behavior.get("typing_speed", "unknown")
    message_length = behavior.get("message_length_category", "unknown")

    # 规则 1：快速短消息 + 深夜 → 可能情绪化
    if emotional_tendency == "likely_emotional" and category in ("chitchat", "career"):
        # 检查是否包含情绪相关词汇
        text_lower = text.lower()
        emotional_indicators = ["唉", "哎", "烦", "累", "难", "烦死了", "受不了"]
        if any(ind in text_lower for ind in emotional_indicators):
            logger.info(f"行为特征调整分类: {category} -> emotional (深夜/快速短消息)")
            return "emotional", confidence + 0.1

    # 规则 2：深思熟虑型 → 保持原分类，提升置信度
    if emotional_tendency == "likely_thoughtful":
        logger.info(f"行为特征确认分类: {category} (深思熟虑型)")
        return category, min(confidence + 0.05, 1.0)

    # 规则 3：深夜 + 求职相关 → 可能焦虑
    if is_late_night and category == "career" and typing_speed == "fast":
        logger.info(f"行为特征调整分类: {category} -> emotional (深夜求职焦虑)")
        return "emotional", confidence + 0.15

    return category, confidence


def guard_agent(state: State) -> Dict:
    """
    GuardAgent 节点函数。

    流程：
      1. 从状态中获取用户消息
      2. 分析用户行为特征（输入频率、消息长度、时间段）
      3. 调用情感分析器获取 emotion_label 和 emotion_score
      4. 对消息进行分类（结合行为特征）
      5. 计算分类置信度
      6. 记录分类反馈
      7. 返回更新后的状态字段

    Args:
        state: 当前 LangGraph 状态

    Returns:
        包含 emotion_label, emotion_score, message_category, user_behavior,
        classification_confidence 的状态更新字典
    """
    message = state.get("current_message", "")
    user_id = state.get("user_id", "default_user")

    # 1. 分析行为特征（多模态分类）
    behavior = behavior_analyzer.analyze_behavior(user_id, message)

    # 2. 情感分析
    emotion_label, emotion_score = sentiment_analyzer.analyze(message)

    # 3. 消息分类（结合行为特征）
    message_category, classification_confidence = _classify_message(message, behavior)

    # 4. 记录分类反馈
    classification_feedback = {
        "user_id": user_id,
        "message": message[:50],
        "category": message_category,
        "confidence": classification_confidence,
        "behavior": behavior,
        "timestamp": behavior.get("timestamp", ""),
    }

    logger.info(
        f"GuardAgent | user={user_id} | "
        f"emotion={emotion_label}({emotion_score:.2f}) | "
        f"category={message_category} (confidence={classification_confidence:.2f}) | "
        f"behavior={behavior.get('emotional_tendency', 'unknown')} | "
        f"message={message[:50]}..."
    )

    return {
        "emotion_label": emotion_label,
        "emotion_score": emotion_score,
        "message_category": message_category,
        "user_behavior": behavior,
        "classification_confidence": classification_confidence,
        "classification_feedback": [classification_feedback],
    }
