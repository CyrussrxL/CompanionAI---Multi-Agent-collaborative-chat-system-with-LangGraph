"""
ResponseComposer —— 响应合成器节点

职责：
  1. 将各 Agent 的回复（coding_response / career_response / general_response）
     与情绪关怀语句拼接成最终的 final_response
  2. 将 assistant 的回复存入记忆，保持对话历史的完整性
  3. 支持日报生成功能（长期记忆的总结能力）

设计理由：
  - ResponseComposer 作为工作流的最终节点，确保所有路径的输出都经过统一处理。
  - 情绪关怀语句前置插入，让用户第一时间感受到 AI 的关心。
  - 将情绪关怀逻辑从独立 Agent 合并到此节点，避免过度设计，
    同时保留基于历史趋势的分级关怀能力。
"""

from typing import Dict

from langchain_openai import ChatOpenAI

from companion_ai.graph.state import State
from companion_ai.memory.vector_store import vector_store
from companion_ai.utils.config import settings
from companion_ai.utils.helpers import get_timestamp
from companion_ai.utils.logger import logger


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=0.7,
        timeout=120,
        max_retries=3,
    )


def _generate_emotion_care(
    emotion_label: str,
    emotion_score: float,
    emotional_trend: list,
    message_category: str,
) -> str:
    """
    根据情绪状态和历史趋势生成关怀语句。

    逻辑：
      1. 检查历史趋势：连续 negative 超过 3 次触发深度关怀
      2. 根据当前情绪标签和分数生成分级关怀
      3. 结合消息类别调整关怀内容
    """
    consecutive_negative = 0
    for score in reversed(emotional_trend):
        if score < 0.4:
            consecutive_negative += 1
        else:
            break

    if consecutive_negative >= 3:
        return (
            "💙 我注意到你最近有些焦虑和疲惫，要不下次试试先深呼吸？"
            "我一直在这里陪着你，不用急着赶路，照顾好自己才是最重要的。"
        )

    if emotion_label == "negative":
        if emotion_score >= 0.7:
            return (
                "🤗 感觉你现在的情绪波动比较大，先别急，"
                "深呼吸一下，慢慢来，我在这里陪你。"
            )
        elif emotion_score >= 0.4:
            return (
                "😊 看起来你有些不太开心，没关系，"
                "每个人都会有低谷的时候，休息一下也许会好一些。"
            )
        else:
            return "🫂 稍微有点低落？没关系的，我在呢。"

    elif emotion_label == "positive":
        if emotion_score >= 0.7:
            return (
                "🌟 看到你状态这么好真开心！继续保持这份热情！"
            )
        else:
            return "👍 不错的心情！希望你能一直保持积极的状态～"

    else:
        if message_category == "coding":
            return "📝 来学习啦？加油！"
        elif message_category == "career":
            return "💼 求职路上有我陪你！"
        else:
            return "👋 你好呀～"


def _build_general_prompt(state: State) -> str:
    """
    构建 general_chat 节点的提示词，用于处理 emotional 和 chitchat 类别的消息。
    """
    from companion_ai.utils.helpers import format_memories, format_user_profile

    message = state.get("current_message", "")
    emotion_label = state.get("emotion_label", "neutral")
    emotion_score = state.get("emotion_score", 0.5)
    memories = state.get("retrieved_memories", [])
    profile = state.get("user_profile", {})

    memories_text = format_memories(memories)
    profile_text = format_user_profile(profile)

    style_instruction = ""
    if emotion_label == "negative":
        style_instruction = (
            "用户当前情绪低落，请用温柔、富有同理心的语气回复，"
            "先倾听和共情，再给出建议。"
        )
    elif emotion_label == "positive":
        style_instruction = (
            "用户当前情绪积极，可以用活泼、鼓励的语气回复。"
        )

    prompt = f"""你是一位温暖、智能的聊天伙伴，名叫 CompanionAI。

## 用户画像
{profile_text}

## 相关历史记忆
{memories_text}

## 当前情绪状态
情绪标签: {emotion_label}, 情绪分数: {emotion_score:.2f}
{style_instruction}

## 用户消息
{message}

请给出温暖、有深度的回复。"""

    return prompt


def general_chat(state: State) -> Dict:
    """
    general_chat 节点函数，处理 emotional 和 chitchat 类别的消息。
    使用大模型通用回复，但需携带记忆和情绪感知。
    """
    user_id = state.get("user_id", "default_user")
    logger.info(f"GeneralChat | user={user_id} | 处理通用对话")

    prompt = _build_general_prompt(state)

    try:
        llm = _get_llm()
        response = llm.invoke(prompt)
        general_response = response.content
    except Exception as e:
        logger.error(f"GeneralChat | LLM 调用失败: {e}")
        general_response = "抱歉，我暂时无法回复，请稍后再试。"

    return {"general_response": general_response}


def response_composer(state: State) -> Dict:
    """
    ResponseComposer 节点函数。

    流程：
      1. 根据情绪状态和历史趋势生成关怀语句
      2. 获取各 Agent 的专业回复
      3. 拼接为 final_response（情绪关怀前置）
      4. 将 assistant 回复存入记忆

    拼接格式：
      [情绪关怀语句]

      ---

      [主要回复内容]
    """
    user_id = state.get("user_id", "default_user")
    emotion_label = state.get("emotion_label", "neutral")
    emotion_score = state.get("emotion_score", 0.5)
    message_category = state.get("message_category", "chitchat")
    user_profile = state.get("user_profile", {})

    # 1. 生成情绪关怀语句
    emotional_trend = user_profile.get("emotional_trend", [])
    emotion_care = _generate_emotion_care(
        emotion_label=emotion_label,
        emotion_score=emotion_score,
        emotional_trend=emotional_trend,
        message_category=message_category,
    )

    # 2. 获取专业回复
    main_response = ""
    if message_category == "coding":
        main_response = state.get("coding_response", "")
    elif message_category == "career":
        main_response = state.get("career_response", "")
    else:
        main_response = state.get("general_response", "")

    if not main_response:
        main_response = "我暂时无法处理这个请求，请稍后再试。"

    # 3. 拼接最终回复
    if emotion_care and emotion_label != "neutral":
        final_response = f"{emotion_care}\n\n---\n\n{main_response}"
    else:
        final_response = main_response

    # 4. 存储到记忆
    timestamp = get_timestamp()
    vector_store.store_conversation(
        user_id=user_id,
        text=final_response,
        emotion=emotion_label,
        category=message_category,
        timestamp=timestamp,
        role="assistant",
    )

    logger.info(
        f"ResponseComposer | user={user_id} | "
        f"category={message_category} | "
        f"emotion={emotion_label} | "
        f"final_response_len={len(final_response)}"
    )

    return {"final_response": final_response}


def generate_daily_report(state: State) -> Dict:
    """
    深度加分项：日报生成功能。
    根据用户今天的学习/情绪情况生成日报摘要。
    """
    user_id = state.get("user_id", "default_user")

    summary_text = vector_store.generate_summary(user_id, "今天的学习和情绪情况")

    prompt = f"""请根据以下用户数据，生成一份今日学习与情绪日报：

{summary_text}

日报格式：
1. 今日学习总结
2. 情绪状态分析
3. 明日建议

请用温暖、鼓励的语气撰写。"""

    try:
        llm = _get_llm()
        response = llm.invoke(prompt)
        daily_report = response.content
    except Exception as e:
        logger.error(f"日报生成失败: {e}")
        daily_report = f"日报生成失败: {str(e)}"

    logger.info(f"DailyReport | user={user_id} | 日报生成完成")

    return {"daily_report": daily_report}
