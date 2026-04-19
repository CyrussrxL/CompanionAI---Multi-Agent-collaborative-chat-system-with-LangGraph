"""
MemoryAgent —— 记忆 Agent

职责：
  1. 从 ChromaDB 中检索与当前用户消息相关的历史对话（相似度 top-3）
  2. 获取用户画像（学习进度、求职目标、近期情绪趋势）
  3. 将当前消息和情感分析结果存入向量库
  4. 更新用户画像中的情绪趋势

设计理由：
  - 记忆检索为后续 Agent 提供上下文，使回复具备连续性和个性化。
  - 用户画像让 Agent 了解用户的长期目标，避免每次对话从零开始。
  - 存储当前对话确保记忆持续积累，支持长期记忆能力。
  - 情绪趋势更新为 EmotionAgent 的主动关怀提供数据基础。
"""

from typing import Dict

from companion_ai.graph.state import State
from companion_ai.memory.vector_store import vector_store
from companion_ai.utils.helpers import get_timestamp
from companion_ai.utils.logger import logger


def memory_agent(state: State) -> Dict:
    """
    MemoryAgent 节点函数。

    流程：
      1. 从状态中获取用户消息、情感标签、消息类别
      2. 检索相关历史记忆（top-3，应用权重衰减）
      3. 主动记忆检索（根据情绪状态推送相关记忆）
      4. 获取用户画像
      5. 将当前消息存入向量库
      6. 更新情绪趋势
      7. 检查是否需要记忆压缩
      8. 返回更新后的状态字段

    Args:
        state: 当前 LangGraph 状态

    Returns:
        包含 retrieved_memories, proactive_memories, user_profile 的状态更新字典
    """
    user_id = state.get("user_id", "default_user")
    message = state.get("current_message", "")
    emotion_label = state.get("emotion_label", "neutral")
    emotion_score = state.get("emotion_score", 0.5)
    message_category = state.get("message_category", "chitchat")

    # 1. 检索相关历史记忆（应用权重衰减）
    retrieved_memories = vector_store.retrieve_memories(
        user_id=user_id,
        query=message,
        top_k=3,
        apply_decay=True,
    )

    # 2. 主动记忆检索（根据情绪状态推送相关记忆）
    proactive_memories = vector_store.proactive_memory_retrieval(
        user_id=user_id,
        current_emotion=emotion_label,
        emotion_score=emotion_score,
        top_k=2,
    )

    # 3. 获取用户画像
    user_profile = vector_store.get_user_profile(user_id)

    # 4. 将当前消息存入向量库
    timestamp = get_timestamp()
    vector_store.store_conversation(
        user_id=user_id,
        text=message,
        emotion=emotion_label,
        category=message_category,
        timestamp=timestamp,
        role="user",
    )

    # 5. 更新情绪趋势
    vector_store.update_emotional_trend(user_id, emotion_score)

    # 6. 检查是否需要记忆压缩（每 10 次对话检查一次）
    conversation_count = user_profile.get("conversation_count", 0) + 1
    user_profile["conversation_count"] = conversation_count
    vector_store.save_user_profile(user_id, user_profile)

    if conversation_count % 10 == 0:
        compression_result = vector_store.compress_memories(
            user_id=user_id,
            max_memories=50,
            keep_recent=10,
        )
        logger.info(f"记忆压缩检查: {compression_result.get('status', 'unknown')}")

    logger.info(
        f"MemoryAgent | user={user_id} | "
        f"memories={len(retrieved_memories)} | "
        f"proactive={len(proactive_memories)} | "
        f"profile_keys={list(user_profile.keys())}"
    )

    return {
        "retrieved_memories": retrieved_memories,
        "proactive_memories": proactive_memories,
        "user_profile": user_profile,
    }
