"""
LangGraph 工作流定义 —— workflow.py

职责：
  定义多 Agent 协作的完整状态图，包括节点、边和条件路由。

工作流设计：
  1. guard → 情感分析 + 消息分类
  2. memory → 检索记忆 + 更新记忆
  3. 根据 message_category 条件路由：
     - coding → coding 节点
     - career → career 节点
     - emotional/chitchat → general_chat 节点
  4. 最后进入 response_composer 节点：情绪关怀 + 拼接最终回复

设计理由：
  - 使用 LangGraph 的 StateGraph 构建有向图，每个 Agent 是一个节点。
  - 条件边（add_conditional_edges）实现基于消息类别的动态路由，
    这是多 Agent 协作的核心——不同类型的消息由不同的专家 Agent 处理。
  - response_composer 节点在所有路径之后执行，整合情绪关怀与专业回复，
    确保每条消息都能获得有温度的响应。
  - 使用 MemorySaver 检查点，支持对话中断后恢复。
  - 图的入口是 guard，出口是 response_composer，形成完整的处理流水线。
"""

from typing import Literal

from langgraph.graph import END, StateGraph

from companion_ai.agents.career_agent import career_agent
from companion_ai.agents.coding_agent import coding_agent
from companion_ai.agents.guard_agent import guard_agent
from companion_ai.agents.memory_agent import memory_agent
from companion_ai.agents.response_composer import general_chat, generate_daily_report, response_composer
from companion_ai.graph.state import State
from companion_ai.utils.logger import logger


def _route_by_category(state: State) -> Literal["coding", "career", "general_chat"]:
    """
    条件路由函数：根据 message_category 决定下一个节点。

    路由规则：
      - coding → coding 节点
      - career → career 节点
      - emotional / chitchat → general_chat 节点

    设计理由：
      条件边是 LangGraph 多 Agent 架构的关键，
      它让不同类型的消息被路由到最合适的专家 Agent，
      避免了单 Agent 需要处理所有类型消息的复杂性。
    """
    category = state.get("message_category", "chitchat")
    logger.info(f"路由决策 | category={category}")

    if category == "coding":
        return "coding"
    elif category == "career":
        return "career"
    else:
        return "general_chat"


def build_graph() -> StateGraph:
    """
    构建多 Agent 协作的 LangGraph 状态图。

    节点：
      - guard: 入口 Agent，情感分析 + 分类
      - memory: 记忆 Agent，检索 + 存储
      - coding: 编程辅导 Agent
      - career: 求职成长 Agent
      - general_chat: 通用对话节点
      - response_composer: 响应合成器，情绪关怀 + 拼接最终回复

    边：
      - guard → memory（固定边）
      - memory → coding/career/general_chat（条件边）
      - coding → response_composer（固定边）
      - career → response_composer（固定边）
      - general_chat → response_composer（固定边）
      - response_composer → END（固定边）
    """
    graph = StateGraph(State)

    graph.add_node("guard", guard_agent)
    graph.add_node("memory", memory_agent)
    graph.add_node("coding", coding_agent)
    graph.add_node("career", career_agent)
    graph.add_node("general_chat", general_chat)
    graph.add_node("response_composer", response_composer)

    graph.set_entry_point("guard")

    graph.add_edge("guard", "memory")

    graph.add_conditional_edges(
        "memory",
        _route_by_category,
        {
            "coding": "coding",
            "career": "career",
            "general_chat": "general_chat",
        },
    )

    graph.add_edge("coding", "response_composer")
    graph.add_edge("career", "response_composer")
    graph.add_edge("general_chat", "response_composer")

    graph.add_edge("response_composer", END)

    logger.info("LangGraph 工作流构建完成")
    return graph


def compile_graph():
    """
    编译工作流图，添加检查点支持。

    使用 MemorySaver 作为检查点存储，支持对话中断后恢复。
    这对于长时间对话场景（如模拟面试）尤其重要。
    """
    from langgraph.checkpoint.memory import MemorySaver

    graph = build_graph()
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("LangGraph 工作流编译完成（含检查点）")
    return compiled


def run_workflow(
    user_id: str,
    message: str,
    thread_id: str = None,
    interview_mode: bool = False,
) -> dict:
    """
    运行完整的工作流，返回最终状态。

    Args:
        user_id: 用户唯一标识
        message: 用户消息
        thread_id: 对话线程 ID（用于检查点恢复）
        interview_mode: 是否开启模拟面试模式

    Returns:
        包含 final_response 的状态字典
    """
    compiled = compile_graph()

    if thread_id is None:
        thread_id = f"thread_{user_id}"

    initial_state = {
        "user_id": user_id,
        "current_message": message,
        "interview_mode": interview_mode,
    }

    config = {"configurable": {"thread_id": thread_id}}

    logger.info(f"工作流启动 | user={user_id} | thread={thread_id}")

    result = compiled.invoke(initial_state, config=config)

    logger.info(
        f"工作流完成 | user={user_id} | "
        f"category={result.get('message_category', '')} | "
        f"emotion={result.get('emotion_label', '')}"
    )

    return result


def run_daily_report(user_id: str) -> str:
    """
    生成用户日报。

    Args:
        user_id: 用户唯一标识

    Returns:
        日报文本
    """
    state = {"user_id": user_id}
    result = generate_daily_report(state)
    return result.get("daily_report", "日报生成失败")
