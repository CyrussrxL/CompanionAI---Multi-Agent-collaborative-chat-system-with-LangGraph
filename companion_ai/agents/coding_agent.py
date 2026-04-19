"""
CodingAgent —— 编程辅导 Agent

职责：
  当 message_category 为 coding 或消息包含代码时激活，提供：
    - 代码/算法思路解释
    - 代码审查（指出潜在 bug、风格问题、优化建议）
    - LeetCode 题目推荐和学习资源（通过 MCP 获取真实题目）
    - 主动学习建议（深度加分项：根据情绪趋势推荐调整学习计划）
    - 工具调用建议：推荐使用 MCP 工具（沙箱执行、LeetCode 题目、GitHub 代码搜索）

设计理由：
  - 使用大模型 API 生成回复，但必须携带 retrieved_memories 和 user_profile，
    确保回复具有上下文连续性和个性化。
  - 主动学习建议功能：当检测到用户情绪为 negative 且疲劳时，
    建议调整学习节奏，体现情绪驱动的智能推荐。
  - 回复风格根据情绪动态调整（深度加分项：情绪驱动的回复风格切换）。
  - 通过 MCP 协议接入外部应用（sandbox、leetcode、github），
    让编程辅导具备代码执行、真实题库、项目参考等能力。
"""

from typing import Dict

from langchain_openai import ChatOpenAI

from companion_ai.graph.state import State
from companion_ai.utils.config import settings
from companion_ai.utils.helpers import format_memories, format_user_profile
from companion_ai.utils.logger import logger


def _get_llm() -> ChatOpenAI:
    """根据配置创建 LLM 实例，支持 DeepSeek 和 OpenAI。"""
    return ChatOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=0.7,
        timeout=120,
        max_retries=3,
    )


def _build_coding_prompt(state: State) -> str:
    """
    构建 CodingAgent 的系统提示词。

    包含：
      - 角色定义
      - 用户画像信息
      - 历史记忆上下文
      - 情绪感知指令（风格切换）
      - 主动学习建议指令
      - MCP 工具推荐说明
    """
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
            "用户当前情绪较低落，请用温柔、鼓励的语气回复，"
            "避免过于技术化或生硬的表达，先关心用户的感受。"
        )
    elif emotion_label == "positive":
        style_instruction = (
            "用户当前情绪积极，可以用活泼、鼓励的语气回复，"
            "适当增加挑战性建议。"
        )

    proactive_learning = ""
    emotional_trend = profile.get("emotional_trend", [])
    if len(emotional_trend) >= 3 and all(
        s < 0.4 for s in emotional_trend[-3:]
    ):
        proactive_learning = (
            "重要：用户近期情绪持续低落，请在回复末尾主动建议调整学习节奏，"
            "例如：'我注意到你最近有些疲惫，要不先休息一下？学习是马拉松，不是短跑。'"
        )

    mcp_tools_desc = _get_mcp_tools_description()

    prompt = f"""你是一位专业的编程学习辅导老师，擅长 Python、算法和数据结构。

## 你的职责
- 解释代码和算法思路
- 审查代码（指出潜在 bug、风格问题、优化建议）
- 推荐 LeetCode 题目或学习资源
- 根据用户的学习进度提供个性化建议

## 可用工具
{mcp_tools_desc}

## 用户画像
{profile_text}

## 相关历史记忆
{memories_text}

## 当前情绪状态
情绪标签: {emotion_label}, 情绪分数: {emotion_score:.2f}
{style_instruction}

{proactive_learning}

## 用户消息
{message}

请给出专业、有针对性的回复。如果涉及代码，请给出具体的代码示例和解释。"""

    return prompt


def _get_mcp_tools_description() -> str:
    """
    获取 MCP 工具描述字符串。

    根据 MCP 启用状态动态生成工具说明。
    """
    if settings.MCP_ENABLED:
        return """你可以通过以下 MCP 工具增强辅导能力：
- execute_code_sandbox(code, language): 在安全沙箱中执行代码，验证正确性
- get_leetcode_problem(problem_id, difficulty, tag): 获取 LeetCode 真实题目信息
- search_leetcode_problems(difficulty, tag, limit): 搜索 LeetCode 题目列表
- search_github_repositories(query, sort, limit): 搜索 GitHub 开源项目
- get_github_repository_info(owner, repo): 获取 GitHub 仓库详细信息
- search_github_code(query, language, limit): 搜索 GitHub 真实代码片段

使用建议：
- 当用户需要验证代码时，建议使用 execute_code_sandbox
- 当用户需要练习题时，建议使用 search_leetcode_problems
- 当用户需要项目参考时，建议使用 search_github_repositories"""
    else:
        return """当前 MCP 工具未启用，你可以使用以下本地工具：
- execute_python_code(code): 执行 Python 代码（本地执行）"""


def coding_agent(state: State) -> Dict:
    """
    CodingAgent 节点函数。

    流程：
      1. 构建包含上下文的提示词
      2. 调用大模型 API 生成回复
      3. 返回 coding_response

    Args:
        state: 当前 LangGraph 状态

    Returns:
        包含 coding_response 的状态更新字典
    """
    user_id = state.get("user_id", "default_user")
    logger.info(f"CodingAgent | user={user_id} | 开始处理编程辅导请求")

    prompt = _build_coding_prompt(state)

    try:
        llm = _get_llm()
        response = llm.invoke(prompt)
        coding_response = response.content
        logger.info(f"CodingAgent | user={user_id} | 回复生成成功")
    except Exception as e:
        logger.error(f"CodingAgent | LLM 调用失败: {e}")
        coding_response = (
            "抱歉，编程辅导服务暂时不可用。请检查 API 配置或稍后重试。\n"
            f"错误信息: {str(e)}"
        )

    return {"coding_response": coding_response}
