"""
CareerAgent —— 求职成长 Agent

职责：
  当 message_category 为 career 或消息包含求职关键词时激活，提供：
    - 根据用户画像给出定制化建议
    - 简历修改要点（通过 MCP 接入 ATS 评分系统）
    - 模拟面试（通过 MCP 获取真实面经和题库）
    - 推荐真实实习公司列表（通过 MCP 获取最新招聘信息）
    - 模拟面试的自动评价（深度加分项：对用户回答打分 0-100 并给出改进建议）
    - 工具调用推荐：推荐使用 MCP 工具（岗位搜索、简历优化、面试经验）

设计理由：
  - CareerAgent 完全基于用户画像定制，而非通用求职建议，
    体现个性化记忆的价值。
  - 模拟面试模式通过状态中的 interview_mode 标志控制，
    支持多轮面试对话。
  - 面试评价使用 LLM 生成结构化评分和改进建议，
    帮助用户有针对性地提升。
  - 输出建议后更新 user_profile 中的求职进度。
  - 通过 MCP 协议接入外部应用（招聘、简历优化、面试经验），
    让求职辅导具备真实岗位信息、ATS 评分、真实面经等能力。
"""

import re
from typing import Dict

from langchain_openai import ChatOpenAI

from companion_ai.graph.state import State
from companion_ai.memory.vector_store import vector_store
from companion_ai.utils.config import settings
from companion_ai.utils.helpers import format_memories, format_user_profile
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


def _build_career_prompt(state: State) -> str:
    """
    构建 CareerAgent 的系统提示词。

    包含用户背景、画像信息、历史记忆和情绪感知指令。
    """
    message = state.get("current_message", "")
    emotion_label = state.get("emotion_label", "neutral")
    emotion_score = state.get("emotion_score", 0.5)
    memories = state.get("retrieved_memories", [])
    profile = state.get("user_profile", {})
    interview_mode = state.get("interview_mode", False)

    memories_text = format_memories(memories)
    profile_text = format_user_profile(profile)

    style_instruction = ""
    if emotion_label == "negative":
        style_instruction = (
            "用户当前情绪较低落，请用温暖、鼓励的语气给出建议，"
            "强调每个人都有自己的节奏，不要过于焦虑。"
        )
    elif emotion_label == "positive":
        style_instruction = (
            "用户当前情绪积极，可以用更有挑战性的建议激励用户继续前进。"
        )

    interview_instruction = ""
    if interview_mode:
        interview_instruction = """
当前处于模拟面试模式：
- 如果用户尚未回答问题，请提出一个面试问题（与 AI/算法/计算机相关）
- 如果用户正在回答问题，请对回答进行评价：
  1. 给出评分（0-100分）
  2. 指出优点
  3. 指出不足和改进建议
  4. 提供参考答案要点
"""

    mcp_tools_desc = _get_career_mcp_tools_description()

    prompt = f"""你是一位资深的求职辅导顾问，专注于帮助学生进入 AI 和科技行业。

## 用户画像
{profile_text}

## 相关历史记忆
{memories_text}

## 当前情绪状态
情绪标签: {emotion_label}, 情绪分数: {emotion_score:.2f}
{style_instruction}

{interview_instruction}

## 可用工具
{mcp_tools_desc}

## 你的能力
1. 简历优化：针对用户的求职目标，给出具体的简历修改建议（条目级别）
2. 实习投递策略：推荐适合的科技公司
3. 模拟面试：提出技术/项目/行为面试问题，评价用户回答
4. 学习路径：根据用户背景推荐学习路线

## 用户消息
{message}

请根据用户画像给出专业、有针对性的求职建议。"""

    return prompt


def _get_career_mcp_tools_description() -> str:
    """
    获取 CareerAgent MCP 工具描述字符串。

    根据 MCP 启用状态动态生成工具说明。
    """
    if settings.CAREER_MCP_ENABLED:
        return """你可以通过以下 MCP 工具增强辅导能力：
- search_job_listings(position, city, experience, limit): 搜索真实招聘岗位信息
- analyze_job_requirements(job_description): 分析岗位 JD 的技能要求
- optimize_resume(resume_text, target_position, target_company): 简历优化（ATS 评分）
- get_interview_experience(company, position, interview_type): 获取真实面试经验
- search_interview_questions(topic, difficulty, limit): 搜索面试题目
- get_salary_info(position, city, experience): 获取岗位薪资信息

使用建议：
- 当用户需要找实习时，建议使用 search_job_listings
- 当用户需要优化简历时，建议使用 optimize_resume
- 当用户准备面试时，建议使用 get_interview_experience 和 search_interview_questions
- 当用户需要了解薪资时，建议使用 get_salary_info"""
    else:
        return """当前 MCP 工具未启用，你可以使用以下本地工具：
- evaluate_resume(resume_text): 简历评分（关键词匹配）
- get_interview_questions(topic, count): 面试题库（固定题库）"""


def _extract_score(text: str) -> float:
    """
    从面试评价文本中提取评分（0-100）。
    """
    score_patterns = [
        r"评分[：:]\s*(\d+)",
        r"得分[：:]\s*(\d+)",
        r"(\d+)\s*[/／]\s*100",
        r"(\d+)\s*分",
    ]
    for pattern in score_patterns:
        match = re.search(pattern, text)
        if match:
            score = float(match.group(1))
            return min(100.0, max(0.0, score))
    return -1.0


def career_agent(state: State) -> Dict:
    """
    CareerAgent 节点函数。

    流程：
      1. 构建包含上下文的提示词
      2. 调用大模型 API 生成回复
      3. 若处于面试模式，提取评分
      4. 更新用户画像中的求职进度
      5. 返回 career_response

    Args:
        state: 当前 LangGraph 状态

    Returns:
        包含 career_response, interview_score 的状态更新字典
    """
    user_id = state.get("user_id", "default_user")
    interview_mode = state.get("interview_mode", False)
    logger.info(
        f"CareerAgent | user={user_id} | "
        f"面试模式={'开启' if interview_mode else '关闭'}"
    )

    prompt = _build_career_prompt(state)

    try:
        llm = _get_llm()
        response = llm.invoke(prompt)
        career_response = response.content
        logger.info(f"CareerAgent | user={user_id} | 回复生成成功")
    except Exception as e:
        logger.error(f"CareerAgent | LLM 调用失败: {e}")
        career_response = (
            "抱歉，求职辅导服务暂时不可用。请检查 API 配置或稍后重试。\n"
            f"错误信息: {str(e)}"
        )

    interview_score = -1.0
    if interview_mode:
        interview_score = _extract_score(career_response)
        if interview_score >= 0:
            logger.info(f"CareerAgent | 面试评分: {interview_score}")

    profile = state.get("user_profile", {})
    if profile:
        job_progress = profile.get("job_progress", {})
        if interview_mode and interview_score >= 0:
            scores = job_progress.get("interview_scores", [])
            scores.append(interview_score)
            job_progress["interview_scores"] = scores[-10:]
        job_progress["last_career_advice"] = career_response[:200]
        profile["job_progress"] = job_progress
        vector_store.save_user_profile(user_id, profile)

    return {
        "career_response": career_response,
        "interview_score": interview_score,
    }
