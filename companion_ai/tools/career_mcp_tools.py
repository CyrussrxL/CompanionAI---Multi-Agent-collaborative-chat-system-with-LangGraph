"""
CareerAgent MCP 工具管理器 —— 求职辅导外部应用集成

职责：
  1. 管理求职相关 MCP 服务器连接
  2. 提供统一的工具调用接口
  3. 支持 JD 招聘、简历优化、面试经验三类 MCP 服务

设计理由：
  - MCP（Model Context Protocol）是开放协议，允许 AI 与外部工具标准化交互
  - 通过统一管理器解耦具体实现，便于扩展新工具
  - 支持 MCP 禁用时的本地回退，确保系统鲁棒性
"""

import httpx
from typing import Dict, Any, Optional, List
from langchain_core.tools import tool

from companion_ai.utils.config import settings
from companion_ai.utils.logger import logger


class CareerMCPClient:
    """
    Career MCP 客户端，封装与 MCP 服务器的通信。
    """

    def __init__(self, server_url: str, server_name: str):
        self.server_url = server_url.rstrip("/")
        self.server_name = server_name
        self._available = False

    def call_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """
        调用 MCP 工具。

        Args:
            tool_name: 工具名称
            params: 工具参数
            timeout: 超时时间（秒）

        Returns:
            工具执行结果
        """
        if not settings.CAREER_MCP_ENABLED:
            return {
                "success": False,
                "error": "Career MCP 功能未启用，请在 .env 中设置 CAREER_MCP_ENABLED=true",
            }

        if not self._available:
            self._check_availability()
            if not self._available:
                return {
                    "success": False,
                    "error": f"Career MCP 服务器不可用: {self.server_name} @ {self.server_url}",
                }

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    f"{self.server_url}/tools/{tool_name}",
                    json={"params": params},
                )
                response.raise_for_status()
                result = response.json()
                logger.info(
                    f"Career MCP 调用成功: {self.server_name}.{tool_name} "
                    f"(params={params})"
                )
                return {"success": True, "data": result}
        except httpx.TimeoutException:
            logger.error(f"Career MCP 调用超时: {self.server_name}.{tool_name}")
            return {
                "success": False,
                "error": f"Career MCP 调用超时 ({timeout}秒)",
            }
        except httpx.HTTPError as e:
            logger.error(f"Career MCP 调用失败: {self.server_name}.{tool_name} - {e}")
            return {
                "success": False,
                "error": f"Career MCP 调用失败: {str(e)}",
            }
        except Exception as e:
            logger.error(f"Career MCP 调用异常: {self.server_name}.{tool_name} - {e}")
            return {
                "success": False,
                "error": f"Career MCP 调用异常: {str(e)}",
            }

    def _check_availability(self) -> bool:
        """检查 MCP 服务器是否可用"""
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(f"{self.server_url}/health")
                if response.status_code == 200:
                    self._available = True
                    logger.info(f"Career MCP 服务器可用: {self.server_name}")
                    return True
        except Exception:
            logger.warning(f"Career MCP 服务器不可用: {self.server_name} @ {self.server_url}")
        self._available = False
        return False

    @property
    def is_available(self) -> bool:
        return self._available


class CareerMCPToolManager:
    """
    Career MCP 工具管理器，管理所有求职相关 MCP 服务器。
    """

    def __init__(self):
        self.servers: Dict[str, CareerMCPClient] = {}
        self._init_servers()

    def _init_servers(self):
        """初始化所有 MCP 服务器连接"""
        self.servers["job"] = CareerMCPClient(
            settings.CAREER_MCP_JOB_URL, "job"
        )
        self.servers["resume"] = CareerMCPClient(
            settings.CAREER_MCP_RESUME_URL, "resume"
        )
        self.servers["interview"] = CareerMCPClient(
            settings.CAREER_MCP_INTERVIEW_URL, "interview"
        )
        logger.info(
            f"Career MCP 工具管理器初始化完成，"
            f"启用状态: {settings.CAREER_MCP_ENABLED}"
        )

    def get_client(self, server_name: str) -> Optional[CareerMCPClient]:
        """获取指定服务器的客户端"""
        return self.servers.get(server_name)

    def call_tool(
        self,
        server_name: str,
        tool_name: str,
        params: Dict[str, Any],
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """
        调用指定服务器的工具。

        Args:
            server_name: 服务器名称（job/resume/interview）
            tool_name: 工具名称
            params: 工具参数
            timeout: 超时时间

        Returns:
            工具执行结果
        """
        client = self.get_client(server_name)
        if not client:
            return {
                "success": False,
                "error": f"未注册的 Career MCP 服务器: {server_name}",
            }
        return client.call_tool(tool_name, params, timeout)


# 全局单例
career_mcp_tool_manager = CareerMCPToolManager()


@tool
def search_job_listings(
    position: str = "AI开发",
    city: str = "北京",
    experience: str = "校招",
    limit: int = 5,
) -> Dict[str, Any]:
    """
    搜索真实招聘岗位信息。

    通过 MCP 接入的招聘服务获取最新岗位信息，包括薪资、技能要求、面试流程等。

    Args:
        position: 岗位名称（如 AI开发、算法工程师、后端开发）
        city: 工作城市
        experience: 经验要求（校招/1-3年/3-5年）
        limit: 返回数量限制

    Returns:
        包含岗位列表的字典
    """
    result = career_mcp_tool_manager.call_tool(
        "job",
        "search_listings",
        {
            "position": position,
            "city": city,
            "experience": experience,
            "limit": limit,
        },
        timeout=15,
    )

    if result.get("success"):
        data = result.get("data", {})
        return {
            "success": True,
            "jobs": data.get("jobs", []),
            "total": data.get("total", 0),
            "source": "mcp_job",
        }
    else:
        logger.warning(f"招聘搜索失败: {result.get('error')}")
        return {
            "success": False,
            "error": result.get("error", "搜索岗位失败"),
            "source": "mcp_job",
        }


@tool
def analyze_job_requirements(job_description: str) -> Dict[str, Any]:
    """
    分析岗位 JD 的技能要求。

    提取 JD 中的关键技能、经验要求、加分项，帮助用户针对性准备。

    Args:
        job_description: 岗位描述文本

    Returns:
        包含技能要求分析的字典
    """
    result = career_mcp_tool_manager.call_tool(
        "job",
        "analyze_requirements",
        {"job_description": job_description},
        timeout=15,
    )

    if result.get("success"):
        data = result.get("data", {})
        return {
            "success": True,
            "required_skills": data.get("required_skills", []),
            "preferred_skills": data.get("preferred_skills", []),
            "experience_requirements": data.get("experience_requirements", []),
            "interview_process": data.get("interview_process", []),
            "source": "mcp_job",
        }
    else:
        logger.warning(f"JD 分析失败: {result.get('error')}")
        return {
            "success": False,
            "error": result.get("error", "分析岗位要求失败"),
            "source": "mcp_job",
        }


@tool
def optimize_resume(
    resume_text: str,
    target_position: str = "AI开发",
    target_company: Optional[str] = None,
) -> Dict[str, Any]:
    """
    简历优化服务。

    通过 MCP 接入的简历优化服务，提供 ATS 兼容性评分、关键词优化、格式建议等。

    Args:
        resume_text: 简历文本
        target_position: 目标岗位
        target_company: 目标公司（可选）

    Returns:
        包含简历优化建议的字典
    """
    params = {
        "resume_text": resume_text,
        "target_position": target_position,
    }
    if target_company:
        params["target_company"] = target_company

    result = career_mcp_tool_manager.call_tool(
        "resume",
        "optimize",
        params,
        timeout=20,
    )

    if result.get("success"):
        data = result.get("data", {})
        return {
            "success": True,
            "ats_score": data.get("ats_score", 0),
            "strengths": data.get("strengths", []),
            "improvements": data.get("improvements", []),
            "keyword_suggestions": data.get("keyword_suggestions", []),
            "format_suggestions": data.get("format_suggestions", []),
            "source": "mcp_resume",
        }
    else:
        logger.warning(f"简历优化失败: {result.get('error')}")
        from companion_ai.tools.career_tools import evaluate_resume
        return evaluate_resume.invoke({"resume_text": resume_text})


@tool
def get_interview_experience(
    company: str = "字节跳动",
    position: str = "AI开发",
    interview_type: str = "技术面",
) -> Dict[str, Any]:
    """
    获取真实面试经验分享。

    通过 MCP 接入的面经服务获取真实面试题目、流程和评价标准。

    Args:
        company: 公司名称
        position: 岗位名称
        interview_type: 面试类型（技术面/HR面/主管面）

    Returns:
        包含面试经验的字典
    """
    result = career_mcp_tool_manager.call_tool(
        "interview",
        "get_experience",
        {
            "company": company,
            "position": position,
            "interview_type": interview_type,
        },
        timeout=15,
    )

    if result.get("success"):
        data = result.get("data", {})
        return {
            "success": True,
            "company": data.get("company"),
            "position": data.get("position"),
            "interview_type": data.get("interview_type"),
            "questions": data.get("questions", []),
            "process": data.get("process", []),
            "tips": data.get("tips", []),
            "difficulty": data.get("difficulty", "medium"),
            "source": "mcp_interview",
        }
    else:
        logger.warning(f"面试经验获取失败: {result.get('error')}")
        return {
            "success": False,
            "error": result.get("error", "获取面试经验失败"),
            "source": "mcp_interview",
        }


@tool
def search_interview_questions(
    topic: str = "AI",
    difficulty: str = "medium",
    limit: int = 5,
) -> Dict[str, Any]:
    """
    搜索面试题目。

    根据主题和难度搜索真实面试题目，包含参考答案。

    Args:
        topic: 面试主题（AI/算法/系统设计/行为面试）
        difficulty: 难度（easy/medium/hard）
        limit: 返回数量限制

    Returns:
        包含面试题目的字典
    """
    result = career_mcp_tool_manager.call_tool(
        "interview",
        "search_questions",
        {
            "topic": topic,
            "difficulty": difficulty,
            "limit": limit,
        },
        timeout=15,
    )

    if result.get("success"):
        data = result.get("data", {})
        return {
            "success": True,
            "questions": data.get("questions", []),
            "total": data.get("total", 0),
            "source": "mcp_interview",
        }
    else:
        logger.warning(f"面试题目搜索失败: {result.get('error')}")
        from companion_ai.tools.career_tools import get_interview_questions
        return get_interview_questions.invoke({"topic": topic, "count": limit})


@tool
def get_salary_info(
    position: str = "AI开发",
    city: str = "北京",
    experience: str = "校招",
) -> Dict[str, Any]:
    """
    获取岗位薪资信息。

    获取目标岗位在指定城市和经验水平的薪资范围。

    Args:
        position: 岗位名称
        city: 城市
        experience: 经验水平

    Returns:
        包含薪资信息的字典
    """
    result = career_mcp_tool_manager.call_tool(
        "job",
        "get_salary",
        {
            "position": position,
            "city": city,
            "experience": experience,
        },
        timeout=15,
    )

    if result.get("success"):
        data = result.get("data", {})
        return {
            "success": True,
            "position": data.get("position"),
            "city": data.get("city"),
            "salary_range": data.get("salary_range", {}),
            "median": data.get("median", 0),
            "percentile_25": data.get("percentile_25", 0),
            "percentile_75": data.get("percentile_75", 0),
            "source": "mcp_job",
        }
    else:
        logger.warning(f"薪资查询失败: {result.get('error')}")
        return {
            "success": False,
            "error": result.get("error", "查询薪资失败"),
            "source": "mcp_job",
        }
