"""
MCP 工具管理器 —— 外部应用集成

职责：
  1. 管理所有 MCP 服务器连接
  2. 提供统一的工具调用接口
  3. 支持 sandbox、leetcode、github 三类 MCP 服务

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


class MCPClient:
    """
    MCP 客户端，封装与 MCP 服务器的通信。
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
        if not settings.MCP_ENABLED:
            return {
                "success": False,
                "error": "MCP 功能未启用，请在 .env 中设置 MCP_ENABLED=true",
            }

        if not self._available:
            self._check_availability()
            if not self._available:
                return {
                    "success": False,
                    "error": f"MCP 服务器不可用: {self.server_name} @ {self.server_url}",
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
                    f"MCP 调用成功: {self.server_name}.{tool_name} "
                    f"(params={params})"
                )
                return {"success": True, "data": result}
        except httpx.TimeoutException:
            logger.error(f"MCP 调用超时: {self.server_name}.{tool_name}")
            return {
                "success": False,
                "error": f"MCP 调用超时 ({timeout}秒)",
            }
        except httpx.HTTPError as e:
            logger.error(f"MCP 调用失败: {self.server_name}.{tool_name} - {e}")
            return {
                "success": False,
                "error": f"MCP 调用失败: {str(e)}",
            }
        except Exception as e:
            logger.error(f"MCP 调用异常: {self.server_name}.{tool_name} - {e}")
            return {
                "success": False,
                "error": f"MCP 调用异常: {str(e)}",
            }

    def _check_availability(self) -> bool:
        """检查 MCP 服务器是否可用"""
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(f"{self.server_url}/health")
                if response.status_code == 200:
                    self._available = True
                    logger.info(f"MCP 服务器可用: {self.server_name}")
                    return True
        except Exception:
            logger.warning(f"MCP 服务器不可用: {self.server_name} @ {self.server_url}")
        self._available = False
        return False

    @property
    def is_available(self) -> bool:
        return self._available


class MCPToolManager:
    """
    MCP 工具管理器，管理所有外部 MCP 服务器。
    """

    def __init__(self):
        self.servers: Dict[str, MCPClient] = {}
        self._init_servers()

    def _init_servers(self):
        """初始化所有 MCP 服务器连接"""
        self.servers["sandbox"] = MCPClient(
            settings.MCP_SANDBOX_URL, "sandbox"
        )
        self.servers["leetcode"] = MCPClient(
            settings.MCP_LEETCODE_URL, "leetcode"
        )
        self.servers["github"] = MCPClient(
            settings.MCP_GITHUB_URL, "github"
        )
        logger.info(
            f"MCP 工具管理器初始化完成，"
            f"启用状态: {settings.MCP_ENABLED}"
        )

    def get_client(self, server_name: str) -> Optional[MCPClient]:
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
            server_name: 服务器名称（sandbox/leetcode/github）
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
                "error": f"未注册的 MCP 服务器: {server_name}",
            }
        return client.call_tool(tool_name, params, timeout)


# 全局单例
mcp_tool_manager = MCPToolManager()


@tool
def execute_code_sandbox(code: str, language: str = "python") -> Dict[str, Any]:
    """
    在安全沙箱中执行代码。

    通过 MCP 接入的 sandbox 服务执行代码，支持多种语言。
    比本地执行更安全，支持资源限制和隔离。

    Args:
        code: 要执行的代码
        language: 编程语言（python/javascript/java 等）

    Returns:
        包含执行结果或错误信息的字典
    """
    result = mcp_tool_manager.call_tool(
        "sandbox",
        "execute_code",
        {"code": code, "language": language},
        timeout=30,
    )

    if result.get("success"):
        data = result.get("data", {})
        return {
            "success": True,
            "stdout": data.get("stdout", ""),
            "stderr": data.get("stderr", ""),
            "exit_code": data.get("exit_code", 0),
            "execution_time": data.get("execution_time", 0),
            "source": "mcp_sandbox",
        }
    else:
        logger.warning(f"沙箱执行失败，回退到本地执行: {result.get('error')}")
        from companion_ai.tools.python_executor import execute_python_code
        return execute_python_code.invoke({"code": code})


@tool
def get_leetcode_problem(problem_id: Optional[int] = None, difficulty: Optional[str] = None, tag: Optional[str] = None) -> Dict[str, Any]:
    """
    获取 LeetCode 题目信息。

    通过 MCP 接入的 leetcode 服务获取真实题目、描述和测试用例。

    Args:
        problem_id: 题目 ID（可选）
        difficulty: 难度（easy/medium/hard）
        tag: 标签（如 dynamic-programming, array, tree 等）

    Returns:
        包含题目信息的字典
    """
    params = {}
    if problem_id is not None:
        params["problem_id"] = problem_id
    if difficulty:
        params["difficulty"] = difficulty
    if tag:
        params["tag"] = tag

    result = mcp_tool_manager.call_tool(
        "leetcode",
        "get_problem",
        params,
        timeout=15,
    )

    if result.get("success"):
        data = result.get("data", {})
        return {
            "success": True,
            "problem_id": data.get("problem_id"),
            "title": data.get("title"),
            "title_cn": data.get("title_cn"),
            "difficulty": data.get("difficulty"),
            "description": data.get("description"),
            "examples": data.get("examples", []),
            "tags": data.get("tags", []),
            "source": "mcp_leetcode",
        }
    else:
        logger.warning(f"LeetCode 获取失败: {result.get('error')}")
        return {
            "success": False,
            "error": result.get("error", "获取题目失败"),
            "source": "mcp_leetcode",
        }


@tool
def search_leetcode_problems(
    difficulty: Optional[str] = None,
    tag: Optional[str] = None,
    limit: int = 5,
) -> Dict[str, Any]:
    """
    搜索 LeetCode 题目列表。

    根据难度和标签搜索题目，用于推荐练习。

    Args:
        difficulty: 难度（easy/medium/hard）
        tag: 标签（如 dynamic-programming, array, tree 等）
        limit: 返回数量限制

    Returns:
        包含题目列表的字典
    """
    params = {"limit": limit}
    if difficulty:
        params["difficulty"] = difficulty
    if tag:
        params["tag"] = tag

    result = mcp_tool_manager.call_tool(
        "leetcode",
        "search_problems",
        params,
        timeout=15,
    )

    if result.get("success"):
        data = result.get("data", {})
        return {
            "success": True,
            "problems": data.get("problems", []),
            "total": data.get("total", 0),
            "source": "mcp_leetcode",
        }
    else:
        logger.warning(f"LeetCode 搜索失败: {result.get('error')}")
        return {
            "success": False,
            "error": result.get("error", "搜索题目失败"),
            "source": "mcp_leetcode",
        }


@tool
def search_github_repositories(query: str, sort: str = "stars", limit: int = 5) -> Dict[str, Any]:
    """
    搜索 GitHub 代码仓库。

    通过 MCP 接入的 GitHub 服务搜索开源项目，用于学习参考。

    Args:
        query: 搜索关键词
        sort: 排序方式（stars/forks/updated）
        limit: 返回数量限制

    Returns:
        包含仓库列表的字典
    """
    result = mcp_tool_manager.call_tool(
        "github",
        "search_repositories",
        {"query": query, "sort": sort, "limit": limit},
        timeout=15,
    )

    if result.get("success"):
        data = result.get("data", {})
        return {
            "success": True,
            "repositories": data.get("repositories", []),
            "total": data.get("total", 0),
            "source": "mcp_github",
        }
    else:
        logger.warning(f"GitHub 搜索失败: {result.get('error')}")
        return {
            "success": False,
            "error": result.get("error", "搜索仓库失败"),
            "source": "mcp_github",
        }


@tool
def get_github_repository_info(owner: str, repo: str) -> Dict[str, Any]:
    """
    获取 GitHub 仓库详细信息。

    获取 README、项目结构、最新提交等信息。

    Args:
        owner: 仓库所有者
        repo: 仓库名称

    Returns:
        包含仓库详细信息的字典
    """
    result = mcp_tool_manager.call_tool(
        "github",
        "get_repository_info",
        {"owner": owner, "repo": repo},
        timeout=15,
    )

    if result.get("success"):
        data = result.get("data", {})
        return {
            "success": True,
            "name": data.get("name"),
            "description": data.get("description"),
            "stars": data.get("stars"),
            "language": data.get("language"),
            "readme": data.get("readme"),
            "topics": data.get("topics", []),
            "source": "mcp_github",
        }
    else:
        logger.warning(f"GitHub 仓库信息获取失败: {result.get('error')}")
        return {
            "success": False,
            "error": result.get("error", "获取仓库信息失败"),
            "source": "mcp_github",
        }


@tool
def search_github_code(query: str, language: str = "python", limit: int = 5) -> Dict[str, Any]:
    """
    搜索 GitHub 代码片段。

    搜索真实项目中的代码用法，用于学习参考。

    Args:
        query: 搜索关键词
        language: 编程语言过滤
        limit: 返回数量限制

    Returns:
        包含代码片段的字典
    """
    result = mcp_tool_manager.call_tool(
        "github",
        "search_code",
        {"query": query, "language": language, "limit": limit},
        timeout=15,
    )

    if result.get("success"):
        data = result.get("data", {})
        return {
            "success": True,
            "code_snippets": data.get("code_snippets", []),
            "total": data.get("total", 0),
            "source": "mcp_github",
        }
    else:
        logger.warning(f"GitHub 代码搜索失败: {result.get('error')}")
        return {
            "success": False,
            "error": result.get("error", "搜索代码失败"),
            "source": "mcp_github",
        }
