import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    项目全局配置，通过 .env 文件或环境变量加载。
    使用 pydantic-settings 实现类型安全的配置管理。
    """

    DEEPSEEK_API_KEY: Optional[str] = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    OPENAI_API_KEY: Optional[str] = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o-mini"

    ALIYUN_API_KEY: Optional[str] = ""
    ALIYUN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ALIYUN_MODEL: str = "qwen-plus"

    LLM_PROVIDER: str = "aliyun"

    CHROMA_PERSIST_DIR: str = "./chroma_data"
    CHROMA_COLLECTION_CONVERSATION: str = "conversations"
    CHROMA_COLLECTION_PROFILE: str = "user_profiles"
    CHROMA_COLLECTION_CLASSIFICATION: str = "classification_seeds"

    EMBEDDING_API_KEY: Optional[str] = ""
    EMBEDDING_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    EMBEDDING_MODEL: str = "text-embedding-v3"

    MCP_SANDBOX_URL: str = "http://localhost:3000"
    MCP_LEETCODE_URL: str = "http://localhost:3001"
    MCP_GITHUB_URL: str = "http://localhost:3002"
    MCP_ENABLED: bool = False

    CAREER_MCP_JOB_URL: str = "http://localhost:3003"
    CAREER_MCP_RESUME_URL: str = "http://localhost:3004"
    CAREER_MCP_INTERVIEW_URL: str = "http://localhost:3005"
    CAREER_MCP_ENABLED: bool = False

    SENTIMENT_MODEL_NAME: str = "distilbert-base-uncased-finetuned-sst-2-english"
    # 默认使用关键词方案，快速启动
    # 需要模型时设为 True 即可
    SENTIMENT_FALLBACK_ENABLED: bool = False

    HF_ENDPOINT: str = "https://hf-mirror.com"

    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "./logs"

    DEFAULT_USER_ID: str = "default_user"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def llm_api_key(self) -> str:
        if self.LLM_PROVIDER == "deepseek":
            return os.getenv("DEEPSEEK_API_KEY", "") or self.DEEPSEEK_API_KEY or ""
        elif self.LLM_PROVIDER == "aliyun":
            return os.getenv("OPENAI_API_KEY", "") or os.getenv("ALIYUN_API_KEY", "") or self.ALIYUN_API_KEY or ""
        return os.getenv("OPENAI_API_KEY", "") or self.OPENAI_API_KEY or ""

    @property
    def llm_base_url(self) -> str:
        if self.LLM_PROVIDER == "deepseek":
            return self.DEEPSEEK_BASE_URL
        elif self.LLM_PROVIDER == "aliyun":
            return self.ALIYUN_BASE_URL
        return self.OPENAI_BASE_URL

    @property
    def llm_model(self) -> str:
        if self.LLM_PROVIDER == "deepseek":
            return self.DEEPSEEK_MODEL
        elif self.LLM_PROVIDER == "aliyun":
            return self.ALIYUN_MODEL
        return self.OPENAI_MODEL

    @property
    def embedding_api_key(self) -> str:
        return os.getenv("EMBEDDING_API_KEY", "") or self.EMBEDDING_API_KEY or self.llm_api_key

    @property
    def embedding_base_url(self) -> str:
        return self.EMBEDDING_BASE_URL

    @property
    def embedding_model(self) -> str:
        return self.EMBEDDING_MODEL


settings = Settings()