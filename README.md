# CompanionAI 🤖

基于 LangGraph 的多 Agent 协作智能聊天伙伴系统

## ✨ 功能特性

- **多 Agent 架构**: Router + 5 个专业化 Agents 协作架构，通过 LangGraph 条件边动态路由
- **智能分类**: OpenAI Embedding API 向量相似度检测 + 代码特征识别 + 关键词回退三层策略，结合用户行为分析动态调整分类
- **向量记忆**: ChromaDB 持久化存储，实现语义检索、记忆权重衰减（时间衰减 + 频率加权）、主动记忆推送（根据情绪状态）、记忆压缩
- **MCP 协议集成**:
  - CodingAgent: 代码执行沙箱、LeetCode 题目获取、GitHub 代码搜索
  - CareerAgent: 真实岗位信息、ATS 简历评分、真实面经
- **情绪关怀**: 基于历史情绪趋势的分级关怀机制，连续低落触发深度关怀
- **前后端分离**: FastAPI 后端 + Streamlit 前端

## 🛠️ 技术栈

Python | LangGraph | LangChain | ChromaDB | OpenAI Embedding API | MCP 协议 | FastAPI | Streamlit | Plotly | Uvicorn

## 📐 架构设计

```
用户消息 → GuardAgent (Router)
             ├─ 情感分析: 规则引擎 + 关键词匹配
             ├─ 消息分类: 向量相似度 + 代码检测 + 关键词回退
             ├─ 行为分析: 输入频率、消息长度、时间段
             └─ 动态路由
          MemoryAgent
             ├─ ChromaDB 向量检索 top-3 相关记忆（权重衰减）
             ├─ 主动记忆推送（根据情绪状态）
             ├─ 获取/更新用户画像
             └─ 存储当前对话 + 更新情绪趋势 + 记忆压缩
       ┌──── 条件路由 ────┐
       ↓         ↓         ↓
  CodingAgent  CareerAgent  GeneralChat
       └────┬────────┬────┘
            ↓
    ResponseComposer
       ├─ 情绪关怀（基于历史趋势分级）
       ├─ 拼接专业回复
       └─ 存储记忆
            ↓
         最终回复
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并配置：

```env
# LLM 配置
OPENAI_API_KEY=你的API密钥
OPENAI_BASE_URL=你的API地址
OPENAI_MODEL=你的模型名称

# Embedding 配置（可选，用于向量分类）
EMBEDDING_API_KEY=你的Embedding API密钥
EMBEDDING_BASE_URL=你的Embedding API地址
EMBEDDING_MODEL=text-embedding-v3

# MCP 配置（可选，用于外部工具集成）
MCP_ENABLED=true
CODE_SANDBOX_API_URL=你的代码沙箱地址
LEETCODE_API_URL=你的LeetCode API地址
GITHUB_API_TOKEN=你的GitHub Token
```

### 3. 启动项目

```bash
# 一键启动（后端 + 前端）
python start_all.py

# 或单独启动
# 后端: uvicorn companion_ai.backend.main:app --reload
# 前端: streamlit run companion_ai/frontend/streamlit_app.py
```

## 🌐 访问地址

- **前端界面**: http://localhost:8501
- **后端 API**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs (Swagger)

## 📁 项目结构

```
CompanionAI/
├── companion_ai/
│   ├── agents/                 # 5 个专业化 Agent
│   │   ├── guard_agent.py      # Router: 情感分析 + 分类 + 行为分析
│   │   ├── memory_agent.py     # 向量检索 + 记忆管理
│   │   ├── coding_agent.py     # 编程辅导 + MCP 工具
│   │   ├── career_agent.py     # 求职辅导 + MCP 工具
│   │   ├── behavior_analyzer.py# 用户行为分析器
│   │   └── response_composer.py# 响应合成: 情绪关怀 + 回复拼接
│   ├── tools/                  # 自定义工具 + MCP 工具
│   │   ├── python_executor.py  # 代码执行
│   │   ├── career_tools.py     # 求职工具
│   │   ├── mcp_tools.py        # CodingAgent MCP 工具
│   │   └── career_mcp_tools.py # CareerAgent MCP 工具
│   ├── graph/                  # LangGraph 定义
│   │   ├── state.py            # 状态定义
│   │   └── workflow.py         # 工作流构建
│   ├── memory/                 # 向量存储
│   │   └── vector_store.py
│   ├── emotion/                # 情感分析
│   │   └── sentiment_analyzer.py
│   ├── data/                   # 分类种子数据
│   │   └── classification_seeds.json
│   ├── backend/                # FastAPI 后端
│   │   └── main.py
│   ├── frontend/               # Streamlit 前端
│   │   └── streamlit_app.py
│   └── utils/                  # 工具函数
│       ├── config.py
│       ├── logger.py
│       └── helpers.py
├── classification_seeds.md     # 分类种子说明文档
├── .env.example
├── .gitignore
├── requirements.txt
├── start_all.py
└── README.md
```

## 🎯 核心技术亮点

### 1. Router + 专业化 Agents 架构

使用 LangGraph 构建有向图，`add_conditional_edges` 实现基于消息类别的动态路由，将单 Agent 的复杂提示词拆解为 5 个专业化 Agent 分工协作，避免提示词膨胀。

### 2. 多模态消息分类

GuardAgent 采用三层分类策略：OpenAI Embedding API 向量相似度检测（主方案）+ 代码特征识别 + 关键词回退（兜底），结合用户行为分析（输入频率、消息长度、时间段）动态调整分类结果，解决传统关键词匹配无法理解语义和同义词的痛点。

### 3. 智能记忆系统

使用 ChromaDB PersistentClient 持久化存储对话历史与用户画像，实现：
- **语义检索**: 基于向量相似度，支持用户隔离
- **记忆权重衰减**: 时间衰减 + 使用频率加权，模拟人类遗忘曲线
- **主动记忆推送**: 根据用户当前情绪状态推送相关记忆
- **记忆压缩**: 定期摘要总结，控制存储增长

### 4. MCP 协议集成

通过 Model Context Protocol 接入外部应用，增强 Agent 能力：
- **CodingAgent**: 代码执行沙箱（安全验证）、LeetCode 题目获取（真实题库）、GitHub 代码搜索（项目参考）
- **CareerAgent**: 真实岗位信息、ATS 简历评分、真实面经
- **智能回退**: MCP 不可用时自动切换本地工具，确保系统鲁棒性

### 5. 情绪关怀机制

ResponseComposer 基于历史情绪趋势的分级关怀：
- 连续低落超过 3 次 → 深度关怀
- 当前负面情绪 → 分级鼓励（根据情绪强度）
- 积极情绪 → 祝贺保持
- 结合消息类别调整关怀内容，打造有"温度"的 AI 交互体验

## 📝 许可证

MIT License
