# CompanionAI 🤖

基于 LangGraph 的多 Agent 协作智能聊天伙伴系统

## ✨ 功能特性

- **多 Agent 架构**: Router + 专业化 Agents 协作架构，通过 LangGraph 条件边动态路由
- **情感感知**: Transformers distilbert 模型 + 关键词回退双层策略，情绪驱动回复风格
- **向量记忆**: ChromaDB 持久化存储，跨会话个性化推荐
- **8 个自定义工具**:
  - 编程工具: execute_python_code, format_python_code, analyze_code_complexity, extract_code_features
  - 求职工具: evaluate_resume, get_interview_questions, recommend_companies, generate_learning_path
- **前后端分离**: FastAPI 后端 + Streamlit 前端
- **情绪关怀**: 基于历史情绪趋势的分级关怀机制

## 🛠️ 技术栈

Python | LangGraph | LangChain | ChromaDB | Transformers | FastAPI | Streamlit | Plotly | Uvicorn

## 📐 架构设计

```
用户消息 → GuardAgent (Router)
             ├─ 情感分析: Transformers + 关键词回退
             ├─ 消息分类: 代码检测 + 关键词评分
             └─ 动态路由
          MemoryAgent
             ├─ ChromaDB 检索 top-3 相关记忆
             ├─ 获取/更新用户画像
             └─ 存储当前对话 + 更新情绪趋势
       ┌──── 条件路由 ────┐
       ↓         ↓         ↓
  CodingAgent  CareerAgent  GeneralChat
       └────┬────────┬────┘
            ↓
       EmotionAgent (分级关怀)
            ↓
       Synthesizer (拼接 + 记忆同步)
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
OPENAI_API_KEY=你的API密钥
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
│   ├── agents/                 # 6 个专业化 Agent
│   │   ├── guard_agent.py      # Router 感知
│   │   ├── memory_agent.py     # 向量检索
│   │   ├── coding_agent.py     # 编程辅导
│   │   ├── career_agent.py     # 求职咨询
│   │   ├── emotion_agent.py    # 情绪关怀
│   │   └── synthesizer.py      # 合成输出
│   ├── tools/                  # 8 个自定义工具
│   │   ├── python_executor.py  # 代码执行
│   │   ├── code_analyzer.py    # 代码分析
│   │   └── career_tools.py     # 求职工具
│   ├── graph/                  # LangGraph 定义
│   │   ├── state.py            # 状态定义
│   │   └── workflow.py         # 工作流构建
│   ├── memory/                 # 向量存储
│   │   └── vector_store.py
│   ├── emotion/                # 情感分析
│   │   └── sentiment_analyzer.py
│   ├── backend/                # FastAPI 后端
│   │   └── main.py
│   ├── frontend/               # Streamlit 前端
│   │   └── streamlit_app.py
│   └── utils/                  # 工具函数
│       ├── config.py
│       ├── logger.py
│       └── helpers.py
├── .env.example
├── .gitignore
├── requirements.txt
├── start_all.py
└── README.md
```

## 🎯 核心技术亮点

### 1. Router + 专业化 Agents 架构

使用 LangGraph 构建有向图，`add_conditional_edges` 实现基于消息类别的动态路由，将单 Agent 的复杂提示词拆解为专业化分工。

### 2. 情感感知前置机制

在 GuardAgent 层完成情感分析，采用 Transformers distilbert 模型 + 中英文关键词回退双层策略，让所有后续 Agent 都能感知用户情绪并调整回复风格。

### 3. 向量记忆系统

使用 ChromaDB PersistentClient 持久化存储对话历史与用户画像，通过语义检索实现跨会话的个性化推荐。

### 4. 自定义工具集成

开发 8 个自定义工具，使用 `@tool` 装饰器定义，Agent 可在回复中主动推荐工具使用。

## 📝 许可证

MIT License
