"""
Streamlit 前端界面 —— streamlit_app.py

职责：
  提供 CompanionAI 的可视化交互界面，包括：
    - 左侧边栏：用户画像编辑、情绪仪表盘、记忆可视化、对话存储管理
    - 右侧主区域：聊天历史（气泡风格）、消息输入、日报生成按钮

设计理由：
  - 使用 Streamlit 快速构建交互式 Web 界面，适合原型展示。
  - 情绪仪表盘使用 plotly 绘制折线图，直观展示情绪趋势。
  - 用户画像可编辑，让用户主动参与个性化设置。
  - 聊天记录使用 session_state 持久化，页面刷新不丢失。
  - 支持对话历史的保存和加载，方便用户管理不同对话。
  - 实现流式输出，提升用户体验。
"""

import json
import os
import sys
import time
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from companion_ai.graph.workflow import run_workflow, run_daily_report
from companion_ai.memory.vector_store import vector_store
from companion_ai.utils.config import settings
from companion_ai.utils.logger import logger


def get_conversations_dir():
    """获取对话存储目录。"""
    return os.path.join(os.path.dirname(__file__), "..", "..", "conversations")


def load_saved_conversations():
    """从本地文件加载已保存的对话列表。"""
    save_dir = get_conversations_dir()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    conversations = []
    for filename in os.listdir(save_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(save_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    conversations.append({
                        "filename": filename,
                        "name": data.get("name", filename),
                        "timestamp": data.get("timestamp", ""),
                        "message_count": len(data.get("chat_history", [])),
                        "user_id": data.get("user_id", ""),
                    })
            except Exception as e:
                logger.error(f"加载对话失败: {e}")
    
    conversations.sort(key=lambda x: x["timestamp"], reverse=True)
    st.session_state.saved_conversations = conversations


def save_conversation_direct(user_id=None):
    """保存当前对话到本地文件（直接保存，不需要用户输入）。"""
    if not st.session_state.chat_history:
        return False, None
    
    save_dir = get_conversations_dir()
    os.makedirs(save_dir, exist_ok=True)
    
    current_user_id = user_id or st.session_state.user_id
    
    if st.session_state.current_conversation_name:
        name = st.session_state.current_conversation_name
        timestamp = datetime.now().isoformat()
    else:
        timestamp = datetime.now().isoformat()
        name = f"{current_user_id}_{timestamp[:19].replace(':', '-')}"
        st.session_state.current_conversation_name = name
    
    filename = f"{name}.json"
    filepath = os.path.join(save_dir, filename)
    
    data = {
        "name": name,
        "timestamp": timestamp,
        "user_id": current_user_id,
        "chat_history": st.session_state.chat_history,
    }
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        load_saved_conversations()
        return True, name
    except Exception as e:
        logger.error(f"保存失败: {e}")
        return False, None


def load_conversation(filename):
    """从本地文件加载对话。"""
    save_dir = get_conversations_dir()
    filepath = os.path.join(save_dir, filename)
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            st.session_state.chat_history = data.get("chat_history", [])
            st.session_state.user_id = data.get("user_id", st.session_state.user_id)
            st.session_state.thread_id = f"thread_{st.session_state.user_id}"
            st.success(f"对话已加载: {data.get('name', filename)}")
            st.rerun()
    except Exception as e:
        st.error(f"加载失败: {e}")


def delete_conversation(filename):
    """删除已保存的对话。"""
    save_dir = get_conversations_dir()
    filepath = os.path.join(save_dir, filename)
    
    try:
        os.remove(filepath)
        load_saved_conversations()
        st.success("对话已删除")
        st.rerun()
    except Exception as e:
        st.error(f"删除失败: {e}")


def stream_output(text, placeholder):
    """模拟流式输出显示文本。"""
    current_text = ""
    for char in text:
        current_text += char
        placeholder.markdown(current_text)
        time.sleep(0.01)
    return current_text


def init_session_state():
    """初始化 Streamlit session state。"""
    if "user_id" not in st.session_state:
        st.session_state.user_id = settings.DEFAULT_USER_ID
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "interview_mode" not in st.session_state:
        st.session_state.interview_mode = False
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"thread_{st.session_state.user_id}"
    if "saved_conversations" not in st.session_state:
        st.session_state.saved_conversations = []
        load_saved_conversations()
    if "streaming_enabled" not in st.session_state:
        st.session_state.streaming_enabled = True
    if "current_conversation_name" not in st.session_state:
        st.session_state.current_conversation_name = None


def create_new_conversation():
    """创建新对话。"""
    if st.session_state.chat_history and st.session_state.current_conversation_name:
        save_conversation_direct()
    st.session_state.chat_history = []
    timestamp = datetime.now().isoformat()
    name = f"{st.session_state.user_id}_{timestamp[:19].replace(':', '-')}"
    st.session_state.current_conversation_name = name
    st.rerun()


def render_sidebar():
    """渲染左侧边栏：用户画像、情绪仪表盘、记忆可视化、对话存储。"""
    with st.sidebar:
        st.title("⚙️ 设置面板")

        st.session_state.user_id = st.text_input(
            "用户 ID",
            value=st.session_state.user_id,
            help="输入你的用户标识，用于记忆和画像管理",
        )

        if st.button("➕ 新建对话"):
            create_new_conversation()

        st.divider()
        st.subheader("👤 用户画像")

        profile = vector_store.get_user_profile(st.session_state.user_id)

        learning_goal = st.text_input(
            "学习目标",
            value=profile.get("learning_goal", "找算法实习"),
        )
        skill_level = st.text_input(
            "当前技能水平",
            value=profile.get("current_skill_level", "Python基础"),
        )
        job_target = st.text_input(
            "求职目标",
            value=profile.get("job_target", "AI开发"),
        )

        if st.button("💾 保存画像"):
            profile["learning_goal"] = learning_goal
            profile["current_skill_level"] = skill_level
            profile["job_target"] = job_target
            vector_store.save_user_profile(st.session_state.user_id, profile)
            st.success("画像已保存！")

        st.divider()
        st.subheader("🎭 情绪仪表盘")

        emotional_trend = profile.get("emotional_trend", [])
        if emotional_trend:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(emotional_trend) + 1)),
                    y=emotional_trend,
                    mode="lines+markers",
                    name="情绪得分",
                    line=dict(color="#6366f1", width=2),
                    marker=dict(size=8),
                )
            )
            fig.update_layout(
                title="最近情绪趋势",
                xaxis_title="对话序号",
                yaxis_title="情绪得分",
                yaxis=dict(range=[0, 1]),
                height=250,
                margin=dict(l=20, r=20, t=40, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

            avg_score = sum(emotional_trend) / len(emotional_trend)
            if avg_score < 0.4:
                st.warning(f"近期平均情绪: {avg_score:.2f} — 看起来你需要休息一下 💙")
            elif avg_score > 0.7:
                st.success(f"近期平均情绪: {avg_score:.2f} — 状态不错！🌟")
            else:
                st.info(f"近期平均情绪: {avg_score:.2f}")
        else:
            st.info("暂无情绪数据，开始对话后将自动记录。")

        st.divider()
        st.subheader("🧠 记忆可视化")

        if st.session_state.chat_history:
            last_user_msg = ""
            for msg in reversed(st.session_state.chat_history):
                if msg["role"] == "user":
                    last_user_msg = msg["content"]
                    break

            if last_user_msg:
                memories = vector_store.retrieve_memories(
                    st.session_state.user_id, last_user_msg, top_k=3
                )
                if memories:
                    for i, mem in enumerate(memories, 1):
                        st.markdown(
                            f"**记忆 {i}** ({mem.get('timestamp', '')})\n\n"
                            f"情绪: {mem.get('emotion', '')} | "
                            f"类别: {mem.get('category', '')}\n\n"
                            f"> {mem.get('text', '')[:100]}..."
                        )
                else:
                    st.info("暂无相关记忆")
        else:
            st.info("开始对话后将显示相关记忆")

        st.divider()
        st.subheader("💾 对话存储")

        st.info(f"当前对话: {st.session_state.current_conversation_name or '未命名'}")

        st.divider()
        st.subheader("📚 已保存对话")

        if st.session_state.saved_conversations:
            for conv in st.session_state.saved_conversations:
                with st.expander(f"📋 {conv['name']} ({conv['message_count']} 条消息)"):
                    st.caption(f"保存时间: {conv['timestamp'][:19]}")
                    st.caption(f"用户 ID: {conv.get('user_id', '未知')}")
                    col_load, col_del = st.columns(2)
                    with col_load:
                        if st.button("📥 加载", key=f"load_{conv['filename']}"):
                            load_conversation(conv['filename'])
                    with col_del:
                        if st.button("🗑️ 删除", key=f"del_{conv['filename']}"):
                            delete_conversation(conv['filename'])
        else:
            st.info("暂无已保存的对话")


def render_chat_area():
    """渲染右侧主区域：聊天历史和输入框。"""
    st.title("🤖 CompanionAI")
    st.caption("你的多 Agent 智能聊天伙伴 — 编程辅导 · 求职建议 · 情绪关怀")

    col1, col2 = st.columns([3, 1])
    with col2:
        interview_toggle = st.toggle(
            "🎯 模拟面试",
            value=st.session_state.interview_mode,
            help="开启后 CareerAgent 将进入模拟面试模式",
        )
        st.session_state.interview_mode = interview_toggle

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("emotion"):
                emotion_emoji = {
                    "positive": "😊",
                    "negative": "😢",
                    "neutral": "😐",
                }.get(msg["emotion"], "")
                st.caption(f"{emotion_emoji} 情绪: {msg['emotion']}")

    if prompt := st.chat_input("输入消息..."):
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt, "emotion": None}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        if not st.session_state.current_conversation_name:
            timestamp = datetime.now().isoformat()
            name = f"{st.session_state.user_id}_{timestamp[:19].replace(':', '-')}"
            st.session_state.current_conversation_name = name

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                with st.spinner("思考中..."):
                    result = run_workflow(
                        user_id=st.session_state.user_id,
                        message=prompt,
                        thread_id=st.session_state.thread_id,
                        interview_mode=st.session_state.interview_mode,
                    )
                    response = result.get("final_response", "抱歉，无法生成回复。")
                    emotion = result.get("emotion_label", "neutral")
                    category = result.get("message_category", "")

                if st.session_state.streaming_enabled:
                    full_response = stream_output(response, message_placeholder)
                else:
                    message_placeholder.markdown(response)
                    full_response = response

                emotion_emoji = {
                    "positive": "😊",
                    "negative": "😢",
                    "neutral": "😐",
                }.get(emotion, "")
                st.caption(
                    f"{emotion_emoji} 情绪: {emotion} | "
                    f"类别: {category}"
                )

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                        "emotion": emotion,
                        "category": category,
                    }
                )

                save_conversation_direct()

            except Exception as e:
                error_msg = f"❌ 处理失败: {str(e)}"
                message_placeholder.error(error_msg)
                logger.error(f"工作流执行失败: {e}")

    col_report, col_clear = st.columns(2)
    with col_report:
        if st.button("📊 生成日报"):
            with st.spinner("正在生成日报..."):
                try:
                    report = run_daily_report(st.session_state.user_id)
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": f"## 📊 今日日报\n\n{report}",
                            "emotion": None,
                        }
                    )
                    save_conversation_direct()
                    st.rerun()
                except Exception as e:
                    st.error(f"日报生成失败: {str(e)}")

    with col_clear:
        if st.button("🗑️ 清空对话"):
            st.session_state.chat_history = []
            st.rerun()


def main():
    """Streamlit 应用入口。"""
    st.set_page_config(
        page_title="CompanionAI",
        page_icon="🤖",
        layout="wide",
    )

    init_session_state()
    render_sidebar()
    render_chat_area()


if __name__ == "__main__":
    main()
