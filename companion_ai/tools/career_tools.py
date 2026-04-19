"""
求职辅导工具模块

包含简历评分、面试题库等求职相关工具。
"""

import random
from typing import Dict, Any
from langchain_core.tools import tool


@tool
def evaluate_resume(resume_text: str) -> Dict[str, Any]:
    """
    简历评分工具（简化版）。
    
    从简历文本中提取关键信息并给出评分（0-100）。
    
    Args:
        resume_text: 简历文本内容
        
    Returns:
        包含评分、优点、改进建议的字典
    """
    try:
        score = 0
        strengths = []
        improvements = []
        
        keywords = {
            "技能": ["Python", "Java", "C++", "算法", "数据结构", "机器学习", "深度学习", "SQL"],
            "项目": ["项目", "实习", "开发", "设计", "实现"],
            "成果": ["优化", "提升", "减少", "增加", "改进", "完成"],
            "经历": ["实习", "工作", "实践", "比赛", "竞赛"],
        }
        
        keyword_count = 0
        for category, words in keywords.items():
            for word in words:
                if word in resume_text:
                    keyword_count += 1
                    strengths.append(f"包含{category}相关关键词：{word}")
        
        score = min(100, 40 + keyword_count * 3)
        
        if len(resume_text) < 200:
            score -= 15
            improvements.append("简历内容偏短，建议补充更多项目细节和成果描述")
        elif len(resume_text) > 500:
            score += 5
            strengths.append("简历内容较为丰富")
        
        if not any(kw in resume_text for kw in ["github", "GitHub", "链接", "link"]):
            improvements.append("建议添加 GitHub 链接或项目链接")
        
        if not any(kw in resume_text for kw in ["%", "倍", "ms", "秒", "提升", "减少"]):
            improvements.append("建议添加量化成果，如性能提升百分比、效率提升等")
        
        score = max(0, min(100, score))
        
        score_level = "待改进"
        if score >= 80:
            score_level = "优秀"
        elif score >= 60:
            score_level = "良好"
        
        return {
            "success": True,
            "score": score,
            "score_level": score_level,
            "strengths": strengths[:5] if strengths else ["简历结构基本完整"],
            "improvements": improvements[:5] if improvements else ["保持当前内容，可继续丰富细节"],
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "score": 0,
        }


@tool
def get_interview_questions(topic: str = "AI", count: int = 3) -> Dict[str, Any]:
    """
    面试题库工具。
    
    根据主题获取面试问题。
    
    Args:
        topic: 面试主题（AI/算法/编程/系统设计/行为面试）
        count: 问题数量
        
    Returns:
        包含面试问题列表的字典
    """
    question_bank = {
        "AI": [
            "请解释什么是过拟合，以及如何防止过拟合？",
            "请说明神经网络中激活函数的作用，并举例几种常见的激活函数？",
            "请解释梯度下降算法及其变种（SGD、Adam）的区别？",
            "什么是 RNN 和 LSTM，它们解决了什么问题？",
            "请解释 Transformer 架构中的自注意力机制？",
        ],
        "算法": [
            "请说明快速排序的时间复杂度和空间复杂度？",
            "请解释动态规划的适用场景和核心思想？",
            "请说明红黑树和 AVL 树的区别？",
            "请解释图的 BFS 和 DFS 遍历算法？",
            "请说明哈希冲突的解决方法？",
        ],
        "编程": [
            "请实现一个单例模式？",
            "请说明 Python 中的 GIL（全局解释器锁）？",
            "请解释进程和线程的区别？",
            "请说明什么是死锁以及如何避免死锁？",
            "请解释 Python 中的装饰器？",
        ],
        "系统设计": [
            "请设计一个高并发的秒杀系统？",
            "请设计一个分布式缓存系统？",
            "请设计一个消息队列系统？",
            "请设计一个短链接服务？",
            "请设计一个实时聊天系统？",
        ],
        "行为面试": [
            "请描述一次你解决的最困难的技术问题？",
            "请举例说明你在团队中如何处理冲突？",
            "请描述一次你学习新技术的经历？",
            "请举例说明你如何应对压力？",
            "请描述一次你主动改进工作流程的经历？",
        ],
    }
    
    topic = topic if topic in question_bank else "AI"
    questions = random.sample(question_bank[topic], min(count, len(question_bank[topic])))
    
    return {
        "success": True,
        "topic": topic,
        "questions": questions,
    }
