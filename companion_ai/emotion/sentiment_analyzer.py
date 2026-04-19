"""
情感分析模块 —— SentimentAnalyzer

职责：
  对用户输入文本进行情感分析，返回情感标签（positive/negative/neutral）和情感分数（0~1）。

设计理由：
  - 优先使用 transformers 的预训练模型 distilbert-base-uncased-finetuned-sst-2-english，
    该模型在 SST-2 数据集上微调，对英文情感分类效果优秀。
  - 由于用户输入可能为中文，模型对中文情感判断可能不够准确，
    因此提供基于关键词的本地回退方案，确保系统在模型加载失败时仍能工作。
  - 回退逻辑基于中英文情感关键词匹配，覆盖常见的情绪表达。
"""

import os
import re
from typing import Tuple

from companion_ai.utils.config import settings
from companion_ai.utils.logger import logger


class SentimentAnalyzer:
    """
    情感分析器，支持 transformers 预训练模型和本地关键词回退两种模式。
    """

    NEGATIVE_KEYWORDS_CN = [
        "焦虑", "累", "烦", "难过", "崩溃", "压力", "抑郁", "沮丧",
        "失望", "担心", "害怕", "疲惫", "无助", "绝望", "痛苦",
        "心累", "烦躁", "郁闷", "低落", "丧", "不想", "放弃",
        "好累", "好烦", "好难", "太难", "好大", "受不了", "烦死",
    ]
    POSITIVE_KEYWORDS_CN = [
        "开心", "谢谢", "高兴", "棒", "喜欢", "兴奋", "满足",
        "成功", "进步", "感谢", "不错", "厉害", "优秀", "自信",
        "愉快", "舒适", "期待", "有趣", "收获", "很好", "太好了",
        "好开心", "好棒",
    ]
    NEGATIVE_KEYWORDS_EN = [
        "anxious", "tired", "annoyed", "sad", "depressed", "stressed",
        "frustrated", "worried", "exhausted", "helpless", "hopeless",
        "painful", "overwhelmed", "upset", "angry", "afraid", "bad",
        "terrible", "awful", "miserable",
    ]
    POSITIVE_KEYWORDS_EN = [
        "happy", "thanks", "great", "good", "love", "excited", "satisfied",
        "success", "progress", "grateful", "nice", "awesome", "excellent",
        "confident", "joyful", "comfortable", "looking forward", "fun",
        "amazing", "wonderful",
    ]

    def __init__(self):
        self.pipeline = None
        self.use_fallback = False
        self._load_model()

    def _load_model(self):
        """
        尝试加载 transformers 预训练情感分析 pipeline。
        若加载失败（网络问题、依赖缺失等），自动切换到关键词回退模式。

        SENTIMENT_FALLBACK_ENABLED 的语义：
          - True（默认）：先尝试加载模型，失败后回退到关键词方案
          - False：直接使用关键词方案，跳过模型加载（适用于无网络或快速启动场景）
        """
        os.environ["HF_ENDPOINT"] = settings.HF_ENDPOINT
        
        if settings.SENTIMENT_FALLBACK_ENABLED:
            try:
                from transformers import pipeline as hf_pipeline

                logger.info(
                    f"正在加载情感分析模型: {settings.SENTIMENT_MODEL_NAME} (镜像源: {settings.HF_ENDPOINT})"
                )
                self.pipeline = hf_pipeline(
                    "sentiment-analysis",
                    model=settings.SENTIMENT_MODEL_NAME,
                )
                logger.info("情感分析模型加载成功")
            except Exception as e:
                logger.warning(
                    f"情感分析模型加载失败: {e}，将使用关键词回退方案"
                )
                self.use_fallback = True
        else:
            self.use_fallback = True
            logger.info("已配置跳过模型加载，直接使用关键词方案")

    def analyze(self, text: str) -> Tuple[str, float]:
        """
        分析文本情感，返回 (emotion_label, emotion_score)。

        逻辑：
          - 对于中文文本，强制使用关键词回退方案（英文模型对中文识别效果差）
          - 对于英文文本，使用模型分析

        Args:
            text: 用户输入文本

        Returns:
            emotion_label: "positive" / "negative" / "neutral"
            emotion_score: 0.0 ~ 1.0 的情感强度分数
        """
        if not text or not text.strip():
            return "neutral", 0.5

        # 检测是否包含中文字符
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
        
        if has_chinese:
            # 中文文本使用关键词回退方案
            return self._analyze_with_keywords(text)
        elif self.pipeline is not None and not self.use_fallback:
            # 英文文本使用模型分析
            return self._analyze_with_model(text)
        else:
            # 其他情况使用关键词方案
            return self._analyze_with_keywords(text)

    def _analyze_with_model(self, text: str) -> Tuple[str, float]:
        """
        使用 transformers pipeline 进行情感分析。
        模型输出 POSITIVE/NEGATIVE 标签和对应置信度分数。
        """
        try:
            result = self.pipeline(text)[0]
            label = result["label"].lower()
            score = round(result["score"], 4)

            if label == "positive":
                return "positive", score
            elif label == "negative":
                return "negative", score
            else:
                return "neutral", 0.5
        except Exception as e:
            logger.error(f"模型情感分析异常: {e}，回退到关键词方案")
            return self._analyze_with_keywords(text)

    def _analyze_with_keywords(self, text: str) -> Tuple[str, float]:
        """
        基于关键词的本地回退情感分析。

        逻辑：
          1. 优先匹配更长的复合关键词（如"好累"优先于"累"），
             避免短词被复合词中的字干扰
          2. 统计文本中出现的正面/负面关键词数量
          3. 根据关键词命中数计算分数
          4. 若正负均无命中，判定为 neutral

        分数计算方式：
          - 命中关键词越多，分数越极端（越接近 0 或 1）
          - neutral 分数固定为 0.5
        """
        text_lower = text.lower()

        all_neg = sorted(
            self.NEGATIVE_KEYWORDS_CN + self.NEGATIVE_KEYWORDS_EN,
            key=len, reverse=True,
        )
        all_pos = sorted(
            self.POSITIVE_KEYWORDS_CN + self.POSITIVE_KEYWORDS_EN,
            key=len, reverse=True,
        )

        neg_matched = []
        temp_text = text_lower
        for kw in all_neg:
            if kw in temp_text:
                neg_matched.append(kw)
                temp_text = temp_text.replace(kw, " " * len(kw), 1)

        pos_matched = []
        temp_text = text_lower
        for kw in all_pos:
            if kw in temp_text:
                pos_matched.append(kw)
                temp_text = temp_text.replace(kw, " " * len(kw), 1)

        neg_count = len(neg_matched)
        pos_count = len(pos_matched)

        if neg_count == 0 and pos_count == 0:
            return "neutral", 0.5

        if neg_count > pos_count:
            score = min(0.95, 0.55 + neg_count * 0.08)
            return "negative", round(score, 4)
        elif pos_count > neg_count:
            score = min(0.95, 0.55 + pos_count * 0.08)
            return "positive", round(score, 4)
        else:
            return "neutral", 0.5

    def contains_code(self, text: str) -> bool:
        """
        检测文本中是否包含代码片段。
        通过常见的代码特征模式匹配：代码块标记、行尾分号、函数定义等。
        """
        code_patterns = [
            r"```[\s\S]*?```",
            r"def\s+\w+\s*\(",
            r"class\s+\w+",
            r"import\s+\w+",
            r"from\s+\w+\s+import",
            r"print\s*\(",
            r"return\s+",
            r"for\s+\w+\s+in\s+",
            r"while\s+\w+",
            r"if\s+\w+.*:",
            r"=\s*\[",
            r"\w+\.\w+\(.*\)",
        ]
        for pattern in code_patterns:
            if re.search(pattern, text):
                return True
        return False


sentiment_analyzer = SentimentAnalyzer()
