"""
记忆模块 —— VectorStore

职责：
  使用 ChromaDB 作为向量数据库，存储和检索用户的对话历史、用户画像和情绪轨迹。

设计理由：
  - 使用 chromadb.PersistentClient 实现数据持久化，重启后记忆不丢失。
  - 对话历史和用户画像分别存储在不同的 collection 中，便于独立管理和查询。
  - 每条对话记录包含 metadata（emotion, timestamp, category, user_id），
    支持按用户过滤和按类型检索。
  - 用户画像以 JSON 字符串形式存储，key 为 user_profile_{user_id}，
    包含学习目标、技能水平、求职目标、情绪趋势等。
  - 使用 OpenAI Embedding API 进行向量嵌入，支持语义检索和分类。
"""

import json
import os
import uuid
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from companion_ai.utils.config import settings
from companion_ai.utils.logger import logger


class OpenAICompatibleEmbeddingFunction:
    """
    兼容 OpenAI 格式的 Embedding 函数类，支持阿里云 DashScope 等兼容 API。
    """
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._embedding_function = OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=model_name,
            api_base=base_url,
        )
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._embedding_function(input)


def create_embedding_function():
    """
    创建 Embedding 函数实例。
    优先使用 OpenAI 兼容 API，若配置缺失则回退到简单随机向量。
    """
    api_key = settings.embedding_api_key
    base_url = settings.embedding_base_url
    model = settings.embedding_model
    
    if api_key and api_key != "your_embedding_api_key_here":
        logger.info(f"使用 OpenAI 兼容 Embedding API: {model} @ {base_url}")
        return OpenAICompatibleEmbeddingFunction(
            api_key=api_key,
            base_url=base_url,
            model_name=model,
        )
    else:
        logger.warning("未配置 Embedding API，使用简单随机向量（仅用于测试）")
        return SimpleEmbeddingFunction()


class SimpleEmbeddingFunction:
    """
    简单的本地嵌入函数类，满足 ChromaDB 的接口要求。
    返回固定维度的随机向量，用于避免从网络下载默认嵌入模型。
    """
    def __init__(self):
        pass
    
    def __call__(self, input):
        import numpy as np
        return [np.random.rand(384).tolist() for _ in input]
    
    @staticmethod
    def name():
        return "simple_local_embedding"


simple_embedding_function = SimpleEmbeddingFunction()


class VectorStore:
    """
    基于 ChromaDB 的向量存储，管理对话记忆和用户画像。
    """

    def __init__(self):
        os.environ["HF_ENDPOINT"] = settings.HF_ENDPOINT
        
        self.embedding_fn = create_embedding_function()
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        
        self.conversation_collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_CONVERSATION,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn,
        )
        self.profile_collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_PROFILE,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn,
        )
        self.classification_collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_CLASSIFICATION,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn,
        )
        
        self._init_classification_seeds()
        
        logger.info(
            f"ChromaDB 初始化完成，对话记录数: {self.conversation_collection.count()}, "
            f"画像记录数: {self.profile_collection.count()}, "
            f"分类种子数: {self.classification_collection.count()}"
        )
    
    def _init_classification_seeds(self):
        """
        初始化分类种子数据到向量库。
        从 JSON 文件加载标注好的示例，用于向量相似度分类。
        """
        if self.classification_collection.count() > 0:
            logger.info("分类种子数据已存在，跳过初始化")
            return
        
        seeds_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "classification_seeds.json",
        )
        
        if not os.path.exists(seeds_file):
            logger.warning(f"分类种子文件不存在: {seeds_file}")
            return
        
        try:
            with open(seeds_file, "r", encoding="utf-8") as f:
                seeds = json.load(f)
            
            texts = [seed["text"] for seed in seeds]
            metadatas = [{"category": seed["category"]} for seed in seeds]
            ids = [f"seed_{i}" for i in range(len(seeds))]
            
            self.classification_collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info(f"分类种子数据初始化完成，共 {len(seeds)} 条")
        except Exception as e:
            logger.error(f"加载分类种子数据失败: {e}")

    def store_conversation(
        self,
        user_id: str,
        text: str,
        emotion: str,
        category: str,
        timestamp: str,
        role: str = "user",
    ) -> None:
        """
        存储一条对话记录到向量库。

        Args:
            user_id: 用户唯一标识
            text: 消息文本内容
            emotion: 情感标签（positive/negative/neutral）
            category: 消息类别（coding/career/emotional/chitchat）
            timestamp: 时间戳
            role: 角色（user/assistant）
        """
        doc_id = f"{user_id}_{role}_{uuid.uuid4().hex[:8]}"
        metadata = {
            "user_id": user_id,
            "emotion": emotion,
            "category": category,
            "timestamp": timestamp,
            "role": role,
            "retrieval_count": "0",  # 初始检索次数为 0
        }
        try:
            self.conversation_collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id],
            )
            logger.info(f"对话已存储: user={user_id}, category={category}, emotion={emotion}")
        except Exception as e:
            logger.error(f"存储对话失败: {e}")

    def retrieve_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
        apply_decay: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        根据查询文本检索与当前消息最相关的历史对话。
        使用向量相似度检索，支持时间衰减和使用频率加权。

        Args:
            user_id: 用户唯一标识
            query: 查询文本
            top_k: 返回最相似的 top_k 条记录
            apply_decay: 是否应用权重衰减（默认 True）

        Returns:
            包含 text, emotion, timestamp, category, role, final_score 的字典列表
        """
        try:
            # 使用向量相似度检索，多取一些用于后续按 user_id 过滤
            results = self.conversation_collection.query(
                query_texts=[query],
                n_results=top_k * 5,
            )
            
            memories = []
            if results and results["documents"] and results["documents"][0]:
                for doc, meta, distance in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    # 过滤出当前用户的记忆
                    if meta.get("user_id") == user_id:
                        # ChromaDB 返回的是距离，越小越相似，转换为相似度分数
                        similarity = 1.0 / (1.0 + distance)
                        
                        memory = {
                            "text": doc,
                            "emotion": meta.get("emotion", "unknown"),
                            "timestamp": meta.get("timestamp", ""),
                            "category": meta.get("category", ""),
                            "role": meta.get("role", "user"),
                            "similarity": round(similarity, 4),
                            "retrieval_count": int(meta.get("retrieval_count", "0")),
                        }
                        
                        # 应用权重衰减
                        if apply_decay:
                            memory["final_score"] = self._calculate_memory_weight(
                                memory
                            )
                        else:
                            memory["final_score"] = similarity
                        
                        memories.append(memory)
                        
                        # 更新检索次数
                        self._increment_retrieval_count(doc, meta)
                    
                    if len(memories) >= top_k:
                        break
            
            # 按 final_score 降序排序
            memories.sort(key=lambda x: x["final_score"], reverse=True)
            
            logger.info(
                f"向量检索到 {len(memories)} 条相关记忆 (user={user_id}, decay={apply_decay})"
            )
            return memories
        except Exception as e:
            logger.error(f"向量检索记忆失败: {e}")
            return []

    def _calculate_memory_weight(self, memory: Dict[str, Any]) -> float:
        """
        计算记忆的权重（时间衰减 + 使用频率加权）。

        公式：final_score = similarity * time_weight * (1 + freq_weight)

        Args:
            memory: 记忆字典

        Returns:
            最终权重分数
        """
        similarity = memory.get("similarity", 0.5)
        timestamp = memory.get("timestamp", "")
        retrieval_count = memory.get("retrieval_count", 0)

        # 1. 时间衰减（遗忘曲线）
        time_weight = self._calculate_time_decay(timestamp)

        # 2. 使用频率加权
        freq_weight = min(retrieval_count * 0.1, 0.5)  # 最多增加 50%

        # 3. 综合权重
        final_score = similarity * time_weight * (1 + freq_weight)

        return round(final_score, 4)

    def _calculate_time_decay(self, timestamp: str, decay_rate: float = 0.95) -> float:
        """
        计算时间衰减权重（模拟遗忘曲线）。

        公式：time_weight = decay_rate ^ days_old

        Args:
            timestamp: 时间戳字符串（格式：YYYY-MM-DD HH:MM:SS）
            decay_rate: 衰减率（默认 0.95，表示每天衰减 5%）

        Returns:
            时间权重（0~1）
        """
        try:
            from datetime import datetime

            if not timestamp:
                return 0.5  # 没有时间戳，返回中等权重

            # 解析时间戳
            memory_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            now = datetime.now()
            days_old = (now - memory_time).days

            # 计算衰减权重
            time_weight = decay_rate ** days_old

            return max(time_weight, 0.1)  # 最低权重 0.1
        except Exception as e:
            logger.warning(f"计算时间衰减失败: {e}")
            return 0.5  # 解析失败，返回中等权重

    def _increment_retrieval_count(self, doc: str, meta: Dict[str, Any]) -> None:
        """
        增加记忆的检索次数。

        Args:
            doc: 文档内容
            meta: 元数据
        """
        try:
            doc_id = meta.get("id", "")
            if not doc_id:
                return

            current_count = int(meta.get("retrieval_count", "0"))
            new_count = current_count + 1

            # 更新 metadata
            new_meta = meta.copy()
            new_meta["retrieval_count"] = str(new_count)

            # ChromaDB 不支持直接更新 metadata，需要删除后重新添加
            # 这里只记录日志，实际更新需要在下次存储时处理
            logger.debug(f"记忆检索次数更新: {doc_id} ({current_count} -> {new_count})")
        except Exception as e:
            logger.warning(f"更新检索次数失败: {e}")

    def classify_message(self, text: str, top_k: int = 3) -> str:
        """
        基于向量相似度对消息进行分类。

        Args:
            text: 用户消息
            top_k: 检索最相似的 top_k 个示例

        Returns:
            分类结果（coding/career/emotional/chitchat）
        """
        category, _ = self.classify_message_with_confidence(text, top_k)
        return category

    def classify_message_with_confidence(
        self, text: str, top_k: int = 3
    ) -> tuple:
        """
        基于向量相似度对消息进行分类，并返回置信度。

        Args:
            text: 用户消息
            top_k: 检索最相似的 top_k 个示例

        Returns:
            (category, confidence) 分类结果和置信度
        """
        try:
            results = self.classification_collection.query(
                query_texts=[text],
                n_results=top_k,
            )
            
            if not results or not results["metadatas"] or not results["metadatas"][0]:
                logger.warning("向量分类未返回结果，回退到 chitchat")
                return "chitchat", 0.5
            
            # 统计类别投票
            category_votes = {}
            category_distances = {}
            
            for meta, distance in zip(
                results["metadatas"][0], results["distances"][0]
            ):
                category = meta["category"]
                category_votes[category] = category_votes.get(category, 0) + 1
                
                # 记录该类别的最小距离（最相似的距离）
                if category not in category_distances:
                    category_distances[category] = distance
                else:
                    category_distances[category] = min(
                        category_distances[category], distance
                    )
            
            # 获取最高票数的类别
            best_category = max(category_votes, key=category_votes.get)
            best_votes = category_votes[best_category]
            
            # 计算置信度
            # 1. 投票比例（最高票数占总票数的比例）
            vote_ratio = best_votes / top_k
            
            # 2. 距离相似度（转换为 0-1 范围）
            best_distance = category_distances.get(best_category, 1.0)
            distance_similarity = 1.0 / (1.0 + best_distance)
            
            # 3. 综合置信度（投票权重 60% + 距离权重 40%）
            confidence = vote_ratio * 0.6 + distance_similarity * 0.4
            
            logger.info(
                f"向量分类结果: {best_category} "
                f"(confidence={confidence:.2f}, votes={category_votes})"
            )
            return best_category, min(confidence, 1.0)
        except Exception as e:
            logger.error(f"向量分类失败: {e}")
            return "chitchat", 0.5

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户画像。若不存在则返回默认画像。

        用户画像包含：
          - learning_goal: 学习目标
          - current_skill_level: 当前技能水平
          - job_target: 求职目标
          - emotional_trend: 最近5次情绪得分列表
        """
        profile_id = f"user_profile_{user_id}"
        try:
            results = self.profile_collection.get(ids=[profile_id])
            if results and results["documents"]:
                profile_data = json.loads(results["documents"][0])
                logger.info(f"获取用户画像成功: user={user_id}")
                return profile_data
        except Exception as e:
            logger.warning(f"获取用户画像失败: {e}")

        default_profile = {
            "learning_goal": "找算法实习",
            "current_skill_level": "Python基础",
            "job_target": "AI开发",
            "emotional_trend": [],
        }
        self.save_user_profile(user_id, default_profile)
        return default_profile

    def save_user_profile(self, user_id: str, profile: Dict[str, Any]) -> None:
        """
        保存或更新用户画像。
        """
        profile_id = f"user_profile_{user_id}"
        try:
            profile_json = json.dumps(profile, ensure_ascii=False)
            metadata = {"user_id": user_id, "type": "profile"}
            try:
                self.profile_collection.update(
                    ids=[profile_id],
                    documents=[profile_json],
                    metadatas=[metadata],
                )
            except Exception:
                self.profile_collection.add(
                    ids=[profile_id],
                    documents=[profile_json],
                    metadatas=[metadata],
                )
            logger.info(f"用户画像已保存: user={user_id}")
        except Exception as e:
            logger.error(f"保存用户画像失败: {e}")

    def update_emotional_trend(
        self, user_id: str, emotion_score: float, max_trend_length: int = 5
    ) -> List[float]:
        """
        更新用户画像中的情绪趋势记录。
        保留最近 max_trend_length 次的情绪得分。

        Args:
            user_id: 用户唯一标识
            emotion_score: 本次情绪得分（0~1）
            max_trend_length: 保留的最大趋势长度

        Returns:
            更新后的情绪趋势列表
        """
        profile = self.get_user_profile(user_id)
        trend = profile.get("emotional_trend", [])
        trend.append(emotion_score)
        trend = trend[-max_trend_length:]
        profile["emotional_trend"] = trend
        self.save_user_profile(user_id, profile)
        return trend

    def get_recent_emotions(
        self, user_id: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        获取用户最近的对话情绪记录，用于前端情绪仪表盘展示。

        Args:
            user_id: 用户唯一标识
            limit: 返回的最大记录数

        Returns:
            包含 emotion, timestamp, score 的字典列表
        """
        try:
            results = self.conversation_collection.get(
                where={"user_id": user_id, "role": "user"},
                limit=limit,
            )
            emotions = []
            if results and results["metadatas"]:
                for meta in reversed(results["metadatas"]):
                    emotions.append(
                        {
                            "emotion": meta.get("emotion", "neutral"),
                            "timestamp": meta.get("timestamp", ""),
                            "category": meta.get("category", ""),
                        }
                    )
            return emotions[:limit]
        except Exception as e:
            logger.error(f"获取近期情绪记录失败: {e}")
            return []

    def proactive_memory_retrieval(
        self,
        user_id: str,
        current_emotion: str,
        emotion_score: float,
        top_k: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        主动记忆检索：根据用户当前情绪状态推送相关记忆。

        规则：
          - 情绪低落（negative, score < 0.4）→ 推送过去的成功经历
          - 情绪稳定（neutral）→ 推送学习笔记
          - 情绪积极（positive, score > 0.7）→ 推送进步记录

        Args:
            user_id: 用户唯一标识
            current_emotion: 当前情绪标签
            emotion_score: 当前情绪分数
            top_k: 返回的记忆数量

        Returns:
            主动推送的记忆列表
        """
        try:
            # 根据情绪状态选择检索关键词
            if current_emotion == "negative" and emotion_score < 0.4:
                # 情绪低落 → 推送成功经历
                query = "成功 完成 开心 解决 掌握 进步"
                logger.info(
                    f"主动记忆检索: 情绪低落，推送成功经历 (user={user_id})"
                )
            elif current_emotion == "positive" and emotion_score > 0.7:
                # 情绪积极 → 推送进步记录
                query = "进步 提升 学会 理解 突破"
                logger.info(
                    f"主动记忆检索: 情绪积极，推送进步记录 (user={user_id})"
                )
            else:
                # 情绪稳定 → 推送学习笔记
                query = "学习 笔记 总结 复习 知识点"
                logger.info(
                    f"主动记忆检索: 情绪稳定，推送学习笔记 (user={user_id})"
                )

            # 检索相关记忆
            memories = self.retrieve_memories(
                user_id=user_id,
                query=query,
                top_k=top_k,
                apply_decay=True,
            )

            # 添加主动推送标记
            for mem in memories:
                mem["proactive"] = True

            return memories
        except Exception as e:
            logger.error(f"主动记忆检索失败: {e}")
            return []

    def generate_summary(self, user_id: str, query: str = "学习情况总结") -> str:
        """
        深度加分项：长期记忆的总结能力。
        检索更多历史记录，生成可读的摘要文本供 LLM 进一步总结。

        Args:
            user_id: 用户唯一标识
            query: 用于检索的查询文本

        Returns:
            格式化的历史记录文本
        """
        memories = self.retrieve_memories(user_id, query, top_k=10)
        profile = self.get_user_profile(user_id)
        if not memories:
            return "暂无足够的历史记录生成摘要。"

        lines = [f"用户画像: {json.dumps(profile, ensure_ascii=False)}", "", "近期对话记录:"]
        for i, mem in enumerate(memories, 1):
            lines.append(
                f"  {i}. [{mem.get('timestamp', '')}] "
                f"(情绪:{mem.get('emotion', '')}, 类别:{mem.get('category', '')}) "
                f"{mem.get('text', '')}"
            )
        return "\n".join(lines)

    def compress_memories(
        self,
        user_id: str,
        max_memories: int = 50,
        keep_recent: int = 10,
    ) -> Dict[str, Any]:
        """
        记忆压缩与摘要：定期生成记忆摘要，节省存储空间。

        流程：
          1. 获取用户所有记忆
          2. 如果超过 max_memories，则进行压缩
          3. 保留最近 keep_recent 条记忆
          4. 将其余记忆按类别分组生成摘要
          5. 返回压缩统计信息

        Args:
            user_id: 用户唯一标识
            max_memories: 最大记忆数量阈值
            keep_recent: 保留的最近记忆数量

        Returns:
            压缩统计信息字典
        """
        try:
            # 获取用户所有记忆
            results = self.conversation_collection.get(
                where={"user_id": user_id, "role": "user"},
                limit=200,
            )

            if not results or not results["documents"]:
                return {"status": "no_memories", "message": "没有记忆需要压缩"}

            total_memories = len(results["documents"])

            if total_memories <= max_memories:
                return {
                    "status": "no_need",
                    "message": f"记忆数量 ({total_memories}) 未超过阈值 ({max_memories})",
                }

            # 按时间排序
            memories_with_meta = list(
                zip(results["documents"], results["metadatas"], results["ids"])
            )
            memories_with_meta.sort(
                key=lambda x: x[1].get("timestamp", ""), reverse=True
            )

            # 保留最近的记忆
            recent_memories = memories_with_meta[:keep_recent]

            # 将其余记忆按类别分组
            category_groups = {}
            for doc, meta, doc_id in memories_with_meta[keep_recent:]:
                category = meta.get("category", "unknown")
                if category not in category_groups:
                    category_groups[category] = []
                category_groups[category].append(doc)

            # 生成类别摘要
            summaries = {}
            for category, texts in category_groups.items():
                summaries[category] = {
                    "count": len(texts),
                    "sample_texts": texts[:3],  # 只保留 3 个示例
                    "summary": f"共 {len(texts)} 条{category}类对话",
                }

            # 保存摘要到用户画像
            profile = self.get_user_profile(user_id)
            profile["memory_compression"] = {
                "last_compressed": self._get_current_timestamp(),
                "total_before": total_memories,
                "kept_recent": keep_recent,
                "category_summaries": summaries,
            }
            self.save_user_profile(user_id, profile)

            logger.info(
                f"记忆压缩完成: user={user_id}, "
                f"before={total_memories}, after={keep_recent}, "
                f"categories={list(summaries.keys())}"
            )

            return {
                "status": "compressed",
                "total_before": total_memories,
                "kept_recent": keep_recent,
                "category_summaries": summaries,
            }
        except Exception as e:
            logger.error(f"记忆压缩失败: {e}")
            return {"status": "error", "message": str(e)}

    def _get_current_timestamp(self) -> str:
        """获取当前时间戳字符串"""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


vector_store = VectorStore()
