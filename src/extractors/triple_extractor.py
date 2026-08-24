import json
import time
from typing import List, Optional
import logging
from openai import OpenAI
from pydantic import ValidationError

from src.schemas.graph_schema import (
    Triple, ExtractionResult, EntityType, RelationType
)
from src.utils.retry_decorator import retry_on_exception

logger = logging.getLogger(__name__)


class TripleExtractor:
    """
    基于LLM的实体关系三元组抽取器
    使用Few-shot + JSON Schema强制结构化输出
    """

    def __init__(
            self,
            api_key: str,
            base_url: str = "https://api.deepseek.com",  # 或其他兼容OpenAI的端点
            model_name: str = "deepseek-chat",
            temperature: float = 0.1,  # 低温度保证确定性
            confidence_threshold: float = 0.7
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold

        # 构建Few-shot示例
        self.few_shot_examples = self._build_few_shot()

    def _build_few_shot(self) -> str:
        """构建Few-shot示例，引导LLM学习抽取模式"""
        return """
        ## 示例1：
        原文: "A项目使用Python和PyTorch框架，由张伟团队负责开发，预计12月上线。"
        输出:
        {
            "triples": [
                {"head": "A项目", "head_type": "项目", "relation": "使用", "tail": "Python", "tail_type": "技术栈", "confidence": 0.95},
                {"head": "A项目", "head_type": "项目", "relation": "使用", "tail": "PyTorch", "tail_type": "技术栈", "confidence": 0.95},
                {"head": "A项目", "head_type": "项目", "relation": "属于", "tail": "张伟团队", "tail_type": "部门", "confidence": 0.90},
                {"head": "A项目", "head_type": "项目", "relation": "后置", "tail": "12月上线", "tail_type": "里程碑", "confidence": 0.85}
            ]
        }

        ## 示例2：
        原文: "B项目依赖Java和Spring Boot，与A项目在架构设计上有相似之处。"
        输出:
        {
            "triples": [
                {"head": "B项目", "head_type": "项目", "relation": "依赖", "tail": "Java", "tail_type": "技术栈", "confidence": 0.92},
                {"head": "B项目", "head_type": "项目", "relation": "依赖", "tail": "Spring Boot", "tail_type": "技术栈", "confidence": 0.92},
                {"head": "B项目", "head_type": "项目", "relation": "引用", "tail": "A项目", "tail_type": "项目", "confidence": 0.78}
            ]
        }
        """

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        entity_types = ", ".join([e.value for e in EntityType])
        relation_types = ", ".join([r.value for r in RelationType])

        return f"""
        你是一个专业的知识图谱实体关系抽取专家。

        ## 任务：
        从给定的非结构化文本中，抽取所有实体（Entity）和关系（Relation），并以JSON格式输出。

        ## 约束条件：
        1. 实体类型必须严格属于以下类型之一：{entity_types}
        2. 关系类型必须严格属于以下类型之一：{relation_types}
        3. 为每个三元组给出置信度分数（0-1），反映抽取的可信度
        4. 只抽取确定性的陈述，不抽取模糊或推测性内容
        5. 保持原文本中的实体名称，不要改写

        ## 输出格式（严格遵守此JSON Schema）：
        {{
            "triples": [
                {{
                    "head": "实体名称",
                    "head_type": "实体类型",
                    "relation": "关系类型",
                    "tail": "实体名称", 
                    "tail_type": "实体类型",
                    "confidence": 0.95
                }}
            ]
        }}

        ## Few-shot示例：
        {self.few_shot_examples}

        ## 注意：
        - 如果原文没有明确的关系，不要强行构造
        - 优先抽取"核心实体"（项目、人员、技术栈）之间的关系
        - 输出必须是合法的JSON，不要包含任何额外说明文字
        """

    @retry_on_exception(max_retries=3, delay=1.0, backoff=2.0)
    def _call_llm(self, chunk_text: str) -> str:
        """调用LLM，带重试机制"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": f"请抽取以下文本中的三元组：\n\n{chunk_text}"}
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"}  # DeepSeek支持JSON模式
        )
        return response.choices[0].message.content

    def extract(self, chunk_text: str, chunk_id: str = "") -> ExtractionResult:
        """
        从文本块中抽取三元组
        Args:
            chunk_text: 待抽取的文本片段
            chunk_id: 文本块ID（用于溯源）
        Returns:
            ExtractionResult: 包含三元组列表和元信息
        """
        start_time = time.time()

        # Step 1: 调用LLM
        raw_response = self._call_llm(chunk_text)

        # Step 2: 解析JSON
        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError as e:
            logger.error(f"LLM返回非JSON格式: {raw_response[:200]}...")
            raise ValueError(f"无效的JSON响应: {e}")

        # Step 3: 校验并构建Triple对象
        valid_triples = []
        total_tokens = 0  # 实际从response中获取

        for triple_dict in data.get("triples", []):
            try:
                # 自动类型转换和验证
                triple = Triple(
                    head=triple_dict["head"].strip(),
                    head_type=EntityType(triple_dict["head_type"]),
                    relation=RelationType(triple_dict["relation"]),
                    tail=triple_dict["tail"].strip(),
                    tail_type=EntityType(triple_dict["tail_type"]),
                    confidence=float(triple_dict["confidence"]),
                    source_chunk=chunk_text[:200] + "...",  # 截断存储
                    metadata={"chunk_id": chunk_id}
                )

                # 置信度过滤
                if triple.confidence >= self.confidence_threshold:
                    valid_triples.append(triple)
                else:
                    logger.debug(f"过滤低置信度三元组: {triple.head} -> {triple.tail} (conf={triple.confidence})")

            except (ValidationError, ValueError) as e:
                logger.warning(f"跳过无效三元组: {triple_dict}, 错误: {e}")
                continue

        elapsed_ms = (time.time() - start_time) * 1000

        logger.info(
            f"抽取完成: 原始{len(data.get('triples', []))}个三元组, "
            f"通过校验{len(valid_triples)}个"
        )

        return ExtractionResult(
            triples=valid_triples,
            total_tokens_used=total_tokens,
            extraction_time_ms=elapsed_ms
        )

    def batch_extract(
            self,
            chunks: List[tuple],  # [(chunk_id, chunk_text), ...]
            batch_size: int = 5
    ) -> List[ExtractionResult]:
        """
        批量抽取（带进度条）
        """
        results = []
        total = len(chunks)

        for idx, (chunk_id, chunk_text) in enumerate(chunks, 1):
            logger.info(f"处理第 {idx}/{total} 个chunk: {chunk_id}")
            try:
                result = self.extract(chunk_text, chunk_id)
                results.append(result)
            except Exception as e:
                logger.error(f"处理chunk {chunk_id} 失败: {e}")
                # 返回空结果但继续处理
                results.append(ExtractionResult(
                    triples=[],
                    total_tokens_used=0,
                    extraction_time_ms=0
                ))

        return results