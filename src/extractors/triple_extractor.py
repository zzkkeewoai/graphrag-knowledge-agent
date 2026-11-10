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

        ## 示例3（企业制度域 · 审批链与职责）：
        原文: "《采购付款审批流程》第3.2条：单笔金额达到5万元及以上的采购付款，须经部门负责人与财务总监二级审批。采购部负责采购执行与供应商对接。"
        输出:
        {
            "triples": [
                {"head": "采购付款审批流程第3.2条", "head_type": "条款", "relation": "依据", "tail": "5万元", "tail_type": "阈值", "confidence": 0.92},
                {"head": "财务总监", "head_type": "角色", "relation": "审批", "tail": "采购付款", "tail_type": "流程", "confidence": 0.95},
                {"head": "部门负责人", "head_type": "角色", "relation": "审批", "tail": "采购付款", "tail_type": "流程", "confidence": 0.95},
                {"head": "采购付款审批流程第3.2条", "head_type": "条款", "relation": "适用于", "tail": "采购付款", "tail_type": "流程", "confidence": 0.88},
                {"head": "采购部", "head_type": "部门", "relation": "负责", "tail": "采购执行", "tail_type": "流程", "confidence": 0.93},
                {"head": "采购部", "head_type": "部门", "relation": "负责", "tail": "供应商对接", "tail_type": "流程", "confidence": 0.90}
            ]
        }

        ## 示例4（企业制度域 · 时限与版本）：
        原文: "《发票管理规定》第4.2条：跨年发票须在次年3月31日前完成抵扣。原《发票管理规定》（2025版）同时废止。"
        输出:
        {
            "triples": [
                {"head": "发票管理规定第4.2条", "head_type": "条款", "relation": "截止于", "tail": "次年3月31日", "tail_type": "时限", "confidence": 0.93},
                {"head": "发票管理规定第4.2条", "head_type": "条款", "relation": "适用于", "tail": "跨年发票", "tail_type": "概念", "confidence": 0.85},
                {"head": "发票管理规定（2026版）", "head_type": "文档", "relation": "替代", "tail": "发票管理规定（2025版）", "tail_type": "文档", "confidence": 0.90}
            ]
        }

        ## 示例5（企业域 · 业务系统与权限）：
        原文: "费用报销流程使用 OA 系统。财务部可以访问 ERP 系统的财务模块。"
        输出:
        {
            "triples": [
                {"head": "费用报销流程", "head_type": "流程", "relation": "使用", "tail": "OA 系统", "tail_type": "系统", "confidence": 0.94},
                {"head": "财务部", "head_type": "部门", "relation": "访问", "tail": "ERP 系统", "tail_type": "系统", "confidence": 0.92}
            ]
        }

        ## 示例6（关键 · 条款再化：一条规则同时承载阈值、审批人、流程）：
        ## 带条件的规则必须抽成「条款」节点，让多个条件都挂在它身上。
        ## 不要拆成互不相干的边——那样"哪个阈值对应哪个审批人"就丢了。
        原文: "第2.2条：单笔采购金额达到5万元且低于50万元的，由部门负责人与财务总监共同审批。"
        输出:
        {
            "triples": [
                {"head": "第2.2条", "head_type": "条款", "relation": "依据", "tail": "5万元", "tail_type": "阈值", "confidence": 0.94},
                {"head": "第2.2条", "head_type": "条款", "relation": "依据", "tail": "50万元", "tail_type": "阈值", "confidence": 0.94},
                {"head": "第2.2条", "head_type": "条款", "relation": "审批", "tail": "财务总监", "tail_type": "角色", "confidence": 0.92},
                {"head": "第2.2条", "head_type": "条款", "relation": "审批", "tail": "部门负责人", "tail_type": "角色", "confidence": 0.92},
                {"head": "第2.2条", "head_type": "条款", "relation": "适用于", "tail": "采购付款", "tail_type": "流程", "confidence": 0.90}
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

        ## 实体类型的含义（防止把制度类概念误标成"概念"）：
        - 条款：制度里的具体条文（如"第3.2条"），是规则的宿主；引用要落到条款，而不是整篇文档
        - 阈值：可比较的数值与单位（5万元 / 90日 / 36小时）
        - 时限：具体日期或周期（次年3月31日 / 每月20日）
        - 流程：可执行的动作或环节（审批 / 抵扣 / 付款 / 开票）
        - 角色：职务而非具体的人（财务总监 / 部门负责人）
        - 组织：外部法人（供应商 / 客户）；公司内部单位用"部门"
        - 系统：业务系统（ERP / OA / CRM / SRM / PLM / MES / HRM / ITSM）；
          注意与技术栈区分——业务系统回答"走哪个系统"，技术栈回答"用什么框架"
        - 项目 / 人员 / 技术栈 / 里程碑 / 文档 / 概念：沿用原义

        ## 关系类型的含义：
        - 负责：部门或角色 → 流程/职责（"谁管什么"；部门职责一律用它，不要用「适用于」）
        - 审批：角色 → 流程（"谁批准什么"）；条款 → 角色（"该条款指定谁审批"）
        - 适用于：条款或系统 → 对象/流程（"这条规矩管什么" / "这个系统用于什么"）
        - 截止于：条款 → 时限（"什么时候之前必须做完"）
        - 依据：流程或条款 → 阈值（"按什么标准"）
        - 替代：新版本 → 旧版本（"谁作废了谁"）

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
        - 优先抽取句子的主干关系（谁 → 做什么 → 对谁），不要为了凑数量而堆叠修饰性实体
        - 制度类文本要抽到「条款」粒度（第 X 条），不要把整篇文档当成一个实体
        - 数值与日期优先标成「阈值」「时限」，不要标成「概念」
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