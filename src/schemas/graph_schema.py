from enum import Enum
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class EntityType(str, Enum):
    """强制约束的实体类型"""
    PROJECT = "项目"
    PERSON = "人员"
    TECH_STACK = "技术栈"
    MILESTONE = "里程碑"
    DEPARTMENT = "部门"
    DOCUMENT = "文档"
    CONCEPT = "概念"

class RelationType(str, Enum):
    """强制约束的关系类型"""
    USES = "使用"
    DEPENDS_ON = "依赖"
    LEADS = "领导"
    BELONGS_TO = "属于"
    PRECEDES = "前置"
    FOLLOWS = "后置"
    REFERENCES = "引用"
    CONTAINS = "包含"

class Triple(BaseModel):
    """三元组数据模型"""
    head: str
    head_type: EntityType
    relation: RelationType
    tail: str
    tail_type: EntityType
    confidence: float = Field(ge=0.0, le=1.0, description="置信度分数")
    source_chunk: str = Field(description="抽取来源的原文片段")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ExtractionResult(BaseModel):
    """抽取结果容器"""
    triples: List[Triple]
    total_tokens_used: int
    extraction_time_ms: float