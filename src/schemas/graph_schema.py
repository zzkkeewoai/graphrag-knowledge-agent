"""图谱数据契约：实体类型、关系类型、三元组。

【本次改了什么，为什么】
原有 7 类实体是为「IT 项目文档」设计的（项目/人员/技术栈/里程碑/部门/文档/概念）。
要支撑「企业制度 / 流程 / 审批」类知识库，这 7 类覆盖不住：
    「5 万元」「次年 3 月 31 日」「二级审批」全都只能塞进「概念」，
    于是图里堆满同类型的「概念」节点，检索时无法区分它们。

【原则：只增不改（很重要）】
不修改、不删除任何已有枚举值，只做增量。三个原因：

  1. 枚举值是**写入图谱的数据契约**。改一个值 = 库里旧节点全部对不上，
     要做数据迁移；旧数据是演示库还能清库重灌，但不能假设永远能清。
  2. 抽取器的 prompt 是动态拼的（src/extractors/triple_extractor.py:67
     `", ".join(e.value for e in EntityType)`），改枚举会立刻改变模型行为。
  3. 旧的项目域题库（data/eval_queries.json）还能当**回归集**用；
     改类型等于把回归基线也一起弄坏了。

所以：7 + 6 = 13 类实体，8 + 5 = 13 类关系。

【一个已知的粗糙点（先记下来，别假装没有）】
「阈值」的实体名天然带单位（"5万元"、"90日"、"36小时"），比较时得先解析数字。
更规范的做法是 name 存规范化数值、metadata 存 {value, unit}，
但那需要抽取器配合，本轮先不做 —— **先让类型对得上，再谈精度**。
"""

from enum import Enum
from typing import List, Dict, Any
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """强制约束的实体类型。

    前 7 类是原有（项目域），后 6 类是新增（企业域）。
    两组并存是刻意的：同一个知识库里可能既有项目文档、也有制度文档，
    而且旧数据不需要迁移。
    """

    # ---------- 原有：项目域 ----------
    PROJECT = "项目"
    PERSON = "人员"
    TECH_STACK = "技术栈"
    MILESTONE = "里程碑"
    DEPARTMENT = "部门"
    DOCUMENT = "文档"
    CONCEPT = "概念"

    # ---------- 新增：企业域 ----------
    # 制度里的具体条目（"第 4.2 条"）。它是规则的宿主，
    # 引用要落到它，而不是落到整篇文档 —— 否则无法判断"引没引对条"。
    CLAUSE = "条款"

    # 5万元 / 90日 / 36小时。必须单独成类，因为它要参与**比较和计算**：
    # "30 万够不够 5 万"、"90 天过没过"。概念做不到这件事。
    THRESHOLD = "阈值"

    # 次年3月31日 / 每月20日。要判断"是否已过"，必须有独立的日期类型。
    DEADLINE = "时限"

    # 审批 / 抵扣 / 付款 / 开票。流程之间是有序的（前置/后置），
    # 而"概念"之间没有边，串不成链。
    PROCESS = "流程"

    # 财务总监 / 部门负责人。注意它和「人员」的区别：
    # 人员是具体的人，角色是职务；审批规则挂在角色上，不挂在某个人身上。
    ROLE = "角色"

    # 供应商 / 客户。外部法人。与内部「部门」区分的理由很实际：
    # "谁审批"的答案只能是内部角色，"向谁付款"的答案是外部组织。
    ORGANIZATION = "组织"

    # ERP / OA / CRM / SRM / PLM / MES / HRM / ITSM。业务系统。
    # 为什么不能复用「技术栈」：实测中 8 个业务系统全被标成技术栈，
    # 和技术栈节点（Python / Neo4j）混在一起。但这两类回答的是不同问题：
    #   "这个项目用什么框架" → 技术栈
    #   "这个流程走哪个系统" → 系统
    # 混在一起，检索时无法按类型过滤，多跳链也会串错。
    SYSTEM = "系统"


class RelationType(str, Enum):
    """强制约束的关系类型。前 8 类原有，后 5 类新增。"""

    # ---------- 原有 ----------
    USES = "使用"
    DEPENDS_ON = "依赖"
    LEADS = "领导"
    BELONGS_TO = "属于"
    PRECEDES = "前置"
    FOLLOWS = "后置"
    REFERENCES = "引用"
    CONTAINS = "包含"

    # ---------- 新增 ----------
    # 角色 → 流程。审批链的核心边：「财务总监 -审批-> 采购付款」。
    APPROVES = "审批"

    # 条款 → 对象/部门。回答"这条规矩管谁"。
    APPLIES_TO = "适用于"

    # 条款 → 时限。回答"什么时候之前必须做完"。
    DUE_BY = "截止于"

    # 流程 → 阈值/口径。回答"按什么标准算"。
    BASED_ON = "依据"

    # 新版本 → 旧版本。版本冲突题的必杀边：
    # 没有它，模型只能靠"生效日期"自己比，很容易引到已废止的那版。
    REPLACES = "替代"

    # 部门/角色 → 流程/职责/对象。回答"谁管什么"。
    # 为什么必须单独加：实测中 40 条「部门→职责」被硬塞进了「适用于」，
    # 而「适用于」本该留给"条款→对象"。一个关系被占用，另一个语义就废了。
    RESPONSIBLE_FOR = "负责"

    # 部门/角色 → 系统。回答"谁能进哪个系统"（权限）。
    # 加它的原因同样是实测：权限句没有对应关系，21 条「部门→系统」
    # 又挤回了「适用于」。**每缺一个关系，最像的那个就会被占满。**
    ACCESS = "访问"


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
