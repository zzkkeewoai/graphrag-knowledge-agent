"""
evidence_validator.py —— 证据验证器（v2.2 新增）

核心语义：
  "相关（relevant）" ≠ "支持结论（support）"。
  检索到相关内容 ≠ 能支撑用户问题中的核心断言。

例如：
  query: "张三负责哪些项目？"    期望关系: "负责"
  graph: 张三 --参与--> 项目A      ← 相关，但不能证明"负责"
  graph: 张三 --负责--> 项目B      ← 正向证据

关键陷阱（本模块刻意规避）：
  relation != expected_relation   ≠   负向证据
  "没有证据证明张三负责项目A"     ≠   "证明张三不负责项目A"
  代码中绝不能因为 relation != expected 就生成 negative evidence，
  只能判定为 insufficient（证据不足）。

设计原则：
  1. 确定性规则优先，不调用 LLM 作为第一选择；
  2. 只读 state / 检索结果，不写入 Neo4j（运行状态只活在 AgentState）；
  3. 最小侵入：不改变 HybridRetriever / RRF / Reranker 的任何逻辑。
"""
import logging
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# ============================================================
# 关系类型词表（轻量 Query Understanding 复用）
# 与 RouterAgent 的关键词路由思路一致，但不重复设计一套 Query
# Understanding —— 这里只负责"从 query 提取期望关系"，供证据匹配。
# ============================================================
RELATION_KEYWORDS = {
    "负责": ["负责", "主管", "掌管", "管理"],
    "参与": ["参与", "参加", "加入"],
    "依赖": ["依赖", "依赖于", "依赖了"],
    "使用": ["使用", "用了", "采用", "用什么"],
    "领导": ["领导", "负责带领", "牵头"],
    "属于": ["属于", "属于哪个", "归属"],
    "引用": ["引用", "引用了", "参照"],
}

# 反向否定词（用于 conflict 检测：文本说"不再/没有负责"）
NEGATION_KEYWORDS = ["不再", "没有", "未", "不负责", "已卸任", "已不再", "已经不再"]

# 疑问/无意义词（用于从 query 提取目标实体时的占位）
_QUESTION_WORDS = ["哪些", "什么", "哪个", "谁", "多少", "如何", "怎么", "呢", "？", "?", "了", "的", "是", "在"]

# 常见宾语/对象词（"张三负责的项目"里"项目"是对象不是实体主语）
_OBJECT_WORDS = ["项目", "系统", "技术", "模块", "产品", "服务", "平台", "任务", "工作", "方面", "有哪些", "什么", "哪个"]


def extract_target_relation(query: str) -> Optional[str]:
    """从 query 提取期望关系类型（确定性规则）。

    query: "张三负责哪些项目？"  →  "负责"
    query: "张三依赖什么系统？"  →  "依赖"
    提取不到返回 None（此时证据验证退化为"是否检索到内容"）。
    """
    for rel, keywords in RELATION_KEYWORDS.items():
        for kw in keywords:
            if kw in query:
                return rel
    return None


def extract_target_entity(query: str, known_entities: List[str]) -> Optional[str]:
    """提取目标实体。

    优先：从检索结果里已知实体中，找出出现在 query 里的那个（最可靠）。
    兜底：query 去掉关系词/疑问词/宾语词后，取首段 2-6 字片段
          （简单启发式，仅作为 known_entities 缺失时的后备，不引入新依赖）。
    """
    # 1. 已知实体匹配（graph_results 里的 entity 出现在 query 中）
    for ent in known_entities:
        if ent and ent in query:
            return ent

    # 2. 启发式：去掉关系词/疑问词/宾语词
    cleaned = query
    for rel, keywords in RELATION_KEYWORDS.items():
        for kw in keywords:
            cleaned = cleaned.replace(kw, "")
    for w in _QUESTION_WORDS + _OBJECT_WORDS:
        cleaned = cleaned.replace(w, "")
    # 再去掉残留标点
    cleaned = re.sub(r"[？?。，,！!、\s]", "", cleaned)
    cleaned = cleaned.strip()
    # 中文人名/实体通常在 2-6 字；取清洗后整体
    if 1 <= len(cleaned) <= 12:
        return cleaned
    return None


def _text_mentions_negation(text: str, entity: str, expected_rel: str) -> bool:
    """判断文本片段是否包含"实体 + 否定词 + 期望关系"（conflict 信号）。

    例：text="张三不再负责项目A"  entity="张三"  expected_rel="负责"
        → 命中否定词"不再" 且同时出现 entity 与 rel → True
    """
    if not entity or not expected_rel:
        return False
    has_negation = any(nw in text for nw in NEGATION_KEYWORDS)
    if not has_negation:
        return False
    # 否定词附近需出现实体与关系（粗粒度同现即可，避免把无关否定算进来）
    has_entity = entity in text
    has_rel = expected_rel in text
    return has_entity and has_rel


# ============================================================
# 文本证据"方向"判断（v2.2 Conflict Resolver 前置）
# 用于区分：
#   文本正向支持："张三负责项目A"           → 支持"负责"
#   文本负向反对："张三不再负责项目A"         → 反对"负责"（conflict 候选）
#   文本中性/无关："张三参与项目A的开发"      → 不支持也不反对
# ============================================================

# 明确的时间信号词（用于优先级判断：最新证据优先）
_TIME_SIGNALS = ["现在", "目前", "当前", "如今", "现由", "截至", "如今是", "现在由", "202", "20", "年", "月"]

# 正向支持模式（实体 + 关系词，无否定）
_SUPPORT_RELS = ["负责", "管理", "掌管", "主管", "领导"]


def _text_supports(text: str, entity: str, expected_rel: str) -> bool:
    """文本是否明确正向支持"entity expected_rel ..."。

    例：text="张三负责项目A，同时负责项目B"  entity="张三" rel="负责" → True
    注意：不把"参与"当成对"负责"的支持。
    """
    if not entity or not expected_rel:
        return False
    if any(nw in text for nw in NEGATION_KEYWORDS):
        return False                       # 带否定的句子不是正向支持
    if entity not in text or expected_rel not in text:
        return False
    # 排除"参与/加入"等弱关系被误当成强关系支持
    if "参与" in text and expected_rel == "负责":
        return False
    return True


def _text_opposes(text: str, entity: str, expected_rel: str) -> bool:
    """文本是否明确反对"entity expected_rel ..."（即冲突源）。

    例：text="张三不再负责项目A，已交接给李四" → True
    """
    if not entity or not expected_rel:
        return False
    has_entity = entity in text
    has_rel = expected_rel in text
    has_negation = any(nw in text for nw in NEGATION_KEYWORDS)
    # 要求否定词与实体/关系同现，且否定作用于该关系
    # "张三不再负责项目A" 命中；"张三没有参与项目B"（rel=负责时）不命中
    if not (has_entity and has_rel and has_negation):
        return False
    # 细化：否定词必须紧邻实体与关系之间或之前（粗粒度：出现在同一子句）
    # 用"不再/没有/未 + rel"或"rel + 已经不再"等模式粗判
    for nw in NEGATION_KEYWORDS:
        if nw in text:
            # 找否定词位置与 rel 位置，要求距离较近（同一短句内）
            pos_nw = text.find(nw)
            pos_rel = text.find(expected_rel)
            if pos_nw >= 0 and pos_rel >= 0 and abs(pos_nw - pos_rel) <= 20:
                return True
    return False


def _text_has_time_signal(text: str) -> bool:
    """文本是否包含明确时间信号（现在/目前/截至/年份…），供优先级判断。"""
    return any(sig in text for sig in _TIME_SIGNALS)


def _text_mentions_entity_rel(text: str, entity: str, expected_rel: str) -> bool:
    """文本是否同时提到实体与关系（无论方向），供 source 兜底判断。"""
    return bool(entity) and bool(expected_rel) and entity in text and expected_rel in text


class EvidenceValidator:
    """证据验证器：判断检索证据是否足以支持用户问题。

    输出（写入 state）：
      evidence_status: sufficient / insufficient / conflict / unknown
      next_action:     answer / targeted_graph / refuse / conflict
      conflict_status: conflict / no_conflict / unknown
    """

    def __init__(self):
        # conflict 处理当前为预留：检测到 conflict 时标记状态并进入 conflict 分支，
        # ConflictResolver 可作为后续扩展点（此处不引入新架构）。
        pass

    # ----------------------------------------------------------
    # 核心判断
    # ----------------------------------------------------------
    def validate(
        self,
        query: str,
        graph_results: List[Dict],
        documents: List[Dict],
        targeted_retried: bool = False,
        known_entities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """对检索结果做证据验证。

        Args:
            query: 用户问题
            graph_results: 图谱检索结果（含 entity/relation/target/confidence）
            documents: 精排后的文档（含 text 等，用于 conflict 文本检测）
            targeted_retried: 是否已做过 targeted graph retrieval
            known_entities: 候选实体（通常来自 graph_results 的 entity 字段）

        Returns:
            {"evidence_status": ..., "next_action": ..., "target_relation": ...,
             "target_entity": ..., "conflict_status": ..., "evidence_detail": [...]}
        """
        target_relation = extract_target_relation(query)
        known = known_entities or [r.get("entity", "") for r in (graph_results or [])]
        target_entity = extract_target_entity(query, known)

        logger.info(
            f"[Evidence] relation={target_relation} entity={target_entity} "
            f"graph_results={len(graph_results or [])} targeted_retried={targeted_retried}"
        )

        # 无目标关系：退化为"有没有检索到内容"
        if not target_relation:
            if graph_results or documents:
                return self._sufficient(query, target_relation, target_entity)
            return self._insufficient(query, target_relation, target_entity, targeted_retried)

        # 有目标关系 → 结构化匹配 graph 的 relation
        positive: List[Dict] = []   # relation == expected 的正向证据
        partial: List[Dict] = []    # 相关但 relation != expected
        for r in graph_results or []:
            rel = r.get("relation", "")
            # 注意：多跳关系形如 "使用-依赖"，这里只看是否包含期望关系
            matched = rel == target_relation or target_relation in str(rel).split("-")
            item = {
                "entity": r.get("entity", ""),
                "relation": rel,
                "target": r.get("target", ""),
                "confidence": r.get("confidence", 0),
                "text": r.get("text", ""),
                "source": "graph",
                "timestamp": r.get("timestamp"),
            }
            if matched:
                positive.append(item)
            elif rel and r.get("target"):
                # 有实体有目标但关系不同 → 相关但不是结论支持（关键：不是 negative！）
                partial.append(item)

        # ---- 文本证据方向判断（v2.2 新增）----
        # vector/documents 文本对"entity expected_rel"的支持/反对方向
        text_positive: List[Dict] = []   # 文本正向支持（如"张三负责项目A"）
        text_negative: List[Dict] = []   # 文本负向反对（如"张三不再负责项目A"）
        for doc in documents or []:
            text = doc.get("text", "")
            item = {
                "text": text,
                "source": "text",
                "doc_id": doc.get("doc_id", ""),
                "timestamp": doc.get("timestamp") or doc.get("date"),
            }
            if _text_opposes(text, target_entity or "", target_relation):
                text_negative.append(item)
            elif _text_supports(text, target_entity or "", target_relation):
                text_positive.append(item)

        # ---- 冲突判定 ----
        # conflict 的定义：存在"支持结论"的证据 且 存在"反对结论"的证据
        #   graph 正向  vs  文本负向   → conflict（如 graph"负责" + text"不再负责"）
        #   graph 无正向 + 文本负向     → conflict（text 明确说当前不负责，与"是否负责"矛盾面）
        graph_supports = bool(positive)
        text_supports_conclusion = bool(text_positive)
        text_opposes_conclusion = bool(text_negative)

        conflict_evidence: List[Dict] = []
        if text_opposes_conclusion and (graph_supports or text_supports_conclusion):
            # 正反证据同时存在 → conflict
            conflict_evidence = [
                {"side": "support", "source": "graph" if graph_supports else "text",
                 "evidence": (positive or text_positive)[0]},
                {"side": "oppose", "source": "text", "evidence": text_negative[0]},
            ]

        evidence_detail = (
            [{"role": "positive", **p} for p in positive]
            + [{"role": "partial", **p} for p in partial]
            + [{"role": "text_support", **t} for t in text_positive]
            + [{"role": "text_oppose", **t} for t in text_negative]
        )

        if conflict_evidence:
            return {
                "evidence_status": "conflict",
                "next_action": "conflict",
                "conflict_status": "detected",
                "conflict_evidence": conflict_evidence,
                "target_relation": target_relation,
                "target_entity": target_entity,
                "evidence_detail": evidence_detail,
            }

        # 任一来源（graph 或 text）明确支持结论 → sufficient
        if graph_supports or text_supports_conclusion:
            return self._sufficient(query, target_relation, target_entity,
                                    evidence_detail=evidence_detail)

        # 没有任何正向证据 → insufficient（无论有没有 partial）
        return self._insufficient(query, target_relation, target_entity,
                                  targeted_retried, evidence_detail=evidence_detail)

    # ----------------------------------------------------------
    # 结果构造
    # ----------------------------------------------------------
    def _sufficient(self, query, rel, entity, evidence_detail=None):
        return {
            "evidence_status": "sufficient",
            "next_action": "answer",
            "conflict_status": "no_conflict",
            "target_relation": rel,
            "target_entity": entity,
            "evidence_detail": evidence_detail or [],
        }

    def _insufficient(self, query, rel, entity, targeted_retried, evidence_detail=None):
        # 已重试过一次仍不足 → 拒答；否则进 targeted graph retrieval
        if targeted_retried:
            return {
                "evidence_status": "insufficient",
                "next_action": "refuse",
                "conflict_status": "no_conflict",
                "target_relation": rel,
                "target_entity": entity,
                "evidence_detail": evidence_detail or [],
            }
        return {
            "evidence_status": "insufficient",
            "next_action": "targeted_graph",
            "conflict_status": "no_conflict",
            "target_relation": rel,
            "target_entity": entity,
            "evidence_detail": evidence_detail or [],
        }
