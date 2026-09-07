"""
conflict_resolver.py —— 冲突解决器（v2.2 新增）

解决的问题：
  Graph 与 Vector（文本）证据互相矛盾时，不能简单"信 Graph"或"信 Vector"，
  也不能把判断丢给 LLM。本模块用确定性规则裁决：

  冲突示例：
    Graph:  张三 --负责--> 项目A        （支持"负责"）
    Vector: "张三已经不再负责项目A，目前由李四负责。"  （反对"负责"）

  裁决优先级（第一版，全部确定性规则，不调 LLM）：
    1. 时间信息：有明确"现在/目前/截至..." 的证据视为最新有效 → 优先
    2. 来源 + 时间戳/置信度：metadata 里存在 timestamp / confidence 则利用
    3. 结构化 vs 非结构化：不默认 Graph 优先（Graph 可能是旧数据）
    4. 无法可靠裁决 → unresolved → refuse（不让 LLM 编答案）

关键原则：
  conflict ≠ insufficient
    insufficient：没有足够证据支持结论（缺正向）
    conflict：存在两个支持相反结论的证据（正反都有）
"""
import logging
from typing import Dict, Any, List, Optional

from src.agents.evidence_validator import _text_has_time_signal

logger = logging.getLogger(__name__)


class ConflictResolver:
    """确定性冲突裁决器。"""

    def resolve(
        self,
        query: str,
        conflict_evidence: List[Dict],
        evidence_detail: List[Dict],
    ) -> Dict[str, Any]:
        """裁决冲突证据。

        Args:
            query: 用户问题
            conflict_evidence: [{"side": "support"/"oppose", "source": "graph"/"text",
                                "evidence": {...}}, ...]（由 EvidenceValidator 提供）
            evidence_detail: 全部证据（含 timestamp/confidence 等元信息）

        Returns:
            {
              "conflict_status": "resolved" | "unresolved",
              "next_action": "answer" | "refuse",
              "conflict_resolution": str,   # 裁决说明（供 answer 引用）
              "winner_side": "support" | "oppose" | None,
            }
        """
        support = next((e for e in conflict_evidence if e.get("side") == "support"), None)
        oppose = next((e for e in conflict_evidence if e.get("side") == "oppose"), None)

        if not support or not oppose:
            return self._unresolved(query, "缺少支持或反对证据，无法裁决")

        sup_ev = support.get("evidence", {})
        opp_ev = oppose.get("evidence", {})

        # ---------- 优先级 1：时间信息 ----------
        # 反对侧（文本）通常携带"现在/目前/不再"等最新信号；
        # 支持侧若来自 Graph，可能是历史入库数据。
        sup_time = sup_ev.get("timestamp") or sup_ev.get("date")
        opp_time = opp_ev.get("timestamp") or opp_ev.get("date")
        opp_text = opp_ev.get("text", "")

        # 1a. 双方都有显式 timestamp → 取最新
        if sup_time and opp_time:
            try:
                if str(sup_time) >= str(opp_time):
                    return self._resolved(query, "support", f"支持侧时间戳 {sup_time} 更新", sup_ev)
                return self._resolved(query, "oppose", f"反对侧时间戳 {opp_time} 更新", opp_ev)
            except Exception as e:
                # 时间戳格式非法 → 降级到下一优先级（不吞异常，记录原因）
                logger.warning(f"[ConflictResolver] 时间戳比较失败，降级: {e}")

        # 1b. 反对侧文本含"现在/目前/不再"等时间信号 → 视为最新状态
        if _text_has_time_signal(opp_text):
            return self._resolved(
                query, "oppose",
                "文本证据明确描述当前状态（含时间信号），优先于历史图谱数据",
                opp_ev,
            )

        # ---------- 优先级 2：来源可靠性 ----------
        # 若双方都有 confidence 且差距明显 → 取高者
        sup_conf = sup_ev.get("confidence")
        opp_conf = opp_ev.get("confidence")
        if sup_conf is not None and opp_conf is not None:
            try:
                if float(sup_conf) >= float(opp_conf) + 0.15:
                    return self._resolved(query, "support",
                                          f"支持侧置信度 {sup_conf} 显著更高", sup_ev)
                if float(opp_conf) >= float(sup_conf) + 0.15:
                    return self._resolved(query, "oppose",
                                          f"反对侧置信度 {opp_conf} 显著更高", opp_ev)
            except Exception as e:
                # 置信度格式非法 → 降级到下一优先级（记录原因）
                logger.warning(f"[ConflictResolver] 置信度比较失败，降级: {e}")

        # 双方都有显式 source（如入库时间来源）时，文本类通常更新
        # 但此处不默认 Graph 优先——无可靠信号则 unresolved

        # ---------- 无法裁决 → unresolved ----------
        return self._unresolved(query, "支持与反对证据均无时间/置信度差异，无法可靠裁决")

    # ----------------------------------------------------------
    def _resolved(self, query, winner_side, reason, winner_ev):
        logger.info(f"[ConflictResolver] resolved: {winner_side} — {reason}")
        return {
            "conflict_status": "resolved",
            "next_action": "answer",
            "conflict_resolution": reason,
            "winner_side": winner_side,
            "winner_evidence": winner_ev,
        }

    def _unresolved(self, query, reason):
        logger.info(f"[ConflictResolver] unresolved: {reason}")
        return {
            "conflict_status": "unresolved",
            "next_action": "refuse",
            "conflict_resolution": reason,
            "winner_side": None,
            "winner_evidence": None,
        }
