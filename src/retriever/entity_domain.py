"""实体 → 业务域 的归属推断（图谱层的域隔离）。

为什么需要：
    向量/BM25 两路可以按 doc_id 过滤，但**图谱那一路没过滤**——实体是从整张图里查的，
    两套语料的实体混在一张图里。实测后果：simco 域问"公司员工多少人"，
    答了 pdf 域年报里的 2,137 人（应为澜海科技的 1,200）。

为什么用"反推"而不是重灌：
    图里没有 source_chunk 属性（关系只存 relation_type/confidence），没法直接回填。
    重灌图要 400+ 次抽取调用；而**块文本里有原文**，只要看某个实体名出现在哪个域的
    块里，就能反推它的域归属——零 LLM 调用，几秒钟算完。

局限（必须写清楚）：
    · 这是**近似**：实体名恰好在另一域的文本里被提及时会被判成跨域（例如"总经理"两边都有）。
      跨域实体本来就应该两个域都放行，所以这类"误判"反而是对的。
    · 长期方案是**写入时就带 domain 属性**（`(name, domain)` 联合主键或实体属性），
      这样不用反推，也不怕文本里偶然提及。
"""

from __future__ import annotations

import logging
from typing import Iterable

logger = logging.getLogger(__name__)


class EntityDomainMap:
    """实体名 → 所属域集合。未收录的实体不参与过滤（放行）。"""

    def __init__(self):
        self._map: dict[str, set[str]] = {}
        self._titles: dict[str, list[str]] = {}
        self._built = False

    def build(self, entity_names: Iterable[str], chunks_by_domain: dict[str, list[str]],
              titles_by_domain: dict[str, list[str]] | None = None,
              min_len: int = 2) -> "EntityDomainMap":
        """用各域的块文本反推实体归属。

        chunks_by_domain: {"pdf": [块文本...], "simco": [...]}
        titles_by_domain: {"pdf": ["对外担保管理制度", ...]} —— 兜底用。

        为什么需要 titles 兜底：抽取出来的实体名常常是**拼接出来的**
        （"对外担保管理制度第四条"），原文里并没有这一串字，纯文本匹配找不到，
        结果这类实体被判成"无域信息"而被放行——串味就漏在这。
        用"实体名包含某域文档标题"作为兜底，能把这类实体归位。
        """
        blobs = {d: "\n".join(texts) for d, texts in chunks_by_domain.items() if texts}
        self._titles = {d: [t for t in ts if len(t) >= 4]
                        for d, ts in (titles_by_domain or {}).items()}
        self._all_domains = set(blobs) or set(self._titles)
        names = [n for n in entity_names if n and len(n) >= min_len]
        for name in names:
            domains = {d for d, blob in blobs.items() if name in blob}
            if domains:
                self._map[name] = domains
        # 通用实体："公司""股东""董事会"这类在**所有域**都出现的词，没有区分力。
        # 不排除它们的话，"公司 —保密义务→ …"这种 pdf 三元组会因为主语"公司"
        # 在 simco 域也出现而被放行，串味就漏在这。
        self._generic = {n for n, ds in self._map.items() if ds >= self._all_domains}
        self._built = True
        logger.info(
            f"实体域归属构建完成：{len(self._map)}/{len(names)} 个实体有域信息，"
            f"其中 {len(self._generic)} 个判为跨域通用（不参与判断）"
        )
        return self

    def domains_of(self, name: str) -> set[str] | None:
        """该实体的域集合；通用实体或未知实体返回 None（= 没有区分信号）。"""
        if not name:
            return None
        if name in self._generic:
            return None
        ds = self._map.get(name)
        if not ds and self._titles:
            ds = {d for d, ts in self._titles.items() if any(t in name for t in ts)}
        return set(ds) if ds else None

    def keep(self, name: str, domains: Iterable[str] | None) -> bool:
        """单实体判断（保留给调用方做粗筛）。"""
        if not domains:
            return True
        ds = self.domains_of(name)
        return True if not ds else bool(ds & set(domains))

    def triple_keep(self, subject: str, obj: str, domains: Iterable[str] | None) -> bool:
        """三元组级判断：用**更具体的那一端**决定去留。

        为什么取"域集合更小"的一端：
            "对外担保管理制度第四条 —审批→ 公司" 里，主语只在 pdf 出现（{pdf}），
            宾语"公司"在 pdf 和 simco 都出现（{pdf,simco}）。若取并集就会因为
            "公司 也在 simco" 而放行，串味就是这么漏的。
            取交集或取更小的那个集合，才能反映出"这条边其实是 pdf 独有的"。
        """
        if not domains:
            return True
        allowed = set(domains)
        s, o = self.domains_of(subject), self.domains_of(obj)
        cands = [x for x in (s, o) if x]
        if not cands:
            return True                      # 两端都没信号：宁可放行，不误杀
        best = min(cands, key=len)           # 更具体的那个端点
        return bool(best & allowed)

    @property
    def built(self) -> bool:
        return self._built
