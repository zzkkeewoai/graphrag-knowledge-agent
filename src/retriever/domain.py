"""域隔离：让同一份知识库里共存多个业务域，检索时不串味。

为什么需要：
    手写制度（simco）、项目域（project）、上市公司真实 PDF（pdf）三套语料
    主题相近（都讲"审批""合同""金额"），混在一张库里检索必然互相挤占。
    云从那类岗的 JD 里明确写「多租户 / 多知识库隔离」，本质就是这件事。

怎么判断一条块属于哪个域：
    不额外存元数据（那要重灌全部语料），而是用**确定性的命名规则**从 doc_id 推：
        "1_对外担保管理制度" → pdf        （PDF 都有数字前缀）
        "projects" / "people" → project
        其它（org / systems / approval_matrix …）→ simco
    规则集中在这一个文件里，换语料只改这里。
"""

from __future__ import annotations

import re
from typing import Iterable, List

_PDF_PREFIX = re.compile(r"^\d+[_\-\s]")
_PROJECT_DOCS = {"projects", "people"}


def domain_of(doc_id: str) -> str:
    """从一个块的 doc_id 推断它属于哪个域。"""
    if not doc_id:
        return "unknown"
    if _PDF_PREFIX.match(doc_id):
        return "pdf"
    if doc_id in _PROJECT_DOCS:
        return "project"
    return "simco"


def parse_domains(spec: str | None) -> List[str] | None:
    """把 "simco,project" 解析成列表；空/None 表示不过滤。"""
    if not spec:
        return None
    out = [x.strip() for x in spec.split(",") if x.strip()]
    return out or None


def keep(doc_id: str, domains: Iterable[str] | None) -> bool:
    """该块是否属于允许的域。"""
    if not domains:
        return True
    return domain_of(doc_id) in set(domains)
