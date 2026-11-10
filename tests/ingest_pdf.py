"""tests/ingest_pdf.py —— 真实 PDF 制度文档 → 条款级切块 → 抽取入库。

为什么单独写：
    原来的语料是我手写的 Markdown，切块天然干净。真实 PDF 有四个坑：
      ① 页脚页码（"1 / 16"）混进正文，成为噪声块
      ② 中文字符之间被插入空格（"第一章 总 则"），正则匹配会失败
      ③ 一条制度跨页断开，切出来是两半
      ④ 只有页码，没有章节信息——检索命中后无法溯源到"第几章第几条"

    这个脚本专门处理这四件事，并把「章/条/页码」写进每块文本与元数据。

设计要点（能讲给面试官听的部分）：
    · **先拼后切**：把整份 PDF 的文本连成一条流再按「第X条」切，跨页条款自动接上；
      页码通过字符偏移反查，不影响切分。
    · **块内自带上下文**：每块以「《文件名》第X章 第Y条：」开头。
      这样 embedding 编码的是"带身份的条款"，而不是一段没有主语的文字。
    · **一实体一组事实的反向选择**：制度文档天然以"条"为单位，每条就是一条规则，
      所以按条切比按段落切更贴合语义。

用法：
    python tests/ingest_pdf.py --dry-run                 # 只解析，打印统计与前几条样本
    python tests/ingest_pdf.py --reset                   # 清库后灌 PDF（图 + 向量）
    python tests/ingest_pdf.py --reset --skip-milvus      # 只建图
    python tests/ingest_pdf.py --dir data/pdf --only 1.pdf
"""

import argparse
import bisect
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pymupdf  # noqa: E402

# 页脚/页眉：整行只有数字，或 "1 / 16" 这种
PAGE_NO = re.compile(r"^\s*\d+\s*(?:/\s*\d+)?\s*$")
CHAPTER = re.compile(r"第[一二三四五六七八九十百]+章")
ARTICLE = re.compile(r"第[一二三四五六七八九十百零〇\d]+条")
CJK_GAP = re.compile(r"(?<=[\u4e00-\u9fa5，。；：、）】])\s+(?=[\u4e00-\u9fa5（【])")


def clean_text(raw: str) -> str:
    """去页码、修中文之间的异常空格。"""
    lines = [ln for ln in raw.splitlines() if not PAGE_NO.match(ln)]
    text = "\n".join(lines)
    text = CJK_GAP.sub("", text)          # 中文之间的空格直接删掉
    text = re.sub(r"[ \t]+", " ", text)   # 其余连续空白压成一个空格
    return text


def table_to_text(page) -> str:
    """把页面里的表格转成可检索的句子。

    为什么必须单独做：PyMuPDF 的 get_text() 是"按版面顺序抽文字"，
    两列表格会被拉平成一串，**值甚至会丢**。实测年报的员工表：
        "母公司在职员工的数量主要子公司在职员工的数量 / 1,966"
    —— 母公司那个 45 直接没了，销售/技术/财务/行政的人数也全丢。
    而表格恰恰是企业文档里数字最集中的地方。

    输出统一成「【表】行名：值；值」——不猜表头。
    早先版本把第一行当表头，结果第一行数据被吃掉、值还对错了列（45=1966）。
    键值表里最重要的是"标签 + 数字"，表头名丢了不影响回答。
    """
    out = []
    try:
        tabs = page.find_tables()
    except Exception:  # noqa: BLE001
        return ""

    def clean(cell) -> str:
        s = (cell or "").replace("\n", " ").strip()
        s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)     # "1 966" → "1966"
        s = re.sub(r"(?<=\d)\s*([,.])\s*(?=\d)", r"\1", s)  # "4 078 . 68" → "4078.68"
        return s.strip(" ,，;；")

    for t in tabs.tables:
        for row in t.extract():
            cells = [clean(c) for c in row]
            cells = [c for c in cells if c]
            if len(cells) >= 2:
                out.append("【表】" + cells[0] + "：" + "；".join(cells[1:]))
            elif cells:
                out.append("【表】" + cells[0])
    return "\n".join(out)


def extract_pdf(path: Path, pages: set[int] | None = None) -> tuple[str, list[int], list[int]]:
    """返回 (全文, 每段起始字符偏移, 对应真实页码)。

    pages: 只解析这些页（1-based）。长文档（比如 258 页的年报）里往往只有几页
    是我们要的（"员工情况""利润分配"），整本灌进去既慢又把无关内容带进检索。
    """
    doc = pymupdf.open(str(path))
    parts: list[str] = []
    offsets: list[int] = []
    page_nums: list[int] = []
    pos = 0
    for i, page in enumerate(doc, 1):
        if pages and i not in pages:
            continue
        offsets.append(pos)
        page_nums.append(i)          # 记录真实页码，别用"第几个选中页"
        t = clean_text(page.get_text())
        extra = table_to_text(page)  # 表格单独抽，补回被拉平丢失的数字
        if extra:
            t = t + "\n" + extra
        parts.append(t)
        pos += len(t) + 1                 # +1 是拼接用的换行
    return "\n".join(parts), offsets, page_nums


def title_of(path: Path, text: str) -> str:
    """标题取文件名（去掉 "1_" 这类序号）。

    不从正文取标题：PDF 里公司名、制度名、副标题经常在同一行，
    取出来是一条长串，塞进块前缀会污染 embedding，而且还会被截断成
    "…第一章总则第" 这种半句。文件名反而是最干净的制度名。
    """
    # 只去掉「9_」这种序号前缀。注意不能写成 ^[\d_\-\s]+ ——
    # 那会把文件名开头的年份也吃掉（"9_2025年年度报告" → "年年度报告"）。
    name = re.sub(r"^\d+[_\-\s]+", "", path.stem).strip()
    return name or path.stem


def _page_of(offsets: list[int], page_nums: list[int] | None, pos: int) -> int:
    """字符偏移 → 真实页码。"""
    idx = max(0, bisect.bisect_right(offsets, pos) - 1)
    if page_nums and idx < len(page_nums):
        return page_nums[idx]
    return idx + 1


def split_clauses(text: str, offsets: list[int], title: str,
                  max_len: int = 900, page_nums: list[int] | None = None):
    """按「第X条」切块，每块带章节前缀。返回 [{text, chapter, article, page}]。"""
    chapters = [(m.start(), m.group()) for m in CHAPTER.finditer(text)]
    articles = list(ARTICLE.finditer(text))

    # 不是所有 PDF 都按「第X条」写。年报、摘要、ESG 报告用的是
    # 「七、报告期末员工情况」「(一) 员工情况」这种小节编号。
    # 没有条款就退化成按小节切，否则会切出 0 块。
    if not articles:
        return _split_by_section(text, offsets, title, max_len, page_nums)

    chunks = []
    for i, m in enumerate(articles):
        start = m.start()
        end = articles[i + 1].start() if i + 1 < len(articles) else len(text)
        # 正文从条号**之后**开始，否则块里会出现"第二条：第二条…"的重复
        body = text[m.end():end].strip().lstrip("：:、 ")
        if len(body) < 10:
            continue

        # 该条属于哪一章：取它之前最近的章标题
        chapter = ""
        for cpos, cname in chapters:
            if cpos <= start:
                chapter = cname
            else:
                break

        # 太长就按句号再切，避免单块过大稀释检索
        pieces = [body] if len(body) <= max_len else _split_long(body, max_len)

        for k, piece in enumerate(pieces):
            head = f"《{title}》{chapter} {m.group()}：" if k == 0 else f"《{title}》{chapter} {m.group()}（续）："
            chunks.append({
                "text": head + piece,
                "chapter": chapter,
                "article": m.group(),
                "page": _page_of(offsets, page_nums, start),   # 真实页码
            })
    return chunks


def _split_by_section(text: str, offsets: list[int], title: str,
                      max_len: int = 900, page_nums: list[int] | None = None):
    """没有「第X条」的文档（年报/摘要/ESG）按小节标题切块。

    小节标题形如「七、报告期末母公司和主要子公司的员工情况」「(一) 员工情况」。
    每块带上所属的一级小节名，保证块内自带上下文。
    """
    # 小节标题不一定在行首：PDF 抽出来的文本常常是"…无异议。七、报告期末员工情况"，
    # 所以句号/分号之后出现「一、二、…」也要认作小节边界。
    SECTION = re.compile(r"(?:^|[\n。；])\s*([一二三四五六七八九十]+、[^\n]{2,40})")
    marks = [(m.start(1), m.group(1).strip()) for m in SECTION.finditer(text)]

    chunks = []

    # 第一个小节标记之前的内容不能丢——年报里"员工情况"的正文
    # 有时就落在前一个小节标题之后、这里之前。
    if marks and marks[0][0] > 20:
        head = f"《{title}》前言："
        for k, piece in enumerate(_split_long(text[: marks[0][0]].strip(), max_len)):
            chunks.append({
                "text": head + piece,
                "chapter": "前言",
                "article": f"§0-{k + 1}",
                "page": _page_of(offsets, page_nums, 0),
            })

    bounds = [pos for pos, _ in marks] or [0]
    for idx, start in enumerate(bounds):
        end = bounds[idx + 1] if idx + 1 < len(bounds) else len(text)
        sec_name = marks[idx][1] if idx < len(marks) else ""
        body = text[start:end].strip()
        if len(body) < 20:
            continue
        for k, piece in enumerate(_split_long(body, max_len)):
            head = f"《{title}》{sec_name}：" if sec_name else f"《{title}》："
            chunks.append({
                "text": head + piece,
                "chapter": sec_name,
                "article": f"§{idx + 1}" + (f"-{k + 1}" if k else ""),
                "page": _page_of(offsets, page_nums, start),
            })
    return chunks


def _split_long(body: str, max_len: int) -> list[str]:
    parts, buf = [], ""
    for sent in re.split(r"(?<=[。；])", body):
        if len(buf) + len(sent) > max_len and buf:
            parts.append(buf)
            buf = sent
        else:
            buf += sent
    if buf:
        parts.append(buf)
    return parts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/pdf")
    parser.add_argument("--only", default="", help="只处理指定文件名，逗号分隔")
    parser.add_argument("--pages", default="", help="只解析指定页（1-based），如 51-53 或 51,52")
    parser.add_argument("--reset", action="store_true", help="先清空 Neo4j（演示库专用）")
    parser.add_argument("--skip-milvus", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="只解析不灌库")
    parser.add_argument("--show", type=int, default=3, help="dry-run 时打印几块样本")
    args = parser.parse_args()

    pages: set[int] | None = None
    if args.pages:
        pages = set()
        for part in args.pages.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                pages.update(range(int(a), int(b) + 1))
            elif part:
                pages.add(int(part))

    d = Path(args.dir)
    # Windows 上 glob 大小写不敏感，同时用 "*.pdf" 和 "*.PDF" 会把同一批文件收两遍。
    # 统一按后缀小写过滤 + 去重。
    files = sorted({p.resolve() for p in d.iterdir() if p.suffix.lower() == ".pdf"})
    files = [Path(p) for p in files]
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        files = [f for f in files if f.name in wanted]
    if not files:
        print(f"[FAIL] {d} 下没有 PDF")
        return 1

    # ---------- 解析 ----------
    all_chunks = []
    print("=" * 70)
    print(f"解析 {len(files)} 份 PDF")
    print("=" * 70)
    for f in files:
        text, offsets, page_nums = extract_pdf(f, pages)
        title = title_of(f, text)
        chunks = split_clauses(text, offsets, title, page_nums=page_nums)
        all_chunks.append((f.stem, chunks))
        avg = sum(len(c["text"]) for c in chunks) / max(1, len(chunks))
        print(f"  {f.name:<28} 字符 {len(text):>6}  条款 {len(chunks):>3}  平均块长 {avg:>5.0f}")

    total = sum(len(c) for _, c in all_chunks)
    print(f"\n合计 {total} 块")

    if args.dry_run:
        for stem, chunks in all_chunks:
            print(f"\n--- {stem} 前 {args.show} 块 ---")
            for c in chunks[: args.show]:
                print(f"[{c['article']} p{c['page']}] {c['text'][:150]}")
        print("\n(dry-run：未灌库)")
        return 0

    # ---------- 灌库 ----------
    from src.config import Config
    from src.database.neo4j_client import Neo4jClient
    from src.extractors.triple_extractor import TripleExtractor
    from src.database.local_vector_store import LocalVectorStore

    if not Config.DEEPSEEK_API_KEY:
        print("[FAIL] .env 缺少 DEEPSEEK_API_KEY")
        return 1

    neo4j = Neo4jClient(uri=Config.NEO4J_URI, user=Config.NEO4J_USER, password=Config.NEO4J_PASSWORD)
    neo4j.connect()
    neo4j.create_indexes()
    if args.reset:
        with neo4j.driver.session() as s:
            before = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            s.run("MATCH (n) DETACH DELETE n")
        print(f"\n[reset] 已清空 Neo4j（删除 {before} 个节点）\n")

    extractor = TripleExtractor(
        api_key=Config.DEEPSEEK_API_KEY,
        base_url=Config.DEEPSEEK_BASE_URL,
        model_name=Config.DEEPSEEK_MODEL,
        temperature=0.1,
        confidence_threshold=0.7,
    )

    store = None
    if not args.skip_milvus:
        store = LocalVectorStore()
        store.connect()
        store.create_collection("document_chunks", drop_existing=args.reset)
        store.load_if_needed()

    offset = 0
    for stem, chunks in all_chunks:
        triples = []
        for i, c in enumerate(chunks, 1):
            try:
                triples.extend(extractor.extract(c["text"], chunk_id=f"{stem}#{c['article']}").triples)
            except Exception as e:  # noqa: BLE001
                print(f"  {stem} {c['article']} 抽取失败：{type(e).__name__}")
        written = neo4j.batch_write(triples).get("success", 0) if triples else 0
        print(f"  {stem}: {len(triples)} 条三元组，写入 {written}")

        if store is not None:
            n = store.insert_chunks(stem, [c["text"] for c in chunks], chunk_offset=offset)
            offset += n
            print(f"    向量写入 {n} 块")

    print("\n" + "=" * 70)
    with neo4j.driver.session() as s:
        nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    print(f"完成：Neo4j {nodes} 节点 / {rels} 关系" + (f"，向量累计 {offset} 块" if store else ""))
    print("=" * 70)

    neo4j.close()
    if store is not None:
        store.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
