"""tests/test_static.py —— 不依赖任何外部服务的静态检查（可放进 CI）。

为什么单独一套：
    完整评测要 Neo4j + 向量库 + LLM API，跑一次要钱、要时间，不适合每次提交都跑。
    但"schema 被改坏、判分器写错、数据集格式崩了"这类问题，**不需要任何外部服务就能查出来**。
    所以把评测分成两层：CI 跑这层（秒级、免费），完整评测放本地手动跑。

覆盖五件事：
    1. datasets/*.jsonl 每行都能解析，必填字段齐全
    2. schema 枚举数量与取值唯一性（改枚举时最容易撞车的地方）
    3. 抽取 prompt 能构建，且关键定义还在（防止有人删了定义段）
    4. 判分器语义：AND / OR / none
    5. 拒答识别 + 文档切段

用法：
    python tests/test_static.py
退出码 0 = 全过。
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ok = 0
fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL  {name}  {detail}")


def load_eval_mod():
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_eval", ROOT / "tests" / "run_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    # ---------- 1. 数据集 ----------
    print("[1] 数据集格式")
    files = sorted((ROOT / "datasets").glob("*.jsonl"))
    check("data集目录非空", bool(files), "datasets/ 下没有 jsonl")
    for f in files:
        raw = [l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        bad = 0
        for i, line in enumerate(raw, 1):
            try:
                case = json.loads(line)
            except json.JSONDecodeError:
                print(f"      {f.name} 第 {i} 行不是合法 JSON")
                bad += 1
                continue
            if not {"id", "query"} <= set(case):
                print(f"      {f.name} 第 {i} 行缺 id/query")
                bad += 1
        check(f"{f.name} 全部可解析", bad == 0, f"{bad} 行有问题")
        print(f"      {f.name}: {len(raw)} 题")

    # ---------- 2. schema ----------
    print("[2] schema 枚举")
    from src.schemas.graph_schema import EntityType, RelationType, Triple

    e_vals = [e.value for e in EntityType]
    r_vals = [r.value for r in RelationType]
    check("实体类型无重复", len(e_vals) == len(set(e_vals)), str(len(e_vals)))
    check("关系类型无重复", len(r_vals) == len(set(r_vals)), str(len(r_vals)))
    check("实体类型 ≥ 14", len(e_vals) >= 14, str(len(e_vals)))
    check("关系类型 ≥ 15", len(r_vals) >= 15, str(len(r_vals)))
    # 关键类型必须还在（防止误删）
    for name in ["条款", "阈值", "流程", "角色", "系统"]:
        check(f"实体含「{name}」", name in e_vals)
    for name in ["审批", "适用于", "依据", "负责", "访问"]:
        check(f"关系含「{name}」", name in r_vals)
    check("Triple 可实例化", Triple is not None)

    # ---------- 3. 抽取 prompt ----------
    print("[3] 抽取 prompt")
    from src.extractors.triple_extractor import TripleExtractor

    ex = TripleExtractor(api_key="static-check-key")   # 只构建 prompt，不发请求
    prompt = ex._build_system_prompt()
    check("prompt 非空", len(prompt) > 500, str(len(prompt)))
    for kw in ["条款", "阈值", "审批", "适用于", "依据"]:
        check(f"prompt 提到「{kw}」", kw in prompt)
    check("prompt 含示例 6（条款再化）", "示例6" in prompt or "示例 6" in prompt)
    # 注意：不能简单查 "优先抽取" —— prompt 里有正面规则「优先抽取句子的主干关系」。
    # 要查的是那句被删掉的项目域专属指令的**具体措辞**。
    check("不再有项目域专属的抽取指令", "人员、技术栈" not in prompt)

    # ---------- 4. 判分器 ----------
    print("[4] 判分器")
    m = load_eval_mod()
    case = {"all": [["甲"], ["乙", "丙"]], "none": ["丁"]}
    r = m.score("这里提到甲和丙", case)
    check("AND + OR 命中", r["ok"] is True, str(r))
    r = m.score("只提到甲", case)
    check("缺 OR 组判错", r["ok"] is False and "乙|丙" in r["missing"], str(r))
    r = m.score("甲丙丁", case)
    check("出现禁止词判错", r["ok"] is False and "丁" in r["forbidden"], str(r))

    # ---------- 5. 拒答识别 + 切段 ----------
    print("[5] 拒答识别与切段")
    for text in ["当前知识库中的信息不足以回答", "根据提供的信息，无法确定", "知识库中没有相关规定"]:
        check(f"识别拒答：{text[:8]}", m.is_refusal(text))
    check("正常回答不算拒答", not m.is_refusal("由财务总监审批"))

    import importlib.util as iu

    spec = iu.spec_from_file_location("ingest", ROOT / "tests" / "ingest_simco.py")
    ing = iu.module_from_spec(spec)
    spec.loader.exec_module(ing)
    md = "# 标题\n\n- 文档编号：X\n\n第 1 条：内容一。\n第 2 条：内容二。\n\n## 小节\n\n正文。\n"
    blocks = ing.split_paragraphs(md)
    check("切段：标题被跳过", all(not b.startswith("# ") for b in blocks), str(blocks))
    check("切段：元数据保留", any("文档元数据" in b for b in blocks), str(blocks))
    check("切段：条款按空行合并成一段", any("第 1 条" in b and "第 2 条" in b for b in blocks), str(blocks))

    print("\n" + "=" * 54)
    print(f"结果: {ok} 通过, {fail} 失败")
    print("=" * 54)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
