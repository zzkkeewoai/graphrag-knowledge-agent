"""
Ragas 评估脚本
基于标准测试集验证 GraphRAG 系统的检索+生成质量

评估维度：
- Context Precision: 检索到的上下文中有多少是真正相关的
- Context Recall: 真正相关的上下文中检索到了多少
- Faithfulness: 生成的答案是否基于检索上下文（而非编造）
- Answer Relevancy: 答案与问题的相关程度
- Answer Correctness: 答案的事实准确性

使用方法：
  cd graphrag_system
  python -m tests.test_ragas_eval
"""
import json
import sys
import os
import time
import logging
from typing import Dict, Any, List
from dataclasses import dataclass, field

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """单条评估结果"""
    query_id: str
    query: str
    category: str
    difficulty: str
    answer: str
    latency_ms: float
    # 评估分数
    context_precision: float = 0.0    # 检索上下文精准率
    context_recall: float = 0.0       # 检索上下文召回率 (Hit@K)
    faithfulness: float = 0.0         # 生成忠实度
    answer_relevancy: float = 0.0     # 答案相关性
    answer_correctness: float = 0.0   # 答案正确性
    # 详细信息
    retrieved_docs: List[str] = field(default_factory=list)
    error: str = ""


class GraphRAGEvaluator:
    """
    GraphRAG 系统评估器

    评估流程：
    1. 加载标准测试集
    2. 逐条运行系统获取答案
    3. 计算 Ragas 兼容的评估指标
    4. 输出评估报告

    支持两种模式：
    - 快速评估（默认）：使用规则 + 关键词匹配
    - 完整评估：调用 Ragas 库（需 pip install ragas）
    """

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.results: List[EvalResult] = []

    def load_queries(self, path: str = None) -> List[Dict]:
        """加载标准测试集"""
        if path is None:
            path = os.path.join(
                os.path.dirname(__file__), '..', 'data', 'eval_queries.json'
            )
        with open(path, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        logger.info(f"加载 {len(queries)} 条测试查询")
        return queries

    def run_evaluation(self, queries: List[Dict] = None, use_ragas: bool = False) -> List[EvalResult]:
        """运行完整评估"""
        if queries is None:
            queries = self.load_queries()

        logger.info(f"开始评估 {len(queries)} 条查询...")
        self.results = []

        for i, q in enumerate(queries, 1):
            logger.info(f"[{i}/{len(queries)}] {q['category']}: {q['query'][:50]}...")

            result = EvalResult(
                query_id=q['id'],
                query=q['query'],
                category=q['category'],
                difficulty=q['difficulty'],
                answer="",
                latency_ms=0.0
            )

            try:
                start = time.time()
                if self.orchestrator:
                    response = self.orchestrator.process(q['query'])
                    result.answer = response.get('answer', '')
                    result.retrieved_docs = [
                        d.get('text', '')[:100]
                        for d in response.get('documents', [])
                    ]
                else:
                    result.answer = "[模拟模式] 根据检索到的上下文，"
                    result.answer += f"相关的实体和关系已找到，涉及 {len(q.get('relevant_docs', []))} 个文档。"

                result.latency_ms = (time.time() - start) * 1000

                # 计算评估指标
                if use_ragas:
                    self._compute_ragas_metrics(result, q)
                else:
                    self._compute_simple_metrics(result, q)

            except Exception as e:
                result.error = str(e)
                logger.error(f"评估失败 [{q['id']}]: {e}")

            self.results.append(result)

        return self.results

    def _compute_simple_metrics(self, result: EvalResult, query: Dict):
        """
        简化版评估指标计算
        基于关键词匹配 + 规则

        生产环境可替换为 Ragas 库：
          from ragas import evaluate
          from ragas.metrics import context_precision, faithfulness, answer_relevancy
        """
        expected = query.get('expected_answer', '')
        answer = result.answer
        relevant_docs = set(query.get('relevant_docs', []))

        # Context Precision: 检索到的文档中有多少是相关的
        retrieved_ids = set()
        for doc_text in result.retrieved_docs:
            for rel_doc in relevant_docs:
                if rel_doc.lower() in doc_text.lower():
                    retrieved_ids.add(rel_doc)
        if result.retrieved_docs:
            result.context_precision = len(retrieved_ids) / len(result.retrieved_docs)

        # Context Recall (Hit@K): 相关文档中有多少被检索到
        if relevant_docs:
            result.context_recall = len(retrieved_ids) / len(relevant_docs)
        else:
            result.context_recall = 1.0

        # Faithfulness: 答案中的事实是否来自检索上下文
        # 简化版：检查答案中的关键词是否在检索文档中出现
        if answer and result.retrieved_docs:
            context_text = ' '.join(result.retrieved_docs).lower()
            answer_words = set(answer.lower().split())
            context_words = set(context_text.split())
            if answer_words:
                overlap = len(answer_words & context_words) / len(answer_words)
                result.faithfulness = min(overlap * 2, 1.0)  # 放大因子
            else:
                result.faithfulness = 0.0

        # Answer Relevancy: 答案与问题的相关性
        query_words = set(query['query'].lower())
        if answer:
            answer_words = set(answer.lower().split())
            if answer_words:
                result.answer_relevancy = len(query_words & answer_words) / len(query_words)

        # Answer Correctness: 答案与期望答案的事实一致性
        if expected:
            expected_words = set(expected.lower().split())
            answer_words_lower = set(answer.lower().split())
            if expected_words:
                recall = len(expected_words & answer_words_lower) / len(expected_words)
                precision = len(expected_words & answer_words_lower) / max(len(answer_words_lower), 1)
                if recall + precision > 0:
                    result.answer_correctness = 2 * recall * precision / (recall + precision)

    def _compute_ragas_metrics(self, result: EvalResult, query: Dict):
        """
        完整 Ragas 评估（需安装 ragas 库）
        pip install ragas
        """
        try:
            from ragas.metrics import context_precision, faithfulness, answer_relevancy
            from ragas import evaluate
            from datasets import Dataset

            ds = Dataset.from_dict({
                "question": [query['query']],
                "answer": [result.answer],
                "contexts": [result.retrieved_docs],
                "ground_truth": [query.get('expected_answer', '')]
            })

            scores = evaluate(
                ds,
                metrics=[context_precision, faithfulness, answer_relevancy]
            )

            result.context_precision = float(scores.get('context_precision', 0))
            result.faithfulness = float(scores.get('faithfulness', 0))
            result.answer_relevancy = float(scores.get('answer_relevancy', 0))

        except ImportError:
            logger.warning("ragas 未安装，回退到简化评估")
            self._compute_simple_metrics(result, query)

    def get_summary(self) -> Dict[str, Any]:
        """生成评估摘要报告"""
        if not self.results:
            return {"error": "无评估结果"}

        # 按类别汇总
        by_category: Dict[str, List[EvalResult]] = {}
        for r in self.results:
            cat = r.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(r)

        # 计算总体指标
        total = len(self.results)
        avg_latency = sum(r.latency_ms for r in self.results) / total
        avg_precision = sum(r.context_precision for r in self.results) / total
        avg_recall = sum(r.context_recall for r in self.results) / total
        avg_faithfulness = sum(r.faithfulness for r in self.results) / total
        avg_relevancy = sum(r.answer_relevancy for r in self.results) / total
        avg_correctness = sum(r.answer_correctness for r in self.results) / total

        error_count = sum(1 for r in self.results if r.error)

        # 按类别统计
        category_stats = {}
        for cat, cat_results in by_category.items():
            n = len(cat_results)
            category_stats[cat] = {
                "count": n,
                "avg_latency_ms": round(sum(r.latency_ms for r in cat_results) / n, 1),
                "avg_context_precision": round(sum(r.context_precision for r in cat_results) / n, 3),
                "avg_answer_correctness": round(sum(r.answer_correctness for r in cat_results) / n, 3),
                "error_count": sum(1 for r in cat_results if r.error)
            }

        # 按难度统计
        by_difficulty: Dict[str, List[EvalResult]] = {}
        for r in self.results:
            diff = r.difficulty
            if diff not in by_difficulty:
                by_difficulty[diff] = []
            by_difficulty[diff].append(r)

        difficulty_stats = {}
        for diff, diff_results in by_difficulty.items():
            n = len(diff_results)
            difficulty_stats[diff] = {
                "count": n,
                "avg_correctness": round(sum(r.answer_correctness for r in diff_results) / n, 3),
            }

        summary = {
            "总查询数": total,
            "错误数": error_count,
            "成功率": f"{(total - error_count) / total:.1%}",
            "--- 核心指标 ---": "",
            "Context Precision": f"{avg_precision:.3f}",
            "Context Recall (Hit@K)": f"{avg_recall:.3f}",
            "Faithfulness (忠实度)": f"{avg_faithfulness:.3f}",
            "Answer Relevancy": f"{avg_relevancy:.3f}",
            "Answer Correctness": f"{avg_correctness:.3f}",
            "--- 性能 ---": "",
            "平均延迟": f"{avg_latency:.0f}ms",
            "--- 按类别 ---": "",
        }
        for cat, stats in category_stats.items():
            summary[f"  {cat}"] = (
                f"Precision={stats['avg_context_precision']:.3f}, "
                f"Correctness={stats['avg_answer_correctness']:.3f}, "
                f"Latency={stats['avg_latency_ms']:.0f}ms"
            )

        summary["--- 按难度 ---"] = ""
        for diff, stats in difficulty_stats.items():
            summary[f"  {diff}"] = (
                f"{stats['count']}条, "
                f"Correctness={stats['avg_correctness']:.3f}"
            )

        return summary

    def print_report(self):
        """打印评估报告"""
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print("  GraphRAG 系统评估报告 (Ragas 兼容)")
        print("=" * 60)
        for key, value in summary.items():
            if value is None:
                print(f"\n  {key}")
            else:
                print(f"  {key}: {value}")
        print("=" * 60)

    def compare_to_baseline(self, baseline_results: List[EvalResult] = None) -> Dict:
        """
        与 Baseline（传统 Vector-RAG）对比
        计算相对提升
        """
        if baseline_results is None:
            # 模拟传统 Vector-RAG baseline 数据
            # 生产环境需实际运行传统 RAG 获取真实数据
            baseline_precision = 0.45
            baseline_correctness = 0.52
        else:
            baseline_precision = sum(r.context_precision for r in baseline_results) / len(baseline_results)
            baseline_correctness = sum(r.answer_correctness for r in baseline_results) / len(baseline_results)

        current_precision = sum(r.context_precision for r in self.results) / len(self.results)
        current_correctness = sum(r.answer_correctness for r in self.results) / len(self.results)

        precision_gain = (current_precision - baseline_precision) / baseline_precision * 100
        correctness_gain = (current_correctness - baseline_correctness) / baseline_correctness * 100

        return {
            "baseline_context_precision": round(baseline_precision, 3),
            "current_context_precision": round(current_precision, 3),
            "context_precision_gain": f"{precision_gain:.0f}%",
            "baseline_answer_correctness": round(baseline_correctness, 3),
            "current_answer_correctness": round(current_correctness, 3),
            "answer_correctness_gain": f"{correctness_gain:.0f}%"
        }


def main():
    """独立运行评估"""
    print("GraphRAG 系统评估 (评估模式)")
    print("-" * 40)

    evaluator = GraphRAGEvaluator()
    queries = evaluator.load_queries()

    # 统计测试集信息
    categories = {}
    for q in queries:
        cat = q['category']
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\n测试集概况: {len(queries)} 条查询")
    for cat, count in categories.items():
        print(f"  {cat}: {count} 条")

    print("\n提示: 启动完整评估需实例化 orchestrator:")
    print("  orchestrator = AgentOrchestrator(neo4j_client, milvus_client)")
    print("  evaluator = GraphRAGEvaluator(orchestrator)")
    print("  evaluator.run_evaluation()")
    print("  evaluator.print_report()")


if __name__ == '__main__':
    main()
