import logging
from typing import List, Dict, Any
from neo4j import GraphDatabase, Result
from src.schemas.graph_schema import Triple
from src.utils.entity_normalizer import normalize_entity_name

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j图数据库客户端"""

    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    def connect(self):
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        logger.info(f"已连接到Neo4j: {self.uri}")

    def close(self):
        if self.driver:
            self.driver.close()

    def create_indexes(self):
        """创建索引加速查询"""
        with self.driver.session() as session:
            session.run("CREATE INDEX entity_name_idx IF NOT EXISTS FOR (n:Entity) ON (n.name)")
            session.run("CREATE INDEX entity_type_idx IF NOT EXISTS FOR (n:Entity) ON (n.type)")
            logger.info("索引创建完成")

    @staticmethod
    def _entity_row(triple: Triple) -> Dict[str, Any]:
        """把三元组转成写入参数 —— **实体名在这里规范化**。

        为什么单独抽成静态方法：
            写路径有两条（单条 `write_triple` / 批量 `batch_write`）。
            如果各自手写一遍规范化，**迟早有一条会漏**（这正是"策略散落在各处"
            的经典失效）。抽出来之后两条路径共用同一份逻辑，
            而且**能脱离 Neo4j 做单元测试**。

        为什么同时写 name 和 display_name：
            name         规范化后的键  → 用于 MERGE，保证同一实体只有一个节点
            display_name 原始写法      → 用于展示（保留人类可读形式）

        为什么 display_name 用 coalesce（首次写入优先）：
            同一个实体可能有多种原始写法（"张三" / "张三 " / "Ａ项目"）。
            如果每次都覆盖，展示出来的名字就取决于**抽取顺序**，不稳定。
            让第一次写入的写法留下，展示名才是确定的。
        """
        return {
            "head": normalize_entity_name(triple.head),
            "head_display": triple.head,
            "head_type": triple.head_type.value,
            "tail": normalize_entity_name(triple.tail),
            "tail_display": triple.tail,
            "tail_type": triple.tail_type.value,
            "relation_type": triple.relation.value,
            "confidence": triple.confidence,
        }

    def write_triple(self, triple: Triple) -> bool:
        """写入单个三元组

        用统一的关系类型 RELATES_TO 存储，通过 relation_type 属性区分关系种类，
        避免 Cypher 中关系类型无法参数化的问题，也无需依赖 APOC 插件。

        实体名经 `normalize_entity_name` 规范化后作为节点唯一键 ——
        否则"同一个实体的不同写法"会静默变成两个节点（见 entity_normalizer 的说明）。
        """
        with self.driver.session() as session:
            cypher = """
            MERGE (h:Entity {name: $head})
            SET h.type = $head_type,
                h.display_name = coalesce(h.display_name, $head_display)
            MERGE (t:Entity {name: $tail})
            SET t.type = $tail_type,
                t.display_name = coalesce(t.display_name, $tail_display)
            MERGE (h)-[r:RELATES_TO]->(t)
            SET r.relation_type = $relation_type,
                r.confidence = $confidence
            RETURN h, r, t
            """
            result = session.run(cypher, **self._entity_row(triple))
            return result.single() is not None

    def batch_write(self, triples: List[Triple], batch_size: int = 500,
                    max_retries: int = 2) -> Dict[str, int]:
        """批量写入（工程化修复 v2.2）

        原实现：for 循环逐条 write_triple() —— 每条一次网络 Round Trip，
        大量数据时开销巨大。
        现实现：
        1. UNWIND 批量写入（500 条/批），大幅减少网络往返
        2. 每个 batch 一个显式事务（session.execute_write）
        3. batch 失败重试（有限次数），而不是整个数据集重来
        4. MERGE 天然幂等：同一条目重复执行不产生重复数据
        """
        success = 0
        failed = 0

        def _run_batch(tx, batch_rows):
            tx.run(
                """
                UNWIND $rows AS row
                MERGE (h:Entity {name: row.head})
                SET h.type = row.head_type,
                    h.display_name = coalesce(h.display_name, row.head_display)
                MERGE (t:Entity {name: row.tail})
                SET t.type = row.tail_type,
                    t.display_name = coalesce(t.display_name, row.tail_display)
                MERGE (h)-[r:RELATES_TO]->(t)
                SET r.relation_type = row.relation_type,
                    r.confidence = row.confidence
                """,
                rows=batch_rows,
            )

        def _to_row(t: Triple) -> dict:
            # 复用 _entity_row —— 批量路径和单条路径必须走同一份规范化逻辑，
            # 否则两条写入路径会产生不一致的节点键
            return Neo4jClient._entity_row(t)

        # 按 batch_size 分批
        for i in range(0, len(triples), batch_size):
            batch = triples[i:i + batch_size]
            rows = [_to_row(t) for t in batch]

            # 有限重试
            for attempt in range(max_retries + 1):
                try:
                    with self.driver.session() as session:
                        session.execute_write(_run_batch, rows)
                    success += len(batch)
                    break
                except Exception as e:
                    logger.error(f"batch 写入失败 (offset={i}, attempt={attempt + 1}): {e}")
                    if attempt < max_retries:
                        import time
                        time.sleep(0.5 * (2 ** attempt))  # 指数退避
                    else:
                        failed += len(batch)
                        # 单条定位失败原因（仅失败 batch 内逐条尝试，降低损失）
                        for t in batch:
                            try:
                                self.write_triple(t)
                                success += 1
                            except Exception as e2:
                                logger.error(f"单条写入失败 {t.head}->{t.tail}: {e2}")
                                failed += 1

        return {"success": success, "failed": failed}

    def query_tech_stack(self, project_name: str) -> List[str]:
        """查询项目使用的技术栈

        入参同样要规范化 —— 写入时用了规范键，查询时必须用同一套规则，
        否则会出现"数据明明写进去了却查不到"的诡异现象。
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Entity {name: $name, type: '项目'})-[r:RELATES_TO]->(t:Entity {type: '技术栈'})
                WHERE r.relation_type = '使用'
                RETURN t.name AS tech
                """,
                name=normalize_entity_name(project_name)
            )
            return [record["tech"] for record in result]

    def query_dependencies(self, project_name: str) -> List[Dict]:
        """查询项目的依赖关系"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p1:Entity {name: $name, type: '项目'})-[r:RELATES_TO]->(p2:Entity {type: '项目'})
                WHERE r.relation_type = '依赖'
                RETURN p2.name AS depends_on, r.confidence AS confidence
                """,
                name=normalize_entity_name(project_name)
            )
            return [{"project": record["depends_on"], "confidence": record["confidence"]} for record in result]

    def query_entity_by_relation(self, entity_name: str, relation_type: str) -> List[Dict]:
        """定向图谱检索（v2.2 Evidence Validator 补充召回用）

        针对用户问题中的 entity + 期望 relation_type 做精确结构化查询，
        只返回 relation_type 完全匹配的关系——用于 Evidence Validator
        判定 insufficient 后补充"正向证据"。

        Args:
            entity_name: 目标实体（如"张三"）
            relation_type: 期望关系类型（如"负责"）

        Returns:
            [{"entity", "relation", "target", "target_type", "confidence"}, ...]
            查询无结果或异常返回空列表（绝不抛错，由调用方走 insufficient/refuse）。
        """
        if not entity_name or not relation_type:
            return []
        try:
            entity_key = normalize_entity_name(entity_name)
        except (TypeError, ValueError):
            # 实体名非法（空 / 全是不可见字符）→ 当作"查不到"处理，
            # 由调用方走 insufficient / refuse，绝不抛错打断整条链路
            logger.warning(f"实体名非法，跳过定向检索: {entity_name!r}")
            return []
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (e:Entity {name: $entity})-[r:RELATES_TO]->(t:Entity)
                    WHERE r.relation_type = $rel
                    RETURN e.name AS entity,
                           r.relation_type AS relation,
                           t.name AS target,
                           t.type AS target_type,
                           r.confidence AS confidence
                    LIMIT 50
                    """,
                    entity=entity_key,
                    rel=relation_type
                )
                return [
                    {
                        "entity": record["entity"],
                        "relation": record["relation"],
                        "target": record["target"],
                        "target_type": record["target_type"],
                        "confidence": record["confidence"],
                    }
                    for record in result
                ]
        except Exception as e:
            logger.warning(f"定向图谱检索异常 (entity={entity_name}, rel={relation_type}): {e}")
            return []