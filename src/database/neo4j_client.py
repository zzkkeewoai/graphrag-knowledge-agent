import logging
from typing import List, Dict, Any
from neo4j import GraphDatabase, Result
from src.schemas.graph_schema import Triple

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

    def write_triple(self, triple: Triple) -> bool:
        """写入单个三元组

        用统一的关系类型 RELATES_TO 存储，通过 relation_type 属性区分关系种类，
        避免 Cypher 中关系类型无法参数化的问题，也无需依赖 APOC 插件。
        """
        with self.driver.session() as session:
            cypher = """
            MERGE (h:Entity {name: $head})
            SET h.type = $head_type
            MERGE (t:Entity {name: $tail})
            SET t.type = $tail_type
            MERGE (h)-[r:RELATES_TO]->(t)
            SET r.relation_type = $relation_type,
                r.confidence = $confidence
            RETURN h, r, t
            """
            result = session.run(
                cypher,
                head=triple.head,
                head_type=triple.head_type.value,
                tail=triple.tail,
                tail_type=triple.tail_type.value,
                relation_type=triple.relation.value,
                confidence=triple.confidence
            )
            return result.single() is not None

    def batch_write(self, triples: List[Triple]) -> Dict[str, int]:
        """批量写入"""
        success = 0
        failed = 0
        for triple in triples:
            try:
                if self.write_triple(triple):
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"写入失败: {e}")
                failed += 1
        return {"success": success, "failed": failed}

    def query_tech_stack(self, project_name: str) -> List[str]:
        """查询项目使用的技术栈"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Entity {name: $name, type: '项目'})-[r:RELATES_TO]->(t:Entity {type: '技术栈'})
                WHERE r.relation_type = '使用'
                RETURN t.name AS tech
                """,
                name=project_name
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
                name=project_name
            )
            return [{"project": record["depends_on"], "confidence": record["confidence"]} for record in result]