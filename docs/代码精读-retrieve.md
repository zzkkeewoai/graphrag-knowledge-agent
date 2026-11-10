# 代码精读：`hybrid_retriever.py::retrieve()`

> 位置：`src/retriever/hybrid_retriever.py` 第 **112–206 行**。
> **这 96 行是整个项目最核心的一段**——三路召回、融合、保底、域过滤全在这。
> 读懂了它，执行流程就通了 70%。

---

## 一、它在链路里的位置

```
orchestrator 的 local_node
      ↓ 调用
LocalAgent.execute(query, domains)          ← src/agents/orchestrator_v2.py
      ↓ 调用
HybridRetriever.retrieve(query, top_k, hops, domains)   ← 就是这一段
      ↓ 返回
{graph_results, vector_results, bm25_results, fused_results}
```

**输入**：一句查询 + 三个参数
**输出**：四份结果（三路原始 + 融合后）

---

## 二、逐段读（8 段）

### 段 1：参数（112–128 行）

```python
def retrieve(self, query, top_k=5, hops=2, domains=None):
    fetch_k = top_k * 4 if domains else top_k
```

**大白话**：`top_k` 是"最后要几条"，`fetch_k` 是"先取几条"。

**为什么乘 4**：有域过滤时，取回来的 5 条可能只有 1 条属于这个域——过滤完就没了。所以**多取 4 倍再过滤再截断**。

**如果去掉会怎样**：域隔离的准确性会崩（过滤后候选不足，答不出来）。

---

### 段 2：图谱召回（130–143 行）

```python
graph_results = self._graph_retrieve(query, hops=hops)
if domains and graph_results:
    emap = self._entity_domain_map()
    graph_results = [r for r in graph_results
                     if emap.triple_keep(r.get("entity"), r.get("target"), domains)]
```

**大白话**：先去 Neo4j 查实体和它的两跳邻居，然后**按域过滤**。

**为什么用 `triple_keep`（三元组级）而不是 `keep`（实体级）**：
主语常是"公司"这种通用词，两个域都有；真正带域信息的是宾语。所以要**看两端的域集合，取更具体的那个**。

**如果去掉过滤**：就是那个串味 bug——simco 域答出 pdf 的 2,137 人。

---

### 段 3：向量召回（145–149 行）

```python
vector_results = self._vector_retrieve(query, fetch_k)
if domains:
    vector_results = [r for r in vector_results
                      if keep(r.get("doc_id"), domains)][:top_k]
```

**大白话**：查询变向量 → 和库里所有块算余弦 → 取前 fetch_k 条 → 按域过滤 → 截到 top_k。

**为什么它最贵**：要跑一次 embedding 模型（约 0.1–0.3 秒），再算 414 次点积。

---

### 段 4：BM25 召回（151–155 行）

```python
bm25_results = self._bm25_retrieve(query, fetch_k)
```

**大白话**：按关键词（字符二元组）查倒排索引，取前 fetch_k 条。

**它补什么**：向量对"第十五条""净资产 50%"这种**字面唯一**的东西不敏感，BM25 专治这个。

---

### 段 5：RRF 融合（157–159 行）

```python
fused = self.reciprocal_rank_fusion([graph_results, vector_results, bm25_results])
```

**大白话**：三份列表，按"名次"打分合成绩，重新排序。

```
每路：score += 1 / (60 + 名次)
```

**为什么不用分数相加**：三路量纲不同（余弦 0~1、BM25 几十分、图谱匹配度）。**RRF 只看名次，天然免疫。**

**致命缺点**：**它不看来源**。谁在某一路上排得靠前，谁的分数就高——**哪怕这一路本身结果很多、很杂**。

---

### 段 6：向量保底（161–184 行）

```python
for v in vector_results:                 # 全部 8 条
    if key not in fused_keys:
        fused.append({**v, "fusion_score": 0.0})
```

**大白话**：把向量召回的所有结果**强行塞进融合列表**，不管 RRF 给它们打多少分。

**为什么必须这么做**：图谱一路能返回 90 条，向量只有 8 条。RRF 一算，**向量被整体挤到窗口外**——而向量是**唯一带原文**的一路，它一没，模型就只能看着三元组猜。

**为什么是"全量"而不是"前 2 名"**：我试过三版——
```
v1 补"向量前 2 名"     → 无效（它们本来就在融合结果里）
v2 补"未入选的前 2 条"  → 第 5 名还是丢（"A项目使用Python"就排第 5）
v3 全部保留            → 有效
```

---

### 段 7：BM25 保底（186–198 行）

```python
for b in bm25_results[:2]:               # 只保前 2
```

**大白话**：同理，BM25 最能说明"精确命中"的那 2 条，也不许被挤掉。

**为什么只保 2 条**：BM25 结果可能很多，全保会把噪声带进来；前 2 名通常就是最精确的字面命中。

---

### 段 8：返回（200–206 行）

```python
return {"graph_results": ..., "vector_results": ..., "bm25_results": ..., "fused_results": ...}
```

**大白话**：三路原始结果都返回，因为下游还要用：
- `fused_results` → 拼成候选文档
- `graph_results` → 给证据验证做结构化匹配
- `vector/bm25` → 拼 text_map 找原文

---

## 三、四个参数的含义（面试常问）

| 参数 | 含义 | 现在值 | 调大/调小会怎样 |
|---|---|---|---|
| `top_k` | 向量/BM25 最终取几条 | 8（`VECTOR_TOP_K`） | 调小省成本但漏证据；调大延迟涨 |
| `hops` | 图谱走几跳 | 2 | 调大召回多但邻居爆炸（实测 90 条） |
| `domains` | 只在这些域检索 | None/`["pdf"]` | 不传=混检会串味 |
| `fetch_k` | 先取几条再过滤 | `top_k*4`（有域时） | 不乘 4，过滤后候选不够 |

---

## 四、三个容易误解的点

**① `fusion_score: 0.0` 不是"这条不重要"**
它是保底塞进去的，赋值 0 只是因为 RRF 没给它算分。**后面的精排会重新打分**，所以不会因为 0 分被淘汰。

**② 保底是"追加"，不是"替换"**
它没有把图谱结果踢出去，只是往列表尾部补。最终谁进上下文，由精排决定。

**③ 域过滤做了两次**
向量/BM25 按 `doc_id` 过滤；图谱按"实体/宾语"过滤。**因为图谱的 doc_id 是 `graph:实体名`，和块的 doc_id 不是一个东西。**

---

## 五、自己动手验证（3 条命令）

```powershell
# ① 看三路各给了什么、融合后谁在窗口里
python tests/debug_fusion.py "对外担保超过净资产50%" --needle 净资产

# ② 看向量召回里目标块排第几
python tests/debug_retrieval.py "对外担保超过净资产50%" --top 8 --contains 50%

# ③ 改一个参数看结果变不变（这就是"配置有没有接线"的检查法）
#    .env 里改 VECTOR_TOP_K=3 → 重启 → 看 document_count 变不变
```

---

## 六、口述版（40 秒，面试用）

「这个函数做三件事：**三路召回、融合、保底**。

三路各取一批候选，先按域过滤。然后用 RRF 融合——只吃排名不吃分数，因为三路量纲不同。

但 RRF 有个致命缺点：不看来源。图谱一路能返 90 条，向量只有 8 条，融合后向量被整体挤出去。而向量是唯一带原文的一路，它一没，模型就看不到证据。

所以我加了保底：**向量全量保留、BM25 保前 2**，让稀缺的那一路整体豁免。」
