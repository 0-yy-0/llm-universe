# 增强阶段 2026 更新设计文档

## 背景

第 6 章增强阶段已有"三层增强 + 选型总结"结构，覆盖 11 种方法。作为 2026 年教程，需补充近两年 RAG 领域重要新技术，修复现有代码缺陷，同步更新导航与选型文档。

## 设计决策记录

### 决策 1：Contextual Retrieval 归属

- **结论**：放在第 3 章 CCH notebook 末尾，作为 CCH 的自然升级。第 6 章上下文增强的"如何选择"部分加引导语指向第 3 章。
- **理由**：两者都是"索引时为 chunk 附加上下文"，属于索引阶段技术。第 6 章的上下文增强（Sentence Window / Small-to-Big / AutoMerging）都是检索时恢复上下文，机制不同。

### 决策 2：GraphRAG 位置与深度

- **结论**：放在第 6 章 `3. 系统增强.ipynb` 末尾，轻量处理（理论 + 纯 dict 最小示例）。
- **理由**：GraphRAG 涉及实体提取、图构建、路由等复杂编排，工程复杂度对标系统增强层。但不引入 neo4j/networkx，用纯 Python dict 模拟。

### 决策 3：Adaptive Retrieval 处理方式

- **结论**：合并到现有"查询路由"节，扩展为"查询路由与自适应检索"。
- **理由**：查询路由回答"走哪条路"，Adaptive Retrieval 回答"走多深"，本质都是检索前策略选择。合并避免方法膨胀。

### 决策 4：Multi-Doc Agent / Agentic RAG 修复

- **结论**：直接修复。route_city → route_cities 支持多城市；Agentic RAG executor 按 step 独立路由。
- **理由**：明确的 bug 修复，无方案分歧。

## 改动清单

| # | 文件 | 改动 | 量级 |
|---|---|---|---|
| 1 | `1. 上下文增强（重构版）.ipynb` | "如何选择"加引导语：chunk 缺文档语境应回第 3 章 CCH/Contextual Retrieval | 小 |
| 2 | `1. 上下文增强（重构版）.ipynb` | 对比表后新增 Late Chunking 纯理论介绍（无代码） | 小 |
| 3 | `2. 流程增强（重构版）.ipynb` | "查询路由"→"查询路由与自适应检索"，追加 query 复杂度分类代码 | 中 |
| 4 | `3. 系统增强.ipynb` | route_city → route_cities，支持多城市路由合并 | 小 |
| 5 | `3. 系统增强.ipynb` | Agentic RAG executor 按 step 独立路由 | 小 |
| 6 | `3. 系统增强.ipynb` | 末尾新增 GraphRAG 轻量介绍（理论 + 纯 dict 最小示例） | 中 |
| 7 | `0. 先导.ipynb` | 导航表 + mermaid 方法地图同步更新 | 小 |
| 8 | `4. 选型总结.md` | 决策图/速查表/推荐组合新增 GraphRAG、自适应检索 | 中 |
| 9 | `readme.md` | 同步方法列表 | 小 |
| 10 | 第 3 章 `3.CCH.ipynb` | 末尾新增"从 CCH 到 Contextual Retrieval"一节 | 中 |

## 约束

- 全部使用 LangChain，不用 LangGraph
- 不新增 pip 依赖
- Late Chunking 只讲理论，不做可运行代码
- GraphRAG 用纯 Python dict 模拟图结构
- 代码风格与现有 notebook 一致（ChatOpenAI + Chroma + OpenAIEmbeddings）
- 教学优先，不追求生产级
