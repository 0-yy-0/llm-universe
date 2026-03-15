# 第六章：增强阶段（重构版）

本目录采用“三层增强 + 选型总结”的结构讲解第六章，避免与前几章重复：

1. `0. 先导：为什么基础 RAG 还不够.ipynb`
2. `1. 上下文增强（重构版）.ipynb`
3. `2. 流程增强（重构版）.ipynb`
4. `3. 系统增强.ipynb`
5. `4. 选型总结.md`

## 本章新增方法（2026）

- 上下文增强：`Sentence Window`、`Small-to-Big`、`AutoMerging`、`Late Chunking（理论）`
- 流程增强：`迭代检索`、`递归检索`、`查询路由与自适应检索`、`Corrective RAG`、`Self-RAG`
- 系统增强：`Memory`、`Multi-Document Agent`、`Agentic RAG`、`GraphRAG`

## 与前几章的边界

- 第3章（索引）已讲：混合检索、元数据、CCH、文档增强
- 第4章（检索）已讲：query 改写、HyDE、step-back、子查询
- 第5章（生成）已讲：压缩、重排、过滤、引用

本章重点是：**定位失败类型 -> 选择增强层级 -> 组合方法落地**。

## 旧版内容说明

- `0. 其他增强（llama-index 版 rag fusion，混合检索，重写查询）.ipynb` 保留为历史草稿，作为补充参考，不再作为主线内容。
- `1. 上下文增强.ipynb` 与 `2. 流程增强.ipynb` 保留原始版本，便于对照。
