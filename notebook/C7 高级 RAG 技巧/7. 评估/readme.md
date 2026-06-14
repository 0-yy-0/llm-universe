# C7.7 工业级 RAG 评估与迭代

## 从 C5 到 C7.7

C5 介绍了基于 Bad Case 的验证迭代方法（人工评估、大模型自动评估），适用于项目原型阶段。本章聚焦**生产环境的系统化评估**，解决三个新挑战：

| 挑战 | C5 原型验证 | C7.7 生产评估 |
|------|-----------|-------------|
| **规模** | 几十条 Bad Case | 数百到上千条分桶评估集 |
| **持续性** | 一次性人工检查 | 每次改动自动回归 |
| **可归因性** | 整体打分 | 分层定位到具体环节 |

## 本章内容

| 序号 | 文件 | 内容 |
|------|------|------|
| 1 | [工业级 RAG 评估体系概述](1.工业级RAG评估体系概述.md) | 方法论总纲：评估全景图、RAG 三元组、工具生态 |
| 2 | [RAG 三元组与分层指标](2.RAG三元组与分层指标.ipynb) | 用 Ragas 对 RAG Pipeline 跑出分层评估报告 |
| 3 | [LLM-as-Judge 评估](3.LLM-as-Judge评估.ipynb) | 手写 Judge Prompt，掌握操作级评估 |
| 4 | [评估集工程](4.评估集工程.ipynb) | 构建、分桶、维护可持续更新的评估集 |
| 5 | [错误归因与根因分析](5.错误归因与根因分析.ipynb) | 拿到低分后，系统化定位并修复问题 |
| 6 | [CI/CD 质量门禁与在线监控](6.CI_CD质量门禁与在线监控.ipynb) | 把评估集成到开发流程，用 DeepEval 做回归测试 |
| 7 | [工具对比与选型](7.工具对比与选型.md) | 6 大工具对比表 + 选型决策树 |

## 前置知识

- 已学完 **C5 系统评估与优化**（了解 Bad Case 评估方法）
- 已学完 **C7.2 ~ C7.6**（了解高级 RAG 各阶段技巧）

## 推荐学习顺序

1. 先读 **概述**（10 分钟）掌握全景
2. 跑 **Notebook 2**，理解分层指标
3. 跑 **Notebook 3**，掌握 LLM-as-Judge
4. 跑 **Notebook 4**，学会构建评估集
5. 跑 **Notebook 5**，学会归因分析
6. 跑 **Notebook 6**，掌握 CI/CD 集成
7. 读 **工具选型**，为实际项目做技术选型

## 环境要求

```bash
pip install ragas deepeval langchain langchain-community langchain-chroma \
    sentence-transformers modelscope pymupdf scikit-learn matplotlib \
    zhipuai python-dotenv datasets tqdm
```

在项目根目录的 `.env` 文件中配置 `ZHIPUAI_API_KEY`。
