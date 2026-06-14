# C7.7 工业级 RAG 评估与迭代 — 设计文档

> 日期：2026-03-15
> 状态：已确认

## 一、定位与边界

### 1.1 定位

C7.7 = **面向生产环境的 RAG 系统化评估闭环**。

与 C5 的关系：
- C5 = 项目原型验证（Bad Case → Prompt 迭代、人工评估 + 简单自动评估）
- C7.7 = 生产系统评估（分层指标 → LLM-as-Judge → 评估集工程 → 归因 → CI/CD → 在线监控）

### 1.2 不讲的内容（避免与 C5 重叠）

- 什么是评估、为什么要评估
- 什么是 Bad Case
- 人工评估的基本概念
- "使用大模型评估"的基础定义
- 检索常见失败模式的枚举（C5/3 已讲）
- 基础 Prompt 优化案例

### 1.3 目标读者

分层覆盖三类读者：
- **入门开发者**：跟着 Notebook 跑通完整评估流程
- **RAG 工程师**：拿到方法论 + 关键代码片段，能在自己项目中落地
- **架构师/技术负责人**：拿到选型决策框架，10 分钟看完全景

## 二、技术约束

| 项目 | 选择 | 原因 |
|------|------|------|
| LLM | 智谱 AI `glm-4-flash` | 与 C7 其他章节一致 |
| Embedding | 本地 `BAAI/bge-small-zh-v1.5` | 零 API 成本，本地可复现 |
| 评估数据 | 南瓜书（复用 C7 已有数据） | 保持教程连贯性 |
| 评估工具 | 不绑定单一工具，每环节选最合适的 | Ragas / DeepEval / 手写 Judge |
| 交付形式 | Jupyter Notebook（每小节一个） | 理论 + 可运行代码混排 |
| 篇幅 | 充实型，类似 C7.2 数据处理的体量 | 每个 Notebook ~25-35 cells |

## 三、文件结构

```
7. 评估/
├── readme.md                              ← 本章导读
├── 1.工业级RAG评估体系概述.md              ← 方法论总纲（架构师入口）
├── 2.RAG三元组与分层指标.ipynb             ← 指标设计 + Ragas 实战
├── 3.LLM-as-Judge评估.ipynb              ← Judge Prompt + 操作级评估
├── 4.评估集工程.ipynb                      ← 构建/分桶/维护评估集
├── 5.错误归因与根因分析.ipynb              ← 从低分到定位到行动
├── 6.CI_CD质量门禁与在线监控.ipynb         ← DeepEval CI + 在线指标
├── 7.工具对比与选型.md                     ← 对比表 + 决策树
├── data/
│   ├── eval_questions.json                ← 分桶评估问题集
│   └── golden_answers.json                ← 标准答案集
└── figures/
```

## 四、各文件详细设计

### 4.1 readme.md — 本章导读

- 用 2-3 句话衔接 C5："C5 介绍了基于 Bad Case 的验证迭代方法，本章聚焦生产环境的三个新挑战：规模、持续性、可归因性。"
- 本章学习路线图
- 各节一句话简介
- 前置知识提示

### 4.2 工业级 RAG 评估体系概述（md，~800 字）

**面向架构师，10 分钟读完全景。**

内容：
1. 生产环境评估 vs 原型验证的 3 个核心差异
   - 规模：从几十条到上千条评估集
   - 持续性：每次改动都需要回归验证
   - 可归因性：低分需要定位到具体环节
2. 评估体系全景图（一张流程图）
   - 分层指标 → LLM-as-Judge → 评估集 → 归因 → CI/CD → 在线监控
3. RAG 三元组速览（Context Relevance / Groundedness / Answer Relevance）
4. 2026 工具生态对比表（6 个工具 × 6 个维度）
5. 本章学习路线图

### 4.3 RAG 三元组与分层指标（ipynb，~30 cells）

**目标：用 Ragas 对 RAG pipeline 跑出分层评估报告。**

理论 cells：
- RAG Triad 详解：三个指标的定义、计算方式、目标阈值（Faithfulness ≥ 0.80, Answer Relevance ≥ 0.85）
- 分层指标体系：
  - 检索层：Context Precision / Context Recall
  - 生成层：Faithfulness / Answer Relevance
  - 系统层：延迟 / Token 成本
- 每个指标的适用场景和局限

代码 cells：
1. 环境准备（modelscope 下载 bge-small-zh-v1.5）
2. 搭建简单 RAG Pipeline（南瓜书 + Chroma + glm-4-flash）
3. 准备评估数据（问题 + ground truth 答案）
4. 安装 Ragas，配置 LLM Provider
5. 运行 Ragas 评估（context_precision, context_recall, faithfulness, answer_relevancy）
6. 输出分层评估报告
7. 可视化：雷达图展示各维度得分
8. 小结 + 实践建议

### 4.4 LLM-as-Judge 评估（ipynb，~35 cells）

**目标：掌握用 LLM 做评估的核心技术，能自定义 Judge。**

理论 cells：
- LLM-as-Judge 原理：为什么用 LLM 评估 LLM
- Judge Prompt 设计原则：
  - 带推理链（CoT）vs 直接打分
  - 评分标准锚定（0-10 分，每档有明确定义）
- 操作级评估：对 pipeline 中检索、生成分别打分
- 采样策略：按比例抽样控制评估成本
- 校准：Judge 评分和人工标注的一致性验证

代码 cells：
1. 手写 Context Relevance Judge Prompt
2. 手写 Faithfulness Judge Prompt
3. 对比实验：直接打分 vs CoT 打分的质量差异
4. 批量评估函数封装
5. 结果统计与分桶分析
6. Cohen's Kappa 一致性验证（模拟人工标注 vs Judge 评分）
7. 小结 + 实践建议

### 4.5 评估集工程（ipynb，~25 cells）

**目标：构建一套可维护的分桶评估集。**

理论 cells：
- 评估集三种类型：
  - Golden Set：核心回归集，每次必跑
  - Regression Set：历史 Bad Case 累积
  - Hard Set：边界挑战用例
- 分桶策略：事实问答 / 多跳推理 / 模糊问题 / 拒答场景
- 评估集持续更新机制

代码 cells：
1. 用 LLM 基于南瓜书自动生成分桶评估问题
2. 构建 JSON 评估集（含问题、预期答案、桶标签、难度标签）
3. 用前面的 Ragas + LLM-as-Judge 跑评估
4. 按桶聚合得分，找出弱点桶
5. 输出分桶评估报告
6. 小结 + 实践建议

### 4.6 错误归因与根因分析（ipynb，~25 cells）

**目标：拿到低分结果后，系统化定位问题。**

理论 cells：
- 归因三步法：
  1. 看分层指标，哪一层分低
  2. 层内细分，是 Recall 低还是 Precision 低
  3. 确认根因，是数据覆盖不够还是 embedding/prompt 问题
- 归因矩阵（表格）：
  - 检索低分 → Recall 低（数据/embedding）or Precision 低（噪声）
  - 生成低分 → Faithfulness 低（幻觉）or Relevance 低（答非所问）
  - 两层都低 → 评估集本身可能有问题
- 从归因到行动：每种根因对应的修复方向

代码 cells：
1. 取评估结果中得分最低的 5 个 case
2. 逐个 case 分析：检索结果 + 生成结果 + Judge 评分
3. 自动归因分类函数
4. 输出归因报告（表格形式）
5. 小结 + 实践建议

### 4.7 CI/CD 质量门禁与在线监控（ipynb，~30 cells）

**目标：把评估集成到开发流程中。**

理论 cells：
- 离线评估：每次改动后自动跑回归
- 质量门禁：设阈值（Faithfulness ≥ 0.80），不过不合并
- 在线监控：追问率、转人工率、用户反馈
- 全链路版本化：索引、embedding、prompt、模型、评估集

代码 cells：
1. 用 DeepEval 定义 pytest 风格的评估测试用例
2. 定义质量门禁阈值
3. 模拟 CI 流程：改一个 prompt → 跑评估 → 对比前后得分
4. 输出版本对比报告
5. 简单在线指标采集示例
6. 小结 + 实践建议

### 4.8 工具对比与选型（md，~600 字）

**面向决策者的选型地图。**

内容：
1. 6 个工具对比表：

| 工具 | 类型 | 核心能力 | CI 集成 | 中文支持 | 适合场景 |
|------|------|---------|---------|---------|---------|
| Ragas | 开源指标库 | 组件级指标最全 | 中 | 好 | 离线评估 |
| DeepEval | 开源测试框架 | pytest 风格，50+ 指标 | 强 | 好 | CI/CD 集成 |
| TruLens | 开源评估框架 | RAG Triad，可编程反馈函数 | 中 | 中 | 实验对比 |
| LangSmith | 商业平台 | LangChain 生态，实验管理 | 强 | 好 | LangChain 用户 |
| Langfuse | 开源可观测 | 操作级评估，OpenTelemetry | 强 | 好 | 在线监控 |
| Arize Phoenix | 开源可观测 | UMAP 嵌入可视化 | 中 | 中 | 调试分析 |

2. 选型决策树（按团队规模 / 技术栈 / 预算推荐）
3. 前沿展望：Agentic RAG 评估、多跳评估

## 五、数据流

```
南瓜书 PDF（C7 已有）
    ↓
简单 RAG Pipeline（Notebook 2 搭建，后续复用）
    ↓
评估问题集（Notebook 4 构建，JSON 格式）
    ↓
Ragas 分层评估得分（Notebook 2 产出）
    ↓
LLM-as-Judge 评分（Notebook 3 产出）
    ↓
归因分析（Notebook 5 消费上述结果）
    ↓
CI 回归测试（Notebook 6 把上述流程自动化）
```

## 六、与 C5 的衔接策略

1. **不重复定义**：不解释"什么是评估""什么是 Bad Case""什么是大模型评估"
2. **不重复枚举**：不列举"检索有哪些常见失败模式"
3. **开头做衔接**：readme.md 用 2-3 句话说清"C5 讲了什么，C7.7 从哪里接上"
4. **交叉引用**：需要基础概念时，用链接指回 C5 对应章节

## 七、每个 Notebook 的统一模板

```
1. 前置知识提示（需要先学哪些章节）
2. 本节目标（一句话）
3. 理论部分（Markdown cells）
4. 环境准备（Code cells）
5. 代码实践（Code cells + 输出）
6. 小结
7. 实践建议
8. 参考文献
```

## 八、依赖库

```
ragas                    # 分层评估指标
deepeval                 # pytest 风格评估测试
langchain                # RAG Pipeline
langchain-community      # Embeddings, VectorStore
langchain-chroma         # Chroma 向量库
sentence-transformers    # bge-small-zh-v1.5
modelscope               # 模型下载
pymupdf                  # PDF 解析
scikit-learn             # Cohen's Kappa
matplotlib               # 可视化
```
