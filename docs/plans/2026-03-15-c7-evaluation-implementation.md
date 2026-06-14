# C7.7 工业级 RAG 评估与迭代 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 llm-universe 教程的 C7 章节创建完整的工业级 RAG 评估模块，包含 2 个 md 文件、4 个可运行 Notebook 和配套数据。

**Architecture:** 按评估链路组织：概述 → 分层指标 → LLM-as-Judge → 评估集工程 → 错误归因 → CI/CD → 工具选型。所有 Notebook 共享同一个 RAG Pipeline（南瓜书 + bge-small-zh-v1.5 + Chroma + glm-4-flash），数据和代码逐步复用。

**Tech Stack:** Python 3.10+, LangChain, Chroma, Ragas, DeepEval, 智谱AI glm-4-flash, BAAI/bge-small-zh-v1.5, matplotlib

---

### Task 1: 创建目录结构和 readme.md

**Files:**
- Create: `notebook/C7 高级 RAG 技巧/7. 评估/readme.md`
- Create: `notebook/C7 高级 RAG 技巧/7. 评估/data/` (目录)
- Create: `notebook/C7 高级 RAG 技巧/7. 评估/figures/` (目录)

**Step 1: 创建目录**

```bash
mkdir -p "notebook/C7 高级 RAG 技巧/7. 评估/data"
mkdir -p "notebook/C7 高级 RAG 技巧/7. 评估/figures"
```

**Step 2: 写 readme.md**

内容要点：
- 开头衔接 C5："C5 介绍了基于 Bad Case 的验证迭代方法（人工评估、大模型自动评估），适用于项目原型。本章聚焦生产环境的系统化评估，解决三个新挑战：规模、持续性、可归因性。"
- 本章各节一句话简介与链接
- 前置知识：需学完 C5、C7.2-C7.6

**Step 3: 确认文件存在**

```bash
ls -la "notebook/C7 高级 RAG 技巧/7. 评估/"
```

---

### Task 2: 编写 1.工业级RAG评估体系概述.md

**Files:**
- Create: `notebook/C7 高级 RAG 技巧/7. 评估/1.工业级RAG评估体系概述.md`

**Step 1: 编写 md 文件**

结构（约 800 字 + 2 张图表）：

```markdown
# 工业级 RAG 评估体系概述

## 一、从原型验证到生产评估

生产环境评估 vs 原型验证的 3 个核心差异表格：
| 维度 | C5 原型验证 | C7 生产评估 |
|------|-----------|-----------|
| 规模 | 几十条 Bad Case | 数百到上千条分桶评估集 |
| 持续性 | 一次性人工检查 | 每次改动自动回归 |
| 可归因性 | 整体打分 | 分层定位到具体环节 |

## 二、评估体系全景图

流程图：分层指标 → LLM-as-Judge → 评估集 → 归因 → CI/CD → 在线监控

## 三、RAG 三元组

- Context Relevance：检索到的上下文和问题的相关性
- Groundedness：生成的答案是否基于检索到的上下文
- Answer Relevance：生成的答案和问题的相关性

目标阈值：Faithfulness ≥ 0.80, Answer Relevance ≥ 0.85

## 四、2026 工具生态

6 个工具对比表（Ragas / DeepEval / TruLens / LangSmith / Langfuse / Arize Phoenix）
维度：类型、核心能力、CI 集成、中文支持、适合场景

## 五、本章学习路线

各节一句话 + 推荐阅读顺序
```

---

### Task 3: 编写 2.RAG三元组与分层指标.ipynb

**Files:**
- Create: `notebook/C7 高级 RAG 技巧/7. 评估/2.RAG三元组与分层指标.ipynb`

**Step 1: 创建 Notebook，按以下 cell 结构编写（约 30 cells）**

**Markdown Cell 1: 标题与前置知识**
```markdown
# RAG 三元组与分层指标

前置知识：C5 系统评估与优化、C7.2-C7.6

本节目标：理解 RAG 三元组指标体系，使用 Ragas 对 RAG pipeline 产出分层评估报告。
```

**Markdown Cell 2: RAG 三元组理论**
```markdown
## 一、RAG 三元组（RAG Triad）

RAG 系统的评估围绕三个核心元素：Query（用户问题）、Context（检索到的上下文）、Response（生成的答案）。
两两之间形成三个评估维度：

### 1. Context Relevance（上下文相关性）
检索到的文档片段是否和用户问题相关。
- 计算方式：对每个检索到的 chunk，用 LLM 判断其与 query 的相关程度，取均值
- 目标阈值：≥ 0.75
- 低分原因：检索器召回了无关内容，或 embedding 质量不足

### 2. Groundedness / Faithfulness（基础性 / 忠实度）
生成的答案是否有检索到的上下文作为事实依据，而非凭空编造。
- 计算方式：将答案拆分为多个 claim，逐个验证是否能在 context 中找到依据
- 目标阈值：≥ 0.80
- 低分原因：LLM 幻觉，或 context 不充分导致 LLM 自行补充

### 3. Answer Relevance（答案相关性）
生成的答案是否回答了用户的问题。
- 计算方式：用 LLM 判断 response 和 query 的匹配程度
- 目标阈值：≥ 0.85
- 低分原因：答非所问，或回答了 context 中的其他内容
```

**Markdown Cell 3: 分层指标体系**
```markdown
## 二、分层指标体系

除 RAG 三元组外，生产环境还需关注更细粒度的分层指标：

### 检索层
| 指标 | 含义 | 公式 |
|------|------|------|
| Context Precision | 检索结果中相关文档的占比 | 相关 chunks / 总 chunks |
| Context Recall | 标准答案中的关键信息被检索到的比例 | 被覆盖的关键点 / 总关键点 |

### 生成层
| 指标 | 含义 |
|------|------|
| Faithfulness | 答案中每个 claim 是否有 context 依据 |
| Answer Relevancy | 答案是否切题 |
| Answer Correctness | 答案与标准答案的语义相似度 |

### 系统层
| 指标 | 含义 |
|------|------|
| 延迟 | 检索 + 生成的端到端耗时 |
| Token 成本 | 每次问答消耗的 token 数 |
| 失败率 | 超时 / 异常 / 空返回的比例 |
```

**Code Cell 4: 环境准备**
```python
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from modelscope import snapshot_download
model_dir = snapshot_download('BAAI/bge-small-zh-v1.5')
print(f"✅ Embedding 模型下载完成: {model_dir}")
```

**Code Cell 5: 初始化 Embedding 和 LLM**
```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatZhipuAI

embedding = HuggingFaceEmbeddings(model_name=model_dir)

api_key = os.environ.get("ZHIPUAI_API_KEY")
llm = ChatZhipuAI(
    model="glm-4-flash",
    temperature=0.0,
    api_key=api_key
)
print("✅ Embedding 和 LLM 初始化完成")
```

**Code Cell 6: 构建 RAG Pipeline**
```python
import re
import json
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

pdf_path = "../3. 索引阶段/data/pumpkin_book.pdf"

def clean_text(text: str) -> str:
    text = re.sub(r'→_→\n.*?←_←', '', text, flags=re.DOTALL)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

loader = PyMuPDFLoader(pdf_path)
pdf_pages = loader.load()
data_pages = pdf_pages[13:-13]

for page in data_pages:
    page.page_content = clean_text(page.page_content)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(data_pages)

vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

print(f"✅ RAG Pipeline 构建完成，共 {len(splits)} 个文档块")
```

**Code Cell 7: 构建问答链**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_template(
    "根据以下上下文回答问题。如果上下文中没有相关信息，请说'根据已有资料无法回答'。\n\n"
    "上下文：\n{context}\n\n"
    "问题：{question}\n\n"
    "回答："
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("什么是信息增益？")
print(f"测试回答：{response[:200]}")
```

**Code Cell 8: 准备评估数据集**
```python
eval_data = [
    {
        "question": "什么是信息增益？",
        "ground_truth": "信息增益是指在得知某个特征的信息后，信息不确定性减少的程度。在决策树中，信息增益越大，说明该特征对分类的贡献越大。"
    },
    {
        "question": "什么是基尼指数？",
        "ground_truth": "基尼指数是度量数据集纯度的一种指标，反映了从数据集中随机抽取两个样本，其类别标记不一致的概率。基尼指数越小，数据集纯度越高。"
    },
    {
        "question": "过拟合和欠拟合的区别是什么？",
        "ground_truth": "过拟合是模型在训练集上表现好但在测试集上表现差，学到了训练数据中的噪声。欠拟合是模型在训练集和测试集上都表现不好，没有学到数据的基本规律。"
    },
    {
        "question": "什么是支持向量机？",
        "ground_truth": "支持向量机是一种二分类模型，通过在特征空间中找到一个最优超平面来实现分类，使得两类样本到超平面的间隔最大化。"
    },
    {
        "question": "朴素贝叶斯分类器的基本原理是什么？",
        "ground_truth": "朴素贝叶斯分类器基于贝叶斯定理，假设各特征之间条件独立，通过计算后验概率来进行分类，选择后验概率最大的类别作为预测结果。"
    },
]
print(f"✅ 准备了 {len(eval_data)} 条评估数据")
```

**Code Cell 9: 生成 RAG 回答用于评估**
```python
from tqdm import tqdm

results = []
for item in tqdm(eval_data, desc="生成回答"):
    docs = retriever.invoke(item["question"])
    answer = rag_chain.invoke(item["question"])
    results.append({
        "question": item["question"],
        "answer": answer,
        "contexts": [doc.page_content for doc in docs],
        "ground_truth": item["ground_truth"]
    })

print(f"✅ 已生成 {len(results)} 条回答")
print(f"\n示例：\n问题：{results[0]['question']}\n回答：{results[0]['answer'][:200]}")
```

**Code Cell 10: 安装 Ragas**
```python
# !pip install ragas
```

**Code Cell 11: 使用 Ragas 进行分层评估**
```python
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

eval_dataset = Dataset.from_dict({
    "question": [r["question"] for r in results],
    "answer": [r["answer"] for r in results],
    "contexts": [r["contexts"] for r in results],
    "ground_truth": [r["ground_truth"] for r in results],
})

ragas_llm = LangchainLLMWrapper(llm)
ragas_emb = LangchainEmbeddingsWrapper(embedding)

eval_result = evaluate(
    dataset=eval_dataset,
    metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
    llm=ragas_llm,
    embeddings=ragas_emb,
)

print("=" * 50)
print("分层评估报告")
print("=" * 50)
for metric, score in eval_result.items():
    print(f"  {metric}: {score:.4f}")
```

**Code Cell 12: 可视化雷达图**
```python
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

metrics = list(eval_result.keys())
scores = list(eval_result.values())

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
scores_plot = scores + [scores[0]]
angles += angles[:1]
metrics_plot = metrics + [metrics[0]]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, scores_plot, alpha=0.25, color='steelblue')
ax.plot(angles, scores_plot, 'o-', linewidth=2, color='steelblue')
ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
ax.set_ylim(0, 1)
ax.set_title("RAG 分层评估雷达图", pad=20, fontsize=14)
plt.tight_layout()
plt.savefig("./figures/ragas_radar.png", dpi=150, bbox_inches='tight')
plt.show()
```

**Markdown Cell 13: 各指标解读**
```markdown
## 三、评估结果解读

### 怎么看这些分数？

| 指标 | 得分区间 | 含义 |
|------|---------|------|
| Context Precision | ≥ 0.80 优 / 0.60-0.80 可 / < 0.60 差 | 检索结果是否精准 |
| Context Recall | ≥ 0.75 优 / 0.50-0.75 可 / < 0.50 差 | 关键信息是否被检索到 |
| Faithfulness | ≥ 0.80 优 / 0.60-0.80 可 / < 0.60 差 | 答案是否有据可查 |
| Answer Relevancy | ≥ 0.85 优 / 0.70-0.85 可 / < 0.70 差 | 答案是否切题 |
```

**Code Cell 14: 逐条查看详细得分**
```python
df = eval_result.to_pandas()
print(df.to_string(index=False))
```

**Markdown Cell 15: 小结与实践建议**
```markdown
## 四、小结

本节介绍了 RAG 三元组和分层指标体系，并使用 Ragas 对 RAG Pipeline 进行了分层评估。

### 实践建议
1. **先定阈值再优化**：为每个指标设定目标阈值，低于阈值的重点优化
2. **检索优先**：如果 Context Precision/Recall 低，先优化检索再调 Prompt
3. **定期跑评估**：每次改动后跑一轮，避免改好了 A 却破坏了 B
4. **指标不是万能的**：数值高不代表用户满意，数值低也不代表一定有问题，需结合人工抽检

### 参考文献
- [Ragas 官方文档](https://docs.ragas.io/)
- [RAG Evaluation: 2026 Metrics and Benchmarks](https://labelyourdata.com/articles/llm-fine-tuning/rag-evaluation)
```

---

### Task 4: 编写 3.LLM-as-Judge评估.ipynb

**Files:**
- Create: `notebook/C7 高级 RAG 技巧/7. 评估/3.LLM-as-Judge评估.ipynb`

**Step 1: 创建 Notebook，按以下结构编写（约 35 cells）**

**核心 Markdown cells 内容要点：**

1. 标题 + 前置知识 + 本节目标
2. 理论：LLM-as-Judge 的原理
   - 核心思想：用 LLM 充当评估员，替代昂贵的人工标注
   - 和 C5 的区别：C5 讲了概念，这里讲设计方法、操作级评估、成本控制
3. Judge Prompt 设计原则
   - 直接打分 vs CoT（带推理链）打分
   - 评分锚定：每一档都有明确定义
   - 示例：Context Relevance Judge Prompt（0-10 分，含评分标准）
4. 操作级评估：检索和生成分开评
5. 采样策略：不是每条都评，按比例控制成本
6. 校准方法：Cohen's Kappa

**核心 Code cells 内容要点：**

1. 复用 Task 3 的 RAG Pipeline 和评估数据
2. 编写 `context_relevance_judge` 函数：
   - 输入 question + context
   - 用 glm-4-flash 打分（0-10）
   - 返回分数 + 推理过程
3. 编写 `faithfulness_judge` 函数：
   - 输入 context + answer
   - 用 glm-4-flash 判断答案中每个 claim 是否有依据
4. 对比实验：直接打分 vs CoT 打分
   - 对同一批数据分别用两种 prompt
   - 比较分数分布差异
5. 批量评估 + 按题目类型聚合
6. Cohen's Kappa 计算（用 sklearn）：
   - 模拟一组人工标注分数
   - 和 Judge 分数做一致性计算
7. 小结：何时用 LLM-as-Judge，何时需人工

---

### Task 5: 编写 4.评估集工程.ipynb

**Files:**
- Create: `notebook/C7 高级 RAG 技巧/7. 评估/4.评估集工程.ipynb`
- Create: `notebook/C7 高级 RAG 技巧/7. 评估/data/eval_questions.json`

**Step 1: 创建 Notebook，按以下结构编写（约 25 cells）**

**核心 Markdown cells 内容要点：**

1. 标题 + 前置知识 + 目标
2. 评估集三种类型（Golden Set / Regression Set / Hard Set）
3. 分桶策略：事实问答 / 推理题 / 模糊问题 / 拒答场景
4. 持续更新机制

**核心 Code cells 内容要点：**

1. 用 LLM 基于南瓜书内容自动生成分桶评估问题
   - Prompt：给定一段文本，生成指定类型的问题 + 答案
   - 类型：factual / reasoning / ambiguous / unanswerable
2. 构建 JSON 评估集，结构：
   ```json
   {
     "question": "...",
     "ground_truth": "...",
     "bucket": "factual",
     "difficulty": "easy",
     "source_page": 42
   }
   ```
3. 保存到 `data/eval_questions.json`
4. 用 Ragas + LLM-as-Judge 跑评估
5. 按桶聚合得分
6. 找出弱点桶，给出优化建议
7. 小结

---

### Task 6: 编写 5.错误归因与根因分析.ipynb

**Files:**
- Create: `notebook/C7 高级 RAG 技巧/7. 评估/5.错误归因与根因分析.ipynb`

**Step 1: 创建 Notebook，按以下结构编写（约 25 cells）**

**核心 Markdown cells 内容要点：**

1. 标题 + 前置知识 + 目标
2. 归因三步法（看分层指标 → 层内细分 → 确认根因）
3. 归因矩阵表格
4. 从归因到行动

**核心 Code cells 内容要点：**

1. 加载前面 Notebook 的评估结果
2. 取得分最低的 5 个 case
3. 逐个展示：question / retrieved contexts / answer / 各指标得分
4. 编写自动归因函数：
   ```python
   def diagnose(case):
       if case["context_recall"] < 0.5:
           return "检索召回不足：数据覆盖或 embedding 问题"
       if case["context_precision"] < 0.5:
           return "检索精度不足：噪声文档过多"
       if case["faithfulness"] < 0.6:
           return "生成幻觉：LLM 未基于上下文回答"
       if case["answer_relevancy"] < 0.7:
           return "答非所问：Prompt 或 LLM 理解问题"
       return "综合表现可接受"
   ```
5. 输出归因报告表格
6. 小结 + 各根因对应的修复方向

---

### Task 7: 编写 6.CI_CD质量门禁与在线监控.ipynb

**Files:**
- Create: `notebook/C7 高级 RAG 技巧/7. 评估/6.CI_CD质量门禁与在线监控.ipynb`

**Step 1: 创建 Notebook，按以下结构编写（约 30 cells）**

**核心 Markdown cells 内容要点：**

1. 标题 + 前置知识 + 目标
2. 离线评估：回归测试，每次改动后自动跑评估集
3. 质量门禁：Faithfulness ≥ 0.80 才能合并
4. 在线监控：追问率、转人工率、用户反馈
5. 全链路版本化

**核心 Code cells 内容要点：**

1. 安装 DeepEval：`pip install deepeval`
2. 用 DeepEval 定义 pytest 风格测试用例：
   ```python
   from deepeval import assert_test
   from deepeval.test_case import LLMTestCase
   from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

   def test_faithfulness():
       test_case = LLMTestCase(
           input="什么是信息增益？",
           actual_output=rag_chain.invoke("什么是信息增益？"),
           retrieval_context=[...],
       )
       metric = FaithfulnessMetric(threshold=0.8, model="glm-4-flash", ...)
       assert_test(test_case, [metric])
   ```
3. 模拟 CI 流程：
   - v1 prompt → 跑评估 → 记录得分
   - 修改 prompt → v2 prompt → 跑评估 → 记录得分
   - 对比表格
4. 质量门禁判断逻辑
5. 版本对比报告输出
6. 在线指标采集示例（记录每次问答的延迟、token 数、是否触发拒答）
7. 小结

---

### Task 8: 编写 7.工具对比与选型.md

**Files:**
- Create: `notebook/C7 高级 RAG 技巧/7. 评估/7.工具对比与选型.md`

**Step 1: 编写 md 文件**

结构（约 600 字 + 2 张表）：

```markdown
# 工具对比与选型

## 一、主流工具对比

| 工具 | 类型 | 核心能力 | CI 集成 | 中文支持 | 适合场景 |
|------|------|---------|---------|---------|---------|
| Ragas | 开源指标库 | 组件级 RAG 指标最全 | 中 | 好 | 离线评估、快速实验 |
| DeepEval | 开源测试框架 | pytest 风格，50+ 指标 | 强 | 好 | CI/CD 集成 |
| TruLens | 开源评估框架 | RAG Triad，可编程反馈函数 | 中 | 中 | 实验对比、仪表盘 |
| LangSmith | 商业平台 | LangChain 生态，实验管理 | 强 | 好 | LangChain 用户 |
| Langfuse | 开源可观测 | 操作级评估，OpenTelemetry | 强 | 好 | 在线监控、生产追踪 |
| Arize Phoenix | 开源可观测 | UMAP 嵌入可视化 | 中 | 中 | 调试分析 |

## 二、选型建议

按团队规模/技术栈/预算的决策树：
- 个人/小团队 + 快速验证 → Ragas
- 有 CI/CD 流程 → DeepEval
- LangChain 技术栈 → LangSmith
- 需要生产监控 → Langfuse
- 全面评估 + 仪表盘 → TruLens

## 三、前沿展望

- Agentic RAG 评估：传统指标不适用于自主决策的 Agent
- 多跳推理评估：需要 hop-aware 的逐步验证
- GraphRAG 评估：迭代检索的精度/召回权衡
```

---

### Task 9: 清理旧文件并验证

**Files:**
- Review: `notebook/C7 高级 RAG 技巧/7. 评估/` 目录下所有文件

**Step 1: 检查所有新文件是否创建成功**

```bash
ls -la "notebook/C7 高级 RAG 技巧/7. 评估/"
```

**Step 2: 确认旧文件的处理方式**

保留旧文件（`1.使用TruLens评估RAG应用.md`、`rag_eval.ipynb` 等）作为参考，不删除。新文件的编号（1-7）会自然排在前面。如果需要区分，可以将旧文件移入 `archive/` 子目录。

**Step 3: 验证所有 Notebook 可以正常打开**

在 Jupyter 中逐个打开确认格式正确。
