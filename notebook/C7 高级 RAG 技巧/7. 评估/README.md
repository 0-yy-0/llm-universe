# 评估：判断改动是否有用

本章把问题、检索资料和回答分开检查，再用固定问题比较改动。你需要同时回答两件事：目标问题有没有改善，原本正常的问题有没有退化。比较前固定问题、证据语料、`top_k`、字符预算和相关性标注 qrels，结果才能解释清楚。

## 学习路径

1. [选择一个问题](选择一个问题.ipynb)：保存一次检索的原始结果与资料范围。
2. [分开检查问题、资料与回答](分开检查问题、资料与回答.ipynb)：先检查问题条件，再看资料缺口，最后核对回答依据。
3. [计算检索和回答指标](计算检索和回答指标.ipynb)：理解命中、排名、必要内容覆盖与原文支持的差别。
4. [比较改动前后](比较改动前后.ipynb)：在固定 6 题上比较 BM25/CCH，保留逐题变化。
5. [分析问题出在哪一步](分析问题出在哪一步.ipynb)：根据失败记录决定下一步改资料、检索还是回答。
6. [端到端验收](端到端验收.ipynb)：按[验收协议](端到端验收协议.md)运行 4 道固定题，把检索、上下文、真实回答、引用与资料不足处理连起来。

## 三种实验分别说明什么

| 实验 | 固定的范围 | 能回答的问题 |
|---|---|---|
| [C3 索引增强比较](../3.%20索引阶段/比较索引增强方法.ipynb) | 3 道适配题、9 页候选语料、BM25、`top_k=3`、同一检索字段预算 | 方法是否适合它针对的问法（method-fit） |
| [本章改动前后比较](比较改动前后.ipynb) | 6 道回归题、同一 evidence 池、`top_k=4`、`char_budget=1200` | 加入 CCH 是否伤害一般问题（general regression guard） |
| [端到端验收](端到端验收.ipynb) | 4 道固定题、实际检索上下文、真实模型回答与独立语义检查 | 整个流程能否保留依据，并在资料不足时拒答 |

C3 保存的局部适配结果中，可靠章节标题（CCH）让“机器学习算法之间有没有绝对更好”这道题的必要页首条排名由 3 提到 2；Document Augmentation 让“集成多个弱学习器”这道题由 2 提到 1。这是检索排名改善，回答质量仍需另查。

C3 的 3 题包含于本章的 6 题，二者并非独立的泛化验证；候选语料、证据粒度、`top_k` 和预算也不同。各专题 Notebook 同样使用各自的输入范围，其改善、持平、退化与不适用结果应留在对应实验中，不能合并成一张跨方法排行榜。

## 运行固定 6 题的检索比较

先按[教程首页](../README.md#运行准备)使用 Python 3.10 与 `llm-universe-c7` kernel，进入 C7 根目录并安装 `requirements-c7.txt`。之后在同一目录运行：

```bash
python scripts/run_benchmark.py
```

需要逐题 JSON 时加 `--json`。脚本只读取统一数据包，缺少数据或依赖时直接报错。这 6 道问题同时标为 `demo` 和 `regression`：

```text
model_selection_with_intro_scope
cross_validation_reliability
ensemble_learning_definition
lda_recursive_derivation
newton_methods_comparison
model_evaluation_followup
```

两种方法分别检索原文 `quote`（`bm25`）和“可靠章节标题 + 原文”（`bm25_cch`）。排序器只接收问题文字与 evidence 的 `evidence_id/page/quote`；排序结束后才读取 `qrels.relevance=1` 计算指标。`reference_answer`、`expected_pages`、`reference_claims` 不参与排序。后续若组装回答上下文，只能使用命中的原文 `quote`，并遵守同一字符预算。

每题保存首条相关证据排名、相关证据覆盖、Recall@4、MRR、命中 ID/页码、上下文字符数，以及相对 BM25 的 `improved`、`unchanged`、`degraded` 或 `tradeoff`；汇总同时保留平均值和四类结果计数。

## 结果与限制

本次保存的真实 Notebook 输出显示：BM25 平均 Recall@4 为 0.750、MRR 为 0.542；CCH 平均 Recall@4 为 0.500、MRR 为 0.500。相对 BM25，CCH 在 4 道题上不变、2 道题退化，没有改善题。交叉验证可靠性题的 Recall@4 从 1.000 降到 0.500，集成学习定义题从 1.000 降到 0.000；LDA 连续推导题两种方法都未命中相关 qrels。因此当前结果只能说明这组 canonical 题上的 CCH 退化与不变，不能表述成全面提升。

BGE 微调实验是在 34 条 frozen test query→evidence 上比较同一向量模型微调前后的 Recall@K、MRR 与逐题排名；CCH general regression guard 则在固定 6 道 canonical 问题和同一证据池上比较 BM25 与 `bm25_cch`。两者的干预对象、样本集合与指标口径不同，不能合并成一个整体收益数字，只能分别保留各自结论。

当前结果不支持在这套索引上全局用 CCH 替代 BM25。evidence 虽能由检查器逐字定位到 PDF，却是围绕教程问题整理的小型标注池，不能把池内排序结果解释为全书召回率。检索指标也不能单独推出生成答案正确率；回答仍要回到原文逐条核对。

微调数据包的 34 条 frozen test 问法来自 17 个 PDF 页面上的固定 token chunks，每个 chunk 最多 1 题。问法由 `glm-4-flash` 生成和初审，再经独立模型语义复核，保留记录均为 `human_verified=false`。这是有限的合成问法分布，与本章 6 道回归题用途不同，均不能外推为自然用户集覆盖。

## 扩展：评审、问题集和持续观察

- [用模型辅助检查回答](用模型辅助检查回答.ipynb)：按明确规则评审，并与固定回归题的已审核判定对照。
- [建设评估问题集](建设评估问题集.ipynb)：理解问题、evidence、qrels 与使用范围怎样进入统一数据包。
- [持续观察教程和线上结果](持续观察教程和线上结果.ipynb)：先运行本地检查，再按需求接入 LLM Judge、Ragas、CI/DeepEval 或 tracing/TruLens。五类接入共用问题、证据、回答记录，第三方框架在该页仅作接入说明。

普通教学运行保存本单元实际输入和输出即可；长期回归、线上观察或跨版本比较时，再记录能解释差异的数据、索引、模型、Prompt 与评分配置。

本章使用的问题来自[统一数据包](../data/README.md)。

[返回教程首页](../README.md)
