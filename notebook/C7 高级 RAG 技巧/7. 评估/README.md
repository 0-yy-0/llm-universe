# 评估：method-fit 专题审计、general regression guard 与端到端收口

这一章说明怎样判断一次检索修改有没有用。先问清楚评估目的，再固定问题、证据语料、`top_k`、字符预算和 qrels；最后同时检查目标任务的适配收益与一般问题的副作用。不同专题的候选语料、问题和预算有各自用途，不能合并成一张跨方法排行榜。

## 先看两套互补证据

索引增强要分开回答两个问题：

1. **C3 method-fit 适配实验**：[比较索引增强方法](../3.%20索引阶段/比较索引增强方法.ipynb) 在同一批 3 道问题、同一组 9 页候选语料、同一个 BM25 检索器、`top_k=3` 和检索字段预算下，观察方法是否解决它要针对的问法。真实保存结果中，可靠章节标题（CCH）让“机器学习算法之间有没有绝对更好”这道目标题的必要页首条排名由 3 提到 2；Document Augmentation 让“集成多个弱学习器”这道目标题由 2 提到 1。它们是检索排名的 method-fit 证据，不是答案质量或全库收益结论。
2. **C7 general regression guard 一般回归护栏**：[比较改动前后](比较改动前后.ipynb) 在同一批 6 道 canonical 回归问题、同一证据池、`top_k=4` 和 `char_budget=1200` 下，只比较 BM25 与 `bm25_cch`。它检查加入 CCH 是否伤害一般问题，不是 CCH 正向效果演示。当前保存的两题退化（交叉验证可靠性、集成学习定义）必须保留，不能换题或用平均数掩盖。

当前 CCH 不应在这套索引上全局替代 BM25；局部适配仍需经过回归护栏。C3 的 3 题包含于 C7 的 6 题，二者不是独立泛化验证；结果差异还来自候选语料、证据粒度、`top_k` 和字符预算。C3 回答局部 method-fit，C7 回答更大证据池下的回归风险，不能直接横向合并。

## 三层评估边界

本章把三类工作分开，避免把一个专题 Notebook 的局部结果误读成全章结论：

1. **专题案例审计**：每个方法 Notebook 使用自己的固定输入，检查必要证据、回答要点和副作用；它们不是同一批题上的全方法横评。
2. **general regression guard**：本页的统一脚本只比较 BM25 与 CCH 检索排序，覆盖下面列出的 6 道 canonical 回归题；不覆盖回答模型，也不覆盖章节中其他专题方法。
3. **端到端 capstone**：[端到端验收](端到端验收.ipynb)按[验收协议](端到端验收协议.md)用少量 canonical 固定回归问题（evidence 已按固定流程审核）串起“问题 → 检索/补查 → 后处理 → 回答 → 引用或拒答”；四题统一由真实模型依据上下文判定 `answerable/insufficient`，验收的是闭环是否可追溯，不替代检索护栏。

## 问法分布边界

数据包中的 34 条 frozen-test 问法来自 17 个 PDF 页面上的固定 token chunks，每个 chunk 最多 1 题。问法由 `glm-4-flash` 生成和初审，再经独立模型语义复核；全部记录均为 `human_verified=false`。它们构成有限的合成问法分布，因此本章的数字和案例只能说明这组 canonical 数据，不能外推为自然用户集覆盖。

## 直接运行 C7 general regression guard（仅 BM25/CCH）

`比较改动前后.ipynb` 调用统一脚本，固定 6 道同时标为 `demo` 和 `regression` 的 canonical 问题：

```text
model_selection_with_intro_scope
cross_validation_reliability
ensemble_learning_definition
lda_recursive_derivation
newton_methods_comparison
model_evaluation_followup
```

脚本入口位于 `scripts/run_benchmark.py`，两个工作目录都可以运行：

```bash
# 仓库根目录
python "notebook/C7 高级 RAG 技巧/scripts/run_benchmark.py"

# C7 教程根目录
cd "notebook/C7 高级 RAG 技巧"
python scripts/run_benchmark.py
```

缺少 canonical 数据或必要依赖时命令直接失败，不会切换到 PDF、向量库、外部模型或其他问题文件。需要更适合程序读取的结果时加 `--json`。

## 当前护栏协议

护栏包含两个当前项目已有方法：

| 方法 | 检索字段 |
| --- | --- |
| `bm25` | canonical evidence 原文 `quote` |
| `bm25_cch` | 可靠章节标题（CCH）+ canonical evidence 原文 |

两者均使用 `top_k=4` 和 `char_budget=1200`。检索函数只接收问题文字与 evidence 的 `evidence_id/page/quote` 投影；所有方法完成排序后才读取 canonical `qrels` 的 `relevance=1`，因此 `reference_answer`、`expected_pages` 和 `reference_claims` 不会成为排序特征。canonical evidence 都由 checker 验证为 PDF 上可逐字定位的原文，但它们是围绕教程问题整理的小型标注池，不是完整 PDF 分块库；本实验只衡量池内排序变化，不能外推为全书召回率。回答上下文若要继续接入，只能从命中 evidence 的原文 `quote` 组成，并受同一字符预算约束。

每道题保存首条相关 evidence 的 `rank`、相关 evidence `coverage`、`Recall@4`、`MRR`、命中 ID/页码、上下文字符数以及相对 BM25 的 `improved`、`unchanged`、`degraded` 或 `tradeoff`。汇总同时报告平均 Recall@4、MRR 和四类结果计数；退化不会被删除或用平均值掩盖。

本次保存的真实 Notebook 输出显示：BM25 平均 Recall@4 为 0.750、MRR 为 0.542；CCH 平均 Recall@4 为 0.500、MRR 为 0.500。相对 BM25，CCH 在 4 道题上不变、2 道题退化，没有改善题。交叉验证可靠性题的 Recall@4 从 1.000 降到 0.500，集成学习定义题从 1.000 降到 0.000；LDA 连续推导题两种方法都未命中相关 qrels。因此当前结果只能说明这组 canonical 题上的 CCH 退化与不变，不能表述成全面提升。

BGE 微调实验是在 34 条 frozen test query→evidence 上比较同一向量模型微调前后的 Recall@K、MRR 与逐题排名；CCH general regression guard 则在固定 6 道 canonical 问题和同一证据池上比较 BM25 与 `bm25_cch`。两者的干预对象、样本集合与指标口径不同，不能合并成一个整体收益数字，只能分别保留各自结论。

## 先完整走一次

1. [C3 可选横向实验](../3.%20索引阶段/比较索引增强方法.ipynb)：先看 3 道目标题上的 method-fit 适配证据。
2. [比较改动前后](比较改动前后.ipynb)：再跑 6 道一般问题的 C7 regression guard，保留每道题的退化。
3. [选择一个问题](选择一个问题.ipynb)：理解如何保存单题的原始检索结果与证据范围。
4. [分开检查问题、资料与回答](分开检查问题、资料与回答.ipynb)：把问题、检索资料和最终回答分开核对。

完成上面四页后，若要验收完整闭环，继续执行[端到端验收](端到端验收.ipynb)，并对照[端到端验收协议](端到端验收协议.md)。普通教学运行不要求记录模型版本、执行时间或源码 revision；只有长期回归、线上监控或跨版本比较时，才记录能解释差异的必要字段。

## 再按需要阅读

1. [计算检索和回答指标](计算检索和回答指标.ipynb)：了解页面命中、排名、回答完整性和原文依据。
2. [用模型辅助检查回答](用模型辅助检查回答.ipynb)：让模型按明确规则评分，再与 canonical 固定回归的已审核判定对照。
3. [建设评估问题集](建设评估问题集.ipynb)：了解问题、evidence 和 qrels 如何进入 canonical 数据包。
4. [分析问题出在哪一步](分析问题出在哪一步.ipynb)：把失败定位到资料范围、分块、检索或回答。
5. [持续观察教程和线上结果](持续观察教程和线上结果.ipynb)：用本地确定性指标做文件/回归检查，并按统一数据契约选择 Ragas、LLM Judge、DeepEval/CI 和 TruLens/tracing 等接入路径；第三方框架在本页只做选型说明，不伪造运行结果。

本章以本地计算、canonical 固定回归和已审核结果核对为主，不要求额外的评估框架。检索指标只能说明相关证据是否被找到，不能单独推出生成答案正确率；回答结论仍需回到原文逐条核对。

本章使用的问题来自[canonical 数据包](../data/README.md)。

[返回教程首页](../README.md)
