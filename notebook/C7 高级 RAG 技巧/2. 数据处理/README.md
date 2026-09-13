# 数据处理

本章从资料读取开始，逐步建立可检索的文本语料，并在证据质量稳定后比较分块、选择向量模型，最后用一个真实的 query→evidence 实验判断是否值得微调。

## 学习导航

1. [读取和清理不同格式的资料](读取和清理不同格式的资料.ipynb)
2. [常用分块方法](常用分块方法.ipynb)
3. [比较分块设置](比较分块设置.ipynb)
4. [分块优化](分块优化.ipynb)
5. [选择向量模型](选择向量模型.ipynb)
6. [什么时候需要微调向量模型](什么时候需要微调向量模型.ipynb)

微调 Notebook 按五种监督结构说明损失选择：query-positive、带二值/连续分数的句对、带类别的句对、显式正负三元组，以及带类别标签的单句。对应方法包括 MNRL、MegaBatchMarginLoss、ContrastiveLoss、CosineSimilarityLoss、SoftmaxLoss、TripletLoss 和四种 Batch Triplet 损失；本章对当前 query→evidence 数据采用 MNRL 做真实训练，其他方法保留原理与适用条件。

## 什么时候才需要微调

向量模型微调不是解析或索引问题的补丁。面对召回不足时，按下面顺序定位：

1. **先检查读取和清理。** 确认 PDF/HTML/Word 的文字没有乱码、页边界和标题层级没有丢失，重复页眉页脚和元数据没有混入正文。原文不完整时，先修解析。
2. **再检查分块。** 检查 chunk 是否在定义、条件和结论之间截断，长度和 overlap 是否适合任务，并用逐字证据验证边界。证据被切开时，先调分块策略。
3. **再检查检索系统。** 分别对 dense、BM25、混合检索和必要的重排做基线；确认索引使用的是最新完整 corpus，query/qrels 与评测集合一致。词面问题、过滤条件或索引错配，先修检索配置。
4. **只有剩下稳定的语义排序缺口时才微调。** 先构造与实际检索任务一致的 query→evidence 数据，按 page/section/query family 分组切分，使用真实 evidence 作为 positive；配置只在 dev 上选择，选定后才在 frozen test 上比较 Recall@1/3/5/10、MRR 和逐题排名。

## 可选微调实验

`data/dataset` 是本实验唯一事实源。最终数据包含 163 条南瓜书 query→evidence pair：101 train、28 dev、34 frozen test，分别来自 55、14、17 个互不重叠的 PDF 页面。split 按 page、`section_id` 和 `query_family_id` 隔离，不共享 qrel evidence；训练 positive 始终是 canonical `evidence.quote`。

100 个候选来源页被确定性切成 287 个原文 chunk，生成、审核、训练和评估共用这些 chunk。问法由 `glm-4-flash` 生成、初审，再由同一模型的另一次独立请求做语义复核；保留记录全部为 `human_verified=false`，不能写成人工标注。被剔除候选及逐条理由见 `data/semantic_review_exclusions.json`；完整字段、去重、解析噪声和数据检查规则见[数据说明](../data/README.md)与[维护说明](../docs/维护说明.md)。

### 运行实验

先按[教程首页的运行准备](../README.md#运行准备)使用 Python 3.10 与 `llm-universe-c7` kernel，进入 C7 根目录并安装 `requirements-c7.txt`。第一次运行前，在同一环境、网络可用时下载固定模型一次：

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"
```

这条命令只负责准备本地 Hugging Face 缓存；完成后 Notebook 和脚本都用 `local_files_only=True` 运行，缓存、依赖或数据缺失会直接失败，不切换模型、不使用 fallback。不要把联网下载放进评测闭环。

训练采用 `MultipleNegativesRankingLoss`（MNRL），并用分组 sampler 避免把共享已知正例的 query 放入同一 batch。当前数据只支撑这次 MNRL 实验；Notebook 仍完整介绍 MegaBatchMarginLoss、ContrastiveLoss、CosineSimilarityLoss、SoftmaxLoss、TripletLoss 和四种 Batch Triplet 损失的原理与选择条件，但不伪造它们在当前数据上的比较结果。

运行[什么时候需要微调向量模型](什么时候需要微调向量模型.ipynb)可重现完整闭环：使用 `BAAI/bge-small-zh-v1.5`，在 dev 上选择配置，保存并重载模型，重新编码 287 个 canonical chunk，最后一次性评估 frozen test。脚本入口是：

```bash
python scripts/run_embedding_finetune.py
```

脚本也从 C7 根目录运行。Notebook 和脚本只读取统一数据包；缺少依赖、缓存或数据时直接报错，不切换模型或使用 fallback。

### 目录中的示意图

- [固定长度分块与 overlap](figures/FixedSizedChunk.png) 用于解释相邻 chunk 的边界关系；它是概念示意，不替代本实验按页面和 BGE tokenizer 生成的 384-token `fixed_token_chunk`。
- [MSE 与 MAE 的曲线](figures/Loss.png) 说明常见回归损失对误差的敏感性；本实验实际训练的是 MNRL，因此不要把该图当作本次训练的 loss 或指标结果。
- [SBERT 与 SoftmaxLoss 流程](figures/SBERT_SoftmaxLoss.png) 展示句对分类式 SBERT 的输入与分类头；本实验使用 query→evidence 的 MNRL 排序，图示用于区分任务范式，并非本次实验的精确模型结构。

## 本次完整实验结果

BGE 向量模型微调实验中，dev 选择了 1 epoch、学习率 `2e-5`。三候选 dev 结果和选中候选的 loss/optimizer steps 已保存在 Notebook 的 `c2-compact-report-v2` 中；在此前未参与配置选择的 34 条 frozen test 上，Recall@10 从 0.9412 提升到 1.0000，Recall@5 从 0.8235 提升到 0.9706，Recall@3 保持 0.7941，Recall@1 从 0.5882 提升到 0.6471，MRR 从 0.6987 提升到 0.7591。逐题排名为 9 条改善、3 条退化、22 条不变，Notebook 同时持久化全部 34 条 baseline/finetuned rank 及退化 query ID。

因此，本例能展示一条真实的小规模微调闭环，但 101 条训练 pair 不代表生产数据已经充分；结论只适用于同一 `glm-4-flash` 生成、初审并经同一模型另一次独立请求复核的合成问法分布，且全部记录 `human_verified=false`，不能外推为自然用户问法上的稳定收益，也不能忽略 3 条逐题退化。生产使用前应另收真实读者问题做独立评估；不能在看到 frozen test 后修改这份测试集。

## 实践任务：为一份小资料建立可解释的分块基线

选一份不超过三页的自己的资料（也可以先用本章随附的 CSV、DOCX 或 PPTX fixture），保存不可变原件，再用本章方法生成两组分块：一组保留结构边界，另一组只改变一个 `chunk_size` 或 overlap。针对两个固定问题，记录目标页/片段排名、片段数量和交给回答模型的字符/token 数；先完成读取和边界抽查，再决定是否需要比较向量模型。

### 自检标准

- 每个 chunk 都有唯一 ID 和原文位置；标题、单位、条件与结论没有在清理时丢失。
- 两组设置只改一个变量，并为每个问题记录 Recall@k/首条相关排名、必要证据是否完整和上下文预算。
- 若证据被截断、解析有乱码或只有一组问题，结论必须写成“基线未建立”，不能直接进入向量微调。

[返回教程首页](../README.md)
