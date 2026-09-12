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

## 本实验的数据和审核边界

`data/dataset` 是本实验唯一事实源。最终数据包含 163 条南瓜书 query→evidence pair：101 train、28 dev、34 frozen test，分别来自 55、14、17 个互不重叠的 PDF 页面。

100 个候选来源页先按 BGE tokenizer 确定性切成 287 个、最多 384 tokens 的原文 chunk。生成、审核、训练和评估使用完全相同的 chunk，不按 query 或答案临时裁剪；每个 chunk 最多生成 1 道原文足以支持的问法，不强制每页凑齐四种类型。3 个生成结果连续未通过长度/schema 校验而被显式拒绝，2 个规范化重复问法被去重，余下 282 个候选中 `glm-4-flash` 初审拒绝 9 个；独立模型语义复核再剔除 110 个存在数学转写错误、对象错配、条件缺失、证据截断、问题不自足或循环回答等明确问题的候选，最终保留 163 条，覆盖其中 86 个来源页。二次剔除及逐条理由保存在 `data/semantic_review_exclusions.json`。全部保留记录明确 `human_verified=false`，不把模型审核写成人工标注。另有 5 个含解析替换符的 fixed chunk 只留在检索 corpus 中，既无 qrel 也不作为训练 positive，只能视为未标注的真实解析噪声，不能视为已确认负例。

split 按 page、`section_id` 和 `query_family_id` 隔离，不共享 qrel evidence。候选 answer 仅用于审核，训练 positive 始终是 canonical `evidence.quote`；训练、开发和测试采用同一任务定义，但来源页面不同。

## 运行实验

先按[教程首页的运行准备](../README.md#运行准备)使用 Python 3.10 与 `llm-universe-c7` kernel，进入 C7 根目录并安装 `requirements-c7.txt`。第一次运行前，在同一环境、网络可用时下载固定模型一次：

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"
```

这条命令只负责准备本地 Hugging Face 缓存；完成后 Notebook 和脚本都用 `local_files_only=True` 运行，缓存、依赖或数据缺失会直接失败，不切换模型、不使用 fallback。不要把联网下载放进评测闭环。

训练采用 `MultipleNegativesRankingLoss`（MNRL）：一个 batch 中第 `i` 个 query 的第 `i` 个真实 evidence 是正例，其他 evidence 是批内负例。分组 sampler 同时检查 positive ID 和完整 qrels，共享任一已知正例的 query 不会进入同一 batch，避免把相关 evidence 当成负例。当前 163 条 query→evidence 数据只支撑这次 MNRL 实验；Notebook 保留 MegaBatchMarginLoss、ContrastiveLoss、CosineSimilarityLoss、SoftmaxLoss、TripletLoss 和四种 Batch Triplet 损失的原理与选择条件，但不伪造这些损失在当前数据上的比较结果。

运行 [什么时候需要微调向量模型](什么时候需要微调向量模型.ipynb) 可重现完整实验：使用 `BAAI/bge-small-zh-v1.5`，在 dev 上选择训练配置，保存并重载选中模型，然后一次性评估 frozen test，并重新编码 287 个 canonical `fixed_token_chunk`，输出 Recall@1/3/5/10、MRR 与逐题变化。脚本入口是：

```bash
python scripts/run_embedding_finetune.py
```

上面的脚本也从 C7 根目录运行。Notebook 和脚本都只读取统一数据包，缺少依赖、缓存模型或数据时直接报错；章节目录下没有另一份 `requirements-c7.txt`。

### 目录中的示意图

- [固定长度分块与 overlap](figures/FixedSizedChunk.png) 用于解释相邻 chunk 的边界关系；它是概念示意，不替代本实验按页面和 BGE tokenizer 生成的 384-token `fixed_token_chunk`。
- [MSE 与 MAE 的曲线](figures/Loss.png) 说明常见回归损失对误差的敏感性；本实验实际训练的是 MNRL，因此不要把该图当作本次训练的 loss 或指标结果。
- [SBERT 与 SoftmaxLoss 流程](figures/SBERT_SoftmaxLoss.png) 展示句对分类式 SBERT 的输入与分类头；本实验使用 query→evidence 的 MNRL 排序，图示用于区分任务范式，并非本次实验的精确模型结构。

## 本次完整实验结果

BGE 向量模型微调实验中，dev 选择了 1 epoch、学习率 `2e-5`。三候选 dev 结果和选中候选的 loss/optimizer steps 已保存在 Notebook 的 `c2-compact-report-v2` 中；在此前未参与配置选择的 34 条 frozen test 上，Recall@10 从 0.9412 提升到 1.0000，Recall@5 从 0.8235 提升到 0.9706，Recall@3 保持 0.7941，Recall@1 从 0.5882 提升到 0.6471，MRR 从 0.6987 提升到 0.7591。逐题排名为 9 条改善、3 条退化、22 条不变，Notebook 同时持久化全部 34 条 baseline/finetuned rank 及退化 query ID。

因此，本例能展示一条真实的小规模微调闭环，但 101 条训练 pair 不代表生产数据已经充分；结论只适用于同一 `glm-4-flash` 生成并经模型复核的合成问法分布，不能外推为自然用户问法上的稳定收益，也不能忽略 3 条逐题退化。生产使用前应另收真实读者问题做独立评估；不能在看到 frozen test 后修改这份测试集。

[返回教程首页](../README.md)
