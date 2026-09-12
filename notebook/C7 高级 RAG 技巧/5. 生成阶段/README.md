# 检索后处理与回答约束

[排序、压缩与回答](排序、压缩与回答.ipynb)把这一章的四种处理放在同一个 Notebook 中。这里的“生成阶段”指把候选资料整理成可供回答的上下文并约束回答依据；固定示例用于教学检查。Notebook 末尾另有标为“可选”的真实 Cross-Encoder 同题比较，实际运行结果和范围写入 Notebook。

## 本章导航

1. [重新排序](排序、压缩与回答.ipynb)：候选中已有正确片段，但排得太后。
2. [上下文压缩](排序、压缩与回答.ipynb)：候选过长时，在字符预算内保留必要原文。
3. [来源标注](排序、压缩与回答.ipynb)：逐条核对结论，再附上实际匹配的页码。
4. [资料不足与拒答](排序、压缩与回答.ipynb)：候选没有支持时说明边界，不根据相似页面猜答案。

四个专题在 Notebook 中按上述顺序出现；它们可以独立使用，不是必须同时打开的一条“生成模型”流水线。

## 本章内容

1. 正确资料已经在候选结果中，但排得太后：重新排序。
2. 正确资料很长，必要结论会被截断：删去无关文字。
3. 回答已经写出：逐条核对原文并标明来源。
4. 资料中根本没有所需内容：不根据相似页面猜答案。

四种处理解决不同问题，不需要同时使用。“标注来源”比较同一段固定示例文字在添加页码前后的差别，并检查错页、漏引和没有原文依据的结论；这里没有调用或评估回答模型。若要接入真实生成器，应把它放在这些检查之后，并重新保存实际回答、引用和拒答结果。

本章的“重新排序”教学例子使用本地 BM25/关键词重排的教学排序器（启发式排序），不是真实 Cross-Encoder，也没有据此伪造模型效果；它保留候选集合不变，展示轻量基线如何改变顺序。Notebook 末尾的可选实验才加载 BAAI/bge-reranker-large，用同一个 canonical query、同一批候选和相同 top-k 做真实比较。

## 可选真实 Cross-Encoder 实验

Notebook 末尾的可选单元使用 BAAI/bge-reranker-large 与 LangChain CrossEncoderReranker。它只对当前 canonical SVM 问题的第一阶段 6 条候选打分，候选集合和 top-k 与轻量基线完全相同；结果保存模型、缓存位置、分数、必要页排名和本次改善/不变/退化，不能外推为整个数据集的效果。

fresh clone 先在 llm-universe-c7 kernel 对应环境，从 C7 唯一依赖入口准备依赖：

    python -m pip install -r "notebook/C7 高级 RAG 技巧/requirements-c7.txt"

上面的 requirements 已包含本 Notebook 实际导入的固定版本：`modelscope==1.33.0`、`langchain==0.3.27`、`langchain-community==0.3.30` 和 `langchain-core==0.3.76`；不要再为本节执行另一套独立的 pip 安装命令。

依赖安装完成后，再准备本实验需要的模型缓存：

    python -c "from modelscope import snapshot_download; print(snapshot_download('BAAI/bge-reranker-large'))"

Notebook 运行时只读取已缓存的指定模型（local_files_only=True）；依赖、配置或权重缺失会直接失败，不切换模型、不伪造输出，也不静默联网。跨编码器的分数、延迟和一次性单题观察范围都需结合自己的开发集重新评估。

建议沿着[引用来源示例](排序、压缩与回答.ipynb)和[资料不足时拒答示例](排序、压缩与回答.ipynb)继续阅读；这两个部分都要求回到候选原文核对，不把生成文字当作证据。完成本章后可进入[处理信息缺口](../6.%20处理信息缺口/README.md)或[评估](../7.%20评估/README.md)。

[返回教程首页](../README.md)
