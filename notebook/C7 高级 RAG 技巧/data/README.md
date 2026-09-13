# C7 数据资产

本目录是 C7 主线的事实边界。南瓜书 PDF、配套向量库和 `dataset/` canonical 数据包来自当前仓库；可执行 Notebook 不从 Git 历史、临时文件或外部资料补写事实。

## 南瓜书资料和向量库

随附 PDF 是 Datawhale 南瓜书项目 v1.9.9，许可为 [CC BY-NC-SA 4.0](https://github.com/datawhalechina/pumpkin-book/blob/master/LICENSE)。PDF SHA-256、向量库关键文件、SQLite collection、维度和记录数登记在 [`asset_manifest.json`](asset_manifest.json)，可由教程 checker 复核。向量库是这份 PDF 的分块文字和向量，不是另一份资料来源；变更 PDF、分块方式或向量模型后必须重建并同步清单。

## Canonical 数据包

`dataset/` 是 C7 唯一的 canonical 数据包，包含问题、原文 evidence、qrels、会话、split 和审核记录。运行时通过 `common/dataset.py` 严格加载：

1. `documents.jsonl` 登记原始资料、许可和 SHA-256。
2. `evidence.jsonl` 保存可追溯的页码、原文片段和规范化 offsets。
3. `queries.jsonl` 保存问题、可回答性、用途和 reference claims。
4. `qrels.jsonl` 将问题绑定到 evidence；`splits.json` 按 section 和 query family 防止泄漏。
5. `sessions.jsonl` 保存连续追问及会话边界，`annotations/` 保存候选与审核状态。

问题集的加载、引用完整性、空训练问题、重复 ID、训练正例水印和 split 泄漏由 `common/dataset.py` 与 `scripts/check_dataset.py` 检查。评估 Notebook 只能通过 canonical loader 读取这些记录；参考答案和 expected evidence 只用于结果检查，不能提前注入检索上下文。

当前数据包包含 163 条 query→evidence pair：101 train、28 dev、34 frozen test，分别覆盖 55、14、17 个互不重叠的 PDF 页面。问题由 `glm-4-flash` 生成和初审，再由同一 `glm-4-flash` 发起另一次独立语义审核请求；保留记录的 `human_verified` 均为 `false`。这是有限的合成问法分布，教程结果不能外推为自然用户集覆盖，也不能把模型审核称为人工核验。

## 多来源 Notebook 的真实资料范围

[构建多轮多来源助手](../6.%20处理信息缺口/构建多轮多来源助手.ipynb) 直接读取以下三个当前项目文件，并为每个文件建立独立检索器：

| source_id | 仓库路径 | 用途 |
| --- | --- | --- |
| `canonical_dataset_manifest` | `data/dataset/manifest.json` | canonical schema、split 数量、训练/开发/冻结测试边界 |
| `data_processing_readme` | `2. 数据处理/README.md` | query→evidence 构建、MNRL 训练和配置选择规则 |
| `evaluation_readme` | `7. 评估/README.md` | method-fit、general regression guard 和端到端评估范围 |

资料路由、各源检索、证据合并和回答都只使用这三个文件的当前文字。每条 claim 绑定 `source_id`、物理行号和该行原文；审计中同时保存模型 raw JSON、解析结果、路由、检索结果、引用绑定和会话写回。这样可以逐条回到真实文件核对，而不是把同一份 PDF 复制成多个来源。

## 重新运行

在 C7 根目录使用 Python 3.10 的 `llm-universe-c7` kernel。基础数据检查不需要模型；需要生成的实验只读取项目根目录 `.env` 中的 `ZHIPUAI_API_KEY`，固定调用 `glm-4-flash`。缺少密钥、依赖、数据或可解析模型输出时直接失败，不替换模型、资料或结果。

有意修改 canonical JSON/JSONL 后，先刷新 manifest，再检查数据：

```bash
python scripts/prepare_dataset.py
python scripts/check_dataset.py --dataset data/dataset
```

`prepare_dataset.py` 只处理显式的当前数据包：它先在隔离副本中验证全部记录，再更新 manifest 中的数量和 SHA-256。它不读取 Git 历史、不寻找其他问题集，也不会替换缺失输入。只检查 manifest 是否已经同步时使用 `python scripts/prepare_dataset.py --check`。
