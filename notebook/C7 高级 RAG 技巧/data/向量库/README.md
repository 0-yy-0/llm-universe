# 教程配套向量库

[南瓜书配套库](南瓜书配套库/)保存了《南瓜书》的文本片段和向量，约 7.2 MB。下载完整教程时保留这个目录即可，无须重新计算整本书的向量。具体的向量检索代码见[结合关键词和向量检索](../../3.%20索引阶段/结合关键词和向量检索.ipynb)。

只看教程时，直接阅读 Notebook 中保存的输出。部分 BM25 示例也会从这份库读取已经切好的 987 个文本片段，因此重新运行这些页面需要 Chroma，但不需要 BGE。重新运行向量检索时，还需要 BGE 模型把**新问题**转成向量；Notebook 只读取本地缓存，不会在实验过程中联网下载。这与重算文档向量是两件事。

## 这份库使用什么配置

| 项目 | 配置 |
|---|---|
| 资料 | [pumpkin_book.pdf](../pumpkin_book.pdf)，PDF 标注版本 1.9.9、2023.03；SHA-256 `371e426890ae8246bca319e433b1d54d0610f015de42df8385cd8eeb610225b1` |
| 向量模型 | `BAAI/bge-small-zh-v1.5`，512 维，向量归一化 |
| 问题编码 | 直接编码原问题，不额外添加检索提示语 |
| 分块 | 先逐页清理页眉和空白，再用 `RecursiveCharacterTextSplitter` 按 256 字符切分，重叠 20 字符 |
| 片段数 | 987 |
| 比较方式 | 余弦距离 |
| 默认返回数 | 4 个片段 |
| Chroma 集合名 | `nb_ctx` |
| 已验证的读取版本 | `chromadb==1.5.5` |

库中的 `page` 从 0 开始；教程加载函数会加 1，显示为 PDF 阅读器中的页码。这里的页码不是书中印刷页码。

## 来源与复用范围

这份库使用 256 字符分块和 20 字符重叠。987 个片段的页码和可见文字已经与当前 PDF 核对；读取函数会去掉 PDF 提取时留下的隐藏控制字符。教程直接保存这套文档向量，不需要读者重新生成。

本地 [`pumpkin_book.pdf`](../pumpkin_book.pdf) 标注版本为 1.9.9，官方来源是 Datawhale 的[南瓜书项目](https://github.com/datawhalechina/pumpkin-book)，对应的[版本发布页](https://github.com/datawhalechina/pumpkin-book/releases/tag/v1.9.9)和[许可文件](https://github.com/datawhalechina/pumpkin-book/blob/master/LICENSE)均可公开核对。原项目采用 CC BY-NC-SA 4.0；这套向量库含有从 PDF 提取的文字，发布和复用时应一并保留署名、非商业使用和相同方式共享的要求。商业使用需要另行取得授权。

## 重新运行：使用独立环境

在仓库根目录执行下面的命令即可准备一个只给本章使用的环境（Python 3.10）：

```bash
python3.10 -m venv .venv-c7-rag
.venv-c7-rag/bin/python -m pip install -U pip
.venv-c7-rag/bin/python -m pip install -r "notebook/C7 高级 RAG 技巧/requirements-c7.txt"
```

依赖入口中的 `chromadb==1.5.5` 是本配套库的已验证读取版本；`pymupdf`、`python-docx`、`python-pptx` 等用于 Notebook 的本地解析，其他未锁定包不代表已在所有平台验证。仓库根目录的依赖配置和 `data_base/vector_db/chroma` 中的现有 Chroma 库不适用于本章。

运行 Notebook 前需显式准备一次 `BAAI/bge-small-zh-v1.5` 缓存：

```bash
.venv-c7-rag/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"
```

这一步可以联网；Notebook 随后使用 `local_files_only=True`，缓存缺失会直接报错。它会保留真实的 BGE 检索结果，不会悄悄改用 BM25。

执行 Notebook 时可从仓库根、C7 根或本章目录启动；`requirements-c7.txt` 已包含 `nbconvert` 和 `ipykernel`，无需额外安装 JupyterLab。下面三个命令分别从三个目录执行同一份数据处理示例并把新输出保存回 Notebook：

```bash
# 仓库根
cd /path/to/llm-universe
.venv-c7-rag/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
  "notebook/C7 高级 RAG 技巧/2. 数据处理/读取和清理不同格式的资料.ipynb"

# 或 C7 根
cd "notebook/C7 高级 RAG 技巧"
../../.venv-c7-rag/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
  "2. 数据处理/读取和清理不同格式的资料.ipynb"

# 或章节目录
cd "notebook/C7 高级 RAG 技巧/2. 数据处理"
../../../.venv-c7-rag/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
  "读取和清理不同格式的资料.ipynb"
```

若使用 VS Code、PyCharm 或已有的 JupyterLab 交互阅读，直接选择这个虚拟环境作为 kernel 即可；这不是运行代码的必需依赖。

如果使用其他 Python 环境，请确认环境中的 Chroma 版本为 `1.5.5`。修改问题或比较检索方法时，可以继续复用这份库。只有更换 PDF、分块方式或向量模型时，才需要为改动后的文档计算向量。实验产生的新库应另放目录，不要覆盖教程附带的文件。
