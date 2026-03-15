# 第六章"增强阶段"重构 — 交接文档

> 创建时间：2026-03-15
> 用途：上下文满了，开新窗口时用来恢复全部进度和决策

---

## 一、项目背景

- 仓库路径：`/Users/zhihu123/Project/other/llm-universe`
- 教程定位：2026 年面向初学者的高级 RAG 教程
- 章节位置：`notebook/C7 高级 RAG 技巧/6. 增强阶段/`
- 整体定位：A 为主（不改前5章），局部吸收 B 的思想（第6章可大改）
- 第7章也可大改，但本轮未动

---

## 二、已确认的设计决策

### 三层结构
```
6. 增强阶段/
  0. 先导：为什么基础 RAG 还不够.ipynb
  1. 上下文增强（重构版）.ipynb
  2. 流程增强（重构版）.ipynb
  3. 系统增强.ipynb
  4. 选型总结.md
  readme.md
```

### 与前几章的边界约定
| 前章已覆盖 | 第六章不重复讲 |
|---|---|
| 第3章：混合检索、元数据、CCH、Document Augmentation | 索引结构、分块方式 |
| 第4章：query 改写、HyDE、step-back、子查询 | query 变换原理 |
| 第5章：上下文压缩(ContextualCompression)、重排(Reranking)、过滤、引用 | 压缩/重排/过滤 |

### 关键设计选择
- 删掉 `Contextual Compression`（第5章已讲）
- 删掉 `LLM 生成内容`（不属于上下文增强）
- `记忆` 从上下文增强移到系统增强
- `Multi-Document Agent` 从流程增强移到系统增强
- `Self-RAG` 降级为自适应检索的案例
- 每个方法统一模板：问题 → 思想 → 示意图 → 最小代码 → 效果 → 适用/局限
- `0. 其他增强` notebook 保留为历史草稿

---

## 三、已创建/修改的文件

### 新建文件（重构版主线）
1. `0. 先导：为什么基础 RAG 还不够.ipynb` — 6 cell，方法地图 + 学习路线
2. `1. 上下文增强（重构版）.ipynb` — 10 cell，Sentence Window / Small-to-Big / AutoMerging
3. `2. 流程增强（重构版）.ipynb` — 11 cell，迭代检索 / 递归检索 / 查询路由 / 自适应+Self-RAG
4. `3. 系统增强.ipynb` — 12 cell，Memory + Multi-Document Agent + baseline对比 + 延伸阅读
5. `4. 选型总结.md` — 速查表 + 落地原则 + 常见误区
6. `readme.md` — 更新为重构版目录入口

### 保留的旧文件（未删除）
- `0. 其他增强（llama-index 版 rag fusion，混合检索，重写查询）.ipynb`
- `1. 上下文增强.ipynb`（原始版，有完整 qna_dict 和评估结果）
- `2. 流程增强.ipynb`（原始版，有 Self-RAG 完整原理 + Multi-Document Agent 完整代码）

---

## 四、未完成的 7 个改进项（按优先级排序）

### P0：必须做

#### 1. 先导补可运行的失败案例
- 用现有 `./data/face.pdf` + `qna_dict`
- 把 chunk_size 设小（如128），跑基础向量检索
- 展示"检索到了但答错了"的具体 case
- 推荐场景：跨段落答案不完整 / chunk 边界切断
- 不需要额外造数据，控制 chunk_size 即可复现

#### 2. 上下文增强补原理拆解
- 现在三个方法都是 `download_llama_pack` 黑盒调用
- 每个方法需要补：
  - "索引时 vs 检索时"流程对比图
  - 3-5 行核心代码展示关键步骤（不是整个 pack）
  - 关键参数说明（窗口大小 / 合并阈值 / 父子块比例）

#### 3. 流程增强代码从伪代码升级为可运行
- `iterative_retrieval` 和 `recursive_retrieval` 当前是独立函数，没接 retriever/llm
- `route_query` 只是关键词匹配，没有 LLM-based routing
- 至少迭代检索和查询路由要给完整可运行示例

### P1：应该做

#### 4. Self-RAG 补回原理讲解
- 旧版 `2. 流程增强.ipynb` 里有完整原理（三环节 + 两张图 + 训练/推理流程）
- 重构版只剩一句话
- 建议：保留原理讲解和示意图，代码可简化

#### 5. Memory 补端到端多轮问答 demo
- 现在只展示了 `ConversationMemory` 类和 `build_retrieval_query`
- 缺少完整流程：提问 → 拼接历史 → 检索 → 生成 → 写回 memory → 下一轮

### P2：锦上添花

#### 6. 选型总结补"方法组合"
- 常见组合：Sentence Window + 查询路由、迭代检索 + 记忆、AutoMerging + Reranking
- 补 2-3 个推荐组合方案

#### 7. 选型总结补"评估指标"列
- 每种方法标注"主要改善什么指标"（召回率 / 忠实性 / 上下文完整性）
- 补一列到速查表

---

## 五、可复用的稳定"造失败"技巧

| 失败场景 | 怎么造 | 对应增强 | 复现难度 |
|---|---|---|---|
| 跨段落答案不完整 | 问一个答案分散在两个不相邻段落的问题 | 上下文增强 | 低 |
| chunk 边界切断 | chunk_size 设为 128，让完整句子被切断 | 上下文增强 | 极低 |
| 需要多步推理 | 问"A的B是什么"，A和B的关系分布在不同段落 | 流程增强 | 低 |
| 多文档混淆 | 多个主题文档混在一个索引，问具体主题 | 系统增强 | 低 |
| 多轮指代消解 | 第一轮问X，第二轮问"那它呢" | 系统增强 | 低 |

---

## 六、数据文件说明

- `./data/face.pdf` — 机器学习面试题（SVM/BN/LN/Transformer/BERT等）
- `./data/mutli_documents_data/` — 18个城市的 wiki 文本 + 已构建的索引
- `./figures/selfrag.png` / `selftoken.jpg` — Self-RAG 原理图（可复用）
- `./figures/rrfsample.png` — RRF 示例图
- `./figures/mda.png` — Multi-Document Agent 示例图

---

## 七、整套教程的完整大纲（供参考）

```
C7 高级 RAG 技巧/
  1. 背景/
  2. 数据处理/
  3. 索引阶段/
  4. 检索阶段/
  5. 生成阶段/
  6. 增强阶段/  ← 当前重构中
  7. 评估/       ← 待重构
```

---

## 八、新窗口的推荐起始指令

```
请阅读 docs/plans/2026-03-15-chapter6-handoff.md，然后继续执行第六章增强阶段的改进。
优先级：P0 的 3 项 > P1 的 2 项 > P2 的 2 项。
从 P0-1（先导补失败案例）开始。
```
