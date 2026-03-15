# 增强阶段章节优化 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 优化 C7 第六章"增强阶段"的文件结构、教学文字和内容深度，使其成为结构统一、Case 驱动、文字与代码比例合理的 2026 RAG 教程。

**Architecture:** 删除 4 个旧文件，重命名 2 个重构版文件，重写 readme.md，在 3 个 notebook 中插入过渡段落/结果分析/学习检查点等 markdown cell，扩充系统增强中的 Agentic RAG 并调整方法顺序。

**Tech Stack:** Jupyter Notebook (.ipynb), Markdown, Python (LangChain + OpenAI)

**Design doc:** `docs/plans/2026-03-15-chapter6-enhancement-redesign.md`

---

### Task 1: 清理旧文件

**Files:**
- Delete: `notebook/C7 高级 RAG 技巧/6. 增强阶段/0. 先导：为什么基础 RAG 还不够.ipynb`
- Delete: `notebook/C7 高级 RAG 技巧/6. 增强阶段/0. 其他增强（llama-index 版 rag fusion，混合检索，重写查询）.ipynb`
- Delete: `notebook/C7 高级 RAG 技巧/6. 增强阶段/1. 上下文增强.ipynb`（旧版，无"重构版"后缀）
- Delete: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`（旧版，无"重构版"后缀）

**Step 1:** 删除上述 4 个文件。

**Step 2:** 验证目录中只剩重构版 notebook + 系统增强 + 选型总结 + readme + data + figures。

**Step 3:** Commit

```bash
git add -A
git commit -m "chore: remove legacy notebooks from chapter 6 enhancement stage"
```

---

### Task 2: 重命名重构版文件

**Files:**
- Rename: `1. 上下文增强（重构版）.ipynb` → `1. 上下文增强.ipynb`
- Rename: `2. 流程增强（重构版）.ipynb` → `2. 流程增强.ipynb`

**Step 1:** `git mv` 重命名两个文件。

**Step 2:** 更新两个 notebook 内部的标题 cell：
- `1. 上下文增强（重构版）` 中 Cell 0 标题去掉"（重构版）"
- `2. 流程增强（重构版）` 中 Cell 0 标题去掉"（重构版）"

**Step 3:** 验证文件名和内部标题一致。

**Step 4:** Commit

```bash
git add -A
git commit -m "refactor: rename restructured notebooks, drop '重构版' suffix"
```

---

### Task 3: 重写 readme.md

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/readme.md`

**Step 1:** 用以下内容替换整个 readme.md：

```markdown
# 第六章：增强阶段

## 基础 RAG 的三个隐含假设

基础 RAG 通常隐含三个假设：一次检索就能覆盖答案所需的全部证据；被召回的片段本身已包含充分上下文；当前问题不依赖历史对话、跨文档关系或系统状态。

当这三个假设不成立时，就需要引入增强策略。

## 当假设不成立时

假设 1 失效（命中但上下文不全）→ **上下文增强**：在检索后恢复邻域、父块或层级上下文。
假设 2 失效（一次检索不够）→ **流程增强**：让系统多走几步——迭代、递归、路由、质量把关、自反思。
假设 3 失效（跨轮/跨文档/状态丢失）→ **系统增强**：引入记忆、多文档路由、知识图谱和 Agent 编排。

## 方法地图

｀｀｀mermaid
flowchart TD
    basic[基础 RAG] --> ctx[上下文增强]
    basic --> flow[流程增强]
    basic --> sys[系统增强]

    ctx --> sw[Sentence Window]
    ctx --> stb[Small-to-Big]
    ctx --> am[AutoMerging]

    flow --> ir[迭代检索]
    flow --> rr[递归检索]
    flow --> qr[查询路由与自适应检索]
    flow --> crag[Corrective RAG]
    flow --> selfrag[Self-RAG]

    sys --> mem[Memory]
    sys --> mda[Multi-Document Agent]
    sys --> graphrag[GraphRAG]
    sys --> agentic[Agentic RAG]
｀｀｀

## 本章内容

1. `1. 上下文增强.ipynb` — 解决"检索相关但上下文不全"
2. `2. 流程增强.ipynb` — 解决"一轮流程不够"
3. `3. 系统增强.ipynb` — 解决"多轮/多文档/状态丢失"
4. `4. 选型总结.md` — 方法组合与成本权衡
```

注意：上面的 mermaid 代码块用的是全角反引号作为占位符，实际写入时换成半角反引号。

**Step 2:** 验证 readme 内容正确，mermaid 语法闭合。

**Step 3:** Commit

```bash
git add readme.md
git commit -m "docs: rewrite chapter 6 readme as minimal entry page"
```

---

### Task 4: 优化 `1. 上下文增强.ipynb`

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/1. 上下文增强.ipynb`

所有改动都是插入 markdown cell，不修改现有代码 cell 逻辑。

**Step 1: 补"本章要解决什么"**

在 Cell 0（标题）之后、Cell 1（统一实验设置说明）之前，插入 markdown cell：

> 内容要点：用 1-2 段文字说明"检索相关但上下文不全"这类失败的本质——不是检索没命中，而是命中的片段太碎，前后支撑信息丢失。这是最常见也最容易修复的 RAG 失败类型。

**Step 2: 补 baseline 失败分析**

在 Cell 4（baseline 代码）之后、Cell 5（"如何阅读本节"）之前，插入 markdown cell：

> 内容要点：解读 baseline 输出——指出回答中哪些地方体现了"命中但不完整"，建立读者对后续方法改善效果的预期。

**Step 3: 补 `simple_eval` 说明**

在 Cell 2（代码 cell，含 simple_eval 定义）中 simple_eval 函数定义上方或旁边添加一行注释说明：

> "简化评估：按关键词命中判断，生产环境建议使用 LLM 评估或人工标注。"

**Step 4: 补 Sentence Window 结果分析 + 过渡**

在 Cell 7（Sentence Window 代码）之后、Cell 8（Small-to-Big 标题）之前，插入 markdown cell：

> 内容要点：解读对比结果，说明 Sentence Window 改善了什么。过渡："Sentence Window 通过恢复邻域解决了局部上下文缺失，但如果证据分散在不同段落层级——比如一段总结在章节开头，细节在后续段落——邻域窗口就不够用了。这引出 Small-to-Big。"

**Step 5: 补 Small-to-Big 结果分析 + 过渡**

在 Cell 9（Small-to-Big 代码）之后、Cell 10（AutoMerging 标题）之前，插入 markdown cell：

> 内容要点：解读对比结果。过渡："Small-to-Big 通过父块回填补全了层级上下文，但如果同一个父块下有多个子块被分散命中，我们还需要决定是否合并——这就是 AutoMerging 要解决的问题。"

**Step 6: 补 AutoMerging 结果分析 + 过渡到 Late Chunking**

在 Cell 11（AutoMerging 代码）之后、Cell 12（最终对比表）之前，插入 markdown cell：

> 内容要点：解读对比结果。过渡到 Late Chunking："前面三种方法都在检索时做上下文恢复。有没有可能在索引阶段就解决这个问题？Late Chunking 提供了一种不同的思路。"

**Step 7: 补下一章衔接**

在 Cell 15（学习检查点）之后，插入 markdown cell：

> "如果你发现问题的根源不是上下文不全，而是一次检索流程本身不够——比如需要多步推理、需要先评估再补检——请继续学习 `2. 流程增强.ipynb`。"

**Step 8:** 验证所有新 cell 位置正确，notebook 可正常打开。

**Step 9:** Commit

```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/1. 上下文增强.ipynb"
git commit -m "docs: add transition paragraphs and result analysis to context enhancement notebook"
```

---

### Task 5: 优化 `2. 流程增强.ipynb`

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`

**Step 1: 补 baseline 失败分析**

在 Cell 4（baseline 代码）之后、Cell 5（五种方法分类表）之前，插入 markdown cell：

> 内容要点：解读 baseline 输出——指出回答只覆盖了问题的一部分，缺少子问题维度。"这不是上下文不全的问题（chunk 大小已经合理），而是流程本身只走了一步。"

**Step 2: 补迭代检索结果分析 + 过渡**

在 Cell 7（迭代检索代码）之后、Cell 8（递归检索标题）之前，插入 markdown cell：

> 内容要点：解读 baseline vs 迭代检索的差异。过渡："迭代检索通过'先答后补'逐步完善答案，但它的子问题是隐式产生的（由 LLM 判断缺什么）。如果问题本身可以显式拆解为独立子问题，递归检索会更高效。"

**Step 3: 补递归检索结果分析 + 过渡**

在 Cell 9（递归检索代码）之后、Cell 10（查询路由标题）之前，插入 markdown cell：

> 内容要点：解读结果。过渡："迭代和递归检索解决了'多走几步'的问题，但还没回答'该走哪条路'——当系统有多个索引或工具时，怎么选？这就是查询路由的职责。"

**Step 4: 拆分 Cell 10-11（查询路由 + 自适应检索）**

当前 Cell 10 是查询路由的 markdown，Cell 11 是一个巨大的代码 cell 同时包含查询路由和自适应检索。需要拆成：

1. Cell 10: 查询路由 markdown（保持不变）
2. Cell 11a: 查询路由代码（只保留 `rule_route`、`llm_route`、`routed_answer` 和路由测试）
3. 新 markdown cell: 查询路由结果分析 + 过渡到自适应检索
4. 新 markdown cell: 自适应检索标题+说明
   > "查询路由决定'走哪条路'，自适应检索决定'走多深'。两者组合形成完整的检索前决策层。"
   > 包含：失败场景、流程、边界、关键决策点
5. Cell 11b: 自适应检索代码（`classify_query_complexity`、`adaptive_retrieval_answer` 和测试）
6. 新 markdown cell: 自适应检索结果分析 + 过渡到 CRAG

**Step 5: 补 CRAG 结果分析 + 过渡**

在 Cell 13（CRAG 代码）之后、Cell 14（Self-RAG 标题）之前，插入 markdown cell：

> 内容要点：解读 CRAG 三种 branch 的实际含义。过渡："CRAG 在生成前做了质量把关，但决策仍然是单次的——评估一次、处理一次。如果我们希望模型自己判断'要不要继续检索'，在多个轮次中动态决策呢？这就是 Self-RAG 的思路。"

**Step 6: 补 Self-RAG 结果分析 + 论文对比说明**

在 Cell 15（Self-RAG 代码）之后、Cell 16（小结）之前，插入 markdown cell：

> 内容要点：
> 1. 结果分析
> 2. "教学简化版 vs 论文原版"说明："本节用 LLM API 模拟了 Self-RAG 的 retrieve→generate→critique 循环。论文原版通过特殊的 reflection tokens 在模型内部完成这一决策，需要专门微调的模型（如 selfrag_llama2_7b）。教学版保留了核心控制逻辑，但决策精度依赖通用 LLM 的判断能力。"

**Step 7: 补全方法对比表**

在 Self-RAG 分析之后、Cell 16（小结）之前，插入 markdown cell：

> 表格包含：方法名 | 控制类型 | 典型修复问题 | 新增复杂度 | 最适合场景

**Step 8:** 验证 notebook 结构正确。

**Step 9:** Commit

```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb"
git commit -m "docs: add transitions, split routing cell, add comparison table to process enhancement notebook"
```

---

### Task 6: 优化 `3. 系统增强.ipynb`

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/3. 系统增强.ipynb`

这是改动量最大的任务。包含：调整方法顺序、补文字、扩充 Agentic RAG。

**Step 1: 补"本章要解决什么"**

在 Cell 0（标题）之后、Cell 1（流程 vs 系统辨析）之前，插入 markdown cell：

> 内容要点："前两章解决的是单次请求内的问题——上下文不全、流程不够。但当系统需要记住上一轮对话、协调多个数据源、或在复杂任务中自主规划执行路径时，单次请求的优化已经不够了。这就是系统增强要解决的问题。"

**Step 2: 补 Memory 结果分析 + 过渡**

在 Memory 代码 cell 之后，插入 markdown cell：

> 内容要点：解读有记忆 vs 无记忆的输出差异。过渡："记忆解决了跨轮状态延续问题。但如果系统面对的是多个不同数据源——每个数据源需要不同的检索策略——单纯的记忆还不够，我们需要路由和工具选择能力。这就是 Multi-Document Agent 的角色。"

**Step 3: 补 Multi-Doc Agent 结果分析 + 过渡**

在 Multi-Doc Agent 对比代码之后，插入 markdown cell：

> 内容要点：解读 Agent vs Baseline 的差异。过渡："Multi-Document Agent 通过路由解决了'去哪找'的问题。但有些问题不只是找对数据源就行——它们需要实体关系推理，需要知道'A 和 B 之间有什么关系'。这类问题需要知识图谱。"

**Step 4: 调整方法顺序——把 GraphRAG 移到 Agentic RAG 前面**

当前顺序：Memory → Multi-Doc → 系统成本边界 → Agentic RAG → GraphRAG
调整为：Memory → Multi-Doc → GraphRAG → 系统成本边界 → Agentic RAG

具体操作：把 GraphRAG 的 markdown cell 和代码 cell 移到 Agentic RAG 之前。

**Step 5: 扩充 GraphRAG 文字**

在现有 GraphRAG markdown cell 中补充：
> "向量检索和图检索的直觉区别：向量检索问的是'哪些文本和我的问题语义相似'，图检索问的是'哪些实体和我查询的实体有关系'。前者擅长局部事实查询，后者擅长全局性问题和多跳推理。"

在 GraphRAG 代码 cell 后插入 markdown cell：
> 结果分析 + `eval()` 安全性警告："本示例用 eval() 解析 LLM 输出的三元组列表，仅供教学演示。生产环境应使用结构化输出（如 JSON mode）或专用实体抽取模型。"
> 过渡："到目前为止，我们有了记忆（管状态）、多文档路由（管工具选择）、知识图谱（管关系推理）。但当任务足够复杂时，光有这些'积木'还不够——我们还需要一个'大脑'来规划执行步骤、调度这些工具、并在出错时自我修正。这就是 Agentic RAG。"

**Step 6: 大幅扩充 Agentic RAG**

替换当前简短的 Agentic RAG markdown，改为多个 markdown cell + 代码 cell 的完整方法块：

6a. **概念动机 markdown：**
> "为什么 Multi-Document Agent 还不够？"
> Multi-Doc Agent 的路由是单步的：选工具→检索→回答。但复杂任务往往需要：先规划要做哪些步骤、逐步执行并获取中间结果、检查结果是否充分、不够就调整计划继续执行。这种"规划→执行→反思"的循环，就是 Agentic RAG 的核心。

6b. **核心架构 markdown：**
> Plan → Execute → Reflect 循环的详细说明：
> - **Planner**：将复杂问题拆解为可执行的步骤序列。每个步骤是一个具体的检索或推理任务。
> - **Executor**：逐步执行计划，调用检索器、路由器、图谱等工具获取中间结果。
> - **Reflector**：审视中间结果是否足够回答原始问题。如果不够，指出缺口，Planner 追加步骤。
> 配一个简洁的 mermaid 流程图。

6c. **与非 Agent 流程对比 markdown：**
> 表格：非 Agent 流程（固定管线）vs Agentic RAG（动态决策），从路径控制、错误恢复、可调试性、延迟成本等维度对比。

6d. **代码 cell：** 保留并扩充现有的 planner/executor/reflector 代码。确保 demo 中 Agentic RAG 能调度前面的 city_retrievers（Multi-Doc）作为工具。如果可行，让它也能调用 graph_retrieve 作为工具之一。

6e. **结果分析 markdown：**
> 对比 Agentic RAG vs 非 Agent 的输出差异。

6f. **生产注意事项 markdown：**
> - 可观测性：必须记录每一步的规划决策、工具调用和中间输出。
> - 回退策略：Agent 路径失败时，应回退到非 Agent 管线保证可用性。
> - 成本控制：每轮 Plan/Execute/Reflect 都消耗 LLM 调用，需设置最大轮次和 token 预算。
> - 建议：先上线非 Agent 路径作为 baseline，再逐步放量 Agentic 策略。

6g. **适用边界 markdown：**
> - 适合：目标复杂、需要多步工具协作的场景
> - 不适合：简单查询（杀鸡用牛刀）、延迟敏感场景
> - 判断标准：如果去掉规划和反思，系统仍能正确回答，那就不需要 Agent

**Step 7: 移动"系统成本与边界" markdown**

把当前的"系统成本与边界" cell 移到 Agentic RAG 生产注意事项之后（或合并到 Agentic RAG 的生产注意事项中），避免它突兀地出现在中间。

**Step 8: 补全方法对比表**

在所有方法之后，插入 markdown cell：

> 表格：方法名 | 解决问题 | 复杂度 | 延迟成本 | 最适合场景

**Step 9: 补学习检查点**

> - 你能解释 Memory 的窗口策略和摘要策略的区别吗？
> - 你能说出 Multi-Document Agent 在什么情况下比单索引更好、什么情况下不值得？
> - 你能画出 Agentic RAG 的 Plan→Execute→Reflect 循环吗？
> - 你知道 GraphRAG 和向量检索各自擅长什么类型的问题吗？

**Step 10:** 验证 notebook 结构和方法顺序正确。

**Step 11:** Commit

```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/3. 系统增强.ipynb"
git commit -m "docs: expand Agentic RAG, reorder methods, add transitions to system enhancement notebook"
```

---

### Task 7: 优化 `4. 选型总结.md`

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/4. 选型总结.md`

**Step 1:** 在文件开头（`# 4. 选型总结` 之后）插入引言段落：

> "读完前三节之后，你已经了解了三层增强的方法和各自的适用场景。本节帮你做最后一步决策：面对具体问题时，应该先选哪个方法、怎么组合、什么时候该停。"

**Step 2:** 将 `## 1.5) 方法选择决策流程` 改为 `## 2) 方法选择决策流程`，后续编号顺延：
- 原 `2)` → `3)`
- 原 `3)` → `4)`
- 原 `4)` → `5)`
- 原 `5)` → `6)`

**Step 3:** Commit

```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/4. 选型总结.md"
git commit -m "docs: add intro paragraph and fix numbering in selection summary"
```

---

### Task 8: 最终验证

**Step 1:** 确认目录结构只有 readme.md + 3 个 notebook + 1 个 md + data + figures。

**Step 2:** 逐个打开 notebook 确认 cell 顺序和内容正确。

**Step 3:** 确认 readme.md 中的文件名引用与实际文件名一致。

**Step 4:** 确认 4. 选型总结.md 编号连续。
