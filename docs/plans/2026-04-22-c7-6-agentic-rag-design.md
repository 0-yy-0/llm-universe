# C7.6 Agentic RAG — 设计文档

> 日期：2026-04-22
> 状态：待用户确认

## 一、定位与边界

### 1.1 为什么补这一节

`6. 增强阶段` 的 `readme.md` 与 `4. 选型总结.md` 的方法地图均把 **Agentic RAG** 列在"系统增强"分支下，但 `3. 系统增强.ipynb` 实际只实现了 Memory 与 Multi-Document Agent，Agentic RAG 缺位。本节补齐这一空缺。

### 1.2 与已有内容的边界

| 已有内容 | 与 Agentic RAG 的差异 |
|---|---|
| `2.` Self-RAG / CRAG | 单 query 内的固定反思模板，无工具调用、无任务规划 |
| `2.` 自适应检索 | 单步策略选择，不进入闭环 |
| `3.` Memory | 解决跨轮指代，不解决任务规划 |
| `3.` Multi-Document Agent | **单步 LLM 路由 + 单次检索**，不是真正的 agent |

**Agentic RAG 的独有点**：Plan-Act-Observe-Reflect 闭环 + 工具集 + trace 可观测。它把前面所有方法都视作可调用的"子能力"。

### 1.3 教学目标

读者读完后能：
1. 用 80 行代码手写一个 ReAct Agent（不依赖 LangGraph）
2. 区分 ReAct vs Plan-then-Execute 两种主流架构的取舍
3. 设计差异化的工具集，而不是只挂一个 retriever
4. 评估 Agent 的"工具调用准确率""trace 长度""答案正确率"三类指标
5. 知道从 Agentic RAG 到 Deep Researcher 需要追加哪些组件

## 二、文件与目录调整

### 2.1 新增文件

```
6. 增强阶段/
├── 4. Agentic RAG.ipynb            ← 新增
├── data/
│   └── agentic_eval.json           ← 新增：6~8 道多步题
```

### 2.2 顺延文件

```
4. 选型总结.md  →  5. 选型总结.md
```

### 2.3 改动文件

- `readme.md`：在「本章内容」列表加入 `4. Agentic RAG.ipynb`，原 4 顺延为 5；mermaid 图无需调整（agentic 节点已经存在）。
- `5. 选型总结.md`（原 `4.`）：在"决策流程"与"速查表"中把 `Agentic RAG` 行的描述改为指向新 notebook，并把"复杂任务组合"案例补一句"详见 6.4 节"。
- `_common.py`：新增 2 个轻量工具函数（详见第六节）。

## 三、Notebook 大纲

总目标：12~15 个 cell，约 1000 行（与 6.2/6.3 同量级）。

```
# 4. Agentic RAG

## 4.0 本节定位与钩子升级
   - 与 2. Self-RAG / 3. Multi-Doc Agent 的差异表
   - 钩子升级：class XxxAgent: ask(q) -> (answer, trace)
     —— 在 6.3 的 ask(q) -> str 基础上多返回一个 trace，便于过程评估

## 4.1 统一实验设置 + 环境准备
   - 复用 _common.py 公共底座
   - 复用 6.3 的 CHAPTER_RETRIEVERS / SOURCE_DESCRIPTIONS

## 4.2 工具集设计
   - 定义 4 个工具（见第四节）
   - 每个工具有统一签名：tool(args_dict) -> str
   - 提供 ToolRegistry 注册器与 to_prompt_schema() 方法

## 4.3 ReAct Agent（手写实现）
   - 主循环：Thought → Action → Observation × N
   - 控制：最大轮次、提前停止、解析失败的兜底
   - inspect：跑 1 道复杂题，打印完整 trace

## 4.4 Plan-then-Execute Agent（手写实现）
   - Step 1：让 LLM 一次性出 task_plan: list[Step]
   - Step 2：按 plan 顺序执行；每步可选"按上一步结果调整剩余 plan"
   - inspect：同一道题，打印 plan + 执行 trace

## 4.5 Baseline：Multi-Doc Agent 适配
   - 复用 6.3 的 MultiDocAgent，包成 ask(q) -> (answer, trace) 形态
   - trace 仅含一步路由 + 一次检索，恰好作为弱 baseline

## 4.6 三方对比评估
   - 5~8 道多步题（agentic_eval.json）
   - 三个指标：
       a. 答案正确率（复用 simple_eval_2pt + DIMENSIONAL_EVAL_PROMPT）
       b. 工具调用准确率（trace 中"调对工具的步骤数 / 总步骤数"）
       c. 平均 trace 长度（间接反映成本）
   - build_compare_table 出三方对比表

## 4.7 何时用谁：决策建议
   - ReAct：探索式问题、工具数 ≤ 5、容忍轮次抖动
   - Plan-then-Execute：可拆解题、可预算控制、需要可解释 plan
   - Multi-Doc Agent：纯路由场景、对延迟敏感

## 4.8 延伸：从 Agentic RAG 到 Deep Researcher
   - 一张架构对比图（mermaid）
   - 200 字说明：从单 agent 到 Planner+Researcher×N+Writer 多 agent；
     从单 query 到长任务记忆 + 报告大纲 + 引用对齐
   - 不实现 demo，留作下一篇

## 4.9 小结
   - 三种 agent 的异同表
   - 与 7. 评估的衔接（trace 级指标会在 7.5 错误归因中复用）
```

## 四、工具集设计

四个工具，签名统一为 `tool_fn(args: dict) -> str`，名字与中文描述都进入 prompt。

| 工具名 | 参数 | 用途 | 实现 |
|---|---|---|---|
| `search_full_corpus` | `{"query": str}` | 全库兜底检索 | 复用 4.1 节构建的全局 retriever |
| `search_chapter` | `{"chapter_id": str, "query": str}` | 章节定向检索 | 复用 6.3 的 `CHAPTER_RETRIEVERS[chapter_id]` |
| `lookup_formula` | `{"formula_id": str}` | 按"式 6.11"返回原文段落 | 用正则在原始 PDF 文本中查"(式 X.Y)"上下若干字符 |
| `calc` | `{"expr": str}` | 简单算术 | Python `eval` + 安全白名单（仅 `+ - * / ** log sqrt`） |

**ToolRegistry 接口约定**：

```python
class ToolRegistry:
    def register(self, name: str, fn: Callable, desc: str, args_schema: dict): ...
    def call(self, name: str, args: dict) -> str: ...        # 带异常兜底
    def to_prompt_schema(self) -> str: ...                    # 拼到 system prompt
```

`to_prompt_schema()` 输出形如：

```
你可以调用的工具：
- search_full_corpus(query: str) — 在南瓜书全库检索
- search_chapter(chapter_id: str, query: str) — 在指定章节检索；chapter_id 可选 ch1_intro/ch2_eval/ch3_linear/ch4_tree
- lookup_formula(formula_id: str) — 按公式编号返回原文，如 "6.11"
- calc(expr: str) — 简单算术
```

**Action 解析协议**（ReAct）：要求 LLM 输出严格 JSON：

```
Thought: ...
Action: {"tool": "search_chapter", "args": {"chapter_id": "ch4_tree", "query": "信息增益"}}
```

解析失败时：① 重试一次（用更严格的 prompt）；② 仍失败则 fallback 到 `search_full_corpus(原始 question)`，并在 trace 标记 `parse_error`。

## 五、题目集草案

新建 `data/agentic_eval.json`，6 道多步题（先草拟，由用户审定）：

```json
[
  {
    "id": "Q1",
    "question": "决策树和支持向量机在二分类时的决策边界形状有何区别？",
    "expected": "决策树边界是分段平行于坐标轴的轴对齐矩形（多个矩形拼接）；支持向量机用线性核时是超平面，用 RBF 核时是非线性平滑曲面。",
    "expected_tools": ["search_chapter:ch4_tree", "search_chapter:ch3_linear"]
  },
  {
    "id": "Q2",
    "question": "南瓜书式 6.11 是什么，几何意义是什么？",
    "expected": "式 6.11 是 SVM 对偶问题的拉格朗日函数；几何上对应在约束下最大化间隔。",
    "expected_tools": ["lookup_formula:6.11", "search_chapter:ch6"]
  },
  {
    "id": "Q3",
    "question": "样本量 N=1000、特征数 d=20 时，KNN 预测一条样本和线性 SVM 预测一条样本的复杂度量级各是多少？",
    "expected": "KNN 预测一条 ≈ O(N·d) = 20000 次操作；线性 SVM 预测一条 ≈ O(d) = 20 次操作。",
    "expected_tools": ["search_chapter:ch3_linear", "calc"]
  },
  {
    "id": "Q4",
    "question": "南瓜书第 2 章和第 4 章都讨论了'剪枝'相关概念，两者的目的有何不同？",
    "expected": "第 2 章讨论的是模型选择中的过拟合控制（评估角度）；第 4 章讨论的是决策树预剪枝/后剪枝（结构角度），目的都是防止过拟合但操作对象不同。",
    "expected_tools": ["search_chapter:ch2_eval", "search_chapter:ch4_tree"]
  },
  {
    "id": "Q5",
    "question": "用基尼指数从样本数 100 的根节点分出 60/40 两子节点，基尼增益是多少？",
    "expected": "根节点 Gini = 1 - (0.6²+0.4²) = 0.48；分裂后加权 Gini = 0.6·G1 + 0.4·G2，需进一步给出子节点类别分布才能算最终值。",
    "expected_tools": ["search_chapter:ch4_tree", "calc"]
  },
  {
    "id": "Q6",
    "question": "ROC 曲线与 PR 曲线的核心区别是什么？哪种场景下 PR 更可靠？",
    "expected": "ROC 横轴 FPR、纵轴 TPR；PR 横轴 Recall、纵轴 Precision。类别极不平衡时 ROC 会高估性能，PR 更可靠。",
    "expected_tools": ["search_chapter:ch2_eval"]
  }
]
```

`expected_tools` 字段是为"工具调用准确率"指标准备的：trace 里的工具调用与 `expected_tools` 求交集即为"调对工具的步骤数"。允许超集（agent 多调一些工具不扣分），但缺失关键工具会显著扣分。

## 六、`_common.py` 新增接口

只加两个轻量工具，不污染现有代码：

```python
def safe_eval_arith(expr: str) -> str:
    """白名单算术求值；非白名单字符直接抛 ValueError。"""

def trace_tool_recall(trace: list[dict], expected_tools: list[str]) -> float:
    """trace 命中 expected_tools 的覆盖率，0~1 浮点数。"""
```

agent 主体（`ReActAgent` / `PlanThenExecuteAgent` / `ToolRegistry`）保持在 notebook 内手写实现，不下沉到 `_common.py`，遵循"教学清晰优先"原则。

## 七、评估指标与对比表

三方对比表的列：

| method | acc_2pt（0~2 均分） | tool_recall（0~1） | mean_trace_steps |
|---|---|---|---|
| MultiDocAgent | 基线值 | 通常 ≤ 1（只能选一个工具） | 1.0 |
| ReActAgent | 期望显著高 | 期望 0.7~0.9 | 期望 2~4 |
| PlanThenExecuteAgent | 期望与 ReAct 接近或略高 | 期望与 ReAct 接近 | 期望 2~3，方差更小 |

读表预期：ReAct 在 trace 长度上波动大但灵活；Plan-then-Execute 在 trace 长度上更稳定，但对"问题可拆解性"敏感；MultiDocAgent 在多步题上明显落败——这正是本节要传达的核心结论。

## 八、与其他章节的衔接

- **6.3** → 本节复用 `CHAPTER_RETRIEVERS`、`SOURCE_DESCRIPTIONS`、`MultiDocAgent`，零冗余构建。
- **7. 评估** → 本节产出的 `trace` 字段会在 7.5 错误归因中作为案例数据；本节的 `tool_recall` 指标在 7.2 三元组指标里有对应位置（"操作正确率"）。
- **新 5. 选型总结**（原 4） → 把"复杂任务组合"案例的脚注补成 "Agentic RAG 见 6.4 节，包含 ReAct 与 Plan-then-Execute 实现对比"。

## 九、开发计划与工作量估算

| 阶段 | 工作量 | 输出 |
|---|---|---|
| 1. 题目集敲定 | 0.5h | `data/agentic_eval.json`（用户过一遍） |
| 2. 工具集 + ToolRegistry | 1.5h | notebook 4.2 节 + `_common.py` 增量 |
| 3. ReActAgent 实现 + inspect | 2h | notebook 4.3 节 |
| 4. PlanThenExecuteAgent + inspect | 2h | notebook 4.4 节 |
| 5. baseline 适配 + 三方对比 | 1.5h | notebook 4.5 / 4.6 节 |
| 6. 决策建议 + Deep Researcher 延伸 | 1h | notebook 4.7 / 4.8 / 4.9 节 |
| 7. readme + 选型总结调整 | 0.5h | 目录联动改动 |
| **合计** | **~9h** | 1 个 notebook + 1 个 data 文件 + 2 个目录文件改动 |

## 十、风险与开放问题

1. **GLM-4-Flash 的 tool-use 能力**：模型可能频繁产出无法解析的 Action JSON。已有 fallback 策略（重试 + 退化为全库检索 + trace 标记），但需要在实现阶段验证失败率，必要时把 prompt 改成更宽松的"键值对"格式。
2. **题目集人工标注**：6 道题的 `expected` 与 `expected_tools` 都是我草拟的，需要用户依据南瓜书原文校对，特别是 Q2/Q5 的公式编号与数值。
3. **评估的随机性**：LLM-as-Judge 与 LLM 自身回答都有随机性。建议每个 agent 跑 1 次（与 6.2/6.3 风格一致），不做多次平均；如果对比表起伏过大，再考虑加 `temperature=0` + 单次跑足。
4. **Deep Researcher 是否需要附一个简化 demo**：当前设计只放架构图。如果你后续希望加一个用 web 搜索的可运行 demo，建议另起一节而不是塞进本节，避免本节超出 L2 边界。

## 十一、自检（spec 自查）

- [x] 占位检查：无 TBD/TODO；所有小节有具体内容
- [x] 内部一致：4 个工具、6 道题、3 方对比、9h 工作量在各节互相印证
- [x] 范围检查：单一 notebook + 必要的目录联动，无范围蔓延
- [x] 歧义检查：`tool_recall` 公式、`Action` 解析协议、Deep Researcher 的"延伸不实现"边界都已写明
