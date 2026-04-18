# C7「增强阶段」三节统一重构与系统增强补全设计

**日期**：2026-04-18
**范围**：`notebook/C7 高级 RAG 技巧/6. 增强阶段/` 三个 notebook 与 `_common.py`
**前序设计**：`docs/superpowers/specs/2026-04-10-c7-context-enhancement-teaching-design.md`（仅针对 6.1 节的教学优先策略，本设计在其基础上扩展并对齐到 6.2、6.3 节）

## 1. 背景与问题

C7 第 6 章「增强阶段」共 3 个 notebook，当前状态如下：

| 节 | 状态 | 主要问题 |
|---|---|---|
| `1. 上下文增强.ipynb`（2859 行） | 内容完整、有同题对比表 | 公共代码自给自足，与 2、3 节不互通 |
| `2. 流程增强.ipynb`（1114 行） | 6 种方法均有实现 | 评估硬编码"对偶问题"单题；公共代码与 1 节复制粘贴；`llm_call` 每次成功强制 `sleep(20)`；无跨方法对比表 |
| `3. 系统增强.ipynb`（127 行） | **仅环境准备，readme 承诺的 4 个方法（Memory / Multi-Document Agent / GraphRAG / Agentic RAG）一个都没实现** | 实质是空壳 |

三节"形似而神不同"：

- **公共底座重复**：PDF 清洗、本地 embedding、Chroma 工厂、`llm_call`、评估函数三节各写一份。
- **评估口径割裂**：1 节 0~2 分（结构化）、2 节 0~10 分自由文本（写死单题）、3 节 ✅/❌ 二值。无法横向比较。
- **节流策略不一致**：1 节默认不 sleep；2、3 节每次成功调用 `sleep(20)`，连续跑 5 道题需 5~10 分钟。
- **3 节未实现**：作为系列收尾的"系统增强"实际为空。

读者在三节之间切换时，必须重新理解"环境准备"的细节，无法形成"上下文 → 流程 → 系统"三层递进的统一认知。

## 2. 目标与非目标

### 2.1 目标

1. **三节共用一个 `_common.py` 公共底座**，删除重复定义（PDF 清洗、embedding、Chroma 工厂、`llm_call`、评估骨架等）。
2. **每节钩子落在本节真正的变化点**，让读者一眼看出"这一节在变什么"。
3. **6.3 节从零写两个真正体现"系统级状态"的方法**：Memory（多轮对话） + Multi-Document Agent（多源路由）。
4. **教学优先**：所有"读者第一次见到的核心新概念"（包括公共底座中本章首次出现的函数）必须在 notebook 中完整展示一次实现，再在后续节复用。
5. **保持现有 6.1 教学策略**（前序 spec 定义的"机制为主、单一题集、不强求全胜"）。

### 2.2 非目标（本轮不做）

- **不新增 6.1、6.2 的方法**。6.1 已覆盖检索后上下文增强的主要思路（Sentence Window / Small-to-Big / AutoMerging + Late Chunking 理论）；6.2 已覆盖 5~6 种流程编排策略。
- **不引入 Contextual Retrieval、RAPTOR**。两者均属于索引阶段（embedding 之前用 LLM 生成上下文摘要 / 层级聚类摘要），按"按 RAG 流水线分章"的归类逻辑应放第 3 章。如需补强需另开 spec。
- **不引入 GraphRAG / Agentic RAG**。GraphRAG 需图数据库依赖过重；Agentic RAG 与 6.2 流程增强语义重叠，本轮先不做。
- **不引入 Late Chunking 可运行 demo**。该方法依赖长上下文 embedding 模型（如 jina-embeddings-v2），与本节本地 `bge-small-zh-v1.5` 不兼容，硬做反而割裂；6.1 现有理论介绍保留。
- **不做大规模评估扫描**（与前序 spec 一致：教学优先，不强求统计显著性）。
- **临时脚本清理**（仓库根 `final_fix_nb.py`、`fix_nb_v4.py`、`temp_eval.py` 等）作为 follow-up 备注，不纳入本轮主线。

## 3. 核心设计原则

> **公共底座只放真正"全节都用且与本节核心无关"的能力**；每节自己定义本节的"钩子点"，钩子点必须落在本节真正变化的地方，让读者一眼看出"这一节在变什么"。

依据这条原则：

- 6.1 节的核心变化点是「检索后如何拼 context」→ 钩子是 `*_context(question) -> str`
- 6.2 节的核心变化点是「检索-生成的流程编排」→ 钩子升一层为 `*_pipeline(question) -> str`
- 6.3 节的核心变化点是「跨轮/跨源的状态」→ 钩子再升一层为 `class XxxSystem: ask(q) -> str`

公共底座 `_common.py` **只服务**这些钩子的"接入与评估"，不做钩子内部的事。

## 4. 公共底座 `_common.py`

**物理位置**：`notebook/C7 高级 RAG 技巧/6. 增强阶段/_common.py`
仅本章 3 节共用。**不向 4、5、7 节扩散**——若以后这些节也需要复用，应另起一份章级共享模块，而非把本章模块路径硬连过去。

### 4.1 进入 `_common.py` 的内容

| 函数 / 类 | 用途 | 首次在 notebook 中完整展示的位置 |
|---|---|---|
| `get_embeddings()` | 本地 bge-small 单例 | 不展示（前几章已讲） |
| `get_cleaned_pdf_documents()` | PDF 加载 + `clean_text` 后按页缓存 | 不展示（前几章已讲） |
| `open_or_build_chroma(persist_dir, docs, ids)` | Chroma 索引工厂 | 不展示（前几章已讲） |
| `llm_call(prompt, sleep_after=0)` | 智谱 ZhipuAI 调用 + 429 退避 | 不展示（前几章已讲） |
| `trim_context_to_budget(text, budget)` | 按 `CONTEXT_CHAR_BUDGET` 截断 | 不展示 |
| `build_rag_generation_prompt(question, context)` | 统一的 RAG 生成 prompt 模板 | 不展示（前几章已讲） |
| `simple_eval_2pt(answer, expected, question, *, prompt_template=None)` | 0~2 分裁判；`prompt_template` 允许各节传入定制 prompt | **6.1 完整 inline 重写** |
| `answer_from_context_fn(build_ctx_fn)` | 1 节专用胶水：把 `*_context` 接到 LLM 上 | **6.1 完整 inline 重写** |
| `run_shared_eval(answer_fn, qna_dict)` | 通用评测流水线（适配 1、2 节） | **6.1 完整 inline 重写** |
| `run_session_eval(system, qna_dict)` | 多轮 / 多源系统评测（顺次 `system.ask(q)`，保留状态） | **6.3 完整 inline 重写** |
| `build_compare_table(*dfs, names)` | 跨方法对比表 | **6.1 完整 inline 重写** |
| `load_qna_subset(qa_path, indices)` | 加载共用题集 | 不展示 |

### 4.2 不进入 `_common.py` 的内容（必须留在各节 notebook 显式可见）

- 6.1 节的 `sentence_map` / `neighbor_map` / `child_to_parent` / `leaf_per_parent` 等结构
- 6.2 节每个 `*_pipeline` 的控制流实现
- 6.3 节每个 system 类（`MemoryRAGSystem`、`MultiDocAgent`）

### 4.3 single source of truth 同步规则

`_common.py` 是 single source of truth。当某函数在 6.1 / 6.2 / 6.3 首次定义时，notebook cell 里给出**完整可运行实现**，`_common.py` 同时存一份**字符相同**的副本。

**code-and-spec sync rule**：修改任何 `_common.py` 函数 → 必须同步修改"首次出现该函数的 notebook cell"，反之亦然。CI / pre-commit 不强制（成本高），由实现期人工保证；spec self-review 阶段抽查。

## 5. 三节钩子 + 评估口径 + sleep 策略

| | **6.1 上下文增强** | **6.2 流程增强** | **6.3 系统增强** |
|---|---|---|---|
| 核心变化点 | context 拼法 | 检索-生成的流程编排 | 跨轮 / 跨源的状态 |
| 钩子签名 | `*_context(q) -> str` | `*_pipeline(q) -> str` | `class XxxSystem: ask(q) -> str` |
| inspect 重点 | 命中片段、扩展窗口、父块 | 每步决策的 trace（多轮检索、子问题、路由分支） | 状态快照（history、condensed query、路由决策） |
| 运行器 | `answer_from_context_fn` + `run_shared_eval` | `run_shared_eval(*_pipeline)`（直接喂 pipeline） | `run_session_eval(system)` |
| 评估口径 | 0~2 分（沿用 1 节当前 prompt） | **维度计分版** 0~2 分：每题列 2~4 个必答要点，每点 0/1，按要点覆盖率映射到 0~2 分 | **行为评估** + 0~2 分综合：Memory 检查后续答案是否引用上文、Multi-Doc 检查路由是否选对 source；接口仍是 0/1/2 |
| sleep 默认 | 0 秒 | 1 秒 | 1 秒 |

### 5.1 评估口径接口一致、内容定制

三节的评估接口都通过 `simple_eval_2pt(answer, expected, question, prompt_template=...)`，**返回都是 0/1/2**，因此：

- 同一份 `build_compare_table` 能拼接三节的结果
- 各节通过传入不同的 `prompt_template` 反映本节核心（要点覆盖 / 行为正确性）

### 5.2 trace 数据结构（6.2 用）

6.2 节定义并 inline 展示：

```python
@dataclass
class Step:
    kind: str         # "retrieve" / "draft" / "route" / "reflect" / "finalize"
    payload: dict     # 各 kind 自己定义内容
```

每个 `*_pipeline` 内部维护 `trace: list[Step]`，`inspect_*` 顺次打印。本结构不放 `_common.py`（只 6.2 用）。

## 6. 教学暴露规则

**总原则**：

- **新出现**：完整 inline 重写一次（让读者真正"建立"这个工具）
- **后续再出现**：`from _common import ...` 复用
- **前几章已讲过的**：不展示，直接 import 或一笔带过

### 6.1 三节展示密度

| 节 | 完整 inline 重写（首次出现的新概念） | 直接 `from _common import` | 不提（前几章已讲） |
|---|---|---|---|
| **6.1** | `simple_eval_2pt`、`answer_from_context_fn`、`run_shared_eval`、`build_compare_table` | — | `get_embeddings`、`get_cleaned_pdf_documents`、`open_or_build_chroma`、`llm_call`、`trim_context_to_budget`、`load_qna_subset` |
| **6.2** | `*_pipeline` 钩子约定（用 `iterative_pipeline` 完整示范）、`Step` 数据结构、维度计分 prompt | `run_shared_eval`、`simple_eval_2pt`（接口）、基础工具 | 同 6.1 |
| **6.3** | `MemoryRAGSystem`（首个 system 类完整示范）、`run_session_eval`、行为评估 prompt | `simple_eval_2pt`（接口）、基础工具 | 同 6.1 |

### 6.2 inline 与 import 的衔接写法

**6.1 首次定义 `simple_eval_2pt` 时**：notebook cell 完整 `def simple_eval_2pt(...)`（**不 import**），后续 6.1 内部直接调用。

**6.2 / 6.3 复用时**：每节开头加一段 markdown：

> 6.1 中介绍过的 `simple_eval_2pt` / `run_shared_eval` / `build_compare_table` 已收纳到 `_common.py`，本节直接 `import` 复用，不再展开。

后续代码：

```python
from _common import simple_eval_2pt, run_shared_eval, build_compare_table
```

读者无需翻回 6.1 即可理解"这些是 6.1 完整定义过的同名函数"。

## 7. 6.3 新增方法骨架

### 7.1 Memory（多轮对话 RAG）

```python
class MemoryRAGSystem:
    def __init__(self, retriever, max_history=5):
        self.retriever = retriever
        self.history: list[tuple[str, str]] = []
        self.max_history = max_history

    def _condense(self, q: str) -> str:
        """LLM 把 (history + 当前 q) 改写成独立的 standalone query。"""
        if not self.history:
            return q
        hist_text = "\n".join(f"Q: {hq}\nA: {ha}" for hq, ha in self.history)
        prompt = (
            "下面是历史对话。请把最新的问题改写成一个不依赖历史的独立问题，"
            "保留所有指代消解后的实体名。只输出改写后的问题，不要任何前缀。\n\n"
            f"历史：\n{hist_text}\n\n最新问题：{q}\n\n改写："
        )
        return llm_call(prompt).strip()

    def ask(self, q: str) -> str:
        condensed = self._condense(q)
        docs = self.retriever.invoke(condensed)
        ctx = "\n\n".join(d.page_content for d in docs)
        ans = llm_call(build_rag_generation_prompt(q, ctx))
        self.history.append((q, ans))
        self.history = self.history[-self.max_history:]
        return ans
```

- **inspect 重点**：condensed query vs 原 query；history 长度；命中是否因为 condensed 改写而变化
- **评测题集**：构造 3~4 个"多轮对话脚本"（每脚本 3~4 轮，后续轮包含"它"、"上面提到的方法"等指代），手工标准答案，存 `data/memory_sessions.json`
- **行为评估 prompt 要点**：要求裁判判断"模型答案是否正确处理了指代"，再综合答案正确性给 0~2 分

### 7.2 Multi-Document Agent（多源路由）

```python
class MultiDocAgent:
    def __init__(self, sources: dict[str, "Retriever"], descriptions: dict[str, str]):
        """sources 例如 {'ch2': retriever_ch2, ...}; descriptions 提供给路由 LLM。"""
        self.sources = sources
        self.descriptions = descriptions

    def _route(self, q: str) -> list[str]:
        """LLM 根据 q 与 source 描述选择 1~N 个 source 名。"""
        desc_text = "\n".join(f"- {k}: {v}" for k, v in self.descriptions.items())
        prompt = (
            "你是一个路由器。给定可选的知识源描述与用户问题，"
            "请输出 1~3 个最相关的 source key（每行一个），不要任何其它内容。\n\n"
            f"可选源：\n{desc_text}\n\n问题：{q}\n\n选择："
        )
        out = llm_call(prompt).strip().splitlines()
        return [k.strip() for k in out if k.strip() in self.sources][:3]

    def ask(self, q: str) -> str:
        chosen = self._route(q) or list(self.sources.keys())[:1]
        docs = []
        for name in chosen:
            docs.extend(self.sources[name].invoke(q))
        ctx = "\n\n".join(d.page_content for d in docs)
        return llm_call(build_rag_generation_prompt(q, ctx))
```

- **source 构造**：把南瓜书按章节切分（按 PDF 目录页范围或章节正则），每章节建独立 Chroma 集合 `./chroma_db/multi_doc/ch{n}`
- **inspect 重点**：路由决策（选了哪些 source、为什么）；各 source 的命中片段
- **评测题集**：从 `train_dataset.json` 挑跨章节题与单章节题各 3~5 道（依据 `page_num` 字段判断所属章节），看路由是否选中正确章节
- **行为评估 prompt 要点**：要求裁判检查"模型实际用到的 source 是否包含问题真正需要的章节"，再综合答案正确性给 0~2 分

## 8. 物理产出

### 8.1 新增

- `notebook/C7 高级 RAG 技巧/6. 增强阶段/_common.py`
- `notebook/C7 高级 RAG 技巧/6. 增强阶段/data/memory_sessions.json`（手工 3~4 个对话脚本）

### 8.2 重写

- `1. 上下文增强.ipynb`：抽出公共底座到 `_common.py` 并 import；首次定义 `simple_eval_2pt` / `answer_from_context_fn` / `run_shared_eval` / `build_compare_table`；其它内容（钩子签名、Sentence Window / Small-to-Big / AutoMerging 实现、同题对比表、选择指南）保持不变。预计代码量减少 20~30%。
- `2. 流程增强.ipynb`：钩子升级为 `*_pipeline`；删硬编码"对偶问题"单题与 0~10 分自由文本评估；引入 `Step` trace；评估口径改为维度计分版的 `simple_eval_2pt`；新增跨方法对比表（`build_compare_table`）；删除 `time.sleep(20)`，改用 `llm_call(..., sleep_after=1)`；用 `_common` 替换所有重复底座。预计代码量减少 30~40%。
- `3. 系统增强.ipynb`：从零写 `MemoryRAGSystem` 与 `MultiDocAgent`；定义 `run_session_eval`；定义行为评估 prompt；包含 inspect 与对比表。

### 8.3 不动

- 第 3、4、5、7 章的所有内容
- 现有 6.1 的方法实现（Sentence Window / Small-to-Big / AutoMerging 算法本身、`CONTEXT_CHAR_BUDGET` 默认值、20 题子集等）
- 现有 6.2 的 6 种方法的核心算法（仅重新组织代码结构与评估，不改算法）
- `readme.md`（或仅微调 6.3 的方法清单与本设计对齐）

## 9. 验收标准

1. **架构**：三节 import `_common`，搜索三节代码不存在重复的 `clean_text` / `get_embeddings` / `llm_call` / `simple_eval` 等定义。
2. **钩子可见性**：每节 notebook 的方法实现单元只剩"本节核心变化"——读者能在 30 行内看完一个方法的 `*_context` / `*_pipeline` / `XxxSystem`。
3. **教学暴露**：每个 `_common` 函数在其首次出现的节里，notebook 内可见到完整 `def`（与 `_common.py` 内字符相同）；后续节复用时用 `import` 且有一段衔接 markdown。
4. **评估**：三节都有"同题对比表"或"行为评估表"作为最后一节；`simple_eval_2pt` 接口三节一致，`build_compare_table` 能拼接。
5. **6.3 可运行**：
   - Memory：在 4 轮对话中至少 1 轮满足「`condensed query != 原 query`」且「retrieve 命中集合相对单题问询发生变化」（客观可观测信号），并且裁判 0~2 分相对"无 memory baseline"提升。
   - Multi-Doc：在 6 道题中至少 4 道路由结果包含问题真正所属的章节 source（按 `train_dataset.json.page_num` 反推章节）。
6. **节流**：6.1 默认不 sleep；6.2、6.3 默认 `sleep_after=1`；三节均无硬编码 `time.sleep(20)`。
7. **代码量**：三节合计代码行数（不含 markdown）相对当前减少 ≥ 25%。
8. **文档同步**：6 章 `readme.md` 中的方法清单与本设计一致（或本设计在范围注明保留 readme 不改）。

## 10. 后续扩展（不在本轮）

- 第 3 章新增 `5.Contextual_Retrieval.ipynb`（CCH 的 chunk 级精细化版本）与 `6.RAPTOR.ipynb`（层级聚类摘要）—— 单开 spec
- 6.3 节补 `GraphRAG`（待图数据库依赖确定后）与 `Agentic RAG`（待与 6.2 边界更明确后）
- 仓库根目录临时脚本（`final_fix_nb.py` 等）的 housekeeping 提交
- `chroma_db/` / `models/` / 评估产物 CSV/JSON 的 `.gitignore` 整理

## 11. 审批

- **产品 / 教学负责人**：待用户确认。
- 确认后进入实现计划（writing-plans skill）。
