# C7 6.2 流程增强题目完善 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完善 `2. 流程增强.ipynb` 的评测题、展示题和少量控制题设计，让每种流程增强方法在保持原始方法行为不变的前提下，都有真实、清晰、可复现且能说明“解决了什么问题”的对比展示。

**Architecture:** 保持 6.1 的教程风格：主评测表只使用 `train_dataset.json` 中的真实题；每种方法另设代表题，通过 trace、baseline 对比和证据差异展示机制有效性；少量额外题只能来自课程语料/真实参考答案的自然问题，不允许为了触发现象改写方法 prompt 或 pipeline。实现上优先在 notebook 内新增轻量题目选择、demo case 和结果面板，不引入新的复杂框架。

**Method Fidelity Rule:** 6.2 是方法复现教程。不得为了演示效果修改原始方法的核心 prompt、控制流、停止条件或判别逻辑。若某个 demo 不理想，先扩大自然题候选池、选择更合适的真实题、调整非方法性的展示参数（如候选题顺序、展示 top-k 文本），或从课程语料中构造自然的非评分案例；不要把“没触发”作为最终教学交付。

**Tech Stack:** Python 3.10、Jupyter notebook、pandas、LangChain retriever、Chroma、智谱 GLM 调用、现有 `_common.py` 公共底座。

---

## File Structure

**Files to inspect:**
- `notebook/C7 高级 RAG 技巧/6. 增强阶段/1. 上下文增强.ipynb` — 对齐 6.1 的主表、分方法面板、真实提分/边界样例风格。
- `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb` — 本计划的主要修改对象。
- `notebook/C7 高级 RAG 技巧/6. 增强阶段/_common.py` — 复用 `load_qna_subset`、`run_shared_eval`、`build_compare_table`、`trim_context_to_budget`、`build_rag_generation_prompt` 等公共功能。
- `notebook/C7 高级 RAG 技巧/3. 索引阶段/data/train_dataset.json` — 主评测题和大多数 demo case 的来源。

**Files to modify:**
- `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`
  - 调整 `QA_INDICES` 为流程增强专用主集。
  - 新增 `baseline_pipeline`。
  - 新增 `DEMO_CASES` / `EXTRA_DEMO_CASES`。
  - 把各 `inspect_*` 从默认第一题改为方法专用代表题。
  - Routing 改为两个真实不同 retriever。
  - 增加分方法真实差异面板。
  - 把最终 summary 改为从当前 `compare_df` 生成。

**Optional output files:**
- `notebook/C7 高级 RAG 技巧/6. 增强阶段/data/flow_enhance_compare_table.csv` — 若 notebook 当前风格需要落盘，则保存主评测表。
- `notebook/C7 高级 RAG 技巧/6. 增强阶段/data/flow_enhance_eval_summary.json` — 若需要与 6.1 对齐，则保存汇总、demo case 和 trace 检查结果。

Do not create new documentation files beyond this plan unless the user explicitly asks.

---

## Task 1: Audit Current Question Set and Candidate Pool

**Files:**
- Inspect: `notebook/C7 高级 RAG 技巧/3. 索引阶段/data/train_dataset.json`
- Inspect: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`
- Modify: none

- [ ] **Step 1: Locate the current 6.2 question definitions**

Open `2. 流程增强.ipynb` and find the cell that currently defines:

```python
QA_INDICES_CONCEPT = [0, 1, 3, 4, 7]
QA_INDICES_DERIVE = [12, 15, 18, 19, 30, 86]
QA_INDICES = QA_INDICES_CONCEPT + QA_INDICES_DERIVE
qna_dict = load_qna_subset(QA_PATH, QA_INDICES)
```

Expected: confirm the notebook currently uses 11 questions and prints a message like `本节使用 11 道题`.

- [ ] **Step 2: Generate a candidate question audit table**

Run this command from the repository root to list candidate questions with heuristic labels:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('notebook/C7 高级 RAG 技巧/3. 索引阶段/data/train_dataset.json')
data = json.loads(p.read_text(encoding='utf-8'))
for i, item in enumerate(data):
    q = item.get('query', '') or ''
    a = item.get('answer', '') or ''
    if not q.strip():
        continue
    signals = []
    if any(x in q for x in ['推导', '证明', '计算', '表达式', '公式', '二阶导数', '海森', '矩阵', 'NumPy']):
        signals.append('derive/calc')
    if any(x in q for x in ['解释', '区别', '为什么', '说明', '比较']):
        signals.append('concept/compare')
    if any(x in q for x in ['并', '同时', '各自', '请简述', '并说明', '给出']):
        signals.append('multi-aspect')
    if len(a) >= 300:
        signals.append('long-answer')
    print(f'{i:03d}\tp{item.get("page_num", "")}\tqlen={len(q)}\talen={len(a)}\t{"|".join(signals)}\t{q[:100]}')
PY
```

Expected: output includes at least indices `0, 1, 4, 5, 7, 12, 15, 18, 19, 36, 42, 62, 63, 86` with non-empty queries.

- [ ] **Step 3: Record the initial candidate pool for implementation**

Use this candidate pool during the remaining tasks:

```python
CANDIDATE_INDICES = [
    0,   # 算法/模型，简单 baseline
    1,   # 泛化能力，长概念多要点
    4,   # 模型评估、经验误差、泛化误差
    5,   # 交叉验证
    7,   # 宏平均/微平均，可能混入 ROC 噪声
    12,  # 期望风险交叉项
    15,  # NumPy 向量化
    18,  # 牛顿法推导 + 实际限制
    19,  # Hessian 推导
    36,  # KKT 条件
    42,  # SVM 核函数
    62,  # JC/FMI/RI 聚类指标比较
    63,  # EM/GMM 参数估计
    86,  # MM 优化
]
```

No code change is required in this task.

- [ ] **Step 4: Commit audit notes only if a file was intentionally created**

Default: do not commit anything for this task. If you created a temporary audit file, delete it before continuing.

---

## Task 2: Replace the Main Evaluation Set with a 12-Question Flow Set

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`

- [ ] **Step 1: Update the QA index cell**

Replace the current question-set cell with this exact content:

```python
# 6.2 流程增强专用主评测集：只用于本节内部比较，不与 6.1 分数直接横比
QA_INDICES_CONCEPT = [0, 1, 4, 5, 7]
QA_INDICES_DERIVE = [12, 15, 18, 19, 36, 42, 86]
QA_INDICES = QA_INDICES_CONCEPT + QA_INDICES_DERIVE
qna_dict = load_qna_subset(QA_PATH, QA_INDICES)
print(f"✅ 本节使用 {len(qna_dict)} 道题（{len(QA_INDICES_CONCEPT)} 概念/比较 + {len(QA_INDICES_DERIVE)} 推导/复杂题）")
print("   本节分数只用于 6.2 内部方法比较；因题集和裁判口径不同，不与 6.1 直接横比。")
print("   想跑得快可只用前 3 道：QA_INDICES = QA_INDICES_CONCEPT[:3]")
```

Expected: notebook now uses 12 questions.

- [ ] **Step 2: Add a short markdown explanation before the cell**

Insert this markdown immediately before the updated QA cell:

```markdown
### 本节题集：流程增强敏感题

6.2 的题集不直接沿用 6.1 的 24 题自然主集，而是选取 12 道更容易暴露“流程差异”的题：概念/比较题负责观察多要点覆盖，推导/复杂题负责观察补检索、拆解、路由和自反思。

注意：本节使用维度计分 prompt，且题集与 6.1 不同，所以分数只用于 6.2 内部横向比较，不与 6.1 的上下文增强分数直接比较。
```

Expected: reader understands why 6.2 does not reuse 6.1's full set.

- [ ] **Step 3: Run the updated QA cell only**

In Jupyter or VS Code, run the markdown + QA cell.

Expected output:

```text
✅ 本节使用 12 道题（5 概念/比较 + 7 推导/复杂题）
本节分数只用于 6.2 内部方法比较；因题集和裁判口径不同，不与 6.1 直接横比。
```

- [ ] **Step 4: Commit**

```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb"
git commit -m "docs(c7): refine flow enhancement question set"
```

Expected: commit succeeds.

---

## Task 3: Add Baseline Pipeline for Real Method Comparisons

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`

- [ ] **Step 1: Add baseline pipeline code after `print_trace`**

Find the cell defining `step` and `print_trace`. After `print_trace`, add:

```python
def baseline_pipeline(question: str) -> tuple[str, list[dict]]:
    trace: list[dict] = []
    docs = retriever.invoke(question)
    ctx = trim_context_to_budget("\n\n".join(d.page_content for d in docs), CONTEXT_CHAR_BUDGET)
    trace.append(step("retrieve", branch="baseline", n_hits=len(docs), context_chars=len(ctx)))
    answer = llm_call(build_rag_generation_prompt(question, ctx))
    trace.append(step("finalize", reason="single_pass"))
    return answer, trace


def inspect_baseline(question: str) -> None:
    print(f"❓ 问题: {question}\n")
    answer, trace = baseline_pipeline(question)
    print(f"🧠 最终答案：\n{answer}\n")
    print("🔍 流程 trace：")
    print_trace(trace)
```

Expected: baseline has the same `(answer, trace)` interface as all 6.2 methods.

- [ ] **Step 2: Add a baseline markdown section**

Before “方法 1：迭代检索”, insert:

```markdown
## Baseline：单次检索 + 单次生成

在看流程增强之前，先保留一个最朴素的对照组：`retrieve → generate`。后面所有方法都必须和它比较，否则无法判断“多走一步”到底带来了改进，还是只是增加了成本。

Baseline 也使用本节共用的 retriever、同样的 `CONTEXT_CHAR_BUDGET` 和同样的维度计分裁判。
```

- [ ] **Step 3: Add a baseline inspect cell**

After the baseline markdown, add a code cell:

```python
inspect_baseline(list(qna_dict.keys())[0])
```

Expected: running this cell prints a two-step trace: `retrieve`, `finalize`.

- [ ] **Step 4: Update the full evaluation cell to include baseline**

Replace the full evaluation cell with:

```python
baseline_df  = eval_pipeline(baseline_pipeline,  qna_dict)
iterative_df = eval_pipeline(iterative_pipeline, qna_dict)
recursive_df = eval_pipeline(recursive_pipeline, qna_dict)
routing_df   = eval_pipeline(routing_pipeline,   qna_dict)
adaptive_df  = eval_pipeline(adaptive_pipeline,  qna_dict)
crag_df      = eval_pipeline(crag_pipeline,      qna_dict)
selfrag_df   = eval_pipeline(self_rag_pipeline,  qna_dict)
```

- [ ] **Step 5: Update the compare table cell to include baseline**

Replace the compare table cell with:

```python
compare_df = build_compare_table(
    [baseline_df, iterative_df, recursive_df, routing_df, adaptive_df, crag_df, selfrag_df],
    names=["baseline", "iterative", "recursive", "routing", "adaptive", "crag", "self_rag"],
)
compare_df
```

Expected: `compare_df` has columns `question`, `baseline`, `iterative`, `recursive`, `routing`, `adaptive`, `crag`, `self_rag`.

- [ ] **Step 6: Commit**

```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb"
git commit -m "feat(c7): add flow enhancement baseline pipeline"
```

Expected: commit succeeds.

---

## Task 4: Add Method-Specific Demo Cases

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`

- [ ] **Step 1: Add helper functions for demo question lookup**

After the QA cell, add:

```python
DEMO_CASES = {
    "baseline": [0],
    "iterative": [18, 19, 42],
    "recursive": [4, 42, 63],
    "routing_general": [5],
    "routing_math": [19],
    "adaptive_simple": [14],
    "adaptive_moderate": [4],
    "adaptive_complex": [42],
    "crag": [7, 62],
    "self_rag": [1, 62, 63],
}

EXTRA_DEMO_CASES = {
    "crag_noise": "请解释宏平均和微平均的区别，不要讨论 ROC 曲线、阈值选择或 AUC。",
}

_all_demo_indices = sorted({idx for values in DEMO_CASES.values() for idx in values})
demo_qna_dict = load_qna_subset(QA_PATH, _all_demo_indices)


def demo_question(case_name: str, pos: int = 0) -> str:
    idx = DEMO_CASES[case_name][pos]
    qna = load_qna_subset(QA_PATH, [idx])
    return next(iter(qna.keys()))


def extra_demo_question(case_name: str) -> str:
    return EXTRA_DEMO_CASES[case_name]

print(f"✅ 方法展示题已准备：{len(_all_demo_indices)} 道数据集题 + {len(EXTRA_DEMO_CASES)} 道控制题")
```

Expected: helper functions can fetch fixed demo questions by method.

- [ ] **Step 2: Change baseline inspect call**

Replace:

```python
inspect_baseline(list(qna_dict.keys())[0])
```

with:

```python
inspect_baseline(demo_question("baseline"))
```

- [ ] **Step 3: Change iterative inspect call**

Replace:

```python
inspect_iterative(list(qna_dict.keys())[0])
```

with:

```python
inspect_iterative(demo_question("iterative"))
```

- [ ] **Step 4: Change recursive inspect call**

Replace:

```python
inspect_recursive(list(qna_dict.keys())[0])
```

with:

```python
inspect_recursive(demo_question("recursive"))
```

- [ ] **Step 5: Change routing inspect call**

Replace:

```python
inspect_routing(list(qna_dict.keys())[0])
```

with:

```python
inspect_routing(demo_question("routing_general"))
inspect_routing(demo_question("routing_math"))
```

Expected: routing section shows one general case and one math case.

- [ ] **Step 6: Change adaptive inspect call**

Replace:

```python
inspect_adaptive(list(qna_dict.keys())[0])
```

with:

```python
inspect_adaptive(demo_question("adaptive_simple"))
inspect_adaptive(demo_question("adaptive_moderate"))
inspect_adaptive(demo_question("adaptive_complex"))
```

Expected: adaptive section attempts to show simple, moderate, and complex delegation.

- [ ] **Step 7: Change CRAG inspect call**

Replace:

```python
inspect_crag(list(qna_dict.keys())[0])
```

with:

```python
inspect_crag(demo_question("crag"))
inspect_crag(extra_demo_question("crag_noise"))
```

Expected: CRAG section shows a natural dataset case and one control case that should make noise filtering easier to observe.

- [ ] **Step 8: Change Self-RAG inspect call**

Replace:

```python
inspect_self_rag(list(qna_dict.keys())[0])
```

with:

```python
inspect_self_rag(demo_question("self_rag"))
```

Expected: Self-RAG section uses a long-answer/multi-metric case instead of the first simple question.

- [ ] **Step 9: Commit**

```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb"
git commit -m "docs(c7): add flow enhancement demo cases"
```

Expected: commit succeeds.

---

## Task 5: Make Routing Use Two Real Retriever Branches

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`

- [ ] **Step 1: Replace the routing simplification markdown**

Find the paragraph that says both `math_index` and `general_index` point to the same retriever. Replace it with:

```markdown
> **本节的教学实现**：为了让路由有真实差异，我们仍然使用同一本南瓜书，但构造两个不同检索配置：
> - `general_index`：较大 chunk、较少 top-k，适合概念解释题；
> - `math_index`：较小 chunk、更多 top-k，适合公式、推导、矩阵类题。
>
> 这仍然不是生产中的完整多源路由；真实系统里可以把两个 branch 换成不同知识库、关键词索引、SQL 或工具调用。
```

- [ ] **Step 2: Replace `INDEX_TO_RETRIEVER` definition**

Replace:

```python
INDEX_TO_RETRIEVER = {
    "math_index": retriever,
    "general_index": retriever,
}
```

with:

```python
general_retriever = build_retriever(chunk_size=512, chunk_overlap=60, k=2)
math_retriever = build_retriever(chunk_size=128, chunk_overlap=20, k=8)

INDEX_TO_RETRIEVER = {
    "math_index": math_retriever,
    "general_index": general_retriever,
}
```

Expected: routing branches now use different chunking and top-k settings.

- [ ] **Step 3: Improve routing trace detail**

In `routing_pipeline`, replace:

```python
trace.append(step("retrieve", branch=branch, n_hits=len(docs)))
```

with:

```python
trace.append(step(
    "retrieve",
    branch=branch,
    n_hits=len(docs),
    context_chars=len(ctx),
    first_hit=docs[0].page_content[:80] if docs else "",
))
```

Expected: trace makes branch differences visible without printing huge contexts.

- [ ] **Step 4: Run routing inspect cells**

Run the routing section cells.

Expected:
- `routing_general` case should route to `general_index`.
- `routing_math` case should route to `math_index`.
- Trace `n_hits` should usually differ: general branch `2`, math branch `8`.

- [ ] **Step 5: If routing branch is wrong, record it as a boundary example rather than hiding it**

If the router sends a case to the wrong branch, keep the output and update the local markdown under routing with this sentence:

```markdown
如果 router 在某次运行中选错 branch，不要把它当成 notebook 错误；这正是查询路由的核心风险：路由层一旦误判，后面的检索和生成都会被带偏。
```

- [ ] **Step 6: Commit**

```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb"
git commit -m "feat(c7): use distinct retrievers for routing demo"
```

Expected: commit succeeds.

---

## Task 6: Add Trace Validation Helpers and Method Panels

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`

- [ ] **Step 1: Add method comparison helper functions**

After `eval_pipeline`, add:

```python
def flow_method_compare(method_df, method_name):
    return build_compare_table(
        [baseline_df, method_df],
        names=["baseline", method_name],
    )


def flow_method_diff_summary(method_df, method_name):
    df = flow_method_compare(method_df, method_name)
    wins = df.index[df[method_name] > df["baseline"]].tolist()
    regressions = df.index[df[method_name] < df["baseline"]].tolist()
    ties = df.index[df[method_name] == df["baseline"]].tolist()
    return df, wins, regressions, ties


def short_text(text, max_chars=80):
    text = " ".join(str(text).split())
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def flow_score_summary(compare_df):
    methods = [c for c in compare_df.columns if c != "question"]
    rows = []
    baseline = compare_df["baseline"]
    for method in methods:
        scores = compare_df[method]
        rows.append({
            "method": method,
            "mean_score_0_2": float(scores.mean()),
            "total_score": int(scores.sum()),
            "strict_wins_vs_baseline": 0 if method == "baseline" else int((scores > baseline).sum()),
            "regressions_vs_baseline": 0 if method == "baseline" else int((scores < baseline).sum()),
            "ties_vs_baseline": len(compare_df) if method == "baseline" else int((scores == baseline).sum()),
        })
    return pd.DataFrame(rows)
```

Expected: helpers mirror 6.1's comparison style but stay local to 6.2.

- [ ] **Step 2: Add trace validation helpers**

After the comparison helpers, add:

```python
def trace_has_kind(trace: list[dict], kind: str) -> bool:
    return any(s.get("kind") == kind for s in trace)


def trace_kinds(trace: list[dict]) -> list[str]:
    return [s.get("kind", "") for s in trace]


def validate_demo_trace(method_name: str, trace: list[dict]) -> dict:
    kinds = trace_kinds(trace)
    if method_name == "iterative":
        ok = kinds.count("retrieve") >= 2 or any(s.get("reason") == "max_rounds_reached" for s in trace)
        expected = "出现第二轮 retrieve，或明确到达 max_rounds"
    elif method_name == "recursive":
        ok = trace_has_kind(trace, "decompose") and trace_has_kind(trace, "sub_answer")
        expected = "出现 decompose 和 sub_answer"
    elif method_name == "routing":
        ok = trace_has_kind(trace, "route") and trace_has_kind(trace, "retrieve")
        expected = "出现 route 和对应 branch retrieve"
    elif method_name == "adaptive":
        ok = trace_has_kind(trace, "classify") and trace_has_kind(trace, "delegate")
        expected = "出现 classify 和 delegate"
    elif method_name == "crag":
        tags = [s.get("tag") for s in trace if s.get("kind") == "grade"]
        ok = any(t in {"partial", "irrelevant"} for t in tags)
        expected = "至少一条证据被判为 partial 或 irrelevant"
    elif method_name == "self_rag":
        verdicts = [s.get("verdict") for s in trace if s.get("kind") == "reflect"]
        ok = "CONTINUE" in verdicts or "FINISH" in verdicts
        expected = "出现 reflect verdict；若没有 CONTINUE，需要在解读中说明教学版局限"
    else:
        ok = bool(trace)
        expected = "trace 非空"
    return {
        "method": method_name,
        "ok": ok,
        "expected": expected,
        "kinds": " → ".join(kinds),
    }
```

Expected: demo validation focuses on mechanism visibility, not just score.

- [ ] **Step 3: Add a method panel display helper**

Add:

```python
def display_flow_method_panel(method_label, method_name, method_df, demo_pipeline, demo_question_text):
    df, wins, regressions, ties = flow_method_diff_summary(method_df, method_name)
    total = int(df[method_name].sum())
    base_total = int(df["baseline"].sum())
    max_score = len(df) * 2
    answer, trace = demo_pipeline(demo_question_text)
    validation = validate_demo_trace(method_name, trace)

    print(f"### {method_label} 真实结果")
    print(f"Baseline 总分: {base_total}/{max_score}")
    print(f"{method_label} 总分: {total}/{max_score} ({total - base_total:+d})")
    print(f"严格提分题: {wins if wins else '无'}")
    print(f"边界/回退题: {regressions if regressions else '无'}")
    print(f"持平题数: {len(ties)}")
    print(f"\n代表题: {demo_question_text}")
    print(f"Trace 检查: {'通过' if validation['ok'] else '需解读'} — {validation['expected']}")
    print(f"Trace 形状: {validation['kinds']}\n")
    print_trace(trace)
```

Expected: method panel can be called after all DataFrames exist.

- [ ] **Step 4: Add a method panel section after `compare_df`**

After the compare table, add markdown:

```markdown
## 分方法真实差异面板

总表只能说明分数变化，不能说明“流程为什么不同”。下面每个方法都选一个代表题，展示真实 trace，并给出它相对 baseline 的提分题、回退题和持平题。
```

Then add a code cell:

```python
display(flow_score_summary(compare_df))

display_flow_method_panel("Iterative", "iterative", iterative_df, iterative_pipeline, demo_question("iterative"))
display_flow_method_panel("Recursive", "recursive", recursive_df, recursive_pipeline, demo_question("recursive"))
display_flow_method_panel("Routing", "routing", routing_df, routing_pipeline, demo_question("routing_math"))
display_flow_method_panel("Adaptive", "adaptive", adaptive_df, adaptive_pipeline, demo_question("adaptive_complex"))
display_flow_method_panel("CRAG", "crag", crag_df, crag_pipeline, demo_question("crag"))
display_flow_method_panel("Self-RAG", "self_rag", selfrag_df, self_rag_pipeline, demo_question("self_rag"))
```

Expected: after full evaluation, the notebook prints per-method summary and trace.

- [ ] **Step 5: Commit**

```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb"
git commit -m "docs(c7): add flow method comparison panels"
```

Expected: commit succeeds.

---

## Task 7: Replace Static Final Results with Dynamic Summary

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`

- [ ] **Step 1: Replace the static “实测结果与解读” section**

Find the markdown section starting with:

```markdown
## 实测结果与解读（本节作者实跑数据，供对照）
```

Replace the static table and fixed claims with:

```markdown
## 实测结果与解读

下面的汇总来自当前 notebook 的 `compare_df`，因此会随模型输出、限流重试、题集调整和随机波动略有变化。读这张表时不要只看均值，更要结合上面的分方法 trace：流程增强的教学重点是“哪一步做了不同决策”。
```

- [ ] **Step 2: Add dynamic summary code**

Immediately after the new markdown, add:

```python
summary_df = flow_score_summary(compare_df)
display(summary_df)

baseline_mean = float(compare_df["baseline"].mean())
rows = []
for method in ["iterative", "recursive", "routing", "adaptive", "crag", "self_rag"]:
    method_mean = float(compare_df[method].mean())
    rows.append({
        "方法": method,
        "均值": round(method_mean, 3),
        "Δ vs baseline": round(method_mean - baseline_mean, 3),
        "严格提分题数": int((compare_df[method] > compare_df["baseline"]).sum()),
        "回退题数": int((compare_df[method] < compare_df["baseline"]).sum()),
    })

display(pd.DataFrame(rows))
```

Expected: final section derives results from the current run.

- [ ] **Step 3: Replace fixed “5 条发现” with conditional reading guidance**

Replace the fixed numbered findings with:

```markdown
### 如何解读本轮结果

1. 如果 `iterative` 提分，优先查看它的 trace 是否真的发生了第二轮检索；如果没有第二轮却提分，说明收益可能来自生成随机性，而不是补检索。
2. 如果 `recursive` 在复杂题上提分，查看 `decompose` 是否拆出了互补子问题；如果在简单题回退，通常是过度拆解造成的。
3. 如果 `routing` 与 baseline 接近，先看两个 branch 的 `first_hit` 和 `n_hits` 是否不同；如果 branch 不同但得分相同，说明路由改变了上下文但未改变裁判分数。
4. 如果 `crag` 提分，查看被过滤的 `partial/irrelevant` 证据；如果回退，可能是 grader 把有用片段降权或过滤了。
5. 如果 `self_rag` 很少触发 `CONTINUE`，这是教学化简版的正常局限：通用 LLM 往往过早自信，不像论文原版有专门 reflection token。
```

Expected: final interpretation no longer depends on stale hard-coded run numbers.

- [ ] **Step 4: Commit**

```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb"
git commit -m "docs(c7): derive flow enhancement summary from current run"
```

Expected: commit succeeds.

---

## Task 8: Mine Effective Natural Demo Cases Without Changing Methods

**Files:**
- Inspect: `notebook/C7 高级 RAG 技巧/3. 索引阶段/data/train_dataset.json`
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`

**Goal:** 找到每个方法的有效自然示例：示例必须展示该方法相对 baseline 解决了什么具体问题，而不是只打印 trace 或简单说“本轮没触发”。不得修改任何方法的核心 prompt、pipeline、停止条件或 grader 逻辑。

- [ ] **Step 1: Add a broad natural candidate pool for demo mining**

In the demo-case cell, add this list below `DEMO_CASES`:

```python
DEMO_MINING_CANDIDATES = {
    "iterative": [12, 18, 19, 36, 42, 46, 50, 53, 60, 62, 63, 67, 72, 80, 86, 98, 102, 108, 117],
    "recursive": [4, 7, 42, 62, 63, 86, 101, 113, 117],
    "routing_general": [1, 5, 7, 14, 54, 62, 68, 84, 85, 101, 114, 115],
    "routing_math": [12, 15, 18, 19, 36, 42, 46, 50, 63, 80, 86, 102, 108, 117],
    "adaptive_simple": [14, 22, 33, 35, 84, 85, 107, 114],
    "adaptive_moderate": [4, 5, 7, 17, 38, 54, 62, 68, 101],
    "adaptive_complex": [19, 42, 46, 63, 80, 86, 102, 108, 117],
    "crag": [7, 9, 16, 20, 38, 49, 54, 62, 68, 85, 101, 115],
    "self_rag": [1, 7, 19, 42, 62, 63, 67, 86, 98, 101, 113, 117],
}
```

Expected: the candidate pool is broad enough to mine examples, but all candidates are natural dataset questions.

- [ ] **Step 2: Add a trace mining helper that does not alter method behavior**

Add this helper near the trace validation helpers:

```python
def mine_demo_case(method_name: str, pipeline_fn, candidate_indices: list[int], *, require_score_gain: bool = False):
    rows = []
    candidate_qna = load_qna_subset(QA_PATH, candidate_indices)
    for idx, question in zip(candidate_indices, candidate_qna.keys()):
        answer, trace = pipeline_fn(question)
        validation = validate_demo_trace(method_name, trace)
        row = {
            "idx": idx,
            "question": question,
            "ok": validation["ok"],
            "trace_shape": validation["kinds"],
            "answer_head": short_text(answer, 120),
        }
        if require_score_gain and "baseline_df" in globals():
            pass
        rows.append(row)
    return pd.DataFrame(rows)
```

Expected: this helper only runs existing pipelines and records traces. It must not change prompts, retriever parameters, or pipeline logic.

- [ ] **Step 3: Mine Iterative candidates until a real two-step example is found**

Run:

```python
iterative_mining_df = mine_demo_case("iterative", iterative_pipeline, DEMO_MINING_CANDIDATES["iterative"])
display(iterative_mining_df[["idx", "ok", "trace_shape", "question"]])
```

Success criteria:

```text
At least one row has ok == True because trace has two retrieve steps or max_rounds_reached.
```

If a row succeeds, set `DEMO_CASES["iterative"]` so that index is first. Example:

```python
"iterative": [FOUND_IDX, 18, 19, 42],
```

Then add a markdown note under Iterative:

```markdown
代表题来自数据集自然题，不修改 iterative prompt 或停止条件。它展示的是原方法在多要点/长推导问题上如何用第一轮草答的缺口线索驱动第二轮检索。
```

If no row succeeds, do not change the method. Continue to Step 4 to construct a natural course-material case.

- [ ] **Step 4: If dataset mining fails, create a natural non-scored course-material Iterative case**

Only if Step 3 finds no effective example, add this to `EXTRA_DEMO_CASES`:

```python
"iterative_course_natural": "请根据南瓜书相关内容解释牛顿法的基本迭代公式，并说明它与梯度下降法在选择下一个迭代点时的区别。"
```

Then run:

```python
inspect_iterative(extra_demo_question("iterative_course_natural"))
```

Success criteria:

```text
The trace shows either a second retrieve or max_rounds_reached, and the answer/trace explains a real missing aspect such as Newton formula vs gradient descent comparison.
```

If it succeeds, use this only in the Iterative inspect cell and add markdown:

```markdown
这道展示题不进入主评测均值；它来自课程语料中的自然学习问题，用来稳定观察原始 iterative 流程如何把“缺少对比/推导细节”的中间判断转成第二轮检索线索。
```

If it still does not succeed, add one more natural course-material case, not an artificial checklist:

```python
"iterative_course_svm": "请根据南瓜书相关内容解释 SVM 中核函数的作用，并说明引入核函数后优化问题会发生什么变化。"
```

Run it once. Choose the first course-material case that shows the iterative mechanism.

- [ ] **Step 5: Mine Recursive candidates for a meaningful decomposition example**

Run:

```python
recursive_mining_df = mine_demo_case("recursive", recursive_pipeline, DEMO_MINING_CANDIDATES["recursive"])
display(recursive_mining_df[["idx", "ok", "trace_shape", "question"]])
```

Success criteria:

```text
The selected row has decompose + at least two sub_answer steps, and the subquestions correspond to distinct requirements rather than paraphrases.
```

Prefer index `4` if it decomposes into model evaluation, empirical error, generalization error, and which error to minimize. Otherwise choose the first row that has distinct subquestions and a better answer than baseline in `compare_df` or a clear trace-level explanation.

- [ ] **Step 6: Mine CRAG candidates for a real filtering example**

Run:

```python
crag_mining_df = mine_demo_case("crag", crag_pipeline, DEMO_MINING_CANDIDATES["crag"])
display(crag_mining_df[["idx", "ok", "trace_shape", "question"]])
```

Success criteria:

```text
The chosen case has at least one partial or irrelevant tag, and the filtered/used evidence changes the generated context in a way the markdown can explain.
```

If no dataset case has useful filtering, add one natural course-material case to `EXTRA_DEMO_CASES`:

```python
"crag_course_natural": "请根据南瓜书相关内容解释宏平均和微平均的区别，以及它们在类别不平衡时各自可能带来的问题。"
```

This is allowed because it is a natural course question, not a prompt designed to mention forbidden topics. Use it only for CRAG inspect, not for `compare_df`.

- [ ] **Step 7: Mine Self-RAG candidates for a useful reflection example**

Run:

```python
selfrag_mining_df = mine_demo_case("self_rag", self_rag_pipeline, DEMO_MINING_CANDIDATES["self_rag"])
display(selfrag_mining_df[["idx", "ok", "trace_shape", "question"]])
```

Success criteria:

```text
The chosen case must show a reflection verdict and a useful answer difference from baseline or a clear self-check behavior. CONTINUE is ideal, but FINISH is acceptable only if the answer is visibly complete and the explanation says Self-RAG's reflection can also stop early when evidence is sufficient.
```

Do not present FINISH as “method did nothing”. The markdown must explain what was checked and why the method stopped.

- [ ] **Step 8: Update `DEMO_CASES` with mined winners**

After mining, update `DEMO_CASES` so the first item for each method is the selected effective example:

```python
DEMO_CASES = {
    "baseline": [0],
    "iterative": [ITERATIVE_WINNER_IDX, 18, 19, 42],
    "recursive": [RECURSIVE_WINNER_IDX, 4, 42, 63],
    "routing_general": [ROUTING_GENERAL_WINNER_IDX],
    "routing_math": [ROUTING_MATH_WINNER_IDX],
    "adaptive_simple": [ADAPTIVE_SIMPLE_WINNER_IDX],
    "adaptive_moderate": [ADAPTIVE_MODERATE_WINNER_IDX],
    "adaptive_complex": [ADAPTIVE_COMPLEX_WINNER_IDX],
    "crag": [CRAG_WINNER_IDX, 7, 62],
    "self_rag": [SELFRAG_WINNER_IDX, 1, 62, 63],
}
```

Replace each `*_WINNER_IDX` with actual integer indices from the mining results. Do not leave symbolic placeholders in the notebook.

- [ ] **Step 9: Add method-specific explanation markdown for each selected demo**

For each method section, add 2-3 sentences after the inspect output:

```markdown
这个代表题有效，不是因为它“难”，而是因为它触发了该方法的核心决策点：<具体决策点>。和 Baseline 相比，读者应重点观察 <trace 中的字段/分支/过滤证据>，这就是本方法解决问题的地方。
```

Fill in the exact decision point:
- Iterative: `missing_hint` drives second retrieve.
- Recursive: `decompose` creates distinct subquestions.
- Routing: `route` selects branch and branch has different `n_hits`/`first_hit`.
- Adaptive: `classify` delegates to different pipelines for different question complexity.
- CRAG: `grade` filters partial/irrelevant evidence before generation.
- Self-RAG: `reflect` decides whether answer is sufficient or needs another round.

- [ ] **Step 10: Commit mined demo cases**

```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb"
git commit -m "docs(c7): mine effective natural flow demos"
```

Expected: commit succeeds.

---

## Task 9: Validate Demo Behavior and Adjust Final Demo Case Choices

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`

This task validates the mined examples from the previous task. Do not weaken the success criteria to “just explain the limitation”; each method section needs a useful example that shows what problem the method solves while preserving original method behavior.

- [ ] **Step 1: Run only demo inspect cells first**

Run these sections without running the full 12-question evaluation:

```text
Baseline inspect
Iterative inspect
Recursive inspect
Routing inspect
Adaptive inspect
CRAG inspect
Self-RAG inspect
```

Expected: each section prints a trace.

- [ ] **Step 2: Check Iterative demo**

Success criteria:

```text
Trace has at least two retrieve steps OR finalize reason is max_rounds_reached.
```

If the first iterative demo question finishes in one round, edit:

```python
"iterative": [18, 19, 42],
```

Move the best-performing candidate to the front. Try in this order:

```python
"iterative": [19, 42, 18],
```

If all three still finish in one round, do not stop with a limitation note. Return to Task 8's mining pool and test additional natural candidates. The final Iterative section must include either a dataset question or a natural course-material question that demonstrates second retrieval or `max_rounds_reached` without changing `ITERATIVE_DRAFT_PROMPT` or `iterative_pipeline`.

- [ ] **Step 3: Check Recursive demo**

Success criteria:

```text
Trace includes decompose and at least two sub_answer entries.
```

If `recursive` over-splits or repeats, move index `4` to the front:

```python
"recursive": [4, 42, 63],
```

Expected: index 4 naturally decomposes into model evaluation, empirical error, generalization error, and which error to minimize.

- [ ] **Step 4: Check Routing demo**

Success criteria:

```text
General case routes to general_index; math case routes to math_index; n_hits differs between branches.
```

If routing chooses the wrong branch, keep the output and add the boundary-risk sentence from Task 5. Do not hard-code the branch.

- [ ] **Step 5: Check Adaptive demo**

Success criteria:

```text
The three adaptive demo calls show at least two different delegate targets; ideal is single_pass, iterative_pipeline, recursive_pipeline.
```

If all three choose the same target, try this ordering:

```python
"adaptive_simple": [14],
"adaptive_moderate": [5],
"adaptive_complex": [63],
```

If still not separated, return to Task 8's adaptive mining candidates. The final Adaptive section must show at least two distinct delegate targets, and ideally all three of `single_pass`, `iterative_pipeline`, and `recursive_pipeline`, without changing `CLASSIFIER_PROMPT` or `adaptive_pipeline`.

- [ ] **Step 6: Check CRAG demo**

Success criteria:

```text
At least one grade tag is partial or irrelevant.
```

If dataset case `7` has no partial/irrelevant, keep the control call:

```python
inspect_crag(extra_demo_question("crag_noise"))
```

Expected: the control question should make ROC/threshold/AUC snippets easier to identify as noise.

- [ ] **Step 7: Check Self-RAG demo**

Success criteria:

```text
Trace includes reflect. CONTINUE is preferred but not required.
```

If no candidate triggers `CONTINUE`, do not force it by modifying `SELF_RAG_PROMPT` or `self_rag_pipeline`. Select a case where Self-RAG's reflection produces a visibly complete answer or a trace that explains why it stopped, and write the section as an effective early-stop example rather than “method did nothing”. If a useful early-stop example cannot be found, return to Task 8's self-RAG mining pool.

- [ ] **Step 8: Commit final demo adjustments**

```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb"
git commit -m "docs(c7): validate flow enhancement demo traces"
```

Expected: commit succeeds.

---

## Task 10: Run Full Evaluation and Save Optional Artifacts

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`
- Optional create/modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/data/flow_enhance_compare_table.csv`
- Optional create/modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/data/flow_enhance_eval_summary.json`

- [ ] **Step 1: Run the full evaluation cells**

Run the cells that compute:

```python
baseline_df
iterative_df
recursive_df
routing_df
adaptive_df
crag_df
selfrag_df
compare_df
summary_df
```

Expected: 12 rows in `compare_df`; all method columns present.

- [ ] **Step 2: Inspect final compare table**

Check:

```python
compare_df.shape
compare_df.columns.tolist()
```

Expected:

```python
(12, 8)
['question', 'baseline', 'iterative', 'recursive', 'routing', 'adaptive', 'crag', 'self_rag']
```

If shape differs because duplicate questions collapsed, verify `qna_dict` contains 12 unique question strings. If not, replace the duplicate index in `QA_INDICES` with candidate `62` or `63`.

- [ ] **Step 3: Optional save artifacts if 6.1-style persistence is desired**

If the notebook already has a data-saving pattern and the user wants parity with 6.1, add this code near the final summary:

```python
from pathlib import Path
import json

out_dir = Path("data")
out_dir.mkdir(parents=True, exist_ok=True)
flow_compare_path = out_dir / "flow_enhance_compare_table.csv"
flow_summary_path = out_dir / "flow_enhance_eval_summary.json"

compare_df.to_csv(flow_compare_path, index=False, encoding="utf-8-sig")

flow_summary = {
    "n_questions": int(len(compare_df)),
    "qa_indices": QA_INDICES,
    "demo_cases": DEMO_CASES,
    "extra_demo_cases": EXTRA_DEMO_CASES,
    "score_summary": flow_score_summary(compare_df).to_dict("records"),
    "files": {
        "compare_table_csv": str(flow_compare_path),
        "eval_summary_json": str(flow_summary_path),
    },
}
with open(flow_summary_path, "w", encoding="utf-8") as f:
    json.dump(flow_summary, f, ensure_ascii=False, indent=2)

print("已写入:", flow_compare_path)
print("已写入:", flow_summary_path)
```

Expected: files are written under the 6.增强阶段 `data/` directory.

- [ ] **Step 4: Commit final evaluated notebook and optional artifacts**

If artifacts are saved and intended to be tracked:

```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb" \
        "notebook/C7 高级 RAG 技巧/6. 增强阶段/data/flow_enhance_compare_table.csv" \
        "notebook/C7 高级 RAG 技巧/6. 增强阶段/data/flow_enhance_eval_summary.json"
git commit -m "data(c7): add flow enhancement evaluation artifacts"
```

If artifacts are not saved or not intended to be tracked:

```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb"
git commit -m "docs(c7): run flow enhancement evaluation summary"
```

Expected: commit succeeds.

---

## Task 11: Final Review Checklist

**Files:**
- Inspect: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`
- Inspect: `notebook/C7 高级 RAG 技巧/6. 增强阶段/4. 选型总结.md`

- [ ] **Step 1: Confirm 6.2 is aligned with 6.1 style**

Check the notebook has all of these:

```text
- 主评测题集说明
- Baseline 对照
- 每个方法的 trace 展示
- compare_df 总表
- flow_score_summary 汇总
- 分方法真实差异面板
- 最终动态解读，不依赖过期手写数字
```

Expected: all items present.

- [ ] **Step 2: Confirm no stale static numbers remain**

Search the notebook text for old hard-coded claims like:

```text
iterative（最佳）
1.45/2
18~25 分钟
routing 和 baseline 分数完全一致
```

Expected: remove or rewrite any stale claims unless they are clearly labeled as an old reference run.

- [ ] **Step 3: Confirm method framing is positive**

Read all headings and summaries. Ensure the wording presents methods as improvements for different failure modes, not as “baseline failed badly”. Use wording like:

```text
Baseline 建立对照
流程增强用于观察多步决策是否带来收益
边界样例用于理解方法适用条件
```

Avoid wording like:

```text
Baseline 太差
某方法一定最好
高级方法必然更强
```

- [ ] **Step 4: Check 4. 选型总结.md only if 6.2 method names changed**

If the notebook still uses the same six method names, do not edit `4. 选型总结.md`.

If a method name changed, update only the relevant row in `4. 选型总结.md` and commit:

```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/4. 选型总结.md"
git commit -m "docs(c7): align flow method naming in selection summary"
```

Expected: no unnecessary summary-doc churn.

- [ ] **Step 5: Final git status**

```bash
git status --short
```

Expected: only intentional notebook/artifact changes remain. Do not commit unrelated existing user changes such as `.claude/`, `.vscode/`, `.worktrees/`, or other notebooks unless explicitly requested.

---

## Self-Review

**Spec coverage:** This plan covers question audit, main QA set replacement, method-specific demo cases, baseline addition, real routing branches, trace validation, dynamic summaries, optional artifacts, and final review.

**Placeholder scan:** No `TBD`, `TODO`, “implement later”, or unspecified test steps remain. Each code change includes exact code blocks.

**Type consistency:** All new pipelines use the existing 6.2 convention `pipeline(question: str) -> tuple[str, list[dict]]`. Comparison helpers assume `baseline_df` is created before method panels are run. `DEMO_CASES` values are lists of dataset indices; `EXTRA_DEMO_CASES` values are literal question strings and are intentionally excluded from `qna_dict`.
