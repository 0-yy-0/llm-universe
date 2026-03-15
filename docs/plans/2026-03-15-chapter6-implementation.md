# Chapter 6 Enhancement Rewrite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild Chapter 6 of the 2026 RAG tutorial into a consistent, beginner-friendly enhancement chapter with runnable failure demos, explicit method internals, and updated 2026 topics including Corrective RAG and Agentic RAG.

**Architecture:** Keep the existing three-layer chapter structure (`上下文增强 / 流程增强 / 系统增强`) and preserve LangChain + LlamaIndex mixed usage where it helps teaching. Each notebook is rewritten to follow the same teaching template: problem, failure case, idea, flow, core code, runnable example, comparison, and scope/limits. API-dependent result demos are kept, but principle cells and core logic cells should run with minimal dependencies.

**Tech Stack:** Jupyter notebooks, Markdown, Python, LangChain, LlamaIndex, OpenAI `gpt-4o-mini`, Chroma/vector retrieval utilities already used in repo.

---

### Task 1: Preflight Decisions And Compatibility Check

**Files:**
- Modify: `docs/plans/2026-03-15-chapter6-implementation.md`
- Check: `notebook/C7 高级 RAG 技巧/6. 增强阶段/1. 上下文增强（重构版）.ipynb`
- Check: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强（重构版）.ipynb`
- Check: `notebook/C7 高级 RAG 技巧/6. 增强阶段/3. 系统增强.ipynb`
- Reference: `docs/plans/2026-03-15-chapter6-design.md`

**Step 1: Check current LlamaIndex pack/API compatibility**

Run:
```bash
python - <<'PY'
mods = [
    "llama_index",
    "llama_index.core.llama_pack",
    "llama_index.agent.openai",
]
for mod in mods:
    try:
        __import__(mod)
        print("OK", mod)
    except Exception as e:
        print("ERR", mod, type(e).__name__, e)
PY
```

Expected: module availability summary; record whether `download_llama_pack` and agent imports still work.

**Step 2: Decide fallback implementations if imports fail**

Write a short note in your working scratchpad:
- If `download_llama_pack` works: keep pack demos and add manual “core logic” cells before them.
- If it fails: replace each pack demo with equivalent hand-written LlamaIndex logic and keep wording unchanged for readers.
- If `OpenAIAgent` import fails: implement Agentic RAG with LangGraph or a simple planner-executor loop.

**Step 3: Confirm common model string**

Search all target notebooks for outdated model names.

Run:
```bash
python - <<'PY'
from pathlib import Path
paths = [
    Path("notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强（重构版）.ipynb"),
    Path("notebook/C7 高级 RAG 技巧/6. 增强阶段/3. 系统增强.ipynb"),
]
for path in paths:
    text = path.read_text()
    for token in ["gpt-3.5-turbo", "gpt-4o-mini", "selfrag_llama2_7b", "download_llama_pack"]:
        if token in text:
            print(path, "contains", token)
PY
```

Expected: a list of model/API strings to normalize during implementation.

**Step 4: Commit compatibility decision notes**

Run:
```bash
git status --short
```

Expected: no notebook edits yet; only plan/doc changes if any.

#### Task 1 Execution Notes (2026-03-15)

- `llama_index` is not installed in current environment (`ModuleNotFoundError`).
- `download_llama_pack` and `OpenAIAgent` pack-based demos are therefore not runnable as-is.
- Fallback decision for this execution:
  - Use simple, explicit LangChain-first runnable cells for all required demos.
  - Keep method explanations unchanged in teaching narrative.
  - Where LlamaIndex pack API was originally intended, provide equivalent manual logic or LangChain alternatives.
  - Normalize model references to `gpt-4o-mini` during subsequent notebook rewrites.

---

### Task 2: Rewrite The Intro Notebook Skeleton

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/0. 先导：为什么基础 RAG 还不够.ipynb`
- Reference: `notebook/C7 高级 RAG 技巧/6. 增强阶段/data/face.pdf`
- Reference: `notebook/C7 高级 RAG 技巧/6. 增强阶段/data/mutli_documents_data/`

**Step 1: Reorder the notebook into “failure-first” flow**

Edit the notebook cells to this order:
1. title + chapter goal
2. baseline assumptions of basic RAG
3. shared setup cell
4. failure case 1 markdown
5. failure case 1 code
6. failure case 1 analysis
7. failure case 2 markdown
8. failure case 2 code
9. failure case 2 analysis
10. failure case 3 markdown
11. failure case 3 code
12. failure case 3 analysis
13. method map
14. learning path
15. boundaries

**Step 2: Write one shared setup code cell**

Include:
- imports
- environment loading pattern consistent with the rest of the tutorial
- data loading for `face.pdf`
- optional helper for showing retrieved chunks

**Step 3: Add failure case 1**

Implement a minimal baseline retrieval with very small chunks on `face.pdf`.

Include:
- chunk size around `128`
- one question whose answer spans multiple sentences/paragraphs
- retrieval output + final answer

Expected teaching point: retrieval is “related” but context is incomplete.

**Step 4: Add failure case 2**

Implement a single-pass retrieval/generation example where one retrieval round is not enough.

Include:
- one question needing decomposition or follow-up retrieval
- one answer showing partial coverage

Expected teaching point: this belongs to flow enhancement, not context enhancement.

**Step 5: Add failure case 3**

Implement a multi-document or multi-turn failure example using the city data.

Include one of:
- mixed-city baseline confusion
- or a second-turn pronoun follow-up without memory

Expected teaching point: this belongs to system enhancement.

**Step 6: Sanity-check notebook structure**

Run:
```bash
python - <<'PY'
import json
path = "notebook/C7 高级 RAG 技巧/6. 增强阶段/0. 先导：为什么基础 RAG 还不够.ipynb"
nb = json.load(open(path))
print("cells", len(nb["cells"]))
for i, cell in enumerate(nb["cells"][:8]):
    src = "".join(cell.get("source", []))
    print(i, cell["cell_type"], src[:50].replace("\n", " "))
PY
```

Expected: cell count increased and the first cells match the failure-first structure.

**Step 7: Commit the intro rewrite**

Run:
```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/0. 先导：为什么基础 RAG 还不够.ipynb"
git commit -m "docs: 重构第六章先导失败案例"
```

Expected: one commit focused on the intro notebook only.

---

### Task 3: Build The Context Enhancement Baseline And Template

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/1. 上下文增强（重构版）.ipynb`
- Reference: `notebook/C7 高级 RAG 技巧/6. 增强阶段/1. 上下文增强.ipynb`

**Step 1: Expand the notebook to include a baseline section**

Add cells for:
- section goal
- unified experiment setup
- plain vector retrieval baseline
- explicit failure example before any enhancement method

**Step 2: Normalize setup code**

Use one shared setup cell with:
- `gpt-4o-mini`
- `face.pdf`
- `qna_dict`
- evaluation helper
- any common splitter/retriever helpers

**Step 3: Add baseline comparison output**

Implement a plain retrieval answer loop over `qna_dict` and store results in a dataframe.

Expected: one dataframe used later alongside Sentence Window / Small-to-Big / AutoMerging outputs.

**Step 4: Add a “how to read this section” markdown cell**

Write a short note telling readers that every method follows the same 8-step template.

**Step 5: Sanity-check notebook cell count and headings**

Run:
```bash
python - <<'PY'
import json
path = "notebook/C7 高级 RAG 技巧/6. 增强阶段/1. 上下文增强（重构版）.ipynb"
nb = json.load(open(path))
print("cells", len(nb["cells"]))
for i, cell in enumerate(nb["cells"][:12]):
    src = "".join(cell.get("source", []))
    print(i, cell["cell_type"], src[:60].replace("\n", " "))
PY
```

Expected: notebook is much larger than the original 10-cell version and includes baseline/template framing.

---

### Task 4: Add Sentence Window Teaching Block

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/1. 上下文增强（重构版）.ipynb`

**Step 1: Write the failure-case markdown for Sentence Window**

Explain:
- hit sentence is relevant
- neighboring sentences carry the missing support

**Step 2: Add “index time vs retrieval time” explanation**

Write a markdown block or simple mermaid diagram showing:
- split into sentences + store neighbors
- retrieve hit sentence
- restore left/right window before generation

**Step 3: Add a core-logic code cell**

Show only the key logic:
- sentence split
- neighbor mapping
- window restoration

Keep this code short and readable.

**Step 4: Keep or replace the pack demo**

- If pack works: keep `SentenceWindowRetrieverPack`
- If pack fails: write equivalent manual LlamaIndex code

**Step 5: Add result comparison cells**

Display:
- method dataframe
- one concrete Q/A example against the baseline

**Step 6: Add applicability/limitations markdown**

Must include:
- best for local context dependence
- not a fix for missing evidence or multi-step reasoning

---

### Task 5: Add Small-to-Big And AutoMerging Teaching Blocks

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/1. 上下文增强（重构版）.ipynb`

**Step 1: Write the Small-to-Big block**

Add:
- failure case markdown
- idea markdown
- index/retrieval flow
- core logic cell for child-to-parent mapping
- runnable example
- result display
- applicability/limits

**Step 2: Write the AutoMerging block**

Add:
- failure case markdown
- idea markdown
- hierarchy merge flow
- core logic cell for leaf retrieval + merge threshold
- runnable example
- result display
- applicability/limits

**Step 3: Add a final comparison table**

Create a summary dataframe or markdown table comparing:
- baseline
- Sentence Window
- Small-to-Big
- AutoMerging

Compare at least:
- typical failure fixed
- added complexity
- best-fit document shape

**Step 4: Add section closing cells**

Add:
- “how to choose”
- learning checkpoint

**Step 5: Run notebook JSON sanity check**

Run:
```bash
python - <<'PY'
import json
path = "notebook/C7 高级 RAG 技巧/6. 增强阶段/1. 上下文增强（重构版）.ipynb"
nb = json.load(open(path))
needles = ["Sentence Window", "Small-to-Big", "AutoMerging", "学习检查点"]
text = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
for needle in needles:
    print(needle, needle in text)
PY
```

Expected: all four checks print `True`.

**Step 6: Commit the context notebook**

Run:
```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/1. 上下文增强（重构版）.ipynb"
git commit -m "docs: 重构第六章上下文增强"
```

Expected: one commit scoped to the context-enhancement notebook.

---

### Task 6: Rebuild The Flow Enhancement Skeleton

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强（重构版）.ipynb`
- Reference: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`
- Reference: `notebook/C7 高级 RAG 技巧/6. 增强阶段/figures/selfrag.png`
- Reference: `notebook/C7 高级 RAG 技巧/6. 增强阶段/figures/selftoken.jpg`

**Step 1: Expand the notebook structure**

Reorder/add cells so the notebook contains:
- section goal
- boundary vs Chapter 4
- shared setup
- single-pass baseline failure
- five method blocks
- summary
- handoff to system enhancement
- learning checkpoint

**Step 2: Add a shared setup code cell**

Include:
- `gpt-4o-mini`
- retriever helper
- any dataset preparation needed for iterative/recursive/routing/CRAG/Self-RAG demos

**Step 3: Add the single-pass baseline failure example**

Use one query that shows why one retrieval round is insufficient.

**Step 4: Replace pseudo-code-only framing**

Remove the current notebook’s dependence on free-floating toy functions without a real retriever/llm connection.

**Step 5: Sanity-check headings before method implementation**

Run:
```bash
python - <<'PY'
import json
path = "notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强（重构版）.ipynb"
nb = json.load(open(path))
print("cells", len(nb["cells"]))
for i, cell in enumerate(nb["cells"][:10]):
    src = "".join(cell.get("source", []))
    print(i, cell["cell_type"], src[:60].replace("\n", " "))
PY
```

Expected: baseline framing is in place before detailed method blocks are added.

---

### Task 7: Implement Iterative Retrieval, Recursive Retrieval, And Query Routing

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强（重构版）.ipynb`

**Step 1: Implement Iterative Retrieval block**

Add:
- failure case
- loop flow diagram
- real runnable code wired to retriever + llm
- before/after answer comparison
- scope/limits

**Step 2: Implement Recursive Retrieval block**

Add:
- failure case
- decomposition flow
- sub-question list example
- runnable code that retrieves per sub-question and merges final answer
- scope/limits

**Step 3: Implement Query Routing block**

Add:
- rule-based routing explanation
- LLM-based routing explanation
- runnable LLM router selecting an index/tool
- result example
- scope/limits

**Step 4: Validate block presence**

Run:
```bash
python - <<'PY'
import json
path = "notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强（重构版）.ipynb"
nb = json.load(open(path))
text = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
for needle in ["迭代检索", "递归检索", "查询路由", "gpt-4o-mini"]:
    print(needle, needle in text)
PY
```

Expected: all checks print `True`.

---

### Task 8: Implement Corrective RAG And Simplified Self-RAG

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强（重构版）.ipynb`

**Step 1: Implement the Corrective RAG block**

Add:
- retrieval-quality failure case
- CRAG flow diagram
- grader logic for retrieved docs
- branch behavior:
  - all relevant
  - partially relevant
  - irrelevant
- runnable example
- direct comparison against plain retrieval

**Step 2: Rebuild Self-RAG as a teaching block, not a legacy model demo**

Add:
- self-rag motivation
- three-stage explanation: retrieve / generate / critique
- both original figures
- training vs inference explanation
- simplified control loop using an LLM judge for “retrieve more or stop”

Do not include:
- local LLaMA 2 downloads
- GGUF model instructions
- legacy pack-specific heavyweight setup

**Step 3: Add section summary and learning checkpoint**

Write:
- when to use each of the five flow-enhancement methods
- one short self-check list for readers

**Step 4: Run notebook sanity check**

Run:
```bash
python - <<'PY'
import json
path = "notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强（重构版）.ipynb"
nb = json.load(open(path))
text = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
for needle in ["Corrective RAG", "Self-RAG", "selfrag.png", "selftoken.jpg", "学习检查点"]:
    print(needle, needle in text)
PY
```

Expected: all checks print `True`.

**Step 5: Commit the flow notebook**

Run:
```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强（重构版）.ipynb"
git commit -m "docs: 重构第六章流程增强"
```

Expected: one commit scoped to the flow-enhancement notebook.

---

### Task 9: Rebuild System Enhancement With Memory And Multi-Document Agent

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/3. 系统增强.ipynb`
- Reference: `notebook/C7 高级 RAG 技巧/6. 增强阶段/figures/mda.png`

**Step 1: Add section framing and shared setup**

Introduce:
- why this is a “system” problem, not just a retrieval-flow problem
- one shared setup cell with normalized model configuration

**Step 2: Rewrite the Memory block**

Add:
- no-memory failure case
- memory concept + where it helps
- end-to-end multi-turn demo
- explicit write-back step
- with-memory vs without-memory comparison
- scope/limits

**Step 3: Rewrite the Multi-Document Agent block**

Keep the existing logic, but reorder it to teach:
- why single-index baselines fail
- per-document agents/tools
- top-level routing
- baseline comparison

Also normalize model usage to `gpt-4o-mini`.

**Step 4: Add system-cost/boundary markdown**

Explain:
- latency
- observability
- permissions
- fallback strategy expectations

**Step 5: Validate notebook block presence**

Run:
```bash
python - <<'PY'
import json
path = "notebook/C7 高级 RAG 技巧/6. 增强阶段/3. 系统增强.ipynb"
nb = json.load(open(path))
text = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
for needle in ["Memory", "Multi-Document Agent", "gpt-4o-mini"]:
    print(needle, needle in text)
PY
```

Expected: all checks print `True`.

---

### Task 10: Add Agentic RAG, Update Summary Files, And Run Final Verification

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/3. 系统增强.ipynb`
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/4. 选型总结.md`
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/readme.md`

**Step 1: Add the Agentic RAG block**

Teach it as:
- planner/executor/reflection loop
- how it composes retrieval, routing, correction, and memory
- one runnable mini demo using the simplest reliable framework from Task 1
- baseline comparison against a non-agent workflow
- applicability/limits

**Step 2: Upgrade `4. 选型总结.md`**

Add:
- Corrective RAG row
- Agentic RAG row
- “主要改善指标” column
- three recommended method combinations

**Step 3: Upgrade `readme.md`**

Reflect the final chapter structure and newly added methods.

**Step 4: Run repository-level structural verification**

Run:
```bash
python - <<'PY'
from pathlib import Path
targets = [
    Path("notebook/C7 高级 RAG 技巧/6. 增强阶段/0. 先导：为什么基础 RAG 还不够.ipynb"),
    Path("notebook/C7 高级 RAG 技巧/6. 增强阶段/1. 上下文增强（重构版）.ipynb"),
    Path("notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强（重构版）.ipynb"),
    Path("notebook/C7 高级 RAG 技巧/6. 增强阶段/3. 系统增强.ipynb"),
    Path("notebook/C7 高级 RAG 技巧/6. 增强阶段/4. 选型总结.md"),
    Path("notebook/C7 高级 RAG 技巧/6. 增强阶段/readme.md"),
]
for path in targets:
    print(path, path.exists(), path.stat().st_size)
PY
```

Expected: all files exist and have non-trivial size.

**Step 5: Run targeted lint/JSON validation**

Run:
```bash
python - <<'PY'
import json
paths = [
    "notebook/C7 高级 RAG 技巧/6. 增强阶段/0. 先导：为什么基础 RAG 还不够.ipynb",
    "notebook/C7 高级 RAG 技巧/6. 增强阶段/1. 上下文增强（重构版）.ipynb",
    "notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强（重构版）.ipynb",
    "notebook/C7 高级 RAG 技巧/6. 增强阶段/3. 系统增强.ipynb",
]
for path in paths:
    with open(path) as f:
        json.load(f)
    print("valid json", path)
PY
```

Expected: all notebooks parse with no JSON error.

**Step 6: Run smoke execution only for non-API-dependent support code if practical**

If you extracted any helper functions into standalone Python snippets, run those snippets directly and verify there is no traceback. Do not claim notebook execution success without actually running the specific cells or commands.

**Step 7: Commit the final batch**

Run:
```bash
git add \
  "notebook/C7 高级 RAG 技巧/6. 增强阶段/3. 系统增强.ipynb" \
  "notebook/C7 高级 RAG 技巧/6. 增强阶段/4. 选型总结.md" \
  "notebook/C7 高级 RAG 技巧/6. 增强阶段/readme.md"
git commit -m "docs: 完成第六章增强阶段重构"
```

Expected: final commit for system notebook + summary docs.

---

### Notes For The Executor

- Treat the existing design doc as the source of truth: `docs/plans/2026-03-15-chapter6-design.md`
- Preserve any unrelated user changes outside the files above.
- Prefer `EditNotebook` for notebook edits.
- After each major notebook rewrite, run JSON sanity validation before moving on.
- If compatibility checks fail, prefer simple explicit code over clever abstractions.
- Do not silently keep outdated 2023-2024 model references if a 2026-safe equivalent is available.

### Suggested New-Session Prompt

Use this in the new session:

```text
请读取 docs/plans/2026-03-15-chapter6-implementation.md，并使用 superpowers:executing-plans 按任务逐步执行。
从 Task 1 开始，不要跳步；每完成一个 Task 做一次最小验证，再继续下一个。
如遇到 LlamaIndex API 不兼容，按计划中的 fallback 执行。
```
