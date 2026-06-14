# C7 6.2 Routing Problem-Solving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Query Routing section in C7 6.2 from a parameter-tuning demo into a problem-solving demo where routing chooses the right evidence strategy for formula-heavy, concept-heavy, and ambiguous questions.

**Architecture:** Keep all changes scoped to the existing 6.2 notebook and its local evaluation helper script. Add a small structured router parser, a `mixed_index` fallback, and a deterministic local evidence selector so routing can improve evidence quality instead of only increasing `k`. Use same-run `METHOD_RECORDS` for the targeted panel, and keep the final teaching claim bounded to trace-backed improvements rather than SOTA-style average-score claims.

**Tech Stack:** Jupyter notebook JSON (`nbformat`), Python 3.10, LangChain retrievers, existing `_common.py` helpers, pandas CSV evaluation outputs, Zhipu LLM API via existing `llm_call`.

---

## File Structure

- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`
  - Owns the teaching implementation for all six flow-enhancement methods.
  - Routing cells around lines 838-930 define `ROUTER_PROMPT`, retrievers, and `routing_pipeline`.
  - Demo case cells around lines 449-454 define `routing_general` and `routing_math` examples.
  - Same-run panel cells around lines 1480-1594 validate routing trace shape.
  - Summary markdown around lines 1602+ explains how to read routing results.
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/_test_routing_fix.py`
  - Owns standalone routing evaluation experiments.
  - Should compare current routing with mixed+filter routing on the same candidate set.
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/4. 选型总结.md`
  - Owns short method-selection guidance.
  - Should state routing is useful when query type changes evidence strategy, not when one generic retriever already covers all cases.
- Optional generated outputs: `notebook/C7 高级 RAG 技巧/6. 增强阶段/_eval_routing_mixed.csv`
  - Stores one reproducible experiment result if the worker runs the full LLM evaluation.
  - Do not require this file for notebook structural validation.

---

### Task 1: Audit Routing Questions

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb:449`
- Reference: `notebook/C7 高级 RAG 技巧/6. 增强阶段/_eval_compare_df.csv`

- [ ] **Step 1: Confirm the current routing demo cases**

Run:

```bash
rg -n 'routing_general|routing_math' 'notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb' | head -20
```

Expected: output includes these exact case lines:

```text
"routing_general": [62, 84],
"routing_math": [12, 19],
```

- [ ] **Step 2: Print current routing wins and regressions**

Run:

```bash
cd 'notebook/C7 高级 RAG 技巧/6. 增强阶段'
/usr/local/Caskroom/miniconda/base/envs/py310/bin/python - <<'PY'
import pandas as pd
from pathlib import Path
p = Path('_eval_compare_df.csv')
if not p.exists():
    raise SystemExit('_eval_compare_df.csv not found; run the notebook evaluation cell first')
df = pd.read_csv(p)
for _, row in df.iterrows():
    delta = int(row['routing']) - int(row['baseline'])
    marker = 'WIN' if delta > 0 else 'REGRESSION' if delta < 0 else 'TIE'
    print(f'{marker:10s} delta={delta:+d} baseline={int(row["baseline"])} routing={int(row["routing"])} question={row["question"][:120]}')
PY
```

Expected: output shows routing has both wins and regressions, proving the next change should target problem-fit rather than claim universal average improvement.

- [ ] **Step 3: Add a markdown note explaining case intent**

In `2. 流程增强.ipynb`, edit the markdown cell immediately before or after `DEMO_CASES` to include this exact paragraph:

```markdown
Routing 的展示题不是为了“刷平均分”，而是为了暴露它解决的具体失败模式：同一批问题里有些需要大块上下文解释概念，有些需要小块高召回保留公式邻近文本。若题目本身用一个通用检索器已经足够，Routing 可能只增加流程成本；因此 targeted cases 必须能说明 router 为什么改变证据策略。
```

If no nearby markdown cell exists, create one immediately before the code cell that defines `DEMO_CASES`.

- [ ] **Step 4: Validate notebook JSON after the markdown edit**

Run:

```bash
python - <<'PY'
import nbformat
from pathlib import Path
p = Path('notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb')
nb = nbformat.read(p, as_version=4)
nbformat.validate(nb)
print(f'valid notebook: {len(nb.cells)} cells')
PY
```

Expected: prints `valid notebook:` and exits with status 0.

- [ ] **Step 5: Commit the question-audit note**

Run:

```bash
git add 'notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb'
git commit -m 'docs: explain routing demo case intent'
```

Expected: commit succeeds. If this repo is being edited without commits, record the exact changed file in the handoff instead of committing.

---

### Task 2: Add Structured Router Decision

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb:882`

- [ ] **Step 1: Replace `ROUTER_PROMPT` with a JSON-output prompt**

In the routing code cell, replace the existing `ROUTER_PROMPT` with:

```python
ROUTER_PROMPT = """
你是 RAG 查询路由器。请根据问题选择最合适的检索策略。

可选策略：
- math_index：题目要求推导、写公式、解释数学表达式、KKT/梯度/Hessian/核函数/EM/MM 等数学步骤。
- general_index：题目要求定义、概念区别、优缺点、应用背景，不需要展开公式推导。
- mixed_index：题目同时包含概念解释和公式推导，或你不确定单一索引是否足够。

只输出 JSON，不要 Markdown，不要解释：
{{"target": "math_index|general_index|mixed_index", "confidence": 0.0到1.0之间的小数, "reason": "不超过20个字"}}

问题：{question}
""".strip()
```

- [ ] **Step 2: Add `parse_router_decision` below `ROUTER_PROMPT`**

Add this function immediately below `ROUTER_PROMPT`:

```python
def parse_router_decision(raw: str) -> dict:
    """Parse router JSON and fall back to mixed_index on malformed or low-confidence output."""
    text = str(raw).strip()
    try:
        if "```" in text:
            text = text.split("```")[1]
            text = text.removeprefix("json").strip()
        data = json.loads(text)
    except Exception:
        lowered = text.lower()
        if "math" in lowered and "general" not in lowered:
            data = {"target": "math_index", "confidence": 0.7, "reason": "legacy math"}
        elif "general" in lowered and "math" not in lowered:
            data = {"target": "general_index", "confidence": 0.7, "reason": "legacy general"}
        else:
            data = {"target": "mixed_index", "confidence": 0.0, "reason": "parse failed"}

    target = str(data.get("target", "mixed_index")).strip().lower()
    if target not in {"math_index", "general_index", "mixed_index"}:
        target = "mixed_index"
    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    if confidence < 0.55:
        target = "mixed_index"
    return {
        "target": target,
        "confidence": confidence,
        "reason": str(data.get("reason", ""))[:40],
    }
```

- [ ] **Step 3: Add lightweight parser self-checks in the same cell**

Add this block after `parse_router_decision`:

```python
assert parse_router_decision('{"target":"math_index","confidence":0.9,"reason":"公式推导"}')["target"] == "math_index"
assert parse_router_decision('{"target":"general_index","confidence":0.4,"reason":"不确定"}')["target"] == "mixed_index"
assert parse_router_decision('math_index')["target"] == "math_index"
assert parse_router_decision('nonsense')["target"] == "mixed_index"
```

- [ ] **Step 4: Validate the parser in notebook text**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
text = '\n'.join(''.join(c.get('source', [])) for c in nb['cells'])
checks = {
    'json_router_prompt': '只输出 JSON' in text,
    'parse_router_decision': 'def parse_router_decision(raw: str) -> dict' in text,
    'low_confidence_fallback': 'confidence < 0.55' in text,
    'mixed_index_present': 'mixed_index' in text,
}
print(checks)
raise SystemExit(0 if all(checks.values()) else 1)
PY
```

Expected: all checks print `True`.

- [ ] **Step 5: Commit structured router parsing**

Run:

```bash
git add 'notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb'
git commit -m 'feat: add structured routing decisions'
```

Expected: commit succeeds. If commits are disabled for this workstream, document the modified file in the handoff.

---

### Task 3: Implement Mixed Retrieval and Evidence Selection

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb:888-918`

- [ ] **Step 1: Add a deterministic evidence selector**

Add this function above `routing_pipeline`:

```python
def select_routing_evidence(question: str, docs: list, max_docs: int = 4) -> list:
    """Keep evidence that overlaps with the question while preserving retriever order."""
    question_tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", question)
        if token.strip()
    }
    scored = []
    seen = set()
    for rank, doc in enumerate(docs):
        content = doc.page_content
        fingerprint = content[:120]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        doc_tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", content[:1200])
            if token.strip()
        }
        overlap = len(question_tokens & doc_tokens)
        formula_bonus = 2 if any(symbol in content for symbol in ["=", "∑", "∂", "∇", "β", "λ", "argmin"]) else 0
        scored.append((overlap + formula_bonus, -rank, doc))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    selected = [doc for _, _, doc in scored[:max_docs]]
    return selected or docs[:max_docs]
```

- [ ] **Step 2: Add `mixed_retrieve`**

Add this function below `INDEX_TO_RETRIEVER`:

```python
def mixed_retrieve(question: str) -> tuple[list, dict]:
    math_docs = math_retriever.invoke(question)
    general_docs = general_retriever.invoke(question)
    selected = select_routing_evidence(question, math_docs[:4] + general_docs[:3], max_docs=4)
    diagnostics = {
        "math_hits": len(math_docs),
        "general_hits": len(general_docs),
        "selected_hits": len(selected),
    }
    return selected, diagnostics
```

- [ ] **Step 3: Replace `routing_pipeline` with mixed-aware routing**

Replace the entire current `routing_pipeline` function with:

```python
def routing_pipeline(question: str) -> tuple[str, list[dict]]:
    trace: list[dict] = []

    raw_decision = llm_call(ROUTER_PROMPT.format(question=question))
    decision = parse_router_decision(raw_decision)
    branch = decision["target"]
    trace.append(step(
        "route",
        target=branch,
        confidence=decision["confidence"],
        reason=decision["reason"],
        raw=raw_decision[:120],
    ))

    if branch == "mixed_index":
        docs, diagnostics = mixed_retrieve(question)
    else:
        chosen_retriever = INDEX_TO_RETRIEVER[branch]
        docs = chosen_retriever.invoke(question)
        docs = select_routing_evidence(question, docs, max_docs=min(4, len(docs)))
        diagnostics = {"selected_hits": len(docs)}

    ctx = trim_context_to_budget("\n\n".join(d.page_content for d in docs), CONTEXT_CHAR_BUDGET)
    trace.append(step(
        "retrieve",
        branch=branch,
        n_hits=len(docs),
        context_chars=len(ctx),
        first_hit=docs[0].page_content[:80] if docs else "",
        **diagnostics,
    ))

    answer = llm_call(build_rag_generation_prompt(question, ctx))
    trace.append(step("finalize", reason="answered"))
    return answer, trace
```

- [ ] **Step 4: Ensure imports exist**

Find the first imports cell in `2. 流程增强.ipynb`. If it does not already import `re`, add:

```python
import re
```

The parser added in Task 2 uses `json`; if `json` is not already imported, add:

```python
import json
```

- [ ] **Step 5: Run structural validation**

Run:

```bash
python - <<'PY'
import json
import nbformat
from pathlib import Path
p = Path('notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb')
nb = nbformat.read(p, as_version=4)
nbformat.validate(nb)
text = '\n'.join(''.join(c.get('source', [])) for c in nb.cells)
checks = {
    'select_routing_evidence': 'def select_routing_evidence(question: str, docs: list, max_docs: int = 4) -> list' in text,
    'mixed_retrieve': 'def mixed_retrieve(question: str) -> tuple[list, dict]' in text,
    'mixed_branch': 'if branch == "mixed_index"' in text,
    'selected_hits_trace': 'selected_hits' in text,
}
print(checks)
raise SystemExit(0 if all(checks.values()) else 1)
PY
```

Expected: all checks print `True`.

- [ ] **Step 6: Commit mixed routing**

Run:

```bash
git add 'notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb'
git commit -m 'feat: add mixed routing evidence selection'
```

Expected: commit succeeds. If commits are disabled, document the modified file in the handoff.

---

### Task 4: Update Trace Validation for Mixed Routing

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb:314-335`
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb:1547-1559`

- [ ] **Step 1: Allow `routing_math` to pass through `mixed_index` only when confidence is low**

In `validate_demo_trace`, replace the routing branch validation block with this exact logic:

```python
    elif method_name == "routing":
        routes = [s for s in trace if s.get("kind") == "route"]
        retrieves = [s for s in trace if s.get("kind") == "retrieve"]
        branches = [s.get("branch") for s in retrieves if s.get("branch")]
        expected_branch = {
            "routing_general": "general_index",
            "routing_math": "math_index",
        }.get(case_key)
        has_retrieval_shape = bool(retrieves) and all(
            {"n_hits", "context_chars", "first_hit", "selected_hits"}.issubset(s.keys())
            for s in retrieves
        )
        low_confidence_mixed = bool(routes) and routes[0].get("target") == "mixed_index" and routes[0].get("confidence", 1.0) < 0.55
        branch_ok = expected_branch is None or expected_branch in branches or low_confidence_mixed
        ok = "route" in kinds and has_retrieval_shape and branch_ok
        expected = "出现 route 和对应 branch retrieve；routing_general/routing_math 优先走预期分支，低置信度可走 mixed_index；暴露 branch/n_hits/context_chars/first_hit/selected_hits"
```

- [ ] **Step 2: Update the routing panel assertion**

In `display_flow_method_panel`, replace the `if method_name == "routing":` assertion block with:

```python
    if method_name == "routing":
        expected_by_key = {
            "routing_general": "general_index",
            "routing_math": "math_index",
        }
        retrieve_by_key = {r["case_key"]: first_retrieve(r) for r in records}
        route_by_key = {
            r["case_key"]: next((s for s in r["trace"] if s.get("kind") == "route"), {})
            for r in records
        }
        for case_key, expected_branch in expected_by_key.items():
            actual_branch = retrieve_by_key[case_key].get("branch")
            low_confidence_mixed = (
                actual_branch == "mixed_index"
                and route_by_key[case_key].get("confidence", 1.0) < 0.55
            )
            assert actual_branch == expected_branch or low_confidence_mixed, retrieve_by_key
        shapes = {
            (
                s.get("branch"),
                s.get("n_hits"),
                s.get("context_chars"),
                s.get("first_hit"),
                s.get("selected_hits"),
            )
            for s in retrieve_by_key.values()
        }
        assert len(shapes) >= 2, "routing branch traces must differ in branch/n_hits/context_chars/first_hit/selected_hits"
```

- [ ] **Step 3: Update the routing trace-signal markdown row**

Replace the Routing row in the trace signal table with:

```markdown
| Routing | `routing_general` / `routing_math` 优先走对应分支；低置信度允许 `mixed_index`；分支的 `branch` / `n_hits` / `context_chars` / `selected_hits` / `first_hit` 至少有可解释差异 |
```

- [ ] **Step 4: Run validation checks**

Run:

```bash
python - <<'PY'
import nbformat
from pathlib import Path
p = Path('notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb')
nb = nbformat.read(p, as_version=4)
nbformat.validate(nb)
text = '\n'.join(''.join(c.get('source', [])) for c in nb.cells)
checks = {
    'low_confidence_mixed': 'low_confidence_mixed' in text,
    'selected_hits_required': '{"n_hits", "context_chars", "first_hit", "selected_hits"}' in text,
    'routing_assertion_updated': 'routing branch traces must differ in branch/n_hits/context_chars/first_hit/selected_hits' in text,
}
print(checks)
raise SystemExit(0 if all(checks.values()) else 1)
PY
```

Expected: all checks print `True`.

- [ ] **Step 5: Commit trace validation updates**

Run:

```bash
git add 'notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb'
git commit -m 'test: validate mixed routing traces'
```

Expected: commit succeeds. If commits are disabled, document the modified file in the handoff.

---

### Task 5: Extend Standalone Routing Experiment

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/_test_routing_fix.py`
- Optional output: `notebook/C7 高级 RAG 技巧/6. 增强阶段/_eval_routing_mixed.csv`

- [ ] **Step 1: Add the same parser and selector to `_test_routing_fix.py`**

Near the top of `_test_routing_fix.py`, add imports and helper functions:

```python
import json
import re


def parse_router_decision(raw: str) -> dict:
    text = str(raw).strip()
    try:
        if "```" in text:
            text = text.split("```")[1]
            text = text.removeprefix("json").strip()
        data = json.loads(text)
    except Exception:
        lowered = text.lower()
        if "math" in lowered and "general" not in lowered:
            data = {"target": "math_index", "confidence": 0.7, "reason": "legacy math"}
        elif "general" in lowered and "math" not in lowered:
            data = {"target": "general_index", "confidence": 0.7, "reason": "legacy general"}
        else:
            data = {"target": "mixed_index", "confidence": 0.0, "reason": "parse failed"}
    target = str(data.get("target", "mixed_index")).strip().lower()
    if target not in {"math_index", "general_index", "mixed_index"}:
        target = "mixed_index"
    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    if confidence < 0.55:
        target = "mixed_index"
    return {"target": target, "confidence": confidence, "reason": str(data.get("reason", ""))[:40]}


def select_routing_evidence(question: str, docs: list, max_docs: int = 4) -> list:
    question_tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", question)
        if token.strip()
    }
    scored = []
    seen = set()
    for rank, doc in enumerate(docs):
        content = doc.page_content
        fingerprint = content[:120]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        doc_tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", content[:1200])
            if token.strip()
        }
        overlap = len(question_tokens & doc_tokens)
        formula_bonus = 2 if any(symbol in content for symbol in ["=", "∑", "∂", "∇", "β", "λ", "argmin"]) else 0
        scored.append((overlap + formula_bonus, -rank, doc))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    selected = [doc for _, _, doc in scored[:max_docs]]
    return selected or docs[:max_docs]
```

- [ ] **Step 2: Replace `ROUTER_PROMPT` in `_test_routing_fix.py`**

Use the same JSON-output prompt from Task 2 Step 1.

- [ ] **Step 3: Add `routing_pipeline_mixed`**

Add this function below the current `routing_pipeline`:

```python
def routing_pipeline_mixed(question: str) -> tuple[str, list[dict]]:
    trace = []
    raw_decision = llm_call(ROUTER_PROMPT.format(question=question))
    decision = parse_router_decision(raw_decision)
    branch = decision["target"]
    trace.append({"kind": "route", **decision})

    if branch == "mixed_index":
        math_docs = math_retriever.invoke(question)
        general_docs = general_retriever_k2.invoke(question)
        docs = select_routing_evidence(question, math_docs[:4] + general_docs[:3], max_docs=4)
        diagnostics = {"math_hits": len(math_docs), "general_hits": len(general_docs), "selected_hits": len(docs)}
    else:
        chosen = math_retriever if branch == "math_index" else general_retriever_k2
        docs = select_routing_evidence(question, chosen.invoke(question), max_docs=4)
        diagnostics = {"selected_hits": len(docs)}

    ctx = trim_context_to_budget("\n\n".join(d.page_content for d in docs), CONTEXT_CHAR_BUDGET)
    trace.append({"kind": "retrieve", "branch": branch, "n_hits": len(docs), "context_chars": len(ctx), **diagnostics})
    answer = llm_call(build_rag_generation_prompt(question, ctx))
    trace.append({"kind": "finalize", "reason": "answered"})
    return answer, trace
```

- [ ] **Step 4: Add a third experiment block**

After the existing `k=4` block, add:

```python
print("\n>>> 测试 routing with mixed_index + evidence filter...")
df_mixed = eval_pipeline("routing_mixed", routing_pipeline_mixed, qna_dict)
df_mixed.to_csv("_eval_routing_mixed.csv", index=False)
total_mixed = int(df_mixed["rag_eval_results"].sum())
mean_mixed = float(df_mixed["rag_eval_results"].mean())
print(f"    routing_mixed: 总分={total_mixed}/{len(df_mixed)*2}, 均值={mean_mixed:.3f}")
```

- [ ] **Step 5: Update the final summary print**

Replace the final summary line with:

```python
print(f"\n汇总: k=2 均值={mean_k2:.3f}, k=4 均值={mean_k4:.3f}, mixed+filter 均值={mean_mixed:.3f}")
```

- [ ] **Step 6: Run Python compile check**

Run:

```bash
cd 'notebook/C7 高级 RAG 技巧/6. 增强阶段'
/usr/local/Caskroom/miniconda/base/envs/py310/bin/python -m py_compile _test_routing_fix.py
```

Expected: no output and exit status 0.

- [ ] **Step 7: Run a two-question smoke experiment before the full run**

Temporarily set this line in `_test_routing_fix.py`:

```python
EVAL_INDICES = [12, 62]
```

Run:

```bash
cd 'notebook/C7 高级 RAG 技巧/6. 增强阶段'
/usr/local/Caskroom/miniconda/base/envs/py310/bin/python -u _test_routing_fix.py
```

Expected: script prints all three sections: `routing_k2`, `routing_k4`, and `routing_mixed`, and creates `_eval_routing_mixed.csv`.

- [ ] **Step 8: Restore the full evaluation set**

Restore:

```python
EVAL_INDICES = sorted(set(QA_INDICES) | set(flatten_demo_indices(DEMO_CASES)))
```

If `_test_routing_fix.py` does not use `DEMO_CASES`, restore the current pre-task expression exactly as it was before Step 7.

- [ ] **Step 9: Commit the experiment harness**

Run:

```bash
git add 'notebook/C7 高级 RAG 技巧/6. 增强阶段/_test_routing_fix.py' 'notebook/C7 高级 RAG 技巧/6. 增强阶段/_eval_routing_mixed.csv'
git commit -m 'test: compare mixed routing evidence selection'
```

Expected: commit succeeds. If commits are disabled, document the modified files and whether `_eval_routing_mixed.csv` was generated.

---

### Task 6: Update Teaching Summary

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb:1602`
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/4. 选型总结.md`

- [ ] **Step 1: Update the notebook comparison guidance**

In `2. 流程增强.ipynb`, in the markdown section `## 怎么读这张对比表`, add this paragraph:

```markdown
对 Routing，不要只看 `k` 或总均值。真正要检查的是：router 是否把“公式/推导题”送到保留公式邻近文本的小块高召回策略，把“概念/比较题”送到大块上下文策略；当问题跨两类证据时，是否用 `mixed_index` 做保守融合，并通过 `selected_hits` 证明没有把所有噪声都塞给生成模型。
```

- [ ] **Step 2: Update the final warning paragraph**

Replace the existing warning paragraph that starts with `如果某个方法在主表中没有提分` with:

```markdown
如果某个方法在主表中没有提分，不要直接写成“方法无效”。先回到该方法的 trace 定位失败环节：router 是否选错分支、mixed 是否因为低置信度被触发、evidence selector 是否丢掉了关键公式、grader 是否过滤了有用证据、recursive 是否拆出重复子问题、Self-RAG 是否过早 `FINISH`、生成阶段是否没有吸收新增证据。只有 trace 能说明失败发生在哪里。
```

- [ ] **Step 3: Update `4. 选型总结.md` routing guidance**

Find the Query Routing / 路由相关小节 in `4. 选型总结.md`. Add this bullet under the routing method guidance:

```markdown
- **适用前提**：问题类型会改变证据策略，例如公式推导题需要小 chunk、高召回和公式邻近文本，概念比较题需要大 chunk 和连续上下文；如果一个通用 retriever 已经能稳定覆盖所有证据，Routing 可能只增加延迟和误路由风险。
```

- [ ] **Step 4: Validate summary text exists**

Run:

```bash
rg -n 'selected_hits|公式/推导题|适用前提|误路由风险' \
  'notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb' \
  'notebook/C7 高级 RAG 技巧/6. 增强阶段/4. 选型总结.md'
```

Expected: output includes matches in both files.

- [ ] **Step 5: Commit teaching summary updates**

Run:

```bash
git add 'notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb' 'notebook/C7 高级 RAG 技巧/6. 增强阶段/4. 选型总结.md'
git commit -m 'docs: clarify routing problem-solving criteria'
```

Expected: commit succeeds. If commits are disabled, document the modified files in the handoff.

---

### Task 7: Final Validation

**Files:**
- Validate: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`
- Validate: `notebook/C7 高级 RAG 技巧/6. 增强阶段/_test_routing_fix.py`
- Validate: `notebook/C7 高级 RAG 技巧/6. 增强阶段/4. 选型总结.md`

- [ ] **Step 1: Run notebook structural validation**

Run:

```bash
python - <<'PY'
import nbformat
from pathlib import Path
p = Path('notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb')
nb = nbformat.read(p, as_version=4)
nbformat.validate(nb)
print(f'valid notebook: {len(nb.cells)} cells')
PY
```

Expected: prints `valid notebook:` and exits with status 0.

- [ ] **Step 2: Run routing invariant checks**

Run:

```bash
python - <<'PY'
import json, re
from pathlib import Path
p = Path('notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
text = '\n'.join(''.join(c.get('source', [])) for c in nb['cells'])
checks = {
    'six_methods': len(re.findall(r'^## 方法 \d：', text, flags=re.M)) == 6,
    'routing_cases': '"routing_general": [62, 84]' in text and '"routing_math": [12, 19]' in text,
    'structured_router': 'def parse_router_decision(raw: str) -> dict' in text,
    'mixed_retrieve': 'def mixed_retrieve(question: str) -> tuple[list, dict]' in text,
    'evidence_selector': 'def select_routing_evidence(question: str, docs: list, max_docs: int = 4) -> list' in text,
    'selected_hits_trace': 'selected_hits' in text,
    'same_run_panel': 'METHOD_RECORDS' in text and '面板不得重新调用 pipeline 生成 trace' in text,
    'no_sota_claim': '不是 SOTA 排名' in text,
}
for name, ok in checks.items():
    print(f'{name}: {ok}')
failed = [name for name, ok in checks.items() if not ok]
raise SystemExit(1 if failed else 0)
PY
```

Expected: all checks print `True`.

- [ ] **Step 3: Run Python compile check**

Run:

```bash
cd 'notebook/C7 高级 RAG 技巧/6. 增强阶段'
/usr/local/Caskroom/miniconda/base/envs/py310/bin/python -m py_compile _common.py _context_enhance_utils.py _run_eval.py _run_eval_continue.py _test_routing_fix.py
```

Expected: no output and exit status 0.

- [ ] **Step 4: Run LLM smoke test for parser and one route**

Run:

```bash
cd 'notebook/C7 高级 RAG 技巧/6. 增强阶段'
/usr/local/Caskroom/miniconda/base/envs/py310/bin/python - <<'PY'
from _common import llm_call
print(llm_call('只输出 OK', sleep_after=0))
PY
```

Expected: prints `OK`. If this fails with network or API error, record it as environment failure, not code failure.

- [ ] **Step 5: Review diff scope**

Run:

```bash
git diff --stat -- \
  'notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb' \
  'notebook/C7 高级 RAG 技巧/6. 增强阶段/_test_routing_fix.py' \
  'notebook/C7 高级 RAG 技巧/6. 增强阶段/4. 选型总结.md'
```

Expected: diff is limited to routing implementation, routing experiment harness, and routing teaching guidance.

- [ ] **Step 6: Final commit**

Run:

```bash
git status --short
git add \
  'notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb' \
  'notebook/C7 高级 RAG 技巧/6. 增强阶段/_test_routing_fix.py' \
  'notebook/C7 高级 RAG 技巧/6. 增强阶段/4. 选型总结.md'
git commit -m 'feat: make routing solve evidence-strategy selection'
```

Expected: commit succeeds. If the workstream intentionally avoids commits, leave the files staged or unstaged according to maintainer preference and provide a handoff summary with exact commands run.

---

## Self-Review

**Spec coverage:** This plan covers the current requirement to think beyond `k` tuning, optimize questions so the method solves a real problem, and prepare work that can be delegated to other agents. Task 1 audits and documents problem-fit questions. Tasks 2-4 implement a structured router, mixed fallback, and trace validation. Task 5 validates the approach with an experiment harness. Task 6 updates teaching docs so claims match evidence. Task 7 provides final validation.

**Placeholder scan:** The plan contains no banned placeholder markers. Every code-changing step includes exact code or exact replacement text. Every validation step includes exact commands and expected outcomes.

**Type consistency:** `parse_router_decision(raw: str) -> dict`, `select_routing_evidence(question: str, docs: list, max_docs: int = 4) -> list`, and `mixed_retrieve(question: str) -> tuple[list, dict]` are defined before use. Trace fields are consistent across implementation and validation: `route.target`, `route.confidence`, `retrieve.branch`, `retrieve.n_hits`, `retrieve.context_chars`, `retrieve.first_hit`, and `retrieve.selected_hits`.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-06-c7-6-2-routing-problem-solving.md`. Two execution options:

**1. Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
