# C7 6.2《流程增强》实现计划

> **给执行 agent：** 必须使用 `superpowers:executing-plans` 按任务执行本计划。小步完成并逐项勾选，不新增第 7 个 runnable 方法，保留六个教学方法：`iterative retrieval`、`recursive decomposition`、`query routing`、`adaptive retrieval`、`CRAG` 和 `Self-RAG`。

**目标：** 重构 `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`，让 6.2 聚焦“单次请求内的 RAG 流程决策”：何时补检索、何时拆解、何时选择检索策略、何时过滤证据、何时自评继续或停止。最终 notebook 必须用同一套 baseline、同一套裁判、同一套运行同时产出主评估表与 method case panel，并为每个方法提供 2~3 个真实可运行、能展示相对 baseline 机制差异的 targeted cases；`routing` 必须覆盖 `general` 与 `math`，`adaptive` 必须覆盖 `simple`、`moderate`、`complex`。

**架构：** 保留 notebook 内的轻量教学实现，不引入新的工程框架。`build_retriever` 保留并使用 `2. 流程增强.ipynb` 内现有定义，本计划不要求从 `_common.py` 导入或迁移 `build_retriever`，也不修改 `_common.py`。公共底座只从 `_common.py` 复用已存在的 `run_shared_eval`、`build_compare_table`、`trim_context_to_budget`、`build_rag_generation_prompt`、`llm_call` 等能力；6.2 本节只在 notebook 内组织 pipeline、trace、demo case、评估表和教学解释。主评估表、method df、targeted/case panel 必须来自同一次评估运行缓存的 `(answer, trace, score, expected)` 证据链，不允许 panel 单独重新调用 pipeline 生成 trace。正式验收的 2~3 个 targeted cases 必须来自 `DEMO_CASES`，并进入同一轮 `METHOD_RECORDS` / `compare_df` 证据链；`EXTRA_DEMO_CASES` 只能作为补充 inspect 或边界演示，不能绕过同跑、同裁判、同 baseline。`query routing` 的边界必须硬限定为单次请求内的检索策略或索引配置选择，例如 chunk 粒度、top-k、数学/概念分支；不进入多文档 agent、工具路由、跨请求记忆或长期状态。

**技术栈：** Python 3.10、Jupyter notebook、nbformat、pandas、LangChain retriever、Chroma、智谱 GLM、现有 `_common.py` 公共底座、`train_dataset.json` 南瓜书 QA 题库。

---

## 实施规则

- 只实现和教学主线直接相关的内容；不新增第 7 个 runnable 方法。
- 不把评估结果写成 SOTA 排名；所有结果说明都必须标注小样本、LLM-as-judge、教学评估。
- 不为了演示效果修改方法核心 prompt、pipeline、停止条件或 grader 逻辑；case 不合适时先扩大自然题候选池。
- 不能把“没触发”或“没提升”当作合格交付。若某方法指标未提升，必须用 trace 定位失败环节，并继续寻找更合适的自然 case，直到有可教学的真实 case。失败定位 case 可以保留用于讲解，但不能替代正式 targeted case。
- 不写 git commit 步骤。本计划只允许写验证命令、notebook smoke test、结构检查和 `git diff -- ...`。

### 任务 1：审计当前 notebook 并吸收旧计划中仍可用的材料

**文件：**
- 检查：`docs/superpowers/specs/2026-05-23-c7-6-2-flow-enhancement-design.md`
- 检查：`docs/superpowers/plans/2026-05-10-c7-6-2-flow-question-design.md`
- 检查：`notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`
- 检查：`notebook/C7 高级 RAG 技巧/6. 增强阶段/_common.py`
- 检查：`notebook/C7 高级 RAG 技巧/3. 索引阶段/data/train_dataset.json`
- 新建：无
- 修改：无
- 验证：下面的 notebook 结构检查命令

- [ ] **步骤 1：导出 notebook 标题和关键符号**

Run from repository root:

```bash
python - <<'PY'
import json
from pathlib import Path

nb_path = Path("notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    lines = [line for line in src.splitlines() if line.startswith("#") or line.startswith("def ") or "QA_INDICES" in line or "DEMO_CASES" in line or "compare_df" in line]
    if lines:
        print(f"\n--- cell {i} {cell['cell_type']} ---")
        for line in lines[:12]:
            print(line)
PY
```

预期结果：输出显示开头边界小节、统一接口、`build_retriever` 的 notebook 内定义、`baseline_pipeline`、六个方法 pipeline、`DEMO_CASES`、评估单元、method panel 和最终总结单元。若缺少任一项，先记录缺失项再编辑。

- [ ] **步骤 2：只把旧计划当作素材库审计**

Use the old plan only to harvest still-valid implementation ideas:

```text
- baseline_pipeline and baseline inspect cell
- DEMO_CASES and broad natural case mining
- routing with two real retriever configurations
- trace validation helpers
- main compare table
- method-level panel
- final summary derived from current run
```

预期结果：如果旧计划任务与最新 spec 冲突，不整段照搬。尤其要移除旧 commit 步骤、移除让 6.2 routing 看起来像工具路由或多源 agent 路由的措辞，并移除静态结果断言。

- [ ] **步骤 3：确认公共 helper 和 notebook 内 retriever 定义**

先在 `2. 流程增强.ipynb` 中确认已经存在 `def build_retriever(...)`，并保留该 notebook 内定义作为本节 retriever 工厂。然后检查 `_common.py`，只确认下面这些导入对 notebook 可用：

```python
import json
from pathlib import Path

from _common import (
    QA_PATH,
    CONTEXT_CHAR_BUDGET,
    trim_context_to_budget,
    build_rag_generation_prompt,
    llm_call,
    load_qna_subset,
    run_shared_eval,
    build_compare_table,
)
```

预期结果：`build_retriever` 来自当前 notebook，不从 `_common.py` 导入；本计划不要求修改 `_common.py`。`json` 和 `Path` 在 notebook setup/import 单元中先导入，供后续 case mining、缓存读取和结构检查使用。执行者只能补充缺失 import，不要用上面的 import 片段整段覆盖 notebook 已有 setup；必须保留 notebook 内 `build_retriever` 及其依赖的现有导入、变量和函数。若 notebook 重复了 `_common.py` 中已存在且稳定的非 retriever helper，可优先导入；不要在本轮迁移 `build_retriever`。

- [ ] **步骤 4：审计候选数据集字段**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("notebook/C7 高级 RAG 技巧/3. 索引阶段/data/train_dataset.json")
data = json.loads(p.read_text(encoding="utf-8"))
for i in [0, 1, 4, 5, 7, 12, 15, 18, 19, 36, 42, 62, 63, 86, 101, 113, 117]:
    item = data[i]
    print(f"{i:03d}\tpage={item.get('page_num')}\tq={item['query'][:70]}\ta_len={len(item['answer'])}")
PY
```

预期结果：所有列出的索引都存在，并且 `query` 与 `answer` 非空。除非 notebook 审计发现更强的自然 case，否则使用这个候选池做主评估和 case mining。

### 任务 2：重构开篇定位与边界导航

**文件：**
- 检查：`notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`
- 修改：`notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`
- 新建：无
- 验证：下面的 markdown heading 和 boundary grep 命令

- [ ] **步骤 1：替换或收紧开篇主旨**

At the top of the notebook, keep the title `# 2. 流程增强`, then ensure the first explanatory block contains this exact teaching thesis:

```markdown
## 这一节在解决什么问题

基础 RAG 通常是一次检索、一次生成。6.2 讨论的不是“把 query 写得更像文档”，也不是“把已检索上下文拼得更完整”，而是：当一次请求中的基础流程不够用时，系统如何改变检索、筛选、生成和自评的顺序与条件。

本节的关键词是 **控制流**：什么时候再检索、什么时候先拆问题、什么时候选择不同检索配置、什么时候过滤证据、什么时候让模型检查自己是否应该继续。
```

预期结果：读者在看到方法名称前，就能识别本节的单一主线。

- [ ] **步骤 2：在主旨后立即添加章节硬边界**

Insert this markdown near the beginning, before method overview:

```markdown
## 与第 4 章 / 6.1 / 6.3 的边界

| 章节 | 关注点 | 本节是否展开 |
|---|---|---|
| 第 4 章 Query/Document 对齐 | 让问题更容易命中文档，例如 query rewrite、HyDE、对齐表示 | 不重复实现，只在必要处引用 |
| 6.1 上下文增强 | 检索后如何组织上下文，例如 Sentence Window、Small-to-Big、AutoMerging | 不重复讲上下文拼接细节 |
| 6.2 流程增强 | 单次请求内如何改变 RAG 控制流 | 本节主线 |
| 6.3 系统增强 | 跨请求、多文档、多工具、记忆、agent 编排 | 不前移到本节 |

因此，本节的 query routing 只表示“单次请求内选择检索策略或索引配置”，例如 chunk 粒度、top-k、数学/概念分支；它不是多文档 agent、不是工具路由，也不维护跨请求状态。
```

预期结果：notebook 第一屏已经阻断 routing 边界漂移。

- [ ] **步骤 3：在六个方法前添加统一观察框架**

Add this markdown before the six-method overview table:

```markdown
## 统一观察框架

后面每个方法都回答同一组问题：

1. 它改变了 RAG 的哪个流程决策点？
2. 它针对什么失败模式？
3. 它的 trace 中应该出现什么信号？
4. 它相对 baseline 改善了证据、答案覆盖，还是只增加了流程成本？
5. 它在哪些题型上可能不值得使用？
```

预期结果：通用阅读问题只出现一次，后续方法小节可以更短、更聚焦。

- [ ] **步骤 4：验证边界措辞**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

nb = json.loads(Path("notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb").read_text(encoding="utf-8"))
text = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
required = [
    "控制流",
    "第 4 章 Query/Document 对齐",
    "6.1 上下文增强",
    "6.3 系统增强",
    "单次请求内选择检索策略或索引配置",
    "不是多文档 agent",
    "不是工具路由",
    "不维护跨请求状态",
]
missing = [s for s in required if s not in text]
print("missing:", missing)
raise SystemExit(1 if missing else 0)
PY
```

预期结果：`missing: []`。

### 任务 3：标准化方法模板并修正方法解释

**文件：**
- 检查：`notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`
- 修改：`notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`
- 新建：无
- 验证：下面的方法标题/顺序断言命令

- [ ] **步骤 1：保持恰好六个 runnable 方法并按决策点排序**

Ensure the notebook has exactly these method headings in this order:

```text
## 方法 1：迭代检索（Iterative Retrieval）
## 方法 2：递归分解（Recursive Decomposition）
## 方法 3：查询路由（Query Routing）
## 方法 4：自适应检索（Adaptive Retrieval）
## 方法 5：Corrective RAG（CRAG）
## 方法 6：Self-RAG（教学化简版）
```

预期结果：没有第七个 `## 方法` 标题，也没有 baseline 加六个方法之外的额外 runnable pipeline。

- [ ] **步骤 2：为每个方法应用相同章节模板**

Each method section should contain these subheadings, with local content filled from the existing implementation:

```markdown
### 直觉
### 流程图
### 适用 / 不适用
### 最小实现
### 看 trace 学到了什么
### 增益与局限
```

预期结果：重复 setup 说明移动到开头或公共单元；每个方法只聚焦自己的决策点和 trace。

- [ ] **步骤 3：修正 iterative 与 recursive 的措辞**

In the recursive section, use this comparison table:

```markdown
|  | 迭代检索 | 递归分解 |
|---|---|---|
| 决策时机 | 先生成草稿，再根据缺口决定是否补检索 | 生成前先拆问题，再分别检索子问题 |
| 解决的问题 | 首轮答案漏要点、证据不足、需要补查 | 原问题包含多个子任务或多跳依赖 |
| trace 信号 | 多次 `retrieve`，或出现缺口提示与继续检索理由 | `decompose` 后出现多个 `sub_answer` |
| 主要风险 | LLM 不知道自己缺什么，可能过早停止 | 子问题拆错或重复，可能稀释主问题 |
```

预期结果：notebook 不再把 iterative 和 recursive 都笼统写成“多走几步”。

- [ ] **步骤 4：集中添加每个方法的 trace 信号**

Add this markdown near the method overview:

```markdown
| 方法 | 必须观察的 trace 信号 |
|---|---|
| Iterative | 是否出现草稿缺口、第二轮 `retrieve` 或明确的停止原因 |
| Recursive | `decompose` 是否产生互补子问题，`sub_answer` 是否覆盖不同角度 |
| Routing | `routing_general` 必须走 `general_index`，`routing_math` 必须走 `math_index`；分支的 `n_hits` / `context_chars` / `first_hit` 至少有可解释差异 |
| Adaptive | `adaptive_simple` / `adaptive_moderate` / `adaptive_complex` 至少触发两个不同 delegate，理想三类都不同 |
| CRAG | `grade` 是否把证据标成 `relevant` / `partial` / `irrelevant`，并用 `before_context_chars` / `after_context_chars` / `dropped_count` 或 context hash 证明上下文真的变化 |
| Self-RAG | `reflect` 后的 `FINISH` / `CONTINUE` 是否带来真实检查效果；至少一个正式 case 需要非负/正向覆盖变化或明确 baseline 对比收益 |
```

预期结果：trace validation criteria 和教学讲解共用同一套语言。

- [ ] **步骤 5：验证方法顺序和数量**

Run:

```bash
python - <<'PY'
import json, re
from pathlib import Path

nb = json.loads(Path("notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb").read_text(encoding="utf-8"))
text = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
methods = re.findall(r"^## 方法 \d：.+$", text, flags=re.M)
print("\n".join(methods))
assert len(methods) == 6, methods
assert "方法 2：递归分解" in methods[1], methods[1]
assert "方法 3：查询路由" in methods[2], methods[2]
PY
```

预期结果：命令成功退出，并且只打印六个方法标题。

### 任务 4：建立 baseline pipeline、严格 trace validation 与同跑评估证据链

**文件：**
- 检查：`notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`
- 检查：`notebook/C7 高级 RAG 技巧/6. 增强阶段/_common.py`
- 修改：`notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`
- 新建：无
- 验证：下面的 baseline/evaluation 符号 smoke command

- [ ] **步骤 1：确保 baseline 使用相同 pipeline 签名**

Keep or add this implementation in the unified interface section:

```python
def baseline_pipeline(question: str) -> tuple[str, list[dict]]:
    trace: list[dict] = []
    docs = retriever.invoke(question)
    ctx = trim_context_to_budget("\n\n".join(d.page_content for d in docs), CONTEXT_CHAR_BUDGET)
    trace.append(step("retrieve", branch="baseline", n_hits=len(docs), context_chars=len(ctx)))
    answer = llm_call(build_rag_generation_prompt(question, ctx))
    trace.append(step("finalize", reason="single_pass"))
    return answer, trace
```

预期结果：baseline 和六个方法一样返回 `(answer, trace)`。

- [ ] **步骤 2：在任何 case mining 之前定义严格 `validate_demo_trace`**

在 notebook 中先替换或定义严格 validation helper，再执行任何 `mine_demo_case(...)`。不要先用宽松标准选 case、后续 panel 再失败；如果 validation helper 发现某题不满足机制要求，该题不能进入正式 `DEMO_CASES` 的优先位置。

Use or update:

```python
def validate_demo_trace(method_name: str, trace: list[dict], case_key: str | None = None) -> dict:
    kinds = [s.get("kind", "") for s in trace]
    if method_name == "iterative":
        has_gap_reason = any(s.get("missing") or "缺" in str(s.get("reason", "")) for s in trace)
        reached_max_rounds = any(s.get("reason") == "max_rounds_reached" for s in trace)
        ok = (kinds.count("retrieve") >= 2 or reached_max_rounds) and has_gap_reason
        expected = "至少两次 retrieve，或明确到达 max_rounds；同时必须出现缺口/补检索原因字段"
    elif method_name == "recursive":
        sub_questions = []
        for s in trace:
            if s.get("kind") == "decompose":
                sub_questions.extend(s.get("sub_questions") or s.get("questions") or [])
            if s.get("kind") == "sub_question":
                sub_questions.append(s.get("question") or s.get("sub_question"))
        unique_sub_questions = {str(q).strip() for q in sub_questions if str(q).strip()}
        diversity_review_ok = any(s.get("sub_question_diversity_ok") is True for s in trace)
        sub_answers = [s for s in trace if s.get("kind") == "sub_answer"]
        ok = "decompose" in kinds and len(sub_answers) >= 2 and (len(unique_sub_questions) >= 2 or diversity_review_ok)
        expected = "出现 decompose，至少两个 sub_answer，且 sub_questions 去重后不少于 2 个或 panel 标记 sub_question_diversity_ok"
    elif method_name == "routing":
        retrieves = [s for s in trace if s.get("kind") == "retrieve"]
        branches = [s.get("branch") for s in retrieves if s.get("branch")]
        expected_branch = {
            "routing_general": "general_index",
            "routing_math": "math_index",
        }.get(case_key)
        has_retrieval_shape = bool(retrieves) and all(
            {"n_hits", "context_chars", "first_hit"}.issubset(s.keys())
            for s in retrieves
        )
        branch_ok = expected_branch is None or expected_branch in branches
        ok = "route" in kinds and has_retrieval_shape and branch_ok
        expected = "出现 route 和对应 branch retrieve；routing_general 必须 general_index，routing_math 必须 math_index；暴露 branch/n_hits/context_chars/first_hit"
    elif method_name == "adaptive":
        delegates = {s.get("target") or s.get("delegate") for s in trace if s.get("kind") == "delegate"}
        ok = "classify" in kinds and bool(delegates)
        expected = "单题出现 classify 和 delegate；跨 simple/moderate/complex panel 至少覆盖两个 delegate，理想三个"
    elif method_name == "crag":
        tags = [s.get("tag") for s in trace if s.get("kind") == "grade"]
        context_changes = [
            s for s in trace
            if s.get("kind") in {"filter", "rebuild_context"}
            and (
                s.get("before_context_chars") != s.get("after_context_chars")
                or s.get("context_hash_before") != s.get("context_hash_after")
                or (s.get("dropped_count") or 0) > 0
            )
        ]
        ok = any(t in {"partial", "irrelevant"} for t in tags) and bool(context_changes)
        expected = "至少一条证据被判为 partial/irrelevant，且 before/after context chars、dropped_count 或 context hash 证明上下文改变"
    elif method_name == "self_rag":
        verdicts = [s.get("verdict") for s in trace if s.get("kind") == "reflect"]
        has_reflection_verdict = any(v in {"CONTINUE", "FINISH"} for v in verdicts)
        coverage_deltas = [
            s.get("answer_coverage_delta")
            for s in trace
            if isinstance(s.get("answer_coverage_delta"), (int, float))
        ]
        has_non_negative_coverage_delta = any(delta >= 0 for delta in coverage_deltas)
        has_positive_coverage_delta = any(delta > 0 for delta in coverage_deltas)
        has_explainable_stop = any(
            s.get("stop_reason") or s.get("coverage_note") or s.get("failure_location") or s.get("failure_reason")
            for s in trace
        )
        ok = has_reflection_verdict and (has_non_negative_coverage_delta or has_positive_coverage_delta or has_explainable_stop)
        expected = "必须有 reflection verdict；正式通过需要非负/正向覆盖变化或 panel 中明确 baseline 对比收益；失败定位字段只能说明失败，不能替代唯一合格 case"
    else:
        ok = bool(trace)
        expected = "trace 非空"
    return {"method": method_name, "case_key": case_key, "ok": ok, "expected": expected, "kinds": " -> ".join(kinds)}
```

预期结果：validation helper 位于 case mining 单元之前。`routing`、`CRAG` 和 `Self-RAG` 的代码级 `ok` 判定不能比文字标准更宽松；`routing` 分支差异、`adaptive` 多 delegate 覆盖和 `Self-RAG` 正式收益仍需在 panel 聚合层 fail-fast 验证。

- [ ] **步骤 3：建立同跑 trace 缓存评估 helper**

使用这个 helper 让所有方法共用 `run_shared_eval`、同一裁判、同一 baseline，并在同一次运行中缓存每题的 `answer`、`trace`、`score`、`expected`。后续 method df、`compare_df`、targeted/case panel 都只能从这些返回值和 `METHOD_RECORDS` 读取证据，不允许为了 panel 再调用 pipeline：

```python
def eval_pipeline_with_trace(method_name: str, pipeline_fn, qna_dict) -> tuple[pd.DataFrame, list[dict]]:
    """Run one pipeline once per question and keep answer/trace/score/expected together."""
    records_by_question: dict[str, dict] = {}

    def answer_fn(question: str) -> str:
        answer, trace = pipeline_fn(question)
        records_by_question[question] = {
            "method": method_name,
            "question": question,
            "expected": qna_dict[question],
            "answer": answer,
            "trace": trace,
        }
        return answer

    df = run_shared_eval(answer_fn, qna_dict, eval_prompt_template=DIMENSIONAL_EVAL_PROMPT)
    score_by_question = dict(zip(df["question"], df["rag_eval_results"]))

    records = []
    for question, record in records_by_question.items():
        records.append({
            **record,
            "score": score_by_question.get(question),
        })
    return df, records
```

预期结果：baseline 和六个方法都通过 `run_shared_eval` 与 `DIMENSIONAL_EVAL_PROMPT` 评估；每个问题只调用一次对应 pipeline；trace 与 score 保存在同一条 record 中。

- [ ] **步骤 4：在 DEMO_CASES 确定后构建同一轮主评估表和 METHOD_RECORDS**

helper 可以在前面先定义，但这个 full evaluation cell 只能在 `DEMO_CASES` / targeted cases 最终确定、且严格 `validate_demo_trace(...)` 已定义并被 mining 使用之后真正执行。`qna_dict` 必须同时包含主评估题和 targeted cases，这样 method panel 可以从同一轮 `METHOD_RECORDS` 读取 trace，而不是另跑 pipeline：

```python
def flatten_demo_indices(demo_cases: dict[str, list[int]]) -> list[int]:
    return sorted({idx for values in demo_cases.values() for idx in values})

EVAL_INDICES = sorted(set(QA_INDICES) | set(flatten_demo_indices(DEMO_CASES)))
qna_dict = load_qna_subset(QA_PATH, EVAL_INDICES)

METHOD_RUNS = {
    "baseline": baseline_pipeline,
    "iterative": iterative_pipeline,
    "recursive": recursive_pipeline,
    "routing": routing_pipeline,
    "adaptive": adaptive_pipeline,
    "crag": crag_pipeline,
    "self_rag": self_rag_pipeline,
}

METHOD_DFS: dict[str, pd.DataFrame] = {}
METHOD_RECORDS: dict[str, list[dict]] = {}

for method_name, pipeline_fn in METHOD_RUNS.items():
    method_df, records = eval_pipeline_with_trace(method_name, pipeline_fn, qna_dict)
    METHOD_DFS[method_name] = method_df
    METHOD_RECORDS[method_name] = records

baseline_df  = METHOD_DFS["baseline"]
iterative_df = METHOD_DFS["iterative"]
recursive_df = METHOD_DFS["recursive"]
routing_df   = METHOD_DFS["routing"]
adaptive_df  = METHOD_DFS["adaptive"]
crag_df      = METHOD_DFS["crag"]
selfrag_df   = METHOD_DFS["self_rag"]

compare_df = build_compare_table(
    [baseline_df, iterative_df, recursive_df, routing_df, adaptive_df, crag_df, selfrag_df],
    names=["baseline", "iterative", "recursive", "routing", "adaptive", "crag", "self_rag"],
)
compare_df
```

预期结果：`compare_df` 每个问题一行并包含这些方法列；`METHOD_RECORDS` 与 `METHOD_DFS` 来自同一个循环、同一批 `qna_dict`、同一个 judge，没有第二套 panel-only pipeline 调用。主评估题和 targeted cases 共享同一 baseline、同一裁判、同一批方法实现。

- [ ] **步骤 5：在 QA 集附近添加评估说明**

Use this markdown:

```markdown
本节评估是教学评估，不是开放域 benchmark 或 SOTA 排名。样本量小，裁判是 LLM-as-judge，结果会受模型版本、限流重试、检索缓存和问题选择影响。主表用于观察整体趋势；分方法 case panel 用于观察适配题型上的机制差异。
```

预期结果：读者不会把表格误读为排行榜。

- [ ] **步骤 6：不运行 LLM 调用的评估符号 smoke check**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

nb = json.loads(Path("notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb").read_text(encoding="utf-8"))
text = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
required = [
    "def baseline_pipeline(question: str) -> tuple[str, list[dict]]",
    "def validate_demo_trace(method_name: str, trace: list[dict], case_key: str | None = None) -> dict",
    "def eval_pipeline_with_trace(method_name: str, pipeline_fn, qna_dict)",
    "def flatten_demo_indices(demo_cases: dict[str, list[int]]) -> list[int]",
    "EVAL_INDICES = sorted(set(QA_INDICES) | set(flatten_demo_indices(DEMO_CASES)))",
    "METHOD_RUNS = {",
    "METHOD_DFS: dict[str, pd.DataFrame] = {}",
    "METHOD_RECORDS: dict[str, list[dict]] = {}",
    'baseline_df  = METHOD_DFS["baseline"]',
    'selfrag_df   = METHOD_DFS["self_rag"]',
    'names=["baseline", "iterative", "recursive", "routing", "adaptive", "crag", "self_rag"]',
    "教学评估",
    "LLM-as-judge",
]
missing = [s for s in required if s not in text]
print("missing:", missing)
raise SystemExit(1 if missing else 0)
PY
```

预期结果：`missing: []`。

### 任务 5：以边界为先实现两个真实 retriever 分支的 query routing

**文件：**
- 检查：`notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`
- 检查：`notebook/C7 高级 RAG 技巧/6. 增强阶段/_common.py`
- 修改：`notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`
- 新建：无
- 验证：下面的 routing branch smoke command

- [ ] **步骤 1：重写 routing 边界说明**

The routing section must include this block:

```markdown
> **本节的 routing 边界**：这里的 query routing 只发生在单次请求内部，用来选择检索策略或索引配置。教学实现使用同一本南瓜书构造两个不同 branch：
>
> - `general_index`：较大 chunk、较少 top-k，适合概念解释和比较题；
> - `math_index`：较小 chunk、更多 top-k，适合公式、推导、矩阵、优化类题。
>
> 它不负责多文档 agent、工具调用路由、跨请求记忆或长期状态。那些属于 6.3 系统增强。
```

预期结果：routing 的边界在方法小节内再次被明确。

- [ ] **步骤 2：确保 routing 使用两个不同 retriever 配置**

使用当前 notebook 内已经定义的 `build_retriever`，不要从 `_common.py` 导入它，也不要为本轮修改 `_common.py`：

```python
general_retriever = build_retriever(chunk_size=512, chunk_overlap=60, k=2)
math_retriever = build_retriever(chunk_size=128, chunk_overlap=20, k=8)

INDEX_TO_RETRIEVER = {
    "math_index": math_retriever,
    "general_index": general_retriever,
}
```

预期结果：routing 不再把两个 branch 指向同一个 retriever；`general_retriever` 与 `math_retriever` 都由 notebook 内现有 `build_retriever` 构建。

- [ ] **步骤 3：让 routing trace 暴露分支差异**

Inside `routing_pipeline`, the retrieve trace must include:

```python
trace.append(step(
    "retrieve",
    branch=branch,
    n_hits=len(docs),
    context_chars=len(ctx),
    first_hit=docs[0].page_content[:80] if docs else "",
))
```

预期结果：panel 可以展示 branch choice、hit count、context length 和 first hit snippet。

- [ ] **步骤 4：使用两个 routing demo calls，并把错误分支视为验收失败**

The routing inspect cell should call both branches:

```python
inspect_routing(demo_question("routing_general"))
inspect_routing(demo_question("routing_math"))
```

预期结果：`routing_general` 必须 route 到 `general_index`，`routing_math` 必须 route 到 `math_index`。如果 LLM router 选错 branch，保留输出作为失败定位 case 并修正 case 选择或 routing 规则；该题不能通过正式 targeted case 验收。两个正式 case 的 retrieve trace 中，`n_hits`、`context_chars`、`first_hit` 至少一项必须有可解释差异，否则说明两个 branch 没有真正改变检索行为。

- [ ] **步骤 5：验证 routing 符号**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

nb = json.loads(Path("notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb").read_text(encoding="utf-8"))
text = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
required = [
    "general_retriever = build_retriever(chunk_size=512, chunk_overlap=60, k=2)",
    "math_retriever = build_retriever(chunk_size=128, chunk_overlap=20, k=8)",
    '"math_index": math_retriever',
    '"general_index": general_retriever',
    "first_hit=docs[0].page_content[:80] if docs else",
    "不是多文档 agent",
    "不是工具调用路由",
]
missing = [s for s in required if s not in text]
print("missing:", missing)
raise SystemExit(1 if missing else 0)
PY
```

预期结果：`missing: []`。

### 任务 6：实现自然 case 挖掘并确定最终 DEMO_CASES

**文件：**
- 检查：`notebook/C7 高级 RAG 技巧/3. 索引阶段/data/train_dataset.json`
- 检查：`notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`
- 修改：`notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`
- 新建：无
- 验证：下面的 demo case 结构命令和 notebook demo 单元

- [ ] **步骤 1：定义自然候选题池**

Add or update this dictionary in the demo case section:

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

预期结果：所有候选题都是数据集中的自然问题。

- [ ] **步骤 2：确认严格 validation helper 已前置**

在添加 mining helper 前，确认任务 4 的 `validate_demo_trace(method_name, trace, case_key=None)` 已经在 notebook 中定义，并且它会被下面的 mining helper 直接调用。不能先用宽松标准选 case，再期待 panel 阶段兜底；若 helper 缺失或仍是旧宽松版本，先回到任务 4 修正。

预期结果：case mining 使用的就是最终交付阻断标准，避免选中后续 panel 必然失败的 case。

- [ ] **步骤 3：添加不改变 pipeline 的 mining helper**

Add:

```python
QA_DATA = json.loads(Path(QA_PATH).read_text(encoding="utf-8"))
QA_INDEX_BY_QUESTION = {item["query"]: i for i, item in enumerate(QA_DATA)}

def mine_demo_case(method_name: str, pipeline_fn, candidate_indices: list[int], case_key: str | None = None) -> pd.DataFrame:
    rows = []
    candidate_qna = load_qna_subset(QA_PATH, candidate_indices)
    for question in candidate_qna.keys():
        answer, trace = pipeline_fn(question)
        validation = validate_demo_trace(method_name, trace, case_key=case_key)
        rows.append({
            "idx": QA_INDEX_BY_QUESTION.get(question),
            "question": question,
            "ok": validation["ok"],
            "case_key": case_key,
            "expected": validation["expected"],
            "trace_shape": validation["kinds"],
            "answer_head": short_text(answer, 140),
        })
    return pd.DataFrame(rows)
```

预期结果：setup/import 单元已经先执行 `import json` 和 `from pathlib import Path`。helper 只运行现有 pipelines 并记录 traces；它不改变 prompts、retriever settings、stopping rules 或 grader 逻辑，也不在循环中重复读取 JSON。

- [ ] **步骤 4：为每个方法挖掘 2~3 个 targeted cases**

Run these notebook cells one method at a time:

```python
iterative_mining_df = mine_demo_case("iterative", iterative_pipeline, DEMO_MINING_CANDIDATES["iterative"], case_key="iterative")
display(iterative_mining_df[["idx", "ok", "trace_shape", "question"]])

recursive_mining_df = mine_demo_case("recursive", recursive_pipeline, DEMO_MINING_CANDIDATES["recursive"], case_key="recursive")
display(recursive_mining_df[["idx", "ok", "trace_shape", "question"]])

routing_general_mining_df = mine_demo_case("routing", routing_pipeline, DEMO_MINING_CANDIDATES["routing_general"], case_key="routing_general")
routing_math_mining_df = mine_demo_case("routing", routing_pipeline, DEMO_MINING_CANDIDATES["routing_math"], case_key="routing_math")
display(routing_general_mining_df[["idx", "ok", "trace_shape", "question"]])
display(routing_math_mining_df[["idx", "ok", "trace_shape", "question"]])

adaptive_simple_mining_df = mine_demo_case("adaptive", adaptive_pipeline, DEMO_MINING_CANDIDATES["adaptive_simple"], case_key="adaptive_simple")
adaptive_moderate_mining_df = mine_demo_case("adaptive", adaptive_pipeline, DEMO_MINING_CANDIDATES["adaptive_moderate"], case_key="adaptive_moderate")
adaptive_complex_mining_df = mine_demo_case("adaptive", adaptive_pipeline, DEMO_MINING_CANDIDATES["adaptive_complex"], case_key="adaptive_complex")
display(adaptive_simple_mining_df[["idx", "ok", "trace_shape", "question"]])
display(adaptive_moderate_mining_df[["idx", "ok", "trace_shape", "question"]])
display(adaptive_complex_mining_df[["idx", "ok", "trace_shape", "question"]])

crag_mining_df = mine_demo_case("crag", crag_pipeline, DEMO_MINING_CANDIDATES["crag"], case_key="crag")
display(crag_mining_df[["idx", "ok", "trace_shape", "question"]])

selfrag_mining_df = mine_demo_case("self_rag", self_rag_pipeline, DEMO_MINING_CANDIDATES["self_rag"], case_key="self_rag")
display(selfrag_mining_df[["idx", "ok", "trace_shape", "question"]])
```

预期结果：每个方法默认保留 2~3 个来自 `DEMO_CASES` 的正式 targeted cases。`routing` 必须同时保留 `routing_general` 与 `routing_math`，并分别通过 case key 的 branch 断言；`adaptive` 必须同时保留 `adaptive_simple`、`adaptive_moderate`、`adaptive_complex`，后续 panel 至少触发两个不同 delegate，理想三类。`iterative`、`recursive`、`crag`、`self_rag` 至少各 2 个；若最终只能保留 1 个，只能在 notebook 中作为失败定位或边界说明，不能当作合格默认交付。

- [ ] **步骤 5：对选中 cases 应用严格成功标准**

Use these criteria before putting a case first in `DEMO_CASES`:

```text
Iterative: at least two retrieve steps, or a clear max-round stop after a missing-information loop; trace must explain whether the second retrieval was driven by a concrete gap.
Recursive: decompose plus at least two sub_answer steps that cover distinct subrequirements; repeated or near-duplicate subquestions do not pass.
Routing: `routing_general` must route to `general_index`; `routing_math` must route to `math_index`. Trace must expose branch, n_hits, context_chars, and first_hit, and panel must show at least one explainable difference among n_hits/context_chars/first_hit.
Adaptive: simple/moderate/complex targeted cases must trigger at least two distinct delegate targets, ideally three; not merely "some two delegates appeared" without matching case keys. If two only, explain which complexity bucket collapsed and why.
CRAG: at least one evidence item is graded partial or irrelevant, and the filtered context changes generation input. Trace must record `before_context_chars`/`after_context_chars` plus `dropped_count`, or `context_hash_before`/`context_hash_after`; validation compares these fields, not just the presence of `filter`/`rebuild_context`.
Self-RAG: reflect alone is not enough; FINISH/CONTINUE must be linked to a real checking effect. At least one formal self-rag targeted case needs non-negative/positive coverage change or explicit score/coverage benefit vs baseline. Failure localization cases can remain for teaching, but cannot be the only "passing" self-rag case.
```

预期结果：没有方法仅因为单元打印了输出就被接受；所有合格 case 都能用 trace 解释机制是否真正生效。

- [ ] **步骤 6：只用具体整数索引更新 DEMO_CASES**

After mining, `DEMO_CASES` must contain only real integer dataset indices:

```python
DEMO_CASES = {
    "baseline": [0],
    "iterative": [19, 18, 42],
    "recursive": [4, 42, 63],
    "routing_general": [5, 7],
    "routing_math": [19, 42],
    "adaptive_simple": [14, 84],
    "adaptive_moderate": [5, 62],
    "adaptive_complex": [63, 86],
    "crag": [7, 62, 101],
    "self_rag": [1, 62, 63],
}
```

预期结果：上面的索引是初始具体配置，并已满足每个方法 2~3 个正式 targeted cases 的默认要求。若 mining 找到更强自然 case，用观察到的整数替换对应整数；不要在 notebook 代码中留下符号标记。所有正式 targeted cases 必须来自 `DEMO_CASES`，并通过后续同一轮 `METHOD_RECORDS` / `compare_df` 验证。

- [ ] **步骤 7：仅当需要补充 inspect 或边界演示时添加自然额外题**

`EXTRA_DEMO_CASES` 不能替代正式 targeted cases，也不能绕过同跑、同裁判、同 baseline。它只能作为补充 inspect 或边界演示，并默认不进入 `compare_df`。Approved extra cases:

```python
EXTRA_DEMO_CASES = {
    "iterative_course_natural": "请根据南瓜书相关内容解释牛顿法的基本迭代公式，并说明它与梯度下降法在选择下一个迭代点时的区别。",
    "crag_course_natural": "请根据南瓜书相关内容解释宏平均和微平均的区别，以及它们在类别不平衡时各自可能带来的问题。",
}
```

预期结果：extra cases 是自然学习问题，不是为迎合方法而构造的人工字符串；它们不进入主 `compare_df`，因此也不算正式验收 case。若某个 extra case 要升级为正式 case，必须进入同一轮缓存体系，例如纳入 `EVAL_INDICES` 和 `METHOD_RECORDS`，或使用单独但同跑同裁判的 `EXTRA_METHOD_RECORDS`；并且必须有 baseline 对照、score、trace、expected/评分说明，不能作为 panel-only trace。

- [ ] **步骤 8：验证 demo case 结构**

Run:

```bash
python - <<'PY'
import ast, json
from pathlib import Path

nb = json.loads(Path("notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb").read_text(encoding="utf-8"))
code = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code")
tree = ast.parse(code)
assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]
demo_nodes = [n.value for n in assignments for t in n.targets if isinstance(t, ast.Name) and t.id == "DEMO_CASES"]
assert demo_nodes, "DEMO_CASES not found"
demo = ast.literal_eval(demo_nodes[-1])
required = {"baseline", "iterative", "recursive", "routing_general", "routing_math", "adaptive_simple", "adaptive_moderate", "adaptive_complex", "crag", "self_rag"}
assert set(demo) == required, sorted(set(demo) ^ required)
for key, values in demo.items():
    assert values and all(isinstance(v, int) for v in values), (key, values)
assert len(demo["routing_general"]) >= 2 and len(demo["routing_math"]) >= 2
assert all(len(demo[k]) >= 2 for k in ["adaptive_simple", "adaptive_moderate", "adaptive_complex"])
assert all(len(demo[k]) >= 2 for k in ["iterative", "recursive", "crag", "self_rag"])
print("DEMO_CASES ok")
PY
```

预期结果：`DEMO_CASES ok`。

### 任务 7：用同跑缓存生成 trace validation、method panel 和 targeted case panel

**文件：**
- 检查：`notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`
- 修改：`notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`
- 新建：无
- 验证：下面的 panel 符号 smoke command

- [ ] **步骤 1：复用已前置的严格 trace validation**

不要在这里重新定义一个更宽松的 `validate_demo_trace`。本任务只确认任务 4 中的严格 helper 已经位于 case mining 之前，并且签名为：

```python
def validate_demo_trace(method_name: str, trace: list[dict], case_key: str | None = None) -> dict:
```

预期结果：validation 检查机制可见性，不只检查分数，且代码级 `ok` 判定不能比文字标准更宽松。`routing` 必须能按 `routing_general` / `routing_math` case key 校验 branch；`CRAG` 必须比较 before/after context 或 hash 字段；`Self-RAG` 必须区分正式有效 case 与失败定位 case。`routing` 的分支差异、`adaptive` 的多 delegate 覆盖和 `Self-RAG` 的至少一个正式收益 case 需要在 panel 聚合层继续验证，不能只靠单题 trace。

- [ ] **步骤 2：保持 method-vs-baseline panel 绑定 `compare_df`**

Use:

```python
def flow_score_summary(compare_df):
    methods = [c for c in compare_df.columns if c != "question"]
    baseline = compare_df["baseline"]
    rows = []
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

预期结果：score summary 使用当前 `compare_df`，不使用手写数字。

- [ ] **步骤 3：添加只读取同跑 records 的 targeted panel helper**

Use:

```python
BASELINE_RECORDS_BY_QUESTION = {
    r["question"]: r for r in METHOD_RECORDS["baseline"]
}

def records_for_cases(method_name: str, case_keys: list[str]) -> list[dict]:
    expected_questions = []
    for case_key in case_keys:
        case_qna = load_qna_subset(QA_PATH, DEMO_CASES[case_key])
        expected_questions.extend((case_key, question) for question in case_qna.keys())

    available = {r["question"]: r for r in METHOD_RECORDS[method_name]}
    missing = [q for _, q in expected_questions if q not in available]
    if missing:
        raise ValueError(
            f"{method_name} targeted cases are not in qna_dict / METHOD_RECORDS: {missing[:2]}"
        )
    records = []
    for case_key, question in expected_questions:
        record = {**available[question], "case_key": case_key}
        records.append(record)
    return records

def trace_shape(trace: list[dict]) -> str:
    return " -> ".join(s.get("kind", "") for s in trace)

def first_retrieve(record: dict) -> dict:
    return next((s for s in record["trace"] if s.get("kind") == "retrieve"), {})

def display_flow_method_panel(method_label: str, method_name: str, case_keys: list[str]):
    records = records_for_cases(method_name, case_keys)
    validations = [validate_demo_trace(method_name, r["trace"], case_key=r["case_key"]) for r in records]

    print(f"### {method_label} 真实结果")
    for record, validation in zip(records, validations):
        base = BASELINE_RECORDS_BY_QUESTION[record["question"]]
        score_delta = record["score"] - base["score"]
        coverage_deltas = [
            s.get("answer_coverage_delta")
            for s in record["trace"]
            if isinstance(s.get("answer_coverage_delta"), (int, float))
        ]
        relation = (
            "win" if record["score"] > base["score"]
            else "regression" if record["score"] < base["score"]
            else "tie"
        )
        print(f"- case_key: {record['case_key']}")
        print(f"  分数关系: {relation} ({record['score']} vs baseline {base['score']}; delta={score_delta})")
        print(f"  问题: {record['question']}")
        print(f"  Baseline answer: {short_text(base['answer'], 120)}")
        print(f"  Baseline trace: {trace_shape(base['trace'])}")
        print(f"  Method answer: {short_text(record['answer'], 120)}")
        print(f"  Method trace: {validation['kinds']}")
        if coverage_deltas:
            print(f"  Coverage delta: {coverage_deltas}")
        print(f"  Trace 检查: {'通过' if validation['ok'] else '需解读'} - {validation['expected']}")
        print_trace(record["trace"])

    failed = [v for v in validations if not v["ok"]]
    if failed:
        raise AssertionError(f"{method_name} trace validation failed: {failed}")

    if method_name == "routing":
        expected_by_key = {
            "routing_general": "general_index",
            "routing_math": "math_index",
        }
        retrieve_by_key = {r["case_key"]: first_retrieve(r) for r in records}
        for case_key, expected_branch in expected_by_key.items():
            assert retrieve_by_key[case_key].get("branch") == expected_branch, retrieve_by_key
        shapes = {
            (
                s.get("n_hits"),
                s.get("context_chars"),
                s.get("first_hit"),
            )
            for s in retrieve_by_key.values()
        }
        assert len(shapes) >= 2, "routing branch traces must differ in n_hits/context_chars/first_hit"
    if method_name == "adaptive":
        delegate_by_key = {
            r["case_key"]: [
                s.get("target") or s.get("delegate")
                for s in r["trace"]
                if s.get("kind") == "delegate"
            ]
            for r in records
        }
        assert all(delegate_by_key.get(k) for k in ["adaptive_simple", "adaptive_moderate", "adaptive_complex"]), delegate_by_key
        delegates = {d for values in delegate_by_key.values() for d in values if d}
        assert len(delegates) >= 2, delegates
    if method_name == "self_rag":
        formal_benefit = any(
            r["score"] >= BASELINE_RECORDS_BY_QUESTION[r["question"]]["score"]
            or any(
                isinstance(s.get("answer_coverage_delta"), (int, float))
                and s.get("answer_coverage_delta") >= 0
                for s in r["trace"]
            )
            for r in records
        )
        assert formal_benefit, "self_rag needs at least one formal case with non-negative coverage delta or baseline benefit"
```

预期结果：panel 只使用 `METHOD_RECORDS` 和 `baseline_df` / method `*_df` 同一轮产生的 score；不会再次调用任何 pipeline，不会形成 panel-only trace。每个 case 同时展示 baseline answer 摘要、baseline trace 形状、method answer 摘要、method trace 形状、score delta 或 coverage delta。`records_for_cases` 若发现 targeted case 不在主评估 `qna_dict` 中，必须报错并先把该 case 纳入同一轮评估。`validate_demo_trace` 的失败不是展示文案，而是交付阻断；`routing` 的 case-key branch 覆盖和分支差异、`adaptive` 的 simple/moderate/complex delegate 覆盖、`self_rag` 至少一个正式收益 case，都由上面的聚合 `assert` fail-fast 保证，不满足就不能进入最终交付。

- [ ] **步骤 4：在 `compare_df` 后添加 targeted/case panel 小节**

Use this markdown:

```markdown
## 分方法真实差异面板

主表回答“整体趋势如何”；下面的 method case panel 回答“在方法适配题型上，机制是否真的触发，证据或答案覆盖是否改善”。两者来自同一轮 `METHOD_DFS`、`METHOD_RECORDS`、`baseline_df`、方法 `*_df` 和同一个维度计分裁判，只是展示层次不同。面板不得重新调用 pipeline 生成 trace。
```

Then use this code cell:

```python
display(flow_score_summary(compare_df))

display_flow_method_panel("Iterative", "iterative", ["iterative"])
display_flow_method_panel("Recursive", "recursive", ["recursive"])
display_flow_method_panel("Routing", "routing", ["routing_general", "routing_math"])
display_flow_method_panel("Adaptive", "adaptive", ["adaptive_simple", "adaptive_moderate", "adaptive_complex"])
display_flow_method_panel("CRAG", "crag", ["crag"])
display_flow_method_panel("Self-RAG", "self_rag", ["self_rag"])
```

预期结果：主评估和 targeted/case panel 同跑、同裁判、同 baseline。`routing` 面板展示 general 和 math；`adaptive` 面板展示 simple、moderate、complex；其它方法至少展示 2 个 targeted cases。

- [ ] **步骤 5：添加失败定位说明**

After the panel, add:

```markdown
如果某个方法在主表中没有提分，不要直接写成“方法无效”。先回到该方法的 trace 定位失败环节：router 是否选错分支、grader 是否过滤了有用证据、recursive 是否拆出重复子问题、Self-RAG 是否过早 `FINISH`、生成阶段是否没有吸收新增证据。只有 trace 能说明失败发生在哪里。
```

预期结果：得分弱的方法仍有基于 trace 的诊断路径。

- [ ] **步骤 6：验证 panel 符号**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

nb = json.loads(Path("notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb").read_text(encoding="utf-8"))
text = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
required = [
    "def validate_demo_trace(method_name: str, trace: list[dict], case_key: str | None = None) -> dict",
    "def flow_score_summary(compare_df)",
    "METHOD_RECORDS: dict[str, list[dict]] = {}",
    "BASELINE_RECORDS_BY_QUESTION = {",
    "def records_for_cases(method_name: str, case_keys: list[str]) -> list[dict]",
    "def display_flow_method_panel(method_label: str, method_name: str, case_keys: list[str])",
    "Baseline answer:",
    "Baseline trace:",
    "Method answer:",
    "Method trace:",
    "delta=",
    "routing branch traces must differ",
    "self_rag needs at least one formal case",
    "## 分方法真实差异面板",
    "同一轮 `METHOD_DFS`",
    "面板不得重新调用 pipeline 生成 trace",
    "同一个维度计分裁判",
    "定位失败环节",
    'display_flow_method_panel("Routing", "routing", ["routing_general", "routing_math"])',
    'display_flow_method_panel("Adaptive", "adaptive", ["adaptive_simple", "adaptive_moderate", "adaptive_complex"])',
]
missing = [s for s in required if s not in text]
print("missing:", missing)
raise SystemExit(1 if missing else 0)
PY
```

预期结果：`missing: []`。

### 任务 8：重写动态总结、结果解读和评估限制

**文件：**
- 检查：`notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`
- 修改：`notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`
- 新建：无
- 验证：下面的 stale static result scan command

- [ ] **步骤 1：用动态总结替换静态结果断言**

Use:

```markdown
## 实测结果与解读

下面的汇总来自当前 notebook 的 `compare_df`，因此会随模型输出、限流重试、题集调整和随机波动略有变化。读这张表时不要只看均值，更要结合上面的分方法 trace：流程增强的教学重点是“哪一步做了不同决策”。
```

Then:

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

预期结果：summary 数字始终来自当前运行。

- [ ] **步骤 2：添加按方法解读结果的规则**

Use:

```markdown
### 如何解读本轮结果

1. 如果 `iterative` 提分，优先查看它的 trace 是否真的发生第二轮检索；如果没有第二轮却提分，收益可能来自生成随机性，而不是补检索。
2. 如果 `recursive` 在复杂题上提分，查看 `decompose` 是否拆出了互补子问题；如果在简单题回退，通常是过度拆解造成的成本。
3. 如果 `routing` 与 baseline 接近，先看两个 branch 的 `first_hit`、`n_hits`、`context_chars` 是否不同；如果 branch 不同但得分相同，说明路由改变了上下文但未改变裁判分数。
4. 如果 `adaptive` 表现不稳定，查看 `classify` 与 `delegate` 是否把题目送进了合适流程；分类错会把后续流程全部带偏。
5. 如果 `crag` 提分，查看被过滤的 `partial` / `irrelevant` 证据；如果回退，可能是 grader 把有用片段降权或过滤了。
6. 如果 `self_rag` 很少触发继续检索，说明教学化简版的通用 LLM 自评偏自信；需要用 trace 说明它是有效早停还是过早停止。
```

预期结果：结果解读以 trace 为先，而不是以排名为先。

- [ ] **步骤 3：添加明确的评估限制**

Use:

```markdown
### 评估限制

这组实验只服务于 6.2 的教学目标：帮助读者看懂流程决策怎样改变 RAG 行为。它不是 SOTA 排名，也不代表开放域泛化结论。限制包括：样本量小、题库来自同一教材、裁判是 LLM-as-judge、不同模型版本可能给出不同分数、部分方法的收益高度依赖题型是否匹配。
```

预期结果：最终叙述不过度声明。

- [ ] **步骤 4：移除过时的静态数字断言**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

nb = json.loads(Path("notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb").read_text(encoding="utf-8"))
text = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
stale = [
    "iterative（最佳）",
    "routing 和 baseline 分数完全一致",
    "一定最好",
    "SOTA",
    "排行榜",
]
hits = [s for s in stale if s in text]
print("static_or_overclaim_hits:", hits)
raise SystemExit(1 if any(s != "排行榜" for s in hits) else 0)
PY
```

预期结果：不再保留过时的固定结果断言。`排行榜` 这个词只能出现在“不是排行榜”等否定句中。

### 任务 9：仅在边界或命名冲突时对齐选型总结

**文件：**
- 检查：`notebook/C7 高级 RAG 技巧/6. 增强阶段/4. 选型总结.md`
- 检查：`notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`
- 修改：仅当现有措辞与最终 6.2 边界或方法命名冲突时，才最小修改 `notebook/C7 高级 RAG 技巧/6. 增强阶段/4. 选型总结.md`
- 新建：无
- 验证：下面的 summary boundary scan command

- [ ] **步骤 1：检查方法命名一致性**

Compare the summary table against the notebook’s six method names:

```text
迭代检索 / Iterative Retrieval
递归分解 / Recursive Decomposition
查询路由 / Query Routing
自适应检索 / Adaptive Retrieval
Corrective RAG / CRAG
Self-RAG
```

预期结果：summary 不以和 notebook 标题冲突的方式重命名 6.2 方法。

- [ ] **步骤 2：仅在存在过时 routing 措辞时修补**

If `4. 选型总结.md` describes 查询路由 as “多数据源或多索引 / 工具命中率” without distinguishing 6.2 from 6.3, minimally rewrite that row to this stricter boundary. The row must contain both `单次请求内` and `6.3`, and it must not keep an unqualified `工具命中率` as the 6.2 routing metric:

```markdown
| 查询路由 | 单次请求内选择检索策略或索引配置；生产系统中可扩展到多源路由 | 路由准确率、分支命中质量 | 中 | 低-中 | 6.2 先学检索策略分支，6.3 再扩展到系统级路由 |
```

预期结果：summary 保留生产视角，但不会让 6.2 看起来像系统级 routing。`工具命中率` 只有在明确限定为 6.3 或系统级多工具路由时才可出现，不能作为 6.2 查询路由的未限定指标。

- [ ] **步骤 3：同步旧称为“递归分解 / Recursive Decomposition”**

If `4. 选型总结.md` still uses `递归检索` for the 6.2 method name, rename that label to `递归分解 / Recursive Decomposition` while preserving the existing meaning and table structure.

预期结果：选型总结与 notebook 方法标题一致，不再混用旧称。

- [ ] **步骤 4：仅在总结仍模糊时添加一条边界说明**

If needed, add this note below the flow-enhancement part of the summary:

```markdown
> 6.2 的流程增强默认限定在单次请求内；涉及跨请求记忆、多工具、多文档 agent 的编排，放到 6.3 系统增强中讨论。
```

预期结果：只做最小修改，不进行无关的 summary 重写。

- [ ] **步骤 5：如已编辑则验证总结边界**

Run:

```bash
python - <<'PY'
from pathlib import Path

p = Path("notebook/C7 高级 RAG 技巧/6. 增强阶段/4. 选型总结.md")
text = p.read_text(encoding="utf-8")
if "查询路由" in text:
    assert "单次请求内" in text and "6.3" in text
    if "工具命中率" in text:
        pos = text.index("工具命中率")
        window = text[max(0, pos - 80):pos + 80]
        assert "6.3" in window, window
assert "递归检索" not in text or "递归分解" in text
print("selection summary checked")
PY
```

预期结果：命令打印 `selection summary checked`。

### 任务 10：Notebook 结构验证与最终复核

**文件：**
- 检查：`notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`
- 检查：`notebook/C7 高级 RAG 技巧/6. 增强阶段/4. 选型总结.md`
- 检查：`notebook/C7 高级 RAG 技巧/6. 增强阶段/_common.py`
- 修改：无；除非验证失败明确指向任务 2-9 的实现错误
- 新建：无
- 验证：JSON validation、text invariant checks、可选 notebook smoke test 和 `git diff -- ...`

- [ ] **步骤 1：验证 notebook JSON 结构**

Run:

```bash
python - <<'PY'
import nbformat
from pathlib import Path

p = Path("notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb")
nb = nbformat.read(p, as_version=4)
nbformat.validate(nb)
print(f"valid notebook: {len(nb.cells)} cells")
PY
```

预期结果：notebook 通过 nbformat v4 验证。

- [ ] **步骤 2：运行不变量检查**

Run:

```bash
python - <<'PY'
import json, re
from pathlib import Path

p = Path("notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))
text = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
checks = {
    "six_methods": len(re.findall(r"^## 方法 \d：", text, flags=re.M)) == 6,
    "has_baseline": "def baseline_pipeline(question: str) -> tuple[str, list[dict]]" in text,
    "has_demo_cases": "DEMO_CASES = {" in text,
    "has_trace_validation": "def validate_demo_trace(method_name: str, trace: list[dict], case_key: str | None = None) -> dict" in text,
    "has_strict_validation_fields": (
        "answer_coverage_delta" in text
        and "sub_question_diversity_ok" in text
        and "before_context_chars" in text
        and "routing_general" in text
        and "routing branch traces must differ" in text
        and "trace validation failed" in text
    ),
    "has_same_run_panel": (
        "## 分方法真实差异面板" in text
        and "METHOD_RECORDS" in text
        and "面板不得重新调用 pipeline 生成 trace" in text
        and "同一个维度计分裁判" in text
        and "交付阻断" in text
    ),
    "has_dynamic_summary": "summary_df = flow_score_summary(compare_df)" in text,
    "has_limits": "不是 SOTA 排名" in text and "LLM-as-judge" in text,
    "routing_boundary": "不是多文档 agent" in text and "不是工具" in text and "跨请求" in text,
}
for name, ok in checks.items():
    print(f"{name}: {ok}")
failed = [name for name, ok in checks.items() if not ok]
raise SystemExit(1 if failed else 0)
PY
```

预期结果：所有检查都打印 `True`。

- [ ] **步骤 3：运行有限 notebook smoke test**

In Jupyter or VS Code, restart kernel and run through:

```text
1. imports and common setup
2. retriever build/load
3. QA index cell
4. DEMO_CASES cell
5. baseline inspect
6. one inspect cell per method
7. strict validate_demo_trace cell before mining
8. final DEMO_CASES / targeted cases cell
9. full evaluation cells only after DEMO_CASES and validation are final, when API quota and time allow
```

预期结果：所有 inspect 单元都产生 answer 和 trace。若因 API 配额或时间跳过完整评估，保留结构检查并说明未运行完整 LLM evaluation。

- [ ] **步骤 4：确认每个方法都有教学 case**

Use the trace outputs to fill this review table in notebook markdown or a nearby final review cell:

```markdown
| 方法 | 代表题来源 | 核心 trace 信号 | 相对 baseline 的教学差异 |
|---|---|---|---|
| Iterative | 数据集或自然课程题 | 第二轮检索或明确停止原因 | 展示草稿缺口如何驱动补检索 |
| Recursive | 数据集题 | decompose + 多个 sub_answer | 展示复杂问题如何拆成互补子任务 |
| Routing | 数据集题 | general/math 两类 case + 不同 branch 的 n_hits/context_chars/first_hit | 展示检索配置选择如何改变证据 |
| Adaptive | 数据集题 | simple/moderate/complex 三类 case + 至少两个 delegate | 展示问题复杂度如何决定流程深度 |
| CRAG | 数据集题，extra 仅作补充 inspect | grade + partial/irrelevant + before/after context chars、dropped_count 或 context hash | 展示证据过滤如何改变上下文 |
| Self-RAG | 数据集题 | reflect verdict + FINISH/CONTINUE 的真实检查效果 | 展示自评如何关联答案覆盖、早停充分性或失败定位 |
```

预期结果：没有任何一行只写“未触发”或只写“无提升”。

- [ ] **步骤 5：检查 diff 范围**

Run:

```bash
git diff -- "notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb" \
           "notebook/C7 高级 RAG 技巧/6. 增强阶段/4. 选型总结.md" \
           "notebook/C7 高级 RAG 技巧/6. 增强阶段/_common.py"
```

预期结果：diff 只包含限定范围内的 notebook 教学修改，以及必要时的最小 selection-summary 措辞修改。`_common.py` 正常情况下没有 diff。

---

## 自审

**Spec 覆盖：** 本实现计划以 `docs/superpowers/specs/2026-05-23-c7-6-2-flow-enhancement-design.md` 为准。它保留恰好六个 runnable 方法，强化章节边界和 query-routing 边界，用同一轮 `METHOD_DFS` / `METHOD_RECORDS` 统一 baseline、主评估、method df 与 targeted panel，要求每个方法默认提供 2~3 个真实可运行 targeted cases，避免 SOTA 式声明，并从当前 `compare_df` 派生最终总结数字。

**旧计划吸收：** 旧计划只作为 baseline pipeline、demo/case mining、双 retriever routing、trace validation、动态总结、主评估表和 method-level panel 的素材来源。commit 步骤、过时静态结果，以及任何与最新 spec 冲突的 routing 语言都要移除或重写。

**开放标记扫描：** 计划包含具体任务、文件范围、命令、预期输出、markdown 片段和代码片段。它不依赖未解决标记或符号化 notebook 值；即使 mining 可以改进 case 选择，计划仍提供具体初始 `DEMO_CASES` 字典，并要求最终值都是真实整数索引。

**类型与顺序一致性：** 所有 runnable 方法，包括 baseline，都使用 `pipeline(question: str) -> tuple[str, list[dict]]`。`validate_demo_trace(method_name, trace, case_key=None)` 必须在 case mining 之前定义并用于 mining，避免先按宽松标准选 case。`eval_pipeline_with_trace` 适配 pipeline 签名到 `run_shared_eval`，并返回同跑的 `(df, records)`；`METHOD_RECORDS` 中每条记录都包含 `answer`、`trace`、`score`、`expected`。`compare_df` 列为 `question`、`baseline`、`iterative`、`recursive`、`routing`、`adaptive`、`crag` 和 `self_rag`。`DEMO_CASES` 将 case 名映射到整数数据集索引列表，正式 targeted cases 必须来自 `DEMO_CASES` 并进入同一轮 `METHOD_RECORDS` / `compare_df`；`EXTRA_DEMO_CASES` 只用于补充 inspect 或边界演示，若升级为正式 case，必须进入同一轮缓存体系并有 baseline 对照、score、trace、expected/评分说明。

**Validation 阻断：** `validate_demo_trace` 不是打印 trace 的辅助说明，而是进入最终交付前的失败条件。`iterative` 必须把缺口/补检索原因纳入 `ok`，`recursive` 必须检查子问题去重或 `sub_question_diversity_ok`，`routing` 必须按 `routing_general -> general_index`、`routing_math -> math_index` 校验，`CRAG` 必须用 before/after context 字段或 hash 证明上下文变化，`self_rag` 必须区分正式有效 case 与失败定位 case；`routing` 分支差异、`adaptive` case-key delegate 覆盖和 `self_rag` 至少一个正式收益 case 由 panel 聚合 `assert` fail-fast 阻断。method panel 必须同时展示 baseline answer 摘要、baseline trace 形状、method answer 摘要、method trace 形状、score delta 或 coverage delta。

**Review 反馈覆盖：** 本计划已覆盖 9 项 Important review：`EXTRA_DEMO_CASES` 不替代正式 case；严格 validation 前置到 mining；setup/import 明确 `json` 与 `Path` 且保留 `build_retriever` 依赖；routing/adaptive 聚合检查按 case key fail-fast；CRAG 比较真实上下文变化；Self-RAG 区分有效 case 与失败定位 case；`4. 选型总结.md` 的 routing 边界、`工具命中率` 和递归命名被收紧；method panel 展示相对 baseline 证据；full evaluation 只在 `DEMO_CASES` 和 validation 最终确定后运行。
