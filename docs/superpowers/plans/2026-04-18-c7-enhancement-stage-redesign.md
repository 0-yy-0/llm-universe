# C7「增强阶段」三节统一重构与系统增强补全 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `notebook/C7 高级 RAG 技巧/6. 增强阶段/` 三个 notebook 重构为「共享 `_common.py` 公共底座 + 每节钩子落在本节核心变化点 + 教学优先的 inline 完整定义」的形态，并从零写完 6.3 节的 Memory 与 Multi-Document Agent 两个方法。

**Architecture:**
- 抽出 `_common.py` 作为三节的 single source of truth（PDF/embedding/Chroma/`llm_call`/评估接口）
- 6.1 钩子 `*_context(q) -> str`、6.2 钩子 `*_pipeline(q) -> str`、6.3 钩子 `class XxxSystem: ask(q) -> str`
- 教学暴露规则：函数首次出现的节里 inline 完整 `def`，后续节 `from _common import` 复用

**Tech Stack:** Python 3.10、Jupyter notebook、`langchain-chroma`、`langchain-text-splitters`、`langchain-community`（HuggingFaceEmbeddings）、`langchain-core`、`zhipuai`、`pymupdf` (`fitz`)、`modelscope`、`pandas`、`pytest`（仅用于 `_common.py` 的轻量 smoke test）

**前置约定：**
- 所有 shell 命令都假定 cwd = `notebook/C7 高级 RAG 技巧/6. 增强阶段/`，除非另注
- 所有 `_common.py` 中函数的字符串字面量、参数名、默认值、错误消息必须与"首次定义节"的 notebook cell 完全一致（single source of truth）
- 每个 task 完成后单独 commit，commit 信息按 conventional commits 风格

**Spec：** `docs/superpowers/specs/2026-04-18-c7-enhancement-stage-redesign.md`

---

## Task 0：准备工作与 .gitignore

**Files:**
- Modify: `.gitignore`
- Create: `notebook/C7 高级 RAG 技巧/6. 增强阶段/data/`（目录）
- Create: `notebook/C7 高级 RAG 技巧/6. 增强阶段/tests/`（目录）

- [ ] **Step 1：检查现状**

```bash
cd "notebook/C7 高级 RAG 技巧/6. 增强阶段"
ls -la
```

Expected: 看到现有 `1. 上下文增强.ipynb`、`2. 流程增强.ipynb`、`3. 系统增强.ipynb`、`readme.md`、`chroma_db/`、`models/`，以及 git status 中的散落 CSV/JSON 评估产物。

- [ ] **Step 2：补 `.gitignore`**

把以下条目追加到仓库根 `.gitignore`（如已有则跳过）：

```
# C7 6.增强阶段 运行产物
notebook/C7 高级 RAG 技巧/6. 增强阶段/chroma_db/
notebook/C7 高级 RAG 技巧/6. 增强阶段/models/
notebook/C7 高级 RAG 技巧/6. 增强阶段/context_enhance_compare_*.csv
notebook/C7 高级 RAG 技巧/6. 增强阶段/context_enhance_compare_*.json
notebook/C7 高级 RAG 技巧/6. 增强阶段/context_enhance_eval_summary.json
notebook/C7 高级 RAG 技巧/6. 增强阶段/eval*.txt
notebook/C7 高级 RAG 技巧/6. 增强阶段/eval*.json
notebook/C7 高级 RAG 技巧/6. 增强阶段/.DS_Store
```

- [ ] **Step 3：从 git 中移除已 stage 的运行产物（如有）**

```bash
git rm -r --cached "notebook/C7 高级 RAG 技巧/6. 增强阶段/chroma_db" 2>/dev/null || true
git rm -r --cached "notebook/C7 高级 RAG 技巧/6. 增强阶段/models" 2>/dev/null || true
```

Expected: 输出已移除的文件列表；如未跟踪则忽略。

- [ ] **Step 4：创建必要的目录占位**

```bash
mkdir -p "notebook/C7 高级 RAG 技巧/6. 增强阶段/data"
mkdir -p "notebook/C7 高级 RAG 技巧/6. 增强阶段/tests"
touch "notebook/C7 高级 RAG 技巧/6. 增强阶段/tests/.gitkeep"
```

- [ ] **Step 5：commit**

```bash
git add .gitignore "notebook/C7 高级 RAG 技巧/6. 增强阶段/tests/.gitkeep"
git commit -m "chore(c7): gitignore enhancement-stage runtime artifacts"
```

---

## Task 1：`_common.py` 基础工具层（无评估部分）

**Files:**
- Create: `notebook/C7 高级 RAG 技巧/6. 增强阶段/_common.py`
- Create: `notebook/C7 高级 RAG 技巧/6. 增强阶段/tests/test_common_basic.py`

**目标：** 实现 PDF / embedding / Chroma / `llm_call` / prompt 模板 / `trim_context_to_budget` / `load_qna_subset` 七组基础工具，并加 smoke test。

- [ ] **Step 1：写 smoke test 文件（先失败）**

创建 `notebook/C7 高级 RAG 技巧/6. 增强阶段/tests/test_common_basic.py`：

```python
"""
_common.py 基础工具的 smoke test。
不调用 LLM、不下载 embedding 模型，只验证函数签名、可 import、纯函数行为。
运行：cd "notebook/C7 高级 RAG 技巧/6. 增强阶段" && python -m pytest tests/test_common_basic.py -v
"""
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(THIS_DIR))


def test_import_common():
    import _common  # noqa


def test_trim_context_to_budget_no_truncate():
    from _common import trim_context_to_budget
    text = "abc"
    assert trim_context_to_budget(text, 10) == "abc"


def test_trim_context_to_budget_truncates_with_marker():
    from _common import trim_context_to_budget
    text = "句子一。" * 200
    out = trim_context_to_budget(text, 50)
    assert "[...下文已按 CONTEXT_CHAR_BUDGET 截断...]" in out
    assert len(out) <= 50 + len("\n\n[...下文已按 CONTEXT_CHAR_BUDGET 截断...]") + 5


def test_build_rag_generation_prompt_contains_question_and_context():
    from _common import build_rag_generation_prompt
    p = build_rag_generation_prompt("Q?", "CTX")
    assert "Q?" in p and "CTX" in p


def test_load_qna_subset_returns_dict_in_order(tmp_path):
    import json
    from _common import load_qna_subset
    qa_path = tmp_path / "qa.json"
    qa_path.write_text(
        json.dumps([
            {"query": "q0", "answer": "a0"},
            {"query": "q1", "answer": "a1"},
            {"query": "q2", "answer": "a2"},
        ]),
        encoding="utf-8",
    )
    out = load_qna_subset(str(qa_path), [0, 2])
    assert list(out.keys()) == ["q0", "q2"]
    assert out["q2"] == "a2"
```

- [ ] **Step 2：跑 smoke test 看到失败**

```bash
cd "notebook/C7 高级 RAG 技巧/6. 增强阶段"
python -m pytest tests/test_common_basic.py -v
```

Expected: 全部失败，错误信息为 `ModuleNotFoundError: No module named '_common'`。

- [ ] **Step 3：创建 `_common.py` 基础工具部分**

创建 `notebook/C7 高级 RAG 技巧/6. 增强阶段/_common.py`：

```python
"""
6. 增强阶段 三节共用的公共底座（single source of truth）。

各 notebook 中"首次出现"的核心新概念（评估、钩子胶水）必须先在 notebook 内
完整 inline 重写一次，再 from _common import 复用；本文件保存与 notebook
首次定义字符相同的副本。基础工具（embedding、PDF、Chroma 等）前几章已讲，
notebook 中不重复展示，直接 import。
"""
from __future__ import annotations

import json
import os
import re
import time
import warnings
from pathlib import Path
from typing import Callable, Iterable

import fitz  # PyMuPDF
import pandas as pd
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from modelscope import snapshot_download

warnings.filterwarnings("ignore")
load_dotenv()


# ---------- 路径与默认参数 ----------

EMBED_MODEL_ID = "BAAI/bge-small-zh-v1.5"
EMBED_MODEL_PATH = f"./models/{EMBED_MODEL_ID}"

PDF_PATH = "../3. 索引阶段/data/pumpkin_book.pdf"
QA_PATH = "../3. 索引阶段/data/train_dataset.json"

CHROMA_COLLECTION = "nb_ctx"
CONTEXT_CHAR_BUDGET = int(os.environ.get("CONTEXT_CHAR_BUDGET", "1150"))

LLM_MODEL = "glm-4-flash-250414"
LLM_MAX_RETRIES = 10


# ---------- embedding（单例） ----------

_pdf_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """本地 bge-small-zh-v1.5 单例，多个 Chroma 索引共用。"""
    global _pdf_embeddings
    if _pdf_embeddings is None:
        path = EMBED_MODEL_PATH
        if not os.path.exists(path):
            cache_dir = Path("./models")
            cache_dir.mkdir(parents=True, exist_ok=True)
            path = snapshot_download(EMBED_MODEL_ID, cache_dir=str(cache_dir))
        _pdf_embeddings = HuggingFaceEmbeddings(
            model_name=path,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _pdf_embeddings


# ---------- PDF 清洗与按页缓存 ----------

def clean_text(text: str) -> str:
    text = re.sub(r"→_→\n欢迎去各大电商平台选购纸质版南瓜书《机器学习公式详解》\n←_←", "", text)
    text = re.sub(r"→_→\n配套视频教程：https://www.bilibili.com/video/BV1Mh411e7VU\n←_←", "", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"\n+", "", text)
    return text


_CLEANED_PDF_DOCS: list[Document] | None = None


def get_cleaned_pdf_documents() -> list[Document]:
    """加载 + 清洗 PDF 一次，三节共用。返回按页 Document 列表。"""
    global _CLEANED_PDF_DOCS
    if _CLEANED_PDF_DOCS is None:
        doc = fitz.open(PDF_PATH)
        pages = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            pages.append(Document(page_content=clean_text(page.get_text()), metadata={"page": i}))
        doc.close()
        _CLEANED_PDF_DOCS = pages
    return _CLEANED_PDF_DOCS


# ---------- Chroma 工厂 ----------

def open_or_build_chroma(persist_directory: str, documents: list[Document], ids: list[str]) -> Chroma:
    """同一 persist 目录复用，已存在集合则跳过 add。"""
    os.makedirs(persist_directory, exist_ok=True)
    emb = get_embeddings()
    vs = Chroma(
        persist_directory=persist_directory,
        embedding_function=emb,
        collection_name=CHROMA_COLLECTION,
        collection_metadata={"hnsw:space": "cosine"},
    )
    if vs._collection.count() > 0:
        print(f"  -> 加载已有索引: {persist_directory}")
        return vs
    print(f"  -> 创建新索引: {persist_directory}")
    vs.add_documents(documents, ids=ids)
    return vs


# ---------- 智谱 LLM 调用（429 退避，可选每次成功后 sleep） ----------

_zhipu_ai_client = None


def llm_call(prompt: str, *, sleep_after: float = 0.0) -> str:
    """智谱 ZhipuAI 调用。429 时按 min(180, 20*n) 秒退避，最多 LLM_MAX_RETRIES 次。
    sleep_after > 0 时在每次成功调用后追加 sleep（用于 6.2 / 6.3 节对抗连续调用限流）。
    """
    from zhipuai import ZhipuAI

    global _zhipu_ai_client
    if _zhipu_ai_client is None:
        _zhipu_ai_client = ZhipuAI(api_key=os.environ.get("ZHIPUAI_API_KEY"))
    client = _zhipu_ai_client
    for attempt in range(LLM_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            if sleep_after > 0:
                time.sleep(sleep_after)
            return resp.choices[0].message.content
        except Exception as e:
            if "429" not in str(e) or attempt == LLM_MAX_RETRIES - 1:
                raise
            wait = min(180, 20 * (attempt + 1))
            print(f"  [速率限制，等待 {wait}s...]")
            time.sleep(wait)
    raise RuntimeError("llm_call exhausted retries without raising")


# ---------- 通用 prompt 与文本工具 ----------

def build_rag_generation_prompt(question: str, context: str) -> str:
    return (
        "仅根据上下文回答问题，如果上下文没有包含完整答案，请仅回答上下文中的内容，不要补充你自己的知识。\n\n"
        f"问题：{question}\n上下文：\n{context}"
    )


def trim_context_to_budget(text: str, budget: int) -> str:
    text = (text or "").strip()
    if len(text) <= budget:
        return text
    head = text[:budget]
    cut = max(head.rfind("\n\n"), head.rfind("。"), head.rfind("；"))
    if cut > int(budget * 0.55):
        head = head[: cut + 1]
    return head.strip() + "\n\n[...下文已按 CONTEXT_CHAR_BUDGET 截断...]"


# ---------- 题集加载 ----------

def load_qna_subset(qa_path: str, indices: Iterable[int]) -> dict[str, str]:
    """从 train_dataset.json 加载指定下标的 (query -> answer) 字典，保持 indices 顺序。"""
    with open(qa_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    out: dict[str, str] = {}
    for i in indices:
        item = pairs[i]
        q, a = item["query"], item["answer"]
        if q and str(q).strip():
            out[q] = a
    return out


# ---------- 评估与运行器（占位，Task 2/3 中实现） ----------
# simple_eval_2pt / answer_from_context_fn / run_shared_eval / build_compare_table → Task 2
# run_session_eval → Task 3
```

- [ ] **Step 4：跑 smoke test 看到通过**

```bash
cd "notebook/C7 高级 RAG 技巧/6. 增强阶段"
python -m pytest tests/test_common_basic.py -v
```

Expected: 5 个 test 全部 PASS。

- [ ] **Step 5：commit**

```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/_common.py" \
        "notebook/C7 高级 RAG 技巧/6. 增强阶段/tests/test_common_basic.py"
git commit -m "feat(c7): add _common.py basic utilities (PDF/embed/chroma/llm_call)"
```

---

## Task 2：`_common.py` 评估接口（6.1 节首次出现的概念）

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/_common.py`
- Create: `notebook/C7 高级 RAG 技巧/6. 增强阶段/tests/test_common_eval.py`

**目标：** 实现 `simple_eval_2pt`、`answer_from_context_fn`、`run_shared_eval`、`build_compare_table` 四个评估接口，并加 smoke test（不真正调 LLM，用 monkeypatch 替身）。

- [ ] **Step 1：写 smoke test**

创建 `tests/test_common_eval.py`：

```python
"""
_common.py 评估接口的 smoke test：用 monkeypatch 替身 llm_call，
验证 0~2 分 parse、对比表拼接、answer_from_context_fn 胶水正确性。
"""
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(THIS_DIR))


def test_parse_eval_score_clean():
    from _common import _parse_eval_score
    assert _parse_eval_score("2") == 2
    assert _parse_eval_score(" 1 \n") == 1
    assert _parse_eval_score("0") == 0


def test_parse_eval_score_dirty():
    from _common import _parse_eval_score
    assert _parse_eval_score("评分：2\n理由：...") == 2
    assert _parse_eval_score("- 1") == 1
    assert _parse_eval_score("garbage with no digit") == 0


def test_simple_eval_2pt_with_monkeypatched_llm(monkeypatch):
    import _common
    monkeypatch.setattr(_common, "llm_call", lambda prompt, **kw: "2")
    assert _common.simple_eval_2pt("ans", "exp", "q") == 2


def test_simple_eval_2pt_uses_custom_template(monkeypatch):
    import _common
    captured = {}

    def fake_llm(prompt, **kw):
        captured["prompt"] = prompt
        return "1"

    monkeypatch.setattr(_common, "llm_call", fake_llm)
    out = _common.simple_eval_2pt(
        "a", "e", "q",
        prompt_template="CUSTOM:{question}|{expected_answer}|{llm_answer}",
    )
    assert out == 1
    assert captured["prompt"] == "CUSTOM:q|e|a"


def test_answer_from_context_fn_pipeline(monkeypatch):
    import _common
    monkeypatch.setattr(_common, "llm_call", lambda prompt, **kw: f"ANS<{prompt[:5]}>")
    fn = _common.answer_from_context_fn(lambda q: f"CTX-of-{q}")
    out = fn("hello")
    assert out.startswith("ANS<")
    assert "hello" not in out  # 只取 prompt 前 5 字符做 marker；对实际生成无要求


def test_run_shared_eval_returns_df(monkeypatch):
    import _common
    monkeypatch.setattr(_common, "llm_call", lambda prompt, **kw: "2")
    qna = {"q1": "a1", "q2": "a2"}
    df = _common.run_shared_eval(lambda q: f"answer-of-{q}", qna)
    assert list(df.columns) == ["question", "llm_answer", "expected_answer", "rag_eval_results"]
    assert len(df) == 2
    assert df["rag_eval_results"].tolist() == [2, 2]


def test_build_compare_table_aligns_on_question():
    import pandas as pd
    from _common import build_compare_table
    df_a = pd.DataFrame({"question": ["q1", "q2"], "rag_eval_results": [2, 0]})
    df_b = pd.DataFrame({"question": ["q1", "q2"], "rag_eval_results": [1, 2]})
    out = build_compare_table([df_a, df_b], names=["A", "B"])
    assert list(out.columns) == ["question", "A", "B"]
    assert out.loc[out["question"] == "q1", "A"].iloc[0] == 2
    assert out.loc[out["question"] == "q2", "B"].iloc[0] == 2
```

- [ ] **Step 2：跑 test 看到失败**

```bash
cd "notebook/C7 高级 RAG 技巧/6. 增强阶段"
python -m pytest tests/test_common_eval.py -v
```

Expected: 全部失败（`AttributeError` 或 `ImportError`，因为 `_common` 还没定义这些函数）。

- [ ] **Step 3：在 `_common.py` 末尾追加评估接口实现**

把以下代码追加到 `_common.py`（替换"占位"那段注释）：

```python
# ---------- 评估接口（6.1 节首次出现，notebook 中 inline 完整重写一次） ----------

DEFAULT_EVAL_PROMPT_2PT = (
    "请作为一名判卷人，按 0～2 分评判「模型答案」回答「用户问题」的质量，并对照「参考答案」核对事实与要点。\n"
    "你只收到下列三段文字，没有 RAG 检索原文；请勿以「未逐字引用原文」为由扣分，除非答案与参考答案明显矛盾。\n"
    "评分标准：\n"
    "2 分：核心结论正确，问题所问的必答要点基本齐全，无明显事实错误；表述可比参考答案更短。\n"
    "1 分：方向基本正确，但明显缺少部分关键点，或解释不完整、表述含糊；无严重编造。\n"
    "0 分：关键结论错误、严重漏答问题核心、或与参考答案明显矛盾、或凭空补充。\n"
    "参考答案中的引导语、举例、排版说明不必复述；勿因未覆盖参考答案中的次要枝节就将 2 分打成 1 分。\n\n"
    "用户问题：{question}\n"
    "参考答案：{expected_answer}\n"
    "模型答案：{llm_answer}\n\n"
    "请仅输出一行，且该行只包含一个字符：0、1 或 2，不要输出任何其它文字。"
)


def _parse_eval_score(raw: str) -> int:
    text = (raw or "").strip()
    for line in text.splitlines():
        s = line.strip()
        s = re.sub(r"^[-*•\d.)]+\s*", "", s)
        if re.fullmatch(r"[012]", s):
            return int(s)
    m = re.search(r"(?<![0-9])([012])(?![0-9])", text)
    if m:
        return int(m.group(1))
    return 0


def simple_eval_2pt(
    llm_answer: str,
    expected_answer: str,
    question: str = "",
    *,
    prompt_template: str | None = None,
) -> int:
    """0~2 分 LLM 裁判。三节都用同一接口；prompt_template 允许各节传入定制模板。
    模板必须包含 {question} / {expected_answer} / {llm_answer} 三个占位符。
    """
    template = prompt_template or DEFAULT_EVAL_PROMPT_2PT
    prompt = template.format(
        question=question,
        expected_answer=expected_answer,
        llm_answer=llm_answer,
    )
    try:
        return _parse_eval_score(llm_call(prompt))
    except Exception as e:
        print(f"评估失败: {e}")
        return 0


def answer_from_context_fn(build_context_fn: Callable[[str], str]) -> Callable[[str], str]:
    """6.1 专用胶水：把 *_context(q)->str 的钩子接到 LLM 上。
    6.2 / 6.3 不需要这个胶水（pipeline / system.ask 已经直接产出答案）。
    """
    def _answer(question: str) -> str:
        context = build_context_fn(question)
        prompt = build_rag_generation_prompt(question, context)
        return llm_call(prompt)
    return _answer


def run_shared_eval(
    answer_fn: Callable[[str], str],
    qna_dict: dict[str, str],
    *,
    eval_prompt_template: str | None = None,
) -> pd.DataFrame:
    """通用评测流水线：逐题调 answer_fn，再用 simple_eval_2pt 打分，返回 DataFrame。
    6.1 / 6.2 都用本函数；6.3 用 run_session_eval。
    eval_prompt_template 用于 6.2 维度计分等需要定制 prompt 的场景。
    """
    rows = []
    for question, expected in qna_dict.items():
        answer = answer_fn(question)
        score = simple_eval_2pt(
            answer, expected, question,
            prompt_template=eval_prompt_template,
        )
        rows.append({
            "question": question,
            "llm_answer": answer,
            "expected_answer": expected,
            "rag_eval_results": score,
        })
    return pd.DataFrame(rows)


def build_compare_table(dfs: list[pd.DataFrame], names: list[str]) -> pd.DataFrame:
    """把多个 run_shared_eval 的结果按 question 列对齐拼接为同题对比表。"""
    if len(dfs) != len(names):
        raise ValueError("dfs 与 names 长度必须一致")
    base = dfs[0][["question"]].copy()
    base[names[0]] = dfs[0]["rag_eval_results"].values
    out = base
    for df, name in zip(dfs[1:], names[1:]):
        out = out.merge(
            df[["question", "rag_eval_results"]].rename(columns={"rag_eval_results": name}),
            on="question",
            how="outer",
        )
    return out.reset_index(drop=True)
```

- [ ] **Step 4：跑 test 看到通过**

```bash
cd "notebook/C7 高级 RAG 技巧/6. 增强阶段"
python -m pytest tests/test_common_eval.py -v
```

Expected: 7 个 test 全部 PASS。

- [ ] **Step 5：commit**

```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/_common.py" \
        "notebook/C7 高级 RAG 技巧/6. 增强阶段/tests/test_common_eval.py"
git commit -m "feat(c7): add 0~2pt judge, run_shared_eval, build_compare_table to _common"
```

---

## Task 3：`_common.py` 系统评估 `run_session_eval`（6.3 节首次出现）

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/_common.py`
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/tests/test_common_eval.py`

**目标：** 实现 `run_session_eval(system, qna_dict)`：顺次调用 `system.ask(q)` 保留状态，输出与 `run_shared_eval` 同结构的 DataFrame。

- [ ] **Step 1：在 test 文件追加 case**

把以下 test 追加到 `tests/test_common_eval.py` 末尾：

```python
def test_run_session_eval_preserves_state(monkeypatch):
    import _common
    monkeypatch.setattr(_common, "llm_call", lambda prompt, **kw: "2")

    class FakeSystem:
        def __init__(self):
            self.calls: list[str] = []

        def ask(self, q: str) -> str:
            self.calls.append(q)
            return f"answer-of-{q}-after-{len(self.calls)}-calls"

    sys_obj = FakeSystem()
    qna = {"q1": "a1", "q2": "a2", "q3": "a3"}
    df = _common.run_session_eval(sys_obj, qna)
    assert list(df["question"]) == ["q1", "q2", "q3"]
    assert sys_obj.calls == ["q1", "q2", "q3"]
    assert "after-3-calls" in df["llm_answer"].iloc[2]
```

- [ ] **Step 2：跑 test 看到失败**

```bash
cd "notebook/C7 高级 RAG 技巧/6. 增强阶段"
python -m pytest tests/test_common_eval.py::test_run_session_eval_preserves_state -v
```

Expected: FAIL，`AttributeError: module '_common' has no attribute 'run_session_eval'`。

- [ ] **Step 3：在 `_common.py` 末尾追加 `run_session_eval`**

```python
# ---------- 系统评测（6.3 节首次出现，notebook 中 inline 完整重写一次） ----------

def run_session_eval(
    system,
    qna_dict: dict[str, str],
    *,
    eval_prompt_template: str | None = None,
) -> pd.DataFrame:
    """6.3 系统增强专用：按 dict 顺序调用 system.ask(q)，复用同一 system 实例
    以保留 memory / history / 路由状态。返回与 run_shared_eval 同结构的 DataFrame。
    """
    rows = []
    for question, expected in qna_dict.items():
        answer = system.ask(question)
        score = simple_eval_2pt(
            answer, expected, question,
            prompt_template=eval_prompt_template,
        )
        rows.append({
            "question": question,
            "llm_answer": answer,
            "expected_answer": expected,
            "rag_eval_results": score,
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4：跑全部 eval test**

```bash
cd "notebook/C7 高级 RAG 技巧/6. 增强阶段"
python -m pytest tests/test_common_eval.py -v
```

Expected: 8 个 test 全部 PASS。

- [ ] **Step 5：commit**

```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/_common.py" \
        "notebook/C7 高级 RAG 技巧/6. 增强阶段/tests/test_common_eval.py"
git commit -m "feat(c7): add run_session_eval for stateful systems (6.3)"
```

---

## Task 4：重构 6.1 `1. 上下文增强.ipynb`

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/1. 上下文增强.ipynb`

**目标：**

1. 保留所有方法（Sentence Window / Small-to-Big / AutoMerging + Late Chunking 理论 + 同题对比）
2. 把基础工具（embedding / PDF / Chroma / `llm_call` / `trim_context_to_budget`）改为 `from _common import ...`，并删掉 inline 定义
3. 保留 `simple_eval_2pt`、`answer_from_context_fn`、`run_shared_eval`、`build_compare_table` 在 notebook 中的 inline 完整定义（与 `_common.py` 字符相同），并紧接一段说明 markdown：「这个函数在 `_common.py` 中也保留了一份相同实现，6.2 / 6.3 节会直接 `from _common import` 复用」
4. 各 `*_context` 钩子签名保持不变；删除冗余的辅助函数（如 `inspect_baseline` 中重复的拼接逻辑可保留为教学示意，但内部应使用 `baseline_expanded_join` 等已有的纯函数）

**前置阅读：** spec §4、§5、§6

- [ ] **Step 1：列出 notebook 当前所有 cell 概览**

```bash
cd "notebook/C7 高级 RAG 技巧/6. 增强阶段"
jupyter nbconvert --to script "1. 上下文增强.ipynb" --stdout 2>/dev/null | grep -nE '^# In\[|^# ##? ' | head -50
```

记录每个章节标题对应的 cell 大致位置，便于后续定位。

- [ ] **Step 2：替换"基础环境"代码 cell**

定位到当前包含 `import fitz / EMBED_MODEL_ID / get_embeddings / get_cleaned_pdf_documents / open_or_build_chroma / llm_call / clean_text / split_sentences_for_window / trim_context_to_budget / build_rag_generation_prompt` 的若干 cell（spec §6.1 表中列为"不展示"的那些）。

把这些 cell 的代码**整段替换**为：

```python
import os
import re
import json
import warnings
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
sys.path.insert(0, ".")
from _common import (
    get_embeddings, get_cleaned_pdf_documents, open_or_build_chroma,
    llm_call, build_rag_generation_prompt, trim_context_to_budget,
    load_qna_subset, CHROMA_COLLECTION, CONTEXT_CHAR_BUDGET,
    PDF_PATH, QA_PATH,
)

# 句级切分仅 6.1 句窗用，留在本节显式可见
def split_sentences_for_window(text: str, max_piece: int = 320):
    raw = [s.strip() for s in re.split(r"(?<=[。！？!?])", text) if s.strip()]
    out = []
    for s in raw:
        if len(s) <= max_piece:
            out.append(s)
        else:
            for i in range(0, len(s), max_piece):
                out.append(s[i : i + max_piece])
    return out

warnings.filterwarnings("ignore")

print("✅ 公共底座（embedding / PDF / Chroma / llm_call / trim）已从 _common 引入")
```

并把上方对应的 markdown 改写为简短引导（不再讲 PDF 加载与 embedding 的细节，只说"前几章已介绍，本节直接复用"）。

- [ ] **Step 3：用 inline 完整定义替换"评估流水线"代码 cell**

定位到当前 `_parse_eval_score / simple_eval / answer_from_context_fn / run_shared_eval` 的 cell。

把这些 cell 的代码替换为以下 4 段（4 段分 4 个 code cell，每段前一个 markdown cell 介绍设计动机；以下用代码块标出每段内容，markdown 介绍由实现者按 spec §6 风格补，每段 markdown 控制在 80 字以内）：

**Cell A（markdown）：** 评分 prompt 与 0~2 分标准（说明评估口径偏严、为什么 0/1/2、为什么用 LLM 裁判）

**Cell B（code）—— 完整 inline `simple_eval_2pt`，与 `_common.py` 字符相同：**

```python
DEFAULT_EVAL_PROMPT_2PT = (
    "请作为一名判卷人，按 0～2 分评判「模型答案」回答「用户问题」的质量，并对照「参考答案」核对事实与要点。\n"
    "你只收到下列三段文字，没有 RAG 检索原文；请勿以「未逐字引用原文」为由扣分，除非答案与参考答案明显矛盾。\n"
    "评分标准：\n"
    "2 分：核心结论正确，问题所问的必答要点基本齐全，无明显事实错误；表述可比参考答案更短。\n"
    "1 分：方向基本正确，但明显缺少部分关键点，或解释不完整、表述含糊；无严重编造。\n"
    "0 分：关键结论错误、严重漏答问题核心、或与参考答案明显矛盾、或凭空补充。\n"
    "参考答案中的引导语、举例、排版说明不必复述；勿因未覆盖参考答案中的次要枝节就将 2 分打成 1 分。\n\n"
    "用户问题：{question}\n"
    "参考答案：{expected_answer}\n"
    "模型答案：{llm_answer}\n\n"
    "请仅输出一行，且该行只包含一个字符：0、1 或 2，不要输出任何其它文字。"
)


def _parse_eval_score(raw: str) -> int:
    text = (raw or "").strip()
    for line in text.splitlines():
        s = line.strip()
        s = re.sub(r"^[-*•\d.)]+\s*", "", s)
        if re.fullmatch(r"[012]", s):
            return int(s)
    m = re.search(r"(?<![0-9])([012])(?![0-9])", text)
    if m:
        return int(m.group(1))
    return 0


def simple_eval_2pt(llm_answer, expected_answer, question="", *, prompt_template=None):
    template = prompt_template or DEFAULT_EVAL_PROMPT_2PT
    prompt = template.format(
        question=question, expected_answer=expected_answer, llm_answer=llm_answer,
    )
    try:
        return _parse_eval_score(llm_call(prompt))
    except Exception as e:
        print(f"评估失败: {e}")
        return 0
```

紧接一个 markdown cell：

> 这个函数已收纳到 `_common.py`（字符相同），6.2 / 6.3 节会直接 `from _common import simple_eval_2pt` 复用，不再重复展示。

**Cell C（markdown + code）—— 完整 inline `answer_from_context_fn`：**

```python
def answer_from_context_fn(build_context_fn):
    """1 节专用胶水：把 *_context 钩子（仅返回 context 字符串）接到 LLM 上。
    6.2 / 6.3 不需要——pipeline / system.ask 已经直接产出答案。"""
    def _answer(question):
        context = build_context_fn(question)
        prompt = build_rag_generation_prompt(question, context)
        return llm_call(prompt)
    return _answer
```

紧接 markdown：

> 同样收纳到 `_common.py`；本节后续与下一节都通过 `import` 使用。

**Cell D（markdown + code）—— 完整 inline `run_shared_eval`：**

```python
def run_shared_eval(answer_fn, qna_dict, *, eval_prompt_template=None):
    """通用评测流水线：逐题调 answer_fn → simple_eval_2pt → DataFrame。"""
    rows = []
    for question, expected in qna_dict.items():
        answer = answer_fn(question)
        score = simple_eval_2pt(
            answer, expected, question, prompt_template=eval_prompt_template,
        )
        rows.append({
            "question": question, "llm_answer": answer,
            "expected_answer": expected, "rag_eval_results": score,
        })
    return pd.DataFrame(rows)
```

紧接 markdown：

> 同样收纳到 `_common.py`。6.2 节的 `*_pipeline` 也是 `q -> str` 的形状，所以可以直接喂给 `run_shared_eval`。

- [ ] **Step 4：替换"题集加载" cell**

把当前手写读 `train_dataset.json` 的 cell 替换为：

```python
QA_INDICES = list(range(20))
qna_dict = load_qna_subset(QA_PATH, QA_INDICES)
print(f"✅ 本节共用 {len(qna_dict)} 题（QA_INDICES={QA_INDICES[:5]}...）")
```

- [ ] **Step 5：在"同题对比"小节末尾用 `build_compare_table` 替换原对比表构建**

定位到当前 `build_compare_table(baseline_df, sentence_window_df, ...)` 局部函数定义并删除；替换为：

**Cell（markdown）：** "对比表也是 6.1 首次出现的概念，先在本节完整定义一次，再放进 `_common.py`。"

**Cell（code）—— 完整 inline 定义，与 `_common.py` 字符相同：**

```python
def build_compare_table(dfs, names):
    """把多个 run_shared_eval 的结果按 question 对齐拼接。"""
    if len(dfs) != len(names):
        raise ValueError("dfs 与 names 长度必须一致")
    base = dfs[0][["question"]].copy()
    base[names[0]] = dfs[0]["rag_eval_results"].values
    out = base
    for df, name in zip(dfs[1:], names[1:]):
        out = out.merge(
            df[["question", "rag_eval_results"]].rename(columns={"rag_eval_results": name}),
            on="question", how="outer",
        )
    return out.reset_index(drop=True)


compare_df = build_compare_table(
    [baseline_df, sentence_window_df, small_to_big_df, auto_merging_df],
    names=["baseline", "sentence_window", "small_to_big", "auto_merging"],
)
compare_df
```

随后保留原"提分题 / 回退 / 持平 / `summarize_context_eval`"的展示代码，但其内部对 DataFrame 列名的引用改用 `compare_df` 的新列。

- [ ] **Step 6：清理重复定义**

搜索 notebook 全文，确认以下名字**只在 inline 定义 cell 处出现一次**：
- `simple_eval_2pt`、`_parse_eval_score`、`answer_from_context_fn`、`run_shared_eval`、`build_compare_table`、`DEFAULT_EVAL_PROMPT_2PT`

删除任何重复的 `def` 或常量定义。

- [ ] **Step 7：执行 notebook 验证可运行**

```bash
cd "notebook/C7 高级 RAG 技巧/6. 增强阶段"
EVAL_MAX_UNIQ_QUESTIONS=2 jupyter nbconvert --to notebook --execute "1. 上下文增强.ipynb" \
  --output "1. 上下文增强.ipynb" --ExecutePreprocessor.timeout=900
```

Expected: 退出码 0，notebook 末尾有 `compare_df` 的输出表，无 import 错误。

- [ ] **Step 8：人工核对**

打开 notebook 检查：

1. 没有任何 cell 重复定义 `clean_text` / `get_embeddings` / `llm_call`
2. inline 定义的 `simple_eval_2pt`、`answer_from_context_fn`、`run_shared_eval`、`build_compare_table` 各仅出现一次
3. 后续这些函数的调用没有 `from _common import` 它们（因为本节已 inline）
4. 所有方法（baseline / SW / Small-to-Big / AutoMerging）的钩子签名仍是 `*_context(question) -> str`

- [ ] **Step 9：commit**

```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/1. 上下文增强.ipynb"
git commit -m "refactor(c7): 6.1 use _common for utils; inline define eval interfaces"
```

---

## Task 5：重构 6.2 `2. 流程增强.ipynb`

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb`

**目标：**

1. 删除复制粘贴的公共底座，改为 `from _common import`
2. 钩子升级为 `*_pipeline(question) -> str`（每个方法返回最终答案）
3. 引入 `Step` 数据结构，每个 `*_pipeline` 内部维护 `trace`，inspect 顺次打印
4. 评估口径升级为「维度计分版」：每题列 2~4 个必答要点，要点覆盖率→0~2 分
5. 删除硬编码的"对偶问题"单题，改用与 6.1 同款的 `qna_dict`（但选不同子集，体现"多维度复杂题"）
6. 删除所有 `time.sleep(20)` 硬编码，统一用 `llm_call(..., sleep_after=1)`
7. 新增跨方法对比表，调用 `build_compare_table` 拼接

**前置阅读：** spec §5（钩子表 + sleep 表）、§6（教学暴露规则）

- [ ] **Step 1：列出 notebook 当前所有方法 cell 与依赖**

```bash
cd "notebook/C7 高级 RAG 技巧/6. 增强阶段"
jupyter nbconvert --to script "2. 流程增强.ipynb" --stdout 2>/dev/null \
  | grep -nE '^# In\[|^# ##? |^def ' | head -80
```

Expected: 看到 6 个方法（迭代 / 递归 / 路由 / CRAG / Self-RAG / 自适应）。记录每个方法的当前 `def` 名，便于改写。

- [ ] **Step 2：替换"环境准备" cell**

把 cell 0~1（含 `import / load_dotenv / ChatZhipuAI / clean_text / llm_call(time.sleep(20)!) / build_chroma_* / load_chunks / build_retriever / evaluate_answer`）整段替换为：

```python
import os
import re
import json
import warnings
from dataclasses import dataclass, field
from typing import Any
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
sys.path.insert(0, ".")
from _common import (
    get_embeddings, get_cleaned_pdf_documents, open_or_build_chroma,
    llm_call as _raw_llm_call, build_rag_generation_prompt,
    trim_context_to_budget, load_qna_subset,
    simple_eval_2pt, run_shared_eval, build_compare_table,
    CHROMA_COLLECTION, CONTEXT_CHAR_BUDGET, PDF_PATH, QA_PATH,
)

warnings.filterwarnings("ignore")


def llm_call(prompt: str) -> str:
    """6.2 节默认每次成功调用后 sleep 1 秒，对抗多步流程的连续调用限流。"""
    return _raw_llm_call(prompt, sleep_after=1.0)


def load_chunks(chunk_size=256, chunk_overlap=20):
    docs = list(get_cleaned_pdf_documents())
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""], keep_separator=True,
    )
    return splitter.split_documents(docs)


def build_retriever(chunk_size=256, chunk_overlap=20, k=4):
    persist_dir = f"./chroma_db/baseline_{chunk_size}_{chunk_overlap}"
    chunks = load_chunks(chunk_size, chunk_overlap)
    ids = [f"b{i}" for i in range(len(chunks))]
    vs = open_or_build_chroma(persist_dir, chunks, ids)
    return vs.as_retriever(search_kwargs={"k": k})


retriever = build_retriever()
print("✅ 6.2 环境准备完成（公共底座来自 _common；本节默认 sleep_after=1.0）")
```

- [ ] **Step 3：增加 markdown + code cell 介绍 6.2 的钩子约定 `*_pipeline` 与 `Step`**

在"环境准备"之后、第一个方法之前，插入：

**Cell（markdown）：**

```markdown
## 本节钩子：`*_pipeline(question) -> str`

6.1 节的钩子是 `*_context(question) -> str`——每个方法只决定"如何拼 context"，
拼完后由公共胶水接到 LLM。但流程增强的核心**就是**控制流：多轮检索、子问题、
路由、自反思。所以本节钩子升一层——**每个方法直接返回最终答案**：

```
*_pipeline(question) -> answer_str
```

由于 `*_pipeline` 也是 `q -> str` 的形状，公共底座的 `run_shared_eval` 不需要修改，
直接把 `*_pipeline` 当 `answer_fn` 喂进去即可。

为了让读者看到"流程的形状"，每个方法内部维护一个 `trace: list[Step]`，
`inspect_*` 顺次打印这些步骤——这是本节 inspect 的重点。
```

**Cell（code）—— 完整 inline 定义 `Step`：**

```python
@dataclass
class Step:
    """一次流程内的中间步骤。kind 例：retrieve / draft / route / reflect / finalize。"""
    kind: str
    payload: dict = field(default_factory=dict)


def print_trace(trace: list[Step]) -> None:
    for i, s in enumerate(trace):
        head = f"[{i+1}] {s.kind}"
        body = ", ".join(f"{k}={repr(v)[:60]}" for k, v in s.payload.items())
        print(f"  {head}  {body}")
```

- [ ] **Step 4：定义本节维度计分 prompt（inline 完整重写）**

新增一个 markdown + code cell：

**Cell（markdown）：**

```markdown
## 本节评估口径：维度计分版 0~2 分

6.2 处理的是「多维度复杂问题」（如"对偶优势 + KKT + Slater"三个独立角度）。
朴素 0~2 分裁判会被语言流畅度干扰。本节定制 prompt：要求裁判先列出
**该题的必答要点**，再按要点覆盖比例映射到 0/1/2，输出仍是一行 0/1/2。
接口仍是 `simple_eval_2pt(answer, expected, question, prompt_template=...)`，
保证 `build_compare_table` 能与 6.1 / 6.3 拼接。
```

**Cell（code）：**

```python
DIMENSIONAL_EVAL_PROMPT = (
    "你是判卷人。先在心里把「参考答案」拆成 2~4 个【必答要点】"
    "（如：方法1、方法2、关键条件、关键定义等），不要输出这些要点。\n"
    "然后比对「模型答案」覆盖了几个要点：\n"
    "- 全部覆盖且无关键事实错误：2\n"
    "- 覆盖一半左右、或部分要点表述含糊但方向正确：1\n"
    "- 大部分要点缺失、或关键事实错误：0\n\n"
    "用户问题：{question}\n参考答案：{expected_answer}\n模型答案：{llm_answer}\n\n"
    "仅输出一行，只包含字符 0、1 或 2，不要任何其它文字。"
)


def eval_pipeline(answer_fn, qna_dict):
    """6.2 节的快捷入口：固定使用维度计分 prompt。"""
    return run_shared_eval(answer_fn, qna_dict, eval_prompt_template=DIMENSIONAL_EVAL_PROMPT)
```

- [ ] **Step 5：替换题集 cell**

把当前 `baseline_query = "在机器学习中，对偶问题..."` 与 `evaluate_answer(...)` 的硬编码段落删除；改为：

```python
QA_INDICES = [0, 1, 3, 4, 7]  # 本节挑"多维度复杂"题；可按需调整
qna_dict = load_qna_subset(QA_PATH, QA_INDICES)
print(f"✅ 本节使用 {len(qna_dict)} 道多维度题（QA_INDICES={QA_INDICES}）")
```

- [ ] **Step 6：把每个方法改写为 `*_pipeline` 钩子（以迭代为示范，其它 5 个方法照葫芦画瓢）**

对**第一个方法（迭代检索）**，把当前的 `iterative_retrieval(query, max_rounds=2)` 改写为：

```python
def iterative_pipeline(question: str, max_rounds: int = 2) -> str:
    trace: list[Step] = []
    context_blocks: list[str] = []
    missing_hint = ""

    for i in range(max_rounds):
        effective_q = question if not missing_hint else f"{question}\n补充检索线索：{missing_hint}"
        docs = retriever.invoke(effective_q)
        context_blocks.extend(d.page_content for d in docs)
        merged_ctx = trim_context_to_budget("\n\n".join(context_blocks[-8:]), CONTEXT_CHAR_BUDGET)
        trace.append(Step("retrieve", {"round": i + 1, "hint": missing_hint, "n_hits": len(docs)}))

        draft_prompt = (
            f"问题：{question}\n上下文：\n{merged_ctx}\n"
            "请先给出当前答案，再用一句话指出仍缺失的关键信息。\n"
            "输出格式：\n当前答案：...\n缺失信息：..."
        )
        draft = llm_call(draft_prompt)
        trace.append(Step("draft", {"round": i + 1, "draft_head": draft[:80]}))

        if "缺失信息：无" in draft or "缺失信息: 无" in draft:
            trace.append(Step("finalize", {"reason": "complete_after_draft"}))
            iterative_pipeline.last_trace = trace  # 给 inspect 使用
            return draft

        missing_hint = draft.split("缺失信息")[-1].strip("：: \n")[:120]

    final = llm_call(build_rag_generation_prompt(question, merged_ctx))
    trace.append(Step("finalize", {"reason": "max_rounds_reached"}))
    iterative_pipeline.last_trace = trace
    return final


def inspect_iterative(question: str) -> None:
    print(f"❓ 问题: {question}\n")
    ans = iterative_pipeline(question)
    print(f"🧠 最终答案：\n{ans}\n")
    print("🔍 流程 trace：")
    print_trace(iterative_pipeline.last_trace)
```

为其它 5 个方法（递归 / 路由 / CRAG / Self-RAG / 自适应）做相同改造：函数名改 `*_pipeline`、内部维护 `trace`、绑定 `*.last_trace`、配对一个 `inspect_*`。

每个方法改完后跑一次 inspect 验证：

```python
inspect_iterative(list(qna_dict.keys())[0])
```

确认能看到 trace 输出。

- [ ] **Step 7：替换原"对每个方法单独调用 evaluate_answer"为统一 eval_pipeline 与对比表**

在所有方法定义之后，加一组 cell：

**Cell（code）：**

```python
iterative_df = eval_pipeline(iterative_pipeline, qna_dict)
recursive_df = eval_pipeline(recursive_pipeline, qna_dict)
routing_df   = eval_pipeline(routing_pipeline,   qna_dict)
crag_df      = eval_pipeline(crag_pipeline,      qna_dict)
selfrag_df   = eval_pipeline(self_rag_pipeline,  qna_dict)
adaptive_df  = eval_pipeline(adaptive_pipeline,  qna_dict)
```

**Cell（code）：**

```python
compare_df = build_compare_table(
    [iterative_df, recursive_df, routing_df, crag_df, selfrag_df, adaptive_df],
    names=["iterative", "recursive", "routing", "crag", "self_rag", "adaptive"],
)
compare_df
```

- [ ] **Step 8：在 notebook 末尾增加结论 markdown**

```markdown
## 跨方法对比读法

`compare_df` 每行一题、每列一个方法的 0~2 分（裁判使用维度计分 prompt）。
注意：本节 6 个方法各有自己的"擅长场景"——比如递归适合可显式拆解的问题，
路由适合多源场景。同题分数高低不是绝对优劣指标，应结合 inspect 输出的 trace
理解"流程是否走对"。
```

- [ ] **Step 9：搜索并删除残留**

```bash
cd "notebook/C7 高级 RAG 技巧/6. 增强阶段"
jupyter nbconvert --to script "2. 流程增强.ipynb" --stdout 2>/dev/null \
  | grep -nE 'time\.sleep\(20|对偶问题|evaluate_answer|ChatZhipuAI|baseline_query'
```

Expected: 无输出（所有硬编码已删）。如有残留，定位修复。

- [ ] **Step 10：执行 notebook 验证**

```bash
cd "notebook/C7 高级 RAG 技巧/6. 增强阶段"
EVAL_MAX_UNIQ_QUESTIONS=2 jupyter nbconvert --to notebook --execute "2. 流程增强.ipynb" \
  --output "2. 流程增强.ipynb" --ExecutePreprocessor.timeout=1800
```

Expected: 退出码 0，末尾有 `compare_df` 表，trace 在每个 inspect 下可见。

- [ ] **Step 11：commit**

```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/2. 流程增强.ipynb"
git commit -m "refactor(c7): 6.2 hooks->pipeline+trace, dimensional 0~2 judge, drop sleep(20)"
```

---

## Task 6：6.3 准备 — Memory 数据集 `memory_sessions.json`

**Files:**
- Create: `notebook/C7 高级 RAG 技巧/6. 增强阶段/data/memory_sessions.json`

**目标：** 手工构造 3~4 个多轮对话脚本，每脚本 3~4 轮，**后续轮含明显指代**（"它"、"上面提到的方法"、"这个公式"等），并附标准答案。

- [ ] **Step 1：写数据文件**

创建 `notebook/C7 高级 RAG 技巧/6. 增强阶段/data/memory_sessions.json`：

```json
{
  "sessions": [
    {
      "id": "s1_decision_tree",
      "title": "决策树连续属性",
      "turns": [
        {
          "q": "决策树在处理连续属性时常用什么离散化方法？",
          "a": "二分法（bi-partition）：对连续属性的取值集合排序后，候选划分点为相邻取值的中点，按信息增益等准则选择最优划分点。"
        },
        {
          "q": "它的候选划分点是怎么确定的？",
          "a": "把该连续属性在样本中的取值升序排列，相邻两个取值的中点构成候选划分点集合。"
        },
        {
          "q": "选择最优划分点的准则是什么？",
          "a": "在所有候选划分点中，选使信息增益（或增益率、基尼指数等）最大的那个作为最终划分点。"
        }
      ]
    },
    {
      "id": "s2_eval_methods",
      "title": "模型评估方法",
      "turns": [
        {
          "q": "机器学习中常用的模型评估方法有哪些？",
          "a": "留出法（hold-out）、交叉验证法（k-fold cross-validation）、自助法（bootstrap）。"
        },
        {
          "q": "上面提到的方法各自适合什么数据规模？",
          "a": "留出法适合数据量较大的场景；交叉验证法适合中等数据量、需要稳定估计；自助法适合数据集较小、难以有效划分训练/测试集的场景。"
        },
        {
          "q": "其中哪种会改变训练集的样本分布？",
          "a": "自助法（bootstrap）通过有放回抽样构造训练集，会改变原始数据的分布，引入估计偏差。"
        }
      ]
    },
    {
      "id": "s3_pr_f1",
      "title": "二分类指标",
      "turns": [
        {
          "q": "什么是查准率（precision）和查全率（recall）？",
          "a": "查准率 = TP / (TP + FP)，预测为正例中真正为正例的比例；查全率 = TP / (TP + FN)，真实正例中被预测为正例的比例。"
        },
        {
          "q": "F1 是怎么把它们组合起来的？",
          "a": "F1 = 2·P·R/(P+R)，是查准率与查全率的调和平均。"
        },
        {
          "q": "如果想偏重其中之一，可以用什么变体？",
          "a": "Fβ：Fβ=(1+β²)·P·R/(β²·P+R)。β>1 偏重查全率，β<1 偏重查准率。"
        }
      ]
    }
  ]
}
```

- [ ] **Step 2：commit**

```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/data/memory_sessions.json"
git commit -m "data(c7): add memory_sessions.json for 6.3 multi-turn dialog eval"
```

---

## Task 7：6.3 编写 — Memory（多轮对话 RAG）

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/3. 系统增强.ipynb`

**目标：** 在 6.3 notebook 中：

1. 替换原"环境准备"为 `from _common import` 的简洁版本
2. 介绍本节钩子约定：`class XxxSystem: ask(q) -> str`
3. inline 完整定义 `MemoryRAGSystem`、`run_session_eval`、行为评估 prompt
4. 跑通 3 个 session，输出 condensed query 与命中变化作为客观信号

**前置阅读：** spec §5、§7.1

- [ ] **Step 1：替换 cell 5（旧环境准备代码）**

把当前 `import os ... ChatZhipuAI ... time.sleep(20) ... simple_eval ✅/❌` 整块代码替换为：

```python
import os
import re
import json
import warnings
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
sys.path.insert(0, ".")
from _common import (
    get_embeddings, get_cleaned_pdf_documents, open_or_build_chroma,
    llm_call as _raw_llm_call, build_rag_generation_prompt,
    trim_context_to_budget, simple_eval_2pt, build_compare_table,
    CHROMA_COLLECTION, CONTEXT_CHAR_BUDGET, PDF_PATH, QA_PATH,
)

warnings.filterwarnings("ignore")


def llm_call(prompt: str) -> str:
    """6.3 节默认每次成功调用后 sleep 1 秒。"""
    return _raw_llm_call(prompt, sleep_after=1.0)


def load_chunks(chunk_size=256, chunk_overlap=20):
    docs = list(get_cleaned_pdf_documents())
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""], keep_separator=True,
    )
    return splitter.split_documents(docs)


def build_retriever(chunk_size=256, chunk_overlap=20, k=4, persist_subdir="baseline_256_20"):
    persist_dir = f"./chroma_db/{persist_subdir}"
    chunks = load_chunks(chunk_size, chunk_overlap)
    ids = [f"b{i}" for i in range(len(chunks))]
    vs = open_or_build_chroma(persist_dir, chunks, ids)
    return vs.as_retriever(search_kwargs={"k": k})


retriever = build_retriever()
print("✅ 6.3 环境准备完成（公共底座来自 _common）")
```

- [ ] **Step 2：在"环境准备"之后插入「本节钩子约定」markdown**

```markdown
## 本节钩子：`class XxxSystem: ask(q) -> str`

6.1 钩子是 `*_context`（变 context 拼法），6.2 钩子是 `*_pipeline`（变流程编排）。
但系统增强的核心是**状态**——跨轮记忆、跨源路由——单题接口 `q -> str` 根本
表达不出来。所以本节钩子升一层：每个方法是一个**带状态的类**，对外提供 `ask(q)`。

```
class MemoryRAGSystem:    def ask(q) -> str   # 内部更新 history
class MultiDocAgent:      def ask(q) -> str   # 内部路由到不同 source
```

公共底座为此提供 `run_session_eval(system, qna_dict)`：按顺序调用 `system.ask`，
**复用同一 system 实例**以保留状态。它是 6.3 首次出现的概念，下面会先 inline
完整定义一次，再放进 `_common.py` 供未来扩展使用。
```

- [ ] **Step 3：inline 完整定义 `run_session_eval` 与行为评估 prompt**

```python
def run_session_eval(system, qna_dict, *, eval_prompt_template=None):
    """按顺序调 system.ask(q)，复用同一 system 实例以保留状态。"""
    rows = []
    for question, expected in qna_dict.items():
        answer = system.ask(question)
        score = simple_eval_2pt(
            answer, expected, question, prompt_template=eval_prompt_template,
        )
        rows.append({
            "question": question, "llm_answer": answer,
            "expected_answer": expected, "rag_eval_results": score,
        })
    return pd.DataFrame(rows)


MEMORY_BEHAVIORAL_EVAL_PROMPT = (
    "你是判卷人。本题是多轮对话中的一轮，「用户问题」可能含指代（如『它』、"
    "『上面提到的方法』）。请按 0~2 分评估「模型答案」：\n"
    "2 分：正确解析指代，并给出与「参考答案」核心一致的回答。\n"
    "1 分：解析了指代但答案部分缺失或方向正确但不完整。\n"
    "0 分：未解析指代、答非所问、或与参考答案明显矛盾。\n\n"
    "用户问题：{question}\n参考答案：{expected_answer}\n模型答案：{llm_answer}\n\n"
    "仅输出一行，只包含字符 0、1 或 2。"
)
```

紧接 markdown：

> 上面 `run_session_eval` 已收纳到 `_common.py`，本节后续与未来章节通过 `import` 复用。

- [ ] **Step 4：inline 完整定义 `MemoryRAGSystem`**

```python
class MemoryRAGSystem:
    """多轮对话 RAG：每次 ask 时，先用 LLM 把 (history, q) 改写成独立 query，
    再正常 retrieve + generate；新 (q, a) 入 history。"""

    def __init__(self, retriever, max_history: int = 5):
        self.retriever = retriever
        self.history: list[tuple[str, str]] = []
        self.max_history = max_history
        self.last_debug: dict = {}

    def _condense(self, q: str) -> str:
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
        ctx = trim_context_to_budget(
            "\n\n".join(d.page_content for d in docs),
            CONTEXT_CHAR_BUDGET,
        )
        ans = llm_call(build_rag_generation_prompt(q, ctx))
        self.last_debug = {
            "original": q, "condensed": condensed,
            "n_hits": len(docs),
            "first_hit_head": docs[0].page_content[:80] if docs else "",
        }
        self.history.append((q, ans))
        self.history = self.history[-self.max_history :]
        return ans
```

- [ ] **Step 5：inspect 与对比基线**

```python
def inspect_memory_session(session: dict) -> None:
    print(f"━━━━━ Session: {session['title']} ━━━━━\n")
    sys = MemoryRAGSystem(retriever)
    for i, turn in enumerate(session["turns"]):
        q = turn["q"]
        ans = sys.ask(q)
        d = sys.last_debug
        print(f"--- Turn {i+1} ---")
        print(f"  原问题：{q}")
        print(f"  Condensed：{d['condensed']}")
        print(f"  改写发生：{'是' if d['condensed'] != q else '否'}")
        print(f"  命中数：{d['n_hits']}；首条命中：{d['first_hit_head']}...")
        print(f"  答案：{ans[:160]}...")
        print()


with open("data/memory_sessions.json", "r", encoding="utf-8") as f:
    SESSIONS = json.load(f)["sessions"]

inspect_memory_session(SESSIONS[0])
```

- [ ] **Step 6：用行为评估跑全部 session 并产出对比表**

```python
def session_to_qna(session):
    return {turn["q"]: turn["a"] for turn in session["turns"]}


def baseline_no_memory_ask_factory():
    """对照组：每题独立 retrieve + generate，无 history。"""
    class NoMemorySystem:
        def __init__(self, retriever):
            self.retriever = retriever
        def ask(self, q):
            docs = self.retriever.invoke(q)
            ctx = trim_context_to_budget(
                "\n\n".join(d.page_content for d in docs), CONTEXT_CHAR_BUDGET,
            )
            return llm_call(build_rag_generation_prompt(q, ctx))
    return NoMemorySystem(retriever)


memory_dfs, baseline_dfs = [], []
for sess in SESSIONS:
    qna = session_to_qna(sess)
    mem_sys = MemoryRAGSystem(retriever)
    base_sys = baseline_no_memory_ask_factory()
    memory_dfs.append(run_session_eval(mem_sys, qna, eval_prompt_template=MEMORY_BEHAVIORAL_EVAL_PROMPT))
    baseline_dfs.append(run_session_eval(base_sys, qna, eval_prompt_template=MEMORY_BEHAVIORAL_EVAL_PROMPT))

memory_all = pd.concat(memory_dfs, ignore_index=True)
baseline_all = pd.concat(baseline_dfs, ignore_index=True)
memory_compare = build_compare_table(
    [baseline_all, memory_all],
    names=["no_memory_baseline", "memory"],
)
memory_compare
```

- [ ] **Step 7：执行 notebook 验证**

```bash
cd "notebook/C7 高级 RAG 技巧/6. 增强阶段"
jupyter nbconvert --to notebook --execute "3. 系统增强.ipynb" \
  --output "3. 系统增强.ipynb" --ExecutePreprocessor.timeout=1800
```

Expected:
- 至少 1 轮 `Condensed` 与原问题不同（验收标准 5 的客观信号）
- `memory_compare` 表中 memory 列至少有一行 > baseline 列

如未达成，调整 condensed prompt 或 session 数据后重跑。

- [ ] **Step 8：commit**

```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/3. 系统增强.ipynb"
git commit -m "feat(c7): 6.3 add MemoryRAGSystem with run_session_eval and behavioral judge"
```

---

## Task 8：6.3 编写 — Multi-Document Agent（多源路由）

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/3. 系统增强.ipynb`

**目标：** 在 Memory 之后追加 `MultiDocAgent`：把南瓜书按章节拆为 3~4 个独立 source（向量库），路由 LLM 选 source，对比"全库 baseline vs 多源路由"。

- [ ] **Step 1：在 Memory 章节之后插入 markdown**

```markdown
## Multi-Document Agent（多源路由）

Memory 解决"跨轮"，Multi-Doc Agent 解决"跨源"。当文档分布在多个领域 / 章节时，
盲检全库会把无关章节的相似片段也召回。Multi-Doc Agent 的核心是：
**每个 source 一个独立的检索器 + 一个 LLM 路由器**——根据问题选 1~3 个 source，
仅在选中的 source 上检索。

下面把南瓜书按页区间切成若干"章节 source"，建独立的 Chroma 集合。
```

- [ ] **Step 2：按页切分章节**

```python
def split_by_page_ranges(page_ranges: dict[str, tuple[int, int]]):
    """page_ranges 例：{'ch1_intro': (0, 18), 'ch2_eval': (19, 45), ...}
    返回 {source_name: list[Document]}。
    """
    pages = get_cleaned_pdf_documents()
    out: dict[str, list[Document]] = {}
    for name, (lo, hi) in page_ranges.items():
        docs = []
        for p in pages:
            page_idx = p.metadata.get("page", -1)
            if lo <= page_idx <= hi and p.page_content.strip():
                docs.append(p)
        out[name] = docs
    return out


SOURCE_PAGE_RANGES = {
    "ch1_intro":     (0, 18),
    "ch2_eval":      (19, 45),
    "ch3_linear":    (46, 75),
    "ch4_tree":      (76, 110),
}

SOURCE_DESCRIPTIONS = {
    "ch1_intro":     "第1章 绪论：机器学习基本术语、假设空间、归纳偏好。",
    "ch2_eval":      "第2章 模型评估与选择：留出法、交叉验证、自助法、查准查全率、ROC、偏差方差。",
    "ch3_linear":    "第3章 线性模型：线性回归、对数几率回归、LDA、多分类、类别不平衡。",
    "ch4_tree":      "第4章 决策树：信息增益、增益率、基尼指数、剪枝、连续与缺失值。",
}


def build_chapter_sources():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=256, chunk_overlap=20,
        separators=["\n\n", "\n", " ", ""], keep_separator=True,
    )
    by_source = split_by_page_ranges(SOURCE_PAGE_RANGES)
    retrievers = {}
    for name, page_docs in by_source.items():
        chunks = splitter.split_documents(page_docs)
        if not chunks:
            print(f"  [warn] source {name} 切完为空，跳过")
            continue
        ids = [f"{name}-{i}" for i in range(len(chunks))]
        vs = open_or_build_chroma(f"./chroma_db/multi_doc/{name}", chunks, ids)
        retrievers[name] = vs.as_retriever(search_kwargs={"k": 4})
    return retrievers


CHAPTER_RETRIEVERS = build_chapter_sources()
print(f"✅ 多源构建完成，可用 sources：{list(CHAPTER_RETRIEVERS.keys())}")
```

- [ ] **Step 3：inline 完整定义 `MultiDocAgent`**

```python
class MultiDocAgent:
    """多源路由 RAG：LLM 根据问题与 source 描述选 1~3 个 source，仅在选中源检索。"""

    def __init__(self, sources: dict, descriptions: dict):
        self.sources = sources
        self.descriptions = descriptions
        self.last_debug: dict = {}

    def _route(self, q: str) -> list[str]:
        desc_text = "\n".join(f"- {k}: {v}" for k, v in self.descriptions.items())
        prompt = (
            "你是一个路由器。给定可选的知识源描述与用户问题，"
            "请输出 1~3 个最相关的 source key（每行一个 key，不要任何其它内容）。\n\n"
            f"可选源：\n{desc_text}\n\n问题：{q}\n\n选择："
        )
        out = llm_call(prompt).strip().splitlines()
        valid = [k.strip() for k in out if k.strip() in self.sources][:3]
        return valid or [next(iter(self.sources))]

    def ask(self, q: str) -> str:
        chosen = self._route(q)
        docs = []
        per_source_hits = {}
        for name in chosen:
            r = self.sources[name].invoke(q)
            docs.extend(r)
            per_source_hits[name] = len(r)
        ctx = trim_context_to_budget(
            "\n\n".join(d.page_content for d in docs), CONTEXT_CHAR_BUDGET,
        )
        ans = llm_call(build_rag_generation_prompt(q, ctx))
        self.last_debug = {"chosen": chosen, "per_source_hits": per_source_hits}
        return ans
```

- [ ] **Step 4：inspect**

```python
def inspect_multi_doc(question: str, agent: MultiDocAgent) -> None:
    print(f"❓ 问题：{question}")
    ans = agent.ask(question)
    print(f"🧭 路由选择：{agent.last_debug['chosen']}")
    print(f"📊 各源命中：{agent.last_debug['per_source_hits']}")
    print(f"💬 答案：{ans[:200]}...\n")


agent = MultiDocAgent(CHAPTER_RETRIEVERS, SOURCE_DESCRIPTIONS)
inspect_multi_doc("决策树连续属性的二分法", agent)
inspect_multi_doc("交叉验证与留出法的区别", agent)
```

- [ ] **Step 5：评估 — 标注题集 expected 章节，对比"全库 baseline vs 多源路由"**

```python
MULTI_DOC_EVAL_PROMPT = (
    "你是判卷人。本题需要从特定章节的知识回答。请按 0~2 分评估「模型答案」：\n"
    "2 分：与「参考答案」核心一致，无关键事实错误。\n"
    "1 分：方向正确但部分要点缺失或表述含糊。\n"
    "0 分：错误回答或与参考答案明显矛盾。\n\n"
    "用户问题：{question}\n参考答案：{expected_answer}\n模型答案：{llm_answer}\n\n"
    "仅输出一行，只包含字符 0、1 或 2。"
)


# 从 train_dataset.json 挑 6 道题，并人工标注它们应路由到的 source（用于客观信号）
MULTI_DOC_QA = {
    # train_idx, expected source key
    0:  "ch4_tree",      # "算法、模型与...关系"——常见放第 1 章绪论；按你的题集核对，必要时改 ch1_intro
    5:  "ch2_eval",      # 交叉验证 vs 留出法
    6:  "ch2_eval",      # F1 与 β
    7:  "ch2_eval",      # 宏 / 微平均
    26: "ch4_tree",      # 决策树连续属性
    27: "ch4_tree",      # 图 4-2 决策划分
}

with open(QA_PATH, "r", encoding="utf-8") as f:
    _all_pairs = json.load(f)
multi_doc_qna = {_all_pairs[i]["query"]: _all_pairs[i]["answer"] for i in MULTI_DOC_QA.keys()}
expected_source_by_question = {_all_pairs[i]["query"]: src for i, src in MULTI_DOC_QA.items()}


# 全库 baseline：用单一 retriever（已有的 retriever）
class FullCorpusSystem:
    def __init__(self, retriever):
        self.retriever = retriever
    def ask(self, q):
        docs = self.retriever.invoke(q)
        ctx = trim_context_to_budget(
            "\n\n".join(d.page_content for d in docs), CONTEXT_CHAR_BUDGET,
        )
        return llm_call(build_rag_generation_prompt(q, ctx))


full_baseline = FullCorpusSystem(retriever)
agent_for_eval = MultiDocAgent(CHAPTER_RETRIEVERS, SOURCE_DESCRIPTIONS)

baseline_md_df = run_session_eval(full_baseline, multi_doc_qna, eval_prompt_template=MULTI_DOC_EVAL_PROMPT)
agent_md_df = run_session_eval(agent_for_eval, multi_doc_qna, eval_prompt_template=MULTI_DOC_EVAL_PROMPT)

multi_doc_compare = build_compare_table(
    [baseline_md_df, agent_md_df], names=["full_corpus", "multi_doc_agent"],
)
multi_doc_compare
```

- [ ] **Step 6：路由命中正确率（客观信号）**

```python
# 重跑一遍 ask 并记录路由
agent_for_check = MultiDocAgent(CHAPTER_RETRIEVERS, SOURCE_DESCRIPTIONS)
correct_routes = 0
total = len(multi_doc_qna)
for q in multi_doc_qna:
    _ = agent_for_check.ask(q)
    chosen = agent_for_check.last_debug["chosen"]
    expected_src = expected_source_by_question[q]
    hit = expected_src in chosen
    correct_routes += int(hit)
    print(f"  Q={q[:40]!s}...  chosen={chosen}  expected={expected_src}  {'✅' if hit else '❌'}")
print(f"\n路由命中正确率：{correct_routes}/{total} = {correct_routes/total:.0%}")
```

Expected: ≥ 4/6 命中（验收标准 5 的客观信号）。如低于该值，调整 `SOURCE_DESCRIPTIONS` 让 LLM 路由更易做对。

- [ ] **Step 7：执行 notebook 验证**

```bash
cd "notebook/C7 高级 RAG 技巧/6. 增强阶段"
jupyter nbconvert --to notebook --execute "3. 系统增强.ipynb" \
  --output "3. 系统增强.ipynb" --ExecutePreprocessor.timeout=2400
```

Expected: 退出码 0；末尾两个对比表 `memory_compare`、`multi_doc_compare` 均渲染；路由正确率 ≥ 4/6。

- [ ] **Step 8：commit**

```bash
git add "notebook/C7 高级 RAG 技巧/6. 增强阶段/3. 系统增强.ipynb"
git commit -m "feat(c7): 6.3 add MultiDocAgent with chapter sources and routing eval"
```

---

## Task 9：收尾 — 验收、readme 同步、_common 同步检查

**Files:**
- Modify: `notebook/C7 高级 RAG 技巧/6. 增强阶段/readme.md`（如需）
- Read-only check: 三节 notebook + `_common.py`

- [ ] **Step 1：验收 §9 验收标准 1（架构）**

```bash
cd "notebook/C7 高级 RAG 技巧/6. 增强阶段"
for nb in "1. 上下文增强.ipynb" "2. 流程增强.ipynb" "3. 系统增强.ipynb"; do
  echo "--- $nb 中重复定义检查 ---"
  jupyter nbconvert --to script "$nb" --stdout 2>/dev/null \
    | grep -nE '^def (clean_text|get_embeddings|get_cleaned_pdf_documents|llm_call|_parse_eval_score|run_session_eval) '
done
```

Expected:
- `clean_text` / `get_embeddings` / `get_cleaned_pdf_documents` / `llm_call`：三节都为空（已 import）
- `_parse_eval_score` / `simple_eval_2pt` 等：仅在 6.1 出现
- `run_session_eval`：仅在 6.3 出现

- [ ] **Step 2：验收 §9 验收标准 3（教学暴露）**

```bash
cd "notebook/C7 高级 RAG 技巧/6. 增强阶段"
echo "--- 6.1 inline 定义检查 ---"
jupyter nbconvert --to script "1. 上下文增强.ipynb" --stdout 2>/dev/null \
  | grep -cE '^def (simple_eval_2pt|answer_from_context_fn|run_shared_eval|build_compare_table) '
echo "--- 6.2 inline 定义 Step 检查 ---"
jupyter nbconvert --to script "2. 流程增强.ipynb" --stdout 2>/dev/null \
  | grep -cE '^class Step|^def print_trace '
echo "--- 6.3 inline 定义检查 ---"
jupyter nbconvert --to script "3. 系统增强.ipynb" --stdout 2>/dev/null \
  | grep -cE '^def run_session_eval|^class MemoryRAGSystem|^class MultiDocAgent '
```

Expected: 6.1 输出 4；6.2 输出 ≥ 2；6.3 输出 3。

- [ ] **Step 3：验证 _common 与首次定义节字符相同（关键一致性）**

```bash
cd "notebook/C7 高级 RAG 技巧/6. 增强阶段"
python -c "
import inspect, sys
sys.path.insert(0, '.')
import _common
funcs = ['simple_eval_2pt', '_parse_eval_score', 'answer_from_context_fn', 'run_shared_eval', 'build_compare_table', 'run_session_eval']
for f in funcs:
    src = inspect.getsource(getattr(_common, f))
    print(f'=== {f} ({len(src)} chars) ===')
    print(src[:120].replace(chr(10), ' / '))
"
```

人工核对：把上面打印出的每个函数的源码与对应 notebook 的 inline cell 比对，确认参数名、默认值、字符串 prompt 完全一致。

如不一致，修正较新的一方（一般是 `_common.py` 跟 notebook 走，因为 notebook 是教学时被读者首先看到的）。

- [ ] **Step 4：验收 §9 验收标准 7（代码量）**

```bash
cd "notebook/C7 高级 RAG 技巧/6. 增强阶段"
git log --all --pretty=format:'%h %s' -- "1. 上下文增强.ipynb" | head -3
echo "---"
for nb in "1. 上下文增强.ipynb" "2. 流程增强.ipynb" "3. 系统增强.ipynb"; do
  cur=$(jupyter nbconvert --to script "$nb" --stdout 2>/dev/null | wc -l)
  echo "  $nb: 当前 $cur 行"
done
```

把 git diff 之前的版本拉出来对比（用 `git show` + 计数），确认三节合计代码行数（不含 markdown）减少 ≥ 25%。如未达成，分析是哪一节膨胀（通常 6.3 从零写会拉高总量），按 spec 验收 7 是"合计"，可以接受 6.3 增、6.1/6.2 减。

- [ ] **Step 5：同步 readme.md（仅在与本设计冲突时改）**

```bash
cd "notebook/C7 高级 RAG 技巧/6. 增强阶段"
cat readme.md
```

如果 readme 中 6.3 仍承诺 4 个方法（含 GraphRAG / Agentic RAG），把这两条改为"后续扩展"备注，明确本轮实现 Memory + Multi-Doc Agent。其它内容不动。

- [ ] **Step 6：跑全部 _common 测试最后一遍**

```bash
cd "notebook/C7 高级 RAG 技巧/6. 增强阶段"
python -m pytest tests/ -v
```

Expected: 全部 PASS（13 个 test）。

- [ ] **Step 7：commit 收尾**

```bash
git add -A "notebook/C7 高级 RAG 技巧/6. 增强阶段"
git commit -m "chore(c7): finalize enhancement-stage redesign — readme & checks" || echo "无新增改动可提交"
```

---

## Self-Review

**1. Spec 覆盖检查**

| Spec 章节 | 实现任务 |
|---|---|
| §1 背景与问题 | （叙述性，无任务） |
| §2 目标 | Tasks 1~9 整体覆盖 |
| §2 非目标 | 计划中未引入 Contextual Retrieval / RAPTOR / GraphRAG / Agentic RAG / Late Chunking demo ✓ |
| §3 设计原则（钩子落本节核心变化点） | Task 4（6.1 钩子保持）、Task 5（6.2 升 pipeline）、Task 7/8（6.3 升 system 类）✓ |
| §4 `_common.py` 边界 | Tasks 1、2、3 ✓ |
| §4.3 single source of truth 同步 | Task 9 Step 3 验证 ✓ |
| §5 三节钩子 + 评估 + sleep | Task 4/5/7-8 + Task 9 验收 ✓ |
| §6 教学暴露规则 | Task 4（6.1 inline 4 个函数）、Task 5（6.2 inline `Step`）、Task 7（6.3 inline `run_session_eval`）✓ |
| §7.1 Memory 骨架 | Task 7 ✓ |
| §7.2 Multi-Doc Agent 骨架 | Task 8 ✓ |
| §8 物理产出（含 `_common.py`、`memory_sessions.json`、三节重写） | Tasks 1-8 ✓ |
| §9 验收标准 | Task 9 Steps 1~5 ✓ |

无遗漏。

**2. Placeholder 检查**

- 所有代码 step 都给出了完整可运行代码（无 "TBD" / "implement later"）
- 所有命令都给出了具体路径与期望输出
- 所有断言都有具体阈值（如"路由命中 ≥ 4/6"、"代码量减少 ≥ 25%"）

**3. 类型与命名一致性**

- `simple_eval_2pt` 在 Tasks 2、4、5、7 中签名一致：`(llm_answer, expected_answer, question='', *, prompt_template=None) -> int`
- `run_shared_eval` 在 Tasks 2、4、5 中签名一致：`(answer_fn, qna_dict, *, eval_prompt_template=None) -> pd.DataFrame`
- `run_session_eval` 在 Tasks 3、7、8 中签名一致：`(system, qna_dict, *, eval_prompt_template=None) -> pd.DataFrame`
- `build_compare_table` 在 Tasks 2、4、5、7、8 中签名一致：`(dfs, names) -> pd.DataFrame`
- `Step` 在 Task 5、（如 6.3 重用则在 Task 7/8）中字段一致：`kind: str`、`payload: dict`
- `MemoryRAGSystem.last_debug` 与 `MultiDocAgent.last_debug` 都用 dict 暴露调试信息，inspect 函数依赖一致

无不一致。
