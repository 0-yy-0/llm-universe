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
