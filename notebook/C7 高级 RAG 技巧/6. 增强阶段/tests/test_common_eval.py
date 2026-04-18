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
