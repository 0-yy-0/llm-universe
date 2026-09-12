"""C6 缺口补查与端到端原始响应审计的独立契约测试；不调用模型。"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
COURSE = ROOT / "notebook" / "C7 高级 RAG 技巧"
CHECKER = COURSE / "7. 评估" / "check_capstone.py"
CAPSTONE = COURSE / "7. 评估" / "端到端验收.ipynb"
CRAG = COURSE / "6. 处理信息缺口" / "检查检索结果后再继续.ipynb"
if str(COURSE) not in sys.path:
    sys.path.insert(0, str(COURSE))
spec = importlib.util.spec_from_file_location("c7_capstone_audit_checker", CHECKER)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def saved_notebook():
    return json.loads(CAPSTONE.read_text(encoding="utf-8"))


def audit_of(notebook):
    return next(
        output["data"][checker.AUDIT_MIME]
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if checker.AUDIT_MIME in output.get("data", {})
    )


def update_raw_artifact(notebook, stage, case_id, raw):
    """同步原始调用 artifact，确保测试进一步检查 raw/派生字段，而非只比副本。"""
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            call = output.get("data", {}).get(checker.RAW_MIME)
            if call and call["stage"] == stage and call["query_id"] == case_id:
                call["raw_response"] = raw
                return
    raise AssertionError("missing raw call")


def test_saved_capstone_replays_real_retrieval_raw_responses_and_gap():
    audit = checker.validate_notebook(saved_notebook())
    gap = next(row for row in audit["cases"] if row["query_id"] == checker.GAP_CASE_ID)
    repair = gap["gap_repair"]
    assert repair["triggered"]
    assert repair["first_round_missing_essential_ids"]
    assert repair["new_evidence"]
    assert repair["retrieval_rounds"]
    assert gap["initial_top_k"] == 1
    assert gap["initial_context_chars"] <= audit["context_budget_chars"]
    assert gap["context_chars"] <= audit["context_budget_chars"]
    for case in audit["cases"]:
        assert case["raw_generation_response"].strip()
        assert case["raw_semantic_response"].strip()
        # 质量通过与否由运行决定，完整性测试不强制覆盖真实负结果。
        assert case["verdict"] in {"pass_answerable", "pass_insufficient", "fail"}


@pytest.mark.parametrize("field", ["raw_generation_response", "raw_semantic_response"])
def test_missing_model_original_cannot_be_replaced_by_derived_flags(field):
    notebook = saved_notebook()
    del audit_of(notebook)["cases"][0][field]
    with pytest.raises((AssertionError, ValueError), match="原文|原始"):
        checker.validate_notebook(notebook)


def test_generation_original_is_reparsed_even_with_valid_evidence_ids():
    notebook = saved_notebook()
    case = audit_of(notebook)["cases"][0]
    generated = checker.parse_strict_json(case["raw_generation_response"])
    generated["answer"] = "所有机器学习算法都存在绝对最优者。"
    raw = json.dumps(generated, ensure_ascii=False)
    case["raw_generation_response"] = raw
    update_raw_artifact(notebook, "generation", case["query_id"], raw)
    with pytest.raises(AssertionError, match="answer.*原文"):
        checker.validate_notebook(notebook)


def test_reviewer_original_is_reparsed_instead_of_trusting_semantic_pass():
    notebook = saved_notebook()
    case = audit_of(notebook)["cases"][0]
    reviewed = checker.parse_strict_json(case["raw_semantic_response"])
    reviewed["claims"][0]["relation"] = "contradicted"
    reviewed["verdict"] = "fail"
    raw = json.dumps(reviewed, ensure_ascii=False)
    case["raw_semantic_response"] = raw
    update_raw_artifact(notebook, "semantic_review", case["query_id"], raw)
    with pytest.raises(AssertionError, match="semantic_claims.*原文"):
        checker.validate_notebook(notebook)


def test_claimed_success_cannot_override_reviewer_failure():
    with pytest.raises(ValueError, match="pass.*未支持"):
        checker.validate_semantic_review(
            json.dumps({
                "claims": [{"claim": "答案没有支持", "relation": "not_found", "evidence_ids": []}],
                "verdict": "pass",
                "reason": "伪造通过",
            }, ensure_ascii=False),
            {"evi_example"}, "answerable",
        )


@pytest.mark.parametrize("field", ["context_chars", "initial_context_chars"])
def test_budget_numbers_are_recomputed_from_exact_context(field):
    notebook = saved_notebook()
    audit_of(notebook)["cases"][0][field] += 1
    with pytest.raises(AssertionError, match="字符"):
        checker.validate_notebook(notebook)


def test_canonical_qrels_and_hydrated_quotes_are_not_trusted_from_audit():
    for field in ("essential_ids", "hydrated_citations"):
        notebook = saved_notebook()
        case = audit_of(notebook)["cases"][0]
        if field == "essential_ids":
            case[field] = []
        else:
            case[field][0]["quote"] += "这是补写的内容。"
        with pytest.raises(AssertionError, match="canonical|essential"):
            checker.validate_notebook(notebook)


def test_gap_query_and_new_evidence_have_to_replay_from_actual_model_output():
    for attack in ("query", "retrieved_ids", "new_evidence"):
        notebook = saved_notebook()
        gap = next(row for row in audit_of(notebook)["cases"] if row["gap_repair"])
        repair = gap["gap_repair"]
        if attack == "query":
            repair["retrieval_rounds"][0]["query"] = "人工偷偷补入的答案语义"
        elif attack == "retrieved_ids":
            repair["retrieval_rounds"][0]["retrieved_ids"].reverse()
        else:
            repair["new_evidence"][0]["quote"] += "伪造原文"
        with pytest.raises(AssertionError, match="补查"):
            checker.validate_notebook(notebook)


def test_gap_trigger_follows_raw_sufficient_and_cannot_be_forced():
    notebook = saved_notebook()
    gap = next(row for row in audit_of(notebook)["cases"] if row["gap_repair"])
    gap["gap_repair"]["triggered"] = False
    with pytest.raises(AssertionError, match="触发条件"):
        checker.validate_notebook(notebook)


def test_honest_quality_negative_is_preserved_and_verified():
    notebook = saved_notebook()
    audit = audit_of(notebook)
    case = audit["cases"][0]
    raw_generation = json.dumps({
        "status": "insufficient", "answer": checker.INSUFFICIENT_ANSWER, "citations": [],
    }, ensure_ascii=False)
    raw_review = json.dumps({
        "claims": [], "verdict": "pass", "reason": "回答保持固定资料不足声明。",
    }, ensure_ascii=False)
    case.update(
        raw_generation_response=raw_generation, response_status="insufficient",
        answer=checker.INSUFFICIENT_ANSWER, citation_ids=[], hydrated_citations=[],
        qrels_positive_citation_ids=[], raw_semantic_response=raw_review,
        semantic_claims=[], semantic_verdict="pass", semantic_reason="回答保持固定资料不足声明。",
        verdict="fail",
        failure_reasons=["model_returned_insufficient", "essential_not_cited", "unsupported_answer_claims"],
    )
    update_raw_artifact(notebook, "generation", case["query_id"], raw_generation)
    update_raw_artifact(notebook, "semantic_review", case["query_id"], raw_review)
    audit["all_passed"] = False
    result = checker.validate_notebook(notebook)
    assert result["all_passed"] is False
    assert result["cases"][0]["verdict"] == "fail"
    # 同一份负结果不能只修改总开关就被宣布通过。
    audit["all_passed"] = True
    with pytest.raises(AssertionError, match="all_passed"):
        checker.validate_notebook(notebook)


def test_honest_untriggered_gap_is_a_valid_negative_audit():
    notebook = saved_notebook()
    audit = audit_of(notebook)
    gap = next(row for row in audit["cases"] if row["query_id"] == checker.GAP_CASE_ID)
    evidence = {
        row["evidence_id"]: row
        for row in (
            json.loads(line)
            for line in (COURSE / "data/dataset/evidence.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    first_id = gap["initial_context_evidence_ids"][0]
    raw_reflection = json.dumps({
        "sufficient": True,
        "missing": [],
        "repair_queries": [],
        "supporting_quotes": [evidence[first_id]["quote"]],
    }, ensure_ascii=False)
    repair = gap["gap_repair"]
    repair.update(
        raw_reflection=raw_reflection,
        reflection=checker.validate_gap_reflection(raw_reflection, [evidence[first_id]]),
        triggered=False,
        trigger_reason="reflection.sufficient=true",
        retrieval_rounds=[],
        first_round_missing_essential_ids=[
            item for item in gap["essential_ids"]
            if item not in gap["initial_context_evidence_ids"]
        ],
        new_evidence=[],
        gap_closed=False,
    )
    update_raw_artifact(notebook, "gap_reflection", gap["query_id"], raw_reflection)
    gap["retrieved_ids"] = list(gap["initial_retrieved_ids"])
    gap["context_evidence_ids"] = list(gap["initial_context_evidence_ids"])
    gap["context_chars"] = gap["initial_context_chars"]

    raw_generation = json.dumps({
        "status": "insufficient", "answer": checker.INSUFFICIENT_ANSWER, "citations": [],
    }, ensure_ascii=False)
    raw_review = json.dumps({
        "claims": [], "verdict": "pass", "reason": "回答保持固定资料不足声明。",
    }, ensure_ascii=False)
    gap.update(
        raw_generation_response=raw_generation,
        response_status="insufficient",
        answer=checker.INSUFFICIENT_ANSWER,
        citation_ids=[],
        hydrated_citations=[],
        qrels_positive_citation_ids=[],
        raw_semantic_response=raw_review,
        semantic_claims=[],
        semantic_verdict="pass",
        semantic_reason="回答保持固定资料不足声明。",
        verdict="fail",
        failure_reasons=[
            "model_returned_insufficient", "essential_not_retrieved",
            "essential_not_in_context", "essential_not_cited", "unsupported_answer_claims",
        ],
    )
    update_raw_artifact(notebook, "generation", gap["query_id"], raw_generation)
    update_raw_artifact(notebook, "semantic_review", gap["query_id"], raw_review)
    audit["all_passed"] = False

    result = checker.validate_notebook(notebook)
    assert result["all_passed"] is False
    assert next(row for row in result["cases"] if row["query_id"] == checker.GAP_CASE_ID)["verdict"] == "fail"


@pytest.mark.parametrize("raw", [
    '{"status":"answerable","status":"insufficient","answer":"x","citations":[]}',
    '{"x":NaN}',
    '{"x":1} and a second answer',
    '["not", "an", "object"]',
])
def test_raw_parser_rejects_duplicate_fields_non_json_constants_and_extra_text(raw):
    with pytest.raises(ValueError):
        checker.parse_strict_json(raw)


def test_gap_reflection_cannot_fabricate_a_supporting_quote():
    raw = json.dumps({
        "sufficient": True, "missing": [], "repair_queries": [],
        "supporting_quotes": ["只存在于问题中的结论"],
    }, ensure_ascii=False)
    with pytest.raises(ValueError, match="绑定"):
        checker.validate_gap_reflection(raw, [{"quote": "本轮真实检索证据"}])


def test_crag_scope_and_manual_answer_hint_are_explicit():
    notebook = json.loads(CRAG.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "受 CRAG 启发的本地纠错" in source
    assert "https://arxiv.org/abs/2401.15884" in source
    assert "知识精炼" in source and "外部搜索" in source
    assert "manual_teaching_hint_with_answer_semantics" in source
    assert "人工补充词，并含有" in source
    assert "运行 Notebook 不联网" not in source
    assert "assert needs_correction" not in source
    assert "max_retries=0" in source and "需要网络" in source
    audits = [
        output["data"][checker.AUDIT_MIME]
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if checker.AUDIT_MIME in output.get("data", {})
    ]
    manual = next(row for row in audits if row.get("repair_query_source"))
    assert manual["repair_query_source"] == "manual_teaching_hint_with_answer_semantics"
    assert manual["manual_hint_limitation"]
