"""C7 LLM-as-Judge calibration notebook contracts."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).parents[2]
COURSE = ROOT / "notebook" / "C7 高级 RAG 技巧"
NOTEBOOK = COURSE / "7. 评估" / "用模型辅助检查回答.ipynb"


def _expanded_cell() -> dict:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return next(
        cell for cell in notebook.cells if cell.get("id") == "expanded-judge-calibration"
    )


def _expanded_source() -> str:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    cell_ids = {
        "expanded-judge-setup",
        "expanded-judge-contract",
        "expanded-judge-calibration",
        "expanded-judge-statistics",
    }
    return "\n".join(
        "".join(cell.source)
        for cell in notebook.cells
        if cell.get("id") in cell_ids
    )


def _expanded_output() -> str:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    output_cell_ids = {"expanded-judge-calibration", "expanded-judge-statistics"}
    return "".join(
        "".join(item.get("text", []))
        for cell in notebook.cells
        if cell.get("id") in output_cell_ids
        for item in cell.outputs
        if item.get("output_type") == "stream"
    )


def _quote_matcher(evidence_by_id: dict[str, dict]):
    tree = ast.parse(_expanded_source())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"quote_match_text", "quote_overlaps_evidence"}
    ]
    namespace = {
        "evidence_by_id": evidence_by_id,
        "unicodedata": __import__("unicodedata"),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace["quote_overlaps_evidence"]


def _batch_validator():
    tree = ast.parse(_expanded_source())
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_complete_calibration_batch"
    )
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace["validate_complete_calibration_batch"]


def test_expanded_calibration_source_has_ten_author_labeled_cases_and_no_fallback() -> None:
    source = _expanded_source()

    assert 'EXPANDED_MODEL = "glm-4-flash"' in source
    assert 'assert len(calibration_samples) == 10' in source
    assert '"label_source": "tutorial_author_annotation"' in source
    assert '"human_verified": False' in source
    assert '"max_retries": 0' in source
    assert "不进行应用层重试或 fallback" in source
    assert "SequenceMatcher" not in source
    assert "if candidate in canonical" in source
    assert "if invalid_quotes:" in source
    assert 'or not claim["evidence"]' in source
    assert "原始返回先原样记录" not in source  # this explanation belongs to the markdown cell
    for category in ("correct", "partial", "unsupported", "citation_mismatch", "should_refuse"):
        assert f'"{category}"' in source or f'("{category}"' in source


def test_expanded_calibration_binds_only_canonical_evidence_and_persists_real_results() -> None:
    cell = _expanded_cell()
    source = _expanded_source()
    evidence_ids = {
        row["evidence_id"]
        for row in (
            json.loads(line)
            for line in (COURSE / "data" / "dataset" / "evidence.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        )
    }
    referenced_ids = {
        token.strip('"')
        for token in source.split('"')
        if token.startswith("evi_") and token != "evi_..."
    }
    assert referenced_ids <= evidence_ids

    output = _expanded_output()
    assert "固定模型 glm-4-flash" in output
    assert "实际调用尝试次数： 10" in output
    assert "成功拿到原始返回： 10" in output
    assert "成功解析： 10 / 10" in output
    assert output.count("原始模型返回：") == 10
    assert "score 混淆计数" in output
    assert "限制：样本来自同一本《南瓜书》" in output


def test_saved_judge_expectations_predictions_and_reported_rates_are_consistent() -> None:
    output = _expanded_output()
    expectations = [
        json.loads(line.split("： ", 1)[1])
        for line in output.splitlines()
        if line.startswith("教程预设期望（human_verified=false）： ")
    ]
    predictions = [
        json.loads(line.split("： ", 1)[1])
        for line in output.splitlines()
        if line.startswith("结构化判断： ")
    ]
    assert len(expectations) == len(predictions) == 10

    score_matches = sum(
        expected["score"] == predicted["score"]
        for expected, predicted in zip(expectations, predictions)
    )
    full_matches = sum(
        expected["score"] == predicted["score"]
        and all(
            expected[field] == predicted[field]
            for field in ("needs_refusal", "answer_action", "citation_ok")
        )
        for expected, predicted in zip(expectations, predictions)
    )
    assert score_matches == 5
    assert full_matches == 3
    assert "分数一致率：50.00%" in output
    assert "完整协议一致率：30.00%" in output


def test_calibration_gold_contract_distinguishes_question_policy_from_actual_answer_action() -> None:
    source = _expanded_source()

    cal_01 = source[source.index('"cal-01"') : source.index('"cal-02"')]
    cal_09 = source[source.index('"cal-09"') : source.index('"cal-10"')]
    cal_10 = source[source.index('"cal-10"') : source.index("assert len(calibration_samples)")]
    assert 'False,\n        "answer"' in cal_01
    assert 'True,\n        "refuse"' in cal_09
    assert 'True,\n        "answer"' in cal_10
    assert cal_10.count("expected_citation_ok=False") == 1


def test_quote_verification_rejects_prefixed_fabrication_and_checks_every_quote() -> None:
    matcher = _quote_matcher({"e1": {"quote": "南瓜书中的完整原文事实。"}})

    assert matcher("完整原文事实。", ["e1"])
    assert not matcher("完整原文事实，然后追加一段虚构结论。", ["e1"])
    quotes = ["完整原文事实。", "完全虚构的第二条引文。"]
    assert not all(matcher(quote, ["e1"]) for quote in quotes)


def test_statistics_rejects_incomplete_or_failed_calibration_batch() -> None:
    validate = _batch_validator()
    samples = [{"sample_id": f"cal-{index:02d}"} for index in range(1, 11)]
    results = [
        {"sample_id": sample["sample_id"], "raw": "{}", "parsed": {}, "error": None}
        for sample in samples
    ]
    validate(results, samples)

    broken = [dict(record) for record in results]
    broken[4]["parsed"] = None
    broken[4]["error"] = "parse failed"
    with pytest.raises(RuntimeError, match="禁止对子集继续计分"):
        validate(broken, samples)

    with pytest.raises(RuntimeError, match="同序、完整"):
        validate(results[:-1], samples)
