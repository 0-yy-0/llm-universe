"""C7 LLM-as-Judge calibration notebook contracts."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).parents[2]
COURSE = ROOT / "notebook" / "C7 高级 RAG 技巧"
NOTEBOOK = COURSE / "7. 评估" / "用模型辅助检查回答.ipynb"


def _expanded_cell() -> dict:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return next(
        cell for cell in notebook.cells if cell.get("id") == "expanded-judge-calibration"
    )


def _quote_matcher(evidence_by_id: dict[str, dict]):
    tree = ast.parse("".join(_expanded_cell().source))
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


def test_expanded_calibration_source_has_ten_author_labeled_cases_and_no_fallback() -> None:
    source = "".join(_expanded_cell().source)

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
    source = "".join(cell.source)
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

    assert cell.outputs
    output = "".join(
        "".join(item.get("text", []))
        for item in cell.outputs
        if item.get("output_type") == "stream"
    )
    assert "固定模型 glm-4-flash" in output
    assert "实际调用尝试次数： 10" in output
    assert "成功拿到原始返回： 10" in output
    assert "成功解析： 10 / 10" in output
    assert output.count("原始模型返回：") == 10
    assert "score 混淆计数" in output
    assert "限制：样本来自同一本《南瓜书》" in output


def test_calibration_gold_contract_distinguishes_question_policy_from_actual_answer_action() -> None:
    source = "".join(_expanded_cell().source)

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
