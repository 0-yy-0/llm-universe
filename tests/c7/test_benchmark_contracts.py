"""Contracts for the C7 same-question retrieval benchmark."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
COURSE = ROOT / "notebook" / "C7 高级 RAG 技巧"
SCRIPT = COURSE / "scripts" / "run_benchmark.py"
CHECKER_SCRIPT = COURSE / "scripts" / "check_tutorial.py"

spec = importlib.util.spec_from_file_location("c7_run_benchmark", SCRIPT)
assert spec and spec.loader
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)

checker_spec = importlib.util.spec_from_file_location("c7_tutorial_checker", CHECKER_SCRIPT)
assert checker_spec and checker_spec.loader
checker = importlib.util.module_from_spec(checker_spec)
checker_spec.loader.exec_module(checker)

if str(COURSE) not in sys.path:
    sys.path.insert(0, str(COURSE))

from common.dataset import load_dataset  # noqa: E402
from common.eval_utils import recall_at_k  # noqa: E402


def test_explicit_query_batch_and_methods_are_shared():
    report = benchmark.run_benchmark()
    config = report["config"]

    assert tuple(config["query_ids"]) == benchmark.DEFAULT_QUERY_IDS
    assert config["query_scope"] == "explicit canonical demo/regression set"
    assert config["methods"] == list(benchmark.METHODS)
    assert config["baseline"] == "bm25"
    assert len(config["methods"]) in (2, 3)

    for query_id in config["query_ids"]:
        methods = report["questions"][query_id]["methods"]
        assert list(methods) == list(benchmark.METHODS)
        assert all(
            len(metrics["retrieved_evidence_ids"]) == config["top_k"]
            for metrics in methods.values()
        )
        assert all(
            metrics["top_k"] == config["top_k"]
            and metrics["char_budget"] == config["char_budget"]
            for metrics in methods.values()
        )


def test_metrics_are_computed_from_canonical_qrels_and_rank():
    package = load_dataset()
    report = benchmark.run_benchmark()
    evidence_rows = benchmark._retrieval_evidence(package)
    evidence_by_id = {row["evidence_id"]: row for row in evidence_rows}

    for query_id, query_result in report["questions"].items():
        relevant = {
            row["evidence_id"]
            for row in package["qrels"]
            if row["query_id"] == query_id and row["relevance"] == 1
        }
        assert relevant
        for metrics in query_result["methods"].values():
            ids = metrics["retrieved_evidence_ids"]
            flags = [evidence_id in relevant for evidence_id in ids]
            expected_recall = recall_at_k(
                flags, total_relevant=len(relevant), k=report["config"]["top_k"]
            )
            assert metrics["coverage"] == pytest.approx(
                len(set(ids) & relevant) / len(relevant)
            )
            assert metrics["recall_at_k"] == pytest.approx(expected_recall)
            rank = next(
                (position for position, evidence_id in enumerate(ids, 1) if evidence_id in relevant),
                None,
            )
            assert metrics["first_relevant_rank"] == rank
            assert metrics["mrr"] == pytest.approx(0.0 if rank is None else 1.0 / rank)
            assert all(evidence_id in evidence_by_id for evidence_id in ids)


def test_saved_compare_notebook_matches_current_local_run():
    """The saved Notebook must describe this exact local benchmark run."""

    current = benchmark.run_benchmark()
    saved = checker._saved_benchmark_reports(
        COURSE / "7. 评估" / "比较改动前后.ipynb"
    )
    assert len(saved) == 1
    benchmark.validate_report(saved[0])
    assert checker._benchmark_reports_equal(current, saved[0])


def test_retrieval_projection_does_not_include_answer_annotations():
    package = load_dataset()
    query_rows = benchmark._selected_queries(package, benchmark.DEFAULT_QUERY_IDS)
    evidence_rows = benchmark._retrieval_evidence(package)

    assert all(set(row) == {"query_id", "text"} for row in query_rows)
    assert all(set(row) == {"evidence_id", "page", "quote"} for row in evidence_rows)
    assert not any("reference_answer" in row for row in query_rows)
    assert not any("expected_pages" in row for row in query_rows)
    assert not any("reference_claims" in row for row in query_rows)


def test_cch_and_baseline_use_one_search_budget():
    package = load_dataset()
    evidence_rows = benchmark._retrieval_evidence(package)
    budget = 31
    for evidence in evidence_rows:
        for method in benchmark.METHODS:
            value = benchmark.compose_index_text(evidence, method, budget)
            assert 0 < len(value) <= budget

    report = benchmark.run_benchmark(top_k=2, char_budget=budget)
    assert report["config"]["top_k"] == 2
    assert report["config"]["char_budget"] == budget
    benchmark.validate_report(report)


def test_entrypoint_finds_course_from_repository_or_course_root():
    assert benchmark.find_course_root(ROOT) == COURSE.resolve()
    assert benchmark.find_course_root(COURSE) == COURSE.resolve()

    root_run = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    course_run = subprocess.run(
        [sys.executable, "scripts/run_benchmark.py", "--json"],
        cwd=COURSE,
        capture_output=True,
        text=True,
        check=True,
    )
    for output in (root_run.stdout, course_run.stdout):
        parsed = json.loads(output)
        assert parsed["validation"]["same_query_batch_for_all_methods"]


def test_missing_dataset_fails_without_fallback(tmp_path: Path):
    with pytest.raises(RuntimeError, match="canonical dataset"):
        benchmark.run_benchmark(dataset_root=tmp_path / "missing")
