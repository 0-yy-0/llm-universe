"""Canonical C7 dataset contract tests."""

from __future__ import annotations

import importlib.util
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
COURSE = ROOT / "notebook" / "C7 高级 RAG 技巧"
DATASET = COURSE / "data" / "dataset"

spec = importlib.util.spec_from_file_location(
    "c7_check_dataset", COURSE / "scripts" / "check_dataset.py"
)
assert spec and spec.loader
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

import sys

if str(COURSE) not in sys.path:
    sys.path.insert(0, str(COURSE))

from common.dataset import (  # noqa: E402
    DatasetContractError,
    chapter_title,
    load_dataset,
    load_annotation,
    load_query_catalog,
    load_query_only,
    load_search_evidence,
    validate_dataset,
)
import common.dataset as dataset_store  # noqa: E402
from common.eval_utils import load_pdf_pages  # noqa: E402


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _copy_dataset(tmp_path: Path) -> Path:
    target = tmp_path / "dataset"
    shutil.copytree(DATASET, target)
    return target


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows)
        + "\n",
        encoding="utf-8",
    )


def _refresh_manifest_sha256(target: Path, file_key: str) -> None:
    """Keep a mutated fixture otherwise valid so the asserted issue is specific."""

    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest["files"][file_key]
    path = target / metadata["path"]
    metadata["sha256"] = checker.dataset_store.sha256_file(path)
    if path.suffix == ".jsonl" and isinstance(metadata.get("record_count"), int):
        metadata["record_count"] = len(_jsonl(path))
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_embedding_builder():
    """Load the C2 build module without invoking its network/tokenizer paths."""

    build_spec = importlib.util.spec_from_file_location(
        "c7_build_embedding_training_data_contract",
        COURSE / "scripts" / "build_embedding_training_data.py",
    )
    assert build_spec and build_spec.loader
    module = importlib.util.module_from_spec(build_spec)
    build_spec.loader.exec_module(module)
    return module


def _load_prepare_module():
    """Load the preparation entry point without running a refresh."""

    prepare_spec = importlib.util.spec_from_file_location(
        "c7_prepare_dataset_contract",
        COURSE / "scripts" / "prepare_dataset.py",
    )
    assert prepare_spec and prepare_spec.loader
    module = importlib.util.module_from_spec(prepare_spec)
    prepare_spec.loader.exec_module(module)
    return module


def _mutate_pdf_provenance(target: Path, mutation: str) -> None:
    evidence_path = target / "evidence.jsonl"
    evidence = _jsonl(evidence_path)
    if mutation == "nontraining_null_offsets":
        row = next(
            item for item in evidence if item["evidence_id"] == "evi_91d0b1e7d0ff"
        )
        row["quote"] = "这段文字并非南瓜书原文"
        row["quote_normalized"] = row["quote"]
        row["offsets"] = None
    elif mutation == "training_page_out_of_range":
        pairs = _jsonl(target / "finetune_pairs.jsonl")
        pair = next(item for item in pairs if item["split"] == "train")
        row = next(
            item
            for item in evidence
            if item["evidence_id"] == pair["positive_evidence_id"]
        )
        row["page"] = 999
        pair["page"] = 999
        _write_jsonl(target / "finetune_pairs.jsonl", pairs)
        _refresh_manifest_sha256(target, "finetune_pairs")
    elif mutation == "document_path_opt_out":
        documents_path = target / "documents.jsonl"
        documents = _jsonl(documents_path)
        documents[0]["path"] = "data/not_the_book.pdf"
        _write_jsonl(documents_path, documents)
        _refresh_manifest_sha256(target, "documents")
    elif mutation == "candidate_source_path_opt_out":
        candidates_path = target / "annotations" / "qa_candidates.jsonl"
        candidates = _jsonl(candidates_path)
        candidates[0]["source_path"] = "data/not_the_book.pdf"
        _write_jsonl(candidates_path, candidates)
        _refresh_manifest_sha256(target, "qa_candidates")
    else:  # pragma: no cover - protects the test helper itself
        raise AssertionError(f"unknown mutation: {mutation}")
    _write_jsonl(evidence_path, evidence)
    _refresh_manifest_sha256(target, "evidence")


def test_canonical_counts_and_current_usage_policy():
    package = load_dataset()
    manifest = package["manifest"]
    counts = package["manifest"]["counts"]
    assert len(package["documents"]) == 1
    for key in ("evidence", "queries", "qrels", "qa_candidates", "review_decisions", "finetune_pairs"):
        assert len(package[key]) == counts[key]
    pair_counts = Counter(row["split"] for row in package["finetune_pairs"])
    for split in ("train", "dev", "test"):
        assert len(package["splits"]["assignments"][split]) == pair_counts[split]
    assert pair_counts["train"] >= 32
    assert pair_counts["dev"] >= 8
    assert pair_counts["test"] >= 8
    assert len(package["splits"]["assignments"]["regression"]) == 70
    current = set(package["manifest"]["usage_policy"]["current_case_ids"])
    assert len(current) == 70
    current_queries = [query for query in package["queries"] if query["query_id"] in current]
    assert len(current_queries) == 70
    assert manifest["usage_policy"]["semantic_training"]["human_verified"] is False
    assert manifest["semantic_audit"]["human_verified"] is False
    assert "candidate_target_counts" not in manifest["split_protocol"]
    assert manifest["split_protocol"]["candidate_actual_counts"] == dict(pair_counts)
    for query in current_queries:
        assert {"demo", "dev", "regression"}.issubset(query["usage"])
        assert {"finetune", "test"}.issubset(query["excluded_from"])


def test_candidate_review_contains_only_final_verified_rows():
    candidates = _jsonl(DATASET / "annotations" / "qa_candidates.jsonl")
    package = load_dataset()
    builder = _load_embedding_builder()
    exclusions = builder.load_secondary_review_exclusions()
    split_counts = Counter(row["split"] for row in package["finetune_pairs"])
    assert len(candidates) == package["manifest"]["counts"]["qa_candidates"]
    assert all(row["query"].strip() for row in candidates)
    assert sum(row["training_eligible"] for row in candidates) == split_counts["train"]
    assert sum(not row["training_eligible"] for row in candidates) == (
        split_counts["dev"] + split_counts["test"]
    )
    assert all(row["human_verified"] is False for row in candidates)
    expected_exclusion_count = package["manifest"]["data_construction"][
        "secondary_review_exclusion_count"
    ]
    assert len(exclusions) == expected_exclusion_count
    assert set(exclusions).isdisjoint({row["candidate_id"] for row in candidates})
    assert expected_exclusion_count == 110
    for row in candidates:
        assert row["status"] == "approved"
        assert row["review_status"] == "verified"
        assert row["evidence_ids"]


def test_query_and_annotation_projections_are_separate():
    package = load_dataset()
    query_rows = load_query_catalog()
    assert len(query_rows) == len(package["queries"])
    current = set(package["manifest"]["usage_policy"]["current_case_ids"])
    assert current <= {case["id"] for case in query_rows}
    assert {case["id"] for case in query_rows} == {
        query["query_id"]
        for query in package["queries"]
        if query["query_id"] in set(case["id"] for case in query_rows)
    }
    assert all(set(row) == {"id", "query"} for row in query_rows)
    annotation = load_annotation("cross_validation_reliability")
    assert annotation["expected_pages"] == [19]
    assert annotation["essential_evidence_spans"]


def test_query_controls_are_runtime_only_and_annotations_keep_evaluation():
    forbidden = {
        "evaluation",
        "citation_pages",
        "should_retrieve",
        "supported_answer_required",
        "reference_answer",
        "expected_pages",
        "reference_claims",
        "source_pages",
        "oracle",
    }

    def assert_no_forbidden(value):
        if isinstance(value, dict):
            assert forbidden.isdisjoint(value)
            for nested in value.values():
                assert_no_forbidden(nested)
        elif isinstance(value, list):
            for nested in value:
                assert_no_forbidden(nested)

    query_rows = _jsonl(DATASET / "queries.jsonl")
    for query in query_rows:
        controls = dataset_store.load_query_controls(query["query_id"])
        assert_no_forbidden(controls)
        expected_evaluation = query.get("interaction", {}).get("evaluation", {})
        annotation = load_annotation(query["query_id"])
        assert annotation["evaluation"] == expected_evaluation


def test_search_evidence_projection_does_not_load_full_package(monkeypatch):
    def fail_full_package(*args, **kwargs):
        raise AssertionError("search projection must not call load_dataset")

    monkeypatch.setattr(dataset_store, "load_dataset", fail_full_package)
    query_rows = load_query_only(["gaussian_mean_estimate"])
    rows = load_search_evidence()
    assert query_rows == [
        {
            "id": "gaussian_mean_estimate",
            "query": query_rows[0]["query"],
        }
    ]
    assert rows
    assert all(set(row) == {"evidence_id", "page", "quote"} for row in rows)


def test_chapter_title_reads_documents_without_full_package(monkeypatch):
    def fail_full_package(*args, **kwargs):
        raise AssertionError("chapter lookup must not call load_dataset")

    dataset_store._chapter_ranges.cache_clear()
    monkeypatch.setattr(dataset_store, "load_dataset", fail_full_package)
    try:
        ranges = dataset_store.load_chapter_ranges()
        assert ranges
        first = ranges[0]
        assert chapter_title(first["start"]) == first["title"]
    finally:
        dataset_store._chapter_ranges.cache_clear()


def test_retrieval_notebooks_keep_annotations_post_hoc():
    background = json.loads(
        (COURSE / "1. 背景" / "为什么基础 RAG 还会答错.ipynb").read_text(
            encoding="utf-8"
        )
    )
    background_code = "\n".join(
        "".join(cell.get("source", []))
        for cell in background["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "load_query_only" in background_code
    assert "load_search_evidence" in background_code
    assert "load_query_records(" not in background_code
    assert "load_evidence_records(" not in background_code
    assert "all_ranked = search" in background_code
    assert background_code.index("all_ranked = search") < background_code.index("qrels = [")

    generation = json.loads(
        (COURSE / "5. 生成阶段" / "排序、压缩与回答.ipynb").read_text(
            encoding="utf-8"
        )
    )
    cross_encoder_code = next(
        "".join(cell.get("source", []))
        for cell in generation["cells"]
        if cell.get("cell_type") == "code"
        and "RERANKER_MODEL_ID" in "".join(cell.get("source", []))
    )
    rerank_complete = cross_encoder_code.index("cross_encoder_ranked = [")
    label_reads = [
        cross_encoder_code.index("load_query_records()"),
        cross_encoder_code.index("load_evidence_records()"),
        cross_encoder_code.index("load_annotation("),
    ]
    assert all(rerank_complete < index for index in label_reads)
    assert rerank_complete < cross_encoder_code.index("reference_claims")
    assert rerank_complete < cross_encoder_code.index("assert canonical_evidence_ids")


def test_checker_accepts_canonical_package():
    assert checker.check_dataset(DATASET) == []


def test_checker_rejects_empty_training_query(tmp_path):
    target = _copy_dataset(tmp_path)
    queries_path = target / "queries.jsonl"
    queries = _jsonl(queries_path)
    splits = json.loads((target / "splits.json").read_text(encoding="utf-8"))
    train_id = splits["assignments"]["train"][0]
    query = next(row for row in queries if row["query_id"] == train_id)
    query["text"] = "   "
    _write_jsonl(queries_path, queries)
    issues = checker.check_dataset(target)
    assert any("空训练问题" in issue for issue in issues)


def test_checker_rejects_dangling_qrel_reference(tmp_path):
    target = _copy_dataset(tmp_path)
    qrels_path = target / "qrels.jsonl"
    qrels = _jsonl(qrels_path)
    qrels[0]["evidence_id"] = "evi_missing"
    _write_jsonl(qrels_path, qrels)
    issues = checker.check_dataset(target)
    assert any("dangling evidence ref" in issue for issue in issues)


def test_checker_rejects_duplicate_query_id(tmp_path):
    target = _copy_dataset(tmp_path)
    queries_path = target / "queries.jsonl"
    queries = _jsonl(queries_path)
    queries.append(dict(queries[0]))
    _write_jsonl(queries_path, queries)
    issues = checker.check_dataset(target)
    assert any("重复 query_id" in issue for issue in issues)


def test_checker_rejects_group_split_leakage(tmp_path):
    target = _copy_dataset(tmp_path)
    splits_path = target / "splits.json"
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    first = splits["assignments"]["train"][0]
    second = splits["assignments"]["test"][0]
    splits["groups"][second]["section_id"] = splits["groups"][first]["section_id"]
    splits_path.write_text(json.dumps(splits), encoding="utf-8")
    issues = checker.check_dataset(target)
    assert any("section split 泄漏" in issue for issue in issues)


def test_checker_rejects_evidence_that_is_not_exact_pdf_text(tmp_path):
    target = _copy_dataset(tmp_path)
    evidence_path = target / "evidence.jsonl"
    rows = _jsonl(evidence_path)
    rows[0]["quote"] = "这不是 PDF 原文"
    rows[0]["offsets"] = {
        "start": 0,
        "end": len(rows[0]["quote"]),
        "unit": "normalized_characters",
        "status": "exact",
    }
    _write_jsonl(evidence_path, rows)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["evidence"]["sha256"] = checker.dataset_store.sha256_file(evidence_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    issues = checker.check_dataset(target)
    assert any("不是声明页码和 offsets 上的逐字 PDF 原文" in issue for issue in issues)


@pytest.mark.parametrize(
    "mutation, expected_issue",
    [
        ("nontraining_null_offsets", "offsets 必须是对象"),
        ("training_page_out_of_range", "page 超出 canonical PDF 页范围"),
    ],
)
def test_pdf_provenance_tampering_is_rejected_by_loader_and_checker(
    tmp_path, mutation, expected_issue
):
    target = _copy_dataset(tmp_path)
    _mutate_pdf_provenance(target, mutation)

    issues = validate_dataset(target)
    assert any(expected_issue in issue for issue in issues)
    with pytest.raises(DatasetContractError, match="canonical PDF|offsets|page"):
        load_dataset(target)
    checker_issues = checker.check_dataset(target)
    assert any(expected_issue in issue for issue in checker_issues)


@pytest.mark.parametrize(
    "mutation", ["document_path_opt_out", "candidate_source_path_opt_out"]
)
def test_document_path_opt_out_is_rejected_by_all_entry_points(tmp_path, mutation):
    target = _copy_dataset(tmp_path)
    _mutate_pdf_provenance(target, mutation)

    issues = validate_dataset(target)
    assert any("canonical document" in issue for issue in issues)
    with pytest.raises(DatasetContractError, match="canonical document"):
        load_dataset(target)
    checker_issues = checker.check_dataset(target)
    assert any("canonical document" in issue for issue in checker_issues)
    prepare = _load_prepare_module()
    with pytest.raises(DatasetContractError, match="canonical document"):
        prepare.prepare_dataset(target)


def test_checker_rejects_query_family_leakage_even_when_sections_differ(tmp_path):
    target = _copy_dataset(tmp_path)
    splits_path = target / "splits.json"
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    train_id = splits["assignments"]["train"][0]
    test_id = splits["assignments"]["test"][0]
    splits["groups"][test_id]["query_family_id"] = splits["groups"][train_id]["query_family_id"]
    splits_path.write_text(json.dumps(splits), encoding="utf-8")
    issues = checker.check_dataset(target)
    assert any("query family split 泄漏" in issue for issue in issues)


def test_fixed_canonical_chunks_are_bounded_and_bind_complete_pdf_text():
    """Fixed chunks must cover their source pages and bind every supervision view."""

    package = load_dataset(DATASET)
    fixed = [
        row
        for row in package["evidence"]
        if row.get("evidence_type") == "fixed_token_chunk"
    ]
    assert fixed, "canonical package must contain the fixed-token C2 chunks"
    fixed_by_id = {row["evidence_id"]: row for row in fixed}
    pdf_pages = {int(row["page"]): row["text"] for row in load_pdf_pages()}

    chunks_by_page: dict[int, list[dict]] = {}
    for row in fixed:
        token_count = row.get("token_count")
        token_budget = row.get("token_budget")
        assert isinstance(token_count, int) and not isinstance(token_count, bool)
        assert isinstance(token_budget, int) and not isinstance(token_budget, bool)
        assert 0 < token_count <= token_budget < 512
        assert row["chunk_id"] == row["evidence_id"]
        offsets = row["offsets"]
        assert offsets["status"] == "exact"
        assert offsets["unit"] == "normalized_characters"
        assert offsets["end"] - offsets["start"] == len(row["quote"])
        assert row["quote"] == pdf_pages[row["page"]][offsets["start"] : offsets["end"]]
        chunks_by_page.setdefault(int(row["page"]), []).append(row)

    # Splitting is page-local and lossless: no gaps, overlaps, or omitted tail.
    for page, rows in chunks_by_page.items():
        cursor = 0
        for row in sorted(rows, key=lambda item: item["offsets"]["start"]):
            offsets = row["offsets"]
            assert offsets["start"] == cursor
            cursor = offsets["end"]
        assert cursor == len(pdf_pages[page])

    for pair in package["finetune_pairs"]:
        positive = fixed_by_id[pair["positive_evidence_id"]]
        assert pair["positive"] == positive["quote"]
        assert pair["page"] == positive["page"]
        assert pair["section_id"] == positive["section_id"]
        assert pair["positive_token_count"] == positive["token_count"]
        assert pair["token_budget"] == positive["token_budget"]

    # Candidate support and the training positive are the same canonical text,
    # rather than independently generated answer/context strings.
    for candidate in package["qa_candidates"]:
        evidence_ids = candidate.get("evidence_ids")
        assert isinstance(evidence_ids, list) and len(evidence_ids) == 1
        positive = fixed_by_id[evidence_ids[0]]
        assert candidate["support_quotes"] == [positive["quote"]]
        verification = candidate["verification"]
        assert verification["quote_exact"] is True
        assert verification["support_quotes_exact"] is True


def test_checker_rejects_candidate_support_quote_drift(tmp_path):
    target = _copy_dataset(tmp_path)
    candidates_path = target / "annotations" / "qa_candidates.jsonl"
    candidates = _jsonl(candidates_path)
    candidate = next(row for row in candidates if row.get("evidence_ids"))
    candidate["support_quotes"] = ["这不是 canonical chunk 原文"]
    _write_jsonl(candidates_path, candidates)
    _refresh_manifest_sha256(target, "qa_candidates")

    issues = checker.check_dataset(target)
    assert any(
        "support_quotes" in issue or "support quote" in issue or "支持引用" in issue
        for issue in issues
    )


def test_checker_rejects_normalized_query_conflict(tmp_path):
    target = _copy_dataset(tmp_path)
    queries_path = target / "queries.jsonl"
    queries = _jsonl(queries_path)
    assert queries[0]["query_id"] != queries[1]["query_id"]
    queries[0]["text"] = "KKT 条件是什么？"
    queries[1]["text"] = "什么是KKT条件"
    _write_jsonl(queries_path, queries)
    _refresh_manifest_sha256(target, "queries")

    # The conflict rule is part of the fixed-chunk/v3 contract.  Make the
    # fixture explicitly v3 even when a pre-rebuild checkout is under test.
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("data_construction", {})["review_type"] = (
        "model_assisted_semantic_training_v3"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    issues = checker.check_dataset(target)
    assert any("规范化 query 冲突" in issue for issue in issues)


@pytest.mark.parametrize("mutation", ["legacy_status", "failed_status", "false"])
def test_checker_rejects_review_status_or_semantic_tampering(tmp_path, mutation):
    target = _copy_dataset(tmp_path)
    pairs_path = target / "finetune_pairs.jsonl"
    pairs = _jsonl(pairs_path)
    pair = next(row for row in pairs if isinstance(row.get("semantic_review"), dict))
    audit = pair["semantic_review"]
    if mutation == "legacy_status":
        pair["review_status"] = "automated_verified"
    elif mutation == "failed_status":
        pair["review_status"] = "failed"
    else:
        audit["positive_fully_supports"] = False
    _write_jsonl(pairs_path, pairs)
    _refresh_manifest_sha256(target, "finetune_pairs")

    issues = checker.check_dataset(target)
    assert any("未通过对应版本语义核验" in issue for issue in issues)


@pytest.mark.parametrize("location", ["cache_metadata", "top_level"])
def test_embedding_cache_fingerprint_mismatch_is_rejected(tmp_path, location):
    builder = _load_embedding_builder()
    source = {
        "evidence_id": "evi_test_chunk",
        "source_page_id": "source_test_page",
        "page": 1,
        "section_id": "doc:page_0001",
        "start": 0,
        "end": 4,
        "token_count": 4,
        "token_budget": 384,
        "text": "测试原文",
        "split": "train",
    }
    metadata = builder.cache_metadata([source], existing_query_norms={"已有问题"})
    partial = builder._new_partial(metadata)
    if location == "cache_metadata":
        partial["cache_metadata"]["source_chunks_fingerprint"] = "0" * 64
    else:
        partial["source_chunks_fingerprint"] = "0" * 64
    cache_path = tmp_path / "generated.partial.json"
    cache_path.write_text(
        json.dumps(partial, ensure_ascii=False), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="不匹配"):
        builder._load_or_init_partial(cache_path, metadata)
