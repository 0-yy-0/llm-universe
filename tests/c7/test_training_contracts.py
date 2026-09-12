from __future__ import annotations

import copy
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
COURSE_ROOT = REPO_ROOT / "notebook" / "C7 高级 RAG 技巧"
if str(COURSE_ROOT) not in sys.path:
    sys.path.insert(0, str(COURSE_ROOT))

from common.dataset import load_dataset  # noqa: E402
from common.training_utils import (  # noqa: E402
    build_unique_positive_batches,
    compare_retrieval,
    evaluate_embeddings,
    load_finetune_pairs,
    train_embedding_model,
    TrainingContractError,
    validate_finetune_pairs,
)


def test_finetune_pairs_are_canonical_query_to_evidence_chunks() -> None:
    package = load_dataset(COURSE_ROOT / "data" / "dataset")
    pairs = load_finetune_pairs(
        COURSE_ROOT / "data" / "dataset" / "finetune_pairs.jsonl",
        evidence_rows=package["evidence"],
        query_rows=package["queries"],
    )
    evidence = {row["evidence_id"]: row for row in package["evidence"]}
    assert len(pairs) == package["manifest"]["counts"]["finetune_pairs"]
    assert {row["split"] for row in pairs} == {"train", "dev", "test"}
    assert {row["split"] for row in pairs}.issubset({"train", "dev", "test"})
    assert all("answer" not in row and "reference_answer" not in row for row in pairs)
    assert all(row["positive"] == evidence[row["positive_evidence_id"]]["quote"] for row in pairs)
    assert all(row["human_verified"] is False for row in pairs)


def test_candidate_review_and_grouped_split_counts() -> None:
    package = load_dataset(COURSE_ROOT / "data" / "dataset")
    pairs = load_finetune_pairs(
        COURSE_ROOT / "data" / "dataset" / "finetune_pairs.jsonl",
        evidence_rows=package["evidence"],
        query_rows=package["queries"],
    )
    by_split = {split: [row for row in pairs if row["split"] == split] for split in ("train", "dev", "test")}
    split_counts = {split: len(rows) for split, rows in by_split.items()}
    expected = package["manifest"]["data_construction"]["retained_pair_count_by_split"]
    assert split_counts == expected
    assert split_counts["train"] >= 32
    assert split_counts["dev"] >= 8
    assert split_counts["test"] >= 8
    pages_by_split = {
        split: {row["page"] for row in rows} for split, rows in by_split.items()
    }
    assert not pages_by_split["train"] & pages_by_split["dev"]
    assert not pages_by_split["train"] & pages_by_split["test"]
    assert not pages_by_split["dev"] & pages_by_split["test"]
    assert sum(row["training_eligible"] is True for row in package["qa_candidates"]) == split_counts["train"]
    assert all(row["status"] == "approved" for row in package["qa_candidates"])
    assert all(row["human_verified"] is False for row in package["qa_candidates"])
    assert all(row["human_verified"] is False for row in package["review_decisions"])


def test_semantic_audit_covers_all_splits_without_human_claims() -> None:
    package = load_dataset(COURSE_ROOT / "data" / "dataset")
    pairs = load_finetune_pairs(
        COURSE_ROOT / "data" / "dataset" / "finetune_pairs.jsonl",
        evidence_rows=package["evidence"],
        query_rows=package["queries"],
    )
    split_counts = Counter(row["split"] for row in pairs)
    assert dict(split_counts) == package["manifest"]["data_construction"]["retained_pair_count_by_split"]
    assert Counter(row["semantic_review"]["status"] for row in pairs) == {"passed": len(pairs)}
    assert all(
        row["review_type"] == "model_assisted_semantic_training_v3"
        and row["human_verified"] is False
        and row["semantic_review"]["review_type"] == "model_assisted_semantic_training_v3"
        and row["semantic_review"]["question_self_contained"] is True
        and row["semantic_review"]["positive_fully_supports"] is True
        and row["semantic_review"]["answer_fully_supported"] is True
        and row["semantic_review"]["support_quotes_sufficient"] is True
        and row["semantic_review"]["human_verified"] is False
        for row in pairs
    )
    train = [row for row in pairs if row["split"] == "train"]
    assert len(train) == split_counts["train"]
    batches = build_unique_positive_batches(train, batch_size=16, seed=42)
    assert sorted(index for batch in batches for index in batch) == list(range(len(train)))
    assert all(
        len({train[index]["positive_evidence_id"] for index in batch}) == len(batch)
        for batch in batches
    )

    candidates = {row["candidate_id"]: row for row in package["qa_candidates"]}
    decisions = {row["candidate_id"]: row for row in package["review_decisions"]}
    queries = {row["query_id"]: row for row in package["queries"]}
    qrels = {
        (row["query_id"], row["evidence_id"]): row for row in package["qrels"]
    }
    for pair in pairs:
        candidate_id = pair["candidate_id"]
        review = pair["semantic_review"]
        assert candidates[candidate_id]["review_type"] == review["review_type"]
        assert candidates[candidate_id]["semantic_review"] == review
        assert decisions[candidate_id]["review_type"] == review["review_type"]
        assert decisions[candidate_id]["semantic_review"] == review
        assert queries[pair["query_id"]]["review_method"] == review["review_type"]
        qrel = qrels[(pair["query_id"], pair["positive_evidence_id"])]
        assert qrel["review_type"] == review["review_type"]
        assert qrel["semantic_review_status"] == review["status"]

    # Quality gates may legitimately remove every retained example of one
    # optional type; do not keep a bad row merely to fill a category quota.
    query_types = {row["query_type"] for row in package["qa_candidates"]}
    assert query_types
    assert query_types <= {"definition", "mechanism", "comparison", "condition"}


def test_retrieval_metrics_and_per_query_comparison() -> None:
    query_ids = ["q1", "q2"]
    evidence_ids = ["e1", "e2", "e3"]
    query_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    baseline_evidence = np.asarray([[1.0, 0.0], [0.0, 0.8], [0.0, 1.0]])
    improved_evidence = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    qrels = [
        {"query_id": "q1", "evidence_id": "e1", "relevance": 1},
        {"query_id": "q2", "evidence_id": "e2", "relevance": 1},
    ]
    baseline = evaluate_embeddings(
        query_ids, query_embeddings, evidence_ids, baseline_evidence, qrels
    )
    after = evaluate_embeddings(
        query_ids, query_embeddings, evidence_ids, improved_evidence, qrels
    )
    assert baseline["Recall@1"] == 0.5
    assert after["Recall@1"] == 0.5
    comparison = compare_retrieval(baseline, after)
    assert comparison["degraded_count"] == 1
    assert comparison["improved_count"] == 1
    assert set(comparison["degraded_query_ids"]) | set(comparison["improved_query_ids"]) == {"q1", "q2"}


def test_recall_at_k_counts_all_relevant_evidence() -> None:
    """Recall@k must divide by every qrel, not collapse to Hit@k."""

    metrics = evaluate_embeddings(
        ["q1"],
        np.asarray([[1.0, 0.0]]),
        ["e1", "e2", "e3"],
        np.asarray([[1.0, 0.0], [0.9, 0.0], [0.0, 1.0]]),
        [
            {"query_id": "q1", "evidence_id": "e1", "relevance": 1},
            {"query_id": "q1", "evidence_id": "e2", "relevance": 1},
        ],
        top_ks=(1, 2),
    )
    assert metrics["Recall@1"] == 0.5
    assert metrics["Recall@2"] == 1.0


def test_compare_retrieval_reports_multi_qrel_tradeoff() -> None:
    """A gain at one cutoff must not hide a loss at another cutoff."""

    before = {
        "per_query": [
            {
                "query_id": "q1",
                "first_relevant_rank": 2,
                "recall_at_k": {"1": 0.0, "3": 1.0},
            }
        ]
    }
    after = {
        "per_query": [
            {
                "query_id": "q1",
                "first_relevant_rank": 1,
                "recall_at_k": {"1": 0.5, "3": 0.5},
            }
        ]
    }

    comparison = compare_retrieval(before, after)
    assert comparison["tradeoff_count"] == 1
    assert comparison["tradeoff_query_ids"] == ["q1"]
    assert comparison["improved_count"] == 0
    assert comparison["degraded_count"] == 0
    assert comparison["per_query"][0]["status"] == "tradeoff"


def test_mnrl_rejects_a_known_multi_positive_as_a_silent_negative() -> None:
    """An unsafe batch must fail instead of training on a false negative."""

    pairs = [
        {"query_id": "q1", "positive_evidence_id": "e1"},
        {"query_id": "q2", "positive_evidence_id": "e2"},
    ]
    known_positive_ids = {
        "q1": {"e1", "e2"},  # e2 is also a known positive for q1
        "q2": {"e2"},
    }
    with pytest.raises(TrainingContractError, match="已知正例"):
        build_unique_positive_batches(
            pairs,
            batch_size=2,
            seed=0,
            known_positive_ids_by_query=known_positive_ids,
        )


def test_mnrl_batches_keep_known_positive_sets_disjoint() -> None:
    pairs = [
        {"query_id": "q1", "positive_evidence_id": "e1"},
        {"query_id": "q2", "positive_evidence_id": "e3"},
        {"query_id": "q3", "positive_evidence_id": "e4"},
    ]
    known_positive_ids = {
        "q1": {"e1", "e2"},
        "q2": {"e3"},
        "q3": {"e4"},
    }
    batches = build_unique_positive_batches(
        pairs,
        batch_size=3,
        seed=0,
        known_positive_ids_by_query=known_positive_ids,
    )
    for batch in batches:
        for left_index, left in enumerate(batch):
            left_known = known_positive_ids[pairs[left]["query_id"]]
            for right in batch[left_index + 1 :]:
                assert left_known.isdisjoint({pairs[right]["positive_evidence_id"]})


@pytest.mark.parametrize("mutation", ["legacy_status", "failed_status", "false"])
def test_training_validator_rejects_review_status_or_semantic_tampering(mutation):
    package = load_dataset(COURSE_ROOT / "data" / "dataset")
    pair = next(
        row
        for row in package["finetune_pairs"]
        if isinstance(row.get("semantic_review"), dict)
    )
    mutated = copy.deepcopy(pair)
    if mutation == "legacy_status":
        mutated["review_status"] = "automated_verified"
    elif mutation == "failed_status":
        mutated["review_status"] = "failed"
    else:
        mutated["semantic_review"]["answer_fully_supported"] = False

    with pytest.raises(TrainingContractError, match="语义核验"):
        validate_finetune_pairs(
            [mutated],
            evidence_rows=package["evidence"],
            query_rows=package["queries"],
        )


def test_training_refuses_nonempty_output_dir_without_reuse(tmp_path):
    output_dir = tmp_path / "training-output"
    output_dir.mkdir()
    sentinel = output_dir / "existing-model-marker"
    sentinel.write_text("keep", encoding="utf-8")
    minimal_pair = {
        "pair_id": "pair_test",
        "query_id": "query_test",
        "query": "测试问题",
        "positive": "测试原文",
        "positive_evidence_id": "evidence_test",
        "page": 1,
        "section_id": "doc:page_0001",
        "query_family_id": "family_test",
        "split": "train",
        "review_status": "automated_verified",
        "human_verified": False,
    }

    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        train_embedding_model(
            [minimal_pair],
            output_dir,
            batch_size=2,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert sorted(path.name for path in output_dir.iterdir()) == [sentinel.name]
