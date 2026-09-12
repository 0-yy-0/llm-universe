"""Negative contracts for C2 generated-item and semantic-audit caches."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
COURSE = ROOT / "notebook" / "C7 高级 RAG 技巧"


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "c7_build_embedding_training_data_integrity",
        COURSE / "scripts" / "build_embedding_training_data.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cache_fixture():
    builder = _load_builder()
    source = {
        "evidence_id": "evi_test_chunk",
        "source_page_id": "source_test_page",
        "doc_id": "doc_pumpkin_book",
        "page": 1,
        "section_id": "doc:page_0001",
        "start": 0,
        "end": 6,
        "token_count": 4,
        "token_budget": 384,
        "text": "测试原文支持答案。",
        "split": "train",
    }
    raw = {
        "type": "definition",
        "question": "测试对象是什么？",
        "answer": "测试对象是原文中的对象。",
        "claims": ["测试对象是原文中的对象。"],
        "support_chunk_ids": [source["evidence_id"]],
    }
    item = builder._normalise_item(
        raw,
        source,
        reserved_query_norms=set(),
        local_query_norms=set(),
    )
    audit = {
        "item_id": item["item_id"],
        "input_hash": builder._audit_input_hash(item, source),
        "question_self_contained": True,
        "positive_fully_supports": True,
        "answer_fully_supported": True,
        "support_quotes_sufficient": True,
    }
    return builder, source, item, audit


@pytest.mark.parametrize("field", ["question", "answer", "claims", "support_quotes", "item_id"])
def test_cached_item_mutations_are_rejected(cache_fixture, field):
    builder, source, item, _ = cache_fixture
    mutated = copy.deepcopy(item)
    if field == "question":
        mutated[field] = "篡改后的问题？"
    elif field == "answer":
        mutated[field] = "篡改后的答案。"
    elif field == "claims":
        mutated[field] = ["篡改后的断言。"]
    elif field == "support_quotes":
        mutated[field] = ["篡改后的引文。"]
    else:
        mutated[field] = "train_item_tampered"

    with pytest.raises(ValueError):
        builder._validate_cached_items([mutated], [source], reserved_query_norms=set())


def test_cached_item_hash_mutation_is_rejected(cache_fixture):
    builder, source, item, _ = cache_fixture
    mutated = copy.deepcopy(item)
    mutated["input_hash"] = "0" * 64
    with pytest.raises(ValueError):
        builder._validate_cached_items([mutated], [source], reserved_query_norms=set())


def test_cached_audit_hash_mutation_is_rejected(cache_fixture):
    builder, source, item, audit = cache_fixture
    mutated = copy.deepcopy(audit)
    mutated["input_hash"] = "0" * 64
    with pytest.raises(ValueError):
        builder._validate_cached_audits(
            [item],
            {item["item_id"]: mutated},
            {source["evidence_id"]: source},
            require_complete=True,
        )


def test_cached_audit_source_identity_mutation_is_rejected(cache_fixture):
    builder, source, item, audit = cache_fixture
    mutated_source = copy.deepcopy(source)
    mutated_source["start"] += 1
    with pytest.raises(ValueError):
        builder._validate_cached_audits(
            [item],
            {item["item_id"]: audit},
            {source["evidence_id"]: mutated_source},
            require_complete=True,
        )
