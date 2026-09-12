#!/usr/bin/env python3
"""Validate the C7 canonical dataset package.

The checker is deliberately strict: a missing file, malformed record, empty
training question, dangling reference, duplicate identifier, or group split
leak makes the command fail with a non-zero exit status.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


COURSE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COURSE_ROOT.parents[1]
if str(COURSE_ROOT) not in sys.path:
    sys.path.insert(0, str(COURSE_ROOT))

from common import dataset as dataset_store  # noqa: E402


DATASET_ROOT = dataset_store.DATASET_ROOT


def _check_file_integrity(root: Path, issues: list[str]) -> None:
    """Check manifest file paths and checksums without relying on old data."""

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        issues.append(f"manifest.json 格式错误：{exc}")
        return
    if not isinstance(manifest, dict):
        issues.append("manifest.json 必须是对象")
        return
    files = manifest.get("files")
    if not isinstance(files, dict):
        issues.append("manifest.files 必须是对象")
        return
    for name, metadata in files.items():
        if not isinstance(metadata, dict) or not isinstance(metadata.get("path"), str):
            issues.append(f"manifest.files.{name} 缺少 path")
            continue
        path = root / metadata["path"]
        if not path.is_file():
            issues.append(f"canonical 文件缺失：{path}")
            continue
        expected = metadata.get("sha256")
        if isinstance(expected, str):
            try:
                actual = dataset_store.sha256_file(path)
            except OSError as exc:
                issues.append(f"无法读取 canonical 文件 {path}：{exc}")
                continue
            if actual != expected:
                issues.append(f"文件 SHA-256 不匹配：{path}")
        count = metadata.get("record_count")
        if isinstance(count, int):
            try:
                actual_count = len(dataset_store.read_jsonl(path))
            except (OSError, ValueError) as exc:
                issues.append(f"无法读取 canonical JSONL {path}：{exc}")
                continue
            if actual_count != count:
                issues.append(f"文件记录数不匹配：{path}（manifest={count}，实际={actual_count}）")


def _check_pdf_source(package: Mapping[str, Any], issues: list[str]) -> None:
    """Require every canonical-PDF evidence row to be an exact PDF span."""

    # Keep source/hash/page/offset/quote rules in one validator so CLI checks,
    # dataset loading, preparation, and training cannot disagree.
    issues.extend(dataset_store.validate_pdf_provenance(package))


def _check_empty_training_queries(
    package: Mapping[str, Any], issues: list[str]
) -> None:
    queries = {
        row.get("query_id"): row
        for row in package.get("queries", [])
        if isinstance(row, dict)
    }
    splits = package.get("splits", {})
    assignments = splits.get("assignments", {}) if isinstance(splits, dict) else {}
    training_ids = assignments.get("train", []) if isinstance(assignments, dict) else []
    for query_id in training_ids:
        row = queries.get(query_id)
        if row is None:
            continue
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            issues.append(f"空训练问题：{query_id}")
        if "finetune" in set(row.get("excluded_from", [])):
            issues.append(f"排除项 query 不得进入 train：{query_id}")
        if set(row.get("usage", [])) & {"demo", "regression"}:
            issues.append(f"demo/regression query 不得进入 train：{query_id}")
    for row in package.get("qa_candidates", []):
        if not isinstance(row, dict) or not row.get("training_eligible"):
            continue
        if not str(row.get("query", "")).strip():
            issues.append(f"空训练候选：{row.get('candidate_id')}")


def _check_fixed_chunk_contract(
    package: Mapping[str, Any], issues: list[str]
) -> None:
    """Check that v3 training/evaluation records share bounded chunks."""

    construction = package.get("manifest", {}).get("data_construction", {})
    evidence_by_id = {
        row.get("evidence_id"): row
        for row in package.get("evidence", [])
        if isinstance(row, dict)
    }
    # Candidate support must be the same canonical fixed chunk as the
    # training positive. validate_dataset checks evidence IDs but used to miss
    # drift in this parallel support_quotes field. Keep this check outside the
    # v3 gate so it also catches drift while a package is being migrated.
    for row in package.get("qa_candidates", []):
        if not isinstance(row, dict):
            continue
        evidence_ids = row.get("evidence_ids")
        if not isinstance(evidence_ids, list) or len(evidence_ids) != 1:
            continue
        positive = evidence_by_id.get(evidence_ids[0])
        if not isinstance(positive, dict):
            continue
        support_quotes = row.get("support_quotes")
        if not isinstance(support_quotes, list) or not support_quotes:
            issues.append(
                "候选 support_quotes 缺失或不是非空列表："
                f"{row.get('candidate_id')}"
            )
            continue
        source_quote = positive.get("quote")
        if not isinstance(source_quote, str):
            continue
        if any(
            not isinstance(quote, str)
            or not quote.strip()
            or quote not in source_quote
            for quote in support_quotes
        ):
            issues.append(
                "候选 support_quotes 不是 evidence 原文片段："
                f"{row.get('candidate_id')}"
            )
        if (
            positive.get("evidence_type") == "fixed_token_chunk"
            and support_quotes != [source_quote]
        ):
            issues.append(
                "fixed chunk 候选 support_quotes 未绑定完整 chunk 原文："
                f"{row.get('candidate_id')}"
            )
    review_type = "model_assisted_semantic_training_v3"
    review_status = "model_assisted_semantically_verified"
    required_semantic_checks = (
        "question_self_contained",
        "positive_fully_supports",
        "answer_fully_supported",
        "support_quotes_sufficient",
    )
    if not isinstance(construction, dict) or construction.get("review_type") != review_type:
        return
    fixed_ids: set[str] = set()
    for row in package.get("evidence", []):
        if not isinstance(row, dict) or row.get("evidence_type") != "fixed_token_chunk":
            continue
        evidence_id = row.get("evidence_id")
        fixed_ids.add(str(evidence_id))
        token_count, token_budget = row.get("token_count"), row.get("token_budget")
        if (
            not isinstance(token_count, int)
            or isinstance(token_count, bool)
            or not isinstance(token_budget, int)
            or isinstance(token_budget, bool)
            or token_count <= 0
            or token_count > token_budget
            or token_budget > 384
            or token_budget >= 512
        ):
            issues.append(f"fixed chunk token budget 无效：{evidence_id}")
        if row.get("chunk_id") != evidence_id:
            issues.append(f"fixed chunk chunk_id 不绑定 evidence_id：{evidence_id}")
    normalized: dict[str, str] = {}
    for row in package.get("queries", []):
        if not isinstance(row, dict):
            continue
        query_id, text = row.get("query_id"), row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        key = dataset_store.normalize_query(text)
        previous = normalized.get(key)
        if previous is not None and previous != query_id:
            issues.append(
                f"规范化 query 冲突：{previous!r} 与 {query_id!r} 均归一为 {key!r}"
            )
        normalized[key] = str(query_id)
    for row in package.get("finetune_pairs", []):
        if not isinstance(row, dict):
            continue
        evidence_id = row.get("positive_evidence_id")
        positive = evidence_by_id.get(evidence_id)
        if not isinstance(positive, dict) or positive.get("evidence_type") != "fixed_token_chunk":
            issues.append(f"训练 positive 不是 fixed_token_chunk：{row.get('pair_id')}")
            continue
        if row.get("positive") != positive.get("quote"):
            issues.append(f"训练 positive 与 fixed chunk 不一致：{row.get('pair_id')}")
        if row.get("positive_token_count") != positive.get("token_count"):
            issues.append(f"训练 positive token_count 不一致：{row.get('pair_id')}")
        if row.get("token_budget") != positive.get("token_budget"):
            issues.append(f"训练 token_budget 不一致：{row.get('pair_id')}")
        review = row.get("semantic_review")
        if (
            row.get("review_status") != review_status
            or row.get("review_type") != review_type
            or row.get("human_verified") is not False
            or not isinstance(review, dict)
            or review.get("status") != "passed"
            or review.get("review_type") != review_type
            or review.get("human_verified") is not False
            or row.get("supervision_type") != "semantic_query_evidence"
            or any(
                review.get(key) is not True for key in required_semantic_checks
            )
        ):
            issues.append(f"训练 pair 缺少真实通过的语义审核：{row.get('pair_id')}")
    expected_budget = construction.get("chunk_token_budget")
    if expected_budget != 384:
        issues.append(f"manifest data_construction.chunk_token_budget 应为 384，实际 {expected_budget!r}")


def _check_dangling_references(
    package: Mapping[str, Any], issues: list[str]
) -> None:
    doc_ids = {row.get("doc_id") for row in package.get("documents", []) if isinstance(row, dict)}
    evidence_ids = {
        row.get("evidence_id") for row in package.get("evidence", []) if isinstance(row, dict)
    }
    query_ids = {row.get("query_id") for row in package.get("queries", []) if isinstance(row, dict)}
    for row in package.get("evidence", []):
        if isinstance(row, dict) and row.get("doc_id") not in doc_ids:
            issues.append(f"dangling doc ref：{row.get('evidence_id')} -> {row.get('doc_id')}")
    for row in package.get("qrels", []):
        if not isinstance(row, dict):
            continue
        if row.get("query_id") not in query_ids:
            issues.append(f"dangling query ref：{row.get('query_id')}")
        if row.get("evidence_id") not in evidence_ids:
            issues.append(f"dangling evidence ref：{row.get('evidence_id')}")
    for row in package.get("queries", []):
        if not isinstance(row, dict):
            continue
        for claim in row.get("reference_claims", []):
            if not isinstance(claim, dict):
                continue
            for evidence_id in claim.get("evidence_ids", []):
                if evidence_id not in evidence_ids:
                    issues.append(f"dangling claim evidence ref：{row.get('query_id')} -> {evidence_id}")
    for row in package.get("qa_candidates", []):
        if not isinstance(row, dict):
            continue
        for evidence_id in row.get("evidence_ids", []):
            if evidence_id not in evidence_ids:
                issues.append(f"dangling candidate evidence ref：{row.get('candidate_id')} -> {evidence_id}")


def _check_duplicate_ids(package: Mapping[str, Any], issues: list[str]) -> None:
    for name, field in dataset_store.ID_FIELDS.items():
        if not field:
            continue
        seen: set[Any] = set()
        for row in package.get(name, []):
            if not isinstance(row, dict):
                continue
            value = row.get(field)
            if value in seen:
                issues.append(f"重复 ID：{name}.{field}={value}")
            seen.add(value)


def _check_group_split_leakage(
    package: Mapping[str, Any], issues: list[str]
) -> None:
    splits = package.get("splits", {})
    if not isinstance(splits, dict):
        return
    assignments = splits.get("assignments", {})
    groups = splits.get("groups", {})
    if not isinstance(assignments, dict) or not isinstance(groups, dict):
        return
    section_to_split: dict[Any, str] = {}
    family_to_split: dict[Any, str] = {}
    evidence_to_split: dict[str, str] = {}
    evidence_by_query: dict[Any, set[str]] = {}
    for row in package.get("qrels", []):
        if isinstance(row, dict):
            evidence_by_query.setdefault(row.get("query_id"), set()).add(row.get("evidence_id"))
    query_to_split: dict[Any, str] = {}
    for split in ("train", "dev", "test"):
        for query_id in assignments.get(split, []):
            previous = query_to_split.get(query_id)
            if previous is not None and previous != split:
                issues.append(f"query 重复分配：{query_id}（{previous}/{split}）")
            query_to_split[query_id] = split
            group = groups.get(query_id)
            if not isinstance(group, dict):
                continue
            section_id = group.get("section_id")
            family_id = group.get("query_family_id")
            previous_split = section_to_split.get(section_id)
            if previous_split is not None and previous_split != split:
                issues.append(f"section split 泄漏：{section_id!r}（{previous_split}/{split}）")
            section_to_split[section_id] = split
            previous_split = family_to_split.get(family_id)
            if previous_split is not None and previous_split != split:
                issues.append(f"query family split 泄漏：{family_id!r}（{previous_split}/{split}）")
            family_to_split[family_id] = split
            for evidence_id in evidence_by_query.get(query_id, set()):
                previous_evidence_split = evidence_to_split.get(evidence_id)
                if previous_evidence_split is not None and previous_evidence_split != split:
                    issues.append(
                        f"evidence split 泄漏：{evidence_id}（{previous_evidence_split}/{split}）"
                    )
                evidence_to_split[evidence_id] = split


def check_dataset(root: str | Path | None = None) -> list[str]:
    """Return all issues; an empty list means the package is valid."""

    package_root = Path(root) if root is not None else DATASET_ROOT
    issues: list[str] = []
    _check_file_integrity(package_root, issues)
    issues.extend(dataset_store.validate_dataset(package_root))
    if issues and not (package_root / "manifest.json").is_file():
        return issues
    try:
        package = dataset_store._read_package(package_root)  # type: ignore[attr-defined]
    except (FileNotFoundError, dataset_store.DatasetContractError) as exc:
        return issues + [str(exc)]
    _check_empty_training_queries(package, issues)
    _check_dangling_references(package, issues)
    _check_duplicate_ids(package, issues)
    _check_group_split_leakage(package, issues)
    _check_fixed_chunk_contract(package, issues)
    _check_pdf_source(package, issues)
    return list(dict.fromkeys(issues))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_ROOT)
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    issues = check_dataset(args.dataset)
    if args.json:
        print(json.dumps({"ok": not issues, "issues": issues}, ensure_ascii=False, indent=2))
    elif issues:
        for issue in issues:
            print(f"ERROR: {issue}", file=sys.stderr)
        print(f"dataset check failed: {len(issues)} issue(s)", file=sys.stderr)
    else:
        print("dataset check passed")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
