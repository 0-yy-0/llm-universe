#!/usr/bin/env python3
"""Finalize the current C7 canonical dataset after intentional record edits.

The command validates every declared JSON/JSONL file in an isolated copy, then
refreshes record counts and SHA-256 values in ``manifest.json``. It never reads
Git history, discovers an alternate dataset, or substitutes missing inputs.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


COURSE_ROOT = Path(__file__).resolve().parents[1]
if str(COURSE_ROOT) not in sys.path:
    sys.path.insert(0, str(COURSE_ROOT))

from common import dataset as dataset_store  # noqa: E402


DEFAULT_DATASET = COURSE_ROOT / "data" / "dataset"


class PreparationError(ValueError):
    """The requested canonical package cannot be finalized safely."""


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"manifest 不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparationError(f"manifest 不是有效 JSON：{path}") from exc
    if not isinstance(value, dict):
        raise PreparationError("manifest 必须是对象")
    if value.get("source_of_truth") != "canonical_dataset":
        raise PreparationError("manifest.source_of_truth 必须为 canonical_dataset")
    if not isinstance(value.get("files"), dict) or not value["files"]:
        raise PreparationError("manifest.files 必须是非空对象")
    return value


def _declared_path(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise PreparationError(f"manifest.files.{label}.path 为空")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise PreparationError(f"manifest.files.{label}.path 越出数据目录") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"canonical 文件不存在：{candidate}")
    return candidate


def build_refreshed_manifest(
    root: Path, *, dataset_version: str | None = None
) -> dict[str, Any]:
    """Return a refreshed manifest without mutating the package."""

    manifest = copy.deepcopy(_load_manifest(root))
    counts: dict[str, int] = {}
    rows_by_name: dict[str, list[dict[str, Any]]] = {}
    for name, metadata in manifest["files"].items():
        if not isinstance(metadata, dict):
            raise PreparationError(f"manifest.files.{name} 必须是对象")
        path = _declared_path(root, metadata.get("path"), name)
        if path.suffix == ".jsonl":
            rows = dataset_store.read_jsonl(path)
            record_count = len(rows)
            metadata["record_count"] = record_count
            counts[name] = record_count
            rows_by_name[name] = rows
        elif path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PreparationError(f"canonical JSON 格式错误：{path}") from exc
            metadata.pop("record_count", None)
        else:
            raise PreparationError(f"不支持的 canonical 文件类型：{path}")
        metadata["sha256"] = dataset_store.sha256_file(path)
    manifest["counts"] = counts
    _refresh_summary_metadata(manifest, rows_by_name, dataset_root=root)
    if dataset_version is not None:
        if not dataset_version.strip():
            raise PreparationError("dataset_version 不能为空")
        manifest["dataset_version"] = dataset_version.strip()
    return manifest


def _validate_candidate(root: Path, manifest: dict[str, Any]) -> None:
    """Validate the new manifest against a disposable package copy."""

    with tempfile.TemporaryDirectory(prefix="c7-dataset-") as temporary:
        candidate_root = Path(temporary) / "dataset"
        shutil.copytree(root, candidate_root)
        (candidate_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        dataset_store.load_dataset(candidate_root)


def _most_common_string(rows: list[dict[str, Any]], field: str) -> str | None:
    values = [row.get(field) for row in rows]
    strings = [value for value in values if isinstance(value, str) and value]
    if not strings:
        return None
    return Counter(strings).most_common(1)[0][0]


def _refresh_summary_metadata(
    manifest: dict[str, Any],
    rows_by_name: dict[str, list[dict[str, Any]]],
    *,
    dataset_root: Path,
) -> None:
    """Refresh count summaries that describe the canonical JSONL records.

    ``manifest.files`` and ``manifest.counts`` are not the only places where
    this package records counts.  The v4 rebuild intentionally keeps the
    legacy tutorial cases alongside the C2 semantic pairs, so stale v2
    summaries can otherwise look plausible while describing a different
    experiment.  Derive only fields whose source rows are present; unrelated
    tutorial metadata remains untouched.
    """

    candidates = rows_by_name.get("qa_candidates", [])
    pairs = rows_by_name.get("finetune_pairs", [])
    evidence = rows_by_name.get("evidence", [])
    pair_counts = {
        split: sum(row.get("split") == split for row in pairs)
        for split in ("train", "dev", "test")
    }
    pair_pages = {
        split: len(
            {
                row.get("page")
                for row in pairs
                if row.get("split") == split and isinstance(row.get("page"), int)
            }
        )
        for split in ("train", "dev", "test")
    }
    review_type = _most_common_string(pairs, "review_type") or _most_common_string(
        candidates, "review_type"
    )
    if review_type is None:
        review_type = "model_assisted_semantic_training_v3"

    usage_policy = manifest.get("usage_policy")
    if not isinstance(usage_policy, dict):
        usage_policy = {}
        manifest["usage_policy"] = usage_policy

    candidate_review = usage_policy.get("candidate_review")
    if not isinstance(candidate_review, dict):
        candidate_review = {}
        usage_policy["candidate_review"] = candidate_review
    candidate_review.update(
        {
            "total": len(candidates),
            "approved": sum(row.get("status") == "approved" for row in candidates),
            "human_verified": sum(
                row.get("human_verified") is True for row in candidates
            ),
            "method": _most_common_string(candidates, "review_method")
            or _most_common_string(candidates, "review_type")
            or candidate_review.get("method"),
        }
    )

    semantic_training = usage_policy.get("semantic_training")
    if not isinstance(semantic_training, dict):
        semantic_training = {}
        usage_policy["semantic_training"] = semantic_training
    semantic_training.update(
        {
            "pair_count": len(pairs),
            "pair_count_by_split": pair_counts,
            "source_page_count_by_split": pair_pages,
            "positive": "exact_fixed_token_chunk",
            "human_verified": bool(pairs)
            and all(row.get("human_verified") is True for row in pairs),
            "review_type": review_type,
            "query_types": sorted(
                {
                    row.get("query_type")
                    for row in candidates
                    if isinstance(row.get("query_type"), str)
                }
            ),
        }
    )

    semantic_audit = manifest.get("semantic_audit")
    if not isinstance(semantic_audit, dict):
        semantic_audit = {}
        manifest["semantic_audit"] = semantic_audit
    semantic_audit.update(
        {
            "method": review_type,
            "human_verified": bool(pairs)
            and all(row.get("human_verified") is True for row in pairs),
            "scope": {
                split: {
                    "source_pages": pair_pages[split],
                    "retained_pairs": pair_counts[split],
                }
                for split in ("train", "dev", "test")
            },
        }
    )

    split_protocol = manifest.get("split_protocol")
    if isinstance(split_protocol, dict):
        # These fields describe the current retained training package.  Do not
        # preserve pre-filtering targets as if they were current record counts.
        split_protocol.pop("candidate_target_counts", None)
        split_protocol["candidate_actual_counts"] = pair_counts
        split_protocol["source_page_counts"] = pair_pages

    construction = manifest.get("data_construction")
    if isinstance(construction, dict):
        construction.pop("generated_item_count", None)
        construction["retained_item_count"] = len(candidates)
        construction["retained_pair_count_by_split"] = pair_counts
        construction["query_types"] = sorted(
            {
                row.get("query_type")
                for row in candidates
                if isinstance(row.get("query_type"), str)
            }
        )
        # This is the number of fixed chunks that are actually supervised by
        # each split, not the number of legacy page/quote evidence rows.
        construction["retained_source_chunk_count_by_split"] = {
            split: len(
                {
                    row.get("positive_evidence_id")
                    for row in pairs
                    if row.get("split") == split
                    and isinstance(row.get("positive_evidence_id"), str)
                }
            )
            for split in ("train", "dev", "test")
        }
        # Keep the manifest's retained-source indexes synchronized with the
        # canonical pairs.  These lists are consumed by audit tooling; leaving
        # removed evidence IDs here would make a refreshed manifest describe
        # records that no longer exist.
        construction["candidate_source_evidence_ids_by_split"] = {
            split: sorted(
                {
                    row.get("positive_evidence_id")
                    for row in pairs
                    if row.get("split") == split
                    and isinstance(row.get("positive_evidence_id"), str)
                }
            )
            for split in ("train", "dev", "test")
        }
        construction["source_pages_by_split"] = {
            split: sorted(
                {
                    row.get("page")
                    for row in pairs
                    if row.get("split") == split and isinstance(row.get("page"), int)
                }
            )
            for split in ("train", "dev", "test")
        }
        construction["source_pages_with_retained_items"] = len(
            {
                row.get("page")
                for row in pairs
                if isinstance(row.get("page"), int)
            }
        )
        construction["fixed_chunk_count"] = sum(
            row.get("evidence_type") == "fixed_token_chunk" for row in evidence
        )

        # The secondary semantic exclusions are an audit file adjacent to the
        # canonical package rather than a JSONL input declared in the package.
        # When that file is present, refresh both related summary counts so a
        # record removal cannot leave a stale manifest that still passes a
        # superficial count check.  Disposable package copies used by
        # ``_validate_candidate`` do not have the adjacent audit file and keep
        # the already refreshed values.
        exclusions_path = dataset_root.parent / "semantic_review_exclusions.json"
        if exclusions_path.is_file():
            try:
                exclusions = json.loads(exclusions_path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PreparationError(
                    f"semantic_review_exclusions 不是有效 JSON：{exclusions_path}"
                ) from exc
            if not isinstance(exclusions, dict) or not isinstance(
                exclusions.get("rejections"), list
            ):
                raise PreparationError(
                    "semantic_review_exclusions.rejections 必须是数组"
                )
            secondary_count = len(exclusions["rejections"])
            previous_secondary = construction.get("secondary_review_exclusion_count")
            previous_rejected = construction.get("rejected_item_count")
            if not isinstance(previous_secondary, int) or not isinstance(
                previous_rejected, int
            ):
                raise PreparationError(
                    "manifest.data_construction 的审核计数必须是整数"
                )
            primary_count = previous_rejected - previous_secondary
            if primary_count < 0:
                raise PreparationError(
                    "manifest.data_construction.rejected_item_count 小于二次剔除数"
                )
            construction["secondary_review_exclusion_count"] = secondary_count
            construction["rejected_item_count"] = primary_count + secondary_count


def prepare_dataset(
    root: Path = DEFAULT_DATASET,
    *,
    dataset_version: str | None = None,
    check_only: bool = False,
) -> dict[str, Any]:
    """Validate and refresh one explicit canonical package."""

    root = root.resolve()
    refreshed = build_refreshed_manifest(root, dataset_version=dataset_version)
    _validate_candidate(root, refreshed)
    current = _load_manifest(root)
    changed = current != refreshed
    if check_only:
        if changed:
            raise PreparationError("manifest 的记录数或 SHA-256 尚未刷新")
    elif changed:
        manifest_path = root / "manifest.json"
        temporary_path = root / ".manifest.json.tmp"
        temporary_path.write_text(
            json.dumps(refreshed, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(manifest_path)
    return {
        "dataset": str(root),
        "dataset_version": refreshed.get("dataset_version"),
        "counts": refreshed["counts"],
        "manifest_changed": changed,
        "mode": "check" if check_only else "refresh",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--dataset-version",
        help="仅在本次数据变更需要发布新版本时显式指定；默认保留现有版本",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只验证 manifest 是否已经与当前 canonical 文件一致",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare_dataset(
        args.dataset,
        dataset_version=args.dataset_version,
        check_only=args.check,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
