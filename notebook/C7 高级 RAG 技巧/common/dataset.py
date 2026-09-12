"""C7 统一数据包的读取与契约校验。

运行时只读取 ``data/dataset``。教程案例、微调样本、证据、划分和审核记录
都由同一份 manifest 管理；教程入口从这些 canonical 记录投影出所需字段。
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


COURSE_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = COURSE_ROOT / "data" / "dataset"
CANONICAL_DOCUMENT_ID = "doc_pumpkin_book"
CANONICAL_PDF_RELATIVE_PATH = "data/pumpkin_book.pdf"
CANONICAL_PDF_PATH = (COURSE_ROOT / "data" / "pumpkin_book.pdf").resolve()
SCHEMA_VERSION = 1
NORMALIZATION_VERSION = "c7-text-normalization-v2"
FIXED_CHUNK_MAX_TOKEN_BUDGET = 384
MODEL_MAX_SEQ_LENGTH = 512
SEMANTIC_REVIEW_TYPE = "model_assisted_semantic_training_v3"
SEMANTIC_REVIEW_STATUS = "model_assisted_semantically_verified"
SEMANTIC_REVIEW_CHECKS = (
    "question_self_contained",
    "positive_fully_supports",
    "answer_fully_supported",
    "support_quotes_sufficient",
)

JSONL_FILES: dict[str, str] = {
    "documents": "documents.jsonl",
    "evidence": "evidence.jsonl",
    "queries": "queries.jsonl",
    "qrels": "qrels.jsonl",
    "sessions": "sessions.jsonl",
    "qa_candidates": "annotations/qa_candidates.jsonl",
    "review_decisions": "annotations/review_decisions.jsonl",
    "finetune_pairs": "finetune_pairs.jsonl",
}
ID_FIELDS: dict[str, str] = {
    "documents": "doc_id",
    "evidence": "evidence_id",
    "queries": "query_id",
    "qrels": None,  # qrels are identified by a compound key
    "sessions": "session_id",
    "qa_candidates": "candidate_id",
    "review_decisions": "decision_id",
    "finetune_pairs": "pair_id",
}
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "documents": (
        "doc_id",
        "path",
        "version",
        "sha256",
        "source",
        "license",
        "source_group",
    ),
    "evidence": (
        "evidence_id",
        "doc_id",
        "section_id",
        "page",
        "quote",
        "offsets",
        "normalization_version",
    ),
    "queries": (
        "query_id",
        "text",
        "task_type",
        "answerability",
        "query_family_id",
        "usage",
        "reference_claims",
    ),
    "qrels": (
        "query_id",
        "evidence_id",
        "relevance",
        "essential",
        "review_status",
    ),
    "sessions": (
        "session_id",
        "mode",
        "turns",
        "query_ids",
    ),
    "qa_candidates": (
        "candidate_id",
        "query",
        "answer",
        "page_num",
        "status",
        "review_status",
        "training_eligible",
        "evidence_ids",
    ),
    "review_decisions": (
        "decision_id",
        "candidate_id",
        "decision",
        "review_status",
        "human_verified",
    ),
    "finetune_pairs": (
        "pair_id",
        "query_id",
        "query",
        "positive",
        "positive_evidence_id",
        "page",
        "section_id",
        "query_family_id",
        "split",
        "review_status",
        "human_verified",
    ),
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class DatasetContractError(ValueError):
    """Canonical package 缺失、格式错误或引用断裂。"""


def normalize_query(value: object) -> str:
    """Normalize query wording for duplicate/conflict detection.

    Whitespace and punctuation are intentionally removed after NFKC
    normalization.  This catches superficial variants such as
    "KKT 条件是什么？" and "什么是KKT条件" without changing the stored
    question text used by the model.
    """

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = "".join(
        character
        for character in text
        if not character.isspace()
        and not unicodedata.category(character).startswith(("P", "S"))
    )
    # Canonicalize the common Chinese definition inversion so
    # “KKT条件是什么” and “什么是KKT条件” conflict as one query.
    if text.startswith("什么是") and len(text) > 3:
        text = text[3:] + "是什么"
    elif text.endswith("是什么") and len(text) > 3:
        text = text[:-3] + "是什么"
    return text


def _root(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else DATASET_ROOT


def _path(root: Path, name: str) -> Path:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"canonical 数据文件不存在：{path}")
    return path


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetContractError(f"JSON 格式错误：{path}") from exc


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """严格读取 JSONL；空行、非对象和坏 JSON 都直接报错。"""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"canonical 数据文件不存在：{path}")
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise DatasetContractError(f"JSONL 编码错误：{path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise DatasetContractError(f"JSONL 不允许空行：{path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetContractError(
                f"JSONL 格式错误：{path}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise DatasetContractError(
                f"JSONL 每行必须是对象：{path}:{line_number}"
            )
        rows.append(value)
    return rows


def _read_package(root: Path) -> dict[str, Any]:
    manifest_path = _path(root, "manifest.json")
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise DatasetContractError("manifest.json 必须是对象")
    package: dict[str, Any] = {"manifest": manifest}
    for key, relative in JSONL_FILES.items():
        package[key] = read_jsonl(_path(root, relative))
    method_cases_path = _path(root, "annotations/method_cases.json")
    method_cases = _load_json(method_cases_path)
    if not isinstance(method_cases, dict):
        raise DatasetContractError("annotations/method_cases.json 必须是对象")
    package["method_cases"] = method_cases
    splits = _load_json(_path(root, "splits.json"))
    if not isinstance(splits, dict):
        raise DatasetContractError("splits.json 必须是对象")
    package["splits"] = splits
    return package


def _id_set(rows: Iterable[Mapping[str, Any]], field: str) -> tuple[set[str], list[str]]:
    values: set[str] = set()
    issues: list[str] = []
    for index, row in enumerate(rows, start=1):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            issues.append(f"第 {index} 行 {field} 为空")
            continue
        if value in values:
            issues.append(f"重复 {field}：{value}")
        values.add(value)
    return values, issues


def _append_required_issues(
    issues: list[str], name: str, rows: Iterable[Mapping[str, Any]]
) -> None:
    required = REQUIRED_FIELDS[name]
    for index, row in enumerate(rows, start=1):
        missing = [field for field in required if field not in row]
        if missing:
            issues.append(f"{name} 第 {index} 行缺少字段：{', '.join(missing)}")


def _validate_offsets(value: Any, label: str, issues: list[str]) -> bool:
    """Validate the structural part of the normalized PDF-span contract."""

    valid = True
    if not isinstance(value, dict):
        issues.append(f"{label} offsets 必须是对象")
        return False
    status = value.get("status")
    if status != "exact":
        issues.append(f"{label} offsets.status 必须为 exact")
        valid = False
    if value.get("unit") != "normalized_characters":
        issues.append(f"{label} offsets.unit 必须为 normalized_characters")
        valid = False
    start, end = value.get("start"), value.get("end")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
    ):
        issues.append(f"{label} offsets 必须包含整数 start/end")
        valid = False
    elif start < 0 or start >= end:
        issues.append(f"{label} offsets 范围无效")
        valid = False
    return valid


def _validate_canonical_document(
    package: Mapping[str, Any], issues: list[str]
) -> bool:
    """Validate the fixed, single-document source declaration.

    The provenance checker must not infer which document is canonical from a
    user-controlled path.  C7 has one source document and every record must
    bind to this exact declaration, including its relative path and digest.
    """

    raw_documents = package.get("documents")
    if not isinstance(raw_documents, list):
        issues.append("canonical document 声明必须是列表")
        return False
    documents = [row for row in raw_documents if isinstance(row, Mapping)]
    if len(documents) != len(raw_documents):
        issues.append("canonical document 声明不能包含非对象记录")
    if len(documents) != 1:
        issues.append("canonical document 必须恰好声明一个文档")

    canonical = next(
        (row for row in documents if row.get("doc_id") == CANONICAL_DOCUMENT_ID),
        None,
    )
    if canonical is None:
        issues.append(
            "canonical document 缺少固定文档 doc_pumpkin_book"
        )
        return False
    if len(documents) == 1 and canonical.get("doc_id") != CANONICAL_DOCUMENT_ID:
        # Kept for clarity if the selection logic is changed later.
        issues.append("canonical document doc_id 不匹配 doc_pumpkin_book")
        return False

    valid = True
    path_value = canonical.get("path")
    if path_value != CANONICAL_PDF_RELATIVE_PATH:
        issues.append(
            "canonical document path 必须为 data/pumpkin_book.pdf"
        )
        valid = False
    else:
        try:
            resolved = (COURSE_ROOT / path_value).resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            issues.append(f"canonical document path 无法解析：{exc}")
            valid = False
        else:
            if resolved != CANONICAL_PDF_PATH:
                issues.append(
                    "canonical document path 未指向固定的 pumpkin_book.pdf"
                )
                valid = False

    expected_sha = canonical.get("sha256")
    if not isinstance(expected_sha, str) or HEX64.fullmatch(expected_sha) is None:
        issues.append("canonical document sha256 必须是 64 位小写十六进制")
        valid = False

    if not CANONICAL_PDF_PATH.is_file():
        issues.append(f"canonical PDF 原始资料缺失：{CANONICAL_PDF_PATH}")
        return False
    try:
        actual_sha = sha256_file(CANONICAL_PDF_PATH)
    except OSError as exc:
        issues.append(f"无法读取 canonical PDF：{exc}")
        return False
    if isinstance(expected_sha, str) and HEX64.fullmatch(expected_sha):
        if actual_sha != expected_sha:
            issues.append("canonical document SHA-256 与实际文件不一致")
            valid = False
    return valid


@lru_cache(maxsize=1)
def _canonical_pdf_page_texts() -> dict[int, str]:
    """Load normalized page text once, using the project's PDF normalizer."""

    # Keep this import lazy: eval_utils imports this module for its dataset
    # constants, while validation itself only runs after module initialization.
    from . import eval_utils

    pages = eval_utils.load_pdf_pages()
    page_texts: dict[int, str] = {}
    for item in pages:
        if not isinstance(item, Mapping):
            raise ValueError("PDF page loader 返回了非对象记录")
        page, text = item.get("page"), item.get("text")
        if (
            not isinstance(page, int)
            or isinstance(page, bool)
            or page <= 0
            or not isinstance(text, str)
        ):
            raise ValueError("PDF page loader 返回了非法页码或文本")
        if page in page_texts:
            raise ValueError(f"PDF page loader 返回重复页码：{page}")
        page_texts[page] = text
    if not page_texts:
        raise ValueError("PDF 没有可读取的规范化页面")
    return page_texts


def validate_pdf_provenance(
    package: Mapping[str, Any],
    *,
    pages: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    """Require every evidence span sourced from the canonical PDF to be exact.

    Offsets are measured against ``eval_utils.load_pdf_pages`` output, which is
    the repository's normalized-character contract.  Invalid page/offset
    records are reported rather than skipped, and no quote is trusted without
    comparing it to the corresponding normalized page span.
    """

    issues: list[str] = []
    document_valid = _validate_canonical_document(package, issues)

    evidence_rows = package.get("evidence")
    if not isinstance(evidence_rows, list):
        issues.append("evidence 必须是记录列表")
        return issues
    for index, row in enumerate(evidence_rows, start=1):
        if not isinstance(row, Mapping):
            issues.append(f"evidence 第 {index} 行必须是对象")
        elif row.get("doc_id") != CANONICAL_DOCUMENT_ID:
            issues.append(
                f"evidence 第 {index} 行必须引用固定 canonical document "
                f"{CANONICAL_DOCUMENT_ID}"
            )

    # Even when the declaration is malformed, the errors above are the
    # fail-closed result.  Do not infer a replacement source or validate
    # offsets against an unrelated document.
    if not document_valid:
        return issues

    try:
        if pages is None:
            page_texts = _canonical_pdf_page_texts()
        else:
            page_texts = {}
            for item in pages:
                if not isinstance(item, Mapping):
                    raise ValueError("PDF page loader 返回了非对象记录")
                page, text = item.get("page"), item.get("text")
                if (
                    not isinstance(page, int)
                    or isinstance(page, bool)
                    or page <= 0
                    or not isinstance(text, str)
                ):
                    raise ValueError("PDF page loader 返回了非法页码或文本")
                if page in page_texts:
                    raise ValueError(f"PDF page loader 返回重复页码：{page}")
                page_texts[page] = text
            if not page_texts:
                raise ValueError("PDF 没有可读取的规范化页面")
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        issues.append(f"无法读取 canonical PDF 规范化文本：{exc}")
        return issues

    for index, row in enumerate(evidence_rows, start=1):
        if not isinstance(row, Mapping) or row.get("doc_id") != CANONICAL_DOCUMENT_ID:
            continue
        label = f"evidence 第 {index} 行"
        page = row.get("page")
        if not isinstance(page, int) or isinstance(page, bool) or page <= 0:
            issues.append(f"{label} page 必须为 canonical PDF 范围内的正整数")
            continue
        page_text = page_texts.get(page)
        if page_text is None:
            issues.append(f"{label} page 超出 canonical PDF 页范围：{page}")
            continue
        quote = row.get("quote")
        if not isinstance(quote, str) or not quote.strip():
            issues.append(f"{label} quote 必须为非空字符串")
            continue
        offsets = row.get("offsets")
        if not _validate_offsets(offsets, label, issues):
            continue
        start, end = offsets["start"], offsets["end"]
        if end > len(page_text):
            issues.append(
                f"{label} offsets 超出第 {page} 页规范化文本范围："
                f"{start}:{end} / {len(page_text)}"
            )
            continue
        if end - start != len(quote):
            issues.append(f"{label} offsets 长度与 quote 不一致")
            continue
        if page_text[start:end] != quote:
            issues.append(f"{label} 不是声明页码和 offsets 上的逐字 PDF 原文")
        quote_normalized = row.get("quote_normalized")
        if quote_normalized is not None and quote_normalized != quote:
            issues.append(f"{label} quote_normalized 必须与 quote 一致")
    return issues


def _validate_records(package: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    manifest = package.get("manifest")
    if not isinstance(manifest, dict):
        return ["manifest 缺失或不是对象"]
    if manifest.get("schema_version") != SCHEMA_VERSION:
        issues.append(f"不支持的 schema_version：{manifest.get('schema_version')!r}")
    if manifest.get("source_of_truth") != "canonical_dataset":
        issues.append("manifest.source_of_truth 必须为 canonical_dataset")
    for name in JSONL_FILES:
        rows = package.get(name)
        if not isinstance(rows, list):
            issues.append(f"{name} 不是记录列表")
            continue
        _append_required_issues(issues, name, rows)
        id_field = ID_FIELDS[name]
        if id_field:
            _, id_issues = _id_set(rows, id_field)
            issues.extend(f"{name}: {issue}" for issue in id_issues)

    documents = package.get("documents", [])
    evidence = package.get("evidence", [])
    queries = package.get("queries", [])
    qrels = package.get("qrels", [])
    sessions = package.get("sessions", [])
    candidates = package.get("qa_candidates", [])
    decisions = package.get("review_decisions", [])
    finetune_pairs = package.get("finetune_pairs", [])
    doc_ids = {row.get("doc_id") for row in documents if isinstance(row, dict)}
    evidence_ids = {row.get("evidence_id") for row in evidence if isinstance(row, dict)}
    query_ids = {row.get("query_id") for row in queries if isinstance(row, dict)}

    for index, row in enumerate(documents, start=1):
        if not isinstance(row, dict):
            continue
        if not isinstance(row.get("path"), str) or not row.get("path"):
            issues.append(f"documents 第 {index} 行 path 为空")
        if not isinstance(row.get("sha256"), str) or not HEX64.fullmatch(row.get("sha256", "")):
            issues.append(f"documents 第 {index} 行 sha256 无效")
        ranges = row.get("chapter_ranges")
        if not isinstance(ranges, list) or not ranges:
            issues.append(f"documents 第 {index} 行 chapter_ranges 必须是非空列表")
        else:
            previous_end = 0
            for range_index, item in enumerate(ranges, start=1):
                if not isinstance(item, dict):
                    issues.append(f"documents 第 {index} 行 chapter_ranges 第 {range_index} 项不是对象")
                    continue
                start, end, title = item.get("start"), item.get("end"), item.get("title")
                if not isinstance(start, int) or not isinstance(end, int) or start > end:
                    issues.append(f"documents 第 {index} 行 chapter_ranges 第 {range_index} 项范围无效")
                elif start <= previous_end:
                    issues.append(f"documents 第 {index} 行 chapter_ranges 必须按页码递增且不重叠")
                else:
                    previous_end = end
                if not isinstance(title, str) or not title.strip():
                    issues.append(f"documents 第 {index} 行 chapter_ranges 第 {range_index} 项标题为空")

    for index, row in enumerate(evidence, start=1):
        if not isinstance(row, dict):
            continue
        label = f"evidence 第 {index} 行"
        if row.get("doc_id") not in doc_ids:
            issues.append(f"{label} 引用了不存在的 doc_id：{row.get('doc_id')!r}")
        page = row.get("page")
        if not isinstance(page, int) or page <= 0:
            issues.append(f"{label} page 必须为正整数")
        if not isinstance(row.get("quote"), str) or not row.get("quote", "").strip():
            issues.append(f"{label} quote 为空")
        _validate_offsets(row.get("offsets"), label, issues)
        if row.get("normalization_version") != NORMALIZATION_VERSION:
            issues.append(f"{label} normalization_version 不匹配")
        if row.get("evidence_type") == "fixed_token_chunk":
            token_count = row.get("token_count")
            token_budget = row.get("token_budget")
            if (
                not isinstance(token_count, int)
                or isinstance(token_count, bool)
                or token_count <= 0
            ):
                issues.append(f"{label} fixed chunk token_count 必须为正整数")
            if (
                not isinstance(token_budget, int)
                or isinstance(token_budget, bool)
                or token_budget <= 0
                or token_budget > FIXED_CHUNK_MAX_TOKEN_BUDGET
                or token_budget >= MODEL_MAX_SEQ_LENGTH
            ):
                issues.append(
                    f"{label} fixed chunk token_budget 必须不超过 {FIXED_CHUNK_MAX_TOKEN_BUDGET}"
                )
            if (
                isinstance(token_count, int)
                and isinstance(token_budget, int)
                and token_count > token_budget
            ):
                issues.append(f"{label} fixed chunk token_count 超过 token_budget")
            offsets = row.get("offsets")
            if (
                isinstance(offsets, dict)
                and isinstance(offsets.get("start"), int)
                and isinstance(offsets.get("end"), int)
                and isinstance(row.get("quote"), str)
                and offsets["end"] - offsets["start"] != len(row["quote"])
            ):
                issues.append(f"{label} fixed chunk offsets 与 quote 长度不一致")

    issues.extend(validate_pdf_provenance(package))

    qrel_keys: set[tuple[Any, Any]] = set()
    qrel_evidence_by_query: dict[str, set[str]] = {}
    for index, row in enumerate(qrels, start=1):
        if not isinstance(row, dict):
            continue
        label = f"qrels 第 {index} 行"
        query_id, evidence_id = row.get("query_id"), row.get("evidence_id")
        key = (query_id, evidence_id)
        if key in qrel_keys:
            issues.append(f"重复 qrel：{query_id!r} -> {evidence_id!r}")
        qrel_keys.add(key)
        if query_id not in query_ids:
            issues.append(f"{label} 引用了不存在的 query_id：{query_id!r}")
        if evidence_id not in evidence_ids:
            issues.append(f"{label} 引用了不存在的 evidence_id：{evidence_id!r}")
        if not isinstance(row.get("relevance"), int) or row.get("relevance") not in (0, 1):
            issues.append(f"{label} relevance 必须为 0 或 1")
        if not isinstance(row.get("essential"), bool):
            issues.append(f"{label} essential 必须为布尔值")
        if not isinstance(row.get("review_status"), str) or not row.get("review_status"):
            issues.append(f"{label} review_status 为空")
        if isinstance(query_id, str) and isinstance(evidence_id, str):
            qrel_evidence_by_query.setdefault(query_id, set()).add(evidence_id)

    for index, row in enumerate(queries, start=1):
        if not isinstance(row, dict):
            continue
        label = f"queries 第 {index} 行"
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            issues.append(f"{label} text 为空")
        usage = row.get("usage")
        if not isinstance(usage, list) or not usage or not all(
            isinstance(item, str) and item for item in usage
        ):
            issues.append(f"{label} usage 必须是非空字符串列表")
        claims = row.get("reference_claims")
        if not isinstance(claims, list):
            issues.append(f"{label} reference_claims 必须是列表")
            continue
        claim_evidence: set[str] = set()
        for claim_index, claim in enumerate(claims, start=1):
            if not isinstance(claim, dict):
                issues.append(f"{label} claim {claim_index} 不是对象")
                continue
            refs = claim.get("evidence_ids")
            if not isinstance(refs, list) or not refs:
                issues.append(f"{label} claim {claim_index} 没有 evidence_ids")
                continue
            for evidence_id in refs:
                claim_evidence.add(evidence_id)
                if evidence_id not in evidence_ids:
                    issues.append(
                        f"{label} claim {claim_index} 引用了不存在的 evidence_id：{evidence_id!r}"
                    )
        if claim_evidence - qrel_evidence_by_query.get(row.get("query_id"), set()):
            issues.append(f"{label} 的 reference_claims 未全部写入 qrels")
        answerability = row.get("answerability")
        if answerability == "answerable" and not qrel_evidence_by_query.get(row.get("query_id")):
            issues.append(f"{label} answerable 但没有 qrels")

    # The v3 package rejects superficial duplicate/conflicting questions
    # globally; the canonical package has no review bypass.
    if manifest.get("data_construction", {}).get("review_type") == SEMANTIC_REVIEW_TYPE:
        normalized_queries: dict[str, str] = {}
        for row in queries:
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                continue
            normalized = normalize_query(row["text"])
            if not normalized:
                continue
            previous = normalized_queries.get(normalized)
            if previous is not None and previous != row.get("query_id"):
                issues.append(
                    "规范化 query 冲突："
                    f"{previous!r} 与 {row.get('query_id')!r} "
                    f"均归一为 {normalized!r}"
                )
            normalized_queries[normalized] = str(row.get("query_id"))

    for index, row in enumerate(sessions, start=1):
        if not isinstance(row, dict):
            continue
        label = f"sessions 第 {index} 行"
        turns = row.get("turns")
        if not isinstance(turns, list) or not turns:
            issues.append(f"{label} turns 必须为非空列表")
        for query_id in row.get("query_ids", []):
            if query_id not in query_ids:
                issues.append(f"{label} 引用了不存在的 query_id：{query_id!r}")

    evidence_by_id = {
        row.get("evidence_id"): row for row in evidence if isinstance(row, dict)
    }
    for index, row in enumerate(candidates, start=1):
        if not isinstance(row, dict):
            continue
        label = f"qa_candidates 第 {index} 行"
        if row.get("source_path") != CANONICAL_PDF_RELATIVE_PATH:
            issues.append(
                f"{label} source_path 必须为 canonical document 的 "
                "data/pumpkin_book.pdf"
            )
        query = row.get("query")
        if row.get("training_eligible"):
            if not isinstance(query, str) or not query.strip():
                issues.append(f"{label} 空问题不能进入训练")
            if row.get("status") != "approved" or row.get("review_status") != "verified":
                issues.append(f"{label} 未核验却标记为可训练")
            candidate_evidence_ids = row.get("evidence_ids")
            if not isinstance(candidate_evidence_ids, list) or len(candidate_evidence_ids) != 1:
                issues.append(f"{label} 可训练候选没有 evidence_ids")
            else:
                source_evidence = evidence_by_id.get(candidate_evidence_ids[0])
                verification = row.get("verification")
                if isinstance(source_evidence, dict) and isinstance(verification, dict):
                    offsets = source_evidence.get("offsets", {})
                    if (
                        verification.get("chunk_start") != offsets.get("start")
                        or verification.get("chunk_end") != offsets.get("end")
                    ):
                        issues.append(f"{label} verification offsets 与 evidence 不一致")
                    if row.get("pdf_page") != source_evidence.get("page"):
                        issues.append(f"{label} pdf_page 与 evidence 不一致")
        for evidence_id in row.get("evidence_ids", []):
            if evidence_id not in evidence_ids:
                issues.append(f"{label} 引用了不存在的 evidence_id：{evidence_id!r}")
        if not isinstance(row.get("page_num"), int) or row.get("page_num") <= 0:
            issues.append(f"{label} page_num 必须为正整数")
        if not isinstance(query, str) or not query.strip():
            if row.get("status") != "rejected" or row.get("review_status") != "rejected":
                issues.append(f"{label} 空问题必须为 rejected")

    candidate_by_id = {
        row.get("candidate_id"): row for row in candidates if isinstance(row, dict)
    }
    candidate_ids = set(candidate_by_id)
    decision_ids, decision_id_issues = _id_set(decisions, "decision_id")
    issues.extend(f"review_decisions: {issue}" for issue in decision_id_issues)
    for index, row in enumerate(decisions, start=1):
        if not isinstance(row, dict):
            continue
        label = f"review_decisions 第 {index} 行"
        if row.get("candidate_id") not in candidate_ids:
            issues.append(
                f"{label} 引用了不存在的 candidate_id：{row.get('candidate_id')!r}"
            )
        if row.get("decision") not in {"pending", "rejected", "approved"}:
            issues.append(f"{label} decision 无效")
        if row.get("human_verified") is not False:
            issues.append(f"{label} 不得伪装成人工核验")
        candidate = candidate_by_id.get(row.get("candidate_id"))
        if isinstance(candidate, dict):
            if row.get("decision") != candidate.get("status"):
                issues.append(f"{label} decision 与 qa_candidate.status 不一致")
            if row.get("evidence_ids") != candidate.get("evidence_ids"):
                issues.append(f"{label} evidence_ids 与 qa_candidate 不一致")
            if row.get("verification") != candidate.get("verification"):
                issues.append(f"{label} verification 与 qa_candidate 不一致")

    query_by_id = {
        row.get("query_id"): row for row in queries if isinstance(row, dict)
    }
    for index, row in enumerate(finetune_pairs, start=1):
        if not isinstance(row, dict):
            continue
        label = f"finetune_pairs 第 {index} 行"
        query = query_by_id.get(row.get("query_id"))
        positive = evidence_by_id.get(row.get("positive_evidence_id"))
        if query is None:
            issues.append(f"{label} 引用了不存在的 query_id：{row.get('query_id')!r}")
        elif row.get("query") != query.get("text"):
            issues.append(f"{label} query 与 canonical query 不一致")
        elif row.get("query_family_id") != query.get("query_family_id"):
            issues.append(f"{label} query_family_id 与 canonical query 不一致")
        if positive is None:
            issues.append(
                f"{label} 引用了不存在的 positive_evidence_id："
                f"{row.get('positive_evidence_id')!r}"
            )
        else:
            if row.get("positive") != positive.get("quote"):
                issues.append(f"{label} positive 与 canonical evidence.quote 不一致")
            if row.get("page") != positive.get("page"):
                issues.append(f"{label} page 与 canonical evidence 不一致")
            if row.get("section_id") != positive.get("section_id"):
                issues.append(f"{label} section_id 与 canonical evidence 不一致")
            if positive.get("evidence_type") == "fixed_token_chunk":
                if not isinstance(row.get("positive_token_count"), int):
                    issues.append(f"{label} fixed chunk pair 缺少 positive_token_count")
                elif row["positive_token_count"] != positive.get("token_count"):
                    issues.append(f"{label} positive_token_count 与 evidence 不一致")
                if row.get("token_budget") != positive.get("token_budget"):
                    issues.append(f"{label} token_budget 与 evidence 不一致")
        positive_text = str(row.get("positive", ""))
        if any(
            marker in positive_text
            for marker in ("欢迎去各大电商平台选购纸质版南瓜书", "配套视频教程", "bilibili.com")
        ):
            issues.append(f"{label} positive 含 PDF 重复水印")
        if row.get("split") not in {"train", "dev", "test"}:
            issues.append(f"{label} split 无效：{row.get('split')!r}")
        review_status = row.get("review_status")
        if review_status == SEMANTIC_REVIEW_STATUS:
            semantic_review = row.get("semantic_review")
            if (
                positive is None
                or positive.get("evidence_type") != "fixed_token_chunk"
                or row.get("review_type") != SEMANTIC_REVIEW_TYPE
                or row.get("supervision_type") != "semantic_query_evidence"
                or not isinstance(semantic_review, dict)
                or semantic_review.get("status") != "passed"
                or semantic_review.get("review_type") != SEMANTIC_REVIEW_TYPE
                or semantic_review.get("human_verified") is not False
                or any(
                    semantic_review.get(key) is not True
                    for key in SEMANTIC_REVIEW_CHECKS
                )
            ):
                issues.append(f"{label} 未通过对应版本语义核验")
        else:
            issues.append(f"{label} 未通过对应版本语义核验")
        if row.get("human_verified") is not False:
            issues.append(f"{label} 不得伪装成人工核验")
        if any(key in row for key in ("answer", "reference_answer", "generated_answer")):
            issues.append(f"{label} 不得包含生成答案字段")

    splits = package.get("splits")
    if not isinstance(splits, dict):
        issues.append("splits.json 缺失或不是对象")
    else:
        if splits.get("schema_version") != SCHEMA_VERSION:
            issues.append("splits.schema_version 不匹配")
        group_keys = splits.get("group_keys")
        if group_keys != ["section_id", "query_family_id"]:
            issues.append("splits.group_keys 必须为 section_id + query_family_id")
        assignments = splits.get("assignments")
        if not isinstance(assignments, dict):
            issues.append("splits.assignments 必须是对象")
            assignments = {}
        partitions = {"train", "dev", "test"}
        seen_queries: dict[str, str] = {}
        groups = splits.get("groups", {})
        if not isinstance(groups, dict):
            issues.append("splits.groups 必须是对象")
            groups = {}
        for partition in partitions:
            values = assignments.get(partition, [])
            if not isinstance(values, list):
                issues.append(f"splits.assignments.{partition} 必须是列表")
                continue
            for query_id in values:
                if query_id not in query_ids:
                    issues.append(f"split {partition} 引用了不存在的 query_id：{query_id!r}")
                    continue
                if query_id in seen_queries:
                    issues.append(
                        f"query_id 同时出现在多个 split：{query_id}（{seen_queries[query_id]} 和 {partition}）"
                    )
                seen_queries[query_id] = partition
                group = groups.get(query_id)
                if not isinstance(group, dict):
                    issues.append(f"split query 缺少 group：{query_id}")
        for index, row in enumerate(finetune_pairs, start=1):
            if not isinstance(row, dict):
                continue
            partition = row.get("split")
            query_id = row.get("query_id")
            if partition in partitions and query_id not in assignments.get(partition, []):
                issues.append(
                    f"finetune_pairs 第 {index} 行 split 与 splits.json 不一致"
                )
        section_to_partition: dict[Any, str] = {}
        family_to_partition: dict[Any, str] = {}
        evidence_to_partition: dict[str, str] = {}
        for query_id, partition in seen_queries.items():
            group = groups.get(query_id, {})
            section_id = group.get("section_id")
            family_id = group.get("query_family_id")
            query = query_by_id.get(query_id, {})
            if family_id != query.get("query_family_id"):
                issues.append(f"split group 的 query_family_id 与 query 不一致：{query_id}")
            if section_id not in set(query.get("evidence_section_ids", [])):
                issues.append(f"split group 的 section_id 与 query 不一致：{query_id}")
            if section_id in section_to_partition and section_to_partition[section_id] != partition:
                issues.append(
                    "section split 泄漏："
                    f"{section_id!r} 同时出现在 {section_to_partition[section_id]} 和 {partition}"
                )
            section_to_partition[section_id] = partition
            if family_id in family_to_partition and family_to_partition[family_id] != partition:
                issues.append(
                    "query family split 泄漏："
                    f"{family_id!r} 同时出现在 {family_to_partition[family_id]} 和 {partition}"
                )
            family_to_partition[family_id] = partition
            for evidence_id in qrel_evidence_by_query.get(query_id, set()):
                previous_evidence_split = evidence_to_partition.get(evidence_id)
                if (
                    previous_evidence_split is not None
                    and previous_evidence_split != partition
                ):
                    issues.append(
                        "evidence split 泄漏："
                        f"{evidence_id} 同时出现在 {previous_evidence_split} 和 {partition}"
                    )
                evidence_to_partition[evidence_id] = partition
        regression = splits.get("regression_query_ids", [])
        if not isinstance(regression, list):
            issues.append("splits.regression_query_ids 必须是列表")
            regression = []
        for query_id in regression:
            if query_id not in query_ids:
                issues.append(f"regression 引用了不存在的 query_id：{query_id!r}")
            if query_id in assignments.get("train", []) or query_id in assignments.get("test", []):
                issues.append(f"regression query 不得进入 train/test：{query_id}")
        regression_assignment = assignments.get("regression", [])
        if not isinstance(regression_assignment, list):
            issues.append("splits.assignments.regression 必须是列表")
        elif set(regression_assignment) != set(regression):
            issues.append("splits.assignments.regression 与 regression_query_ids 不一致")
        assigned_experiment = set().union(
            *(set(assignments.get(name, [])) for name in ("train", "dev", "test"))
        )
        overlap = set(regression) & assigned_experiment
        if overlap:
            issues.append(f"regression query 不得进入训练实验分区：{sorted(overlap)}")

    counts = manifest.get("counts")
    if isinstance(counts, dict):
        for key in JSONL_FILES:
            expected = counts.get(key)
            actual = len(package.get(key, []))
            if expected is not None and expected != actual:
                issues.append(f"manifest counts.{key}={expected}，实际为 {actual}")
    else:
        issues.append("manifest.counts 必须是对象")

    policy = manifest.get("usage_policy", {})
    current_ids = policy.get("current_case_ids", []) if isinstance(policy, dict) else []
    if not isinstance(current_ids, list):
        issues.append("manifest.usage_policy.current_case_ids 必须是列表")
    else:
        by_id = {row.get("query_id"): row for row in queries if isinstance(row, dict)}
        for query_id in current_ids:
            row = by_id.get(query_id)
            if row is None:
                issues.append(f"usage_policy 引用了不存在的 query_id：{query_id!r}")
                continue
            usage = set(row.get("usage", []))
            if not {"demo", "dev", "regression"}.issubset(usage):
                issues.append(f"当前 70 题 usage 不完整：{query_id}")
            if {"finetune", "test"} & usage:
                issues.append(f"当前 70 题不得用于 finetune/test：{query_id}")
            excluded = set(row.get("excluded_from", []))
            if not {"finetune", "test"}.issubset(excluded):
                issues.append(f"当前 70 题缺少 finetune/test 排除标记：{query_id}")
    return issues


def validate_dataset(root: str | Path | None = None) -> list[str]:
    """返回所有契约问题，不在 checker 中隐藏第一个错误。"""

    package_root = _root(root)
    try:
        package = _read_package(package_root)
    except (FileNotFoundError, DatasetContractError) as exc:
        return [str(exc)]
    return _validate_records(package)


def load_dataset(root: str | Path | None = None) -> dict[str, Any]:
    """严格读取并验证完整 canonical package。"""

    package_root = _root(root)
    package = _read_package(package_root)
    issues = _validate_records(package)
    if issues:
        raise DatasetContractError("; ".join(issues))
    return package


def _package_or_default(package: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return load_dataset() if package is None else dict(package)


def load_query_records() -> list[dict[str, Any]]:
    return list(load_dataset()["queries"])


def load_evidence_records() -> list[dict[str, Any]]:
    return list(load_dataset()["evidence"])


def load_search_evidence(
    *,
    root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Read the canonical evidence projection used as a search corpus.

    Unlike :func:`load_evidence_records`, this entry point reads only
    ``evidence.jsonl``.  Retrieval code can therefore load the corpus without
    opening queries, qrels, reference answers, or review annotations.
    """

    rows = read_jsonl(_path(_root(root), JSONL_FILES["evidence"]))
    projected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        evidence_id = row.get("evidence_id")
        page = row.get("page")
        quote = row.get("quote")
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise DatasetContractError(f"evidence 第 {index} 行 evidence_id 为空")
        if evidence_id in seen:
            raise DatasetContractError(f"重复 evidence_id：{evidence_id}")
        if not isinstance(page, int) or isinstance(page, bool) or page < 1:
            raise DatasetContractError(f"evidence 第 {index} 行 page 无效：{evidence_id}")
        if not isinstance(quote, str) or not quote.strip():
            raise DatasetContractError(f"evidence 第 {index} 行 quote 为空：{evidence_id}")
        seen.add(evidence_id)
        projected.append(
            {"evidence_id": evidence_id, "page": page, "quote": quote}
        )
    return projected


def load_qrel_records() -> list[dict[str, Any]]:
    return list(load_dataset()["qrels"])


@lru_cache(maxsize=1)
def _chapter_ranges() -> tuple[tuple[int, int, str], ...]:
    # Chapter lookup is needed while constructing retrieval text.  Keep it
    # independent from the full package so this path does not load queries,
    # qrels, or annotations as a side effect.
    documents = read_jsonl(_path(DATASET_ROOT, JSONL_FILES["documents"]))
    document = next(
        (row for row in documents if row.get("doc_id") == CANONICAL_DOCUMENT_ID),
        None,
    )
    if document is None:
        raise DatasetContractError(
            f"documents.jsonl 缺少 canonical 文档：{CANONICAL_DOCUMENT_ID}"
        )
    chapter_ranges = document.get("chapter_ranges")
    if not isinstance(chapter_ranges, list):
        raise DatasetContractError("canonical 文档缺少 chapter_ranges 列表")
    return tuple(
        (int(item["start"]), int(item["end"]), str(item["title"]))
        for item in chapter_ranges
    )


def load_chapter_ranges() -> list[dict[str, Any]]:
    """Return the single canonical PDF page-to-chapter mapping."""

    return [
        {"start": start, "end": end, "title": title}
        for start, end, title in _chapter_ranges()
    ]


def chapter_title(page: int) -> str:
    """Resolve a PDF page through canonical document metadata."""

    page = int(page)
    for start, end, title in _chapter_ranges():
        if start <= page <= end:
            return title
    raise KeyError(f"证据页没有 canonical 章节标题：{page}")


def load_query_only(
    case_ids: Iterable[str] | None = None,
    *,
    root: str | Path | None = None,
) -> list[dict[str, str]]:
    """Read only ``query_id``/``text`` from the canonical query file.

    This is intentionally independent from :func:`load_dataset`: a retrieval
    or generation pipeline must be able to start without opening qrels,
    reference answers, expected pages, or review annotations.  The returned
    projection uses the small ``id``/``query`` vocabulary consumed by the
    teaching notebooks and contains no annotation keys.
    """

    if isinstance(case_ids, (str, bytes)):
        raise TypeError("case_ids 必须是案例 ID 序列，而不是单个字符串")
    requested = None if case_ids is None else [str(item) for item in case_ids]
    wanted_set = set(requested or ())
    if requested is not None and any(not item.strip() for item in requested):
        raise ValueError("case_ids 不能包含空字符串")

    # Do not call load_query_records/load_dataset here.  Those helpers are
    # useful for complete evaluation and necessarily read the whole package.
    rows = read_jsonl(_path(_root(root), JSONL_FILES["queries"]))
    by_id: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows, start=1):
        query_id = row.get("query_id")
        text = row.get("text")
        if not isinstance(query_id, str) or not query_id.strip():
            raise DatasetContractError(f"queries 第 {index} 行 query_id 为空")
        if query_id in by_id:
            raise DatasetContractError(f"重复 query_id：{query_id}")
        if not isinstance(text, str) or not text.strip():
            raise DatasetContractError(f"queries 第 {index} 行 text 为空：{query_id}")
        if requested is None or query_id in wanted_set:
            by_id[query_id] = {"id": query_id, "query": text}
    wanted = list(by_id) if requested is None else requested
    wanted_set = set(wanted)
    missing = wanted_set - set(by_id)
    if missing:
        raise KeyError(f"问题集缺少案例：{sorted(missing)}")
    # Preserve caller order while returning each requested ID once.
    return [by_id[query_id] for query_id in dict.fromkeys(wanted)]


def load_query_controls(
    case_id: str,
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Read non-answerability controls needed to execute one query flow.

    Interaction steps and access-scope settings describe how a caller should
    issue a query; they are intentionally separate from both the tiny query
    projection and the post-result evaluation annotation.  This entry point
    never returns interaction.evaluation, expected pages, reference answers,
    source-page labels, or evidence spans.
    """

    query_rows = read_jsonl(_path(_root(root), JSONL_FILES["queries"]))
    query = next(
        (row for row in query_rows if row.get("query_id") == str(case_id)),
        None,
    )
    if query is None:
        raise KeyError(f"问题集缺少案例：{case_id}")
    interaction = query.get("interaction", {})
    if interaction is None:
        return {}
    if not isinstance(interaction, Mapping):
        raise DatasetContractError(f"案例 {case_id} interaction 必须是对象")

    # Only fields needed to execute the query flow may cross this boundary.
    # In particular, interaction.evaluation contains gold pages and answer
    # expectations and must remain unavailable until load_annotation() runs.
    controls: dict[str, Any] = {}
    for field in (
        "mode",
        "dependency",
        "max_hops",
        "expected_stop_condition",
    ):
        if field in interaction:
            controls[field] = interaction[field]

    if "steps" in interaction:
        steps = interaction["steps"]
        if not isinstance(steps, list):
            raise DatasetContractError(f"案例 {case_id} interaction.steps 必须是列表")
        projected_steps = []
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, Mapping):
                raise DatasetContractError(
                    f"案例 {case_id} interaction.steps 第 {index} 项必须是对象"
                )
            projected_steps.append(
                {
                    field: step[field]
                    for field in ("id", "query", "verified_context")
                    if field in step
                }
            )
        controls["steps"] = projected_steps

    if "turns" in interaction:
        turns = interaction["turns"]
        if not isinstance(turns, list):
            raise DatasetContractError(f"案例 {case_id} interaction.turns 必须是列表")
        projected_turns = []
        for index, turn in enumerate(turns, start=1):
            if not isinstance(turn, Mapping):
                raise DatasetContractError(
                    f"案例 {case_id} interaction.turns 第 {index} 项必须是对象"
                )
            projected_turns.append(
                {
                    field: turn[field]
                    for field in ("role", "q")
                    if field in turn
                }
            )
        controls["turns"] = projected_turns

    if "restricted_pages" in interaction:
        restricted_pages = interaction["restricted_pages"]
        if not isinstance(restricted_pages, list) or any(
            not isinstance(page, int) or isinstance(page, bool) or page < 1
            for page in restricted_pages
        ):
            raise DatasetContractError(
                f"案例 {case_id} interaction.restricted_pages 必须是正整数列表"
            )
        controls["restricted_pages"] = list(restricted_pages)

    return controls


def _annotation_spans(
    query: Mapping[str, Any], evidence_by_id: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    claims = []
    for claim in query.get("reference_claims", []):
        spans = []
        for evidence_id in claim.get("evidence_ids", []):
            evidence = evidence_by_id[evidence_id]
            spans.append({"page": int(evidence["page"]), "quote": str(evidence["quote"])})
        claims.append({"claim": str(claim.get("claim", "")), "spans": spans})
    return claims


def load_annotation(
    case_id: str,
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Read one case's evaluation annotation after a pipeline has run.

    The function deliberately reads only ``queries.jsonl`` and
    ``evidence.jsonl``.  It is the post-result counterpart to
    :func:`load_query_only`; callers must not use it to initialize retrieval.
    Interaction evaluation and step-level source-page labels are returned
    here, after the retrieval result exists.
    """

    package_root = _root(root)
    query_rows = read_jsonl(_path(package_root, JSONL_FILES["queries"]))
    query = next(
        (row for row in query_rows if row.get("query_id") == str(case_id)),
        None,
    )
    if query is None:
        raise KeyError(f"问题集缺少案例：{case_id}")
    evidence_rows = read_jsonl(_path(package_root, JSONL_FILES["evidence"]))
    evidence_by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(evidence_rows, start=1):
        evidence_id = row.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise DatasetContractError(f"evidence 第 {index} 行 evidence_id 为空")
        if evidence_id in evidence_by_id:
            raise DatasetContractError(f"重复 evidence_id：{evidence_id}")
        evidence_by_id[evidence_id] = row
    claims = query.get("reference_claims", [])
    if not isinstance(claims, list):
        raise DatasetContractError(f"案例 {case_id} reference_claims 必须是列表")
    for claim in claims:
        if not isinstance(claim, Mapping):
            raise DatasetContractError(f"案例 {case_id} reference_claims 项必须是对象")
        for evidence_id in claim.get("evidence_ids", []):
            if evidence_id not in evidence_by_id:
                raise DatasetContractError(
                    f"案例 {case_id} 引用了不存在的 evidence：{evidence_id}"
                )
    interaction = query.get("interaction", {})
    if interaction is None:
        interaction = {}
    if not isinstance(interaction, Mapping):
        raise DatasetContractError(f"案例 {case_id} interaction 必须是对象")
    evaluation = interaction.get("evaluation", {})
    if evaluation is None:
        evaluation = {}
    if not isinstance(evaluation, Mapping):
        raise DatasetContractError(
            f"案例 {case_id} interaction.evaluation 必须是对象"
        )
    step_annotations = []
    steps = interaction.get("steps", [])
    if steps is not None:
        if not isinstance(steps, list):
            raise DatasetContractError(f"案例 {case_id} interaction.steps 必须是列表")
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, Mapping):
                raise DatasetContractError(
                    f"案例 {case_id} interaction.steps 第 {index} 项必须是对象"
                )
            if "source_pages" in step:
                source_pages = step["source_pages"]
                if not isinstance(source_pages, list) or any(
                    not isinstance(page, int) or isinstance(page, bool) or page < 1
                    for page in source_pages
                ):
                    raise DatasetContractError(
                        f"案例 {case_id} interaction.steps 第 {index} 项 source_pages 无效"
                    )
                step_annotations.append(
                    {
                        "id": str(step.get("id", "")),
                        "source_pages": list(source_pages),
                    }
                )

    annotation = {
        "expected_pages": [int(page) for page in query.get("expected_pages", [])],
        "essential_evidence_spans": _annotation_spans(query, evidence_by_id),
        "reference_answer": str(query.get("reference_answer", "")),
        "expected_keywords": list(query.get("expected_keywords", [])),
        "evaluation": dict(evaluation),
    }
    if step_annotations:
        annotation["step_annotations"] = step_annotations
    return annotation


def load_query_catalog(
    *,
    root: str | Path | None = None,
) -> list[dict[str, str]]:
    """Return the complete query-only projection for tutorial browsing."""

    return load_query_only(None, root=root)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "CANONICAL_DOCUMENT_ID",
    "CANONICAL_PDF_RELATIVE_PATH",
    "CANONICAL_PDF_PATH",
    "COURSE_ROOT",
    "DATASET_ROOT",
    "DatasetContractError",
    "JSONL_FILES",
    "NORMALIZATION_VERSION",
    "load_annotation",
    "load_chapter_ranges",
    "load_dataset",
    "load_evidence_records",
    "load_search_evidence",
    "load_qrel_records",
    "load_query_catalog",
    "load_query_controls",
    "load_query_only",
    "load_query_records",
    "normalize_query",
    "chapter_title",
    "read_jsonl",
    "sha256_file",
    "validate_pdf_provenance",
    "validate_dataset",
]
