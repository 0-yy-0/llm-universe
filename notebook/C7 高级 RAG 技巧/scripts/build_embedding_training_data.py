#!/usr/bin/env python3
"""Build the C2 semantic query-to-evidence package.

The source of every generated pair is a deterministic, non-overlapping
substring of one canonical Pumpkin Book PDF page.  The substring is measured
with the tokenizer of the BGE model and is kept below a conservative budget,
so the generator, audit, training and evaluation all see the same text.
Generation and semantic auditing both use the real glm-4-flash API.  A
missing/invalid response or stale cache is an error; this command never
fabricates a result or switches to another model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCRIPT = Path(__file__).resolve()
COURSE_ROOT = SCRIPT.parents[1]
DATASET_ROOT = COURSE_ROOT / "data" / "dataset"
SECONDARY_REVIEW_PATH = COURSE_ROOT / "data" / "semantic_review_exclusions.json"
if str(COURSE_ROOT) not in sys.path:
    sys.path.insert(0, str(COURSE_ROOT))

from common.dataset import load_dataset, normalize_query  # noqa: E402
from common.eval_utils import find_local_bge_model, load_pdf_pages, normalize_text  # noqa: E402
from common.nontraining_utils import RAG_LLM_MODEL, llm_call  # noqa: E402


MODEL = "glm-4-flash"
# Version 6 invalidates every v5 cache.  Generation and semantic-audit API
# calls are now one-shot: an invalid response is a hard error, while an
# explicit empty ``items`` list remains a valid candidate rejection.
BUILD_SCHEMA_VERSION = 6
ITEM_SCHEMA_VERSION = 2
AUDIT_SCHEMA_VERSION = 2
CACHE_INTEGRITY_VERSION = 1
CHUNK_TOKEN_BUDGET = 384
MODEL_MAX_SEQ_LENGTH = 512
BATCH_CHUNKS = 1
# Keep each API request small enough to complete within the hard request
# timeout. Two short requests are reliable while the former 8-chunk requests
# repeatedly stalled even with a single worker.
MAX_WORKERS = 2
REVIEW_TYPE = "model_assisted_semantic_training_v3"
GENERATION_PROMPT_VERSION = "c2-fixed-token-chunk-generation-v5.0-one-shot"
AUDIT_PROMPT_VERSION = "c2-fixed-token-chunk-audit-v5.0-one-shot"
GENERATION_TYPES = {"definition", "mechanism", "comparison", "condition"}
MAX_QUESTION_CHARS = 100
MAX_ANSWER_CHARS = 240
MAX_CLAIMS = 3
MAX_CLAIM_CHARS = 140
FORBIDDEN_QUERY_FRAGMENTS = (
    "根据提供的上下文",
    "根据上下文信息",
    "请解释以下陈述",
    "上述内容",
    "这段文字",
    "这里的",
    "该式",
    "该公式",
    "最后一个不等式",
    "in the chunk",
    "described in the chunk",
)


class QueryConflictError(ValueError):
    """A generated question collides after canonical normalization."""


def stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _source_identity_payload(source: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return every source field that can affect a positive example.

    The cache must be bound to the actual canonical chunk, rather than only to
    its id.  Keeping this list explicit also makes the hash stable when the
    source mapping contains unrelated bookkeeping fields.
    """

    source = source or {}
    return {
        key: source.get(key)
        for key in (
            "evidence_id",
            "source_page_id",
            "doc_id",
            "section_id",
            "page",
            "start",
            "end",
            "token_count",
            "token_budget",
            "text",
            "split",
        )
    }


def _item_integrity_payload(
    item: Mapping[str, Any], source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Canonical item payload used to bind a normalized row to its source."""

    return {
        "cache_integrity_version": CACHE_INTEGRITY_VERSION,
        "item_schema_version": ITEM_SCHEMA_VERSION,
        "item_id": item.get("item_id"),
        "source_evidence_id": item.get("source_evidence_id"),
        "page": item.get("page"),
        "section_id": item.get("section_id"),
        "type": item.get("type"),
        "question": item.get("question"),
        "answer": item.get("answer"),
        "claims": item.get("claims"),
        "support_quotes": item.get("support_quotes"),
        "support_chunk_ids": item.get("support_chunk_ids"),
        "source": _source_identity_payload(source),
    }


def _item_input_hash(
    item: Mapping[str, Any], source: Mapping[str, Any] | None,
) -> str:
    return sha256_text(_item_integrity_payload(item, source))


def _audit_input_payload(
    item: Mapping[str, Any], source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Canonical semantic-audit input, including the positive identity."""

    source_payload = _source_identity_payload(source)
    # A direct item-only call is still deterministic.  Internal cache paths
    # always pass the canonical source mapping, which supplies the complete
    # source identity and text.
    if source is None:
        source_payload["text"] = (
            item.get("support_quotes", [None])[0]
            if isinstance(item.get("support_quotes"), list)
            and item.get("support_quotes")
            else None
        )
    return {
        "cache_integrity_version": CACHE_INTEGRITY_VERSION,
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "item_id": item.get("item_id"),
        "question": item.get("question"),
        "answer": item.get("answer"),
        "claims": item.get("claims"),
        "support_quotes": item.get("support_quotes"),
        "positive_evidence_id": item.get("source_evidence_id"),
        "positive_source_identity": source_payload,
    }


def _audit_input_hash(
    item: Mapping[str, Any], source: Mapping[str, Any] | None,
) -> str:
    return sha256_text(_audit_input_payload(item, source))


def chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def parse_strict_json(raw: str) -> dict[str, Any]:
    text = str(raw).strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        first_line, separator, rest = text.partition("\n")
        if (
            separator
            and first_line.strip().lower() in {fence, fence + "json"}
            and rest.rstrip().endswith(fence)
        ):
            text = rest.rstrip()[: -len(fence)].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"glm-4-flash 未返回严格 JSON：{raw!r}") from exc
    if not isinstance(value, dict):
        raise ValueError("glm-4-flash 返回值必须是 JSON object")
    return value


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_secondary_review_exclusions(
    path: Path = SECONDARY_REVIEW_PATH,
) -> dict[str, str]:
    """Load the current independent model-review rejection set."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取二次语义审核清单：{path}") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "review_method",
        "human_verified",
        "rejections",
    }:
        raise ValueError("二次语义审核清单顶层字段无效")
    if value["schema_version"] != 1 or value["review_method"] != "independent_model_review":
        raise ValueError("二次语义审核清单版本或方法无效")
    if value["human_verified"] is not False or not isinstance(value["rejections"], list):
        raise ValueError("二次语义审核不能伪装成人工核验")
    result: dict[str, str] = {}
    for row in value["rejections"]:
        if not isinstance(row, dict) or set(row) != {"candidate_id", "reason"}:
            raise ValueError("二次语义审核 rejection 字段无效")
        candidate_id = row.get("candidate_id")
        reason = row.get("reason")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id.startswith("semantic_candidate_")
            or not isinstance(reason, str)
            or not reason.strip()
            or candidate_id in result
        ):
            raise ValueError("二次语义审核 rejection ID/reason 无效或重复")
        result[candidate_id] = reason.strip()
    return result


@lru_cache(maxsize=1)
def _bge_tokenizer():
    """Load the local BGE tokenizer; a missing cache is a hard failure."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment contract
        raise RuntimeError("构建固定 token chunk 需要 transformers；禁止 fallback") from exc
    model_path = find_local_bge_model()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            use_fast=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("无法加载本地 BGE tokenizer；禁止改用其他 tokenizer") from exc
    if tokenizer is None:
        raise RuntimeError("本地 BGE tokenizer 为空")
    return tokenizer


def token_count(text: str) -> int:
    value = _bge_tokenizer()(
        text,
        add_special_tokens=True,
        truncation=False,
        verbose=False,
    )
    ids = value.get("input_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError("BGE tokenizer 没有返回 input_ids")
    return len(ids)


def split_fixed_token_chunks(
    source_page_id: str,
    text: str,
    *,
    token_budget: int = CHUNK_TOKEN_BUDGET,
) -> list[dict[str, Any]]:
    """Split a page once, deterministically, without consulting any query."""

    if token_budget <= 0 or token_budget >= MODEL_MAX_SEQ_LENGTH:
        raise ValueError("fixed token budget 必须小于 BGE 的 512 token 输入上限")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"来源页为空：{source_page_id}")
    result: list[dict[str, Any]] = []
    start = 0
    while start < len(text):
        low, high = start + 1, len(text)
        best: int | None = None
        while low <= high:
            middle = (low + high) // 2
            count = token_count(text[start:middle])
            if count <= token_budget:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best is None or best <= start:
            raise ValueError(
                f"无法在 token budget 内切分 {source_page_id}，offset={start}"
            )
        value = text[start:best]
        count = token_count(value)
        if count <= 0 or count > token_budget:
            raise AssertionError("固定 chunk token 数量超出预算")
        evidence_id = stable_id(
            "evi_ft_chunk",
            source_page_id,
            start,
            best,
            value,
            token_budget,
        )
        result.append(
            {
                "evidence_id": evidence_id,
                "chunk_id": evidence_id,
                "source_page_id": source_page_id,
                "start": start,
                "end": best,
                "text": value,
                "token_count": count,
                "token_budget": token_budget,
            }
        )
        start = best
    return result


def _configured_source_pages(package: Mapping[str, Any]) -> dict[str, list[int]]:
    construction = package["manifest"].get("data_construction", {})
    configured_pages = construction.get("source_pages_by_split")
    result: dict[str, list[int]] = {}
    if isinstance(configured_pages, dict) and set(configured_pages) == {"train", "dev", "test"}:
        for split in ("train", "dev", "test"):
            values = configured_pages[split]
            if not isinstance(values, list) or not values:
                raise ValueError(f"manifest source_pages_by_split 无效：{split}")
            if any(not isinstance(page, int) or page <= 0 for page in values):
                raise ValueError(f"manifest source_pages_by_split 页码无效：{split}")
            result[split] = list(dict.fromkeys(values))
    else:
        configured_ids = construction.get("candidate_source_evidence_ids_by_split")
        if not isinstance(configured_ids, dict) or set(configured_ids) != {"train", "dev", "test"}:
            raise ValueError("manifest 缺少 C2 来源页清单")
        evidence_by_id = {row["evidence_id"]: row for row in package["evidence"]}
        for split in ("train", "dev", "test"):
            values = configured_ids[split]
            if not isinstance(values, list) or not values:
                raise ValueError(f"manifest candidate source list 无效：{split}")
            pages: list[int] = []
            for evidence_id in values:
                evidence = evidence_by_id.get(evidence_id)
                if evidence is None:
                    raise ValueError(f"manifest 来源 evidence 不存在：{evidence_id}")
                if not isinstance(evidence.get("page"), int) or evidence["page"] <= 0:
                    raise ValueError(f"manifest 来源页码无效：{evidence_id}")
                pages.append(int(evidence["page"]))
            result[split] = list(dict.fromkeys(pages))
    all_pages = [page for values in result.values() for page in values]
    if len(all_pages) != len(set(all_pages)):
        raise ValueError("C2 来源页在 train/dev/test 之间重复")
    return result


def load_sources(
    package: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load canonical PDF pages and materialize fixed-token chunks once."""

    page_rows = {int(row["page"]): str(row["text"]) for row in load_pdf_pages()}
    source_pages = _configured_source_pages(package)
    evidence_by_id = {row["evidence_id"]: row for row in package["evidence"]}
    configured_ids = package["manifest"].get("data_construction", {}).get(
        "candidate_source_evidence_ids_by_split", {}
    )
    page_section: dict[int, str] = {}
    if isinstance(configured_ids, dict):
        for values in configured_ids.values():
            for evidence_id in values:
                row = evidence_by_id.get(evidence_id)
                if isinstance(row, dict) and isinstance(row.get("page"), int):
                    page = int(row["page"])
                    page_section[page] = str(
                        row.get("section_id", f"doc_pumpkin_book:page_{page:04d}")
                    )
    sources: list[dict[str, Any]] = []
    for split in ("train", "dev", "test"):
        for page in source_pages[split]:
            text = page_rows.get(page)
            if not text:
                raise ValueError(f"PDF 缺少 C2 来源页：{page}")
            source_page_id = stable_id("source_page", "doc_pumpkin_book", page)
            section_id = page_section.get(page, f"doc_pumpkin_book:page_{page:04d}")
            for chunk in split_fixed_token_chunks(source_page_id, text):
                sources.append(
                    {
                        **chunk,
                        "doc_id": "doc_pumpkin_book",
                        "page": page,
                        "section_id": section_id,
                        "split": split,
                    }
                )
    if not sources:
        raise ValueError("C2 fixed-token source chunks 为空")
    split_counts = {
        split: sum(row["split"] == split for row in sources)
        for split in ("train", "dev", "test")
    }
    if any(count == 0 for count in split_counts.values()):
        raise ValueError(f"fixed-token chunks 缺少 split：{split_counts}")
    return sorted(sources, key=lambda row: (row["split"], row["page"], row["start"])), []


def _base_query_norms(package: Mapping[str, Any]) -> set[str]:
    return {
        normalize_query(row["text"])
        for row in package["queries"]
        if isinstance(row.get("text"), str)
        and row.get("text", "").strip()
        and not str(row.get("query_id", "")).startswith("semantic_query_")
    }


def _source_fingerprint(sources: Sequence[Mapping[str, Any]]) -> str:
    return sha256_text(
        [
            {
                "evidence_id": row["evidence_id"],
                "source_page_id": row["source_page_id"],
                "page": row["page"],
                "section_id": row["section_id"],
                "start": row["start"],
                "end": row["end"],
                "token_count": row["token_count"],
                "token_budget": row["token_budget"],
                "text": row["text"],
                "split": row["split"],
            }
            for row in sources
        ]
    )


def cache_metadata(
    sources: Sequence[Mapping[str, Any]],
    *,
    existing_query_norms: Iterable[str] = (),
) -> dict[str, Any]:
    source_fingerprint = _source_fingerprint(sources)
    prompt_fingerprint = sha256_text(
        {
            "generation_prompt_version": GENERATION_PROMPT_VERSION,
            "audit_prompt_version": AUDIT_PROMPT_VERSION,
            "generation_schema": {
                "top_level": "chunks",
                "item_fields": ["type", "question", "answer", "claims", "support_chunk_ids"],
                "max_items_per_chunk": 1,
                "batch_chunks": BATCH_CHUNKS,
                "max_question_chars": MAX_QUESTION_CHARS,
                "max_answer_chars": MAX_ANSWER_CHARS,
                "max_claims": MAX_CLAIMS,
                "max_claim_chars": MAX_CLAIM_CHARS,
            },
            "audit_schema": {
                "top_level": "items",
                "boolean_fields": [
                    "question_self_contained",
                    "positive_fully_supports",
                    "answer_fully_supported",
                    "support_quotes_sufficient",
                ],
                "input_hash_field": "input_hash",
                "audit_schema_version": AUDIT_SCHEMA_VERSION,
            },
            "item_schema_version": ITEM_SCHEMA_VERSION,
            "cache_integrity_version": CACHE_INTEGRITY_VERSION,
        }
    )
    input_fingerprint = sha256_text(
        {
            "schema_version": BUILD_SCHEMA_VERSION,
            "item_schema_version": ITEM_SCHEMA_VERSION,
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            "cache_integrity_version": CACHE_INTEGRITY_VERSION,
            "model": MODEL,
            "chunk_token_budget": CHUNK_TOKEN_BUDGET,
            "batch_chunks": BATCH_CHUNKS,
            "model_max_seq_length": MODEL_MAX_SEQ_LENGTH,
            "source_chunks_fingerprint": source_fingerprint,
            "prompt_fingerprint": prompt_fingerprint,
            "existing_query_norms": sorted(set(existing_query_norms)),
        }
    )
    return {
        "schema_version": BUILD_SCHEMA_VERSION,
        "item_schema_version": ITEM_SCHEMA_VERSION,
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "cache_integrity_version": CACHE_INTEGRITY_VERSION,
        "model": MODEL,
        "chunk_token_budget": CHUNK_TOKEN_BUDGET,
        "batch_chunks": BATCH_CHUNKS,
        "model_max_seq_length": MODEL_MAX_SEQ_LENGTH,
        "generation_prompt_version": GENERATION_PROMPT_VERSION,
        "audit_prompt_version": AUDIT_PROMPT_VERSION,
        "prompt_fingerprint": prompt_fingerprint,
        "source_chunks_fingerprint": source_fingerprint,
        "input_fingerprint": input_fingerprint,
    }


def generation_prompt(
    batch: Sequence[Mapping[str, Any]],
    *,
    reserved_query_norms: Iterable[str] = (),
) -> str:
    records = [
        {
            "chunk_id": row["evidence_id"],
            "page": row["page"],
            "text": row["text"],
        }
        for row in batch
    ]
    reserved = sorted(set(reserved_query_norms))
    return f"""你在为《南瓜书》中文检索器构造 query→evidence 弱监督数据。只能阅读每个 CHUNK 的 TEXT，不得使用外部知识或页面之外的上下文。

必须逐一处理 CHUNKS 中的每个 chunk_id，并在输出 chunks 数组中为每个 id 保留恰好一个对象（即使 items 为空也不能省略）。每个 CHUNK 最多生成 1 道问题；TEXT 只有不足以支持一个自包含问答时就输出空 items，不要为了凑数量补题，也不要按页面固定生成四类。若生成问题，type 只能是 definition、mechanism、comparison、condition 之一，但类型只描述原文真正支持的问法。

每道问题必须独立可理解，不出现“根据上下文/上述内容”等模板；question 不超过 100 个字符，answer 不超过 240 个字符；claims 为 1 到 3 条，每条不超过 140 个字符。answer 和 claims 只需概括直接回答问题的最小事实，不能整段抄写公式推导或 TEXT，也不得添加公式含义、例子、条件或推论。support_chunk_ids 必须只写当前 CHUNK 的原样 chunk_id，不能自造 ID。问题不能与 RESERVED_QUERY_NORMS 规范化后重复，也不能和本批其他问题重复。

只输出严格 JSON，不要代码围栏：
{{"chunks":[{{"chunk_id":"...","items":[{{"type":"definition|mechanism|comparison|condition","question":"...","answer":"...","claims":["..."],"support_chunk_ids":["..."]}}]}}]}}

CHUNKS:
{json.dumps(records, ensure_ascii=False)}

RESERVED_QUERY_NORMS:
{json.dumps(reserved, ensure_ascii=False)}"""


def _normalise_item(
    item: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    reserved_query_norms: set[str],
    local_query_norms: set[str],
) -> dict[str, Any]:
    required = {"type", "question", "answer", "claims", "support_chunk_ids"}
    if set(item) != required:
        raise ValueError(f"{source['evidence_id']} item 字段不严格")
    item_type = item.get("type")
    if item_type not in GENERATION_TYPES:
        raise ValueError(f"问题类型无效：{item_type!r}")
    question = normalize_text(item.get("question"))
    answer = normalize_text(item.get("answer"))
    raw_claims = item.get("claims")
    if not isinstance(raw_claims, list) or not raw_claims:
        raise ValueError(f"{source['evidence_id']} 问题缺少 claims")
    claims = [normalize_text(claim) for claim in raw_claims]
    if not question or not answer or any(not claim for claim in claims):
        raise ValueError(f"{source['evidence_id']} 问题/答案/claims 不能为空")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"{source['evidence_id']} question 超过长度限制")
    if len(answer) > MAX_ANSWER_CHARS:
        raise ValueError(f"{source['evidence_id']} answer 超过长度限制")
    if len(claims) > MAX_CLAIMS or any(len(claim) > MAX_CLAIM_CHARS for claim in claims):
        raise ValueError(f"{source['evidence_id']} claims 超过数量或长度限制")
    if any(fragment in question for fragment in FORBIDDEN_QUERY_FRAGMENTS):
        raise ValueError(f"问题依赖上下文模板：{question!r}")
    query_norm = normalize_query(question)
    if not query_norm or query_norm in reserved_query_norms or query_norm in local_query_norms:
        raise QueryConflictError(f"规范化 query 冲突：{question!r}")
    support_ids = item.get("support_chunk_ids")
    if support_ids != [source["evidence_id"]]:
        raise ValueError(
            f"{question!r} 必须绑定当前已有 chunk：{source['evidence_id']}"
        )
    local_query_norms.add(query_norm)
    normalized = {
        "item_id": stable_id("train_item", source["evidence_id"], query_norm),
        "source_evidence_id": source["evidence_id"],
        "page": source["page"],
        "section_id": source["section_id"],
        "type": item_type,
        "question": question,
        "answer": answer,
        "claims": claims,
        "support_quotes": [source["text"]],
        "support_chunk_ids": [source["evidence_id"]],
    }
    normalized["input_hash"] = _item_input_hash(normalized, source)
    return normalized


def validate_generation(
    value: Mapping[str, Any],
    batch: Sequence[Mapping[str, Any]],
    *,
    reserved_query_norms: Iterable[str] = (),
    drop_query_conflicts: bool = False,
) -> list[dict[str, Any]]:
    if set(value) != {"chunks"} or not isinstance(value["chunks"], list):
        raise ValueError("生成 JSON 顶层必须只有 chunks")
    expected = {row["evidence_id"]: row for row in batch}
    returned_ids = {row.get("chunk_id") for row in value["chunks"] if isinstance(row, dict)}
    if returned_ids != set(expected):
        raise ValueError("生成结果 chunk_id 不完整或越界")
    output: list[dict[str, Any]] = []
    local_query_norms: set[str] = set()
    for chunk_result in value["chunks"]:
        if not isinstance(chunk_result, dict) or set(chunk_result) != {"chunk_id", "items"}:
            raise ValueError("chunk 结果字段不严格")
        source = expected[chunk_result["chunk_id"]]
        items = chunk_result["items"]
        if not isinstance(items, list) or len(items) > 1:
            raise ValueError(f"{source['evidence_id']} 每个 chunk 最多 1 道问法")
        for item in items:
            try:
                output.append(
                    _normalise_item(
                        item,
                        source,
                        reserved_query_norms=set(reserved_query_norms),
                        local_query_norms=local_query_norms,
                    )
                )
            except QueryConflictError:
                if not drop_query_conflicts:
                    raise
    return output


def audit_prompt(
    items: Sequence[Mapping[str, Any]],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
) -> str:
    records = []
    for item in items:
        evidence_id = item["source_evidence_id"]
        if evidence_id not in evidence_by_id:
            raise ValueError(f"audit item 引用未知 chunk：{evidence_id}")
        records.append(
            {
                "item_id": item["item_id"],
                "question": item["question"],
                "answer": item["answer"],
                "claims": item["claims"],
                "positive_evidence_id": evidence_id,
                "support_quotes": item["support_quotes"],
                "input_hash": _audit_input_hash(item, evidence_by_id[evidence_id]),
            }
        )
    evidence = [
        {
            "evidence_id": evidence_id,
            "source_page_id": evidence_by_id[evidence_id]["source_page_id"],
            "section_id": evidence_by_id[evidence_id]["section_id"],
            "page": evidence_by_id[evidence_id]["page"],
            "start": evidence_by_id[evidence_id]["start"],
            "end": evidence_by_id[evidence_id]["end"],
            "token_count": evidence_by_id[evidence_id]["token_count"],
            "token_budget": evidence_by_id[evidence_id]["token_budget"],
            "text": evidence_by_id[evidence_id]["text"],
        }
        for evidence_id in sorted({item["source_evidence_id"] for item in items})
    ]
    return f"""你是训练数据的第二阶段语义审核器。只能依据 INPUT 和 EVIDENCE，不得使用外部知识。逐条检查：问题自包含；positive evidence 能完整回答问题；answer 的全部主要事实被 positive 直接支持；support_quotes 与 positive 完全一致且足够。不能因为主题相似就通过。

所有字段都必须真实反映检查结果。不要修写数据，不要把不确定项放行。只输出严格 JSON，不要代码围栏、解释或额外字段：
{{"items":[{{"item_id":"...","input_hash":"64 lowercase hex chars","question_self_contained":true,"positive_fully_supports":true,"answer_fully_supported":true,"support_quotes_sufficient":true}}]}}

INPUT:
{json.dumps(records, ensure_ascii=False)}

EVIDENCE:
{json.dumps(evidence, ensure_ascii=False)}"""


def validate_audit(
    value: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    evidence_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError("审核 JSON 必须是 object")
    if set(value) != {"items"} or not isinstance(value["items"], list):
        raise ValueError("审核 JSON 顶层必须只有 items")
    if any(
        not isinstance(item, Mapping) or not isinstance(item.get("item_id"), str)
        for item in items
    ):
        raise ValueError("审核输入 item_id 必须是字符串")
    expected = {item["item_id"]: item for item in items}
    rows = value["items"]
    if len(rows) != len(expected) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("审核 item_id 不完整或越界")
    if {row.get("item_id") for row in rows} != set(expected):
        raise ValueError("审核 item_id 不完整或越界")
    fields = {
        "item_id",
        "input_hash",
        "question_self_contained",
        "positive_fully_supports",
        "answer_fully_supported",
        "support_quotes_sufficient",
    }
    checked: dict[str, dict[str, Any]] = {}
    for row in rows:
        if set(row) != fields:
            raise ValueError("审核 item 字段不严格")
        boolean_fields = fields - {"item_id", "input_hash"}
        if any(not isinstance(row[key], bool) for key in boolean_fields):
            raise ValueError(f"审核布尔字段必须是真实 bool：{row}")
        item_id = row["item_id"]
        item = expected[item_id]
        source = None
        if evidence_by_id is not None:
            evidence_id = item.get("source_evidence_id")
            source = evidence_by_id.get(evidence_id)
            if source is None:
                raise ValueError(f"审核 item 引用了未知 positive chunk：{evidence_id}")
        if row["input_hash"] != _audit_input_hash(item, source):
            raise ValueError(f"{item_id} semantic audit input hash 不匹配")
        checked[item_id] = dict(row)
    return checked


def _validate_audit_entry(
    audit: Mapping[str, Any],
    item: Mapping[str, Any],
    source: Mapping[str, Any] | None,
) -> None:
    fields = {
        "item_id",
        "input_hash",
        "question_self_contained",
        "positive_fully_supports",
        "answer_fully_supported",
        "support_quotes_sufficient",
    }
    if set(audit) != fields:
        raise ValueError(f"{item.get('item_id')} 审核 item 字段不严格")
    if audit.get("item_id") != item.get("item_id"):
        raise ValueError(f"{item.get('item_id')} audit item_id 不匹配")
    if not isinstance(audit.get("input_hash"), str):
        raise ValueError(f"{item.get('item_id')} audit input_hash 必须是字符串")
    digest = audit["input_hash"]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{item.get('item_id')} audit input_hash 格式无效")
    boolean_fields = fields - {"item_id", "input_hash"}
    if any(not isinstance(audit.get(key), bool) for key in boolean_fields):
        raise ValueError(f"{item.get('item_id')} 审核布尔值必须是真实 bool")
    expected_hash = _audit_input_hash(item, source)
    if digest != expected_hash:
        raise ValueError(f"{item.get('item_id')} semantic audit input hash 不匹配")


def _validate_cached_items(
    items: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    *,
    reserved_query_norms: Iterable[str],
    allow_query_conflicts: bool = False,
) -> None:
    if not isinstance(items, list):
        raise ValueError("生成缓存 items 必须是 list")
    source_by_id = {row["evidence_id"]: row for row in sources}
    seen_ids: set[str] = set()
    seen_norms = set(reserved_query_norms)
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("生成缓存 item 必须是 object")
        expected_fields = {
            "item_id",
            "source_evidence_id",
            "page",
            "section_id",
            "type",
            "question",
            "answer",
            "claims",
            "support_quotes",
            "support_chunk_ids",
            "input_hash",
        }
        if set(item) != expected_fields:
            raise ValueError("生成缓存 item schema 不严格；禁止复用")
        if not isinstance(item.get("item_id"), str):
            raise ValueError("生成缓存 item_id 必须是字符串")
        source_id = item.get("source_evidence_id")
        if not isinstance(source_id, str):
            raise ValueError("生成缓存 source_evidence_id 必须是字符串")
        source = source_by_id.get(source_id)
        if source is None:
            raise ValueError(f"生成缓存引用了不存在的 source chunk：{source_id}")
        if item.get("item_id") in seen_ids:
            raise ValueError(f"生成缓存 item_id 重复：{item.get('item_id')}")
        seen_ids.add(item.get("item_id"))

        # Re-run the current generation normalizer against every cached row.
        # This checks all fields, recomputes the stable id, and recomputes the
        # item input hash.  A source/quote-only check is deliberately not
        # sufficient for cache reuse.
        raw_item = {
            key: item[key]
            for key in ("type", "question", "answer", "claims", "support_chunk_ids")
        }
        try:
            normalized = _normalise_item(
                raw_item,
                source,
                reserved_query_norms=set(),
                local_query_norms=set(),
            )
        except QueryConflictError as exc:
            raise ValueError(f"{item.get('item_id')} 缓存 query 规范化失败") from exc
        if dict(item) != normalized:
            raise ValueError(
                f"{item.get('item_id')} 缓存 item 完整 schema/稳定 item_id/input hash 不匹配"
            )
        question = normalize_text(item.get("question"))
        query_norm = normalize_query(question)
        if not query_norm:
            raise ValueError(f"生成缓存规范化 query 冲突：{question!r}")
        if query_norm in seen_norms and not allow_query_conflicts:
            raise ValueError(f"生成缓存规范化 query 冲突：{question!r}")
        seen_norms.add(query_norm)
        if item.get("support_chunk_ids") != [source_id]:
            raise ValueError(f"{item.get('item_id')} 未绑定唯一 canonical chunk")
        if item.get("support_quotes") != [source["text"]]:
            raise ValueError(f"{item.get('item_id')} support quote 不是 chunk 原文")
        if item.get("page") != source["page"] or item.get("section_id") != source["section_id"]:
            raise ValueError(f"{item.get('item_id')} 来源页/section 不一致")


def _validate_cached_audits(
    items: Sequence[Mapping[str, Any]],
    audits: Mapping[str, Any],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
    *,
    require_complete: bool = False,
) -> None:
    """Validate every cached audit and bind it to the current item/source."""

    if not isinstance(audits, dict):
        raise ValueError("生成缓存 audits 必须是 object")
    if not isinstance(items, list) or any(
        not isinstance(item, Mapping) or not isinstance(item.get("item_id"), str)
        for item in items
    ):
        raise ValueError("审核缓存输入 items 必须包含字符串 item_id")
    items_by_id = {item["item_id"]: item for item in items}
    audit_ids = set(audits)
    item_ids = set(items_by_id)
    if not audit_ids.issubset(item_ids):
        extra = sorted(audit_ids - item_ids)
        raise ValueError(f"审核缓存包含不存在的 item：{extra[:5]}")
    if require_complete and audit_ids != item_ids:
        missing = sorted(item_ids - audit_ids)
        raise ValueError(f"审核缓存缺少 item：{missing[:5]}")
    for item_id, audit in audits.items():
        if not isinstance(audit, Mapping):
            raise ValueError(f"{item_id} 审核缓存必须是 object")
        item = items_by_id[item_id]
        source = evidence_by_id.get(item.get("source_evidence_id"))
        if source is None:
            raise ValueError(f"{item_id} 审核缓存引用未知 source chunk")
        _validate_audit_entry(audit, item, source)


def _validate_generated_by_source(
    value: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    *,
    reserved_query_norms: Iterable[str],
    require_all_sources: bool = False,
) -> list[Mapping[str, Any]]:
    """Validate the per-source item cache before it can be reused."""

    generated_by_source = value.get("generated_by_source")
    if not isinstance(generated_by_source, dict):
        raise ValueError("生成缓存缺少 generated_by_source")
    source_ids = {row["evidence_id"] for row in sources}
    unknown_source_ids = set(generated_by_source) - source_ids
    if unknown_source_ids:
        raise ValueError(
            f"生成缓存包含未知 source chunks：{sorted(unknown_source_ids)[:5]}"
        )
    if require_all_sources and set(generated_by_source) != source_ids:
        missing = sorted(source_ids - set(generated_by_source))
        raise ValueError(f"生成缓存缺少 source chunks：{missing[:5]}")
    if any(not isinstance(rows, list) for rows in generated_by_source.values()):
        raise ValueError("生成缓存 generated_by_source 的每项必须是 list")
    all_items = [
        item
        for rows in generated_by_source.values()
        for item in rows
    ]
    _validate_cached_items(
        all_items,
        sources,
        reserved_query_norms=reserved_query_norms,
        allow_query_conflicts=True,
    )
    return all_items


def _validate_cache_metadata(
    value: Mapping[str, Any],
    expected_metadata: Mapping[str, Any],
    *,
    label: str,
) -> None:
    actual = value.get("cache_metadata")
    if actual != dict(expected_metadata):
        raise ValueError(
            f"{label} 与当前来源 chunks、prompt/schema/model 不匹配；禁止复用"
        )
    for key, expected in expected_metadata.items():
        if value.get(key) != expected:
            raise ValueError(f"{label} 顶层 {key} 与当前构建不匹配")
    if not isinstance(value.get("generated_by_source"), dict):
        raise ValueError(f"{label} 缺少 generated_by_source")
    if not isinstance(value.get("generation_rejections_by_source"), dict):
        raise ValueError(f"{label} 缺少 generation_rejections_by_source")
    if not isinstance(value.get("audits"), dict):
        raise ValueError(f"{label} 缺少 audits")


def _new_partial(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cache_metadata": dict(metadata),
        **dict(metadata),
        "generated_by_source": {},
        "generation_rejections_by_source": {},
        "audits": {},
    }


def _load_or_init_partial(
    path: Path,
    metadata: Mapping[str, Any],
    *,
    sources: Sequence[Mapping[str, Any]] | None = None,
    reserved_query_norms: Iterable[str] = (),
) -> dict[str, Any]:
    if not path.is_file():
        return _new_partial(metadata)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"partial cache 不是有效 JSON：{path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"partial cache 必须是 object：{path}")
    _validate_cache_metadata(value, metadata, label="partial cache")
    if sources is not None:
        all_existing = _validate_generated_by_source(
            value,
            sources,
            reserved_query_norms=reserved_query_norms,
        )
        _validate_cached_audits(
            all_existing,
            value["audits"],
            {row["evidence_id"]: row for row in sources},
        )
    return value


def _save_partial(path: Path, partial: Mapping[str, Any]) -> None:
    write_json(path, partial)


def generate_dataset(
    sources: list[dict[str, Any]],
    *,
    cache_path: Path,
    partial_cache_path: Path,
    existing_query_norms: Iterable[str] = (),
) -> dict[str, Any]:
    if RAG_LLM_MODEL != MODEL:
        raise RuntimeError(f"本构建固定要求 {MODEL}，实际为 {RAG_LLM_MODEL}")
    existing_query_norms = set(existing_query_norms)
    metadata = cache_metadata(sources, existing_query_norms=existing_query_norms)
    partial = _load_or_init_partial(
        partial_cache_path,
        metadata,
        sources=sources,
        reserved_query_norms=existing_query_norms,
    )
    generated_by_source = partial["generated_by_source"]
    generation_rejections = partial.get("generation_rejections_by_source")
    if not isinstance(generation_rejections, dict):
        raise ValueError("partial cache 缺少 generation_rejections_by_source")
    all_existing = [
        item
        for rows in generated_by_source.values()
        if isinstance(rows, list)
        for item in rows
    ]
    _validate_cached_items(
        all_existing,
        sources,
        reserved_query_norms=existing_query_norms,
        # A process can stop after generation writes batches but before the
        # deterministic global dedup pass. Validate those rows structurally,
        # then let the pass below resolve conflicts in source order.
        allow_query_conflicts=True,
    )
    source_by_id = {row["evidence_id"]: row for row in sources}
    _validate_cached_audits(
        all_existing,
        partial["audits"],
        source_by_id,
    )
    source_norms = set(existing_query_norms)
    source_norms.update(normalize_query(item["question"]) for item in all_existing)
    generation_conflict_count = 0
    pending_generation = []
    for number, batch in enumerate(chunks(sources, BATCH_CHUNKS), 1):
        batch_ids = [row["evidence_id"] for row in batch]
        if all(source_id in generated_by_source for source_id in batch_ids):
            print(f"generation {number}: cached", flush=True)
            continue
        pending_generation.append((number, list(batch), batch_ids))

    def _generate(job):
        number, batch, batch_ids = job
        prompt = generation_prompt(
            batch, reserved_query_norms=existing_query_norms
        )
        # One llm_call means one synchronous SDK/HTTP request.  Do not turn a
        # malformed response into a hidden second request: callers need the
        # failure and its exact request count to remain observable.
        raw = llm_call(prompt, max_tokens=700)
        value = parse_strict_json(raw)
        rows = validate_generation(
            value,
            batch,
            reserved_query_norms=existing_query_norms,
            drop_query_conflicts=True,
        )
        response_items = [
            item
            for chunk_result in value.get("chunks", [])
            if isinstance(chunk_result, dict)
            for item in (chunk_result.get("items", []) or [])
        ]
        return number, batch_ids, rows, len(response_items) - len(rows), {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_generate, job) for job in pending_generation]
        for future in as_completed(futures):
            number, batch_ids, rows, conflicts, rejected_sources = future.result()
            generation_conflict_count += conflicts
            for source_id in batch_ids:
                generated_by_source[source_id] = [
                    row for row in rows if row["source_evidence_id"] == source_id
                ]
                generation_rejections.pop(source_id, None)
            generation_rejections.update(rejected_sources)
            _save_partial(partial_cache_path, partial)
            print(
                f"generation {number}: {len(rows)} retained candidates"
                + (f", {len(rejected_sources)} schema-rejected" if rejected_sources else ""),
                flush=True,
            )

    generated = [
        item
        for source in sources
        for item in generated_by_source.get(source["evidence_id"], [])
    ]
    if set(generated_by_source) != set(source_by_id):
        missing = sorted(set(source_by_id) - set(generated_by_source))
        raise ValueError(f"生成缓存缺少 source chunks：{missing[:5]}")
    deduplicated: list[dict[str, Any]] = []
    seen_norms = set(existing_query_norms)
    for source in sources:
        source_id = source["evidence_id"]
        kept: list[dict[str, Any]] = []
        for item in generated_by_source.get(source_id, []):
            normalized = normalize_query(item["question"])
            if normalized in seen_norms:
                generation_conflict_count += 1
                continue
            seen_norms.add(normalized)
            kept.append(item)
            deduplicated.append(item)
        generated_by_source[source_id] = kept
    generated = deduplicated
    _validate_cached_items(
        generated,
        sources,
        reserved_query_norms=existing_query_norms,
    )
    _save_partial(partial_cache_path, partial)
    audits = partial["audits"]
    evidence_by_id = {row["evidence_id"]: row for row in sources}
    pending_audits = []
    for number, batch_sources in enumerate(chunks(sources, BATCH_CHUNKS), 1):
        ids = {row["evidence_id"] for row in batch_sources}
        batch_items = [row for row in generated if row["source_evidence_id"] in ids]
        if not batch_items:
            continue
        if all(item["item_id"] in audits for item in batch_items):
            print(f"audit {number}: cached", flush=True)
            continue
        pending_audits.append((number, batch_items))

    def _audit(job):
        number, batch_items = job
        prompt = audit_prompt(batch_items, evidence_by_id)
        raw = llm_call(prompt, max_tokens=350)
        checked = validate_audit(
            parse_strict_json(raw),
            batch_items,
            evidence_by_id,
        )
        return number, batch_items, checked

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_audit, job) for job in pending_audits]
        for future in as_completed(futures):
            number, batch_items, checked = future.result()
            audits.update(checked)
            _save_partial(partial_cache_path, partial)
            print(f"audit {number}: {len(batch_items)} items", flush=True)

    generated_ids = {item["item_id"] for item in generated}
    if not generated_ids.issubset(audits):
        raise ValueError("语义审核缓存缺少 item 或布尔结果")
    _validate_cached_audits(
        generated,
        audits,
        evidence_by_id,
        require_complete=True,
    )
    retained: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    boolean_fields = (
        "question_self_contained",
        "positive_fully_supports",
        "answer_fully_supported",
        "support_quotes_sufficient",
    )
    for item in generated:
        audit = audits[item["item_id"]]
        if not isinstance(audit, dict) or any(
            not isinstance(audit.get(key), bool) for key in boolean_fields
        ):
            raise ValueError(f"{item['item_id']} 审核布尔值缺失或不是 bool")
        if all(audit[key] for key in boolean_fields):
            retained.append(item)
        else:
            failed.append({"item_id": item["item_id"], "audit": dict(audit)})
    source_ids_by_split = {
        split: {row["evidence_id"] for row in sources if row["split"] == split}
        for split in ("train", "dev", "test")
    }
    retained_by_split = {
        split: sum(item["source_evidence_id"] in source_ids_by_split[split] for item in retained)
        for split in ("train", "dev", "test")
    }
    minimums = {"train": 32, "dev": 8, "test": 8}
    if any(retained_by_split[split] < minimums[split] for split in minimums):
        raise ValueError(f"语义审核后各 split 合格 pair 数不足：{retained_by_split}")
    retained_audits = {item["item_id"]: dict(audits[item["item_id"]]) for item in retained}
    generated_payload = {
        "cache_metadata": metadata,
        **metadata,
        "generated_by_source": {
            source_id: rows
            for source_id, rows in partial["generated_by_source"].items()
            if source_id in {source["evidence_id"] for source in sources}
        },
        "generation_rejections_by_source": dict(generation_rejections),
        "items": retained,
        "audits": retained_audits,
        "curation": {
            "source_pages_reviewed": len({row["source_page_id"] for row in sources}),
            "source_pages_retained": len({row["source_page_id"] for row in sources}),
            "source_pages_excluded": 0,
            "source_chunks_reviewed": len(sources),
            "source_chunks_with_retained_items": len(
                {item["source_evidence_id"] for item in retained}
            ),
            "retained_items_by_split": retained_by_split,
            "rejected_item_count": len(failed),
            "generation_schema_rejection_count": len(generation_rejections),
            "generation_query_conflict_count": generation_conflict_count,
            "rejected_items": failed,
            "rule": "reject_schema-invalid_generation_and_false_semantic_audit; never_exclude_source_page",
        },
    }
    _validate_cache_metadata(generated_payload, metadata, label="generated cache")
    _validate_cached_items(
        generated_payload["items"],
        sources,
        reserved_query_norms=existing_query_norms,
    )
    _validate_cached_audits(
        generated_payload["items"],
        generated_payload["audits"],
        evidence_by_id,
        require_complete=True,
    )
    write_json(cache_path, generated_payload)
    return generated_payload


def _review_from_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    boolean_fields = (
        "question_self_contained",
        "positive_fully_supports",
        "answer_fully_supported",
        "support_quotes_sufficient",
    )
    if any(audit.get(key) is not True for key in boolean_fields):
        raise ValueError("只有四个审核布尔值全部为真实 True 的 item 才能进入 canonical")
    return {
        **{key: bool(audit[key]) for key in boolean_fields},
        "status": "passed",
        "review_type": REVIEW_TYPE,
        "human_verified": False,
        "note": "通过真实 glm-4-flash 语义审核；未声称人工核验。",
    }


def _generated_ids(package: Mapping[str, Any]) -> tuple[set[str], set[str], set[str]]:
    query_ids = {
        str(row.get("query_id"))
        for row in package["queries"]
        if str(row.get("query_id", "")).startswith("semantic_query_")
    }
    candidate_ids = {
        str(row.get("candidate_id"))
        for row in package["qa_candidates"]
        if str(row.get("candidate_id", "")).startswith("semantic_candidate_")
    }
    evidence_ids = {
        str(row.get("evidence_id"))
        for row in package["evidence"]
        if str(row.get("evidence_id", "")).startswith("evi_ft_")
    }
    return query_ids, candidate_ids, evidence_ids


def _base_split(package: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(package["splits"])
    for split in ("train", "dev", "test"):
        value["assignments"][split] = [
            query_id
            for query_id in value["assignments"].get(split, [])
            if not str(query_id).startswith("semantic_query_")
        ]
    value["groups"] = {
        query_id: group
        for query_id, group in value.get("groups", {}).items()
        if not str(query_id).startswith("semantic_query_")
    }
    return value


def build_records(
    package: Mapping[str, Any],
    sources: list[dict[str, Any]],
    generated: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = cache_metadata(sources, existing_query_norms=_base_query_norms(package))
    _validate_cache_metadata(generated, metadata, label="generated cache")
    items = generated.get("items")
    audits = generated.get("audits")
    if not isinstance(items, list) or not isinstance(audits, dict):
        raise ValueError("generated 缺少 items/audits")
    _validate_generated_by_source(
        generated,
        sources,
        reserved_query_norms=_base_query_norms(package),
    )
    _validate_cached_items(
        items,
        sources,
        reserved_query_norms=_base_query_norms(package),
    )
    source_by_id = {row["evidence_id"]: row for row in sources}
    _validate_cached_audits(
        items,
        audits,
        source_by_id,
        require_complete=True,
    )
    exclusions = load_secondary_review_exclusions()
    candidate_id_by_item = {
        item["item_id"]: stable_id(
            "semantic_candidate",
            stable_id(
                "semantic_query",
                item["source_evidence_id"],
                normalize_query(item["question"]),
            ),
        )
        for item in items
    }
    generated_candidate_ids = set(candidate_id_by_item.values())
    missing_exclusions = sorted(set(exclusions) - generated_candidate_ids)
    if missing_exclusions:
        raise ValueError(
            "二次语义审核清单与本次生成结果不匹配；必须重新审核并更新清单："
            f"{missing_exclusions[:5]}"
        )
    items = [
        item
        for item in items
        if candidate_id_by_item[item["item_id"]] not in exclusions
    ]
    reviews = {
        item["item_id"]: _review_from_audit(audits[item["item_id"]])
        for item in items
    }
    query_ids_to_remove, candidate_ids_to_remove, evidence_ids_to_remove = _generated_ids(package)
    base_evidence = [
        dict(row)
        for row in package["evidence"]
        if row.get("evidence_id") not in evidence_ids_to_remove
    ]
    chunk_evidence = []
    for source in sources:
        chunk_evidence.append(
            {
                "evidence_id": source["evidence_id"],
                "chunk_id": source["evidence_id"],
                "source_page_id": source["source_page_id"],
                "doc_id": source["doc_id"],
                "section_id": source["section_id"],
                "page": source["page"],
                "quote": source["text"],
                "quote_normalized": source["text"],
                "offsets": {
                    "status": "exact",
                    "unit": "normalized_characters",
                    "start": source["start"],
                    "end": source["end"],
                },
                "normalization_version": "c7-text-normalization-v2",
                "evidence_type": "fixed_token_chunk",
                "selection_method": "deterministic_bge_tokenizer_budget",
                "token_count": source["token_count"],
                "token_budget": source["token_budget"],
                "source_case_ids": [],
                "claim_ids": [],
            }
        )
    evidence_rows = base_evidence + chunk_evidence
    queries = [
        dict(row)
        for row in package["queries"]
        if row.get("query_id") not in query_ids_to_remove
    ]
    qrels = [
        dict(row)
        for row in package["qrels"]
        if row.get("query_id") not in query_ids_to_remove
    ]
    candidates = [
        dict(row)
        for row in package["qa_candidates"]
        if row.get("candidate_id") not in candidate_ids_to_remove
    ]
    decisions = [
        dict(row)
        for row in package["review_decisions"]
        if row.get("candidate_id") not in candidate_ids_to_remove
    ]
    pairs = [
        dict(row)
        for row in package["finetune_pairs"]
        if row.get("query_id") not in query_ids_to_remove
    ]
    split = _base_split(package)
    ordinal = 0
    for item in items:
        source = source_by_id[item["source_evidence_id"]]
        review = reviews[item["item_id"]]
        split_name = source["split"]
        query_id = stable_id(
            "semantic_query",
            source["evidence_id"],
            normalize_query(item["question"]),
        )
        candidate_id = stable_id("semantic_candidate", query_id)
        pair_id = stable_id("pair", query_id)
        family_id = stable_id(
            "semantic_family",
            source["source_page_id"],
            source["evidence_id"],
        )
        ordinal += 1
        verification = {
            "method": REVIEW_TYPE,
            "result": "approved",
            "pdf_page": source["page"],
            "chunk_start": source["start"],
            "chunk_end": source["end"],
            "quote_exact": True,
            "support_quotes_exact": True,
            "human_verified": False,
            "normalization_version": "c7-text-normalization-v2",
            "semantic_review": review,
        }
        queries.append(
            {
                "query_id": query_id,
                "text": item["question"],
                "task_type": "rag_qa",
                "answerability": "answerable",
                "query_family_id": family_id,
                "usage": ["finetune"] if split_name == "train" else ["evaluation"],
                "excluded_from": (
                    ["dev", "test", "regression"]
                    if split_name == "train"
                    else ["train", "test", "regression"]
                    if split_name == "dev"
                    else ["train", "dev", "regression"]
                ),
                "reference_claims": [
                    {
                        "claim_id": stable_id("claim", query_id, index),
                        "claim": claim,
                        "evidence_ids": [source["evidence_id"]],
                    }
                    for index, claim in enumerate(item["claims"], 1)
                ],
                "evidence_section_ids": [source["section_id"]],
                "expected_pages": [source["page"]],
                "source_candidate_id": candidate_id,
                "review_status": "model_assisted_semantically_verified",
                "review_method": REVIEW_TYPE,
                "semantic_review": review,
            }
        )
        qrels.append(
            {
                "query_id": query_id,
                "evidence_id": source["evidence_id"],
                "relevance": 1,
                "essential": True,
                "review_status": "model_assisted_semantically_verified",
                "review_method": REVIEW_TYPE,
                "review_type": REVIEW_TYPE,
                "semantic_review_status": "passed",
            }
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "source": "pumpkin_book_fixed_token_chunk_generation",
                "source_model": MODEL,
                "source_path": "data/pumpkin_book.pdf",
                "source_ordinal": ordinal,
                "query": item["question"],
                "answer": item["answer"],
                "claims": item["claims"],
                "support_quotes": [source["text"]],
                "query_type": item["type"],
                "page_num": source["page"],
                "status": "approved",
                "review_status": "verified",
                "reviewer": None,
                "human_verified": False,
                "training_eligible": split_name == "train",
                "evidence_ids": [source["evidence_id"]],
                "pdf_page": source["page"],
                "review_method": REVIEW_TYPE,
                "verification": verification,
                "semantic_review": review,
                "review_type": REVIEW_TYPE,
                "supervision_type": "semantic_query_evidence",
            }
        )
        decisions.append(
            {
                "decision_id": stable_id("decision", candidate_id),
                "candidate_id": candidate_id,
                "decision": "approved",
                "review_status": "verified",
                "human_verified": False,
                "reason": "model_assisted_semantic_audit_passed",
                "evidence_ids": [source["evidence_id"]],
                "review_method": REVIEW_TYPE,
                "verification": verification,
                "review_type": REVIEW_TYPE,
                "semantic_review": review,
            }
        )
        pairs.append(
            {
                "pair_id": pair_id,
                "query_id": query_id,
                "candidate_id": candidate_id,
                "query": item["question"],
                "positive": source["text"],
                "positive_evidence_id": source["evidence_id"],
                "positive_token_count": source["token_count"],
                "token_budget": source["token_budget"],
                "page": source["page"],
                "section_id": source["section_id"],
                "query_family_id": family_id,
                "split": split_name,
                "review_status": "model_assisted_semantically_verified",
                "human_verified": False,
                "verification_method": REVIEW_TYPE,
                "review_type": REVIEW_TYPE,
                "supervision_type": "semantic_query_evidence",
                "semantic_review": review,
            }
        )
        split["assignments"][split_name].append(query_id)
        split["groups"][query_id] = {
            "section_id": source["section_id"],
            "query_family_id": family_id,
        }
    for split_name in ("train", "dev", "test"):
        split["assignments"][split_name].sort()
    return {
        "evidence": evidence_rows,
        "queries": queries,
        "qrels": qrels,
        "qa_candidates": candidates,
        "review_decisions": decisions,
        "finetune_pairs": pairs,
        "splits": split,
    }


def _construction_metadata(
    sources: Sequence[Mapping[str, Any]],
    generated: Mapping[str, Any],
    records: Mapping[str, Any],
) -> dict[str, Any]:
    pages_by_split = {
        split: sorted({int(row["page"]) for row in sources if row["split"] == split})
        for split in ("train", "dev", "test")
    }
    chunks_by_split = {
        split: sorted(row["evidence_id"] for row in sources if row["split"] == split)
        for split in ("train", "dev", "test")
    }
    pair_counts = {
        split: sum(row["split"] == split for row in records["finetune_pairs"])
        for split in ("train", "dev", "test")
    }
    source_page_by_id = {row["evidence_id"]: row["source_page_id"] for row in sources}
    retained_pages = {
        source_page_by_id[row["positive_evidence_id"]]
        for row in records["finetune_pairs"]
        if row["positive_evidence_id"] in source_page_by_id
    }
    secondary_exclusions = load_secondary_review_exclusions()
    return {
        "method": "pumpkin_book_fixed_token_chunk_semantic_query_evidence_v3",
        "generator_model": MODEL,
        "auditor_model": MODEL,
        "source_pages_reviewed": len({row["source_page_id"] for row in sources}),
        "source_pages_retained": len({row["source_page_id"] for row in sources}),
        "source_pages_excluded": 0,
        "source_pages_with_retained_items": len(retained_pages),
        "source_chunks_reviewed": len(sources),
        "source_chunk_count_by_split": {
            split: len(chunks_by_split[split]) for split in ("train", "dev", "test")
        },
        "primary_audit_input_count": len(generated["items"])
        + generated["curation"]["rejected_item_count"],
        "primary_audit_retained_item_count": len(generated["items"]),
        "retained_item_count": len(records["finetune_pairs"]),
        "retained_pair_count_by_split": pair_counts,
        "source_pages_by_split": pages_by_split,
        "candidate_source_evidence_ids_by_split": chunks_by_split,
        "exclusion_rule": "reject_schema-invalid_generation, false_primary_audit, and independent_secondary_review; never_exclude_source_page",
        "query_types": sorted(
            {
                row["query_type"]
                for row in records["qa_candidates"]
                if row.get("review_type") == REVIEW_TYPE
            }
        ),
        "support_binding": "generator_must_bind_existing_fixed_token_chunk_id",
        "chunk_token_budget": CHUNK_TOKEN_BUDGET,
        "model_max_seq_length": MODEL_MAX_SEQ_LENGTH,
        "review_type": REVIEW_TYPE,
        "human_verified": False,
        "primary_audit_rejected_item_count": generated["curation"]["rejected_item_count"],
        "generation_query_conflict_count": generated["curation"][
            "generation_query_conflict_count"
        ],
        "secondary_review_exclusion_count": len(secondary_exclusions),
        "secondary_review_method": "independent_model_review",
        "rejected_item_count": generated["curation"]["rejected_item_count"]
        + len(secondary_exclusions),
        "generation_schema_rejection_count": generated["curation"][
            "generation_schema_rejection_count"
        ],
        "cache_fingerprint_fields": [
            "schema_version",
            "item_schema_version",
            "audit_schema_version",
            "cache_integrity_version",
            "model",
            "chunk_token_budget",
            "generation_prompt_version",
            "audit_prompt_version",
            "prompt_fingerprint",
            "source_chunks_fingerprint",
            "input_fingerprint",
        ],
        "cache_metadata": generated["cache_metadata"],
    }


def apply_records(
    records: Mapping[str, Any],
    *,
    dataset_root: Path,
    sources: Sequence[Mapping[str, Any]],
    generated: Mapping[str, Any],
) -> None:
    write_jsonl(dataset_root / "evidence.jsonl", records["evidence"])
    write_jsonl(dataset_root / "queries.jsonl", records["queries"])
    write_jsonl(dataset_root / "qrels.jsonl", records["qrels"])
    write_jsonl(
        dataset_root / "annotations" / "qa_candidates.jsonl",
        records["qa_candidates"],
    )
    write_jsonl(
        dataset_root / "annotations" / "review_decisions.jsonl",
        records["review_decisions"],
    )
    write_jsonl(dataset_root / "finetune_pairs.jsonl", records["finetune_pairs"])
    write_json(dataset_root / "splits.json", records["splits"])
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["data_construction"] = _construction_metadata(sources, generated, records)
    write_json(manifest_path, manifest)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=COURSE_ROOT / ".cache" / "c2_semantic_training_v5",
        help="cache directory bound to the current fixed chunks and prompt schema",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="审核通过后重建并写入 canonical train/dev/test 三个分区及其索引",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="仅复用与当前来源 chunks、prompt/schema/model 指纹完全匹配的完整缓存",
    )
    args = parser.parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    cache_dir = args.cache_dir.resolve()
    cache_path = cache_dir / "generated.json"
    partial_path = cache_dir / "generated.partial.json"
    package = load_dataset(dataset_root)
    sources, _ = load_sources(package)
    existing_query_norms = _base_query_norms(package)
    metadata = cache_metadata(sources, existing_query_norms=existing_query_norms)
    if args.reuse_cache:
        if not cache_path.is_file():
            raise FileNotFoundError(f"生成缓存不存在：{cache_path}")
        value = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("完整生成缓存必须是 object")
        _validate_cache_metadata(value, metadata, label="完整生成缓存")
        generated = value
        _validate_generated_by_source(
            generated,
            sources,
            reserved_query_norms=existing_query_norms,
            require_all_sources=True,
        )
        _validate_cached_items(
            generated["items"],
            sources,
            reserved_query_norms=existing_query_norms,
        )
        _validate_cached_audits(
            generated["items"],
            generated["audits"],
            {row["evidence_id"]: row for row in sources},
            require_complete=True,
        )
    else:
        generated = generate_dataset(
            sources,
            cache_path=cache_path,
            partial_cache_path=partial_path,
            existing_query_norms=existing_query_norms,
        )
    records = build_records(package, sources, generated)
    print(
        json.dumps(
            {
                "source_chunks": len(sources),
                "generated_items": len(generated["items"]),
                "records": {
                    key: len(value)
                    for key, value in records.items()
                    if isinstance(value, list)
                },
                "retained_items_by_split": generated["curation"]["retained_items_by_split"],
                "rejected_item_count": generated["curation"]["rejected_item_count"],
                "generation_schema_rejection_count": generated["curation"][
                    "generation_schema_rejection_count"
                ],
                "chunk_token_budget": CHUNK_TOKEN_BUDGET,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.apply:
        apply_records(
            records,
            dataset_root=dataset_root,
            sources=sources,
            generated=generated,
        )
        print("canonical train/dev/test partitions rebuilt from fixed-token chunks")
    else:
        print("dry run only; pass --apply to rebuild canonical train/dev/test partitions")


if __name__ == "__main__":
    main()
