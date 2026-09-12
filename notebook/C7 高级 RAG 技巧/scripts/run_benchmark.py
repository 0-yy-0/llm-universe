#!/usr/bin/env python3
"""Run the C7 same-question retrieval benchmark.

The benchmark deliberately has a small, explicit question set.  Retrieval
receives only query text and canonical evidence text.  qrels are read after
all methods have produced their ranked lists, so reference evidence cannot
silently become a search feature.

The command is usable from either the repository root::

    python "notebook/C7 高级 RAG 技巧/scripts/run_benchmark.py"

or the C7 tutorial root::

    python scripts/run_benchmark.py

Missing or malformed canonical data is an error.  There is intentionally no
fallback to another case file, a PDF, an existing vector store,
or an external model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
COURSE_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = COURSE_ROOT.parents[1]
if str(COURSE_ROOT) not in sys.path:
    sys.path.insert(0, str(COURSE_ROOT))

from common.dataset import DatasetContractError, chapter_title, load_dataset  # noqa: E402
from common.eval_utils import build_bm25_search, normalize_text, recall_at_k  # noqa: E402


# Every selected query is explicitly curated for both tutorial demonstration
# and regression use in the canonical package.  Keeping this tuple in source
# makes the comparison reproducible and prevents method-specific sampling.
DEFAULT_QUERY_IDS: tuple[str, ...] = (
    "model_selection_with_intro_scope",
    "cross_validation_reliability",
    "ensemble_learning_definition",
    "lda_recursive_derivation",
    "newton_methods_comparison",
    "model_evaluation_followup",
)

DEFAULT_TOP_K = 4
DEFAULT_CHAR_BUDGET = 1200
BASELINE_METHOD = "bm25"
METHODS: tuple[str, ...] = (BASELINE_METHOD, "bm25_cch")

class BenchmarkContractError(ValueError):
    """Raised when benchmark configuration or output violates its contract."""


def find_course_root(start: str | Path | None = None) -> Path:
    """Find the C7 root from a repository-root or C7-root working directory.

    The script's own location is also considered, which keeps imports stable
    when the command is launched through an absolute path.
    """

    origin = Path(start or Path.cwd()).resolve()
    candidates: list[Path] = []
    for folder in (origin, *origin.parents, COURSE_ROOT):
        candidates.extend((folder, folder / "notebook" / "C7 高级 RAG 技巧"))
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (
            (candidate / "data" / "dataset" / "manifest.json").is_file()
            and (candidate / "common" / "dataset.py").is_file()
        ):
            return candidate
    raise FileNotFoundError(
        "找不到 C7 canonical dataset；请从仓库根或 C7 教程根运行。"
    )


def _as_nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkContractError(f"{label} 必须是非空文字")
    return normalize_text(value)


def validate_config(
    query_ids: Sequence[str] = DEFAULT_QUERY_IDS,
    top_k: int = DEFAULT_TOP_K,
    char_budget: int = DEFAULT_CHAR_BUDGET,
) -> tuple[tuple[str, ...], int, int]:
    """Validate the shared benchmark configuration before any retrieval."""

    ids = tuple(str(item) for item in query_ids)
    if not ids or any(not item.strip() for item in ids):
        raise BenchmarkContractError("query_ids 必须是非空字符串序列")
    if len(set(ids)) != len(ids):
        raise BenchmarkContractError("query_ids 不得重复")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise BenchmarkContractError("top_k 必须为正整数")
    if not isinstance(char_budget, int) or isinstance(char_budget, bool) or char_budget <= 0:
        raise BenchmarkContractError("char_budget 必须为正整数")
    if len(METHODS) < 2 or len(METHODS) > 3:
        raise BenchmarkContractError("横评方法数必须为 2～3")
    if METHODS[0] != BASELINE_METHOD:
        raise BenchmarkContractError("第一个方法必须是 BM25 基线")
    return ids, top_k, char_budget


def _selected_queries(
    package: Mapping[str, Any], query_ids: Sequence[str]
) -> list[dict[str, Any]]:
    rows = package.get("queries")
    if not isinstance(rows, list):
        raise BenchmarkContractError("canonical queries 不是记录列表")
    by_id = {row.get("query_id"): row for row in rows if isinstance(row, dict)}
    selected: list[dict[str, Any]] = []
    for query_id in query_ids:
        row = by_id.get(query_id)
        if row is None:
            raise BenchmarkContractError(f"canonical queries 缺少问题：{query_id}")
        if row.get("task_type") != "rag_qa":
            raise BenchmarkContractError(f"benchmark 只能使用 rag_qa 问题：{query_id}")
        if row.get("answerability") != "answerable":
            raise BenchmarkContractError(f"benchmark 只能使用 answerable 问题：{query_id}")
        usage = row.get("usage")
        if not isinstance(usage, list) or not {"demo", "regression"}.issubset(usage):
            raise BenchmarkContractError(
                f"问题必须同时标记 demo/regression：{query_id}"
            )
        # Retrieval gets this narrow projection only.  In particular it does
        # not receive reference_claims, expected_pages, or reference_answer.
        selected.append(
            {
                "query_id": query_id,
                "text": _as_nonempty_text(row.get("text"), f"问题 {query_id} text"),
            }
        )
    return selected


def _retrieval_evidence(package: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = package.get("evidence")
    if not isinstance(rows, list) or not rows:
        raise BenchmarkContractError("canonical evidence 不能为空")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise BenchmarkContractError("canonical evidence 每项必须是对象")
        evidence_id = row.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise BenchmarkContractError("evidence_id 必须是非空文字")
        if evidence_id in seen:
            raise BenchmarkContractError(f"evidence_id 重复：{evidence_id}")
        seen.add(evidence_id)
        page = row.get("page")
        if not isinstance(page, int) or page <= 0:
            raise BenchmarkContractError(f"evidence {evidence_id} page 无效")
        quote = _as_nonempty_text(row.get("quote"), f"evidence {evidence_id} quote")
        # No qrel or query annotations are carried into this projection.
        result.append({"evidence_id": evidence_id, "page": page, "quote": quote})
    return result


def compose_index_text(
    evidence: Mapping[str, Any], method: str, char_budget: int
) -> str:
    """Build a bounded search field while retaining original evidence text."""

    if method == BASELINE_METHOD:
        text = str(evidence["quote"])
    elif method == "bm25_cch":
        text = f"{chapter_title(int(evidence['page']))}\n{evidence['quote']}"
    else:
        raise KeyError(f"未知横评方法：{method}")
    text = normalize_text(text)
    if not text:
        raise BenchmarkContractError(
            f"方法 {method} 生成了空检索字段：{evidence.get('evidence_id')}"
        )
    return text[:char_budget]


def _build_method_searcher(
    evidence_rows: Sequence[Mapping[str, Any]], method: str, char_budget: int
) -> Callable[[str, int], list[dict[str, Any]]]:
    """Wrap the shared BM25 helper and map synthetic rows to stable IDs."""

    index_rows = [
        {"page": index, "text": compose_index_text(row, method, char_budget)}
        for index, row in enumerate(evidence_rows, start=1)
    ]
    search = build_bm25_search(index_rows)

    def run(query: str, top_k: int) -> list[dict[str, Any]]:
        hits = search(query, top_k=top_k)
        result: list[dict[str, Any]] = []
        for hit in hits:
            # build_bm25_search returns the synthetic row's one-based index.
            position = int(hit.page) - 1
            if not 0 <= position < len(evidence_rows):
                raise BenchmarkContractError("BM25 结果无法映回 canonical evidence")
            source = evidence_rows[position]
            result.append(
                {
                    "evidence_id": str(source["evidence_id"]),
                    "page": int(source["page"]),
                    "score": round(float(hit.score), 8),
                    "index_characters": len(
                        compose_index_text(source, method, char_budget)
                    ),
                }
            )
        if len(result) > top_k:
            raise BenchmarkContractError(f"方法 {method} 返回超过 top_k")
        return result

    return run


def retrieve_all(
    query_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int = DEFAULT_TOP_K,
    char_budget: int = DEFAULT_CHAR_BUDGET,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Run every method over exactly the same query and evidence batch."""

    validate_config([str(row.get("query_id", "")) for row in query_rows], top_k, char_budget)
    if not evidence_rows:
        raise BenchmarkContractError("evidence_rows 不能为空")
    searchers = {
        method: _build_method_searcher(evidence_rows, method, char_budget)
        for method in METHODS
    }
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for query in query_rows:
        query_id = _as_nonempty_text(query.get("query_id"), "query_id")
        text = _as_nonempty_text(query.get("text"), f"问题 {query_id} text")
        result[query_id] = {
            method: searchers[method](text, top_k) for method in METHODS
        }
    return result


def _relevance_by_query(
    package: Mapping[str, Any], query_ids: Sequence[str]
) -> dict[str, set[str]]:
    """Read positive qrels only after retrieval has completed."""

    rows = package.get("qrels")
    if not isinstance(rows, list):
        raise BenchmarkContractError("canonical qrels 不是记录列表")
    wanted = set(query_ids)
    evidence_ids = {
        row.get("evidence_id")
        for row in package.get("evidence", [])
        if isinstance(row, dict)
    }
    relevant: dict[str, set[str]] = {query_id: set() for query_id in query_ids}
    for row in rows:
        if not isinstance(row, dict) or row.get("query_id") not in wanted:
            continue
        if row.get("relevance") == 1:
            evidence_id = row.get("evidence_id")
            if evidence_id not in evidence_ids:
                raise BenchmarkContractError(
                    f"qrel 引用了不存在的 evidence：{evidence_id}"
                )
            relevant[str(row["query_id"])].add(str(evidence_id))
    missing = [query_id for query_id, ids in relevant.items() if not ids]
    if missing:
        raise BenchmarkContractError(f"选中问题没有正 qrels：{missing}")
    return relevant


def _evidence_pages(
    evidence_rows: Sequence[Mapping[str, Any]], evidence_ids: Iterable[str]
) -> set[int]:
    by_id = {str(row["evidence_id"]): int(row["page"]) for row in evidence_rows}
    pages: set[int] = set()
    for evidence_id in evidence_ids:
        if evidence_id not in by_id:
            raise BenchmarkContractError(f"结果无法映回 evidence 页：{evidence_id}")
        pages.add(by_id[evidence_id])
    return pages


def _first_rank(hits: Sequence[Mapping[str, Any]], relevant: set[str]) -> int | None:
    return next(
        (
            rank
            for rank, hit in enumerate(hits, start=1)
            if str(hit.get("evidence_id")) in relevant
        ),
        None,
    )


def _mrr(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def _outcome(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> str:
    """Compare recall first and rank second, preserving visible trade-offs."""

    before_recall = float(before["recall_at_k"])
    after_recall = float(after["recall_at_k"])
    before_rank = (
        int(before.get("top_k", len(before.get("retrieved_evidence_ids", [])))) + 1
        if before["first_relevant_rank"] is None
        else int(before["first_relevant_rank"])
    )
    after_rank = (
        int(after.get("top_k", len(after.get("retrieved_evidence_ids", [])))) + 1
        if after["first_relevant_rank"] is None
        else int(after["first_relevant_rank"])
    )
    improved = after_recall > before_recall or after_rank < before_rank
    degraded = after_recall < before_recall or after_rank > before_rank
    if improved and degraded:
        return "tradeoff"
    if improved:
        return "improved"
    if degraded:
        return "degraded"
    return "unchanged"


def _metric_record(
    hits: Sequence[Mapping[str, Any]],
    relevant: set[str],
    evidence_rows: Sequence[Mapping[str, Any]],
    top_k: int,
    char_budget: int,
) -> dict[str, Any]:
    ids = [str(hit["evidence_id"]) for hit in hits]
    if len(ids) != len(set(ids)):
        raise BenchmarkContractError("单个方法结果包含重复 evidence_id")
    flags = [evidence_id in relevant for evidence_id in ids]
    recall = recall_at_k(flags, total_relevant=len(relevant), k=top_k)
    rank = _first_rank(hits, relevant)
    found = sorted(set(ids) & relevant)
    relevant_pages = _evidence_pages(evidence_rows, relevant)
    found_pages = _evidence_pages(evidence_rows, found)
    # Context always comes from canonical quote text, never from CCH or any
    # other index-only field.  A single shared budget is applied to every
    # method before the character count is reported.
    text_by_id = {str(row["evidence_id"]): str(row["quote"]) for row in evidence_rows}
    context = "\n\n".join(text_by_id[evidence_id] for evidence_id in ids)
    context = context[:char_budget]
    return {
        "retrieved_evidence_ids": ids,
        "retrieved_pages": [int(hit["page"]) for hit in hits],
        "relevant_evidence_ids_found": found,
        "top_k": top_k,
        "first_relevant_rank": rank,
        "rank": rank,
        "relevant_rank": rank,  # concise alias for readers of the table
        "coverage": len(found) / len(relevant),
        "recall_at_k": recall,
        "mrr": _mrr(rank),
        "relevant_pages_found": sorted(found_pages),
        "relevant_page_coverage": (
            len(found_pages) / len(relevant_pages) if relevant_pages else 0.0
        ),
        "context_characters": len(context),
        "char_budget": char_budget,
        "hits": [dict(hit) for hit in hits],
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _summary(
    method: str,
    question_results: Mapping[str, Mapping[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    rows = list(question_results.values())
    outcomes = {name: 0 for name in ("improved", "unchanged", "degraded", "tradeoff")}
    if method == BASELINE_METHOD:
        outcomes = {"baseline": len(rows), **outcomes}
    else:
        for row in rows:
            outcome = row["outcome_vs_bm25"]
            if outcome not in outcomes:
                raise BenchmarkContractError(f"未知比较结果：{outcome}")
            outcomes[outcome] += 1
    rank_values = [
        float(top_k + 1 if row["first_relevant_rank"] is None else row["first_relevant_rank"])
        for row in rows
    ]
    return {
        "question_count": len(rows),
        "mean_coverage": _mean(float(row["coverage"]) for row in rows),
        "mean_recall_at_k": _mean(float(row["recall_at_k"]) for row in rows),
        "mrr": _mean(float(row["mrr"]) for row in rows),
        "mean_first_relevant_rank_with_miss_as_top_k_plus_one": _mean(rank_values),
        "mean_context_characters": _mean(
            float(row["context_characters"]) for row in rows
        ),
        "outcomes_vs_bm25": outcomes,
    }


def validate_report(report: Mapping[str, Any]) -> None:
    """Validate metric, method, query, and budget consistency."""

    if report.get("schema_version") != 1:
        raise BenchmarkContractError("benchmark schema_version 必须为 1")
    config = report.get("config")
    if not isinstance(config, dict):
        raise BenchmarkContractError("benchmark 缺少 config")
    query_ids, top_k, char_budget = validate_config(
        config.get("query_ids", ()), config.get("top_k"), config.get("char_budget")
    )
    if config.get("methods") != list(METHODS):
        raise BenchmarkContractError("方法列表未保持统一")
    questions = report.get("questions")
    if not isinstance(questions, dict) or tuple(questions) != query_ids:
        raise BenchmarkContractError("问题集合未在方法间保持一致")
    summaries = report.get("summary")
    if not isinstance(summaries, dict) or set(summaries) != set(METHODS):
        raise BenchmarkContractError("汇总缺少方法")
    for query_id in query_ids:
        item = questions[query_id]
        if not isinstance(item, dict):
            raise BenchmarkContractError(f"问题结果不是对象：{query_id}")
        methods = item.get("methods")
        if not isinstance(methods, dict) or list(methods) != list(METHODS):
            raise BenchmarkContractError(f"问题方法集合不一致：{query_id}")
        baseline = methods[BASELINE_METHOD]
        for method in METHODS:
            metrics = methods[method]
            if not isinstance(metrics, dict):
                raise BenchmarkContractError(f"方法指标不是对象：{query_id}/{method}")
            ids = metrics.get("retrieved_evidence_ids")
            if not isinstance(ids, list) or len(ids) != top_k:
                raise BenchmarkContractError(
                    f"{query_id}/{method} 未返回统一 top_k={top_k}"
                )
            if metrics.get("top_k") != top_k:
                raise BenchmarkContractError(f"{query_id}/{method} top_k 记录不一致")
            if metrics.get("char_budget") != char_budget:
                raise BenchmarkContractError(f"{query_id}/{method} 字符预算不一致")
            if not isinstance(metrics.get("context_characters"), int) or not 0 <= metrics["context_characters"] <= char_budget:
                raise BenchmarkContractError(f"{query_id}/{method} 上下文超过字符预算")
            hits = metrics.get("hits")
            if not isinstance(hits, list) or len(hits) != top_k:
                raise BenchmarkContractError(f"{query_id}/{method} hits 与 top_k 不一致")
            if any(
                not isinstance(hit, dict)
                or not isinstance(hit.get("index_characters"), int)
                or not 0 < hit["index_characters"] <= char_budget
                for hit in hits
            ):
                raise BenchmarkContractError(f"{query_id}/{method} 检索字段超过字符预算")
            for name in ("coverage", "recall_at_k", "mrr"):
                value = metrics.get(name)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                    raise BenchmarkContractError(f"{query_id}/{method} {name} 无效")
            rank = metrics.get("first_relevant_rank")
            if rank is not None and (not isinstance(rank, int) or not 1 <= rank <= top_k):
                raise BenchmarkContractError(f"{query_id}/{method} rank 无效")
            if metrics.get("rank") != rank or metrics.get("relevant_rank") != rank:
                raise BenchmarkContractError(f"{query_id}/{method} rank 别名不一致")
            if method == BASELINE_METHOD:
                if metrics.get("outcome_vs_bm25") != "baseline":
                    raise BenchmarkContractError("BM25 必须标记 baseline")
            else:
                expected = _outcome(baseline, metrics)
                if metrics.get("outcome_vs_bm25") != expected:
                    raise BenchmarkContractError(
                        f"{query_id}/{method} outcome 与指标不一致"
                    )


def run_benchmark(
    *,
    dataset_root: str | Path | None = None,
    query_ids: Sequence[str] = DEFAULT_QUERY_IDS,
    top_k: int = DEFAULT_TOP_K,
    char_budget: int = DEFAULT_CHAR_BUDGET,
) -> dict[str, Any]:
    """Run and return the complete same-question benchmark report."""

    ids, top_k, char_budget = validate_config(query_ids, top_k, char_budget)
    if dataset_root is None:
        root = find_course_root()
        dataset_path = root / "data" / "dataset"
    else:
        dataset_path = Path(dataset_root).resolve()
    try:
        package = load_dataset(dataset_path)
    except (FileNotFoundError, DatasetContractError) as exc:
        raise RuntimeError(f"canonical dataset 无法读取：{dataset_path}") from exc

    query_rows = _selected_queries(package, ids)
    evidence_rows = _retrieval_evidence(package)

    # Crucial leakage boundary: qrels are not passed to retrieve_all.  Only
    # after every method has ranked the same query/evidence projection do we
    # load positive relevance labels to compute metrics.
    retrieval = retrieve_all(
        query_rows, evidence_rows, top_k=top_k, char_budget=char_budget
    )
    relevant = _relevance_by_query(package, ids)
    query_text = {str(row["query_id"]): str(row["text"]) for row in query_rows}
    questions: dict[str, Any] = {}
    for query_id in ids:
        baseline_metrics: dict[str, Any] | None = None
        methods: dict[str, Any] = {}
        for method in METHODS:
            metrics = _metric_record(
                retrieval[query_id][method],
                relevant[query_id],
                evidence_rows,
                top_k,
                char_budget,
            )
            if method == BASELINE_METHOD:
                metrics["outcome_vs_bm25"] = "baseline"
                baseline_metrics = metrics
            else:
                if baseline_metrics is None:
                    raise BenchmarkContractError("必须先计算 BM25 基线")
                metrics["outcome_vs_bm25"] = _outcome(baseline_metrics, metrics)
            methods[method] = metrics
        questions[query_id] = {
            "query": query_text[query_id],
            "relevant_evidence_count": len(relevant[query_id]),
            "methods": methods,
        }

    summaries: dict[str, Any] = {}
    for method in METHODS:
        summaries[method] = _summary(
            method,
            {query_id: questions[query_id]["methods"][method] for query_id in ids},
            top_k,
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "c7_same_question_retrieval",
        "config": {
            "query_ids": list(ids),
            "query_scope": "explicit canonical demo/regression set",
            "top_k": top_k,
            "char_budget": char_budget,
            "methods": list(METHODS),
            "baseline": BASELINE_METHOD,
            "qrels": "canonical qrels relevance=1, loaded after retrieval",
            "answer_context_source": "canonical evidence.quote only",
            "evidence_pool_scope": "curated exact-PDF evidence pool, not the full PDF chunk corpus",
        },
        "questions": questions,
        "summary": summaries,
        "validation": {
            "same_query_batch_for_all_methods": True,
            "same_top_k_for_all_methods": True,
            "same_char_budget_for_all_methods": True,
            "qrels_used_for_metrics": True,
            "target_pages_or_reference_answers_used_for_retrieval": False,
            "evidence_pool_is_full_pdf": False,
            "fallback_used": False,
        },
    }
    validate_report(report)
    return report


def render_text(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable report for terminal/Notebook output."""

    config = report["config"]
    lines = [
        "C7 同题统一横评",
        (
            f"题数={len(config['query_ids'])}；方法={','.join(config['methods'])}；"
            f"top_k={config['top_k']}；char_budget={config['char_budget']}"
        ),
        "",
        "逐题结果（rank / coverage / Recall@k / MRR / 相对 BM25）",
    ]
    for query_id in config["query_ids"]:
        item = report["questions"][query_id]
        lines.append(f"- {query_id}: {item['query']}")
        for method in config["methods"]:
            metrics = item["methods"][method]
            rank = metrics["first_relevant_rank"]
            lines.append(
                f"  {method}: rank={rank if rank is not None else 'miss'}; "
                f"coverage={metrics['coverage']:.3f}; "
                f"Recall@{config['top_k']}={metrics['recall_at_k']:.3f}; "
                f"MRR={metrics['mrr']:.3f}; "
                f"outcome={metrics['outcome_vs_bm25']}"
            )
    lines.extend(("", "汇总"))
    for method in config["methods"]:
        summary = report["summary"][method]
        lines.append(
            f"- {method}: Recall@{config['top_k']}={summary['mean_recall_at_k']:.3f}; "
            f"MRR={summary['mrr']:.3f}; outcomes={summary['outcomes_vs_bm25']}"
        )
    lines.extend(
        (
            "",
            "验证：同题/同 top-k/同字符预算；qrels 只用于检索完成后的指标；"
            "检索未读取目标页或参考答案；证据池是标注过的小型原文池而非全 PDF；无 fallback。",
        )
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="canonical dataset 目录；默认从当前目录发现 C7/data/dataset",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="只输出完整 JSON（默认同时输出人类可读摘要和 JSON）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_benchmark(dataset_root=args.dataset)
    except (BenchmarkContractError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 1
    if not args.json:
        print(render_text(report))
        print()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
