#!/usr/bin/env python3
"""只读检查端到端 capstone 的 Notebook 结构和已保存验收输出。"""

from __future__ import annotations

import json
from pathlib import Path


COURSE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = COURSE_ROOT / "7. 评估" / "端到端验收.ipynb"
AUDIT_MIME = "application/vnd.llm-universe.tutorial-audit+json"


def _stream_text(output: dict) -> str:
    value = output.get("text", "")
    return "".join(value) if isinstance(value, list) else str(value)


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise AssertionError("capstone Notebook cells 结构无效")
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    if len(code_cells) < 6:
        raise AssertionError("capstone 必须包含检索、预算、生成、语义检查和事后检查代码单元")
    if any(not isinstance(cell.get("execution_count"), int) for cell in code_cells):
        raise AssertionError("capstone 尚未保存所有代码单元的执行记录")

    cell_sources = ["".join(cell.get("source", [])) for cell in code_cells]
    sources = "\n".join(cell_sources)
    required = (
        "read_jsonl",
        "build_bm25_chunk_search",
        "CONTEXT_BUDGET",
        "apply_context_budget",
        "len(context_text[case_id]) != used",
        "llm_call",
        "json.loads",
        "insufficient",
        "evidence_by_id",
        "SEMANTIC_JUDGE_PROMPT",
        "semantic_results",
        "hydrated_citations",
        "supported",
        "contradicted",
        "not_found",
        "emit_tutorial_audit",
    )
    missing = [token for token in required if token not in sources]
    if missing:
        raise AssertionError(f"capstone 缺少关键闭环代码：{missing}")
    forbidden_calls = ("load_dataset(", "load_query_records(", "load_evidence_records(", "load_qrel_records(")
    if any(token in sources for token in forbidden_calls):
        raise AssertionError("capstone 不应通过完整数据集读取器加载 C2 训练候选")

    def _cell_index(fragment: str) -> int:
        matches = [index for index, source in enumerate(cell_sources) if fragment in source]
        if len(matches) != 1:
            raise AssertionError(f"capstone 必须有且只有一个 {fragment!r} 代码单元")
        return matches[0]

    retrieve_index = _cell_index("retrieved =")
    budget_index = _cell_index("context_rows =")
    generation_index = _cell_index("raw_model = {}")
    semantic_index = _cell_index("semantic_results = {}")
    posthoc_index = _cell_index("qrels_by_case =")
    if not retrieve_index < budget_index < generation_index < semantic_index < posthoc_index:
        raise AssertionError("capstone 代码顺序必须是检索 → 上下文预算 → 生成 → 语义检查 → 事后验收")
    pre_generation = "\n".join(cell_sources[:generation_index])
    leaked_fields = (
        "answerability",
        "reference_claims",
        "reference_answer",
        "expected_pages",
        "qrels.jsonl",
        '"relevance"',
        '"essential"',
        '"usage"',
    )
    leaked = [field for field in leaked_fields if field in pre_generation]
    if leaked:
        raise AssertionError(f"生成前源码出现评测标签或 qrels：{leaked}")
    if "query_meta" in "\n".join(cell_sources[:generation_index]):
        raise AssertionError("query metadata 必须在模型生成之后读取")
    if "qrels.jsonl" not in cell_sources[posthoc_index] or "queries.jsonl" not in cell_sources[posthoc_index]:
        raise AssertionError("query metadata 与 qrels 必须在 post-hoc 代码单元读取")
    if cell_sources[generation_index].count("llm_call(") != 1:
        raise AssertionError("生成代码单元必须对每道题执行一次统一 llm_call")
    if cell_sources[semantic_index].count("llm_call(") != 1:
        raise AssertionError("语义检查代码单元必须对每道题执行一次独立 llm_call")
    semantic_pre_hoc = "\n".join(cell_sources[generation_index:posthoc_index])
    semantic_leaked = [field for field in leaked_fields if field in semantic_pre_hoc]
    if semantic_leaked:
        raise AssertionError(f"模型生成/语义检查阶段出现事后评测标签：{semantic_leaked}")
    if "INSUFFICIENT_ANSWER" not in cell_sources[generation_index]:
        raise AssertionError("生成代码必须声明严格的 insufficient 固定拒答句")

    outputs = [output for cell in code_cells for output in cell.get("outputs", [])]
    errors = [output for output in outputs if output.get("output_type") == "error"]
    if errors:
        raise AssertionError("capstone Notebook 保存了错误输出")
    audits = []
    stream = []
    for output in outputs:
        if output.get("output_type") == "stream":
            stream.append(_stream_text(output))
        data = output.get("data", {})
        if isinstance(data, dict) and isinstance(data.get(AUDIT_MIME), dict):
            audits.append(data[AUDIT_MIME])
    if len(audits) != 1:
        raise AssertionError(f"capstone 应保存一条结构化 audit，实际为 {len(audits)} 条")
    audit = audits[0]
    if audit.get("trace_kind") != "c7_endpoint_capstone" or audit.get("model") != "glm-4-flash":
        raise AssertionError("capstone audit 没有声明真实 glm-4-flash 闭环")
    if audit.get("retriever") != "bm25" or audit.get("evidence_scope") != "canonical evidence_type=quote only":
        raise AssertionError("capstone audit 的检索或证据范围不符合协议")
    if audit.get("all_passed") is not True:
        raise AssertionError("capstone audit 未通过")
    semantic_judge = audit.get("semantic_judge")
    if (
        not isinstance(semantic_judge, dict)
        or semantic_judge.get("model") != "glm-4-flash"
        or semantic_judge.get("calls") != 4
        or not isinstance(semantic_judge.get("limitation"), str)
        or not semantic_judge["limitation"].strip()
    ):
        raise AssertionError("capstone audit 必须保存独立语义评审的模型、调用次数和局限")

    cases = audit.get("cases", [])
    if len(cases) != 4 or any(
        case.get("model_called") is not True
        or case.get("semantic_judge_model_called") is not True
        for case in cases
    ):
        raise AssertionError("capstone 必须对固定 4 道题全部真实调用模型")
    answerable = [case for case in cases if case.get("answerability") == "answerable"]
    refusals = [case for case in cases if case.get("answerability") != "answerable"]
    if len(answerable) < 3 or not refusals:
        raise AssertionError("capstone 应包含至少 3 道 answerable 和 1 道拒答题")
    if sum(len(case.get("essential_ids", [])) >= 2 for case in answerable) < 2:
        raise AssertionError("capstone 应包含至少 2 道多证据 answerable 题")
    for case in answerable:
        essential = set(case.get("essential_ids", []))
        if not essential <= set(case.get("retrieved_ids", [])):
            raise AssertionError(f"essential evidence 未进入检索结果：{case.get('query_id')}")
        if not essential <= set(case.get("context_evidence_ids", [])):
            raise AssertionError(f"essential evidence 未进入上下文：{case.get('query_id')}")
        if not essential <= set(case.get("citation_ids", [])):
            raise AssertionError(f"模型引用未覆盖 essential evidence：{case.get('query_id')}")
        claims = case.get("semantic_claims")
        if (
            case.get("response_status") != "answerable"
            or case.get("verdict") != "pass_answerable"
            or case.get("semantic_verdict") != "pass"
            or not isinstance(claims, list)
            or not claims
            or any(claim.get("relation") != "supported" for claim in claims)
            or any(
                not set(claim.get("evidence_ids", [])).issubset(set(case.get("citation_ids", [])))
                for claim in claims
            )
        ):
            raise AssertionError(f"answerable 题验收失败：{case.get('query_id')}")
    for case in refusals:
        if (
            case.get("response_status") != "insufficient"
            or case.get("answer") != "资料不足，无法根据提供的上下文回答。"
            or case.get("citation_ids")
            or case.get("semantic_verdict") != "pass"
            or case.get("semantic_claims") != []
            or case.get("verdict") != "pass_insufficient"
        ):
            raise AssertionError(f"insufficient 题验收失败：{case.get('query_id')}")
    model_call_summaries = sum(text.count("glm-4-flash 返回") for text in stream)
    semantic_call_summaries = sum(text.count("semantic judge glm-4-flash") for text in stream)
    if model_call_summaries != 4 or semantic_call_summaries != 4:
        raise AssertionError(
            "Notebook 必须保存 4 次生成调用和 4 次独立语义评审调用摘要，"
            f"实际为 {model_call_summaries}+{semantic_call_summaries} 次"
        )
    if not any("quote=" in text for text in stream):
        raise AssertionError("Notebook 没有保存逐字 quote hydrate 输出")

    print(f"capstone check passed: {len(cases)} cases, {len(answerable)} answerable, {len(refusals)} insufficient")


if __name__ == "__main__":
    main()
