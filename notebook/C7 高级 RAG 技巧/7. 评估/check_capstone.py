#!/usr/bin/env python3
"""重放本地检索、预算和原始模型响应，检查 capstone 审计完整性。

不调用模型。默认允许如实记录的质量负结果；--require-all-passed 另行要求
所有回答与信息缺口补查通过，避免把 audit 完整误称为回答全部正确。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

COURSE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = COURSE_ROOT / "7. 评估" / "端到端验收.ipynb"
AUDIT_MIME = "application/vnd.llm-universe.tutorial-audit+json"
RAW_MIME = "application/vnd.llm-universe.capstone-raw+json"
INSUFFICIENT_ANSWER = "资料不足，无法根据提供的上下文回答。"
CASE_IDS = (
    "model_selection_with_intro_scope",
    "cross_validation_reliability",
    "model_evaluation_and_macro_micro",
    "book_evidence_boundary",
)
GAP_CASE_ID = "model_evaluation_and_macro_micro"
CONTEXT_BUDGET = 1800
TOP_K = 8


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parse_strict_json(raw: str) -> dict:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("原始模型响应必须是非空字符串")
    payload = raw.strip()
    if payload.startswith("~~~"):
        raise ValueError("不接受非 JSON 传输外壳")
    if payload.startswith(chr(96) * 3):
        match = re.fullmatch(r"\x60{3}json\s*\n(.*?)\n\x60{3}", payload, re.DOTALL)
        if match is None:
            raise ValueError("模型 JSON 围栏不完整或带有额外文字")
        payload = match.group(1)

    def reject_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"模型 JSON 重复字段：{key}")
            value[key] = item
        return value

    def reject_constant(value):
        raise ValueError(f"模型 JSON 含非有限数字：{value}")

    value = json.loads(
        payload, object_pairs_hook=reject_duplicates, parse_constant=reject_constant
    )
    if not isinstance(value, dict):
        raise ValueError("模型 JSON 必须是对象")
    return value


def string_list(value, field: str, *, minimum: int = 0, maximum: int = 100) -> list[str]:
    if (
        not isinstance(value, list)
        or not minimum <= len(value) <= maximum
        or any(not isinstance(item, str) or not item.strip() for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{field} 必须是非空字符串组成的唯一列表，长度 {minimum}..{maximum}")
    return value


def validate_generation(raw: str, allowed_ids: set[str]) -> dict:
    value = parse_strict_json(raw)
    if set(value) != {"status", "answer", "citations"}:
        raise ValueError("原始生成 JSON 字段不严格")
    if value["status"] not in ("answerable", "insufficient"):
        raise ValueError("原始生成 status 非法")
    if not isinstance(value["answer"], str) or not value["answer"].strip():
        raise ValueError("原始生成 answer 必须非空")
    citations = value["citations"]
    if not isinstance(citations, list) or any(
        not isinstance(item, dict) or set(item) != {"evidence_id"} for item in citations
    ):
        raise ValueError("原始生成 citations 必须只含 evidence_id")
    ids = string_list([item["evidence_id"] for item in citations], "citations")
    if not set(ids) <= allowed_ids:
        raise ValueError("原始生成引用了上下文以外的 evidence")
    if value["status"] == "insufficient":
        if value["answer"] != INSUFFICIENT_ANSWER or ids:
            raise ValueError("insufficient 必须保持固定拒答句与空 citations")
    elif not ids:
        raise ValueError("answerable 必须有 evidence_id 引用")
    return {**value, "citation_ids": ids}


def validate_semantic_review(raw: str, allowed_ids: set[str], status: str) -> dict:
    value = parse_strict_json(raw)
    if set(value) != {"claims", "verdict", "reason"}:
        raise ValueError("原始语义 reviewer JSON 字段不严格")
    if value["verdict"] not in ("pass", "fail"):
        raise ValueError("原始 reviewer verdict 非法")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("原始 reviewer reason 必须非空")
    claims = value["claims"]
    if not isinstance(claims, list) or len(claims) > 5:
        raise ValueError("原始 reviewer claims 必须是最多 5 项的列表")
    if status == "insufficient":
        if claims:
            raise ValueError("拒答的 reviewer 不应凭空生成 claims")
    elif status != "answerable" or not claims:
        raise ValueError("answerable 的 reviewer 必须逐条列出 claims")
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {"claim", "relation", "evidence_ids"}:
            raise ValueError("原始 reviewer claim 字段不严格")
        if not isinstance(claim["claim"], str) or not claim["claim"].strip():
            raise ValueError("原始 reviewer claim 为空")
        relation = claim["relation"]
        if relation not in ("supported", "contradicted", "not_found"):
            raise ValueError("原始 reviewer relation 非法")
        ids = string_list(claim["evidence_ids"], "reviewer evidence_ids")
        if not set(ids) <= allowed_ids:
            raise ValueError("原始 reviewer 引用了回答没有引用的 evidence")
        if (relation == "not_found" and ids) or (relation != "not_found" and not ids):
            raise ValueError("原始 reviewer relation 与 evidence_ids 不一致")
    # fail 是可以持久化的质量负结果，不用 pass 覆盖模型原判。
    if value["verdict"] == "pass" and any(c["relation"] != "supported" for c in claims):
        raise ValueError("原始 reviewer 同时给出 pass 和未支持的结论")
    return value


def validate_gap_reflection(raw: str, initial_evidence: list[dict]) -> dict:
    """复用 C6 的四字段反思契约，并将支持引文绑定到首轮实际证据。"""
    value = parse_strict_json(raw)
    if set(value) != {"sufficient", "missing", "repair_queries", "supporting_quotes"}:
        raise ValueError("缺口反思字段必须精确为 C6 的四字段契约")
    if type(value["sufficient"]) is not bool:
        raise ValueError("缺口反思 sufficient 必须是 boolean")
    missing = string_list(value["missing"], "missing", maximum=12)
    queries = string_list(value["repair_queries"], "repair_queries", maximum=3)
    quotes = string_list(value["supporting_quotes"], "supporting_quotes", maximum=12)
    if value["sufficient"]:
        if missing or queries or not quotes:
            raise ValueError("资料足够时必须有 supporting_quotes，且不能要求补查")
    elif not missing or not queries:
        raise ValueError("资料不足时必须同时给出 missing 与 repair_queries")
    from common.eval_utils import normalize_text

    candidates = [normalize_text(row["quote"]) for row in initial_evidence]
    normalized = [normalize_text(quote) for quote in quotes]
    if len(normalized) != len(set(normalized)) or any(
        not quote or not any(quote in candidate for candidate in candidates)
        for quote in normalized
    ):
        raise ValueError("缺口反思 supporting_quotes 不能绑定到首轮实际证据")
    return value


def context_block(row: dict) -> str:
    return f"[evidence_id={row['evidence_id']} page={row['page']}]\n{row['quote']}"


def budget_ids(ids: list[str], evidence: dict, budget: int) -> tuple[list[str], int]:
    selected = []
    used = 0
    for evidence_id in ids:
        block = context_block(evidence[evidence_id])
        if not selected and len(block) > budget:
            raise ValueError("首条 evidence 超过上下文预算")
        added = len(block) + (2 if selected else 0)
        if used + added > budget:
            break
        selected.append(evidence_id)
        used += added
    require(bool(selected), "预算后没有 evidence")
    return selected, used


def evaluate_case(case: dict, metadata: dict, qrels: list[dict]) -> tuple[str, list[str]]:
    """用原始模型解析结果和事后 qrels 计算质量结论，不抛弃负结果。"""
    essential = {r["evidence_id"] for r in qrels if r["relevance"] == 1 and r.get("essential")}
    positives = {r["evidence_id"] for r in qrels if r["relevance"] == 1}
    failures = []
    if metadata["answerability"] == "answerable":
        if case["response_status"] != "answerable":
            failures.append("model_returned_insufficient")
        if not essential:
            failures.append("missing_essential_qrels")
        for field, label in (
            ("retrieved_ids", "essential_not_retrieved"),
            ("context_evidence_ids", "essential_not_in_context"),
            ("citation_ids", "essential_not_cited"),
        ):
            if not essential <= set(case[field]):
                failures.append(label)
        if not case["semantic_claims"] or any(
            c["relation"] != "supported" for c in case["semantic_claims"]
        ):
            failures.append("unsupported_answer_claims")
        success = "pass_answerable"
    else:
        if positives:
            failures.append("unanswerable_has_positive_qrels")
        if (
            case["response_status"] != "insufficient"
            or case["answer"] != INSUFFICIENT_ANSWER
            or case["citation_ids"]
            or case["semantic_claims"]
        ):
            failures.append("invalid_insufficient")
        success = "pass_insufficient"
    if case["semantic_verdict"] != "pass":
        failures.append("semantic_reviewer_failed")
    return ("fail" if failures else success), failures


def validate_notebook(notebook: dict, course_root: Path = COURSE_ROOT) -> dict:
    if str(course_root) not in sys.path:
        sys.path.insert(0, str(course_root))
    from common.dataset import read_jsonl
    from common.eval_utils import build_bm25_chunk_search

    cells = notebook.get("cells", [])
    code = [cell for cell in cells if cell.get("cell_type") == "code"]
    require(len(code) >= 7, "capstone 缺少检索、预算、缺口补查、生成、语义与事后单元")
    require(all(type(c.get("execution_count")) is int for c in code), "未保存全部执行记录")
    sources = ["".join(c.get("source", [])) for c in code]

    def cell_index(fragment):
        matches = [i for i, source in enumerate(sources) if fragment in source]
        require(len(matches) == 1, f"必须恰有一个 {fragment!r} 代码单元")
        return matches[0]

    stages = [cell_index(fragment) for fragment in (
        "retrieved =", "context_rows =", "gap_raw = llm_call(",
        "raw_model = {}", "semantic_results = {}", "qrels_by_case =",
    )]
    require(stages == sorted(stages) and len(set(stages)) == 6, "capstone 阶段顺序错误")
    pre_hoc = "\n".join(sources[:stages[-1]])
    leaked = ("answerability", "reference_claims", "reference_answer", "expected_pages",
              "qrels.jsonl", '"relevance"', '"essential"', '"usage"', "query_meta")
    require(not any(token in pre_hoc for token in leaked), "生成/反思/reviewer 前读取评测标签")
    require("len(context_text[case_id]) != used" in "\n".join(sources), "缺少实际字符预算检查")
    require("INSUFFICIENT_ANSWER" in sources[stages[3]], "缺少严格拒答校验")
    for index in (stages[2], stages[3], stages[4]):
        require(sources[index].count("llm_call(") == 1, "每个模型阶段必须只有一次显式调用位置")
    require(not any(token in "\n".join(sources) for token in (
        "load_dataset(", "load_query_records(", "load_evidence_records(", "load_qrel_records("
    )), "capstone 不应加载 C2 训练候选")

    outputs = [o for cell in code for o in cell.get("outputs", [])]
    require(not any(o.get("output_type") == "error" for o in outputs), "Notebook 保存了执行错误")
    audits = [o["data"][AUDIT_MIME] for o in outputs if AUDIT_MIME in o.get("data", {})]
    raw_calls = [o["data"][RAW_MIME] for o in outputs if RAW_MIME in o.get("data", {})]
    require(len(audits) == 1, "必须保存一条完整 capstone audit")
    audit = audits[0]
    require(audit.get("schema_version") == 2, "audit schema_version 必须为 2")
    for key, expected in {
        "trace_kind": "c7_endpoint_capstone", "model": "glm-4-flash",
        "api_key_source": ".env:ZHIPUAI_API_KEY", "max_retries": 0,
        "retriever": "bm25", "evidence_scope": "canonical evidence_type=quote only",
        "top_k": TOP_K, "context_budget_chars": CONTEXT_BUDGET,
        "gap_case_id": GAP_CASE_ID,
    }.items():
        require(audit.get(key) == expected, f"audit {key} 不符合协议")
    require(audit.get("semantic_judge", {}).get("calls") == len(CASE_IDS), "reviewer 调用次数错误")
    require(audit["semantic_judge"].get("model") == "glm-4-flash", "reviewer 模型错误")
    require(bool(audit["semantic_judge"].get("limitation")), "缺少 reviewer 局限说明")
    cases = audit.get("cases", [])
    require([c.get("query_id") for c in cases] == list(CASE_IDS), "固定四题不完整或顺序错误")

    dataset_root = course_root / "data" / "dataset"
    queries = {r["query_id"]: r for r in read_jsonl(dataset_root / "queries.jsonl")}
    evidence = {
        r["evidence_id"]: {k: r[k] for k in ("evidence_id", "page", "quote")}
        for r in read_jsonl(dataset_root / "evidence.jsonl") if r.get("evidence_type") == "quote"
    }
    qrel_rows = read_jsonl(dataset_root / "qrels.jsonl")
    search = build_bm25_chunk_search([
        {"chunk_id": r["evidence_id"], "pages": [r["page"]], "text": r["quote"]}
        for r in evidence.values()
    ])
    call_index = {}
    for call in raw_calls:
        key = (call.get("stage"), call.get("query_id"))
        require(key not in call_index, f"原始响应 artifact 重复：{key}")
        require(call.get("model") == "glm-4-flash" and call.get("max_retries") == 0, "原始调用配置不符")
        call_index[key] = call.get("raw_response")
    require(len(call_index) == 2 * len(CASE_IDS) + 1, "必须持久化 4 次生成、4 次 reviewer、1 次缺口反思")
    gap_passed = False
    for case in cases:
        case_id = case["query_id"]
        metadata = queries[case_id]
        require("regression" in metadata.get("usage", []), f"不是 regression 问题：{case_id}")
        require(case.get("question") == metadata["text"], f"问题文字被改写：{case_id}")
        initial_top_k = 1 if case_id == GAP_CASE_ID else TOP_K
        initial_ids = [r.chunk_id for r in search(metadata["text"], top_k=initial_top_k)]
        require(case.get("initial_top_k") == initial_top_k, "首轮召回设置不符")
        require(case.get("initial_retrieved_ids") == initial_ids, f"首轮检索无法重放：{case_id}")
        initial_context, initial_chars = budget_ids(initial_ids, evidence, CONTEXT_BUDGET)
        require(case.get("initial_context_evidence_ids") == initial_context, "首轮上下文无法重放")
        require(case.get("initial_context_chars") == initial_chars, "首轮字符数不符")
        merged_ids = list(initial_ids)
        repair = case.get("gap_repair")
        if case_id == GAP_CASE_ID:
            require(isinstance(repair, dict), "缺少 C6 缺口补查轨迹")
            raw_gap = repair.get("raw_reflection")
            require(call_index.get(("gap_reflection", case_id)) == raw_gap, "缺口原文与原始调用 artifact 不符")
            reflected = validate_gap_reflection(raw_gap, [evidence[i] for i in initial_context])
            require(repair.get("reflection") == reflected, "缺口派生字段与原始响应不符")
            triggered = not reflected["sufficient"]
            require(repair.get("triggered") is triggered, "缺口触发条件与原始 sufficient 不符")
            reason = "reflection.sufficient=false: " + "；".join(reflected["missing"]) if triggered else "reflection.sufficient=true"
            require(repair.get("trigger_reason") == reason, "补查触发理由被修改")
            rounds = repair.get("retrieval_rounds", [])
            require([r.get("query") for r in rounds] == reflected["repair_queries"], "补查词不是模型原始输出")
            for round_ in rounds:
                ids = [r.chunk_id for r in search(round_["query"], top_k=TOP_K)]
                require(round_.get("retrieved_ids") == ids, "补查结果无法由 BM25 重放")
                merged_ids.extend(i for i in ids if i not in merged_ids)
        else:
            require(repair is None, "非缺口题不应暗中补查")
        require(case.get("retrieved_ids") == merged_ids, "最终候选不是实际检索的稳定去重合并")
        context_ids, used = budget_ids(merged_ids, evidence, CONTEXT_BUDGET)
        require(case.get("context_evidence_ids") == context_ids, "最终上下文预算无法重放")
        require(case.get("context_chars") == used <= CONTEXT_BUDGET, "最终字符预算错误")
        generation_raw = case.get("raw_generation_response")
        reviewer_raw = case.get("raw_semantic_response")
        require(call_index.get(("generation", case_id)) == generation_raw, "生成原文与调用 artifact 不符")
        require(call_index.get(("semantic_review", case_id)) == reviewer_raw, "reviewer 原文与调用 artifact 不符")
        generated = validate_generation(generation_raw, set(context_ids))
        reviewed = validate_semantic_review(reviewer_raw, set(generated["citation_ids"]), generated["status"])
        for key, value in {
            "response_status": generated["status"], "answer": generated["answer"],
            "citation_ids": generated["citation_ids"],
            "hydrated_citations": [evidence[i] for i in generated["citation_ids"]],
            "semantic_claims": reviewed["claims"], "semantic_verdict": reviewed["verdict"],
            "semantic_reason": reviewed["reason"], "answerability": metadata["answerability"],
        }.items():
            require(case.get(key) == value, f"{case_id}: 派生字段 {key} 与原文/canonical 不符")
        require(case.get("model_called") is True and case.get("semantic_judge_model_called") is True, "缺少真实模型调用记录")
        qrels = [r for r in qrel_rows if r["query_id"] == case_id]
        essential = {r["evidence_id"] for r in qrels if r["relevance"] == 1 and r.get("essential")}
        positives = {r["evidence_id"] for r in qrels if r["relevance"] == 1}
        require(case.get("essential_ids") == sorted(essential), "essential IDs 与 canonical qrels 不符")
        require(case.get("qrels_positive_citation_ids") == sorted(set(generated["citation_ids"]) & positives), "正相关引用集合不符")
        verdict, failures = evaluate_case(case, metadata, qrels)
        require(case.get("verdict") == verdict and case.get("failure_reasons") == failures, "质量结论并非由真实响应重新计算")
        if repair is not None:
            missing_before = sorted(essential - set(initial_context))
            new_ids = [i for i in context_ids if i not in initial_context]
            require(repair.get("first_round_missing_essential_ids") == missing_before, "首轮事后缺口与 qrels 不符")
            require(repair.get("new_evidence") == [evidence[i] for i in new_ids], "补查新证据不符")
            gap_passed = bool(repair["triggered"] and missing_before and new_ids and
                              essential <= set(context_ids) and verdict == "pass_answerable")
            require(repair.get("gap_closed") is gap_passed, "缺口闭环结论被强制改写")
    all_passed = gap_passed and all(c["verdict"].startswith("pass_") for c in cases)
    require(audit.get("all_passed") is all_passed, "all_passed 与重算结果不符")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-all-passed", action="store_true", help="另行要求全部质量验收通过")
    args = parser.parse_args()
    audit = validate_notebook(json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8")))
    passed = sum(case["verdict"].startswith("pass_") for case in audit["cases"])
    print(f"capstone audit verified: {len(audit['cases'])} cases, {passed} quality passed; all_passed={audit['all_passed']}")
    for case in audit["cases"]:
        print(f"  {case['query_id']}: {case['verdict']} {case['failure_reasons']}")
    if args.require_all_passed and not audit["all_passed"]:
        raise SystemExit("capstone quality acceptance failed; negative results remain in the audit")


if __name__ == "__main__":
    main()
