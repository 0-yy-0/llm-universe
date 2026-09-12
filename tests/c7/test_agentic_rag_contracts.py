"""Agentic RAG 的语义和执行轨迹契约。

这些检查只读取已保存的 Notebook JSON 和 canonical 查询标注，不执行 Notebook，因而
不会触发模型或向量检索调用。重点是检查 ``Plan → Act → Observe → Verify →
Repair → Verify → Answer`` 的真实闭环，而不只是检查几个字段是否存在。
"""

from __future__ import annotations

import ast
import json
import re
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
COURSE = ROOT / "notebook" / "C7 高级 RAG 技巧"
NOTEBOOK = COURSE / "6. 处理信息缺口" / "让系统决定怎样检索.ipynb"
QUERIES = COURSE / "data" / "dataset" / "queries.jsonl"
AUDIT_MIME = "application/vnd.llm-universe.tutorial-audit+json"
CASE_IDS = {"agentic_mds_ksvd", "agentic_kpca_centering"}
ANSWER_ALIASES = {
    "answer_check",
    "agentic_answer",
    "answer_draft",
    "answer_revision",
    "final_answer",
}

if str(COURSE) not in sys.path:
    sys.path.insert(0, str(COURSE))
from common.eval_utils import normalize_text  # noqa: E402


def _notebook() -> dict:
    value = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _code_text() -> str:
    cells = _notebook().get("cells", [])
    return "\n".join(
        "".join(str(part) for part in cell.get("source", []))
        for cell in cells
        if isinstance(cell, dict) and cell.get("cell_type") == "code"
    )


def _function_source(function_name: str) -> str:
    """Return one top-level Notebook function without executing the cell."""

    source = _code_text()
    tree = ast.parse(source)
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ),
        None,
    )
    assert function is not None, function_name
    lines = source.splitlines()
    return "\n".join(lines[function.lineno - 1 : function.end_lineno])


def _saved_audits() -> list[dict]:
    audits: list[dict] = []
    for cell in _notebook().get("cells", []):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if not isinstance(output, dict):
                continue
            data = output.get("data")
            value = data.get(AUDIT_MIME) if isinstance(data, dict) else None
            values = value if isinstance(value, list) else [value]
            audits.extend(item for item in values if isinstance(item, dict))
    return audits


def _audits_by_case() -> dict[str, dict]:
    audits = _saved_audits()
    by_case = {
        str(audit.get("case_id")): audit
        for audit in audits
        if isinstance(audit.get("case_id"), str)
    }
    assert set(by_case) == CASE_IDS
    return by_case


def _query_annotations() -> dict[str, dict]:
    rows = [
        json.loads(line)
        for line in QUERIES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {str(row["query_id"]): row for row in rows if row.get("query_id") in CASE_IDS}


def _canonical_evidence() -> dict[str, dict]:
    path = COURSE / "data" / "dataset" / "evidence.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {str(row["evidence_id"]): row for row in rows if row.get("evidence_id")}


def _essential_qrel_ids(annotation: dict) -> set[str]:
    return {
        str(evidence_id)
        for reference_claim in annotation.get("reference_claims", [])
        if isinstance(reference_claim, dict)
        for evidence_id in reference_claim.get("evidence_ids", [])
        if isinstance(evidence_id, str)
    }


def _final_evidence_catalog(audit: dict) -> dict[str, dict]:
    """Read the evidence catalog actually passed to the final answer stage.

    The final page list is not enough: a page can contain several canonical
    quotes, so a fabricated ID from the same page must not satisfy the answer
    contract.  The audit should therefore persist the final merged catalog (a
    dict or list is accepted to keep the check independent of presentation).
    """

    outputs = audit.get("model_outputs")
    assert isinstance(outputs, dict)
    for container in (outputs, audit):
        for name in ("candidate_catalog", "final_evidence_catalog", "final_evidence"):
            value = container.get(name)
            if isinstance(value, dict):
                rows = list(value.values())
            elif isinstance(value, list):
                rows = value
            else:
                continue
            catalog = {
                str(row["evidence_id"]): row
                for row in rows
                if isinstance(row, dict)
                and row.get("evidence_id")
                and isinstance(row.get("quote"), str)
            }
            if catalog:
                return catalog
    raise AssertionError(
        "最终 audit 必须保存传给答案阶段的 candidate_catalog，不能只保存页码"
    )


def _trace(audit: dict) -> list[dict]:
    outputs = audit.get("model_outputs")
    assert isinstance(outputs, dict)
    value = outputs.get("trace")
    assert isinstance(value, list)
    assert all(isinstance(step, dict) and isinstance(step.get("step"), str) for step in value)
    return value


def _stage_parsed(audit: dict, *names: str) -> dict | None:
    outputs = audit.get("model_outputs")
    if not isinstance(outputs, dict):
        return None
    for name in names:
        value = outputs.get(name)
        if isinstance(value, dict) and isinstance(value.get("parsed"), dict):
            return value["parsed"]
    return None


def _final_answer(audit: dict) -> dict:
    """最终答案唯一从 ``model_outputs.answer`` 读取。"""

    outputs = audit.get("model_outputs")
    assert isinstance(outputs, dict)
    assert not (ANSWER_ALIASES & set(outputs)), (
        "model_outputs 不得保留历史答案别名",
        ANSWER_ALIASES & set(outputs),
    )
    value = outputs.get("answer")
    assert isinstance(value, dict), "最终答案必须唯一保存在 model_outputs.answer"
    parsed = value.get("parsed")
    assert isinstance(parsed, dict), "model_outputs.answer.parsed 必须是结构化对象"
    return parsed


def _claim_bindings(claim: object) -> tuple[list[str], list[dict]]:
    """Validate the hydrated multi-evidence claim shape without aliases."""

    assert isinstance(claim, dict), "每条 answer claim 必须是对象"
    assert set(claim) == {"statement", "evidence_ids", "evidence"}, (
        "answered claim 必须精确包含 statement、evidence_ids、evidence",
        claim,
    )
    assert isinstance(claim["statement"], str) and claim["statement"].strip()
    evidence_ids = claim["evidence_ids"]
    evidence = claim["evidence"]
    assert isinstance(evidence_ids, list) and evidence_ids, claim
    assert all(isinstance(item, str) and item.strip() for item in evidence_ids), claim
    assert len(evidence_ids) == len(set(evidence_ids)), (
        "每条 claim 的 evidence_ids 必须非空且唯一",
        claim,
    )
    assert isinstance(evidence, list) and evidence, claim
    assert len(evidence) == len(evidence_ids), claim
    for item in evidence:
        assert isinstance(item, dict), claim
        assert set(item) == {"evidence_id", "page", "quote"}, (
            "每条 evidence 必须精确包含 evidence_id、page、quote",
            item,
        )
        assert isinstance(item["evidence_id"], str) and item["evidence_id"].strip(), item
        assert isinstance(item["page"], int) and not isinstance(item["page"], bool), item
        assert isinstance(item["quote"], str) and item["quote"].strip(), item
    assert [item["evidence_id"] for item in evidence] == evidence_ids, claim
    return evidence_ids, evidence


def _extract_pure_function(function_name: str) -> object:
    """只编译 Notebook 中一个纯 parser 及其依赖，绝不初始化模型/检索器。"""

    tree = ast.parse(_code_text())
    definitions: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node

    # Include only the dependency closure of the pure parser.  This keeps the test
    # independent of model/retriever initialization while allowing the Notebook to
    # factor normalization into a small helper (rather than forcing a name here).
    wanted = {function_name}
    nodes: list[ast.AST] = []
    while wanted:
        name = wanted.pop()
        node = definitions.get(name)
        if node is None or node in nodes:
            continue
        nodes.append(node)
        referenced = {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        }
        wanted.update(referenced & definitions.keys())
    assert any(isinstance(node, ast.FunctionDef) and node.name == function_name for node in nodes)
    namespace: dict[str, object] = {
        "json": json,
        "re": re,
        "unicodedata": __import__("unicodedata"),
        "normalize_text": normalize_text,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace[function_name]


def _extract_agentic_parser() -> object:
    return _extract_pure_function("parse_agentic_verify")


def test_agentic_saved_audits_have_two_cases_and_structured_stage_records():
    by_case = _audits_by_case()
    for case_id, audit in by_case.items():
        outputs = audit.get("model_outputs")
        assert isinstance(outputs, dict), case_id
        assert not (ANSWER_ALIASES & set(outputs)), (
            case_id,
            "model_outputs 不得保留历史答案别名",
            ANSWER_ALIASES & set(outputs),
        )
        for stage in ("plan", "verify", "answer"):
            value = outputs.get(stage)
            assert isinstance(value, dict), (case_id, stage)
            assert isinstance(value.get("raw"), str) and value["raw"].strip(), (case_id, stage)
            assert "parsed" in value, (case_id, stage)
        assert _trace(audit), case_id


def test_answerable_cases_finish_sufficient_and_answered_with_essential_evidence():
    """固定的两个 Agentic 案例必须真正走到可回答的最终状态。"""

    annotations = _query_annotations()
    assert set(annotations) == CASE_IDS
    for case_id, audit in _audits_by_case().items():
        annotation = annotations[case_id]
        assert annotation.get("answerability") == "answerable", (case_id, annotation)

        final_verify = _stage_parsed(audit, "verify")
        assert isinstance(final_verify, dict), case_id
        assert final_verify.get("sufficient") is True, (case_id, final_verify)
        assert final_verify.get("missing") == [], (case_id, final_verify)

        answer = _final_answer(audit)
        assert answer.get("status") == "answered", (case_id, answer)
        claims = answer.get("claims")
        assert isinstance(claims, list) and claims, (case_id, answer)
        claimed_ids = {
            evidence_id
            for claim in claims
            for evidence_id in _claim_bindings(claim)[0]
        }

        # The final catalog is the saved, actual evidence set given to the
        # answer stage.  Every canonical evidence item marked essential by the
        # query must survive into that final set.
        essential_ids = _essential_qrel_ids(annotation)
        assert essential_ids, (case_id, annotation)
        final_catalog = _final_evidence_catalog(audit)
        assert essential_ids <= set(final_catalog), (
            case_id,
            "最终 candidate_catalog 必须保留全部 essential evidence",
            essential_ids,
            set(final_catalog),
        )
        assert essential_ids <= claimed_ids, (
            case_id,
            "claims 合并引用必须覆盖全部 essential qrels",
            essential_ids,
            claimed_ids,
        )


def test_verify_prompt_uses_actual_compact_canonical_evidence_payload():
    """拦截 verify_once 的真实 prompt，确认传入的是最终精简 catalog。"""

    tree = ast.parse(_code_text())
    verify_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "verify_once"
    )
    prompt_calls: list[str] = []
    catalog_calls: list[list] = []
    compact_catalog = {
        "e_compact": {
            "evidence_id": "e_compact",
            "page": 7,
            "quote": "canonical quote",
        }
    }
    hits = [SimpleNamespace(page=7, text="full retrieved context", chunk_id="h1")]
    requirements = [{"requirement_id": "action_1", "query": "q", "purpose": "p"}]

    def candidate_evidence_catalog(actual_hits: list) -> dict[str, dict]:
        catalog_calls.append(actual_hits)
        return compact_catalog

    def call_glm_once(prompt: str, *, max_tokens: int = 900) -> str:
        prompt_calls.append(prompt)
        return json.dumps(
            {"sufficient": True, "covered": ["action_1"], "missing": []},
            ensure_ascii=False,
        )

    namespace = {
        "candidate_evidence_catalog": candidate_evidence_catalog,
        "call_glm_once": call_glm_once,
        "parse_agentic_verify": lambda raw, requirement_ids: json.loads(raw),
        "observed_payload": lambda actual_hits: [{"text": "full retrieved context"}],
        "json": json,
    }
    exec(compile(ast.Module(body=[verify_node], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    result = namespace["verify_once"]("question", hits, "initial", requirements)

    assert result["parsed"] == {"sufficient": True, "covered": ["action_1"], "missing": []}
    assert catalog_calls == [hits]
    assert len(prompt_calls) == 1
    prompt = prompt_calls[0]
    assert json.dumps(list(compact_catalog.values()), ensure_ascii=False) in prompt
    assert "字段只有 evidence_id、page、quote" in prompt
    assert "full retrieved context" in prompt


def test_agentic_generation_does_not_leak_qrels_or_annotations():
    """生成阶段只消费检索结果；qrels 仅由测试离线读取核验。"""

    source = _code_text()
    for forbidden in ("load_qrel_records", "ESSENTIAL_EVIDENCE_BY_CASE", "required_evidence_ids"):
        assert forbidden not in source, f"Notebook 生成代码不得依赖 {forbidden}"

    for function_name in ("parse_agentic_answer", "agentic_pipeline"):
        function_source = _function_source(function_name)
        for forbidden in (
            "qrels",
            "reference_claims",
            "expected_pages",
            "expected_keywords",
            "annotation",
        ):
            assert not re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(forbidden)}(?![A-Za-z0-9_])",
                function_source,
                flags=re.IGNORECASE,
            ), f"{function_name} 不得使用 {forbidden}"


def test_agentic_verify_parser_rejects_normalized_covered_missing_overlap():
    parse_agentic_verify = _extract_agentic_parser()
    overlapping = {
        "sufficient": False,
        "covered": ["MDS 的降维目标"],
        "missing": ["MDS降维目标"],
    }
    with pytest.raises((TypeError, ValueError), match="(?i)covered|missing|重复|互斥"):
        parse_agentic_verify(json.dumps(overlapping, ensure_ascii=False))


def test_agentic_answer_parser_rejects_unsupported_status_and_fabricated_claims():
    parse_agentic_answer = _extract_pure_function("parse_agentic_answer")
    with pytest.raises((TypeError, ValueError), match="(?i)资料不足|answer|insufficient"):
        parse_agentic_answer(
            json.dumps(
                {"answer": "资料不足，无法确定。", "status": "insufficient", "claims": []},
                ensure_ascii=False,
            ),
            {},
            expected_sufficient=False,
        )
    insufficient = parse_agentic_answer(
        json.dumps(
            {"status": "insufficient", "claims": []},
            ensure_ascii=False,
        ),
        {},
        expected_sufficient=False,
    )
    assert insufficient["answer"] == "资料不足，无法可靠回答。"
    assert insufficient["status"] == "insufficient"
    assert insufficient["claims"] == []
    with pytest.raises((TypeError, ValueError), match="(?i)claims|insufficient|status"):
        parse_agentic_answer(
            json.dumps(
                {
                    "status": "insufficient",
                    "claims": [{"statement": "没有证据的事实", "evidence_ids": ["e_fake"]}],
                },
                ensure_ascii=False,
            ),
            {},
            expected_sufficient=False,
        )
    with pytest.raises((TypeError, ValueError), match="(?i)evidence_id|真实候选"):
        parse_agentic_answer(
            json.dumps(
                {
                    "status": "answered",
                    "claims": [{"statement": "事实", "evidence_ids": ["e_fake"]}],
                },
                ensure_ascii=False,
            ),
            {},
            expected_sufficient=True,
        )
    with pytest.raises((TypeError, ValueError), match="(?i)evidence_id|evidence_ids|字段"):
        parse_agentic_answer(
            json.dumps(
                {
                    "status": "answered",
                    # The former singular evidence_id shape is intentionally
                    # rejected; the parser must not keep a compatibility path.
                    "claims": [{"statement": "事实", "evidence_id": "e1"}],
                },
                ensure_ascii=False,
            ),
            {"e1": {"evidence_id": "e1", "page": 7, "quote": "事实原文"}},
            expected_sufficient=True,
        )
    with pytest.raises((TypeError, ValueError), match="(?i)duplicate|重复|唯一|evidence_ids"):
        parse_agentic_answer(
            json.dumps(
                {
                    "status": "answered",
                    "claims": [{"statement": "事实", "evidence_ids": ["e1", "e1"]}],
                },
                ensure_ascii=False,
            ),
            {"e1": {"evidence_id": "e1", "page": 7, "quote": "事实原文"}},
            expected_sufficient=True,
        )
    parsed = parse_agentic_answer(
        json.dumps(
                {
                    "status": "answered",
                    "claims": [{"statement": "事实原文事实补充原文", "evidence_ids": ["e1", "e2"]}],
                },
                ensure_ascii=False,
            ),
        {
            "e1": {"evidence_id": "e1", "page": 7, "quote": "事实原文"},
            "e2": {"evidence_id": "e2", "page": 8, "quote": "事实补充原文"},
        },
        expected_sufficient=True,
    )
    assert parsed["answer"] == "事实原文事实补充原文"
    assert parsed["claims"] == [
        {
            "statement": "事实原文事实补充原文",
            "evidence_ids": ["e1", "e2"],
            "evidence": [
                {"evidence_id": "e1", "page": 7, "quote": "事实原文"},
                {"evidence_id": "e2", "page": 8, "quote": "事实补充原文"},
            ],
        }
    ]
    with pytest.raises((TypeError, ValueError), match="(?i)statement|支持|answer"):
        parse_agentic_answer(
            json.dumps(
                {
                    "status": "answered",
                    "claims": [{"statement": "事实。", "evidence_ids": ["e1"]}],
                },
                ensure_ascii=False,
            ),
            {"e1": {"evidence_id": "e1", "page": 7, "quote": "事实原文"}},
            expected_sufficient=True,
        )


def test_agentic_claim_parser_rejects_semantic_reversals_not_just_low_overlap():
    """反义改写即使共享大部分词，也必须因非 quote 拼接而拒绝。"""

    parse_agentic_answer = _extract_pure_function("parse_agentic_answer")
    cases = [
        (
            "MDS 算法的降维准则是不要求原始空间中样本之间的距离在低维空间中得以保持。",
            {"e_mds": {"evidence_id": "e_mds", "page": 133, "quote": "MDS 算法的降维准则是要求原始空间中样本之间的距离在低维空间中 得以保持"}},
        ),
        (
            "即使xi已进行中心化，但zi仍然是中心化的，此时本节推导仍然成立。",
            {"e_kpca": {"evidence_id": "e_kpca", "page": 132, "quote": "即使xi 已进行中心化, 但zi 却不一定是中心化的, 此时本节推导 均不再成立"}},
        ),
    ]
    for statement, catalog in cases:
        with pytest.raises((TypeError, ValueError), match="(?i)statement|支持|quote|answer"):
            parse_agentic_answer(
                json.dumps(
                    {
                        "status": "answered",
                        "claims": [{"statement": statement, "evidence_ids": list(catalog)}],
                    },
                    ensure_ascii=False,
                ),
                catalog,
                expected_sufficient=True,
            )


def test_agentic_claim_parser_preserves_numeric_operator_symbols():
    """负号、小数点、斜杠和百分号变化不能被版式规范化吞掉。"""

    parse_agentic_answer = _extract_pure_function("parse_agentic_answer")
    counterexamples = [
        (
            "值为1",
            {"e_negative": {"evidence_id": "e_negative", "page": 7, "quote": "值为-1"}},
        ),
        (
            "误差为001",
            {"e_decimal": {"evidence_id": "e_decimal", "page": 7, "quote": "误差为0.01"}},
        ),
        (
            "比例为12",
            {"e_fraction": {"evidence_id": "e_fraction", "page": 7, "quote": "比例为1/2"}},
        ),
        (
            "准确率从10%下降到20%",
            {"e_percent": {"evidence_id": "e_percent", "page": 7, "quote": "准确率从10%提高到20%"}},
        ),
        (
            "x=1",
            {"e_not_equal": {"evidence_id": "e_not_equal", "page": 7, "quote": "x!=1"}},
        ),
        (
            "3",
            {"e_factorial": {"evidence_id": "e_factorial", "page": 7, "quote": "3!"}},
        ),
        (
            "xi",
            {"e_subscript": {"evidence_id": "e_subscript", "page": 7, "quote": "x_i"}},
        ),
        (
            "a*b+c",
            {"e_parentheses": {"evidence_id": "e_parentheses", "page": 7, "quote": "a*(b+c)"}},
        ),
    ]
    for statement, catalog in counterexamples:
        with pytest.raises((TypeError, ValueError), match="(?i)statement|支持|quote|answer"):
            parse_agentic_answer(
                json.dumps(
                    {
                        "status": "answered",
                        "claims": [{"statement": statement, "evidence_ids": list(catalog)}],
                    },
                    ensure_ascii=False,
                ),
                catalog,
                expected_sufficient=True,
            )

    statement = "数值为-1，误差为0.01，比例为1/2，变化为10%"
    catalog = {
        "e_symbols": {
            "evidence_id": "e_symbols",
            "page": 7,
            "quote": "数值为 -1，误差为 0.01，比例为 1/2，变化为 10%",
        }
    }
    parsed = parse_agentic_answer(
        json.dumps(
            {
                "status": "answered",
                "claims": [{"statement": statement, "evidence_ids": ["e_symbols"]}],
            },
            ensure_ascii=False,
        ),
        catalog,
        expected_sufficient=True,
    )
    assert parsed["claims"][0]["evidence_ids"] == ["e_symbols"]


def test_agentic_runtime_uses_search_projection_without_loading_full_package(monkeypatch):
    """AST 定位运行时入口，再用 monkeypatch 证明它不经 load_dataset。"""

    tree = ast.parse(_code_text())
    dataset_imports = [
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "common.dataset"
        for alias in node.names
    ]
    assert "load_search_evidence" in dataset_imports
    assert "load_evidence_records" not in dataset_imports
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_search_evidence"
        for node in ast.walk(tree)
    )

    import common.dataset as dataset

    def fail_full_package(*args, **kwargs):
        raise AssertionError("Agentic retrieval must not call load_dataset")

    monkeypatch.setattr(dataset, "load_dataset", fail_full_package)
    rows = dataset.load_search_evidence()
    assert rows and all(set(row) == {"evidence_id", "page", "quote"} for row in rows)


def test_simulated_unit_repair_reverify_pipeline_control_flow_with_mocks():
    """仅 simulated/unit：mock glm/search 实际执行 repair→reverify 分支。"""

    source = _code_text()
    tree = ast.parse(source)
    pipeline_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "agentic_pipeline")
    events: list[str] = []
    answer_prompts: list[str] = []
    plan = {
        "goal": "mock goal",
        "actions": [
            {"action_id": "action_1", "tool": "dense", "query": "q1", "purpose": "r1"},
            {"action_id": "action_2", "tool": "bm25", "query": "q2", "purpose": "r2"},
        ],
    }
    requirements = [
        {"requirement_id": "action_1", "query": "q1", "purpose": "r1"},
        {"requirement_id": "action_2", "query": "q2", "purpose": "r2"},
    ]

    class Hit:
        def __init__(self, page: int, chunk_id: str):
            self.page = page
            self.chunk_id = chunk_id
            self.text = f"mock quote {chunk_id}"

    def call_glm_once(prompt: str, *, max_tokens: int = 900) -> str:
        if "规划器" in prompt:
            events.append("glm:plan")
            return "plan"
        if "修复器" in prompt:
            events.append("glm:repair")
            return "repair"
        events.append("glm:answer")
        answer_prompts.append(prompt)
        return "answer"

    def run_tool(tool: str, query: str, top_k: int = 4):
        events.append(f"search:{tool}:{query}")
        page = {"q1": 133, "q2": 146, "repair-query": 146}[query]
        return [Hit(page, query)], {"mock": True}

    def verify_once(question: str, hits: list, phase: str, requirements: list[dict]) -> dict:
        events.append(f"verify:{phase}")
        sufficient = phase == "after_repair"
        parsed = {
            "sufficient": sufficient,
            "covered": [item["requirement_id"] for item in requirements] if sufficient else ["action_1"],
            "missing": [] if sufficient else ["action_2"],
        }
        return {"step": "verify", "phase": phase, "raw": phase, "parsed": parsed, "pages": [hit.page for hit in hits]}

    def parse_agentic_answer(raw: str, candidate_catalog: dict[str, dict], *, expected_sufficient: bool) -> dict:
        events.append("answer:parse")
        return {
            "answer": "mock answer",
            "status": "answered",
            "claims": [{"statement": "mock answer", "evidence_ids": ["e1"], "evidence": [{"evidence_id": "e1", "page": 146, "quote": "mock quote"}]}],
        }

    def merge_action_evidence(action_hits: list[list], repair_hits: list | None = None, *, limit: int = 8):
        merged = [hit for group in action_hits for hit in group]
        if repair_hits is not None:
            merged.extend(repair_hits)
        return merged[:limit]

    namespace = {
        "call_glm_once": call_glm_once,
        "parse_agentic_plan": lambda raw: plan,
        "plan_requirements": lambda value: requirements,
        "run_tool": run_tool,
        "merge_action_evidence": merge_action_evidence,
        "verify_once": verify_once,
        "parse_agentic_repair": lambda raw, used_queries: {"tool": "dense", "query": "repair-query", "purpose": "repair purpose"},
        "candidate_evidence_catalog": lambda hits, **kwargs: {"e1": {"evidence_id": "e1", "page": 146, "quote": "mock quote"}},
        "parse_agentic_answer": parse_agentic_answer,
        "observed_payload": lambda hits: [],
        "format_context": lambda hits, max_chars=7000: "mock context",
        "json": json,
    }
    exec(compile(ast.Module(body=[pipeline_node], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    _, answer, outputs = namespace["agentic_pipeline"]("mock question")

    assert answer["status"] == "answered"
    assert outputs["stage_call_counts"] == {
        "plan": 1,
        "verify_initial": 1,
        "repair": 1,
        "verify_after_repair": 1,
        "answer": 1,
    }
    assert events == [
        "glm:plan",
        "search:dense:q1",
        "search:bm25:q2",
        "verify:initial",
        "glm:repair",
        "search:dense:repair-query",
        "verify:after_repair",
        "glm:answer",
        "answer:parse",
    ]
    assert len(answer_prompts) == 1
    assert 'ID=e1' in answer_prompts[0]
    assert "每条 claim" in answer_prompts[0]
    assert "1..4 个 evidence_id" in answer_prompts[0]
    assert "最终对象的键集合必须恰好是 {status, claims}" in answer_prompts[0]
    assert "只输出 status 和 claims，不输出 answer" in answer_prompts[0]
    assert '"answer":' not in answer_prompts[0]
    assert "每条 claim 恰好只列一个 evidence_id" not in answer_prompts[0]
    assert "恰好只列一个 evidence_id" not in answer_prompts[0]
    assert "唯一 claim" not in answer_prompts[0]
    assert "全部候选 evidence_id" not in answer_prompts[0]
    assert [step["step"] for step in outputs["trace"]] == [
        "plan",
        "retrieve",
        "retrieve",
        "verify",
        "repair",
        "verify",
        "stop",
        "answer",
    ]


def test_agentic_verify_partitions_stable_planned_requirement_ids():
    parse_agentic_verify = _extract_agentic_parser()
    requirement_ids = ["action_1", "action_2"]
    assert parse_agentic_verify(
        json.dumps(
            {"sufficient": True, "covered": requirement_ids, "missing": []},
            ensure_ascii=False,
        ),
        requirement_ids,
    )["sufficient"] is True
    assert parse_agentic_verify(
        json.dumps(
            {"sufficient": False, "covered": ["action_1"], "missing": ["action_2"]},
            ensure_ascii=False,
        ),
        requirement_ids,
    )["missing"] == ["action_2"]
    with pytest.raises((TypeError, ValueError), match="(?i)requirement|covered|missing|ID"):
        parse_agentic_verify(
            json.dumps(
                {"sufficient": True, "covered": ["action_1"], "missing": []},
                ensure_ascii=False,
            ),
            requirement_ids,
        )
    with pytest.raises((TypeError, ValueError), match="(?i)sufficient|missing"):
        parse_agentic_verify(
            json.dumps(
                {"sufficient": True, "covered": ["action_1"], "missing": ["action_2"]},
                ensure_ascii=False,
            ),
            requirement_ids,
        )


def test_agentic_plan_requires_stable_action_ids():
    parse_agentic_plan = _extract_pure_function("parse_agentic_plan")
    valid = parse_agentic_plan(
        json.dumps(
            {
                "goal": "g",
                "actions": [
                    {"action_id": "action_1", "tool": "dense", "query": "q1", "purpose": "p1"},
                    {"action_id": "action_2", "tool": "bm25", "query": "q2", "purpose": "p2"},
                ],
            },
            ensure_ascii=False,
        )
    )
    assert [item["action_id"] for item in valid["actions"]] == ["action_1", "action_2"]
    with pytest.raises((TypeError, ValueError), match="(?i)action_id|stable|编号"):
        parse_agentic_plan(
            json.dumps(
                {
                    "goal": "g",
                    "actions": [
                        {"action_id": "action_2", "tool": "dense", "query": "q1", "purpose": "p1"},
                    ],
                },
                ensure_ascii=False,
            )
        )
def test_action_evidence_merge_allocates_a_slot_to_each_plan_group():
    merge_action_evidence = _extract_pure_function("merge_action_evidence")

    def hit(page: int, chunk_id: str) -> SimpleNamespace:
        return SimpleNamespace(page=page, chunk_id=chunk_id, text=f"片段-{chunk_id}")

    merged = merge_action_evidence(
        [[hit(133, "mds")], [hit(146, "ksvd")]],
        [hit(125, "repair")],
        limit=2,
    )
    assert {item.page for item in merged} == {133, 146}


def test_mixed_pca_mds_catalog_keeps_both_topics_through_repair_merge():
    """明确的混合主题不应被固定语义过滤，repair 合并也不能丢掉任一类。"""

    catalog_from_rows = _extract_pure_function("candidate_catalog_from_rows")
    merge_action_evidence = _extract_pure_function("merge_action_evidence")

    pca = SimpleNamespace(page=7, chunk_id="pca", text="PCA 需要中心化数据")
    mds = SimpleNamespace(page=7, chunk_id="mds", text="MDS 要求保持距离")
    repair = SimpleNamespace(page=7, chunk_id="repair", text="MDS 要求保持距离")
    rows = [
        {"evidence_id": "e_pca", "page": 7, "quote": "PCA 需要中心化数据"},
        {"evidence_id": "e_mds", "page": 7, "quote": "MDS 要求保持距离"},
    ]

    merged = merge_action_evidence([[pca], [mds]], [repair], limit=8)
    catalog = catalog_from_rows(merged, rows)
    assert {"e_pca", "e_mds"} <= set(catalog)


def test_hybrid_rrf_keeps_same_page_complementary_chunks_at_hit_grain():
    rank_fusion = _extract_pure_function("rank_fusion")

    def hit(page: int, text: str, chunk_id: str | None = None) -> SimpleNamespace:
        return SimpleNamespace(page=page, text=text, chunk_id=chunk_id, score=1.0)

    dense_chunk = hit(146, "Sparse Coding Stage", "dense_sparse")
    dense_other = hit(133, "MDS distance", "dense_mds")
    bm25_page = hit(146, "完整页同时包含 Codebook Update Stage", None)
    fused = rank_fusion([dense_chunk, dense_other], [bm25_page], top_k=3)
    assert {item.chunk_id for item in fused} == {"dense_sparse", "dense_mds", None}
    assert any(item.text == bm25_page.text for item in fused)


def test_saved_verify_records_have_disjoint_normalized_topics():
    """实际保存的 verify 也必须遵守互斥约束，不能只把约束写在 prompt 中。"""
    requirement_key = _extract_pure_function("_requirement_key")

    for case_id, audit in _audits_by_case().items():
        outputs = audit["model_outputs"]
        verify_records = []
        for name in ("verify", "final_verify", "verify_final"):
            value = outputs.get(name)
            if isinstance(value, dict) and isinstance(value.get("parsed"), dict):
                verify_records.append(value["parsed"])
        verify_records.extend(
            step.get("parsed")
            for step in _trace(audit)
            if step.get("step") in {"verify", "final_verify", "verify_final"}
            and isinstance(step.get("parsed"), dict)
        )
        assert verify_records, case_id
        for record in verify_records:
            covered = record.get("covered")
            missing = record.get("missing")
            assert isinstance(covered, list) and isinstance(missing, list), (case_id, record)
            covered_topics = {requirement_key(item) for item in covered}
            missing_topics = {requirement_key(item) for item in missing}
            assert covered_topics.isdisjoint(missing_topics), (case_id, record)


def test_agentic_repair_merge_preserves_each_plan_action_and_essential_pages():
    audit = _audits_by_case()["agentic_mds_ksvd"]
    trace = _trace(audit)
    plan = _stage_parsed(audit, "plan")
    assert isinstance(plan, dict)
    actions = plan.get("actions")
    assert isinstance(actions, list) and len(actions) >= 2
    retrieve_steps = [step for step in trace if step.get("step") == "retrieve"]
    final_catalog = _final_evidence_catalog(audit)
    final_pages = {
        int(page)
        for page in audit.get("after", {}).get("pages", [])
        if (isinstance(page, int) and not isinstance(page, bool))
        or (isinstance(page, str) and page.strip().isdigit())
    }
    assert final_pages, "最终 audit 必须保存实际上下文页码"
    for action in actions:
        assert isinstance(action, dict)
        matches = [step for step in retrieve_steps if step.get("query") == action.get("query")]
        assert matches, f"plan action 没有对应实际 retrieve：{action}"
        candidate_pages = {
            int(page)
            for page in matches[0].get("pages", [])
            if (isinstance(page, int) and not isinstance(page, bool))
            or (isinstance(page, str) and page.strip().isdigit())
        }
        candidate_ids = {
            evidence_id
            for item in matches[0].get("evidence", [])
            if isinstance(item, dict)
            for evidence_id in item.get("matched_evidence_ids", [])
            if isinstance(evidence_id, str)
        }
        if candidate_ids:
            assert candidate_ids & set(final_catalog), (
                "repair 合并不能把某个 plan action 找到的关键 evidence 整体挤掉",
                action,
                candidate_ids,
                set(final_catalog),
            )
        else:
            # Keep old saved traces diagnosable while the explicit evidence
            # audit is being introduced; the final catalog remains mandatory.
            assert candidate_pages & final_pages, (
                "repair 合并不能把某个 plan action 找到的候选整体挤掉",
                action,
                candidate_pages,
                final_pages,
            )
    annotation = _query_annotations()["agentic_mds_ksvd"]
    expected_pages = set(annotation["expected_pages"])
    assert expected_pages <= final_pages
    assert audit.get("after", {}).get("required_page_coverage") == pytest.approx(1.0)


def test_agentic_repair_is_followed_by_final_verify_before_any_answer():
    audit = _audits_by_case()["agentic_mds_ksvd"]
    trace = _trace(audit)
    verify_indexes = [
        index
        for index, step in enumerate(trace)
        if step.get("step") in {"verify", "verify_final", "final_verify"}
    ]
    answer_indexes = [index for index, step in enumerate(trace) if step.get("step") == "answer"]
    assert verify_indexes and answer_indexes and max(verify_indexes) < min(answer_indexes)
    counts = audit["model_outputs"].get("stage_call_counts", {})
    assert isinstance(counts, dict)
    assert not (ANSWER_ALIASES & set(counts)), (
        "stage_call_counts 不得保留历史答案别名",
        ANSWER_ALIASES & set(counts),
    )
    repair_indexes = [index for index, step in enumerate(trace) if step.get("step") == "repair"]
    assert int(counts.get("repair", 0)) == len(repair_indexes)
    if not repair_indexes:
        # A sufficient initial verify legitimately skips repair; the final
        # verify-before-answer ordering is still required and is checked above.
        assert int(counts.get("verify_initial", 0)) == 1
        assert int(counts.get("verify_after_repair", 0)) == 0
        assert int(counts.get("answer", 0)) == 1
        return
    repair_index = repair_indexes[0]
    after_repair = [index for index in verify_indexes if index > repair_index]
    assert after_repair, "repair 后必须重新 verify，不能直接生成答案"
    assert min(after_repair) < min(answer_indexes)
    assert int(counts.get("repair", 0)) == 1
    verify_count = counts.get("verify")
    if verify_count is None:
        verify_count = sum(
            int(counts.get(key, 0))
            for key in ("verify_initial", "verify_after_repair", "verify_final")
        )
    assert int(verify_count) >= 2
    assert int(counts.get("answer", 0)) == 1


def test_agentic_answer_claims_bind_to_final_evidence_ids_and_quotes():
    canonical = _canonical_evidence()
    annotations = _query_annotations()
    for case_id, audit in _audits_by_case().items():
        annotation = annotations[case_id]
        assert annotation.get("answerability") == "answerable", (case_id, annotation)
        answer = _final_answer(audit)
        assert set(answer) == {"answer", "status", "claims"}, (case_id, answer)
        final_catalog = _final_evidence_catalog(audit)
        assert set(final_catalog) <= set(canonical), (case_id, final_catalog)
        for evidence_id, final_row in final_catalog.items():
            row = canonical[evidence_id]
            assert final_row["evidence_id"] == evidence_id, (case_id, final_row)
            assert int(final_row["page"]) == int(row["page"]), (case_id, final_row)
            assert str(final_row["quote"]) == str(row["quote"]), (case_id, final_row)
        assert answer.get("status") == "answered", (case_id, answer)
        claims = answer.get("claims")
        assert isinstance(claims, list) and claims, (case_id, answer)
        claimed_ids: set[str] = set()
        pages = audit.get("after", {}).get("pages", [])
        final_pages = {
            int(page)
            for page in pages
            if (isinstance(page, int) and not isinstance(page, bool))
            or (isinstance(page, str) and page.strip().isdigit())
        }
        for claim in claims:
            evidence_ids, evidence_items = _claim_bindings(claim)
            claimed_ids.update(evidence_ids)
            for evidence_id in evidence_ids:
                assert evidence_id in canonical, (case_id, evidence_id)
                assert evidence_id in final_catalog, (
                    case_id,
                    evidence_id,
                    "claim 必须引用最终合并上下文中的 evidence_id",
                )
                row = canonical[evidence_id]
                final_row = final_catalog[evidence_id]
                assert final_row["evidence_id"] == evidence_id, (case_id, evidence_id)
                assert int(final_row["page"]) == int(row["page"]), (case_id, evidence_id)
                assert str(final_row["quote"]) == str(row["quote"]), (case_id, evidence_id)
                assert int(row["page"]) in final_pages, (case_id, evidence_id, final_pages)
                evidence_item = next(
                    item for item in evidence_items if item["evidence_id"] == evidence_id
                )
                assert evidence_item["page"] == int(row["page"]), (case_id, evidence_item)
                assert evidence_item["quote"] == str(row["quote"]), (case_id, evidence_item)
        essential_ids = _essential_qrel_ids(annotation)
        assert essential_ids <= set(final_catalog), (case_id, essential_ids, final_catalog)
        assert essential_ids <= claimed_ids, (
            case_id,
            "所有 essential qrels evidence_id 都必须被 claims 合并引用",
            essential_ids,
            claimed_ids,
        )


def test_agentic_source_has_explicit_insufficient_guard_and_no_fake_fallback_path():
    source = _code_text()
    assert re.search(r"\binsufficient\b", source, flags=re.IGNORECASE)
    assert "claims" in source
    tree = ast.parse(source)
    fake_assignments: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id.lower()
            if name == "fallback" or name.startswith("fallback_") or name in {
                "retry",
                "default_answer",
                "default_claims",
                "default_evidence",
            }:
                fake_assignments.append(target.id)
    assert not fake_assignments, f"Agentic RAG 不得写入 fallback/default/retry 假结果：{fake_assignments}"

    for audit in _saved_audits():
        outputs = audit.get("model_outputs")
        if not isinstance(outputs, dict):
            continue
        assert not (ANSWER_ALIASES & set(outputs)), (
            "model_outputs 不得保留历史答案别名",
            ANSWER_ALIASES & set(outputs),
        )
        counts = outputs.get("stage_call_counts")
        if isinstance(counts, dict):
            assert int(counts.get("plan", 0)) == 1
            assert int(counts.get("repair", 0)) <= 1
            assert not (ANSWER_ALIASES & set(counts)), (
                "stage_call_counts 不得保留历史答案别名",
                ANSWER_ALIASES & set(counts),
            )
            assert int(counts.get("answer", 0)) == 1
        assert outputs.get("fallback_used") is not True
