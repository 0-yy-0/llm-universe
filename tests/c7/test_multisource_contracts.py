import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).parents[2] / "notebook/C7 高级 RAG 技巧"
NOTEBOOK = ROOT / "6. 处理信息缺口/构建多轮多来源助手.ipynb"
MANIFEST = ROOT / "data/asset_manifest.json"
AUDIT_MIME = "application/vnd.llm-universe.tutorial-audit+json"
DIRECT_SOURCE_IDS = {
    "canonical_dataset_manifest",
    "data_processing_readme",
    "evaluation_readme",
}
GENERATION_ORACLE_MARKERS = (
    "0.7941",
    "Recall@10",
    "Recall@5",
    "Recall@1",
    "data_processing_readme:L61",
    "evaluation_readme:L65",
    "expected_status",
    "expected_entity",
    "required_source_ids",
    "必须返回 status=conflict",
    "完整顺序句",
    "均有所提升",
)
REMOVED_HISTORY_MARKERS = (
    "C6",
    "supplementary",
    "5 折",
    "5折",
    "10 折",
    "10折",
    "勘误",
    "更新通知",
    "旧配置",
    "迁移",
)


def _load_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _saved_audit():
    notebook = _load_notebook()
    audits = [
        output["data"][AUDIT_MIME]
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if AUDIT_MIME in output.get("data", {})
        and "rounds" in output["data"][AUDIT_MIME]
    ]
    assert len(audits) == 1
    return audits[0]


def _parse_raw(raw):
    text = raw.strip()
    if text.startswith(chr(96) * 3):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    return json.loads(text)


def test_manifest_declares_three_hashed_current_sources():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files = manifest["assets"]["multisource_sources"]["files"]
    assert {item["source_id"] for item in files} == DIRECT_SOURCE_IDS
    assert manifest["assets"]["multisource_sources"]["required_files"] == [
        "data/dataset/manifest.json",
        "2. 数据处理/README.md",
        "7. 评估/README.md",
    ]
    for item in files:
        path = ROOT / item["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]


def test_notebook_has_real_model_contract_and_parsable_code():
    notebook = _load_notebook()
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    assert 'MODEL_NAME = "glm-4-flash"' in source
    assert "load_zhipuai_api_key()" in source
    assert "ZhipuAI(api_key=api_key, max_retries=0)" in source
    assert "fallback" not in source.lower()
    assert "ensure_conflict_claims" not in source
    assert "normalize_insufficient_claims" not in source
    assert "auto_bound_conflict_claims" not in source
    assert "normalized_insufficient_claims" not in source
    assert set(re.findall(r'"source_id": "([^"]+)"', source)) >= DIRECT_SOURCE_IDS


def test_in_scope_index_followup_does_not_clear_history():
    notebook = _load_notebook()
    conversation_source = "".join(notebook["cells"][3]["source"])
    namespace = {"dataclass": dataclass, "field": field, "re": re}
    exec(conversation_source, namespace)
    state = namespace["ConversationState"]("index-followup")
    state.topic = "c7_training_evaluation"
    state.history = [
        {
            "turn_id": 1,
            "focus_entity": "南瓜书",
            "focus_candidates": ["南瓜书"],
        }
    ]
    decision = namespace["resolve_followup"](
        "那 CCH 索引增强在 general regression guard 上的 Recall/MRR 是什么？",
        state,
    )
    assert decision["action"] == "inherit"
    assert len(state.history) == 1


def test_generation_prompt_questions_and_calls_have_no_answer_oracle():
    notebook = _load_notebook()
    code_source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    tree = ast.parse(code_source)
    answer_function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "answer_with_evidence"
    )
    answer_source = ast.get_source_segment(code_source, answer_function)
    assert answer_source is not None
    assert not any(marker in answer_source for marker in GENERATION_ORACLE_MARKERS)
    assert [arg.arg for arg in answer_function.args.args] == [
        "question",
        "rewritten_query",
        "retrieved",
    ]

    questions_assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "questions"
            for target in node.targets
        )
    )
    questions = ast.literal_eval(questions_assignment.value)
    assert isinstance(questions, list) and len(questions) == 3
    assert all(isinstance(question, str) for question in questions)
    assert not any(
        marker in question
        for question in questions
        for marker in GENERATION_ORACLE_MARKERS
    )
    assert all("必须" not in question and "expected_" not in question for question in questions)

    ask_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "ask"
    ]
    assert ask_calls
    for call in ask_calls:
        assert not call.keywords
        assert not any(
            keyword in ast.unparse(call)
            for keyword in ("expected_status", "expected_entity", "required_source_ids")
        )


def test_answer_call_has_no_expected_validation_arguments():
    notebook = _load_notebook()
    code_source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    tree = ast.parse(code_source)
    answer_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "answer_with_evidence"
    ]
    assert len(answer_calls) == 1
    call = answer_calls[0]
    assert not call.keywords
    assert len(call.args) == 3
    assert not any(
        marker in ast.unparse(call) for marker in GENERATION_ORACLE_MARKERS
    )


def test_claim_hydration_only_adds_source_text_to_existing_claims():
    notebook = _load_notebook()
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    tree = ast.parse(source)
    hydrator = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "hydrate_claim_quotes"
    )
    subscript_assignments = []
    mutating_calls = []
    deletes = []
    for node in ast.walk(hydrator):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript):
                    key = target.slice
                    if isinstance(key, ast.Constant):
                        subscript_assignments.append(key.value)
        elif isinstance(node, ast.Delete):
            deletes.append(node)
        elif isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Attribute) and function.attr in {
                "append",
                "clear",
                "extend",
                "insert",
                "pop",
                "remove",
            }:
                mutating_calls.append(function.attr)
    assert subscript_assignments == ["statement", "quote"]
    assert not deletes
    assert not mutating_calls


def test_answer_validator_rejects_instead_of_rewriting_model_answer():
    notebook = _load_notebook()
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    tree = ast.parse(source)
    validator = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_model_answer"
    )
    for node in ast.walk(validator):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            assert not (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "value"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "answer"
            )


def test_real_round_does_not_force_conflict_and_fixture_is_memory_only():
    notebook = _load_notebook()
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "detect_conflict" not in source
    assert "conflict_signals" not in source
    assert "simulated_conflict_rows" in source
    assert '"path": "<memory>"' in source
    assert "simulated_conflict_validation" in source
    assert "simulated_conflict" in source


def test_saved_audit_covers_routing_citations_noncomparable_scopes_insufficiency_and_boundaries():
    audit = _saved_audit()
    assert audit["model"] == "glm-4-flash"
    assert audit["api_key_source"] == ".env:ZHIPUAI_API_KEY"
    assert audit["client_max_retries"] == 0
    assert set(audit["source_catalog"]) == DIRECT_SOURCE_IDS
    assert len(audit["raw_outputs"]) == 4
    assert all(isinstance(raw, str) and raw.strip() for raw in audit["raw_outputs"])

    rounds = audit["rounds"]
    assert [round_["status"] for round_ in rounds] == [
        "answered",
        "answered",
        "answered",
    ]
    assert rounds[1]["rewrite_action"] == "inherit"
    assert rounds[1]["history_used_turn"] == rounds[0]["turn_id"]
    first_expected_sources = {
        "canonical_dataset_manifest",
        "data_processing_readme",
    }
    first_raw = _parse_raw(rounds[0]["raw"])
    assert {
        claim["source_id"] for claim in first_raw["claims"]
    } == first_expected_sources
    assert {
        claim["source_id"] for claim in rounds[0]["parsed"]["claims"]
    } == first_expected_sources
    assert set(rounds[2]["cited_source_ids"]) == {
        "data_processing_readme",
        "evaluation_readme",
    }
    assert set(rounds[2]["source_ids"]) == {
        "data_processing_readme",
        "evaluation_readme",
    }
    second_answer = rounds[1]["answer"]
    assert all(
        marker in second_answer
        for marker in ("dev", "选择", "frozen test")
    )
    second_raw = _parse_raw(rounds[1]["raw"])
    assert any("dev" in claim["statement"] for claim in rounds[1]["parsed"]["claims"])
    assert any(
        "frozen test" in claim["statement"]
        for claim in rounds[1]["parsed"]["claims"]
    )
    assert all(
        claim["source_id"] == "data_processing_readme"
        for claim in second_raw["claims"]
    )
    for round_ in rounds:
        raw = _parse_raw(round_["raw"])
        assert set(raw) == {"focus_entity", "entities", "status", "claims"}
        assert set(round_["parsed"]) == set(raw) | {"answer"}
        for field in set(raw) - {"claims"}:
            assert round_["parsed"][field] == raw[field]
        assert len(raw["claims"]) == len(round_["parsed"]["claims"])
        for raw_claim, parsed_claim in zip(
            raw["claims"], round_["parsed"]["claims"]
        ):
            assert parsed_claim["source_id"] == raw_claim["source_id"]
            assert parsed_claim["evidence_id"] == raw_claim["evidence_id"]
            assert set(raw_claim) == {"source_id", "evidence_id"}
            assert set(parsed_claim) == set(raw_claim) | {"statement", "quote"}
            assert re.sub(r"\s+", "", parsed_claim["statement"]) == re.sub(
                r"\s+", "", parsed_claim["quote"]
            )
        by_id = {
            evidence["evidence_id"]: evidence
            for evidence in round_["retrieved"]
        }
        for claim in round_["claims"]:
            assert claim["source_id"] in round_["source_ids"]
            assert claim["evidence_id"] in by_id
            assert claim["quote"] == by_id[claim["evidence_id"]]["quote"]
    claims = {
        (claim["source_id"], claim["evidence_id"])
        for claim in rounds[2]["claims"]
    }
    assert claims >= {
        ("data_processing_readme", "data_processing_readme:L61"),
        ("evaluation_readme", "evaluation_readme:L65"),
        ("evaluation_readme", "evaluation_readme:L67"),
    }
    third_raw = _parse_raw(rounds[2]["raw"])
    third_text = json.dumps(rounds[2]["parsed"], ensure_ascii=False)
    assert all(
        metric in third_text
        for metric in ("Recall@10", "Recall@5", "Recall@1", "MRR")
    )
    assert "提升" in third_text
    assert re.search(r"Recall@3.{0,12}(?:保持|不变)", third_text)
    assert "0.7941" in third_text
    assert "逐题" in third_text
    assert "均有所提升" not in third_text
    assert not re.search(r"dev.{0,8}(?:评估|比较).*Recall", third_text)
    assert {
        claim["source_id"] for claim in third_raw["claims"]
    } == {"data_processing_readme", "evaluation_readme"}
    third_claim_ids = {
        (claim["source_id"], claim["evidence_id"])
        for claim in third_raw["claims"]
    }
    assert (
        "data_processing_readme",
        "data_processing_readme:L61",
    ) in third_claim_ids
    assert (
        "evaluation_readme",
        "evaluation_readme:L65",
    ) in third_claim_ids
    assert (
        "evaluation_readme",
        "evaluation_readme:L67",
    ) in third_claim_ids
    assert {
        claim["source_id"] for claim in rounds[2]["parsed"]["claims"]
    } == {"data_processing_readme", "evaluation_readme"}
    assert "BGE" in third_text or "微调" in third_text
    assert "frozen test" in third_text
    assert "CCH" in third_text
    assert "34" in third_text
    assert "4 道题" in third_text and "2 道题" in third_text
    assert "不能合并成一个整体收益数字" in third_text

    processing_claim = next(
        claim
        for claim in rounds[2]["parsed"]["claims"]
        if claim["evidence_id"] == "data_processing_readme:L61"
    )
    assert "Recall@3 保持 0.7941" in processing_claim["quote"]
    assert "Recall@10" in processing_claim["quote"]
    evaluation_claim = next(
        claim
        for claim in rounds[2]["parsed"]["claims"]
        if claim["evidence_id"] == "evaluation_readme:L65"
    )
    assert "不能表述成全面提升" in evaluation_claim["quote"]

    missing = audit["missing_information"]
    missing_raw = _parse_raw(missing["raw"])
    assert set(missing_raw) == {"focus_entity", "entities", "status", "claims"}
    assert missing_raw["status"] == "insufficient"
    assert missing_raw["claims"] == []
    assert missing["parsed"]["answer"] == "资料不足，无法可靠回答。"
    assert missing["parsed"]["claims"] == missing_raw["claims"]
    assert missing["audit"] == {
        "status": "insufficient",
        "claims": [],
        "no_fabricated_evidence": True,
    }
    runtime_probe = audit["runtime_state_probe"]
    runtime_first = runtime_probe["first"]
    assert len(runtime_first["focus_candidates"]) >= 2
    assert set(runtime_first["focus_candidates"]) >= {"南瓜书", "C7"}
    assert runtime_probe["ambiguous_followup"]["status"] == "clarify"
    assert runtime_probe["legitimate_index_followup"]["action"] != "new_topic"
    assert runtime_probe["ambiguous_followup"]["raw"] is None
    assert runtime_probe["entity_switch"]["action"] == "explicit_entity"
    assert "南瓜书" not in runtime_probe["entity_switch"]["query"]
    assert "C7" in runtime_probe["entity_switch"]["query"]
    assert runtime_probe["history_writeback"]
    assert all(
        {"entities", "focus_candidates"} <= set(item)
        for item in runtime_probe["history_writeback"]
    )
    topic_switch = audit["topic_switch"]
    assert topic_switch["result"]["status"] == "new_topic"
    assert topic_switch["result"]["history_size_after"] == 0
    assert topic_switch["followup"]["status"] == "clarify"
    assert topic_switch["followup"]["raw"] is None
    assert topic_switch["followup"]["history_size_after"] == 0
    assert audit["history_writeback"]
    assert all(
        {"entities", "focus_candidates"} <= set(item)
        for item in audit["history_writeback"]
    )
    assert all(audit["boundary_checks"].values())


def test_removed_history_and_supplementary_material_are_absent():
    paths = [
        ROOT / "README.md",
        ROOT / "6. 处理信息缺口/README.md",
        ROOT / "data/README.md",
        ROOT / "data/asset_manifest.json",
        NOTEBOOK,
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert not any(marker in text for marker in REMOVED_HISTORY_MARKERS)
    supplementary = ROOT / "data/supplementary"
    assert not supplementary.exists() or not any(supplementary.iterdir())
