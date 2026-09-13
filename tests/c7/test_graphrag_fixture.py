"""Checks for the saved, deterministic GraphRAG teaching fixture."""

from __future__ import annotations

import json
import sys
from pathlib import Path

COURSE = Path(__file__).parents[2] / "notebook" / "C7 高级 RAG 技巧"
if str(COURSE) not in sys.path:
    sys.path.insert(0, str(COURSE))

from common.dataset import (
    CANONICAL_PDF_RELATIVE_PATH,
    NORMALIZATION_VERSION,
    load_evidence_records,
)
from common.eval_utils import load_pdf_pages, normalize_text


ROOT = Path(__file__).parents[2]
NOTEBOOK = (
    ROOT
    / "notebook"
    / "C7 高级 RAG 技巧"
    / "6. 处理信息缺口"
    / "让系统选择资料来源.ipynb"
)
AUDIT_MIME = "application/vnd.llm-universe.tutorial-audit+json"


def _graph_audit() -> dict:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            audit = data.get(AUDIT_MIME)
            if (
                isinstance(audit, dict)
                and audit.get("case_id") == "lda_multihop"
                and audit.get("method")
                == "受限 GraphRAG（确定性南瓜书 fixture）"
            ):
                return audit
    raise AssertionError("Notebook 没有保存 GraphRAG fixture 审计输出")


def test_graphrag_edges_bind_exact_canonical_evidence() -> None:
    audit = _graph_audit()
    evidence_by_id = {
        row["evidence_id"]: row for row in load_evidence_records()
    }
    page_text_by_number = {
        int(row["page"]): normalize_text(row["text"])
        for row in load_pdf_pages()
    }

    assert audit["fixture"]["automatic_extraction"] is False
    assert audit["fixture"]["model_called"] is False
    assert audit["fixture"]["source"] == CANONICAL_PDF_RELATIVE_PATH
    assert audit["entity_resolution"] == {
        "canonical_node": "entity:lda",
        "aliases": ["LDA", "线性判别分析"],
        "source": CANONICAL_PDF_RELATIVE_PATH,
        "version": NORMALIZATION_VERSION,
    }

    required_fields = {
        "subject",
        "relation",
        "object",
        "evidence_id",
        "source",
        "page",
        "quote",
    }
    assert audit["edges"]
    for edge in audit["edges"]:
        assert required_fields <= edge.keys()
        source = evidence_by_id[edge["evidence_id"]]
        assert edge["source"] == CANONICAL_PDF_RELATIVE_PATH
        assert edge["page"] == source["page"]
        assert edge["quote"] == source["quote"]
        offsets = source["offsets"]
        page_text = page_text_by_number[edge["page"]]
        assert page_text[offsets["start"] : offsets["end"]] == edge["quote"]


def test_graphrag_path_is_bounded_and_covers_the_second_hop() -> None:
    audit = _graph_audit()
    paths = audit["paths"]
    assert paths
    assert all(path["hops"] <= audit["max_hops"] == 2 for path in paths)
    target = next(
        path
        for path in paths
        if path["nodes"][-1] == "N−1 个最大广义特征值对应的特征向量"
    )
    assert target["hops"] == 2
    assert target["evidence_ids"] == [
        "evi_677d8872b6b8",
        "evi_94ff9315c02b",
    ]
    assert audit["baseline"]["scope_pages"] == [41, 44]
    assert audit["baseline"]["retrieved_pages"] == [41]
    assert audit["baseline"]["required_evidence_coverage"] == 0.5
    assert audit["graph"]["required_evidence_hits"] == [
        "evi_677d8872b6b8",
        "evi_94ff9315c02b",
    ]
    assert audit["graph"]["required_evidence_coverage"] == 1.0
