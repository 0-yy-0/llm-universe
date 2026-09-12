import hashlib
import importlib.util
import json
import sqlite3
from types import SimpleNamespace
from pathlib import Path


ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location("tutorial_checker", ROOT / "notebook/C7 高级 RAG 技巧/scripts/check_tutorial.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def _fixture(tmp_path, *, pdf=b"pdf", count=987, dimension=512):
    (tmp_path / "data/store/index").mkdir(parents=True)
    (tmp_path / "data/pumpkin_book.pdf").write_bytes(pdf)
    db = tmp_path / "data/store/chroma.sqlite3"
    conn = sqlite3.connect(db)
    conn.executescript("CREATE TABLE collections (id TEXT, name TEXT, dimension INTEGER);"
                       "CREATE TABLE segments (id TEXT, collection TEXT);"
                       "CREATE TABLE embeddings (id INTEGER, segment_id TEXT);")
    conn.execute("INSERT INTO collections VALUES ('c','nb_ctx',?)", (dimension,))
    conn.execute("INSERT INTO segments VALUES ('s','c')")
    conn.executemany("INSERT INTO embeddings VALUES (?, 's')", [(i,) for i in range(count)])
    conn.commit(); conn.close()
    manifest = {"schema_version": 1, "assets": {
        "pdf": {"path": "data/pumpkin_book.pdf", "sha256": hashlib.sha256(pdf).hexdigest(),
                "source": "fixture", "license": "fixture"},
        "vector_store": {"path": "data/store", "required_files": ["chroma.sqlite3", "index/file.bin"],
                          "collection_name": "nb_ctx", "dimension": dimension, "record_count": count,
                          "model": "BAAI/bge-small-zh-v1.5", "normalized_embeddings": True,
                          "chunk": {"strategy": "RecursiveCharacterTextSplitter", "size": 256, "overlap": 20},
                          "source": "fixture PDF", "license": "fixture license",
                          "verified_facts": ["fixture"]}}}
    (tmp_path / "data/asset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "data/store/index/file.bin").write_bytes(b"x")


def _multisource_assets(tmp_path):
    records = [
        ("data/dataset/manifest.json", "canonical_dataset_manifest", "{}\n"),
        ("2. 数据处理/README.md", "data_processing_readme", "# data processing\n"),
        ("7. 评估/README.md", "evaluation_readme", "# evaluation\n"),
    ]
    files = []
    for path_value, source_id, content in records:
        path = tmp_path / path_value
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        files.append(
            {
                "path": path_value,
                "source_id": source_id,
                "source_type": "fixture",
                "purpose": "fixture",
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        )
    return {
        "required_files": [path_value for path_value, _, _ in records],
        "source_of_truth": "current_repository_files",
        "source_scope": "fixture",
        "files": files,
    }


def test_multisource_manifest_paths_ids_and_hashes_pass(tmp_path):
    record = _multisource_assets(tmp_path)
    issues = []
    checker._check_multisource_sources(record, issues, tmp_path)
    assert issues == []


def test_multisource_manifest_hash_and_source_id_mismatch_fail(tmp_path):
    record = _multisource_assets(tmp_path)
    record["files"][0]["sha256"] = "0" * 64
    record["files"][1]["source_id"] = "wrong_source"
    issues = []
    checker._check_multisource_sources(record, issues, tmp_path)
    assert any("SHA-256" in issue for issue in issues)
    assert any("source_id" in issue for issue in issues)


def test_manifest_and_sqlite_fixture_pass(tmp_path):
    _fixture(tmp_path)
    issues = []
    checker._check_manifest(issues, tmp_path / "data/asset_manifest.json", tmp_path)
    assert issues == []


def test_manifest_hash_and_missing_file_fail(tmp_path):
    _fixture(tmp_path)
    manifest = json.loads((tmp_path / "data/asset_manifest.json").read_text())
    manifest["assets"]["pdf"]["sha256"] = "0" * 64
    (tmp_path / "data/store/index/file.bin").unlink()
    (tmp_path / "data/asset_manifest.json").write_text(json.dumps(manifest))
    issues = []
    checker._check_manifest(issues, tmp_path / "data/asset_manifest.json", tmp_path)
    assert any("SHA-256" in issue for issue in issues)
    assert any("关键文件" in issue for issue in issues)


def test_sqlite_metadata_mismatch_fails(tmp_path):
    _fixture(tmp_path)
    manifest = json.loads((tmp_path / "data/asset_manifest.json").read_text())
    manifest["assets"]["vector_store"]["record_count"] = 986
    manifest["assets"]["vector_store"]["dimension"] = 511
    (tmp_path / "data/asset_manifest.json").write_text(json.dumps(manifest))
    issues = []
    checker._check_manifest(issues, tmp_path / "data/asset_manifest.json", tmp_path)
    assert any("维度" in issue for issue in issues)
    assert any("记录数" in issue for issue in issues)


def test_vector_recipe_mismatch_fails(tmp_path):
    _fixture(tmp_path)
    manifest = json.loads((tmp_path / "data/asset_manifest.json").read_text())
    manifest["assets"]["vector_store"]["chunk"]["size"] = 999
    (tmp_path / "data/asset_manifest.json").write_text(json.dumps(manifest))
    issues = []
    checker._check_manifest(issues, tmp_path / "data/asset_manifest.json", tmp_path)
    assert any("chunk" in issue for issue in issues)


def test_fenced_markdown_links_are_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "COURSE_ROOT", tmp_path)
    md = tmp_path / "guide.md"
    md.write_text("````\n[example](missing.txt)\n````\n", encoding="utf-8")
    issues = []
    checker._check_links(md, issues, md.read_text(), 1)
    assert issues == []


def test_contextual_provenance_mismatch_fails(tmp_path, monkeypatch):
    path = tmp_path / "3. 索引阶段/给片段补充所属上下文.ipynb"
    path.parent.mkdir(parents=True)
    audit = {"model_outputs": {"candidate_chunks": ["p1"], "validated_contexts": {"p2": "bad"}},
             "provenance": {"status": "saved", "index_scope": "fixture",
                            "candidate_selection": "uses expected_pages", "raw_output_coverage": "partial"}}
    notebook = {"cells": [{"cell_type": "code", "outputs": [{"output_type": "display_data",
        "data": {checker.TUTORIAL_AUDIT_MIME: audit}}]}]}
    path.write_text(json.dumps(notebook), encoding="utf-8")
    monkeypatch.setattr(checker, "COURSE_ROOT", tmp_path)
    issues = []
    checker._check_contextual_provenance(issues)
    assert any("candidate_selection" in issue or "validated_contexts" in issue for issue in issues)


def _benchmark_report(marker="current"):
    return {
        "schema_version": 1,
        "benchmark": "c7_same_question_retrieval",
        "config": {"marker": marker},
        "questions": {},
        "summary": {},
        "validation": {"fallback_used": False},
    }


def _write_saved_benchmark_notebook(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    output = "benchmark contract passed\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": 1,
                        "outputs": [{"output_type": "stream", "text": output}],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_saved_benchmark_report_parser_reads_structured_json(tmp_path):
    path = tmp_path / "compare.ipynb"
    report = _benchmark_report()
    _write_saved_benchmark_notebook(path, report)

    assert checker._saved_benchmark_reports(path) == [report]


def test_benchmark_consistency_checker_uses_current_local_run(tmp_path, monkeypatch):
    report = _benchmark_report()
    path = tmp_path / "7. 评估/比较改动前后.ipynb"
    _write_saved_benchmark_notebook(path, report)
    module = SimpleNamespace(
        run_benchmark=lambda: report,
        validate_report=lambda value: None,
    )
    monkeypatch.setattr(checker, "_load_local_benchmark", lambda root: module)

    issues = []
    checker._check_benchmark_notebook_consistency(issues, tmp_path)
    assert issues == []

    stale = _benchmark_report("stale")
    _write_saved_benchmark_notebook(path, stale)
    issues = []
    checker._check_benchmark_notebook_consistency(issues, tmp_path)
    assert any("run_benchmark()" in issue and "不一致" in issue for issue in issues)
