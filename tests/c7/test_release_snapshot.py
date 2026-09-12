"""Read-only release whitelist checks for the C7 tutorial."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts" / "check_c7_release_snapshot.py"
spec = importlib.util.spec_from_file_location("c7_release_snapshot", SCRIPT)
assert spec and spec.loader
snapshot_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot_module)


def test_current_snapshot_has_entrypoints_and_complete_vector_bundle():
    snapshot = snapshot_module.build_snapshot(ROOT)

    assert snapshot["release_whitelist"]["missing"] == []
    assert snapshot["vector_bundle"]["error"] is None
    assert snapshot["vector_bundle"]["missing"] == []
    assert snapshot["forbidden"]["in_release_scope"] == []
    assert "README.md" in snapshot["release_whitelist"]["exact_required_files"]
    assert "tests/c7/test_benchmark_contracts.py" in snapshot["release_whitelist"]["exact_required_files"]
    manifest_required = snapshot["release_whitelist"]["manifest_required_files"]
    assert "notebook/C7 高级 RAG 技巧/data/dataset/evidence.jsonl" in manifest_required
    assert "notebook/C7 高级 RAG 技巧/2. 数据处理/README.md" in manifest_required
    assert "notebook/C7 高级 RAG 技巧/6. 处理信息缺口/构建多轮多来源助手.ipynb" in manifest_required


def test_forbidden_names_are_never_treated_as_release_files():
    forbidden = [
        ".DS_Store",
        ".codex",
        ".luna_stage",
        ".tmp",
        "tmp",
        "__pycache__",
        "example.pyc",
    ]

    assert all(snapshot_module._is_forbidden_component(name) for name in forbidden)
    assert not snapshot_module._is_release_path(Path(".DS_Store"))


def test_required_and_vector_files_must_be_in_git_index(tmp_path, monkeypatch):
    """Only a valid Git-index scope is releasable, not a bare working tree."""

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    # Match the production pathspecs so this exercises a real in-scope index,
    # rather than a fixture path that the checker cannot observe.
    required = Path("README.md")
    vector = Path("notebook/C7 高级 RAG 技巧/data/store/vector.bin")
    (tmp_path / required).write_text("required", encoding="utf-8")
    (tmp_path / vector).parent.mkdir(parents=True)
    (tmp_path / vector).write_bytes(b"vector")

    monkeypatch.setattr(snapshot_module, "REQUIRED_FILES", (required,))
    monkeypatch.setattr(snapshot_module, "RELEASE_SCOPE_PREFIXES", ())
    monkeypatch.setattr(snapshot_module, "RELEASE_SCOPE_EXACT", {required.as_posix(), vector.as_posix()})
    monkeypatch.setattr(
        snapshot_module,
        "_manifest_vector_files",
        lambda root: ([vector], None),
    )
    monkeypatch.setattr(
        snapshot_module,
        "_manifest_required_files",
        lambda root: ([], None),
    )

    snapshot = snapshot_module.build_snapshot(tmp_path)

    assert snapshot["passed"] is False
    assert snapshot["release_whitelist"]["untracked_required"] == [required.as_posix()]
    assert snapshot["vector_bundle"]["untracked"] == [vector.as_posix()]

    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "--", required.as_posix(), vector.as_posix()],
        check=True,
    )
    staged = snapshot_module.build_snapshot(tmp_path)
    assert staged["passed"] is True
    assert staged["release_whitelist"]["untracked_required"] == []
    assert staged["vector_bundle"]["untracked"] == []
    assert required.as_posix() in staged["release_whitelist"]["index_candidates"]
    assert vector.as_posix() in staged["release_whitelist"]["index_candidates"]

    (tmp_path / required).write_text("changed after staging", encoding="utf-8")
    stale_index = snapshot_module.build_snapshot(tmp_path)
    assert stale_index["passed"] is False
    assert stale_index["release_whitelist"]["unstaged_release_changes"] == [
        required.as_posix()
    ]
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "--", required.as_posix()],
        check=True,
    )
    assert snapshot_module.build_snapshot(tmp_path)["passed"] is True

    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "rm",
            "--cached",
            "-q",
            "--",
            required.as_posix(),
            vector.as_posix(),
        ],
        check=True,
    )
    unstaged_again = snapshot_module.build_snapshot(tmp_path)
    assert unstaged_again["passed"] is False
    assert unstaged_again["release_whitelist"]["untracked_required"] == [required.as_posix()]
    assert unstaged_again["vector_bundle"]["untracked"] == [vector.as_posix()]


def test_manifest_required_assets_are_present_and_indexed(tmp_path, monkeypatch):
    """Manifest sections must add physical files to the release gate."""

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    c7_root = Path("notebook/C7 高级 RAG 技巧")
    manifest_path = c7_root / "data/asset_manifest.json"
    required = Path("README.md")
    manifest = {
        "assets": {
            "vector_store": {"path": "data/vector", "required_files": ["index.bin"]},
            "canonical_dataset": {
                "path": "data/dataset",
                "required_files": ["manifest.json", "evidence.jsonl"],
            },
            "multisource_sources": {
                "required_files": [
                    "data/dataset/manifest.json",
                    "2. 数据处理/README.md",
                ]
            },
            "complete_experiments": {
                "required_notebooks": [
                    "6. 处理信息缺口/构建多轮多来源助手.ipynb"
                ]
            },
        }
    }
    files = {
        required: "required",
        manifest_path: json.dumps(manifest, ensure_ascii=False),
        c7_root / "data/dataset/manifest.json": "dataset manifest",
        c7_root / "data/dataset/evidence.jsonl": "evidence",
        c7_root / "2. 数据处理/README.md": "data processing",
        c7_root / "6. 处理信息缺口/构建多轮多来源助手.ipynb": "notebook",
    }
    for path, content in files.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    monkeypatch.setattr(snapshot_module, "REQUIRED_FILES", (required, manifest_path))
    monkeypatch.setattr(snapshot_module, "RELEASE_SCOPE_PREFIXES", (c7_root.as_posix() + "/",))
    monkeypatch.setattr(
        snapshot_module,
        "RELEASE_SCOPE_EXACT",
        {required.as_posix(), manifest_path.as_posix()},
    )
    monkeypatch.setattr(snapshot_module, "_manifest_vector_files", lambda root: ([], None))

    initial = snapshot_module.build_snapshot(tmp_path)
    required_paths = set(initial["release_whitelist"]["manifest_required_files"])
    assert required_paths == {
        "notebook/C7 高级 RAG 技巧/data/dataset/manifest.json",
        "notebook/C7 高级 RAG 技巧/data/dataset/evidence.jsonl",
        "notebook/C7 高级 RAG 技巧/2. 数据处理/README.md",
        "notebook/C7 高级 RAG 技巧/6. 处理信息缺口/构建多轮多来源助手.ipynb",
    }
    assert initial["release_whitelist"]["missing"] == []
    assert initial["passed"] is False
    assert set(required_paths) <= set(initial["release_whitelist"]["untracked_required"])

    subprocess.run(["git", "-C", str(tmp_path), "add", "--", "."], check=True)
    staged = snapshot_module.build_snapshot(tmp_path)
    assert staged["passed"] is True
    assert staged["release_whitelist"]["missing"] == []
    assert staged["release_whitelist"]["untracked_required"] == []

    missing_path = c7_root / "data/dataset/evidence.jsonl"
    (tmp_path / missing_path).unlink()
    missing = snapshot_module.build_snapshot(tmp_path)
    assert missing["passed"] is False
    assert missing_path.as_posix() in missing["release_whitelist"]["missing"]

    (tmp_path / missing_path).write_text("evidence", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "--", missing_path.as_posix()], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "rm", "--cached", "-q", "--", missing_path.as_posix()],
        check=True,
    )
    not_indexed = snapshot_module.build_snapshot(tmp_path)
    assert not_indexed["passed"] is False
    assert not_indexed["release_whitelist"]["untracked_required"] == [
        missing_path.as_posix()
    ]
