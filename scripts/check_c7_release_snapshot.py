#!/usr/bin/env python3
"""Read-only snapshot of the files intended for the C7 tutorial release.

The command does not stage, delete, or modify anything.  It reports the
release whitelist, checks the canonical/vector bundles named by the C7 asset
manifest, and calls out local-workspace debris such as ``.DS_Store``,
``.codex*``, ``.luna*`` and temporary/cache directories.  The default check
only fails for missing required release files or forbidden files *inside* the
release scope; ``--all`` also makes ambient forbidden paths a failure.

Examples::

    python scripts/check_c7_release_snapshot.py
    python scripts/check_c7_release_snapshot.py --json
    python scripts/check_c7_release_snapshot.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
C7_ROOT = Path("notebook/C7 高级 RAG 技巧")
ASSET_MANIFEST = C7_ROOT / "data" / "asset_manifest.json"

# These files are the human-facing entry points and the executable checks
# that must be present even when the broader C7/tutorial scopes are listed.
REQUIRED_FILES = (
    Path("README.md"),
    Path("docs/README.md"),
    Path("scripts/check_c7_release_snapshot.py"),
    C7_ROOT / "README.md",
    C7_ROOT / "docs/维护说明.md",
    C7_ROOT / "data/asset_manifest.json",
    C7_ROOT / "data/pumpkin_book.pdf",
    C7_ROOT / "data/dataset/manifest.json",
    C7_ROOT / "scripts/check_tutorial.py",
    C7_ROOT / "scripts/run_benchmark.py",
    C7_ROOT / "7. 评估/README.md",
    C7_ROOT / "7. 评估/比较改动前后.ipynb",
    C7_ROOT / "7. 评估/端到端验收.ipynb",
    C7_ROOT / "7. 评估/端到端验收协议.md",
    Path("tests/c7/test_tutorial_checker.py"),
    Path("tests/c7/test_benchmark_contracts.py"),
)

# A release scope deliberately includes the complete C7 tutorial and C7
# tests.  The explicit list above makes the important entry points visible
# in the report; the scopes prevent a partial directory from looking complete.
RELEASE_SCOPE_PREFIXES = (
    C7_ROOT.as_posix() + "/",
    "tests/c7/",
)
RELEASE_SCOPE_EXACT = {path.as_posix() for path in REQUIRED_FILES}


def _is_forbidden_component(component: str) -> bool:
    """Return whether a path component is local debris, not release input."""

    lower = component.lower()
    if lower in {
        ".ds_store",
        ".codex",
        ".codex_tmp",
        ".codex-transfer",
        ".luna",
        ".pytest_cache",
        ".cache",
        "__pycache__",
        ".ipynb_checkpoints",
        "tmp",
        "temp",
    }:
        return True
    return lower.startswith((".codex", ".luna", ".tmp", ".test_tmp", ".pytest")) or lower.endswith(
        (".pyc", ".pyo")
    )


def _is_release_path(relative: Path) -> bool:
    value = relative.as_posix()
    return value in RELEASE_SCOPE_EXACT or any(
        value.startswith(prefix) for prefix in RELEASE_SCOPE_PREFIXES
    )


def _git_candidates(root: Path, pathspecs: list[str] | None = None) -> set[Path]:
    """Collect tracked and non-ignored untracked paths without changing Git."""

    try:
        command = [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ]
        if pathspecs:
            command.extend(["--", *pathspecs])
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {
        Path(raw.decode("utf-8"))
        for raw in completed.stdout.split(b"\0")
        if raw
    }


def _git_index_candidates(root: Path, pathspecs: list[str] | None = None) -> set[Path]:
    """Collect paths present in the Git index (tracked or staged).

    A file that merely exists in the working tree is deliberately not an
    index candidate.  Release checks use this set for required assets so a
    developer cannot obtain PASS from an untracked local PDF/vector bundle.
    """

    try:
        command = [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--cached",
            "-z",
        ]
        if pathspecs:
            command.extend(["--", *pathspecs])
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {
        Path(raw.decode("utf-8"))
        for raw in completed.stdout.split(b"\0")
        if raw
    }


def _git_unstaged_candidates(root: Path, pathspecs: list[str] | None = None) -> set[Path]:
    """Collect release paths whose working-tree content differs from the index."""

    try:
        command = ["git", "-C", str(root), "diff", "--name-only", "-z"]
        if pathspecs:
            command.extend(["--", *pathspecs])
        completed = subprocess.run(command, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {
        Path(raw.decode("utf-8"))
        for raw in completed.stdout.split(b"\0")
        if raw
    }


def _filesystem_forbidden(root: Path) -> set[Path]:
    """Find forbidden names even when Git ignores them (for example caches)."""

    found: set[Path] = set()
    for current, directories, files in os.walk(root):
        current_path = Path(current)
        relative_current = current_path.relative_to(root)
        if relative_current != Path(".") and any(
            _is_forbidden_component(part) for part in relative_current.parts
        ):
            found.add(relative_current)
            directories[:] = []
            continue
        kept_directories: list[str] = []
        for name in directories:
            if name in {".git", ".worktrees", "node_modules", ".venv", "venv"}:
                continue
            relative = (current_path / name).relative_to(root)
            if _is_forbidden_component(name):
                found.add(relative)
            else:
                kept_directories.append(name)
        directories[:] = kept_directories
        for name in files:
            relative = (current_path / name).relative_to(root)
            if _is_forbidden_component(name):
                found.add(relative)
    return found


def _collapse_paths(paths: set[Path] | list[Path]) -> list[str]:
    """Keep only the shortest path when a forbidden directory has children."""

    ordered = sorted(set(paths), key=lambda path: (len(path.parts), path.as_posix()))
    kept: list[Path] = []
    for path in ordered:
        if any(parent == path or parent in path.parents for parent in kept):
            continue
        kept.append(path)
    return [path.as_posix() for path in sorted(kept, key=lambda value: value.as_posix())]


def _manifest_vector_files(root: Path) -> tuple[list[Path], str | None]:
    """Read the manifest-declared vector bundle paths, without guessing names."""

    try:
        manifest = json.loads((root / ASSET_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return [], f"无法读取 {ASSET_MANIFEST}: {error}"
    try:
        vector = manifest["assets"]["vector_store"]
        directory = Path(vector["path"])
        required = vector["required_files"]
        if (
            directory.is_absolute()
            or ".." in directory.parts
            or not isinstance(required, list)
            or not required
        ):
            raise ValueError("vector_store.path/required_files 无效")
        names = [Path(name) for name in required if isinstance(name, str)]
        if len(names) != len(required) or any(
            name.is_absolute() or ".." in name.parts for name in names
        ):
            raise ValueError("vector_store.required_files 必须是相对路径")
        paths = [C7_ROOT / directory / name for name in names]
    except (KeyError, TypeError, ValueError) as error:
        return [], f"asset_manifest 的 vector_store 无效：{error}"
    return paths, None


def _manifest_required_files(root: Path) -> tuple[list[Path], str | None]:
    """Read all non-vector release paths declared by the C7 asset manifest.

    Manifest paths are relative to the C7 chapter root.  The canonical dataset
    entries are names relative to ``canonical_dataset.path``; the multisource
    and complete-experiment entries are already C7-root-relative paths.  Keep
    this parsing strict so a malformed manifest cannot make the release check
    silently skip required files.
    """

    def relative_path(value: Any, field: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} 必须是非空字符串")
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{field} 必须是 C7 根目录下的相对路径")
        return path

    def relative_list(value: Any, field: str) -> list[Path]:
        if not isinstance(value, list) or not value:
            raise ValueError(f"{field} 必须是非空列表")
        paths = [relative_path(item, field) for item in value]
        if len(set(paths)) != len(paths):
            raise ValueError(f"{field} 不能包含重复路径")
        return paths

    try:
        manifest = json.loads((root / ASSET_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return [], f"无法读取 {ASSET_MANIFEST}: {error}"
    try:
        assets = manifest["assets"]
        canonical = assets["canonical_dataset"]
        dataset_directory = relative_path(
            canonical["path"], "canonical_dataset.path"
        )
        canonical_files = relative_list(
            canonical["required_files"], "canonical_dataset.required_files"
        )
        multisource_files = relative_list(
            assets["multisource_sources"]["required_files"],
            "multisource_sources.required_files",
        )
        experiment_notebooks = relative_list(
            assets["complete_experiments"]["required_notebooks"],
            "complete_experiments.required_notebooks",
        )
    except (KeyError, TypeError, ValueError) as error:
        return [], f"asset_manifest 的 required assets 无效：{error}"

    paths = [C7_ROOT / dataset_directory / name for name in canonical_files]
    paths.extend(C7_ROOT / path for path in multisource_files)
    paths.extend(C7_ROOT / path for path in experiment_notebooks)
    # A path can be named by more than one manifest section (for example the
    # canonical manifest is also a multisource source).  Preserve declaration
    # order while checking each physical file only once.
    return list(dict.fromkeys(paths)), None


def build_snapshot(root: Path = REPO_ROOT, *, fail_on_ambient: bool = False) -> dict[str, Any]:
    """Build a JSON-serializable, read-only release snapshot."""

    root = root.resolve()
    manifest_required_paths, manifest_required_error = _manifest_required_files(root)
    required_files = list(dict.fromkeys([*REQUIRED_FILES, *manifest_required_paths]))
    missing = [path.as_posix() for path in required_files if not (root / path).is_file()]
    vector_paths, vector_error = _manifest_vector_files(root)
    vector_missing = [path.as_posix() for path in vector_paths if not (root / path).is_file()]

    pathspecs = [
        C7_ROOT.as_posix(),
        "tests/c7",
        "README.md",
        "docs/README.md",
        "scripts/check_c7_release_snapshot.py",
    ]
    candidates = _git_candidates(root, pathspecs)
    index_candidates = _git_index_candidates(root, pathspecs)
    unstaged_candidates = _git_unstaged_candidates(root, pathspecs)
    required_untracked = [
        path.as_posix()
        for path in required_files
        if (root / path).is_file() and path not in index_candidates
    ]
    vector_untracked = [
        path.as_posix()
        for path in vector_paths
        if (root / path).is_file() and path not in index_candidates
    ]
    index_release_candidates = sorted(
        path.as_posix() for path in index_candidates if _is_release_path(path)
    )
    working_tree_release_candidates = sorted(
        path.as_posix()
        for path in candidates - index_candidates
        if _is_release_path(path)
    )
    unstaged_release_changes = sorted(
        path.as_posix() for path in unstaged_candidates if _is_release_path(path)
    )
    filesystem_forbidden = _filesystem_forbidden(root)
    candidate_forbidden = {path for path in candidates if any(
        _is_forbidden_component(part) for part in path.parts
    )}
    forbidden_in_scope = _collapse_paths(
        {path for path in candidate_forbidden if _is_release_path(path)}
    )
    ignored_forbidden_in_scope = _collapse_paths(
        {
            path
            for path in filesystem_forbidden
            if _is_release_path(path) and path not in candidate_forbidden
        }
    )
    ambient_forbidden = _collapse_paths(
        {path for path in filesystem_forbidden if not _is_release_path(path)}
    )
    required_present = [
        path.as_posix() for path in required_files if (root / path).is_file()
    ]
    passed = (
        not missing
        and manifest_required_error is None
        and not required_untracked
        and not vector_missing
        and not vector_untracked
        and vector_error is None
        and not forbidden_in_scope
        and not working_tree_release_candidates
        and not unstaged_release_changes
    )
    if fail_on_ambient and ambient_forbidden:
        passed = False
    return {
        "passed": passed,
        "repository": root.as_posix(),
        "release_whitelist": {
            "exact_required_files": [path.as_posix() for path in required_files],
            "manifest_required_files": [
                path.as_posix() for path in manifest_required_paths
            ],
            "manifest_error": manifest_required_error,
            "directory_scopes": [f"{prefix}**" for prefix in RELEASE_SCOPE_PREFIXES],
            "required_present": required_present,
            "missing": missing,
            "untracked_required": required_untracked,
            "index_candidates": index_release_candidates,
            "release_candidates": index_release_candidates,
            "working_tree_candidates": working_tree_release_candidates,
            "unstaged_release_changes": unstaged_release_changes,
        },
        "vector_bundle": {
            "manifest": ASSET_MANIFEST.as_posix(),
            "required_files": [path.as_posix() for path in vector_paths],
            "missing": vector_missing,
            "untracked": vector_untracked,
            "error": vector_error,
        },
        "forbidden": {
            "in_release_scope": forbidden_in_scope,
            "ignored_in_release_scope": ignored_forbidden_in_scope,
            "ambient_not_in_release_scope": ambient_forbidden,
            "policy": [
                ".DS_Store, .codex*, .luna*, .tmp*/tmp/temp, .cache",
                ".pytest_cache, __pycache__, .ipynb_checkpoints, *.pyc/*.pyo",
            ],
        },
        "mode": "all" if fail_on_ambient else "release_scope",
    }


def _print_human(snapshot: dict[str, Any]) -> None:
    status = "PASS" if snapshot["passed"] else "FAIL"
    print(f"C7 release snapshot: {status} ({snapshot['mode']})")
    release = snapshot["release_whitelist"]
    vector = snapshot["vector_bundle"]
    forbidden = snapshot["forbidden"]
    print(f"required files: {len(release['required_present'])}/{len(release['exact_required_files'])}")
    if release.get("untracked_required"):
        print(f"required files not in Git index: {len(release['untracked_required'])}")
        for path in release["untracked_required"]:
            print(f"  - {path}")
    if release.get("working_tree_candidates"):
        print("untracked release files:")
        for path in release["working_tree_candidates"]:
            print(f"  - {path}")
    if release.get("unstaged_release_changes"):
        print("working-tree release changes not synchronized to the index:")
        for path in release["unstaged_release_changes"]:
            print(f"  - {path}")
    if release.get("manifest_error"):
        print(f"manifest required-assets error: {release['manifest_error']}")
    print(f"vector bundle: {len(vector['required_files']) - len(vector['missing'])}/{len(vector['required_files'])}")
    if vector.get("untracked"):
        print(f"vector files not in Git index: {len(vector['untracked'])}")
        for path in vector["untracked"]:
            print(f"  - {path}")
    if release["missing"]:
        print("missing:")
        for path in release["missing"]:
            print(f"  - {path}")
    if vector.get("error"):
        print(f"vector manifest error: {vector['error']}")
    if vector["missing"]:
        print("missing vector files:")
        for path in vector["missing"]:
            print(f"  - {path}")
    print(f"forbidden in release scope (would block release): {len(forbidden['in_release_scope'])}")
    print(f"ignored debris in release scope (do not stage): {len(forbidden['ignored_in_release_scope'])}")
    for path in forbidden["ignored_in_release_scope"]:
        print(f"  - {path}")
    print(f"ambient forbidden (do not stage): {len(forbidden['ambient_not_in_release_scope'])}")
    for path in forbidden["ambient_not_in_release_scope"]:
        print(f"  - {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="ambient forbidden paths also fail the check")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args(argv)
    snapshot = build_snapshot(fail_on_ambient=args.all)
    if args.json:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    else:
        _print_human(snapshot)
    return 0 if snapshot["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
