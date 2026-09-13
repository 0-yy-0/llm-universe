"""C7 notebook metadata and dependency contracts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import nbformat
from packaging.requirements import Requirement


ROOT = Path(__file__).parents[2]
COURSE = ROOT / "notebook" / "C7 高级 RAG 技巧"
REQUIREMENTS = COURSE / "requirements-c7.txt"
EXPECTED_NOTEBOOK_COUNT = 40
EXPECTED_KERNEL = "llm-universe-c7"
EXPECTED_PYTHON_MAJOR_MINOR = (3, 10)
CELL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _notebooks() -> list[Path]:
    return sorted(COURSE.rglob("*.ipynb"))


def _load_notebook(path: Path) -> dict:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    # Validate the serialized document without normalizing it first. The
    # explicit ID checks below must see missing/duplicate IDs as committed.
    nbformat.validate(notebook, version=4, version_minor=5)
    return notebook


def test_all_c7_notebooks_have_uniform_runtime_metadata_and_valid_ids() -> None:
    notebooks = _notebooks()
    assert len(notebooks) == EXPECTED_NOTEBOOK_COUNT

    global_ids: dict[str, tuple[Path, int]] = {}
    for path in notebooks:
        notebook = _load_notebook(path)
        metadata = notebook["metadata"]
        assert metadata["kernelspec"]["name"] == EXPECTED_KERNEL, path
        version = metadata["language_info"]["version"]
        match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?", version)
        assert match is not None, (path, version)
        assert tuple(map(int, match.groups())) == EXPECTED_PYTHON_MAJOR_MINOR, (
            path,
            version,
        )

        local_ids: set[str] = set()
        for index, cell in enumerate(notebook["cells"]):
            cell_id = cell.get("id")
            assert isinstance(cell_id, str) and cell_id, (path, index)
            assert CELL_ID_PATTERN.fullmatch(cell_id), (path, index, cell_id)
            assert cell_id not in local_ids, (path, index, cell_id)
            assert cell_id not in global_ids, (
                path,
                index,
                cell_id,
                global_ids.get(cell_id),
            )
            local_ids.add(cell_id)
            global_ids[cell_id] = (path, index)


def test_c7_requirements_pin_direct_test_and_runtime_imports() -> None:
    pins: dict[str, str] = {}
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        pins[requirement.name] = str(requirement.specifier)

    assert pins["pytest"] == "==9.0.3"
    assert pins["huggingface-hub"] == "==0.35.1"
