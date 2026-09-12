#!/usr/bin/env python3
"""检查教程文件、问题集以及 Notebook 中已经保存的结果。

脚本只读取文件，不运行 Notebook、不调用模型，也不重新计算文档向量。
除文件和结构检查外，会在本地重跑一次不调用 API 的 BM25/CCH 核心
benchmark，并把当前 ``run_benchmark()`` 报告与比较 Notebook 保存的结构化
JSON 对齐；这样不会把过期输出误当成当前 canonical 数据的结果。
结果审计只处理 Notebook 输出中的统一审计 MIME（before/after JSON），其他格式会报告为 unable_to_judge。
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import re
import sqlite3
import sys
from pathlib import Path
from urllib.parse import unquote
COURSE_ROOT = Path(__file__).resolve().parents[1]
if str(COURSE_ROOT) not in sys.path:
    sys.path.insert(0, str(COURSE_ROOT))
from common import dataset as dataset_store
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
CASE_JSON = re.compile(r'\{\s*"case_id"\s*:\s*"([^"]+)"')
TUTORIAL_AUDIT_MIME = "application/vnd.llm-universe.tutorial-audit+json"
BENCHMARK_SCRIPT = Path("scripts/run_benchmark.py")
BENCHMARK_NOTEBOOK = Path("7. 评估/比较改动前后.ipynb")
BENCHMARK_NAME = "c7_same_question_retrieval"
LAST_RESULT_AUDIT: dict[str, object] = {}
CANONICAL_MANIFEST = Path("data/dataset/manifest.json")
MULTISOURCE_REQUIRED_FILES = (
    "data/dataset/manifest.json",
    "2. 数据处理/README.md",
    "7. 评估/README.md",
)
MULTISOURCE_SOURCE_IDS = {
    "canonical_dataset_manifest",
    "data_processing_readme",
    "evaluation_readme",
}
REQUIRED_EXPERIMENTS = (
    "2. 数据处理/什么时候需要微调向量模型.ipynb",
    "3. 索引阶段/比较索引增强方法.ipynb",
    "6. 处理信息缺口/构建多轮多来源助手.ipynb",
)
AGENTIC_NOTEBOOK = Path("6. 处理信息缺口/让系统决定怎样检索.ipynb")
AGENTIC_CASE_IDS = frozenset({"agentic_mds_ksvd", "agentic_kpca_centering"})
RESERVED_NOTEBOOKS = {
    "2. 数据处理/什么时候需要微调向量模型.ipynb",
    "3. 索引阶段/比较索引增强方法.ipynb",
    "6. 处理信息缺口/构建多轮多来源助手.ipynb",
    "7. 评估/比较改动前后.ipynb",
}
def _check_links(path: Path, issues: list[str], text: str | None = None, notebook_cell: int | None = None) -> None:
    text = path.read_text(encoding="utf-8") if text is None else text
    location = f"（Notebook Markdown 单元 {notebook_cell}）" if notebook_cell else ""
    # fenced code 中的示例命令/文字不是教程链接，避免把示例路径误报为缺失文件。
    text = re.sub(r"```[^\n]*\n.*?```", "", text, flags=re.DOTALL)
    for raw_target in MARKDOWN_LINK.findall(text):
        target = raw_target.strip().split("#", 1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        resolved = (path.parent / unquote(target)).resolve()
        try:
            resolved.relative_to(COURSE_ROOT.resolve())
        except ValueError:
            continue
        if not resolved.exists():
            issues.append(f"链接目标不存在：{path.relative_to(COURSE_ROOT)}{location} -> {target}")


MANIFEST_PATH = COURSE_ROOT / "data" / "asset_manifest.json"


def _check_manifest(issues: list[str], manifest_path: Path = MANIFEST_PATH, root: Path = COURSE_ROOT) -> None:
    """检查资产清单及可用的 SQLite 元数据，不依赖 Chroma。"""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        issues.append(f"资产清单无法读取：{error}")
        return
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        issues.append("资产清单 schema_version 必须为 1")
        return
    assets = manifest.get("assets")
    if not isinstance(assets, dict):
        issues.append("资产清单缺少 assets 对象")
        return
    pdf = assets.get("pdf")
    if not isinstance(pdf, dict):
        issues.append("资产清单缺少 pdf 记录")
    else:
        _check_hashed_asset(pdf, "pdf", issues, root)
    vector = assets.get("vector_store")
    if not isinstance(vector, dict):
        issues.append("资产清单缺少 vector_store 记录")
        return
    path_value = vector.get("path")
    if not isinstance(path_value, str) or not path_value:
        issues.append("vector_store.path 必须是非空相对路径")
        return
    vector_path = (root / path_value).resolve()
    try:
        vector_path.relative_to(root.resolve())
    except ValueError:
        issues.append("vector_store.path 必须位于教程根目录内")
        return
    required_files = vector.get("required_files")
    if not isinstance(required_files, list) or not required_files:
        issues.append("vector_store.required_files 必须是非空列表")
    else:
        for name in required_files:
            if not isinstance(name, str) or not (vector_path / name).is_file():
                issues.append(f"向量库缺少关键文件：{path_value}/{name}")
    for key, expected in (("collection_name", "nb_ctx"), ("dimension", 512), ("record_count", 987)):
        if vector.get(key) != expected:
            issues.append(f"向量库清单 {key} 不符：应为 {expected!r}")
    for key, expected in (("model", "BAAI/bge-small-zh-v1.5"), ("normalized_embeddings", True)):
        if vector.get(key) != expected:
            issues.append(f"向量库清单 {key} 不符：应为 {expected!r}")
    chunk = vector.get("chunk")
    expected_chunk = {"strategy": "RecursiveCharacterTextSplitter", "size": 256, "overlap": 20}
    if chunk != expected_chunk:
        issues.append(f"向量库清单 chunk 不符：应为 {expected_chunk!r}")
    for key in ("source", "license"):
        if not isinstance(vector.get(key), str) or not vector[key].strip():
            issues.append(f"向量库清单缺少 {key} 说明")
    if not isinstance(vector.get("verified_facts"), list) or not vector["verified_facts"]:
        issues.append("向量库必须列出 verified_facts")
    sqlite_path = vector_path / "chroma.sqlite3"
    if sqlite_path.is_file():
        _check_vector_sqlite(sqlite_path, vector, issues)
    if manifest.get("source_of_truth") == "canonical_dataset":
        _check_declared_project_assets(assets, issues, root)


def _check_hashed_asset(record: dict, label: str, issues: list[str], root: Path) -> None:
    path_value, digest = record.get("path"), record.get("sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        issues.append(f"{label} 必须包含相对 path 和 64 位小写 sha256")
        return
    path = (root / path_value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        issues.append(f"{label}.path 必须位于教程根目录内")
        return
    if not path.is_file():
        issues.append(f"资产文件不存在：{path_value}")
    elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        issues.append(f"资产 SHA-256 不匹配：{path_value}")
    if not isinstance(record.get("source"), str) or not record["source"].strip():
        issues.append(f"{label} 缺少来源说明")
    if not isinstance(record.get("license"), str) or not record["license"].strip():
        issues.append(f"{label} 缺少许可说明")


def _check_declared_project_assets(
    assets: dict[str, object], issues: list[str], root: Path
) -> None:
    """按 asset_manifest 中的声明检查 canonical、多来源资料和实验入口。"""

    canonical = assets.get("canonical_dataset")
    if not isinstance(canonical, dict):
        issues.append("资产清单缺少 canonical_dataset 记录")
    else:
        dataset_path_value = canonical.get("path")
        if not isinstance(dataset_path_value, str) or not dataset_path_value:
            issues.append("canonical_dataset.path 必须是非空相对路径")
        else:
            dataset_path = (root / dataset_path_value).resolve()
            try:
                dataset_path.relative_to(root.resolve())
            except ValueError:
                issues.append("canonical_dataset.path 必须位于教程根目录内")
            else:
                required = canonical.get("required_files")
                if not isinstance(required, list) or not required:
                    issues.append("canonical_dataset.required_files 必须是非空列表")
                else:
                    for name in required:
                        if not isinstance(name, str) or not (dataset_path / name).is_file():
                            issues.append(f"canonical 数据包缺少文件：{dataset_path_value}/{name}")
                manifest_ref = canonical.get("manifest")
                if not isinstance(manifest_ref, str) or not (root / manifest_ref).is_file():
                    issues.append("canonical_dataset.manifest 不存在")

    multisource = assets.get("multisource_sources")
    if not isinstance(multisource, dict):
        issues.append("资产清单缺少 multisource_sources 记录")
    else:
        _check_multisource_sources(multisource, issues, root)

    experiments = assets.get("complete_experiments")
    if not isinstance(experiments, dict):
        issues.append("资产清单缺少 complete_experiments 记录")
    else:
        notebooks = experiments.get("required_notebooks")
        if not isinstance(notebooks, list) or not notebooks:
            issues.append("complete_experiments.required_notebooks 必须是非空列表")
        else:
            for name in notebooks:
                if not isinstance(name, str) or not (root / name).is_file():
                    issues.append(f"完整实验 Notebook 清单文件不存在：{name}")

def _check_multisource_sources(
    record: dict[str, object], issues: list[str], root: Path
) -> None:
    """校验三份当前项目资料的路径、source_id 和 SHA-256 声明。"""
    required = record.get("required_files")
    if required != list(MULTISOURCE_REQUIRED_FILES):
        issues.append(
            "multisource_sources.required_files 必须按固定顺序列出三份当前资料："
            f"{list(MULTISOURCE_REQUIRED_FILES)!r}"
        )
    files = record.get("files")
    if not isinstance(files, list) or len(files) != len(MULTISOURCE_REQUIRED_FILES):
        issues.append("multisource_sources.files 必须恰好列出三份当前资料")
        return
    seen_paths: set[str] = set()
    seen_source_ids: set[str] = set()
    expected_by_path = {
        path: source_id
        for path, source_id in zip(
            MULTISOURCE_REQUIRED_FILES,
            (
                "canonical_dataset_manifest",
                "data_processing_readme",
                "evaluation_readme",
            ),
        )
    }
    for index, item in enumerate(files):
        label = f"multisource_sources.files[{index}]"
        if not isinstance(item, dict):
            issues.append(f"{label} 必须是对象")
            continue
        path_value = item.get("path")
        source_id = item.get("source_id")
        digest = item.get("sha256")
        if not isinstance(path_value, str) or not path_value:
            issues.append(f"{label}.path 必须是非空相对路径")
            continue
        if path_value in seen_paths:
            issues.append(f"{label}.path 重复：{path_value}")
        seen_paths.add(path_value)
        if path_value not in expected_by_path:
            issues.append(f"{label}.path 不是当前三份资料：{path_value}")
        elif source_id != expected_by_path[path_value]:
            issues.append(
                f"{label}.source_id 与路径不匹配：{path_value} 应为 {expected_by_path[path_value]}"
            )
        if not isinstance(source_id, str) or source_id not in MULTISOURCE_SOURCE_IDS:
            issues.append(f"{label}.source_id 不是允许的当前来源：{source_id!r}")
        elif source_id in seen_source_ids:
            issues.append(f"{label}.source_id 重复：{source_id}")
        else:
            seen_source_ids.add(source_id)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            issues.append(f"{label}.sha256 必须是 64 位小写十六进制")
            continue
        path = (root / path_value).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            issues.append(f"{label}.path 必须位于教程根目录内：{path_value}")
            continue
        if not path.is_file():
            issues.append(f"当前资料文件不存在：{path_value}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            issues.append(f"当前资料 SHA-256 不匹配：{path_value}")
    if seen_paths != set(MULTISOURCE_REQUIRED_FILES):
        issues.append(
            "multisource_sources.files 与 required_files 不一致："
            f"应为 {list(MULTISOURCE_REQUIRED_FILES)!r}，实际为 {sorted(seen_paths)!r}"
        )
    if seen_source_ids != MULTISOURCE_SOURCE_IDS:
        issues.append(
            "multisource_sources.files source_id 集合不一致："
            f"应为 {sorted(MULTISOURCE_SOURCE_IDS)!r}，实际为 {sorted(seen_source_ids)!r}"
        )


def _check_vector_sqlite(path: Path, record: dict, issues: list[str]) -> None:
    try:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        collection = db.execute(
            "SELECT name, dimension FROM collections WHERE name = ?", (record["collection_name"],)
        ).fetchone()
        count = db.execute(
            "SELECT COUNT(*) FROM embeddings WHERE segment_id IN "
            "(SELECT id FROM segments WHERE collection = (SELECT id FROM collections WHERE name = ?))",
            (record["collection_name"],),
        ).fetchone()[0]
        db.close()
    except (OSError, sqlite3.Error) as error:
        issues.append(f"向量库 SQLite 无法只读检查：{error}")
        return
    if collection is None:
        issues.append(f"SQLite 中找不到 collection：{record['collection_name']}")
    else:
        if collection[1] != record["dimension"]:
            issues.append(f"SQLite 向量维度不符：{collection[1]!r}")
    if count != record["record_count"]:
        issues.append(f"SQLite 向量记录数不符：{count!r}")

def _check_notebook(path: Path, issues: list[str]) -> None:
    relative = path.relative_to(COURSE_ROOT)
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        issues.append(f"Notebook 无法读取：{relative} ({error})")
        return
    cells = notebook.get("cells") if isinstance(notebook, dict) else None
    if not isinstance(cells, list):
        issues.append(f"Notebook 结构无效：{relative}")
        return
    for index, cell in enumerate(cells, 1):
        if not isinstance(cell, dict) or cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", "")
        source = "".join(str(part) for part in source) if isinstance(source, list) else source
        if isinstance(source, str):
            _check_links(path, issues, source, index)
    code_cells = [cell for cell in cells if isinstance(cell, dict) and cell.get("cell_type") == "code"]
    outputs = []
    for cell in code_cells:
        if isinstance(cell.get("outputs"), list):
            outputs.extend(output for output in cell["outputs"] if isinstance(output, dict))
    if any(output.get("output_type") == "error" for output in outputs):
        issues.append(f"Notebook 保存了报错输出：{relative}")
    missing = [index + 1 for index, cell in enumerate(code_cells) if not isinstance(cell.get("execution_count"), int)]
    if missing:
        issues.append(f"Notebook 有代码单元没有保存执行记录：{relative}（单元 {missing}）")
    if code_cells and not outputs:
        issues.append(f"Notebook 没有保存任何输出：{relative}")


def _code_sources(path: Path) -> tuple[dict[str, object] | None, list[str]]:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, []
    if not isinstance(notebook, dict):
        return None, []
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        return notebook, []
    return notebook, [
        "".join(str(part) for part in cell.get("source", []))
        for cell in cells
        if isinstance(cell, dict) and cell.get("cell_type") == "code"
    ]


def _check_c7_evaluation_contract(issues: list[str]) -> None:
    """检查 C7 两个模型评估入口的反 fallback 与独立语义验收契约。"""

    capstone_path = COURSE_ROOT / "7. 评估" / "端到端验收.ipynb"
    notebook, capstone_sources = _code_sources(capstone_path)
    if notebook is not None:
        capstone_text = "\n".join(capstone_sources)
        required = (
            "apply_context_budget",
            "len(context_text[case_id]) != used",
            "SEMANTIC_JUDGE_PROMPT",
            "semantic_results",
            "hydrated_citations",
            "supported",
            "contradicted",
            "not_found",
            "semantic_judge_model_called",
        )
        missing = [token for token in required if token not in capstone_text]
        if missing:
            issues.append(f"C7 capstone 缺少语义/预算契约：{missing}")
        try:
            retrieve_index = next(i for i, source in enumerate(capstone_sources) if "retrieved =" in source)
            budget_index = next(i for i, source in enumerate(capstone_sources) if "context_rows =" in source)
            generation_index = next(i for i, source in enumerate(capstone_sources) if "raw = llm_call(" in source)
            semantic_index = next(i for i, source in enumerate(capstone_sources) if "semantic_results = {}" in source)
            posthoc_index = next(i for i, source in enumerate(capstone_sources) if "qrels_by_case =" in source)
        except StopIteration:
            pass
        else:
            if not retrieve_index < budget_index < generation_index < semantic_index < posthoc_index:
                issues.append("C7 capstone 顺序必须是检索 → 预算 → 生成 → 独立语义检查 → post-hoc")
            leaked_fields = (
                "answerability", "reference_claims", "reference_answer",
                "expected_pages", "qrels.jsonl", '"relevance"', '"essential"', '"usage"',
            )
            pre_hoc = "\n".join(capstone_sources[:semantic_index])
            if any(field in pre_hoc for field in leaked_fields):
                issues.append("C7 capstone 生成前不得读取评测标签/qrels")
            if capstone_sources[generation_index].count("llm_call(") != 1:
                issues.append("C7 capstone 生成阶段必须对每道题调用一次 glm-4-flash")
            if capstone_sources[semantic_index].count("llm_call(") != 1:
                issues.append("C7 capstone 语义阶段必须对每道题独立调用一次 glm-4-flash")

    judge_path = COURSE_ROOT / "7. 评估" / "用模型辅助检查回答.ipynb"
    judge_notebook, judge_sources = _code_sources(judge_path)
    if judge_notebook is not None:
        judge_text = "\n".join(judge_sources)
        if "default_score" in judge_text:
            issues.append("C7 模型辅助评估不得使用 default_score fallback")
        if "if not isinstance(evidence, list): evidence = []" in judge_text:
            issues.append("C7 模型辅助评估不得静默清空非法 evidence")
        for token in ("json.loads", "raise ValueError", "parse_improved_judge", "verify_evidence", "NOTHING_FOUND"):
            if token not in judge_text:
                issues.append(f"C7 模型辅助评估缺少严格校验：{token}")
        if "status': '解析失败'" in judge_text or "未生成可用分数" in judge_text:
            issues.append("C7 模型辅助评估解析失败必须直接抛错，不能保存失败分数路径")
        saved = "\n".join(_saved_output_texts(judge_path))
        if "实际调用尝试次数" not in saved or "额外调用：" not in saved:
            issues.append("C7 模型辅助评估没有保存真实校准调用输出")

def _output_text(output: dict[str, object]) -> str:
    if output.get("output_type") == "stream":
        text = output.get("text", "")
    else:
        data = output.get("data", {})
        text = data.get("text/plain", data.get("text/markdown", "")) if isinstance(data, dict) else ""
    if isinstance(text, list):
        text = "".join(str(part) for part in text)
    return text if isinstance(text, str) else ""

def _saved_output_texts(path: Path) -> list[str]:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(notebook, dict):
        return []
    texts = []
    for cell in notebook.get("cells", []):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if not isinstance(output, dict):
                continue
            data = output.get("data")
            audit = data.get(TUTORIAL_AUDIT_MIME) if isinstance(data, dict) else None
            if isinstance(audit, dict):
                audit = {"case_id": audit.get("case_id"), **audit}
                texts.append(json.dumps(audit, ensure_ascii=False))
            elif isinstance(audit, list):
                texts.extend(
                    json.dumps({"case_id": record.get("case_id"), **record}, ensure_ascii=False)
                    for record in audit
                    if isinstance(record, dict)
                )
            if (text := _output_text(output)).strip():
                texts.append(text)
    return texts


def _saved_audit_records(path: Path) -> list[dict[str, object]]:
    """Extract structured tutorial-audit records without executing a Notebook."""

    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(notebook, dict):
        return []
    records: list[dict[str, object]] = []
    for cell in notebook.get("cells", []):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if not isinstance(output, dict):
                continue
            data = output.get("data")
            audit = data.get(TUTORIAL_AUDIT_MIME) if isinstance(data, dict) else None
            values = audit if isinstance(audit, list) else [audit]
            records.extend(
                value
                for value in values
                if isinstance(value, dict) and isinstance(value.get("case_id"), str)
            )
    return records


def _agentic_page(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    if isinstance(value, str) and value.strip().isdigit() and int(value.strip()) > 0:
        return int(value.strip())
    return None


def _agentic_pages(value: object) -> set[int]:
    if not isinstance(value, dict) or not isinstance(value.get("pages"), list):
        return set()
    return {
        page
        for raw_page in value["pages"]
        if (page := _agentic_page(raw_page)) is not None
    }


def _agentic_catalog(
    value: object, case_id: str
) -> tuple[dict[str, dict[str, object]], list[str]]:
    """Validate the saved candidate catalog shape and return it keyed by evidence ID."""

    errors: list[str] = []
    if isinstance(value, dict):
        entries = list(value.items())
    elif isinstance(value, list):
        entries = [(None, item) for item in value]
    else:
        return {}, [f"Agentic {case_id} 缺少非空 candidate_catalog 对象/列表"]
    if not entries:
        errors.append(f"Agentic {case_id} candidate_catalog 不能为空")

    catalog: dict[str, dict[str, object]] = {}
    for key, row in entries:
        if not isinstance(row, dict):
            errors.append(f"Agentic {case_id} candidate_catalog 含非对象候选")
            continue
        evidence_id = row.get("evidence_id")
        quote = row.get("quote")
        page = _agentic_page(row.get("page"))
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            errors.append(f"Agentic {case_id} candidate_catalog 候选缺少 evidence_id")
            continue
        evidence_id = evidence_id.strip()
        if key is not None and str(key) != evidence_id:
            errors.append(
                f"Agentic {case_id} candidate_catalog 键与 evidence_id 不一致：{key!r} != {evidence_id!r}"
            )
        if not isinstance(quote, str) or not quote.strip():
            errors.append(f"Agentic {case_id} candidate_catalog {evidence_id} 缺少 quote")
        if page is None:
            errors.append(f"Agentic {case_id} candidate_catalog {evidence_id} 缺少正整数 page")
        if evidence_id in catalog:
            errors.append(f"Agentic {case_id} candidate_catalog 重复 evidence_id：{evidence_id}")
        catalog[evidence_id] = row
    return catalog, errors


def _agentic_raw_payload(value: object) -> dict[str, object] | None:
    """Parse the saved answer-stage raw object without accepting extra prose."""

    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[0].strip().lower() not in {"```", "```json"} or lines[-1].strip() != "```":
            return None
        text = "\n".join(lines[1:-1]).strip()
    if "```" in text:
        return None
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_agentic_canonical(issues: list[str]) -> dict[str, dict[str, object]]:
    path = COURSE_ROOT / "data" / "dataset" / "evidence.jsonl"
    try:
        rows = dataset_store.read_jsonl(path)
    except (OSError, ValueError) as error:
        issues.append(f"Agentic canonical evidence 无法读取：{error}")
        return {}
    return {
        str(row["evidence_id"]): row
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("evidence_id"), str)
        and row.get("evidence_id")
    }


def _load_agentic_query_qrels(
    issues: list[str],
) -> tuple[dict[str, dict[str, object]], dict[str, set[str]]]:
    """Load answerability and essential qrels for the fixed Agentic cases."""

    dataset_root = COURSE_ROOT / "data" / "dataset"
    try:
        query_rows = dataset_store.read_jsonl(dataset_root / "queries.jsonl")
        qrel_rows = dataset_store.read_jsonl(dataset_root / "qrels.jsonl")
    except (OSError, ValueError) as error:
        issues.append(f"Agentic canonical queries/qrels 无法读取：{error}")
        return {}, {}
    queries: dict[str, dict[str, object]] = {}
    for row in query_rows:
        query_id = row.get("query_id") if isinstance(row, dict) else None
        if not isinstance(query_id, str) or not query_id:
            continue
        if query_id in queries:
            issues.append(f"Agentic canonical queries 存在重复 query_id：{query_id}")
        queries[query_id] = row
    essential: dict[str, set[str]] = {case_id: set() for case_id in AGENTIC_CASE_IDS}
    for row in qrel_rows:
        if not isinstance(row, dict):
            continue
        query_id, evidence_id = row.get("query_id"), row.get("evidence_id")
        if query_id not in AGENTIC_CASE_IDS or not isinstance(evidence_id, str) or not evidence_id:
            continue
        if row.get("essential") is True:
            essential[query_id].add(evidence_id)
    return queries, essential


def _check_agentic_contract(issues: list[str]) -> None:
    """检查 Agentic RAG 固定案例的真实、唯一、可追溯最终答案契约。"""

    path = COURSE_ROOT / AGENTIC_NOTEBOOK
    if not path.is_file():
        issues.append(f"缺少 Agentic RAG Notebook：{AGENTIC_NOTEBOOK}")
        return
    records = _saved_audit_records(path)
    case_ids = [str(record.get("case_id")) for record in records]
    if len(records) != len(AGENTIC_CASE_IDS) or set(case_ids) != set(AGENTIC_CASE_IDS):
        issues.append(
            "Agentic RAG 必须且只能保存两个固定案例 "
            f"{sorted(AGENTIC_CASE_IDS)!r}，实际为 {case_ids!r}"
        )
    duplicates = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
    if duplicates:
        issues.append(f"Agentic RAG 固定案例不得保存重复 answer/audit：{duplicates!r}")
    canonical = _load_agentic_canonical(issues)
    queries, essential_by_case = _load_agentic_query_qrels(issues)

    for case_id in sorted(AGENTIC_CASE_IDS):
        matching = [record for record in records if record.get("case_id") == case_id]
        if len(matching) != 1:
            continue
        record = matching[0]
        if record.get("method") != "让模型选择检索方式（Agentic Retrieval）":
            issues.append(f"Agentic {case_id} audit method 不匹配：{record.get('method')!r}")
        outputs = record.get("model_outputs")
        if not isinstance(outputs, dict):
            issues.append(f"Agentic {case_id} 缺少 model_outputs 对象")
            continue

        plan_stage = outputs.get("plan")
        plan = plan_stage.get("parsed") if isinstance(plan_stage, dict) else None
        planned_ids: list[str] = []
        if not isinstance(plan, dict) or not isinstance(plan.get("actions"), list) or not plan["actions"]:
            issues.append(f"Agentic {case_id} plan 必须保存非空 actions")
        else:
            for index, action in enumerate(plan["actions"]):
                expected_id = f"action_{index + 1}"
                action_id = action.get("action_id") if isinstance(action, dict) else None
                if action_id != expected_id:
                    issues.append(
                        f"Agentic {case_id} plan action[{index}] 必须使用稳定 action_id {expected_id!r}：{action_id!r}"
                    )
                elif action_id in planned_ids:
                    issues.append(f"Agentic {case_id} plan action_id 重复：{action_id!r}")
                else:
                    planned_ids.append(action_id)

        trace = outputs.get("trace")
        trace_steps = trace if isinstance(trace, list) else []
        if not isinstance(trace, list):
            issues.append(f"Agentic {case_id} 缺少 model_outputs.trace 列表")
        for step in trace_steps:
            if not isinstance(step, dict) or step.get("step") != "retrieve":
                continue
            action_id = step.get("action_id")
            requirement_id = step.get("requirement_id")
            if action_id != requirement_id or action_id not in planned_ids:
                issues.append(
                    f"Agentic {case_id} retrieve 必须绑定一个已规划且同值的 action_id/requirement_id：{step!r}"
                )
        verify_records = []
        for name in ("verify_initial", "verify_after_repair", "verify"):
            value = outputs.get(name)
            parsed = value.get("parsed") if isinstance(value, dict) else None
            if isinstance(parsed, dict):
                verify_records.append(parsed)
        verify_records.extend(
            step.get("parsed")
            for step in trace_steps
            if isinstance(step, dict)
            and step.get("step") == "verify"
            and isinstance(step.get("parsed"), dict)
        )
        for verify_record in verify_records:
            covered = verify_record.get("covered")
            missing = verify_record.get("missing")
            sufficient = verify_record.get("sufficient")
            if not isinstance(covered, list) or not isinstance(missing, list):
                issues.append(f"Agentic {case_id} verify covered/missing 必须是列表：{verify_record!r}")
                continue
            if type(sufficient) is not bool:
                issues.append(f"Agentic {case_id} verify.sufficient 必须是 boolean：{verify_record!r}")
            if sufficient != (not missing):
                issues.append(f"Agentic {case_id} verify.sufficient 必须严格等价于 missing 为空：{verify_record!r}")
            observed_ids = covered + missing
            ids_are_strings = all(isinstance(item, str) and item.strip() for item in observed_ids)
            if planned_ids and (
                not ids_are_strings
                or len(observed_ids) != len(planned_ids)
                or len(set(observed_ids)) != len(observed_ids)
                or set(observed_ids) != set(planned_ids)
            ):
                issues.append(
                    f"Agentic {case_id} verify 必须把每个 planned requirement/action ID 恰好放入 covered 或 missing：{verify_record!r}"
                )
        step_names = [
            step.get("step") for step in trace_steps if isinstance(step, dict)
        ]
        answer_indexes = [index for index, step in enumerate(trace_steps)
                          if isinstance(step, dict) and step.get("step") == "answer"]
        if len(answer_indexes) != 1:
            issues.append(f"Agentic {case_id} 必须恰好有一个最终 answer stage：{step_names!r}")
        legacy_answers = [
            name for name in ("answer_draft", "answer_check", "answer_revision")
            if name in step_names
        ]
        if legacy_answers:
            issues.append(
                f"Agentic {case_id} 不得保存多个旧版答案阶段：{legacy_answers!r}"
            )
        repair_indexes = [index for index, step in enumerate(trace_steps)
                          if isinstance(step, dict) and step.get("step") == "repair"]
        if len(repair_indexes) > 1:
            issues.append(f"Agentic {case_id} repair 只能执行一次")
        verify_indexes = [index for index, step in enumerate(trace_steps)
                          if isinstance(step, dict) and step.get("step") == "verify"]
        if not verify_indexes:
            issues.append(f"Agentic {case_id} 缺少 verify stage")
        if answer_indexes:
            answer_index = answer_indexes[0]
            before_answer = [index for index in verify_indexes if index < answer_index]
            if not before_answer:
                issues.append(f"Agentic {case_id} answer 前必须存在最终 verify")
            if repair_indexes:
                repair_index = repair_indexes[0]
                after_repair = [index for index in verify_indexes
                                if repair_index < index < answer_index]
                if not after_repair:
                    issues.append(
                        f"Agentic {case_id} repair 后必须先 final verify，再生成 answer"
                    )

        counts = outputs.get("stage_call_counts")
        if not isinstance(counts, dict):
            issues.append(f"Agentic {case_id} 缺少 stage_call_counts")
        else:
            if counts.get("plan") != 1:
                issues.append(f"Agentic {case_id} plan 必须恰好调用一次：{counts.get('plan')!r}")
            if counts.get("answer") != 1:
                issues.append(f"Agentic {case_id} answer 必须恰好调用一次：{counts.get('answer')!r}")
            repair_count = counts.get("repair", 0)
            if not isinstance(repair_count, int) or isinstance(repair_count, bool) or not 0 <= repair_count <= 1:
                issues.append(f"Agentic {case_id} repair 调用次数必须为 0 或 1：{repair_count!r}")
            old_counts = {
                key: counts.get(key)
                for key in ("answer_draft", "answer_check", "answer_revision")
                if isinstance(counts.get(key), int) and counts.get(key) > 0
            }
            if old_counts:
                issues.append(f"Agentic {case_id} 存在旧版多答案调用计数：{old_counts!r}")

        final_verify = outputs.get("verify")
        final_verify_parsed = final_verify.get("parsed") if isinstance(final_verify, dict) else None
        if not isinstance(final_verify_parsed, dict) or type(final_verify_parsed.get("sufficient")) is not bool:
            issues.append(f"Agentic {case_id} 缺少结构化最终 verify.sufficient")

        answer_stage = outputs.get("answer")
        answer = answer_stage.get("parsed") if isinstance(answer_stage, dict) else None
        if not isinstance(answer_stage, dict) or not isinstance(answer_stage.get("raw"), str) or not answer_stage["raw"].strip():
            issues.append(f"Agentic {case_id} 缺少唯一 answer 的原始模型输出")
        raw_answer = _agentic_raw_payload(answer_stage.get("raw") if isinstance(answer_stage, dict) else None)
        if raw_answer is None or set(raw_answer) != {"status", "claims"}:
            issues.append(
                f"Agentic {case_id} answer raw 必须是只含 status、claims 的 JSON 对象"
            )
        if not isinstance(answer, dict):
            issues.append(f"Agentic {case_id} answer 必须保存结构化 parsed 对象")
            continue
        if record.get("answer") != answer:
            issues.append(f"Agentic {case_id} 顶层 answer 必须与 model_outputs.answer 一致")
        if set(answer) != {"answer", "status", "claims"}:
            issues.append(f"Agentic {case_id} answer 字段必须精确为 answer、status、claims")
            continue
        answer_text, status, claims = answer["answer"], answer["status"], answer["claims"]
        if not isinstance(answer_text, str) or not answer_text.strip():
            issues.append(f"Agentic {case_id} answer.answer 必须是非空字符串")
        if status not in {"answered", "insufficient"}:
            issues.append(f"Agentic {case_id} answer.status 非法：{status!r}")
        if not isinstance(claims, list):
            issues.append(f"Agentic {case_id} answer.claims 必须是列表")
            claims = []
        if status == "insufficient" and claims != []:
            issues.append(f"Agentic {case_id} status=insufficient 时 claims 必须为空")
        if status == "insufficient" and answer_text != "资料不足，无法可靠回答。":
            issues.append(
                f"Agentic {case_id} status=insufficient 时 answer 必须精确为：资料不足，无法可靠回答。"
            )
        if status == "answered" and not claims:
            issues.append(f"Agentic {case_id} status=answered 时 claims 不能为空")
        if status == "answered" and isinstance(claims, list):
            statements = [
                claim.get("statement")
                for claim in claims
                if isinstance(claim, dict) and isinstance(claim.get("statement"), str)
            ]
            if len(statements) == len(claims) and answer_text != "".join(statements):
                issues.append(
                    f"Agentic {case_id} answer 必须严格等于按顺序拼接的 claim statements"
                )
        if isinstance(final_verify_parsed, dict) and type(final_verify_parsed.get("sufficient")) is bool:
            expected_status = "answered" if final_verify_parsed["sufficient"] else "insufficient"
            if status != expected_status:
                issues.append(
                    f"Agentic {case_id} answer.status 与最终 verify 不一致："
                    f"{status!r} != {expected_status!r}"
                )

        catalog, catalog_errors = _agentic_catalog(outputs.get("candidate_catalog"), case_id)
        issues.extend(catalog_errors)
        after_pages = _agentic_pages(record.get("after"))
        for evidence_id, row in catalog.items():
            page = _agentic_page(row.get("page"))
            if page is not None and page not in after_pages:
                issues.append(
                    f"Agentic {case_id} candidate_catalog {evidence_id} 不在最终 merged pages：{page}"
                )
            canonical_row = canonical.get(evidence_id)
            if canonical_row is None:
                issues.append(f"Agentic {case_id} candidate_catalog 使用未知 evidence_id：{evidence_id}")
                continue
            if row.get("quote") != canonical_row.get("quote"):
                issues.append(f"Agentic {case_id} candidate_catalog {evidence_id} quote 与 canonical 不一致")
            if page is not None and page != _agentic_page(canonical_row.get("page")):
                issues.append(f"Agentic {case_id} candidate_catalog {evidence_id} page 与 canonical 不一致")

        query = queries.get(case_id)
        essential_ids = essential_by_case.get(case_id, set())
        if query is None:
            issues.append(f"Agentic {case_id} 不在 canonical queries 中")
        elif query.get("answerability") == "answerable" and not essential_ids:
            issues.append(f"Agentic {case_id} answerable query 没有 canonical essential qrels")
        if (
            isinstance(query, dict)
            and query.get("answerability") == "answerable"
            and essential_ids
            and essential_ids <= set(catalog)
        ):
            if not isinstance(final_verify_parsed, dict) or final_verify_parsed.get("sufficient") is not True:
                issues.append(
                    f"Agentic {case_id} 教程质量问题：answerable query 的最终 candidate_catalog 已覆盖全部 essential evidence，"
                    "但 final verify.sufficient 不是 true"
                )
            if status != "answered" or not claims:
                issues.append(
                    f"Agentic {case_id} 教程质量问题：answerable query 的最终 candidate_catalog 已覆盖全部 essential evidence，"
                    "但 answer.status 不是 answered 或 claims 为空"
                )
        if status != "answered" or not isinstance(claims, list):
            continue
        claimed_evidence_ids: set[str] = set()
        for index, claim in enumerate(claims):
            if not isinstance(claim, dict):
                issues.append(f"Agentic {case_id} claim[{index}] 必须是对象")
                continue
            if set(claim) != {"statement", "evidence_ids", "evidence"}:
                issues.append(
                    f"Agentic {case_id} claim[{index}] 字段必须精确为 statement、evidence_ids、evidence"
                )
                continue
            statement = claim.get("statement")
            if not isinstance(statement, str) or not statement.strip():
                issues.append(f"Agentic {case_id} claim[{index}] 缺少 statement")
            raw_ids = claim.get("evidence_ids")
            if not isinstance(raw_ids, list) or not raw_ids:
                issues.append(f"Agentic {case_id} claim[{index}] evidence_ids 必须是非空列表")
                continue
            evidence_ids: list[str] = []
            invalid_id = False
            for id_index, raw_id in enumerate(raw_ids):
                if not isinstance(raw_id, str) or not raw_id.strip():
                    issues.append(
                        f"Agentic {case_id} claim[{index}].evidence_ids[{id_index}] 必须是非空字符串"
                    )
                    invalid_id = True
                    continue
                evidence_id = raw_id.strip()
                if evidence_id in evidence_ids:
                    issues.append(
                        f"Agentic {case_id} claim[{index}] evidence_ids 不得重复：{evidence_id}"
                    )
                    invalid_id = True
                evidence_ids.append(evidence_id)
            raw_evidence = claim.get("evidence")
            if not isinstance(raw_evidence, list) or not raw_evidence:
                issues.append(f"Agentic {case_id} claim[{index}] evidence 必须是非空列表")
                continue
            if len(raw_evidence) != len(evidence_ids):
                issues.append(
                    f"Agentic {case_id} claim[{index}] evidence 数量必须与 evidence_ids 一致"
                )
                invalid_id = True
            evidence_item_ids: list[str] = []
            claim_valid = not invalid_id
            for evidence_index, item in enumerate(raw_evidence):
                if not isinstance(item, dict) or set(item) != {"evidence_id", "page", "quote"}:
                    issues.append(
                        f"Agentic {case_id} claim[{index}].evidence[{evidence_index}] 字段必须精确为 evidence_id、page、quote"
                    )
                    claim_valid = False
                    continue
                evidence_id = item.get("evidence_id")
                page = item.get("page")
                quote = item.get("quote")
                if not isinstance(evidence_id, str) or not evidence_id.strip():
                    issues.append(
                        f"Agentic {case_id} claim[{index}].evidence[{evidence_index}] evidence_id 必须是非空字符串"
                    )
                    claim_valid = False
                    continue
                evidence_id = evidence_id.strip()
                if evidence_id in evidence_item_ids:
                    issues.append(
                        f"Agentic {case_id} claim[{index}] evidence 中 evidence_id 不得重复：{evidence_id}"
                    )
                    claim_valid = False
                evidence_item_ids.append(evidence_id)
                if not isinstance(page, int) or isinstance(page, bool) or page <= 0:
                    issues.append(
                        f"Agentic {case_id} claim[{index}] evidence[{evidence_index}] page 必须为正整数"
                    )
                    claim_valid = False
                if not isinstance(quote, str) or not quote.strip():
                    issues.append(
                        f"Agentic {case_id} claim[{index}] evidence[{evidence_index}] quote 必须为非空字符串"
                    )
                    claim_valid = False
                candidate = catalog.get(evidence_id)
                canonical_row = canonical.get(evidence_id)
                if candidate is None:
                    issues.append(
                        f"Agentic {case_id} claim[{index}] evidence_id 不在最终 candidate_catalog：{evidence_id}"
                    )
                    claim_valid = False
                    continue
                if canonical_row is None:
                    issues.append(
                        f"Agentic {case_id} claim[{index}] 使用未知 canonical evidence_id：{evidence_id}"
                    )
                    claim_valid = False
                    continue
                candidate_page = _agentic_page(candidate.get("page"))
                canonical_page = _agentic_page(canonical_row.get("page"))
                if page != candidate_page or page != canonical_page:
                    issues.append(
                        f"Agentic {case_id} claim[{index}] evidence[{evidence_index}] page 未与 candidate_catalog/canonical 一致：{evidence_id}"
                    )
                    claim_valid = False
                if quote != candidate.get("quote") or quote != canonical_row.get("quote"):
                    issues.append(
                        f"Agentic {case_id} claim[{index}] evidence[{evidence_index}] quote 未与 candidate_catalog/canonical 一致：{evidence_id}"
                    )
                    claim_valid = False
            if set(evidence_item_ids) != set(evidence_ids):
                issues.append(
                    f"Agentic {case_id} claim[{index}] evidence 与 evidence_ids 的 ID 集合不一致"
                )
                claim_valid = False
            if claim_valid:
                claimed_evidence_ids.update(evidence_ids)

        if (
            isinstance(query, dict)
            and query.get("answerability") == "answerable"
            and essential_ids
        ):
            missing_claim_ids = sorted(essential_ids - claimed_evidence_ids)
            if missing_claim_ids:
                issues.append(
                    f"Agentic {case_id} 教程质量问题：answerable 案例 claims 合并引用未覆盖 essential qrels：{missing_claim_ids!r}"
                )


_BENCHMARK_JSON_START = re.compile(r'\{\s*"schema_version"\s*:')


def _saved_benchmark_reports(path: Path) -> list[dict[str, object]]:
    """Extract saved core-benchmark reports from Notebook outputs.

    The comparison Notebook prints the report with ``json.dumps`` after a
    short status line.  A later version may also publish it through the
    tutorial-audit MIME, so both forms are accepted.  This function only
    parses saved output; it never executes Notebook code.
    """

    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(notebook, dict):
        return []

    decoder = json.JSONDecoder()
    reports: list[dict[str, object]] = []
    seen: set[str] = set()

    def add_report(value: object) -> None:
        if not isinstance(value, dict) or value.get("benchmark") != BENCHMARK_NAME:
            return
        try:
            key = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            return
        if key not in seen:
            seen.add(key)
            reports.append(value)

    for cell in notebook.get("cells", []):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if not isinstance(output, dict):
                continue
            data = output.get("data")
            audit = data.get(TUTORIAL_AUDIT_MIME) if isinstance(data, dict) else None
            audit_values = audit if isinstance(audit, list) else [audit]
            for value in audit_values:
                add_report(value)
            if isinstance(data, dict):
                json_output = data.get("application/json")
                if isinstance(json_output, dict):
                    add_report(json_output)
                elif isinstance(json_output, list):
                    for value in json_output:
                        add_report(value)
                elif isinstance(json_output, str):
                    try:
                        add_report(json.loads(json_output))
                    except json.JSONDecodeError:
                        pass

            text = _output_text(output)
            for match in _BENCHMARK_JSON_START.finditer(text):
                try:
                    value, _ = decoder.raw_decode(text[match.start():])
                except json.JSONDecodeError:
                    continue
                add_report(value)
    return reports


def _benchmark_reports_equal(left: object, right: object) -> bool:
    """Compare reports independent of JSON object key order."""

    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    try:
        left_text = json.dumps(left, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        right_text = json.dumps(right, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return False
    return left_text == right_text


def _load_local_benchmark(root: Path):
    """Import the local benchmark module without invoking an external model."""

    path = root / BENCHMARK_SCRIPT
    if not path.is_file():
        raise FileNotFoundError(f"缺少核心 benchmark 脚本：{BENCHMARK_SCRIPT}")
    spec = importlib.util.spec_from_file_location("c7_saved_benchmark", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载核心 benchmark 脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_benchmark_notebook_consistency(
    issues: list[str], root: Path | None = None
) -> None:
    """Recompute the local benchmark and compare it with saved Notebook JSON.

    ``run_benchmark()`` is intentionally the only computation here.  Its
    retrieval path is pure local BM25/CCH and does not call a model or API;
    any data/configuration failure is reported as an issue instead of being
    hidden by a fallback.
    """

    root = COURSE_ROOT if root is None else root
    notebook_path = root / BENCHMARK_NOTEBOOK
    if not notebook_path.is_file():
        issues.append(f"缺少核心 benchmark Notebook：{BENCHMARK_NOTEBOOK}")
        return
    saved = _saved_benchmark_reports(notebook_path)
    if len(saved) != 1:
        issues.append(
            "C7 核心 benchmark Notebook 必须保存且仅保存一份结构化报告："
            f"实际找到 {len(saved)} 份"
        )
        return

    try:
        benchmark = _load_local_benchmark(root)
        current = benchmark.run_benchmark()
        benchmark.validate_report(current)
        benchmark.validate_report(saved[0])
    except Exception as error:  # noqa: BLE001 - checker must report, not mask, failures
        issues.append(f"C7 核心 benchmark 当前运行无法校验：{error}")
        return
    if not _benchmark_reports_equal(current, saved[0]):
        issues.append(
            "C7 核心 benchmark 当前 run_benchmark() 与比较 Notebook 保存的结构化结果不一致；"
            "请在 canonical 数据或 benchmark 配置固定后重新执行 Notebook 并保存输出"
        )


def _check_contextual_provenance(issues: list[str]) -> None:
    """校验 Contextual Retrieval 保存输出仍带有可追溯的来源范围。"""
    path = COURSE_ROOT / "3. 索引阶段" / "给片段补充所属上下文.ipynb"
    if not path.is_file():
        return
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    found = False
    for text in _saved_output_texts(path):
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or "provenance" not in value:
            continue
        found = True
        provenance = value.get("provenance")
        outputs = value.get("model_outputs")
        if not isinstance(provenance, dict) or not all(isinstance(provenance.get(k), str) and provenance[k].strip()
                                                       for k in ("status", "index_scope", "candidate_selection", "raw_output_coverage")):
            issues.append("Contextual Retrieval 输出缺少完整 provenance")
        selection = provenance.get("candidate_selection", "") if isinstance(provenance, dict) else ""
        selection_lower = selection.lower()
        mentions_labels = re.search(r"expected_pages|reference_answer", selection_lower)
        explicitly_excludes = re.search(r"(?:no|without|不(?:使用|读取|依赖))\s*[^.]*?(?:expected_pages|reference_answer)", selection_lower)
        if mentions_labels and not explicitly_excludes:
            issues.append("Contextual Retrieval candidate_selection 不得使用 expected_pages/reference_answer")
        if not isinstance(outputs, dict) or not isinstance(outputs.get("candidate_chunks"), list):
            issues.append("Contextual Retrieval 输出缺少 candidate_chunks")
            continue
        candidates = set(map(str, outputs["candidate_chunks"]))
        for key in ("validated_contexts", "raw_generated_contexts", "validation_notes"):
            values = outputs.get(key)
            if values is not None and (not isinstance(values, dict) or not set(map(str, values)) <= candidates):
                issues.append(f"Contextual Retrieval {key} 超出 candidate_chunks")
    if not found:
        issues.append("Contextual Retrieval Notebook 没有保存 provenance 输出")


def _check_canonical_file_integrity(dataset_root: Path, issues: list[str]) -> None:
    manifest_path = dataset_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict):
        return
    for name, metadata in files.items():
        if not isinstance(metadata, dict) or not isinstance(metadata.get("path"), str):
            issues.append(f"canonical manifest.files.{name} 缺少 path")
            continue
        relative = Path(metadata["path"])
        path = (dataset_root / relative).resolve()
        try:
            path.relative_to(dataset_root.resolve())
        except ValueError:
            issues.append(f"canonical 文件路径越出数据包：{metadata['path']}")
            continue
        if not path.is_file():
            issues.append(f"canonical 文件缺失：{metadata['path']}")
            continue
        expected = metadata.get("sha256")
        if isinstance(expected, str) and dataset_store.sha256_file(path) != expected:
            issues.append(f"canonical 文件 SHA-256 不匹配：{metadata['path']}")
        count = metadata.get("record_count")
        if isinstance(count, int) and path.suffix == ".jsonl":
            try:
                actual = len(dataset_store.read_jsonl(path))
            except (OSError, ValueError) as error:
                issues.append(f"canonical JSONL 无法读取：{metadata['path']}（{error}）")
            else:
                if actual != count:
                    issues.append(
                        f"canonical 文件记录数不匹配：{metadata['path']}（manifest={count}，实际={actual}）"
                    )


def _check_project_catalog(issues: list[str]) -> None:
    """检查 canonical package、三份当前多来源资料和完整实验入口。"""

    dataset_root = COURSE_ROOT / "data" / "dataset"
    manifest_path = COURSE_ROOT / CANONICAL_MANIFEST
    if not manifest_path.is_file():
        issues.append(f"缺少 canonical manifest：{CANONICAL_MANIFEST}")
    else:
        _check_canonical_file_integrity(dataset_root, issues)
        try:
            issues.extend(f"canonical dataset：{issue}" for issue in dataset_store.validate_dataset(dataset_root))
        except (OSError, ValueError) as error:
            issues.append(f"canonical dataset 无法校验：{error}")
    for relative in REQUIRED_EXPERIMENTS:
        path = COURSE_ROOT / relative
        if not path.is_file():
            issues.append(f"缺少完整实验 Notebook：{relative}")


def _check_legacy_references(issues: list[str]) -> None:
    """禁止可迁移代码重新依赖已淘汰的旧问题文件名。"""

    marker = "rag" + "_" + "cases"
    for path in COURSE_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in {".ipynb", ".py", ".md"}:
            continue
        relative = path.relative_to(COURSE_ROOT).as_posix()
        if relative in RESERVED_NOTEBOOKS or relative.startswith("common/"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if marker in text:
            issues.append(f"发现直接旧问题集引用：{relative}")


def _find_saved_case(texts: list[str], case_id: str) -> dict[str, object] | None:
    decoder = json.JSONDecoder()
    fallback = None
    for text in texts:
        for match in CASE_JSON.finditer(text):
            if match.group(1) != case_id:
                continue
            try:
                value, _ = decoder.raw_decode(text[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("case_id") == case_id:
                if "method" in value or "role" in value:
                    return value
                fallback = value
    return fallback

def _number(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None

def _metric_text(value: float) -> str:
    return "未命中" if math.isinf(value) else f"{value:g}"

def _read_comparable_metrics(result: dict[str, object]) -> tuple[dict[str, object] | None, str]:
    pages = []
    for side in ("before", "after"):
        value = result.get(side)
        if not isinstance(value, dict):
            return None, f"缺少 {side} 对象"
        if not isinstance(value.get("pages"), list):
            return None, f"{side}.pages 不是列表"
        coverage = _number(value.get("required_page_coverage"))
        raw_rank = value.get("first_required_rank")
        rank = float("inf") if raw_rank is None else _number(raw_rank)
        if coverage is None or rank is None:
            return None, f"{side} 缺少可比较的排名或必要页覆盖率"
        if rank < 0:
            return None, f"{side}.first_required_rank 不能为负数"
        pages.append((coverage, rank))

    comparison = result.get("comparison")
    if comparison is None:
        primary = None
    elif not isinstance(comparison, dict):
        return None, "comparison 不是对象"
    else:
        name = comparison.get("name")
        before, after = _number(comparison.get("before")), _number(comparison.get("after"))
        higher_is_better = comparison.get("higher_is_better")
        if not isinstance(name, str) or not name.strip():
            return None, "comparison.name 不能为空"
        if before is None or after is None:
            return None, "comparison 缺少可比较的 before/after 数值"
        if not isinstance(higher_is_better, bool):
            return None, "comparison.higher_is_better 必须是布尔值"
        primary = {"name": name.strip(), "before": before, "after": after, "higher_is_better": higher_is_better}
    return {"pages": (pages[0], pages[1]), "comparison": primary}, ""

def _audit_purpose(
    purpose: str, improved: bool, regressed: bool, label: str, tradeoff: str | None = None,
) -> tuple[str, str]:
    if purpose == "说明不适用或限制":
        return "limited", f"限制案例：{label}可比较，结果不据此判定成败"
    if purpose == "确认没有改坏":
        return ("passed", f"复查{label}没有退化（允许不变或改善）") if not regressed else (
            "failed", f"复查{label}发生退化"
        )
    if purpose == "main":
        return ("passed", f"正文{label}严格改善") if improved and not regressed else (
            "failed", f"正文{label}没有严格改善，或发生退化"
        )
    if purpose == "再次改善":
        if not improved:
            return "failed", f"复查{label}没有任何严格改善"
        if regressed and tradeoff is not None:
            return "passed", tradeoff
        return ("passed", f"复查{label}严格改善") if not regressed else (
            "failed", f"复查{label}没有严格改善，或发生退化"
        )
    return "unable_to_judge", f"未知的复查用途：{purpose}"

def _audit_saved_case(result: dict[str, object], purpose: str = "再次改善") -> tuple[str, str]:
    metrics, reason = _read_comparable_metrics(result)
    if metrics is None:
        return "unable_to_judge", reason
    (before_coverage, before_rank), (after_coverage, after_rank) = metrics["pages"]
    primary = metrics["comparison"]
    if primary is not None:
        before, after = primary["before"], primary["after"]
        improved = after > before if primary["higher_is_better"] else after < before
        regressed = after < before if primary["higher_is_better"] else after > before
        label = f"comparison「{primary['name']}」{before:g}→{after:g}"
        return _audit_purpose(purpose, improved, regressed, label)

    coverage_improved, rank_improved = after_coverage > before_coverage, after_rank < before_rank
    coverage_regressed, rank_regressed = after_coverage < before_coverage, after_rank > before_rank
    label = (
        f"必要页覆盖率 {_metric_text(before_coverage)}→{_metric_text(after_coverage)}，"
        f"首条排名 {_metric_text(before_rank)}→{_metric_text(after_rank)}"
    )
    tradeoff = (
        f"复查必要页覆盖率 {_metric_text(before_coverage)}→{_metric_text(after_coverage)} 严格改善，"
        f"首条排名 {_metric_text(before_rank)}→{_metric_text(after_rank)} 作为可解释权衡记录"
    )
    if purpose == "再次改善":
        if coverage_regressed and rank_improved:
            return "failed", (
                f"复查必要页覆盖率 {_metric_text(before_coverage)}→{_metric_text(after_coverage)} 下降，"
                f"不能用首条排名 {_metric_text(before_rank)}→{_metric_text(after_rank)} 的改善抵消"
            )
        if coverage_improved and rank_regressed:
            return "passed", tradeoff
        if coverage_regressed or rank_regressed:
            return "failed", (
                f"复查出现不被允许的权衡：必要页覆盖率 {_metric_text(before_coverage)}→{_metric_text(after_coverage)}，"
                f"首条排名 {_metric_text(before_rank)}→{_metric_text(after_rank)}"
            )
    return _audit_purpose(
        purpose, coverage_improved or rank_improved, coverage_regressed or rank_regressed, label, tradeoff
    )

def _comparison_name(result: dict[str, object]) -> str | None:
    comparison = result.get("comparison")
    name = comparison.get("name") if isinstance(comparison, dict) else None
    return name.strip() if isinstance(name, str) and name.strip() else None

def _audit_method_results(methods: dict[str, object], cases_by_id: dict[str, dict[str, object]], issues: list[str]) -> dict[str, object]:
    cache: dict[Path, list[str]] = {}
    checked: list[str] = []
    main_count = check_count = 0
    limitations: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    tradeoffs: list[dict[str, object]] = []
    unable: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    for method, usage in methods.items():
        if not isinstance(usage, dict):
            unable.append({"method": method, "role": "method", "reason": "方法记录不是对象"})
            continue
        notebook, main_id, check_id = usage.get("notebook"), usage.get("main"), usage.get("check")
        purpose = usage.get("check_purpose")
        if not isinstance(notebook, str) or not isinstance(main_id, str) or main_id not in cases_by_id:
            unable.append({"method": method, "role": "main", "case_id": main_id, "reason": "正文问题或 Notebook 不可用"})
            main_id = None
        if not isinstance(check_id, str) or check_id not in cases_by_id:
            unable.append({"method": method, "role": "check", "case_id": check_id, "reason": "复查问题或 Notebook 不可用"})
            check_id = None
        if not isinstance(notebook, str):
            continue
        path = COURSE_ROOT / notebook
        if not path.is_file():
            unable.extend({"method": method, "role": role, "case_id": case_id, "reason": "对应 Notebook 不存在"}
                          for role, case_id in (("main", main_id), ("check", check_id)) if case_id is not None)
            continue
        cache.setdefault(path, _saved_output_texts(path))
        statuses: dict[str, str] = {}
        for role, case_id in (("main", main_id), ("check", check_id)):
            if case_id is None:
                continue
            result = _find_saved_case(cache[path], case_id)
            audit_purpose = "main" if role == "main" else purpose
            if result is None:
                status, reason = "unable_to_judge", "没有保存标准 before/after JSON"
            elif result.get("method") != method:
                status, reason = "unable_to_judge", f"标准结果 method 不匹配：{result.get('method')!r}"
                issues.append(f"方法 {method} 的{role}标准结果 method 不匹配：{case_id}（{reason}）")
            elif result.get("role") != role:
                status, reason = "unable_to_judge", f"标准结果 role 不匹配：{result.get('role')!r}"
                issues.append(f"方法 {method} 的{role}标准结果 role 不匹配：{case_id}（{reason}）")
            elif not isinstance(audit_purpose, str):
                status, reason = "unable_to_judge", "没有说明复查题用途"
            else:
                status, reason = _audit_saved_case(result, audit_purpose)
            if result is not None and (name := _comparison_name(result)):
                comparisons.append({"method": method, "role": role, "case_id": case_id, "name": name})
            statuses[role] = status
            entry = {"method": method, "role": role, "case_id": case_id, "reason": reason}
            if status == "failed":
                failed.append(entry)
                issues.append(f"方法 {method} 的{('正文' if role == 'main' else purpose)}结果未符合审计要求：{case_id}（{reason}）")
            elif status == "unable_to_judge":
                unable.append(entry)
            elif status == "limited":
                limitations.append(entry)
            elif "可解释权衡" in reason:
                tradeoffs.append(entry)
        if statuses.get("main") == "passed":
            main_count += 1
        if statuses.get("check") in {"passed", "limited"}:
            check_count += 1
        if statuses.get("main") == "passed" and statuses.get("check") in {"passed", "limited"}:
            checked.append(method)
    for entry in unable:
        issues.append(
            f"方法 {entry.get('method')} 的{entry.get('role')}结果无法判断："
            f"{entry.get('case_id', '')}（{entry.get('reason', '未说明原因')}）"
        )
    return {
        "method_count": len(methods), "checked_count": len(checked), "main_checked_count": main_count,
        "check_checked_count": check_count, "failed_count": len(failed),
        "unable_to_judge_count": len({entry["method"] for entry in unable}), "checked": checked,
        "limitations": limitations, "comparisons": comparisons, "tradeoffs": tradeoffs,
        "failed": failed, "unable_to_judge": unable,
    }

def _check_canonical_dataset(issues: list[str]) -> None:
    global LAST_RESULT_AUDIT
    dataset_root = COURSE_ROOT / "data" / "dataset"
    try:
        package = dataset_store._read_package(dataset_root)
    except (OSError, ValueError) as error:
        issues.append(f"canonical 数据包无法读取：{error}")
        return
    queries = package.get("queries", [])
    case_ids = {
        row.get("query_id")
        for row in queries
        if isinstance(row, dict) and isinstance(row.get("query_id"), str)
    }
    if len(case_ids) != len(queries):
        issues.append("canonical queries 必须全部有唯一的 query_id")
    methods = package.get("method_cases")
    if not isinstance(methods, dict) or not methods:
        issues.append("canonical package 缺少方法和问题的对应关系")
        return
    cases_by_id = {
        row["query_id"]: row
        for row in queries
        if isinstance(row, dict) and isinstance(row.get("query_id"), str)
    }
    role_uses: dict[str, list[str]] = {}
    for method, usage in methods.items():
        if not isinstance(usage, dict):
            issues.append(f"方法 {method} 没有分别记录正文问题和复核问题")
            continue
        main_case, check_case = usage.get("main"), usage.get("check")
        for label, case_id in (("正文问题", main_case), ("复核问题", check_case)):
            if case_id not in case_ids:
                issues.append(f"方法 {method} 的{label}不存在于 canonical queries：{case_id}")
        if main_case == check_case:
            issues.append(f"方法 {method} 的正文问题和复核问题不能相同")
        for role, case_id in (("main", main_case), ("check", check_case)):
            if isinstance(case_id, str):
                role_uses.setdefault(case_id, []).append(f"{method}.{role}")
        if usage.get("check_purpose") not in {"再次改善", "确认没有改坏", "说明不适用或限制"}:
            issues.append(f"方法 {method} 没有说明复查题的用途")
        notebook = usage.get("notebook")
        notebook_path = COURSE_ROOT / notebook if isinstance(notebook, str) else None
        if notebook_path is None or not notebook_path.is_file():
            issues.append(f"方法 {method} 对应的 Notebook 不存在：{notebook}")
            continue
        notebook_text = notebook_path.read_text(encoding="utf-8")
        for label, case_id in (("正文问题", main_case), ("复核问题", check_case)):
            if isinstance(case_id, str) and case_id not in notebook_text:
                issues.append(f"方法 {method} 的{label}没有出现在对应 Notebook：{case_id}")
    repeated = {case_id: roles for case_id, roles in role_uses.items() if len(roles) > 1}
    if repeated:
        issues.append(f"方法的正文/复核问题在 main 或 check 中重复使用：{repeated}")
    for case_id in sorted(case_ids - set(role_uses)):
        usage = cases_by_id[case_id].get("usage", [])
        if not isinstance(usage, list) or not usage:
            issues.append(f"问题 {case_id} 未被任何方法使用且没有 usage")
    LAST_RESULT_AUDIT = _audit_method_results(methods, cases_by_id, issues)

def check() -> tuple[list[str], int]:
    global LAST_RESULT_AUDIT
    LAST_RESULT_AUDIT = {}
    issues: list[str] = []
    markdown_files, notebook_files = sorted(COURSE_ROOT.rglob("*.md")), sorted(COURSE_ROOT.rglob("*.ipynb"))
    for path in markdown_files:
        _check_links(path, issues)
    for path in notebook_files:
        _check_notebook(path, issues)
    for required in (
        COURSE_ROOT / "data" / "pumpkin_book.pdf",
        COURSE_ROOT / "data" / "dataset" / "manifest.json",
        COURSE_ROOT / "data" / "向量库" / "南瓜书配套库" / "chroma.sqlite3",
    ):
        if not required.is_file():
            issues.append(f"缺少教程文件：{required.relative_to(COURSE_ROOT)}")
    _check_manifest(issues)
    _check_project_catalog(issues)
    _check_contextual_provenance(issues)
    _check_canonical_dataset(issues)
    _check_legacy_references(issues)
    _check_c7_evaluation_contract(issues)
    _check_agentic_contract(issues)
    _check_benchmark_notebook_consistency(issues)
    return issues, len(notebook_files)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查教程文件、问题集以及已保存的 Notebook 结果")
    parser.add_argument("--verbose", action="store_true", help="显示已检查、限制、失败和暂不能判断明细")
    args = parser.parse_args(argv)
    issues, notebook_count = check()
    audit = LAST_RESULT_AUDIT
    if args.verbose:
        result_audit = audit
    else:
        keys = ("method_count", "checked_count", "main_checked_count", "check_checked_count", "failed_count", "unable_to_judge_count")
        result_audit = {key: audit.get(key, 0) for key in keys}
        if audit.get("failed"):
            result_audit["failed"] = audit["failed"]
    print(json.dumps({"passed": not issues, "notebook_count": notebook_count, "issues": issues, "result_audit": result_audit}, ensure_ascii=False, indent=2))
    return 0 if not issues else 1
if __name__ == "__main__":
    raise SystemExit(main())
