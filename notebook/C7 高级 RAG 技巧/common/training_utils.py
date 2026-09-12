"""C2 向量模型微调实验的训练与检索评估契约。

这个模块只接受 canonical dataset 中已经通过页码/片段核验的
``finetune_pairs.jsonl``。训练正例字段是 PDF 的真实 chunk（``positive``），
不会把候选行里的生成答案当作训练目标。模型、数据和依赖都必须由调用方
明确提供；任何导入或加载失败都会直接抛错，不会静默切换到其他模型或数据。
"""

from __future__ import annotations

import json
import inspect
import math
import random
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MODEL_NAME = "BAAI/bge-small-zh-v1.5"
MODEL_MAX_SEQ_LENGTH = 512
FIXED_CHUNK_TOKEN_BUDGET = 384
C7_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_REVIEW_TYPE = "model_assisted_semantic_training_v3"
SEMANTIC_REVIEW_STATUS = "model_assisted_semantically_verified"
SEMANTIC_REVIEW_CHECKS = (
    "question_self_contained",
    "positive_fully_supports",
    "answer_fully_supported",
    "support_quotes_sufficient",
)
DEFAULT_TOP_KS = (1, 3, 5, 10)
PAIR_FIELDS = (
    "pair_id",
    "query_id",
    "query",
    "positive",
    "positive_evidence_id",
    "page",
    "section_id",
    "query_family_id",
    "split",
    "review_status",
    "human_verified",
)


class TrainingContractError(ValueError):
    """训练数据、评估引用或模型契约不满足要求。"""


def _portable_artifact_path(path: str | Path) -> str:
    """Return a stable project-relative artifact reference for reports.

    Runtime loaders still receive explicit ``Path`` objects.  Persisted
    reports must not capture a contributor's home directory or checkout path;
    artifacts inside C7 are therefore written relative to the tutorial root,
    while an externally supplied test/output directory is represented by its
    final component.
    """

    value = Path(path).resolve()
    try:
        return value.relative_to(C7_ROOT.resolve()).as_posix()
    except ValueError:
        return value.name


def _validate_semantic_review(
    row: Mapping[str, Any], pair_id: str, *, require_details: bool = True
) -> None:
    """Require the one current semantic-review contract for every pair.

    Legacy automatic-review states deliberately have no acceptance path:
    canonical training rows must identify the v3 model review and its full
    passed audit, even when a caller only wants to inspect the pair shape.
    """

    if row.get("review_status") != SEMANTIC_REVIEW_STATUS:
        raise TrainingContractError(f"{pair_id} 未通过 v3 语义核验")
    if not require_details:
        return
    semantic_review = row.get("semantic_review")
    if (
        row.get("review_type") != SEMANTIC_REVIEW_TYPE
        or row.get("supervision_type") != "semantic_query_evidence"
        or not isinstance(semantic_review, Mapping)
        or semantic_review.get("status") != "passed"
        or semantic_review.get("review_type") != SEMANTIC_REVIEW_TYPE
        or semantic_review.get("human_verified") is not False
        or any(semantic_review.get(key) is not True for key in SEMANTIC_REVIEW_CHECKS)
    ):
        raise TrainingContractError(f"{pair_id} 未通过 v3 语义核验")


def _assert_empty_output_dir(output_dir: str | Path) -> Path:
    """Return a writable, empty directory without ever reusing artifacts.

    An existing empty directory is safe to use (this is useful when a caller
    creates it with ``mktemp``), but any existing file, symlink, or directory
    entry is a hard error.  Keeping this check in one helper makes the
    training and experiment entry points obey the same no-reuse contract.
    """

    output_path = Path(output_dir)
    if output_path.is_symlink() or output_path.exists():
        if not output_path.is_dir() or output_path.is_symlink():
            raise FileExistsError(f"训练输出路径已有对象，拒绝覆盖：{output_path}")
        try:
            next(output_path.iterdir())
        except StopIteration:
            return output_path
        raise FileExistsError(f"训练输出目录已有内容，拒绝覆盖：{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.mkdir()
    return output_path


def _write_json(path: str | Path, value: object) -> None:
    """Write JSON atomically and reject non-finite metric values."""

    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"训练数据文件不存在：{path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise TrainingContractError(f"JSONL 不允许空行：{path}:{line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TrainingContractError(f"JSONL 每行必须是对象：{path}:{line_number}")
        rows.append(value)
    return rows


def _by_id(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise TrainingContractError(f"记录缺少非空 {field}")
        if value in result:
            raise TrainingContractError(f"重复 {field}：{value}")
        result[value] = row
    return result


def validate_finetune_pairs(
    pairs: Sequence[Mapping[str, Any]],
    *,
    evidence_rows: Sequence[Mapping[str, Any]] | None = None,
    query_rows: Sequence[Mapping[str, Any]] | None = None,
    require_reviewed: bool = True,
) -> list[dict[str, Any]]:
    """检查 query→真实 evidence chunk 的训练行，并返回普通 dict 副本。

    ``answer``/``reference_answer`` 等字段一旦出现在 pair 中就拒绝，避免
    后续训练代码误把生成答案当作 positive。若提供 canonical evidence/query，
    还会检查 ID、页码、section、核验状态和 positive 与原文 chunk 的逐字一致性。
    """

    evidence_by_id = _by_id(evidence_rows, "evidence_id") if evidence_rows is not None else {}
    query_by_id = _by_id(query_rows, "query_id") if query_rows is not None else {}
    seen: set[str] = set()
    checked: list[dict[str, Any]] = []
    required = set(PAIR_FIELDS)
    for index, source in enumerate(pairs, 1):
        if not isinstance(source, Mapping):
            raise TrainingContractError(f"finetune pair 第 {index} 行必须是对象")
        row = dict(source)
        missing = sorted(required - row.keys())
        if missing:
            raise TrainingContractError(f"finetune pair 第 {index} 行缺少字段：{', '.join(missing)}")
        pair_id = row["pair_id"]
        if not isinstance(pair_id, str) or not pair_id or pair_id in seen:
            raise TrainingContractError(f"无效或重复 pair_id：{pair_id!r}")
        seen.add(pair_id)
        if any(key in row for key in ("answer", "reference_answer", "generated_answer")):
            raise TrainingContractError(f"{pair_id} 不得包含生成答案字段")
        if not isinstance(row["query"], str) or not row["query"].strip():
            raise TrainingContractError(f"{pair_id} query 为空")
        if not isinstance(row["positive"], str) or not row["positive"].strip():
            raise TrainingContractError(f"{pair_id} positive 必须为非空原文 chunk")
        for field in ("query_id", "section_id", "query_family_id"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise TrainingContractError(f"{pair_id} {field} 必须为非空字符串")
        if any(
            marker in row["positive"]
            for marker in ("欢迎去各大电商平台选购纸质版南瓜书", "配套视频教程", "bilibili.com")
        ):
            raise TrainingContractError(f"{pair_id} positive 含 PDF 重复水印")
        if not isinstance(row["split"], str) or row["split"] not in {"train", "dev", "test"}:
            raise TrainingContractError(f"{pair_id} split 无效：{row['split']!r}")
        if not isinstance(row["page"], int) or row["page"] <= 0:
            raise TrainingContractError(f"{pair_id} page 必须为正整数")
        # Keep the historical keyword for API compatibility, but never let it
        # relax the canonical review contract.  A shape-only caller must not
        # be able to pass a failed or incomplete semantic audit downstream.
        if row["human_verified"] is not False:
            raise TrainingContractError(f"{pair_id} 不得伪装成人工核验")
        _validate_semantic_review(row, pair_id)
        evidence_id = row["positive_evidence_id"]
        if not isinstance(evidence_id, str) or not evidence_id:
            raise TrainingContractError(f"{pair_id} positive_evidence_id 为空")
        evidence = evidence_by_id.get(evidence_id)
        if evidence is not None:
            if row["positive"] != evidence.get("quote"):
                raise TrainingContractError(f"{pair_id} positive 不是 evidence.quote 的原文 chunk")
            if row["page"] != evidence.get("page"):
                raise TrainingContractError(f"{pair_id} page 与 evidence.page 不一致")
            if row["section_id"] != evidence.get("section_id"):
                raise TrainingContractError(f"{pair_id} section_id 与 evidence 不一致")
            if evidence.get("evidence_type") != "fixed_token_chunk":
                raise TrainingContractError(
                    f"{pair_id} positive evidence 必须是 fixed_token_chunk，禁止整页 positive"
                )
            token_count = evidence.get("token_count")
            token_budget = evidence.get("token_budget")
            if (
                not isinstance(token_count, int)
                or isinstance(token_count, bool)
                or not isinstance(token_budget, int)
                or isinstance(token_budget, bool)
                or token_count <= 0
                or token_count > token_budget
                or token_budget > FIXED_CHUNK_TOKEN_BUDGET
                or token_budget >= MODEL_MAX_SEQ_LENGTH
            ):
                raise TrainingContractError(f"{pair_id} positive chunk token budget 无效")
            if row.get("positive_token_count") != token_count:
                raise TrainingContractError(f"{pair_id} positive_token_count 与 evidence 不一致")
            if row.get("token_budget") != token_budget:
                raise TrainingContractError(f"{pair_id} token_budget 与 evidence 不一致")
        elif evidence_rows is not None:
            raise TrainingContractError(
                f"{pair_id} positive_evidence_id 引用了不存在的 evidence：{evidence_id!r}"
            )
        query = query_by_id.get(row["query_id"])
        if query is not None:
            if row["query"] != query.get("text"):
                raise TrainingContractError(f"{pair_id} query 与 canonical query 不一致")
            if query.get("query_family_id") != row["query_family_id"]:
                raise TrainingContractError(f"{pair_id} query_family_id 不一致")
            claims = query.get("reference_claims", [])
            refs = {ref for claim in claims if isinstance(claim, Mapping) for ref in claim.get("evidence_ids", [])}
            if evidence_id not in refs:
                raise TrainingContractError(f"{pair_id} positive evidence 不在 query reference_claims 中")
        elif query_rows is not None:
            raise TrainingContractError(
                f"{pair_id} query_id 引用了不存在的 canonical query：{row['query_id']!r}"
            )
        checked.append(row)
    if not checked:
        raise TrainingContractError("finetune_pairs 不能为空")
    return checked


def load_finetune_pairs(
    path: str | Path,
    *,
    evidence_rows: Sequence[Mapping[str, Any]] | None = None,
    query_rows: Sequence[Mapping[str, Any]] | None = None,
    split: str | None = None,
) -> list[dict[str, Any]]:
    """读取并校验 finetune_pairs；split 过滤发生在完整校验之后。"""

    pairs = validate_finetune_pairs(
        _jsonl(path), evidence_rows=evidence_rows, query_rows=query_rows
    )
    if split is None:
        return pairs
    if split not in {"train", "dev", "test"}:
        raise TrainingContractError(f"split 无效：{split!r}")
    selected = [row for row in pairs if row["split"] == split]
    if not selected:
        raise TrainingContractError(f"split={split} 没有训练行")
    return selected


def seed_everything(seed: int = 42) -> None:
    """设置训练和 DataLoader 使用的随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - 运行环境契约错误
        raise RuntimeError("训练需要已安装的 torch，禁止依赖 fallback") from exc
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sentence_transformers():
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
    except ImportError as exc:  # pragma: no cover - 运行环境契约错误
        raise RuntimeError("训练需要已安装的 sentence-transformers，禁止模型 fallback") from exc
    return InputExample, SentenceTransformer, losses


def _load_sentence_transformer_from_path(
    sentence_transformer_cls: Any, path: Path, *, device: str
):
    """Load an explicitly local model across sentence-transformers versions.

    ``local_files_only`` was added to newer sentence-transformers releases,
    while the pinned environment used by this repository may expose only the
    older constructor.  The path is resolved locally before this function is
    called, and no model identifier is passed to the constructor, so omitting
    the unsupported keyword cannot trigger a remote fallback.
    """

    try:
        parameters = inspect.signature(sentence_transformer_cls).parameters
    except (TypeError, ValueError) as exc:  # pragma: no cover - unusual proxy class
        raise RuntimeError("无法检查 SentenceTransformer 加载接口") from exc
    kwargs: dict[str, Any] = {"device": device}
    if "local_files_only" in parameters:
        kwargs["local_files_only"] = True
    try:
        return sentence_transformer_cls(str(path), **kwargs)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"无法加载本地 SentenceTransformer：{path}") from exc


def _assert_model_artifacts(path: Path) -> None:
    """Reject incomplete local snapshots before a loader can seek the network."""

    required_files = ("config.json", "modules.json")
    missing = [name for name in required_files if not (path / name).is_file()]
    if not (path / "model.safetensors").is_file() and not (
        path / "pytorch_model.bin"
    ).is_file():
        missing.append("model.safetensors or pytorch_model.bin")
    if missing:
        raise FileNotFoundError(
            f"本地模型缓存不完整：{path}；缺少 {', '.join(missing)}"
        )


@lru_cache(maxsize=1)
def _local_base_model_path() -> Path:
    """Resolve the one allowed BGE model from the local Hugging Face cache."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - environment contract
        raise RuntimeError("加载本地 BGE 需要安装 huggingface_hub；禁止 fallback") from exc
    try:
        path = Path(snapshot_download(MODEL_NAME, local_files_only=True)).resolve()
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(
            f"本地 Hugging Face 缓存中没有 {MODEL_NAME}；禁止联网下载或模型 fallback"
        ) from exc
    if not path.is_dir():
        raise FileNotFoundError(f"本地 BGE snapshot 不是目录：{path}")
    _assert_model_artifacts(path)
    return path


def load_model(
    model_name: str = MODEL_NAME,
    *,
    local_files_only: bool = True,
    device: str = "cpu",
):
    """加载唯一指定的 BGE 模型；缺文件时直接失败。"""

    if model_name != MODEL_NAME:
        raise TrainingContractError(
            f"C2 实验固定使用 {MODEL_NAME}，收到 {model_name!r}"
        )
    if local_files_only is not True:
        raise TrainingContractError("C2 模型只能从本地缓存加载；禁止联网 fallback")
    _, SentenceTransformer, _ = _sentence_transformers()
    return _load_sentence_transformer_from_path(
        SentenceTransformer, _local_base_model_path(), device=device
    )


def load_saved_model(path: str | Path, *, device: str = "cpu"):
    """重载本次训练保存的独立目录，不回退到基础模型。"""

    _, SentenceTransformer, _ = _sentence_transformers()
    path = Path(path)
    if not path.is_dir() or path.is_symlink():
        raise FileNotFoundError(f"训练输出目录不存在：{path}")
    _assert_model_artifacts(path)
    return _load_sentence_transformer_from_path(SentenceTransformer, path, device=device)


def build_training_examples(pairs: Sequence[Mapping[str, Any]]):
    """构造 query→positive 的 InputExample；不读取 answer 字段。"""

    InputExample, _, _ = _sentence_transformers()
    return [InputExample(texts=[row["query"], row["positive"]]) for row in pairs]


def build_unique_positive_batches(
    pairs: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    seed: int,
    known_positive_ids_by_query: Mapping[str, set[str]] | None = None,
) -> list[list[int]]:
    """让已知互为正例的 query 永远不在同一个 MNRL batch 中。"""

    if batch_size < 2:
        raise TrainingContractError("batch_size 必须至少为 2")
    if not pairs:
        raise TrainingContractError("MNRL batch 输入不能为空")
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(pairs):
        if not isinstance(row, Mapping):
            raise TrainingContractError(f"MNRL 第 {index + 1} 行必须是对象")
        query_id = row.get("query_id")
        evidence_id = row.get("positive_evidence_id")
        if not isinstance(query_id, str) or not query_id:
            raise TrainingContractError(f"MNRL 第 {index + 1} 行 query_id 无效")
        if not isinstance(evidence_id, str) or not evidence_id:
            raise TrainingContractError(
                f"MNRL 第 {index + 1} 行 positive_evidence_id 无效"
            )
        grouped.setdefault(evidence_id, []).append(index)
    rng = random.Random(seed)
    for indices in grouped.values():
        rng.shuffle(indices)
    batches: list[list[int]] = []
    for round_index in range(max(map(len, grouped.values()))):
        active = [
            evidence_id
            for evidence_id, indices in grouped.items()
            if round_index < len(indices)
        ]
        rng.shuffle(active)
        for start in range(0, len(active), batch_size):
            batch_ids = active[start : start + batch_size]
            batch = [grouped[evidence_id][round_index] for evidence_id in batch_ids]
            if len({pairs[index]["positive_evidence_id"] for index in batch}) != len(batch):
                raise TrainingContractError("MNRL batch 出现重复 positive_evidence_id")
            query_ids = [str(pairs[index]["query_id"]) for index in batch]
            if len(set(query_ids)) != len(query_ids):
                raise TrainingContractError(
                    "MNRL batch 出现重复 query_id，无法保证多正例安全"
                )
            if known_positive_ids_by_query is not None:
                positive_sets = [
                    _known_positive_set(
                        known_positive_ids_by_query,
                        str(pairs[index]["query_id"]),
                        str(pairs[index]["positive_evidence_id"]),
                    )
                    for index in batch
                ]
                for left in range(len(positive_sets)):
                    for right in range(left + 1, len(positive_sets)):
                        if positive_sets[left] & positive_sets[right]:
                            raise TrainingContractError(
                                "MNRL batch 把已知正例当成其他 query 的负例"
                            )
            batches.append(batch)
    if sorted(index for batch in batches for index in batch) != list(range(len(pairs))):
        raise TrainingContractError("MNRL batch 未恰好覆盖全部训练样本")
    return batches


def _known_positive_set(
    known_positive_ids_by_query: Mapping[str, Iterable[str]],
    query_id: str,
    own_positive_id: str,
) -> set[str]:
    """Normalize a known-positive map and always include the row's own ID."""

    raw = known_positive_ids_by_query.get(query_id)
    if raw is None:
        return {own_positive_id}
    if isinstance(raw, str):
        values = {raw}
    else:
        try:
            values = set(raw)
        except TypeError as exc:
            raise TrainingContractError(
                f"query {query_id} 的已知正例集合无效"
            ) from exc
    if any(not isinstance(value, str) or not value for value in values):
        raise TrainingContractError(f"query {query_id} 的已知正例 ID 无效")
    values.add(own_positive_id)
    return values


def train_embedding_model(
    pairs: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    model_name: str = MODEL_NAME,
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    seed: int = 42,
    device: str = "cpu",
    known_positive_ids_by_query: Mapping[str, set[str]] | None = None,
):
    """使用 MultipleNegativesRankingLoss 完成至少一轮真实训练并保存模型。"""

    output_path = _assert_empty_output_dir(output_dir)
    if epochs < 1:
        raise TrainingContractError("epochs 必须至少为 1")
    if batch_size < 2:
        raise TrainingContractError("MultipleNegativesRankingLoss 需要 batch_size 至少为 2")
    if not pairs:
        raise TrainingContractError("训练行不能为空")
    seed_everything(seed)
    InputExample, _, losses = _sentence_transformers()
    # Re-check row shape here so callers cannot bypass the pair contract by
    # constructing InputExample objects directly.
    checked = validate_finetune_pairs(pairs, require_reviewed=True)
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("训练需要 torch.utils.data.DataLoader，禁止 fallback") from exc
    model = load_model(model_name, local_files_only=True, device=device)
    _ensure_no_truncation(
        model,
        [text for row in checked for text in (row["query"], row["positive"])],
    )
    examples = [InputExample(texts=[row["query"], row["positive"]]) for row in checked]
    loss = losses.MultipleNegativesRankingLoss(model)
    torch = __import__("torch")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    started = time.perf_counter()
    model.train()
    loss_history: list[float] = []
    batch_count_per_epoch: list[int] = []
    for epoch in range(epochs):
        batch_sampler = build_unique_positive_batches(
            checked,
            batch_size=batch_size,
            seed=seed + epoch,
            known_positive_ids_by_query=known_positive_ids_by_query,
        )
        batch_count_per_epoch.append(len(batch_sampler))
        loader = DataLoader(
            examples,
            batch_sampler=batch_sampler,
            collate_fn=model.smart_batching_collate,
        )
        for sentence_features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            sentence_features = [
                {key: value.to(device) for key, value in features.items()}
                for features in sentence_features
            ]
            labels = labels.to(device)
            batch_loss = loss(sentence_features, labels)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_history.append(float(batch_loss.detach().cpu().item()))
    # SentenceTransformer.fit is unavailable without the optional datasets
    # package in v5.2.  The explicit PyTorch loop above is the same
    # MultipleNegativesRankingLoss objective and keeps this experiment's
    # dependency contract minimal and auditable.
    model.save(str(output_path), create_model_card=False)
    duration = time.perf_counter() - started
    if not (output_path / "config_sentence_transformers.json").is_file():
        raise RuntimeError(f"训练未在独立目录保存 SentenceTransformer 配置：{output_path}")
    if not loss_history:
        raise RuntimeError("训练没有产生 optimizer step；禁止保存空实验")
    return {
        "model": model,
        "output_dir": _portable_artifact_path(output_path),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_pair_count": len(checked),
        "duration_seconds": duration,
        "loss_first": loss_history[0],
        "loss_last": loss_history[-1],
        "optimizer_steps": len(loss_history),
        "unique_positive_per_batch": True,
        "train_unique_positive_count": len(
            {row["positive_evidence_id"] for row in checked}
        ),
        "batch_count_per_epoch": batch_count_per_epoch,
    }


def _ensure_no_truncation(model: Any, texts: Sequence[str]) -> list[int]:
    """Fail before encoding if any text would be truncated by the model."""

    if not texts:
        raise TrainingContractError("模型输入不能为空")
    max_length = getattr(model, "max_seq_length", None)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        try:
            first_module = model._first_module()
            tokenizer = getattr(first_module, "tokenizer", None)
            if max_length is None:
                max_length = getattr(first_module, "max_seq_length", None)
        except Exception as exc:  # noqa: BLE001
            raise TrainingContractError("无法读取指定模型 tokenizer/max_seq_length") from exc
    if tokenizer is None or not isinstance(max_length, int) or max_length <= 0:
        raise TrainingContractError("模型必须暴露 tokenizer 和 max_seq_length")
    try:
        encoded = tokenizer(
            list(texts),
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        input_ids = encoded.get("input_ids")
    except Exception as exc:  # noqa: BLE001
        raise TrainingContractError("无法用模型 tokenizer 检查输入长度") from exc
    if not isinstance(input_ids, list) or len(input_ids) != len(texts):
        raise TrainingContractError("模型 tokenizer 返回的 input_ids 行数不匹配")
    lengths = [len(row) for row in input_ids if isinstance(row, list)]
    if len(lengths) != len(texts):
        raise TrainingContractError("模型 tokenizer 返回了无效 input_ids")
    if any(length > max_length for length in lengths):
        raise TrainingContractError(
            f"输入超过模型 max_seq_length={max_length}；禁止静默截断"
        )
    return lengths


def encode_texts(model: Any, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
    """用给定模型编码文本；空 corpus 或编码失败都直接抛错。"""

    if batch_size < 1:
        raise TrainingContractError("编码 batch_size 必须至少为 1")
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise TrainingContractError("编码文本必须是非空字符串列表")
    _ensure_no_truncation(model, texts)
    try:
        encoded = model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("指定模型编码失败；禁止编码 fallback") from exc
    matrix = np.asarray(encoded, dtype=np.float32)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != len(texts)
        or matrix.shape[1] < 1
        or not np.isfinite(matrix).all()
    ):
        raise TrainingContractError("模型返回的 embedding 形状不匹配")
    return matrix


def _rank_metrics(
    query_id: str,
    ranked_ids: Sequence[str],
    relevant_ids: set[str],
    top_ks: Sequence[int],
) -> dict[str, Any]:
    first_rank: int | None = None
    for index, evidence_id in enumerate(ranked_ids, 1):
        if evidence_id in relevant_ids:
            first_rank = index
            break
    recalls = {
        str(k): len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)
        for k in top_ks
    }
    return {
        "query_id": query_id,
        "relevant_evidence_ids": sorted(relevant_ids),
        "first_relevant_rank": first_rank,
        "mrr": 0.0 if first_rank is None else 1.0 / first_rank,
        "recall_at_k": recalls,
        "ranked_evidence_ids": list(ranked_ids[: max(top_ks)]),
    }


def evaluate_embeddings(
    query_ids: Sequence[str],
    query_embeddings: np.ndarray,
    evidence_ids: Sequence[str],
    evidence_embeddings: np.ndarray,
    qrels: Sequence[Mapping[str, Any]],
    *,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
) -> dict[str, Any]:
    """基于已编码的完整 evidence corpus 计算 Recall@k、MRR 和逐题排名。"""

    if not top_ks or any(k < 1 for k in top_ks):
        raise TrainingContractError("top_ks 必须是正整数列表")
    if not query_ids or not evidence_ids:
        raise TrainingContractError("评估 query/evidence corpus 不能为空")
    top_ks = tuple(sorted(set(int(k) for k in top_ks)))
    if any(not isinstance(query_id, str) or not query_id.strip() for query_id in query_ids) or any(
        not isinstance(evidence_id, str) or not evidence_id.strip()
        for evidence_id in evidence_ids
    ):
        raise TrainingContractError("评估 query/evidence ID 不能为空")
    if len(set(query_ids)) != len(query_ids):
        raise TrainingContractError("评估 query_id 不能重复")
    if len(set(evidence_ids)) != len(evidence_ids):
        raise TrainingContractError("评估 evidence_id 不能重复")
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    evidence_embeddings = np.asarray(evidence_embeddings, dtype=np.float32)
    if (
        query_embeddings.ndim != 2
        or evidence_embeddings.ndim != 2
        or query_embeddings.shape[0] != len(query_ids)
        or evidence_embeddings.shape[0] != len(evidence_ids)
        or query_embeddings.shape[1] != evidence_embeddings.shape[1]
        or query_embeddings.shape[1] < 1
        or not np.isfinite(query_embeddings).all()
        or not np.isfinite(evidence_embeddings).all()
    ):
        raise TrainingContractError("embedding 行数与 ID 数量不一致")
    evidence_id_set = set(evidence_ids)
    relevant: dict[str, set[str]] = {query_id: set() for query_id in query_ids}
    query_id_set = set(query_ids)
    for row in qrels:
        query_id, evidence_id = row.get("query_id"), row.get("evidence_id")
        if query_id not in query_id_set:
            continue
        if evidence_id not in evidence_id_set:
            raise TrainingContractError(f"qrel 引用了未编码的 evidence：{evidence_id!r}")
        if row.get("relevance") == 1:
            relevant[query_id].add(evidence_id)
    missing = [query_id for query_id, refs in relevant.items() if not refs]
    if missing:
        raise TrainingContractError(f"评估 query 没有 relevance=1 qrel：{missing[:3]}")
    scores = np.matmul(np.asarray(query_embeddings), np.asarray(evidence_embeddings).T)
    order = np.argsort(-scores, axis=1, kind="stable")
    per_query: list[dict[str, Any]] = []
    for row_index, query_id in enumerate(query_ids):
        ranked = [evidence_ids[index] for index in order[row_index].tolist()]
        per_query.append(_rank_metrics(query_id, ranked, relevant[query_id], top_ks))
    metrics: dict[str, Any] = {
        "query_count": len(query_ids),
        "corpus_count": len(evidence_ids),
        "top_ks": list(top_ks),
        "mrr": float(np.mean([row["mrr"] for row in per_query])),
        "recall_at_k": {
            str(k): float(np.mean([row["recall_at_k"][str(k)] for row in per_query]))
            for k in top_ks
        },
        "per_query": per_query,
    }
    # Keep explicit labels in the persisted result so the report can be read
    # without guessing whether a key means Recall or a rank cutoff.
    for k, value in metrics["recall_at_k"].items():
        metrics[f"Recall@{k}"] = value
    return metrics


def evaluate_retrieval(
    model: Any,
    query_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    qrels: Sequence[Mapping[str, Any]],
    *,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
    batch_size: int = 32,
) -> dict[str, Any]:
    """重新编码 query 与完整 evidence corpus 后评估检索。"""

    if not query_rows or not evidence_rows:
        raise TrainingContractError("评估 query/evidence rows 不能为空")
    query_ids: list[str] = []
    query_texts: list[str] = []
    for index, row in enumerate(query_rows, 1):
        if not isinstance(row, Mapping):
            raise TrainingContractError(f"评估 query 第 {index} 行必须是对象")
        query_id, text = row.get("query_id"), row.get("text")
        if not isinstance(query_id, str) or not query_id.strip():
            raise TrainingContractError(f"评估 query 第 {index} 行 query_id 无效")
        if not isinstance(text, str) or not text.strip():
            raise TrainingContractError(f"评估 query 第 {index} 行 text 无效")
        query_ids.append(query_id)
        query_texts.append(text)
    evidence_ids: list[str] = []
    evidence_texts: list[str] = []
    for index, row in enumerate(evidence_rows, 1):
        if not isinstance(row, Mapping):
            raise TrainingContractError(f"评估 evidence 第 {index} 行必须是对象")
        evidence_id, quote = row.get("evidence_id"), row.get("quote")
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise TrainingContractError(f"评估 evidence 第 {index} 行 evidence_id 无效")
        if not isinstance(quote, str) or not quote.strip():
            raise TrainingContractError(f"评估 evidence 第 {index} 行 quote 无效")
        evidence_ids.append(evidence_id)
        evidence_texts.append(quote)
    query_embeddings = encode_texts(model, query_texts, batch_size=batch_size)
    evidence_embeddings = encode_texts(model, evidence_texts, batch_size=batch_size)
    return evaluate_embeddings(
        query_ids,
        query_embeddings,
        evidence_ids,
        evidence_embeddings,
        qrels,
        top_ks=top_ks,
    )


def compare_retrieval(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """逐题比较重载模型和原始 BGE 的 rank/Recall 变化。"""

    before_rows = {row["query_id"]: row for row in before.get("per_query", [])}
    after_rows = {row["query_id"]: row for row in after.get("per_query", [])}
    if set(before_rows) != set(after_rows):
        raise TrainingContractError("baseline 与 finetuned 的评估 query 集合不一致")
    per_query: list[dict[str, Any]] = []
    improved: list[str] = []
    degraded: list[str] = []
    tradeoff: list[str] = []
    unchanged: list[str] = []
    for query_id in before_rows:
        old, new = before_rows[query_id], after_rows[query_id]
        old_rank = old.get("first_relevant_rank")
        new_rank = new.get("first_relevant_rank")
        rank_delta = None if old_rank is None or new_rank is None else old_rank - new_rank
        old_recalls = old.get("recall_at_k", {})
        new_recalls = new.get("recall_at_k", {})
        recall_delta = {
            str(k): float(new_recalls.get(str(k), 0.0) - old_recalls.get(str(k), 0.0))
            for k in sorted(set(old_recalls) | set(new_recalls), key=int)
        }
        has_improvement = (rank_delta is not None and rank_delta > 0) or any(
            value > 0 for value in recall_delta.values()
        )
        has_degradation = (rank_delta is not None and rank_delta < 0) or any(
            value < 0 for value in recall_delta.values()
        )
        if has_improvement and has_degradation:
            tradeoff.append(query_id)
            status = "tradeoff"
        elif has_improvement:
            improved.append(query_id)
            status = "improved"
        elif has_degradation:
            degraded.append(query_id)
            status = "degraded"
        else:
            unchanged.append(query_id)
            status = "unchanged"
        per_query.append(
            {
                "query_id": query_id,
                "baseline_rank": old_rank,
                "finetuned_rank": new_rank,
                "rank_delta_positive_is_better": rank_delta,
                "recall_delta": recall_delta,
                "status": status,
            }
        )
    per_query.sort(key=lambda row: row["query_id"])
    return {
        "query_count": len(per_query),
        "improved_query_ids": sorted(improved),
        "degraded_query_ids": sorted(degraded),
        "tradeoff_query_ids": sorted(tradeoff),
        "unchanged_query_ids": sorted(unchanged),
        "improved_count": len(improved),
        "degraded_count": len(degraded),
        "tradeoff_count": len(tradeoff),
        "unchanged_count": len(unchanged),
        "per_query": per_query,
    }


def _query_split(
    package: Mapping[str, Any], split: str, *, query_ids: set[str] | None = None
) -> list[dict[str, Any]]:
    splits = package.get("splits")
    if not isinstance(splits, Mapping):
        raise TrainingContractError("package 缺少 splits")
    assignments = splits.get("assignments")
    if not isinstance(assignments, Mapping):
        raise TrainingContractError("package.splits.assignments 无效")
    ids = assignments.get(split, [])
    if not isinstance(ids, list):
        raise TrainingContractError(f"package split={split} assignments 无效")
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(package.get("queries", []), 1):
        if not isinstance(row, Mapping):
            raise TrainingContractError(f"package queries 第 {index} 行必须是对象")
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise TrainingContractError(f"package queries 第 {index} 行 query_id 无效")
        if query_id in by_id:
            raise TrainingContractError(f"package queries 重复 query_id：{query_id}")
        by_id[query_id] = row
    rows = [
        by_id[query_id]
        for query_id in ids
        if query_id in by_id and (query_ids is None or query_id in query_ids)
    ]
    if not rows:
        raise TrainingContractError(f"package split={split} 为空")
    return rows


def run_experiment(
    package: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    model_name: str = MODEL_NAME,
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    seed: int = 42,
    device: str = "cpu",
    selection_configs: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """仅用 dev 选择配置，再一次性打开 frozen test 做最终比较。"""

    # Reserve the experiment directory before touching data or models.  A
    # partially completed previous run is intentionally not resumable.
    output_path = _assert_empty_output_dir(output_dir)
    checked_pairs = validate_finetune_pairs(
        pairs,
        evidence_rows=package["evidence"],
        query_rows=package["queries"],
    )
    required_splits = ("train", "dev", "test")
    pairs_by_split = {
        split: [row for row in checked_pairs if row["split"] == split]
        for split in required_splits
    }
    missing_splits = [split for split in required_splits if not pairs_by_split[split]]
    if missing_splits:
        raise TrainingContractError(
            "finetune_pairs 每个 train/dev/test split 至少需要 1 条："
            f"缺少 {', '.join(missing_splits)}"
        )
    train_pairs = pairs_by_split["train"]
    pair_query_ids_by_split = {
        split: {row["query_id"] for row in rows}
        for split, rows in pairs_by_split.items()
    }
    splits = package.get("splits")
    assignments = splits.get("assignments") if isinstance(splits, Mapping) else None
    if not isinstance(assignments, Mapping):
        raise TrainingContractError("package.splits.assignments 缺失或无效")
    for split, query_ids in pair_query_ids_by_split.items():
        assigned = assignments.get(split)
        if not isinstance(assigned, list):
            raise TrainingContractError(f"package split={split} assignments 缺失或无效")
        if not query_ids.issubset(set(assigned)):
            raise TrainingContractError(
                f"finetune_pairs split={split} 的 query 未全部出现在 splits.json"
            )
    dev_queries = _query_split(
        package, "dev", query_ids=pair_query_ids_by_split["dev"]
    )
    test_queries = _query_split(
        package, "test", query_ids=pair_query_ids_by_split["test"]
    )
    qrels = package.get("qrels")
    if not isinstance(qrels, Sequence) or isinstance(qrels, (str, bytes)):
        raise TrainingContractError("package.qrels 必须是记录列表")
    experiment_evidence = [
        row
        for row in package["evidence"]
        if row.get("evidence_type") == "fixed_token_chunk"
    ]
    if not experiment_evidence:
        raise TrainingContractError("C2 实验缺少 fixed_token_chunk 检索语料")
    experiment_evidence_ids = {row["evidence_id"] for row in experiment_evidence}
    experiment_query_ids = set().union(*pair_query_ids_by_split.values())
    experiment_qrels = [
        row
        for row in qrels
        if row.get("query_id") in experiment_query_ids
        and row.get("evidence_id") in experiment_evidence_ids
    ]
    qrel_query_ids = {
        row.get("query_id")
        for row in experiment_qrels
        if row.get("relevance") == 1
    }
    missing_qrels = sorted(experiment_query_ids - qrel_query_ids)
    if missing_qrels:
        raise TrainingContractError(
            f"C2 query 在 fixed_token_chunk 语料中没有正例：{missing_qrels[:3]}"
        )
    base_model = load_model(model_name, local_files_only=True, device=device)
    baseline_dev = evaluate_retrieval(
        base_model,
        dev_queries,
        experiment_evidence,
        experiment_qrels,
        batch_size=batch_size,
    )
    del base_model

    if selection_configs is None:
        configs = [
            {"epochs": epochs, "learning_rate": 5e-6},
            {"epochs": epochs, "learning_rate": 1e-5},
            {"epochs": epochs, "learning_rate": learning_rate},
        ]
    else:
        configs = list(selection_configs)
    normalized_configs: list[dict[str, Any]] = []
    seen_configs: set[tuple[int, float]] = set()
    for config_index, config in enumerate(configs, 1):
        if not isinstance(config, Mapping):
            raise TrainingContractError(f"selection config 第 {config_index} 项必须是对象")
        try:
            candidate = {
                "epochs": int(config["epochs"]),
                "learning_rate": float(config["learning_rate"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise TrainingContractError(
                f"selection config 第 {config_index} 项缺少有效 epochs/learning_rate"
            ) from exc
        if (
            candidate["epochs"] < 1
            or not math.isfinite(candidate["learning_rate"])
            or candidate["learning_rate"] <= 0
        ):
            raise TrainingContractError(
                f"selection config 第 {config_index} 项超出训练参数范围"
            )
        key = (candidate["epochs"], candidate["learning_rate"])
        if key not in seen_configs:
            seen_configs.add(key)
            normalized_configs.append(candidate)
    if not normalized_configs:
        raise TrainingContractError("dev selection 至少需要一个训练配置")

    candidates: list[dict[str, Any]] = []
    train_query_ids = pair_query_ids_by_split["train"]
    train_positive_ids_by_query: dict[str, set[str]] = {
        query_id: set() for query_id in train_query_ids
    }
    for row in experiment_qrels:
        if not isinstance(row, Mapping) or row.get("relevance") != 1:
            continue
        query_id, evidence_id = row.get("query_id"), row.get("evidence_id")
        if query_id not in train_query_ids:
            continue
        if not isinstance(query_id, str) or not isinstance(evidence_id, str):
            raise TrainingContractError("qrels 的 train 正例 ID 必须是非空字符串")
        if not query_id or not evidence_id:
            raise TrainingContractError("qrels 的 train 正例 ID 不能为空")
        train_positive_ids_by_query[query_id].add(evidence_id)
    if any(not values for values in train_positive_ids_by_query.values()):
        missing = sorted(
            query_id
            for query_id, values in train_positive_ids_by_query.items()
            if not values
        )
        raise TrainingContractError(
            f"train query 没有 relevance=1 qrel，无法保护多正例：{missing[:3]}"
        )
    for index, config in enumerate(normalized_configs, 1):
        candidate_dir = output_path / f"candidate_{index:02d}"
        training = train_embedding_model(
            train_pairs,
            candidate_dir,
            model_name=model_name,
            epochs=config["epochs"],
            batch_size=batch_size,
            learning_rate=config["learning_rate"],
            seed=seed,
            device=device,
            known_positive_ids_by_query=train_positive_ids_by_query,
        )
        training.pop("model", None)
        # ``candidate_dir`` is retained as a local Path for loading.  The
        # result and progress files only carry a stable relative reference.
        training["output_dir"] = _portable_artifact_path(candidate_dir)
        reloaded = load_saved_model(candidate_dir, device=device)
        dev_metrics = evaluate_retrieval(
            reloaded,
            dev_queries,
            experiment_evidence,
            experiment_qrels,
            batch_size=batch_size,
        )
        del reloaded
        candidates.append(
            {
                "candidate_id": index,
                "config": config,
                "model_dir": candidate_dir.relative_to(output_path).as_posix(),
                "training": training,
                "dev": dev_metrics,
                "dev_comparison": compare_retrieval(baseline_dev, dev_metrics),
            }
        )
        _write_json(output_path / "dev_selection_progress.json", candidates)

    def selection_key(candidate: Mapping[str, Any]) -> tuple[float, ...]:
        metrics = candidate["dev"]
        return (
            float(metrics["Recall@10"]),
            float(metrics["Recall@5"]),
            float(metrics["Recall@3"]),
            float(metrics["Recall@1"]),
            float(metrics["mrr"]),
        )

    selected = max(candidates, key=selection_key)
    base_model = load_model(model_name, local_files_only=True, device=device)
    baseline_test = evaluate_retrieval(
        base_model,
        test_queries,
        experiment_evidence,
        experiment_qrels,
        batch_size=batch_size,
    )
    del base_model
    selected_model = load_saved_model(
        output_path / selected["model_dir"],
        device=device,
    )
    finetuned_test = evaluate_retrieval(
        selected_model,
        test_queries,
        experiment_evidence,
        experiment_qrels,
        batch_size=batch_size,
    )
    del selected_model
    finetuned_dev = selected["dev"]
    training = selected["training"]
    result = {
        "experiment": "c2_query_to_pumpkin_evidence",
        "model_name": model_name,
        "dataset": {
            "dataset_id": package.get("manifest", {}).get("dataset_id")
            if isinstance(package.get("manifest"), Mapping)
            else None,
            "dataset_version": package.get("manifest", {}).get("dataset_version")
            if isinstance(package.get("manifest"), Mapping)
            else None,
        },
        "positive_field": "positive=canonical evidence.quote",
        "retrieval_corpus": "canonical fixed_token_chunk only",
        "retrieval_corpus_size": len(experiment_evidence),
        "corpus_reencoded_after_reload": True,
        "selection_policy": "dev_only_lexicographic_recall10_5_3_1_then_mrr",
        "frozen_test_opened_after_selection": True,
        "pair_count_by_split": {
            split: len(rows) for split, rows in pairs_by_split.items()
        },
        "selection_candidates": candidates,
        "selected_candidate_id": selected["candidate_id"],
        "selected_model_dir": selected["model_dir"],
        "train_pair_count": len(train_pairs),
        "dev_query_count": len(dev_queries),
        "frozen_test_query_count": len(test_queries),
        "training": training,
        "baseline": {"dev": baseline_dev, "frozen_test": baseline_test},
        "finetuned": {"dev": finetuned_dev, "frozen_test": finetuned_test},
        "comparison": {
            "dev": compare_retrieval(baseline_dev, finetuned_dev),
            "frozen_test": compare_retrieval(baseline_test, finetuned_test),
        },
    }
    _write_json(output_path / "experiment_results.json", result)
    return result
