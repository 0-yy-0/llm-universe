"""Run the C2 query-to-Pumpkin-Book evidence fine-tuning experiment.

The command deliberately has one model/data path: BAAI/bge-small-zh-v1.5 and
the canonical dataset package.  A missing dependency, model file, evidence
reference, or output artifact is an error rather than a fallback condition.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).resolve()
COURSE_ROOT = SCRIPT.parents[1]
REPO_ROOT = COURSE_ROOT.parents[1]
if str(COURSE_ROOT) not in sys.path:
    sys.path.insert(0, str(COURSE_ROOT))

from common.dataset import load_dataset  # noqa: E402
from common.training_utils import (  # noqa: E402
    MODEL_NAME,
    load_finetune_pairs,
    run_experiment,
)


def _portable_reference(path: str | Path) -> str:
    """Keep persisted CLI summaries independent of the local checkout path."""

    value = Path(path).resolve()
    try:
        return value.relative_to(COURSE_ROOT.resolve()).as_posix()
    except ValueError:
        return value.name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=COURSE_ROOT / "data" / "dataset",
        help="canonical dataset root (default: notebook/C7 高级 RAG 技巧/data/dataset)",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=None,
        help="canonical query-to-evidence pair file (default: dataset-root/finetune_pairs.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "independent empty model/result directory; omitted means create a "
            "fresh unique directory under C7/.cache"
        ),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = args.dataset_root.resolve()
    pairs_path = (
        args.pairs if args.pairs is not None else dataset_root / "finetune_pairs.jsonl"
    ).resolve()
    canonical_pairs_path = (dataset_root / "finetune_pairs.jsonl").resolve()
    if pairs_path != canonical_pairs_path:
        raise ValueError(
            "C2 训练只允许使用 dataset-root/finetune_pairs.jsonl；"
            "禁止切换到其他 pair 文件"
        )
    if args.output_dir is None:
        cache_root = COURSE_ROOT / ".cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        output_dir = Path(
            tempfile.mkdtemp(prefix="c2_embedding_finetune_", dir=str(cache_root))
        ).resolve()
    else:
        output_dir = args.output_dir.resolve()
    package = load_dataset(dataset_root)
    pairs = load_finetune_pairs(
        pairs_path,
        evidence_rows=package["evidence"],
        query_rows=package["queries"],
    )
    result = run_experiment(
        package,
        pairs,
        output_dir,
        model_name=MODEL_NAME,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device="cpu",
    )
    comparison = result["comparison"]["frozen_test"]
    candidate_dev = [
        {
            "candidate_id": candidate["candidate_id"],
            "config": candidate["config"],
            "model_dir": candidate["model_dir"],
            "Recall@1": candidate["dev"]["Recall@1"],
            "Recall@5": candidate["dev"]["Recall@5"],
            "Recall@10": candidate["dev"]["Recall@10"],
            "MRR": candidate["dev"]["mrr"],
        }
        for candidate in result["selection_candidates"]
    ]
    summary = {
        "experiment": result["experiment"],
        "model_name": result["model_name"],
        "output_dir": _portable_reference(output_dir),
        "train_pair_count": result["train_pair_count"],
        "dev_query_count": result["dev_query_count"],
        "frozen_test_query_count": result["frozen_test_query_count"],
        "candidate_dev": candidate_dev,
        "selected_candidate_id": result["selected_candidate_id"],
        "selected_model_dir": result["selected_model_dir"],
        "selected_dev": {
            "Recall@1": result["finetuned"]["dev"]["Recall@1"],
            "Recall@5": result["finetuned"]["dev"]["Recall@5"],
            "Recall@10": result["finetuned"]["dev"]["Recall@10"],
            "MRR": result["finetuned"]["dev"]["mrr"],
        },
        "training_seconds": result["training"].get("duration_seconds"),
        "baseline_frozen_test": {
            "Recall@1": result["baseline"]["frozen_test"]["Recall@1"],
            "Recall@5": result["baseline"]["frozen_test"]["Recall@5"],
            "MRR": result["baseline"]["frozen_test"]["mrr"],
        },
        "finetuned_frozen_test": {
            "Recall@1": result["finetuned"]["frozen_test"]["Recall@1"],
            "Recall@5": result["finetuned"]["frozen_test"]["Recall@5"],
            "MRR": result["finetuned"]["frozen_test"]["mrr"],
        },
        "frozen_test_improved": comparison["improved_count"],
        "frozen_test_degraded": comparison["degraded_count"],
        "frozen_test_unchanged": comparison["unchanged_count"],
        "results_file": "experiment_results.json",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
