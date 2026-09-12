"""教程 Notebook 共用的本地读取、检索与结果检查函数。"""

from __future__ import annotations

import math
import re
import warnings
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from . import dataset as dataset_store


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = dataset_store.DATASET_ROOT
# 保留现有 Notebook 使用的常量名；它指向统一数据包中的 canonical JSONL。
PDF_PATH = ROOT / "data" / "pumpkin_book.pdf"
DEFAULT_VECTOR_PATH = ROOT / "data" / "向量库" / "南瓜书配套库"
BGE_MODEL_ID = "BAAI/bge-small-zh-v1.5"
TOKEN_RE = re.compile(r"[一-鿿]|[a-z0-9]+", re.IGNORECASE)
CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]+")
PDF_PRINT_AD = re.compile(
    r"→_→\s*欢迎去各大电商平台选购纸质版南瓜书《机器学习公式详解》\s*←_←"
)
PDF_VIDEO_AD = re.compile(
    r"→_→\s*配套视频教程：https?://www\.bilibili\.com/video/[^\s←]+\s*←_←"
)

TUTORIAL_AUDIT_MIME = "application/vnd.llm-universe.tutorial-audit+json"

def emit_tutorial_audit(payload: dict) -> None:
    """把本次运行的审计记录写入 Notebook 的结构化输出。

    前端没有这个 MIME 的 renderer 时不会渲染长 JSON；记录仍会随
    ``display_data`` 保存在 ipynb 中，检查脚本可以直接读取它。普通
    Python 脚本没有 IPython display 环境时不打印机器记录。
    """

    if not isinstance(payload, dict):
        raise TypeError("教程审计记录必须是字典")
    try:
        from IPython import get_ipython
        from IPython.display import display
    except ImportError:
        return
    if get_ipython() is None:
        return
    display({TUTORIAL_AUDIT_MIME: payload}, raw=True)


@dataclass
class Evidence:
    page: int
    text: str
    score: float
    # 页级检索仍可使用前三个参数；片段检索额外保留身份，避免同页片段互相覆盖。
    chunk_id: str | None = None


@dataclass
class ChunkEvidence:
    chunk_id: str
    pages: list[int]
    text: str
    score: float


def normalize_text(text: object) -> str:
    value = CONTROL_CHARS.sub(" ", str(text or ""))
    value = PDF_PRINT_AD.sub(" ", value)
    value = PDF_VIDEO_AD.sub(" ", value)
    return re.sub(r"\s+", " ", value).strip()


def load_pdf_pages() -> list[dict]:
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"没有找到 PDF：{PDF_PATH}")
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("读取 PDF 需要安装 pymupdf") from exc

    document = fitz.open(str(PDF_PATH))
    pages = []
    try:
        for number, page in enumerate(document, start=1):
            text = normalize_text(page.get_text("text"))
            if not text:
                continue
            blocks = [
                normalize_text(block[4])
                for block in page.get_text("blocks")
                if len(block) > 4 and normalize_text(block[4])
            ]
            pages.append({"page": number, "text": text, "blocks": blocks or [text]})
    finally:
        document.close()
    if not pages:
        raise ValueError("PDF 没有可读取的文本")
    return pages


def load_query_catalog() -> list[dict[str, str]]:
    """Return all query text/IDs without loading evaluation annotations."""

    return dataset_store.load_query_catalog()


def load_query_controls(case_id: str) -> dict:
    """Return query-flow controls without answer-evaluation annotations."""

    return dataset_store.load_query_controls(case_id)


def find_local_bge_model() -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("解析本地 BGE 缓存需要安装 huggingface_hub") from exc
    try:
        cached = Path(snapshot_download(BGE_MODEL_ID, local_files_only=True))
    except Exception as exc:  # noqa: BLE001 - 转成明确的单一路径契约错误
        raise FileNotFoundError(f"本地 Hugging Face 缓存中没有 {BGE_MODEL_ID}") from exc
    if not (cached / "config.json").is_file() or not (cached / "vocab.txt").is_file():
        raise FileNotFoundError(f"本地模型缓存不完整：{cached}")
    return cached


def load_default_collection():
    """打开教程随附的向量库，不创建空库，也不重新计算向量。"""
    if not (DEFAULT_VECTOR_PATH / "chroma.sqlite3").is_file():
        raise FileNotFoundError(f"没有找到教程向量库：{DEFAULT_VECTOR_PATH}")
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as exc:
        raise RuntimeError("读取教程向量库需要安装 chromadb==1.5.5") from exc

    version = tuple(int(part) for part in chromadb.__version__.split(".")[:3])
    if version != (1, 5, 5):
        raise RuntimeError("请在独立环境中安装 chromadb==1.5.5。")
    client = chromadb.PersistentClient(
        path=str(DEFAULT_VECTOR_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection("nb_ctx", embedding_function=None)
    if not collection.count():
        raise ValueError("教程向量库没有片段，请重新下载。")
    return collection


def load_default_chunks(collection=None) -> list[dict]:
    """读取已经分好的片段；显示页码从 1 开始。"""
    collection = collection or load_default_collection()
    data = collection.get(include=["documents", "metadatas"])
    rows = [
        {
            "chunk_id": chunk_id,
            "pages": [int(metadata["page"]) + 1],
            "text": normalize_text(text),
        }
        for chunk_id, text, metadata in zip(
            data["ids"], data["documents"], data["metadatas"]
        )
    ]
    return sorted(rows, key=lambda row: int(row["chunk_id"][1:]))


def build_default_chunk_search(
    collection=None, model=None
) -> Callable[[str, int], list[ChunkEvidence]]:
    """复用保存好的文档向量，只计算新问题的向量。"""
    collection = collection or load_default_collection()
    if model is None:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="IProgress not found.*",
                )
                from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("向量检索需要安装 sentence-transformers") from exc
        # 模型必须已经存在于本地环境；不以模型 ID 触发下载或切换到另一
        # 个来源，缺失时让 find_local_bge_model 的错误直接暴露。
        model = SentenceTransformer(str(find_local_bge_model()), device="cpu")

    def search(query: str, top_k: int = 4) -> list[ChunkEvidence]:
        if top_k <= 0:
            return []
        vector = model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )
        data = collection.query(
            query_embeddings=vector.tolist(),
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            ChunkEvidence(
                chunk_id,
                [int(metadata["page"]) + 1],
                normalize_text(text),
                1 - float(distance),
            )
            for chunk_id, metadata, text, distance in zip(
                data["ids"][0],
                data["metadatas"][0],
                data["documents"][0],
                data["distances"][0],
            )
        ]

    return search


def _page_number(item: object) -> int:
    if isinstance(item, Evidence):
        return item.page
    if isinstance(item, dict):
        return int(item.get("page", 0))
    return int(getattr(item, "page", 0))


def _page_text(item: object) -> str:
    if isinstance(item, Evidence):
        return item.text
    if isinstance(item, dict):
        return str(item.get("text", ""))
    return str(getattr(item, "text", ""))


def _lexical_tokens(text: object) -> list[str]:
    return [match.lower() for match in TOKEN_RE.findall(normalize_text(text))]


def build_bm25_search(
    pages: Sequence[dict], k1: float = 1.5, b: float = 0.75
) -> Callable[[str, int], list[Evidence]]:
    rows = list(pages)
    if not rows:
        raise ValueError("语料不能为空")
    tokens = [_lexical_tokens(_page_text(row)) for row in rows]
    frequencies = [Counter(item) for item in tokens]
    document_frequency = Counter(term for item in tokens for term in set(item))
    lengths = [len(item) for item in tokens]
    average_length = sum(lengths) / len(lengths)

    def search(query: str, top_k: int = 5) -> list[Evidence]:
        if top_k <= 0:
            return []
        scores = []
        for index, frequency in enumerate(frequencies):
            score = 0.0
            for term in _lexical_tokens(query):
                count = frequency.get(term, 0)
                if not count:
                    continue
                inverse = math.log(
                    1
                    + (len(rows) - document_frequency[term] + 0.5)
                    / (document_frequency[term] + 0.5)
                )
                denominator = count + k1 * (
                    1 - b + b * lengths[index] / max(average_length, 1e-9)
                )
                score += inverse * count * (k1 + 1) / denominator
            scores.append((index, score))
        ranked = sorted(scores, key=lambda pair: (-pair[1], pair[0]))[:top_k]
        return [
            Evidence(_page_number(rows[index]), _page_text(rows[index]), float(score))
            for index, score in ranked
        ]

    return search


def page_coverage(
    expected_pages: Iterable[int], evidence: Iterable[Evidence]
) -> tuple[int, float, list[int]]:
    expected = {int(page) for page in expected_pages}
    found = {_page_number(item) for item in evidence}
    hits = sorted(expected & found)
    rate = len(hits) / len(expected) if expected else math.nan
    return len(hits), rate, hits


def precision_at_k(relevant: Iterable[object], k: int) -> float:
    """返回前 ``k`` 条中的相关比例。

    若结果少于 k，分母是实际返回条数（而不是 k）；因此这是“已返回结果的
    precision@k”，不会因为检索器只返回一条就凭空制造未返回的负例。
    """
    if k <= 0:
        return 0.0
    values = list(relevant)[:k]
    return sum(bool(value) for value in values) / len(values) if values else 0.0


def recall_at_k(relevant: Iterable[object], total_relevant: int, k: int) -> float:
    """前 k 条召回的相关证据数 / 评估集中的全部相关证据数。"""
    if total_relevant <= 0:
        raise ValueError("total_relevant 必须为正数")
    if k <= 0:
        return 0.0
    hits = sum(bool(value) for value in list(relevant)[:k])
    if hits > total_relevant:
        raise ValueError("total_relevant 不能小于已命中的相关项数")
    return hits / total_relevant


def average_precision(relevant: Iterable[object], total_relevant: int) -> float:
    """按全部相关证据归一化的 AP；漏召回时不会仍为 1。

    ``total_relevant`` 必须来自完整评估标注，不能用当前返回结果中的命中数猜测。
    """
    values = list(relevant)
    if total_relevant <= 0:
        raise ValueError("total_relevant 必须为正数")
    hits = 0
    area = 0.0
    for rank, flag in enumerate(values, start=1):
        if bool(flag):
            hits += 1
            area += hits / rank
    if hits > total_relevant:
        raise ValueError("total_relevant 不能小于已命中的相关项数")
    return area / total_relevant


def make_fixed_chunks(
    pages: Sequence[dict], chunk_size: int = 260, overlap: int = 0
) -> list[dict]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size 必须为正，overlap 必须小于 chunk_size")
    step = chunk_size - overlap
    chunks = []
    for page in pages:
        text = _page_text(page)
        for index, start in enumerate(range(0, len(text), step), start=1):
            part = normalize_text(text[start : start + chunk_size])
            if part:
                chunks.append(
                    {
                        "chunk_id": f"p{_page_number(page)}_fixed_{index}",
                        "pages": [_page_number(page)],
                        "text": part,
                    }
                )
    return chunks


def make_recursive_chunks(
    pages: Sequence[dict], chunk_size: int = 260, overlap: int = 0
) -> list[dict]:
    """优先在句尾收块，并在同页相邻片段间重复 ``overlap`` 个字符。

    字符数以 ``normalize_text`` 后的页文本为准，片段是它的连续切片。
    下一块从上一块末尾回退 overlap 个字符，所以重叠前缀可能从句中
    开始。若长度上限内没有能继续前进的句尾，则按字符硬切；页间不重叠。
    """

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size 必须为正，overlap 必须小于 chunk_size")
    chunks = []
    for page in pages:
        text = normalize_text(_page_text(page))
        sentence_ends = []
        cursor = 0
        for sentence in split_sentences(text):
            cursor = text.index(sentence, cursor) + len(sentence)
            sentence_ends.append(cursor)

        start = 0
        number = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary_index = bisect_right(sentence_ends, end) - 1
                # 必须在重叠前缀之外加入新字符，避免短句或大 overlap 导致循环。
                if boundary_index >= 0 and sentence_ends[boundary_index] > start + overlap:
                    end = sentence_ends[boundary_index]
            part = text[start:end]
            if part.strip():
                number += 1
                chunks.append(
                    {
                        "chunk_id": f"p{_page_number(page)}_sentence_{number}",
                        "pages": [_page_number(page)],
                        "text": part,
                    }
                )
            if end == len(text):
                break
            start = end - overlap
    return chunks


def split_sentences(text: object) -> list[str]:
    """中文句末无需空格；英文句末需空格，避免拆开小数等内容。"""

    return [
        part.strip()
        for part in re.split(r"(?<=[。！？])\s*|(?<=[.!?])\s+", str(text or ""))
        if part.strip()
    ]


def build_bm25_chunk_search(
    chunks: Sequence[dict], k1: float = 1.5, b: float = 0.75
) -> Callable[[str, int], list[ChunkEvidence]]:
    rows = list(chunks)
    search_pages = build_bm25_search(
        [{"page": index, "text": row.get("text", "")} for index, row in enumerate(rows)],
        k1,
        b,
    )

    def search(query: str, top_k: int = 5) -> list[ChunkEvidence]:
        return [
            ChunkEvidence(
                rows[item.page]["chunk_id"],
                list(rows[item.page].get("pages", [])),
                rows[item.page].get("text", ""),
                item.score,
            )
            for item in search_pages(query, top_k)
        ]

    return search


__all__ = [
    "ChunkEvidence",
    "DATASET_ROOT",
    "Evidence",
    "TUTORIAL_AUDIT_MIME",
    "build_bm25_chunk_search",
    "build_bm25_search",
    "build_default_chunk_search",
    "find_local_bge_model",
    "load_default_chunks",
    "load_default_collection",
    "load_pdf_pages",
    "load_query_catalog",
    "load_query_controls",
    "make_fixed_chunks",
    "make_recursive_chunks",
    "split_sentences",
    "precision_at_k",
    "recall_at_k",
    "average_precision",
    "page_coverage",
    "emit_tutorial_audit",
]
