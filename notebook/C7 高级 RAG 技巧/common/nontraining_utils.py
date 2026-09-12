"""无需训练的 RAG 方法共用的小工具。

这里的函数有意把问题读取和评估标注读取分开：检索、改写和生成阶段只
读取 ``id``、``query``，结果产生后才读取 expected_pages 等评估字段。
"""

from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from . import dataset as dataset_store
from .eval_utils import (
    BGE_MODEL_ID,
    ChunkEvidence,
    Evidence,
    build_default_chunk_search,
    build_bm25_chunk_search,
    build_bm25_search,
    emit_tutorial_audit,
    find_local_bge_model,
    load_pdf_pages,
    normalize_text,
)


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
PROJECT_ENV_PATH = PROJECT_ROOT / ".env"
PROJECT_ENV_RELATIVE_PATH = ".env"
DATASET_ROOT = dataset_store.DATASET_ROOT
RAG_LLM_MODEL = "glm-4-flash"


def load_zhipuai_api_key() -> str:
    """严格读取项目根目录 ``.env`` 中的智谱 API 密钥。

    不把配置注入进进程环境，也不读取 shell 环境变量；所有直接调用方都应
    通过这个 helper 获取密钥。缺少 dotenv、配置文件或非空密钥时直接抛错。
    """

    from dotenv import dotenv_values

    path = PROJECT_ENV_PATH
    if not path.is_file():
        raise FileNotFoundError(f"没有找到项目根目录 .env：{path}")
    values = dotenv_values(path, interpolate=False)
    api_key = values.get("ZHIPUAI_API_KEY")
    if not isinstance(api_key, str) or not api_key.strip():
        raise RuntimeError(
            "项目根目录 .env 没有设置非空 ZHIPUAI_API_KEY，无法执行真实生成"
        )
    return api_key.strip()


def load_query_only(case_ids: Sequence[str] | None = None) -> list[dict[str, str]]:
    """只投影读取案例 id 和 query，供检索/生成阶段使用。"""

    return dataset_store.load_query_only(case_ids)


def load_query_catalog() -> list[dict[str, str]]:
    """Return every query as the same annotation-free projection."""

    return dataset_store.load_query_catalog()


def load_query_controls(case_id: str) -> dict:
    """读取执行查询所需的交互/权限控制，不带答案评估字段。"""

    return dataset_store.load_query_controls(case_id)


def load_annotation(case_id: str) -> dict:
    """在结果产生后读取评估标注；调用位置应放在 pipeline 之后。"""

    return dataset_store.load_annotation(case_id)


def llm_call(
    prompt: str,
    *,
    max_tokens: int = 900,
    retries: int = 1,
) -> str:
    """使用项目根目录密钥发出一次同步模型请求。

    ``retries`` 仅作为显式保护参数：传入大于一的值会在发出请求前失败。
    SDK 自身也固定为 ``max_retries=0``，因此
    一次 Python 调用至多对应一次 HTTP 请求；解析或服务错误会原样抛给
    调用方，由调用方决定是否停止，而不是在这里悄悄重放请求。
    """

    if (
        not isinstance(retries, int)
        or isinstance(retries, bool)
        or retries != 1
    ):
        raise ValueError("llm_call 只允许 retries=1；不支持应用层重试")

    api_key = load_zhipuai_api_key()
    from zhipuai import ZhipuAI

    client = ZhipuAI(api_key=api_key, max_retries=0)
    response = client.chat.completions.create(
        model=RAG_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=60,
    )
    content = response.choices[0].message.content
    if not content or not str(content).strip():
        raise RuntimeError("模型返回空文字")
    return str(content).strip()


def parse_json_object(raw: str) -> dict:
    """严格解析完整 JSON 对象，可选完整 Markdown JSON 围栏。"""

    text = str(raw or "").strip()
    if text.startswith("```"):
        fenced = re.fullmatch(
            r"```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fenced is None:
            raise ValueError("模型没有返回完整 JSON 围栏")
        text = fenced.group("body").strip()
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("模型没有返回合法的完整 JSON 对象") from error
    if not isinstance(value, dict):
        raise ValueError("模型返回的不是 JSON 对象")
    return value


def parse_json_list(raw: str, key: str) -> list[str]:
    value = parse_json_object(raw)
    items = value.get(key)
    if not isinstance(items, list):
        raise ValueError(f"模型 JSON 缺少列表字段 {key!r}：{raw!r}")
    result = []
    for raw_item in items:
        item = raw_item
        if isinstance(raw_item, dict):
            # 规划器有时会把子问题写成 {"subquestion": "..."}，提取其文字而非把字典转成检索词。
            item = raw_item.get("subquestion", raw_item.get("query", raw_item.get("question", "")))
            if not item:
                # 兼容模型用“问题1”“问题2”等标签包装字符串的 JSON。
                item = next((candidate for candidate in raw_item.values()
                             if isinstance(candidate, str)), "")
        text = str(item).strip()
        if text:
            result.append(text)
    if not result:
        raise ValueError(f"模型返回的 {key!r} 为空：{raw!r}")
    return result


def build_page_dense_search(pages: Sequence[dict] | None = None):
    """用本地 BGE 为 PDF 页建立一次页级 dense 检索。"""

    rows = list(load_pdf_pages() if pages is None else pages)
    if not rows:
        raise ValueError("pages 不能为空；显式传入空序列不会回退到全量 PDF")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="IProgress not found.*")
        from sentence_transformers import SentenceTransformer

    # 只使用已安装的本地模型；缺失时让 find_local_bge_model 直接报错。
    model = SentenceTransformer(str(find_local_bge_model()), device="cpu")
    texts = [str(row.get("text", "")) for row in rows]
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    matrix = np.asarray(vectors, dtype=np.float32)

    def search(query: str, top_k: int = 5) -> list[Evidence]:
        if top_k <= 0:
            return []
        vector = np.asarray(
            model.encode([str(query)], normalize_embeddings=True, show_progress_bar=False)[0],
            dtype=np.float32,
        )
        scores = matrix @ vector
        order = np.argsort(-scores, kind="stable")[: min(top_k, len(rows))]
        return [
            Evidence(
                int(rows[index]["page"]),
                str(rows[index].get("text", "")),
                float(scores[index]),
            )
            for index in order
        ]

    return search


def build_reused_chunk_search(collection=None):
    """复用教程随附的向量库，只为新问题计算向量。"""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="IProgress not found.*")
        chunk_search = build_default_chunk_search(collection=collection)

    def search(query: str, top_k: int = 5) -> list[Evidence]:
        if top_k <= 0:
            return []
        # 向量库按片段存储；保留 chunk_id，不能因同页而丢掉互补证据。
        chunks: list[ChunkEvidence] = chunk_search(query, top_k=min(top_k, 987))
        result: list[Evidence] = []
        for chunk in chunks:
            if not chunk.pages:
                continue
            page = int(chunk.pages[0])
            result.append(Evidence(page, str(chunk.text), float(chunk.score), chunk.chunk_id))
            if len(result) >= top_k:
                break
        return result

    return search


def build_bm25_page_search(pages: Sequence[dict] | None = None):
    rows = list(load_pdf_pages() if pages is None else pages)
    if not rows:
        raise ValueError("pages 不能为空；显式传入空序列不会回退到全量 PDF")
    return build_bm25_search(rows)


def evidence_payload(items: Iterable[Evidence], chars: int = 420) -> list[dict]:
    return [
        {
            "page": int(item.page),
            **({"chunk_id": item.chunk_id} if item.chunk_id else {}),
            "score": round(float(item.score), 5),
            "text": re.sub(r"\s+", " ", item.text).strip()[:chars],
        }
        for item in items
    ]


def unique_evidence(items: Iterable[Evidence], limit: int = 8) -> list[Evidence]:
    seen: set[object] = set()
    result: list[Evidence] = []
    for item in items:
        # 有 chunk 身份时按 chunk 去重；页级 Evidence 按页和文本去重。
        key = (
            ("chunk", item.chunk_id)
            if item.chunk_id
            else ("page_text", int(item.page), normalize_text(item.text))
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
        if len(result) >= limit:
            break
    return result


def rank_and_coverage(items: Iterable[Evidence], expected_pages: Iterable[int]) -> dict:
    pages = [int(item.page) for item in items]
    expected = {int(page) for page in expected_pages}
    found = sorted(expected & set(pages))
    first_rank = next((index for index, page in enumerate(pages, 1) if page in expected), None)
    return {
        "pages": pages,
        "first_required_rank": first_rank,
        "required_pages_found": found,
        "required_page_coverage": len(found) / len(expected) if expected else math.nan,
    }


def print_method_report(record: dict) -> None:
    """保存改前/改后及真实生成产物；普通 Notebook 输出保持简洁。"""

    annotation = record["annotation"]
    before = rank_and_coverage(record["before"], annotation["expected_pages"])
    after = rank_and_coverage(record["after"], annotation["expected_pages"])
    emit_tutorial_audit({
        "case_id": record["case_id"],
        "query": record["query"],
        "before": before,
        "after": after,
        "model_outputs": record.get("model_outputs", {}),
        "annotation_check_after_result": {
            "expected_pages": annotation["expected_pages"],
            "evidence_pages": sorted({
                int(span.get("page"))
                for claim in annotation["essential_evidence_spans"]
                if isinstance(claim, dict)
                for span in claim.get("spans", [])
                if isinstance(span, dict) and str(span.get("page", "")).isdigit()
            }),
        },
    })


def answer_prompt(question: str, context: str) -> str:
    return (
        "仅根据下面的资料回答问题；资料没有支持的内容就明确说资料不足，不能补充外部知识。\n\n"
        f"问题：{question}\n资料：\n{context}"
    )


def format_context(items: Iterable[Evidence], max_chars: int = 6000) -> str:
    text = "\n\n".join(f"[第 {item.page} 页]\n{item.text}" for item in items)
    return text[:max_chars]


__all__ = [
    "DATASET_ROOT",
    "PROJECT_ENV_PATH",
    "PROJECT_ENV_RELATIVE_PATH",
    "PROJECT_ROOT",
    "RAG_LLM_MODEL",
    "answer_prompt",
    "build_bm25_page_search",
    "build_bm25_chunk_search",
    "build_page_dense_search",
    "build_reused_chunk_search",
    "evidence_payload",
    "emit_tutorial_audit",
    "format_context",
    "load_annotation",
    "load_query_catalog",
    "load_query_controls",
    "load_query_only",
    "load_zhipuai_api_key",
    "llm_call",
    "parse_json_list",
    "parse_json_object",
    "print_method_report",
    "rank_and_coverage",
    "unique_evidence",
]
