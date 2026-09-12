"""C7 分块、父片段选择与拒答边界的行为回归。"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys

import pytest


COURSE = Path(__file__).parents[2] / "notebook" / "C7 高级 RAG 技巧"
CONTEXT_NOTEBOOK = COURSE / "6. 处理信息缺口" / "按句子和父子片段补充上下文.ipynb"
GENERATION_NOTEBOOK = COURSE / "5. 生成阶段" / "排序、压缩与回答.ipynb"
if str(COURSE) not in sys.path:
    sys.path.insert(0, str(COURSE))

from common.eval_utils import (  # noqa: E402
    ChunkEvidence,
    build_bm25_search,
    make_recursive_chunks,
    normalize_text,
    split_sentences,
)


def _notebook_functions(path, names, namespace):
    """只执行指定函数定义；行为测试不加载模型或运行整本 notebook。"""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    definitions = [
        node for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in definitions} == set(names)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_recursive_overlap_applies_between_normal_chinese_sentences():
    text = "第一句。第二句。第三句。第四句。"
    pages = [{"page": 1, "text": text}]
    plain = make_recursive_chunks(pages, chunk_size=9, overlap=0)
    overlapping = make_recursive_chunks(pages, chunk_size=9, overlap=2)

    assert [row["text"] for row in plain] == ["第一句。第二句。", "第三句。第四句。"]
    assert [row["text"] for row in overlapping] == [
        "第一句。第二句。", "句。第三句。", "句。第四句。",
    ]


@pytest.mark.parametrize(
    ("text", "chunk_size", "overlap"),
    [
        ("abcdefghijklmnop", 6, 2),
        ("短。句。也。要。前。进。", 5, 4),
        ("  中文一。\n\n中文二！ English 1.2. Next sentence.  ", 13, 3),
        ("没有句号的一个超长句子也必须完整保留", 7, 1),
    ],
)
def test_recursive_overlap_preserves_text_and_respects_length(text, chunk_size, overlap):
    chunks = make_recursive_chunks(
        [{"page": 3, "text": text}], chunk_size=chunk_size, overlap=overlap,
    )
    assert all(0 < len(row["text"]) <= chunk_size for row in chunks)
    for left, right in zip(chunks, chunks[1:]):
        assert left["text"][-overlap:] == right["text"][:overlap]
    reconstructed = chunks[0]["text"] + "".join(row["text"][overlap:] for row in chunks[1:])
    assert reconstructed == normalize_text(text)


def test_recursive_overlap_does_not_cross_page_boundaries():
    chunks = make_recursive_chunks(
        [{"page": 1, "text": "甲甲。乙乙。"}, {"page": 2, "text": "丙丙。丁丁。"}],
        chunk_size=5,
        overlap=1,
    )
    second_page = [row for row in chunks if row["pages"] == [2]]
    assert second_page[0]["text"] == "丙丙。"
    assert all("甲" not in row["text"] and "乙" not in row["text"] for row in second_page)


def test_common_sentence_boundaries_handle_adjacent_chinese_and_decimal_numbers():
    assert split_sentences("值为0.5。下一句！还有一句？Value 1.2. Next.") == [
        "值为0.5。", "下一句！", "还有一句？", "Value 1.2.", "Next.",
    ]


def _sentence_window_namespace():
    return _notebook_functions(
        CONTEXT_NOTEBOOK,
        ["sentence_parts", "sentence_spans", "sentence_window_from_hit"],
        {
            "split_sentences": split_sentences,
            "page_by_number": {
                1: {"text": "第一句。第二句。第三句。第四句。"},
                2: {"text": "其他页内容。"},
            },
        },
    )


def test_sentence_window_uses_common_chinese_boundaries_and_current_page_index():
    namespace = _sentence_window_namespace()
    hit = ChunkEvidence("hit", [1], "第二句", 1.0)
    assert namespace["sentence_parts"]("甲。乙！丙？") == split_sentences("甲。乙！丙？")
    assert namespace["sentence_window_from_hit"](hit, radius=1) == ["第一句。", "第二句。", "第三句。"]
    assert namespace["sentence_window_from_hit"](hit, radius=10) == [
        "第一句。", "第二句。", "第三句。", "第四句。",
    ]


def test_sentence_window_rejects_an_anchor_missing_from_its_page():
    namespace = _sentence_window_namespace()
    with pytest.raises(ValueError, match="精确定位"):
        namespace["sentence_window_from_hit"](ChunkEvidence("bad", [1], "不存在的句子", 1.0))


@pytest.mark.parametrize("reverse_parents", [False, True])
def test_auto_merge_selects_greater_hit_ratio_even_if_that_parent_is_later(reverse_parents):
    child_ids = ["a1", "a2", "b1", "b2", "a3"]
    hits = [ChunkEvidence(child_id, [1], child_id, 1.0) for child_id in child_ids]
    groups = {
        "a": ["a1", "a2", "a3", "a4", "a5", "a6"],
        "b": ["b1", "b2", "b3"],
    }
    if reverse_parents:
        groups = dict(reversed(list(groups.items())))
    namespace = _notebook_functions(
        CONTEXT_NOTEBOOK,
        ["ranked_merge_candidates", "auto_merge", "auto_merge_context"],
        {
            "CANDIDATE_K": 5,
            "auto_search": lambda query, top_k: hits[:top_k],
            "auto_child_ids_by_parent": groups,
        },
    )
    selected_hits, parent, ratio = namespace["auto_merge"]("same query")
    assert parent == "b" and ratio == pytest.approx(2 / 3)
    assert [hit.chunk_id for hit in selected_hits] == ["b1", "b2"]
    merged = namespace["auto_merge_context"](
        child_ids, {}, groups, {child_id: child_id for child_id in child_ids},
        {"a": "父 A", "b": "父 B"}, threshold=0.25,
    )
    assert merged == "父 B\n\n父 A"


def test_auto_merge_breaks_equal_ratios_by_retrieval_rank_and_deduplicates_hits():
    namespace = _notebook_functions(CONTEXT_NOTEBOOK, ["ranked_merge_candidates"], {})
    rank = namespace["ranked_merge_candidates"]
    groups = {"a": ["a1", "a2", "a3"], "b": ["b1", "b2", "b3"]}
    assert [item["parent_id"] for item in rank(["b1", "a1", "a2", "b2"], groups)] == ["b", "a"]
    assert rank(["a1", "a1", "a1"], groups) == []
    assert rank(["a1", "a2"], {"a": ["a1", "a2", "a3", "a4"]}, threshold=0.5) == []


def _generation_namespace():
    pages = [{"page": 1, "text": "SVM用于分类。 CUDART是另一名称。"}]
    return _notebook_functions(
        GENERATION_NOTEBOOK,
        ["extract_from_source", "missing_technical_identifiers", "answer_from_source"],
        {
            "pages": pages,
            "page_search": build_bm25_search(pages),
            "build_bm25_search": build_bm25_search,
            "clean_text": normalize_text,
            "re": re,
        },
    )


@pytest.mark.parametrize(
    ("query", "missing"),
    [
        ("书中CUDA版本是什么？", ["CUDA"]),
        ("MNIST的样本数是多少？", ["MNIST"]),
        ("有没有CUDA和MNIST，以及CUDA建议？", ["CUDA", "MNIST"]),
    ],
)
def test_generation_refuses_missing_ascii_identifiers_adjacent_to_chinese(query, missing):
    namespace = _generation_namespace()
    counter = {"calls": 0}
    assert namespace["missing_technical_identifiers"](query) == missing
    page, answer = namespace["answer_from_source"](query, check_boundary=True, counter=counter)
    assert page == 1 and counter == {"calls": 1}
    assert "资料中没有找到" in answer and "不能仅依据" in answer
    assert all(identifier in answer for identifier in missing)


def test_generation_does_not_refuse_present_identifier_or_match_inside_ascii_word():
    namespace = _generation_namespace()
    assert namespace["missing_technical_identifiers"]("书中SVM用于什么？") == []
    assert namespace["missing_technical_identifiers"]("preCUDApost是一个整体") == []
    _, answer = namespace["answer_from_source"]("书中SVM用于什么？", check_boundary=True)
    assert "SVM" in answer and "资料中没有找到" not in answer
