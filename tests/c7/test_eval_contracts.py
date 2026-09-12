"""C7 评估与片段身份的最小契约测试。"""

from pathlib import Path
import sys
import types
import pytest


COURSE = Path(__file__).parents[2] / "notebook" / "C7 高级 RAG 技巧"
if str(COURSE) not in sys.path:
    sys.path.insert(0, str(COURSE))

from common.eval_utils import average_precision, precision_at_k, recall_at_k
from common.eval_utils import Evidence
import common.nontraining_utils as nontraining_utils
from common.nontraining_utils import llm_call, unique_evidence


def _fake_zhipuai(monkeypatch, *, content="模型结果", error=None):
    calls = []

    response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=content)
            )
        ]
    )

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if error is not None:
                raise error
            return response

    clients = []

    class FakeZhipuAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chat = types.SimpleNamespace(
                completions=FakeCompletions()
            )
            clients.append(self)

    module = types.ModuleType("zhipuai")
    module.ZhipuAI = FakeZhipuAI
    monkeypatch.setitem(sys.modules, "zhipuai", module)
    monkeypatch.setattr(
        nontraining_utils,
        "load_zhipuai_api_key",
        lambda: "test-key",
    )
    return clients, calls


def test_llm_call_success_uses_one_http_request_and_disables_sdk_retries(monkeypatch):
    clients, calls = _fake_zhipuai(monkeypatch, content="  成功结果  ")

    assert llm_call("问题", max_tokens=123) == "成功结果"
    assert len(clients) == 1
    assert clients[0].kwargs == {"api_key": "test-key", "max_retries": 0}
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 123


def test_llm_call_service_error_is_propagated_after_one_request(monkeypatch):
    clients, calls = _fake_zhipuai(
        monkeypatch,
        error=RuntimeError("service unavailable"),
    )

    with pytest.raises(RuntimeError, match="service unavailable"):
        llm_call("问题")
    assert len(clients) == 1
    assert len(calls) == 1


def test_llm_call_empty_response_fails_without_a_second_request(monkeypatch):
    clients, calls = _fake_zhipuai(monkeypatch, content="   ")

    with pytest.raises(RuntimeError, match="空文字"):
        llm_call("问题")
    assert len(clients) == 1
    assert len(calls) == 1


def test_llm_call_rejects_application_retries_before_http(monkeypatch):
    clients, calls = _fake_zhipuai(monkeypatch)

    with pytest.raises(ValueError, match="不支持应用层重试"):
        llm_call("问题", retries=2)
    assert clients == []
    assert calls == []


def test_missing_relevant_evidence_lowers_recall_and_average_precision():
    flags = [1, 0]
    assert recall_at_k(flags, total_relevant=2, k=2) == 0.5
    assert average_precision(flags, total_relevant=2) == 0.5


def test_recall_rejects_impossible_total_relevant():
    with pytest.raises(ValueError, match="必须为正数"):
        recall_at_k([False], total_relevant=0, k=1)
    with pytest.raises(ValueError, match="不能小于"):
        recall_at_k([True, True], total_relevant=1, k=2)


def test_average_precision_requires_complete_annotation_denominator():
    with pytest.raises(TypeError):
        average_precision([1])
    with pytest.raises(ValueError):
        average_precision([1], total_relevant=0)
    with pytest.raises(ValueError):
        average_precision([1, 1], total_relevant=1)


def test_precision_at_k_uses_returned_count_when_fewer_than_k():
    assert precision_at_k([1], k=4) == 1.0


def test_same_page_different_chunks_are_retained():
    items = [
        Evidence(3, "第一片段", 0.9, "p3_a"),
        Evidence(3, "第二片段", 0.8, "p3_b"),
    ]
    assert [item.chunk_id for item in unique_evidence(items)] == ["p3_a", "p3_b"]


def test_duplicate_chunk_is_removed():
    items = [
        Evidence(3, "片段", 0.9, "p3_a"),
        Evidence(3, "片段", 0.8, "p3_a"),
    ]
    assert len(unique_evidence(items)) == 1


def test_same_page_without_chunk_id_uses_text_identity():
    items = [
        Evidence(3, "第一片段", 0.9),
        Evidence(3, "第二片段", 0.8),
        Evidence(3, "  第一片段  ", 0.7),
    ]
    assert [item.text for item in unique_evidence(items)] == ["第一片段", "第二片段"]
