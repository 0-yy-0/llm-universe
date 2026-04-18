"""
6. 增强阶段 三节共用的公共底座（single source of truth）。

各 notebook 中"首次出现"的核心新概念（评估、钩子胶水）必须先在 notebook 内
完整 inline 重写一次，再 from _common import 复用；本文件保存与 notebook
首次定义字符相同的副本。基础工具（embedding、PDF、Chroma 等）前几章已讲，
notebook 中不重复展示，直接 import。
"""
from __future__ import annotations

import json
import os
import re
import time
import warnings
from pathlib import Path
from typing import Callable, Iterable

import fitz  # PyMuPDF
import pandas as pd
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from modelscope import snapshot_download

warnings.filterwarnings("ignore")
load_dotenv()


# ---------- 路径与默认参数 ----------

EMBED_MODEL_ID = "BAAI/bge-small-zh-v1.5"
EMBED_MODEL_PATH = f"./models/{EMBED_MODEL_ID}"

PDF_PATH = "../3. 索引阶段/data/pumpkin_book.pdf"
QA_PATH = "../3. 索引阶段/data/train_dataset.json"

CHROMA_COLLECTION = "nb_ctx"
CONTEXT_CHAR_BUDGET = int(os.environ.get("CONTEXT_CHAR_BUDGET", "1150"))

LLM_MODEL = "glm-4-flash-250414"
LLM_MAX_RETRIES = 10


# ---------- embedding（单例） ----------

_pdf_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """本地 bge-small-zh-v1.5 单例，多个 Chroma 索引共用。"""
    global _pdf_embeddings
    if _pdf_embeddings is None:
        path = EMBED_MODEL_PATH
        if not os.path.exists(path):
            cache_dir = Path("./models")
            cache_dir.mkdir(parents=True, exist_ok=True)
            path = snapshot_download(EMBED_MODEL_ID, cache_dir=str(cache_dir))
        _pdf_embeddings = HuggingFaceEmbeddings(
            model_name=path,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _pdf_embeddings


# ---------- PDF 清洗与按页缓存 ----------

def clean_text(text: str) -> str:
    text = re.sub(r"→_→\n欢迎去各大电商平台选购纸质版南瓜书《机器学习公式详解》\n←_←", "", text)
    text = re.sub(r"→_→\n配套视频教程：https://www.bilibili.com/video/BV1Mh411e7VU\n←_←", "", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"\n+", "", text)
    return text


_CLEANED_PDF_DOCS: list[Document] | None = None


def get_cleaned_pdf_documents() -> list[Document]:
    """加载 + 清洗 PDF 一次，三节共用。返回按页 Document 列表。"""
    global _CLEANED_PDF_DOCS
    if _CLEANED_PDF_DOCS is None:
        doc = fitz.open(PDF_PATH)
        pages = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            pages.append(Document(page_content=clean_text(page.get_text()), metadata={"page": i}))
        doc.close()
        _CLEANED_PDF_DOCS = pages
    return _CLEANED_PDF_DOCS


# ---------- Chroma 工厂 ----------

def open_or_build_chroma(persist_directory: str, documents: list[Document], ids: list[str]) -> Chroma:
    """同一 persist 目录复用，已存在集合则跳过 add。"""
    os.makedirs(persist_directory, exist_ok=True)
    emb = get_embeddings()
    vs = Chroma(
        persist_directory=persist_directory,
        embedding_function=emb,
        collection_name=CHROMA_COLLECTION,
        collection_metadata={"hnsw:space": "cosine"},
    )
    if vs._collection.count() > 0:
        print(f"  -> 加载已有索引: {persist_directory}")
        return vs
    print(f"  -> 创建新索引: {persist_directory}")
    vs.add_documents(documents, ids=ids)
    return vs


# ---------- 智谱 LLM 调用（429 退避，可选每次成功后 sleep） ----------

_zhipu_ai_client = None


def llm_call(prompt: str, *, sleep_after: float = 0.0) -> str:
    """智谱 ZhipuAI 调用。429 时按 min(180, 20*n) 秒退避，最多 LLM_MAX_RETRIES 次。
    sleep_after > 0 时在每次成功调用后追加 sleep（用于 6.2 / 6.3 节对抗连续调用限流）。
    """
    from zhipuai import ZhipuAI

    global _zhipu_ai_client
    if _zhipu_ai_client is None:
        _zhipu_ai_client = ZhipuAI(api_key=os.environ.get("ZHIPUAI_API_KEY"))
    client = _zhipu_ai_client
    for attempt in range(LLM_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            if sleep_after > 0:
                time.sleep(sleep_after)
            return resp.choices[0].message.content
        except Exception as e:
            if "429" not in str(e) or attempt == LLM_MAX_RETRIES - 1:
                raise
            wait = min(180, 20 * (attempt + 1))
            print(f"  [速率限制，等待 {wait}s...]")
            time.sleep(wait)
    raise RuntimeError("llm_call exhausted retries without raising")


# ---------- 通用 prompt 与文本工具 ----------

def build_rag_generation_prompt(question: str, context: str) -> str:
    return (
        "仅根据上下文回答问题，如果上下文没有包含完整答案，请仅回答上下文中的内容，不要补充你自己的知识。\n\n"
        f"问题：{question}\n上下文：\n{context}"
    )


def trim_context_to_budget(text: str, budget: int) -> str:
    text = (text or "").strip()
    if len(text) <= budget:
        return text
    head = text[:budget]
    cut = max(head.rfind("\n\n"), head.rfind("。"), head.rfind("；"))
    if cut > int(budget * 0.55):
        head = head[: cut + 1]
    return head.strip() + "\n\n[...下文已按 CONTEXT_CHAR_BUDGET 截断...]"


# ---------- 题集加载 ----------

def load_qna_subset(qa_path: str, indices: Iterable[int]) -> dict[str, str]:
    """从 train_dataset.json 加载指定下标的 (query -> answer) 字典，保持 indices 顺序。"""
    with open(qa_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    out: dict[str, str] = {}
    for i in indices:
        item = pairs[i]
        q, a = item["query"], item["answer"]
        if q and str(q).strip():
            out[q] = a
    return out


# ---------- 评估与运行器（占位，Task 2/3 中实现） ----------
# simple_eval_2pt / answer_from_context_fn / run_shared_eval / build_compare_table → Task 2
# run_session_eval → Task 3
