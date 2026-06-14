#!/usr/bin/env python3
"""Mini eval: 5 questions, baseline + 3 methods"""
import sys
import warnings
sys.path.insert(0, ".")
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from _common import (
    get_cleaned_pdf_documents, open_or_build_chroma,
    llm_call as _raw_llm_call, build_rag_generation_prompt,
    trim_context_to_budget,
    run_shared_eval, build_compare_table, CONTEXT_CHAR_BUDGET,
)

warnings.filterwarnings("ignore")

def llm_call(prompt: str) -> str:
    return _raw_llm_call(prompt, sleep_after=1.0)

import json
with open("difficult_dataset.json", "r", encoding="utf-8") as f:
    diff_data = json.load(f)

# Use first 5 questions
qna_dict = {d["query"]: d["answer"] for d in diff_data[:5]}
print(f"Mini eval: {len(qna_dict)} questions")

def build_retriever(chunk_size=256, chunk_overlap=20, k=4):
    docs = list(get_cleaned_pdf_documents())
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""], keep_separator=True,
    )
    chunks = splitter.split_documents(docs)
    ids = [f"b{i}" for i in range(len(chunks))]
    vs = open_or_build_chroma(f"./chroma_db/baseline_{chunk_size}_{chunk_overlap}", chunks, ids)
    return vs.as_retriever(search_kwargs={"k": k})

retriever = build_retriever()

def baseline_pipeline(question):
    docs = retriever.invoke(question)
    ctx = trim_context_to_budget("\n\n".join(d.page_content for d in docs), CONTEXT_CHAR_BUDGET)
    return llm_call(build_rag_generation_prompt(question, ctx)), []

DIMENSIONAL_EVAL_PROMPT = (
    "你是判卷人。先在心里把「参考答案」拆成 2~4 个【必答要点】"
    "（如：方法1、方法2、关键条件、关键定义等），不要输出这些要点。\n"
    "然后比对「模型答案」覆盖了几个要点：\n"
    "- 全部覆盖且无关键事实错误：2\n"
    "- 覆盖一半左右、或部分要点表述含糊但方向正确：1\n"
    "- 大部分要点缺失、或关键事实错误：0\n\n"
    "用户问题：{question}\n参考答案：{expected_answer}\n模型答案：{llm_answer}\n\n"
    "仅输出一行，只包含字符 0、1 或 2，不要任何其它文字。"
)

print("\nRunning baseline...")
base_df = run_shared_eval(lambda q: baseline_pipeline(q)[0], qna_dict, eval_prompt_template=DIMENSIONAL_EVAL_PROMPT)
print(f"Baseline: mean={base_df['rag_eval_results'].mean():.3f}, total={base_df['rag_eval_results'].sum()}/{len(base_df)*2}")

for q, score in zip(base_df['question'], base_df['rag_eval_results']):
    print(f"  {score}: {q[:50]}...")
