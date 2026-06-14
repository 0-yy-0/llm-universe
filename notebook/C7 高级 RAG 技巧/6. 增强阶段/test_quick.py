#!/usr/bin/env python3
import sys
sys.path.insert(0, ".")

# Quick test: just load and verify setup
from _common import get_embeddings, get_cleaned_pdf_documents

print("Loading embeddings...")
emb = get_embeddings()
print(f"Embeddings loaded: {emb}")

print("Loading PDF...")
docs = get_cleaned_pdf_documents()
print(f"PDF loaded: {len(docs)} pages")

print("Loading dataset...")
import json
with open("difficult_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"Dataset loaded: {len(data)} questions")
for i, d in enumerate(data[:3]):
    print(f"  Q{i+1}: {d['query'][:50]}... (p{d['page_num']})")

print("\nSetup OK!")
