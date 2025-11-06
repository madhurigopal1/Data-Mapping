# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 15:37:02 2025

@author: 424646
"""

import pandas as pd
import fitz  # PyMuPDF
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
# Load BERT model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0)

# Load CSV
csv_df = pd.read_csv("sample.csv")  # Replace with actual file path

# Load PDF
pdf_doc = fitz.open("policy_mapping_report.pdf")  # Replace with actual file path
pdf_sections = [page.get_text().strip() for page in pdf_doc if page.get_text().strip()]

# Compare each column with PDF sections
results = {}
for column in csv_df.columns:
    column_text = column + " " + " ".join(csv_df[column].dropna().astype(str).tolist())
    column_emb = get_embedding(column_text).unsqueeze(0)

    section_scores = []
    for section in pdf_sections:
        section_emb = get_embedding(section).unsqueeze(0)
        score = cosine_similarity(column_emb.numpy(), section_emb.numpy())[0][0]
        section_scores.append((section[:100], score))  # Preview first 100 chars

    section_scores.sort(key=lambda x: x[1], reverse=True)
    results[column] = section_scores[:3]

# Print results
for col, matches in results.items():
    print(f"\nColumn: {col}")
    for i, (section, score) in enumerate(matches, 1):
        print(f"  Match {i}: Confidence={score:.2f}, Section Preview='{section}...'")
