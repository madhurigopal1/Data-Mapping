# -*- coding: utf-8 -*-
"""
FastAPI app to compare CSV columns with PDF sections using BERT embeddings.
Generates JSON mapping and Excel file for download.
"""

import os
import tempfile
import pandas as pd
import fitz  # PyMuPDF
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load BERT model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0)

@app.post("/compare")
async def compare_files(csv_file: UploadFile = File(...), pdf_file: UploadFile = File(...)):
    # Save uploaded files temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_csv:
        temp_csv.write(await csv_file.read())
        csv_path = temp_csv.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(await pdf_file.read())
        pdf_path = temp_pdf.name

    # Read CSV
    csv_df = pd.read_csv(csv_path)

    # Read PDF
    pdf_doc = fitz.open(pdf_path)
    pdf_sections = [page.get_text().strip() for page in pdf_doc if page.get_text().strip()]

    # Mapping results
    mapping_results = []
    excel_rows = []
    for column in csv_df.columns:
        column_text = column + " " + " ".join(csv_df[column].dropna().astype(str).tolist())
        column_emb = get_embedding(column_text).unsqueeze(0)

        section_scores = []
        for section in pdf_sections:
            section_emb = get_embedding(section).unsqueeze(0)
            score = cosine_similarity(column_emb.numpy(), section_emb.numpy())[0][0]
            section_scores.append({"section_preview": section[:150], "confidence": round(score, 2)})

        section_scores.sort(key=lambda x: x["confidence"], reverse=True)
        top_matches = section_scores[:3]

        mapping_results.append({
            "csv_column": column,
            "top_matches": top_matches
        })

        # Prepare rows for Excel
        for match in top_matches:
            excel_rows.append({
                "CSV Column": column,
                "PDF Section Preview": match["section_preview"],
                "Confidence": match["confidence"]
            })

    # Save Excel file
    excel_df = pd.DataFrame(excel_rows)
    excel_path = os.path.join(tempfile.gettempdir(), "mapping_results.xlsx")
    excel_df.to_excel(excel_path, index=False)

    return JSONResponse(content={
        "mappings": mapping_results,
        "excel_download_path": excel_path
    })

# Run the API on port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("compare_api:app", host="127.0.0.1", port=8000, reload=True)