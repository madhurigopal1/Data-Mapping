import pandas as pd
import fitz  # PyMuPDF
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Dummy embedding function for illustration
def get_embedding(text: str):
    return np.random.rand(768)

# File paths (update these with your actual file locations)
csv_path = "sample.csv"
pdf_path = "sample.pdf"

# Read CSV
try:
    csv_df = pd.read_csv(csv_path)
except Exception as e:
    print(f"CSV read error: {str(e)}")
    exit()

# Read PDF
try:
    pdf_doc = fitz.open(pdf_path)
    pdf_sections = [
        {"text": page.get_text().strip(), "page": i + 1}
        for i, page in enumerate(pdf_doc)
        if page.get_text().strip()
    ]
except Exception as e:
    print(f"PDF read error: {str(e)}")
    exit()

# Compare each column with PDF sections
results = {}
for column in csv_df.columns:
    column_text = column + " " + " ".join(csv_df[column].dropna().astype(str).tolist())
    column_emb = get_embedding(column_text).reshape(1, -1)

    section_scores = []
    for section in pdf_sections:
        section_emb = get_embedding(section["text"]).reshape(1, -1)
        score = cosine_similarity(column_emb, section_emb)[0][0]
        if score > 0.5:  # Confidence threshold
            section_scores.append({
                "section_preview": section["text"][:100],
                "confidence": round(score, 2),
                "page": section["page"]
            })

    section_scores.sort(key=lambda x: x["confidence"], reverse=True)
    results[column] = section_scores[:3]  # Top 3 matches

# Print results
for column, matches in results.items():
    print(f"\nColumn: {column}")
    for match in matches:
        print(f"  Page {match['page']} (Confidence: {match['confidence']}): {match['section_preview']}")