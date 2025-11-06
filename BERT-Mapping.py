# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 15:34:18 2025

@author: 424646
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from fastapi import FastAPI, Request
import uvicorn

# Load BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

# Function to get sentence embeddings using BERT
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0)  # [CLS] token

# Simulated CSV and PDF text
csv_texts = [
    "Company revenue increased by 20% in Q3",
    "New product launched in September"
]
pdf_texts = [
    "The company reported a 20% revenue growth in the third quarter",
    "A product was introduced in September"
]

# Create labeled pairs
data_pairs = [{"text1": c, "text2": p, "label": 1} for c, p in zip(csv_texts, pdf_texts)]
data_pairs += [
    {"text1": csv_texts[0], "text2": "CEO resigned in October", "label": 0},
    {"text1": csv_texts[1], "text2": "Annual report published in December", "label": 0}
]

# Dataset class
class TextPairDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.embeddings = [
            torch.cat((get_embedding(d["text1"]), get_embedding(d["text2"])))
            for d in data
        ]
        self.labels = [d["label"] for d in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.embeddings[idx], torch.tensor(self.labels[idx])

# Train-test split
df = pd.DataFrame(data_pairs)
train_df = df.sample(frac=0.5, random_state=42)
test_df = df.drop(train_df.index)

train_dataset = TextPairDataset(train_df.to_dict(orient="records"))
test_dataset = TextPairDataset(test_df.to_dict(orient="records"))

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2)

# Simple classifier
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.fc(x)

model = SimpleClassifier(input_dim=768*2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(3):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=-1)
        preds = torch.argmax(probs, dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

accuracy = accuracy_score(all_labels, all_preds)
print("Model Accuracy:", accuracy)

# Confidence scores
for i, (inputs, labels) in enumerate(test_loader):
    outputs = model(inputs)
    probs = torch.nn.functional.softmax(outputs, dim=-1)
    for j in range(len(labels)):
        print(f"Pair {i*2 + j + 1}: Match={bool(torch.argmax(probs[j]))}, Confidence={probs[j][1].item():.2f}")

# FastAPI Web Service
#app = FastAPI()

#@app.post("/compare")
def compare_texts(data):
#    data = await request.json()
    emb1 = get_embedding(data[0])
    emb2 = get_embedding(data[1])
    inputs = torch.cat((emb1, emb2)).unsqueeze(0)
    outputs = model(inputs)
    probs = torch.nn.functional.softmax(outputs, dim=-1)
    confidence = probs[0][1].item()
    match = torch.argmax(probs, dim=1).item()
    return {"match": bool(match), "confidence": confidence}

# Run the app directly from Python
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

#compare_texts(data_pairs)