"""
FastAPI application that uses a Hugging Face cross-encoder reranker
to rank documents based on their similarity to a given query.

POST /api/v1/rerank
Request: { "query": "...", "documents": [ { "id": ..., "text": "..." }, ... ] }
Response: { "data": [ { "id": ..., "similarity": ... }, ... ] }
"""

import os
import logging
from uuid import UUID
from typing import List, Union
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configuration
host = os.getenv("HOST", "127.0.0.1")
port = int(os.getenv("PORT", "8787"))
max_length = int(os.getenv("MAX_LENGTH", "512"))
model_name = os.getenv("MODEL", "BAAI/bge-reranker-v2-m3")
device = os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.info("host: %s", host)
logging.info("port: %d", port)
logging.info("max_length: %d", max_length)
logging.info("model: %s", model_name)
logging.info("device: %s", device)

# Load model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model = model.to(device)
model.eval()

app = FastAPI()

# Data models
class Document(BaseModel):
    id: Union[int, str, UUID]
    text: str

class RequestData(BaseModel):
    query: str
    documents: List[Document]

    def construct_pairs(self):
        return [[self.query, doc.text] for doc in self.documents]

class ResponseData(BaseModel):
    id: Union[int, str, UUID]
    similarity: float

# Endpoint
@app.post("/api/v1/rerank")
async def rerank_documents(request: RequestData):
    logging.info("Reranking %d documents for query: %.50s...", len(request.documents), request.query)

    # Guard: no documents → return empty list
    if not request.documents:
        logging.info("No documents supplied, returning empty result.")
        return {"data": []}

    # Optional: filter out empty/whitespace-only texts
    docs = [doc for doc in request.documents if (doc.text or "").strip()]
    if not docs:
        logging.info("All documents empty after filtering, returning empty result.")
        return {"data": []}

    pairs = [[request.query, doc.text] for doc in docs]

    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length
        ).to(device)
        scores = model(**inputs, return_dict=True).logits.view(-1).float()

    response = [
        ResponseData(id=doc.id, similarity=score.item())
        for doc, score in zip(docs, scores)
    ]
    response.sort(key=lambda r: r.similarity, reverse=True)

    return {"data": response}

# Basic healthcheck
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

# Entrypoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=host, port=port)
