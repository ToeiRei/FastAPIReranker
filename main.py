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

    pairs = request.construct_pairs()
    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length
        ).to(device)
        scores = model(**inputs, return_dict=True).logits.view(-1).float()

    # Build response objects
    response = [
        ResponseData(id=doc.id, similarity=score.item())
        for doc, score in zip(request.documents, scores)
    ]
    response.sort(key=lambda r: r.similarity, reverse=True)

    return {"data": response}

# Entrypoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=host, port=port)
