# FastAPIReranker

A lightweight FastAPI service that exposes Hugging Face cross‑encoder rerankers (default: [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)) over HTTP.  
Useful as a plug‑and‑play component in RAG pipelines to improve retrieval quality by reranking candidate documents against a query.

---

## ✨ Features
- 🚀 Simple REST API (`/api/v1/rerank`)  
- 🧠 Strong multilingual reranker model by default  
- ⚙️ Configurable via environment variables (model, device, max length, host, port)  
- 🐳 Docker‑ready, with Compose support  
- 🔌 Easy to integrate with Langflow, Qdrant, Haystack, or custom RAG stacks  

---

## 📦 Installation

### Local
```bash
git clone https://github.com/yourname/FastAPIReranker.git
cd FastAPIReranker
pip install -r requirements.txt
DEVICE=cuda MAX_LENGTH=1024 python main.py
```

### Docker
```bash
docker build -t fastapi-reranker .
docker run -e HOST=0.0.0.0 -e PORT=8787 -p 8787:8787 fastapi-reranker
```

### Docker Compose
```bash
docker compose up --build
```

---

## ⚙️ Environment Variables

| Variable     | Default                        | Description                                  |
|--------------|--------------------------------|----------------------------------------------|
| `HOST`       | `127.0.0.1`                    | Host to bind the service                     |
| `PORT`       | `8787`                         | Port to listen on                            |
| `MAX_LENGTH` | `512`                          | Max sequence length for tokenizer            |
| `MODEL`      | `BAAI/bge-reranker-v2-m3`      | Hugging Face model to use                    |
| `DEVICE`     | auto (`cuda`/`mps`/`cpu`)      | Device to run on                             |

---

## 🔗 API Usage

### Request
```bash
curl -X POST "http://127.0.0.1:8787/api/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is panda?",
    "documents": [
      {"id": 1, "text": "meow."},
      {"id": 2, "text": "The food bowl is empty."},
      {"id": 3, "text": "I like cats."}
    ]
  }'
```

### Response
```json
{
  "data": [
    {"id": 2, "similarity": 5.265044212341309},
    {"id": 3, "similarity": -7.278249263763428},
    {"id": 1, "similarity": -8.183815002441406}
  ]
}
```

---

## 🛠 Integration ideas
- **Langflow**: Call via API node between Retriever and Prompt Template.  
- **Qdrant**: Use as a rerank step after vector search.  
- **Custom RAG**: Drop in as a microservice to improve relevance.  

---

## 📜 License
MIT — free to use, modify, and share.
