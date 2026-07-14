# rag-textmining

RAG pipeline for Vietnamese Land Law retrieval and answer generation.

## Prerequisites

- Docker and Docker Compose
- A `.env` file at the project root
- Source data in `workspace/data/original/landlaw.md`
- Chunked data in `workspace/data/chunked/landlaw_chunks.json`

The `.env` file should contain:

```env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
```

Do not commit real API keys.

## Run With Docker

Start the container from the project root:

```bash
docker-compose up -d --build
```

Open a shell inside the app container:

```bash
docker exec -it rag-textmining-api /bin/bash
```

Inside the container, the workspace is mounted at `/app`:

```bash
cd /app
```

## Ingest Data

Run ingestion before retrieval so Qdrant has the embedded chunks:

```bash
python scripts/ingest.py
```

This reads:

```text
data/chunked/landlaw_chunks.json
```

and upserts vectors into the default Qdrant collection:

```text
landlaw
```

If `landlaw_chunks.json` does not exist yet, generate chunks from `data/original/landlaw.md` first by using the `chunking()` function in `scripts/ingest.py`, then run embedding ingestion.

## Test Retrieval Only

Use `--no-generate` first. This tests normalization, metadata extraction, Qdrant retrieval, reranking, reference expansion, and final context construction without calling the LLM.

```bash
python scripts/retrieve.py --query "Theo Điều 188 Luật Đất đai quy định gì?" --no-generate
```

Expected output is JSON with fields such as:

```text
query
normalized_query
filters
chunks
expanded_chunks
context
```

Try another query:

```bash
python scripts/retrieve.py --query "Điều kiện chuyển nhượng quyền sử dụng đất là gì?" --no-generate
```

## Test Full RAG Generation

After retrieval-only works, run without `--no-generate`:

```bash
python scripts/retrieve.py --query "Theo Điều 188 Luật Đất đai quy định gì?"
```

This will:

1. Retrieve relevant chunks from Qdrant.
2. Expand referenced legal articles.
3. Build a Vietnamese prompt with metadata and content.
4. Call Groq through `LLMManager`.
5. Print the generated Vietnamese answer.

## Optional Collection Override

The default collection is `landlaw`. To test another collection:

```bash
python scripts/retrieve.py --query "Theo Điều 188 Luật Đất đai quy định gì?" --collection your_collection --no-generate
```

## Run Outside Docker

Docker is recommended. Running outside Docker is only useful if your local Python environment has all dependencies installed and can reach Qdrant.

From the project root:

```bash
cd workspace
pip install -r requirements.txt
python scripts/retrieve.py --query "Theo Điều 188 Luật Đất đai quy định gì?" --no-generate
```

If local Python cannot import packages such as `qdrant_client`, `sentence_transformers`, or `groq`, run inside Docker instead.

## Useful Checks

Show CLI help:

```bash
python scripts/retrieve.py --help
```

Check container status:

```bash
docker ps
```

Stop the container:

```bash
docker-compose down
```

## Project Structure

```text
project/
|-- workspace/
|   |-- data/
|   |   |-- chunked/
|   |   |   `-- landlaw_chunks.json
|   |   `-- original/
|   |       `-- landlaw.md
|   |-- scripts/
|   |   |-- ingest.py
|   |   `-- retrieve.py
|   `-- src/
|       |-- configs/
|       |-- common/
|       |-- database/
|       |-- data_pipeline/
|       |-- llm/
|       `-- retriever/
|-- .env
|-- docker-compose.yml
|-- Dockerfile
`-- README.md
```
