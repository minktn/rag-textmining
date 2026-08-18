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

## Run Outside Docker (with uv)

From the project root:

```bash
uv sync
uv run python workspace/scripts/retrieve.py --query "Theo Điều 188 Luật Đất đai quy định gì?" --no-generate
```

## Useful Checks

Show CLI help:

```bash
uv run python workspace/scripts/retrieve.py --help
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
|   |   |-- evaluate.py
|   |   |-- ingest.py
|   |   |-- retrieve.py
|   |   `-- test.py
|   `-- src/
|       |-- common/
|       |-- configs/
|       |-- database/
|       |   |-- chunker/
|       |   `-- embedder/
|       |-- evaluation/
|       |-- generation/
|       |-- retriever/
|       `-- utils/
|-- .env
|-- docker-compose.yml
|-- Dockerfile
|-- pyproject.toml
`-- README.md
```
## USED PAPER:
- Hypothetical Document Embedding: https://arxiv.org/pdf/2212.10496 (**Done**)
- RAG-fusion: https://arxiv.org/pdf/2402.03367 (**Done**)
- Filter-reranker: https://arxiv.org/pdf/2303.08559
- Token_elimination: https://arxiv.org/pdf/2310.13682
- CoG: https://arxiv.org/pdf/2307.06962
- Active RAG: https://arxiv.org/pdf/2305.06983
- SELF-RAG: https://arxiv.org/pdf/2310.11511
- Late chunking: https://arxiv.org/pdf/2409.04701 (**Done**)
- Corrective RAG: https://arxiv.org/pdf/2401.15884 (**Finetuning on progress**)

