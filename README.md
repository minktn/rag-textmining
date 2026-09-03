# VietLegal RAG — Hệ thống RAG Pháp Luật Việt Nam (Luật Đất đai 2024)

Pipeline tìm kiếm, truy xuất tri thức và trả lời câu hỏi pháp lý tự động dựa trên các kỹ thuật RAG tiên tiến (Dense Retrieval, Contriever, GraphRAG, HyDE, RAG-Fusion, CRAG, Filter-then-Rerank, Prompt Compression).

---

## 📋 Yêu cầu tiên quyết (Prerequisites)

1. **Python >= 3.10** & Công cụ quản lý gói **`uv`** (khuyên dùng):
   ```bash
   pip install uv
   ```
2. **Node.js >= 18** & **npm** (để chạy giao diện UI).
3. **File cấu hình môi trường `.env`** tại thư mục gốc của dự án:
   Tạo file `.env` từ `.env.example`:
   ```env
   # LLM API Keys
   NVIDIA_API_KEY=your_nvidia_api_key
   GROQ_API_KEY=your_groq_api_key
   GOOGLE_API_KEY=your_google_api_key

   # Vector Database (Qdrant Cloud hoặc Local)
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key

   # Neo4j Graph Database (Tùy chọn nếu dùng Neo4j Cloud/Local)
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password
   ```

---

## 🚀 Thứ tự chạy các lệnh (Execution Order)

### Bước 1: Cài đặt thư viện phụ thuộc (Dependencies)

Cài đặt toàn bộ môi trường Python qua `uv`:
```bash
uv sync
```

---

### Bước 2: Chuẩn bị cơ sở dữ liệu & Ingest dữ liệu (Data Pipeline)

#### 2.1. Phân đoạn dữ liệu thô (Chunking)
Phân đoạn file văn bản luật `landlaw.md` thành `landlaw_chunks.json`:
```bash
uv run python Scripts/ingest.py --chunk-only
```

#### 2.2. Nhúng dữ liệu vào cơ sở dữ liệu (Vector / Contriever / Graph)
* **Nhúng vào Vector DB cơ sở (BAAI/bge-m3, 1024 dims)**:
  ```bash
  uv run python Scripts/ingest.py --store base
  ```
  *(Tùy chọn: Thêm `--late` nếu muốn sử dụng Late Chunking)*

* **Nhúng vào Contriever Store (facebook/mcontriever-msmarco, 768 dims)**:
  ```bash
  uv run python Scripts/ingest.py --store contriever
  ```

* **Xây dựng Knowledge Graph / Graph Database (Microsoft GraphRAG)**:
  ```bash
  uv run python Scripts/ingest.py --store graph --build-graph
  ```

* **Kiểm tra trạng thái nạp dữ liệu**:
  ```bash
  uv run python Scripts/ingest.py --status
  ```

---

### Bước 3: Khởi động hệ thống & Dịch vụ (Services)

Mở **3 cửa sổ Terminal riêng biệt** để khởi động đầy đủ các dịch vụ:

#### Terminal 1: Khởi động NLI Microservice (BamiBERT-ViLegalNLI — Port 8001)
> *Dịch vụ phân loại Entailment/Contradiction phục vụ Corrective RAG (CRAG).*
```bash
uv run python -m src.models.nli_model.api
```

#### Terminal 2: Khởi động Backend API Server (FastAPI — Port 8002)
> *Cung cấp REST API cho Chatbot RAG và Evaluation Playground.*
```bash
uv run uvicorn src.api.server:app --host 127.0.0.1 --port 8002
```

#### Terminal 3: Khởi động Giao diện Web (React Vite — Port 5173)
```bash
cd ui
npm install
npm run dev
```
👉 Truy cập giao diện ứng dụng tại: **http://localhost:5173**

---

### Bước 4: Thử nghiệm nhanh bằng dòng lệnh (CLI Testing)

#### 4.1. Kiểm tra truy xuất ngữ cảnh (Retrieval only — không gọi LLM)
```bash
uv run python Scripts/retrieve.py --query "Hạn mức giao đất ở là bao nhiêu?" --no-generate
```

#### 4.2. Kiểm tra toàn bộ pipeline sinh câu trả lời (Full RAG Generation)
```bash
uv run python Scripts/retrieve.py --query "Hạn mức giao đất ở là bao nhiêu?"
```

---

### Bước 5: Chạy đánh giá định lượng (Evaluation Pipeline)

Chạy script đánh giá benchmark trên tập câu hỏi Luật Đất đai 2024:

* **Đánh giá cơ bản nhanh (Basic Metrics: F1, BLEU, ROUGE-L, Recall@K, Precision@K)**:
  ```bash
  uv run python Scripts/evaluate.py --limit 10 --skip-ragas
  ```

* **Đánh giá đầy đủ có RAGAS (LLM-as-a-judge: Faithfulness, Answer Relevancy, Context Precision/Recall)**:
  ```bash
  uv run python Scripts/evaluate.py --limit 5
  ```

* **Đánh giá với Graph Database**:
  ```bash
  uv run python Scripts/evaluate.py --graph --limit 5
  ```

* **Đánh giá với RAG-Fusion**:
  ```bash
  uv run python Scripts/evaluate.py --rag-fusion --limit 5
  ```

---

## 📚 Nghiên cứu khoa học áp dụng (Used Papers)
- **Hypothetical Document Embedding (HyDE)**: [arXiv:2212.10496](https://arxiv.org/pdf/2212.10496)
- **RAG-Fusion**: [arXiv:2402.03367](https://arxiv.org/pdf/2402.03367)
- **Filter-then-Rerank**: [arXiv:2303.08559](https://arxiv.org/pdf/2303.08559)
- **Prompt Compression (LongLLMLingua)**: [llmlingua.com](https://llmlingua.com/longllmlingua.html)
- **Self-RAG**: [arXiv:2310.11511](https://arxiv.org/pdf/2310.11511)
- **Late Chunking**: [arXiv:2409.04701](https://arxiv.org/pdf/2409.04701)
- **Corrective RAG (CRAG)**: [arXiv:2401.15884](https://arxiv.org/pdf/2401.15884) *(Mô hình BamiBERT ViLegalNLI đã tinh chỉnh)*
- **Contriever**: [arXiv:2112.09118](https://arxiv.org/abs/2112.09118)
- **GraphRAG**: [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
