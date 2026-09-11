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

## ⚙️ Cấu hình hệ thống (`src/config/settings.py`)

Các tham số chính được chỉnh sửa trực tiếp trong file [`src/config/settings.py`](src/config/settings.py):

* **Mô hình LLM & Chế độ suy luận**:
  * `LLM_SERVICE` / `SUB_LLM_SERVICE`: Dịch vụ LLM (`"nvidia"`, `"groq"`, `"google"`, `"local"`).
  * `LLM_MODE` / `SUB_LLM_MODE`: Chế độ sinh lời giải (`"reason"` hoặc `"base"`).
* **Mô hình Embedding**:
  * `EMBEDDING_MODEL = "BAAI/bge-m3"` (1024 chiều, dùng cho Vector DB và Graph DB).
  * `CONTRIEVER_MODEL = "facebook/mcontriever-msmarco"` (768 chiều).
* **Cấu hình Truy xuất Two-Stage & RRF (Two-Stage Retrieval & Reciprocal Rank Fusion)**:
  * `RETRIEVAL_BM25 = 100`: Số lượng ứng viên được lọc trước qua BM25 (Giai đoạn 1).
  * `RETRIEVAL_DENSE = 20`: Số lượng ứng viên được chọn ra sau khi kết hợp Dense + RRF (Giai đoạn 2 & 3).
  * `RRF_K = 60`: Hằng số làm mượt cho Reciprocal Rank Fusion (RRF).
  * `RERANK_LIMIT = 5`: Số lượng ngữ cảnh đưa vào reranker Cross-Encoder (`BAAI/bge-reranker-v2-m3`).
* **Đánh giá Benchmark (Evaluation & RAGAS)**:
  * `RAGAS_SERVICE`: LLM Judge cho RAGAS (`"nvidia"`, `"groq"`, `"google"`).
  * `EVAL_BATCH_SIZE`: Kích thước batch chia câu hỏi (mặc định: `10`).
  * `EVAL_MAX_WORKERS`: Số luồng đa nhiệm chạy RAG pipeline (mặc định: `4`).
  * `RAGAS_MAX_WORKERS`: Số luồng gửi song song tới RAGAS LLM Judge (mặc định: `4`).
* **Tiền / Hậu xử lý mặc định**:
  * `ADAPTABLE_PREPROCESS`: Danh sách tiền xử lý có thể xài chung với các pp khác.
  * `ADAPTABLE_POSTPROCESS`: Danh sách hậu xử lý có thể xài chung với các pp khác (ví dụ: `["prompt_compression"]`).

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

#### 2.3. Lập chỉ mục BM25 (BM25 Indexing qua `bm25.pkl`)
Hệ thống sử dụng mô hình BM25Okapi gọn nhẹ (lưu qua file `bm25.pkl`) để phục vụ giai đoạn lọc trước (pre-filtering) top 100 ứng viên:
* **Vector Database**: Lưu tại `db/vector_database/bm25.pkl` (quản lý 666 chunks luật, dùng chung cho cả Baseline BGE-M3 và Contriever vì cùng chung một kiểu phân đoạn).
* **Graph Database**: Lưu tại `db/graph_database/bm25.pkl` (quản lý 170 text units trích xuất từ Knowledge Graph).

Lệnh chạy lập chỉ mục BM25:
```bash
# Lập chỉ mục cho toàn bộ Vector DB và Graph DB:
uv run python Scripts/ingest.py --bm25 --store all

# Hoặc lập chỉ mục riêng cho từng store:
uv run python Scripts/ingest.py --bm25 --store base
uv run python Scripts/ingest.py --bm25 --store graph
```
*(Bạn cũng có thể dùng `python -m src.database.ingest --bm25` hoặc `python -m src.database.store_manager --bm25` với cùng đối số)*

#### 2.4. Kiểm tra trạng thái nạp dữ liệu:
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

#### 4.1. Cơ chế Truy xuất Hai Giai đoạn (Two-Stage Hybrid RRF Pipeline)
Mặc định, mọi truy vấn tìm kiếm đều tự động thực thi qua pipeline tối ưu 2 giai đoạn:
1. **Giai đoạn 1 (BM25 Top 100)**: Luôn dùng BM25 để lọc ra top 100 ứng viên (`RETRIEVAL_BM25 = 100`) dựa trên từ khóa câu hỏi.
2. **Giai đoạn 2 (Dense Search trong Top 100)**: Lấy top 100 ứng viên này để tính điểm tương đồng ngữ nghĩa (cosine similarity) trong Vector Store / Graph Store.
3. **Giai đoạn 3 (Reciprocal Rank Fusion - RRF)**: Chạy thuật toán RRF với $k = 60$ (`RRF_K = 60`) để hợp nhất thứ hạng giữa BM25 và Dense:
   $$RRF(d) = \frac{1}{60 + rank_{BM25}(d)} + \frac{1}{60 + rank_{Dense}(d)}$$
   Sắp xếp điểm số giảm dần và chọn ra top 20 ứng viên tốt nhất (`RETRIEVAL_DENSE = 20`).
4. **Giai đoạn 4 (Cross-Encoder Reranking & Context Building)**: Đưa 20 ứng viên qua mô hình Cross-Encoder `BAAI/bge-reranker-v2-m3` để chọn top 5 (`RERANK_LIMIT = 5`), kết hợp mở rộng các điều luật viện dẫn (`expand_references`).

#### 4.2. Kiểm tra truy xuất ngữ cảnh (Retrieval only — không gọi LLM)
```bash
uv run python Scripts/retrieve.py --query "Hạn mức giao đất ở là bao nhiêu?" --no-generate
```

#### 4.3. Kiểm tra toàn bộ pipeline sinh câu trả lời (Full RAG Generation)
```bash
uv run python Scripts/retrieve.py --query "Hạn mức giao đất ở là bao nhiêu?"
```


---

### Bước 5: Chạy đánh giá định lượng (Evaluation Pipeline)

Chạy script đánh giá benchmark trên tập câu hỏi Luật Đất đai 2024 với cơ chế **Streaming per-case**, **Auto-Resume** và **Checkpointing per-batch cho RAGAS**.

#### 5.1. Các chế độ Retriever chính (`--retriever-mode`)

* **1. Chế độ Dense cơ bản (Base BGE-M3 trên Qdrant)**:
  ```bash
  uv run python Scripts/evaluate.py --retriever-mode base --postprocessing filter_rerank --limit 10
  ```

* **2. Chế độ Contriever (facebook/contriever trên Qdrant)**:
  ```bash
  uv run python Scripts/evaluate.py --retriever-mode contriever --limit 10
  ```

* **3. Chế độ Graph Database (Microsoft GraphRAG)**:
  ```bash
  uv run python Scripts/evaluate.py --retriever-mode graph
  # hoặc tùy chọn phương thức truy vấn Graph (local | global | drift | basic):
  uv run python Scripts/evaluate.py --retriever-mode graph --graph-method local
  # Cờ --graph là alias viết tắt tương đương:
  uv run python Scripts/evaluate.py --graph
  ```

#### 5.2. Các kịch bản đánh giá nâng cao

* **Đánh giá tiêu chuẩn đầy đủ (Phase 1 tuần tự an toàn GPU + RAGAS đa luồng)**:
  ```bash
  uv run python Scripts/evaluate.py --retriever-mode base --postprocessing filter_rerank --llm-service google --sub-llm-service google --ragas-service nvidia --max-workers 1 --ragas-max-workers 4
  ```

* **Đánh giá cơ bản nhanh (Bỏ qua LLM Judge để tiết kiệm API call)**:
  ```bash
  uv run python Scripts/evaluate.py --retriever-mode base --limit 10 --skip-ragas
  ```

* **Đánh giá với RAG-Fusion & Tiền/Hậu xử lý nâng cao**:
  ```bash
  uv run python Scripts/evaluate.py --retriever-mode base --advanced rag_fusion --preprocessing hyde --postprocessing filter_rerank --limit 10
  ```

* **Kết hợp GraphRAG & RAG-Fusion**:
  ```bash
  uv run python Scripts/evaluate.py --retriever-mode graph --advanced rag_fusion
  ```

#### 5.3. Tùy chỉnh Concurrency & Auto-Resume

* **Tùy chỉnh số luồng độc lập (Decoupled Concurrency)**:
  - `--max-workers <N>`: Số luồng xử lý đồng thời cho Phase 1 (Retrieval & Generation). Đặt `1` khi dùng mô hình reranker trên CUDA để tránh nghẽn VRAM.
  - `--ragas-max-workers <N>`: Số luồng xử lý đồng thời cho Phase 3 (RAGAS LLM-as-a-judge). Mặc định `4` để bắn song song các request LLM Judge giúp hoàn thành nhanh chóng.
  - `--ragas-batch-size <N>`: Kích thước batch cho RAGAS (mặc định: `1` - xong câu nào ghi câu đó ngay lập tức xuống đĩa).
  - `--batch-size <N>`: Kích thước batch xử lý Phase 1 (mặc định: `10`).

* **Cơ chế Bảo Lưu & Auto-Resume**:
  - Hệ thống tự động nhận diện phiên trước nếu trùng khớp metadata cấu hình và tiếp tục các câu hỏi còn thiếu.
  - **Bảo lưu 100%**: Các câu hỏi đã có đủ 4 điểm RAGAS hợp lệ sẽ không bị đánh giá lại, hệ thống chỉ chạy bổ sung cho các câu bị thiếu/lỗi (`None`).
  - Dùng cờ `--no-resume` nếu muốn bỏ qua phiên trước và đánh giá mới hoàn toàn từ đầu:
    ```bash
    uv run python Scripts/evaluate.py --no-resume
    ```

* **Lưu trữ & Chạy song song không xung đột (Parallel Multi-CMD Execution)**:
  - File báo cáo chi tiết theo thời gian: `db/results/eval_report_YYYYMMDD_HHMMSS.json`.
  - File đồng bộ mới nhất theo Signature cấu hình pipeline: `db/results/eval_latest_{signature}.json`.
    - Ví dụ: `eval_latest_base_hyde_crag_nvidia_nvidia_google.json`
    - Nếu là advanced: `eval_latest_base_<advanced_method>_nvidia_nvidia_google.json`
    - Nếu là graph kèm postprocessing: `eval_latest_graph_filter_rerank_nvidia_nvidia_google.json`
  - Cơ chế này cho phép chạy song song nhiều terminal với các phương pháp khác nhau mà hoàn toàn không bị ghi đè hay lỗi lock file.
  - Ở đầu JSON luôn có khối `summary_metrics` tổng hợp điểm trung bình macro-average lũy kế theo thời gian thực, và từng case trong `detailed_results` đều được chuẩn hóa đầy đủ 100% các metrics (Retrieval, Generation, Latency, RAGAS).

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

