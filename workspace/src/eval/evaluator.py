"""
RAG Evaluator — Pipeline Execution
====================================
Orchestrate retrieve → generate → evaluate cho từng câu hỏi.
Chỉ gọi Qdrant MỘT LẦN per question (loại bỏ duplicate query).
"""

import time

from src.data_pipeline import DenseEmbedder
from src.database import DBManager
from src.llm import LLMManager


class RAGEvaluator:
    """Orchestrator cho toàn bộ RAG evaluation pipeline.

    Tách riêng các bước retrieve / generate / evaluate để dễ mở rộng
    và test từng phần độc lập.
    """

    def __init__(
        self,
        embedder: DenseEmbedder,
        db_manager: DBManager,
        llm_manager: LLMManager,
        model_name: str,
        collection_name: str = 'landlaw',
        top_k: int = 5,
    ):
        self.embedder = embedder
        self.db_manager = db_manager
        self.llm_manager = llm_manager
        self.model_name = model_name
        self.collection_name = collection_name
        self.top_k = top_k

    # ── Retrieve ──────────────────────────────────────────────────

    def retrieve(self, question: str) -> dict:
        """Truy vấn Qdrant MỘT LẦN DUY NHẤT, trả về contexts + payloads + latency.

        Trước refactor, code gọi 2 lần Qdrant cho cùng 1 câu hỏi:
          1. db_manager.query_dense()  → chỉ lấy content
          2. db_manager.client.query_points() → lấy payload
        Bây giờ chỉ gọi query_points() 1 lần, extract cả hai.
        """
        t0 = time.perf_counter()
        query_vector = self.embedder.embed_single(question)

        raw_results = self.db_manager.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using='dense',
            limit=self.top_k,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        # Extract content + full payload from single query
        contexts = [
            point.payload.get('content', '')
            for point in raw_results.points
        ]
        payloads = [point.payload for point in raw_results.points]

        # Format docs cho LLM prompt (tương thích với LLMManager.construct_prompt)
        docs = [{'content': c} for c in contexts]

        return {
            'contexts': contexts,
            'payloads': payloads,
            'docs': docs,
            'latency_ms': round(latency_ms, 2),
        }

    # ── Generate ──────────────────────────────────────────────────

    def generate(self, question: str, docs: list[dict]) -> dict:
        """Generate answer từ LLM, trả về answer + latency."""
        t0 = time.perf_counter()
        prompt = self.llm_manager.construct_prompt(question, docs=docs)
        answer = self.llm_manager.generate_response(prompt, model_name=self.model_name)
        latency_ms = (time.perf_counter() - t0) * 1000

        if answer is None:
            answer = ""

        return {
            'answer': answer,
            'latency_ms': round(latency_ms, 2),
        }

    # ── Single question ───────────────────────────────────────────

    def evaluate_single(self, item: dict) -> dict:
        """Chạy retrieve + generate cho 1 câu hỏi, trả về kết quả chi tiết."""
        from src.eval.metrics import MetricsCalculator

        question = item['question']

        retrieval = self.retrieve(question)
        generation = self.generate(question, retrieval['docs'])

        # Trích xuất law_id chuẩn hóa từ retrieved payloads để so sánh
        retrieved_law_ids = MetricsCalculator.extract_law_ids_from_payloads(
            retrieval['payloads']
        )

        return {
            'id': item['id'],
            'question': question,
            'question_type': item.get('question_type', ''),
            'ground_truth': item['answer'],
            'generated_answer': generation['answer'],
            'retrieved_contexts': retrieval['contexts'],
            'retrieved_payloads': retrieval['payloads'],
            'retrieved_law_ids': retrieved_law_ids,
            'law_id': item.get('law_id', {}),
            'retrieval_latency_ms': retrieval['latency_ms'],
            'generation_latency_ms': generation['latency_ms'],
            'e2e_latency_ms': round(retrieval['latency_ms'] + generation['latency_ms'], 2),
        }

    # ── Full dataset ──────────────────────────────────────────────

    def evaluate_all(self, questions: list[dict]) -> list[dict]:
        """Chạy pipeline cho toàn bộ dataset với progress indicator."""
        results = []
        total = len(questions)

        for idx, item in enumerate(questions, 1):
            qid = item['id']
            print(f"\r  ⏳ [{idx}/{total}] {qid}: {item['question'][:60]}...", end='', flush=True)

            result = self.evaluate_single(item)
            results.append(result)

        print()  # newline after progress
        return results
