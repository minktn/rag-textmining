"""
SLM Filter: Local Small Language Model Document Classifier (4-bit Quantized)
=============================================================================
First-stage filtering component of the Filter-then-Rerank paradigm (EMNLP 2023).
Evaluates candidate text chunks via a single forward-pass logit distribution:
- Easy Relevant  (s >= tau_high): High confidence of relevance, retained immediately.
- Easy Irrelevant (s <= tau_low) : High confidence of irrelevance, discarded.
- Hard Samples   (tau_low < s < tau_high): Ambiguous cases, routed to LLM Reranker.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Fixed local SLM model configuration (independent of settings.py)
DEFAULT_SLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SLMFilter:
    """Document relevance filter driven by a local Small Language Model (SLM) in 4-bit."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        tau_high: float = 0.70,
        tau_low: float = 0.20,
        batch_size: int = 4,
    ):
        self.model_name = model_name or DEFAULT_SLM_MODEL
        self.device = device or DEFAULT_DEVICE
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.batch_size = batch_size

        self._tokenizer = None
        self._model = None
        self._token_a_id: Optional[int] = None
        self._token_b_id: Optional[int] = None

    def _load_model(self) -> None:
        """Lazily initialize the local SLM with 4-bit quantization and cache target choice token IDs."""
        if self._model is not None and self._tokenizer is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"[SLMFilter] Loading 4-bit local SLM: '{self.model_name}' on device '{self.device}'...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        bnb_config = None
        if self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device != "cuda":
            self._model.to(self.device)
        self._model.eval()

        # Cache canonical ASCII token IDs for binary evaluation choices 'A' and 'B'
        self._token_a_id = self._tokenizer.encode("A", add_special_tokens=False)[0]
        self._token_b_id = self._tokenizer.encode("B", add_special_tokens=False)[0]
        logger.info(f"[SLMFilter] Ready (4-bit). Choice token IDs: A={self._token_a_id}, B={self._token_b_id}")

    def score_chunk(self, query: str, chunk_text: str) -> float:
        """
        Compute continuous relevance confidence s in [0, 1] via single forward-pass logits.

        Evaluates the conditional probability P(A | prompt) over binary choices
        A (Relevant) and B (Irrelevant), with a coverage threshold to detect distribution shifts.
        """
        self._load_model()

        prompt = (
            f"<|im_start|>system\n"
            f"Given a legal query and a retrieved legal document passage, evaluate whether the passage is relevant to answering the query.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Query: {query}\n\n"
            f"Document Passage:\n{chunk_text[:1200]}\n\n"
            f"Is the passage relevant to answering the query?\n"
            f"A. Relevant\n"
            f"B. Irrelevant\n\n"
            f"Answer:<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)

            prob_a = probs[self._token_a_id].item()
            prob_b = probs[self._token_b_id].item()
            coverage = prob_a + prob_b

            # Fallback to neutral score if probability mass escapes canonical choice tokens
            if coverage < 0.25:
                logger.debug(f"[SLMFilter] Low choice coverage ({coverage:.4f} < 0.25). Marked as hard sample.")
                return 0.50

            score = prob_a / coverage
            return float(round(score, 4))

    def filter_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Partition candidates into easy relevant, hard ambiguous, and dropped subsets.

        Parameters
        ----------
        query : str
            User search query.
        chunks : List[Dict[str, Any]]
            Retrieved candidate chunks to evaluate.

        Returns
        -------
        Tuple[List[Dict], List[Dict], List[Dict]]
            (easy_chunks, hard_chunks, dropped_chunks)
        """
        if not chunks:
            return [], [], []

        easy_chunks: List[Dict[str, Any]] = []
        hard_chunks: List[Dict[str, Any]] = []
        dropped_chunks: List[Dict[str, Any]] = []

        for chunk in chunks:
            content = chunk.get("content") or chunk.get("text") or ""
            try:
                score = self.score_chunk(query, content)
            except Exception as e:
                logger.warning(f"[SLMFilter] Scoring failure on chunk: {e}. Defaulting to hard sample.")
                score = 0.50

            chunk_copy = dict(chunk)
            chunk_copy["slm_score"] = round(score, 4)

            if score >= self.tau_high:
                chunk_copy["filter_status"] = "easy_relevant"
                easy_chunks.append(chunk_copy)
            elif score <= self.tau_low:
                chunk_copy["filter_status"] = "easy_irrelevant"
                dropped_chunks.append(chunk_copy)
            else:
                chunk_copy["filter_status"] = "hard_sample"
                hard_chunks.append(chunk_copy)

        # Fallback mechanism: salvage highest-scoring candidates if all fall below threshold
        if not easy_chunks and not hard_chunks and dropped_chunks:
            dropped_chunks.sort(key=lambda x: x.get("slm_score", 0.0), reverse=True)
            salvaged_count = min(len(dropped_chunks), self.batch_size)
            hard_chunks = dropped_chunks[:salvaged_count]
            dropped_chunks = dropped_chunks[salvaged_count:]
            for c in hard_chunks:
                c["filter_status"] = "hard_sample"

        logger.info(
            f"[SLMFilter] Filtering completed: {len(chunks)} chunks -> "
            f"Easy: {len(easy_chunks)}, Hard: {len(hard_chunks)}, Dropped: {len(dropped_chunks)}"
        )
        return easy_chunks, hard_chunks, dropped_chunks
