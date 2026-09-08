"""
Filter-then-Rerank: LLM Multi-Choice Reranker
=============================================
Implements candidate reranking for hard/ambiguous samples identified
by the SLM filter, utilizing multi-choice prompting and in-context examples.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.generation.sub_llm_manager import SubLLMManager

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Please answer the multi-choice questions below. Note that you need to strictly "
    "follow the format ending with 'Correct Answer: ([your choice])'"
)

FEW_SHOT_DEMONSTRATIONS: List[Dict[str, str]] = [
    {
        "role": "user",
        "content": (
            "Sentence: Thời kỳ quy hoạch sử dụng đất quốc gia và quy hoạch sử dụng đất cấp tỉnh là bao nhiêu năm?\n\n"
            "Candidate Choices:\n"
            "(a) [Điều 61. Hệ thống quy hoạch, kế hoạch sử dụng đất]: 1. Hệ thống quy hoạch, kế hoạch sử dụng đất bao gồm: a) Quy hoạch, kế hoạch sử dụng đất quốc gia; b) Quy hoạch, kế hoạch sử dụng đất cấp tỉnh...\n"
            "(b) [Điều 62. Thời kỳ quy hoạch, kế hoạch sử dụng đất]: 1. Thời kỳ, tầm nhìn quy hoạch sử dụng đất quốc gia thực hiện theo quy định của Luật Quy hoạch. 2. Thời kỳ, tầm nhìn quy hoạch sử dụng đất cấp tỉnh thống nhất với thời kỳ, tầm nhìn của quy hoạch tỉnh...\n"
            "(c) None: Không có đoạn nào phù hợp.\n\n"
        ),
    },
    {
        "role": "assistant",
        "content": "Correct Answer: (b)",
    },
    {
        "role": "user",
        "content": (
            "Sentence: Các thủ tục hành chính về đất đai được thực hiện bằng hình thức trực tuyến trên môi trường điện tử có giá trị pháp lý tương đương với hình thức nộp trực tiếp không?\n\n"
            "Candidate Choices:\n"
            "(a) [Điều 224. Nguyên tắc thực hiện thủ tục hành chính về đất đai]: 4. Các thủ tục hành chính về đất đai được thực hiện bằng hình thức trực tiếp, qua hệ thống bưu chính hoặc trên môi trường điện tử và các hình thức này có giá trị pháp lý hoàn toàn như nhau.\n"
            "(b) [Điều 223. Các thủ tục hành chính về đất đai]: 1. Các thủ tục hành chính về đất đai bao gồm: a) Thủ tục thu hồi đất, giao đất, cho thuê đất...\n"
            "(c) None: Không có đoạn nào phù hợp.\n\n"
        ),
    },
    {
        "role": "assistant",
        "content": "Correct Answer: (a)",
    },
]


class LLMReranker:
    """Multi-choice candidate reranker for hard/ambiguous document samples."""

    def __init__(
        self,
        sub_llm_manager: Optional[SubLLMManager] = None,
        model_name: Optional[str] = None,
        topk: int = 5,
    ):
        if sub_llm_manager is not None and sub_llm_manager.service != "local":
            self.sub_llm = sub_llm_manager
        else:
            self.sub_llm = SubLLMManager(service="nvidia")

        self.service = self.sub_llm.service
        if model_name:
            self.model_name = model_name
        elif self.service == "nvidia":
            self.model_name = getattr(settings, "NVIDIA_LLM", "nvidia/nemotron-3.5-lightning-30b-a3b")
        elif self.service in ("gemini", "google"):
            self.model_name = getattr(settings, "GEMINI_LLM", "gemma-4-31b-it")
        elif self.service == "groq":
            self.model_name = getattr(settings, "GROQ_LLM", "llama-3.3-70b-versatile")
        else:
            self.model_name = self.sub_llm.get_default_model(self.service)

        self.topk = topk
        logger.info(
            f"[LLMReranker] Initialized reranker: service='{self.service}', "
            f"model='{self.model_name}', topk={self.topk}"
        )

    def build_prompt(self, query: str, candidate_chunks: List[Dict[str, Any]]) -> str:
        """Format user query and candidate passages into a standardized multi-choice prompt."""
        lines = [f"Sentence: {query}\n", "Candidate Choices:"]
        for i, chunk in enumerate(candidate_chunks[:self.topk]):
            letter = chr(ord("a") + i)
            meta = chunk.get("metadata") or chunk.get("payload") or {}
            title = meta.get("article") or meta.get("source") or f"Tài liệu {i+1}"
            raw_text = chunk.get("content") or chunk.get("text") or ""
            snippet = raw_text[:1200].replace("\n", " ").strip()
            lines.append(f"({letter}) [{title}]: {snippet}")

        none_letter = chr(ord("a") + min(len(candidate_chunks), self.topk))
        lines.append(f"({none_letter}) None: Không có đoạn nào trong các lựa chọn trên chứa thông tin giải quyết câu hỏi.")
        return "\n".join(lines) + "\n\n"

    def parse_res(self, raw_res: str, num_candidates: int) -> List[int]:
        """Parse candidate choice indices from model response text using regex pattern matching."""
        if not raw_res:
            return []

        text_to_search = raw_res
        match = re.search(r"Correct Answer:\s*(.+)", raw_res, re.IGNORECASE)
        if match:
            text_to_search = match.group(1)

        choices = re.findall(r"\(([a-zA-Z])\)", text_to_search)
        if not choices:
            choices = re.findall(r"\b([a-zA-Z])\b", text_to_search)

        selected_indices: List[int] = []
        for choice in choices:
            idx = ord(choice.lower()) - ord("a")
            if 0 <= idx < num_candidates:
                if idx not in selected_indices:
                    selected_indices.append(idx)
            elif idx == num_candidates:
                break
        return selected_indices

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=2, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_llm_multi_choice(self, user_prompt: str) -> str:
        """Execute multi-choice prompt against the configured LLM service with strict token bounds."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(FEW_SHOT_DEMONSTRATIONS)
        messages.append({"role": "user", "content": user_prompt})

        if self.service == "nvidia" and hasattr(self.sub_llm, "nvidia_client"):
            response = self.sub_llm.nvidia_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=64,
                stop=["\n\n", "\n"],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                timeout=30.0,
            )
            return response.choices[0].message.content or ""

        elif self.service in ("google", "gemini") and hasattr(self.sub_llm, "gemini_key"):
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
            from langchain_core.output_parsers import StrOutputParser
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.sub_llm.gemini_key,
                temperature=0.0,
                max_output_tokens=64,
            )
            lc_messages = [SystemMessage(content=SYSTEM_PROMPT)]
            for demo in FEW_SHOT_DEMONSTRATIONS:
                if demo["role"] == "user":
                    lc_messages.append(HumanMessage(content=demo["content"]))
                else:
                    lc_messages.append(AIMessage(content=demo["content"]))
            lc_messages.append(HumanMessage(content=user_prompt))

            chain = llm | StrOutputParser()
            return chain.invoke(lc_messages)

        else:
            orig_prompt = self.sub_llm.system_prompt
            try:
                self.sub_llm.system_prompt = SYSTEM_PROMPT
                res = self.sub_llm.generate_response(
                    prompt=user_prompt,
                    model_name=self.model_name,
                    service=self.service,
                )
                return str(res)
            finally:
                self.sub_llm.system_prompt = orig_prompt

    def rerank_hard_samples(
        self,
        query: str,
        hard_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Evaluate and rerank ambiguous/hard document chunks using multi-choice LLM reasoning.

        Parameters
        ----------
        query : str
            User query.
        hard_chunks : List[Dict[str, Any]]
            Ambiguous chunks identified by the first-stage SLM filter.

        Returns
        -------
        List[Dict[str, Any]]
            Re-scored and ranked candidates.
        """
        if not hard_chunks:
            return []

        candidates = hard_chunks[:self.topk]
        overflow_chunks = hard_chunks[self.topk:]

        user_prompt = self.build_prompt(query, candidates)
        try:
            raw_response = self._call_llm_multi_choice(user_prompt)
            selected_indices = self.parse_res(raw_response, len(candidates))
        except Exception as err:
            logger.error(f"[LLMReranker] Reranking inference error: {err}. Preserving initial ranking.")
            selected_indices = list(range(len(candidates)))
            raw_response = "fallback"

        rank_scores = [round(0.95 - (i * 0.10), 4) for i in range(len(candidates))]
        reranked_chunks: List[Dict[str, Any]] = []

        for i, chunk in enumerate(candidates):
            chunk_copy = dict(chunk)
            chunk_copy["source"] = "slm_filter_llm_reranked"
            chunk_copy["llm_raw_response"] = raw_response

            if i in selected_indices:
                rank_pos = selected_indices.index(i)
                assigned_score = rank_scores[rank_pos] if rank_pos < len(rank_scores) else 0.70
                chunk_copy["llm_rerank_score"] = assigned_score
                chunk_copy["final_score"] = assigned_score
                chunk_copy["llm_choice_selected"] = True
            else:
                base_score = chunk.get("slm_score", 0.50)
                assigned_score = round(max(0.10, base_score * 0.4), 4)
                chunk_copy["llm_rerank_score"] = assigned_score
                chunk_copy["final_score"] = assigned_score
                chunk_copy["llm_choice_selected"] = False

            reranked_chunks.append(chunk_copy)

        for chunk in overflow_chunks:
            chunk_copy = dict(chunk)
            chunk_copy["source"] = "slm_filter_overflow_unranked"
            chunk_copy["final_score"] = round(chunk.get("slm_score", 0.30) * 0.3, 4)
            reranked_chunks.append(chunk_copy)

        reranked_chunks.sort(key=lambda c: c.get("final_score", 0.0), reverse=True)
        return reranked_chunks
