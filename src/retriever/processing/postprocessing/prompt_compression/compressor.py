from typing import List, Dict, Any
import logging

try:
    from llmlingua import PromptCompressor
except ImportError:
    PromptCompressor = None
    logging.warning("Please install llmlingua: pip install llmlingua")

class LongLLMLinguaCompressor:
    def __init__(
        self,
        # Nên dùng mô hình Causal LM (nhỏ gọn) có khả năng tiếng Việt tốt để tính Perplexity
        # Qwen2.5-1.5B hoặc 0.5B là lựa chọn tuyệt vời cho tiếng Việt trong LongLLMLingua
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", 
        device: str = "cuda" # Hoặc "cpu" nếu không có GPU
    ):
        """
        Khởi tạo Prompt Compressor bằng thư viện LLMLingua.
        Sử dụng kỹ thuật LongLLMLingua (đặc trị cho RAG).
        """
        if PromptCompressor is None:
            raise ImportError("Thư viện llmlingua chưa được cài đặt.")
            
        self.compressor = PromptCompressor(
            model_name=model_name,
            device_map=device,
        )
        
    def compress_retrieved_context(
        self, 
        question: str, 
        contexts: List[str], 
        target_token: int = 500
    ) -> Dict[str, Any]:
        """
        Nén các đoạn văn bản truy xuất được dựa trên nhận thức về câu hỏi (Question-Aware).
        
        Args:
            question: Câu hỏi của người dùng
            contexts: Danh sách các đoạn văn bản thô truy xuất từ Qdrant / Graph DB
            target_token: Số token tối đa muốn giữ lại sau khi nén
            
        Returns:
            Dictionary chứa ngữ cảnh đã nén và các thống kê
        """
        # LongLLMLingua khuyến nghị cách cấu hình tham số như sau cho RAG:
        results = self.compressor.compress_prompt(
            context=contexts,
            question=question,
            target_token=target_token,
            rank_method="longllmlingua",         # Sử dụng LongLLMLingua
            condition_compare=True,              # Bật so sánh điều kiện
            condition_in_question="after",       # Đặt câu hỏi ở sau để tính Perplexity tốt hơn
            context_budget="+100",
            dynamic_context_compression_ratio=0.4, # Tỷ lệ nén động (giữ lại 40%)
            reorder_context="original"           # Sắp xếp ngữ cảnh: 'original' hoặc 'two_stage'
        )
        
        origin_tokens = results.get("origin_tokens", 0)
        compressed_tokens = results.get("compressed_tokens", 0)
        saving_ratio = (
            (origin_tokens - compressed_tokens) / origin_tokens
            if origin_tokens > 0 else 0.0
        )
        
        # Kết quả trả về gồm có 'compressed_prompt' (chuỗi đã nén)
        return {
            "compressed_context": results.get("compressed_prompt", ""),
            "origin_tokens": origin_tokens,
            "compressed_tokens": compressed_tokens,
            "saving_ratio": saving_ratio
        }

