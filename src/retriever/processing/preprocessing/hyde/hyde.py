import torch

from src.generation.sub_llm_manager import SubLLMManager
from src.database.embedder import DenseEmbedder
from src.config import settings

class HyDE:
    def __init__(self, llm_manager: SubLLMManager = None, embedder: DenseEmbedder = None):
        """
        Initialize the HyDE (Hypothetical Document Embeddings) pipeline.
        Sử dụng DenseEmbedder mặc định của dự án để phù hợp với Qdrant collection.
        """
        self.llm_manager = llm_manager or SubLLMManager()
        
        # Nếu không truyền embedder, tự động khởi tạo bằng config
        if embedder:
            self.embedder = embedder
        else:
            dense_model = getattr(settings, "EMBEDDING_MODEL", "BAAI/bge-m3")
            self.embedder = DenseEmbedder(dense_model)

    def create_hypothetic_document(self, query: str) -> str:
        """
        1) prompt bằng LLM để sinh ra một tài liệu giả định dựa trên câu hỏi của người dùng.
        """
        prompt = f"""Please write a legal answering passage in Vietnamese to answer the question in detail.
Question: {query}
Passage:"""
        
        # Backup system prompt cũ và thiết lập prompt phù hợp cho instruction HyDE
        original_system_prompt = self.llm_manager.system_prompt
        self.llm_manager.system_prompt = "You are a helpful AI assistant."
        
        try:
            hypothetic_document = self.llm_manager.generate_response(prompt)
        finally:
            self.llm_manager.system_prompt = original_system_prompt
            
        return hypothetic_document.strip() if hypothetic_document else ""

    def get_embeddings(self, document: str) -> list:
        """
        2) Chạy tài liệu giả định qua model DenseEmbedder hiện tại của dự án
        để đảm bảo cùng số chiều vector với Database Qdrant.
        """
        # DenseEmbedder.embed_single sẽ trả về embedding chuẩn của dự án (thường là list[float])
        embeddings = self.embedder.embed_single(document)
        return embeddings

    def process(self, query: str) -> dict:
        """
        Thực hiện toàn bộ pipeline HyDE: sinh tài liệu giả định và tạo vector nhúng.
        """
        hypothetic_document = self.create_hypothetic_document(query)
        
        if not hypothetic_document:
            raise RuntimeError("LLM không thể sinh ra tài liệu giả định (Hypothetical Document).")
            
        embeddings = self.get_embeddings(hypothetic_document)
        
        return {
            "query": query,
            "hypothetic_document": hypothetic_document,
            "embeddings": embeddings
        }
