import logging
from typing import Optional, Any

from src.generation.sub_llm_manager import SubLLMManager

logger = logging.getLogger(__name__)

REWRITE_PROMPT_TEMPLATE = (
    "Bạn là chuyên gia tra cứu thông tin pháp luật Việt Nam.\n"
    "Hãy phân tích câu hỏi người dùng và viết lại thành 1 câu truy vấn ngắn gọn, chuẩn xác thuật ngữ Luật để tra cứu trong cơ sở dữ liệu.\n\n"
    "Quy tắc:\n"
    "1. Giữ nguyên ý nghĩa cốt lõi của câu hỏi.\n"
    "2. CHỈ TRẢ VỀ DUY NHẤT 1 CÂU TRUY VẤN VIẾT LẠI, KHÔNG GIẢI THÍCH, KHÔNG THÊM BẤT KỲ LỜI DẪN NÀO.\n\n"
    "Câu hỏi gốc: {query}\n"
    "Câu truy vấn viết lại:"
)


def rewrite_query(query: str, llm: Optional[Any] = None) -> str:
    """Viết lại câu hỏi người dùng thành câu truy vấn chuẩn luật trước khi retrieval."""
    if not query or not query.strip():
        return query

    try:
        if llm is not None and hasattr(llm, "invoke"):
            # Hỗ trợ LangChain model nếu được truyền vào
            from langchain_core.prompts import PromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            prompt_tmpl = PromptTemplate.from_template(REWRITE_PROMPT_TEMPLATE)
            chain = prompt_tmpl | llm | StrOutputParser()
            rewritten = str(chain.invoke({"query": query.strip()})).strip().strip("'\"")
        else:
            # Sử dụng SubLLMManager chuẩn của hệ thống
            sub_llm = llm or SubLLMManager()
            orig_system_prompt = sub_llm.system_prompt
            sub_llm.system_prompt = "Bạn là chuyên gia tra cứu thông tin pháp luật Việt Nam."
            try:
                formatted_prompt = REWRITE_PROMPT_TEMPLATE.format(query=query.strip())
                rewritten = sub_llm.generate_response(formatted_prompt)
            finally:
                sub_llm.system_prompt = orig_system_prompt

        if rewritten:
            if isinstance(rewritten, list):
                rewritten = " ".join(str(x) for x in rewritten)
            else:
                rewritten = str(rewritten)
            rewritten = rewritten.strip().strip("'\"")
            # Nếu có nhiều dòng, lấy dòng đầu tiên không rỗng
            if "\n" in rewritten:
                lines = [
                    l.strip()
                    for l in rewritten.split("\n")
                    if l.strip() and not l.strip().startswith("```")
                ]
                rewritten = lines[0] if lines else rewritten

            return rewritten if rewritten else query
        return query
    except Exception as e:
        logger.warning(f"Lỗi khi rewrite query '{query}': {e}. Sử dụng query gốc.")
        return query
