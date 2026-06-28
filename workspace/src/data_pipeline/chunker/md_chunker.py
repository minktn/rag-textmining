import uuid
import hashlib
from .base_chunker import BaseChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class MDChunker(BaseChunker):
	def __init__(self, headers_to_split_on=None, chunk_size=1000, chunk_overlap=100):
		if headers_to_split_on is None:
			headers_to_split_on = [
				("#", "source"),
				("##", "chapter"),
				("###", "section"),
				("####", "article"),
			]
		
		self.md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
		self.recursive_splitter = RecursiveCharacterTextSplitter(
			chunk_size=chunk_size,
			chunk_overlap=chunk_overlap,
			separators=["\n\n", "\n", ".", " "]
		)

	def chunking(self, text: str):
		md_docs = self.md_splitter.split_text(text)
		raw_chunks = self.recursive_splitter.split_documents(md_docs)
		
		chunks = []
		for doc in raw_chunks:
			salt = doc.metadata.get("article", "no_article")
			id = hashlib.md5(f"{salt}::{doc.page_content}".encode("utf-8")).hexdigest()
			chunks.append({
				"id": str(uuid.UUID(id)),
				"metadata": doc.metadata,
				"content": doc.page_content,
			})

		return chunks