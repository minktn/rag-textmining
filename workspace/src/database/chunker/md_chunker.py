import uuid
import hashlib
import re
from .base_chunker import BaseChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class MDChunker(BaseChunker):
	def __init__(self, headers_to_split_on=None, chunk_size=1000):
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
			chunk_overlap=0,
			separators=["\n[clause]"]
		)

	def chunking(self, text: str):
		md_docs = self.md_splitter.split_text(text)
		raw_chunks = self.recursive_splitter.split_documents(md_docs)
		
		chunks = []
		for doc in raw_chunks:
			salt = doc.metadata.get("article", "no_article")
			id = hashlib.md5(f"{salt}::{doc.page_content}".encode("utf-8")).hexdigest()

			clause_tag = re.compile(r'\[clause\] (\d+)')
			res = clause_tag.findall(doc.page_content)
			content = clause_tag.sub(r'\1', doc.page_content)

			clause_nos = [int(num) for num in res] if res else []

			metadata = doc.metadata
			metadata['clause_nos'] = clause_nos

			chunks.append({
				"id": str(uuid.UUID(id)),
				"metadata": metadata,
				"content": content,
			})

		return chunks