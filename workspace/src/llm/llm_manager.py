import time
from groq import Groq

class LLMManager:
	# Rate limiting: 30 RPM -> 1 request every 2.0 seconds minimum delay
	_last_request_time = 0.0
	_min_delay = 2.0

	def __init__(self, api_key, temperature=0.1):
		from groq import Groq

		self.client = Groq(api_key=api_key)
		self.temperature = temperature

		self.system_prompt = (
			'Bạn là trợ lý AI chuyên nghiệp. '
			'Hãy trả lời bằng tiếng Việt '
			'dựa trên ngữ cảnh được cung cấp.'
		)

	def construct_prompt(self, query, docs=None):
		if docs:
			context = '\n\n'.join(
				self._format_doc(doc, index)
				for index, doc in enumerate(docs, start=1)
			)
			return f'NGỮ CẢNH:\n{context}\n\nCÂU HỎI:\n{query}'
		return f'CÂU HỎI:\n{query}'

	def _format_doc(self, doc, index):
		metadata = doc.get('metadata') or doc.get('payload') or {}
		lines = [f'Tài liệu {index}:']

		for key, label in (
			('source', 'Nguồn'),
			('chapter', 'Chương'),
			('section', 'Mục'),
			('article', 'Điều'),
			('article_no', 'Số điều'),
			('chapter_no', 'Số chương'),
			('section_no', 'Số mục'),
			('clause_nos', 'Các khoản'),
			('ref_article_nos', 'Điều được viện dẫn'),
		):
			value = metadata.get(key)
			if value not in (None, '', []):
				lines.append(f'{label}: {value}')

		if doc.get('dense_score') is not None:
			lines.append(f'Điểm truy xuất dense: {doc["dense_score"]}')
		if doc.get('rerank_score') is not None:
			lines.append(f'Điểm xếp hạng lại: {doc["rerank_score"]}')
		if doc.get('source'):
			lines.append(f'Nguồn truy xuất: {doc["source"]}')

		lines.append('Nội dung:')
		lines.append(str(doc.get('content', '')))
		return '\n'.join(lines)

	def generate_response(self, prompt, model_name):
		# Enforce rate limit delay
		now = time.time()
		elapsed = now - LLMManager._last_request_time
		if elapsed < LLMManager._min_delay:
			time.sleep(LLMManager._min_delay - elapsed)
		LLMManager._last_request_time = time.time()

		try:
			response = self.client.chat.completions.create(
				messages=[
					{'role': 'system', 'content': self.system_prompt},
					{'role': 'user', 'content': prompt}
				],
				model=model_name,
				temperature=self.temperature
			)
			return response.choices[0].message.content

		except Exception as e:
			print(f'Lỗi khi tạo câu trả lời: {e}')
			return None
