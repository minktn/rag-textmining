from groq import Groq

class LLMManager:
	def __init__(self, api_key, temperature=0.1):
		self.client = Groq(api_key=api_key)
		self.temperature = temperature

		self.system_prompt = (
            'Bạn là trợ lý AI chuyên nghiệp. Hãy trả lời bằng tiếng Việt '
            'dựa trên ngữ cảnh được cung cấp.'
        )

	def construct_prompt(self, query, docs=None):
		if docs:
			context = '\n\n'.join(
				[f'Tài liệu {i+1}: {doc["content"]}' for i, doc in enumerate(docs)]
			)
			return f'NGỮ CẢNH:\n{context}\n\nCÂU HỎI:\n{query}'
		else:
			return f'CÂU HỎI:\n{query}'

	def generate_response(self, prompt, model_name):
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
			print(f'Error generating response: {e}')
			return None