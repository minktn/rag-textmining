import re
import unicodedata
from typing import Any, Mapping


class LegalMetadataProcessor:
	METADATA_FIELDS = [
		'article_no',
		'chapter_no',
		'section_no',
		'clause_nos',
		'ref_article_nos',
		'normalized_source_text',
	]

	ARTICLE_RE = re.compile(r'(?<!\w)(?:điều|article)\s*\.?\s*([0-9]+[a-z]?)')
	CHAPTER_RE = re.compile(r'(?<!\w)(?:chương|chapter)\s*\.?\s*([0-9]+|[ivxlcdm]+)')
	SECTION_RE = re.compile(r'(?<!\w)(?:mục|section)\s*\.?\s*([0-9]+|[ivxlcdm]+)')
	CLAUSE_RE = re.compile(r'(?<!\w)(?:khoản|clause)\s*\.?\s*([0-9]+[a-z]?)')
	CLAUSE_MARKER_RE = re.compile(r'(?:^|[\n\r])\s*([0-9]+)\.\s+')
	WHITESPACE_RE = re.compile(r'\s+')

	ROMAN_VALUES = {
		'i': 1,
		'v': 5,
		'x': 10,
		'l': 50,
		'c': 100,
		'd': 500,
		'm': 1000,
	}

	def normalize_unicode(self, text: Any) -> str:
		return unicodedata.normalize('NFC', self._as_text(text))

	def normalize_whitespace(self, text: Any) -> str:
		return self.WHITESPACE_RE.sub(' ', self.normalize_unicode(text)).strip()

	def normalize_text(self, text: Any, lowercase: bool = True) -> str:
		normalized = self.normalize_whitespace(text)
		if lowercase:
			return normalized.lower()
		return normalized

	def extract_references(self, text: Any, current_article_no: int | None = None) -> dict[str, Any]:
		normalized = self.normalize_text(text)
		article_nos = self._find_numbers(self.ARTICLE_RE, normalized)
		chapter_nos = self._find_numbers(self.CHAPTER_RE, normalized)
		section_nos = self._find_numbers(self.SECTION_RE, normalized)
		clause_nos = self.extract_clause_numbers(text)

		primary_article_no = article_nos[0] if article_nos else None
		if current_article_no is not None:
			ref_article_nos = [number for number in article_nos if number != current_article_no]
		else:
			ref_article_nos = article_nos[1:]

		return {
			'article_no': primary_article_no,
			'chapter_no': chapter_nos[0] if chapter_nos else None,
			'section_no': section_nos[0] if section_nos else None,
			'clause_nos': clause_nos,
			'ref_article_nos': self._unique_ints(ref_article_nos),
		}

	def extract_clause_numbers(self, text: Any) -> list[int]:
		normalized = self.normalize_text(text)
		return self._unique_ints(
			self._find_numbers(self.CLAUSE_RE, normalized)
			+ self._find_numbers(self.CLAUSE_MARKER_RE, self.normalize_unicode(text))
		)

	def enrich_payload(self, metadata: Mapping[str, Any] | None, content: Any = '') -> dict[str, Any]:
		payload = dict(metadata or {})
		content_text = self._as_text(content if content is not None else payload.get('content', ''))

		content_refs = self.extract_references(content_text)
		article_no = self._first_number(self.ARTICLE_RE, payload.get('article', ''))
		chapter_no = self._first_number(self.CHAPTER_RE, payload.get('chapter', ''))
		section_no = self._first_number(self.SECTION_RE, payload.get('section', ''))

		payload['article_no'] = article_no if article_no is not None else content_refs['article_no']
		payload['chapter_no'] = chapter_no if chapter_no is not None else content_refs['chapter_no']
		payload['section_no'] = section_no if section_no is not None else content_refs['section_no']
		payload['clause_nos'] = self.extract_clause_numbers(content_text)
		payload['ref_article_nos'] = self.extract_references(
			content_text,
			current_article_no=payload['article_no'],
		)['ref_article_nos']
		payload['normalized_source_text'] = self.build_normalized_source_text(payload, content_text)

		return payload

	def build_normalized_source_text(self, metadata: Mapping[str, Any] | None, content: Any = '') -> str:
		metadata = metadata or {}
		parts = [
			metadata.get('source', ''),
			metadata.get('chapter', ''),
			metadata.get('section', ''),
			metadata.get('article', ''),
			content,
		]
		return self.normalize_text(' '.join(self._as_text(part) for part in parts if part))

	def _find_numbers(self, pattern: re.Pattern[str], text: Any) -> list[int]:
		numbers = []
		for match in pattern.finditer(self._as_text(text)):
			number = self._parse_legal_number(match.group(1))
			if number is not None:
				numbers.append(number)
		return self._unique_ints(numbers)

	def _first_number(self, pattern: re.Pattern[str], text: Any) -> int | None:
		numbers = self._find_numbers(pattern, self.normalize_text(text))
		return numbers[0] if numbers else None

	def _parse_legal_number(self, value: Any) -> int | None:
		token = self.normalize_text(value)
		digit_match = re.match(r'([0-9]+)', token)
		if digit_match:
			return int(digit_match.group(1))
		if re.fullmatch(r'[ivxlcdm]+', token):
			return self._roman_to_int(token)
		return None

	def _roman_to_int(self, value: str) -> int | None:
		total = 0
		previous = 0
		for char in reversed(value):
			current = self.ROMAN_VALUES.get(char)
			if current is None:
				return None
			if current < previous:
				total -= current
			else:
				total += current
				previous = current
		return total or None

	def _unique_ints(self, values: list[int]) -> list[int]:
		seen = set()
		unique = []
		for value in values:
			if value not in seen:
				seen.add(value)
				unique.append(value)
		return unique

	def _as_text(self, value: Any) -> str:
		if value is None:
			return ''
		return str(value)


LEGAL_METADATA_FIELDS = LegalMetadataProcessor.METADATA_FIELDS
