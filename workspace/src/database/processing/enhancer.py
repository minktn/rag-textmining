import uuid
from typing import List, Dict, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
from langchain_core.language_models.llms import LLM
from rag.utils.logger import RAGLogger
from rag.utils.progress import ProgressTracker


class LLMIndexingEnhancer:
    def __init__(self, llm: LLM, logger: RAGLogger):
        self.llm = llm
        self.log = logger

    def _call_llm(self, prompt: str) -> str:
        try:
            result = self.llm.invoke(prompt)
            return result.strip() if result else ""
        except Exception as e:
            self.log.warning(f"LLM call failed: {e}")
            return ""

    def generate_hypothetical_questions(self, chunk_text: str, n: int = 3) -> List[str]:
        prompt = (
            f"Given the following text, generate {n} specific questions that this text directly answers.\n"
            f"Output only the questions, one per line, no numbering, no extra text.\n\n"
            f"Text:\n{chunk_text}\n\nQuestions:"
        )
        result = self._call_llm(prompt)
        questions = [q.strip() for q in result.split("\n") if q.strip() and "?" in q]
        return questions[:n]

    def generate_summary(self, chunk_text: str) -> str:
        prompt = (
            f"Summarize the following text in 1-2 concise sentences capturing the key information.\n"
            f"Output only the summary, no preamble.\n\n"
            f"Text:\n{chunk_text}\n\nSummary:"
        )
        return self._call_llm(prompt)

    def extract_entities(self, chunk_text: str) -> List[str]:
        prompt = (
            f"Extract the most important technical terms, entities, and keywords from the text.\n"
            f"Output as a comma-separated list only, no explanation.\n\n"
            f"Text:\n{chunk_text}\n\nKeywords:"
        )
        result = self._call_llm(prompt)
        return [e.strip() for e in result.split(",") if e.strip()][:10]

    def enhance_chunks(
        self,
        chunks: List[Document],
        progress: ProgressTracker = None,
        max_workers: int = 1,
    ) -> Tuple[List[Document], List[Tuple[str, Document]]]:
        self.log.step(f"LLM-enhancing {len(chunks)} chunks at index time")
        self.log.info("Techniques: HyQGen + Summary Embedding + Entity Extraction")

        if max_workers > 1:
            self.log.info(f"Parallel enhancement enabled: max_workers={max_workers}")
            return self._enhance_chunks_parallel(chunks, progress, max_workers)
        else:
            return self._enhance_chunks_sequential(chunks, progress)

    def _enhance_single_chunk(self, i: int, chunk: Document) -> Dict[str, Any]:
        text = chunk.page_content
        questions = self.generate_hypothetical_questions(text, n=3)
        summary = self.generate_summary(text)
        entities = self.extract_entities(text)
        return {
            "index": i,
            "questions": questions,
            "summary": summary,
            "entities": entities,
        }

    def _enhance_chunks_parallel(
        self,
        chunks: List[Document],
        progress: ProgressTracker = None,
        max_workers: int = 4,
    ) -> Tuple[List[Document], List[Tuple[str, Document]]]:
        enhanced_parent_chunks = []
        auxiliary_docs_with_parent: List[Tuple[str, Document]] = []

        results: List[Any] = [None] * len(chunks)
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._enhance_single_chunk, i, chunk): i
                for i, chunk in enumerate(chunks)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results[idx] = result
                except Exception as e:
                    self.log.warning(f"  Chunk {idx} enhancement failed: {e}")
                    results[idx] = {"index": idx, "questions": [], "summary": "", "entities": []}

                completed += 1
                if progress:
                    progress.update(completed, len(chunks), f"LLM enhancing chunk {completed}/{len(chunks)}")
                if completed % 10 == 0:
                    self.log.info(f"  [{completed}/{len(chunks)}] chunks enhanced so far")

        for i, chunk in enumerate(chunks):
            r = results[i]
            chunk.metadata["llm_summary"] = r["summary"]
            chunk.metadata["llm_entities"] = ", ".join(r["entities"])
            chunk.metadata["llm_questions"] = " | ".join(r["questions"])
            enhanced_parent_chunks.append(chunk)

            for q in r["questions"]:
                if q:
                    aux_doc = Document(
                        page_content=q,
                        metadata={
                            **chunk.metadata,
                            "doc_type": "hypothetical_question",
                            "parent_text_snippet": chunk.page_content[:200],
                        },
                    )
                    auxiliary_docs_with_parent.append((chunk.page_content, aux_doc))

            if r["summary"]:
                aux_doc = Document(
                    page_content=r["summary"],
                    metadata={
                        **chunk.metadata,
                        "doc_type": "llm_summary",
                    },
                )
                auxiliary_docs_with_parent.append((chunk.page_content, aux_doc))

        return enhanced_parent_chunks, auxiliary_docs_with_parent

    def _enhance_chunks_sequential(
        self,
        chunks: List[Document],
        progress: ProgressTracker = None,
    ) -> Tuple[List[Document], List[Tuple[str, Document]]]:
        enhanced_parent_chunks = []
        auxiliary_docs_with_parent: List[Tuple[str, Document]] = []

        for i, chunk in enumerate(chunks):
            if progress:
                progress.update(i, len(chunks), f"LLM enhancing chunk {i+1}/{len(chunks)}")

            text = chunk.page_content
            questions = self.generate_hypothetical_questions(text, n=3)
            summary = self.generate_summary(text)
            entities = self.extract_entities(text)

            chunk.metadata["llm_summary"] = summary
            chunk.metadata["llm_entities"] = ", ".join(entities)
            chunk.metadata["llm_questions"] = " | ".join(questions)
            enhanced_parent_chunks.append(chunk)

            for q in questions:
                if q:
                    aux_doc = Document(
                        page_content=q,
                        metadata={
                            **chunk.metadata,
                            "doc_type": "hypothetical_question",
                            "parent_text_snippet": text[:200],
                        },
                    )
                    auxiliary_docs_with_parent.append((chunk.page_content, aux_doc))

            if summary:
                aux_doc = Document(
                    page_content=summary,
                    metadata={
                        **chunk.metadata,
                        "doc_type": "llm_summary",
                    },
                )
                auxiliary_docs_with_parent.append((chunk.page_content, aux_doc))

        return enhanced_parent_chunks, auxiliary_docs_with_parent
