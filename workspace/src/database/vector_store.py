import json
import logging
from pathlib import Path
from tqdm import tqdm
from qdrant_client.models import PointStruct, models

# Setup logging configuration for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create file handler targeting the database folder
log_file = Path(__file__).resolve().parent / "upload_data.log"
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
if not logger.handlers:
    logger.addHandler(file_handler)

class VectorStore:
    def __init__(self, db_manager, dense_embedder, sparse_embedder, metadata_processor, chunker=None):
        self.db_manager = db_manager
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        self.metadata_processor = metadata_processor
        self.chunker = chunker

    def _enrich_content(self, chunk):
        metadata = chunk.get('metadata', {})
        content = chunk.get('content', '')

        if not metadata:
            return content

        metadata_str = ''
        for key, value in metadata.items():
            metadata_str += f"[{key}: {value}] "
        
        enriched_content = f"{metadata_str.strip()}\n{content}"

        return enriched_content

    def chunk_and_save(self, input_file: Path, output_file: Path):
        """Chunks an input markdown file and saves to JSON."""
        if not self.chunker:
            logger.error("Chunker is not initialized.")
            raise ValueError("Chunker is not initialized.")
            
        logger.info(f"Starting chunking for {input_file}")
        with open(input_file, "r", encoding="utf-8") as file:
            content = file.read()

        chunks = self.chunker.chunking(content)
        self.chunker.save_to_json(chunks, output_file)
        
        msg = f"Saved {len(chunks)} chunks to {output_file}"
        logger.info(msg)
        print(msg)
        return chunks

    def ingest_documents(self, collection_name: str, chunks_json_path: Path):
        """Reads chunked JSON, generates embeddings, and upserts to Qdrant."""
        logger.info(f"Starting ingestion process for collection '{collection_name}' from {chunks_json_path}")
        
        with open(chunks_json_path, "r", encoding="utf-8") as file:
            chunks = json.load(file)
            
        logger.info(f"Loaded {len(chunks)} chunks. Generating embeddings...")
        print(f"Loaded {len(chunks)} chunks. Generating embeddings...")
        
        # Enrich content
        enriched_chunks = [self._enrich_content(chunk) for chunk in chunks]
        
        # Embed
        dense_embeddings = self.dense_embedder.embed_batch(enriched_chunks)
        sparse_embeddings = self.sparse_embedder.embed_batch(enriched_chunks)
        
        logger.info("Embeddings generated successfully. Setting up collection...")
        vector_size = self.dense_embedder.model.get_sentence_embedding_dimension()
        self.db_manager.setup_collection(collection_name, vector_size=vector_size, recreate=True)
        
        points = []
        logger.info("Constructing payload points...")
        
        for chunk, dense_embedding, sparse_embedding in tqdm(
            zip(chunks, dense_embeddings, sparse_embeddings), 
            total=len(chunks), 
            desc="Constructing and preparing points for upload"
        ):
            content = chunk.get('content', '')
            payload = self.metadata_processor.enrich_payload(chunk.get('metadata', {}), content)
            payload['content'] = content
            
            points.append(
                PointStruct(
                    id=chunk['id'],
                    vector={
                        'dense': dense_embedding,
                        'sparse': models.SparseVector(
                            indices=sparse_embedding['indices'],
                            values=sparse_embedding['values']
                        )
                    },
                    payload=payload
                )
            )
            
        logger.info(f"Upserting {len(points)} points to collection '{collection_name}'...")
        print(f"Upserting {len(points)} points to collection '{collection_name}'...")
        
        self.db_manager.upsert_points(collection_name, points)
        
        logger.info("Ingestion completed successfully.")
        print("Ingestion completed successfully.")

    def ingest_documents_late(self, collection_name: str, chunks_json_path: Path, max_tokens: int = 8192):
        """Reads chunked JSON, uses LATE CHUNKING to generate embeddings, and upserts to Qdrant."""
        from transformers import AutoTokenizer, AutoModel
        import torch
        from src.configs import settings
        from src.database.chunker.late_chunking import chunked_pooling

        logger.info(f"Starting LATE CHUNKING ingestion process for collection '{collection_name}' from {chunks_json_path}")
        
        with open(chunks_json_path, "r", encoding="utf-8") as file:
            chunks = json.load(file)

        logger.info(f"Loaded {len(chunks)} chunks. Generating late chunking embeddings...")
        print(f"Loaded {len(chunks)} chunks. Generating late chunking embeddings...")

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(settings.LONG_DENSE_EMBEDDING_MODEL)
        model = AutoModel.from_pretrained(settings.LONG_DENSE_EMBEDDING_MODEL)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # Enrich content
        for chunk in chunks:
            chunk['enriched_content'] = self._enrich_content(chunk)

        points = []
        
        # We group chunks into batches up to max_tokens
        current_batch_chunks = []
        current_batch_token_count = 0
        batches = []

        for chunk in chunks:
            text = chunk['enriched_content']
            tokens = tokenizer.tokenize(text)
            token_len = len(tokens) + 2 # +2 for SEP, CLS

            if current_batch_token_count + token_len > max_tokens - 100 and current_batch_chunks:
                batches.append(current_batch_chunks)
                current_batch_chunks = [chunk]
                current_batch_token_count = token_len
            else:
                current_batch_chunks.append(chunk)
                current_batch_token_count += token_len
        
        if current_batch_chunks:
            batches.append(current_batch_chunks)

        logger.info(f"Created {len(batches)} batches for late chunking.")
        print(f"Created {len(batches)} batches for late chunking.")

        # Dynamically get the hidden size for the current model (e.g. 1024 for bge-m3)
        vector_size = model.config.hidden_size

        self.db_manager.setup_collection(collection_name, vector_size=vector_size, recreate=True)

        # Process each batch
        for batch_idx, batch in enumerate(tqdm(batches, desc="Processing Late Chunking Batches")):
            combined_text = ""
            chunk_char_spans = []
            
            for chunk in batch:
                start_char = len(combined_text)
                text = chunk['enriched_content']
                combined_text += text
                end_char = len(combined_text)
                chunk_char_spans.append((start_char, end_char))
                combined_text += "\n" # separator

            inputs = tokenizer(
                combined_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_tokens,
                add_special_tokens=True
            )
            
            inputs_gpu = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs_gpu)
                token_embeddings = outputs.last_hidden_state
            
            span_annotations = []
            for (char_start, char_end) in chunk_char_spans:
                # Dùng char_to_token trực tiếp từ BatchEncoding của HuggingFace (chuẩn xác 100% với tiếng Việt & XLM-RoBERTa)
                token_start = inputs.char_to_token(0, char_start)
                # Lấy token của char ngay trước char_end
                token_end_inclusive = inputs.char_to_token(0, max(char_start, char_end - 1))
                
                if token_start is None:
                    # Fallback tìm token gần nhất phía sau
                    for c in range(char_start, char_end):
                        t = inputs.char_to_token(0, c)
                        if t is not None:
                            token_start = t
                            break
                if token_start is None:
                    token_start = 1

                if token_end_inclusive is None:
                    # Fallback tìm token gần nhất phía trước
                    for c in range(char_end - 1, char_start - 1, -1):
                        t = inputs.char_to_token(0, c)
                        if t is not None:
                            token_end_inclusive = t
                            break
                if token_end_inclusive is None:
                    token_end = token_start + 1
                else:
                    token_end = token_end_inclusive + 1
                
                span_annotations.append((token_start, max(token_start + 1, token_end)))
            
            # chunked_pooling signature: (model_output, span_annotation: list, max_length=None)
            pooled_embeddings = chunked_pooling(
                [token_embeddings], 
                [span_annotations], 
                max_length=max_tokens
            )[0]

            # Audit Stage: Log chi tiết token spans để kiểm tra
            if batch_idx == 0:
                logger.info("=== STAGE AUDIT: LATE CHUNKING SPANS (BATCH 0) ===")
                for i in range(min(3, len(batch))):
                    s_tok, e_tok = span_annotations[i]
                    c_text = batch[i]['enriched_content'][:50].replace('\n', ' ')
                    logger.info(f"Chunk [{batch[i]['id'][:8]}] CharSpan: {chunk_char_spans[i]} -> TokenSpan: ({s_tok}, {e_tok}) | Length: {e_tok - s_tok} tokens | Text Snippet: '{c_text}...'")

            enriched_texts = [c['enriched_content'] for c in batch]
            sparse_embeddings = self.sparse_embedder.embed_batch(enriched_texts)

            for idx, chunk in enumerate(batch):
                dense_embedding = pooled_embeddings[idx]
                sparse_embedding = sparse_embeddings[idx]
                
                content = chunk.get('content', '')
                payload = self.metadata_processor.enrich_payload(chunk.get('metadata', {}), content)
                payload['content'] = content
                
                points.append(
                    PointStruct(
                        id=chunk['id'],
                        vector={
                            'dense': dense_embedding.tolist(),
                            'sparse': models.SparseVector(
                                indices=sparse_embedding['indices'],
                                values=sparse_embedding['values']
                            )
                        },
                        payload=payload
                    )
                )

        logger.info(f"Upserting {len(points)} points to collection '{collection_name}'...")
        print(f"Upserting {len(points)} points to collection '{collection_name}'...")
        
        self.db_manager.upsert_points(collection_name, points)
        
        logger.info("Late chunking ingestion completed successfully.")
        print("Late chunking ingestion completed successfully.")
