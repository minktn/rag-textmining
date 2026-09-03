"""
Basic Metrics — Retrieval & Generation
========================================
MetricsCalculator: tính toán tất cả custom metrics.

Retrieval: Hit Rate, MRR, Recall@K, Precision@K, nDCG
Generation: F1, Exact Match, BLEU-1, ROUGE-L
"""

import math
from collections import Counter

# ── NLTK setup (tải tokenizer nếu chưa có) ────────────────────────────
import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ── ROUGE-L ───────────────────────────────────────────────────────────
try:
    from rouge_score import rouge_scorer as _rouge_scorer_module
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

# ── Shared text processing ────────────────────────────────────────────
from .text_processing import normalize_text, tokenize_vi


class MetricsCalculator:
    """Tính toán tất cả custom metrics (retrieval + generation).

    Retrieval metrics: Hit Rate, MRR, Recall@K, Precision@K, nDCG
    Generation metrics: F1, Exact Match, BLEU-1, ROUGE-L
    """

    # ══════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _normalize_law_id(law_id) -> list[dict]:
        """Chuẩn hóa law_id thành danh sách các law identifier dicts.

        Hỗ trợ các định dạng:
          - Dict đơn: {'article_no': 121, 'clause_nos': [1]}
          - List các dict (multi-hop): [{'article_no': 121, ...}, {'article_no': 156, ...}]

        Returns:
            List[dict] — mỗi dict chứa {'article_no': int, 'clause_nos': set[int]}
        """
        items = []
        if not law_id:
            return items

        raw_list = law_id if isinstance(law_id, list) else [law_id]

        for item in raw_list:
            if not isinstance(item, dict):
                continue
            art = item.get('article_no')
            if art is None:
                continue
            clause_nos = set(item.get('clause_nos', []))
            items.append({
                'article_no': art,
                'clause_nos': clause_nos,
            })

        return items

    @staticmethod
    def _get_relevant_articles(law_id) -> set:
        """Trích xuất tập relevant article numbers từ law_id (backward-compatible).

        Dùng cho các metric chỉ cần biết article_no level.
        """
        normalized = MetricsCalculator._normalize_law_id(law_id)
        return {item['article_no'] for item in normalized}

    @staticmethod
    def _is_payload_relevant(payload: dict, gt_law_ids: list[dict]) -> bool:
        """Kiểm tra một payload có khớp với bất kỳ ground truth law_id nào không."""
        p_article = payload.get('article_no')
        p_clauses = set(payload.get('clause_nos', []))

        if p_article is None:
            from src.common.legal_metadata import LegalMetadataProcessor
            extracted_meta = LegalMetadataProcessor().enrich_payload(payload, payload.get('content', ''))
            p_article = extracted_meta.get('article_no')
            if not p_clauses:
                p_clauses = set(extracted_meta.get('clause_nos', []))

        if p_article is None and 'content' in payload:
            extracted = MetricsCalculator.extract_payload_from_text(payload['content'])
            p_article = extracted['article_no']
            if not p_clauses:
                p_clauses = set(extracted['clause_nos'])

        if p_article is None:
            return False

        for gt in gt_law_ids:
            if p_article != gt['article_no']:
                continue
            # Article khớp → kiểm tra clause
            if not gt['clause_nos']:
                # Ground truth không chỉ định clause cụ thể → soft match
                return True
            if p_clauses & gt['clause_nos']:
                # Có overlap clause → exact match
                return True

        return False

    @staticmethod
    def _build_relevance_list(payloads: list[dict], gt_law_ids: list[dict]) -> list[int]:
        """Tạo binary relevance list dựa trên matching article_no + clause_nos."""
        return [
            1 if MetricsCalculator._is_payload_relevant(p, gt_law_ids) else 0
            for p in payloads
        ]

    @staticmethod
    def extract_payload_from_text(text: str) -> dict:
        """Trích xuất thông tin article_no và clause_nos từ văn bản thuần bằng regex.

        Giúp tính toán đầy đủ các metrics retrieval cho GraphRAG hoặc các hệ thống chỉ trả về text context.
        """
        import re
        if not text:
            return {'article_no': None, 'clause_nos': [], 'content': ''}

        art_match = re.search(r'Điều\s+(\d+)', text, re.IGNORECASE)
        article_no = int(art_match.group(1)) if art_match else None

        clause_matches = re.findall(r'Khoản\s+(\d+)', text, re.IGNORECASE)
        clause_nos = [int(c) for c in clause_matches] if clause_matches else []

        return {
            'article_no': article_no,
            'clause_nos': clause_nos,
            'content': text,
        }

    @staticmethod
    def extract_law_ids_from_payloads(payloads: list[dict]) -> list[dict]:
        """Trích xuất law_id chuẩn hóa từ danh sách retrieved payloads.

        Trả về list các dict dạng:
            {'article_no': int, 'chapter_no': int, 'section_no': int|None, 'clause_nos': list[int]}
        """
        law_ids = []
        for p in payloads:
            art = p.get('article_no')
            clauses = p.get('clause_nos', [])
            chap = p.get('chapter_no')
            sec = p.get('section_no')

            if art is None:
                from src.common.legal_metadata import LegalMetadataProcessor
                extracted_meta = LegalMetadataProcessor().enrich_payload(p, p.get('content', ''))
                art = extracted_meta.get('article_no')
                if chap is None:
                    chap = extracted_meta.get('chapter_no')
                if sec is None:
                    sec = extracted_meta.get('section_no')
                if not clauses:
                    clauses = extracted_meta.get('clause_nos', [])

            if art is None and 'content' in p:
                extracted = MetricsCalculator.extract_payload_from_text(p['content'])
                art = extracted['article_no']
                if not clauses:
                    clauses = extracted['clause_nos']

            law_ids.append({
                'article_no': art,
                'chapter_no': chap,
                'section_no': sec,
                'clause_nos': clauses,
            })
        return law_ids

    # ══════════════════════════════════════════════════════════════
    # RETRIEVAL METRICS
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def compute_retrieval_hit(payloads: list[dict], law_id) -> bool:
        """Kiểm tra có ít nhất 1 retrieved context khớp ground truth law_id không."""
        gt_law_ids = MetricsCalculator._normalize_law_id(law_id)
        if not gt_law_ids:
            return False

        for payload in payloads:
            if MetricsCalculator._is_payload_relevant(payload, gt_law_ids):
                return True
        return False

    @staticmethod
    def compute_mrr(payloads: list[dict], law_id) -> float:
        """Mean Reciprocal Rank — 1/rank của context đúng đầu tiên."""
        gt_law_ids = MetricsCalculator._normalize_law_id(law_id)
        if not gt_law_ids:
            return 0.0

        for rank, payload in enumerate(payloads, 1):
            if MetricsCalculator._is_payload_relevant(payload, gt_law_ids):
                return 1.0 / rank
        return 0.0

    @staticmethod
    def compute_recall_at_k(payloads: list[dict], law_id) -> float:
        """Recall@K = |relevant ground truth items matched| / |total ground truth items|.

        Tỷ lệ ground truth law_ids được tìm thấy trong top-K retrieved payloads.
        """
        gt_law_ids = MetricsCalculator._normalize_law_id(law_id)
        if not gt_law_ids:
            return 0.0

        matched_count = 0
        for gt in gt_law_ids:
            for p in payloads:
                if MetricsCalculator._is_payload_relevant(p, [gt]):
                    matched_count += 1
                    break  # Chỉ cần 1 payload khớp cho gt này

        return matched_count / len(gt_law_ids)

    @staticmethod
    def compute_precision_at_k(payloads: list[dict], law_id) -> float:
        """Precision@K = |relevant payloads in top-K| / K.

        Tỷ lệ documents trong top-K là relevant.
        """
        gt_law_ids = MetricsCalculator._normalize_law_id(law_id)
        if not gt_law_ids or not payloads:
            return 0.0

        relevance = MetricsCalculator._build_relevance_list(payloads, gt_law_ids)
        return sum(relevance) / len(relevance)

    @staticmethod
    def compute_ndcg(payloads: list[dict], law_id) -> float:
        """Normalized Discounted Cumulative Gain.

        Đánh giá chất lượng ranking: context relevant ở vị trí cao hơn → score cao hơn.
        DCG = Σ rel_i / log2(i + 1)
        nDCG = DCG / IDCG (ideal DCG nếu xếp tất cả relevant lên đầu)
        """
        gt_law_ids = MetricsCalculator._normalize_law_id(law_id)
        if not gt_law_ids or not payloads:
            return 0.0

        relevance = MetricsCalculator._build_relevance_list(payloads, gt_law_ids)

        # DCG
        dcg = sum(
            rel / math.log2(i + 2)  # i+2 vì i bắt đầu từ 0, log2(1) = 0
            for i, rel in enumerate(relevance)
        )

        # IDCG (ideal: tất cả relevant docs xếp trước)
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = sum(
            rel / math.log2(i + 2)
            for i, rel in enumerate(ideal_relevance)
        )

        if idcg == 0:
            return 0.0

        return dcg / idcg

    # ══════════════════════════════════════════════════════════════
    # GENERATION METRICS
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def compute_f1(prediction: str, reference: str) -> float:
        """Token-level F1 giữa prediction và reference."""
        pred_tokens = tokenize_vi(prediction)
        ref_tokens = tokenize_vi(reference)

        if not pred_tokens or not ref_tokens:
            return 0.0

        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            return 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    @staticmethod
    def compute_exact_match(prediction: str, reference: str) -> bool:
        """Exact match sau normalize."""
        return normalize_text(prediction) == normalize_text(reference)

    @staticmethod
    def compute_bleu1(prediction: str, reference: str) -> float:
        """BLEU-1 score (unigram precision)."""
        pred_tokens = tokenize_vi(prediction)
        ref_tokens = tokenize_vi(reference)

        if not pred_tokens or not ref_tokens:
            return 0.0

        smoother = SmoothingFunction().method1
        return sentence_bleu(
            [ref_tokens], pred_tokens,
            weights=(1.0, 0, 0, 0),
            smoothing_function=smoother,
        )

    @staticmethod
    def compute_rouge_l(prediction: str, reference: str) -> float:
        """ROUGE-L F1 score — đánh giá longest common subsequence.

        ROUGE-L bổ sung cho BLEU vì nó đo recall (BLEU thiên về precision).
        Phù hợp cho đánh giá câu trả lời tiếng Việt dài.

        Fallback: nếu rouge-score chưa cài, tính ROUGE-L thủ công bằng LCS.
        """
        if HAS_ROUGE:
            scorer = _rouge_scorer_module.RougeScorer(['rougeL'], use_stemmer=False)
            scores = scorer.score(reference, prediction)
            return scores['rougeL'].fmeasure

        # ── Fallback: tính LCS-based ROUGE-L thủ công ──────────────
        pred_tokens = tokenize_vi(prediction)
        ref_tokens = tokenize_vi(reference)

        if not pred_tokens or not ref_tokens:
            return 0.0

        # LCS via dynamic programming
        m, n = len(ref_tokens), len(pred_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i - 1] == pred_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs_len = dp[m][n]

        if lcs_len == 0:
            return 0.0

        precision = lcs_len / n
        recall = lcs_len / m
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    # ══════════════════════════════════════════════════════════════
    # PER-QUESTION BUNDLES
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def compute_retrieval_metrics(payloads: list[dict], law_id) -> dict:
        """Tính tất cả retrieval metrics cho 1 câu hỏi."""
        return {
            'retrieval_hit': MetricsCalculator.compute_retrieval_hit(payloads, law_id),
            'mrr': round(MetricsCalculator.compute_mrr(payloads, law_id), 4),
            'recall_at_k': round(MetricsCalculator.compute_recall_at_k(payloads, law_id), 4),
            'precision_at_k': round(MetricsCalculator.compute_precision_at_k(payloads, law_id), 4),
            'ndcg': round(MetricsCalculator.compute_ndcg(payloads, law_id), 4),
        }

    @staticmethod
    def compute_generation_metrics(prediction: str, reference: str) -> dict:
        """Tính tất cả generation metrics cho 1 câu hỏi."""
        return {
            'exact_match': MetricsCalculator.compute_exact_match(prediction, reference),
            'f1_score': round(MetricsCalculator.compute_f1(prediction, reference), 4),
            'bleu_1': round(MetricsCalculator.compute_bleu1(prediction, reference), 4),
            'rouge_l': round(MetricsCalculator.compute_rouge_l(prediction, reference), 4),
        }

    # ══════════════════════════════════════════════════════════════
    # AGGREGATE
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def aggregate(results: list[dict], top_k: int = 5, is_graph_mode: bool = False) -> dict:
        """Tổng hợp metrics cho toàn bộ dataset.

        Parameters
        ----------
        results : list[dict]
            Danh sách kết quả evaluation từng câu hỏi.
        top_k : int
            Số lượng tài liệu retrieve.
        is_graph_mode : bool
            Nếu True (Graph database/GraphRAG), bỏ qua các chỉ số retrieval chunk-level
            (Recall@K, Precision@K, MRR, nDCG) vì dữ liệu đồ thị dạng Entity/Community,
            chỉ tập trung vào Generation metrics và RAGAS metrics.
        """
        total = len(results)
        if total == 0:
            return {}

        # Tự động nhận diện graph mode nếu các item có cờ is_graph
        if not is_graph_mode and any(r.get('is_graph') for r in results):
            is_graph_mode = True

        # Accumulators
        hit_count = 0
        mrr_sum = 0.0
        recall_sum = 0.0
        precision_sum = 0.0
        ndcg_sum = 0.0

        em_count = 0
        f1_sum = 0.0
        bleu1_sum = 0.0
        rouge_l_sum = 0.0

        retrieval_lat_sum = 0.0
        generation_lat_sum = 0.0
        e2e_lat_sum = 0.0

        for r in results:
            # ── Retrieval metrics (Chỉ tính khi KHÔNG PHẢI Graph Mode) ─
            if not is_graph_mode:
                ret_metrics = MetricsCalculator.compute_retrieval_metrics(
                    r.get('retrieved_payloads', []), r.get('law_id', {})
                )
                r.update(ret_metrics)

                hit_count += int(ret_metrics['retrieval_hit'])
                mrr_sum += ret_metrics['mrr']
                recall_sum += ret_metrics['recall_at_k']
                precision_sum += ret_metrics['precision_at_k']
                ndcg_sum += ret_metrics['ndcg']
            else:
                # Đánh dấu N/A cho retrieval chunks trong Graph Mode
                r['retrieval_hit'] = None
                r['mrr'] = None
                r['recall_at_k'] = None
                r['precision_at_k'] = None
                r['ndcg'] = None

            # ── Generation metrics ─────────────────────────────
            gen_metrics = MetricsCalculator.compute_generation_metrics(
                r.get('generated_answer', ''), r.get('ground_truth', '')
            )
            r.update(gen_metrics)

            em_count += int(gen_metrics['exact_match'])
            f1_sum += gen_metrics['f1_score']
            bleu1_sum += gen_metrics['bleu_1']
            rouge_l_sum += gen_metrics['rouge_l']

            # ── Latency ────────────────────────────────────────
            retrieval_lat_sum += r.get('retrieval_latency_ms', 0)
            generation_lat_sum += r.get('generation_latency_ms', 0)
            e2e_lat_sum += r.get('e2e_latency_ms', 0)

        aggregate = {
            'total_questions': total,
            'top_k': top_k,
            'is_graph_mode': is_graph_mode,
            # Retrieval (None nếu là Graph mode)
            'retrieval_hit_rate': round(hit_count / total, 4) if not is_graph_mode else None,
            'mrr': round(mrr_sum / total, 4) if not is_graph_mode else None,
            'avg_recall_at_k': round(recall_sum / total, 4) if not is_graph_mode else None,
            'avg_precision_at_k': round(precision_sum / total, 4) if not is_graph_mode else None,
            'avg_ndcg': round(ndcg_sum / total, 4) if not is_graph_mode else None,
            'retrieval_notice': (
                "Skipped for Graph Database (Entity/Community context). Only RAGAS LLM-as-a-judge applies."
                if is_graph_mode else None
            ),
            # Generation
            'exact_match_rate': round(em_count / total, 4),
            'avg_f1': round(f1_sum / total, 4),
            'avg_bleu_1': round(bleu1_sum / total, 4),
            'avg_rouge_l': round(rouge_l_sum / total, 4),
            # Latency
            'avg_retrieval_latency_ms': round(retrieval_lat_sum / total, 2),
            'avg_generation_latency_ms': round(generation_lat_sum / total, 2),
            'avg_e2e_latency_ms': round(e2e_lat_sum / total, 2),
        }

        # Per question-type breakdown
        type_groups: dict[str, list[dict]] = {}
        for r in results:
            qt = r.get('question_type', 'unknown')
            type_groups.setdefault(qt, []).append(r)

        per_type = {}
        for qt, items in type_groups.items():
            n = len(items)
            per_type[qt] = {
                'count': n,
                'hit_rate': round(sum(1 for i in items if i.get('retrieval_hit')) / n, 4) if not is_graph_mode else None,
                'avg_recall_at_k': round(sum(i.get('recall_at_k', 0) for i in items) / n, 4) if not is_graph_mode else None,
                'avg_f1': round(sum(i.get('f1_score', 0) for i in items) / n, 4),
                'avg_bleu_1': round(sum(i.get('bleu_1', 0) for i in items) / n, 4),
                'avg_rouge_l': round(sum(i.get('rouge_l', 0) for i in items) / n, 4),
            }
        aggregate['per_question_type'] = per_type

        return aggregate
