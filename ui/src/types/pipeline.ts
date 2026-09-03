export interface PipelineConfig {
  database: 'base' | 'contriever' | 'graph';
  advanced: string | null; // e.g. 'rag_fusion'
  preprocessing: string[]; // e.g. ['hyde']
  postprocessing: string[]; // e.g. ['filter_rerank', 'crag', 'prompt_compression']
  llm_service?: string;
  sub_llm_service?: string;
  top_k?: number;
}

export interface EvalOptionItem {
  id: string;
  name: string;
  desc: string;
}

export interface EvalInfo {
  dataset_name: string;
  law: string;
  total_questions: number;
  default_seed: number;
  available_options: {
    database: EvalOptionItem[];
    advanced: EvalOptionItem[];
    preprocessing: EvalOptionItem[];
    postprocessing: EvalOptionItem[];
    llm_services: string[];
  };
}

export interface EvalLawId {
  article_no: number | null;
  chapter_no: number | null;
  section_no: number | null;
  clause_nos: number[];
}

export interface EvalQuestionResult {
  id: string;
  question: string;
  question_type: string;
  ground_truth: string;
  generated_answer: string;
  retrieved_contexts: string[];
  retrieved_law_ids: EvalLawId[];
  law_id?: Record<string, any>;
  is_graph?: boolean;
  retrieval_latency_ms: number;
  generation_latency_ms: number;
  e2e_latency_ms: number;
}

export interface EvalReport {
  timestamp?: string;
  metadata?: {
    timestamp?: string;
    configuration?: {
      retriever_mode?: string;
      advanced_method?: string | null;
      preprocessing?: string[];
      postprocessing?: string[];
      llm_service?: string;
      sub_llm_service?: string;
      ragas_service?: string;
      top_k?: number;
    };
    eval_dataset?: {
      file?: string;
      law?: string;
      version?: string;
      total_questions?: number;
    };
  };
  summary_metrics?: {
    is_graph_mode?: boolean;
    retrieval?: {
      hit_rate?: number;
      mrr?: number;
      recall_at_k?: number;
      precision_at_k?: number;
      ndcg?: number;
      latency_ms?: number;
    };
    generation?: {
      exact_match?: number;
      f1?: number;
      bleu_1?: number;
      rouge_l?: number;
      latency_ms?: number;
      e2e_latency_ms?: number;
    };
    ragas?: {
      faithfulness?: number;
      answer_relevancy?: number;
      context_precision?: number;
      context_recall?: number;
    };
    per_question_type?: Record<
      string,
      {
        count: number;
        hit_rate: number;
        avg_recall_at_k: number;
        avg_f1: number;
        avg_bleu_1: number;
        avg_rouge_l: number;
      }
    >;
  };
  detailed_results?: EvalQuestionResult[];
}
