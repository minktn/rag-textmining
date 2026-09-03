import { RAGApiResponse } from '../types/chat';
import { EvalInfo, EvalReport, PipelineConfig } from '../types/pipeline';

const API_BASE_URL = 'http://localhost:8002';

export async function checkServerHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, { method: 'GET' });
    return res.ok;
  } catch (error) {
    return false;
  }
}

export async function sendRAGQuery(
  query: string,
  rag: boolean,
  top_k?: number,
  sessionId?: string,
  config?: PipelineConfig
): Promise<RAGApiResponse> {
  const payload: Record<string, any> = {
    query,
    rag,
    session_id: sessionId,
  };
  if (typeof top_k === 'number') {
    payload.top_k = top_k;
  }
  if (config) {
    payload.database = config.database;
    payload.advanced = config.advanced;
    payload.preprocessing = config.preprocessing;
    payload.postprocessing = config.postprocessing;
    if (config.llm_service) payload.llm_service = config.llm_service;
    if (config.sub_llm_service) payload.sub_llm_service = config.sub_llm_service;
  }

  const response = await fetch(`${API_BASE_URL}/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Lỗi từ máy chủ (${response.status})`);
  }

  return response.json();
}

export async function getEvalInfo(): Promise<EvalInfo> {
  const response = await fetch(`${API_BASE_URL}/eval/info`, {
    method: 'GET',
  });
  if (!response.ok) {
    throw new Error(`Không thể lấy thông tin tập dữ liệu (${response.status})`);
  }
  return response.json();
}

export async function runEvaluation(params: {
  limit: number;
  random_sample: boolean;
  random_seed?: number;
  retriever_mode: string;
  advanced?: string | null;
  preprocessing?: string[];
  postprocessing?: string[];
  llm_service?: string;
  sub_llm_service?: string;
  ragas_service?: string;
  skip_ragas?: boolean;
  top_k?: number;
}): Promise<EvalReport> {
  const response = await fetch(`${API_BASE_URL}/eval/run`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Lỗi khi chạy đánh giá (${response.status})`);
  }

  return response.json();
}

export async function getLatestEval(): Promise<EvalReport> {
  const response = await fetch(`${API_BASE_URL}/eval/latest`, {
    method: 'GET',
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Chưa có báo cáo gần đây (${response.status})`);
  }
  return response.json();
}