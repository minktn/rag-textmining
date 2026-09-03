import React from 'react';
import { 
  X, 
  Settings2, 
  Database, 
  Sparkles, 
  Cpu, 
  Layers, 
  Info, 
  RotateCcw,
  CheckCircle2
} from 'lucide-react';
import { PipelineConfig } from '../types/pipeline';

interface PipelineSettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: PipelineConfig;
  onChangeConfig: (newConfig: PipelineConfig) => void;
}

export const DEFAULT_PIPELINE_CONFIG: PipelineConfig = {
  database: 'base',
  advanced: null,
  preprocessing: [],
  postprocessing: [],
  llm_service: 'nvidia',
  sub_llm_service: 'nvidia',
  top_k: 5,
};

export const PipelineSettingsModal: React.FC<PipelineSettingsModalProps> = ({
  isOpen,
  onClose,
  config,
  onChangeConfig,
}) => {
  if (!isOpen) return null;

  const isAdvancedSelected = Boolean(config.advanced);

  const handleDatabaseChange = (db: 'base' | 'contriever' | 'graph') => {
    onChangeConfig({ ...config, database: db });
  };

  const handleToggleAdvanced = (adv: string) => {
    if (config.advanced === adv) {
      onChangeConfig({ ...config, advanced: null });
    } else {
      onChangeConfig({ ...config, advanced: adv });
    }
  };

  const handleTogglePreprocessing = (method: string) => {
    const exists = config.preprocessing.includes(method);
    const updated = exists
      ? config.preprocessing.filter((m) => m !== method)
      : [...config.preprocessing, method];
    onChangeConfig({ ...config, preprocessing: updated });
  };

  const handleTogglePostprocessing = (method: string) => {
    const isAdaptable = method === 'prompt_compression';
    if (config.postprocessing.includes(method)) {
      onChangeConfig({
        ...config,
        postprocessing: config.postprocessing.filter((m) => m !== method),
      });
    } else {
      if (isAdaptable) {
        onChangeConfig({
          ...config,
          postprocessing: [...config.postprocessing, method],
        });
      } else {
        // Tối đa 1 phương thức non-adaptable
        const onlyAdaptable = config.postprocessing.filter((m) => m === 'prompt_compression');
        onChangeConfig({
          ...config,
          postprocessing: [...onlyAdaptable, method],
        });
      }
    }
  };

  const handleResetBaseline = () => {
    onChangeConfig(DEFAULT_PIPELINE_CONFIG);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-fadeIn">
      <div 
        className="w-full max-w-2xl bg-sidebarBg border border-borderDark/90 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-borderDark/80 flex items-center justify-between bg-cardBg/50">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-accentGreen/15 text-accentGreen border border-accentGreen/25">
              <Settings2 className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-gray-100 flex items-center gap-2">
                Cấu hình Pipeline RAG
                <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-950/80 text-emerald-400 border border-emerald-800/50">
                  {config.database === 'graph' ? 'Graph RAG' : config.advanced ? 'Advanced RAG' : 'Standard Baseline'}
                </span>
              </h2>
              <p className="text-xs text-gray-400">
                Tùy chỉnh các thành phần can thiệp cho các lần hỏi đáp (Inference) tiếp theo
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-hoverBg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-6 text-sm text-gray-300">
          {/* Section 1: Database Selector */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-xs font-semibold text-gray-200 uppercase tracking-wider">
              <Database className="w-4 h-4 text-emerald-400" />
              <span>1. Cơ sở dữ liệu tri thức (Database)</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {[
                { id: 'base', name: 'Vector DB (Base)', desc: 'BGE-M3 Dense + CrossEncoder Reranker', icon: '⚡' },
                { id: 'contriever', name: 'Contriever', desc: 'Dense search với mContriever', icon: '🔍' },
                { id: 'graph', name: 'Graph Database', desc: 'Microsoft GraphRAG trên Neo4j / Local KG', icon: '🕸️' },
              ].map((item) => {
                const isSelected = config.database === item.id;
                return (
                  <label
                    key={item.id}
                    onClick={() => handleDatabaseChange(item.id as any)}
                    className={`relative p-3.5 rounded-xl border cursor-pointer transition-all flex flex-col gap-1.5 ${
                      isSelected
                        ? 'border-accentGreen bg-accentGreen/10 text-white shadow-md'
                        : 'border-borderDark/70 bg-cardBg/40 hover:bg-cardBg hover:border-gray-600 text-gray-300'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-xs flex items-center gap-1.5">
                        <span>{item.icon}</span>
                        {item.name}
                      </span>
                      <input
                        type="radio"
                        name="db_select"
                        checked={isSelected}
                        onChange={() => {}}
                        className="text-accentGreen focus:ring-accentGreen h-3.5 w-3.5"
                      />
                    </div>
                    <span className="text-[11px] text-gray-400 leading-snug">{item.desc}</span>
                  </label>
                );
              })}
            </div>
          </div>

          {/* Section 2: Advanced Retrieval */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-xs font-semibold text-gray-200 uppercase tracking-wider">
                <Sparkles className="w-4 h-4 text-amber-400" />
                <span>2. Phương thức truy xuất nâng cao (Advanced Method)</span>
              </div>
              <span className="text-[11px] text-gray-500">Ưu tiên cao nhất — thay thế Pre/Post</span>
            </div>

            <div className="space-y-2">
              {[
                {
                  id: 'rag_fusion',
                  name: 'RAG-Fusion',
                  desc: 'Mở rộng câu hỏi thành nhiều biến thể qua Sub-LLM, truy vấn song song và gộp kết quả bằng Reciprocal Rank Fusion (RRF).',
                },
              ].map((item) => {
                const checked = config.advanced === item.id;
                return (
                  <label
                    key={item.id}
                    onClick={() => handleToggleAdvanced(item.id)}
                    className={`flex items-start gap-3 p-3.5 rounded-xl border cursor-pointer transition-all ${
                      checked
                        ? 'border-amber-500/80 bg-amber-500/10 text-white shadow-sm'
                        : 'border-borderDark/70 bg-cardBg/40 hover:bg-cardBg hover:border-gray-600 text-gray-300'
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => {}}
                      className="mt-0.5 rounded text-amber-500 focus:ring-amber-500 h-4 w-4 bg-inputBg border-gray-600"
                    />
                    <div className="flex flex-col gap-1">
                      <span className="font-semibold text-xs text-gray-100 flex items-center gap-1.5">
                        {item.name}
                        {checked && <CheckCircle2 className="w-3.5 h-3.5 text-amber-400" />}
                      </span>
                      <span className="text-[11px] text-gray-400 leading-relaxed">{item.desc}</span>
                    </div>
                  </label>
                );
              })}
            </div>

            {isAdvancedSelected && (
              <div className="flex items-center gap-2 p-2.5 rounded-lg bg-amber-950/40 border border-amber-800/50 text-amber-300 text-xs">
                <Info className="w-4 h-4 flex-shrink-0" />
                <span>
                  Đang chọn <strong>Advanced</strong>. Preprocessing và Postprocessing sẽ được tạm thời bỏ qua theo kiến trúc Strategy.
                </span>
              </div>
            )}
          </div>

          {/* Section 3: Preprocessing & Postprocessing */}
          <div className={`space-y-4 transition-opacity ${isAdvancedSelected ? 'opacity-40 pointer-events-none' : ''}`}>
            {/* Preprocessing */}
            <div className="space-y-2.5">
              <div className="flex items-center gap-2 text-xs font-semibold text-gray-200 uppercase tracking-wider">
                <Layers className="w-4 h-4 text-cyan-400" />
                <span>3. Tiền xử lý truy vấn (Preprocessing)</span>
              </div>
              <div className="space-y-2">
                {[
                  {
                    id: 'hyde',
                    name: 'HyDE (Hypothetical Document Embeddings)',
                    desc: 'Sinh văn bản câu trả lời giả định trước khi tính toán embedding, tăng cường độ tương đồng ngữ nghĩa.',
                  },
                ].map((item) => {
                  const checked = config.preprocessing.includes(item.id);
                  return (
                    <label
                      key={item.id}
                      onClick={() => !isAdvancedSelected && handleTogglePreprocessing(item.id)}
                      className={`flex items-start gap-3 p-3 rounded-xl border cursor-pointer transition-all ${
                        checked
                          ? 'border-cyan-500/80 bg-cyan-500/10 text-white'
                          : 'border-borderDark/70 bg-cardBg/40 hover:bg-cardBg hover:border-gray-600 text-gray-300'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        disabled={isAdvancedSelected}
                        onChange={() => {}}
                        className="mt-0.5 rounded text-cyan-500 focus:ring-cyan-500 h-4 w-4 bg-inputBg border-gray-600"
                      />
                      <div className="flex flex-col gap-0.5">
                        <span className="font-semibold text-xs text-gray-100">{item.name}</span>
                        <span className="text-[11px] text-gray-400">{item.desc}</span>
                      </div>
                    </label>
                  );
                })}
              </div>
            </div>

            {/* Postprocessing */}
            <div className="space-y-2.5">
              <div className="flex items-center gap-2 text-xs font-semibold text-gray-200 uppercase tracking-wider">
                <Cpu className="w-4 h-4 text-purple-400" />
                <span>4. Hậu xử lý kết quả (Postprocessing)</span>
              </div>
              <div className="space-y-2">
                {[
                  {
                    id: 'filter_rerank',
                    name: 'Filter-then-Rerank',
                    desc: 'Mô hình 2 tầng: SLM Local (1.5B) lọc sơ bộ -> NVIDIA LLM Reranker phân loại chuyên sâu cho các mẫu khó.',
                  },
                  {
                    id: 'crag',
                    name: 'Corrective RAG (CRAG)',
                    desc: 'Đánh giá độ tin cậy của tài liệu bằng BamiBERT NLI; hiệu chỉnh và tìm kiếm bổ sung nếu context không đạt.',
                  },
                  {
                    id: 'prompt_compression',
                    name: 'Prompt Compression (LongLLMLingua)',
                    desc: 'Nén các đoạn văn bản dài để loại bỏ từ dư thừa, tiết kiệm token và tăng tốc sinh câu trả lời.',
                  },
                ].map((item) => {
                  const checked = config.postprocessing.includes(item.id);
                  return (
                    <label
                      key={item.id}
                      onClick={() => !isAdvancedSelected && handleTogglePostprocessing(item.id)}
                      className={`flex items-start gap-3 p-3 rounded-xl border cursor-pointer transition-all ${
                        checked
                          ? 'border-purple-500/80 bg-purple-500/10 text-white'
                          : 'border-borderDark/70 bg-cardBg/40 hover:bg-cardBg hover:border-gray-600 text-gray-300'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        disabled={isAdvancedSelected}
                        onChange={() => {}}
                        className="mt-0.5 rounded text-purple-500 focus:ring-purple-500 h-4 w-4 bg-inputBg border-gray-600"
                      />
                      <div className="flex flex-col gap-0.5">
                        <span className="font-semibold text-xs text-gray-100">{item.name}</span>
                        <span className="text-[11px] text-gray-400">{item.desc}</span>
                      </div>
                    </label>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Section 4: LLM Services */}
          <div className="space-y-3 pt-2 border-t border-borderDark/60">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-semibold text-gray-300 mb-1.5">
                  LLM Generator Chính:
                </label>
                <select
                  value={config.llm_service || 'nvidia'}
                  onChange={(e) => onChangeConfig({ ...config, llm_service: e.target.value })}
                  className="w-full bg-inputBg border border-borderDark/80 rounded-xl px-3 py-2 text-xs text-gray-200 focus:outline-none focus:border-accentGreen"
                >
                  <option value="nvidia">NVIDIA NIM (Nemotron 3 Ultra 550B)</option>
                  <option value="groq">Groq (Llama 3.3 70B Versatile)</option>
                  <option value="google">Google Gemini (Gemma 4 31B)</option>
                  <option value="local">Local Model (ViLegalQwen 1.5B)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-gray-300 mb-1.5">
                  Sub-LLM (Mở rộng & Rerank):
                </label>
                <select
                  value={config.sub_llm_service || 'nvidia'}
                  onChange={(e) => onChangeConfig({ ...config, sub_llm_service: e.target.value })}
                  className="w-full bg-inputBg border border-borderDark/80 rounded-xl px-3 py-2 text-xs text-gray-200 focus:outline-none focus:border-accentGreen"
                >
                  <option value="nvidia">NVIDIA NIM</option>
                  <option value="local">Local Model (SLM)</option>
                  <option value="groq">Groq</option>
                  <option value="google">Google Gemini</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3.5 border-t border-borderDark/80 bg-cardBg/50 flex items-center justify-between">
          <button
            type="button"
            onClick={handleResetBaseline}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-gray-400 hover:text-white rounded-lg hover:bg-hoverBg transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Khôi phục Baseline</span>
          </button>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-xs font-medium rounded-xl bg-accentGreen hover:bg-accentGreenHover text-white shadow transition-colors"
            >
              Áp dụng & Đóng
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
