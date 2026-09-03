import React, { useState, useEffect } from 'react';
import { 
  Play, 
  RotateCcw, 
  AlertCircle, 
  CheckCircle, 
  Database, 
  Sparkles, 
  Layers, 
  Cpu, 
  Shuffle, 
  Clock, 
  Award, 
  FileText, 
  ChevronDown, 
  ChevronUp, 
  BarChart3, 
  Zap 
} from 'lucide-react';
import { EvalInfo, EvalReport } from '../types/pipeline';
import { getEvalInfo, runEvaluation, getLatestEval } from '../services/api';

export const EvaluationPlayground: React.FC = () => {
  // ── Eval Info & State ─────────────────────────────────────────
  const [evalInfo, setEvalInfo] = useState<EvalInfo | null>(null);
  const [maxQuestions, setMaxQuestions] = useState<number>(114);
  const [report, setReport] = useState<EvalReport | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [infoMsg, setInfoMsg] = useState<string | null>(null);

  // ── Pipeline Settings ─────────────────────────────────────────
  const [database, setDatabase] = useState<'base' | 'contriever' | 'graph'>('base');
  const [advanced, setAdvanced] = useState<string | null>(null);
  const [preprocessing, setPreprocessing] = useState<string[]>([]);
  const [postprocessing, setPostprocessing] = useState<string[]>([]);
  const [llmService, setLlmService] = useState<string>('nvidia');
  const [subLlmService, setSubLlmService] = useState<string>('nvidia');

  // ── Evaluation Execution Settings ─────────────────────────────
  const [sampleCount, setSampleCount] = useState<number>(1);
  const [isRandomSample, setIsRandomSample] = useState<boolean>(true);
  const [randomSeed, setRandomSeed] = useState<number>(42);
  const [skipRagas, setSkipRagas] = useState<boolean>(true);
  const [ragasService, setRagasService] = useState<string>('nvidia');

  // ── UI Filter & Accordions ────────────────────────────────────
  const [expandedQuestionIds, setExpandedQuestionIds] = useState<Record<string, boolean>>({});
  const [searchQuery, setSearchQuery] = useState<string>('');

  // ── Load dataset info on mount ────────────────────────────────
  useEffect(() => {
    const fetchInfo = async () => {
      try {
        const info = await getEvalInfo();
        setEvalInfo(info);
        if (info.total_questions > 0) {
          setMaxQuestions(info.total_questions);
        }
      } catch (err: any) {
        console.warn('Could not fetch eval info:', err);
      }
    };
    fetchInfo();

    // Tự động nạp kết quả gần nhất nếu có
    const loadLatest = async () => {
      try {
        const latest = await getLatestEval();
        setReport(latest);
      } catch (err) {
        // Chưa có kết quả gần đây
      }
    };
    loadLatest();
  }, []);

  // ── Input Validation for N ────────────────────────────────────
  const isInputEmpty = sampleCount === null || sampleCount === undefined || isNaN(sampleCount);
  const isTooSmall = !isInputEmpty && sampleCount <= 0;
  const isTooLarge = !isInputEmpty && sampleCount > maxQuestions;
  const isInputValid = !isInputEmpty && !isTooSmall && !isTooLarge;

  let validationError: string | null = null;
  if (isInputEmpty) {
    validationError = 'Vui lòng nhập số lượng mẫu thử (N).';
  } else if (isTooSmall) {
    validationError = '⚠️ Số lượng mẫu thử phải lớn hơn 0 (N > 0).';
  } else if (isTooLarge) {
    validationError = `⚠️ Vượt quá số lượng mẫu tối đa trong tập dữ liệu (tối đa ${maxQuestions} mẫu).`;
  }

  // ── Handlers ──────────────────────────────────────────────────
  const handleToggleAdvanced = (val: string) => {
    setAdvanced((prev) => (prev === val ? null : val));
  };

  const handleTogglePreprocessing = (val: string) => {
    setPreprocessing((prev) =>
      prev.includes(val) ? prev.filter((item) => item !== val) : [...prev, val]
    );
  };

  const handleTogglePostprocessing = (method: string) => {
    const isAdaptable = method === 'prompt_compression';
    if (postprocessing.includes(method)) {
      setPostprocessing((prev) => prev.filter((item) => item !== method));
    } else {
      if (isAdaptable) {
        setPostprocessing((prev) => [...prev, method]);
      } else {
        // Tối đa 1 phương thức non-adaptable (filter_rerank hoặc crag)
        setPostprocessing((prev) => {
          const onlyAdaptable = prev.filter((item) => item === 'prompt_compression');
          return [...onlyAdaptable, method];
        });
      }
    }
  };

  const handleToggleQuestion = (id: string) => {
    setExpandedQuestionIds((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const handleLoadLatest = async () => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const res = await getLatestEval();
      setReport(res);
      setInfoMsg('Đã nạp báo cáo đánh giá mới nhất thành công.');
      setTimeout(() => setInfoMsg(null), 3000);
    } catch (err: any) {
      setErrorMsg(err.message || 'Không thể tải báo cáo mới nhất.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunEvaluation = async () => {
    if (!isInputValid || isLoading) return;

    setIsLoading(true);
    setErrorMsg(null);
    setInfoMsg(null);

    try {
      const result = await runEvaluation({
        limit: sampleCount,
        random_sample: isRandomSample,
        random_seed: randomSeed,
        retriever_mode: database,
        advanced: advanced,
        preprocessing: advanced ? [] : preprocessing,
        postprocessing: advanced ? [] : postprocessing,
        llm_service: llmService,
        sub_llm_service: subLlmService,
        ragas_service: ragasService,
        skip_ragas: skipRagas,
        top_k: 5,
      });
      setReport(result);
      setInfoMsg(`Đánh giá thành công ${sampleCount} mẫu câu hỏi!`);
      setTimeout(() => setInfoMsg(null), 4000);
    } catch (err: any) {
      setErrorMsg(err.message || 'Có lỗi xảy ra trong quá trình đánh giá.');
    } finally {
      setIsLoading(false);
    }
  };

  const isAdvancedSelected = Boolean(advanced);

  // Lọc danh sách câu hỏi chi tiết
  const filteredQuestions = (report?.detailed_results || []).filter((q) => {
    if (!searchQuery.trim()) return true;
    const qText = q.question.toLowerCase();
    const qId = q.id.toLowerCase();
    const s = searchQuery.toLowerCase().trim();
    return qText.includes(s) || qId.includes(s);
  });

  return (
    <div className="flex-1 overflow-y-auto bg-chatBg text-gray-100 p-4 sm:p-6 lg:p-8 space-y-8">
      {/* ── Top Header ───────────────────────────────────────── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-borderDark/60 pb-6">
        <div>
          <div className="flex items-center gap-3 mb-1.5">
            <div className="p-2.5 rounded-2xl bg-gradient-to-br from-emerald-500/20 to-teal-500/10 text-emerald-400 border border-emerald-500/30 shadow-lg">
              <BarChart3 className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2.5">
                Evaluation Playground
                <span className="text-xs font-medium px-2.5 py-0.5 rounded-full bg-emerald-950/80 text-emerald-400 border border-emerald-800/50">
                  {evalInfo?.law || 'Pháp luật Đất đai 2024'}
                </span>
              </h1>
              <p className="text-xs text-gray-400">
                Môi trường đánh giá định lượng Retrieval & Generation độc lập với Chatbot
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2.5">
          <button
            type="button"
            onClick={handleLoadLatest}
            disabled={isLoading}
            className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-cardBg hover:bg-hoverBg border border-borderDark/80 text-gray-300 hover:text-white text-xs font-medium transition-all shadow-sm"
            title="Đọc lại kết quả mới nhất đã lưu tại db/results/eval_latest.json"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Nạp kết quả gần nhất</span>
          </button>
        </div>
      </div>

      {/* Notifications */}
      {infoMsg && (
        <div className="p-3.5 rounded-xl bg-emerald-950/50 border border-emerald-600/50 text-emerald-300 text-xs flex items-center gap-2.5 animate-fadeIn">
          <CheckCircle className="w-4 h-4 flex-shrink-0" />
          <span>{infoMsg}</span>
        </div>
      )}
      {errorMsg && (
        <div className="p-3.5 rounded-xl bg-red-950/50 border border-red-600/50 text-red-300 text-xs flex items-center gap-2.5 animate-fadeIn">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* ── Control Panel: Settings & Execution ──────────────── */}
      <div className="bg-sidebarBg/90 border border-borderDark/80 rounded-2xl shadow-xl overflow-hidden">
        <div className="px-6 py-4 border-b border-borderDark/70 bg-cardBg/40 flex items-center justify-between">
          <span className="text-sm font-semibold text-gray-200 flex items-center gap-2">
            <Zap className="w-4 h-4 text-emerald-400" />
            Bảng điều khiển thử nghiệm (Pipeline & Test Set Settings)
          </span>
          <span className="text-xs text-gray-400">
            Tập dữ liệu: <strong className="text-gray-200">eval_landlaw_2024.json</strong> ({maxQuestions} câu hỏi)
          </span>
        </div>

        <div className="p-6 space-y-6">
          {/* Row 1: Pipeline Configuration (Tickboxes) */}
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Database Selection */}
            <div className="space-y-2.5">
              <label className="text-xs font-semibold text-gray-300 uppercase tracking-wider flex items-center gap-1.5">
                <Database className="w-3.5 h-3.5 text-emerald-400" />
                Cơ sở dữ liệu
              </label>
              <div className="space-y-1.5">
                {[
                  { id: 'base', name: 'Vector DB (BGE-M3)', desc: 'Dense Search + CrossEncoder' },
                  { id: 'contriever', name: 'Contriever', desc: 'mContriever Dense' },
                  { id: 'graph', name: 'Graph Database', desc: 'Microsoft GraphRAG' },
                ].map((item) => (
                  <label
                    key={item.id}
                    onClick={() => setDatabase(item.id as any)}
                    className={`flex items-center justify-between p-2.5 rounded-xl border cursor-pointer text-xs transition-all ${
                      database === item.id
                        ? 'border-accentGreen bg-accentGreen/15 text-white font-medium shadow-sm'
                        : 'border-borderDark/60 bg-cardBg/30 hover:bg-cardBg/70 text-gray-300'
                    }`}
                  >
                    <span>{item.name}</span>
                    <input
                      type="radio"
                      name="eval_db"
                      checked={database === item.id}
                      onChange={() => {}}
                      className="text-accentGreen focus:ring-accentGreen h-3.5 w-3.5"
                    />
                  </label>
                ))}
              </div>
            </div>

            {/* Advanced Method */}
            <div className="space-y-2.5">
              <label className="text-xs font-semibold text-gray-300 uppercase tracking-wider flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                Advanced Method
              </label>
              <div className="space-y-1.5">
                {[
                  { id: 'rag_fusion', name: 'RAG-Fusion', desc: 'Multi-query + RRF' },
                ].map((item) => {
                  const checked = advanced === item.id;
                  return (
                    <label
                      key={item.id}
                      onClick={() => handleToggleAdvanced(item.id)}
                      className={`flex items-center justify-between p-2.5 rounded-xl border cursor-pointer text-xs transition-all ${
                        checked
                          ? 'border-amber-500/80 bg-amber-500/15 text-white font-medium shadow-sm'
                          : 'border-borderDark/60 bg-cardBg/30 hover:bg-cardBg/70 text-gray-300'
                      }`}
                    >
                      <span className="flex items-center gap-1.5">
                        {item.name}
                        <span className="text-[10px] text-gray-400">({item.desc})</span>
                      </span>
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => {}}
                        className="rounded text-amber-500 focus:ring-amber-500 h-3.5 w-3.5 bg-inputBg border-gray-600"
                      />
                    </label>
                  );
                })}
              </div>
              {isAdvancedSelected && (
                <p className="text-[10px] text-amber-400/90 leading-tight">
                  ⚡ Advanced sẽ tự động thay thế Preprocessing & Postprocessing.
                </p>
              )}
            </div>

            {/* Preprocessing */}
            <div className={`space-y-2.5 ${isAdvancedSelected ? 'opacity-30 pointer-events-none' : ''}`}>
              <label className="text-xs font-semibold text-gray-300 uppercase tracking-wider flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-cyan-400" />
                Preprocessing
              </label>
              <div className="space-y-1.5">
                {[
                  { id: 'hyde', name: 'HyDE', desc: 'Hypothetical Embeddings' },
                ].map((item) => {
                  const checked = preprocessing.includes(item.id);
                  return (
                    <label
                      key={item.id}
                      onClick={() => !isAdvancedSelected && handleTogglePreprocessing(item.id)}
                      className={`flex items-center justify-between p-2.5 rounded-xl border cursor-pointer text-xs transition-all ${
                        checked
                          ? 'border-cyan-500/80 bg-cyan-500/15 text-white font-medium shadow-sm'
                          : 'border-borderDark/60 bg-cardBg/30 hover:bg-cardBg/70 text-gray-300'
                      }`}
                    >
                      <span className="flex items-center gap-1.5">
                        {item.name}
                        <span className="text-[10px] text-gray-400">({item.desc})</span>
                      </span>
                      <input
                        type="checkbox"
                        checked={checked}
                        disabled={isAdvancedSelected}
                        onChange={() => {}}
                        className="rounded text-cyan-500 focus:ring-cyan-500 h-3.5 w-3.5 bg-inputBg border-gray-600"
                      />
                    </label>
                  );
                })}
              </div>
            </div>

            {/* Postprocessing */}
            <div className={`space-y-2.5 ${isAdvancedSelected ? 'opacity-30 pointer-events-none' : ''}`}>
              <label className="text-xs font-semibold text-gray-300 uppercase tracking-wider flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-purple-400" />
                Postprocessing
              </label>
              <div className="space-y-1.5">
                {[
                  { id: 'filter_rerank', name: 'Filter-then-Rerank', desc: 'SLM 1.5B + NVIDIA' },
                  { id: 'crag', name: 'CRAG (NLI)', desc: 'Corrective Verification' },
                  { id: 'prompt_compression', name: 'Prompt Compression', desc: 'LongLLMLingua' },
                ].map((item) => {
                  const checked = postprocessing.includes(item.id);
                  return (
                    <label
                      key={item.id}
                      onClick={() => !isAdvancedSelected && handleTogglePostprocessing(item.id)}
                      className={`flex items-center justify-between p-2 rounded-xl border cursor-pointer text-xs transition-all ${
                        checked
                          ? 'border-purple-500/80 bg-purple-500/15 text-white font-medium shadow-sm'
                          : 'border-borderDark/60 bg-cardBg/30 hover:bg-cardBg/70 text-gray-300'
                      }`}
                    >
                      <span className="truncate pr-2">{item.name}</span>
                      <input
                        type="checkbox"
                        checked={checked}
                        disabled={isAdvancedSelected}
                        onChange={() => {}}
                        className="rounded text-purple-500 focus:ring-purple-500 h-3.5 w-3.5 bg-inputBg border-gray-600 flex-shrink-0"
                      />
                    </label>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Row 1.5: LLM & RAGAS Settings */}
          <div className="pt-4 border-t border-borderDark/60 grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
            <div>
              <label className="block text-gray-400 mb-1.5 font-semibold">LLM Generator:</label>
              <select
                value={llmService}
                onChange={(e) => setLlmService(e.target.value)}
                className="w-full bg-inputBg border border-borderDark/80 rounded-xl px-3 py-2 text-xs text-gray-200 focus:outline-none focus:border-accentGreen"
              >
                <option value="nvidia">NVIDIA NIM (Nemotron 3 Ultra)</option>
                <option value="groq">Groq (Llama 3.3 70B)</option>
                <option value="google">Google Gemini</option>
                <option value="local">Local (ViLegalQwen 1.5B)</option>
              </select>
            </div>
            <div>
              <label className="block text-gray-400 mb-1.5 font-semibold">Sub-LLM (Expansion & Rerank):</label>
              <select
                value={subLlmService}
                onChange={(e) => setSubLlmService(e.target.value)}
                className="w-full bg-inputBg border border-borderDark/80 rounded-xl px-3 py-2 text-xs text-gray-200 focus:outline-none focus:border-accentGreen"
              >
                <option value="nvidia">NVIDIA NIM</option>
                <option value="local">Local Model (SLM)</option>
                <option value="groq">Groq</option>
                <option value="google">Google Gemini</option>
              </select>
            </div>
            <div>
              <label className="block text-gray-400 mb-1.5 font-semibold">Chỉ số RAGAS:</label>
              <div className="flex items-center gap-2 mt-2">
                <label
                  onClick={() => setSkipRagas(!skipRagas)}
                  className="flex items-center gap-2 cursor-pointer text-gray-300"
                >
                  <input
                    type="checkbox"
                    checked={skipRagas}
                    onChange={() => {}}
                    className="rounded text-accentGreen focus:ring-accentGreen h-3.5 w-3.5 bg-inputBg border-gray-600"
                  />
                  <span>Bỏ qua RAGAS (chạy nhanh)</span>
                </label>
                {!skipRagas && (
                  <select
                    value={ragasService}
                    onChange={(e) => setRagasService(e.target.value)}
                    className="bg-inputBg border border-borderDark/80 rounded-lg px-2 py-1 text-[11px] text-gray-200"
                  >
                    <option value="nvidia">NVIDIA</option>
                    <option value="groq">Groq</option>
                    <option value="google">Google</option>
                  </select>
                )}
              </div>
            </div>
          </div>

          {/* Row 2: Sampling & Execution Parameters */}
          <div className="p-4 rounded-xl bg-cardBg/40 border border-borderDark/60 grid grid-cols-1 md:grid-cols-12 gap-5 items-center">
            {/* Number of Samples (N) */}
            <div className="md:col-span-5 space-y-1.5">
              <div className="flex items-center justify-between">
                <label className="text-xs font-semibold text-gray-200 flex items-center gap-1.5">
                  <span>Số lượng mẫu thử (N):</span>
                  <span className="text-[11px] text-gray-400">1 → {maxQuestions}</span>
                </label>
                {/* Shortcuts */}
                <div className="flex items-center gap-1">
                  {[1, 5, 10, 20, maxQuestions].map((n) => (
                    <button
                      key={n}
                      type="button"
                      onClick={() => setSampleCount(n)}
                      className={`px-1.5 py-0.5 rounded text-[10px] font-medium transition-colors ${
                        sampleCount === n
                          ? 'bg-accentGreen text-white'
                          : 'bg-inputBg text-gray-400 hover:text-gray-200'
                      }`}
                    >
                      {n === maxQuestions ? 'Max' : n}
                    </button>
                  ))}
                </div>
              </div>

              <div className="relative">
                <input
                  type="number"
                  min={1}
                  max={maxQuestions}
                  value={isInputEmpty ? '' : sampleCount}
                  onChange={(e) => {
                    const val = e.target.value === '' ? (null as any) : parseInt(e.target.value, 10);
                    setSampleCount(val);
                  }}
                  className={`w-full bg-inputBg border rounded-xl px-3.5 py-2 text-sm text-white focus:outline-none transition-all ${
                    validationError
                      ? 'border-red-500/80 bg-red-950/20 text-red-200 focus:border-red-500'
                      : 'border-borderDark/80 focus:border-accentGreen'
                  }`}
                  placeholder={`Nhập số lượng từ 1 đến ${maxQuestions}`}
                />
              </div>

              {/* Validation Warning / Error Text */}
              {validationError ? (
                <div className="text-[11px] text-red-400 font-medium flex items-center gap-1 animate-fadeIn">
                  <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
                  <span>{validationError}</span>
                </div>
              ) : (
                <div className="text-[11px] text-emerald-400 flex items-center gap-1">
                  <CheckCircle className="w-3.5 h-3.5 flex-shrink-0" />
                  <span>Hợp lệ: Sẽ đánh giá {sampleCount}/{maxQuestions} câu hỏi</span>
                </div>
              )}
            </div>

            {/* Random Flag & Seed */}
            <div className="md:col-span-4 space-y-1.5">
              <label
                onClick={() => setIsRandomSample(!isRandomSample)}
                className="flex items-center gap-2.5 cursor-pointer text-xs font-semibold text-gray-200"
              >
                <input
                  type="checkbox"
                  checked={isRandomSample}
                  onChange={() => {}}
                  className="rounded text-accentGreen focus:ring-accentGreen h-4 w-4 bg-inputBg border-gray-600"
                />
                <span className="flex items-center gap-1.5">
                  <Shuffle className="w-3.5 h-3.5 text-accentGreen" />
                  Lấy mẫu ngẫu nhiên (Random Sampling)
                </span>
              </label>

              <div className="flex items-center gap-2 pl-6">
                <span className="text-[11px] text-gray-400">Random Seed:</span>
                <input
                  type="number"
                  disabled={!isRandomSample}
                  value={randomSeed}
                  onChange={(e) => setRandomSeed(parseInt(e.target.value, 10) || 42)}
                  className="w-16 bg-inputBg border border-borderDark/80 rounded-lg px-2 py-1 text-xs text-center text-white disabled:opacity-40"
                />
                <span className="text-[10px] text-gray-500">Mặc định: 42</span>
              </div>
            </div>

            {/* Execute Run Button */}
            <div className="md:col-span-3 flex justify-end">
              <button
                type="button"
                disabled={!isInputValid || isLoading}
                onClick={handleRunEvaluation}
                className={`w-full py-3 px-4 rounded-xl font-bold text-xs flex items-center justify-center gap-2 shadow-lg transition-all ${
                  isInputValid && !isLoading
                    ? 'bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700 text-white shadow-emerald-950/50 cursor-pointer active:scale-[0.98]'
                    : 'bg-gray-800 text-gray-500 border border-borderDark/40 cursor-not-allowed'
                }`}
              >
                {isLoading ? (
                  <>
                    <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    <span>Đang đánh giá {sampleCount} câu...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-white" />
                    <span>BẮT ĐẦU ĐÁNH GIÁ</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ── Results Display Section ──────────────────────────── */}
      {/* ── Results Display Section ──────────────────────────── */}
      {report && (report.summary_metrics || report.detailed_results) ? (
        <div className="space-y-6 animate-fadeIn">
          {/* Metadata Bar */}
          <div className="flex flex-wrap items-center justify-between gap-3 p-4 rounded-2xl bg-cardBg/60 border border-borderDark/80 text-xs">
            <div className="flex flex-wrap items-center gap-3">
              <span className="flex items-center gap-1.5 text-gray-300">
                <Clock className="w-3.5 h-3.5 text-emerald-400" />
                Thời gian: <strong>{new Date(report.metadata?.timestamp || report.timestamp || Date.now()).toLocaleString('vi-VN')}</strong>
              </span>
              <span className="text-gray-600">•</span>
              <span className="text-gray-300">
                Đã test: <strong className="text-white">{report.detailed_results?.length || 0} mẫu</strong>
              </span>
              <span className="text-gray-600">•</span>
              <span className="text-gray-300">
                Retriever: <strong className="text-emerald-400">{report.metadata?.configuration?.retriever_mode || database}</strong>
              </span>
              <span className="text-gray-600">•</span>
              <span className="text-gray-300">
                Advanced: <strong className="text-amber-400">{report.metadata?.configuration?.advanced_method || 'None'}</strong>
              </span>
            </div>
            <div className="flex items-center gap-2 text-gray-400 text-[11px]">
              <span>LLM: {report.metadata?.configuration?.llm_service || llmService}</span>
            </div>
          </div>

          {/* KPI Metric Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3.5">
            {/* Hit Rate */}
            <div className="p-4 rounded-2xl bg-sidebarBg border border-borderDark/80 shadow-md flex flex-col justify-between">
              <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wider">Hit Rate</span>
              <div className="my-2">
                <span className="text-2xl font-black text-emerald-400">
                  {((report.summary_metrics?.retrieval?.hit_rate ?? 0) * 100).toFixed(1)}%
                </span>
              </div>
              <span className="text-[10px] text-gray-500">Tỷ lệ tìm trúng điều luật</span>
            </div>

            {/* MRR */}
            <div className="p-4 rounded-2xl bg-sidebarBg border border-borderDark/80 shadow-md flex flex-col justify-between">
              <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wider">MRR</span>
              <div className="my-2">
                <span className="text-2xl font-black text-teal-300">
                  {(report.summary_metrics?.retrieval?.mrr ?? 0).toFixed(4)}
                </span>
              </div>
              <span className="text-[10px] text-gray-500">Mean Reciprocal Rank</span>
            </div>

            {/* Recall@5 */}
            <div className="p-4 rounded-2xl bg-sidebarBg border border-borderDark/80 shadow-md flex flex-col justify-between">
              <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wider">Recall@5</span>
              <div className="my-2">
                <span className="text-2xl font-black text-cyan-400">
                  {(report.summary_metrics?.retrieval?.recall_at_k ?? 0).toFixed(4)}
                </span>
              </div>
              <span className="text-[10px] text-gray-500">Độ phủ tài liệu liên quan</span>
            </div>

            {/* F1 Score */}
            <div className="p-4 rounded-2xl bg-sidebarBg border border-borderDark/80 shadow-md flex flex-col justify-between">
              <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wider">Avg F1 Score</span>
              <div className="my-2">
                <span className="text-2xl font-black text-amber-400">
                  {(report.summary_metrics?.generation?.f1 ?? 0).toFixed(4)}
                </span>
              </div>
              <span className="text-[10px] text-gray-500">Độ chính xác text sinh ra</span>
            </div>

            {/* ROUGE-L */}
            <div className="p-4 rounded-2xl bg-sidebarBg border border-borderDark/80 shadow-md flex flex-col justify-between">
              <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wider">ROUGE-L</span>
              <div className="my-2">
                <span className="text-2xl font-black text-orange-400">
                  {(report.summary_metrics?.generation?.rouge_l ?? 0).toFixed(4)}
                </span>
              </div>
              <span className="text-[10px] text-gray-500">Trùng khớp chuỗi dài nhất</span>
            </div>

            {/* Latency */}
            <div className="p-4 rounded-2xl bg-sidebarBg border border-borderDark/80 shadow-md flex flex-col justify-between">
              <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wider">E2E Latency</span>
              <div className="my-2">
                <span className="text-2xl font-black text-purple-400">
                  {((report.summary_metrics?.generation?.e2e_latency_ms ?? 0) / 1000).toFixed(1)}s
                </span>
              </div>
              <span className="text-[10px] text-gray-500">
                Ret: {((report.summary_metrics?.retrieval?.latency_ms ?? 0) / 1000).toFixed(1)}s
              </span>
            </div>
          </div>

          {/* Breakdown by Question Type */}
          {report.summary_metrics?.per_question_type && Object.keys(report.summary_metrics.per_question_type).length > 0 && (
            <div className="bg-sidebarBg border border-borderDark/80 rounded-2xl overflow-hidden shadow-md">
              <div className="px-6 py-3.5 border-b border-borderDark/60 bg-cardBg/30 flex items-center justify-between">
                <h3 className="text-xs font-semibold text-gray-200 uppercase tracking-wider flex items-center gap-2">
                  <Award className="w-4 h-4 text-emerald-400" />
                  Chỉ số theo từng dạng câu hỏi (Question Type Breakdown)
                </h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-cardBg/50 text-gray-400 border-b border-borderDark/50">
                    <tr>
                      <th className="py-2.5 px-4 font-semibold">Dạng câu hỏi</th>
                      <th className="py-2.5 px-4 font-semibold">Số lượng (N)</th>
                      <th className="py-2.5 px-4 font-semibold">Hit Rate</th>
                      <th className="py-2.5 px-4 font-semibold">Recall@K</th>
                      <th className="py-2.5 px-4 font-semibold">F1 Score</th>
                      <th className="py-2.5 px-4 font-semibold">BLEU-1</th>
                      <th className="py-2.5 px-4 font-semibold">ROUGE-L</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-borderDark/30 text-gray-300">
                    {Object.entries(report.summary_metrics.per_question_type).map(([qType, stats]: [string, any]) => (
                      <tr key={qType} className="hover:bg-cardBg/40 transition-colors">
                        <td className="py-3 px-4 font-semibold text-white capitalize">{qType}</td>
                        <td className="py-3 px-4">{stats.count}</td>
                        <td className="py-3 px-4 text-emerald-400 font-bold">{((stats.hit_rate ?? 0) * 100).toFixed(1)}%</td>
                        <td className="py-3 px-4">{(stats.avg_recall_at_k ?? 0).toFixed(4)}</td>
                        <td className="py-3 px-4 font-medium text-amber-300">{(stats.avg_f1 ?? 0).toFixed(4)}</td>
                        <td className="py-3 px-4">{(stats.avg_bleu_1 ?? 0).toFixed(4)}</td>
                        <td className="py-3 px-4">{(stats.avg_rouge_l ?? 0).toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Detailed Question Results */}
          <div className="bg-sidebarBg border border-borderDark/80 rounded-2xl overflow-hidden shadow-md">
            <div className="px-6 py-4 border-b border-borderDark/60 bg-cardBg/30 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <h3 className="text-xs font-semibold text-gray-200 uppercase tracking-wider flex items-center gap-2">
                <FileText className="w-4 h-4 text-emerald-400" />
                Chi tiết từng câu hỏi ({filteredQuestions.length}/{report.detailed_results?.length || 0})
              </h3>
              <div className="w-full sm:w-64">
                <input
                  type="text"
                  placeholder="Lọc theo ID hoặc nội dung..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full bg-inputBg border border-borderDark/80 rounded-xl px-3 py-1.5 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-accentGreen"
                />
              </div>
            </div>

            <div className="divide-y divide-borderDark/40">
              {filteredQuestions.map((q) => {
                const isExpanded = Boolean(expandedQuestionIds[q.id]);
                return (
                  <div key={q.id} className="p-4 hover:bg-cardBg/20 transition-colors space-y-3">
                    {/* Item Header */}
                    <div 
                      onClick={() => handleToggleQuestion(q.id)}
                      className="flex items-start justify-between gap-4 cursor-pointer"
                    >
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="px-2 py-0.5 rounded bg-emerald-950/70 border border-emerald-700/40 text-emerald-400 font-mono text-[11px] font-bold">
                            {q.id}
                          </span>
                          <span className="text-[10px] px-2 py-0.5 rounded bg-gray-800 text-gray-400 uppercase font-semibold">
                            {q.question_type}
                          </span>
                          <span className="text-[10px] text-gray-500">
                            {(q.e2e_latency_ms / 1000).toFixed(1)}s
                          </span>
                        </div>
                        <h4 className="text-sm font-semibold text-gray-100 leading-snug">
                          {q.question}
                        </h4>
                      </div>

                      <div className="flex items-center gap-2 flex-shrink-0 pt-1">
                        <span className="text-xs text-gray-400 flex items-center gap-1">
                          {q.retrieved_law_ids?.length || 0} Điều luật
                        </span>
                        {isExpanded ? (
                          <ChevronUp className="w-4 h-4 text-gray-400" />
                        ) : (
                          <ChevronDown className="w-4 h-4 text-gray-400" />
                        )}
                      </div>
                    </div>

                    {/* Retrieved Law IDs Badges */}
                    <div className="flex flex-wrap gap-1.5 items-center">
                      <span className="text-[11px] text-gray-400 mr-1">Law IDs:</span>
                      {q.retrieved_law_ids && q.retrieved_law_ids.length > 0 ? (
                        q.retrieved_law_ids.map((lid, idx) => (
                          <span
                            key={idx}
                            className="inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-md bg-cardBg border border-borderDark/80 text-gray-200"
                          >
                            <strong className="text-emerald-400">
                              {lid.article_no !== null ? `Điều ${lid.article_no}` : 'N/A'}
                            </strong>
                            {lid.clause_nos && lid.clause_nos.length > 0 && (
                              <span className="text-gray-400">
                                (Khoản {lid.clause_nos.join(', ')})
                              </span>
                            )}
                            {lid.chapter_no && (
                              <span className="text-gray-500 text-[10px]">
                                [Chương {lid.chapter_no}]
                              </span>
                            )}
                          </span>
                        ))
                      ) : (
                        <span className="text-[11px] text-gray-500 italic">Không tìm thấy</span>
                      )}
                    </div>

                    {/* Expanded Content */}
                    {isExpanded && (
                      <div className="pt-3 border-t border-borderDark/40 space-y-4 text-xs animate-fadeIn">
                        {/* Answers Comparison */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <div className="p-3.5 rounded-xl bg-cardBg/40 border border-borderDark/60 space-y-1.5">
                            <span className="font-semibold text-emerald-400 block uppercase tracking-wider text-[10px]">
                              Ground Truth (Đáp án chuẩn):
                            </span>
                            <p className="text-gray-300 leading-relaxed whitespace-pre-wrap">
                              {q.ground_truth || 'N/A'}
                            </p>
                          </div>

                          <div className="p-3.5 rounded-xl bg-cardBg/40 border border-borderDark/60 space-y-1.5">
                            <span className="font-semibold text-teal-400 block uppercase tracking-wider text-[10px]">
                              Generated Answer (Mô hình sinh):
                            </span>
                            <p className="text-gray-200 leading-relaxed whitespace-pre-wrap">
                              {q.generated_answer || 'N/A'}
                            </p>
                          </div>
                        </div>

                        {/* Retrieved Contexts */}
                        {q.retrieved_contexts && q.retrieved_contexts.length > 0 && (
                          <div className="space-y-2">
                            <span className="font-semibold text-gray-400 uppercase tracking-wider text-[10px]">
                              Ngữ cảnh trích xuất ({q.retrieved_contexts.length} đoạn):
                            </span>
                            <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
                              {q.retrieved_contexts.map((ctx, cIdx) => (
                                <div
                                  key={cIdx}
                                  className="p-2.5 rounded-lg bg-inputBg border border-borderDark/60 text-gray-300 text-[11px] leading-relaxed"
                                >
                                  <span className="text-emerald-400 font-mono font-bold mr-1.5">
                                    [Đoạn {cIdx + 1}]
                                  </span>
                                  {ctx}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      ) : (
        /* Empty State */
        <div className="p-12 text-center rounded-2xl bg-sidebarBg/50 border border-borderDark/50 space-y-3">
          <div className="w-12 h-12 rounded-2xl bg-cardBg text-gray-400 mx-auto flex items-center justify-center">
            <BarChart3 className="w-6 h-6" />
          </div>
          <h3 className="text-base font-semibold text-gray-200">Chưa có kết quả đánh giá nào</h3>
          <p className="text-xs text-gray-400 max-w-sm mx-auto">
            Chọn cấu hình pipeline, nhập số lượng mẫu thử (N) và ấn <strong>BẮT ĐẦU ĐÁNH GIÁ</strong> để chạy thử nghiệm.
          </p>
        </div>
      )}
    </div>
  );
};
