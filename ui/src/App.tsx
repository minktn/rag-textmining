import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { ChatArea } from './components/ChatArea';
import { ChatInput } from './components/ChatInput';
import { PipelineSettingsModal, DEFAULT_PIPELINE_CONFIG } from './components/PipelineSettingsModal';
import { EvaluationPlayground } from './components/EvaluationPlayground';
import { useChatStorage } from './hooks/useChatStorage';
import { sendRAGQuery, checkServerHealth } from './services/api';
import { Message } from './types/chat';
import { PipelineConfig } from './types/pipeline';

const PIPELINE_CONFIG_STORAGE_KEY = 'vietlegal_rag_pipeline_config';

export const App: React.FC = () => {
  const {
    sessions,
    currentSession,
    activeSessionId,
    setActiveSessionId,
    createSession,
    deleteSession,
    renameSession,
    addMessage,
    clearCurrentMessages,
  } = useChatStorage();

  const [activeTab, setActiveTab] = useState<'chat' | 'eval'>('chat');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [loadingSessionIds, setLoadingSessionIds] = useState<Record<string, boolean>>({});
  const [serverOnline, setServerOnline] = useState(true);

  // Pipeline configuration (persisted to localStorage)
  const [pipelineConfig, setPipelineConfig] = useState<PipelineConfig>(() => {
    try {
      const saved = localStorage.getItem(PIPELINE_CONFIG_STORAGE_KEY);
      if (saved) return JSON.parse(saved);
    } catch (e) {
      console.error('Failed to parse saved pipeline config', e);
    }
    return DEFAULT_PIPELINE_CONFIG;
  });

  const handleUpdatePipelineConfig = (newConfig: PipelineConfig) => {
    setPipelineConfig(newConfig);
    try {
      localStorage.setItem(PIPELINE_CONFIG_STORAGE_KEY, JSON.stringify(newConfig));
    } catch (e) {
      console.error('Failed to save pipeline config', e);
    }
  };

  // Trạng thái loading chỉ tính riêng cho session đang mở hiện tại
  const isCurrentSessionLoading = Boolean(
    currentSession?.id && loadingSessionIds[currentSession.id]
  );

  // Check health periodically
  useEffect(() => {
    const ping = async () => {
      const ok = await checkServerHealth();
      setServerOnline(ok);
    };
    ping();
    const interval = setInterval(ping, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleSendMessage = async (queryText: string) => {
    if (!currentSession) return;

    const targetSessionId = currentSession.id;

    const userMessage: Message = {
      id: 'msg_' + Date.now(),
      role: 'user',
      content: queryText,
      timestamp: new Date().toISOString(),
    };

    addMessage(targetSessionId, userMessage);
    setLoadingSessionIds((prev) => ({ ...prev, [targetSessionId]: true }));

    try {
      const res = await sendRAGQuery(
        queryText,
        true,
        pipelineConfig.top_k || 5,
        targetSessionId,
        pipelineConfig
      );

      const assistantMessage: Message = {
        id: res.request_id || 'msg_' + Date.now(),
        role: 'assistant',
        content: res.answer,
        timestamp: res.timestamp || new Date().toISOString(),
        rag_used: true,
        retrieval_mode: res.retrieval_mode,
        rewritten_query: res.rewritten_query,
        nli_verification: res.nli_verification,
        retrieved_chunks: res.retrieved_chunks,
        latency_ms: res.latency_ms,
        input_sha256: res.input_sha256,
      };

      addMessage(targetSessionId, assistantMessage);
    } catch (error: any) {
      const errorMessage: Message = {
        id: 'err_' + Date.now(),
        role: 'assistant',
        content: `⚠️ Không thể kết nối tới máy chủ RAG: ${error.message || error}. Hãy đảm bảo backend FastAPI đang chạy trên cổng 8002.`,
        timestamp: new Date().toISOString(),
        rag_used: true,
      };
      addMessage(targetSessionId, errorMessage);
    } finally {
      setLoadingSessionIds((prev) => {
        const next = { ...prev };
        delete next[targetSessionId];
        return next;
      });
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-chatBg text-gray-100 font-sans">
      {/* Sidebar (Chỉ hiển thị hoặc phục vụ lịch sử chat) */}
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        isOpen={isSidebarOpen}
        serverOnline={serverOnline}
        loadingSessionIds={loadingSessionIds}
        onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
        onSelectSession={(id) => {
          setActiveSessionId(id);
          setActiveTab('chat');
        }}
        onCreateSession={() => {
          createSession(true);
          setActiveTab('chat');
        }}
        onDeleteSession={deleteSession}
        onRenameSession={renameSession}
      />

      {/* Main Area */}
      <main className="flex-1 flex flex-col h-full min-w-0 overflow-hidden relative">
        <Header
          title={currentSession?.title || 'Đoạn chat mới'}
          activeTab={activeTab}
          pipelineConfig={pipelineConfig}
          isLoading={isCurrentSessionLoading}
          onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
          onClearChat={clearCurrentMessages}
          onChangeTab={setActiveTab}
          onOpenSettings={() => setIsSettingsOpen(true)}
        />

        {activeTab === 'chat' ? (
          <>
            <ChatArea
              messages={currentSession?.messages || []}
              isLoading={isCurrentSessionLoading}
              onSendSuggestion={handleSendMessage}
            />

            <ChatInput
              isLoading={isCurrentSessionLoading}
              pipelineConfig={pipelineConfig}
              onSendMessage={handleSendMessage}
            />
          </>
        ) : (
          <EvaluationPlayground />
        )}

        {/* Pipeline Settings Modal */}
        <PipelineSettingsModal
          isOpen={isSettingsOpen}
          onClose={() => setIsSettingsOpen(false)}
          config={pipelineConfig}
          onChangeConfig={handleUpdatePipelineConfig}
        />
      </main>
    </div>
  );
};

export default App;
