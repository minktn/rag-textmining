import React from 'react';
import { 
  Menu, 
  Trash2, 
  Settings2, 
  MessageSquare, 
  BarChart3 
} from 'lucide-react';
import { PipelineConfig } from '../types/pipeline';

interface HeaderProps {
  title: string;
  activeTab: 'chat' | 'eval';
  pipelineConfig: PipelineConfig;
  isLoading?: boolean;
  onToggleSidebar: () => void;
  onClearChat: () => void;
  onChangeTab: (tab: 'chat' | 'eval') => void;
  onOpenSettings: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  title,
  activeTab,
  pipelineConfig,
  onToggleSidebar,
  onClearChat,
  onChangeTab,
  onOpenSettings,
}) => {
  const getDatabaseLabel = () => {
    if (pipelineConfig.database === 'graph') return 'Graph DB';
    if (pipelineConfig.database === 'contriever') return 'Contriever';
    return 'Vector DB (BGE-M3)';
  };

  return (
    <header className="h-14 border-b border-borderDark/60 bg-sidebarBg/40 backdrop-blur px-4 flex items-center justify-between sticky top-0 z-30">
      <div className="flex items-center gap-3">
        <button
          onClick={onToggleSidebar}
          className="p-2 rounded-lg text-gray-400 hover:bg-hoverBg hover:text-white transition-colors"
          title="Ẩn / Hiện Sidebar"
        >
          <Menu className="w-5 h-5" />
        </button>

        {/* Tab Navigation: Chat vs. Evaluation */}
        <div className="flex items-center p-1 bg-cardBg/90 border border-borderDark/80 rounded-xl">
          <button
            onClick={() => onChangeTab('chat')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === 'chat'
                ? 'bg-accentGreen text-white shadow-sm'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <MessageSquare className="w-3.5 h-3.5" />
            <span>Trò chuyện</span>
          </button>

          <button
            onClick={() => onChangeTab('eval')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === 'eval'
                ? 'bg-accentGreen text-white shadow-sm'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <BarChart3 className="w-3.5 h-3.5" />
            <span>Playground Đánh giá</span>
          </button>
        </div>

        {/* Active Title */}
        {activeTab === 'chat' && (
          <div className="hidden lg:flex items-center gap-2 text-xs text-gray-400 pl-2 border-l border-borderDark/60">
            <span className="text-gray-200 font-semibold truncate max-w-[150px]">{title}</span>
            <span className="text-[11px] px-2 py-0.5 rounded-full bg-cardBg border border-borderDark/60 text-emerald-400 font-medium">
              {getDatabaseLabel()}
            </span>
            {pipelineConfig.advanced && (
              <span className="text-[11px] px-2 py-0.5 rounded-full bg-amber-950/60 border border-amber-800/40 text-amber-300 font-medium">
                {pipelineConfig.advanced}
              </span>
            )}
          </div>
        )}
      </div>

      <div className="flex items-center gap-2.5">
        {/* Settings Button */}
        <button
          onClick={onOpenSettings}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-cardBg hover:bg-hoverBg border border-borderDark/80 text-gray-300 hover:text-white text-xs font-medium transition-colors shadow-sm"
          title="Cấu hình Pipeline RAG (Database, Advanced, Pre/Post Processing)"
        >
          <Settings2 className="w-4 h-4 text-accentGreen" />
          <span className="hidden sm:inline">Cấu hình RAG</span>
        </button>

        {activeTab === 'chat' && (
          /* Clear Chat Button */
          <button
            onClick={onClearChat}
            className="p-2 text-gray-400 hover:text-red-400 hover:bg-hoverBg rounded-lg transition-colors"
            title="Xóa lịch sử hội thoại hiện tại"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        )}
      </div>
    </header>
  );
};