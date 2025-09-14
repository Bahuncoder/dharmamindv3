// Enhanced Rishi Guidance Panel - integrated within regular chat
import React, { useState } from 'react';

interface Rishi {
  id: string;
  name: string;
  sanskrit: string;
  specialization: string[];
  greeting: string;
  available: boolean;
  requires_upgrade?: boolean;
  teaching_style?: string;
  archetype?: string;
}

interface RishiGuidancePanelProps {
  onRishiSelect: (rishiId: string) => void;
  selectedRishi: string;
  userSubscription: string;
  isDemo?: boolean;
  availableRishis: Rishi[];
}

export const RishiGuidancePanel: React.FC<RishiGuidancePanelProps> = ({
  onRishiSelect,
  selectedRishi,
  userSubscription,
  isDemo = false,
  availableRishis
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="rishi-guidance-panel mb-4">
      {/* Toggle Button */}
      <button
        className="w-full flex items-center justify-between px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 bg-gradient-to-r from-orange-50 to-amber-50 border border-orange-200 hover:from-orange-100 hover:to-amber-100"
        style={{ 
          color: 'var(--color-text-primary, #1f2937)',
          borderColor: selectedRishi ? 'var(--color-primary-saffron, #f59e0b)' : undefined 
        }}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center space-x-3">
          <span className="text-lg">🧘</span>
          <div className="text-left">
            <div className="font-semibold">Spiritual Guidance</div>
            {selectedRishi && (
              <div className="text-xs text-orange-600">
                {availableRishis.find(r => r.id === selectedRishi)?.name || 'Active'}
              </div>
            )}
          </div>
        </div>
        <div className="flex items-center space-x-2">
          {selectedRishi && (
            <span className="text-xs px-2 py-1 rounded-full bg-orange-100 text-orange-700">
              Active
            </span>
          )}
          <span className={`transform transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`}>
            ▼
          </span>
        </div>
      </button>

      {/* Expandable Rishi Selection */}
      {isExpanded && (
        <div className="mt-2 p-3 bg-white rounded-lg border border-orange-200 shadow-sm space-y-2">
          <div className="text-xs text-gray-600 mb-3">
            Choose a spiritual guide for wisdom and guidance:
          </div>
          
          {/* Clear Selection */}
          <button
            className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-sm transition-all duration-200 ${
              !selectedRishi 
                ? 'bg-gray-100 border border-gray-300 text-gray-800' 
                : 'bg-gray-50 hover:bg-gray-100 text-gray-600'
            }`}
            onClick={() => onRishiSelect('')}
          >
            <div className="flex items-center space-x-3">
              <span className="text-lg">💬</span>
              <span>Regular Chat Only</span>
            </div>
            {!selectedRishi && (
              <span className="text-xs px-2 py-1 rounded-full bg-green-100 text-green-700">
                Active
              </span>
          )}
        </button>
        
        {/* Available Rishis */}
        {availableRishis.map((rishi) => (
          <button
            key={rishi.id}
            className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-sm transition-all duration-200 ${
              selectedRishi === rishi.id 
                ? 'bg-gradient-to-r from-orange-100 to-amber-100 border border-orange-300 text-orange-800' 
                : 'bg-gray-50 hover:bg-orange-50 text-gray-700 border border-gray-200'
            }`}
            onClick={() => onRishiSelect(rishi.id)}
            disabled={!rishi.available}
          >
            <div className="flex items-center space-x-3">
              <span className="text-lg">{getRishiIcon(rishi.id)}</span>
              <div className="text-left">
                <div className="font-medium">{rishi.name}</div>
                <div className="text-xs text-gray-500">{rishi.specialization.slice(0, 2).join(', ')}</div>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              {selectedRishi === rishi.id && (
                <span className="text-xs px-2 py-1 rounded-full bg-orange-200 text-orange-800">
                  Active
                </span>
              )}
              {rishi.requires_upgrade && userSubscription === 'basic' && !isDemo && (
                <span className="text-xs bg-yellow-400 text-yellow-900 px-1 rounded">✨</span>
              )}
            </div>
          </button>
        ))}
          
          {/* Subscription Info */}
          {userSubscription === 'basic' && !isDemo && (
            <div className="mt-3 text-xs text-orange-700 bg-orange-100 px-3 py-2 rounded-lg">
              🆓 Free: Patanjali available • Upgrade for 4 more Rishis
            </div>
          )}
          
          {isDemo && (
            <div className="mt-3 text-xs text-blue-700 bg-blue-100 px-3 py-2 rounded-lg">
              🚀 Demo: All 5 Rishis available for exploration
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const getRishiIcon = (rishiId: string) => {
  const icons: Record<string, string> = {
    patanjali: '🧘',
    vyasa: '📚', 
    valmiki: '💖',
    adi_shankara: '✨',
    narada: '🎵'
  };
  return icons[rishiId] || '🕉️';
};

const getRishiColor = (rishiId: string) => {
  const colors: Record<string, string> = {
    patanjali: 'from-blue-500 to-indigo-600',
    vyasa: 'from-green-500 to-emerald-600',
    valmiki: 'from-pink-500 to-rose-600',
    adi_shankara: 'from-purple-500 to-violet-600',
    narada: 'from-yellow-500 to-orange-600'
  };
  return colors[rishiId] || 'from-gray-500 to-gray-600';
};
