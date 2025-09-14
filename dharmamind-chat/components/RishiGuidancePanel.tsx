// Professional Rishi Selector for Sidebar
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

interface RishiSelectorProps {
  onRishiSelect: (rishiId: string) => void;
  selectedRishi: string;
  userSubscription: string;
  isDemo?: boolean;
  availableRishis: Rishi[];
}

export const RishiSelector: React.FC<RishiSelectorProps> = ({
  onRishiSelect,
  selectedRishi,
  userSubscription,
  isDemo = false,
  availableRishis
}) => {
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

  return (
    <div className="rishi-selector">
      <h3 className="text-xs font-semibold uppercase tracking-wider mb-3 px-2 text-gray-500">
        AI Advisors
      </h3>
      
      <div className="space-y-2">
        {/* Default AI */}
        <button
          className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-sm transition-all duration-200 ${
            !selectedRishi 
              ? 'bg-white border border-green-200 text-gray-800 shadow-sm' 
              : 'bg-gray-50 hover:bg-gray-100 text-gray-600'
          }`}
          onClick={() => onRishiSelect('')}
        >
          <div className="flex items-center space-x-3">
            <span className="text-lg">🤖</span>
            <span className="font-medium">Standard AI</span>
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
            className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-sm transition-all duration-200 ${
              selectedRishi === rishi.id 
                ? 'bg-white border border-orange-200 text-orange-800 shadow-sm' 
                : 'bg-gray-50 hover:bg-gray-100 text-gray-700'
            }`}
            onClick={() => onRishiSelect(rishi.id)}
            disabled={!rishi.available}
          >
            <div className="flex items-center space-x-3">
              <span className="text-lg">{getRishiIcon(rishi.id)}</span>
              <div className="text-left">
                <div className="font-medium">{rishi.name}</div>
                <div className="text-xs text-gray-500">{rishi.specialization.slice(0, 2).join(' • ')}</div>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              {selectedRishi === rishi.id && (
                <span className="text-xs px-2 py-1 rounded-full bg-orange-100 text-orange-700">
                  Active
                </span>
              )}
              {rishi.requires_upgrade && userSubscription === 'basic' && !isDemo && (
                <span className="text-xs bg-yellow-400 text-yellow-900 px-1.5 py-0.5 rounded">Pro</span>
              )}
            </div>
          </button>
        ))}
        
        {/* Subscription Info */}
        {userSubscription === 'basic' && !isDemo && (
          <div className="mt-3 text-xs text-amber-700 bg-amber-50 px-3 py-2 rounded-lg border border-amber-200">
            🆓 Free: Patanjali • <span className="font-semibold">Upgrade for 4 more advisors</span>
          </div>
        )}
        
        {isDemo && (
          <div className="mt-3 text-xs text-blue-700 bg-blue-50 px-3 py-2 rounded-lg border border-blue-200">
            🚀 Demo: All advisors available
          </div>
        )}
      </div>
    </div>
  );
};
