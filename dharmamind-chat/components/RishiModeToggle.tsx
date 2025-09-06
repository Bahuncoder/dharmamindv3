// Simple toggle component that extends your existing chat
import React, { useState } from 'react';

interface RishiModeToggleProps {
  currentMode: 'regular' | 'rishi';
  onModeChange: (mode: 'regular' | 'rishi') => void;
  userSubscription: string;
  isDemo?: boolean;
}

export const RishiModeToggle: React.FC<RishiModeToggleProps> = ({
  currentMode,
  onModeChange,
  userSubscription,
  isDemo = false
}) => {
  return (
    <div className="flex items-center space-x-3">
      <div className="flex items-center space-x-2 p-1 rounded-xl bg-gray-100 dark:bg-gray-800">
        <button
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
            currentMode === 'regular' 
              ? 'bg-white shadow-sm text-gray-900 dark:bg-gray-700 dark:text-white' 
              : 'text-gray-600 hover:text-gray-800 dark:text-gray-300 dark:hover:text-white'
          }`}
          onClick={() => onModeChange('regular')}
        >
          🤖 Regular Chat
        </button>
        <button
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 flex items-center space-x-2 ${
            currentMode === 'rishi' 
              ? 'bg-gradient-to-r from-orange-500 to-yellow-600 text-white shadow-sm' 
              : 'text-gray-600 hover:text-gray-800 dark:text-gray-300 dark:hover:text-white'
          }`}
          onClick={() => onModeChange('rishi')}
        >
          <span>🧘 Rishi Mode</span>
          {userSubscription === 'basic' && currentMode !== 'rishi' && (
            <span className="text-xs bg-yellow-400 text-yellow-900 px-1 rounded">✨</span>
          )}
        </button>
      </div>
      
      {currentMode === 'rishi' && userSubscription === 'basic' && !isDemo && (
        <div className="text-xs text-orange-700 bg-orange-100 px-3 py-1 rounded-full">
          🆓 Free: Patanjali available
        </div>
      )}

      {currentMode === 'rishi' && isDemo && (
        <div className="text-xs text-blue-700 bg-blue-100 px-3 py-1 rounded-full">
          🚀 Demo: All 3 Rishis available
        </div>
      )}
    </div>
  );
};

// Simple Rishi selector that reuses your existing components
export const RishiSelector: React.FC<{
  onSelectRishi: (rishi: string) => void;
  availableRishis: any[];
}> = ({ onSelectRishi, availableRishis }) => {
  return (
    <div className="space-y-4">
      <div className="text-center text-gray-600 mb-6">
        <p>Choose your spiritual guide for personalized wisdom and guidance</p>
      </div>
      
      <div className="grid gap-4">
        {availableRishis.map(rishi => (
          <div 
            key={rishi.id}
            className={`p-6 border rounded-xl transition-all duration-200 cursor-pointer ${
              rishi.available 
                ? 'border-green-200 bg-green-50 hover:border-green-300 hover:shadow-md' 
                : 'border-gray-200 bg-gray-50 opacity-60 cursor-not-allowed'
            }`}
            onClick={() => rishi.available && onSelectRishi(rishi.id)}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-3 mb-2">
                  <h3 className="text-xl font-bold text-gray-900">{rishi.name}</h3>
                  {!rishi.available && (
                    <span className="bg-yellow-100 text-yellow-800 text-xs px-2 py-1 rounded-full font-medium">
                      🔒 Premium
                    </span>
                  )}
                </div>
                <p className="text-sm text-orange-600 font-medium mb-2">{rishi.sanskrit}</p>
                <div className="text-sm text-gray-600 mb-3">
                  <strong>Specializes in:</strong> {rishi.specialization.join(', ')}
                </div>
                <p className="text-sm text-gray-700 italic leading-relaxed">
                  "{rishi.greeting}"
                </p>
              </div>
            </div>
            
            {rishi.available && (
              <div className="mt-4 pt-4 border-t border-green-200">
                <button className="w-full bg-gradient-to-r from-green-500 to-emerald-600 text-white font-medium py-2 px-4 rounded-lg hover:from-green-600 hover:to-emerald-700 transition-all duration-200">
                  Select {rishi.name}
                </button>
              </div>
            )}
            
            {!rishi.available && (
              <div className="mt-4 pt-4 border-t border-gray-200">
                <button className="w-full bg-gray-300 text-gray-600 font-medium py-2 px-4 rounded-lg cursor-not-allowed">
                  Requires Premium Subscription
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
      
      <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="flex items-start space-x-3">
          <span className="text-blue-500 text-lg">ℹ️</span>
          <div className="text-sm text-blue-800">
            <p className="font-medium mb-1">How Rishi Mode Works:</p>
            <p>Each Rishi offers guidance based on their specific teachings and spiritual expertise. Choose the one whose wisdom resonates with your current question or life situation.</p>
          </div>
        </div>
      </div>
    </div>
  );
};
