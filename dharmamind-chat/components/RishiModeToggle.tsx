// Simple toggle component that extends your existing chat
import React, { useState } from 'react';

interface RishiModeToggleProps {
  currentMode: 'regular' | 'rishi';
  onModeChange: (mode: 'regular' | 'rishi') => void;
  userSubscription: string;
}

export const RishiModeToggle: React.FC<RishiModeToggleProps> = ({
  currentMode,
  onModeChange,
  userSubscription
}) => {
  return (
    <div className="mode-toggle">
      <div className="toggle-buttons">
        <button
          className={`mode-btn ${currentMode === 'regular' ? 'active' : ''}`}
          onClick={() => onModeChange('regular')}
        >
          🤖 Regular Chat
        </button>
        <button
          className={`mode-btn ${currentMode === 'rishi' ? 'active' : ''}`}
          onClick={() => onModeChange('rishi')}
        >
          🧘 Rishi Mode
          {userSubscription === 'free' && (
            <span className="premium-badge">✨</span>
          )}
        </button>
      </div>
      
      {currentMode === 'rishi' && userSubscription === 'free' && (
        <div className="mode-info">
          <span className="info-text">
            🆓 Free: 1 conversation/day with Patanjali
          </span>
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
    <div className="rishi-selector">
      <h3>🕉️ Choose Your Spiritual Guide:</h3>
      <div className="rishis-grid">
        {availableRishis.map(rishi => (
          <div 
            key={rishi.id}
            className={`rishi-card ${rishi.available ? 'available' : 'locked'}`}
            onClick={() => rishi.available && onSelectRishi(rishi.id)}
          >
            <div className="rishi-info">
              <h4>{rishi.name}</h4>
              <p className="sanskrit">{rishi.sanskrit}</p>
              <div className="specialization">
                {rishi.specialization.join(', ')}
              </div>
              {!rishi.available && (
                <div className="upgrade-prompt">
                  🔒 Premium Required
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
