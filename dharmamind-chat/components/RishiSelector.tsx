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
  const [isOpen, setIsOpen] = useState(false);

  const getRishiIcon = (rishiId: string) => {
    const icons: Record<string, string> = {
      atri: '🧘',           // Master of tapasya (meditation/austerity)
      bhrigu: '⭐',         // Great astrologer
      vashishta: '�',      // Guru of Lord Rama, symbol of wisdom
      vishwamitra: '🕉️',   // Creator of Gayatri Mantra
      gautama: '🙏',        // Known for deep meditation and dharma
      jamadagni: '⚡',      // Father of Parashurama, symbol of discipline
      kashyapa: '�'        // Father of beings, cosmic creator
    };
    return icons[rishiId] || '🕉️';
  };

  const selectedRishiData = availableRishis.find(r => r.id === selectedRishi);

  return (
    <div className="rishi-selector">
      <h3 className="text-xs font-semibold uppercase tracking-wider mb-3 px-2" style={{ color: 'var(--color-text-secondary, #6b7280)' }}>
        Rishi
      </h3>
      
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-3 py-3 rounded-lg text-sm transition-all duration-200"
        style={{
          backgroundColor: 'var(--color-background-secondary, #ffffff)',
          border: '1px solid var(--color-border-primary, #10b981)',
          color: 'var(--color-text-primary, #1f2937)'
        }}
      >
        <div className="flex items-center space-x-3">
          <span className="text-lg">
            {selectedRishiData ? getRishiIcon(selectedRishi) : '🕉️'}
          </span>
          <div className="text-left">
            <div className="font-medium">
              {selectedRishiData ? selectedRishiData.name : 'Select Rishi Guide'}
            </div>
            {selectedRishiData && (
              <div className="text-xs" style={{ color: 'var(--color-text-secondary, #6b7280)' }}>
                {selectedRishiData.specialization.slice(0, 2).join(' • ')}
              </div>
            )}
          </div>
        </div>
        <div className="flex items-center space-x-2">
          {selectedRishiData && (
            <span className="text-xs px-2 py-1 rounded-full" style={{
              backgroundColor: 'var(--color-success-light, #d1fae5)',
              color: 'var(--color-success-dark, #065f46)'
            }}>
              ✓ Active
            </span>
          )}
          <svg 
            className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {isOpen && (
        <div className="mt-2 space-y-1 rounded-lg shadow-lg overflow-hidden" style={{
          backgroundColor: 'var(--color-background-secondary, #ffffff)',
          border: '1px solid var(--color-border-primary, #10b981)'
        }}>
          {availableRishis.map((rishi) => (
            <button
              key={rishi.id}
              className="w-full flex items-center justify-between px-3 py-2.5 text-sm transition-all duration-200"
              style={{
                backgroundColor: selectedRishi === rishi.id ? 'var(--color-success-light, #d1fae5)' : 'transparent',
                color: selectedRishi === rishi.id ? 'var(--color-success-dark, #065f46)' : 'var(--color-text-primary, #1f2937)'
              }}
              onClick={() => {
                onRishiSelect(rishi.id);
                setIsOpen(false);
              }}
            >
              <div className="flex items-center space-x-3">
                <span className="text-lg">{getRishiIcon(rishi.id)}</span>
                <div className="text-left">
                  <div className="font-medium">{rishi.name}</div>
                  <div className="text-xs" style={{ color: 'var(--color-text-secondary, #6b7280)' }}>
                    {rishi.specialization.slice(0, 2).join(' • ')}
                  </div>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                {selectedRishi === rishi.id && (
                  <span className="text-xs px-2 py-1 rounded-full" style={{
                    backgroundColor: 'var(--color-success-light, #d1fae5)',
                    color: 'var(--color-success-dark, #065f46)'
                  }}>
                    ✓
                  </span>
                )}
                {rishi.requires_upgrade && userSubscription === 'basic' && !isDemo && (
                  <span className="text-xs px-1.5 py-0.5 rounded font-medium" style={{
                    backgroundColor: 'var(--color-accent, #f59e0b)',
                    color: 'white'
                  }}>
                    Pro
                  </span>
                )}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};
