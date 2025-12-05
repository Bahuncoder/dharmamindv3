/**
 * 🎨 COLOR MANAGER COMPONENT
 * 
 * This demonstrates how easy it is to change colors system-wide.
 * Change brand colors here and watch the entire app update instantly!
 */

import React, { useState } from 'react';
import { useColors } from '../contexts/ColorContext';

const ColorManager: React.FC = () => {
  const { currentTheme, availableThemes, changeTheme, updateBrandColors, resetToDefault } = useColors();
  const [isOpen, setIsOpen] = useState(false);
  const [customColors, setCustomColors] = useState({
    brandPrimary: currentTheme.colors.brandPrimary,
    brandAccent: currentTheme.colors.brandAccent,
    brandSecondary: currentTheme.colors.brandSecondary,
  });

  const handleCustomColorChange = (colorType: keyof typeof customColors, value: string) => {
    const newColors = { ...customColors, [colorType]: value };
    setCustomColors(newColors);
    updateBrandColors({ [colorType]: value });
  };

  return (
    <div className="fixed bottom-4 right-4 z-50">
      {/* Toggle Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="btn-primary p-3 rounded-full shadow-lg hover:shadow-xl transition-all duration-300"
        title="Color Manager"
      >
        🎨
      </button>

      {/* Color Panel */}
      {isOpen && (
        <div className="absolute bottom-16 right-0 bg-white border-2 border-brand-accent rounded-xl p-6 shadow-xl min-w-80 max-h-96 overflow-y-auto">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-primary">🎨 Master Color System</h3>
            <button
              onClick={() => setIsOpen(false)}
              className="text-secondary hover:text-primary text-xl"
            >
              ×
            </button>
          </div>

          {/* Current Theme Display */}
          <div className="mb-4 p-3 bg-section-light rounded-lg">
            <p className="text-sm font-medium text-primary">Current Theme:</p>
            <p className="text-brand-primary font-semibold">{currentTheme.displayName || currentTheme.name}</p>
            {currentTheme.description && (
              <p className="text-xs text-secondary mt-1">{currentTheme.description}</p>
            )}
          </div>

          {/* Predefined Themes */}
          <div className="mb-6">
            <h4 className="text-sm font-semibold text-primary mb-3">🎯 Predefined Themes</h4>
            <div className="grid grid-cols-1 gap-2">
              {availableThemes.map((theme) => (
                <button
                  key={theme.name}
                  onClick={() => changeTheme(theme.name)}
                  className={`p-3 rounded-lg border-2 text-left transition-all ${
                    currentTheme.name === theme.name
                      ? 'border-brand-accent bg-blue-50'
                      : 'border-brand-accent hover:border-brand-accent hover:bg-section-light'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className="flex gap-1">
                      <div 
                        className="w-4 h-4 rounded-full border"
                        style={{ backgroundColor: theme.colors.brandPrimary }}
                      />
                      <div 
                        className="w-4 h-4 rounded-full border"
                        style={{ backgroundColor: theme.colors.brandAccent }}
                      />
                      <div 
                        className="w-4 h-4 rounded-full border"
                        style={{ backgroundColor: theme.colors.brandSecondary }}
                      />
                    </div>
                    <div>
                      <p className="font-medium text-sm">{theme.displayName}</p>
                      <p className="text-xs text-secondary">{theme.description}</p>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Custom Colors */}
          <div className="mb-4">
            <h4 className="text-sm font-semibold text-primary mb-3">🎨 Custom Colors</h4>
            <div className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-secondary mb-1">
                  Primary (Main brand color)
                </label>
                <div className="flex gap-2">
                  <input
                    type="color"
                    value={customColors.brandPrimary}
                    onChange={(e) => handleCustomColorChange('brandPrimary', e.target.value)}
                    className="w-12 h-8 rounded border border-brand-accent cursor-pointer"
                  />
                  <input
                    type="text"
                    value={customColors.brandPrimary}
                    onChange={(e) => handleCustomColorChange('brandPrimary', e.target.value)}
                    className="flex-1 px-2 py-1 text-xs border border-brand-accent rounded"
                    placeholder="#6b7280"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-secondary mb-1">
                  Accent (Borders & highlights)
                </label>
                <div className="flex gap-2">
                  <input
                    type="color"
                    value={customColors.brandAccent}
                    onChange={(e) => handleCustomColorChange('brandAccent', e.target.value)}
                    className="w-12 h-8 rounded border border-brand-accent cursor-pointer"
                  />
                  <input
                    type="text"
                    value={customColors.brandAccent}
                    onChange={(e) => handleCustomColorChange('brandAccent', e.target.value)}
                    className="flex-1 px-2 py-1 text-xs border border-brand-accent rounded"
                    placeholder="#10b981"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-secondary mb-1">
                  Secondary (Supporting actions)
                </label>
                <div className="flex gap-2">
                  <input
                    type="color"
                    value={customColors.brandSecondary}
                    onChange={(e) => handleCustomColorChange('brandSecondary', e.target.value)}
                    className="w-12 h-8 rounded border border-brand-accent cursor-pointer"
                  />
                  <input
                    type="text"
                    value={customColors.brandSecondary}
                    onChange={(e) => handleCustomColorChange('brandSecondary', e.target.value)}
                    className="flex-1 px-2 py-1 text-xs border border-brand-accent rounded"
                    placeholder="#8b5cf6"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-2">
            <button
              onClick={resetToDefault}
              className="btn-outline btn-sm flex-1"
            >
              Reset Default
            </button>
            <button
              onClick={() => setIsOpen(false)}
              className="btn-primary btn-sm flex-1"
            >
              Close
            </button>
          </div>

          {/* Info */}
          <div className="mt-4 p-3 bg-blue-50 rounded-lg">
            <p className="text-xs text-blue-700">
              <strong>💡 How it works:</strong> These colors control the entire application. 
              Change them here and watch buttons, cards, borders, and all UI elements 
              update instantly across every page!
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default ColorManager;
