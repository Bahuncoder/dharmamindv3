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
        <div className="absolute bottom-16 right-0 bg-neutral-100 border-2 border-gold-400 rounded-xl p-6 shadow-xl min-w-80 max-h-96 overflow-y-auto">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-neutral-900">🎨 Master Color System</h3>
            <button
              onClick={() => setIsOpen(false)}
              className="text-neutral-600 hover:text-gold-600 text-xl"
            >
              ×
            </button>
          </div>

          {/* Current Theme Display */}
          <div className="mb-4 p-3 bg-neutral-100 rounded-lg">
            <p className="text-sm font-medium text-neutral-900">Current Theme:</p>
            <p className="text-neutral-900 font-semibold">{currentTheme.displayName || currentTheme.name}</p>
            {currentTheme.description && (
              <p className="text-xs text-neutral-600 mt-1">{currentTheme.description}</p>
            )}
          </div>

          {/* Predefined Themes */}
          <div className="mb-6">
            <h4 className="text-sm font-semibold text-neutral-900 mb-3">🎯 Predefined Themes</h4>
            <div className="grid grid-cols-1 gap-2">
              {availableThemes.map((theme) => (
                <button
                  key={theme.name}
                  onClick={() => changeTheme(theme.name)}
                  className={`p-3 rounded-lg border-2 text-left transition-all ${
                    currentTheme.name === theme.name
                      ? 'border-gold-400 bg-gold-50'
                      : 'border-gold-400 hover:border-gold-400 hover:bg-neutral-100'
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
                      <p className="text-xs text-neutral-600">{theme.description}</p>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Custom Colors */}
          <div className="mb-4">
            <h4 className="text-sm font-semibold text-neutral-900 mb-3">🎨 Custom Colors</h4>
            <div className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-neutral-600 mb-1">
                  Primary (Main brand color)
                </label>
                <div className="flex gap-2">
                  <input
                    type="color"
                    value={customColors.brandPrimary}
                    onChange={(e) => handleCustomColorChange('brandPrimary', e.target.value)}
                    className="w-12 h-8 rounded border border-gold-400 cursor-pointer"
                  />
                  <input
                    type="text"
                    value={customColors.brandPrimary}
                    onChange={(e) => handleCustomColorChange('brandPrimary', e.target.value)}
                    className="flex-1 px-2 py-1 text-xs border border-gold-400 rounded"
                    placeholder="#6b7280"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-neutral-600 mb-1">
                  Accent (Borders & highlights)
                </label>
                <div className="flex gap-2">
                  <input
                    type="color"
                    value={customColors.brandAccent}
                    onChange={(e) => handleCustomColorChange('brandAccent', e.target.value)}
                    className="w-12 h-8 rounded border border-gold-400 cursor-pointer"
                  />
                  <input
                    type="text"
                    value={customColors.brandAccent}
                    onChange={(e) => handleCustomColorChange('brandAccent', e.target.value)}
                    className="flex-1 px-2 py-1 text-xs border border-gold-400 rounded"
                    placeholder="#10b981"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-neutral-600 mb-1">
                  Secondary (Supporting actions)
                </label>
                <div className="flex gap-2">
                  <input
                    type="color"
                    value={customColors.brandSecondary}
                    onChange={(e) => handleCustomColorChange('brandSecondary', e.target.value)}
                    className="w-12 h-8 rounded border border-gold-400 cursor-pointer"
                  />
                  <input
                    type="text"
                    value={customColors.brandSecondary}
                    onChange={(e) => handleCustomColorChange('brandSecondary', e.target.value)}
                    className="flex-1 px-2 py-1 text-xs border border-gold-400 rounded"
                    placeholder="#059669"
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
          <div className="mt-4 p-3 bg-gold-50 rounded-lg">
            <p className="text-xs text-gold-700">
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
