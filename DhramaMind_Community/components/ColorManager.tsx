/**
 * Color Manager Component
 * 
 * This component provides an interface for changing colors throughout the app.
 * Perfect for settings page or admin panel.
 */

import React, { useState } from 'react';
import { useColors } from '../contexts/ColorContext';

const ColorManager: React.FC = () => {
  const { currentTheme, availableThemes, changeTheme, updateColors } = useColors();
  const [customColors, setCustomColors] = useState({
    primaryStart: currentTheme.colors.primaryStart,
    primaryEnd: currentTheme.colors.primaryEnd,
  });

  const handleThemeChange = (themeName: string) => {
    changeTheme(themeName);
  };

  const handleCustomColorChange = (colorKey: string, value: string) => {
    const newColors = { ...customColors, [colorKey]: value };
    setCustomColors(newColors);
    
    // Update colors in real-time
    updateColors(newColors);
  };

  const resetToDefault = () => {
    changeTheme('dharma-default');
    setCustomColors({
      primaryStart: '#d97706',
      primaryEnd: '#059669',
    });
  };

  return (
    <div className="color-manager p-6 bg-white dark:bg-neutral-800 rounded-lg shadow-lg max-w-md mx-auto">
      <h3 className="text-xl font-semibold mb-6 text-neutral-900 dark:text-neutral-100">
        Color Theme Manager
      </h3>
      
      {/* Predefined Themes */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-3">
          Predefined Themes
        </label>
        <div className="grid grid-cols-1 gap-2">
          {availableThemes.map((theme) => (
            <button
              key={theme.name}
              onClick={() => handleThemeChange(theme.name)}
              className={`flex items-center p-3 rounded-lg border-2 transition-all ${
                currentTheme.name === theme.name
                  ? 'border-primary bg-primary bg-opacity-10'
                  : 'border-neutral-200 dark:border-neutral-600 hover:border-primary'
              }`}
            >
              <div
                className="w-6 h-6 rounded-full mr-3"
                style={{
                  background: `linear-gradient(45deg, ${theme.colors.primaryStart}, ${theme.colors.primaryEnd})`
                }}
              />
              <span className="text-sm font-medium capitalize text-neutral-700 dark:text-neutral-300">
                {theme.name.replace('-', ' ')}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Custom Colors */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-3">
          Custom Colors
        </label>
        
        <div className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-neutral-600 dark:text-neutral-400 mb-1">
              Primary Start Color
            </label>
            <div className="flex items-center space-x-2">
              <input
                type="color"
                value={customColors.primaryStart}
                onChange={(e) => handleCustomColorChange('primaryStart', e.target.value)}
                className="w-12 h-10 rounded border-2 border-neutral-200 dark:border-neutral-600 cursor-pointer"
              />
              <input
                type="text"
                value={customColors.primaryStart}
                onChange={(e) => handleCustomColorChange('primaryStart', e.target.value)}
                className="flex-1 px-3 py-2 border border-neutral-200 dark:border-neutral-600 rounded-md text-sm bg-white dark:bg-neutral-700 text-neutral-900 dark:text-neutral-100"
                placeholder="#d97706"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-neutral-600 dark:text-neutral-400 mb-1">
              Primary End Color
            </label>
            <div className="flex items-center space-x-2">
              <input
                type="color"
                value={customColors.primaryEnd}
                onChange={(e) => handleCustomColorChange('primaryEnd', e.target.value)}
                className="w-12 h-10 rounded border-2 border-neutral-200 dark:border-neutral-600 cursor-pointer"
              />
              <input
                type="text"
                value={customColors.primaryEnd}
                onChange={(e) => handleCustomColorChange('primaryEnd', e.target.value)}
                className="flex-1 px-3 py-2 border border-neutral-200 dark:border-neutral-600 rounded-md text-sm bg-white dark:bg-neutral-700 text-neutral-900 dark:text-neutral-100"
                placeholder="#059669"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Preview */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-3">
          Preview
        </label>
        <div className="space-y-3">
          <button className="btn-primary w-full">
            Primary Button
          </button>
          <button className="btn-contact w-full">
            Contact Sales Button
          </button>
          <button className="btn-enterprise w-full">
            Enterprise Button
          </button>
          <div className="bg-primary-gradient p-4 rounded-lg">
            <span className="text-white font-medium">Gradient Background</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="avatar-primary">DM</div>
            <span className="text-neutral-700 dark:text-neutral-300">Logo Preview</span>
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex space-x-2">
        <button
          onClick={resetToDefault}
          className="flex-1 px-4 py-2 border border-neutral-200 dark:border-neutral-600 rounded-md text-sm font-medium text-neutral-700 dark:text-neutral-300 hover:bg-neutral-50 dark:hover:bg-neutral-700 transition-colors"
        >
          Reset to Default
        </button>
        <button className="btn-primary flex-1">
          Save Settings
        </button>
      </div>

      {/* Current Theme Info */}
      <div className="mt-4 p-3 bg-neutral-50 dark:bg-neutral-700 rounded-lg">
        <div className="text-xs text-neutral-600 dark:text-neutral-400">
          <strong>Current Theme:</strong> {currentTheme.name}
        </div>
        <div className="text-xs text-neutral-600 dark:text-neutral-400 mt-1">
          <strong>Colors:</strong> {currentTheme.colors.primaryStart} → {currentTheme.colors.primaryEnd}
        </div>
      </div>
    </div>
  );
};

export default ColorManager;
