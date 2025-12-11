/**
 * Color Manager Component
 * 
 * This component provides an interface for changing colors throughout the app.
 * Perfect for settings page or admin panel.
 */

import React, { useState } from 'react';
import { useColors } from '../contexts/ColorContext';
import Button from './Button';

const ColorManager: React.FC = () => {
  const { currentTheme, availableThemes, changeTheme, updateColors } = useColors();
  const [customColors, setCustomColors] = useState({
    primaryStart: currentTheme.colors.primaryStart,
    primaryEnd: currentTheme.colors.primaryEnd,
  });
  const [isPreviewMode, setIsPreviewMode] = useState(false);
  const [savedSuccess, setSavedSuccess] = useState(false);

  const handleThemeChange = (themeName: string) => {
    changeTheme(themeName);
    const selectedTheme = availableThemes.find(t => t.name === themeName);
    if (selectedTheme) {
      setCustomColors({
        primaryStart: selectedTheme.colors.primaryStart,
        primaryEnd: selectedTheme.colors.primaryEnd,
      });
    }
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
<<<<<<< HEAD
      primaryStart: '#10b981',
=======
      primaryStart: '#d97706',
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
      primaryEnd: '#059669',
    });
  };

  const handleSave = () => {
    setSavedSuccess(true);
    setTimeout(() => setSavedSuccess(false), 2000);
  };

  const getThemeDisplayName = (name: string) => {
    return name.split('-').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  };

  return (
    <div className="color-manager card-elevated p-8 max-w-2xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h3 className="text-3xl font-black text-primary mb-2">
            🎨 Color Theme Manager
          </h3>
          <p className="text-secondary font-semibold">
            Customize your DharmaMind experience with beautiful color themes
          </p>
        </div>
        <button
          onClick={() => setIsPreviewMode(!isPreviewMode)}
          className={`px-4 py-2 rounded-lg font-bold text-sm transition-all ${
            isPreviewMode 
              ? 'bg-primary text-white' 
              : 'bg-neutral-100 text-secondary hover:bg-neutral-200'
          }`}
        >
          {isPreviewMode ? '👁️ Exit Preview' : '👀 Preview Mode'}
        </button>
      </div>
      
      {/* Theme Selection Grid */}
      <div className="mb-8">
        <label className="block text-xl font-black text-primary mb-4">
          ✨ Predefined Themes
        </label>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {availableThemes.map((theme) => (
            <button
              key={theme.name}
              onClick={() => handleThemeChange(theme.name)}
              className={`group flex items-center p-5 rounded-xl border-2 transition-all duration-300 hover:shadow-lg ${
                currentTheme.name === theme.name
                  ? 'border-primary bg-primary-gradient bg-opacity-10 shadow-md transform scale-105'
                  : 'border-border-light hover:border-primary hover:shadow-md'
              }`}
            >
              <div className="flex items-center space-x-4 w-full">
                <div
                  className="w-12 h-12 rounded-xl shadow-md group-hover:scale-110 transition-transform"
                  style={{
                    background: `linear-gradient(135deg, ${theme.colors.primaryStart}, ${theme.colors.primaryEnd})`
                  }}
                />
                <div className="flex-1 text-left">
                  <div className="font-bold text-primary text-lg">
                    {getThemeDisplayName(theme.name)}
                  </div>
                  <div className="text-sm text-secondary font-medium">
                    {theme.colors.primaryStart} → {theme.colors.primaryEnd}
                  </div>
                </div>
                {currentTheme.name === theme.name && (
                  <div className="text-2xl animate-pulse">✓</div>
                )}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Custom Colors Section */}
      <div className="mb-8">
        <label className="block text-xl font-black text-primary mb-4">
          🎯 Custom Colors
        </label>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-3">
            <label className="block text-sm font-bold text-secondary uppercase tracking-wide">
              Primary Start Color
            </label>
            <div className="flex items-center space-x-3">
              <div className="relative">
                <input
                  type="color"
                  value={customColors.primaryStart}
                  onChange={(e) => handleCustomColorChange('primaryStart', e.target.value)}
                  className="w-16 h-16 rounded-xl border-2 border-border-light cursor-pointer hover:border-primary transition-colors shadow-md"
                />
                <div className="absolute -top-1 -right-1 w-5 h-5 bg-primary rounded-full text-white text-xs flex items-center justify-center font-bold">
                  1
                </div>
              </div>
              <input
                type="text"
                value={customColors.primaryStart}
                onChange={(e) => handleCustomColorChange('primaryStart', e.target.value)}
                className="flex-1 px-4 py-3 border-2 border-border-light rounded-xl text-sm bg-primary-bg text-primary font-bold focus:border-primary focus:ring-2 focus:ring-focus transition-all"
<<<<<<< HEAD
                placeholder="#10b981"
=======
                placeholder="#d97706"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
              />
            </div>
          </div>

          <div className="space-y-3">
            <label className="block text-sm font-bold text-secondary uppercase tracking-wide">
              Primary End Color
            </label>
            <div className="flex items-center space-x-3">
              <div className="relative">
                <input
                  type="color"
                  value={customColors.primaryEnd}
                  onChange={(e) => handleCustomColorChange('primaryEnd', e.target.value)}
                  className="w-16 h-16 rounded-xl border-2 border-border-light cursor-pointer hover:border-primary transition-colors shadow-md"
                />
                <div className="absolute -top-1 -right-1 w-5 h-5 bg-primary rounded-full text-white text-xs flex items-center justify-center font-bold">
                  2
                </div>
              </div>
              <input
                type="text"
                value={customColors.primaryEnd}
                onChange={(e) => handleCustomColorChange('primaryEnd', e.target.value)}
                className="flex-1 px-4 py-3 border-2 border-border-light rounded-xl text-sm bg-primary-bg text-primary font-bold focus:border-primary focus:ring-2 focus:ring-focus transition-all"
                placeholder="#059669"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Live Preview */}
      <div className="mb-8">
        <label className="block text-xl font-black text-primary mb-4">
          👀 Live Preview
        </label>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <Button variant="primary" fullWidth>
              <span className="mr-2">🚀</span>
              Primary Button
            </Button>
            <Button variant="outline" fullWidth>
              <span className="mr-2">📞</span>
              Outline Button
            </Button>
            <Button variant="enterprise" fullWidth>
              <span className="mr-2">💼</span>
              Enterprise Button
            </Button>
          </div>
          <div className="space-y-4">
            <div className="bg-primary-gradient p-6 rounded-xl shadow-lg">
              <div className="text-white font-bold text-lg">🌸 DharmaMind</div>
              <div className="text-white opacity-90 text-sm">Gradient Background</div>
            </div>
            <div className="flex items-center space-x-4 p-4 bg-primary-bg rounded-xl border-2 border-border-light">
              <div className="w-12 h-12 bg-primary-gradient rounded-full flex items-center justify-center text-white font-bold text-lg">
                DM
              </div>
              <div>
                <div className="font-bold text-primary">Logo Preview</div>
                <div className="text-sm text-secondary">Brand Identity</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex flex-col md:flex-row space-y-3 md:space-y-0 md:space-x-4">
        <Button
          variant="outline"
          onClick={resetToDefault}
          className="flex-1"
          icon={<span>🔄</span>}
          iconPosition="left"
        >
          Reset to Default
        </Button>
        <Button
          variant="primary"
          onClick={handleSave}
          className="flex-1"
          icon={savedSuccess ? <span>✅</span> : <span>💾</span>}
          iconPosition="left"
        >
          {savedSuccess ? 'Saved Successfully!' : 'Save Settings'}
        </Button>
      </div>

      {/* Current Theme Info */}
      <div className="mt-8 p-6 bg-neutral-50 rounded-xl">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-sm font-bold text-secondary uppercase tracking-wide">
              Current Theme
            </div>
            <div className="text-lg font-black text-primary">
              {getThemeDisplayName(currentTheme.name)}
            </div>
          </div>
          <div>
            <div className="text-sm font-bold text-secondary uppercase tracking-wide">
              Start Color
            </div>
            <div className="flex items-center justify-center space-x-2">
              <div 
                className="w-6 h-6 rounded-full shadow-sm"
                style={{ backgroundColor: currentTheme.colors.primaryStart }}
              />
              <span className="text-sm font-bold text-primary">
                {currentTheme.colors.primaryStart}
              </span>
            </div>
          </div>
          <div>
            <div className="text-sm font-bold text-secondary uppercase tracking-wide">
              End Color
            </div>
            <div className="flex items-center justify-center space-x-2">
              <div 
                className="w-6 h-6 rounded-full shadow-sm"
                style={{ backgroundColor: currentTheme.colors.primaryEnd }}
              />
              <span className="text-sm font-bold text-primary">
                {currentTheme.colors.primaryEnd}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ColorManager;
