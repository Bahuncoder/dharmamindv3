/**
 * Color Theme Context and Hook for DharmaMind
 * 
 * This provides programmatic access to change colors throughout the app.
 * When colors are updated, all components using this system will automatically update.
 */

import React, { createContext, useContext, useEffect, useState } from 'react';

interface ColorTheme {
  name: string;
  colors: {
    primaryStart: string;
    primaryEnd: string;
    primaryHoverStart: string;
    primaryHoverEnd: string;
    primary: string;
    primaryHover: string;
    borderPrimary: string;      // Gold border color
    borderSecondary: string;    // Secondary border color
    textPrimary: string;        // Text colors
    textSecondary: string;      // Secondary text color
    background: string;         // Background color
    backgroundSecondary: string; // Secondary background
    // Error colors
    error: string;
    errorBackground: string;
    errorText: string;
    // Success colors
    success: string;
    successBackground: string;
    successText: string;
    // Warning colors
    warning: string;
    warningBackground: string;
    warningText: string;
    // Shadow colors
    shadowLight: string;
    shadowMedium: string;
    shadowStrong: string;
    // Surface colors
    surfaceLight: string;
    surfaceDark: string;
    // Additional utility colors
    accent: string;
    wisdom: string;
    lotus: string;
  };
}

interface ColorContextType {
  currentTheme: ColorTheme;
  availableThemes: ColorTheme[];
  changeTheme: (themeName: string) => void;
  updateColors: (colors: Partial<ColorTheme['colors']>) => void;
}

// Predefined themes
const themes: ColorTheme[] = [
  {
    name: 'light-gray-gold', // Gold theme
    colors: {
      primaryStart: '#f8fafc',        // Very light gray
      primaryEnd: '#f1f5f9',          // Light gray
      primaryHoverStart: '#e2e8f0',   // Medium light gray
      primaryHoverEnd: '#cbd5e1',     // Medium gray
      primary: '#f8fafc',             // Very light gray backgrounds
      primaryHover: '#e2e8f0',        // Medium light gray on hover
      borderPrimary: '#d4a854',       // Gold for borders
      borderSecondary: '#e4b864',     // Light gold for secondary borders
      textPrimary: '#1f2937',         // Dark gray text
      textSecondary: '#6b7280',       // Medium gray text
      background: '#f1f5f9',          // Light gray background for user messages
      backgroundSecondary: '#ffffff', // White background for AI messages
      // Error colors
      error: '#dc2626',               // Red
      errorBackground: '#fef2f2',     // Light red background
      errorText: '#991b1b',           // Dark red text
      // Success colors
      success: '#059669',             // Dark emerald
      successBackground: '#f0fdf4',   // Light green background
      successText: '#166534',         // Dark green text
      // Warning colors
      warning: '#d97706',             // Orange
      warningBackground: '#fffbeb',   // Light yellow background
      warningText: '#92400e',         // Dark orange text
      // Shadow colors
      shadowLight: 'rgba(0, 0, 0, 0.1)',
      shadowMedium: 'rgba(0, 0, 0, 0.2)',
      shadowStrong: 'rgba(0, 0, 0, 0.4)',
      // Surface colors
      surfaceLight: '#ffffff',
      surfaceDark: '#f8fafc',
      // Additional utility colors
      accent: '#6b7280',              // Gray accent
      wisdom: '#4facfe',              // Blue wisdom
      lotus: '#fa709a',               // Pink lotus
    }
  }
];

const ColorContext = createContext<ColorContextType | undefined>(undefined);

export const ColorProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [currentTheme, setCurrentTheme] = useState<ColorTheme>(themes[0]);

  // Helper function to convert hex to RGB
  const hexToRgb = (hex: string) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16)
    } : null;
  };

  // Apply colors to CSS custom properties
  const applyColors = (colors: ColorTheme['colors']) => {
    const root = document.documentElement;
    root.style.setProperty('--color-primary-start', colors.primaryStart);
    root.style.setProperty('--color-primary-end', colors.primaryEnd);
    root.style.setProperty('--color-primary-hover-start', colors.primaryHoverStart);
    root.style.setProperty('--color-primary-hover-end', colors.primaryHoverEnd);
    root.style.setProperty('--color-primary', colors.primary);
    root.style.setProperty('--color-primary-hover', colors.primaryHover);

    // Border colors - Emerald green for borders
    root.style.setProperty('--color-border-primary', colors.borderPrimary);
    root.style.setProperty('--color-border-secondary', colors.borderSecondary);

    // Text colors
    root.style.setProperty('--color-text-primary', colors.textPrimary);
    root.style.setProperty('--color-text-secondary', colors.textSecondary);

    // Background colors
    root.style.setProperty('--color-background', colors.background);
    root.style.setProperty('--color-background-secondary', colors.backgroundSecondary);

    // Error colors
    root.style.setProperty('--color-error', colors.error);
    root.style.setProperty('--color-error-background', colors.errorBackground);
    root.style.setProperty('--color-error-text', colors.errorText);

    // Success colors
    root.style.setProperty('--color-success', colors.success);
    root.style.setProperty('--color-success-background', colors.successBackground);
    root.style.setProperty('--color-success-text', colors.successText);

    // Warning colors
    root.style.setProperty('--color-warning', colors.warning);
    root.style.setProperty('--color-warning-background', colors.warningBackground);
    root.style.setProperty('--color-warning-text', colors.warningText);

    // Shadow colors
    root.style.setProperty('--color-shadow-light', colors.shadowLight);
    root.style.setProperty('--color-shadow-medium', colors.shadowMedium);
    root.style.setProperty('--color-shadow-strong', colors.shadowStrong);

    // Surface colors
    root.style.setProperty('--color-surface-light', colors.surfaceLight);
    root.style.setProperty('--color-surface-dark', colors.surfaceDark);

    // Additional utility colors
    root.style.setProperty('--color-accent', colors.accent);
    root.style.setProperty('--color-wisdom', colors.wisdom);
    root.style.setProperty('--color-lotus', colors.lotus);

    // Update focus ring color to use gold border color
    root.style.setProperty('--color-focus', colors.borderPrimary);
    const borderRgb = hexToRgb(colors.borderPrimary);
    if (borderRgb) {
      root.style.setProperty('--color-focus-ring', `rgba(${borderRgb.r}, ${borderRgb.g}, ${borderRgb.b}, 0.2)`);
    }
  };

  // Change to a predefined theme
  const changeTheme = (themeName: string) => {
    const theme = themes.find(t => t.name === themeName);
    if (theme) {
      setCurrentTheme(theme);
      applyColors(theme.colors);
    }
  };

  // Update specific colors while keeping the rest
  const updateColors = (newColors: Partial<ColorTheme['colors']>) => {
    const updatedTheme = {
      ...currentTheme,
      colors: { ...currentTheme.colors, ...newColors }
    };
    setCurrentTheme(updatedTheme);
    applyColors(updatedTheme.colors);
  };

  // Apply initial theme on mount
  useEffect(() => {
    applyColors(currentTheme.colors);
  }, []);

  const contextValue: ColorContextType = {
    currentTheme,
    availableThemes: themes,
    changeTheme,
    updateColors,
  };

  return (
    <ColorContext.Provider value={contextValue}>
      {children}
    </ColorContext.Provider>
  );
};

// Custom hook to use the color context
export const useColor = (): ColorContextType => {
  const context = useContext(ColorContext);
  if (context === undefined) {
    throw new Error('useColor must be used within a ColorProvider');
  }
  return context;
};

// Named export for the context itself (if needed)
export { ColorContext };
