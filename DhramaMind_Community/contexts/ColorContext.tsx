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
    // Add more color properties as needed
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
    name: 'dharma-default',
    colors: {
      primaryStart: '#d97706', // amber-600
      primaryEnd: '#059669',   // emerald-600
      primaryHoverStart: '#b45309', // amber-700
      primaryHoverEnd: '#047857',   // emerald-700
      primary: '#d97706',
      primaryHover: '#b45309',
    }
  },
  {
    name: 'ocean-blue',
    colors: {
      primaryStart: '#0ea5e9', // sky-500
      primaryEnd: '#0284c7',   // sky-600
      primaryHoverStart: '#0369a1', // sky-700
      primaryHoverEnd: '#075985',   // sky-800
      primary: '#0ea5e9',
      primaryHover: '#0369a1',
    }
  },
  {
    name: 'forest-green',
    colors: {
      primaryStart: '#059669', // emerald-600
      primaryEnd: '#047857',   // emerald-700
      primaryHoverStart: '#065f46', // emerald-800
      primaryHoverEnd: '#064e3b',   // emerald-900
      primary: '#059669',
      primaryHover: '#065f46',
    }
  },
  {
    name: 'royal-purple',
    colors: {
      primaryStart: '#8b5cf6', // violet-500
      primaryEnd: '#7c3aed',   // violet-600
      primaryHoverStart: '#6d28d9', // violet-700
      primaryHoverEnd: '#5b21b6',   // violet-800
      primary: '#8b5cf6',
      primaryHover: '#6d28d9',
    }
  },
  {
    name: 'sunset-orange',
    colors: {
      primaryStart: '#f97316', // orange-500
      primaryEnd: '#ea580c',   // orange-600
      primaryHoverStart: '#c2410c', // orange-700
      primaryHoverEnd: '#9a3412',   // orange-800
      primary: '#f97316',
      primaryHover: '#c2410c',
    }
  }
];

const ColorContext = createContext<ColorContextType | undefined>(undefined);

export const ColorProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [currentTheme, setCurrentTheme] = useState<ColorTheme>(themes[0]);

  // Apply colors to CSS custom properties
  const applyColors = (colors: ColorTheme['colors']) => {
    const root = document.documentElement;
    root.style.setProperty('--color-primary-start', colors.primaryStart);
    root.style.setProperty('--color-primary-end', colors.primaryEnd);
    root.style.setProperty('--color-primary-hover-start', colors.primaryHoverStart);
    root.style.setProperty('--color-primary-hover-end', colors.primaryHoverEnd);
    root.style.setProperty('--color-primary', colors.primary);
    root.style.setProperty('--color-primary-hover', colors.primaryHover);
    
    // Update focus ring color to match primary
    const primaryRgb = hexToRgb(colors.primary);
    if (primaryRgb) {
      root.style.setProperty('--color-focus', colors.primary);
      root.style.setProperty('--color-focus-ring', `rgba(${primaryRgb.r}, ${primaryRgb.g}, ${primaryRgb.b}, 0.2)`);
    }
  };

  // Helper function to convert hex to RGB
  const hexToRgb = (hex: string) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16)
    } : null;
  };

  // Change to a predefined theme
  const changeTheme = (themeName: string) => {
    const theme = themes.find(t => t.name === themeName);
    if (theme) {
      setCurrentTheme(theme);
      applyColors(theme.colors);
      
      // Save to localStorage
      localStorage.setItem('dharma-color-theme', themeName);
    }
  };

  // Update specific colors while keeping the current theme
  const updateColors = (newColors: Partial<ColorTheme['colors']>) => {
    const updatedTheme = {
      ...currentTheme,
      colors: { ...currentTheme.colors, ...newColors }
    };
    setCurrentTheme(updatedTheme);
    applyColors(updatedTheme.colors);
    
    // Save custom colors to localStorage
    localStorage.setItem('dharma-custom-colors', JSON.stringify(updatedTheme.colors));
  };

  // Load saved theme on mount
  useEffect(() => {
    const savedTheme = localStorage.getItem('dharma-color-theme');
    const savedColors = localStorage.getItem('dharma-custom-colors');
    
    if (savedColors) {
      // Apply custom colors
      const customColors = JSON.parse(savedColors);
      const customTheme = {
        name: 'custom',
        colors: customColors
      };
      setCurrentTheme(customTheme);
      applyColors(customColors);
    } else if (savedTheme) {
      // Apply saved theme
      changeTheme(savedTheme);
    } else {
      // Apply default theme
      applyColors(currentTheme.colors);
    }
  }, []);

  const value: ColorContextType = {
    currentTheme,
    availableThemes: themes,
    changeTheme,
    updateColors
  };

  return (
    <ColorContext.Provider value={value}>
      {children}
    </ColorContext.Provider>
  );
};

// Hook to use the color system
export const useColors = () => {
  const context = useContext(ColorContext);
  if (!context) {
    throw new Error('useColors must be used within a ColorProvider');
  }
  return context;
};

// Higher-order component for easy integration
export const withColors = <P extends object>(Component: React.ComponentType<P>) => {
  return (props: P) => (
    <ColorProvider>
      <Component {...props} />
    </ColorProvider>
  );
};
