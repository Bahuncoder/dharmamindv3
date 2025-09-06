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
    borderPrimary: string;      // Emerald border color
    borderSecondary: string;    // Secondary border color
    textPrimary: string;        // Text colors
    textSecondary: string;      // Secondary text color
    background: string;         // Background color
    backgroundSecondary: string; // Secondary background
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
    name: 'light-gray-emerald', // New default theme
    colors: {
      primaryStart: '#f8fafc',        // Very light gray
      primaryEnd: '#f1f5f9',          // Light gray
      primaryHoverStart: '#e2e8f0',   // Medium light gray
      primaryHoverEnd: '#cbd5e1',     // Medium gray
      primary: '#f8fafc',             // Very light gray backgrounds
      primaryHover: '#e2e8f0',        // Medium light gray on hover
      borderPrimary: '#10b981',       // Emerald green for borders
      borderSecondary: '#6ee7b7',     // Light emerald for secondary borders
      textPrimary: '#1f2937',         // Dark gray text
      textSecondary: '#6b7280',       // Medium gray text
      background: '#ffffff',          // White background
      backgroundSecondary: '#f9fafb', // Very light background
    }
  },
  {
    name: 'dharma-light-gray', // Renamed old theme
    colors: {
      primaryStart: '#f3f4f6',        // Light gray
      primaryEnd: '#e5e7eb',          // Slightly darker light gray
      primaryHoverStart: '#e5e7eb',   // Darker light gray
      primaryHoverEnd: '#d1d5db',     // Medium gray
      primary: '#f3f4f6',             // Light gray backgrounds
      primaryHover: '#e5e7eb',        // Darker light gray on hover
      borderPrimary: '#10b981',       // Emerald green borders
      borderSecondary: '#6ee7b7',     // Light emerald borders
      textPrimary: '#1f2937',         // Dark gray text
      textSecondary: '#6b7280',       // Medium gray text
      background: '#ffffff',          // White background
      backgroundSecondary: '#f9fafb', // Very light background
    }
  },
  {
    name: 'ocean-blue',
    colors: {
      primaryStart: '#007BFF',        // Blue
      primaryEnd: '#0056B3',          // Darker blue
      primaryHoverStart: '#0056B3',   // Darker blue
      primaryHoverEnd: '#004085',     // Even darker blue
      primary: '#007BFF',
      primaryHover: '#0056B3',
      borderPrimary: '#007BFF',       // Blue borders
      borderSecondary: '#66b3ff',     // Light blue borders
      textPrimary: '#ffffff',         // White text
      textSecondary: '#e6f3ff',       // Light blue text
      background: '#f0f8ff',          // Light blue background
      backgroundSecondary: '#e6f3ff', // Very light blue
    }
  },
  {
    name: 'forest-emerald',
    colors: {
      primaryStart: '#32A370',        // Emerald green
      primaryEnd: '#2D8B5F',          // Darker emerald
      primaryHoverStart: '#2D8B5F',   // Darker emerald
      primaryHoverEnd: '#28784F',     // Even darker emerald
      primary: '#32A370',
      primaryHover: '#2D8B5F',
      borderPrimary: '#32A370',       // Emerald borders
      borderSecondary: '#7dd3a0',     // Light emerald borders
      textPrimary: '#ffffff',         // White text
      textSecondary: '#ecfdf5',       // Light green text
      background: '#f0fdf4',          // Light green background
      backgroundSecondary: '#ecfdf5', // Very light green
    }
  },
  {
    name: 'royal-purple',
    colors: {
      primaryStart: '#8B5CF6',        // Purple
      primaryEnd: '#7C3AED',          // Darker purple
      primaryHoverStart: '#7C3AED',   // Darker purple
      primaryHoverEnd: '#6D28D9',     // Even darker purple
      primary: '#8B5CF6',
      primaryHover: '#7C3AED',
      borderPrimary: '#8B5CF6',       // Purple borders
      borderSecondary: '#c4b5fd',     // Light purple borders
      textPrimary: '#ffffff',         // White text
      textSecondary: '#f3f0ff',       // Light purple text
      background: '#faf5ff',          // Light purple background
      backgroundSecondary: '#f3f0ff', // Very light purple
    }
  },
  {
    name: 'sunset-saffron',
    colors: {
      primaryStart: '#F2A300',        // Saffron orange
      primaryEnd: '#D4910A',          // Darker saffron
      primaryHoverStart: '#D4910A',   // Darker saffron
      primaryHoverEnd: '#B8860B',     // Even darker saffron
      primary: '#F2A300',
      primaryHover: '#D4910A',
      borderPrimary: '#F2A300',       // Saffron borders
      borderSecondary: '#fcd34d',     // Light saffron borders
      textPrimary: '#ffffff',         // White text
      textSecondary: '#fffbeb',       // Light yellow text
      background: '#fffbeb',          // Light yellow background
      backgroundSecondary: '#fef3c7', // Very light yellow
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
    
    // Border colors - Emerald green for borders
    root.style.setProperty('--color-border-primary', colors.borderPrimary);
    root.style.setProperty('--color-border-secondary', colors.borderSecondary);
    
    // Text colors
    root.style.setProperty('--color-text-primary', colors.textPrimary);
    root.style.setProperty('--color-text-secondary', colors.textSecondary);
    
    // Background colors
    root.style.setProperty('--color-background', colors.background);
    root.style.setProperty('--color-background-secondary', colors.backgroundSecondary);
    
    // Update focus ring color to use emerald border color
    root.style.setProperty('--color-focus', colors.borderPrimary);
    const borderRgb = hexToRgb(colors.borderPrimary);
    if (borderRgb) {
      root.style.setProperty('--color-focus-ring', `rgba(${borderRgb.r}, ${borderRgb.g}, ${borderRgb.b}, 0.2)`);
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
