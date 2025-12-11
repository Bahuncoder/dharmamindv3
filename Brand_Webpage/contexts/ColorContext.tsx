/**
 * 🎨 MASTER COLOR THEME SYSTEM for DharmaMind
 * 
 * This integrates with the Master Color System in colors.css
 * When you change colors here, they update throughout the entire app instantly.
 * 
 * 🎯 HOW IT WORKS:
 * 1. This context updates CSS custom properties in colors.css
 * 2. All components use those CSS variables
 * 3. Change brand colors = entire app updates automatically
 */

import React, { createContext, useContext, useEffect, useState } from 'react';

interface ColorTheme {
  name: string;
  displayName: string;
  description: string;
  colors: {
    // Master brand colors that control everything
    brandPrimary: string;      // Main brand color
    brandAccent: string;       // Accent/border color  
    brandSecondary: string;    // Secondary actions
    
    // Auto-derived hover states
    brandPrimaryHover: string;
    brandAccentHover: string;
    brandSecondaryHover: string;
    
    // Legacy support (auto-mapped from brand colors)
    primaryStart: string;
    primaryEnd: string;
    primaryHoverStart: string;
    primaryHoverEnd: string;
    primary: string;
    primaryHover: string;
  };
}

interface ColorContextType {
  currentTheme: ColorTheme;
  availableThemes: ColorTheme[];
  changeTheme: (themeName: string) => void;
  updateBrandColors: (colors: { brandPrimary?: string; brandAccent?: string; brandSecondary?: string }) => void;
  resetToDefault: () => void;
}

// 🎨 PREDEFINED THEMES - All using the new master system
const themes: ColorTheme[] = [
  {
    name: 'dharma-default',
    displayName: 'Dharma Emerald',
    description: 'Spiritual emerald green - the official DharmaMind theme',
    colors: {
      brandPrimary: '#10b981',      // Emerald-500: Main brand
      brandAccent: '#10b981',       // Emerald-500: Accents/borders
      brandSecondary: '#059669',    // Emerald-600: Secondary actions
      brandPrimaryHover: '#059669', // Emerald-600
      brandAccentHover: '#047857',  // Emerald-700
      brandSecondaryHover: '#047857', // Emerald-700
      // Legacy mappings
      primaryStart: '#10b981',
      primaryEnd: '#059669',
      primaryHoverStart: '#059669',
      primaryHoverEnd: '#047857',
      primary: '#10b981',
      primaryHover: '#059669',
    }
  },
  {
    name: 'ocean-serenity',
    displayName: 'Ocean Serenity',
    description: 'Calming ocean blues with teal accents',
    colors: {
      brandPrimary: '#0ea5e9',      // Sky-500
      brandAccent: '#14b8a6',       // Teal-500
      brandSecondary: '#3b82f6',    // Blue-500
      brandPrimaryHover: '#0369a1', // Sky-700
      brandAccentHover: '#0f766e',  // Teal-600
      brandSecondaryHover: '#2563eb', // Blue-600
      primaryStart: '#0ea5e9',
      primaryEnd: '#14b8a6',
      primaryHoverStart: '#0369a1',
      primaryHoverEnd: '#0f766e',
      primary: '#0ea5e9',
      primaryHover: '#0369a1',
    }
  },
  {
    name: 'forest-wisdom',
    displayName: 'Forest Wisdom',
    description: 'Natural forest greens for mindful focus',
    colors: {
      brandPrimary: '#059669',      // Emerald-600
      brandAccent: '#10b981',       // Emerald-500
      brandSecondary: '#22c55e',    // Green-500
      brandPrimaryHover: '#047857', // Emerald-700
      brandAccentHover: '#059669',  // Emerald-600
      brandSecondaryHover: '#16a34a', // Green-600
      primaryStart: '#059669',
      primaryEnd: '#10b981',
      primaryHoverStart: '#047857',
      primaryHoverEnd: '#059669',
      primary: '#059669',
      primaryHover: '#047857',
    }
  },
  {
    name: 'royal-purple',
    displayName: 'Royal Purple',
    description: 'Elegant purple tones for spiritual depth',
    colors: {
      brandPrimary: '#8b5cf6',      // Violet-500
      brandAccent: '#a855f7',       // Purple-500
      brandSecondary: '#ec4899',    // Pink-500
      brandPrimaryHover: '#7c3aed', // Violet-600
      brandAccentHover: '#9333ea',  // Purple-600
      brandSecondaryHover: '#db2777', // Pink-600
      primaryStart: '#8b5cf6',
      primaryEnd: '#a855f7',
      primaryHoverStart: '#7c3aed',
      primaryHoverEnd: '#9333ea',
      primary: '#8b5cf6',
      primaryHover: '#7c3aed',
    }
  },
  {
    name: 'emerald-focus',
    displayName: 'Pure Emerald',
    description: 'Pure emerald tones for clarity and growth',
    colors: {
      brandPrimary: '#10b981',      // Emerald-500
      brandAccent: '#059669',       // Emerald-600
      brandSecondary: '#14b8a6',    // Teal-500
      brandPrimaryHover: '#059669', // Emerald-600
      brandAccentHover: '#047857',  // Emerald-700
      brandSecondaryHover: '#0f766e', // Teal-600
      primaryStart: '#10b981',
      primaryEnd: '#059669',
      primaryHoverStart: '#059669',
      primaryHoverEnd: '#047857',
      primary: '#10b981',
      primaryHover: '#059669',
    }
  },
  {
    name: 'sunset-warmth',
    displayName: 'Sunset Warmth',
    description: 'Warm sunset colors for comfort and energy',
    colors: {
      brandPrimary: '#f97316',      // Orange-500
      brandAccent: '#eab308',       // Yellow-500
      brandSecondary: '#ef4444',    // Red-500
      brandPrimaryHover: '#ea580c', // Orange-600
      brandAccentHover: '#ca8a04',  // Yellow-600
      brandSecondaryHover: '#dc2626', // Red-600
      primaryStart: '#f97316',
      primaryEnd: '#eab308',
      primaryHoverStart: '#ea580c',
      primaryHoverEnd: '#ca8a04',
      primary: '#f97316',
      primaryHover: '#ea580c',
    }
  }
];

const ColorContext = createContext<ColorContextType | undefined>(undefined);

export const ColorProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [currentTheme, setCurrentTheme] = useState<ColorTheme>(themes[0]); // Default to dharma-gray

  // 🎯 MASTER COLOR APPLICATION FUNCTION
  // This updates the CSS custom properties that control the entire app
  const applyColors = (colors: ColorTheme['colors']) => {
    const root = document.documentElement;
    
    // 🔥 UPDATE MASTER BRAND COLORS (These control everything!)
    root.style.setProperty('--brand-primary', colors.brandPrimary);
    root.style.setProperty('--brand-primary-hover', colors.brandPrimaryHover);
    root.style.setProperty('--brand-accent', colors.brandAccent);
    root.style.setProperty('--brand-accent-hover', colors.brandAccentHover);
    root.style.setProperty('--brand-secondary', colors.brandSecondary);
    root.style.setProperty('--brand-secondary-hover', colors.brandSecondaryHover);
    
    // 🔄 UPDATE ALL DERIVED COLORS (Auto-update from brand colors)
    root.style.setProperty('--color-primary', colors.brandPrimary);
    root.style.setProperty('--color-primary-hover', colors.brandPrimaryHover);
    root.style.setProperty('--color-accent', colors.brandAccent);
    root.style.setProperty('--color-accent-hover', colors.brandAccentHover);
    root.style.setProperty('--color-secondary', colors.brandSecondary);
    root.style.setProperty('--color-secondary-hover', colors.brandSecondaryHover);
    
    // 🎨 UPDATE LEGACY SUPPORT COLORS
    root.style.setProperty('--color-primary-start', colors.brandPrimary);
    root.style.setProperty('--color-primary-end', colors.brandAccent);
    root.style.setProperty('--color-primary-hover-start', colors.brandPrimaryHover);
    root.style.setProperty('--color-primary-hover-end', colors.brandAccentHover);
    
    // 🎯 UPDATE FOCUS AND INTERACTION COLORS
    root.style.setProperty('--color-focus', colors.brandAccent);
    const accentRgb = hexToRgb(colors.brandAccent);
    if (accentRgb) {
      root.style.setProperty('--color-focus-ring', `rgba(${accentRgb.r}, ${accentRgb.g}, ${accentRgb.b}, 0.2)`);
    }
    
    // 🎨 UPDATE GRADIENTS
    root.style.setProperty('--brand-gradient', `linear-gradient(135deg, ${colors.brandPrimary} 0%, ${colors.brandAccent} 100%)`);
    root.style.setProperty('--brand-gradient-hover', `linear-gradient(135deg, ${colors.brandPrimaryHover} 0%, ${colors.brandAccentHover} 100%)`);
    
    console.log('🎨 Color theme applied:', {
      primary: colors.brandPrimary,
      accent: colors.brandAccent,
      secondary: colors.brandSecondary
    });
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

  // 🔄 CHANGE TO A PREDEFINED THEME
  const changeTheme = (themeName: string) => {
    const theme = themes.find(t => t.name === themeName);
    if (theme) {
      setCurrentTheme(theme);
      applyColors(theme.colors);
      
      // Save to localStorage
      localStorage.setItem('dharma-color-theme', themeName);
      console.log('🎨 Theme changed to:', theme.displayName);
    }
  };

  // 🎨 UPDATE SPECIFIC BRAND COLORS (Advanced usage)
  const updateBrandColors = (newColors: { brandPrimary?: string; brandAccent?: string; brandSecondary?: string }) => {
    const updatedColors = { ...currentTheme.colors };
    
    if (newColors.brandPrimary) {
      updatedColors.brandPrimary = newColors.brandPrimary;
      updatedColors.primary = newColors.brandPrimary;
      updatedColors.primaryStart = newColors.brandPrimary;
      // Auto-generate hover state (darker)
      updatedColors.brandPrimaryHover = darkenColor(newColors.brandPrimary, 20);
      updatedColors.primaryHover = updatedColors.brandPrimaryHover;
      updatedColors.primaryHoverStart = updatedColors.brandPrimaryHover;
    }
    
    if (newColors.brandAccent) {
      updatedColors.brandAccent = newColors.brandAccent;
      updatedColors.primaryEnd = newColors.brandAccent;
      // Auto-generate hover state (darker)
      updatedColors.brandAccentHover = darkenColor(newColors.brandAccent, 20);
      updatedColors.primaryHoverEnd = updatedColors.brandAccentHover;
    }
    
    if (newColors.brandSecondary) {
      updatedColors.brandSecondary = newColors.brandSecondary;
      // Auto-generate hover state (darker)
      updatedColors.brandSecondaryHover = darkenColor(newColors.brandSecondary, 20);
    }
    
    const customTheme = {
      ...currentTheme,
      name: 'custom',
      displayName: 'Custom Theme',
      description: 'Your custom color configuration',
      colors: updatedColors
    };
    
    setCurrentTheme(customTheme);
    applyColors(updatedColors);
    
    // Save custom colors to localStorage
    localStorage.setItem('dharma-custom-colors', JSON.stringify(updatedColors));
    console.log('🎨 Custom colors applied:', newColors);
  };

  // Helper function to darken a color
  const darkenColor = (hex: string, percent: number) => {
    const rgb = hexToRgb(hex);
    if (!rgb) return hex;
    
    const factor = (100 - percent) / 100;
    const r = Math.round(rgb.r * factor);
    const g = Math.round(rgb.g * factor);
    const b = Math.round(rgb.b * factor);
    
    return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
  };

  // 🔄 RESET TO DEFAULT THEME
  const resetToDefault = () => {
    changeTheme('dharma-default');
    localStorage.removeItem('dharma-custom-colors');
    console.log('🎨 Reset to default theme');
  };

  // 🚀 LOAD SAVED THEME ON MOUNT
  useEffect(() => {
    const savedTheme = localStorage.getItem('dharma-color-theme');
    const savedColors = localStorage.getItem('dharma-custom-colors');
    
    if (savedColors) {
      // Apply custom colors
      try {
        const customColors = JSON.parse(savedColors);
        const customTheme = {
          name: 'custom',
          displayName: 'Custom Theme',
          description: 'Your saved custom colors',
          colors: customColors
        };
        setCurrentTheme(customTheme);
        applyColors(customColors);
        console.log('🎨 Loaded custom colors from storage');
      } catch (error) {
        console.warn('🎨 Failed to load custom colors, using default');
        applyColors(themes[0].colors);
      }
    } else if (savedTheme) {
      // Apply saved theme
      changeTheme(savedTheme);
    } else {
      // Apply default theme
      applyColors(currentTheme.colors);
      console.log('🎨 Applied default theme');
    }
  }, []);

  const value: ColorContextType = {
    currentTheme,
    availableThemes: themes,
    changeTheme,
    updateBrandColors,
    resetToDefault
  };

  return (
    <ColorContext.Provider value={value}>
      {children}
    </ColorContext.Provider>
  );
};

// 🎨 HOOK TO USE THE COLOR SYSTEM
export const useColors = () => {
  const context = useContext(ColorContext);
  if (!context) {
    throw new Error('useColors must be used within a ColorProvider');
  }
  return context;
};

// 🎨 HIGHER-ORDER COMPONENT FOR EASY INTEGRATION
export const withColors = <P extends object>(Component: React.ComponentType<P>) => {
  return (props: P) => (
    <ColorProvider>
      <Component {...props} />
    </ColorProvider>
  );
};
