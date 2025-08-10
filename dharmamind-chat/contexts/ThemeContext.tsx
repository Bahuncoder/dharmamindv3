import React, { createContext, useContext, useState, useEffect } from 'react';

interface ThemeContextType {
  isDarkMode: boolean;
  toggleDarkMode: () => void;
  spiritualMode: boolean;
  toggleSpiritualMode: () => void;
  currentTheme: 'light' | 'dark' | 'spiritual';
  setCurrentTheme: (theme: 'light' | 'dark' | 'spiritual') => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

interface ThemeProviderProps {
  children: React.ReactNode;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [spiritualMode, setSpiritualMode] = useState(false);
  const [currentTheme, setCurrentTheme] = useState<'light' | 'dark' | 'spiritual'>('light');

  // Load theme preference from localStorage
  useEffect(() => {
    const savedTheme = localStorage.getItem('dharmamind-theme');
    const savedSpiritual = localStorage.getItem('dharmamind-spiritual-mode');
    
    if (savedTheme) {
      setCurrentTheme(savedTheme as 'light' | 'dark' | 'spiritual');
      setIsDarkMode(savedTheme === 'dark' || savedTheme === 'spiritual');
      setSpiritualMode(savedTheme === 'spiritual');
    } else {
      // Auto-detect system preference
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      if (prefersDark) {
        setCurrentTheme('dark');
        setIsDarkMode(true);
      }
    }
    
    if (savedSpiritual === 'true') {
      setSpiritualMode(true);
    }
  }, []);

  // Apply theme classes to document
  useEffect(() => {
    const root = document.documentElement;
    
    // Remove all theme classes
    root.classList.remove('light', 'dark', 'spiritual');
    
    // Add current theme class
    root.classList.add(currentTheme);
    
    // Apply CSS variables based on theme
    if (currentTheme === 'spiritual') {
      root.style.setProperty('--bg-primary', '#0a0a0f');
      root.style.setProperty('--bg-secondary', '#1a1a2e');
      root.style.setProperty('--bg-tertiary', '#16213e');
      root.style.setProperty('--text-primary', '#f7fafc');
      root.style.setProperty('--text-secondary', '#e2e8f0');
      root.style.setProperty('--text-muted', '#a0aec0');
      root.style.setProperty('--accent-primary', '#4299e1');
      root.style.setProperty('--accent-secondary', '#63b3ed');
      root.style.setProperty('--border-color', '#2d3748');
      root.style.setProperty('--spiritual-glow', '#ffd700');
      root.style.setProperty('--meditation-purple', '#805ad5');
    } else if (currentTheme === 'dark') {
      root.style.setProperty('--bg-primary', '#1a202c');
      root.style.setProperty('--bg-secondary', '#2d3748');
      root.style.setProperty('--bg-tertiary', '#4a5568');
      root.style.setProperty('--text-primary', '#f7fafc');
      root.style.setProperty('--text-secondary', '#e2e8f0');
      root.style.setProperty('--text-muted', '#a0aec0');
      root.style.setProperty('--accent-primary', '#4299e1');
      root.style.setProperty('--accent-secondary', '#63b3ed');
      root.style.setProperty('--border-color', '#4a5568');
    } else {
      // Light mode - use default CSS variables
      root.style.removeProperty('--bg-primary');
      root.style.removeProperty('--bg-secondary');
      root.style.removeProperty('--bg-tertiary');
      root.style.removeProperty('--text-primary');
      root.style.removeProperty('--text-secondary');
      root.style.removeProperty('--text-muted');
      root.style.removeProperty('--accent-primary');
      root.style.removeProperty('--accent-secondary');
      root.style.removeProperty('--border-color');
    }
  }, [currentTheme]);

  const toggleDarkMode = () => {
    const newTheme = isDarkMode ? 'light' : 'dark';
    setCurrentTheme(newTheme);
    setIsDarkMode(!isDarkMode);
    setSpiritualMode(false);
    localStorage.setItem('dharmamind-theme', newTheme);
  };

  const toggleSpiritualMode = () => {
    if (spiritualMode) {
      // Exit spiritual mode
      setCurrentTheme('light');
      setIsDarkMode(false);
      setSpiritualMode(false);
      localStorage.setItem('dharmamind-theme', 'light');
    } else {
      // Enter spiritual mode
      setCurrentTheme('spiritual');
      setIsDarkMode(true);
      setSpiritualMode(true);
      localStorage.setItem('dharmamind-theme', 'spiritual');
    }
    localStorage.setItem('dharmamind-spiritual-mode', (!spiritualMode).toString());
  };

  const handleSetCurrentTheme = (theme: 'light' | 'dark' | 'spiritual') => {
    setCurrentTheme(theme);
    setIsDarkMode(theme === 'dark' || theme === 'spiritual');
    setSpiritualMode(theme === 'spiritual');
    localStorage.setItem('dharmamind-theme', theme);
    localStorage.setItem('dharmamind-spiritual-mode', (theme === 'spiritual').toString());
  };

  return (
    <ThemeContext.Provider value={{
      isDarkMode,
      toggleDarkMode,
      spiritualMode,
      toggleSpiritualMode,
      currentTheme,
      setCurrentTheme: handleSetCurrentTheme
    }}>
      {children}
    </ThemeContext.Provider>
  );
};
