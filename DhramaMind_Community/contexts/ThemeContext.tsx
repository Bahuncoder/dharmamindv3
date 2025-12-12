/**
 * 🌓 Theme Context - Dark/Light Mode Toggle
 * ==========================================
 * 
 * Provides system-wide theme management with:
 * - Dark/Light mode toggle
 * - System preference detection
 * - LocalStorage persistence
 * - Smooth transitions
 */

import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';

type Theme = 'light' | 'dark' | 'system';

interface ThemeContextType {
    theme: Theme;
    resolvedTheme: 'light' | 'dark';
    setTheme: (theme: Theme) => void;
    toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

const STORAGE_KEY = 'dharmamind-community-theme';

interface ThemeProviderProps {
    children: ReactNode;
    defaultTheme?: Theme;
}

export function ThemeProvider({ children, defaultTheme = 'system' }: ThemeProviderProps) {
    const [theme, setThemeState] = useState<Theme>(defaultTheme);
    const [resolvedTheme, setResolvedTheme] = useState<'light' | 'dark'>('light');
    const [mounted, setMounted] = useState(false);

    // Get system preference
    const getSystemTheme = (): 'light' | 'dark' => {
        if (typeof window !== 'undefined') {
            return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        }
        return 'light';
    };

    // Resolve theme based on setting
    const resolveTheme = (t: Theme): 'light' | 'dark' => {
        if (t === 'system') {
            return getSystemTheme();
        }
        return t;
    };

    // Load theme from localStorage on mount
    useEffect(() => {
        setMounted(true);
        const stored = localStorage.getItem(STORAGE_KEY) as Theme | null;
        if (stored && ['light', 'dark', 'system'].includes(stored)) {
            setThemeState(stored);
        }
    }, []);

    // Update resolved theme and apply to document
    useEffect(() => {
        if (!mounted) return;

        const resolved = resolveTheme(theme);
        setResolvedTheme(resolved);

        // Apply to document
        const root = document.documentElement;
        root.classList.remove('light', 'dark');
        root.classList.add(resolved);

        // Update meta theme-color
        const metaThemeColor = document.querySelector('meta[name="theme-color"]');
        if (metaThemeColor) {
            metaThemeColor.setAttribute('content', resolved === 'dark' ? '#0a0a0a' : '#ffffff');
        }
    }, [theme, mounted]);

    // Listen for system preference changes
    useEffect(() => {
        if (!mounted || theme !== 'system') return;

        const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
        const handler = () => setResolvedTheme(getSystemTheme());

        mediaQuery.addEventListener('change', handler);
        return () => mediaQuery.removeEventListener('change', handler);
    }, [theme, mounted]);

    const setTheme = (newTheme: Theme) => {
        setThemeState(newTheme);
        localStorage.setItem(STORAGE_KEY, newTheme);
    };

    const toggleTheme = () => {
        const newTheme = resolvedTheme === 'dark' ? 'light' : 'dark';
        setTheme(newTheme);
    };

    // Prevent flash on initial load
    if (!mounted) {
        return (
            <ThemeContext.Provider value={{ theme, resolvedTheme, setTheme, toggleTheme }}>
                {children}
            </ThemeContext.Provider>
        );
    }

    return (
        <ThemeContext.Provider value={{ theme, resolvedTheme, setTheme, toggleTheme }}>
            {children}
        </ThemeContext.Provider>
    );
}

export function useTheme() {
    const context = useContext(ThemeContext);
    if (context === undefined) {
        throw new Error('useTheme must be used within a ThemeProvider');
    }
    return context;
}

export default ThemeContext;
