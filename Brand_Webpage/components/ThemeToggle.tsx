/**
 * 🌓 Theme Toggle Component
 * ==========================
 * 
 * A beautiful animated toggle button for switching between
 * dark and light modes with smooth transitions.
 */

import React from 'react';
import { useTheme } from '../contexts/ThemeContext';

interface ThemeToggleProps {
  className?: string;
  showLabel?: boolean;
}

export default function ThemeToggle({ className = '', showLabel = false }: ThemeToggleProps) {
  const { resolvedTheme, toggleTheme, theme, setTheme } = useTheme();
  const isDark = resolvedTheme === 'dark';

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      {showLabel && (
        <span className="text-sm text-neutral-600 dark:text-neutral-400">
          {isDark ? 'Dark' : 'Light'}
        </span>
      )}
      
      <button
        onClick={toggleTheme}
        className="relative w-14 h-7 rounded-full bg-neutral-200 dark:bg-neutral-700 
                   transition-colors duration-300 focus:outline-none focus:ring-2 
                   focus:ring-[#d4a854] focus:ring-offset-2 dark:focus:ring-offset-neutral-900"
        aria-label={`Switch to ${isDark ? 'light' : 'dark'} mode`}
        title={`Switch to ${isDark ? 'light' : 'dark'} mode`}
      >
        {/* Track */}
        <span className="absolute inset-0 rounded-full overflow-hidden">
          {/* Stars (visible in dark mode) */}
          <span className={`absolute inset-0 transition-opacity duration-300 ${isDark ? 'opacity-100' : 'opacity-0'}`}>
            <span className="absolute w-1 h-1 bg-white rounded-full top-1 left-2 animate-pulse" />
            <span className="absolute w-0.5 h-0.5 bg-white rounded-full top-3 left-4" />
            <span className="absolute w-1 h-1 bg-white rounded-full top-2 left-6 animate-pulse" style={{ animationDelay: '0.5s' }} />
          </span>
        </span>

        {/* Thumb with sun/moon */}
        <span
          className={`absolute top-0.5 w-6 h-6 rounded-full shadow-md transform transition-all duration-300 
                      ${isDark ? 'translate-x-7 bg-neutral-800' : 'translate-x-0.5 bg-white'}`}
        >
          {/* Sun icon */}
          <svg
            className={`absolute inset-0 w-6 h-6 p-1 text-[#d4a854] transition-opacity duration-300 
                       ${isDark ? 'opacity-0' : 'opacity-100'}`}
            fill="currentColor"
            viewBox="0 0 20 20"
          >
            <path
              fillRule="evenodd"
              d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z"
              clipRule="evenodd"
            />
          </svg>

          {/* Moon icon */}
          <svg
            className={`absolute inset-0 w-6 h-6 p-1.5 text-[#d4a854] transition-opacity duration-300 
                       ${isDark ? 'opacity-100' : 'opacity-0'}`}
            fill="currentColor"
            viewBox="0 0 20 20"
          >
            <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
          </svg>
        </span>
      </button>

      {/* Optional: Theme selector dropdown */}
      {showLabel && (
        <select
          value={theme}
          onChange={(e) => setTheme(e.target.value as 'light' | 'dark' | 'system')}
          className="ml-2 text-xs bg-transparent border border-neutral-300 dark:border-neutral-600 
                     rounded px-2 py-1 text-neutral-600 dark:text-neutral-400 cursor-pointer
                     focus:outline-none focus:ring-1 focus:ring-[#d4a854]"
        >
          <option value="light">Light</option>
          <option value="dark">Dark</option>
          <option value="system">System</option>
        </select>
      )}
    </div>
  );
}
