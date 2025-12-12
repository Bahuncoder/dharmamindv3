/**
 * 🌓 Theme Toggle Component
 * =========================
 * 
 * Animated toggle for Dark/Light mode switching
 * - Beautiful sun/moon icons with smooth transitions
 * - Optional dropdown for system preference
 * - Accessible with proper aria labels
 */

import React from 'react';
import { useTheme } from '../contexts/ThemeContext';

interface ThemeToggleProps {
    className?: string;
    showLabel?: boolean;
}

const ThemeToggle: React.FC<ThemeToggleProps> = ({
    className = '',
    showLabel = false
}) => {
    const { resolvedTheme, toggleTheme } = useTheme();
    const isDark = resolvedTheme === 'dark';

    return (
        <button
            onClick={toggleTheme}
            className={`
        relative inline-flex items-center justify-center
        w-10 h-10 rounded-lg
        text-neutral-600 dark:text-neutral-400
        hover:text-gold-600 dark:hover:text-gold-400
        hover:bg-neutral-100 dark:hover:bg-neutral-800
        transition-all duration-300
        focus:outline-none focus:ring-2 focus:ring-gold-600/20
        ${className}
      `}
            aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
            title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
        >
            {/* Sun Icon */}
            <svg
                className={`
          w-5 h-5 absolute
          transition-all duration-500 ease-in-out
          ${isDark ? 'opacity-0 rotate-90 scale-0' : 'opacity-100 rotate-0 scale-100'}
        `}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
            >
                <circle cx="12" cy="12" r="5" />
                <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
            </svg>

            {/* Moon Icon */}
            <svg
                className={`
          w-5 h-5 absolute
          transition-all duration-500 ease-in-out
          ${isDark ? 'opacity-100 rotate-0 scale-100' : 'opacity-0 -rotate-90 scale-0'}
        `}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
            >
                <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" />
            </svg>

            {/* Optional Label */}
            {showLabel && (
                <span className="ml-8 text-sm font-medium">
                    {isDark ? 'Dark' : 'Light'}
                </span>
            )}
        </button>
    );
};

export default ThemeToggle;
