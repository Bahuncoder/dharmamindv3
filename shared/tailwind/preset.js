/**
 * ═══════════════════════════════════════════════════════════════════════════
 * 🕉️ DHARMAMIND UNIFIED TAILWIND PRESET
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * Shared Tailwind configuration for all DharmaMind products:
 * - dharmamind.ai (Chat App)
 * - dharmamind.org (Community)
 * - dharmamind.com (Brand Website)
 * 
 * Design System: Minimalist Grayscale + Gold Accents
 * 
 * Usage in each app's tailwind.config.js:
 * const sharedPreset = require('../shared/tailwind/preset');
 * module.exports = { presets: [sharedPreset], ... }
 */

module.exports = {
    theme: {
        extend: {
            colors: {
                // ═══════════════════════════════════════════════════════════════
                // NEUTRAL PALETTE (Stone-based grays for warmth)
                // ═══════════════════════════════════════════════════════════════
                neutral: {
                    50: '#fafaf9',   // Page background (lightest)
                    100: '#f5f5f4',  // Card/surface background
                    200: '#e7e5e4',  // Borders, dividers
                    300: '#d6d3d1',  // Hover states, secondary borders
                    400: '#a8a29e',  // Disabled text, placeholders
                    500: '#78716c',  // Muted text
                    600: '#57534e',  // Secondary text
                    700: '#44403c',  // Primary text (lighter variant)
                    800: '#292524',  // Primary text (dark)
                    900: '#1c1917',  // Headings, emphasis (darkest)
                    950: '#0c0a09',  // Extra dark (rare use)
                },

                // ═══════════════════════════════════════════════════════════════
                // GOLD ACCENT (Primary action color)
                // ═══════════════════════════════════════════════════════════════
                gold: {
                    50: '#fefdf8',
                    100: '#f4e4bd',  // Light gold backgrounds
                    200: '#edd9a0',
                    300: '#e5c878',
                    400: '#ddb84d',
                    500: '#d4a854',  // DEFAULT - primary buttons
                    600: '#d4a854',  // Same as 500 (main accent)
                    700: '#b8860b',  // Hover states
                    800: '#8b6914',
                    900: '#5c4a0f',
                },

                // ═══════════════════════════════════════════════════════════════
                // SEMANTIC COLORS
                // ═══════════════════════════════════════════════════════════════
                success: {
                    50: '#f0fdf4',
                    100: '#dcfce7',
                    500: '#22c55e',
                    600: '#16a34a',
                    700: '#15803d',
                },
                warning: {
                    50: '#fffbeb',
                    100: '#fef3c7',
                    500: '#f59e0b',
                    600: '#d97706',
                    700: '#b45309',
                },
                error: {
                    50: '#fef2f2',
                    100: '#fee2e2',
                    500: '#ef4444',
                    600: '#dc2626',
                    700: '#b91c1c',
                },
                info: {
                    50: '#eff6ff',
                    100: '#dbeafe',
                    500: '#3b82f6',
                    600: '#2563eb',
                    700: '#1d4ed8',
                },

                // ═══════════════════════════════════════════════════════════════
                // UI SHORTCUTS (Semantic names for common uses)
                // ═══════════════════════════════════════════════════════════════
                ui: {
                    bg: '#fafaf9',         // neutral-50  - page background
                    surface: '#f5f5f4',    // neutral-100 - card background
                    border: '#e7e5e4',     // neutral-200 - borders
                    hover: '#d6d3d1',      // neutral-300 - hover states
                },
                text: {
                    primary: '#1c1917',    // neutral-900 - headings
                    secondary: '#57534e',  // neutral-600 - body text
                    muted: '#78716c',      // neutral-500 - muted
                    light: '#a8a29e',      // neutral-400 - very light
                },

                // ═══════════════════════════════════════════════════════════════
                // LEGACY ALIASES (for backwards compatibility)
                // ═══════════════════════════════════════════════════════════════
                primary: {
                    50: '#fafaf9',
                    100: '#f5f5f4',
                    200: '#e7e5e4',
                    300: '#d6d3d1',
                    400: '#a8a29e',
                    500: '#78716c',
                    600: '#57534e',
                    700: '#44403c',
                    800: '#292524',
                    900: '#1c1917',
                },
                accent: {
                    50: '#fefdf8',
                    100: '#f4e4bd',
                    500: '#d4a854',
                    600: '#d4a854',
                    700: '#b8860b',
                },
                dharma: {
                    50: '#fefdf8',
                    100: '#f4e4bd',
                    500: '#d4a854',
                    600: '#d4a854',
                    700: '#b8860b',
                },
            },

            // ═══════════════════════════════════════════════════════════════
            // TYPOGRAPHY
            // ═══════════════════════════════════════════════════════════════
            fontFamily: {
                sans: [
                    'Inter',
                    '-apple-system',
                    'BlinkMacSystemFont',
                    'Segoe UI',
                    'Roboto',
                    'Helvetica Neue',
                    'Arial',
                    'sans-serif',
                ],
                serif: [
                    'Crimson Pro',
                    'Georgia',
                    'Cambria',
                    'Times New Roman',
                    'Times',
                    'serif',
                ],
                mono: [
                    'JetBrains Mono',
                    'Fira Code',
                    'Menlo',
                    'Monaco',
                    'Consolas',
                    'Liberation Mono',
                    'monospace',
                ],
            },

            // ═══════════════════════════════════════════════════════════════
            // SPACING & SIZING
            // ═══════════════════════════════════════════════════════════════
            spacing: {
                '18': '4.5rem',
                '22': '5.5rem',
                '128': '32rem',
                '144': '36rem',
            },
            maxWidth: {
                '8xl': '88rem',
                '9xl': '96rem',
            },

            // ═══════════════════════════════════════════════════════════════
            // BORDER RADIUS
            // ═══════════════════════════════════════════════════════════════
            borderRadius: {
                '4xl': '2rem',
                '5xl': '2.5rem',
            },

            // ═══════════════════════════════════════════════════════════════
            // SHADOWS
            // ═══════════════════════════════════════════════════════════════
            boxShadow: {
                'soft': '0 2px 15px -3px rgba(0, 0, 0, 0.07), 0 10px 20px -2px rgba(0, 0, 0, 0.04)',
                'soft-lg': '0 10px 40px -10px rgba(0, 0, 0, 0.1)',
                'glow': '0 0 20px rgba(212, 168, 84, 0.3)',
                'glow-lg': '0 0 40px rgba(212, 168, 84, 0.4)',
                'inner-soft': 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.05)',
            },

            // ═══════════════════════════════════════════════════════════════
            // ANIMATIONS
            // ═══════════════════════════════════════════════════════════════
            animation: {
                'fade-in': 'fadeIn 0.5s ease-out',
                'fade-in-up': 'fadeInUp 0.5s ease-out',
                'fade-in-down': 'fadeInDown 0.5s ease-out',
                'slide-in-right': 'slideInRight 0.3s ease-out',
                'slide-in-left': 'slideInLeft 0.3s ease-out',
                'pulse-soft': 'pulseSoft 2s ease-in-out infinite',
                'float': 'float 3s ease-in-out infinite',
                'shimmer': 'shimmer 2s linear infinite',
            },
            keyframes: {
                fadeIn: {
                    '0%': { opacity: '0' },
                    '100%': { opacity: '1' },
                },
                fadeInUp: {
                    '0%': { opacity: '0', transform: 'translateY(10px)' },
                    '100%': { opacity: '1', transform: 'translateY(0)' },
                },
                fadeInDown: {
                    '0%': { opacity: '0', transform: 'translateY(-10px)' },
                    '100%': { opacity: '1', transform: 'translateY(0)' },
                },
                slideInRight: {
                    '0%': { opacity: '0', transform: 'translateX(20px)' },
                    '100%': { opacity: '1', transform: 'translateX(0)' },
                },
                slideInLeft: {
                    '0%': { opacity: '0', transform: 'translateX(-20px)' },
                    '100%': { opacity: '1', transform: 'translateX(0)' },
                },
                pulseSoft: {
                    '0%, 100%': { opacity: '1' },
                    '50%': { opacity: '0.7' },
                },
                float: {
                    '0%, 100%': { transform: 'translateY(0)' },
                    '50%': { transform: 'translateY(-10px)' },
                },
                shimmer: {
                    '0%': { backgroundPosition: '-200% 0' },
                    '100%': { backgroundPosition: '200% 0' },
                },
            },

            // ═══════════════════════════════════════════════════════════════
            // TRANSITIONS
            // ═══════════════════════════════════════════════════════════════
            transitionDuration: {
                '400': '400ms',
            },
            transitionTimingFunction: {
                'smooth': 'cubic-bezier(0.4, 0, 0.2, 1)',
            },
        },
    },
    plugins: [],
};
