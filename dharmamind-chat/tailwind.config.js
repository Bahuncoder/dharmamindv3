/** 
 * 🕉️ DharmaMind Chat - Professional Design System
 * Tailwind CSS Configuration
 * 
 * Design Philosophy: Minimalist, calm, sophisticated
 * Inspired by: Linear, Notion, Apple
 * 
 * COLOR SYSTEM: Grayscale + Gold Accent
 * - Dark/Light grays for text and UI
 * - Gold accent for highlights and CTAs
 * 
 * @type {import('tailwindcss').Config} 
 */

module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        serif: ['Crimson Pro', 'Georgia', 'serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      fontSize: {
        'xs': ['0.75rem', { lineHeight: '1rem' }],
        'sm': ['0.8125rem', { lineHeight: '1.25rem' }],
        'base': ['0.9375rem', { lineHeight: '1.5rem' }],
        'lg': ['1.0625rem', { lineHeight: '1.75rem' }],
        'xl': ['1.25rem', { lineHeight: '1.75rem' }],
        '2xl': ['1.5rem', { lineHeight: '2rem' }],
        '3xl': ['1.875rem', { lineHeight: '2.25rem' }],
      },
      colors: {
        // ============================================
        // MINIMALIST: Dark gray for TEXT, light grays for UI, gold for accents
        // ============================================
        brand: {
          primary: '#1c1917',    // Near Black - TEXT ONLY
          secondary: '#d4a854',  // Warm Gold - buttons, accents, CTAs
          accent: '#d4a854',     // Warm Gold - highlights
          light: '#78716c',      // Medium Gray - secondary text
          muted: '#a8a29e',      // Light Gray - placeholder text
        },
        // Text colors (dark grays - ONLY place for dark colors)
        text: {
          primary: '#1c1917',    // Main text
          secondary: '#57534e',  // Secondary text
          muted: '#78716c',      // Muted/placeholder
          light: '#a8a29e',      // Very light text
        },
        // UI colors (light grays for backgrounds, borders, surfaces)
        ui: {
          bg: '#fafaf9',         // Page background
          surface: '#f5f5f4',    // Card/surface background
          border: '#e7e5e4',     // Borders
          hover: '#d6d3d1',      // Hover states
        },
        // Gold accent (buttons, links, highlights)
        gold: {
          DEFAULT: '#d4a854',
          50: '#fdfaf3',
          100: '#faf3e0',
          200: '#f4e4bd',
          300: '#ecd090',
          400: '#e2b85e',
          500: '#d4a854',
          600: '#b8860b',
          700: '#a06c2c',
          800: '#82552a',
          900: '#6b4626',
          950: '#3b2412',
        },
        // Keep accent same as gold
        accent: {
          DEFAULT: '#d4a854',
          light: '#f4e4bd',
          dark: '#b8860b',
        },
        50: '#fdfbf7',
        100: '#f9f3e8',
        200: '#f1e2c8',
        300: '#e8cca0',
        400: '#daa520',
        500: '#b8860b',
        600: '#996d0a',
        700: '#7a5708',
        800: '#5c4106',
        900: '#3d2b04',
      },
      // Neutral - Warm Stone (same across all apps)
      stone: {
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
        950: '#0c0a09',
      },
      // Override default neutral
      neutral: {
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
        950: '#0c0a09',
      },
    },
    spacing: {
      '18': '4.5rem',
      '22': '5.5rem',
    },
    borderRadius: {
      'sm': '0.25rem',
      'md': '0.375rem',
      'lg': '0.5rem',
      'xl': '0.75rem',
      '2xl': '1rem',
      '3xl': '1.5rem',
    },
    boxShadow: {
      'xs': '0 1px 2px rgba(0, 0, 0, 0.04)',
      'sm': '0 1px 3px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04)',
      'md': '0 4px 6px -1px rgba(0, 0, 0, 0.06), 0 2px 4px -1px rgba(0, 0, 0, 0.04)',
      'lg': '0 10px 15px -3px rgba(0, 0, 0, 0.06), 0 4px 6px -2px rgba(0, 0, 0, 0.03)',
      'xl': '0 20px 25px -5px rgba(0, 0, 0, 0.06), 0 10px 10px -5px rgba(0, 0, 0, 0.02)',
      'primary': '0 4px 14px rgba(28, 25, 23, 0.1)',
      'gold': '0 4px 14px rgba(212, 168, 84, 0.2)',
      'inner': 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.05)',
    },
    transitionDuration: {
      '150': '150ms',
      '200': '200ms',
      '300': '300ms',
    },
    transitionTimingFunction: {
      'smooth': 'cubic-bezier(0.4, 0, 0.2, 1)',
    },
    animation: {
      'fade-in': 'fadeIn 0.2s ease-out',
      'slide-up': 'slideUp 0.2s ease-out',
      'spin-slow': 'spin 1.5s linear infinite',
    },
    keyframes: {
      fadeIn: {
        '0%': { opacity: '0' },
        '100%': { opacity: '1' },
      },
      slideUp: {
        '0%': { opacity: '0', transform: 'translateY(8px)' },
        '100%': { opacity: '1', transform: 'translateY(0)' },
      },
    },
    typography: {
      DEFAULT: {
        css: {
          color: '#1c1917',
          a: {
            color: '#b8860b',
            '&:hover': {
              color: '#d4a854',
            },
          },
          h1: { color: '#1c1917' },
          h2: { color: '#1c1917' },
          h3: { color: '#1c1917' },
          h4: { color: '#1c1917' },
          strong: { color: '#1c1917' },
          code: {
            color: '#44403c',
            backgroundColor: '#f5f5f4',
            borderRadius: '0.25rem',
            padding: '0.125rem 0.25rem',
          },
        },
      },
    },
  },
},
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
};

