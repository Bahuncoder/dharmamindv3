/**
 * DharmaMind Community - Tailwind Config
 * Minimalist Design: Grayscale + Gold Accent
 */

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        // Brand colors - Minimalist Grayscale + Gold
        // Dark gray ONLY for text, light grays for UI, gold for accents
        brand: {
          primary: '#1c1917',    // Near Black - TEXT ONLY
          secondary: '#d4a854',  // Warm Gold - buttons, accents, CTAs
          accent: '#d4a854',     // Warm Gold - highlights
          light: '#78716c',      // Medium Gray - secondary text
          muted: '#a8a29e',      // Light Gray - placeholder text
        },
        // Text colors (dark grays)
        text: {
          primary: '#1c1917',    // Main text
          secondary: '#57534e',  // Secondary text
          muted: '#78716c',      // Muted/placeholder
          light: '#a8a29e',      // Very light text
        },
        // UI colors (light grays for backgrounds, borders)
        ui: {
          bg: '#fafaf9',         // Page background
          surface: '#f5f5f4',    // Card/surface background
          border: '#e7e5e4',     // Borders
          hover: '#d6d3d1',      // Hover states
        },
        // Gold accent (buttons, links, highlights)
        gold: {
          DEFAULT: '#d4a854',
          light: '#f4e4bd',
          dark: '#b8860b',
        },
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
      fontFamily: {
        sans: ['system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
