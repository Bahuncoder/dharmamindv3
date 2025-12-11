/**
 * DharmaMind Brand Webpage - Tailwind Config
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
        // Dark gray ONLY for text, medium grays for UI, gold for accents
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
        // UI colors (medium grays for backgrounds, borders)
        ui: {
          bg: '#f5f5f4',         // Page background (darker)
          surface: '#e7e5e4',    // Card/surface background (darker)
          border: '#d6d3d1',     // Borders (darker)
          hover: '#a8a29e',      // Hover states (darker)
        },
        // Gold accent (buttons, links, highlights)
        gold: {
          50: '#fefce8',
          100: '#fef9c3',
          200: '#fef08a',
          300: '#fde047',
          400: '#facc15',
          500: '#eab308',
          600: '#d4a854',         // Main gold
          700: '#b8860b',         // Dark gold
          800: '#854d0e',
          900: '#713f12',
          DEFAULT: '#d4a854',
          light: '#f4e4bd',
          dark: '#b8860b',
        },
        // Override neutral with warm grays (medium tones)
        neutral: {
          50: '#f5f5f4',         // Was #fafaf9 - now slightly darker
          100: '#e7e5e4',        // Was #f5f5f4 - now medium light
          200: '#d6d3d1',        // Was #e7e5e4 - now medium
          300: '#a8a29e',        // Was #d6d3d1 - now medium dark
          400: '#78716c',        // Was #a8a29e 
          500: '#57534e',        // Was #78716c
          600: '#44403c',        // Was #57534e
          700: '#292524',        // Was #44403c
          800: '#1c1917',        // Was #292524
          900: '#0c0a09',        // Was #1c1917
          950: '#0a0908',
        },
      },
      fontFamily: {
        sans: ['system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
      borderRadius: {
        sm: '0.25rem',
        DEFAULT: '0.5rem',
        md: '0.5rem',
        lg: '0.75rem',
        xl: '1rem',
        '2xl': '1.5rem',
      },
      boxShadow: {
        sm: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
        DEFAULT: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
        md: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
        lg: '0 10px 15px -3px rgb(0 0 0 / 0.1)',
      },
    },
  },
  plugins: [],
};
