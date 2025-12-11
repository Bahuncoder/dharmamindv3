<<<<<<< HEAD
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
=======
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: 'class', // Enable class-based dark mode
  theme: {
    extend: {
      colors: {
        dharma: {
          50: '#fdf8f0',
          100: '#faf0e1',
          200: '#f4dfc2',
          300: '#ebc899',
          400: '#e0aa6e',
          500: '#d69143',
          600: '#c87d39',
          700: '#a66730',
          800: '#85532c',
          900: '#6e4528',
          950: '#3b2213',
        },
        lotus: {
          50: '#fef7f0',
          100: '#fdede0',
          200: '#fad9c0',
          300: '#f6bf95',
          400: '#f09968',
          500: '#ea7944',
          600: '#db5f2a',
          700: '#b64920',
          800: '#913b1f',
          900: '#74331e',
          950: '#3e170d',
        },
        peace: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
          950: '#082f49',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        serif: ['Crimson Text', 'serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'lotus-bloom': 'lotusBloom 2s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        lotusBloom: {
          '0%, 100%': { transform: 'scale(1)' },
          '50%': { transform: 'scale(1.05)' },
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
}
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
