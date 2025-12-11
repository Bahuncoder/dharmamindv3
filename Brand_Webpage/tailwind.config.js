<<<<<<< HEAD
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
        // Override Tailwind's default primary color to emerald green
        primary: {
          50: '#ecfdf5',
          100: '#d1fae5',
          200: '#a7f3d0',
          300: '#6ee7b7',
          400: '#34d399',
          500: '#10b981',  // Our main emerald green
          600: '#059669',
          700: '#047857',
          800: '#065f46',
          900: '#064e3b',
          950: '#022c22',
        },
        dharma: {
          50: '#f0fdf4',  // emerald-50
          100: '#dcfce7', // emerald-100
          200: '#bbf7d0', // emerald-200
          300: '#86efac', // emerald-300
          400: '#4ade80', // emerald-400
          500: '#22c55e', // emerald-500
          600: '#16a34a', // emerald-600
          700: '#15803d', // emerald-700
          800: '#166534', // emerald-800
          900: '#14532d', // emerald-900
          950: '#052e16', // emerald-950
        },
        lotus: {
          50: '#f9fafb',  // gray-50
          100: '#f3f4f6', // gray-100
          200: '#e5e7eb', // gray-200
          300: '#d1d5db', // gray-300
          400: '#9ca3af', // gray-400
          500: '#6b7280', // gray-500
          600: '#4b5563', // gray-600
          700: '#374151', // gray-700
          800: '#1f2937', // gray-800
          900: '#111827', // gray-900
          950: '#030712', // gray-950
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
