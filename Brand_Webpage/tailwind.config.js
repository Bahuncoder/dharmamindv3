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
