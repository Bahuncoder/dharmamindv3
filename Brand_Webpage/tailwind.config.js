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
        // PRIMARY: Light Gray (main backgrounds, cards, etc.)
        primary: {
          50: '#f9fafb',   // Lightest gray
          100: '#f3f4f6',  // Main primary color (light gray)
          200: '#e5e7eb',  // Hover states
          300: '#d1d5db',  // Darker gray
          400: '#9ca3af',  // Medium gray
          500: '#6b7280',  // Text gray
          600: '#4b5563',  // Secondary text
          700: '#374151',  // Dark text
          800: '#1f2937',  // Main text
          900: '#111827',  // Darkest
        },

        // ACCENT: Emerald (ONLY for borders and highlights)
        accent: {
          50: '#ecfdf5',   // Lightest emerald
          100: '#d1fae5',  // Light emerald borders
          200: '#a7f3d0',  // 
          300: '#6ee7b7',  //
          400: '#34d399',  //
          500: '#10b981',  // Main accent color (emerald for borders)
          600: '#059669',  // Border hover
          700: '#047857',  // Dark borders
          800: '#065f46',  //
          900: '#064e3b',  //
          950: '#022c22',  // Darkest
        },

        // Keep dharma, lotus, peace for backward compatibility but simplified
        dharma: {
          500: '#10b981',  // Just emerald for borders
        },
        lotus: {
          100: '#f3f4f6',  // Just light gray
          500: '#6b7280',  // Just medium gray for text
        },
        peace: {
          100: '#f3f4f6',  // Map to light gray
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
