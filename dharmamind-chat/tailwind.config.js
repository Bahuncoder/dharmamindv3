/** 
 * 🕉️ DharmaMind Chat (dharmamind.ai)
 * Tailwind CSS Configuration
 * 
 * Purpose: Personal AI Spiritual Companion
 * Accent: Sacred Emerald (#10b981)
 * 
 * @type {import('tailwindcss').Config} 
 */

const dmPreset = require('../shared/tailwind.preset.js');

module.exports = {
  presets: [dmPreset],
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      // Chat specific overrides
      colors: {
        // Product-specific accent (Sacred Emerald for spiritual depth)
        'product-accent': {
          DEFAULT: '#10b981',
          light: '#d1fae5',
          dark: '#047857',
        },
        // Chat-specific color mappings
        'chat-bg': {
          primary: '#fafaf9',
          secondary: '#f5f5f4',
          dark: '#1c1917',
        },
        'chat-text': {
          primary: '#1c1917',
          secondary: '#57534e',
          muted: '#78716c',
        },
      },
      backgroundImage: {
        // Chat specific gradients
        'hero-gradient': 'linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(13, 148, 136, 0.05) 100%)',
        'chat-gradient': 'linear-gradient(135deg, #10b981 0%, #0d9488 100%)',
        'rishi-gradient': 'linear-gradient(135deg, #10b981 0%, #8b5cf6 100%)',
        'message-gradient': 'linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%)',
      },
      boxShadow: {
        'chat': '0 4px 20px rgba(16, 185, 129, 0.15)',
        'chat-lg': '0 10px 40px rgba(16, 185, 129, 0.2)',
        'message': '0 2px 8px rgba(0, 0, 0, 0.05)',
        'rishi': '0 4px 20px rgba(139, 92, 246, 0.2)',
      },
      // Additional chat animations
      animation: {
        'typing': 'typing 1.4s ease-in-out infinite',
        'message-in': 'messageIn 0.3s ease-out forwards',
      },
      keyframes: {
        typing: {
          '0%, 100%': { opacity: '0.3' },
          '50%': { opacity: '1' },
        },
        messageIn: {
          '0%': { opacity: '0', transform: 'translateY(10px) scale(0.95)' },
          '100%': { opacity: '1', transform: 'translateY(0) scale(1)' },
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
};
