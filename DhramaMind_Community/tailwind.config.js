/** 
 * 🕉️ DharmaMind Community (dharmamind.org)
 * Tailwind CSS Configuration
 * 
 * Purpose: Digital Sangha - Connection & Engagement
 * Accent: Warm Gold (#f59e0b)
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
      // Community specific overrides
      colors: {
        // Product-specific accent (Warm Gold for community warmth)
        'product-accent': {
          DEFAULT: '#f59e0b',
          light: '#fef3c7',
          dark: '#b45309',
        },
      },
      backgroundImage: {
        // Community specific gradients
        'hero-gradient': 'linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, rgba(245, 158, 11, 0.1) 50%, rgba(16, 185, 129, 0.05) 100%)',
        'community-gradient': 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
        'connect-gradient': 'linear-gradient(135deg, #10b981 0%, #f59e0b 100%)',
      },
      boxShadow: {
        'community': '0 25px 50px -12px rgba(245, 158, 11, 0.15), 0 10px 25px -5px rgba(16, 185, 129, 0.1)',
        'card-warm': '0 20px 40px rgba(245, 158, 11, 0.15), 0 5px 15px rgba(0, 0, 0, 0.05)',
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
};
