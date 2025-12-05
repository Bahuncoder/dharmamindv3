/** 
 * 🕉️ DharmaMind Brand Website (dharmamind.com)
 * Tailwind CSS Configuration
 * 
 * Purpose: Enterprise & Authority positioning
 * Accent: Premium Purple (#8b5cf6)
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
      // Brand Website specific overrides
      colors: {
        // Product-specific accent (Premium Purple for enterprise)
        'product-accent': {
          DEFAULT: '#8b5cf6',
          light: '#ede9fe',
          dark: '#5b21b6',
        },
      },
      backgroundImage: {
        // Brand page specific gradients
        'hero-gradient': 'linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, rgba(139, 92, 246, 0.1) 50%, rgba(245, 158, 11, 0.05) 100%)',
        'enterprise-gradient': 'linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%)',
        'cta-gradient': 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
      },
      boxShadow: {
        'hero': '0 25px 50px -12px rgba(16, 185, 129, 0.15), 0 10px 25px -5px rgba(139, 92, 246, 0.1)',
        'card-hover': '0 20px 40px rgba(16, 185, 129, 0.15), 0 5px 15px rgba(0, 0, 0, 0.05)',
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
};
