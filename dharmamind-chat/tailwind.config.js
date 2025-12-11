/**
 * DharmaMind Chat - Tailwind Config
 * Uses shared unified design system
 */

const sharedPreset = require('../shared/tailwind/preset');

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: 'class',
  presets: [sharedPreset],
  theme: {
    extend: {
      // Chat-specific extensions (if any)
    },
  },
  plugins: [],
};
