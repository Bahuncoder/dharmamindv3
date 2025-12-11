/**
 * DharmaMind Brand Webpage - Tailwind Config
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
      // Brand Website uses slightly darker backgrounds
      colors: {
        ui: {
          bg: '#f5f5f4',         // neutral-100 instead of neutral-50
          surface: '#e7e5e4',    // neutral-200 instead of neutral-100
          border: '#d6d3d1',     // neutral-300 instead of neutral-200
          hover: '#a8a29e',      // neutral-400 instead of neutral-300
        },
      },
    },
  },
  plugins: [],
};
