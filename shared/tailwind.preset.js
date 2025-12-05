/**
 * ═══════════════════════════════════════════════════════════════════════════
 * 🕉️ DHARMAMIND UNIVERSAL TAILWIND PRESET
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * Shared Tailwind configuration for all DharmaMind products:
 * - dharmamind.ai (Chat)
 * - dharmamind.org (Community)
 * - dharmamind.com (Brand Website)
 * 
 * Import this preset in each product's tailwind.config.js:
 * const dmPreset = require('../shared/tailwind.preset.js');
 * module.exports = { presets: [dmPreset], ... }
 */

module.exports = {
  theme: {
    extend: {
      colors: {
        // ═══════════════════════════════════════════════════════════════
        // 🎨 BRAND CORE COLORS
        // ═══════════════════════════════════════════════════════════════
        
        // Primary Brand - Sacred Emerald (Heart Chakra, Growth, Trust)
        brand: {
          DEFAULT: '#10b981',
          50: '#ecfdf5',
          100: '#d1fae5',
          200: '#a7f3d0',
          300: '#6ee7b7',
          400: '#34d399',
          500: '#10b981',
          600: '#059669',
          700: '#047857',
          800: '#065f46',
          900: '#064e3b',
          950: '#022c22',
        },
        
        // Secondary Brand - Sacred Gold (Wisdom, Spirituality, Excellence)
        gold: {
          DEFAULT: '#f59e0b',
          50: '#fffbeb',
          100: '#fef3c7',
          200: '#fde68a',
          300: '#fcd34d',
          400: '#fbbf24',
          500: '#f59e0b',
          600: '#d97706',
          700: '#b45309',
          800: '#92400e',
          900: '#78350f',
        },
        
        // Tertiary Brand - Spiritual Purple (Crown Chakra, Premium)
        spiritual: {
          DEFAULT: '#8b5cf6',
          50: '#f5f3ff',
          100: '#ede9fe',
          200: '#ddd6fe',
          300: '#c4b5fd',
          400: '#a78bfa',
          500: '#8b5cf6',
          600: '#7c3aed',
          700: '#6d28d9',
          800: '#5b21b6',
          900: '#4c1d95',
        },
        
        // Quaternary Brand - Wisdom Teal (Third Eye, Clarity)
        wisdom: {
          DEFAULT: '#0d9488',
          50: '#f0fdfa',
          100: '#ccfbf1',
          200: '#99f6e4',
          300: '#5eead4',
          400: '#2dd4bf',
          500: '#14b8a6',
          600: '#0d9488',
          700: '#0f766e',
          800: '#115e59',
          900: '#134e4a',
        },
        
        // ═══════════════════════════════════════════════════════════════
        // 🌓 NEUTRAL COLORS (Earthy, Grounded)
        // ═══════════════════════════════════════════════════════════════
        
        earth: {
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
        
        // ═══════════════════════════════════════════════════════════════
        // 📱 PRODUCT-SPECIFIC ACCENTS
        // ═══════════════════════════════════════════════════════════════
        
        // Chat accent (dharmamind.ai) - Deep Emerald
        'chat-accent': {
          DEFAULT: '#10b981',
          light: '#d1fae5',
          dark: '#047857',
        },
        
        // Community accent (dharmamind.org) - Warm Gold
        'community-accent': {
          DEFAULT: '#f59e0b',
          light: '#fef3c7',
          dark: '#b45309',
        },
        
        // Brand accent (dharmamind.com) - Premium Purple
        'brand-accent': {
          DEFAULT: '#8b5cf6',
          light: '#ede9fe',
          dark: '#5b21b6',
        },
        
        // ═══════════════════════════════════════════════════════════════
        // 🎭 SEMANTIC COLORS
        // ═══════════════════════════════════════════════════════════════
        
        success: {
          DEFAULT: '#10b981',
          light: '#d1fae5',
          dark: '#047857',
        },
        warning: {
          DEFAULT: '#f59e0b',
          light: '#fef3c7',
          dark: '#b45309',
        },
        error: {
          DEFAULT: '#ef4444',
          light: '#fee2e2',
          dark: '#b91c1c',
        },
        info: {
          DEFAULT: '#0d9488',
          light: '#ccfbf1',
          dark: '#0f766e',
        },
        
        // ═══════════════════════════════════════════════════════════════
        // 🔄 LEGACY ALIASES (backwards compatibility)
        // ═══════════════════════════════════════════════════════════════
        
        dharma: {
          50: '#ecfdf5',
          100: '#d1fae5',
          200: '#a7f3d0',
          300: '#6ee7b7',
          400: '#34d399',
          500: '#10b981',
          600: '#059669',
          700: '#047857',
          800: '#065f46',
          900: '#064e3b',
        },
        accent: {
          50: '#ecfdf5',
          100: '#d1fae5',
          200: '#a7f3d0',
          300: '#6ee7b7',
          400: '#34d399',
          500: '#10b981',
          600: '#059669',
          700: '#047857',
          800: '#065f46',
          900: '#064e3b',
        },
        primary: {
          50: '#f9fafb',
          100: '#f3f4f6',
          200: '#e5e7eb',
          300: '#d1d5db',
          400: '#9ca3af',
          500: '#6b7280',
          600: '#4b5563',
          700: '#374151',
          800: '#1f2937',
          900: '#111827',
        },
      },
      
      // ═══════════════════════════════════════════════════════════════
      // 🔤 TYPOGRAPHY
      // ═══════════════════════════════════════════════════════════════
      
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        serif: ['Crimson Text', 'Georgia', 'serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        sanskrit: ['Noto Sans Devanagari', 'Poppins', 'sans-serif'],
      },
      
      fontSize: {
        '2xs': ['0.625rem', { lineHeight: '0.875rem' }],
      },
      
      // ═══════════════════════════════════════════════════════════════
      // 📐 SPACING & SIZING
      // ═══════════════════════════════════════════════════════════════
      
      borderRadius: {
        '4xl': '2rem',
        '5xl': '2.5rem',
      },
      
      // ═══════════════════════════════════════════════════════════════
      // 🌫️ SHADOWS
      // ═══════════════════════════════════════════════════════════════
      
      boxShadow: {
        'brand': '0 4px 20px rgba(16, 185, 129, 0.25)',
        'brand-lg': '0 10px 40px rgba(16, 185, 129, 0.3)',
        'gold': '0 4px 20px rgba(245, 158, 11, 0.25)',
        'gold-lg': '0 10px 40px rgba(245, 158, 11, 0.3)',
        'spiritual': '0 4px 20px rgba(139, 92, 246, 0.25)',
        'spiritual-lg': '0 10px 40px rgba(139, 92, 246, 0.3)',
        'glow': '0 0 30px rgba(16, 185, 129, 0.15)',
        'glow-gold': '0 0 30px rgba(245, 158, 11, 0.15)',
        'inner-glow': 'inset 0 0 20px rgba(16, 185, 129, 0.1)',
      },
      
      // ═══════════════════════════════════════════════════════════════
      // ✨ GRADIENTS
      // ═══════════════════════════════════════════════════════════════
      
      backgroundImage: {
        // Brand Gradients
        'gradient-brand': 'linear-gradient(135deg, #10b981 0%, #0f766e 100%)',
        'gradient-gold': 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
        'gradient-spiritual': 'linear-gradient(135deg, #8b5cf6 0%, #10b981 100%)',
        'gradient-premium': 'linear-gradient(135deg, #f59e0b 0%, #8b5cf6 100%)',
        'gradient-cosmos': 'linear-gradient(135deg, #1c1917 0%, #4c1d95 50%, #064e3b 100%)',
        'gradient-sunrise': 'linear-gradient(135deg, #fef3c7 0%, #d1fae5 50%, #ede9fe 100%)',
        
        // Mesh Gradients
        'mesh-brand': `
          radial-gradient(at 40% 20%, rgba(16, 185, 129, 0.15) 0px, transparent 50%),
          radial-gradient(at 80% 0%, rgba(245, 158, 11, 0.1) 0px, transparent 50%),
          radial-gradient(at 0% 50%, rgba(139, 92, 246, 0.08) 0px, transparent 50%),
          radial-gradient(at 80% 50%, rgba(16, 185, 129, 0.1) 0px, transparent 50%),
          radial-gradient(at 0% 100%, rgba(245, 158, 11, 0.1) 0px, transparent 50%)
        `,
        
        // Radial & Conic
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
      },
      
      // ═══════════════════════════════════════════════════════════════
      // 🎭 ANIMATIONS
      // ═══════════════════════════════════════════════════════════════
      
      animation: {
        // Fade animations
        'fade-in': 'fadeIn 0.5s ease-out',
        'fade-in-up': 'fadeInUp 0.5s ease-out forwards',
        'fade-in-down': 'fadeInDown 0.5s ease-out forwards',
        
        // Slide animations
        'slide-up': 'slideUp 0.3s ease-out',
        'slide-down': 'slideDown 0.3s ease-out',
        'slide-left': 'slideLeft 0.3s ease-out',
        'slide-right': 'slideRight 0.3s ease-out',
        
        // Scale animations
        'scale-in': 'scaleIn 0.3s ease-out',
        'scale-out': 'scaleOut 0.3s ease-out',
        
        // Special effects
        'float': 'float 3s ease-in-out infinite',
        'pulse-glow': 'pulseGlow 2s ease-in-out infinite',
        'breathe': 'breathe 4s ease-in-out infinite',
        'shimmer': 'shimmer 2s infinite linear',
        'spin-slow': 'spin 8s linear infinite',
        
        // Sacred animations
        'lotus-bloom': 'lotusBloom 3s ease-in-out infinite',
        'chakra-glow': 'chakraGlow 4s ease-in-out infinite',
        'om-pulse': 'omPulse 2s ease-in-out infinite',
      },
      
      keyframes: {
        // Fade keyframes
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        fadeInDown: {
          '0%': { opacity: '0', transform: 'translateY(-20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        
        // Slide keyframes
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideDown: {
          '0%': { transform: 'translateY(-20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideLeft: {
          '0%': { transform: 'translateX(20px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        slideRight: {
          '0%': { transform: 'translateX(-20px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        
        // Scale keyframes
        scaleIn: {
          '0%': { opacity: '0', transform: 'scale(0.95)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        scaleOut: {
          '0%': { opacity: '1', transform: 'scale(1)' },
          '100%': { opacity: '0', transform: 'scale(0.95)' },
        },
        
        // Special effect keyframes
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        pulseGlow: {
          '0%, 100%': { boxShadow: '0 4px 20px rgba(16, 185, 129, 0.25)' },
          '50%': { boxShadow: '0 8px 40px rgba(16, 185, 129, 0.4)' },
        },
        breathe: {
          '0%, 100%': { transform: 'scale(1)', opacity: '0.8' },
          '50%': { transform: 'scale(1.05)', opacity: '1' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        
        // Sacred keyframes
        lotusBloom: {
          '0%, 100%': { transform: 'scale(1) rotate(0deg)', opacity: '0.9' },
          '50%': { transform: 'scale(1.05) rotate(2deg)', opacity: '1' },
        },
        chakraGlow: {
          '0%, 100%': { filter: 'drop-shadow(0 0 5px rgba(16, 185, 129, 0.3))' },
          '50%': { filter: 'drop-shadow(0 0 20px rgba(245, 158, 11, 0.4))' },
        },
        omPulse: {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(245, 158, 11, 0.4)' },
          '50%': { boxShadow: '0 0 0 20px rgba(245, 158, 11, 0)' },
        },
      },
      
      // ═══════════════════════════════════════════════════════════════
      // ⏱️ TRANSITIONS
      // ═══════════════════════════════════════════════════════════════
      
      transitionTimingFunction: {
        'bounce': 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
        'breath': 'cubic-bezier(0.4, 0, 0.2, 1)',
      },
      
      transitionDuration: {
        '400': '400ms',
        '600': '600ms',
        '800': '800ms',
        '2000': '2000ms',
        '4000': '4000ms',
      },
      
      // ═══════════════════════════════════════════════════════════════
      // 🔲 BACKDROP
      // ═══════════════════════════════════════════════════════════════
      
      backdropBlur: {
        xs: '2px',
        '3xl': '64px',
      },
    },
  },
  
  plugins: [],
};

