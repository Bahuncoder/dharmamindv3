import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Enhanced utility function to merge Tailwind CSS classes with conflict resolution
 * Combines clsx for conditional classes and tailwind-merge for conflict resolution
 * 
 * @param inputs - Array of class values (strings, objects, arrays, etc.)
 * @returns Merged and deduplicated class string
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Advanced class composition utility for complex component styling
 * Supports nested objects, arrays, and conditional logic
 * 
 * @param baseClasses - Base classes that should always be applied
 * @param conditionalClasses - Object with conditions as keys and classes as values
 * @param additionalClasses - Additional classes to merge
 * @returns Composed class string
 */
export function composeClasses(
  baseClasses: string,
  conditionalClasses: Record<string, boolean | undefined> = {},
  additionalClasses: string = ''
): string {
  const conditionalClassList = Object.entries(conditionalClasses)
    .filter(([, condition]) => condition)
    .map(([className]) => className);

  return cn(baseClasses, ...conditionalClassList, additionalClasses);
}

/**
 * Responsive class utility for breakpoint-specific styling
 * 
 * @param classes - Object with breakpoint keys and class values
 * @returns Responsive class string
 */
export function responsive(classes: {
  default?: string;
  sm?: string;
  md?: string;
  lg?: string;
  xl?: string;
  '2xl'?: string;
}): string {
  const responsiveClasses = [];
  
  if (classes.default) responsiveClasses.push(classes.default);
  if (classes.sm) responsiveClasses.push(`sm:${classes.sm}`);
  if (classes.md) responsiveClasses.push(`md:${classes.md}`);
  if (classes.lg) responsiveClasses.push(`lg:${classes.lg}`);
  if (classes.xl) responsiveClasses.push(`xl:${classes.xl}`);
  if (classes['2xl']) responsiveClasses.push(`2xl:${classes['2xl']}`);
  
  return responsiveClasses.join(' ');
}

/**
 * Color variant utility for consistent theming
 * 
 * @param variant - Color variant name
 * @param type - Type of styling (background, text, border, etc.)
 * @returns Color class string
 */
export function colorVariant(
  variant: 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'info' | 'spiritual',
  type: 'bg' | 'text' | 'border' | 'ring' = 'bg'
): string {
  const variants = {
    primary: {
      bg: 'bg-gray-500 hover:bg-gray-600',
      text: 'text-gray-600',
      border: 'border-emerald-500',
      ring: 'ring-emerald-500'
    },
    secondary: {
      bg: 'bg-gray-600 hover:bg-gray-700',
      text: 'text-gray-600',
      border: 'border-gray-600',
      ring: 'ring-gray-500'
    },
    success: {
      bg: 'bg-green-600 hover:bg-green-700',
      text: 'text-green-600',
      border: 'border-green-600',
      ring: 'ring-green-500'
    },
    warning: {
      bg: 'bg-yellow-600 hover:bg-yellow-700',
      text: 'text-yellow-600',
      border: 'border-yellow-600',
      ring: 'ring-yellow-500'
    },
    error: {
      bg: 'bg-red-600 hover:bg-red-700',
      text: 'text-red-600',
      border: 'border-red-600',
      ring: 'ring-red-500'
    },
    info: {
      bg: 'bg-blue-600 hover:bg-blue-700',
      text: 'text-blue-600',
      border: 'border-blue-600',
      ring: 'ring-blue-500'
    },
    spiritual: {
      bg: 'bg-purple-600 hover:bg-purple-700',
      text: 'text-purple-600',
      border: 'border-purple-600',
      ring: 'ring-purple-500'
    }
  };

  return variants[variant][type];
}

/**
 * Animation utility for consistent motion design
 * 
 * @param animation - Animation type
 * @returns Animation class string
 */
export function animation(
  animation: 'fadeIn' | 'slideUp' | 'slideDown' | 'scaleIn' | 'spin' | 'pulse' | 'bounce'
): string {
  const animations = {
    fadeIn: 'animate-fadeIn',
    slideUp: 'animate-slideUp',
    slideDown: 'animate-slideDown',
    scaleIn: 'animate-scaleIn',
    spin: 'animate-spin',
    pulse: 'animate-pulse',
    bounce: 'animate-bounce'
  };

  return animations[animation];
}

/**
 * Shadow utility for consistent elevation design
 * 
 * @param level - Shadow level (0-5)
 * @returns Shadow class string
 */
export function shadow(level: 0 | 1 | 2 | 3 | 4 | 5): string {
  const shadows = {
    0: 'shadow-none',
    1: 'shadow-sm',
    2: 'shadow-md',
    3: 'shadow-lg',
    4: 'shadow-xl',
    5: 'shadow-2xl'
  };

  return shadows[level];
}

/**
 * Spacing utility for consistent margin and padding
 * 
 * @param spacing - Spacing value
 * @param type - Type of spacing (margin, padding)
 * @param direction - Direction (all, x, y, t, r, b, l)
 * @returns Spacing class string
 */
export function spacing(
  spacing: 0 | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 32,
  type: 'm' | 'p' = 'p',
  direction: 'all' | 'x' | 'y' | 't' | 'r' | 'b' | 'l' = 'all'
): string {
  const directionMap = {
    all: '',
    x: 'x',
    y: 'y',
    t: 't',
    r: 'r',
    b: 'b',
    l: 'l'
  };

  return `${type}${directionMap[direction]}-${spacing}`;
}

export default cn;
