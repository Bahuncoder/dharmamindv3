/**
 * DharmaMind - Professional Guide Avatar
 * Clean, minimal avatar component for AI guides
 */

import React from 'react';

// Professional color scheme for guides
export const RISHI_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  marici: { bg: 'bg-amber-50', text: 'text-amber-700', border: 'border-amber-200' },
  atri: { bg: 'bg-indigo-50', text: 'text-indigo-700', border: 'border-indigo-200' },
  angiras: { bg: 'bg-orange-50', text: 'text-orange-700', border: 'border-orange-200' },
  pulastya: { bg: 'bg-blue-50', text: 'text-blue-700', border: 'border-blue-200' },
  pulaha: { bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-200' },
  kratu: { bg: 'bg-red-50', text: 'text-red-700', border: 'border-red-200' },
  bhrigu: { bg: 'bg-purple-50', text: 'text-purple-700', border: 'border-purple-200' },
  vasishta: { bg: 'bg-teal-50', text: 'text-teal-700', border: 'border-teal-200' },
  daksha: { bg: 'bg-slate-50', text: 'text-slate-700', border: 'border-slate-200' },
  default: { bg: 'bg-gray-50', text: 'text-gray-700', border: 'border-gray-200' },
};

// For backwards compatibility
export const RISHI_SACRED_COLORS = RISHI_COLORS;

interface RishiAvatarProps {
  rishiId: string;
  rishiName?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  animated?: boolean;
  showGlow?: boolean;
  className?: string;
}

const sizeClasses = {
  sm: 'w-8 h-8 text-sm',
  md: 'w-10 h-10 text-base',
  lg: 'w-12 h-12 text-lg',
  xl: 'w-16 h-16 text-xl',
};

export const RishiAvatar: React.FC<RishiAvatarProps> = ({
  rishiId,
  rishiName,
  size = 'md',
  animated = false,
  showGlow = false,
  className = '',
}) => {
  const colors = RISHI_COLORS[rishiId] || RISHI_COLORS.default;
  const displayName = rishiName || rishiId;
  const initial = displayName.charAt(0).toUpperCase();

  return (
    <div
      className={`
        ${sizeClasses[size]}
        ${colors.bg}
        ${colors.text}
        rounded-full
        flex items-center justify-center
        font-semibold
        border
        ${colors.border}
        ${showGlow ? 'ring-2 ring-offset-2 ring-teal-500/30' : ''}
        ${animated ? 'transition-transform hover:scale-105' : ''}
        ${className}
      `}
    >
      {initial}
    </div>
  );
};

export default RishiAvatar;
