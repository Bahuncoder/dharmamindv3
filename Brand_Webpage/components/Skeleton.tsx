/**
 * 🦴 Skeleton Loading Components
 * ==============================
 * 
 * Fast, beautiful skeleton loaders for content placeholders
 * - Shimmer animation effect
 * - Various preset shapes
 * - Dark mode support
 */

import React from 'react';

// Base shimmer animation class
const shimmerClass = `
  relative overflow-hidden
  before:absolute before:inset-0
  before:-translate-x-full
  before:animate-[shimmer_1.5s_infinite]
  before:bg-gradient-to-r
  before:from-transparent before:via-white/20 before:to-transparent
`;

interface SkeletonProps {
  className?: string;
  animate?: boolean;
}

// Basic skeleton block
export const Skeleton: React.FC<SkeletonProps & { 
  width?: string | number;
  height?: string | number;
  rounded?: 'none' | 'sm' | 'md' | 'lg' | 'full';
}> = ({ 
  className = '', 
  animate = true,
  width,
  height,
  rounded = 'md'
}) => {
  const roundedClasses = {
    none: '',
    sm: 'rounded-sm',
    md: 'rounded-md',
    lg: 'rounded-lg',
    full: 'rounded-full'
  };

  return (
    <div
      className={`
        bg-neutral-200 dark:bg-neutral-700
        ${roundedClasses[rounded]}
        ${animate ? shimmerClass : ''}
        ${className}
      `}
      style={{ width, height }}
    />
  );
};

// Text line skeleton
export const SkeletonText: React.FC<SkeletonProps & {
  lines?: number;
  lastLineWidth?: string;
}> = ({ 
  className = '', 
  animate = true,
  lines = 3,
  lastLineWidth = '60%'
}) => (
  <div className={`space-y-2 ${className}`}>
    {Array.from({ length: lines }).map((_, i) => (
      <Skeleton
        key={i}
        height={16}
        width={i === lines - 1 ? lastLineWidth : '100%'}
        animate={animate}
      />
    ))}
  </div>
);

// Avatar skeleton
export const SkeletonAvatar: React.FC<SkeletonProps & {
  size?: 'sm' | 'md' | 'lg' | 'xl';
}> = ({ 
  className = '', 
  animate = true,
  size = 'md'
}) => {
  const sizeClasses = {
    sm: 'w-8 h-8',
    md: 'w-10 h-10',
    lg: 'w-12 h-12',
    xl: 'w-16 h-16'
  };

  return (
    <Skeleton
      rounded="full"
      animate={animate}
      className={`${sizeClasses[size]} ${className}`}
    />
  );
};

// Card skeleton
export const SkeletonCard: React.FC<SkeletonProps & {
  hasImage?: boolean;
  imageHeight?: number;
}> = ({ 
  className = '', 
  animate = true,
  hasImage = true,
  imageHeight = 200
}) => (
  <div className={`bg-white dark:bg-neutral-800 rounded-xl border border-neutral-200 dark:border-neutral-700 overflow-hidden ${className}`}>
    {hasImage && (
      <Skeleton height={imageHeight} animate={animate} rounded="none" className="w-full" />
    )}
    <div className="p-4 space-y-3">
      <Skeleton height={20} animate={animate} className="w-3/4" />
      <SkeletonText lines={2} animate={animate} />
      <div className="flex items-center gap-2 pt-2">
        <SkeletonAvatar size="sm" animate={animate} />
        <Skeleton height={14} animate={animate} className="w-24" />
      </div>
    </div>
  </div>
);

// Table row skeleton
export const SkeletonTableRow: React.FC<SkeletonProps & {
  columns?: number;
}> = ({ 
  className = '', 
  animate = true,
  columns = 4
}) => (
  <div className={`flex items-center gap-4 py-3 px-4 border-b border-neutral-200 dark:border-neutral-700 ${className}`}>
    {Array.from({ length: columns }).map((_, i) => (
      <Skeleton
        key={i}
        height={16}
        animate={animate}
        className={i === 0 ? 'w-1/4' : 'flex-1'}
      />
    ))}
  </div>
);

// List item skeleton
export const SkeletonListItem: React.FC<SkeletonProps & {
  hasAvatar?: boolean;
  hasAction?: boolean;
}> = ({ 
  className = '', 
  animate = true,
  hasAvatar = true,
  hasAction = false
}) => (
  <div className={`flex items-center gap-3 py-3 ${className}`}>
    {hasAvatar && <SkeletonAvatar size="md" animate={animate} />}
    <div className="flex-1 space-y-2">
      <Skeleton height={16} animate={animate} className="w-2/3" />
      <Skeleton height={12} animate={animate} className="w-1/2" />
    </div>
    {hasAction && <Skeleton height={32} animate={animate} className="w-20" rounded="lg" />}
  </div>
);

// Header skeleton
export const SkeletonHeader: React.FC<SkeletonProps> = ({ 
  className = '', 
  animate = true
}) => (
  <div className={`flex items-center justify-between py-4 ${className}`}>
    <div className="flex items-center gap-3">
      <Skeleton height={32} width={32} animate={animate} rounded="lg" />
      <Skeleton height={24} width={120} animate={animate} />
    </div>
    <div className="flex items-center gap-4">
      <Skeleton height={16} width={60} animate={animate} />
      <Skeleton height={16} width={60} animate={animate} />
      <Skeleton height={36} width={100} animate={animate} rounded="lg" />
    </div>
  </div>
);

// Page skeleton (full page loading)
export const SkeletonPage: React.FC<SkeletonProps & {
  hasHeader?: boolean;
  hasSidebar?: boolean;
}> = ({ 
  className = '', 
  animate = true,
  hasHeader = true,
  hasSidebar = false
}) => (
  <div className={`min-h-screen bg-neutral-50 dark:bg-neutral-900 ${className}`}>
    {hasHeader && (
      <div className="border-b border-neutral-200 dark:border-neutral-700 px-6">
        <SkeletonHeader animate={animate} />
      </div>
    )}
    <div className="flex">
      {hasSidebar && (
        <div className="w-64 border-r border-neutral-200 dark:border-neutral-700 p-4 space-y-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <SkeletonListItem key={i} animate={animate} hasAvatar={false} />
          ))}
        </div>
      )}
      <div className="flex-1 p-6">
        <div className="max-w-4xl mx-auto space-y-6">
          <Skeleton height={40} animate={animate} className="w-1/3" />
          <SkeletonText lines={4} animate={animate} />
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pt-4">
            {Array.from({ length: 3 }).map((_, i) => (
              <SkeletonCard key={i} animate={animate} imageHeight={150} />
            ))}
          </div>
        </div>
      </div>
    </div>
  </div>
);

// Pricing card skeleton
export const SkeletonPricingCard: React.FC<SkeletonProps> = ({ 
  className = '', 
  animate = true
}) => (
  <div className={`bg-white dark:bg-neutral-800 rounded-xl border border-neutral-200 dark:border-neutral-700 p-6 ${className}`}>
    <Skeleton height={24} animate={animate} className="w-1/3 mb-2" />
    <Skeleton height={40} animate={animate} className="w-1/2 mb-4" />
    <Skeleton height={16} animate={animate} className="w-full mb-6" />
    <div className="space-y-3 mb-6">
      {Array.from({ length: 4 }).map((_, i) => (
        <div key={i} className="flex items-center gap-2">
          <Skeleton height={16} width={16} animate={animate} rounded="full" />
          <Skeleton height={14} animate={animate} className="flex-1" />
        </div>
      ))}
    </div>
    <Skeleton height={44} animate={animate} rounded="lg" className="w-full" />
  </div>
);

export default {
  Skeleton,
  SkeletonText,
  SkeletonAvatar,
  SkeletonCard,
  SkeletonTableRow,
  SkeletonListItem,
  SkeletonHeader,
  SkeletonPage,
  SkeletonPricingCard
};
