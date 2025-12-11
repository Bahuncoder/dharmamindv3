import React, { useState, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import FeedbackModal from './FeedbackModal';
import { useColors } from '../contexts/ColorContext';
import { cn } from '../utils/cn';

// Enhanced interfaces for comprehensive feedback system
interface FeedbackButtonProps {
  conversationId?: string;
  messageId?: string;
  variant?: 'floating' | 'inline' | 'compact' | 'minimal' | 'spiritual';
  position?: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left' | 'center';
  size?: 'sm' | 'md' | 'lg' | 'xl';
  color?: 'primary' | 'secondary' | 'spiritual' | 'success' | 'warning' | 'error';
  icon?: 'message' | 'heart' | 'star' | 'thumbs' | 'custom';
  customIcon?: React.ReactNode;
  label?: string;
  showBadge?: boolean;
  badgeCount?: number;
  autoHide?: boolean;
  hideDelay?: number;
  tooltipText?: string;
  disabled?: boolean;
  className?: string;
  analytics?: {
    category?: string;
    action?: string;
    label?: string;
  };
  onFeedbackSubmitted?: (feedback: any) => void;
  zIndex?: number;
}

// Complex styling configurations for different variants
const variantConfigurations = {
  floating: {
    base: 'fixed btn-primary shadow-lg hover:shadow-xl transform transition-all duration-300 hover:scale-105 backdrop-blur-sm border border-card-border',
    colors: {
      primary: 'btn-primary',
      secondary: 'btn-secondary',
      spiritual: 'btn-primary',
      success: 'btn-primary',
      warning: 'btn-primary',
      error: 'btn-primary'
    }
  },
  inline: {
    base: 'btn-primary transition-all duration-300 hover:scale-102 inline-flex',
    colors: {
      primary: 'btn-primary',
      secondary: 'btn-secondary',
      spiritual: 'btn-primary',
      success: 'btn-primary',
      warning: 'btn-primary',
      error: 'btn-primary'
    }
  },
  compact: {
    base: 'transition-all duration-300 hover:scale-102 inline-flex border border-card-border',
    colors: {
<<<<<<< HEAD
      primary: 'card-background text-neutral-900 hover:bg-primary hover:text-white',
      secondary: 'card-background text-neutral-600 hover:bg-secondary hover:text-white',
      spiritual: 'card-background text-neutral-900 hover:bg-primary hover:text-white',
      success: 'card-background text-neutral-900 hover:bg-primary hover:text-white',
      warning: 'card-background text-neutral-900 hover:bg-primary hover:text-white',
      error: 'card-background text-neutral-900 hover:bg-primary hover:text-white'
=======
      primary: 'card-background text-primary hover:bg-primary hover:text-white',
      secondary: 'card-background text-secondary hover:bg-secondary hover:text-white',
      spiritual: 'card-background text-primary hover:bg-primary hover:text-white',
      success: 'card-background text-primary hover:bg-primary hover:text-white',
      warning: 'card-background text-primary hover:bg-primary hover:text-white',
      error: 'card-background text-primary hover:bg-primary hover:text-white'
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
    }
  },
  minimal: {
    base: 'transition-all duration-300 hover:scale-105 bg-transparent border-none',
    colors: {
<<<<<<< HEAD
      primary: 'text-neutral-900 hover:text-gold-600-dark hover:bg-page-background',
      secondary: 'text-neutral-600 hover:text-gold-600-dark hover:bg-page-background',
      spiritual: 'text-neutral-900 hover:text-gold-600-dark hover:bg-page-background',
      success: 'text-neutral-900 hover:text-gold-600-dark hover:bg-page-background',
      warning: 'text-neutral-900 hover:text-gold-600-dark hover:bg-page-background',
      error: 'text-neutral-900 hover:text-gold-600-dark hover:bg-page-background'
=======
      primary: 'text-primary hover:text-primary-dark hover:bg-page-background',
      secondary: 'text-secondary hover:text-secondary-dark hover:bg-page-background',
      spiritual: 'text-primary hover:text-primary-dark hover:bg-page-background',
      success: 'text-primary hover:text-primary-dark hover:bg-page-background',
      warning: 'text-primary hover:text-primary-dark hover:bg-page-background',
      error: 'text-primary hover:text-primary-dark hover:bg-page-background'
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
    }
  },
  spiritual: {
    base: 'relative overflow-hidden btn-primary shadow-lg hover:shadow-xl transform transition-all duration-500',
    colors: {
      primary: 'btn-primary',
      secondary: 'btn-secondary',
      spiritual: 'btn-primary',
      success: 'btn-primary',
      warning: 'btn-primary',
      error: 'btn-primary'
    }
  }
};

// Position configurations for floating variant
const positionConfigurations = {
  'bottom-right': 'bottom-6 right-6',
  'bottom-left': 'bottom-6 left-6',
  'top-right': 'top-6 right-6',
  'top-left': 'top-6 left-6',
  'center': 'top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2'
};

// Size configurations
const sizeConfigurations = {
  sm: {
    padding: 'px-2 py-1',
    text: 'text-xs',
    icon: 'w-3 h-3',
    gap: 'gap-1',
    rounded: 'rounded-md'
  },
  md: {
    padding: 'px-3 py-2',
    text: 'text-sm',
    icon: 'w-4 h-4',
    gap: 'gap-2',
    rounded: 'rounded-lg'
  },
  lg: {
    padding: 'px-4 py-3',
    text: 'text-base',
    icon: 'w-5 h-5',
    gap: 'gap-2',
    rounded: 'rounded-lg'
  },
  xl: {
    padding: 'px-6 py-4',
    text: 'text-lg',
    icon: 'w-6 h-6',
    gap: 'gap-3',
    rounded: 'rounded-xl'
  }
};

// Icon library for different feedback types
const iconLibrary = {
  message: (className: string) => (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
      />
    </svg>
  ),
  heart: (className: string) => (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"
      />
    </svg>
  ),
  star: (className: string) => (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z"
      />
    </svg>
  ),
  thumbs: (className: string) => (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H2v-9a2 2 0 012-2h3v11z"
      />
    </svg>
  )
};

// Enhanced Badge Component
const FeedbackBadge: React.FC<{ count: number; size: string }> = ({ count, size }) => {
  const badgeSize = size === 'sm' ? 'w-4 h-4 text-xs' : size === 'md' ? 'w-5 h-5 text-xs' : 'w-6 h-6 text-sm';
  
  return (
    <motion.div
      className={cn(
        'absolute -top-1 -right-1 bg-accent text-white rounded-full flex items-center justify-center font-bold',
        badgeSize
      )}
      initial={{ scale: 0 }}
      animate={{ scale: 1 }}
      transition={{ type: "spring", stiffness: 500, damping: 30 }}
    >
      {count > 99 ? '99+' : count}
    </motion.div>
  );
};

// Enhanced Tooltip Component
const FeedbackTooltip: React.FC<{ text: string; children: React.ReactNode }> = ({ text, children }) => {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <div
      className="relative"
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {children}
      <AnimatePresence>
        {isVisible && (
          <motion.div
<<<<<<< HEAD
            className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 card-background border border-card-border text-neutral-900 text-sm rounded-lg whitespace-nowrap z-50"
=======
            className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 card-background border border-card-border text-primary text-sm rounded-lg whitespace-nowrap z-50"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
            initial={{ opacity: 0, y: 10, scale: 0.8 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.8 }}
            transition={{ duration: 0.2 }}
          >
            {text}
            <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-card-border"></div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

// Main Enhanced Feedback Button Component
const FeedbackButton: React.FC<FeedbackButtonProps> = ({
  conversationId,
  messageId,
  variant = 'floating',
  position = 'bottom-right',
  size = 'md',
  color = 'primary',
  icon = 'message',
  customIcon,
  label,
  showBadge = false,
  badgeCount = 0,
  autoHide = false,
  hideDelay = 3000,
  tooltipText,
  disabled = false,
  className = '',
  analytics,
  onFeedbackSubmitted,
  zIndex = 40
}) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isVisible, setIsVisible] = useState(true);

  // Memoized configurations for performance optimization
  const variantConfig = useMemo(() => variantConfigurations[variant], [variant]);
  const colorConfig = useMemo(() => variantConfig.colors[color], [variantConfig, color]);
  const sizeConfig = useMemo(() => sizeConfigurations[size], [size]);
  const positionClass = useMemo(() => 
    variant === 'floating' ? positionConfigurations[position] : '', 
    [variant, position]
  );

  // Enhanced click handler with analytics
  const handleClick = useCallback(() => {
    if (disabled) return;

    // Analytics tracking
    if (analytics) {
      if (typeof window !== 'undefined' && (window as any).gtag) {
        (window as any).gtag('event', analytics.action || 'feedback_button_click', {
          event_category: analytics.category || 'engagement',
          event_label: analytics.label || `${variant}_${color}`
        });
      }
    }

    setIsModalOpen(true);
  }, [disabled, analytics, variant, color]);

  // Auto-hide functionality
  React.useEffect(() => {
    if (autoHide && hideDelay > 0) {
      const timer = setTimeout(() => setIsVisible(false), hideDelay);
      return () => clearTimeout(timer);
    }
  }, [autoHide, hideDelay]);

  // Feedback submission handler
  const handleFeedbackSubmitted = useCallback((feedback: any) => {
    onFeedbackSubmitted?.(feedback);
    setIsModalOpen(false);
  }, [onFeedbackSubmitted]);

  // Complex class composition
  const buttonClasses = useMemo(() => {
    const baseClasses = [
      'relative',
      'flex',
      'items-center',
      'justify-center',
      'font-medium',
      'cursor-pointer',
      'focus:outline-none',
      'focus:ring-4',
      'focus:ring-offset-2',
      'select-none',
      'transition-all',
      'duration-300'
    ];

    if (disabled) {
      baseClasses.push('opacity-50', 'cursor-not-allowed');
    }

    return cn(
      baseClasses,
      variantConfig.base,
      colorConfig,
      sizeConfig.padding,
      sizeConfig.text,
      sizeConfig.gap,
      sizeConfig.rounded,
      positionClass,
      className
    );
  }, [
    variantConfig,
    colorConfig,
    sizeConfig,
    positionClass,
    disabled,
    className
  ]);

  // Render icon based on type or custom icon
  const renderIcon = useCallback(() => {
    if (customIcon) {
      return <span className={sizeConfig.icon}>{customIcon}</span>;
    }
    
    if (icon === 'custom') return null;
    
    const iconFunction = iconLibrary[icon as keyof typeof iconLibrary];
    return iconFunction ? iconFunction(sizeConfig.icon) : null;
  }, [customIcon, icon, sizeConfig.icon]);

  // Spiritual variant special effects
  const spiritualEffects = variant === 'spiritual' && (
    <>
      <motion.div
        className="absolute inset-0 bg-gradient-to-r from-purple-400/20 via-pink-400/20 to-blue-400/20"
        animate={{
          x: ['-100%', '100%'],
        }}
        transition={{
          duration: 3,
          repeat: Infinity,
          ease: "linear"
        }}
      />
      <motion.div
<<<<<<< HEAD
        className="absolute inset-0 bg-neutral-100/10"
=======
        className="absolute inset-0 bg-white/10"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
        animate={{
          opacity: [0, 0.3, 0],
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      />
    </>
  );

  // Motion variants for different animations
  const motionVariants = {
    floating: {
      initial: { opacity: 0, y: 20, scale: 0.8 },
      animate: { opacity: 1, y: 0, scale: 1 },
      exit: { opacity: 0, y: 20, scale: 0.8 },
      hover: { scale: 1.05, y: -2 },
      tap: { scale: 0.95 }
    },
    inline: {
      initial: { opacity: 0, x: -20 },
      animate: { opacity: 1, x: 0 },
      exit: { opacity: 0, x: -20 },
      hover: { scale: 1.02 },
      tap: { scale: 0.98 }
    },
    compact: {
      initial: { opacity: 0 },
      animate: { opacity: 1 },
      exit: { opacity: 0 },
      hover: { scale: 1.02 },
      tap: { scale: 0.98 }
    },
    minimal: {
      initial: { opacity: 0 },
      animate: { opacity: 1 },
      exit: { opacity: 0 },
      hover: { scale: 1.05 },
      tap: { scale: 0.95 }
    },
    spiritual: {
      initial: { opacity: 0, scale: 0.5, rotate: -180 },
      animate: { opacity: 1, scale: 1, rotate: 0 },
      exit: { opacity: 0, scale: 0.5, rotate: 180 },
      hover: { scale: 1.1, rotate: 5 },
      tap: { scale: 0.9, rotate: -5 }
    }
  };

  const currentVariant = motionVariants[variant];

  if (!isVisible) return null;

  const buttonContent = (
    <AnimatePresence>
      <motion.button
        className={buttonClasses}
        onClick={handleClick}
        disabled={disabled}
        style={{ zIndex }}
        variants={currentVariant}
        initial="initial"
        animate="animate"
        exit="exit"
        whileHover="hover"
        whileTap="tap"
        transition={{
          type: "spring",
          stiffness: 400,
          damping: 25,
          delay: variant === 'floating' ? 0.5 : 0
        }}
      >
        {spiritualEffects}
        
        <div className="relative z-10 flex items-center gap-2">
          {renderIcon()}
          {label && <span>{label}</span>}
          {!label && variant !== 'minimal' && (
            <span>
              {variant === 'compact' ? 'Feedback' : 'Share Feedback'}
            </span>
          )}
        </div>

        {showBadge && badgeCount > 0 && (
          <FeedbackBadge count={badgeCount} size={size} />
        )}
      </motion.button>
    </AnimatePresence>
  );

  return (
    <>
      {tooltipText ? (
        <FeedbackTooltip text={tooltipText}>
          {buttonContent}
        </FeedbackTooltip>
      ) : (
        buttonContent
      )}

      <FeedbackModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        conversationId={conversationId}
        messageId={messageId}
      />
    </>
  );
};

export default FeedbackButton;
export type { FeedbackButtonProps };
