import React from 'react';

interface LoadingSpinnerProps {
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  variant?: 'default' | 'dharmic' | 'pulse' | 'dots';
  text?: string;
  className?: string;
  fullScreen?: boolean;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'md',
  variant = 'dharmic',
  text,
  className = '',
  fullScreen = false
}) => {
  const sizes = {
    xs: { spinner: 'w-4 h-4', text: 'text-xs', container: 'gap-2' },
    sm: { spinner: 'w-6 h-6', text: 'text-sm', container: 'gap-2' },
    md: { spinner: 'w-10 h-10', text: 'text-base', container: 'gap-3' },
    lg: { spinner: 'w-14 h-14', text: 'text-lg', container: 'gap-4' },
    xl: { spinner: 'w-20 h-20', text: 'text-xl', container: 'gap-5' }
  };

  const DefaultSpinner = () => (
    <div className={`${sizes[size].spinner} relative`}>
      <div className="absolute inset-0 rounded-full border-4 border-gray-200"></div>
      <div className="absolute inset-0 rounded-full border-4 border-emerald-500 border-t-transparent animate-spin"></div>
    </div>
  );

  const DharmicSpinner = () => (
    <div className={`${sizes[size].spinner} relative`}>
      {/* Outer ring */}
      <div className="absolute inset-0 rounded-full border-4 border-emerald-200 animate-pulse"></div>
      
      {/* Middle spinning ring */}
      <div className="absolute inset-1 rounded-full border-4 border-transparent border-t-emerald-500 border-r-emerald-500 animate-spin"></div>
      
      {/* Inner spinning ring (opposite direction) */}
      <div 
        className="absolute inset-2 rounded-full border-2 border-transparent border-b-emerald-400 border-l-emerald-400 animate-spin"
        style={{ animationDirection: 'reverse', animationDuration: '0.75s' }}
      ></div>
      
      {/* Center Om symbol */}
      <div className="absolute inset-0 flex items-center justify-center">
        <span 
          className="text-emerald-600 animate-pulse"
          style={{ fontSize: size === 'xs' ? '0.5rem' : size === 'sm' ? '0.625rem' : size === 'md' ? '0.875rem' : size === 'lg' ? '1.125rem' : '1.5rem' }}
        >
          ॐ
        </span>
      </div>
      
      {/* Glow effect */}
      <div className="absolute inset-0 rounded-full bg-emerald-500/10 blur-md animate-pulse"></div>
    </div>
  );

  const PulseSpinner = () => (
    <div className={`${sizes[size].spinner} relative flex items-center justify-center`}>
      {/* Ripple effects */}
      <div className="absolute inset-0 rounded-full bg-emerald-500/30 animate-ping"></div>
      <div 
        className="absolute inset-2 rounded-full bg-emerald-500/40 animate-ping"
        style={{ animationDelay: '0.2s' }}
      ></div>
      <div 
        className="absolute inset-4 rounded-full bg-emerald-500/50 animate-ping"
        style={{ animationDelay: '0.4s' }}
      ></div>
      
      {/* Center dot */}
      <div className="relative w-2 h-2 rounded-full bg-emerald-600"></div>
    </div>
  );

  const DotsSpinner = () => (
    <div className={`flex items-center ${sizes[size].container}`}>
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className={`${size === 'xs' ? 'w-1.5 h-1.5' : size === 'sm' ? 'w-2 h-2' : size === 'md' ? 'w-3 h-3' : size === 'lg' ? 'w-4 h-4' : 'w-5 h-5'} 
            rounded-full bg-emerald-500 animate-bounce`}
          style={{ animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </div>
  );

  const spinners = {
    default: <DefaultSpinner />,
    dharmic: <DharmicSpinner />,
    pulse: <PulseSpinner />,
    dots: <DotsSpinner />
  };

  const content = (
    <div className={`flex flex-col items-center ${sizes[size].container} ${className}`}>
      {spinners[variant]}
      
      {text && (
        <p className={`${sizes[size].text} text-gray-600 font-medium animate-pulse`}>
          {text}
        </p>
      )}
    </div>
  );

  if (fullScreen) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-white/80 backdrop-blur-sm">
        {content}
      </div>
    );
  }

  return content;
};

// Page loading component with beautiful animation
export const LoadingPage: React.FC<{ message?: string; submessage?: string }> = ({ 
  message = 'Loading...',
  submessage
}) => (
  <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-b from-gray-50 to-white">
    {/* Background orbs */}
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-emerald-200/30 rounded-full blur-3xl animate-pulse"></div>
      <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-emerald-300/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
    </div>
    
    <div className="relative z-10 flex flex-col items-center">
      {/* Large dharmic spinner */}
      <div className="mb-8">
        <LoadingSpinner size="xl" variant="dharmic" />
      </div>
      
      {/* Loading text */}
      <h2 className="text-2xl font-bold text-gray-800 mb-2 animate-fade-in">
        {message}
      </h2>
      
      {submessage && (
        <p className="text-gray-500 animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
          {submessage}
        </p>
      )}
      
      {/* Progress dots */}
      <div className="mt-6">
        <LoadingSpinner size="sm" variant="dots" />
      </div>
    </div>
  </div>
);

// Skeleton loader for content
export const Skeleton: React.FC<{ 
  className?: string;
  variant?: 'text' | 'circular' | 'rectangular';
  width?: string;
  height?: string;
}> = ({ 
  className = '', 
  variant = 'rectangular',
  width,
  height
}) => {
  const baseClasses = 'bg-gray-200 animate-pulse';
  
  const variants = {
    text: 'rounded',
    circular: 'rounded-full',
    rectangular: 'rounded-lg'
  };

  return (
    <div 
      className={`${baseClasses} ${variants[variant]} ${className}`}
      style={{ width, height }}
    />
  );
};

export default LoadingSpinner;
