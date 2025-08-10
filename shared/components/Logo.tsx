import React from 'react';

interface LogoProps {
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | 'avatar';
  showText?: boolean;
  onClick?: () => void;
  className?: string;
}

const Logo: React.FC<LogoProps> = ({ 
  size = 'md', 
  showText = true, 
  onClick, 
  className = '' 
}) => {
  const sizeClasses = {
    xs: {
      container: 'w-6 h-6',
      text: 'text-xs',
      titleText: 'text-sm'
    },
    sm: {
      container: 'w-8 h-8', 
      text: 'text-sm',
      titleText: 'text-lg'
    },
    md: {
      container: 'w-10 h-10',
      text: 'text-lg', 
      titleText: 'text-xl'
    },
    lg: {
      container: 'w-12 h-12',
      text: 'text-xl',
      titleText: 'text-2xl'
    },
    xl: {
      container: 'w-16 h-16',
      text: 'text-2xl',
      titleText: 'text-3xl'
    },
    avatar: {
      container: 'w-8 h-8',
      text: 'text-sm',
      titleText: 'text-lg'
    }
  };

  const currentSize = sizeClasses[size];
  
  const LogoContent = () => (
    <>
      <div className={`${currentSize.container} bg-gradient-to-r from-amber-600 to-emerald-600 rounded-${size === 'avatar' ? 'full' : 'lg'} flex items-center justify-center shadow-md`}>
        <span className={`text-white font-bold ${currentSize.text}`}>🕉</span>
      </div>
      {showText && (
        <span className={`font-semibold text-stone-800 ${currentSize.titleText} ml-3`}>
          DharmaMind
        </span>
      )}
    </>
  );

  if (onClick) {
    return (
      <button 
        onClick={onClick}
        className={`flex items-center hover:opacity-80 transition-opacity ${className}`}
      >
        <LogoContent />
      </button>
    );
  }

  return (
    <div className={`flex items-center ${className}`}>
      <LogoContent />
    </div>
  );
};

export default Logo;
