import React from 'react';
import Image from 'next/image';

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
      image: 'w-6 h-6',
      imageSize: 24,
      text: 'text-xs',
      titleText: 'text-sm'
    },
    sm: {
      container: 'w-8 h-8',
      image: 'w-8 h-8',
      imageSize: 32,
      text: 'text-sm',
      titleText: 'text-lg'
    },
    md: {
      container: 'w-10 h-10',
      image: 'w-10 h-10',
      imageSize: 40,
      text: 'text-lg',
      titleText: 'text-xl'
    },
    lg: {
      container: 'w-12 h-12',
      image: 'w-12 h-12',
      imageSize: 48,
      text: 'text-xl',
      titleText: 'text-2xl'
    },
    xl: {
      container: 'w-16 h-16',
      image: 'w-16 h-16',
      imageSize: 64,
      text: 'text-2xl',
      titleText: 'text-3xl'
    },
    avatar: {
      container: 'w-8 h-8',
      image: 'w-8 h-8',
      imageSize: 32,
      text: 'text-sm',
      titleText: 'text-lg'
    }
  };

  const currentSize = sizeClasses[size];

  const LogoContent = () => (
    <>
      <div className={`${currentSize.container} rounded-${size === 'avatar' ? 'full' : 'lg'} overflow-hidden shadow-lg bg-neutral-100 border border-neutral-300 relative`}>
        <Image
          src="/logo.jpeg"
          alt="DharmaMind Logo"
          width={currentSize.imageSize}
          height={currentSize.imageSize}
          className={`${currentSize.image} object-contain filter contrast-125 saturate-125 brightness-105 hue-rotate-0`}
          style={{
            imageRendering: 'crisp-edges',
            filter: 'contrast(1.3) saturate(1.2) brightness(1.05) sharpen(1.5)'
          }}
          priority
          quality={100}
        />
        {/* Gold accent border at bottom */}
        <div className="absolute bottom-0 left-0 right-0 h-1 bg-gold-600"></div>
      </div>
      {showText && (
        <span className={`font-bold text-neutral-900 ${currentSize.titleText} ml-3 tracking-tight drop-shadow-sm`}>
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
