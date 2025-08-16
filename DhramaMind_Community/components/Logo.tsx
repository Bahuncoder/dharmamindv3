import React from 'react';
import Image from 'next/image';
import Link from 'next/link';

interface LogoProps {
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | 'avatar';
  showText?: boolean;
  onClick?: () => void;
  href?: string; // For making logo clickable
  external?: boolean; // For external links
  className?: string;
}

const Logo: React.FC<LogoProps> = ({ 
  size = 'md', 
  showText = true, 
  onClick, 
  href,
  external = false,
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
      <div className={`${currentSize.container} relative overflow-hidden rounded-${size === 'avatar' ? 'full' : 'lg'} shadow-md border-2 border-border-primary bg-white`}>
        <Image
          src="/logo.jpeg"
          alt="DharmaMind Logo"
          fill
          className="object-contain p-1"
          priority
          quality={100}
          sizes={`${currentSize.container}`}
          style={{
            objectFit: 'contain'
          }}
        />
      </div>
      {showText && (
        <div className="ml-3">
          <div className={`font-black text-primary ${currentSize.titleText} tracking-tight`}>
            DharmaMind
          </div>
          <div className={`font-semibold text-secondary ${size === 'xl' ? 'text-sm' : 'text-xs'} -mt-1 tracking-wide`}>
            AI with Soul Powered by Dharma
          </div>
        </div>
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

  if (href) {
    if (external) {
      return (
        <a 
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className={`flex items-center hover:opacity-80 transition-opacity ${className}`}
        >
          <LogoContent />
        </a>
      );
    } else {
      return (
        <Link 
          href={href}
          className={`flex items-center hover:opacity-80 transition-opacity ${className}`}
        >
          <LogoContent />
        </Link>
      );
    }
  }

  return (
    <div className={`flex items-center ${className}`}>
      <LogoContent />
    </div>
  );
};

export default Logo;
