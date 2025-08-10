import React from 'react';
import { useRouter } from 'next/router';
import Image from 'next/image';
import { useAuth } from '../contexts/AuthContext';

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
  const router = useRouter();
  const { user, isAuthenticated } = useAuth();

  const handleClick = () => {
    if (onClick) {
      onClick();
    } else {
      // Always navigate to chat page when logo is clicked
      router.push('/chat');
    }
  };

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
      <div className={`${currentSize.container} rounded-${size === 'avatar' ? 'full' : 'lg'} flex items-center justify-center overflow-hidden relative`}
           style={{
             background: 'linear-gradient(135deg, var(--color-accent), var(--color-accent-hover))',
             border: '3px solid var(--color-logo-emerald)',
             boxShadow: '0 4px 16px rgba(16, 185, 129, 0.3), 0 2px 8px rgba(249, 115, 22, 0.2)'
           }}>
        <Image
          src="/logo.jpeg"
          alt="DharmaMind Logo"
          width={parseInt(currentSize.container.split('-')[1]) * 4}
          height={parseInt(currentSize.container.split('-')[1]) * 4}
          className="object-cover rounded-inherit relative z-10"
          priority
        />
        {/* Professional overlay gradient */}
        <div 
          className="absolute inset-0 rounded-inherit"
          style={{
            background: 'linear-gradient(135deg, rgba(249, 115, 22, 0.1), rgba(16, 185, 129, 0.1))',
            zIndex: 5
          }}
        />
      </div>
      {showText && (
        <div className="ml-3">
          <span className={`font-semibold ${currentSize.titleText}`}
                style={{color: 'var(--color-text-primary)'}}>
            DharmaMind-AI with Soul
          </span>
          {isAuthenticated && user && (
            <div className={`${size === 'xs' || size === 'sm' ? 'text-xs' : 'text-sm'} font-medium`}
                 style={{color: 'var(--color-emerald)'}}>
              Current: {user.subscription_plan.charAt(0).toUpperCase() + user.subscription_plan.slice(1)}
            </div>
          )}
        </div>
      )}
    </>
  );

  return (
    <button 
      onClick={handleClick}
      className={`flex items-center transition-all duration-300 hover:scale-105 group ${className}`}
      style={{
        filter: 'hover:drop-shadow(0 8px 16px rgba(16, 185, 129, 0.4))'
      }}
    >
      <LogoContent />
    </button>
  );
};

export default Logo;
