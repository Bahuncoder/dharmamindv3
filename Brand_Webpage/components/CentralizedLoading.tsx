/**
 * 🔄 DharmaMind Centralized Loading Components
 * 
 * Unified loading UI components to replace scattered loading implementations
 */

import React from 'react';
import Logo from './Logo';

// ===============================
// TYPES & INTERFACES
// ===============================

export interface LoadingSpinnerProps {
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  color?: 'primary' | 'secondary' | 'white' | 'gray';
  className?: string;
}

export interface LoadingButtonProps {
  isLoading: boolean;
  loadingText?: string;
  children: React.ReactNode;
  className?: string;
  disabled?: boolean;
  onClick?: () => void;
  type?: 'button' | 'submit';
}

export interface LoadingOverlayProps {
  isLoading: boolean;
  message?: string;
  progress?: number;
  children: React.ReactNode;
  className?: string;
  blur?: boolean;
}

export interface LoadingPageProps {
  message?: string;
  submessage?: string;
  showLogo?: boolean;
  progress?: number;
}

// ===============================
// LOADING SPINNER COMPONENT
// ===============================

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'md',
  color = 'primary',
  className = ''
}) => {
  const sizeClasses = {
    xs: 'w-3 h-3',
    sm: 'w-4 h-4', 
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12'
  };

  const colorClasses = {
    primary: 'border-amber-600 border-t-transparent',
<<<<<<< HEAD
    secondary: 'border-gold-600 border-t-transparent', 
    white: 'border-white border-t-transparent',
    gray: 'border-neutral-300 border-t-gray-600'
=======
    secondary: 'border-emerald-600 border-t-transparent', 
    white: 'border-white border-t-transparent',
    gray: 'border-gray-300 border-t-gray-600'
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
  };

  return (
    <div 
      className={`border-2 rounded-full animate-spin ${sizeClasses[size]} ${colorClasses[color]} ${className}`}
      role="status"
      aria-label="Loading"
    >
      <span className="sr-only">Loading...</span>
    </div>
  );
};

// ===============================
// LOADING BUTTON COMPONENT  
// ===============================

export const LoadingButton: React.FC<LoadingButtonProps> = ({
  isLoading,
  loadingText = 'Loading...',
  children,
  className = '',
  disabled = false,
  onClick,
  type = 'button'
}) => {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled || isLoading}
      className={`flex items-center justify-center space-x-2 transition-all duration-200 ${
        isLoading ? 'cursor-not-allowed opacity-75' : ''
      } ${className}`}
    >
      {isLoading && <LoadingSpinner size="sm" color="white" />}
      <span>{isLoading ? loadingText : children}</span>
    </button>
  );
};

// ===============================
// LOADING OVERLAY COMPONENT
// ===============================

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  isLoading,
  message = 'Loading...',
  progress,
  children,
  className = '',
  blur = true
}) => {
  return (
    <div className={`relative ${className}`}>
      {children}
      
      {isLoading && (
<<<<<<< HEAD
        <div className={`absolute inset-0 flex flex-col items-center justify-center bg-neutral-100 bg-opacity-90 z-50 ${
          blur ? 'backdrop-blur-sm' : ''
        }`}>
          <div className="flex flex-col items-center space-y-4 p-6 bg-neutral-100 rounded-lg shadow-lg border border-neutral-300">
            <LoadingSpinner size="lg" color="primary" />
            
            {message && (
              <p className="text-sm font-medium text-neutral-900 text-center">
=======
        <div className={`absolute inset-0 flex flex-col items-center justify-center bg-white bg-opacity-90 z-50 ${
          blur ? 'backdrop-blur-sm' : ''
        }`}>
          <div className="flex flex-col items-center space-y-4 p-6 bg-white rounded-lg shadow-lg border border-gray-200">
            <LoadingSpinner size="lg" color="primary" />
            
            {message && (
              <p className="text-sm font-medium text-gray-700 text-center">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                {message}
              </p>
            )}
            
            {progress !== undefined && (
              <div className="w-48">
<<<<<<< HEAD
                <div className="w-full bg-primary-background rounded-full h-2">
                  <div 
                    className="bg-gradient-to-r from-gold-600 to-gold-700 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
                  />
                </div>
                <p className="text-xs text-neutral-600 text-center mt-1">
=======
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-gradient-to-r from-amber-600 to-emerald-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
                  />
                </div>
                <p className="text-xs text-gray-500 text-center mt-1">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                  {Math.round(progress)}% complete
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

// ===============================
// FULL PAGE LOADING COMPONENT
// ===============================

export const LoadingPage: React.FC<LoadingPageProps> = ({
  message = 'Loading your spiritual companion...',
  submessage,
  showLogo = true,
  progress
}) => {
  return (
    <div className="min-h-screen bg-primary-background-main flex items-center justify-center">
      <div className="text-center max-w-md mx-auto px-6">
        
        {showLogo && (
          <div className="flex justify-center mb-8">
            <Logo 
              size="xl" 
              showText={false}
              className="animate-pulse"
            />
          </div>
        )}
        
<<<<<<< HEAD
        <h1 className="text-2xl font-bold mb-2 text-neutral-800">DharmaMind</h1>
        
        <p className="text-neutral-600 mb-6">{message}</p>
        
        {submessage && (
          <p className="text-sm text-neutral-500 mb-6">{submessage}</p>
=======
        <h1 className="text-2xl font-bold mb-2 text-stone-800">DharmaMind</h1>
        
        <p className="text-stone-600 mb-6">{message}</p>
        
        {submessage && (
          <p className="text-sm text-stone-500 mb-6">{submessage}</p>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
        )}
        
        <div className="flex justify-center mb-6">
          <LoadingSpinner size="lg" color="primary" />
        </div>
        
        {progress !== undefined && (
          <div className="w-full max-w-xs mx-auto">
<<<<<<< HEAD
            <div className="w-full bg-neutral-200 rounded-full h-2">
              <div 
                className="bg-gradient-to-r from-gold-600 to-gold-700 h-2 rounded-full transition-all duration-500"
                style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
              />
            </div>
            <p className="text-xs text-neutral-500 text-center mt-2">
=======
            <div className="w-full bg-stone-200 rounded-full h-2">
              <div 
                className="bg-gradient-to-r from-amber-600 to-emerald-600 h-2 rounded-full transition-all duration-500"
                style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
              />
            </div>
            <p className="text-xs text-stone-500 text-center mt-2">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
              {Math.round(progress)}% complete
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

// ===============================
// LOADING CARD COMPONENT
// ===============================

export const LoadingCard: React.FC<{
  message?: string;
  className?: string;
}> = ({
  message = 'Loading...',
  className = ''
}) => {
  return (
<<<<<<< HEAD
    <div className={`bg-neutral-100 rounded-lg shadow-sm border border-neutral-300 p-8 ${className}`}>
      <div className="flex flex-col items-center space-y-4">
        <LoadingSpinner size="lg" color="primary" />
        <p className="text-sm font-medium text-neutral-600">{message}</p>
=======
    <div className={`bg-white rounded-lg shadow-sm border border-gray-200 p-8 ${className}`}>
      <div className="flex flex-col items-center space-y-4">
        <LoadingSpinner size="lg" color="primary" />
        <p className="text-sm font-medium text-gray-600">{message}</p>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
      </div>
    </div>
  );
};

// ===============================
// LOADING SKELETON COMPONENT
// ===============================

export const LoadingSkeleton: React.FC<{
  lines?: number;
  avatar?: boolean;
  className?: string;
}> = ({
  lines = 3,
  avatar = false,
  className = ''
}) => {
  return (
    <div className={`animate-pulse ${className}`}>
      {avatar && (
        <div className="flex items-center space-x-4 mb-4">
<<<<<<< HEAD
          <div className="w-10 h-10 bg-primary-background rounded-full" />
          <div className="flex-1 space-y-2">
            <div className="h-4 bg-primary-background rounded w-1/4" />
            <div className="h-3 bg-primary-background rounded w-1/3" />
=======
          <div className="w-10 h-10 bg-gray-300 rounded-full" />
          <div className="flex-1 space-y-2">
            <div className="h-4 bg-gray-300 rounded w-1/4" />
            <div className="h-3 bg-gray-300 rounded w-1/3" />
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
          </div>
        </div>
      )}
      
      <div className="space-y-3">
        {Array.from({ length: lines }, (_, i) => (
          <div 
            key={i}
<<<<<<< HEAD
            className={`h-4 bg-primary-background rounded ${
=======
            className={`h-4 bg-gray-300 rounded ${
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
              i === lines - 1 ? 'w-2/3' : 'w-full'
            }`}
          />
        ))}
      </div>
    </div>
  );
};

// ===============================
// INLINE LOADING COMPONENT
// ===============================

export const InlineLoading: React.FC<{
  message?: string;
  size?: 'sm' | 'md';
  className?: string;
}> = ({
  message = 'Loading...',
  size = 'sm',
  className = ''
}) => {
  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      <LoadingSpinner size={size} color="primary" />
<<<<<<< HEAD
      <span className={`${size === 'sm' ? 'text-sm' : 'text-base'} text-neutral-600`}>
=======
      <span className={`${size === 'sm' ? 'text-sm' : 'text-base'} text-gray-600`}>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
        {message}
      </span>
    </div>
  );
};
