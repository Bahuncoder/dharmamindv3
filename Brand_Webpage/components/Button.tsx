import React from 'react';

interface ButtonProps {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'outline' | 'soft' | 'ghost' | 'danger' | 'enterprise' | 'contact' | 'gradient';
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  onClick?: () => void;
  disabled?: boolean;
  type?: 'button' | 'submit' | 'reset';
  className?: string;
  loading?: boolean;
  icon?: React.ReactNode;
  iconPosition?: 'left' | 'right';
  fullWidth?: boolean;
  animate?: boolean;
}

const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  onClick,
  disabled = false,
  type = 'button',
  className = '',
  loading = false,
  icon,
  iconPosition = 'left',
  fullWidth = false,
  animate = true
}) => {
  const baseClasses = `
    inline-flex items-center justify-center font-semibold rounded-xl 
    transition-all duration-300 ease-out
    focus:outline-none focus:ring-2 focus:ring-offset-2 
    disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none
    relative overflow-hidden group
    ${fullWidth ? 'w-full' : ''}
    ${animate && !disabled ? 'hover:-translate-y-0.5 active:translate-y-0 active:scale-[0.98]' : ''}
  `;
  
  const variants = {
    primary: `
      bg-gradient-to-r from-emerald-500 to-emerald-600 text-white
      shadow-lg shadow-emerald-500/25
      hover:shadow-xl hover:shadow-emerald-500/30
      focus:ring-emerald-500/50
    `,
    gradient: `
      bg-gradient-to-r from-emerald-500 via-emerald-600 to-teal-600 text-white
      shadow-lg shadow-emerald-500/25
      hover:shadow-xl hover:shadow-emerald-500/40
      focus:ring-emerald-500/50
    `,
    secondary: `
      bg-gray-100 text-gray-800 border-2 border-emerald-500
      hover:bg-gray-200 hover:border-emerald-600
      shadow-sm hover:shadow-md
      focus:ring-emerald-500/30
    `,
    outline: `
      bg-transparent text-emerald-600 border-2 border-emerald-500
      hover:bg-emerald-50 hover:border-emerald-600
      focus:ring-emerald-500/30
    `,
    soft: `
      bg-emerald-50 text-emerald-700 border border-emerald-200
      hover:bg-emerald-100 hover:border-emerald-300
      focus:ring-emerald-500/30
    `,
    contact: `
      bg-white text-gray-800 border-2 border-emerald-500
      hover:bg-emerald-50 hover:border-emerald-600
      shadow-sm hover:shadow-md
      focus:ring-emerald-500/30
    `,
    enterprise: `
      bg-gradient-to-r from-gray-800 to-gray-900 text-white
      border-2 border-emerald-500
      shadow-lg shadow-gray-900/25
      hover:shadow-xl hover:from-gray-700 hover:to-gray-800
      focus:ring-emerald-500/50
    `,
    ghost: `
      bg-transparent text-gray-600 
      hover:text-gray-900 hover:bg-gray-100
      focus:ring-gray-500/30
    `,
    danger: `
      bg-gradient-to-r from-red-500 to-red-600 text-white
      shadow-lg shadow-red-500/25
      hover:shadow-xl hover:shadow-red-500/30
      focus:ring-red-500/50
    `
  };

  const sizes = {
    xs: 'px-3 py-1.5 text-xs gap-1.5',
    sm: 'px-4 py-2 text-sm gap-2',
    md: 'px-5 py-2.5 text-sm gap-2',
    lg: 'px-6 py-3 text-base gap-2.5',
    xl: 'px-8 py-4 text-lg gap-3'
  };

  const iconSizes = {
    xs: 'w-3.5 h-3.5',
    sm: 'w-4 h-4',
    md: 'w-4 h-4',
    lg: 'w-5 h-5',
    xl: 'w-6 h-6'
  };

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled || loading}
      className={`${baseClasses} ${variants[variant]} ${sizes[size]} ${className}`}
    >
      {/* Shimmer effect */}
      {animate && !disabled && (
        <span className="absolute inset-0 -translate-x-full group-hover:translate-x-full transition-transform duration-700 bg-gradient-to-r from-transparent via-white/20 to-transparent" />
      )}
      
      {/* Loading spinner */}
      {loading && (
        <svg 
          className={`animate-spin ${iconSizes[size]} ${iconPosition === 'left' ? '-ml-0.5' : '-mr-0.5'}`} 
          fill="none" 
          viewBox="0 0 24 24"
        >
          <circle 
            className="opacity-25" 
            cx="12" cy="12" r="10" 
            stroke="currentColor" 
            strokeWidth="4"
          />
          <path 
            className="opacity-75" 
            fill="currentColor" 
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          />
        </svg>
      )}
      
      {/* Left icon */}
      {icon && iconPosition === 'left' && !loading && (
        <span className={`${iconSizes[size]} flex items-center justify-center transition-transform group-hover:scale-110`}>
          {icon}
        </span>
      )}
      
      {/* Button text */}
      <span className="relative z-10">{children}</span>
      
      {/* Right icon */}
      {icon && iconPosition === 'right' && !loading && (
        <span className={`${iconSizes[size]} flex items-center justify-center transition-transform group-hover:scale-110 group-hover:translate-x-0.5`}>
          {icon}
        </span>
      )}
    </button>
  );
};

export default Button;
