import React from 'react';

interface ButtonProps {
  children: React.ReactNode;
<<<<<<< HEAD
  variant?: 'primary' | 'secondary' | 'outline' | 'soft' | 'ghost' | 'danger' | 'enterprise' | 'contact' | 'gradient';
=======
  variant?: 'primary' | 'secondary' | 'outline' | 'soft' | 'ghost' | 'danger' | 'enterprise' | 'contact';
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  onClick?: () => void;
  disabled?: boolean;
  type?: 'button' | 'submit' | 'reset';
  className?: string;
  loading?: boolean;
<<<<<<< HEAD
  icon?: React.ReactNode;
  iconPosition?: 'left' | 'right';
  fullWidth?: boolean;
  animate?: boolean;
=======
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
}

const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  onClick,
  disabled = false,
  type = 'button',
  className = '',
<<<<<<< HEAD
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
      bg-gradient-to-r from-gold-600 to-gold-700 text-white
      shadow-lg shadow-gold-500/25
      hover:shadow-xl hover:shadow-gold-500/30
      focus:ring-gold-500/50
    `,
    gradient: `
      bg-gradient-to-r from-gold-500 via-gold-600 to-gold-700 text-white
      shadow-lg shadow-gold-500/25
      hover:shadow-xl hover:shadow-gold-500/40
      focus:ring-gold-500/50
    `,
    secondary: `
      bg-neutral-100 text-neutral-800 border-2 border-gold-500
      hover:bg-neutral-200 hover:border-gold-600
      shadow-sm hover:shadow-md
      focus:ring-gold-500/30
    `,
    outline: `
      bg-transparent text-gold-600 border-2 border-gold-500
      hover:bg-gold-50 hover:border-gold-600
      focus:ring-gold-500/30
    `,
    soft: `
      bg-gold-50 text-gold-700 border border-gold-200
      hover:bg-gold-100 hover:border-gold-300
      focus:ring-gold-500/30
    `,
    contact: `
      bg-neutral-100 text-neutral-800 border-2 border-gold-500
      hover:bg-gold-50 hover:border-gold-600
      shadow-sm hover:shadow-md
      focus:ring-gold-500/30
    `,
    enterprise: `
      bg-gradient-to-r from-gray-800 to-gray-900 text-white
      border-2 border-gold-500
      shadow-lg shadow-gray-900/25
      hover:shadow-xl hover:from-gray-700 hover:to-gray-800
      focus:ring-gold-500/50
    `,
    ghost: `
      bg-transparent text-neutral-600 
      hover:text-neutral-900 hover:bg-neutral-100
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
=======
  loading = false
}) => {
  const baseClasses = 'inline-flex items-center justify-center font-medium rounded-lg transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed';
  
  const variants = {
    primary: 'btn-primary shadow-md hover:shadow-lg focus:ring-focus',
    secondary: 'btn-secondary focus:ring-focus',
    outline: 'btn-outline focus:ring-focus',
    soft: 'btn-soft focus:ring-focus',
    contact: 'btn-contact focus:ring-focus', // Contact sales specific styling
    enterprise: 'btn-enterprise shadow-md hover:shadow-lg focus:ring-focus', // Enterprise specific styling
    ghost: 'text-neutral-600 hover:text-neutral-900 hover:bg-neutral-100 focus:ring-focus',
    danger: 'bg-error text-white hover:bg-red-700 focus:ring-red-500/50 shadow-md hover:shadow-lg'
  };

  const sizes = {
    xs: 'px-2 py-1 text-xs',
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-sm',
    lg: 'px-6 py-3 text-base',
    xl: 'px-8 py-4 text-lg'
  };

  const variantClasses = variants[variant];
  const sizeClasses = sizes[size];
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled || loading}
<<<<<<< HEAD
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
=======
      className={`${baseClasses} ${variantClasses} ${sizeClasses} ${className}`}
    >
      {loading && (
        <svg className="animate-spin -ml-1 mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
      )}
      {children}
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
    </button>
  );
};

export default Button;
