import React from 'react';
import { motion } from 'framer-motion';
import { 
  SunIcon, 
  MoonIcon,
  SparklesIcon
} from '@heroicons/react/24/outline';
import { useTheme } from '../contexts/ThemeContext';

interface ThemeToggleProps {
  className?: string;
  showLabels?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

const ThemeToggle: React.FC<ThemeToggleProps> = ({ 
  className = '', 
  showLabels = false,
  size = 'md'
}) => {
  const { currentTheme, setCurrentTheme } = useTheme();

  const themes = [
    {
      id: 'light' as const,
      name: 'Light',
      icon: SunIcon,
      color: '#f7fafc',
      bgColor: '#fbbf24'
    },
    {
      id: 'dark' as const,
      name: 'Dark', 
      icon: MoonIcon,
      color: '#1a202c',
      bgColor: '#4a5568'
    },
    {
      id: 'spiritual' as const,
      name: 'Spiritual',
      icon: SparklesIcon,
      color: '#0a0a0f',
      bgColor: 'linear-gradient(135deg, #805ad5, #4299e1, #ffd700)'
    }
  ];

  const getSizeClasses = () => {
    switch (size) {
      case 'sm': return 'w-8 h-8';
      case 'lg': return 'w-12 h-12';
      default: return 'w-10 h-10';
    }
  };

  const getIconSize = () => {
    switch (size) {
      case 'sm': return 'w-4 h-4';
      case 'lg': return 'w-6 h-6';
      default: return 'w-5 h-5';
    }
  };

  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      {showLabels && (
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Theme:
        </span>
      )}
      
      <div className="flex items-center space-x-1 p-1 bg-gray-100 dark:bg-gray-800 rounded-lg">
        {themes.map((theme) => {
          const IconComponent = theme.icon;
          const isActive = currentTheme === theme.id;
          
          return (
            <motion.button
              key={theme.id}
              onClick={() => setCurrentTheme(theme.id)}
              className={`
                ${getSizeClasses()}
                relative flex items-center justify-center rounded-md
                transition-all duration-200 ease-in-out
                ${isActive 
                  ? 'shadow-md' 
                  : 'hover:bg-gray-200 dark:hover:bg-gray-700'
                }
              `}
              style={{
                background: isActive 
                  ? theme.id === 'spiritual' 
                    ? theme.bgColor 
                    : theme.bgColor
                  : 'transparent'
              }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              title={`Switch to ${theme.name} mode`}
            >
              <IconComponent 
                className={`
                  ${getIconSize()}
                  ${isActive 
                    ? theme.id === 'spiritual' ? 'text-white' : 'text-white'
                    : 'text-gray-600 dark:text-gray-400'
                  }
                `}
              />
              
              {/* Spiritual mode special effects */}
              {theme.id === 'spiritual' && isActive && (
                <>
                  <motion.div
                    className="absolute inset-0 rounded-md"
                    style={{
                      background: 'linear-gradient(135deg, #805ad5, #4299e1, #ffd700)',
                      opacity: 0.8
                    }}
                    animate={{
                      background: [
                        'linear-gradient(135deg, #805ad5, #4299e1, #ffd700)',
                        'linear-gradient(135deg, #4299e1, #ffd700, #805ad5)',
                        'linear-gradient(135deg, #ffd700, #805ad5, #4299e1)',
                        'linear-gradient(135deg, #805ad5, #4299e1, #ffd700)'
                      ]
                    }}
                    transition={{
                      duration: 3,
                      repeat: Infinity,
                      ease: 'linear'
                    }}
                  />
                  <motion.div
                    className="absolute inset-0 rounded-md bg-gradient-to-r from-transparent via-white to-transparent"
                    style={{ opacity: 0.2 }}
                    animate={{
                      x: ['-100%', '100%']
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      ease: 'linear'
                    }}
                  />
                </>
              )}
              
              {/* Active indicator */}
              {isActive && (
                <motion.div
                  className="absolute -bottom-1 left-1/2 transform -translate-x-1/2 w-1 h-1 bg-current rounded-full"
                  layoutId="theme-indicator"
                  transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                />
              )}
            </motion.button>
          );
        })}
      </div>
      
      {/* Current theme label */}
      {showLabels && (
        <span className="text-sm text-gray-600 dark:text-gray-400 capitalize">
          {currentTheme === 'spiritual' && '✨'} {currentTheme}
        </span>
      )}
    </div>
  );
};

export default ThemeToggle;
