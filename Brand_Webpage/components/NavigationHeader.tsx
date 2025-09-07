import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';
import Logo from './Logo';
import Button from './Button';

interface NavigationHeaderProps {
  title?: string;
  showBackButton?: boolean;
  showHomeButton?: boolean;
  customBackAction?: () => void;
  className?: string;
}

const NavigationHeader: React.FC<NavigationHeaderProps> = ({
  title,
  showBackButton = true,
  showHomeButton = false,
  customBackAction,
  className = ""
}) => {
  const router = useRouter();
  const [canGoBack, setCanGoBack] = useState(false);

  useEffect(() => {
    // Check if we have history to go back to
    // We'll use a more intelligent approach by checking sessionStorage
    const hasHistory = typeof window !== 'undefined' && 
      (window.history.length > 1 || sessionStorage.getItem('dharma_nav_history') !== null);
    setCanGoBack(hasHistory);

    // Track navigation history
    if (typeof window !== 'undefined') {
      const currentPath = router.asPath;
      const history = JSON.parse(sessionStorage.getItem('dharma_nav_history') || '[]');
      
      // Only add to history if it's different from the last entry
      if (history.length === 0 || history[history.length - 1] !== currentPath) {
        history.push(currentPath);
        // Keep only last 10 entries
        if (history.length > 10) {
          history.shift();
        }
        sessionStorage.setItem('dharma_nav_history', JSON.stringify(history));
      }
    }
  }, [router.asPath]);

  const handleBack = () => {
    if (customBackAction) {
      customBackAction();
      return;
    }

    // Smart back navigation
    if (typeof window !== 'undefined') {
      const history = JSON.parse(sessionStorage.getItem('dharma_nav_history') || '[]');
      
      if (history.length > 1) {
        // Remove current page and go to previous
        history.pop();
        const previousPage = history[history.length - 1];
        
        // Update history
        sessionStorage.setItem('dharma_nav_history', JSON.stringify(history));
        
        // Navigate to previous page
        if (previousPage && previousPage !== router.asPath) {
          router.push(previousPage);
        } else {
          // Fallback to browser back
          router.back();
        }
      } else if (canGoBack) {
        // Use browser back as fallback
        router.back();
      } else {
        // Ultimate fallback - go home
        router.push('/');
      }
    } else {
      // Server-side fallback
      router.push('/');
    }
  };

  const handleHome = () => {
    // Clear history when going home
    if (typeof window !== 'undefined') {
      sessionStorage.setItem('dharma_nav_history', JSON.stringify(['/']));
    }
    router.push('/');
  };

  return (
    <motion.header
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className={`bg-white/90 backdrop-blur-sm border-b border-gray-200/50 sticky top-0 z-50 ${className}`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-4">
            {/* Smart Back Button */}
            {showBackButton && (
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleBack}
                className="flex items-center space-x-2 px-3 py-2 rounded-lg text-gray-600 hover:text-primary hover:bg-gray-50 transition-all duration-200 group"
                title={canGoBack ? "Go back to previous page" : "Go to homepage"}
              >
                <svg 
                  className="w-5 h-5 transition-transform group-hover:-translate-x-0.5" 
                  fill="none" 
                  stroke="currentColor" 
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                <span className="font-medium hidden sm:block">
                  {canGoBack ? 'Back' : 'Home'}
                </span>
              </motion.button>
            )}
            
            {/* Separator */}
            {showBackButton && (showHomeButton || title) && (
              <div className="h-6 w-px bg-gray-200"></div>
            )}
            
            {/* Home Button */}
            {showHomeButton && (
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleHome}
                className="flex items-center space-x-2 px-3 py-2 rounded-lg text-gray-600 hover:text-primary hover:bg-gray-50 transition-all duration-200"
                title="Go to homepage"
              >
                <svg 
                  className="w-5 h-5" 
                  fill="none" 
                  stroke="currentColor" 
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                </svg>
                <span className="font-medium hidden sm:block">Home</span>
              </motion.button>
            )}
            
            {/* Logo - Always clickable to home */}
            <Logo 
              size="sm"
              onClick={handleHome}
              className="cursor-pointer hover:opacity-80 transition-opacity"
            />
            
            {/* Page Title */}
            {title && (
              <>
                <div className="h-6 w-px bg-gray-200"></div>
                <h1 className="text-lg font-semibold text-gray-900 hidden md:block">
                  {title}
                </h1>
              </>
            )}
          </div>

          {/* Right Side Navigation */}
          <nav className="hidden md:flex items-center space-x-6">
            <button
              onClick={() => router.push('/features')}
              className="text-gray-600 hover:text-primary text-sm font-medium transition-colors"
            >
              Features
            </button>
            <button
              onClick={() => router.push('/pricing')}
              className="text-gray-600 hover:text-primary text-sm font-medium transition-colors"
            >
              Pricing
            </button>
            <button
              onClick={() => router.push('/about')}
              className="text-gray-600 hover:text-primary text-sm font-medium transition-colors"
            >
              About
            </button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => router.push('/contact')}
            >
              Contact
            </Button>
          </nav>

          {/* Mobile Menu Button */}
          <button
            onClick={() => {
              // You can implement mobile menu here
              console.log('Mobile menu clicked');
            }}
            className="md:hidden p-2 rounded-lg text-gray-600 hover:text-primary hover:bg-gray-50 transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        </div>
      </div>
    </motion.header>
  );
};

export default NavigationHeader;
