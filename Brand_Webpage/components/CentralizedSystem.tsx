/**
 * 🕉️ DharmaMind Centralized System Hub
 * 
 * Central hub for all UI components, routing, and system management
 * Single source of truth for all application features
 */

import React, { createContext, useContext, useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '../contexts/AuthContext';
import { ContactButton, SupportSection } from './CentralizedSupport';
import CentralizedSubscriptionModal from './CentralizedSubscriptionModal';
import { LoadingSpinner } from './CentralizedLoading';
import { ErrorMessage } from './CentralizedError';
import AuthComponent from './AuthComponent';
import FeedbackButton from './FeedbackButton';
import Logo from './Logo';
import Button from './Button';

// ===============================
// CENTRALIZED SYSTEM CONTEXT
// ===============================

interface CentralizedSystemContextType {
  // Navigation
  navigateTo: (path: string) => void;
  goToAuth: (mode?: 'login' | 'signup') => void;
  goToSubscription: () => void;
  goToHelp: () => void;
  goToSupport: () => void;
  goToHome: () => void;
  
  // Modals
  showAuthModal: boolean;
  showSubscriptionModal: boolean;
  toggleAuthModal: (show?: boolean) => void;
  toggleSubscriptionModal: (show?: boolean) => void;
  
  // System state
  isLoading: boolean;
  error: string | null;
  clearError: () => void;
  
  // External redirects
  redirectToChat: () => void;
  redirectToExternalAuth: () => void;
}

const CentralizedSystemContext = createContext<CentralizedSystemContextType | null>(null);

// ===============================
// CENTRALIZED SYSTEM PROVIDER
// ===============================

export const CentralizedSystemProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const router = useRouter();
  const { user, isAuthenticated } = useAuth();
  
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [showSubscriptionModal, setShowSubscriptionModal] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Navigation functions
  const navigateTo = (path: string) => {
    setIsLoading(true);
    router.push(path).finally(() => setIsLoading(false));
  };

  const goToAuth = (mode: 'login' | 'signup' = 'login') => {
    // Navigate to professional auth page
    navigateTo(`/auth?mode=${mode}`);
  };

  const goToSubscription = () => {
    if (isAuthenticated) {
      setShowSubscriptionModal(true);
    } else {
      navigateTo('/subscription');
    }
  };

  const goToHelp = () => navigateTo('/help');
  const goToSupport = () => navigateTo('/support');
  const goToHome = () => navigateTo('/');

  // Modal functions
  const toggleAuthModal = (show?: boolean) => {
    setShowAuthModal(show !== undefined ? show : !showAuthModal);
  };

  const toggleSubscriptionModal = (show?: boolean) => {
    setShowSubscriptionModal(show !== undefined ? show : !showSubscriptionModal);
  };

  // External redirects
  const redirectToChat = () => {
    window.location.href = 'https://dharmamind.ai';
  };

  const redirectToExternalAuth = () => {
    window.location.href = 'https://dharmamind.ai/auth';
  };

  // Error management
  const clearError = () => setError(null);

  const value: CentralizedSystemContextType = {
    navigateTo,
    goToAuth,
    goToSubscription,
    goToHelp,
    goToSupport,
    goToHome,
    showAuthModal,
    showSubscriptionModal,
    toggleAuthModal,
    toggleSubscriptionModal,
    isLoading,
    error,
    clearError,
    redirectToChat,
    redirectToExternalAuth
  };

  return (
    <CentralizedSystemContext.Provider value={value}>
      {children}
      
      {/* Global Modals */}
      {showAuthModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <div 
            className="absolute inset-0 bg-black bg-opacity-50 backdrop-blur-sm"
            onClick={() => toggleAuthModal(false)}
          ></div>
          <div className="relative z-10">
            <AuthComponent
              onClose={() => toggleAuthModal(false)}
              redirectTo="/"
            />
          </div>
        </div>
      )}
      
      {showSubscriptionModal && (
        <CentralizedSubscriptionModal
          isOpen={showSubscriptionModal}
          onClose={() => toggleSubscriptionModal(false)}
          currentPlan={user?.subscription_plan || 'basic'}
        />
      )}
      
      {/* Global Loading Overlay */}
      {isLoading && (
        <div className="fixed inset-0 bg-black/20 backdrop-blur-sm z-50 flex items-center justify-center">
          <LoadingSpinner size="lg" />
        </div>
      )}
      
      {/* Global Error Display */}
      {error && (
        <div className="fixed top-4 right-4 z-50">
          <ErrorMessage 
            error={error} 
            onDismiss={clearError}
            variant="toast"
          />
        </div>
      )}
    </CentralizedSystemContext.Provider>
  );
};

// ===============================
// CENTRALIZED SYSTEM HOOK
// ===============================

export const useCentralizedSystem = () => {
  const context = useContext(CentralizedSystemContext);
  if (!context) {
    throw new Error('useCentralizedSystem must be used within CentralizedSystemProvider');
  }
  return context;
};

// ===============================
// CENTRALIZED UI COMPONENTS
// ===============================

// Centralized Header Component
export const CentralizedHeader: React.FC<{
  showNavigation?: boolean;
  variant?: 'minimal' | 'full';
}> = ({ showNavigation = true, variant = 'full' }) => {
  const { goToHome, goToAuth, toggleSubscriptionModal, redirectToChat } = useCentralizedSystem();
  const { user, isAuthenticated } = useAuth();

  return (
    <header className="border-b border-stone-200/50 bg-white/95 backdrop-blur-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Logo 
            size="sm"
            showText={true}
            onClick={goToHome}
          />
          
          {showNavigation && variant === 'full' && (
            <nav className="hidden md:flex space-x-6">
              <ContactButton type="help" variant="link" />
              <ContactButton type="support" variant="link" />
            </nav>
          )}

          <div className="flex items-center space-x-4">
            {isAuthenticated ? (
              <>
                <div className="hidden sm:flex items-center space-x-3">
                  <div className="text-right">
                    <div className="text-sm font-medium text-stone-900">
                      {user?.email?.split('@')[0] || 'User'}
                    </div>
                    <div className="text-xs text-stone-500">
                      {user?.subscription_plan || 'Basic Plan'}
                    </div>
                  </div>
                </div>
                <Button
                  onClick={redirectToChat}
                  variant="primary"
                  size="sm"
                >
                  Open Chat
                </Button>
              </>
            ) : (
              <>
                <Button
                  onClick={() => goToAuth('login')}
                  variant="outline"
                  size="sm"
                >
                  Sign In
                </Button>
                <Button
                  onClick={redirectToChat}
                  variant="primary"
                  size="sm"
                >
                  Try Demo
                </Button>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};

// Centralized Footer Component
export const CentralizedFooter: React.FC<{
  variant?: 'minimal' | 'detailed';
}> = ({ variant = 'minimal' }) => {
  const { goToHome, goToHelp, goToSupport, goToAuth } = useCentralizedSystem();

  if (variant === 'minimal') {
    return (
      <footer className="border-t border-gray-200 bg-white py-4">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <p className="text-sm text-gray-600">
            © 2025 DharmaMind. All rights reserved.
          </p>
        </div>
      </footer>
    );
  }

  return (
    <footer className="border-t border-gray-200 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <Logo size="sm" onClick={goToHome} />
            <p className="text-sm text-gray-600 mt-4">
              Your AI wisdom companion for spiritual growth and conscious decision-making.
            </p>
          </div>
          
          <div>
            <h3 className="text-sm font-semibold text-gray-900 mb-4">Support</h3>
            <div className="space-y-2">
              <ContactButton type="help" variant="link" size="sm">
                Help Center
              </ContactButton>
              <ContactButton type="support" variant="link" size="sm">
                Contact Support
              </ContactButton>
            </div>
          </div>
          
          <div>
            <h3 className="text-sm font-semibold text-gray-900 mb-4">Account</h3>
            <div className="space-y-2">
              <button 
                onClick={() => goToAuth('login')}
                className="block text-sm text-gray-600 hover:text-gray-900"
              >
                Sign In
              </button>
              <button 
                onClick={() => goToAuth('signup')}
                className="block text-sm text-gray-600 hover:text-gray-900"
              >
                Sign Up
              </button>
            </div>
          </div>
        </div>
        
        <div className="mt-8 pt-8 border-t border-gray-200 text-center">
          <p className="text-sm text-gray-600">
            © 2025 DharmaMind. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};

// Centralized Page Layout
export const CentralizedLayout: React.FC<{
  children: React.ReactNode;
  title?: string;
  showHeader?: boolean;
  showFooter?: boolean;
  showFeedback?: boolean;
  headerVariant?: 'minimal' | 'full';
  footerVariant?: 'minimal' | 'detailed';
}> = ({ 
  children, 
  title,
  showHeader = true,
  showFooter = true,
  showFeedback = true,
  headerVariant = 'full',
  footerVariant = 'minimal'
}) => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-amber-50 to-stone-50">
      {showHeader && <CentralizedHeader variant={headerVariant} />}
      
      <main className="flex-1">
        {title && (
          <div className="bg-gradient-to-br from-amber-50 to-emerald-50 py-8">
            <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
              <h1 className="text-3xl font-bold text-gray-900">{title}</h1>
            </div>
          </div>
        )}
        {children}
      </main>
      
      {showFooter && <CentralizedFooter variant={footerVariant} />}
      {showFeedback && <FeedbackButton variant="floating" />}
    </div>
  );
};

export default CentralizedSystemProvider;
