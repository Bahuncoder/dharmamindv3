/**
 * Protected Route Component
 * Wraps components that require authentication
 */

import React, { ReactNode } from 'react';
import { useAuth } from '../contexts/AuthContext';
import AuthComponent from './AuthComponent';

interface ProtectedRouteProps {
  children: ReactNode;
  requiresPaidPlan?: boolean;
  fallback?: ReactNode;
  redirectTo?: string;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  requiresPaidPlan = false,
  fallback,
  redirectTo
}) => {
  const { isAuthenticated, user, isLoading } = useAuth();

  // Show loading state
  if (isLoading) {
    return (
      <div className="min-h-screen bg-neutral-50 dark:bg-neutral-900 flex items-center justify-center">
        <div className="text-center">
          <div className="w-14 h-14 mx-auto mb-5 rounded-xl overflow-hidden border-2 border-gold-500/30 shadow-md">
            <img
              src="/logo.jpeg"
              alt="DharmaMind"
              className="w-full h-full object-cover"
            />
          </div>
          <div className="flex items-center justify-center gap-1">
            <div className="w-1.5 h-1.5 bg-gold-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
            <div className="w-1.5 h-1.5 bg-gold-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
            <div className="w-1.5 h-1.5 bg-gold-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
          </div>
        </div>
      </div>
    );
  }

  // Not authenticated - show auth component
  if (!isAuthenticated || !user) {
    if (fallback) {
      return <>{fallback}</>;
    }
    return <AuthComponent redirectTo={redirectTo} />;
  }

  // Check if paid plan is required
  if (requiresPaidPlan && user.subscription_plan === 'basic') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-neutral-900 to-indigo-900 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-8 text-center">
          <div className="mb-6">
            <div className="w-16 h-16 bg-gold-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-gold-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m0 0v2m0-2h2m-2 0H10m4-4V9a4 4 0 00-8 0v2m0 0V9a2 2 0 012-2h2a2 2 0 012 2v2" />
              </svg>
            </div>
            <h2 className="text-2xl font-bold text-gray-800 mb-2">Premium Feature</h2>
            <p className="text-gray-600">
              This feature requires a paid subscription plan.
            </p>
          </div>

          <div className="space-y-3">
            <button
              onClick={() => window.location.href = '/?upgrade=true'}
              className="w-full bg-gradient-to-r from-gold-600 to-gold-600 text-white py-3 px-4 rounded-lg hover:from-gold-700 hover:to-purple-700 transition-all duration-200 font-medium"
            >
              Upgrade Now
            </button>
            <button
              onClick={() => window.location.href = '/chat?demo=true'}
              className="w-full bg-gray-100 text-gray-700 py-3 px-4 rounded-lg hover:bg-gray-200 transition-all duration-200 font-medium"
            >
              Back to Chat
            </button>
          </div>
        </div>
      </div>
    );
  }

  // User is authenticated and has required permissions
  return <>{children}</>;
};

export default ProtectedRoute;
