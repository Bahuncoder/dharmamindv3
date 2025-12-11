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
<<<<<<< HEAD
      <div className="min-h-screen bg-gradient-to-br from-neutral-800 via-neutral-900 to-neutral-800 flex items-center justify-center">
=======
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 flex items-center justify-center">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
        <div className="text-white text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
          <p>Loading...</p>
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

<<<<<<< HEAD
  // Check if paid plan is required (basic is the free tier)
  if (requiresPaidPlan && user.subscription_plan === 'basic') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-neutral-800 via-neutral-900 to-neutral-800 flex items-center justify-center p-4">
        <div className="bg-neutral-100 rounded-2xl shadow-2xl w-full max-w-md p-8 text-center">
          <div className="mb-6">
            <div className="w-16 h-16 bg-neutral-100 rounded-full flex items-center justify-center mx-auto mb-4">
=======
  // Check if paid plan is required
  if (requiresPaidPlan && user.subscription_plan === 'free') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-8 text-center">
          <div className="mb-6">
            <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
              <svg className="w-8 h-8 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m0 0v2m0-2h2m-2 0H10m4-4V9a4 4 0 00-8 0v2m0 0V9a2 2 0 012-2h2a2 2 0 012 2v2" />
              </svg>
            </div>
<<<<<<< HEAD
            <h2 className="text-2xl font-bold text-neutral-900 mb-2">Premium Feature</h2>
            <p className="text-neutral-600">
              This feature requires a paid subscription plan.
            </p>
          </div>

          <div className="space-y-3">
            <button
              onClick={() => window.location.href = '/?upgrade=true'}
              className="w-full bg-gradient-to-r from-gold-600 to-gold-700 text-white py-3 px-4 rounded-lg hover:from-gold-700 hover:to-gold-800 transition-all duration-200 font-medium"
=======
            <h2 className="text-2xl font-bold text-gray-800 mb-2">Premium Feature</h2>
            <p className="text-gray-600">
              This feature requires a paid subscription plan.
            </p>
          </div>
          
          <div className="space-y-3">
            <button
              onClick={() => window.location.href = '/?upgrade=true'}
              className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 px-4 rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-200 font-medium"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
            >
              Upgrade Now
            </button>
            <button
              onClick={() => window.location.href = '/'}
<<<<<<< HEAD
              className="w-full bg-neutral-100 text-neutral-900 py-3 px-4 rounded-lg hover:bg-neutral-100 transition-all duration-200 font-medium"
=======
              className="w-full bg-gray-100 text-gray-700 py-3 px-4 rounded-lg hover:bg-gray-200 transition-all duration-200 font-medium"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
            >
              Back to Home
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
