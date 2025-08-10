/**
 * Authentication Modal Component for DharmaMind
 * Handles login, registration, and demo access in modal format
 */

import React, { useState } from 'react';
import { useRouter } from 'next/router';
import { signIn } from 'next-auth/react';
import { useAuth } from '../contexts/AuthContext';
import Logo from './Logo';

interface AuthComponentProps {
  onClose?: () => void;
  mode?: 'login' | 'register';
  redirectTo?: string;
  hideDemo?: boolean;
}

const AuthComponent: React.FC<AuthComponentProps> = ({ 
  onClose, 
  mode: initialMode = 'login',
  redirectTo,
  hideDemo = false
}) => {
  const router = useRouter();
  const { login, register, demoLogin, isLoading } = useAuth();
  const [mode, setMode] = useState<'login' | 'register'>(initialMode);
  const [formData, setFormData] = useState({
    first_name: '',
    last_name: '',
    email: '',
    password: '',
    confirmPassword: ''
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (mode === 'register' && !formData.first_name.trim()) {
      newErrors.first_name = 'First name is required';
    }

    if (mode === 'register' && !formData.last_name.trim()) {
      newErrors.last_name = 'Last name is required';
    }

    if (!formData.email.trim()) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Invalid email format';
    }

    if (!formData.password) {
      newErrors.password = 'Password is required';
    } else if (formData.password.length < 6) {
      newErrors.password = 'Password must be at least 6 characters';
    }

    if (mode === 'register' && formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) return;

    setIsSubmitting(true);
    setErrors({});

    try {
      let response;
      
      if (mode === 'login') {
        response = await login({
          email: formData.email,
          password: formData.password
        });
      } else {
        response = await register({
          first_name: formData.first_name,
          last_name: formData.last_name,
          email: formData.email,
          password: formData.password,
          accept_terms: true,
          accept_privacy: true
        });
      }

      if (response.success) {
        // Success! Close modal or redirect
        if (onClose) onClose();
        if (redirectTo) {
          window.location.href = redirectTo;
        }
      } else {
        setErrors({ general: response.error || 'Authentication failed' });
      }
    } catch (error) {
      console.error('Auth error:', error);
      setErrors({ general: 'Network error. Please try again.' });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDemoLogin = async (plan: 'free' | 'premium' | 'enterprise') => {
    setIsSubmitting(true);
    try {
      // Navigate directly to chat in demo mode
      if (onClose) onClose();
      router.push('/chat?demo=true');
    } catch (error) {
      console.error('Demo navigation error:', error);
      // Fallback to window.location
      window.location.href = '/chat?demo=true';
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: '' }));
    }
  };

  // Add Google OAuth handler
  const handleGoogleAuth = async () => {
    setIsSubmitting(true);
    try {
      const result = await signIn('google', {
        callbackUrl: redirectTo || '/chat',
        redirect: false,
      });

      if (result?.error) {
        setErrors({ general: 'Google authentication failed. Please try again.' });
      } else if (result?.url) {
        window.location.href = result.url;
      }
    } catch (error) {
      console.error('Google auth error:', error);
      setErrors({ general: 'Network error. Please try again.' });
    } finally {
      setIsSubmitting(false);
    }
  };

  // Navigate to full auth page for complex flows
  const navigateToFullAuth = (authMode: string) => {
    if (onClose) onClose();
    router.push(`/auth?mode=${authMode}`);
  };

  const toggleMode = () => {
    setMode(prev => prev === 'login' ? 'register' : 'login');
    setErrors({});
    setFormData({
      first_name: '',
      last_name: '',
      email: formData.email, // Keep email when switching
      password: '',
      confirmPassword: ''
    });
  };

  return (
    <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-8 relative">
      {/* Close Button */}
      {onClose && (
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-gray-600 text-2xl z-10"
        >
          ×
        </button>
      )}

      {/* Header */}
      <div className="text-center mb-8">
        <div className="mb-4">
          <Logo size="sm" showText={false} />
        </div>
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          {mode === 'login' ? 'Welcome Back' : 'Join DharmaMind'}
        </h2>
        <p className="text-gray-600">
          {mode === 'login' 
            ? 'Continue your spiritual journey' 
            : 'Begin your path to enlightenment'
          }
        </p>
      </div>

      {/* General Error */}
      {errors.general && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-600 text-sm">{errors.general}</p>
        </div>
      )}

      {/* Quick Google Sign In */}
      <div className="mb-6">
        <button
          onClick={handleGoogleAuth}
          disabled={isSubmitting || isLoading}
          className="w-full inline-flex justify-center py-3 px-4 border border-gray-300 rounded-lg shadow-sm bg-white text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300"
        >
          <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24">
            <path
              fill="currentColor"
              d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
            />
            <path
              fill="currentColor"
              d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
            />
            <path
              fill="currentColor"
              d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
            />
            <path
              fill="currentColor"
              d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
            />
          </svg>
          {isSubmitting || isLoading ? 'Please wait...' : `${mode === 'register' ? 'Sign up' : 'Sign in'} with Google`}
        </button>
      </div>

      <div className="relative mb-6">
        <div className="absolute inset-0 flex items-center">
          <div className="w-full border-t border-gray-300" />
        </div>
        <div className="relative flex justify-center text-sm">
          <span className="px-2 bg-white text-gray-500">Or</span>
        </div>
      </div>

      {/* Simplified Form for Modal */}
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Name fields for registration */}
        {mode === 'register' && (
          <div className="grid grid-cols-2 gap-3">
            <div>
              <input
                type="text"
                name="first_name"
                value={formData.first_name}
                onChange={handleInputChange}
                className={`w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                  errors.first_name ? 'border-red-500' : 'border-gray-300'
                }`}
                placeholder="First name"
                required
              />
              {errors.first_name && <p className="text-red-500 text-xs mt-1">{errors.first_name}</p>}
            </div>

            <div>
              <input
                type="text"
                name="last_name"
                value={formData.last_name}
                onChange={handleInputChange}
                className={`w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                  errors.last_name ? 'border-red-500' : 'border-gray-300'
                }`}
                placeholder="Last name"
                required
              />
              {errors.last_name && <p className="text-red-500 text-xs mt-1">{errors.last_name}</p>}
            </div>
          </div>
        )}

        {/* Email field */}
        <div>
          <input
            type="email"
            name="email"
            value={formData.email}
            onChange={handleInputChange}
            className={`w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 ${
              errors.email ? 'border-red-500' : 'border-gray-300'
            }`}
            placeholder="Email address"
            required
          />
          {errors.email && <p className="text-red-500 text-xs mt-1">{errors.email}</p>}
        </div>

        {/* Password field */}
        <div>
          <input
            type="password"
            name="password"
            value={formData.password}
            onChange={handleInputChange}
            className={`w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 ${
              errors.password ? 'border-red-500' : 'border-gray-300'
            }`}
            placeholder="Password"
            required
          />
          {errors.password && <p className="text-red-500 text-xs mt-1">{errors.password}</p>}
        </div>

        {/* Confirm Password field */}
        {mode === 'register' && (
          <div>
            <input
              type="password"
              name="confirmPassword"
              value={formData.confirmPassword}
              onChange={handleInputChange}
              className={`w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                errors.confirmPassword ? 'border-red-500' : 'border-gray-300'
              }`}
              placeholder="Confirm password"
              required
            />
            {errors.confirmPassword && <p className="text-red-500 text-xs mt-1">{errors.confirmPassword}</p>}
          </div>
        )}

        {/* Submit Button */}
        <button
          type="submit"
          disabled={isSubmitting || isLoading}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
        >
          {isSubmitting || isLoading ? 'Processing...' : (mode === 'login' ? 'Sign In' : 'Create Account')}
        </button>
      </form>

      {/* Mode Toggle */}
      <div className="mt-6 text-center">
        <p className="text-gray-600 text-sm">
          {mode === 'login' ? "Don't have an account? " : "Already have an account? "}
          <button
            onClick={toggleMode}
            className="text-blue-600 hover:text-blue-700 font-medium"
          >
            {mode === 'login' ? 'Sign up' : 'Sign in'}
          </button>
        </p>
      </div>

      {/* Advanced Options */}
      <div className="mt-4 text-center">
        <button
          onClick={() => navigateToFullAuth('forgot-password')}
          className="text-sm text-gray-500 hover:text-gray-700"
        >
          Forgot password?
        </button>
      </div>

      {/* Demo Access */}
      {!hideDemo && (
        <div className="mt-8 pt-6 border-t border-gray-200">
          <p className="text-center text-gray-600 mb-4 text-sm">Or try DharmaMind instantly:</p>
          <div className="space-y-2">
            <button
              onClick={() => handleDemoLogin('free')}
              disabled={isSubmitting}
              className="w-full bg-emerald-100 text-emerald-700 py-2 px-4 rounded-lg hover:bg-emerald-200 transition-colors disabled:opacity-50 text-sm"
            >
              🆓 Demo Mode (Free)
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default AuthComponent;
