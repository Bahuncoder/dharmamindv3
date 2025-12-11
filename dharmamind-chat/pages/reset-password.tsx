import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Link from 'next/link';
import { motion } from 'framer-motion';
import Logo from '../components/Logo';

export default function ResetPassword() {
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [token, setToken] = useState('');
  const router = useRouter();

  useEffect(() => {
    // Get reset token from URL
    if (router.query.token) {
      setToken(router.query.token as string);
    }
  }, [router.query]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    // Validation
    if (password !== confirmPassword) {
      setError('Passwords do not match');
      setIsLoading(false);
      return;
    }

    if (password.length < 8) {
      setError('Password must be at least 8 characters long');
      setIsLoading(false);
      return;
    }

    if (!token) {
      setError('Invalid reset token. Please request a new password reset.');
      setIsLoading(false);
      return;
    }

    try {
      const response = await fetch('/api/auth/reset-password', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          token,
          password 
        }),
      });

      if (response.ok) {
        setSuccess(true);
        // Redirect to chat after 3 seconds
        setTimeout(() => {
          router.push('/chat');
        }, 3000);
      } else {
        const errorData = await response.json();
        setError(errorData.message || 'Failed to reset password');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  if (success) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-stone-100 via-amber-50 to-gold-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
        <div className="sm:mx-auto sm:w-full sm:max-w-md">
          <Link href="/chat" className="flex justify-center">
            <Logo size="md" showText={true} />
          </Link>
        </div>

        <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
          <div className="bg-white/80 backdrop-blur-sm py-8 px-4 shadow-lg sm:rounded-2xl sm:px-10 border border-stone-200/50 text-center">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="w-16 h-16 bg-success-100 rounded-full flex items-center justify-center mx-auto mb-4"
            >
              <svg className="w-8 h-8 text-success-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </motion.div>
            
            <h2 className="text-2xl font-bold text-stone-800 mb-4">Password Reset Successfully</h2>
            
            <p className="text-stone-600 mb-6">
              Your password has been updated successfully. You can now sign in with your new password.
            </p>
            
            <p className="text-sm text-stone-500 mb-8">
              Redirecting you to chat page in 3 seconds...
            </p>
            
            <Link
              href="/chat"
              className="w-full inline-flex justify-center py-3 px-4 border border-transparent text-sm font-medium rounded-xl text-white bg-gradient-to-r from-amber-600 to-gold-600 hover:from-amber-700 hover:to-gold-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gold-500/50 transition-all duration-300 shadow-md hover:shadow-lg"
            >
              Continue to Chat
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-stone-100 via-amber-50 to-gold-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
      <div className="sm:mx-auto sm:w-full sm:max-w-md">
        <Link href="/chat" className="flex justify-center">
          <Logo size="md" showText={true} />
        </Link>
        <h2 className="mt-6 text-center text-3xl font-extrabold text-stone-800">
          Reset your password
        </h2>
        <p className="mt-2 text-center text-sm text-stone-600">
          Enter your new password below
        </p>
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
        <div className="bg-white/80 backdrop-blur-sm py-8 px-4 shadow-lg sm:rounded-2xl sm:px-10 border border-stone-200/50">
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-4 p-3 bg-red-50/80 border border-red-200/50 rounded-xl text-red-700 text-sm backdrop-blur-sm"
            >
              {error}
            </motion.div>
          )}

          <form className="space-y-6" onSubmit={handleSubmit}>
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-stone-700">
                New password
              </label>
              <div className="mt-1">
                <input
                  id="password"
                  name="password"
                  type="password"
                  autoComplete="new-password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="appearance-none block w-full px-4 py-3 border border-stone-300/50 rounded-xl shadow-sm placeholder-stone-400 focus:outline-none focus:ring-2 focus:ring-gold-500/50 focus:border-gold-500/50 transition-all duration-300 bg-white/70 backdrop-blur-sm"
                  placeholder="Enter your new password"
                />
              </div>
              <p className="mt-1 text-xs text-stone-500">
                Must be at least 8 characters long
              </p>
            </div>

            <div>
              <label htmlFor="confirmPassword" className="block text-sm font-medium text-stone-700">
                Confirm new password
              </label>
              <div className="mt-1">
                <input
                  id="confirmPassword"
                  name="confirmPassword"
                  type="password"
                  autoComplete="new-password"
                  required
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="appearance-none block w-full px-4 py-3 border border-stone-300/50 rounded-xl shadow-sm placeholder-stone-400 focus:outline-none focus:ring-2 focus:ring-gold-500/50 focus:border-gold-500/50 transition-all duration-300 bg-white/70 backdrop-blur-sm"
                  placeholder="Confirm your new password"
                />
              </div>
            </div>

            <div>
              <motion.button
                type="submit"
                disabled={isLoading || !token}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="group relative w-full flex justify-center py-3 px-4 border border-transparent text-sm font-medium rounded-xl text-white bg-gradient-to-r from-amber-600 to-gold-600 hover:from-amber-700 hover:to-gold-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gold-500/50 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 shadow-md hover:shadow-lg"
              >
                {isLoading ? (
                  <div className="flex items-center space-x-2">
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    <span>Updating password...</span>
                  </div>
                ) : (
                  'Update password'
                )}
              </motion.button>
            </div>
          </form>

          <div className="mt-6">
            <div className="text-center">
              <span className="text-sm text-stone-600">
                Remember your password?{' '}
                <Link href="/chat" className="font-medium text-gold-600 hover:text-gold-700 transition-colors duration-200">
                  Back to Chat
                </Link>
              </span>
            </div>
          </div>

          {!token && (
            <div className="mt-8 pt-6 border-t border-stone-200/50">
              <p className="text-xs text-red-500 text-center">
                Invalid or missing reset token. Please request a new password reset.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
