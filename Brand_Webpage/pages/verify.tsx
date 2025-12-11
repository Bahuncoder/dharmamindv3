import React, { useState } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '../contexts/AuthContext';
import Logo from '../components/Logo';

export default function VerifyAccount() {
  const router = useRouter();
  const { login } = useAuth();
  const [email, setEmail] = useState(router.query.email || '');
  const [code, setCode] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [resendLoading, setResendLoading] = useState(false);

  const handleVerify = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      const response = await fetch('/api/auth/verify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: email,
          code: code
        }),
      });

      const data = await response.json();

      if (response.ok && data.success) {
        // User is verified, redirect to login to complete the flow
        router.push('/auth?mode=login&verified=true');
      } else {
        setError(data.message || 'Verification failed. Please try again.');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleResendCode = async () => {
    setResendLoading(true);
    setError('');

    try {
      const response = await fetch('/api/auth/resend-code', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email }),
      });

      const data = await response.json();

      if (response.ok) {
        alert('Verification code sent! Please check your email.');
      } else {
        setError(data.message || 'Failed to resend code.');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setResendLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-primary-background-light flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <Logo size="lg" />
<<<<<<< HEAD
          <h1 className="text-3xl font-bold text-neutral-900 mt-4">Verify Your Account</h1>
          <p className="text-neutral-600 mt-2">
=======
          <h1 className="text-3xl font-bold text-gray-900 mt-4">Verify Your Account</h1>
          <p className="text-gray-600 mt-2">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
            We've sent a verification code to your email
          </p>
        </div>

        {/* Verification Form */}
<<<<<<< HEAD
        <div className="bg-neutral-100 rounded-2xl shadow-xl p-8">
          <form onSubmit={handleVerify} className="space-y-6">
            {/* Email Display */}
            <div>
              <label className="block text-sm font-medium text-neutral-900 mb-2">
=======
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <form onSubmit={handleVerify} className="space-y-6">
            {/* Email Display */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                Email Address
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
<<<<<<< HEAD
                className="w-full px-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-500 focus:border-transparent bg-neutral-100"
=======
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent bg-gray-50"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                placeholder="your.email@example.com"
                required
                readOnly={!!router.query.email}
              />
            </div>

            {/* Verification Code */}
            <div>
<<<<<<< HEAD
              <label className="block text-sm font-medium text-neutral-900 mb-2">
=======
              <label className="block text-sm font-medium text-gray-700 mb-2">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                Verification Code
              </label>
              <input
                type="text"
                value={code}
                onChange={(e) => setCode(e.target.value.replace(/\\D/g, '').slice(0, 6))}
<<<<<<< HEAD
                className="w-full px-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-500 focus:border-transparent text-center text-2xl font-mono tracking-widest"
=======
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent text-center text-2xl font-mono tracking-widest"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                placeholder="000000"
                maxLength={6}
                required
              />
<<<<<<< HEAD
              <p className="text-sm text-neutral-600 mt-1">
=======
              <p className="text-sm text-gray-500 mt-1">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                Enter the 6-digit code sent to your email
              </p>
            </div>

            {/* Error Message */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <p className="text-red-700 text-sm">{error}</p>
              </div>
            )}

            {/* Verify Button */}
            <button
              type="submit"
              disabled={isLoading || code.length !== 6}
<<<<<<< HEAD
              className="w-full bg-gold-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-gold-700 focus:outline-none focus:ring-2 focus:ring-gold-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
=======
              className="w-full bg-emerald-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-emerald-700 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
            >
              {isLoading ? (
                <div className="flex items-center justify-center">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                  Verifying...
                </div>
              ) : (
                'Verify Account'
              )}
            </button>

            {/* Resend Code */}
            <div className="text-center">
<<<<<<< HEAD
              <p className="text-sm text-neutral-600 mb-2">
=======
              <p className="text-sm text-secondary mb-2">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                Didn't receive the code?
              </p>
              <button
                type="button"
                onClick={handleResendCode}
                disabled={resendLoading}
<<<<<<< HEAD
                className="text-neutral-600 hover:text-gold-600 font-medium text-sm disabled:opacity-50"
=======
                className="text-secondary hover:text-primary font-medium text-sm disabled:opacity-50"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
              >
                {resendLoading ? 'Sending...' : 'Resend Code'}
              </button>
            </div>
          </form>

          {/* Back to Login */}
<<<<<<< HEAD
          <div className="mt-8 pt-6 border-t border-neutral-300 text-center">
            <button
              onClick={() => router.push('/auth?mode=login')}
              className="text-neutral-600 hover:text-gold-600 text-sm"
=======
          <div className="mt-8 pt-6 border-t border-gray-200 text-center">
            <button
              onClick={() => router.push('/auth?mode=login')}
              className="text-gray-600 hover:text-gray-800 text-sm"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
            >
              ← Back to Login
            </button>
          </div>
        </div>

        {/* Security Note */}
        <div className="mt-6 text-center">
<<<<<<< HEAD
          <p className="text-xs text-neutral-600">
=======
          <p className="text-xs text-gray-500">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
            🔒 Your verification code expires in 10 minutes for security
          </p>
        </div>
      </div>
    </div>
  );
}
