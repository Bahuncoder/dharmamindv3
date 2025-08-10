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
          <h1 className="text-3xl font-bold text-gray-900 mt-4">Verify Your Account</h1>
          <p className="text-gray-600 mt-2">
            We've sent a verification code to your email
          </p>
        </div>

        {/* Verification Form */}
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <form onSubmit={handleVerify} className="space-y-6">
            {/* Email Display */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Email Address
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent bg-gray-50"
                placeholder="your.email@example.com"
                required
                readOnly={!!router.query.email}
              />
            </div>

            {/* Verification Code */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Verification Code
              </label>
              <input
                type="text"
                value={code}
                onChange={(e) => setCode(e.target.value.replace(/\\D/g, '').slice(0, 6))}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent text-center text-2xl font-mono tracking-widest"
                placeholder="000000"
                maxLength={6}
                required
              />
              <p className="text-sm text-gray-500 mt-1">
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
              className="w-full bg-emerald-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-emerald-700 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
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
              <p className="text-sm text-secondary mb-2">
                Didn't receive the code?
              </p>
              <button
                type="button"
                onClick={handleResendCode}
                disabled={resendLoading}
                className="text-secondary hover:text-primary font-medium text-sm disabled:opacity-50"
              >
                {resendLoading ? 'Sending...' : 'Resend Code'}
              </button>
            </div>
          </form>

          {/* Back to Login */}
          <div className="mt-8 pt-6 border-t border-gray-200 text-center">
            <button
              onClick={() => router.push('/auth?mode=login')}
              className="text-gray-600 hover:text-gray-800 text-sm"
            >
              ← Back to Login
            </button>
          </div>
        </div>

        {/* Security Note */}
        <div className="mt-6 text-center">
          <p className="text-xs text-gray-500">
            🔒 Your verification code expires in 10 minutes for security
          </p>
        </div>
      </div>
    </div>
  );
}
