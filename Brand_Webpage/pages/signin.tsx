/**
 * DharmaMind Sign In - Professional Sign In Page
 * Uses unified Layout component for consistent header/footer
 */

import React, { useState } from 'react';
import Link from 'next/link';
import Layout from '../components/layout/Layout';

const SignInPage: React.FC = () => {
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    rememberMe: false,
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      // Handle sign in - redirect to chat app
      window.location.href = 'https://dharmamind.ai/login';
    } catch (err) {
      setError('An error occurred. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Layout
      title="Sign In"
      description="Sign in to your DharmaMind account to access personalized AI guidance."
    >
      {/* Hero */}
      <section className="pt-32 pb-8 px-6 bg-neutral-100">
        <div className="max-w-md mx-auto text-center">
          <div className="w-16 h-16 bg-gold-100 rounded-2xl flex items-center justify-center mx-auto mb-6">
            <svg className="w-8 h-8 text-gold-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
          </div>
          <h1 className="text-4xl font-semibold text-neutral-900 leading-tight tracking-tight mb-4">
            Welcome back
          </h1>
          <p className="text-lg text-neutral-600">
            Sign in to continue your journey with DharmaMind
          </p>
        </div>
      </section>

      {/* Sign In Form */}
      <section className="py-8 px-6 bg-neutral-100">
        <div className="max-w-md mx-auto">
          <div className="bg-white p-8 rounded-2xl border border-neutral-200 shadow-sm">
            {error && (
              <div className="mb-6 p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm">
                {error}
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label htmlFor="email" className="block text-sm font-medium text-neutral-900 mb-2">
                  Email address
                </label>
                <input
                  type="email"
                  id="email"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  className="w-full px-4 py-3 bg-neutral-50 border border-neutral-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-gold-600 focus:border-transparent text-neutral-900 placeholder-neutral-400"
                  placeholder="you@example.com"
                  required
                />
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <label htmlFor="password" className="block text-sm font-medium text-neutral-900">
                    Password
                  </label>
                  <Link href="/forgot-password" className="text-sm text-gold-600 hover:text-gold-700 transition-colors">
                    Forgot password?
                  </Link>
                </div>
                <input
                  type="password"
                  id="password"
                  value={formData.password}
                  onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                  className="w-full px-4 py-3 bg-neutral-50 border border-neutral-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-gold-600 focus:border-transparent text-neutral-900 placeholder-neutral-400"
                  placeholder="••••••••"
                  required
                />
              </div>

              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="rememberMe"
                  checked={formData.rememberMe}
                  onChange={(e) => setFormData({ ...formData, rememberMe: e.target.checked })}
                  className="w-4 h-4 text-gold-600 border-neutral-300 rounded focus:ring-gold-600"
                />
                <label htmlFor="rememberMe" className="ml-2 text-sm text-neutral-600">
                  Keep me signed in
                </label>
              </div>

              <button
                type="submit"
                disabled={isLoading}
                className="w-full px-6 py-3 bg-gold-600 text-white font-medium rounded-lg hover:bg-gold-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? 'Signing in...' : 'Sign in'}
              </button>
            </form>

            {/* Divider */}
            <div className="relative my-8">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-neutral-200"></div>
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-4 bg-white text-neutral-500">or continue with</span>
              </div>
            </div>

            {/* Social Sign In */}
            <div className="space-y-3">
              <button
                type="button"
                className="w-full px-6 py-3 bg-white border border-neutral-300 text-neutral-700 font-medium rounded-lg hover:bg-neutral-50 transition-colors flex items-center justify-center gap-3"
              >
                <svg className="w-5 h-5" viewBox="0 0 24 24">
                  <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                  <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                  <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
                  <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
                </svg>
                Continue with Google
              </button>
            </div>

            {/* Sign Up Link */}
            <p className="text-center text-sm text-neutral-600 mt-8">
              Don&apos;t have an account?{' '}
              <Link href="/signup" className="text-gold-600 hover:text-gold-700 font-medium transition-colors">
                Create one free
              </Link>
            </p>
          </div>
        </div>
      </section>

      {/* Trust Indicators */}
      <section className="py-16 px-6 bg-neutral-100">
        <div className="max-w-4xl mx-auto">
          <div className="grid md:grid-cols-3 gap-8 text-center">
            <div>
              <div className="w-12 h-12 bg-gold-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                <svg className="w-6 h-6 text-gold-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              </div>
              <h3 className="font-semibold text-neutral-900 mb-1">Secure</h3>
              <p className="text-sm text-neutral-600">Your data is encrypted and protected</p>
            </div>
            <div>
              <div className="w-12 h-12 bg-gold-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                <svg className="w-6 h-6 text-gold-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              </div>
              <h3 className="font-semibold text-neutral-900 mb-1">Private</h3>
              <p className="text-sm text-neutral-600">Your conversations stay confidential</p>
            </div>
            <div>
              <div className="w-12 h-12 bg-gold-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                <svg className="w-6 h-6 text-gold-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h3 className="font-semibold text-neutral-900 mb-1">Fast</h3>
              <p className="text-sm text-neutral-600">Instant access to AI guidance</p>
            </div>
          </div>
        </div>
      </section>
    </Layout>
  );
};

export default SignInPage;
