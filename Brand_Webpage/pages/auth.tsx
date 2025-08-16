import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Link from 'next/link';
import { motion } from 'framer-motion';
import Head from 'next/head';
import { useCentralizedSystem } from '../components/CentralizedSystem';
import Logo from '../components/Logo';
import Button from '../components/Button';

type AuthMode = 'login' | 'signup' | 'forgot-password' | 'verify';

export default function Auth() {
  const router = useRouter();
  const { goToHome } = useCentralizedSystem();
  
  // Get mode from URL query or default to login
  const [mode, setMode] = useState<AuthMode>('login');
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    password: '',
    confirmPassword: '',
    verificationCode: ''
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  // Initialize mode from URL params
  useEffect(() => {
    const { mode: urlMode, email } = router.query;
    if (urlMode && ['login', 'signup', 'forgot-password', 'verify'].includes(urlMode as string)) {
      setMode(urlMode as AuthMode);
    }
    if (email && typeof email === 'string') {
      setFormData(prev => ({ ...prev, email }));
    }
  }, [router.query]);

  // Update URL when mode changes
  const changeMode = (newMode: AuthMode) => {
    setMode(newMode);
    router.push(`/auth?mode=${newMode}`, undefined, { shallow: true });
    setError('');
    setSuccess('');
    // Reset form except email
    setFormData(prev => ({
      firstName: '',
      lastName: '',
      password: '',
      confirmPassword: '',
      verificationCode: '',
      email: prev.email
    }));
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    if (error) setError('');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setIsLoading(true);

    try {
      if (mode === 'login') {
        const response = await fetch('/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            email: formData.email, 
            password: formData.password 
          }),
        });

        if (response.ok) {
          window.location.href = 'https://dharmamind.ai';
        } else {
          const errorData = await response.json();
          setError(errorData.message || 'Login failed');
        }
      } 
      else if (mode === 'signup') {
        const response = await fetch('/api/auth/register', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: `${formData.firstName} ${formData.lastName}`,
            email: formData.email,
            password: formData.password,
            plan: router.query.plan || 'free'
          }),
        });

        const data = await response.json();
        if (response.ok && data.success) {
          setSuccess('Verification code sent! Redirecting to verification...');
          setTimeout(() => {
            changeMode('verify');
          }, 2000);
        } else {
          setError(data.message || 'Registration failed');
        }
      }
      else if (mode === 'forgot-password') {
        const response = await fetch('/api/auth/forgot-password', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email: formData.email }),
        });

        if (response.ok) {
          setSuccess('Password reset instructions sent to your email!');
        } else {
          const errorData = await response.json();
          setError(errorData.message || 'Failed to send reset email');
        }
      }
      else if (mode === 'verify') {
        const response = await fetch('/api/auth/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            email: formData.email,
            code: formData.verificationCode
          }),
        });

        const data = await response.json();
        if (response.ok && data.success) {
          setSuccess('Account verified successfully! Redirecting...');
          setTimeout(() => {
            window.location.href = 'https://dharmamind.ai';
          }, 2000);
        } else {
          setError(data.message || 'Verification failed');
        }
      }
    } catch (err) {
      setError('Something went wrong. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const getPageConfig = () => {
    switch (mode) {
      case 'signup':
        return {
          title: 'Create your account',
          subtitle: 'Begin your journey with AI soul',
          submitText: 'Create account'
        };
      case 'forgot-password':
        return {
          title: 'Reset your password',
          subtitle: 'Enter your email to receive reset instructions',
          submitText: 'Send reset email'
        };
      case 'verify':
        return {
          title: 'Verify your account',
          subtitle: 'Enter the verification code sent to your email',
          submitText: 'Verify account'
        };
      default:
        return {
          title: 'Sign in to your account',
          subtitle: 'Continue your journey with AI soul',
          submitText: 'Sign in'
        };
    }
  };

  const config = getPageConfig();

  return (
    <>
      <Head>
        <title>{`${config.title} - DharmaMind`}</title>
        <meta name="description" content="DharmaMind authentication - Sign in or create your account" />
      </Head>

      <div className="min-h-screen bg-gradient-to-br from-stone-100 via-amber-50 to-emerald-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
        <div className="sm:mx-auto sm:w-full sm:max-w-md">
          <div className="flex justify-center">
            <Logo 
              size="md" 
              showText={true} 
              onClick={() => goToHome()}
              className="cursor-pointer"
            />
          </div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-stone-800">
            {config.title}
          </h2>
          <p className="mt-2 text-center text-sm text-stone-600">
            {config.subtitle}
          </p>
        </div>

        <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
          <div className="bg-white/80 backdrop-blur-sm py-8 px-4 shadow-lg sm:rounded-2xl sm:px-10 border border-stone-200/50">
            
            {/* Back to Home Button */}
            <div className="flex justify-between items-center mb-6">
              <button
                onClick={() => goToHome()}
                className="inline-flex items-center text-sm text-stone-600 hover:text-stone-900 transition-colors"
              >
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                Back to Home
              </button>
            </div>

            {/* Error Message */}
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-4 p-3 bg-red-50/80 border border-red-200/50 rounded-xl text-red-700 text-sm backdrop-blur-sm"
              >
                {error}
              </motion.div>
            )}

            {/* Success Message */}
            {success && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-4 p-3 bg-green-50/80 border border-green-200/50 rounded-xl text-green-700 text-sm backdrop-blur-sm"
              >
                {success}
              </motion.div>
            )}

            <form className="space-y-6" onSubmit={handleSubmit}>
              {/* Name fields for signup */}
              {mode === 'signup' && (
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label htmlFor="firstName" className="block text-sm font-medium text-stone-700">
                      First name
                    </label>
                    <div className="mt-1">
                      <input
                        id="firstName"
                        name="firstName"
                        type="text"
                        autoComplete="given-name"
                        required
                        value={formData.firstName}
                        onChange={handleInputChange}
                        className="appearance-none block w-full px-4 py-3 border border-stone-300/50 rounded-xl shadow-sm placeholder-stone-400 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500/50 transition-all duration-300 bg-white/70 backdrop-blur-sm"
                        placeholder="First name"
                      />
                    </div>
                  </div>

                  <div>
                    <label htmlFor="lastName" className="block text-sm font-medium text-stone-700">
                      Last name
                    </label>
                    <div className="mt-1">
                      <input
                        id="lastName"
                        name="lastName"
                        type="text"
                        autoComplete="family-name"
                        required
                        value={formData.lastName}
                        onChange={handleInputChange}
                        className="appearance-none block w-full px-4 py-3 border border-stone-300/50 rounded-xl shadow-sm placeholder-stone-400 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500/50 transition-all duration-300 bg-white/70 backdrop-blur-sm"
                        placeholder="Last name"
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* Email field */}
              {mode !== 'verify' && (
                <div>
                  <label htmlFor="email" className="block text-sm font-medium text-stone-700">
                    Email address
                  </label>
                  <div className="mt-1">
                    <input
                      id="email"
                      name="email"
                      type="email"
                      autoComplete="email"
                      required
                      value={formData.email}
                      onChange={handleInputChange}
                      className="appearance-none block w-full px-4 py-3 border border-stone-300/50 rounded-xl shadow-sm placeholder-stone-400 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500/50 transition-all duration-300 bg-white/70 backdrop-blur-sm"
                      placeholder="Enter your email"
                    />
                  </div>
                </div>
              )}

              {/* Password fields */}
              {(mode === 'login' || mode === 'signup') && (
                <div>
                  <label htmlFor="password" className="block text-sm font-medium text-stone-700">
                    Password
                  </label>
                  <div className="mt-1">
                    <input
                      id="password"
                      name="password"
                      type="password"
                      autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                      required
                      value={formData.password}
                      onChange={handleInputChange}
                      className="appearance-none block w-full px-4 py-3 border border-stone-300/50 rounded-xl shadow-sm placeholder-stone-400 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500/50 transition-all duration-300 bg-white/70 backdrop-blur-sm"
                      placeholder={mode === 'signup' ? 'Create a password' : 'Enter your password'}
                    />
                  </div>
                  {mode === 'signup' && (
                    <p className="mt-1 text-xs text-stone-500">
                      Must be at least 8 characters long
                    </p>
                  )}
                </div>
              )}

              {/* Confirm Password for signup */}
              {mode === 'signup' && (
                <div>
                  <label htmlFor="confirmPassword" className="block text-sm font-medium text-stone-700">
                    Confirm password
                  </label>
                  <div className="mt-1">
                    <input
                      id="confirmPassword"
                      name="confirmPassword"
                      type="password"
                      autoComplete="new-password"
                      required
                      value={formData.confirmPassword}
                      onChange={handleInputChange}
                      className="appearance-none block w-full px-4 py-3 border border-stone-300/50 rounded-xl shadow-sm placeholder-stone-400 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500/50 transition-all duration-300 bg-white/70 backdrop-blur-sm"
                      placeholder="Confirm your password"
                    />
                  </div>
                </div>
              )}

              {/* Verification Code for verify mode */}
              {mode === 'verify' && (
                <div>
                  <label htmlFor="verificationCode" className="block text-sm font-medium text-stone-700">
                    Verification Code
                  </label>
                  <div className="mt-1">
                    <input
                      id="verificationCode"
                      name="verificationCode"
                      type="text"
                      required
                      value={formData.verificationCode}
                      onChange={handleInputChange}
                      className="appearance-none block w-full px-4 py-3 border border-stone-300/50 rounded-xl shadow-sm placeholder-stone-400 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500/50 transition-all duration-300 bg-white/70 backdrop-blur-sm"
                      placeholder="Enter verification code"
                    />
                  </div>
                  <p className="mt-1 text-xs text-stone-500">
                    Check your email ({formData.email}) for the verification code
                  </p>
                </div>
              )}

              {/* Forgot Password link for login */}
              {mode === 'login' && (
                <div className="flex items-center justify-between">
                  <div></div>
                  <div className="text-sm">
                    <button
                      type="button"
                      onClick={() => changeMode('forgot-password')}
                      className="font-medium text-emerald-600 hover:text-emerald-700 transition-colors duration-200"
                    >
                      Forgot your password?
                    </button>
                  </div>
                </div>
              )}

              {/* Terms and Privacy for signup */}
              {mode === 'signup' && (
                <div className="space-y-3">
                  <div className="flex items-start">
                    <input
                      id="acceptTerms"
                      name="acceptTerms"
                      type="checkbox"
                      required
                      className="h-4 w-4 text-emerald-600 focus:ring-emerald-500/50 border-stone-300 rounded mt-0.5"
                    />
                    <label htmlFor="acceptTerms" className="ml-3 block text-sm text-stone-700">
                      I agree to the{' '}
                      <Link href="/terms" className="text-emerald-600 hover:text-emerald-700 underline">
                        Terms of Service
                      </Link>
                    </label>
                  </div>

                  <div className="flex items-start">
                    <input
                      id="acceptPrivacy"
                      name="acceptPrivacy"
                      type="checkbox"
                      required
                      className="h-4 w-4 text-emerald-600 focus:ring-emerald-500/50 border-stone-300 rounded mt-0.5"
                    />
                    <label htmlFor="acceptPrivacy" className="ml-3 block text-sm text-stone-700">
                      I agree to the{' '}
                      <Link href="/privacy" className="text-emerald-600 hover:text-emerald-700 underline">
                        Privacy Policy
                      </Link>
                    </label>
                  </div>
                </div>
              )}

              {/* Submit Button */}
              <div>
                <Button
                  type="submit"
                  variant="primary"
                  size="lg"
                  disabled={isLoading}
                  className="w-full"
                >
                  {isLoading ? 'Processing...' : config.submitText}
                </Button>
              </div>
            </form>

            {/* Mode Switching Links */}
            <div className="mt-6">
              <div className="text-center space-y-2">
                {mode === 'login' && (
                  <span className="text-sm text-stone-600">
                    Don't have an account?{' '}
                    <button
                      onClick={() => changeMode('signup')}
                      className="font-medium text-emerald-600 hover:text-emerald-700 transition-colors duration-200"
                    >
                      Sign up for free
                    </button>
                  </span>
                )}
                
                {mode === 'signup' && (
                  <span className="text-sm text-stone-600">
                    Already have an account?{' '}
                    <button
                      onClick={() => changeMode('login')}
                      className="font-medium text-emerald-600 hover:text-emerald-700 transition-colors duration-200"
                    >
                      Sign in
                    </button>
                  </span>
                )}

                {(mode === 'forgot-password' || mode === 'verify') && (
                  <span className="text-sm text-stone-600">
                    Remember your password?{' '}
                    <button
                      onClick={() => changeMode('login')}
                      className="font-medium text-emerald-600 hover:text-emerald-700 transition-colors duration-200"
                    >
                      Sign in
                    </button>
                  </span>
                )}
              </div>
            </div>

            <div className="mt-8 pt-6 border-t border-stone-200/50">
              <p className="text-xs text-stone-500 text-center">
                Secure authentication powered by DharmaMind
              </p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
