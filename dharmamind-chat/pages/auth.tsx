import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { signIn, getSession } from 'next-auth/react';
import { useAuth } from '../contexts/AuthContext';
import Head from 'next/head';
import Logo from '../components/Logo';
import { useError, useFormError, validationRules } from '../hooks/useError';
import { useLoading } from '../hooks/useLoading';
import { useNavigation, ROUTES } from '../hooks/useNavigation';
import { ErrorMessage, FieldError, FormErrors } from '../components/CentralizedError';
import { LoadingButton } from '../components/CentralizedLoading';

type AuthMode = 'login' | 'signup' | 'forgot-password' | 'verify';

export default function Auth() {
  const router = useRouter();
  const { login, register } = useAuth();
  const navigation = useNavigation();
  
  // Centralized state management
  const loading = useLoading();
  const formError = useFormError();
  
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
  const [options, setOptions] = useState({
    rememberMe: false,
    acceptTerms: false,
    acceptPrivacy: false,
    marketingConsent: false
  });
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

  // Get return URL from query params
  const getReturnUrl = () => {
    const { returnUrl } = router.query;
    return typeof returnUrl === 'string' ? returnUrl : '/chat';
  };

  // Check if user is already authenticated
  useEffect(() => {
    const checkSession = async () => {
      const session = await getSession();
      if (session) {
        navigation.goToChat();
      }
    };
    checkSession();
  }, [navigation]);

  // Update URL when mode changes
  const changeMode = (newMode: AuthMode) => {
    setMode(newMode);
    router.push(`/auth?mode=${newMode}`, undefined, { shallow: true });
    formError.clearAllErrors();
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
    const { name, value, type, checked } = e.target;
    
    if (type === 'checkbox') {
      setOptions(prev => ({ ...prev, [name]: checked }));
    } else {
      setFormData(prev => ({ ...prev, [name]: value }));
      // Clear field error when user starts typing
      if (formError.errors[name]) {
        formError.clearFieldError(name);
      }
    }
  };

  const validateForm = (): boolean => {
    const rules: Record<string, any[]> = {};
    
    // Email validation for all modes except verify
    if (mode !== 'verify') {
      rules.email = [
        validationRules.required('Email'),
        validationRules.email()
      ];
    }

    // Name validation for signup
    if (mode === 'signup') {
      rules.firstName = [validationRules.required('First name')];
      rules.lastName = [validationRules.required('Last name')];
    }

    // Password validation for login and signup
    if (mode === 'login' || mode === 'signup') {
      rules.password = [
        validationRules.required('Password'),
        validationRules.minLength(8)
      ];
    }

    // Confirm password for signup
    if (mode === 'signup') {
      rules.confirmPassword = [
        validationRules.confirmPassword(formData.password)
      ];
    }

    // Verification code for verify mode
    if (mode === 'verify') {
      rules.verificationCode = [validationRules.required('Verification code')];
    }

    const isFormValid = formError.validateForm(formData, rules);

    // Additional validations for signup
    if (mode === 'signup' && isFormValid) {
      if (!options.acceptTerms || !options.acceptPrivacy) {
        formError.setError('Please accept the Terms of Service and Privacy Policy');
        return false;
      }
    }

    return isFormValid;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) return;

    setSuccess('');
    
    await loading.withLoading(async () => {
      if (mode === 'login') {
        // Try NextAuth credentials first for test accounts
        const result = await signIn('credentials', {
          email: formData.email,
          password: formData.password,
          redirect: false,
        });

        if (result?.ok && !result?.error) {
          const returnUrl = getReturnUrl();
          router.push(returnUrl);
          return;
        }

        // Fallback to custom API
        const response = await fetch('/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            email: formData.email, 
            password: formData.password 
          }),
        });

        if (response.ok) {
          const returnUrl = getReturnUrl();
          router.push(returnUrl);
        } else {
          const errorData = await response.json();
          formError.setError(errorData.message || 'Login failed. Please check your credentials or create an account.');
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
            plan: router.query.plan || 'basic'
          }),
        });

        const data = await response.json();
        if (response.ok && data.success) {
          setSuccess('Verification code sent! Redirecting to verification...');
          setTimeout(() => {
            changeMode('verify');
          }, 2000);
        } else {
          formError.setError(data.message || 'Registration failed');
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
          formError.setError(errorData.message || 'Failed to send reset email');
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
          const returnUrl = getReturnUrl();
          setSuccess(`Account verified successfully! Redirecting to ${returnUrl === '/chat' ? 'chat' : 'home'}...`);
          setTimeout(() => {
            router.push(returnUrl);
          }, 2000);
        } else {
          formError.setError(data.message || 'Verification failed');
        }
      }
    }, 'Authentication failed. Please try again.');
  };

  const handleGoogleAuth = async () => {
    await loading.withLoading(async () => {
      const returnUrl = getReturnUrl();
      const result = await signIn('google', {
        callbackUrl: returnUrl,
        redirect: false,
      });

      if (result?.error) {
        formError.setError('Google authentication failed. Please try again.');
      } else if (result?.url) {
        window.location.href = result.url;
      }
    }, 'Google authentication failed. Please try again.');
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
        <title>{`${config.title} - DharmaMind-AI with Soul`}</title>
        <meta name="description" content="DharmaMind authentication - Sign in or create your account" />
      </Head>

      <div className="min-h-screen bg-gradient-to-br from-orange-50 via-amber-50 to-emerald-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8" style={{
        background: 'linear-gradient(to bottom right, var(--color-bg-light), #FFF7ED, #ECFDF5)'
      }}>
        <div className="sm:mx-auto sm:w-full sm:max-w-md">
          <div className="flex justify-center">
            <Logo 
              size="md" 
              showText={true} 
              className="cursor-pointer"
            />
          </div>
          <h2 className="mt-6 text-center text-3xl font-extrabold" style={{ color: 'var(--color-text-primary)' }}>
            {config.title}
          </h2>
          <p className="mt-2 text-center text-sm" style={{ color: 'var(--color-text-secondary)' }}>
            {config.subtitle}
          </p>
        </div>

        <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
          <div className="bg-white/80 backdrop-blur-sm py-8 px-4 shadow-lg sm:rounded-2xl sm:px-10" style={{
            backgroundColor: 'var(--color-bg-card)',
            borderColor: 'var(--color-border-light)'
          }}>
            
            {/* Back to Chat Button */}
            <div className="flex justify-between items-center mb-6">
              <button
                onClick={() => router.push('/chat')}
                className="inline-flex items-center text-sm transition-colors hover:opacity-80"
                style={{ 
                  color: 'var(--color-text-secondary)'
                }}
              >
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                Back to Chat
              </button>
            </div>

            {/* Error Messages */}
            <ErrorMessage 
              error={formError.error} 
              onDismiss={formError.clearError}
              className="mb-4"
            />
            
            {/* Form Field Errors */}
            <FormErrors 
              errors={formError.errors}
              className="mb-4"
            />

            {/* Success Message */}
            {success && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-4 p-3 rounded-xl text-sm backdrop-blur-sm"
                style={{
                  backgroundColor: 'var(--color-success-light)',
                  borderColor: 'var(--color-success)',
                  color: 'var(--color-success)'
                }}
              >
                {success}
              </motion.div>
            )}

            <form className="space-y-6" onSubmit={handleSubmit}>
              {/* Name fields for signup */}
              {mode === 'signup' && (
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label htmlFor="firstName" className="block text-sm font-medium" style={{ color: 'var(--color-text-primary)' }}>
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
                        className="appearance-none block w-full px-4 py-3 rounded-xl shadow-sm transition-all duration-300 backdrop-blur-sm"
                        style={{
                          borderColor: 'var(--color-border-light)',
                          backgroundColor: 'var(--color-bg-card)',
                          color: 'var(--color-text-primary)'
                        }}
                        onFocus={(e) => {
                          (e.target as HTMLInputElement).style.borderColor = 'var(--color-success)';
                          (e.target as HTMLInputElement).style.boxShadow = '0 0 0 2px rgba(50, 163, 112, 0.1)';
                        }}
                        onBlur={(e) => {
                          (e.target as HTMLInputElement).style.borderColor = 'var(--color-border-light)';
                          (e.target as HTMLInputElement).style.boxShadow = 'none';
                        }}
                        placeholder="First name"
                      />
                    </div>
                  </div>

                  <div>
                    <label htmlFor="lastName" className="block text-sm font-medium" style={{ color: 'var(--color-text-primary)' }}>
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
                        className="appearance-none block w-full px-4 py-3 rounded-xl shadow-sm transition-all duration-300 backdrop-blur-sm"
                        style={{
                          borderColor: 'var(--color-border-light)',
                          backgroundColor: 'var(--color-bg-card)',
                          color: 'var(--color-text-primary)'
                        }}
                        onFocus={(e) => {
                          (e.target as HTMLInputElement).style.borderColor = 'var(--color-success)';
                          (e.target as HTMLInputElement).style.boxShadow = '0 0 0 2px rgba(50, 163, 112, 0.1)';
                        }}
                        onBlur={(e) => {
                          (e.target as HTMLInputElement).style.borderColor = 'var(--color-border-light)';
                          (e.target as HTMLInputElement).style.boxShadow = 'none';
                        }}
                        placeholder="Last name"
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* Email field */}
              {mode !== 'verify' && (
                <div>
                  <label htmlFor="email" className="block text-sm font-medium" style={{ color: 'var(--color-text-primary)' }}>
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
                      className="appearance-none block w-full px-4 py-3 rounded-xl shadow-sm transition-all duration-300 backdrop-blur-sm"
                      style={{
                        borderColor: 'var(--color-border-light)',
                        backgroundColor: 'var(--color-bg-card)',
                        color: 'var(--color-text-primary)'
                      }}
                      onFocus={(e) => {
                        (e.target as HTMLInputElement).style.borderColor = 'var(--color-success)';
                        (e.target as HTMLInputElement).style.boxShadow = '0 0 0 2px rgba(50, 163, 112, 0.1)';
                      }}
                      onBlur={(e) => {
                        (e.target as HTMLInputElement).style.borderColor = 'var(--color-border-light)';
                        (e.target as HTMLInputElement).style.boxShadow = 'none';
                      }}
                      placeholder="Enter your email"
                    />
                  </div>
                </div>
              )}

              {/* Password fields */}
              {(mode === 'login' || mode === 'signup') && (
                <div>
                  <label htmlFor="password" className="block text-sm font-medium" style={{ color: 'var(--color-text-primary)' }}>
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
                      className="appearance-none block w-full px-4 py-3 rounded-xl shadow-sm transition-all duration-300 backdrop-blur-sm"
                      style={{
                        borderColor: 'var(--color-border-light)',
                        backgroundColor: 'var(--color-bg-card)',
                        color: 'var(--color-text-primary)'
                      }}
                      onFocus={(e) => {
                        (e.target as HTMLInputElement).style.borderColor = 'var(--color-success)';
                        (e.target as HTMLInputElement).style.boxShadow = '0 0 0 2px rgba(50, 163, 112, 0.1)';
                      }}
                      onBlur={(e) => {
                        (e.target as HTMLInputElement).style.borderColor = 'var(--color-border-light)';
                        (e.target as HTMLInputElement).style.boxShadow = 'none';
                      }}
                      placeholder={mode === 'signup' ? 'Create a password' : 'Enter your password'}
                    />
                  </div>
                  {mode === 'signup' && (
                    <p className="mt-1 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
                      Must be at least 8 characters long
                    </p>
                  )}
                </div>
              )}

              {/* Confirm Password for signup */}
              {mode === 'signup' && (
                <div>
                  <label htmlFor="confirmPassword" className="block text-sm font-medium" style={{ color: 'var(--color-text-primary)' }}>
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

              {/* Remember Me for login */}
              {mode === 'login' && (
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <input
                      id="rememberMe"
                      name="rememberMe"
                      type="checkbox"
                      checked={options.rememberMe}
                      onChange={handleInputChange}
                      className="h-4 w-4 text-emerald-600 focus:ring-emerald-500/50 border-stone-300 rounded"
                    />
                    <label htmlFor="rememberMe" className="ml-2 block text-sm text-stone-700">
                      Remember me
                    </label>
                  </div>

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
                      checked={options.acceptTerms}
                      onChange={handleInputChange}
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
                      checked={options.acceptPrivacy}
                      onChange={handleInputChange}
                      className="h-4 w-4 text-emerald-600 focus:ring-emerald-500/50 border-stone-300 rounded mt-0.5"
                    />
                    <label htmlFor="acceptPrivacy" className="ml-3 block text-sm text-stone-700">
                      I agree to the{' '}
                      <Link href="/privacy" className="text-emerald-600 hover:text-emerald-700 underline">
                        Privacy Policy
                      </Link>
                    </label>
                  </div>

                  <div className="flex items-start">
                    <input
                      id="marketingConsent"
                      name="marketingConsent"
                      type="checkbox"
                      checked={options.marketingConsent}
                      onChange={handleInputChange}
                      className="h-4 w-4 text-emerald-600 focus:ring-emerald-500/50 border-stone-300 rounded mt-0.5"
                    />
                    <label htmlFor="marketingConsent" className="ml-3 block text-sm" style={{ color: 'var(--color-text-primary)' }}>
                      I'd like to receive product updates and special offers (optional)
                    </label>
                  </div>
                </div>
              )}

              {/* Submit Button */}
              <div className="mt-8">
                <LoadingButton
                  type="submit"
                  isLoading={loading.isLoading}
                  loadingText="Processing..."
                  className="btn-primary w-full py-4 px-6 text-base font-semibold rounded-xl transition-all duration-300"
                >
                  {config.submitText}
                </LoadingButton>
              </div>
            </form>

            {/* Google OAuth - Only for login and signup */}
            {(mode === 'login' || mode === 'signup') && (
              <div className="mt-6">
                <div className="relative">
                  <div className="absolute inset-0 flex items-center">
                    <div className="w-full border-t" style={{ borderColor: 'var(--color-border-medium)' }} />
                  </div>
                  <div className="relative flex justify-center text-sm">
                    <span className="px-2" style={{ 
                      backgroundColor: 'var(--color-bg-card)', 
                      color: 'var(--color-text-secondary)' 
                    }}>Or continue with</span>
                  </div>
                </div>

                <div className="mt-6">
                  <LoadingButton
                    onClick={handleGoogleAuth}
                    isLoading={loading.isLoading}
                    loadingText="Authenticating..."
                    className="btn-google w-full py-4 px-6 text-base font-semibold rounded-xl"
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
                    {mode === 'signup' ? 'Sign up' : 'Sign in'} with Google
                  </LoadingButton>
                </div>
              </div>
            )}

            {/* Mode Switching Links */}
            <div className="mt-8">
              <div className="text-center space-y-3">
                {mode === 'login' && (
                  <span className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                    Don't have an account?{' '}
                    <button
                      onClick={() => changeMode('signup')}
                      className="font-semibold transition-colors duration-200 hover:opacity-80"
                      style={{ color: 'var(--color-logo-emerald)' }}
                    >
                      Sign up for free
                    </button>
                  </span>
                )}
                
                {mode === 'signup' && (
                  <span className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                    Already have an account?{' '}
                    <button
                      onClick={() => changeMode('login')}
                      className="font-semibold transition-colors duration-200 hover:opacity-80"
                      style={{ color: 'var(--color-logo-emerald)' }}
                    >
                      Sign in
                    </button>
                  </span>
                )}

                {(mode === 'forgot-password' || mode === 'verify') && (
                  <span className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
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
