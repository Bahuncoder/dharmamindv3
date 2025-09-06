import React, { useState } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import Logo from '../../components/Logo';

const AdminLogin: React.FC = () => {
  const router = useRouter();
  const [credentials, setCredentials] = useState({
    email: '',
    password: ''
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      // Secure authentication via backend API
      const response = await fetch('/api/admin/auth', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: credentials.email,
          password: credentials.password
        })
      });

      const data = await response.json();

      if (response.ok && data.success) {
        // Store secure session token
        localStorage.setItem('dharma_admin_token', data.token);
        router.push('/admin/dashboard');
      } else {
        setError(data.message || 'Invalid credentials. Please try again.');
      }
    } catch (error) {
      setError('Authentication service unavailable. Please try again later.');
    }
    
    setIsLoading(false);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setCredentials({
      ...credentials,
      [e.target.name]: e.target.value
    });
  };

  return (
    <>
      <Head>
        <title>Admin Login - DharmaMind</title>
        <meta name="description" content="DharmaMind Admin Panel Login" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/logo.jpeg" />
      </Head>
      <div className="min-h-screen bg-secondary-bg flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-md w-full space-y-8">
          {/* Header */}
          <div className="text-center">
            <div className="flex justify-center mb-6">
              <Logo size="lg" showText={true} />
            </div>
            <h2 className="text-3xl font-black text-primary tracking-tight">
              Admin Login
            </h2>
            <p className="mt-2 text-sm text-secondary font-semibold">
              Access the DharmaMind content management system
            </p>
          </div>

          {/* Login Form */}
          <div className="card-primary p-8">
            <form className="space-y-6" onSubmit={handleSubmit}>
              {error && (
                <div className="bg-error-light border border-error text-error px-4 py-3 rounded-lg text-sm font-semibold">
                  {error}
                </div>
              )}
              
              <div>
                <label htmlFor="email" className="block text-sm font-bold text-primary mb-2">
                  Email Address
                </label>
                <input
                  id="email"
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  value={credentials.email}
                  onChange={handleChange}
                  className="w-full px-4 py-3 border border-medium rounded-lg focus:border-primary focus:outline-none font-semibold"
                  placeholder="admin@dharmamind.com"
                />
              </div>

              <div>
                <label htmlFor="password" className="block text-sm font-bold text-primary mb-2">
                  Password
                </label>
                <input
                  id="password"
                  name="password"
                  type="password"
                  autoComplete="current-password"
                  required
                  value={credentials.password}
                  onChange={handleChange}
                  className="w-full px-4 py-3 border border-medium rounded-lg focus:border-primary focus:outline-none font-semibold"
                  placeholder="Enter your password"
                />
              </div>

              <div>
                <button
                  type="submit"
                  disabled={isLoading}
                  className="w-full btn-primary px-4 py-3 rounded-lg font-black text-lg disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoading ? 'Signing in...' : 'Sign in to Admin Panel'}
                </button>
              </div>
            </form>

            {/* Security Notice */}
            <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
              <h4 className="text-sm font-bold text-red-800 mb-2">Security Notice:</h4>
              <div className="text-xs text-red-700 font-semibold space-y-1">
                <p>This admin panel uses secure authentication.</p>
                <p>Contact your system administrator for access credentials.</p>
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="text-center">
            <p className="text-xs text-muted font-semibold">
              Protected by DharmaMind Security • 
              <a href="/" className="text-primary hover:underline ml-1">
                Back to Site
              </a>
            </p>
          </div>
        </div>
      </div>
    </>
  );
};

export default AdminLogin;
