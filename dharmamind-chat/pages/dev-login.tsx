/**
 * Development Login Page
 * Allows direct login without redirecting to brand website
 * Only available in development mode
 */

import { useState } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '../contexts/AuthContext';
import Logo from '../components/Logo';

const DevLoginPage = () => {
    const router = useRouter();
    const { login, demoLogin, guestLogin } = useAuth();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    // Only allow in development
    const isDev = process.env.NODE_ENV === 'development';

    if (!isDev) {
        return (
            <div className="min-h-screen bg-neutral-50 dark:bg-neutral-900 flex items-center justify-center">
                <div className="text-center">
                    <p className="text-red-500">Dev login is only available in development mode.</p>
                </div>
            </div>
        );
    }

    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setIsLoading(true);

        try {
            const result = await login({ email, password });
            if (result.success) {
                router.push('/chat');
            } else {
                setError(result.error || 'Login failed');
            }
        } catch (err) {
            setError('Login failed');
        } finally {
            setIsLoading(false);
        }
    };

    const handleDemoLogin = async (plan: 'basic' | 'pro' | 'max' | 'enterprise') => {
        setIsLoading(true);
        try {
            const result = await demoLogin(plan);
            if (result.success) {
                router.push('/chat');
            } else {
                setError(result.error || 'Demo login failed');
            }
        } catch (err) {
            setError('Demo login failed');
        } finally {
            setIsLoading(false);
        }
    };

    const handleGuestLogin = () => {
        guestLogin();
        router.push('/chat');
    };

    return (
        <div className="min-h-screen bg-neutral-50 dark:bg-neutral-900 flex items-center justify-center p-4">
            <div className="w-full max-w-md">
                {/* Header */}
                <div className="text-center mb-8">
                    <div className="flex justify-center mb-4">
                        <Logo size="lg" showText={false} />
                    </div>
                    <h1 className="text-2xl font-bold text-neutral-900 dark:text-white">Dev Login</h1>
                    <p className="text-sm text-neutral-500 mt-1">Development mode only</p>
                </div>

                {/* Error Message */}
                {error && (
                    <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm">
                        {error}
                    </div>
                )}

                {/* Login Form */}
                <form onSubmit={handleLogin} className="bg-white dark:bg-neutral-800 rounded-xl shadow-lg p-6 mb-4">
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1">
                                Email
                            </label>
                            <input
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                className="w-full px-4 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-700 text-neutral-900 dark:text-white focus:ring-2 focus:ring-gold-500 focus:border-transparent"
                                placeholder="dev@example.com"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1">
                                Password
                            </label>
                            <input
                                type="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                className="w-full px-4 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-700 text-neutral-900 dark:text-white focus:ring-2 focus:ring-gold-500 focus:border-transparent"
                                placeholder="••••••••"
                            />
                        </div>
                        <button
                            type="submit"
                            disabled={isLoading}
                            className="w-full py-2.5 bg-gold-500 hover:bg-gold-600 text-white font-medium rounded-lg transition-colors disabled:opacity-50"
                        >
                            {isLoading ? 'Logging in...' : 'Login'}
                        </button>
                    </div>
                </form>

                {/* Quick Demo Logins */}
                <div className="bg-white dark:bg-neutral-800 rounded-xl shadow-lg p-6 mb-4">
                    <h3 className="text-sm font-semibold text-neutral-700 dark:text-neutral-300 mb-3">
                        Quick Demo Login (No backend needed)
                    </h3>
                    <div className="grid grid-cols-2 gap-2">
                        <button
                            onClick={() => handleDemoLogin('basic')}
                            disabled={isLoading}
                            className="py-2 px-3 bg-neutral-100 hover:bg-neutral-200 dark:bg-neutral-700 dark:hover:bg-neutral-600 text-neutral-700 dark:text-neutral-300 text-sm font-medium rounded-lg transition-colors"
                        >
                            Basic Plan
                        </button>
                        <button
                            onClick={() => handleDemoLogin('pro')}
                            disabled={isLoading}
                            className="py-2 px-3 bg-blue-100 hover:bg-blue-200 dark:bg-blue-900/30 dark:hover:bg-blue-900/50 text-blue-700 dark:text-blue-300 text-sm font-medium rounded-lg transition-colors"
                        >
                            Pro Plan
                        </button>
                        <button
                            onClick={() => handleDemoLogin('max')}
                            disabled={isLoading}
                            className="py-2 px-3 bg-purple-100 hover:bg-purple-200 dark:bg-purple-900/30 dark:hover:bg-purple-900/50 text-purple-700 dark:text-purple-300 text-sm font-medium rounded-lg transition-colors"
                        >
                            Max Plan
                        </button>
                        <button
                            onClick={() => handleDemoLogin('enterprise')}
                            disabled={isLoading}
                            className="py-2 px-3 bg-gold-100 hover:bg-gold-200 dark:bg-gold-900/30 dark:hover:bg-gold-900/50 text-gold-700 dark:text-gold-300 text-sm font-medium rounded-lg transition-colors"
                        >
                            Enterprise
                        </button>
                    </div>
                </div>

                {/* Guest Login */}
                <div className="bg-white dark:bg-neutral-800 rounded-xl shadow-lg p-6">
                    <button
                        onClick={handleGuestLogin}
                        className="w-full py-2.5 border-2 border-neutral-300 dark:border-neutral-600 hover:border-gold-500 dark:hover:border-gold-500 text-neutral-700 dark:text-neutral-300 font-medium rounded-lg transition-colors"
                    >
                        Continue as Guest
                    </button>
                    <p className="text-xs text-neutral-500 text-center mt-2">
                        Limited features, no account needed
                    </p>
                </div>

                {/* Back Link */}
                <div className="text-center mt-6">
                    <a href="/" className="text-sm text-gold-600 hover:text-gold-700">
                        ← Back to Home
                    </a>
                </div>
            </div>
        </div>
    );
};

export default DevLoginPage;
