/**
 * Chat Login - Redirects to Central Auth (Brand Webpage)
 */

import { useEffect } from 'react';
import { useRouter } from 'next/router';

const AUTH_URL = process.env.NEXT_PUBLIC_AUTH_URL || 'http://localhost:3001';

const LoginPage = () => {
  const router = useRouter();
  const { callbackUrl } = router.query;

  useEffect(() => {
    // Redirect to central auth with callback to chat
    const callback = callbackUrl || `${window.location.origin}/chat`;
    const authUrl = `${AUTH_URL}/auth/login?callbackUrl=${encodeURIComponent(callback as string)}`;
    window.location.href = authUrl;
  }, [callbackUrl]);

  return (
    <div className="min-h-screen bg-neutral-50 dark:bg-neutral-900 flex items-center justify-center">
      <div className="text-center">
        <div className="w-12 h-12 mx-auto mb-4 rounded-xl overflow-hidden border-2 border-gold-500/30 shadow-md">
          <img src="/logo.jpeg" alt="DharmaMind" className="w-full h-full object-cover" />
        </div>
        <p className="text-neutral-600 dark:text-neutral-400 text-sm">Redirecting to sign in...</p>
        <div className="flex items-center justify-center gap-1 mt-3">
          <div className="w-1.5 h-1.5 bg-gold-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
          <div className="w-1.5 h-1.5 bg-gold-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
          <div className="w-1.5 h-1.5 bg-gold-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;

