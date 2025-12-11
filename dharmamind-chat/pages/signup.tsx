/**
 * Chat Signup - Redirects to Central Auth (Brand Webpage)
 */

import { useEffect } from 'react';
import { useRouter } from 'next/router';

const AUTH_URL = process.env.NEXT_PUBLIC_AUTH_URL || 'http://localhost:3001';

const SignupPage = () => {
  const router = useRouter();
  const { callbackUrl } = router.query;

  useEffect(() => {
    // Redirect to central auth with callback to chat
    const callback = callbackUrl || `${window.location.origin}/chat`;
    const authUrl = `${AUTH_URL}/auth/signup?callbackUrl=${encodeURIComponent(callback as string)}`;
    window.location.href = authUrl;
  }, [callbackUrl]);

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="text-center">
        <div className="w-8 h-8 bg-gray-900 rounded-lg flex items-center justify-center mx-auto mb-4">
          <span className="text-white font-bold text-sm">D</span>
        </div>
        <p className="text-gray-500">Redirecting to sign up...</p>
      </div>
    </div>
  );
};

export default SignupPage;

