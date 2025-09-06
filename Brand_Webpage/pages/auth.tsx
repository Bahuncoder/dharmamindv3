import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';
import { useCentralizedSystem } from '../components/CentralizedSystem';
import Logo from '../components/Logo';
import brandAuth from '../services/brandAuth';
import type { DharmaMindUser } from '../services/brandAuth';

export default function Auth() {
  const router = useRouter();
  const { goToHome } = useCentralizedSystem();
  const [isLoading, setIsLoading] = useState(true);
  const [user, setUser] = useState<DharmaMindUser | null>(null);

  useEffect(() => {
    const handleAuth = async () => {
      // Check if user is already authenticated
      if (brandAuth.isAuthenticated()) {
        const currentUser = await brandAuth.getCurrentUser();
        if (currentUser) {
          setUser(currentUser);
          // Redirect to intended page or home
          const returnUrl = router.query.returnUrl as string || '/';
          router.push(returnUrl);
          return;
        }
      }

      // Check for SSO token from central auth (Chat App)
      const authResult = await brandAuth.checkAutoLogin();
      if (authResult?.success && authResult.user) {
        setUser(authResult.user);
        const returnUrl = router.query.returnUrl as string || '/';
        router.push(returnUrl);
        return;
      }

      // If no auth found, redirect to central auth immediately
      handleCentralAuthRedirect();
    };

    handleAuth();
  }, [router]);

  const handleCentralAuthRedirect = () => {
    const mode = router.query.mode as string || 'login';
    const returnUrl = router.query.returnUrl as string || '/';
    
    // Redirect to Chat App (Central Auth Hub)
    brandAuth.redirectToCentralAuth(mode as 'login' | 'signup', returnUrl);
  };

  const handleRegisterRedirect = () => {
    const returnUrl = router.query.returnUrl as string || '/';
    brandAuth.redirectToCentralAuth('signup', returnUrl);
  };

  // Show loading/redirect screen
  return (
    <>
      <Head>
        <title>Redirecting to DharmaMind Login</title>
        <meta name="description" content="Redirecting to DharmaMind central authentication..." />
      </Head>

      <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-white to-gray-50 flex items-center justify-center px-4 relative overflow-hidden">
        {/* Background decoration - same as chat app */}
        <div className="absolute inset-0 -z-10">
          <div className="absolute top-10 left-10 w-72 h-72 bg-emerald-100/30 rounded-full mix-blend-multiply filter blur-xl animate-blob"></div>
          <div className="absolute top-0 right-4 w-72 h-72 bg-gray-100/40 rounded-full mix-blend-multiply filter blur-xl animate-blob animation-delay-2000"></div>
          <div className="absolute -bottom-8 left-20 w-72 h-72 bg-emerald-50/50 rounded-full mix-blend-multiply filter blur-xl animate-blob animation-delay-4000"></div>
        </div>

        <div className="text-center">
          <div className="bg-white/80 backdrop-blur-lg py-12 px-8 shadow-2xl rounded-3xl border border-white/20 relative max-w-md mx-auto">
            {/* Glass morphism effect */}
            <div className="absolute inset-0 bg-gradient-to-br from-white/60 to-white/30 rounded-3xl"></div>
            <div className="relative z-10">
              
              {/* Logo */}
              <div className="mb-8">
                <div className="relative inline-block">
                  <Logo 
                    size="lg" 
                    showText={true} 
                    className="cursor-pointer transform hover:scale-105 transition-transform duration-300"
                    onClick={() => goToHome()}
                  />
                  <div className="absolute -inset-2 bg-gradient-to-r from-emerald-200 to-emerald-100 rounded-full opacity-20 blur-lg"></div>
                </div>
              </div>

              {/* Loading Message */}
              <div className="mb-8">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-600 mx-auto mb-4"></div>
                <h2 className="text-3xl font-bold bg-gradient-to-r from-gray-900 via-emerald-800 to-gray-900 bg-clip-text text-transparent mb-4">
                  Connecting to DharmaMind
                </h2>
                <p className="text-lg text-gray-600 font-medium mb-6">
                  Redirecting you to secure authentication...
                </p>
                <div className="text-sm text-gray-500">
                  <p className="mb-2">✨ Single sign-on across all platforms</p>
                  <p className="mb-2">🔒 Secure authentication</p>
                  <p>🌟 Seamless experience</p>
                </div>
              </div>

              {/* Manual redirect buttons */}
              <div className="space-y-4">
                <button
                  onClick={handleCentralAuthRedirect}
                  className="w-full bg-gradient-to-r from-emerald-600 to-emerald-700 text-white px-6 py-3 rounded-xl font-bold hover:from-emerald-700 hover:to-emerald-800 transition-all duration-200 shadow-lg hover:shadow-xl"
                >
                  Continue to Login
                </button>
                
                <p className="text-xs text-gray-500">
                  Don't have an account?{' '}
                  <button
                    onClick={handleRegisterRedirect}
                    className="text-emerald-600 hover:text-emerald-700 font-medium underline"
                  >
                    Create one here
                  </button>
                </p>
              </div>

              {/* Platform indicator */}
              <div className="mt-8 pt-6 border-t border-gray-200/50">
                <p className="text-xs text-gray-500 mb-2">DharmaMind Ecosystem:</p>
                <div className="flex justify-center gap-2 text-xs">
                  <span className="px-2 py-1 bg-emerald-100 text-emerald-700 rounded-full">Main Site</span>
                  <span className="px-2 py-1 bg-gray-100 text-gray-600 rounded-full">AI Chat</span>
                  <span className="px-2 py-1 bg-gray-100 text-gray-600 rounded-full">Community</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Auto-redirect script */}
      <script
        dangerouslySetInnerHTML={{
          __html: `
            // Auto-redirect after 3 seconds if not already redirected
            setTimeout(function() {
              if (window.location.pathname === '/auth') {
                const mode = '${router.query.mode || 'login'}';
                const returnUrl = '${router.query.returnUrl || '/'}';
                window.location.href = 'https://dharmamind.ai/auth?mode=' + mode + '&returnUrl=' + encodeURIComponent(window.location.origin + '/auth?returnUrl=' + encodeURIComponent(returnUrl)) + '&platform=brand';
              }
            }, 3000);
          `
        }}
      />
    </>
  );
}

