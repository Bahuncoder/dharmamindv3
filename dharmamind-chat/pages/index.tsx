import React, { useEffect, useCallback } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';
import { useAuth } from '../contexts/AuthContext';

const HomePage: React.FC = () => {
  const router = useRouter();
  const { isAuthenticated, isGuest, guestLogin } = useAuth();

  // Quick redirect to chat
  const initializeApp = useCallback(() => {
    setTimeout(() => {
      if (!isAuthenticated && !isGuest) {
        guestLogin();
      }
      router.replace('/chat');
    }, 300);
  }, [router, isAuthenticated, isGuest, guestLogin]);

  useEffect(() => {
    initializeApp();
  }, [initializeApp]);

  return (
    <>
      <Head>
        <title>DharmaMind - AI with Soul</title>
        <meta name="description" content="Experience AI with soul - Conscious conversations guided by dharmic wisdom" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <meta name="theme-color" content="#d4a854" />
        <link rel="icon" type="image/x-icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-neutral-50 dark:bg-neutral-900 flex items-center justify-center">
        <div className="text-center">
          {/* Logo */}
          <div className="w-16 h-16 mx-auto mb-6 rounded-xl overflow-hidden border-2 border-gold-500/30 shadow-lg">
            <img
              src="/logo.jpeg"
              alt="DharmaMind"
              className="w-full h-full object-cover"
            />
          </div>

          {/* Title */}
          <h1 className="text-2xl font-semibold text-neutral-800 dark:text-neutral-100 mb-2">
            DharmaMind
          </h1>
          <p className="text-sm text-neutral-500 dark:text-neutral-400 mb-6">
            AI with Soul
          </p>

          {/* Loading dots */}
          <div className="flex items-center justify-center gap-1">
            <div className="w-1.5 h-1.5 bg-gold-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
            <div className="w-1.5 h-1.5 bg-gold-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
            <div className="w-1.5 h-1.5 bg-gold-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
          </div>
        </div>
      </div>
    </>
  );
};

export default HomePage;
