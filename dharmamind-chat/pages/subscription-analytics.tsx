/**
 * 🕉️ DharmaMind Enhanced Subscription Analytics Page
 * 
 * Advanced subscription management with usage analytics,
 * predictive insights, and intelligent upgrade recommendations
 */

import React from 'react';
import { GetServerSideProps } from 'next';
import Head from 'next/head';
import { useSession } from 'next-auth/react';
import { motion } from 'framer-motion';
import { useColor } from '../contexts/ColorContext';
import SubscriptionDashboard from '../components/SubscriptionDashboard';
import { useNavigation } from '../hooks/useNavigation';
import Logo from '../components/Logo';

// ===============================
// SUBSCRIPTION ANALYTICS PAGE
// ===============================

const SubscriptionAnalyticsPage: React.FC = () => {
  const { data: session } = useSession();
  const { goToAuth } = useNavigation();
  const { currentTheme } = useColor();

  // Redirect to auth if not logged in (optional, can allow demo mode)
  if (!session) {
    return (
      <div className="min-h-screen bg-neutral-50 dark:bg-neutral-900">
        <Head>
          <title>Subscription Analytics - DharmaMind</title>
          <meta name="description" content="Advanced subscription management and usage analytics for DharmaMind AI spiritual companion" />
        </Head>

        <div className="flex items-center justify-center min-h-screen p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white dark:bg-neutral-800 rounded-lg shadow-lg p-8 max-w-md w-full text-center"
          >
            <Logo size="xl" showText={false} className="justify-center mb-4" />
            <h1 className="text-2xl font-bold text-neutral-900 dark:text-white mb-2">
              Subscription Analytics
            </h1>
            <p className="text-neutral-600 dark:text-neutral-400 mb-6">
              Sign in to view your detailed usage analytics and subscription insights
            </p>

            <div className="space-y-3">
              <button
                onClick={() => goToAuth('login')}
                className="w-full px-4 py-2 text-white font-medium rounded-lg hover:opacity-90 transition-all bg-gold-500 hover:bg-gold-600"
              >
                Sign In
              </button>
              <button
                onClick={() => goToAuth('signup')}
                className="w-full px-4 py-2 font-medium rounded-lg hover:bg-gold-50 dark:hover:bg-gold-900/20 transition-all border border-gold-500 text-gold-600 dark:text-gold-400"
              >
                Sign Up
              </button>
            </div>

            {/* Demo Banner */}
            <div className="mt-6 p-3 bg-neutral-100 dark:bg-neutral-700 border border-neutral-300 dark:border-neutral-600 rounded-lg">
              <p className="text-sm text-neutral-800 dark:text-neutral-200">
                💡 <strong>Demo Mode:</strong> You can still explore basic analytics without signing in
              </p>
              <button className="mt-2 text-xs text-gold-600 dark:text-gold-400 hover:text-gold-700 underline">
                Continue as Guest
              </button>
            </div>
          </motion.div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-neutral-50 dark:bg-neutral-900">
      <Head>
        <title>Subscription Analytics - DharmaMind</title>
        <meta name="description" content="Advanced subscription management and usage analytics for DharmaMind AI" />
        <meta name="keywords" content="subscription, analytics, usage tracking, dharma, spiritual AI" />
      </Head>

      {/* Navigation Header */}
      <header className="bg-white dark:bg-neutral-800 shadow-sm border-b border-neutral-200 dark:border-neutral-700">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Logo size="md" showText={true} />
              <div>
                <p className="text-sm text-neutral-600 dark:text-neutral-400">Subscription Analytics</p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <div className="text-sm text-neutral-600 dark:text-neutral-400">
                Welcome, {session.user?.name || session.user?.email}
              </div>
              <button
                className="px-3 py-1 text-sm font-medium rounded-lg hover:bg-gold-50 dark:hover:bg-gold-900/20 transition-all border border-gold-500/40 text-gold-600 dark:text-gold-400"
              >
                Back to Chat
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="py-8">
        <SubscriptionDashboard />
      </main>

      {/* Footer */}
      <footer className="bg-white dark:bg-neutral-800 border-t border-neutral-200 dark:border-neutral-700 mt-12">
        <div className="max-w-6xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between text-sm text-neutral-600 dark:text-neutral-400">
            <div>
              © 2024 DharmaMind. All rights reserved.
            </div>
            <div className="flex space-x-4">
              <a href="/privacy" className="hover:text-gold-600 dark:hover:text-gold-400">Privacy</a>
              <a href="/terms" className="hover:text-gold-600 dark:hover:text-gold-400">Terms</a>
              <a href="/help" className="hover:text-gold-600 dark:hover:text-gold-400">Help</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

// ===============================
// SERVER-SIDE PROPS
// ===============================

export const getServerSideProps: GetServerSideProps = async (context) => {
  // You can add server-side authentication checks here
  // For now, we'll let the client handle authentication

  return {
    props: {}
  };
};

export default SubscriptionAnalyticsPage;
