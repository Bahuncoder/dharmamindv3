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
      <div className="min-h-screen bg-gradient-to-br from-purple-50 via-neutral-100 to-indigo-50">
        <Head>
          <title>Subscription Analytics - DharmaMind</title>
          <meta name="description" content="Advanced subscription management and usage analytics for DharmaMind AI spiritual companion" />
        </Head>

        <div className="flex items-center justify-center min-h-screen p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white rounded-lg shadow-lg p-8 max-w-md w-full text-center"
          >
            <Logo size="xl" showText={false} className="justify-center mb-4" />
            <h1 className="text-2xl font-bold text-gray-900 mb-2">
              Subscription Analytics
            </h1>
            <p className="text-gray-600 mb-6">
              Sign in to view your detailed usage analytics and subscription insights
            </p>
            
            <div className="space-y-3">
              <button
                onClick={() => goToAuth('login')}
                className="w-full px-4 py-2 text-white font-medium rounded-lg hover:opacity-90 transition-all"
                style={{ backgroundColor: currentTheme.colors.primary }}
              >
                Sign In
              </button>
              <button
                onClick={() => goToAuth('signup')}
                className="w-full px-4 py-2 font-medium rounded-lg hover:bg-opacity-10 transition-all"
                style={{ 
                  borderColor: currentTheme.colors.primary,
                  color: currentTheme.colors.primary,
                  borderWidth: '1px',
                  borderStyle: 'solid'
                }}
              >
                Sign Up
              </button>
            </div>

            {/* Demo Banner */}
            <div className="mt-6 p-3 bg-neutral-100 border border-neutral-300 rounded-lg">
              <p className="text-sm text-neutral-800">
                💡 <strong>Demo Mode:</strong> You can still explore basic analytics without signing in
              </p>
              <button className="mt-2 text-xs text-gold-600 hover:text-neutral-800 underline">
                Continue as Guest
              </button>
            </div>
          </motion.div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-neutral-100 to-indigo-50">
      <Head>
        <title>Subscription Analytics - DharmaMind</title>
        <meta name="description" content="Advanced subscription management and usage analytics for DharmaMind AI" />
        <meta name="keywords" content="subscription, analytics, usage tracking, dharma, spiritual AI" />
      </Head>

      {/* Navigation Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Logo size="md" showText={true} />
              <div>
                <p className="text-sm text-gray-600">Subscription Analytics</p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-600">
                Welcome, {session.user?.name || session.user?.email}
              </div>
              <button 
                className="px-3 py-1 text-sm font-medium rounded-lg hover:bg-opacity-10 transition-all"
                style={{
                  color: currentTheme.colors.primary,
                  borderColor: currentTheme.colors.primary + '40',
                  borderWidth: '1px',
                  borderStyle: 'solid'
                }}
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
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-6xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between text-sm text-gray-600">
            <div>
              © 2024 DharmaMind. All rights reserved.
            </div>
            <div className="flex space-x-4">
              <a href="/privacy" className="hover:text-gold-600">Privacy</a>
              <a href="/terms" className="hover:text-gold-600">Terms</a>
              <a href="/help" className="hover:text-gold-600">Help</a>
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
