import React, { useEffect } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import Logo from '../components/Logo';
import { CentralizedFeedbackButton } from '../components/CentralizedSupport';

const StatusPage: React.FC = () => {
  const router = useRouter();

  // Redirect to centralized status page immediately
  useEffect(() => {
    const params = new URLSearchParams();
    params.append('source', 'chat_app');

    const statusUrl = `https://dharmamind.com/status?${params.toString()}`;
    window.location.href = statusUrl;
  }, []);

  return (
    <>
      <Head>
        <title>System Status - DharmaMind</title>
        <meta name="description" content="Real-time status of DharmaMind services, API health, and system performance monitoring" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-neutral-50 dark:bg-neutral-900">
        {/* Header */}
        <header className="border-b border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-800">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <Logo
                size="sm"
                showText={true}
                onClick={() => router.push('/')}
              />

              <nav className="flex items-center space-x-8">
                <button
                  onClick={() => router.push('/')}
                  className="text-neutral-600 dark:text-neutral-300 hover:text-neutral-900 dark:hover:text-white text-sm font-medium"
                >
                  Home
                </button>
              </nav>
            </div>
          </div>
        </header>

        {/* Content */}
        <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="text-center mb-16">
            <h1 className="text-4xl font-bold text-neutral-900 dark:text-white mb-6">System Status</h1>
            <p className="text-xl text-neutral-600 dark:text-neutral-400">Redirecting to real-time status monitoring...</p>
          </div>

          {/* Loading State */}
          <div className="text-center py-16">
            <div className="w-16 h-16 mx-auto mb-6 rounded-xl overflow-hidden border-2 border-gold-500/30 shadow-md">
              <img src="/logo.jpeg" alt="DharmaMind" className="w-full h-full object-cover" />
            </div>
            <div className="flex items-center justify-center gap-1 mb-4">
              <div className="w-2 h-2 bg-gold-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
              <div className="w-2 h-2 bg-gold-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
              <div className="w-2 h-2 bg-gold-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
            </div>
            <p className="text-lg text-neutral-600 dark:text-neutral-400 mb-4">Loading system status...</p>
            <p className="text-sm text-neutral-500 dark:text-neutral-500 mb-8">
              You're being redirected to our comprehensive status monitoring at dharmamind.com
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <CentralizedFeedbackButton
                variant="button"
                size="lg"
                className="mx-auto"
              >
                System Status
              </CentralizedFeedbackButton>

              <CentralizedFeedbackButton

                variant="button"
                size="lg"
                className="mx-auto"
              >
                API Status
              </CentralizedFeedbackButton>
            </div>
          </div>

          {/* Quick Status Preview */}
          <div className="grid md:grid-cols-3 gap-6 mt-16">
            <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-sm border border-gold-200 dark:border-gold-900/30 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-neutral-900 dark:text-white">Chat Services</h3>
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-gold-500 rounded-full mr-2"></div>
                  <span className="text-sm text-gold-600 dark:text-gold-400 font-medium">Operational</span>
                </div>
              </div>
              <p className="text-neutral-600 dark:text-neutral-400 text-sm">All chat services running normally</p>
            </div>

            <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-sm border border-gold-200 dark:border-gold-900/30 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-neutral-900 dark:text-white">API Gateway</h3>
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-gold-500 rounded-full mr-2"></div>
                  <span className="text-sm text-gold-600 dark:text-gold-400 font-medium">Operational</span>
                </div>
              </div>
              <p className="text-neutral-600 dark:text-neutral-400 text-sm">API responding normally</p>
            </div>

            <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-sm border border-gold-200 dark:border-gold-900/30 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-neutral-900 dark:text-white">Authentication</h3>
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-gold-500 rounded-full mr-2"></div>
                  <span className="text-sm text-gold-600 dark:text-gold-400 font-medium">Operational</span>
                </div>
              </div>
              <p className="text-neutral-600 dark:text-neutral-400 text-sm">Login services operational</p>
            </div>
          </div>

          <div className="mt-8 text-center">
            <p className="text-sm text-neutral-500 dark:text-neutral-500">
              For detailed metrics, incident history, and real-time monitoring, visit our comprehensive status page.
            </p>
          </div>
        </main>
      </div>
    </>
  );
};

export default StatusPage;
