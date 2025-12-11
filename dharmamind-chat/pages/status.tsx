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

      <div className="min-h-screen bg-gradient-to-br from-green-50 via-white to-gold-50">
        {/* Header */}
        <header className="border-b border-gray-200 bg-white">
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
                  className="text-gray-600 hover:text-gray-900 text-sm font-medium"
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
            <h1 className="text-4xl font-bold text-gray-900 mb-6">System Status</h1>
            <p className="text-xl text-gray-600">Redirecting to real-time status monitoring...</p>
          </div>

          {/* Loading State */}
          <div className="text-center py-16">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-green-600 mx-auto mb-6"></div>
            <p className="text-lg text-gray-600 mb-4">Loading system status...</p>
            <p className="text-sm text-gray-500 mb-8">
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
            <div className="bg-white rounded-lg shadow-sm border border-green-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">Chat Services</h3>
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-green-400 rounded-full mr-2"></div>
                  <span className="text-sm text-success-600 font-medium">Operational</span>
                </div>
              </div>
              <p className="text-gray-600 text-sm">All chat services running normally</p>
            </div>

            <div className="bg-white rounded-lg shadow-sm border border-green-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">API Gateway</h3>
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-green-400 rounded-full mr-2"></div>
                  <span className="text-sm text-success-600 font-medium">Operational</span>
                </div>
              </div>
              <p className="text-gray-600 text-sm">API responding normally</p>
            </div>

            <div className="bg-white rounded-lg shadow-sm border border-green-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">Authentication</h3>
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-green-400 rounded-full mr-2"></div>
                  <span className="text-sm text-success-600 font-medium">Operational</span>
                </div>
              </div>
              <p className="text-gray-600 text-sm">Login services operational</p>
            </div>
          </div>

          <div className="mt-8 text-center">
            <p className="text-sm text-gray-500">
              For detailed metrics, incident history, and real-time monitoring, visit our comprehensive status page.
            </p>
          </div>
        </main>
      </div>
    </>
  );
};

export default StatusPage;
