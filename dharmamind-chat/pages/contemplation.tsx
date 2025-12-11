import React from 'react';
import { GetServerSideProps } from 'next';
import Head from 'next/head';
import Link from 'next/link';
import { 
  ArrowLeftIcon, 
  HeartIcon, 
  LightBulbIcon 
} from '@heroicons/react/24/outline';
import DeepContemplationInterface from '../components/DeepContemplationInterface';

interface ContemplationPageProps {
  // Add any server-side props if needed
}

const ContemplationPage: React.FC<ContemplationPageProps> = () => {
  const handleInsightCapture = (insight: string) => {
    console.log('Insight captured:', insight);
    // Here you could save to local storage, send to backend, etc.
  };

  const handleSessionComplete = (summary: any) => {
    console.log('Session completed:', summary);
    // Handle session completion - could show summary, save progress, etc.
  };

  return (
    <>
      <Head>
        <title>Deep Contemplation - DharmaMind</title>
        <meta 
          name="description" 
          content="Experience profound spiritual contemplation with AI-guided meditation, Sanskrit wisdom, and dharmic insights for inner transformation." 
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-indigo-900">
        {/* Header */}
        <header className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border-b border-gray-200 dark:border-gray-700 sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              <div className="flex items-center space-x-4">
                <Link href="/chat" className="flex items-center text-gray-600 dark:text-gray-300 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors">
                  <ArrowLeftIcon className="h-5 w-5 mr-2" />
                  Back to Chat
                </Link>
                <div className="h-6 border-l border-gray-300 dark:border-gray-600"></div>
                <h1 className="text-xl font-semibold text-gray-900 dark:text-white flex items-center">
                  <HeartIcon className="h-6 w-6 text-rose-500 mr-2" />
                  Deep Contemplation
                </h1>
              </div>
              <div className="flex items-center space-x-2">
                <LightBulbIcon className="h-5 w-5 text-amber-500" />
                <span className="text-sm text-gray-600 dark:text-gray-400">AI-Guided Spiritual Practice</span>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Introduction Card */}
          <div className="mb-8 bg-white/70 dark:bg-gray-800/70 backdrop-blur-sm rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                🕉️ Sacred Space for Deep Contemplation
              </h2>
              <p className="text-gray-600 dark:text-gray-300 max-w-3xl mx-auto leading-relaxed">
                Welcome to a sacred space where ancient wisdom meets modern mindfulness. 
                This AI-guided contemplation system draws from the depths of dharmic traditions, 
                offering personalized meditation practices, Sanskrit mantras, and profound insights 
                to support your journey toward inner clarity and spiritual awakening.
              </p>
            </div>
          </div>

          {/* Feature Highlights */}
          <div className="grid md:grid-cols-3 gap-6 mb-8">
            <div className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <div className="text-center">
                <div className="bg-indigo-100 dark:bg-indigo-900/50 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-4">
                  <HeartIcon className="h-6 w-6 text-indigo-600 dark:text-indigo-400" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Personalized Guidance</h3>
                <p className="text-gray-600 dark:text-gray-300 text-sm">
                  AI-powered contemplation tailored to your spiritual path and current needs
                </p>
              </div>
            </div>

            <div className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <div className="text-center">
                <div className="bg-amber-100 dark:bg-amber-900/50 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-amber-600 dark:text-amber-400 text-lg font-bold">ॐ</span>
                </div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Sanskrit Wisdom</h3>
                <p className="text-gray-600 dark:text-gray-300 text-sm">
                  Authentic mantras and teachings from ancient spiritual traditions
                </p>
              </div>
            </div>

            <div className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <div className="text-center">
                <div className="bg-gold-100 dark:bg-gold-900/50 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-4">
                  <LightBulbIcon className="h-6 w-6 text-gold-600 dark:text-gold-400" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Deep Insights</h3>
                <p className="text-gray-600 dark:text-gray-300 text-sm">
                  Capture and reflect on profound realizations during your practice
                </p>
              </div>
            </div>
          </div>

          {/* Deep Contemplation Interface */}
          <div className="bg-white/70 dark:bg-gray-800/70 backdrop-blur-sm rounded-xl shadow-lg border border-gray-200 dark:border-gray-700">
            <DeepContemplationInterface 
              onInsightCapture={handleInsightCapture}
              onSessionComplete={handleSessionComplete}
            />
          </div>

          {/* Footer Note */}
          <div className="mt-8 text-center">
            <p className="text-gray-500 dark:text-gray-400 text-sm italic">
              "The mind is everything. What you think you become." - Buddha
            </p>
          </div>
        </main>
      </div>
    </>
  );
};

export default ContemplationPage;

// Optional: Add server-side props if you need to fetch data
export const getServerSideProps: GetServerSideProps = async (context) => {
  // You could fetch user's contemplation history, preferences, etc.
  return {
    props: {
      // Add any props you want to pass to the component
    },
  };
};
