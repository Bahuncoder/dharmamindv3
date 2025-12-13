import React from 'react';
import Head from 'next/head';
import Link from 'next/link';
import Logo from '../components/Logo';

const DemoLinksPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>DharmaMind Demo Links</title>
        <meta name="description" content="Demo links for DharmaMind chat app" />
      </Head>

      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-amber-50 to-stone-50">
        <div className="max-w-4xl mx-auto px-4 py-16">

          <div className="text-center mb-12">
            <Logo size="lg" />
            <h1 className="text-3xl font-bold text-stone-900 mt-6 mb-4">
              DharmaMind Beta - Demo Links
            </h1>
            <p className="text-stone-600">
              Simple demo links for the beta chat app
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

            {/* Beta Demo Link */}
            <div className="bg-white rounded-xl shadow-sm border border-stone-200 p-6">
              <div className="text-center">
                <span className="text-4xl mb-4 block">🚀</span>
                <h3 className="text-xl font-semibold text-stone-900 mb-3">
                  Beta Demo Chat
                </h3>
                <p className="text-stone-600 mb-6">
                  Go directly to demo chat mode. Perfect for beta testing and sharing.
                </p>
                <Link
                  href="/chat?demo=true"
                  className="inline-block bg-gradient-to-r from-amber-500 to-gold-500 text-white px-6 py-3 rounded-lg font-medium hover:from-amber-600 hover:to-gold-600 transition-colors"
                >
                  Try Beta Demo
                </Link>
                <div className="mt-4 text-sm text-stone-500">
                  URL: <code>/chat?demo=true</code>
                </div>
              </div>
            </div>

            {/* Direct Chat Link */}
            <div className="bg-white rounded-xl shadow-sm border border-stone-200 p-6">
              <div className="text-center">
                <span className="text-4xl mb-4 block">💬</span>
                <h3 className="text-xl font-semibold text-stone-900 mb-3">
                  Direct Chat
                </h3>
                <p className="text-stone-600 mb-6">
                  Go directly to chat (may redirect based on auth status).
                </p>
                <Link
                  href="/chat"
                  className="inline-block bg-gradient-to-r from-gold-500 to-gold-600 text-white px-6 py-3 rounded-lg font-medium hover:from-gold-600 hover:to-gold-700 transition-colors"
                >
                  Go to Chat
                </Link>
                <div className="mt-4 text-sm text-stone-500">
                  URL: <code>/chat</code>
                </div>
              </div>
            </div>

          </div>

          <div className="mt-12 bg-amber-50 border border-amber-200 rounded-xl p-6">
            <h4 className="text-lg font-semibold text-amber-900 mb-3 flex items-center">
              <span className="mr-2">ℹ️</span>
              Beta Demo Flow
            </h4>
            <div className="space-y-3 text-amber-800">
              <div>
                <strong>Beta Demo Link:</strong> Takes users directly to demo chat mode - no onboarding friction, perfect for testing and sharing.
              </div>
              <div>
                <strong>Direct Chat:</strong> For authenticated users or redirects to appropriate flow based on user status.
              </div>
            </div>
          </div>

          <div className="mt-8 text-center">
            <Link
              href="/"
              className="text-stone-600 hover:text-stone-900 font-medium"
            >
              ← Back to Home
            </Link>
          </div>

        </div>
      </div>
    </>
  );
};

export default DemoLinksPage;
