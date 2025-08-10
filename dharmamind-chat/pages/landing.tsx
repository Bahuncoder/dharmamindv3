import React from 'react';
import Head from 'next/head';
import Link from 'next/link';
import Logo from '../components/Logo';

const LandingPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>DharmaMind-AI with Soul. Powered by Dharma.</title>
        <meta name="description" content="Experience beta AI conversations guided by dharmic wisdom. Simple, direct chat with conscious technology." />
      </Head>

      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-amber-50 to-stone-50">
        
        {/* Header */}
        <header className="border-b border-stone-200/50 bg-white/95 backdrop-blur-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <Logo size="sm" showText={true} />
              <div className="flex items-center space-x-4">
                <Link
                  href="/login"
                  className="text-stone-600 hover:text-stone-900 font-medium"
                >
                  Sign In
                </Link>
                <Link
                  href="/?demo=true"
                  className="bg-gradient-to-r from-amber-500 to-emerald-500 text-white px-4 py-2 rounded-lg font-medium hover:from-amber-600 hover:to-emerald-600 transition-colors"
                >
                  Try Beta Demo
                </Link>
              </div>
            </div>
          </div>
        </header>

        {/* Hero Section */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          
          <div className="text-center mb-16">
            <h1 className="text-5xl font-bold text-stone-900 mb-6">
              AI with <span className="text-transparent bg-clip-text bg-gradient-to-r from-amber-600 to-emerald-600">Soul powered by Dharma</span>
            </h1>
            <p className="text-xl text-stone-600 mb-8 max-w-3xl mx-auto">
              Experience beta AI conversations that transcend information to offer genuine wisdom, 
              guided by timeless spiritual principles. No setup required - just start chatting.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                href="/?demo=true"
                className="bg-gradient-to-r from-amber-500 to-emerald-500 text-white px-8 py-4 rounded-xl font-semibold text-lg hover:from-amber-600 hover:to-emerald-600 transition-colors shadow-lg"
              >
                🚀 Try Beta Demo
              </Link>
              <Link
                href="/auth?mode=signup"
                className="bg-white text-stone-700 px-8 py-4 rounded-xl font-semibold text-lg border border-stone-300 hover:bg-stone-50 transition-colors"
              >
                📝 Sign Up
              </Link>
            </div>
            
            <p className="text-sm text-stone-500 mt-4">
              No setup required • Start chatting immediately
            </p>
          </div>

          {/* Features Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
            
            <div className="bg-white rounded-xl shadow-sm border border-stone-200 p-6 text-center">
              <div className="text-4xl mb-4">🧘</div>
              <h3 className="text-xl font-semibold text-stone-900 mb-3">Spiritual Guidance</h3>
              <p className="text-stone-600">
                AI conversations rooted in dharmic wisdom and spiritual principles for meaningful insights.
              </p>
            </div>

            <div className="bg-white rounded-xl shadow-sm border border-stone-200 p-6 text-center">
              <div className="text-4xl mb-4">💬</div>
              <h3 className="text-xl font-semibold text-stone-900 mb-3">Pure Chat Experience</h3>
              <p className="text-stone-600">
                Streamlined interface focused entirely on meaningful conversations without distractions.
              </p>
            </div>

            <div className="bg-white rounded-xl shadow-sm border border-stone-200 p-6 text-center">
              <div className="text-4xl mb-4">🌱</div>
              <h3 className="text-xl font-semibold text-stone-900 mb-3">Conscious Technology</h3>
              <p className="text-stone-600">
                Technology designed with awareness, serving your spiritual growth and conscious evolution.
              </p>
            </div>

          </div>

          {/* CTA Section */}
          <div className="bg-gradient-to-r from-amber-50 to-emerald-50 rounded-2xl border border-amber-200 p-8 text-center">
            <h2 className="text-3xl font-bold text-stone-900 mb-4">
              Ready for Beta AI Wisdom Chat?
            </h2>
            <p className="text-stone-600 mb-6 text-lg">
              Jump straight into meaningful conversations with our beta AI. No setup, no barriers - just wisdom.
            </p>
            <Link
              href="/?demo=true"
              className="bg-gradient-to-r from-amber-500 to-emerald-500 text-white px-8 py-4 rounded-xl font-semibold text-lg hover:from-amber-600 hover:to-emerald-600 transition-colors shadow-lg inline-block"
            >
              Start Beta Chat 🚀
            </Link>
          </div>

        </main>

        {/* Footer */}
        <footer className="border-t border-stone-200 bg-stone-50 py-8">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <p className="text-stone-600">
              Built with ❤️ for conscious technology and spiritual growth
            </p>
            <div className="mt-4 space-x-6">
              <Link href="/demo-links" className="text-stone-500 hover:text-stone-700">
                Demo Links
              </Link>
              <Link href="/auth?mode=login" className="text-stone-500 hover:text-stone-700">
                Sign In
              </Link>
              <Link href="/privacy" className="text-stone-500 hover:text-stone-700">
                Privacy
              </Link>
              <Link href="/terms" className="text-stone-500 hover:text-stone-700">
                Terms
              </Link>
            </div>
          </div>
        </footer>

      </div>
    </>
  );
};

export default LandingPage;
