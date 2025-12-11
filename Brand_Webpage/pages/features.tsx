/**
 * DharmaMind Features - Professional Features Page
 */

import React from 'react';
import Head from 'next/head';
import Link from 'next/link';

const FeaturesPage: React.FC = () => {
  const features = [
    {
      category: 'Intelligence',
      items: [
        {
          title: 'Contextual Understanding',
          description: 'AI that comprehends nuance, context, and the deeper meaning behind your questions.',
        },
        {
          title: 'Specialized Guides',
          description: 'Domain-specific AI assistants for leadership, wellness, philosophy, and personal growth.',
        },
        {
          title: 'Adaptive Learning',
          description: 'The system learns your preferences and communication style over time.',
        },
      ],
    },
    {
      category: 'Principles',
      items: [
        {
          title: 'Ethical Framework',
          description: 'Built on time-tested wisdom principles that prioritize integrity and long-term value.',
        },
        {
          title: 'Balanced Perspective',
          description: 'Guidance that considers multiple viewpoints and encourages thoughtful reflection.',
        },
        {
          title: 'Growth Oriented',
          description: 'Every interaction is designed to support genuine personal and professional development.',
        },
      ],
    },
    {
      category: 'Platform',
      items: [
        {
          title: 'Cross-Platform',
          description: 'Access DharmaMind from web, mobile, or integrate via API into your existing tools.',
        },
        {
          title: 'Conversation History',
          description: 'Your conversations are saved and searchable for future reference.',
        },
        {
          title: 'Privacy First',
          description: 'Your data is encrypted and never used to train external models without consent.',
        },
      ],
    },
  ];

  return (
    <>
      <Head>
        <title>Features - DharmaMind</title>
        <meta name="description" content="Explore DharmaMind features: contextual AI, specialized guides, ethical framework, and enterprise-ready platform." />
      </Head>

      <div className="min-h-screen bg-neutral-100">
        {/* Navigation */}
        <header className="fixed top-0 left-0 right-0 z-50 bg-neutral-100/90 backdrop-blur-sm border-b border-neutral-300">
          <div className="max-w-6xl mx-auto px-6">
            <div className="flex items-center justify-between h-16">
              <Link href="/" className="flex items-center gap-2">
                <div className="w-8 h-8 bg-neutral-900 rounded-lg flex items-center justify-center">
                  <span className="text-white font-bold text-sm">D</span>
                </div>
                <span className="font-semibold text-neutral-900">DharmaMind</span>
              </Link>
              <nav className="hidden md:flex items-center gap-8">
                <Link href="/features" className="text-sm text-neutral-900 font-medium">Features</Link>
                <Link href="/pricing" className="text-sm text-neutral-600 hover:text-neutral-900 transition-colors">Pricing</Link>
                <Link href="/enterprise" className="text-sm text-neutral-600 hover:text-neutral-900 transition-colors">Enterprise</Link>
                <Link href="/about" className="text-sm text-neutral-600 hover:text-neutral-900 transition-colors">About</Link>
              </nav>
              <div className="flex items-center gap-3">
                <Link href="https://dharmamind.ai" className="text-sm text-neutral-600 hover:text-neutral-900 hidden sm:block">Sign in</Link>
                <Link href="https://dharmamind.ai" className="px-4 py-2 bg-neutral-900 text-white text-sm font-medium rounded-lg hover:bg-neutral-800 transition-colors">
                  Get Started
                </Link>
              </div>
            </div>
          </div>
        </header>

        <main>
          {/* Hero */}
          <section className="pt-32 pb-16 px-6">
            <div className="max-w-4xl mx-auto text-center">
              <h1 className="text-5xl font-semibold text-neutral-900 leading-tight tracking-tight mb-6">
                Features that matter
              </h1>
              <p className="text-xl text-neutral-500 max-w-2xl mx-auto leading-relaxed">
                Thoughtfully designed capabilities that help you make better decisions 
                and achieve meaningful growth.
              </p>
            </div>
          </section>

          {/* Features by Category */}
          {features.map((category, i) => (
            <section key={i} className={`py-20 px-6 ${i % 2 === 1 ? 'bg-neutral-50' : ''}`}>
              <div className="max-w-6xl mx-auto">
                <h2 className="text-sm font-medium text-neutral-400 uppercase tracking-wider mb-8">
                  {category.category}
                </h2>
                <div className="grid md:grid-cols-3 gap-8">
                  {category.items.map((item, j) => (
                    <div key={j} className="bg-neutral-100 p-8 rounded-xl border border-neutral-300">
                      <h3 className="text-lg font-semibold text-neutral-900 mb-3">{item.title}</h3>
                      <p className="text-neutral-500 leading-relaxed">{item.description}</p>
                    </div>
                  ))}
                </div>
              </div>
            </section>
          ))}

          {/* Comparison */}
          <section className="py-20 px-6">
            <div className="max-w-4xl mx-auto">
              <h2 className="text-3xl font-semibold text-neutral-900 text-center mb-12">
                Why DharmaMind?
              </h2>
              <div className="grid md:grid-cols-2 gap-6">
                <div className="p-6 rounded-xl border border-neutral-300">
                  <h3 className="font-medium text-neutral-400 mb-4">Traditional AI Chatbots</h3>
                  <ul className="space-y-3">
                    {[
                      'Generic responses',
                      'No ethical framework',
                      'Short-term focused',
                      'One-size-fits-all',
                    ].map((item, i) => (
                      <li key={i} className="flex items-center gap-3 text-neutral-500">
                        <span className="w-1.5 h-1.5 bg-neutral-300 rounded-full"></span>
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="p-6 rounded-xl border border-neutral-900 bg-neutral-900 text-white">
                  <h3 className="font-medium text-neutral-300 mb-4">DharmaMind</h3>
                  <ul className="space-y-3">
                    {[
                      'Context-aware guidance',
                      'Principled decision-making',
                      'Long-term growth focus',
                      'Specialized expertise',
                    ].map((item, i) => (
                      <li key={i} className="flex items-center gap-3">
                        <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </section>

          {/* CTA */}
          <section className="py-20 px-6 bg-neutral-50">
            <div className="max-w-4xl mx-auto text-center">
              <h2 className="text-3xl font-semibold text-neutral-900 mb-4">
                Experience the difference
              </h2>
              <p className="text-lg text-neutral-500 mb-8">
                Start your free trial and see how DharmaMind can help you.
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <Link
                  href="https://dharmamind.ai"
                  className="w-full sm:w-auto px-8 py-3 bg-neutral-900 text-white font-medium rounded-lg hover:bg-neutral-800 transition-colors"
                >
                  Start free trial
                </Link>
                <Link
                  href="/pricing"
                  className="w-full sm:w-auto px-8 py-3 text-neutral-700 font-medium hover:text-neutral-900 transition-colors"
                >
                  View pricing
                </Link>
              </div>
            </div>
          </section>
        </main>

        {/* Footer */}
        <footer className="border-t border-neutral-300 py-12 px-6">
          <div className="max-w-6xl mx-auto">
            <div className="flex flex-col md:flex-row items-center justify-between gap-6">
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 bg-neutral-900 rounded flex items-center justify-center">
                  <span className="text-white font-bold text-xs">D</span>
                </div>
                <span className="text-sm text-neutral-500">© 2024 DharmaMind. All rights reserved.</span>
              </div>
              <div className="flex items-center gap-6">
                <Link href="/privacy" className="text-sm text-neutral-500 hover:text-neutral-700">Privacy</Link>
                <Link href="/terms" className="text-sm text-neutral-500 hover:text-neutral-700">Terms</Link>
                <Link href="/contact" className="text-sm text-neutral-500 hover:text-neutral-700">Contact</Link>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </>
  );
};

export default FeaturesPage;
