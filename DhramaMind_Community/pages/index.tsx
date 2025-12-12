/**
 * DharmaMind Community - Homepage
 * Clean, professional design with improved readability
 */

import React from 'react';
import Head from 'next/head';
import Link from 'next/link';

const HomePage: React.FC = () => {
  const discussions = [
    { id: 1, title: 'Best practices for mindful leadership', author: 'Alex M.', replies: 24, category: 'Leadership' },
    { id: 2, title: 'How AI can support better decision-making', author: 'Sarah K.', replies: 18, category: 'Technology' },
    { id: 3, title: 'Building resilience in uncertain times', author: 'James L.', replies: 31, category: 'Growth' },
    { id: 4, title: 'Integrating wisdom principles in daily work', author: 'Maya P.', replies: 12, category: 'Practice' },
  ];

  const topics = [
    { name: 'Leadership', count: 234 },
    { name: 'Personal Growth', count: 189 },
    { name: 'Decision Making', count: 156 },
    { name: 'Technology', count: 142 },
    { name: 'Wellness', count: 98 },
  ];

  return (
    <>
      <Head>
        <title>DharmaMind Community</title>
        <meta name="description" content="Connect with thoughtful professionals. Share insights, learn, grow together." />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-neutral-50">
        {/* Navigation */}
        <header className="bg-white border-b border-neutral-200">
          <div className="max-w-6xl mx-auto px-6">
            <div className="flex items-center justify-between h-16">
              <Link href="/" className="flex items-center gap-3">
                <img src="/logo.jpeg" alt="DharmaMind" className="w-8 h-8 rounded-lg object-cover" />
                <span className="text-lg font-semibold text-neutral-900">Community</span>
              </Link>

              <nav className="hidden md:flex items-center gap-8">
                <Link href="/discussions" className="text-base text-neutral-600 hover:text-neutral-900 transition-colors">
                  Discussions
                </Link>
                <Link href="/members" className="text-base text-neutral-600 hover:text-neutral-900 transition-colors">
                  Members
                </Link>
                <Link href="/events" className="text-base text-neutral-600 hover:text-neutral-900 transition-colors">
                  Events
                </Link>
              </nav>

              <div className="flex items-center gap-4">
                <Link href="/login" className="text-base text-neutral-600 hover:text-neutral-900 transition-colors">
                  Sign in
                </Link>
                <Link href="/signup" className="px-5 py-2 bg-gold-600 text-white text-base font-medium rounded-lg hover:bg-gold-700 transition-colors">
                  Join
                </Link>
              </div>
            </div>
          </div>
        </header>

        <main>
          {/* Hero */}
          <section className="bg-white border-b border-neutral-200">
            <div className="max-w-6xl mx-auto px-6 py-20">
              <div className="flex justify-center mb-10">
                <img src="/logo.jpeg" alt="DharmaMind" className="w-24 h-24 rounded-2xl shadow-sm object-cover" />
              </div>

              <p className="text-sm font-medium text-neutral-500 uppercase tracking-widest text-center mb-6">
                AI with Soul
              </p>

              <div className="max-w-2xl mx-auto text-center">
                <h1 className="text-4xl md:text-5xl font-semibold text-neutral-900 mb-5">
                  A space for thoughtful conversation
                </h1>
                <p className="text-xl text-neutral-600 mb-8">
                  Connect with people who value depth, growth, and meaningful dialogue.
                </p>
                <div className="flex items-center justify-center gap-4">
                  <Link href="/signup" className="px-8 py-3 bg-gold-600 text-white text-lg font-medium rounded-lg hover:bg-gold-700 transition-colors">
                    Join free
                  </Link>
                  <Link href="/about" className="px-8 py-3 text-neutral-600 text-lg font-medium hover:text-neutral-900 transition-colors">
                    Learn more
                  </Link>
                </div>
              </div>
            </div>
          </section>

          {/* Stats */}
          <section className="bg-white border-b border-neutral-200">
            <div className="max-w-6xl mx-auto px-6 py-10">
              <div className="flex flex-wrap items-center justify-center gap-16 text-center">
                <div>
                  <p className="text-4xl font-bold text-neutral-900">12k+</p>
                  <p className="text-base text-neutral-500 mt-1">Members</p>
                </div>
                <div>
                  <p className="text-4xl font-bold text-neutral-900">5.4k+</p>
                  <p className="text-base text-neutral-500 mt-1">Discussions</p>
                </div>
                <div>
                  <p className="text-4xl font-bold text-neutral-900">89</p>
                  <p className="text-base text-neutral-500 mt-1">Countries</p>
                </div>
              </div>
            </div>
          </section>

          {/* Main Content */}
          <section className="py-12 px-6">
            <div className="max-w-6xl mx-auto">
              <div className="grid lg:grid-cols-3 gap-8">
                {/* Discussions */}
                <div className="lg:col-span-2">
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="text-2xl font-semibold text-neutral-900">Recent Discussions</h2>
                    <Link href="/discussions" className="text-base text-neutral-500 hover:text-neutral-700 transition-colors">
                      View all →
                    </Link>
                  </div>

                  <div className="space-y-4">
                    {discussions.map((discussion) => (
                      <Link
                        key={discussion.id}
                        href={`/discussions/${discussion.id}`}
                        className="block bg-white p-5 rounded-xl border border-neutral-200 hover:border-neutral-300 hover:shadow-sm transition-all"
                      >
                        <div className="flex items-start justify-between gap-4">
                          <div className="flex-1 min-w-0">
                            <span className="inline-block px-3 py-1 text-sm text-neutral-600 bg-neutral-100 rounded-full mb-2">
                              {discussion.category}
                            </span>
                            <h3 className="text-lg font-medium text-neutral-900 mb-2">{discussion.title}</h3>
                            <p className="text-base text-neutral-500">by {discussion.author}</p>
                          </div>
                          <div className="text-right flex-shrink-0">
                            <p className="text-xl font-semibold text-neutral-900">{discussion.replies}</p>
                            <p className="text-sm text-neutral-500">replies</p>
                          </div>
                        </div>
                      </Link>
                    ))}
                  </div>

                  <div className="mt-6 text-center">
                    <Link href="/discussions/new" className="inline-flex items-center gap-2 text-base text-neutral-600 hover:text-neutral-900 transition-colors">
                      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
                      </svg>
                      Start a discussion
                    </Link>
                  </div>
                </div>

                {/* Sidebar */}
                <div className="space-y-6">
                  {/* Topics */}
                  <div className="bg-white p-6 rounded-xl border border-neutral-200">
                    <h3 className="text-lg font-semibold text-neutral-900 mb-4">Popular Topics</h3>
                    <div className="space-y-3">
                      {topics.map((topic, i) => (
                        <Link
                          key={i}
                          href={`/topics/${topic.name.toLowerCase().replace(' ', '-')}`}
                          className="flex items-center justify-between text-base hover:bg-neutral-50 -mx-3 px-3 py-2 rounded-lg transition-colors"
                        >
                          <span className="text-neutral-700">{topic.name}</span>
                          <span className="text-sm text-neutral-500">{topic.count}</span>
                        </Link>
                      ))}
                    </div>
                  </div>

                  {/* Guidelines */}
                  <div className="bg-white p-6 rounded-xl border border-neutral-200">
                    <h3 className="text-lg font-semibold text-neutral-900 mb-4">Community Guidelines</h3>
                    <ul className="text-base text-neutral-600 space-y-3">
                      <li className="flex items-start gap-2">
                        <span className="text-gold-600">•</span>
                        <span>Be respectful and constructive</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-gold-600">•</span>
                        <span>Share knowledge freely</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-gold-600">•</span>
                        <span>Help others grow</span>
                      </li>
                    </ul>
                  </div>

                  {/* CTA */}
                  <div className="bg-gold-600 p-6 rounded-xl">
                    <h3 className="text-lg font-semibold text-white mb-2">Try DharmaMind AI</h3>
                    <p className="text-base text-gold-100 mb-4">
                      Personalized guidance powered by AI.
                    </p>
                    <Link href="https://dharmamind.ai" className="block w-full text-center px-4 py-2.5 bg-white text-gold-700 font-medium rounded-lg text-base hover:bg-gold-50 transition-colors">
                      Get started free
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </main>

        {/* Footer */}
        <footer className="bg-white border-t border-neutral-200 py-10 px-6 mt-8">
          <div className="max-w-6xl mx-auto">
            <div className="flex flex-col md:flex-row items-center justify-between gap-6">
              <div className="flex items-center gap-3">
                <img src="/logo.jpeg" alt="DharmaMind" className="w-6 h-6 rounded object-cover" />
                <span className="text-base text-neutral-500">© 2024 DharmaMind Community</span>
              </div>
              <div className="flex items-center gap-8">
                <Link href="/privacy" className="text-base text-neutral-500 hover:text-neutral-700 transition-colors">Privacy</Link>
                <Link href="/terms" className="text-base text-neutral-500 hover:text-neutral-700 transition-colors">Terms</Link>
                <Link href="https://dharmamind.ai" className="text-base text-neutral-500 hover:text-neutral-700 transition-colors">DharmaMind</Link>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </>
  );
};

export default HomePage;
