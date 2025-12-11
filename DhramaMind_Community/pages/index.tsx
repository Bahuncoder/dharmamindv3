<<<<<<< HEAD
/**
 * DharmaMind Community - Homepage
 * Clean, professional design
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
            <div className="flex items-center justify-between h-14">
              <Link href="/" className="flex items-center gap-3">
                <img src="/logo.jpeg" alt="DharmaMind" className="w-7 h-7 rounded-lg object-cover" />
                <span className="font-medium text-neutral-900">Community</span>
              </Link>

              <nav className="hidden md:flex items-center gap-6">
                <Link href="/discussions" className="text-sm text-neutral-600 hover:text-neutral-900 transition-colors">
                  Discussions
                </Link>
                <Link href="/members" className="text-sm text-neutral-600 hover:text-neutral-900 transition-colors">
                  Members
                </Link>
                <Link href="/events" className="text-sm text-neutral-600 hover:text-neutral-900 transition-colors">
                  Events
                </Link>
              </nav>

              <div className="flex items-center gap-3">
                <Link href="/login" className="text-sm text-neutral-600 hover:text-neutral-900 transition-colors">
                  Sign in
                </Link>
                <Link href="/signup" className="px-4 py-1.5 bg-neutral-900 text-white text-sm font-medium rounded-lg hover:bg-neutral-800 transition-colors">
                  Join
                </Link>
              </div>
            </div>
          </div>
        </header>

        <main>
          {/* Hero */}
          <section className="bg-white border-b border-neutral-200">
            <div className="max-w-6xl mx-auto px-6 py-16">
              <div className="flex justify-center mb-8">
                <img src="/logo.jpeg" alt="DharmaMind" className="w-20 h-20 rounded-2xl shadow-sm object-cover" />
              </div>
              
              <p className="text-xs font-medium text-neutral-500 uppercase tracking-widest text-center mb-4">
                AI with Soul
              </p>

              <div className="max-w-xl mx-auto text-center">
                <h1 className="text-3xl font-semibold text-neutral-900 mb-3">
                  A space for thoughtful conversation
                </h1>
                <p className="text-neutral-600 mb-6">
                  Connect with people who value depth, growth, and meaningful dialogue.
                </p>
                <div className="flex items-center justify-center gap-3">
                  <Link href="/signup" className="px-6 py-2 bg-neutral-900 text-white font-medium rounded-lg hover:bg-neutral-800 transition-colors text-sm">
                    Join free
                  </Link>
                  <Link href="/about" className="px-6 py-2 text-neutral-600 font-medium hover:text-neutral-900 transition-colors text-sm">
                    Learn more
                  </Link>
                </div>
              </div>
            </div>
          </section>

          {/* Stats */}
          <section className="bg-white border-b border-neutral-200">
            <div className="max-w-6xl mx-auto px-6 py-6">
              <div className="flex flex-wrap items-center justify-center gap-12 text-center">
                <div>
                  <p className="text-2xl font-semibold text-neutral-900">12k+</p>
                  <p className="text-xs text-neutral-500">Members</p>
                </div>
                <div>
                  <p className="text-2xl font-semibold text-neutral-900">5.4k+</p>
                  <p className="text-xs text-neutral-500">Discussions</p>
                </div>
                <div>
                  <p className="text-2xl font-semibold text-neutral-900">89</p>
                  <p className="text-xs text-neutral-500">Countries</p>
                </div>
              </div>
            </div>
          </section>

          {/* Main Content */}
          <section className="py-8 px-6">
            <div className="max-w-6xl mx-auto">
              <div className="grid lg:grid-cols-3 gap-6">
                {/* Discussions */}
                <div className="lg:col-span-2">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="font-medium text-neutral-900">Recent Discussions</h2>
                    <Link href="/discussions" className="text-xs text-neutral-500 hover:text-neutral-700 transition-colors">
                      View all
                    </Link>
                  </div>

                  <div className="space-y-2">
                    {discussions.map((discussion) => (
                      <Link
                        key={discussion.id}
                        href={`/discussions/${discussion.id}`}
                        className="block bg-white p-4 rounded-lg border border-neutral-200 hover:border-neutral-300 transition-colors"
                      >
                        <div className="flex items-start justify-between gap-4">
                          <div className="flex-1 min-w-0">
                            <span className="inline-block px-2 py-0.5 text-xs text-neutral-500 bg-neutral-100 rounded mb-1.5">
                              {discussion.category}
                            </span>
                            <h3 className="text-sm font-medium text-neutral-900 mb-1">{discussion.title}</h3>
                            <p className="text-xs text-neutral-500">by {discussion.author}</p>
                          </div>
                          <div className="text-right flex-shrink-0">
                            <p className="text-sm font-medium text-neutral-900">{discussion.replies}</p>
                            <p className="text-xs text-neutral-500">replies</p>
                          </div>
                        </div>
                      </Link>
                    ))}
                  </div>

                  <div className="mt-4 text-center">
                    <Link href="/discussions/new" className="inline-flex items-center gap-1.5 text-sm text-neutral-600 hover:text-neutral-900 transition-colors">
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
                      </svg>
                      Start a discussion
                    </Link>
                  </div>
                </div>

                {/* Sidebar */}
                <div className="space-y-4">
                  {/* Topics */}
                  <div className="bg-white p-4 rounded-lg border border-neutral-200">
                    <h3 className="text-sm font-medium text-neutral-900 mb-3">Topics</h3>
                    <div className="space-y-2">
                      {topics.map((topic, i) => (
                        <Link
                          key={i}
                          href={`/topics/${topic.name.toLowerCase().replace(' ', '-')}`}
                          className="flex items-center justify-between text-sm hover:bg-neutral-50 -mx-2 px-2 py-1 rounded transition-colors"
                        >
                          <span className="text-neutral-600">{topic.name}</span>
                          <span className="text-xs text-neutral-500">{topic.count}</span>
                        </Link>
                      ))}
                    </div>
                  </div>

                  {/* Guidelines */}
                  <div className="bg-white p-4 rounded-lg border border-neutral-200">
                    <h3 className="text-sm font-medium text-neutral-900 mb-3">Guidelines</h3>
                    <ul className="text-xs text-neutral-600 space-y-1.5">
                      <li>• Be respectful and constructive</li>
                      <li>• Share knowledge freely</li>
                      <li>• Help others grow</li>
                    </ul>
                  </div>

                  {/* CTA */}
                  <div className="bg-neutral-900 p-4 rounded-lg">
                    <h3 className="text-sm font-medium text-white mb-1">Try DharmaMind AI</h3>
                    <p className="text-xs text-neutral-400 mb-3">
                      Personalized guidance powered by AI.
                    </p>
                    <Link href="http://localhost:3000" className="block w-full text-center px-3 py-1.5 bg-white text-neutral-900 font-medium rounded text-xs hover:bg-neutral-100 transition-colors">
                      Get started
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </main>

        {/* Footer */}
        <footer className="bg-white border-t border-neutral-200 py-8 px-6 mt-8">
          <div className="max-w-6xl mx-auto">
            <div className="flex flex-col md:flex-row items-center justify-between gap-4">
              <div className="flex items-center gap-2">
                <img src="/logo.jpeg" alt="DharmaMind" className="w-5 h-5 rounded object-cover" />
                <span className="text-xs text-neutral-500">© 2024 DharmaMind Community</span>
              </div>
              <div className="flex items-center gap-6">
                <Link href="/privacy" className="text-xs text-neutral-500 hover:text-neutral-700 transition-colors">Privacy</Link>
                <Link href="/terms" className="text-xs text-neutral-500 hover:text-neutral-700 transition-colors">Terms</Link>
                <Link href="http://localhost:3001" className="text-xs text-neutral-500 hover:text-neutral-700 transition-colors">DharmaMind</Link>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </>
  );
};

export default HomePage;
=======
import React from 'react';
import Head from 'next/head';
import Link from 'next/link';
import Navigation from '../components/Navigation';

const HomePage: React.FC = () => {
  return (
    <>
      <Head>
        <title>DharmaMind - AI with Soul Powered by Dharma</title>
        <meta name="description" content="DharmaMind - where ancient dharma wisdom meets modern AI consciousness. Join our spiritual digital sangha and explore conscious artificial intelligence." />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/logo.jpeg" />
      </Head>
      <div className="min-h-screen bg-secondary-bg">
        <Navigation />
        <main className="w-full max-w-6xl mx-auto px-4 py-12">
          {/* Hero Section */}
          <section className="text-center mb-20">
            <h1 className="text-6xl md:text-7xl font-black text-primary mb-8 leading-tight tracking-tight" style={{fontWeight: '900'}}>
              DharmaMind
            </h1>
            <p className="text-2xl font-black text-secondary mb-6 max-w-3xl mx-auto tracking-wide">
              AI with Soul Powered by Dharma
            </p>
            <p className="text-xl text-secondary mb-12 max-w-4xl mx-auto leading-relaxed font-semibold">
              Where ancient wisdom meets artificial intelligence. Experience spiritual guidance through conscious AI that honors dharma traditions while embracing modern consciousness exploration.
            </p>
            <div className="flex flex-col sm:flex-row gap-6 justify-center items-center mb-12">
              <a 
                href="https://dharmamind.ai" 
                target="_blank" 
                rel="noopener noreferrer"
                className="btn-primary px-8 py-4 rounded-lg font-bold text-lg hover:opacity-90 transition-opacity duration-200 shadow-lg"
              >
                Experience DharmaMind AI →
              </a>
              <Link 
                href="/community"
                className="btn-outline px-8 py-4 rounded-lg font-bold text-lg hover:bg-primary hover:text-white transition-all duration-200"
              >
                Join Our Community
              </Link>
            </div>
            <div className="flex justify-center gap-8 text-center">
              <div>
                <div className="text-4xl font-black text-primary">10K+</div>
                <div className="text-sm font-semibold text-muted">Active Members</div>
              </div>
              <div>
                <div className="text-4xl font-black text-primary">500+</div>
                <div className="text-sm font-semibold text-muted">Daily Conversations</div>
              </div>
              <div>
                <div className="text-4xl font-black text-primary">50+</div>
                <div className="text-sm font-semibold text-muted">Countries</div>
              </div>
            </div>
          </section>

          {/* What We Offer */}
          <section className="mb-20">
            <div className="text-center mb-16">
              <h2 className="text-4xl md:text-5xl font-black text-primary mb-6 tracking-tight">
                Ancient Wisdom, Modern Intelligence
              </h2>
              <p className="text-xl text-secondary max-w-3xl mx-auto font-semibold leading-relaxed">
                Discover how artificial intelligence can deepen your spiritual practice while staying true to timeless dharma principles.
              </p>
            </div>
            <div className="grid gap-12 md:grid-cols-2 lg:grid-cols-3">
              <div className="text-center group">
                <div className="bg-primary-gradient-light w-24 h-24 rounded-2xl flex items-center justify-center mb-8 mx-auto group-hover:scale-110 transition-all duration-300 shadow-lg">
                  <span className="text-4xl">🧘‍♂️</span>
                </div>
                <h3 className="text-2xl font-black text-primary mb-4 tracking-tight">AI-Guided Meditation</h3>
                <p className="text-secondary font-semibold leading-relaxed">
                  Personalized meditation sessions powered by AI that understands your spiritual journey and adapts to your practice needs.
                </p>
              </div>
              <div className="text-center group">
                <div className="bg-primary-gradient-light-alt w-24 h-24 rounded-2xl flex items-center justify-center mb-8 mx-auto group-hover:scale-110 transition-all duration-300 shadow-lg">
                  <span className="text-4xl">📿</span>
                </div>
                <h3 className="text-2xl font-black text-primary mb-4 tracking-tight">Dharma Conversations</h3>
                <p className="text-secondary font-semibold leading-relaxed">
                  Engage in meaningful discussions about spiritual concepts with AI that draws from Buddhist, Hindu, and other wisdom traditions.
                </p>
              </div>
              <div className="text-center group">
                <div className="bg-primary-gradient-light w-24 h-24 rounded-2xl flex items-center justify-center mb-8 mx-auto group-hover:scale-110 transition-all duration-300 shadow-lg">
                  <span className="text-4xl">🌟</span>
                </div>
                <h3 className="text-2xl font-black text-primary mb-4 tracking-tight">Conscious Community</h3>
                <p className="text-secondary font-semibold leading-relaxed">
                  Connect with thousands of practitioners worldwide who are exploring the intersection of spirituality and artificial intelligence.
                </p>
              </div>
            </div>
          </section>

          {/* Featured Content */}
          <section className="mb-20">
            <div className="text-center mb-16">
              <h2 className="text-4xl md:text-5xl font-black text-primary mb-6 tracking-tight">
                Explore & Learn
              </h2>
              <p className="text-xl text-secondary max-w-3xl mx-auto font-semibold">
                Dive into our resources designed to support your spiritual growth and understanding of conscious AI.
              </p>
            </div>
            <div className="grid gap-8 md:grid-cols-2">
              <div className="card-elevated overflow-hidden hover:shadow-xl transition-all duration-300 group">
                <div className="bg-primary-gradient-light h-48 flex items-center justify-center">
                  <span className="text-6xl group-hover:scale-110 transition-transform duration-300">💬</span>
                </div>
                <div className="p-8">
                  <h3 className="text-2xl font-black text-primary mb-4 tracking-tight">Community Discussions</h3>
                  <p className="text-secondary mb-6 font-semibold leading-relaxed">
                    Join thousands of spiritual seekers in meaningful conversations about dharma, meditation, and the role of AI in spiritual growth.
                  </p>
                  <Link 
                    href="/community"
                    className="btn-primary px-6 py-3 rounded-lg font-black hover:opacity-90 transition-opacity"
                  >
                    Join Community →
                  </Link>
                </div>
              </div>
              
              <div className="card-elevated overflow-hidden hover:shadow-xl transition-all duration-300 group">
                <div className="bg-primary-gradient-light-alt h-48 flex items-center justify-center">
                  <span className="text-6xl group-hover:scale-110 transition-transform duration-300">📚</span>
                </div>
                <div className="p-8">
                  <h3 className="text-2xl font-black text-primary mb-4 tracking-tight">Wisdom Articles</h3>
                  <p className="text-secondary mb-6 font-semibold leading-relaxed">
                    Read insightful articles on meditation, philosophy, and practical wisdom for integrating dharma into modern life.
                  </p>
                  <Link 
                    href="/blog"
                    className="btn-outline px-6 py-3 rounded-lg font-black hover:bg-primary hover:text-white transition-all"
                  >
                    Read Blog →
                  </Link>
                </div>
              </div>
            </div>
          </section>

          {/* Call to Action */}
          <section className="card-elevated p-12 md:p-16 text-center bg-primary-gradient-light">
            <h3 className="text-4xl md:text-5xl font-black text-primary mb-6 tracking-tight">
              Begin Your Journey
            </h3>
            <p className="text-xl text-secondary mb-10 max-w-3xl mx-auto font-medium leading-relaxed">
              Experience the future of spiritual guidance where ancient dharma wisdom meets cutting-edge AI consciousness. Your enlightened digital journey starts here.
            </p>
            <div className="flex flex-col sm:flex-row gap-6 justify-center items-center">
              <a 
                href="https://dharmamind.ai" 
                target="_blank" 
                rel="noopener noreferrer"
                className="btn-primary px-10 py-4 rounded-lg font-black text-xl hover:opacity-90 transition-opacity duration-200 shadow-lg"
              >
                Start Conversation →
              </a>
              <Link 
                href="/community"
                className="btn-outline px-10 py-4 rounded-lg font-black text-xl hover:bg-primary hover:text-white transition-all duration-200"
              >
                Explore Community
              </Link>
            </div>
            <p className="text-base text-muted mt-8 font-semibold">
              Free spiritual AI guidance • Dharma-powered insights • Authentic consciousness exploration
            </p>
          </section>
        </main>
      </div>
    </>
  );
};

export default HomePage;
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
