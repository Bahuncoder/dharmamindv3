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
