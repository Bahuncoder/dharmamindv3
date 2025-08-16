import React from 'react';
import Head from 'next/head';
import Navigation from '../components/Navigation';

const BlogPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>DharmaMind Blog - AI with Soul Powered by Dharma</title>
        <meta name="description" content="Explore wisdom, insights, and transformative content from DharmaMind - where ancient dharma meets modern AI consciousness." />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/logo.jpeg" />
      </Head>
      <div className="min-h-screen bg-secondary-bg">
        <Navigation />
        <main className="w-full max-w-6xl mx-auto px-4 py-12">
          {/* Hero Section */}
          <section className="text-center mb-16">
            <h1 className="text-6xl md:text-7xl font-black text-primary mb-8 leading-tight tracking-tight">
              DharmaMind <span className="text-secondary font-black">Blog</span>
            </h1>
            <p className="text-2xl font-bold text-secondary mb-6 max-w-3xl mx-auto tracking-wide">
              AI with Soul Powered by Dharma
            </p>
            <p className="text-xl text-secondary mb-8 max-w-3xl mx-auto leading-relaxed font-semibold">
              Explore timeless wisdom, AI consciousness insights, and transformative practices that bridge ancient dharma with modern artificial intelligence through DharmaMind.
            </p>
            <div className="flex flex-wrap justify-center gap-4 text-sm text-muted">
              <span className="px-3 py-1 bg-primary-gradient-light rounded-full">Mindfulness</span>
              <span className="px-3 py-1 bg-primary-gradient-light-alt rounded-full">Spiritual Growth</span>
              <span className="px-3 py-1 bg-primary-gradient-light rounded-full">Community Stories</span>
              <span className="px-3 py-1 bg-primary-gradient-light-alt rounded-full">Ancient Wisdom</span>
            </div>
          </section>

          {/* Featured Article */}
          <section className="mb-16">
            <div className="card-elevated overflow-hidden">
              <div className="grid md:grid-cols-2 gap-0">
                <div className="bg-primary-gradient-light flex items-center justify-center h-64 md:h-auto">
                  <div className="text-center">
                    <span className="text-6xl mb-4 block">🕉</span>
                    <span className="text-lg font-medium text-primary">Featured Article</span>
                  </div>
                </div>
                <div className="p-8 md:p-12">
                  <div className="text-sm text-muted mb-3 flex items-center gap-2">
                    <span className="px-2 py-1 bg-primary text-white text-xs rounded">Featured</span>
                    <span>August 5, 2025</span>
                    <span>•</span>
                    <span>12 min read</span>
                  </div>
                  <h2 className="text-4xl font-black text-primary mb-6 leading-tight tracking-tight">
                    The Art of Mindful Living: Integrating Ancient Wisdom into Modern Life
                  </h2>
                  <p className="text-secondary mb-6 leading-relaxed font-medium">
                    In our fast-paced world, the ancient art of mindfulness offers a sanctuary of peace and clarity. Discover how traditional meditation practices can transform your daily routine and bring deeper meaning to every moment.
                  </p>
                  <button className="btn-primary px-6 py-3 rounded-lg font-bold hover:opacity-90 transition-opacity">
                    Read Full Article →
                  </button>
                </div>
              </div>
            </div>
          </section>

          {/* Categories Filter */}
          <section className="mb-12">
            <div className="flex flex-wrap justify-center gap-3">
              <button className="px-4 py-2 bg-primary text-white rounded-lg font-medium">All Posts</button>
              <button className="px-4 py-2 bg-neutral-100 text-secondary hover:bg-neutral-200 rounded-lg font-medium transition-colors">Meditation</button>
              <button className="px-4 py-2 bg-neutral-100 text-secondary hover:bg-neutral-200 rounded-lg font-medium transition-colors">Philosophy</button>
              <button className="px-4 py-2 bg-neutral-100 text-secondary hover:bg-neutral-200 rounded-lg font-medium transition-colors">Community</button>
              <button className="px-4 py-2 bg-neutral-100 text-secondary hover:bg-neutral-200 rounded-lg font-medium transition-colors">Practices</button>
            </div>
          </section>
          
          {/* Blog posts grid */}
          <section className="grid gap-8 md:grid-cols-2 lg:grid-cols-3 mb-16">
            <article className="card-primary overflow-hidden hover:shadow-lg transition-all duration-300 group">
              <div className="bg-primary-gradient-light h-48 flex items-center justify-center">
                <span className="text-4xl group-hover:scale-110 transition-transform duration-300">🧘‍♀️</span>
              </div>
              <div className="p-6">
                <div className="flex items-center gap-2 text-sm text-muted mb-3">
                  <span className="px-2 py-1 bg-success-light text-success text-xs rounded">Meditation</span>
                  <span>August 3, 2025</span>
                  <span>•</span>
                  <span>8 min read</span>
                </div>
                <h3 className="text-2xl font-black text-primary mb-4 line-clamp-2 tracking-tight">
                  Building a Sustainable Daily Meditation Practice
                </h3>
                <p className="text-secondary mb-4 line-clamp-3 font-semibold">
                  Learn practical strategies to establish and maintain a meditation routine that fits seamlessly into your busy lifestyle, backed by scientific research and ancient wisdom.
                </p>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-8 bg-primary-gradient rounded-full flex items-center justify-center text-white text-sm font-medium">
                      S
                    </div>
                    <span className="text-sm text-muted">Sarah Chen</span>
                  </div>
                  <button className="text-primary hover:underline text-sm font-bold">
                    Read More →
                  </button>
                </div>
              </div>
            </article>
            
            <article className="card-primary overflow-hidden hover:shadow-lg transition-all duration-300 group">
              <div className="bg-primary-gradient-light-alt h-48 flex items-center justify-center">
                <span className="text-4xl group-hover:scale-110 transition-transform duration-300">🌱</span>
              </div>
              <div className="p-6">
                <div className="flex items-center gap-2 text-sm text-muted mb-3">
                  <span className="px-2 py-1 bg-info-light text-info text-xs rounded">Philosophy</span>
                  <span>August 1, 2025</span>
                  <span>•</span>
                  <span>6 min read</span>
                </div>
                <h3 className="text-2xl font-bold text-primary mb-4 line-clamp-2 tracking-tight">
                  The Four Noble Truths in Modern Psychology
                </h3>
                <p className="text-secondary mb-4 line-clamp-3 font-medium">
                  Exploring the profound connections between Buddhist philosophy and contemporary therapeutic approaches to understanding and alleviating human suffering.
                </p>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-8 bg-primary-gradient rounded-full flex items-center justify-center text-white text-sm font-medium">
                      M
                    </div>
                    <span className="text-sm text-muted">Dr. Michael Torres</span>
                  </div>
                  <button className="text-primary hover:underline text-sm font-medium">
                    Read More →
                  </button>
                </div>
              </div>
            </article>
            
            <article className="card-primary overflow-hidden hover:shadow-lg transition-all duration-300 group">
              <div className="bg-primary-gradient-light h-48 flex items-center justify-center">
                <span className="text-4xl group-hover:scale-110 transition-transform duration-300">💫</span>
              </div>
              <div className="p-6">
                <div className="flex items-center gap-2 text-sm text-muted mb-3">
                  <span className="px-2 py-1 bg-warning-light text-warning text-xs rounded">Community</span>
                  <span>July 29, 2025</span>
                  <span>•</span>
                  <span>4 min read</span>
                </div>
                <h3 className="text-2xl font-bold text-primary mb-4 line-clamp-2 tracking-tight">
                  Finding Sangha: The Power of Spiritual Community
                </h3>
                <p className="text-secondary mb-4 line-clamp-3 font-medium">
                  Personal stories from our community members about how connecting with like-minded practitioners has deepened their spiritual journey and personal growth.
                </p>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-8 bg-primary-gradient rounded-full flex items-center justify-center text-white text-sm font-medium">
                      A
                    </div>
                    <span className="text-sm text-muted">Aisha Patel</span>
                  </div>
                  <button className="text-primary hover:underline text-sm font-bold">
                    Read More →
                  </button>
                </div>
              </div>
            </article>

            <article className="card-primary overflow-hidden hover:shadow-lg transition-all duration-300 group">
              <div className="bg-primary-gradient-light-alt h-48 flex items-center justify-center">
                <span className="text-4xl group-hover:scale-110 transition-transform duration-300">📿</span>
              </div>
              <div className="p-6">
                <div className="flex items-center gap-2 text-sm text-muted mb-3">
                  <span className="px-2 py-1 bg-success-light text-success text-xs rounded">Practices</span>
                  <span>July 27, 2025</span>
                  <span>•</span>
                  <span>10 min read</span>
                </div>
                <h3 className="text-2xl font-bold text-primary mb-4 line-clamp-2 tracking-tight">
                  Mala Meditation: A Guide to Mantra Practice
                </h3>
                <p className="text-secondary mb-4 line-clamp-3 font-medium">
                  Discover the transformative power of mala beads and mantra repetition in cultivating focus, intention, and spiritual connection in your daily practice.
                </p>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-8 bg-primary-gradient rounded-full flex items-center justify-center text-white text-sm font-medium">
                      R
                    </div>
                    <span className="text-sm text-muted">Ravi Sharma</span>
                  </div>
                  <button className="text-primary hover:underline text-sm font-medium">
                    Read More →
                  </button>
                </div>
              </div>
            </article>

            <article className="card-primary overflow-hidden hover:shadow-lg transition-all duration-300 group">
              <div className="bg-primary-gradient-light h-48 flex items-center justify-center">
                <span className="text-4xl group-hover:scale-110 transition-transform duration-300">🌸</span>
              </div>
              <div className="p-6">
                <div className="flex items-center gap-2 text-sm text-muted mb-3">
                  <span className="px-2 py-1 bg-info-light text-info text-xs rounded">Philosophy</span>
                  <span>July 25, 2025</span>
                  <span>•</span>
                  <span>7 min read</span>
                </div>
                <h3 className="text-xl font-semibold text-primary mb-3 line-clamp-2">
                  Impermanence and Joy: Lessons from Cherry Blossoms
                </h3>
                <p className="text-secondary mb-4 line-clamp-3">
                  Reflecting on the Buddhist concept of impermanence through the fleeting beauty of cherry blossoms and what it teaches us about appreciating life's precious moments.
                </p>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-8 bg-primary-gradient rounded-full flex items-center justify-center text-white text-sm font-medium">
                      Y
                    </div>
                    <span className="text-sm text-muted">Yuki Tanaka</span>
                  </div>
                  <button className="text-primary hover:underline text-sm font-medium">
                    Read More →
                  </button>
                </div>
              </div>
            </article>

            <article className="card-primary overflow-hidden hover:shadow-lg transition-all duration-300 group">
              <div className="bg-primary-gradient-light-alt h-48 flex items-center justify-center">
                <span className="text-4xl group-hover:scale-110 transition-transform duration-300">🔔</span>
              </div>
              <div className="p-6">
                <div className="flex items-center gap-2 text-sm text-muted mb-3">
                  <span className="px-2 py-1 bg-success-light text-success text-xs rounded">Practices</span>
                  <span>July 23, 2025</span>
                  <span>•</span>
                  <span>5 min read</span>
                </div>
                <h3 className="text-xl font-semibold text-primary mb-3 line-clamp-2">
                  Sound Bath Healing: The Science of Vibrational Therapy
                </h3>
                <p className="text-secondary mb-4 line-clamp-3">
                  Understanding how sound frequencies affect our consciousness and well-being, exploring the therapeutic benefits of singing bowls, chimes, and other healing instruments.
                </p>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-8 bg-primary-gradient rounded-full flex items-center justify-center text-white text-sm font-medium">
                      L
                    </div>
                    <span className="text-sm text-muted">Luna Rodriguez</span>
                  </div>
                  <button className="text-primary hover:underline text-sm font-medium">
                    Read More →
                  </button>
                </div>
              </div>
            </article>
          </section>

          {/* Load More Button */}
          <section className="text-center mb-16">
            <button className="btn-outline px-8 py-3 rounded-lg font-medium hover:bg-primary hover:text-white transition-all duration-200">
              Load More Articles
            </button>
          </section>

          {/* Newsletter Signup */}
          <section className="card-elevated p-8 md:p-12 text-center bg-primary-gradient-light">
            <h3 className="text-4xl font-black text-primary mb-6 tracking-tight">Stay Connected</h3>
            <p className="text-xl text-secondary mb-8 max-w-2xl mx-auto font-medium">
              Subscribe to our newsletter and receive weekly insights, meditation guides, and updates from the DharmaMind community.
            </p>
            <div className="max-w-md mx-auto flex gap-3">
              <input 
                type="email" 
                placeholder="Enter your email address"
                className="flex-1 px-4 py-3 rounded-lg border border-medium focus:border-primary focus:outline-none"
              />
              <button className="btn-primary px-6 py-3 rounded-lg font-bold whitespace-nowrap">
                Subscribe
              </button>
            </div>
            <p className="text-sm text-muted mt-3 font-semibold">
              No spam, unsubscribe at any time.
            </p>
          </section>
        </main>
      </div>
    </>
  );
};

export default BlogPage;
