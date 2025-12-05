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
      
      <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white">
        <Navigation />
        
        {/* Hero Section with Modern Design */}
        <main className="w-full">
          <section className="relative min-h-[90vh] flex items-center justify-center overflow-hidden">
            {/* Animated Background */}
            <div className="absolute inset-0 overflow-hidden">
              {/* Gradient Mesh */}
              <div className="absolute inset-0 bg-gradient-to-br from-gray-50 via-white to-emerald-50/30"></div>
              
              {/* Floating Orbs */}
              <div className="absolute top-20 left-10 w-72 h-72 bg-emerald-500/10 rounded-full blur-3xl animate-float"></div>
              <div className="absolute bottom-20 right-10 w-96 h-96 bg-emerald-400/10 rounded-full blur-3xl animate-float" style={{ animationDelay: '1s' }}></div>
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-emerald-300/5 rounded-full blur-3xl"></div>
            </div>

            <div className="relative z-10 max-w-6xl mx-auto px-4 py-20 text-center">
              {/* Badge */}
              <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/80 backdrop-blur-xl rounded-full border border-emerald-200 shadow-lg mb-8 animate-fade-in-up">
                <span className="text-2xl">🕉️</span>
                <span className="text-sm font-semibold text-gray-800">AI with Soul</span>
              </div>

              {/* Main Heading */}
              <h1 className="text-6xl md:text-8xl font-black text-gray-900 mb-6 leading-tight tracking-tight animate-fade-in-up stagger-1">
                DharmaMind
              </h1>
              
              {/* Subheading */}
              <p className="text-2xl md:text-3xl font-bold text-gray-700 mb-6 animate-fade-in-up stagger-2">
                AI with Soul <span className="text-emerald-600">Powered by Dharma</span>
              </p>
              
              {/* Description */}
              <p className="text-lg md:text-xl text-gray-600 mb-12 max-w-3xl mx-auto leading-relaxed animate-fade-in-up stagger-3">
                Where ancient wisdom meets artificial intelligence. Experience spiritual guidance through 
                conscious AI that honors dharma traditions while embracing modern consciousness exploration.
              </p>

              {/* CTA Buttons */}
              <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-16 animate-fade-in-up stagger-4">
                <a 
                  href="https://dharmamind.ai" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="group relative px-8 py-4 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white rounded-xl font-bold text-lg shadow-accent hover:shadow-accent-lg transition-all duration-300 hover:-translate-y-1 overflow-hidden"
                >
                  <span className="relative z-10 flex items-center gap-2">
                    Experience DharmaMind AI
                    <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                    </svg>
                  </span>
                  {/* Shimmer effect */}
                  <div className="absolute inset-0 -top-1 -bottom-1 bg-gradient-to-r from-transparent via-white/20 to-transparent skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></div>
                </a>
                
                <Link 
                  href="/community"
                  className="px-8 py-4 bg-white text-gray-800 rounded-xl font-bold text-lg border-2 border-emerald-500 hover:bg-emerald-50 transition-all duration-300 hover:-translate-y-1"
                >
                  Join Our Community
                </Link>
              </div>

              {/* Stats */}
              <div className="flex flex-wrap justify-center gap-12 animate-fade-in-up stagger-5">
                {[
                  { number: '10K+', label: 'Active Members' },
                  { number: '500+', label: 'Daily Conversations' },
                  { number: '50+', label: 'Countries' }
                ].map((stat, index) => (
                  <div key={index} className="text-center">
                    <div className="text-4xl md:text-5xl font-black text-gray-900 mb-1">{stat.number}</div>
                    <div className="text-sm font-semibold text-gray-500">{stat.label}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Scroll Indicator */}
            <div className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce">
              <div className="w-6 h-10 border-2 border-emerald-400 rounded-full flex justify-center">
                <div className="w-1.5 h-3 bg-emerald-500 rounded-full mt-2 animate-pulse"></div>
              </div>
            </div>
          </section>

          {/* What We Offer Section */}
          <section className="py-24 bg-white relative overflow-hidden">
            {/* Background decoration */}
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-emerald-500 to-transparent opacity-30"></div>
            
            <div className="max-w-6xl mx-auto px-4">
              <div className="text-center mb-16">
                <div className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-50 rounded-full border border-emerald-200 mb-6">
                  <span className="text-xl">✨</span>
                  <span className="text-sm font-semibold text-emerald-700">What We Offer</span>
                </div>
                <h2 className="text-4xl md:text-5xl font-black text-gray-900 mb-6">
                  Ancient Wisdom, Modern Intelligence
                </h2>
                <p className="text-xl text-gray-600 max-w-3xl mx-auto">
                  Discover how artificial intelligence can deepen your spiritual practice while staying true to timeless dharma principles.
                </p>
              </div>

              <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
                {[
                  {
                    emoji: '🧘‍♂️',
                    title: 'AI-Guided Meditation',
                    description: 'Personalized meditation sessions powered by AI that understands your spiritual journey and adapts to your practice needs.',
                    gradient: 'from-emerald-500/10 to-emerald-600/20'
                  },
                  {
                    emoji: '📿',
                    title: 'Dharma Conversations',
                    description: 'Engage in meaningful discussions about spiritual concepts with AI that draws from Hindu, Vedantic, and Puranic wisdom traditions.',
                    gradient: 'from-teal-500/10 to-emerald-500/20'
                  },
                  {
                    emoji: '🌟',
                    title: 'Conscious Community',
                    description: 'Connect with thousands of practitioners worldwide who are exploring the intersection of spirituality and artificial intelligence.',
                    gradient: 'from-emerald-400/10 to-teal-500/20'
                  }
                ].map((feature, index) => (
                  <div 
                    key={index}
                    className="group relative bg-white rounded-2xl p-8 border border-gray-100 shadow-sm hover:shadow-xl hover:border-emerald-200 transition-all duration-500 hover:-translate-y-2"
                  >
                    {/* Hover gradient */}
                    <div className={`absolute inset-0 bg-gradient-to-br ${feature.gradient} opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-2xl`}></div>
                    
                    <div className="relative z-10">
                      <div className="w-16 h-16 bg-gradient-to-br from-emerald-100 to-emerald-200 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300 shadow-md">
                        <span className="text-3xl">{feature.emoji}</span>
                      </div>
                      <h3 className="text-2xl font-bold text-gray-900 mb-4">{feature.title}</h3>
                      <p className="text-gray-600 leading-relaxed">{feature.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* Featured Content Section */}
          <section className="py-24 bg-gradient-to-b from-gray-50 to-white">
            <div className="max-w-6xl mx-auto px-4">
              <div className="text-center mb-16">
                <div className="inline-flex items-center gap-2 px-4 py-2 bg-white rounded-full border border-emerald-200 shadow-sm mb-6">
                  <span className="text-xl">📚</span>
                  <span className="text-sm font-semibold text-emerald-700">Resources</span>
                </div>
                <h2 className="text-4xl md:text-5xl font-black text-gray-900 mb-6">
                  Explore & Learn
                </h2>
                <p className="text-xl text-gray-600 max-w-3xl mx-auto">
                  Dive into our resources designed to support your spiritual growth and understanding of conscious AI.
                </p>
              </div>

              <div className="grid gap-8 md:grid-cols-2">
                {/* Community Card */}
                <div className="group bg-white rounded-3xl overflow-hidden border border-gray-100 shadow-sm hover:shadow-2xl transition-all duration-500 hover:-translate-y-2">
                  <div className="h-48 bg-gradient-to-br from-emerald-100 to-emerald-200 flex items-center justify-center relative overflow-hidden">
                    <span className="text-7xl group-hover:scale-125 transition-transform duration-500">💬</span>
                    <div className="absolute inset-0 bg-gradient-to-t from-white/50 to-transparent"></div>
                  </div>
                  <div className="p-8">
                    <h3 className="text-2xl font-bold text-gray-900 mb-4">Community Discussions</h3>
                    <p className="text-gray-600 mb-6 leading-relaxed">
                      Join thousands of spiritual seekers in meaningful conversations about dharma, meditation, and the role of AI in spiritual growth.
                    </p>
                    <Link 
                      href="/community"
                      className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white rounded-xl font-bold hover:shadow-accent transition-all duration-300"
                    >
                      Join Community
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                      </svg>
                    </Link>
                  </div>
                </div>

                {/* Blog Card */}
                <div className="group bg-white rounded-3xl overflow-hidden border border-gray-100 shadow-sm hover:shadow-2xl transition-all duration-500 hover:-translate-y-2">
                  <div className="h-48 bg-gradient-to-br from-teal-100 to-emerald-100 flex items-center justify-center relative overflow-hidden">
                    <span className="text-7xl group-hover:scale-125 transition-transform duration-500">📚</span>
                    <div className="absolute inset-0 bg-gradient-to-t from-white/50 to-transparent"></div>
                  </div>
                  <div className="p-8">
                    <h3 className="text-2xl font-bold text-gray-900 mb-4">Wisdom Articles</h3>
                    <p className="text-gray-600 mb-6 leading-relaxed">
                      Read insightful articles on meditation, philosophy, and practical wisdom for integrating dharma into modern life.
                    </p>
                    <Link 
                      href="/blog"
                      className="inline-flex items-center gap-2 px-6 py-3 bg-white text-gray-800 rounded-xl font-bold border-2 border-emerald-500 hover:bg-emerald-50 transition-all duration-300"
                    >
                      Read Blog
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                      </svg>
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* CTA Section */}
          <section className="py-24 relative overflow-hidden">
            {/* Background */}
            <div className="absolute inset-0 bg-gradient-to-br from-emerald-50 via-white to-teal-50"></div>
            <div className="absolute top-0 left-1/4 w-96 h-96 bg-emerald-200/30 rounded-full blur-3xl"></div>
            <div className="absolute bottom-0 right-1/4 w-80 h-80 bg-teal-200/30 rounded-full blur-3xl"></div>
            
            <div className="relative z-10 max-w-4xl mx-auto px-4 text-center">
              <div className="bg-white/80 backdrop-blur-xl rounded-3xl p-12 md:p-16 border border-emerald-100 shadow-xl">
                <div className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-50 rounded-full border border-emerald-200 mb-8">
                  <span className="text-xl">🙏</span>
                  <span className="text-sm font-semibold text-emerald-700">Start Your Journey</span>
                </div>
                
                <h3 className="text-4xl md:text-5xl font-black text-gray-900 mb-6">
                  Begin Your Journey
                </h3>
                <p className="text-xl text-gray-600 mb-10 max-w-2xl mx-auto leading-relaxed">
                  Experience the future of spiritual guidance where ancient dharma wisdom meets cutting-edge AI consciousness. Your enlightened digital journey starts here.
                </p>
                
                <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-8">
                  <a 
                    href="https://dharmamind.ai" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="group px-10 py-4 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white rounded-xl font-bold text-lg shadow-accent hover:shadow-accent-lg transition-all duration-300 hover:-translate-y-1 flex items-center gap-2"
                  >
                    Start Conversation
                    <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                    </svg>
                  </a>
                  
                  <Link 
                    href="/community"
                    className="px-10 py-4 bg-white text-gray-800 rounded-xl font-bold text-lg border-2 border-emerald-500 hover:bg-emerald-50 transition-all duration-300 hover:-translate-y-1"
                  >
                    Explore Community
                  </Link>
                </div>
                
                <p className="text-sm text-gray-500 flex items-center justify-center gap-4 flex-wrap">
                  <span className="flex items-center gap-1"><span className="text-emerald-500">✓</span> Free spiritual AI guidance</span>
                  <span className="flex items-center gap-1"><span className="text-emerald-500">✓</span> Dharma-powered insights</span>
                  <span className="flex items-center gap-1"><span className="text-emerald-500">✓</span> Authentic wisdom</span>
                </p>
              </div>
            </div>
          </section>

          {/* Footer */}
          <footer className="py-12 bg-gray-900 text-white">
            <div className="max-w-6xl mx-auto px-4">
              <div className="flex flex-col md:flex-row items-center justify-between gap-6">
                <div className="flex items-center gap-3">
                  <span className="text-2xl">🕉️</span>
                  <span className="text-xl font-bold">DharmaMind</span>
                </div>
                <p className="text-gray-400 text-sm">
                  © 2024 DharmaMind. AI with Soul Powered by Dharma.
                </p>
                <div className="flex gap-6">
                  <Link href="/privacy" className="text-gray-400 hover:text-emerald-400 transition-colors">Privacy</Link>
                  <Link href="/terms" className="text-gray-400 hover:text-emerald-400 transition-colors">Terms</Link>
                  <Link href="/contact" className="text-gray-400 hover:text-emerald-400 transition-colors">Contact</Link>
                </div>
              </div>
            </div>
          </footer>
        </main>
      </div>
    </>
  );
};

export default HomePage;
