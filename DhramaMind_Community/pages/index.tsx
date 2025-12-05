/**
 * 🕉️ DharmaMind Community - Digital Sangha
 * dharmamind.org
 * 
 * Purpose: Connection, Community, Spiritual Growth Together
 * Accent: Sacred Gold (#f59e0b)
 */

import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import Navigation from '../components/Navigation';

// Animation variants
const fadeInUp = {
  initial: { opacity: 0, y: 30 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6 }
};

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.1
    }
  }
};

const HomePage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('all');
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  // Community stats
  const stats = [
    { number: '15K+', label: 'Community Members', icon: '👥' },
    { number: '500+', label: 'Daily Discussions', icon: '💬' },
    { number: '108', label: 'Wisdom Topics', icon: '📿' },
    { number: '50+', label: 'Countries United', icon: '🌍' }
  ];

  // Features of the community
  const features = [
    {
      icon: '🧘',
      title: 'Guided Sangha Sessions',
      description: 'Join live meditation and discussion sessions with spiritual guides and fellow seekers from around the world.',
      color: 'brand',
      gradient: 'from-brand-500/10 to-brand-600/20'
    },
    {
      icon: '📚',
      title: 'Wisdom Library',
      description: 'Access our vast collection of dharmic teachings, scriptures, and interpretations curated by scholars.',
      color: 'gold',
      gradient: 'from-gold-500/10 to-gold-600/20'
    },
    {
      icon: '💡',
      title: 'Ask the Rishis',
      description: 'Get personalized spiritual guidance from our AI Rishis - 9 wise sages with unique domains of wisdom.',
      color: 'spiritual',
      gradient: 'from-spiritual-500/10 to-spiritual-600/20'
    },
    {
      icon: '🤝',
      title: 'Study Circles',
      description: 'Form or join study groups focused on specific texts, practices, or philosophical traditions.',
      color: 'wisdom',
      gradient: 'from-wisdom-500/10 to-wisdom-600/20'
    },
    {
      icon: '🌟',
      title: 'Growth Tracking',
      description: 'Track your spiritual journey with insights, milestones, and personalized practice recommendations.',
      color: 'brand',
      gradient: 'from-brand-500/10 to-gold-500/20'
    },
    {
      icon: '🎯',
      title: 'Daily Dharma',
      description: 'Receive daily wisdom, mantras, and contemplation topics tailored to your spiritual path.',
      color: 'gold',
      gradient: 'from-gold-500/10 to-brand-500/20'
    }
  ];

  // Recent discussions
  const discussions = [
    { title: 'Understanding Karma in Modern Life', author: 'Arjun S.', replies: 45, category: 'Philosophy' },
    { title: 'Meditation Techniques for Beginners', author: 'Priya M.', replies: 89, category: 'Practice' },
    { title: 'The Role of AI in Spiritual Growth', author: 'Ravi K.', replies: 67, category: 'Technology' },
    { title: 'Bhagavad Gita Chapter 2 Study', author: 'Deepa L.', replies: 112, category: 'Scripture' }
  ];

  return (
    <>
      <Head>
        <title>DharmaMind Community - Digital Sangha | AI with Soul</title>
        <meta name="description" content="Join the DharmaMind community - a digital sangha where seekers connect, learn, and grow together through dharmic wisdom and conscious AI." />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="theme-color" content="#10b981" />
        <link rel="icon" href="/logo.jpeg" />
      </Head>
      
      <div className="min-h-screen bg-earth-50">
        <Navigation />
        
        <main className="w-full">
          {/* Hero Section */}
          <section className="relative min-h-[90vh] flex items-center justify-center overflow-hidden">
            {/* Animated Background */}
            <div className="absolute inset-0">
              {/* Mesh gradient */}
              <div className="absolute inset-0 bg-mesh-brand opacity-50"></div>
              <div className="absolute inset-0 bg-gradient-to-br from-earth-50 via-brand-50/30 to-gold-50/20"></div>
              
              {/* Floating orbs with community warmth */}
              <motion.div
                className="absolute top-20 left-10 w-72 h-72 rounded-full blur-3xl"
                style={{ background: 'radial-gradient(circle, rgba(16, 185, 129, 0.15), transparent)' }}
                animate={{
                  scale: [1, 1.2, 1],
                  x: [0, 30, 0],
                  y: [0, -20, 0]
                }}
                transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }}
              />
              <motion.div
                className="absolute bottom-20 right-10 w-96 h-96 rounded-full blur-3xl"
                style={{ background: 'radial-gradient(circle, rgba(245, 158, 11, 0.12), transparent)' }}
                animate={{
                  scale: [1, 1.1, 1],
                  x: [0, -20, 0],
                  y: [0, 30, 0]
                }}
                transition={{ duration: 10, repeat: Infinity, ease: 'easeInOut', delay: 1 }}
              />
              <motion.div
                className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full blur-3xl"
                style={{ background: 'radial-gradient(circle, rgba(139, 92, 246, 0.08), transparent)' }}
                animate={{ scale: [1, 1.05, 1] }}
                transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut' }}
              />
            </div>

            <div className="relative z-10 max-w-6xl mx-auto px-4 py-20 text-center">
              {/* Product Ecosystem Navigation */}
              <motion.div 
                className="flex justify-center gap-2 mb-8"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
              >
                <a href="https://dharmamind.ai" className="px-4 py-2 rounded-full text-sm font-medium bg-white/60 backdrop-blur-xl border border-brand-200 text-brand-700 hover:bg-brand-50 transition-all">
                  💬 Chat
                </a>
                <span className="px-4 py-2 rounded-full text-sm font-bold bg-gold-500 text-white shadow-gold">
                  🌐 Community
                </span>
                <a href="https://dharmamind.com" className="px-4 py-2 rounded-full text-sm font-medium bg-white/60 backdrop-blur-xl border border-spiritual-200 text-spiritual-700 hover:bg-spiritual-50 transition-all">
                  🏢 Enterprise
                </a>
              </motion.div>

              {/* Badge */}
              <motion.div 
                className="inline-flex items-center gap-3 px-6 py-3 bg-white/80 backdrop-blur-xl rounded-full border border-gold-200 shadow-lg mb-8"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.3 }}
              >
                <motion.span 
                  className="text-3xl"
                  animate={{ rotate: [0, 10, -10, 0] }}
                  transition={{ duration: 4, repeat: Infinity }}
                >
                  🙏
                </motion.span>
                <span className="text-base font-semibold text-earth-800">Digital Sangha</span>
                <span className="px-2 py-0.5 rounded-full bg-gold-100 text-gold-700 text-xs font-bold">LIVE</span>
              </motion.div>

              {/* Main Heading */}
              <motion.h1 
                className="text-6xl md:text-8xl font-black text-earth-900 mb-6 leading-tight tracking-tight"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4, duration: 0.8 }}
              >
                DharmaMind{' '}
                <span className="bg-gradient-to-r from-gold-500 to-gold-600 bg-clip-text text-transparent">
                  Community
                </span>
              </motion.h1>
              
              {/* Subheading */}
              <motion.p 
                className="text-2xl md:text-3xl font-bold text-earth-700 mb-6"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.6 }}
              >
                Where Seekers <span className="text-brand-600">Connect</span> • <span className="text-gold-600">Learn</span> • <span className="text-spiritual-600">Grow</span>
              </motion.p>
              
              {/* Description */}
              <motion.p 
                className="text-lg md:text-xl text-earth-600 mb-12 max-w-3xl mx-auto leading-relaxed"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.7 }}
              >
                Join thousands of spiritual seekers in our digital sangha. Share wisdom, 
                participate in guided sessions, and grow together on your dharmic journey.
              </motion.p>

              {/* CTA Buttons */}
              <motion.div 
                className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-16"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.8 }}
              >
                <Link 
                  href="/community"
                  className="group relative px-10 py-4 bg-gradient-to-r from-gold-500 to-gold-600 text-white rounded-2xl font-bold text-lg shadow-gold hover:shadow-gold-lg transition-all duration-300 hover:-translate-y-1 overflow-hidden"
                >
                  <span className="relative z-10 flex items-center gap-2">
                    Join the Sangha
                    <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                    </svg>
                  </span>
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></div>
                </Link>
                
                <a 
                  href="https://dharmamind.ai"
                  className="px-10 py-4 bg-white text-earth-800 rounded-2xl font-bold text-lg border-2 border-brand-500 hover:bg-brand-50 transition-all duration-300 hover:-translate-y-1 flex items-center gap-2"
                >
                  <span>🤖</span>
                  Try AI Chat
                </a>
              </motion.div>

              {/* Stats */}
              <motion.div 
                className="grid grid-cols-2 md:grid-cols-4 gap-6"
                variants={staggerContainer}
                initial="initial"
                animate="animate"
              >
                {stats.map((stat, index) => (
                  <motion.div 
                    key={index} 
                    className="bg-white/70 backdrop-blur-xl rounded-2xl p-6 border border-earth-100 shadow-sm hover:shadow-lg transition-all duration-300 hover:-translate-y-1"
                    variants={fadeInUp}
                    whileHover={{ scale: 1.02 }}
                  >
                    <span className="text-3xl mb-2 block">{stat.icon}</span>
                    <div className="text-3xl md:text-4xl font-black text-earth-900 mb-1">{stat.number}</div>
                    <div className="text-sm font-medium text-earth-500">{stat.label}</div>
                  </motion.div>
                ))}
              </motion.div>
            </div>

            {/* Scroll Indicator */}
            <motion.div 
              className="absolute bottom-8 left-1/2 -translate-x-1/2"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1.5 }}
            >
              <motion.div 
                className="w-8 h-12 border-2 border-gold-400 rounded-full flex justify-center cursor-pointer"
                animate={{ y: [0, 5, 0] }}
                transition={{ duration: 2, repeat: Infinity }}
                onClick={() => window.scrollTo({ top: window.innerHeight, behavior: 'smooth' })}
              >
                <motion.div 
                  className="w-2 h-4 bg-gradient-to-b from-gold-500 to-brand-500 rounded-full mt-2"
                  animate={{ opacity: [1, 0.3, 1], y: [0, 8, 0] }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
              </motion.div>
            </motion.div>
          </section>

          {/* Features Section */}
          <section className="py-24 bg-white relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-brand-500 via-gold-500 to-spiritual-500 opacity-50"></div>
            
            <div className="max-w-7xl mx-auto px-4">
              <motion.div 
                className="text-center mb-16"
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
              >
                <div className="inline-flex items-center gap-2 px-4 py-2 bg-gold-50 rounded-full border border-gold-200 mb-6">
                  <span className="text-xl">✨</span>
                  <span className="text-sm font-semibold text-gold-700">Community Features</span>
                </div>
                <h2 className="text-4xl md:text-5xl font-black text-earth-900 mb-6">
                  Everything You Need to{' '}
                  <span className="bg-gradient-to-r from-brand-500 to-gold-500 bg-clip-text text-transparent">Grow Together</span>
                </h2>
                <p className="text-xl text-earth-600 max-w-3xl mx-auto">
                  Our community is designed to support every aspect of your spiritual journey with tools, 
                  guidance, and connections that matter.
                </p>
              </motion.div>

              <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                {features.map((feature, index) => (
                  <motion.div 
                    key={index}
                    className="group relative bg-white rounded-2xl p-8 border border-earth-100 shadow-sm hover:shadow-xl transition-all duration-500 hover:-translate-y-2 overflow-hidden"
                    initial={{ opacity: 0, y: 30 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: index * 0.1 }}
                  >
                    <div className={`absolute inset-0 bg-gradient-to-br ${feature.gradient} opacity-0 group-hover:opacity-100 transition-opacity duration-500`}></div>
                    
                    <div className="relative z-10">
                      <div className={`w-16 h-16 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300 shadow-md bg-gradient-to-br from-${feature.color}-100 to-${feature.color}-200`}>
                        <span className="text-3xl">{feature.icon}</span>
                      </div>
                      <h3 className="text-xl font-bold text-earth-900 mb-3">{feature.title}</h3>
                      <p className="text-earth-600 leading-relaxed">{feature.description}</p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </section>

          {/* Recent Discussions Section */}
          <section className="py-24 bg-gradient-to-b from-earth-50 to-white">
            <div className="max-w-6xl mx-auto px-4">
              <motion.div 
                className="text-center mb-12"
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
              >
                <div className="inline-flex items-center gap-2 px-4 py-2 bg-white rounded-full border border-brand-200 shadow-sm mb-6">
                  <span className="text-xl">💬</span>
                  <span className="text-sm font-semibold text-brand-700">Active Discussions</span>
                </div>
                <h2 className="text-4xl md:text-5xl font-black text-earth-900 mb-4">
                  Join the Conversation
                </h2>
                <p className="text-xl text-earth-600">See what the community is exploring today</p>
              </motion.div>

              <div className="space-y-4 mb-12">
                {discussions.map((discussion, index) => (
                  <motion.div
                    key={index}
                    className="group bg-white rounded-2xl p-6 border border-earth-100 shadow-sm hover:shadow-lg hover:border-brand-200 transition-all duration-300 cursor-pointer"
                    initial={{ opacity: 0, x: -30 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: index * 0.1 }}
                    whileHover={{ x: 5 }}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <span className="inline-block px-3 py-1 rounded-full text-xs font-semibold bg-brand-100 text-brand-700 mb-2">
                          {discussion.category}
                        </span>
                        <h4 className="text-lg font-bold text-earth-900 group-hover:text-brand-600 transition-colors">
                          {discussion.title}
                        </h4>
                        <p className="text-sm text-earth-500 mt-1">Started by {discussion.author}</p>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-brand-600">{discussion.replies}</div>
                          <div className="text-xs text-earth-500">replies</div>
                        </div>
                        <svg className="w-5 h-5 text-earth-400 group-hover:text-brand-500 group-hover:translate-x-1 transition-all" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>

              <motion.div 
                className="text-center"
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
              >
                <Link 
                  href="/community"
                  className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-brand-500 to-brand-600 text-white rounded-2xl font-bold text-lg shadow-brand hover:shadow-brand-lg transition-all duration-300 hover:-translate-y-1"
                >
                  View All Discussions
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </Link>
              </motion.div>
            </div>
          </section>

          {/* CTA Section */}
          <section className="py-24 relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-cosmos"></div>
            <div className="absolute inset-0 bg-[url('/pattern.svg')] opacity-5"></div>
            
            <div className="relative z-10 max-w-4xl mx-auto px-4 text-center">
              <motion.div 
                className="bg-white/10 backdrop-blur-xl rounded-3xl p-12 md:p-16 border border-white/20 shadow-2xl"
                initial={{ opacity: 0, scale: 0.95 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
              >
                <motion.div 
                  className="text-6xl mb-6"
                  animate={{ rotate: [0, 5, -5, 0], scale: [1, 1.1, 1] }}
                  transition={{ duration: 4, repeat: Infinity }}
                >
                  🙏
                </motion.div>
                
                <h3 className="text-4xl md:text-5xl font-black text-white mb-6">
                  Your Sangha Awaits
                </h3>
                <p className="text-xl text-white/80 mb-10 max-w-2xl mx-auto leading-relaxed">
                  Join thousands of seekers on the path to wisdom. Together, we learn, grow, 
                  and support each other's dharmic journey.
                </p>
                
                <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-8">
                  <Link 
                    href="/community"
                    className="group px-10 py-4 bg-gradient-to-r from-gold-500 to-gold-600 text-white rounded-2xl font-bold text-lg shadow-gold hover:shadow-gold-lg transition-all duration-300 hover:-translate-y-1 flex items-center gap-2"
                  >
                    Join Free Today
                    <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                    </svg>
                  </Link>
                  
                  <a 
                    href="https://dharmamind.ai"
                    className="px-10 py-4 bg-white/10 backdrop-blur text-white rounded-2xl font-bold text-lg border-2 border-white/30 hover:bg-white/20 transition-all duration-300 hover:-translate-y-1"
                  >
                    Explore AI Chat
                  </a>
                </div>
                
                <p className="text-sm text-white/60 flex items-center justify-center gap-4 flex-wrap">
                  <span className="flex items-center gap-1"><span className="text-brand-400">✓</span> Free forever plan</span>
                  <span className="flex items-center gap-1"><span className="text-gold-400">✓</span> Active community</span>
                  <span className="flex items-center gap-1"><span className="text-spiritual-400">✓</span> Expert guidance</span>
                </p>
              </motion.div>
            </div>
          </section>

          {/* Footer */}
          <footer className="py-16 bg-earth-900 text-white">
            <div className="max-w-6xl mx-auto px-4">
              {/* Product Links */}
              <div className="flex flex-wrap justify-center gap-4 mb-12">
                <a href="https://dharmamind.ai" className="px-6 py-3 rounded-xl bg-earth-800 hover:bg-brand-600 transition-colors flex items-center gap-2">
                  <span>💬</span>
                  <span>DharmaMind Chat</span>
                </a>
                <span className="px-6 py-3 rounded-xl bg-gold-600 flex items-center gap-2">
                  <span>🌐</span>
                  <span>Community</span>
                </span>
                <a href="https://dharmamind.com" className="px-6 py-3 rounded-xl bg-earth-800 hover:bg-spiritual-600 transition-colors flex items-center gap-2">
                  <span>🏢</span>
                  <span>Enterprise</span>
                </a>
              </div>

              <div className="flex flex-col md:flex-row items-center justify-between gap-6 pt-8 border-t border-earth-800">
                <div className="flex items-center gap-3">
                  <span className="text-3xl">🕉️</span>
                  <div>
                    <span className="text-xl font-bold">DharmaMind</span>
                    <span className="text-gold-500 ml-2 text-sm font-medium">Community</span>
                  </div>
                </div>
                <p className="text-earth-400 text-sm">
                  © 2024 DharmaMind. AI with Soul Powered by Dharma.
                </p>
                <div className="flex gap-6">
                  <Link href="/privacy" className="text-earth-400 hover:text-brand-400 transition-colors">Privacy</Link>
                  <Link href="/terms" className="text-earth-400 hover:text-brand-400 transition-colors">Terms</Link>
                  <Link href="/contact" className="text-earth-400 hover:text-brand-400 transition-colors">Contact</Link>
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
