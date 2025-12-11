<<<<<<< HEAD
/**
 * DharmaMind About - Professional About Page
 * Uses unified Layout component for consistent header/footer
 */

import React from 'react';
import Layout from '../components/layout/Layout';
import Link from 'next/link';
import { siteConfig } from '../config/site.config';

const AboutPage: React.FC = () => {
  const values = [
    {
      icon: (
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
        </svg>
      ),
      title: 'Ancient Wisdom',
      description: 'Grounded in Sanātana Dharma—the eternal principles of truth, righteousness, and cosmic harmony that have guided seekers for millennia.',
    },
    {
      icon: (
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
      ),
      title: 'Modern Intelligence',
      description: 'Powered by cutting-edge AI trained specifically on dharmic texts, philosophical traditions, and contemplative practices.',
    },
    {
      icon: (
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
        </svg>
      ),
      title: 'Ethical Foundation',
      description: 'Built with Ahimsa (non-harm) at its core. We prioritize your wellbeing, privacy, and spiritual growth over engagement metrics.',
    },
    {
      icon: (
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
        </svg>
      ),
      title: 'Universal Access',
      description: 'Making the profound wisdom of dharmic traditions accessible to everyone, regardless of background or prior knowledge.',
    },
  ];

  return (
    <Layout
      title="About"
      description="Learn about DharmaMind's mission to bridge ancient wisdom with modern AI for meaningful personal growth."
    >
      {/* Hero */}
      <section className="pt-32 pb-20 px-6 bg-neutral-100">
        <div className="max-w-4xl mx-auto">
          <p className="text-gold-600 font-medium mb-4">Our Story</p>
          <h1 className="text-5xl font-semibold text-neutral-900 leading-tight tracking-tight mb-6">
            Where ancient wisdom meets<br />modern intelligence
          </h1>
          <p className="text-xl text-neutral-600 max-w-3xl leading-relaxed">
            DharmaMind was founded with a singular vision: to make the timeless wisdom of
            Sanātana Dharma accessible through the power of artificial intelligence—creating
            a bridge between sacred knowledge and everyday life.
          </p>
        </div>
      </section>

      {/* Mission & Vision */}
      <section className="py-20 px-6 bg-neutral-50">
        <div className="max-w-5xl mx-auto">
          <div className="grid md:grid-cols-2 gap-16">
            <div>
              <div className="w-12 h-12 bg-gold-100 rounded-xl flex items-center justify-center mb-6">
                <svg className="w-6 h-6 text-gold-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
              </div>
              <h2 className="text-2xl font-semibold text-neutral-900 mb-4">Our Vision</h2>
              <p className="text-lg text-neutral-600 leading-relaxed">
                A world where everyone has access to a wise, patient guide—one that draws from
                humanity's deepest spiritual traditions to help navigate life's challenges with
                clarity, purpose, and inner peace.
              </p>
            </div>
            <div>
              <div className="w-12 h-12 bg-gold-100 rounded-xl flex items-center justify-center mb-6">
                <svg className="w-6 h-6 text-gold-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                </svg>
              </div>
              <h2 className="text-2xl font-semibold text-neutral-900 mb-4">Our Mission</h2>
              <p className="text-lg text-neutral-600 leading-relaxed">
                To preserve and transmit dharmic wisdom through technology that respects the
                depth of these traditions while making them practical, personalized, and
                immediately applicable to modern life.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* The Problem We're Solving */}
      <section className="py-20 px-6 bg-neutral-100">
        <div className="max-w-3xl mx-auto">
          <h2 className="text-3xl font-semibold text-neutral-900 mb-8 text-center">Why DharmaMind Exists</h2>
          <div className="space-y-6 text-lg text-neutral-600">
            <p>
              In an age of infinite information, we face a paradox: <span className="text-neutral-900 font-medium">the more we know, the less clear we feel</span>.
              We're overwhelmed with content yet starving for meaning. Surrounded by answers, yet uncertain about the right questions.
            </p>
            <p>
              Meanwhile, the world's most profound wisdom traditions—texts like the Bhagavad Gita, Upanishads,
              and Yoga Sutras—remain locked behind barriers of language, cultural context, and scholarly
              interpretation. Their transformative insights feel inaccessible to those who need them most.
            </p>
            <p>
              <span className="text-neutral-900 font-medium">DharmaMind bridges this gap.</span> We've trained
              our AI on the authentic sources of Sanātana Dharma, guided by scholars and practitioners,
              to create a companion that can meet you where you are—whether you're seeking practical guidance
              for daily challenges or deeper understanding of life's eternal questions.
            </p>
          </div>
        </div>
      </section>

      {/* Values */}
      <section className="py-20 px-6 bg-neutral-50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-semibold text-neutral-900 mb-4">Our Principles</h2>
            <p className="text-lg text-neutral-600 max-w-2xl mx-auto">
              The values that guide everything we build
            </p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {values.map((value, i) => (
              <div key={i} className="bg-white p-6 rounded-2xl border border-neutral-200">
                <div className="text-gold-600 mb-4">{value.icon}</div>
                <h3 className="text-lg font-semibold text-neutral-900 mb-3">{value.title}</h3>
                <p className="text-neutral-600 text-sm leading-relaxed">{value.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Founder Section */}
      <section className="py-20 px-6 bg-neutral-100">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-semibold text-neutral-900 mb-4">Leadership</h2>
          </div>
          <div className="bg-white p-8 md:p-12 rounded-2xl border border-neutral-200">
            <div className="flex flex-col md:flex-row items-center md:items-start gap-8">
              <div className="w-32 h-32 bg-gradient-to-br from-gold-100 to-gold-200 rounded-2xl flex items-center justify-center flex-shrink-0">
                <span className="text-4xl font-semibold text-gold-700">SP</span>
              </div>
              <div>
                <h3 className="text-2xl font-semibold text-neutral-900 mb-1">{siteConfig.company.founder}</h3>
                <p className="text-gold-600 font-medium mb-4">Founder & CEO</p>
                <p className="text-neutral-600 leading-relaxed mb-4">
                  A technologist with deep roots in dharmic tradition, {siteConfig.company.founder.split(' ')[0]} founded DharmaMind
                  to solve a problem he experienced personally: the challenge of integrating spiritual
                  wisdom into the demands of modern professional life.
                </p>
                <p className="text-neutral-600 leading-relaxed">
                  "I grew up with these teachings, but it took years to understand how to apply them
                  practically. I built DharmaMind so others wouldn't have to wait that long. Everyone
                  deserves a wise guide available whenever they need one."
                </p>
                <div className="mt-6">
                  <a
                    href={siteConfig.social.twitter}
                    className="text-neutral-400 hover:text-gold-600 transition-colors inline-flex items-center gap-2"
                  >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                    </svg>
                    <span className="text-sm">@{siteConfig.social.twitter.split('/').pop()}</span>
                  </a>
=======
import React from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import Button from '../components/Button';
import Logo from '../components/Logo';
import Footer from '../components/Footer';
import SEOHead from '../components/SEOHead';

const AboutPage: React.FC = () => {
  const router = useRouter();

  const teamMembers = [
    {
      name: "Dr. Arjun Patel",
      role: "Founder & CEO",
      bio: "PhD in Computer Science with 15+ years in AI research. Dedicated practitioner of Vedantic philosophy.",
      image: "🧘‍♂️",
      expertise: ["AI Research", "Vedantic Philosophy", "Product Strategy"]
    },
    {
      name: "Priya Sharma",
      role: "Chief Technology Officer",
      bio: "Former Google AI researcher with expertise in natural language processing and Sanskrit computational linguistics.",
      image: "👩‍💻",
      expertise: ["NLP", "Sanskrit Computing", "Machine Learning"]
    },
    {
      name: "Swami Krishnananda",
      role: "Spiritual Advisor",
      bio: "Renowned scholar of Advaita Vedanta with 30+ years of teaching experience in traditional ashrams.",
      image: "🙏",
      expertise: ["Advaita Vedanta", "Sanskrit", "Spiritual Teaching"]
    },
    {
      name: "Maya Chen",
      role: "Head of Product",
      bio: "Product strategy expert focused on mindful technology and ethical AI development.",
      image: "💡",
      expertise: ["Product Design", "UX Strategy", "Ethical AI"]
    }
  ];

  const values = [
    {
      icon: "🕉️",
      title: "Authentic Wisdom",
      description: "We draw from authentic ancient texts and teachings, ensuring accuracy and respect for traditional knowledge.",
      details: "Our content is verified by Sanskrit scholars and spiritual practitioners to maintain authenticity."
    },
    {
      icon: "🤖",
      title: "Ethical AI",
      description: "Our AI is developed with ethics at its core, prioritizing user wellbeing and responsible technology use.",
      details: "We follow strict ethical guidelines and ensure our AI promotes positive spiritual growth."
    },
    {
      icon: "🔒",
      title: "Privacy First",
      description: "Your spiritual journey is personal. We protect your data with the highest security standards.",
      details: "End-to-end encryption and zero-knowledge architecture protect your conversations."
    },
    {
      icon: "🌍",
      title: "Global Accessibility",
      description: "Making ancient wisdom accessible to everyone, regardless of background or location.",
      details: "Available in multiple languages with culturally sensitive adaptations."
    }
  ];

  const achievements = [
    {
      metric: "50,000+",
      label: "Active Users",
      description: "Spiritual seekers worldwide"
    },
    {
      metric: "1M+",
      label: "Conversations",
      description: "Meaningful dialogues facilitated"
    },
    {
      metric: "99.9%",
      label: "Uptime",
      description: "Reliable spiritual guidance"
    },
    {
      metric: "25+",
      label: "Languages",
      description: "Making wisdom accessible globally"
    }
  ];

  const milestones = [
    {
      year: "2023",
      quarter: "Q1",
      title: "Company Founded",
      description: "DharmaMind was established with a mission to democratize spiritual wisdom through AI.",
      icon: "🌱"
    },
    {
      year: "2023",
      quarter: "Q3",
      title: "Research Phase",
      description: "Extensive research into Sanskrit texts and collaboration with spiritual scholars began.",
      icon: "📚"
    },
    {
      year: "2024",
      quarter: "Q1",
      title: "AI Model Development",
      description: "Started developing our proprietary AI model trained on classical spiritual texts.",
      icon: "🧠"
    },
    {
      year: "2024",
      quarter: "Q3",
      title: "Beta Testing Launch",
      description: "Conducted extensive beta testing with spiritual practitioners and scholars worldwide.",
      icon: "🧪"
    },
    {
      year: "2024",
      quarter: "Q4",
      title: "Platform Refinement",
      description: "Refined the platform based on user feedback and spiritual advisor recommendations.",
      icon: "⚡"
    },
    {
      year: "2025",
      quarter: "Q1",
      title: "Public Launch",
      description: "Officially launched DharmaMind platform to serve seekers on their spiritual journey.",
      icon: "🚀"
    }
  ];

  return (
    <>
      <SEOHead 
        title="About DharmaMind - Ethical AI for Personal Growth & Finding Purpose"
        description="Learn how DharmaMind combines AI technology with wisdom traditions to help you find life purpose, make ethical decisions, and live consciously. Join our community of growth-minded individuals."
        keywords="DharmaMind AI, ethical AI companion, purpose-driven AI guide, conversational philosophy AI, AI for personal ethics, meaningful life AI, conscious living community, personal transformation, finding life purpose"
      />

      <div className="min-h-screen bg-section-light">
        {/* Header */}
        <header className="bg-white border-b border-light sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center space-x-4">
                <button
                  onClick={() => router.back()}
                  className="flex items-center space-x-2 text-gray-600 hover:text-primary transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                  <span className="font-medium">Back</span>
                </button>
                
                <div className="h-6 w-px bg-border-medium"></div>
                
                <Logo 
                  size="sm"
                  onClick={() => router.push('/')}
                  className="cursor-pointer"
                />
              </div>

              <nav className="hidden md:flex items-center space-x-6">
                <button
                  onClick={() => router.push('/features')}
                  className="text-secondary hover:text-primary text-sm font-medium transition-colors"
                >
                  Features
                </button>
                <button
                  onClick={() => router.push('/pricing')}
                  className="text-secondary hover:text-primary text-sm font-medium transition-colors"
                >
                  Pricing
                </button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => router.push('/contact')}
                >
                  Contact Us
                </Button>
              </nav>
            </div>
          </div>
        </header>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          {/* Hero Section */}
          <div className="text-center mb-20">
            <div className="animate-fade-in">
              <h1 className="text-5xl md:text-6xl font-bold text-primary mb-6">
                About DharmaMind
              </h1>
              <div className="w-24 h-1 bg-brand-gradient mx-auto mb-8"></div>
              <p className="text-xl md:text-2xl text-secondary max-w-4xl mx-auto mb-8 leading-relaxed">
                We're on a mission to make ancient spiritual wisdom accessible to everyone through 
                ethical AI technology, bridging the gap between timeless teachings and modern life.
              </p>
              <div className="flex flex-wrap justify-center gap-4 mb-12">
                {achievements.map((achievement, index) => (
                  <div key={index} className="bg-white rounded-xl p-6 shadow-sm border border-light text-center min-w-[150px]">
                    <div className="text-2xl font-bold text-brand-accent mb-1">{achievement.metric}</div>
                    <div className="text-sm font-semibold text-primary mb-1">{achievement.label}</div>
                    <div className="text-xs text-secondary">{achievement.description}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Mission & Vision */}
          <div className="mb-20">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
              <div className="animate-slide-in-left">
                <h2 className="text-4xl font-bold text-primary mb-6">Our Mission</h2>
                <p className="text-secondary text-lg leading-relaxed mb-6">
                  In an increasingly complex world, we believe that ancient wisdom holds the keys to inner peace, 
                  purposeful living, and spiritual growth. DharmaMind was created to make these profound teachings 
                  accessible to anyone seeking guidance on their spiritual journey.
                </p>
                <p className="text-secondary text-lg leading-relaxed mb-8">
                  Through our AI platform, we preserve and share the timeless wisdom of the Vedas, Upanishads, 
                  Bhagavad Gita, and other sacred texts, helping modern seekers apply these teachings to their daily lives.
                </p>
                <div className="space-y-4">
                  <div className="flex items-start space-x-3">
                    <span className="text-brand-accent text-xl">✨</span>
                    <span className="text-secondary">Democratize access to spiritual wisdom</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <span className="text-brand-accent text-xl">🌱</span>
                    <span className="text-secondary">Foster personal and spiritual growth</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <span className="text-brand-accent text-xl">🤝</span>
                    <span className="text-secondary">Bridge ancient wisdom with modern life</span>
                  </div>
                </div>
              </div>
              <div className="content-card text-center animate-slide-in-right">
                <div className="text-8xl mb-6">🧘‍♀️</div>
                <h3 className="text-2xl font-bold text-primary mb-4">Ancient Wisdom, Modern Technology</h3>
                <p className="text-secondary">
                  We combine the profound insights of ancient spiritual traditions with cutting-edge AI 
                  to create a personalized spiritual companion for the digital age.
                </p>
              </div>
            </div>
          </div>

          {/* Core Values */}
          <div className="mb-20">
            <div className="text-center mb-12">
              <h2 className="text-4xl font-bold text-primary mb-4">Our Core Values</h2>
              <p className="text-xl text-secondary max-w-3xl mx-auto">
                These principles guide everything we do and shape how we approach our mission.
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {values.map((value, index) => (
                <div
                  key={index}
                  className="content-card hover:shadow-lg transition-all duration-300 animate-fade-in-up"
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <div className="flex items-start space-x-4">
                    <div className="text-4xl">{value.icon}</div>
                    <div className="flex-1">
                      <h3 className="text-xl font-semibold text-primary mb-3">{value.title}</h3>
                      <p className="text-secondary leading-relaxed mb-3">{value.description}</p>
                      <p className="text-sm text-muted">{value.details}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Team Section */}
          <div className="mb-20">
            <div className="text-center mb-12">
              <h2 className="text-4xl font-bold text-primary mb-4">Meet Our Team</h2>
              <p className="text-xl text-secondary max-w-3xl mx-auto">
                A diverse team of technologists, spiritual practitioners, and scholars working together 
                to bridge ancient wisdom with modern technology.
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
              {teamMembers.map((member, index) => (
                <div
                  key={index}
                  className="content-card text-center hover:shadow-lg transition-all duration-300 animate-fade-in-up"
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <div className="text-6xl mb-4">{member.image}</div>
                  <h3 className="text-lg font-semibold text-primary mb-2">{member.name}</h3>
                  <p className="text-brand-accent font-medium mb-3">{member.role}</p>
                  <p className="text-secondary text-sm leading-relaxed mb-4">{member.bio}</p>
                  <div className="space-y-1">
                    {member.expertise.map((skill, skillIndex) => (
                      <span
                        key={skillIndex}
                        className="inline-block bg-bg-secondary text-xs px-2 py-1 rounded-full text-secondary mr-1 mb-1"
                      >
                        {skill}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Journey Timeline */}
          <div className="mb-20">
            <div className="text-center mb-12">
              <h2 className="text-4xl font-bold text-primary mb-4">Our Journey</h2>
              <p className="text-xl text-secondary max-w-3xl mx-auto">
                From concept to reality - the story of how DharmaMind came to life.
              </p>
            </div>
            <div className="max-w-5xl mx-auto">
              <div className="relative">
                {/* Timeline line */}
                <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-border-light hidden md:block"></div>
                
                {milestones.map((milestone, index) => (
                  <div
                    key={index}
                    className="relative flex items-start mb-12 animate-slide-in-left"
                    style={{ animationDelay: `${index * 0.1}s` }}
                  >
                    <div className="hidden md:flex absolute left-6 w-6 h-6 bg-brand-accent rounded-full border-4 border-white shadow-sm items-center justify-center text-xs">
                      {milestone.icon}
                    </div>
                    <div className="content-card md:ml-16 w-full">
                      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-3">
                        <div className="flex items-center space-x-3 mb-2 sm:mb-0">
                          <span className="bg-brand-gradient text-white px-3 py-1 rounded-full text-sm font-semibold">
                            {milestone.year} {milestone.quarter}
                          </span>
                          <h3 className="text-lg font-semibold text-primary">{milestone.title}</h3>
                        </div>
                        <span className="text-2xl md:hidden">{milestone.icon}</span>
                      </div>
                      <p className="text-secondary">{milestone.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Technology & Approach */}
          <div className="mb-20">
            <div className="content-card">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
                <div className="text-center lg:text-left animate-slide-in-left">
                  <div className="text-8xl mb-6">🤖</div>
                  <h2 className="text-4xl font-bold text-primary mb-6">Our Technology</h2>
                  <p className="text-secondary text-lg leading-relaxed">
                    Cutting-edge AI meets ancient wisdom through our proprietary technology stack.
                  </p>
                </div>
                <div className="animate-slide-in-right">
                  <p className="text-secondary text-lg leading-relaxed mb-6">
                    Our AI is trained on authentic Sanskrit texts, scholarly translations, and commentaries from 
                    renowned spiritual teachers. We use advanced natural language processing to understand context 
                    and provide relevant, personalized guidance.
                  </p>
                  <div className="space-y-4">
                    <div className="flex items-start space-x-3">
                      <span className="text-brand-accent text-xl">✓</span>
                      <span className="text-secondary">Trained on classical texts: Bhagavad Gita, Upanishads, Vedas</span>
                    </div>
                    <div className="flex items-start space-x-3">
                      <span className="text-brand-accent text-xl">✓</span>
                      <span className="text-secondary">Reviewed by Sanskrit scholars and spiritual teachers</span>
                    </div>
                    <div className="flex items-start space-x-3">
                      <span className="text-brand-accent text-xl">✓</span>
                      <span className="text-secondary">Continuous learning from user interactions</span>
                    </div>
                    <div className="flex items-start space-x-3">
                      <span className="text-brand-accent text-xl">✓</span>
                      <span className="text-secondary">Privacy-first architecture with end-to-end encryption</span>
                    </div>
                    <div className="flex items-start space-x-3">
                      <span className="text-brand-accent text-xl">✓</span>
                      <span className="text-secondary">Multi-language support with cultural sensitivity</span>
                    </div>
                  </div>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                </div>
              </div>
            </div>
          </div>
<<<<<<< HEAD
        </div>
      </section>

      {/* What Makes Us Different */}
      <section className="py-20 px-6 bg-neutral-50">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-3xl font-semibold text-neutral-900 mb-12 text-center">What Makes DharmaMind Different</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="text-4xl font-semibold text-gold-600 mb-2">10,000+</div>
              <p className="text-neutral-600">Sacred texts and commentaries in our training data</p>
            </div>
            <div className="text-center">
              <div className="text-4xl font-semibold text-gold-600 mb-2">5,000+</div>
              <p className="text-neutral-600">Years of wisdom tradition preserved and accessible</p>
            </div>
            <div className="text-center">
              <div className="text-4xl font-semibold text-gold-600 mb-2">100%</div>
              <p className="text-neutral-600">Privacy-first—your spiritual journey stays yours</p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-6 bg-neutral-900">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-semibold text-white mb-4">Begin your journey</h2>
          <p className="text-lg text-neutral-400 mb-8 max-w-2xl mx-auto">
            Join thousands who have discovered a new way to access ancient wisdom.
            Start your conversation with DharmaMind today.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              href="https://dharmamind.ai"
              className="px-8 py-3 bg-gold-600 text-white font-medium rounded-lg hover:bg-gold-700 transition-colors"
            >
              Try DharmaMind Free
            </Link>
            <Link
              href="/contact"
              className="px-8 py-3 bg-transparent text-white font-medium rounded-lg border border-neutral-600 hover:border-neutral-500 transition-colors"
            >
              Contact Us
            </Link>
          </div>
        </div>
      </section>
    </Layout>
=======

          {/* Call to Action */}
          <div className="text-center animate-fade-in">
            <div className="content-card-featured">
              <h2 className="text-4xl font-bold text-primary mb-6">Join Our Mission</h2>
              <p className="text-xl text-secondary mb-8 max-w-3xl mx-auto leading-relaxed">
                Whether you're beginning your spiritual journey or seeking deeper understanding, 
                DharmaMind is here to guide you with wisdom from the ages.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button
                  variant="primary"
                  size="lg"
                  onClick={() => router.push('/')}
                >
                  Start Your Journey
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  onClick={() => router.push('/contact')}
                >
                  Contact Us
                </Button>
              </div>
            </div>
          </div>
        </div>

        <Footer />
      </div>
    </>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
  );
};

export default AboutPage;
