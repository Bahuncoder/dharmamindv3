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
                </div>
              </div>
            </div>
          </div>
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
  );
};

export default AboutPage;
