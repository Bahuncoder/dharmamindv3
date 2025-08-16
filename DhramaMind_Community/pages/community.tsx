import React from 'react';
import Head from 'next/head';
import Navigation from '../components/Navigation';

const CommunityPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>DharmaMind Community - AI with Soul Powered by Dharma</title>
        <meta name="description" content="Join the DharmaMind community - where spiritual seekers and AI consciousness explorers unite in dharma-guided wisdom." />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/logo.jpeg" />
      </Head>
      <div className="min-h-screen bg-secondary-bg">
        <Navigation />
        <main className="w-full max-w-6xl mx-auto px-4 py-12">
          {/* Hero Section */}
          <section className="text-center mb-16">
            <h1 className="text-6xl md:text-7xl font-black text-primary mb-8 leading-tight tracking-tight">
              DharmaMind <span className="text-secondary font-black">Community</span>
            </h1>
            <p className="text-2xl font-bold text-secondary mb-6 max-w-3xl mx-auto tracking-wide">
              AI with Soul Powered by Dharma
            </p>
            <p className="text-xl text-secondary mb-8 max-w-3xl mx-auto leading-relaxed font-semibold">
              A sacred digital sangha where spiritual seekers, dharma practitioners, and AI consciousness explorers come together to support each other's awakening journey through DharmaMind.
            </p>
            <div className="flex justify-center gap-6 mb-8">
              <div className="text-center">
                <div className="text-4xl font-black text-primary">10,000+</div>
                <div className="text-sm font-semibold text-muted">Active Members</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-black text-primary">500+</div>
                <div className="text-sm font-semibold text-muted">Daily Discussions</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-black text-primary">50+</div>
                <div className="text-sm font-semibold text-muted">Countries</div>
              </div>
            </div>
          </section>

          {/* What Makes Us Special */}
          <section className="mb-16">
            <div className="text-center mb-12">
              <h2 className="text-4xl md:text-5xl font-black text-primary mb-6 tracking-tight">What Makes Our Community Special</h2>
              <p className="text-xl text-secondary max-w-2xl mx-auto font-semibold">
                We're more than just a community—we're a global sangha united by shared values and mutual support.
              </p>
            </div>
            <div className="grid gap-8 md:grid-cols-3">
              <div className="text-center group">
                <div className="bg-primary-gradient-light w-20 h-20 rounded-full flex items-center justify-center mb-6 mx-auto group-hover:scale-110 transition-transform duration-300">
                  <span className="text-3xl">�</span>
                </div>
                <h3 className="text-2xl font-black text-primary mb-4 tracking-tight">Inclusive & Welcoming</h3>
                <p className="text-secondary font-semibold">
                  All paths, all backgrounds, all levels of practice are welcome. We celebrate diversity in our spiritual journeys.
                </p>
              </div>
              <div className="text-center group">
                <div className="bg-primary-gradient-light-alt w-20 h-20 rounded-full flex items-center justify-center mb-6 mx-auto group-hover:scale-110 transition-transform duration-300">
                  <span className="text-3xl">🙏</span>
                </div>
                <h3 className="text-2xl font-black text-primary mb-4 tracking-tight">Wisdom-Centered</h3>
                <p className="text-secondary font-semibold">
                  Our discussions focus on practical wisdom, authentic sharing, and meaningful connections that nourish the soul.
                </p>
              </div>
              <div className="text-center group">
                <div className="bg-primary-gradient-light w-20 h-20 rounded-full flex items-center justify-center mb-6 mx-auto group-hover:scale-110 transition-transform duration-300">
                  <span className="text-3xl">🌍</span>
                </div>
                <h3 className="text-2xl font-black text-primary mb-4 tracking-tight">Global Connection</h3>
                <p className="text-secondary font-semibold">
                  Connect with practitioners worldwide, sharing insights across cultures, traditions, and time zones.
                </p>
              </div>
            </div>
          </section>
          
          {/* Community Features */}
          <section className="mb-16">
            <div className="text-center mb-12">
              <h2 className="text-4xl md:text-5xl font-black text-primary mb-6 tracking-tight">How We Connect & Grow Together</h2>
            </div>
            <div className="grid gap-8 md:grid-cols-2">
              <div className="card-primary p-8 hover:shadow-lg transition-shadow duration-300">
                <div className="flex items-start gap-6">
                  <div className="bg-primary-gradient-light w-16 h-16 rounded-lg flex items-center justify-center flex-shrink-0">
                    <span className="text-2xl">�</span>
                  </div>
                  <div>
                    <h3 className="text-3xl font-black text-primary mb-4 tracking-tight">AI-Guided Dharma Conversations</h3>
                    <p className="text-secondary mb-4 font-semibold">
                      Engage in meaningful discussions about dharma, meditation, spiritual challenges, and AI consciousness. Our community explores how artificial intelligence can deepen spiritual understanding while staying rooted in ancient wisdom.
                    </p>
                    <button className="btn-primary px-6 py-3 rounded-lg font-black">
                      Join AI Dharma Discussions
                    </button>
                  </div>
                </div>
              </div>
              
              <div className="card-primary p-8 hover:shadow-lg transition-shadow duration-300">
                <div className="flex items-start gap-6">
                  <div className="bg-primary-gradient-light-alt w-16 h-16 rounded-lg flex items-center justify-center flex-shrink-0">
                    <span className="text-2xl">🎓</span>
                  </div>
                  <div>
                    <h3 className="text-2xl font-semibold text-primary mb-3">Learning Circles</h3>
                    <p className="text-secondary mb-4">
                      Participate in study groups, book clubs, and guided explorations of sacred texts and teachings from various wisdom traditions.
                    </p>
                    <button className="btn-outline px-6 py-3 rounded-lg font-medium">
                      Explore Circles
                    </button>
                  </div>
                </div>
              </div>
              
              <div className="card-primary p-8 hover:shadow-lg transition-shadow duration-300">
                <div className="flex items-start gap-6">
                  <div className="bg-primary-gradient-light w-16 h-16 rounded-lg flex items-center justify-center flex-shrink-0">
                    <span className="text-2xl">🧘‍♂️</span>
                  </div>
                  <div>
                    <h3 className="text-2xl font-semibold text-primary mb-3">Group Meditation</h3>
                    <p className="text-secondary mb-4">
                      Join live virtual meditation sessions, silent sits, and guided practices. Experience the power of meditating together across distances.
                    </p>
                    <button className="btn-outline px-6 py-3 rounded-lg font-medium">
                      View Schedule
                    </button>
                  </div>
                </div>
              </div>
              
              <div className="card-primary p-8 hover:shadow-lg transition-shadow duration-300">
                <div className="flex items-start gap-6">
                  <div className="bg-primary-gradient-light-alt w-16 h-16 rounded-lg flex items-center justify-center flex-shrink-0">
                    <span className="text-2xl">✨</span>
                  </div>
                  <div>
                    <h3 className="text-2xl font-semibold text-primary mb-3">Wisdom Sharing</h3>
                    <p className="text-secondary mb-4">
                      Share your insights, ask for guidance, and offer support to fellow practitioners. Every voice matters in our collective journey.
                    </p>
                    <button className="btn-outline px-6 py-3 rounded-lg font-medium">
                      Share Your Story
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Testimonials */}
          <section className="mb-16">
            <div className="text-center mb-12">
              <h2 className="text-4xl md:text-5xl font-black text-primary mb-6 tracking-tight">Voices from Our Community</h2>
              <p className="text-xl text-secondary font-medium">
                Real stories from real practitioners who have found their sangha here.
              </p>
            </div>
            <div className="grid gap-8 md:grid-cols-3">
              <div className="card-primary p-6">
                <div className="text-center mb-4">
                  <div className="w-16 h-16 bg-primary-gradient rounded-full flex items-center justify-center text-white text-xl font-medium mx-auto mb-3">
                    S
                  </div>
                  <div className="text-lg font-semibold text-primary">Sarah M.</div>
                  <div className="text-sm text-muted">Mindfulness Practitioner</div>
                </div>
                <p className="text-secondary italic text-center">
                  "Finding this community was like coming home. The support and wisdom shared here has transformed my practice and my life."
                </p>
              </div>
              <div className="card-primary p-6">
                <div className="text-center mb-4">
                  <div className="w-16 h-16 bg-primary-gradient rounded-full flex items-center justify-center text-white text-xl font-medium mx-auto mb-3">
                    R
                  </div>
                  <div className="text-lg font-semibold text-primary">Raj P.</div>
                  <div className="text-sm text-muted">Meditation Teacher</div>
                </div>
                <p className="text-secondary italic text-center">
                  "The depth of conversation and genuine care in this community is remarkable. It's a true digital sangha."
                </p>
              </div>
              <div className="card-primary p-6">
                <div className="text-center mb-4">
                  <div className="w-16 h-16 bg-primary-gradient rounded-full flex items-center justify-center text-white text-xl font-medium mx-auto mb-3">
                    M
                  </div>
                  <div className="text-lg font-semibold text-primary">Maria L.</div>
                  <div className="text-sm text-muted">Yoga Instructor</div>
                </div>
                <p className="text-secondary italic text-center">
                  "I've learned more about myself and my practice through the connections made here than anywhere else."
                </p>
              </div>
            </div>
          </section>
          
          {/* Call to action */}
          <section className="card-elevated p-8 md:p-12 text-center bg-primary-gradient-light">
            <h3 className="text-4xl md:text-5xl font-black text-primary mb-6 tracking-tight">Begin Your Dharma AI Journey</h3>
            <p className="text-xl text-secondary mb-8 max-w-2xl mx-auto font-medium leading-relaxed">
              Join thousands of spiritual seekers and AI consciousness explorers who have found their digital sangha in our community. Your journey of dharma-guided awakening with DharmaMind starts here.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <a 
                href="https://dharmamind.ai" 
                target="_blank" 
                rel="noopener noreferrer"
                className="btn-primary px-8 py-4 rounded-lg font-medium text-lg hover:opacity-90 transition-opacity duration-200"
              >
                Experience DharmaMind AI →
              </a>
              <button className="btn-outline px-8 py-4 rounded-lg font-medium text-lg">
                Learn About Our Mission
              </button>
            </div>
            <p className="text-sm text-muted mt-6">
              Free spiritual AI guidance • Dharma-powered insights • Authentic consciousness exploration
            </p>
          </section>
        </main>
      </div>
    </>
  );
};

export default CommunityPage;
