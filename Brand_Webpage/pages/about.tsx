import React from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import Button from '../components/Button';
import Logo from '../components/Logo';
import Footer from '../components/Footer';
import SEOHead from '../components/SEOHead';
import BrandHeader from '../components/BrandHeader';

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
        {/* Professional Brand Header with Breadcrumbs */}
        <BrandHeader
          breadcrumbs={[
            { label: 'Home', href: '/' },
            { label: 'About', href: '/about' }
          ]}
        />

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
                </div>
              </div>
            </div>
          </div>

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
  );
};

export default AboutPage;
