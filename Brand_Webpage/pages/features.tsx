import React from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import Button from '../components/Button';
import Logo from '../components/Logo';
import Footer from '../components/Footer';

const FeaturesPage: React.FC = () => {
  const router = useRouter();

  const features = [
    {
      icon: "🤖",
      title: "AI-Powered Spiritual Guidance",
      description: "Interact with an AI trained on ancient spiritual texts to receive personalized wisdom and guidance.",
      benefits: [
        "24/7 availability for spiritual consultation",
        "Personalized responses based on your spiritual journey",
        "Accurate interpretations from authentic texts",
        "Progressive learning that adapts to your growth"
      ],
      useCases: [
        "Daily spiritual guidance and reflection",
        "Understanding complex philosophical concepts",
        "Meditation and mindfulness practices",
        "Life decision making with spiritual wisdom"
      ]
    },
    {
      icon: "📚",
      title: "Ancient Text Library",
      description: "Access to a comprehensive library of sacred texts including Bhagavad Gita, Upanishads, and Vedas.",
      benefits: [
        "Original Sanskrit texts with translations",
        "Multiple scholarly interpretations",
        "Cross-referenced verses and concepts",
        "Audio narrations and chanting guides"
      ],
      useCases: [
        "Study and research of spiritual texts",
        "Finding relevant verses for specific situations",
        "Learning Sanskrit and pronunciation",
        "Daily reading and contemplation practices"
      ]
    },
    {
      icon: "🧘‍♀️",
      title: "Personalized Meditation Programs",
      description: "Customized meditation and mindfulness programs based on your spiritual goals and preferences.",
      benefits: [
        "Programs tailored to your experience level",
        "Progressive difficulty and complexity",
        "Various meditation techniques and styles",
        "Progress tracking and insights"
      ],
      useCases: [
        "Building consistent meditation practice",
        "Learning different meditation techniques",
        "Stress reduction and mental clarity",
        "Spiritual development and growth"
      ]
    },
    {
      icon: "🌟",
      title: "Daily Wisdom & Insights",
      description: "Receive daily spiritual insights, quotes, and teachings relevant to your current life situation.",
      benefits: [
        "Contextual wisdom for daily challenges",
        "Inspiring quotes and teachings",
        "Practical applications of ancient wisdom",
        "Reflection prompts and exercises"
      ],
      useCases: [
        "Morning inspiration and motivation",
        "Guidance during difficult times",
        "Spiritual growth and self-reflection",
        "Building wisdom-based daily habits"
      ]
    },
    {
      icon: "🕯️",
      title: "Ritual & Practice Guidance",
      description: "Learn and practice traditional spiritual rituals, ceremonies, and daily practices with proper guidance.",
      benefits: [
        "Step-by-step ritual instructions",
        "Cultural context and significance",
        "Adaptation for modern lifestyles",
        "Personal practice recommendations"
      ],
      useCases: [
        "Morning and evening spiritual routines",
        "Festival and ceremony participation",
        "Personal spiritual milestones",
        "Building sacred daily practices"
      ]
    },
    {
      icon: "🌸",
      title: "Mindful Living Integration",
      description: "Integrate spiritual principles into everyday activities for a more mindful and purposeful life.",
      benefits: [
        "Practical spirituality for daily life",
        "Mindful approaches to work and relationships",
        "Ethical decision-making frameworks",
        "Balanced lifestyle recommendations"
      ],
      useCases: [
        "Mindful eating and living practices",
        "Spiritual approach to work and career",
        "Improving relationships and communication",
        "Finding purpose and meaning in life"
      ]
    }
  ];

  const technologies = [
    {
      icon: "🧠",
      title: "Advanced Natural Language Processing",
      description: "Our AI understands context, emotion, and spiritual concepts to provide meaningful responses.",
      tech: "GPT-4 based architecture with spiritual domain specialization"
    },
    {
      icon: "🔐",
      title: "Privacy-First Architecture",
      description: "End-to-end encryption ensures your spiritual journey remains completely private.",
      tech: "Zero-knowledge encryption with local data processing"
    },
    {
      icon: "🌐",
      title: "Multi-Language Support",
      description: "Available in 25+ languages with culturally sensitive adaptations.",
      tech: "Advanced translation models with cultural context awareness"
    },
    {
      icon: "📱",
      title: "Cross-Platform Accessibility",
      description: "Access your spiritual guidance anywhere, anytime, on any device.",
      tech: "Progressive Web App with offline capabilities"
    }
  ];

  const testimonials = [
    {
      name: "Dr. Sarah Johnson",
      role: "Philosophy Professor",
      text: "DharmaMind has revolutionized how I study and teach ancient wisdom. The AI's understanding of complex philosophical concepts is remarkable.",
      rating: 5
    },
    {
      name: "Michael Chen",
      role: "Software Engineer",
      text: "As a busy professional, having 24/7 access to spiritual guidance has been transformative. The personalized meditations fit perfectly into my schedule.",
      rating: 5
    },
    {
      name: "Priya Sharma",
      role: "Yoga Instructor",
      text: "The authentic Sanskrit texts and pronunciation guides have enhanced my teaching. My students love the daily wisdom insights.",
      rating: 5
    }
  ];

  return (
    <>
      <Head>
        <title>Features - DharmaMind | AI-Powered Spiritual Guidance Platform</title>
        <meta name="description" content="Discover DharmaMind's comprehensive features: AI-powered spiritual guidance, ancient text library, personalized meditation programs, and mindful living integration." />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
        <meta property="og:title" content="Features - DharmaMind AI Spiritual Platform" />
        <meta property="og:description" content="Explore our AI-powered spiritual guidance features designed to bring ancient wisdom into modern life." />
        <meta property="og:type" content="website" />
      </Head>

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
                  onClick={() => router.push('/about')}
                  className="text-secondary hover:text-primary text-sm font-medium transition-colors"
                >
                  About
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
                Powerful Features for Your Spiritual Journey
              </h1>
              <div className="w-24 h-1 bg-brand-gradient mx-auto mb-8"></div>
              <p className="text-xl md:text-2xl text-secondary max-w-4xl mx-auto mb-12 leading-relaxed">
                Discover comprehensive tools and features designed to guide you on your spiritual path, 
                combining ancient wisdom with modern technology for a transformative experience.
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
                  onClick={() => window.open('https://demo.dharmamind.ai', '_blank')}
                >
                  Try Demo
                </Button>
              </div>
            </div>
          </div>

          {/* Core Features */}
          <div className="mb-20">
            <div className="text-center mb-16">
              <h2 className="text-4xl font-bold text-primary mb-4">Core Features</h2>
              <p className="text-xl text-secondary max-w-3xl mx-auto">
                Comprehensive tools designed to support every aspect of your spiritual growth and development.
              </p>
            </div>

            <div className="space-y-16">
              {features.map((feature, index) => (
                <div
                  key={index}
                  className={`grid grid-cols-1 lg:grid-cols-2 gap-12 items-center ${
                    index % 2 === 1 ? 'lg:grid-flow-col-dense' : ''
                  }`}
                >
                  <div className={`${index % 2 === 1 ? 'lg:col-start-2' : ''} animate-slide-in-left`}>
                    <div className="flex items-center space-x-4 mb-6">
                      <div className="text-5xl">{feature.icon}</div>
                      <h3 className="text-3xl font-bold text-primary">{feature.title}</h3>
                    </div>
                    <p className="text-lg text-secondary leading-relaxed mb-8">
                      {feature.description}
                    </p>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                      <div>
                        <h4 className="text-lg font-semibold text-primary mb-4">Key Benefits</h4>
                        <ul className="space-y-2">
                          {feature.benefits.map((benefit, benefitIndex) => (
                            <li key={benefitIndex} className="flex items-start space-x-2">
                              <span className="text-brand-accent mt-1">✓</span>
                              <span className="text-secondary text-sm">{benefit}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                      
                      <div>
                        <h4 className="text-lg font-semibold text-primary mb-4">Use Cases</h4>
                        <ul className="space-y-2">
                          {feature.useCases.map((useCase, useCaseIndex) => (
                            <li key={useCaseIndex} className="flex items-start space-x-2">
                              <span className="text-brand-accent mt-1">•</span>
                              <span className="text-secondary text-sm">{useCase}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                  
                  <div className={`${index % 2 === 1 ? 'lg:col-start-1' : ''} animate-slide-in-right`}>
                    <div className="content-card text-center bg-gradient-to-br from-brand-primary/5 to-brand-accent/10">
                      <div className="text-8xl mb-6">{feature.icon}</div>
                      <div className="bg-white/80 backdrop-blur rounded-lg p-6">
                        <h4 className="text-xl font-semibold text-primary mb-4">Feature Highlights</h4>
                        <div className="grid grid-cols-2 gap-4 text-center">
                          <div className="bg-brand-primary/10 rounded-lg p-3">
                            <div className="text-lg font-bold text-brand-accent">24/7</div>
                            <div className="text-xs text-secondary">Available</div>
                          </div>
                          <div className="bg-brand-primary/10 rounded-lg p-3">
                            <div className="text-lg font-bold text-brand-accent">AI</div>
                            <div className="text-xs text-secondary">Powered</div>
                          </div>
                          <div className="bg-brand-primary/10 rounded-lg p-3">
                            <div className="text-lg font-bold text-brand-accent">∞</div>
                            <div className="text-xs text-secondary">Personalized</div>
                          </div>
                          <div className="bg-brand-primary/10 rounded-lg p-3">
                            <div className="text-lg font-bold text-brand-accent">🔒</div>
                            <div className="text-xs text-secondary">Private</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Technology Stack */}
          <div className="mb-20">
            <div className="text-center mb-12">
              <h2 className="text-4xl font-bold text-primary mb-4">Built on Cutting-Edge Technology</h2>
              <p className="text-xl text-secondary max-w-3xl mx-auto">
                Our platform leverages the latest advances in AI and technology to deliver an exceptional spiritual experience.
              </p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {technologies.map((tech, index) => (
                <div
                  key={index}
                  className="content-card hover:shadow-lg transition-all duration-300 animate-fade-in-up"
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <div className="flex items-start space-x-4">
                    <div className="text-4xl">{tech.icon}</div>
                    <div className="flex-1">
                      <h3 className="text-xl font-semibold text-primary mb-3">{tech.title}</h3>
                      <p className="text-secondary leading-relaxed mb-4">{tech.description}</p>
                      <div className="bg-bg-secondary px-3 py-2 rounded-lg">
                        <span className="text-sm text-muted font-mono">{tech.tech}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* User Testimonials */}
          <div className="mb-20">
            <div className="text-center mb-12">
              <h2 className="text-4xl font-bold text-primary mb-4">What Our Users Say</h2>
              <p className="text-xl text-secondary max-w-3xl mx-auto">
                Real experiences from people who have transformed their spiritual journey with DharmaMind.
              </p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {testimonials.map((testimonial, index) => (
                <div
                  key={index}
                  className="content-card text-center hover:shadow-lg transition-all duration-300 animate-fade-in-up"
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <div className="flex justify-center mb-4">
                    {[...Array(testimonial.rating)].map((_, i) => (
                      <span key={i} className="text-yellow-400 text-xl">⭐</span>
                    ))}
                  </div>
                  <p className="text-secondary leading-relaxed mb-6 italic">"{testimonial.text}"</p>
                  <div>
                    <h4 className="text-lg font-semibold text-primary">{testimonial.name}</h4>
                    <p className="text-brand-accent text-sm">{testimonial.role}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Feature Comparison */}
          <div className="mb-20">
            <div className="content-card">
              <div className="text-center mb-12">
                <h2 className="text-4xl font-bold text-primary mb-4">Why Choose DharmaMind?</h2>
                <p className="text-xl text-secondary max-w-3xl mx-auto">
                  See how DharmaMind compares to traditional spiritual guidance methods.
                </p>
              </div>
              
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border-light">
                      <th className="text-left py-4 text-primary font-semibold">Feature</th>
                      <th className="text-center py-4 text-primary font-semibold">DharmaMind</th>
                      <th className="text-center py-4 text-secondary">Traditional Methods</th>
                    </tr>
                  </thead>
                  <tbody className="space-y-4">
                    <tr className="border-b border-border-light/50">
                      <td className="py-4 text-secondary">Availability</td>
                      <td className="py-4 text-center text-brand-accent font-semibold">24/7</td>
                      <td className="py-4 text-center text-muted">Limited Hours</td>
                    </tr>
                    <tr className="border-b border-border-light/50">
                      <td className="py-4 text-secondary">Personalization</td>
                      <td className="py-4 text-center text-brand-accent font-semibold">AI-Powered</td>
                      <td className="py-4 text-center text-muted">General Advice</td>
                    </tr>
                    <tr className="border-b border-border-light/50">
                      <td className="py-4 text-secondary">Cost</td>
                      <td className="py-4 text-center text-brand-accent font-semibold">Affordable</td>
                      <td className="py-4 text-center text-muted">Expensive</td>
                    </tr>
                    <tr className="border-b border-border-light/50">
                      <td className="py-4 text-secondary">Text Access</td>
                      <td className="py-4 text-center text-brand-accent font-semibold">Comprehensive Library</td>
                      <td className="py-4 text-center text-muted">Limited Resources</td>
                    </tr>
                    <tr className="border-b border-border-light/50">
                      <td className="py-4 text-secondary">Privacy</td>
                      <td className="py-4 text-center text-brand-accent font-semibold">Complete Anonymity</td>
                      <td className="py-4 text-center text-muted">Personal Disclosure</td>
                    </tr>
                    <tr>
                      <td className="py-4 text-secondary">Progress Tracking</td>
                      <td className="py-4 text-center text-brand-accent font-semibold">AI Analytics</td>
                      <td className="py-4 text-center text-muted">Manual Notes</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Call to Action */}
          <div className="text-center animate-fade-in">
            <div className="content-card-featured">
              <h2 className="text-4xl font-bold text-primary mb-6">Ready to Transform Your Spiritual Journey?</h2>
              <p className="text-xl text-secondary mb-8 max-w-3xl mx-auto leading-relaxed">
                Join thousands of seekers who have discovered the power of AI-enhanced spiritual guidance. 
                Start your journey today with our comprehensive feature set.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center mb-8">
                <Button
                  variant="primary"
                  size="lg"
                  onClick={() => router.push('/')}
                >
                  Get Started Free
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  onClick={() => router.push('/pricing')}
                >
                  View Pricing
                </Button>
              </div>
              <p className="text-sm text-muted">
                No credit card required • 7-day free trial • Cancel anytime
              </p>
            </div>
          </div>
        </div>

        <Footer />
      </div>
    </>
  );
};

export default FeaturesPage;
