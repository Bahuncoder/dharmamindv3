import React, { useEffect } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import { useSession } from 'next-auth/react';
import { motion, useScroll, useTransform, AnimatePresence } from 'framer-motion';
import { useAuth } from '../contexts/AuthContext';
import { useCentralizedSystem } from '../components/CentralizedSystem';
import Logo from '../components/Logo';
import Button from '../components/Button';
import Footer from '../components/Footer';
import { ContactButton } from '../components/CentralizedSupport';
import { useLoading } from '../hooks/useLoading';
import { LoadingPage } from '../components/CentralizedLoading';

const EnterprisePage: React.FC = () => {
  const router = useRouter();
  const { data: session, status } = useSession();
  const { user, isAuthenticated } = useAuth();
  const { isLoading, withLoading } = useLoading();
  const { goToAuth, toggleSubscriptionModal, toggleAuthModal } = useCentralizedSystem();

  useEffect(() => {
    // Check for subscription upgrade request in URL
    if (router.query.upgrade === 'true') {
      toggleSubscriptionModal(true);
    }
  }, [router.query]);

  const handleGetStarted = () => {
    goToAuth('signup');
  };

  const handleTryFree = async () => {
    // Navigate to dharmamind.ai for demo
    window.location.href = 'https://dharmamind.ai';
  };

  // Show loading while checking session
  if (status === 'loading') {
    return (
      <LoadingPage 
        message="Loading enterprise solutions..."
        submessage="Preparing your business intelligence platform"
        showLogo={true}
      />
    );
  }

  const enterpriseFeatures = [
    {
      icon: '🏢',
      title: 'Team Integration',
      description: 'Seamlessly integrate DharmaMind into your existing workflow and team structure.'
    },
    {
      icon: '🔐',
      title: 'Enterprise Security',
      description: 'Advanced security features including data encryption, secure access controls, and audit logs.'
    },
    {
      icon: '🛠️',
      title: 'Custom Deployment',
      description: 'Flexible deployment options including cloud, on-premise, or hybrid solutions.'
    },
    {
      icon: '📊',
      title: 'Usage Analytics',
      description: 'Comprehensive insights into team engagement and platform utilization patterns.'
    },
    {
      icon: '🔧',
      title: 'API Access',
      description: 'Full API integration capabilities for seamless connection with your existing tools.'
    },
    {
      icon: '👥',
      title: 'User Management',
      description: 'Advanced administration tools for managing users, permissions, and organizational structure.'
    },
    {
      icon: '🎯',
      title: 'Custom Configuration',
      description: 'Tailor DharmaMind to match your organization\'s specific needs and values.'
    },
    {
      icon: '📞',
      title: 'Priority Support',
      description: 'Dedicated support team with priority response times and personalized assistance.'
    }
  ];

  const pricingTiers = [
    {
      name: 'Professional',
      price: 'Custom',
      period: 'pricing',
      description: 'Tailored solutions for growing teams',
      features: [
        'Custom user capacity',
        'Team management tools',
        'Basic integrations',
        'Standard support',
        'Secure deployment'
      ],
      cta: 'Contact Sales'
    },
    {
      name: 'Enterprise',
      price: 'Custom',
      period: 'pricing',
      description: 'Advanced features for large organizations',
      features: [
        'Unlimited users',
        'Advanced SSO integration',
        'Custom deployment options',
        'Priority support',
        'Custom training',
        'Analytics dashboard'
      ],
      cta: 'Schedule Demo',
      popular: true
    },
    {
      name: 'Enterprise Plus',
      price: 'Custom',
      period: 'pricing',
      description: 'Premium solutions with white-label options',
      features: [
        'Full platform customization',
        'On-premise deployment',
        'Dedicated infrastructure',
        'White-label branding',
        '24/7 dedicated support',
        'SLA guarantees'
      ],
      cta: 'Contact Sales'
    }
  ];

  return (
    <>
      <Head>
        <title>Enterprise Solutions - DharmaMind</title>
        <meta name="description" content="Enterprise-grade AI spiritual guidance platform with SSO, custom deployment, and dedicated support" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-primary-background-main">
        {/* Header */}
        <header className="border-b border-light bg-white/80 backdrop-blur-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              {/* Logo */}
              <Logo 
                size="sm"
                onClick={() => router.push('/')}
              />

              {/* Navigation */}
              <nav className="hidden md:flex items-center space-x-8">
                <button 
                  onClick={() => router.push('/')}
                  className="text-secondary hover:text-primary text-sm font-medium"
                >
                  Home
                </button>
                <button 
                  onClick={() => router.push('/docs')}
                  className="text-secondary hover:text-primary text-sm font-medium"
                >
                  Documentation
                </button>
                <ContactButton 
                  variant="link"
                  prefillCategory="partnership"
                  className="text-secondary hover:text-primary text-sm font-medium"
                >
                  Contact
                </ContactButton>
              </nav>

              {/* Actions */}
              <div className="flex items-center space-x-4">
                {session || isAuthenticated ? (
                  <>
                    <div className="hidden sm:flex items-center space-x-3">
                      <div className="text-right">
                        <div className="text-sm font-medium text-primary">
                          {(session?.user?.email || user?.email)?.split('@')[0] || 'User'}
                        </div>
                        <div className="text-xs text-secondary">
                          {user?.subscription_plan || 'Basic Plan'}
                        </div>
                      </div>
                    </div>
                    <Button
                      onClick={handleTryFree}
                      variant="primary"
                      size="sm"
                    >
                      Open Chat
                    </Button>
                  </>
                ) : (
                  <>
                    <Button
                      onClick={() => toggleAuthModal(true)}
                      variant="outline"
                      size="sm"
                    >
                      Sign In
                    </Button>
                    <Button
                      onClick={handleTryFree}
                      variant="primary"
                      size="sm"
                    >
                      Try Demo
                    </Button>
                  </>
                )}
              </div>
            </div>
          </div>
        </header>

        {/* Enhanced Hero Section */}
        <div className="relative bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white overflow-hidden">
          {/* Background Patterns */}
          <div className="absolute inset-0 opacity-10">
            <div className="absolute top-0 left-0 w-full h-full bg-[radial-gradient(circle_at_50%_50%,rgba(255,255,255,0.1),transparent_50%)]"></div>
            <div className="absolute top-1/4 right-1/4 w-96 h-96 bg-brand-primary/20 rounded-full blur-3xl"></div>
            <div className="absolute bottom-1/4 left-1/4 w-96 h-96 bg-brand-accent/20 rounded-full blur-3xl"></div>
          </div>
          
          <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="text-center"
            >
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.6, delay: 0.2 }}
                className="inline-flex items-center px-6 py-3 bg-white/10 backdrop-blur-xl rounded-full mb-8 border border-white/20"
              >
                <span className="text-brand-accent mr-2">🏢</span>
                <span className="text-sm font-medium">Enterprise-Grade AI Wisdom Platform</span>
              </motion.div>
              
              <h1 className="text-6xl lg:text-7xl font-bold mb-6 bg-gradient-to-r from-white via-gray-100 to-white bg-clip-text text-transparent">
                DharmaMind for
                <br />
                <span className="bg-gradient-to-r from-brand-primary via-brand-accent to-brand-secondary bg-clip-text text-transparent">
                  Enterprise
                </span>
              </h1>
              
              <p className="text-xl lg:text-2xl text-gray-300 mb-8 max-w-4xl mx-auto leading-relaxed">
                Transform your organization with AI-powered ethical decision-making, employee well-being, 
                and purpose-driven leadership at enterprise scale.
              </p>
              
              <div className="flex flex-col lg:flex-row gap-6 justify-center mb-12">
                <motion.div
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <Button
                    onClick={() => router.push('/demo-request')}
                    className="bg-gradient-to-r from-brand-primary to-brand-accent hover:from-brand-primary/90 hover:to-brand-accent/90 text-white px-10 py-4 text-lg font-semibold rounded-xl shadow-2xl border-0"
                  >
                    <span className="mr-2">🚀</span>
                    Schedule Enterprise Demo
                  </Button>
                </motion.div>
                
                <motion.div
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <Button
                    onClick={() => router.push('/technical-specs')}
                    className="bg-white/10 backdrop-blur-xl hover:bg-white/20 text-white px-10 py-4 text-lg font-semibold rounded-xl border border-white/30"
                  >
                    <span className="mr-2">📋</span>
                    Technical Specifications
                  </Button>
                </motion.div>
              </div>
              
              {/* Trust Indicators */}
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.4 }}
                className="flex flex-wrap items-center justify-center gap-8 text-sm text-gray-400"
              >
                <div className="flex items-center space-x-2">
                  <span className="text-green-400">✓</span>
                  <span>SOC 2 Type II Compliant</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-green-400">✓</span>
                  <span>GDPR & CCPA Ready</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-green-400">✓</span>
                  <span>99.9% Uptime SLA</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-green-400">✓</span>
                  <span>Enterprise SSO</span>
                </div>
              </motion.div>
            </motion.div>
          </div>
        </div>

        {/* Enterprise Statistics */}
        <div className="bg-white py-16 border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-8 text-center">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6 }}
              >
                <div className="text-4xl font-bold text-gray-900 mb-2">500+</div>
                <div className="text-sm text-gray-600">Enterprise Clients</div>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: 0.1 }}
              >
                <div className="text-4xl font-bold text-gray-900 mb-2">2M+</div>
                <div className="text-sm text-gray-600">Users Supported</div>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: 0.2 }}
              >
                <div className="text-4xl font-bold text-gray-900 mb-2">47%</div>
                <div className="text-sm text-gray-600">Reduction in Turnover</div>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: 0.3 }}
              >
                <div className="text-4xl font-bold text-gray-900 mb-2">99.9%</div>
                <div className="text-sm text-gray-600">Platform Uptime</div>
              </motion.div>
            </div>
          </div>
        </div>

        {/* Vision Statement */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-primary mb-6">
              Our Enterprise Vision
            </h2>
            <p className="text-lg text-secondary max-w-4xl mx-auto leading-relaxed">
              At DharmaMind, our vision extends beyond the individual. We believe that a world of purpose-driven 
              individuals will create a world of purpose-driven organizations. Our enterprise plan translates 
              the principles of timeless wisdom into tangible, business-oriented outcomes, fostering a culture 
              of clarity, purpose, and peace in the modern workplace.
            </p>
          </div>
        </div>

        {/* Enhanced Three Core Products */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-6">
              Enterprise AI Solutions
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Three integrated platforms designed to transform organizational culture, 
              decision-making, and employee engagement at scale.
            </p>
          </motion.div>
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
            
            {/* DharmaMind for Teams - Enhanced */}
            <motion.div 
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
              className="relative group"
            >
              <div className="bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-200 rounded-2xl p-8 h-full hover:shadow-2xl transition-all duration-300 transform hover:scale-[1.02]">
                <div className="w-16 h-16 mb-6 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-2xl flex items-center justify-center">
                  <span className="text-3xl">👥</span>
                </div>
                <h3 className="text-2xl font-bold text-gray-900 mb-4">DharmaMind for Teams</h3>
                <p className="text-blue-700 mb-6 font-semibold text-lg">Employee Well-being & Performance Platform</p>
                <p className="text-gray-700 mb-6 leading-relaxed">
                  Enterprise-grade AI coaching platform providing personalized guidance for professional development, 
                  stress management, and workplace well-being. Includes advanced analytics and management dashboards.
                </p>
                
                <div className="space-y-4 mb-8">
                  <h4 className="font-semibold text-gray-900 mb-3">Enterprise Features:</h4>
                  <div className="space-y-3 text-sm text-gray-700">
                    <div className="flex items-start space-x-3">
                      <span className="text-green-500 mt-0.5">✓</span>
                      <div>
                        <strong>AI-Powered Wellness Hub:</strong> Personalized mindfulness, stress reduction, and productivity coaching
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <span className="text-green-500 mt-0.5">✓</span>
                      <div>
                        <strong>Manager Dashboard:</strong> Team wellness insights without compromising individual privacy
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <span className="text-green-500 mt-0.5">✓</span>
                      <div>
                        <strong>Integration Suite:</strong> Slack, Teams, Workday, and 50+ enterprise tools
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <span className="text-green-500 mt-0.5">✓</span>
                      <div>
                        <strong>Advanced Analytics:</strong> Employee engagement metrics and burnout prevention insights
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="bg-blue-100 border border-blue-200 rounded-xl p-4 mb-6">
                  <p className="text-sm text-blue-800">
                    <strong>ROI Impact:</strong> 47% reduction in employee turnover, 62% improvement in engagement scores
                  </p>
                </div>

                <div className="flex flex-wrap gap-2 mb-6">
                  <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-xs font-medium">SSO Integration</span>
                  <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-xs font-medium">HIPAA Compliant</span>
                  <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-xs font-medium">24/7 Support</span>
                </div>
              </div>
            </motion.div>

            {/* DharmaMind Karma - Enhanced */}
            <motion.div 
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="relative group"
            >
              <div className="bg-gradient-to-br from-purple-50 to-violet-50 border-2 border-purple-300 rounded-2xl p-8 h-full hover:shadow-2xl transition-all duration-300 transform hover:scale-[1.02] relative">
                <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                  <span className="bg-gradient-to-r from-purple-600 to-violet-600 text-white px-4 py-1 rounded-full text-sm font-semibold">
                    Most Popular
                  </span>
                </div>
                <div className="w-16 h-16 mb-6 bg-gradient-to-r from-purple-500 to-violet-600 rounded-2xl flex items-center justify-center">
                  <span className="text-3xl">⚖️</span>
                </div>
                <h3 className="text-2xl font-bold text-gray-900 mb-4">DharmaMind Karma</h3>
                <p className="text-purple-700 mb-6 font-semibold text-lg">Ethical Leadership & Decision Framework</p>
                <p className="text-gray-700 mb-6 leading-relaxed">
                  Advanced ethical AI system for corporate governance, helping leadership teams make values-aligned 
                  decisions with comprehensive stakeholder impact analysis and ethical compliance monitoring.
                </p>
                
                <div className="space-y-4 mb-8">
                  <h4 className="font-semibold text-gray-900 mb-3">Leadership Tools:</h4>
                  <div className="space-y-3 text-sm text-gray-700">
                    <div className="flex items-start space-x-3">
                      <span className="text-green-500 mt-0.5">✓</span>
                      <div>
                        <strong>Ethical Decision Engine:</strong> AI-powered analysis of moral and stakeholder implications
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <span className="text-green-500 mt-0.5">✓</span>
                      <div>
                        <strong>Governance Dashboard:</strong> Real-time compliance monitoring and risk assessment
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <span className="text-green-500 mt-0.5">✓</span>
                      <div>
                        <strong>Values Integration:</strong> Custom ethical frameworks aligned with company values
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <span className="text-green-500 mt-0.5">✓</span>
                      <div>
                        <strong>Audit Trail:</strong> Complete decision history with ethical reasoning documentation
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="bg-purple-100 border border-purple-200 rounded-xl p-4 mb-6">
                  <p className="text-sm text-purple-800">
                    <strong>Business Impact:</strong> 34% improvement in stakeholder trust, 28% reduction in legal risks
                  </p>
                </div>

                <div className="flex flex-wrap gap-2 mb-6">
                  <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-xs font-medium">C-Suite Ready</span>
                  <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-xs font-medium">Risk Management</span>
                  <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-xs font-medium">Compliance</span>
                </div>
              </div>
            </motion.div>

            {/* DharmaMind Pathways - Enhanced */}
            <motion.div 
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8, delay: 0.4 }}
              className="relative group"
            >
              <div className="bg-gradient-to-br from-emerald-50 to-teal-50 border border-emerald-200 rounded-2xl p-8 h-full hover:shadow-2xl transition-all duration-300 transform hover:scale-[1.02]">
                <div className="w-16 h-16 mb-6 bg-gradient-to-r from-emerald-500 to-teal-600 rounded-2xl flex items-center justify-center">
                  <span className="text-3xl">🧭</span>
                </div>
                <h3 className="text-2xl font-bold text-gray-900 mb-4">DharmaMind Pathways</h3>
                <p className="text-emerald-700 mb-6 font-semibold text-lg">Career Development & Purpose Alignment</p>
                <p className="text-gray-700 mb-6 leading-relaxed">
                  Revolutionary career guidance platform that aligns individual purpose with organizational goals, 
                  featuring AI-powered career mapping, succession planning, and talent development insights.
                </p>
                
                <div className="space-y-4 mb-8">
                  <h4 className="font-semibold text-gray-900 mb-3">Talent Solutions:</h4>
                  <div className="space-y-3 text-sm text-gray-700">
                    <div className="flex items-start space-x-3">
                      <span className="text-green-500 mt-0.5">✓</span>
                      <div>
                        <strong>Purpose-Driven Career Mapping:</strong> AI-guided professional development aligned with values
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <span className="text-green-500 mt-0.5">✓</span>
                      <div>
                        <strong>Succession Planning:</strong> Identify and develop future leaders based on purpose-fit
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <span className="text-green-500 mt-0.5">✓</span>
                      <div>
                        <strong>Retention Analytics:</strong> Predict and prevent talent flight with purpose alignment metrics
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <span className="text-green-500 mt-0.5">✓</span>
                      <div>
                        <strong>Skills Development:</strong> Personalized learning paths based on dharmic career goals
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="bg-emerald-100 border border-emerald-200 rounded-xl p-4 mb-6">
                  <p className="text-sm text-emerald-800">
                    <strong>Talent ROI:</strong> 58% increase in internal promotions, 41% improvement in retention rates
                  </p>
                </div>

                <div className="flex flex-wrap gap-2 mb-6">
                  <span className="px-3 py-1 bg-emerald-100 text-emerald-700 rounded-full text-xs font-medium">HR Integration</span>
                  <span className="px-3 py-1 bg-emerald-100 text-emerald-700 rounded-full text-xs font-medium">Performance Management</span>
                  <span className="px-3 py-1 bg-emerald-100 text-emerald-700 rounded-full text-xs font-medium">Leadership Development</span>
                </div>
              </div>
            </motion.div>
          </div>
        </div>

        {/* Enterprise Security & Compliance */}
        <div className="bg-gray-50 py-20">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
              className="text-center mb-16"
            >
              <h2 className="text-4xl font-bold text-gray-900 mb-6">
                Enterprise-Grade Security & Compliance
              </h2>
              <p className="text-xl text-gray-600 max-w-3xl mx-auto">
                Built from the ground up with enterprise security, privacy, and compliance as core requirements.
              </p>
            </motion.div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 mb-16">
              
              {/* Security Features */}
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.8 }}
                className="space-y-8"
              >
                <div className="flex items-start space-x-4">
                  <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center flex-shrink-0">
                    <span className="text-blue-600 text-xl">🔒</span>
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">End-to-End Encryption</h3>
                    <p className="text-gray-600">All data encrypted in transit and at rest using AES-256 encryption with customer-managed keys.</p>
                  </div>
                </div>

                <div className="flex items-start space-x-4">
                  <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center flex-shrink-0">
                    <span className="text-green-600 text-xl">🛡️</span>
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">Zero-Trust Architecture</h3>
                    <p className="text-gray-600">Multi-layered security with continuous verification and least-privilege access controls.</p>
                  </div>
                </div>

                <div className="flex items-start space-x-4">
                  <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center flex-shrink-0">
                    <span className="text-purple-600 text-xl">🔐</span>
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">Enterprise SSO</h3>
                    <p className="text-gray-600">SAML 2.0, OpenID Connect, and OAuth 2.0 integration with all major identity providers.</p>
                  </div>
                </div>

                <div className="flex items-start space-x-4">
                  <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center flex-shrink-0">
                    <span className="text-red-600 text-xl">🚨</span>
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">Advanced Threat Detection</h3>
                    <p className="text-gray-600">Real-time monitoring with AI-powered anomaly detection and automated response systems.</p>
                  </div>
                </div>
              </motion.div>

              {/* Compliance Features */}
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.8 }}
                className="space-y-8"
              >
                <div className="flex items-start space-x-4">
                  <div className="w-12 h-12 bg-indigo-100 rounded-lg flex items-center justify-center flex-shrink-0">
                    <span className="text-indigo-600 text-xl">📋</span>
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">SOC 2 Type II Certified</h3>
                    <p className="text-gray-600">Annual third-party audits ensuring the highest standards of security and availability.</p>
                  </div>
                </div>

                <div className="flex items-start space-x-4">
                  <div className="w-12 h-12 bg-yellow-100 rounded-lg flex items-center justify-center flex-shrink-0">
                    <span className="text-yellow-600 text-xl">🌍</span>
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">GDPR & CCPA Compliant</h3>
                    <p className="text-gray-600">Full compliance with global privacy regulations including data portability and right to deletion.</p>
                  </div>
                </div>

                <div className="flex items-start space-x-4">
                  <div className="w-12 h-12 bg-teal-100 rounded-lg flex items-center justify-center flex-shrink-0">
                    <span className="text-teal-600 text-xl">⚕️</span>
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">HIPAA Ready</h3>
                    <p className="text-gray-600">Healthcare-grade privacy controls with Business Associate Agreements available.</p>
                  </div>
                </div>

                <div className="flex items-start space-x-4">
                  <div className="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center flex-shrink-0">
                    <span className="text-orange-600 text-xl">📊</span>
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">Complete Audit Trails</h3>
                    <p className="text-gray-600">Comprehensive logging and monitoring with tamper-proof audit trails for compliance reporting.</p>
                  </div>
                </div>
              </motion.div>
            </div>

            {/* Compliance Badges */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
              className="flex flex-wrap justify-center gap-8 opacity-70"
            >
              <div className="flex items-center space-x-2 text-gray-600">
                <span className="text-2xl">🏆</span>
                <span className="font-medium">SOC 2 Type II</span>
              </div>
              <div className="flex items-center space-x-2 text-gray-600">
                <span className="text-2xl">🛡️</span>
                <span className="font-medium">ISO 27001</span>
              </div>
              <div className="flex items-center space-x-2 text-gray-600">
                <span className="text-2xl">🔒</span>
                <span className="font-medium">GDPR Compliant</span>
              </div>
              <div className="flex items-center space-x-2 text-gray-600">
                <span className="text-2xl">⚕️</span>
                <span className="font-medium">HIPAA Ready</span>
              </div>
              <div className="flex items-center space-x-2 text-gray-600">
                <span className="text-2xl">🇺🇸</span>
                <span className="font-medium">CCPA Compliant</span>
              </div>
            </motion.div>
          </div>
        </div>

        {/* Enhanced Deployment Options */}
        <div className="bg-white py-20">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
              className="text-center mb-16"
            >
              <h2 className="text-4xl font-bold text-gray-900 mb-6">
                Flexible Deployment Architecture
              </h2>
              <p className="text-xl text-gray-600 max-w-3xl mx-auto">
                Deploy DharmaMind in the environment that best meets your security, compliance, and operational requirements.
              </p>
            </motion.div>
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              
              {/* Cloud Deployment */}
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.8 }}
                className="bg-gradient-to-br from-blue-50 to-cyan-50 border border-blue-200 rounded-2xl p-8 text-center hover:shadow-xl transition-all duration-300"
              >
                <div className="w-20 h-20 mx-auto mb-6 bg-gradient-to-r from-blue-500 to-cyan-600 rounded-2xl flex items-center justify-center">
                  <span className="text-4xl">☁️</span>
                </div>
                <h3 className="text-2xl font-semibold text-gray-900 mb-4">Multi-Cloud SaaS</h3>
                <p className="text-gray-600 mb-6 leading-relaxed">
                  Fully managed cloud deployment across AWS, Azure, and GCP with automatic scaling, 
                  global CDN, and 99.9% uptime SLA guarantee.
                </p>
                <div className="space-y-3 text-sm text-gray-700 mb-8 text-left">
                  <div className="flex items-center space-x-2">
                    <span className="text-green-500">✓</span>
                    <span>Zero infrastructure management</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-green-500">✓</span>
                    <span>Automatic updates & patches</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-green-500">✓</span>
                    <span>Global content delivery</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-green-500">✓</span>
                    <span>24/7 monitoring & support</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-green-500">✓</span>
                    <span>Fastest time to deployment</span>
                  </div>
                </div>
                <div className="bg-blue-100 rounded-lg p-3 mb-6">
                  <p className="text-sm text-blue-800 font-medium">Best for: Growing teams, rapid deployment, global access</p>
                </div>
                <Button
                  onClick={() => router.push('/demo-request')}
                  className="w-full bg-gradient-to-r from-blue-500 to-cyan-600 text-white border-0"
                >
                  Start Cloud Trial
                </Button>
              </motion.div>

              {/* Private Cloud Deployment */}
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.8, delay: 0.1 }}
                className="bg-gradient-to-br from-purple-50 to-pink-50 border-2 border-purple-300 rounded-2xl p-8 text-center hover:shadow-xl transition-all duration-300 relative"
              >
                <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                  <span className="bg-gradient-to-r from-purple-600 to-pink-600 text-white px-4 py-1 rounded-full text-sm font-semibold">
                    Enterprise Choice
                  </span>
                </div>
                <div className="w-20 h-20 mx-auto mb-6 bg-gradient-to-r from-purple-500 to-pink-600 rounded-2xl flex items-center justify-center">
                  <span className="text-4xl">🏛️</span>
                </div>
                <h3 className="text-2xl font-semibold text-gray-900 mb-4">Private Cloud</h3>
                <p className="text-gray-600 mb-6 leading-relaxed">
                  Dedicated cloud environment with enhanced security controls, custom compliance configurations, 
                  and enterprise-grade SLA guarantees.
                </p>
                <div className="space-y-3 text-sm text-gray-700 mb-8 text-left">
                  <div className="flex items-center space-x-2">
                    <span className="text-green-500">✓</span>
                    <span>Dedicated infrastructure</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-green-500">✓</span>
                    <span>Enhanced security controls</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-green-500">✓</span>
                    <span>Custom compliance frameworks</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-green-500">✓</span>
                    <span>Priority support & SLA</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-green-500">✓</span>
                    <span>Custom integrations</span>
                  </div>
                </div>
                <div className="bg-purple-100 rounded-lg p-3 mb-6">
                  <p className="text-sm text-purple-800 font-medium">Best for: Large enterprises, regulated industries, high security</p>
                </div>
                <Button
                  onClick={() => router.push('/demo-request')}
                  className="w-full bg-gradient-to-r from-purple-500 to-pink-600 text-white border-0"
                >
                  Schedule Enterprise Demo
                </Button>
              </motion.div>

              {/* On-Premise Deployment */}
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.8, delay: 0.2 }}
                className="bg-gradient-to-br from-emerald-50 to-teal-50 border border-emerald-200 rounded-2xl p-8 text-center hover:shadow-xl transition-all duration-300"
              >
                <div className="w-20 h-20 mx-auto mb-6 bg-gradient-to-r from-emerald-500 to-teal-600 rounded-2xl flex items-center justify-center">
                  <span className="text-4xl">🏢</span>
                </div>
                <h3 className="text-2xl font-semibold text-gray-900 mb-4">On-Premise</h3>
                <p className="text-gray-600 mb-6 leading-relaxed">
                  Complete control with on-premise deployment in your data center. Perfect for organizations 
                  with strict data sovereignty and air-gapped requirements.
                </p>
                <div className="space-y-3 text-sm text-gray-700 mb-8 text-left">
                  <div className="flex items-center space-x-2">
                    <span className="text-green-500">✓</span>
                    <span>Complete data sovereignty</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-green-500">✓</span>
                    <span>Air-gapped deployment option</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-green-500">✓</span>
                    <span>Custom security policies</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-green-500">✓</span>
                    <span>Offline operation capability</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-green-500">✓</span>
                    <span>White-label customization</span>
                  </div>
                </div>
                <div className="bg-emerald-100 rounded-lg p-3 mb-6">
                  <p className="text-sm text-emerald-800 font-medium">Best for: Government, defense, financial services, healthcare</p>
                </div>
                <Button
                  onClick={() => router.push('/demo-request')}
                  className="w-full bg-gradient-to-r from-emerald-500 to-teal-600 text-white border-0"
                >
                  Contact Enterprise Sales
                </Button>
              </motion.div>
            </div>
          </div>
        </div>

        {/* Value Proposition */}
        <div className="bg-section-light py-16">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-16">
              <h2 className="text-3xl font-bold text-primary mb-6">
                Our Value Proposition
              </h2>
              <p className="text-xl text-secondary max-w-3xl mx-auto mb-12">
                Our B2B offerings are not just a benefit; they are a strategic investment in your organization's future.
              </p>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              
              {/* Resilient Workforce */}
              <div className="content-card text-center">
                <div className="w-16 h-16 mx-auto mb-6 bg-primary-clean rounded-full flex items-center justify-center">
                  <span className="text-3xl">💪</span>
                </div>
                <h3 className="text-2xl font-bold text-primary mb-4">A Resilient Workforce</h3>
                <p className="text-secondary">
                  Build a team equipped with the emotional and mental tools to navigate stress and change with 
                  confidence and clarity.
                </p>
              </div>

              {/* Highly Engaged Culture */}
              <div className="content-card text-center">
                <div className="w-16 h-16 mx-auto mb-6 bg-primary-clean rounded-full flex items-center justify-center">
                  <span className="text-3xl">🎯</span>
                </div>
                <h3 className="text-2xl font-bold text-primary mb-4">A Highly Engaged Culture</h3>
                <p className="text-secondary">
                  Foster a workforce that is motivated by purpose and aligned with shared values, leading to 
                  exceptional performance and job satisfaction.
                </p>
              </div>

              {/* Trusted Brand */}
              <div className="content-card text-center">
                <div className="w-16 h-16 mx-auto mb-6 bg-primary-clean rounded-full flex items-center justify-center">
                  <span className="text-3xl">🌟</span>
                </div>
                <h3 className="text-2xl font-bold text-primary mb-4">A Trusted Brand</h3>
                <p className="text-secondary">
                  Become an organization known for ethical leadership, integrity, and commitment to building 
                  a better future for all stakeholders.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Pricing */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-primary mb-4">
              Enterprise Pricing
            </h2>
            <p className="text-xl text-secondary">
              Transparent pricing that scales with your organization
            </p>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {pricingTiers.map((tier, index) => (
              <div key={index} className={`relative rounded-xl p-8 ${
                tier.popular 
                  ? 'pricing-card-featured' 
                  : 'content-card'
              }`}>
                {tier.popular && (
                  <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                    <span className="badge-popular">
                      Most Popular
                    </span>
                  </div>
                )}
                
                <div className="text-center">
                  <h3 className="text-xl font-semibold text-primary mb-2">{tier.name}</h3>
                  <div className="mb-4">
                    <span className="text-4xl font-bold text-primary">{tier.price}</span>
                    <span className="text-secondary">{tier.period}</span>
                  </div>
                  <p className="text-secondary mb-6">{tier.description}</p>
                </div>
                
                <ul className="space-y-3 mb-8">
                  {tier.features.map((feature, featureIndex) => (
                    <li key={featureIndex} className="flex items-center space-x-3">
                      <span className="text-success flex-shrink-0">✓</span>
                      <span className="text-primary">{feature}</span>
                    </li>
                  ))}
                </ul>
                
                <Button 
                  variant={tier.popular ? "primary" : "outline"}
                  className="w-full"
                  onClick={() => router.push('/demo-request')}
                >
                  {tier.cta}
                </Button>
              </div>
            ))}
          </div>
        </div>

        {/* Enterprise ROI Calculator */}
        <div className="bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white py-20">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
              className="text-center mb-16"
            >
              <h2 className="text-4xl font-bold mb-6">
                Calculate Your Enterprise ROI
              </h2>
              <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                See how DharmaMind can impact your organization's bottom line through improved employee engagement, 
                retention, and decision-making quality.
              </p>
            </motion.div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
              
              {/* ROI Stats */}
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.8 }}
                className="space-y-8"
              >
                <div className="bg-white/10 backdrop-blur-xl rounded-2xl p-6">
                  <div className="text-3xl font-bold text-green-400 mb-2">$2.4M</div>
                  <div className="text-lg font-semibold mb-2">Average Annual Savings</div>
                  <div className="text-gray-300 text-sm">For 1,000+ employee organizations</div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-white/10 backdrop-blur-xl rounded-xl p-4">
                    <div className="text-2xl font-bold text-blue-400 mb-1">47%</div>
                    <div className="text-sm text-gray-300">Turnover Reduction</div>
                  </div>
                  <div className="bg-white/10 backdrop-blur-xl rounded-xl p-4">
                    <div className="text-2xl font-bold text-purple-400 mb-1">62%</div>
                    <div className="text-sm text-gray-300">Engagement Increase</div>
                  </div>
                  <div className="bg-white/10 backdrop-blur-xl rounded-xl p-4">
                    <div className="text-2xl font-bold text-green-400 mb-1">34%</div>
                    <div className="text-sm text-gray-300">Better Decisions</div>
                  </div>
                  <div className="bg-white/10 backdrop-blur-xl rounded-xl p-4">
                    <div className="text-2xl font-bold text-yellow-400 mb-1">18M</div>
                    <div className="text-sm text-gray-300">Recovery Time</div>
                  </div>
                </div>

                <div className="bg-gradient-to-r from-brand-primary/20 to-brand-accent/20 backdrop-blur-xl rounded-2xl p-6 border border-white/20">
                  <h3 className="text-xl font-semibold mb-4">Typical ROI Breakdown</h3>
                  <div className="space-y-3 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-300">Reduced Hiring Costs</span>
                      <span className="text-white font-medium">$850K/year</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Improved Productivity</span>
                      <span className="text-white font-medium">$1.2M/year</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Better Decision Making</span>
                      <span className="text-white font-medium">$360K/year</span>
                    </div>
                    <div className="border-t border-white/20 pt-3 mt-3">
                      <div className="flex justify-between font-semibold">
                        <span>Total Annual Benefit</span>
                        <span className="text-green-400">$2.41M</span>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>

              {/* Contact Form */}
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.8 }}
                className="bg-white/10 backdrop-blur-xl rounded-2xl p-8 border border-white/20"
              >
                <h3 className="text-2xl font-bold mb-6">Get Your Custom ROI Report</h3>
                <form className="space-y-4">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <input
                      type="text"
                      placeholder="First Name"
                      className="bg-white/10 border border-white/30 rounded-lg px-4 py-3 text-white placeholder-gray-300 focus:outline-none focus:border-brand-accent"
                    />
                    <input
                      type="text"
                      placeholder="Last Name"
                      className="bg-white/10 border border-white/30 rounded-lg px-4 py-3 text-white placeholder-gray-300 focus:outline-none focus:border-brand-accent"
                    />
                  </div>
                  <input
                    type="email"
                    placeholder="Work Email"
                    className="w-full bg-white/10 border border-white/30 rounded-lg px-4 py-3 text-white placeholder-gray-300 focus:outline-none focus:border-brand-accent"
                  />
                  <input
                    type="text"
                    placeholder="Company Name"
                    className="w-full bg-white/10 border border-white/30 rounded-lg px-4 py-3 text-white placeholder-gray-300 focus:outline-none focus:border-brand-accent"
                  />
                  <select className="w-full bg-white/10 border border-white/30 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-brand-accent">
                    <option value="">Company Size</option>
                    <option value="100-500">100-500 employees</option>
                    <option value="500-1000">500-1,000 employees</option>
                    <option value="1000-5000">1,000-5,000 employees</option>
                    <option value="5000+">5,000+ employees</option>
                  </select>
                  <Button
                    type="submit"
                    className="w-full bg-gradient-to-r from-brand-primary to-brand-accent text-white py-3 font-semibold rounded-lg border-0"
                  >
                    Calculate My ROI
                  </Button>
                </form>
                <p className="text-xs text-gray-400 mt-4">
                  We'll send you a detailed ROI analysis within 24 hours.
                </p>
              </motion.div>
            </div>
          </div>
        </div>

        {/* Enhanced Customer Success Stories */}
        <div className="bg-white py-20">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
              className="text-center mb-16"
            >
              <h2 className="text-4xl font-bold text-gray-900 mb-6">
                Trusted by Industry Leaders
              </h2>
              <p className="text-xl text-gray-600 max-w-3xl mx-auto">
                Organizations across industries are transforming their cultures with DharmaMind's ethical AI platform.
              </p>
            </motion.div>
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-16">
              
              {/* Case Study 1 */}
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.8 }}
                className="bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-200 rounded-2xl p-8"
              >
                <div className="flex items-center mb-6">
                  <div className="w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center mr-4">
                    <span className="text-white text-xl">🏦</span>
                  </div>
                  <div>
                    <div className="font-semibold text-lg text-gray-900">Global Financial Services</div>
                    <div className="text-blue-600 text-sm">Fortune 500 • 45,000 employees</div>
                  </div>
                </div>
                <p className="text-gray-700 mb-6 leading-relaxed">
                  "DharmaMind Karma transformed our risk management culture. Leadership teams now make more thoughtful, 
                  stakeholder-conscious decisions. We've seen a 40% reduction in regulatory issues."
                </p>
                <div className="grid grid-cols-2 gap-4 text-center text-sm">
                  <div className="bg-white rounded-lg p-3">
                    <div className="text-2xl font-bold text-blue-600">40%</div>
                    <div className="text-gray-600">Fewer Compliance Issues</div>
                  </div>
                  <div className="bg-white rounded-lg p-3">
                    <div className="text-2xl font-bold text-green-600">$12M</div>
                    <div className="text-gray-600">Annual Savings</div>
                  </div>
                </div>
              </motion.div>

              {/* Case Study 2 */}
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.8, delay: 0.1 }}
                className="bg-gradient-to-br from-purple-50 to-pink-50 border border-purple-200 rounded-2xl p-8"
              >
                <div className="flex items-center mb-6">
                  <div className="w-12 h-12 bg-purple-600 rounded-full flex items-center justify-center mr-4">
                    <span className="text-white text-xl">🏥</span>
                  </div>
                  <div>
                    <div className="font-semibold text-lg text-gray-900">Healthcare Network</div>
                    <div className="text-purple-600 text-sm">Regional Leader • 15,000 employees</div>
                  </div>
                </div>
                <p className="text-gray-700 mb-6 leading-relaxed">
                  "DharmaMind for Teams became essential during the pandemic. Our healthcare workers found peace and 
                  resilience through AI-guided wellness. Burnout rates dropped dramatically."
                </p>
                <div className="grid grid-cols-2 gap-4 text-center text-sm">
                  <div className="bg-white rounded-lg p-3">
                    <div className="text-2xl font-bold text-purple-600">65%</div>
                    <div className="text-gray-600">Reduced Burnout</div>
                  </div>
                  <div className="bg-white rounded-lg p-3">
                    <div className="text-2xl font-bold text-green-600">89%</div>
                    <div className="text-gray-600">Staff Satisfaction</div>
                  </div>
                </div>
              </motion.div>

              {/* Case Study 3 */}
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.8, delay: 0.2 }}
                className="bg-gradient-to-br from-emerald-50 to-teal-50 border border-emerald-200 rounded-2xl p-8"
              >
                <div className="flex items-center mb-6">
                  <div className="w-12 h-12 bg-emerald-600 rounded-full flex items-center justify-center mr-4">
                    <span className="text-white text-xl">⚡</span>
                  </div>
                  <div>
                    <div className="font-semibold text-lg text-gray-900">Technology Unicorn</div>
                    <div className="text-emerald-600 text-sm">Startup • 8,000 employees</div>
                  </div>
                </div>
                <p className="text-gray-700 mb-6 leading-relaxed">
                  "DharmaMind Pathways revolutionized our talent development. Employees find deeper purpose in their work, 
                  leading to unprecedented retention and internal promotion rates."
                </p>
                <div className="grid grid-cols-2 gap-4 text-center text-sm">
                  <div className="bg-white rounded-lg p-3">
                    <div className="text-2xl font-bold text-emerald-600">72%</div>
                    <div className="text-gray-600">Internal Promotions</div>
                  </div>
                  <div className="bg-white rounded-lg p-3">
                    <div className="text-2xl font-bold text-green-600">91%</div>
                    <div className="text-gray-600">Employee Retention</div>
                  </div>
                </div>
              </motion.div>
            </div>

            {/* Industry Logos */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
              className="text-center"
            >
              <p className="text-gray-500 mb-8">Trusted across industries</p>
              <div className="flex flex-wrap justify-center items-center gap-12 opacity-60">
                <div className="text-2xl font-bold text-gray-400">🏦 Financial Services</div>
                <div className="text-2xl font-bold text-gray-400">🏥 Healthcare</div>
                <div className="text-2xl font-bold text-gray-400">⚡ Technology</div>
                <div className="text-2xl font-bold text-gray-400">🏭 Manufacturing</div>
                <div className="text-2xl font-bold text-gray-400">🎓 Education</div>
              </div>
            </motion.div>
          </div>
        </div>

        {/* CTA Section */}
        <div className="bg-section-light">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16 text-center">
            <h2 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-6">
                Start Your Enterprise Journey Today
              </h2>
              <p className="text-xl text-gray-600 mb-10 max-w-3xl mx-auto leading-relaxed">
                Join leading organizations using DharmaMind for ethical AI decision-making, 
                employee well-being, and purpose-driven leadership transformation.
              </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button
                variant="primary"
                onClick={() => router.push('/demo-request')}
                className="px-8 py-4 text-lg font-medium shadow-lg"
              >
                Schedule Demo
              </Button>
              <Button
                variant="outline"
                onClick={() => router.push('/technical-specs')}
                className="px-8 py-4 text-lg font-medium"
              >
                Download Technical Specs
              </Button>
            </div>
            
            <div className="mt-8 text-sm text-gray-500">
              <p>� contact@dharmamind.com • Schedule a personalized demo</p>
              <p className="mt-2">We're here to help you find the right solution for your organization</p>
            </div>
          </div>
        </div>

        {/* Centralized Footer */}
        <Footer variant="professional" />
      </div>
    </>
  );
};

export default EnterprisePage;
