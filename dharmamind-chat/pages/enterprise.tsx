import React from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import Logo from '../components/Logo';
import { ContactButton } from '../components/CentralizedSupport';

const EnterprisePage: React.FC = () => {
  const router = useRouter();

  const enterpriseFeatures = [
    {
      icon: '🏢',
      title: 'Single Sign-On (SSO)',
      description: 'Integrate with your existing identity providers including Active Directory, LDAP, SAML, and OAuth.'
    },
    {
      icon: '🔐',
      title: 'Advanced Security',
      description: 'Enterprise-grade security with custom encryption keys, audit logs, and compliance reporting.'
    },
    {
      icon: '🛠️',
      title: 'Custom Deployment',
      description: 'On-premise, private cloud, or hybrid deployment options to meet your infrastructure requirements.'
    },
    {
      icon: '📊',
      title: 'Analytics & Reporting',
      description: 'Comprehensive usage analytics, conversation insights, and custom reporting dashboards.'
    },
    {
      icon: '🔧',
      title: 'API Integration',
      description: 'Full REST API access with custom rate limits and dedicated support for seamless integration.'
    },
    {
      icon: '👥',
      title: 'Team Management',
      description: 'Advanced user management, role-based permissions, and departmental organization tools.'
    },
    {
      icon: '🎯',
      title: 'Custom Training',
      description: 'Train DharmaMind on your organization-specific content, policies, and knowledge base.'
    },
    {
      icon: '📞',
      title: 'Dedicated Support',
      description: '24/7 dedicated support with assigned customer success manager and priority response.'
    }
  ];

  const pricingTiers = [
    {
      name: 'Enterprise Starter',
      price: '$2,500',
      period: '/month',
      description: 'Perfect for medium-sized organizations',
      features: [
        'Up to 500 users',
        'SSO integration',
        'Basic analytics',
        'Email support',
        'Standard SLA'
      ],
      cta: 'Start Trial'
    },
    {
      name: 'Enterprise Pro',
      price: '$7,500',
      period: '/month',
      description: 'Advanced features for large organizations',
      features: [
        'Up to 2,000 users',
        'Advanced SSO & LDAP',
        'Custom deployment options',
        'Advanced analytics',
        'Dedicated support',
        'Custom integrations'
      ],
      cta: 'Contact Sales',
      popular: true
    },
    {
      name: 'Enterprise Custom',
      price: 'Custom',
      period: 'pricing',
      description: 'Tailored solutions for enterprise needs',
      features: [
        'Unlimited users',
        'On-premise deployment',
        'Custom AI training',
        'White-label options',
        'Service level agreements',
        '24/7 dedicated support'
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

      <div className="min-h-screen bg-primary-background">
        {/* Header */}
        <header className="bg-white/80 backdrop-blur-sm border-b border-stone-200/50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              {/* Logo */}
              <Logo 
                size="sm"
                onClick={() => router.push('/')}
              />

              <nav className="flex items-center space-x-8">
                <button 
                  onClick={() => router.push('/')}
                  className="text-stone-600 hover:text-stone-900 text-sm font-medium"
                >
                  Home
                </button>
                <button 
                  onClick={() => router.push('/docs')}
                  className="text-stone-600 hover:text-stone-900 text-sm font-medium"
                >
                  Documentation
                </button>
                <ContactButton 
                  variant="link"
                  prefillCategory="partnership"
                  className="text-stone-600 hover:text-stone-900 text-sm font-medium"
                >
                  Contact
                </ContactButton>
                <button 
                  onClick={() => router.push('/auth?mode=login')}
                  className="bg-primary-gradient text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-primary-gradient transition-all duration-300"
                >
                  Sign In
                </button>
              </nav>
            </div>
          </div>
        </header>

        {/* Hero Section */}
        <div className="bg-primary-background-light py-20">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center">
              <h1 className="text-5xl font-bold text-stone-900 mb-6">
                DharmaMind for Business
              </h1>
              <p className="text-xl text-stone-600 mb-8 max-w-4xl mx-auto">
                Empowering organizations to build purpose-driven, ethically-minded, and resilient workforces through AI-powered wisdom
              </p>
              
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <button
                  onClick={() => router.push('/demo-request')}
                  className="btn-primary px-8 py-4 rounded-lg text-lg font-medium transition-all duration-300 shadow-lg"
                >
                  Schedule Enterprise Demo
                </button>
                <button
                  onClick={() => router.push('/technical-specs')}
                  className="btn-contact px-8 py-4 rounded-lg text-lg font-medium transition-all duration-300"
                >
                  Download Technical Specs
                </button>
              </div>
              
              <div className="mt-8 flex flex-col sm:flex-row items-center justify-center space-y-2 sm:space-y-0 sm:space-x-8 text-sm text-stone-600">
                <div className="flex items-center space-x-2">
                  <span className="text-green-500">✓</span>
                  <span>SOC 2 Type II Certified</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-green-500">✓</span>
                  <span>GDPR & HIPAA Compliant</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-green-500">✓</span>
                  <span>99.9% Uptime SLA</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Vision Statement */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-stone-900 mb-6">
              Our Enterprise Vision
            </h2>
            <p className="text-lg text-stone-600 max-w-4xl mx-auto leading-relaxed">
              At DharmaMind, our vision extends beyond the individual. We believe that a world of purpose-driven 
              individuals will create a world of purpose-driven organizations. Our enterprise plan translates 
              the principles of timeless wisdom into tangible, business-oriented outcomes, fostering a culture 
              of clarity, purpose, and peace in the modern workplace.
            </p>
          </div>
        </div>

        {/* Three Core Products */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-stone-900 mb-4">
              Our Enterprise Solutions
            </h2>
            <p className="text-xl text-stone-600 max-w-3xl mx-auto">
              Three powerful AI-driven tools designed to transform your organization from the inside out
            </p>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            
            {/* DharmaMind for Teams */}
            <div className="bg-white rounded-xl shadow-lg border border-stone-200 p-8 hover:shadow-xl transition-shadow">
              <div className="w-16 h-16 mb-6 bg-primary-gradient-light rounded-lg flex items-center justify-center">
                <span className="text-3xl">👥</span>
              </div>
              <h3 className="text-2xl font-bold text-stone-900 mb-4">DharmaMind for Teams</h3>
              <p className="text-stone-600 mb-6 font-medium">Cultivating Well-being and Performance</p>
              <p className="text-stone-600 mb-6">
                A private, corporate version of our core AI, acting as a confidential and wise assistant for every employee. 
                Designed to be a proactive tool for mental well-being and professional development.
              </p>
              
              <h4 className="font-semibold text-stone-900 mb-3">Key Features:</h4>
              <ul className="text-sm text-stone-600 space-y-2 mb-6">
                <li>✓ AI-Guided Mindfulness: Personalized exercises for focus and stress reduction</li>
                <li>✓ Conversational Coaching: Navigate difficult situations with clarity</li>
                <li>✓ Professional Development: Value-based goal-setting and skill enhancement</li>
                <li>✓ Private Reflection Space: Confidential guidance for work-life balance</li>
              </ul>
              
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
                <p className="text-sm text-amber-800">
                  <strong>Impact:</strong> Increased employee satisfaction, reduced burnout, and a culture of support
                </p>
              </div>
            </div>

            {/* DharmaMind Karma */}
            <div className="bg-white rounded-xl shadow-lg border border-stone-200 p-8 hover:shadow-xl transition-shadow">
              <div className="w-16 h-16 mb-6 bg-primary-gradient-light-alt rounded-lg flex items-center justify-center">
                <span className="text-3xl">⚖️</span>
              </div>
              <h3 className="text-2xl font-bold text-stone-900 mb-4">DharmaMind Karma</h3>
              <p className="text-stone-600 mb-6 font-medium">The Ethical Decision Framework</p>
              <p className="text-stone-600 mb-6">
                Our answer to the growing need for ethical leadership. Provides a powerful framework for leaders 
                and teams to integrate ethical principles into daily decision-making processes.
              </p>
              
              <h4 className="font-semibold text-stone-900 mb-3">Key Features:</h4>
              <ul className="text-sm text-stone-600 space-y-2 mb-6">
                <li>✓ Ethical Analysis Engine: Evaluate moral and social implications of decisions</li>
                <li>✓ Integrity Training Modules: Conscious leadership and ethical governance</li>
                <li>✓ Values Alignment Tools: Define, measure, and uphold core ethical values</li>
                <li>✓ Stakeholder Impact Analysis: Consider all parties in complex decisions</li>
              </ul>
              
              <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-4">
                <p className="text-sm text-emerald-800">
                  <strong>Impact:</strong> Build lasting trust with customers, attract top talent, transparent culture
                </p>
              </div>
            </div>

            {/* DharmaMind Pathways */}
            <div className="bg-white rounded-xl shadow-lg border border-stone-200 p-8 hover:shadow-xl transition-shadow">
              <div className="w-16 h-16 mb-6 bg-primary-gradient-light rounded-lg flex items-center justify-center">
                <span className="text-3xl">🧭</span>
              </div>
              <h3 className="text-2xl font-bold text-stone-900 mb-4">DharmaMind Pathways</h3>
              <p className="text-stone-600 mb-6 font-medium">Aligning Purpose with Profession</p>
              <p className="text-stone-600 mb-6">
                A revolutionary career guidance tool that helps employees align their professional roles with 
                their personal purpose, solving the greatest challenge in the modern workplace: employee engagement.
              </p>
              
              <h4 className="font-semibold text-stone-900 mb-3">Key Features:</h4>
              <ul className="text-sm text-stone-600 space-y-2 mb-6">
                <li>✓ Purpose-Driven Career Mapping: Discover professional Dharma with AI guidance</li>
                <li>✓ Managerial Coaching Tools: Dashboard for supporting team growth</li>
                <li>✓ Fulfillment Metrics: Measure impact of purpose on engagement</li>
                <li>✓ Personalized Journey: Identify unique skills, passions, and purpose</li>
              </ul>
              
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <p className="text-sm text-blue-800">
                  <strong>Impact:</strong> Reduced turnover, increased retention, highly motivated workforce
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Deployment Options */}
        <div className="bg-white border-y border-stone-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
            <div className="text-center mb-16">
              <h2 className="text-3xl font-bold text-stone-900 mb-4">
                Flexible Deployment Options
              </h2>
              <p className="text-xl text-stone-600">
                Deploy DharmaMind where it works best for your organization
              </p>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              
              {/* Cloud */}
              <div className="text-center p-8 border border-stone-200 rounded-xl">
                <div className="w-16 h-16 mx-auto mb-6 bg-primary-gradient-light rounded-full flex items-center justify-center">
                  <span className="text-3xl">☁️</span>
                </div>
                <h3 className="text-xl font-semibold text-stone-900 mb-4">Cloud Hosted</h3>
                <p className="text-stone-600 mb-6">
                  Fully managed cloud deployment with automatic updates, scaling, and 99.9% uptime SLA.
                </p>
                <ul className="text-sm text-stone-600 space-y-2 mb-6">
                  <li>✓ No infrastructure management</li>
                  <li>✓ Automatic scaling</li>
                  <li>✓ Global CDN</li>
                  <li>✓ 24/7 monitoring</li>
                </ul>
                <button className="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 transition-colors">
                  Learn More
                </button>
              </div>

              {/* Private Cloud */}
              <div className="text-center p-8 border-2 border-emerald-200 rounded-xl bg-emerald-50 relative">
                <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                  <span className="bg-emerald-600 text-white px-3 py-1 rounded-full text-sm font-medium">
                    Most Popular
                  </span>
                </div>
                <div className="w-16 h-16 mx-auto mb-6 bg-primary-gradient-light rounded-full flex items-center justify-center">
                  <span className="text-3xl">🏛️</span>
                </div>
                <h3 className="text-xl font-semibold text-stone-900 mb-4">Private Cloud</h3>
                <p className="text-stone-600 mb-6">
                  Dedicated cloud environment with enhanced security and compliance controls.
                </p>
                <ul className="text-sm text-stone-600 space-y-2 mb-6">
                  <li>✓ Dedicated infrastructure</li>
                  <li>✓ Enhanced security</li>
                  <li>✓ Custom compliance</li>
                  <li>✓ SLA guarantees</li>
                </ul>
                <button className="w-full bg-emerald-600 text-white py-3 rounded-lg hover:bg-emerald-700 transition-colors" onClick={() => router.push('/demo-request')}>
                  Schedule Demo
                </button>
              </div>

              {/* On-Premise */}
              <div className="text-center p-8 border border-stone-200 rounded-xl">
                <div className="w-16 h-16 mx-auto mb-6 bg-primary-gradient-light-alt rounded-full flex items-center justify-center">
                  <span className="text-3xl">🏢</span>
                </div>
                <h3 className="text-xl font-semibold text-stone-900 mb-4">On-Premise</h3>
                <p className="text-stone-600 mb-6">
                  Complete control with on-premise deployment in your own data center or private cloud.
                </p>
                <ul className="text-sm text-stone-600 space-y-2 mb-6">
                  <li>✓ Complete data control</li>
                  <li>✓ Custom security policies</li>
                  <li>✓ Regulatory compliance</li>
                  <li>✓ Offline capability</li>
                </ul>
                <button className="btn-enterprise w-full py-3 rounded-lg transition-colors" onClick={() => router.push('/demo-request')}>
                  Schedule Enterprise Demo
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Value Proposition */}
        <div className="bg-primary-background-light-horizontal py-16">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-16">
              <h2 className="text-3xl font-bold text-stone-900 mb-6">
                Our Value Proposition
              </h2>
              <p className="text-xl text-stone-600 max-w-3xl mx-auto mb-12">
                Our B2B offerings are not just a benefit; they are a strategic investment in your organization's future.
              </p>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              
              {/* Resilient Workforce */}
              <div className="bg-white rounded-xl shadow-lg border border-stone-200 p-8 text-center">
                <div className="w-16 h-16 mx-auto mb-6 bg-primary-gradient-light rounded-full flex items-center justify-center">
                  <span className="text-3xl">💪</span>
                </div>
                <h3 className="text-2xl font-bold text-stone-900 mb-4">A Resilient Workforce</h3>
                <p className="text-stone-600">
                  Build a team equipped with the emotional and mental tools to navigate stress and change with 
                  confidence and clarity.
                </p>
              </div>

              {/* Highly Engaged Culture */}
              <div className="bg-white rounded-xl shadow-lg border border-stone-200 p-8 text-center">
                <div className="w-16 h-16 mx-auto mb-6 bg-primary-gradient-light-alt rounded-full flex items-center justify-center">
                  <span className="text-3xl">🎯</span>
                </div>
                <h3 className="text-2xl font-bold text-stone-900 mb-4">A Highly Engaged Culture</h3>
                <p className="text-stone-600">
                  Foster a workforce that is motivated by purpose and aligned with shared values, leading to 
                  exceptional performance and job satisfaction.
                </p>
              </div>

              {/* Trusted Brand */}
              <div className="bg-white rounded-xl shadow-lg border border-stone-200 p-8 text-center">
                <div className="w-16 h-16 mx-auto mb-6 bg-primary-gradient-light rounded-full flex items-center justify-center">
                  <span className="text-3xl">🌟</span>
                </div>
                <h3 className="text-2xl font-bold text-stone-900 mb-4">A Trusted Brand</h3>
                <p className="text-stone-600">
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
            <h2 className="text-3xl font-bold text-stone-900 mb-4">
              Enterprise Pricing
            </h2>
            <p className="text-xl text-stone-600">
              Transparent pricing that scales with your organization
            </p>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {pricingTiers.map((tier, index) => (
              <div key={index} className={`relative rounded-xl p-8 ${
                tier.popular 
                  ? 'border-2 border-emerald-200 bg-emerald-50' 
                  : 'border border-stone-200 bg-white'
              }`}>
                {tier.popular && (
                  <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                    <span className="bg-emerald-600 text-white px-4 py-1 rounded-full text-sm font-medium">
                      Most Popular
                    </span>
                  </div>
                )}
                
                <div className="text-center">
                  <h3 className="text-xl font-semibold text-stone-900 mb-2">{tier.name}</h3>
                  <div className="mb-4">
                    <span className="text-4xl font-bold text-stone-900">{tier.price}</span>
                    <span className="text-stone-600">{tier.period}</span>
                  </div>
                  <p className="text-stone-600 mb-6">{tier.description}</p>
                </div>
                
                <ul className="space-y-3 mb-8">
                  {tier.features.map((feature, featureIndex) => (
                    <li key={featureIndex} className="flex items-center space-x-3">
                      <span className="text-green-500 flex-shrink-0">✓</span>
                      <span className="text-stone-700">{feature}</span>
                    </li>
                  ))}
                </ul>
                
                <ContactButton 
                  variant="button"
                  prefillCategory="partnership"
                  className={`w-full py-3 px-6 rounded-lg font-medium transition-all duration-300 ${
                    tier.popular
                      ? 'bg-emerald-600 text-white hover:bg-emerald-700'
                      : 'bg-primary-gradient hover:bg-primary-gradient text-white'
                  }`}
                >
                  {tier.cta}
                </ContactButton>
              </div>
            ))}
          </div>
        </div>

        {/* Customer Success */}
        <div className="bg-primary-gradient text-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
            <div className="text-center mb-16">
              <h2 className="text-3xl font-bold mb-4">
                Trusted by Leading Organizations
              </h2>
              <p className="text-xl text-stone-300">
                Join thousands of organizations using DharmaMind for ethical AI guidance
              </p>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
              
              {/* Testimonial 1 */}
              <div className="bg-testimonial rounded-lg p-8">
                <div className="flex items-center mb-6">
                  <div className="w-12 h-12 bg-primary-gradient rounded-full flex items-center justify-center mr-4">
                    <span className="text-white font-bold">TG</span>
                  </div>
                  <div>
                    <div className="font-semibold">Tech Global Inc.</div>
                    <div className="text-testimonial-title text-sm">Fortune 500 Technology Company</div>
                  </div>
                </div>
                <blockquote className="text-testimonial-quote mb-4">
                  "DharmaMind for Teams has transformed our workplace culture. Employee well-being is at an all-time high, 
                  and our teams are making more ethical, purpose-driven decisions. The ROI in employee satisfaction is remarkable."
                </blockquote>
                <div className="text-sm text-testimonial-author">
                  — Sarah Chen, Chief People Officer
                </div>
              </div>

              {/* Testimonial 2 */}
              <div className="bg-testimonial rounded-lg p-8">
                <div className="flex items-center mb-6">
                  <div className="w-12 h-12 bg-primary-gradient rounded-full flex items-center justify-center mr-4">
                    <span className="text-white font-bold">HC</span>
                  </div>
                  <div>
                    <div className="font-semibold">HealthCare United</div>
                    <div className="text-testimonial-title text-sm">Healthcare Organization</div>
                  </div>
                </div>
                <blockquote className="text-testimonial-quote mb-4">
                  "DharmaMind Pathways helped us reduce turnover by 40%. Our employees are now aligned with their purpose, 
                  and productivity has never been higher. It's a game-changer for employee engagement."
                </blockquote>
                <div className="text-sm text-testimonial-author">
                  — Dr. Michael Rodriguez, Chief Human Resources Officer
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* CTA Section */}
        <div className="bg-primary-background-light-horizontal">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16 text-center">
            <h2 className="text-3xl font-bold text-stone-900 mb-4">
              Ready to Transform Your Organization? 🚀
            </h2>
            <p className="text-xl text-stone-600 mb-8">
              Join leading organizations using DharmaMind for ethical AI decision-making
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button
                onClick={() => router.push('/demo-request')}
                className="bg-primary-gradient text-white px-8 py-4 rounded-lg text-lg font-medium hover:bg-primary-gradient transition-all duration-300 shadow-lg"
              >
                Schedule Demo
              </button>
              <button
                onClick={() => router.push('/technical-specs')}
                className="btn-contact-alt px-8 py-4 rounded-lg text-lg font-medium transition-all duration-300"
              >
                Download Technical Specs
              </button>
            </div>
            
            <div className="mt-8 text-sm text-stone-500">
              <p>📞 enterprise@dharmamind.com • +1 (555) 123-4567</p>
              <p className="mt-2">Available for demo: Monday-Friday, 9 AM - 6 PM PST</p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="border-t border-stone-200 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <div className="text-center">
              <Logo 
                size="xs"
                className="justify-center mb-4"
              />
              <p className="text-sm text-stone-600">
                © 2025 DharmaMind. All rights reserved.
              </p>
            </div>
          </div>
        </footer>
      </div>
    </>
  );
};

export default EnterprisePage;
