import React, { useState } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import { useNavigation } from '../hooks/useNavigation';
import Button from '../components/Button';
import Logo from '../components/Logo';
import { ContactForm } from '../components/CentralizedSupport';
<<<<<<< HEAD
import BrandHeader from '../components/BrandHeader';
=======
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc

const HelpAndSupportPage: React.FC = () => {
  const router = useRouter();
  const navigation = useNavigation();
  const [activeTab, setActiveTab] = useState<'help' | 'contact'>('help');

  const faqs = [
    {
      question: "What is DharmaMind?",
      answer: "DharmaMind is an AI-powered platform that provides spiritual guidance, wisdom, and support based on ancient Hindu philosophy and teachings. We combine traditional dharmic knowledge with modern AI to offer personalized spiritual insights."
    },
    {
      question: "How does the AI guidance work?",
      answer: "Our AI is trained on classical Hindu texts, philosophical teachings, and spiritual wisdom. It provides personalized guidance while maintaining respect for traditional teachings and adapting them to modern contexts."
    },
    {
      question: "What subscription plans are available?",
      answer: "We offer Free, Premium, and Enterprise plans. Free includes basic guidance, Premium offers unlimited sessions and advanced features, while Enterprise provides custom solutions for organizations and institutions."
    },
    {
      question: "Is my data secure and private?",
      answer: "Yes, we take privacy seriously. All conversations are encrypted, and we never share personal data with third parties. Your spiritual journey remains private and secure."
    },
    {
      question: "Can I use DharmaMind on mobile devices?",
      answer: "Yes, DharmaMind is fully responsive and works seamlessly on desktop, tablet, and mobile devices. You can access your spiritual guidance anywhere, anytime."
    },
    {
      question: "How do I cancel my subscription?",
      answer: "You can cancel your subscription anytime from your account settings. Go to Settings > Subscription > Cancel Subscription. Your access will continue until the end of your billing period."
    },
    {
      question: "Do you offer refunds?",
      answer: "We offer a 30-day money-back guarantee for all premium subscriptions. If you're not satisfied within the first 30 days, contact our support team for a full refund."
    },
    {
      question: "Can I get guidance in languages other than English?",
      answer: "Currently, DharmaMind primarily operates in English, but we're working on adding support for Sanskrit, Hindi, and other regional languages. This feature will be available in future updates."
    },
    {
      question: "How accurate is the spiritual guidance?",
      answer: "Our AI is trained on authentic sources and reviewed by spiritual scholars. However, remember that AI guidance should complement, not replace, human wisdom and your own intuition in your spiritual journey."
    },
    {
      question: "Is there a limit to how many questions I can ask?",
      answer: "Free users get 10 questions per month. Premium users have unlimited questions. Enterprise users get custom limits based on their plan."
    }
  ];

  const supportOptions = [
    {
      icon: "📚",
      title: "Knowledge Base",
      description: "Browse our comprehensive guides and tutorials",
      action: () => setActiveTab('help')
    },
    {
      icon: "💬",
      title: "Live Chat",
      description: "Chat with our support team (Available 9 AM - 6 PM PST)",
      action: () => setActiveTab('contact')
    },
    {
      icon: "📧",
      title: "Email Support",
      description: "Send us a detailed message and we'll respond within 24 hours",
      action: () => setActiveTab('contact')
    },
    {
      icon: "📞",
      title: "Phone Support",
      description: "Call us at +1 (555) 123-4567 for urgent matters",
      action: () => window.open('tel:+15551234567')
    }
  ];

  return (
    <>
      <Head>
        <title>Help & Support - DharmaMind</title>
        <meta name="description" content="Get help and support for DharmaMind. Find answers to common questions and contact our support team." />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-primary-background">
<<<<<<< HEAD
        {/* Professional Brand Header with Breadcrumbs */}
        <BrandHeader
          breadcrumbs={[
            { label: 'Home', href: '/' },
            { label: 'Help', href: '/help' }
          ]}
        />
=======
        {/* Header */}
        <header className="bg-white border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center space-x-4">
                <button
                  onClick={() => router.back()}
                  className="flex items-center space-x-2 text-secondary hover:text-primary transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                  <span className="font-medium">Back</span>
                </button>
                
                <div className="h-6 w-px bg-gray-300"></div>
                
                <Logo 
                  size="sm"
                  onClick={() => router.push('/')}
                  className="cursor-pointer"
                />
              </div>
            </div>
          </div>
        </header>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          {/* Hero Section */}
          <div className="text-center mb-16">
<<<<<<< HEAD
            <h1 className="text-4xl font-bold text-neutral-900 mb-6">
              Help & Support Center 🙏
            </h1>
            <p className="text-xl text-neutral-600 max-w-3xl mx-auto">
              Find answers to your questions or get in touch with our support team.
=======
            <h1 className="text-4xl font-bold text-primary mb-6">
              Help & Support Center 🙏
            </h1>
            <p className="text-xl text-secondary max-w-3xl mx-auto">
              Find answers to your questions or get in touch with our support team. 
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
              We're here to help you on your spiritual journey.
            </p>
          </div>

          {/* Tab Navigation */}
          <div className="flex justify-center mb-12">
<<<<<<< HEAD
            <div className="bg-neutral-200 rounded-lg p-1 flex">
              <button
                onClick={() => setActiveTab('help')}
                className={`px-6 py-2 rounded-md font-medium transition-colors ${activeTab === 'help'
                  ? 'bg-neutral-100 text-neutral-900 shadow-sm'
                  : 'text-neutral-600 hover:text-gold-600'
                  }`}
=======
            <div className="bg-gray-100 rounded-lg p-1 flex">
              <button
                onClick={() => setActiveTab('help')}
                className={`px-6 py-2 rounded-md font-medium transition-colors ${
                  activeTab === 'help'
                    ? 'bg-white text-primary shadow-sm'
                    : 'text-secondary hover:text-primary'
                }`}
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
              >
                📚 Help Center
              </button>
              <button
                onClick={() => setActiveTab('contact')}
<<<<<<< HEAD
                className={`px-6 py-2 rounded-md font-medium transition-colors ${activeTab === 'contact'
                  ? 'bg-neutral-100 text-neutral-900 shadow-sm'
                  : 'text-neutral-600 hover:text-gold-600'
                  }`}
=======
                className={`px-6 py-2 rounded-md font-medium transition-colors ${
                  activeTab === 'contact'
                    ? 'bg-white text-primary shadow-sm'
                    : 'text-secondary hover:text-primary'
                }`}
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
              >
                💬 Contact Support
              </button>
            </div>
          </div>

          {/* Help Center Tab */}
          {activeTab === 'help' && (
            <div className="space-y-12">
              {/* Quick Support Options */}
              <div>
<<<<<<< HEAD
                <h2 className="text-2xl font-bold text-neutral-900 mb-8 text-center">
=======
                <h2 className="text-2xl font-bold text-primary mb-8 text-center">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                  How can we help you today?
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  {supportOptions.map((option, index) => (
                    <div
                      key={index}
                      onClick={option.action}
<<<<<<< HEAD
                      className="bg-neutral-100 border border-neutral-300 rounded-xl p-6 text-center hover:shadow-lg transition-shadow cursor-pointer"
                    >
                      <div className="text-3xl mb-4">{option.icon}</div>
                      <h3 className="text-lg font-semibold text-neutral-900 mb-2">
                        {option.title}
                      </h3>
                      <p className="text-neutral-600 text-sm">
=======
                      className="bg-white border border-gray-200 rounded-xl p-6 text-center hover:shadow-lg transition-shadow cursor-pointer"
                    >
                      <div className="text-3xl mb-4">{option.icon}</div>
                      <h3 className="text-lg font-semibold text-primary mb-2">
                        {option.title}
                      </h3>
                      <p className="text-secondary text-sm">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                        {option.description}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              {/* FAQ Section */}
              <div>
<<<<<<< HEAD
                <h2 className="text-2xl font-bold text-neutral-900 mb-8 text-center">
=======
                <h2 className="text-2xl font-bold text-primary mb-8 text-center">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                  Frequently Asked Questions
                </h2>
                <div className="max-w-4xl mx-auto space-y-4">
                  {faqs.map((faq, index) => (
                    <details
                      key={index}
<<<<<<< HEAD
                      className="bg-neutral-100 border border-neutral-300 rounded-lg"
                    >
                      <summary className="p-6 cursor-pointer hover:bg-neutral-100 transition-colors">
                        <span className="text-lg font-semibold text-neutral-900">
=======
                      className="bg-white border border-gray-200 rounded-lg"
                    >
                      <summary className="p-6 cursor-pointer hover:bg-gray-50 transition-colors">
                        <span className="text-lg font-semibold text-primary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                          {faq.question}
                        </span>
                      </summary>
                      <div className="px-6 pb-6">
<<<<<<< HEAD
                        <p className="text-neutral-600 leading-relaxed">
=======
                        <p className="text-secondary leading-relaxed">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                          {faq.answer}
                        </p>
                      </div>
                    </details>
                  ))}
                </div>
              </div>

              {/* Additional Resources */}
              <div className="bg-primary-background-light-horizontal rounded-xl p-8">
                <div className="text-center">
<<<<<<< HEAD
                  <h2 className="text-2xl font-bold text-neutral-900 mb-4">
                    Still need help? 🤝
                  </h2>
                  <p className="text-neutral-600 mb-6">
=======
                  <h2 className="text-2xl font-bold text-primary mb-4">
                    Still need help? 🤝
                  </h2>
                  <p className="text-secondary mb-6">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    Can't find what you're looking for? Our support team is ready to assist you.
                  </p>
                  <div className="flex flex-col sm:flex-row gap-4 justify-center">
                    <Button
                      variant="primary"
                      onClick={() => setActiveTab('contact')}
                    >
                      Contact Support
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => setActiveTab('contact')}
                    >
                      Start Live Chat
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Contact Support Tab */}
          {activeTab === 'contact' && (
            <div className="max-w-2xl mx-auto">
<<<<<<< HEAD
              <div className="bg-neutral-100 rounded-xl border border-neutral-300 p-8">
                <div className="text-center mb-8">
                  <h2 className="text-2xl font-bold text-neutral-900 mb-4">
                    Contact Our Support Team
                  </h2>
                  <p className="text-neutral-600">
=======
              <div className="bg-white rounded-xl border border-gray-200 p-8">
                <div className="text-center mb-8">
                  <h2 className="text-2xl font-bold text-primary mb-4">
                    Contact Our Support Team
                  </h2>
                  <p className="text-secondary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    Send us a message and we'll get back to you within 24 hours.
                  </p>
                </div>

<<<<<<< HEAD
                <ContactForm
=======
                <ContactForm 
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                  initialCategory="support"
                  onSubmit={async (data) => {
                    // Handle form submission
                    console.log('Support form submitted:', data);
                  }}
                />
              </div>

              {/* Alternative Contact Methods */}
              <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
<<<<<<< HEAD
                <div className="bg-neutral-100 rounded-lg border border-neutral-300 p-6 text-center">
                  <div className="text-2xl mb-3">📧</div>
                  <h3 className="font-semibold text-neutral-900 mb-2">Email Us</h3>
                  <p className="text-neutral-600 text-sm mb-3">
                    For detailed inquiries
                  </p>
                  <a
                    href="mailto:support@dharmamind.com"
                    className="text-neutral-900 hover:text-gold-600-dark font-medium"
=======
                <div className="bg-white rounded-lg border border-gray-200 p-6 text-center">
                  <div className="text-2xl mb-3">📧</div>
                  <h3 className="font-semibold text-primary mb-2">Email Us</h3>
                  <p className="text-secondary text-sm mb-3">
                    For detailed inquiries
                  </p>
                  <a 
                    href="mailto:support@dharmamind.com"
                    className="text-primary hover:text-primary-dark font-medium"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                  >
                    support@dharmamind.com
                  </a>
                </div>

<<<<<<< HEAD
                <div className="bg-neutral-100 rounded-lg border border-neutral-300 p-6 text-center">
                  <div className="text-2xl mb-3">📞</div>
                  <h3 className="font-semibold text-neutral-900 mb-2">Call Us</h3>
                  <p className="text-neutral-600 text-sm mb-3">
                    Mon-Fri, 9 AM - 6 PM PST
                  </p>
                  <a
                    href="tel:+15551234567"
                    className="text-neutral-900 hover:text-gold-600-dark font-medium"
=======
                <div className="bg-white rounded-lg border border-gray-200 p-6 text-center">
                  <div className="text-2xl mb-3">📞</div>
                  <h3 className="font-semibold text-primary mb-2">Call Us</h3>
                  <p className="text-secondary text-sm mb-3">
                    Mon-Fri, 9 AM - 6 PM PST
                  </p>
                  <a 
                    href="tel:+15551234567"
                    className="text-primary hover:text-primary-dark font-medium"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                  >
                    +1 (555) 123-4567
                  </a>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
<<<<<<< HEAD
        <footer className="bg-neutral-100 border-t border-neutral-300 mt-16">
=======
        <footer className="bg-white border-t border-gray-200 mt-16">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-8">
              {/* Product Column */}
              <div>
<<<<<<< HEAD
                <h3 className="text-sm font-semibold text-neutral-900 mb-4">Product</h3>
=======
                <h3 className="text-sm font-semibold text-primary mb-4">Product</h3>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                <ul className="space-y-3">
                  <li>
                    <button
                      onClick={() => router.push('/#features')}
<<<<<<< HEAD
                      className="text-sm text-neutral-600 hover:text-gold-600 transition-colors"
=======
                      className="text-sm text-secondary hover:text-primary transition-colors"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    >
                      Features
                    </button>
                  </li>
                  <li>
                    <button
                      onClick={() => router.push('/pricing')}
<<<<<<< HEAD
                      className="text-sm text-neutral-600 hover:text-gold-600 transition-colors"
=======
                      className="text-sm text-secondary hover:text-primary transition-colors"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    >
                      Pricing
                    </button>
                  </li>
                  <li>
                    <button
                      onClick={() => router.push('/enterprise')}
<<<<<<< HEAD
                      className="text-sm text-neutral-600 hover:text-gold-600 transition-colors"
=======
                      className="text-sm text-secondary hover:text-primary transition-colors"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    >
                      Enterprise
                    </button>
                  </li>
                  <li>
                    <button
                      onClick={() => router.push('/api-docs')}
<<<<<<< HEAD
                      className="text-sm text-neutral-600 hover:text-gold-600 transition-colors"
=======
                      className="text-sm text-secondary hover:text-primary transition-colors"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    >
                      API
                    </button>
                  </li>
                </ul>
              </div>

              {/* Support Column */}
              <div>
<<<<<<< HEAD
                <h3 className="text-sm font-semibold text-neutral-900 mb-4">Support</h3>
=======
                <h3 className="text-sm font-semibold text-primary mb-4">Support</h3>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                <ul className="space-y-3">
                  <li>
                    <button
                      onClick={() => setActiveTab('help')}
<<<<<<< HEAD
                      className="text-sm text-neutral-600 hover:text-gold-600 transition-colors"
=======
                      className="text-sm text-secondary hover:text-primary transition-colors"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    >
                      Help
                    </button>
                  </li>
                  <li>
                    <button
                      onClick={() => router.push('/contact')}
<<<<<<< HEAD
                      className="text-sm text-neutral-600 hover:text-gold-600 transition-colors"
=======
                      className="text-sm text-secondary hover:text-primary transition-colors"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    >
                      Contact
                    </button>
                  </li>
                  <li>
                    <button
                      onClick={() => router.push('/docs')}
<<<<<<< HEAD
                      className="text-sm text-neutral-600 hover:text-gold-600 transition-colors"
=======
                      className="text-sm text-secondary hover:text-primary transition-colors"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    >
                      Documentation
                    </button>
                  </li>
                  <li>
                    <button
                      onClick={() => router.push('/status')}
<<<<<<< HEAD
                      className="text-sm text-neutral-600 hover:text-gold-600 transition-colors"
=======
                      className="text-sm text-secondary hover:text-primary transition-colors"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    >
                      Status
                    </button>
                  </li>
                </ul>
              </div>

              {/* Company Column */}
              <div>
<<<<<<< HEAD
                <h3 className="text-sm font-semibold text-neutral-900 mb-4">Company</h3>
=======
                <h3 className="text-sm font-semibold text-primary mb-4">Company</h3>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                <ul className="space-y-3">
                  <li>
                    <button
                      onClick={() => router.push('/about')}
<<<<<<< HEAD
                      className="text-sm text-neutral-600 hover:text-gold-600 transition-colors"
=======
                      className="text-sm text-secondary hover:text-primary transition-colors"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    >
                      About
                    </button>
                  </li>
                  <li>
                    <button
                      onClick={() => router.push('/careers')}
<<<<<<< HEAD
                      className="text-sm text-neutral-600 hover:text-gold-600 transition-colors"
=======
                      className="text-sm text-secondary hover:text-primary transition-colors"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    >
                      Careers
                    </button>
                  </li>
                  <li>
                    <button
                      onClick={() => router.push('/news')}
<<<<<<< HEAD
                      className="text-sm text-neutral-600 hover:text-gold-600 transition-colors"
=======
                      className="text-sm text-secondary hover:text-primary transition-colors"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    >
                      News
                    </button>
                  </li>
                  <li>
                    <button
                      onClick={() => router.push('/blog')}
<<<<<<< HEAD
                      className="text-sm text-neutral-600 hover:text-gold-600 transition-colors"
=======
                      className="text-sm text-secondary hover:text-primary transition-colors"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    >
                      Blog
                    </button>
                  </li>
                </ul>
              </div>

              {/* Legal Column */}
              <div>
<<<<<<< HEAD
                <h3 className="text-sm font-semibold text-neutral-900 mb-4">Legal</h3>
=======
                <h3 className="text-sm font-semibold text-primary mb-4">Legal</h3>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                <ul className="space-y-3">
                  <li>
                    <button
                      onClick={() => router.push('/privacy')}
<<<<<<< HEAD
                      className="text-sm text-neutral-600 hover:text-gold-600 transition-colors"
=======
                      className="text-sm text-secondary hover:text-primary transition-colors"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    >
                      Privacy
                    </button>
                  </li>
                  <li>
                    <button
                      onClick={() => router.push('/terms')}
<<<<<<< HEAD
                      className="text-sm text-neutral-600 hover:text-gold-600 transition-colors"
=======
                      className="text-sm text-secondary hover:text-primary transition-colors"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    >
                      Terms
                    </button>
                  </li>
                  <li>
                    <button
                      onClick={() => router.push('/security')}
<<<<<<< HEAD
                      className="text-sm text-neutral-600 hover:text-gold-600 transition-colors"
=======
                      className="text-sm text-secondary hover:text-primary transition-colors"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    >
                      Security
                    </button>
                  </li>
                  <li>
                    <button
                      onClick={() => router.push('/cookies')}
<<<<<<< HEAD
                      className="text-sm text-neutral-600 hover:text-gold-600 transition-colors"
=======
                      className="text-sm text-secondary hover:text-primary transition-colors"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    >
                      Cookies
                    </button>
                  </li>
                </ul>
              </div>
            </div>

            {/* Bottom Section */}
<<<<<<< HEAD
            <div className="border-t border-neutral-300 pt-8">
              <div className="flex flex-col md:flex-row items-center justify-between">
                <div className="flex items-center space-x-4 mb-4 md:mb-0">
                  <Logo
=======
            <div className="border-t border-gray-200 pt-8">
              <div className="flex flex-col md:flex-row items-center justify-between">
                <div className="flex items-center space-x-4 mb-4 md:mb-0">
                  <Logo 
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    size="xs"
                    onClick={() => router.push('/')}
                    className="cursor-pointer"
                  />
<<<<<<< HEAD
                  <span className="text-sm text-neutral-600">
                    © 2025 DharmaMind. All rights reserved.
                  </span>
                </div>

=======
                  <span className="text-sm text-secondary">
                    © 2025 DharmaMind. All rights reserved.
                  </span>
                </div>
                
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                <div className="flex items-center space-x-6">
                  <a
                    href="https://twitter.com/dharmamindai"
                    target="_blank"
                    rel="noopener noreferrer"
<<<<<<< HEAD
                    className="text-neutral-600 hover:text-gold-600 transition-colors"
                  >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M23.953 4.57a10 10 0 01-2.825.775 4.958 4.958 0 002.163-2.723c-.951.555-2.005.959-3.127 1.184a4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.936 4.936 0 004.604 3.417 9.867 9.867 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.053 0 13.998-7.496 13.998-13.985 0-.21 0-.42-.015-.63A9.935 9.935 0 0024 4.59z" />
=======
                    className="text-gray-400 hover:text-secondary transition-colors"
                  >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M23.953 4.57a10 10 0 01-2.825.775 4.958 4.958 0 002.163-2.723c-.951.555-2.005.959-3.127 1.184a4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.936 4.936 0 004.604 3.417 9.867 9.867 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.053 0 13.998-7.496 13.998-13.985 0-.21 0-.42-.015-.63A9.935 9.935 0 0024 4.59z"/>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    </svg>
                  </a>
                  <a
                    href="https://linkedin.com/company/dharmamindai"
                    target="_blank"
                    rel="noopener noreferrer"
<<<<<<< HEAD
                    className="text-neutral-600 hover:text-gold-600 transition-colors"
                  >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
=======
                    className="text-gray-400 hover:text-secondary transition-colors"
                  >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    </svg>
                  </a>
                  <a
                    href="https://github.com/dharmamind"
                    target="_blank"
                    rel="noopener noreferrer"
<<<<<<< HEAD
                    className="text-neutral-600 hover:text-gold-600 transition-colors"
                  >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
=======
                    className="text-gray-400 hover:text-secondary transition-colors"
                  >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    </svg>
                  </a>
                </div>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </>
  );
};

export default HelpAndSupportPage;
