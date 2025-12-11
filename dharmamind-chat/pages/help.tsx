import React, { useState } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import Logo from '../components/Logo';
import { ContactButton, SupportSection } from '../components/CentralizedSupport';
import { siteConfig } from '../config/shared.config';

interface FAQItem {
  question: string;
  answer: string;
  category: 'getting-started' | 'account' | 'billing' | 'features' | 'technical' | 'api';
}

const HelpPage: React.FC = () => {
  const router = useRouter();
  const [activeCategory, setActiveCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedFAQ, setExpandedFAQ] = useState<number | null>(null);

  // Use FAQs from shared config - single source of truth
  const faqData: FAQItem[] = siteConfig.support.categories.flatMap(cat => {
    // Map support FAQ items that match this category
    const categoryFaqs = [
      // Getting Started
      ...(cat.id === 'getting-started' ? [
        {
          question: "How do I get started with DharmaMind?",
          answer: "Simply click 'Get Started' on our homepage to create a free account. You can also try our demo version without signing up to explore the platform's capabilities.",
          category: "getting-started" as const
        },
        {
          question: "What makes DharmaMind different from other AI assistants?",
          answer: "DharmaMind combines cutting-edge AI technology with ancient dharmic wisdom. Our 32-dimensional wisdom architecture ensures responses are not just intelligent, but ethically grounded and spiritually meaningful.",
          category: "getting-started" as const
        },
        {
          question: "Can I use DharmaMind for free?",
          answer: `Yes! We offer a free tier with ${siteConfig.pricing.plans[0].limits.conversationsPerMonth} conversations per month. You can upgrade to ${siteConfig.pricing.plans[1].name} or ${siteConfig.pricing.plans[2].name} plans for unlimited access and advanced features.`,
          category: "getting-started" as const
        }
      ] : []),
      // Account
      ...(cat.id === 'account' ? [
        {
          question: "How do I reset my password?",
          answer: "Click 'Forgot Password' on the login page, enter your email address, and follow the instructions sent to your email to reset your password.",
          category: "account" as const
        },
        {
          question: "Can I change my email address?",
          answer: "Yes, go to Settings > Account > Email to update your email address. You'll need to verify the new email before the change takes effect.",
          category: "account" as const
        },
        {
          question: "How do I delete my account?",
          answer: "Go to Settings > Privacy & Data > Delete Account. Please note that this action is permanent and cannot be undone.",
          category: "account" as const
        }
      ] : []),
      // Billing
      ...(cat.id === 'billing' ? [
        {
          question: "What payment methods do you accept?",
          answer: "We accept all major credit cards (Visa, MasterCard, American Express, Discover) through Stripe. Enterprise customers can also pay via invoice.",
          category: "billing" as const
        },
        {
          question: "Can I cancel my subscription anytime?",
          answer: `Yes, you can cancel your subscription at any time from your Settings page. Your access will continue until the end of your current billing period.${siteConfig.pricing.guarantees.cancelAnytime ? ' No questions asked.' : ''}`,
          category: "billing" as const
        },
        {
          question: "Do you offer refunds?",
          answer: `We offer a ${siteConfig.pricing.guarantees.moneyBack.days}-day money-back guarantee for all paid plans. Contact our support team if you're not satisfied with your experience.`,
          category: "billing" as const
        },
        {
          question: "What's the pricing?",
          answer: `We offer three plans: Free ($0/month with ${siteConfig.pricing.plans[0].limits.conversationsPerMonth} conversations), ${siteConfig.pricing.plans[1].name} ($${siteConfig.pricing.plans[1].price.monthly}/month for unlimited access), and ${siteConfig.pricing.plans[2].name} (custom pricing for organizations).`,
          category: "billing" as const
        }
      ] : []),
      // Features
      ...(cat.id === 'features' ? [
        {
          question: "What are the 32 wisdom modules?",
          answer: "Our 32-dimensional wisdom architecture includes dharmic concepts like Karma, Moksha, Viveka, Shakti, Ahimsa, and 27 others. Each conversation is processed through relevant philosophical frameworks.",
          category: "features" as const
        },
        {
          question: "How does the chat history work?",
          answer: `All your conversations are automatically saved. ${siteConfig.pricing.plans[1].name} users have unlimited history, while Free users can access the last ${siteConfig.pricing.plans[0].limits.historyDays} days. You can also export your chat history.`,
          category: "features" as const
        },
        {
          question: "Can I use DharmaMind offline?",
          answer: "DharmaMind requires an internet connection as it relies on cloud-based AI processing. However, your recent chat history is cached locally for quick access.",
          category: "features" as const
        }
      ] : []),
      // Technical
      ...(cat.id === 'technical' ? [
        {
          question: "Is my data secure?",
          answer: "Yes, we use enterprise-grade security including AES-256 encryption, SOC 2 compliance, and regular security audits. Your conversations are private and never shared.",
          category: "technical" as const
        },
        {
          question: "What browsers are supported?",
          answer: "DharmaMind works on all modern browsers including Chrome, Firefox, Safari, and Edge. We recommend using the latest version for the best experience.",
          category: "technical" as const
        },
        {
          question: "Is there a mobile app?",
          answer: "We're developing native iOS and Android apps (coming soon). Currently, our web app is fully responsive and works great on mobile browsers.",
          category: "technical" as const
        }
      ] : []),
      // API
      ...(cat.id === 'api' ? [
        {
          question: "How do I get API access?",
          answer: `API access is available for ${siteConfig.pricing.plans[2].name} customers. Visit our API documentation at ${siteConfig.support.apiDocs} to learn more.`,
          category: "api" as const
        },
        {
          question: "What are the API rate limits?",
          answer: "Rate limits vary by plan. Enterprise customers get custom limits based on their needs. See our API documentation for details.",
          category: "api" as const
        }
      ] : [])
    ];
    return categoryFaqs;
  });

  // Use categories from shared config
  const categories = [
    { id: 'all', name: 'All Topics', icon: '📚' },
    ...siteConfig.support.categories
  ];
  const filteredFAQs = faqData.filter(faq => {
    const matchesCategory = activeCategory === 'all' || faq.category === activeCategory;
    const matchesSearch = faq.question.toLowerCase().includes(searchQuery.toLowerCase()) ||
      faq.answer.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  const toggleFAQ = (index: number) => {
    setExpandedFAQ(expandedFAQ === index ? null : index);
  };

  return (
    <>
      <Head>
        <title>Help Center - DharmaMind</title>
        <meta name="description" content="Find answers to common questions about DharmaMind, our AI-powered spiritual guidance platform" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="border-b border-gray-200 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <button
                onClick={() => router.push('/')}
                className="hover:opacity-80 transition-opacity"
              >
                <Logo size="sm" showText={true} />
              </button>

              <nav className="flex items-center space-x-8">
                <button
                  onClick={() => router.push('/')}
                  className="text-gray-600 hover:text-gray-900 text-sm font-medium"
                >
                  Home
                </button>
                <ContactButton
                  variant="link"
                  prefillCategory="support"
                  className="text-gray-600 hover:text-gray-900 text-sm font-medium"
                >
                  Contact
                </ContactButton>
                <button
                  onClick={() => router.push('/auth?mode=login')}
                  className="bg-gradient-to-r from-amber-600 to-emerald-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:from-amber-700 hover:to-emerald-700 transition-all duration-300"
                >
                  Sign In
                </button>
              </nav>
            </div>
          </div>
        </header>

        {/* Hero Section */}
        <div className="bg-gradient-to-br from-amber-50 to-emerald-50 py-16">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <h1 className="text-4xl font-bold text-gray-900 mb-4">
              How can we help you? 🙏
            </h1>
            <p className="text-xl text-gray-600 mb-8">
              Find answers to common questions about DharmaMind
            </p>

            {/* Search Bar */}
            <div className="max-w-2xl mx-auto relative">
              <input
                type="text"
                placeholder="Search for help articles..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full px-6 py-4 text-lg border border-gray-300 rounded-xl focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 shadow-sm"
              />
              <div className="absolute right-4 top-1/2 transform -translate-y-1/2">
                <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">

            {/* Category Sidebar */}
            <div className="lg:col-span-1">
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 sticky top-8">
                <h3 className="font-semibold text-gray-900 mb-4">Categories</h3>
                <div className="space-y-2">
                  {categories.map((category) => (
                    <button
                      key={category.id}
                      onClick={() => setActiveCategory(category.id)}
                      className={`w-full text-left px-3 py-2 rounded-lg transition-colors ${activeCategory === category.id
                          ? 'bg-emerald-100 text-emerald-800 font-medium'
                          : 'text-gray-600 hover:bg-gray-100'
                        }`}
                    >
                      <span className="mr-2">{category.icon}</span>
                      {category.name}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* FAQ Content */}
            <div className="lg:col-span-3">
              {filteredFAQs.length === 0 ? (
                <div className="text-center py-12">
                  <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
                    <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 12h6m-6-4h6m2 5.291A7.962 7.962 0 0112 20.4a7.962 7.962 0 01-8-7.8v-.6C4 6.477 7.477 3 12 3s8 3.477 8 8v.6c0 2.196-.896 4.182-2.344 5.591z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">No results found</h3>
                  <p className="text-gray-500">Try adjusting your search or browse different categories.</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {filteredFAQs.map((faq, index) => (
                    <div key={index} className="bg-white rounded-lg shadow-sm border border-gray-200">
                      <button
                        onClick={() => toggleFAQ(index)}
                        className="w-full text-left px-6 py-4 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-inset"
                      >
                        <div className="flex items-center justify-between">
                          <h3 className="text-lg font-medium text-gray-900">{faq.question}</h3>
                          <svg
                            className={`w-5 h-5 text-gray-500 transition-transform duration-200 ${expandedFAQ === index ? 'transform rotate-180' : ''
                              }`}
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                          </svg>
                        </div>
                      </button>

                      {expandedFAQ === index && (
                        <div className="px-6 pb-4">
                          <div className="border-t border-gray-200 pt-4">
                            <p className="text-gray-700 leading-relaxed">{faq.answer}</p>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* Contact Support - Use Centralized Support Section */}
              <SupportSection
                title="Still need help? 🤝"
                subtitle="Can't find what you're looking for? Our support team is here to help."
                showLinks={true}
                className="mt-12 bg-gradient-to-r from-amber-50 to-emerald-50 rounded-lg p-8 text-center"
              />
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="border-t border-gray-200 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <div className="text-center">
              <button
                onClick={() => router.push('/')}
                className="flex justify-center mx-auto mb-4 hover:opacity-80 transition-opacity"
              >
                <Logo size="sm" showText={true} />
              </button>
              <p className="text-sm text-gray-600">
                © 2025 DharmaMind. All rights reserved.
              </p>
            </div>
          </div>
        </footer>
      </div>
    </>
  );
};

export default HelpPage;
