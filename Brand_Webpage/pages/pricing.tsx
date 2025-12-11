<<<<<<< HEAD
import { Layout, Section, Button } from '../components';
import { siteConfig } from '../config/site.config';
import { useState } from 'react';
import Link from 'next/link';

export default function Pricing() {
  const [billingPeriod, setBillingPeriod] = useState<'monthly' | 'annual'>('monthly');

  const plans = [
    {
      name: 'Free',
      price: { monthly: '$0', annual: '$0' },
      period: 'forever',
      description: 'For individuals exploring AI-guided spiritual growth.',
      features: [
        '10 conversations per month',
        'Basic AI guidance',
        'Access to core wisdom traditions',
        'Community Discord access',
        'Email support',
      ],
      cta: 'Get Started Free',
      href: 'https://chat.dharmamind.ai',
      highlighted: false,
      badge: null,
    },
    {
      name: 'Pro',
      price: { monthly: '$19', annual: '$15' },
      period: billingPeriod === 'monthly' ? '/month' : '/month, billed annually',
      description: 'For dedicated practitioners seeking deeper guidance.',
      features: [
        'Unlimited conversations',
        'All specialized spiritual guides',
        'Full conversation history',
        'Personalized practice recommendations',
        'Priority support',
        'Advanced insights & analytics',
        'Guided meditation library',
      ],
      cta: 'Start 14-Day Free Trial',
      href: 'https://chat.dharmamind.ai/signup?plan=pro',
      highlighted: true,
      badge: 'Most Popular',
    },
    {
      name: 'Enterprise',
      price: { monthly: 'Custom', annual: 'Custom' },
      period: 'per organization',
      description: 'For teams, temples, and wellness organizations.',
      features: [
        'Everything in Pro',
        'Unlimited team members',
        'Custom AI training on your teachings',
        'SSO/SAML authentication',
        'Private deployment options',
        'Dedicated account manager',
        'Custom integrations & API',
        'SLA guarantee (99.9% uptime)',
      ],
      cta: 'Contact Sales',
      href: '/contact?type=enterprise',
      highlighted: false,
      badge: null,
    },
  ];

  const faqs = [
    {
      q: 'Can I try DharmaMind for free?',
      a: 'Yes! Our Free plan gives you 10 conversations per month at no cost. You can also start a 14-day free trial of Pro with full access to all features.',
    },
    {
      q: 'What happens after my free trial ends?',
      a: "You'll automatically be moved to the Free plan unless you choose to upgrade. We'll never charge you without your explicit consent.",
    },
    {
      q: 'Can I cancel anytime?',
      a: 'Absolutely. You can cancel your subscription at any time from your account settings. No questions asked, no hidden fees.',
    },
    {
      q: 'What payment methods do you accept?',
      a: 'We accept all major credit cards (Visa, Mastercard, American Express) through our secure payment processor Stripe. Enterprise customers can pay via invoice.',
    },
    {
      q: 'Is there a discount for annual billing?',
      a: 'Yes! When you choose annual billing, you save approximately 20% compared to monthly billing. That\'s like getting 2+ months free!',
    },
    {
      q: 'Do you offer discounts for spiritual organizations?',
      a: 'Yes, we offer special pricing for temples, monasteries, ashrams, and non-profit spiritual organizations. Contact us to learn more.',
    },
  ];

  const comparisonFeatures = [
    { feature: 'Monthly conversations', free: '10', pro: 'Unlimited', enterprise: 'Unlimited' },
    { feature: 'AI guidance quality', free: 'Basic', pro: 'Advanced', enterprise: 'Advanced + Custom' },
    { feature: 'Wisdom traditions', free: 'Core traditions', pro: 'All traditions', enterprise: 'All + Custom' },
    { feature: 'Conversation history', free: '7 days', pro: 'Unlimited', enterprise: 'Unlimited' },
    { feature: 'Meditation library', free: '—', pro: '✓', enterprise: '✓' },
    { feature: 'Practice recommendations', free: '—', pro: '✓', enterprise: '✓' },
    { feature: 'API access', free: '—', pro: '—', enterprise: '✓' },
    { feature: 'Custom training', free: '—', pro: '—', enterprise: '✓' },
    { feature: 'SSO/SAML', free: '—', pro: '—', enterprise: '✓' },
    { feature: 'Support', free: 'Community', pro: 'Priority', enterprise: 'Dedicated' },
    { feature: 'SLA', free: '—', pro: '—', enterprise: '99.9%' },
  ];

  return (
    <Layout
      title="Pricing"
      description="Simple, transparent pricing for spiritual growth. Start free, upgrade when you're ready."
    >
      {/* Hero */}
      <Section background="light" padding="xl">
        <div className="max-w-3xl mx-auto text-center">
          <h1 className="text-4xl md:text-6xl font-bold text-neutral-900 mb-6">
            Simple, Transparent Pricing
          </h1>
          <p className="text-xl text-neutral-600 leading-relaxed mb-8">
            Start your spiritual journey for free. Upgrade when you're ready for deeper guidance.
            No hidden fees, cancel anytime.
          </p>

          {/* Billing Toggle */}
          <div className="inline-flex items-center bg-neutral-100 rounded-full p-1 border border-neutral-200">
            <button
              onClick={() => setBillingPeriod('monthly')}
              className={`px-6 py-2 rounded-full text-sm font-medium transition-colors ${billingPeriod === 'monthly'
                ? 'bg-gold-600 text-white'
                : 'text-neutral-600 hover:text-neutral-900'
                }`}
            >
              Monthly
            </button>
            <button
              onClick={() => setBillingPeriod('annual')}
              className={`px-6 py-2 rounded-full text-sm font-medium transition-colors ${billingPeriod === 'annual'
                ? 'bg-gold-600 text-white'
                : 'text-neutral-600 hover:text-neutral-900'
                }`}
            >
              Annual
              <span className="ml-2 text-xs bg-gold-100 text-gold-700 px-2 py-0.5 rounded-full">
                Save 20%
              </span>
            </button>
          </div>
        </div>
      </Section>

      {/* Pricing Cards */}
      <Section>
        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {plans.map((plan) => (
            <div
              key={plan.name}
              className={`relative p-8 rounded-2xl transition-all ${plan.highlighted
                ? 'bg-gold-600 text-white shadow-xl scale-105'
                : 'bg-neutral-100 border border-neutral-200 hover:shadow-lg'
                }`}
            >
              {/* Badge */}
              {plan.badge && (
                <span className="absolute -top-3 left-1/2 -translate-x-1/2 px-4 py-1 bg-gold-600 text-neutral-900 text-xs font-semibold rounded-full">
                  {plan.badge}
                </span>
              )}

              {/* Plan Name */}
              <p className={`text-sm font-medium mb-2 ${plan.highlighted ? 'text-white/80' : 'text-neutral-500'}`}>
                {plan.name}
              </p>

              {/* Price */}
              <div className="flex items-baseline gap-1 mb-2">
                <span className="text-4xl font-bold">
                  {plan.price[billingPeriod]}
                </span>
                <span className={plan.highlighted ? 'text-white/70' : 'text-neutral-500'}>
                  {plan.period}
                </span>
              </div>

              {/* Description */}
              <p className={`mb-6 ${plan.highlighted ? 'text-white/80' : 'text-neutral-500'}`}>
                {plan.description}
              </p>

              {/* Features */}
              <ul className="space-y-3 mb-8">
                {plan.features.map((feature, j) => (
                  <li key={j} className="flex items-start gap-3 text-sm">
                    <svg
                      className={`w-5 h-5 flex-shrink-0 mt-0.5 ${plan.highlighted ? 'text-gold-600' : 'text-gold-600'}`}
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                    <span className={plan.highlighted ? 'text-white' : 'text-neutral-700'}>
                      {feature}
                    </span>
                  </li>
                ))}
              </ul>

              {/* CTA */}
              <Button
                href={plan.href}
                variant={plan.highlighted ? 'secondary' : 'primary'}
                className="w-full"
                external={plan.href.startsWith('http')}
              >
                {plan.cta}
              </Button>
            </div>
          ))}
        </div>
      </Section>

      {/* Feature Comparison Table */}
      <Section background="light">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-2xl font-bold text-neutral-900 text-center mb-8">
            Compare Plans
          </h2>

          <div className="bg-neutral-100 rounded-2xl border border-neutral-200 overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="border-b border-neutral-200">
                  <th className="text-left p-4 font-medium text-neutral-500">Feature</th>
                  <th className="text-center p-4 font-semibold text-neutral-900">Free</th>
                  <th className="text-center p-4 font-semibold text-neutral-900 bg-gold-50">Pro</th>
                  <th className="text-center p-4 font-semibold text-neutral-900">Enterprise</th>
                </tr>
              </thead>
              <tbody>
                {comparisonFeatures.map((row, i) => (
                  <tr key={i} className={i % 2 === 0 ? 'bg-neutral-100' : 'bg-neutral-100'}>
                    <td className="p-4 text-neutral-700">{row.feature}</td>
                    <td className="p-4 text-center text-neutral-600">{row.free}</td>
                    <td className="p-4 text-center text-neutral-900 bg-gold-50 font-medium">{row.pro}</td>
                    <td className="p-4 text-center text-neutral-600">{row.enterprise}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </Section>

      {/* FAQ */}
      <Section>
        <div className="max-w-3xl mx-auto">
          <h2 className="text-2xl font-bold text-neutral-900 text-center mb-12">
            Frequently Asked Questions
          </h2>
          <div className="space-y-4">
            {faqs.map((faq, i) => (
              <div key={i} className="bg-neutral-100 p-6 rounded-xl">
                <h3 className="font-semibold text-neutral-900 mb-2">{faq.q}</h3>
                <p className="text-neutral-600">{faq.a}</p>
              </div>
            ))}
          </div>

          <div className="text-center mt-8">
            <Link href="/faq" className="text-neutral-900 hover:underline">
              View all FAQs →
            </Link>
          </div>
        </div>
      </Section>

      {/* Money Back Guarantee */}
      <Section background="light">
        <div className="max-w-2xl mx-auto text-center">
          <span className="text-5xl mb-4 block">🛡️</span>
          <h2 className="text-2xl font-bold text-neutral-900 mb-4">
            30-Day Money Back Guarantee
          </h2>
          <p className="text-neutral-600">
            Not satisfied? We'll refund your payment within 30 days, no questions asked.
            We're confident you'll love DharmaMind, but we want you to feel completely secure.
          </p>
        </div>
      </Section>

      {/* Enterprise CTA */}
      <Section>
        <div className="bg-neutral-200 border border-neutral-300 rounded-3xl p-8 md:p-12 text-center">
          <h2 className="text-3xl font-bold text-neutral-900 mb-4">
            Need a Custom Solution?
          </h2>
          <p className="text-neutral-600 mb-8 max-w-2xl mx-auto">
            Whether you're a spiritual organization, wellness company, or educational institution,
            we can create a custom solution tailored to your needs.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button href="/contact?type=enterprise" variant="primary" size="lg">
              Contact Sales
            </Button>
            <Button
              href="/enterprise"
              variant="outline"
              size="lg"
            >
              Learn About Enterprise
            </Button>
          </div>
        </div>
      </Section>
    </Layout>
  );
}
=======
import React from 'react';
import Head from 'next/head';
import { motion } from 'framer-motion';
import { useRouter } from 'next/router';
import Logo from '../components/Logo';
import Footer from '../components/Footer';
import { useCentralizedSystem } from '../components/CentralizedSystem';
import { useSubscription } from '../hooks/useSubscription';
import { useAuth } from '../contexts/AuthContext';

const PricingPage: React.FC = () => {
  const router = useRouter();
  const { toggleSubscriptionModal, toggleAuthModal } = useCentralizedSystem();
  const { subscriptionPlans, isLoading } = useSubscription();
  const { isAuthenticated, user } = useAuth();

  const handleSelectPlan = (planId: string) => {
    if (isAuthenticated) {
      toggleSubscriptionModal(true);
    } else {
      toggleAuthModal(true);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-white to-brand-accent/5">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-brand-accent mx-auto mb-4"></div>
          <p className="text-secondary">Loading pricing plans...</p>
        </div>
      </div>
    );
  }

  return (
    <>
      <Head>
        <title>Pricing - DharmaMind</title>
        <meta name="description" content="Choose the perfect plan for your spiritual journey with DharmaMind" />
      </Head>

      <div className="min-h-screen bg-gradient-to-br from-white via-brand-accent/5 to-white">
        {/* Header */}
        <header className="relative z-50 bg-white/95 backdrop-blur-xl border-b border-gray-100">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center py-4">
              <Logo />
              <nav className="hidden md:flex space-x-8">
                <button onClick={() => router.push('/')} className="text-secondary hover:text-primary transition-colors">
                  Home
                </button>
                <button onClick={() => router.push('/features')} className="text-secondary hover:text-primary transition-colors">
                  Features
                </button>
                <button onClick={() => router.push('/about')} className="text-secondary hover:text-primary transition-colors">
                  About
                </button>
              </nav>
              <div className="flex items-center space-x-4">
                {isAuthenticated ? (
                  <span className="text-secondary">Welcome, {user?.first_name || 'User'}</span>
                ) : (
                  <button 
                    onClick={() => toggleAuthModal(true)}
                    className="btn-outline"
                  >
                    Sign In
                  </button>
                )}
              </div>
            </div>
          </div>
        </header>

        {/* Hero Section */}
        <section className="py-20">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              <div className="inline-flex items-center space-x-3 bg-white/80 backdrop-blur-xl rounded-full px-6 py-3 border border-gray-200 shadow-lg mb-8">
                <span className="text-2xl">💎</span>
                <span className="text-lg font-semibold text-primary">Pricing</span>
              </div>
              <h1 className="text-5xl md:text-6xl font-bold text-primary mb-6">
                Choose Your Spiritual Journey
              </h1>
              <p className="text-xl text-secondary max-w-3xl mx-auto">
                Start your path to enlightenment with our AI-powered dharmic guidance
              </p>
            </motion.div>
          </div>
        </section>

        {/* Centralized Pricing Section */}
        <section className="py-12">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            {subscriptionPlans && subscriptionPlans.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {subscriptionPlans.map((plan, index) => (
                  <motion.div
                    key={plan.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: index * 0.1 }}
                    className={`relative bg-white rounded-3xl p-8 shadow-xl border transition-all duration-300 hover:shadow-2xl hover:scale-105 ${
                      plan.popular ? 'border-brand-accent border-2' : 'border-gray-200'
                    }`}
                  >
                    {plan.popular && (
                      <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                        <span className="bg-brand-accent text-white px-4 py-1 text-sm font-medium rounded-full">
                          Most Popular
                        </span>
                      </div>
                    )}
                    
                    <div className="text-center mb-8">
                      <div className="w-16 h-16 bg-brand-gradient rounded-2xl flex items-center justify-center mx-auto mb-4">
                        <span className="text-white text-2xl">
                          {plan.tier === 'basic' ? '🌱' : plan.tier === 'pro' ? '🧘‍♀️' : '⭐'}
                        </span>
                      </div>
                      <h3 className="text-2xl font-bold text-primary mb-2">{plan.name}</h3>
                      <p className="text-secondary">{plan.description}</p>
                    </div>

                    <div className="text-center mb-8">
                      <span className="text-5xl font-bold text-primary">${plan.price.monthly}</span>
                      <span className="text-secondary">/month</span>
                      {plan.price.yearly && plan.price.yearly < plan.price.monthly * 12 && (
                        <div className="text-sm text-brand-accent mt-2">
                          Save ${(plan.price.monthly * 12) - plan.price.yearly}/year with annual billing
                        </div>
                      )}
                    </div>

                    <ul className="space-y-4 text-secondary mb-8">
                      {plan.features.map((feature, featureIndex) => (
                        <li key={featureIndex} className="flex items-center">
                          <div className="w-5 h-5 bg-brand-accent rounded-full flex items-center justify-center mr-3">
                            <span className="text-white text-xs">✓</span>
                          </div>
                          {feature.description || feature.feature_id}
                        </li>
                      ))}
                    </ul>

                    <button
                      onClick={() => handleSelectPlan(plan.id)}
                      className={`w-full py-3 rounded-xl font-medium transition-all duration-300 ${
                        plan.popular
                          ? 'bg-brand-gradient text-white hover:shadow-lg'
                          : 'border-2 border-brand-accent text-brand-accent hover:bg-brand-accent hover:text-white'
                      }`}
                    >
                      {plan.tier === 'basic' ? 'Get Started Free' : `Choose ${plan.name}`}
                    </button>
                  </motion.div>
                ))}
              </div>
            ) : (
              <div className="text-center py-20">
                <p className="text-xl text-secondary">Loading pricing plans...</p>
                <div className="mt-8">
                  <button
                    onClick={() => toggleSubscriptionModal(true)}
                    className="btn-primary"
                  >
                    View All Plans
                  </button>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* CTA Section */}
        <section className="py-20 bg-gradient-to-r from-brand-primary/5 to-brand-accent/5">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
            >
              <h2 className="text-4xl font-bold text-primary mb-6">
                Ready to Begin Your Journey?
              </h2>
              <p className="text-xl text-secondary mb-8">
                Join thousands of practitioners finding clarity and wisdom with DharmaMind
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <button
                  onClick={() => isAuthenticated ? toggleSubscriptionModal(true) : toggleAuthModal(true)}
                  className="btn-primary text-lg px-8 py-4"
                >
                  Get Started Today
                </button>
                <button
                  onClick={() => router.push('/features')}
                  className="btn-outline text-lg px-8 py-4"
                >
                  Explore Features
                </button>
              </div>
            </motion.div>
          </div>
        </section>

        <Footer />
      </div>
    </>
  );
};

export default PricingPage;
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
