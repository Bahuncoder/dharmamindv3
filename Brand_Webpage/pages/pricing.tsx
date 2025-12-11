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
