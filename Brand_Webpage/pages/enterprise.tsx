import Link from 'next/link';
import { Layout, Section, Button } from '../components';

export default function EnterprisePage() {
  const features = [
    {
      title: 'Team Management',
      description: 'Centralized dashboard for managing users, roles, and permissions across your organization.',
      icon: '👥',
    },
    {
      title: 'Advanced Security',
      description: 'SOC 2 Type II compliant with end-to-end encryption, SSO/SAML, and advanced access controls.',
      icon: '🛡️',
    },
    {
      title: 'Custom Deployment',
      description: 'Flexible options including cloud, on-premise, or hybrid deployment to meet your requirements.',
      icon: '🏗️',
    },
    {
      title: 'API & Integrations',
      description: 'RESTful API with comprehensive documentation. Pre-built integrations for popular tools.',
      icon: '🔌',
    },
    {
      title: 'Analytics Dashboard',
      description: 'Detailed insights into usage patterns, team engagement, and ROI metrics.',
      icon: '📊',
    },
    {
      title: 'Priority Support',
      description: 'Dedicated account manager, 24/7 support, and guaranteed response times via SLA.',
      icon: '💬',
    },
  ];

  const securityFeatures = [
    'SOC 2 Type II Certified',
    'End-to-end encryption (AES-256)',
    'SSO/SAML 2.0 integration',
    'Role-based access control',
    'Audit logs and compliance reporting',
    'GDPR and CCPA compliant',
  ];

  const pricingFeatures = [
    'Unlimited users',
    'Custom deployment',
    'Dedicated support',
    'SLA guarantee',
    'Custom integrations',
  ];

  return (
    <Layout title="Enterprise" description="Enterprise-grade AI platform for organizations. Advanced security, custom deployment, and dedicated support.">
      {/* Hero Section */}
      <Section background="white" padding="xl">
        <div className="max-w-4xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-gold-100 rounded-full text-sm text-gold-700 mb-8">
            Enterprise Solutions
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-neutral-900 leading-tight mb-6">
            Built for organizations
            <br />
            <span className="text-neutral-400">that demand more</span>
          </h1>
          <p className="text-xl text-neutral-600 max-w-2xl mx-auto mb-10 leading-relaxed">
            Deploy DharmaMind across your organization with enterprise-grade security,
            custom integrations, and dedicated support.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Button href="/contact" size="lg">
              Contact Sales
            </Button>
            <Button href="https://dharmamind.ai" variant="outline" size="lg" external>
              Start free trial
            </Button>
          </div>
        </div>
      </Section>

      {/* Features Grid */}
      <Section background="light" padding="xl">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-neutral-900 mb-4">Enterprise Features</h2>
            <p className="text-lg text-neutral-600 max-w-2xl mx-auto">
              Everything your organization needs to scale AI adoption securely and efficiently.
            </p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, i) => (
              <div key={i} className="bg-white p-8 rounded-xl border border-neutral-200 hover:shadow-md transition-shadow">
                <div className="w-12 h-12 bg-gold-100 rounded-lg flex items-center justify-center mb-5 text-2xl">
                  {feature.icon}
                </div>
                <h3 className="text-lg font-semibold text-neutral-900 mb-2">{feature.title}</h3>
                <p className="text-neutral-600 leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* Security Section */}
      <Section background="white" padding="xl">
        <div className="max-w-6xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-3xl font-bold text-neutral-900 mb-6">
                Enterprise-grade security
              </h2>
              <p className="text-lg text-neutral-600 mb-8 leading-relaxed">
                Your data security is our top priority. DharmaMind is built with security
                at every layer to meet the strictest compliance requirements.
              </p>
              <ul className="space-y-4">
                {securityFeatures.map((item, i) => (
                  <li key={i} className="flex items-center gap-3 text-neutral-700">
                    <span className="text-gold-600">✓</span>
                    {item}
                  </li>
                ))}
              </ul>
            </div>
            <div className="bg-neutral-100 rounded-2xl p-12 flex items-center justify-center">
              <div className="text-center">
                <div className="w-20 h-20 bg-gold-100 rounded-2xl mx-auto mb-4 flex items-center justify-center shadow-sm">
                  <span className="text-4xl">🛡️</span>
                </div>
                <p className="text-sm text-neutral-600">Security first approach</p>
              </div>
            </div>
          </div>
        </div>
      </Section>

      {/* Pricing */}
      <Section background="light" padding="xl">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold text-neutral-900 mb-4">Simple, transparent pricing</h2>
          <p className="text-lg text-neutral-600 mb-12">
            Custom pricing based on your organization&apos;s needs. Contact us for a quote.
          </p>
          <div className="bg-white p-8 rounded-xl border border-neutral-200 max-w-md mx-auto">
            <p className="text-sm text-neutral-500 uppercase tracking-wider mb-2">Enterprise</p>
            <p className="text-4xl font-bold text-neutral-900 mb-4">Custom</p>
            <p className="text-neutral-600 mb-6">
              Tailored to your organization&apos;s size and requirements
            </p>
            <ul className="text-left space-y-3 mb-8">
              {pricingFeatures.map((item, i) => (
                <li key={i} className="flex items-center gap-3 text-neutral-700 text-sm">
                  <span className="text-gold-600">✓</span>
                  {item}
                </li>
              ))}
            </ul>
            <Button href="/contact" className="w-full">
              Contact Sales
            </Button>
          </div>
        </div>
      </Section>

      {/* CTA */}
      <Section background="dark" padding="xl">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">Ready to transform your organization?</h2>
          <p className="text-lg text-neutral-400 mb-8">
            Schedule a demo with our enterprise team to see how DharmaMind can help.
          </p>
          <Button href="/contact" size="lg">
            Schedule a demo
          </Button>
        </div>
      </Section>
    </Layout>
  );
}
