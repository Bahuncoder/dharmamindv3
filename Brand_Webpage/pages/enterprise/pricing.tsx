import React, { useState } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';
import BrandHeader from '../../components/BrandHeader';
import Footer from '../../components/Footer';
import Button from '../../components/Button';

const EnterprisePricingPage: React.FC = () => {
    const router = useRouter();
    const [billingCycle, setBillingCycle] = useState<'monthly' | 'annual'>('monthly');

    const pricingTiers = [
        {
            name: "Starter",
            description: "Perfect for small teams exploring dharmic decision-making",
            monthlyPrice: 99,
            annualPrice: 999,
            userLimit: "Up to 25 users",
            features: [
                "Core AI Ethics Engine",
                "Basic meditation guidance",
                "Team wellness dashboard",
                "Email support",
                "Standard integrations",
                "Mobile & web access"
            ],
            limitations: [
                "Limited to basic dharmic principles",
                "Standard response times"
            ],
            color: "border-gold-400 bg-neutral-100",
            buttonText: "Start Free Trial"
        },
        {
            name: "Professional",
            description: "Comprehensive solution for growing organizations",
            monthlyPrice: 299,
            annualPrice: 2999,
            userLimit: "Up to 100 users",
            features: [
                "Advanced AI Ethics Engine",
                "Personalized wisdom paths",
                "Advanced analytics & insights",
                "Priority support",
                "Custom integrations",
                "API access",
                "Advanced security features",
                "Custom branding options"
            ],
            limitations: [],
            color: "border-gold-400 bg-accent-50",
            buttonText: "Start Professional",
            popular: true
        },
        {
            name: "Enterprise",
            description: "Full-scale dharmic transformation for large organizations",
            monthlyPrice: null,
            annualPrice: null,
            userLimit: "Unlimited users",
            features: [
                "Complete AI Wisdom Platform",
                "Custom dharmic frameworks",
                "White-label solutions",
                "Dedicated success manager",
                "24/7 premium support",
                "On-premise deployment",
                "Custom integrations",
                "Advanced compliance tools",
                "Executive dashboard",
                "Multi-region support"
            ],
            limitations: [],
            color: "border-gold-400 bg-neutral-100",
            buttonText: "Contact Sales",
            isCustom: true
        }
    ];

    const addOns = [
        {
            name: "Advanced Analytics",
            description: "Deep insights into team consciousness and organizational alignment",
            monthlyPrice: 49,
            features: [
                "Consciousness metrics tracking",
                "Predictive wellness analytics",
                "Custom report generation",
                "Real-time dashboards"
            ]
        },
        {
            name: "Custom Integration",
            description: "Seamless connection with your existing business systems",
            monthlyPrice: 99,
            features: [
                "Custom API development",
                "Legacy system integration",
                "Data migration assistance",
                "Ongoing integration support"
            ]
        },
        {
            name: "Training & Onboarding",
            description: "Comprehensive team training and change management",
            monthlyPrice: 199,
            features: [
                "Executive leadership training",
                "Team workshops",
                "Change management consulting",
                "Ongoing coaching sessions"
            ]
        }
    ];

    const comparisonFeatures = [
        {
            category: "Core Features",
            features: [
                { name: "AI Ethics Engine", starter: "Basic", professional: "Advanced", enterprise: "Custom" },
                { name: "User Limit", starter: "25", professional: "100", enterprise: "Unlimited" },
                { name: "Meditation Library", starter: "✓", professional: "✓", enterprise: "✓" },
                { name: "Mobile Access", starter: "✓", professional: "✓", enterprise: "✓" }
            ]
        },
        {
            category: "Analytics & Insights",
            features: [
                { name: "Basic Dashboard", starter: "✓", professional: "✓", enterprise: "✓" },
                { name: "Advanced Analytics", starter: "—", professional: "✓", enterprise: "✓" },
                { name: "Custom Reports", starter: "—", professional: "Limited", enterprise: "Unlimited" },
                { name: "Executive Dashboard", starter: "—", professional: "—", enterprise: "✓" }
            ]
        },
        {
            category: "Support & Services",
            features: [
                { name: "Email Support", starter: "✓", professional: "✓", enterprise: "✓" },
                { name: "Priority Support", starter: "—", professional: "✓", enterprise: "✓" },
                { name: "24/7 Support", starter: "—", professional: "—", enterprise: "✓" },
                { name: "Dedicated Manager", starter: "—", professional: "—", enterprise: "✓" }
            ]
        },
        {
            category: "Enterprise Features",
            features: [
                { name: "Custom Branding", starter: "—", professional: "Basic", enterprise: "Full" },
                { name: "API Access", starter: "—", professional: "Standard", enterprise: "Enterprise" },
                { name: "On-Premise Deployment", starter: "—", professional: "—", enterprise: "✓" },
                { name: "Multi-Region Support", starter: "—", professional: "—", enterprise: "✓" }
            ]
        }
    ];

    const calculateAnnualSavings = (monthly: number, annual: number) => {
        const annualTotal = monthly * 12;
        const savings = annualTotal - annual;
        const percentage = Math.round((savings / annualTotal) * 100);
        return { savings, percentage };
    };

    return (
        <>
            <Head>
                <title>Enterprise Pricing - DharmaMind</title>
                <meta name="description" content="Transparent pricing for enterprise dharmic transformation with flexible plans and custom solutions" />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <link rel="icon" href="/favicon.ico" />
            </Head>

            <div className="min-h-screen bg-neutral-100">
                {/* Enterprise Brand Header */}
                <BrandHeader
                    isEnterprise={true}
                    breadcrumbs={[
                        { label: 'Home', href: '/' },
                        { label: 'Enterprise', href: '/enterprise' },
                        { label: 'Pricing', href: '/enterprise/pricing' }
                    ]}
                />

                {/* Hero Section */}
                <div className="bg-neutral-100 border-b-2 border-gold-400 py-16">
                    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.8 }}
                            className="text-center"
                        >
                            <h1 className="text-4xl lg:text-5xl font-bold text-neutral-900 mb-6">
                                Enterprise <span className="text-gold-600 border-b-4 border-gold-400">Pricing</span>
                            </h1>
                            <p className="text-xl text-neutral-600 max-w-3xl mx-auto leading-relaxed mb-8">
                                Transparent, scalable pricing that grows with your dharmic transformation journey. All plans include our core wisdom engine.
                            </p>

                            {/* Billing Toggle */}
                            <div className="flex items-center justify-center space-x-4 mb-8">
                                <span className={`text-sm ${billingCycle === 'monthly' ? 'text-gold-600 font-semibold' : 'text-neutral-600'}`}>
                                    Monthly
                                </span>
                                <button
                                    onClick={() => setBillingCycle(billingCycle === 'monthly' ? 'annual' : 'monthly')}
                                    className="relative inline-flex h-6 w-11 items-center rounded-full bg-primary-background border-2 border-gold-400 transition-colors focus:outline-none focus:ring-2 focus:ring-brand-accent focus:ring-offset-2"
                                >
                                    <span
                                        className={`inline-block h-4 w-4 transform rounded-full bg-accent transition-transform ${billingCycle === 'annual' ? 'translate-x-6' : 'translate-x-1'
                                            }`}
                                    />
                                </button>
                                <span className={`text-sm ${billingCycle === 'annual' ? 'text-gold-600 font-semibold' : 'text-neutral-600'}`}>
                                    Annual
                                    <span className="ml-1 text-xs bg-accent text-white px-2 py-1 rounded-full">Save 15%</span>
                                </span>
                            </div>
                        </motion.div>
                    </div>
                </div>

                {/* Pricing Tiers */}
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-16">
                        {pricingTiers.map((tier, index) => (
                            <motion.div
                                key={index}
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ duration: 0.6, delay: index * 0.1 }}
                                className={`${tier.color} border-2 rounded-lg p-6 relative ${tier.popular ? 'transform scale-105 shadow-lg' : ''}`}
                            >
                                {tier.popular && (
                                    <div className="absolute -top-3 left-1/2 transform -translate-x-1/2 bg-accent text-white px-4 py-1 rounded-full text-sm font-medium">
                                        Most Popular
                                    </div>
                                )}

                                <div className="mb-6">
                                    <h3 className="text-2xl font-bold text-neutral-900 mb-2">{tier.name}</h3>
                                    <p className="text-neutral-600 text-sm mb-4">{tier.description}</p>

                                    <div className="mb-4">
                                        {tier.isCustom ? (
                                            <div className="text-3xl font-bold text-gold-600">Custom</div>
                                        ) : (
                                            <div>
                                                <div className="text-3xl font-bold text-gold-600">
                                                    ${billingCycle === 'monthly' ? tier.monthlyPrice : Math.round(tier.annualPrice! / 12)}
                                                    <span className="text-lg text-neutral-600">/month</span>
                                                </div>
                                                {billingCycle === 'annual' && (
                                                    <div className="text-sm text-neutral-600">
                                                        ${tier.annualPrice}/year
                                                        {tier.monthlyPrice && tier.annualPrice && (
                                                            <span className="ml-2 text-gold-600 font-medium">
                                                                (Save ${calculateAnnualSavings(tier.monthlyPrice, tier.annualPrice).savings})
                                                            </span>
                                                        )}
                                                    </div>
                                                )}
                                            </div>
                                        )}
                                    </div>

                                    <div className="text-sm text-neutral-600 mb-6">{tier.userLimit}</div>
                                </div>

                                <ul className="space-y-3 mb-8">
                                    {tier.features.map((feature, idx) => (
                                        <li key={idx} className="flex items-center space-x-2">
                                            <span className="text-gold-600 text-sm">✓</span>
                                            <span className="text-sm text-neutral-900">{feature}</span>
                                        </li>
                                    ))}
                                </ul>

                                <Button
                                    className={`w-full ${tier.popular ? 'bg-accent hover:bg-accent text-white' : 'bg-neutral-100 hover:bg-primary-background text-neutral-900'} border-2 border-gold-400 py-3`}
                                    onClick={() => router.push(tier.isCustom ? '/contact' : '/signup')}
                                >
                                    {tier.buttonText}
                                </Button>
                            </motion.div>
                        ))}
                    </div>
                </div>

                {/* Add-ons */}
                <div className="bg-primary-background border-t-2 border-gold-400 py-16">
                    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            className="text-center mb-12"
                        >
                            <h2 className="text-3xl font-bold text-neutral-900 mb-4 border-b-4 border-gold-400 inline-block pb-2">
                                Add-on Services
                            </h2>
                            <p className="text-lg text-neutral-600">
                                Enhance your DharmaMind experience with specialized services
                            </p>
                        </motion.div>

                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                            {addOns.map((addon, index) => (
                                <motion.div
                                    key={index}
                                    initial={{ opacity: 0, y: 20 }}
                                    whileInView={{ opacity: 1, y: 0 }}
                                    viewport={{ once: true }}
                                    transition={{ duration: 0.6, delay: index * 0.1 }}
                                    className="bg-neutral-100 border-2 border-gold-400 rounded-lg p-6 hover:shadow-lg transition-all duration-300"
                                >
                                    <div className="mb-6">
                                        <h3 className="text-xl font-semibold text-neutral-900 mb-2">{addon.name}</h3>
                                        <p className="text-neutral-600 text-sm mb-4">{addon.description}</p>
                                        <div className="text-2xl font-bold text-gold-600">
                                            ${addon.monthlyPrice}
                                            <span className="text-lg text-neutral-600">/month</span>
                                        </div>
                                    </div>

                                    <ul className="space-y-2 mb-6">
                                        {addon.features.map((feature, idx) => (
                                            <li key={idx} className="flex items-center space-x-2">
                                                <span className="text-gold-600 text-sm">✓</span>
                                                <span className="text-sm text-neutral-900">{feature}</span>
                                            </li>
                                        ))}
                                    </ul>

                                    <Button
                                        className="w-full bg-accent hover:bg-accent text-white border-2 border-gold-400 py-2"
                                        onClick={() => router.push('/contact')}
                                    >
                                        Add to Plan
                                    </Button>
                                </motion.div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Feature Comparison Table */}
                <div className="bg-neutral-100 border-t-2 border-gold-400 py-16">
                    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            className="text-center mb-12"
                        >
                            <h2 className="text-3xl font-bold text-neutral-900 mb-4 border-b-4 border-gold-400 inline-block pb-2">
                                Feature Comparison
                            </h2>
                            <p className="text-lg text-neutral-600">
                                Detailed comparison of features across all plans
                            </p>
                        </motion.div>

                        <div className="overflow-x-auto">
                            <table className="w-full bg-primary-background border-2 border-gold-400 rounded-lg">
                                <thead>
                                    <tr className="border-b-2 border-gold-400">
                                        <th className="p-4 text-left text-neutral-900 font-semibold">Feature</th>
                                        <th className="p-4 text-center text-neutral-900 font-semibold">Starter</th>
                                        <th className="p-4 text-center text-neutral-900 font-semibold">Professional</th>
                                        <th className="p-4 text-center text-neutral-900 font-semibold">Enterprise</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {comparisonFeatures.map((category, categoryIndex) => (
                                        <React.Fragment key={categoryIndex}>
                                            <tr>
                                                <td className="p-4 font-bold text-neutral-900 bg-neutral-100 border-b border-gold-400/30" colSpan={4}>
                                                    {category.category}
                                                </td>
                                            </tr>
                                            {category.features.map((feature, featureIndex) => (
                                                <motion.tr
                                                    key={featureIndex}
                                                    initial={{ opacity: 0, x: -20 }}
                                                    whileInView={{ opacity: 1, x: 0 }}
                                                    viewport={{ once: true }}
                                                    transition={{ duration: 0.5, delay: (categoryIndex * category.features.length + featureIndex) * 0.05 }}
                                                    className="border-b border-gold-400/30"
                                                >
                                                    <td className="p-4 text-neutral-900">{feature.name}</td>
                                                    <td className="p-4 text-center text-neutral-600">{feature.starter}</td>
                                                    <td className="p-4 text-center text-neutral-600">{feature.professional}</td>
                                                    <td className="p-4 text-center text-gold-600 font-semibold">{feature.enterprise}</td>
                                                </motion.tr>
                                            ))}
                                        </React.Fragment>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                {/* FAQ Section */}
                <div className="bg-primary-background border-t-2 border-gold-400 py-16">
                    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            className="text-center mb-12"
                        >
                            <h2 className="text-3xl font-bold text-neutral-900 mb-4 border-b-4 border-gold-400 inline-block pb-2">
                                Frequently Asked Questions
                            </h2>
                        </motion.div>

                        <div className="space-y-6">
                            {[
                                {
                                    question: "Can I switch between plans?",
                                    answer: "Yes, you can upgrade or downgrade your plan at any time. Changes take effect at your next billing cycle."
                                },
                                {
                                    question: "Is there a free trial available?",
                                    answer: "Yes, all plans come with a 14-day free trial. No credit card required to get started."
                                },
                                {
                                    question: "What payment methods do you accept?",
                                    answer: "We accept all major credit cards, ACH transfers, and wire transfers for enterprise customers."
                                },
                                {
                                    question: "Do you offer discounts for non-profits?",
                                    answer: "Yes, we offer special pricing for qualifying non-profit organizations. Contact our sales team for details."
                                }
                            ].map((faq, index) => (
                                <motion.div
                                    key={index}
                                    initial={{ opacity: 0, y: 20 }}
                                    whileInView={{ opacity: 1, y: 0 }}
                                    viewport={{ once: true }}
                                    transition={{ duration: 0.5, delay: index * 0.1 }}
                                    className="bg-neutral-100 border-2 border-gold-400 rounded-lg p-6"
                                >
                                    <h3 className="text-lg font-semibold text-neutral-900 mb-3">{faq.question}</h3>
                                    <p className="text-neutral-600">{faq.answer}</p>
                                </motion.div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* CTA Section */}
                <div className="bg-neutral-100 border-t-2 border-gold-400 py-16">
                    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            className="space-y-6"
                        >
                            <h2 className="text-3xl font-bold text-neutral-900">
                                Ready to Transform Your Organization?
                            </h2>
                            <p className="text-lg text-neutral-600 max-w-2xl mx-auto">
                                Start your 14-day free trial today or speak with our sales team about custom enterprise solutions.
                            </p>
                            <div className="flex flex-col sm:flex-row gap-4 justify-center">
                                <Button
                                    className="bg-accent hover:bg-accent text-white border-2 border-gold-400 px-8 py-3"
                                    onClick={() => router.push('/signup')}
                                >
                                    Start Free Trial
                                </Button>
                                <Button
                                    className="bg-neutral-100 hover:bg-primary-background text-neutral-900 border-2 border-gold-400 px-8 py-3"
                                    onClick={() => router.push('/contact')}
                                >
                                    Contact Sales
                                </Button>
                            </div>
                        </motion.div>
                    </div>
                </div>

                <Footer />
            </div>
        </>
    );
};

export default EnterprisePricingPage;
