import React from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';
import BrandHeader from '../../components/BrandHeader';
import Footer from '../../components/Footer';
import Button from '../../components/Button';

const EnterpriseSupportPage: React.FC = () => {
    const router = useRouter();

    const supportTiers = [
        {
            name: "Essential",
            price: "Included",
            description: "Basic support for getting started with DharmaMind",
            features: [
                "Email support (48h response)",
                "Knowledge base access",
                "Community forum access",
                "Basic onboarding materials",
                "Monthly office hours"
            ],
            color: "border-gold-400 bg-neutral-100"
        },
        {
            name: "Professional",
            price: "$2,500/month",
            description: "Enhanced support for growing organizations",
            features: [
                "Priority email support (24h response)",
                "Phone support (business hours)",
                "Dedicated success manager",
                "Custom training sessions",
                "Implementation guidance",
                "Quarterly business reviews"
            ],
            color: "border-gold-400 bg-accent-50",
            popular: true
        },
        {
            name: "Enterprise",
            price: "Custom",
            description: "White-glove support for mission-critical deployments",
            features: [
                "24/7 phone & email support",
                "Dedicated technical account manager",
                "On-site implementation support",
                "Custom integrations support",
                "SLA guarantees",
                "Priority feature requests"
            ],
            color: "border-gold-400 bg-neutral-100"
        }
    ];

    const supportChannels = [
        {
            icon: "📧",
            title: "Email Support",
            description: "Get help via email with detailed responses from our expert team",
            availability: "24/7 submission, response within SLA",
            action: "Send Email"
        },
        {
            icon: "📞",
            title: "Phone Support",
            description: "Direct phone access to our support engineers for urgent issues",
            availability: "Business hours (PST) for Pro+",
            action: "Call Now"
        },
        {
            icon: "💬",
            title: "Live Chat",
            description: "Real-time chat support for quick questions and guidance",
            availability: "Business hours (PST)",
            action: "Start Chat"
        },
        {
            icon: "🎥",
            title: "Screen Sharing",
            description: "One-on-one screen sharing sessions for complex troubleshooting",
            availability: "By appointment",
            action: "Schedule Session"
        }
    ];

    const resources = [
        {
            icon: "📚",
            title: "Knowledge Base",
            description: "Comprehensive documentation and tutorials",
            items: ["Setup guides", "Integration docs", "Troubleshooting", "Best practices"]
        },
        {
            icon: "🎓",
            title: "Training Center",
            description: "Interactive courses and certification programs",
            items: ["Admin certification", "User training", "Custom workshops", "Webinar series"]
        },
        {
            icon: "👥",
            title: "Community Forum",
            description: "Connect with other users and share experiences",
            items: ["User discussions", "Feature requests", "Success stories", "Expert tips"]
        },
        {
            icon: "🔧",
            title: "Developer Hub",
            description: "Technical resources for developers and IT teams",
            items: ["API documentation", "SDK libraries", "Code samples", "Integration guides"]
        }
    ];

    const slaMetrics = [
        { metric: "Initial Response", essential: "48 hours", professional: "24 hours", enterprise: "4 hours" },
        { metric: "Resolution Time", essential: "5 business days", professional: "3 business days", enterprise: "Same day" },
        { metric: "Uptime SLA", essential: "99.5%", professional: "99.9%", enterprise: "99.95%" },
        { metric: "Support Hours", essential: "Business hours", professional: "Extended hours", enterprise: "24/7" }
    ];

    return (
        <>
            <Head>
                <title>Enterprise Support - DharmaMind</title>
                <meta name="description" content="Comprehensive enterprise support with dedicated success managers, training, and 24/7 assistance" />
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
                        { label: 'Support', href: '/enterprise/support' }
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
                                Enterprise <span className="text-gold-600 border-b-4 border-gold-400">Support</span>
                            </h1>
                            <p className="text-xl text-neutral-600 max-w-3xl mx-auto leading-relaxed">
                                Comprehensive support services designed to ensure your success with dedicated experts, training, and 24/7 assistance.
                            </p>
                        </motion.div>
                    </div>
                </div>

                {/* Support Tiers */}
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        className="text-center mb-12"
                    >
                        <h2 className="text-3xl font-bold text-neutral-900 mb-4 border-b-4 border-gold-400 inline-block pb-2">
                            Support Plans
                        </h2>
                        <p className="text-lg text-neutral-600">
                            Choose the level of support that matches your organization's needs
                        </p>
                    </motion.div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
                        {supportTiers.map((tier, index) => (
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
                                <div className="text-center mb-6">
                                    <h3 className="text-2xl font-bold text-neutral-900 mb-2">{tier.name}</h3>
                                    <div className="text-3xl font-bold text-gold-600 mb-2">{tier.price}</div>
                                    <p className="text-neutral-600">{tier.description}</p>
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
                                    className={`w-full ${tier.popular ? 'bg-accent hover:bg-accent text-white' : 'bg-neutral-100 hover:bg-primary-background text-neutral-900'} border-2 border-gold-400 py-2`}
                                    onClick={() => router.push('/contact')}
                                >
                                    Get Started
                                </Button>
                            </motion.div>
                        ))}
                    </div>
                </div>

                {/* Support Channels */}
                <div className="bg-neutral-50 border-t-2 border-gold-400 py-16">
                    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            className="text-center mb-12"
                        >
                            <h2 className="text-3xl font-bold text-neutral-900 mb-4 border-b-4 border-gold-400 inline-block pb-2">
                                Support Channels
                            </h2>
                            <p className="text-lg text-neutral-600">
                                Multiple ways to get help when you need it
                            </p>
                        </motion.div>

                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                            {supportChannels.map((channel, index) => (
                                <motion.div
                                    key={index}
                                    initial={{ opacity: 0, y: 20 }}
                                    whileInView={{ opacity: 1, y: 0 }}
                                    viewport={{ once: true }}
                                    transition={{ duration: 0.6, delay: index * 0.1 }}
                                    className="bg-neutral-100 border-2 border-gold-400 rounded-lg p-6 text-center hover:shadow-lg transition-all duration-300"
                                >
                                    <div className="text-4xl mb-4">{channel.icon}</div>
                                    <h3 className="text-lg font-semibold text-neutral-900 mb-2">{channel.title}</h3>
                                    <p className="text-neutral-600 text-sm mb-3">{channel.description}</p>
                                    <p className="text-xs text-gold-600 mb-4">{channel.availability}</p>
                                    <Button
                                        className="bg-accent hover:bg-accent text-white border-2 border-gold-400 px-4 py-2 text-sm"
                                        onClick={() => router.push('/contact')}
                                    >
                                        {channel.action}
                                    </Button>
                                </motion.div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* SLA Metrics */}
                <div className="bg-neutral-100 border-t-2 border-gold-400 py-16">
                    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            className="text-center mb-12"
                        >
                            <h2 className="text-3xl font-bold text-neutral-900 mb-4 border-b-4 border-gold-400 inline-block pb-2">
                                Service Level Agreements
                            </h2>
                            <p className="text-lg text-neutral-600">
                                Guaranteed response times and service levels
                            </p>
                        </motion.div>

                        <div className="overflow-x-auto">
                            <table className="w-full bg-neutral-50 border-2 border-gold-400 rounded-lg">
                                <thead>
                                    <tr className="border-b-2 border-gold-400">
                                        <th className="p-4 text-left text-neutral-900 font-semibold">Metric</th>
                                        <th className="p-4 text-center text-neutral-900 font-semibold">Essential</th>
                                        <th className="p-4 text-center text-neutral-900 font-semibold">Professional</th>
                                        <th className="p-4 text-center text-neutral-900 font-semibold">Enterprise</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {slaMetrics.map((row, index) => (
                                        <motion.tr
                                            key={index}
                                            initial={{ opacity: 0, x: -20 }}
                                            whileInView={{ opacity: 1, x: 0 }}
                                            viewport={{ once: true }}
                                            transition={{ duration: 0.5, delay: index * 0.1 }}
                                            className="border-b border-gold-400/30"
                                        >
                                            <td className="p-4 font-medium text-neutral-900">{row.metric}</td>
                                            <td className="p-4 text-center text-neutral-600">{row.essential}</td>
                                            <td className="p-4 text-center text-neutral-600">{row.professional}</td>
                                            <td className="p-4 text-center text-gold-600 font-semibold">{row.enterprise}</td>
                                        </motion.tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                {/* Resources */}
                <div className="bg-neutral-50 border-t-2 border-gold-400 py-16">
                    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            className="text-center mb-12"
                        >
                            <h2 className="text-3xl font-bold text-neutral-900 mb-4 border-b-4 border-gold-400 inline-block pb-2">
                                Self-Service Resources
                            </h2>
                            <p className="text-lg text-neutral-600">
                                Comprehensive resources to help you succeed independently
                            </p>
                        </motion.div>

                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                            {resources.map((resource, index) => (
                                <motion.div
                                    key={index}
                                    initial={{ opacity: 0, y: 20 }}
                                    whileInView={{ opacity: 1, y: 0 }}
                                    viewport={{ once: true }}
                                    transition={{ duration: 0.6, delay: index * 0.1 }}
                                    className="bg-neutral-100 border-2 border-gold-400 rounded-lg p-6 hover:shadow-lg transition-all duration-300"
                                >
                                    <div className="text-4xl mb-4">{resource.icon}</div>
                                    <h3 className="text-lg font-semibold text-neutral-900 mb-2">{resource.title}</h3>
                                    <p className="text-neutral-600 text-sm mb-4">{resource.description}</p>
                                    <ul className="space-y-2">
                                        {resource.items.map((item, idx) => (
                                            <li key={idx} className="flex items-center space-x-2">
                                                <span className="text-gold-600 text-xs">•</span>
                                                <span className="text-xs text-neutral-900">{item}</span>
                                            </li>
                                        ))}
                                    </ul>
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
                                Ready to Get the Support You Need?
                            </h2>
                            <p className="text-lg text-neutral-600 max-w-2xl mx-auto">
                                Contact our team to discuss your support requirements and find the perfect plan for your organization.
                            </p>
                            <div className="flex flex-col sm:flex-row gap-4 justify-center">
                                <Button
                                    className="bg-accent hover:bg-accent text-white border-2 border-gold-400 px-8 py-3"
                                    onClick={() => router.push('/contact')}
                                >
                                    Contact Support
                                </Button>
                                <Button
                                    className="bg-neutral-100 hover:bg-primary-background text-neutral-900 border-2 border-gold-400 px-8 py-3"
                                    onClick={() => router.push('/enterprise/pricing')}
                                >
                                    View Pricing
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

export default EnterpriseSupportPage;
