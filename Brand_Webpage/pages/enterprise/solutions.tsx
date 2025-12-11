import React from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';
import BrandHeader from '../../components/BrandHeader';
import Footer from '../../components/Footer';
import Button from '../../components/Button';

const EnterpriseSolutionsPage: React.FC = () => {
    const router = useRouter();

    const solutions = [
        {
            icon: "🧠",
            title: "AI Ethics Advisory",
            description: "Integrated ethical decision-making framework for complex business choices",
            features: [
                "Real-time ethics assessment",
                "Stakeholder impact analysis",
                "Regulatory compliance guidance",
                "Cultural sensitivity checks"
            ]
        },
        {
            icon: "🌱",
            title: "Employee Wellbeing Platform",
            description: "Comprehensive mental health and spiritual wellness solution",
            features: [
                "Personalized meditation programs",
                "Stress management tools",
                "Work-life balance coaching",
                "Team wellness analytics"
            ]
        },
        {
            icon: "🎯",
            title: "Purpose-Driven Leadership",
            description: "Transform leadership through dharmic principles and conscious decision-making",
            features: [
                "Leadership assessment tools",
                "Values alignment workshops",
                "Executive coaching programs",
                "Cultural transformation guidance"
            ]
        },
        {
            icon: "📊",
            title: "Organizational Insights",
            description: "Deep analytics on team consciousness, engagement, and spiritual alignment",
            features: [
                "Consciousness metrics dashboard",
                "Team alignment scoring",
                "Cultural health indicators",
                "Predictive wellness analytics"
            ]
        },
        {
            icon: "🔗",
            title: "Integration Hub",
            description: "Seamlessly connect with your existing HR, CRM, and business intelligence tools",
            features: [
                "Single sign-on (SSO) support",
                "API-first architecture",
                "Custom workflow automation",
                "Real-time data synchronization"
            ]
        },
        {
            icon: "🛡️",
            title: "Enterprise Security",
            description: "Bank-grade security with spiritual wisdom protection protocols",
            features: [
                "End-to-end encryption",
                "SOC 2 Type II compliance",
                "GDPR & CCPA compliance",
                "Zero-trust architecture"
            ]
        }
    ];

    const useCases = [
        {
            industry: "Healthcare",
            challenge: "Ethical decision-making in patient care",
            solution: "AI-powered ethics guidance for medical professionals",
            result: "40% improvement in decision confidence"
        },
        {
            industry: "Financial Services",
            challenge: "Balancing profit with social responsibility",
            solution: "Dharmic principles integration in investment decisions",
            result: "25% increase in ESG compliance scores"
        },
        {
            industry: "Technology",
            challenge: "Employee burnout and lack of purpose",
            solution: "Comprehensive wellbeing and purpose alignment platform",
            result: "50% reduction in turnover, 35% increase in engagement"
        }
    ];

    return (
        <>
            <Head>
                <title>Enterprise Solutions - DharmaMind</title>
                <meta name="description" content="Comprehensive enterprise solutions for ethical AI, employee wellbeing, and purpose-driven leadership" />
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
                        { label: 'Solutions', href: '/enterprise/solutions' }
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
                                Enterprise <span className="text-gold-600" style={{ borderBottom: '4px solid var(--brand-accent)' }}>Solutions</span>
                            </h1>
                            <p className="text-xl text-neutral-600 max-w-3xl mx-auto leading-relaxed">
                                Transform your organization with comprehensive AI-powered solutions that integrate ancient wisdom with modern business needs.
                            </p>
                        </motion.div>
                    </div>
                </div>

                {/* Solutions Grid */}
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                        {solutions.map((solution, index) => (
                            <motion.div
                                key={index}
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ duration: 0.6, delay: index * 0.1 }}
                                className="bg-neutral-100 border-2 border-gold-400 rounded-lg p-6 hover:shadow-lg transition-all duration-300"
                            >
                                <div className="text-4xl mb-4">{solution.icon}</div>
                                <h3 className="text-xl font-semibold text-neutral-900 mb-3">{solution.title}</h3>
                                <p className="text-neutral-600 mb-4">{solution.description}</p>
                                <ul className="space-y-2">
                                    {solution.features.map((feature, idx) => (
                                        <li key={idx} className="flex items-center space-x-2">
                                            <span className="text-gold-600 text-sm">✓</span>
                                            <span className="text-neutral-900 text-sm">{feature}</span>
                                        </li>
                                    ))}
                                </ul>
                            </motion.div>
                        ))}
                    </div>
                </div>

                {/* Use Cases */}
                <div className="bg-neutral-50 border-t-2 border-gold-400 py-16">
                    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            className="text-center mb-12"
                        >
                            <h2 className="text-3xl font-bold text-neutral-900 mb-4 inline-block pb-2" style={{ borderBottom: '4px solid var(--brand-accent)' }}>
                                Industry Use Cases
                            </h2>
                            <p className="text-lg text-neutral-600">
                                See how organizations across industries are transforming with DharmaMind
                            </p>
                        </motion.div>

                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                            {useCases.map((useCase, index) => (
                                <motion.div
                                    key={index}
                                    initial={{ opacity: 0, x: -20 }}
                                    whileInView={{ opacity: 1, x: 0 }}
                                    viewport={{ once: true }}
                                    transition={{ duration: 0.6, delay: index * 0.2 }}
                                    className="bg-neutral-100 border-2 border-gold-400 rounded-lg p-6"
                                >
                                    <h3 className="text-lg font-semibold text-gold-600 mb-3">{useCase.industry}</h3>
                                    <div className="space-y-3">
                                        <div>
                                            <h4 className="text-sm font-medium text-neutral-900">Challenge:</h4>
                                            <p className="text-sm text-neutral-600">{useCase.challenge}</p>
                                        </div>
                                        <div>
                                            <h4 className="text-sm font-medium text-neutral-900">Solution:</h4>
                                            <p className="text-sm text-neutral-600">{useCase.solution}</p>
                                        </div>
                                        <div>
                                            <h4 className="text-sm font-medium text-neutral-900">Result:</h4>
                                            <p className="text-sm font-semibold text-gold-600">{useCase.result}</p>
                                        </div>
                                    </div>
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
                                Schedule a personalized demo to see how our enterprise solutions can address your unique challenges.
                            </p>
                            <div className="flex flex-col sm:flex-row gap-4 justify-center">
                                <Button
                                    variant="enterprise"
                                    size="lg"
                                    onClick={() => router.push('/contact')}
                                >
                                    Schedule Demo
                                </Button>
                                <Button
                                    variant="outline"
                                    size="lg"
                                    onClick={() => router.push('/enterprise/security')}
                                >
                                    View Security Details
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

export default EnterpriseSolutionsPage;
