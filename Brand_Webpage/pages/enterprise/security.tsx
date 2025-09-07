import React from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';
import BrandHeader from '../../components/BrandHeader';
import Footer from '../../components/Footer';
import Button from '../../components/Button';

const EnterpriseSecurityPage: React.FC = () => {
    const router = useRouter();

    const securityFeatures = [
        {
            icon: "🔒",
            title: "End-to-End Encryption",
            description: "All data encrypted in transit and at rest with AES-256 encryption",
            details: [
                "256-bit AES encryption for data at rest",
                "TLS 1.3 for data in transit",
                "Hardware security modules (HSM)",
                "Key rotation every 90 days"
            ]
        },
        {
            icon: "🛡️",
            title: "Zero Trust Architecture",
            description: "Never trust, always verify - comprehensive identity and access management",
            details: [
                "Multi-factor authentication (MFA)",
                "Role-based access control (RBAC)",
                "Continuous verification",
                "Network segmentation"
            ]
        },
        {
            icon: "📋",
            title: "Compliance Ready",
            description: "Built to meet the strictest regulatory requirements",
            details: [
                "SOC 2 Type II certified",
                "GDPR & CCPA compliant",
                "HIPAA ready configuration",
                "ISO 27001 aligned"
            ]
        },
        {
            icon: "🔍",
            title: "Continuous Monitoring",
            description: "24/7 security monitoring with AI-powered threat detection",
            details: [
                "Real-time threat detection",
                "Behavioral anomaly analysis",
                "Security incident response",
                "Automated threat mitigation"
            ]
        },
        {
            icon: "💾",
            title: "Data Protection",
            description: "Comprehensive data lifecycle management and protection",
            details: [
                "Automated data classification",
                "Data loss prevention (DLP)",
                "Secure backup & recovery",
                "Right to be forgotten compliance"
            ]
        },
        {
            icon: "🔐",
            title: "Identity & Access",
            description: "Enterprise-grade identity management with seamless integration",
            details: [
                "Single Sign-On (SSO) support",
                "SAML 2.0 & OpenID Connect",
                "Directory service integration",
                "Privileged access management"
            ]
        }
    ];

    const certifications = [
        { name: "SOC 2 Type II", icon: "🏆", status: "Certified" },
        { name: "ISO 27001", icon: "🌐", status: "Aligned" },
        { name: "GDPR", icon: "🇪🇺", status: "Compliant" },
        { name: "CCPA", icon: "🏛️", status: "Compliant" },
        { name: "HIPAA", icon: "🏥", status: "Ready" },
        { name: "FedRAMP", icon: "🏛️", status: "In Progress" }
    ];

    const securityPillars = [
        {
            title: "Confidentiality",
            description: "Your data remains private and accessible only to authorized individuals",
            icon: "🤫"
        },
        {
            title: "Integrity",
            description: "Data accuracy and completeness is maintained throughout its lifecycle",
            icon: "✅"
        },
        {
            title: "Availability",
            description: "99.9% uptime SLA with redundant systems and disaster recovery",
            icon: "🚀"
        },
        {
            title: "Accountability",
            description: "Complete audit trails and transparent security reporting",
            icon: "📊"
        }
    ];

    return (
        <>
            <Head>
                <title>Enterprise Security - DharmaMind</title>
                <meta name="description" content="Enterprise-grade security with bank-level protection for your spiritual wisdom and business data" />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <link rel="icon" href="/favicon.ico" />
            </Head>

            <div className="min-h-screen bg-section-light">
                {/* Enterprise Brand Header */}
                <BrandHeader
                    isEnterprise={true}
                    breadcrumbs={[
                        { label: 'Home', href: '/' },
                        { label: 'Enterprise', href: '/enterprise' },
                        { label: 'Security', href: '/enterprise/security' }
                    ]}
                />

                {/* Hero Section */}
                <div className="bg-brand-primary border-b-2 border-brand-accent py-16">
                    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.8 }}
                            className="text-center"
                        >
                            <h1 className="text-4xl lg:text-5xl font-bold text-primary mb-6">
                                Enterprise <span className="text-brand-accent" style={{ borderBottom: '4px solid var(--brand-accent)' }}>Security</span>
                            </h1>
                            <p className="text-xl text-secondary max-w-3xl mx-auto leading-relaxed">
                                Bank-grade security protecting your spiritual wisdom and business data with the highest standards of privacy and compliance.
                            </p>
                        </motion.div>
                    </div>
                </div>

                {/* Security Pillars */}
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        className="text-center mb-12"
                    >
                        <h2 className="text-3xl font-bold text-primary mb-4 border-b-4 border-brand-accent inline-block pb-2">
                            Security Foundation
                        </h2>
                        <p className="text-lg text-secondary">
                            Built on four fundamental pillars of information security
                        </p>
                    </motion.div>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
                        {securityPillars.map((pillar, index) => (
                            <motion.div
                                key={index}
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ duration: 0.6, delay: index * 0.1 }}
                                className="bg-brand-primary border-2 border-brand-accent rounded-lg p-6 text-center"
                            >
                                <div className="text-4xl mb-4">{pillar.icon}</div>
                                <h3 className="text-xl font-semibold text-primary mb-3">{pillar.title}</h3>
                                <p className="text-secondary">{pillar.description}</p>
                            </motion.div>
                        ))}
                    </div>
                </div>

                {/* Security Features */}
                <div className="bg-primary-background border-t-2 border-brand-accent py-16">
                    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            className="text-center mb-12"
                        >
                            <h2 className="text-3xl font-bold text-primary mb-4 border-b-4 border-brand-accent inline-block pb-2">
                                Comprehensive Security Features
                            </h2>
                            <p className="text-lg text-secondary">
                                Multi-layered security approach protecting every aspect of your data
                            </p>
                        </motion.div>

                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                            {securityFeatures.map((feature, index) => (
                                <motion.div
                                    key={index}
                                    initial={{ opacity: 0, y: 20 }}
                                    whileInView={{ opacity: 1, y: 0 }}
                                    viewport={{ once: true }}
                                    transition={{ duration: 0.6, delay: index * 0.1 }}
                                    className="bg-brand-primary border-2 border-brand-accent rounded-lg p-6 hover:shadow-lg transition-all duration-300"
                                >
                                    <div className="text-4xl mb-4">{feature.icon}</div>
                                    <h3 className="text-xl font-semibold text-primary mb-3">{feature.title}</h3>
                                    <p className="text-secondary mb-4">{feature.description}</p>
                                    <ul className="space-y-2">
                                        {feature.details.map((detail, idx) => (
                                            <li key={idx} className="flex items-center space-x-2">
                                                <span className="text-brand-accent text-sm">✓</span>
                                                <span className="text-sm text-primary">{detail}</span>
                                            </li>
                                        ))}
                                    </ul>
                                </motion.div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Certifications */}
                <div className="bg-brand-primary border-t-2 border-brand-accent py-16">
                    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            className="text-center mb-12"
                        >
                            <h2 className="text-3xl font-bold text-primary mb-4 border-b-4 border-brand-accent inline-block pb-2">
                                Compliance & Certifications
                            </h2>
                            <p className="text-lg text-secondary">
                                Meeting the highest standards for security and privacy
                            </p>
                        </motion.div>

                        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-6">
                            {certifications.map((cert, index) => (
                                <motion.div
                                    key={index}
                                    initial={{ opacity: 0, scale: 0.8 }}
                                    whileInView={{ opacity: 1, scale: 1 }}
                                    viewport={{ once: true }}
                                    transition={{ duration: 0.5, delay: index * 0.1 }}
                                    className="bg-primary-background border-2 border-brand-accent rounded-lg p-4 text-center hover:shadow-lg transition-all duration-300"
                                >
                                    <div className="text-3xl mb-2">{cert.icon}</div>
                                    <h3 className="text-sm font-semibold text-primary mb-1">{cert.name}</h3>
                                    <span className="text-xs text-brand-accent font-medium">{cert.status}</span>
                                </motion.div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Security Documentation */}
                <div className="bg-primary-background border-t-2 border-brand-accent py-16">
                    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            className="text-center mb-12"
                        >
                            <h2 className="text-3xl font-bold text-primary mb-4 border-b-4 border-brand-accent inline-block pb-2">
                                Security Documentation
                            </h2>
                            <p className="text-lg text-secondary">
                                Transparent security practices and detailed documentation
                            </p>
                        </motion.div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <motion.div
                                initial={{ opacity: 0, x: -20 }}
                                whileInView={{ opacity: 1, x: 0 }}
                                viewport={{ once: true }}
                                className="bg-brand-primary border-2 border-brand-accent rounded-lg p-6"
                            >
                                <h3 className="text-lg font-semibold text-primary mb-3">🔐 Security Whitepaper</h3>
                                <p className="text-secondary mb-4">Comprehensive overview of our security architecture and practices</p>
                                <Button className="bg-accent hover:bg-accent text-white border-2 border-brand-accent px-4 py-2 text-sm">
                                    Download PDF
                                </Button>
                            </motion.div>

                            <motion.div
                                initial={{ opacity: 0, x: 20 }}
                                whileInView={{ opacity: 1, x: 0 }}
                                viewport={{ once: true }}
                                className="bg-brand-primary border-2 border-brand-accent rounded-lg p-6"
                            >
                                <h3 className="text-lg font-semibold text-primary mb-3">📋 Compliance Report</h3>
                                <p className="text-secondary mb-4">Latest audit results and compliance status updates</p>
                                <Button className="bg-accent hover:bg-accent text-white border-2 border-brand-accent px-4 py-2 text-sm">
                                    View Report
                                </Button>
                            </motion.div>
                        </div>
                    </div>
                </div>

                {/* CTA Section */}
                <div className="bg-brand-primary border-t-2 border-brand-accent py-16">
                    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            className="space-y-6"
                        >
                            <h2 className="text-3xl font-bold text-primary">
                                Questions About Our Security?
                            </h2>
                            <p className="text-lg text-secondary max-w-2xl mx-auto">
                                Our security team is ready to answer your questions and provide additional documentation as needed.
                            </p>
                            <div className="flex flex-col sm:flex-row gap-4 justify-center">
                                <Button
                                    className="bg-accent hover:bg-accent text-white border-2 border-brand-accent px-8 py-3"
                                    onClick={() => router.push('/contact')}
                                >
                                    Contact Security Team
                                </Button>
                                <Button
                                    className="bg-brand-primary hover:bg-primary-background text-primary border-2 border-brand-accent px-8 py-3"
                                    onClick={() => router.push('/enterprise/support')}
                                >
                                    Enterprise Support
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

export default EnterpriseSecurityPage;
