import { Layout, Section } from '../components';

export default function SecurityPage() {
    const securityFeatures = [
        {
            icon: '🔐',
            title: 'End-to-End Encryption',
            description: 'All conversations are encrypted with AES-256 encryption, ensuring your spiritual discussions remain private and secure.'
        },
        {
            icon: '🛡️',
            title: 'Enterprise Security',
            description: 'SOC 2 Type II compliant infrastructure with regular penetration testing and security audits.'
        },
        {
            icon: '🔒',
            title: 'Zero-Knowledge Architecture',
            description: 'We use zero-knowledge principles - your personal conversations cannot be accessed by our team.'
        },
        {
            icon: '🌐',
            title: 'GDPR & CCPA Compliant',
            description: 'Full compliance with global privacy regulations including GDPR, CCPA, and other data protection laws.'
        },
        {
            icon: '🔑',
            title: 'Multi-Factor Authentication',
            description: 'Secure your account with MFA using authenticator apps, SMS, or hardware security keys.'
        },
        {
            icon: '🏛️',
            title: 'Data Sovereignty',
            description: 'Choose where your data is stored with region-specific data centers and on-premise deployment options.'
        }
    ];

    const certifications = [
        { name: 'SOC 2 Type II', status: 'Certified', icon: '✅' },
        { name: 'ISO 27001', status: 'In Progress', icon: '🔄' },
        { name: 'HIPAA', status: 'Compliant', icon: '✅' },
        { name: 'PCI DSS', status: 'Certified', icon: '✅' }
    ];

    return (
        <Layout title="Security" description="Learn about DharmaMind's security measures, compliance, and data protection practices">
            {/* Hero Section */}
            <Section background="light" padding="xl">
                <div className="max-w-4xl mx-auto text-center">
                    <div className="w-16 h-16 mx-auto mb-6 bg-gold-100 rounded-full flex items-center justify-center">
                        <span className="text-3xl">🛡️</span>
                    </div>
                    <h1 className="text-4xl md:text-5xl font-bold text-neutral-900 mb-4">
                        Enterprise-Grade Security
                    </h1>
                    <p className="text-xl text-neutral-600 mb-8 max-w-2xl mx-auto">
                        Your spiritual journey deserves the highest level of privacy and protection.
                        Learn how we safeguard your conversations and data.
                    </p>
                    <div className="inline-flex items-center px-6 py-3 bg-white rounded-lg shadow-sm border border-neutral-200">
                        <span className="text-gold-600 mr-2">✅</span>
                        <span className="font-medium text-neutral-900">Zero security incidents in our history</span>
                    </div>
                </div>
            </Section>

            {/* Security Features */}
            <Section background="white" padding="xl">
                <div className="max-w-6xl mx-auto">
                    <h2 className="text-3xl font-bold text-neutral-900 mb-12 text-center">
                        Built with Security First
                    </h2>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                        {securityFeatures.map((feature, index) => (
                            <div key={index} className="bg-neutral-50 rounded-lg border border-neutral-200 p-6 hover:shadow-md transition-shadow">
                                <div className="w-12 h-12 mb-4 bg-gold-100 rounded-lg flex items-center justify-center">
                                    <span className="text-2xl">{feature.icon}</span>
                                </div>
                                <h3 className="text-lg font-semibold text-neutral-900 mb-3">{feature.title}</h3>
                                <p className="text-neutral-600 leading-relaxed">{feature.description}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </Section>

            {/* Technical Details */}
            <Section background="light" padding="xl">
                <div className="max-w-6xl mx-auto">
                    <h2 className="text-3xl font-bold text-neutral-900 mb-12 text-center">
                        Technical Security Details
                    </h2>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
                        {/* Data Protection */}
                        <div>
                            <h3 className="text-xl font-semibold text-neutral-900 mb-6">🔒 Data Protection</h3>
                            <div className="space-y-4">
                                <div className="flex items-start space-x-3">
                                    <div className="w-2 h-2 bg-gold-600 rounded-full mt-2"></div>
                                    <div>
                                        <div className="font-medium text-neutral-900">Encryption at Rest</div>
                                        <div className="text-neutral-600 text-sm">AES-256 encryption for all stored data</div>
                                    </div>
                                </div>
                                <div className="flex items-start space-x-3">
                                    <div className="w-2 h-2 bg-gold-600 rounded-full mt-2"></div>
                                    <div>
                                        <div className="font-medium text-neutral-900">Encryption in Transit</div>
                                        <div className="text-neutral-600 text-sm">TLS 1.3 for all data transmission</div>
                                    </div>
                                </div>
                                <div className="flex items-start space-x-3">
                                    <div className="w-2 h-2 bg-gold-600 rounded-full mt-2"></div>
                                    <div>
                                        <div className="font-medium text-neutral-900">Key Management</div>
                                        <div className="text-neutral-600 text-sm">Hardware Security Modules (HSMs)</div>
                                    </div>
                                </div>
                                <div className="flex items-start space-x-3">
                                    <div className="w-2 h-2 bg-gold-600 rounded-full mt-2"></div>
                                    <div>
                                        <div className="font-medium text-neutral-900">Data Retention</div>
                                        <div className="text-neutral-600 text-sm">Configurable retention policies</div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Infrastructure Security */}
                        <div>
                            <h3 className="text-xl font-semibold text-neutral-900 mb-6">🏗️ Infrastructure Security</h3>
                            <div className="space-y-4">
                                <div className="flex items-start space-x-3">
                                    <div className="w-2 h-2 bg-gold-600 rounded-full mt-2"></div>
                                    <div>
                                        <div className="font-medium text-neutral-900">Network Security</div>
                                        <div className="text-neutral-600 text-sm">VPC isolation, firewalls, DDoS protection</div>
                                    </div>
                                </div>
                                <div className="flex items-start space-x-3">
                                    <div className="w-2 h-2 bg-gold-600 rounded-full mt-2"></div>
                                    <div>
                                        <div className="font-medium text-neutral-900">Access Controls</div>
                                        <div className="text-neutral-600 text-sm">Role-based access with least privilege</div>
                                    </div>
                                </div>
                                <div className="flex items-start space-x-3">
                                    <div className="w-2 h-2 bg-gold-600 rounded-full mt-2"></div>
                                    <div>
                                        <div className="font-medium text-neutral-900">Monitoring</div>
                                        <div className="text-neutral-600 text-sm">24/7 security monitoring and alerting</div>
                                    </div>
                                </div>
                                <div className="flex items-start space-x-3">
                                    <div className="w-2 h-2 bg-gold-600 rounded-full mt-2"></div>
                                    <div>
                                        <div className="font-medium text-neutral-900">Incident Response</div>
                                        <div className="text-neutral-600 text-sm">Documented procedures and rapid response</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </Section>

            {/* Certifications */}
            <Section background="white" padding="xl">
                <div className="max-w-4xl mx-auto">
                    <h2 className="text-3xl font-bold text-neutral-900 mb-12 text-center">
                        Compliance & Certifications
                    </h2>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                        {certifications.map((cert, index) => (
                            <div key={index} className="text-center p-6 bg-neutral-50 rounded-lg border border-neutral-200">
                                <div className="text-3xl mb-3">{cert.icon}</div>
                                <div className="font-semibold text-neutral-900">{cert.name}</div>
                                <div className="text-sm text-neutral-500">{cert.status}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </Section>

            {/* CTA Section */}
            <Section background="dark" padding="xl">
                <div className="max-w-4xl mx-auto text-center">
                    <h2 className="text-3xl font-bold mb-4">
                        Questions About Security?
                    </h2>
                    <p className="text-neutral-400 mb-8 max-w-2xl mx-auto">
                        Our security team is happy to discuss our practices and answer any questions
                        about how we protect your data.
                    </p>
                    <div className="flex flex-col sm:flex-row gap-4 justify-center">
                        <a
                            href="mailto:security@dharmamind.com"
                            className="bg-gold-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-gold-700 transition-colors"
                        >
                            Contact Security Team
                        </a>
                        <a
                            href="/privacy"
                            className="bg-neutral-700 text-white px-6 py-3 rounded-lg font-medium hover:bg-neutral-600 transition-colors"
                        >
                            Read Privacy Policy
                        </a>
                    </div>
                </div>
            </Section>
        </Layout>
    );
}
