import { Layout, Section, Button, Card } from '../components';
import { siteConfig } from '../config/site.config';

export default function Products() {
    return (
        <Layout
            title="Products"
            description="AI-powered tools for spiritual growth, from personal guidance to enterprise solutions."
        >
            {/* Hero Section */}
            <Section background="light" padding="xl">
                <div className="max-w-3xl">
                    <h1 className="text-4xl md:text-6xl font-bold text-neutral-900 mb-6">
                        Products
                    </h1>
                    <p className="text-xl text-neutral-600 leading-relaxed">
                        From personal spiritual guidance to enterprise solutions, our products
                        are designed to serve humanity's highest aspirations while respecting
                        all wisdom traditions.
                    </p>
                </div>
            </Section>

            {/* Products Grid */}
            <Section>
                <div className="space-y-24">
                    {siteConfig.products.featured.map((product, index) => (
                        <div
                            key={product.id}
                            id={product.id}
                            className={`grid md:grid-cols-2 gap-12 items-center ${index % 2 === 1 ? 'md:flex-row-reverse' : ''
                                }`}
                        >
                            {/* Content */}
                            <div className={index % 2 === 1 ? 'md:order-2' : ''}>
                                <span className="text-5xl mb-6 block">{product.icon}</span>
                                <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-3">
                                    {product.name}
                                </h2>
                                <p className="text-lg text-neutral-600 mb-2">{product.tagline}</p>
                                <p className="text-neutral-600 mb-6">{product.description}</p>

                                <ul className="space-y-3 mb-8">
                                    {product.features.map((feature) => (
                                        <li key={feature} className="flex items-center text-neutral-700">
                                            <span className="text-gold-600 mr-3">✓</span>
                                            {feature}
                                        </li>
                                    ))}
                                </ul>

                                <Button href={product.href} size="lg">
                                    {product.cta}
                                </Button>
                            </div>

                            {/* Visual */}
                            <div className={`${index % 2 === 1 ? 'md:order-1' : ''}`}>
                                <div className="bg-gradient-to-br from-neutral-100 to-neutral-200 rounded-2xl aspect-square flex items-center justify-center">
                                    <span className="text-8xl opacity-50">{product.icon}</span>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </Section>

            {/* Use Cases */}
            <Section background="light">
                <div className="text-center mb-12">
                    <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-4">
                        Use Cases
                    </h2>
                    <p className="text-neutral-600 max-w-2xl mx-auto">
                        Our products serve a wide range of needs across personal growth,
                        education, healthcare, and cultural preservation.
                    </p>
                </div>

                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {siteConfig.products.useCases.map((useCase) => (
                        <div
                            key={useCase.title}
                            className="p-6 rounded-2xl border border-neutral-200 bg-neutral-100 hover:shadow-lg transition-shadow"
                        >
                            <span className="text-4xl mb-4 block">{useCase.icon}</span>
                            <h3 className="text-lg font-semibold text-neutral-900 mb-2">
                                {useCase.title}
                            </h3>
                            <p className="text-neutral-600">{useCase.description}</p>
                        </div>
                    ))}
                </div>
            </Section>

            {/* Pricing Preview */}
            <Section>
                <div className="bg-neutral-100 rounded-3xl border border-neutral-200 p-8 md:p-12">
                    <div className="grid md:grid-cols-3 gap-8">
                        {/* Free Tier */}
                        <div className="p-6 rounded-2xl border border-neutral-200">
                            <h3 className="text-xl font-semibold text-neutral-900 mb-2">Free</h3>
                            <p className="text-3xl font-bold text-neutral-900 mb-4">$0<span className="text-base font-normal text-neutral-500">/month</span></p>
                            <p className="text-neutral-600 mb-6">Perfect for exploring spiritual AI</p>
                            <ul className="space-y-3 mb-6">
                                <li className="flex items-center text-sm text-neutral-600">
                                    <span className="text-gold-600 mr-2">✓</span>
                                    50 conversations/month
                                </li>
                                <li className="flex items-center text-sm text-neutral-600">
                                    <span className="text-gold-600 mr-2">✓</span>
                                    Basic guidance
                                </li>
                                <li className="flex items-center text-sm text-neutral-600">
                                    <span className="text-gold-600 mr-2">✓</span>
                                    Community support
                                </li>
                            </ul>
                            <Button href="https://chat.dharmamind.ai" variant="outline" className="w-full">
                                Get Started
                            </Button>
                        </div>

                        {/* Pro Tier */}
                        <div className="p-6 rounded-2xl border-2 border-gold-400 relative">
                            <span className="absolute -top-3 left-6 px-3 py-1 bg-gold-600 text-white text-xs rounded-full">
                                Most Popular
                            </span>
                            <h3 className="text-xl font-semibold text-neutral-900 mb-2">Pro</h3>
                            <p className="text-3xl font-bold text-neutral-900 mb-4">$19<span className="text-base font-normal text-neutral-500">/month</span></p>
                            <p className="text-neutral-600 mb-6">For dedicated practitioners</p>
                            <ul className="space-y-3 mb-6">
                                <li className="flex items-center text-sm text-neutral-600">
                                    <span className="text-gold-600 mr-2">✓</span>
                                    Unlimited conversations
                                </li>
                                <li className="flex items-center text-sm text-neutral-600">
                                    <span className="text-gold-600 mr-2">✓</span>
                                    Advanced guidance
                                </li>
                                <li className="flex items-center text-sm text-neutral-600">
                                    <span className="text-gold-600 mr-2">✓</span>
                                    Priority support
                                </li>
                                <li className="flex items-center text-sm text-neutral-600">
                                    <span className="text-gold-600 mr-2">✓</span>
                                    Progress tracking
                                </li>
                            </ul>
                            <Button href="https://chat.dharmamind.ai/pro" className="w-full">
                                Start Free Trial
                            </Button>
                        </div>

                        {/* Enterprise Tier */}
                        <div className="p-6 rounded-2xl border border-neutral-200">
                            <h3 className="text-xl font-semibold text-neutral-900 mb-2">Enterprise</h3>
                            <p className="text-3xl font-bold text-neutral-900 mb-4">Custom</p>
                            <p className="text-neutral-600 mb-6">For organizations at scale</p>
                            <ul className="space-y-3 mb-6">
                                <li className="flex items-center text-sm text-neutral-600">
                                    <span className="text-gold-600 mr-2">✓</span>
                                    Custom deployment
                                </li>
                                <li className="flex items-center text-sm text-neutral-600">
                                    <span className="text-gold-600 mr-2">✓</span>
                                    Dedicated support
                                </li>
                                <li className="flex items-center text-sm text-neutral-600">
                                    <span className="text-gold-600 mr-2">✓</span>
                                    SLA guarantees
                                </li>
                                <li className="flex items-center text-sm text-neutral-600">
                                    <span className="text-gold-600 mr-2">✓</span>
                                    Training included
                                </li>
                            </ul>
                            <Button href={`mailto:${siteConfig.company.email}`} variant="outline" className="w-full">
                                Contact Sales
                            </Button>
                        </div>
                    </div>
                </div>
            </Section>

            {/* API Section */}
            <Section background="dark">
                <div className="grid md:grid-cols-2 gap-12 items-center">
                    <div>
                        <span className="text-sm font-medium text-neutral-400 uppercase tracking-wider mb-4 block">
                            For Developers
                        </span>
                        <h2 className="text-3xl md:text-4xl font-bold mb-6">
                            Build with DharmaMind API
                        </h2>
                        <p className="text-neutral-300 mb-6">
                            Integrate spiritual AI into your applications with our powerful REST API.
                            Perfect for wellness apps, meditation platforms, and educational tools.
                        </p>
                        <ul className="space-y-3 mb-8">
                            <li className="flex items-center text-neutral-300">
                                <span className="text-green-400 mr-3">✓</span>
                                Simple REST API
                            </li>
                            <li className="flex items-center text-neutral-300">
                                <span className="text-green-400 mr-3">✓</span>
                                Real-time streaming
                            </li>
                            <li className="flex items-center text-neutral-300">
                                <span className="text-green-400 mr-3">✓</span>
                                Multi-language SDKs
                            </li>
                            <li className="flex items-center text-neutral-300">
                                <span className="text-green-400 mr-3">✓</span>
                                Comprehensive docs
                            </li>
                        </ul>
                        <div className="flex gap-4">
                            <Button href="/docs" variant="secondary">
                                View Documentation
                            </Button>
                            <Button
                                href="/products#api"
                                variant="outline"
                                className="border-white text-white hover:bg-neutral-100 hover:text-neutral-900"
                            >
                                Get API Key
                            </Button>
                        </div>
                    </div>

                    <div className="bg-neutral-800 rounded-2xl p-6 font-mono text-sm">
                        <div className="text-neutral-400 mb-2"># Example API call</div>
                        <div className="text-green-400">curl</div>
                        <div className="text-white ml-4">-X POST https://api.dharmamind.ai/v1/chat \</div>
                        <div className="text-white ml-4">-H "Authorization: Bearer $API_KEY" \</div>
                        <div className="text-white ml-4">-d '{`{`}</div>
                        <div className="text-white ml-8">"message": "What is karma?",</div>
                        <div className="text-white ml-8">"context": "beginner"</div>
                        <div className="text-white ml-4">{`}`}'</div>
                    </div>
                </div>
            </Section>

            {/* CTA */}
            <Section>
                <div className="text-center max-w-2xl mx-auto">
                    <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-6">
                        Ready to Begin?
                    </h2>
                    <p className="text-xl text-neutral-600 mb-8">
                        Start your spiritual journey with AI that understands the depth of ancient wisdom.
                    </p>
                    <div className="flex flex-col sm:flex-row gap-4 justify-center">
                        <Button href="https://chat.dharmamind.ai" size="lg" external>
                            Try DharmaMind Free
                        </Button>
                        <Button href={`mailto:${siteConfig.company.email}`} variant="outline" size="lg">
                            Talk to Sales
                        </Button>
                    </div>
                </div>
            </Section>
        </Layout>
    );
}
