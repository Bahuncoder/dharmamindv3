import { Layout, Section, Button } from '../components';
import { siteConfig } from '../config/site.config';
import Image from 'next/image';

export default function Press() {
    return (
        <Layout
            title="Press"
            description="News, announcements, and media resources from DharmaMind."
        >
            {/* Hero Section */}
            <Section background="light" padding="xl">
                <div className="max-w-3xl">
                    <h1 className="text-4xl md:text-6xl font-bold text-neutral-900 mb-6">
                        Press & Media
                    </h1>
                    <p className="text-xl text-neutral-600 leading-relaxed">
                        The latest news, announcements, and media resources from {siteConfig.company.name}.
                        For press inquiries, contact us at{' '}
                        <a href={`mailto:${siteConfig.company.pressEmail}`} className="text-neutral-900 hover:underline">
                            {siteConfig.company.pressEmail}
                        </a>
                    </p>
                </div>
            </Section>

            {/* Press Kit */}
            <Section>
                <div className="grid md:grid-cols-2 gap-12 items-center">
                    <div>
                        <h2 className="text-3xl font-bold text-neutral-900 mb-4">
                            Press Kit
                        </h2>
                        <p className="text-neutral-600 mb-6">
                            Download our official press kit including logos, brand guidelines,
                            executive photos, and company fact sheet.
                        </p>
                        <div className="space-y-4">
                            <Button href="/press-kit.zip" variant="primary" size="lg">
                                Download Press Kit
                            </Button>
                            <p className="text-sm text-neutral-500">
                                Includes: Logos (PNG, SVG), Brand Guidelines, Executive Bios & Photos
                            </p>
                        </div>
                    </div>
                    <div className="bg-neutral-100 rounded-2xl p-8">
                        <h3 className="font-semibold text-neutral-900 mb-4">Quick Facts</h3>
                        <dl className="space-y-3">
                            <div className="flex justify-between">
                                <dt className="text-neutral-600">Founded</dt>
                                <dd className="font-medium text-neutral-900">{siteConfig.company.founded}</dd>
                            </div>
                            <div className="flex justify-between">
                                <dt className="text-neutral-600">Headquarters</dt>
                                <dd className="font-medium text-neutral-900">{siteConfig.company.headquarters}</dd>
                            </div>
                            <div className="flex justify-between">
                                <dt className="text-neutral-600">Team Size</dt>
                                <dd className="font-medium text-neutral-900">{siteConfig.stats.company.find(s => s.label === 'Team Members')?.value}</dd>
                            </div>
                            <div className="flex justify-between">
                                <dt className="text-neutral-600">Funding</dt>
                                <dd className="font-medium text-neutral-900">{siteConfig.stats.company.find(s => s.label === 'Funding Raised')?.value}</dd>
                            </div>
                        </dl>
                    </div>
                </div>
            </Section>

            {/* Recent News */}
            <Section background="light">
                <div className="mb-12">
                    <h2 className="text-3xl font-bold text-neutral-900 mb-4">Recent News</h2>
                    <p className="text-neutral-600">
                        The latest announcements and coverage about {siteConfig.company.name}.
                    </p>
                </div>

                <div className="space-y-6">
                    {siteConfig.press.coverage.map((item, index) => (
                        <a
                            key={index}
                            href={item.link}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="block bg-neutral-100 rounded-xl p-6 border border-neutral-200 hover:shadow-lg hover:border-gold-400/30 transition-all group"
                        >
                            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                                <div className="flex-grow">
                                    <div className="flex items-center gap-3 mb-2">
                                        <span className="text-sm font-medium text-neutral-900">{item.outlet}</span>
                                        <span className="text-neutral-300">•</span>
                                        <span className="text-sm text-neutral-500">{item.date}</span>
                                    </div>
                                    <h3 className="text-lg font-semibold text-neutral-900 group-hover:text-neutral-900 transition-colors">
                                        {item.title}
                                    </h3>
                                </div>
                                <div className="flex-shrink-0">
                                    <span className="inline-flex items-center text-neutral-900 group-hover:translate-x-1 transition-transform">
                                        Read article
                                        <svg className="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                                        </svg>
                                    </span>
                                </div>
                            </div>
                        </a>
                    ))}
                </div>
            </Section>

            {/* Press Releases */}
            <Section>
                <div className="mb-12">
                    <h2 className="text-3xl font-bold text-neutral-900 mb-4">Press Releases</h2>
                </div>

                <div className="grid md:grid-cols-2 gap-6">
                    {[
                        {
                            date: "December 2025",
                            title: "DharmaMind Launches DharmaLLM 2.0 with Enhanced Spiritual Understanding",
                            excerpt: "New model demonstrates unprecedented understanding of cross-cultural spiritual traditions while maintaining ethical guidelines.",
                        },
                        {
                            date: "October 2025",
                            title: "DharmaMind Raises $30M Series B to Expand Global Access",
                            excerpt: "Funding will accelerate international expansion and support for additional languages and spiritual traditions.",
                        },
                        {
                            date: "July 2025",
                            title: "DharmaMind Partners with Leading Universities for AI Ethics Research",
                            excerpt: "Collaboration aims to establish new standards for ethical AI development in spiritual and mental wellness applications.",
                        },
                        {
                            date: "March 2025",
                            title: "DharmaMind Chat Reaches 50 Million Conversations Milestone",
                            excerpt: "Platform growth reflects increasing demand for AI-assisted spiritual guidance and personal development tools.",
                        },
                    ].map((release, index) => (
                        <div
                            key={index}
                            className="p-6 rounded-xl border border-neutral-200 hover:shadow-md transition-shadow"
                        >
                            <span className="text-sm text-neutral-500">{release.date}</span>
                            <h3 className="text-lg font-semibold text-neutral-900 mt-2 mb-3">
                                {release.title}
                            </h3>
                            <p className="text-neutral-600 text-sm mb-4">
                                {release.excerpt}
                            </p>
                            <Button href="#" variant="ghost" size="sm">
                                Read Full Release →
                            </Button>
                        </div>
                    ))}
                </div>
            </Section>

            {/* Brand Assets */}
            <Section background="light">
                <div className="text-center mb-12">
                    <h2 className="text-3xl font-bold text-neutral-900 mb-4">Brand Assets</h2>
                    <p className="text-neutral-600 max-w-2xl mx-auto">
                        Official logos and brand assets for press and partner use.
                        Please follow our brand guidelines when using these assets.
                    </p>
                </div>

                <div className="grid md:grid-cols-3 gap-6">
                    {/* Logo Light */}
                    <div className="bg-neutral-100 rounded-xl p-8 border border-neutral-200 text-center">
                        <div className="h-24 flex items-center justify-center mb-4">
                            <Image
                                src="/logo.jpeg"
                                alt="DharmaMind Logo"
                                width={80}
                                height={80}
                                className="rounded-lg"
                            />
                        </div>
                        <h3 className="font-medium text-neutral-900 mb-2">Primary Logo</h3>
                        <p className="text-sm text-neutral-500 mb-4">For light backgrounds</p>
                        <Button href="/logo.jpeg" variant="outline" size="sm">
                            Download PNG
                        </Button>
                    </div>

                    {/* Logo Dark */}
                    <div className="bg-neutral-800 rounded-xl p-8 text-center">
                        <div className="h-24 flex items-center justify-center mb-4">
                            <Image
                                src="/logo.jpeg"
                                alt="DharmaMind Logo"
                                width={80}
                                height={80}
                                className="rounded-lg"
                            />
                        </div>
                        <h3 className="font-medium text-white mb-2">Logo on Dark</h3>
                        <p className="text-sm text-white/70 mb-4">For dark backgrounds</p>
                        <Button href="/logo.jpeg" variant="secondary" size="sm">
                            Download PNG
                        </Button>
                    </div>

                    {/* Wordmark */}
                    <div className="bg-neutral-100 rounded-xl p-8 border border-neutral-200 text-center">
                        <div className="h-24 flex items-center justify-center mb-4">
                            <span className="text-3xl font-bold text-neutral-900">DharmaMind</span>
                        </div>
                        <h3 className="font-medium text-neutral-900 mb-2">Wordmark</h3>
                        <p className="text-sm text-neutral-500 mb-4">Text-only version</p>
                        <Button href="#" variant="outline" size="sm">
                            Download SVG
                        </Button>
                    </div>
                </div>
            </Section>

            {/* Contact */}
            <Section>
                <div className="bg-neutral-200 border border-neutral-300 rounded-3xl p-8 md:p-12 text-center">
                    <h2 className="text-3xl font-bold text-neutral-900 mb-4">Media Inquiries</h2>
                    <p className="text-neutral-600 mb-6 max-w-2xl mx-auto">
                        For press inquiries, interview requests, or additional information,
                        please contact our communications team.
                    </p>
                    <div className="flex flex-col sm:flex-row gap-4 justify-center">
                        <Button
                            href={`mailto:${siteConfig.company.pressEmail}`}
                            variant="secondary"
                            size="lg"
                        >
                            {siteConfig.company.pressEmail}
                        </Button>
                    </div>
                </div>
            </Section>
        </Layout>
    );
}
