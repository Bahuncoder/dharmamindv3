import { Layout, Section, Button } from '../components';
import { siteConfig } from '../config/site.config';

export default function Charter() {
    return (
        <Layout
            title="Charter"
            description="Our founding principles and commitments to responsible AI development."
        >
            {/* Hero Section */}
            <Section background="dark" padding="xl">
                <div className="max-w-3xl">
                    <h1 className="text-4xl md:text-6xl font-bold mb-6">
                        The {siteConfig.company.name} Charter
                    </h1>
                    <p className="text-xl text-neutral-300 leading-relaxed">
                        {siteConfig.charter.preamble}
                    </p>
                </div>
            </Section>

            {/* Principles */}
            <Section>
                <div className="max-w-4xl mx-auto">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl font-bold text-neutral-900 mb-4">
                            Our Founding Principles
                        </h2>
                        <p className="text-neutral-600">
                            These principles guide every decision we make, from product development
                            to hiring to partnerships.
                        </p>
                    </div>

                    <div className="space-y-12">
                        {siteConfig.charter.principles.map((principle) => (
                            <div
                                key={principle.number}
                                className="grid md:grid-cols-12 gap-6 items-start"
                            >
                                {/* Number */}
                                <div className="md:col-span-2">
                                    <div className="w-16 h-16 rounded-full bg-gold-600 text-white flex items-center justify-center text-2xl font-bold">
                                        {principle.number}
                                    </div>
                                </div>

                                {/* Content */}
                                <div className="md:col-span-10">
                                    <h3 className="text-2xl font-bold text-neutral-900 mb-3">
                                        {principle.title}
                                    </h3>
                                    <p className="text-neutral-600 mb-4 leading-relaxed">
                                        {principle.description}
                                    </p>
                                    <div className="bg-neutral-50 rounded-xl p-4 border-l-4 border-gold-600">
                                        <p className="text-sm text-neutral-700">
                                            <span className="font-semibold">Our Commitment:</span>{' '}
                                            {principle.commitment}
                                        </p>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </Section>

            {/* Ethics Board */}
            <Section background="light">
                <div className="max-w-4xl mx-auto">
                    <div className="text-center mb-12">
                        <h2 className="text-3xl font-bold text-neutral-900 mb-4">
                            Ethics Governance
                        </h2>
                        <p className="text-neutral-600 max-w-2xl mx-auto">
                            {siteConfig.charter.governance.description}
                        </p>
                    </div>

                    <div className="grid md:grid-cols-2 gap-6">
                        {siteConfig.charter.governance.members.map((member) => (
                            <div
                                key={member.name}
                                className="bg-neutral-100 rounded-2xl border border-neutral-200 p-6 flex items-center"
                            >
                                <div className="w-14 h-14 rounded-full bg-gradient-to-br from-neutral-200 to-neutral-300 flex items-center justify-center mr-4">
                                    <span className="text-2xl">👤</span>
                                </div>
                                <div>
                                    <h3 className="font-semibold text-neutral-900">{member.name}</h3>
                                    <p className="text-sm text-neutral-600">{member.role}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </Section>

            {/* Safety Practices */}
            <Section>
                <div className="max-w-4xl mx-auto">
                    <div className="text-center mb-12">
                        <h2 className="text-3xl font-bold text-neutral-900 mb-4">
                            How We Put Principles Into Practice
                        </h2>
                        <p className="text-neutral-600">
                            Principles only matter if they're lived. Here's how we implement ours.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-2 gap-6">
                        {siteConfig.safety.practices.map((practice, index) => (
                            <div
                                key={index}
                                className="flex items-start p-4 bg-neutral-50 rounded-xl"
                            >
                                <span className="text-gold-600 mr-3 mt-0.5">✓</span>
                                <span className="text-neutral-700">{practice}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </Section>

            {/* Key Commitments */}
            <Section background="dark">
                <div className="max-w-4xl mx-auto">
                    <div className="text-center mb-12">
                        <h2 className="text-3xl font-bold mb-4">
                            Non-Negotiable Commitments
                        </h2>
                        <p className="text-neutral-400">
                            Some things we will never compromise on, regardless of business pressure.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-3 gap-8">
                        {siteConfig.safety.commitments.map((commitment) => (
                            <div key={commitment.title} className="text-center">
                                <div className="w-16 h-16 rounded-full bg-neutral-100/10 flex items-center justify-center mx-auto mb-4">
                                    <span className="text-3xl">🛡️</span>
                                </div>
                                <h3 className="text-lg font-semibold mb-3">{commitment.title}</h3>
                                <p className="text-neutral-400 text-sm">{commitment.description}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </Section>

            {/* Dharma Values */}
            <Section background="light">
                <div className="max-w-4xl mx-auto">
                    <div className="text-center mb-12">
                        <h2 className="text-3xl font-bold text-neutral-900 mb-4">
                            Rooted in Dharma
                        </h2>
                        <p className="text-neutral-600">
                            Our principles are inspired by timeless wisdom from Dharmic traditions.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-2 gap-6">
                        {siteConfig.dharmaFaq.slice(0, 4).map((item) => (
                            <div
                                key={item.term}
                                className="bg-neutral-100 rounded-2xl border border-neutral-200 p-6"
                            >
                                <h3 className="text-lg font-semibold text-neutral-900 mb-2">
                                    {item.term}
                                </h3>
                                <p className="text-neutral-600 text-sm">
                                    {item.answer}
                                </p>
                            </div>
                        ))}
                    </div>
                </div>
            </Section>

            {/* Accountability */}
            <Section>
                <div className="max-w-4xl mx-auto text-center">
                    <h2 className="text-3xl font-bold text-neutral-900 mb-6">
                        Hold Us Accountable
                    </h2>
                    <p className="text-xl text-neutral-600 mb-8 max-w-2xl mx-auto">
                        We publish this charter publicly because we want to be held accountable.
                        If you see us falling short of these principles, please let us know.
                    </p>
                    <div className="flex flex-col sm:flex-row gap-4 justify-center">
                        <Button href={`mailto:ethics@dharmamind.ai`} size="lg">
                            Contact Ethics Board
                        </Button>
                        <Button href="/safety" variant="outline" size="lg">
                            Learn More About Safety
                        </Button>
                    </div>
                </div>
            </Section>

            {/* Version & Date */}
            <Section background="light" padding="sm">
                <div className="max-w-4xl mx-auto text-center text-sm text-neutral-500">
                    <p>
                        Charter Version 1.0 · Last Updated December 2024 ·
                        <a href="/charter/history" className="text-neutral-700 hover:underline ml-1">
                            View Version History
                        </a>
                    </p>
                </div>
            </Section>
        </Layout>
    );
}
