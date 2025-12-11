import { Layout, Section, Button } from '../components';
import { siteConfig } from '../config/site.config';

export default function Safety() {
    return (
        <Layout
            title="Safety"
            description="Our commitment to building AI that is safe, ethical, and respectful of all spiritual traditions."
        >
            {/* Hero Section */}
            <Section background="dark" padding="xl">
                <div className="max-w-3xl">
                    <h1 className="text-4xl md:text-6xl font-bold mb-6">
                        Safety & Ethics
                    </h1>
                    <p className="text-xl text-neutral-300 leading-relaxed">
                        We believe AI in spiritual contexts demands the highest ethical standards.
                        Our commitment to safety isn't just policy—it's the foundation of everything we build.
                    </p>
                </div>
            </Section>

            {/* Core Principles */}
            <Section>
                <div className="text-center mb-16">
                    <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-4">
                        Our Safety Principles
                    </h2>
                    <p className="text-xl text-neutral-600 max-w-2xl mx-auto">
                        Six principles guide every decision we make about AI safety and ethics.
                    </p>
                </div>

                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
                    {siteConfig.safety.principles.map((principle) => (
                        <div
                            key={principle.title}
                            className="p-8 rounded-2xl border border-neutral-200 bg-neutral-100 hover:shadow-lg transition-shadow"
                        >
                            <span className="text-4xl mb-4 block">{principle.icon}</span>
                            <h3 className="text-xl font-semibold text-neutral-900 mb-3">
                                {principle.title}
                            </h3>
                            <p className="text-neutral-600">{principle.description}</p>
                        </div>
                    ))}
                </div>
            </Section>

            {/* Safety in Practice */}
            <Section background="light">
                <div className="grid md:grid-cols-2 gap-12 items-center">
                    <div>
                        <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-6">
                            Safety in Practice
                        </h2>
                        <p className="text-neutral-600 mb-8">
                            Principles are only meaningful when put into practice. Here's how we
                            implement our safety commitments across every aspect of our work.
                        </p>

                        <div className="space-y-4">
                            {siteConfig.safety.practices.map((practice, index) => (
                                <div
                                    key={index}
                                    className="flex items-start p-4 bg-neutral-100 rounded-xl border border-neutral-200"
                                >
                                    <span className="text-gold-600 mr-3 mt-0.5">✓</span>
                                    <span className="text-neutral-700">{practice}</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="bg-neutral-100 rounded-2xl border border-neutral-200 p-8">
                        <h3 className="text-xl font-semibold text-neutral-900 mb-6">
                            Key Commitments
                        </h3>
                        <div className="space-y-6">
                            {siteConfig.safety.commitments.map((commitment) => (
                                <div key={commitment.title}>
                                    <h4 className="font-medium text-neutral-900 mb-2">
                                        {commitment.title}
                                    </h4>
                                    <p className="text-neutral-600 text-sm">{commitment.description}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </Section>

            {/* Dharmic Foundation */}
            <Section>
                <div className="text-center mb-12">
                    <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-4">
                        Rooted in Dharmic Principles
                    </h2>
                    <p className="text-neutral-600 max-w-2xl mx-auto">
                        Our approach to AI safety is informed by timeless wisdom from Dharmic traditions.
                    </p>
                </div>

                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {siteConfig.dharmaFaq.map((item) => (
                        <div
                            key={item.term}
                            className="p-6 rounded-2xl border border-neutral-200 bg-neutral-100"
                        >
                            <h3 className="text-lg font-semibold text-neutral-900 mb-2">
                                {item.term}
                            </h3>
                            <p className="text-neutral-600 text-sm">{item.answer}</p>
                        </div>
                    ))}
                </div>
            </Section>

            {/* Responsible AI Framework */}
            <Section background="dark">
                <div className="max-w-4xl mx-auto">
                    <div className="text-center mb-12">
                        <h2 className="text-3xl md:text-4xl font-bold mb-4">
                            Responsible AI Framework
                        </h2>
                        <p className="text-neutral-400">
                            Our systematic approach to ensuring AI safety across the development lifecycle.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-2 gap-8">
                        {[
                            {
                                title: "Design",
                                items: [
                                    "Ethics review for all new features",
                                    "Safety-first architecture decisions",
                                    "Inclusive design principles",
                                ],
                            },
                            {
                                title: "Development",
                                items: [
                                    "Bias testing throughout development",
                                    "Red teaming by diverse groups",
                                    "Continuous safety monitoring",
                                ],
                            },
                            {
                                title: "Deployment",
                                items: [
                                    "Staged rollouts with safety gates",
                                    "Real-time content monitoring",
                                    "Rapid response protocols",
                                ],
                            },
                            {
                                title: "Evaluation",
                                items: [
                                    "Regular third-party audits",
                                    "User feedback integration",
                                    "Public safety reporting",
                                ],
                            },
                        ].map((phase) => (
                            <div key={phase.title} className="bg-neutral-100/5 rounded-xl p-6">
                                <h3 className="text-lg font-semibold mb-4">{phase.title}</h3>
                                <ul className="space-y-2">
                                    {phase.items.map((item) => (
                                        <li key={item} className="flex items-center text-neutral-300 text-sm">
                                            <span className="text-green-400 mr-2">•</span>
                                            {item}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        ))}
                    </div>
                </div>
            </Section>

            {/* Ethics Board */}
            <Section>
                <div className="grid md:grid-cols-2 gap-12 items-center">
                    <div>
                        <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-6">
                            Ethics Advisory Board
                        </h2>
                        <p className="text-neutral-600 mb-6">
                            {siteConfig.charter.governance.description}
                        </p>
                        <p className="text-neutral-600 mb-8">
                            Our ethics board reviews major product decisions, investigates concerns,
                            and publishes annual reports on our safety practices.
                        </p>
                        <Button href="/charter">Read Our Charter</Button>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        {siteConfig.charter.governance.members.map((member) => (
                            <div
                                key={member.name}
                                className="p-4 rounded-xl border border-neutral-200 bg-neutral-100"
                            >
                                <div className="w-12 h-12 rounded-full bg-neutral-100 flex items-center justify-center mb-3">
                                    <span className="text-xl">👤</span>
                                </div>
                                <h4 className="font-medium text-neutral-900">{member.name}</h4>
                                <p className="text-sm text-neutral-500">{member.role}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </Section>

            {/* Report a Concern */}
            <Section background="light">
                <div className="text-center max-w-2xl mx-auto">
                    <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-6">
                        Report a Concern
                    </h2>
                    <p className="text-neutral-600 mb-8">
                        If you've encountered content or behavior from our AI that concerns you,
                        we want to know. All reports are reviewed by our safety team.
                    </p>
                    <div className="flex flex-col sm:flex-row gap-4 justify-center">
                        <Button href="mailto:safety@dharmamind.ai" size="lg">
                            Report Safety Issue
                        </Button>
                        <Button href={`mailto:ethics@dharmamind.ai`} variant="outline" size="lg">
                            Contact Ethics Board
                        </Button>
                    </div>
                </div>
            </Section>

            {/* Resources */}
            <Section>
                <div className="text-center mb-12">
                    <h2 className="text-2xl font-bold text-neutral-900 mb-4">
                        Safety Resources
                    </h2>
                </div>

                <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
                    {[
                        {
                            title: "Safety Documentation",
                            description: "Technical details about our safety systems and processes.",
                            link: "/docs/safety",
                            icon: "📄",
                        },
                        {
                            title: "Annual Safety Report",
                            description: "Our yearly review of safety incidents and improvements.",
                            link: "/safety/report-2024",
                            icon: "📊",
                        },
                        {
                            title: "Research Papers",
                            description: "Our published research on AI safety in spiritual contexts.",
                            link: "/research#safety",
                            icon: "📚",
                        },
                    ].map((resource) => (
                        <a
                            key={resource.title}
                            href={resource.link}
                            className="p-6 rounded-2xl border border-neutral-200 bg-neutral-100 hover:shadow-lg hover:border-neutral-300 transition-all group"
                        >
                            <span className="text-3xl mb-4 block">{resource.icon}</span>
                            <h3 className="text-lg font-semibold text-neutral-900 group-hover:text-neutral-700 mb-2">
                                {resource.title}
                            </h3>
                            <p className="text-neutral-600 text-sm">{resource.description}</p>
                        </a>
                    ))}
                </div>
            </Section>
        </Layout>
    );
}
