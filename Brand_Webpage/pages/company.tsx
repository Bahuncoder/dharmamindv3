import { Layout, Section, Button, StatCard } from '../components';
import { siteConfig } from '../config/site.config';

export default function Company() {
    return (
        <Layout
            title="Company"
            description="Learn about DharmaMind's mission to democratize spiritual wisdom through ethical AI."
        >
            {/* Hero Section */}
            <Section background="light" padding="xl">
                <div className="max-w-3xl">
                    <h1 className="text-4xl md:text-6xl font-bold text-neutral-900 mb-6">
                        About {siteConfig.company.name}
                    </h1>
                    <p className="text-xl text-neutral-600 leading-relaxed">
                        {siteConfig.company.description}
                    </p>
                </div>
            </Section>

            {/* Mission & Vision */}
            <Section>
                <div className="grid md:grid-cols-2 gap-12">
                    <div>
                        <span className="text-sm font-medium text-neutral-500 uppercase tracking-wider mb-4 block">
                            Our Mission
                        </span>
                        <h2 className="text-2xl md:text-3xl font-bold text-neutral-900 mb-4">
                            {siteConfig.company.mission}
                        </h2>
                    </div>
                    <div>
                        <span className="text-sm font-medium text-neutral-500 uppercase tracking-wider mb-4 block">
                            Our Vision
                        </span>
                        <h2 className="text-2xl md:text-3xl font-bold text-neutral-900 mb-4">
                            {siteConfig.company.vision}
                        </h2>
                    </div>
                </div>
            </Section>

            {/* Stats */}
            <Section background="dark">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
                    {siteConfig.stats.company.map((stat) => (
                        <StatCard
                            key={stat.label}
                            value={stat.value}
                            label={stat.label}
                            dark
                        />
                    ))}
                </div>
            </Section>

            {/* Our Story */}
            <Section>
                <div className="max-w-4xl mx-auto">
                    <div className="text-center mb-12">
                        <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-4">
                            Our Story
                        </h2>
                    </div>

                    <div className="prose prose-lg max-w-none text-neutral-600">
                        <p>
                            {siteConfig.company.name} was founded in {siteConfig.company.founded} with a simple but
                            ambitious vision: to make the world's spiritual wisdom accessible to everyone through
                            the power of ethical AI.
                        </p>
                        <p>
                            Our founder, Sudip Paudel, envisioned a future where ancient spiritual wisdom
                            could be accessible to everyone through modern technology. While AI was advancing
                            rapidly, it wasn't getting wiser. It couldn't help people with the questions that
                            matter most: Who am I? What is my purpose? How should I live?
                        </p>
                        <p>
                            Deeply connected to his own spiritual roots and studying Vedantic philosophy
                            and other wisdom traditions, Sudip realized that while humanity had accumulated
                            thousands of years of profound spiritual insight, access to this wisdom remained
                            limited—requiring years of study, the right teachers, or the privilege of time
                            and resources.
                        </p>
                        <p>
                            {siteConfig.company.name} was born from the conviction that technology could help
                            bridge this gap—if built thoughtfully, ethically, and in deep collaboration with
                            authentic wisdom keepers.
                        </p>
                        <p>
                            Today, our team includes AI researchers, philosophers, Sanskrit scholars, Buddhist
                            monks, psychologists, and spiritual practitioners. We're united by a shared belief
                            that AI can serve humanity's highest aspirations, not just its commercial needs.
                        </p>
                    </div>
                </div>
            </Section>

            {/* Timeline */}
            <Section background="light">
                <div className="text-center mb-12">
                    <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-4">
                        Our Journey
                    </h2>
                    <p className="text-neutral-600">Key milestones in our story</p>
                </div>

                <div className="max-w-3xl mx-auto">
                    <div className="relative">
                        {/* Timeline line */}
                        <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-neutral-200" />

                        {/* Timeline items */}
                        <div className="space-y-8">
                            {siteConfig.timeline.map((event, index) => (
                                <div key={index} className="relative flex items-start">
                                    {/* Dot */}
                                    <div className="absolute left-6 w-4 h-4 rounded-full bg-gold-600 border-4 border-white" />

                                    {/* Content */}
                                    <div className="ml-16 pb-8">
                                        <span className="text-sm font-medium text-neutral-500">
                                            {event.quarter} {event.year}
                                        </span>
                                        <h3 className="text-lg font-semibold text-neutral-900 mt-1">
                                            {event.title}
                                        </h3>
                                        <p className="text-neutral-600 mt-1">{event.description}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </Section>

            {/* Values */}
            <Section>
                <div className="text-center mb-12">
                    <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-4">
                        Our Values
                    </h2>
                    <p className="text-neutral-600 max-w-2xl mx-auto">
                        These principles guide every decision we make, from product development to hiring.
                    </p>
                </div>

                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
                    {[
                        {
                            title: "Ahimsa (Non-Harm)",
                            description: "Our AI is designed to never cause psychological, emotional, or spiritual harm. User wellbeing always comes first.",
                            icon: "🛡️",
                        },
                        {
                            title: "Satya (Truth)",
                            description: "We're honest about what AI can and cannot do. We never claim our AI has spiritual authority.",
                            icon: "✨",
                        },
                        {
                            title: "Seva (Service)",
                            description: "We exist to serve humanity's spiritual growth, not to maximize engagement or profit.",
                            icon: "🙏",
                        },
                        {
                            title: "Shraddha (Respect)",
                            description: "We approach all wisdom traditions with deep respect, working with authentic scholars and practitioners.",
                            icon: "📿",
                        },
                        {
                            title: "Viveka (Discernment)",
                            description: "We make careful, thoughtful decisions about what our AI should and shouldn't do.",
                            icon: "🧠",
                        },
                        {
                            title: "Karuna (Compassion)",
                            description: "We build technology with compassion for all beings, especially those who are suffering.",
                            icon: "💝",
                        },
                    ].map((value) => (
                        <div
                            key={value.title}
                            className="p-8 rounded-2xl border border-neutral-200 bg-neutral-100 hover:shadow-lg transition-shadow"
                        >
                            <span className="text-4xl mb-4 block">{value.icon}</span>
                            <h3 className="text-xl font-semibold text-neutral-900 mb-3">
                                {value.title}
                            </h3>
                            <p className="text-neutral-600">{value.description}</p>
                        </div>
                    ))}
                </div>
            </Section>

            {/* Leadership Preview */}
            <Section background="light">
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-12">
                    <div>
                        <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-2">
                            Leadership
                        </h2>
                        <p className="text-neutral-600">
                            Meet the team building {siteConfig.company.name}
                        </p>
                    </div>
                    <Button href="/team" variant="outline" className="mt-4 md:mt-0">
                        View Full Team
                    </Button>
                </div>

                <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
                    {siteConfig.team.leadership.map((member) => (
                        <div key={member.name} className="text-center">
                            <div className="w-24 h-24 rounded-full bg-gradient-to-br from-neutral-200 to-neutral-300 mx-auto mb-4 flex items-center justify-center">
                                <span className="text-4xl opacity-50">👤</span>
                            </div>
                            <h3 className="font-semibold text-neutral-900">{member.name}</h3>
                            <p className="text-sm text-neutral-600">{member.role}</p>
                        </div>
                    ))}
                </div>
            </Section>

            {/* Contact/Locations */}
            <Section>
                <div className="grid md:grid-cols-2 gap-12">
                    <div>
                        <h2 className="text-3xl font-bold text-neutral-900 mb-6">
                            Get in Touch
                        </h2>
                        <div className="space-y-4">
                            <div>
                                <span className="text-sm font-medium text-neutral-500 block mb-1">General Inquiries</span>
                                <a href={`mailto:${siteConfig.company.email}`} className="text-neutral-900 hover:underline">
                                    {siteConfig.company.email}
                                </a>
                            </div>
                            <div>
                                <span className="text-sm font-medium text-neutral-500 block mb-1">Press</span>
                                <a href={`mailto:${siteConfig.company.pressEmail}`} className="text-neutral-900 hover:underline">
                                    {siteConfig.company.pressEmail}
                                </a>
                            </div>
                            <div>
                                <span className="text-sm font-medium text-neutral-500 block mb-1">Careers</span>
                                <a href={`mailto:${siteConfig.company.careersEmail}`} className="text-neutral-900 hover:underline">
                                    {siteConfig.company.careersEmail}
                                </a>
                            </div>
                            <div>
                                <span className="text-sm font-medium text-neutral-500 block mb-1">Support</span>
                                <a href={`mailto:${siteConfig.company.supportEmail}`} className="text-neutral-900 hover:underline">
                                    {siteConfig.company.supportEmail}
                                </a>
                            </div>
                        </div>
                    </div>

                    <div>
                        <h2 className="text-3xl font-bold text-neutral-900 mb-6">
                            Headquarters
                        </h2>
                        <p className="text-neutral-600 mb-4">
                            {siteConfig.company.headquarters}
                        </p>
                        <div className="bg-neutral-100 rounded-2xl aspect-video flex items-center justify-center">
                            <span className="text-4xl">📍</span>
                        </div>
                    </div>
                </div>
            </Section>

            {/* Press */}
            <Section background="light">
                <div className="text-center mb-12">
                    <h2 className="text-3xl font-bold text-neutral-900 mb-4">
                        In the Press
                    </h2>
                </div>

                <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
                    {siteConfig.press.coverage.map((item) => (
                        <a
                            key={item.outlet}
                            href={item.link}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="p-6 rounded-2xl border border-neutral-200 bg-neutral-100 hover:shadow-lg hover:border-neutral-300 transition-all"
                        >
                            <span className="text-2xl font-semibold text-neutral-400 mb-4 block">
                                {item.outlet}
                            </span>
                            <h3 className="font-medium text-neutral-900 mb-2">{item.title}</h3>
                            <span className="text-sm text-neutral-500">
                                {new Date(item.date).toLocaleDateString('en-US', {
                                    year: 'numeric',
                                    month: 'long',
                                    day: 'numeric'
                                })}
                            </span>
                        </a>
                    ))}
                </div>

                <div className="text-center mt-8">
                    <Button href={siteConfig.press.kit} variant="outline">
                        Download Press Kit
                    </Button>
                </div>
            </Section>

            {/* CTA */}
            <Section>
                <div className="bg-neutral-200 border border-neutral-300 rounded-3xl p-8 md:p-16 text-center">
                    <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-6">
                        Join Us on This Journey
                    </h2>
                    <p className="text-xl text-neutral-600 mb-8 max-w-2xl mx-auto">
                        Whether you're looking to use our products, join our team, or partner with us,
                        we'd love to hear from you.
                    </p>
                    <div className="flex flex-col sm:flex-row gap-4 justify-center">
                        <Button href="/careers" variant="primary" size="lg">
                            View Open Positions
                        </Button>
                        <Button
                            href={`mailto:${siteConfig.company.email}`}
                            variant="outline"
                            size="lg"
                        >
                            Contact Us
                        </Button>
                    </div>
                </div>
            </Section>
        </Layout>
    );
}
