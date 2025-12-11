import { Layout, Section, Button } from '../components';
import { siteConfig } from '../config/site.config';

export default function Research() {
    return (
        <Layout
            title="Research"
            description="Advancing the science of ethical AI for spiritual growth and human flourishing."
        >
            {/* Hero Section */}
            <Section background="light" padding="xl">
                <div className="max-w-3xl">
                    <h1 className="text-4xl md:text-6xl font-bold text-neutral-900 mb-6">
                        Research at {siteConfig.company.name}
                    </h1>
                    <p className="text-xl text-neutral-600 leading-relaxed">
                        We're pioneering new approaches to building AI systems that understand the depth
                        and nuance of spiritual traditions while maintaining the highest ethical standards.
                    </p>
                </div>
            </Section>

            {/* Research Areas */}
            <Section>
                <div className="mb-12">
                    <h2 className="text-3xl font-bold text-neutral-900 mb-4">Research Areas</h2>
                    <p className="text-neutral-600 max-w-2xl">
                        Our research spans multiple domains, from language understanding to computer vision,
                        all focused on serving humanity's spiritual needs.
                    </p>
                </div>

                <div className="grid md:grid-cols-2 gap-6">
                    {siteConfig.research.areas.map((area) => (
                        <div
                            key={area.id}
                            id={area.id}
                            className="p-8 rounded-2xl border border-neutral-200 bg-neutral-100 hover:shadow-lg transition-shadow"
                        >
                            <div className="flex items-start justify-between mb-4">
                                <span className="text-4xl">{area.icon}</span>
                                <span className="px-3 py-1 bg-gold-100 text-gold-700 text-xs rounded-full">
                                    {area.status}
                                </span>
                            </div>
                            <h3 className="text-xl font-semibold text-neutral-900 mb-3">{area.title}</h3>
                            <p className="text-neutral-600 mb-4">{area.description}</p>
                            <div className="text-sm text-neutral-500">
                                {area.papers} papers published
                            </div>
                        </div>
                    ))}
                </div>
            </Section>

            {/* Publications */}
            <Section background="light" id="publications">
                <div className="mb-12">
                    <h2 className="text-3xl font-bold text-neutral-900 mb-4">Publications</h2>
                    <p className="text-neutral-600 max-w-2xl">
                        Our peer-reviewed papers contribute to the broader scientific community.
                    </p>
                </div>

                {/* Featured Publications */}
                <div className="mb-12">
                    <h3 className="text-lg font-semibold text-neutral-900 mb-6">Featured Papers</h3>
                    <div className="grid md:grid-cols-2 gap-6">
                        {siteConfig.research.publications.filter(p => p.featured).map((paper) => (
                            <div
                                key={paper.title}
                                className="p-6 rounded-2xl border border-neutral-200 bg-neutral-100"
                            >
                                <div className="flex items-center gap-2 mb-3">
                                    <span className="px-2 py-1 bg-gold-600 text-white text-xs rounded-full">Featured</span>
                                    <span className="text-sm text-neutral-500">{paper.venue} · {paper.year}</span>
                                </div>
                                <h4 className="text-lg font-semibold text-neutral-900 mb-2">{paper.title}</h4>
                                <p className="text-sm text-neutral-600 mb-3">{paper.authors.join(', ')}</p>
                                <p className="text-neutral-500 text-sm mb-4 line-clamp-3">{paper.abstract}</p>
                                <Button href={paper.link} variant="outline" size="sm">
                                    Read Paper →
                                </Button>
                            </div>
                        ))}
                    </div>
                </div>

                {/* All Publications */}
                <div>
                    <h3 className="text-lg font-semibold text-neutral-900 mb-6">All Publications</h3>
                    <div className="space-y-4">
                        {siteConfig.research.publications.filter(p => !p.featured).map((paper) => (
                            <div
                                key={paper.title}
                                className="p-4 rounded-xl border border-neutral-200 bg-neutral-100 flex items-center justify-between"
                            >
                                <div>
                                    <h4 className="font-medium text-neutral-900">{paper.title}</h4>
                                    <p className="text-sm text-neutral-500">
                                        {paper.authors.join(', ')} · {paper.venue} · {paper.year}
                                    </p>
                                </div>
                                <Button href={paper.link} variant="ghost" size="sm">
                                    PDF →
                                </Button>
                            </div>
                        ))}
                    </div>
                </div>
            </Section>

            {/* Open Source */}
            <Section>
                <div className="mb-12">
                    <h2 className="text-3xl font-bold text-neutral-900 mb-4">Open Source</h2>
                    <p className="text-neutral-600 max-w-2xl">
                        We believe in open research. These tools and datasets are available
                        for the community to use and build upon.
                    </p>
                </div>

                <div className="grid md:grid-cols-3 gap-6">
                    {siteConfig.research.openSource.map((project) => (
                        <a
                            key={project.name}
                            href={project.link}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="p-6 rounded-2xl border border-neutral-200 bg-neutral-100 hover:shadow-lg hover:border-neutral-300 transition-all group"
                        >
                            <div className="flex items-center justify-between mb-4">
                                <span className="px-2 py-1 bg-neutral-100 text-neutral-600 text-xs rounded-full">
                                    {project.language}
                                </span>
                                <div className="flex items-center text-neutral-500 text-sm">
                                    <span className="mr-1">⭐</span>
                                    {project.stars}
                                </div>
                            </div>
                            <h3 className="text-lg font-semibold text-neutral-900 group-hover:text-neutral-700 mb-2">
                                {project.name}
                            </h3>
                            <p className="text-neutral-600 text-sm">{project.description}</p>
                        </a>
                    ))}
                </div>

                <div className="text-center mt-12">
                    <Button href={siteConfig.social.github} variant="outline" external>
                        View All Repositories on GitHub
                    </Button>
                </div>
            </Section>

            {/* Join Research Team */}
            <Section background="dark">
                <div className="text-center max-w-2xl mx-auto">
                    <h2 className="text-3xl md:text-4xl font-bold mb-6">
                        Join Our Research Team
                    </h2>
                    <p className="text-neutral-300 mb-8">
                        We're looking for researchers passionate about the intersection of
                        AI, ethics, and spirituality. Help us advance the science of human flourishing.
                    </p>
                    <div className="flex flex-col sm:flex-row gap-4 justify-center">
                        <Button href="/careers" variant="secondary">
                            View Research Positions
                        </Button>
                        <Button
                            href={`mailto:${siteConfig.company.email}`}
                            variant="outline"
                            className="border-white text-white hover:bg-neutral-100 hover:text-neutral-900"
                        >
                            Contact Research Team
                        </Button>
                    </div>
                </div>
            </Section>
        </Layout>
    );
}
