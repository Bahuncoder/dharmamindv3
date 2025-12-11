import { Layout, Section, TeamCard, Button } from '../components';
import { siteConfig } from '../config/site.config';

export default function Team() {
    return (
        <Layout
            title="Team"
            description="Meet the team building AI that serves humanity's highest aspirations."
        >
            {/* Hero Section */}
            <Section background="light" padding="xl">
                <div className="max-w-3xl">
                    <h1 className="text-4xl md:text-6xl font-bold text-neutral-900 mb-6">
                        The People Behind {siteConfig.company.name}
                    </h1>
                    <p className="text-xl text-neutral-600 leading-relaxed">
                        We're a diverse team of engineers, researchers, philosophers, and practitioners
                        united by a shared mission: building AI that honors wisdom traditions while
                        advancing human potential.
                    </p>
                </div>
            </Section>

            {/* Leadership Section */}
            <Section>
                <div className="mb-12">
                    <h2 className="text-3xl font-bold text-neutral-900 mb-4">Leadership</h2>
                    <p className="text-neutral-600 max-w-2xl">
                        Our leadership team combines deep expertise in AI, philosophy, and spiritual traditions.
                    </p>
                </div>

                <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
                    {siteConfig.team.leadership.map((member) => (
                        <TeamCard key={member.name} member={member} type="leadership" />
                    ))}
                </div>
            </Section>

            {/* Advisors Section */}
            <Section background="light">
                <div className="mb-12">
                    <h2 className="text-3xl font-bold text-neutral-900 mb-4">Advisory Board</h2>
                    <p className="text-neutral-600 max-w-2xl">
                        We're guided by world-renowned experts in AI ethics, spirituality, philosophy, and psychology.
                    </p>
                </div>

                <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
                    {siteConfig.team.advisors.map((advisor) => (
                        <TeamCard key={advisor.name} member={advisor} type="advisor" />
                    ))}
                </div>
            </Section>

            {/* Team Stats */}
            <Section>
                <div className="text-center mb-12">
                    <h2 className="text-3xl font-bold text-neutral-900 mb-4">Our Team by the Numbers</h2>
                    <p className="text-neutral-600">
                        Growing fast, but staying true to our values.
                    </p>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-12">
                    {siteConfig.stats.company.map((stat) => (
                        <div key={stat.label} className="text-center">
                            <div className="text-4xl md:text-5xl font-bold text-neutral-900 mb-2">
                                {stat.value}
                            </div>
                            <div className="text-neutral-600">{stat.label}</div>
                        </div>
                    ))}
                </div>

                {/* Department Breakdown */}
                <div className="bg-neutral-50 rounded-2xl p-8">
                    <h3 className="text-xl font-semibold text-neutral-900 mb-6">Team Composition</h3>
                    <div className="space-y-4">
                        {siteConfig.team.departments.map((dept) => (
                            <div key={dept.name} className="flex items-center">
                                <span className="w-32 text-neutral-600">{dept.name}</span>
                                <div className="flex-grow h-4 bg-neutral-200 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-gold-600 rounded-full"
                                        style={{
                                            width: `${(dept.count / siteConfig.team.departments.reduce((a, b) => a + b.count, 0)) * 100}%`
                                        }}
                                    />
                                </div>
                                <span className="w-16 text-right text-gold-600 font-medium">{dept.count}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </Section>

            {/* Values Section */}
            <Section background="light">
                <div className="text-center mb-12">
                    <h2 className="text-3xl font-bold text-neutral-900 mb-4">What Unites Us</h2>
                    <p className="text-neutral-500 max-w-2xl mx-auto">
                        Beyond skills and expertise, we share a deep commitment to these values.
                    </p>
                </div>

                <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
                    {siteConfig.careers.values.map((value) => (
                        <div key={value.title} className="text-center">
                            <span className="text-4xl mb-4 block">{value.icon}</span>
                            <h3 className="text-lg font-semibold mb-2">{value.title}</h3>
                            <p className="text-neutral-400 text-sm">{value.description}</p>
                        </div>
                    ))}
                </div>
            </Section>

            {/* Join Us CTA */}
            <Section>
                <div className="bg-gradient-to-br from-neutral-100 to-neutral-50 rounded-3xl p-8 md:p-12 text-center">
                    <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-4">
                        Want to Join Our Team?
                    </h2>
                    <p className="text-xl text-neutral-600 mb-8 max-w-2xl mx-auto">
                        We're always looking for talented people who are passionate about
                        building technology that serves humanity's highest aspirations.
                    </p>
                    <div className="flex flex-col sm:flex-row gap-4 justify-center">
                        <Button href="/careers" size="lg">
                            View Open Positions
                        </Button>
                        <Button href={`mailto:${siteConfig.company.careersEmail}`} variant="outline" size="lg">
                            Get in Touch
                        </Button>
                    </div>
                </div>
            </Section>
        </Layout>
    );
}
