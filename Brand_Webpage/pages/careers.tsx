import { Layout, Section, JobCard, Button } from '../components';
import { siteConfig } from '../config/site.config';

export default function Careers() {
    const totalOpenings = siteConfig.careers.openings.length;

    // Group jobs by department
    const jobsByDepartment = siteConfig.careers.departments.reduce((acc, dept) => {
        acc[dept.name] = siteConfig.careers.openings.filter(job => job.department === dept.name);
        return acc;
    }, {} as Record<string, typeof siteConfig.careers.openings>);

    return (
        <Layout
            title="Careers"
            description="Join DharmaMind and help build AI that serves humanity's highest aspirations."
        >
            {/* Hero Section */}
            <Section background="dark" padding="xl">
                <div className="max-w-3xl">
                    <span className="inline-block px-4 py-2 bg-neutral-100/10 rounded-full text-sm mb-6">
                        {totalOpenings} Open Positions
                    </span>
                    <h1 className="text-4xl md:text-6xl font-bold mb-6">
                        Build the Future of Spiritual AI
                    </h1>
                    <p className="text-xl text-neutral-300 leading-relaxed mb-8">
                        Join a team of engineers, researchers, philosophers, and practitioners
                        working on technology that truly matters. Help us create AI that honors
                        ancient wisdom while advancing human potential.
                    </p>
                    <Button href="#openings" variant="secondary" size="lg">
                        View Open Positions
                    </Button>
                </div>
            </Section>

            {/* Why Join Us */}
            <Section>
                <div className="text-center mb-16">
                    <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-4">
                        Why Join {siteConfig.company.name}?
                    </h2>
                    <p className="text-xl text-neutral-600 max-w-2xl mx-auto">
                        We're building something meaningful, and we want you to be part of it.
                    </p>
                </div>

                <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
                    {siteConfig.careers.values.map((value) => (
                        <div key={value.title} className="text-center">
                            <span className="text-5xl mb-6 block">{value.icon}</span>
                            <h3 className="text-xl font-semibold text-neutral-900 mb-3">
                                {value.title}
                            </h3>
                            <p className="text-neutral-600">
                                {value.description}
                            </p>
                        </div>
                    ))}
                </div>
            </Section>

            {/* Benefits */}
            <Section background="light">
                <div className="grid md:grid-cols-2 gap-12 items-center">
                    <div>
                        <h2 className="text-3xl font-bold text-neutral-900 mb-6">
                            Benefits & Perks
                        </h2>
                        <p className="text-neutral-600 mb-8">
                            We believe in taking care of our team so they can do their best work.
                            Here's what we offer:
                        </p>
                        <ul className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                            {siteConfig.careers.benefits.map((benefit) => (
                                <li key={benefit} className="flex items-center">
                                    <span className="text-gold-600 mr-3">✓</span>
                                    <span className="text-neutral-700">{benefit}</span>
                                </li>
                            ))}
                        </ul>
                    </div>

                    <div className="bg-neutral-100 rounded-2xl border border-neutral-200 p-8">
                        <h3 className="text-xl font-semibold text-neutral-900 mb-6">
                            Team by Department
                        </h3>
                        <div className="space-y-4">
                            {siteConfig.careers.departments.map((dept) => (
                                <div key={dept.name} className="flex justify-between items-center p-4 bg-neutral-50 rounded-xl">
                                    <div>
                                        <div className="font-medium text-neutral-900">{dept.name}</div>
                                        <div className="text-sm text-neutral-500">{dept.description}</div>
                                    </div>
                                    {dept.openRoles > 0 && (
                                        <span className="px-3 py-1 bg-gold-600 text-white text-sm rounded-full">
                                            {dept.openRoles} open
                                        </span>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </Section>

            {/* Open Positions */}
            <Section id="openings">
                <div className="text-center mb-12">
                    <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-4">
                        Open Positions
                    </h2>
                    <p className="text-neutral-600">
                        Find your next role. All positions are remote-friendly unless otherwise noted.
                    </p>
                </div>

                {Object.entries(jobsByDepartment).map(([department, jobs]) => (
                    jobs.length > 0 && (
                        <div key={department} className="mb-12">
                            <h3 className="text-xl font-semibold text-neutral-900 mb-6 flex items-center">
                                {department}
                                <span className="ml-3 px-2 py-1 bg-neutral-100 text-neutral-600 text-sm rounded-full">
                                    {jobs.length}
                                </span>
                            </h3>
                            <div className="grid md:grid-cols-2 gap-4">
                                {jobs.map((job) => (
                                    <JobCard key={job.id} job={job} />
                                ))}
                            </div>
                        </div>
                    )
                ))}

                {/* Don't see your role? */}
                <div className="bg-neutral-50 rounded-2xl p-8 text-center mt-12">
                    <h3 className="text-xl font-semibold text-neutral-900 mb-3">
                        Don't see your role?
                    </h3>
                    <p className="text-neutral-600 mb-6 max-w-lg mx-auto">
                        We're always looking for talented people. Send us your resume
                        and tell us how you'd like to contribute.
                    </p>
                    <Button href={`mailto:${siteConfig.company.careersEmail}`} variant="outline">
                        Send Your Resume
                    </Button>
                </div>
            </Section>

            {/* Hiring Process */}
            <Section background="light">
                <div className="text-center mb-12">
                    <h2 className="text-3xl font-bold text-neutral-900 mb-4">
                        Our Hiring Process
                    </h2>
                    <p className="text-neutral-600 max-w-2xl mx-auto">
                        We've designed our process to be respectful of your time while
                        giving us the information we need to make great hiring decisions.
                    </p>
                </div>

                <div className="grid md:grid-cols-4 gap-8">
                    {[
                        {
                            step: "1",
                            title: "Application",
                            description: "Submit your resume and tell us why you're interested in DharmaMind.",
                            duration: "5 min"
                        },
                        {
                            step: "2",
                            title: "Intro Call",
                            description: "Chat with our recruiting team about your background and interests.",
                            duration: "30 min"
                        },
                        {
                            step: "3",
                            title: "Technical/Skills",
                            description: "Depending on the role, a technical interview or skills assessment.",
                            duration: "1-2 hours"
                        },
                        {
                            step: "4",
                            title: "Team Interview",
                            description: "Meet your potential teammates and learn more about the role.",
                            duration: "1-2 hours"
                        },
                    ].map((phase) => (
                        <div key={phase.step} className="text-center">
                            <div className="w-12 h-12 rounded-full bg-gold-600 text-white flex items-center justify-center text-xl font-bold mx-auto mb-4">
                                {phase.step}
                            </div>
                            <h3 className="text-lg font-semibold text-neutral-900 mb-2">
                                {phase.title}
                            </h3>
                            <p className="text-neutral-600 text-sm mb-2">
                                {phase.description}
                            </p>
                            <span className="text-xs text-neutral-500">~{phase.duration}</span>
                        </div>
                    ))}
                </div>
            </Section>

            {/* Location */}
            <Section>
                <div className="grid md:grid-cols-2 gap-12 items-center">
                    <div>
                        <h2 className="text-3xl font-bold text-neutral-900 mb-6">
                            Work From Anywhere
                        </h2>
                        <p className="text-neutral-600 mb-6">
                            We're a remote-first company with team members across the globe.
                            We believe great work can happen anywhere, and we've built our
                            culture around asynchronous communication and intentional collaboration.
                        </p>
                        <p className="text-neutral-600 mb-6">
                            For those who prefer office life, we have a headquarters in
                            {siteConfig.company.headquarters} with optional in-person collaboration days.
                        </p>
                        <div className="flex items-center space-x-2 text-neutral-500">
                            <span>📍</span>
                            <span>{siteConfig.company.headquarters}</span>
                        </div>
                    </div>

                    <div className="bg-gradient-to-br from-neutral-100 to-neutral-200 rounded-2xl aspect-video flex items-center justify-center">
                        <span className="text-6xl">🌍</span>
                    </div>
                </div>
            </Section>

            {/* CTA */}
            <Section background="dark">
                <div className="text-center">
                    <h2 className="text-3xl md:text-4xl font-bold mb-4">
                        Ready to Make an Impact?
                    </h2>
                    <p className="text-xl text-neutral-300 mb-8 max-w-2xl mx-auto">
                        Join us in building AI that serves humanity's highest aspirations.
                        Your work will touch millions of lives.
                    </p>
                    <div className="flex flex-col sm:flex-row gap-4 justify-center">
                        <Button href="#openings" variant="secondary" size="lg">
                            Browse Open Roles
                        </Button>
                        <Button
                            href={`mailto:${siteConfig.company.careersEmail}`}
                            variant="outline"
                            size="lg"
                            className="border-white text-white hover:bg-neutral-100 hover:text-neutral-900"
                        >
                            Contact Recruiting
                        </Button>
                    </div>
                </div>
            </Section>
        </Layout>
    );
}
