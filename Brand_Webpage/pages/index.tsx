import Link from 'next/link';
import { Layout, Section, StatCard, Button, TestimonialCard } from '../components';
import { siteConfig } from '../config/site.config';

export default function Home() {
  return (
    <Layout>
      {/* Hero Section */}
      <Section background="white" padding="xl" className="relative overflow-hidden">
        {/* Background gradient */}
        <div className="absolute inset-0 bg-gradient-to-b from-neutral-50 to-white -z-10" />

        <div className="text-center max-w-4xl mx-auto">
          {/* Announcement Banner */}
          <Link
            href="/blog/introducing-dharmallm-2"
            className="inline-flex items-center px-4 py-2 bg-gold-100 rounded-full text-sm text-gold-700 mb-8 hover:bg-gold-200 transition-colors"
          >
            <span className="bg-gold-600 text-white text-xs px-2 py-0.5 rounded-full mr-2">New</span>
            Introducing DharmaLLM 2.0 →
          </Link>

          <h1 className="text-5xl md:text-7xl font-bold text-neutral-900 mb-6 leading-tight">
            {siteConfig.company.tagline}
          </h1>

          <p className="text-xl md:text-2xl text-neutral-600 mb-10 leading-relaxed max-w-2xl mx-auto">
            {siteConfig.company.description}
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button href="https://chat.dharmamind.ai" size="lg" external>
              Try DharmaMind Chat
            </Button>
            <Button href="/research" variant="outline" size="lg">
              Explore Our Research
            </Button>
          </div>
        </div>
      </Section>

      {/* Trusted By Section */}
      <Section padding="md" className="border-y border-neutral-100">
        <div className="text-center">
          <p className="text-sm font-medium text-neutral-400 uppercase tracking-wider mb-8">
            Trusted by organizations worldwide
          </p>
          <div className="flex flex-wrap items-center justify-center gap-8 md:gap-12 opacity-60 grayscale hover:opacity-100 hover:grayscale-0 transition-all duration-500">
            {/* Partner/Customer Logos - Using text placeholders that can be replaced with real logos */}
            {[
              { name: "Stanford", icon: "🎓" },
              { name: "MIT", icon: "🔬" },
              { name: "Google", icon: "🔍" },
              { name: "Microsoft", icon: "💻" },
              { name: "Harvard", icon: "📚" },
              { name: "OpenAI", icon: "🤖" },
            ].map((partner) => (
              <div key={partner.name} className="flex items-center space-x-2 text-neutral-500">
                <span className="text-2xl">{partner.icon}</span>
                <span className="text-lg font-semibold">{partner.name}</span>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* Mission Statement */}
      <Section background="dark">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl md:text-5xl font-bold mb-6 leading-tight">
            {siteConfig.company.mission}
          </h2>
          <div className="flex justify-center">
            <Link href="/company" className="text-neutral-400 hover:text-white transition-colors inline-flex items-center">
              Learn about our approach →
            </Link>
          </div>
        </div>
      </Section>

      {/* Stats */}
      <Section background="light">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          {siteConfig.stats.hero.map((stat) => (
            <StatCard
              key={stat.label}
              value={stat.value}
              label={stat.label}
              description={stat.description}
            />
          ))}
        </div>
      </Section>

      {/* Products Section */}
      <Section>
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-4">Our Products</h2>
          <p className="text-xl text-neutral-600 max-w-2xl mx-auto">
            Tools for spiritual growth, powered by ethical AI.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {siteConfig.products.featured.map((product) => (
            <Link
              key={product.id}
              href={product.href}
              className="group p-8 rounded-2xl border border-neutral-200 bg-neutral-100 hover:shadow-xl hover:border-gold-400 transition-all"
            >
              <span className="text-4xl mb-4 block">{product.icon}</span>
              <h3 className="text-xl font-semibold text-neutral-900 mb-2 group-hover:text-gold-600">
                {product.name}
              </h3>
              <p className="text-neutral-500 text-sm mb-3">{product.tagline}</p>
              <p className="text-neutral-600 mb-4">{product.description}</p>
              <span className="text-gold-600 font-medium inline-flex items-center">
                {product.cta} <span className="ml-2 group-hover:translate-x-1 transition-transform">→</span>
              </span>
            </Link>
          ))}
        </div>

        <div className="text-center mt-12">
          <Button href="/products" variant="outline">
            Explore All Products
          </Button>
        </div>
      </Section>

      {/* Research Highlight */}
      <Section background="light">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          <div>
            <span className="text-sm font-medium text-neutral-500 uppercase tracking-wider mb-4 block">Research</span>
            <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-6">
              Advancing the Science of Spiritual AI
            </h2>
            <p className="text-neutral-600 mb-6">
              Our research team is pioneering new approaches to building AI systems that
              understand the depth and nuance of spiritual traditions while maintaining
              the highest ethical standards.
            </p>
            <div className="space-y-4 mb-8">
              {siteConfig.research.areas.slice(0, 3).map((area) => (
                <div key={area.id} className="flex items-center">
                  <span className="text-2xl mr-3">{area.icon}</span>
                  <div>
                    <div className="font-medium text-neutral-900">{area.title}</div>
                    <div className="text-sm text-neutral-500">{area.papers} papers published</div>
                  </div>
                </div>
              ))}
            </div>
            <Button href="/research">View All Research</Button>
          </div>

          <div className="bg-neutral-100 rounded-2xl border border-neutral-200 p-8">
            <h3 className="font-semibold text-neutral-900 mb-6">Latest Publication</h3>
            {siteConfig.research.publications.filter(p => p.featured).slice(0, 1).map((paper) => (
              <div key={paper.title}>
                <h4 className="text-lg font-medium text-neutral-900 mb-2">{paper.title}</h4>
                <p className="text-sm text-neutral-600 mb-4">{paper.authors.join(', ')}</p>
                <p className="text-neutral-500 text-sm mb-4">{paper.abstract}</p>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-neutral-500">{paper.venue}</span>
                  <Button href={paper.link} variant="ghost" size="sm">Read Paper →</Button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* Testimonials */}
      <Section>
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-4">
            What People Are Saying
          </h2>
          <p className="text-neutral-600">
            Hear from practitioners, teachers, and seekers using DharmaMind.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {siteConfig.testimonials.map((testimonial, index) => (
            <TestimonialCard key={index} testimonial={testimonial} />
          ))}
        </div>
      </Section>

      {/* Safety Section */}
      <Section background="light">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          <div className="order-2 md:order-1">
            <div className="grid grid-cols-2 gap-4">
              {siteConfig.safety.principles.slice(0, 4).map((principle) => (
                <div
                  key={principle.title}
                  className="bg-neutral-100 rounded-xl border border-neutral-200 p-4"
                >
                  <span className="text-2xl mb-2 block">{principle.icon}</span>
                  <h4 className="font-medium text-neutral-900 text-sm">{principle.title}</h4>
                </div>
              ))}
            </div>
          </div>

          <div className="order-1 md:order-2">
            <span className="text-sm font-medium text-neutral-500 uppercase tracking-wider mb-4 block">Safety</span>
            <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-6">
              Built on Ethical Foundations
            </h2>
            <p className="text-neutral-600 mb-6">
              We believe AI in spiritual contexts requires the highest ethical standards.
              Our systems are designed with safety, respect, and user wellbeing at their core.
            </p>
            <Button href="/safety">Read Our Safety Principles</Button>
          </div>
        </div>
      </Section>

      {/* Enterprise Section */}
      <Section>
        <div className="bg-neutral-200 border border-neutral-300 rounded-3xl p-8 md:p-16">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <span className="text-sm font-medium text-gold-600 uppercase tracking-wider mb-4 block">Enterprise</span>
              <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-6">
                {siteConfig.company.name} for Organizations
              </h2>
              <p className="text-neutral-600 mb-8">
                Bring the power of ethical spiritual AI to your wellness organization,
                educational institution, or healthcare provider. Custom solutions,
                dedicated support, and enterprise-grade security.
              </p>
              <div className="flex flex-col sm:flex-row gap-4">
                <Button href="/products#enterprise" variant="primary">
                  Learn More
                </Button>
                <Button
                  href={`mailto:${siteConfig.company.email}`}
                  variant="outline"
                >
                  Contact Sales
                </Button>
              </div>
            </div>

            <div className="hidden md:block">
              <div className="grid grid-cols-2 gap-4 text-center">
                {siteConfig.products.useCases.slice(0, 4).map((useCase) => (
                  <div key={useCase.title} className="bg-neutral-100 border border-neutral-200 rounded-xl p-4">
                    <span className="text-3xl mb-2 block">{useCase.icon}</span>
                    <div className="text-sm font-medium text-neutral-700">{useCase.title}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </Section>

      {/* Press/Trust Section */}
      <Section background="light">
        <div className="text-center">
          <p className="text-sm text-neutral-500 uppercase tracking-wider mb-8">As Featured In</p>
          <div className="flex flex-wrap justify-center items-center gap-8 opacity-50">
            {siteConfig.press.coverage.map((item) => (
              <span key={item.outlet} className="text-2xl font-semibold text-neutral-400">
                {item.outlet}
              </span>
            ))}
          </div>
        </div>
      </Section>

      {/* Final CTA */}
      <Section>
        <div className="text-center max-w-3xl mx-auto">
          <h2 className="text-3xl md:text-5xl font-bold text-neutral-900 mb-6">
            Begin Your Journey
          </h2>
          <p className="text-xl text-neutral-600 mb-10">
            Experience AI that understands the depth of spiritual wisdom.
            Start a conversation that could change your perspective.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button href="https://chat.dharmamind.ai" size="lg" external>
              Try DharmaMind Free
            </Button>
            <Button href="/company" variant="outline" size="lg">
              Learn About Us
            </Button>
          </div>
        </div>
      </Section>
    </Layout>
  );
}
