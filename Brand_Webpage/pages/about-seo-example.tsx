import Head from 'next/head';
import SEOHead from '../components/SEOHead';

export default function About() {
  return (
    <>
      <SEOHead 
        title="About DharmaMind"
        description="Learn about DharmaMind's mission to combine ancient wisdom with modern AI technology. Discover how we're making spiritual guidance accessible through artificial intelligence."
        keywords="about dharmamind, AI spirituality, dharma technology, wisdom AI, spiritual AI platform, universal wisdom"
      />

      <div className="min-h-screen bg-primary-background-main">
        {/* Your existing about content */}
        <div className="container mx-auto px-4 py-16">
          <h1 className="text-4xl font-bold text-primary-text mb-8">
            About DharmaMind
          </h1>
          <div className="text-lg text-primary-text-muted">
            <p className="mb-6">
              DharmaMind represents the harmonious fusion of ancient wisdom and cutting-edge artificial intelligence. 
              Our platform bridges the gap between timeless spiritual teachings and modern technology, making profound 
              insights accessible to seekers worldwide.
            </p>
            <p className="mb-6">
              Founded on the principles of dharma - the path of righteousness and spiritual truth - our AI is trained 
              not just on data, but on the accumulated wisdom of generations of spiritual teachers, philosophers, and 
              enlightened beings from various traditions.
            </p>
            <p>
              We believe that technology, when guided by wisdom and compassion, can serve as a powerful tool for 
              personal growth, inner peace, and collective enlightenment.
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
