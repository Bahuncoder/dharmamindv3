import SEOHead from '../components/SEOHead';

export default function Enterprise() {
  return (
    <>
      <SEOHead 
        title="Enterprise Solutions - Purpose-Driven Employee Engagement"
        description="Transform your workplace with DharmaMind's ethical AI solutions. Improve employee retention, workplace well-being, and corporate culture with purpose-driven engagement tools."
        keywords="purpose-driven employee engagement, workplace well-being solutions, ethical leadership training, corporate culture improvement, employee retention strategies, AI for professional development, mindfulness for corporate teams, workplace purpose alignment"
      />

      <div className="min-h-screen bg-primary-background-main">
        <div className="container mx-auto px-4 py-16">
          <h1 className="text-4xl font-bold text-primary-text mb-8">
            Enterprise Solutions for Purpose-Driven Organizations
          </h1>
          <div className="text-lg text-primary-text-muted">
            <p className="mb-6">
              Transform your workplace culture with DharmaMind's ethical AI solutions designed for 
              purpose-driven employee engagement and workplace well-being.
            </p>
            
            <h2 className="text-2xl font-semibold text-primary-text mb-4">
              Our Enterprise Offerings:
            </h2>
            
            <ul className="space-y-4 mb-8">
              <li className="flex items-start">
                <span className="text-brand-accent mr-3">•</span>
                <div>
                  <strong>Ethical Leadership Training:</strong> AI-powered coaching for conscious leadership development
                </div>
              </li>
              <li className="flex items-start">
                <span className="text-brand-accent mr-3">•</span>
                <div>
                  <strong>Employee Retention Strategies:</strong> Purpose alignment tools to reduce turnover
                </div>
              </li>
              <li className="flex items-start">
                <span className="text-brand-accent mr-3">•</span>
                <div>
                  <strong>Workplace Well-being Solutions:</strong> Stress management and mindfulness programs
                </div>
              </li>
              <li className="flex items-start">
                <span className="text-brand-accent mr-3">•</span>
                <div>
                  <strong>Corporate Culture Improvement:</strong> Values-based decision making frameworks
                </div>
              </li>
            </ul>

            <p className="mb-6">
              Our AI for professional development helps create mindful corporate teams aligned with 
              meaningful work and conscious action.
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
