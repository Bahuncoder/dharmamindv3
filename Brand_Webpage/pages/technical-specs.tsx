import React, { useState } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import Logo from '../components/Logo';
import { ContactButton } from '../components/CentralizedSupport';

interface DownloadItem {
  id: string;
  title: string;
  description: string;
  fileSize: string;
  icon: string;
  category: 'api' | 'security' | 'integration' | 'overview';
}

const TechnicalSpecsPage: React.FC = () => {
  const router = useRouter();
  const [contactInfo, setContactInfo] = useState({
    name: '',
    email: '',
    company: ''
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [hasAccess, setHasAccess] = useState(false);

  const downloadItems: DownloadItem[] = [
    {
      id: 'api-reference',
      title: 'DharmaMind API Reference',
      description: 'Complete REST API documentation with endpoints, authentication, and response schemas',
      fileSize: '2.4 MB',
      icon: '🔌',
      category: 'api'
    },
    {
      id: 'security-whitepaper',
      title: 'Security & Compliance Whitepaper',
      description: 'Detailed security architecture, encryption standards, and compliance certifications',
      fileSize: '1.8 MB',
      icon: '🔒',
      category: 'security'
    },
    {
      id: 'integration-guide',
      title: 'Enterprise Integration Guide',
      description: 'Step-by-step integration guide for SSO, LDAP, and enterprise systems',
      fileSize: '3.1 MB',
      icon: '🔗',
      category: 'integration'
    },
    {
      id: 'architecture-overview',
      title: 'System Architecture Overview',
      description: 'Technical architecture diagrams and infrastructure requirements',
      fileSize: '2.7 MB',
      icon: '🏗️',
      category: 'overview'
    },
    {
      id: 'sdk-documentation',
      title: 'JavaScript/Python SDK Documentation',
      description: 'Client libraries and SDK documentation for custom integrations',
      fileSize: '1.5 MB',
      icon: '⚙️',
      category: 'api'
    },
    {
      id: 'deployment-guide',
      title: 'Cloud Deployment Guide',
      description: 'AWS, Azure, and GCP deployment configurations and best practices',
      fileSize: '2.2 MB',
      icon: '☁️',
      category: 'integration'
    },
    {
      id: 'performance-benchmarks',
      title: 'Performance & Scalability Report',
      description: 'Load testing results, performance metrics, and scaling recommendations',
      fileSize: '1.9 MB',
      icon: '📊',
      category: 'overview'
    },
    {
      id: 'compliance-report',
      title: 'GDPR & SOC 2 Compliance Report',
      description: 'Detailed compliance documentation and certification reports',
      fileSize: '3.3 MB',
      icon: '📋',
      category: 'security'
    }
  ];

  const handleContactSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    // Simulate API call
    setTimeout(() => {
      setIsSubmitting(false);
      setHasAccess(true);
    }, 2000);
  };

  const handleDownload = (item: DownloadItem) => {
    // Simulate download
    const link = document.createElement('a');
    link.href = '#';
    link.download = `${item.title.replace(/\s+/g, '_')}.pdf`;
    link.click();
    
    // Show download notification
    alert(`Downloading ${item.title}...`);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setContactInfo({
      ...contactInfo,
      [e.target.name]: e.target.value
    });
  };

  const getCategoryColor = (category: string) => {
    // Use centralized color system for all categories
    return 'bg-primary-clean text-primary border border-stone-200';
  };

  return (
    <div className="min-h-screen bg-primary-background">
      <Head>
        <title>Technical Specifications - DharmaMind Enterprise</title>
        <meta name="description" content="Download comprehensive technical documentation and specifications for DharmaMind Enterprise" />
      </Head>

      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-stone-200/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Logo 
              size="sm"
              onClick={() => router.push('/')}
            />
            <div className="flex items-center space-x-4">
              <button 
                onClick={() => router.push('/demo-request')}
                className="text-secondary hover:text-primary font-medium"
              >
                Schedule Demo
              </button>
              <button 
                onClick={() => router.push('/')}
                className="text-secondary hover:text-primary"
              >
                Back to Home
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            📚 Technical Specifications & Documentation
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Access comprehensive technical documentation, API references, security whitepapers, 
            and integration guides for DharmaMind Enterprise.
          </p>
        </div>

        {/* Access Form or Downloads */}
        {!hasAccess ? (
          <div className="max-w-2xl mx-auto">
            <div className="bg-white rounded-lg shadow-lg border border-stone-200 p-8">
              <div className="text-center mb-6">
                <div className="w-16 h-16 bg-primary-clean rounded-full flex items-center justify-center mx-auto mb-4 border border-stone-200">
                  <svg className="w-8 h-8 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path>
                  </svg>
                </div>
                <h2 className="text-2xl font-bold text-primary mb-2">
                  Access Technical Documentation
                </h2>
                <p className="text-secondary">
                  Provide your business information to download our complete technical documentation package.
                </p>
              </div>

              <form onSubmit={handleContactSubmit} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Full Name *
                  </label>
                  <input
                    type="text"
                    name="name"
                    required
                    value={contactInfo.name}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Business Email *
                  </label>
                  <input
                    type="email"
                    name="email"
                    required
                    value={contactInfo.email}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Company Name *
                  </label>
                  <input
                    type="text"
                    name="company"
                    required
                    value={contactInfo.company}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                  />
                </div>
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className={`w-full bg-emerald-600 text-white px-6 py-3 rounded-lg font-medium transition-all duration-300 ${
                    isSubmitting 
                      ? 'opacity-50 cursor-not-allowed' 
                      : 'hover:bg-emerald-700'
                  }`}
                >
                  {isSubmitting ? (
                    <div className="flex items-center justify-center">
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                      Verifying Access...
                    </div>
                  ) : (
                    'Get Access to Technical Docs'
                  )}
                </button>
              </form>

              <div className="mt-6 p-4 bg-gray-50 rounded-lg">
                <h3 className="font-semibold text-gray-900 mb-2">What's included:</h3>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• Complete API documentation and SDKs</li>
                  <li>• Security architecture and compliance reports</li>
                  <li>• Integration guides for enterprise systems</li>
                  <li>• Performance benchmarks and scalability data</li>
                  <li>• Deployment guides for cloud platforms</li>
                </ul>
              </div>
            </div>
          </div>
        ) : (
          <div>
            {/* Filter Buttons */}
            <div className="flex flex-wrap justify-center gap-2 mb-8">
              {['all', 'api', 'security', 'integration', 'overview'].map((filter) => (
                <button
                  key={filter}
                  className={`px-4 py-2 rounded-lg font-medium transition-all duration-300 ${
                    filter === 'all' 
                      ? 'bg-emerald-600 text-white' 
                      : 'bg-white text-gray-700 border border-gray-200 hover:bg-gray-50'
                  }`}
                >
                  {filter === 'all' ? 'All Documents' : filter.charAt(0).toUpperCase() + filter.slice(1)}
                </button>
              ))}
            </div>

            {/* Download Grid */}
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {downloadItems.map((item) => (
                <div key={item.id} className="bg-white rounded-lg border border-stone-200 p-6 hover:shadow-lg transition-all duration-300">
                  <div className="flex items-start justify-between mb-4">
                    <div className="text-3xl">{item.icon}</div>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getCategoryColor(item.category)}`}>
                      {item.category}
                    </span>
                  </div>
                  
                  <h3 className="font-semibold text-gray-900 mb-2">{item.title}</h3>
                  <p className="text-sm text-gray-600 mb-4 line-clamp-2">{item.description}</p>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">{item.fileSize}</span>
                    <button
                      onClick={() => handleDownload(item)}
                      className="bg-emerald-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-emerald-700 transition-all duration-300"
                    >
                      Download
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {/* Additional Resources */}
            <div className="mt-12 bg-white rounded-lg border border-stone-200 p-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">
                🚀 Ready to Get Started?
              </h2>
              
              <div className="grid md:grid-cols-3 gap-6">
                <div className="text-center p-6 bg-primary-clean rounded-lg border border-stone-200">
                  <div className="text-3xl mb-3">🎯</div>
                  <h3 className="font-semibold text-primary mb-2">Schedule a Demo</h3>
                  <p className="text-sm text-secondary mb-4">
                    See DharmaMind in action with a personalized demo
                  </p>
                  <button
                    onClick={() => router.push('/demo-request')}
                    className="btn-primary px-4 py-2 rounded-lg text-sm font-medium"
                  >
                    Book Demo
                  </button>
                </div>
                
                <div className="text-center p-6 bg-primary-clean rounded-lg border border-stone-200">
                  <div className="text-3xl mb-3">💬</div>
                  <h3 className="font-semibold text-primary mb-2">Ask Questions</h3>
                  <p className="text-sm text-secondary mb-4">
                    Get technical support from our engineering team
                  </p>
                  <ContactButton
                    variant="button"
                    prefillCategory="support"
                    className="btn-primary px-4 py-2 rounded-lg text-sm font-medium"
                  >
                    Contact Support
                  </ContactButton>
                </div>
                
                <div className="text-center p-6 bg-primary-clean rounded-lg border border-stone-200">
                  <div className="text-3xl mb-3">🛠️</div>
                  <h3 className="font-semibold text-primary mb-2">Start Building</h3>
                  <p className="text-sm text-secondary mb-4">
                    Try our API sandbox environment
                  </p>
                  <button
                    className="btn-primary px-4 py-2 rounded-lg text-sm font-medium"
                  >
                    API Sandbox
                  </button>
                </div>
              </div>
            </div>

            {/* Success Message */}
            <div className="mt-8 bg-primary-clean border border-stone-200 rounded-lg p-6">
              <div className="flex items-center">
                <div className="w-8 h-8 bg-primary-clean border border-stone-200 rounded-full flex items-center justify-center mr-4">
                  <svg className="w-5 h-5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                  </svg>
                </div>
                <div>
                  <h3 className="font-semibold text-primary">Access Granted!</h3>
                  <p className="text-secondary text-sm">
                    You now have access to all technical documentation. Download any files you need for your evaluation.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TechnicalSpecsPage;
