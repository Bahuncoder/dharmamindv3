import React from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import Logo from '../components/Logo';
import { ContactButton } from '../components/CentralizedSupport';

const DocsPage: React.FC = () => {
  const router = useRouter();

  const docSections = [
    {
      title: '🚀 Getting Started',
      items: [
        { title: 'Quick Start Guide', description: 'Get up and running with DharmaMind in minutes' },
        { title: 'Account Setup', description: 'Create and configure your DharmaMind account' },
        { title: 'First Conversation', description: 'Learn how to interact with our AI' }
      ]
    },
    {
      title: '💬 Using DharmaMind',
      items: [
        { title: 'Chat Interface', description: 'Navigate and use the chat interface effectively' },
        { title: 'Wisdom Modules', description: 'Understanding our 32-dimensional wisdom architecture' },
        { title: 'Best Practices', description: 'Tips for meaningful spiritual conversations' }
      ]
    },
    {
      title: '🔧 API Documentation',
      items: [
        { title: 'REST API Reference', description: 'Complete API endpoints and parameters' },
        { title: 'Authentication', description: 'Secure API access using JWT tokens' },
        { title: 'Rate Limits', description: 'Understanding API usage limits and guidelines' }
      ]
    },
    {
      title: '⚙️ Advanced Features',
      items: [
        { title: 'Enterprise Integration', description: 'SSO, LDAP, and enterprise features' },
        { title: 'Custom Deployments', description: 'On-premise and private cloud options' },
        { title: 'Webhooks', description: 'Event-driven integrations with your systems' }
      ]
    }
  ];

  return (
    <>
      <Head>
        <title>Documentation - DharmaMind</title>
        <meta name="description" content="Complete documentation for DharmaMind AI platform including API reference, guides, and tutorials" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-primary-background">
        {/* Header */}
        <header className="border-b border-gray-200 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <button 
                onClick={() => router.push('/')}
                className="hover:opacity-80 transition-opacity"
              >
                <Logo size="sm" showText={true} />
              </button>

              <nav className="flex items-center space-x-8">
                <button 
                  onClick={() => router.push('/')}
                  className="text-secondary hover:text-primary text-sm font-medium"
                >
                  Home
                </button>
                <button 
                  onClick={() => router.push('/help')}
                  className="text-secondary hover:text-primary text-sm font-medium"
                >
                  Help
                </button>
                <ContactButton 
                  variant="link"
                  prefillCategory="support"
                  className="text-secondary hover:text-primary text-sm font-medium"
                >
                  Contact
                </ContactButton>
                <button 
                  onClick={() => router.push('/login')}
                  className="btn-primary px-4 py-2 rounded-lg text-sm font-medium"
                >
                  Sign In
                </button>
              </nav>
            </div>
          </div>
        </header>

        {/* Hero Section */}
        <div className="bg-primary-clean py-16">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <h1 className="text-4xl font-bold text-primary mb-4">
              📚 Documentation
            </h1>
            <p className="text-xl text-secondary mb-8">
              Everything you need to integrate and use DharmaMind
            </p>
            
            {/* Quick Links */}
            <div className="flex flex-wrap justify-center gap-4">
              <button className="bg-white px-6 py-3 rounded-lg shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
                <div className="text-primary font-medium">Quick Start</div>
                <div className="text-sm text-secondary">Get started in 5 minutes</div>
              </button>
              <button className="bg-white px-6 py-3 rounded-lg shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
                <div className="text-primary font-medium">API Reference</div>
                <div className="text-sm text-secondary">Complete API docs</div>
              </button>
              <button className="bg-white px-6 py-3 rounded-lg shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
                <div className="text-primary font-medium">Examples</div>
                <div className="text-sm text-secondary">Code samples & tutorials</div>
              </button>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          
          {/* Documentation Sections */}
          <div className="space-y-12">
            {docSections.map((section, sectionIndex) => (
              <div key={sectionIndex}>
                <h2 className="text-2xl font-bold text-primary mb-8">{section.title}</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {section.items.map((item, itemIndex) => (
                    <div key={itemIndex} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow cursor-pointer">
                      <h3 className="font-semibold text-primary mb-2">{item.title}</h3>
                      <p className="text-secondary text-sm mb-4">{item.description}</p>
                      <div className="flex items-center text-primary text-sm font-medium">
                        Read more
                        <svg className="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* API Quick Reference */}
          <div className="mt-16 bg-primary-clean rounded-lg p-8 border border-gray-200">
            <h2 className="text-2xl font-bold mb-6 text-primary">🔗 API Quick Reference</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              
              {/* Authentication */}
              <div>
                <h3 className="text-lg font-semibold mb-4 text-primary">Authentication</h3>
                <div className="bg-gray-800 rounded-lg p-4">
                  <code className="text-sm">
                    <div className="text-gray-400">// JWT Token Authentication</div>
                    <div className="text-white">headers: {`{`}</div>
                    <div className="text-gray-300 ml-4">'Authorization': 'Bearer YOUR_JWT_TOKEN'</div>
                    <div className="text-white">{`}`}</div>
                  </code>
                </div>
              </div>

              {/* Chat Endpoint */}
              <div>
                <h3 className="text-lg font-semibold mb-4 text-primary">Chat Endpoint</h3>
                <div className="bg-gray-800 rounded-lg p-4">
                  <code className="text-sm">
                    <div className="text-gray-400">// Send message to DharmaMind</div>
                    <div className="text-green-400">POST</div>
                    <div className="text-white">/api/chat</div>
                    <div className="text-gray-300 mt-2">{`{ "message": "Your question" }`}</div>
                  </code>
                </div>
              </div>
            </div>

            <div className="mt-6 flex items-center space-x-4">
              <button className="btn-primary px-4 py-2 rounded-lg font-medium">
                View Full API Docs
              </button>
              <button className="btn-outline px-4 py-2 rounded-lg font-medium">
                Download Postman Collection
              </button>
            </div>
          </div>

          {/* SDK Information */}
          <div className="mt-16">
            <h2 className="text-2xl font-bold text-primary mb-8">🛠️ SDKs & Libraries</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              
              {/* JavaScript SDK */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 text-center">
                <div className="w-12 h-12 mx-auto mb-4 bg-primary-clean rounded-lg flex items-center justify-center border border-gray-200">
                  <span className="text-2xl">📜</span>
                </div>
                <h3 className="font-semibold text-primary mb-2">JavaScript</h3>
                <p className="text-secondary text-sm mb-4">npm install dharmamind-js</p>
                <button className="text-primary font-medium text-sm hover:text-secondary">
                  View Docs →
                </button>
              </div>

              {/* Python SDK */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 text-center">
                <div className="w-12 h-12 mx-auto mb-4 bg-blue-100 rounded-lg flex items-center justify-center">
                  <span className="text-2xl">🐍</span>
                </div>
                <h3 className="font-semibold text-primary mb-2">Python</h3>
                <p className="text-secondary text-sm mb-4">pip install dharmamind</p>
                <button className="text-emerald-600 font-medium text-sm hover:text-emerald-700">
                  View Docs →
                </button>
              </div>

              {/* React Components */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 text-center">
                <div className="w-12 h-12 mx-auto mb-4 bg-cyan-100 rounded-lg flex items-center justify-center">
                  <span className="text-2xl">⚛️</span>
                </div>
                <h3 className="font-semibold text-primary mb-2">React</h3>
                <p className="text-secondary text-sm mb-4">npm install @dharmamind/react</p>
                <button className="text-emerald-600 font-medium text-sm hover:text-emerald-700">
                  View Docs →
                </button>
              </div>

              {/* REST API */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 text-center">
                <div className="w-12 h-12 mx-auto mb-4 bg-green-100 rounded-lg flex items-center justify-center">
                  <span className="text-2xl">🔌</span>
                </div>
                <h3 className="font-semibold text-primary mb-2">REST API</h3>
                <p className="text-secondary text-sm mb-4">Direct HTTP integration</p>
                <button className="text-emerald-600 font-medium text-sm hover:text-emerald-700">
                  View Docs →
                </button>
              </div>
            </div>
          </div>

          {/* Support Section */}
          <div className="mt-16 bg-primary-clean rounded-lg p-8 text-center border border-gray-200">
            <h3 className="text-xl font-semibold text-primary mb-4">
              Need help with integration? 🤝
            </h3>
            <p className="text-secondary mb-6">
              Our developer support team is here to help you get integrated quickly.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <ContactButton
                variant="button"
                prefillCategory="support"
                className="btn-primary px-6 py-3 rounded-lg font-medium"
              >
                Contact Developer Support
              </ContactButton>
              <button
                onClick={() => router.push('/help')}
                className="border border-gray-300 text-gray-700 px-6 py-3 rounded-lg font-medium hover:bg-gray-50 transition-colors"
              >
                Browse Help Center
              </button>
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="border-t border-gray-200 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <div className="text-center">
              <button 
                onClick={() => router.push('/')}
                className="flex justify-center mx-auto mb-4 hover:opacity-80 transition-opacity"
              >
                <Logo size="sm" showText={true} />
              </button>
              <p className="text-sm text-secondary">
                © 2025 DharmaMind. All rights reserved.
              </p>
            </div>
          </div>
        </footer>
      </div>
    </>
  );
};

export default DocsPage;
