import React from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import { ContactButton } from '../components/CentralizedSupport';

const ApiDocsPage: React.FC = () => {
  const router = useRouter();

  const endpoints = [
    {
      method: 'POST',
      path: '/api/auth/login',
      description: 'Authenticate user and receive JWT token',
      params: ['email', 'password']
    },
    {
      method: 'POST',
      path: '/api/auth/register',
      description: 'Create new user account',
      params: ['name', 'email', 'password']
    },
    {
      method: 'POST',
      path: '/api/chat',
      description: 'Send message to DharmaMind AI',
      params: ['message', 'conversation_id?']
    },
    {
      method: 'GET',
      path: '/api/conversations',
      description: 'Get user conversation history',
      params: []
    },
    {
      method: 'GET',
      path: '/api/user/profile',
      description: 'Get current user profile',
      params: []
    }
  ];

  const getMethodColor = (method: string) => {
    switch (method) {
      case 'GET':
        return 'bg-green-100 text-green-800';
      case 'POST':
        return 'bg-blue-100 text-blue-800';
      case 'PUT':
        return 'bg-yellow-100 text-yellow-800';
      case 'DELETE':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <>
      <Head>
        <title>API Documentation - DharmaMind</title>
        <meta name="description" content="Complete REST API documentation for DharmaMind platform integration" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="border-b border-gray-200 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <button 
                onClick={() => router.push('/')}
                className="flex items-center space-x-3 hover:opacity-80 transition-opacity"
              >
                <div className="w-8 h-8 bg-gradient-to-r from-amber-600 to-emerald-600 rounded-lg flex items-center justify-center">
                  <span className="text-white text-sm font-bold">🕉</span>
                </div>
                <span className="text-xl font-semibold text-gray-900">DharmaMind API</span>
              </button>

              <nav className="flex items-center space-x-8">
                <button 
                  onClick={() => router.push('/')}
                  className="text-gray-600 hover:text-gray-900 text-sm font-medium"
                >
                  Home
                </button>
                <button 
                  onClick={() => router.push('/docs')}
                  className="text-gray-600 hover:text-gray-900 text-sm font-medium"
                >
                  Documentation
                </button>
                <ContactButton 
                  variant="link"
                  prefillCategory="support"
                  className="text-gray-600 hover:text-gray-900 text-sm font-medium"
                >
                  Contact
                </ContactButton>
                <button 
                  onClick={() => router.push('/auth?mode=login')}
                  className="bg-gradient-to-r from-amber-600 to-emerald-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:from-amber-700 hover:to-emerald-700 transition-all duration-300"
                >
                  Sign In
                </button>
              </nav>
            </div>
          </div>
        </header>

        {/* Hero Section */}
        <div className="bg-gradient-to-br from-amber-50 to-emerald-50 py-16">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <h1 className="text-4xl font-bold text-gray-900 mb-4">
              🔌 DharmaMind API
            </h1>
            <p className="text-xl text-gray-600 mb-8">
              Integrate dharmic AI wisdom into your applications with our REST API
            </p>
            
            <div className="inline-flex items-center px-6 py-3 bg-white rounded-lg shadow-sm border border-gray-200">
              <span className="text-gray-600 mr-2">Base URL:</span>
              <code className="font-mono text-emerald-600 font-medium">
                https://api.dharmamind.com/v1
              </code>
            </div>
          </div>
        </div>

        {/* Quick Start */}
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-8">🚀 Quick Start</h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-16">
            
            {/* Authentication */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h3 className="text-xl font-semibold text-gray-900 mb-4">1. Authentication</h3>
              <p className="text-gray-600 mb-4">
                First, authenticate to receive a JWT token:
              </p>
              <div className="bg-stone-800 rounded-lg p-4 overflow-x-auto">
                <code className="text-green-400 text-sm">
                  <div className="text-blue-400">POST</div>
                  <div className="text-white">/api/auth/login</div>
                  <div className="mt-2 text-gray-300">{`{`}</div>
                  <div className="text-amber-300 ml-4">"email": "user@example.com",</div>
                  <div className="text-amber-300 ml-4">"password": "your_password"</div>
                  <div className="text-gray-300">{`}`}</div>
                </code>
              </div>
            </div>

            {/* Making Requests */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h3 className="text-xl font-semibold text-gray-900 mb-4">2. Making Requests</h3>
              <p className="text-gray-600 mb-4">
                Include the JWT token in your requests:
              </p>
              <div className="bg-stone-800 rounded-lg p-4 overflow-x-auto">
                <code className="text-sm">
                  <div className="text-gray-400">// Headers</div>
                  <div className="text-amber-300">Authorization: Bearer YOUR_JWT_TOKEN</div>
                  <div className="text-amber-300">Content-Type: application/json</div>
                </code>
              </div>
            </div>
          </div>

          {/* API Endpoints */}
          <h2 className="text-3xl font-bold text-gray-900 mb-8">📋 API Endpoints</h2>
          
          <div className="space-y-6">
            {endpoints.map((endpoint, index) => (
              <div key={index} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <div className="flex items-center space-x-4 mb-4">
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${getMethodColor(endpoint.method)}`}>
                    {endpoint.method}
                  </span>
                  <code className="text-lg font-mono text-gray-900">{endpoint.path}</code>
                </div>
                
                <p className="text-gray-600 mb-4">{endpoint.description}</p>
                
                {endpoint.params.length > 0 && (
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">Parameters:</h4>
                    <div className="flex flex-wrap gap-2">
                      {endpoint.params.map((param, paramIndex) => (
                        <code key={paramIndex} className="px-2 py-1 bg-gray-100 text-gray-800 rounded text-sm">
                          {param}
                        </code>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Example Response */}
          <div className="mt-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-8">📄 Example Response</h2>
            
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h3 className="text-xl font-semibold text-gray-900 mb-4">Chat API Response</h3>
              <div className="bg-stone-800 rounded-lg p-6 overflow-x-auto">
                <code className="text-sm">
                  <div className="text-gray-300">{`{`}</div>
                  <div className="text-amber-300 ml-4">"success": true,</div>
                  <div className="text-amber-300 ml-4">"conversation_id": "conv_123456",</div>
                  <div className="text-amber-300 ml-4">"response": {`{`}</div>
                  <div className="text-green-400 ml-8">"message": "Thank you for your question about finding inner peace...",</div>
                  <div className="text-green-400 ml-8">"wisdom_modules": ["moksha", "viveka", "ahimsa"],</div>
                  <div className="text-green-400 ml-8">"timestamp": "2025-01-29T10:30:00Z"</div>
                  <div className="text-amber-300 ml-4">{`},`}</div>
                  <div className="text-amber-300 ml-4">"usage": {`{`}</div>
                  <div className="text-blue-400 ml-8">"tokens_used": 150,</div>
                  <div className="text-blue-400 ml-8">"remaining_quota": 850</div>
                  <div className="text-amber-300 ml-4">{`}`}</div>
                  <div className="text-gray-300">{`}`}</div>
                </code>
              </div>
            </div>
          </div>

          {/* Rate Limits */}
          <div className="mt-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-8">⚡ Rate Limits</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 text-center">
                <div className="text-3xl mb-3">🆓</div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Free Tier</h3>
                <div className="text-2xl font-bold text-blue-600 mb-1">100</div>
                <div className="text-gray-600 text-sm">requests/hour</div>
              </div>
              
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 text-center">
                <div className="text-3xl mb-3">💼</div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Professional</h3>
                <div className="text-2xl font-bold text-emerald-600 mb-1">1,000</div>
                <div className="text-gray-600 text-sm">requests/hour</div>
              </div>
              
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 text-center">
                <div className="text-3xl mb-3">🏢</div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Enterprise</h3>
                <div className="text-2xl font-bold text-purple-600 mb-1">Custom</div>
                <div className="text-gray-600 text-sm">unlimited available</div>
              </div>
            </div>
          </div>

          {/* Error Codes */}
          <div className="mt-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-8">🚨 Error Codes</h2>
            
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Code</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Description</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Solution</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  <tr>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-red-600">400</td>
                    <td className="px-6 py-4 text-sm text-gray-900">Bad Request</td>
                    <td className="px-6 py-4 text-sm text-gray-600">Check request parameters</td>
                  </tr>
                  <tr>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-red-600">401</td>
                    <td className="px-6 py-4 text-sm text-gray-900">Unauthorized</td>
                    <td className="px-6 py-4 text-sm text-gray-600">Verify JWT token</td>
                  </tr>
                  <tr>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-red-600">429</td>
                    <td className="px-6 py-4 text-sm text-gray-900">Rate Limited</td>
                    <td className="px-6 py-4 text-sm text-gray-600">Reduce request frequency</td>
                  </tr>
                  <tr>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-red-600">500</td>
                    <td className="px-6 py-4 text-sm text-gray-900">Server Error</td>
                    <td className="px-6 py-4 text-sm text-gray-600">Contact support</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* SDKs */}
          <div className="mt-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-8">📦 SDKs & Libraries</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 text-center">
                <div className="text-4xl mb-4">📜</div>
                <h3 className="font-semibold text-gray-900 mb-2">JavaScript</h3>
                <code className="text-sm text-gray-600 mb-4 block">npm install dharmamind</code>
                <button className="text-emerald-600 hover:text-emerald-700 font-medium text-sm">
                  View Docs →
                </button>
              </div>

              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 text-center">
                <div className="text-4xl mb-4">🐍</div>
                <h3 className="font-semibold text-gray-900 mb-2">Python</h3>
                <code className="text-sm text-gray-600 mb-4 block">pip install dharmamind</code>
                <button className="text-emerald-600 hover:text-emerald-700 font-medium text-sm">
                  View Docs →
                </button>
              </div>

              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 text-center">
                <div className="text-4xl mb-4">💎</div>
                <h3 className="font-semibold text-gray-900 mb-2">Ruby</h3>
                <code className="text-sm text-gray-600 mb-4 block">gem install dharmamind</code>
                <button className="text-emerald-600 hover:text-emerald-700 font-medium text-sm">
                  View Docs →
                </button>
              </div>

              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 text-center">
                <div className="text-4xl mb-4">🔌</div>
                <h3 className="font-semibold text-gray-900 mb-2">cURL</h3>
                <code className="text-sm text-gray-600 mb-4 block">Direct HTTP calls</code>
                <button className="text-emerald-600 hover:text-emerald-700 font-medium text-sm">
                  View Examples →
                </button>
              </div>
            </div>
          </div>

          {/* Support */}
          <div className="mt-16 bg-gradient-to-r from-amber-50 to-emerald-50 rounded-lg p-8 text-center">
            <h3 className="text-xl font-semibold text-gray-900 mb-4">
              Need API Support? 🤝
            </h3>
            <p className="text-gray-600 mb-6">
              Our developer support team is here to help you integrate successfully.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <ContactButton
                variant="button"
                prefillCategory="support"
                className="bg-gradient-to-r from-amber-600 to-emerald-600 text-white px-6 py-3 rounded-lg font-medium hover:from-amber-700 hover:to-emerald-700 transition-all duration-300"
              >
                Contact Developers
              </ContactButton>
              <button
                onClick={() => router.push('/docs')}
                className="border border-gray-300 text-gray-700 px-6 py-3 rounded-lg font-medium hover:bg-gray-50 transition-colors"
              >
                View Full Docs
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
                className="flex items-center justify-center space-x-3 mx-auto mb-4 hover:opacity-80 transition-opacity"
              >
                <div className="w-8 h-8 bg-gradient-to-r from-amber-600 to-emerald-600 rounded-lg flex items-center justify-center">
                  <span className="text-white text-sm font-bold">🕉</span>
                </div>
                <span className="text-xl font-semibold text-gray-900">DharmaMind</span>
              </button>
              <p className="text-sm text-gray-600">
                © 2025 DharmaMind. All rights reserved.
              </p>
            </div>
          </div>
        </footer>
      </div>
    </>
  );
};

export default ApiDocsPage;
