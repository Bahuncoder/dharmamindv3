import React from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import Logo from '../components/Logo';
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
<<<<<<< HEAD
        return 'bg-primary-clean text-neutral-900 border border-neutral-300';
      case 'POST':
        return 'bg-primary-clean text-neutral-900 border border-neutral-300';
      case 'PUT':
        return 'bg-primary-clean text-neutral-900 border border-neutral-300';
      case 'DELETE':
        return 'bg-primary-clean text-neutral-900 border border-neutral-300';
      default:
        return 'bg-primary-clean text-neutral-900 border border-neutral-300';
=======
        return 'bg-primary-clean text-primary border border-gray-200';
      case 'POST':
        return 'bg-primary-clean text-primary border border-gray-200';
      case 'PUT':
        return 'bg-primary-clean text-primary border border-gray-200';
      case 'DELETE':
        return 'bg-primary-clean text-primary border border-gray-200';
      default:
        return 'bg-primary-clean text-primary border border-gray-200';
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
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

      <div className="min-h-screen bg-primary-background">
        {/* Header */}
<<<<<<< HEAD
        <header className="border-b border-neutral-300 bg-neutral-100">
=======
        <header className="border-b border-gray-200 bg-white">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center space-x-3">
                <Logo 
                  size="sm"
                  showText={false}
                  onClick={() => router.push('/')}
                  className="hover:opacity-80 transition-opacity"
                />
<<<<<<< HEAD
                <span className="text-xl font-semibold text-neutral-900">DharmaMind API</span>
=======
                <span className="text-xl font-semibold text-primary">DharmaMind API</span>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
              </div>

              <nav className="flex items-center space-x-8">
                <button 
                  onClick={() => router.push('/')}
<<<<<<< HEAD
                  className="text-neutral-600 hover:text-gold-600 text-sm font-medium"
=======
                  className="text-secondary hover:text-primary text-sm font-medium"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                >
                  Home
                </button>
                <button 
                  onClick={() => router.push('/docs')}
<<<<<<< HEAD
                  className="text-neutral-600 hover:text-gold-600 text-sm font-medium"
=======
                  className="text-secondary hover:text-primary text-sm font-medium"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                >
                  Documentation
                </button>
                <ContactButton 
                  variant="link"
                  prefillCategory="support"
<<<<<<< HEAD
                  className="text-neutral-600 hover:text-gold-600 text-sm font-medium"
=======
                  className="text-secondary hover:text-primary text-sm font-medium"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
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
<<<<<<< HEAD
            <h1 className="text-4xl font-bold text-neutral-900 mb-4">
              🔌 DharmaMind API
            </h1>
            <p className="text-xl text-neutral-600 mb-8">
              Integrate dharmic AI wisdom into your applications with our REST API
            </p>
            
            <div className="inline-flex items-center px-6 py-3 bg-neutral-100 rounded-lg shadow-sm border border-neutral-300">
              <span className="text-neutral-600 mr-2">Base URL:</span>
              <code className="font-mono text-neutral-900 font-medium">
=======
            <h1 className="text-4xl font-bold text-primary mb-4">
              🔌 DharmaMind API
            </h1>
            <p className="text-xl text-secondary mb-8">
              Integrate dharmic AI wisdom into your applications with our REST API
            </p>
            
            <div className="inline-flex items-center px-6 py-3 bg-white rounded-lg shadow-sm border border-gray-200">
              <span className="text-secondary mr-2">Base URL:</span>
              <code className="font-mono text-primary font-medium">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                https://api.dharmamind.com/v1
              </code>
            </div>
          </div>
        </div>

        {/* Quick Start */}
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
<<<<<<< HEAD
          <h2 className="text-3xl font-bold text-neutral-900 mb-8">🚀 Quick Start</h2>
=======
          <h2 className="text-3xl font-bold text-primary mb-8">🚀 Quick Start</h2>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-16">
            
            {/* Authentication */}
<<<<<<< HEAD
            <div className="bg-neutral-100 rounded-lg shadow-sm border border-neutral-300 p-6">
              <h3 className="text-xl font-semibold text-neutral-900 mb-4">1. Authentication</h3>
              <p className="text-neutral-600 mb-4">
                First, authenticate to receive a JWT token:
              </p>
              <div className="bg-neutral-800 rounded-lg p-4 overflow-x-auto">
                <code className="text-neutral-900 text-sm">
                  <div className="text-neutral-600">POST</div>
                  <div className="text-white">/api/auth/login</div>
                  <div className="mt-2 text-neutral-600">{`{`}</div>
                  <div className="text-neutral-600 ml-4">"email": "user@example.com",</div>
                  <div className="text-neutral-600 ml-4">"password": "your_password"</div>
                  <div className="text-neutral-600">{`}`}</div>
=======
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h3 className="text-xl font-semibold text-primary mb-4">1. Authentication</h3>
              <p className="text-secondary mb-4">
                First, authenticate to receive a JWT token:
              </p>
              <div className="bg-stone-800 rounded-lg p-4 overflow-x-auto">
                <code className="text-primary text-sm">
                  <div className="text-secondary">POST</div>
                  <div className="text-white">/api/auth/login</div>
                  <div className="mt-2 text-gray-300">{`{`}</div>
                  <div className="text-gray-300 ml-4">"email": "user@example.com",</div>
                  <div className="text-gray-300 ml-4">"password": "your_password"</div>
                  <div className="text-gray-300">{`}`}</div>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                </code>
              </div>
            </div>

            {/* Making Requests */}
<<<<<<< HEAD
            <div className="bg-neutral-100 rounded-lg shadow-sm border border-neutral-300 p-6">
              <h3 className="text-xl font-semibold text-neutral-900 mb-4">2. Making Requests</h3>
              <p className="text-neutral-600 mb-4">
                Include the JWT token in your requests:
              </p>
              <div className="bg-neutral-800 rounded-lg p-4 overflow-x-auto">
                <code className="text-sm">
                  <div className="text-neutral-600">// Headers</div>
                  <div className="text-neutral-600">Authorization: Bearer YOUR_JWT_TOKEN</div>
                  <div className="text-neutral-600">Content-Type: application/json</div>
=======
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h3 className="text-xl font-semibold text-primary mb-4">2. Making Requests</h3>
              <p className="text-secondary mb-4">
                Include the JWT token in your requests:
              </p>
              <div className="bg-stone-800 rounded-lg p-4 overflow-x-auto">
                <code className="text-sm">
                  <div className="text-gray-400">// Headers</div>
                  <div className="text-gray-300">Authorization: Bearer YOUR_JWT_TOKEN</div>
                  <div className="text-gray-300">Content-Type: application/json</div>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                </code>
              </div>
            </div>
          </div>

          {/* API Endpoints */}
<<<<<<< HEAD
          <h2 className="text-3xl font-bold text-neutral-900 mb-8">📋 API Endpoints</h2>
          
          <div className="space-y-6">
            {endpoints.map((endpoint, index) => (
              <div key={index} className="bg-neutral-100 rounded-lg shadow-sm border border-neutral-300 p-6">
=======
          <h2 className="text-3xl font-bold text-primary mb-8">📋 API Endpoints</h2>
          
          <div className="space-y-6">
            {endpoints.map((endpoint, index) => (
              <div key={index} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                <div className="flex items-center space-x-4 mb-4">
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${getMethodColor(endpoint.method)}`}>
                    {endpoint.method}
                  </span>
<<<<<<< HEAD
                  <code className="text-lg font-mono text-neutral-900">{endpoint.path}</code>
                </div>
                
                <p className="text-neutral-600 mb-4">{endpoint.description}</p>
                
                {endpoint.params.length > 0 && (
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-2">Parameters:</h4>
                    <div className="flex flex-wrap gap-2">
                      {endpoint.params.map((param, paramIndex) => (
                        <code key={paramIndex} className="px-2 py-1 bg-primary-clean text-neutral-900 border border-neutral-300 rounded text-sm">
=======
                  <code className="text-lg font-mono text-primary">{endpoint.path}</code>
                </div>
                
                <p className="text-secondary mb-4">{endpoint.description}</p>
                
                {endpoint.params.length > 0 && (
                  <div>
                    <h4 className="font-medium text-primary mb-2">Parameters:</h4>
                    <div className="flex flex-wrap gap-2">
                      {endpoint.params.map((param, paramIndex) => (
                        <code key={paramIndex} className="px-2 py-1 bg-primary-clean text-primary border border-gray-200 rounded text-sm">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
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
<<<<<<< HEAD
            <h2 className="text-3xl font-bold text-neutral-900 mb-8">📄 Example Response</h2>
            
            <div className="bg-neutral-100 rounded-lg shadow-sm border border-neutral-300 p-6">
              <h3 className="text-xl font-semibold text-neutral-900 mb-4">Chat API Response</h3>
              <div className="bg-neutral-800 rounded-lg p-6 overflow-x-auto">
                <code className="text-sm">
                  <div className="text-neutral-600">{`{`}</div>
                  <div className="text-neutral-600 ml-4">"success": true,</div>
                  <div className="text-neutral-600 ml-4">"conversation_id": "conv_123456",</div>
                  <div className="text-neutral-600 ml-4">"response": {`{`}</div>
                  <div className="text-neutral-600 ml-8">"message": "Thank you for your question about finding inner peace...",</div>
                  <div className="text-neutral-600 ml-8">"wisdom_modules": ["moksha", "viveka", "ahimsa"],</div>
                  <div className="text-neutral-600 ml-8">"timestamp": "2025-01-29T10:30:00Z"</div>
                  <div className="text-neutral-600 ml-4">{`},`}</div>
                  <div className="text-neutral-600 ml-4">"usage": {`{`}</div>
                  <div className="text-neutral-600 ml-8">"tokens_used": 150,</div>
                  <div className="text-neutral-600 ml-8">"remaining_quota": 850</div>
                  <div className="text-neutral-600 ml-4">{`}`}</div>
                  <div className="text-neutral-600">{`}`}</div>
=======
            <h2 className="text-3xl font-bold text-primary mb-8">📄 Example Response</h2>
            
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h3 className="text-xl font-semibold text-primary mb-4">Chat API Response</h3>
              <div className="bg-stone-800 rounded-lg p-6 overflow-x-auto">
                <code className="text-sm">
                  <div className="text-gray-300">{`{`}</div>
                  <div className="text-gray-300 ml-4">"success": true,</div>
                  <div className="text-gray-300 ml-4">"conversation_id": "conv_123456",</div>
                  <div className="text-gray-300 ml-4">"response": {`{`}</div>
                  <div className="text-gray-300 ml-8">"message": "Thank you for your question about finding inner peace...",</div>
                  <div className="text-gray-300 ml-8">"wisdom_modules": ["moksha", "viveka", "ahimsa"],</div>
                  <div className="text-gray-300 ml-8">"timestamp": "2025-01-29T10:30:00Z"</div>
                  <div className="text-gray-300 ml-4">{`},`}</div>
                  <div className="text-gray-300 ml-4">"usage": {`{`}</div>
                  <div className="text-gray-300 ml-8">"tokens_used": 150,</div>
                  <div className="text-gray-300 ml-8">"remaining_quota": 850</div>
                  <div className="text-gray-300 ml-4">{`}`}</div>
                  <div className="text-gray-300">{`}`}</div>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                </code>
              </div>
            </div>
          </div>

          {/* Rate Limits */}
          <div className="mt-16">
<<<<<<< HEAD
            <h2 className="text-3xl font-bold text-neutral-900 mb-8">⚡ Rate Limits</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-neutral-100 rounded-lg shadow-sm border border-neutral-300 p-6 text-center">
                <div className="text-3xl mb-3">🆓</div>
                <h3 className="text-lg font-semibold text-neutral-900 mb-2">Free Tier</h3>
                <div className="text-2xl font-bold text-neutral-900 mb-1">100</div>
                <div className="text-neutral-600 text-sm">requests/hour</div>
              </div>
              
              <div className="bg-neutral-100 rounded-lg shadow-sm border border-neutral-300 p-6 text-center">
                <div className="text-3xl mb-3">💼</div>
                <h3 className="text-lg font-semibold text-neutral-900 mb-2">Professional</h3>
                <div className="text-2xl font-bold text-neutral-900 mb-1">1,000</div>
                <div className="text-neutral-600 text-sm">requests/hour</div>
              </div>
              
              <div className="bg-neutral-100 rounded-lg shadow-sm border border-neutral-300 p-6 text-center">
                <div className="text-3xl mb-3">🏢</div>
                <h3 className="text-lg font-semibold text-neutral-900 mb-2">Enterprise</h3>
                <div className="text-2xl font-bold text-neutral-900 mb-1">Custom</div>
                <div className="text-neutral-600 text-sm">unlimited available</div>
=======
            <h2 className="text-3xl font-bold text-primary mb-8">⚡ Rate Limits</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 text-center">
                <div className="text-3xl mb-3">🆓</div>
                <h3 className="text-lg font-semibold text-primary mb-2">Free Tier</h3>
                <div className="text-2xl font-bold text-primary mb-1">100</div>
                <div className="text-secondary text-sm">requests/hour</div>
              </div>
              
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 text-center">
                <div className="text-3xl mb-3">💼</div>
                <h3 className="text-lg font-semibold text-primary mb-2">Professional</h3>
                <div className="text-2xl font-bold text-primary mb-1">1,000</div>
                <div className="text-secondary text-sm">requests/hour</div>
              </div>
              
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 text-center">
                <div className="text-3xl mb-3">🏢</div>
                <h3 className="text-lg font-semibold text-primary mb-2">Enterprise</h3>
                <div className="text-2xl font-bold text-primary mb-1">Custom</div>
                <div className="text-secondary text-sm">unlimited available</div>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
              </div>
            </div>
          </div>

          {/* Error Codes */}
          <div className="mt-16">
<<<<<<< HEAD
            <h2 className="text-3xl font-bold text-neutral-900 mb-8">🚨 Error Codes</h2>
            
            <div className="bg-neutral-100 rounded-lg shadow-sm border border-neutral-300 overflow-hidden">
              <table className="w-full">
                <thead className="bg-neutral-100">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-neutral-600 uppercase tracking-wider">Code</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-neutral-600 uppercase tracking-wider">Description</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-neutral-600 uppercase tracking-wider">Solution</th>
=======
            <h2 className="text-3xl font-bold text-primary mb-8">🚨 Error Codes</h2>
            
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Code</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Description</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Solution</th>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  <tr>
<<<<<<< HEAD
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-neutral-900">400</td>
                    <td className="px-6 py-4 text-sm text-neutral-900">Bad Request</td>
                    <td className="px-6 py-4 text-sm text-neutral-600">Check request parameters</td>
                  </tr>
                  <tr>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-neutral-900">401</td>
                    <td className="px-6 py-4 text-sm text-neutral-900">Unauthorized</td>
                    <td className="px-6 py-4 text-sm text-neutral-600">Verify JWT token</td>
                  </tr>
                  <tr>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-neutral-900">429</td>
                    <td className="px-6 py-4 text-sm text-neutral-900">Rate Limited</td>
                    <td className="px-6 py-4 text-sm text-neutral-600">Reduce request frequency</td>
                  </tr>
                  <tr>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-neutral-900">500</td>
                    <td className="px-6 py-4 text-sm text-neutral-900">Server Error</td>
                    <td className="px-6 py-4 text-sm text-neutral-600">Contact support</td>
=======
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-primary">400</td>
                    <td className="px-6 py-4 text-sm text-primary">Bad Request</td>
                    <td className="px-6 py-4 text-sm text-secondary">Check request parameters</td>
                  </tr>
                  <tr>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-primary">401</td>
                    <td className="px-6 py-4 text-sm text-primary">Unauthorized</td>
                    <td className="px-6 py-4 text-sm text-secondary">Verify JWT token</td>
                  </tr>
                  <tr>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-primary">429</td>
                    <td className="px-6 py-4 text-sm text-primary">Rate Limited</td>
                    <td className="px-6 py-4 text-sm text-secondary">Reduce request frequency</td>
                  </tr>
                  <tr>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-primary">500</td>
                    <td className="px-6 py-4 text-sm text-primary">Server Error</td>
                    <td className="px-6 py-4 text-sm text-secondary">Contact support</td>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* SDKs */}
          <div className="mt-16">
<<<<<<< HEAD
            <h2 className="text-3xl font-bold text-neutral-900 mb-8">📦 SDKs & Libraries</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              
              <div className="bg-neutral-100 rounded-lg shadow-sm border border-neutral-300 p-6 text-center">
                <div className="text-4xl mb-4">📜</div>
                <h3 className="font-semibold text-neutral-900 mb-2">JavaScript</h3>
                <code className="text-sm text-neutral-600 mb-4 block">npm install dharmamind</code>
                <button className="text-neutral-900 hover:text-gold-600 font-medium text-sm">
=======
            <h2 className="text-3xl font-bold text-primary mb-8">📦 SDKs & Libraries</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 text-center">
                <div className="text-4xl mb-4">📜</div>
                <h3 className="font-semibold text-primary mb-2">JavaScript</h3>
                <code className="text-sm text-secondary mb-4 block">npm install dharmamind</code>
                <button className="text-primary hover:text-secondary font-medium text-sm">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                  View Docs →
                </button>
              </div>

<<<<<<< HEAD
              <div className="bg-neutral-100 rounded-lg shadow-sm border border-neutral-300 p-6 text-center">
                <div className="text-4xl mb-4">🐍</div>
                <h3 className="font-semibold text-neutral-900 mb-2">Python</h3>
                <code className="text-sm text-neutral-600 mb-4 block">pip install dharmamind</code>
                <button className="text-neutral-900 hover:text-gold-600 font-medium text-sm">
=======
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 text-center">
                <div className="text-4xl mb-4">🐍</div>
                <h3 className="font-semibold text-primary mb-2">Python</h3>
                <code className="text-sm text-secondary mb-4 block">pip install dharmamind</code>
                <button className="text-primary hover:text-secondary font-medium text-sm">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                  View Docs →
                </button>
              </div>

<<<<<<< HEAD
              <div className="bg-neutral-100 rounded-lg shadow-sm border border-neutral-300 p-6 text-center">
                <div className="text-4xl mb-4">💎</div>
                <h3 className="font-semibold text-neutral-900 mb-2">Ruby</h3>
                <code className="text-sm text-neutral-600 mb-4 block">gem install dharmamind</code>
                <button className="text-neutral-900 hover:text-gold-600 font-medium text-sm">
=======
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 text-center">
                <div className="text-4xl mb-4">💎</div>
                <h3 className="font-semibold text-primary mb-2">Ruby</h3>
                <code className="text-sm text-secondary mb-4 block">gem install dharmamind</code>
                <button className="text-primary hover:text-secondary font-medium text-sm">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                  View Docs →
                </button>
              </div>

<<<<<<< HEAD
              <div className="bg-neutral-100 rounded-lg shadow-sm border border-neutral-300 p-6 text-center">
                <div className="text-4xl mb-4">🔌</div>
                <h3 className="font-semibold text-neutral-900 mb-2">cURL</h3>
                <code className="text-sm text-neutral-600 mb-4 block">Direct HTTP calls</code>
                <button className="text-neutral-900 hover:text-gold-600 font-medium text-sm">
=======
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 text-center">
                <div className="text-4xl mb-4">🔌</div>
                <h3 className="font-semibold text-primary mb-2">cURL</h3>
                <code className="text-sm text-secondary mb-4 block">Direct HTTP calls</code>
                <button className="text-primary hover:text-secondary font-medium text-sm">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                  View Examples →
                </button>
              </div>
            </div>
          </div>

          {/* Support */}
<<<<<<< HEAD
          <div className="mt-16 bg-primary-clean rounded-lg p-8 text-center border border-neutral-300">
            <h3 className="text-xl font-semibold text-neutral-900 mb-4">
              Need API Support? 🤝
            </h3>
            <p className="text-neutral-600 mb-6">
=======
          <div className="mt-16 bg-primary-clean rounded-lg p-8 text-center border border-gray-200">
            <h3 className="text-xl font-semibold text-primary mb-4">
              Need API Support? 🤝
            </h3>
            <p className="text-secondary mb-6">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
              Our developer support team is here to help you integrate successfully.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <ContactButton
                variant="button"
                prefillCategory="support"
                className="btn-primary px-6 py-3 rounded-lg font-medium"
              >
                Contact Developers
              </ContactButton>
              <button
                onClick={() => router.push('/docs')}
<<<<<<< HEAD
                className="border border-neutral-300 text-neutral-900 px-6 py-3 rounded-lg font-medium hover:bg-neutral-100 transition-colors"
=======
                className="border border-gray-300 text-primary px-6 py-3 rounded-lg font-medium hover:bg-gray-50 transition-colors"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
              >
                View Full Docs
              </button>
            </div>
          </div>
        </div>

        {/* Footer */}
<<<<<<< HEAD
        <footer className="border-t border-neutral-300 bg-neutral-100">
=======
        <footer className="border-t border-gray-200 bg-white">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <div className="text-center">
              <Logo 
                size="sm"
                onClick={() => router.push('/')}
                className="justify-center mb-4 hover:opacity-80 transition-opacity"
              />
<<<<<<< HEAD
              <p className="text-sm text-neutral-600">
=======
              <p className="text-sm text-secondary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
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
