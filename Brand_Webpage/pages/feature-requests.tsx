import React, { useState } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import Logo from '../components/Logo';
<<<<<<< HEAD
import BrandHeader from '../components/BrandHeader';
=======
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc

interface FeatureRequest {
  id: string;
  title: string;
  description: string;
  category: 'ui-ux' | 'ai-features' | 'integrations' | 'performance' | 'mobile' | 'other';
  status: 'submitted' | 'under-review' | 'planned' | 'in-development' | 'completed';
  votes: number;
  submittedBy: string;
  submittedAt: Date;
}

const FeatureRequestsPage: React.FC = () => {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<'submit' | 'browse'>('submit');
  const [formData, setFormData] = useState({
    title: '',
    description: '',
    category: 'ai-features' as const
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  // Sample feature requests for the browse section
  const sampleRequests: FeatureRequest[] = [
    {
      id: '1',
      title: 'Voice-to-Text Chat Interface',
      description: 'Add ability to speak questions instead of typing for more natural conversations',
      category: 'ai-features',
      status: 'in-development',
      votes: 47,
      submittedBy: 'User',
      submittedAt: new Date('2024-01-15')
    },
    {
      id: '2',
      title: 'Dark Mode Theme',
      description: 'Implement a dark mode option for better usability in low-light environments',
      category: 'ui-ux',
      status: 'completed',
      votes: 32,
      submittedBy: 'User',
      submittedAt: new Date('2024-01-10')
    },
    {
      id: '3',
      title: 'Mobile App',
      description: 'Native mobile application for iOS and Android with offline capabilities',
      category: 'mobile',
      status: 'planned',
      votes: 68,
      submittedBy: 'User',
      submittedAt: new Date('2024-01-20')
    }
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
<<<<<<< HEAD

    // Simulate form submission
    await new Promise(resolve => setTimeout(resolve, 1000));

=======
    
    // Simulate form submission
    await new Promise(resolve => setTimeout(resolve, 1000));
    
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
    setSubmitted(true);
    setIsSubmitting(false);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const getStatusColor = (status: FeatureRequest['status']) => {
    switch (status) {
<<<<<<< HEAD
      case 'submitted': return 'bg-neutral-100 text-neutral-900';
      case 'under-review': return 'bg-gold-100 text-gold-700';
      case 'planned': return 'bg-gold-100 text-gold-700';
      case 'in-development': return 'bg-gold-100 text-gold-700';
      case 'completed': return 'bg-gold-100 text-green-800';
      default: return 'bg-neutral-100 text-neutral-900';
=======
      case 'submitted': return 'bg-gray-100 text-gray-800';
      case 'under-review': return 'bg-blue-100 text-blue-800';
      case 'planned': return 'bg-emerald-100 text-emerald-800';
      case 'in-development': return 'bg-purple-100 text-purple-800';
      case 'completed': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
    }
  };

  const getStatusText = (status: FeatureRequest['status']) => {
    switch (status) {
      case 'submitted': return 'Submitted';
      case 'under-review': return 'Under Review';
      case 'planned': return 'Planned';
      case 'in-development': return 'In Development';
      case 'completed': return 'Completed';
      default: return 'Unknown';
    }
  };

  return (
    <>
      <Head>
        <title>Feature Requests - DharmaMind</title>
        <meta name="description" content="Submit feature requests and see what's coming to DharmaMind" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

<<<<<<< HEAD
      <div className="min-h-screen bg-neutral-100">
        {/* Professional Brand Header with Breadcrumbs */}
        <BrandHeader
          breadcrumbs={[
            { label: 'Home', href: '/' },
            { label: 'Feature Requests', href: '/feature-requests' }
          ]}
        />

        {/* Main Content */}
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">

          {/* Hero Section */}
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-neutral-900 mb-4">
              Shape the Future of DharmaMind
            </h1>
            <p className="text-lg text-neutral-600 max-w-2xl mx-auto">
=======
      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="bg-white border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center space-x-4">
                <button
                  onClick={() => router.back()}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                </button>
                <Logo 
                  size="sm"
                  onClick={() => router.push('/')}
                />
                <h1 className="text-xl font-semibold text-gray-900">Feature Requests</h1>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          
          {/* Hero Section */}
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-gray-900 mb-4">
              Shape the Future of DharmaMind
            </h1>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
              Your ideas matter. Submit feature requests and help us build the AI spiritual companion you've always envisioned.
            </p>
          </div>

          {/* Tabs */}
<<<<<<< HEAD
          <div className="flex space-x-1 bg-neutral-100 p-1 rounded-lg mb-8">
            <button
              onClick={() => setActiveTab('submit')}
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${activeTab === 'submit'
                ? 'bg-neutral-100 text-neutral-900 shadow-sm'
                : 'text-neutral-600 hover:text-gold-600'
                }`}
=======
          <div className="flex space-x-1 bg-gray-100 p-1 rounded-lg mb-8">
            <button
              onClick={() => setActiveTab('submit')}
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
                activeTab === 'submit'
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
            >
              Submit Request
            </button>
            <button
              onClick={() => setActiveTab('browse')}
<<<<<<< HEAD
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${activeTab === 'browse'
                ? 'bg-neutral-100 text-neutral-900 shadow-sm'
                : 'text-neutral-600 hover:text-gold-600'
                }`}
=======
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
                activeTab === 'browse'
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
            >
              Browse Requests
            </button>
          </div>

          {/* Submit Tab */}
          {activeTab === 'submit' && (
<<<<<<< HEAD
            <div className="bg-neutral-100 rounded-lg shadow-sm border border-neutral-300 p-6">
              {submitted ? (
                <div className="text-center py-8">
                  <div className="w-16 h-16 bg-gold-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <svg className="w-8 h-8 text-gold-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold text-neutral-900 mb-2">Request Submitted!</h3>
                  <p className="text-neutral-600 mb-6">
=======
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              {submitted ? (
                <div className="text-center py-8">
                  <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <svg className="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">Request Submitted!</h3>
                  <p className="text-gray-600 mb-6">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    Thank you for your feature request. Our team will review it and update you on its progress.
                  </p>
                  <button
                    onClick={() => {
                      setSubmitted(false);
                      setFormData({ title: '', description: '', category: 'ai-features' });
                      setActiveTab('browse');
                    }}
<<<<<<< HEAD
                    className="bg-gradient-to-r from-gold-600 to-gold-700 text-white px-6 py-2 rounded-lg hover:from-gold-700 hover:to-gold-800 transition-colors"
=======
                    className="bg-gradient-to-r from-amber-600 to-emerald-600 text-white px-6 py-2 rounded-lg hover:from-amber-700 hover:to-emerald-700 transition-colors"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                  >
                    Browse Other Requests
                  </button>
                </div>
              ) : (
                <form onSubmit={handleSubmit} className="space-y-6">
                  <div>
<<<<<<< HEAD
                    <label htmlFor="title" className="block text-sm font-medium text-neutral-900 mb-2">
=======
                    <label htmlFor="title" className="block text-sm font-medium text-gray-700 mb-2">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                      Feature Title *
                    </label>
                    <input
                      type="text"
                      id="title"
                      name="title"
                      required
                      value={formData.title}
                      onChange={handleInputChange}
                      placeholder="Brief, descriptive title for your feature request"
<<<<<<< HEAD
                      className="w-full px-4 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-500 focus:border-transparent"
=======
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    />
                  </div>

                  <div>
<<<<<<< HEAD
                    <label htmlFor="category" className="block text-sm font-medium text-neutral-900 mb-2">
=======
                    <label htmlFor="category" className="block text-sm font-medium text-gray-700 mb-2">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                      Category *
                    </label>
                    <select
                      id="category"
                      name="category"
                      required
                      value={formData.category}
                      onChange={handleInputChange}
<<<<<<< HEAD
                      className="w-full px-4 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-500 focus:border-transparent"
=======
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    >
                      <option value="ai-features">AI Features</option>
                      <option value="ui-ux">UI/UX</option>
                      <option value="integrations">Integrations</option>
                      <option value="performance">Performance</option>
                      <option value="mobile">Mobile</option>
                      <option value="other">Other</option>
                    </select>
                  </div>

                  <div>
<<<<<<< HEAD
                    <label htmlFor="description" className="block text-sm font-medium text-neutral-900 mb-2">
=======
                    <label htmlFor="description" className="block text-sm font-medium text-gray-700 mb-2">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                      Detailed Description *
                    </label>
                    <textarea
                      id="description"
                      name="description"
                      required
                      rows={6}
                      value={formData.description}
                      onChange={handleInputChange}
                      placeholder="Describe your feature request in detail. Include the problem it solves, how it would work, and why it would be valuable."
<<<<<<< HEAD
                      className="w-full px-4 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-500 focus:border-transparent"
=======
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    />
                  </div>

                  <div className="flex justify-end">
                    <button
                      type="submit"
                      disabled={isSubmitting}
<<<<<<< HEAD
                      className="bg-gradient-to-r from-gold-600 to-gold-700 text-white px-6 py-2 rounded-lg hover:from-gold-700 hover:to-gold-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
=======
                      className="bg-gradient-to-r from-amber-600 to-emerald-600 text-white px-6 py-2 rounded-lg hover:from-amber-700 hover:to-emerald-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    >
                      {isSubmitting ? 'Submitting...' : 'Submit Request'}
                    </button>
                  </div>
                </form>
              )}
            </div>
          )}

          {/* Browse Tab */}
          {activeTab === 'browse' && (
            <div className="space-y-6">
              <div className="text-center">
<<<<<<< HEAD
                <p className="text-neutral-600">
=======
                <p className="text-gray-600">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                  See what features the community is requesting and track development progress.
                </p>
              </div>

              <div className="space-y-4">
                {sampleRequests.map((request) => (
<<<<<<< HEAD
                  <div key={request.id} className="bg-neutral-100 rounded-lg shadow-sm border border-neutral-300 p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex-1">
                        <h3 className="text-lg font-semibold text-neutral-900 mb-2">
                          {request.title}
                        </h3>
                        <p className="text-neutral-600 mb-3">
                          {request.description}
                        </p>
                        <div className="flex items-center space-x-4 text-sm text-neutral-600">
=======
                  <div key={request.id} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex-1">
                        <h3 className="text-lg font-semibold text-gray-900 mb-2">
                          {request.title}
                        </h3>
                        <p className="text-gray-600 mb-3">
                          {request.description}
                        </p>
                        <div className="flex items-center space-x-4 text-sm text-gray-500">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                          <span className="capitalize">{request.category.replace('-', ' ')}</span>
                          <span>•</span>
                          <span>{request.submittedAt.toLocaleDateString()}</span>
                        </div>
                      </div>
                      <div className="flex flex-col items-end space-y-2 ml-4">
                        <span className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(request.status)}`}>
                          {getStatusText(request.status)}
                        </span>
<<<<<<< HEAD
                        <div className="flex items-center space-x-1 text-sm text-neutral-600">
=======
                        <div className="flex items-center space-x-1 text-sm text-gray-500">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                          </svg>
                          <span>{request.votes}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="text-center py-8">
<<<<<<< HEAD
                <p className="text-neutral-600 text-sm">
=======
                <p className="text-gray-500 text-sm">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                  Want to see your idea here? Switch to the Submit tab to create a new feature request.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default FeatureRequestsPage;
