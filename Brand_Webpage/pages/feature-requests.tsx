import React, { useState } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import Logo from '../components/Logo';
import BrandHeader from '../components/BrandHeader';

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

    // Simulate form submission
    await new Promise(resolve => setTimeout(resolve, 1000));

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
      case 'submitted': return 'bg-neutral-100 text-neutral-900';
      case 'under-review': return 'bg-gold-100 text-gold-700';
      case 'planned': return 'bg-gold-100 text-gold-700';
      case 'in-development': return 'bg-gold-100 text-gold-700';
      case 'completed': return 'bg-gold-100 text-green-800';
      default: return 'bg-neutral-100 text-neutral-900';
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
              Your ideas matter. Submit feature requests and help us build the AI spiritual companion you've always envisioned.
            </p>
          </div>

          {/* Tabs */}
          <div className="flex space-x-1 bg-neutral-100 p-1 rounded-lg mb-8">
            <button
              onClick={() => setActiveTab('submit')}
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${activeTab === 'submit'
                ? 'bg-neutral-100 text-neutral-900 shadow-sm'
                : 'text-neutral-600 hover:text-gold-600'
                }`}
            >
              Submit Request
            </button>
            <button
              onClick={() => setActiveTab('browse')}
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${activeTab === 'browse'
                ? 'bg-neutral-100 text-neutral-900 shadow-sm'
                : 'text-neutral-600 hover:text-gold-600'
                }`}
            >
              Browse Requests
            </button>
          </div>

          {/* Submit Tab */}
          {activeTab === 'submit' && (
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
                    Thank you for your feature request. Our team will review it and update you on its progress.
                  </p>
                  <button
                    onClick={() => {
                      setSubmitted(false);
                      setFormData({ title: '', description: '', category: 'ai-features' });
                      setActiveTab('browse');
                    }}
                    className="bg-gradient-to-r from-gold-600 to-gold-700 text-white px-6 py-2 rounded-lg hover:from-gold-700 hover:to-gold-800 transition-colors"
                  >
                    Browse Other Requests
                  </button>
                </div>
              ) : (
                <form onSubmit={handleSubmit} className="space-y-6">
                  <div>
                    <label htmlFor="title" className="block text-sm font-medium text-neutral-900 mb-2">
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
                      className="w-full px-4 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-500 focus:border-transparent"
                    />
                  </div>

                  <div>
                    <label htmlFor="category" className="block text-sm font-medium text-neutral-900 mb-2">
                      Category *
                    </label>
                    <select
                      id="category"
                      name="category"
                      required
                      value={formData.category}
                      onChange={handleInputChange}
                      className="w-full px-4 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-500 focus:border-transparent"
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
                    <label htmlFor="description" className="block text-sm font-medium text-neutral-900 mb-2">
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
                      className="w-full px-4 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-500 focus:border-transparent"
                    />
                  </div>

                  <div className="flex justify-end">
                    <button
                      type="submit"
                      disabled={isSubmitting}
                      className="bg-gradient-to-r from-gold-600 to-gold-700 text-white px-6 py-2 rounded-lg hover:from-gold-700 hover:to-gold-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
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
                <p className="text-neutral-600">
                  See what features the community is requesting and track development progress.
                </p>
              </div>

              <div className="space-y-4">
                {sampleRequests.map((request) => (
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
                          <span className="capitalize">{request.category.replace('-', ' ')}</span>
                          <span>•</span>
                          <span>{request.submittedAt.toLocaleDateString()}</span>
                        </div>
                      </div>
                      <div className="flex flex-col items-end space-y-2 ml-4">
                        <span className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(request.status)}`}>
                          {getStatusText(request.status)}
                        </span>
                        <div className="flex items-center space-x-1 text-sm text-neutral-600">
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
                <p className="text-neutral-600 text-sm">
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
