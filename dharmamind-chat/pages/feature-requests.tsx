import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import Logo from '../components/Logo';
import { CentralizedFeedbackButton } from '../components/CentralizedSupport';

interface FeatureRequest {
  id: string;
  title: string;
  description: string;
  category: 'ui-ux' | 'ai-features' | 'integrations' | 'performance' | 'security' | 'mobile' | 'enterprise';
  priority: 'low' | 'medium' | 'high' | 'critical';
  votes: number;
  status: 'submitted' | 'under-review' | 'planned' | 'in-progress' | 'completed' | 'declined';
  submittedBy: string;
  submittedAt: string;
  tags: string[];
}

interface NewFeatureRequest {
  title: string;
  description: string;
  category: FeatureRequest['category'];
  priority: FeatureRequest['priority'];
  tags: string[];
}

const FeatureRequestsPage: React.FC = () => {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<'browse' | 'submit' | 'roadmap'>('browse');
  const [features, setFeatures] = useState<FeatureRequest[]>([]);
  const [newRequest, setNewRequest] = useState<NewFeatureRequest>({
    title: '',
    description: '',
    category: 'ui-ux',
    priority: 'medium',
    tags: []
  });
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [showCentralizedOption, setShowCentralizedOption] = useState(true);

  // Sample feature requests data
  const sampleFeatures: FeatureRequest[] = [
    {
      id: '1',
      title: 'Dark Mode Theme',
      description: 'Add a dark mode option for better user experience during night time usage',
      category: 'ui-ux',
      priority: 'medium',
      votes: 142,
      status: 'in-progress',
      submittedBy: 'user123',
      submittedAt: '2024-01-15',
      tags: ['dark-mode', 'ui', 'accessibility']
    },
    {
      id: '2',
      title: 'Voice Input Support',
      description: 'Allow users to input questions via voice recognition',
      category: 'ai-features',
      priority: 'high',
      votes: 89,
      status: 'planned',
      submittedBy: 'user456',
      submittedAt: '2024-01-20',
      tags: ['voice', 'input', 'accessibility']
    },
    {
      id: '3',
      title: 'Slack Integration',
      description: 'Integrate DharmaMind with Slack for team conversations',
      category: 'integrations',
      priority: 'medium',
      votes: 67,
      status: 'under-review',
      submittedBy: 'user789',
      submittedAt: '2024-01-25',
      tags: ['slack', 'integration', 'teams']
    },
    {
      id: '4',
      title: 'Mobile App',
      description: 'Native mobile application for iOS and Android',
      category: 'mobile',
      priority: 'high',
      votes: 203,
      status: 'planned',
      submittedBy: 'user101',
      submittedAt: '2024-01-10',
      tags: ['mobile', 'ios', 'android', 'app']
    },
    {
      id: '5',
      title: 'Export Conversations',
      description: 'Allow users to export their conversation history in various formats',
      category: 'ui-ux',
      priority: 'low',
      votes: 34,
      status: 'submitted',
      submittedBy: 'user102',
      submittedAt: '2024-02-01',
      tags: ['export', 'history', 'backup']
    }
  ];

  useEffect(() => {
    setFeatures(sampleFeatures);
  }, []);

  const handleSubmitRequest = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      // Here you would normally submit to your API
      const request: FeatureRequest = {
        id: Date.now().toString(),
        ...newRequest,
        votes: 0,
        status: 'submitted',
        submittedBy: 'current_user',
        submittedAt: new Date().toISOString().split('T')[0]
      };

      setFeatures(prev => [request, ...prev]);
      setNewRequest({
        title: '',
        description: '',
        category: 'ui-ux',
        priority: 'medium',
        tags: []
      });
      setActiveTab('browse');
      
      console.log('Feature request submitted:', request);
    } catch (error) {
      console.error('Error submitting feature request:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleVote = (featureId: string) => {
    setFeatures(prev => 
      prev.map(feature => 
        feature.id === featureId 
          ? { ...feature, votes: feature.votes + 1 }
          : feature
      )
    );
  };

  const getStatusColor = (status: FeatureRequest['status']) => {
    const colors = {
      'submitted': 'bg-gray-100 text-gray-800',
      'under-review': 'bg-yellow-100 text-yellow-800',
      'planned': 'bg-blue-100 text-blue-800',
      'in-progress': 'bg-orange-100 text-orange-800',
      'completed': 'bg-green-100 text-green-800',
      'declined': 'bg-red-100 text-red-800'
    };
    return colors[status];
  };

  const getCategoryIcon = (category: FeatureRequest['category']) => {
    const icons = {
      'ui-ux': '🎨',
      'ai-features': '🤖',
      'integrations': '🔗',
      'performance': '⚡',
      'security': '🔒',
      'mobile': '📱',
      'enterprise': '🏢'
    };
    return icons[category];
  };

  const filteredFeatures = features.filter(feature => {
    const matchesSearch = feature.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         feature.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         feature.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    const matchesCategory = categoryFilter === 'all' || feature.category === categoryFilter;
    return matchesSearch && matchesCategory;
  });

  return (
    <>
      <Head>
        <title>Feature Requests - DharmaMind</title>
        <meta name="description" content="Submit feature requests, vote on upcoming features, and view our product roadmap" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-gradient-to-br from-orange-50 via-white to-pink-50">
        {/* Header */}
        <header className="border-b border-gray-200 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <Logo 
                size="sm"
                showText={true}
                onClick={() => router.push('/')}
              />

              <nav className="flex items-center space-x-8">
                <button 
                  onClick={() => router.push('/')}
                  className="text-gray-600 hover:text-gray-900 text-sm font-medium"
                >
                  Home
                </button>
              </nav>
            </div>
          </div>
        </header>

        {/* Centralized Option Banner */}
        {showCentralizedOption && (
          <div className="bg-gradient-to-r from-orange-500 to-pink-500 text-white">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="text-sm font-medium">
                    🚀 New! Submit feature requests directly to our enhanced product portal
                  </span>
                </div>
                <div className="flex items-center space-x-3">
                  <CentralizedFeedbackButton 
                    variant="link"
                    size="sm"
                    className="text-white hover:text-orange-100"
                  >
                    Go to Enhanced Portal
                  </CentralizedFeedbackButton>
                  <button
                    onClick={() => setShowCentralizedOption(false)}
                    className="text-white hover:text-orange-100"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-gray-900 mb-4">Feature Requests</h1>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Help shape the future of DharmaMind by submitting feature requests, voting on ideas, and tracking our development roadmap.
            </p>
          </div>

          {/* Tabs */}
          <div className="flex justify-center mb-8">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-1">
              <button
                onClick={() => setActiveTab('browse')}
                className={`px-6 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'browse'
                    ? 'bg-orange-100 text-orange-700'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                Browse Features
              </button>
              <button
                onClick={() => setActiveTab('submit')}
                className={`px-6 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'submit'
                    ? 'bg-orange-100 text-orange-700'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                Submit Request
              </button>
              <button
                onClick={() => setActiveTab('roadmap')}
                className={`px-6 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'roadmap'
                    ? 'bg-orange-100 text-orange-700'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                Roadmap
              </button>
            </div>
          </div>

          {/* Browse Features Tab */}
          {activeTab === 'browse' && (
            <div className="space-y-6">
              {/* Search and Filters */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <div className="flex flex-col sm:flex-row gap-4">
                  <div className="flex-1">
                    <input
                      type="text"
                      placeholder="Search features..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500"
                    />
                  </div>
                  <div>
                    <select
                      value={categoryFilter}
                      onChange={(e) => setCategoryFilter(e.target.value)}
                      className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500"
                    >
                      <option value="all">All Categories</option>
                      <option value="ui-ux">UI/UX</option>
                      <option value="ai-features">AI Features</option>
                      <option value="integrations">Integrations</option>
                      <option value="performance">Performance</option>
                      <option value="security">Security</option>
                      <option value="mobile">Mobile</option>
                      <option value="enterprise">Enterprise</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Feature List */}
              <div className="grid gap-6">
                {filteredFeatures.map((feature) => (
                  <div key={feature.id} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <span className="text-2xl">{getCategoryIcon(feature.category)}</span>
                          <h3 className="text-xl font-semibold text-gray-900">{feature.title}</h3>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(feature.status)}`}>
                            {feature.status.replace('-', ' ')}
                          </span>
                        </div>
                        <p className="text-gray-600 mb-3">{feature.description}</p>
                        <div className="flex items-center gap-4 text-sm text-gray-500">
                          <span>Category: {feature.category.replace('-', ' ')}</span>
                          <span>Priority: {feature.priority}</span>
                          <span>Submitted: {feature.submittedAt}</span>
                        </div>
                        <div className="flex flex-wrap gap-2 mt-3">
                          {feature.tags.map((tag, index) => (
                            <span key={index} className="px-2 py-1 bg-gray-100 text-gray-700 rounded-md text-xs">
                              #{tag}
                            </span>
                          ))}
                        </div>
                      </div>
                      <div className="ml-6 text-center">
                        <button
                          onClick={() => handleVote(feature.id)}
                          className="bg-orange-100 hover:bg-orange-200 text-orange-700 px-4 py-2 rounded-lg font-medium transition-colors"
                        >
                          ⬆️ {feature.votes}
                        </button>
                        <p className="text-xs text-gray-500 mt-1">votes</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Submit Request Tab */}
          {activeTab === 'submit' && (
            <div className="max-w-2xl mx-auto">
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-6">Submit Feature Request</h2>
                
                {/* Enhanced Portal Option */}
                <div className="bg-orange-50 border border-orange-200 rounded-lg p-4 mb-6">
                  <div className="flex items-start">
                    <svg className="w-5 h-5 text-orange-500 mr-3 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <div className="flex-1">
                      <h3 className="text-sm font-medium text-orange-800 mb-1">Enhanced Feature Portal Available</h3>
                      <p className="text-sm text-orange-700 mb-3">
                        Use our new enhanced feature portal for better tracking, voting, and collaboration with our product team.
                      </p>
                      <CentralizedFeedbackButton 
                        
                        variant="button"
                        size="sm"
                      >
                        Submit via Enhanced Portal
                      </CentralizedFeedbackButton>
                    </div>
                  </div>
                </div>

                <form onSubmit={handleSubmitRequest} className="space-y-6">
                  <div>
                    <label htmlFor="title" className="block text-sm font-medium text-gray-700 mb-2">
                      Feature Title *
                    </label>
                    <input
                      type="text"
                      id="title"
                      required
                      value={newRequest.title}
                      onChange={(e) => setNewRequest(prev => ({ ...prev, title: e.target.value }))}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500"
                      placeholder="Brief, descriptive title for your feature request"
                    />
                  </div>

                  <div>
                    <label htmlFor="description" className="block text-sm font-medium text-gray-700 mb-2">
                      Description *
                    </label>
                    <textarea
                      id="description"
                      required
                      rows={4}
                      value={newRequest.description}
                      onChange={(e) => setNewRequest(prev => ({ ...prev, description: e.target.value }))}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500"
                      placeholder="Detailed description of the feature, including use cases and benefits"
                    />
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                      <label htmlFor="category" className="block text-sm font-medium text-gray-700 mb-2">
                        Category
                      </label>
                      <select
                        id="category"
                        value={newRequest.category}
                        onChange={(e) => setNewRequest(prev => ({ ...prev, category: e.target.value as FeatureRequest['category'] }))}
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500"
                      >
                        <option value="ui-ux">UI/UX</option>
                        <option value="ai-features">AI Features</option>
                        <option value="integrations">Integrations</option>
                        <option value="performance">Performance</option>
                        <option value="security">Security</option>
                        <option value="mobile">Mobile</option>
                        <option value="enterprise">Enterprise</option>
                      </select>
                    </div>

                    <div>
                      <label htmlFor="priority" className="block text-sm font-medium text-gray-700 mb-2">
                        Priority
                      </label>
                      <select
                        id="priority"
                        value={newRequest.priority}
                        onChange={(e) => setNewRequest(prev => ({ ...prev, priority: e.target.value as FeatureRequest['priority'] }))}
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500"
                      >
                        <option value="low">Low</option>
                        <option value="medium">Medium</option>
                        <option value="high">High</option>
                        <option value="critical">Critical</option>
                      </select>
                    </div>
                  </div>

                  <div>
                    <label htmlFor="tags" className="block text-sm font-medium text-gray-700 mb-2">
                      Tags (comma-separated)
                    </label>
                    <input
                      type="text"
                      id="tags"
                      value={newRequest.tags.join(', ')}
                      onChange={(e) => setNewRequest(prev => ({ 
                        ...prev, 
                        tags: e.target.value.split(',').map(tag => tag.trim()).filter(tag => tag.length > 0)
                      }))}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500"
                      placeholder="e.g., mobile, dark-mode, accessibility"
                    />
                  </div>

                  <button
                    type="submit"
                    disabled={loading}
                    className="w-full bg-gradient-to-r from-orange-600 to-pink-600 text-white py-3 rounded-lg font-medium hover:from-orange-700 hover:to-pink-700 transition-colors disabled:opacity-50"
                  >
                    {loading ? 'Submitting...' : 'Submit Feature Request'}
                  </button>
                </form>
              </div>
            </div>
          )}

          {/* Roadmap Tab */}
          {activeTab === 'roadmap' && (
            <div className="space-y-8">
              <div className="text-center">
                <h2 className="text-2xl font-bold text-gray-900 mb-4">Development Roadmap</h2>
                <p className="text-gray-600 max-w-2xl mx-auto">
                  See what we're working on and what's coming next in DharmaMind's development.
                </p>
              </div>

              {/* Enhanced Roadmap Option */}
              <div className="bg-gradient-to-r from-orange-100 to-pink-100 border border-orange-200 rounded-lg p-6">
                <div className="text-center">
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">📋 Interactive Product Roadmap</h3>
                  <p className="text-gray-600 mb-4">
                    View our comprehensive interactive roadmap with timeline, milestones, and detailed feature planning.
                  </p>
                  <CentralizedFeedbackButton 
                    
                    variant="button"
                    size="lg"
                  >
                    View Interactive Roadmap
                  </CentralizedFeedbackButton>
                </div>
              </div>

              {/* Roadmap Sections */}
              <div className="grid md:grid-cols-3 gap-8">
                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                    🚀 <span className="ml-2">In Progress</span>
                  </h3>
                  <div className="space-y-3">
                    {features.filter(f => f.status === 'in-progress').map(feature => (
                      <div key={feature.id} className="p-3 bg-orange-50 rounded-lg">
                        <h4 className="font-medium text-gray-900">{feature.title}</h4>
                        <p className="text-sm text-gray-600 mt-1">{feature.votes} votes</p>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                    📅 <span className="ml-2">Planned</span>
                  </h3>
                  <div className="space-y-3">
                    {features.filter(f => f.status === 'planned').map(feature => (
                      <div key={feature.id} className="p-3 bg-blue-50 rounded-lg">
                        <h4 className="font-medium text-gray-900">{feature.title}</h4>
                        <p className="text-sm text-gray-600 mt-1">{feature.votes} votes</p>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                    ✅ <span className="ml-2">Completed</span>
                  </h3>
                  <div className="space-y-3">
                    {features.filter(f => f.status === 'completed').map(feature => (
                      <div key={feature.id} className="p-3 bg-green-50 rounded-lg">
                        <h4 className="font-medium text-gray-900">{feature.title}</h4>
                        <p className="text-sm text-gray-600 mt-1">{feature.votes} votes</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Feedback Section */}
          <div className="mt-16 bg-gradient-to-r from-pink-100 to-orange-100 rounded-lg p-8 text-center">
            <h3 className="text-xl font-bold text-gray-900 mb-4">Have General Product Feedback?</h3>
            <p className="text-gray-600 mb-6">
              Share your thoughts, suggestions, or report issues with our product team.
            </p>
            <CentralizedFeedbackButton 
              
              variant="button"
              size="lg"
            >
              Share Product Feedback
            </CentralizedFeedbackButton>
          </div>
        </main>
      </div>
    </>
  );
};

export default FeatureRequestsPage;
