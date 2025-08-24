import React, { useState } from 'react';
import Head from 'next/head';
import Navigation from '../components/Navigation';
import EnhancedCommunityDashboard from '../components/EnhancedCommunityDashboard';

const CommunityPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [selectedCategory, setSelectedCategory] = useState('all');

  const categories = [
    { id: 'all', name: 'All Discussions', count: 0 },
    { id: 'life-decisions', name: 'Life Decisions', count: 0 },
    { id: 'personal-growth', name: 'Personal Growth', count: 0 },
    { id: 'ethics', name: 'Ethical Guidance', count: 0 },
    { id: 'relationships', name: 'Relationships', count: 0 },
    { id: 'purpose', name: 'Purpose & Career', count: 0 },
    { id: 'mental-clarity', name: 'Mental Clarity', count: 0 }
  ];

  const discussions: any[] = [];

  const members: any[] = [];

  const events: any[] = [];

  const resources: any[] = [];

  const filteredDiscussions = selectedCategory === 'all' 
    ? discussions 
    : discussions.filter(d => d.category === selectedCategory);

  return (
    <>
      <Head>
        <title>DharmaMind Community - Connect, Learn, Grow Together</title>
        <meta name="description" content="Join our vibrant community of spiritual seekers exploring AI consciousness and dharma wisdom. Connect with like-minded practitioners worldwide." />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/logo.jpeg" />
      </Head>
      <div className="min-h-screen bg-secondary-bg">
        <Navigation />
        <main className="w-full max-w-6xl mx-auto px-4 py-12">
          {/* Hero Section */}
          <section className="text-center mb-16">
            <h1 className="text-6xl md:text-7xl font-black text-primary mb-8 leading-tight tracking-tight">
              DharmaMind <span className="text-secondary font-black">Community</span>
            </h1>
            <p className="text-2xl font-bold text-secondary mb-6 max-w-3xl mx-auto tracking-wide">
              AI with Soul. Powered by Dharma.
            </p>
            <p className="text-xl text-secondary mb-8 max-w-3xl mx-auto leading-relaxed font-semibold">
              Join thousands who are making wiser choices, finding inner clarity, and living with purpose through ethical AI guidance.
            </p>
            <div className="flex justify-center gap-8 text-center mb-8">
              <div>
                <div className="text-4xl font-black text-primary">0</div>
                <div className="text-sm font-semibold text-muted">Active Members</div>
              </div>
              <div>
                <div className="text-4xl font-black text-primary">4.9★</div>
                <div className="text-sm font-semibold text-muted">Community Rating</div>
              </div>
              <div>
                <div className="text-4xl font-black text-primary">0</div>
                <div className="text-sm font-semibold text-muted">Countries</div>
              </div>
            </div>
          </section>

          {/* Navigation Tabs */}
          <section className="mb-12">
            <div className="flex flex-wrap justify-center gap-2 mb-8">
              <button
                onClick={() => setActiveTab('dashboard')}
                className={`px-6 py-3 rounded-lg font-bold transition-all duration-200 ${
                  activeTab === 'dashboard'
                    ? 'bg-primary text-white shadow-lg'
                    : 'bg-neutral-100 text-secondary hover:bg-neutral-200'
                }`}
              >
                🏠 Dashboard
              </button>
              <button
                onClick={() => setActiveTab('discussions')}
                className={`px-6 py-3 rounded-lg font-bold transition-all duration-200 ${
                  activeTab === 'discussions'
                    ? 'bg-primary text-white shadow-lg'
                    : 'bg-neutral-100 text-secondary hover:bg-neutral-200'
                }`}
              >
                💬 Discussions
              </button>
              <button
                onClick={() => setActiveTab('events')}
                className={`px-6 py-3 rounded-lg font-bold transition-all duration-200 ${
                  activeTab === 'events'
                    ? 'bg-primary text-white shadow-lg'
                    : 'bg-neutral-100 text-secondary hover:bg-neutral-200'
                }`}
              >
                📅 Events
              </button>
              <button
                onClick={() => setActiveTab('resources')}
                className={`px-6 py-3 rounded-lg font-bold transition-all duration-200 ${
                  activeTab === 'resources'
                    ? 'bg-primary text-white shadow-lg'
                    : 'bg-neutral-100 text-secondary hover:bg-neutral-200'
                }`}
              >
                📚 Resources
              </button>
              <button
                onClick={() => setActiveTab('members')}
                className={`px-6 py-3 rounded-lg font-bold transition-all duration-200 ${
                  activeTab === 'members'
                    ? 'bg-primary text-white shadow-lg'
                    : 'bg-neutral-100 text-secondary hover:bg-neutral-200'
                }`}
              >
                Members
              </button>
            </div>
          </section>

          {/* Enhanced Dashboard Tab */}
          {activeTab === 'dashboard' && (
            <div className="-mx-4">
              <EnhancedCommunityDashboard />
            </div>
          )}

          {/* Discussions Tab */}
          {activeTab === 'discussions' && (
            <section>
              {/* Category Filter */}
              <div className="mb-8">
                <div className="flex flex-wrap gap-2 justify-center">
                  {categories.map((category) => (
                    <button
                      key={category.id}
                      onClick={() => setSelectedCategory(category.id)}
                      className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                        selectedCategory === category.id
                          ? 'bg-primary text-white'
                          : 'bg-neutral-100 text-secondary hover:bg-neutral-200'
                      }`}
                    >
                      {category.name} ({category.count})
                    </button>
                  ))}
                </div>
              </div>

              {/* Featured Discussions */}
              <div className="mb-8">
                <h3 className="text-2xl font-black text-primary mb-6">Featured Discussions</h3>
                {filteredDiscussions.filter(d => d.featured).length > 0 ? (
                  <div className="grid gap-6 md:grid-cols-2">
                    {filteredDiscussions.filter(d => d.featured).map((discussion) => (
                      <div key={discussion.id} className="card-elevated p-6 hover:shadow-lg transition-all duration-300">
                        <div className="flex items-start justify-between mb-4">
                          <span className="px-3 py-1 bg-primary text-white text-xs rounded-full font-medium">
                            Featured
                          </span>
                          <span className="text-sm text-muted">{discussion.time}</span>
                        </div>
                        <h4 className="text-xl font-bold text-primary mb-3 line-clamp-2">
                          {discussion.title}
                        </h4>
                        <p className="text-secondary mb-4 line-clamp-2">
                          {discussion.content}
                        </p>
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <div className="w-8 h-8 bg-primary-gradient rounded-full flex items-center justify-center text-white text-sm font-medium">
                              {discussion.author.charAt(0)}
                            </div>
                            <span className="text-sm text-muted font-medium">{discussion.author}</span>
                          </div>
                          <div className="flex items-center gap-4 text-sm text-muted">
                            <span>💬 {discussion.replies}</span>
                            <span>❤️ {discussion.likes}</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="card-primary p-12 text-center">
                    <div className="text-6xl mb-4">💭</div>
                    <h4 className="text-xl font-bold text-primary mb-2">No Featured Discussions Yet</h4>
                    <p className="text-secondary">Featured discussions will appear here once added by admin.</p>
                  </div>
                )}
              </div>

              {/* All Discussions */}
              <div>
                <h3 className="text-2xl font-black text-primary mb-6">Recent Discussions</h3>
                {filteredDiscussions.length > 0 ? (
                  <div className="space-y-4">
                    {filteredDiscussions.map((discussion) => (
                      <div key={discussion.id} className="card-primary p-6 hover:shadow-md transition-all duration-300">
                        <div className="flex items-start justify-between mb-3">
                          <h4 className="text-lg font-bold text-primary line-clamp-1 flex-1 mr-4">
                            {discussion.title}
                          </h4>
                          <span className="text-sm text-muted whitespace-nowrap">{discussion.time}</span>
                        </div>
                        <p className="text-secondary mb-4 line-clamp-2">
                          {discussion.content}
                        </p>
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <div className="w-8 h-8 bg-primary-gradient rounded-full flex items-center justify-center text-white text-sm font-medium">
                              {discussion.author.charAt(0)}
                            </div>
                            <span className="text-sm text-muted font-medium">{discussion.author}</span>
                          </div>
                          <div className="flex items-center gap-4 text-sm text-muted">
                            <span>💬 {discussion.replies}</span>
                            <span>❤️ {discussion.likes}</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="card-primary p-12 text-center">
                    <div className="text-6xl mb-4">🌱</div>
                    <h4 className="text-xl font-bold text-primary mb-2">Community Growing Soon</h4>
                    <p className="text-secondary">Discussions will appear here as our community grows and members share their experiences.</p>
                    <div className="mt-6">
                      <a 
                        href="https://dharmamind.ai" 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="btn-primary px-6 py-3 rounded-lg font-bold"
                      >
                        Start Your Journey →
                      </a>
                    </div>
                  </div>
                )}
              </div>
            </section>
          )}

          {/* Events Tab */}
          {activeTab === 'events' && (
            <section>
              <h3 className="text-3xl font-black text-primary mb-8 text-center">Upcoming Community Events</h3>
              {events.length > 0 ? (
                <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                  {events.map((event) => (
                    <div key={event.id} className="card-elevated p-6 text-center hover:shadow-lg transition-all duration-300">
                      <div className="bg-primary-gradient-light w-16 h-16 rounded-full flex items-center justify-center mb-4 mx-auto">
                        <span className="text-2xl">📅</span>
                      </div>
                      <h4 className="text-xl font-bold text-primary mb-3">{event.title}</h4>
                      <div className="text-secondary mb-4 space-y-1">
                        <p className="font-semibold">{event.date}</p>
                        <p>{event.time}</p>
                        <p className="text-sm">
                          <span className="inline-block w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                          {event.type}
                        </p>
                      </div>
                      <p className="text-sm text-muted mb-4">{event.attendees} attending</p>
                      <button className="btn-primary px-6 py-2 rounded-lg font-bold w-full">
                        Join Event
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="card-primary p-12 text-center">
                  <div className="text-6xl mb-4">📅</div>
                  <h4 className="text-xl font-bold text-primary mb-2">No Events Scheduled</h4>
                  <p className="text-secondary">Community events will be posted here as they are scheduled.</p>
                </div>
              )}
            </section>
          )}

          {/* Resources Tab */}
          {activeTab === 'resources' && (
            <section>
              <h3 className="text-3xl font-black text-primary mb-8 text-center">Community Resources</h3>
              {resources.length > 0 ? (
                <div className="grid gap-6 md:grid-cols-2">
                  {resources.map((resource, index) => (
                    <div key={index} className="card-primary p-6 hover:shadow-md transition-all duration-300">
                      <div className="flex items-start justify-between mb-4">
                        <div className="flex items-center gap-3">
                          <div className="w-12 h-12 bg-primary-gradient-light rounded-lg flex items-center justify-center">
                            <span className="text-xl">
                              {resource.type === 'Guide' ? '📖' : 
                               resource.type === 'Audio' ? '🎧' :
                               resource.type === 'Article' ? '📄' : '📋'}
                            </span>
                          </div>
                          <div>
                            <h4 className="text-lg font-bold text-primary">{resource.title}</h4>
                            <p className="text-sm text-muted">{resource.type} • {resource.category}</p>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted">
                          📥 {resource.downloads.toLocaleString()} downloads
                        </span>
                        <button className="btn-outline px-4 py-2 rounded-lg font-medium text-sm">
                          Access Resource
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="card-primary p-12 text-center">
                  <div className="text-6xl mb-4">📚</div>
                  <h4 className="text-xl font-bold text-primary mb-2">Resources Coming Soon</h4>
                  <p className="text-secondary">Learning resources and guides will be available here soon.</p>
                </div>
              )}
            </section>
          )}

          {/* Members Tab */}
          {activeTab === 'members' && (
            <section>
              <h3 className="text-3xl font-black text-primary mb-8 text-center">Community Members</h3>
              {members.length > 0 ? (
                <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                  {members.map((member, index) => (
                    <div key={index} className="card-primary p-6 text-center hover:shadow-md transition-all duration-300">
                      <div className="w-16 h-16 bg-primary-gradient rounded-full flex items-center justify-center text-white text-xl font-bold mb-4 mx-auto">
                        {member.avatar}
                      </div>
                      <h4 className="text-lg font-bold text-primary mb-2">{member.name}</h4>
                      <p className="text-sm text-secondary font-medium mb-3">{member.role}</p>
                      <div className="flex flex-wrap justify-center gap-1 mb-4">
                        {member.specialties.map((specialty: string, idx: number) => (
                          <span key={idx} className="px-2 py-1 bg-primary-gradient-light text-xs rounded-full">
                            {specialty}
                          </span>
                        ))}
                      </div>
                      <div className="text-sm text-muted space-y-1">
                        <p>{member.contributions} contributions</p>
                        <p>Joined {member.joinDate}</p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="card-primary p-12 text-center">
                  <div className="text-6xl mb-4">👥</div>
                  <h4 className="text-xl font-bold text-primary mb-2">Community Growing</h4>
                  <p className="text-secondary">Member profiles will appear here as our community grows.</p>
                  <div className="mt-6">
                    <a 
                      href="https://dharmamind.ai" 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="btn-primary px-6 py-3 rounded-lg font-bold"
                    >
                      Be Among the First →
                    </a>
                  </div>
                </div>
              )}
            </section>
          )}

          {/* Call to Action */}
          <section className="mt-16 card-elevated p-8 md:p-12 text-center bg-primary-gradient-light">
            <h3 className="text-4xl font-black text-primary mb-6 tracking-tight">Join Our Community</h3>
            <p className="text-xl text-secondary mb-8 max-w-2xl mx-auto font-medium">
              Ready to make wiser choices and live with greater clarity and purpose? Connect with others on the same journey.
            </p>
            <a 
              href="https://dharmamind.ai" 
              target="_blank" 
              rel="noopener noreferrer"
              className="btn-primary px-8 py-3 rounded-lg font-bold text-lg hover:opacity-90 transition-opacity"
            >
              Experience DharmaMind →
            </a>
            <p className="text-sm text-muted mt-4 font-semibold">
              Free to join • Immediate access • 4.9★ rating • Global community
            </p>
          </section>
        </main>
      </div>
    </>
  );
};

export default CommunityPage;
