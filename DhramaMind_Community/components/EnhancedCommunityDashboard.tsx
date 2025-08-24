import React, { useState, useEffect } from 'react';
import { useColors } from '../contexts/ColorContext';
import Button from './Button';

interface CommunityStats {
  totalMembers: number;
  activeToday: number;
  postsToday: number;
  commentsToday: number;
  weeklyGrowth: number;
  monthlyEngagement: number;
  popularTopics: string[];
  recentJoins: Array<{
    id: string;
    name: string;
    avatar: string;
    joinedAt: string;
    isOnline: boolean;
  }>;
}

interface CommunityActivity {
  id: string;
  type: 'discussion' | 'post' | 'comment' | 'join' | 'achievement';
  title: string;
  description: string;
  author: string;
  timestamp: string;
  engagement: number;
  tags: string[];
}

const EnhancedCommunityDashboard: React.FC = () => {
  const { currentTheme } = useColors();
  const [isLoading, setIsLoading] = useState(true);
  const [activeFilter, setActiveFilter] = useState('all');
  const [timeRange, setTimeRange] = useState('today');
  
  const [stats, setStats] = useState<CommunityStats>({
    totalMembers: 2847,
    activeToday: 234,
    postsToday: 18,
    commentsToday: 156,
    weeklyGrowth: 8.5,
    monthlyEngagement: 73.2,
    popularTopics: ['Meditation', 'Mindfulness', 'Buddhism', 'Self-improvement', 'Mental Health'],
    recentJoins: [
      {
        id: '1',
        name: 'Sarah Chen',
        avatar: '👩‍💼',
        joinedAt: '2 hours ago',
        isOnline: true
      },
      {
        id: '2',
        name: 'Michael Rodriguez',
        avatar: '👨‍🎓',
        joinedAt: '4 hours ago',
        isOnline: true
      },
      {
        id: '3',
        name: 'Emily Watson',
        avatar: '👩‍🔬',
        joinedAt: '6 hours ago',
        isOnline: false
      },
      {
        id: '4',
        name: 'David Kim',
        avatar: '👨‍💻',
        joinedAt: '8 hours ago',
        isOnline: true
      }
    ]
  });

  const [activities, setActivities] = useState<CommunityActivity[]>([
    {
      id: '1',
      type: 'discussion',
      title: 'How to maintain consistent meditation practice?',
      description: 'Looking for advice on building a sustainable daily meditation routine...',
      author: 'Alex Thompson',
      timestamp: '5 minutes ago',
      engagement: 12,
      tags: ['meditation', 'habits', 'beginner']
    },
    {
      id: '2',
      type: 'post',
      title: 'The Science Behind Mindful Breathing',
      description: 'New research reveals fascinating insights about breathwork and neuroplasticity...',
      author: 'Dr. Lisa Patel',
      timestamp: '23 minutes ago',
      engagement: 28,
      tags: ['science', 'breathing', 'research']
    },
    {
      id: '3',
      type: 'achievement',
      title: 'Community Milestone: 5000 Meditation Sessions',
      description: 'Our community has collectively completed 5000 meditation sessions this month!',
      author: 'DharmaMind Team',
      timestamp: '1 hour ago',
      engagement: 45,
      tags: ['milestone', 'celebration', 'community']
    },
    {
      id: '4',
      type: 'comment',
      title: 'Insightful comment on Buddhist Philosophy',
      description: 'A deep dive into the Four Noble Truths and their practical applications...',
      author: 'Wisdom Seeker',
      timestamp: '2 hours ago',
      engagement: 15,
      tags: ['philosophy', 'buddhism', 'wisdom']
    }
  ]);

  useEffect(() => {
    // Simulate loading with realistic delay
    const timer = setTimeout(() => setIsLoading(false), 1200);
    return () => clearTimeout(timer);
  }, []);

  const getActivityIcon = (type: string) => {
    const icons = {
      discussion: '💬',
      post: '📝',
      comment: '💭',
      join: '👋',
      achievement: '🏆'
    };
    return icons[type as keyof typeof icons] || '📋';
  };

  const getEngagementColor = (engagement: number) => {
    if (engagement >= 30) return 'text-success';
    if (engagement >= 15) return 'text-warning';
    return 'text-info';
  };

  const filteredActivities = activities.filter(activity => 
    activeFilter === 'all' || activity.type === activeFilter
  );

  if (isLoading) {
    return (
      <div className="min-h-screen bg-secondary-bg flex items-center justify-center">
        <div className="text-center">
          <div className="relative">
            <div className="animate-spin rounded-full h-20 w-20 border-4 border-neutral-200 mx-auto mb-6"></div>
            <div className="animate-spin rounded-full h-20 w-20 border-4 border-primary border-t-transparent absolute top-0 left-1/2 transform -translate-x-1/2"></div>
          </div>
          <h2 className="text-2xl font-black text-primary mb-2">Loading Community Dashboard</h2>
          <p className="text-lg text-secondary font-semibold">Gathering the latest community insights...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-secondary-bg">
      {/* Header Section */}
      <div className="bg-primary-gradient text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="text-center">
            <h1 className="text-5xl font-black mb-4 tracking-tight">
              🌸 DharmaMind Community
            </h1>
            <p className="text-xl font-semibold opacity-90 max-w-2xl mx-auto">
              A thriving community of {stats.totalMembers.toLocaleString()} mindful souls 
              on their journey to inner peace and wisdom
            </p>
            <div className="mt-8 flex justify-center space-x-4">
              <div className="text-center">
                <div className="text-3xl font-black">{stats.activeToday}</div>
                <div className="text-sm font-semibold opacity-80">Active Today</div>
              </div>
              <div className="w-px bg-white opacity-30"></div>
              <div className="text-center">
                <div className="text-3xl font-black">{stats.weeklyGrowth}%</div>
                <div className="text-sm font-semibold opacity-80">Weekly Growth</div>
              </div>
              <div className="w-px bg-white opacity-30"></div>
              <div className="text-center">
                <div className="text-3xl font-black">{stats.monthlyEngagement}%</div>
                <div className="text-sm font-semibold opacity-80">Engagement Rate</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Quick Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="card-elevated p-6 hover:shadow-xl transition-all duration-300 group">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-bold text-secondary uppercase tracking-wide">Total Members</p>
                <p className="text-3xl font-black text-primary group-hover:scale-105 transition-transform">
                  {stats.totalMembers.toLocaleString()}
                </p>
              </div>
              <div className="text-4xl group-hover:animate-bounce">👥</div>
            </div>
            <div className="mt-4 flex items-center">
              <span className="text-sm font-semibold text-success">↗ +{stats.weeklyGrowth}%</span>
              <span className="text-xs text-muted ml-2">this week</span>
            </div>
          </div>

          <div className="card-elevated p-6 hover:shadow-xl transition-all duration-300 group">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-bold text-secondary uppercase tracking-wide">Active Today</p>
                <p className="text-3xl font-black text-primary group-hover:scale-105 transition-transform">
                  {stats.activeToday}
                </p>
              </div>
              <div className="text-4xl group-hover:animate-pulse">🟢</div>
            </div>
            <div className="mt-4 flex items-center">
              <span className="text-sm font-semibold text-info">Live activity</span>
            </div>
          </div>

          <div className="card-elevated p-6 hover:shadow-xl transition-all duration-300 group">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-bold text-secondary uppercase tracking-wide">Posts Today</p>
                <p className="text-3xl font-black text-primary group-hover:scale-105 transition-transform">
                  {stats.postsToday}
                </p>
              </div>
              <div className="text-4xl group-hover:rotate-12 transition-transform">📝</div>
            </div>
            <div className="mt-4 flex items-center">
              <span className="text-sm font-semibold text-success">Quality content</span>
            </div>
          </div>

          <div className="card-elevated p-6 hover:shadow-xl transition-all duration-300 group">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-bold text-secondary uppercase tracking-wide">Comments Today</p>
                <p className="text-3xl font-black text-primary group-hover:scale-105 transition-transform">
                  {stats.commentsToday}
                </p>
              </div>
              <div className="text-4xl group-hover:animate-bounce">💬</div>
            </div>
            <div className="mt-4 flex items-center">
              <span className="text-sm font-semibold text-warning">Engaging discussions</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Activity Feed */}
          <div className="lg:col-span-2 space-y-6">
            {/* Activity Filters */}
            <div className="card-primary p-6">
              <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
                <h2 className="text-2xl font-black text-primary">Community Activity</h2>
                <div className="flex space-x-2">
                  {['all', 'discussion', 'post', 'achievement', 'comment'].map((filter) => (
                    <button
                      key={filter}
                      onClick={() => setActiveFilter(filter)}
                      className={`px-4 py-2 rounded-lg font-bold text-sm transition-all ${
                        activeFilter === filter
                          ? 'bg-primary text-white shadow-md'
                          : 'bg-neutral-100 text-secondary hover:bg-neutral-200'
                      }`}
                    >
                      {filter.charAt(0).toUpperCase() + filter.slice(1)}
                    </button>
                  ))}
                </div>
              </div>

              {/* Activities List */}
              <div className="space-y-4">
                {filteredActivities.map((activity) => (
                  <div key={activity.id} className="border border-border-light rounded-lg p-5 hover:shadow-md hover:border-border-primary transition-all duration-300 group">
                    <div className="flex items-start space-x-4">
                      <div className="flex-shrink-0">
                        <div className="w-12 h-12 bg-primary-gradient rounded-full flex items-center justify-center text-white text-xl group-hover:scale-110 transition-transform">
                          {getActivityIcon(activity.type)}
                        </div>
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between mb-2">
                          <h3 className="text-lg font-bold text-primary group-hover:text-primary-hover transition-colors">
                            {activity.title}
                          </h3>
                          <span className={`text-sm font-semibold ${getEngagementColor(activity.engagement)}`}>
                            {activity.engagement} ❤️
                          </span>
                        </div>
                        <p className="text-secondary font-medium mb-3 line-clamp-2">
                          {activity.description}
                        </p>
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-4 text-sm">
                            <span className="font-semibold text-primary">{activity.author}</span>
                            <span className="text-muted">{activity.timestamp}</span>
                          </div>
                          <div className="flex space-x-2">
                            {activity.tags.map((tag) => (
                              <span key={tag} className="px-2 py-1 bg-neutral-100 text-secondary text-xs font-semibold rounded-full">
                                #{tag}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-6 text-center">
                <Button variant="outline" size="lg" className="font-bold">
                  Load More Activities
                </Button>
              </div>
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Popular Topics */}
            <div className="card-elevated p-6">
              <h3 className="text-xl font-black text-primary mb-4 flex items-center">
                <span className="mr-2">🔥</span>
                Trending Topics
              </h3>
              <div className="space-y-3">
                {stats.popularTopics.map((topic, index) => (
                  <div key={topic} className="flex items-center justify-between p-3 bg-neutral-50 rounded-lg hover:bg-neutral-100 transition-colors cursor-pointer group">
                    <span className="font-semibold text-primary group-hover:text-primary-hover transition-colors">
                      {topic}
                    </span>
                    <span className="text-xs bg-primary text-white px-2 py-1 rounded-full font-bold">
                      #{index + 1}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Recent Joins */}
            <div className="card-elevated p-6">
              <h3 className="text-xl font-black text-primary mb-4 flex items-center">
                <span className="mr-2">👋</span>
                New Members
              </h3>
              <div className="space-y-4">
                {stats.recentJoins.map((member) => (
                  <div key={member.id} className="flex items-center space-x-3 p-3 hover:bg-neutral-50 rounded-lg transition-colors group">
                    <div className="relative">
                      <div className="w-10 h-10 bg-primary-gradient rounded-full flex items-center justify-center text-white text-lg">
                        {member.avatar}
                      </div>
                      {member.isOnline && (
                        <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-success rounded-full border-2 border-white"></div>
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-semibold text-primary group-hover:text-primary-hover transition-colors">
                        {member.name}
                      </p>
                      <p className="text-xs text-muted">
                        Joined {member.joinedAt}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-4">
                <Button variant="outline" size="sm" className="w-full font-bold">
                  View All Members
                </Button>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="card-elevated p-6">
              <h3 className="text-xl font-black text-primary mb-4 flex items-center">
                <span className="mr-2">⚡</span>
                Quick Actions
              </h3>
              <div className="space-y-3">
                <Button variant="primary" size="md" className="w-full font-bold">
                  <span className="mr-2">✍️</span>
                  Start Discussion
                </Button>
                <Button variant="outline" size="md" className="w-full font-bold">
                  <span className="mr-2">📝</span>
                  Share Insight
                </Button>
                <Button variant="outline" size="md" className="w-full font-bold">
                  <span className="mr-2">🧘</span>
                  Join Meditation
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EnhancedCommunityDashboard;
