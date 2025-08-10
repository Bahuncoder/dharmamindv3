import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import Logo from '../../components/Logo';

interface DashboardStats {
  totalPosts: number;
  totalViews: number;
  totalUsers: number;
  activeUsers: number;
  newComments: number;
  pendingReviews: number;
}

interface RecentActivity {
  id: string;
  type: 'post' | 'comment' | 'user' | 'login';
  title: string;
  description: string;
  timestamp: string;
  status: 'success' | 'warning' | 'info';
}

const AdminDashboard: React.FC = () => {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState('overview');
  const [isLoading, setIsLoading] = useState(true);
  const [stats, setStats] = useState<DashboardStats>({
    totalPosts: 42,
    totalViews: 15420,
    totalUsers: 1250,
    activeUsers: 89,
    newComments: 23,
    pendingReviews: 5
  });

  const [recentActivity] = useState<RecentActivity[]>([
    {
      id: '1',
      type: 'post',
      title: 'New Blog Post Published',
      description: 'The Art of Mindful Living: Integrating Ancient Wisdom',
      timestamp: '2 minutes ago',
      status: 'success'
    },
    {
      id: '2',
      type: 'comment',
      title: 'New Comment Awaiting Moderation',
      description: 'Comment on "Building a Sustainable Meditation Practice"',
      timestamp: '15 minutes ago',
      status: 'warning'
    },
    {
      id: '3',
      type: 'user',
      title: 'New User Registration',
      description: 'Sarah Johnson joined the community',
      timestamp: '1 hour ago',
      status: 'info'
    },
    {
      id: '4',
      type: 'login',
      title: 'Admin Login',
      description: 'Successful login from IP: 192.168.1.100',
      timestamp: '2 hours ago',
      status: 'success'
    }
  ]);

  useEffect(() => {
    // Check authentication
    const isAuthenticated = localStorage.getItem('dharma_admin');
    if (!isAuthenticated) {
      router.push('/admin/login');
      return;
    }
    
    // Simulate loading
    const timer = setTimeout(() => setIsLoading(false), 800);
    return () => clearTimeout(timer);
  }, [router]);

  const handleLogout = () => {
    localStorage.removeItem('dharma_admin');
    router.push('/admin/login');
  };

  const getActivityIcon = (type: string, status: string) => {
    const icons = {
      post: '📝',
      comment: '💬',
      user: '👤',
      login: '🔐'
    };
    return icons[type as keyof typeof icons] || '📋';
  };

  const getStatusColor = (status: string) => {
    const colors = {
      success: 'text-success bg-success-light',
      warning: 'text-warning bg-warning-light',
      info: 'text-info bg-info-light'
    };
    return colors[status as keyof typeof colors] || 'text-secondary bg-neutral-100';
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-secondary-bg flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-xl font-semibold text-secondary">Loading Admin Dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <>
      <Head>
        <title>Admin Dashboard - DharmaMind</title>
        <meta name="description" content="DharmaMind Admin Dashboard - Content Management System" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/logo.jpeg" />
      </Head>
      
      <div className="min-h-screen bg-secondary-bg">
        {/* Top Navigation Bar */}
        <nav className="bg-primary-bg shadow-lg border-b-2 border-border-primary">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              <div className="flex items-center">
                <Logo size="sm" showText={true} />
                <div className="ml-4 hidden md:block">
                  <span className="text-lg font-black text-primary">Admin Dashboard</span>
                </div>
              </div>
              
              <div className="flex items-center space-x-4">
                <div className="relative">
                  <button className="p-2 rounded-full bg-warning-light text-warning hover:bg-warning hover:text-white transition-colors">
                    <span className="text-lg">🔔</span>
                    <span className="absolute -top-1 -right-1 bg-error text-white text-xs rounded-full h-5 w-5 flex items-center justify-center font-bold">
                      {stats.pendingReviews}
                    </span>
                  </button>
                </div>
                <button 
                  onClick={handleLogout}
                  className="btn-outline px-4 py-2 rounded-lg font-bold text-sm hover:bg-primary hover:text-white transition-all"
                >
                  Logout
                </button>
              </div>
            </div>
          </div>
        </nav>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Page Header */}
          <div className="mb-8">
            <h1 className="text-4xl font-black text-primary mb-2 tracking-tight">
              Welcome Back, Admin
            </h1>
            <p className="text-xl text-secondary font-semibold">
              Manage your DharmaMind content and community
            </p>
          </div>

          {/* Tab Navigation */}
          <div className="mb-8">
            <div className="border-b border-medium">
              <nav className="-mb-px flex space-x-8">
                {[
                  { id: 'overview', label: 'Overview', icon: '📊' },
                  { id: 'content', label: 'Content Management', icon: '📝' },
                  { id: 'users', label: 'User Management', icon: '👥' },
                  { id: 'analytics', label: 'Analytics', icon: '📈' },
                  { id: 'settings', label: 'Settings', icon: '⚙️' }
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`py-2 px-1 border-b-2 font-bold text-sm transition-colors ${
                      activeTab === tab.id
                        ? 'border-primary text-primary'
                        : 'border-transparent text-secondary hover:text-primary hover:border-medium'
                    }`}
                  >
                    <span className="mr-2">{tab.icon}</span>
                    {tab.label}
                  </button>
                ))}
              </nav>
            </div>
          </div>

          {/* Dashboard Content */}
          {activeTab === 'overview' && (
            <div className="space-y-8">
              {/* Stats Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="card-primary p-6 hover:shadow-lg transition-shadow">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-bold text-secondary uppercase tracking-wide">Total Posts</p>
                      <p className="text-3xl font-black text-primary">{stats.totalPosts}</p>
                    </div>
                    <div className="text-4xl">📝</div>
                  </div>
                  <div className="mt-4">
                    <span className="text-sm font-semibold text-success">+12% from last month</span>
                  </div>
                </div>

                <div className="card-primary p-6 hover:shadow-lg transition-shadow">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-bold text-secondary uppercase tracking-wide">Total Views</p>
                      <p className="text-3xl font-black text-primary">{stats.totalViews.toLocaleString()}</p>
                    </div>
                    <div className="text-4xl">👁️</div>
                  </div>
                  <div className="mt-4">
                    <span className="text-sm font-semibold text-success">+23% from last month</span>
                  </div>
                </div>

                <div className="card-primary p-6 hover:shadow-lg transition-shadow">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-bold text-secondary uppercase tracking-wide">Total Users</p>
                      <p className="text-3xl font-black text-primary">{stats.totalUsers.toLocaleString()}</p>
                    </div>
                    <div className="text-4xl">👥</div>
                  </div>
                  <div className="mt-4">
                    <span className="text-sm font-semibold text-success">+8% from last month</span>
                  </div>
                </div>

                <div className="card-primary p-6 hover:shadow-lg transition-shadow">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-bold text-secondary uppercase tracking-wide">Active Now</p>
                      <p className="text-3xl font-black text-primary">{stats.activeUsers}</p>
                    </div>
                    <div className="text-4xl">🟢</div>
                  </div>
                  <div className="mt-4">
                    <span className="text-sm font-semibold text-info">Live user count</span>
                  </div>
                </div>
              </div>

              {/* Quick Actions */}
              <div className="card-elevated p-8">
                <h3 className="text-2xl font-black text-primary mb-6 tracking-tight">Quick Actions</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <button 
                    onClick={() => setActiveTab('content')}
                    className="btn-primary p-4 rounded-lg font-bold text-center hover:opacity-90 transition-opacity"
                  >
                    <div className="text-2xl mb-2">✍️</div>
                    Create New Post
                  </button>
                  <button 
                    onClick={() => setActiveTab('users')}
                    className="btn-outline p-4 rounded-lg font-bold text-center hover:bg-primary hover:text-white transition-all"
                  >
                    <div className="text-2xl mb-2">👤</div>
                    Manage Users
                  </button>
                  <button 
                    onClick={() => setActiveTab('analytics')}
                    className="btn-outline p-4 rounded-lg font-bold text-center hover:bg-primary hover:text-white transition-all"
                  >
                    <div className="text-2xl mb-2">📊</div>
                    View Analytics
                  </button>
                  <button 
                    onClick={() => setActiveTab('settings')}
                    className="btn-outline p-4 rounded-lg font-bold text-center hover:bg-primary hover:text-white transition-all"
                  >
                    <div className="text-2xl mb-2">⚙️</div>
                    Site Settings
                  </button>
                </div>
              </div>

              {/* Recent Activity */}
              <div className="card-primary p-8">
                <h3 className="text-2xl font-black text-primary mb-6 tracking-tight">Recent Activity</h3>
                <div className="space-y-4">
                  {recentActivity.map((activity) => (
                    <div key={activity.id} className="flex items-start space-x-4 p-4 hover:bg-neutral-50 rounded-lg transition-colors">
                      <div className={`p-2 rounded-full ${getStatusColor(activity.status)}`}>
                        <span className="text-lg">{getActivityIcon(activity.type, activity.status)}</span>
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-bold text-primary">{activity.title}</p>
                        <p className="text-sm text-secondary font-medium">{activity.description}</p>
                        <p className="text-xs text-muted font-semibold">{activity.timestamp}</p>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-6">
                  <button className="btn-outline px-4 py-2 rounded-lg font-bold text-sm hover:bg-primary hover:text-white transition-all">
                    View All Activity
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Content Management Tab */}
          {activeTab === 'content' && (
            <div className="space-y-8">
              <div className="card-elevated p-8">
                <h3 className="text-3xl font-black text-primary mb-6 tracking-tight">Content Management</h3>
                <p className="text-lg text-secondary font-semibold mb-6">
                  Create, edit, and manage all your blog posts and website content.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  <button 
                    onClick={() => router.push('/admin/editor')}
                    className="card-primary p-6 text-left hover:shadow-lg transition-all group"
                  >
                    <div className="text-4xl mb-4 group-hover:scale-110 transition-transform">✍️</div>
                    <h4 className="text-xl font-black text-primary mb-2">Create New Post</h4>
                    <p className="text-secondary font-semibold">Write and publish new blog articles</p>
                  </button>
                  <button className="card-primary p-6 text-left hover:shadow-lg transition-all group">
                    <div className="text-4xl mb-4 group-hover:scale-110 transition-transform">📝</div>
                    <h4 className="text-xl font-black text-primary mb-2">Manage Posts</h4>
                    <p className="text-secondary font-semibold">Edit existing blog posts and articles</p>
                  </button>
                  <button className="card-primary p-6 text-left hover:shadow-lg transition-all group">
                    <div className="text-4xl mb-4 group-hover:scale-110 transition-transform">🏷️</div>
                    <h4 className="text-xl font-black text-primary mb-2">Categories & Tags</h4>
                    <p className="text-secondary font-semibold">Organize content with categories</p>
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Other tabs would be implemented similarly */}
          {activeTab !== 'overview' && activeTab !== 'content' && (
            <div className="card-elevated p-12 text-center">
              <div className="text-6xl mb-4">🚧</div>
              <h3 className="text-2xl font-black text-primary mb-4 tracking-tight">
                {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} Section
              </h3>
              <p className="text-lg text-secondary font-semibold">
                This section is under development and will be available soon.
              </p>
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default AdminDashboard;
