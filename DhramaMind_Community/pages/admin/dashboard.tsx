import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import Logo from '../../components/Logo';
import { SessionManager } from '../../utils/secureStorage';
import { useToast } from '../../contexts/ToastContext';

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
  const { success, error } = useToast();
  const [activeTab, setActiveTab] = useState('overview');
  const [isLoading, setIsLoading] = useState(true);
  const [stats, setStats] = useState<DashboardStats>({
    totalPosts: 42,
    totalViews: 15420,
<<<<<<< HEAD
    totalUsers: 2847,
    activeUsers: 234,
    newComments: 156,
    pendingReviews: 3
  });

=======
    totalUsers: 1250,
    activeUsers: 89,
    newComments: 23,
    pendingReviews: 5
  });
  
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
  const [refreshing, setRefreshing] = useState(false);

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
    // Check authentication using secure session
    const checkAuth = () => {
      if (!SessionManager.isAdminAuthenticated()) {
        error('Authentication Required', 'Please log in to access the admin dashboard.');
        router.push('/admin/login');
        return;
      }
<<<<<<< HEAD

=======
      
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
      // Simulate loading
      const timer = setTimeout(() => setIsLoading(false), 800);
      return () => clearTimeout(timer);
    };

    checkAuth();
  }, [router, error]);

  const handleLogout = () => {
    SessionManager.clearAdminSession();
    success('Logged Out', 'You have been successfully logged out.');
    router.push('/admin/login');
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500));
      // Update stats with new data
      setStats(prev => ({
        ...prev,
        totalViews: prev.totalViews + Math.floor(Math.random() * 100),
        activeUsers: Math.floor(Math.random() * 150) + 50,
        newComments: Math.floor(Math.random() * 30) + 10
      }));
      success('Dashboard Refreshed', 'Latest data has been loaded.');
    } catch (err) {
      error('Refresh Failed', 'Unable to refresh dashboard data. Please try again.');
    } finally {
      setRefreshing(false);
    }
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
<<<<<<< HEAD

=======
      
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
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
<<<<<<< HEAD

              <div className="flex items-center space-x-4">
                <button
=======
              
              <div className="flex items-center space-x-4">
                <button 
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                  onClick={handleRefresh}
                  disabled={refreshing}
                  className="p-2 rounded-full bg-neutral-100 text-secondary hover:bg-neutral-200 hover:text-primary transition-colors disabled:opacity-50"
                  title="Refresh Dashboard"
                >
                  <span className={`text-lg ${refreshing ? 'animate-spin' : ''}`}>
                    {refreshing ? '⟳' : '🔄'}
                  </span>
                </button>
                <div className="relative">
                  <button className="p-2 rounded-full bg-warning-light text-warning hover:bg-warning hover:text-white transition-colors">
                    <span className="text-lg">🔔</span>
                    <span className="absolute -top-1 -right-1 bg-error text-white text-xs rounded-full h-5 w-5 flex items-center justify-center font-bold animate-pulse">
                      {stats.pendingReviews}
                    </span>
                  </button>
                </div>
<<<<<<< HEAD
                <button
=======
                <button 
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
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
<<<<<<< HEAD
                    className={`py-2 px-1 border-b-2 font-bold text-sm transition-colors ${activeTab === tab.id
                        ? 'border-primary text-primary'
                        : 'border-transparent text-secondary hover:text-primary hover:border-medium'
                      }`}
=======
                    className={`py-2 px-1 border-b-2 font-bold text-sm transition-colors ${
                      activeTab === tab.id
                        ? 'border-primary text-primary'
                        : 'border-transparent text-secondary hover:text-primary hover:border-medium'
                    }`}
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
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
                <div className="card-primary p-6 hover:shadow-xl transition-all duration-300 group relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-20 h-20 bg-primary opacity-5 rounded-full -mr-10 -mt-10 group-hover:scale-110 transition-transform"></div>
                  <div className="flex items-center justify-between relative z-10">
                    <div>
                      <p className="text-sm font-bold text-secondary uppercase tracking-wide">Total Posts</p>
                      <p className="text-3xl font-black text-primary group-hover:scale-105 transition-transform">{stats.totalPosts}</p>
                    </div>
                    <div className="text-4xl group-hover:animate-bounce">📝</div>
                  </div>
                  <div className="mt-4 relative z-10">
                    <span className="text-sm font-semibold text-success">+12% from last month</span>
                  </div>
                </div>

                <div className="card-primary p-6 hover:shadow-xl transition-all duration-300 group relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-20 h-20 bg-info opacity-5 rounded-full -mr-10 -mt-10 group-hover:scale-110 transition-transform"></div>
                  <div className="flex items-center justify-between relative z-10">
                    <div>
                      <p className="text-sm font-bold text-secondary uppercase tracking-wide">Total Views</p>
                      <p className="text-3xl font-black text-primary group-hover:scale-105 transition-transform">{stats.totalViews.toLocaleString()}</p>
                    </div>
                    <div className="text-4xl group-hover:animate-pulse">👁️</div>
                  </div>
                  <div className="mt-4 relative z-10">
                    <span className="text-sm font-semibold text-success">+23% from last month</span>
                  </div>
                </div>

                <div className="card-primary p-6 hover:shadow-xl transition-all duration-300 group relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-20 h-20 bg-warning opacity-5 rounded-full -mr-10 -mt-10 group-hover:scale-110 transition-transform"></div>
                  <div className="flex items-center justify-between relative z-10">
                    <div>
                      <p className="text-sm font-bold text-secondary uppercase tracking-wide">Total Users</p>
                      <p className="text-3xl font-black text-primary group-hover:scale-105 transition-transform">{stats.totalUsers.toLocaleString()}</p>
                    </div>
                    <div className="text-4xl group-hover:rotate-12 transition-transform">👥</div>
                  </div>
                  <div className="mt-4 relative z-10">
                    <span className="text-sm font-semibold text-success">+8% from last month</span>
                  </div>
                </div>

                <div className="card-primary p-6 hover:shadow-xl transition-all duration-300 group relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-20 h-20 bg-success opacity-5 rounded-full -mr-10 -mt-10 group-hover:scale-110 transition-transform"></div>
                  <div className="flex items-center justify-between relative z-10">
                    <div>
                      <p className="text-sm font-bold text-secondary uppercase tracking-wide">Active Now</p>
                      <p className="text-3xl font-black text-primary group-hover:scale-105 transition-transform">{stats.activeUsers}</p>
                    </div>
                    <div className="text-4xl">
                      <span className="inline-block animate-ping absolute text-2xl">🟢</span>
                      <span className="text-4xl">🟢</span>
                    </div>
                  </div>
                  <div className="mt-4 relative z-10">
                    <span className="text-sm font-semibold text-info">Live user count</span>
                  </div>
                </div>
              </div>

              {/* Quick Actions */}
              <div className="card-elevated p-8">
                <h3 className="text-2xl font-black text-primary mb-6 tracking-tight">Quick Actions</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
<<<<<<< HEAD
                  <button
=======
                  <button 
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    onClick={() => setActiveTab('content')}
                    className="btn-primary p-6 rounded-xl font-bold text-center hover:opacity-90 transition-all duration-300 group"
                  >
                    <div className="text-3xl mb-3 group-hover:scale-110 transition-transform">✍️</div>
                    <div className="text-lg">Create New Post</div>
                    <div className="text-sm opacity-80 mt-1">Write & publish</div>
                  </button>
<<<<<<< HEAD
                  <button
=======
                  <button 
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    onClick={() => setActiveTab('users')}
                    className="btn-outline p-6 rounded-xl font-bold text-center hover:bg-primary hover:text-white transition-all duration-300 group"
                  >
                    <div className="text-3xl mb-3 group-hover:scale-110 transition-transform">👤</div>
                    <div className="text-lg">Manage Users</div>
                    <div className="text-sm opacity-70 mt-1 group-hover:opacity-100">Community control</div>
                  </button>
<<<<<<< HEAD
                  <button
=======
                  <button 
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    onClick={() => setActiveTab('analytics')}
                    className="btn-outline p-6 rounded-xl font-bold text-center hover:bg-primary hover:text-white transition-all duration-300 group"
                  >
                    <div className="text-3xl mb-3 group-hover:scale-110 transition-transform">📊</div>
                    <div className="text-lg">View Analytics</div>
                    <div className="text-sm opacity-70 mt-1 group-hover:opacity-100">Insights & trends</div>
                  </button>
<<<<<<< HEAD
                  <button
=======
                  <button 
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    onClick={() => setActiveTab('settings')}
                    className="btn-outline p-6 rounded-xl font-bold text-center hover:bg-primary hover:text-white transition-all duration-300 group"
                  >
                    <div className="text-3xl mb-3 group-hover:scale-110 transition-transform">⚙️</div>
                    <div className="text-lg">Site Settings</div>
                    <div className="text-sm opacity-70 mt-1 group-hover:opacity-100">Configure system</div>
                  </button>
                </div>
              </div>

              {/* Recent Activity */}
              <div className="card-primary p-8">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-2xl font-black text-primary tracking-tight">Recent Activity</h3>
                  <button className="text-sm font-bold text-secondary hover:text-primary transition-colors">
                    View All →
                  </button>
                </div>
                <div className="space-y-4">
                  {recentActivity.map((activity) => (
                    <div key={activity.id} className="flex items-start space-x-4 p-4 hover:bg-neutral-50 rounded-xl transition-all duration-300 group cursor-pointer">
                      <div className={`p-3 rounded-xl ${getStatusColor(activity.status)} group-hover:scale-110 transition-transform`}>
                        <span className="text-xl">{getActivityIcon(activity.type, activity.status)}</span>
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-lg font-bold text-primary group-hover:text-primary-hover transition-colors">{activity.title}</p>
                        <p className="text-sm text-secondary font-medium mt-1 line-clamp-2">{activity.description}</p>
                        <p className="text-xs text-muted font-semibold mt-2 flex items-center">
                          <span className="mr-2">🕐</span>
                          {activity.timestamp}
                        </p>
                      </div>
                      <div className="flex-shrink-0">
                        <button className="p-2 rounded-lg bg-neutral-100 text-secondary hover:bg-primary hover:text-white transition-all opacity-0 group-hover:opacity-100">
                          <span>→</span>
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-6">
                  <button className="btn-outline px-6 py-3 rounded-xl font-bold text-sm hover:bg-primary hover:text-white transition-all w-full md:w-auto">
                    <span className="mr-2">📋</span>
                    View Complete Activity Log
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
<<<<<<< HEAD
                  <button
=======
                  <button 
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
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
