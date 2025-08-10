import React, { useState } from 'react';
import { useRouter } from 'next/router';
import { useSession, signOut } from 'next-auth/react';
import Head from 'next/head';
import Logo from '../components/Logo';
import { ContactButton } from '../components/CentralizedSupport';
import CentralizedSubscriptionModal from '../components/CentralizedSubscriptionModal';

interface User {
  name: string;
  email: string;
  plan?: 'basic' | 'pro' | 'max' | 'enterprise';
  isGuest?: boolean;
}

type SettingsCategory = 'general' | 'notifications' | 'personalization' | 'data-controls' | 'security' | 'account';

const SettingsPage: React.FC = () => {
  const router = useRouter();
  const { data: session, status } = useSession();
  const [user, setUser] = useState<User | null>(null);
  const [activeCategory, setActiveCategory] = useState<SettingsCategory>('general');
  const [isSubscriptionModalOpen, setIsSubscriptionModalOpen] = useState(false);
  
  // Settings states
  const [language, setLanguage] = useState('English');
  const [responseNotifications, setResponseNotifications] = useState(true);
  const [taskNotifications, setTaskNotifications] = useState(true);
  const [theme, setTheme] = useState('light');
  const [fontSize, setFontSize] = useState('medium');
  const [autoSave, setAutoSave] = useState(true);
  const [improveModel, setImproveModel] = useState(true);
  const [multifactorAuth, setMultifactorAuth] = useState(false);

  React.useEffect(() => {
    // Check if user came from demo mode
    const { demo } = router.query;
    
    if (demo === 'true') {
      const demoUser: User = {
        name: 'Guest User',
        email: 'demo@dharmamind.com',
        isGuest: true,
        plan: 'basic'
      };
      setUser(demoUser);
      return;
    }

    // Handle authenticated users with NextAuth session
    if (session?.user) {
      const authUser: User = {
        name: session.user.name || 'User',
        email: session.user.email || '',
        plan: 'pro'
      };
      setUser(authUser);
      return;
    }

    // If no session and not demo, redirect to login
    if (status !== 'loading') {
      router.push('/');
    }
  }, [router, session, status]);

  const handleSignOut = async () => {
    if (user?.isGuest) {
      router.push('/');
    } else {
      signOut({ callbackUrl: '/', redirect: true });
    }
  };

  const categories = [
    { id: 'general' as const, name: 'General', icon: '⚙️' },
    { id: 'notifications' as const, name: 'Notifications', icon: '🔔' },
    { id: 'personalization' as const, name: 'Personalization', icon: '🎨' },
    { id: 'data-controls' as const, name: 'Data Controls', icon: '📊' },
    { id: 'security' as const, name: 'Security', icon: '🔒' },
    { id: 'account' as const, name: 'Account', icon: '👤' },
  ];

  const exportChatHistory = () => {
    const dataStr = JSON.stringify([], null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    const exportFileDefaultName = 'dharmamind-chat-history.json';
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  const deleteAllChats = () => {
    if (confirm('Are you sure you want to delete all your chat history? This action cannot be undone.')) {
      localStorage.removeItem('dharma_chat_history');
      alert('All chat history has been deleted.');
    }
  };

  const deleteAccount = () => {
    if (confirm('Are you sure you want to delete your account? This action cannot be undone and will permanently delete all your data.')) {
      // Handle account deletion
      handleSignOut();
    }
  };

  const renderCategoryContent = () => {
    switch (activeCategory) {
      case 'general':
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">General Settings</h3>
              
              {/* Language */}
              <div className="flex items-center justify-between py-4 border-b border-gray-200">
                <div>
                  <h4 className="text-sm font-medium text-gray-900">Language</h4>
                  <p className="text-sm text-gray-500">Choose your preferred language</p>
                </div>
                <select
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                  className="mt-1 block w-32 pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-emerald-500 focus:border-emerald-500 sm:text-sm rounded-md"
                >
                  <option value="English">English</option>
                  <option value="Spanish">Spanish</option>
                  <option value="French">French</option>
                  <option value="German">German</option>
                  <option value="Hindi">Hindi</option>
                  <option value="Sanskrit">Sanskrit</option>
                </select>
              </div>
            </div>
          </div>
        );

      case 'notifications':
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Notification Preferences</h3>
              
              {/* Response Notifications */}
              <div className="flex items-center justify-between py-4 border-b border-gray-200">
                <div>
                  <h4 className="text-sm font-medium text-gray-900">Response Notifications</h4>
                  <p className="text-sm text-gray-500">Get notified when DharmaMind responds</p>
                </div>
                <button
                  onClick={() => setResponseNotifications(!responseNotifications)}
                  className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 ${
                    responseNotifications ? 'bg-emerald-600' : 'bg-gray-200'
                  }`}
                >
                  <span
                    className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                      responseNotifications ? 'translate-x-5' : 'translate-x-0'
                    }`}
                  />
                </button>
              </div>

              {/* Task Notifications */}
              <div className="flex items-center justify-between py-4 border-b border-gray-200">
                <div>
                  <h4 className="text-sm font-medium text-gray-900">Task Notifications</h4>
                  <p className="text-sm text-gray-500">Get notified about important tasks and reminders</p>
                </div>
                <button
                  onClick={() => setTaskNotifications(!taskNotifications)}
                  className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 ${
                    taskNotifications ? 'bg-emerald-600' : 'bg-gray-200'
                  }`}
                >
                  <span
                    className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                      taskNotifications ? 'translate-x-5' : 'translate-x-0'
                    }`}
                  />
                </button>
              </div>
            </div>
          </div>
        );

      case 'personalization':
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Personalization</h3>
              
              {/* Theme */}
              <div className="flex items-center justify-between py-4 border-b border-gray-200">
                <div>
                  <h4 className="text-sm font-medium text-gray-900">Theme</h4>
                  <p className="text-sm text-gray-500">Choose your preferred color theme</p>
                </div>
                <select
                  value={theme}
                  onChange={(e) => setTheme(e.target.value)}
                  className="mt-1 block w-32 pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-emerald-500 focus:border-emerald-500 sm:text-sm rounded-md"
                >
                  <option value="light">Light</option>
                  <option value="dark">Dark</option>
                  <option value="auto">Auto</option>
                </select>
              </div>

              {/* Font Size */}
              <div className="flex items-center justify-between py-4 border-b border-gray-200">
                <div>
                  <h4 className="text-sm font-medium text-gray-900">Font Size</h4>
                  <p className="text-sm text-gray-500">Adjust text size for better readability</p>
                </div>
                <select
                  value={fontSize}
                  onChange={(e) => setFontSize(e.target.value)}
                  className="mt-1 block w-32 pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-emerald-500 focus:border-emerald-500 sm:text-sm rounded-md"
                >
                  <option value="small">Small</option>
                  <option value="medium">Medium</option>
                  <option value="large">Large</option>
                </select>
              </div>
            </div>
          </div>
        );

      case 'data-controls':
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Data Controls</h3>
              
              {/* Archived Chats */}
              <div className="flex items-center justify-between py-4 border-b border-gray-200">
                <div>
                  <h4 className="text-sm font-medium text-gray-900">Archived Chats</h4>
                  <p className="text-sm text-gray-500">View and manage your archived conversations</p>
                </div>
                <button
                  onClick={() => router.push('/conversations')}
                  className="inline-flex items-center px-3 py-1.5 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-emerald-500"
                >
                  View Archives
                </button>
              </div>

              {/* Delete All Chats */}
              <div className="flex items-center justify-between py-4 border-b border-gray-200">
                <div>
                  <h4 className="text-sm font-medium text-gray-900">Delete All Chats</h4>
                  <p className="text-sm text-gray-500">Permanently delete all conversation history</p>
                </div>
                <button
                  onClick={deleteAllChats}
                  className="inline-flex items-center px-3 py-1.5 border border-red-300 shadow-sm text-sm font-medium rounded-md text-red-700 bg-white hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                >
                  Delete All
                </button>
              </div>

              {/* Export Data */}
              <div className="flex items-center justify-between py-4 border-b border-gray-200">
                <div>
                  <h4 className="text-sm font-medium text-gray-900">Export Data</h4>
                  <p className="text-sm text-gray-500">Download your conversation history</p>
                </div>
                <button
                  onClick={exportChatHistory}
                  className="inline-flex items-center px-3 py-1.5 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-emerald-500"
                >
                  Export
                </button>
              </div>

              {/* Improve Model */}
              <div className="flex items-center justify-between py-4 border-b border-gray-200">
                <div>
                  <h4 className="text-sm font-medium text-gray-900">Improve the Model Forever</h4>
                  <p className="text-sm text-gray-500">Help improve DharmaMind by sharing your interactions</p>
                </div>
                <button
                  onClick={() => setImproveModel(!improveModel)}
                  className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 ${
                    improveModel ? 'bg-emerald-600' : 'bg-gray-200'
                  }`}
                >
                  <span
                    className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                      improveModel ? 'translate-x-5' : 'translate-x-0'
                    }`}
                  />
                </button>
              </div>

              {/* Shared Links */}
              <div className="flex items-center justify-between py-4 border-b border-gray-200">
                <div>
                  <h4 className="text-sm font-medium text-gray-900">Shared Links</h4>
                  <p className="text-sm text-gray-500">Manage conversations you've shared</p>
                </div>
                <button
                  onClick={() => alert('Shared links management coming soon!')}
                  className="inline-flex items-center px-3 py-1.5 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-emerald-500"
                >
                  Manage
                </button>
              </div>
            </div>
          </div>
        );

      case 'security':
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Security Settings</h3>
              
              {/* Logout This Device */}
              <div className="flex items-center justify-between py-4 border-b border-gray-200">
                <div>
                  <h4 className="text-sm font-medium text-gray-900">Logout of This Device</h4>
                  <p className="text-sm text-gray-500">Sign out from this device only</p>
                </div>
                <button
                  onClick={handleSignOut}
                  className="inline-flex items-center px-3 py-1.5 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-emerald-500"
                >
                  Logout
                </button>
              </div>

              {/* Multifactor Authentication */}
              <div className="flex items-center justify-between py-4 border-b border-gray-200">
                <div>
                  <h4 className="text-sm font-medium text-gray-900">Multifactor Authentication</h4>
                  <p className="text-sm text-gray-500">Add an extra layer of security to your account</p>
                </div>
                <button
                  onClick={() => setMultifactorAuth(!multifactorAuth)}
                  className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 ${
                    multifactorAuth ? 'bg-emerald-600' : 'bg-gray-200'
                  }`}
                >
                  <span
                    className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                      multifactorAuth ? 'translate-x-5' : 'translate-x-0'
                    }`}
                  />
                </button>
              </div>

              {/* Logout All Devices */}
              <div className="flex items-center justify-between py-4 border-b border-gray-200">
                <div>
                  <h4 className="text-sm font-medium text-gray-900">Logout of All Devices</h4>
                  <p className="text-sm text-gray-500">Sign out from all devices and browsers</p>
                </div>
                <button
                  onClick={() => {
                    if (confirm('Are you sure you want to logout from all devices?')) {
                      handleSignOut();
                    }
                  }}
                  className="inline-flex items-center px-3 py-1.5 border border-red-300 shadow-sm text-sm font-medium rounded-md text-red-700 bg-white hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                >
                  Logout All
                </button>
              </div>
            </div>
          </div>
        );

      case 'account':
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Account Management</h3>
              
              {/* Account Information */}
              <div className="py-4 border-b border-gray-200">
                <h4 className="text-sm font-medium text-gray-900 mb-2">Account Information</h4>
                <div className="space-y-2">
                  <p className="text-sm text-gray-600">
                    <span className="font-medium">Name:</span> {user?.name || 'User'}
                  </p>
                  <p className="text-sm text-gray-600">
                    <span className="font-medium">Email:</span> {user?.email || 'Not provided'}
                  </p>
                  <p className="text-sm text-gray-600">
                    <span className="font-medium">Plan:</span> 
                    <span className="ml-1 capitalize">{user?.plan || 'Basic'}</span>
                    {!user?.isGuest && (
                      <button
                        onClick={() => setIsSubscriptionModalOpen(true)}
                        className="ml-2 text-emerald-600 hover:text-emerald-800 text-sm font-medium"
                      >
                        Upgrade
                      </button>
                    )}
                  </p>
                </div>
              </div>

              {/* Delete Account */}
              <div className="flex items-center justify-between py-4">
                <div>
                  <h4 className="text-sm font-medium text-gray-900">Delete Account</h4>
                  <p className="text-sm text-gray-500">Permanently delete your account and all data</p>
                </div>
                <button
                  onClick={deleteAccount}
                  className="inline-flex items-center px-3 py-1.5 border border-red-300 shadow-sm text-sm font-medium rounded-md text-red-700 bg-white hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                >
                  Delete Account
                </button>
              </div>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  if (status === 'loading') {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
      </div>
    );
  }

  return (
    <>
      <Head>
        <title>Settings - DharmaMind</title>
        <meta name="description" content="Manage your DharmaMind account settings and preferences" />
      </Head>

      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="bg-white shadow-sm border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <button
                onClick={() => router.push('/')}
                className="flex items-center space-x-2 text-gray-900 hover:text-gray-700"
              >
                <Logo size="sm" showText={true} />
              </button>
              
              <div className="flex items-center space-x-4">
                <ContactButton variant="link" />
                <button
                  onClick={() => router.push('/chat')}
                  className="text-gray-600 hover:text-gray-900 font-medium"
                >
                  Back to Chat
                </button>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col lg:flex-row gap-6">
            
            {/* Sidebar */}
            <div className="lg:w-64 flex-shrink-0">
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
                <h2 className="text-lg font-medium text-gray-900 mb-4">Settings</h2>
                <nav className="space-y-1">
                  {categories.map((category) => (
                    <button
                      key={category.id}
                      onClick={() => setActiveCategory(category.id)}
                      className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                        activeCategory === category.id
                          ? 'bg-emerald-100 text-emerald-800'
                          : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                      }`}
                    >
                      <span className="mr-3">{category.icon}</span>
                      {category.name}
                    </button>
                  ))}
                </nav>
              </div>
            </div>

            {/* Content Area */}
            <div className="flex-1">
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                {renderCategoryContent()}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Centralized Subscription Modal */}
      <CentralizedSubscriptionModal
        isOpen={isSubscriptionModalOpen}
        onClose={() => setIsSubscriptionModalOpen(false)}
        currentPlan={user?.plan || 'basic'}
      />
    </>
  );
};

export default SettingsPage;
