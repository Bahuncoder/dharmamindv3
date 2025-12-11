import React, { useState } from 'react';
import { useRouter } from 'next/router';
import { useSession, signOut } from 'next-auth/react';
import Head from 'next/head';
import Logo from '../components/Logo';
import { ContactButton } from '../components/CentralizedSupport';

interface User {
  name: string;
  email: string;
  plan?: 'free' | 'professional' | 'enterprise';
  isGuest?: boolean;
}

const SettingsPage: React.FC = () => {
  const router = useRouter();
  const { data: session, status } = useSession();
  const [user, setUser] = useState<User | null>(null);
  const [notifications, setNotifications] = useState(true);
  const [dataSharing, setDataSharing] = useState(false);
  const [learnFromInteractions, setLearnFromInteractions] = useState(true);
  const [chatHistory, setChatHistory] = useState<any[]>([]);

  React.useEffect(() => {
    // Check if user came from demo mode
    const { demo } = router.query;
    
    if (demo === 'true') {
      // Create demo user for settings
      const demoUser: User = {
        name: 'Guest User',
        email: 'demo@dharmamind.com',
        isGuest: true,
        plan: 'free'
      };
      setUser(demoUser);
      return;
    }

    // Handle authenticated users with NextAuth session
    if (session?.user) {
      const authUser: User = {
        name: session.user.name || 'User',
        email: session.user.email || '',
        plan: 'professional' // Google users get pro access
      };
      setUser(authUser);
      return;
    }

    // Check legacy localStorage (fallback)
    const userData = localStorage.getItem('dharma_user');
    if (userData) {
      setUser(JSON.parse(userData));
      return;
    }

    // No user found - redirect to home unless still loading
    if (status !== 'loading') {
      router.push('/');
    }
  }, [session, status, router]);

  React.useEffect(() => {
    // Load chat history based on user type
    if (user?.isGuest) {
      // For demo users, load from sessionStorage
      const history = JSON.parse(sessionStorage.getItem('demo_chat_history') || '[]');
      setChatHistory(history);
    } else if (user) {
      // For real users, load from localStorage
      const history = JSON.parse(localStorage.getItem('dharma_chat_history') || '[]');
      setChatHistory(history);
    }
  }, [user]);

  const handleLogout = async () => {
    if (session) {
      // Sign out from NextAuth
      await signOut({ redirect: false });
    }
    // Clear any localStorage data
    localStorage.removeItem('dharma_user');
    localStorage.removeItem('dharma_chat_history');
    
    // Redirect to chat in demo mode instead of welcome
  window.location.href = 'https://dharmamind.ai';
  };

  const handleBackToChat = () => {
  window.location.href = 'https://dharmamind.ai';
  };

  const clearAllChatHistory = () => {
    if (confirm('Are you sure you want to delete all chat history? This action cannot be undone.')) {
      if (user?.isGuest) {
        // For demo users, clear sessionStorage
        sessionStorage.removeItem('demo_chat_history');
      } else {
        // For real users, clear localStorage  
        localStorage.removeItem('dharma_chat_history');
      }
      setChatHistory([]);
    }
  };

  const deleteAllMyData = () => {
    const confirmMessage = user?.isGuest 
      ? 'Are you sure you want to delete all your demo data? This will clear your current session data.'
      : 'Are you sure you want to delete ALL your data? This will permanently delete your account, all conversations, and settings. This action cannot be undone.';
    
    if (confirm(confirmMessage)) {
      if (user?.isGuest) {
        // For demo users, just clear session data
        sessionStorage.clear();
        alert('Demo data cleared successfully.');
        router.push('/');
      } else {
        // For real users, clear everything and sign out
        localStorage.removeItem('dharma_chat_history');
        localStorage.removeItem('dharma_user');
        
        if (session) {
          signOut({ callbackUrl: '/', redirect: true });
        } else {
          router.push('/');
        }
        
        alert('All your data has been deleted successfully.');
      }
    }
  };

  const downloadUserData = () => {
    const userData = {
      user: user,
      chatHistory: chatHistory,
      settings: {
        notifications,
        dataSharing,
        learnFromInteractions
      },
      exportDate: new Date().toISOString()
    };
    
    const dataStr = JSON.stringify(userData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `dharmamind-user-data-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const exportChatHistory = () => {
    const dataStr = JSON.stringify(chatHistory, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    const filename = user?.isGuest 
      ? `dharmamind-demo-chat-history-${new Date().toISOString().split('T')[0]}.json`
      : `dharmamind-chat-history-${new Date().toISOString().split('T')[0]}.json`;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  };

  if (!user) return null;

  return (
    <>
      <Head>
        <title>Settings - DharmaMind</title>
        <meta name="description" content="Manage your DharmaMind preferences and account settings" />
      </Head>
      
      <div className="min-h-screen bg-neutral-100 dark:bg-neutral-900 transition-colors">
        {/* Header */}
        <div className="bg-neutral-100 dark:bg-primary-background border-b border-neutral-300 dark:border-gold-400">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center space-x-4">
                <button
                  onClick={handleBackToChat}
                  className="text-neutral-600 hover:text-gold-600 dark:text-neutral-600 dark:hover:text-gold-600"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                </button>
                <h1 className="text-xl font-semibold text-neutral-900 dark:text-white">Settings</h1>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-gradient-to-r from-gold-600 to-gold-700 rounded-full flex items-center justify-center">
                  <span className="text-white text-sm font-medium">
                    {user.name.charAt(0).toUpperCase()}
                  </span>
                </div>
                <span className="text-sm text-neutral-900 dark:text-neutral-600">{user.name}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="space-y-8">
            
            {/* Account Section */}
            <div className="bg-neutral-100 dark:bg-primary-background rounded-lg shadow-sm border border-neutral-300 dark:border-gold-400">
              <div className="px-6 py-4 border-b border-neutral-300 dark:border-gold-400">
                <h2 className="text-lg font-medium text-neutral-900 dark:text-white">Account</h2>
              </div>
              <div className="px-6 py-4 space-y-4">
                <div>
                  <label className="block text-sm font-medium text-neutral-900 dark:text-neutral-600 mb-1">
                    Name
                  </label>
                  <input
                    type="text"
                    value={user.name}
                    className="w-full px-3 py-2 border border-gold-400 dark:border-gold-400 rounded-md focus:outline-none focus:ring-2 focus:ring-brand-accent focus:border-transparent bg-neutral-100 dark:bg-primary-background text-neutral-900 dark:text-white"
                    readOnly
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-neutral-900 dark:text-neutral-600 mb-1">
                    Email
                  </label>
                  <input
                    type="email"
                    value={user.email}
                    className="w-full px-3 py-2 border border-gold-400 dark:border-gold-400 rounded-md focus:outline-none focus:ring-2 focus:ring-brand-accent focus:border-transparent bg-neutral-100 dark:bg-primary-background text-neutral-900 dark:text-white"
                    readOnly
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-neutral-900 dark:text-neutral-600 mb-1">
                    Current Plan
                  </label>
                  <div className="flex items-center space-x-3">
                    <span className="px-3 py-1 bg-neutral-100 dark:bg-primary-background text-neutral-900 dark:text-neutral-600 text-sm rounded-full capitalize">
                      {user.plan || 'free'}
                    </span>
                    <button 
                      onClick={() => router.push('/subscription')}
                      className="text-sm text-neutral-600 dark:text-neutral-600 hover:text-gold-600 dark:hover:text-gold-600 hover:underline transition-colors"
                    >
                      Manage subscription
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* App Preferences Section */}
            <div className="bg-neutral-100 dark:bg-primary-background rounded-lg shadow-sm border border-neutral-300 dark:border-gold-400">
              <div className="px-6 py-4 border-b border-neutral-300 dark:border-gold-400">
                <h2 className="text-lg font-medium text-neutral-900 dark:text-white">App Preferences</h2>
              </div>
              <div className="px-6 py-4 space-y-6">

                {/* Notifications */}
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-sm font-medium text-neutral-900">
                      Email notifications
                    </label>
                    <p className="text-sm text-neutral-600">
                      Receive updates about your spiritual journey
                    </p>
                  </div>
                  <button
                    onClick={() => setNotifications(!notifications)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                      notifications ? 'bg-neutral-100' : 'bg-primary-background'
                    }`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-neutral-100 transition-transform ${
                        notifications ? 'translate-x-6' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>

                {/* Data Sharing */}
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-sm font-medium text-neutral-900">
                      Help improve DharmaMind
                    </label>
                    <p className="text-sm text-neutral-600">
                      Share anonymized data to enhance AI responses
                    </p>
                  </div>
                  <button
                    onClick={() => setDataSharing(!dataSharing)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                      dataSharing ? 'bg-neutral-100' : 'bg-primary-background'
                    }`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-neutral-100 transition-transform ${
                        dataSharing ? 'translate-x-6' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>

                {/* Learn from Interactions */}
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-sm font-medium text-neutral-900">
                      Learn from my interactions
                    </label>
                    <p className="text-sm text-neutral-600">
                      Allow AI to learn from your conversations for better personalized responses
                    </p>
                  </div>
                  <button
                    onClick={() => setLearnFromInteractions(!learnFromInteractions)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                      learnFromInteractions ? 'bg-neutral-100' : 'bg-primary-background'
                    }`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-neutral-100 transition-transform ${
                        learnFromInteractions ? 'translate-x-6' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>
              </div>
            </div>

            {/* Chat History Section */}
            <div className="bg-neutral-100 dark:bg-primary-background rounded-lg shadow-sm border border-neutral-300 dark:border-gold-400">
              <div className="px-6 py-4 border-b border-neutral-300 dark:border-gold-400">
                <h2 className="text-lg font-medium text-neutral-900 dark:text-white">Chat History</h2>
                <p className="text-sm text-neutral-600 dark:text-neutral-600 mt-1">
                  Manage your conversation history and data
                </p>
                {user?.isGuest && (
                  <p className="text-sm text-neutral-600 mt-1">
                    Demo mode: History is temporary and will be cleared when you leave the app
                  </p>
                )}
              </div>
              <div className="px-6 py-4 space-y-6">
                
                {/* Chat History Stats */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-neutral-100 rounded-lg p-4">
                    <div className="text-sm font-medium text-neutral-600">Total Conversations</div>
                    <div className="text-2xl font-bold text-neutral-900">{chatHistory.length}</div>
                  </div>
                  <div className="bg-neutral-100 rounded-lg p-4">
                    <div className="text-sm font-medium text-neutral-600">Messages Sent</div>
                    <div className="text-2xl font-bold text-neutral-900">
                      {chatHistory.reduce((total, chat) => total + chat.messages?.length || 0, 0)}
                    </div>
                  </div>
                  <div className="bg-neutral-100 rounded-lg p-4">
                    <div className="text-sm font-medium text-neutral-600">Storage Used</div>
                    <div className="text-2xl font-bold text-neutral-900">
                      {Math.round(JSON.stringify(chatHistory).length / 1024)} KB
                    </div>
                  </div>
                </div>

                {/* Chat History Actions */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 border border-neutral-300 rounded-lg">
                    <div>
                      <h3 className="text-sm font-medium text-neutral-900">Export Chat History</h3>
                      <p className="text-sm text-neutral-600">Download all your conversations as a JSON file</p>
                    </div>
                    <button
                      onClick={exportChatHistory}
                      className="px-4 py-2 bg-neutral-100 hover:bg-primary-background text-neutral-900 rounded-md text-sm font-medium transition-colors"
                    >
                      Export
                    </button>
                  </div>

                  <div className="flex items-center justify-between p-4 border border-red-200 rounded-lg bg-red-50">
                    <div>
                      <h3 className="text-sm font-medium text-red-900">Clear All Chat History</h3>
                      <p className="text-sm text-red-600">Permanently delete all conversation history. This action cannot be undone.</p>
                    </div>
                    <button
                      onClick={clearAllChatHistory}
                      className="px-4 py-2 bg-red-100 hover:bg-red-200 text-red-700 rounded-md text-sm font-medium transition-colors"
                    >
                      Clear All
                    </button>
                  </div>
                </div>

                {/* Recent Conversations Preview */}
                {chatHistory.length > 0 && (
                  <div>
                    <h3 className="text-sm font-medium text-neutral-900 mb-3">Recent Conversations</h3>
                    <div className="space-y-2 max-h-48 overflow-y-auto">
                      {chatHistory.slice(0, 5).map((chat, index) => (
                        <div key={chat.id || index} className="flex items-center justify-between p-3 bg-neutral-100 rounded-lg">
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-neutral-900 truncate">
                              {chat.title || 'Untitled Conversation'}
                            </p>
                            <p className="text-xs text-neutral-600">
                              {new Date(chat.lastUpdate).toLocaleDateString()}
                            </p>
                          </div>
                          <div className="text-xs text-neutral-600">
                            {chat.messages?.length || 0} messages
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Privacy Section */}
            <div className="bg-neutral-100 dark:bg-primary-background rounded-lg shadow-sm border border-neutral-300 dark:border-gold-400">
              <div className="px-6 py-4 border-b border-neutral-300 dark:border-gold-400">
                <h2 className="text-lg font-medium text-neutral-900 dark:text-white">Privacy & Data</h2>
              </div>
              <div className="px-6 py-4 space-y-4">
                <button 
                  onClick={downloadUserData}
                  className="w-full text-left px-4 py-3 border border-neutral-300 dark:border-gold-400 rounded-lg hover:bg-neutral-100 dark:hover:bg-primary-background transition-colors"
                >
                  <div className="text-sm font-medium text-neutral-900 dark:text-white">Download your data</div>
                  <div className="text-sm text-neutral-600 dark:text-neutral-600">Export all your conversations and data</div>
                </button>
                <button 
                  onClick={deleteAllMyData}
                  className="w-full text-left px-4 py-3 border border-red-200 dark:border-red-600 rounded-lg hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors text-red-600 dark:text-red-400"
                >
                  <div className="text-sm font-medium">Delete my data</div>
                  <div className="text-sm text-red-500 dark:text-red-400">
                    {user?.isGuest 
                      ? 'Clear all demo session data' 
                      : 'Permanently delete your account and all data'
                    }
                  </div>
                </button>
              </div>
            </div>

            {/* Support Section */}
            <div className="bg-neutral-100 rounded-lg shadow-sm border border-neutral-300">
              <div className="px-6 py-4 border-b border-neutral-300">
                <h2 className="text-lg font-medium text-neutral-900">Support</h2>
              </div>
              <div className="px-6 py-4 space-y-4">
                <button 
                  onClick={() => router.push('/help')}
                  className="w-full text-left px-4 py-3 border border-neutral-300 rounded-lg hover:bg-neutral-100 transition-colors"
                >
                  <div className="text-sm font-medium text-neutral-900">Help Center</div>
                  <div className="text-sm text-neutral-600">Get help and find answers</div>
                </button>
                <ContactButton 
                  variant="card"
                  prefillCategory="support"
                  className="w-full text-left px-4 py-3 border border-neutral-300 rounded-lg hover:bg-neutral-100 transition-colors"
                >
                  <div className="text-sm font-medium text-neutral-900">Contact Support</div>
                  <div className="text-sm text-neutral-600">Get in touch with our team</div>
                </ContactButton>
                <button 
                  onClick={() => router.push('/feature-requests')}
                  className="w-full text-left px-4 py-3 border border-neutral-300 rounded-lg hover:bg-neutral-100 transition-colors"
                >
                  <div className="text-sm font-medium text-neutral-900">Feature Requests</div>
                  <div className="text-sm text-neutral-600">Suggest new features</div>
                </button>
              </div>
            </div>

            {/* Sign Out */}
            <div className="flex justify-center pt-8">
              <button
                onClick={handleLogout}
                className="px-6 py-2 border border-gold-400 rounded-lg text-neutral-900 hover:bg-neutral-100 transition-colors"
              >
                Sign out
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default SettingsPage;
