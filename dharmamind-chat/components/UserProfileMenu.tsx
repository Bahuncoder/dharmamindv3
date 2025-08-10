import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { SparklesIcon } from '@heroicons/react/24/outline';
import { useColors } from '../contexts/ColorContext';
import { useAuth } from '../contexts/AuthContext';
import { useSubscription } from '../contexts/SubscriptionContext';
import { useRouter } from 'next/router';
import SettingsModal from './SettingsModal';

interface UserProfileMenuProps {
  onUpgrade: () => void;
  onClose: () => void;
  user?: any; // Accept user prop to override AuthContext user
  isDemo?: boolean; // Whether user is in demo mode
  onLogout?: () => void; // Custom logout handler
}

const UserProfileMenu: React.FC<UserProfileMenuProps> = ({ onUpgrade, onClose, user: propUser, isDemo = false, onLogout: customLogout }) => {
  const { currentTheme } = useColors();
  const { user: authUser, isAuthenticated, logout } = useAuth();
  const { upgradePlan } = useSubscription();
  const router = useRouter();
  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false);

  // Use prop user if provided, otherwise use auth user
  const user = propUser || authUser;

  const handleUpgrade = () => {
    onClose();
    onUpgrade();
  };

  const handleSettings = () => {
    setIsSettingsModalOpen(true);
  };

  const handleConversations = () => {
    onClose();
    if (isDemo) {
      router.push('/conversations?demo=true');
    } else {
      router.push('/conversations');
    }
  };

  const handleHelp = () => {
    onClose();
    window.open('https://dharmamind.com/help', '_blank');
  };

  const handleLogout = () => {
    onClose();
    if (customLogout) {
      customLogout();
    } else {
      localStorage.clear();
      logout();
      router.push('/');
    }
  };

  const handleManageSubscription = () => {
    onClose();
    onUpgrade();
  };

  // Always render the dropdown - show guest menu if no user
  if (!user) {
    return (
      <>
        <motion.div
          initial={{ opacity: 0, y: 10, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 10, scale: 0.95 }}
          transition={{ duration: 0.15 }}
        className="absolute left-0 right-0 bottom-full mb-2 bg-white rounded-lg shadow-xl border border-gray-200 overflow-hidden z-50"
        style={{ 
          boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)' 
        }}
      >
        {/* Guest User Info Section */}
        <div className="px-4 py-3 border-b border-gray-100 bg-gradient-to-r from-gray-50 to-gray-100">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-full flex items-center justify-center bg-gray-200 ring-2 ring-white">
              <span className="text-lg font-semibold text-gray-600">👤</span>
            </div>
            <div className="flex-1">
              <p className="font-semibold text-gray-900">Guest User</p>
              <p className="text-xs text-gray-500">Try DharmaMind for free</p>
            </div>
          </div>
        </div>

        {/* Upgrade Section for Guests */}
        <div className="px-4 py-3">
          <div className="text-center mb-3">
            <div className="inline-flex items-center px-3 py-1 rounded-full bg-yellow-100 text-yellow-800 text-sm font-medium mb-2">
              <SparklesIcon className="h-4 w-4 mr-1" />
              Limited Access
            </div>
            <p className="text-sm text-gray-600">Unlock unlimited spiritual guidance</p>
          </div>

          {/* Upgrade Button for Guests */}
          <button
            onClick={handleUpgrade}
            className="w-full py-3 px-4 rounded-lg text-white font-semibold transition-all duration-200 hover:shadow-lg transform hover:-translate-y-0.5 mb-3"
            style={{ 
              background: 'linear-gradient(135deg, #f59e0b 0%, #10b981 100%)',
              boxShadow: '0 4px 14px 0 rgba(245, 158, 11, 0.3)'
            }}
          >
            <div className="flex items-center justify-center space-x-2">
              <SparklesIcon className="h-4 w-4" />
              <span>Upgrade to Pro</span>
            </div>
          </button>

          {/* Guest Limitations */}
          <div className="bg-amber-50 rounded-lg p-3 border border-amber-200">
            <p className="text-xs font-semibold text-amber-800 mb-2">Guest Limitations:</p>
            <ul className="text-xs text-amber-700 space-y-1">
              <li>• 3 messages per session</li>
              <li>• No conversation history</li>
              <li>• Basic responses only</li>
            </ul>
            <p className="text-xs text-amber-600 mt-2 text-center">
              Use Sign In/Sign Up buttons above to get full access
            </p>
          </div>
        </div>
      </motion.div>
      
      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsModalOpen}
        onClose={() => setIsSettingsModalOpen(false)}
        user={user}
      />
      </>
    );
  }

  // Authenticated User Menu
  return (
    <>
      <motion.div
        initial={{ opacity: 0, y: 10, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 10, scale: 0.95 }}
      transition={{ duration: 0.15 }}
      className="absolute left-0 right-0 bottom-full mb-2 bg-white rounded-lg shadow-xl border border-gray-200 overflow-hidden z-50"
      style={{ 
        boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)' 
      }}
    >
      {/* User Info Header */}
      <div className="px-4 py-3 border-b border-gray-100 bg-gradient-to-r from-blue-50 to-emerald-50">
        <div className="flex items-center space-x-3">
          <div 
            className="w-12 h-12 rounded-full flex items-center justify-center ring-2 ring-white"
            style={{ backgroundColor: currentTheme.colors.primary + '20' }}
          >
            <span 
              className="text-xl font-semibold"
              style={{ color: currentTheme.colors.primary }}
            >
              {(user.first_name || user.name || user.email)?.[0]?.toUpperCase() || 'U'}
            </span>
          </div>
          <div className="flex-1">
            <p className="font-semibold text-gray-900">
              {user.first_name && user.last_name 
                ? `${user.first_name} ${user.last_name}` 
                : user.name || user.first_name || 'User'}
            </p>
            <p className="text-xs text-gray-500 font-mono">ID: {user.email?.split('@')[0] || 'user-001'}</p>
            <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium mt-1 ${
              (user.subscription_plan || user.plan) === 'basic' 
                ? 'bg-yellow-100 text-yellow-800' 
                : 'bg-green-100 text-green-800'
            }`}>
              {(user.subscription_plan || user.plan) === 'basic' ? (
                <svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                </svg>
              ) : (
                <svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
              )}
              {user.isGuest ? 'Guest User' :
               (user.subscription_plan || user.plan) === 'basic' ? 'Basic Plan' : 
               (user.subscription_plan || user.plan) === 'pro' ? 'Pro Plan' : 
               (user.subscription_plan || user.plan) === 'max' ? 'Max Plan' : 'Premium Plan'}
            </div>
          </div>
        </div>
      </div>

      {/* Menu Options */}
      <div className="py-2">
        {/* Upgrade Plan - Only for Basic Users */}
        {((user.subscription_plan || user.plan) === 'basic' || user.isGuest) && (
          <button
            onClick={handleUpgrade}
            className="w-full flex items-center px-4 py-3 text-sm font-medium text-left hover:bg-gradient-to-r hover:from-amber-50 hover:to-emerald-50 transition-all duration-200 group border-b border-gray-50"
          >
            <div className="w-8 h-8 rounded-lg bg-gradient-to-r from-amber-500 to-emerald-600 flex items-center justify-center mr-3 group-hover:scale-110 transition-transform">
              <SparklesIcon className="w-4 h-4 text-white" />
            </div>
            <div className="flex-1">
              <p className="font-semibold text-gray-900 group-hover:text-amber-700">Upgrade Plan</p>
              <p className="text-xs text-gray-500">Unlock unlimited features</p>
            </div>
            <div className="text-amber-500 group-hover:text-amber-600">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </div>
          </button>
        )}

        {/* My Conversations */}
        <button
          onClick={handleConversations}
          className="w-full flex items-center px-4 py-3 text-sm font-medium text-left hover:bg-gray-50 transition-colors group"
        >
          <div className="w-8 h-8 rounded-lg bg-indigo-100 flex items-center justify-center mr-3 group-hover:bg-indigo-200 transition-colors">
            <svg className="w-4 h-4 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
          </div>
          <div className="flex-1">
            <p className="font-medium text-gray-900 group-hover:text-gray-700">My Conversations</p>
            <p className="text-xs text-gray-500">View your chat history</p>
          </div>
          <div className="text-gray-400 group-hover:text-gray-600">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </div>
        </button>

        {/* Account Settings */}
        <button
          onClick={handleSettings}
          className="w-full flex items-center px-4 py-3 text-sm font-medium text-left hover:bg-gray-50 transition-colors group"
        >
          <div className="w-8 h-8 rounded-lg bg-blue-100 flex items-center justify-center mr-3 group-hover:bg-blue-200 transition-colors">
            <svg className="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </div>
          <div className="flex-1">
            <p className="font-medium text-gray-900 group-hover:text-gray-700">Account Settings</p>
            <p className="text-xs text-gray-500">Manage your profile & preferences</p>
          </div>
          <div className="text-gray-400 group-hover:text-gray-600">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </div>
        </button>

        {/* Manage Subscription - For Premium Users */}
        {((user.subscription_plan || user.plan) !== 'basic' && !user.isGuest) && (
          <button
            onClick={handleManageSubscription}
            className="w-full flex items-center px-4 py-3 text-sm font-medium text-left hover:bg-gray-50 transition-colors group"
          >
            <div className="w-8 h-8 rounded-lg bg-green-100 flex items-center justify-center mr-3 group-hover:bg-green-200 transition-colors">
              <svg className="w-4 h-4 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" />
              </svg>
            </div>
            <div className="flex-1">
              <p className="font-medium text-gray-900 group-hover:text-gray-700">Manage Subscription</p>
              <p className="text-xs text-gray-500">Update billing & plan details</p>
            </div>
            <div className="text-gray-400 group-hover:text-gray-600">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </div>
          </button>
        )}

        {/* Help & Support */}
        <button
          onClick={handleHelp}
          className="w-full flex items-center px-4 py-3 text-sm font-medium text-left hover:bg-gray-50 transition-colors group"
        >
          <div className="w-8 h-8 rounded-lg bg-purple-100 flex items-center justify-center mr-3 group-hover:bg-purple-200 transition-colors">
            <svg className="w-4 h-4 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div className="flex-1">
            <p className="font-medium text-gray-900 group-hover:text-gray-700">Help & Support</p>
            <p className="text-xs text-gray-500">Get help and contact support</p>
          </div>
          <div className="text-gray-400 group-hover:text-gray-600">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
            </svg>
          </div>
        </button>

        {/* Demo Mode Info - Only for demo users */}
        {isDemo && (
          <>
            <div className="border-t border-gray-100 my-2"></div>
            <div className="px-4 py-3 bg-blue-50 mx-2 rounded-lg">
              <div className="text-center">
                <div className="text-sm font-medium text-blue-800 mb-1">
                  🚀 Demo Mode
                </div>
                <p className="text-xs text-blue-600 mb-2">
                  You're exploring DharmaMind! Sign up to save your conversations and unlock full features.
                </p>
                <div className="flex space-x-2">
                  <button
                    onClick={() => {
                      onClose();
                      router.push('/auth?mode=signup');
                    }}
                    className="flex-1 px-2 py-1 bg-blue-600 text-white text-xs rounded-md hover:bg-blue-700 transition-colors"
                  >
                    Sign Up
                  </button>
                  <button
                    onClick={() => {
                      onClose();
                      router.push('/auth?mode=login');
                    }}
                    className="flex-1 px-2 py-1 border border-blue-600 text-blue-600 text-xs rounded-md hover:bg-blue-50 transition-colors"
                  >
                    Sign In
                  </button>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Divider - Only show if Sign Out will be displayed */}
        {!isDemo && (
          <div className="border-t border-gray-100 my-2"></div>
        )}

        {/* Logout - Only for non-demo users */}
        {!isDemo && (
          <button
            onClick={handleLogout}
            className="w-full flex items-center px-4 py-3 text-sm font-medium text-left hover:bg-red-50 transition-colors group"
          >
            <div className="w-8 h-8 rounded-lg bg-red-100 flex items-center justify-center mr-3 group-hover:bg-red-200 transition-colors">
              <svg className="w-4 h-4 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
              </svg>
            </div>
            <div className="flex-1">
              <p className="font-medium text-red-700 group-hover:text-red-800">Sign Out</p>
              <p className="text-xs text-red-500">Log out of your account</p>
            </div>
          </button>
        )}
      </div>
    </motion.div>
    
    {/* Settings Modal */}
    <SettingsModal
      isOpen={isSettingsModalOpen}
      onClose={() => setIsSettingsModalOpen(false)}
      user={user}
    />
    </>
  );
};

export default UserProfileMenu;
