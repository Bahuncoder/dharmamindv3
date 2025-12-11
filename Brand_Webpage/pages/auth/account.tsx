/**
 * DharmaMind Account Settings
 * Central account management for all platforms
 */

import React, { useState } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { useSession, signOut } from 'next-auth/react';
import { useRouter } from 'next/router';

const AccountPage: React.FC = () => {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<'profile' | 'security' | 'billing'>('profile');

  // Redirect to login if not authenticated
  if (status === 'loading') {
    return (
      <div className="min-h-screen bg-neutral-50 flex items-center justify-center">
        <div className="text-neutral-500">Loading...</div>
      </div>
    );
  }

  if (status === 'unauthenticated') {
    router.push('/auth/login?callbackUrl=/auth/account');
    return null;
  }

  const handleSignOut = () => {
    signOut({ callbackUrl: '/' });
  };

  return (
    <>
      <Head>
        <title>Account - DharmaMind</title>
        <meta name="description" content="Manage your DharmaMind account" />
      </Head>

      <div className="min-h-screen bg-neutral-50">
        {/* Navigation */}
        <header className="bg-neutral-100 border-b border-neutral-300">
          <div className="max-w-4xl mx-auto px-6">
            <div className="flex items-center justify-between h-16">
              <Link href="/" className="flex items-center gap-2">
                <div className="w-8 h-8 bg-neutral-900 rounded-lg flex items-center justify-center">
                  <span className="text-white font-bold text-sm">D</span>
                </div>
                <span className="font-semibold text-neutral-900">DharmaMind</span>
              </Link>
              <button
                onClick={handleSignOut}
                className="text-sm text-neutral-500 hover:text-neutral-700"
              >
                Sign out
              </button>
            </div>
          </div>
        </header>

        <main className="max-w-4xl mx-auto px-6 py-12">
          <h1 className="text-2xl font-semibold text-neutral-900 mb-8">Account Settings</h1>

          {/* Tabs */}
          <div className="flex gap-6 border-b border-neutral-300 mb-8">
            {[
              { id: 'profile', label: 'Profile' },
              { id: 'security', label: 'Security' },
              { id: 'billing', label: 'Billing' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`pb-4 text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'text-neutral-900 border-b-2 border-neutral-900'
                    : 'text-neutral-500 hover:text-neutral-700'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Profile Tab */}
          {activeTab === 'profile' && (
            <div className="space-y-8">
              <div className="bg-neutral-100 p-6 rounded-xl border border-neutral-300">
                <h2 className="text-lg font-semibold text-neutral-900 mb-6">Profile Information</h2>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-neutral-700 mb-2">Name</label>
                    <input
                      type="text"
                      defaultValue={session?.user?.name || ''}
                      className="w-full px-4 py-3 border border-neutral-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-900"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-neutral-700 mb-2">Email</label>
                    <input
                      type="email"
                      defaultValue={session?.user?.email || ''}
                      className="w-full px-4 py-3 border border-neutral-300 rounded-lg bg-neutral-50"
                      disabled
                    />
                    <p className="mt-1 text-xs text-neutral-400">Email cannot be changed</p>
                  </div>
                </div>
                <button className="mt-6 px-6 py-2.5 bg-neutral-900 text-white text-sm font-medium rounded-lg hover:bg-neutral-800">
                  Save changes
                </button>
              </div>

              <div className="bg-neutral-100 p-6 rounded-xl border border-neutral-300">
                <h2 className="text-lg font-semibold text-neutral-900 mb-4">Connected Apps</h2>
                <p className="text-sm text-neutral-500 mb-4">
                  Your DharmaMind account gives you access to all our platforms:
                </p>
                <div className="space-y-3">
                  {[
                    { name: 'DharmaMind Chat', url: 'https://dharmamind.ai', status: 'Connected' },
                    { name: 'DharmaMind Community', url: 'https://community.dharmamind.org', status: 'Connected' },
                  ].map((app, i) => (
                    <div key={i} className="flex items-center justify-between p-3 bg-neutral-50 rounded-lg">
                      <div>
                        <p className="font-medium text-neutral-900">{app.name}</p>
                        <p className="text-xs text-neutral-500">{app.url}</p>
                      </div>
                      <span className="text-xs text-gold-600 font-medium">{app.status}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Security Tab */}
          {activeTab === 'security' && (
            <div className="space-y-8">
              <div className="bg-neutral-100 p-6 rounded-xl border border-neutral-300">
                <h2 className="text-lg font-semibold text-neutral-900 mb-6">Change Password</h2>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-neutral-700 mb-2">Current password</label>
                    <input
                      type="password"
                      className="w-full px-4 py-3 border border-neutral-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-900"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-neutral-700 mb-2">New password</label>
                    <input
                      type="password"
                      className="w-full px-4 py-3 border border-neutral-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-900"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-neutral-700 mb-2">Confirm new password</label>
                    <input
                      type="password"
                      className="w-full px-4 py-3 border border-neutral-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-900"
                    />
                  </div>
                </div>
                <button className="mt-6 px-6 py-2.5 bg-neutral-900 text-white text-sm font-medium rounded-lg hover:bg-neutral-800">
                  Update password
                </button>
              </div>

              <div className="bg-neutral-100 p-6 rounded-xl border border-neutral-300">
                <h2 className="text-lg font-semibold text-neutral-900 mb-4">Sessions</h2>
                <p className="text-sm text-neutral-500 mb-4">
                  Manage your active sessions across devices.
                </p>
                <div className="p-3 bg-neutral-50 rounded-lg flex items-center justify-between">
                  <div>
                    <p className="font-medium text-neutral-900">Current session</p>
                    <p className="text-xs text-neutral-500">This device</p>
                  </div>
                  <span className="text-xs text-gold-600 font-medium">Active</span>
                </div>
              </div>

              <div className="bg-neutral-100 p-6 rounded-xl border border-red-200">
                <h2 className="text-lg font-semibold text-red-600 mb-4">Danger Zone</h2>
                <p className="text-sm text-neutral-500 mb-4">
                  Permanently delete your account and all associated data.
                </p>
                <button className="px-6 py-2.5 bg-red-600 text-white text-sm font-medium rounded-lg hover:bg-red-700">
                  Delete account
                </button>
              </div>
            </div>
          )}

          {/* Billing Tab */}
          {activeTab === 'billing' && (
            <div className="space-y-8">
              <div className="bg-neutral-100 p-6 rounded-xl border border-neutral-300">
                <h2 className="text-lg font-semibold text-neutral-900 mb-4">Current Plan</h2>
                <div className="flex items-center justify-between p-4 bg-neutral-50 rounded-lg">
                  <div>
                    <p className="font-semibold text-neutral-900 capitalize">
                      {(session?.user as any)?.plan || 'Free'} Plan
                    </p>
                    <p className="text-sm text-neutral-500">
                      {(session?.user as any)?.plan === 'free' 
                        ? '10 conversations per month' 
                        : 'Unlimited conversations'}
                    </p>
                  </div>
                  <Link
                    href="/pricing"
                    className="px-4 py-2 bg-neutral-900 text-white text-sm font-medium rounded-lg hover:bg-neutral-800"
                  >
                    Upgrade
                  </Link>
                </div>
              </div>

              <div className="bg-neutral-100 p-6 rounded-xl border border-neutral-300">
                <h2 className="text-lg font-semibold text-neutral-900 mb-4">Payment Method</h2>
                <p className="text-sm text-neutral-500">No payment method on file.</p>
                <button className="mt-4 px-6 py-2.5 border border-neutral-300 text-neutral-700 text-sm font-medium rounded-lg hover:bg-neutral-50">
                  Add payment method
                </button>
              </div>

              <div className="bg-neutral-100 p-6 rounded-xl border border-neutral-300">
                <h2 className="text-lg font-semibold text-neutral-900 mb-4">Billing History</h2>
                <p className="text-sm text-neutral-500">No billing history available.</p>
              </div>
            </div>
          )}
        </main>
      </div>
    </>
  );
};

export default AccountPage;

