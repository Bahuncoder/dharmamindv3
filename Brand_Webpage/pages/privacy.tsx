import React from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import Logo from '../components/Logo';
import { ContactButton } from '../components/CentralizedSupport';

const PrivacyPage: React.FC = () => {
  const router = useRouter();

  return (
    <>
      <Head>
        <title>Privacy Policy - DharmaMind</title>
        <meta name="description" content="DharmaMind Privacy Policy - How we protect and handle your data" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="border-b border-gray-200 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center space-x-4">
                <button
                  onClick={() => router.back()}
                  className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                  <span className="font-medium">Back</span>
                </button>
                
                <div className="h-6 w-px bg-gray-300"></div>
                
                <Logo 
                  size="sm"
                  showText={true}
                  onClick={() => router.push('/')}
                  className="cursor-pointer"
                />
              </div>

              <nav className="flex items-center space-x-8">
                <button 
                  onClick={() => router.push('/')}
                  className="text-gray-600 hover:text-gray-900 text-sm font-medium"
                >
                  Home
                </button>
                <ContactButton 
                  variant="link"
                  prefillCategory="general"
                  className="text-gray-600 hover:text-gray-900 text-sm font-medium"
                >
                  Contact
                </ContactButton>
              </nav>
            </div>
          </div>
        </header>

        {/* Content */}
        <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
            <h1 className="text-3xl font-bold text-gray-900 mb-8">Privacy Policy</h1>
            
            <div className="prose prose-gray max-w-none">
              <p className="text-lg text-gray-600 mb-8">
                Last updated: {new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}
              </p>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold text-gray-900 mb-4">Our Commitment to Privacy</h2>
                <p className="text-gray-700 mb-4">
                  At DharmaMind, we are committed to protecting your privacy and personal information. This Privacy Policy 
                  explains how we collect, use, and safeguard your data when you use our AI spiritual guidance platform.
                </p>
                <p className="text-gray-700">
                  We believe that privacy is a fundamental right, and we have designed our services with privacy-by-design 
                  principles, ensuring that your spiritual journey remains personal and secure.
                </p>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold text-gray-900 mb-4">Information We Collect</h2>
                
                <h3 className="text-xl font-semibold text-gray-800 mb-3">Account Information</h3>
                <ul className="list-disc list-inside text-gray-700 mb-4 space-y-2">
                  <li>Email address for account creation and communication</li>
                  <li>Name for personalization of your experience</li>
                  <li>Password (encrypted and never stored in plain text)</li>
                  <li>Subscription and billing information</li>
                </ul>

                <h3 className="text-xl font-semibold text-gray-800 mb-3">Conversation Data</h3>
                <ul className="list-disc list-inside text-gray-700 mb-4 space-y-2">
                  <li>Chat messages and conversations with our AI</li>
                  <li>Spiritual preferences and interests</li>
                  <li>Usage patterns to improve our service</li>
                  <li>Session data and interaction history</li>
                </ul>

                <h3 className="text-xl font-semibold text-gray-800 mb-3">Technical Information</h3>
                <ul className="list-disc list-inside text-gray-700 space-y-2">
                  <li>Device information and browser type</li>
                  <li>IP address and location data (generalized)</li>
                  <li>Usage analytics and performance metrics</li>
                  <li>Error logs and debugging information</li>
                </ul>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold text-gray-900 mb-4">How We Use Your Information</h2>
                <ul className="list-disc list-inside text-gray-700 space-y-2">
                  <li><strong>Provide Services:</strong> To deliver personalized AI spiritual guidance</li>
                  <li><strong>Improve Experience:</strong> To enhance our AI responses and platform functionality</li>
                  <li><strong>Communication:</strong> To send important updates and spiritual insights</li>
                  <li><strong>Security:</strong> To protect against fraud and unauthorized access</li>
                  <li><strong>Analytics:</strong> To understand usage patterns and improve our service</li>
                  <li><strong>Support:</strong> To provide customer support and resolve issues</li>
                </ul>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold text-gray-900 mb-4">Data Protection & Security</h2>
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                  <p className="text-blue-800 font-medium">🔒 Your spiritual conversations are encrypted and protected</p>
                </div>
                <ul className="list-disc list-inside text-gray-700 space-y-2">
                  <li>End-to-end encryption for all conversations</li>
                  <li>Secure cloud storage with industry-standard protection</li>
                  <li>Regular security audits and vulnerability assessments</li>
                  <li>Limited access controls for our team members</li>
                  <li>Automatic logout and session management</li>
                </ul>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold text-gray-900 mb-4">Your Privacy Rights</h2>
                <p className="text-gray-700 mb-4">You have the following rights regarding your personal data:</p>
                <ul className="list-disc list-inside text-gray-700 space-y-2">
                  <li><strong>Access:</strong> Request a copy of your personal data</li>
                  <li><strong>Correction:</strong> Update or correct inaccurate information</li>
                  <li><strong>Deletion:</strong> Request deletion of your account and data</li>
                  <li><strong>Portability:</strong> Export your conversation history</li>
                  <li><strong>Objection:</strong> Object to certain data processing activities</li>
                  <li><strong>Restriction:</strong> Request limitation of data processing</li>
                </ul>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold text-gray-900 mb-4">Chat History & User Control</h2>
                <p className="text-gray-700 mb-4">
                  We understand that your spiritual conversations are deeply personal. You have complete control over your chat history:
                </p>
                <ul className="list-disc list-inside text-gray-700 space-y-2">
                  <li>View all your conversation history in your account settings</li>
                  <li>Clear individual conversations or entire chat history</li>
                  <li>Export your conversations for personal records</li>
                  <li>Control whether we learn from your interactions to improve responses</li>
                  <li>Set privacy preferences for data retention</li>
                </ul>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold text-gray-900 mb-4">Data Sharing</h2>
                <p className="text-gray-700 mb-4">
                  We do not sell, rent, or trade your personal information. We may share data only in these limited circumstances:
                </p>
                <ul className="list-disc list-inside text-gray-700 space-y-2">
                  <li><strong>Service Providers:</strong> Trusted partners who help us operate our platform</li>
                  <li><strong>Legal Requirements:</strong> When required by law or to protect our rights</li>
                  <li><strong>Business Transfers:</strong> In case of merger or acquisition (with notice)</li>
                  <li><strong>Consent:</strong> When you explicitly provide consent</li>
                </ul>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold text-gray-900 mb-4">Cookies & Analytics</h2>
                <p className="text-gray-700 mb-4">
                  We use cookies and similar technologies to improve your experience:
                </p>
                <ul className="list-disc list-inside text-gray-700 space-y-2">
                  <li>Essential cookies for platform functionality</li>
                  <li>Analytics cookies to understand usage patterns</li>
                  <li>Preference cookies to remember your settings</li>
                  <li>Security cookies to protect against fraud</li>
                </ul>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold text-gray-900 mb-4">Contact Us</h2>
                <p className="text-gray-700 mb-4">
                  If you have any questions about this Privacy Policy or your data, please contact us:
                </p>
                <div className="bg-gray-100 rounded-lg p-4">
                  <p className="text-gray-800"><strong>Email:</strong> privacy@dharmamind.com</p>
                  <p className="text-gray-800"><strong>Support:</strong> support@dharmamind.com</p>
                  <p className="text-gray-800"><strong>Address:</strong> DharmaMind Privacy Team</p>
                </div>
              </section>

              <section>
                <h2 className="text-2xl font-semibold text-gray-900 mb-4">Changes to This Policy</h2>
                <p className="text-gray-700">
                  We may update this Privacy Policy from time to time. We will notify you of any material changes 
                  by email or through our platform. Your continued use of DharmaMind after such changes constitutes 
                  acceptance of the updated policy.
                </p>
              </section>
            </div>
          </div>
        </main>

        {/* Footer */}
        <footer className="border-t border-gray-200 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <div className="text-center">
              <Logo 
                size="sm"
                showText={true}
                onClick={() => router.push('/welcome')}
                className="justify-center mb-4"
              />
              <p className="text-sm text-slate-600">
                © 2025 DharmaMind. All rights reserved.
              </p>
            </div>
          </div>
        </footer>
      </div>
    </>
  );
};

export default PrivacyPage;
