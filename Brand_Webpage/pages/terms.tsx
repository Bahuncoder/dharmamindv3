import React from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import Logo from '../components/Logo';
import BrandHeader from '../components/BrandHeader';

const TermsPage: React.FC = () => {
  const router = useRouter();

  return (
    <>
      <Head>
        <title>Terms of Service - DharmaMind</title>
        <meta name="description" content="DharmaMind Terms of Service - Legal terms and conditions for using our platform" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-section-light">
        {/* Professional Brand Header with Breadcrumbs */}
        <BrandHeader
          breadcrumbs={[
            { label: 'Home', href: '/' },
            { label: 'Terms of Service', href: '/terms' }
          ]}
        />

        {/* Content */}
        <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="bg-white rounded-lg shadow-sm border border-light p-8">
            <h1 className="text-3xl font-bold text-primary mb-8">Terms of Service</h1>

            <div className="prose prose-gray max-w-none">
              <p className="text-lg text-secondary mb-8">
                Last updated: {new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}
              </p>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold text-primary mb-4">Welcome to DharmaMind</h2>
                <p className="text-primary mb-4">
                  These Terms of Service ("Terms") govern your access to and use of DharmaMind's AI spiritual guidance
                  platform, including our website, mobile applications, and services (collectively, the "Service").
                </p>
                <p className="text-primary">
                  By accessing or using our Service, you agree to be bound by these Terms. If you disagree with any
                  part of these terms, you may not access the Service.
                </p>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold text-primary mb-4">Acceptance of Terms</h2>
                <p className="text-primary mb-4">
                  By creating an account or using our Service, you confirm that:
                </p>
                <ul className="list-disc list-inside text-primary space-y-2">
                  <li>You are at least 18 years old or have parental consent</li>
                  <li>You have the legal capacity to enter into these Terms</li>
                  <li>You will use the Service in accordance with these Terms</li>
                  <li>All information you provide is accurate and complete</li>
                </ul>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold text-primary mb-4">Description of Service</h2>
                <p className="text-primary mb-4">
                  DharmaMind provides AI-powered spiritual guidance and wisdom based on ancient teachings including:
                </p>
                <ul className="list-disc list-inside text-primary space-y-2">
                  <li>Interactive conversations with our AI spiritual guide</li>
                  <li>Personalized insights based on dharmic principles</li>
                  <li>Access to ancient wisdom from Gita, Upanishads, and Vedas</li>
                  <li>Spiritual growth tracking and recommendations</li>
                  <li>Community features and shared learning experiences</li>
                </ul>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold text-primary mb-4">User Accounts</h2>

                <h3 className="text-xl font-semibold text-primary mb-3">Account Creation</h3>
                <ul className="list-disc list-inside text-primary mb-4 space-y-2">
                  <li>You must provide accurate and complete information</li>
                  <li>You are responsible for maintaining account security</li>
                  <li>You must notify us immediately of any unauthorized access</li>
                  <li>One person may only create one account</li>
                </ul>

                <h3 className="text-xl font-semibold text-primary mb-3">Account Termination</h3>
                <p className="text-primary">
                  You may terminate your account at any time. We may suspend or terminate your account if you
                  violate these Terms or engage in harmful behavior.
                </p>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold text-primary mb-4">Acceptable Use</h2>

                <h3 className="text-xl font-semibold text-primary mb-3">You May:</h3>
                <ul className="list-disc list-inside text-primary mb-4 space-y-2">
                  <li>Use the Service for personal spiritual growth and learning</li>
                  <li>Share insights with others in accordance with our community guidelines</li>
                  <li>Provide feedback to help us improve the Service</li>
                  <li>Export your personal conversation history</li>
                </ul>

                <h3 className="text-xl font-semibold text-primary mb-3">You May Not:</h3>
                <ul className="list-disc list-inside text-primary space-y-2">
                  <li>Use the Service for illegal or harmful activities</li>
                  <li>Attempt to reverse engineer or exploit our AI technology</li>
                  <li>Share your account credentials with others</li>
                  <li>Harass, abuse, or harm other users</li>
                  <li>Distribute spam, malware, or harmful content</li>
                  <li>Violate any applicable laws or regulations</li>
                </ul>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold text-primary mb-4">Subscription and Billing</h2>

                <h3 className="text-xl font-semibold text-primary mb-3">Free Trial</h3>
                <p className="text-primary mb-4">
                  We offer a free trial period during which you can explore our basic features. No payment
                  information is required for the trial.
                </p>

                <h3 className="text-xl font-semibold text-primary mb-3">Paid Subscriptions</h3>
                <ul className="list-disc list-inside text-primary mb-4 space-y-2">
                  <li>Subscriptions are billed monthly or annually</li>
                  <li>Payment is due at the start of each billing cycle</li>
                  <li>You can cancel your subscription at any time</li>
                  <li>Refunds are provided according to our refund policy</li>
                </ul>

                <h3 className="text-xl font-semibold text-primary mb-3">Refund Policy</h3>
                <p className="text-primary">
                  We offer a 30-day money-back guarantee for annual subscriptions and a 7-day guarantee for
                  monthly subscriptions. Contact our support team to request a refund.
                </p>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold text-primary mb-4">Intellectual Property</h2>

                <h3 className="text-xl font-semibold text-primary mb-3">Our Content</h3>
                <p className="text-primary mb-4">
                  The Service, including all AI responses, spiritual content, and platform features, is owned by
                  DharmaMind and protected by intellectual property laws.
                </p>

                <h3 className="text-xl font-semibold text-primary mb-3">Your Content</h3>
                <p className="text-primary mb-4">
                  You retain ownership of your conversations and personal data. By using our Service, you grant us
                  a license to process your conversations to provide and improve our Service.
                </p>

                <h3 className="text-xl font-semibold text-primary mb-3">Ancient Wisdom</h3>
                <p className="text-primary">
                  The spiritual teachings from Gita, Upanishads, and Vedas are part of humanity's shared heritage.
                  Our AI interpretations and presentations of this wisdom are our intellectual property.
                </p>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold text-primary mb-4">Privacy and Data</h2>
                <p className="text-primary mb-4">
                  Your privacy is important to us. Please review our Privacy Policy to understand how we collect,
                  use, and protect your information.
                </p>
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <p className="text-blue-800">
                    <strong>Chat History Control:</strong> You have complete control over your conversation history.
                    You can view, export, or delete your chats at any time through your account settings.
                  </p>
                </div>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold text-primary mb-4">AI Limitations & Disclaimers</h2>

                <div className="bg-primary-clean border border-stone-200 rounded-lg p-4 mb-4">
                  <h3 className="text-lg font-semibold text-primary mb-2">Important Notice</h3>
                  <p className="text-primary">
                    DharmaMind provides AI-generated spiritual guidance for educational and personal growth purposes.
                    Our AI is not a substitute for professional counseling, therapy, or medical advice.
                  </p>
                </div>

                <ul className="list-disc list-inside text-primary space-y-2">
                  <li>AI responses are generated based on training data and algorithms</li>
                  <li>Responses may not always be accurate or appropriate for your situation</li>
                  <li>Use your own judgment when applying AI guidance to your life</li>
                  <li>Seek professional help for serious mental health or spiritual concerns</li>
                  <li>We are not responsible for decisions made based on AI responses</li>
                </ul>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold text-primary mb-4">Limitation of Liability</h2>
                <p className="text-primary mb-4">
                  To the fullest extent permitted by law, DharmaMind shall not be liable for any indirect,
                  incidental, special, consequential, or punitive damages arising from your use of the Service.
                </p>
                <p className="text-primary">
                  Our total liability for any claims related to the Service shall not exceed the amount you
                  paid for the Service in the 12 months preceding the claim.
                </p>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold text-primary mb-4">Changes to Terms</h2>
                <p className="text-primary mb-4">
                  We may update these Terms from time to time. We will notify you of material changes by:
                </p>
                <ul className="list-disc list-inside text-primary space-y-2">
                  <li>Email notification to your registered address</li>
                  <li>In-app notification when you next use the Service</li>
                  <li>Updated "Last modified" date at the top of these Terms</li>
                </ul>
              </section>

              <section className="mb-8">
                <h2 className="text-2xl font-semibold text-primary mb-4">Contact Information</h2>
                <p className="text-primary mb-4">
                  If you have questions about these Terms, please contact us:
                </p>
                <div className="bg-brand-primary rounded-lg p-4">
                  <p className="text-primary"><strong>Email:</strong> legal@dharmamind.com</p>
                  <p className="text-primary"><strong>Support:</strong> support@dharmamind.com</p>
                  <p className="text-primary"><strong>Address:</strong> DharmaMind Legal Team</p>
                </div>
              </section>

              <section>
                <h2 className="text-2xl font-semibold text-primary mb-4">Governing Law</h2>
                <p className="text-primary">
                  These Terms shall be governed by and construed in accordance with applicable laws.
                  Any disputes arising from these Terms or your use of the Service shall be resolved
                  through binding arbitration.
                </p>
              </section>
            </div>
          </div>
        </main>

        {/* Footer */}
        <footer className="border-t border-light bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <div className="text-center">
              <Logo
                size="sm"
                showText={true}
                onClick={() => router.push('/welcome')}
                className="justify-center mb-4"
              />
              <p className="text-sm text-secondary">
                © 2025 DharmaMind. All rights reserved.
              </p>
            </div>
          </div>
        </footer>
      </div>
    </>
  );
};

export default TermsPage;
