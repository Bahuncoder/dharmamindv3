import React from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import Logo from '../components/Logo';
import { ContactForm } from '../components/CentralizedSupport';

const ContactPage: React.FC = () => {
  const router = useRouter();

  const handleFormSubmit = async (data: any) => {
    // Handle form submission
    console.log('Contact form submitted:', data);
    // You can add your API call here
  };

  return (
    <>
      <Head>
        <title>Contact Us - DharmaMind</title>
        <meta name="description" content="Get in touch with the DharmaMind team for support, questions, or feedback" />
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
                <button 
                  onClick={() => router.push('/help')}
                  className="text-gray-600 hover:text-gray-900 text-sm font-medium"
                >
                  Help
                </button>
              </nav>
            </div>
          </div>
        </header>

        {/* Content */}
        <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
            
            {/* Contact Form */}
            <div className="bg-white rounded-lg shadow-sm border border-neutral-200 p-8">
              <h1 className="text-3xl font-bold text-gray-900 mb-6">Get in Touch</h1>
              <p className="text-gray-600 mb-8">
                We'd love to hear from you. Send us a message and we'll respond as soon as possible.
              </p>

              <ContactForm 
                onSubmit={handleFormSubmit}
                compact={false}
              />
            </div>

            {/* Contact Information */}
            <div className="space-y-8">
              
              {/* Contact Details */}
              <div className="bg-white rounded-lg shadow-sm border border-neutral-200 p-8">
                <h2 className="text-xl font-bold text-gray-900 mb-6">Contact Information</h2>
                
                <div className="space-y-4">
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 text-gray-500 mt-1">
                      <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">Email</p>
                      <p className="text-gray-600">support@dharmamind.com</p>
                    </div>
                  </div>

                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 text-gray-500 mt-1">
                      <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">Response Time</p>
                      <p className="text-gray-600">Usually within 24 hours</p>
                    </div>
                  </div>

                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 text-gray-500 mt-1">
                      <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                      </svg>
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">Office</p>
                      <p className="text-gray-600">Remote-first team<br />Serving users globally</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* FAQ Quick Links */}
              <div className="bg-white rounded-lg shadow-sm border border-neutral-200 p-8">
                <h2 className="text-xl font-bold text-gray-900 mb-6">Quick Help</h2>
                
                <div className="space-y-3">
                  <button 
                    onClick={() => router.push('/help')}
                    className="w-full text-left p-3 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                  >
                    <p className="font-medium text-gray-900">Help Center</p>
                    <p className="text-sm text-gray-600">Browse our knowledge base</p>
                  </button>
                  
                  <button 
                    onClick={() => router.push('/docs')}
                    className="w-full text-left p-3 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                  >
                    <p className="font-medium text-gray-900">Documentation</p>
                    <p className="text-sm text-gray-600">API and integration guides</p>
                  </button>
                  
                  <button 
                    onClick={() => router.push('/status')}
                    className="w-full text-left p-3 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                  >
                    <p className="font-medium text-gray-900">System Status</p>
                    <p className="text-sm text-gray-600">Check service availability</p>
                  </button>
                </div>
              </div>
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

export default ContactPage;
