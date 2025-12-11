/**
 * DharmaMind Contact - Professional Contact Page
 * Uses unified Layout component for consistent header/footer
 */

import React, { useState } from 'react';
import Layout from '../components/layout/Layout';

const ContactPage: React.FC = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    company: '',
    message: '',
    type: 'general',
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Handle form submission
    console.log('Form submitted:', formData);
  };

  return (
    <Layout
      title="Contact"
      description="Get in touch with the DharmaMind team. We are here to help."
    >
      {/* Hero */}
      <section className="pt-32 pb-16 px-6 bg-neutral-100">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-5xl font-semibold text-neutral-900 leading-tight tracking-tight mb-6">
            Get in touch
          </h1>
          <p className="text-xl text-neutral-600 max-w-2xl mx-auto leading-relaxed">
            Have a question or want to learn more? We would love to hear from you.
          </p>
        </div>
      </section>

      {/* Contact Form */}
      <section className="py-12 px-6 bg-neutral-100">
        <div className="max-w-2xl mx-auto">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <label htmlFor="name" className="block text-sm font-medium text-neutral-900 mb-2">
                  Name
                </label>
                <input
                  type="text"
                  id="name"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="w-full px-4 py-3 bg-white border border-neutral-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-gold-600 focus:border-transparent text-neutral-900"
                  placeholder="Your name"
                  required
                />
              </div>
              <div>
                <label htmlFor="email" className="block text-sm font-medium text-neutral-900 mb-2">
                  Email
                </label>
                <input
                  type="email"
                  id="email"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  className="w-full px-4 py-3 bg-white border border-neutral-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-gold-600 focus:border-transparent text-neutral-900"
                  placeholder="you@company.com"
                  required
                />
              </div>
            </div>

            <div>
              <label htmlFor="company" className="block text-sm font-medium text-neutral-900 mb-2">
                Company (optional)
              </label>
              <input
                type="text"
                id="company"
                value={formData.company}
                onChange={(e) => setFormData({ ...formData, company: e.target.value })}
                className="w-full px-4 py-3 bg-white border border-neutral-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-gold-600 focus:border-transparent text-neutral-900"
                placeholder="Your company"
              />
            </div>

            <div>
              <label htmlFor="type" className="block text-sm font-medium text-neutral-900 mb-2">
                How can we help?
              </label>
              <select
                id="type"
                value={formData.type}
                onChange={(e) => setFormData({ ...formData, type: e.target.value })}
                className="w-full px-4 py-3 bg-white border border-neutral-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-gold-600 focus:border-transparent text-neutral-900"
              >
                <option value="general">General inquiry</option>
                <option value="sales">Sales / Enterprise</option>
                <option value="support">Support</option>
                <option value="partnership">Partnership</option>
                <option value="press">Press / Media</option>
              </select>
            </div>

            <div>
              <label htmlFor="message" className="block text-sm font-medium text-neutral-900 mb-2">
                Message
              </label>
              <textarea
                id="message"
                value={formData.message}
                onChange={(e) => setFormData({ ...formData, message: e.target.value })}
                rows={5}
                className="w-full px-4 py-3 bg-white border border-neutral-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-gold-600 focus:border-transparent resize-none text-neutral-900"
                placeholder="Tell us how we can help..."
                required
              />
            </div>

            <button
              type="submit"
              className="w-full px-6 py-3 bg-gold-600 text-white font-medium rounded-lg hover:bg-gold-700 transition-colors"
            >
              Send message
            </button>
          </form>
        </div>
      </section>

      {/* Contact Info */}
      <section className="py-20 px-6 bg-neutral-50">
        <div className="max-w-4xl mx-auto">
          <div className="grid md:grid-cols-3 gap-8 text-center">
            <div>
              <div className="w-12 h-12 bg-gold-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                <svg className="w-6 h-6 text-gold-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
              </div>
              <h3 className="font-semibold text-neutral-900 mb-1">Email</h3>
              <p className="text-neutral-600">hello@dharmamind.ai</p>
            </div>
            <div>
              <div className="w-12 h-12 bg-gold-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                <svg className="w-6 h-6 text-gold-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17 8h2a2 2 0 012 2v6a2 2 0 01-2 2h-2v4l-4-4H9a1.994 1.994 0 01-1.414-.586m0 0L11 14h4a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2v4l.586-.586z" />
                </svg>
              </div>
              <h3 className="font-semibold text-neutral-900 mb-1">Community</h3>
              <a href="https://dharmamind.org" target="_blank" rel="noopener noreferrer" className="text-gold-600 hover:text-gold-700 transition-colors">dharmamind.org</a>
            </div>
            <div>
              <div className="w-12 h-12 bg-gold-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                <svg className="w-6 h-6 text-gold-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="font-semibold text-neutral-900 mb-1">Response time</h3>
              <p className="text-neutral-600">Within 24 hours</p>
            </div>
          </div>
        </div>
      </section>
    </Layout>
  );
};

export default ContactPage;
