import React, { useState } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import Logo from '../components/Logo';

interface FormData {
  firstName: string;
  lastName: string;
  email: string;
  company: string;
  jobTitle: string;
  phone: string;
  companySize: string;
  industry: string;
  useCase: string;
  preferredDate: string;
  preferredTime: string;
  timeZone: string;
  additionalInfo: string;
}

const DemoRequestPage: React.FC = () => {
  const router = useRouter();
  const [formData, setFormData] = useState<FormData>({
    firstName: '',
    lastName: '',
    email: '',
    company: '',
    jobTitle: '',
    phone: '',
    companySize: '',
    industry: '',
    useCase: '',
    preferredDate: '',
    preferredTime: '',
    timeZone: '',
    additionalInfo: ''
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    // Simulate API call
    setTimeout(() => {
      setIsSubmitting(false);
      setSubmitted(true);
    }, 2000);
  };

  if (submitted) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-stone-50 via-amber-50 to-emerald-50">
        <Head>
          <title>Demo Request Submitted - DharmaMind Enterprise</title>
          <meta name="description" content="Your enterprise demo request has been submitted successfully" />
        </Head>

        {/* Header */}
        <header className="bg-white/80 backdrop-blur-sm border-b border-stone-200/50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              <Logo 
                size="sm"
                onClick={() => router.push('/')}
              />
              <button 
                onClick={() => router.push('/')}
                className="text-stone-600 hover:text-stone-900"
              >
                Back to Home
              </button>
            </div>
          </div>
        </header>

        {/* Success Message */}
        <div className="max-w-2xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="text-center">
            <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-6">
              <svg className="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
              </svg>
            </div>
            <h1 className="text-3xl font-bold text-gray-900 mb-4">
              🙏 Demo Request Submitted Successfully
            </h1>
            <div className="bg-white rounded-lg p-8 shadow-sm border border-stone-200 mb-8">
              <p className="text-lg text-gray-700 mb-4">
                Thank you for your interest in DharmaMind Enterprise! Our team will review your request and reach out within 24 hours to schedule your personalized demo.
              </p>
              <div className="bg-amber-50 rounded-lg p-4 border border-amber-200">
                <h3 className="font-semibold text-amber-800 mb-2">What happens next?</h3>
                <ul className="text-amber-700 text-sm space-y-1">
                  <li>• Our enterprise specialist will contact you within 24 hours</li>
                  <li>• We'll schedule a 45-minute personalized demo at your preferred time</li>
                  <li>• You'll receive a custom demo environment to explore</li>
                  <li>• We'll discuss your specific use cases and requirements</li>
                </ul>
              </div>
            </div>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button
                onClick={() => router.push('/')}
                className="bg-gradient-to-r from-amber-600 via-orange-500 to-red-500 text-white px-6 py-3 rounded-lg font-medium hover:from-amber-700 hover:via-orange-600 hover:to-red-600 transition-all duration-300"
              >
                Back to Home
              </button>
              <button
                onClick={() => router.push('/technical-specs')}
                className="border-2 border-amber-600 text-amber-700 px-6 py-3 rounded-lg font-medium hover:bg-amber-50 transition-all duration-300"
              >
                Download Technical Specs
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-stone-50 via-amber-50 to-emerald-50">
      <Head>
        <title>Schedule Enterprise Demo - DharmaMind</title>
        <meta name="description" content="Schedule a personalized demo of DharmaMind Enterprise for your organization" />
      </Head>

      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-stone-200/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Logo 
              size="sm"
              onClick={() => router.push('/')}
            />
            <button 
              onClick={() => router.push('/')}
              className="text-stone-600 hover:text-stone-900"
            >
              Back to Home
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            🏢 Schedule Your Enterprise Demo
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Experience how DharmaMind can transform your organization's decision-making 
            and employee well-being with AI-powered spiritual wisdom.
          </p>
        </div>

        {/* Demo Benefits */}
        <div className="grid md:grid-cols-3 gap-6 mb-12">
          <div className="bg-white rounded-lg p-6 shadow-sm border border-stone-200">
            <div className="text-3xl mb-3">🎯</div>
            <h3 className="font-semibold text-gray-900 mb-2">Personalized Demo</h3>
            <p className="text-gray-600 text-sm">Tailored to your industry and specific use cases</p>
          </div>
          <div className="bg-white rounded-lg p-6 shadow-sm border border-stone-200">
            <div className="text-3xl mb-3">⚡</div>
            <h3 className="font-semibold text-gray-900 mb-2">45-Minute Session</h3>
            <p className="text-gray-600 text-sm">Comprehensive overview and Q&A with our experts</p>
          </div>
          <div className="bg-white rounded-lg p-6 shadow-sm border border-stone-200">
            <div className="text-3xl mb-3">🔧</div>
            <h3 className="font-semibold text-gray-900 mb-2">Custom Environment</h3>
            <p className="text-gray-600 text-sm">Test environment configured for your organization</p>
          </div>
        </div>

        {/* Form */}
        <div className="bg-white rounded-lg shadow-lg border border-stone-200 p-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Contact Information */}
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Contact Information</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    First Name *
                  </label>
                  <input
                    type="text"
                    name="firstName"
                    required
                    value={formData.firstName}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Last Name *
                  </label>
                  <input
                    type="text"
                    name="lastName"
                    required
                    value={formData.lastName}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Business Email *
                  </label>
                  <input
                    type="email"
                    name="email"
                    required
                    value={formData.email}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Phone Number
                  </label>
                  <input
                    type="tel"
                    name="phone"
                    value={formData.phone}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                  />
                </div>
              </div>
            </div>

            {/* Company Information */}
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Company Information</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Company Name *
                  </label>
                  <input
                    type="text"
                    name="company"
                    required
                    value={formData.company}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Job Title *
                  </label>
                  <input
                    type="text"
                    name="jobTitle"
                    required
                    value={formData.jobTitle}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Company Size *
                  </label>
                  <select
                    name="companySize"
                    required
                    value={formData.companySize}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                  >
                    <option value="">Select company size</option>
                    <option value="1-10">1-10 employees</option>
                    <option value="11-50">11-50 employees</option>
                    <option value="51-200">51-200 employees</option>
                    <option value="201-1000">201-1,000 employees</option>
                    <option value="1001-5000">1,001-5,000 employees</option>
                    <option value="5000+">5,000+ employees</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Industry *
                  </label>
                  <select
                    name="industry"
                    required
                    value={formData.industry}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                  >
                    <option value="">Select industry</option>
                    <option value="technology">Technology</option>
                    <option value="healthcare">Healthcare</option>
                    <option value="finance">Financial Services</option>
                    <option value="education">Education</option>
                    <option value="manufacturing">Manufacturing</option>
                    <option value="retail">Retail</option>
                    <option value="consulting">Consulting</option>
                    <option value="government">Government</option>
                    <option value="nonprofit">Non-profit</option>
                    <option value="other">Other</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Demo Details */}
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Demo Preferences</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Primary Use Case *
                  </label>
                  <select
                    name="useCase"
                    required
                    value={formData.useCase}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                  >
                    <option value="">Select primary use case</option>
                    <option value="employee-wellness">Employee Wellness & Mental Health</option>
                    <option value="leadership-guidance">Leadership & Decision Making</option>
                    <option value="conflict-resolution">Conflict Resolution</option>
                    <option value="team-building">Team Building & Culture</option>
                    <option value="change-management">Change Management</option>
                    <option value="customer-support">Customer Support Enhancement</option>
                    <option value="training-development">Training & Development</option>
                    <option value="other">Other</option>
                  </select>
                </div>
                
                <div className="grid md:grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Preferred Date
                    </label>
                    <input
                      type="date"
                      name="preferredDate"
                      value={formData.preferredDate}
                      onChange={handleInputChange}
                      min={new Date().toISOString().split('T')[0]}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Preferred Time
                    </label>
                    <select
                      name="preferredTime"
                      value={formData.preferredTime}
                      onChange={handleInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                    >
                      <option value="">Select time</option>
                      <option value="9:00 AM">9:00 AM</option>
                      <option value="10:00 AM">10:00 AM</option>
                      <option value="11:00 AM">11:00 AM</option>
                      <option value="1:00 PM">1:00 PM</option>
                      <option value="2:00 PM">2:00 PM</option>
                      <option value="3:00 PM">3:00 PM</option>
                      <option value="4:00 PM">4:00 PM</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Time Zone *
                    </label>
                    <select
                      name="timeZone"
                      required
                      value={formData.timeZone}
                      onChange={handleInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                    >
                      <option value="">Select timezone</option>
                      <option value="PST">Pacific (PST)</option>
                      <option value="MST">Mountain (MST)</option>
                      <option value="CST">Central (CST)</option>
                      <option value="EST">Eastern (EST)</option>
                      <option value="GMT">GMT</option>
                      <option value="CET">Central European</option>
                      <option value="JST">Japan Standard</option>
                      <option value="AEST">Australian Eastern</option>
                    </select>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Additional Information
                  </label>
                  <textarea
                    name="additionalInfo"
                    rows={4}
                    value={formData.additionalInfo}
                    onChange={handleInputChange}
                    placeholder="Tell us about your specific requirements, challenges, or questions..."
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                  />
                </div>
              </div>
            </div>

            {/* Submit Button */}
            <div className="pt-6 border-t border-gray-200">
              <button
                type="submit"
                disabled={isSubmitting}
                className={`w-full bg-gradient-to-r from-amber-600 via-orange-500 to-red-500 text-white px-8 py-4 rounded-lg text-lg font-medium transition-all duration-300 shadow-lg ${
                  isSubmitting 
                    ? 'opacity-50 cursor-not-allowed' 
                    : 'hover:from-amber-700 hover:via-orange-600 hover:to-red-600 hover:shadow-xl'
                }`}
              >
                {isSubmitting ? (
                  <div className="flex items-center justify-center">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                    Submitting Request...
                  </div>
                ) : (
                  'Schedule Enterprise Demo'
                )}
              </button>
              <p className="text-xs text-gray-500 text-center mt-3">
                By submitting this form, you agree to our Terms of Service and Privacy Policy
              </p>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default DemoRequestPage;
