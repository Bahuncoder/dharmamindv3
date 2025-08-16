import React from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import Logo from '../components/Logo';
import { ContactButton } from '../components/CentralizedSupport';

const SecurityPage: React.FC = () => {
  const router = useRouter();

  const securityFeatures = [
    {
      icon: '🔐',
      title: 'End-to-End Encryption',
      description: 'All conversations are encrypted with AES-256 encryption, ensuring your spiritual discussions remain private and secure.'
    },
    {
      icon: '🛡️',
      title: 'Enterprise Security',
      description: 'SOC 2 Type II compliant infrastructure with regular penetration testing and security audits.'
    },
    {
      icon: '🔒',
      title: 'Zero-Knowledge Architecture',
      description: 'We use zero-knowledge principles - your personal conversations cannot be accessed by our team.'
    },
    {
      icon: '🌐',
      title: 'GDPR & CCPA Compliant',
      description: 'Full compliance with global privacy regulations including GDPR, CCPA, and other data protection laws.'
    },
    {
      icon: '🔑',
      title: 'Multi-Factor Authentication',
      description: 'Secure your account with MFA using authenticator apps, SMS, or hardware security keys.'
    },
    {
      icon: '🏛️',
      title: 'Data Sovereignty',
      description: 'Choose where your data is stored with region-specific data centers and on-premise deployment options.'
    }
  ];

  const certifications = [
    { name: 'SOC 2 Type II', status: 'Certified', icon: '✅' },
    { name: 'ISO 27001', status: 'In Progress', icon: '🔄' },
    { name: 'HIPAA', status: 'Compliant', icon: '✅' },
    { name: 'PCI DSS', status: 'Certified', icon: '✅' }
  ];

  return (
    <>
      <Head>
        <title>Security - DharmaMind</title>
        <meta name="description" content="Learn about DharmaMind's security measures, compliance, and data protection practices" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="border-b border-gray-200 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <button 
                onClick={() => router.push('/')}
                className="hover:opacity-80 transition-opacity"
              >
                <Logo size="sm" showText={true} />
              </button>

              <nav className="flex items-center space-x-8">
                <button 
                  onClick={() => router.push('/')}
                  className="text-gray-600 hover:text-gray-900 text-sm font-medium"
                >
                  Home
                </button>
                <button 
                  onClick={() => router.push('/privacy')}
                  className="text-gray-600 hover:text-gray-900 text-sm font-medium"
                >
                  Privacy
                </button>
                <ContactButton 
                  variant="link"
                  prefillCategory="general"
                  className="text-gray-600 hover:text-gray-900 text-sm font-medium"
                >
                  Contact
                </ContactButton>
                <button 
                  onClick={() => router.push('/auth?mode=login')}
                  className="bg-gradient-to-r from-amber-600 to-emerald-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:from-amber-700 hover:to-emerald-700 transition-all duration-300"
                >
                  Sign In
                </button>
              </nav>
            </div>
          </div>
        </header>

        {/* Hero Section */}
        <div className="bg-gradient-to-br from-amber-50 to-emerald-50 py-16">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <div className="w-16 h-16 mx-auto mb-6 bg-gradient-to-r from-amber-600 to-emerald-600 rounded-full flex items-center justify-center">
              <span className="text-white text-2xl">🛡️</span>
            </div>
            <h1 className="text-4xl font-bold text-gray-900 mb-4">
              Enterprise-Grade Security
            </h1>
            <p className="text-xl text-gray-600 mb-8">
              Your spiritual journey deserves the highest level of privacy and protection. 
              Learn how we safeguard your conversations and data.
            </p>
            
            <div className="inline-flex items-center px-6 py-3 bg-white rounded-lg shadow-sm border border-gray-200">
              <span className="text-green-600 mr-2">✅</span>
              <span className="font-medium text-gray-900">
                Zero security incidents in our history
              </span>
            </div>
          </div>
        </div>

        {/* Security Features */}
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-12 text-center">
            Built with Security First
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {securityFeatures.map((feature, index) => (
              <div key={index} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
                <div className="w-12 h-12 mb-4 bg-gradient-to-r from-amber-100 to-emerald-100 rounded-lg flex items-center justify-center">
                  <span className="text-2xl">{feature.icon}</span>
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-3">{feature.title}</h3>
                <p className="text-gray-600 leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Technical Details */}
        <div className="bg-white border-y border-gray-200">
          <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-12 text-center">
              Technical Security Details
            </h2>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
              
              {/* Data Protection */}
              <div>
                <h3 className="text-xl font-semibold text-gray-900 mb-6">🔒 Data Protection</h3>
                <div className="space-y-4">
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-emerald-600 rounded-full mt-2"></div>
                    <div>
                      <div className="font-medium text-gray-900">Encryption at Rest</div>
                      <div className="text-gray-600 text-sm">AES-256 encryption for all stored data</div>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-emerald-600 rounded-full mt-2"></div>
                    <div>
                      <div className="font-medium text-gray-900">Encryption in Transit</div>
                      <div className="text-gray-600 text-sm">TLS 1.3 for all data transmission</div>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-emerald-600 rounded-full mt-2"></div>
                    <div>
                      <div className="font-medium text-gray-900">Key Management</div>
                      <div className="text-gray-600 text-sm">Hardware Security Modules (HSMs)</div>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-emerald-600 rounded-full mt-2"></div>
                    <div>
                      <div className="font-medium text-gray-900">Data Retention</div>
                      <div className="text-gray-600 text-sm">Configurable retention policies</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Infrastructure Security */}
              <div>
                <h3 className="text-xl font-semibold text-gray-900 mb-6">🏗️ Infrastructure Security</h3>
                <div className="space-y-4">
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-blue-600 rounded-full mt-2"></div>
                    <div>
                      <div className="font-medium text-gray-900">Network Security</div>
                      <div className="text-gray-600 text-sm">VPC isolation, firewalls, DDoS protection</div>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-blue-600 rounded-full mt-2"></div>
                    <div>
                      <div className="font-medium text-gray-900">Access Control</div>
                      <div className="text-gray-600 text-sm">Role-based access with principle of least privilege</div>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-blue-600 rounded-full mt-2"></div>
                    <div>
                      <div className="font-medium text-gray-900">Monitoring</div>
                      <div className="text-gray-600 text-sm">24/7 security monitoring and alerting</div>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-blue-600 rounded-full mt-2"></div>
                    <div>
                      <div className="font-medium text-gray-900">Backups</div>
                      <div className="text-gray-600 text-sm">Encrypted, geographically distributed backups</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Compliance & Certifications */}
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-12 text-center">
            Compliance & Certifications
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
            {certifications.map((cert, index) => (
              <div key={index} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 text-center">
                <div className="text-3xl mb-3">{cert.icon}</div>
                <h3 className="font-semibold text-gray-900 mb-2">{cert.name}</h3>
                <span className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${
                  cert.status === 'Certified' || cert.status === 'Compliant'
                    ? 'bg-green-100 text-green-800'
                    : 'bg-emerald-100 text-emerald-800'
                }`}>
                  {cert.status}
                </span>
              </div>
            ))}
          </div>

          <div className="bg-gradient-to-r from-amber-50 to-emerald-50 rounded-lg p-8 text-center">
            <h3 className="text-xl font-semibold text-gray-900 mb-4">
              Transparency Report
            </h3>
            <p className="text-gray-600 mb-6">
              We believe in transparency. View our latest security audit results and compliance reports.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button className="bg-gradient-to-r from-amber-600 to-emerald-600 text-white px-6 py-3 rounded-lg font-medium hover:from-amber-700 hover:to-emerald-700 transition-all duration-300">
                View Audit Reports
              </button>
              <button className="border border-gray-300 text-gray-700 px-6 py-3 rounded-lg font-medium hover:bg-gray-50 transition-colors">
                Download Certificates
              </button>
            </div>
          </div>
        </div>

        {/* Security Practices */}
        <div className="bg-gradient-to-r from-amber-600 to-emerald-600 text-white">
          <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
            <h2 className="text-3xl font-bold mb-12 text-center">
              Our Security Practices
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              
              {/* Development Security */}
              <div className="bg-gray-800 rounded-lg p-8">
                <h3 className="text-xl font-semibold mb-6 text-emerald-400">🔧 Development Security</h3>
                <ul className="space-y-3 text-gray-300">
                  <li className="flex items-start space-x-2">
                    <span className="text-emerald-400 mt-1">•</span>
                    <span>Secure coding practices and code reviews</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-emerald-400 mt-1">•</span>
                    <span>Automated security testing in CI/CD pipelines</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-emerald-400 mt-1">•</span>
                    <span>Regular dependency updates and vulnerability scanning</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-emerald-400 mt-1">•</span>
                    <span>Static and dynamic application security testing</span>
                  </li>
                </ul>
              </div>

              {/* Operational Security */}
              <div className="bg-gray-800 rounded-lg p-8">
                <h3 className="text-xl font-semibold mb-6 text-blue-400">🛡️ Operational Security</h3>
                <ul className="space-y-3 text-gray-300">
                  <li className="flex items-start space-x-2">
                    <span className="text-blue-400 mt-1">•</span>
                    <span>24/7 security operations center (SOC)</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-blue-400 mt-1">•</span>
                    <span>Automated threat detection and response</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-blue-400 mt-1">•</span>
                    <span>Regular penetration testing by third parties</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-blue-400 mt-1">•</span>
                    <span>Incident response and disaster recovery plans</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Contact Security */}
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16 text-center">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Have Security Questions? 🔒
          </h2>
          <p className="text-lg text-gray-600 mb-8">
            Our security team is here to address any concerns or questions you may have.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <ContactButton
              variant="button"
              prefillCategory="support"
              className="bg-gradient-to-r from-amber-600 to-emerald-600 text-white px-8 py-3 rounded-lg font-medium hover:from-amber-700 hover:to-emerald-700 transition-all duration-300"
            >
              Contact Security Team
            </ContactButton>
            <button className="border border-gray-300 text-gray-700 px-8 py-3 rounded-lg font-medium hover:bg-gray-50 transition-colors">
              Report Security Issue
            </button>
          </div>
          
          <div className="mt-8 text-sm text-gray-500">
            <p>🔐 security@dharmamind.com</p>
            <p className="mt-2">
              For urgent security matters, please encrypt your message using our PGP key.
            </p>
          </div>
        </div>

        {/* Footer */}
        <footer className="border-t border-gray-200 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <div className="text-center">
              <button 
                onClick={() => router.push('/')}
                className="flex justify-center mx-auto mb-4 hover:opacity-80 transition-opacity"
              >
                <Logo size="sm" showText={true} />
              </button>
              <p className="text-sm text-gray-600">
                © 2025 DharmaMind. All rights reserved.
              </p>
            </div>
          </div>
        </footer>
      </div>
    </>
  );
};

export default SecurityPage;
