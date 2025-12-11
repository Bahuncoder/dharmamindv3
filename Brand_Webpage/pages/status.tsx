import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import Logo from '../components/Logo';
import { ContactButton } from '../components/CentralizedSupport';

interface ServiceStatus {
  name: string;
  status: 'operational' | 'degraded' | 'down' | 'maintenance';
  uptime: string;
  responseTime: string;
  description: string;
}

interface Incident {
  id: string;
  title: string;
  status: 'investigating' | 'identified' | 'monitoring' | 'resolved';
  severity: 'low' | 'medium' | 'high' | 'critical';
  createdAt: string;
  updatedAt: string;
  description: string;
  updates: Array<{
    timestamp: string;
    message: string;
    status: string;
  }>;
}

const StatusPage: React.FC = () => {
  const router = useRouter();
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [overallStatus, setOverallStatus] = useState<'operational' | 'degraded' | 'down'>('operational');

  useEffect(() => {
    // Simulate fetching service status
    const mockServices: ServiceStatus[] = [
      {
        name: 'DharmaMind API',
        status: 'operational',
        uptime: '99.9%',
        responseTime: '120ms',
        description: 'Core AI chat and wisdom processing'
      },
      {
        name: 'Authentication Service',
        status: 'operational',
        uptime: '99.8%',
        responseTime: '85ms',
        description: 'User login and registration'
      },
      {
        name: 'Chat Interface',
        status: 'operational',
        uptime: '100%',
        responseTime: '45ms',
        description: 'Web application frontend'
      },
      {
        name: 'Payment Processing',
        status: 'operational',
        uptime: '99.9%',
        responseTime: '210ms',
        description: 'Subscription and billing'
      },
      {
        name: 'Mobile App',
        status: 'operational',
        uptime: '99.7%',
        responseTime: '90ms',
        description: 'iOS and Android applications'
      },
      {
        name: 'Wisdom Database',
        status: 'operational',
        uptime: '100%',
        responseTime: '15ms',
        description: 'Dharmic knowledge and wisdom modules'
      }
    ];

    const mockIncidents: Incident[] = [
      {
        id: '1',
        title: 'Scheduled Maintenance - Database Optimization',
        status: 'resolved',
        severity: 'low',
        createdAt: '2025-01-28T10:00:00Z',
        updatedAt: '2025-01-28T12:30:00Z',
        description: 'Routine database optimization to improve response times.',
        updates: [
          {
            timestamp: '2025-01-28T12:30:00Z',
            message: 'Maintenance completed successfully. All services restored.',
            status: 'resolved'
          },
          {
            timestamp: '2025-01-28T10:00:00Z',
            message: 'Scheduled maintenance began. Minor impact expected.',
            status: 'monitoring'
          }
        ]
      }
    ];

    setServices(mockServices);
    setIncidents(mockIncidents);

    // Calculate overall status
    const hasDown = mockServices.some(s => s.status === 'down');
    const hasDegraded = mockServices.some(s => s.status === 'degraded');
    
    if (hasDown) {
      setOverallStatus('down');
    } else if (hasDegraded) {
      setOverallStatus('degraded');
    } else {
      setOverallStatus('operational');
    }
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'operational':
<<<<<<< HEAD
        return 'text-gold-600 bg-gold-100';
      case 'degraded':
        return 'text-gold-600 bg-gold-100';
      case 'down':
        return 'text-red-600 bg-red-100';
      case 'maintenance':
        return 'text-gold-600 bg-gold-100';
      default:
        return 'text-neutral-600 bg-neutral-100';
=======
        return 'text-green-600 bg-green-100';
      case 'degraded':
        return 'text-emerald-600 bg-emerald-100';
      case 'down':
        return 'text-red-600 bg-red-100';
      case 'maintenance':
        return 'text-blue-600 bg-blue-100';
      default:
        return 'text-gray-600 bg-gray-100';
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'operational':
        return '✅';
      case 'degraded':
        return '⚠️';
      case 'down':
        return '❌';
      case 'maintenance':
        return '🔧';
      default:
        return '❓';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'low':
<<<<<<< HEAD
        return 'text-gold-600 bg-gold-100';
      case 'medium':
        return 'text-gold-600 bg-gold-100';
=======
        return 'text-green-600 bg-green-100';
      case 'medium':
        return 'text-emerald-600 bg-emerald-100';
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
      case 'high':
        return 'text-red-600 bg-red-100';
      case 'critical':
        return 'text-red-600 bg-red-100';
      default:
<<<<<<< HEAD
        return 'text-neutral-600 bg-neutral-100';
=======
        return 'text-gray-600 bg-gray-100';
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <>
      <Head>
        <title>System Status - DharmaMind</title>
        <meta name="description" content="Real-time status of DharmaMind services and systems" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

<<<<<<< HEAD
      <div className="min-h-screen bg-neutral-100">
        {/* Header */}
        <header className="border-b border-neutral-300 bg-neutral-100">
=======
      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="border-b border-gray-200 bg-white">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
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
<<<<<<< HEAD
                  className="text-neutral-600 hover:text-gold-600 text-sm font-medium"
=======
                  className="text-secondary hover:text-primary text-sm font-medium"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                >
                  Home
                </button>
                <button 
                  onClick={() => router.push('/help')}
<<<<<<< HEAD
                  className="text-neutral-600 hover:text-gold-600 text-sm font-medium"
=======
                  className="text-secondary hover:text-primary text-sm font-medium"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                >
                  Help
                </button>
                <ContactButton 
                  variant="link"
                  prefillCategory="support"
<<<<<<< HEAD
                  className="text-neutral-600 hover:text-gold-600 text-sm font-medium"
=======
                  className="text-secondary hover:text-primary text-sm font-medium"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                >
                  Contact
                </ContactButton>
              </nav>
            </div>
          </div>
        </header>

        {/* Overall Status */}
        <div className={`py-8 ${
          overallStatus === 'operational' 
            ? 'bg-green-50 border-green-200' 
            : overallStatus === 'degraded'
<<<<<<< HEAD
            ? 'bg-gold-50 border-gold-200'
=======
            ? 'bg-emerald-50 border-emerald-200'
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
            : 'bg-red-50 border-red-200'
        } border-b`}>
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <div className="flex items-center justify-center space-x-3 mb-4">
              <span className="text-3xl">
                {overallStatus === 'operational' ? '✅' : overallStatus === 'degraded' ? '⚠️' : '❌'}
              </span>
<<<<<<< HEAD
              <h1 className="text-3xl font-bold text-neutral-900">
=======
              <h1 className="text-3xl font-bold text-gray-900">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                {overallStatus === 'operational' 
                  ? 'All Systems Operational' 
                  : overallStatus === 'degraded'
                  ? 'Some Systems Degraded'
                  : 'System Issues Detected'
                }
              </h1>
            </div>
<<<<<<< HEAD
            <p className="text-lg text-neutral-600">
=======
            <p className="text-lg text-gray-600">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
              {overallStatus === 'operational' 
                ? 'DharmaMind is running smoothly. All services are functioning normally.' 
                : overallStatus === 'degraded'
                ? 'Some services are experiencing minor issues. We are working to resolve them.'
                : 'We are experiencing service disruptions and working to restore normal operation.'
              }
            </p>
<<<<<<< HEAD
            <div className="mt-4 text-sm text-neutral-600">
=======
            <div className="mt-4 text-sm text-gray-500">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
              Last updated: {new Date().toLocaleString()}
            </div>
          </div>
        </div>

        {/* Services Status */}
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
<<<<<<< HEAD
          <h2 className="text-2xl font-bold text-neutral-900 mb-8">Service Status</h2>
          
          <div className="space-y-4">
            {services.map((service, index) => (
              <div key={index} className="bg-neutral-100 rounded-lg shadow-sm border border-neutral-300 p-6">
=======
          <h2 className="text-2xl font-bold text-gray-900 mb-8">Service Status</h2>
          
          <div className="space-y-4">
            {services.map((service, index) => (
              <div key={index} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <span className="text-2xl">{getStatusIcon(service.status)}</span>
                    <div>
<<<<<<< HEAD
                      <h3 className="font-semibold text-neutral-900">{service.name}</h3>
                      <p className="text-sm text-neutral-600">{service.description}</p>
=======
                      <h3 className="font-semibold text-gray-900">{service.name}</h3>
                      <p className="text-sm text-gray-600">{service.description}</p>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-6">
                    <div className="text-center">
<<<<<<< HEAD
                      <div className="text-sm font-medium text-neutral-900">{service.uptime}</div>
                      <div className="text-xs text-neutral-600">Uptime</div>
                    </div>
                    <div className="text-center">
                      <div className="text-sm font-medium text-neutral-900">{service.responseTime}</div>
                      <div className="text-xs text-neutral-600">Response</div>
=======
                      <div className="text-sm font-medium text-gray-900">{service.uptime}</div>
                      <div className="text-xs text-gray-500">Uptime</div>
                    </div>
                    <div className="text-center">
                      <div className="text-sm font-medium text-gray-900">{service.responseTime}</div>
                      <div className="text-xs text-gray-500">Response</div>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    </div>
                    <span className={`px-3 py-1 rounded-full text-sm font-medium capitalize ${getStatusColor(service.status)}`}>
                      {service.status}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Incidents */}
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 pb-12">
<<<<<<< HEAD
          <h2 className="text-2xl font-bold text-neutral-900 mb-8">Recent Incidents</h2>
          
          {incidents.length === 0 ? (
            <div className="bg-neutral-100 rounded-lg shadow-sm border border-neutral-300 p-8 text-center">
              <div className="w-16 h-16 mx-auto mb-4 bg-gold-100 rounded-full flex items-center justify-center">
                <span className="text-2xl">✅</span>
              </div>
              <h3 className="text-lg font-medium text-neutral-900 mb-2">No Recent Incidents</h3>
              <p className="text-neutral-600">All systems have been running smoothly with no reported incidents in the last 30 days.</p>
=======
          <h2 className="text-2xl font-bold text-gray-900 mb-8">Recent Incidents</h2>
          
          {incidents.length === 0 ? (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center">
              <div className="w-16 h-16 mx-auto mb-4 bg-green-100 rounded-full flex items-center justify-center">
                <span className="text-2xl">✅</span>
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Recent Incidents</h3>
              <p className="text-gray-600">All systems have been running smoothly with no reported incidents in the last 30 days.</p>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
            </div>
          ) : (
            <div className="space-y-6">
              {incidents.map((incident) => (
<<<<<<< HEAD
                <div key={incident.id} className="bg-neutral-100 rounded-lg shadow-sm border border-neutral-300 p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <h3 className="font-semibold text-neutral-900">{incident.title}</h3>
=======
                <div key={incident.id} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <h3 className="font-semibold text-gray-900">{incident.title}</h3>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                        <span className={`px-2 py-1 rounded-full text-xs font-medium capitalize ${getSeverityColor(incident.severity)}`}>
                          {incident.severity}
                        </span>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium capitalize ${getStatusColor(incident.status)}`}>
                          {incident.status}
                        </span>
                      </div>
<<<<<<< HEAD
                      <p className="text-neutral-600 mb-3">{incident.description}</p>
                      <div className="text-sm text-neutral-600">
=======
                      <p className="text-gray-600 mb-3">{incident.description}</p>
                      <div className="text-sm text-gray-500">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                        Created: {formatDate(incident.createdAt)} • 
                        Updated: {formatDate(incident.updatedAt)}
                      </div>
                    </div>
                  </div>
                  
                  {/* Incident Updates */}
<<<<<<< HEAD
                  <div className="border-t border-neutral-300 pt-4">
                    <h4 className="font-medium text-neutral-900 mb-3">Updates</h4>
                    <div className="space-y-3">
                      {incident.updates.map((update, updateIndex) => (
                        <div key={updateIndex} className="flex items-start space-x-3">
                          <div className="w-2 h-2 bg-neutral-100 rounded-full mt-2"></div>
                          <div className="flex-1">
                            <div className="text-sm text-neutral-900">{update.message}</div>
                            <div className="text-xs text-neutral-600">{formatDate(update.timestamp)}</div>
=======
                  <div className="border-t border-gray-200 pt-4">
                    <h4 className="font-medium text-gray-900 mb-3">Updates</h4>
                    <div className="space-y-3">
                      {incident.updates.map((update, updateIndex) => (
                        <div key={updateIndex} className="flex items-start space-x-3">
                          <div className="w-2 h-2 bg-gray-400 rounded-full mt-2"></div>
                          <div className="flex-1">
                            <div className="text-sm text-gray-900">{update.message}</div>
                            <div className="text-xs text-gray-500">{formatDate(update.timestamp)}</div>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Historical Uptime */}
<<<<<<< HEAD
        <div className="bg-neutral-100 border-t border-neutral-300">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
            <h2 className="text-2xl font-bold text-neutral-900 mb-8">90-Day Uptime History</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="text-3xl font-bold text-gold-600 mb-2">99.9%</div>
                <div className="text-neutral-600">Overall Uptime</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-gold-600 mb-2">1.2s</div>
                <div className="text-neutral-600">Avg Response Time</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-purple-600 mb-2">0</div>
                <div className="text-neutral-600">Critical Incidents</div>
              </div>
            </div>

            <div className="mt-8 p-6 bg-neutral-100 rounded-lg">
              <div className="flex items-center justify-between mb-4">
                <span className="text-sm font-medium text-neutral-900">Past 90 days</span>
                <div className="flex items-center space-x-4 text-xs text-neutral-600">
=======
        <div className="bg-white border-t border-gray-200">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
            <h2 className="text-2xl font-bold text-gray-900 mb-8">90-Day Uptime History</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="text-3xl font-bold text-green-600 mb-2">99.9%</div>
                <div className="text-gray-600">Overall Uptime</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600 mb-2">1.2s</div>
                <div className="text-gray-600">Avg Response Time</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-purple-600 mb-2">0</div>
                <div className="text-gray-600">Critical Incidents</div>
              </div>
            </div>

            <div className="mt-8 p-6 bg-gray-50 rounded-lg">
              <div className="flex items-center justify-between mb-4">
                <span className="text-sm font-medium text-gray-700">Past 90 days</span>
                <div className="flex items-center space-x-4 text-xs text-gray-500">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                  <div className="flex items-center space-x-1">
                    <div className="w-3 h-3 bg-green-500 rounded"></div>
                    <span>Operational</span>
                  </div>
                  <div className="flex items-center space-x-1">
<<<<<<< HEAD
                    <div className="w-3 h-3 bg-gold-600 rounded"></div>
=======
                    <div className="w-3 h-3 bg-emerald-500 rounded"></div>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    <span>Degraded</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <div className="w-3 h-3 bg-red-500 rounded"></div>
                    <span>Down</span>
                  </div>
                </div>
              </div>
              
              {/* Simplified uptime visualization */}
              <div className="grid grid-cols-90 gap-1">
                {Array.from({length: 90}, (_, i) => (
                  <div 
                    key={i} 
                    className="h-8 bg-green-500 rounded-sm" 
                    title={`Day ${90-i}: Operational`}
                  ></div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
<<<<<<< HEAD
        <footer className="border-t border-neutral-300 bg-neutral-100">
=======
        <footer className="border-t border-gray-200 bg-white">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <div className="text-center">
              <button 
                onClick={() => router.push('/')}
                className="flex justify-center mx-auto mb-4 hover:opacity-80 transition-opacity"
              >
                <Logo size="sm" showText={true} />
              </button>
<<<<<<< HEAD
              <p className="text-sm text-neutral-600">
=======
              <p className="text-sm text-gray-600">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                © 2025 DharmaMind. All rights reserved.
              </p>
            </div>
          </div>
        </footer>
      </div>
    </>
  );
};

export default StatusPage;
