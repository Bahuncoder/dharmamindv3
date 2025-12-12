/**
 * DharmaMind Community - Event Detail Page
 */

import React, { useState } from 'react';
import { useRouter } from 'next/router';
import { Layout } from '../../components/Layout';
import Link from 'next/link';

// Mock data - would come from API
const mockEvent = {
  id: 1,
  title: 'Mindful Tech: Balancing Innovation & Wisdom',
  description: `Join us for an inspiring panel discussion exploring how technology leaders can integrate mindfulness practices into their work and organizations.

Our expert panelists will share practical strategies for:
- Maintaining focus and clarity in fast-paced tech environments
- Building cultures of psychological safety and authentic communication
- Using AI tools mindfully and ethically
- Creating sustainable work practices that support wellbeing

Whether you're a tech professional, entrepreneur, or simply curious about the intersection of technology and wisdom traditions, this event offers valuable insights for navigating our digital age with intention and presence.`,
  type: 'Virtual',
  date: 'March 15, 2024',
  time: '10:00 AM - 12:00 PM PST',
  location: 'Zoom (link sent upon registration)',
  image: null,
  category: 'Workshop',
  host: {
    name: 'DharmaMind Community',
    avatar: 'D',
  },
  speakers: [
    { name: 'Dr. Sarah Chen', role: 'AI Ethics Researcher', avatar: 'S' },
    { name: 'Marcus Williams', role: 'Mindfulness Coach', avatar: 'M' },
    { name: 'Priya Sharma', role: 'Tech Entrepreneur', avatar: 'P' },
  ],
  attendees: 156,
  maxAttendees: 200,
  isRegistered: false,
  price: 'Free',
};

const EventDetailPage: React.FC = () => {
  const router = useRouter();
  const { id } = router.query;
  const [isRegistered, setIsRegistered] = useState(mockEvent.isRegistered);
  const [isRegistering, setIsRegistering] = useState(false);
  
  const event = mockEvent; // In production, fetch by id

  const handleRSVP = async () => {
    setIsRegistering(true);
    // In production, this would call an API
    await new Promise(resolve => setTimeout(resolve, 1000));
    setIsRegistered(true);
    setIsRegistering(false);
  };

  const spotsRemaining = event.maxAttendees - event.attendees;
  const spotsPercentage = (event.attendees / event.maxAttendees) * 100;

  return (
    <Layout title={event.title} description={`Event: ${event.title}`}>
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Breadcrumb */}
        <nav className="mb-8">
          <Link href="/events" className="text-gold-600 hover:text-gold-700 transition-colors">
            ← Back to Events
          </Link>
        </nav>

        <div className="grid md:grid-cols-3 gap-8">
          {/* Main Content */}
          <div className="md:col-span-2">
            {/* Header */}
            <div className="mb-6">
              <div className="flex items-center gap-2 mb-3">
                <span className="px-3 py-1 text-sm text-gold-700 bg-gold-100 rounded-full">
                  {event.category}
                </span>
                <span className="px-3 py-1 text-sm text-neutral-600 bg-neutral-100 rounded-full">
                  {event.type}
                </span>
              </div>
              <h1 className="text-3xl font-bold text-neutral-900">{event.title}</h1>
            </div>

            {/* Event Image Placeholder */}
            <div className="bg-gradient-to-br from-gold-100 to-gold-200 rounded-2xl h-64 flex items-center justify-center mb-8">
              <div className="text-center">
                <div className="text-6xl mb-2">🧘</div>
                <p className="text-gold-700 font-medium">{event.category}</p>
              </div>
            </div>

            {/* Description */}
            <div className="bg-white rounded-2xl border border-neutral-200 p-8 mb-8">
              <h2 className="text-xl font-semibold text-neutral-900 mb-4">About This Event</h2>
              <div className="prose prose-neutral max-w-none">
                {event.description.split('\n\n').map((paragraph, i) => (
                  <p key={i} className="text-neutral-700 leading-relaxed mb-4 whitespace-pre-line">
                    {paragraph}
                  </p>
                ))}
              </div>
            </div>

            {/* Speakers */}
            <div className="bg-white rounded-2xl border border-neutral-200 p-8">
              <h2 className="text-xl font-semibold text-neutral-900 mb-6">Speakers</h2>
              <div className="space-y-4">
                {event.speakers.map((speaker, i) => (
                  <div key={i} className="flex items-center gap-4">
                    <div className="w-12 h-12 bg-gold-100 rounded-full flex items-center justify-center">
                      <span className="text-xl font-bold text-gold-700">{speaker.avatar}</span>
                    </div>
                    <div>
                      <p className="font-semibold text-neutral-900">{speaker.name}</p>
                      <p className="text-sm text-neutral-500">{speaker.role}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Sidebar */}
          <div className="md:col-span-1">
            <div className="sticky top-8 space-y-6">
              {/* Registration Card */}
              <div className="bg-white rounded-2xl border border-neutral-200 p-6">
                <div className="text-center mb-6">
                  <p className="text-3xl font-bold text-neutral-900">{event.price}</p>
                </div>

                {isRegistered ? (
                  <div className="bg-green-50 border border-green-200 rounded-lg p-4 text-center mb-4">
                    <svg className="w-8 h-8 text-green-600 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    <p className="text-green-800 font-medium">You're registered!</p>
                    <p className="text-green-600 text-sm">Check your email for details</p>
                  </div>
                ) : (
                  <button
                    onClick={handleRSVP}
                    disabled={isRegistering || spotsRemaining === 0}
                    className="w-full px-6 py-3 bg-gold-600 text-white font-medium rounded-lg hover:bg-gold-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed mb-4"
                  >
                    {isRegistering ? 'Registering...' : spotsRemaining === 0 ? 'Event Full' : 'Register Now'}
                  </button>
                )}

                {/* Spots Progress */}
                <div className="mb-4">
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-neutral-500">{event.attendees} registered</span>
                    <span className="text-neutral-500">{spotsRemaining} spots left</span>
                  </div>
                  <div className="h-2 bg-neutral-100 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gold-500 rounded-full transition-all"
                      style={{ width: `${spotsPercentage}%` }}
                    />
                  </div>
                </div>

                <button className="w-full px-4 py-2 text-neutral-600 hover:text-neutral-900 transition-colors text-sm">
                  Share Event
                </button>
              </div>

              {/* Event Details Card */}
              <div className="bg-white rounded-2xl border border-neutral-200 p-6">
                <h3 className="font-semibold text-neutral-900 mb-4">Event Details</h3>
                <div className="space-y-4">
                  <div className="flex items-start gap-3">
                    <svg className="w-5 h-5 text-gold-600 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <div>
                      <p className="font-medium text-neutral-900">{event.date}</p>
                      <p className="text-sm text-neutral-500">{event.time}</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <svg className="w-5 h-5 text-gold-600 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                    <div>
                      <p className="font-medium text-neutral-900">{event.type}</p>
                      <p className="text-sm text-neutral-500">{event.location}</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Host Card */}
              <div className="bg-white rounded-2xl border border-neutral-200 p-6">
                <h3 className="font-semibold text-neutral-900 mb-4">Hosted by</h3>
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-gold-100 rounded-full flex items-center justify-center">
                    <span className="text-lg font-bold text-gold-700">{event.host.avatar}</span>
                  </div>
                  <p className="font-medium text-neutral-900">{event.host.name}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default EventDetailPage;
