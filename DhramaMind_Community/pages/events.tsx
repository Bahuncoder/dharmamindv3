/**
 * DharmaMind Community - Events Page
 */

import React from 'react';
import Link from 'next/link';
import { Layout } from '../components/Layout';

const EventsPage: React.FC = () => {
    const upcomingEvents = [
        { id: 1, title: 'Mindful Leadership Workshop', date: 'Dec 15, 2024', time: '10:00 AM PST', type: 'Workshop', attendees: 156 },
        { id: 2, title: 'AI & Ethics: A Thoughtful Discussion', date: 'Dec 18, 2024', time: '2:00 PM PST', type: 'Discussion', attendees: 89 },
        { id: 3, title: 'Year-End Reflection Circle', date: 'Dec 28, 2024', time: '11:00 AM PST', type: 'Community', attendees: 234 },
    ];

    const pastEvents = [
        { id: 4, title: 'Building Resilience in Teams', date: 'Nov 20, 2024', type: 'Workshop', attendees: 178 },
        { id: 5, title: 'Morning Meditation Sessions', date: 'Nov 15, 2024', type: 'Practice', attendees: 312 },
    ];

    return (
        <Layout title="Events" description="Join workshops, discussions, and community gatherings.">
            <main className="max-w-6xl mx-auto px-6 py-12">
                {/* Page Header */}
                <div className="mb-10">
                    <h1 className="text-4xl font-bold text-neutral-900 mb-3">Events</h1>
                    <p className="text-lg text-neutral-600">
                        Join workshops, discussions, and community gatherings.
                    </p>
                </div>

                {/* Upcoming Events */}
                <section className="mb-12">
                    <h2 className="text-2xl font-semibold text-neutral-900 mb-6">Upcoming Events</h2>
                    <div className="grid md:grid-cols-3 gap-6">
                        {upcomingEvents.map((event) => (
                            <Link
                                key={event.id}
                                href={`/events/${event.id}`}
                                className="bg-white p-6 rounded-xl border border-neutral-200 hover:border-gold-300 hover:shadow-md transition-all"
                            >
                                <span className="inline-block px-3 py-1 text-sm text-gold-700 bg-gold-100 rounded-full mb-4">
                                    {event.type}
                                </span>
                                <h3 className="text-xl font-semibold text-neutral-900 mb-3">{event.title}</h3>
                                <div className="space-y-2 text-base text-neutral-500">
                                    <div className="flex items-center gap-2">
                                        <svg className="w-5 h-5 text-gold-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                        </svg>
                                        <span>{event.date}</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <svg className="w-5 h-5 text-gold-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                        </svg>
                                        <span>{event.time}</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <svg className="w-5 h-5 text-gold-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                                        </svg>
                                        <span>{event.attendees} attending</span>
                                    </div>
                                </div>
                                <button className="mt-6 w-full px-4 py-2 bg-gold-600 text-white font-medium rounded-lg hover:bg-gold-700 transition-colors">
                                    RSVP
                                </button>
                            </Link>
                        ))}
                    </div>
                </section>

                {/* Past Events */}
                <section>
                    <h2 className="text-2xl font-semibold text-neutral-900 mb-6">Past Events</h2>
                    <div className="space-y-4">
                        {pastEvents.map((event) => (
                            <Link
                                key={event.id}
                                href={`/events/${event.id}`}
                                className="flex items-center justify-between bg-white p-6 rounded-xl border border-neutral-200 hover:border-gold-300 transition-all"
                            >
                                <div>
                                    <span className="inline-block px-3 py-1 text-sm text-neutral-500 bg-neutral-100 rounded-full mb-2">
                                        {event.type}
                                    </span>
                                    <h3 className="text-lg font-semibold text-neutral-900">{event.title}</h3>
                                    <p className="text-base text-neutral-500">{event.date}</p>
                                </div>
                                <div className="text-right">
                                    <p className="text-2xl font-bold text-neutral-900">{event.attendees}</p>
                                    <p className="text-sm text-neutral-500">attended</p>
                                </div>
                            </Link>
                        ))}
                    </div>
                </section>
            </main>
        </Layout>
    );
};

export default EventsPage;
