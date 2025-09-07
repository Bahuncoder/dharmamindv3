import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import Navigation from '../components/Navigation';
import CommunityDiscussions from '../components/CommunityDiscussions';

const CommunityFeedPage: React.FC = () => {
    const [activeSection, setActiveSection] = useState('discussions');

    const sections = [
        { id: 'discussions', name: 'Discussions', icon: '💬' },
        { id: 'events', name: 'Events', icon: '📅' },
        { id: 'resources', name: 'Resources', icon: '📚' },
        { id: 'members', name: 'Members', icon: '👥' }
    ];

    return (
        <>
            <Head>
                <title>Community Feed - DharmaMind Community</title>
                <meta name="description" content="Stay connected with the DharmaMind community. Join discussions, find events, and connect with fellow spiritual seekers." />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <link rel="icon" href="/logo.jpeg" />
            </Head>

            <div className="min-h-screen bg-secondary-bg">
                <Navigation />

                {/* Header Section */}
                <div className="bg-primary-gradient text-white py-12">
                    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                        <div className="text-center">
                            <h1 className="text-5xl font-black mb-4 tracking-tight">
                                🌸 Community Feed
                            </h1>
                            <p className="text-xl font-semibold opacity-90 max-w-2xl mx-auto">
                                Stay connected with your spiritual community. Share wisdom, ask questions, and grow together.
                            </p>
                        </div>
                    </div>
                </div>

                {/* Section Navigation */}
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
                    <div className="flex flex-wrap gap-2 justify-center mb-8">
                        {sections.map(section => (
                            <button
                                key={section.id}
                                onClick={() => setActiveSection(section.id)}
                                className={`px-6 py-3 rounded-lg font-bold transition-all duration-200 flex items-center gap-2 ${activeSection === section.id
                                        ? 'bg-primary-gradient text-white shadow-lg'
                                        : 'bg-primary text-secondary hover:bg-bg-tertiary'
                                    }`}
                            >
                                <span>{section.icon}</span>
                                <span>{section.name}</span>
                            </button>
                        ))}
                    </div>

                    {/* Content Section */}
                    <div className="min-h-screen">
                        {activeSection === 'discussions' && <CommunityDiscussions />}

                        {activeSection === 'events' && (
                            <div className="text-center py-20">
                                <span className="text-6xl mb-4 block">📅</span>
                                <h2 className="text-3xl font-black text-primary mb-4">Community Events</h2>
                                <p className="text-xl text-secondary mb-8 max-w-2xl mx-auto">
                                    Join our meditation sessions, dharma talks, and community gatherings.
                                </p>
                                <div className="bg-primary rounded-xl p-8 max-w-2xl mx-auto">
                                    <div className="grid gap-4">
                                        <div className="bg-primary-gradient-light p-6 rounded-lg">
                                            <h3 className="text-xl font-bold text-primary mb-2">🧘‍♀️ Weekly Group Meditation</h3>
                                            <p className="text-secondary mb-2">Every Sunday at 10:00 AM PST</p>
                                            <p className="text-sm text-muted">Join our virtual meditation session for inner peace and community connection.</p>
                                        </div>

                                        <div className="bg-primary-gradient-light p-6 rounded-lg">
                                            <h3 className="text-xl font-bold text-primary mb-2">☸️ Dharma Discussion Circle</h3>
                                            <p className="text-secondary mb-2">Every Wednesday at 7:00 PM PST</p>
                                            <p className="text-sm text-muted">Explore Buddhist teachings and their applications in modern life.</p>
                                        </div>

                                        <div className="bg-primary-gradient-light p-6 rounded-lg">
                                            <h3 className="text-xl font-bold text-primary mb-2">🎋 Monthly Community Retreat</h3>
                                            <p className="text-secondary mb-2">First Saturday of each month</p>
                                            <p className="text-sm text-muted">Deep practice sessions with guided meditation and wisdom teachings.</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {activeSection === 'resources' && (
                            <div className="text-center py-20">
                                <span className="text-6xl mb-4 block">📚</span>
                                <h2 className="text-3xl font-black text-primary mb-4">Learning Resources</h2>
                                <p className="text-xl text-secondary mb-8 max-w-2xl mx-auto">
                                    Explore our curated collection of dharma teachings, guided meditations, and wisdom resources.
                                </p>
                                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
                                    <div className="bg-primary p-6 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300">
                                        <span className="text-3xl mb-4 block">📖</span>
                                        <h3 className="text-lg font-bold text-primary mb-2">Essential Texts</h3>
                                        <p className="text-secondary text-sm">Core Buddhist texts and modern interpretations</p>
                                    </div>

                                    <div className="bg-primary p-6 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300">
                                        <span className="text-3xl mb-4 block">🎧</span>
                                        <h3 className="text-lg font-bold text-primary mb-2">Guided Meditations</h3>
                                        <p className="text-secondary text-sm">Audio practices for daily meditation</p>
                                    </div>

                                    <div className="bg-primary p-6 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300">
                                        <span className="text-3xl mb-4 block">🎥</span>
                                        <h3 className="text-lg font-bold text-primary mb-2">Dharma Talks</h3>
                                        <p className="text-secondary text-sm">Video teachings from experienced practitioners</p>
                                    </div>
                                </div>
                            </div>
                        )}

                        {activeSection === 'members' && (
                            <div className="text-center py-20">
                                <span className="text-6xl mb-4 block">👥</span>
                                <h2 className="text-3xl font-black text-primary mb-4">Community Members</h2>
                                <p className="text-xl text-secondary mb-8 max-w-2xl mx-auto">
                                    Connect with fellow practitioners on the dharma path.
                                </p>
                                <div className="bg-primary rounded-xl p-8 max-w-4xl mx-auto">
                                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                                        {/* Featured Members */}
                                        <div className="bg-primary-gradient-light p-6 rounded-lg text-center">
                                            <div className="w-16 h-16 bg-primary-gradient rounded-full flex items-center justify-center text-white text-xl font-bold mb-4 mx-auto">
                                                👩‍💼
                                            </div>
                                            <h3 className="font-bold text-primary mb-1">Sarah Chen</h3>
                                            <p className="text-sm text-secondary mb-2">Community Guide</p>
                                            <div className="flex flex-wrap justify-center gap-1">
                                                <span className="bg-neutral-200 px-2 py-1 rounded text-xs">Meditation</span>
                                                <span className="bg-neutral-200 px-2 py-1 rounded text-xs">Mindfulness</span>
                                            </div>
                                        </div>

                                        <div className="bg-primary-gradient-light p-6 rounded-lg text-center">
                                            <div className="w-16 h-16 bg-primary-gradient rounded-full flex items-center justify-center text-white text-xl font-bold mb-4 mx-auto">
                                                👨‍🎓
                                            </div>
                                            <h3 className="font-bold text-primary mb-1">Michael Torres</h3>
                                            <p className="text-sm text-secondary mb-2">Dharma Scholar</p>
                                            <div className="flex flex-wrap justify-center gap-1">
                                                <span className="bg-neutral-200 px-2 py-1 rounded text-xs">Philosophy</span>
                                                <span className="bg-neutral-200 px-2 py-1 rounded text-xs">Teaching</span>
                                            </div>
                                        </div>

                                        <div className="bg-primary-gradient-light p-6 rounded-lg text-center">
                                            <div className="w-16 h-16 bg-primary-gradient rounded-full flex items-center justify-center text-white text-xl font-bold mb-4 mx-auto">
                                                👩‍💻
                                            </div>
                                            <h3 className="font-bold text-primary mb-1">Emma Watson</h3>
                                            <p className="text-sm text-secondary mb-2">Mindful Professional</p>
                                            <div className="flex flex-wrap justify-center gap-1">
                                                <span className="bg-neutral-200 px-2 py-1 rounded text-xs">Work-Life</span>
                                                <span className="bg-neutral-200 px-2 py-1 rounded text-xs">Balance</span>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="mt-8 pt-6 border-t border-border-light">
                                        <p className="text-secondary font-semibold mb-4">
                                            Join our growing community of 2,847+ mindful practitioners
                                        </p>
                                        <a
                                            href="https://dharmamind.ai"
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="btn-primary px-6 py-3 rounded-lg font-bold hover:opacity-90 transition-opacity"
                                        >
                                            Connect with DharmaMind
                                        </a>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </>
    );
};

export default CommunityFeedPage;
