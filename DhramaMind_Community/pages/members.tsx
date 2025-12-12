/**
 * DharmaMind Community - Members Page
 */

import React from 'react';
import Link from 'next/link';
import { Layout } from '../components/Layout';

const MembersPage: React.FC = () => {
    const members = [
        { id: 1, name: 'Alex Martinez', role: 'Leadership Coach', posts: 156, joined: 'Jan 2024', avatar: 'A' },
        { id: 2, name: 'Sarah Kim', role: 'Tech Entrepreneur', posts: 89, joined: 'Feb 2024', avatar: 'S' },
        { id: 3, name: 'James Liu', role: 'Mindfulness Teacher', posts: 234, joined: 'Dec 2023', avatar: 'J' },
        { id: 4, name: 'Maya Patel', role: 'Wellness Consultant', posts: 67, joined: 'Mar 2024', avatar: 'M' },
        { id: 5, name: 'David Roberts', role: 'Executive Coach', posts: 112, joined: 'Jan 2024', avatar: 'D' },
        { id: 6, name: 'Lisa Thompson', role: 'HR Director', posts: 78, joined: 'Feb 2024', avatar: 'L' },
        { id: 7, name: 'Michael Chen', role: 'Software Engineer', posts: 45, joined: 'Apr 2024', avatar: 'M' },
        { id: 8, name: 'Emma Wilson', role: 'Life Coach', posts: 198, joined: 'Nov 2023', avatar: 'E' },
    ];

    return (
        <Layout title="Members" description="Meet the thoughtful professionals in our community.">
            <main className="max-w-6xl mx-auto px-6 py-12">
                {/* Page Header */}
                <div className="mb-10">
                    <h1 className="text-4xl font-bold text-neutral-900 mb-3">Members</h1>
                    <p className="text-lg text-neutral-600">
                        Connect with thoughtful professionals from around the world.
                    </p>
                </div>

                {/* Stats */}
                <div className="grid grid-cols-3 gap-6 mb-10">
                    <div className="bg-white p-6 rounded-xl border border-neutral-200 text-center">
                        <p className="text-3xl font-bold text-neutral-900">12,450</p>
                        <p className="text-base text-neutral-500">Total Members</p>
                    </div>
                    <div className="bg-white p-6 rounded-xl border border-neutral-200 text-center">
                        <p className="text-3xl font-bold text-neutral-900">89</p>
                        <p className="text-base text-neutral-500">Countries</p>
                    </div>
                    <div className="bg-white p-6 rounded-xl border border-neutral-200 text-center">
                        <p className="text-3xl font-bold text-neutral-900">1,234</p>
                        <p className="text-base text-neutral-500">New This Month</p>
                    </div>
                </div>

                {/* Members Grid */}
                <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
                    {members.map((member) => (
                        <Link
                            key={member.id}
                            href={`/members/${member.id}`}
                            className="bg-white p-6 rounded-xl border border-neutral-200 hover:border-gold-300 hover:shadow-md transition-all text-center"
                        >
                            <div className="w-16 h-16 bg-gold-100 rounded-full flex items-center justify-center mx-auto mb-4">
                                <span className="text-2xl font-bold text-gold-700">{member.avatar}</span>
                            </div>
                            <h3 className="text-lg font-semibold text-neutral-900 mb-1">{member.name}</h3>
                            <p className="text-base text-neutral-500 mb-3">{member.role}</p>
                            <div className="flex justify-center gap-4 text-sm text-neutral-400">
                                <span>{member.posts} posts</span>
                                <span>·</span>
                                <span>Joined {member.joined}</span>
                            </div>
                        </Link>
                    ))}
                </div>

                {/* Load More */}
                <div className="mt-10 text-center">
                    <button className="px-6 py-3 bg-white border border-neutral-200 text-neutral-700 text-base font-medium rounded-lg hover:border-gold-300 hover:text-gold-600 transition-colors">
                        Load more members
                    </button>
                </div>
            </main>
        </Layout>
    );
};

export default MembersPage;
