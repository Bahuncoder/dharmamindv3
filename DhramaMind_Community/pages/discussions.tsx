/**
 * DharmaMind Community - Discussions Page
 */

import React from 'react';
import Link from 'next/link';
import { Layout } from '../components/Layout';

const DiscussionsPage: React.FC = () => {
    const discussions = [
        { id: 1, title: 'Best practices for mindful leadership', author: 'Alex M.', replies: 24, category: 'Leadership', date: '2 hours ago' },
        { id: 2, title: 'How AI can support better decision-making', author: 'Sarah K.', replies: 18, category: 'Technology', date: '5 hours ago' },
        { id: 3, title: 'Building resilience in uncertain times', author: 'James L.', replies: 31, category: 'Growth', date: '1 day ago' },
        { id: 4, title: 'Integrating wisdom principles in daily work', author: 'Maya P.', replies: 12, category: 'Practice', date: '2 days ago' },
        { id: 5, title: 'The role of meditation in productivity', author: 'David R.', replies: 45, category: 'Wellness', date: '3 days ago' },
        { id: 6, title: 'Creating meaningful connections remotely', author: 'Lisa T.', replies: 28, category: 'Leadership', date: '4 days ago' },
    ];

    const categories = ['All', 'Leadership', 'Technology', 'Growth', 'Practice', 'Wellness'];

    return (
        <Layout title="Discussions" description="Join thoughtful discussions on leadership, growth, and meaningful topics.">
            <main className="max-w-6xl mx-auto px-6 py-12">
                {/* Page Header */}
                <div className="mb-10">
                    <h1 className="text-4xl font-bold text-neutral-900 mb-3">Discussions</h1>
                    <p className="text-lg text-neutral-600">
                        Explore conversations on leadership, growth, and meaningful topics.
                    </p>
                </div>

                {/* Categories */}
                <div className="flex flex-wrap gap-3 mb-8">
                    {categories.map((cat) => (
                        <button
                            key={cat}
                            className={`px-4 py-2 rounded-lg text-base font-medium transition-colors ${cat === 'All'
                                ? 'bg-gold-600 text-white'
                                : 'bg-white border border-neutral-200 text-neutral-600 hover:border-gold-300 hover:text-gold-600'
                                }`}
                        >
                            {cat}
                        </button>
                    ))}
                </div>

                {/* Discussions List */}
                <div className="space-y-4">
                    {discussions.map((discussion) => (
                        <Link
                            key={discussion.id}
                            href={`/discussions/${discussion.id}`}
                            className="block bg-white p-6 rounded-xl border border-neutral-200 hover:border-gold-300 hover:shadow-md transition-all"
                        >
                            <div className="flex items-start justify-between gap-6">
                                <div className="flex-1">
                                    <span className="inline-block px-3 py-1 text-sm text-gold-700 bg-gold-100 rounded-full mb-3">
                                        {discussion.category}
                                    </span>
                                    <h3 className="text-xl font-semibold text-neutral-900 mb-2">{discussion.title}</h3>
                                    <p className="text-base text-neutral-500">
                                        by {discussion.author} · {discussion.date}
                                    </p>
                                </div>
                                <div className="text-right flex-shrink-0">
                                    <p className="text-2xl font-bold text-neutral-900">{discussion.replies}</p>
                                    <p className="text-sm text-neutral-500">replies</p>
                                </div>
                            </div>
                        </Link>
                    ))}
                </div>

                {/* Start Discussion CTA */}
                <div className="mt-10 text-center">
                    <Link
                        href="/discussions/new"
                        className="inline-flex items-center gap-2 px-6 py-3 bg-gold-600 text-white text-lg font-medium rounded-lg hover:bg-gold-700 transition-colors"
                    >
                        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
                        </svg>
                        Start a new discussion
                    </Link>
                </div>
            </main>
        </Layout>
    );
};

export default DiscussionsPage;
