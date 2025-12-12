/**
 * DharmaMind Community - Discussion Detail Page
 */

import React, { useState } from 'react';
import { useRouter } from 'next/router';
import { Layout } from '../../components/Layout';
import Link from 'next/link';

// Mock data - would come from API
const mockDiscussion = {
    id: 1,
    title: 'Best practices for mindful leadership',
    content: `Leadership in today's fast-paced world requires more than just strategic thinking—it demands mindfulness, emotional intelligence, and a deep understanding of human dynamics.

I've been exploring how ancient wisdom traditions can inform modern leadership practices. Here are some principles I've found particularly valuable:

**1. Present-moment awareness**
Leaders who practice mindfulness are better able to respond thoughtfully rather than react impulsively. This leads to better decision-making and more authentic connections with team members.

**2. Compassionate communication**
Drawing from Buddhist principles of right speech, effective leaders communicate with honesty, kindness, and a genuine concern for others' wellbeing.

**3. Non-attachment to outcomes**
While goals are important, attachment to specific outcomes can create stress and rigid thinking. Mindful leaders hold goals lightly while remaining committed to the process.

What practices have you found helpful in your leadership journey? I'd love to hear your experiences.`,
    author: {
        name: 'Alex Martinez',
        role: 'Leadership Coach',
        avatar: 'A',
        posts: 156,
    },
    category: 'Leadership',
    date: '2 hours ago',
    replies: [
        {
            id: 1,
            author: { name: 'Sarah Kim', avatar: 'S', role: 'Tech Entrepreneur' },
            content: 'This resonates deeply. I\'ve found that starting each day with a brief meditation has transformed how I approach challenges. The concept of non-attachment has been particularly powerful—it helps me stay creative when things don\'t go as planned.',
            date: '1 hour ago',
            likes: 12,
        },
        {
            id: 2,
            author: { name: 'James Liu', avatar: 'J', role: 'Mindfulness Teacher' },
            content: 'Beautiful synthesis of wisdom traditions and modern leadership! I would add that regular self-reflection practices are essential. Journaling or contemplation helps leaders understand their own patterns and blind spots.',
            date: '45 minutes ago',
            likes: 8,
        },
        {
            id: 3,
            author: { name: 'Maya Patel', avatar: 'M', role: 'Wellness Consultant' },
            content: 'The point about compassionate communication is crucial. I\'ve seen teams transform when leaders prioritize psychological safety and genuine care. It creates space for innovation and authentic collaboration.',
            date: '30 minutes ago',
            likes: 15,
        },
    ],
};

const DiscussionDetailPage: React.FC = () => {
    const router = useRouter();
    const { id } = router.query;
    const [replyContent, setReplyContent] = useState('');

    const discussion = mockDiscussion; // In production, fetch by id

    const handleSubmitReply = (e: React.FormEvent) => {
        e.preventDefault();
        // Handle reply submission
        console.log('Reply:', replyContent);
        setReplyContent('');
    };

    return (
        <Layout title={discussion.title} description={`Discussion: ${discussion.title}`}>
            <div className="max-w-4xl mx-auto px-6 py-12">
                {/* Breadcrumb */}
                <nav className="mb-8">
                    <Link href="/discussions" className="text-gold-600 hover:text-gold-700 transition-colors">
                        ← Back to Discussions
                    </Link>
                </nav>

                {/* Main Post */}
                <article className="bg-white rounded-2xl border border-neutral-200 p-8 mb-8">
                    <div className="mb-6">
                        <span className="inline-block px-3 py-1 text-sm text-gold-700 bg-gold-100 rounded-full mb-4">
                            {discussion.category}
                        </span>
                        <h1 className="text-3xl font-bold text-neutral-900 mb-4">{discussion.title}</h1>

                        {/* Author Info */}
                        <div className="flex items-center gap-4">
                            <div className="w-12 h-12 bg-gold-100 rounded-full flex items-center justify-center">
                                <span className="text-xl font-bold text-gold-700">{discussion.author.avatar}</span>
                            </div>
                            <div>
                                <p className="font-semibold text-neutral-900">{discussion.author.name}</p>
                                <p className="text-sm text-neutral-500">{discussion.author.role} · {discussion.date}</p>
                            </div>
                        </div>
                    </div>

                    {/* Content */}
                    <div className="prose prose-neutral max-w-none">
                        {discussion.content.split('\n\n').map((paragraph, i) => (
                            <p key={i} className="text-neutral-700 leading-relaxed mb-4 whitespace-pre-line">
                                {paragraph}
                            </p>
                        ))}
                    </div>

                    {/* Actions */}
                    <div className="flex items-center gap-6 mt-8 pt-6 border-t border-neutral-200">
                        <button className="flex items-center gap-2 text-neutral-500 hover:text-gold-600 transition-colors">
                            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                            </svg>
                            <span>Like</span>
                        </button>
                        <button className="flex items-center gap-2 text-neutral-500 hover:text-gold-600 transition-colors">
                            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
                            </svg>
                            <span>Share</span>
                        </button>
                        <button className="flex items-center gap-2 text-neutral-500 hover:text-gold-600 transition-colors">
                            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
                            </svg>
                            <span>Save</span>
                        </button>
                    </div>
                </article>

                {/* Replies Section */}
                <section>
                    <h2 className="text-xl font-semibold text-neutral-900 mb-6">
                        {discussion.replies.length} Replies
                    </h2>

                    {/* Reply Form */}
                    <form onSubmit={handleSubmitReply} className="bg-white rounded-xl border border-neutral-200 p-6 mb-6">
                        <textarea
                            value={replyContent}
                            onChange={(e) => setReplyContent(e.target.value)}
                            placeholder="Share your thoughts..."
                            className="w-full px-4 py-3 border border-neutral-200 rounded-lg focus:ring-2 focus:ring-gold-500 focus:border-transparent resize-none"
                            rows={4}
                        />
                        <div className="flex justify-end mt-4">
                            <button
                                type="submit"
                                disabled={!replyContent.trim()}
                                className="px-6 py-2 bg-gold-600 text-white font-medium rounded-lg hover:bg-gold-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                Post Reply
                            </button>
                        </div>
                    </form>

                    {/* Replies List */}
                    <div className="space-y-4">
                        {discussion.replies.map((reply) => (
                            <div key={reply.id} className="bg-white rounded-xl border border-neutral-200 p-6">
                                <div className="flex items-start gap-4">
                                    <div className="w-10 h-10 bg-gold-100 rounded-full flex items-center justify-center flex-shrink-0">
                                        <span className="text-lg font-bold text-gold-700">{reply.author.avatar}</span>
                                    </div>
                                    <div className="flex-1">
                                        <div className="flex items-center gap-2 mb-2">
                                            <p className="font-semibold text-neutral-900">{reply.author.name}</p>
                                            <span className="text-sm text-neutral-400">·</span>
                                            <p className="text-sm text-neutral-500">{reply.date}</p>
                                        </div>
                                        <p className="text-neutral-700 leading-relaxed">{reply.content}</p>
                                        <div className="flex items-center gap-4 mt-4">
                                            <button className="flex items-center gap-1 text-sm text-neutral-500 hover:text-gold-600 transition-colors">
                                                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                                                </svg>
                                                <span>{reply.likes}</span>
                                            </button>
                                            <button className="text-sm text-neutral-500 hover:text-gold-600 transition-colors">
                                                Reply
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </section>
            </div>
        </Layout>
    );
};

export default DiscussionDetailPage;
