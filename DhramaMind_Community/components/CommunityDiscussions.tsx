import React, { useState } from 'react';
import { useNotifications } from '../contexts/NotificationContext';

interface Discussion {
    id: string;
    title: string;
    description: string;
    category: string;
    author: {
        name: string;
        avatar: string;
        role: string;
    };
    createdAt: Date;
    replies: number;
    likes: number;
    isLiked?: boolean;
    tags: string[];
    lastActivity: Date;
    isPinned?: boolean;
}

const CommunityDiscussions: React.FC = () => {
    const { addNotification } = useNotifications();
    const [activeCategory, setActiveCategory] = useState('all');
    const [searchTerm, setSearchTerm] = useState('');
    const [sortBy, setSortBy] = useState('recent');

    const categories = [
        { id: 'all', name: 'All Topics', count: 24, icon: '📚' },
        { id: 'meditation', name: 'Meditation', count: 8, icon: '🧘‍♀️' },
        { id: 'dharma', name: 'Dharma Wisdom', count: 6, icon: '☸️' },
        { id: 'daily-life', name: 'Daily Life', count: 5, icon: '🌱' },
        { id: 'community', name: 'Community', count: 3, icon: '👥' },
        { id: 'questions', name: 'Questions', count: 2, icon: '❓' },
    ];

    const [discussions] = useState<Discussion[]>([
        {
            id: '1',
            title: 'How to maintain consistent meditation practice?',
            description: 'I have been struggling to keep up with my daily meditation routine. Looking for practical advice and experiences from fellow practitioners.',
            category: 'meditation',
            author: {
                name: 'Sarah Chen',
                avatar: '👩‍💼',
                role: 'Community Guide'
            },
            createdAt: new Date(Date.now() - 2 * 60 * 60 * 1000),
            replies: 12,
            likes: 28,
            isLiked: false,
            tags: ['meditation', 'routine', 'consistency'],
            lastActivity: new Date(Date.now() - 30 * 60 * 1000),
            isPinned: true
        },
        {
            id: '2',
            title: 'Understanding the Four Noble Truths in modern context',
            description: 'How can we apply the Four Noble Truths to navigate stress and challenges in our contemporary world?',
            category: 'dharma',
            author: {
                name: 'Michael Torres',
                avatar: '👨‍🎓',
                role: 'Dharma Scholar'
            },
            createdAt: new Date(Date.now() - 4 * 60 * 60 * 1000),
            replies: 18,
            likes: 42,
            isLiked: true,
            tags: ['four-noble-truths', 'philosophy', 'modern-life'],
            lastActivity: new Date(Date.now() - 45 * 60 * 1000),
            isPinned: false
        },
        {
            id: '3',
            title: 'Creating a peaceful workspace for remote work',
            description: 'Share your tips and setups for maintaining mindfulness and calm while working from home.',
            category: 'daily-life',
            author: {
                name: 'Emma Watson',
                avatar: '👩‍💻',
                role: 'Mindful Professional'
            },
            createdAt: new Date(Date.now() - 6 * 60 * 60 * 1000),
            replies: 9,
            likes: 23,
            isLiked: false,
            tags: ['workspace', 'remote-work', 'mindfulness'],
            lastActivity: new Date(Date.now() - 1 * 60 * 60 * 1000),
            isPinned: false
        },
        {
            id: '4',
            title: 'Welcome new members! Share your spiritual journey',
            description: 'A warm welcome thread for new community members to introduce themselves and share what brings them to the dharma path.',
            category: 'community',
            author: {
                name: 'DharmaMind Team',
                avatar: '🕉️',
                role: 'Community Admin'
            },
            createdAt: new Date(Date.now() - 12 * 60 * 60 * 1000),
            replies: 34,
            likes: 67,
            isLiked: true,
            tags: ['welcome', 'introductions', 'community'],
            lastActivity: new Date(Date.now() - 15 * 60 * 1000),
            isPinned: true
        },
        {
            id: '5',
            title: 'Dealing with difficult emotions through mindfulness',
            description: 'How do you use mindfulness and Buddhist teachings to work with anger, fear, and sadness in healthy ways?',
            category: 'meditation',
            author: {
                name: 'Alex Thompson',
                avatar: '👨‍⚕️',
                role: 'Mindfulness Practitioner'
            },
            createdAt: new Date(Date.now() - 18 * 60 * 60 * 1000),
            replies: 15,
            likes: 31,
            isLiked: false,
            tags: ['emotions', 'mindfulness', 'mental-health'],
            lastActivity: new Date(Date.now() - 2 * 60 * 60 * 1000),
            isPinned: false
        }
    ]);

    const handleLike = (discussionId: string) => {
        // In a real app, this would make an API call
        addNotification({
            type: 'success',
            title: 'Discussion Liked!',
            message: 'Thank you for engaging with the community.',
            read: false
        });
    };

    const handleReply = (discussionId: string) => {
        // In a real app, this would navigate to the discussion detail page
        addNotification({
            type: 'info',
            title: 'Ready to Reply',
            message: 'Share your thoughts and wisdom with the community.',
            read: false
        });
    };

    const formatTime = (date: Date) => {
        const now = new Date();
        const diff = now.getTime() - date.getTime();
        const hours = Math.floor(diff / (1000 * 60 * 60));

        if (hours < 1) {
            const minutes = Math.floor(diff / (1000 * 60));
            return `${minutes}m ago`;
        } else if (hours < 24) {
            return `${hours}h ago`;
        } else {
            const days = Math.floor(hours / 24);
            return `${days}d ago`;
        }
    };

    const filteredDiscussions = discussions.filter(discussion => {
        const matchesCategory = activeCategory === 'all' || discussion.category === activeCategory;
        const matchesSearch = discussion.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
            discussion.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
            discussion.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
        return matchesCategory && matchesSearch;
    });

    return (
        <div className="max-w-6xl mx-auto p-6">
            {/* Header */}
            <div className="mb-8">
                <h1 className="text-3xl font-black text-primary mb-2">Community Discussions</h1>
                <p className="text-lg text-secondary font-semibold">
                    Connect, share wisdom, and grow together on the dharma path
                </p>
            </div>

            {/* Search and Filters */}
            <div className="mb-8 bg-primary rounded-xl p-6 shadow-lg">
                <div className="flex flex-col lg:flex-row gap-4">
                    {/* Search */}
                    <div className="flex-1">
                        <div className="relative">
                            <span className="absolute left-3 top-1/2 transform -translate-y-1/2 text-lg">🔍</span>
                            <input
                                type="text"
                                placeholder="Search discussions, topics, or tags..."
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                                className="w-full pl-12 pr-4 py-3 rounded-lg border border-border-light focus:border-primary focus:ring-2 focus:ring-focus-ring outline-none transition-colors"
                            />
                        </div>
                    </div>

                    {/* Sort */}
                    <div className="lg:w-48">
                        <select
                            value={sortBy}
                            onChange={(e) => setSortBy(e.target.value)}
                            className="w-full px-4 py-3 rounded-lg border border-border-light focus:border-primary focus:ring-2 focus:ring-focus-ring outline-none transition-colors"
                        >
                            <option value="recent">Most Recent</option>
                            <option value="popular">Most Popular</option>
                            <option value="replies">Most Replies</option>
                        </select>
                    </div>

                    {/* New Discussion Button */}
                    <button
                        onClick={() => addNotification({
                            type: 'info',
                            title: 'Start New Discussion',
                            message: 'Feature coming soon! Share your thoughts with the community.',
                            read: false
                        })}
                        className="bg-primary-gradient text-white px-6 py-3 rounded-lg font-bold hover:opacity-90 transition-opacity whitespace-nowrap"
                    >
                        + New Discussion
                    </button>
                </div>
            </div>

            <div className="grid lg:grid-cols-4 gap-8">
                {/* Categories Sidebar */}
                <div className="lg:col-span-1">
                    <div className="bg-primary rounded-xl p-6 sticky top-6">
                        <h3 className="text-lg font-black text-primary mb-4">Categories</h3>
                        <div className="space-y-2">
                            {categories.map(category => (
                                <button
                                    key={category.id}
                                    onClick={() => setActiveCategory(category.id)}
                                    className={`w-full flex items-center justify-between p-3 rounded-lg transition-all duration-200 ${activeCategory === category.id
                                            ? 'bg-primary-gradient text-white shadow-lg'
                                            : 'text-secondary hover:bg-bg-tertiary'
                                        }`}
                                >
                                    <div className="flex items-center gap-3">
                                        <span className="text-lg">{category.icon}</span>
                                        <span className="font-semibold">{category.name}</span>
                                    </div>
                                    <span className={`text-sm px-2 py-1 rounded-full ${activeCategory === category.id ? 'bg-white bg-opacity-20' : 'bg-neutral-200'
                                        }`}>
                                        {category.count}
                                    </span>
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Discussions List */}
                <div className="lg:col-span-3 space-y-4">
                    {filteredDiscussions.map(discussion => (
                        <div
                            key={discussion.id}
                            className={`bg-primary rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 border-l-4 ${discussion.isPinned ? 'border-l-warning' : 'border-l-primary'
                                }`}
                        >
                            {/* Discussion Header */}
                            <div className="flex items-start justify-between mb-4">
                                <div className="flex items-center gap-3 flex-1">
                                    <div className="flex items-center gap-2">
                                        <span className="text-2xl">{discussion.author.avatar}</span>
                                        <div>
                                            <h4 className="font-bold text-primary text-sm">{discussion.author.name}</h4>
                                            <p className="text-xs text-muted">{discussion.author.role}</p>
                                        </div>
                                    </div>
                                    {discussion.isPinned && (
                                        <span className="bg-warning text-white px-2 py-1 rounded text-xs font-semibold">
                                            📌 Pinned
                                        </span>
                                    )}
                                </div>
                                <span className="text-sm text-muted font-medium">
                                    {formatTime(discussion.createdAt)}
                                </span>
                            </div>

                            {/* Discussion Content */}
                            <div className="mb-4">
                                <h3 className="text-xl font-black text-primary mb-2 hover:text-secondary cursor-pointer transition-colors">
                                    {discussion.title}
                                </h3>
                                <p className="text-secondary leading-relaxed">
                                    {discussion.description}
                                </p>
                            </div>

                            {/* Tags */}
                            <div className="flex flex-wrap gap-2 mb-4">
                                {discussion.tags.map(tag => (
                                    <span
                                        key={tag}
                                        className="bg-neutral-200 text-secondary px-3 py-1 rounded-full text-sm font-medium hover:bg-neutral-300 cursor-pointer transition-colors"
                                    >
                                        #{tag}
                                    </span>
                                ))}
                            </div>

                            {/* Discussion Footer */}
                            <div className="flex items-center justify-between pt-4 border-t border-border-light">
                                <div className="flex items-center gap-6">
                                    <button
                                        onClick={() => handleLike(discussion.id)}
                                        className={`flex items-center gap-2 px-3 py-1 rounded-lg transition-all duration-200 ${discussion.isLiked
                                                ? 'bg-error text-white'
                                                : 'text-secondary hover:bg-error hover:text-white'
                                            }`}
                                    >
                                        <span>❤️</span>
                                        <span className="font-semibold">{discussion.likes}</span>
                                    </button>

                                    <button
                                        onClick={() => handleReply(discussion.id)}
                                        className="flex items-center gap-2 text-secondary hover:bg-primary hover:text-white px-3 py-1 rounded-lg transition-all duration-200"
                                    >
                                        <span>💬</span>
                                        <span className="font-semibold">{discussion.replies}</span>
                                    </button>
                                </div>

                                <div className="text-sm text-muted font-medium">
                                    Last activity: {formatTime(discussion.lastActivity)}
                                </div>
                            </div>
                        </div>
                    ))}

                    {filteredDiscussions.length === 0 && (
                        <div className="text-center py-16 bg-primary rounded-xl">
                            <span className="text-6xl mb-4 block">🔍</span>
                            <h3 className="text-xl font-black text-primary mb-2">No discussions found</h3>
                            <p className="text-secondary">
                                Try adjusting your search or category filters
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default CommunityDiscussions;
