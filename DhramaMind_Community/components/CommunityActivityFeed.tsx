import React, { useState, useEffect } from 'react';

interface User {
    id: string;
    name: string;
    avatar: string;
    role: string;
    isOnline: boolean;
    lastSeen: Date;
    contributions: number;
    joinDate: string;
    badges: string[];
    specialties: string[];
}

interface Activity {
    id: string;
    type: 'join' | 'post' | 'comment' | 'like' | 'achievement';
    user: User;
    content: string;
    timestamp: Date;
    relatedTo?: string;
}

const CommunityActivityFeed: React.FC = () => {
    const [activities, setActivities] = useState<Activity[]>([]);
    const [onlineUsers, setOnlineUsers] = useState<User[]>([]);
    const [filter, setFilter] = useState<string>('all');

    // Sample data
    useEffect(() => {
        const sampleUsers: User[] = [
            {
                id: '1',
                name: 'Sarah Chen',
                avatar: '👩‍💼',
                role: 'Community Guide',
                isOnline: true,
                lastSeen: new Date(),
                contributions: 125,
                joinDate: 'March 2024',
                badges: ['🌟 Top Contributor', '🧘‍♀️ Meditation Master'],
                specialties: ['Mindfulness', 'Daily Practice']
            },
            {
                id: '2',
                name: 'Michael Torres',
                avatar: '👨‍🎓',
                role: 'Dharma Scholar',
                isOnline: true,
                lastSeen: new Date(),
                contributions: 89,
                joinDate: 'February 2024',
                badges: ['📚 Wisdom Keeper', '☸️ Dharma Teacher'],
                specialties: ['Buddhist Philosophy', 'Meditation']
            },
            {
                id: '3',
                name: 'Emma Watson',
                avatar: '👩‍💻',
                role: 'Mindful Professional',
                isOnline: false,
                lastSeen: new Date(Date.now() - 30 * 60 * 1000),
                contributions: 56,
                joinDate: 'April 2024',
                badges: ['💼 Work-Life Balance'],
                specialties: ['Workplace Mindfulness', 'Stress Management']
            }
        ];

        const sampleActivities: Activity[] = [
            {
                id: '1',
                type: 'post',
                user: sampleUsers[0],
                content: 'shared insights on "Maintaining consistency in meditation practice"',
                timestamp: new Date(Date.now() - 5 * 60 * 1000),
                relatedTo: 'How to maintain consistent meditation practice?'
            },
            {
                id: '2',
                type: 'join',
                user: {
                    id: '4',
                    name: 'Alex Kim',
                    avatar: '👨‍⚕️',
                    role: 'New Member',
                    isOnline: true,
                    lastSeen: new Date(),
                    contributions: 0,
                    joinDate: 'Today',
                    badges: ['🌱 New Journey'],
                    specialties: []
                },
                content: 'joined the community',
                timestamp: new Date(Date.now() - 15 * 60 * 1000)
            },
            {
                id: '3',
                type: 'achievement',
                user: sampleUsers[1],
                content: 'earned the "Wisdom Keeper" badge for sharing valuable insights',
                timestamp: new Date(Date.now() - 45 * 60 * 1000)
            },
            {
                id: '4',
                type: 'comment',
                user: sampleUsers[2],
                content: 'added thoughtful commentary to the Four Noble Truths discussion',
                timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
                relatedTo: 'Understanding the Four Noble Truths in modern context'
            },
            {
                id: '5',
                type: 'like',
                user: sampleUsers[0],
                content: 'appreciated a post about mindful workspace setup',
                timestamp: new Date(Date.now() - 3 * 60 * 60 * 1000),
                relatedTo: 'Creating a peaceful workspace for remote work'
            }
        ];

        setOnlineUsers(sampleUsers.filter(user => user.isOnline));
        setActivities(sampleActivities);
    }, []);

    const getActivityIcon = (type: string) => {
        const icons = {
            join: '👋',
            post: '📝',
            comment: '💭',
            like: '❤️',
            achievement: '🏆'
        };
        return icons[type as keyof typeof icons] || '📋';
    };

    const getActivityColor = (type: string) => {
        const colors = {
            join: 'border-l-success',
            post: 'border-l-info',
            comment: 'border-l-primary',
            like: 'border-l-error',
            achievement: 'border-l-warning'
        };
        return colors[type as keyof typeof colors] || 'border-l-neutral-400';
    };

    const formatTime = (timestamp: Date) => {
        const now = new Date();
        const diff = now.getTime() - timestamp.getTime();
        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(minutes / 60);

        if (hours > 0) return `${hours}h ago`;
        return `${minutes}m ago`;
    };

    const filteredActivities = filter === 'all'
        ? activities
        : activities.filter(activity => activity.type === filter);

    return (
        <div className="max-w-6xl mx-auto p-6">
            <div className="grid lg:grid-cols-4 gap-8">
                {/* Main Activity Feed */}
                <div className="lg:col-span-3">
                    <div className="mb-6">
                        <h2 className="text-2xl font-black text-primary mb-4">Community Activity</h2>

                        {/* Activity Filters */}
                        <div className="flex flex-wrap gap-2 mb-6">
                            {[
                                { id: 'all', name: 'All Activity', icon: '📋' },
                                { id: 'join', name: 'New Members', icon: '👋' },
                                { id: 'post', name: 'Posts', icon: '📝' },
                                { id: 'comment', name: 'Comments', icon: '💭' },
                                { id: 'achievement', name: 'Achievements', icon: '🏆' }
                            ].map(filterOption => (
                                <button
                                    key={filterOption.id}
                                    onClick={() => setFilter(filterOption.id)}
                                    className={`px-4 py-2 rounded-lg font-semibold transition-all duration-200 flex items-center gap-2 ${filter === filterOption.id
                                            ? 'bg-primary text-white'
                                            : 'bg-neutral-100 text-secondary hover:bg-neutral-200'
                                        }`}
                                >
                                    <span>{filterOption.icon}</span>
                                    <span className="hidden sm:inline">{filterOption.name}</span>
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Activity List */}
                    <div className="space-y-4">
                        {filteredActivities.map(activity => (
                            <div
                                key={activity.id}
                                className={`bg-primary rounded-lg p-6 shadow-lg hover:shadow-xl transition-all duration-300 border-l-4 ${getActivityColor(activity.type)}`}
                            >
                                <div className="flex items-start gap-4">
                                    <div className="flex-shrink-0">
                                        <div className="relative">
                                            <div className="w-12 h-12 bg-primary-gradient rounded-full flex items-center justify-center text-white text-lg font-bold">
                                                {activity.user.avatar}
                                            </div>
                                            {activity.user.isOnline && (
                                                <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-success rounded-full border-2 border-white"></div>
                                            )}
                                        </div>
                                    </div>

                                    <div className="flex-1">
                                        <div className="flex items-start justify-between">
                                            <div>
                                                <div className="flex items-center gap-2 mb-1">
                                                    <span className="text-lg">{getActivityIcon(activity.type)}</span>
                                                    <h3 className="font-bold text-primary">{activity.user.name}</h3>
                                                    <span className="text-sm bg-neutral-200 px-2 py-1 rounded text-secondary">
                                                        {activity.user.role}
                                                    </span>
                                                </div>
                                                <p className="text-secondary">
                                                    {activity.content}
                                                    {activity.relatedTo && (
                                                        <span className="block mt-1 text-sm text-primary font-semibold">
                                                            "{activity.relatedTo}"
                                                        </span>
                                                    )}
                                                </p>
                                            </div>
                                            <span className="text-sm text-muted font-medium">
                                                {formatTime(activity.timestamp)}
                                            </span>
                                        </div>

                                        {/* User badges for achievement activities */}
                                        {activity.type === 'achievement' && activity.user.badges.length > 0 && (
                                            <div className="mt-3 flex flex-wrap gap-2">
                                                {activity.user.badges.map(badge => (
                                                    <span key={badge} className="bg-warning bg-opacity-20 text-warning px-3 py-1 rounded-full text-sm font-medium">
                                                        {badge}
                                                    </span>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        ))}

                        {filteredActivities.length === 0 && (
                            <div className="text-center py-12 bg-primary rounded-lg">
                                <span className="text-4xl mb-4 block">🌱</span>
                                <h3 className="text-xl font-bold text-primary mb-2">No activity yet</h3>
                                <p className="text-secondary">Be the first to contribute to the community!</p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Sidebar */}
                <div className="lg:col-span-1 space-y-6">
                    {/* Online Members */}
                    <div className="bg-primary rounded-lg p-6 sticky top-6">
                        <h3 className="text-lg font-black text-primary mb-4 flex items-center gap-2">
                            <span className="w-3 h-3 bg-success rounded-full animate-pulse"></span>
                            Online Members ({onlineUsers.length})
                        </h3>
                        <div className="space-y-3">
                            {onlineUsers.map(user => (
                                <div key={user.id} className="flex items-center gap-3 p-3 rounded-lg hover:bg-bg-tertiary transition-colors cursor-pointer">
                                    <div className="relative">
                                        <div className="w-10 h-10 bg-primary-gradient rounded-full flex items-center justify-center text-white text-sm font-bold">
                                            {user.avatar}
                                        </div>
                                        <div className="absolute -bottom-1 -right-1 w-3 h-3 bg-success rounded-full border-2 border-white"></div>
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <p className="text-sm font-bold text-primary truncate">{user.name}</p>
                                        <p className="text-xs text-muted">{user.role}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Community Stats */}
                    <div className="bg-primary rounded-lg p-6">
                        <h3 className="text-lg font-black text-primary mb-4">Community Stats</h3>
                        <div className="space-y-4">
                            <div className="flex justify-between items-center">
                                <span className="text-secondary font-semibold">Total Members</span>
                                <span className="font-bold text-primary">2,847</span>
                            </div>
                            <div className="flex justify-between items-center">
                                <span className="text-secondary font-semibold">Active Today</span>
                                <span className="font-bold text-success">234</span>
                            </div>
                            <div className="flex justify-between items-center">
                                <span className="text-secondary font-semibold">New This Week</span>
                                <span className="font-bold text-info">67</span>
                            </div>
                            <div className="flex justify-between items-center">
                                <span className="text-secondary font-semibold">Discussions</span>
                                <span className="font-bold text-primary">1,456</span>
                            </div>
                        </div>
                    </div>

                    {/* Quick Actions */}
                    <div className="bg-primary rounded-lg p-6">
                        <h3 className="text-lg font-black text-primary mb-4">Quick Actions</h3>
                        <div className="space-y-3">
                            <button className="w-full bg-primary-gradient text-white px-4 py-3 rounded-lg font-bold hover:opacity-90 transition-opacity flex items-center justify-center gap-2">
                                <span>📝</span>
                                <span>New Discussion</span>
                            </button>
                            <button className="w-full bg-neutral-100 text-secondary px-4 py-3 rounded-lg font-bold hover:bg-neutral-200 transition-colors flex items-center justify-center gap-2">
                                <span>📅</span>
                                <span>View Events</span>
                            </button>
                            <button className="w-full bg-neutral-100 text-secondary px-4 py-3 rounded-lg font-bold hover:bg-neutral-200 transition-colors flex items-center justify-center gap-2">
                                <span>👥</span>
                                <span>Find Members</span>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default CommunityActivityFeed;
