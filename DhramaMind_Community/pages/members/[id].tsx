/**
 * DharmaMind Community - Member Profile Page
 */

import React from 'react';
import { useRouter } from 'next/router';
import { Layout } from '../../components/Layout';
import Link from 'next/link';

// Mock data - would come from API
const mockMember = {
  id: 1,
  name: 'Alex Martinez',
  role: 'Leadership Coach',
  avatar: 'A',
  bio: 'Leadership coach and mindfulness practitioner with 15+ years of experience helping executives and teams develop authentic, conscious leadership skills. Passionate about bridging ancient wisdom traditions with modern organizational challenges.',
  location: 'San Francisco, CA',
  joined: 'January 2024',
  website: 'https://alexmartinez.com',
  stats: {
    discussions: 156,
    replies: 523,
    likes: 1248,
    following: 89,
    followers: 312,
  },
  badges: [
    { name: 'Top Contributor', icon: '🏆' },
    { name: 'Mindfulness Master', icon: '🧘' },
    { name: 'Community Builder', icon: '🤝' },
  ],
  interests: ['Leadership', 'Mindfulness', 'Philosophy', 'Wellness', 'Technology'],
  recentActivity: [
    {
      type: 'discussion',
      title: 'Best practices for mindful leadership',
      date: '2 hours ago',
      id: 1,
    },
    {
      type: 'reply',
      title: 'How to maintain focus in a distracted world',
      date: '1 day ago',
      id: 2,
    },
    {
      type: 'discussion',
      title: 'The intersection of AI and human wisdom',
      date: '3 days ago',
      id: 3,
    },
  ],
};

const MemberProfilePage: React.FC = () => {
  const router = useRouter();
  const { id } = router.query;
  
  const member = mockMember; // In production, fetch by id

  return (
    <Layout title={`${member.name} - Profile`} description={`${member.name}'s profile on DharmaMind Community`}>
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Breadcrumb */}
        <nav className="mb-8">
          <Link href="/members" className="text-gold-600 hover:text-gold-700 transition-colors">
            ← Back to Members
          </Link>
        </nav>

        {/* Profile Header */}
        <div className="bg-white rounded-2xl border border-neutral-200 p-8 mb-8">
          <div className="flex flex-col md:flex-row items-start gap-6">
            {/* Avatar */}
            <div className="w-24 h-24 bg-gold-100 rounded-full flex items-center justify-center flex-shrink-0">
              <span className="text-4xl font-bold text-gold-700">{member.avatar}</span>
            </div>

            {/* Info */}
            <div className="flex-1">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h1 className="text-2xl font-bold text-neutral-900">{member.name}</h1>
                  <p className="text-neutral-600">{member.role}</p>
                </div>
                <button className="px-6 py-2 bg-gold-600 text-white font-medium rounded-lg hover:bg-gold-700 transition-colors">
                  Follow
                </button>
              </div>

              <p className="text-neutral-700 mt-4 leading-relaxed">{member.bio}</p>

              {/* Meta Info */}
              <div className="flex flex-wrap items-center gap-4 mt-4 text-sm text-neutral-500">
                <span className="flex items-center gap-1">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  {member.location}
                </span>
                <span className="flex items-center gap-1">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  Joined {member.joined}
                </span>
                {member.website && (
                  <a href={member.website} target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 text-gold-600 hover:text-gold-700">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                    </svg>
                    Website
                  </a>
                )}
              </div>

              {/* Badges */}
              <div className="flex flex-wrap gap-2 mt-4">
                {member.badges.map((badge, i) => (
                  <span key={i} className="inline-flex items-center gap-1 px-3 py-1 bg-gold-50 text-gold-700 text-sm rounded-full">
                    <span>{badge.icon}</span>
                    <span>{badge.name}</span>
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
          {[
            { label: 'Discussions', value: member.stats.discussions },
            { label: 'Replies', value: member.stats.replies },
            { label: 'Likes', value: member.stats.likes },
            { label: 'Following', value: member.stats.following },
            { label: 'Followers', value: member.stats.followers },
          ].map((stat, i) => (
            <div key={i} className="bg-white rounded-xl border border-neutral-200 p-4 text-center">
              <p className="text-2xl font-bold text-neutral-900">{stat.value.toLocaleString()}</p>
              <p className="text-sm text-neutral-500">{stat.label}</p>
            </div>
          ))}
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {/* Interests */}
          <div className="md:col-span-1">
            <div className="bg-white rounded-xl border border-neutral-200 p-6">
              <h2 className="font-semibold text-neutral-900 mb-4">Interests</h2>
              <div className="flex flex-wrap gap-2">
                {member.interests.map((interest, i) => (
                  <span key={i} className="px-3 py-1 bg-neutral-100 text-neutral-600 text-sm rounded-full">
                    {interest}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Recent Activity */}
          <div className="md:col-span-2">
            <div className="bg-white rounded-xl border border-neutral-200 p-6">
              <h2 className="font-semibold text-neutral-900 mb-4">Recent Activity</h2>
              <div className="space-y-4">
                {member.recentActivity.map((activity, i) => (
                  <Link
                    key={i}
                    href={`/discussions/${activity.id}`}
                    className="block p-4 bg-neutral-50 rounded-lg hover:bg-neutral-100 transition-colors"
                  >
                    <div className="flex items-center gap-2 text-sm text-neutral-500 mb-1">
                      <span className={activity.type === 'discussion' ? 'text-gold-600' : 'text-neutral-500'}>
                        {activity.type === 'discussion' ? 'Started discussion' : 'Replied to'}
                      </span>
                      <span>·</span>
                      <span>{activity.date}</span>
                    </div>
                    <p className="text-neutral-900 font-medium">{activity.title}</p>
                  </Link>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default MemberProfilePage;
