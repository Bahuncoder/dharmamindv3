import React from 'react';
import Head from 'next/head';
import Navigation from '../components/Navigation';
import CommunityActivityFeed from '../components/CommunityActivityFeed';

const CommunityInsights: React.FC = () => {
    return (
        <>
            <Head>
                <title>Community Insights - DharmaMind Community</title>
                <meta name="description" content="Real-time insights into the DharmaMind community activity, engagement, and growth." />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <link rel="icon" href="/logo.jpeg" />
            </Head>

            <div className="min-h-screen bg-secondary-bg">
                <Navigation />

                {/* Header */}
                <div className="bg-primary-gradient text-white py-12">
                    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                        <div className="text-center">
                            <h1 className="text-5xl font-black mb-4 tracking-tight">
                                📊 Community Insights
                            </h1>
                            <p className="text-xl font-semibold opacity-90 max-w-2xl mx-auto">
                                Real-time activity feed and community engagement metrics
                            </p>
                        </div>
                    </div>
                </div>

                {/* Activity Feed */}
                <CommunityActivityFeed />
            </div>
        </>
    );
};

export default CommunityInsights;
