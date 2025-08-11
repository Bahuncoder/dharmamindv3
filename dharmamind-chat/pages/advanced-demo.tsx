import React from 'react';
import Head from 'next/head';
import AdvancedSpiritualChatInterface from '../components/AdvancedSpiritualChatInterface';

const AdvancedDemoPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Advanced Spiritual Chat - Universal Dharma</title>
        <meta name="description" content="Experience the most advanced spiritual chat interface with glass morphism, floating orbs, and mindfulness features." />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      
      <AdvancedSpiritualChatInterface />
    </>
  );
};

export default AdvancedDemoPage;
