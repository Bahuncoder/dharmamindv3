/**
 * 🕉️ DharmaMind Subscription Page
 * 
 * Redirects to pricing section on homepage and opens centralized subscription modal
 * Handles redirects from dharmamind.ai with plan targeting
 */

import React, { useEffect } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';
import { useCentralizedSystem } from '../components/CentralizedSystem';

const SubscriptionPage: React.FC = () => {
  const router = useRouter();
  const { toggleSubscriptionModal } = useCentralizedSystem();

  useEffect(() => {
    // Get parameters from URL
    const { plan, upgrade, source } = router.query;
    
    // Track the source of the redirect
    if (source === 'ai' || source === 'dharmamind-ai') {
      console.log('User redirected from dharmamind.ai to subscription page');
    }
    
    // Redirect to homepage pricing section and open modal
    if (router.isReady) {
      // Redirect to the pricing section on homepage
      router.push('/#pricing');
      
      // Open the centralized subscription modal
      setTimeout(() => {
        toggleSubscriptionModal(true);
      }, 100);
    }
  }, [router.query, router.isReady, toggleSubscriptionModal, router]);

  return (
    <>
      <Head>
        <title>Subscription Plans - DharmaMind AI</title>
        <meta 
          name="description" 
          content="Choose your spiritual journey with DharmaMind AI. Basic, Pro, and Max plans available with dharmic guidance and AI wisdom." 
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-gradient-to-br from-neutral-50 via-neutral-100 to-neutral-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gold-600 mx-auto mb-4"></div>
          <p className="text-neutral-600">Redirecting to pricing...</p>
        </div>
      </div>
    </>
  );
};

export default SubscriptionPage;
