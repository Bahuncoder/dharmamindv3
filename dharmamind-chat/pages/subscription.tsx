import React, { useEffect } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';

const SubscriptionPage: React.FC = () => {
  const router = useRouter();

  useEffect(() => {
    // Redirect to chat with subscription modal open
    // This ensures consistent user experience using the centralized subscription modal
    router.push('/chat?subscription=true');
  }, [router]);

  return (
    <>
      <Head>
        <title>Subscription - DharmaMind</title>
        <meta name="description" content="Manage your DharmaMind subscription" />
      </Head>
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-amber-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Redirecting to subscription management...</p>
        </div>
      </div>
    </>
  );
};

export default SubscriptionPage;
