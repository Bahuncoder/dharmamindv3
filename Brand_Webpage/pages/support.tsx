import React from 'react';
import Head from 'next/head';
import { SupportSection } from '../components/CentralizedSupport';

const SupportPage: React.FC = () => (
  <>
    <Head>
      <title>Support - DharmaMind</title>
      <meta name="description" content="Get support for DharmaMind, your AI wisdom companion" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <link rel="icon" href="/favicon.ico" />
    </Head>
    <div className="min-h-screen flex flex-col items-center justify-center bg-neutral-100">
      <SupportSection 
        title="Support"
        subtitle="Get help, contact support, or browse our help center."
        showLinks={true}
        className="max-w-xl w-full mt-12 bg-gradient-to-r from-neutral-100 to-neutral-200 rounded-lg p-8 text-center"
      />
    </div>
  </>
);

export default SupportPage;
