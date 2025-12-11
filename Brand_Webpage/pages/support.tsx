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
<<<<<<< HEAD
    <div className="min-h-screen flex flex-col items-center justify-center bg-neutral-100">
=======
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
      <SupportSection 
        title="Support"
        subtitle="Get help, contact support, or browse our help center."
        showLinks={true}
<<<<<<< HEAD
        className="max-w-xl w-full mt-12 bg-gradient-to-r from-neutral-100 to-neutral-200 rounded-lg p-8 text-center"
=======
        className="max-w-xl w-full mt-12 bg-gradient-to-r from-amber-50 to-emerald-50 rounded-lg p-8 text-center"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
      />
    </div>
  </>
);

export default SupportPage;
