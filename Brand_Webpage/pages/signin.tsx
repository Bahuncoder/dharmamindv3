import React from 'react';
import Head from 'next/head';

export default function SignIn() {
  return (
    <>
      <Head>
        <title>Sign In - DharmaMind</title>
      </Head>
      <div className="min-h-screen bg-neutral-100 flex items-center justify-center">
        <div className="bg-neutral-100 p-8 rounded-lg shadow-md">
          <h1 className="text-2xl font-bold mb-4">Sign In Test Page</h1>
          <p>This is a test sign-in page to verify routing works.</p>
        </div>
      </div>
    </>
  );
}
