import React from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useRouter } from 'next/router';
import { useEffect } from 'react';

export default function Welcome() {
  const { user } = useAuth();
  const router = useRouter();

  useEffect(() => {
    // Redirect to chat if user is already authenticated
    if (user) {
      router.push('/chat');
    }
  }, [user, router]);

  if (user) {
    return <div>Redirecting...</div>;
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-6">
            Welcome to DharmaMind
          </h1>
          <p className="text-xl text-gray-600 mb-8">
            Your spiritual companion for wisdom and guidance
          </p>
          <div className="space-x-4">
            <button
              onClick={() => router.push('/login')}
              className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors"
            >
              Sign In
            </button>
            <button
              onClick={() => router.push('/signup')}
              className="bg-white text-blue-600 border border-blue-600 px-6 py-3 rounded-lg hover:bg-blue-50 transition-colors"
            >
              Sign Up
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
