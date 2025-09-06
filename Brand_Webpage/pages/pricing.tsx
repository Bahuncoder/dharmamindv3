import React from 'react';
import Head from 'next/head';
import { motion } from 'framer-motion';
import { useRouter } from 'next/router';
import Logo from '../components/Logo';
import Footer from '../components/Footer';
import { useCentralizedSystem } from '../components/CentralizedSystem';
import { useSubscription } from '../hooks/useSubscription';
import { useAuth } from '../contexts/AuthContext';

const PricingPage: React.FC = () => {
  const router = useRouter();
  const { toggleSubscriptionModal, toggleAuthModal } = useCentralizedSystem();
  const { subscriptionPlans, isLoading } = useSubscription();
  const { isAuthenticated, user } = useAuth();

  const handleSelectPlan = (planId: string) => {
    if (isAuthenticated) {
      toggleSubscriptionModal(true);
    } else {
      toggleAuthModal(true);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-white to-brand-accent/5">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-brand-accent mx-auto mb-4"></div>
          <p className="text-secondary">Loading pricing plans...</p>
        </div>
      </div>
    );
  }

  return (
    <>
      <Head>
        <title>Pricing - DharmaMind</title>
        <meta name="description" content="Choose the perfect plan for your spiritual journey with DharmaMind" />
      </Head>

      <div className="min-h-screen bg-gradient-to-br from-white via-brand-accent/5 to-white">
        {/* Header */}
        <header className="relative z-50 bg-white/95 backdrop-blur-xl border-b border-gray-100">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center py-4">
              <Logo />
              <nav className="hidden md:flex space-x-8">
                <button onClick={() => router.push('/')} className="text-secondary hover:text-primary transition-colors">
                  Home
                </button>
                <button onClick={() => router.push('/features')} className="text-secondary hover:text-primary transition-colors">
                  Features
                </button>
                <button onClick={() => router.push('/about')} className="text-secondary hover:text-primary transition-colors">
                  About
                </button>
              </nav>
              <div className="flex items-center space-x-4">
                {isAuthenticated ? (
                  <span className="text-secondary">Welcome, {user?.first_name || 'User'}</span>
                ) : (
                  <button 
                    onClick={() => toggleAuthModal(true)}
                    className="btn-outline"
                  >
                    Sign In
                  </button>
                )}
              </div>
            </div>
          </div>
        </header>

        {/* Hero Section */}
        <section className="py-20">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              <div className="inline-flex items-center space-x-3 bg-white/80 backdrop-blur-xl rounded-full px-6 py-3 border border-gray-200 shadow-lg mb-8">
                <span className="text-2xl">💎</span>
                <span className="text-lg font-semibold text-primary">Pricing</span>
              </div>
              <h1 className="text-5xl md:text-6xl font-bold text-primary mb-6">
                Choose Your Spiritual Journey
              </h1>
              <p className="text-xl text-secondary max-w-3xl mx-auto">
                Start your path to enlightenment with our AI-powered dharmic guidance
              </p>
            </motion.div>
          </div>
        </section>

        {/* Centralized Pricing Section */}
        <section className="py-12">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            {subscriptionPlans && subscriptionPlans.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {subscriptionPlans.map((plan, index) => (
                  <motion.div
                    key={plan.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: index * 0.1 }}
                    className={`relative bg-white rounded-3xl p-8 shadow-xl border transition-all duration-300 hover:shadow-2xl hover:scale-105 ${
                      plan.popular ? 'border-brand-accent border-2' : 'border-gray-200'
                    }`}
                  >
                    {plan.popular && (
                      <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                        <span className="bg-brand-accent text-white px-4 py-1 text-sm font-medium rounded-full">
                          Most Popular
                        </span>
                      </div>
                    )}
                    
                    <div className="text-center mb-8">
                      <div className="w-16 h-16 bg-brand-gradient rounded-2xl flex items-center justify-center mx-auto mb-4">
                        <span className="text-white text-2xl">
                          {plan.tier === 'basic' ? '🌱' : plan.tier === 'pro' ? '🧘‍♀️' : '⭐'}
                        </span>
                      </div>
                      <h3 className="text-2xl font-bold text-primary mb-2">{plan.name}</h3>
                      <p className="text-secondary">{plan.description}</p>
                    </div>

                    <div className="text-center mb-8">
                      <span className="text-5xl font-bold text-primary">${plan.price.monthly}</span>
                      <span className="text-secondary">/month</span>
                      {plan.price.yearly && plan.price.yearly < plan.price.monthly * 12 && (
                        <div className="text-sm text-brand-accent mt-2">
                          Save ${(plan.price.monthly * 12) - plan.price.yearly}/year with annual billing
                        </div>
                      )}
                    </div>

                    <ul className="space-y-4 text-secondary mb-8">
                      {plan.features.map((feature, featureIndex) => (
                        <li key={featureIndex} className="flex items-center">
                          <div className="w-5 h-5 bg-brand-accent rounded-full flex items-center justify-center mr-3">
                            <span className="text-white text-xs">✓</span>
                          </div>
                          {feature.description || feature.feature_id}
                        </li>
                      ))}
                    </ul>

                    <button
                      onClick={() => handleSelectPlan(plan.id)}
                      className={`w-full py-3 rounded-xl font-medium transition-all duration-300 ${
                        plan.popular
                          ? 'bg-brand-gradient text-white hover:shadow-lg'
                          : 'border-2 border-brand-accent text-brand-accent hover:bg-brand-accent hover:text-white'
                      }`}
                    >
                      {plan.tier === 'basic' ? 'Get Started Free' : `Choose ${plan.name}`}
                    </button>
                  </motion.div>
                ))}
              </div>
            ) : (
              <div className="text-center py-20">
                <p className="text-xl text-secondary">Loading pricing plans...</p>
                <div className="mt-8">
                  <button
                    onClick={() => toggleSubscriptionModal(true)}
                    className="btn-primary"
                  >
                    View All Plans
                  </button>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* CTA Section */}
        <section className="py-20 bg-gradient-to-r from-brand-primary/5 to-brand-accent/5">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
            >
              <h2 className="text-4xl font-bold text-primary mb-6">
                Ready to Begin Your Journey?
              </h2>
              <p className="text-xl text-secondary mb-8">
                Join thousands of practitioners finding clarity and wisdom with DharmaMind
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <button
                  onClick={() => isAuthenticated ? toggleSubscriptionModal(true) : toggleAuthModal(true)}
                  className="btn-primary text-lg px-8 py-4"
                >
                  Get Started Today
                </button>
                <button
                  onClick={() => router.push('/features')}
                  className="btn-outline text-lg px-8 py-4"
                >
                  Explore Features
                </button>
              </div>
            </motion.div>
          </div>
        </section>

        <Footer />
      </div>
    </>
  );
};

export default PricingPage;
