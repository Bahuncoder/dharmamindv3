import React from 'react';
import Head from 'next/head';

export default function ButtonDemo() {
  return (
    <>
      <Head>
        <title>Button Demo - DharmaMind</title>
      </Head>
      
      <div className="min-h-screen" style={{ backgroundColor: 'var(--color-bg-light)' }}>
        <div className="container mx-auto px-4 py-16">
          <h1 className="text-4xl font-bold text-center mb-16" style={{ color: 'var(--color-text-primary)' }}>
            DharmaMind Button Showcase
          </h1>
          
          <div className="max-w-4xl mx-auto space-y-12">
            
            {/* Primary Button */}
            <div className="text-center">
              <h2 className="text-2xl font-semibold mb-6" style={{ color: 'var(--color-text-primary)' }}>
                Primary Button (4px Emerald Border)
              </h2>
              <button className="btn-primary mx-4">
                Sign Up Now
              </button>
              <button className="btn-primary mx-4">
                Get Started
              </button>
              <button className="btn-primary mx-4">
                Learn More
              </button>
            </div>
            
            {/* Premium Button */}
            <div className="text-center">
              <h2 className="text-2xl font-semibold mb-6" style={{ color: 'var(--color-text-primary)' }}>
                Premium Button (5px Emerald Border + Shadow)
              </h2>
              <button className="btn-premium mx-4">
                Start Your Journey
              </button>
              <button className="btn-premium mx-4">
                Subscribe Premium
              </button>
            </div>
            
            {/* Ultra Bordered Button */}
            <div className="text-center">
              <h2 className="text-2xl font-semibold mb-6" style={{ color: 'var(--color-text-primary)' }}>
                Ultra Bordered Button (6px + Double Shadow)
              </h2>
              <button className="btn-bordered mx-4">
                Join DharmaMind
              </button>
              <button className="btn-bordered mx-4">
                Transform Your Life
              </button>
            </div>
            
            {/* Color Showcase */}
            <div className="bg-white p-8 rounded-2xl shadow-lg">
              <h2 className="text-2xl font-semibold mb-6 text-center" style={{ color: 'var(--color-text-primary)' }}>
                Color Combination Showcase
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="text-center">
                  <h3 className="text-lg font-medium mb-4" style={{ color: 'var(--color-text-primary)' }}>
                    Emerald Green Border
                  </h3>
                  <div 
                    className="w-32 h-16 mx-auto rounded-lg border-4"
                    style={{ 
                      borderColor: 'var(--color-success)',
                      backgroundColor: 'var(--color-bg-light)'
                    }}
                  ></div>
                  <p className="mt-2 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                    #32A370
                  </p>
                </div>
                <div className="text-center">
                  <h3 className="text-lg font-medium mb-4" style={{ color: 'var(--color-text-primary)' }}>
                    Saffron Orange Interior
                  </h3>
                  <div 
                    className="w-32 h-16 mx-auto rounded-lg"
                    style={{ 
                      background: 'linear-gradient(135deg, var(--color-primary-saffron), var(--color-primary-saffron-dark))'
                    }}
                  ></div>
                  <p className="mt-2 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                    #F2A300 → #D4910A
                  </p>
                </div>
              </div>
            </div>
            
            {/* Interactive Demo */}
            <div className="text-center">
              <h2 className="text-2xl font-semibold mb-6" style={{ color: 'var(--color-text-primary)' }}>
                Interactive Demo - Hover to See Effects
              </h2>
              <div className="flex justify-center space-x-6 flex-wrap gap-4">
                <button className="btn-primary">
                  Hover Me (Primary)
                </button>
                <button className="btn-premium">
                  Hover Me (Premium)
                </button>
                <button className="btn-bordered">
                  Hover Me (Ultra Border)
                </button>
              </div>
            </div>
            
          </div>
        </div>
      </div>
    </>
  );
}
