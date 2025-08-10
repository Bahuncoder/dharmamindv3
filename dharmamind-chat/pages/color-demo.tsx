import React from 'react';
import Head from 'next/head';
import Link from 'next/link';

const ColorDemoPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>DharmaMind Color System Demo</title>
        <meta name="description" content="Showcase of the new DharmaMind color system with light gray backgrounds and charcoal text" />
      </Head>

      <div className="min-h-screen p-8" style={{backgroundColor: 'var(--color-bg-primary)'}}>
        
        <div className="max-w-6xl mx-auto">
          
          {/* Header */}
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold mb-4" style={{color: 'var(--color-text-primary)'}}>
              DharmaMind Color System
            </h1>
            <p className="text-lg" style={{color: 'var(--color-text-secondary)'}}>
              Mindful design with light gray backgrounds, charcoal text, and purposeful colors
            </p>
          </div>

          {/* Color Palette */}
          <div className="bg-card-primary p-8 rounded-xl mb-8">
            <h2 className="text-2xl font-bold mb-6" style={{color: 'var(--color-text-primary)'}}>
              Primary Colors
            </h2>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="w-20 h-20 rounded-lg mx-auto mb-3" 
                     style={{backgroundColor: 'var(--color-primary-saffron)'}}></div>
                <h3 className="font-semibold" style={{color: 'var(--color-text-primary)'}}>Saffron</h3>
                <p className="text-sm" style={{color: 'var(--color-text-secondary)'}}>#F2A300</p>
                <p className="text-xs" style={{color: 'var(--color-text-muted)'}}>CTA Buttons</p>
              </div>

              <div className="text-center">
                <div className="w-20 h-20 rounded-lg mx-auto mb-3" 
                     style={{backgroundColor: 'var(--color-primary-emerald)'}}></div>
                <h3 className="font-semibold" style={{color: 'var(--color-text-primary)'}}>Emerald</h3>
                <p className="text-sm" style={{color: 'var(--color-text-secondary)'}}>#32A370</p>
                <p className="text-xs" style={{color: 'var(--color-text-muted)'}}>Success</p>
              </div>

              <div className="text-center">
                <div className="w-20 h-20 rounded-lg mx-auto mb-3" 
                     style={{backgroundColor: 'var(--color-text-primary)'}}></div>
                <h3 className="font-semibold" style={{color: 'var(--color-text-primary)'}}>Charcoal</h3>
                <p className="text-sm" style={{color: 'var(--color-text-secondary)'}}>#2C2C2C</p>
                <p className="text-xs" style={{color: 'var(--color-text-muted)'}}>Primary Text</p>
              </div>

              <div className="text-center">
                <div className="w-20 h-20 rounded-lg mx-auto mb-3" 
                     style={{backgroundColor: 'var(--color-text-secondary)'}}></div>
                <h3 className="font-semibold" style={{color: 'var(--color-text-primary)'}}>Medium Gray</h3>
                <p className="text-sm" style={{color: 'var(--color-text-secondary)'}}>#6E6E6E</p>
                <p className="text-xs" style={{color: 'var(--color-text-muted)'}}>Secondary Text</p>
              </div>
            </div>
          </div>

          {/* Button Showcase */}
          <div className="bg-card-primary p-8 rounded-xl mb-8">
            <h2 className="text-2xl font-bold mb-6" style={{color: 'var(--color-text-primary)'}}>
              Button Styles
            </h2>
            
            <div className="space-y-6">
              
              {/* Primary CTA Buttons */}
              <div>
                <h3 className="text-lg font-semibold mb-4" style={{color: 'var(--color-text-primary)'}}>
                  Primary Call-to-Action (Saffron Orange)
                </h3>
                <div className="flex flex-wrap gap-4">
                  <button className="btn-primary">
                    🚀 Start Chat Now
                  </button>
                  <button className="btn-primary">
                    Send Message
                  </button>
                  <button className="btn-primary">
                    Begin Session
                  </button>
                </div>
                <p className="text-sm mt-2" style={{color: 'var(--color-text-muted)'}}>
                  Solid saffron orange (#F2A300) with white text for maximum attention
                </p>
              </div>

              {/* Success/Confirmation Buttons */}
              <div>
                <h3 className="text-lg font-semibold mb-4" style={{color: 'var(--color-text-primary)'}}>
                  Success & Confirmation (Emerald Green)
                </h3>
                <div className="flex flex-wrap gap-4">
                  <button className="btn-success">
                    ✅ Confirm
                  </button>
                  <button className="btn-success">
                    Complete
                  </button>
                  <button className="btn-success">
                    Save Changes
                  </button>
                </div>
                <p className="text-sm mt-2" style={{color: 'var(--color-text-muted)'}}>
                  Emerald green (#32A370) for positive outcomes and confirmations
                </p>
              </div>

              {/* Secondary Buttons */}
              <div>
                <h3 className="text-lg font-semibold mb-4" style={{color: 'var(--color-text-primary)'}}>
                  Secondary Actions (Ghost Style)
                </h3>
                <div className="flex flex-wrap gap-4">
                  <button className="btn-secondary">
                    Cancel
                  </button>
                  <button className="btn-secondary">
                    Go Back
                  </button>
                  <button className="btn-secondary">
                    Learn More
                  </button>
                </div>
                <p className="text-sm mt-2" style={{color: 'var(--color-text-muted)'}}>
                  Transparent background with saffron border and charcoal text
                </p>
              </div>

              {/* Ghost Buttons */}
              <div>
                <h3 className="text-lg font-semibold mb-4" style={{color: 'var(--color-text-primary)'}}>
                  Subtle Actions (Light Gray)
                </h3>
                <div className="flex flex-wrap gap-4">
                  <button className="btn-ghost">
                    View Details
                  </button>
                  <button className="btn-ghost">
                    Settings
                  </button>
                  <button className="btn-ghost">
                    Help
                  </button>
                </div>
                <p className="text-sm mt-2" style={{color: 'var(--color-text-muted)'}}>
                  Light gray background (#EEEEEE) for low-priority actions
                </p>
              </div>

            </div>
          </div>

          {/* Background Showcase */}
          <div className="bg-card-primary p-8 rounded-xl mb-8">
            <h2 className="text-2xl font-bold mb-6" style={{color: 'var(--color-text-primary)'}}>
              Background System
            </h2>
            
            <div className="grid md:grid-cols-3 gap-4">
              <div className="p-6 rounded-lg" style={{backgroundColor: 'var(--color-bg-primary)'}}>
                <h4 className="font-semibold mb-2" style={{color: 'var(--color-text-primary)'}}>
                  Primary Background
                </h4>
                <p className="text-sm" style={{color: 'var(--color-text-secondary)'}}>
                  Light gray (#F7F7F7) - Main app background
                </p>
              </div>

              <div className="p-6 rounded-lg" style={{backgroundColor: 'var(--color-bg-white)'}}>
                <h4 className="font-semibold mb-2" style={{color: 'var(--color-text-primary)'}}>
                  Card Background
                </h4>
                <p className="text-sm" style={{color: 'var(--color-text-secondary)'}}>
                  Pure white (#FFFFFF) - Cards and modals
                </p>
              </div>

              <div className="p-6 rounded-lg" style={{backgroundColor: 'var(--color-bg-secondary)'}}>
                <h4 className="font-semibold mb-2" style={{color: 'var(--color-text-primary)'}}>
                  Secondary Background
                </h4>
                <p className="text-sm" style={{color: 'var(--color-text-secondary)'}}>
                  Slightly darker (#EEEEEE) - Sections
                </p>
              </div>
            </div>
          </div>

          {/* Typography Showcase */}
          <div className="bg-card-primary p-8 rounded-xl mb-8">
            <h2 className="text-2xl font-bold mb-6" style={{color: 'var(--color-text-primary)'}}>
              Typography System
            </h2>
            
            <div className="space-y-4">
              <div>
                <h1 className="text-4xl font-bold" style={{color: 'var(--color-text-primary)'}}>
                  Primary Heading Text
                </h1>
                <p className="text-sm" style={{color: 'var(--color-text-muted)'}}>
                  Deep charcoal gray (#2C2C2C) for maximum readability
                </p>
              </div>

              <div>
                <p className="text-lg" style={{color: 'var(--color-text-secondary)'}}>
                  Secondary body text for descriptions and less important content
                </p>
                <p className="text-sm" style={{color: 'var(--color-text-muted)'}}>
                  Medium gray (#6E6E6E) creates clear hierarchy
                </p>
              </div>

              <div>
                <p className="text-sm" style={{color: 'var(--color-text-muted)'}}>
                  Muted text for timestamps, metadata, and subtle information
                </p>
                <p className="text-xs" style={{color: 'var(--color-text-muted)'}}>
                  Light gray (#999999) for minimal visual weight
                </p>
              </div>
            </div>
          </div>

          {/* Back to App */}
          <div className="text-center">
            <Link href="/" className="btn-primary">
              ← Back to DharmaMind
            </Link>
          </div>

        </div>
      </div>
    </>
  );
};

export default ColorDemoPage;
