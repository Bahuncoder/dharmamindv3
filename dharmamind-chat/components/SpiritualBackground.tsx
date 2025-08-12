import React from 'react';

interface SpiritualBackgroundProps {
  className?: string;
  variant?: 'default' | 'meditation' | 'minimal';
}

const SpiritualBackground: React.FC<SpiritualBackgroundProps> = ({ 
  className = '', 
  variant = 'default' 
}) => {
  return (
    <div className={`spiritual-background ${variant} ${className}`}>
      {/* Floating Orbs */}
      <div className="floating-orbs-container">
        <div className="floating-orb large" style={{ top: '10%', left: '20%', animationDelay: '0s' }}></div>
        <div className="floating-orb medium" style={{ top: '60%', right: '15%', animationDelay: '2s' }}></div>
        <div className="floating-orb small" style={{ top: '30%', left: '70%', animationDelay: '4s' }}></div>
        <div className="floating-orb medium" style={{ bottom: '20%', left: '10%', animationDelay: '6s' }}></div>
        <div className="floating-orb small" style={{ top: '80%', right: '40%', animationDelay: '8s' }}></div>
      </div>
      
      {/* Sacred Geometry Background */}
      <div className="sacred-geometry-bg"></div>
      
      {/* Lotus Patterns */}
      <div className="lotus-pattern" style={{ top: '15%', right: '5%', animationDelay: '0s' }}></div>
      <div className="lotus-pattern" style={{ bottom: '25%', left: '3%', animationDelay: '10s' }}></div>
    </div>
  );
};

export default SpiritualBackground;
