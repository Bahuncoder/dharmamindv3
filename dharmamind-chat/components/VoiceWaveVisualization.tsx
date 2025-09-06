import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';

interface VoiceWaveVisualizationProps {
  isActive: boolean;
  audioLevel?: number;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
}

const VoiceWaveVisualization: React.FC<VoiceWaveVisualizationProps> = ({
  isActive,
  audioLevel = 0.5,
  className = '',
  size = 'md'
}) => {
  const [bars, setBars] = useState<number[]>([]);
  
  const barCount = size === 'sm' ? 5 : size === 'md' ? 8 : 12;
  
  useEffect(() => {
    if (!isActive) {
      setBars(new Array(barCount).fill(0.2));
      return;
    }

    const interval = setInterval(() => {
      const newBars = Array.from({ length: barCount }, () => {
        const baseLevel = audioLevel * 0.8;
        const variation = (Math.random() - 0.5) * 0.4;
        return Math.max(0.1, Math.min(1, baseLevel + variation));
      });
      setBars(newBars);
    }, 100);

    return () => clearInterval(interval);
  }, [isActive, audioLevel, barCount]);

  const getBarHeight = (index: number) => {
    if (!isActive) return '20%';
    const height = bars[index] || 0.2;
    return `${height * 100}%`;
  };

  const getSizeClasses = () => {
    switch (size) {
      case 'sm': return 'w-8 h-6';
      case 'md': return 'w-12 h-8';
      case 'lg': return 'w-16 h-12';
    }
  };

  const getBarWidth = () => {
    switch (size) {
      case 'sm': return '2px';
      case 'md': return '3px';
      case 'lg': return '4px';
    }
  };

  return (
    <div className={`flex items-end justify-center space-x-1 ${getSizeClasses()} ${className}`}>
      {Array.from({ length: barCount }).map((_, index) => (
        <motion.div
          key={index}
          className={`bg-gradient-to-t ${
            isActive 
              ? 'from-emerald-400 to-emerald-600' 
              : 'from-gray-300 to-gray-400'
          } rounded-t-sm`}
          style={{ width: getBarWidth() }}
          animate={{
            height: getBarHeight(index),
          }}
          transition={{
            duration: 0.1,
            ease: "easeOut"
          }}
        />
      ))}
    </div>
  );
};

export default VoiceWaveVisualization;
