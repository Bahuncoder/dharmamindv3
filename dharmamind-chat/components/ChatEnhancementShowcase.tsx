import React from 'react';
import { motion } from 'framer-motion';
import { 
  SparklesIcon, 
  ChatBubbleLeftRightIcon,
  MicrophoneIcon,
  MagnifyingGlassIcon,
  HeartIcon,
  StarIcon,
  EyeIcon,
  CursorArrowRaysIcon
} from '@heroicons/react/24/outline';

const ChatEnhancementShowcase: React.FC = () => {
  const features = [
    {
      icon: SparklesIcon,
      title: "Enhanced Message Bubbles",
      description: "Interactive actions, spiritual metadata, and beautiful animations",
      color: "from-emerald-400 to-emerald-600"
    },
    {
      icon: ChatBubbleLeftRightIcon,
      title: "Advanced Message Input",
      description: "Quick prompts, file attachments, voice input, and emoji picker",
      color: "from-blue-400 to-blue-600"
    },
    {
      icon: MicrophoneIcon,
      title: "Voice Integration",
      description: "Speech-to-text input with visual feedback and spiritual audio",
      color: "from-purple-400 to-purple-600"
    },
    {
      icon: MagnifyingGlassIcon,
      title: "Message Search",
      description: "Full-text search with smart filtering and tag organization",
      color: "from-yellow-400 to-yellow-600"
    },
    {
      icon: HeartIcon,
      title: "Spiritual Reactions",
      description: "Express gratitude, insights, and spiritual connection",
      color: "from-pink-400 to-pink-600"
    },
    {
      icon: StarIcon,
      title: "Meditation Mode",
      description: "Immersive dark theme with breathing guide for contemplation",
      color: "from-indigo-400 to-indigo-600"
    },
    {
      icon: EyeIcon,
      title: "Floating Background",
      description: "Sacred geometry, lotus patterns, and spiritual orbs",
      color: "from-green-400 to-green-600"
    },
    {
      icon: CursorArrowRaysIcon,
      title: "Floating Action Menu",
      description: "Quick access to notes, journal, insights, and more",
      color: "from-red-400 to-red-600"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-emerald-50 p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <motion.h1 
            className="text-5xl font-bold bg-gradient-to-r from-emerald-600 via-blue-600 to-purple-600 bg-clip-text text-transparent mb-6"
            animate={{ 
              backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'] 
            }}
            transition={{ 
              duration: 5, 
              repeat: Infinity, 
              ease: "linear" 
            }}
          >
            DharmaMind Chat Enhanced
          </motion.h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
            Experience the most advanced spiritual AI chat interface with premium features, 
            beautiful animations, and intuitive interactions designed for wisdom seekers.
          </p>
        </motion.div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 mb-16">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ 
                scale: 1.05,
                boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1)"
              }}
              className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-gray-200/50 shadow-lg hover:shadow-xl transition-all duration-300"
            >
              <div className={`w-12 h-12 rounded-xl bg-gradient-to-r ${feature.color} flex items-center justify-center mb-4 shadow-lg`}>
                <feature.icon className="w-6 h-6 text-white" />
              </div>
              <h3 className="font-semibold text-gray-900 mb-2">{feature.title}</h3>
              <p className="text-sm text-gray-600 leading-relaxed">{feature.description}</p>
            </motion.div>
          ))}
        </div>

        {/* Key Benefits */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="bg-gradient-to-r from-emerald-500 to-blue-600 rounded-3xl p-8 text-white text-center"
        >
          <h2 className="text-3xl font-bold mb-6">Spiritual Technology Perfection</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div>
              <div className="text-4xl font-bold mb-2">🧘‍♂️</div>
              <h3 className="font-semibold mb-2">Mindful Design</h3>
              <p className="text-emerald-100">Every element crafted for contemplative interaction</p>
            </div>
            <div>
              <div className="text-4xl font-bold mb-2">✨</div>
              <h3 className="font-semibold mb-2">Premium Experience</h3>
              <p className="text-emerald-100">Glass morphism, smooth animations, and spiritual aesthetics</p>
            </div>
            <div>
              <div className="text-4xl font-bold mb-2">🕉️</div>
              <h3 className="font-semibold mb-2">Dharmic Intelligence</h3>
              <p className="text-emerald-100">AI responses aligned with ancient wisdom traditions</p>
            </div>
          </div>
        </motion.div>

        {/* Technical Excellence */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1 }}
          className="mt-16 text-center"
        >
          <h2 className="text-2xl font-bold text-gray-900 mb-8">Built with Modern Technology</h2>
          <div className="flex flex-wrap justify-center gap-4">
            {[
              'React 18', 'TypeScript', 'Framer Motion', 'Tailwind CSS', 
              'Next.js', 'WebRTC', 'Speech API', 'Glass Morphism'
            ].map((tech, index) => (
              <motion.span
                key={tech}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 1 + index * 0.1 }}
                className="px-4 py-2 bg-gray-100 text-gray-700 rounded-full font-medium shadow-sm hover:shadow-md transition-all duration-200"
              >
                {tech}
              </motion.span>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default ChatEnhancementShowcase;
