import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '../contexts/AuthContext';

interface OnboardingStep {
  id: string;
  title: string;
  description: string;
  icon: string;
  component: React.ReactNode;
  isComplete?: boolean;
}

interface UserPreferences {
  spiritualTradition: string;
  experienceLevel: string;
  primaryGoals: string[];
  communicationStyle: string;
  language: string;
  notifications: boolean;
  voiceInput: boolean;
}

interface UserOnboardingProps {
  isDemo?: boolean;
}

const UserOnboarding: React.FC<UserOnboardingProps> = ({ isDemo = false }) => {
  const router = useRouter();
  const { user } = useAuth();
  const [currentStep, setCurrentStep] = useState(0);
  const [preferences, setPreferences] = useState<UserPreferences>({
    spiritualTradition: '',
    experienceLevel: '',
    primaryGoals: [],
    communicationStyle: '',
    language: 'en',
    notifications: true,
    voiceInput: false
  });
  const [isCompleting, setIsCompleting] = useState(false);

  // Mock update function for now
  const updateUserPreferences = async (prefs: UserPreferences) => {
    // This would normally call an API
    console.log('Updating user preferences:', prefs);
    return Promise.resolve();
  };

  const steps: OnboardingStep[] = [
    {
      id: 'welcome',
      title: 'Welcome to DharmaMind',
      description: 'Let\'s personalize your spiritual AI companion',
      icon: '🕉️',
      component: <WelcomeStep />
    },
    {
      id: 'tradition',
      title: 'Spiritual Tradition',
      description: 'Which wisdom tradition resonates with you most?',
      icon: '🏛️',
      component: <TraditionStep 
        value={preferences.spiritualTradition}
        onChange={(value) => setPreferences(prev => ({ ...prev, spiritualTradition: value }))}
      />
    },
    {
      id: 'experience',
      title: 'Experience Level',
      description: 'How would you describe your spiritual journey?',
      icon: '🌱',
      component: <ExperienceStep
        value={preferences.experienceLevel}
        onChange={(value) => setPreferences(prev => ({ ...prev, experienceLevel: value }))}
      />
    },
    {
      id: 'goals',
      title: 'Your Goals',
      description: 'What would you like guidance with?',
      icon: '🎯',
      component: <GoalsStep
        value={preferences.primaryGoals}
        onChange={(value) => setPreferences(prev => ({ ...prev, primaryGoals: value }))}
      />
    },
    {
      id: 'communication',
      title: 'Communication Style',
      description: 'How would you prefer DharmaMind to respond?',
      icon: '💬',
      component: <CommunicationStep
        value={preferences.communicationStyle}
        onChange={(value) => setPreferences(prev => ({ ...prev, communicationStyle: value }))}
      />
    },
    {
      id: 'features',
      title: 'Features & Settings',
      description: 'Customize your experience',
      icon: '⚙️',
      component: <FeaturesStep
        preferences={preferences}
        onChange={(updates) => setPreferences(prev => ({ ...prev, ...updates }))}
      />
    }
  ];

  const currentStepData = steps[currentStep];
  const progress = ((currentStep + 1) / steps.length) * 100;
  const isLastStep = currentStep === steps.length - 1;
  const canProceed = validateCurrentStep();

  function validateCurrentStep(): boolean {
    switch (currentStepData.id) {
      case 'welcome':
        return true;
      case 'tradition':
        return preferences.spiritualTradition !== '';
      case 'experience':
        return preferences.experienceLevel !== '';
      case 'goals':
        return preferences.primaryGoals.length > 0;
      case 'communication':
        return preferences.communicationStyle !== '';
      case 'features':
        return true;
      default:
        return false;
    }
  }

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(prev => prev + 1);
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(prev => prev - 1);
    }
  };

  const handleComplete = async () => {
    setIsCompleting(true);
    try {
      // Save preferences
      if (!isDemo) {
        await updateUserPreferences(preferences);
      }
      
      // Store onboarding completion
      localStorage.setItem('dharmamind-onboarding-complete', 'true');
      localStorage.setItem('dharmamind-preferences', JSON.stringify(preferences));
      
      // Redirect based on mode
      if (isDemo) {
        router.push('/chat?demo=true&welcome=true');
      } else {
        router.push('/chat?welcome=true');
      }
    } catch (error) {
      console.error('Failed to complete onboarding:', error);
      // Continue anyway
      if (isDemo) {
        router.push('/chat?demo=true&welcome=true');
      } else {
        router.push('/chat?welcome=true');
      }
    } finally {
      setIsCompleting(false);
    }
  };

  const handleSkip = () => {
    localStorage.setItem('dharmamind-onboarding-skipped', 'true');
    if (isDemo) {
      router.push('/chat?demo=true');
    } else {
      router.push('/chat');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-stone-100 via-amber-50 to-emerald-50">
      {/* Header */}
      <div className="mobile-header">
        <div className="mobile-header-content">
          <div className="flex items-center">
            <span className="text-2xl mr-3">{currentStepData.icon}</span>
            <div>
              <h1 className="text-lg font-bold text-stone-800">
                {currentStepData.title}
              </h1>
              <p className="text-sm text-stone-600">
                Step {currentStep + 1} of {steps.length}
              </p>
            </div>
          </div>
          <button
            onClick={handleSkip}
            className="text-stone-500 hover:text-stone-700 text-sm font-medium"
          >
            Skip
          </button>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="px-4 py-2 bg-white/80">
        <div className="w-full bg-stone-200 rounded-full h-2">
          <div 
            className="bg-gradient-to-r from-amber-500 to-emerald-500 h-2 rounded-full transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 px-4 py-6">
        <div className="max-w-md mx-auto">
          <div className="text-center mb-8">
            <p className="text-stone-600 leading-relaxed">
              {currentStepData.description}
            </p>
          </div>

          {/* Step Component */}
          <div className="mb-8">
            {currentStepData.component}
          </div>

          {/* Navigation */}
          <div className="flex items-center justify-between gap-4">
            <button
              onClick={handlePrevious}
              disabled={currentStep === 0}
              className="btn btn-outline px-6 py-3 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Previous
            </button>

            {isLastStep ? (
              <button
                onClick={handleComplete}
                disabled={!canProceed || isCompleting}
                className="btn btn-primary px-6 py-3 flex-1 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isCompleting ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                    Completing...
                  </>
                ) : (
                  <>
                    Complete Setup
                    <span className="ml-2">🎉</span>
                  </>
                )}
              </button>
            ) : (
              <button
                onClick={handleNext}
                disabled={!canProceed}
                className="btn btn-primary px-6 py-3 flex-1 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
                <span className="ml-2">→</span>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// Individual Step Components
const WelcomeStep: React.FC = () => (
  <div className="text-center">
    <div className="mb-6">
      <div className="w-24 h-24 bg-gradient-to-br from-amber-100 to-emerald-100 rounded-full flex items-center justify-center mx-auto mb-4">
        <span className="text-4xl">🕉️</span>
      </div>
      <h2 className="text-2xl font-bold text-stone-800 mb-2">
        Welcome to DharmaMind
      </h2>
      <p className="text-stone-600 leading-relaxed">
        Your AI companion for spiritual growth, inner clarity, and wisdom guidance. 
        Let's customize your experience to align with your unique spiritual journey.
      </p>
    </div>
    <div className="bg-gradient-to-r from-amber-50 to-emerald-50 rounded-lg p-4 border border-amber-200">
      <p className="text-sm text-stone-700">
        <strong>✨ What to expect:</strong> Personalized guidance, wisdom from ancient traditions, 
        and AI responses tailored to your spiritual path and communication preferences.
      </p>
    </div>
  </div>
);

interface StepProps {
  value: any;
  onChange: (value: any) => void;
}

const TraditionStep: React.FC<StepProps> = ({ value, onChange }) => {
  const traditions = [
    { id: 'universal', name: 'Universal Wisdom', description: 'All traditions combined', icon: '🌍' },
    { id: 'vedanta', name: 'Vedanta', description: 'Non-dualistic philosophy', icon: '🏛️' },
    { id: 'yoga', name: 'Yoga', description: 'Union of mind, body, spirit', icon: '🧘' },
    { id: 'buddhism', name: 'Buddhism', description: 'Path to enlightenment', icon: '☸️' },
    { id: 'christianity', name: 'Christianity', description: 'Love and service', icon: '✝️' },
    { id: 'islam', name: 'Islam', description: 'Submission and peace', icon: '☪️' },
    { id: 'other', name: 'Other/Multiple', description: 'Mix of traditions', icon: '🤝' }
  ];

  return (
    <div className="space-y-3">
      {traditions.map((tradition) => (
        <button
          key={tradition.id}
          onClick={() => onChange(tradition.id)}
          className={`w-full p-4 rounded-lg border-2 text-left transition-all ${
            value === tradition.id 
              ? 'border-emerald-500 bg-emerald-50' 
              : 'border-stone-200 bg-white hover:border-stone-300'
          }`}
        >
          <div className="flex items-center">
            <span className="text-2xl mr-4">{tradition.icon}</span>
            <div>
              <h3 className="font-medium text-stone-800">{tradition.name}</h3>
              <p className="text-sm text-stone-600">{tradition.description}</p>
            </div>
          </div>
        </button>
      ))}
    </div>
  );
};

const ExperienceStep: React.FC<StepProps> = ({ value, onChange }) => {
  const levels = [
    { 
      id: 'beginner', 
      name: 'Beginner', 
      description: 'New to spiritual practices',
      detail: 'I\'m just starting my spiritual journey'
    },
    { 
      id: 'exploring', 
      name: 'Exploring', 
      description: 'Learning and experimenting',
      detail: 'I\'ve tried some practices and am curious to learn more'
    },
    { 
      id: 'practicing', 
      name: 'Practicing', 
      description: 'Regular spiritual practice',
      detail: 'I have established spiritual practices and seek deeper understanding'
    },
    { 
      id: 'advanced', 
      name: 'Advanced', 
      description: 'Deep experience and study',
      detail: 'I have extensive experience and help guide others'
    }
  ];

  return (
    <div className="space-y-3">
      {levels.map((level) => (
        <button
          key={level.id}
          onClick={() => onChange(level.id)}
          className={`w-full p-4 rounded-lg border-2 text-left transition-all ${
            value === level.id 
              ? 'border-emerald-500 bg-emerald-50' 
              : 'border-stone-200 bg-white hover:border-stone-300'
          }`}
        >
          <h3 className="font-medium text-stone-800 mb-1">{level.name}</h3>
          <p className="text-sm text-stone-600 mb-2">{level.description}</p>
          <p className="text-xs text-stone-500 italic">{level.detail}</p>
        </button>
      ))}
    </div>
  );
};

const GoalsStep: React.FC<StepProps> = ({ value, onChange }) => {
  const goals = [
    { id: 'meditation', name: 'Meditation & Mindfulness', icon: '🧘‍♀️' },
    { id: 'purpose', name: 'Life Purpose & Direction', icon: '🧭' },
    { id: 'relationships', name: 'Relationships & Love', icon: '💝' },
    { id: 'career', name: 'Career & Leadership', icon: '📈' },
    { id: 'healing', name: 'Emotional Healing', icon: '💚' },
    { id: 'wisdom', name: 'Ancient Wisdom & Philosophy', icon: '📚' },
    { id: 'growth', name: 'Personal Growth', icon: '🌱' },
    { id: 'peace', name: 'Inner Peace & Calm', icon: '☮️' }
  ];

  const toggleGoal = (goalId: string) => {
    const currentGoals = value || [];
    if (currentGoals.includes(goalId)) {
      onChange(currentGoals.filter((id: string) => id !== goalId));
    } else {
      onChange([...currentGoals, goalId]);
    }
  };

  return (
    <div>
      <p className="text-sm text-stone-600 mb-4">Select all that interest you:</p>
      <div className="grid grid-cols-2 gap-3">
        {goals.map((goal) => (
          <button
            key={goal.id}
            onClick={() => toggleGoal(goal.id)}
            className={`p-3 rounded-lg border-2 text-center transition-all ${
              value?.includes(goal.id) 
                ? 'border-emerald-500 bg-emerald-50' 
                : 'border-stone-200 bg-white hover:border-stone-300'
            }`}
          >
            <span className="text-2xl mb-2 block">{goal.icon}</span>
            <p className="text-xs font-medium text-stone-800">{goal.name}</p>
          </button>
        ))}
      </div>
    </div>
  );
};

const CommunicationStep: React.FC<StepProps> = ({ value, onChange }) => {
  const styles = [
    {
      id: 'compassionate',
      name: 'Compassionate & Gentle',
      description: 'Warm, understanding, nurturing responses',
      example: '"I understand this is difficult. Let\'s explore this gently together..."'
    },
    {
      id: 'practical',
      name: 'Practical & Direct',
      description: 'Clear, actionable, straightforward guidance',
      example: '"Here are three specific steps you can take today..."'
    },
    {
      id: 'scholarly',
      name: 'Scholarly & Detailed',
      description: 'In-depth, philosophical, well-referenced',
      example: '"According to the Bhagavad Gita, this principle teaches us..."'
    },
    {
      id: 'poetic',
      name: 'Poetic & Inspirational',
      description: 'Metaphorical, beautiful, uplifting language',
      example: '"Like a lotus rising from muddy waters, your challenges..."'
    }
  ];

  return (
    <div className="space-y-3">
      {styles.map((style) => (
        <button
          key={style.id}
          onClick={() => onChange(style.id)}
          className={`w-full p-4 rounded-lg border-2 text-left transition-all ${
            value === style.id 
              ? 'border-emerald-500 bg-emerald-50' 
              : 'border-stone-200 bg-white hover:border-stone-300'
          }`}
        >
          <h3 className="font-medium text-stone-800 mb-1">{style.name}</h3>
          <p className="text-sm text-stone-600 mb-2">{style.description}</p>
          <p className="text-xs text-stone-500 italic">{style.example}</p>
        </button>
      ))}
    </div>
  );
};

interface FeaturesStepProps {
  preferences: UserPreferences;
  onChange: (updates: Partial<UserPreferences>) => void;
}

const FeaturesStep: React.FC<FeaturesStepProps> = ({ preferences, onChange }) => {
  const languages = [
    { code: 'en', name: 'English' },
    { code: 'hi', name: 'हिन्दी (Hindi)' },
    { code: 'sa', name: 'संस्कृत (Sanskrit)' },
    { code: 'es', name: 'Español' },
    { code: 'fr', name: 'Français' },
    { code: 'de', name: 'Deutsch' }
  ];

  return (
    <div className="space-y-6">
      {/* Language Selection */}
      <div>
        <h3 className="font-medium text-stone-800 mb-3">Primary Language</h3>
        <select
          value={preferences.language}
          onChange={(e) => onChange({ language: e.target.value })}
          className="w-full p-3 border border-stone-300 rounded-lg bg-white"
        >
          {languages.map((lang) => (
            <option key={lang.code} value={lang.code}>
              {lang.name}
            </option>
          ))}
        </select>
      </div>

      {/* Feature Toggles */}
      <div className="space-y-4">
        <div className="flex items-center justify-between p-3 bg-white rounded-lg border border-stone-200">
          <div>
            <h4 className="font-medium text-stone-800">Notifications</h4>
            <p className="text-sm text-stone-600">Daily wisdom and reminders</p>
          </div>
          <button
            onClick={() => onChange({ notifications: !preferences.notifications })}
            className={`w-12 h-6 rounded-full transition-colors ${
              preferences.notifications ? 'bg-emerald-500' : 'bg-stone-300'
            }`}
          >
            <div className={`w-5 h-5 bg-white rounded-full transition-transform ${
              preferences.notifications ? 'translate-x-6' : 'translate-x-0.5'
            }`} />
          </button>
        </div>

        <div className="flex items-center justify-between p-3 bg-white rounded-lg border border-stone-200">
          <div>
            <h4 className="font-medium text-stone-800">Voice Input</h4>
            <p className="text-sm text-stone-600">Speak your questions aloud</p>
          </div>
          <button
            onClick={() => onChange({ voiceInput: !preferences.voiceInput })}
            className={`w-12 h-6 rounded-full transition-colors ${
              preferences.voiceInput ? 'bg-emerald-500' : 'bg-stone-300'
            }`}
          >
            <div className={`w-5 h-5 bg-white rounded-full transition-transform ${
              preferences.voiceInput ? 'translate-x-6' : 'translate-x-0.5'
            }`} />
          </button>
        </div>
      </div>

      {/* Summary */}
      <div className="bg-gradient-to-r from-amber-50 to-emerald-50 rounded-lg p-4 border border-amber-200">
        <h4 className="font-medium text-stone-800 mb-2">Your Personalized Setup:</h4>
        <ul className="text-sm text-stone-700 space-y-1">
          <li>• Tradition: {preferences.spiritualTradition}</li>
          <li>• Experience: {preferences.experienceLevel}</li>
          <li>• Goals: {preferences.primaryGoals.length} selected</li>
          <li>• Style: {preferences.communicationStyle}</li>
        </ul>
      </div>
    </div>
  );
};

export default UserOnboarding;
