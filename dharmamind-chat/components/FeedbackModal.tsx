import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface FeedbackModalProps {
  isOpen: boolean;
  onClose: () => void;
  conversationId?: string;
  messageId?: string;
}

interface FeedbackData {
  feedback_type: string;
  title: string;
  content: string;
  overall_rating?: number;
  response_quality?: number;
  helpfulness?: number;
  spiritual_value?: number;
  user_email?: string;
  allow_contact: boolean;
  share_anonymously: boolean;
}

const FeedbackModal: React.FC<FeedbackModalProps> = ({ 
  isOpen, 
  onClose, 
  conversationId, 
  messageId 
}) => {
  const [step, setStep] = useState(1);
  const [feedbackData, setFeedbackData] = useState<FeedbackData>({
    feedback_type: 'general',
    title: '',
    content: '',
    allow_contact: false,
    share_anonymously: true
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const feedbackTypes = [
    { value: 'general', label: '💬 General Feedback', description: 'Share your overall experience' },
    { value: 'bug_report', label: '🐛 Bug Report', description: 'Report technical issues' },
    { value: 'feature_request', label: '✨ Feature Request', description: 'Suggest new features' },
    { value: 'content_quality', label: '📝 Content Quality', description: 'Feedback on response quality' },
    { value: 'performance', label: '⚡ Performance', description: 'Speed and responsiveness issues' },
    { value: 'dharmic_concern', label: '🕉️ Dharmic Concern', description: 'Spiritual or ethical concerns' },
    { value: 'spiritual_guidance', label: '🙏 Spiritual Guidance', description: 'Feedback on spiritual content' },
    { value: 'user_experience', label: '👤 User Experience', description: 'Interface and usability feedback' }
  ];

  const handleInputChange = (field: keyof FeedbackData, value: any) => {
    setFeedbackData(prev => ({ ...prev, [field]: value }));
  };

  const handleRatingChange = (field: keyof FeedbackData, rating: number) => {
    setFeedbackData(prev => ({ ...prev, [field]: rating }));
  };

  const handleSubmit = async () => {
    if (!feedbackData.title.trim() || !feedbackData.content.trim()) {
      alert('Please fill in both title and content fields.');
      return;
    }

    setIsSubmitting(true);
    
    try {
      const payload = {
        ...feedbackData,
        conversation_id: conversationId,
        message_id: messageId,
        browser_info: navigator.userAgent,
        device_info: `${navigator.platform} - ${window.screen.width}x${window.screen.height}`
      };

      const response = await fetch('/api/v1/feedback/submit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      });

      if (response.ok) {
        setSubmitted(true);
        setTimeout(() => {
          onClose();
          setSubmitted(false);
          setStep(1);
          setFeedbackData({
            feedback_type: 'general',
            title: '',
            content: '',
            allow_contact: false,
            share_anonymously: true
          });
        }, 2000);
      } else {
        throw new Error('Failed to submit feedback');
      }
    } catch (error) {
      console.error('Error submitting feedback:', error);
      alert('Failed to submit feedback. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const nextStep = () => {
    if (step < 3) setStep(step + 1);
  };

  const prevStep = () => {
    if (step > 1) setStep(step - 1);
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-stone-900/50 backdrop-blur-sm flex items-center justify-center z-50 p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto"
          onClick={e => e.stopPropagation()}
        >
          {submitted ? (
            <div className="p-8 text-center">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="w-16 h-16 bg-gradient-to-r from-emerald-500 to-emerald-600 rounded-full flex items-center justify-center mx-auto mb-4"
              >
                <span className="text-2xl">✓</span>
              </motion.div>
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                Thank You! 🙏
              </h3>
              <p className="text-gray-600 dark:text-gray-300 mb-4">
                Your feedback has been received and will help us improve DharmaMind for everyone.
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                We truly appreciate your contribution to our spiritual AI journey.
              </p>
            </div>
          ) : (
            <>
              {/* Header */}
              <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
                <div>
                  <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                    Share Your Feedback
                  </h2>
                  <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                    Help us improve your DharmaMind experience
                  </p>
                </div>
                <button
                  onClick={onClose}
                  className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              {/* Progress Bar */}
              <div className="px-6 py-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Step {step} of 3
                  </span>
                  <span className="text-sm text-gray-500 dark:text-gray-400">
                    {step === 1 ? 'Type & Content' : step === 2 ? 'Ratings' : 'Details'}
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-gradient-to-r from-gray-500 to-emerald-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${(step / 3) * 100}%` }}
                  />
                </div>
              </div>

              <div className="p-6">
                {/* Step 1: Feedback Type and Content */}
                {step === 1 && (
                  <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="space-y-6"
                  >
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                        What type of feedback would you like to share?
                      </label>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {feedbackTypes.map((type) => (
                          <button
                            key={type.value}
                            onClick={() => handleInputChange('feedback_type', type.value)}
                            className={`text-left p-3 rounded-lg border-2 transition-all ${
                              feedbackData.feedback_type === type.value
                                ? 'border-emerald-500 bg-emerald-50 dark:bg-emerald-900/20'
                                : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                            }`}
                          >
                            <div className="font-medium text-sm text-gray-900 dark:text-white">
                              {type.label}
                            </div>
                            <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                              {type.description}
                            </div>
                          </button>
                        ))}
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Title *
                      </label>
                      <input
                        type="text"
                        value={feedbackData.title}
                        onChange={(e) => handleInputChange('title', e.target.value)}
                        placeholder="Brief summary of your feedback"
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 dark:bg-gray-700 dark:text-white"
                        maxLength={200}
                      />
                      <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        {feedbackData.title.length}/200 characters
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Details *
                      </label>
                      <textarea
                        value={feedbackData.content}
                        onChange={(e) => handleInputChange('content', e.target.value)}
                        placeholder="Please share your detailed feedback, suggestions, or concerns..."
                        rows={4}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 dark:bg-gray-700 dark:text-white resize-none"
                        maxLength={4000}
                      />
                      <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        {feedbackData.content.length}/4000 characters
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* Step 2: Ratings */}
                {step === 2 && (
                  <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="space-y-6"
                  >
                    <div className="text-center mb-6">
                      <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                        Rate Your Experience
                      </h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        Help us understand how we're doing (optional)
                      </p>
                    </div>

                    {[
                      { key: 'overall_rating' as keyof FeedbackData, label: 'Overall Experience', icon: '⭐' },
                      { key: 'response_quality' as keyof FeedbackData, label: 'Response Quality', icon: '💬' },
                      { key: 'helpfulness' as keyof FeedbackData, label: 'Helpfulness', icon: '🤝' },
                      { key: 'spiritual_value' as keyof FeedbackData, label: 'Spiritual Value', icon: '🕉️' }
                    ].map((rating) => (
                      <div key={rating.key} className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <span className="text-xl">{rating.icon}</span>
                          <span className="font-medium text-gray-700 dark:text-gray-300">
                            {rating.label}
                          </span>
                        </div>
                        <div className="flex space-x-2">
                          {[1, 2, 3, 4, 5].map((star) => (
                            <button
                              key={star}
                              onClick={() => handleRatingChange(rating.key, star)}
                              className={`w-8 h-8 rounded-full transition-colors ${
                                (feedbackData[rating.key] as number) >= star
                                  ? 'bg-gradient-to-r from-gray-400 to-gray-500 text-white'
                                  : 'bg-gray-200 dark:bg-gray-700 text-gray-400 hover:bg-gray-300 dark:hover:bg-gray-600'
                              }`}
                            >
                              ★
                            </button>
                          ))}
                        </div>
                      </div>
                    ))}
                  </motion.div>
                )}

                {/* Step 3: Contact Details */}
                {step === 3 && (
                  <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="space-y-6"
                  >
                    <div className="text-center mb-6">
                      <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                        Contact & Privacy
                      </h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        Let us know how you'd like us to handle your feedback
                      </p>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Email Address (Optional)
                      </label>
                      <input
                        type="email"
                        value={feedbackData.user_email || ''}
                        onChange={(e) => handleInputChange('user_email', e.target.value)}
                        placeholder="your.email@example.com"
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 dark:bg-gray-700 dark:text-white"
                      />
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        Only needed if you want us to follow up with you
                      </p>
                    </div>

                    <div className="space-y-4">
                      <label className="flex items-start space-x-3">
                        <input
                          type="checkbox"
                          checked={feedbackData.allow_contact}
                          onChange={(e) => handleInputChange('allow_contact', e.target.checked)}
                          className="mt-1 rounded border-gray-300 text-emerald-600 focus:ring-emerald-500"
                        />
                        <div>
                          <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
                            Allow follow-up contact
                          </div>
                          <div className="text-xs text-gray-500 dark:text-gray-400">
                            We may reach out to clarify your feedback or inform you of improvements
                          </div>
                        </div>
                      </label>

                      <label className="flex items-start space-x-3">
                        <input
                          type="checkbox"
                          checked={feedbackData.share_anonymously}
                          onChange={(e) => handleInputChange('share_anonymously', e.target.checked)}
                          className="mt-1 rounded border-gray-300 text-emerald-600 focus:ring-emerald-500"
                        />
                        <div>
                          <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
                            Share anonymously for improvements
                          </div>
                          <div className="text-xs text-gray-500 dark:text-gray-400">
                            Help us improve DharmaMind by sharing your feedback anonymously with our team
                          </div>
                        </div>
                      </label>
                    </div>
                  </motion.div>
                )}
              </div>

              {/* Footer */}
              <div className="flex items-center justify-between p-6 border-t border-gray-200 dark:border-gray-700">
                <button
                  onClick={step === 1 ? onClose : prevStep}
                  className="px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 font-medium transition-colors"
                >
                  {step === 1 ? 'Cancel' : 'Previous'}
                </button>

                <div className="flex space-x-3">
                  {step < 3 ? (
                    <button
                      onClick={nextStep}
                      disabled={step === 1 && (!feedbackData.title.trim() || !feedbackData.content.trim())}
                      className="px-6 py-2 bg-gradient-to-r from-gray-500 to-emerald-500 text-white font-medium rounded-lg hover:from-gray-600 hover:to-emerald-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    >
                      Next
                    </button>
                  ) : (
                    <button
                      onClick={handleSubmit}
                      disabled={isSubmitting}
                      className="px-6 py-2 bg-gradient-to-r from-gray-500 to-emerald-500 text-white font-medium rounded-lg hover:from-gray-600 hover:to-emerald-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center space-x-2"
                    >
                      {isSubmitting ? (
                        <>
                          <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                          <span>Submitting...</span>
                        </>
                      ) : (
                        <span>Submit Feedback</span>
                      )}
                    </button>
                  )}
                </div>
              </div>
            </>
          )}
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default FeedbackModal;
