import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useColors } from '../contexts/ColorContext';

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
        className="fixed inset-0 bg-page-background/80 backdrop-blur-sm flex items-center justify-center z-50 p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="card-background border border-card-border rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto"
          onClick={e => e.stopPropagation()}
        >
          {submitted ? (
            <div className="p-8 text-center">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="w-16 h-16 btn-primary rounded-full flex items-center justify-center mx-auto mb-4"
              >
                <span className="text-2xl">✓</span>
              </motion.div>
<<<<<<< HEAD
              <h3 className="text-2xl font-bold text-neutral-900 mb-2">
                Thank You! 🙏
              </h3>
              <p className="text-neutral-600 mb-4">
                Your feedback has been received and will help us improve DharmaMind for everyone.
              </p>
              <p className="text-sm text-neutral-600">
=======
              <h3 className="text-2xl font-bold text-primary mb-2">
                Thank You! 🙏
              </h3>
              <p className="text-secondary mb-4">
                Your feedback has been received and will help us improve DharmaMind for everyone.
              </p>
              <p className="text-sm text-secondary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                We truly appreciate your contribution to our spiritual AI journey.
              </p>
            </div>
          ) : (
            <>
              {/* Header */}
              <div className="flex items-center justify-between p-6 border-b border-card-border">
                <div>
<<<<<<< HEAD
                  <h2 className="text-xl font-semibold text-neutral-900">
                    Share Your Feedback
                  </h2>
                  <p className="text-sm text-neutral-600 mt-1">
=======
                  <h2 className="text-xl font-semibold text-primary">
                    Share Your Feedback
                  </h2>
                  <p className="text-sm text-secondary mt-1">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    Help us improve your DharmaMind experience
                  </p>
                </div>
                <button
                  onClick={onClose}
<<<<<<< HEAD
                  className="text-neutral-600 hover:text-gold-600 transition-colors"
=======
                  className="text-secondary hover:text-primary transition-colors"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              {/* Progress Bar */}
              <div className="px-6 py-4">
                <div className="flex items-center justify-between mb-2">
<<<<<<< HEAD
                  <span className="text-sm font-medium text-neutral-900">
                    Step {step} of 3
                  </span>
                  <span className="text-sm text-neutral-600">
=======
                  <span className="text-sm font-medium text-primary">
                    Step {step} of 3
                  </span>
                  <span className="text-sm text-secondary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    {step === 1 ? 'Type & Content' : step === 2 ? 'Ratings' : 'Details'}
                  </span>
                </div>
                <div className="w-full bg-card-border rounded-full h-2">
                  <div 
                    className="btn-primary h-2 rounded-full transition-all duration-300"
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
<<<<<<< HEAD
                      <label className="block text-sm font-medium text-neutral-900 mb-3">
=======
                      <label className="block text-sm font-medium text-primary mb-3">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                        What type of feedback would you like to share?
                      </label>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {feedbackTypes.map((type) => (
                          <button
                            key={type.value}
                            onClick={() => handleInputChange('feedback_type', type.value)}
                            className={`text-left p-3 rounded-lg border-2 transition-all ${
                              feedbackData.feedback_type === type.value
                                ? 'border-primary bg-primary/10'
                                : 'border-card-border hover:border-primary/50'
                            }`}
                          >
<<<<<<< HEAD
                            <div className="font-medium text-sm text-neutral-900">
                              {type.label}
                            </div>
                            <div className="text-xs text-neutral-600 mt-1">
=======
                            <div className="font-medium text-sm text-primary">
                              {type.label}
                            </div>
                            <div className="text-xs text-secondary mt-1">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                              {type.description}
                            </div>
                          </button>
                        ))}
                      </div>
                    </div>

                    <div>
<<<<<<< HEAD
                      <label className="block text-sm font-medium text-neutral-900 mb-2">
=======
                      <label className="block text-sm font-medium text-primary mb-2">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                        Title *
                      </label>
                      <input
                        type="text"
                        value={feedbackData.title}
                        onChange={(e) => handleInputChange('title', e.target.value)}
                        placeholder="Brief summary of your feedback"
                        className="input-primary"
                        maxLength={200}
                      />
<<<<<<< HEAD
                      <div className="text-xs text-neutral-600 mt-1">
=======
                      <div className="text-xs text-secondary mt-1">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                        {feedbackData.title.length}/200 characters
                      </div>
                    </div>

                    <div>
<<<<<<< HEAD
                      <label className="block text-sm font-medium text-neutral-900 mb-2">
=======
                      <label className="block text-sm font-medium text-primary mb-2">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                        Details *
                      </label>
                      <textarea
                        value={feedbackData.content}
                        onChange={(e) => handleInputChange('content', e.target.value)}
                        placeholder="Please share your detailed feedback, suggestions, or concerns..."
                        rows={4}
                        className="input-primary resize-none"
                        maxLength={4000}
                      />
<<<<<<< HEAD
                      <div className="text-xs text-neutral-600 mt-1">
=======
                      <div className="text-xs text-secondary mt-1">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
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
<<<<<<< HEAD
                      <h3 className="text-lg font-medium text-neutral-900 mb-2">
                        Rate Your Experience
                      </h3>
                      <p className="text-sm text-neutral-600">
=======
                      <h3 className="text-lg font-medium text-primary mb-2">
                        Rate Your Experience
                      </h3>
                      <p className="text-sm text-secondary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
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
<<<<<<< HEAD
                          <span className="font-medium text-neutral-900">
=======
                          <span className="font-medium text-primary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
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
                                  ? 'btn-primary text-white'
<<<<<<< HEAD
                                  : 'card-background text-neutral-600 hover:bg-primary hover:text-white'
=======
                                  : 'card-background text-secondary hover:bg-primary hover:text-white'
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
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
<<<<<<< HEAD
                      <h3 className="text-lg font-medium text-neutral-900 mb-2">
                        Contact & Privacy
                      </h3>
                      <p className="text-sm text-neutral-600">
=======
                      <h3 className="text-lg font-medium text-primary mb-2">
                        Contact & Privacy
                      </h3>
                      <p className="text-sm text-secondary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                        Let us know how you'd like us to handle your feedback
                      </p>
                    </div>

                    <div>
<<<<<<< HEAD
                      <label className="block text-sm font-medium text-neutral-900 mb-2">
=======
                      <label className="block text-sm font-medium text-primary mb-2">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                        Email Address (Optional)
                      </label>
                      <input
                        type="email"
                        value={feedbackData.user_email || ''}
                        onChange={(e) => handleInputChange('user_email', e.target.value)}
                        placeholder="your.email@example.com"
                        className="input-primary"
                      />
<<<<<<< HEAD
                      <p className="text-xs text-neutral-600 mt-1">
=======
                      <p className="text-xs text-secondary mt-1">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                        Only needed if you want us to follow up with you
                      </p>
                    </div>

                    <div className="space-y-4">
                      <label className="flex items-start space-x-3">
                        <input
                          type="checkbox"
                          checked={feedbackData.allow_contact}
                          onChange={(e) => handleInputChange('allow_contact', e.target.checked)}
<<<<<<< HEAD
                          className="mt-1 rounded border-card-border text-neutral-900 focus:ring-primary"
                        />
                        <div>
                          <div className="text-sm font-medium text-neutral-900">
                            Allow follow-up contact
                          </div>
                          <div className="text-xs text-neutral-600">
=======
                          className="mt-1 rounded border-card-border text-primary focus:ring-primary"
                        />
                        <div>
                          <div className="text-sm font-medium text-primary">
                            Allow follow-up contact
                          </div>
                          <div className="text-xs text-secondary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                            We may reach out to clarify your feedback or inform you of improvements
                          </div>
                        </div>
                      </label>

                      <label className="flex items-start space-x-3">
                        <input
                          type="checkbox"
                          checked={feedbackData.share_anonymously}
                          onChange={(e) => handleInputChange('share_anonymously', e.target.checked)}
<<<<<<< HEAD
                          className="mt-1 rounded border-card-border text-neutral-900 focus:ring-primary"
                        />
                        <div>
                          <div className="text-sm font-medium text-neutral-900">
                            Share anonymously for improvements
                          </div>
                          <div className="text-xs text-neutral-600">
=======
                          className="mt-1 rounded border-card-border text-primary focus:ring-primary"
                        />
                        <div>
                          <div className="text-sm font-medium text-primary">
                            Share anonymously for improvements
                          </div>
                          <div className="text-xs text-secondary">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                            Help us improve DharmaMind by sharing your feedback anonymously with our team
                          </div>
                        </div>
                      </label>
                    </div>
                  </motion.div>
                )}
              </div>

              {/* Footer */}
              <div className="flex items-center justify-between p-6 border-t border-card-border">
                <button
                  onClick={step === 1 ? onClose : prevStep}
<<<<<<< HEAD
                  className="px-4 py-2 text-neutral-600 hover:text-gold-600 font-medium transition-colors"
=======
                  className="px-4 py-2 text-secondary hover:text-primary font-medium transition-colors"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                >
                  {step === 1 ? 'Cancel' : 'Previous'}
                </button>

                <div className="flex space-x-3">
                  {step < 3 ? (
                    <button
                      onClick={nextStep}
                      disabled={step === 1 && (!feedbackData.title.trim() || !feedbackData.content.trim())}
                      className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Next
                    </button>
                  ) : (
                    <button
                      onClick={handleSubmit}
                      disabled={isSubmitting}
                      className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
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
