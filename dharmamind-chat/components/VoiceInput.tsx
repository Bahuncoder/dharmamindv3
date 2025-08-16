import React, { useState, useRef, useEffect } from 'react';

interface VoiceInputProps {
  onTranscript: (transcript: string) => void;
  onError?: (error: string) => void;
  className?: string;
  disabled?: boolean;
  language?: string;
  spiritualMode?: boolean;
  showVisualFeedback?: boolean;
}

type RecognitionState = 'idle' | 'listening' | 'processing' | 'error';

interface SpeechRecognitionResult {
  [key: number]: {
    transcript: string;
    confidence: number;
  };
  isFinal: boolean;
  length: number;
}

interface SpeechRecognitionResultList {
  [key: number]: SpeechRecognitionResult;
  length: number;
}

interface SpeechRecognitionEvent {
  results: SpeechRecognitionResultList;
  resultIndex: number;
}

interface SpeechRecognitionErrorEvent {
  error: string;
  message: string;
}

declare global {
  interface Window {
    SpeechRecognition: any;
    webkitSpeechRecognition: any;
  }
}

const VoiceInput: React.FC<VoiceInputProps> = ({
  onTranscript,
  onError,
  className = '',
  disabled = false,
  language = 'en-US',
  spiritualMode = false,
  showVisualFeedback = true
}) => {
  const [state, setState] = useState<RecognitionState>('idle');
  const [transcript, setTranscript] = useState('');
  const [confidence, setConfidence] = useState(0);
  const recognitionRef = useRef<any>(null);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Check for browser support
  const isSupported = typeof window !== 'undefined' && 
    (window.SpeechRecognition || window.webkitSpeechRecognition);

  useEffect(() => {
    if (!isSupported) {
      return;
    }

    // Initialize speech recognition
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();

    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = language;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      setState('listening');
      setTranscript('');
      setConfidence(0);
      
      // Set timeout for auto-stop
      timeoutRef.current = setTimeout(() => {
        if (recognitionRef.current) {
          recognitionRef.current.stop();
        }
      }, 30000); // 30 seconds max
    };

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let finalTranscript = '';
      let interimTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          finalTranscript += result[0].transcript;
          setConfidence(result[0].confidence);
        } else {
          interimTranscript += result[0].transcript;
        }
      }

      const currentTranscript = finalTranscript || interimTranscript;
      setTranscript(currentTranscript);

      if (finalTranscript) {
        setState('processing');
        onTranscript(finalTranscript.trim());
        
        // Clear timeout since we got a result
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current);
          timeoutRef.current = null;
        }
      }
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      setState('error');
      const errorMessage = getErrorMessage(event.error);
      
      if (onError) {
        onError(errorMessage);
      }
      
      // Clear timeout on error
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
      
      // Reset to idle after a delay
      setTimeout(() => {
        setState('idle');
      }, 2000);
    };

    recognition.onend = () => {
      if (state === 'listening') {
        setState('idle');
      }
      
      // Clear timeout when recognition ends
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };

    recognitionRef.current = recognition;

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.abort();
      }
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [isSupported, language, onTranscript, onError, state]);

  const getErrorMessage = (error: string): string => {
    switch (error) {
      case 'no-speech':
        return 'No speech detected. Please try again.';
      case 'audio-capture':
        return 'Audio capture failed. Check your microphone.';
      case 'not-allowed':
        return 'Microphone access denied. Please allow microphone access.';
      case 'network':
        return 'Network error. Please check your connection.';
      case 'service-not-allowed':
        return 'Speech recognition service not available.';
      case 'bad-grammar':
        return 'Speech recognition grammar error.';
      case 'language-not-supported':
        return 'Selected language not supported.';
      default:
        return 'Speech recognition error. Please try again.';
    }
  };

  const startListening = () => {
    if (!isSupported || disabled || state !== 'idle') {
      return;
    }

    // Request microphone permission
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ audio: true })
        .then(() => {
          recognitionRef.current?.start();
        })
        .catch((error) => {
          setState('error');
          if (onError) {
            onError('Microphone access denied. Please allow microphone access.');
          }
          setTimeout(() => setState('idle'), 2000);
        });
    } else {
      recognitionRef.current?.start();
    }
  };

  const stopListening = () => {
    if (recognitionRef.current && state === 'listening') {
      recognitionRef.current.stop();
    }
  };

  const getButtonIcon = () => {
    if (spiritualMode) {
      switch (state) {
        case 'listening':
          return '🕉️';
        case 'processing':
          return '🧘‍♂️';
        case 'error':
          return '⚠️';
        default:
          return '🎙️';
      }
    }
    
    switch (state) {
      case 'listening':
        return '🎤';
      case 'processing':
        return '⏳';
      case 'error':
        return '❌';
      default:
        return '🎙️';
    }
  };

  const getButtonTitle = () => {
    if (spiritualMode) {
      switch (state) {
        case 'listening':
          return 'Speaking with mindfulness... Click to pause';
        case 'processing':
          return 'Contemplating your words...';
        case 'error':
          return 'Technical difficulty. Click to try again';
        default:
          return 'Share your thoughts with voice';
      }
    }
    
    switch (state) {
      case 'listening':
        return 'Listening... Click to stop';
      case 'processing':
        return 'Processing speech...';
      case 'error':
        return 'Error occurred. Click to try again';
      default:
        return 'Click to start voice input';
    }
  };

  const getButtonClass = () => {
    const baseClass = `voice-input-btn ${className}`;
    
    switch (state) {
      case 'listening':
        return `${baseClass} recording`;
      case 'processing':
        return `${baseClass} processing`;
      case 'error':
        return `${baseClass} error`;
      default:
        return baseClass;
    }
  };

  if (!isSupported) {
    return null; // Don't render if not supported
  }

  return (
    <div className="voice-input-container">
      <button
        type="button"
        className={getButtonClass()}
        onClick={state === 'listening' ? stopListening : startListening}
        disabled={disabled || state === 'processing'}
        title={getButtonTitle()}
        aria-label={getButtonTitle()}
      >
        <span className="voice-icon" role="img" aria-hidden="true">
          {getButtonIcon()}
        </span>
      </button>
      
      {/* Live transcript display (optional) */}
      {transcript && state === 'listening' && (
        <div className="voice-transcript">
          <div className="voice-transcript-content">
            <span className="voice-transcript-text">{transcript}</span>
            {confidence > 0 && (
              <span className="voice-confidence">
                ({Math.round(confidence * 100)}% confident)
              </span>
            )}
          </div>
        </div>
      )}
      
      {/* Visual feedback for listening state */}
      {state === 'listening' && (
        <div className="voice-listening-indicator">
          <div className="voice-wave">
            <div className="voice-wave-bar"></div>
            <div className="voice-wave-bar"></div>
            <div className="voice-wave-bar"></div>
            <div className="voice-wave-bar"></div>
          </div>
          <p className="voice-status-text">Listening for your voice...</p>
        </div>
      )}
      
      <style jsx>{`
        .voice-input-container {
          position: relative;
        }
        
        .voice-transcript {
          position: absolute;
          bottom: 60px;
          left: 50%;
          transform: translateX(-50%);
          background: rgba(0, 0, 0, 0.8);
          color: white;
          padding: 8px 12px;
          border-radius: 8px;
          font-size: 0.9rem;
          max-width: 200px;
          z-index: 1001;
        }
        
        .voice-transcript-content {
          text-align: center;
        }
        
        .voice-confidence {
          display: block;
          font-size: 0.7rem;
          opacity: 0.7;
          margin-top: 2px;
        }
        
        .voice-listening-indicator {
          position: fixed;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          background: rgba(0, 0, 0, 0.9);
          color: white;
          padding: 20px;
          border-radius: 16px;
          text-align: center;
          z-index: 1002;
          backdrop-filter: blur(10px);
        }
        
        .voice-wave {
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 3px;
          margin-bottom: 12px;
        }
        
        .voice-wave-bar {
          width: 3px;
          height: 20px;
          background: #48bb78;
          border-radius: 2px;
          animation: voice-wave-animation 1.2s ease-in-out infinite;
        }
        
        .voice-wave-bar:nth-child(2) {
          animation-delay: 0.1s;
        }
        
        .voice-wave-bar:nth-child(3) {
          animation-delay: 0.2s;
        }
        
        .voice-wave-bar:nth-child(4) {
          animation-delay: 0.3s;
        }
        
        .voice-status-text {
          margin: 0;
          font-size: 0.9rem;
          color: #48bb78;
        }
        
        @keyframes voice-wave-animation {
          0%, 40%, 100% {
            transform: scaleY(0.4);
          }
          20% {
            transform: scaleY(1.0);
          }
        }
        
        @media (max-width: 768px) {
          .voice-transcript {
            max-width: 90vw;
            left: 50%;
            transform: translateX(-50%);
          }
          
          .voice-listening-indicator {
            max-width: 80vw;
            padding: 16px;
          }
        }
      `}</style>
    </div>
  );
};

export default VoiceInput;
