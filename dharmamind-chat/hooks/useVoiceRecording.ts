import { useState, useRef, useCallback, useEffect } from 'react';

interface UseVoiceRecordingOptions {
    onTranscript: (text: string) => void;
    onError?: (error: string) => void;
    language?: string;
    maxDuration?: number; // in seconds
}

interface UseVoiceRecordingReturn {
    isRecording: boolean;
    recordingTime: number;
    isSupported: boolean;
    supportType: 'native' | 'fallback' | 'none';
    startRecording: () => Promise<void>;
    stopRecording: () => void;
    toggleRecording: () => void;
}

// Check for native Speech Recognition support
const getSpeechRecognition = (): any => {
    if (typeof window === 'undefined') return null;
    return (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
};

// Check for MediaRecorder support (for fallback recording)
const getMediaRecorderSupport = (): boolean => {
    if (typeof window === 'undefined') return false;
    return !!(navigator.mediaDevices?.getUserMedia && window.MediaRecorder);
};

export const useVoiceRecording = ({
    onTranscript,
    onError,
    language = 'en-US',
    maxDuration = 60
}: UseVoiceRecordingOptions): UseVoiceRecordingReturn => {
    const [isRecording, setIsRecording] = useState(false);
    const [recordingTime, setRecordingTime] = useState(0);

    const recognitionRef = useRef<any>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const streamRef = useRef<MediaStream | null>(null);
    const timerRef = useRef<NodeJS.Timeout | null>(null);
    const maxDurationTimerRef = useRef<NodeJS.Timeout | null>(null);

    // Determine support type
    const SpeechRecognition = getSpeechRecognition();
    const hasNativeSupport = !!SpeechRecognition;
    const hasFallbackSupport = getMediaRecorderSupport();

    const supportType: 'native' | 'fallback' | 'none' =
        hasNativeSupport ? 'native' :
            hasFallbackSupport ? 'fallback' : 'none';

    const isSupported = supportType !== 'none';

    // Start timer
    const startTimer = useCallback(() => {
        setRecordingTime(0);
        timerRef.current = setInterval(() => {
            setRecordingTime(prev => prev + 1);
        }, 1000);
    }, []);

    // Stop timer
    const stopTimer = useCallback(() => {
        if (timerRef.current) {
            clearInterval(timerRef.current);
            timerRef.current = null;
        }
        if (maxDurationTimerRef.current) {
            clearTimeout(maxDurationTimerRef.current);
            maxDurationTimerRef.current = null;
        }
        setRecordingTime(0);
    }, []);

    // Cleanup function
    const cleanup = useCallback(() => {
        if (recognitionRef.current) {
            try {
                recognitionRef.current.stop();
            } catch (e) { }
            recognitionRef.current = null;
        }
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            try {
                mediaRecorderRef.current.stop();
            } catch (e) { }
        }
        mediaRecorderRef.current = null;
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
        audioChunksRef.current = [];
        stopTimer();
        setIsRecording(false);
    }, [stopTimer]);

    // Native Speech Recognition (Chrome, Edge, Safari)
    const startNativeRecording = useCallback(async () => {
        try {
            // Request microphone permission first
            await navigator.mediaDevices.getUserMedia({ audio: true });

            const recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = language;
            recognition.maxAlternatives = 1;

            let finalTranscript = '';

            recognition.onstart = () => {
                setIsRecording(true);
                startTimer();

                // Auto-stop after max duration
                maxDurationTimerRef.current = setTimeout(() => {
                    if (recognitionRef.current) {
                        recognition.stop();
                    }
                }, maxDuration * 1000);
            };

            recognition.onresult = (event: any) => {
                let interimTranscript = '';

                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const result = event.results[i];
                    if (result.isFinal) {
                        finalTranscript += result[0].transcript + ' ';
                    } else {
                        interimTranscript += result[0].transcript;
                    }
                }

                // Send the transcript (final + interim for real-time feedback)
                const currentText = (finalTranscript + interimTranscript).trim();
                if (currentText) {
                    onTranscript(currentText);
                }
            };

            recognition.onerror = (event: any) => {
                console.error('Speech recognition error:', event.error);
                cleanup();

                const errorMessages: Record<string, string> = {
                    'no-speech': 'No speech detected. Please try again.',
                    'audio-capture': 'Microphone not found. Please check your device.',
                    'not-allowed': 'Microphone access denied. Please allow microphone access.',
                    'network': 'Network error. Please check your connection.',
                    'aborted': 'Recording was cancelled.',
                };

                onError?.(errorMessages[event.error] || 'Voice recognition error. Please try again.');
            };

            recognition.onend = () => {
                cleanup();
            };

            recognitionRef.current = recognition;
            recognition.start();
        } catch (error: any) {
            console.error('Error starting native recording:', error);
            cleanup();
            onError?.('Could not access microphone. Please check your permissions.');
        }
    }, [SpeechRecognition, language, maxDuration, onTranscript, onError, startTimer, cleanup]);

    // Fallback: Record audio and send to transcription API (Firefox, older browsers)
    const startFallbackRecording = useCallback(async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 16000
                }
            });
            streamRef.current = stream;

            // Determine best supported format
            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : MediaRecorder.isTypeSupported('audio/webm')
                    ? 'audio/webm'
                    : MediaRecorder.isTypeSupported('audio/mp4')
                        ? 'audio/mp4'
                        : 'audio/wav';

            const mediaRecorder = new MediaRecorder(stream, { mimeType });
            mediaRecorderRef.current = mediaRecorder;
            audioChunksRef.current = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });

                // Send to backend for transcription
                try {
                    const formData = new FormData();
                    formData.append('audio', audioBlob, `recording.${mimeType.includes('webm') ? 'webm' : 'mp4'}`);
                    formData.append('language', language);

                    const response = await fetch('/api/transcribe', {
                        method: 'POST',
                        body: formData
                    });

                    if (response.ok) {
                        const data = await response.json();
                        if (data.transcript) {
                            onTranscript(data.transcript);
                        } else {
                            onError?.('Could not transcribe audio. Please try again.');
                        }
                    } else {
                        throw new Error('Transcription failed');
                    }
                } catch (error) {
                    console.error('Transcription error:', error);
                    onError?.('Could not transcribe audio. Please try again.');
                }

                cleanup();
            };

            mediaRecorder.onerror = () => {
                cleanup();
                onError?.('Recording error. Please try again.');
            };

            setIsRecording(true);
            startTimer();
            mediaRecorder.start(1000); // Collect data every second

            // Auto-stop after max duration
            maxDurationTimerRef.current = setTimeout(() => {
                if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
                    mediaRecorderRef.current.stop();
                }
            }, maxDuration * 1000);

        } catch (error: any) {
            console.error('Error starting fallback recording:', error);
            cleanup();

            if (error.name === 'NotAllowedError') {
                onError?.('Microphone access denied. Please allow microphone access.');
            } else if (error.name === 'NotFoundError') {
                onError?.('No microphone found. Please connect a microphone.');
            } else {
                onError?.('Could not start recording. Please try again.');
            }
        }
    }, [language, maxDuration, onTranscript, onError, startTimer, cleanup]);

    // Start recording (auto-selects method)
    const startRecording = useCallback(async () => {
        if (isRecording) return;

        if (!isSupported) {
            onError?.('Voice input is not supported in your browser. Please use a modern browser.');
            return;
        }

        if (supportType === 'native') {
            await startNativeRecording();
        } else {
            await startFallbackRecording();
        }
    }, [isRecording, isSupported, supportType, startNativeRecording, startFallbackRecording, onError]);

    // Stop recording
    const stopRecording = useCallback(() => {
        if (recognitionRef.current) {
            try {
                recognitionRef.current.stop();
            } catch (e) { }
        }
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            mediaRecorderRef.current.stop();
        } else {
            cleanup();
        }
    }, [cleanup]);

    // Toggle recording
    const toggleRecording = useCallback(() => {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    }, [isRecording, startRecording, stopRecording]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            cleanup();
        };
    }, [cleanup]);

    return {
        isRecording,
        recordingTime,
        isSupported,
        supportType,
        startRecording,
        stopRecording,
        toggleRecording
    };
};

export default useVoiceRecording;
