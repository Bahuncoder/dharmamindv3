import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  Play, 
  Pause, 
  Square, 
  Heart, 
  Lightbulb, 
  Clock, 
  BookOpen,
  RotateCcw,
  CheckCircle
} from 'lucide-react';

interface ContemplationSession {
  session_id: string;
  practice_type: string;
  tradition: string;
  depth_level: string;
  duration_minutes: number;
  guidance_text: string;
  sanskrit_wisdom?: string;
  reflection_prompts: string[];
  mantras: string[];
  created_at: string;
}

interface ContemplationGuidance {
  guidance: {
    instruction: string;
    encouragement: string;
    technique: string;
  };
  next_prompt: string;
  mantra_suggestion: string;
  time_remaining: number;
  depth_assessment: string;
}

interface DeepContemplationProps {
  onInsightCapture?: (insight: string) => void;
  onSessionComplete?: (summary: any) => void;
}

const DeepContemplationInterface: React.FC<DeepContemplationProps> = ({
  onInsightCapture,
  onSessionComplete
}) => {
  const [session, setSession] = useState<ContemplationSession | null>(null);
  const [isActive, setIsActive] = useState(false);
  const [timeElapsed, setTimeElapsed] = useState(0);
  const [currentGuidance, setCurrentGuidance] = useState<ContemplationGuidance | null>(null);
  const [insights, setInsights] = useState<string[]>([]);
  const [currentInsight, setCurrentInsight] = useState('');
  const [currentState, setCurrentState] = useState('peaceful');
  const [practiceTypes] = useState([
    { value: 'breath_awareness', label: 'Breath Awareness', icon: '🌬️', description: 'Mindful awareness of natural breathing' },
    { value: 'loving_kindness', label: 'Loving Kindness', icon: '💝', description: 'Cultivation of universal love and compassion' },
    { value: 'wisdom_reflection', label: 'Wisdom Reflection', icon: '🔮', description: 'Deep contemplation of spiritual teachings' },
    { value: 'death_contemplation', label: 'Death Contemplation', icon: '🕯️', description: 'Profound reflection on mortality and meaning' },
    { value: 'impermanence', label: 'Impermanence', icon: '🍃', description: 'Understanding the transient nature of all things' },
    { value: 'self_inquiry', label: 'Self Inquiry', icon: '🪞', description: 'Investigation into the true nature of self' }
  ]);

  const [traditions] = useState([
    { value: 'universal', label: 'Universal', description: 'Accessible to all spiritual backgrounds' },
    { value: 'vedanta', label: 'Vedanta', description: 'Non-dual wisdom from ancient India' },
    { value: 'buddhist', label: 'Buddhist', description: 'Mindfulness and wisdom practices' },
    { value: 'yoga', label: 'Yoga', description: 'Union of body, mind, and spirit' },
    { value: 'zen', label: 'Zen', description: 'Direct pointing to awakened awareness' }
  ]);

  // Timer effect
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isActive && session) {
      interval = setInterval(() => {
        setTimeElapsed(prev => {
          if (prev >= session.duration_minutes * 60) {
            setIsActive(false);
            return session.duration_minutes * 60;
          }
          return prev + 1;
        });
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isActive, session]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const startContemplation = async (practiceType: string, tradition: string = 'universal', duration: number = 20) => {
    try {
      const response = await fetch('/api/v1/contemplation/begin', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          practice_type: practiceType,
          tradition: tradition,
          duration_minutes: duration,
          depth_level: 'focused'
        })
      });

      if (response.ok) {
        const newSession = await response.json();
        setSession(newSession);
        setIsActive(true);
        setTimeElapsed(0);
        setInsights([]);
      }
    } catch (error) {
      console.error('Failed to start contemplation:', error);
    }
  };

  const pauseContemplation = () => {
    setIsActive(false);
  };

  const resumeContemplation = () => {
    setIsActive(true);
  };

  const requestGuidance = async () => {
    if (!session) return;

    try {
      const response = await fetch('/api/v1/contemplation/guide', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: session.session_id,
          current_state: currentState
        })
      });

      if (response.ok) {
        const guidance = await response.json();
        setCurrentGuidance(guidance.guidance);
      }
    } catch (error) {
      console.error('Failed to get guidance:', error);
    }
  };

  const captureInsight = async () => {
    if (!session || !currentInsight.trim()) return;

    try {
      const response = await fetch('/api/v1/contemplation/insight', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: session.session_id,
          insight: currentInsight,
          integration_intention: ''
        })
      });

      if (response.ok) {
        setInsights([...insights, currentInsight]);
        setCurrentInsight('');
        onInsightCapture?.(currentInsight);
      }
    } catch (error) {
      console.error('Failed to capture insight:', error);
    }
  };

  const completeContemplation = async () => {
    if (!session) return;

    try {
      const response = await fetch('/api/v1/contemplation/complete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: session.session_id,
          completion_reflection: ''
        })
      });

      if (response.ok) {
        const summary = await response.json();
        setIsActive(false);
        setSession(null);
        setTimeElapsed(0);
        onSessionComplete?.(summary);
      }
    } catch (error) {
      console.error('Failed to complete contemplation:', error);
    }
  };

  const progress = session ? (timeElapsed / (session.duration_minutes * 60)) * 100 : 0;

  if (!session) {
    return (
      <div className="max-w-4xl mx-auto p-6 space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span className="text-2xl">🧘‍♂️</span>
              Deep Contemplation System
            </CardTitle>
            <p className="text-muted-foreground">
              Authentic spiritual practices for profound inner transformation
            </p>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {practiceTypes.map((practice) => (
                <Card 
                  key={practice.value} 
                  className="cursor-pointer hover:bg-accent transition-colors"
                  onClick={() => startContemplation(practice.value)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-center gap-3 mb-2">
                      <span className="text-2xl">{practice.icon}</span>
                      <h3 className="font-semibold">{practice.label}</h3>
                    </div>
                    <p className="text-sm text-muted-foreground">{practice.description}</p>
                  </CardContent>
                </Card>
              ))}
            </div>

            <div className="mt-6 p-4 bg-muted rounded-lg">
              <h3 className="font-semibold mb-2">🕉️ Spiritual Traditions Available</h3>
              <div className="flex flex-wrap gap-2">
                {traditions.map((tradition) => (
                  <Badge key={tradition.value} variant="outline">
                    {tradition.label}
                  </Badge>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      {/* Session Header */}
      <Card>
        <CardHeader>
          <div className="flex justify-between items-start">
            <div>
              <CardTitle className="flex items-center gap-2">
                <span className="text-2xl">
                  {practiceTypes.find(p => p.value === session.practice_type)?.icon || '🧘‍♂️'}
                </span>
                {practiceTypes.find(p => p.value === session.practice_type)?.label || session.practice_type}
              </CardTitle>
              <div className="flex gap-2 mt-2">
                <Badge variant="outline">{session.tradition}</Badge>
                <Badge variant="outline">{session.depth_level}</Badge>
                <Badge variant="outline">{session.duration_minutes} min</Badge>
              </div>
            </div>
            <div className="text-right">
              <div className="text-2xl font-mono">{formatTime(timeElapsed)}</div>
              <div className="text-sm text-muted-foreground">
                / {formatTime(session.duration_minutes * 60)}
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Progress value={progress} className="mb-4" />
          <div className="flex gap-2">
            {!isActive ? (
              <Button onClick={resumeContemplation} size="sm">
                <Play className="w-4 h-4 mr-2" />
                {timeElapsed > 0 ? 'Resume' : 'Begin'}
              </Button>
            ) : (
              <Button onClick={pauseContemplation} variant="outline" size="sm">
                <Pause className="w-4 h-4 mr-2" />
                Pause
              </Button>
            )}
            <Button onClick={completeContemplation} variant="outline" size="sm">
              <Square className="w-4 h-4 mr-2" />
              Complete
            </Button>
            <Button onClick={requestGuidance} variant="outline" size="sm">
              <BookOpen className="w-4 h-4 mr-2" />
              Guidance
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Practice Guidance */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="w-5 h-5" />
            Practice Guidance
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <p className="text-sm leading-relaxed">{session.guidance_text}</p>
            </div>
            
            {session.sanskrit_wisdom && (
              <div className="p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg">
                <h4 className="font-semibold text-sm mb-2">🕉️ Sanskrit Wisdom</h4>
                <p className="text-sm italic">{session.sanskrit_wisdom}</p>
              </div>
            )}

            {currentGuidance && (
              <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                <h4 className="font-semibold text-sm mb-2">✨ Live Guidance</h4>
                <p className="text-sm mb-2"><strong>Instruction:</strong> {currentGuidance.guidance.instruction}</p>
                <p className="text-sm mb-2"><strong>Encouragement:</strong> {currentGuidance.guidance.encouragement}</p>
                <p className="text-sm"><strong>Technique:</strong> {currentGuidance.guidance.technique}</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* State Selection & Mantra */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Heart className="w-5 h-5" />
              Current State
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-2">
              {['peaceful', 'distracted', 'insightful', 'resistant', 'profound'].map((state) => (
                <Button
                  key={state}
                  variant={currentState === state ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setCurrentState(state)}
                  className="text-xs"
                >
                  {state}
                </Button>
              ))}
            </div>
            {session.mantras.length > 0 && (
              <div className="mt-4 p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                <h4 className="font-semibold text-sm mb-2">🔮 Sacred Mantra</h4>
                <p className="text-sm italic text-center">{session.mantras[0]}</p>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Lightbulb className="w-5 h-5" />
              Capture Insights
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <textarea
                value={currentInsight}
                onChange={(e) => setCurrentInsight(e.target.value)}
                placeholder="What wisdom is arising? Capture your insights here..."
                className="w-full p-3 border rounded-lg text-sm resize-none"
                rows={3}
              />
              <Button
                onClick={captureInsight}
                size="sm"
                disabled={!currentInsight.trim()}
                className="w-full"
              >
                <CheckCircle className="w-4 h-4 mr-2" />
                Capture Insight
              </Button>
              {insights.length > 0 && (
                <div className="text-xs text-muted-foreground">
                  {insights.length} insight{insights.length !== 1 ? 's' : ''} captured
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Reflection Prompts */}
      {session.reflection_prompts.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <RotateCcw className="w-5 h-5" />
              Reflection Prompts
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {session.reflection_prompts.map((prompt, index) => (
                <div key={index} className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                  <p className="text-sm">{prompt}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default DeepContemplationInterface;
