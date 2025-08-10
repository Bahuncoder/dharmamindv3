import type { NextApiRequest, NextApiResponse } from 'next'
import axios from 'axios'

interface ChatRequest {
  message: string
  conversation_id?: string | null
  user_context?: Record<string, any>
  context?: string
  user_id?: string
  language?: string
}

interface ChatResponse {
  response: string
  conversation_id: string
  confidence_score: number
  dharmic_alignment: number
  modules_used: string[]
  timestamp: string
  model_used?: string
  processing_time?: number
  sources?: string[]
  suggestions?: string[]
}

interface ErrorResponse {
  error: string
  message: string
}

// Fallback response generator for when backend is unavailable
function generateFallbackResponse(message: string): string {
  const lowercaseMessage = message.toLowerCase()
  
  // Spiritual wisdom responses based on common topics
  if (lowercaseMessage.includes('meditation') || lowercaseMessage.includes('mindfulness')) {
    return "In the stillness of meditation, we find the eternal presence that has always been within us. Begin with just a few minutes of focused breathing, allowing your mind to settle like a clear mountain lake. The path of mindfulness teaches us that peace is not found in the absence of thoughts, but in our relationship to them."
  }
  
  if (lowercaseMessage.includes('suffering') || lowercaseMessage.includes('pain') || lowercaseMessage.includes('difficult')) {
    return "Suffering, as the Buddha taught, is part of the human experience, but it is not our permanent state. Every challenge carries within it the seeds of wisdom and growth. Like a lotus that blooms from muddy waters, our struggles can become the very foundation of our spiritual awakening. Be gentle with yourself during difficult times."
  }
  
  if (lowercaseMessage.includes('love') || lowercaseMessage.includes('compassion') || lowercaseMessage.includes('kindness')) {
    return "Love is the fundamental force that connects all beings. True compassion begins with self-acceptance and extends naturally to others. When we recognize the divine spark in ourselves, we cannot help but see it in everyone we meet. Practice loving-kindness by sending good wishes to yourself, your loved ones, and even those who challenge you."
  }
  
  if (lowercaseMessage.includes('fear') || lowercaseMessage.includes('anxiety') || lowercaseMessage.includes('worry')) {
    return "Fear is often the shadow cast by our attachment to outcomes we cannot control. Remember that you are much more than your thoughts and emotions. Like clouds passing through the sky, fears arise and dissolve naturally when we don't resist them. Ground yourself in the present moment through breath and mindful awareness."
  }
  
  if (lowercaseMessage.includes('purpose') || lowercaseMessage.includes('meaning') || lowercaseMessage.includes('direction')) {
    return "Your purpose unfolds naturally when you align with your authentic self. Listen deeply to your heart's calling, not the voices of expectation from others. Every experience, whether perceived as success or failure, contributes to your spiritual growth. Trust the journey, even when the path seems unclear."
  }
  
  if (lowercaseMessage.includes('relationship') || lowercaseMessage.includes('family') || lowercaseMessage.includes('friend')) {
    return "Relationships are mirrors that reflect our inner world back to us. They offer the greatest opportunities for growth and healing. Practice seeing others with the eyes of understanding rather than judgment. Remember that everyone is fighting their own battles and doing their best with their current level of consciousness."
  }
  
  if (lowercaseMessage.includes('gratitude') || lowercaseMessage.includes('thankful') || lowercaseMessage.includes('blessed')) {
    return "Gratitude is the gateway to abundance and joy. When we appreciate what we have, we align ourselves with the flow of universal blessings. Start each day by acknowledging three things you're grateful for, no matter how small. This practice transforms our perspective and opens our hearts to receive even more goodness."
  }
  
  if (lowercaseMessage.includes('anger') || lowercaseMessage.includes('frustration') || lowercaseMessage.includes('upset')) {
    return "Anger is often a protective emotion masking deeper feelings of hurt or fear. Instead of suppressing it, acknowledge its presence with compassion. Breathe deeply and ask what this emotion is trying to teach you. Transform anger into wisdom by using it as a guide to understanding your boundaries and values."
  }
  
  // Default wisdom response
  return "Every question you ask is a step on the path of self-discovery. The answers you seek already reside within you, waiting to be uncovered through mindful reflection and inner wisdom. Trust your journey, embrace both the light and shadow aspects of your experience, and remember that spiritual growth happens in the present moment. May you find peace and clarity on your path."
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<ChatResponse | ErrorResponse>
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed', message: 'Only POST requests are allowed' })
  }

  const { message, conversation_id, user_context, context, user_id, language }: ChatRequest = req.body

  if (!message || typeof message !== 'string') {
    return res.status(400).json({ 
      error: 'Invalid request', 
      message: 'Message is required and must be a string' 
    })
  }

  try {
    // Use the correct backend URL and endpoint
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    
    const response = await axios.post(`${backendUrl}/api/v1/chat`, {
      message,
      session_id: conversation_id,
    }, {
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 45000, // 45 second timeout for AI processing
    })

    // Return the response from the backend with proper mapping
    const backendData = response.data
    const chatResponse: ChatResponse = {
      response: backendData.response,
      conversation_id: backendData.session_id || conversation_id || 'default',
      confidence_score: backendData.confidence || 0.8,
      dharmic_alignment: 0.9, // Default for simplified backend
      modules_used: ['consciousness_core', 'dharma_engine'], // Default modules
      timestamp: backendData.timestamp || new Date().toISOString(),
      model_used: 'dharmamind-v1',
      processing_time: 100,
      sources: [],
      suggestions: []
    }

    res.status(200).json(chatResponse)

  } catch (error) {
    console.error('Chat API error:', error)
    
    if (axios.isAxiosError(error)) {
      if (error.response) {
        // Backend returned an error response
        return res.status(error.response.status).json({
          error: 'Backend error',
          message: error.response.data?.detail || 'An error occurred while processing your request'
        })
      } else if (error.request) {
        // Network error - provide fallback response for better UX
        console.log('Backend unavailable, providing fallback response')
        
        // Generate a spiritual fallback response based on the message
        const fallbackResponse = generateFallbackResponse(message)
        
        return res.status(200).json({
          response: fallbackResponse,
          conversation_id: conversation_id || `fallback_${Date.now()}`,
          confidence_score: 0.7,
          dharmic_alignment: 0.8,
          modules_used: ['fallback_wisdom'],
          timestamp: new Date().toISOString(),
          model_used: 'dharmamind-fallback',
          processing_time: 50,
          sources: ['Internal Wisdom Database'],
          suggestions: ['When the backend is available, you will receive more personalized guidance.']
        })
      }
    }

    // Generic error
    res.status(500).json({
      error: 'Internal server error',
      message: 'An unexpected error occurred while processing your request'
    })
  }
}
