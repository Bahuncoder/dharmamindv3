import type { NextApiRequest, NextApiResponse } from 'next'
import chatService from '../../services/chatService'

interface ChatRequest {
  message: string
  conversation_id?: string | null
  user_context?: Record<string, any>
  context?: string
  user_id?: string
  language?: string
  user?: {
    name?: string
    email?: string
    plan?: string
    isGuest?: boolean
  }
}

interface ChatResponse {
  response: string
  conversation_id?: string
  confidence_score?: number
  dharmic_alignment?: number
  modules_used?: string[]
  timestamp?: string
  model_used?: string
  processing_time?: number
  sources?: string[]
  suggestions?: string[]
}

interface ErrorResponse {
  error: string
  message: string
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<ChatResponse | ErrorResponse>
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed', message: 'Only POST requests are allowed' })
  }

  const { message, conversation_id, user_context, context, user_id, language, user }: ChatRequest = req.body

  if (!message || typeof message !== 'string') {
    return res.status(400).json({ 
      error: 'Invalid request', 
      message: 'Message is required and must be a string' 
    })
  }

  try {
    console.log('Chat API: Processing message for user:', user?.name || user_id || 'anonymous')
    console.log('Message:', message.substring(0, 100) + '...')
    
    // Use the chatService to connect to the comprehensive backend on port 8000
    const response = await chatService.sendMessage(
      message,
      conversation_id || undefined,
      user_id || user?.email || 'anonymous'
    )

    console.log('Chat API: Backend response received successfully')
    res.status(200).json(response)

  } catch (error) {
    console.error('Chat API error:', error)
    
    // Check if it's a connection error to provide fallback
    if (error instanceof Error && error.message.includes('Unable to connect to backend')) {
      console.log('Chat API: Backend unavailable, providing fallback response')
      
      // Generate a spiritual fallback response using the chatService
      const fallbackResponse = chatService.generateEnhancedFallbackResponse(message)
      
      return res.status(200).json(fallbackResponse)
    }

    // Return error response
    res.status(500).json({
      error: 'Internal server error',
      message: error instanceof Error ? error.message : 'An unexpected error occurred while processing your request'
    })
  }
}
