/**
 * 🕉️ DharmaMind Usage Tracking API
 * 
 * Endpoint for tracking and syncing usage statistics
 * Supports real-time usage monitoring and analytics
 */

import { NextApiRequest, NextApiResponse } from 'next';
import { getServerSession } from 'next-auth/next';
// import { authOptions } from '../auth/[...nextauth]';

// ===============================
// TYPES
// ===============================

interface UsageStats {
  messagesUsed: number;
  messagesLimit: number;
  wisdomModulesUsed: number;
  wisdomModulesLimit: number;
  apiRequestsUsed: number;
  apiRequestsLimit: number;
  resetDate: string;
  periodStart: string;
  periodEnd: string;
}

interface UsageTrackingRequest {
  feature: 'messages' | 'wisdom_modules' | 'api_requests';
  amount: number;
  usage: UsageStats;
}

interface UsageResponse {
  success: boolean;
  usage?: UsageStats;
  canProceed?: boolean;
  message?: string;
  error?: string;
}

// ===============================
// MOCK DATABASE
// ===============================

// In a real app, this would be in your database
const userUsageData = new Map<string, UsageStats>();

const getDefaultUsage = (): UsageStats => {
  const now = new Date();
  const resetDate = new Date(now.getFullYear(), now.getMonth() + 1, 1);
  
  return {
    messagesUsed: 0,
    messagesLimit: 50, // Free plan default
    wisdomModulesUsed: 0,
    wisdomModulesLimit: 5,
    apiRequestsUsed: 0,
    apiRequestsLimit: 0,
    resetDate: resetDate.toISOString(),
    periodStart: now.toISOString(),
    periodEnd: resetDate.toISOString(),
  };
};

// ===============================
// API HANDLER
// ===============================

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<UsageResponse>
) {
  try {
    // Simple user identification - in production, use proper auth
    const userId = req.headers['x-user-id'] as string || 'demo_user';

    switch (req.method) {
      case 'GET':
        return handleGetUsage(req, res, userId);
      
      case 'POST':
        return handleTrackUsage(req, res, userId);
      
      case 'PUT':
        return handleUpdateUsage(req, res, userId);
      
      case 'DELETE':
        return handleResetUsage(req, res, userId);
      
      default:
        res.setHeader('Allow', ['GET', 'POST', 'PUT', 'DELETE']);
        return res.status(405).json({
          success: false,
          error: `Method ${req.method} not allowed`
        });
    }
  } catch (error) {
    console.error('Usage tracking API error:', error);
    return res.status(500).json({
      success: false,
      error: 'Internal server error'
    });
  }
}

// ===============================
// HANDLER FUNCTIONS
// ===============================

async function handleGetUsage(
  req: NextApiRequest,
  res: NextApiResponse<UsageResponse>,
  userId: string
) {
  try {
    let usage = userUsageData.get(userId);
    
    if (!usage) {
      usage = getDefaultUsage();
      userUsageData.set(userId, usage);
    }
    
    // Check if usage period has reset
    const resetDate = new Date(usage.resetDate);
    if (new Date() > resetDate) {
      usage = getDefaultUsage();
      userUsageData.set(userId, usage);
    }
    
    return res.status(200).json({
      success: true,
      usage
    });
  } catch (error) {
    console.error('Get usage error:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to retrieve usage data'
    });
  }
}

async function handleTrackUsage(
  req: NextApiRequest,
  res: NextApiResponse<UsageResponse>,
  userId: string
) {
  try {
    const { feature, amount = 1, usage: clientUsage }: UsageTrackingRequest = req.body;
    
    if (!feature || !['messages', 'wisdom_modules', 'api_requests'].includes(feature)) {
      return res.status(400).json({
        success: false,
        error: 'Invalid feature specified'
      });
    }
    
    // Get current usage or initialize
    let currentUsage = userUsageData.get(userId) || getDefaultUsage();
    
    // Check if usage period has reset
    const resetDate = new Date(currentUsage.resetDate);
    if (new Date() > resetDate) {
      currentUsage = getDefaultUsage();
    }
    
    // Track usage and check limits
    let canProceed = true;
    const newUsage = { ...currentUsage };
    
    switch (feature) {
      case 'messages':
        if (newUsage.messagesLimit !== -1 && newUsage.messagesUsed >= newUsage.messagesLimit) {
          canProceed = false;
        } else {
          newUsage.messagesUsed += amount;
        }
        break;
      case 'wisdom_modules':
        if (newUsage.wisdomModulesLimit !== -1 && newUsage.wisdomModulesUsed >= newUsage.wisdomModulesLimit) {
          canProceed = false;
        } else {
          newUsage.wisdomModulesUsed += amount;
        }
        break;
      case 'api_requests':
        if (newUsage.apiRequestsLimit !== -1 && newUsage.apiRequestsUsed >= newUsage.apiRequestsLimit) {
          canProceed = false;
        } else {
          newUsage.apiRequestsUsed += amount;
        }
        break;
    }
    
    if (canProceed) {
      userUsageData.set(userId, newUsage);
    }
    
    return res.status(200).json({
      success: true,
      usage: newUsage,
      canProceed,
      message: canProceed ? 'Usage tracked successfully' : `${feature} limit reached`
    });
  } catch (error) {
    console.error('Track usage error:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to track usage'
    });
  }
}

async function handleUpdateUsage(
  req: NextApiRequest,
  res: NextApiResponse<UsageResponse>,
  userId: string
) {
  try {
    const { usage }: { usage: UsageStats } = req.body;
    
    if (!usage) {
      return res.status(400).json({
        success: false,
        error: 'Usage data required'
      });
    }
    
    userUsageData.set(userId, usage);
    
    return res.status(200).json({
      success: true,
      usage,
      message: 'Usage updated successfully'
    });
  } catch (error) {
    console.error('Update usage error:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to update usage'
    });
  }
}

async function handleResetUsage(
  req: NextApiRequest,
  res: NextApiResponse<UsageResponse>,
  userId: string
) {
  try {
    const newUsage = getDefaultUsage();
    userUsageData.set(userId, newUsage);
    
    return res.status(200).json({
      success: true,
      usage: newUsage,
      message: 'Usage reset successfully'
    });
  } catch (error) {
    console.error('Reset usage error:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to reset usage'
    });
  }
}

// ===============================
// UTILITIES
// ===============================

/**
 * Get usage statistics for analytics
 */
export async function getUserUsageStats(userId: string): Promise<UsageStats | null> {
  try {
    let usage = userUsageData.get(userId);
    
    if (!usage) {
      return null;
    }
    
    // Check if usage period has reset
    const resetDate = new Date(usage.resetDate);
    if (new Date() > resetDate) {
      return getDefaultUsage();
    }
    
    return usage;
  } catch (error) {
    console.error('Get user usage stats error:', error);
    return null;
  }
}

/**
 * Check if user can perform an action
 */
export async function canUserPerformAction(
  userId: string, 
  feature: 'messages' | 'wisdom_modules' | 'api_requests'
): Promise<boolean> {
  try {
    const usage = await getUserUsageStats(userId);
    if (!usage) return true; // Allow if no usage data
    
    switch (feature) {
      case 'messages':
        return usage.messagesLimit === -1 || usage.messagesUsed < usage.messagesLimit;
      case 'wisdom_modules':
        return usage.wisdomModulesLimit === -1 || usage.wisdomModulesUsed < usage.wisdomModulesLimit;
      case 'api_requests':
        return usage.apiRequestsLimit === -1 || usage.apiRequestsUsed < usage.apiRequestsLimit;
      default:
        return false;
    }
  } catch (error) {
    console.error('Can user perform action error:', error);
    return false;
  }
}

/**
 * Update user plan limits
 */
export async function updateUserPlanLimits(
  userId: string,
  planLimits: {
    messagesLimit?: number;
    wisdomModulesLimit?: number;
    apiRequestsLimit?: number;
  }
): Promise<boolean> {
  try {
    let usage = userUsageData.get(userId) || getDefaultUsage();
    
    const updatedUsage = {
      ...usage,
      ...planLimits
    };
    
    userUsageData.set(userId, updatedUsage);
    return true;
  } catch (error) {
    console.error('Update user plan limits error:', error);
    return false;
  }
}
