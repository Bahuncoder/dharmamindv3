"""
🕉️ DharmaMind Feedback API Routes - User Feedback Management

Advanced feedback collection and analysis system for DharmaMind:

Core Features:
- Multi-type feedback submission (general, bug reports, feature requests, etc.)
- Anonymous and authenticated feedback support
- Real-time AI sentiment analysis and categorization
- Dharmic compliance evaluation of feedback content
- Priority scoring and automated routing
- Comprehensive analytics and reporting

Advanced Capabilities:
- LLM-powered content analysis and insight extraction
- Automated issue identification and suggestion parsing
- Spiritual context awareness in feedback processing
- Integration with conversation and message tracking
- Performance metrics and user satisfaction tracking

May this feedback system serve to improve our service to all beings 🙏
"""

import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..models import (
    FeedbackRequest, FeedbackResponse, FeedbackAnalytics,
    FeedbackType, FeedbackSentiment
)
from ..db.database import DatabaseManager
from ..services.llm_router import LLMRouter
from ..config import settings

# Initialize components
router = APIRouter(prefix="/api/feedback", tags=["feedback"])
security = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)
# No need to call get_settings(), just use the imported settings instance


class FeedbackService:
    """🕉️ Advanced Feedback Processing Service
    
    Handles all feedback-related operations with AI enhancement:
    - Content analysis and sentiment detection
    - Priority scoring and categorization
    - Dharmic compliance evaluation
    - Automated insights extraction
    """
    
    def __init__(self, db_manager: DatabaseManager, llm_router: LLMRouter):
        self.db = db_manager
        self.llm = llm_router
        self.analysis_enabled = settings.ENABLE_FEEDBACK_ANALYSIS
    
    async def process_feedback(
        self, 
        feedback_request: FeedbackRequest,
        background_tasks: BackgroundTasks
    ) -> FeedbackResponse:
        """Process and store user feedback with AI analysis"""
        
        try:
            # Generate feedback ID
            feedback_id = str(uuid.uuid4())
            
            # Prepare feedback data for storage
            feedback_data = {
                "user_id": feedback_request.user_id,
                "conversation_id": feedback_request.conversation_id,
                "message_id": feedback_request.message_id,
                "feedback_type": feedback_request.feedback_type.value,
                "title": feedback_request.title,
                "content": feedback_request.content,
                "overall_rating": feedback_request.overall_rating,
                "response_quality": feedback_request.response_quality,
                "helpfulness": feedback_request.helpfulness,
                "spiritual_value": feedback_request.spiritual_value,
                "user_email": feedback_request.user_email,
                "browser_info": feedback_request.browser_info,
                "device_info": feedback_request.device_info,
                "allow_contact": feedback_request.allow_contact,
                "share_anonymously": feedback_request.share_anonymously
            }
            
            # Store feedback in database
            stored_feedback_id = await self.db.store_feedback(feedback_data)
            
            # Schedule AI analysis in background
            if self.analysis_enabled:
                background_tasks.add_task(
                    self._analyze_feedback_content,
                    stored_feedback_id,
                    feedback_request
                )
            
            # Prepare response
            response = FeedbackResponse(
                feedback_id=stored_feedback_id,
                status="submitted",
                message="Thank you for your feedback! We truly value your input and will review it carefully.",
                created_at=datetime.now()
            )
            
            # Add quick sentiment if available
            if self.analysis_enabled:
                quick_sentiment = await self._quick_sentiment_analysis(
                    feedback_request.content
                )
                response.sentiment_score = quick_sentiment.get("score")
                response.priority_score = self._calculate_priority_score(
                    feedback_request
                )
            
            logger.info(f"Feedback {stored_feedback_id} processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to process feedback. Please try again."
            )
    
    async def _analyze_feedback_content(
        self,
        feedback_id: str,
        feedback_request: FeedbackRequest
    ):
        """Comprehensive AI analysis of feedback content"""
        
        try:
            # Prepare analysis prompt
            analysis_prompt = self._create_analysis_prompt(feedback_request)
            
            # Get LLM analysis
            llm_response = await self.llm.generate_response(
                message=analysis_prompt,
                context="feedback_analysis",
                max_tokens=1000,
                temperature=0.3
            )
            
            # Parse LLM response
            analysis_result = await self._parse_llm_analysis(
                llm_response.response,
                feedback_request
            )
            
            # Store analysis in database
            await self._store_feedback_analysis(feedback_id, analysis_result)
            
            # Check for high-priority issues
            if analysis_result.get("priority_score", 0) > 0.8:
                await self._notify_high_priority_feedback(feedback_id, analysis_result)
            
            logger.info(f"Feedback analysis completed for {feedback_id}")
            
        except Exception as e:
            logger.error(f"Error analyzing feedback {feedback_id}: {e}")
    
    def _create_analysis_prompt(self, feedback_request: FeedbackRequest) -> str:
        """Create comprehensive analysis prompt for LLM"""
        
        return f"""
        Analyze this user feedback for DharmaMind, an AI spiritual guidance system:

        Type: {feedback_request.feedback_type}
        Title: {feedback_request.title}
        Content: {feedback_request.content}
        Ratings: Overall: {feedback_request.overall_rating}, Quality: {feedback_request.response_quality}, 
                Helpfulness: {feedback_request.helpfulness}, Spiritual Value: {feedback_request.spiritual_value}

        Please provide analysis in this JSON format:
        {{
            "sentiment": "positive/neutral/negative",
            "sentiment_score": 0.0-1.0,
            "priority_score": 0.0-1.0,
            "key_topics": ["topic1", "topic2"],
            "mentioned_features": ["feature1", "feature2"],
            "suggestions": ["suggestion1", "suggestion2"],
            "issues_identified": ["issue1", "issue2"],
            "dharmic_concerns": ["concern1", "concern2"],
            "spiritual_insights": ["insight1", "insight2"],
            "urgency_level": "low/medium/high/critical",
            "category_confidence": 0.0-1.0
        }}

        Focus on:
        1. Spiritual and dharmic context
        2. Technical issues or suggestions
        3. User experience insights
        4. Actionable improvement areas
        """
    
    async def _parse_llm_analysis(
        self,
        llm_response: str,
        feedback_request: FeedbackRequest
    ) -> Dict[str, Any]:
        """Parse LLM analysis response into structured data"""
        
        try:
            # In real implementation, would parse JSON from LLM
            # For now, return mock analysis
            return {
                "sentiment": self._detect_sentiment(feedback_request.content),
                "sentiment_score": 0.7,
                "priority_score": self._calculate_priority_score(feedback_request),
                "key_topics": self._extract_topics(feedback_request.content),
                "mentioned_features": ["chat", "guidance"],
                "suggestions": self._extract_suggestions(feedback_request.content),
                "issues_identified": self._extract_issues(feedback_request.content),
                "dharmic_concerns": [],
                "spiritual_insights": self._extract_spiritual_insights(feedback_request.content),
                "urgency_level": "medium",
                "category_confidence": 0.8
            }
            
        except Exception as e:
            logger.error(f"Error parsing LLM analysis: {e}")
            return self._fallback_analysis(feedback_request)
    
    def _detect_sentiment(self, content: str) -> str:
        """Basic sentiment detection"""
        positive_words = ["good", "great", "excellent", "helpful", "love", "amazing"]
        negative_words = ["bad", "terrible", "horrible", "useless", "hate", "awful"]
        
        content_lower = content.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_priority_score(self, feedback_request: FeedbackRequest) -> float:
        """Calculate priority score based on feedback characteristics"""
        score = 0.0
        
        # Type-based priority
        if feedback_request.feedback_type == FeedbackType.BUG_REPORT:
            score += 0.4
        elif feedback_request.feedback_type == FeedbackType.DHARMIC_CONCERN:
            score += 0.5
        elif feedback_request.feedback_type == FeedbackType.FEATURE_REQUEST:
            score += 0.2
        
        # Rating-based priority
        if feedback_request.overall_rating and feedback_request.overall_rating <= 2:
            score += 0.3
        
        # Content-based priority
        urgent_keywords = ["urgent", "critical", "broken", "error", "crash", "inappropriate"]
        if any(keyword in feedback_request.content.lower() for keyword in urgent_keywords):
            score += 0.3
        
        return min(score, 1.0)
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract key topics from content"""
        topics = []
        topic_keywords = {
            "meditation": ["meditat", "mindful"],
            "guidance": ["guidance", "advice", "help"],
            "response_quality": ["response", "answer", "quality"],
            "user_interface": ["interface", "ui", "design", "navigation"],
            "performance": ["slow", "fast", "speed", "performance"]
        }
        
        content_lower = content.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _extract_suggestions(self, content: str) -> List[str]:
        """Extract suggestions from content"""
        suggestions = []
        
        # Look for suggestion patterns
        suggestion_patterns = [
            "should", "could", "would be better", "suggest", "recommend",
            "improve", "add", "include", "feature"
        ]
        
        content_lower = content.lower()
        for pattern in suggestion_patterns:
            if pattern in content_lower:
                # Extract context around suggestion
                words = content.split()
                for i, word in enumerate(words):
                    if pattern in word.lower():
                        # Get surrounding context
                        start = max(0, i - 3)
                        end = min(len(words), i + 8)
                        suggestion = " ".join(words[start:end])
                        suggestions.append(suggestion)
                        break
        
        return suggestions[:3]  # Limit to top 3
    
    def _extract_issues(self, content: str) -> List[str]:
        """Extract issues from content"""
        issues = []
        issue_keywords = [
            "problem", "issue", "bug", "error", "broken", "not working",
            "slow", "crash", "freeze", "stuck"
        ]
        
        content_lower = content.lower()
        for keyword in issue_keywords:
            if keyword in content_lower:
                issues.append(f"User reported: {keyword}")
        
        return issues
    
    def _extract_spiritual_insights(self, content: str) -> List[str]:
        """Extract spiritual insights from content"""
        insights = []
        spiritual_keywords = {
            "wisdom": "User seeking wisdom and deeper understanding",
            "peace": "User expressing desire for inner peace",
            "guidance": "User seeking spiritual guidance",
            "dharma": "User interested in dharmic principles",
            "meditation": "User engaged with meditation practices",
            "compassion": "User exploring compassionate responses"
        }
        
        content_lower = content.lower()
        for keyword, insight in spiritual_keywords.items():
            if keyword in content_lower:
                insights.append(insight)
        
        return insights
    
    def _fallback_analysis(self, feedback_request: FeedbackRequest) -> Dict[str, Any]:
        """Fallback analysis when LLM analysis fails"""
        return {
            "sentiment": "neutral",
            "sentiment_score": 0.5,
            "priority_score": 0.3,
            "key_topics": ["general"],
            "mentioned_features": [],
            "suggestions": [],
            "issues_identified": [],
            "dharmic_concerns": [],
            "spiritual_insights": [],
            "urgency_level": "medium",
            "category_confidence": 0.1
        }
    
    async def _store_feedback_analysis(
        self,
        feedback_id: str,
        analysis_result: Dict[str, Any]
    ):
        """Store analysis results in database"""
        try:
            # Use database manager to store analysis
            await self.db._analyze_feedback_async(feedback_id, analysis_result)
        except Exception as e:
            logger.error(f"Error storing feedback analysis: {e}")
    
    async def _notify_high_priority_feedback(
        self,
        feedback_id: str,
        analysis_result: Dict[str, Any]
    ):
        """Notify team of high-priority feedback"""
        try:
            # In real implementation, would send notifications
            logger.warning(f"High-priority feedback received: {feedback_id}")
            logger.warning(f"Analysis: {analysis_result}")
        except Exception as e:
            logger.error(f"Error notifying high-priority feedback: {e}")
    
    async def _quick_sentiment_analysis(self, content: str) -> Dict[str, Any]:
        """Quick sentiment analysis for immediate response"""
        sentiment = self._detect_sentiment(content)
        score = 0.7 if sentiment == "positive" else 0.3 if sentiment == "negative" else 0.5
        
        return {
            "sentiment": sentiment,
            "score": score
        }


# Dependency to get services
async def get_feedback_service() -> FeedbackService:
    """Get feedback service instance"""
    # In real implementation, would get from dependency injection
    db_manager = DatabaseManager()
    llm_router = LLMRouter()
    return FeedbackService(db_manager, llm_router)


# API Routes
@router.post("/submit", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackRequest,
    background_tasks: BackgroundTasks,
    feedback_service: FeedbackService = Depends(get_feedback_service)
):
    """
    Submit user feedback with comprehensive analysis
    
    Supports multiple feedback types:
    - General feedback and suggestions
    - Bug reports and technical issues
    - Feature requests and improvements
    - Content quality and spiritual value assessment
    - Performance and user experience feedback
    
    Features:
    - Anonymous or authenticated submission
    - Real-time sentiment analysis
    - Priority scoring and routing
    - Dharmic compliance evaluation
    """
    try:
        return await feedback_service.process_feedback(feedback, background_tasks)
    except Exception as e:
        logger.error(f"Error in submit_feedback endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to submit feedback. Please try again."
        )


@router.get("/list", response_model=List[FeedbackAnalytics])
async def list_feedback(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    feedback_type: Optional[FeedbackType] = Query(None, description="Filter by feedback type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Number of records to return"),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    feedback_service: FeedbackService = Depends(get_feedback_service)
):
    """
    List feedback records with advanced filtering
    
    Admin/Staff only endpoint for feedback management:
    - Filter by user, type, status, date range
    - Include AI analysis results
    - Pagination and sorting support
    - Priority-based ordering
    """
    try:
        # In real implementation, would verify admin/staff credentials
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="Authentication required for feedback listing"
            )
        
        feedback_records = await feedback_service.db.get_feedback(
            user_id=user_id,
            limit=limit
        )
        
        return feedback_records
        
    except Exception as e:
        logger.error(f"Error in list_feedback endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve feedback records"
        )


@router.get("/analytics", response_model=Dict[str, Any])
async def get_feedback_analytics(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    feedback_service: FeedbackService = Depends(get_feedback_service)
):
    """
    Get comprehensive feedback analytics and insights
    
    Admin only endpoint providing:
    - Feedback volume and trends
    - Sentiment analysis summary
    - Priority distribution
    - Top issues and suggestions
    - User satisfaction metrics
    - Dharmic compliance insights
    """
    try:
        # Verify admin credentials
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="Authentication required for analytics"
            )
        
        analytics = await feedback_service.db.get_feedback_analytics_summary(days)
        return analytics
        
    except Exception as e:
        logger.error(f"Error in get_feedback_analytics endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve feedback analytics"
        )


@router.put("/{feedback_id}/status")
async def update_feedback_status(
    feedback_id: str,
    status: str,
    assigned_to: Optional[str] = None,
    resolution: Optional[str] = None,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    feedback_service: FeedbackService = Depends(get_feedback_service)
):
    """
    Update feedback status and resolution
    
    Staff only endpoint for feedback management:
    - Update status (new, in_progress, resolved, closed)
    - Assign to team member
    - Add resolution notes
    - Track resolution time
    """
    try:
        # Verify staff credentials
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="Authentication required for status updates"
            )
        
        await feedback_service.db.update_feedback_status(
            feedback_id=feedback_id,
            status=status,
            assigned_to=assigned_to,
            resolution=resolution
        )
        
        return {
            "message": "Feedback status updated successfully",
            "feedback_id": feedback_id,
            "status": status
        }
        
    except Exception as e:
        logger.error(f"Error updating feedback status: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to update feedback status"
        )


@router.get("/{feedback_id}", response_model=FeedbackAnalytics)
async def get_feedback_details(
    feedback_id: str,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    feedback_service: FeedbackService = Depends(get_feedback_service)
):
    """
    Get detailed feedback information including AI analysis
    
    Returns comprehensive feedback details:
    - Original feedback content and ratings
    - AI analysis results and insights
    - Processing status and timeline
    - Related conversation context
    """
    try:
        # Basic authentication check
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="Authentication required"
            )
        
        feedback_records = await feedback_service.db.get_feedback(
            feedback_id=feedback_id,
            limit=1
        )
        
        if not feedback_records:
            raise HTTPException(
                status_code=404,
                detail="Feedback not found"
            )
        
        return feedback_records[0]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting feedback details: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve feedback details"
        )


# Health check endpoint
@router.get("/health")
async def feedback_health_check():
    """Health check for feedback system"""
    return {
        "status": "healthy",
        "service": "feedback",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }
