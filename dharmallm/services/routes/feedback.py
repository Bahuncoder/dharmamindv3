"""User feedback routes"""

import logging
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackSubmission(BaseModel):
    """Feedback submission model"""
    user_id: str
    feedback_type: str  # "bug", "feature", "general"
    title: str
    description: str
    rating: Optional[int] = None


@router.post("/submit")
async def submit_feedback(feedback: FeedbackSubmission):
    """Submit user feedback"""
    logger.info(f"Feedback received from user: {feedback.user_id}")
    return {
        "success": True,
        "feedback_id": f"fb_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "message": "Thank you for your feedback"
    }


@router.get("/list")
async def list_feedback(user_id: Optional[str] = None):
    """List feedback submissions"""
    return {
        "feedbacks": [],
        "count": 0
    }
