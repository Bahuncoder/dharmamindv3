"""
Notification Service for DharmaMind platform

Handles email notifications, SMS, and push notifications for user engagement,
verification, and spiritual guidance updates.
"""

import asyncio
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any
from datetime import datetime
import random
import string

logger = logging.getLogger(__name__)

class NotificationService:
    """Service for handling various types of notifications"""
    
    def __init__(self):
        self.smtp_server = "localhost"  # Configure as needed
        self.smtp_port = 587
        self.smtp_username = None
        self.smtp_password = None
        self.from_email = "noreply@dharmamind.ai"
        
    async def initialize(self) -> bool:
        """Initialize the notification service"""
        try:
            logger.info("Notification service initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize notification service: {e}")
            return False
    
    def generate_verification_code(self, length: int = 6) -> str:
        """Generate a random verification code"""
        return ''.join(random.choices(string.digits, k=length))
    
    async def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        html_body: Optional[str] = None
    ) -> bool:
        """Send email notification"""
        try:
            # This is a mock implementation for development
            # In production, this would use actual SMTP or email service
            logger.info(f"📧 Mock email sent to {to_email}")
            logger.info(f"Subject: {subject}")
            logger.info(f"Body: {body[:100]}...")
            
            # Simulate email sending delay
            await asyncio.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False
    
    async def send_verification_email(
        self,
        to_email: str,
        verification_code: str,
        user_name: Optional[str] = None
    ) -> bool:
        """Send email verification code"""
        try:
            subject = "🕉️ DharmaMind - Email Verification"
            
            body = f"""
Namaste {user_name or 'Friend'},

Welcome to DharmaMind - Your journey of spiritual wisdom begins here.

Your email verification code is: {verification_code}

Please enter this code to verify your email address and complete your registration.

This code will expire in 10 minutes.

May your path be filled with light and wisdom.

🕉️ DharmaMind Team
            """.strip()
            
            html_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #8B4513;">🕉️ DharmaMind Email Verification</h2>
                    <p>Namaste {user_name or 'Friend'},</p>
                    <p>Welcome to DharmaMind - Your journey of spiritual wisdom begins here.</p>
                    
                    <div style="background: #f9f9f9; padding: 20px; border-radius: 5px; text-align: center; margin: 20px 0;">
                        <h3 style="color: #8B4513;">Your Verification Code</h3>
                        <p style="font-size: 24px; font-weight: bold; color: #FF6347; letter-spacing: 3px;">{verification_code}</p>
                    </div>
                    
                    <p>Please enter this code to verify your email address and complete your registration.</p>
                    <p><em>This code will expire in 10 minutes.</em></p>
                    
                    <hr style="margin: 30px 0;">
                    <p style="color: #666; font-style: italic;">May your path be filled with light and wisdom.</p>
                    <p style="color: #8B4513; font-weight: bold;">🕉️ DharmaMind Team</p>
                </div>
            </body>
            </html>
            """
            
            return await self.send_email(to_email, subject, body, html_body)
            
        except Exception as e:
            logger.error(f"Failed to send verification email to {to_email}: {e}")
            return False
    
    async def send_password_reset_email(
        self,
        to_email: str,
        reset_token: str,
        user_name: Optional[str] = None
    ) -> bool:
        """Send password reset email"""
        try:
            subject = "🕉️ DharmaMind - Password Reset Request"
            
            body = f"""
Namaste {user_name or 'Friend'},

We received a request to reset your DharmaMind account password.

Your password reset code is: {reset_token}

If you didn't request this password reset, please ignore this email.

This code will expire in 30 minutes.

🕉️ DharmaMind Team
            """.strip()
            
            return await self.send_email(to_email, subject, body)
            
        except Exception as e:
            logger.error(f"Failed to send password reset email to {to_email}: {e}")
            return False
    
    async def send_welcome_email(
        self,
        to_email: str,
        user_name: str
    ) -> bool:
        """Send welcome email after successful registration"""
        try:
            subject = "🕉️ Welcome to DharmaMind - Your Spiritual Journey Begins"
            
            body = f"""
Namaste {user_name},

Welcome to the DharmaMind family! 🙏

You have successfully joined a community dedicated to spiritual growth, wisdom, and compassionate living. Here's what you can explore:

🧘 Personal Spiritual Guidance
📚 Ancient Wisdom & Modern Insights  
🌟 Meditation & Mindfulness Practices
💫 Dharmic Living Principles
🤝 Supportive Spiritual Community

Your journey of self-discovery and enlightenment starts now. Feel free to ask me anything about:
- Meditation techniques
- Spiritual philosophy
- Life guidance based on dharmic principles
- Personal growth and development

Remember: "The path of wisdom is not about reaching a destination, but about transforming through the journey itself."

May your path be filled with peace, wisdom, and joy.

With loving-kindness,
🕉️ DharmaMind Team
            """.strip()
            
            return await self.send_email(to_email, subject, body)
            
        except Exception as e:
            logger.error(f"Failed to send welcome email to {to_email}: {e}")
            return False
    
    async def send_daily_wisdom(
        self,
        to_email: str,
        user_name: str,
        wisdom_quote: str,
        wisdom_explanation: Optional[str] = None
    ) -> bool:
        """Send daily wisdom notification"""
        try:
            subject = "🌅 Daily Wisdom from DharmaMind"
            
            body = f"""
Good morning {user_name},

Here's your daily dose of wisdom:

"{wisdom_quote}"

{wisdom_explanation or ''}

Reflect on this throughout your day and see how it applies to your current life situation.

🕉️ DharmaMind
            """.strip()
            
            return await self.send_email(to_email, subject, body)
            
        except Exception as e:
            logger.error(f"Failed to send daily wisdom to {to_email}: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check notification service health"""
        return {
            "status": "healthy",
            "service": "notification",
            "email_enabled": True,
            "sms_enabled": False,  # Not implemented
            "push_enabled": False,  # Not implemented
        }

# Global notification service instance
_notification_service: Optional[NotificationService] = None

async def get_notification_service() -> NotificationService:
    """Get the global notification service instance"""
    global _notification_service
    
    if _notification_service is None:
        _notification_service = NotificationService()
        await _notification_service.initialize()
    
    return _notification_service

async def send_verification_code(
    email: str,
    phone: Optional[str] = None,
    code: Optional[str] = None,
    user_name: Optional[str] = None
) -> bool:
    """Send verification code to email (and optionally phone)"""
    try:
        service = await get_notification_service()
        
        # If no code provided, generate one
        if not code:
            code = service.generate_verification_code()
        
        # Log for development
        logger.info(f"📧 Verification code for {email}: {code}")
        print(f"📧 Verification code for {email}: {code}")
        
        # Try to send email (will gracefully fail if SMTP not configured)
        try:
            await service.send_verification_email(email, code, user_name)
        except Exception as e:
            logger.warning(f"Email send failed (OK for dev): {e}")
        
        return True
            
    except Exception as e:
        logger.error(f"Failed to send verification code to {email}: {e}")
        return False

async def send_password_reset_code(
    email: str,
    user_name: Optional[str] = None
) -> Optional[str]:
    """Send password reset code to email and return the code"""
    try:
        service = await get_notification_service()
        code = service.generate_verification_code(length=8)
        
        success = await service.send_password_reset_email(email, code, user_name)
        
        if success:
            return code
        else:
            return None
            
    except Exception as e:
        logger.error(f"Failed to send password reset code to {email}: {e}")
        return None
