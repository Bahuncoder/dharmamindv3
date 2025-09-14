"""
📨 Notification Service
=======================

Handles email notifications, SMS, and other communication channels for DharmaMind.
"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime
import asyncio
import warnings

try:
    import aiosmtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
except ImportError as e:
    warnings.warn(f"Email dependencies not available: {e}")
    aiosmtplib = None
    MIMEText = None
    MIMEMultipart = None

logger = logging.getLogger(__name__)

class NotificationType(str, Enum):
    """Types of notifications"""
    EMAIL_VERIFICATION = "email_verification"
    PASSWORD_RESET = "password_reset"
    SPIRITUAL_REMINDER = "spiritual_reminder"
    RISHI_MESSAGE = "rishi_message"
    SYSTEM_ALERT = "system_alert"
    WELCOME = "welcome"

class NotificationStatus(str, Enum):
    """Status of notifications"""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    DELIVERED = "delivered"

class NotificationChannel(str, Enum):
    """Notification delivery channels"""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"

class NotificationService:
    """📨 Notification service for DharmaMind"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Email configuration (use environment variables in production)
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.smtp_username = "dharma@example.com"
        self.smtp_password = "app_password"
        self.from_email = "DharmaMind <dharma@example.com>"
        
        # Notification templates
        self.templates = self._load_templates()
        
        # Notification history (in-memory for development)
        self.notification_history: List[Dict[str, Any]] = []
        
        self.logger.info("📨 Notification service initialized")
    
    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """Load notification templates"""
        return {
            NotificationType.EMAIL_VERIFICATION: {
                "subject": "🕉️ Verify Your DharmaMind Account",
                "html": """
                <html>
                <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; text-align: center;">
                        <h1 style="color: white; margin: 0;">🕉️ DharmaMind</h1>
                        <p style="color: white; margin: 5px 0 0 0;">Your Spiritual AI Companion</p>
                    </div>
                    <div style="padding: 30px 20px;">
                        <h2 style="color: #333;">Welcome to Your Spiritual Journey!</h2>
                        <p style="color: #666; line-height: 1.6;">
                            Namaste! Thank you for joining DharmaMind. To complete your registration, 
                            please verify your email address using the code below:
                        </p>
                        <div style="background: #f8f9fa; border: 2px solid #667eea; border-radius: 8px; padding: 20px; text-align: center; margin: 20px 0;">
                            <h3 style="color: #667eea; margin: 0; font-size: 24px; letter-spacing: 2px;">{verification_code}</h3>
                        </div>
                        <p style="color: #666; line-height: 1.6;">
                            This code will expire in 10 minutes. If you didn't request this verification, 
                            please ignore this email.
                        </p>
                        <div style="margin-top: 30px; padding: 20px; background: #f0f8ff; border-radius: 8px;">
                            <p style="color: #4a90e2; margin: 0; font-style: italic;">
                                "The journey of a thousand miles begins with a single step." - Lao Tzu
                            </p>
                        </div>
                    </div>
                    <div style="background: #f8f9fa; padding: 20px; text-align: center; border-top: 1px solid #e9ecef;">
                        <p style="color: #666; margin: 0; font-size: 12px;">
                            DharmaMind - Bringing Ancient Wisdom to Modern Life
                        </p>
                    </div>
                </body>
                </html>
                """,
                "text": "DharmaMind Verification Code: {verification_code}. This code expires in 10 minutes."
            },
            NotificationType.PASSWORD_RESET: {
                "subject": "🔐 DharmaMind Password Reset",
                "html": """
                <html>
                <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; text-align: center;">
                        <h1 style="color: white; margin: 0;">🕉️ DharmaMind</h1>
                    </div>
                    <div style="padding: 30px 20px;">
                        <h2 style="color: #333;">Password Reset Request</h2>
                        <p style="color: #666; line-height: 1.6;">
                            We received a request to reset your password. Use the code below to reset it:
                        </p>
                        <div style="background: #f8f9fa; border: 2px solid #e74c3c; border-radius: 8px; padding: 20px; text-align: center; margin: 20px 0;">
                            <h3 style="color: #e74c3c; margin: 0; font-size: 24px; letter-spacing: 2px;">{reset_code}</h3>
                        </div>
                        <p style="color: #666; line-height: 1.6;">
                            This code will expire in 15 minutes. If you didn't request this reset, 
                            please ignore this email and your password will remain unchanged.
                        </p>
                    </div>
                </body>
                </html>
                """,
                "text": "DharmaMind Password Reset Code: {reset_code}. This code expires in 15 minutes."
            },
            NotificationType.WELCOME: {
                "subject": "🙏 Welcome to DharmaMind - Your Spiritual Journey Begins",
                "html": """
                <html>
                <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; text-align: center;">
                        <h1 style="color: white; margin: 0;">🕉️ DharmaMind</h1>
                        <p style="color: white; margin: 5px 0 0 0;">Welcome to Your Spiritual AI Companion</p>
                    </div>
                    <div style="padding: 30px 20px;">
                        <h2 style="color: #333;">Namaste, {username}!</h2>
                        <p style="color: #666; line-height: 1.6;">
                            Welcome to DharmaMind! You've taken the first step on a transformative 
                            spiritual journey guided by the wisdom of the ancient Rishis.
                        </p>
                        <div style="background: #f0f8ff; border-radius: 8px; padding: 20px; margin: 20px 0;">
                            <h3 style="color: #4a90e2; margin-top: 0;">What You Can Explore:</h3>
                            <ul style="color: #666; line-height: 1.8;">
                                <li>🧘 Guided meditation with personalized Rishi wisdom</li>
                                <li>💭 Emotional intelligence and healing support</li>
                                <li>📚 Ancient Sanskrit teachings made accessible</li>
                                <li>🌟 Spiritual practices tailored to your journey</li>
                            </ul>
                        </div>
                        <div style="margin-top: 30px; padding: 20px; background: #fff8e1; border-radius: 8px; border-left: 4px solid #ff9800;">
                            <p style="color: #f57c00; margin: 0; font-style: italic;">
                                "यत्र योगेश्वरः कृष्णो यत्र पार्थो धनुर्धरः। तत्र श्रीर्विजयो भूतिर्ध्रुवा नीतिर्मतिर्मम॥"
                            </p>
                            <p style="color: #666; margin: 10px 0 0 0; font-size: 12px;">
                                "Where there is Krishna and Arjuna, there is prosperity, victory, happiness and sound morality." - Bhagavad Gita
                            </p>
                        </div>
                    </div>
                </body>
                </html>
                """,
                "text": "Welcome to DharmaMind, {username}! Begin your spiritual journey with personalized Rishi guidance."
            }
        }
    
    async def send_email(
        self, 
        to_email: str, 
        subject: str, 
        html_content: str, 
        text_content: Optional[str] = None
    ) -> bool:
        """Send email notification"""
        try:
            if not aiosmtplib:
                # Fallback: log the email instead of sending
                self.logger.info(f"📧 [MOCK EMAIL] To: {to_email}, Subject: {subject}")
                self.logger.info(f"📧 [MOCK EMAIL] Content: {text_content or html_content[:100]}...")
                return True
            
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.from_email
            message["To"] = to_email
            
            # Add text and HTML parts
            if text_content:
                text_part = MIMEText(text_content, "plain")
                message.attach(text_part)
            
            html_part = MIMEText(html_content, "html")
            message.attach(html_part)
            
            # Send email
            await aiosmtplib.send(
                message,
                hostname=self.smtp_server,
                port=self.smtp_port,
                start_tls=True,
                username=self.smtp_username,
                password=self.smtp_password,
            )
            
            self.logger.info(f"📧 Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            self.logger.error(f"📧 Failed to send email to {to_email}: {e}")
            return False
    
    async def send_verification_code(self, email: str, code: str) -> bool:
        """Send email verification code"""
        template = self.templates[NotificationType.EMAIL_VERIFICATION]
        
        html_content = template["html"].format(verification_code=code)
        text_content = template["text"].format(verification_code=code)
        
        success = await self.send_email(
            to_email=email,
            subject=template["subject"],
            html_content=html_content,
            text_content=text_content
        )
        
        # Log notification
        self._log_notification(
            notification_type=NotificationType.EMAIL_VERIFICATION,
            recipient=email,
            status=NotificationStatus.SENT if success else NotificationStatus.FAILED,
            data={"code": code}
        )
        
        return success
    
    async def send_password_reset_code(self, email: str, code: str) -> bool:
        """Send password reset code"""
        template = self.templates[NotificationType.PASSWORD_RESET]
        
        html_content = template["html"].format(reset_code=code)
        text_content = template["text"].format(reset_code=code)
        
        success = await self.send_email(
            to_email=email,
            subject=template["subject"],
            html_content=html_content,
            text_content=text_content
        )
        
        self._log_notification(
            notification_type=NotificationType.PASSWORD_RESET,
            recipient=email,
            status=NotificationStatus.SENT if success else NotificationStatus.FAILED,
            data={"code": code}
        )
        
        return success
    
    async def send_welcome_email(self, email: str, username: str) -> bool:
        """Send welcome email to new user"""
        template = self.templates[NotificationType.WELCOME]
        
        html_content = template["html"].format(username=username)
        text_content = template["text"].format(username=username)
        
        success = await self.send_email(
            to_email=email,
            subject=template["subject"],
            html_content=html_content,
            text_content=text_content
        )
        
        self._log_notification(
            notification_type=NotificationType.WELCOME,
            recipient=email,
            status=NotificationStatus.SENT if success else NotificationStatus.FAILED,
            data={"username": username}
        )
        
        return success
    
    def _log_notification(
        self, 
        notification_type: NotificationType,
        recipient: str,
        status: NotificationStatus,
        data: Optional[Dict[str, Any]] = None
    ):
        """Log notification for tracking"""
        log_entry = {
            "id": len(self.notification_history) + 1,
            "type": notification_type.value,
            "recipient": recipient,
            "status": status.value,
            "timestamp": datetime.now(),
            "data": data or {}
        }
        
        self.notification_history.append(log_entry)
        self.logger.info(f"📊 Notification logged: {notification_type.value} to {recipient} - {status.value}")
    
    def get_notification_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent notification history"""
        return self.notification_history[-limit:]

# Global notification service instance
_notification_service: Optional[NotificationService] = None

def get_notification_service() -> NotificationService:
    """Get global notification service instance"""
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service

def create_notification_service() -> NotificationService:
    """Create new notification service instance"""
    return NotificationService()

# Convenience functions for common operations
async def send_verification_code(email: str, code: str) -> bool:
    """Send verification code email"""
    service = get_notification_service()
    return await service.send_verification_code(email, code)

async def send_password_reset_code(email: str, code: str) -> bool:
    """Send password reset code email"""
    service = get_notification_service()
    return await service.send_password_reset_code(email, code)

async def send_welcome_email(email: str, username: str) -> bool:
    """Send welcome email"""
    service = get_notification_service()
    return await service.send_welcome_email(email, username)

# Export commonly used classes and functions
__all__ = [
    'NotificationService',
    'NotificationType',
    'NotificationStatus',
    'NotificationChannel',
    'get_notification_service',
    'create_notification_service',
    'send_verification_code',
    'send_password_reset_code',
    'send_welcome_email'
]
