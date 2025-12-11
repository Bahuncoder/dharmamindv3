<<<<<<< HEAD
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
    user_name: Optional[str] = None
) -> Optional[str]:
    """Send verification code to email and return the code"""
    try:
        service = await get_notification_service()
        code = service.generate_verification_code()
        
        success = await service.send_verification_email(email, code, user_name)
        
        if success:
            return code
        else:
            return None
            
    except Exception as e:
        logger.error(f"Failed to send verification code to {email}: {e}")
        return None

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
=======
"""
Email and SMS Service for DharmaMind
===================================

Handles sending verification codes via email and SMS
"""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import os

logger = logging.getLogger(__name__)

# Mock Twilio for development (avoid import errors)
try:
    from twilio.rest import Client
except ImportError:
    Client = None

logger = logging.getLogger(__name__)


class NotificationService:
    """Service for sending email and SMS notifications"""
    
    def __init__(self):
        # Email configuration
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        
        # SMS configuration (Twilio)
        self.twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_phone = os.getenv("TWILIO_PHONE_NUMBER")
        
        # Initialize Twilio client if credentials are available
        self.twilio_client = None
        if self.twilio_sid and self.twilio_token and Client is not None:
            try:
                self.twilio_client = Client(self.twilio_sid, self.twilio_token)
            except Exception as e:
                logger.warning(f"Failed to initialize Twilio client: {e}")
    
    async def send_verification_email(self, email: str, code: str, name: str = "User") -> bool:
        """Send verification code via email"""
        if not self.smtp_user or not self.smtp_password:
            logger.warning("Email credentials not configured, using mock sending")
            print(f"🔔 [EMAIL MOCK] Verification code {code} sent to {email}")
            return True
        
        try:
            # Create email content
            subject = "🕉️ DharmaMind - Verify Your Account"
            
            html_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background: linear-gradient(135deg, #ff7b00 0%, #ff8c00 100%); padding: 20px; text-align: center;">
                    <h1 style="color: white; margin: 0;">🕉️ DharmaMind</h1>
                    <p style="color: white; margin: 5px 0;">Universal Wisdom Platform</p>
                </div>
                
                <div style="padding: 30px; background: #f9f9f9;">
                    <h2 style="color: #333;">Welcome, {name}!</h2>
                    <p style="color: #666; line-height: 1.6;">
                        Thank you for joining DharmaMind. To complete your registration and 
                        unlock the wisdom of dharma, please verify your email with the code below:
                    </p>
                    
                    <div style="background: white; padding: 20px; border-radius: 8px; text-align: center; margin: 20px 0;">
                        <h3 style="color: #ff7b00; font-size: 32px; letter-spacing: 8px; margin: 0;">{code}</h3>
                        <p style="color: #999; font-size: 12px;">This code expires in 10 minutes</p>
                    </div>
                    
                    <p style="color: #666; line-height: 1.6;">
                        If you didn't create a DharmaMind account, please ignore this email.
                    </p>
                    
                    <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd;">
                        <p style="color: #999; font-size: 12px; text-align: center;">
                            DharmaMind - AI with Soul powered by Dharma<br>
                            This is an automated message, please do not reply.
                        </p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            text_body = f"""
            🕉️ DharmaMind - Verify Your Account
            
            Welcome, {name}!
            
            Thank you for joining DharmaMind. To complete your registration and 
            unlock the wisdom of dharma, please verify your email with this code:
            
            Verification Code: {code}
            
            This code expires in 10 minutes.
            
            If you didn't create a DharmaMind account, please ignore this email.
            
            DharmaMind - AI with Soul powered by Dharma
            """
            
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.smtp_user
            msg["To"] = email
            
            # Add both text and HTML parts
            part1 = MIMEText(text_body, "plain")
            part2 = MIMEText(html_body, "html")
            
            msg.attach(part1)
            msg.attach(part2)
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"✅ Verification email sent to {email}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send verification email to {email}: {e}")
            return False
    
    async def send_verification_sms(self, phone: str, code: str, name: str = "User") -> bool:
        """Send verification code via SMS"""
        if not self.twilio_client:
            logger.warning("SMS credentials not configured, using mock sending")
            print(f"🔔 [SMS MOCK] Verification code {code} sent to {phone}")
            return True
        
        try:
            message_body = f"""
🕉️ DharmaMind Verification

Hello {name},

Your verification code is: {code}

This code expires in 10 minutes.

If you didn't request this, ignore this message.
            """.strip()
            
            message = self.twilio_client.messages.create(
                body=message_body,
                from_=self.twilio_phone,
                to=phone
            )
            
            logger.info(f"✅ Verification SMS sent to {phone}, SID: {message.sid}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send verification SMS to {phone}: {e}")
            return False
    
    async def send_verification_code(self, email: Optional[str], phone: Optional[str], code: str, name: str = "User") -> bool:
        """Send verification code via email or SMS"""
        if email:
            return await self.send_verification_email(email, code, name)
        elif phone:
            return await self.send_verification_sms(phone, code, name)
        else:
            logger.error("No email or phone provided for verification")
            return False


# Global service instance
notification_service = NotificationService()


async def send_verification_code(email: Optional[str], phone: Optional[str], code: str, name: str = "User") -> bool:
    """Global function to send verification codes"""
    return await notification_service.send_verification_code(email, phone, code, name)
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
