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
