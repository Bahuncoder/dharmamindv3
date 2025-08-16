"""
Enterprise Email Service for DharmaMind
Secure email verification, notifications, and communication tracking
"""

import asyncio
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formatdate, make_msgid
from typing import Dict, Any, Optional, List
import logging
import json
import os
from datetime import datetime, timedelta
import secrets
import uuid

logger = logging.getLogger(__name__)

class EmailProvider:
    """Base email provider interface"""
    
    async def send_email(self, to_email: str, subject: str, html_content: str, 
                        text_content: str = None, template_data: Dict = None) -> Dict[str, Any]:
        raise NotImplementedError

class SMTPEmailProvider(EmailProvider):
    """SMTP email provider for enterprise email delivery"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, 
                 password: str, use_tls: bool = True, from_email: str = None):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.from_email = from_email or username
        
    async def send_email(self, to_email: str, subject: str, html_content: str,
                        text_content: str = None, template_data: Dict = None) -> Dict[str, Any]:
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = to_email
            msg['Date'] = formatdate(localtime=True)
            msg['Message-ID'] = make_msgid()
            
            # Add text content
            if text_content:
                text_part = MIMEText(text_content, 'plain', 'utf-8')
                msg.attach(text_part)
            
            # Add HTML content
            html_part = MIMEText(html_content, 'html', 'utf-8')
            msg.attach(html_part)
            
            # Send email
            context = ssl.create_default_context() if self.use_tls else None
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls(context=context)
                server.login(self.username, self.password)
                result = server.send_message(msg)
                
            return {
                "success": True,
                "message_id": msg['Message-ID'],
                "to_email": to_email,
                "sent_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return {
                "success": False,
                "error": str(e),
                "to_email": to_email
            }

class EmailService:
    """Enterprise email service with templates and tracking"""
    
    def __init__(self, email_provider: EmailProvider, database_manager, 
                 base_url: str = "https://dharmamind.ai"):
        self.email_provider = email_provider
        self.db = database_manager
        self.base_url = base_url
        
        # Email templates
        self.templates = {
            "email_verification": {
                "subject": "Verify Your DharmaMind Account",
                "template": self._get_verification_template()
            },
            "password_reset": {
                "subject": "Reset Your DharmaMind Password",
                "template": self._get_password_reset_template()
            },
            "welcome": {
                "subject": "Welcome to DharmaMind - Your Spiritual Journey Begins",
                "template": self._get_welcome_template()
            },
            "subscription_confirmation": {
                "subject": "Subscription Confirmed - Welcome to Premium DharmaMind",
                "template": self._get_subscription_template()
            },
            "security_alert": {
                "subject": "Security Alert - DharmaMind Account Activity",
                "template": self._get_security_alert_template()
            }
        }
    
    def _get_verification_template(self) -> str:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Verify Your DharmaMind Account</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f8f9fa; }
                .container { max-width: 600px; margin: 0 auto; background-color: white; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 20px; text-align: center; }
                .header h1 { color: white; margin: 0; font-size: 28px; }
                .header .logo { font-size: 36px; margin-bottom: 10px; }
                .content { padding: 40px 30px; }
                .content h2 { color: #333; margin-bottom: 20px; }
                .content p { color: #666; line-height: 1.6; margin-bottom: 20px; }
                .verify-button { display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                               color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; 
                               font-weight: bold; margin: 20px 0; }
                .verify-button:hover { opacity: 0.9; }
                .footer { background-color: #f8f9fa; padding: 30px; text-align: center; color: #666; font-size: 14px; }
                .security-note { background-color: #f8f9fa; border-left: 4px solid #667eea; padding: 15px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">🕉️</div>
                    <h1>DharmaMind</h1>
                    <p style="color: #e8f4fd; margin: 10px 0 0 0;">Universal Wisdom AI Platform</p>
                </div>
                
                <div class="content">
                    <h2>Welcome to Your Spiritual Journey</h2>
                    
                    <p>Dear {{first_name}},</p>
                    
                    <p>Thank you for joining DharmaMind, where ancient wisdom meets modern AI to guide your spiritual journey. 
                       To complete your registration and access your personalized spiritual companion, please verify your email address.</p>
                    
                    <div style="text-align: center;">
                        <a href="{{verification_url}}" class="verify-button">Verify My Email Address</a>
                    </div>
                    
                    <p>This verification link will expire in 24 hours for your security.</p>
                    
                    <div class="security-note">
                        <strong>Security Note:</strong> If you didn't create a DharmaMind account, please ignore this email. 
                        Your email address will not be added to our system.
                    </div>
                    
                    <p>Once verified, you'll have access to:</p>
                    <ul>
                        <li>🧠 Personalized spiritual AI guidance</li>
                        <li>📚 Access to universal wisdom and teachings</li>
                        <li>🎯 Dharmic compliance and ethical AI responses</li>
                        <li>🔒 Secure, private conversations</li>
                        <li>📈 Track your spiritual growth journey</li>
                    </ul>
                    
                    <p>If you have any questions, our support team is here to help at support@dharmamind.ai</p>
                    
                    <p>May your journey be filled with wisdom and peace.</p>
                    
                    <p>With gratitude,<br>The DharmaMind Team</p>
                </div>
                
                <div class="footer">
                    <p>DharmaMind - AI with Soul powered by Dharma</p>
                    <p>This email was sent to {{email}}. If you didn't request this, please ignore this message.</p>
                    <p>© 2025 DharmaMind. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_password_reset_template(self) -> str:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Reset Your DharmaMind Password</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f8f9fa; }
                .container { max-width: 600px; margin: 0 auto; background-color: white; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 20px; text-align: center; }
                .header h1 { color: white; margin: 0; font-size: 28px; }
                .header .logo { font-size: 36px; margin-bottom: 10px; }
                .content { padding: 40px 30px; }
                .reset-button { display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                              color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; 
                              font-weight: bold; margin: 20px 0; }
                .security-warning { background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; 
                                  padding: 15px; margin: 20px 0; }
                .footer { background-color: #f8f9fa; padding: 30px; text-align: center; color: #666; font-size: 14px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">🕉️</div>
                    <h1>DharmaMind</h1>
                </div>
                
                <div class="content">
                    <h2>Password Reset Request</h2>
                    
                    <p>Dear {{first_name}},</p>
                    
                    <p>We received a request to reset the password for your DharmaMind account ({{email}}).</p>
                    
                    <div style="text-align: center;">
                        <a href="{{reset_url}}" class="reset-button">Reset My Password</a>
                    </div>
                    
                    <p>This password reset link will expire in 1 hour for your security.</p>
                    
                    <div class="security-warning">
                        <strong>⚠️ Security Notice:</strong> If you didn't request this password reset, please ignore this email. 
                        Your password will remain unchanged. Consider enabling two-factor authentication for additional security.
                    </div>
                    
                    <p>For your account security, this request was made from:</p>
                    <ul>
                        <li>IP Address: {{ip_address}}</li>
                        <li>Time: {{request_time}}</li>
                        <li>Location: {{location}}</li>
                    </ul>
                    
                    <p>If you need assistance, contact us at support@dharmamind.ai</p>
                    
                    <p>Stay secure,<br>The DharmaMind Security Team</p>
                </div>
                
                <div class="footer">
                    <p>© 2025 DharmaMind. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_welcome_template(self) -> str:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Welcome to DharmaMind</title>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">🕉️</div>
                    <h1>Welcome to DharmaMind</h1>
                </div>
                
                <div class="content">
                    <h2>Your Spiritual Journey Begins Now</h2>
                    
                    <p>Dear {{first_name}},</p>
                    
                    <p>Welcome to DharmaMind! Your email has been verified and your account is now active.</p>
                    
                    <p>🌟 <strong>What's Next?</strong></p>
                    <ul>
                        <li>Start your first spiritual conversation with our AI companion</li>
                        <li>Explore our wisdom library with teachings from various traditions</li>
                        <li>Set up your personal spiritual preferences</li>
                        <li>Consider upgrading to unlock advanced features</li>
                    </ul>
                    
                    <div style="text-align: center;">
                        <a href="{{dashboard_url}}" class="verify-button">Start My Journey</a>
                    </div>
                    
                    <p>May your path be illuminated with wisdom and compassion.</p>
                    
                    <p>With gratitude,<br>The DharmaMind Team</p>
                </div>
                
                <div class="footer">
                    <p>© 2025 DharmaMind. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_subscription_template(self) -> str:
        return """
        <!DOCTYPE html>
        <html>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">🕉️</div>
                    <h1>Subscription Confirmed</h1>
                </div>
                
                <div class="content">
                    <h2>Welcome to Premium DharmaMind</h2>
                    
                    <p>Dear {{first_name}},</p>
                    
                    <p>Your {{plan_name}} subscription has been confirmed! You now have access to premium features.</p>
                    
                    <p><strong>Your Plan Details:</strong></p>
                    <ul>
                        <li>Plan: {{plan_name}}</li>
                        <li>Billing: ${{amount}} {{interval}}</li>
                        <li>Next billing date: {{next_billing_date}}</li>
                    </ul>
                    
                    <p><strong>Premium Features Unlocked:</strong></p>
                    <ul>
                        <li>🚀 Unlimited spiritual conversations</li>
                        <li>🎯 Advanced dharmic analysis</li>
                        <li>📚 Complete wisdom library access</li>
                        <li>🔮 Personalized spiritual insights</li>
                        <li>💬 Priority support</li>
                    </ul>
                    
                    <div style="text-align: center;">
                        <a href="{{dashboard_url}}" class="verify-button">Access Premium Features</a>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_security_alert_template(self) -> str:
        return """
        <!DOCTYPE html>
        <html>
        <body>
            <div class="container">
                <div class="header" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);">
                    <div class="logo">🔒</div>
                    <h1>Security Alert</h1>
                </div>
                
                <div class="content">
                    <h2>Account Security Notification</h2>
                    
                    <p>Dear {{first_name}},</p>
                    
                    <p>We detected {{event_type}} on your DharmaMind account:</p>
                    
                    <div class="security-warning">
                        <strong>Event Details:</strong>
                        <ul>
                            <li>Activity: {{event_description}}</li>
                            <li>Time: {{event_time}}</li>
                            <li>IP Address: {{ip_address}}</li>
                            <li>Location: {{location}}</li>
                        </ul>
                    </div>
                    
                    <p>If this was you, no action is needed. If you don't recognize this activity, please:</p>
                    <ol>
                        <li>Change your password immediately</li>
                        <li>Enable two-factor authentication</li>
                        <li>Review your account activity</li>
                        <li>Contact support if you need assistance</li>
                    </ol>
                    
                    <div style="text-align: center;">
                        <a href="{{security_url}}" class="reset-button">Secure My Account</a>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
    
    async def send_verification_email(self, user_data: Dict[str, Any], verification_token: str) -> bool:
        """Send email verification"""
        verification_url = f"{self.base_url}/auth/verify-email?token={verification_token}"
        
        template_data = {
            "first_name": user_data.get("first_name", "Fellow Seeker"),
            "email": user_data["email"],
            "verification_url": verification_url
        }
        
        html_content = self._render_template("email_verification", template_data)
        
        result = await self.email_provider.send_email(
            to_email=user_data["email"],
            subject=self.templates["email_verification"]["subject"],
            html_content=html_content,
            template_data=template_data
        )
        
        # Log email communication
        if self.db:
            await self._log_email_communication(
                user_id=user_data.get("user_id"),
                email_type="email_verification",
                email_address=user_data["email"],
                status="sent" if result["success"] else "failed",
                template_data=template_data
            )
        
        return result["success"]
    
    async def send_password_reset_email(self, user_data: Dict[str, Any], reset_token: str, 
                                      ip_address: str = None) -> bool:
        """Send password reset email"""
        reset_url = f"{self.base_url}/auth/reset-password?token={reset_token}"
        
        template_data = {
            "first_name": user_data.get("first_name", "Fellow Seeker"),
            "email": user_data["email"],
            "reset_url": reset_url,
            "ip_address": ip_address or "Unknown",
            "request_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "location": "Unknown"  # Could integrate with IP geolocation service
        }
        
        html_content = self._render_template("password_reset", template_data)
        
        result = await self.email_provider.send_email(
            to_email=user_data["email"],
            subject=self.templates["password_reset"]["subject"],
            html_content=html_content,
            template_data=template_data
        )
        
        return result["success"]
    
    async def send_welcome_email(self, user_data: Dict[str, Any]) -> bool:
        """Send welcome email after verification"""
        dashboard_url = f"{self.base_url}/dashboard"
        
        template_data = {
            "first_name": user_data.get("first_name", "Fellow Seeker"),
            "dashboard_url": dashboard_url
        }
        
        html_content = self._render_template("welcome", template_data)
        
        result = await self.email_provider.send_email(
            to_email=user_data["email"],
            subject=self.templates["welcome"]["subject"],
            html_content=html_content,
            template_data=template_data
        )
        
        return result["success"]
    
    async def send_security_alert(self, user_data: Dict[str, Any], event_type: str, 
                                 event_details: Dict[str, Any]) -> bool:
        """Send security alert email"""
        security_url = f"{self.base_url}/account/security"
        
        template_data = {
            "first_name": user_data.get("first_name", "Fellow Seeker"),
            "event_type": event_type,
            "event_description": event_details.get("description", event_type),
            "event_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "ip_address": event_details.get("ip_address", "Unknown"),
            "location": event_details.get("location", "Unknown"),
            "security_url": security_url
        }
        
        html_content = self._render_template("security_alert", template_data)
        
        result = await self.email_provider.send_email(
            to_email=user_data["email"],
            subject=self.templates["security_alert"]["subject"],
            html_content=html_content,
            template_data=template_data
        )
        
        return result["success"]
    
    def _render_template(self, template_name: str, data: Dict[str, Any]) -> str:
        """Render email template with data"""
        template = self.templates[template_name]["template"]
        
        # Simple template rendering (in production, use Jinja2 or similar)
        for key, value in data.items():
            template = template.replace(f"{{{{{key}}}}}", str(value))
        
        return template
    
    async def _log_email_communication(self, user_id: str, email_type: str, 
                                     email_address: str, status: str, 
                                     template_data: Dict = None):
        """Log email communication for tracking"""
        try:
            if self.db and hasattr(self.db, 'pool') and self.db.pool:
                async with self.db.pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO email_communications (
                            user_id, email_type, email_address, status, template_data
                        ) VALUES ($1, $2, $3, $4, $5);
                    """, user_id, email_type, email_address, status, json.dumps(template_data))
        except Exception as e:
            logger.error(f"Failed to log email communication: {e}")

# Email service configuration
def create_email_service(database_manager, config: Dict[str, Any]) -> EmailService:
    """Factory function to create email service with provider"""
    
    if config.get("provider") == "smtp":
        email_provider = SMTPEmailProvider(
            smtp_server=config["smtp_server"],
            smtp_port=config["smtp_port"],
            username=config["username"],
            password=config["password"],
            use_tls=config.get("use_tls", True),
            from_email=config.get("from_email")
        )
    else:
        raise ValueError(f"Unsupported email provider: {config.get('provider')}")
    
    return EmailService(
        email_provider=email_provider,
        database_manager=database_manager,
        base_url=config.get("base_url", "https://dharmamind.ai")
    )
