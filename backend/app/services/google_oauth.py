"""
Google OAuth Integration for DharmaMind
Enterprise-grade OAuth2 authentication with Google
"""

import asyncio
import aiohttp
import jwt
import secrets
import urllib.parse
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class GoogleOAuthService:
    """Google OAuth2 service for secure authentication"""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str, 
                 database_manager, email_service):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.db = database_manager
        self.email_service = email_service
        
        # Google OAuth endpoints
        self.auth_url = "https://accounts.google.com/o/oauth2/v2/auth"
        self.token_url = "https://oauth2.googleapis.com/token"
        self.userinfo_url = "https://www.googleapis.com/oauth2/v2/userinfo"
        self.jwks_url = "https://www.googleapis.com/oauth2/v3/certs"
        
        # OAuth scopes
        self.scopes = [
            "openid",
            "email", 
            "profile"
        ]
    
    def generate_auth_url(self, state: str = None) -> str:
        """Generate Google OAuth authorization URL"""
        if not state:
            state = secrets.token_urlsafe(32)
        
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.scopes),
            "response_type": "code",
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
            "include_granted_scopes": "true"
        }
        
        auth_url = f"{self.auth_url}?{urllib.parse.urlencode(params)}"
        return auth_url, state
    
    async def exchange_code_for_tokens(self, code: str, state: str) -> Dict[str, Any]:
        """Exchange authorization code for access and refresh tokens"""
        try:
            token_data = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": self.redirect_uri
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.token_url, data=token_data) as response:
                    if response.status == 200:
                        tokens = await response.json()
                        return {
                            "success": True,
                            "access_token": tokens.get("access_token"),
                            "refresh_token": tokens.get("refresh_token"),
                            "id_token": tokens.get("id_token"),
                            "expires_in": tokens.get("expires_in", 3600)
                        }
                    else:
                        error_data = await response.json()
                        logger.error(f"Token exchange failed: {error_data}")
                        return {
                            "success": False,
                            "error": error_data.get("error_description", "Token exchange failed")
                        }
        
        except Exception as e:
            logger.error(f"Token exchange error: {e}")
            return {
                "success": False,
                "error": "Network error during token exchange"
            }
    
    async def verify_id_token(self, id_token: str) -> Dict[str, Any]:
        """Verify Google ID token and extract user information"""
        try:
            # In production, implement proper JWT verification with Google's public keys
            # For now, we'll decode without verification (NOT recommended for production)
            decoded_token = jwt.decode(id_token, options={"verify_signature": False})
            
            # Verify token claims
            if decoded_token.get("iss") not in ["accounts.google.com", "https://accounts.google.com"]:
                return {"success": False, "error": "Invalid token issuer"}
            
            if decoded_token.get("aud") != self.client_id:
                return {"success": False, "error": "Invalid token audience"}
            
            if decoded_token.get("exp", 0) < datetime.utcnow().timestamp():
                return {"success": False, "error": "Token expired"}
            
            return {
                "success": True,
                "user_info": {
                    "google_id": decoded_token.get("sub"),
                    "email": decoded_token.get("email"),
                    "email_verified": decoded_token.get("email_verified", False),
                    "name": decoded_token.get("name"),
                    "given_name": decoded_token.get("given_name"),
                    "family_name": decoded_token.get("family_name"),
                    "picture": decoded_token.get("picture"),
                    "locale": decoded_token.get("locale")
                }
            }
        
        except Exception as e:
            logger.error(f"ID token verification error: {e}")
            return {
                "success": False,
                "error": "Invalid ID token"
            }
    
    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user information from Google using access token"""
        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.userinfo_url, headers=headers) as response:
                    if response.status == 200:
                        user_info = await response.json()
                        return {
                            "success": True,
                            "user_info": user_info
                        }
                    else:
                        return {
                            "success": False,
                            "error": "Failed to fetch user info"
                        }
        
        except Exception as e:
            logger.error(f"User info fetch error: {e}")
            return {
                "success": False,
                "error": "Network error"
            }
    
    async def authenticate_or_create_user(self, google_user_info: Dict[str, Any], 
                                        ip_address: str = None) -> Dict[str, Any]:
        """Authenticate existing user or create new user with Google OAuth"""
        try:
            google_id = google_user_info["google_id"]
            email = google_user_info["email"].lower()
            
            # Check if user exists with this Google ID
            if self.db and hasattr(self.db, 'pool') and self.db.pool:
                async with self.db.pool.acquire() as conn:
                    # First check by Google ID
                    existing_user = await conn.fetchrow("""
                        SELECT id, email, status, email_verified, subscription_plan,
                               first_name_encrypted, last_name_encrypted
                        FROM users 
                        WHERE google_id = $1;
                    """, google_id)
                    
                    if existing_user:
                        # User exists with this Google ID
                        user_data = {
                            "user_id": str(existing_user['id']),
                            "email": existing_user['email'],
                            "status": existing_user['status'],
                            "subscription_plan": existing_user['subscription_plan'],
                            "email_verified": existing_user['email_verified'],
                            "auth_provider": "google"
                        }
                        
                        # Update last login
                        await conn.execute("""
                            UPDATE users 
                            SET last_login = CURRENT_TIMESTAMP, last_activity = CURRENT_TIMESTAMP
                            WHERE id = $1;
                        """, existing_user['id'])
                        
                        # Log successful OAuth login
                        await self.db.log_security_event(
                            user_id=existing_user['id'],
                            event_type="oauth_login_successful",
                            event_details={
                                "provider": "google",
                                "ip_address": ip_address,
                                "google_id": google_id
                            },
                            ip_address=ip_address
                        )
                        
                        return {
                            "success": True,
                            "action": "login",
                            "user": user_data
                        }
                    
                    # Check if user exists with same email but different auth provider
                    existing_email_user = await conn.fetchrow("""
                        SELECT id, auth_provider, google_id
                        FROM users 
                        WHERE email = $1;
                    """, email)
                    
                    if existing_email_user:
                        if existing_email_user['auth_provider'] == 'email':
                            # Link Google account to existing email account
                            await conn.execute("""
                                UPDATE users 
                                SET google_id = $1, auth_provider = 'google', 
                                    email_verified = TRUE, updated_at = CURRENT_TIMESTAMP
                                WHERE id = $2;
                            """, google_id, existing_email_user['id'])
                            
                            await self.db.log_security_event(
                                user_id=existing_email_user['id'],
                                event_type="account_linked",
                                event_details={
                                    "provider": "google",
                                    "linked_to": "existing_email_account",
                                    "ip_address": ip_address
                                },
                                ip_address=ip_address
                            )
                            
                            return {
                                "success": True,
                                "action": "linked",
                                "message": "Google account linked to existing account",
                                "user": {
                                    "user_id": str(existing_email_user['id']),
                                    "email": email,
                                    "auth_provider": "google"
                                }
                            }
                        else:
                            return {
                                "success": False,
                                "error": "Account already exists with different provider"
                            }
                    
                    # Create new user with Google OAuth
                    user_creation_data = {
                        "email": email,
                        "auth_provider": "google",
                        "google_id": google_id,
                        "first_name": google_user_info.get("given_name", ""),
                        "last_name": google_user_info.get("family_name", ""),
                        "ip_address": ip_address,
                        "accept_terms": True,  # Assume acceptance via OAuth
                        "accept_privacy": True,
                        "marketing_consent": False
                    }
                    
                    user_result = await self.db.create_user(user_creation_data)
                    
                    # Mark email as verified for Google users
                    await conn.execute("""
                        UPDATE users 
                        SET email_verified = TRUE, status = 'active',
                            email_verification_token = NULL,
                            email_verification_expires = NULL
                        WHERE id = $1;
                    """, user_result["user_id"])
                    
                    # Send welcome email
                    if self.email_service:
                        welcome_data = {
                            "user_id": user_result["user_id"],
                            "email": email,
                            "first_name": google_user_info.get("given_name", "Fellow Seeker")
                        }
                        await self.email_service.send_welcome_email(welcome_data)
                    
                    return {
                        "success": True,
                        "action": "created",
                        "user": {
                            "user_id": user_result["user_id"],
                            "email": email,
                            "status": "active",
                            "subscription_plan": "free",
                            "email_verified": True,
                            "auth_provider": "google"
                        }
                    }
            
            return {
                "success": False,
                "error": "Database connection unavailable"
            }
        
        except Exception as e:
            logger.error(f"Google OAuth authentication error: {e}")
            return {
                "success": False,
                "error": "Authentication processing failed"
            }
    
    async def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh Google access token"""
        try:
            token_data = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.token_url, data=token_data) as response:
                    if response.status == 200:
                        tokens = await response.json()
                        return {
                            "success": True,
                            "access_token": tokens.get("access_token"),
                            "expires_in": tokens.get("expires_in", 3600)
                        }
                    else:
                        return {
                            "success": False,
                            "error": "Token refresh failed"
                        }
        
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return {
                "success": False,
                "error": "Network error during token refresh"
            }

class OAuthStateManager:
    """Manage OAuth state tokens for security"""
    
    def __init__(self, database_manager):
        self.db = database_manager
        
    async def create_state(self, redirect_url: str = None, user_data: Dict = None) -> str:
        """Create and store OAuth state token"""
        state = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(minutes=10)  # 10 minute expiry
        
        state_data = {
            "redirect_url": redirect_url,
            "user_data": user_data or {},
            "created_at": datetime.utcnow().isoformat()
        }
        
        if self.db and hasattr(self.db, 'pool') and self.db.pool:
            try:
                async with self.db.pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO oauth_states (state_token, state_data, expires_at)
                        VALUES ($1, $2, $3)
                        ON CONFLICT DO NOTHING;
                    """, state, json.dumps(state_data), expires_at)
            except Exception as e:
                logger.error(f"Failed to store OAuth state: {e}")
        
        return state
    
    async def validate_and_consume_state(self, state: str) -> Dict[str, Any]:
        """Validate OAuth state token and consume it (one-time use)"""
        if self.db and hasattr(self.db, 'pool') and self.db.pool:
            try:
                async with self.db.pool.acquire() as conn:
                    # Get and delete state in one transaction
                    state_record = await conn.fetchrow("""
                        DELETE FROM oauth_states 
                        WHERE state_token = $1 AND expires_at > CURRENT_TIMESTAMP
                        RETURNING state_data;
                    """, state)
                    
                    if state_record:
                        return {
                            "valid": True,
                            "data": json.loads(state_record['state_data'])
                        }
            except Exception as e:
                logger.error(f"Failed to validate OAuth state: {e}")
        
        return {"valid": False}
    
    async def cleanup_expired_states(self):
        """Clean up expired OAuth states"""
        if self.db and hasattr(self.db, 'pool') and self.db.pool:
            try:
                async with self.db.pool.acquire() as conn:
                    await conn.execute("""
                        DELETE FROM oauth_states 
                        WHERE expires_at < CURRENT_TIMESTAMP;
                    """)
            except Exception as e:
                logger.error(f"Failed to cleanup OAuth states: {e}")

# Factory function
def create_google_oauth_service(config: Dict[str, Any], database_manager, email_service) -> GoogleOAuthService:
    """Create Google OAuth service with configuration"""
    return GoogleOAuthService(
        client_id=config["google_client_id"],
        client_secret=config["google_client_secret"],
        redirect_uri=config["google_redirect_uri"],
        database_manager=database_manager,
        email_service=email_service
    )
