"""
🗄️ DharmaMind Data Management Service

Comprehensive data handling for:
- User information and profiles
- Conversation history and chat data
- Session management
- Security events and audit logs
- Data encryption and privacy protection

Features:
- Secure data storage with encryption
- Conversation persistence
- User data analytics
- Privacy compliance (GDPR ready)
- Data export and import
- Audit trail maintenance
"""

import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from pathlib import Path

from services.advanced_security import (
    security_manager, 
    encrypt_sensitive_data, 
    decrypt_sensitive_data,
    audit_security_event,
    SecurityLevel,
    EncryptionType
)
from services.secret_manager import get_secret

logger = logging.getLogger(__name__)


class DataType(str, Enum):
    """Types of data stored in the system"""
    USER_PROFILE = "user_profile"
    CONVERSATION = "conversation"
    CHAT_MESSAGE = "chat_message"
    SESSION = "session"
    SECURITY_EVENT = "security_event"
    USER_PREFERENCE = "user_preference"
    ANALYTICS = "analytics"


class PrivacyLevel(str, Enum):
    """Privacy levels for data classification"""
    PUBLIC = "public"           # Can be shared freely
    INTERNAL = "internal"       # Internal use only
    CONFIDENTIAL = "confidential"  # Encrypted storage required
    RESTRICTED = "restricted"   # Highest security level


@dataclass
class ChatMessage:
    """Individual chat message structure"""
    message_id: str
    user_id: str
    conversation_id: str
    message: str
    response: str
    timestamp: datetime
    message_type: str = "user_query"  # user_query, system_response, etc.
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "message": self.message,
            "response": self.response,
            "timestamp": self.timestamp.isoformat(),
            "message_type": self.message_type,
            "metadata": self.metadata or {}
        }


@dataclass
class Conversation:
    """Complete conversation structure"""
    conversation_id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[ChatMessage]
    tags: List[str] = None
    summary: str = None
    is_archived: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "messages": [msg.to_dict() for msg in self.messages],
            "tags": self.tags or [],
            "summary": self.summary,
            "is_archived": self.is_archived,
            "message_count": len(self.messages)
        }


@dataclass
class UserData:
    """Complete user data structure"""
    user_id: str
    email: str
    profile: Dict[str, Any]
    conversations: List[str]  # conversation IDs
    preferences: Dict[str, Any]
    analytics: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    last_activity: datetime
    privacy_settings: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "email": self.email,
            "profile": self.profile,
            "conversations": self.conversations,
            "preferences": self.preferences,
            "analytics": self.analytics,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "privacy_settings": self.privacy_settings or {}
        }


class DataManager:
    """🗄️ Comprehensive data management system"""
    
    def __init__(self):
        self.storage_path = Path("data/user_data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory caches for performance
        self.users_cache: Dict[str, UserData] = {}
        self.conversations_cache: Dict[str, Conversation] = {}
        self.sessions_cache: Dict[str, Dict[str, Any]] = {}
        
        # Data classification for privacy compliance with enhanced security
        self.data_classification = {
            "email": SecurityLevel.CONFIDENTIAL,
            "password_hash": SecurityLevel.RESTRICTED,
            "phone": SecurityLevel.CONFIDENTIAL,
            "chat_messages": SecurityLevel.INTERNAL,
            "preferences": SecurityLevel.INTERNAL,
            "analytics": SecurityLevel.INTERNAL,
            "profile": SecurityLevel.CONFIDENTIAL,
            "privacy_settings": SecurityLevel.RESTRICTED
        }
        
        logger.info("🗄️ Data Manager initialized")
    
    # ===============================
    # USER DATA MANAGEMENT
    # ===============================
    
    async def create_user(self, user_data: Dict[str, Any]) -> UserData:
        """Create new user with comprehensive data structure"""
        user_id = str(uuid.uuid4())
        current_time = datetime.utcnow()
        
        # Create user data structure
        user = UserData(
            user_id=user_id,
            email=user_data["email"],
            profile={
                "first_name": user_data.get("first_name"),
                "last_name": user_data.get("last_name"),
                "phone": user_data.get("phone"),
                "timezone": user_data.get("timezone", "UTC"),
                "subscription_plan": user_data.get("subscription_plan", "free"),
                "status": "active",
                "email_verified": True,
                "auth_provider": user_data.get("auth_provider", "email")
            },
            conversations=[],
            preferences={
                "notifications": True,
                "theme": "system",
                "language": "en",
                "privacy_mode": False,
                "data_retention_days": 365
            },
            analytics={
                "login_count": 0,
                "total_conversations": 0,
                "total_messages": 0,
                "last_login": None,
                "favorite_topics": []
            },
            created_at=current_time,
            updated_at=current_time,
            last_activity=current_time,
            privacy_settings={
                "share_analytics": False,
                "export_data_allowed": True,
                "marketing_consent": user_data.get("marketing_consent", False)
            }
        )
        
        # Cache and persist with enhanced security
        self.users_cache[user_id] = user
        await self._persist_user_data(user)
        
        # Audit log for user creation
        await audit_security_event(
            event_type="user_creation",
            action="create_user", 
            resource_id=user_id,
            result="success",
            user_id=user_id,
            metadata={"email": user.email, "auth_provider": user.profile.get("auth_provider")}
        )
        
        logger.info(f"Created user: {user_id} ({user.email})")
        return user
    
    async def get_user(self, user_id: str) -> Optional[UserData]:
        """Get user data by ID"""
        if user_id in self.users_cache:
            return self.users_cache[user_id]
        
        # Try to load from persistent storage
        user_data = await self._load_user_data(user_id)
        if user_data:
            self.users_cache[user_id] = user_data
        
        return user_data
    
    async def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user data"""
        user = await self.get_user(user_id)
        if not user:
            return False
        
        # Update profile fields
        if "profile" in updates:
            user.profile.update(updates["profile"])
        
        # Update preferences
        if "preferences" in updates:
            user.preferences.update(updates["preferences"])
        
        # Update privacy settings
        if "privacy_settings" in updates:
            user.privacy_settings.update(updates["privacy_settings"])
        
        user.updated_at = datetime.utcnow()
        
        # Persist changes
        await self._persist_user_data(user)
        
        logger.info(f"Updated user: {user_id}")
        return True
    
    # ===============================
    # CONVERSATION MANAGEMENT
    # ===============================
    
    async def create_conversation(self, user_id: str, title: str = None) -> Conversation:
        """Create new conversation for user"""
        conversation_id = f"conv_{user_id}_{int(datetime.utcnow().timestamp())}"
        current_time = datetime.utcnow()
        
        conversation = Conversation(
            conversation_id=conversation_id,
            user_id=user_id,
            title=title or f"Conversation {current_time.strftime('%Y-%m-%d %H:%M')}",
            created_at=current_time,
            updated_at=current_time,
            messages=[]
        )
        
        # Cache and persist
        self.conversations_cache[conversation_id] = conversation
        await self._persist_conversation(conversation)
        
        # Update user's conversation list
        user = await self.get_user(user_id)
        if user:
            user.conversations.append(conversation_id)
            user.analytics["total_conversations"] += 1
            await self._persist_user_data(user)
        
        logger.info(f"Created conversation: {conversation_id} for user: {user_id}")
        return conversation
    
    async def add_message_to_conversation(self, conversation_id: str, user_message: str, 
                                        ai_response: str, metadata: Dict[str, Any] = None) -> ChatMessage:
        """Add message to conversation"""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation not found: {conversation_id}")
        
        message_id = str(uuid.uuid4())
        current_time = datetime.utcnow()
        
        chat_message = ChatMessage(
            message_id=message_id,
            user_id=conversation.user_id,
            conversation_id=conversation_id,
            message=user_message,
            response=ai_response,
            timestamp=current_time,
            metadata=metadata
        )
        
        conversation.messages.append(chat_message)
        conversation.updated_at = current_time
        
        # Update user analytics
        user = await self.get_user(conversation.user_id)
        if user:
            user.analytics["total_messages"] += 1
            user.last_activity = current_time
            await self._persist_user_data(user)
        
        # Persist conversation
        await self._persist_conversation(conversation)
        
        logger.info(f"Added message to conversation: {conversation_id}")
        return chat_message
    
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID"""
        if conversation_id in self.conversations_cache:
            return self.conversations_cache[conversation_id]
        
        # Try to load from persistent storage
        conversation = await self._load_conversation(conversation_id)
        if conversation:
            self.conversations_cache[conversation_id] = conversation
        
        return conversation
    
    async def get_user_conversations(self, user_id: str, limit: int = 50, 
                                   include_archived: bool = False) -> List[Conversation]:
        """Get all conversations for a user"""
        user = await self.get_user(user_id)
        if not user:
            return []
        
        conversations = []
        for conv_id in user.conversations[-limit:]:  # Get most recent
            conversation = await self.get_conversation(conv_id)
            if conversation and (include_archived or not conversation.is_archived):
                conversations.append(conversation)
        
        return sorted(conversations, key=lambda x: x.updated_at, reverse=True)
    
    # ===============================
    # SESSION MANAGEMENT
    # ===============================
    
    async def create_session(self, user_id: str, session_data: Dict[str, Any]) -> str:
        """Create user session"""
        session_token = f"session_{uuid.uuid4().hex}"
        current_time = datetime.utcnow()
        
        session = {
            "session_token": session_token,
            "user_id": user_id,
            "created_at": current_time.isoformat(),
            "last_activity": current_time.isoformat(),
            "ip_address": session_data.get("ip_address"),
            "user_agent": session_data.get("user_agent"),
            "expires_at": (current_time + timedelta(hours=24)).isoformat(),
            "is_active": True
        }
        
        self.sessions_cache[session_token] = session
        
        # Update user login analytics
        user = await self.get_user(user_id)
        if user:
            user.analytics["login_count"] += 1
            user.analytics["last_login"] = current_time.isoformat()
            user.last_activity = current_time
            await self._persist_user_data(user)
        
        logger.info(f"Created session: {session_token} for user: {user_id}")
        return session_token
    
    async def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Validate and refresh session"""
        if session_token not in self.sessions_cache:
            return None
        
        session = self.sessions_cache[session_token]
        
        # Check expiry
        expires_at = datetime.fromisoformat(session["expires_at"])
        if datetime.utcnow() > expires_at:
            del self.sessions_cache[session_token]
            return None
        
        # Update last activity
        session["last_activity"] = datetime.utcnow().isoformat()
        
        return session
    
    # ===============================
    # DATA ANALYTICS
    # ===============================
    
    async def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user analytics"""
        user = await self.get_user(user_id)
        if not user:
            return {}
        
        conversations = await self.get_user_conversations(user_id)
        
        # Calculate conversation analytics
        total_messages = sum(len(conv.messages) for conv in conversations)
        avg_messages_per_conversation = total_messages / len(conversations) if conversations else 0
        
        # Topic analysis (simplified)
        all_messages = []
        for conv in conversations:
            all_messages.extend([msg.message for msg in conv.messages])
        
        return {
            "user_id": user_id,
            "basic_stats": user.analytics,
            "conversation_stats": {
                "total_conversations": len(conversations),
                "total_messages": total_messages,
                "avg_messages_per_conversation": round(avg_messages_per_conversation, 2),
                "archived_conversations": sum(1 for conv in conversations if conv.is_archived)
            },
            "activity_timeline": {
                "account_created": user.created_at.isoformat(),
                "last_activity": user.last_activity.isoformat(),
                "days_active": (datetime.utcnow() - user.created_at).days
            },
            "recent_activity": {
                "conversations_this_week": len([
                    conv for conv in conversations 
                    if conv.updated_at > datetime.utcnow() - timedelta(days=7)
                ]),
                "messages_this_week": sum(
                    len([msg for msg in conv.messages 
                        if msg.timestamp > datetime.utcnow() - timedelta(days=7)])
                    for conv in conversations
                )
            }
        }
    
    # ===============================
    # DATA PRIVACY & EXPORT
    # ===============================
    
    async def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all user data for GDPR compliance"""
        user = await self.get_user(user_id)
        if not user:
            return {}
        
        conversations = await self.get_user_conversations(user_id, limit=1000, include_archived=True)
        
        export_data = {
            "export_info": {
                "export_date": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "data_format": "JSON"
            },
            "user_profile": user.to_dict(),
            "conversations": [conv.to_dict() for conv in conversations],
            "analytics": await self.get_user_analytics(user_id)
        }
        
        logger.info(f"Exported data for user: {user_id}")
        return export_data
    
    async def delete_user_data(self, user_id: str, keep_analytics: bool = False) -> bool:
        """Delete user data (GDPR right to be forgotten)"""
        user = await self.get_user(user_id)
        if not user:
            return False
        
        # Delete conversations
        for conv_id in user.conversations:
            if conv_id in self.conversations_cache:
                del self.conversations_cache[conv_id]
            await self._delete_conversation_file(conv_id)
        
        # Delete user data
        if user_id in self.users_cache:
            del self.users_cache[user_id]
        await self._delete_user_file(user_id)
        
        # Remove sessions
        sessions_to_remove = [
            token for token, session in self.sessions_cache.items()
            if session.get("user_id") == user_id
        ]
        for token in sessions_to_remove:
            del self.sessions_cache[token]
        
        logger.info(f"Deleted data for user: {user_id}")
        return True
    
    # ===============================
    # PRIVATE PERSISTENCE METHODS
    # ===============================
    
    async def _persist_user_data(self, user: UserData):
        """Persist user data to file"""
        user_file = self.storage_path / f"user_{user.user_id}.json"
        
        # Encrypt sensitive data
        user_dict = user.to_dict()
        encrypted_data = await self._encrypt_sensitive_fields(user_dict)
        
        user_file.write_text(json.dumps(encrypted_data, indent=2))
    
    async def _persist_conversation(self, conversation: Conversation):
        """Persist conversation to file"""
        conv_file = self.storage_path / f"conv_{conversation.conversation_id}.json"
        conv_file.write_text(json.dumps(conversation.to_dict(), indent=2))
    
    async def _load_user_data(self, user_id: str) -> Optional[UserData]:
        """Load user data from file"""
        user_file = self.storage_path / f"user_{user_id}.json"
        if not user_file.exists():
            return None
        
        try:
            encrypted_data = json.loads(user_file.read_text())
            user_dict = await self._decrypt_sensitive_fields(encrypted_data)
            
            # Convert back to UserData object
            user_dict["created_at"] = datetime.fromisoformat(user_dict["created_at"])
            user_dict["updated_at"] = datetime.fromisoformat(user_dict["updated_at"])
            user_dict["last_activity"] = datetime.fromisoformat(user_dict["last_activity"])
            
            return UserData(**user_dict)
        except Exception as e:
            logger.error(f"Error loading user data: {e}")
            return None
    
    async def _load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Load conversation from file"""
        conv_file = self.storage_path / f"conv_{conversation_id}.json"
        if not conv_file.exists():
            return None
        
        try:
            conv_dict = json.loads(conv_file.read_text())
            
            # Convert timestamps
            conv_dict["created_at"] = datetime.fromisoformat(conv_dict["created_at"])
            conv_dict["updated_at"] = datetime.fromisoformat(conv_dict["updated_at"])
            
            # Convert messages
            messages = []
            for msg_dict in conv_dict["messages"]:
                msg_dict["timestamp"] = datetime.fromisoformat(msg_dict["timestamp"])
                messages.append(ChatMessage(**msg_dict))
            
            conv_dict["messages"] = messages
            
            return Conversation(**conv_dict)
        except Exception as e:
            logger.error(f"Error loading conversation: {e}")
            return None
    
    async def _encrypt_sensitive_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive fields using advanced security"""
        encrypted_data = data.copy()
        
        # Encrypt fields based on classification
        for field, security_level in self.data_classification.items():
            if field in encrypted_data and encrypted_data[field] is not None:
                try:
                    # Convert to string for encryption if needed
                    field_data = encrypted_data[field]
                    if not isinstance(field_data, (str, bytes)):
                        field_data = json.dumps(field_data)
                    
                    # Encrypt with appropriate security level
                    encrypted_package = await encrypt_sensitive_data(
                        field_data, 
                        security_level
                    )
                    
                    # Store encrypted package
                    encrypted_data[f"{field}_encrypted"] = encrypted_package
                    # Remove original field
                    del encrypted_data[field]
                    
                except Exception as e:
                    logger.error(f"Failed to encrypt field {field}: {e}")
                    # Keep original data if encryption fails
                    pass
        
        return encrypted_data
    
    async def _decrypt_sensitive_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive fields using advanced security"""
        decrypted_data = data.copy()
        
        # Find and decrypt encrypted fields
        encrypted_fields = [key for key in data.keys() if key.endswith('_encrypted')]
        
        for encrypted_field in encrypted_fields:
            try:
                original_field = encrypted_field.replace('_encrypted', '')
                encrypted_package = data[encrypted_field]
                
                # Decrypt the data
                decrypted_bytes = await decrypt_sensitive_data(encrypted_package)
                
                # Try to parse as JSON first, fallback to string
                try:
                    decrypted_value = json.loads(decrypted_bytes.decode())
                except:
                    decrypted_value = decrypted_bytes.decode()
                
                # Restore original field
                decrypted_data[original_field] = decrypted_value
                # Remove encrypted field
                del decrypted_data[encrypted_field]
                
            except Exception as e:
                logger.error(f"Failed to decrypt field {encrypted_field}: {e}")
                # Keep encrypted data if decryption fails
                pass
        
        return decrypted_data
    
    async def _delete_user_file(self, user_id: str):
        """Delete user data file"""
        user_file = self.storage_path / f"user_{user_id}.json"
        if user_file.exists():
            user_file.unlink()
    
    async def _delete_conversation_file(self, conversation_id: str):
        """Delete conversation file"""
        conv_file = self.storage_path / f"conv_{conversation_id}.json"
        if conv_file.exists():
            conv_file.unlink()


# Global data manager instance
data_manager = DataManager()


# Convenience functions for integration
async def create_user_data(user_data: Dict[str, Any]) -> UserData:
    """Create user with comprehensive data structure"""
    return await data_manager.create_user(user_data)


async def store_chat_message(user_id: str, message: str, response: str) -> ChatMessage:
    """Store chat message in user's conversation"""
    # Get or create active conversation
    conversations = await data_manager.get_user_conversations(user_id, limit=1)
    
    if conversations and not conversations[0].is_archived:
        conversation = conversations[0]
    else:
        conversation = await data_manager.create_conversation(user_id)
    
    return await data_manager.add_message_to_conversation(
        conversation.conversation_id, message, response
    )


async def get_user_chat_history(user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get user's recent chat history"""
    conversations = await data_manager.get_user_conversations(user_id, limit=limit)
    
    chat_history = []
    for conv in conversations:
        for msg in conv.messages[-5:]:  # Last 5 messages per conversation
            chat_history.append({
                "message": msg.message,
                "response": msg.response,
                "timestamp": msg.timestamp.isoformat(),
                "conversation_id": conv.conversation_id
            })
    
    return sorted(chat_history, key=lambda x: x["timestamp"], reverse=True)


if __name__ == "__main__":
    # Demo the data manager
    async def demo():
        print("🗄️ DharmaMind Data Manager Demo")
        print("=" * 50)
        
        # Create test user
        user_data = {
            "email": "demo@dharmamind.ai",
            "first_name": "Demo",
            "last_name": "User",
            "phone": "+1-555-0123"
        }
        
        user = await data_manager.create_user(user_data)
        print(f"Created user: {user.user_id}")
        
        # Store some chat messages
        msg1 = await store_chat_message(
            user.user_id,
            "What is the meaning of life?",
            "The meaning of life is to find your purpose and live it fully."
        )
        print(f"Stored message: {msg1.message_id}")
        
        # Get analytics
        analytics = await data_manager.get_user_analytics(user.user_id)
        print(f"User analytics: {analytics['conversation_stats']}")
        
        # Export data
        export_data = await data_manager.export_user_data(user.user_id)
        print(f"Exported {len(export_data['conversations'])} conversations")
    
    asyncio.run(demo())
