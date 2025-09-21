"""
User preferences model for DharmaMind platform

Defines user preferences for chat behavior, notifications, and interface customization.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class ThemePreference(str, Enum):
    """UI theme preferences"""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"
    SPIRITUAL = "spiritual"

class LanguagePreference(str, Enum):
    """Language preferences"""
    ENGLISH = "en"
    HINDI = "hi"
    SANSKRIT = "sa"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    CHINESE = "zh"
    JAPANESE = "ja"

class NotificationLevel(str, Enum):
    """Notification level preferences"""
    ALL = "all"
    IMPORTANT = "important"
    MINIMAL = "minimal"
    NONE = "none"

class UserPreferences(BaseModel):
    """User preferences for personalized experience"""
    user_id: str = Field(..., description="Associated user ID")
    
    # Interface preferences
    theme: ThemePreference = Field(default=ThemePreference.AUTO, description="UI theme preference")
    language: LanguagePreference = Field(default=LanguagePreference.ENGLISH, description="Interface language")
    font_size: str = Field(default="medium", description="Font size preference")
    animation_enabled: bool = Field(default=True, description="Enable UI animations")
    
    # Chat preferences
    response_style: str = Field(default="balanced", description="AI response style preference")
    spiritual_depth: str = Field(default="medium", description="Preferred spiritual depth level")
    wisdom_tradition_focus: Optional[str] = Field(default=None, description="Focus on specific tradition")
    include_sanskrit_terms: bool = Field(default=True, description="Include Sanskrit terminology")
    include_citations: bool = Field(default=True, description="Include source citations")
    
    # Notification preferences
    notification_level: NotificationLevel = Field(default=NotificationLevel.IMPORTANT, description="Notification level")
    email_notifications: bool = Field(default=True, description="Enable email notifications")
    push_notifications: bool = Field(default=True, description="Enable push notifications")
    daily_wisdom: bool = Field(default=True, description="Receive daily wisdom messages")
    session_reminders: bool = Field(default=True, description="Session practice reminders")
    
    # Privacy preferences
    data_sharing: bool = Field(default=False, description="Allow anonymized data sharing")
    analytics_tracking: bool = Field(default=True, description="Allow analytics tracking")
    personalization: bool = Field(default=True, description="Enable personalized experience")
    
    # Content preferences
    content_filters: List[str] = Field(default_factory=list, description="Content filtering preferences")
    blocked_topics: List[str] = Field(default_factory=list, description="Topics to avoid")
    preferred_teachers: List[str] = Field(default_factory=list, description="Preferred spiritual teachers")
    
    # Audio/Visual preferences
    voice_enabled: bool = Field(default=False, description="Enable voice responses")
    preferred_voice: Optional[str] = Field(default=None, description="Preferred voice for audio")
    background_sounds: bool = Field(default=False, description="Enable background sounds")
    meditation_timer: bool = Field(default=True, description="Enable meditation timer")
    
    # Advanced preferences
    experimental_features: bool = Field(default=False, description="Enable experimental features")
    advanced_mode: bool = Field(default=False, description="Enable advanced user interface")
    developer_mode: bool = Field(default=False, description="Enable developer features")
    
    # Custom settings
    custom_settings: Dict[str, Any] = Field(default_factory=dict, description="Custom user settings")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Preferences creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }

class UserPreferencesUpdate(BaseModel):
    """Model for updating user preferences"""
    theme: Optional[ThemePreference] = Field(default=None, description="UI theme preference")
    language: Optional[LanguagePreference] = Field(default=None, description="Interface language")
    font_size: Optional[str] = Field(default=None, description="Font size preference")
    response_style: Optional[str] = Field(default=None, description="AI response style")
    spiritual_depth: Optional[str] = Field(default=None, description="Spiritual depth preference")
    notification_level: Optional[NotificationLevel] = Field(default=None, description="Notification level")
    email_notifications: Optional[bool] = Field(default=None, description="Email notifications")
    push_notifications: Optional[bool] = Field(default=None, description="Push notifications")
    data_sharing: Optional[bool] = Field(default=None, description="Data sharing preference")
    voice_enabled: Optional[bool] = Field(default=None, description="Voice responses")
    experimental_features: Optional[bool] = Field(default=None, description="Experimental features")
    custom_settings: Optional[Dict[str, Any]] = Field(default=None, description="Custom settings")