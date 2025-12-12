"""
Memory Manager Service
=====================

Manages conversation memory and context for the DharmaMind system.
Temporary implementation for backward compatibility.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages conversation memory and context"""
    
    def __init__(self):
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.user_contexts: Dict[str, Dict[str, Any]] = {}
        logger.info("Memory Manager initialized")
    
    def add_message(self, conversation_id: str, message: Dict[str, Any], user_id: Optional[str] = None):
        """Add a message to conversation memory"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        message_with_timestamp = {
            **message,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id
        }
        
        self.conversations[conversation_id].append(message_with_timestamp)
        logger.debug(f"Added message to conversation {conversation_id}")
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversations.get(conversation_id, [])
    
    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get user context"""
        return self.user_contexts.get(user_id, {})
    
    def update_user_context(self, user_id: str, context: Dict[str, Any]):
        """Update user context"""
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = {}
        
        self.user_contexts[user_id].update(context)
        logger.debug(f"Updated context for user {user_id}")
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation history"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Cleared conversation {conversation_id}")

# Global memory manager instance
_memory_manager = None

def get_memory_manager() -> MemoryManager:
    """Get or create memory manager instance"""
    global _memory_manager
    
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    
    return _memory_manager
