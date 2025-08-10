"""
Module Selector Service

Selects the best-fit Dharma modules for a given query.
Analyzes user input and conversation context to determine which
spiritual/philosophical modules should be engaged.

Modules are defined in YAML files with expertise areas and capabilities.
"""

import asyncio
import logging
import os
import yaml
from typing import List, Optional, Dict, Any
from pathlib import Path

from ..models.chat import ChatMessage, ModuleInfo
from ..config import settings

logger = logging.getLogger(__name__)

class ModuleSelector:
    """Service for selecting appropriate Dharma modules"""
    
    def __init__(self):
        self.modules: Dict[str, ModuleInfo] = {}
        self.module_embeddings: Dict[str, Any] = {}
        self.expertise_map: Dict[str, List[str]] = {}
        
    async def initialize(self):
        """Initialize module selector with available modules"""
        logger.info("Initializing Module Selector...")
        
        try:
            await self._load_modules()
            await self._build_expertise_map()
            logger.info(f"Module Selector initialized with {len(self.modules)} modules")
            
        except Exception as e:
            logger.error(f"Failed to initialize Module Selector: {e}")
            raise
    
    async def _load_modules(self):
        """Load module definitions from YAML files"""
        module_path = Path(settings.MODULE_CONFIG_PATH)
        
        if not module_path.exists():
            logger.warning(f"Module path does not exist: {module_path}")
            await self._create_default_modules()
            return
            
        # Load YAML module definitions
        for yaml_file in module_path.glob("*.yaml"):
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    module_data = yaml.safe_load(f)
                    
                module_info = ModuleInfo(
                    name=module_data['name'],
                    description=module_data['description'],
                    category=module_data.get('category', 'general'),
                    expertise_areas=module_data.get('expertise_areas', []),
                    confidence=1.0,
                    yaml_path=str(yaml_file)
                )
                
                self.modules[module_info.name] = module_info
                logger.debug(f"Loaded module: {module_info.name}")
                
            except Exception as e:
                logger.error(f"Error loading module from {yaml_file}: {e}")
    
    async def _create_default_modules(self):
        """Create default Dharma modules if none exist"""
        logger.info("Creating default Dharma modules...")
        
        default_modules = [
            {
                "name": "karma",
                "description": "Understanding of karma, action, and consequence",
                "category": "dharmic_principles",
                "expertise_areas": ["action", "consequence", "ethics", "moral_responsibility", "past_actions", "future_outcomes"]
            },
            {
                "name": "dharma",
                "description": "Righteous living and ethical conduct",
                "category": "dharmic_principles", 
                "expertise_areas": ["righteousness", "duty", "ethics", "moral_conduct", "life_purpose", "social_responsibility"]
            },
            {
                "name": "meditation",
                "description": "Mindfulness, meditation, and inner peace practices",
                "category": "spiritual_practice",
                "expertise_areas": ["mindfulness", "meditation", "inner_peace", "concentration", "awareness", "mental_training"]
            },
            {
                "name": "yoga",
                "description": "Unity of mind, body, and spirit through yogic practices",
                "category": "spiritual_practice",
                "expertise_areas": ["unity", "physical_practice", "spiritual_discipline", "mind_body_connection", "pranayama", "asanas"]
            },
            {
                "name": "wisdom",
                "description": "Ancient wisdom and timeless spiritual insights",
                "category": "knowledge",
                "expertise_areas": ["wisdom", "spiritual_insight", "ancient_knowledge", "life_guidance", "understanding", "enlightenment"]
            },
            {
                "name": "compassion",
                "description": "Loving-kindness and compassion for all beings",
                "category": "virtues",
                "expertise_areas": ["compassion", "loving_kindness", "empathy", "care", "understanding", "universal_love"]
            },
            {
                "name": "peace", 
                "description": "Inner peace and harmony in daily life",
                "category": "virtues",
                "expertise_areas": ["peace", "harmony", "tranquility", "calm", "serenity", "balance"]
            },
            {
                "name": "scripture",
                "description": "Insights from sacred texts and scriptures",
                "category": "knowledge",
                "expertise_areas": ["sacred_texts", "scriptures", "vedas", "upanishads", "bhagavad_gita", "ancient_wisdom"]
            }
        ]
        
        for module_data in default_modules:
            module_info = ModuleInfo(
                name=module_data['name'],
                description=module_data['description'],
                category=module_data['category'],
                expertise_areas=module_data['expertise_areas'],
                confidence=1.0
            )
            self.modules[module_info.name] = module_info
    
    async def _build_expertise_map(self):
        """Build reverse map from expertise areas to modules"""
        self.expertise_map = {}
        
        for module_name, module_info in self.modules.items():
            for expertise in module_info.expertise_areas:
                if expertise not in self.expertise_map:
                    self.expertise_map[expertise] = []
                self.expertise_map[expertise].append(module_name)
    
    async def select_modules(
        self,
        message: str,
        context: Optional[str] = None,
        history: Optional[List[ChatMessage]] = None,
        max_modules: int = 3
    ) -> List[ModuleInfo]:
        """Select best-fit modules for a message"""
        
        try:
            # Analyze message content
            message_analysis = await self._analyze_message(message, context, history)
            
            # Score modules based on relevance
            module_scores = await self._score_modules(message_analysis)
            
            # Select top modules
            selected = self._select_top_modules(module_scores, max_modules)
            
            logger.info(f"Selected modules for message: {[m.name for m in selected]}")
            return selected
            
        except Exception as e:
            logger.error(f"Error selecting modules: {e}")
            # Return default modules if selection fails
            return await self._get_default_modules()
    
    async def _analyze_message(
        self,
        message: str,
        context: Optional[str] = None,
        history: Optional[List[ChatMessage]] = None
    ) -> Dict[str, Any]:
        """Analyze message to understand intent and topics"""
        
        # Convert to lowercase for analysis
        text = message.lower()
        if context:
            text += " " + context.lower()
        
        # Add recent history context
        if history:
            recent_messages = " ".join([msg.content.lower() for msg in history[-3:]])
            text += " " + recent_messages
        
        # Extract keywords and themes
        keywords = self._extract_keywords(text)
        themes = self._identify_themes(keywords)
        intent = self._classify_intent(text)
        emotional_tone = self._analyze_emotional_tone(text)
        
        return {
            "keywords": keywords,
            "themes": themes,
            "intent": intent,
            "emotional_tone": emotional_tone,
            "message_length": len(message),
            "has_question": "?" in message
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        # Simple keyword extraction (could be enhanced with NLP)
        dharmic_keywords = [
            "dharma", "karma", "yoga", "meditation", "peace", "wisdom", "compassion",
            "suffering", "happiness", "enlightenment", "consciousness", "mindfulness",
            "balance", "harmony", "truth", "love", "kindness", "anger", "fear",
            "worry", "stress", "anxiety", "depression", "joy", "gratitude",
            "purpose", "meaning", "life", "death", "rebirth", "moksha", "nirvana",
            "scripture", "vedas", "upanishads", "gita", "spiritual", "divine",
            "god", "universe", "soul", "self", "ego", "attachment", "detachment"
        ]
        
        found_keywords = []
        for keyword in dharmic_keywords:
            if keyword in text:
                found_keywords.append(keyword)
                
        return found_keywords
    
    def _identify_themes(self, keywords: List[str]) -> List[str]:
        """Identify themes from keywords"""
        theme_mapping = {
            "spiritual_practice": ["meditation", "yoga", "mindfulness", "spiritual"],
            "life_challenges": ["suffering", "anxiety", "stress", "worry", "fear", "depression"],
            "virtues": ["compassion", "love", "kindness", "peace", "harmony"],
            "philosophy": ["dharma", "karma", "truth", "wisdom", "consciousness"],
            "emotions": ["anger", "joy", "happiness", "gratitude"],
            "life_purpose": ["purpose", "meaning", "enlightenment", "moksha"],
            "sacred_knowledge": ["scripture", "vedas", "upanishads", "gita"]
        }
        
        themes = []
        for theme, theme_keywords in theme_mapping.items():
            if any(keyword in keywords for keyword in theme_keywords):
                themes.append(theme)
                
        return themes
    
    def _classify_intent(self, text: str) -> str:
        """Classify the intent of the message"""
        if any(word in text for word in ["how", "what", "why", "when", "where"]):
            return "question"
        elif any(word in text for word in ["help", "guidance", "advice", "suggest"]):
            return "seeking_guidance"
        elif any(word in text for word in ["sad", "depressed", "anxious", "worried", "suffering"]):
            return "emotional_support"
        elif any(word in text for word in ["learn", "understand", "know", "explain"]):
            return "learning"
        elif any(word in text for word in ["meditate", "practice", "spiritual", "peace"]):
            return "spiritual_practice"
        else:
            return "general_conversation"
    
    def _analyze_emotional_tone(self, text: str) -> str:
        """Analyze emotional tone of the message"""
        positive_words = ["happy", "joy", "grateful", "peace", "love", "wonderful", "great", "good"]
        negative_words = ["sad", "angry", "frustrated", "worried", "anxious", "depressed", "suffering"]
        neutral_words = ["question", "wonder", "think", "consider", "understand"]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if negative_count > positive_count:
            return "negative"
        elif positive_count > negative_count:
            return "positive"
        else:
            return "neutral"
    
    async def _score_modules(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Score modules based on message analysis"""
        scores = {}
        
        for module_name, module_info in self.modules.items():
            score = 0.0
            
            # Score based on keyword matches
            keyword_matches = 0
            for keyword in analysis["keywords"]:
                if keyword in module_info.expertise_areas or keyword in module_info.name.lower():
                    keyword_matches += 1
            
            score += keyword_matches * 0.3
            
            # Score based on theme matches
            theme_matches = 0
            for theme in analysis["themes"]:
                if theme in module_info.category or any(theme in area for area in module_info.expertise_areas):
                    theme_matches += 1
            
            score += theme_matches * 0.4
            
            # Intent-based scoring
            intent_score = self._score_by_intent(analysis["intent"], module_info)
            score += intent_score * 0.2
            
            # Emotional tone adjustment
            tone_adjustment = self._adjust_for_emotional_tone(analysis["emotional_tone"], module_info)
            score += tone_adjustment * 0.1
            
            scores[module_name] = score
            
        return scores
    
    def _score_by_intent(self, intent: str, module_info: ModuleInfo) -> float:
        """Score module based on message intent"""
        intent_module_mapping = {
            "emotional_support": ["compassion", "peace", "wisdom"],
            "spiritual_practice": ["meditation", "yoga", "peace"],
            "learning": ["wisdom", "scripture", "dharma"],
            "seeking_guidance": ["dharma", "karma", "wisdom"],
            "question": ["wisdom", "scripture"]
        }
        
        if intent in intent_module_mapping:
            if module_info.name in intent_module_mapping[intent]:
                return 1.0
        
        return 0.0
    
    def _adjust_for_emotional_tone(self, tone: str, module_info: ModuleInfo) -> float:
        """Adjust score based on emotional tone"""
        tone_adjustments = {
            "negative": {
                "compassion": 0.3,
                "peace": 0.3,
                "meditation": 0.2
            },
            "positive": {
                "wisdom": 0.1,
                "dharma": 0.1
            }
        }
        
        if tone in tone_adjustments and module_info.name in tone_adjustments[tone]:
            return tone_adjustments[tone][module_info.name]
        
        return 0.0
    
    def _select_top_modules(self, scores: Dict[str, float], max_modules: int) -> List[ModuleInfo]:
        """Select top scoring modules"""
        # Sort by score descending
        sorted_modules = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top modules with non-zero scores
        selected = []
        for module_name, score in sorted_modules[:max_modules]:
            if score > 0:
                module_info = self.modules[module_name]
                # Update confidence based on score
                module_info.confidence = min(score, 1.0)
                selected.append(module_info)
        
        # Ensure at least one module is selected
        if not selected and self.modules:
            default_module = list(self.modules.values())[0]
            default_module.confidence = 0.5
            selected.append(default_module)
            
        return selected
    
    async def select_wisdom_modules(
        self,
        question: str,
        category: Optional[str] = None,
        urgency: str = "normal"
    ) -> List[ModuleInfo]:
        """Select modules specifically for wisdom requests"""
        
        # Prioritize wisdom-related modules
        wisdom_modules = ["wisdom", "scripture", "dharma", "meditation"]
        
        if category:
            # Add category-specific modules
            category_modules = {
                "dharma": ["dharma", "karma", "scripture"],
                "meditation": ["meditation", "peace", "yoga"],
                "life": ["wisdom", "dharma", "compassion"],
                "relationships": ["compassion", "love", "dharma"],
                "suffering": ["compassion", "peace", "wisdom"]
            }
            
            if category in category_modules:
                wisdom_modules.extend(category_modules[category])
        
        # Remove duplicates while preserving order
        wisdom_modules = list(dict.fromkeys(wisdom_modules))
        
        # Return available modules
        selected = []
        for module_name in wisdom_modules[:3]:  # Max 3 modules
            if module_name in self.modules:
                module = self.modules[module_name]
                module.confidence = 0.9  # High confidence for wisdom requests
                selected.append(module)
        
        return selected
    
    async def get_available_modules(self) -> List[ModuleInfo]:
        """Get all available modules"""
        return list(self.modules.values())
    
    async def _get_default_modules(self) -> List[ModuleInfo]:
        """Get default modules when selection fails"""
        default_names = ["wisdom", "dharma", "compassion"]
        defaults = []
        
        for name in default_names:
            if name in self.modules:
                module = self.modules[name]
                module.confidence = 0.5
                defaults.append(module)
        
        return defaults if defaults else list(self.modules.values())[:1]
    
    async def health_check(self) -> bool:
        """Check if module selector is healthy"""
        return len(self.modules) > 0


# Dependency injection function for FastAPI
_module_selector_instance = None


def get_module_selector() -> ModuleSelector:
    """Get the module selector instance (singleton pattern)"""
    global _module_selector_instance
    if _module_selector_instance is None:
        _module_selector_instance = ModuleSelector()
    return _module_selector_instance
