"""
LLM Router Service

Routes requests to appropriate LLMs (GPT-4, DharmaLLM, Claude, etc.)
based on content type, complexity, and availability.

Handles:
- Model selection and routing
- Response generation coordination
- Fallback management
- Performance monitoring
"""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from ..models.chat import ChatMessage, ModuleInfo
from ..config import settings

logger = logging.getLogger(__name__)

class LLMProvider(str, Enum):
    """Available LLM providers"""
    DHARMALLM = "dharmallm"
    OPENAI_GPT4 = "openai-gpt4"
    OPENAI_GPT35 = "openai-gpt35" 
    ANTHROPIC_CLAUDE = "anthropic-claude"
    HUGGINGFACE = "huggingface"

@dataclass
class LLMResponse:
    """LLM response data"""
    content: str
    model_name: str
    provider: LLMProvider
    processing_time: float
    tokens_used: int
    confidence: float
    metadata: Dict[str, Any]

class LLMRouter:
    """Main LLM routing and coordination service"""
    
    def __init__(self):
        self.providers = {}
        self.fallback_chain = [
            LLMProvider.DHARMALLM,
            LLMProvider.OPENAI_GPT4,
            LLMProvider.ANTHROPIC_CLAUDE,
            LLMProvider.OPENAI_GPT35
        ]
        self.performance_metrics = {}
        
    async def initialize(self):
        """Initialize LLM providers"""
        logger.info("Initializing LLM Router...")
        
        try:
            # Initialize DharmaLLM (local model)
            if settings.DHARMALLM_MODEL_PATH:
                await self._init_dharmallm()
                
            # Initialize OpenAI
            if settings.OPENAI_API_KEY:
                await self._init_openai()
                
            # Initialize Anthropic
            if settings.ANTHROPIC_API_KEY:
                await self._init_anthropic()
                
            # Initialize HuggingFace
            if settings.HUGGINGFACE_API_KEY:
                await self._init_huggingface()
                
            logger.info(f"LLM Router initialized with {len(self.providers)} providers")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM Router: {e}")
            raise
    
    async def _init_dharmallm(self):
        """Initialize DharmaLLM local model"""
        try:
            # This would typically load a local transformer model
            # For now, we'll create a placeholder
            self.providers[LLMProvider.DHARMALLM] = {
                "status": "ready",
                "model_path": settings.DHARMALLM_MODEL_PATH,
                "max_length": settings.DHARMALLM_MAX_LENGTH,
                "temperature": settings.DHARMALLM_TEMPERATURE
            }
            logger.info("DharmaLLM provider initialized")
        except Exception as e:
            logger.error(f"Failed to initialize DharmaLLM: {e}")
    
    async def _init_openai(self):
        """Initialize OpenAI provider"""
        try:
            # Initialize OpenAI client
            self.providers[LLMProvider.OPENAI_GPT4] = {
                "status": "ready",
                "api_key": settings.OPENAI_API_KEY,
                "model": "gpt-4"
            }
            self.providers[LLMProvider.OPENAI_GPT35] = {
                "status": "ready", 
                "api_key": settings.OPENAI_API_KEY,
                "model": "gpt-3.5-turbo"
            }
            logger.info("OpenAI providers initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
    
    async def _init_anthropic(self):
        """Initialize Anthropic Claude"""
        try:
            self.providers[LLMProvider.ANTHROPIC_CLAUDE] = {
                "status": "ready",
                "api_key": settings.ANTHROPIC_API_KEY,
                "model": "claude-3-opus-20240229"
            }
            logger.info("Anthropic provider initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic: {e}")
    
    async def _init_huggingface(self):
        """Initialize HuggingFace provider"""
        try:
            self.providers[LLMProvider.HUGGINGFACE] = {
                "status": "ready",
                "api_key": settings.HUGGINGFACE_API_KEY
            }
            logger.info("HuggingFace provider initialized")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace: {e}")
    
    async def generate_response(
        self,
        message: str,
        context: Optional[str] = None,
        modules: Optional[List[ModuleInfo]] = None,
        language: str = "en",
        history: Optional[List[ChatMessage]] = None
    ) -> LLMResponse:
        """Generate response using best available LLM"""
        
        start_time = time.time()
        
        try:
            # Select best LLM for this request
            selected_provider = await self._select_provider(message, modules, context)
            
            logger.info(f"Routing request to {selected_provider}")
            
            # Generate response
            response = await self._generate_with_provider(
                provider=selected_provider,
                message=message,
                context=context,
                modules=modules,
                language=language,
                history=history
            )
            
            # Update performance metrics
            processing_time = time.time() - start_time
            await self._update_metrics(selected_provider, processing_time, True)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Try fallback providers
            return await self._generate_with_fallback(message, context, modules, language, history)
    
    async def _select_provider(
        self,
        message: str,
        modules: Optional[List[ModuleInfo]] = None,
        context: Optional[str] = None
    ) -> LLMProvider:
        """Select best LLM provider for request"""
        
        # Check if this is dharmic content (prefer DharmaLLM)
        if self._is_dharmic_content(message, modules):
            if LLMProvider.DHARMALLM in self.providers:
                return LLMProvider.DHARMALLM
        
        # For complex reasoning, prefer GPT-4
        if self._is_complex_reasoning(message):
            if LLMProvider.OPENAI_GPT4 in self.providers:
                return LLMProvider.OPENAI_GPT4
                
        # For general chat, use most available/fastest
        return await self._get_fastest_available_provider()
    
    def _is_dharmic_content(self, message: str, modules: Optional[List[ModuleInfo]] = None) -> bool:
        """Check if content is dharmic/spiritual"""
        dharmic_keywords = [
            "dharma", "karma", "yoga", "meditation", "spiritual", "wisdom",
            "consciousness", "enlightenment", "scripture", "vedic", "hindu",
            "buddhist", "sanskrit", "mantra", "chakra", "moksha"
        ]
        
        message_lower = message.lower()
        keyword_matches = sum(1 for keyword in dharmic_keywords if keyword in message_lower)
        
        # Also check if dharmic modules are being used
        dharmic_modules = modules and any("dharma" in m.name.lower() for m in modules)
        
        return keyword_matches >= 2 or bool(dharmic_modules)
    
    def _is_complex_reasoning(self, message: str) -> bool:
        """Check if message requires complex reasoning"""
        complex_indicators = [
            "analyze", "compare", "explain why", "what if", "how does",
            "relationship between", "cause and effect", "pros and cons",
            "evaluate", "assess", "determine", "reasoning", "logic"
        ]
        
        message_lower = message.lower()
        return any(indicator in message_lower for indicator in complex_indicators)
    
    async def _get_fastest_available_provider(self) -> LLMProvider:
        """Get the fastest available provider based on metrics"""
        available_providers = [
            provider for provider in self.fallback_chain 
            if provider in self.providers
        ]
        
        if not available_providers:
            raise Exception("No LLM providers available")
            
        # Return first available (could be enhanced with actual performance metrics)
        return available_providers[0]
    
    async def _generate_with_provider(
        self,
        provider: LLMProvider,
        message: str,
        context: Optional[str] = None,
        modules: List[ModuleInfo] = None,
        language: str = "en",
        history: List[ChatMessage] = None
    ) -> LLMResponse:
        """Generate response with specific provider"""
        
        # Build prompt with context and modules
        full_prompt = await self._build_prompt(message, context, modules, language, history)
        
        if provider == LLMProvider.DHARMALLM:
            return await self._generate_dharmallm(full_prompt)
        elif provider == LLMProvider.OPENAI_GPT4:
            return await self._generate_openai(full_prompt, "gpt-4")
        elif provider == LLMProvider.OPENAI_GPT35:
            return await self._generate_openai(full_prompt, "gpt-3.5-turbo")
        elif provider == LLMProvider.ANTHROPIC_CLAUDE:
            return await self._generate_anthropic(full_prompt)
        else:
            raise Exception(f"Unsupported provider: {provider}")
    
    async def _build_prompt(
        self,
        message: str,
        context: Optional[str] = None,
        modules: List[ModuleInfo] = None,
        language: str = "en",
        history: List[ChatMessage] = None
    ) -> str:
        """Build complete prompt with context"""
        
        prompt_parts = []
        
        # System context
        prompt_parts.append(
            "You are DharmaMind, a wise AI assistant that provides guidance "
            "based on ancient dharmic wisdom and modern understanding. "
            "Your responses should be compassionate, practical, and respectful of all traditions."
        )
        
        # Add module context
        if modules:
            module_context = "Drawing wisdom from these areas: " + ", ".join(m.name for m in modules)
            prompt_parts.append(module_context)
        
        # Add conversation history
        if history:
            history_context = "Previous conversation:\n"
            for msg in history[-5:]:  # Last 5 messages
                history_context += f"{msg.role.value}: {msg.content}\n"
            prompt_parts.append(history_context)
        
        # Add current context
        if context:
            prompt_parts.append(f"Context: {context}")
        
        # Add language preference
        if language != "en":
            prompt_parts.append(f"Please respond in {language}")
        
        # Add user message
        prompt_parts.append(f"User: {message}")
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    async def _generate_dharmallm(self, prompt: str) -> LLMResponse:
        """Generate response using DharmaLLM"""
        start_time = time.time()
        
        # Placeholder for actual DharmaLLM implementation
        # This would typically use transformers library
        response_content = (
            "🕉️ This is a response from DharmaLLM, drawing upon ancient wisdom "
            "and modern understanding to provide guidance. "
            "[Actual implementation would use local transformer model]"
        )
        
        processing_time = time.time() - start_time
        
        return LLMResponse(
            content=response_content,
            model_name="DharmaLLM-7B",
            provider=LLMProvider.DHARMALLM,
            processing_time=processing_time,
            tokens_used=len(response_content.split()),
            confidence=0.85,
            metadata={"source": "local", "temperature": settings.DHARMALLM_TEMPERATURE}
        )
    
    async def _generate_openai(self, prompt: str, model: str) -> LLMResponse:
        """Generate response using OpenAI"""
        start_time = time.time()
        
        # Placeholder for actual OpenAI implementation
        response_content = (
            f"This is a response from {model}. "
            "[Actual implementation would use OpenAI API]"
        )
        
        processing_time = time.time() - start_time
        
        return LLMResponse(
            content=response_content,
            model_name=model,
            provider=LLMProvider.OPENAI_GPT4 if "gpt-4" in model else LLMProvider.OPENAI_GPT35,
            processing_time=processing_time,
            tokens_used=len(response_content.split()),
            confidence=0.90,
            metadata={"source": "openai", "model": model}
        )
    
    async def _generate_anthropic(self, prompt: str) -> LLMResponse:
        """Generate response using Anthropic Claude"""
        start_time = time.time()
        
        # Placeholder for actual Anthropic implementation
        response_content = (
            "This is a response from Claude. "
            "[Actual implementation would use Anthropic API]"
        )
        
        processing_time = time.time() - start_time
        
        return LLMResponse(
            content=response_content,
            model_name="claude-3-opus",
            provider=LLMProvider.ANTHROPIC_CLAUDE,
            processing_time=processing_time,
            tokens_used=len(response_content.split()),
            confidence=0.88,
            metadata={"source": "anthropic"}
        )
    
    async def _generate_with_fallback(
        self,
        message: str,
        context: Optional[str] = None,
        modules: List[ModuleInfo] = None,
        language: str = "en",
        history: List[ChatMessage] = None
    ) -> LLMResponse:
        """Try fallback providers if primary fails"""
        
        for provider in self.fallback_chain:
            if provider in self.providers:
                try:
                    logger.info(f"Trying fallback provider: {provider}")
                    return await self._generate_with_provider(
                        provider, message, context, modules, language, history
                    )
                except Exception as e:
                    logger.warning(f"Fallback provider {provider} failed: {e}")
                    continue
        
        raise Exception("All LLM providers failed")
    
    async def _update_metrics(self, provider: LLMProvider, processing_time: float, success: bool):
        """Update performance metrics"""
        if provider not in self.performance_metrics:
            self.performance_metrics[provider] = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_time": 0.0,
                "average_time": 0.0
            }
        
        metrics = self.performance_metrics[provider]
        metrics["total_requests"] += 1
        
        if success:
            metrics["successful_requests"] += 1
            metrics["total_time"] += processing_time
            metrics["average_time"] = metrics["total_time"] / metrics["successful_requests"]
    
    async def generate_wisdom_response(
        self,
        question: str,
        modules: List[ModuleInfo],
        category: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Generate specialized wisdom response enhanced with spiritual knowledge base"""
        
        # Get relevant wisdom from knowledge base
        relevant_wisdom = await self._get_contextual_wisdom(question, category, user_context)
        
        # Build wisdom-focused prompt with knowledge enhancement
        context_parts = []
        
        if category:
            context_parts.append(f"Wisdom category: {category}")
        
        if user_context:
            context_parts.append(f"User context: {user_context}")
        
        # Add relevant wisdom from knowledge base
        if relevant_wisdom:
            wisdom_context = "Relevant spiritual wisdom:\n"
            for i, wisdom in enumerate(relevant_wisdom[:3], 1):  # Limit to top 3
                wisdom_context += f"{i}. {wisdom['title']} ({wisdom['tradition']}): {wisdom['text'][:200]}...\n"
            context_parts.append(wisdom_context)
        
        context = "\n".join(context_parts) if context_parts else None
        
        # Enhanced prompt instruction
        enhanced_question = f"""
        {question}
        
        Instructions: Provide a compassionate, wise response drawing from the profound wisdom of 
        Sanatan Dharma (the eternal way) - the world's oldest continuous spiritual tradition. 
        Present these ancient insights as universal human wisdom applicable to modern life.
        
        If relevant wisdom is provided above, integrate those insights naturally while explaining:
        - The universal principle behind the teaching
        - Practical application in daily life  
        - How it addresses the core human concern
        - Sanskrit terms with clear meanings when helpful
        
        Focus on empowerment, practical transformation, and timeless wisdom that serves all humanity.
        Present knowledge scientifically and respectfully, not as religious doctrine.
        """
        
        response = await self.generate_response(
            message=enhanced_question,
            context=context,
            modules=modules
        )
        
        # Enhance response metadata with wisdom sources
        if relevant_wisdom and hasattr(response, 'metadata'):
            response.metadata['wisdom_sources'] = [
                {
                    'title': w['title'],
                    'tradition': w['tradition'],
                    'relevance_score': w.get('relevance_score', 0.5)
                }
                for w in relevant_wisdom[:3]
            ]
        
        return response
    
    async def _get_contextual_wisdom(
        self, 
        question: str, 
        category: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get contextual wisdom from the spiritual knowledge base"""
        try:
            # Import here to avoid circular imports
            from ..chakra_modules.knowledge_base import get_knowledge_base
            
            kb = get_knowledge_base()
            if not kb.is_initialized:
                await kb.initialize()
            
            # Determine search strategy based on context
            wisdom_results = []
            
            # Check if this is an emotional query
            emotion_keywords = ["angry", "sad", "fear", "anxious", "worry", "depressed", "lonely", "confused"]
            detected_emotion = None
            for emotion in emotion_keywords:
                if emotion in question.lower():
                    detected_emotion = emotion
                    break
            
            if detected_emotion:
                wisdom_results = await kb.get_wisdom_for_emotion(detected_emotion)
            
            # Check if this is asking for practices
            elif any(word in question.lower() for word in ["practice", "meditation", "how to", "exercise"]):
                practice_focus = category or "general spiritual practice"
                wisdom_results = await kb.get_practice_recommendations(practice_focus)
            
            # Check if this is a life situation
            elif any(word in question.lower() for word in ["relationship", "work", "career", "family", "decision"]):
                context_dict = user_context or {}
                wisdom_results = await kb.get_guidance_for_situation(question, context_dict)
            
            # General wisdom search
            else:
                search_category = None
                if category and category.lower() in ["meditation", "compassion", "wisdom", "practice", "ethics"]:
                    search_category = category.lower()
                
                wisdom_results = await kb.search_wisdom_semantically(
                    query=question,
                    limit=3,
                    category=search_category
                )
            
            logger.info(f"Retrieved {len(wisdom_results)} wisdom items for query: '{question[:50]}...'")
            return wisdom_results
            
        except Exception as e:
            logger.warning(f"Failed to retrieve contextual wisdom: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check if LLM router is healthy"""
        return len(self.providers) > 0 and any(
            provider.get("status") == "ready" 
            for provider in self.providers.values()
        )
    
    async def analyze_feedback(
        self,
        feedback_type: str,
        title: str,
        content: str,
        ratings: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """Analyze user feedback using LLM for insights and categorization"""
        
        try:
            # Build analysis prompt
            analysis_prompt = self._build_feedback_analysis_prompt(
                feedback_type, title, content, ratings
            )
            
            # Generate analysis using LLM
            response = await self.generate_response(
                message=analysis_prompt,
                context="feedback_analysis",
                max_tokens=1000,
                temperature=0.2  # Lower temperature for more consistent analysis
            )
            
            # Parse the response
            analysis_result = await self._parse_feedback_analysis(response.response)
            
            logger.info(f"Feedback analysis completed for type: {feedback_type}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing feedback: {e}")
            return self._fallback_feedback_analysis(feedback_type, content)
    
    def _build_feedback_analysis_prompt(
        self,
        feedback_type: str,
        title: str,
        content: str,
        ratings: Optional[Dict[str, int]] = None
    ) -> str:
        """Build comprehensive feedback analysis prompt"""
        
        ratings_text = ""
        if ratings:
            ratings_text = f"""
Ratings provided:
- Overall: {ratings.get('overall_rating', 'N/A')}/5
- Response Quality: {ratings.get('response_quality', 'N/A')}/5
- Helpfulness: {ratings.get('helpfulness', 'N/A')}/5
- Spiritual Value: {ratings.get('spiritual_value', 'N/A')}/5
"""
        
        return f"""
Analyze this user feedback for DharmaMind, an AI spiritual guidance system.

Provide analysis in JSON format with these exact keys:

{{
    "sentiment": "positive|neutral|negative",
    "sentiment_score": 0.0-1.0,
    "priority_score": 0.0-1.0,
    "urgency_level": "low|medium|high|critical",
    "key_topics": ["topic1", "topic2", ...],
    "mentioned_features": ["feature1", "feature2", ...],
    "suggestions": ["suggestion1", "suggestion2", ...],
    "issues_identified": ["issue1", "issue2", ...],
    "dharmic_concerns": ["concern1", "concern2", ...],
    "spiritual_insights": ["insight1", "insight2", ...],
    "user_intent": "seeking_help|reporting_bug|suggesting_feature|expressing_satisfaction|other",
    "category_confidence": 0.0-1.0,
    "actionable_items": ["action1", "action2", ...],
    "response_recommendation": "acknowledge|investigate|implement|forward_to_team|follow_up"
}}

Feedback Details:
Type: {feedback_type}
Title: {title}
Content: {content}
{ratings_text}

Focus on:
1. Spiritual and dharmic context awareness
2. Technical aspects and user experience
3. Actionable insights for improvement
4. Priority based on impact and urgency
5. Dharmic compliance concerns if any

Analyze thoroughly but be concise in extracted items.
"""
    
    async def _parse_feedback_analysis(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM feedback analysis response"""
        
        try:
            # In a real implementation, would parse JSON from LLM response
            # For now, return structured mock analysis based on response content
            
            # Extract key patterns
            response_lower = llm_response.lower()
            
            # Sentiment analysis
            sentiment = "neutral"
            sentiment_score = 0.5
            
            positive_indicators = ["good", "great", "excellent", "helpful", "love", "amazing", "perfect"]
            negative_indicators = ["bad", "terrible", "horrible", "useless", "hate", "awful", "broken"]
            
            positive_count = sum(1 for word in positive_indicators if word in response_lower)
            negative_count = sum(1 for word in negative_indicators if word in response_lower)
            
            if positive_count > negative_count:
                sentiment = "positive"
                sentiment_score = 0.7 + (positive_count * 0.1)
            elif negative_count > positive_count:
                sentiment = "negative"
                sentiment_score = 0.3 - (negative_count * 0.1)
            
            sentiment_score = max(0.0, min(1.0, sentiment_score))
            
            # Priority scoring
            priority_score = 0.3
            if "urgent" in response_lower or "critical" in response_lower:
                priority_score += 0.4
            if "bug" in response_lower or "error" in response_lower:
                priority_score += 0.3
            if "slow" in response_lower or "performance" in response_lower:
                priority_score += 0.2
            
            priority_score = min(1.0, priority_score)
            
            # Extract topics and features
            topics = self._extract_topics_from_text(llm_response)
            features = self._extract_features_from_text(llm_response)
            suggestions = self._extract_suggestions_from_text(llm_response)
            issues = self._extract_issues_from_text(llm_response)
            
            return {
                "sentiment": sentiment,
                "sentiment_score": sentiment_score,
                "priority_score": priority_score,
                "urgency_level": "high" if priority_score > 0.7 else "medium" if priority_score > 0.4 else "low",
                "key_topics": topics,
                "mentioned_features": features,
                "suggestions": suggestions,
                "issues_identified": issues,
                "dharmic_concerns": self._extract_dharmic_concerns(llm_response),
                "spiritual_insights": self._extract_spiritual_insights(llm_response),
                "user_intent": self._determine_user_intent(llm_response),
                "category_confidence": 0.8,
                "actionable_items": suggestions + issues,
                "response_recommendation": self._get_response_recommendation(priority_score, sentiment)
            }
            
        except Exception as e:
            logger.error(f"Error parsing feedback analysis: {e}")
            return self._fallback_feedback_analysis("general", llm_response)
    
    def _extract_topics_from_text(self, text: str) -> List[str]:
        """Extract key topics from text"""
        topics = []
        text_lower = text.lower()
        
        topic_keywords = {
            "meditation": ["meditat", "mindful", "awareness"],
            "guidance": ["guidance", "advice", "help", "support"],
            "response_quality": ["response", "answer", "quality", "accuracy"],
            "user_interface": ["interface", "ui", "design", "navigation", "layout"],
            "performance": ["slow", "fast", "speed", "performance", "loading"],
            "spiritual_content": ["spiritual", "dharma", "wisdom", "enlightenment"],
            "features": ["feature", "functionality", "capability"],
            "bugs": ["bug", "error", "crash", "broken", "issue"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics[:5]  # Limit to top 5 topics
    
    def _extract_features_from_text(self, text: str) -> List[str]:
        """Extract mentioned features from text"""
        features = []
        text_lower = text.lower()
        
        feature_keywords = {
            "chat": ["chat", "conversation", "messaging"],
            "search": ["search", "find", "lookup"],  
            "settings": ["settings", "preferences", "configuration"],
            "meditation_guide": ["meditation", "guide", "practice"],
            "wisdom_quotes": ["quotes", "wisdom", "sayings"],
            "user_profile": ["profile", "account", "user"]
        }
        
        for feature, keywords in feature_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                features.append(feature)
        
        return features[:3]  # Limit to top 3 features
    
    def _extract_suggestions_from_text(self, text: str) -> List[str]:
        """Extract suggestions from text"""
        suggestions = []
        text_lower = text.lower()
        
        suggestion_patterns = [
            "should", "could", "would be better", "suggest", "recommend",
            "improve", "add", "include", "feature", "enhancement"
        ]
        
        for pattern in suggestion_patterns:
            if pattern in text_lower:
                # Extract context around suggestion
                words = text.split()
                for i, word in enumerate(words):
                    if pattern in word.lower():
                        start = max(0, i - 3)
                        end = min(len(words), i + 8)
                        suggestion = " ".join(words[start:end])
                        suggestions.append(suggestion)
                        break
        
        return suggestions[:3]  # Limit to top 3 suggestions
    
    def _extract_issues_from_text(self, text: str) -> List[str]:
        """Extract issues from text"""
        issues = []
        text_lower = text.lower()
        
        issue_keywords = [
            "problem", "issue", "bug", "error", "broken", "not working",
            "slow", "crash", "freeze", "stuck", "confusing", "difficult"
        ]
        
        for keyword in issue_keywords:
            if keyword in text_lower:
                issues.append(f"User reported: {keyword}")
        
        return list(set(issues))[:3]  # Remove duplicates and limit
    
    def _extract_dharmic_concerns(self, text: str) -> List[str]:
        """Extract dharmic or spiritual concerns"""
        concerns = []
        text_lower = text.lower()
        
        concern_keywords = {
            "inappropriate_content": ["inappropriate", "offensive", "wrong"],
            "spiritual_accuracy": ["inaccurate", "misleading", "wrong teaching"],
            "ethical_concerns": ["unethical", "harmful", "concerning"],
            "dharmic_violations": ["against dharma", "non-dharmic", "spiritually wrong"]
        }
        
        for concern_type, keywords in concern_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                concerns.append(concern_type)
        
        return concerns
    
    def _extract_spiritual_insights(self, text: str) -> List[str]:
        """Extract spiritual insights from feedback"""
        insights = []
        text_lower = text.lower()
        
        spiritual_indicators = {
            "seeking_wisdom": ["wisdom", "knowledge", "understanding", "learning"],
            "spiritual_growth": ["growth", "development", "progress", "journey"],
            "meditation_interest": ["meditation", "mindfulness", "practice"],
            "dharmic_living": ["dharma", "righteous", "ethical", "moral"],
            "inner_peace": ["peace", "calm", "serenity", "tranquil"],
            "compassion": ["compassion", "kindness", "empathy", "caring"]
        }
        
        for insight_type, keywords in spiritual_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                insights.append(f"User showing interest in {insight_type}")
        
        return insights[:3]  # Limit to top 3 insights
    
    def _determine_user_intent(self, text: str) -> str:
        """Determine primary user intent"""
        text_lower = text.lower()
        
        intent_patterns = {
            "seeking_help": ["help", "support", "guidance", "how to"],
            "reporting_bug": ["bug", "error", "broken", "crash", "issue"],
            "suggesting_feature": ["suggest", "add", "feature", "would like", "request"],
            "expressing_satisfaction": ["thank", "great", "excellent", "love", "amazing"],
            "expressing_dissatisfaction": ["disappointed", "frustrated", "bad", "terrible"]
        }
        
        for intent, patterns in intent_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return intent
        
        return "other"
    
    def _get_response_recommendation(self, priority_score: float, sentiment: str) -> str:
        """Get recommended response action"""
        if priority_score > 0.8:
            return "investigate"
        elif priority_score > 0.6:
            return "follow_up"
        elif sentiment == "negative":
            return "acknowledge"
        elif sentiment == "positive":
            return "acknowledge"
        else:
            return "forward_to_team"
    
    def _fallback_feedback_analysis(self, feedback_type: str, content: str) -> Dict[str, Any]:
        """Fallback analysis when LLM fails"""
        return {
            "sentiment": "neutral",
            "sentiment_score": 0.5,
            "priority_score": 0.3 if feedback_type == "bug_report" else 0.2,
            "urgency_level": "medium" if feedback_type == "bug_report" else "low",
            "key_topics": ["general"],
            "mentioned_features": [],
            "suggestions": [],
            "issues_identified": [],
            "dharmic_concerns": [],
            "spiritual_insights": [],
            "user_intent": "other",
            "category_confidence": 0.1,
            "actionable_items": [],
            "response_recommendation": "acknowledge"
        }


# Dependency injection function for FastAPI
_llm_router_instance = None

def get_llm_router() -> LLMRouter:
    """Get the LLM router instance (singleton pattern)"""
    global _llm_router_instance
    if _llm_router_instance is None:
        _llm_router_instance = LLMRouter()
    return _llm_router_instance
