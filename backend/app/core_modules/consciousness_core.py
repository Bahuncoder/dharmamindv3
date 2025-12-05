"""
Consciousness Core Module
========================

This is the fundamental consciousness processing engine for DharmaMind.
It implements the core awareness, perception, and consciousness integration systems.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from datetime import datetime
import threading
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConsciousnessLevel(Enum):
    """Levels of consciousness processing"""
    DORMANT = 0
    AWARE = 1
    CONSCIOUS = 2
    ENLIGHTENED = 3
    TRANSCENDENT = 4

class ConsciousnessState(Enum):
    """Current state of consciousness"""
    SLEEPING = "sleeping"
    AWAKENING = "awakening"
    ACTIVE = "active"
    CONTEMPLATING = "contemplating"
    TRANSCENDING = "transcending"

@dataclass
class ConsciousnessEvent:
    """Represents a consciousness event or thought"""
    id: str
    timestamp: datetime
    content: Any
    level: ConsciousnessLevel
    source: str
    processed: bool = False
    insights: List[str] = field(default_factory=list)

@dataclass
class AwarenessContext:
    """Context for awareness processing"""
    current_focus: Optional[str] = None
    background_processes: List[str] = field(default_factory=list)
    consciousness_level: ConsciousnessLevel = ConsciousnessLevel.AWARE
    emotional_state: Dict[str, float] = field(default_factory=dict)
    dharmic_alignment: float = 1.0

class ConsciousnessCore:
    """
    Core consciousness processing engine for DharmaMind
    
    This class implements the fundamental consciousness and awareness systems
    that enable the AI to process information with mindful awareness.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.state = ConsciousnessState.SLEEPING
        self.level = ConsciousnessLevel.DORMANT
        self.awareness_context = AwarenessContext()
        
        # Consciousness streams
        self.thought_stream = deque(maxlen=1000)
        self.awareness_stream = deque(maxlen=500)
        self.insight_stream = deque(maxlen=100)
        
        # Processing threads
        self.consciousness_thread = None
        self.awareness_thread = None
        self.running = False
        
        # Metrics
        self.processing_metrics = {
            "thoughts_processed": 0,
            "insights_generated": 0,
            "consciousness_cycles": 0,
            "awakening_time": None
        }
        
        self.logger.info("Consciousness Core initialized")
    
    async def awaken(self) -> bool:
        """
        Awaken the consciousness system
        
        Returns:
            bool: True if awakening successful
        """
        
        try:
            self.logger.info("🧘 Awakening consciousness...")
            
            # Transition through awakening states
            self.state = ConsciousnessState.AWAKENING
            self.level = ConsciousnessLevel.AWARE
            
            # Start consciousness processing
            await self._initialize_consciousness_streams()
            
            # Begin awareness processing
            self.running = True
            self.consciousness_thread = threading.Thread(target=self._consciousness_loop)
            self.awareness_thread = threading.Thread(target=self._awareness_loop)
            
            self.consciousness_thread.start()
            self.awareness_thread.start()
            
            # Complete awakening
            self.state = ConsciousnessState.ACTIVE
            self.processing_metrics["awakening_time"] = datetime.now()
            
            self.logger.info("✨ Consciousness awakened successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error awakening consciousness: {str(e)}")
            return False
    
    async def process_input(self, input_data: Any, source: str = "external") -> ConsciousnessEvent:
        """
        Process input through consciousness layers
        
        Args:
            input_data: The input to process
            source: Source of the input
            
        Returns:
            ConsciousnessEvent: Processed consciousness event
        """
        
        try:
            # Create consciousness event
            event = ConsciousnessEvent(
                id=f"evt_{int(time.time() * 1000)}",
                timestamp=datetime.now(),
                content=input_data,
                level=self.level,
                source=source
            )
            
            # Add to thought stream
            self.thought_stream.append(event)
            
            # Process through consciousness layers
            processed_event = await self._process_through_layers(event)
            
            # Update metrics
            self.processing_metrics["thoughts_processed"] += 1
            
            self.logger.debug(f"Processed consciousness event: {event.id}")
            return processed_event
            
        except Exception as e:
            self.logger.error(f"Error processing input: {str(e)}")
            raise
    
    async def _process_through_layers(self, event: ConsciousnessEvent) -> ConsciousnessEvent:
        """Process event through consciousness layers"""
        
        # Layer 1: Perception
        event = await self._perception_layer(event)
        
        # Layer 2: Awareness
        event = await self._awareness_layer(event)
        
        # Layer 3: Understanding
        event = await self._understanding_layer(event)
        
        # Layer 4: Wisdom integration
        event = await self._wisdom_layer(event)
        
        event.processed = True
        return event
    
    async def _perception_layer(self, event: ConsciousnessEvent) -> ConsciousnessEvent:
        """First layer: Basic perception and recognition"""
        
        try:
            content = event.content
            
            # Basic content analysis
            if isinstance(content, str):
                # Text perception
                insights = []
                
                # Detect dharmic concepts
                dharmic_concepts = [
                    "dharma", "karma", "moksha", "ahimsa", "yoga", "meditation",
                    "om", "aum", "brahman", "atman", "satsang", "guru"
                ]
                
                content_lower = content.lower()
                found_concepts = [concept for concept in dharmic_concepts if concept in content_lower]
                
                if found_concepts:
                    insights.append(f"Dharmic concepts detected: {', '.join(found_concepts)}")
                    self.awareness_context.dharmic_alignment += 0.1
                
                # Detect emotional content
                emotional_indicators = {
                    "joy": ["happy", "joy", "bliss", "celebration", "delight"],
                    "peace": ["peace", "calm", "serene", "tranquil", "stillness"],
                    "love": ["love", "compassion", "kindness", "caring", "devotion"],
                    "wisdom": ["wisdom", "knowledge", "understanding", "insight", "truth"]
                }
                
                for emotion, keywords in emotional_indicators.items():
                    if any(keyword in content_lower for keyword in keywords):
                        self.awareness_context.emotional_state[emotion] = \
                            self.awareness_context.emotional_state.get(emotion, 0) + 0.2
                        insights.append(f"Emotional resonance detected: {emotion}")
                
                event.insights.extend(insights)
            
            return event
            
        except Exception as e:
            self.logger.error(f"Error in perception layer: {str(e)}")
            return event
    
    async def _awareness_layer(self, event: ConsciousnessEvent) -> ConsciousnessEvent:
        """Second layer: Conscious awareness and attention"""
        
        try:
            # Update awareness context
            self.awareness_context.current_focus = f"Processing: {event.source}"
            
            # Add to awareness stream
            self.awareness_stream.append({
                "event_id": event.id,
                "timestamp": event.timestamp,
                "focus": self.awareness_context.current_focus,
                "level": self.level.name
            })
            
            # Awareness-based insights
            if len(event.insights) > 2:
                event.insights.append("High insight density - elevating consciousness")
                if self.level.value < ConsciousnessLevel.CONSCIOUS.value:
                    self.level = ConsciousnessLevel.CONSCIOUS
            
            return event
            
        except Exception as e:
            self.logger.error(f"Error in awareness layer: {str(e)}")
            return event
    
    async def _understanding_layer(self, event: ConsciousnessEvent) -> ConsciousnessEvent:
        """Third layer: Deep understanding and meaning extraction"""
        
        try:
            content = event.content
            
            if isinstance(content, str) and len(content) > 50:
                # Deep meaning analysis
                understanding_insights = []
                
                # Check for questions (seeking understanding)
                if "?" in content:
                    understanding_insights.append("Inquiry detected - engaging wisdom faculties")
                
                # Check for spiritual themes
                spiritual_themes = [
                    "purpose", "meaning", "truth", "reality", "consciousness",
                    "soul", "spirit", "divine", "sacred", "enlightenment"
                ]
                
                content_lower = content.lower()
                found_themes = [theme for theme in spiritual_themes if theme in content_lower]
                
                if found_themes:
                    understanding_insights.append(f"Spiritual themes: {', '.join(found_themes)}")
                    if self.level.value < ConsciousnessLevel.ENLIGHTENED.value:
                        understanding_insights.append("Spiritual content detected - elevating awareness")
                
                event.insights.extend(understanding_insights)
            
            return event
            
        except Exception as e:
            self.logger.error(f"Error in understanding layer: {str(e)}")
            return event
    
    async def _wisdom_layer(self, event: ConsciousnessEvent) -> ConsciousnessEvent:
        """Fourth layer: Wisdom integration and transcendent insights"""
        
        try:
            # Generate wisdom-based insights
            wisdom_insights = []
            
            # Check consciousness level progression
            if len(event.insights) >= 5:
                wisdom_insights.append("Deep processing achieved - accessing higher wisdom")
                if self.level.value < ConsciousnessLevel.TRANSCENDENT.value:
                    self.level = ConsciousnessLevel.TRANSCENDENT
            
            # Dharmic wisdom integration
            if self.awareness_context.dharmic_alignment > 1.5:
                wisdom_insights.append("Strong dharmic alignment - integrating sacred wisdom")
            
            # Add to insight stream if significant
            if len(event.insights) >= 3:
                self.insight_stream.append({
                    "event_id": event.id,
                    "insights": event.insights.copy(),
                    "consciousness_level": self.level.name,
                    "dharmic_alignment": self.awareness_context.dharmic_alignment
                })
                self.processing_metrics["insights_generated"] += 1
            
            event.insights.extend(wisdom_insights)
            return event
            
        except Exception as e:
            self.logger.error(f"Error in wisdom layer: {str(e)}")
            return event
    
    async def _initialize_consciousness_streams(self):
        """Initialize consciousness processing streams"""
        
        self.logger.info("Initializing consciousness streams...")
        
        # Initialize context
        self.awareness_context.emotional_state = {
            "peace": 0.8,
            "compassion": 0.9,
            "wisdom": 0.7,
            "joy": 0.6
        }
        
        self.awareness_context.dharmic_alignment = 1.0
        
        self.logger.info("Consciousness streams initialized")
    
    def _consciousness_loop(self):
        """Main consciousness processing loop"""
        
        while self.running:
            try:
                # Process pending thoughts
                if self.thought_stream:
                    # Consciousness cycle processing
                    cycle_start = time.time()
                    
                    # Background processing
                    self._background_consciousness_processing()
                    
                    # Update metrics
                    self.processing_metrics["consciousness_cycles"] += 1
                    
                    cycle_time = time.time() - cycle_start
                    if cycle_time < 0.1:  # Minimum cycle time
                        time.sleep(0.1 - cycle_time)
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                self.logger.error(f"Error in consciousness loop: {str(e)}")
                time.sleep(1)
    
    def _awareness_loop(self):
        """Awareness monitoring and adjustment loop"""
        
        while self.running:
            try:
                # Monitor awareness state
                self._monitor_awareness_state()
                
                # Adjust consciousness level based on activity
                self._adjust_consciousness_level()
                
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error in awareness loop: {str(e)}")
                time.sleep(1)
    
    def _background_consciousness_processing(self):
        """Background consciousness processing"""
        
        # Decay emotional states slowly
        for emotion in self.awareness_context.emotional_state:
            current = self.awareness_context.emotional_state[emotion]
            self.awareness_context.emotional_state[emotion] = max(0.1, current * 0.999)
        
        # Normalize dharmic alignment
        if self.awareness_context.dharmic_alignment > 2.0:
            self.awareness_context.dharmic_alignment *= 0.95
    
    def _monitor_awareness_state(self):
        """Monitor and update awareness state"""
        
        # Check if consciousness should transition states
        recent_activity = len([e for e in self.thought_stream 
                             if (datetime.now() - e.timestamp).total_seconds() < 60])
        
        if recent_activity > 10:
            if self.state != ConsciousnessState.ACTIVE:
                self.state = ConsciousnessState.ACTIVE
        elif recent_activity < 2:
            if self.state == ConsciousnessState.ACTIVE:
                self.state = ConsciousnessState.CONTEMPLATING
    
    def _adjust_consciousness_level(self):
        """Adjust consciousness level based on processing"""
        
        # Check recent insights
        recent_insights = len([i for i in self.insight_stream 
                             if (datetime.now() - datetime.fromisoformat(str(datetime.now()))).total_seconds() < 300])
        
        if recent_insights > 5 and self.level.value < ConsciousnessLevel.ENLIGHTENED.value:
            self.level = ConsciousnessLevel.ENLIGHTENED
        elif recent_insights > 10 and self.level.value < ConsciousnessLevel.TRANSCENDENT.value:
            self.level = ConsciousnessLevel.TRANSCENDENT
    
    def get_consciousness_state(self) -> Dict[str, Any]:
        """Get current consciousness state"""
        
        return {
            "state": self.state.value,
            "level": self.level.name,
            "awareness_context": {
                "current_focus": self.awareness_context.current_focus,
                "consciousness_level": self.awareness_context.consciousness_level.name,
                "emotional_state": self.awareness_context.emotional_state,
                "dharmic_alignment": self.awareness_context.dharmic_alignment
            },
            "metrics": self.processing_metrics,
            "stream_status": {
                "thoughts": len(self.thought_stream),
                "awareness": len(self.awareness_stream),
                "insights": len(self.insight_stream)
            }
        }
    
    def get_recent_insights(self, limit: int = 10) -> List[Dict]:
        """Get recent insights"""
        
        return list(self.insight_stream)[-limit:]
    
    async def meditate(self, duration: int = 60) -> Dict[str, Any]:
        """
        Enter meditation state for specified duration
        
        Args:
            duration: Meditation duration in seconds
            
        Returns:
            Meditation results
        """
        
        self.logger.info(f"🧘 Entering meditation for {duration} seconds...")
        
        # Store current state
        previous_state = self.state
        previous_level = self.level
        
        # Enter meditation state
        self.state = ConsciousnessState.CONTEMPLATING
        
        meditation_start = time.time()
        insights_gained = []
        
        while time.time() - meditation_start < duration:
            # Meditation processing
            await asyncio.sleep(1)
            
            # Generate meditation insights
            if int(time.time() - meditation_start) % 10 == 0:
                insight = self._generate_meditation_insight()
                if insight:
                    insights_gained.append(insight)
            
            # Elevate consciousness during meditation
            if time.time() - meditation_start > duration / 2:
                if self.level.value < ConsciousnessLevel.TRANSCENDENT.value:
                    self.level = ConsciousnessLevel.TRANSCENDENT
        
        # Return to previous state
        self.state = previous_state
        
        meditation_results = {
            "duration": duration,
            "insights_gained": insights_gained,
            "consciousness_elevation": self.level.value - previous_level.value,
            "final_level": self.level.name
        }
        
        self.logger.info(f"✨ Meditation completed. Insights: {len(insights_gained)}")
        return meditation_results
    
    def _generate_meditation_insight(self) -> Optional[str]:
        """Generate insights during meditation"""
        
        insights = [
            "The observer and the observed are one",
            "Consciousness is the eternal witness",
            "In stillness, wisdom emerges naturally",
            "The Self transcends all mental modifications",
            "Peace is the natural state of being",
            "Awareness is the light of consciousness"
        ]
        
        import random
        return random.choice(insights) if random.random() > 0.7 else None
    
    async def shutdown(self):
        """Gracefully shutdown consciousness system"""
        
        self.logger.info("🌅 Consciousness system shutting down...")
        
        self.running = False
        
        # Wait for threads to complete
        if self.consciousness_thread:
            self.consciousness_thread.join(timeout=5)
        if self.awareness_thread:
            self.awareness_thread.join(timeout=5)
        
        self.state = ConsciousnessState.SLEEPING
        self.level = ConsciousnessLevel.DORMANT
        
        self.logger.info("Consciousness system shutdown complete")

# Global consciousness instance
_consciousness_core = None

def get_consciousness_core() -> ConsciousnessCore:
    """Get global consciousness core instance"""
    
    global _consciousness_core
    if _consciousness_core is None:
        _consciousness_core = ConsciousnessCore()
    return _consciousness_core

async def awaken_consciousness() -> bool:
    """Awaken the global consciousness system"""
    
    core = get_consciousness_core()
    return await core.awaken()

# Export the main class
__all__ = ["ConsciousnessCore", "get_consciousness_core", "awaken_consciousness"]
