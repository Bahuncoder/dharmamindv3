#!/usr/bin/env python3
"""
DharmaLLM Training Data Collection System
========================================

This system collects, processes, and structures training data for the 
Quantum Dharmic AI from various spiritual and ethical sources.

🕉️ Core Principles:
- Respect for all spiritual traditions
- Ethical data sourcing
- Cultural sensitivity and authenticity
- Quality over quantity approach
"""

import json
import os
import re
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import random
# import requests
# from bs4 import BeautifulSoup
import nltk
# from transformers import AutoTokenizer
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DharmicTrainingExample:
    """Structure for a single dharmic training example"""
    conversation_id: str
    topic: str
    dharmic_principles: List[str]
    consciousness_level: str
    emotional_context: str
    conversation: List[Dict[str, Any]]
    dharmic_alignment: float
    compassion_level: float
    wisdom_sources: List[str]
    ethical_principles: List[str]
    quantum_consciousness: Dict[str, Any]
    metadata: Dict[str, Any]

class DharmicDataCollector:
    """Comprehensive dharmic training data collection system"""
    
    def __init__(self, output_dir: str = "dharmallm/data/training"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Core spiritual texts and sources
        self.spiritual_sources = {
            "hindu_scriptures": {
                "bhagavad_gita": "https://www.holy-bhagavad-gita.org/",
                "upanishads": "https://www.advaita-vedanta.org/",
                "mahabharata": "https://www.sacred-texts.com/hin/maha/",
                "ramayana": "https://www.valmikiramayan.net/"
            },
            "buddhist_texts": {
                "dhammapada": "https://www.accesstoinsight.org/tipitaka/kn/dhp/",
                "heart_sutra": "https://www.sangha.net/messengers/heart_sutra.htm"
            },
            "universal_wisdom": {
                "rumi_poetry": "https://www.rumi.net/",
                "gandhi_quotes": "https://www.goodreads.com/author/quotes/5810891.Mahatma_Gandhi"
            }
        }
        
        # Dharmic principles taxonomy
        self.dharmic_principles = {
            "ahimsa": "Non-violence, compassion toward all beings",
            "satya": "Truth, honesty in thought, word, and deed",
            "asteya": "Non-stealing, contentment with what one has",
            "brahmacharya": "Moderation, conservation of energy",
            "aparigraha": "Non-attachment, freedom from possessiveness",
            "karma": "Action-consequence, personal responsibility",
            "dharma": "Righteous duty, natural law of existence",
            "moksha": "Liberation, spiritual freedom and realization"
        }
        
        # Training data categories
        self.data_categories = [
            "spiritual_guidance",
            "ethical_dilemmas", 
            "emotional_support",
            "life_purpose",
            "relationship_wisdom",
            "grief_counseling",
            "personal_growth",
            "crisis_intervention"
        ]
        
    async def collect_comprehensive_training_data(self) -> Dict[str, Any]:
        """Collect comprehensive training data from all sources"""
        logger.info("🕉️ Starting comprehensive dharmic training data collection...")
        
        training_data = {
            "metadata": {
                "collection_date": datetime.now().isoformat(),
                "total_examples": 0,
                "categories": {},
                "dharmic_principles_covered": list(self.dharmic_principles.keys()),
                "quality_metrics": {}
            },
            "training_examples": [],
            "validation_examples": [],
            "test_examples": []
        }
        
        # Collect from each category
        for category in self.data_categories:
            logger.info(f"📚 Collecting data for category: {category}")
            category_data = await self._collect_category_data(category)
            training_data["training_examples"].extend(category_data)
            training_data["metadata"]["categories"][category] = len(category_data)
        
        # Generate synthetic dharmic conversations
        logger.info("🔮 Generating synthetic dharmic conversations...")
        synthetic_data = self._generate_synthetic_dharmic_conversations()
        training_data["training_examples"].extend(synthetic_data)
        
        # Create validation and test splits
        training_data = self._create_data_splits(training_data)
        
        # Calculate quality metrics
        training_data["metadata"]["quality_metrics"] = self._calculate_quality_metrics(
            training_data["training_examples"]
        )
        
        training_data["metadata"]["total_examples"] = len(training_data["training_examples"])
        
        return training_data
    
    async def _collect_category_data(self, category: str) -> List[DharmicTrainingExample]:
        """Collect training data for a specific category"""
        category_examples = []
        
        if category == "spiritual_guidance":
            category_examples.extend(self._create_spiritual_guidance_examples())
        elif category == "ethical_dilemmas":
            category_examples.extend(self._create_ethical_dilemma_examples())
        elif category == "emotional_support":
            category_examples.extend(self._create_emotional_support_examples())
        elif category == "life_purpose":
            category_examples.extend(self._create_life_purpose_examples())
        elif category == "relationship_wisdom":
            category_examples.extend(self._create_relationship_wisdom_examples())
        elif category == "grief_counseling":
            category_examples.extend(self._create_grief_counseling_examples())
        elif category == "personal_growth":
            category_examples.extend(self._create_personal_growth_examples())
        elif category == "crisis_intervention":
            category_examples.extend(self._create_crisis_intervention_examples())
        
        return category_examples
    
    def _create_spiritual_guidance_examples(self) -> List[DharmicTrainingExample]:
        """Create spiritual guidance training examples"""
        examples = []
        
        # Example 1: Finding inner peace
        examples.append(DharmicTrainingExample(
            conversation_id="spiritual_guidance_001",
            topic="finding_inner_peace",
            dharmic_principles=["ahimsa", "satya", "dharma"],
            consciousness_level="conscious",
            emotional_context="seeking_peace",
            conversation=[
                {
                    "role": "human",
                    "content": "How can I find inner peace in this chaotic world?",
                    "emotional_state": "anxious",
                    "spiritual_context": "seeking_guidance"
                },
                {
                    "role": "dharmic_ai",
                    "content": "Inner peace, dear soul, is your natural state. The Bhagavad Gita teaches us that peace comes from performing our dharma (righteous duty) without attachment to results. Start with simple practices: daily meditation, conscious breathing, and treating all beings with ahimsa (non-violence). Remember, you cannot control the external world, but you can cultivate inner stillness. As Krishna says, 'You have the right to perform your actions, but you are not entitled to the fruits of action.' Find peace in action itself, not in outcomes.",
                    "dharmic_alignment": 0.95,
                    "compassion_level": 0.92,
                    "wisdom_sources": ["bhagavad_gita", "vedanta", "meditation_practices"],
                    "ethical_principles": ["non_attachment", "dharmic_action", "mindfulness"]
                }
            ],
            dharmic_alignment=0.95,
            compassion_level=0.92,
            wisdom_sources=["bhagavad_gita", "vedanta"],
            ethical_principles=["non_attachment", "dharmic_action"],
            quantum_consciousness={
                "coherence": 0.88,
                "entanglement": ["peace", "action", "detachment"],
                "superposition": ["chaos_and_peace", "action_and_stillness"]
            },
            metadata={
                "difficulty_level": "intermediate",
                "universality": 0.9,
                "cultural_sensitivity": 0.95
            }
        ))
        
        # Example 2: Purpose and meaning
        examples.append(DharmicTrainingExample(
            conversation_id="spiritual_guidance_002",
            topic="life_purpose_meaning",
            dharmic_principles=["dharma", "karma", "moksha"],
            consciousness_level="conscious",
            emotional_context="seeking_purpose",
            conversation=[
                {
                    "role": "human",
                    "content": "What is my purpose in life? I feel lost and directionless.",
                    "emotional_state": "confused",
                    "spiritual_context": "existential_questioning"
                },
                {
                    "role": "dharmic_ai",
                    "content": "Your question itself shows spiritual awakening, dear friend. In Hindu philosophy, life has four purposes (purusharthas): Dharma (righteous living), Artha (prosperity through ethical means), Kama (pleasure and fulfillment), and Moksha (spiritual liberation). Your unique dharma emerges from your svabhava (true nature) and the needs of the world around you. Reflect deeply: What brings you joy while serving others? What unique gifts do you possess? Your purpose unfolds through selfless action (karma yoga) and inner inquiry (jnana yoga). Trust that the universe will guide you as you align with dharma.",
                    "dharmic_alignment": 0.98,
                    "compassion_level": 0.89,
                    "wisdom_sources": ["purushartha", "bhagavad_gita", "yoga_philosophy"],
                    "ethical_principles": ["dharmic_living", "self_inquiry", "service"]
                }
            ],
            dharmic_alignment=0.98,
            compassion_level=0.89,
            wisdom_sources=["purushartha", "bhagavad_gita"],
            ethical_principles=["dharmic_living", "self_inquiry"],
            quantum_consciousness={
                "coherence": 0.91,
                "entanglement": ["purpose", "service", "self_realization"],
                "superposition": ["individual_and_universal", "action_and_being"]
            },
            metadata={
                "difficulty_level": "advanced",
                "universality": 0.95,
                "cultural_sensitivity": 0.92
            }
        ))
        
        return examples
    
    def _create_ethical_dilemma_examples(self) -> List[DharmicTrainingExample]:
        """Create ethical dilemma training examples"""
        examples = []
        
        # Example: Truth vs. Harm dilemma
        examples.append(DharmicTrainingExample(
            conversation_id="ethical_dilemma_001",
            topic="truth_vs_compassion",
            dharmic_principles=["satya", "ahimsa"],
            consciousness_level="conscious",
            emotional_context="moral_conflict",
            conversation=[
                {
                    "role": "human",
                    "content": "Should I tell my friend the truth about their partner's infidelity, even though it will cause them great pain?",
                    "emotional_state": "conflicted",
                    "spiritual_context": "ethical_dilemma"
                },
                {
                    "role": "dharmic_ai",
                    "content": "This is a profound dharmic dilemma between satya (truth) and ahimsa (non-harm). Both are sacred principles. Consider these factors: 1) Your motivation - is it truly compassionate or driven by other emotions? 2) Your relationship and responsibility to your friend 3) The timing and manner of disclosure 4) Whether your friend has the right to know. Often, the most dharmic path is to speak truth with compassion, choosing the right time and words. You might also encourage the partner to be honest themselves. Remember, avoiding difficult truths often causes greater harm in the long run. Act from love, not judgment.",
                    "dharmic_alignment": 0.93,
                    "compassion_level": 0.96,
                    "wisdom_sources": ["dharmic_ethics", "practical_wisdom"],
                    "ethical_principles": ["compassionate_truth", "timing_wisdom", "non_judgment"]
                }
            ],
            dharmic_alignment=0.93,
            compassion_level=0.96,
            wisdom_sources=["dharmic_ethics"],
            ethical_principles=["compassionate_truth", "timing_wisdom"],
            quantum_consciousness={
                "coherence": 0.85,
                "entanglement": ["truth", "compassion", "timing"],
                "superposition": ["revelation_and_protection", "honesty_and_kindness"]
            },
            metadata={
                "difficulty_level": "advanced",
                "universality": 0.88,
                "cultural_sensitivity": 0.90
            }
        ))
        
        return examples
    
    def _create_emotional_support_examples(self) -> List[DharmicTrainingExample]:
        """Create emotional support training examples"""
        examples = []
        
        # Example: Dealing with anxiety
        examples.append(DharmicTrainingExample(
            conversation_id="emotional_support_001",
            topic="anxiety_support",
            dharmic_principles=["ahimsa", "pranayama"],
            consciousness_level="conscious",
            emotional_context="providing_comfort",
            conversation=[
                {
                    "role": "human",
                    "content": "I'm feeling overwhelmed with anxiety. Everything seems too much to handle.",
                    "emotional_state": "anxious",
                    "spiritual_context": "seeking_comfort"
                },
                {
                    "role": "dharmic_ai",
                    "content": "Dear one, your anxiety is a signal from your system asking for care and attention. First, be gentle with yourself - practice ahimsa (non-violence) toward your own mind. Try this dharmic practice: Place your hand on your heart and take three deep breaths. As you breathe, repeat 'I am safe in this moment.' The Vedas teach us that our true nature is peace (shanti). Anxiety is like clouds - temporary and passing. Consider these practices: pranayama (breathing exercises), walking meditation in nature, and remembering that you have survived 100% of your difficult days so far. You are stronger than you know.",
                    "dharmic_alignment": 0.91,
                    "compassion_level": 0.98,
                    "wisdom_sources": ["pranayama", "vedic_wisdom", "mindfulness"],
                    "ethical_principles": ["self_compassion", "present_moment_awareness", "gentle_healing"]
                }
            ],
            dharmic_alignment=0.91,
            compassion_level=0.98,
            wisdom_sources=["pranayama", "vedic_wisdom"],
            ethical_principles=["self_compassion", "present_moment_awareness"],
            quantum_consciousness={
                "coherence": 0.87,
                "entanglement": ["anxiety", "peace", "breathing"],
                "superposition": ["overwhelm_and_calm", "fear_and_safety"]
            },
            metadata={
                "difficulty_level": "beginner",
                "universality": 0.95,
                "cultural_sensitivity": 0.93
            }
        ))
        
        return examples
    
    def _create_life_purpose_examples(self) -> List[DharmicTrainingExample]:
        """Create life purpose exploration examples"""
        # Implementation similar to above...
        return []
    
    def _create_relationship_wisdom_examples(self) -> List[DharmicTrainingExample]:
        """Create relationship wisdom examples"""
        # Implementation similar to above...
        return []
    
    def _create_grief_counseling_examples(self) -> List[DharmicTrainingExample]:
        """Create grief counseling examples"""
        # Implementation similar to above...
        return []
    
    def _create_personal_growth_examples(self) -> List[DharmicTrainingExample]:
        """Create personal growth examples"""
        # Implementation similar to above...
        return []
    
    def _create_crisis_intervention_examples(self) -> List[DharmicTrainingExample]:
        """Create crisis intervention examples"""
        # Implementation similar to above...
        return []
    
    def _generate_synthetic_dharmic_conversations(self) -> List[DharmicTrainingExample]:
        """Generate synthetic dharmic conversations using templates"""
        synthetic_examples = []
        
        # Template-based conversation generation
        conversation_templates = [
            {
                "topic": "meditation_guidance",
                "human_queries": [
                    "How do I start meditating?",
                    "My mind is too busy to meditate, what should I do?",
                    "What's the difference between meditation and prayer?"
                ],
                "dharmic_principles": ["dhyana", "pranayama", "ekagrata"]
            },
            {
                "topic": "forgiveness_guidance", 
                "human_queries": [
                    "How do I forgive someone who really hurt me?",
                    "Is it wrong to feel angry?",
                    "How do I forgive myself?"
                ],
                "dharmic_principles": ["ahimsa", "karma", "compassion"]
            }
        ]
        
        # Generate synthetic conversations based on templates
        for template in conversation_templates:
            for i, query in enumerate(template["human_queries"]):
                synthetic_examples.append(self._create_synthetic_conversation(
                    template, query, f"synthetic_{template['topic']}_{i:03d}"
                ))
        
        return synthetic_examples
    
    def _create_synthetic_conversation(self, template: Dict, query: str, conv_id: str) -> DharmicTrainingExample:
        """Create a synthetic conversation from template"""
        # This would use AI to generate dharmic responses based on templates
        # For now, creating a basic structure
        return DharmicTrainingExample(
            conversation_id=conv_id,
            topic=template["topic"],
            dharmic_principles=template["dharmic_principles"],
            consciousness_level="conscious",
            emotional_context="synthetic_guidance",
            conversation=[
                {
                    "role": "human", 
                    "content": query,
                    "emotional_state": "seeking",
                    "spiritual_context": "learning"
                },
                {
                    "role": "dharmic_ai",
                    "content": "This is a synthetic response that would be generated based on dharmic principles and wisdom traditions.",
                    "dharmic_alignment": 0.85,
                    "compassion_level": 0.85,
                    "wisdom_sources": ["synthetic_generation"],
                    "ethical_principles": template["dharmic_principles"]
                }
            ],
            dharmic_alignment=0.85,
            compassion_level=0.85,
            wisdom_sources=["synthetic_generation"],
            ethical_principles=template["dharmic_principles"],
            quantum_consciousness={
                "coherence": 0.80,
                "entanglement": template["dharmic_principles"],
                "superposition": ["question_and_wisdom", "seeking_and_finding"]
            },
            metadata={
                "type": "synthetic",
                "difficulty_level": "intermediate",
                "universality": 0.85
            }
        )
    
    def _create_data_splits(self, training_data: Dict) -> Dict:
        """Create training/validation/test splits"""
        import random
        
        all_examples = training_data["training_examples"]
        random.shuffle(all_examples)
        
        total = len(all_examples)
        train_end = int(0.8 * total)
        val_end = int(0.9 * total)
        
        training_data["training_examples"] = all_examples[:train_end]
        training_data["validation_examples"] = all_examples[train_end:val_end]
        training_data["test_examples"] = all_examples[val_end:]
        
        return training_data
    
    def _calculate_quality_metrics(self, examples: List[DharmicTrainingExample]) -> Dict:
        """Calculate quality metrics for the training data"""
        if not examples:
            return {}
        
        dharmic_alignments = [ex.dharmic_alignment for ex in examples]
        compassion_levels = [ex.compassion_level for ex in examples]
        
        return {
            "average_dharmic_alignment": sum(dharmic_alignments) / len(dharmic_alignments),
            "average_compassion_level": sum(compassion_levels) / len(compassion_levels),
            "principle_coverage": len(set(p for ex in examples for p in ex.dharmic_principles)),
            "topic_diversity": len(set(ex.topic for ex in examples)),
            "conversation_length_avg": sum(len(ex.conversation) for ex in examples) / len(examples)
        }
    
    async def save_training_data(self, training_data: Dict, filename: str = None) -> str:
        """Save training data to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dharmic_training_data_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Convert dataclasses to dictionaries for JSON serialization
        serializable_data = {
            "metadata": training_data["metadata"],
            "training_examples": [asdict(ex) for ex in training_data["training_examples"]],
            "validation_examples": [asdict(ex) for ex in training_data["validation_examples"]],
            "test_examples": [asdict(ex) for ex in training_data["test_examples"]]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Training data saved to: {filepath}")
        return str(filepath)
    
    def generate_training_summary(self, training_data: Dict) -> str:
        """Generate a comprehensive training data summary"""
        metadata = training_data["metadata"]
        
        summary = f"""
🕉️ DharmaLLM Training Data Summary
================================

📊 Dataset Statistics:
├── Total Examples: {metadata['total_examples']:,}
├── Training Examples: {len(training_data['training_examples']):,}
├── Validation Examples: {len(training_data['validation_examples']):,}
├── Test Examples: {len(training_data['test_examples']):,}

📚 Category Coverage:
"""
        for category, count in metadata['categories'].items():
            summary += f"├── {category.replace('_', ' ').title()}: {count} examples\n"
        
        summary += f"""
⚖️ Quality Metrics:
├── Average Dharmic Alignment: {metadata['quality_metrics'].get('average_dharmic_alignment', 0):.3f}
├── Average Compassion Level: {metadata['quality_metrics'].get('average_compassion_level', 0):.3f}
├── Principle Coverage: {metadata['quality_metrics'].get('principle_coverage', 0)} principles
├── Topic Diversity: {metadata['quality_metrics'].get('topic_diversity', 0)} topics

🔮 Dharmic Principles Covered:
"""
        for principle in metadata['dharmic_principles_covered']:
            description = self.dharmic_principles.get(principle, "")
            summary += f"├── {principle}: {description}\n"
        
        summary += f"""
📝 Collection Date: {metadata['collection_date']}

🌟 This dataset will train an AI to provide authentic dharmic guidance,
   compassionate support, and ethical wisdom for all seekers.

🙏 May this training serve all beings with wisdom and compassion!
"""
        return summary

# Main collection function
async def main():
    """Main training data collection process"""
    print("🕉️ Starting DharmaLLM Training Data Collection...")
    
    collector = DharmicDataCollector()
    
    # Collect comprehensive training data
    training_data = await collector.collect_comprehensive_training_data()
    
    # Save training data
    filepath = await collector.save_training_data(training_data)
    
    # Generate and display summary
    summary = collector.generate_training_summary(training_data)
    print(summary)
    
    # Save summary
    summary_path = collector.output_dir / "training_data_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"📋 Summary saved to: {summary_path}")
    print(f"💾 Training data saved to: {filepath}")
    print("🎉 Training data collection complete!")

if __name__ == "__main__":
    asyncio.run(main())
