"""
🧪 Test Fixtures for DharmaMind Platform
=====================================

Comprehensive test fixtures and test data for the DharmaMind spiritual guidance platform:
- Spiritual content and scenarios
- User profiles and interaction patterns  
- Mock services and test environments
- Performance testing data
- Spiritual knowledge base samples
- Cultural and linguistic test cases

These fixtures provide realistic test scenarios while respecting
the sacred nature of spiritual content and traditions.
"""

import pytest
import asyncio
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

# Spiritual Test Content
SPIRITUAL_CONCEPTS = [
    {
        'name': 'meditation',
        'category': 'practice',
        'description': 'A practice of focused attention and awareness',
        'traditions': ['Buddhist', 'Hindu', 'Christian', 'Secular'],
        'keywords': ['mindfulness', 'concentration', 'awareness', 'sitting', 'breathing'],
        'related_emotions': ['peace', 'calm', 'clarity', 'centeredness']
    },
    {
        'name': 'compassion',
        'category': 'virtue',
        'description': 'Deep awareness of the suffering of another coupled with the wish to relieve it',
        'traditions': ['Buddhist', 'Christian', 'Islamic', 'Universal'],
        'keywords': ['kindness', 'empathy', 'love', 'care', 'mercy'],
        'related_emotions': ['love', 'empathy', 'tenderness', 'understanding']
    },
    {
        'name': 'dharma',
        'category': 'principle',
        'description': 'Righteous living, duty, and natural law',
        'traditions': ['Hindu', 'Buddhist', 'Jain'],
        'keywords': ['duty', 'righteousness', 'path', 'purpose', 'truth'],
        'related_emotions': ['purpose', 'clarity', 'determination', 'alignment']
    },
    {
        'name': 'surrender',
        'category': 'attitude',
        'description': 'Letting go of the ego\'s desire to control outcomes',
        'traditions': ['Hindu', 'Christian', 'Sufi', 'Universal'],
        'keywords': ['letting go', 'trust', 'faith', 'acceptance', 'devotion'],
        'related_emotions': ['peace', 'relief', 'trust', 'humility']
    },
    {
        'name': 'mindfulness',
        'category': 'practice',
        'description': 'Present-moment awareness without judgment',
        'traditions': ['Buddhist', 'Secular', 'Universal'],
        'keywords': ['presence', 'awareness', 'attention', 'observation', 'now'],
        'related_emotions': ['clarity', 'peace', 'awareness', 'groundedness']
    }
]

SPIRITUAL_PRACTICES = [
    {
        'name': 'daily_meditation',
        'type': 'contemplative',
        'duration': '10-60 minutes',
        'description': 'Regular sitting practice for developing awareness',
        'instructions': [
            'Find a quiet, comfortable space',
            'Sit with spine straight and relaxed',
            'Focus on breath or chosen object',
            'Notice when mind wanders, gently return attention',
            'End with gratitude and dedication'
        ],
        'benefits': ['reduced stress', 'increased clarity', 'emotional regulation'],
        'suitable_for': ['beginners', 'intermediate', 'advanced']
    },
    {
        'name': 'loving_kindness_meditation',
        'type': 'heart-opening',
        'duration': '15-30 minutes',
        'description': 'Cultivation of universal love and compassion',
        'instructions': [
            'Begin with self-directed loving wishes',
            'Extend to loved ones',
            'Include neutral people',
            'Embrace difficult relationships',
            'Radiate to all beings everywhere'
        ],
        'benefits': ['increased compassion', 'reduced anger', 'greater connection'],
        'suitable_for': ['all levels', 'those with relationship difficulties']
    },
    {
        'name': 'contemplative_reading',
        'type': 'study',
        'duration': '20-45 minutes',
        'description': 'Deep reflection on spiritual texts and teachings',
        'instructions': [
            'Choose meaningful spiritual text',
            'Read slowly with full attention',
            'Pause to reflect on meaning',
            'Consider personal application',
            'Journal insights and questions'
        ],
        'benefits': ['deepened understanding', 'intellectual insight', 'wisdom development'],
        'suitable_for': ['intellectually inclined', 'students of wisdom traditions']
    }
]

EMOTIONAL_STATES = [
    {
        'primary_emotion': 'anxiety',
        'intensity': 'high',
        'description': 'Feeling overwhelmed and worried about the future',
        'common_triggers': ['uncertainty', 'major life changes', 'financial stress'],
        'spiritual_guidance': ['present moment awareness', 'trust in higher wisdom', 'breathing practices'],
        'recommended_practices': ['mindfulness meditation', 'prayer', 'nature connection'],
        'cultural_variations': {
            'western': 'focus on individual coping strategies',
            'eastern': 'emphasis on impermanence and non-attachment',
            'indigenous': 'connection to community and ancestral wisdom'
        }
    },
    {
        'primary_emotion': 'grief',
        'intensity': 'medium',
        'description': 'Processing loss and coming to terms with change',
        'common_triggers': ['death of loved one', 'end of relationship', 'major life transition'],
        'spiritual_guidance': ['honoring the healing process', 'finding meaning in suffering', 'community support'],
        'recommended_practices': ['ritual and ceremony', 'contemplative prayer', 'memory practices'],
        'cultural_variations': {
            'western': 'individual processing and therapy',
            'eastern': 'understanding impermanence and karma',
            'indigenous': 'communal grieving and ancestral connection'
        }
    },
    {
        'primary_emotion': 'confusion',
        'intensity': 'medium',
        'description': 'Feeling lost and uncertain about direction',
        'common_triggers': ['major decisions', 'spiritual questioning', 'life transitions'],
        'spiritual_guidance': ['patient discernment', 'seeking wise counsel', 'inner listening'],
        'recommended_practices': ['discernment meditation', 'spiritual direction', 'journaling'],
        'cultural_variations': {
            'western': 'rational analysis and goal-setting',
            'eastern': 'surrender and following the Tao',
            'indigenous': 'vision questing and community guidance'
        }
    },
    {
        'primary_emotion': 'peace',
        'intensity': 'high',
        'description': 'Feeling centered, calm, and aligned',
        'common_triggers': ['successful practice', 'natural beauty', 'spiritual insight'],
        'spiritual_guidance': ['cultivating gratitude', 'sharing peace with others', 'deepening practice'],
        'recommended_practices': ['gratitude meditation', 'service to others', 'advanced contemplation'],
        'cultural_variations': {
            'western': 'maintaining work-life balance',
            'eastern': 'deepening samadhi and wisdom',
            'indigenous': 'contributing to community harmony'
        }
    }
]

USER_PROFILES = [
    {
        'id': 'beginner_seeker',
        'experience_level': 'beginner',
        'primary_interests': ['stress relief', 'basic meditation', 'life purpose'],
        'cultural_background': 'western',
        'preferred_style': 'practical and simple',
        'time_availability': 'limited',
        'learning_preference': 'step-by-step guidance',
        'spiritual_background': 'secular/agnostic',
        'challenges': ['busy lifestyle', 'skepticism', 'impatience']
    },
    {
        'id': 'intermediate_practitioner',
        'experience_level': 'intermediate',
        'primary_interests': ['deepening practice', 'emotional healing', 'wisdom development'],
        'cultural_background': 'mixed_eastern_western',
        'preferred_style': 'balanced theory and practice',
        'time_availability': 'moderate',
        'learning_preference': 'experiential learning',
        'spiritual_background': 'Buddhist/meditation experience',
        'challenges': ['spiritual dryness', 'balancing traditions', 'integration']
    },
    {
        'id': 'advanced_student',
        'experience_level': 'advanced',
        'primary_interests': ['non-dual awareness', 'teaching others', 'service'],
        'cultural_background': 'traditional_eastern',
        'preferred_style': 'subtle and profound',
        'time_availability': 'dedicated',
        'learning_preference': 'contemplative inquiry',
        'spiritual_background': 'long-term practitioner',
        'challenges': ['spiritual pride', 'teaching readiness', 'embodiment']
    },
    {
        'id': 'grief_processor',
        'experience_level': 'varies',
        'primary_interests': ['healing', 'meaning-making', 'hope'],
        'cultural_background': 'varies',
        'preferred_style': 'gentle and compassionate',
        'time_availability': 'varies',
        'learning_preference': 'supportive community',
        'spiritual_background': 'seeking comfort and meaning',
        'challenges': ['overwhelming emotions', 'questioning faith', 'isolation']
    }
]

TEST_QUERIES = [
    {
        'query': 'I feel anxious about the future and can\'t sleep',
        'expected_emotional_states': ['anxiety', 'worry', 'restlessness'],
        'expected_concepts': ['present moment awareness', 'impermanence', 'trust'],
        'expected_practices': ['breathing meditation', 'body relaxation', 'gratitude practice'],
        'difficulty': 'moderate',
        'cultural_context': 'universal'
    },
    {
        'query': 'How can I find my life purpose and meaning?',
        'expected_emotional_states': ['confusion', 'seeking', 'longing'],
        'expected_concepts': ['dharma', 'calling', 'service', 'self-discovery'],
        'expected_practices': ['contemplative reflection', 'journaling', 'spiritual direction'],
        'difficulty': 'complex',
        'cultural_context': 'western_individualistic'
    },
    {
        'query': 'I am grieving the loss of my parent and feel lost',
        'expected_emotional_states': ['grief', 'sadness', 'confusion', 'longing'],
        'expected_concepts': ['impermanence', 'love transcending death', 'healing process'],
        'expected_practices': ['memorial meditation', 'grief rituals', 'community support'],
        'difficulty': 'sensitive',
        'cultural_context': 'universal'
    },
    {
        'query': 'My meditation practice feels dry and I\'m losing motivation',
        'expected_emotional_states': ['frustration', 'doubt', 'spiritual dryness'],
        'expected_concepts': ['spiritual seasons', 'patience', 'faith', 'perseverance'],
        'expected_practices': ['practice variation', 'spiritual reading', 'teacher consultation'],
        'difficulty': 'intermediate',
        'cultural_context': 'practitioner_community'
    },
    {
        'query': 'मैं शांति और आनंद की तलाश में हूं',  # Hindi: I am seeking peace and joy
        'expected_emotional_states': ['seeking', 'longing', 'aspiration'],
        'expected_concepts': ['shanti', 'ananda', 'sadhana', 'guru'],
        'expected_practices': ['mantra meditation', 'yoga', 'satsang'],
        'difficulty': 'moderate',
        'cultural_context': 'hindu_tradition'
    }
]

# Test Data Fixtures

@pytest.fixture
def spiritual_concepts():
    """Fixture providing spiritual concepts for testing"""
    return SPIRITUAL_CONCEPTS.copy()

@pytest.fixture
def spiritual_practices():
    """Fixture providing spiritual practices for testing"""
    return SPIRITUAL_PRACTICES.copy()

@pytest.fixture
def emotional_states():
    """Fixture providing emotional states for testing"""
    return EMOTIONAL_STATES.copy()

@pytest.fixture
def user_profiles():
    """Fixture providing user profiles for testing"""
    return USER_PROFILES.copy()

@pytest.fixture
def test_queries():
    """Fixture providing test queries for testing"""
    return TEST_QUERIES.copy()

@pytest.fixture
def sample_user_id():
    """Fixture providing a sample user ID for testing"""
    return f"test_user_{random.randint(1000, 9999)}"

@pytest.fixture
def sample_query():
    """Fixture providing a sample spiritual query"""
    return random.choice(TEST_QUERIES)['query']

# Mock Service Fixtures

@dataclass
class MockEmbeddingService:
    """Mock embedding service for testing"""
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate mock embedding based on text hash"""
        # Simple mock: hash text and convert to normalized floats
        text_hash = hash(text)
        embedding = []
        for i in range(384):  # Standard sentence transformer size
            embedding.append(((text_hash + i) % 10000) / 10000.0 - 0.5)
        return embedding
    
    async def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute mock similarity"""
        # Simple cosine similarity approximation
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = sum(a * a for a in embedding1) ** 0.5
        norm2 = sum(b * b for b in embedding2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0

@dataclass
class MockLLMService:
    """Mock LLM service for testing"""
    
    async def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate mock LLM response"""
        # Simple pattern matching for realistic responses
        prompt_lower = prompt.lower()
        
        response_templates = {
            'anxiety': {
                'content': 'When anxiety arises, remember that this too shall pass. Focus on your breath and ground yourself in the present moment.',
                'confidence': 0.85,
                'emotional_tone': 'compassionate',
                'suggested_practices': ['breathing meditation', 'mindfulness']
            },
            'grief': {
                'content': 'Grief is love with nowhere to go. Honor your feelings and know that healing happens in its own time.',
                'confidence': 0.90,
                'emotional_tone': 'gentle',
                'suggested_practices': ['memorial meditation', 'community support']
            },
            'purpose': {
                'content': 'Your purpose emerges when you align your unique gifts with the world\'s needs. Listen deeply to your heart.',
                'confidence': 0.80,
                'emotional_tone': 'encouraging',
                'suggested_practices': ['contemplative reflection', 'service']
            },
            'meditation': {
                'content': 'Meditation is like training a puppy - be patient and gentle with your wandering mind.',
                'confidence': 0.85,
                'emotional_tone': 'instructive',
                'suggested_practices': ['daily sitting', 'loving-kindness']
            }
        }
        
        # Find best matching template
        best_match = 'meditation'  # default
        for key, template in response_templates.items():
            if key in prompt_lower:
                best_match = key
                break
        
        response = response_templates[best_match].copy()
        response['prompt_length'] = len(prompt)
        response['generated_at'] = datetime.now().isoformat()
        
        return response

@pytest.fixture
def mock_embedding_service():
    """Fixture providing mock embedding service"""
    return MockEmbeddingService()

@pytest.fixture
def mock_llm_service():
    """Fixture providing mock LLM service"""
    return MockLLMService()

# Performance Testing Fixtures

@pytest.fixture
def performance_test_data():
    """Fixture providing data for performance testing"""
    return {
        'small_queries': [
            'peace',
            'help me',
            'I am sad',
            'meditation',
            'love'
        ],
        'medium_queries': [
            'I am feeling anxious about my job interview tomorrow',
            'How can I develop more compassion for difficult people?',
            'What is the purpose of suffering in spiritual growth?',
            'I want to start meditating but don\'t know where to begin',
            'How do I balance spiritual practice with family responsibilities?'
        ],
        'large_queries': [
            'I have been practicing meditation for several years now, and while I initially experienced significant benefits like reduced stress and increased clarity, lately I feel like my practice has become stagnant and I\'m questioning whether I\'m making real progress on my spiritual journey toward greater wisdom and compassion, especially when I still get triggered by the same situations and people that have always challenged me.',
            'As someone who grew up in a Christian household but has been drawn to Buddhist and Hindu teachings in recent years, I\'m struggling to integrate these different spiritual traditions in a way that feels authentic and not culturally appropriative, while also honoring my family\'s religious background and maintaining those important relationships, all while trying to find my own genuine spiritual path that resonates with my deepest values and experiences.',
            'After experiencing a profound spiritual awakening during a retreat last year, I have been trying to maintain that sense of connection and clarity in my daily life, but the demands of my high-stress career, relationship challenges, and the general chaos of modern life seem to constantly pull me away from that centered state, and I\'m looking for practical wisdom on how to bridge the gap between peak spiritual experiences and the mundane reality of everyday existence.'
        ],
        'concurrent_user_scenarios': [
            {'users': 10, 'query_rate': 1, 'duration': 30},
            {'users': 25, 'query_rate': 2, 'duration': 60},
            {'users': 50, 'query_rate': 3, 'duration': 45},
            {'users': 100, 'query_rate': 1, 'duration': 120}
        ],
        'cache_test_scenarios': [
            {'cache_size': 100, 'operations': 500, 'hit_ratio_target': 0.6},
            {'cache_size': 500, 'operations': 2000, 'hit_ratio_target': 0.75},
            {'cache_size': 1000, 'operations': 5000, 'hit_ratio_target': 0.85}
        ]
    }

# Integration Testing Fixtures

@pytest.fixture
def integration_test_scenarios():
    """Fixture providing integration test scenarios"""
    return [
        {
            'scenario_name': 'complete_guidance_workflow',
            'description': 'Full spiritual guidance session from query to response',
            'steps': [
                'analyze_emotional_state',
                'search_spiritual_knowledge',
                'generate_personalized_response',
                'cache_results',
                'update_user_profile'
            ],
            'expected_duration': 0.5,  # seconds
            'success_criteria': {
                'emotional_confidence': 0.7,
                'knowledge_relevance': 0.8,
                'response_coherence': 0.8
            }
        },
        {
            'scenario_name': 'multi_turn_conversation',
            'description': 'Extended conversation with context maintenance',
            'steps': [
                'initial_query_processing',
                'follow_up_question_1',
                'follow_up_question_2',
                'clarification_request',
                'final_integration'
            ],
            'expected_duration': 2.0,  # seconds
            'success_criteria': {
                'context_maintenance': 0.8,
                'conversation_coherence': 0.75,
                'personalization_improvement': 0.7
            }
        },
        {
            'scenario_name': 'crisis_support_workflow',
            'description': 'Handling sensitive emotional crisis situations',
            'steps': [
                'crisis_detection',
                'emotional_stabilization',
                'resource_recommendation',
                'follow_up_scheduling',
                'safety_protocol_activation'
            ],
            'expected_duration': 1.0,  # seconds
            'success_criteria': {
                'crisis_detection_accuracy': 0.9,
                'response_appropriateness': 0.95,
                'safety_protocol_engagement': 1.0
            }
        }
    ]

# Cultural and Linguistic Test Fixtures

@pytest.fixture
def multilingual_test_data():
    """Fixture providing multilingual test data"""
    return {
        'hindi': {
            'queries': [
                'मैं अशांत हूं और मुझे शांति चाहिए',  # I am restless and need peace
                'ध्यान कैसे करें?',  # How to meditate?
                'जीवन का उद्देश्य क्या है?'  # What is the purpose of life?
            ],
            'concepts': ['शांति', 'ध्यान', 'धर्म', 'कर्म', 'मोक्ष'],
            'emotions': ['शांति', 'प्रेम', 'करुणा', 'क्रोध', 'भय']
        },
        'sanskrit': {
            'queries': [
                'शान्तिः किम्?',  # What is peace?
                'ध्यानस्य फलम् किम्?'  # What is the fruit of meditation?
            ],
            'concepts': ['आत्मा', 'ब्रह्म', 'धर्म', 'अहिंसा', 'सत्य'],
            'emotions': ['आनन्द', 'शान्ति', 'प्रेम', 'करुणा']
        },
        'chinese': {
            'queries': [
                '我如何找到内心的平静？',  # How do I find inner peace?
                '什么是正念？'  # What is mindfulness?
            ],
            'concepts': ['道', '禅', '慈悲', '智慧', '平静'],
            'emotions': ['平静', '喜悦', '慈悲', '智慧']
        },
        'arabic': {
            'queries': [
                'كيف أجد السلام الداخلي؟',  # How do I find inner peace?
                'ما هو التأمل؟'  # What is meditation?
            ],
            'concepts': ['سلام', 'صبر', 'حكمة', 'رحمة', 'محبة'],
            'emotions': ['سلام', 'فرح', 'حب', 'رحمة']
        }
    }

@pytest.fixture
def cultural_context_data():
    """Fixture providing cultural context test data"""
    return {
        'western_individualistic': {
            'values': ['personal growth', 'autonomy', 'self-actualization'],
            'approaches': ['therapy-informed', 'scientific', 'practical'],
            'concerns': ['work-life balance', 'achievement', 'relationships']
        },
        'eastern_collective': {
            'values': ['harmony', 'duty', 'interconnectedness'],
            'approaches': ['traditional wisdom', 'community-centered', 'ritual-based'],
            'concerns': ['family honor', 'social harmony', 'ancestral respect']
        },
        'indigenous_earth_centered': {
            'values': ['nature connection', 'ancestral wisdom', 'community healing'],
            'approaches': ['ceremony', 'story-telling', 'land-based'],
            'concerns': ['cultural preservation', 'environmental health', 'generational healing']
        },
        'secular_scientific': {
            'values': ['evidence-based', 'rational', 'measurable'],
            'approaches': ['research-informed', 'psychological', 'pragmatic'],
            'concerns': ['mental health', 'productivity', 'wellbeing metrics']
        }
    }

# Specialized Test Utilities

class SpiritualTestValidator:
    """Utility class for validating spiritual content appropriateness"""
    
    @staticmethod
    def validate_cultural_sensitivity(content: str, cultural_context: str) -> Tuple[bool, List[str]]:
        """Validate cultural sensitivity of spiritual content"""
        issues = []
        
        # Check for cultural appropriation indicators
        if cultural_context == 'western_individualistic':
            eastern_terms = ['karma', 'dharma', 'nirvana', 'samsara']
            for term in eastern_terms:
                if term in content.lower() and 'appropriation' not in content.lower():
                    # Check if used with proper context
                    if not any(context_word in content.lower() for context_word in ['tradition', 'teaching', 'understanding']):
                        issues.append(f"Potential cultural appropriation of term: {term}")
        
        # Check for religious exclusivity
        exclusive_terms = ['only way', 'true path', 'wrong belief']
        for term in exclusive_terms:
            if term in content.lower():
                issues.append(f"Potentially exclusive language: {term}")
        
        # Check for gender inclusivity
        gendered_terms = ['mankind', 'he/his'] 
        for term in gendered_terms:
            if term in content.lower() and 'inclusive' not in content.lower():
                issues.append(f"Non-inclusive language: {term}")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def assess_spiritual_depth(content: str) -> Dict[str, float]:
        """Assess the spiritual depth of content"""
        depth_indicators = {
            'wisdom_words': ['wisdom', 'insight', 'understanding', 'truth', 'awareness'],
            'practice_words': ['practice', 'meditation', 'prayer', 'contemplation', 'reflection'],
            'virtue_words': ['compassion', 'love', 'kindness', 'patience', 'forgiveness'],
            'transcendent_words': ['divine', 'sacred', 'holy', 'transcendent', 'infinite']
        }
        
        content_lower = content.lower()
        scores = {}
        
        for category, words in depth_indicators.items():
            matches = sum(1 for word in words if word in content_lower)
            scores[category] = min(matches / len(words), 1.0)
        
        overall_depth = sum(scores.values()) / len(scores)
        scores['overall_depth'] = overall_depth
        
        return scores

@pytest.fixture
def spiritual_test_validator():
    """Fixture providing spiritual test validator"""
    return SpiritualTestValidator()

# Export all fixtures for easy import
__all__ = [
    'SPIRITUAL_CONCEPTS',
    'SPIRITUAL_PRACTICES', 
    'EMOTIONAL_STATES',
    'USER_PROFILES',
    'TEST_QUERIES',
    'MockEmbeddingService',
    'MockLLMService',
    'SpiritualTestValidator',
    'spiritual_concepts',
    'spiritual_practices',
    'emotional_states',
    'user_profiles',
    'test_queries',
    'sample_user_id',
    'sample_query',
    'mock_embedding_service',
    'mock_llm_service',
    'performance_test_data',
    'integration_test_scenarios',
    'multilingual_test_data',
    'cultural_context_data',
    'spiritual_test_validator'
]