"""
Darshana Engine - Six Classical Schools of Hindu Philosophy Integration
=====================================================================

This module implements the Six Classical Darshanas (दर्शन) - the orthodox
schools of Hindu philosophy that provide systematic approaches to understanding
reality, consciousness, and liberation.

The Six Darshanas:
1. Vedanta (वेदान्त) - Metaphysics and Self-Realization
2. Yoga (योग) - Practical Meditation and Discipline
3. Samkhya (साङ्ख्य) - Dualistic Philosophy of Consciousness & Matter
4. Nyaya (न्याय) - Logic and Critical Reasoning
5. Vaisheshika (वैशेषिक) - Atomism and Categories of Reality
6. Mimamsa (मीमांसा) - Ritualism and Dharmic Action

🕉️ Through systematic philosophical inquiry, may all beings discover truth
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DarshanaType(Enum):
    """The Six Classical Darshanas"""
    VEDANTA = "vedanta"      # वेदान्त - End of Vedas, Metaphysics
    YOGA = "yoga"            # योग - Union, Practical Discipline
    SAMKHYA = "samkhya"      # साङ्ख्य - Enumeration, Dualism
    NYAYA = "nyaya"          # न्याय - Logic, Reasoning
    VAISHESHIKA = "vaisheshika"  # वैशेषिक - Atomism, Categories
    MIMAMSA = "mimamsa"      # मीमांसा - Investigation, Ritualism


class QueryType(Enum):
    """Types of philosophical queries"""
    METAPHYSICAL = "metaphysical"      # What is reality?
    PRACTICAL = "practical"            # How should I practice?
    LOGICAL = "logical"                # How do we know this?
    DUALISTIC = "dualistic"            # Spirit vs matter questions
    CATEGORICAL = "categorical"        # Classification questions
    RITUALISTIC = "ritualistic"       # Dharmic action questions


@dataclass
class DarshanaContext:
    """Context for philosophical processing"""
    primary_darshana: DarshanaType
    secondary_darshanas: List[DarshanaType] = field(default_factory=list)
    query_type: QueryType = QueryType.METAPHYSICAL
    philosophical_depth: float = 0.5  # 0-1 scale
    sanskrit_preference: bool = False
    user_background: str = "beginner"  # beginner, intermediate, advanced


@dataclass
class PhilosophicalResponse:
    """Response from darshana processing"""
    primary_perspective: str
    darshana_used: DarshanaType
    secondary_perspectives: Dict[DarshanaType, str] = field(
        default_factory=dict)
    sanskrit_terms: List[Dict[str, str]] = field(default_factory=list)
    scriptural_references: List[str] = field(default_factory=list)
    practical_guidance: Optional[str] = None
    philosophical_depth_score: float = 0.0
    integration_notes: Optional[str] = None

class DarshanaClassifier:
    """Classifies user queries into appropriate darshana categories"""
    
    def __init__(self):
        self.classification_patterns = {
            DarshanaType.VEDANTA: {
                'keywords': [
                    'atman', 'brahman', 'self', 'reality', 'consciousness', 'maya',
                    'who am i', 'what is real', 'ultimate truth', 'self-realization',
                    'moksha', 'liberation', 'identity', 'existence', 'being'
                ],
                'concepts': [
                    'metaphysics', 'ontology', 'ultimate reality', 'non-duality',
                    'self-inquiry', 'witness consciousness', 'sat-chit-ananda'
                ]
            },
            DarshanaType.YOGA: {
                'keywords': [
                    'meditation', 'practice', 'how to', 'asana', 'pranayama',
                    'dharana', 'dhyana', 'samadhi', 'discipline', 'control',
                    'mind control', 'spiritual practice', 'path', 'method'
                ],
                'concepts': [
                    'practical spirituality', 'mind training', 'eight limbs',
                    'concentration', 'meditation techniques', 'yoga sutras'
                ]
            },
            DarshanaType.SAMKHYA: {
                'keywords': [
                    'purusha', 'prakriti', 'consciousness', 'matter', 'evolution',
                    'creation', 'duality', 'spirit vs matter', 'gunas', 'tattvas',
                    'sattva', 'rajas', 'tamas', 'manifestation'
                ],
                'concepts': [
                    'cosmic evolution', 'consciousness-matter duality', 'twenty-five principles',
                    'three gunas', 'primordial nature'
                ]
            },
            DarshanaType.NYAYA: {
                'keywords': [
                    'logic', 'reasoning', 'proof', 'evidence', 'argument',
                    'how do we know', 'valid knowledge', 'pramana', 'inference',
                    'perception', 'testimony', 'comparison', 'debate'
                ],
                'concepts': [
                    'epistemology', 'logical reasoning', 'means of knowledge',
                    'syllogism', 'fallacies', 'critical thinking'
                ]
            },
            DarshanaType.VAISHESHIKA: {
                'keywords': [
                    'atoms', 'elements', 'categories', 'classification', 'physical world',
                    'substance', 'quality', 'action', 'dravya', 'guna', 'karma',
                    'eternal atoms', 'material cause'
                ],
                'concepts': [
                    'atomic theory', 'categories of reality', 'physics',
                    'material world', 'six categories'
                ]
            },
            DarshanaType.MIMAMSA: {
                'keywords': [
                    'dharma', 'duty', 'ritual', 'action', 'karma', 'sacrifice',
                    'vedic rituals', 'righteous action', 'moral duty', 'yajna',
                    'what should i do', 'right action'
                ],
                'concepts': [
                    'dharmic action', 'ritual philosophy', 'vedic interpretation',
                    'moral philosophy', 'duty-based ethics'
                ]
            }
        }
        
        self.query_type_patterns = {
            QueryType.METAPHYSICAL: [
                'what is', 'nature of', 'reality', 'existence', 'being',
                'ultimate', 'absolute', 'essence', 'truth'
            ],
            QueryType.PRACTICAL: [
                'how to', 'how can i', 'practice', 'method', 'technique',
                'steps', 'way', 'path', 'approach'
            ],
            QueryType.LOGICAL: [
                'why', 'how do we know', 'proof', 'evidence', 'reasoning',
                'logic', 'argument', 'justify', 'validate'
            ],
            QueryType.DUALISTIC: [
                'difference between', 'vs', 'spirit and matter', 'consciousness and',
                'body and soul', 'mind and', 'material and spiritual'
            ],
            QueryType.CATEGORICAL: [
                'types of', 'kinds of', 'categories', 'classification',
                'what are the', 'how many', 'enumerate'
            ],
            QueryType.RITUALISTIC: [
                'should i', 'duty', 'obligation', 'right action', 'dharma',
                'proper way', 'correct method', 'traditional approach'
            ]
        }
    
    def classify_query(self, query: str, context: Optional[str] = None) -> DarshanaContext:
        """Classify query into appropriate darshana and type"""
        query_lower = query.lower()
        combined_text = f"{query_lower} {context.lower() if context else ''}"
        
        # Score each darshana
        darshana_scores = {}
        for darshana, patterns in self.classification_patterns.items():
            score = 0.0
            
            # Keyword matching
            for keyword in patterns['keywords']:
                if keyword in combined_text:
                    score += 1.0
            
            # Concept matching (partial matches)
            for concept in patterns['concepts']:
                concept_words = concept.split()
                matches = sum(1 for word in concept_words if word in combined_text)
                if matches > 0:
                    score += matches / len(concept_words) * 0.5
            
            darshana_scores[darshana] = score
        
        # Determine primary darshana
        primary_darshana = max(darshana_scores, key=darshana_scores.get)
        
        # Determine secondary darshanas (score > 0.5)
        secondary_darshanas = [
            darshana for darshana, score in darshana_scores.items()
            if score > 0.5 and darshana != primary_darshana
        ]
        
        # Classify query type
        query_type_scores = {}
        for qtype, patterns in self.query_type_patterns.items():
            score = sum(1 for pattern in patterns if pattern in combined_text)
            query_type_scores[qtype] = score
        
        query_type = max(query_type_scores, key=query_type_scores.get)
        
        # Determine philosophical depth
        depth_indicators = [
            'ultimate', 'absolute', 'transcendent', 'essence', 'reality',
            'consciousness', 'being', 'existence', 'truth', 'liberation'
        ]
        depth_score = sum(1 for indicator in depth_indicators if indicator in combined_text)
        philosophical_depth = min(1.0, depth_score / 5.0)
        
        # Check for Sanskrit preference
        sanskrit_terms = ['atman', 'brahman', 'dharma', 'karma', 'moksha', 'yoga', 'samadhi']
        sanskrit_preference = any(term in query_lower for term in sanskrit_terms)
        
        return DarshanaContext(
            primary_darshana=primary_darshana,
            secondary_darshanas=secondary_darshanas[:2],  # Limit to 2 secondary
            query_type=query_type,
            philosophical_depth=philosophical_depth,
            sanskrit_preference=sanskrit_preference,
            user_background="intermediate" if philosophical_depth > 0.6 else "beginner"
        )

class VedantaProcessor:
    """Vedanta (वेदान्त) - Metaphysics and Self-Realization"""
    
    def __init__(self):
        self.core_concepts = {
            'atman': 'The individual Self, pure consciousness',
            'brahman': 'The universal Self, absolute reality',
            'maya': 'The cosmic illusion that veils true reality',
            'moksha': 'Liberation from the cycle of birth and death',
            'satchitananda': 'Existence-Consciousness-Bliss, nature of Brahman'
        }
        
        self.key_texts = [
            'Brahma Sutras', 'Upanishads', 'Bhagavad Gita',
            'Advaita Vedanta texts', 'Shankaracharya commentaries'
        ]
    
    async def process_query(self, query: str, context: DarshanaContext) -> PhilosophicalResponse:
        """Process query through Vedantic lens"""
        response = PhilosophicalResponse(
            primary_perspective="",
            darshana_used=DarshanaType.VEDANTA
        )
        
        # Determine specific Vedantic approach
        if any(term in query.lower() for term in ['who am i', 'self', 'identity']):
            response.primary_perspective = self._self_inquiry_response(query, context)
            response.sanskrit_terms.append({'atman': 'Individual Self/Soul'})
            response.sanskrit_terms.append({'brahman': 'Universal Self/Absolute'})
            
        elif any(term in query.lower() for term in ['reality', 'truth', 'existence']):
            response.primary_perspective = self._reality_inquiry_response(query, context)
            response.sanskrit_terms.append({'sat': 'Existence/Being'})
            response.sanskrit_terms.append({'maya': 'Cosmic illusion'})
            
        elif any(term in query.lower() for term in ['liberation', 'moksha', 'freedom']):
            response.primary_perspective = self._liberation_response(query, context)
            response.sanskrit_terms.append({'moksha': 'Liberation'})
            response.sanskrit_terms.append({'jivanmukta': 'Liberated while living'})
            
        else:
            response.primary_perspective = self._general_vedantic_response(query, context)
        
        # Add scriptural references
        response.scriptural_references = [
            "Chandogya Upanishad 6.8.7 - Tat tvam asi (That thou art)",
            "Mandukya Upanishad - Analysis of consciousness states",
            "Bhagavad Gita 2.20 - The eternal nature of the Self"
        ]
        
        # Practical guidance
        response.practical_guidance = "Engage in self-inquiry (atma-vichara): Ask 'Who am I?' and trace the 'I' thought to its source in pure consciousness."
        
        response.philosophical_depth_score = 0.9
        return response
    
    def _self_inquiry_response(self, query: str, context: DarshanaContext) -> str:
        if context.user_background == "beginner":
            return """From the Vedantic perspective, the question "Who am I?" is the most fundamental inquiry. You are not the body, mind, or thoughts - these are temporary and changing. Your true nature is the Atman (Self), which is pure consciousness, eternal and unchanging. This Self is identical with Brahman, the absolute reality that underlies all existence."""
        else:
            return """Vedanta teaches that the Self (Atman) is sat-chit-ananda - existence, consciousness, and bliss. Through self-inquiry (atma-vichara), one realizes that all identification with body-mind is superimposed on this pure consciousness. The ultimate realization is 'Aham Brahmasmi' - I am Brahman, recognizing the non-dual nature of reality."""
    
    def _reality_inquiry_response(self, query: str, context: DarshanaContext) -> str:
        return """Vedanta reveals that ultimate reality (Brahman) is one without a second (Advaita). What we perceive as the multiplicitous world is Maya - not unreal, but a projection upon the one unchanging reality. Like waves on the ocean or gold ornaments, the world is a real appearance of Brahman, but Brahman itself transcends all forms and names."""
    
    def _liberation_response(self, query: str, context: DarshanaContext) -> str:
        return """Liberation (Moksha) in Vedanta is not something to be attained but rather the recognition of what you already are. It is the removal of ignorance (avidya) that makes you feel separate from Brahman. Through knowledge (jnana), you realize your true nature as the Self, free from all limitations and suffering."""
    
    def _general_vedantic_response(self, query: str, context: DarshanaContext) -> str:
        return """Vedanta, the culmination of Vedic wisdom, teaches that ultimate reality is Brahman - pure consciousness without attributes. The goal is to realize the identity between the individual Self (Atman) and this universal consciousness, transcending all duality and limitation."""

class YogaProcessor:
    """Yoga (योग) - Practical Meditation and Discipline"""
    
    def __init__(self):
        self.eight_limbs = [
            'Yama (Ethical restraints)', 'Niyama (Observances)', 
            'Asana (Postures)', 'Pranayama (Breath control)',
            'Pratyahara (Withdrawal of senses)', 'Dharana (Concentration)',
            'Dhyana (Meditation)', 'Samadhi (Absorption)'
        ]
    
    async def process_query(self, query: str, context: DarshanaContext) -> PhilosophicalResponse:
        """Process query through Yoga philosophy lens"""
        response = PhilosophicalResponse(
            primary_perspective="",
            darshana_used=DarshanaType.YOGA
        )
        
        if any(term in query.lower() for term in ['meditation', 'meditate', 'dhyana']):
            response.primary_perspective = self._meditation_guidance(query, context)
            response.practical_guidance = "Begin with 5-10 minutes daily: Sit comfortably, focus on breath, gently return attention when mind wanders."
            
        elif any(term in query.lower() for term in ['practice', 'how to', 'method']):
            response.primary_perspective = self._practice_guidance(query, context)
            response.practical_guidance = "Follow the eight-limbed path gradually: Start with ethical conduct (yamas/niyamas), then asana and pranayama."
            
        elif any(term in query.lower() for term in ['mind', 'thoughts', 'control']):
            response.primary_perspective = self._mind_control_guidance(query, context)
            response.practical_guidance = "Practice witness consciousness: Observe thoughts without judgment, gradually developing detachment."
            
        else:
            response.primary_perspective = self._general_yoga_response(query, context)
        
        # Add relevant Sanskrit terms
        response.sanskrit_terms.extend([
            {'yoga': 'Union of individual consciousness with universal consciousness'},
            {'samadhi': 'Absorption, highest state of meditation'},
            {'chitta': 'Mind-stuff, consciousness'},
            {'vritti': 'Mental modifications, thought-waves'}
        ])
        
        response.scriptural_references = [
            "Yoga Sutras 1.2 - Yoga is the cessation of modifications of the mind",
            "Yoga Sutras 2.29 - The eight limbs of yoga",
            "Bhagavad Gita 6.19 - Steady mind in meditation"
        ]
        
        response.philosophical_depth_score = 0.7
        return response
    
    def _meditation_guidance(self, query: str, context: DarshanaContext) -> str:
        return """Yoga defines meditation (dhyana) as sustained concentration leading to samadhi (absorption). The practice progresses through dharana (focused attention), dhyana (sustained awareness), and samadhi (unity consciousness). Regular practice purifies the mind and reveals your true Self."""
    
    def _practice_guidance(self, query: str, context: DarshanaContext) -> str:
        return """The Yoga path follows Patanjali's eight limbs (Ashtanga): Begin with ethical conduct (yamas and niyamas), establish physical stability (asana), develop breath awareness (pranayama), then progress to concentration and meditation practices."""
    
    def _mind_control_guidance(self, query: str, context: DarshanaContext) -> str:
        return """Yoga teaches that the mind (chitta) consists of constant modifications (vrittis). Through practice (abhyasa) and detachment (vairagya), these mental fluctuations gradually cease, revealing the true Self as pure consciousness."""
    
    def _general_yoga_response(self, query: str, context: DarshanaContext) -> str:
        return """Yoga, literally meaning 'union,' is the systematic method for achieving the unity of individual consciousness with universal consciousness. It provides practical techniques for purifying body, breath, and mind, leading to Self-realization."""

class NyayaProcessor:
    """Nyaya (न्याय) - Logic and Critical Reasoning"""
    
    def __init__(self):
        self.pramanas = [
            'Pratyaksha (Direct perception)',
            'Anumana (Inference)', 
            'Upamana (Comparison)',
            'Shabda (Verbal testimony)'
        ]
    
    async def process_query(self, query: str, context: DarshanaContext) -> PhilosophicalResponse:
        """Process query through logical analysis"""
        response = PhilosophicalResponse(
            primary_perspective="",
            darshana_used=DarshanaType.NYAYA
        )
        
        if any(term in query.lower() for term in ['how do we know', 'evidence', 'proof']):
            response.primary_perspective = self._epistemological_response(query, context)
            
        elif any(term in query.lower() for term in ['logic', 'reasoning', 'argument']):
            response.primary_perspective = self._logical_analysis(query, context)
            
        else:
            response.primary_perspective = self._general_nyaya_response(query, context)
        
        response.sanskrit_terms.extend([
            {'pramana': 'Means of valid knowledge'},
            {'pratyaksha': 'Direct perception'},
            {'anumana': 'Logical inference'},
            {'tarka': 'Logical reasoning'}
        ])
        
        response.practical_guidance = "Apply the four means of knowledge: Observe directly, infer logically, compare with known examples, and accept valid testimony."
        response.philosophical_depth_score = 0.8
        
        return response
    
    def _epistemological_response(self, query: str, context: DarshanaContext) -> str:
        return """Nyaya provides four valid means of knowledge (pramanas): Direct perception for immediate experience, inference for logical deduction, comparison for understanding similarities, and verbal testimony from reliable sources. Valid knowledge requires careful verification through these methods."""
    
    def _logical_analysis(self, query: str, context: DarshanaContext) -> str:
        return """Nyaya logic follows a five-step syllogism: Statement, reason, universal rule, application, and conclusion. This systematic approach ensures valid reasoning and helps identify logical fallacies in arguments."""
    
    def _general_nyaya_response(self, query: str, context: DarshanaContext) -> str:
        return """Nyaya philosophy emphasizes critical thinking and logical analysis as paths to truth. It provides systematic methods for valid reasoning and knowledge acquisition, essential for any serious philosophical inquiry."""

# Additional processors for Samkhya, Vaisheshika, and Mimamsa would follow similar patterns...

class SamkhyaProcessor:
    """Samkhya (साङ्ख्य) - Dualistic Philosophy of Consciousness & Matter"""
    
    def __init__(self):
        self.core_principles = {
            'purusha': 'Pure consciousness, the eternal witness',
            'prakriti': 'Primordial matter, source of manifestation',
            'sattva': 'Quality of purity, balance, harmony',
            'rajas': 'Quality of activity, passion, movement',
            'tamas': 'Quality of inertia, darkness, ignorance'
        }
        
        self.twenty_five_tattvas = [
            'Purusha', 'Prakriti', 'Mahat', 'Ahamkara', 'Manas',
            'Five sense organs', 'Five action organs', 'Five subtle elements',
            'Five gross elements'
        ]
    
    async def process_query(self, query: str, context: DarshanaContext) -> PhilosophicalResponse:
        """Process query through Samkhya dualistic lens"""
        response = PhilosophicalResponse(
            primary_perspective="",
            darshana_used=DarshanaType.SAMKHYA
        )
        
        if any(term in query.lower() for term in ['consciousness', 'matter', 'duality', 'purusha', 'prakriti']):
            response.primary_perspective = self._consciousness_matter_analysis(query, context)
            response.sanskrit_terms.extend([
                {'purusha': 'Pure consciousness, the witness'},
                {'prakriti': 'Primordial matter, creative principle'}
            ])
            
        elif any(term in query.lower() for term in ['creation', 'evolution', 'manifestation']):
            response.primary_perspective = self._cosmic_evolution_response(query, context)
            response.sanskrit_terms.extend([
                {'sattva': 'Quality of purity and balance'},
                {'rajas': 'Quality of activity and passion'},
                {'tamas': 'Quality of inertia and ignorance'}
            ])
            
        elif any(term in query.lower() for term in ['gunas', 'qualities', 'sattva', 'rajas', 'tamas']):
            response.primary_perspective = self._three_gunas_response(query, context)
            response.sanskrit_terms.extend([
                {'guna': 'Fundamental quality or attribute'},
                {'trigunatmika': 'Having the nature of three gunas'}
            ])
            
        else:
            response.primary_perspective = self._general_samkhya_response(query, context)
        
        response.scriptural_references = [
            "Samkhya Karika - Classical text on Samkhya philosophy",
            "Bhagavad Gita 7.4-5 - Description of lower and higher nature",
            "Samkhya Sutras - Systematic exposition of dualistic philosophy"
        ]
        
        response.practical_guidance = "Cultivate discrimination (viveka) between the eternal witness consciousness (Purusha) and the ever-changing phenomena of matter (Prakriti)."
        response.philosophical_depth_score = 0.85
        
        return response
    
    def _consciousness_matter_analysis(self, query: str, context: DarshanaContext) -> str:
        if context.user_background == "beginner":
            return """Samkhya teaches that reality consists of two eternal principles: Purusha (consciousness) and Prakriti (matter). Purusha is pure awareness - inactive, unchanging, and merely witnessing. Prakriti is the active creative principle that manifests as the entire material universe through its three qualities (gunas). Liberation comes from realizing you are the witnessing consciousness, not the material body-mind complex."""
        else:
            return """In Samkhya's dualistic framework, Purusha represents pure consciousness (cit) - inactive, attributeless, and eternal. Prakriti is the unconscious creative matrix possessing three gunas (sattva, rajas, tamas) that evolve into the twenty-five tattvas of manifestation. The proximity of Purusha catalyzes Prakriti's evolution, yet Purusha remains forever untouched. Kaivalya (isolation) is achieved when the evolutes of Prakriti cease their dance before the motionless witness."""
    
    def _cosmic_evolution_response(self, query: str, context: DarshanaContext) -> str:
        return """Samkhya describes cosmic evolution as Prakriti's systematic unfoldment in the presence of Purusha. From the equilibrium of three gunas arises Mahat (cosmic intelligence), then Ahamkara (ego-principle), followed by mind, senses, and elements. This evolution is both cosmic and individual - the same principles that create the universe operate in personal experience. Understanding this process reveals how consciousness becomes apparently bound and how it can achieve liberation."""
    
    def _three_gunas_response(self, query: str, context: DarshanaContext) -> str:
        return """The three gunas are fundamental qualities pervading all of Prakriti: Sattva brings clarity, harmony, and knowledge; Rajas creates activity, passion, and attachment; Tamas produces inertia, ignorance, and delusion. Every phenomenon represents a unique combination of these three. Spiritual evolution involves increasing sattva while transcending all three gunas to realize the guna-less nature of pure consciousness."""
    
    def _general_samkhya_response(self, query: str, context: DarshanaContext) -> str:
        return """Samkhya provides a systematic analysis of existence through twenty-five principles (tattvas), explaining how consciousness and matter interact to create the experienced world. This knowledge (jnana) of the fundamental duality between Purusha and Prakriti is itself the means of liberation, freeing consciousness from identification with material modifications."""


class VaisheshikaProcessor:
    """Vaisheshika (वैशेषिक) - Atomism and Categories of Reality"""
    
    def __init__(self):
        self.six_categories = {
            'dravya': 'Substance - the material substrate',
            'guna': 'Quality - attributes inherent in substances',
            'karma': 'Action - movement and change',
            'samanya': 'Generality - universal characteristics',
            'vishesha': 'Particularity - individual differences',
            'samavaya': 'Inherence - inseparable relation'
        }
        
        self.nine_substances = [
            'Earth (Prithvi)', 'Water (Ap)', 'Fire (Tejas)', 'Air (Vayu)',
            'Space (Akasha)', 'Time (Kala)', 'Direction (Dik)', 'Soul (Atman)', 'Mind (Manas)'
        ]
    
    async def process_query(self, query: str, context: DarshanaContext) -> PhilosophicalResponse:
        """Process query through Vaisheshika categorical analysis"""
        response = PhilosophicalResponse(
            primary_perspective="",
            darshana_used=DarshanaType.VAISHESHIKA
        )
        
        if any(term in query.lower() for term in ['atoms', 'elements', 'physical', 'material']):
            response.primary_perspective = self._atomic_theory_response(query, context)
            response.sanskrit_terms.extend([
                {'anu': 'Atom, indivisible particle'},
                {'paramanu': 'Ultimate atom'}
            ])
            
        elif any(term in query.lower() for term in ['categories', 'classification', 'types']):
            response.primary_perspective = self._six_categories_response(query, context)
            response.sanskrit_terms.extend([
                {'padarthas': 'Categories of reality'},
                {'dravya': 'Substance'},
                {'guna': 'Quality'}
            ])
            
        elif any(term in query.lower() for term in ['substance', 'quality', 'action']):
            response.primary_perspective = self._substance_analysis(query, context)
            response.sanskrit_terms.append({'samavaya': 'Inherent relation'})
            
        else:
            response.primary_perspective = self._general_vaisheshika_response(query, context)
        
        response.scriptural_references = [
            "Vaisheshika Sutras - Kanada's foundational text",
            "Padarthadharmasamgraha - Systematic exposition of categories",
            "Vaisheshika philosophy in Nyaya-Vaisheshika synthesis"
        ]
        
        response.practical_guidance = "Develop precise discrimination (viveka) in observing the categories of experience - substances, qualities, actions, and relations."
        response.philosophical_depth_score = 0.75
        
        return response
    
    def _atomic_theory_response(self, query: str, context: DarshanaContext) -> str:
        return """Vaisheshika presents an atomic theory where the physical universe consists of eternal, indivisible atoms (paramanus) of four elements: earth, water, fire, and air. These atoms combine in specific proportions to form larger molecules and objects. The atomic combinations and separations explain creation and dissolution of the material world. This ancient atomic theory remarkably parallels modern scientific understanding while maintaining a spiritual framework that includes the soul as a distinct, non-atomic substance."""
    
    def _six_categories_response(self, query: str, context: DarshanaContext) -> str:
        return """Vaisheshika organizes all reality into six fundamental categories (padarthas): Substance (dravya) - the material substrate; Quality (guna) - attributes like color and taste; Action (karma) - movement and change; Generality (samanya) - universal characteristics; Particularity (vishesha) - individual uniqueness; and Inherence (samavaya) - the inseparable relation between substances and their qualities. This systematic categorization provides a comprehensive framework for understanding the structure of reality."""
    
    def _substance_analysis(self, query: str, context: DarshanaContext) -> str:
        return """In Vaisheshika, substances (dravya) are the fundamental entities that possess qualities and perform actions. The nine eternal substances include the four physical elements (earth, water, fire, air), space, time, direction, individual souls, and mind. Each substance has specific qualities and capabilities. The soul (atman) is a unique substance - consciousness, distinct from both physical atoms and the mind, capable of knowledge, desire, and action."""
    
    def _general_vaisheshika_response(self, query: str, context: DarshanaContext) -> str:
        return """Vaisheshika philosophy provides a systematic analysis of the physical and metaphysical components of reality through its atomic theory and categorical framework. It offers both a scientific understanding of material composition and a spiritual recognition of the soul's distinct nature, bridging empirical observation with metaphysical insight."""


class MimamsaProcessor:
    """Mimamsa (मीमांसा) - Ritualism and Dharmic Action"""
    
    def __init__(self):
        self.core_principles = {
            'dharma': 'Righteous action as cosmic law',
            'karma': 'Action and its inevitable consequences',
            'yajna': 'Sacred sacrifice and ritual offering',
            'mantras': 'Sacred sounds with inherent power',
            'apurva': 'Unseen potency of ritual action'
        }
        
        self.sources_of_dharma = [
            'Vedic injunctions (Vidhi)', 'Vedic prohibitions (Nishedha)',
            'Traditional practice (Achara)', 'Conscience (Atmatusti)'
        ]
    
    async def process_query(self, query: str, context: DarshanaContext) -> PhilosophicalResponse:
        """Process query through Mimamsa dharmic lens"""
        response = PhilosophicalResponse(
            primary_perspective="",
            darshana_used=DarshanaType.MIMAMSA
        )
        
        if any(term in query.lower() for term in ['dharma', 'duty', 'righteous', 'moral']):
            response.primary_perspective = self._dharma_analysis(query, context)
            response.sanskrit_terms.extend([
                {'dharma': 'Righteous action, cosmic law'},
                {'svadharma': 'Individual duty based on nature'}
            ])
            
        elif any(term in query.lower() for term in ['ritual', 'sacrifice', 'yajna', 'ceremony']):
            response.primary_perspective = self._ritual_philosophy_response(query, context)
            response.sanskrit_terms.extend([
                {'yajna': 'Sacred sacrifice or offering'},
                {'yajamana': 'Performer of sacrifice'}
            ])
            
        elif any(term in query.lower() for term in ['action', 'karma', 'should i', 'what to do']):
            response.primary_perspective = self._action_guidance(query, context)
            response.sanskrit_terms.extend([
                {'apurva': 'Unseen result of righteous action'},
                {'nishkama karma': 'Desireless action'}
            ])
            
        else:
            response.primary_perspective = self._general_mimamsa_response(query, context)
        
        response.scriptural_references = [
            "Purva Mimamsa Sutras - Jaimini's systematic exposition",
            "Bhagavad Gita 3.9 - Action as yajna (sacrifice)",
            "Vedic mantras and ritual prescriptions"
        ]
        
        response.practical_guidance = "Perform your duties (svadharma) without attachment to results, treating all actions as offerings (yajna) to the cosmic order."
        response.philosophical_depth_score = 0.8
        
        return response
    
    def _dharma_analysis(self, query: str, context: DarshanaContext) -> str:
        if context.user_background == "beginner":
            return """Mimamsa teaches that dharma is not mere human convention but cosmic law revealed through the Vedas. Your dharma (righteous duty) depends on your nature, stage of life, and circumstances. Actions performed in accordance with dharma create positive consequences (punya), while adharmic actions create negative consequences (papa). The key is to act according to scriptural guidance and traditional wisdom rather than personal preference."""
        else:
            return """In Mimamsa, dharma represents the eternal, self-validating moral order embedded in the fabric of existence. It operates through the principle of apurva - the unseen potency that connects righteous action with its inevitable result. Dharma is known primarily through Vedic injunctions (vidhis) and prohibitions (nishedhas), supplemented by traditional practice (achara) and refined conscience (atmatusti). This system ensures cosmic justice through the impersonal law of karma."""
    
    def _ritual_philosophy_response(self, query: str, context: DarshanaContext) -> str:
        return """Mimamsa views rituals (yajnas) as more than symbolic acts - they are precise technologies for harmonizing individual consciousness with cosmic order. Sacred mantras possess inherent potency (shakti), and properly performed rituals generate apurva, an unseen force that produces both immediate and future benefits. Even daily actions can be transformed into yajna through proper attitude and dedication, making ordinary life a continuous sacred offering."""
    
    def _action_guidance(self, query: str, context: DarshanaContext) -> str:
        return """Mimamsa provides clear guidance for ethical action: Follow your svadharma (individual duty) based on your nature and life circumstances. Act according to Vedic principles rather than personal desires. Perform actions as yajna (sacred offering) without attachment to results. This approach ensures that your actions contribute to cosmic harmony while gradually purifying consciousness and creating positive karma."""
    
    def _general_mimamsa_response(self, query: str, context: DarshanaContext) -> str:
        return """Mimamsa establishes the philosophical foundation for dharmic living through systematic interpretation of Vedic injunctions. It demonstrates how proper action, performed with correct understanding and attitude, becomes a path of spiritual evolution that honors both individual development and cosmic harmony."""

class DarshanaEngine:
    """Main engine coordinating all six darshana processors"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.classifier = DarshanaClassifier()
        
        # Initialize processors
        self.processors = {
            DarshanaType.VEDANTA: VedantaProcessor(),
            DarshanaType.YOGA: YogaProcessor(),
            DarshanaType.NYAYA: NyayaProcessor(),
            DarshanaType.SAMKHYA: SamkhyaProcessor(),
            DarshanaType.VAISHESHIKA: VaisheshikaProcessor(),
            DarshanaType.MIMAMSA: MimamsaProcessor(),
        }
        
        self.processing_metrics = {
            "queries_processed": 0,
            "darshana_usage": {darshana.value: 0 for darshana in DarshanaType},
            "average_philosophical_depth": 0.0
        }
        
        self.logger.info("Darshana Engine initialized with classical Hindu philosophy")
    
    async def initialize(self) -> bool:
        """Initialize the darshana engine"""
        try:
            self.logger.info("🕉️ Initializing Darshana Engine...")
            
            # Initialize all processors
            for darshana_type, processor in self.processors.items():
                if hasattr(processor, 'initialize'):
                    await processor.initialize()
            
            self.logger.info("✅ Darshana Engine initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Darshana Engine: {e}")
            return False
    
    async def process_philosophical_query(
        self, 
        query: str, 
        context: Optional[str] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> PhilosophicalResponse:
        """Main method to process queries through appropriate darshana"""
        try:
            # Classify query into appropriate darshana
            darshana_context = self.classifier.classify_query(query, context)
            
            # Apply user preferences if provided
            if user_preferences:
                if 'preferred_darshana' in user_preferences:
                    darshana_context.primary_darshana = DarshanaType(user_preferences['preferred_darshana'])
                if 'background' in user_preferences:
                    darshana_context.user_background = user_preferences['background']
            
            # Get primary processor
            primary_processor = self.processors.get(darshana_context.primary_darshana)
            if not primary_processor:
                # Fallback to Vedanta if processor not available
                primary_processor = self.processors[DarshanaType.VEDANTA]
                darshana_context.primary_darshana = DarshanaType.VEDANTA
            
            # Process through primary darshana
            response = await primary_processor.process_query(query, darshana_context)
            
            # Add secondary perspectives if available
            for secondary_darshana in darshana_context.secondary_darshanas:
                if secondary_darshana in self.processors:
                    secondary_processor = self.processors[secondary_darshana]
                    secondary_response = await secondary_processor.process_query(query, darshana_context)
                    response.secondary_perspectives[secondary_darshana] = secondary_response.primary_perspective
            
            # Add integration notes if multiple darshanas involved
            if response.secondary_perspectives:
                response.integration_notes = self._generate_integration_notes(
                    darshana_context.primary_darshana, 
                    list(response.secondary_perspectives.keys())
                )
            
            # Update metrics
            self.processing_metrics["queries_processed"] += 1
            self.processing_metrics["darshana_usage"][darshana_context.primary_darshana.value] += 1
            
            self.logger.info(f"Processed query through {darshana_context.primary_darshana.value}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing philosophical query: {e}")
            raise
    
    def _generate_integration_notes(
        self, 
        primary: DarshanaType, 
        secondaries: List[DarshanaType]
    ) -> str:
        """Generate notes on how different darshanas complement each other"""
        integration_notes = f"While {primary.value} provides the primary perspective, "
        
        if DarshanaType.YOGA in secondaries:
            integration_notes += "Yoga offers practical methods for realization, "
        if DarshanaType.NYAYA in secondaries:
            integration_notes += "Nyaya provides logical validation, "
        if DarshanaType.SAMKHYA in secondaries:
            integration_notes += "Samkhya explains the cosmic process, "
        
        integration_notes += "creating a comprehensive understanding that honors the systematic approach of classical Hindu philosophy."
        
        return integration_notes
    
    def get_darshana_info(self) -> Dict[str, Any]:
        """Get information about available darshanas"""
        return {
            "available_darshanas": [
                {
                    "name": darshana.value,
                    "sanskrit": self._get_sanskrit_name(darshana),
                    "focus": self._get_focus_area(darshana),
                    "available": darshana in self.processors
                }
                for darshana in DarshanaType
            ],
            "processing_metrics": self.processing_metrics
        }
    
    def _get_sanskrit_name(self, darshana: DarshanaType) -> str:
        sanskrit_names = {
            DarshanaType.VEDANTA: "वेदान्त",
            DarshanaType.YOGA: "योग", 
            DarshanaType.SAMKHYA: "साङ्ख्य",
            DarshanaType.NYAYA: "न्याय",
            DarshanaType.VAISHESHIKA: "वैशेषिक",
            DarshanaType.MIMAMSA: "मीमांसा"
        }
        return sanskrit_names.get(darshana, "")
    
    def _get_focus_area(self, darshana: DarshanaType) -> str:
        focus_areas = {
            DarshanaType.VEDANTA: "Metaphysics & Self-Realization",
            DarshanaType.YOGA: "Practical Discipline & Meditation",
            DarshanaType.SAMKHYA: "Dualistic Cosmology", 
            DarshanaType.NYAYA: "Logic & Critical Reasoning",
            DarshanaType.VAISHESHIKA: "Atomism & Categories",
            DarshanaType.MIMAMSA: "Ritualism & Dharmic Action"
        }
        return focus_areas.get(darshana, "")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current status of darshana engine"""
        return {
            "status": "active",
            "available_darshanas": len(self.processors),
            "total_darshanas": len(DarshanaType),
            "processing_metrics": self.processing_metrics,
            "health_score": 0.9
        }

# Global darshana engine instance
_darshana_engine = None

def get_darshana_engine() -> DarshanaEngine:
    """Get global darshana engine instance"""
    global _darshana_engine
    if _darshana_engine is None:
        _darshana_engine = DarshanaEngine()
    return _darshana_engine

# Export main classes
__all__ = [
    "DarshanaEngine", "DarshanaType", "DarshanaContext", 
    "PhilosophicalResponse", "get_darshana_engine"
]
