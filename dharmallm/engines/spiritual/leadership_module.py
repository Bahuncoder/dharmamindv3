"""
👑 Leadership Module - Dharmic Leadership and Righteous Governance
Complete leadership system based on Ramarajya, Krishna's guidance, and Chanakya's wisdom
Develops leaders who serve with dharma, wisdom, and compassion
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class LeadershipType(Enum):
    """Types of dharmic leadership"""
    RAJAS_LEADER = "rajas_leader"         # Righteous king/president
    ACHARYA_LEADER = "acharya_leader"     # Teacher and guide
    SEVA_LEADER = "seva_leader"           # Service-oriented leader
    SPIRITUAL_LEADER = "spiritual_leader" # Guru and wisdom keeper
    COMMUNITY_LEADER = "community_leader" # Local and social leader
    FAMILY_LEADER = "family_leader"       # Household and family guide

class LeadershipLevel(Enum):
    """Levels of leadership development"""
    EMERGING = "emerging"           # New leader learning basics
    DEVELOPING = "developing"       # Building skills and character
    ESTABLISHED = "established"     # Proven leader with experience
    MASTERFUL = "masterful"         # Wise leader inspiring others
    SAGE_LEADER = "sage_leader"     # Transcendent leadership through wisdom

class LeadershipPrinciple(Enum):
    """Core dharmic leadership principles"""
    DHARMIC_DUTY = "dharmic_duty"           # Following righteous path
    SELFLESS_SERVICE = "selfless_service"   # Leading through seva
    WISDOM_GUIDANCE = "wisdom_guidance"     # Decision-making with wisdom
    COMPASSION = "compassion"               # Leading with heart
    INTEGRITY = "integrity"                 # Truthfulness and authenticity
    COURAGE = "courage"                     # Facing challenges bravely
    HUMILITY = "humility"                   # Leading without ego
    JUSTICE = "justice"                     # Fair and righteous governance

class LeadershipChallenge(Enum):
    """Common leadership challenges"""
    EGO_INFLATION = "ego_inflation"               # Pride and arrogance
    DECISION_PARALYSIS = "decision_paralysis"     # Fear of making decisions
    PEOPLE_PLEASING = "people_pleasing"           # Avoiding difficult decisions
    POWER_CORRUPTION = "power_corruption"         # Misuse of authority
    BURNOUT = "burnout"                          # Overwhelm and exhaustion
    CRITICISM_HANDLING = "criticism_handling"     # Dealing with opposition
    DELEGATION_DIFFICULTY = "delegation"          # Inability to trust others
    VISION_CLARITY = "vision_clarity"            # Unclear purpose and direction

@dataclass
class LeadershipGuidance:
    """Comprehensive leadership guidance"""
    level: LeadershipLevel
    leadership_type: LeadershipType
    core_principles: List[LeadershipPrinciple]
    daily_practices: List[str]
    decision_frameworks: List[str]
    character_qualities: List[str]
    challenges_to_overcome: Dict[str, str]
    scriptural_examples: List[str]
    practical_applications: List[str]

@dataclass
class LeadershipDecision:
    """Framework for dharmic decision making"""
    situation: str
    stakeholders: List[str]
    dharmic_considerations: List[str]
    potential_outcomes: Dict[str, List[str]]
    recommended_approach: str
    implementation_steps: List[str]
    success_metrics: List[str]

class LeadershipInsight(BaseModel):
    """Insight from leadership module"""
    leadership_type: str = Field(description="Type of leadership role")
    core_principle: str = Field(description="Most relevant leadership principle")
    guidance: str = Field(description="Specific leadership guidance")
    challenge_solution: str = Field(description="Solution for current challenge")

class LeadershipResponse(BaseModel):
    """Response from Leadership module"""
    leadership_level: str = Field(description="Current level of leadership development")
    leadership_type: str = Field(description="Primary type of leadership role")
    core_principles: List[str] = Field(description="Key dharmic leadership principles")
    daily_practices: List[str] = Field(description="Daily leadership development practices")
    decision_framework: List[str] = Field(description="Framework for making dharmic decisions")
    character_development: List[str] = Field(description="Character qualities to cultivate")
    challenge_solutions: Dict[str, str] = Field(description="Solutions for leadership challenges")
    scriptural_wisdom: str = Field(description="Relevant scriptural guidance")
    practical_applications: List[str] = Field(description="Practical ways to apply leadership")
    vision_guidance: str = Field(description="Guidance on developing clear vision")

class LeadershipModule:
    """
    👑 Leadership Module - Dharmic Leadership and Righteous Governance
    
    Based on authentic dharmic leadership examples:
    - Lord Rama's Ramarajya (ideal governance)
    - Krishna's guidance in Bhagavad Gita
    - Chanakya's Arthashastra (political wisdom)
    - Bhishma's leadership principles
    
    Develops leaders who serve through dharma, not ego
    """
    
    def __init__(self):
        self.name = "Leadership"
        self.color = "👑"
        self.element = "Dharmic Authority"
        self.principles = ["Service", "Wisdom", "Justice", "Compassion"]
        self.guidance_levels = self._initialize_guidance_levels()
        self.leadership_examples = self._initialize_scriptural_examples()
        self.decision_frameworks = self._initialize_decision_frameworks()
        self.challenge_solutions = self._initialize_challenge_solutions()
        
    def _initialize_guidance_levels(self) -> Dict[LeadershipLevel, LeadershipGuidance]:
        """Initialize guidance for different levels of leadership"""
        return {
            LeadershipLevel.EMERGING: LeadershipGuidance(
                level=LeadershipLevel.EMERGING,
                leadership_type=LeadershipType.COMMUNITY_LEADER,
                core_principles=[
                    LeadershipPrinciple.SELFLESS_SERVICE,
                    LeadershipPrinciple.HUMILITY,
                    LeadershipPrinciple.INTEGRITY
                ],
                daily_practices=[
                    "Morning reflection on service opportunities",
                    "Study biographies of dharmic leaders",
                    "Practice listening deeply to others",
                    "Evening self-assessment of leadership actions",
                    "Seek mentorship from experienced leaders"
                ],
                decision_frameworks=[
                    "Will this serve the greater good?",
                    "Is this action dharmic and ethical?",
                    "How will this affect all stakeholders?",
                    "Am I acting from ego or service?"
                ],
                character_qualities=[
                    "Develop patience and emotional regulation",
                    "Cultivate genuine care for others' wellbeing",
                    "Build trustworthiness through consistent actions",
                    "Practice speaking truth with kindness"
                ],
                challenges_to_overcome={
                    "Self-doubt": "Build confidence through small leadership successes and study",
                    "Fear of mistakes": "View errors as learning opportunities, not failures",
                    "Seeking approval": "Focus on dharmic duty rather than pleasing everyone"
                },
                scriptural_examples=[
                    "Young Rama's obedience and service to parents",
                    "Arjuna's willingness to learn from Krishna",
                    "Hanuman's devoted service to his mission"
                ],
                practical_applications=[
                    "Lead small projects or initiatives",
                    "Serve in community organizations",
                    "Practice conflict resolution in daily life",
                    "Mentor newcomers in your field"
                ]
            ),
            
            LeadershipLevel.DEVELOPING: LeadershipGuidance(
                level=LeadershipLevel.DEVELOPING,
                leadership_type=LeadershipType.SEVA_LEADER,
                core_principles=[
                    LeadershipPrinciple.WISDOM_GUIDANCE,
                    LeadershipPrinciple.JUSTICE,
                    LeadershipPrinciple.COURAGE,
                    LeadershipPrinciple.COMPASSION
                ],
                daily_practices=[
                    "Strategic thinking and planning sessions",
                    "Regular study of dharmic texts and leadership",
                    "Practice difficult conversations with compassion",
                    "Build networks focused on service, not personal gain",
                    "Regular feedback gathering from those you serve"
                ],
                decision_frameworks=[
                    "What would Krishna advise in this situation?",
                    "How can I balance competing interests fairly?",
                    "What are the long-term consequences of this choice?",
                    "How can I involve others in the decision process?"
                ],
                character_qualities=[
                    "Develop strategic thinking while maintaining ethics",
                    "Build resilience to handle criticism and setbacks",
                    "Cultivate ability to see multiple perspectives",
                    "Practice delegation while maintaining accountability"
                ],
                challenges_to_overcome={
                    "People pleasing": "Learn to make necessary but unpopular decisions",
                    "Micromanagement": "Trust others and focus on empowerment",
                    "Overwhelm": "Prioritize effectively and delegate appropriately"
                },
                scriptural_examples=[
                    "Rama's decisions during forest exile",
                    "Krishna's strategic guidance during Kurukshetra",
                    "Chandragupta's learning under Chanakya"
                ],
                practical_applications=[
                    "Lead larger teams and organizations",
                    "Take on challenging projects with ethical dimensions",
                    "Serve on boards or committees",
                    "Develop other leaders through mentoring"
                ]
            ),
            
            LeadershipLevel.ESTABLISHED: LeadershipGuidance(
                level=LeadershipLevel.ESTABLISHED,
                leadership_type=LeadershipType.RAJAS_LEADER,
                core_principles=[
                    LeadershipPrinciple.DHARMIC_DUTY,
                    LeadershipPrinciple.WISDOM_GUIDANCE,
                    LeadershipPrinciple.JUSTICE,
                    LeadershipPrinciple.COURAGE
                ],
                daily_practices=[
                    "Deep contemplation on dharmic duty and purpose",
                    "Regular consultation with wise advisors",
                    "Study of governance and leadership classics",
                    "Personal practice maintaining spiritual discipline",
                    "Regular connection with people being served"
                ],
                decision_frameworks=[
                    "How does this align with dharmic principles?",
                    "What would ideal leaders like Rama do?",
                    "How can I serve the greatest number while protecting minorities?",
                    "What is my duty regardless of personal preferences?"
                ],
                character_qualities=[
                    "Unwavering commitment to dharmic principles",
                    "Ability to make difficult decisions for greater good",
                    "Balance of firmness and compassion",
                    "Vision for long-term welfare of all"
                ],
                challenges_to_overcome={
                    "Power corruption": "Regular spiritual practice and accountability",
                    "Isolation": "Maintain connection with all levels of society",
                    "Decision fatigue": "Build strong advisory systems"
                },
                scriptural_examples=[
                    "Rama's establishment of Ramarajya",
                    "Yudhishthira's righteous governance",
                    "Bharata's selfless leadership while Rama was in exile"
                ],
                practical_applications=[
                    "Lead major organizations or institutions",
                    "Take responsibility for significant social impact",
                    "Influence policy and systemic change",
                    "Establish lasting institutions for service"
                ]
            ),
            
            LeadershipLevel.MASTERFUL: LeadershipGuidance(
                level=LeadershipLevel.MASTERFUL,
                leadership_type=LeadershipType.ACHARYA_LEADER,
                core_principles=[
                    LeadershipPrinciple.WISDOM_GUIDANCE,
                    LeadershipPrinciple.DHARMIC_DUTY,
                    LeadershipPrinciple.SELFLESS_SERVICE,
                    LeadershipPrinciple.HUMILITY
                ],
                daily_practices=[
                    "Teaching and mentoring next generation of leaders",
                    "Deep study and contemplation of eternal principles",
                    "Living as example of dharmic leadership",
                    "Serving as advisor to other leaders",
                    "Preserving and transmitting wisdom traditions"
                ],
                decision_frameworks=[
                    "How will this serve humanity's highest evolution?",
                    "What would the rishis and sages counsel?",
                    "How can I guide without controlling?",
                    "What legacy am I creating for future generations?"
                ],
                character_qualities=[
                    "Complete detachment from personal gain",
                    "Natural wisdom that inspires others",
                    "Ability to see the highest potential in all",
                    "Balance of strength and gentleness"
                ],
                challenges_to_overcome={
                    "Relevance": "Adapt ancient wisdom to modern contexts",
                    "Succession": "Prepare worthy successors",
                    "Detachment": "Serve without attachment to outcomes"
                },
                scriptural_examples=[
                    "Krishna as the ultimate teacher-leader",
                    "Bhishma's guidance throughout his life",
                    "Dronacharya's role as teacher of warriors"
                ],
                practical_applications=[
                    "Establish schools and training institutions",
                    "Write and teach on dharmic leadership",
                    "Serve as elder statesman or advisor",
                    "Create systems that will outlast personal involvement"
                ]
            ),
            
            LeadershipLevel.SAGE_LEADER: LeadershipGuidance(
                level=LeadershipLevel.SAGE_LEADER,
                leadership_type=LeadershipType.SPIRITUAL_LEADER,
                core_principles=[
                    LeadershipPrinciple.DHARMIC_DUTY,
                    LeadershipPrinciple.WISDOM_GUIDANCE,
                    LeadershipPrinciple.SELFLESS_SERVICE,
                    LeadershipPrinciple.HUMILITY
                ],
                daily_practices=[
                    "Living in constant awareness of dharmic duty",
                    "Leading through presence and example",
                    "Offering guidance when sought, silence when not",
                    "Serving the divine plan through human leadership",
                    "Maintaining beginner's mind despite experience"
                ],
                decision_frameworks=[
                    "What serves the divine will and cosmic order?",
                    "How can I be an instrument of higher purpose?",
                    "What does dharma require in this moment?",
                    "How can I serve while remaining empty of ego?"
                ],
                character_qualities=[
                    "Complete surrender to divine will",
                    "Natural authority without claiming power",
                    "Spontaneous wisdom appropriate to each situation",
                    "Universal compassion without favoritism"
                ],
                challenges_to_overcome={
                    "Human limitations": "Accept role as instrument, not doer",
                    "World's resistance": "Serve faithfully regardless of reception",
                    "Personal desires": "Transcend all personal agenda"
                },
                scriptural_examples=[
                    "Krishna's role as cosmic teacher and guide",
                    "Rama as embodiment of ideal leadership",
                    "The Rishis as guardians of dharmic wisdom"
                ],
                practical_applications=[
                    "Serve as spiritual guide and teacher",
                    "Influence through wisdom rather than position",
                    "Establish enduring dharmic institutions",
                    "Live as inspiration for future generations"
                ]
            )
        }
    
    def _initialize_scriptural_examples(self) -> Dict[str, Dict[str, Any]]:
        """Initialize examples from scriptures and history"""
        return {
            "Rama_Leadership": {
                "context": "Ideal governance and personal conduct",
                "principles": [
                    "Put duty before personal preference",
                    "Listen to all subjects equally",
                    "Maintain dharma even in difficult situations",
                    "Lead by personal example"
                ],
                "lessons": [
                    "True leaders serve rather than rule",
                    "Personal integrity is foundation of public trust",
                    "Difficult decisions must be made for greater good",
                    "A leader's conduct affects entire society"
                ]
            },
            
            "Krishna_Guidance": {
                "context": "Strategic leadership and spiritual wisdom",
                "principles": [
                    "Guide without controlling",
                    "Teach through practical situations",
                    "Balance strategic thinking with dharmic principles",
                    "Serve the divine plan through human action"
                ],
                "lessons": [
                    "True leadership is teaching and empowerment",
                    "Wisdom must be applied to real-world challenges",
                    "A leader must sometimes bear the burden of difficult decisions",
                    "Personal transformation enables transformational leadership"
                ]
            },
            
            "Chanakya_Wisdom": {
                "context": "Political strategy and governance principles",
                "principles": [
                    "Combine idealism with practical realism",
                    "Protect dharma through strategic action",
                    "Build systems that outlast individuals",
                    "Balance different interests for overall welfare"
                ],
                "lessons": [
                    "Effective leadership requires both vision and execution",
                    "Systems and institutions matter more than personalities",
                    "Leaders must sometimes work behind the scenes",
                    "Long-term thinking prevents short-term disasters"
                ]
            }
        }
    
    def _initialize_decision_frameworks(self) -> List[str]:
        """Initialize dharmic decision-making frameworks"""
        return [
            "Dharmic Analysis: Is this action aligned with dharma?",
            "Stakeholder Impact: Who will be affected and how?",
            "Long-term Consequences: What are the extended effects?",
            "Personal Motivation: Am I acting from ego or service?",
            "Resource Stewardship: Am I using resources wisely?",
            "Precedent Setting: What example does this set?",
            "Universal Principles: Would this be good if everyone did it?",
            "Guru's Guidance: What would my teachers advise?"
        ]
    
    def _initialize_challenge_solutions(self) -> Dict[LeadershipChallenge, Dict[str, Any]]:
        """Initialize solutions for leadership challenges"""
        return {
            LeadershipChallenge.EGO_INFLATION: {
                "symptoms": ["Arrogance", "Not listening to others", "Believing own praise"],
                "solutions": [
                    "Regular spiritual practice and self-reflection",
                    "Seek feedback from trusted advisors",
                    "Remember that leadership is service, not privilege",
                    "Study stories of humble leaders"
                ],
                "practices": ["Daily humility reflection", "Serving in anonymous ways"],
                "mantras": ["न अहम् (Not I)", "सेवक (Servant)"]
            },
            
            LeadershipChallenge.DECISION_PARALYSIS: {
                "symptoms": ["Avoiding decisions", "Over-analyzing", "Seeking excessive input"],
                "solutions": [
                    "Develop clear decision-making frameworks",
                    "Set decision deadlines",
                    "Accept that perfect information is rarely available",
                    "Learn from decisions rather than regretting them"
                ],
                "practices": ["Daily decision practice", "Quick decision exercises"],
                "mantras": ["धर्मे युद्धे (In dharmic battle)", "कर्मण्येवाधिकारस्ते (Your right is to action)"]
            },
            
            LeadershipChallenge.POWER_CORRUPTION: {
                "symptoms": ["Using position for personal gain", "Different rules for self"],
                "solutions": [
                    "Regular spiritual accountability",
                    "Transparent decision-making processes",
                    "Remember power is temporary stewardship",
                    "Focus on leaving positive legacy"
                ],
                "practices": ["Daily dharmic review", "Anonymous service"],
                "mantras": ["राजधर्म (Royal duty)", "लोकसंग्रह (Welfare of the world)"]
            },
            
            LeadershipChallenge.BURNOUT: {
                "symptoms": ["Exhaustion", "Cynicism", "Decreased effectiveness"],
                "solutions": [
                    "Delegate effectively to capable team members",
                    "Maintain spiritual practices for renewal",
                    "Set boundaries between work and personal time",
                    "Remember that you are not indispensable"
                ],
                "practices": ["Regular retreat and reflection", "Physical exercise"],
                "mantras": ["ईश्वरप्रणिधान (Surrender to Divine)", "शांति (Peace)"]
            }
        }
    
    def assess_leadership_level(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> LeadershipLevel:
        """Assess user's current leadership level"""
        query_lower = query.lower()
        context = user_context or {}
        
        # Check for sage/master level indicators
        if any(word in query_lower for word in ["sage", "wisdom keeper", "spiritual leader", "transcendent"]):
            return LeadershipLevel.SAGE_LEADER
        
        # Check for masterful level indicators
        if any(word in query_lower for word in ["master", "teacher", "mentor", "guide others", "established leader"]):
            return LeadershipLevel.MASTERFUL
        
        # Check for established level indicators
        if any(word in query_lower for word in ["organization", "institution", "major responsibility", "governance"]):
            return LeadershipLevel.ESTABLISHED
        
        # Check for developing level indicators
        if any(word in query_lower for word in ["team", "project", "developing", "building skills"]):
            return LeadershipLevel.DEVELOPING
        
        # Default to emerging
        return LeadershipLevel.EMERGING
    
    def identify_leadership_type(self, query: str, context: Dict[str, Any]) -> LeadershipType:
        """Identify the type of leadership being discussed"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["spiritual", "teacher", "guru", "guide"]):
            return LeadershipType.SPIRITUAL_LEADER
        elif any(word in query_lower for word in ["government", "politics", "policy", "governance"]):
            return LeadershipType.RAJAS_LEADER
        elif any(word in query_lower for word in ["service", "volunteer", "help", "community"]):
            return LeadershipType.SEVA_LEADER
        elif any(word in query_lower for word in ["family", "household", "children", "home"]):
            return LeadershipType.FAMILY_LEADER
        elif any(word in query_lower for word in ["teach", "education", "mentor", "training"]):
            return LeadershipType.ACHARYA_LEADER
        else:
            return LeadershipType.COMMUNITY_LEADER
    
    def identify_leadership_challenges(self, query: str, context: Dict[str, Any]) -> List[LeadershipChallenge]:
        """Identify leadership challenges mentioned in query"""
        challenges = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["ego", "pride", "arrogant"]):
            challenges.append(LeadershipChallenge.EGO_INFLATION)
        
        if any(word in query_lower for word in ["decision", "choice", "unsure", "paralyzed"]):
            challenges.append(LeadershipChallenge.DECISION_PARALYSIS)
        
        if any(word in query_lower for word in ["burnout", "exhausted", "overwhelmed"]):
            challenges.append(LeadershipChallenge.BURNOUT)
        
        if any(word in query_lower for word in ["criticism", "opposition", "resistance"]):
            challenges.append(LeadershipChallenge.CRITICISM_HANDLING)
        
        if any(word in query_lower for word in ["delegate", "trust", "control"]):
            challenges.append(LeadershipChallenge.DELEGATION_DIFFICULTY)
        
        return challenges if challenges else [LeadershipChallenge.VISION_CLARITY]
    
    def get_challenge_solutions(self, challenges: List[LeadershipChallenge]) -> Dict[str, str]:
        """Get solutions for identified challenges"""
        solutions = {}
        
        for challenge in challenges:
            challenge_data = self.challenge_solutions.get(challenge, {})
            solutions[challenge.value] = "; ".join(challenge_data.get("solutions", ["Practice patience and seek guidance"]))
        
        return solutions
    
    def get_vision_guidance(self, leadership_type: LeadershipType, level: LeadershipLevel) -> str:
        """Provide guidance on developing clear leadership vision"""
        if level in [LeadershipLevel.EMERGING, LeadershipLevel.DEVELOPING]:
            return "Focus on understanding your purpose and how you can serve others. Start with small, clear goals and build from there."
        elif level == LeadershipLevel.ESTABLISHED:
            return "Develop a long-term vision that serves the greatest good. Consider how your leadership can create lasting positive change."
        else:  # Masterful/Sage
            return "Your vision should align with dharmic principles and serve the evolution of consciousness. Think in terms of generations, not just immediate results."
    
    def get_practical_applications(self, level: LeadershipLevel, leadership_type: LeadershipType) -> List[str]:
        """Get practical applications for leadership development"""
        guidance = self.guidance_levels.get(level)
        base_applications = guidance.practical_applications if guidance else []
        
        # Add type-specific applications
        type_applications = []
        if leadership_type == LeadershipType.SPIRITUAL_LEADER:
            type_applications.extend(["Offer spiritual guidance", "Create sacred spaces", "Teach dharmic principles"])
        elif leadership_type == LeadershipType.SEVA_LEADER:
            type_applications.extend(["Organize service projects", "Help those in need", "Build community support"])
        
        return base_applications + type_applications
    
    async def process_leadership_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> LeadershipResponse:
        """Process leadership-related query and provide comprehensive guidance"""
        try:
            context = user_context or {}
            
            # Assess leadership aspects
            level = self.assess_leadership_level(query, context)
            leadership_type = self.identify_leadership_type(query, context)
            challenges = self.identify_leadership_challenges(query, context)
            
            # Get guidance
            guidance = self.guidance_levels.get(level)
            if not guidance:
                return self._create_fallback_response()
            
            # Generate specific guidance
            challenge_solutions = self.get_challenge_solutions(challenges)
            vision_guidance = self.get_vision_guidance(leadership_type, level)
            practical_applications = self.get_practical_applications(level, leadership_type)
            
            # Select scriptural example
            scriptural_examples = list(self.leadership_examples.keys())
            example_key = scriptural_examples[0] if scriptural_examples else "Krishna_Guidance"
            example = self.leadership_examples.get(example_key, {})
            scriptural_wisdom = f"Example of {example_key.replace('_', ' ')}: {example.get('lessons', ['Lead through dharma'])[0]}"
            
            return LeadershipResponse(
                leadership_level=level.value,
                leadership_type=leadership_type.value,
                core_principles=[principle.value for principle in guidance.core_principles],
                daily_practices=guidance.daily_practices,
                decision_framework=guidance.decision_frameworks,
                character_development=guidance.character_qualities,
                challenge_solutions=challenge_solutions,
                scriptural_wisdom=scriptural_wisdom,
                practical_applications=practical_applications,
                vision_guidance=vision_guidance
            )
            
        except Exception as e:
            logger.error(f"Error processing leadership query: {e}")
            return self._create_fallback_response()
    
    def _create_fallback_response(self) -> LeadershipResponse:
        """Create fallback response when processing fails"""
        return LeadershipResponse(
            leadership_level="emerging",
            leadership_type="community_leader",
            core_principles=["selfless_service", "integrity", "humility"],
            daily_practices=[
                "Morning reflection on service opportunities",
                "Study examples of dharmic leaders",
                "Practice listening deeply to others",
                "Evening self-assessment of leadership actions"
            ],
            decision_framework=self.decision_frameworks,
            character_development=[
                "Develop patience and emotional regulation",
                "Cultivate genuine care for others",
                "Build trustworthiness through consistent actions",
                "Practice speaking truth with kindness"
            ],
            challenge_solutions={
                "self_doubt": "Build confidence through small successes and continuous learning",
                "fear_of_mistakes": "View errors as learning opportunities rather than failures"
            },
            scriptural_wisdom="Example of Rama: True leaders serve rather than rule, maintaining dharma even in difficult situations.",
            practical_applications=[
                "Lead small projects or initiatives",
                "Serve in community organizations",
                "Practice conflict resolution",
                "Mentor others in your field"
            ],
            vision_guidance="Focus on understanding your purpose and how you can serve others. Start with small, clear goals and build from there."
        )
    
    def get_leadership_insight(self, principle: LeadershipPrinciple) -> Optional[LeadershipInsight]:
        """Get specific insight about a leadership principle"""
        principle_guidance = {
            LeadershipPrinciple.DHARMIC_DUTY: "Lead according to dharmic principles, not personal preferences",
            LeadershipPrinciple.SELFLESS_SERVICE: "True leadership is service to others, not self-aggrandizement",
            LeadershipPrinciple.WISDOM_GUIDANCE: "Make decisions based on wisdom and long-term thinking",
            LeadershipPrinciple.COMPASSION: "Lead with heart while maintaining necessary firmness"
        }
        
        guidance = principle_guidance.get(principle, "Lead with dharma and service")
        
        return LeadershipInsight(
            leadership_type="dharmic_leader",
            core_principle=principle.value,
            guidance=guidance,
            challenge_solution="Practice daily self-reflection and seek guidance from wise mentors"
        )

# Global instance
_leadership_module = None

def get_leadership_module() -> LeadershipModule:
    """Get global Leadership module instance"""
    global _leadership_module
    if _leadership_module is None:
        _leadership_module = LeadershipModule()
    return _leadership_module

# Factory function for easy access
def create_leadership_guidance(query: str, user_context: Optional[Dict[str, Any]] = None) -> LeadershipResponse:
    """Factory function to create leadership guidance"""
    import asyncio
    module = get_leadership_module()
    return asyncio.run(module.process_leadership_query(query, user_context))
