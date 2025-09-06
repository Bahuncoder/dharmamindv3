"""
💼 Career Crisis Module - Dharmic Guidance for Professional Challenges
Spiritual wisdom for job loss, career transitions, workplace stress,
and finding one's purpose in work through dharmic principles
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CareerCrisisType(Enum):
    """Types of career crises"""
    JOB_LOSS = "job_loss"
    CAREER_CHANGE = "career_change"
    WORKPLACE_STRESS = "workplace_stress"
    PURPOSE_SEEKING = "purpose_seeking"
    ETHICAL_DILEMMA = "ethical_dilemma"
    PROMOTION_FAILURE = "promotion_failure"
    BURNOUT = "burnout"
    AGE_DISCRIMINATION = "age_discrimination"


class CareerApproach(Enum):
    """Dharmic approaches to career challenges"""
    DHARMA_WORK = "dharma_work"          # Finding dharmic purpose
    KARMA_YOGA = "karma_yoga"            # Work as spiritual practice
    SURRENDER = "surrender"              # Divine will acceptance
    SKILL_BUILDING = "skill_building"    # Developing capabilities
    NETWORKING = "networking"            # Building relationships
    PATIENCE = "patience"                # Waiting for right opportunity


@dataclass
class CareerGuidance:
    """Comprehensive career crisis guidance"""
    crisis_type: CareerCrisisType
    spiritual_perspective: str
    dharmic_approach: List[CareerApproach]
    daily_practices: List[str]
    practical_steps: List[str]
    mindset_shifts: List[str]
    prayers_mantras: List[str]
    scriptural_wisdom: str
    long_term_vision: List[str]


class CareerCrisisResponse(BaseModel):
    """Response from Career Crisis module"""
    crisis_type: str = Field(description="Type of career crisis")
    spiritual_perspective: str = Field(description="Dharmic view")
    practical_guidance: List[str] = Field(description="Practical steps")
    daily_practices: List[str] = Field(description="Daily practices")
    mindset_transformation: List[str] = Field(description="Mental shifts")
    prayers_mantras: List[str] = Field(description="Spiritual practices")
    dharmic_wisdom: str = Field(description="Scriptural guidance")
    long_term_vision: List[str] = Field(description="Future planning")


class CareerCrisisModule:
    """
    💼 Career Crisis Module - Dharmic Guidance for Professional Challenges
    
    Provides spiritual wisdom for:
    - Job loss and unemployment struggles
    - Career transitions and purpose finding
    - Workplace stress and ethical dilemmas
    - Professional burnout and recovery
    - Age-related career challenges
    
    Based on karma yoga principles and dharmic work ethics.
    """
    
    def __init__(self):
        self.name = "Career Crisis"
        self.color = "💼"
        self.element = "Professional Dharma"
        self.principles = ["Right Livelihood", "Karma Yoga",
                           "Purpose Alignment", "Skill Development"]
        self.guidance_types = self._initialize_career_guidance()
        self.mantras = self._initialize_career_mantras()
        self.success_principles = self._initialize_success_principles()
    
    def _initialize_career_guidance(self) -> Dict[CareerCrisisType, CareerGuidance]:
        """Initialize guidance for different career crises"""
        return {
            CareerCrisisType.JOB_LOSS: CareerGuidance(
                crisis_type=CareerCrisisType.JOB_LOSS,
                spiritual_perspective=(
                    "Job loss is often divine redirection toward your true "
                    "dharma. It creates space for reflection on what truly "
                    "serves your soul's purpose and society's wellbeing."
                ),
                dharmic_approach=[
                    CareerApproach.SURRENDER,
                    CareerApproach.DHARMA_WORK,
                    CareerApproach.SKILL_BUILDING
                ],
                daily_practices=[
                    "Morning gratitude for new opportunities",
                    "Skill development or learning time",
                    "Networking with positive intention",
                    "Evening reflection on lessons learned",
                    "Service activities to maintain purpose"
                ],
                practical_steps=[
                    "Update resume with dharmic language",
                    "Network with integrity and service mindset",
                    "Explore careers aligned with values",
                    "Maintain financial discipline",
                    "Use time for spiritual growth"
                ],
                mindset_shifts=[
                    "From victim to student of life",
                    "From desperation to divine timing trust",
                    "From any job to right livelihood",
                    "From fear to faith in provision",
                    "From loss to opportunity for growth"
                ],
                prayers_mantras=[
                    "Om Gam Ganapataye Namaha - Remove obstacles",
                    "Shri Ram Jai Ram - Divine provision",
                    "Om Hreem Shreem Klim - Prosperity mantra"
                ],
                scriptural_wisdom=(
                    "Bhagavad Gita 18.46: 'A person can attain perfection by "
                    "worshipping God through the performance of their natural "
                    "work.' Trust that right work will come."
                ),
                long_term_vision=[
                    "Identify work that serves others and self",
                    "Build skills that create genuine value",
                    "Develop multiple income streams ethically",
                    "Create work-life integration not just balance",
                    "Become mentor for others facing similar challenges"
                ]
            ),
            
            CareerCrisisType.BURNOUT: CareerGuidance(
                crisis_type=CareerCrisisType.BURNOUT,
                spiritual_perspective=(
                    "Burnout indicates disconnection from dharmic purpose. "
                    "It calls for returning to work as worship and finding "
                    "sacred meaning in professional service."
                ),
                dharmic_approach=[
                    CareerApproach.KARMA_YOGA,
                    CareerApproach.SURRENDER,
                    CareerApproach.PATIENCE
                ],
                daily_practices=[
                    "Begin work with dedication to service",
                    "Take conscious breaks for breath awareness",
                    "Practice detachment from results",
                    "End workday with gratitude ritual",
                    "Evening meditation for stress release"
                ],
                practical_steps=[
                    "Set boundaries between work and personal time",
                    "Delegate tasks where possible",
                    "Request workload adjustment",
                    "Take earned vacation time",
                    "Seek counseling if needed"
                ],
                mindset_shifts=[
                    "From endless striving to sustainable effort",
                    "From perfectionism to excellence with detachment",
                    "From work as burden to work as service",
                    "From external validation to inner satisfaction",
                    "From competition to collaboration"
                ],
                prayers_mantras=[
                    "Om Namah Shivaya - Inner peace mantra",
                    "So Hum - I am That (divine connection)",
                    "Om Shanti - Peace in all activities"
                ],
                scriptural_wisdom=(
                    "Bhagavad Gita 2.47: 'You have the right to perform your "
                    "duties, but not to the fruits of action.' Focus on "
                    "process, not just outcomes."
                ),
                long_term_vision=[
                    "Create sustainable work practices",
                    "Integrate spiritual practices into workday",
                    "Find mentor or spiritual guide for work-life",
                    "Consider career pivot if values misaligned",
                    "Become advocate for healthy work culture"
                ]
            ),
            
            CareerCrisisType.PURPOSE_SEEKING: CareerGuidance(
                crisis_type=CareerCrisisType.PURPOSE_SEEKING,
                spiritual_perspective=(
                    "The search for purpose is the soul's call to align work "
                    "with dharma. Your unique skills and passions point toward "
                    "how you can serve the world meaningfully."
                ),
                dharmic_approach=[
                    CareerApproach.DHARMA_WORK,
                    CareerApproach.SKILL_BUILDING,
                    CareerApproach.SURRENDER
                ],
                daily_practices=[
                    "Morning intention setting for purpose discovery",
                    "Journaling on values and passions",
                    "Skill assessment and development",
                    "Service activities to explore interests",
                    "Evening gratitude for growth opportunities"
                ],
                practical_steps=[
                    "Complete personality and skills assessments",
                    "Interview people in interesting careers",
                    "Volunteer in areas of potential interest",
                    "Take courses or workshops to explore",
                    "Start side projects aligned with interests"
                ],
                mindset_shifts=[
                    "From external expectations to inner calling",
                    "From money-first to purpose-first thinking",
                    "From single career to portfolio approach",
                    "From certainty-seeking to exploration mindset",
                    "From perfect path to evolving journey"
                ],
                prayers_mantras=[
                    "Om Aim Saraswati Namaha - Wisdom and clarity",
                    "Gayatri Mantra - Divine illumination",
                    "Om Gum Guruve Namaha - Guidance from teachers"
                ],
                scriptural_wisdom=(
                    "Bhagavad Gita 18.45: 'Everyone can attain perfection by "
                    "doing their natural work with devotion.' Your dharma "
                    "is unique to you."
                ),
                long_term_vision=[
                    "Align career with core values and strengths",
                    "Create work that serves others meaningfully",
                    "Build expertise in areas of natural talent",
                    "Integrate spiritual growth with professional growth",
                    "Become example of purposeful living for others"
                ]
            )
        }
    
    def _initialize_career_mantras(self) -> Dict[str, List[str]]:
        """Initialize mantras for different career situations"""
        return {
            "job_search": [
                "Om Shreem Hreem Klim Maha Lakshmiyei Namaha",
                "Om Gam Ganapataye Namaha",
                "Shri Ram Jai Ram Jai Jai Ram"
            ],
            "workplace_peace": [
                "Om Namah Shivaya",
                "Om Shanti Shanti Shanti",
                "So Hum - I am That"
            ],
            "career_clarity": [
                "Om Aim Saraswati Namaha",
                "Gayatri Mantra",
                "Om Gum Guruve Namaha"
            ]
        }
    
    def _initialize_success_principles(self) -> List[str]:
        """Initialize dharmic success principles"""
        return [
            "Right livelihood that harms none",
            "Work as worship and service to Divine",
            "Skill development as spiritual practice",
            "Honest effort with detachment from results",
            "Sharing knowledge and mentoring others",
            "Balancing material needs with spiritual growth",
            "Creating value for society through work"
        ]
    
    def assess_crisis_type(self, query: str, context: Dict[str, Any]) -> CareerCrisisType:
        """Assess the type of career crisis from query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["lost job", "fired", "laid off", "unemployed"]):
            return CareerCrisisType.JOB_LOSS
        elif any(word in query_lower for word in ["burnout", "burned out", "exhausted", "overwhelmed"]):
            return CareerCrisisType.BURNOUT
        elif any(word in query_lower for word in ["purpose", "meaning", "calling", "passion"]):
            return CareerCrisisType.PURPOSE_SEEKING
        elif any(word in query_lower for word in ["career change", "transition", "switch"]):
            return CareerCrisisType.CAREER_CHANGE
        elif any(word in query_lower for word in ["stress", "pressure", "workplace", "toxic"]):
            return CareerCrisisType.WORKPLACE_STRESS
        elif any(word in query_lower for word in ["ethics", "moral", "wrong", "unethical"]):
            return CareerCrisisType.ETHICAL_DILEMMA
        else:
            return CareerCrisisType.PURPOSE_SEEKING
    
    async def process_career_crisis_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> CareerCrisisResponse:
        """Process career crisis query and provide dharmic guidance"""
        try:
            context = user_context or {}
            
            # Assess crisis type
            crisis_type = self.assess_crisis_type(query, context)
            
            # Get guidance
            guidance = self.guidance_types.get(crisis_type)
            if not guidance:
                return self._create_fallback_response()
            
            return CareerCrisisResponse(
                crisis_type=crisis_type.value,
                spiritual_perspective=guidance.spiritual_perspective,
                practical_guidance=guidance.practical_steps,
                daily_practices=guidance.daily_practices,
                mindset_transformation=guidance.mindset_shifts,
                prayers_mantras=guidance.prayers_mantras,
                dharmic_wisdom=guidance.scriptural_wisdom,
                long_term_vision=guidance.long_term_vision
            )
            
        except Exception as e:
            logger.error(f"Error processing career crisis query: {e}")
            return self._create_fallback_response()
    
    def _create_fallback_response(self) -> CareerCrisisResponse:
        """Create fallback response when processing fails"""
        return CareerCrisisResponse(
            crisis_type="general_career_support",
            spiritual_perspective="All work can become spiritual practice when done with right intention",
            practical_guidance=["Assess skills and values", "Network with integrity", "Stay open to opportunities"],
            daily_practices=["Morning intention for meaningful work", "Work as service practice", "Evening gratitude"],
            mindset_transformation=["From job to calling", "From survival to service", "From fear to faith"],
            prayers_mantras=["Om Gam Ganapataye Namaha", "Gayatri Mantra", "Om Namah Shivaya"],
            dharmic_wisdom="Right livelihood is one of the foundations of spiritual life",
            long_term_vision=["Align work with dharma", "Serve others through profession", "Integrate growth with service"]
        )


# Global instance
_career_crisis_module = None

def get_career_crisis_module() -> CareerCrisisModule:
    """Get global Career Crisis module instance"""
    global _career_crisis_module
    if _career_crisis_module is None:
        _career_crisis_module = CareerCrisisModule()
    return _career_crisis_module

# Factory function for easy access
def create_career_crisis_guidance(query: str, context: Optional[Dict[str, Any]] = None) -> CareerCrisisResponse:
    """Create career crisis guidance response"""
    module = get_career_crisis_module()
    import asyncio
    return asyncio.run(module.process_career_crisis_query(query, context))
