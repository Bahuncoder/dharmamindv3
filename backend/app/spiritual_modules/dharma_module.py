"""
🔵 Dharma Module - Righteous Path Module
Life purpose, moral decisions, duty
Based on Bhagavad Gita, Manusmriti, Mahabharata
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class DharmaType(Enum):
    """Types of dharma"""
    SVADHARMA = "svadharma"  # Personal dharma
    YUGADHARMA = "yugadharma"  # Age-appropriate dharma
    APADHARMA = "apadharma"  # Emergency dharma
    RAJADHARMA = "rajadharma"  # Leadership dharma
    KULLADHARMA = "kulladharma"  # Family dharma

class LifeStage(Enum):
    """Life stages (Ashramas)"""
    BRAHMACHARYA = "brahmacharya"  # Student
    GRIHASTHA = "grihastha"  # Householder
    VANAPRASTHA = "vanaprastha"  # Forest dweller
    SANNYASA = "sannyasa"  # Renunciant

@dataclass
class DharmaGuidance:
    """Dharmic guidance structure"""
    dharma_type: DharmaType
    life_stage: LifeStage
    primary_duty: str
    moral_principle: str
    gita_reference: str
    practical_action: str
    potential_conflict: str
    resolution_wisdom: str

class DharmaResponse(BaseModel):
    """Response from Dharma module"""
    dharma_type: str = Field(description="Type of dharma applicable")
    life_stage: str = Field(description="Current life stage")
    primary_duties: List[str] = Field(description="Primary dharmic duties")
    moral_guidance: str = Field(description="Moral and ethical guidance")
    gita_wisdom: str = Field(description="Relevant Bhagavad Gita teaching")
    practical_steps: List[str] = Field(description="Practical steps to follow dharma")
    potential_conflicts: List[str] = Field(description="Potential dharmic conflicts")
    conflict_resolution: str = Field(description="How to resolve dharmic conflicts")
    daily_dharma_practice: List[str] = Field(description="Daily dharmic practices")
    
class DharmaModule:
    """
    🔵 Dharma Module - Righteous Path Module
    
    Guides users in understanding and following their dharma (righteous duty)
    based on Bhagavad Gita, Manusmriti, and Mahabharata teachings.
    """
    
    def __init__(self):
        self.name = "Dharma"
        self.color = "🔵"
        self.element = "righteousness"
        self.dharma_database = self._initialize_dharma_teachings()
        
    def _initialize_dharma_teachings(self) -> Dict[tuple, DharmaGuidance]:
        """Initialize dharma teachings database"""
        teachings = {}
        
        # Brahmacharya (Student) stage teachings
        teachings[(DharmaType.SVADHARMA, LifeStage.BRAHMACHARYA)] = DharmaGuidance(
            dharma_type=DharmaType.SVADHARMA,
            life_stage=LifeStage.BRAHMACHARYA,
            primary_duty="Acquire knowledge, practice discipline, serve the guru",
            moral_principle="Purity in thought, word, and deed",
            gita_reference="BG 4.34: Approach a spiritual master with questions and service",
            practical_action="Focus on learning, avoid distractions, practice celibacy",
            potential_conflict="Desire for pleasure vs. study obligations",
            resolution_wisdom="Temporary sacrifice leads to permanent wisdom"
        )
        
        # Grihastha (Householder) stage teachings
        teachings[(DharmaType.SVADHARMA, LifeStage.GRIHASTHA)] = DharmaGuidance(
            dharma_type=DharmaType.SVADHARMA,
            life_stage=LifeStage.GRIHASTHA,
            primary_duty="Support family, earn righteously, serve society",
            moral_principle="Balance material and spiritual duties",
            gita_reference="BG 3.20: King Janaka achieved perfection through action",
            practical_action="Work with dedication, provide for dependents, practice charity",
            potential_conflict="Family needs vs. spiritual aspirations",
            resolution_wisdom="See family service as worship of the Divine"
        )
        
        # Leadership dharma
        teachings[(DharmaType.RAJADHARMA, LifeStage.GRIHASTHA)] = DharmaGuidance(
            dharma_type=DharmaType.RAJADHARMA,
            life_stage=LifeStage.GRIHASTHA,
            primary_duty="Protect the righteous, punish the wicked, maintain order",
            moral_principle="Justice with compassion",
            gita_reference="BG 4.8: I appear to protect the righteous and destroy evil",
            practical_action="Lead by example, make fair decisions, protect the weak",
            potential_conflict="Personal interest vs. public good",
            resolution_wisdom="A leader's personal desires must serve the greater good"
        )
        
        return teachings
        
    def assess_life_stage(self, user_context: Dict[str, Any]) -> LifeStage:
        """Assess user's current life stage"""
        age = user_context.get('age', 25)
        family_status = user_context.get('family_status', 'single')
        occupation = user_context.get('occupation', 'student')
        
        if age < 25 or occupation == 'student':
            return LifeStage.BRAHMACHARYA
        elif age < 50 and family_status in ['married', 'family']:
            return LifeStage.GRIHASTHA
        elif age < 70:
            return LifeStage.VANAPRASTHA
        else:
            return LifeStage.SANNYASA
            
    def determine_dharma_type(self, query: str, context: Dict[str, Any]) -> DharmaType:
        """Determine appropriate dharma type based on situation"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["leader", "responsibility", "authority"]):
            return DharmaType.RAJADHARMA
        elif any(word in query_lower for word in ["family", "parents", "children"]):
            return DharmaType.KULLADHARMA
        elif any(word in query_lower for word in ["emergency", "crisis", "difficult"]):
            return DharmaType.APADHARMA
        elif any(word in query_lower for word in ["modern", "current", "today"]):
            return DharmaType.YUGADHARMA
        else:
            return DharmaType.SVADHARMA
            
    def get_dharmic_duties(self, dharma_type: DharmaType, life_stage: LifeStage) -> List[str]:
        """Get dharmic duties based on type and life stage"""
        duties_map = {
            (DharmaType.SVADHARMA, LifeStage.BRAHMACHARYA): [
                "Pursue knowledge with dedication",
                "Practice self-discipline and celibacy",
                "Serve teachers and elders",
                "Develop good character"
            ],
            (DharmaType.SVADHARMA, LifeStage.GRIHASTHA): [
                "Provide for family responsibly",
                "Earn through righteous means",
                "Practice charity and hospitality",
                "Raise children with good values"
            ],
            (DharmaType.RAJADHARMA, LifeStage.GRIHASTHA): [
                "Protect the innocent and weak",
                "Maintain justice and order",
                "Lead by moral example",
                "Make decisions for collective good"
            ],
            (DharmaType.KULLADHARMA, LifeStage.GRIHASTHA): [
                "Honor parents and elders",
                "Maintain family traditions",
                "Support family members",
                "Preserve family honor"
            ]
        }
        
        return duties_map.get(
            (dharma_type, life_stage),
            ["Follow truth and righteousness", "Serve others selflessly", "Practice non-violence", "Maintain purity"]
        )
        
    def get_gita_wisdom(self, dharma_type: DharmaType) -> str:
        """Get relevant Bhagavad Gita wisdom"""
        wisdom_map = {
            DharmaType.SVADHARMA: "BG 3.35: Better is one's own dharma, though imperfectly performed, than the dharma of another well performed",
            DharmaType.RAJADHARMA: "BG 4.8: I incarnate to protect the righteous, destroy the wicked, and establish dharma",
            DharmaType.KULLADHARMA: "BG 7.11: I am the strength of the strong, devoid of passion and desire",
            DharmaType.APADHARMA: "BG 2.31: Considering your dharma, you should not waver, for there is nothing higher for a warrior than a righteous war",
            DharmaType.YUGADHARMA: "BG 3.21: Whatever a great person does, others follow; whatever standards they set, the world pursues"
        }
        return wisdom_map.get(dharma_type, "BG 18.66: Abandon all dharmas and surrender unto Me; I shall deliver you from all sins")
        
    def identify_dharmic_conflicts(self, query: str) -> List[str]:
        """Identify potential dharmic conflicts"""
        conflicts = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["family", "work", "duty"]):
            conflicts.append("Conflict between family duty and professional obligations")
            
        if any(word in query_lower for word in ["truth", "lie", "honest"]):
            conflicts.append("Tension between absolute truth and compassionate speech")
            
        if any(word in query_lower for word in ["money", "wealth", "profit"]):
            conflicts.append("Balancing material needs with spiritual values")
            
        if any(word in query_lower for word in ["friend", "loyalty", "relationship"]):
            conflicts.append("Personal loyalty vs. moral righteousness")
            
        return conflicts if conflicts else ["Balancing personal desires with dharmic duty"]
        
    def resolve_dharmic_conflict(self, conflicts: List[str]) -> str:
        """Provide wisdom for resolving dharmic conflicts"""
        if "family duty" in conflicts[0]:
            return "Krishna's teaching: Perform all duties as worship to the Divine. Both family and work can be forms of service when done with the right attitude."
        elif "truth" in conflicts[0]:
            return "Mahabharata wisdom: Speak truth that is pleasant; do not speak unpleasant truth. This is the eternal dharma."
        elif "material" in conflicts[0]:
            return "Gita wisdom: One who is not disturbed by incessant flow of desires that enter like rivers into the ocean achieves peace."
        else:
            return "When in doubt, choose the path that causes least harm and serves the greater good. Consult wise elders and scriptures."
            
    async def process_dharma_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> DharmaResponse:
        """Process dharma query and provide guidance"""
        try:
            if user_context is None:
                user_context = {}
                
            # Assess current situation
            life_stage = self.assess_life_stage(user_context)
            dharma_type = self.determine_dharma_type(query, user_context)
            
            # Get dharmic guidance
            duties = self.get_dharmic_duties(dharma_type, life_stage)
            gita_wisdom = self.get_gita_wisdom(dharma_type)
            conflicts = self.identify_dharmic_conflicts(query)
            resolution = self.resolve_dharmic_conflict(conflicts)
            
            # Generate practical steps
            practical_steps = self._generate_practical_steps(dharma_type, life_stage)
            
            # Daily dharma practices
            daily_practices = [
                "Morning prayer or meditation",
                "Act according to your dharma without attachment to results",
                "Practice truthfulness in speech",
                "Serve others according to your capacity",
                "Study sacred texts for guidance"
            ]
            
            # Moral guidance
            moral_guidance = self._generate_moral_guidance(dharma_type, query)
            
            return DharmaResponse(
                dharma_type=dharma_type.value,
                life_stage=life_stage.value,
                primary_duties=duties,
                moral_guidance=moral_guidance,
                gita_wisdom=gita_wisdom,
                practical_steps=practical_steps,
                potential_conflicts=conflicts,
                conflict_resolution=resolution,
                daily_dharma_practice=daily_practices
            )
            
        except Exception as e:
            logger.error(f"Error processing dharma query: {e}")
            return self._create_fallback_response()
            
    def _generate_practical_steps(self, dharma_type: DharmaType, life_stage: LifeStage) -> List[str]:
        """Generate practical steps for following dharma"""
        if dharma_type == DharmaType.SVADHARMA and life_stage == LifeStage.BRAHMACHARYA:
            return [
                "Create a study schedule and stick to it",
                "Practice daily meditation or prayer",
                "Avoid unnecessary distractions",
                "Serve your teachers with humility",
                "Develop good habits and character"
            ]
        elif dharma_type == DharmaType.SVADHARMA and life_stage == LifeStage.GRIHASTHA:
            return [
                "Work diligently in your profession",
                "Provide for your family's needs",
                "Practice charity within your means",
                "Spend quality time with family",
                "Balance material and spiritual pursuits"
            ]
        elif dharma_type == DharmaType.RAJADHARMA:
            return [
                "Make decisions based on collective good",
                "Lead by moral example",
                "Listen to advice from wise counselors",
                "Protect the weak and innocent",
                "Maintain justice without favoritism"
            ]
        else:
            return [
                "Follow your conscience and inner guidance",
                "Consult sacred texts and wise people",
                "Act without attachment to results",
                "Practice non-violence in thought and deed",
                "Serve according to your capacity"
            ]
            
    def _generate_moral_guidance(self, dharma_type: DharmaType, query: str) -> str:
        """Generate moral guidance based on context"""
        base_guidance = "Follow the path of righteousness (dharma) as taught in the scriptures. "
        
        if "decision" in query.lower():
            return base_guidance + "When making decisions, consider: Is it truthful? Is it beneficial? Does it harm anyone? Will I be proud of this choice?"
        elif "conflict" in query.lower():
            return base_guidance + "In conflicts, seek the middle path that honors both justice and compassion. Sometimes the right choice requires sacrifice."
        elif "work" in query.lower():
            return base_guidance + "In work, perform your duties with excellence but without attachment to results. See your work as service to the Divine."
        else:
            return base_guidance + "Let truth, non-violence, purity, and self-control guide all your actions."
            
    def _create_fallback_response(self) -> DharmaResponse:
        """Create fallback response when processing fails"""
        return DharmaResponse(
            dharma_type="svadharma",
            life_stage="grihastha",
            primary_duties=[
                "Speak truth",
                "Practice non-violence",
                "Serve others",
                "Follow righteousness"
            ],
            moral_guidance="Follow your dharma according to your nature and circumstances, as taught in the Bhagavad Gita",
            gita_wisdom="BG 3.35: Better is one's own dharma, though imperfectly performed, than the dharma of another well performed",
            practical_steps=[
                "Act according to your conscience",
                "Consult wise elders",
                "Study sacred texts",
                "Practice self-reflection"
            ],
            potential_conflicts=["Balancing personal desires with duty"],
            conflict_resolution="Choose the path that serves the greater good and causes least harm",
            daily_dharma_practice=[
                "Morning meditation",
                "Truthful speech",
                "Selfless service",
                "Evening reflection"
            ]
        )

# Global instance
_dharma_module = None

def get_dharma_module() -> DharmaModule:
    """Get global Dharma module instance"""
    global _dharma_module
    if _dharma_module is None:
        _dharma_module = DharmaModule()
    return _dharma_module
