"""
💰 Financial Crisis Module - Dharmic Approach to Money Challenges
Spiritual wisdom for debt, poverty, financial stress, and building
wealth through dharmic principles of abundance and right relationship with money
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FinancialCrisisType(Enum):
    """Types of financial crises"""
    DEBT_CRISIS = "debt_crisis"
    JOB_LOSS_POVERTY = "job_loss_poverty"
    BUSINESS_FAILURE = "business_failure"
    MEDICAL_EXPENSES = "medical_expenses"
    FAMILY_FINANCIAL_STRESS = "family_financial_stress"
    POVERTY_MINDSET = "poverty_mindset"
    WEALTH_GUILT = "wealth_guilt"
    FINANCIAL_PLANNING = "financial_planning"


class FinancialApproach(Enum):
    """Dharmic approaches to financial challenges"""
    ABUNDANCE_MINDSET = "abundance_mindset"    # Trusting divine provision
    RIGHT_EARNING = "right_earning"            # Ethical income sources
    CONSCIOUS_SPENDING = "conscious_spending"   # Mindful money use
    GENEROUS_GIVING = "generous_giving"        # Tithing and charity
    SIMPLE_LIVING = "simple_living"            # Reducing wants
    MONEY_MEDITATION = "money_meditation"      # Spiritual money practice


@dataclass
class FinancialGuidance:
    """Comprehensive financial crisis guidance"""
    crisis_type: FinancialCrisisType
    spiritual_perspective: str
    dharmic_approach: List[FinancialApproach]
    daily_practices: List[str]
    practical_steps: List[str]
    mindset_shifts: List[str]
    prayers_mantras: List[str]
    scriptural_wisdom: str
    abundance_affirmations: List[str]


class FinancialCrisisResponse(BaseModel):
    """Response from Financial Crisis module"""
    crisis_type: str = Field(description="Type of financial crisis")
    spiritual_perspective: str = Field(description="Dharmic money view")
    practical_guidance: List[str] = Field(description="Practical steps")
    daily_practices: List[str] = Field(description="Daily practices")
    mindset_transformation: List[str] = Field(description="Mental shifts")
    prayers_mantras: List[str] = Field(description="Money mantras")
    dharmic_wisdom: str = Field(description="Scriptural guidance")
    abundance_practices: List[str] = Field(description="Abundance building")


class FinancialCrisisModule:
    """
    💰 Financial Crisis Module - Dharmic Approach to Money Challenges
    
    Provides spiritual wisdom for:
    - Debt and financial stress management
    - Building abundance consciousness
    - Right relationship with money and wealth
    - Overcoming poverty mindset
    - Ethical wealth creation
    - Generous giving and receiving
    
    Based on dharmic principles of Artha (righteous prosperity).
    """
    
    def __init__(self):
        self.name = "Financial Crisis"
        self.color = "💰"
        self.element = "Dharmic Prosperity"
        self.principles = ["Right Earning", "Conscious Spending",
                           "Generous Giving", "Abundance Trust"]
        self.guidance_types = self._initialize_financial_guidance()
        self.prosperity_mantras = self._initialize_prosperity_mantras()
        self.abundance_principles = self._initialize_abundance_principles()
    
    def _initialize_financial_guidance(self) -> Dict[FinancialCrisisType, FinancialGuidance]:
        """Initialize guidance for different financial crises"""
        return {
            FinancialCrisisType.DEBT_CRISIS: FinancialGuidance(
                crisis_type=FinancialCrisisType.DEBT_CRISIS,
                spiritual_perspective=(
                    "Debt is often a teacher of discipline and mindful consumption. "
                    "It calls us to distinguish between needs and wants, and to "
                    "rebuild our relationship with money through conscious choices."
                ),
                dharmic_approach=[
                    FinancialApproach.CONSCIOUS_SPENDING,
                    FinancialApproach.SIMPLE_LIVING,
                    FinancialApproach.RIGHT_EARNING
                ],
                daily_practices=[
                    "Morning gratitude for what you have",
                    "Track all expenses mindfully",
                    "Before purchases: Need vs Want meditation",
                    "Evening review of money choices",
                    "Prosperity mantras and affirmations"
                ],
                practical_steps=[
                    "List all debts and create payment plan",
                    "Cut unnecessary expenses immediately",
                    "Increase income through ethical means",
                    "Negotiate with creditors honestly",
                    "Seek financial counseling if needed"
                ],
                mindset_shifts=[
                    "From victim to responsible creator",
                    "From shame to learning opportunity",
                    "From scarcity to abundance mindset",
                    "From instant gratification to patience",
                    "From money fear to money wisdom"
                ],
                prayers_mantras=[
                    "Om Shreem Hreem Klim Maha Lakshmiyei Namaha",
                    "Om Gam Ganapataye Namaha - Remove obstacles",
                    "Abundance affirmation: 'Divine provides all I need'"
                ],
                scriptural_wisdom=(
                    "Chanakya Niti: 'A person should not be too honest. "
                    "Straight trees are cut first.' Be wise with money, "
                    "neither too trusting nor too miserly."
                ),
                abundance_affirmations=[
                    "I am worthy of financial abundance",
                    "Money flows to me through right action",
                    "I use money wisely for highest good",
                    "Divine provides for all my needs",
                    "I attract prosperity through service"
                ]
            ),
            
            FinancialCrisisType.POVERTY_MINDSET: FinancialGuidance(
                crisis_type=FinancialCrisisType.POVERTY_MINDSET,
                spiritual_perspective=(
                    "Poverty consciousness is a spiritual block that prevents "
                    "us from receiving divine abundance. It often stems from "
                    "past karma or limiting beliefs about worthiness."
                ),
                dharmic_approach=[
                    FinancialApproach.ABUNDANCE_MINDSET,
                    FinancialApproach.GENEROUS_GIVING,
                    FinancialApproach.MONEY_MEDITATION
                ],
                daily_practices=[
                    "Abundance visualization meditation",
                    "Give something (time/money) daily",
                    "Practice gratitude for all possessions",
                    "Affirm worthiness of prosperity",
                    "Study lives of dharmic wealthy people"
                ],
                practical_steps=[
                    "Identify and challenge limiting beliefs",
                    "Start small savings habit",
                    "Learn money management skills",
                    "Invest in self-development",
                    "Associate with abundance-minded people"
                ],
                mindset_shifts=[
                    "From 'not enough' to 'more than enough'",
                    "From unworthiness to divine worthiness",
                    "From hoarding to flowing",
                    "From fear-based to love-based money decisions",
                    "From competition to collaboration mindset"
                ],
                prayers_mantras=[
                    "Om Hreem Shreem Klim Parameshwari Swaha",
                    "Lakshmi Gayatri Mantra for abundance",
                    "I am prosperous and blessed mantra"
                ],
                scriptural_wisdom=(
                    "Rigveda: 'May we be blessed with abundance, prosperity, "
                    "and the wisdom to use it for the welfare of all.' "
                    "Wealth is divine blessing when used rightly."
                ),
                abundance_affirmations=[
                    "The universe is abundant and so am I",
                    "I deserve prosperity and success",
                    "Money is energy that flows through me",
                    "I create value and receive abundance",
                    "My wealth blesses others and myself"
                ]
            ),
            
            FinancialCrisisType.WEALTH_GUILT: FinancialGuidance(
                crisis_type=FinancialCrisisType.WEALTH_GUILT,
                spiritual_perspective=(
                    "Wealth guilt blocks both enjoyment and generous giving. "
                    "Money is divine energy - the guilt is in how it's earned "
                    "or used, not in having it."
                ),
                dharmic_approach=[
                    FinancialApproach.GENEROUS_GIVING,
                    FinancialApproach.RIGHT_EARNING,
                    FinancialApproach.CONSCIOUS_SPENDING
                ],
                daily_practices=[
                    "Gratitude for ability to help others",
                    "Daily acts of generous giving",
                    "Money blessing and dedication ritual",
                    "Study dharmic wealth principles",
                    "Practice receiving with grace"
                ],
                practical_steps=[
                    "Set percentage for charitable giving",
                    "Use wealth for family and community welfare",
                    "Invest in ethical and sustainable ventures",
                    "Create employment for others",
                    "Support spiritual and educational causes"
                ],
                mindset_shifts=[
                    "From guilt to gratitude and responsibility",
                    "From hoarding to flowing wealth",
                    "From fear of judgment to confident giving",
                    "From personal wealth to community resource",
                    "From money as evil to money as divine tool"
                ],
                prayers_mantras=[
                    "Om Maha Lakshmiyei Namaha - Sacred wealth",
                    "May my wealth serve the highest good",
                    "I am a channel for divine abundance"
                ],
                scriptural_wisdom=(
                    "Bhagavad Gita 3.13: 'Those who eat food offered in "
                    "sacrifice are freed from sin. But those who cook only "
                    "for themselves eat sin.' Share your wealth."
                ),
                abundance_affirmations=[
                    "My wealth is a sacred trust",
                    "I use money to serve divine will",
                    "Prosperity flows through me to others",
                    "I am blessed to be a blessing",
                    "My abundance uplifts all beings"
                ]
            )
        }
    
    def _initialize_prosperity_mantras(self) -> Dict[str, List[str]]:
        """Initialize mantras for prosperity and abundance"""
        return {
            "wealth_attraction": [
                "Om Shreem Hreem Klim Maha Lakshmiyei Namaha",
                "Om Hreem Shreem Klim Parameshwari Swaha",
                "Om Gam Ganapataye Namaha"
            ],
            "debt_removal": [
                "Om Gam Ganapataye Namaha",
                "Om Namo Narayanaya",
                "Hanuman Chalisa for strength"
            ],
            "abundance_mindset": [
                "Gayatri Mantra for divine wisdom",
                "Om Mani Padme Hum for compassion",
                "So Hum - I am abundant consciousness"
            ]
        }
    
    def _initialize_abundance_principles(self) -> List[str]:
        """Initialize dharmic abundance principles"""
        return [
            "Earn through righteous means only",
            "Give generously from all income",
            "Spend consciously on needs first",
            "Save and invest wisely for future",
            "Use wealth to serve others",
            "Maintain gratitude for all provision",
            "Trust in divine abundance"
        ]
    
    def assess_crisis_type(self, query: str, context: Dict[str, Any]) -> FinancialCrisisType:
        """Assess the type of financial crisis from query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["debt", "owe", "credit", "loan"]):
            return FinancialCrisisType.DEBT_CRISIS
        elif any(word in query_lower for word in ["poor", "poverty", "broke", "can't afford"]):
            return FinancialCrisisType.POVERTY_MINDSET
        elif any(word in query_lower for word in ["guilt", "wealthy", "rich", "too much"]):
            return FinancialCrisisType.WEALTH_GUILT
        elif any(word in query_lower for word in ["business", "failed", "bankruptcy"]):
            return FinancialCrisisType.BUSINESS_FAILURE
        elif any(word in query_lower for word in ["medical", "hospital", "health cost"]):
            return FinancialCrisisType.MEDICAL_EXPENSES
        else:
            return FinancialCrisisType.FINANCIAL_PLANNING
    
    async def process_financial_crisis_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> FinancialCrisisResponse:
        """Process financial crisis query and provide dharmic guidance"""
        try:
            context = user_context or {}
            
            # Assess crisis type
            crisis_type = self.assess_crisis_type(query, context)
            
            # Get guidance
            guidance = self.guidance_types.get(crisis_type)
            if not guidance:
                return self._create_fallback_response()
            
            return FinancialCrisisResponse(
                crisis_type=crisis_type.value,
                spiritual_perspective=guidance.spiritual_perspective,
                practical_guidance=guidance.practical_steps,
                daily_practices=guidance.daily_practices,
                mindset_transformation=guidance.mindset_shifts,
                prayers_mantras=guidance.prayers_mantras,
                dharmic_wisdom=guidance.scriptural_wisdom,
                abundance_practices=guidance.abundance_affirmations
            )
            
        except Exception as e:
            logger.error(f"Error processing financial crisis query: {e}")
            return self._create_fallback_response()
    
    def _create_fallback_response(self) -> FinancialCrisisResponse:
        """Create fallback response when processing fails"""
        return FinancialCrisisResponse(
            crisis_type="general_financial_support",
            spiritual_perspective="Money is divine energy to be used wisely for service and growth",
            practical_guidance=["Create budget and stick to it", "Distinguish needs from wants", "Seek multiple income sources"],
            daily_practices=["Morning gratitude for provisions", "Mindful spending decisions", "Evening money reflection"],
            mindset_transformation=["From scarcity to abundance", "From fear to trust", "From hoarding to flowing"],
            prayers_mantras=["Om Shreem Maha Lakshmiyei Namaha", "Om Gam Ganapataye Namaha", "Gayatri Mantra"],
            dharmic_wisdom="Right earning and generous giving create sustainable prosperity",
            abundance_practices=["Practice gratitude daily", "Give something each day", "Visualize abundance"]
        )


# Global instance
_financial_crisis_module = None

def get_financial_crisis_module() -> FinancialCrisisModule:
    """Get global Financial Crisis module instance"""
    global _financial_crisis_module
    if _financial_crisis_module is None:
        _financial_crisis_module = FinancialCrisisModule()
    return _financial_crisis_module

# Factory function for easy access
def create_financial_crisis_guidance(query: str, context: Optional[Dict[str, Any]] = None) -> FinancialCrisisResponse:
    """Create financial crisis guidance response"""
    module = get_financial_crisis_module()
    import asyncio
    return asyncio.run(module.process_financial_crisis_query(query, context))
