"""
🏠 Grihastha Module - The Sacred Householder Path
Complete system for balancing spiritual growth with family and worldly responsibilities
Based on traditional Grihastha dharma and modern family life integration
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class HouseholderStage(Enum):
    """Stages within householder life"""
    YOUNG_ADULT = "young_adult"          # Starting career and relationships
    MARRIED_EARLY = "married_early"      # Early marriage, establishing home
    FAMILY_BUILDING = "family_building"  # Having and raising young children
    FAMILY_MATURE = "family_mature"      # Children growing, career advancing
    EMPTY_NEST = "empty_nest"           # Children independent, pre-retirement
    ELDER = "elder"                     # Retirement, grandparent phase


class LifeArea(Enum):
    """Areas of householder life"""
    MARRIAGE = "marriage"               # Spousal relationship
    PARENTING = "parenting"            # Raising children
    CAREER = "career"                  # Work and profession
    FINANCES = "finances"              # Money management
    HEALTH = "health"                  # Physical and mental wellbeing
    SPIRITUALITY = "spirituality"      # Spiritual practice and growth
    COMMUNITY = "community"            # Social and community involvement
    SERVICE = "service"                # Dharmic service and contribution


class HouseholderChallenge(Enum):
    """Common challenges in householder life"""
    WORK_LIFE_BALANCE = "work_life_balance"
    FINANCIAL_STRESS = "financial_stress"
    RELATIONSHIP_CONFLICT = "relationship_conflict"
    PARENTING_GUIDANCE = "parenting_guidance"
    TIME_MANAGEMENT = "time_management"
    SPIRITUAL_INTEGRATION = "spiritual_integration"
    SOCIAL_OBLIGATIONS = "social_obligations"
    AGING_PARENTS = "aging_parents"


class RelationshipType(Enum):
    """Types of relationships in householder life"""
    SPOUSE = "spouse"                   # Marriage relationship
    PARENT_CHILD = "parent_child"       # Parenting relationship
    FAMILY_EXTENDED = "family_extended" # Extended family relationships
    PROFESSIONAL = "professional"       # Work relationships
    COMMUNITY = "community"            # Community and social relationships


@dataclass
class GrihasthaGuidance:
    """Comprehensive householder guidance"""
    stage: HouseholderStage
    primary_dharma: List[str]
    daily_practices: List[str]
    spiritual_integration: List[str]
    relationship_wisdom: Dict[str, str]
    challenge_solutions: Dict[str, str]
    balance_strategies: List[str]
    service_opportunities: List[str]
    practical_applications: List[str]
    progress_indicators: List[str]


@dataclass
class HouseholderPractice:
    """Specific practice for householder life"""
    name: str
    area: LifeArea
    description: str
    steps: List[str]
    benefits: List[str]
    frequency: str
    integration_tips: List[str]


class GrihasthaResponse(BaseModel):
    """Response from Grihastha module"""
    householder_stage: str = Field(description="Current stage of householder life")
    primary_dharma: List[str] = Field(description="Primary duties and responsibilities")
    spiritual_integration: List[str] = Field(description="Ways to integrate spirituality")
    relationship_guidance: Dict[str, str] = Field(description="Relationship wisdom")
    daily_practices: List[str] = Field(description="Daily householder practices")
    challenge_solutions: Dict[str, str] = Field(description="Solutions for life challenges")
    balance_strategies: List[str] = Field(description="Strategies for work-life balance")
    service_opportunities: List[str] = Field(description="Ways to serve through family life")
    practical_applications: List[str] = Field(description="Practical ways to apply teachings")
    wisdom_guidance: str = Field(description="Core wisdom for householder path")


class GrihasthaModule:
    """
    🏠 Grihastha Module - The Sacred Householder Path
    
    Based on traditional Grihastha dharma:
    - Ashramas (life stages) teachings from Dharma Shastras
    - Mahabharata and Ramayana examples of ideal householders
    - Bhagavad Gita teachings on action in the world
    - Modern integration of ancient wisdom
    
    The householder path teaches that family life itself is spiritual practice
    when approached with dharma, service, and consciousness.
    """
    
    def __init__(self):
        self.name = "Grihastha"
        self.color = "🏠"
        self.element = "Integration"
        self.principles = ["Dharmic Living", "Family Service", "Balance", "Community"]
        self.guidance_levels = self._initialize_guidance_levels()
        self.householder_practices = self._initialize_householder_practices()
        self.challenge_solutions = self._initialize_challenge_solutions()
        self.relationship_wisdom = self._initialize_relationship_wisdom()
        
    def _initialize_guidance_levels(self) -> Dict[HouseholderStage, GrihasthaGuidance]:
        """Initialize guidance for different householder stages"""
        return {
            HouseholderStage.YOUNG_ADULT: GrihasthaGuidance(
                stage=HouseholderStage.YOUNG_ADULT,
                primary_dharma=[
                    "Establish financial independence and career foundation",
                    "Develop character and life skills through education and experience",
                    "Build healthy relationships and find suitable life partner",
                    "Create strong spiritual foundation for future family life",
                    "Practice service to community and society"
                ],
                daily_practices=[
                    "Morning intention setting for dharmic goals",
                    "Work approached as spiritual practice and service",
                    "Evening reflection on character development",
                    "Study of spiritual texts and wisdom traditions",
                    "Regular service to others and community involvement"
                ],
                spiritual_integration=[
                    "Establish regular meditation or prayer practice",
                    "Study spiritual texts relevant to life stage",
                    "Find spiritual community and mentors",
                    "Practice ethical living in all relationships",
                    "Use challenges as opportunities for growth"
                ],
                relationship_wisdom={
                    "dating": "Seek partner who shares your values and supports your growth",
                    "friendship": "Cultivate friendships based on mutual respect and shared dharma",
                    "professional": "Maintain integrity and kindness in all work relationships",
                    "family": "Honor parents while establishing your own path"
                },
                challenge_solutions={
                    "career_uncertainty": "Focus on developing skills and character, trust the process",
                    "relationship_confusion": "Look for dharmic compatibility, not just attraction",
                    "financial_pressure": "Live within means, save regularly, practice gratitude"
                },
                balance_strategies=[
                    "Set clear priorities based on dharmic principles",
                    "Create boundaries between work, relationships, and personal time",
                    "Practice saying no to activities that don't align with values",
                    "Include spiritual practice in daily routine"
                ],
                service_opportunities=[
                    "Volunteer with organizations serving youth or education",
                    "Mentor younger people in your field or community",
                    "Support elderly family members and neighbors",
                    "Contribute skills to charitable causes"
                ],
                practical_applications=[
                    "Use work as opportunity to practice patience and service",
                    "Apply spiritual principles in dating and relationship choices",
                    "Practice gratitude and contentment with current circumstances",
                    "Build savings and financial discipline for future family"
                ],
                progress_indicators=[
                    "Growing sense of purpose and direction in life",
                    "Ability to maintain spiritual practice amid busy schedule",
                    "Healthy relationships based on mutual respect",
                    "Financial responsibility and independence"
                ]
            ),
            
            HouseholderStage.MARRIED_EARLY: GrihasthaGuidance(
                stage=HouseholderStage.MARRIED_EARLY,
                primary_dharma=[
                    "Build strong marital foundation based on love and dharma",
                    "Establish household routines and sacred traditions",
                    "Balance individual growth with couple's development",
                    "Prepare spiritually and practically for potential parenthood",
                    "Create home environment that supports spiritual growth"
                ],
                daily_practices=[
                    "Morning gratitude practice with spouse",
                    "Daily appreciation and kind words to partner",
                    "Shared spiritual practice or discussion",
                    "Mindful communication and active listening",
                    "Evening reflection on day's relationship dynamics"
                ],
                spiritual_integration=[
                    "Create sacred space and rituals in home",
                    "Support each other's individual spiritual practices",
                    "Study spiritual texts together as couple",
                    "Practice seeing spouse as divine manifestation",
                    "Include service projects as couple activity"
                ],
                relationship_wisdom={
                    "marriage": "Approach marriage as spiritual partnership for mutual growth",
                    "in_laws": "Honor extended family while maintaining couple boundaries",
                    "friends": "Nurture friendships that support your marriage",
                    "neighbors": "Be good neighbors and community members"
                },
                challenge_solutions={
                    "adjustment_difficulties": "Practice patience, communicate openly, seek guidance when needed",
                    "financial_planning": "Create joint budget, discuss money values, plan for future",
                    "career_demands": "Support each other's professional growth while prioritizing relationship"
                },
                balance_strategies=[
                    "Schedule regular one-on-one time without distractions",
                    "Balance time with friends/family and couple time",
                    "Create boundaries around work to protect relationship time",
                    "Practice gratitude for spouse daily"
                ],
                service_opportunities=[
                    "Serve together at community organizations",
                    "Host gatherings that build community",
                    "Support other couples in your community",
                    "Contribute to causes important to both partners"
                ],
                practical_applications=[
                    "Practice patience and forgiveness in daily interactions",
                    "Use household chores as opportunities for mindfulness",
                    "Apply conflict resolution skills with love and understanding",
                    "Create meaningful traditions and celebrations"
                ],
                progress_indicators=[
                    "Deepening intimacy and understanding with spouse",
                    "Successful creation of shared spiritual practices",
                    "Growing ability to navigate conflicts constructively",
                    "Increased sense of home as sacred space"
                ]
            ),
            
            HouseholderStage.FAMILY_BUILDING: GrihasthaGuidance(
                stage=HouseholderStage.FAMILY_BUILDING,
                primary_dharma=[
                    "Raise children with love, values, and spiritual foundation",
                    "Provide physical, emotional, and spiritual security for family",
                    "Maintain marriage relationship while focusing on parenting",
                    "Balance career advancement with family responsibilities",
                    "Model dharmic living for children through daily example"
                ],
                daily_practices=[
                    "Morning family prayer or gratitude circle",
                    "Mindful presence during all child interactions",
                    "Teaching moments woven throughout daily activities",
                    "Evening family time without electronic distractions",
                    "Bedtime blessings and loving words to children"
                ],
                spiritual_integration=[
                    "Include children in age-appropriate spiritual activities",
                    "Tell stories from spiritual traditions during family time",
                    "Practice patience and compassion as spiritual discipline",
                    "See parenting challenges as opportunities for growth",
                    "Create family service projects and community involvement"
                ],
                relationship_wisdom={
                    "parenting": "Discipline with love, teach through example, honor each child's nature",
                    "marriage": "Maintain couple connection despite parenting demands",
                    "extended_family": "Welcome grandparents' involvement while maintaining authority",
                    "community": "Build network of families with shared values"
                },
                challenge_solutions={
                    "overwhelm": "Ask for help, prioritize essential tasks, practice self-compassion",
                    "work_life_balance": "Set clear boundaries, focus on presence over perfection",
                    "financial_strain": "Budget carefully, focus on needs vs. wants, practice gratitude"
                },
                balance_strategies=[
                    "Alternate parenting duties to give each parent breaks",
                    "Schedule regular date nights to maintain marriage",
                    "Practice presence - be fully where you are",
                    "Let go of perfectionism in favor of love and connection"
                ],
                service_opportunities=[
                    "Volunteer at children's schools and activities",
                    "Host playdates and community gatherings",
                    "Support other families with young children",
                    "Teach children to help others through family service projects"
                ],
                practical_applications=[
                    "Use daily routines as opportunities for teaching values",
                    "Practice emotional regulation to model for children",
                    "Create family traditions that reinforce spiritual values",
                    "Handle parenting stress with breathing and mindfulness"
                ],
                progress_indicators=[
                    "Children showing kindness, respect, and spiritual awareness",
                    "Maintaining loving marriage despite parenting challenges",
                    "Growing wisdom and patience in handling difficulties",
                    "Family functioning as harmonious, loving unit"
                ]
            ),
            
            HouseholderStage.FAMILY_MATURE: GrihasthaGuidance(
                stage=HouseholderStage.FAMILY_MATURE,
                primary_dharma=[
                    "Guide adolescent children through identity development with wisdom",
                    "Advance in career while maintaining ethical principles",
                    "Increase community service and leadership responsibilities",
                    "Care for aging parents with love and respect",
                    "Prepare for eventual transition to empty nest phase"
                ],
                daily_practices=[
                    "Deep listening and guidance for adolescent children",
                    "Professional work approached with integrity and service",
                    "Regular study of wisdom traditions for deeper understanding",
                    "Community involvement and mentoring of younger people",
                    "Care and communication with aging parents"
                ],
                spiritual_integration=[
                    "Deepen personal spiritual practice with increased stability",
                    "Share wisdom and experience with younger generation",
                    "Practice letting go as children become more independent",
                    "Find meaning through service to community and society",
                    "Prepare spiritually for next phase of life"
                ],
                relationship_wisdom={
                    "adolescents": "Guide with wisdom while allowing independence and learning",
                    "marriage": "Rediscover partnership as parenting demands decrease",
                    "aging_parents": "Honor and care for elders while managing other responsibilities",
                    "community": "Use your experience to mentor and lead others"
                },
                challenge_solutions={
                    "teen_rebellion": "Stay connected with love while maintaining boundaries",
                    "career_pressure": "Remember that integrity matters more than advancement",
                    "sandwich_generation": "Seek support and practice self-care while caring for others"
                },
                balance_strategies=[
                    "Delegate responsibilities to prepare teens for independence",
                    "Maintain marriage focus as children need less direct attention",
                    "Create support systems for eldercare responsibilities",
                    "Practice wisdom in choosing commitments and obligations"
                ],
                service_opportunities=[
                    "Mentor young professionals and parents",
                    "Lead community organizations and initiatives",
                    "Support elderly community members",
                    "Share skills and experience through teaching or volunteering"
                ],
                practical_applications=[
                    "Use life experience to guide others with compassion",
                    "Apply spiritual principles in professional leadership",
                    "Practice detachment while remaining lovingly engaged",
                    "Create legacy through positive influence on others"
                ],
                progress_indicators=[
                    "Adolescent children developing into responsible young adults",
                    "Recognition as wise leader in community or profession",
                    "Successful balance of multiple life responsibilities",
                    "Deepening spiritual understanding and practice"
                ]
            ),
            
            HouseholderStage.EMPTY_NEST: GrihasthaGuidance(
                stage=HouseholderStage.EMPTY_NEST,
                primary_dharma=[
                    "Support adult children's independence while maintaining connection",
                    "Rediscover marriage relationship as couple rather than co-parents",
                    "Increase focus on spiritual development and wisdom cultivation",
                    "Share experience and wisdom through mentoring and teaching",
                    "Prepare practically and spiritually for retirement years"
                ],
                daily_practices=[
                    "Deepened meditation or contemplative practice",
                    "Regular study of spiritual and wisdom literature",
                    "Mentoring and guidance of younger people",
                    "Quality time with spouse rediscovering relationship",
                    "Service to community through wisdom and experience"
                ],
                spiritual_integration=[
                    "Focus on contemplative and devotional practices",
                    "Share spiritual wisdom through teaching or writing",
                    "Practice letting go of need to control outcomes",
                    "Find meaning through service rather than achievement",
                    "Prepare for eventual transition beyond householder stage"
                ],
                relationship_wisdom={
                    "adult_children": "Offer wisdom when asked, support without interfering",
                    "marriage": "Rediscover each other as individuals, not just parents",
                    "grandchildren": "Enjoy grandparent role without competing with parents",
                    "community": "Serve as elder and wisdom keeper"
                },
                challenge_solutions={
                    "identity_shift": "Find new purpose in wisdom sharing and deeper service",
                    "relationship_changes": "Invest in rediscovering spouse and rebuilding intimacy",
                    "health_concerns": "Accept aging gracefully while maintaining vitality"
                },
                balance_strategies=[
                    "Balance involvement with adult children with respect for their independence",
                    "Create new routines and rituals as couple",
                    "Practice letting go while remaining available when needed",
                    "Focus on quality rather than quantity in relationships"
                ],
                service_opportunities=[
                    "Mentor young families and professionals",
                    "Volunteer with organizations serving youth or elderly",
                    "Share skills through teaching or consulting",
                    "Support community initiatives with wisdom and experience"
                ],
                practical_applications=[
                    "Use freedom from daily parenting to deepen spiritual practice",
                    "Travel and experience life from spiritual perspective",
                    "Write or teach to share life wisdom with others",
                    "Practice gratitude for successful completion of active parenting"
                ],
                progress_indicators=[
                    "Successful transition to supportive rather than active parenting",
                    "Renewed intimacy and connection with spouse",
                    "Recognition as wise elder in community",
                    "Peace and fulfillment in current life phase"
                ]
            ),
            
            HouseholderStage.ELDER: GrihasthaGuidance(
                stage=HouseholderStage.ELDER,
                primary_dharma=[
                    "Enjoy and guide grandchildren with wisdom and love",
                    "Share life wisdom with younger generations",
                    "Maintain health and independence as long as possible",
                    "Prepare spiritually for transition beyond physical life",
                    "Leave positive legacy through example and influence"
                ],
                daily_practices=[
                    "Extended contemplative and devotional practice",
                    "Sharing wisdom and stories with family and community",
                    "Gratitude practice for life's blessings and lessons",
                    "Service through presence, wisdom, and blessing others",
                    "Preparation for eventual transition through spiritual practice"
                ],
                spiritual_integration=[
                    "Life becomes primarily spiritual practice and service",
                    "Focus on what truly matters: love, wisdom, service",
                    "Practice letting go of material attachments",
                    "Prepare for eventual departure from physical form",
                    "Radiate peace and wisdom through simple presence"
                ],
                relationship_wisdom={
                    "grandchildren": "Offer unconditional love and wisdom without interfering",
                    "adult_children": "Bless their lives while accepting their choices",
                    "spouse": "Cherish remaining time together with deep appreciation",
                    "community": "Serve as repository of wisdom and blessing"
                },
                challenge_solutions={
                    "health_limitations": "Accept changes gracefully while maintaining dignity",
                    "loss_of_friends": "Practice acceptance while treasuring remaining relationships",
                    "purpose_questions": "Find meaning in wisdom sharing and spiritual preparation"
                },
                balance_strategies=[
                    "Balance activity with rest according to energy levels",
                    "Focus on essential relationships and activities",
                    "Practice acceptance of life's natural cycles",
                    "Maintain hope and gratitude despite challenges"
                ],
                service_opportunities=[
                    "Serve as wisdom keeper and storyteller for family",
                    "Mentor and bless younger people through your example",
                    "Support community through presence and prayers",
                    "Leave written or recorded wisdom for future generations"
                ],
                practical_applications=[
                    "Use life experience to offer perspective on problems",
                    "Practice patience and acceptance in face of limitations",
                    "Share family history and values with descendants",
                    "Approach death preparation as final spiritual practice"
                ],
                progress_indicators=[
                    "Peace and acceptance of life's natural progression",
                    "Recognition as wise elder and blessing to others",
                    "Successful completion of householder dharma",
                    "Readiness for eventual transition beyond physical form"
                ]
            )
        }
    
    def _initialize_householder_practices(self) -> Dict[LifeArea, List[HouseholderPractice]]:
        """Initialize practices for different areas of householder life"""
        return {
            LifeArea.MARRIAGE: [
                HouseholderPractice(
                    name="Daily Appreciation Practice",
                    area=LifeArea.MARRIAGE,
                    description="Expressing gratitude and appreciation for your spouse daily",
                    steps=[
                        "Each morning, silently appreciate your spouse's qualities",
                        "Express one specific appreciation during the day",
                        "Share gratitude for something they did recently",
                        "Practice forgiveness for daily irritations",
                        "End day with loving-kindness meditation for spouse"
                    ],
                    benefits=[
                        "Strengthened emotional bond",
                        "Increased marital satisfaction",
                        "Reduced conflict and tension",
                        "Greater awareness of spouse's contributions"
                    ],
                    frequency="Daily",
                    integration_tips=[
                        "Include appreciation in morning routine",
                        "Use meal times for gratitude sharing",
                        "Write appreciation notes occasionally",
                        "Practice even during difficult periods"
                    ]
                ),
                HouseholderPractice(
                    name="Sacred Communication",
                    area=LifeArea.MARRIAGE,
                    description="Dharmic principles for healthy marital communication",
                    steps=[
                        "Listen with full attention and presence",
                        "Speak truth with kindness and timing",
                        "Address issues promptly and constructively",
                        "Use 'I' statements to express feelings",
                        "Seek understanding before seeking to be understood"
                    ],
                    benefits=[
                        "Deeper intimacy and understanding",
                        "Reduced misunderstandings",
                        "Faster conflict resolution",
                        "Stronger partnership"
                    ],
                    frequency="As needed, ideally daily",
                    integration_tips=[
                        "Set aside daily time for meaningful conversation",
                        "Practice during routine activities",
                        "Use before addressing any concerns",
                        "Apply during both easy and difficult conversations"
                    ]
                )
            ],
            
            LifeArea.PARENTING: [
                HouseholderPractice(
                    name="Dharmic Parenting",
                    area=LifeArea.PARENTING,
                    description="Raising children with dharmic values and principles",
                    steps=[
                        "Model the behavior and values you want to teach",
                        "Teach through stories and examples rather than lecturing",
                        "Create opportunities for children to serve others",
                        "Encourage questions and independent thinking",
                        "Balance discipline with love and understanding"
                    ],
                    benefits=[
                        "Children develop strong moral compass",
                        "Better family harmony and respect",
                        "Children prepared for ethical living",
                        "Stronger parent-child bonds"
                    ],
                    frequency="Continuous lifestyle",
                    integration_tips=[
                        "Use daily situations as teaching moments",
                        "Include children in family service activities",
                        "Share age-appropriate spiritual stories",
                        "Create family rituals and traditions"
                    ]
                ),
                HouseholderPractice(
                    name="Mindful Presence with Children",
                    area=LifeArea.PARENTING,
                    description="Being fully present and attentive with children",
                    steps=[
                        "Put away devices when interacting with children",
                        "Get down to child's physical level for conversations",
                        "Listen actively to their thoughts and feelings",
                        "Respond to their needs with patience and love",
                        "Create special one-on-one time with each child"
                    ],
                    benefits=[
                        "Children feel valued and heard",
                        "Stronger parent-child connection",
                        "Better behavior and cooperation",
                        "Reduced parenting stress"
                    ],
                    frequency="Daily interactions",
                    integration_tips=[
                        "Start with short periods of full attention",
                        "Use routine activities like bedtime",
                        "Practice deep breathing when feeling impatient",
                        "Remember that presence is a gift to children"
                    ]
                )
            ],
            
            LifeArea.CAREER: [
                HouseholderPractice(
                    name="Work as Spiritual Practice",
                    area=LifeArea.CAREER,
                    description="Approaching work with dharmic principles and spiritual awareness",
                    steps=[
                        "Begin work day with intention to serve through your role",
                        "Practice mindfulness and presence during work tasks",
                        "Treat colleagues with respect and kindness",
                        "Maintain integrity and honesty in all business dealings",
                        "Dedicate fruits of work to service of greater good"
                    ],
                    benefits=[
                        "Work becomes meaningful and fulfilling",
                        "Reduced work stress and burnout",
                        "Better relationships with colleagues",
                        "Spiritual growth through daily activities"
                    ],
                    frequency="Daily work practice",
                    integration_tips=[
                        "Start with morning intention setting",
                        "Take mindful breaks throughout day",
                        "Practice gratitude for work opportunities",
                        "Use challenges as opportunities for growth"
                    ]
                )
            ],
            
            LifeArea.SPIRITUALITY: [
                HouseholderPractice(
                    name="Daily Spiritual Integration",
                    area=LifeArea.SPIRITUALITY,
                    description="Maintaining spiritual practice within busy householder life",
                    steps=[
                        "Establish consistent morning spiritual practice",
                        "Use routine activities as mindfulness opportunities",
                        "Take short spiritual breaks throughout day",
                        "Practice gratitude and prayer before meals",
                        "End day with reflection and surrender"
                    ],
                    benefits=[
                        "Maintains spiritual connection amid busy life",
                        "Reduces stress and increases peace",
                        "Provides guidance for daily decisions",
                        "Supports overall wellbeing and balance"
                    ],
                    frequency="Daily integration",
                    integration_tips=[
                        "Start with short, sustainable practices",
                        "Use commute time for prayer or meditation",
                        "Include family in spiritual activities",
                        "Adapt practices to life circumstances"
                    ]
                )
            ]
        }
    
    def _initialize_challenge_solutions(self) -> Dict[HouseholderChallenge, Dict[str, Any]]:
        """Initialize solutions for common householder challenges"""
        return {
            HouseholderChallenge.WORK_LIFE_BALANCE: {
                "description": "Balancing career demands with family time and personal needs",
                "solutions": [
                    "Set clear boundaries between work and family time",
                    "Prioritize activities based on values and importance",
                    "Learn to say no to non-essential commitments",
                    "Create transition rituals between work and family time",
                    "Practice presence and full engagement in current activity"
                ],
                "daily_practices": [
                    "Morning intention setting for day's priorities",
                    "Transition ritual when leaving work",
                    "Device-free time with family",
                    "Evening reflection on day's balance"
                ],
                "wisdom": "Perfect balance is a myth; aim for conscious choices and presence in each moment."
            },
            
            HouseholderChallenge.FINANCIAL_STRESS: {
                "description": "Anxiety and conflict around money, security, and financial goals",
                "solutions": [
                    "Create realistic budget and stick to it",
                    "Build emergency fund for security",
                    "Focus on gratitude for current resources",
                    "Invest in education and skill development",
                    "Practice generous giving within means"
                ],
                "daily_practices": [
                    "Gratitude for financial resources",
                    "Mindful spending decisions",
                    "Regular saving and investment",
                    "Avoiding impulse purchases"
                ],
                "wisdom": "True wealth is contentment with what you have while working responsibly toward goals."
            },
            
            HouseholderChallenge.SPIRITUAL_INTEGRATION: {
                "description": "Difficulty maintaining spiritual practice amid busy family life",
                "solutions": [
                    "Integrate spirituality into daily activities",
                    "Start with short, sustainable practices",
                    "Include family in spiritual activities when possible",
                    "Use routine activities as mindfulness opportunities",
                    "Create sacred space and time in home"
                ],
                "daily_practices": [
                    "Morning spiritual intention",
                    "Mindful daily activities",
                    "Gratitude and prayer throughout day",
                    "Evening reflection and surrender"
                ],
                "wisdom": "Spiritual practice is not separate from life but the consciousness with which you live."
            }
        }
    
    def _initialize_relationship_wisdom(self) -> Dict[RelationshipType, Dict[str, Any]]:
        """Initialize wisdom for different relationship types"""
        return {
            RelationshipType.SPOUSE: {
                "principles": [
                    "Mutual respect and appreciation",
                    "Open and honest communication",
                    "Shared spiritual and life goals",
                    "Balance of independence and togetherness",
                    "Service to each other's growth"
                ],
                "daily_practices": [
                    "Express gratitude and appreciation",
                    "Listen with full attention",
                    "Support each other's individual growth",
                    "Share household and family responsibilities",
                    "Maintain physical and emotional intimacy"
                ],
                "wisdom": "Marriage is a spiritual partnership where both souls support each other's journey to the Divine."
            },
            
            RelationshipType.PARENT_CHILD: {
                "principles": [
                    "Unconditional love and acceptance",
                    "Teaching through example and experience",
                    "Balancing structure with freedom",
                    "Respecting child's individual nature",
                    "Preparing child for independent living"
                ],
                "daily_practices": [
                    "Give full attention during interactions",
                    "Model the behavior you want to see",
                    "Use natural consequences for learning",
                    "Celebrate child's unique gifts and progress",
                    "Create meaningful family traditions"
                ],
                "wisdom": "Children are souls entrusted to your care; guide them to discover their own divine nature."
            }
        }
    
    def assess_householder_stage(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> HouseholderStage:
        """Assess user's current householder stage"""
        query_lower = query.lower()
        context = user_context or {}
        
        # Check for stage-specific indicators
        if any(word in query_lower for word in ["young adult", "starting career", "single", "dating"]):
            return HouseholderStage.YOUNG_ADULT
        elif any(word in query_lower for word in ["newly married", "newlywed", "just married"]):
            return HouseholderStage.MARRIED_EARLY
        elif any(word in query_lower for word in ["young children", "toddler", "baby", "parenting"]):
            return HouseholderStage.FAMILY_BUILDING
        elif any(word in query_lower for word in ["teenager", "adolescent", "teen"]):
            return HouseholderStage.FAMILY_MATURE
        elif any(word in query_lower for word in ["empty nest", "children left", "adult children"]):
            return HouseholderStage.EMPTY_NEST
        elif any(word in query_lower for word in ["retired", "retirement", "grandparent", "elder"]):
            return HouseholderStage.ELDER
        
        # Default to family building (most common)
        return HouseholderStage.FAMILY_BUILDING
    
    def identify_householder_challenges(self, query: str, context: Dict[str, Any]) -> List[HouseholderChallenge]:
        """Identify householder challenges mentioned in query"""
        challenges = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["work life balance", "career family", "time for family"]):
            challenges.append(HouseholderChallenge.WORK_LIFE_BALANCE)
        
        if any(word in query_lower for word in ["money", "financial", "budget", "debt", "income"]):
            challenges.append(HouseholderChallenge.FINANCIAL_STRESS)
        
        if any(word in query_lower for word in ["marriage problem", "relationship conflict", "arguing"]):
            challenges.append(HouseholderChallenge.RELATIONSHIP_CONFLICT)
        
        if any(word in query_lower for word in ["parenting", "children", "kids", "raising"]):
            challenges.append(HouseholderChallenge.PARENTING_GUIDANCE)
        
        if any(word in query_lower for word in ["spiritual practice", "meditation", "prayer", "no time for"]):
            challenges.append(HouseholderChallenge.SPIRITUAL_INTEGRATION)
        
        return challenges if challenges else [HouseholderChallenge.WORK_LIFE_BALANCE]
    
    def get_challenge_solutions(self, challenges: List[HouseholderChallenge]) -> Dict[str, str]:
        """Get solutions for identified challenges"""
        solutions = {}
        
        for challenge in challenges:
            challenge_data = self.challenge_solutions.get(challenge, {})
            solutions[challenge.value] = "; ".join(challenge_data.get("solutions", ["Practice patience and seek guidance"])[:2])
        
        return solutions
    
    async def process_grihastha_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> GrihasthaResponse:
        """Process householder-related query and provide comprehensive guidance"""
        try:
            context = user_context or {}
            
            # Assess householder aspects
            stage = self.assess_householder_stage(query, context)
            challenges = self.identify_householder_challenges(query, context)
            
            # Get guidance
            guidance = self.guidance_levels.get(stage)
            if not guidance:
                return self._create_fallback_response()
            
            # Generate specific guidance
            challenge_solutions = self.get_challenge_solutions(challenges)
            
            # Create relationship guidance
            relationship_guidance = {}
            for rel_type in [RelationshipType.SPOUSE, RelationshipType.PARENT_CHILD]:
                rel_data = self.relationship_wisdom.get(rel_type, {})
                relationship_guidance[rel_type.value] = rel_data.get("wisdom", "Practice love and understanding")
            
            return GrihasthaResponse(
                householder_stage=stage.value,
                primary_dharma=guidance.primary_dharma,
                spiritual_integration=guidance.spiritual_integration,
                relationship_guidance=relationship_guidance,
                daily_practices=guidance.daily_practices,
                challenge_solutions=challenge_solutions,
                balance_strategies=guidance.balance_strategies,
                service_opportunities=guidance.service_opportunities,
                practical_applications=guidance.practical_applications,
                wisdom_guidance=f"The householder path teaches that family life itself is spiritual practice when lived with dharma, love, and service. {guidance.progress_indicators[0] if guidance.progress_indicators else ''}"
            )
            
        except Exception as e:
            logger.error(f"Error processing grihastha query: {e}")
            return self._create_fallback_response()
    
    def _create_fallback_response(self) -> GrihasthaResponse:
        """Create fallback response when processing fails"""
        return GrihasthaResponse(
            householder_stage="family_building",
            primary_dharma=[
                "Raise children with love and dharmic values",
                "Maintain loving marriage relationship",
                "Provide for family's material and spiritual needs",
                "Serve community through family example"
            ],
            spiritual_integration=[
                "Include family in daily spiritual activities",
                "Practice mindfulness during household tasks",
                "Use parenting challenges as spiritual growth opportunities",
                "Create sacred space and time in home"
            ],
            relationship_guidance={
                "spouse": "Marriage is a spiritual partnership where both souls support each other's journey",
                "parent_child": "Children are souls entrusted to your care; guide them with love and wisdom"
            },
            daily_practices=[
                "Morning family gratitude practice",
                "Mindful presence during family interactions",
                "Evening reflection on dharmic living",
                "Regular appreciation expression to family members"
            ],
            challenge_solutions={
                "work_life_balance": "Set clear boundaries and practice presence in each moment",
                "spiritual_integration": "Integrate spirituality into daily activities rather than separating it"
            },
            balance_strategies=[
                "Focus on what truly matters most in each situation",
                "Practice saying no to non-essential commitments",
                "Create family time that is protected from outside demands",
                "Remember that perfect balance is less important than conscious choices"
            ],
            service_opportunities=[
                "Serve community through your family's positive example",
                "Include children in family service projects",
                "Support other families in your community",
                "Volunteer at children's schools and activities"
            ],
            practical_applications=[
                "Use daily household tasks as mindfulness practice",
                "Apply patience and compassion in family conflicts",
                "Create meaningful family traditions and celebrations",
                "Practice gratitude for family blessings and challenges"
            ],
            wisdom_guidance="The householder path teaches that family life itself is spiritual practice when approached with dharma, love, and service to others."
        )


# Global instance
_grihastha_module = None

def get_grihastha_module() -> GrihasthaModule:
    """Get global Grihastha module instance"""
    global _grihastha_module
    if _grihastha_module is None:
        _grihastha_module = GrihasthaModule()
    return _grihastha_module

# Factory function for easy access
def create_grihastha_guidance(query: str, user_context: Optional[Dict[str, Any]] = None) -> GrihasthaResponse:
    """Factory function to create grihastha guidance"""
    import asyncio
    module = get_grihastha_module()
    return asyncio.run(module.process_grihastha_query(query, user_context))
