"""
🤝 Satsang Module - Spiritual Community and Sacred Association
Complete system for building and participating in spiritual community
Based on traditional Satsang principles and modern spiritual fellowship
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SatsangLevel(Enum):
    """Levels of spiritual community engagement"""
    SEEKER = "seeker"                   # Looking for spiritual community
    PARTICIPANT = "participant"         # Actively participating in satsang
    CONTRIBUTOR = "contributor"         # Contributing to community growth
    FACILITATOR = "facilitator"         # Helping guide community activities
    ELDER = "elder"                     # Wise elder serving community


class SatsangType(Enum):
    """Types of spiritual community"""
    STUDY_GROUP = "study_group"                     # Scripture study
    MEDITATION_GROUP = "meditation_group"           # Group meditation
    SERVICE_COMMUNITY = "service_community"         # Community service
    DEVOTIONAL_GATHERING = "devotional_gathering"   # Kirtan, worship
    WISDOM_CIRCLE = "wisdom_circle"                 # Sharing wisdom
    ONLINE_SATSANG = "online_satsang"               # Virtual community
    FAMILY_SATSANG = "family_satsang"               # Family practice
    TEACHER_STUDENT = "teacher_student"             # Teacher-disciple


class SatsangBenefit(Enum):
    """Benefits of spiritual community"""
    SPIRITUAL_SUPPORT = "spiritual_support"         # Encouragement
    SHARED_WISDOM = "shared_wisdom"                 # Learning from others
    ACCOUNTABILITY = "accountability"               # Practice support
    INSPIRATION = "inspiration"                     # Motivation
    SERVICE_OPPORTUNITY = "service_opportunity"     # Serve others
    PROTECTION = "protection"                       # Shield negativity
    TRANSFORMATION = "transformation"               # Accelerated growth
    BELONGING = "belonging"                         # Spiritual family


class SatsangChallenge(Enum):
    """Common challenges in spiritual community"""
    EGO_CONFLICTS = "ego_conflicts"                 # Personality clashes
    SPIRITUAL_MATERIALISM = "spiritual_materialism" # Using for ego
    DEPENDENCY = "dependency"                       # Over-reliance
    GOSSIP_DRAMA = "gossip_drama"                   # Community politics
    DIFFERENT_PATHS = "different_paths"             # Conflicting approaches
    TIME_COMMITMENT = "time_commitment"             # Balancing duties
    LEADERSHIP_ISSUES = "leadership_issues"         # Authority problems
    ISOLATION_TENDENCY = "isolation_tendency"       # Solitary preference


class SatsangPractice(Enum):
    """Practices for spiritual community"""
    GROUP_MEDITATION = "group_meditation"           # Collective meditation
    SCRIPTURE_STUDY = "scripture_study"             # Study spiritual texts
    DEVOTIONAL_SINGING = "devotional_singing"       # Kirtan, bhajan, hymns
    SERVICE_PROJECTS = "service_projects"           # Community service
    WISDOM_SHARING = "wisdom_sharing"               # Share insights
    SILENT_SATSANG = "silent_satsang"               # Sitting in silence
    SPIRITUAL_DISCUSSION = "spiritual_discussion"   # Dharmic conversations
    CELEBRATION = "celebration"                     # Spiritual festivals


@dataclass
class SatsangGuidance:
    """Comprehensive satsang guidance"""
    level: SatsangLevel
    primary_teaching: str
    community_practices: List[str]
    leadership_skills: List[str]
    daily_integration: List[str]
    challenge_solutions: Dict[str, str]
    relationship_wisdom: List[str]
    service_opportunities: List[str]
    practical_applications: List[str]
    progress_indicators: List[str]


@dataclass
class SatsangInsight:
    """Insight about spiritual community"""
    community_type: SatsangType
    benefit: SatsangBenefit
    teaching: str
    practice_method: str
    integration_tip: str


class SatsangResponse(BaseModel):
    """Response from Satsang module"""
    satsang_level: str = Field(description="Current community engagement level")
    community_guidance: str = Field(description="Guidance on spiritual community")
    participation_practices: List[str] = Field(
        description="Ways to participate in satsang"
    )
    leadership_development: List[str] = Field(
        description="Skills for community leadership"
    )
    daily_integration: List[str] = Field(
        description="Daily practices for community spirit"
    )
    challenge_solutions: Dict[str, str] = Field(
        description="Solutions for community challenges"
    )
    relationship_wisdom: List[str] = Field(
        description="Wisdom for spiritual relationships"
    )
    service_opportunities: List[str] = Field(
        description="Ways to serve the community"
    )
    practical_applications: List[str] = Field(
        description="Practical community building"
    )
    scriptural_wisdom: str = Field(description="Traditional teachings on satsang")


class SatsangModule:
    """
    🤝 Satsang Module - Spiritual Community and Sacred Association
    
    Based on traditional Satsang teachings:
    - Upanishads on the power of spiritual company
    - Bhagavad Gita teachings on spiritual fellowship
    - Tulsi Das: "Satsang gatein durlabh bhavati" (Good company is rare and precious)
    - Modern spiritual community principles
    
    Satsang literally means "association with truth" or "gathering with good people"
    It's considered one of the most powerful means of spiritual transformation.
    """
    
    def __init__(self):
        self.name = "Satsang"
        self.color = "🤝"
        self.element = "Community"
        self.principles = ["Truth Association", "Mutual Support", "Shared Wisdom", "Service"]
        self.guidance_levels = self._initialize_guidance_levels()
        self.community_practices = self._initialize_community_practices()
        self.challenge_solutions = self._initialize_challenge_solutions()
        self.relationship_wisdom = self._initialize_relationship_wisdom()
        
    def _initialize_guidance_levels(self) -> Dict[SatsangLevel, SatsangGuidance]:
        """Initialize guidance for different levels of spiritual community engagement"""
        return {
            SatsangLevel.SEEKER: SatsangGuidance(
                level=SatsangLevel.SEEKER,
                primary_teaching="The company you keep shapes who you become. Seek out those who inspire your highest nature and support your spiritual growth.",
                community_practices=[
                    "Attend local spiritual gatherings and events",
                    "Join meditation groups or spiritual study circles",
                    "Participate in devotional singing or chanting",
                    "Engage in community service with like-minded people",
                    "Seek out wise teachers and mentors"
                ],
                leadership_skills=[
                    "Learn to listen deeply and without judgment",
                    "Practice humility and openness to learning",
                    "Develop genuine interest in others' spiritual journey",
                    "Cultivate patience with different levels of understanding"
                ],
                daily_integration=[
                    "Morning prayer for guidance in finding spiritual community",
                    "Mindful interaction with all people you meet",
                    "Evening reflection on spiritual influences in your day",
                    "Reading about spiritual communities and traditions",
                    "Online participation in spiritual forums or groups"
                ],
                challenge_solutions={
                    "shyness": "Start with online communities or small groups where you feel safer",
                    "skepticism": "Visit different communities to find authentic spiritual atmosphere",
                    "time_constraints": "Begin with minimal commitment and gradually increase involvement"
                },
                relationship_wisdom=[
                    "Quality of spiritual friends matters more than quantity",
                    "Look for those who practice what they preach",
                    "Seek people who inspire growth rather than comfort ego",
                    "Be patient in finding your spiritual family"
                ],
                service_opportunities=[
                    "Help with setup and cleanup at spiritual events",
                    "Offer to bring food or materials for gatherings",
                    "Share your skills (music, writing, organizing) with community",
                    "Welcome newcomers and help them feel included"
                ],
                practical_applications=[
                    "Research local spiritual centers and communities",
                    "Attend different types of gatherings to find your fit",
                    "Start conversations about spiritual topics with open-minded friends",
                    "Create study group if none exists in your area"
                ],
                progress_indicators=[
                    "Finding communities that inspire and support your growth",
                    "Feeling more connected to like-minded spiritual seekers",
                    "Experiencing encouragement and accountability in practice",
                    "Beginning to form meaningful spiritual friendships"
                ]
            ),
            
            SatsangLevel.PARTICIPANT: SatsangGuidance(
                level=SatsangLevel.PARTICIPANT,
                primary_teaching="True participation in satsang means bringing your whole self - your questions, struggles, and insights - to the community with sincerity.",
                community_practices=[
                    "Regular attendance at spiritual gatherings",
                    "Active participation in group meditation and discussions",
                    "Sharing your insights and experiences appropriately",
                    "Supporting other community members in their journey",
                    "Contributing to community activities and events"
                ],
                leadership_skills=[
                    "Practice vulnerable and honest sharing",
                    "Develop skills in group facilitation and discussion",
                    "Learn to ask questions that deepen understanding",
                    "Cultivate ability to hold space for others' emotions"
                ],
                daily_integration=[
                    "Morning intention to contribute positively to community",
                    "Practice presence and deep listening in all interactions",
                    "Carry community's wisdom and support throughout your day",
                    "Evening gratitude for spiritual friends and teachers",
                    "Regular communication with spiritual friends"
                ],
                challenge_solutions={
                    "comparison": "Focus on your own growth rather than measuring against others",
                    "shyness_in_sharing": "Start with small, genuine contributions and build confidence",
                    "time_management": "Prioritize quality participation over quantity of activities"
                },
                relationship_wisdom=[
                    "Spiritual friendship requires both giving and receiving",
                    "Be genuine rather than trying to appear spiritual",
                    "Support others without trying to fix or teach them",
                    "Practice patience with different personalities and approaches"
                ],
                service_opportunities=[
                    "Assist with organizing community events and activities",
                    "Offer transportation to those who need it",
                    "Help with community communication and outreach",
                    "Support new members in feeling welcomed and included"
                ],
                practical_applications=[
                    "Create regular schedule for community participation",
                    "Form study partnerships or accountability relationships",
                    "Practice spiritual principles in interactions with community",
                    "Use community time as intensive spiritual practice"
                ],
                progress_indicators=[
                    "Feeling genuine connection and belonging in spiritual community",
                    "Regular participation becomes natural and fulfilling",
                    "Others begin to seek your friendship and support",
                    "Community participation enhances rather than burdens daily life"
                ]
            ),
            
            SatsangLevel.CONTRIBUTOR: SatsangGuidance(
                level=SatsangLevel.CONTRIBUTOR,
                primary_teaching="Contributing to satsang is not about being perfect but about offering your unique gifts in service of collective spiritual growth.",
                community_practices=[
                    "Lead or co-facilitate group activities and discussions",
                    "Organize service projects and community outreach",
                    "Mentor newcomers and support their integration",
                    "Share teachings and insights from your practice",
                    "Help resolve conflicts and maintain harmony"
                ],
                leadership_skills=[
                    "Develop teaching and presentation abilities",
                    "Learn conflict resolution and mediation skills",
                    "Practice discernment in guidance and advice giving",
                    "Cultivate ability to inspire without ego involvement"
                ],
                daily_integration=[
                    "Morning reflection on how to serve community today",
                    "Carry responsibility for community wellbeing throughout day",
                    "Practice leadership qualities in all life interactions",
                    "Evening assessment of service and leadership effectiveness",
                    "Regular planning and preparation for community activities"
                ],
                challenge_solutions={
                    "burnout": "Balance service with self-care and personal practice",
                    "ego_inflation": "Remember you are servant, not savior of community",
                    "criticism": "Use feedback as opportunity for growth and improvement"
                },
                relationship_wisdom=[
                    "Leadership in satsang is about empowerment, not control",
                    "Model the spiritual qualities you hope to inspire",
                    "Be willing to learn from anyone, regardless of experience level",
                    "Handle disagreements with wisdom and compassion"
                ],
                service_opportunities=[
                    "Teach classes or lead workshops on spiritual topics",
                    "Coordinate community service projects",
                    "Provide pastoral care and support during difficult times",
                    "Represent community in interfaith or social activities"
                ],
                practical_applications=[
                    "Develop skills in public speaking and group facilitation",
                    "Study deeply to offer authentic wisdom and guidance",
                    "Create systems and structures that support community growth",
                    "Build partnerships with other spiritual communities"
                ],
                progress_indicators=[
                    "Community members naturally turn to you for guidance",
                    "Successful leadership of community projects and activities",
                    "Ability to inspire others while remaining humble",
                    "Community growth and harmony under your influence"
                ]
            ),
            
            SatsangLevel.FACILITATOR: SatsangGuidance(
                level=SatsangLevel.FACILITATOR,
                primary_teaching="The highest art of satsang facilitation is to create space where truth can emerge naturally, not to impose your understanding on others.",
                community_practices=[
                    "Create and hold sacred space for deep spiritual sharing",
                    "Guide community through challenges and transitions",
                    "Facilitate wisdom emergence rather than teaching content",
                    "Support other leaders and maintain community vision",
                    "Bridge different spiritual traditions and approaches"
                ],
                leadership_skills=[
                    "Master the art of asking powerful questions",
                    "Develop intuitive sense of group dynamics and needs",
                    "Practice holding multiple perspectives simultaneously",
                    "Cultivate ability to channel wisdom rather than ego"
                ],
                daily_integration=[
                    "Morning attunement to highest wisdom for community service",
                    "Practice facilitator consciousness in all interactions",
                    "Maintain awareness of community field throughout day",
                    "Evening surrender of leadership outcomes to Divine will",
                    "Continuous study and deepening of spiritual understanding"
                ],
                challenge_solutions={
                    "power_dynamics": "Use authority to empower others, not to dominate",
                    "group_conflicts": "Stay neutral while guiding toward resolution",
                    "spiritual_bypass": "Address practical issues alongside spiritual ones"
                },
                relationship_wisdom=[
                    "True spiritual authority comes from love and wisdom, not position",
                    "Create safety for vulnerability and authentic expression",
                    "Honor the divine in every community member",
                    "Balance guidance with allowing natural group wisdom to emerge"
                ],
                service_opportunities=[
                    "Train and mentor other community leaders",
                    "Facilitate healing and reconciliation processes",
                    "Guide community visioning and strategic planning",
                    "Serve as bridge between community and outside resources"
                ],
                practical_applications=[
                    "Develop deep listening and intuitive guidance abilities",
                    "Study various facilitation methods and spiritual traditions",
                    "Create rituals and practices that support community bonding",
                    "Build networks with other spiritual communities and teachers"
                ],
                progress_indicators=[
                    "Community members experience profound growth and healing",
                    "Group wisdom emerges naturally in gatherings you facilitate",
                    "Other communities seek your guidance and collaboration",
                    "Sustainable community systems develop under your leadership"
                ]
            ),
            
            SatsangLevel.ELDER: SatsangGuidance(
                level=SatsangLevel.ELDER,
                primary_teaching="The elder in satsang serves by being a living example of spiritual maturity, offering wisdom through presence more than words.",
                community_practices=[
                    "Serve as repository of community wisdom and history",
                    "Offer guidance and blessing to community leaders",
                    "Hold vision of community's highest potential",
                    "Model spiritual maturity and graceful aging",
                    "Bridge traditions and prepare next generation of leaders"
                ],
                leadership_skills=[
                    "Embody wisdom and compassion in all interactions",
                    "Offer guidance with detachment from outcomes",
                    "Practice deep listening to community needs and direction",
                    "Serve as anchor of stability during community changes"
                ],
                daily_integration=[
                    "Morning blessing and prayers for community wellbeing",
                    "Living as example of spiritual principles throughout day",
                    "Offering silent blessing and support to all you encounter",
                    "Evening gratitude for opportunity to serve community",
                    "Continuous preparation for eventual transition of leadership"
                ],
                challenge_solutions={
                    "relevance_concerns": "Share wisdom in ways that speak to current needs",
                    "succession_anxiety": "Trust in community's ability to continue growing",
                    "physical_limitations": "Adapt service to current capabilities while maintaining contribution"
                },
                relationship_wisdom=[
                    "Offer wisdom when asked, silence when not",
                    "See the divine potential in every community member",
                    "Practice letting go while remaining lovingly available",
                    "Bless the community's evolution even when it differs from your vision"
                ],
                service_opportunities=[
                    "Mentor and bless younger community leaders",
                    "Preserve and transmit community wisdom and traditions",
                    "Serve as counselor and wise advisor",
                    "Bridge community with broader spiritual lineages"
                ],
                practical_applications=[
                    "Document community history and wisdom for future generations",
                    "Create systems for ongoing leadership development",
                    "Establish enduring structures that support community mission",
                    "Prepare graceful transition of active leadership responsibilities"
                ],
                progress_indicators=[
                    "Community recognizes you as source of wisdom and stability",
                    "Younger leaders naturally seek your guidance and blessing",
                    "Community continues to thrive with your support but not dependence",
                    "Your presence brings peace and inspiration to community gatherings"
                ]
            )
        }
    
    def _initialize_community_practices(self) -> Dict[SatsangPractice, List[str]]:
        """Initialize specific practices for spiritual community"""
        return {
            SatsangPractice.GROUP_MEDITATION: [
                "Sit in circle for collective meditation",
                "Synchronize breathing as group",
                "Share silence and presence together",
                "Close with group Om or blessing"
            ],
            
            SatsangPractice.SCRIPTURE_STUDY: [
                "Choose sacred text for community study",
                "Take turns reading passages aloud",
                "Share insights and personal applications",
                "Discuss practical implementation in daily life"
            ],
            
            SatsangPractice.DEVOTIONAL_SINGING: [
                "Practice kirtan, bhajan, or spiritual songs",
                "Use music to open hearts and create unity",
                "Include both traditional and contemporary spiritual music",
                "Create opportunities for musical expression and participation"
            ],
            
            SatsangPractice.SERVICE_PROJECTS: [
                "Organize community service activities",
                "Serve together at local charities",
                "Support community members in need",
                "Engage in environmental or social justice projects"
            ],
            
            SatsangPractice.WISDOM_SHARING: [
                "Create opportunities for members to share insights",
                "Hold storytelling circles about spiritual experiences",
                "Invite guest teachers and speakers",
                "Facilitate peer teaching and learning"
            ],
            
            SatsangPractice.SILENT_SATSANG: [
                "Gather simply to sit in silence together",
                "Practice presence without agenda or program",
                "Allow grace and blessing to emerge naturally",
                "Honor the power of collective stillness"
            ]
        }
    
    def _initialize_challenge_solutions(self) -> Dict[SatsangChallenge, Dict[str, Any]]:
        """Initialize solutions for spiritual community challenges"""
        return {
            SatsangChallenge.EGO_CONFLICTS: {
                "description": "Personality clashes and spiritual competition within community",
                "solutions": [
                    "Address conflicts directly with compassion and honesty",
                    "Focus on common spiritual goals rather than personality differences",
                    "Practice humility and willingness to admit mistakes",
                    "Seek mediation from wise community elders when needed",
                    "Remember that conflicts can be opportunities for growth"
                ],
                "practices": ["Group conflict resolution sessions", "Personal forgiveness practice"],
                "wisdom": "Spiritual community is not about perfect people but about imperfect people supporting each other's growth."
            },
            
            SatsangChallenge.SPIRITUAL_MATERIALISM: {
                "description": "Using spiritual community for ego enhancement rather than genuine growth",
                "solutions": [
                    "Regularly examine your motivations for community participation",
                    "Focus on service rather than recognition or status",
                    "Practice humility and beginner's mind",
                    "Seek feedback from trusted community members",
                    "Remember that spiritual growth is measured internally, not externally"
                ],
                "practices": ["Regular self-examination", "Anonymous service"],
                "wisdom": "True spiritual attainment is recognized by humility, not by claims of achievement."
            },
            
            SatsangChallenge.DEPENDENCY: {
                "description": "Over-reliance on community without developing inner strength",
                "solutions": [
                    "Maintain balance between community support and personal practice",
                    "Develop individual spiritual disciplines alongside group activities",
                    "Take breaks from community to strengthen personal relationship with divine",
                    "Practice making decisions based on inner guidance",
                    "Support others rather than always seeking support"
                ],
                "practices": ["Personal retreat time", "Individual spiritual practice"],
                "wisdom": "Healthy satsang supports your direct relationship with the divine, not replaces it."
            },
            
            SatsangChallenge.GOSSIP_DRAMA: {
                "description": "Community politics and interpersonal drama distracting from spiritual focus",
                "solutions": [
                    "Establish clear community guidelines about speech and conduct",
                    "Practice speaking directly to people rather than about them",
                    "Focus conversations on spiritual topics rather than personalities",
                    "Address gossip immediately when it arises",
                    "Model the behavior you want to see in community"
                ],
                "practices": ["Right speech practice", "Direct communication"],
                "wisdom": "Sacred speech honors the divine in all beings and serves the highest good."
            }
        }
    
    def _initialize_relationship_wisdom(self) -> List[str]:
        """Initialize wisdom for spiritual relationships"""
        return [
            "In satsang, we see each person as a manifestation of the divine worthy of respect",
            "Spiritual friendship involves both giving and receiving support on the path",
            "True satsang creates safety for vulnerability and authentic expression",
            "The purpose of spiritual community is mutual encouragement toward the highest truth",
            "In conflict, remember that both parties are souls learning and growing",
            "Spiritual authority comes from wisdom and love, not from position or claims",
            "Honor the unique path and pace of each person's spiritual journey",
            "Practice presence and deep listening as forms of loving service",
            "Share your struggles and insights honestly to help others feel less alone",
            "Create space for silence and reflection, not just talking and activity"
        ]
    
    def assess_satsang_level(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> SatsangLevel:
        """Assess user's current level of community engagement"""
        query_lower = query.lower()
        context = user_context or {}
        
        # Check for elder level indicators
        if any(word in query_lower for word in ["elder", "senior member", "community wisdom", "mentoring leaders"]):
            return SatsangLevel.ELDER
        
        # Check for facilitator level indicators
        if any(word in query_lower for word in ["facilitating", "leading community", "group guidance", "teaching"]):
            return SatsangLevel.FACILITATOR
        
        # Check for contributor level indicators
        if any(word in query_lower for word in ["contributing", "organizing", "helping community", "service"]):
            return SatsangLevel.CONTRIBUTOR
        
        # Check for participant level indicators
        if any(word in query_lower for word in ["participating", "attending", "member of", "active in"]):
            return SatsangLevel.PARTICIPANT
        
        # Default to seeker
        return SatsangLevel.SEEKER
    
    def identify_community_challenges(self, query: str, context: Dict[str, Any]) -> List[SatsangChallenge]:
        """Identify community challenges mentioned in query"""
        challenges = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["conflict", "personality clash", "competition", "ego"]):
            challenges.append(SatsangChallenge.EGO_CONFLICTS)
        
        if any(word in query_lower for word in ["spiritual ego", "showing off", "recognition", "status"]):
            challenges.append(SatsangChallenge.SPIRITUAL_MATERIALISM)
        
        if any(word in query_lower for word in ["dependent", "relying too much", "can't function", "codependent"]):
            challenges.append(SatsangChallenge.DEPENDENCY)
        
        if any(word in query_lower for word in ["gossip", "drama", "politics", "talking about"]):
            challenges.append(SatsangChallenge.GOSSIP_DRAMA)
        
        if any(word in query_lower for word in ["different paths", "disagreement", "conflicting approaches"]):
            challenges.append(SatsangChallenge.DIFFERENT_PATHS)
        
        if any(word in query_lower for word in ["no time", "too busy", "scheduling", "commitment"]):
            challenges.append(SatsangChallenge.TIME_COMMITMENT)
        
        return challenges if challenges else [SatsangChallenge.EGO_CONFLICTS]
    
    def get_challenge_solutions(self, challenges: List[SatsangChallenge]) -> Dict[str, str]:
        """Get solutions for identified challenges"""
        solutions = {}
        
        for challenge in challenges:
            challenge_data = self.challenge_solutions.get(challenge, {})
            solutions[challenge.value] = "; ".join(challenge_data.get("solutions", ["Practice patience and seek guidance"])[:2])
        
        return solutions
    
    def get_scriptural_wisdom(self, level: SatsangLevel) -> str:
        """Get scriptural wisdom appropriate to satsang level"""
        wisdom_map = {
            SatsangLevel.SEEKER: "Mundaka Upanishad: 'When two people sit together and share spiritual wisdom, the divine is present.'",
            SatsangLevel.PARTICIPANT: "Bhagavad Gita 9.30: 'Even if someone is engaged in the most abominable activities, if they are devoted to spiritual practice, they should be considered saintly.'",
            SatsangLevel.CONTRIBUTOR: "Tulsi Das: 'Satsang gatein durlabh bhavati' - Good company is rare and precious, treasure it.",
            SatsangLevel.FACILITATOR: "Tao Te Ching: 'The best leaders are those who serve. The people hardly know they exist.'",
            SatsangLevel.ELDER: "Upanishads: 'From the unreal, lead me to the real; from darkness, lead me to light; from death, lead me to immortality.'"
        }
        return wisdom_map.get(level, "Sanskrit: Satsang means 'association with truth' - seek the company of those who inspire your highest nature.")
    
    async def process_satsang_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> SatsangResponse:
        """Process satsang-related query and provide comprehensive guidance"""
        try:
            context = user_context or {}
            
            # Assess satsang aspects
            level = self.assess_satsang_level(query, context)
            challenges = self.identify_community_challenges(query, context)
            
            # Get guidance
            guidance = self.guidance_levels.get(level)
            if not guidance:
                return self._create_fallback_response()
            
            # Generate specific guidance
            challenge_solutions = self.get_challenge_solutions(challenges)
            scriptural_wisdom = self.get_scriptural_wisdom(level)
            
            return SatsangResponse(
                satsang_level=level.value,
                community_guidance=guidance.primary_teaching,
                participation_practices=guidance.community_practices,
                leadership_development=guidance.leadership_skills,
                daily_integration=guidance.daily_integration,
                challenge_solutions=challenge_solutions,
                relationship_wisdom=guidance.relationship_wisdom,
                service_opportunities=guidance.service_opportunities,
                practical_applications=guidance.practical_applications,
                scriptural_wisdom=scriptural_wisdom
            )
            
        except Exception as e:
            logger.error(f"Error processing satsang query: {e}")
            return self._create_fallback_response()
    
    def _create_fallback_response(self) -> SatsangResponse:
        """Create fallback response when processing fails"""
        return SatsangResponse(
            satsang_level="seeker",
            community_guidance="The company you keep shapes who you become. Seek out those who inspire your highest nature and support your spiritual growth.",
            participation_practices=[
                "Attend local spiritual gatherings and events",
                "Join meditation groups or spiritual study circles",
                "Participate in devotional singing or chanting",
                "Engage in community service with like-minded people"
            ],
            leadership_development=[
                "Learn to listen deeply and without judgment",
                "Practice humility and openness to learning",
                "Develop genuine interest in others' spiritual journey",
                "Cultivate patience with different levels of understanding"
            ],
            daily_integration=[
                "Morning prayer for guidance in finding spiritual community",
                "Mindful interaction with all people you meet",
                "Evening reflection on spiritual influences in your day",
                "Reading about spiritual communities and traditions"
            ],
            challenge_solutions={
                "shyness": "Start with online communities or small groups where you feel safer",
                "skepticism": "Visit different communities to find authentic spiritual atmosphere"
            },
            relationship_wisdom=self.relationship_wisdom[:5],
            service_opportunities=[
                "Help with setup and cleanup at spiritual events",
                "Offer to bring food or materials for gatherings",
                "Share your skills with community",
                "Welcome newcomers and help them feel included"
            ],
            practical_applications=[
                "Research local spiritual centers and communities",
                "Attend different types of gatherings to find your fit",
                "Start conversations about spiritual topics with open-minded friends",
                "Create study group if none exists in your area"
            ],
            scriptural_wisdom="Sanskrit: Satsang means 'association with truth' - seek the company of those who inspire your highest nature."
        )
    
    def get_satsang_insight(self, community_type: SatsangType, benefit: SatsangBenefit) -> Optional[SatsangInsight]:
        """Get specific insight about satsang type and benefit"""
        type_teachings = {
            SatsangType.STUDY_GROUP: "Study together deepens understanding through shared inquiry",
            SatsangType.MEDITATION_GROUP: "Group meditation amplifies the peace and presence of individual practice",
            SatsangType.SERVICE_COMMUNITY: "Serving together creates bonds of love and shared purpose",
            SatsangType.DEVOTIONAL_GATHERING: "Devotional practices open hearts and create divine connection"
        }
        
        benefit_practices = {
            SatsangBenefit.SPIRITUAL_SUPPORT: "Share your challenges and victories with spiritual friends",
            SatsangBenefit.SHARED_WISDOM: "Learn from others' experiences and insights",
            SatsangBenefit.ACCOUNTABILITY: "Commit to practices with community support",
            SatsangBenefit.INSPIRATION: "Allow others' dedication to inspire your own practice"
        }
        
        teaching = type_teachings.get(community_type, "All forms of satsang serve spiritual growth")
        practice = benefit_practices.get(benefit, "Participate with openness and sincerity")
        
        return SatsangInsight(
            community_type=community_type,
            benefit=benefit,
            teaching=teaching,
            practice_method=practice,
            integration_tip="Carry the spirit of satsang into all your relationships"
        )


# Global instance
_satsang_module = None

def get_satsang_module() -> SatsangModule:
    """Get global Satsang module instance"""
    global _satsang_module
    if _satsang_module is None:
        _satsang_module = SatsangModule()
    return _satsang_module

# Factory function for easy access
def create_satsang_guidance(query: str, user_context: Optional[Dict[str, Any]] = None) -> SatsangResponse:
    """Factory function to create satsang guidance"""
    import asyncio
    module = get_satsang_module()
    return asyncio.run(module.process_satsang_query(query, user_context))
