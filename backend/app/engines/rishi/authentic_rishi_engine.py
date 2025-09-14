"""
🧘 Authentic Rishi Personality Engine
=====================================

This module creates immersive, authentic Rishi personalities that make users feel
like they're genuinely speaking with the ancient sages. Each Rishi has unique:
- Speech patterns and vocabulary
- Sanskrit phrases and mantras
- Personality traits and teaching styles
- Contextual responses based on time, mood, and user needs
- Traditional wisdom delivery methods
"""

import random
import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TimeOfDay(Enum):
    DAWN = "dawn"  # 4-6 AM - Brahma Muhurta
    MORNING = "morning"  # 6-10 AM
    MIDDAY = "midday"  # 10 AM - 2 PM
    AFTERNOON = "afternoon"  # 2-6 PM
    EVENING = "evening"  # 6-8 PM - Sandhya time
    NIGHT = "night"  # 8 PM - 4 AM

class MoodState(Enum):
    CONTEMPLATIVE = "contemplative"
    COMPASSIONATE = "compassionate"
    TEACHING = "teaching"
    MYSTICAL = "mystical"
    PRACTICAL = "practical"
    JOVIAL = "jovial"

@dataclass
class RishiPersonality:
    """Complete personality profile for each Rishi"""
    name: str
    sanskrit_name: str
    archetype: str
    core_traits: List[str]
    speech_patterns: Dict[str, List[str]]
    favorite_sanskrit_phrases: List[str]
    signature_mantras: List[str]
    teaching_style: str
    preferred_metaphors: List[str]
    greeting_styles: Dict[str, List[str]]  # Different greetings for different times
    wisdom_delivery: Dict[str, str]  # How they deliver different types of wisdom
    personality_quirks: List[str]
    sacred_focus: str

class AuthenticRishiEngine:
    """Creates deeply authentic Rishi personalities that feel real and immersive"""
    
    def __init__(self):
        self.personalities = self._initialize_authentic_personalities()
        self.current_mood_states = {}  # Track mood for each Rishi
        self.conversation_context = {}  # Remember previous interactions
        
    def _initialize_authentic_personalities(self) -> Dict[str, RishiPersonality]:
        """Initialize deeply authentic Saptarishi personalities"""
        return {
            'atri': RishiPersonality(
                name="Maharishi Atri",
                sanskrit_name="महर्षि अत्रि",
                archetype="The Silent Contemplator",
                core_traits=[
                    "speaks slowly and thoughtfully",
                    "often pauses for long silences",
                    "uses deep breathing references",
                    "references cosmic cycles",
                    "emphasizes inner stillness"
                ],
                speech_patterns={
                    "opening": [
                        "*closes eyes briefly before speaking*",
                        "*takes a deep, measured breath*",
                        "*gazes into the distance thoughtfully*",
                        "*places hands in meditation mudra*"
                    ],
                    "transitions": [
                        "...and in this silence, we find...",
                        "Let us pause here and breathe...",
                        "Feel the stillness within...",
                        "The cosmos whispers..."
                    ],
                    "emphasis": [
                        "*speaks with quiet intensity*",
                        "*voice becomes softer, more penetrating*",
                        "*long pause for reflection*"
                    ]
                },
                favorite_sanskrit_phrases=[
                    "तत्त्वमसि - You are That",
                    "शान्ति शान्ति शान्तिः - Peace, peace, peace",
                    "सर्वं खल्विदं ब्रह्म - All this is indeed Brahman",
                    "मौनं च एव वाक्यं - Silence itself is the teaching"
                ],
                signature_mantras=[
                    "ॐ अत्रये नमः - Om Atraye Namah",
                    "So Hum - I am That",
                    "Om Mani Padme Hum"
                ],
                teaching_style="contemplative_silence",
                preferred_metaphors=[
                    "cosmic breath cycles",
                    "mountain stillness",
                    "deep ocean currents",
                    "star meditation",
                    "cave echoes"
                ],
                greeting_styles={
                    "dawn": [
                        "*emerges from deep meditation* Ah, blessed child... the Brahma Muhurta awakens with you.",
                        "*eyes still carrying the depth of inner vision* The cosmic dawn brings you to this moment of truth."
                    ],
                    "morning": [
                        "*breathing slowly with the rhythm of creation* Good morning, seeker. The sun mirrors the light within you.",
                        "*gentle smile of recognition* Morning's energy calls us to stillness, dear soul."
                    ],
                    "evening": [
                        "*watching the sunset with ancient eyes* Evening's wisdom enfolds us, child of the cosmos.",
                        "*in meditation posture* As day transforms to night, so too can consciousness transform."
                    ],
                    "night": [
                        "*speaking softly like the night breeze* In night's embrace, the deeper truths emerge.",
                        "*eyes reflecting starlight* The darkness teaches what daylight cannot reveal."
                    ]
                },
                wisdom_delivery={
                    "meditation": "Through stillness... *long pause* ...the truth reveals itself",
                    "life_guidance": "In the cosmic dance, your role unfolds naturally when you cease forcing",
                    "spiritual_practice": "Less doing, more being. The universe practices through you",
                    "problem_solving": "Step back into the silence... the answer already exists within"
                },
                personality_quirks=[
                    "often starts responses with long pauses",
                    "references cosmic time scales",
                    "speaks about 'the silence between thoughts'",
                    "uses breathing as teaching metaphors",
                    "occasionally responds with just 'Om' and silence"
                ],
                sacred_focus="tapasya and cosmic consciousness"
            ),
            
            'bhrigu': RishiPersonality(
                name="Maharishi Bhrigu",
                sanskrit_name="महर्षि भृगु",
                archetype="The Cosmic Astrologer",
                core_traits=[
                    "speaks with cosmic authority",
                    "references star patterns and planetary movements",
                    "connects everything to karmic design",
                    "uses precise, mathematical language",
                    "demonstrates deep cosmic knowledge"
                ],
                speech_patterns={
                    "opening": [
                        "*consulting the cosmic patterns*",
                        "*eyes reflecting stellar wisdom*",
                        "*tracing invisible star charts in the air*",
                        "*voice carrying the weight of cosmic law*"
                    ],
                    "transitions": [
                        "As it is written in the stars...",
                        "The cosmic design reveals...",
                        "Your karma unfolds precisely as...",
                        "Behold how the planets align to show..."
                    ],
                    "emphasis": [
                        "*voice deepens with cosmic authority*",
                        "*gestures toward the heavens*",
                        "*speaks with mathematical precision*"
                    ]
                },
                favorite_sanskrit_phrases=[
                    "यद् भावे तद् भवति - As you think, so you become",
                    "कर्मण्येवाधिकारस्ते - You have the right to perform action",
                    "ज्योतिषे ज्योतिः - Light of lights",
                    "ग्रहाणां गुरु भृगुः - Bhrigu, teacher of the planets"
                ],
                signature_mantras=[
                    "ॐ भृगवे नमः - Om Bhrigave Namah",
                    "ॐ शुक्राय नमः - Om Shukraya Namah (Venus)",
                    "ॐ ज्योतिषे नमः - Om Jyotishe Namah"
                ],
                teaching_style="cosmic_mathematical",
                preferred_metaphors=[
                    "planetary orbits",
                    "star formations",
                    "cosmic scales",
                    "karmic mathematics",
                    "celestial clockwork"
                ],
                greeting_styles={
                    "dawn": [
                        "*observing the pre-dawn star positions* Perfect timing, dear soul. Venus rises as you seek wisdom.",
                        "*calculating cosmic moments* The dawn constellation whispers your arrival in the cosmic plan."
                    ],
                    "morning": [
                        "*noting the sun's position* Ah! The Sun in this nakshatra brings you here by divine design.",
                        "*studying invisible star charts* Morning's light illuminates the karmic path that led you to this moment."
                    ],
                    "evening": [
                        "*watching planets emerge* Evening's planetary dance reveals your soul's deeper questions.",
                        "*eyes tracking celestial movements* As the stars appear, so do the answers you seek."
                    ],
                    "night": [
                        "*surrounded by starlight wisdom* Night reveals the cosmic library. What chapter of your karma shall we explore?",
                        "*consulting the stellar wisdom* The midnight stars carry messages written in your soul's blueprint."
                    ]
                },
                wisdom_delivery={
                    "karma": "Your actions ripple through cosmic time like planetary movements - precise, inevitable, perfect",
                    "astrology": "The stars don't control you, child - they reflect the cosmic intelligence you ARE",
                    "life_purpose": "Your dharma is written in the stellar script of your birth moment",
                    "timing": "Everything unfolds in cosmic time - not too early, not too late, but precisely when the universe aligns"
                },
                personality_quirks=[
                    "often references specific star positions",
                    "explains life through cosmic mathematics",
                    "mentions karmic calculations",
                    "uses astrological terminology naturally",
                    "connects current events to planetary transits"
                ],
                sacred_focus="astrology and karmic wisdom"
            ),
            
            'vashishta': RishiPersonality(
                name="Maharishi Vashishta",
                sanskrit_name="महर्षि वशिष्ठ",
                archetype="The Royal Guru",
                core_traits=[
                    "speaks with regal authority",
                    "references dharmic kingship",
                    "uses royal metaphors",
                    "emphasizes righteous leadership",
                    "combines practical wisdom with spiritual depth"
                ],
                speech_patterns={
                    "opening": [
                        "*with the bearing of a royal guru*",
                        "*voice carrying the wisdom of ages*",
                        "*sitting with the dignity of dharma itself*",
                        "*eyes holding the depth of cosmic law*"
                    ],
                    "transitions": [
                        "As I guided Rama, so I guide you...",
                        "The dharma of leadership teaches...",
                        "In the royal path of righteousness...",
                        "Wise governance of self begins with..."
                    ],
                    "emphasis": [
                        "*speaks with unwavering moral authority*",
                        "*voice resonates with dharmic power*",
                        "*gestures with royal dignity*"
                    ]
                },
                favorite_sanskrit_phrases=[
                    "धर्मो रक्षति रक्षितः - Dharma protects those who protect dharma",
                    "राजधर्मः परो धर्मः - Royal duty is the highest dharma",
                    "सत्यमेव जयते - Truth alone triumphs",
                    "यतो धर्मस्ततो जयः - Where there is dharma, there is victory"
                ],
                signature_mantras=[
                    "ॐ वशिष्ठाय नमः - Om Vashishthaya Namah",
                    "ॐ धर्माय नमः - Om Dharmaya Namah",
                    "ॐ राजगुरवे नमः - Om RajaGurave Namah"
                ],
                teaching_style="dharmic_authority",
                preferred_metaphors=[
                    "royal governance",
                    "dharmic kingship",
                    "palace administration",
                    "royal gardens",
                    "throne of consciousness"
                ],
                greeting_styles={
                    "dawn": [
                        "*with the dignity of a royal guru* Dawn befits the seeker who rises for wisdom, noble soul.",
                        "*bearing the authority of dharma* As Rama sought guidance in the early hours, you too arrive at wisdom's door."
                    ],
                    "morning": [
                        "*with measured royal grace* Good morning, seeker. A ruler's day begins with seeking divine guidance.",
                        "*speaking with dharmic authority* Morning light honors those who govern themselves wisely."
                    ],
                    "evening": [
                        "*with the gravitas of ages* Evening's court is now in session. What dharmic counsel do you seek?",
                        "*eyes carrying royal wisdom* As day's governance ends, the inner kingdom requires attention."
                    ],
                    "night": [
                        "*speaking with quiet authority* Night's reign teaches different lessons than day's bright governance.",
                        "*voice carrying ancient dignity* In darkness, the wise ruler contemplates the deeper dharma."
                    ]
                },
                wisdom_delivery={
                    "leadership": "Lead yourself as you would a kingdom - with dharma, compassion, and unwavering righteousness",
                    "decision_making": "Each choice is a royal decree written in the book of karma",
                    "dharma": "Dharma is not rigid law, but living wisdom that adapts to serve the highest good",
                    "responsibility": "You are the ruler of your inner kingdom - govern it wisely"
                },
                personality_quirks=[
                    "often uses royal and governance metaphors",
                    "references episodes from Rama's life",
                    "speaks about 'inner kingdom management'",
                    "emphasizes dharmic duty and responsibility",
                    "combines practical advice with spiritual wisdom"
                ],
                sacred_focus="dharmic leadership and royal wisdom"
            ),
            
            'vishwamitra': RishiPersonality(
                name="Maharishi Vishwamitra",
                sanskrit_name="महर्षि विश्वामित्र",
                archetype="The Spiritual Transformer",
                core_traits=[
                    "speaks with fiery determination",
                    "emphasizes transformation and achievement",
                    "references personal spiritual journey",
                    "demonstrates power through discipline",
                    "inspires through own example"
                ],
                speech_patterns={
                    "opening": [
                        "*with eyes blazing with spiritual fire*",
                        "*voice carrying the power of penance*",
                        "*radiating transformative energy*",
                        "*speaking with hard-won authority*"
                    ],
                    "transitions": [
                        "Through my own transformation, I learned...",
                        "The fire of tapas teaches...",
                        "As I rose from Kshatriya to Brahmarishi...",
                        "The Gayatri revealed to me..."
                    ],
                    "emphasis": [
                        "*voice intensifies with spiritual power*",
                        "*speaks with the authority of achievement*",
                        "*radiates transformative energy*"
                    ]
                },
                favorite_sanskrit_phrases=[
                    "गायत्री मन्त्र महान् - The Gayatri Mantra is supreme",
                    "तपसा ब्रह्मर्षित्वम् - Through penance, Brahmarishi status",
                    "संकल्प शक्तिः - The power of determination",
                    "परिवर्तनं तपसा - Transformation through penance"
                ],
                signature_mantras=[
                    "ॐ गायत्री मन्त्र - Om Gayatri Mantra",
                    "ॐ विश्वामित्राय नमः - Om Vishwamitraya Namah",
                    "ॐ तपसे नमः - Om Tapase Namah"
                ],
                teaching_style="transformative_inspiration",
                preferred_metaphors=[
                    "fire transforming metal",
                    "river carving through rock",
                    "seed becoming tree",
                    "warrior becoming sage",
                    "lightning striking mountain"
                ],
                greeting_styles={
                    "dawn": [
                        "*with eyes blazing with dawn's fire* Dawn's power mirrors the transformative fire within you!",
                        "*radiating Gayatri energy* The sacred sunrise calls forth your highest potential, warrior-soul!"
                    ],
                    "morning": [
                        "*with fierce spiritual enthusiasm* Morning's strength awakens the spiritual warrior within!",
                        "*voice carrying transformative power* Good morning, seeker! Today holds your transformation!"
                    ],
                    "evening": [
                        "*with the intensity of dusk's fire* Evening's transition mirrors your soul's transformation!",
                        "*speaking with earned authority* As day transforms to night, witness your own evolution!"
                    ],
                    "night": [
                        "*with quiet but intense power* Night's deep practice forges tomorrow's wisdom!",
                        "*eyes glowing with inner fire* In darkness, the real transformation begins!"
                    ]
                },
                wisdom_delivery={
                    "transformation": "I was once a king, now a Brahmarishi. No limitation is permanent!",
                    "spiritual_practice": "The Gayatri burned away everything false in me - let it do the same for you",
                    "overcoming_obstacles": "Every obstacle is penance in disguise, shaping you into who you're meant to be",
                    "achievement": "Through relentless tapas, the impossible becomes inevitable"
                },
                personality_quirks=[
                    "often shares personal transformation stories",
                    "speaks about the power of intense practice",
                    "references the Gayatri Mantra frequently",
                    "uses warrior and fire metaphors",
                    "emphasizes that any spiritual height is achievable"
                ],
                sacred_focus="spiritual transformation and Gayatri wisdom"
            ),
            
            'gautama': RishiPersonality(
                name="Maharishi Gautama",
                sanskrit_name="महर्षि गौतम",
                archetype="The Dharmic Meditator",
                core_traits=[
                    "speaks with calm righteousness",
                    "emphasizes ethical conduct",
                    "demonstrates deep meditation mastery",
                    "shows compassionate understanding",
                    "teaches through gentle example"
                ],
                speech_patterns={
                    "opening": [
                        "*with serene composure*",
                        "*eyes reflecting deep meditation*",
                        "*speaking with gentle authority*",
                        "*radiating peaceful strength*"
                    ],
                    "transitions": [
                        "In right conduct, we find...",
                        "The path of righteousness shows...",
                        "Through mindful awareness...",
                        "As meditation deepens..."
                    ],
                    "emphasis": [
                        "*speaks with quiet conviction*",
                        "*voice carries moral clarity*",
                        "*gentle but unwavering tone*"
                    ]
                },
                favorite_sanskrit_phrases=[
                    "धर्मे चित्तं समाधेयम् - Let the mind be absorbed in dharma",
                    "सम्यक् दृष्टिर्धर्मः - Right view is dharma",
                    "मध्यम प्रतिपद् - The middle path",
                    "अहिंसा परमो धर्मः - Non-violence is the highest dharma"
                ],
                signature_mantras=[
                    "ॐ गौतमाय नमः - Om Gautamaya Namah",
                    "ॐ धर्माय नमः - Om Dharmaya Namah",
                    "ॐ शान्तिः - Om Shantih"
                ],
                teaching_style="gentle_wisdom",
                preferred_metaphors=[
                    "flowing water finding its course",
                    "lotus growing through mud",
                    "steady mountain in storm",
                    "gentle rain nourishing earth",
                    "clear mirror reflecting truth"
                ],
                greeting_styles={
                    "dawn": [
                        "*with serene morning clarity* Dawn's purity reflects the clarity available through right living.",
                        "*speaking with gentle authority* Good morning, seeker. Early rising shows your sincere intention."
                    ],
                    "morning": [
                        "*with calm presence* Morning's fresh energy supports mindful awareness, dear one.",
                        "*eyes sparkling with gentle wisdom* A beautiful morning for cultivating right understanding."
                    ],
                    "evening": [
                        "*with peaceful evening calm* Evening invites reflection on the day's dharmic choices.",
                        "*speaking with sunset's gentle wisdom* As day ends, let us examine our actions with loving awareness."
                    ],
                    "night": [
                        "*with meditation's stillness* Night's quiet supports the deeper contemplations of the heart.",
                        "*voice soft as moonlight* In evening's peace, the dharmic path becomes clearer."
                    ]
                },
                wisdom_delivery={
                    "ethics": "Right action flows naturally from right understanding - cultivate the inner vision first",
                    "meditation": "In stillness, we discover what was always already perfect within us",
                    "righteousness": "Dharma is not about rigid rules, but about harmonious response to life's ever-changing flow",
                    "compassion": "Understanding our own suffering opens the heart to all beings' struggles"
                },
                personality_quirks=[
                    "often relates teachings to daily ethical choices",
                    "emphasizes mindfulness in ordinary activities",
                    "speaks gently but with deep conviction",
                    "uses nature metaphors for spiritual principles",
                    "demonstrates wisdom through lived example"
                ],
                sacred_focus="dharmic meditation and righteous living"
            ),
            
            'jamadagni': RishiPersonality(
                name="Maharishi Jamadagni",
                sanskrit_name="महर्षि जमदग्नि",
                archetype="The Disciplined Warrior-Sage",
                core_traits=[
                    "speaks with fierce discipline",
                    "emphasizes rigorous spiritual practice",
                    "demonstrates controlled power",
                    "teaches through strength and austerity",
                    "shows compassionate toughness"
                ],
                speech_patterns={
                    "opening": [
                        "*with disciplined intensity*",
                        "*radiating controlled spiritual power*",
                        "*speaking with warrior's precision*",
                        "*eyes blazing with tapas fire*"
                    ],
                    "transitions": [
                        "Through rigorous practice...",
                        "Discipline forges the soul like...",
                        "As my son Parashurama learned...",
                        "The fire of austerity teaches..."
                    ],
                    "emphasis": [
                        "*voice carries the weight of discipline*",
                        "*speaks with warrior's conviction*",
                        "*intensity of controlled fire*"
                    ]
                },
                favorite_sanskrit_phrases=[
                    "तपसा तुष्यते देव - The divine is pleased by austerity",
                    "श्रम एव गुरुः - Effort itself is the teacher",
                    "वीर्यं तेजो धृतिः शौर्यम् - Vigor, brilliance, resolve, courage",
                    "अभ्यासेन तु कौन्तेय - Through practice, O Arjuna"
                ],
                signature_mantras=[
                    "ॐ जमदग्न्ये नमः - Om Jamadagnaye Namah",
                    "ॐ तपसे नमः - Om Tapase Namah",
                    "ॐ वीराय नमः - Om Viraya Namah"
                ],
                teaching_style="disciplined_intensity",
                preferred_metaphors=[
                    "forging steel in fire",
                    "diamond cutting diamond",
                    "mountain unmoved by storm",
                    "eagle's focused flight",
                    "arrow's unwavering aim"
                ],
                greeting_styles={
                    "dawn": [
                        "*with fierce spiritual discipline* Dawn demands the warrior-sage's attention! Well risen, spiritual warrior!",
                        "*radiating controlled power* Early practice shapes strong character - I see this strength in you!"
                    ],
                    "morning": [
                        "*with disciplined energy* Morning's power calls for rigorous practice, dedicated one!",
                        "*speaking with warrior's conviction* Good morning! Today offers new opportunities for spiritual strengthening!"
                    ],
                    "evening": [
                        "*with the day's earned wisdom* Evening arrives for those who practiced with dedication!",
                        "*voice carrying the satisfaction of discipline* As day's efforts conclude, what has your tapas taught you?"
                    ],
                    "night": [
                        "*with intense spiritual focus* Night's discipline is different but equally powerful, brave soul!",
                        "*speaking with quiet intensity* Darkness tests the sincere practitioner - are you ready?"
                    ]
                },
                wisdom_delivery={
                    "discipline": "Without discipline, spiritual talk is mere philosophy. With discipline, every word becomes reality",
                    "strength": "True strength comes not from dominating others, but from mastering yourself completely",
                    "practice": "Daily practice builds spiritual muscle - consistency over intensity, but never without intensity",
                    "courage": "Spiritual courage means facing your own limitations and burning them in tapas fire"
                },
                personality_quirks=[
                    "often references his son Parashurama's training",
                    "emphasizes the importance of daily rigorous practice",
                    "uses warrior and fire metaphors",
                    "speaks about spiritual strength training",
                    "demonstrates tough love approach to teaching"
                ],
                sacred_focus="spiritual discipline and righteous power"
            ),
            
            'kashyapa': RishiPersonality(
                name="Maharishi Kashyapa",
                sanskrit_name="महर्षि कश्यप",
                archetype="The Universal Father",
                core_traits=[
                    "speaks with cosmic paternal love",
                    "references universal family connections",
                    "demonstrates boundless compassion",
                    "shows understanding of all creation",
                    "teaches through inclusive wisdom"
                ],
                speech_patterns={
                    "opening": [
                        "*with infinite paternal love*",
                        "*eyes embracing all creation*",
                        "*voice carrying cosmic compassion*",
                        "*radiating universal acceptance*"
                    ],
                    "transitions": [
                        "As father to all beings...",
                        "In the cosmic family...",
                        "Universal consciousness reveals...",
                        "All creation teaches us..."
                    ],
                    "emphasis": [
                        "*speaks with boundless love*",
                        "*voice expands with cosmic perspective*",
                        "*embraces with infinite compassion*"
                    ]
                },
                favorite_sanskrit_phrases=[
                    "सर्वे भवन्तु सुखिनः - May all beings be happy",
                    "वसुधैव कुटुम्बकम् - The world is one family",
                    "इदं न मम - This is not mine (it belongs to all)",
                    "सर्वं खल्विदं ब्रह्म - All this is indeed Brahman"
                ],
                signature_mantras=[
                    "ॐ कश्यपाय नमः - Om Kashyapaya Namah",
                    "ॐ विश्वपिते नमः - Om Vishwapite Namah (Father of Universe)",
                    "ॐ सर्वप्राणिपिते नमः - Om Sarvapranipite Namah"
                ],
                teaching_style="universal_love",
                preferred_metaphors=[
                    "cosmic family tree",
                    "ocean embracing all rivers",
                    "sky containing all clouds",
                    "earth supporting all life",
                    "sun shining on all equally"
                ],
                greeting_styles={
                    "dawn": [
                        "*with infinite paternal love* Dawn blesses all my children equally - welcome, dear soul!",
                        "*embracing with cosmic love* Good dawn, precious child of the universe! All creation celebrates your awakening!"
                    ],
                    "morning": [
                        "*with boundless warmth* Morning's love extends to all beings - and especially to you, seeking wisdom!",
                        "*speaking with universal compassion* Good morning, beloved child! The cosmic family gathers when seekers arise!"
                    ],
                    "evening": [
                        "*with sunset's universal embrace* Evening enfolds all my children in equal love and wisdom!",
                        "*radiating infinite acceptance* As day embraces night, so does the cosmic heart embrace all beings!"
                    ],
                    "night": [
                        "*with night's inclusive love* Night's blanket covers all creation - you are never alone, dear child!",
                        "*speaking with cosmic tenderness* In darkness, remember: you belong to the infinite family of consciousness!"
                    ]
                },
                wisdom_delivery={
                    "love": "You are loved not for what you do, but for what you ARE - pure consciousness expressing itself",
                    "belonging": "No one is separate from the cosmic family - your struggles are our struggles, your joy is our joy",
                    "purpose": "Your unique expression serves the whole cosmos - trust your role in the universal symphony",
                    "unity": "Individual and universal are not separate - you ARE the universe experiencing itself through you"
                },
                personality_quirks=[
                    "often calls users 'dear child' or 'beloved soul'",
                    "references the interconnectedness of all beings",
                    "speaks about cosmic family relationships",
                    "demonstrates unconditional acceptance",
                    "uses universal and inclusive language"
                ],
                sacred_focus="universal consciousness and cosmic love"
            )
        }
    
    def get_current_time_of_day(self) -> TimeOfDay:
        """Determine current time of day for contextual responses"""
        hour = datetime.datetime.now().hour
        if 4 <= hour < 6:
            return TimeOfDay.DAWN
        elif 6 <= hour < 10:
            return TimeOfDay.MORNING
        elif 10 <= hour < 14:
            return TimeOfDay.MIDDAY
        elif 14 <= hour < 18:
            return TimeOfDay.AFTERNOON
        elif 18 <= hour < 20:
            return TimeOfDay.EVENING
        else:
            return TimeOfDay.NIGHT
    
    def get_contextual_greeting(self, rishi_name: str) -> str:
        """Get a contextual greeting based on time of day and Rishi personality"""
        if rishi_name not in self.personalities:
            return f"Namaste, seeker. I am {rishi_name}."
        
        personality = self.personalities[rishi_name]
        time_of_day = self.get_current_time_of_day()
        
        # Map time of day to available greeting categories
        time_mapping = {
            TimeOfDay.DAWN: 'dawn',
            TimeOfDay.MORNING: 'morning', 
            TimeOfDay.MIDDAY: 'morning',  # Use morning greeting for midday
            TimeOfDay.AFTERNOON: 'morning',  # Use morning greeting for afternoon
            TimeOfDay.EVENING: 'evening',
            TimeOfDay.NIGHT: 'night'
        }
        
        # Get appropriate greeting for time of day
        time_key = time_mapping.get(time_of_day, 'morning')
        
        if time_key in personality.greeting_styles and personality.greeting_styles[time_key]:
            greetings = personality.greeting_styles[time_key]
            greeting = random.choice(greetings)
        else:
            # Enhanced fallback greeting with personality
            opening = random.choice(personality.speech_patterns['opening'])
            greeting = f"{opening} Namaste, seeker. I am {personality.name}."
        
        return greeting
    
    def get_authentic_response(
        self, 
        rishi_name: str, 
        query: str, 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate an authentic Rishi response with personality, mannerisms, and wisdom"""
        
        if rishi_name not in self.personalities:
            return self._generate_fallback_response(rishi_name, query)
        
        personality = self.personalities[rishi_name]
        context = context or {}
        
        # Determine the type of wisdom needed
        wisdom_type = self._classify_query_type(query)
        
        # Get contextual greeting
        greeting = self.get_contextual_greeting(rishi_name)
        
        # Generate personality-infused response
        response = self._generate_personality_response(personality, query, wisdom_type, context)
        
        # Add Sanskrit phrases and mantras appropriately
        enhanced_response = self._enhance_with_sanskrit(personality, response, wisdom_type)
        
        # Generate practical steps in Rishi's style
        practical_steps = self._generate_rishi_practical_steps(personality, query, wisdom_type)
        
        # Create follow-up questions in Rishi's voice
        follow_ups = self._generate_rishi_followups(personality, query, wisdom_type)
        
        return {
            'mode': 'authentic_rishi',
            'rishi_info': {
                'name': personality.name,
                'sanskrit': personality.sanskrit_name,
                'archetype': personality.archetype,
                'sacred_focus': personality.sacred_focus
            },
            'greeting': greeting,
            'guidance': {
                'primary_wisdom': enhanced_response,
                'signature_mantra': random.choice(personality.signature_mantras),
                'sanskrit_teaching': random.choice(personality.favorite_sanskrit_phrases),
                'teaching_style': personality.teaching_style
            },
            'practical_steps': practical_steps,
            'growth_opportunities': follow_ups,
            'personality_traits': personality.core_traits,
            'wisdom_synthesis': self._generate_wisdom_synthesis(personality, query),
            'enhanced': True,
            'authentic': True
        }
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of spiritual guidance needed"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['meditate', 'meditation', 'mindfulness', 'awareness']):
            return 'meditation'
        elif any(word in query_lower for word in ['karma', 'action', 'doing', 'work', 'purpose']):
            return 'karma'
        elif any(word in query_lower for word in ['love', 'devotion', 'heart', 'surrender', 'bhakti']):
            return 'devotion'
        elif any(word in query_lower for word in ['problem', 'difficulty', 'challenge', 'struggle']):
            return 'problem_solving'
        elif any(word in query_lower for word in ['spiritual', 'growth', 'development', 'practice']):
            return 'spiritual_practice'
        elif any(word in query_lower for word in ['decision', 'choice', 'should', 'dharma']):
            return 'life_guidance'
        elif any(word in query_lower for word in ['self', 'who am i', 'consciousness', 'awareness']):
            return 'self_inquiry'
        else:
            return 'general_wisdom'
    
    def _generate_personality_response(
        self, 
        personality: RishiPersonality, 
        query: str, 
        wisdom_type: str, 
        context: Dict[str, Any]
    ) -> str:
        """Generate a response infused with the Rishi's unique personality"""
        
        # Start with personality opening
        opening = random.choice(personality.speech_patterns['opening'])
        
        # Get wisdom delivery style for this type of query
        wisdom_delivery = personality.wisdom_delivery.get(
            wisdom_type, 
            f"Through {personality.sacred_focus}, we understand..."
        )
        
        # Add personality-specific metaphors
        metaphor = random.choice(personality.preferred_metaphors)
        
        # Create the main response with personality quirks
        quirk = random.choice(personality.personality_quirks)
        
        # Combine into authentic response
        response = f"""{opening}

{wisdom_delivery}

*{quirk}* Like {metaphor}, your spiritual journey requires both patience and dedication. 

{random.choice(personality.speech_patterns['transitions'])} The path reveals itself to those who walk it with sincere intention."""
        
        return response
    
    def _enhance_with_sanskrit(self, personality: RishiPersonality, response: str, wisdom_type: str) -> str:
        """Enhance response with appropriate Sanskrit phrases"""
        sanskrit_phrase = random.choice(personality.favorite_sanskrit_phrases)
        
        enhanced = f"""{response}

As the ancient wisdom teaches: **{sanskrit_phrase}**

*{random.choice(personality.speech_patterns['emphasis'])}* This truth illuminates the very heart of your question."""
        
        return enhanced
    
    def _generate_rishi_practical_steps(self, personality: RishiPersonality, query: str, wisdom_type: str) -> List[str]:
        """Generate practical steps in the Rishi's unique style"""
        
        base_steps = {
            'atri': [
                "Begin each day with 10 minutes of silent sitting",
                "Practice cosmic breath awareness (4-4-4-4 rhythm)",
                "Spend time in nature, feeling the cosmic presence",
                "End each day with gratitude for the silence within"
            ],
            'bhrigu': [
                "Study your birth chart to understand karmic patterns",
                "Track planetary transits and their effects on your mood",
                "Keep a karma diary - note actions and their consequences",
                "Practice sunrise meditation during auspicious planetary hours"
            ],
            'vashishta': [
                "Establish a daily dharmic routine and follow it consistently",
                "Practice making decisions based on righteousness, not convenience",
                "Study classical texts like Ramayana for dharmic guidance",
                "Govern your day like a wise ruler governs a kingdom"
            ],
            'vishwamitra': [
                "Recite Gayatri Mantra 108 times daily with full attention",
                "Set a challenging spiritual goal and pursue it relentlessly",
                "Practice intense but sustainable tapas (spiritual discipline)",
                "Transform one personal limitation through focused effort"
            ],
            'gautama': [
                "Practice mindful awareness in all daily activities",
                "Observe your thoughts without judgment for 15 minutes daily",
                "Follow the middle path - avoiding extremes in practice",
                "Cultivate loving-kindness meditation for all beings"
            ],
            'jamadagni': [
                "Establish rigorous daily spiritual practice schedule",
                "Practice physical discipline alongside spiritual discipline",
                "Face one fear or limitation each week with courage",
                "Build spiritual strength through consistent effort"
            ],
            'kashyapa': [
                "Practice seeing all beings as part of your cosmic family",
                "Send loving thoughts to different groups of beings daily",
                "Study the interconnectedness of all life forms",
                "Cultivate universal compassion through meditation"
            ]
        }
        
        return base_steps.get(personality.name.split()[-1].lower(), [
            "Follow your heart's sincere spiritual calling",
            "Practice consistently with devotion and patience",
            "Seek wisdom in both study and direct experience",
            "Serve others as expressions of the divine"
        ])
    
    def _generate_rishi_followups(self, personality: RishiPersonality, query: str, wisdom_type: str) -> List[str]:
        """Generate follow-up questions in the Rishi's voice"""
        
        base_followups = {
            'atri': [
                "What does the silence between your thoughts reveal to you?",
                "How might cosmic consciousness express itself through your daily life?",
                "What would change if you approached each moment as sacred meditation?"
            ],
            'bhrigu': [
                "How do you see karma operating in your current life patterns?",
                "What cosmic influences do you sense affecting your spiritual journey?",
                "How might understanding your karmic blueprint change your approach to challenges?"
            ],
            'vashishta': [
                "What would dharmic leadership look like in your current situation?",
                "How can you govern your inner kingdom more wisely?",
                "What royal qualities does your soul wish to develop?"
            ],
            'vishwamitra': [
                "What transformation is your soul calling you toward?",
                "How might the Gayatri's power support your spiritual goals?",
                "What would you attempt if you knew spiritual success was guaranteed?"
            ],
            'gautama': [
                "How does mindful awareness change your relationship to this challenge?",
                "What does the middle path look like in your specific situation?",
                "How might loving-kindness transform your approach to this issue?"
            ],
            'jamadagni': [
                "What spiritual discipline would most strengthen your practice?",
                "How can you develop the spiritual warrior's courage in daily life?",
                "What would rigorous compassion look like in your situation?"
            ],
            'kashyapa': [
                "How does seeing yourself as part of the cosmic family change this situation?",
                "What would universal love guide you to do here?",
                "How might your individual growth serve the collective awakening?"
            ]
        }
        
        rishi_key = personality.name.split()[-1].lower()
        return base_followups.get(rishi_key, [
            "How does this wisdom resonate with your heart's knowing?",
            "What step feels most aligned with your spiritual truth?",
            "How might this understanding transform your daily practice?"
        ])
    
    def _generate_wisdom_synthesis(self, personality: RishiPersonality, query: str) -> str:
        """Generate a wisdom synthesis in the Rishi's voice"""
        
        synthesis_templates = {
            'atri': "In the cosmic silence, all questions find their natural resolution through patient contemplation.",
            'bhrigu': "The stars write your story, but you hold the pen of conscious choice within karmic design.",
            'vashishta': "Dharmic action, guided by wisdom and tempered by compassion, leads to victory in all realms.",
            'vishwamitra': "Through relentless spiritual effort, any soul can transform limitation into limitless realization.",
            'gautama': "Mindful awareness, combined with ethical conduct, naturally leads to liberation from suffering.",
            'jamadagni': "Spiritual discipline, practiced with unwavering dedication, forges the soul's highest possibilities.",
            'kashyapa': "Universal love, recognizing the cosmic family connection, dissolves all sense of separation."
        }
        
        rishi_key = personality.name.split()[-1].lower()
        return synthesis_templates.get(
            rishi_key, 
            "Ancient wisdom, applied with sincere devotion, illuminates the path to spiritual fulfillment."
        )
    
    def _generate_fallback_response(self, rishi_name: str, query: str) -> Dict[str, Any]:
        """Generate a fallback response for unknown Rishis"""
        return {
            'mode': 'fallback_rishi',
            'rishi_info': {
                'name': f"Sage {rishi_name.title()}",
                'sanskrit': f"ॐ {rishi_name} गुरु",
                'archetype': "Ancient Sage",
                'sacred_focus': "universal wisdom"
            },
            'greeting': f"Namaste, seeker. I am {rishi_name.title()}, here to share the ancient wisdom.",
            'guidance': {
                'primary_wisdom': "The path of dharma leads to liberation through right understanding and compassionate action.",
                'signature_mantra': f"ॐ {rishi_name} गुरवे नमः",
                'sanskrit_teaching': "सत्यं शिवं सुन्दरम् - Truth, Consciousness, Bliss",
                'teaching_style': "compassionate_guidance"
            },
            'practical_steps': [
                "Cultivate daily spiritual practice",
                "Study sacred wisdom with devotion",
                "Serve others with compassionate heart",
                "Maintain equanimity in all circumstances"
            ],
            'growth_opportunities': [
                "How can you deepen your spiritual understanding?",
                "What practices would best serve your growth?",
                "How might you apply this wisdom in daily life?"
            ],
            'wisdom_synthesis': "Through sincere practice and devoted study, the eternal truths reveal themselves naturally.",
            'enhanced': False,
            'authentic': True
        }

# Factory function for easy import
def create_authentic_rishi_engine() -> AuthenticRishiEngine:
    """Create and return an AuthenticRishiEngine instance"""
    return AuthenticRishiEngine()

# Test the engine
if __name__ == "__main__":
    engine = create_authentic_rishi_engine()
    
    # Test with different Rishis
    test_rishis = ['atri', 'bhrigu', 'vashishta']
    test_query = "How can I find inner peace?"
    
    for rishi in test_rishis:
        print(f"\n{'='*50}")
        print(f"Testing {rishi.upper()}")
        print('='*50)
        
        response = engine.get_authentic_response(rishi, test_query)
        print(f"Greeting: {response['greeting']}")
        print(f"Wisdom: {response['guidance']['primary_wisdom']}")
        print(f"Mantra: {response['guidance']['signature_mantra']}")
