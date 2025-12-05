#!/usr/bin/env python3
"""
🔥 Vedic Ritual Engine - Authentic Ceremony & Worship System
===========================================================

Comprehensive system for authentic Vedic rituals, pujas, yajnas,
and daily worship practices based on traditional Vedic procedures.

Features:
- Complete puja vidhi (worship procedures)
- Yajna/Homa ceremony guidance
- Ritual purification protocols
- Vedic calendar integration
- Astrological timing (muhurta)
"""

import logging
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RitualType(Enum):
    """Types of Vedic rituals"""

    DAILY_PUJA = "daily_puja"
    YAJNA = "yajna"
    HOMA = "homa"
    SANDHYA_VANDANA = "sandhya_vandana"
    ARATI = "arati"
    ABHISHEKA = "abhisheka"
    VRATA = "vrata"
    SAMSKARA = "samskara"


class PujaStep(Enum):
    """Steps in traditional puja"""

    DHYANA = "dhyana"  # Meditation
    AVAHANA = "avahana"  # Invocation
    ASANA = "asana"  # Offering seat
    PADYA = "padya"  # Washing feet
    ARGHYA = "arghya"  # Water offering
    ACHAMANIYA = "achamaniya"  # Sipping water
    SNANA = "snana"  # Bathing
    VASTRA = "vastra"  # Clothing
    YAJNOPAVITA = "yajnopavita"  # Sacred thread
    GANDHA = "gandha"  # Fragrance
    PUSHPA = "pushpa"  # Flowers
    DHUPA = "dhupa"  # Incense
    DIPA = "dipa"  # Lamp
    NAIVEDYA = "naivedya"  # Food offering
    TAMBULA = "tambula"  # Betel leaves
    DAKSINA = "daksina"  # Monetary offering
    ARATI_FINAL = "arati_final"  # Final lamp ceremony
    PRADAKSHINA = "pradakshina"  # Circumambulation
    NAMASKARA = "namaskara"  # Prostration
    VISARJANA = "visarjana"  # Farewell


class Muhurta(Enum):
    """Auspicious timings"""

    BRAHMA_MUHURTA = "brahma_muhurta"  # 4-6 AM
    PRATAH_KALA = "pratah_kala"  # 6-10 AM
    MADHYAHNA = "madhyahna"  # 12-1 PM
    APARAHNA = "aparahna"  # 1-3 PM
    SAYAM_KALA = "sayam_kala"  # 6-7 PM
    PRADOSHA = "pradosha"  # 7:30-9 PM


@dataclass
class RitualItem:
    """Item required for ritual"""

    name: str
    sanskrit_name: str
    purpose: str
    preparation: str
    alternative: Optional[str] = None


@dataclass
class MantraSequence:
    """Mantra sequence for ritual step"""

    step: PujaStep
    mantras: List[str]
    actions: List[str]
    visualization: str
    duration: str


@dataclass
class RitualProcedure:
    """Complete ritual procedure"""

    ritual_type: RitualType
    name: str
    purpose: str
    deity: str
    required_items: List[RitualItem]
    preparation_steps: List[str]
    main_sequence: List[MantraSequence]
    completion_steps: List[str]
    benefits: List[str]
    restrictions: List[str]
    ideal_timing: List[Muhurta]


class VedicRitualEngine:
    """
    🔥 Authentic Vedic Ritual & Ceremony System

    Provides step-by-step guidance for traditional Vedic worship,
    ceremonies, and spiritual practices.
    """

    def __init__(self):
        self.ritual_database = self._initialize_ritual_database()
        self.puja_items = self._initialize_puja_items()
        self.mantra_sequences = self._initialize_mantra_sequences()
        self.muhurta_calculator = self._initialize_muhurta_system()
        logger.info("🔥 Vedic Ritual Engine initialized")

    def get_daily_puja_guide(self, deity: str = "Ganesha") -> RitualProcedure:
        """Get complete daily puja procedure"""

        return RitualProcedure(
            ritual_type=RitualType.DAILY_PUJA,
            name=f"Daily Puja to Lord {deity}",
            purpose="Daily worship for spiritual growth and divine blessing",
            deity=deity,
            required_items=self._get_daily_puja_items(),
            preparation_steps=[
                "Wake up during Brahma Muhurta (4-6 AM)",
                "Complete personal cleanliness (bath, clean clothes)",
                "Clean the puja space and arrange items",
                "Light a lamp and incense",
                "Sit facing east or north in meditation posture",
            ],
            main_sequence=self._get_daily_puja_sequence(deity),
            completion_steps=[
                "Offer final arati with deep devotion",
                "Circumambulate the deity 3 times",
                "Prostrate with complete surrender",
                "Distribute prasadam to family members",
                "Maintain silence for few minutes in gratitude",
            ],
            benefits=[
                "Purification of mind and environment",
                "Increase in devotion and spiritual awareness",
                "Protection from negative influences",
                "Blessing for the day's activities",
                "Strengthening connection with the divine",
            ],
            restrictions=[
                "Avoid eating before completion",
                "Maintain cleanliness throughout",
                "Avoid distractions and conversations",
                "Women during menstruation may offer mental worship",
            ],
            ideal_timing=[Muhurta.BRAHMA_MUHURTA, Muhurta.PRATAH_KALA],
        )

    def get_yajna_procedure(
        self, yajna_type: str = "Agnihotra"
    ) -> RitualProcedure:
        """Get complete yajna procedure"""

        return RitualProcedure(
            ritual_type=RitualType.YAJNA,
            name=f"{yajna_type} Yajna",
            purpose="Fire sacrifice for cosmic harmony and" +
                    "spiritual purification",
            deity="Agni (Fire God)",
            required_items=self._get_yajna_items(),
            preparation_steps=[
                "Fast for specified period before yajna",
                "Prepare sacred fire pit (kunda) according to specifications",
                "Arrange sacrificial materials (ghee, grains, herbs)",
                "Invoke Agni with proper mantras",
                "Purify self and space with water and mantras",
            ],
            main_sequence=self._get_yajna_sequence(yajna_type),
            completion_steps=[
                "Offer final oblations with gratitude",
                "Allow fire to consume all offerings completely",
                "Collect sacred ash (bhasma) for spiritual use",
                "Thank all deities and cosmic forces",
                "Distribute remaining prasadam",
            ],
            benefits=[
                "Purification of atmosphere and environment",
                "Cosmic harmony and planetary peace",
                "Spiritual evolution and karmic cleansing",
                "Divine protection and blessings",
                "Connection with Vedic tradition",
            ],
            restrictions=[
                "Strict fasting and purity requirements",
                "Proper pronunciation of mantras essential",
                "Fire safety and environmental considerations",
                "Qualified guidance recommended for major yajnas",
            ],
            ideal_timing=[Muhurta.BRAHMA_MUHURTA, Muhurta.SAYAM_KALA],
        )

    def get_sandhya_vandana_guide(
        self, varna: str = "general"
    ) -> RitualProcedure:
        """Get complete Sandhya Vandana procedure"""

        return RitualProcedure(
            ritual_type=RitualType.SANDHYA_VANDANA,
            name="Sandhya Vandana (Twilight Worship)",
            purpose="Daily obligation for spiritual purification and" +
                    "divine connection",
            deity="Savitri (Sun God)",
            required_items=[
                RitualItem(
                    "Water",
                    "Jalam",
                    "Purification",
                    "Clean river or well water",
                ),
                RitualItem(
                    "Sacred thread",
                    "Yajnopavita",
                    "Spiritual purity",
                    "Cotton thread blessed by guru",
                ),
                RitualItem(
                    "Tilaka materials",
                    "Tilaka dravya",
                    "Spiritual marking",
                    "Sandalwood, ash, or clay",
                ),
                RitualItem(
                    "Rudraksha mala",
                    "Rudraksha mala",
                    "Chanting",
                    "108-bead counting",
                ),
            ],
            preparation_steps=[
                "Face east during morning, west during evening",
                "Wear clean clothes and sacred thread",
                "Apply tilaka marks on forehead and body",
                "Sit in comfortable meditation posture",
                "Begin with purification mantras",
            ],
            main_sequence=self._get_sandhya_sequence(),
            completion_steps=[
                "Complete with Gayatri mantra meditation",
                "Offer salutations to all directions",
                "Pray for welfare of all beings",
                "Touch the earth in gratitude",
                "Maintain silence for few minutes",
            ],
            benefits=[
                "Daily spiritual discipline and purification",
                "Connection with cosmic rhythms",
                "Mental clarity and emotional balance",
                "Protection from negative influences",
                "Fulfillment of dharmic obligations",
            ],
            restrictions=[
                "Must be performed at exact twilight times",
                "Complete focus and proper pronunciation required",
                "Sacred thread must be worn properly",
                "Avoid during eclipse periods",
            ],
            ideal_timing=[Muhurta.PRATAH_KALA, Muhurta.SAYAM_KALA],
        )

    def calculate_auspicious_timing(
        self, ritual_type: RitualType, date: date
    ) -> Dict[str, Any]:
        """Calculate auspicious timing for ritual"""

        timing_info = {
            "date": date.isoformat(),
            "ritual_type": ritual_type.value,
            "auspicious_periods": [],
            "inauspicious_periods": [],
            "best_timing": None,
            "lunar_phase": self._get_lunar_phase(date),
            "nakshatra": self._get_nakshatra(date),
            "tithi": self._get_tithi(date),
        }

        # Calculate timing based on ritual type
        if ritual_type == RitualType.DAILY_PUJA:
            timing_info["auspicious_periods"] = [
                {
                    "period": "Brahma Muhurta",
                    "time": "4:00-6:00 AM",
                    "significance": "Most sacred time",
                },
                {
                    "period": "Pratah Kala",
                    "time": "6:00-10:00 AM",
                    "significance": "Morning worship",
                },
                {
                    "period": "Sayam Kala",
                    "time": "6:00-7:00 PM",
                    "significance": "Evening prayers",
                },
            ]
            timing_info["best_timing"] = "Brahma Muhurta (4:00-6:00 AM)"

        elif ritual_type == RitualType.YAJNA:
            timing_info["auspicious_periods"] = [
                {
                    "period": "Brahma Muhurta",
                    "time": "4:00-6:00 AM",
                    "significance": "Sacred fire ceremony",
                },
                {
                    "period": "Pradosha",
                    "time": "7:30-9:00 PM",
                    "significance": "Evening yajna",
                },
            ]
            timing_info["best_timing"] = (
                "Based on lunar calendar and nakshatra"
            )

        return timing_info

    def _initialize_ritual_database(self) -> Dict[str, Any]:
        """Initialize comprehensive ritual database"""
        return {
            "daily_rituals": {
                "morning_prayers": "Complete morning spiritual routine",
                "sandhya_vandana": "Twilight worship obligations",
                "evening_prayers": "Complete evening spiritual routine",
            },
            "weekly_rituals": {
                "ekadashi_vrata": "Fasting on 11th lunar day",
                "pradosha_vrata": "13th lunar day observance",
            },
            "seasonal_rituals": {
                "navaratri": "Nine nights of divine mother worship",
                "shivaratri": "Great night of Lord Shiva",
                "diwali": "Festival of lights celebration",
            },
            "life_cycle_rituals": {
                "jatakarma": "Birth ceremony",
                "upanayana": "Sacred thread ceremony",
                "vivaha": "Marriage ceremony",
                "antyesti": "Final rites",
            },
        }

    def _initialize_puja_items(self) -> List[RitualItem]:
        """Initialize essential puja items"""
        return [
            RitualItem(
                "Copper water pot",
                "Kalasha",
                "Holy water container",
                "Fill with pure water",
            ),
            RitualItem(
                "Oil lamp", "Dipa", "Light offering", "Use sesame oil or ghee"
            ),
            RitualItem(
                "Incense sticks",
                "Dhupa",
                "Fragrance offering",
                "Natural ingredients only",
            ),
            RitualItem(
                "Flowers",
                "Pushpa",
                "Beauty offering",
                "Fresh, fragrant flowers",
            ),
            RitualItem(
                "Fruits", "Phala", "Food offering", "Fresh, seasonal fruits"
            ),
            RitualItem(
                "Sandalwood paste",
                "Chandana",
                "Cooling fragrance",
                "Pure sandalwood powder",
            ),
            RitualItem(
                "Sacred rice",
                "Akshata",
                "Prosperity offering",
                "Unbroken rice grains",
            ),
            RitualItem(
                "Bell", "Ghanta", "Sound offering", "Bronze or brass bell"
            ),
            RitualItem(
                "Conch shell",
                "Shankha",
                "Sacred sound",
                "Natural conch for blowing",
            ),
            RitualItem(
                "Sacred thread",
                "Yajnopavita",
                "Spiritual purity",
                "Three-strand cotton thread",
            ),
        ]

    def _get_daily_puja_items(self) -> List[RitualItem]:
        """Get items specifically for daily puja"""
        return [
            RitualItem(
                "Deity image/murti",
                "Murti",
                "Divine presence",
                "Consecrated image or statue",
            ),
            RitualItem(
                "Water vessel",
                "Kalasha",
                "Purification",
                "Copper or silver vessel",
            ),
            RitualItem(
                "Flowers",
                "Pushpa",
                "Devotional offering",
                "Fresh flowers like jasmine, rose",
            ),
            RitualItem(
                "Incense",
                "Dhupa",
                "Fragrant offering",
                "Natural sandalwood incense",
            ),
            RitualItem(
                "Oil lamp", "Dipa", "Light offering", "Ghee or sesame oil lamp"
            ),
            RitualItem(
                "Food offerings",
                "Naivedya",
                "Divine nourishment",
                "Fresh fruits, sweets, rice",
            ),
            RitualItem(
                "Sacred rice",
                "Akshata",
                "Auspicious offering",
                "Turmeric-mixed rice",
            ),
            RitualItem(
                "Sandalwood paste",
                "Chandana",
                "Cooling application",
                "Pure sandalwood",
            ),
            RitualItem(
                "Bell", "Ghanta", "Divine attention", "Bronze bell for arati"
            ),
        ]

    def _get_yajna_items(self) -> List[RitualItem]:
        """Get items specifically for yajna"""
        return [
            RitualItem(
                "Fire pit",
                "Homa kunda",
                "Sacred fire space",
                "Square or round brick structure",
            ),
            RitualItem("Ghee", "Ghrita", "Fire offering", "Pure cow ghee"),
            RitualItem(
                "Sacred wood",
                "Samidha",
                "Fire fuel",
                "Specific woods like mango, pipal",
            ),
            RitualItem(
                "Grains",
                "Dhanya",
                "Abundance offering",
                "Rice, barley, sesame",
            ),
            RitualItem(
                "Herbs",
                "Aushadhi",
                "Healing offering",
                "Specific medicinal herbs",
            ),
            RitualItem("Honey", "Madhu", "Sweetness offering", "Pure honey"),
            RitualItem("Milk", "Ksheera", "Purity offering", "Fresh cow milk"),
            RitualItem(
                "Sacred thread",
                "Yajnopavita",
                "Purification",
                "Sanctified thread",
            ),
            RitualItem(
                "Water vessel",
                "Kalasha",
                "Purification water",
                "Copper vessel with sacred water",
            ),
        ]

    def _get_daily_puja_sequence(self, deity: str) -> List[MantraSequence]:
        """Get mantra sequence for daily puja"""
        return [
            MantraSequence(
                step=PujaStep.DHYANA,
                mantras=[
                    "ॐ गं गणपतये नमः",
                    f"Meditation mantra for Lord {deity}",
                ],
                actions=["Sit in meditation", "Visualize deity"],
                visualization=f"Visualize Lord {deity} in radiant form",
                duration="2-3 minutes",
            ),
            MantraSequence(
                step=PujaStep.AVAHANA,
                mantras=[
                    "आगच्छ देव देवेश आगच्छ परमेश्वर",
                    f"Invocation mantras for {deity}",
                ],
                actions=["Ring bell", "Offer water"],
                visualization="Invite deity to be present",
                duration="1-2 minutes",
            ),
            MantraSequence(
                step=PujaStep.PUSHPA,
                mantras=["ॐ पुष्पं समर्पयामि", "Flower offering mantras"],
                actions=["Offer flowers", "Express devotion"],
                visualization="Deity accepting flower offerings",
                duration="2-3 minutes",
            ),
            MantraSequence(
                step=PujaStep.ARATI_FINAL,
                mantras=["आरती mantras", "ॐ जय जगदीश हरे"],
                actions=["Wave lamp", "Ring bell"],
                visualization="Divine light blessing all",
                duration="3-5 minutes",
            ),
        ]

    def _get_yajna_sequence(self, yajna_type: str) -> List[MantraSequence]:
        """Get mantra sequence for yajna"""
        return [
            MantraSequence(
                step=PujaStep.DHYANA,
                mantras=["ॐ अग्नये नमः", "Agni invocation mantras"],
                actions=["Light fire", "Offer ghee"],
                visualization="Sacred fire as divine presence",
                duration="5 minutes",
            ),
            MantraSequence(
                step=PujaStep.NAIVEDYA,
                mantras=["Oblation mantras", "स्वाहा"],
                actions=["Offer grains", "Pour ghee"],
                visualization="Offerings ascending to gods",
                duration="15-30 minutes",
            ),
        ]

    def _get_sandhya_sequence(self) -> List[MantraSequence]:
        """Get Sandhya Vandana sequence"""
        return [
            MantraSequence(
                step=PujaStep.ACHAMANIYA,
                mantras=["ॐ केशवाय नमः", "Water purification mantras"],
                actions=["Sip water thrice", "Purify self"],
                visualization="Inner purification",
                duration="2 minutes",
            ),
            MantraSequence(
                step=PujaStep.DHYANA,
                mantras=["ॐ भूर्भुवः स्वः", "Gayatri mantra"],
                actions=["Meditate on sun", "Chant Gayatri"],
                visualization="Divine light within",
                duration="10-15 minutes",
            ),
        ]

    def _initialize_muhurta_system(self) -> Dict[str, Any]:
        """Initialize auspicious timing system"""
        return {
            "daily_muhurtas": {
                "brahma_muhurta": {
                    "start": "4:00",
                    "end": "6:00",
                    "significance": "Most sacred",
                },
                "pratah_kala": {
                    "start": "6:00",
                    "end": "10:00",
                    "significance": "Morning worship",
                },
                "madhyahna": {
                    "start": "12:00",
                    "end": "13:00",
                    "significance": "Noon prayers",
                },
                "sayam_kala": {
                    "start": "18:00",
                    "end": "19:00",
                    "significance": "Evening worship",
                },
            },
            "inauspicious_periods": {
                "rahu_kala": "Inauspicious period ruled by Rahu",
                "yamaganda": "Period ruled by Yama",
                "gulika": "Period ruled by Saturn",
            },
        }

    def _get_lunar_phase(self, date: date) -> str:
        """Calculate lunar phase (simplified)"""
        # Simplified lunar phase calculation
        day_of_month = date.day
        if 1 <= day_of_month <= 7:
            return "Waxing Moon (Shukla Paksha)"
        elif 8 <= day_of_month <= 15:
            return "Waxing Moon (Shukla Paksha)"
        elif 16 <= day_of_month <= 22:
            return "Waning Moon (Krishna Paksha)"
        else:
            return "Waning Moon (Krishna Paksha)"

    def _get_nakshatra(self, date: date) -> str:
        """Calculate nakshatra (simplified)"""
        # Simplified nakshatra calculation
        nakshatras = [
            "Ashwini",
            "Bharani",
            "Krittika",
            "Rohini",
            "Mrigashira",
            "Ardra",
            "Punarvasu",
            "Pushya",
            "Ashlesha",
            "Magha",
            "Purva Phalguni",
            "Uttara Phalguni",
            "Hasta",
            "Chitra",
            "Swati",
            "Vishakha",
            "Anuradha",
            "Jyeshtha",
            "Mula",
            "Purva Ashadha",
            "Uttara Ashadha",
            "Shravana",
            "Dhanishta",
            "Shatabhisha",
            "Purva Bhadrapada",
            "Uttara Bhadrapada",
            "Revati",
        ]
        return nakshatras[date.day % 27]

    def _get_tithi(self, date: date) -> str:
        """Calculate tithi (lunar day)"""
        # Simplified tithi calculation
        tithis = [
            "Pratipada",
            "Dwitiya",
            "Tritiya",
            "Chaturthi",
            "Panchami",
            "Shashthi",
            "Saptami",
            "Ashtami",
            "Navami",
            "Dashami",
            "Ekadashi",
            "Dwadashi",
            "Trayodashi",
            "Chaturdashi",
            "Purnima/Amavasya",
        ]
        return tithis[date.day % 15]


# Global instance
_vedic_ritual_engine = None


def get_vedic_ritual_engine() -> VedicRitualEngine:
    """Get global Vedic Ritual Engine instance"""
    global _vedic_ritual_engine
    if _vedic_ritual_engine is None:
        _vedic_ritual_engine = VedicRitualEngine()
    return _vedic_ritual_engine
