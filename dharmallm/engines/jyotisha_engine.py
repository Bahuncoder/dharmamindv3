#!/usr/bin/env python3
"""
🌟 Advanced Jyotisha Engine - Vedic Astrology System
===================================================

Comprehensive Vedic astrology system providing authentic astrological
analysis, predictions, remedies, and spiritual guidance based on
traditional Jyotisha principles.

Features:
- Complete horoscope calculation and analysis
- Muhurta (auspicious timing) calculations
- Dasha (planetary periods) analysis
- Gemstone and remedy recommendations
- Karma analysis through planetary positions
- Transit predictions and their effects
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class Planet(Enum):
    """Nine planets in Vedic astrology"""

    SUN = "surya"  # Surya
    MOON = "chandra"  # Chandra
    MARS = "mangal"  # Mangal
    MERCURY = "budha"  # Budha
    JUPITER = "guru"  # Guru/Brihaspati
    VENUS = "shukra"  # Shukra
    SATURN = "shani"  # Shani
    RAHU = "rahu"  # North Node
    KETU = "ketu"  # South Node


class Zodiac(Enum):
    """Twelve zodiac signs"""

    ARIES = "mesha"  # Mesha
    TAURUS = "vrishabha"  # Vrishabha
    GEMINI = "mithuna"  # Mithuna
    CANCER = "karkataka"  # Karkataka
    LEO = "simha"  # Simha
    VIRGO = "kanya"  # Kanya
    LIBRA = "tula"  # Tula
    SCORPIO = "vrishchika"  # Vrishchika
    SAGITTARIUS = "dhanu"  # Dhanu
    CAPRICORN = "makara"  # Makara
    AQUARIUS = "kumbha"  # Kumbha
    PISCES = "meena"  # Meena


class House(Enum):
    """Twelve houses in Vedic astrology"""

    FIRST = 1  # Lagna - Self, personality
    SECOND = 2  # Dhana - Wealth, speech
    THIRD = 3  # Sahaja - Siblings, courage
    FOURTH = 4  # Sukha - Home, mother
    FIFTH = 5  # Putra - Children, education
    SIXTH = 6  # Ari - Enemies, health
    SEVENTH = 7  # Kalatra - Spouse, partnership
    EIGHTH = 8  # Ayu - Longevity, transformation
    NINTH = 9  # Dharma - Religion, father
    TENTH = 10  # Karma - Career, reputation
    ELEVENTH = 11  # Labha - Gains, friends
    TWELFTH = 12  # Vyaya - Loss, moksha


class Nakshatra(Enum):
    """27 Nakshatras (lunar mansions)"""

    ASHWINI = "ashwini"
    BHARANI = "bharani"
    KRITTIKA = "krittika"
    ROHINI = "rohini"
    MRIGASHIRA = "mrigashira"
    ARDRA = "ardra"
    PUNARVASU = "punarvasu"
    PUSHYA = "pushya"
    ASHLESHA = "ashlesha"
    MAGHA = "magha"
    PURVA_PHALGUNI = "purva_phalguni"
    UTTARA_PHALGUNI = "uttara_phalguni"
    HASTA = "hasta"
    CHITRA = "chitra"
    SWATI = "swati"
    VISHAKHA = "vishakha"
    ANURADHA = "anuradha"
    JYESHTHA = "jyeshtha"
    MULA = "mula"
    PURVA_ASHADHA = "purva_ashadha"
    UTTARA_ASHADHA = "uttara_ashadha"
    SHRAVANA = "shravana"
    DHANISHTA = "dhanishta"
    SHATABHISHA = "shatabhisha"
    PURVA_BHADRAPADA = "purva_bhadrapada"
    UTTARA_BHADRAPADA = "uttara_bhadrapada"
    REVATI = "revati"


@dataclass
class PlanetaryPosition:
    """Position of a planet in the horoscope"""

    planet: Planet
    longitude: float  # Degrees from 0-360
    zodiac_sign: Zodiac
    house: House
    nakshatra: Nakshatra
    nakshatra_pada: int  # 1-4
    retrograde: bool = False


@dataclass
class Horoscope:
    """Complete Vedic horoscope"""

    birth_datetime: datetime
    birth_place: str
    latitude: float
    longitude: float
    lagna: Zodiac  # Ascendant sign
    planetary_positions: List[PlanetaryPosition]
    house_cusps: List[float]  # Starting degrees of each house
    ayanamsa: float  # Precession correction


@dataclass
class MuhurtaRecommendation:
    """Auspicious timing recommendation"""

    activity: str
    recommended_time: datetime
    duration: timedelta
    planetary_support: List[Planet]
    beneficial_nakshatras: List[Nakshatra]
    reasons: List[str]
    precautions: List[str]


@dataclass
class Remedy:
    """Astrological remedy"""

    remedy_type: str  # Gemstone, mantra, charity, etc.
    description: str
    planet_benefited: Planet
    implementation: str
    duration: str
    benefits: List[str]
    cost_estimate: Optional[str] = None


class AdvancedJyotishaEngine:
    """
    🌟 Comprehensive Vedic Astrology Engine

    Provides authentic Jyotisha analysis, predictions, and remedies
    based on traditional Vedic astrology principles.
    """

    def __init__(self):
        self.planetary_data = self._initialize_planetary_data()
        self.nakshatra_data = self._initialize_nakshatra_data()
        self.remedies_database = self._initialize_remedies_database()
        self.muhurta_rules = self._initialize_muhurta_rules()
        logger.info("🌟 Advanced Jyotisha Engine initialized")

    def calculate_horoscope(
        self,
        birth_datetime: datetime,
        birth_place: str,
        latitude: float,
        longitude: float,
    ) -> Horoscope:
        """Calculate complete Vedic horoscope"""

        # Calculate ayanamsa (precession correction)
        ayanamsa = self._calculate_ayanamsa(birth_datetime)

        # Calculate planetary positions
        planetary_positions = []
        for planet in Planet:
            position = self._calculate_planet_position(planet, birth_datetime, ayanamsa)
            planetary_positions.append(position)

        # Calculate lagna (ascendant)
        lagna = self._calculate_lagna(birth_datetime, latitude, longitude, ayanamsa)

        # Calculate house cusps
        house_cusps = self._calculate_house_cusps(lagna, birth_datetime)

        return Horoscope(
            birth_datetime=birth_datetime,
            birth_place=birth_place,
            latitude=latitude,
            longitude=longitude,
            lagna=lagna,
            planetary_positions=planetary_positions,
            house_cusps=house_cusps,
            ayanamsa=ayanamsa,
        )

    def analyze_personality(self, horoscope: Horoscope) -> Dict[str, Any]:
        """Analyze personality from horoscope"""

        lagna_lord = self._get_sign_lord(horoscope.lagna)
        moon_sign = self._find_planet_sign(horoscope, Planet.MOON)
        sun_sign = self._find_planet_sign(horoscope, Planet.SUN)

        analysis = {
            "basic_nature": {
                "lagna_sign": horoscope.lagna.value,
                "lagna_lord": lagna_lord.value,
                "moon_sign": moon_sign.value,
                "sun_sign": sun_sign.value,
                "dominant_elements": self._analyze_elemental_balance(horoscope),
            },
            "personality_traits": self._get_personality_traits(horoscope),
            "strengths": self._analyze_strengths(horoscope),
            "challenges": self._analyze_challenges(horoscope),
            "life_purpose": self._analyze_life_purpose(horoscope),
            "spiritual_inclinations": self._analyze_spiritual_nature(horoscope),
        }

        return analysis

    def calculate_muhurta(
        self,
        activity: str,
        preferred_date: date,
        location: Tuple[float, float],
    ) -> List[MuhurtaRecommendation]:
        """Calculate auspicious timing for activity"""

        recommendations = []

        # Check multiple days around preferred date
        for days_offset in range(-2, 3):
            check_date = preferred_date + timedelta(days=days_offset)
            daily_muhurtas = self._calculate_daily_muhurtas(
                activity, check_date, location
            )
            recommendations.extend(daily_muhurtas)

        # Sort by auspiciousness score
        recommendations.sort(
            key=lambda x: self._calculate_auspiciousness_score(x), reverse=True
        )

        return recommendations[:5]  # Return top 5 recommendations

    def recommend_remedies(
        self, horoscope: Horoscope, specific_issues: List[str] = None
    ) -> List[Remedy]:
        """Recommend astrological remedies"""

        remedies = []

        # Analyze planetary weaknesses
        weak_planets = self._identify_weak_planets(horoscope)

        for planet in weak_planets:
            planet_remedies = self._get_planet_remedies(planet, horoscope)
            remedies.extend(planet_remedies)

        # Add specific issue remedies
        if specific_issues:
            for issue in specific_issues:
                issue_remedies = self._get_issue_specific_remedies(issue, horoscope)
                remedies.extend(issue_remedies)

        return remedies

    def analyze_career_prospects(self, horoscope: Horoscope) -> Dict[str, Any]:
        """Analyze career prospects from horoscope"""

        tenth_house_analysis = self._analyze_house(horoscope, House.TENTH)
        tenth_lord = self._get_house_lord(horoscope, House.TENTH)

        return {
            "career_indicators": {
                "tenth_house_sign": tenth_house_analysis["sign"],
                "tenth_lord": tenth_lord.value,
                "planets_in_tenth": tenth_house_analysis["planets"],
                "tenth_lord_position": self._find_planet_house(horoscope, tenth_lord),
            },
            "suitable_professions": self._get_suitable_professions(horoscope),
            "career_timing": self._analyze_career_timing(horoscope),
            "business_vs_job": self._analyze_business_indicators(horoscope),
            "wealth_prospects": self._analyze_wealth_indicators(horoscope),
            "reputation_factors": self._analyze_reputation_factors(horoscope),
        }

    def analyze_relationships(self, horoscope: Horoscope) -> Dict[str, Any]:
        """Analyze relationship prospects"""

        seventh_house = self._analyze_house(horoscope, House.SEVENTH)
        venus_analysis = self._analyze_planet_strength(horoscope, Planet.VENUS)
        mars_analysis = self._analyze_planet_strength(horoscope, Planet.MARS)

        return {
            "marriage_indicators": {
                "seventh_house_analysis": seventh_house,
                "venus_strength": venus_analysis,
                "mars_influence": mars_analysis,
                "marriage_timing": self._predict_marriage_timing(horoscope),
            },
            "spouse_characteristics": self._predict_spouse_nature(horoscope),
            "relationship_challenges": self._identify_relationship_challenges(
                horoscope
            ),
            "compatibility_factors": self._analyze_compatibility_factors(horoscope),
            "remedies_for_marriage": self._get_marriage_remedies(horoscope),
        }

    def predict_major_periods(self, horoscope: Horoscope) -> Dict[str, Any]:
        """Predict major planetary periods (Dasha)"""

        # Calculate Vimshottari Dasha
        moon_nakshatra = self._find_planet_nakshatra(horoscope, Planet.MOON)
        moon_position_in_nakshatra = self._calculate_nakshatra_position(
            horoscope, Planet.MOON
        )

        dasha_sequence = self._calculate_vimshottari_dasha(
            moon_nakshatra, moon_position_in_nakshatra
        )

        current_dasha = self._get_current_dasha(dasha_sequence, datetime.now())

        return {
            "current_major_period": current_dasha,
            "upcoming_periods": dasha_sequence[:10],  # Next 10 periods
            "period_predictions": self._predict_dasha_effects(
                horoscope, dasha_sequence[:5]
            ),
            "important_transitions": self._identify_important_transitions(
                dasha_sequence
            ),
            "remedial_measures": self._get_dasha_remedies(current_dasha, horoscope),
        }

    def _initialize_planetary_data(self) -> Dict[Planet, Dict[str, Any]]:
        """Initialize planetary characteristics"""
        return {
            Planet.SUN: {
                "nature": "malefic",
                "element": "fire",
                "significations": [
                    "soul",
                    "father",
                    "authority",
                    "government",
                    "health",
                ],
                "exaltation": Zodiac.ARIES,
                "debilitation": Zodiac.LIBRA,
                "own_signs": [Zodiac.LEO],
                "friendly_planets": [Planet.MOON, Planet.MARS, Planet.JUPITER],
                "enemy_planets": [Planet.VENUS, Planet.SATURN],
                "neutral_planets": [Planet.MERCURY],
            },
            Planet.MOON: {
                "nature": "benefic",
                "element": "water",
                "significations": [
                    "mind",
                    "mother",
                    "emotions",
                    "public",
                    "water",
                ],
                "exaltation": Zodiac.TAURUS,
                "debilitation": Zodiac.SCORPIO,
                "own_signs": [Zodiac.CANCER],
                "friendly_planets": [Planet.SUN, Planet.MERCURY],
                "enemy_planets": [],
                "neutral_planets": [
                    Planet.MARS,
                    Planet.JUPITER,
                    Planet.VENUS,
                    Planet.SATURN,
                ],
            },
            Planet.MARS: {
                "nature": "malefic",
                "element": "fire",
                "significations": [
                    "energy",
                    "brothers",
                    "land",
                    "courage",
                    "surgery",
                ],
                "exaltation": Zodiac.CAPRICORN,
                "debilitation": Zodiac.CANCER,
                "own_signs": [Zodiac.ARIES, Zodiac.SCORPIO],
                "friendly_planets": [Planet.SUN, Planet.MOON, Planet.JUPITER],
                "enemy_planets": [Planet.MERCURY],
                "neutral_planets": [Planet.VENUS, Planet.SATURN],
            },
            Planet.MERCURY: {
                "nature": "benefic",
                "element": "earth",
                "significations": [
                    "intelligence",
                    "communication",
                    "business",
                    "education",
                ],
                "exaltation": Zodiac.VIRGO,
                "debilitation": Zodiac.PISCES,
                "own_signs": [Zodiac.GEMINI, Zodiac.VIRGO],
                "friendly_planets": [Planet.SUN, Planet.VENUS],
                "enemy_planets": [Planet.MOON],
                "neutral_planets": [
                    Planet.MARS,
                    Planet.JUPITER,
                    Planet.SATURN,
                ],
            },
            Planet.JUPITER: {
                "nature": "benefic",
                "element": "ether",
                "significations": [
                    "wisdom",
                    "guru",
                    "children",
                    "dharma",
                    "wealth",
                ],
                "exaltation": Zodiac.CANCER,
                "debilitation": Zodiac.CAPRICORN,
                "own_signs": [Zodiac.SAGITTARIUS, Zodiac.PISCES],
                "friendly_planets": [Planet.SUN, Planet.MOON, Planet.MARS],
                "enemy_planets": [Planet.MERCURY, Planet.VENUS],
                "neutral_planets": [Planet.SATURN],
            },
            Planet.VENUS: {
                "nature": "benefic",
                "element": "water",
                "significations": [
                    "love",
                    "beauty",
                    "luxury",
                    "spouse",
                    "arts",
                ],
                "exaltation": Zodiac.PISCES,
                "debilitation": Zodiac.VIRGO,
                "own_signs": [Zodiac.TAURUS, Zodiac.LIBRA],
                "friendly_planets": [Planet.MERCURY, Planet.SATURN],
                "enemy_planets": [Planet.SUN, Planet.MOON],
                "neutral_planets": [Planet.MARS, Planet.JUPITER],
            },
            Planet.SATURN: {
                "nature": "malefic",
                "element": "air",
                "significations": [
                    "discipline",
                    "delay",
                    "longevity",
                    "karma",
                    "service",
                ],
                "exaltation": Zodiac.LIBRA,
                "debilitation": Zodiac.ARIES,
                "own_signs": [Zodiac.CAPRICORN, Zodiac.AQUARIUS],
                "friendly_planets": [Planet.MERCURY, Planet.VENUS],
                "enemy_planets": [Planet.SUN, Planet.MOON, Planet.MARS],
                "neutral_planets": [Planet.JUPITER],
            },
            Planet.RAHU: {
                "nature": "malefic",
                "element": "air",
                "significations": [
                    "illusion",
                    "foreign",
                    "technology",
                    "sudden events",
                ],
                "exaltation": Zodiac.TAURUS,
                "debilitation": Zodiac.SCORPIO,
                "own_signs": [],
                "friendly_planets": [
                    Planet.MERCURY,
                    Planet.VENUS,
                    Planet.SATURN,
                ],
                "enemy_planets": [Planet.SUN, Planet.MOON, Planet.MARS],
                "neutral_planets": [Planet.JUPITER],
            },
            Planet.KETU: {
                "nature": "malefic",
                "element": "fire",
                "significations": [
                    "spirituality",
                    "detachment",
                    "moksha",
                    "research",
                ],
                "exaltation": Zodiac.SCORPIO,
                "debilitation": Zodiac.TAURUS,
                "own_signs": [],
                "friendly_planets": [Planet.MARS, Planet.JUPITER],
                "enemy_planets": [Planet.SUN, Planet.MOON],
                "neutral_planets": [
                    Planet.MERCURY,
                    Planet.VENUS,
                    Planet.SATURN,
                ],
            },
        }

    def _initialize_nakshatra_data(self) -> Dict[Nakshatra, Dict[str, Any]]:
        """Initialize nakshatra characteristics"""
        return {
            Nakshatra.ASHWINI: {
                "ruling_planet": Planet.KETU,
                "symbol": "Horse's head",
                "deity": "Ashwini Kumaras",
                "nature": "Swift, healing",
                "element": "Earth",
                "characteristics": [
                    "Quick action",
                    "Healing abilities",
                    "Initiative",
                ],
            },
            Nakshatra.BHARANI: {
                "ruling_planet": Planet.VENUS,
                "symbol": "Yoni (female organ)",
                "deity": "Yama",
                "nature": "Nurturing, transformative",
                "element": "Earth",
                "characteristics": [
                    "Creativity",
                    "Transformation",
                    "Responsibility",
                ],
            },
            # Add all 27 nakshatras...
        }

    def _initialize_remedies_database(self) -> Dict[Planet, List[Remedy]]:
        """Initialize remedies database"""
        return {
            Planet.SUN: [
                Remedy(
                    remedy_type="Gemstone",
                    description="Ruby (Manikya) - 3-6 carats in gold ring",
                    planet_benefited=Planet.SUN,
                    implementation="Wear on ring finger of right hand on Sunday morning",
                    duration="Lifelong or minimum 3 years",
                    benefits=[
                        "Increased confidence",
                        "Better health",
                        "Authority",
                        "Success",
                    ],
                    cost_estimate="₹50,000 - ₹5,00,000+",
                ),
                Remedy(
                    remedy_type="Mantra",
                    description="Surya Mantra - ॐ ह्रां ह्रीं ह्रौं सः सूर्याय नमः",
                    planet_benefited=Planet.SUN,
                    implementation="Chant 7000 times daily or"
                    + "108 times for 40 days",
                    duration="40 days minimum, lifelong for best results",
                    benefits=[
                        "Solar energy",
                        "Vitality",
                        "Leadership",
                        "Fame",
                    ],
                    cost_estimate="Free",
                ),
                Remedy(
                    remedy_type="Charity",
                    description="Donate wheat," + "jaggery, copper items on Sundays",
                    planet_benefited=Planet.SUN,
                    implementation="Every Sunday morning to needy or temples",
                    duration="Continuous",
                    benefits=[
                        "Karmic purification",
                        "Solar blessings",
                        "Health",
                    ],
                    cost_estimate="₹100-500 weekly",
                ),
            ],
            # Add remedies for all planets...
        }

    def _calculate_ayanamsa(self, birth_datetime: datetime) -> float:
        """Calculate ayanamsa (precession correction)"""
        # Simplified Lahiri ayanamsa calculation
        # Base ayanamsa for 1900: 22.46°
        years_since_1900 = birth_datetime.year - 1900
        ayanamsa = 22.46 + (years_since_1900 * 50.23 / 3600)  # Simplified formula
        return ayanamsa

    def _calculate_planet_position(
        self, planet: Planet, birth_datetime: datetime, ayanamsa: float
    ) -> PlanetaryPosition:
        """Calculate planet position (simplified)"""
        # This is a simplified calculation - real implementation would use
        # Swiss Ephemeris or similar astronomical calculation library

        # Simplified position calculation for demo
        base_positions = {
            Planet.SUN: 80.0,
            Planet.MOON: 120.0,
            Planet.MARS: 200.0,
            Planet.MERCURY: 85.0,
            Planet.JUPITER: 150.0,
            Planet.VENUS: 100.0,
            Planet.SATURN: 300.0,
            Planet.RAHU: 250.0,
            Planet.KETU: 70.0,
        }

        # Adjust for birth time (very simplified)
        longitude = (base_positions[planet] + birth_datetime.day * 2) % 360

        # Apply ayanamsa correction
        longitude = (longitude - ayanamsa) % 360

        # Determine zodiac sign
        zodiac_signs = list(Zodiac)
        sign_index = int(longitude // 30)
        zodiac_sign = zodiac_signs[sign_index]

        # Determine nakshatra
        nakshatra_index = int(longitude / (360 / 27))
        nakshatras = list(Nakshatra)
        nakshatra = nakshatras[nakshatra_index % 27]

        # Nakshatra pada
        pada = int((longitude % (360 / 27)) / (360 / 27 / 4)) + 1

        return PlanetaryPosition(
            planet=planet,
            longitude=longitude,
            zodiac_sign=zodiac_sign,
            house=House.FIRST,  # Simplified - would calculate actual house
            nakshatra=nakshatra,
            nakshatra_pada=pada,
            retrograde=False,  # Simplified
        )

    def _calculate_lagna(
        self,
        birth_datetime: datetime,
        latitude: float,
        longitude: float,
        ayanamsa: float,
    ) -> Zodiac:
        """Calculate lagna (ascendant)"""
        # Simplified lagna calculation
        # Real implementation would use sidereal time calculations

        hour_factor = birth_datetime.hour * 15  # 15 degrees per hour
        lagna_longitude = (hour_factor + longitude / 4) % 360
        lagna_longitude = (lagna_longitude - ayanamsa) % 360

        zodiac_signs = list(Zodiac)
        sign_index = int(lagna_longitude // 30)
        return zodiac_signs[sign_index]

    def _calculate_house_cusps(
        self, lagna: Zodiac, birth_datetime: datetime
    ) -> List[float]:
        """Calculate house cusp positions"""
        # Simplified equal house system
        lagna_degree = list(Zodiac).index(lagna) * 30
        cusps = []

        for i in range(12):
            cusp = (lagna_degree + i * 30) % 360
            cusps.append(cusp)

        return cusps

    # Helper methods for analysis

    def _get_sign_lord(self, sign: Zodiac) -> Planet:
        """Get ruling planet of zodiac sign"""
        lords = {
            Zodiac.ARIES: Planet.MARS,
            Zodiac.TAURUS: Planet.VENUS,
            Zodiac.GEMINI: Planet.MERCURY,
            Zodiac.CANCER: Planet.MOON,
            Zodiac.LEO: Planet.SUN,
            Zodiac.VIRGO: Planet.MERCURY,
            Zodiac.LIBRA: Planet.VENUS,
            Zodiac.SCORPIO: Planet.MARS,
            Zodiac.SAGITTARIUS: Planet.JUPITER,
            Zodiac.CAPRICORN: Planet.SATURN,
            Zodiac.AQUARIUS: Planet.SATURN,
            Zodiac.PISCES: Planet.JUPITER,
        }
        return lords[sign]

    def _find_planet_sign(self, horoscope: Horoscope, planet: Planet) -> Zodiac:
        """Find zodiac sign of a planet"""
        for pos in horoscope.planetary_positions:
            if pos.planet == planet:
                return pos.zodiac_sign
        return Zodiac.ARIES  # Default

    def _analyze_elemental_balance(self, horoscope: Horoscope) -> Dict[str, int]:
        """Analyze elemental balance in horoscope"""
        elements = {"fire": 0, "earth": 0, "air": 0, "water": 0}

        sign_elements = {
            Zodiac.ARIES: "fire",
            Zodiac.LEO: "fire",
            Zodiac.SAGITTARIUS: "fire",
            Zodiac.TAURUS: "earth",
            Zodiac.VIRGO: "earth",
            Zodiac.CAPRICORN: "earth",
            Zodiac.GEMINI: "air",
            Zodiac.LIBRA: "air",
            Zodiac.AQUARIUS: "air",
            Zodiac.CANCER: "water",
            Zodiac.SCORPIO: "water",
            Zodiac.PISCES: "water",
        }

        for pos in horoscope.planetary_positions:
            element = sign_elements[pos.zodiac_sign]
            elements[element] += 1

        return elements

    def _get_personality_traits(self, horoscope: Horoscope) -> List[str]:
        """Get personality traits from horoscope"""
        traits = []

        # Add traits based on lagna
        lagna_traits = {
            Zodiac.ARIES: [
                "Energetic",
                "Leadership",
                "Impulsive",
                "Pioneering",
            ],
            Zodiac.TAURUS: ["Stable", "Practical", "Persistent", "Sensual"],
            Zodiac.GEMINI: [
                "Communicative",
                "Versatile",
                "Curious",
                "Adaptable",
            ],
            Zodiac.CANCER: [
                "Emotional",
                "Nurturing",
                "Intuitive",
                "Protective",
            ],
            Zodiac.LEO: ["Creative", "Generous", "Dramatic", "Authoritative"],
            Zodiac.VIRGO: [
                "Analytical",
                "Perfectionist",
                "Service-oriented",
                "Detail-oriented",
            ],
            Zodiac.LIBRA: ["Harmonious", "Diplomatic", "Artistic", "Balanced"],
            Zodiac.SCORPIO: [
                "Intense",
                "Transformative",
                "Mysterious",
                "Passionate",
            ],
            Zodiac.SAGITTARIUS: [
                "Philosophical",
                "Adventurous",
                "Optimistic",
                "Truth-seeking",
            ],
            Zodiac.CAPRICORN: [
                "Ambitious",
                "Disciplined",
                "Practical",
                "Responsible",
            ],
            Zodiac.AQUARIUS: [
                "Innovative",
                "Humanitarian",
                "Independent",
                "Unique",
            ],
            Zodiac.PISCES: [
                "Intuitive",
                "Compassionate",
                "Artistic",
                "Spiritual",
            ],
        }

        traits.extend(lagna_traits.get(horoscope.lagna, []))
        return traits

    def _analyze_strengths(self, horoscope: Horoscope) -> List[str]:
        """Analyze strengths from horoscope"""
        strengths = []

        # Check exalted planets
        for pos in horoscope.planetary_positions:
            planet_data = self.planetary_data[pos.planet]
            if pos.zodiac_sign == planet_data.get("exaltation"):
                strengths.append(
                    f"Strong {pos.planet.value} brings {', '.join(planet_data['significations'][:2])}"
                )

        return strengths

    def _analyze_challenges(self, horoscope: Horoscope) -> List[str]:
        """Analyze challenges from horoscope"""
        challenges = []

        # Check debilitated planets
        for pos in horoscope.planetary_positions:
            planet_data = self.planetary_data[pos.planet]
            if pos.zodiac_sign == planet_data.get("debilitation"):
                significations = ", ".join(planet_data["significations"][:2])
                challenges.append(
                    f"Weak {pos.planet.value} may cause issues with {significations}"
                )

        return challenges

    # Additional helper methods would be implemented here...

    def _analyze_life_purpose(self, horoscope: Horoscope) -> str:
        """Analyze life purpose from horoscope"""
        return "Detailed life purpose analysis based on planetary positions"

    def _analyze_spiritual_nature(self, horoscope: Horoscope) -> Dict[str, Any]:
        """Analyze spiritual inclinations"""
        return {
            "spiritual_potential": "High",
            "recommended_practices": [
                "Meditation",
                "Mantra chanting",
                "Service",
            ],
            "spiritual_timing": "After age 35",
        }

    # More methods would be implemented for complete functionality...


# Global instance
_jyotisha_engine = None


def get_jyotisha_engine() -> AdvancedJyotishaEngine:
    """Get global Jyotisha Engine instance"""
    global _jyotisha_engine
    if _jyotisha_engine is None:
        _jyotisha_engine = AdvancedJyotishaEngine()
    return _jyotisha_engine
